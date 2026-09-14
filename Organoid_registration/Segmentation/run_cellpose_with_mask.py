#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Patch-wise 3D Cellpose prediction with an aligned and expanded foreground mask."""

import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from cellpose import models
from nibabel.processing import resample_from_to
from scipy.ndimage import distance_transform_edt

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


@dataclass
class SegmentationConfig:
    patch_size_zyx: Tuple[int, int, int] = (60, 192, 192)
    overlap_fraction: Tuple[float, float, float] = (0.3, 0.1, 0.1)
    pretrained_model: str = "cpsam"
    gpu_device: Optional[int] = 0
    diameter: Optional[float] = None
    cellprob_threshold: float = 0.0
    batch_size: int = 2
    flow3D_smooth: float = 0.0
    save_empty_patches: bool = True
    skip_existing_predictions: bool = True
    skip_empty_mask_patches: bool = True
    skip_zero_input_patches: bool = True
    print_patch_summary: bool = True
    print_patch_debug: bool = True
    mask_threshold: float = 0.0
    mask_expansion_fraction: float = 0.01  # Final bounding-box size increases by about 10%.
    image_spatial_unit: str = "um"
    mask_spatial_unit: str = "mm"
    print_affine: bool = True
    print_mask_stats: bool = True
    save_aligned_mask: bool = True
    aligned_mask_filename: str = "foreground_mask_aligned_expanded.nii.gz"


def print_patch_line(message: str):
    """Print without breaking the tqdm progress bar."""
    if HAS_TQDM:
        tqdm.write(message)
    else:
        print(message)


def get_form_code(header, form_name: str) -> int:
    code = int(header[f"{form_name}_code"])
    return code if code > 0 else 1


def print_nifti_meta(name: str, nii: nib.Nifti1Image, print_affine: bool = True, print_stats: bool = False):
    header = nii.header
    spatial_unit, temporal_unit = header.get_xyzt_units()
    _, qform_code = nii.get_qform(coded=True)
    _, sform_code = nii.get_sform(coded=True)
    print("\n" + "=" * 70)
    print(f"NIFTI METADATA: {name}")
    print("=" * 70)
    print(f"Shape:              {nii.shape}")
    print(f"Data dtype:         {header.get_data_dtype()}")
    print(f"Spacing / zooms:    {header.get_zooms()}")
    print(f"Orientation:        {nib.aff2axcodes(nii.affine)}")
    print(f"Spatial unit:       {spatial_unit}")
    print(f"Temporal unit:      {temporal_unit}")
    print(f"qform code:         {qform_code}")
    print(f"sform code:         {sform_code}")
    if print_affine:
        print("Affine:\n", nii.affine, sep="")
    if print_stats:
        data = np.asanyarray(nii.dataobj)
        finite = data[np.isfinite(data)]
        print(f"Number of voxels:   {data.size}")
        print(f"Nonzero voxels:     {int(np.count_nonzero(data))}")
        if finite.size:
            print(f"Minimum value:      {float(finite.min())}")
            print(f"Maximum value:      {float(finite.max())}")
            print(f"Mean value:         {float(finite.mean())}")
    print("=" * 70)


def convert_nifti_spatial_unit(nii: nib.Nifti1Image, source_unit: str, target_unit: str) -> nib.Nifti1Image:
    scales = {("mm", "mm"): 1.0, ("um", "um"): 1.0, ("mm", "um"): 1000.0, ("um", "mm"): 0.001}
    source_unit = source_unit.lower().replace("µm", "um")
    target_unit = target_unit.lower().replace("µm", "um")
    if (source_unit, target_unit) not in scales:
        raise ValueError(f"Unsupported conversion: {source_unit} -> {target_unit}")
    scale = scales[(source_unit, target_unit)]
    if scale == 1.0:
        return nii
    affine = nii.affine.copy()
    affine[:3, :] *= scale
    header = nii.header.copy()
    header.set_xyzt_units("micron" if target_unit == "um" else "mm")
    converted = nib.Nifti1Image(np.asanyarray(nii.dataobj), affine, header)
    converted.set_qform(affine, code=get_form_code(nii.header, "qform"))
    converted.set_sform(affine, code=get_form_code(nii.header, "sform"))
    return converted


def rebuild_mask_geometry_from_image(mask_nii: nib.Nifti1Image, image_nii: nib.Nifti1Image) -> nib.Nifti1Image:
    """Use image orientation/origin while preserving the mask's lower-resolution grid."""
    image_shape = np.asarray(image_nii.shape[:3], dtype=float)
    mask_shape = np.asarray(mask_nii.shape[:3], dtype=float)
    if np.any(mask_shape <= 0):
        raise ValueError(f"Invalid mask shape: {mask_nii.shape[:3]}")
    scale = image_shape / mask_shape
    affine = image_nii.affine.copy()
    affine[:3, :3] = image_nii.affine[:3, :3] @ np.diag(scale)
    header = mask_nii.header.copy()
    spatial_unit, temporal_unit = image_nii.header.get_xyzt_units()
    if spatial_unit in ("micron", "mm", "meter"):
        header.set_xyzt_units(spatial_unit, temporal_unit)
    corrected = nib.Nifti1Image(np.asanyarray(mask_nii.dataobj), affine, header)
    corrected.set_qform(affine, code=1)
    corrected.set_sform(affine, code=1)
    print(f"[INFO] Mask/image scale XYZ: {tuple(scale)}")
    return corrected


def normalize_masked_patch(image_xyz: np.ndarray, mask_xyz: np.ndarray) -> np.ndarray:
    """Normalize with foreground intensities only and keep background at zero."""
    output = np.zeros_like(image_xyz, dtype=np.float32)
    valid = mask_xyz & np.isfinite(image_xyz)
    values = image_xyz[valid]
    if values.size == 0:
        return output
    lower, upper = np.percentile(values, (1.0, 99.0))
    if upper <= lower:
        return output
    output[valid] = np.clip((image_xyz[valid] - lower) / (upper - lower), 0.0, 1.0)
    return output


def expand_binary_mask(binary_mask: np.ndarray, spacing_xyz, expansion_fraction: float) -> np.ndarray:
    """Expand each side by half the requested fraction of the largest physical bounding-box size."""
    if not 0 <= expansion_fraction < 1:
        raise ValueError("mask_expansion_fraction must satisfy 0 <= value < 1.")
    if expansion_fraction == 0:
        return binary_mask
    coordinates = np.argwhere(binary_mask)
    if coordinates.size == 0:
        raise ValueError("Cannot expand an empty foreground mask.")
    bbox_voxels = coordinates.max(axis=0) - coordinates.min(axis=0) + 1
    spacing_xyz = np.asarray(spacing_xyz, dtype=float)
    bbox_physical = bbox_voxels * spacing_xyz
    expansion_distance = float(bbox_physical.max()) * expansion_fraction / 2.0
    distance_outside = distance_transform_edt(~binary_mask, sampling=spacing_xyz)
    expanded = distance_outside <= expansion_distance
    print("\n" + "=" * 70)
    print("FOREGROUND MASK EXPANSION")
    print("=" * 70)
    print(f"Requested total increase: {expansion_fraction * 100:.1f}%")
    print(f"Expansion per side:       {expansion_distance:.3f} {('µm' if spacing_xyz.max() > 1 else 'mm')}")
    print(f"Bounding box XYZ:         {tuple(int(v) for v in bbox_voxels)} voxels")
    print(f"Foreground before:        {int(np.count_nonzero(binary_mask))}")
    print(f"Foreground after:         {int(np.count_nonzero(expanded))}")
    print(f"Foreground ratio after:   {float(expanded.mean()):.6f}")
    print("=" * 70)
    return expanded


def load_foreground_mask(mask_path: str, image_nii: nib.Nifti1Image, config: SegmentationConfig) -> np.ndarray:
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    mask_nii = nib.load(mask_path)
    if len(mask_nii.shape) not in (3, 4) or (len(mask_nii.shape) == 4 and mask_nii.shape[3] != 1):
        raise ValueError(f"Expected a 3D or singleton-4D mask, got {mask_nii.shape}.")
    print_nifti_meta("Original foreground mask", mask_nii, config.print_affine, config.print_mask_stats)
    mask_nii = convert_nifti_spatial_unit(mask_nii, config.mask_spatial_unit, config.image_spatial_unit)
    mask_nii = rebuild_mask_geometry_from_image(mask_nii, image_nii)
    print_nifti_meta("Mask aligned to input image", mask_nii, config.print_affine)
    resampled = resample_from_to(mask_nii, (image_nii.shape[:3], image_nii.affine), order=0, mode="constant", cval=0)
    mask_xyz = np.asanyarray(resampled.dataobj)
    if mask_xyz.ndim == 4:
        mask_xyz = mask_xyz[..., 0]
    binary_mask = mask_xyz > config.mask_threshold
    if not np.any(binary_mask):
        raise ValueError("Resampled foreground mask contains no foreground voxels.")
    binary_mask = expand_binary_mask(binary_mask, image_nii.header.get_zooms()[:3], config.mask_expansion_fraction)
    print_nifti_meta("Mask resampled to input grid before expansion", resampled, config.print_affine, config.print_mask_stats)
    print("\n" + "=" * 70)
    print("FINAL EXPANDED FOREGROUND MASK")
    print("=" * 70)
    print(f"Shape:              {binary_mask.shape}")
    print(f"Data dtype:         {binary_mask.dtype}")
    print(f"Nonzero voxels:     {int(np.count_nonzero(binary_mask))}")
    print(f"Foreground ratio:   {float(binary_mask.mean()):.6f}")
    print("=" * 70)
    same_geometry = (resampled.shape[:3] == image_nii.shape[:3] and
                     np.allclose(resampled.affine, image_nii.affine, atol=1e-5))
    print(f"[INFO] Final mask geometry matches input image: {same_geometry}")
    return binary_mask


def generate_axis_starts(full_size: int, patch_size: int, overlap: float) -> List[int]:
    if full_size <= 0 or patch_size <= 0:
        raise ValueError("Image and patch dimensions must be positive.")
    if not 0 <= overlap < 1:
        raise ValueError("Each overlap fraction must satisfy 0 <= overlap < 1.")
    if patch_size >= full_size:
        return [0]
    stride = max(1, int(round(patch_size * (1.0 - overlap))))
    starts = list(range(0, full_size - patch_size + 1, stride))
    if starts[-1] != full_size - patch_size:
        starts.append(full_size - patch_size)
    return starts


def generate_patch_grid(volume_shape_zyx, patch_size_zyx, overlap_fraction):
    axes = [generate_axis_starts(volume_shape_zyx[i], patch_size_zyx[i], overlap_fraction[i]) for i in range(3)]
    patches = []
    for z0 in axes[0]:
        for y0 in axes[1]:
            for x0 in axes[2]:
                patches.append((slice(z0, min(z0 + patch_size_zyx[0], volume_shape_zyx[0])),
                                slice(y0, min(y0 + patch_size_zyx[1], volume_shape_zyx[1])),
                                slice(x0, min(x0 + patch_size_zyx[2], volume_shape_zyx[2]))))
    return patches


def get_device(gpu_device: Optional[int]) -> torch.device:
    if gpu_device is None:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device cuda:{gpu_device} was requested, but CUDA is unavailable.")
    if not 0 <= gpu_device < torch.cuda.device_count():
        raise ValueError(f"Invalid GPU index {gpu_device}; available indices: 0-{torch.cuda.device_count() - 1}.")
    return torch.device(f"cuda:{gpu_device}")


def choose_mask_dtype(max_label: int):
    return np.uint8 if max_label <= 255 else np.uint16 if max_label <= 65535 else np.uint32


def save_binary_mask(mask_xyz: np.ndarray, reference_nii: nib.Nifti1Image, output_path: str):
    """Save a binary mask on the exact grid and physical space of the reference image."""
    mask_xyz = mask_xyz.astype(np.uint8, copy=False)
    header = reference_nii.header.copy()
    header.set_data_shape(mask_xyz.shape)
    header.set_data_dtype(np.uint8)
    mask_nii = nib.Nifti1Image(mask_xyz, reference_nii.affine, header)
    mask_nii.set_qform(reference_nii.affine, code=1)
    mask_nii.set_sform(reference_nii.affine, code=1)
    nib.save(mask_nii, output_path)


def save_patch_mask(mask_zyx: np.ndarray, patch_slices_zyx, image_nii: nib.Nifti1Image, output_path: str):
    pz, py, px = patch_slices_zyx
    mask_xyz = np.transpose(mask_zyx, (2, 1, 0))
    mask_xyz = mask_xyz.astype(choose_mask_dtype(int(mask_xyz.max())), copy=False)
    affine = image_nii.affine.copy()
    affine[:3, 3] = (image_nii.affine @ np.array([px.start, py.start, pz.start, 1.0]))[:3]
    header = image_nii.header.copy()
    header.set_data_shape(mask_xyz.shape)
    header.set_data_dtype(mask_xyz.dtype)
    patch_nii = nib.Nifti1Image(mask_xyz, affine, header)
    patch_nii.set_qform(affine, code=get_form_code(image_nii.header, "qform"))
    patch_nii.set_sform(affine, code=get_form_code(image_nii.header, "sform"))
    nib.save(patch_nii, output_path)


def save_patch_volume(volume_xyz: np.ndarray, patch_slices_zyx, image_nii: nib.Nifti1Image, output_path: str):
    pz, py, px = patch_slices_zyx
    volume_xyz = volume_xyz.astype(np.float32, copy=False)
    affine = image_nii.affine.copy()
    affine[:3, 3] = (image_nii.affine @ np.array([px.start, py.start, pz.start, 1.0]))[:3]
    header = image_nii.header.copy()
    header.set_data_shape(volume_xyz.shape)
    header.set_data_dtype(np.float32)
    patch_nii = nib.Nifti1Image(volume_xyz, affine, header)
    patch_nii.set_qform(affine, code=get_form_code(image_nii.header, "qform"))
    patch_nii.set_sform(affine, code=get_form_code(image_nii.header, "sform"))
    nib.save(patch_nii, output_path)


def format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def segment_large_volume(image_path: str, output_dir: str, config: SegmentationConfig, foreground_mask_path: Optional[str] = None):
    start_time = time.perf_counter()
    patch_dir = os.path.join(output_dir, "patch_predictions")
    volume_dir = os.path.join(output_dir, "patch_volumes")
    os.makedirs(patch_dir, exist_ok=True)
    os.makedirs(volume_dir, exist_ok=True)

    image_nii = nib.load(image_path)
    if len(image_nii.shape) not in (3, 4):
        raise ValueError(f"Expected a 3D or 4D image, got {image_nii.shape}.")
    print_nifti_meta("Original input image", image_nii, config.print_affine)

    shape_xyz = image_nii.shape[:3]
    shape_zyx = shape_xyz[::-1]
    spacing_xyz = image_nii.header.get_zooms()[:3]
    anisotropy = float(spacing_xyz[2]) / ((float(spacing_xyz[0]) + float(spacing_xyz[1])) / 2.0)

    foreground_mask = load_foreground_mask(foreground_mask_path, image_nii, config) if foreground_mask_path else None

    if foreground_mask is not None and config.save_aligned_mask:
        aligned_mask_path = os.path.join(output_dir, config.aligned_mask_filename)
        save_binary_mask(foreground_mask, image_nii, aligned_mask_path)
        saved_mask_nii = nib.load(aligned_mask_path)
        same_shape = saved_mask_nii.shape[:3] == image_nii.shape[:3]
        same_affine = np.allclose(saved_mask_nii.affine, image_nii.affine, atol=1e-5)
        print(f"[INFO] Saved aligned foreground mask: {aligned_mask_path}")
        print(f"[INFO] Saved mask matches image grid: shape={same_shape}, affine={same_affine}")

    patches = generate_patch_grid(shape_zyx, config.patch_size_zyx, config.overlap_fraction)
    device = get_device(config.gpu_device)

    print(f"[INFO] Shape XYZ/ZYX: {shape_xyz} / {shape_zyx}")
    print(f"[INFO] Spacing XYZ: {spacing_xyz} | Orientation: {nib.aff2axcodes(image_nii.affine)}")
    print(f"[INFO] Anisotropy Z/XY: {anisotropy:.3f} | Total patches: {len(patches)}")
    print(f"[INFO] Device: {device}" + (f" | GPU: {torch.cuda.get_device_name(config.gpu_device)}" if device.type == "cuda" else ""))

    model = models.CellposeModel(pretrained_model=config.pretrained_model, device=device)
    predicted = skipped_existing = skipped_empty = skipped_zero_input = empty_predictions = 0
    iterator = tqdm(patches, desc="Cellpose patches", unit="patch", dynamic_ncols=True) if HAS_TQDM else patches

    for index, (pz, py, px) in enumerate(iterator):
        patch_start = time.perf_counter()

        filename = f"patch_{index:06d}_z{pz.start}-{pz.stop}_y{py.start}-{py.stop}_x{px.start}-{px.stop}.nii.gz"
        output_path = os.path.join(patch_dir, filename)
        volume_output_path = os.path.join(volume_dir, filename)

        if config.skip_existing_predictions and os.path.isfile(output_path):
            skipped_existing += 1
            if config.print_patch_summary:
                print_patch_line(f"[PATCH {index + 1:03d}/{len(patches):03d}] existing | {filename}")
            continue

        patch_xyz = np.asarray(image_nii.dataobj[px, py, pz], dtype=np.float32)
        if patch_xyz.ndim == 4:
            patch_xyz = patch_xyz[..., 0]

        mask_nonzero = None
        if foreground_mask is not None:
            mask_patch = foreground_mask[px, py, pz]
            mask_nonzero = int(np.count_nonzero(mask_patch))

            if config.skip_empty_mask_patches and mask_nonzero == 0:
                skipped_empty += 1
                if config.save_empty_patches:
                    save_patch_mask(np.zeros(patch_xyz.shape[::-1], dtype=np.uint8), (pz, py, px), image_nii, output_path)
                if config.print_patch_summary:
                    print_patch_line(f"[PATCH {index + 1:03d}/{len(patches):03d}] outside mask | predicted_nonzero=0")
                continue
        else:
            mask_patch = np.isfinite(patch_xyz) & (patch_xyz != 0)

        patch_xyz = normalize_masked_patch(patch_xyz, mask_patch)

        save_patch_volume(patch_xyz, (pz, py, px), image_nii, volume_output_path)

        patch_zyx = np.nan_to_num(np.transpose(patch_xyz, (2, 1, 0)), nan=0.0, posinf=0.0, neginf=0.0)
        input_nonzero = int(np.count_nonzero(patch_zyx))
        input_min = float(patch_zyx.min())
        input_max = float(patch_zyx.max())
        input_mean = float(patch_zyx.mean())

        if config.print_patch_debug:
            print_patch_line(f"[DEBUG] Patch {index + 1:03d} | mask_nonzero={mask_nonzero} | input_nonzero={input_nonzero} | range=({input_min:.3f}, {input_max:.3f}) | mean={input_mean:.3f}")

        if config.skip_zero_input_patches and input_nonzero == 0:
            skipped_zero_input += 1
            if config.save_empty_patches:
                save_patch_mask(np.zeros(patch_zyx.shape, dtype=np.uint8), (pz, py, px), image_nii, output_path)
            if config.print_patch_summary:
                print_patch_line(f"[PATCH {index + 1:03d}/{len(patches):03d}] zero input | predicted_nonzero=0")
            continue

        try:
            masks, _, _ = model.eval(patch_zyx, diameter=config.diameter, do_3D=True, z_axis=0,
                                     channel_axis=None, anisotropy=anisotropy,
                                     cellprob_threshold=config.cellprob_threshold,
                                     flow3D_smooth=config.flow3D_smooth,
                                     batch_size=config.batch_size, normalize=False)
        except torch.cuda.OutOfMemoryError as error:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            raise RuntimeError(f"CUDA out of memory at patch {index}, Z={pz.start}:{pz.stop}, Y={py.start}:{py.stop}, X={px.start}:{px.stop}.") from error

        masks = masks.astype(np.int32, copy=False)
        label_count = int(masks.max())
        prediction_nonzero = int(np.count_nonzero(masks))

        if config.print_patch_summary:
            print_patch_line(f"[PATCH {index + 1:03d}/{len(patches):03d}] predicted | labels={label_count} | nonzero_voxels={prediction_nonzero} | time={time.perf_counter() - patch_start:.1f}s")

        if config.print_patch_debug:
            print_patch_line(f"[DEBUG] Patch {index + 1:03d} | cellpose_labels={label_count} | predicted_voxels={prediction_nonzero}")

        if label_count == 0:
            empty_predictions += 1

        if config.save_empty_patches or masks.max() > 0:
            save_patch_mask(masks, (pz, py, px), image_nii, output_path)

        predicted += 1

        if HAS_TQDM:
            iterator.set_postfix(last=f"{time.perf_counter() - patch_start:.1f}s", predicted=predicted,
                                 existing=skipped_existing, outside=skipped_empty, zero_input=skipped_zero_input,
                                 empty_pred=empty_predictions, elapsed=format_seconds(time.perf_counter() - start_time))
        elif index % 10 == 0 or index + 1 == len(patches):
            print(f"[INFO] Patch {index + 1}/{len(patches)} | predicted={predicted} | existing={skipped_existing} | "
                  f"outside={skipped_empty} | zero_input={skipped_zero_input} | empty_pred={empty_predictions}")

    print("\n" + "=" * 70)
    print("CELLPOSE PATCH PREDICTION FINISHED")
    print("=" * 70)
    print(f"Total time:             {format_seconds(time.perf_counter() - start_time)}")
    print(f"Total patches:          {len(patches)}")
    print(f"Cellpose predictions:   {predicted}")
    print(f"Existing skipped:       {skipped_existing}")
    print(f"Outside-mask skipped:   {skipped_empty}")
    print(f"Zero-input skipped:     {skipped_zero_input}")
    print(f"Empty predictions:      {empty_predictions}")
    print(f"Patch output directory: {patch_dir}")
    print(f"Patch volume directory: {volume_dir}")
    print("=" * 70)


if __name__ == "__main__":
    CONFIG = SegmentationConfig()

    # IMAGE_PATH = (r"/autofs/arch11/DATA/HOMES/yuchen/Minor/Organoids/Nifti_data/"
    #               r"case2_young_Paw_Ventricles/data_new/case2_Topro3_origin.nii")
    # FOREGROUND_MASK_PATH = (r"/autofs/arch11/DATA/HOMES/yuchen/Minor/Organoids/Nifti_data/"
    #                         r"case2_young_Paw_Ventricles/data_new/case2_mask_high_FA.nii.gz")

    IMAGE_PATH = (r"/autofs/arch11/DATA/HOMES/yuchen/Minor/Organoids/Nifti_data/"
                  r"case1_mature/data_new/case1_Axon_origin.nii")
    FOREGROUND_MASK_PATH = (r"/autofs/arch11/DATA/HOMES/yuchen/Minor/Organoids/Nifti_data/"
                            r"case1_mature/data_new/case1_dmri_6_mask_after.nii.gz")

    OUTPUT_DIR = (r"/autofs/arch11/DATA/HOMES/yuchen/Minor/Organoids/Nifti_data/seg/"
                  r"case1_axon_test")

    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Input image not found: {IMAGE_PATH}")

    segment_large_volume(IMAGE_PATH, OUTPUT_DIR, CONFIG, None)