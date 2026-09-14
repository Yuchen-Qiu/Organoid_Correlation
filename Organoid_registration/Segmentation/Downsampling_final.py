#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate microscopy maps on native and custom dMRI grids.

Outputs:
1. Binary microscopy map on native dMRI grid
2. Binary microscopy map after eroded mask
3. Point-count map on native dMRI grid
4. Point-count map after eroded mask
5. Point-count maps on custom dMRI grids
6. Custom-grid eroded masks
7. Masked point-count maps on custom grids

Grid construction and microscopy-point mapping follow the same logic
as the correlation script.
"""

import os
import csv
import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform
from nibabel.processing import resample_from_to


# ============================================================
# User settings
# ============================================================

MERGED_DIR = r"/autofs/arch11/DATA/HOMES/yuchen/Minor/Organoids/Nifti_data/seg/case2_cellpose_patch_centers_origin_192_192_96/merged"

BINARY_INPUT_PATH = os.path.join(MERGED_DIR, "merged_binary_mask.nii.gz")
CENTER_CSV_PATH = os.path.join(MERGED_DIR, "component_centers.csv")

MRI_REFERENCE_PATH = r"/autofs/arch11/DATA/HOMES/yuchen/Minor/Organoids/Nifti_data/case2_young/data_new/registered_FA_rab.nii.gz"

# Change this to your actual eroded mask.
ERODED_MASK_PATH = r"/autofs/arch11/DATA/HOMES/yuchen/Minor/Organoids/Nifti_data/case2_young/data_final/registered_Addition_native_Bvalue/erosion_mask_registered_native.nii.gz"

OUTPUT_DIR = r"/autofs/arch11/DATA/HOMES/yuchen/Minor/Organoids/Nifti_data/case2_young/Seg"

# Extra analysis grids.
# Same meaning as TARGET_DMRI_SPACING_UM in the correlation script.
CUSTOM_GRID_SIZES_UM = [
    (3.6, 3.6, 3.6),
    (20.0, 20.0, 20.0),
    (44.79, 44.79, 44.79),
]

MICROSCOPY_WORLD_UNIT = "um"
DMRI_WORLD_UNIT = "mm"

SAVE_RESAMPLED_ERODED_MASK = True


# ============================================================
# Output paths
# ============================================================

BINARY_NATIVE_PATH = os.path.join(
    OUTPUT_DIR,
    "Topro3_downsampling_high.nii.gz",
)

BINARY_NATIVE_MASKED_PATH = os.path.join(OUTPUT_DIR, "Topro3_downsampling_high_masked.nii.gz",)

POINT_COUNT_NATIVE_PATH = os.path.join(OUTPUT_DIR, "Topro3_point_count_native.nii.gz",)

POINT_COUNT_NATIVE_MASKED_PATH = os.path.join(OUTPUT_DIR, "Topro3_point_count_native_masked.nii.gz",)

ERODED_MASK_NATIVE_PATH = os.path.join(OUTPUT_DIR, "eroded_mask_native.nii.gz",)


# ============================================================
# Geometry helpers
# ============================================================

def to_um_factor(unit):
    unit = unit.lower()

    if unit in ("um", "µm"):
        return 1.0

    if unit == "mm":
        return 1000.0

    raise ValueError(f"Unsupported unit: {unit}")


def affine_to_um(affine, unit):
    output = affine.copy().astype(np.float64)
    output[:3, :] *= to_um_factor(unit)
    return output


def get_spacing_um(nii, unit):
    return np.linalg.norm(affine_to_um(nii.affine, unit,)[:3, :3], axis=0,)


def print_geometry(name, nii, unit):
    print()
    print("========================================")
    print(name)
    print("========================================")
    print("[INFO] Shape XYZ:      ", nii.shape[:3])
    print("[INFO] Spacing (um):   ", get_spacing_um(nii, unit))
    print("[INFO] Orientation:    ", nib.aff2axcodes(nii.affine))
    print("[INFO] Origin:         ", nii.affine[:3, 3])
    print("[INFO] qform code:     ", int(nii.header["qform_code"]))
    print("[INFO] sform code:     ", int(nii.header["sform_code"]))


def check_fov(microscopy_nii, dmri_nii):
    microscopy_shape = np.asarray(microscopy_nii.shape[:3], dtype=np.float64,)

    dmri_shape = np.asarray(dmri_nii.shape[:3], dtype=np.float64,)

    microscopy_spacing_um = get_spacing_um(microscopy_nii, MICROSCOPY_WORLD_UNIT,)

    dmri_spacing_um = get_spacing_um(dmri_nii, DMRI_WORLD_UNIT,)

    microscopy_fov_um = (microscopy_shape * microscopy_spacing_um)

    dmri_fov_um = (dmri_shape * dmri_spacing_um)

    difference_um = np.abs(microscopy_fov_um - dmri_fov_um)

    tolerance_um = np.maximum(microscopy_spacing_um, dmri_spacing_um,)

    print()
    print("========================================")
    print("Physical FOV check")
    print("========================================")
    print("[INFO] Microscopy FOV (um):", microscopy_fov_um)
    print("[INFO] dMRI FOV (um):      ", dmri_fov_um)
    print("[INFO] Difference (um):    ", difference_um)

    if np.any(difference_um > tolerance_um):
        print("[WARNING] Microscopy and dMRI FOVs differ by more " "than one voxel. Continue anyway.")


# ============================================================
# Same custom-grid logic as correlation script
# ============================================================

def build_target_geometry(
    nii,
    target_spacing_um,
    spatial_unit,
):
    current_spacing_um = get_spacing_um(nii, spatial_unit,)

    target_spacing_um = np.asarray(target_spacing_um, dtype=np.float64,)

    if (target_spacing_um.shape != (3,) or np.any(target_spacing_um <= 0)):
        raise ValueError("target_spacing_um must contain " "three positive values.")

    current_affine = nii.affine.astype(np.float64)

    current_linear = current_affine[:3, :3,]

    current_spacing_native = np.linalg.norm(current_linear, axis=0,)

    target_spacing_native = (target_spacing_um / to_um_factor(spatial_unit))

    direction = (current_linear / current_spacing_native)

    target_linear = (direction * target_spacing_native)

    current_shape = np.asarray(nii.shape[:3], dtype=np.float64,)

    target_shape = np.maximum(1, np.round(current_shape * current_spacing_um / target_spacing_um).astype(int),)

    old_edge_origin = nib.affines.apply_affine(current_affine, [-0.5, -0.5, -0.5],)

    target_translation = (old_edge_origin + target_linear @ np.array([0.5, 0.5, 0.5]))

    target_affine = np.eye(4, dtype=np.float64,)

    target_affine[:3, :3] = target_linear
    target_affine[:3, 3] = target_translation

    old_fov_um = (current_shape * current_spacing_um)

    new_fov_um = (target_shape.astype(np.float64) * target_spacing_um)

    print()
    print("========================================")
    print("Build custom dMRI grid")
    print("========================================")
    print("[INFO] Native shape:", tuple(nii.shape[:3]),)
    print("[INFO] Native spacing (um):", current_spacing_um,)
    print("[INFO] Native FOV (um):", old_fov_um,)
    print("[INFO] Target shape:", tuple(target_shape),)
    print("[INFO] Target spacing (um):", target_spacing_um,)
    print("[INFO] Target FOV (um):", new_fov_um,)
    print("[INFO] FOV difference (um):", new_fov_um - old_fov_um,)

    return (tuple(target_shape), target_affine,)


# ============================================================
# CSV
# ============================================================

def load_centers(csv_path):
    centers = []

    with open(csv_path, "r", newline="", encoding="utf-8",) as file:

        reader = csv.DictReader(file)

        required = { "voxel_center_x", "voxel_center_y", "voxel_center_z", }

        if not required.issubset(reader.fieldnames or []):
            raise ValueError("component_centers.csv must contain " "voxel_center_x/y/z.")

        for row in reader:
            try:
                centers.append([
                    float(row["voxel_center_x"]),
                    float(row["voxel_center_y"]),
                    float(row["voxel_center_z"]),
                ])
            except (ValueError, TypeError):
                continue

    centers = np.asarray(centers, dtype=np.float64,)

    if len(centers) == 0:
        raise ValueError("No valid component centers found.")

    return centers


# ============================================================
# NIfTI saving
# ============================================================

def save_nifti(
    data,
    affine,
    source_nii,
    output_path,
    dtype,
):
    data = data.astype(dtype, copy=False,)

    header = source_nii.header.copy()

    header.set_data_shape(data.shape)

    header.set_data_dtype(dtype)

    spacing = np.linalg.norm(affine[:3, :3], axis=0,)

    header.set_zooms(tuple(spacing))

    output_nii = nib.Nifti1Image(data, affine, header,)

    _, qcode = source_nii.get_qform(coded=True)

    _, scode = source_nii.get_sform(coded=True)

    qcode = (int(qcode) if qcode is not None and int(qcode) > 0 else 1)

    scode = (int(scode) if scode is not None and int(scode) > 0 else 1)

    output_nii.set_qform(affine, qcode,)

    output_nii.set_sform(affine, scode,)

    nib.save(output_nii, output_path,)

    print("[INFO] Saved:", output_path)


# ============================================================
# Binary microscopy -> native dMRI grid
# ============================================================

def generate_binary_native(
    microscopy_data,
    microscopy_nii,
    dmri_nii,
):
    microscopy_spacing_um = get_spacing_um(microscopy_nii, MICROSCOPY_WORLD_UNIT,)

    dmri_spacing_um = get_spacing_um(dmri_nii, DMRI_WORLD_UNIT,)

    scale_xyz = (dmri_spacing_um / microscopy_spacing_um)

    matrix = np.diag(scale_xyz)

    offset = (scale_xyz - 1.0) / 2.0

    binary = (microscopy_data > 0).astype(np.uint8)

    output = affine_transform(
        binary,
        matrix=matrix,
        offset=offset,
        output_shape=dmri_nii.shape[:3],
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    )

    return (output > 0).astype(np.uint8)


# ============================================================
# Component center -> analysis grid
# Same mapping logic as correlation script
# ============================================================

def map_points_to_grid(
    points_voxel,
    microscopy_nii,
    target_shape,
    target_spacing_um,
):
    microscopy_shape = np.asarray(microscopy_nii.shape[:3], dtype=np.int64,)

    target_shape = np.asarray(target_shape, dtype=np.int64,)

    microscopy_spacing_um = get_spacing_um(microscopy_nii, MICROSCOPY_WORLD_UNIT,)

    target_spacing_um = np.asarray(target_spacing_um, dtype=np.float64,)

    inside_microscopy = (
        (points_voxel[:, 0] >= -0.5)
        & (points_voxel[:, 0] < microscopy_shape[0] - 0.5)
        & (points_voxel[:, 1] >= -0.5)
        & (points_voxel[:, 1] < microscopy_shape[1] - 0.5)
        & (points_voxel[:, 2] >= -0.5)
        & (points_voxel[:, 2] < microscopy_shape[2] - 0.5)
    )

    points_um = (points_voxel + 0.5) * microscopy_spacing_um

    indices = np.floor(points_um / target_spacing_um).astype(np.int64)

    inside_target = (
        (indices[:, 0] >= 0)
        & (indices[:, 0] < target_shape[0])
        & (indices[:, 1] >= 0)
        & (indices[:, 1] < target_shape[1])
        & (indices[:, 2] >= 0)
        & (indices[:, 2] < target_shape[2])
    )

    valid = (inside_microscopy & inside_target)

    valid_indices = indices[valid]

    flat_ids = np.ravel_multi_index(valid_indices.T, tuple(target_shape),)

    counts = np.bincount(flat_ids, minlength=int(np.prod(target_shape)),)

    count_map = counts.reshape(tuple(target_shape)).astype(np.uint32)

    print("[INFO] Centers total:", len(points_voxel),)

    print("[INFO] Centers inside microscopy:", int(inside_microscopy.sum()),)

    print("[INFO] Centers mapped inside grid:", int(valid.sum()),)

    print("[INFO] Centers outside grid:", int((inside_microscopy & ~inside_target).sum()),)

    print("[INFO] Non-zero grid voxels:", int(np.count_nonzero(count_map)),)

    print("[INFO] Maximum count / voxel:", int(count_map.max()),)

    print("[INFO] Sum of count map:", int(count_map.sum()),)

    if (int(count_map.sum()) != int(valid.sum())):
        raise RuntimeError("Point count was not preserved.")

    return count_map


# ============================================================
# Eroded mask resampling
# ============================================================

def resample_mask(
    mask_nii,
    target_shape,
    target_affine,
):
    mask_data = (np.asanyarray(mask_nii.dataobj) > 0).astype(np.uint8)

    if (mask_data.shape == tuple(target_shape) and np.allclose(mask_nii.affine, target_affine, atol=1e-5,)):
        return mask_data

    clean_mask_nii = nib.Nifti1Image(mask_data, mask_nii.affine,)

    resampled = resample_from_to(
        clean_mask_nii,
        (
            tuple(target_shape),
            target_affine,
        ),
        order=0,
        mode="constant",
        cval=0,
    ).get_fdata(
        dtype=np.float32
    )

    return (resampled > 0.5).astype(np.uint8)


# ============================================================
# Grid tag
# ============================================================

def grid_tag(spacing_um):
    spacing_um = np.asarray(spacing_um, dtype=np.float64,)

    if np.allclose(spacing_um, spacing_um[0],):
        value = spacing_um[0]

        if float(value).is_integer():
            return f"{int(value)}um"

        return f"{value:g}um"

    values = [(f"{int(v)}" if float(v).is_integer() else f"{v:g}") for v in spacing_um]

    return ("x".join(values) + "um")


# ============================================================
# Main
# ============================================================

def main():
    for path in (BINARY_INPUT_PATH, CENTER_CSV_PATH, MRI_REFERENCE_PATH, ERODED_MASK_PATH,):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    os.makedirs(OUTPUT_DIR, exist_ok=True,)

    microscopy_nii = nib.load(BINARY_INPUT_PATH)

    dmri_nii = nib.load(MRI_REFERENCE_PATH)

    eroded_nii = nib.load(ERODED_MASK_PATH)

    microscopy_data = np.asanyarray(microscopy_nii.dataobj)

    if microscopy_data.ndim != 3:
        raise ValueError("Microscopy binary input must be 3D.")

    print_geometry("Microscopy geometry", microscopy_nii, MICROSCOPY_WORLD_UNIT,)

    print_geometry("Native dMRI geometry", dmri_nii, DMRI_WORLD_UNIT,)

    print_geometry("Eroded mask geometry", eroded_nii, DMRI_WORLD_UNIT,)

    check_fov(microscopy_nii, dmri_nii,)

    centers_voxel = load_centers(CENTER_CSV_PATH)

    print()
    print("========================================")
    print("Component centers")
    print("========================================")
    print("[INFO] Total centers:", len(centers_voxel),)

    # ========================================================
    # Native eroded mask
    # ========================================================

    native_shape = tuple(
        dmri_nii.shape[:3]
    )

    native_affine = (dmri_nii.affine.copy())

    native_spacing_um = get_spacing_um(dmri_nii, DMRI_WORLD_UNIT,)

    native_eroded_mask = resample_mask(eroded_nii, native_shape, native_affine,)

    print()
    print("========================================")
    print("Native eroded mask")
    print("========================================")
    print("[INFO] Mask voxels:", int(native_eroded_mask.sum()),)

    if SAVE_RESAMPLED_ERODED_MASK:
        save_nifti(native_eroded_mask, native_affine, dmri_nii, ERODED_MASK_NATIVE_PATH, np.uint8,)

    # ========================================================
    # Binary -> native dMRI
    # ========================================================

    print()
    print("========================================")
    print("Binary microscopy -> native dMRI")
    print("========================================")

    binary_native = generate_binary_native(microscopy_data, microscopy_nii, dmri_nii,)

    binary_native_masked = (binary_native * native_eroded_mask).astype(np.uint8)

    print("[INFO] Binary voxels before mask:", int(binary_native.sum()),)

    print("[INFO] Binary voxels after mask:", int(binary_native_masked.sum()),)

    print("[INFO] Binary voxels removed:", int(binary_native.sum() - binary_native_masked.sum()),)

    save_nifti(binary_native, native_affine, dmri_nii, BINARY_NATIVE_PATH, np.uint8,)

    save_nifti(binary_native_masked, native_affine, dmri_nii, BINARY_NATIVE_MASKED_PATH, np.uint8,)

    # ========================================================
    # Point count -> native dMRI
    # ========================================================

    print()
    print("========================================")
    print("Point count -> native dMRI")
    print("========================================")

    native_count = map_points_to_grid(centers_voxel, microscopy_nii, native_shape, native_spacing_um,)

    native_count_masked = (native_count * native_eroded_mask.astype(np.uint32))

    print("[INFO] Points before mask:", int(native_count.sum()),)

    print("[INFO] Points after mask:", int(native_count_masked.sum()),)

    print("[INFO] Points removed by mask:", int(native_count.sum() - native_count_masked.sum()),)

    save_nifti(native_count, native_affine, dmri_nii, POINT_COUNT_NATIVE_PATH, np.uint32,)

    save_nifti(native_count_masked, native_affine, dmri_nii, POINT_COUNT_NATIVE_MASKED_PATH, np.uint32,)

    # ========================================================
    # Custom analysis grids
    # ========================================================

    for requested_spacing in CUSTOM_GRID_SIZES_UM:

        requested_spacing = np.asarray(requested_spacing, dtype=np.float64,)

        tag = grid_tag(requested_spacing)

        print()
        print("=" * 70)
        print(f"CUSTOM GRID: {tag}")
        print("=" * 70)

        target_shape, target_affine = (build_target_geometry(dmri_nii, requested_spacing, DMRI_WORLD_UNIT,))

        # ----------------------------------------------------
        # Point count
        # ----------------------------------------------------

        custom_count = map_points_to_grid(
            centers_voxel,
            microscopy_nii,
            target_shape,
            requested_spacing,
        )

        # ----------------------------------------------------
        # Eroded mask -> custom grid
        # nearest neighbour
        # ----------------------------------------------------

        custom_eroded_mask = resample_mask(
            eroded_nii,
            target_shape,
            target_affine,
        )

        custom_count_masked = (custom_count * custom_eroded_mask.astype(np.uint32))

        print("[INFO] Custom mask voxels:", int(custom_eroded_mask.sum()),)

        print("[INFO] Points before mask:", int(custom_count.sum()),)

        print("[INFO] Points after mask:", int(custom_count_masked.sum()),)

        print("[INFO] Points removed by mask:", int(custom_count.sum() - custom_count_masked.sum()),)

        count_path = os.path.join(OUTPUT_DIR, f"Topro3_point_count_{tag}.nii.gz",)

        masked_count_path = os.path.join(OUTPUT_DIR, f"Topro3_point_count_{tag}_masked.nii.gz",)

        custom_mask_path = os.path.join(OUTPUT_DIR, f"eroded_mask_{tag}.nii.gz",)

        save_nifti(custom_count, target_affine, dmri_nii, count_path, np.uint32,)

        save_nifti(custom_count_masked, target_affine, dmri_nii, masked_count_path, np.uint32,)

        if SAVE_RESAMPLED_ERODED_MASK:
            save_nifti(custom_eroded_mask, target_affine, dmri_nii, custom_mask_path, np.uint8,)

    # ========================================================
    # Finish
    # ========================================================

    print()
    print("=" * 70)
    print("FINISHED")
    print("=" * 70)

    print("Binary native:")
    print(" ", BINARY_NATIVE_PATH,)

    print("Binary native masked:")
    print(" ", BINARY_NATIVE_MASKED_PATH,)

    print("Point count native:")
    print(" ", POINT_COUNT_NATIVE_PATH,)

    print("Point count native masked:")
    print(" ", POINT_COUNT_NATIVE_MASKED_PATH,)

    for spacing in CUSTOM_GRID_SIZES_UM:
        tag = grid_tag(spacing)

        print(f"Point count {tag}:")
        print(" ", os.path.join(OUTPUT_DIR, f"Topro3_point_count_{tag}.nii.gz",),)

        print(f"Point count {tag} masked:")
        print(" ", os.path.join(OUTPUT_DIR, f"Topro3_point_count_{tag}_masked.nii.gz",),)

    print("=" * 70)


if __name__ == "__main__":
    main()