#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

WORK_DIR = r"/autofs/arch11/DATA/HOMES/yuchen/Minor/Organoids/Nifti_data/case2_young/data_final"
CENTER_CSV_PATH = os.path.join(WORK_DIR, "component_centers.csv")
MICROSCOPY_REFERENCE_PATH = os.path.join(WORK_DIR, "case2_Topro3_origin.nii")
MASK_PATH = os.path.join(WORK_DIR, "case2_mask_high_FA.nii.gz")
REFERENCE_DMRI_PATH = os.path.join(WORK_DIR, "registered_Addition_native_Bvalue/registered/dti_FA_final_LAS_registered_native.nii.gz")
OUTPUT_PATH = os.path.join(WORK_DIR, "microscopy_density_inverted_masked.nii.gz")

CSV_WORLD_UNIT = "um"
MICROSCOPY_WORLD_UNIT = "um"
DMRI_WORLD_UNIT = "mm"
DENSITY_UNIT_VOLUME_UM3 = 8000


def to_um_factor(unit):
    if unit.lower() in ("um", "µm"):
        return 1.0
    if unit.lower() == "mm":
        return 1000.0
    raise ValueError(f"Unsupported unit: {unit}")


def affine_to_um(affine, unit):
    out = affine.copy().astype(np.float64)
    out[:3, :] *= to_um_factor(unit)
    return out


def get_spacing_um(nii, unit):
    return np.linalg.norm(affine_to_um(nii.affine, unit)[:3, :3], axis=0)


def get_voxel_volume_um3(nii, unit):
    return float(abs(np.linalg.det(affine_to_um(nii.affine, unit)[:3, :3])))


def load_centers_voxel(csv_path, microscopy_nii):
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        has_world = {"world_x", "world_y", "world_z"}.issubset(fields)
        has_voxel = {"voxel_center_x", "voxel_center_y", "voxel_center_z"}.issubset(fields)
        if not has_world and not has_voxel:
            raise ValueError(f"Unsupported center CSV columns: {fields}")

        centers = []
        if has_world:
            factor = to_um_factor(CSV_WORLD_UNIT)
            for row in reader:
                try:
                    centers.append([float(row["world_x"]) * factor,
                                    float(row["world_y"]) * factor,
                                    float(row["world_z"]) * factor])
                except (ValueError, TypeError):
                    pass
            centers = np.asarray(centers, dtype=np.float64)
            centers = nib.affines.apply_affine(
                np.linalg.inv(affine_to_um(microscopy_nii.affine, MICROSCOPY_WORLD_UNIT)), centers)
        else:
            for row in reader:
                try:
                    centers.append([float(row["voxel_center_x"]),
                                    float(row["voxel_center_y"]),
                                    float(row["voxel_center_z"])])
                except (ValueError, TypeError):
                    pass
            centers = np.asarray(centers, dtype=np.float64)

    if len(centers) == 0:
        raise ValueError("No valid nuclei centres found.")
    return centers


def load_mask_on_reference(mask_path, reference_nii):
    mask_nii = nib.load(mask_path)
    mask = np.asanyarray(mask_nii.dataobj)
    if mask.ndim == 4:
        mask = mask[..., 0]
    mask = mask > 0

    if mask.shape == reference_nii.shape[:3] and np.allclose(mask_nii.affine, reference_nii.affine, atol=1e-5):
        return mask

    return resample_from_to(
        nib.Nifti1Image(mask.astype(np.uint8), mask_nii.affine),
        (reference_nii.shape[:3], reference_nii.affine),
        order=0, mode="constant", cval=0
    ).get_fdata(dtype=np.float32) > 0.5


def map_nuclei_to_dmri(centers_voxel, microscopy_nii, dmri_nii):
    microscopy_shape = np.asarray(microscopy_nii.shape[:3])
    dmri_shape = np.asarray(dmri_nii.shape[:3])
    microscopy_spacing_um = get_spacing_um(microscopy_nii, MICROSCOPY_WORLD_UNIT)
    dmri_spacing_um = get_spacing_um(dmri_nii, DMRI_WORLD_UNIT)

    inside_micro = np.all((centers_voxel >= -0.5) & (centers_voxel < microscopy_shape - 0.5), axis=1)
    points_um = (centers_voxel + 0.5) * microscopy_spacing_um
    indices = np.floor(points_um / dmri_spacing_um).astype(np.int64)
    inside_dmri = np.all((indices >= 0) & (indices < dmri_shape), axis=1)
    indices = indices[inside_micro & inside_dmri]

    flat_ids = np.ravel_multi_index(indices.T, tuple(dmri_shape))
    counts = np.bincount(flat_ids, minlength=int(np.prod(dmri_shape)))
    return counts.reshape(tuple(dmri_shape)).astype(np.int32)


for path in (CENTER_CSV_PATH, MICROSCOPY_REFERENCE_PATH, MASK_PATH, REFERENCE_DMRI_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

microscopy_nii = nib.load(MICROSCOPY_REFERENCE_PATH)
reference_nii = nib.load(REFERENCE_DMRI_PATH)
centers_voxel = load_centers_voxel(CENTER_CSV_PATH, microscopy_nii)
foreground_mask = load_mask_on_reference(MASK_PATH, reference_nii)
nuclei_count = map_nuclei_to_dmri(centers_voxel, microscopy_nii, reference_nii)

voxel_volume_um3 = get_voxel_volume_um3(reference_nii, DMRI_WORLD_UNIT)
density = nuclei_count.astype(np.float64) / voxel_volume_um3 * DENSITY_UNIT_VOLUME_UM3

# Background is defined ONLY by MASK_PATH. Zero-density voxels inside the mask are valid foreground.
output = np.zeros(reference_nii.shape[:3], dtype=np.float32)
values = density[foreground_mask]

if values.size == 0:
    raise ValueError("Foreground mask contains no voxels on the reference dMRI grid.")

dmin = float(values.min())
dmax = float(values.max())
if dmax > dmin:
    output[foreground_mask] = (dmax - values) / (dmax - dmin)
else:
    output[foreground_mask] = 1.0

header = reference_nii.header.copy()
out_nii = nib.Nifti1Image(output, reference_nii.affine, header)
out_nii.set_data_dtype(np.float32)
qform, qcode = reference_nii.get_qform(coded=True)
sform, scode = reference_nii.get_sform(coded=True)
if qform is not None:
    out_nii.set_qform(qform, int(qcode))
if sform is not None:
    out_nii.set_sform(sform, int(scode))
nib.save(out_nii, OUTPUT_PATH)

print("[INFO] Nuclei centres:", len(centers_voxel))
print("[INFO] Foreground mask voxels:", int(foreground_mask.sum()))
print("[INFO] Density range inside mask:", dmin, dmax)
print("[INFO] Output intensity range inside mask:", float(output[foreground_mask].min()), float(output[foreground_mask].max()))
print("[INFO] Saved:", OUTPUT_PATH)
