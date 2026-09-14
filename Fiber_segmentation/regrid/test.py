#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Audit NIfTI geometry through the Cellpose pipeline and save header-preserved test maps."""

import glob
import os

import nibabel as nib
import numpy as np
import pandas as pd


# ============================================================
# User settings
# ============================================================

DATA_DIR = (
    r"/autofs/arch11/DATA/HOMES/yuchen/Minor/Organoids/Nifti_data/"
    r"case2_young_Paw_Ventricles/data_new"
)

WORK_DIR = (
    r"/autofs/arch11/DATA/HOMES/yuchen/Minor/Organoids/Nifti_data/seg/"
    r"cellpose_patch_centers_origin_masked_input_expand10"
)

ORIGIN_PATH = os.path.join(DATA_DIR, "case2_Topro3_origin.nii")
HIGH_PATH = os.path.join(DATA_DIR, "case2_Topro3_high.nii")
PATCH_DIR = os.path.join(WORK_DIR, "patch_predictions")
MERGED_DIR = os.path.join(WORK_DIR, "merged_test_results")
INSTANCE_PATH = os.path.join(MERGED_DIR, "merged_instance_mask.nii.gz")
CENTER_MASK_PATH = os.path.join(MERGED_DIR, "component_centers_mask.nii.gz")
CENTER_CSV_PATH = os.path.join(MERGED_DIR, "component_centers.csv")
OUTPUT_DIR = os.path.join(MERGED_DIR, "header_preserved_tests")

MIN_COMPONENT_VOXELS = 0


# ============================================================
# Helper functions
# ============================================================

def print_geometry(name, path):
    """Print stored forms, nibabel fallback geometry and physical extents."""
    if not os.path.exists(path):
        print(f"\n[MISSING] {name}: {path}")
        return None

    nii = nib.load(path)
    header = nii.header
    qform, qcode = nii.get_qform(coded=True)
    sform, scode = nii.get_sform(coded=True)
    base_affine = header.get_base_affine()
    zooms = np.asarray(header.get_zooms()[:3], dtype=np.float64)
    shape = np.asarray(nii.shape[:3], dtype=np.int64)
    extent_native = (shape - 1) * zooms
    spatial_unit = header.get_xyzt_units()[0]
    unit_to_mm = 0.001 if spatial_unit == "micron" else 1000.0 if spatial_unit == "meter" else 1.0

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    print("Path:", path)
    print("Shape XYZ:", tuple(int(v) for v in shape))
    print("Spacing XYZ:", tuple(float(v) for v in zooms))
    print("Spatial unit:", spatial_unit)
    print("qform/sform codes:", int(qcode), int(scode))
    print("Uses nibabel fallback affine:", bool(int(qcode) == 0 and int(scode) == 0))
    print("Loaded affine equals base affine:", bool(np.allclose(nii.affine, base_affine, atol=1e-6)))
    print("Axis codes from loaded affine:", nib.aff2axcodes(nii.affine))
    print("Full extent XYZ in mm:", tuple(float(v) for v in extent_native * unit_to_mm))
    print("Half extent XYZ in native unit:", tuple(float(v) for v in extent_native / 2.0))
    print("Loaded affine:\n", nii.affine, sep="")
    print("Base affine:\n", base_affine, sep="")
    print("Stored qform:\n", qform, sep="")
    print("Stored sform:\n", sform, sep="")
    return nii


def save_like_reference_without_promoting_forms(data, reference_nii, output_path):
    """Preserve the reference header and leave absent qform/sform absent."""
    header = reference_nii.header.copy()
    header.set_data_shape(data.shape)
    header.set_data_dtype(np.float32)
    qform, qcode = reference_nii.get_qform(coded=True)
    sform, scode = reference_nii.get_sform(coded=True)
    output_nii = nib.Nifti1Image(data.astype(np.float32, copy=False), None, header)

    if int(qcode) > 0:
        output_nii.set_qform(qform, code=int(qcode))
    else:
        output_nii.header["qform_code"] = 0

    if int(scode) > 0:
        output_nii.set_sform(sform, code=int(scode))
    else:
        output_nii.header["sform_code"] = 0

    nib.save(output_nii, output_path)


def build_count_map(coordinates, target_shape):
    """Round coordinates and count centers on one target voxel grid."""
    indices = np.rint(coordinates).astype(np.int64)
    valid = (
        (indices[:, 0] >= 0) & (indices[:, 0] < target_shape[0]) &
        (indices[:, 1] >= 0) & (indices[:, 1] < target_shape[1]) &
        (indices[:, 2] >= 0) & (indices[:, 2] < target_shape[2])
    )
    count_map = np.zeros(target_shape, dtype=np.float32)
    indices = indices[valid]
    if indices.size:
        np.add.at(count_map, (indices[:, 0], indices[:, 1], indices[:, 2]), 1.0)
    return count_map, int(np.count_nonzero(valid)), int(np.count_nonzero(~valid))


# ============================================================
# Audit complete pipeline
# ============================================================

origin_nii = print_geometry("ORIGINAL SEGMENTATION INPUT", ORIGIN_PATH)
high_nii = print_geometry("HIGH REFERENCE", HIGH_PATH)

patch_paths = sorted(glob.glob(os.path.join(PATCH_DIR, "*.nii")) + glob.glob(os.path.join(PATCH_DIR, "*.nii.gz")))
if patch_paths:
    print_geometry("FIRST SAVED PATCH", patch_paths[0])
else:
    print("\n[MISSING] No patch prediction found in:", PATCH_DIR)

print_geometry("MERGED INSTANCE MASK", INSTANCE_PATH)
print_geometry("COMPONENT CENTER MASK", CENTER_MASK_PATH)

if origin_nii is None or high_nii is None:
    raise RuntimeError("Origin and high images are required.")

origin_qcode = int(origin_nii.header["qform_code"])
origin_scode = int(origin_nii.header["sform_code"])
high_qcode = int(high_nii.header["qform_code"])
high_scode = int(high_nii.header["sform_code"])

print("\n" + "=" * 80)
print("AUTOMATIC DIAGNOSIS")
print("=" * 80)

if origin_qcode == 0 and origin_scode == 0:
    print("[FOUND] Origin has no valid qform or sform.")
    print("[FOUND] origin_nii.affine is a nibabel fallback affine, not a stored physical origin.")
else:
    print("[INFO] Origin contains at least one valid stored transform.")

if high_qcode == 0 and high_scode == 0:
    print("[FOUND] High has no valid qform or sform.")
    print("[FOUND] high_nii.affine is a nibabel fallback affine, not a stored physical origin.")
else:
    print("[INFO] High contains at least one valid stored transform.")

high_spacing = np.asarray(high_nii.header.get_zooms()[:3], dtype=np.float64)
high_shape = np.asarray(high_nii.shape[:3], dtype=np.int64)
high_extent = (high_shape - 1) * high_spacing
unit = high_nii.header.get_xyzt_units()[0]
unit_to_mm = 0.001 if unit == "micron" else 1000.0 if unit == "meter" else 1.0
print("High full FOV XYZ in mm:", tuple(float(v) for v in high_extent * unit_to_mm))
print("A manual shift close to one of these values indicates an axis sign/header mismatch.")


# ============================================================
# Save decisive target-grid tests
# ============================================================

if not os.path.exists(CENTER_CSV_PATH):
    raise FileNotFoundError(CENTER_CSV_PATH)

os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(CENTER_CSV_PATH)
required = ["voxel_count", "voxel_center_x", "voxel_center_y", "voxel_center_z"]
missing = [column for column in required if column not in df.columns]
if missing:
    raise ValueError(f"Missing CSV columns: {missing}")

df = df[required].dropna().copy()
if MIN_COMPONENT_VOXELS > 0:
    df = df[df["voxel_count"] >= MIN_COMPONENT_VOXELS].copy()

source_shape = np.asarray(origin_nii.shape[:3], dtype=np.float64)
target_shape = np.asarray(high_nii.shape[:3], dtype=np.int64)
source_voxel = np.column_stack([
    df["voxel_center_x"].to_numpy(dtype=np.float64),
    df["voxel_center_y"].to_numpy(dtype=np.float64),
    df["voxel_center_z"].to_numpy(dtype=np.float64),
])

scale_center = (source_shape - 1.0) / np.maximum(target_shape.astype(np.float64) - 1.0, 1.0)
normal_coordinates = source_voxel / scale_center
flip_y_coordinates = normal_coordinates.copy()
flip_y_coordinates[:, 1] = target_shape[1] - 1 - flip_y_coordinates[:, 1]

normal_map, normal_inside, normal_outside = build_count_map(normal_coordinates, tuple(target_shape))
flip_y_map, flip_inside, flip_outside = build_count_map(flip_y_coordinates, tuple(target_shape))

normal_path = os.path.join(OUTPUT_DIR, "scale_center_normal_header_preserved.nii.gz")
flip_y_path = os.path.join(OUTPUT_DIR, "scale_center_flip_y_header_preserved.nii.gz")
save_like_reference_without_promoting_forms(normal_map, high_nii, normal_path)
save_like_reference_without_promoting_forms(flip_y_map, high_nii, flip_y_path)

print("\n" + "=" * 80)
print("HEADER-PRESERVED TEST OUTPUTS")
print("=" * 80)
print("Normal inside/outside:", normal_inside, normal_outside)
print("Flip-Y inside/outside:", flip_inside, flip_outside)
print("Normal:", normal_path)
print("Flip Y:", flip_y_path)
print("Load HIGH plus these two files in a fresh Slicer scene.")
print("If normal aligns, the earlier Y flip was entirely caused by promoted fallback geometry.")
print("If flip Y aligns, the high pixel array itself is Y-reversed relative to origin.")
print("=" * 80)