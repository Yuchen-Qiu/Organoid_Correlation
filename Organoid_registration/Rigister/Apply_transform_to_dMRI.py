#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import glob
import json
import shutil
import subprocess
import numpy as np
import SimpleITK as sitk

# ============================================================
# User settings
# ============================================================

ROOT = "/autofs/arch11/DATA/HOMES/yuchen/Minor/Organoids/Nifti_data"
DATA = os.path.join(ROOT, "case2_young", "data_final")
B_DIR = os.path.join(DATA, "data_final_bvalue")

FIXED_PATH = os.path.join(DATA, "case2_Topro3_high.nii")
MASK_PATH = os.path.join(DATA, "case2_FA_mask_LAS_eroded_2vox.nii.gz")

SCALAR_IMAGE_PATHS = [
    os.path.join(DATA, "dti_FA_final_LAS.nii.gz"),
    os.path.join(DATA, "dti_MD_final_LAS.nii.gz"),
    os.path.join(DATA, "dti_S0_final_LAS.nii.gz"),
]

B_IMAGES = [os.path.join(B_DIR, f"case2_bvalue_{i:03d}.nii.gz") for i in range(1, 10)]

OUTPUT_DIR = os.path.join(DATA, "registered_final")
REGISTERED_DIR = os.path.join(OUTPUT_DIR, "registered")
MASKED_REGISTERED_DIR = os.path.join(OUTPUT_DIR, "registered_masked")
QC_DIR = os.path.join(OUTPUT_DIR, "QC")

TRANSFORMIX = "/homes/yuchen/software/elastix-5.3.1/bin/transformix"
COMPOSITE_MANIFEST = os.path.join(
    ROOT, "registration_final_test", "case2_Bvalue_rigid_affine_bspline",
    "COMPOSITE_TRANSFORM", "composite_manifest.json"
)

MI_BINS = 64


# ============================================================
# Helpers
# ============================================================

def load_composite_transform():
    with open(COMPOSITE_MANIFEST, "r") as f:
        manifest = json.load(f)
    path = manifest["final_composite_transform"]
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def read_transform_parameter_file(path):
    params = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("(") and line.endswith(")"):
                parts = line[1:-1].split(None, 1)
                if len(parts) == 2:
                    params[parts[0]] = parts[1]
    return params


def write_transform_parameter_file(params, path):
    with open(path, "w") as f:
        for key, value in params.items():
            f.write(f"({key} {value})\n")


def make_native_spacing_transform(composite_transform, fixed_img, native_mri_img, output_path):
    params = read_transform_parameter_file(composite_transform)

    fixed_size = np.asarray(fixed_img.GetSize(), dtype=np.int64)
    fixed_spacing = np.asarray(fixed_img.GetSpacing(), dtype=np.float64)
    native_spacing = np.asarray(native_mri_img.GetSpacing(), dtype=np.float64)
    fixed_extent = (fixed_size - 1) * fixed_spacing
    output_size = np.round(fixed_extent / native_spacing).astype(np.int64) + 1
    output_size = np.maximum(output_size, 1)

    origin = np.asarray(fixed_img.GetOrigin(), dtype=np.float64)
    direction = np.asarray(fixed_img.GetDirection(), dtype=np.float64).reshape(3, 3)

    params["Size"] = " ".join(str(int(v)) for v in output_size)
    params["Index"] = "0 0 0"
    params["Spacing"] = " ".join(f"{float(v):.17g}" for v in native_spacing)
    params["Origin"] = " ".join(f"{float(v):.17g}" for v in origin)
    params["Direction"] = " ".join(f"{float(v):.17g}" for v in direction.flatten(order="F"))
    params["UseDirectionCosines"] = '"true"'
    params["ResampleInterpolator"] = '"FinalLinearInterpolator"'
    params["FinalBSplineInterpolationOrder"] = "1"
    params["Resampler"] = '"DefaultResampler"'
    params["DefaultPixelValue"] = "0"
    params["ResultImagePixelType"] = '"float"'
    params["ResultImageFormat"] = '"nii"'
    params["CompressResultImage"] = '"false"'

    write_transform_parameter_file(params, output_path)
    return output_size, native_spacing


def make_mask_transform(native_transform_path, mask_transform_path):
    params = read_transform_parameter_file(native_transform_path)
    params["ResampleInterpolator"] = '"FinalNearestNeighborInterpolator"'
    params["FinalBSplineInterpolationOrder"] = "0"
    params["ResultImagePixelType"] = '"unsigned char"'
    write_transform_parameter_file(params, mask_transform_path)


def find_result_image(folder):
    candidates = [
        os.path.join(folder, "result.nii"),
        os.path.join(folder, "result.nii.gz"),
        os.path.join(folder, "result.mhd"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    found = sorted(glob.glob(os.path.join(folder, "result.*")))
    if not found:
        raise RuntimeError(f"Transformix result not found: {folder}")
    return found[0]


def transform_image(input_path, transform_path, output_path, temp_dir, pixel_type=sitk.sitkFloat32):
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    subprocess.run([
        TRANSFORMIX,
        "-in", input_path,
        "-out", temp_dir,
        "-tp", transform_path,
    ], check=True)

    result_path = find_result_image(temp_dir)
    result = sitk.ReadImage(result_path, pixel_type)
    sitk.WriteImage(result, output_path)
    return result


def create_native_reference(fixed_img, output_size, output_spacing):
    reference = sitk.Image([int(v) for v in output_size], sitk.sitkFloat32)
    reference.SetSpacing(tuple(float(v) for v in output_spacing))
    reference.SetOrigin(fixed_img.GetOrigin())
    reference.SetDirection(fixed_img.GetDirection())
    return reference


def resample_microscopy_to_native(fixed_img, reference, output_path):
    result = sitk.Resample(
        fixed_img, reference, sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkLinear, 0.0, sitk.sitkFloat32
    )
    sitk.WriteImage(result, output_path)
    return result


def check_same_geometry(img1, img2):
    if img1.GetSize() != img2.GetSize():
        return False
    if not np.allclose(img1.GetSpacing(), img2.GetSpacing(), atol=1e-8):
        return False
    if not np.allclose(img1.GetOrigin(), img2.GetOrigin(), atol=1e-8):
        return False
    if not np.allclose(img1.GetDirection(), img2.GetDirection(), atol=1e-8):
        return False
    return True


def apply_mask(image, mask, output_path):
    if not check_same_geometry(image, mask):
        raise RuntimeError("Image and mask geometry mismatch.")

    data = sitk.GetArrayFromImage(image).astype(np.float32)
    mask_data = sitk.GetArrayFromImage(mask) > 0
    data[~mask_data] = 0

    output = sitk.GetImageFromArray(data.astype(np.float32))
    output.CopyInformation(image)
    sitk.WriteImage(output, output_path)
    return output


def calculate_mi_ncc(microscopy, moving_img):
    if not check_same_geometry(microscopy, moving_img):
        raise RuntimeError("Microscopy and moving image geometry mismatch.")

    fixed = sitk.GetArrayFromImage(microscopy).astype(np.float64)
    moving = sitk.GetArrayFromImage(moving_img).astype(np.float64)
    valid = np.isfinite(fixed) & np.isfinite(moving)
    fixed = fixed[valid]
    moving = moving[valid]

    if fixed.size < 100:
        return np.nan, np.nan, int(fixed.size)

    f_min, f_max = np.percentile(fixed, [1, 99])
    m_min, m_max = np.percentile(moving, [1, 99])

    if abs(f_max - f_min) < 1e-12 or abs(m_max - m_min) < 1e-12:
        mi = np.nan
    else:
        fixed_mi = np.clip(fixed, f_min, f_max)
        moving_mi = np.clip(moving, m_min, m_max)
        hist, _, _ = np.histogram2d(
            fixed_mi, moving_mi, bins=MI_BINS,
            range=[[f_min, f_max], [m_min, m_max]]
        )

        if np.sum(hist) <= 0:
            mi = np.nan
        else:
            pxy = hist / np.sum(hist)
            px = np.sum(pxy, axis=1)
            py = np.sum(pxy, axis=0)
            expected = px[:, None] * py[None, :]
            valid_hist = (pxy > 0) & (expected > 0)
            mi = np.sum(pxy[valid_hist] * np.log(pxy[valid_hist] / expected[valid_hist]))

    fixed_centered = fixed - np.mean(fixed)
    moving_centered = moving - np.mean(moving)
    denominator = np.sqrt(np.sum(fixed_centered ** 2) * np.sum(moving_centered ** 2))

    if denominator < 1e-12:
        ncc = np.nan
    else:
        ncc = np.sum(fixed_centered * moving_centered) / denominator

    return float(mi), float(ncc), int(fixed.size)


def print_geometry(name, img):
    print(name)
    print(f"  Size:      {img.GetSize()}")
    print(f"  Spacing:   {img.GetSpacing()}")
    print(f"  Origin:    {img.GetOrigin()}")
    print(f"  Direction: {img.GetDirection()}")


def get_image_name(path):
    name = os.path.basename(path)
    if name.endswith(".nii.gz"):
        return name[:-7]
    return os.path.splitext(name)[0]


# ============================================================
# Main
# ============================================================

def main():
    required_paths = [
        TRANSFORMIX,
        FIXED_PATH,
        MASK_PATH,
        COMPOSITE_MANIFEST,
        *SCALAR_IMAGE_PATHS,
        *B_IMAGES,
    ]

    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    for folder in [OUTPUT_DIR, REGISTERED_DIR, MASKED_REGISTERED_DIR, QC_DIR]:
        os.makedirs(folder, exist_ok=True)

    composite_transform = load_composite_transform()
    fixed_img = sitk.ReadImage(FIXED_PATH, sitk.sitkFloat32)
    native_mri_img = sitk.ReadImage(B_IMAGES[0], sitk.sitkFloat32)

    print("\n============================================================")
    print("Input geometry")
    print("============================================================")
    print_geometry("Original b-value:", native_mri_img)
    print_geometry("Fixed microscopy:", fixed_img)

    for path in SCALAR_IMAGE_PATHS:
        img = sitk.ReadImage(path, sitk.sitkFloat32)
        if not check_same_geometry(native_mri_img, img):
            raise RuntimeError(
                f"{os.path.basename(path)} does not have the same original geometry as the b-value images."
            )

    native_transform = os.path.join(OUTPUT_DIR, "TransformParameters_NATIVE_SPACING.txt")
    output_size, output_spacing = make_native_spacing_transform(
        composite_transform, fixed_img, native_mri_img, native_transform
    )

    print("\n============================================================")
    print("Output grid")
    print("============================================================")
    print(f"Size: {tuple(int(v) for v in output_size)}")
    print(f"Spacing: {tuple(float(v) for v in output_spacing)}")
    print(f"Origin: {fixed_img.GetOrigin()}")
    print(f"Direction: {fixed_img.GetDirection()}")

    native_reference = create_native_reference(fixed_img, output_size, output_spacing)

    microscopy_native_path = os.path.join(QC_DIR, "microscopy_native_spacing.nii.gz")
    microscopy_native = resample_microscopy_to_native(
        fixed_img, native_reference, microscopy_native_path
    )

    print("\n============================================================")
    print("Transforming erosion mask")
    print("============================================================")

    mask_transform = os.path.join(OUTPUT_DIR, "TransformParameters_NATIVE_SPACING_MASK.txt")
    make_mask_transform(native_transform, mask_transform)

    registered_mask_path = os.path.join(
        OUTPUT_DIR, "erosion_mask_registered_native.nii.gz"
    )
    registered_mask = transform_image(
        MASK_PATH,
        mask_transform,
        registered_mask_path,
        os.path.join(OUTPUT_DIR, "_transformix_mask"),
        sitk.sitkUInt8,
    )

    mask_arr = sitk.GetArrayFromImage(registered_mask) > 0
    registered_mask = sitk.GetImageFromArray(mask_arr.astype(np.uint8))
    registered_mask.CopyInformation(native_reference)
    sitk.WriteImage(registered_mask, registered_mask_path)
    print(f"Registered mask voxels: {int(mask_arr.sum())}")

    print("\n============================================================")
    print("Transforming scalar images")
    print("============================================================")

    for input_path in SCALAR_IMAGE_PATHS:
        name = get_image_name(input_path)
        registered_path = os.path.join(
            REGISTERED_DIR, f"{name}_registered_native.nii.gz"
        )
        masked_path = os.path.join(
            MASKED_REGISTERED_DIR, f"{name}_registered_native_masked.nii.gz"
        )

        registered = transform_image(
            input_path,
            native_transform,
            registered_path,
            os.path.join(OUTPUT_DIR, "_transformix_tmp", name),
        )
        apply_mask(registered, registered_mask, masked_path)

        print(f"{name}:")
        print(f"  Registered: {registered_path}")
        print(f"  Masked:     {masked_path}")

    print("\n============================================================")
    print("Transforming b-images")
    print("============================================================")

    metrics = []

    for i, input_path in enumerate(B_IMAGES, start=1):
        name = f"case2_bvalue_{i:03d}"
        registered_path = os.path.join(
            REGISTERED_DIR, f"{name}_registered_native.nii.gz"
        )
        masked_path = os.path.join(
            MASKED_REGISTERED_DIR, f"{name}_registered_native_masked.nii.gz"
        )

        registered = transform_image(
            input_path,
            native_transform,
            registered_path,
            os.path.join(OUTPUT_DIR, "_transformix_tmp", f"b{i:03d}"),
        )
        apply_mask(registered, registered_mask, masked_path)

        mi, ncc, n_voxels = calculate_mi_ncc(microscopy_native, registered)
        metrics.append({
            "bimage": f"b{i:03d}",
            "MI_registered": mi,
            "NCC_registered": ncc,
            "valid_voxels": n_voxels,
        })

        print(
            f"[{i}/{len(B_IMAGES)}] b{i:03d} | "
            f"MI={mi:.6f} | NCC={ncc:.6f}"
        )

    metrics_path = os.path.join(QC_DIR, "bvalue_vs_microscopy_metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        fieldnames = ["bimage", "MI_registered", "NCC_registered", "valid_voxels"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)

    print("\n============================================================")
    print("Finished")
    print("============================================================")
    print("\nRegistered mask:")
    print(registered_mask_path)
    print("\nRegistered images:")
    print(REGISTERED_DIR)
    print("\nRegistered + masked images:")
    print(MASKED_REGISTERED_DIR)
    print("\nMicroscopy at native spacing:")
    print(microscopy_native_path)
    print("\nMetrics:")
    print(metrics_path)


if __name__ == "__main__":
    main()
