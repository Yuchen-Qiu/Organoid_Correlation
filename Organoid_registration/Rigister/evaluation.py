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

# Fixed microscopy image
FIXED_PATH = os.path.join(DATA, "case2_Topro3_high.nii")

# Fixed evaluation mask in microscopy/fixed space
MASK_PATH = os.path.join(DATA, "case2_FA_mask_LAS_dilated_10vox_registered_native.nii.gz")

# Original, unregistered MRI images used for evaluation
BVALUE9_PATH = os.path.join(B_DIR, "case2_bvalue_009.nii.gz")
MD_PATH = os.path.join(DATA, "case2_MD_LAS.nii.gz")
FA_PATH = os.path.join(DATA, "case2_FA_LAS.nii.gz")
S0_PATH = os.path.join(DATA, "case2_S0_LAS.nii.gz")

TRANSFORMIX = "/homes/yuchen/software/elastix-5.3.1/bin/transformix"

REGISTRATION_DIR = os.path.join(
    ROOT,
    "registration_native_pipeline",
    "case2_FA_rigid_affine",
)
INPUT_IMAGE_LABEL = "FA"
REGISTRATION_LABEL = "Rigid_Affine"

EVAL_IMAGES = {
    "BVALUE9": BVALUE9_PATH,
    "MD": MD_PATH,
    "FA": FA_PATH,
    "S0": S0_PATH,
}

RUN_NAME = (
    INPUT_IMAGE_LABEL.replace(":", "_").replace(",", "_").replace(" ", "_")
    + "_"
    + REGISTRATION_LABEL.replace("+", "_").replace(" ", "_")
)

OUTPUT_DIR = os.path.join(DATA, "registration_metric_evaluation_erosion1voxelmask", RUN_NAME)
REGISTERED_DIR = os.path.join(OUTPUT_DIR, "registered")
QC_DIR = os.path.join(OUTPUT_DIR, "QC")
TABLE_CSV = os.path.join(DATA, "registration_metric_table.csv")
MI_BINS = 64


# ============================================================
# Helpers
# ============================================================

def load_transform_from_registration_dir(registration_dir):
    manifest_path = os.path.join(
        registration_dir,
        "COMPOSITE_TRANSFORM",
        "composite_manifest.json",
    )

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Composite manifest not found: {manifest_path}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    if "final_composite_transform" not in manifest:
        raise KeyError(f"'final_composite_transform' not found in: {manifest_path}")

    transform_path = manifest["final_composite_transform"]

    if not os.path.isabs(transform_path):
        transform_path = os.path.normpath(
            os.path.join(os.path.dirname(manifest_path), transform_path)
        )

    if not os.path.exists(transform_path):
        raise FileNotFoundError(f"Final composite transform not found: {transform_path}")

    return manifest_path, transform_path


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


def make_native_spacing_transform(transform_path, fixed_img, native_mri_img, output_path):
    params = read_transform_parameter_file(transform_path)

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
    params["Direction"] = " ".join(
        f"{float(v):.17g}" for v in direction.flatten(order="F")
    )
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


def transform_image(input_path, transform_path, output_path, temp_dir):
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)

    os.makedirs(temp_dir, exist_ok=True)

    subprocess.run(
        [TRANSFORMIX, "-in", input_path, "-out", temp_dir, "-tp", transform_path],
        check=True,
    )

    result_path = find_result_image(temp_dir)
    result = sitk.ReadImage(result_path, sitk.sitkFloat32)
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
        fixed_img,
        reference,
        sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )

    sitk.WriteImage(result, output_path)
    return result


def resample_mask_to_native(mask_img, reference, output_path):
    result = sitk.Resample(
        mask_img,
        reference,
        sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )

    result = sitk.Cast(result > 0, sitk.sitkUInt8)
    sitk.WriteImage(result, output_path)

    return result


def check_same_geometry(img1, img2):
    return (
        img1.GetSize() == img2.GetSize()
        and np.allclose(img1.GetSpacing(), img2.GetSpacing(), atol=1e-8)
        and np.allclose(img1.GetOrigin(), img2.GetOrigin(), atol=1e-8)
        and np.allclose(img1.GetDirection(), img2.GetDirection(), atol=1e-8)
    )


def calculate_mi_ncc(microscopy, moving_img, mask_img=None):
    if not check_same_geometry(microscopy, moving_img):
        raise RuntimeError("Microscopy and registered image geometry mismatch.")

    fixed = sitk.GetArrayFromImage(microscopy).astype(np.float64)
    moving = sitk.GetArrayFromImage(moving_img).astype(np.float64)

    valid = np.isfinite(fixed) & np.isfinite(moving)

    if mask_img is not None:
        if not check_same_geometry(microscopy, mask_img):
            raise RuntimeError("Microscopy and mask geometry mismatch.")

        mask = sitk.GetArrayFromImage(mask_img) > 0
        valid &= mask

    fixed = fixed[valid]
    moving = moving[valid]

    if fixed.size < 100:
        return np.nan, np.nan, int(fixed.size)

    # MI: 1-99 percentile clipping, 64 bins
    f_min, f_max = np.percentile(fixed, [1, 99])
    m_min, m_max = np.percentile(moving, [1, 99])

    if abs(f_max - f_min) < 1e-12 or abs(m_max - m_min) < 1e-12:
        mi = np.nan
    else:
        fixed_mi = np.clip(fixed, f_min, f_max)
        moving_mi = np.clip(moving, m_min, m_max)

        hist, _, _ = np.histogram2d(
            fixed_mi,
            moving_mi,
            bins=MI_BINS,
            range=[[f_min, f_max], [m_min, m_max]],
        )

        if np.sum(hist) <= 0:
            mi = np.nan
        else:
            pxy = hist / np.sum(hist)
            px = np.sum(pxy, axis=1)
            py = np.sum(pxy, axis=0)
            expected = px[:, None] * py[None, :]
            valid_hist = (pxy > 0) & (expected > 0)

            mi = np.sum(
                pxy[valid_hist]
                * np.log(pxy[valid_hist] / expected[valid_hist])
            )

    # NCC: original unclipped values
    fixed_centered = fixed - np.mean(fixed)
    moving_centered = moving - np.mean(moving)

    denominator = np.sqrt(
        np.sum(fixed_centered ** 2)
        * np.sum(moving_centered ** 2)
    )

    if denominator < 1e-12:
        ncc = np.nan
    else:
        ncc = np.sum(fixed_centered * moving_centered) / denominator

    return float(mi), float(ncc), int(fixed.size)


def save_run_metrics(metrics, output_path):
    fieldnames = [
        "image",
        "MI_global",
        "NCC_global",
        "valid_voxels_global",
        "MI_mask",
        "NCC_mask",
        "valid_voxels_mask",
        "registered_path",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for name in ["BVALUE9", "MD", "FA", "S0"]:
            writer.writerow({
                "image": name,
                "MI_global": f"{metrics[name]['MI_global']:.9f}",
                "NCC_global": f"{metrics[name]['NCC_global']:.9f}",
                "valid_voxels_global": metrics[name]["valid_voxels_global"],
                "MI_mask": f"{metrics[name]['MI_mask']:.9f}",
                "NCC_mask": f"{metrics[name]['NCC_mask']:.9f}",
                "valid_voxels_mask": metrics[name]["valid_voxels_mask"],
                "registered_path": metrics[name]["registered_path"],
            })


def upsert_summary_table(metrics):
    fieldnames = [
        "Input image",
        "Registration",
        "MI global(b-value 9)",
        "NCC global(b-value 9)",
        "MI mask(b-value 9)",
        "NCC mask(b-value 9)",
        "MI global(MD)",
        "NCC global(MD)",
        "MI mask(MD)",
        "NCC mask(MD)",
        "MI global(FA)",
        "NCC global(FA)",
        "MI mask(FA)",
        "NCC mask(FA)",
        "MI global(S0)",
        "NCC global(S0)",
        "MI mask(S0)",
        "NCC mask(S0)",
    ]

    new_row = {
        "Input image": INPUT_IMAGE_LABEL,
        "Registration": REGISTRATION_LABEL,

        "MI global(b-value 9)": f"{metrics['BVALUE9']['MI_global']:.9f}",
        "NCC global(b-value 9)": f"{metrics['BVALUE9']['NCC_global']:.9f}",
        "MI mask(b-value 9)": f"{metrics['BVALUE9']['MI_mask']:.9f}",
        "NCC mask(b-value 9)": f"{metrics['BVALUE9']['NCC_mask']:.9f}",

        "MI global(MD)": f"{metrics['MD']['MI_global']:.9f}",
        "NCC global(MD)": f"{metrics['MD']['NCC_global']:.9f}",
        "MI mask(MD)": f"{metrics['MD']['MI_mask']:.9f}",
        "NCC mask(MD)": f"{metrics['MD']['NCC_mask']:.9f}",

        "MI global(FA)": f"{metrics['FA']['MI_global']:.9f}",
        "NCC global(FA)": f"{metrics['FA']['NCC_global']:.9f}",
        "MI mask(FA)": f"{metrics['FA']['MI_mask']:.9f}",
        "NCC mask(FA)": f"{metrics['FA']['NCC_mask']:.9f}",

        "MI global(S0)": f"{metrics['S0']['MI_global']:.9f}",
        "NCC global(S0)": f"{metrics['S0']['NCC_global']:.9f}",
        "MI mask(S0)": f"{metrics['S0']['MI_mask']:.9f}",
        "NCC mask(S0)": f"{metrics['S0']['NCC_mask']:.9f}",
    }

    rows = []
    replaced = False

    if os.path.exists(TABLE_CSV):
        with open(TABLE_CSV, "r", newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                if (
                    row.get("Input image") == INPUT_IMAGE_LABEL
                    and row.get("Registration") == REGISTRATION_LABEL
                ):
                    rows.append(new_row)
                    replaced = True
                else:
                    rows.append({key: row.get(key, "") for key in fieldnames})

    if not replaced:
        rows.append(new_row)

    with open(TABLE_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_geometry(name, img):
    print(name)
    print(f"  Size:      {img.GetSize()}")
    print(f"  Spacing:   {img.GetSpacing()}")
    print(f"  Origin:    {img.GetOrigin()}")
    print(f"  Direction: {img.GetDirection()}")


# ============================================================
# Main
# ============================================================

def main():
    required_paths = [
        TRANSFORMIX,
        REGISTRATION_DIR,
        FIXED_PATH,
        MASK_PATH,
        BVALUE9_PATH,
        MD_PATH,
        FA_PATH,
        S0_PATH,
    ]

    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    manifest_path, transform_path = load_transform_from_registration_dir(
        REGISTRATION_DIR
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REGISTERED_DIR, exist_ok=True)
    os.makedirs(QC_DIR, exist_ok=True)

    fixed_img = sitk.ReadImage(FIXED_PATH, sitk.sitkFloat32)
    mask_img = sitk.ReadImage(MASK_PATH, sitk.sitkUInt8)
    native_mri_img = sitk.ReadImage(BVALUE9_PATH, sitk.sitkFloat32)

    print("\n============================================================")
    print(f"Run: {INPUT_IMAGE_LABEL} | {REGISTRATION_LABEL}")
    print("============================================================")
    print(f"Registration dir: {REGISTRATION_DIR}")
    print(f"Manifest: {manifest_path}")
    print(f"Transform: {transform_path}")
    print(f"Evaluation mask: {MASK_PATH}")

    print_geometry("Native MRI reference:", native_mri_img)
    print_geometry("Fixed microscopy:", fixed_img)
    print_geometry("Fixed evaluation mask:", mask_img)

    native_transform = os.path.join(
        OUTPUT_DIR,
        "TransformParameters_NATIVE_SPACING.txt",
    )

    output_size, output_spacing = make_native_spacing_transform(
        transform_path,
        fixed_img,
        native_mri_img,
        native_transform,
    )

    native_reference = create_native_reference(
        fixed_img,
        output_size,
        output_spacing,
    )

    microscopy_native_path = os.path.join(
        QC_DIR,
        "microscopy_native_spacing.nii.gz",
    )

    microscopy_native = resample_microscopy_to_native(
        fixed_img,
        native_reference,
        microscopy_native_path,
    )

    mask_native_path = os.path.join(
        QC_DIR,
        "evaluation_mask_native_spacing.nii.gz",
    )

    mask_native = resample_mask_to_native(
        mask_img,
        native_reference,
        mask_native_path,
    )

    print(
        f"Evaluation mask voxels: "
        f"{np.sum(sitk.GetArrayFromImage(mask_native) > 0)}"
    )

    output_names = {
        "BVALUE9": "bvalue_009_registered_native.nii.gz",
        "MD": "MD_registered_native.nii.gz",
        "FA": "FA_registered_native.nii.gz",
        "S0": "S0_registered_native.nii.gz",
    }

    metrics = {}

    print("\n============================================================")
    print("Transforming B9 / MD / FA / S0 and calculating MI + NCC")
    print("============================================================")

    for name in ["BVALUE9", "MD", "FA", "S0"]:
        registered_path = os.path.join(
            REGISTERED_DIR,
            output_names[name],
        )

        registered = transform_image(
            EVAL_IMAGES[name],
            native_transform,
            registered_path,
            os.path.join(OUTPUT_DIR, "_transformix_tmp", name),
        )

        mi_global, ncc_global, n_global = calculate_mi_ncc(
            microscopy_native,
            registered,
        )

        mi_mask, ncc_mask, n_mask = calculate_mi_ncc(
            microscopy_native,
            registered,
            mask_native,
        )

        metrics[name] = {
            "MI_global": mi_global,
            "NCC_global": ncc_global,
            "valid_voxels_global": n_global,
            "MI_mask": mi_mask,
            "NCC_mask": ncc_mask,
            "valid_voxels_mask": n_mask,
            "registered_path": registered_path,
        }

        print(
            f"{name:8s} | "
            f"Global MI={mi_global:.6f}, NCC={ncc_global:.6f}, n={n_global} | "
            f"Mask MI={mi_mask:.6f}, NCC={ncc_mask:.6f}, n={n_mask}"
        )

    run_metrics_path = os.path.join(
        QC_DIR,
        "metrics_B9_MD_FA_S0.csv",
    )

    save_run_metrics(metrics, run_metrics_path)
    upsert_summary_table(metrics)

    print("\n============================================================")
    print("Result")
    print("============================================================")

    for name in ["BVALUE9", "MD", "FA", "S0"]:
        print(
            f"{name:8s} | "
            f"Global MI={metrics[name]['MI_global']:.6f}, "
            f"NCC={metrics[name]['NCC_global']:.6f} | "
            f"Mask MI={metrics[name]['MI_mask']:.6f}, "
            f"NCC={metrics[name]['NCC_mask']:.6f}"
        )

    print("\nRegistered images:")
    print(REGISTERED_DIR)

    print("\nEvaluation mask:")
    print(mask_native_path)

    print("\nThis-run metrics:")
    print(run_metrics_path)

    print("\nSummary table:")
    print(TABLE_CSV)


if __name__ == "__main__":
    main()