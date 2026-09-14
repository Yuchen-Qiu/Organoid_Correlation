#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion


# ============================================================
# User settings
# ============================================================

# Single image mode:
IMAGE_PATH = r"/autofs/arch11/DATA/HOMES/yuchen/Minor/Organoids/Nifti_data/case2_young/data_final/dti_MD_final_LAS.nii.gz"
# IMAGE_PATH = None
# Batch mode:
# Set IMAGE_PATH = None and specify INPUT_DIR
# INPUT_DIR = r"/autofs/arch11/DATA/HOMES/yuchen/Minor/Organoids/Nifti_data/case2_young/data_final/data_final_bvalue"
INPUT_DIR = None

MASK_PATH = r"/autofs/arch11/DATA/HOMES/yuchen/Minor/Organoids/Nifti_data/case2_young/data_final/case2_FA_mask_LAS.nii.gz"

OUTPUT_DIR = r"/autofs/arch11/DATA/HOMES/yuchen/Minor/Organoids/Nifti_data/case2_young/case2_MD_mask_LAS_eroded_2vox.nii.gz"

EROSION_VOXELS = 2


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if IMAGE_PATH is not None and INPUT_DIR is not None:
        raise ValueError("Use either IMAGE_PATH or INPUT_DIR, not both.")

    if IMAGE_PATH is not None:
        if not os.path.isfile(IMAGE_PATH):
            raise FileNotFoundError(IMAGE_PATH)
        image_paths = [IMAGE_PATH]
        print("[INFO] Mode: single image")

    elif INPUT_DIR is not None:
        if not os.path.isdir(INPUT_DIR):
            raise FileNotFoundError(INPUT_DIR)

        image_paths = [
            os.path.join(INPUT_DIR, f)
            for f in sorted(os.listdir(INPUT_DIR))
            if f.endswith(".nii") or f.endswith(".nii.gz")
        ]

        if len(image_paths) == 0:
            raise FileNotFoundError(f"No NIfTI files found in: {INPUT_DIR}")

        print("[INFO] Mode: batch")
        print(f"[INFO] Found {len(image_paths)} images")

    else:
        raise ValueError("Please specify IMAGE_PATH or INPUT_DIR.")

    print("[INFO] Loading mask...")
    mask_img = nib.load(MASK_PATH)
    mask = mask_img.get_fdata() > 0

    print(f"[INFO] Mask shape: {mask.shape}")
    print(f"[INFO] Mask spacing: {mask_img.header.get_zooms()[:3]}")
    print(f"[INFO] Original mask voxels: {np.sum(mask)}")

    structure = np.ones((3, 3, 3), dtype=bool)

    eroded_mask = binary_erosion(
        mask,
        structure=structure,
        iterations=EROSION_VOXELS,
        border_value=0
    )

    removed_outline = mask & (~eroded_mask)

    original_voxels = np.sum(mask)
    eroded_voxels = np.sum(eroded_mask)
    removed_voxels = np.sum(removed_outline)

    print(f"[INFO] Eroded mask voxels: {eroded_voxels}")
    print(f"[INFO] Removed boundary voxels: {removed_voxels}")
    print(f"[INFO] Remaining ratio: {eroded_voxels / max(original_voxels, 1) * 100:.2f}%")

    if eroded_voxels == 0:
        raise ValueError("Eroded mask is empty.")

    mask_name = os.path.basename(MASK_PATH)

    if mask_name.endswith(".nii.gz"):
        mask_name = mask_name[:-7]
    elif mask_name.endswith(".nii"):
        mask_name = mask_name[:-4]

    eroded_mask_path = os.path.join(
        OUTPUT_DIR,
        f"{mask_name}_eroded_{EROSION_VOXELS}vox.nii.gz"
    )

    outline_path = os.path.join(
        OUTPUT_DIR,
        f"{mask_name}_removed_outline_{EROSION_VOXELS}vox.nii.gz"
    )

    mask_header = mask_img.header.copy()
    mask_header.set_data_dtype(np.uint8)

    eroded_img = nib.Nifti1Image(
        eroded_mask.astype(np.uint8),
        mask_img.affine,
        mask_header.copy()
    )

    outline_img = nib.Nifti1Image(
        removed_outline.astype(np.uint8),
        mask_img.affine,
        mask_header.copy()
    )

    qform, qcode = mask_img.get_qform(coded=True)
    sform, scode = mask_img.get_sform(coded=True)

    if qform is not None:
        eroded_img.set_qform(qform, int(qcode))
        outline_img.set_qform(qform, int(qcode))

    if sform is not None:
        eroded_img.set_sform(sform, int(scode))
        outline_img.set_sform(sform, int(scode))

    nib.save(eroded_img, eroded_mask_path)
    nib.save(outline_img, outline_path)

    print(f"[SAVE] {eroded_mask_path}")
    print(f"[SAVE] {outline_path}")

    successful = 0
    failed = 0

    for i, image_path in enumerate(image_paths, start=1):
        print("\n========================================")
        print(f"[{i}/{len(image_paths)}] {os.path.basename(image_path)}")
        print("========================================")

        try:
            image_img = nib.load(image_path)
            image = image_img.get_fdata(dtype=np.float32)

            print(f"[INFO] Shape: {image.shape}")
            print(f"[INFO] Spacing: {image_img.header.get_zooms()[:3]}")
            print(f"[INFO] Image dtype: {image_img.header.get_data_dtype()}")
            print(f"[INFO] Image range: {np.nanmin(image):.8g} - {np.nanmax(image):.8g}")

            if image.shape != mask.shape:
                raise ValueError(
                    f"Image/mask shape mismatch:\n"
                    f"Image: {image.shape}\n"
                    f"Mask:  {mask.shape}"
                )

            if not np.allclose(image_img.affine, mask_img.affine, atol=1e-5):
                raise ValueError(
                    "Image and mask affine matrices do not match.\n"
                    f"Image affine:\n{image_img.affine}\n"
                    f"Mask affine:\n{mask_img.affine}"
                )

            masked_image = image.copy()
            masked_image[~eroded_mask] = 0

            image_name = os.path.basename(image_path)

            if image_name.endswith(".nii.gz"):
                image_name = image_name[:-7]
            elif image_name.endswith(".nii"):
                image_name = image_name[:-4]

            output_path = os.path.join(
                OUTPUT_DIR,
                f"{image_name}_eroded_{EROSION_VOXELS}vox.nii.gz"
            )

            image_header = image_img.header.copy()
            image_header.set_data_dtype(np.float32)
            image_header.set_slope_inter(1.0, 0.0)

            masked_img = nib.Nifti1Image(
                masked_image.astype(np.float32),
                image_img.affine,
                image_header
            )

            qform, qcode = image_img.get_qform(coded=True)
            sform, scode = image_img.get_sform(coded=True)

            if qform is not None:
                masked_img.set_qform(qform, int(qcode))

            if sform is not None:
                masked_img.set_sform(sform, int(scode))

            nib.save(masked_img, output_path)

            print(f"[SAVE] {output_path}")

            saved_img = nib.load(output_path)
            saved = saved_img.get_fdata(dtype=np.float32)

            diff = np.abs(saved[eroded_mask] - image[eroded_mask])

            print(f"[CHECK] Original inside range: {np.nanmin(image[eroded_mask]):.8g} - {np.nanmax(image[eroded_mask]):.8g}")
            print(f"[CHECK] Saved inside range: {np.nanmin(saved[eroded_mask]):.8g} - {np.nanmax(saved[eroded_mask]):.8g}")
            print(f"[CHECK] Max intensity difference: {np.nanmax(diff):.8g}")
            print(f"[CHECK] Mean intensity difference: {np.nanmean(diff):.8g}")
            print(f"[CHECK] Outside nonzero voxels: {np.count_nonzero(saved[~eroded_mask])}")

            if np.allclose(
                saved[eroded_mask],
                image[eroded_mask],
                rtol=0,
                atol=1e-7
            ):
                print("[CHECK] PASS: Internal intensities are unchanged.")
            else:
                print("[CHECK] WARNING: Internal intensities changed after saving.")

            successful += 1

        except Exception as e:
            print(f"[ERROR] {e}")
            failed += 1

    print("\n========================================")
    print("Finished")
    print("========================================")
    print(f"[INFO] Total: {len(image_paths)}")
    print(f"[INFO] Successful: {successful}")
    print(f"[INFO] Failed: {failed}")


if __name__ == "__main__":
    main()