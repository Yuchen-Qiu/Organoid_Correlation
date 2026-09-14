#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import nibabel as nib
import numpy as np


# ============================================================
# User settings
# ============================================================

FA_PATH = r"/autofs/arch11/DATA/HOMES/yuchen/Minor/Organoids/Nifti_data/case2_young/data_final/dti_FA_final.nii.gz"
MD_PATH = r"/autofs/arch11/DATA/HOMES/yuchen/Minor/Organoids/Nifti_data/case2_young/data_final/dti_MD_final.nii.gz"
S0_PATH = r"/autofs/arch11/DATA/HOMES/yuchen/Minor/Organoids/Nifti_data/case2_young/data_final/dti_S0_final.nii.gz"
MASK_PATH = r"/autofs/arch11/DATA/HOMES/yuchen/Minor/Organoids/Nifti_data/case2_young/data_final/case2_FA_mask.nii.gz"

OUTPUT_DIR = r"/autofs/arch11/DATA/HOMES/yuchen/Minor/Organoids/Nifti_data/case2_young/data_final/LAS"

TARGET_AXCODES = ("L", "A", "S")


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = {
        "FA": FA_PATH,
        "MD": MD_PATH,
        "S0": S0_PATH,
        "MASK": MASK_PATH,
    }

    target_ornt = nib.orientations.axcodes2ornt(TARGET_AXCODES)

    for name, path in files.items():
        img = nib.load(path)
        data = np.asanyarray(img.dataobj)
        affine = img.affine

        current_axcodes = nib.aff2axcodes(affine)
        current_ornt = nib.orientations.io_orientation(affine)
        transform = nib.orientations.ornt_transform(current_ornt, target_ornt)

        print("\n" + "=" * 50)
        print(name)
        print("=" * 50)
        print(f"Input: {path}")
        print(f"Shape: {img.shape}")
        print(f"Orientation: {current_axcodes} -> {TARGET_AXCODES}")

        if current_axcodes == TARGET_AXCODES:
            new_data = data
            new_affine = affine.copy()
            print("[INFO] Already LAS, no reorientation needed.")
        else:
            new_data = nib.orientations.apply_orientation(data, transform)
            new_affine = affine @ nib.orientations.inv_ornt_aff(
                transform, img.shape[:3]
            )

        header = img.header.copy()
        filename = os.path.basename(path)

        if filename.endswith(".nii.gz"):
            filename = filename[:-7] + "_LAS.nii.gz"
        elif filename.endswith(".nii"):
            filename = filename[:-4] + "_LAS.nii"
        else:
            filename = filename + "_LAS"

        output_path = os.path.join(OUTPUT_DIR, filename)

        out = nib.Nifti1Image(new_data, new_affine, header=header)
        out.set_qform(new_affine, code=1)
        out.set_sform(new_affine, code=1)
        nib.save(out, output_path)

        check = nib.load(output_path)

        print(f"Output shape: {check.shape}")
        print(f"Output spacing: {check.header.get_zooms()[:3]}")
        print(f"Output orientation: {nib.aff2axcodes(check.affine)}")
        print("Output affine:")
        print(check.affine)
        print(f"Saved: {output_path}")

        if np.issubdtype(data.dtype, np.integer):
            print(f"Input unique values: {np.unique(data)[:20]}")
            print(f"Output unique values: {np.unique(new_data)[:20]}")
        else:
            print(f"Input range: {np.nanmin(data):.8g} ~ {np.nanmax(data):.8g}")
            print(f"Output range: {np.nanmin(new_data):.8g} ~ {np.nanmax(new_data):.8g}")


if __name__ == "__main__":
    main()