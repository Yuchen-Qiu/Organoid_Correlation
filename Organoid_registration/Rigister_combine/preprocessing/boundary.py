#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create a boundary contour NIfTI from an existing binary mask NIfTI.

Recommended use after registration/resampling:
    1. Transform/resample the full mask using nearest-neighbor interpolation.
    2. Run this script on the transformed mask.
    3. Save the boundary in the same grid/affine/header as the transformed mask.

Boundary method:
    inner_boundary = mask - erosion(mask)
"""

import os
import numpy as np
import nibabel as nib

from skimage.morphology import binary_erosion, ball, closing
from scipy.ndimage import binary_fill_holes


# ============================================================
# I/O helpers
# ============================================================

def inspect_nifti(path):
    print("\n========== NIfTI INSPECTION ==========")

    nii = nib.load(path)
    data = nii.get_fdata(dtype=np.float32)
    header = nii.header

    print(f"File: {path}")
    print(f"Data shape: {data.shape}")
    print(f"Data ndim: {data.ndim}")

    print("\nHeader dim:")
    print(header["dim"])

    print("\nHeader pixdim:")
    print(header["pixdim"])

    print("\nHeader zooms:")
    print(header.get_zooms())

    print("\nHeader xyzt_units:")
    print(header.get_xyzt_units())

    print("\nAffine:")
    print(nii.affine)

    print("\nqform code:", int(header["qform_code"]))
    print("qform:")
    print(nii.get_qform())

    print("\nsform code:", int(header["sform_code"]))
    print("sform:")
    print(nii.get_sform())

    print("\nData intensity:")
    print(f"  min  = {np.nanmin(data)}")
    print(f"  max  = {np.nanmax(data)}")
    print(f"  mean = {np.nanmean(data)}")

    print("======================================\n")

    return nii, data


def prepare_3d_data(data, channel_axis=None, channel=0):
    """
    Convert loaded mask data to 3D.
    If input is 4D with a singleton/channel axis, extract or squeeze it.
    """
    print("\n========== PREPARE 3D DATA ==========")
    print(f"Raw data shape: {data.shape}")

    if data.ndim == 3:
        data_3d = data
        print("Input is already 3D.")

    elif data.ndim == 4:
        if channel_axis is not None:
            print(f"Extracting channel {channel} from axis {channel_axis}")

            if channel_axis < 0:
                channel_axis = data.ndim + channel_axis

            if channel_axis < 0 or channel_axis >= data.ndim:
                raise ValueError(f"Invalid channel_axis: {channel_axis}")

            if channel >= data.shape[channel_axis]:
                raise ValueError(
                    f"channel={channel} is out of range for axis {channel_axis}, "
                    f"which has size {data.shape[channel_axis]}"
                )

            data = np.take(data, indices=channel, axis=channel_axis)
            print(f"Shape after channel extraction: {data.shape}")
        else:
            print("No channel extraction. Only squeezing singleton dimensions.")

        old_shape = data.shape
        data = np.squeeze(data)
        print(f"Shape after squeeze: {old_shape} -> {data.shape}")
        data_3d = data

    else:
        raise ValueError(f"Input must be 3D or 4D. Current shape: {data.shape}")

    if data_3d.ndim != 3:
        raise ValueError(
            f"After preparation, data is still not 3D. Current shape: {data_3d.shape}."
        )

    print(f"Final 3D data shape: {data_3d.shape}")
    print("=====================================\n")

    return data_3d.astype(np.float32)


# ============================================================
# Boundary extraction
# ============================================================

def binarize_mask(mask_data, threshold=0.5):
    """
    Convert mask image to boolean mask.

    For nearest-neighbor-transformed masks, values should usually be 0/1.
    threshold=0.5 is safe if values are exactly 0 and 1, or slightly non-binary.
    """
    print("\n========== BINARIZE MASK ==========")
    print(f"Threshold: > {threshold}")

    mask = np.nan_to_num(mask_data, nan=0.0) > threshold

    print(f"Foreground voxels: {int(mask.sum())}")
    if int(mask.sum()) == 0:
        raise RuntimeError("Mask is empty after thresholding. Check input mask or threshold.")

    print("==================================\n")
    return mask


def clean_mask_before_boundary(mask, do_closing=False, closing_radius=1, do_fill_holes=False):
    """
    Optional cleanup for transformed masks.

    Use this only when the transformed mask has tiny gaps/holes.
    For strict quantitative work, keep both options False unless needed.
    """
    print("\n========== OPTIONAL MASK CLEANUP ==========")
    print(f"do_closing     : {do_closing}")
    print(f"closing_radius : {closing_radius}")
    print(f"do_fill_holes  : {do_fill_holes}")

    out = mask.astype(bool)

    if do_closing:
        out = closing(out, ball(closing_radius))
        print(f"Foreground after closing: {int(out.sum())}")

    if do_fill_holes:
        out = binary_fill_holes(out)
        print(f"Foreground after fill holes: {int(out.sum())}")

    print("==========================================\n")
    return out


def get_boundary_from_mask(mask, boundary_radius=1):
    """
    Extract an inner boundary contour from a binary 3D mask.

    Method:
        boundary = mask - erosion(mask)

    boundary_radius controls boundary thickness.
    boundary_radius=1 usually gives approximately 1-voxel-thick boundary.
    """
    print("\n========== BOUNDARY EXTRACTION ==========")
    print("Method: boundary = mask - erosion(mask)")
    print(f"boundary_radius: {boundary_radius}")

    mask_bool = mask.astype(bool)

    if boundary_radius is None or boundary_radius <= 0:
        print("boundary_radius <= 0, returning original mask as boundary.")
        boundary = mask_bool
    else:
        eroded = binary_erosion(mask_bool, ball(boundary_radius))
        boundary = mask_bool & (~eroded)

    boundary = boundary.astype(np.uint8)

    print(f"Boundary voxels: {int(boundary.sum())}")
    print("=========================================\n")

    return boundary


# ============================================================
# Save using input mask spatial information
# ============================================================

def save_mask_like_reference(mask, reference_nii, output_path):
    """
    Save binary image with the same affine/header spatial info as the input mask.
    """
    if mask.ndim != 3:
        raise ValueError(f"Only 3D mask can be saved. Current shape: {mask.shape}")

    mask = mask.astype(np.uint8)

    header = reference_nii.header.copy()
    header.set_data_dtype(np.uint8)
    header["dim"][0] = 3
    header["dim"][1] = mask.shape[0]
    header["dim"][2] = mask.shape[1]
    header["dim"][3] = mask.shape[2]
    header["dim"][4:] = 1

    out = nib.Nifti1Image(mask, reference_nii.affine, header=header)

    qform_code = int(reference_nii.header["qform_code"])
    sform_code = int(reference_nii.header["sform_code"])

    if qform_code > 0:
        out.set_qform(reference_nii.get_qform(), code=qform_code)

    if sform_code > 0:
        out.set_sform(reference_nii.get_sform(), code=sform_code)

    nib.save(out, output_path)
    print(f"Saved: {output_path}")


def verify_output(path):
    print("\n========== VERIFY OUTPUT ==========")

    nii = nib.load(path)

    print(f"File: {path}")
    print(f"Shape: {nii.shape}")
    print(f"Zooms: {nii.header.get_zooms()}")
    print(f"Units: {nii.header.get_xyzt_units()}")

    data = nii.get_fdata(dtype=np.float32)
    print(f"Unique values approximately: {np.unique(data)[:10]}")
    print(f"Foreground voxels: {int((data > 0).sum())}")

    print("\nAffine:")
    print(nii.affine)

    print("\nqform code:", int(nii.header["qform_code"]))
    print("qform:")
    print(nii.get_qform())

    print("\nsform code:", int(nii.header["sform_code"]))
    print("sform:")
    print(nii.get_sform())

    print("===================================\n")


# ============================================================
# Pipeline: existing mask -> boundary
# ============================================================

def run_mask_to_boundary(
    input_mask_path,
    output_boundary_path=None,
    channel_axis=None,
    channel=0,
    mask_threshold=0.5,
    boundary_radius=1,
    do_closing=False,
    closing_radius=1,
    do_fill_holes=False,
):
    reference_nii, raw_data = inspect_nifti(input_mask_path)

    mask_data_3d = prepare_3d_data(
        data=raw_data,
        channel_axis=channel_axis,
        channel=channel,
    )

    mask = binarize_mask(mask_data_3d, threshold=mask_threshold)

    mask = clean_mask_before_boundary(
        mask=mask,
        do_closing=do_closing,
        closing_radius=closing_radius,
        do_fill_holes=do_fill_holes,
    )

    boundary = get_boundary_from_mask(
        mask=mask,
        boundary_radius=boundary_radius,
    )

    if output_boundary_path is None:
        if input_mask_path.endswith(".nii.gz"):
            output_boundary_path = input_mask_path.replace(".nii.gz", "_boundary.nii.gz")
        elif input_mask_path.endswith(".nii"):
            output_boundary_path = input_mask_path.replace(".nii", "_boundary.nii")
        else:
            output_boundary_path = input_mask_path + "_boundary.nii.gz"

    os.makedirs(os.path.dirname(output_boundary_path), exist_ok=True)

    save_mask_like_reference(
        mask=boundary,
        reference_nii=reference_nii,
        output_path=output_boundary_path,
    )

    verify_output(output_boundary_path)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # Use your existing/transformed mask here.
    # Recommended: transform/resample the full mask first using nearest neighbor,
    # then run this script to extract boundary in the target grid.
    input_mask_path = "/autofs/arch11/DATA/HOMES/yuchen/Minor/Organoids/Nifti_data/results_mask/case2_FA_single_rigid_affine_bspline_FA_mask_post.nii.gz"

    output_boundary_path = "/autofs/arch11/DATA/HOMES/yuchen/Minor/Organoids/Nifti_data/results_mask/boundary/case2_FA_single_rigid_affine_bspline_FA_boundary_post.nii.gz"

    # If your mask is 4D, e.g. (X, Y, Z, 1), use channel_axis=3.
    # If your mask is already 3D, channel_axis can stay None.
    channel_axis = None
    channel = 0

    # For binary masks, 0.5 is standard.
    mask_threshold = 0.5

    # 1 = thin boundary; 2 or 3 = thicker boundary.
    boundary_radius = 1

    # Optional cleanup for transformed masks with small holes/gaps.
    # Keep False first. Turn on only if transformed mask is visibly broken.
    do_closing = False
    closing_radius = 1
    do_fill_holes = False

    run_mask_to_boundary(
        input_mask_path=input_mask_path,
        output_boundary_path=output_boundary_path,
        channel_axis=channel_axis,
        channel=channel,
        mask_threshold=mask_threshold,
        boundary_radius=boundary_radius,
        do_closing=do_closing,
        closing_radius=closing_radius,
        do_fill_holes=do_fill_holes,
    )
