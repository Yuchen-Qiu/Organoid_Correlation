#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Directly segment 3D organoid outer boundary from microscopy NIfTI image.

No corrected image.
No clean affine.
No spacing conversion.
No affine rebuilding.

This version:
    1. Reads original NIfTI
    2. Converts 4D to 3D if needed
    3. Segments organoid mask
    4. Saves mask using original affine/header information
"""

import os
import numpy as np
import nibabel as nib

from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, closing, ball
from scipy.ndimage import gaussian_filter, binary_fill_holes, label


# ============================================================
# Inspection
# ============================================================

def inspect_nifti(input_path):
    print("\n========== NIfTI INSPECTION ==========")

    nii = nib.load(input_path)
    data = nii.get_fdata(dtype=np.float32)
    header = nii.header

    print(f"File: {input_path}")
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


# ============================================================
# Prepare 3D data
# ============================================================

def prepare_3d_data(data, channel_axis=None, channel=0):
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

    if data_3d.shape[2] == 1:
        print("\nWARNING:")
        print("Final 3D data has only one Z slice.")

    print("=====================================\n")

    return data_3d.astype(np.float32)


# ============================================================
# Segmentation helpers
# ============================================================

def normalize_01(img):
    img = img.astype(np.float32)

    img_min = np.nanmin(img)
    img_max = np.nanmax(img)

    print(f"Intensity before normalization: {img_min} ~ {img_max}")

    if img_max - img_min < 1e-8:
        raise ValueError("Image intensity range is almost zero. Cannot normalize.")

    return (img - img_min) / (img_max - img_min)


def keep_largest_component(mask):
    labeled, num = label(mask)

    print(f"Number of connected components: {num}")

    if num == 0:
        raise RuntimeError(
            "No foreground object found. Try lowering threshold or checking channel."
        )

    if num == 1:
        return mask

    counts = np.bincount(labeled.ravel())
    counts[0] = 0

    largest_label = np.argmax(counts)
    largest_size = counts[largest_label]

    print(f"Largest component label: {largest_label}")
    print(f"Largest component size: {largest_size}")

    return labeled == largest_label


def segment_organoid_array(
    img,
    gaussian_sigma=1.0,
    threshold=None,
    min_size=2000,
    closing_radius=3,
    fill_holes=True,
):
    print("\n========== SEGMENTATION ==========")
    print(f"Input image shape: {img.shape}")

    print("\nStep 1. Normalize")
    img = normalize_01(img)

    print("\nStep 2. Gaussian smoothing")
    print(f"Gaussian sigma: {gaussian_sigma}")

    if gaussian_sigma is not None and gaussian_sigma > 0:
        img_smooth = gaussian_filter(img, sigma=gaussian_sigma)
    else:
        img_smooth = img

    print("\nStep 3. Threshold")
    if threshold is None:
        thr = threshold_otsu(img_smooth)
        print(f"Otsu threshold: {thr}")
    else:
        thr = float(threshold)
        print(f"Manual threshold: {thr}")

    mask = img_smooth > thr
    print(f"Foreground after threshold: {int(mask.sum())}")

    print("\nStep 4. Remove small objects")
    print(f"min_size: {min_size}")
    mask = remove_small_objects(mask, min_size=min_size)
    print(f"Foreground after removing small objects: {int(mask.sum())}")

    print("\nStep 5. Closing")
    print(f"closing_radius: {closing_radius}")
    if closing_radius is not None and closing_radius > 0:
        mask = closing(mask, ball(closing_radius))
    print(f"Foreground after closing: {int(mask.sum())}")

    if fill_holes:
        print("\nStep 6. Fill holes")
        mask = binary_fill_holes(mask)
        print(f"Foreground after filling holes: {int(mask.sum())}")

    print("\nStep 7. Keep largest component")
    mask = keep_largest_component(mask)

    mask = mask.astype(np.uint8)

    print(f"Final mask shape: {mask.shape}")
    print(f"Final foreground voxels: {int(mask.sum())}")
    print("==================================\n")

    return mask


# ============================================================
# Save mask using original spatial information
# ============================================================

def save_mask_like_original(mask, reference_nii, output_path):
    """
    Save mask with original affine/header.
    This avoids changing spacing, origin, direction, and volume.
    """

    if mask.ndim != 3:
        raise ValueError(f"Only 3D mask can be saved. Current shape: {mask.shape}")

    mask = mask.astype(np.uint8)

    original_affine = reference_nii.affine
    original_header = reference_nii.header.copy()

    # Make header 3D-compatible and uint8
    original_header.set_data_dtype(np.uint8)
    original_header["dim"][0] = 3
    original_header["dim"][1] = mask.shape[0]
    original_header["dim"][2] = mask.shape[1]
    original_header["dim"][3] = mask.shape[2]
    original_header["dim"][4:] = 1

    out = nib.Nifti1Image(mask, original_affine, header=original_header)

    # Keep original qform/sform if available
    qform = reference_nii.get_qform()
    sform = reference_nii.get_sform()
    qform_code = int(reference_nii.header["qform_code"])
    sform_code = int(reference_nii.header["sform_code"])

    if qform_code > 0:
        out.set_qform(qform, code=qform_code)

    if sform_code > 0:
        out.set_sform(sform, code=sform_code)

    nib.save(out, output_path)

    print(f"Saved mask: {output_path}")


def verify_output(path):
    print("\n========== VERIFY OUTPUT ==========")

    nii = nib.load(path)

    print(f"File: {path}")
    print(f"Shape: {nii.shape}")
    print(f"Zooms: {nii.header.get_zooms()}")
    print(f"Units: {nii.header.get_xyzt_units()}")

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
# Pipeline
# ============================================================

def run_pipeline(
    input_path,
    output_mask_path,
    channel_axis=None,
    channel=0,
    gaussian_sigma=1.0,
    threshold=None,
    min_size=2000,
    closing_radius=3,
    fill_holes=True,
):
    reference_nii, raw_data = inspect_nifti(input_path)

    img_3d = prepare_3d_data(
        data=raw_data,
        channel_axis=channel_axis,
        channel=channel,
    )

    mask = segment_organoid_array(
        img=img_3d,
        gaussian_sigma=gaussian_sigma,
        threshold=threshold,
        min_size=min_size,
        closing_radius=closing_radius,
        fill_holes=fill_holes,
    )

    save_mask_like_original(
        mask=mask,
        reference_nii=reference_nii,
        output_path=output_mask_path,
    )

    verify_output(output_mask_path)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    input_path = "/autofs/arch11/DATA/HOMES/yuchen/Minor/Organoids/case2_high_Zo-1_Map2_max.nii.gz"

    output_mask_path = "/autofs/arch11/DATA/HOMES/yuchen/Minor/Organoids/case2_high_Zo-1_Map2_max_mask.nii.gz"

    # Your current shape is probably:
    #   (512, 512, 239, 1)
    #
    # This means X, Y, Z, C.
    # Extract channel 0 from axis 3.
    channel_axis = 3
    channel = 0

    gaussian_sigma = 1.0

    # None means Otsu.
    # You can manually set e.g. threshold = 0.1 after normalization.
    threshold = None

    min_size = 2000
    closing_radius = 3
    fill_holes = True

    run_pipeline(
        input_path=input_path,
        output_mask_path=output_mask_path,
        channel_axis=channel_axis,
        channel=channel,
        gaussian_sigma=gaussian_sigma,
        threshold=threshold,
        min_size=min_size,
        closing_radius=closing_radius,
        fill_holes=fill_holes,
    )