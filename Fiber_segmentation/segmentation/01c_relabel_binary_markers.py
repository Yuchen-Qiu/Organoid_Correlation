import os
import numpy as np
import nibabel as nib
from skimage.measure import label


# ============================================================
# Paths
# ============================================================
work_dir = "/autofs/arch11/DATA/HOMES/yuchen/phantom/output_manual_seed"

input_binary_path = os.path.join(work_dir, "markers_manual_binary.nii.gz")
output_labeled_path = os.path.join(work_dir, "markers_manual_labeled.nii.gz")


# ============================================================
# Load binary markers
# ============================================================
nii = nib.load(input_binary_path)
binary = nii.get_fdata()

print("Input binary marker shape:", binary.shape)
print("Input non-zero voxels:", np.sum(binary > 0))


# ============================================================
# Convert to 2D if needed
# ============================================================
if binary.ndim == 3 and binary.shape[2] == 1:
    binary_2d = binary[:, :, 0]
    is_2d_slice = True
elif binary.ndim == 2:
    binary_2d = binary
    is_2d_slice = False
else:
    raise ValueError(f"Expected 2D image or 3D image with one slice, got shape {binary.shape}")


# ============================================================
# Relabel connected seed points
# ============================================================
# connectivity=1 means 4-connected in 2D
# 如果你希望斜对角也算连接，可以改成 connectivity=2
labeled_2d = label(binary_2d > 0, connectivity=1).astype(np.uint16)

print("Number of labeled seed components:", labeled_2d.max())


# ============================================================
# Convert back to original shape
# ============================================================
if is_2d_slice:
    labeled = labeled_2d[:, :, None]
else:
    labeled = labeled_2d


# ============================================================
# Save labeled markers
# ============================================================
header = nii.header.copy()
out_nii = nib.Nifti1Image(labeled, nii.affine, header)
out_nii.set_data_dtype(np.uint16)

nib.save(out_nii, output_labeled_path)

print("Saved labeled seed mask:")
print(output_labeled_path)
print("Use this file for region growing.")