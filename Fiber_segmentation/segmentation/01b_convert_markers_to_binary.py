import os
import numpy as np
import nibabel as nib


# ============================================================
# Paths
# ============================================================
work_dir = "/autofs/arch11/DATA/HOMES/yuchen/phantom/output_manual_seed"

input_marker_path = os.path.join(work_dir, "markers_auto.nii.gz")
output_binary_path = os.path.join(work_dir, "markers_manual_binary.nii.gz")


# ============================================================
# Load markers
# ============================================================
nii = nib.load(input_marker_path)
markers = nii.get_fdata()

print("Input marker shape:", markers.shape)
print("Input marker max label:", np.max(markers))
print("Input non-zero voxels:", np.sum(markers > 0))


# ============================================================
# Convert to binary
# ============================================================
binary = (markers > 0).astype(np.uint8)

print("Binary non-zero voxels:", np.sum(binary > 0))


# ============================================================
# Save binary seed mask
# ============================================================
header = nii.header.copy()
out_nii = nib.Nifti1Image(binary, nii.affine, header)
out_nii.set_data_dtype(np.uint8)

nib.save(out_nii, output_binary_path)

print("Saved binary seed mask:")
print(output_binary_path)
print("Now manually edit this file, then run 01c_relabel_binary_markers.py")