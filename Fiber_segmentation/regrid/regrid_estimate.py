import os
import nibabel as nib
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

s_map_path = "/autofs/arch11/DATA/HOMES/yuchen/phantom/phantom_25um/forSV_newNIFTI_adjusted.nii"

output_dir = "/autofs/arch11/DATA/HOMES/yuchen/phantom/phantom_25um/sv_output"
os.makedirs(output_dir, exist_ok=True)

sv_map_out = os.path.join(output_dir, "sv_map_um-1.nii.gz")
csv_out = os.path.join(output_dir, "sv_values.csv")


# ============================================================
# Load S map
# ============================================================

nii = nib.load(s_map_path)
s_data = nii.get_fdata()

header = nii.header.copy()
affine = nii.affine

# voxel spacing from header
zooms = header.get_zooms()

voxel_size_x = float(zooms[0])
voxel_size_y = float(zooms[1])

print("Image shape:", s_data.shape)
print("Header voxel size:", zooms)


# ============================================================
# Unit check
# ============================================================
# NIfTI spacing is often in mm.
# If your diffusion voxel is 50 µm, header may be 0.05 mm.
# Therefore convert mm -> µm if spacing looks like mm.

if voxel_size_x < 1 and voxel_size_y < 1:
    print("Voxel size seems to be in mm, converting to µm.")
    voxel_size_x_um = voxel_size_x * 1000
    voxel_size_y_um = voxel_size_y * 1000
else:
    print("Voxel size seems to already be in µm.")
    voxel_size_x_um = voxel_size_x
    voxel_size_y_um = voxel_size_y

print("Voxel size x in µm:", voxel_size_x_um)
print("Voxel size y in µm:", voxel_size_y_um)


# ============================================================
# Calculate S/V
# ============================================================
# Chantal's formula:
# V = voxel_size_x * voxel_size_y
# because z cancels out.

V_um2 = voxel_size_x_um * voxel_size_y_um

print("V used for S/V calculation:", V_um2, "µm²")

sv_data = np.zeros_like(s_data, dtype=np.float32)

valid_mask = s_data > 0
sv_data[valid_mask] = s_data[valid_mask] / V_um2

print("Number of non-zero S voxels:", np.sum(valid_mask))
print("S/V min:", np.min(sv_data[valid_mask]) if np.any(valid_mask) else 0)
print("S/V max:", np.max(sv_data[valid_mask]) if np.any(valid_mask) else 0)


# ============================================================
# Save S/V map
# ============================================================

sv_nii = nib.Nifti1Image(sv_data.astype(np.float32), affine, header)
sv_nii.set_data_dtype(np.float32)
nib.save(sv_nii, sv_map_out)

print("Saved S/V map to:")
print(sv_map_out)


# ============================================================
# Save CSV only for voxels with S > 0
# ============================================================

indices = np.argwhere(valid_mask)

rows = []

for idx in indices:
    x, y = int(idx[0]), int(idx[1])

    if s_data.ndim == 2:
        z = 0
        s_value = s_data[x, y]
        sv_value = sv_data[x, y]
    else:
        z = int(idx[2])
        s_value = s_data[x, y, z]
        sv_value = sv_data[x, y, z]

    rows.append({
        "voxel_x": x,
        "voxel_y": y,
        "voxel_z": z,
        "S_um": round(float(s_value), 4),
        "V_um2": round(float(V_um2), 4),
        "S_over_V_um-1": round(float(sv_value), 6),
    })

df = pd.DataFrame(rows)
df.to_csv(csv_out, index=False)

print("Saved CSV to:")
print(csv_out)