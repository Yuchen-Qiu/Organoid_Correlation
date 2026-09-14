import os
import numpy as np
import nibabel as nib


# ============================================================
# Paths
# ============================================================

seg_path = "/autofs/arch11/DATA/HOMES/yuchen/phantom/phantom_25um/watershed_instance_mask.nii.gz"
center_path = "/autofs/arch11/DATA/HOMES/yuchen/phantom/phantom_25um/centroid_points.nii.gz"

output_dir = "/autofs/arch11/DATA/HOMES/yuchen/phantom/phantom_25um/grid_output"
os.makedirs(output_dir, exist_ok=True)

grid_line_path = os.path.join(output_dir, "diffusion_grid_lines_on_GRE.nii.gz")
overlay_path = os.path.join(output_dir, "segmentation_centers_grid_overlay.nii.gz")
instance_grid_path = os.path.join(output_dir, "watershed_instance_mask_with_diffusion_grid.nii.gz")

# ============================================================
# Parameters
# ============================================================

grid_size_um = 50.0


# ============================================================
# Load segmentation
# ============================================================

seg_img = nib.load(seg_path)
seg_data = seg_img.get_fdata()
seg_affine = seg_img.affine
seg_header = seg_img.header.copy()

print("Segmentation shape:", seg_data.shape)

zooms_mm = seg_img.header.get_zooms()[:3]
voxel_size_um = tuple(float(z) * 1000.0 for z in zooms_mm)

print("GRE voxel size (µm):", voxel_size_um)

sx_um = voxel_size_um[0]
sy_um = voxel_size_um[1]

factor_x = grid_size_um / sx_um
factor_y = grid_size_um / sy_um

print("Grid size (µm):", grid_size_um)
print("Grid factor in pixels:", factor_x, factor_y)


# ============================================================
# Load center / marker image
# ============================================================

center_img = nib.load(center_path)
center_data = center_img.get_fdata()

print("Center image shape:", center_data.shape)

if center_data.shape != seg_data.shape:
    raise ValueError(
        f"Segmentation and center image shapes do not match: "
        f"{seg_data.shape} vs {center_data.shape}"
    )


# ============================================================
# Prepare 2D images
# ============================================================

if seg_data.ndim == 3:
    seg_2d = seg_data[:, :, 0]
else:
    seg_2d = seg_data

if center_data.ndim == 3:
    center_2d = center_data[:, :, 0]
else:
    center_2d = center_data

nx, ny = seg_2d.shape

seg_mask = seg_2d > 0
center_mask = center_2d > 0


# ============================================================
# Draw 50 x 50 µm grid lines
# ============================================================

grid_lines_2d = np.zeros((nx, ny), dtype=np.uint8)

x = 0.0
x_positions = []
while x < nx:
    xi = int(round(x))
    if 0 <= xi < nx:
        grid_lines_2d[xi, :] = 1
        x_positions.append(xi)
    x += factor_x

y = 0.0
y_positions = []
while y < ny:
    yi = int(round(y))
    if 0 <= yi < ny:
        grid_lines_2d[:, yi] = 1
        y_positions.append(yi)
    y += factor_y

grid_mask = grid_lines_2d > 0

print("Number of vertical grid lines:", len(x_positions))
print("Number of horizontal grid lines:", len(y_positions))


# ============================================================
# Save grid line image
# ============================================================

grid_lines_3d = grid_lines_2d[:, :, None]

grid_img = nib.Nifti1Image(grid_lines_3d, seg_affine, seg_header)
grid_img.set_data_dtype(np.uint8)
nib.save(grid_img, grid_line_path)

print("Saved grid lines:", grid_line_path)

# ============================================================
# Save watershed instance mask with grid
#
# Original watershed instance labels are preserved.
# Grid lines are assigned a new label:
# grid_label = max original label + 1
# ============================================================

instance_2d = seg_2d.copy()

max_instance_label = int(np.max(instance_2d))
grid_label = max_instance_label + 1

print("Max original watershed label:", max_instance_label)
print("Grid line label:", grid_label)

instance_grid_2d = instance_2d.copy()
instance_grid_2d[grid_mask] = grid_label

instance_grid_3d = instance_grid_2d[:, :, None]

# Use integer dtype that can safely store all instance labels
instance_grid_img = nib.Nifti1Image(
    instance_grid_3d.astype(np.uint16),
    seg_affine,
    seg_header
)
instance_grid_img.set_data_dtype(np.uint16)

nib.save(instance_grid_img, instance_grid_path)

print("Saved watershed instance mask with grid:", instance_grid_path)

# ============================================================
# Create overlay image
#
# 0 = background
# 1 = segmentation
# 2 = grid line
# 3 = center point
# 4 = center point exactly on grid line
#
# Priority:
# segmentation first
# grid overwrites segmentation
# center overwrites both
# center on grid gets special label 4
# ============================================================

overlay_2d = np.zeros((nx, ny), dtype=np.uint8)

overlay_2d[seg_mask] = 1
overlay_2d[grid_mask] = 2
overlay_2d[center_mask] = 3
overlay_2d[center_mask & grid_mask] = 4

overlay_3d = overlay_2d[:, :, None]

overlay_img = nib.Nifti1Image(overlay_3d, seg_affine, seg_header)
overlay_img.set_data_dtype(np.uint8)
nib.save(overlay_img, overlay_path)

print("Saved overlay:", overlay_path)


# ============================================================
# Statistics
# ============================================================

print("\n--- Statistics ---")
print("Segmentation pixels:", int(np.sum(seg_mask)))
print("Center pixels:", int(np.sum(center_mask)))
print("Grid pixels:", int(np.sum(grid_mask)))
print("Center pixels exactly on grid line:", int(np.sum(center_mask & grid_mask)))

print("\nLabel meaning in overlay:")
print("0 = background")
print("1 = segmentation")
print("2 = grid line")
print("3 = center point")
print("4 = center point on grid line")

print("Done.")