import os
import numpy as np
import pandas as pd

from scipy import ndimage as ndi
from skimage.measure import regionprops, regionprops_table

from seg_utils import (
    save_nii,
    load_nii_as_2d,
    read_and_normalize_image,
    prepare_manual_markers,
    region_growing_from_markers,
)


# ============================================================
# Parameters
# ============================================================
input_path = "/autofs/arch11/DATA/HOMES/yuchen/phantom/5mmIprobe_3mm_phantom.nii"
work_dir = "/autofs/arch11/DATA/HOMES/yuchen/phantom/output_manual_seed"

manual_marker_path = os.path.join(work_dir, "markers_manual_labeled.nii.gz")
dot_mask_path = os.path.join(work_dir, "dot_mask_binary.nii.gz")

output_dir = os.path.join(work_dir, "region_growing_result_nolimitz")
os.makedirs(output_dir, exist_ok=True)

spacing_um = 8.0
ball_diameter_um = 25.0

ball_diameter_px = ball_diameter_um / spacing_um
ball_radius_px = ball_diameter_px / 2.0
expected_area_px = np.pi * ball_radius_px ** 2

# Region growing should not grow much larger than the expected ball radius
max_radius = ball_radius_px * 1.5

# Area filtering based on expected object area
min_area = max(1, int(expected_area_px * 0.4))
max_area = int(expected_area_px * 2.5)
min_area = 1
max_area = 50
intensity_tolerance = 0.20
use_region_mean = False

print("Expected ball diameter px:", ball_diameter_px)
print("Expected ball radius px:", ball_radius_px)
print("Expected area px:", expected_area_px)
print("Using max_radius:", max_radius)
print("Using min_area:", min_area)
print("Using max_area:", max_area)


# ============================================================
# 1. Read image
# ============================================================
img, nii = read_and_normalize_image(input_path)


# ============================================================
# 2. Read manual markers
# ============================================================
if not os.path.exists(manual_marker_path):
    raise FileNotFoundError(
        f"Cannot find {manual_marker_path}\n"
        f"Please copy markers_auto.nii.gz to markers_manual_labeled.nii.gz and edit it first."
    )

markers = prepare_manual_markers(manual_marker_path)


# ============================================================
# 3. Read dot mask
# ============================================================
if not os.path.exists(dot_mask_path):
    raise FileNotFoundError(
        f"Cannot find {dot_mask_path}\n"
        f"Please run 01_generate_blob_seeds.py first."
    )

dot_mask_data, _ = load_nii_as_2d(dot_mask_path)
dot_mask = dot_mask_data > 0

print("Dot mask pixels:", dot_mask.sum())


# ============================================================
# 4. Region growing
# ============================================================
labels_rg_raw = region_growing_from_markers(
    img=img,
    markers=markers,
    mask=dot_mask,
    intensity_tolerance=intensity_tolerance,
    max_radius=max_radius,
    use_region_mean=use_region_mean,
)

print("Raw region growing objects:", labels_rg_raw.max())


# ============================================================
# 5. Area filtering
# ============================================================
clean = np.zeros_like(labels_rg_raw, dtype=np.int32)
new_id = 1

for r in regionprops(labels_rg_raw):
    if min_area <= r.area <= max_area:
        clean[labels_rg_raw == r.label] = new_id
        new_id += 1

labels_rg = clean.astype(np.uint16)

print("Final region growing objects:", labels_rg.max())


# ============================================================
# 6. Binary mask and centroids
# ============================================================
binary_mask = (labels_rg > 0).astype(np.uint8)

props = regionprops_table(
    labels_rg,
    intensity_image=img,
    properties=[
        "label",
        "area",
        "centroid",
        "equivalent_diameter_area",
        "mean_intensity",
    ],
)

df = pd.DataFrame(props)
df = df.rename(columns={
    "centroid-0": "centroid_y",
    "centroid-1": "centroid_x",
})

df.to_csv(
    os.path.join(output_dir, "region_growing_centroids.csv"),
    index=False
)


# ============================================================
# 7. Centroid point mask
# ============================================================
centroid_points = np.zeros_like(labels_rg, dtype=np.uint8)

for r in regionprops(labels_rg):
    cy, cx = r.centroid
    yi = int(round(cy))
    xi = int(round(cx))

    if 0 <= yi < labels_rg.shape[0] and 0 <= xi < labels_rg.shape[1]:
        centroid_points[yi, xi] = 1

centroid_points_vis = ndi.binary_dilation(
    centroid_points > 0,
    structure=np.ones((3, 3))
).astype(np.uint8)


# ============================================================
# 8. Save outputs
# ============================================================
save_nii(
    labels_rg,
    os.path.join(output_dir, "region_growing_instance_mask.nii.gz"),
    nii,
    dtype=np.uint16
)

save_nii(
    binary_mask,
    os.path.join(output_dir, "region_growing_binary_mask.nii.gz"),
    nii,
    dtype=np.uint8
)

save_nii(
    labels_rg_raw.astype(np.uint16),
    os.path.join(output_dir, "region_growing_raw_instance_mask.nii.gz"),
    nii,
    dtype=np.uint16
)

save_nii(
    markers.astype(np.uint16),
    os.path.join(output_dir, "markers_used_for_region_growing.nii.gz"),
    nii,
    dtype=np.uint16
)

save_nii(
    centroid_points,
    os.path.join(output_dir, "centroid_points.nii.gz"),
    nii,
    dtype=np.uint8
)

save_nii(
    centroid_points_vis,
    os.path.join(output_dir, "centroid_points_vis.nii.gz"),
    nii,
    dtype=np.uint8
)

save_nii(
    img,
    os.path.join(output_dir, "original_normalized.nii.gz"),
    nii,
    dtype=np.float32
)

print("Done.")
print("Output directory:", output_dir)
print("Final region growing objects:", labels_rg.max())
print(df.head())
