import os
import numpy as np
import pandas as pd

from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from skimage.feature import blob_log

from seg_utils import save_nii, read_and_normalize_image


# ============================================================
# Parameters
# ============================================================
input_path = "/autofs/arch11/DATA/HOMES/yuchen/phantom/5mmIprobe_3mm_phantom.nii"
output_dir = "/autofs/arch11/DATA/HOMES/yuchen/phantom/output_manual_seed"
os.makedirs(output_dir, exist_ok=True)

spacing_um = 8.0
ball_diameter_um = 25.0

blob_threshold = 0.02
blob_smooth_sigma = 0.4
min_object_size = 2


# ============================================================
# 1. Read image
# ============================================================
img, nii = read_and_normalize_image(input_path)


# ============================================================
# 2. Disk mask
# ============================================================
disk_thresh = threshold_otsu(img)
disk_binary = img > disk_thresh

lab = label(disk_binary)
regions = regionprops(lab)

if len(regions) == 0:
    raise RuntimeError("No disk region found.")

largest_label = max(regions, key=lambda r: r.area).label

disk_mask = lab == largest_label
disk_mask = ndi.binary_fill_holes(disk_mask)

disk_mask = ndi.binary_dilation(
    disk_mask,
    structure=np.ones((15, 15))
)


# Fill holes again after morphology
disk_mask = ndi.binary_fill_holes(disk_mask)

# Slightly expand disk to include edge dots
disk_mask = ndi.binary_erosion(
    disk_mask,
    structure=np.ones((15, 15))
)


# ============================================================
# 3. Rough dot mask
# ============================================================
inside_values = img[disk_mask]
dot_thresh = threshold_otsu(inside_values)

dot_mask = (img < dot_thresh) & disk_mask
dot_mask = remove_small_objects(dot_mask, min_size=min_object_size)


# ============================================================
# 4. Blob detection
# ============================================================
blob_img = np.zeros_like(img, dtype=np.float32)
blob_img[disk_mask] = 1.0 - img[disk_mask]
blob_img_smooth = ndi.gaussian_filter(blob_img, sigma=blob_smooth_sigma)

ball_diameter_px = ball_diameter_um / spacing_um
ball_radius_px = ball_diameter_px / 2.0
expected_sigma = ball_radius_px / np.sqrt(2)

min_sigma = expected_sigma * 0.8
max_sigma = expected_sigma * 1.2

print("Ball diameter px:", ball_diameter_px)
print("Ball radius px:", ball_radius_px)
print("Expected sigma:", expected_sigma)

blobs = blob_log(
    blob_img_smooth,
    min_sigma=min_sigma,
    max_sigma=max_sigma,
    num_sigma=15,
    threshold=blob_threshold,
    overlap=0.5,
)

print("Raw blobs:", len(blobs))


# ============================================================
# 5. Convert blobs to markers
# ============================================================
markers = np.zeros_like(img, dtype=np.int32)

marker_id = 1
valid_blobs = []

for y, x, sigma in blobs:
    yi = int(round(y))
    xi = int(round(x))

    if yi < 0 or yi >= img.shape[0] or xi < 0 or xi >= img.shape[1]:
        continue

    if not disk_mask[yi, xi]:
        continue

    # 如果这个条件太严格导致漏 seed，可以注释掉
    if not dot_mask[yi, xi]:
        continue

    markers[yi, xi] = marker_id

    valid_blobs.append([
        marker_id,
        y,
        x,
        yi,
        xi,
        sigma,
        np.sqrt(2) * sigma,
        2 * np.sqrt(2) * sigma,
    ])

    marker_id += 1

print("Valid blob seeds:", marker_id - 1)


# ============================================================
# 6. Save files for manual correction
# ============================================================
save_nii(
    markers.astype(np.uint16),
    os.path.join(output_dir, "markers_auto.nii.gz"),
    nii,
    dtype=np.uint16
)

save_nii(
    img,
    os.path.join(output_dir, "original_normalized.nii.gz"),
    nii,
    dtype=np.float32
)

save_nii(
    (img * 255).astype(np.uint8),
    os.path.join(output_dir, "original_normalized_uint8.nii.gz"),
    nii,
    dtype=np.uint8
)

save_nii(
    disk_mask.astype(np.uint8),
    os.path.join(output_dir, "disk_mask.nii.gz"),
    nii,
    dtype=np.uint8
)

save_nii(
    dot_mask.astype(np.uint8),
    os.path.join(output_dir, "dot_mask_binary.nii.gz"),
    nii,
    dtype=np.uint8
)

save_nii(
    blob_img.astype(np.float32),
    os.path.join(output_dir, "blob_input.nii.gz"),
    nii,
    dtype=np.float32
)

save_nii(
    blob_img_smooth.astype(np.float32),
    os.path.join(output_dir, "blob_input_smooth.nii.gz"),
    nii,
    dtype=np.float32
)

blob_df = pd.DataFrame(
    valid_blobs,
    columns=[
        "marker_id",
        "blob_y",
        "blob_x",
        "seed_y_round",
        "seed_x_round",
        "sigma",
        "estimated_radius_px",
        "estimated_diameter_px",
    ]
)

blob_df.to_csv(
    os.path.join(output_dir, "blob_seeds_auto.csv"),
    index=False
)

print("Done: automatic seeds generated.")
print("Output directory:", output_dir)
print("Next step: copy markers_auto.nii.gz to markers_manual.nii.gz and manually edit it.")