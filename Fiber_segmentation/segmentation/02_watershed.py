import os
import numpy as np
import nibabel as nib
import pandas as pd

from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import remove_small_objects
from skimage.morphology import binary_closing, binary_opening, disk
from skimage.segmentation import watershed


# ============================================================
# Paths
# ============================================================
input_path = "/autofs/arch11/DATA/HOMES/yuchen/phantom/5mmIprobe_3mm_phantom.nii"

work_dir = "/autofs/arch11/DATA/HOMES/yuchen/phantom/output_manual_seed"

seed_path = os.path.join(work_dir, "markers_manual_labeled.nii.gz")
output_dir = "/autofs/arch11/DATA/HOMES/yuchen/phantom/output_manual_seed/watershed_from_existing_seeds_nolimit_30voxel"
os.makedirs(output_dir, exist_ok=True)


# ============================================================
# Helper: save 2D image as 3D NIfTI mask / volume
# ============================================================
def save_nii(data, out_path, reference_nii, dtype=None):
    arr = np.asarray(data)

    if dtype is not None:
        arr = arr.astype(dtype)

    if arr.ndim == 2:
        arr = arr[:, :, None]

    affine = reference_nii.affine
    header = reference_nii.header.copy()

    out_nii = nib.Nifti1Image(arr, affine, header)

    if dtype is not None:
        out_nii.set_data_dtype(dtype)

    nib.save(out_nii, out_path)


# ============================================================
# Helper: save RGB visualization as NIfTI
# ============================================================
def save_rgb_nii(rgb, out_path, reference_nii):
    rgb = np.asarray(rgb).astype(np.uint8)

    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with shape H x W x 3, got {rgb.shape}")

    rgb = rgb[:, :, None, :]

    affine = reference_nii.affine
    header = reference_nii.header.copy()

    out_nii = nib.Nifti1Image(rgb, affine, header)
    out_nii.set_data_dtype(np.uint8)

    nib.save(out_nii, out_path)


# ============================================================
# 1. Read image
# ============================================================
nii = nib.load(input_path)
img_raw = nii.get_fdata().astype(np.float32)

print("Original image shape:", img_raw.shape)

if img_raw.ndim == 3 and img_raw.shape[2] == 1:
    img = img_raw[:, :, 0]
elif img_raw.ndim == 2:
    img = img_raw
else:
    raise ValueError(f"Expected 2D image or 3D image with one slice, got shape {img_raw.shape}")

# Normalize image to 0-1
img = img - np.nanmin(img)
img = img / (np.nanmax(img) + 1e-8)

# Denoise / smooth only for mask generation
img_for_mask = ndi.median_filter(img, size=3)
# ============================================================
# 2. Find large phantom disk mask
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

print("Disk mask pixels:", np.sum(disk_mask))


# ============================================================
# 3. Segment dark dots roughly
# ============================================================
inside_values = img[disk_mask]
dot_thresh = threshold_otsu(inside_values)

dot_mask = (img < dot_thresh) & disk_mask
dot_mask = remove_small_objects(dot_mask, min_size=2)

print("Dot mask pixels:", np.sum(dot_mask))


# ============================================================
# 4. Read existing seeds and convert to watershed markers
# ============================================================
seed_nii = nib.load(seed_path)
seed_raw = seed_nii.get_fdata()

print("Seed raw shape:", seed_raw.shape)

if seed_raw.ndim == 3 and seed_raw.shape[2] == 1:
    seed = seed_raw[:, :, 0]
elif seed_raw.ndim == 2:
    seed = seed_raw
else:
    raise ValueError(f"Expected 2D seed or 3D seed with one slice, got shape {seed_raw.shape}")

if seed.shape != img.shape:
    raise ValueError(
        f"Seed shape {seed.shape} does not match image shape {img.shape}. "
        "The seed mask and image must be in the same pixel space."
    )

seed_nonzero = seed > 0

print("Seed non-zero pixels before masking:", np.sum(seed_nonzero))

# 只保留 dot_mask 内的 seed
# 这一步很重要：seed 必须在 watershed 的 mask 里面，否则不会生长
seed_nonzero = seed_nonzero & dot_mask

print("Seed non-zero pixels inside dot_mask:", np.sum(seed_nonzero))

if np.sum(seed_nonzero) == 0:
    raise RuntimeError(
        "No seed pixels remain inside dot_mask. "
        "Check whether seed and image are aligned, or whether dot_mask is too strict."
    )

unique_seed_values = np.unique(seed[seed_nonzero])
unique_seed_values = unique_seed_values[unique_seed_values > 0]

print("Unique positive seed values inside dot_mask:", len(unique_seed_values))


# ------------------------------------------------------------
# Case 1: labeled seed mask
# ------------------------------------------------------------
# 如果 seed 里有很多不同 label，例如 1,2,3,...，直接保留 label。
# 但为了避免 label 不连续或 float 问题，这里重新 relabel 一次。
if len(unique_seed_values) > 1:
    print("Existing seeds look like labeled markers. Relabeling them cleanly...")

    markers = np.zeros_like(seed, dtype=np.int32)
    new_id = 1

    for old_id in unique_seed_values:
        obj = (seed == old_id) & dot_mask
        if np.sum(obj) == 0:
            continue

        markers[obj] = new_id
        new_id += 1

# ------------------------------------------------------------
# Case 2: binary seed mask
# ------------------------------------------------------------
# 如果 seed 只有 0/1，那就用 connected components 转成 labeled markers。
else:
    print("Existing seeds look like binary markers. Connected-component labeling...")

    markers = label(seed_nonzero).astype(np.int32)


print("Marker pixels:", np.sum(markers > 0))
print("Number of marker labels:", markers.max())

if markers.max() == 0:
    raise RuntimeError("No valid watershed markers found.")


# ============================================================
# 5. Watershed on distance transform with max expansion limit
# ============================================================

max_expand_voxels = 2

# Each pixel stores distance to the nearest seed pixel
seed_distance = ndi.distance_transform_edt(markers == 0)

# Only allow watershed to grow inside dot_mask AND within 30 voxels from any seed
growth_limit_mask = dot_mask & (seed_distance <= max_expand_voxels)

distance = ndi.distance_transform_edt(dot_mask)

labels_ws = watershed(
    -distance,
    markers,
    mask=growth_limit_mask,
)

print("Growth limit mask pixels:", np.sum(growth_limit_mask))
print("Raw watershed max label:", labels_ws.max())
print("Raw watershed actual objects:", len(regionprops(labels_ws)))

# ============================================================
# 6. Area filtering
# ============================================================
clean = np.zeros_like(labels_ws, dtype=np.int32)
new_id = 1

# 根据你的球直径 25 um、spacing 8 um，大概 area 是 7.7 px
# 可以先放宽一点，避免全被删掉
min_area = 1
max_area = 50

for r in regionprops(labels_ws):
    if min_area <= r.area <= max_area:
        clean[labels_ws == r.label] = new_id
        new_id += 1

labels_ws = clean.astype(np.uint16)

print("Final watershed objects:", labels_ws.max())


# ============================================================
# 7. Create binary mask
# ============================================================
binary_mask = (labels_ws > 0).astype(np.uint8)


# ============================================================
# 8. Save centroids as CSV
# ============================================================
props = regionprops_table(
    labels_ws,
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
    os.path.join(output_dir, "watershed_centroids.csv"),
    index=False
)


# ============================================================
# 9. Create centroid point mask
# ============================================================
centroid_points = np.zeros_like(labels_ws, dtype=np.uint8)

for r in regionprops(labels_ws):
    cy, cx = r.centroid

    yi = int(round(cy))
    xi = int(round(cx))

    if 0 <= yi < labels_ws.shape[0] and 0 <= xi < labels_ws.shape[1]:
        centroid_points[yi, xi] = 1

centroid_points_vis = ndi.binary_dilation(
    centroid_points > 0,
    structure=np.ones((3, 3))
).astype(np.uint8)


# ============================================================
# 10. Same-color shaded visualization
# ============================================================
def make_same_color_shaded_labels(
    labels,
    base_color=(0.1, 0.8, 0.25),
    background_color=(0, 0, 0),
    edge_darkness=0.35,
    center_brightness=1.35,
    boundary_darkness=0.12,
    overall_darkness=1.0,
):
    labels = labels.astype(np.int32)
    h, w = labels.shape

    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[:, :] = np.array(background_color, dtype=np.float32)

    base_color = np.array(base_color, dtype=np.float32)
    fg = labels > 0

    shading = np.zeros_like(labels, dtype=np.float32)

    for lab_id in np.unique(labels):
        if lab_id == 0:
            continue

        obj = labels == lab_id
        dist = ndi.distance_transform_edt(obj)

        if dist.max() > 0:
            dist_norm = dist / dist.max()
        else:
            dist_norm = dist

        obj_shading = edge_darkness + (center_brightness - edge_darkness) * dist_norm
        shading[obj] = obj_shading[obj]

    for c in range(3):
        rgb[..., c][fg] = base_color[c] * shading[fg]

    boundary = np.zeros_like(labels, dtype=bool)

    boundary[:-1, :] |= (labels[:-1, :] != labels[1:, :]) & fg[:-1, :]
    boundary[1:, :]  |= (labels[:-1, :] != labels[1:, :]) & fg[1:, :]

    boundary[:, :-1] |= (labels[:, :-1] != labels[:, 1:]) & fg[:, :-1]
    boundary[:, 1:]  |= (labels[:, :-1] != labels[:, 1:]) & fg[:, 1:]

    boundary = ndi.binary_dilation(
        boundary,
        structure=np.ones((2, 2))
    )

    boundary = boundary & fg

    rgb[fg] *= overall_darkness
    rgb[boundary] *= boundary_darkness

    rgb = np.clip(rgb, 0, 1)

    return (rgb * 255).astype(np.uint8)


def make_scalar_shaded_labels(
    labels,
    edge_darkness=0.35,
    center_brightness=1.35,
    boundary_darkness=0.12,
    overall_darkness=1.0,
):
    labels = labels.astype(np.int32)
    scalar = np.zeros_like(labels, dtype=np.float32)
    fg = labels > 0

    for lab_id in np.unique(labels):
        if lab_id == 0:
            continue

        obj = labels == lab_id
        dist = ndi.distance_transform_edt(obj)

        if dist.max() > 0:
            dist_norm = dist / dist.max()
        else:
            dist_norm = dist

        obj_shading = edge_darkness + (center_brightness - edge_darkness) * dist_norm
        scalar[obj] = obj_shading[obj]

    boundary = np.zeros_like(labels, dtype=bool)

    boundary[:-1, :] |= (labels[:-1, :] != labels[1:, :]) & fg[:-1, :]
    boundary[1:, :]  |= (labels[:-1, :] != labels[1:, :]) & fg[1:, :]

    boundary[:, :-1] |= (labels[:, :-1] != labels[:, 1:]) & fg[:, :-1]
    boundary[:, 1:]  |= (labels[:, :-1] != labels[:, 1:]) & fg[:, 1:]

    boundary = ndi.binary_dilation(
        boundary,
        structure=np.ones((2, 2))
    )

    boundary = boundary & fg

    scalar[fg] *= overall_darkness
    scalar[boundary] *= boundary_darkness

    scalar = np.clip(scalar, 0, 1)

    return scalar.astype(np.float32)


shaded_rgb = make_same_color_shaded_labels(
    labels_ws,
    base_color=(0.1, 0.8, 0.25),
    edge_darkness=0.35,
    center_brightness=1.35,
    boundary_darkness=0.12,
    overall_darkness=1.0,
)

shaded_scalar = make_scalar_shaded_labels(
    labels_ws,
    edge_darkness=0.35,
    center_brightness=1.35,
    boundary_darkness=0.12,
    overall_darkness=1.0,
)


# ============================================================
# 11. Save all outputs as NIfTI
# ============================================================

# Main masks
save_nii(
    labels_ws,
    os.path.join(output_dir, "watershed_instance_mask.nii.gz"),
    nii,
    dtype=np.uint16
)

save_nii(
    binary_mask,
    os.path.join(output_dir, "watershed_binary_mask.nii.gz"),
    nii,
    dtype=np.uint8
)

# Intermediate images / masks
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
    markers.astype(np.uint16),
    os.path.join(output_dir, "markers_from_existing_seeds.nii.gz"),
    nii,
    dtype=np.uint16
)

save_nii(
    distance.astype(np.float32),
    os.path.join(output_dir, "distance_map.nii.gz"),
    nii,
    dtype=np.float32
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

# Visualization only
save_nii(
    shaded_scalar,
    os.path.join(output_dir, "watershed_same_color_shaded_scalar.nii.gz"),
    nii,
    dtype=np.float32
)

save_rgb_nii(
    shaded_rgb,
    os.path.join(output_dir, "watershed_same_color_shaded_rgb.nii.gz"),
    nii
)


print("Done.")
print("Output directory:", output_dir)
print("Final watershed objects:", labels_ws.max())
print(df.head())
