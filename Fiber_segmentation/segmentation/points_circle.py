import os
import numpy as np
import nibabel as nib
import pandas as pd


# ============================================================
# User settings
# ============================================================

base_dir = (
    "/autofs/arch11/DATA/HOMES/yuchen/Minor/Phantom/"
    "output_manual_seed/watershed_from_existing_seeds_nolimit"
)

centroid_csv_path = os.path.join(base_dir, "watershed_centroids.csv")
reference_nii_path = os.path.join(base_dir, "watershed_instance_mask.nii.gz")
alternative_reference_nii_path = os.path.join(base_dir, "original_normalized.nii.gz")

output_dir = os.path.join(base_dir, "centroid_rings_highres")
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# High-resolution factor
# Example:
#   original spacing = 0.008 mm
#   factor = 4  --> new spacing = 0.002 mm
# -----------------------------
upsample_factor = 8

# -----------------------------
# Physical circle size
# 25 um = 0.025 mm
# -----------------------------
circle_diameter_mm = 0.025
circle_radius_mm = circle_diameter_mm / 2.0

# Ring thickness in physical size
# For factor=4, new spacing will be 0.002 mm, so 0.004 mm is about 2 high-res voxels
ring_thickness_mm = 0.002

# Whether instance rings should avoid overwriting previous labels
# True: later rings only fill empty voxels
# False: later rings may overwrite earlier labels
instance_no_overlap = True


# ============================================================
# Helper functions
# ============================================================

def check_file_exists(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{name} not found:\n{path}\n\n"
            "Please check whether the path is correct."
        )


def get_2d_shape_from_reference(ref_data):
    """
    Accept:
      - 2D image: H x W
      - 3D one-slice image: H x W x 1
    """
    if ref_data.ndim == 2:
        return ref_data.shape, False
    elif ref_data.ndim == 3 and ref_data.shape[2] == 1:
        return ref_data[:, :, 0].shape, True
    else:
        raise ValueError(
            f"Expected 2D image or 3D image with one slice, got shape {ref_data.shape}"
        )


def get_spacing_xy_mm(ref_nii):
    zooms = ref_nii.header.get_zooms()
    if len(zooms) < 2:
        raise ValueError(f"Cannot read x/y spacing from zooms: {zooms}")

    # axis 0 -> y, axis 1 -> x
    spacing_y_mm = float(zooms[0])
    spacing_x_mm = float(zooms[1])
    return spacing_y_mm, spacing_x_mm


def build_highres_affine_and_shape(ref_nii, image_shape_2d, upsample_factor):
    """
    Build a high-resolution grid that preserves the same physical field of view.

    Original image:
        shape = (H, W)
        affine columns:
            col0 -> axis 0 direction vector
            col1 -> axis 1 direction vector

    New image:
        shape = (H*F, W*F)
        spacing = old_spacing / F
        affine columns = old columns / F

    To preserve the same outer physical bounds, the new origin is shifted by:
        -0.5 * old_vec * (1 - 1/F)
    along both in-plane axes.
    """
    H, W = image_shape_2d
    F = float(upsample_factor)

    old_affine = ref_nii.affine.copy()
    new_affine = old_affine.copy()

    v0 = old_affine[:3, 0].copy()   # axis 0 vector
    v1 = old_affine[:3, 1].copy()   # axis 1 vector
    t = old_affine[:3, 3].copy()    # origin

    new_affine[:3, 0] = v0 / F
    new_affine[:3, 1] = v1 / F

    shift = -0.5 * v0 * (1.0 - 1.0 / F) - 0.5 * v1 * (1.0 - 1.0 / F)
    new_affine[:3, 3] = t + shift

    new_shape_2d = (int(H * upsample_factor), int(W * upsample_factor))
    return new_affine, new_shape_2d


def original_index_to_highres_index(c, upsample_factor):
    """
    Convert original index coordinate to high-resolution index coordinate,
    while preserving physical position under the bounds-preserving affine.

    Formula:
        c_high = F * c + (F - 1) / 2
               = (c + 0.5) * F - 0.5
    """
    F = float(upsample_factor)
    return (c + 0.5) * F - 0.5


def draw_ring_physical(
    mask,
    center_y,
    center_x,
    radius_mm,
    thickness_mm,
    spacing_y_mm,
    spacing_x_mm,
    value=1,
    overwrite=True,
):
    """
    Draw a ring (circle boundary) on a 2D mask using physical distances.

    center_y, center_x can be floats.
    """
    h, w = mask.shape

    cy = float(center_y)
    cx = float(center_x)

    outer_radius_mm = radius_mm + thickness_mm / 2.0
    inner_radius_mm = max(0.0, radius_mm - thickness_mm / 2.0)

    outer_radius_y_vox = outer_radius_mm / spacing_y_mm
    outer_radius_x_vox = outer_radius_mm / spacing_x_mm

    y_min = max(0, int(np.floor(cy - outer_radius_y_vox - 1)))
    y_max = min(h, int(np.ceil(cy + outer_radius_y_vox + 1)) + 1)
    x_min = max(0, int(np.floor(cx - outer_radius_x_vox - 1)))
    x_max = min(w, int(np.ceil(cx + outer_radius_x_vox + 1)) + 1)

    yy, xx = np.ogrid[y_min:y_max, x_min:x_max]

    dist_mm = np.sqrt(
        ((yy - cy) * spacing_y_mm) ** 2 +
        ((xx - cx) * spacing_x_mm) ** 2
    )

    ring = (dist_mm >= inner_radius_mm) & (dist_mm <= outer_radius_mm)

    local = mask[y_min:y_max, x_min:x_max]

    if overwrite:
        local[ring] = value
    else:
        ring_no_overlap = ring & (local == 0)
        local[ring_no_overlap] = value


def save_highres_nii(
    data_2d,
    reference_nii,
    out_path,
    new_affine,
    upsample_factor,
    reference_was_3d_singleton,
    dtype,
):
    """
    Save high-resolution output while preserving physical consistency.
    """
    arr = np.asarray(data_2d).astype(dtype)

    old_zooms = reference_nii.header.get_zooms()
    header = reference_nii.header.copy()

    if reference_was_3d_singleton:
        arr = arr[:, :, None]
        if len(old_zooms) >= 3:
            new_zooms = (
                float(old_zooms[0]) / upsample_factor,
                float(old_zooms[1]) / upsample_factor,
                float(old_zooms[2]),
            )
        else:
            new_zooms = (
                float(old_zooms[0]) / upsample_factor,
                float(old_zooms[1]) / upsample_factor,
                1.0,
            )
    else:
        new_zooms = (
            float(old_zooms[0]) / upsample_factor,
            float(old_zooms[1]) / upsample_factor,
        )

    out_nii = nib.Nifti1Image(arr, new_affine, header)
    out_nii.set_data_dtype(dtype)

    # Update zooms to match new spacing
    out_nii.header.set_zooms(new_zooms)

    nib.save(out_nii, out_path)


# ============================================================
# 1. Check inputs
# ============================================================

check_file_exists(centroid_csv_path, "Centroid CSV")

if not os.path.exists(reference_nii_path):
    print("Reference NIfTI not found:")
    print(reference_nii_path)
    print("Trying alternative reference:")
    print(alternative_reference_nii_path)
    reference_nii_path = alternative_reference_nii_path

check_file_exists(reference_nii_path, "Reference NIfTI")


# ============================================================
# 2. Load reference image
# ============================================================

ref_nii = nib.load(reference_nii_path)
ref_data = ref_nii.get_fdata()

image_shape_2d, reference_was_3d_singleton = get_2d_shape_from_reference(ref_data)
spacing_y_mm, spacing_x_mm = get_spacing_xy_mm(ref_nii)

new_affine, highres_shape_2d = build_highres_affine_and_shape(
    ref_nii,
    image_shape_2d,
    upsample_factor,
)

highres_spacing_y_mm = spacing_y_mm / upsample_factor
highres_spacing_x_mm = spacing_x_mm / upsample_factor

print("Reference NIfTI:", reference_nii_path)
print("Original 2D shape:", image_shape_2d)
print("Original spacing y/x (mm):", spacing_y_mm, spacing_x_mm)

print("Upsample factor:", upsample_factor)
print("High-res 2D shape:", highres_shape_2d)
print("High-res spacing y/x (mm):", highres_spacing_y_mm, highres_spacing_x_mm)

print("Circle diameter (mm):", circle_diameter_mm)
print("Circle radius (mm):", circle_radius_mm)
print("Ring thickness (mm):", ring_thickness_mm)

print("Circle diameter in original voxels:",
      circle_diameter_mm / spacing_y_mm,
      circle_diameter_mm / spacing_x_mm)

print("Circle diameter in high-res voxels:",
      circle_diameter_mm / highres_spacing_y_mm,
      circle_diameter_mm / highres_spacing_x_mm)


# ============================================================
# 3. Load centroid CSV
# ============================================================

df = pd.read_csv(centroid_csv_path)

required_cols = ["centroid_y", "centroid_x"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(
            f"Column '{col}' not found in CSV.\n"
            f"Available columns: {list(df.columns)}"
        )

print("Number of centroids:", len(df))


# ============================================================
# 4. Create high-resolution ring masks
# ============================================================

# Binary mask: all rings merged together
ring_mask_binary = np.zeros(highres_shape_2d, dtype=np.uint8)

# Instance mask: each ring gets a unique label
# Use uint16 if object count is not too large; use uint32 if needed
if len(df) <= 65535:
    instance_dtype = np.uint16
else:
    instance_dtype = np.uint32

ring_mask_instance = np.zeros(highres_shape_2d, dtype=instance_dtype)

for i, (_, row) in enumerate(df.iterrows(), start=1):
    cy_orig = float(row["centroid_y"])
    cx_orig = float(row["centroid_x"])

    cy_high = original_index_to_highres_index(cy_orig, upsample_factor)
    cx_high = original_index_to_highres_index(cx_orig, upsample_factor)

    # Binary output
    draw_ring_physical(
        mask=ring_mask_binary,
        center_y=cy_high,
        center_x=cx_high,
        radius_mm=circle_radius_mm,
        thickness_mm=ring_thickness_mm,
        spacing_y_mm=highres_spacing_y_mm,
        spacing_x_mm=highres_spacing_x_mm,
        value=1,
        overwrite=True,
    )

    # Instance output
    draw_ring_physical(
        mask=ring_mask_instance,
        center_y=cy_high,
        center_x=cx_high,
        radius_mm=circle_radius_mm,
        thickness_mm=ring_thickness_mm,
        spacing_y_mm=highres_spacing_y_mm,
        spacing_x_mm=highres_spacing_x_mm,
        value=i,
        overwrite=not instance_no_overlap,
    )

print("Binary ring pixels:", np.sum(ring_mask_binary > 0))
print("Instance ring pixels:", np.sum(ring_mask_instance > 0))


# ============================================================
# 5. Save outputs
# ============================================================

binary_output_path = os.path.join(
    output_dir,
    f"centroid_rings_binary_25um_factor{upsample_factor}.nii.gz"
)

instance_output_path = os.path.join(
    output_dir,
    f"centroid_rings_instance_25um_factor{upsample_factor}.nii.gz"
)

save_highres_nii(
    data_2d=ring_mask_binary,
    reference_nii=ref_nii,
    out_path=binary_output_path,
    new_affine=new_affine,
    upsample_factor=upsample_factor,
    reference_was_3d_singleton=reference_was_3d_singleton,
    dtype=np.uint8,
)

save_highres_nii(
    data_2d=ring_mask_instance,
    reference_nii=ref_nii,
    out_path=instance_output_path,
    new_affine=new_affine,
    upsample_factor=upsample_factor,
    reference_was_3d_singleton=reference_was_3d_singleton,
    dtype=instance_dtype,
)

print("Done.")
print("Saved binary mask:  ", binary_output_path)
print("Saved instance mask:", instance_output_path)