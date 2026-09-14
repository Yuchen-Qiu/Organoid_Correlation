import os
import json
import numpy as np
import SimpleITK as sitk


# ============================================================
# User settings
# ============================================================

fixed_path = (
    "/autofs/arch11/DATA/HOMES/yuchen/Minor/Organoids/Nifti_data/"
    "case2_young_Paw_Ventricles/case2_Topro3_high.nii.gz"
)

output_dir = (
    "/autofs/arch11/DATA/HOMES/yuchen/Minor/Organoids/Nifti_data/synthetic_Topro3_flip"
)

sample_prefix = "synthetic_combo"

NUM_SYNTHETIC_IMAGES = 1
RANDOM_SEED = 42


# ============================================================
# Geometric augmentation settings
# ============================================================

# Combined xyz random rotation.
# Each synthetic image gets independent x/y/z rotation.
ROT_X_RANGE_DEG = (-30.0, 30.0)
ROT_Y_RANGE_DEG = (-30.0, 30.0)
ROT_Z_RANGE_DEG = (-30.0, 30.0)

# Translation is sampled as a fraction of object bounding-box size.
# Example:
#   0.05 means max translation is 5% of object size in each axis.
#   0.10 means max translation is 10% of object size in each axis.
TRANSLATION_FRACTION = 0.2

# Random flip around object center.
# First round: keep this False.
# After rotation + translation works, change to True if needed.
ENABLE_RANDOM_FLIP = True
FLIP_PROB_X = 0.5
FLIP_PROB_Y = 0.5
FLIP_PROB_Z = 0.5

# ============================================================
# Intensity augmentation settings
# ============================================================

# First round: keep this False.
# After geometric validation works, change to True if you want robustness testing.
ENABLE_INTENSITY_AUG = False

INTENSITY_SCALE_RANGE = (0.95, 1.05)
INTENSITY_BIAS_RANGE = (-0.02, 0.02)

# Gaussian blur sigma in physical units.
BLUR_SIGMA_RANGE = (0.0, 0.3)

# Noise std relative to image intensity range.
NOISE_STD_RANGE = (0.0, 0.01)


# ============================================================
# Resampling settings
# ============================================================

# Use linear for intensity/density image.
# Use nearest neighbor if you are transforming binary masks or label images.
INTERPOLATOR = sitk.sitkLinear

# Background outside image after resampling.
DEFAULT_VALUE = 0.0


# ============================================================
# Helper functions
# ============================================================

def ensure_3d_image(img):
    """
    If image is 4D with last dimension size 1, extract the 3D volume.
    Otherwise keep it unchanged.
    """
    dim = img.GetDimension()

    if dim == 3:
        return img

    if dim == 4:
        size = list(img.GetSize())
        if size[3] != 1:
            raise ValueError(f"4D image has size {size}; last dimension is not 1.")

        extractor = sitk.ExtractImageFilter()
        extractor.SetSize([size[0], size[1], size[2], 0])
        extractor.SetIndex([0, 0, 0, 0])
        return extractor.Execute(img)

    raise ValueError(f"Only 3D or 4D image is supported, got dimension={dim}.")


def get_image_center_physical(img):
    """
    Compute image FOV center in physical coordinates.
    """
    size = np.array(img.GetSize(), dtype=float)
    center_index = (size - 1.0) / 2.0

    return np.array(
        img.TransformContinuousIndexToPhysicalPoint(center_index.tolist()),
        dtype=float
    )


def make_foreground_mask_array(img):
    """
    Estimate foreground using Otsu threshold.
    Fallback to positive voxels if Otsu produces too few voxels.

    Returns:
        mask_arr in numpy order: z, y, x
    """
    img_float = sitk.Cast(img, sitk.sitkFloat32)

    otsu = sitk.OtsuThresholdImageFilter()
    otsu.SetInsideValue(0)
    otsu.SetOutsideValue(1)
    mask_img = otsu.Execute(img_float)

    mask_arr = sitk.GetArrayFromImage(mask_img).astype(bool)
    img_arr = sitk.GetArrayFromImage(img_float)

    if np.count_nonzero(mask_arr) < 20:
        mask_arr = img_arr > 0

    if np.count_nonzero(mask_arr) < 20:
        raise ValueError("Too few foreground voxels. Cannot estimate object.")

    return mask_arr


def get_object_center_physical(img):
    """
    Compute object center from foreground voxels.
    The returned center is in physical coordinates.
    """
    mask_arr = make_foreground_mask_array(img)

    # numpy index order: z, y, x
    zyx = np.argwhere(mask_arr)

    # convert to SimpleITK index order: x, y, z
    xyz_indices = zyx[:, [2, 1, 0]].astype(float)

    # subsample for speed
    max_points = 200000
    if xyz_indices.shape[0] > max_points:
        rng = np.random.default_rng(123)
        idx = rng.choice(xyz_indices.shape[0], size=max_points, replace=False)
        xyz_indices = xyz_indices[idx]

    physical_points = np.array([
        img.TransformContinuousIndexToPhysicalPoint(p.tolist())
        for p in xyz_indices
    ], dtype=float)

    center = physical_points.mean(axis=0)
    return center


def get_object_bbox_physical_size(img):
    """
    Estimate object bounding-box size in physical units.

    Returns:
        np.array([size_x, size_y, size_z])
    """
    mask_arr = make_foreground_mask_array(img)

    # numpy order: z, y, x
    zyx = np.argwhere(mask_arr)

    z_min, y_min, x_min = zyx.min(axis=0)
    z_max, y_max, x_max = zyx.max(axis=0)

    spacing = np.array(img.GetSpacing(), dtype=float)  # x, y, z

    size_x = (x_max - x_min + 1) * spacing[0]
    size_y = (y_max - y_min + 1) * spacing[1]
    size_z = (z_max - z_min + 1) * spacing[2]

    return np.array([size_x, size_y, size_z], dtype=float)


def rotation_matrix_x(angle_deg):
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)

    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ], dtype=float)


def rotation_matrix_y(angle_deg):
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)

    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ], dtype=float)


def rotation_matrix_z(angle_deg):
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)

    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ], dtype=float)


def combined_rotation_matrix_xyz(rot_x_deg, rot_y_deg, rot_z_deg):
    """
    Combined xyz rotation.

    R = Rz @ Ry @ Rx means:
        first rotate around x,
        then rotate around y,
        then rotate around z.
    """
    Rx = rotation_matrix_x(rot_x_deg)
    Ry = rotation_matrix_y(rot_y_deg)
    Rz = rotation_matrix_z(rot_z_deg)

    return Rz @ Ry @ Rx


def flip_matrix(flip_x=False, flip_y=False, flip_z=False):
    """
    Mirror flip along image physical x/y/z axes.

    This is performed around the transform center.
    """
    sx = -1.0 if flip_x else 1.0
    sy = -1.0 if flip_y else 1.0
    sz = -1.0 if flip_z else 1.0

    return np.diag([sx, sy, sz]).astype(float)


def make_affine_transform(matrix, center, translation):
    """
    Create a SimpleITK AffineTransform from matrix, center and translation.

    The transform is the conceptual forward transform:
        original fixed image -> synthetic moving image
    """
    tx = sitk.AffineTransform(3)
    tx.SetMatrix(matrix.reshape(-1).tolist())
    tx.SetCenter(center.tolist())
    tx.SetTranslation(np.asarray(translation, dtype=float).tolist())

    return tx


def resample_with_forward_transform(img, forward_transform):
    """
    Generate synthetic moving image.

    Conceptually:
        synthetic_moving = forward_transform(original_fixed)

    SimpleITK Resample expects an output-to-input transform.
    Therefore, we pass inverse(forward_transform).
    """
    inverse_transform = forward_transform.GetInverse()

    out = sitk.Resample(
        img,
        img,                 # keep the same grid as original fixed image
        inverse_transform,
        INTERPOLATOR,
        DEFAULT_VALUE,
        sitk.sitkFloat32
    )

    out.CopyInformation(img)
    return out


def sample_uniform(rng, range_tuple):
    low, high = range_tuple
    return float(rng.uniform(low, high))


def apply_intensity_augmentation(img, rng):
    """
    Optional intensity augmentation:
        blur -> scale + bias -> noise
    """
    if not ENABLE_INTENSITY_AUG:
        return img, {
            "enabled": False
        }

    img = sitk.Cast(img, sitk.sitkFloat32)

    scale = sample_uniform(rng, INTENSITY_SCALE_RANGE)
    bias = sample_uniform(rng, INTENSITY_BIAS_RANGE)
    blur_sigma = sample_uniform(rng, BLUR_SIGMA_RANGE)
    noise_std_relative = sample_uniform(rng, NOISE_STD_RANGE)

    if blur_sigma > 0:
        img = sitk.SmoothingRecursiveGaussian(img, blur_sigma)

    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    finite_mask = np.isfinite(arr)

    if not np.any(finite_mask):
        raise ValueError("No finite voxel values found during intensity augmentation.")

    val_min = float(arr[finite_mask].min())
    val_max = float(arr[finite_mask].max())
    val_range = max(val_max - val_min, 1e-8)

    arr = arr * scale + bias

    noise_std_abs = noise_std_relative * val_range
    if noise_std_abs > 0:
        arr += rng.normal(
            loc=0.0,
            scale=noise_std_abs,
            size=arr.shape
        ).astype(np.float32)

    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)

    return out, {
        "enabled": True,
        "intensity_scale": scale,
        "intensity_bias": bias,
        "blur_sigma": blur_sigma,
        "noise_std_relative": noise_std_relative,
        "noise_std_absolute": noise_std_abs
    }


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)

    img = sitk.ReadImage(fixed_path)
    img = ensure_3d_image(img)
    img = sitk.Cast(img, sitk.sitkFloat32)

    image_center = get_image_center_physical(img)
    object_center = get_object_center_physical(img)
    object_bbox_size = get_object_bbox_physical_size(img)

    trans_x_range = (
        -TRANSLATION_FRACTION * object_bbox_size[0],
         TRANSLATION_FRACTION * object_bbox_size[0]
    )
    trans_y_range = (
        -TRANSLATION_FRACTION * object_bbox_size[1],
         TRANSLATION_FRACTION * object_bbox_size[1]
    )
    trans_z_range = (
        -TRANSLATION_FRACTION * object_bbox_size[2],
         TRANSLATION_FRACTION * object_bbox_size[2]
    )

    print("Image center physical:")
    print(image_center)

    print("Object center physical:")
    print(object_center)

    print("Object bbox physical size:")
    print(object_bbox_size)

    print("Translation ranges:")
    print("x:", trans_x_range)
    print("y:", trans_y_range)
    print("z:", trans_z_range)

    center = object_center

    summary = {
        "source_fixed_path": fixed_path,
        "output_dir": output_dir,
        "num_synthetic_images": NUM_SYNTHETIC_IMAGES,
        "random_seed": RANDOM_SEED,
        "image_center_physical": image_center.tolist(),
        "object_center_physical": object_center.tolist(),
        "object_bbox_physical_size": object_bbox_size.tolist(),
        "used_transform_center": "object_center_physical",
        "rotation_ranges_deg": {
            "x": list(ROT_X_RANGE_DEG),
            "y": list(ROT_Y_RANGE_DEG),
            "z": list(ROT_Z_RANGE_DEG)
        },
        "translation_fraction": TRANSLATION_FRACTION,
        "translation_ranges": {
            "x": list(trans_x_range),
            "y": list(trans_y_range),
            "z": list(trans_z_range)
        },
        "enable_random_flip": ENABLE_RANDOM_FLIP,
        "flip_probabilities": {
            "x": FLIP_PROB_X,
            "y": FLIP_PROB_Y,
            "z": FLIP_PROB_Z
        },
        "enable_intensity_augmentation": ENABLE_INTENSITY_AUG,
        "generated_cases": []
    }

    for i in range(NUM_SYNTHETIC_IMAGES):
        # -------------------------
        # Random combined xyz rotation
        # -------------------------
        rot_x = sample_uniform(rng, ROT_X_RANGE_DEG)
        rot_y = sample_uniform(rng, ROT_Y_RANGE_DEG)
        rot_z = sample_uniform(rng, ROT_Z_RANGE_DEG)

        # -------------------------
        # Object-size-based translation
        # -------------------------
        tx = sample_uniform(rng, trans_x_range)
        ty = sample_uniform(rng, trans_y_range)
        tz = sample_uniform(rng, trans_z_range)

        translation = [tx, ty, tz]

        # -------------------------
        # Optional random flip
        # -------------------------
        if ENABLE_RANDOM_FLIP:
            flip_x = bool(rng.random() < FLIP_PROB_X)
            flip_y = bool(rng.random() < FLIP_PROB_Y)
            flip_z = bool(rng.random() < FLIP_PROB_Z)
        else:
            flip_x = False
            flip_y = False
            flip_z = False

        R = combined_rotation_matrix_xyz(rot_x, rot_y, rot_z)
        F = flip_matrix(flip_x, flip_y, flip_z)

        # Combined transform:
        # first flip, then rotate.
        combined_matrix = R @ F

        forward_transform = make_affine_transform(
            matrix=combined_matrix,
            center=center,
            translation=translation
        )

        synthetic_img = resample_with_forward_transform(
            img=img,
            forward_transform=forward_transform
        )

        synthetic_img, intensity_info = apply_intensity_augmentation(
            synthetic_img,
            rng
        )

        out_img_path = os.path.join(
            output_dir,
            f"{sample_prefix}_{i+1:02d}.nii.gz"
        )

        sitk.WriteImage(synthetic_img, out_img_path)

        case_info = {
            "case_id": i + 1,
            "output_image": out_img_path,
            "rotation_deg": {
                "x": rot_x,
                "y": rot_y,
                "z": rot_z
            },
            "translation": {
                "x": tx,
                "y": ty,
                "z": tz
            },
            "translation_fraction": TRANSLATION_FRACTION,
            "flip": {
                "x": flip_x,
                "y": flip_y,
                "z": flip_z
            },
            "transform_center_physical": center.tolist(),
            "combined_forward_matrix": combined_matrix.tolist(),
            "intensity_augmentation": intensity_info
        }

        summary["generated_cases"].append(case_info)

        print("=" * 70)
        print(f"Generated synthetic image {i+1}")
        print(f"Saved to: {out_img_path}")
        print(f"Rotation: x={rot_x:.2f}, y={rot_y:.2f}, z={rot_z:.2f}")
        print(f"Translation: x={tx:.3f}, y={ty:.3f}, z={tz:.3f}")
        print(f"Flip: x={flip_x}, y={flip_y}, z={flip_z}")

    out_json_path = os.path.join(
        output_dir,
        f"{sample_prefix}_summary.json"
    )

    with open(out_json_path, "w") as f:
        json.dump(summary, f, indent=4)

    print("=" * 70)
    print("All synthetic images generated.")
    print("Summary saved to:")
    print(out_json_path)


if __name__ == "__main__":
    main()