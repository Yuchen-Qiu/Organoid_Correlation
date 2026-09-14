#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import SimpleITK as sitk

from functions_weighted import (
    apply_intensity_augmentation,
    combined_rotation_matrix_xyz,
    ensure_3d_image,
    flip_matrix,
    get_image_center_physical,
    get_object_bbox_physical_size,
    get_object_center_physical,
    make_affine_transform,
    resample_with_forward_transform,
    sample_uniform,
    write_json,
)

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

ROT_X_RANGE_DEG = (-30.0, 30.0)
ROT_Y_RANGE_DEG = (-30.0, 30.0)
ROT_Z_RANGE_DEG = (-30.0, 30.0)

TRANSLATION_FRACTION = 0.2

ENABLE_RANDOM_FLIP = True
FLIP_PROB_X = 0.5
FLIP_PROB_Y = 0.5
FLIP_PROB_Z = 0.5

# ============================================================
# Intensity augmentation settings
# ============================================================

ENABLE_INTENSITY_AUG = False

INTENSITY_SCALE_RANGE = (0.95, 1.05)
INTENSITY_BIAS_RANGE = (-0.02, 0.02)
BLUR_SIGMA_RANGE = (0.0, 0.3)
NOISE_STD_RANGE = (0.0, 0.01)

# ============================================================
# Resampling settings
# ============================================================

INTERPOLATOR = sitk.sitkLinear
DEFAULT_VALUE = 0.0


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
        TRANSLATION_FRACTION * object_bbox_size[0],
    )
    trans_y_range = (
        -TRANSLATION_FRACTION * object_bbox_size[1],
        TRANSLATION_FRACTION * object_bbox_size[1],
    )
    trans_z_range = (
        -TRANSLATION_FRACTION * object_bbox_size[2],
        TRANSLATION_FRACTION * object_bbox_size[2],
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
            "z": list(ROT_Z_RANGE_DEG),
        },
        "translation_fraction": TRANSLATION_FRACTION,
        "translation_ranges": {
            "x": list(trans_x_range),
            "y": list(trans_y_range),
            "z": list(trans_z_range),
        },
        "enable_random_flip": ENABLE_RANDOM_FLIP,
        "flip_probabilities": {
            "x": FLIP_PROB_X,
            "y": FLIP_PROB_Y,
            "z": FLIP_PROB_Z,
        },
        "enable_intensity_augmentation": ENABLE_INTENSITY_AUG,
        "generated_cases": [],
    }

    for i in range(NUM_SYNTHETIC_IMAGES):
        rot_x = sample_uniform(rng, ROT_X_RANGE_DEG)
        rot_y = sample_uniform(rng, ROT_Y_RANGE_DEG)
        rot_z = sample_uniform(rng, ROT_Z_RANGE_DEG)

        tx = sample_uniform(rng, trans_x_range)
        ty = sample_uniform(rng, trans_y_range)
        tz = sample_uniform(rng, trans_z_range)
        translation = [tx, ty, tz]

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
        combined_matrix = R @ F

        forward_transform = make_affine_transform(
            matrix=combined_matrix,
            center=center,
            translation=translation,
        )

        synthetic_img = resample_with_forward_transform(
            img=img,
            forward_transform=forward_transform,
            interpolator=INTERPOLATOR,
            default_value=DEFAULT_VALUE,
        )

        synthetic_img, intensity_info = apply_intensity_augmentation(
            synthetic_img,
            rng,
            enabled=ENABLE_INTENSITY_AUG,
            intensity_scale_range=INTENSITY_SCALE_RANGE,
            intensity_bias_range=INTENSITY_BIAS_RANGE,
            blur_sigma_range=BLUR_SIGMA_RANGE,
            noise_std_range=NOISE_STD_RANGE,
        )

        out_img_path = os.path.join(output_dir, f"{sample_prefix}_{i+1:02d}.nii.gz")
        sitk.WriteImage(synthetic_img, out_img_path)

        case_info = {
            "case_id": i + 1,
            "output_image": out_img_path,
            "rotation_deg": {"x": rot_x, "y": rot_y, "z": rot_z},
            "translation": {"x": tx, "y": ty, "z": tz},
            "translation_fraction": TRANSLATION_FRACTION,
            "flip": {"x": flip_x, "y": flip_y, "z": flip_z},
            "transform_center_physical": center.tolist(),
            "combined_forward_matrix": combined_matrix.tolist(),
            "intensity_augmentation": intensity_info,
        }
        summary["generated_cases"].append(case_info)

        print("=" * 70)
        print(f"Generated synthetic image {i+1}")
        print(f"Saved to: {out_img_path}")
        print(f"Rotation: x={rot_x:.2f}, y={rot_y:.2f}, z={rot_z:.2f}")
        print(f"Translation: x={tx:.3f}, y={ty:.3f}, z={tz:.3f}")
        print(f"Flip: x={flip_x}, y={flip_y}, z={flip_z}")

    out_json_path = os.path.join(output_dir, f"{sample_prefix}_summary.json")
    write_json(out_json_path, summary)

    print("=" * 70)
    print("All synthetic images generated.")
    print("Summary saved to:")
    print(out_json_path)


if __name__ == "__main__":
    main()
