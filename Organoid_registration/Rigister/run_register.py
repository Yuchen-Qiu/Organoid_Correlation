#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import shutil
import numpy as np
import SimpleITK as sitk

from functions_weighted import (
    ElastixConfig,
    BSplineConfig,
    CandidateConfig,
    MetricConfig,
    ensure_dir,
    get_output_root,
    run_initial_rigid_overlap,
    run_lowres_candidate_search,
    run_highres_final_registration,
    write_json,
    apply_transform_with_transformix,
    compose_candidate_transform,
    read_transform_parameter_file,
    write_transform_parameter_file,
)

# ============================================================
# Configuration
# ============================================================

ROOT = "/autofs/arch11/DATA/HOMES/yuchen/Minor/Organoids/Nifti_data"
CASE_DIR = os.path.join(ROOT, "case2_young")
DATA = os.path.join(CASE_DIR, "data_final")
B_DIR = os.path.join(DATA, "data_final_bvalue")

# ------------------------------------------------------------
# Current registration experiment
# Edit ONLY this section when switching FA / MD / FA+MD / b-images.
# ------------------------------------------------------------

sample_id_base = "case2_FA"

fixed_search_path = os.path.join(DATA, "case2_Topro3_low.nii")
fixed_final_path = os.path.join(DATA, "case2_Topro3_high.nii")
initial_fixed_path = fixed_final_path

# Primary moving image.
raw_moving_path = os.path.join(DATA, "dti_FA_final_LAS_eroded_1vox.nii.gz")
primary_metric = "AdvancedMattesMutualInformation"
primary_weight = 2.0

# Optional extra moving images for weighted multi-metric registration.
# Keep [] for single-image registration.

# extra_multimetric_raw_pairs = [
#     # {
#     #     "name": f"topro3_vs_b{i:03d}",
#     #     "fixed_path": fixed_final_path,
#     #     "raw_moving_path": os.path.join(
#     #         B_DIR,
#     #         f"case2_bvalue_{i:03d}.nii.gz"
#     #     ),
#     #     "metric": "AdvancedMattesMutualInformation",
#     #     "weight": 1.0,
#     # }
#     # for i in range(0-9)
# ]

extra_multimetric_raw_pairs = [
    # {
    #     "name": "topro3_vs_md",
    #     "fixed_path": fixed_final_path,
    #     "raw_moving_path": os.path.join(DATA, "dti_MD_final_LAS_eroded_1vox.nii.gz"),
    #     "metric": "AdvancedMattesMutualInformation",
    #     "weight": 1.0,
    # },
]

# Examples:
#
# FA only:
# sample_id_base = "case2_FA"
# raw_moving_path = os.path.join(DATA, "dti_FA_fixed.nii.gz")
# extra_multimetric_raw_pairs = []
#
# MD only:
# sample_id_base = "case2_MD"
# raw_moving_path = os.path.join(DATA, "dti_MD_fixed.nii.gz")
# extra_multimetric_raw_pairs = []
#
# FA + MD:
# sample_id_base = "case2_FA_MD"
# raw_moving_path = os.path.join(DATA, "dti_FA_fixed.nii.gz")
# extra_multimetric_raw_pairs = [
#     {
#         "name": "topro3_vs_md",
#         "fixed_path": fixed_final_path,
#         "raw_moving_path": os.path.join(DATA, "dti_MD_fixed.nii.gz"),
#         "metric": "AdvancedMattesMutualInformation",
#         "weight": 1.0,
#     },
# ]
#
# b0 + ... + b9:
# sample_id_base = "case2_B_IMAGES"
# raw_moving_path = os.path.join(B_DIR, "case2_bvalue_000.nii.gz")
# extra_multimetric_raw_pairs = [
#     {
#         "name": f"topro3_vs_b{i}",
#         "fixed_path": fixed_final_path,
#         "raw_moving_path": os.path.join(B_DIR, f"case2_bvalue_{i:03d}.nii.gz"),
#         "metric": "AdvancedMattesMutualInformation",
#         "weight": 1.0,
#     }
#     for i in range(1, 10)
# ]

BASE_OUTPUT_DIR = os.path.join(ROOT, "registration_native_pipeline")

RUN_INITIAL_RIGID = True
LOW_SEARCH_RUN_MODE = "rigid"
FINAL_RUN_MODE = "rigid_affine"
USE_STAGE1_RIGID_AS_HIGHRES_INITIALIZATION = True
USE_HIGHRES_DERIVED_LOW_FIXED = False
RUN_COMPOSITE_QC = True



elastix_config = ElastixConfig(
    max_iterations="512",
    spatial_samples="8192",
    histogram_bins="32",
    initialization_method="GeometricalCenter",
    result_image_format="nii",
    log_to_console=False,
    log_to_file=True,
)

bspline_config = BSplineConfig(
    final_grid_spacing_in_physical_units="0.4",
    number_of_resolutions="4",
    bspline_transform_spline_order="3",
)

candidate_config = CandidateConfig(
    roll_step_deg=30,
    flip_long_axis_options=(False, True),
    moving_pca_mask_percentile=30.0,
    pca_max_points=300000,
    random_seed=1234,
)

metric_config = MetricConfig(
    fixed_mask_percentile=30.0,
    moving_mask_percentile=30.0,
    min_overlap_ratio=0.01,
    hist_bins=64,
    score_method="nmi",
)

# ============================================================
# Helpers
# ============================================================

def exact_transform_files(folder):
    if not os.path.isdir(folder):
        raise RuntimeError(f"Transform folder not found: {folder}")
    files = []
    for name in os.listdir(folder):
        m = re.fullmatch(r"TransformParameters\.(\d+)\.txt", name)
        if m:
            files.append((int(m.group(1)), os.path.join(folder, name)))
    files.sort(key=lambda x: x[0])
    if not files:
        raise RuntimeError(f"No TransformParameters.N.txt found in: {folder}")
    return [p for _, p in files]

def set_initial_transform(params, previous_path):
    params.pop("InitialTransformParametersFileName", None)
    params["InitialTransformParameterFileName"] = (
        '"NoInitialTransform"' if previous_path is None
        else f'"{os.path.abspath(previous_path)}"'
    )
    params["HowToCombineTransforms"] = '"Compose"'

def set_output_geometry(params, reference_img):
    size = reference_img.GetSize()
    spacing = reference_img.GetSpacing()
    origin = reference_img.GetOrigin()
    direction = np.asarray(reference_img.GetDirection(), dtype=np.float64).reshape(3, 3)
    params["Size"] = " ".join(str(int(v)) for v in size)
    params["Index"] = "0 0 0"
    params["Spacing"] = " ".join(f"{float(v):.17g}" for v in spacing)
    params["Origin"] = " ".join(f"{float(v):.17g}" for v in origin)
    params["Direction"] = " ".join(f"{float(v):.17g}" for v in direction.flatten(order="F"))
    params["UseDirectionCosines"] = '"true"'
    params["ResultImagePixelType"] = '"float"'
    params["ResultImageFormat"] = '"nii"'

def copy_relinked_chain(src_files, dst_dir, prefix, previous_path, reference_img):
    ensure_dir(dst_dir)
    last_path = previous_path
    copied = []
    for i, src in enumerate(src_files):
        dst = os.path.join(dst_dir, f"{prefix}_{i:02d}.txt")
        params = read_transform_parameter_file(src)
        set_initial_transform(params, last_path)
        set_output_geometry(params, reference_img)
        write_transform_parameter_file(params, dst)
        copied.append(dst)
        last_path = dst
    return last_path, copied

def write_candidate_affine_transform(dst_path, candidate_tx, previous_path, reference_img):
    matrix = np.asarray(candidate_tx.GetMatrix(), dtype=np.float64)
    center = np.asarray(candidate_tx.GetCenter(), dtype=np.float64)
    translation = np.asarray(candidate_tx.GetTranslation(), dtype=np.float64)
    size = reference_img.GetSize()
    spacing = reference_img.GetSpacing()
    origin = reference_img.GetOrigin()
    direction = np.asarray(reference_img.GetDirection(), dtype=np.float64).reshape(3, 3)
    transform_parameters = np.concatenate([matrix, translation])
    params = {
        "Transform": '"AffineTransform"',
        "NumberOfParameters": "12",
        "TransformParameters": " ".join(f"{float(v):.17g}" for v in transform_parameters),
        "InitialTransformParameterFileName": (
            '"NoInitialTransform"' if previous_path is None
            else f'"{os.path.abspath(previous_path)}"'
        ),
        "HowToCombineTransforms": '"Compose"',
        "FixedImageDimension": "3",
        "MovingImageDimension": "3",
        "FixedInternalImagePixelType": '"float"',
        "MovingInternalImagePixelType": '"float"',
        "Size": " ".join(str(int(v)) for v in size),
        "Index": "0 0 0",
        "Spacing": " ".join(f"{float(v):.17g}" for v in spacing),
        "Origin": " ".join(f"{float(v):.17g}" for v in origin),
        "Direction": " ".join(f"{float(v):.17g}" for v in direction.flatten(order="F")),
        "UseDirectionCosines": '"true"',
        "CenterOfRotationPoint": " ".join(f"{float(v):.17g}" for v in center),
        "ResampleInterpolator": '"FinalLinearInterpolator"',
        "FinalBSplineInterpolationOrder": "1",
        "Resampler": '"DefaultResampler"',
        "DefaultPixelValue": "0",
        "ResultImageFormat": '"nii"',
        "ResultImagePixelType": '"float"',
        "CompressResultImage": '"false"',
    }
    write_transform_parameter_file(params, dst_path)
    return dst_path

def build_full_composite_chain(
    output_root,
    initial_rigid_dir,
    low_search_summary,
    final_summary,
    reference_img,
):
    if not RUN_INITIAL_RIGID:
        raise RuntimeError(
            "Full original-image composite export currently requires RUN_INITIAL_RIGID=True."
        )

    composite_dir = os.path.join(output_root, "COMPOSITE_TRANSFORM")
    if os.path.isdir(composite_dir):
        shutil.rmtree(composite_dir)
    ensure_dir(composite_dir)

    # IMPORTANT:
    # Each resampling transform maps output/fixed coordinates -> input/moving coordinates.
    # Sequential image generation was:
    #   original --T0--> step0 image --C--> candidate --T1--> stage1 --T2--> final
    # Therefore the single mapping from final/output space back to the ORIGINAL image is:
    #   T_total = T0 o C o T1 o T2
    # Elastix "Compose" evaluates current(initial(x)), so the transform files must be
    # linked from INNER to OUTER in the reverse stage order:
    #   T2 -> T1 -> C -> T0.
    previous = None

    final_files = exact_transform_files(final_summary["final_registration_dir"])
    previous, final_copied = copy_relinked_chain(
        final_files, composite_dir, "00_final_inner", previous, reference_img
    )

    best_low = low_search_summary["best_lowres_candidate"]

    stage1_files = exact_transform_files(best_low["registration_dir_lowres"])
    previous, stage1_copied = copy_relinked_chain(
        stage1_files, composite_dir, "01_stage1", previous, reference_img
    )

    pca_info = low_search_summary["moving_pca"]
    candidate_tx = compose_candidate_transform(
        center=np.asarray(pca_info["center"], dtype=np.float64),
        long_axis=np.asarray(pca_info["long_axis"], dtype=np.float64),
        flip_long_axis=bool(best_low["flip_long_axis"]),
        roll_deg=float(best_low["roll_deg"]),
    )
    pca_path = os.path.join(composite_dir, "02_pca_candidate.txt")
    previous = write_candidate_affine_transform(
        pca_path, candidate_tx, previous, reference_img
    )

    step0_files = exact_transform_files(initial_rigid_dir)
    previous, step0_copied = copy_relinked_chain(
        step0_files, composite_dir, "03_step0_outer", previous, reference_img
    )

    manifest = {
        "final_composite_transform": previous,
        "step0_files": step0_copied,
        "pca_transform": pca_path,
        "stage1_files": stage1_copied,
        "final_files": final_copied,
        "mapping_formula": "T_total = T_step0 o T_pca o T_stage1 o T_final",
        "chain_link_order_inner_to_outer": [
            "Stage2 final registration",
            "Stage1 low-res rigid",
            "PCA flip/roll",
            "Step0 initial rigid",
        ],
    }
    write_json(os.path.join(composite_dir, "composite_manifest.json"), manifest)
    return manifest

def prepare_extra_pairs_after_initial_rigid(
    output_root,
    initial_rigid_dir,
):
    if not extra_multimetric_raw_pairs:
        return []

    prepared_root = os.path.join(output_root, "EXTRA_MOVING_AFTER_INITIAL_RIGID")
    ensure_dir(prepared_root)

    if not RUN_INITIAL_RIGID:
        return [
            {
                "name": p["name"],
                "fixed_path": p.get("fixed_path", fixed_final_path),
                "moving_path": p["raw_moving_path"],
                "raw_moving_path": p["raw_moving_path"],
                "metric": "AdvancedMattesMutualInformation",
                "weight": float(p.get("weight", 1.0)),
            }
            for p in extra_multimetric_raw_pairs
        ]

    initial_transform_path = os.path.join(initial_rigid_dir, "TransformParameters.1.txt")
    if not os.path.exists(initial_transform_path):
        raise RuntimeError(f"Missing initial rigid transform: {initial_transform_path}")

    reference_img = sitk.ReadImage(initial_fixed_path, sitk.sitkFloat32)
    prepared = []

    for pair in extra_multimetric_raw_pairs:
        raw_img = sitk.ReadImage(pair["raw_moving_path"], sitk.sitkFloat32)
        pair_dir = os.path.join(prepared_root, pair["name"])
        ensure_dir(pair_dir)

        moved_img = apply_transform_with_transformix(
            moving_sitk=raw_img,
            transform_param_path=initial_transform_path,
            output_dir=pair_dir,
            config=elastix_config,
            reference_sitk=reference_img,
        )

        moved_path = os.path.join(pair_dir, f"{pair['name']}_after_INITIAL_RIGID.nii.gz")
        sitk.WriteImage(moved_img, moved_path)

        prepared.append(
            {
                "name": pair["name"],
                "fixed_path": pair.get("fixed_path", fixed_final_path),
                "moving_path": moved_path,
                "raw_moving_path": pair["raw_moving_path"],
                "metric": pair.get("metric", "AdvancedMattesMutualInformation"),
                "weight": float(pair.get("weight", 1.0)),
            }
        )

    return prepared

def run_composite_qc(raw_moving_path, composite_transform_path, fixed_path, output_root):
    if not RUN_COMPOSITE_QC:
        return ""

    qc_dir = os.path.join(output_root, "COMPOSITE_QC")
    ensure_dir(qc_dir)
    raw_img = sitk.ReadImage(raw_moving_path, sitk.sitkFloat32)
    fixed_img = sitk.ReadImage(fixed_path, sitk.sitkFloat32)

    qc_img = apply_transform_with_transformix(
        moving_sitk=raw_img,
        transform_param_path=composite_transform_path,
        output_dir=qc_dir,
        config=elastix_config,
        reference_sitk=fixed_img,
    )

    qc_path = os.path.join(qc_dir, "primary_original_transformed_once.nii.gz")
    sitk.WriteImage(qc_img, qc_path)
    return qc_path

# ============================================================
# One registration run
# ============================================================

def run_one_experiment():
    primary_path = raw_moving_path
    sample_id = sample_id_base
    output_root = get_output_root(
        base_dir=BASE_OUTPUT_DIR,
        sample_id=sample_id,
        run_name=FINAL_RUN_MODE,
    )
    ensure_dir(output_root)

    print("\n============================================================")
    print(f"Experiment: {sample_id}")
    print("============================================================")
    print(f"Primary: {primary_path}")
    print(f"Extra metrics: {len(extra_multimetric_raw_pairs)}")
    print(f"Output: {output_root}")

    if RUN_INITIAL_RIGID:
        initial_rigid_dir = os.path.join(output_root, "INITIAL_RIGID_OVERLAP")
        initial_summary = run_initial_rigid_overlap(
            fixed_path=initial_fixed_path,
            moving_path=primary_path,
            output_dir=initial_rigid_dir,
            sample_id=sample_id,
            config=elastix_config,
        )
        moving_for_registration = initial_summary["result_path"]
    else:
        raise RuntimeError("This version expects RUN_INITIAL_RIGID=True.")

    extra_pairs = prepare_extra_pairs_after_initial_rigid(
        output_root=output_root,
        initial_rigid_dir=initial_rigid_dir,
    )

    low_search_summary = run_lowres_candidate_search(
        moving_path=moving_for_registration,
        fixed_search_path=fixed_search_path,
        fixed_final_path=fixed_final_path,
        output_root=output_root,
        sample_id=sample_id,
        candidate_config=candidate_config,
        elastix_config=elastix_config,
        metric_config=metric_config,
        use_highres_derived_low_fixed=USE_HIGHRES_DERIVED_LOW_FIXED,
        low_search_run_mode=LOW_SEARCH_RUN_MODE,
    )

    final_summary = run_highres_final_registration(
        moving_path=moving_for_registration,
        fixed_final_path=fixed_final_path,
        output_root=output_root,
        sample_id=sample_id,
        low_search_summary=low_search_summary,
        candidate_config=candidate_config,
        elastix_config=elastix_config,
        metric_config=metric_config,
        final_run_mode=FINAL_RUN_MODE,
        use_stage1_rigid_as_highres_initialization=USE_STAGE1_RIGID_AS_HIGHRES_INITIALIZATION,
        bspline_config=bspline_config,
        final_multimetric_pairs=extra_pairs if extra_pairs else None,
        primary_metric=primary_metric,
        primary_weight=primary_weight,
    )

    fixed_img = sitk.ReadImage(fixed_final_path, sitk.sitkFloat32)
    composite_manifest = build_full_composite_chain(
        output_root=output_root,
        initial_rigid_dir=initial_rigid_dir,
        low_search_summary=low_search_summary,
        final_summary=final_summary,
        reference_img=fixed_img,
    )

    qc_path = run_composite_qc(
        raw_moving_path=primary_path,
        composite_transform_path=composite_manifest["final_composite_transform"],
        fixed_path=fixed_final_path,
        output_root=output_root,
    )

    summary = {
        "experiment": sample_id,
        "sample_id": sample_id,
        "primary_path": primary_path,
        "extra_pairs": extra_multimetric_raw_pairs,
        "initial_rigid": initial_summary,
        "low_search": low_search_summary,
        "final_registration": final_summary,
        "composite_transform": composite_manifest,
        "composite_qc_image": qc_path,
    }
    write_json(os.path.join(output_root, "pipeline_summary_with_composite.json"), summary)

    print("\nFinal composite transform:")
    print(composite_manifest["final_composite_transform"])
    if qc_path:
        print("Composite single-resample QC:")
        print(qc_path)

    return summary

# ============================================================
# Main
# ============================================================

def main():
    required_paths = [
        initial_fixed_path,
        fixed_search_path,
        fixed_final_path,
        raw_moving_path,
    ] + [p["raw_moving_path"] for p in extra_multimetric_raw_pairs]

    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    ensure_dir(BASE_OUTPUT_DIR)
    result = run_one_experiment()

    print("\n============================================================")
    print("Registration finished.")
    print("============================================================")
    print(f"Sample ID: {sample_id_base}")
    print(
        "Composite transform: "
        f"{result['composite_transform']['final_composite_transform']}"
    )

if __name__ == "__main__":
    main()
