import os
import csv
import json
import math
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import itk
import SimpleITK as sitk


# ============================================================
# Configuration containers
# ============================================================

@dataclass
class ElastixConfig:
    max_iterations: str = "512"
    spatial_samples: str = "8192"
    histogram_bins: str = "32"
    initialization_method: str = "GeometricalCenter"  # or "CenterOfGravity"
    result_image_format: str = "nii"
    log_to_console: bool = False
    log_to_file: bool = True


@dataclass
class BSplineConfig:
    # Conservative default for organoid-scale deformation.
    # Larger value = smoother / less local deformation.
    # Smaller value = more local / more flexible deformation.
    final_grid_spacing_in_physical_units: str = "0.4"
    number_of_resolutions: str = "4"
    bspline_transform_spline_order: str = "3"


@dataclass
class CandidateConfig:
    roll_step_deg: int = 30
    flip_long_axis_options: Tuple[bool, bool] = (False, True)
    moving_pca_mask_percentile: float = 30.0
    pca_max_points: int = 300000
    random_seed: int = 1234


@dataclass
class MetricConfig:
    fixed_mask_percentile: float = 30.0
    moving_mask_percentile: float = 30.0
    min_overlap_ratio: float = 0.01
    hist_bins: int = 64
    score_method: str = "nmi"  # recommended: "nmi"


# ============================================================
# Basic file / image helpers
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_output_root(base_dir: str, sample_id: str, run_name: str) -> str:
    out = os.path.join(base_dir, f"{sample_id}_{run_name}")
    ensure_dir(out)
    return out


def print_image_info(name: str, img_sitk: sitk.Image) -> None:
    print(f"\n[{name}]")
    print(f"  Size      : {img_sitk.GetSize()}")
    print(f"  Spacing   : {img_sitk.GetSpacing()}")
    print(f"  Origin    : {img_sitk.GetOrigin()}")
    print(f"  Direction : {img_sitk.GetDirection()}")


def print_physical_extent(name: str, img_sitk: sitk.Image) -> None:
    size = np.array(img_sitk.GetSize(), dtype=np.float64)
    spacing = np.array(img_sitk.GetSpacing(), dtype=np.float64)
    extent = (size - 1.0) * spacing

    print(f"\n[{name} physical extent estimate]")
    print(f"  extent x/y/z: {extent.tolist()}")


def warn_if_fixed_geometry_mismatch(fixed_search_img: sitk.Image, fixed_final_img: sitk.Image) -> None:
    """
    Low-res and high-res fixed images may have different size/spacing,
    but their origin/direction should normally be consistent.
    """
    origin_search = np.array(fixed_search_img.GetOrigin(), dtype=np.float64)
    origin_final = np.array(fixed_final_img.GetOrigin(), dtype=np.float64)

    direction_search = np.array(fixed_search_img.GetDirection(), dtype=np.float64)
    direction_final = np.array(fixed_final_img.GetDirection(), dtype=np.float64)

    origin_diff = np.linalg.norm(origin_search - origin_final)
    direction_diff = np.linalg.norm(direction_search - direction_final)

    print("\n[Fixed image geometry check]")
    print(f"  origin difference norm    : {origin_diff}")
    print(f"  direction difference norm : {direction_diff}")

    if origin_diff > 1e-3:
        print("  WARNING: low-res and high-res fixed images have different origins.")

    if direction_diff > 1e-6:
        print("  WARNING: low-res and high-res fixed images have different directions.")

    print_physical_extent("Low-res fixed", fixed_search_img)
    print_physical_extent("High-res fixed", fixed_final_img)


def make_lowres_fixed_from_highres(
    fixed_high_img: sitk.Image,
    low_reference_img: sitk.Image,
) -> sitk.Image:
    """
    Generate a low-resolution fixed image from the high-resolution fixed image.

    The low-res file is used only for target size. The generated image keeps
    the high-res origin/direction and recalculates spacing from high-res extent.
    This reduces low/high fixed orientation mismatch.
    """
    out_size = tuple(int(v) for v in low_reference_img.GetSize())

    high_size = np.array(fixed_high_img.GetSize(), dtype=np.float64)
    high_spacing = np.array(fixed_high_img.GetSpacing(), dtype=np.float64)
    high_extent = (high_size - 1.0) * high_spacing

    out_size_arr = np.array(out_size, dtype=np.float64)
    out_spacing = np.zeros(3, dtype=np.float64)

    for i in range(3):
        if out_size_arr[i] > 1:
            out_spacing[i] = high_extent[i] / (out_size_arr[i] - 1.0)
        else:
            out_spacing[i] = high_spacing[i]

    reference = sitk.Image(out_size, sitk.sitkFloat32)
    reference.SetOrigin(fixed_high_img.GetOrigin())
    reference.SetDirection(fixed_high_img.GetDirection())
    reference.SetSpacing(tuple(out_spacing.tolist()))

    low_from_high = sitk.Resample(
        fixed_high_img,
        reference,
        sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )

    return low_from_high


def copy_information_if_same_size(img: sitk.Image, reference_img: sitk.Image) -> sitk.Image:
    if tuple(img.GetSize()) == tuple(reference_img.GetSize()):
        img.CopyInformation(reference_img)
    return img


def sitk_to_itk_float(sitk_img: sitk.Image):
    arr = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
    itk_img = itk.image_view_from_array(arr)

    itk_img.SetOrigin(sitk_img.GetOrigin())
    itk_img.SetSpacing(sitk_img.GetSpacing())
    itk_img.SetDirection(
        itk.matrix_from_array(
            np.array(sitk_img.GetDirection(), dtype=np.float64).reshape(3, 3)
        )
    )

    return itk_img


def itk_to_sitk(itk_img) -> sitk.Image:
    arr = itk.array_view_from_image(itk_img)
    sitk_img = sitk.GetImageFromArray(arr.astype(np.float32))

    sitk_img.SetOrigin(tuple(itk_img.GetOrigin()))
    sitk_img.SetSpacing(tuple(itk_img.GetSpacing()))

    direction_matrix = np.array(itk.array_from_matrix(itk_img.GetDirection()))
    sitk_img.SetDirection(tuple(direction_matrix.flatten()))

    return sitk_img


def write_csv(csv_path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def write_json(json_path: str, obj: Dict[str, Any]) -> None:
    with open(json_path, "w") as f:
        json.dump(obj, f, indent=2)


def copy_if_exists(src: str, dst: str) -> None:
    if src and os.path.exists(src):
        shutil.copy2(src, dst)


def copy_transform_files(src_dir: str, dst_dir: str, prefix: str) -> None:
    """
    Copy elastix transform files and log files to a summary location.

    This is intentionally dynamic because different run modes produce
    different numbers of transform files:
      rigid                 -> TransformParameters.0/1.txt
      rigid_affine          -> TransformParameters.0/1/2.txt
      rigid_bspline         -> TransformParameters.0/1/2.txt
      rigid_affine_bspline  -> TransformParameters.0/1/2/3.txt
    """
    ensure_dir(dst_dir)

    if not os.path.exists(src_dir):
        return

    fnames = []

    transform_fnames = [
        fname for fname in os.listdir(src_dir)
        if fname.startswith("TransformParameters.") and fname.endswith(".txt")
    ]

    def transform_index(fname: str) -> int:
        try:
            return int(fname.split(".")[1])
        except Exception:
            return 9999

    fnames.extend(sorted(transform_fnames, key=transform_index))

    for extra in ["elastix.log", "transformix.log"]:
        if os.path.exists(os.path.join(src_dir, extra)):
            fnames.append(extra)

    for fname in fnames:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, f"{prefix}_{fname}")
        shutil.copy2(src, dst)


def safe_int_angle(value: float) -> int:
    return int(round(float(value)))


# ============================================================
# Elastix parameter setup and registration
# ============================================================

def set_common_parameters(
    parameter_map,
    config: ElastixConfig,
):
    parameter_map["AutomaticTransformInitialization"] = ["true"]
    parameter_map["AutomaticTransformInitializationMethod"] = [config.initialization_method]

    parameter_map["MaximumNumberOfIterations"] = [config.max_iterations]
    parameter_map["NumberOfSpatialSamples"] = [config.spatial_samples]
    parameter_map["NumberOfHistogramBins"] = [config.histogram_bins]

    parameter_map["CheckNumberOfSamples"] = ["false"]
    parameter_map["MaximumNumberOfSamplingAttempts"] = ["1"]

    parameter_map["WriteResultImage"] = ["true"]
    parameter_map["ResultImageFormat"] = [config.result_image_format]
    parameter_map["WriteIterationInfo"] = ["false"]

    return parameter_map



def normalize_metric_weights(pairs: List[Dict[str, Any]]) -> List[float]:
    """
    Normalize metric weights so that their sum is 1.
    This is not required by elastix, but makes experiments easier to compare.
    """
    if not pairs:
        raise ValueError("At least one metric pair is required.")

    weights = [float(p.get("weight", 1.0)) for p in pairs]
    weight_sum = sum(weights)

    if weight_sum <= 0:
        raise ValueError("Sum of metric weights must be > 0.")

    return [w / weight_sum for w in weights]


def set_multimetric_common_parameters(
    parameter_map,
    config: ElastixConfig,
    pairs: List[Dict[str, Any]],
):
    """
    Configure an elastix parameter map for weighted multi-metric registration.

    Each pair contributes one metric to the same transform optimization.
    The first fixed/moving image pair is also the pair whose moving image is
    returned by elastix as the main result image.
    """
    n_metrics = len(pairs)
    metrics = [p.get("metric", "AdvancedMattesMutualInformation") for p in pairs]
    weights = normalize_metric_weights(pairs)

    parameter_map["Registration"] = ["MultiMetricMultiResolutionRegistration"]
    parameter_map["Metric"] = metrics

    parameter_map["Interpolator"] = ["BSplineInterpolator"] * n_metrics
    parameter_map["FixedImagePyramid"] = ["FixedRecursiveImagePyramid"] * n_metrics
    parameter_map["MovingImagePyramid"] = ["MovingRecursiveImagePyramid"] * n_metrics
    parameter_map["ImageSampler"] = ["Random"] * n_metrics

    parameter_map["UseRelativeWeights"] = ["false"]
    for i, weight in enumerate(weights):
        parameter_map[f"Metric{i}Weight"] = [str(weight)]

    parameter_map["AutomaticTransformInitialization"] = ["true"]
    parameter_map["AutomaticTransformInitializationMethod"] = [config.initialization_method]

    parameter_map["MaximumNumberOfIterations"] = [config.max_iterations]
    parameter_map["NumberOfSpatialSamples"] = [config.spatial_samples]
    parameter_map["NumberOfHistogramBins"] = [config.histogram_bins]

    parameter_map["CheckNumberOfSamples"] = ["false"]
    parameter_map["MaximumNumberOfSamplingAttempts"] = ["1"]

    parameter_map["WriteResultImage"] = ["true"]
    parameter_map["ResultImageFormat"] = [config.result_image_format]
    parameter_map["WriteIterationInfo"] = ["false"]

    return parameter_map


def print_multimetric_summary(pairs: List[Dict[str, Any]]) -> None:
    weights = normalize_metric_weights(pairs)
    print("\n[Weighted multi-metric setup]")
    for i, (pair, weight) in enumerate(zip(pairs, weights)):
        print(f"  Metric {i}:")
        print(f"    name   : {pair.get('name', f'pair_{i}')}")
        print(f"    metric : {pair.get('metric', 'AdvancedMattesMutualInformation')}")
        print(f"    weight : {weight:.4f}")

def build_bspline_parameter_map(
    elastix_config: ElastixConfig,
    bspline_config: Optional[BSplineConfig] = None,
    multimetric_pairs: Optional[List[Dict[str, Any]]] = None,
):
    if bspline_config is None:
        bspline_config = BSplineConfig()

    bspline = itk.ParameterObject.New().GetDefaultParameterMap("bspline")

    if multimetric_pairs is None:
        bspline = set_common_parameters(bspline, elastix_config)
        n_metrics = 1
    else:
        bspline = set_multimetric_common_parameters(
            bspline,
            elastix_config,
            multimetric_pairs,
        )
        n_metrics = len(multimetric_pairs)

    bspline["AutomaticTransformInitialization"] = ["false"]

    bspline["Transform"] = ["BSplineTransform"]

    if multimetric_pairs is None:
        bspline["Metric"] = ["AdvancedMattesMutualInformation"]

    bspline["Optimizer"] = ["AdaptiveStochasticGradientDescent"]

    bspline["Interpolator"] = ["BSplineInterpolator"] * n_metrics

    bspline["ResampleInterpolator"] = ["FinalBSplineInterpolator"]
    bspline["Resampler"] = ["DefaultResampler"]

    bspline["FinalGridSpacingInPhysicalUnits"] = [
        str(bspline_config.final_grid_spacing_in_physical_units)
    ]
    bspline["NumberOfResolutions"] = [str(bspline_config.number_of_resolutions)]
    bspline["BSplineTransformSplineOrder"] = [
        str(bspline_config.bspline_transform_spline_order)
    ]

    return bspline


def build_parameter_object(
    run_mode: str,
    config: ElastixConfig,
    bspline_config: Optional[BSplineConfig] = None,
    multimetric_pairs: Optional[List[Dict[str, Any]]] = None,
):
    """
    Supported run_mode:
      - "rigid"                : translation + rigid
      - "rigid_affine"         : translation + rigid + affine
      - "rigid_bspline"        : translation + rigid + B-spline
      - "rigid_affine_bspline" : translation + rigid + affine + B-spline
    """
    parameter_object = itk.ParameterObject.New()

    translation = parameter_object.GetDefaultParameterMap("translation")
    rigid = parameter_object.GetDefaultParameterMap("rigid")
    affine = parameter_object.GetDefaultParameterMap("affine")

    if multimetric_pairs is None:
        translation = set_common_parameters(translation, config)
        rigid = set_common_parameters(rigid, config)
        affine = set_common_parameters(affine, config)
    else:
        translation = set_multimetric_common_parameters(translation, config, multimetric_pairs)
        rigid = set_multimetric_common_parameters(rigid, config, multimetric_pairs)
        affine = set_multimetric_common_parameters(affine, config, multimetric_pairs)

    bspline = build_bspline_parameter_map(
        elastix_config=config,
        bspline_config=bspline_config,
        multimetric_pairs=multimetric_pairs,
    )

    if run_mode == "rigid":
        parameter_object.AddParameterMap(translation)
        parameter_object.AddParameterMap(rigid)

    elif run_mode == "rigid_affine":
        parameter_object.AddParameterMap(translation)
        parameter_object.AddParameterMap(rigid)
        parameter_object.AddParameterMap(affine)

    elif run_mode == "rigid_bspline":
        parameter_object.AddParameterMap(translation)
        parameter_object.AddParameterMap(rigid)
        parameter_object.AddParameterMap(bspline)

    elif run_mode == "rigid_affine_bspline":
        parameter_object.AddParameterMap(translation)
        parameter_object.AddParameterMap(rigid)
        parameter_object.AddParameterMap(affine)
        parameter_object.AddParameterMap(bspline)

    else:
        raise ValueError(
            "run_mode must be one of: "
            "'rigid', 'rigid_affine', 'rigid_bspline', "
            f"'rigid_affine_bspline', got: {run_mode}"
        )

    return parameter_object


def run_elastix_registration(
    fixed_sitk: sitk.Image,
    moving_sitk: sitk.Image,
    output_dir: str,
    run_mode: str,
    config: ElastixConfig,
    final_output_name: Optional[str] = None,
    bspline_config: Optional[BSplineConfig] = None,
):
    """
    Run elastix registration and save a clear final result image.

    Returns:
      result_sitk, final_output_path
    """
    ensure_dir(output_dir)

    fixed_itk = sitk_to_itk_float(fixed_sitk)
    moving_itk = sitk_to_itk_float(moving_sitk)

    parameter_object = build_parameter_object(
        run_mode=run_mode,
        config=config,
        bspline_config=bspline_config,
        multimetric_pairs=None,
    )

    result_image_itk, _ = itk.elastix_registration_method(
        fixed_itk,
        moving_itk,
        parameter_object=parameter_object,
        output_directory=output_dir,
        log_to_console=config.log_to_console,
        log_to_file=config.log_to_file,
    )

    result_sitk = itk_to_sitk(result_image_itk)
    result_sitk = copy_information_if_same_size(result_sitk, fixed_sitk)

    if final_output_name is None:
        final_output_name = f"registered_{run_mode}_result.nii.gz"

    final_output_path = os.path.join(output_dir, final_output_name)
    sitk.WriteImage(result_sitk, final_output_path)

    return result_sitk, final_output_path




def run_elastix_registration_multimetric(
    fixed_sitk_list: List[sitk.Image],
    moving_sitk_list: List[sitk.Image],
    metric_pairs: List[Dict[str, Any]],
    output_dir: str,
    run_mode: str,
    config: ElastixConfig,
    final_output_name: Optional[str] = None,
    bspline_config: Optional[BSplineConfig] = None,
):
    """
    Run weighted multi-metric elastix registration.

    The length/order of fixed_sitk_list, moving_sitk_list, and metric_pairs
    must match. The output image returned by elastix corresponds to the moving
    image of pair 0.

    Returns:
      result_sitk, final_output_path, transform_parameter_object
    """
    ensure_dir(output_dir)

    if not (len(fixed_sitk_list) == len(moving_sitk_list) == len(metric_pairs)):
        raise ValueError(
            "fixed_sitk_list, moving_sitk_list, and metric_pairs must have the same length."
        )
    if len(metric_pairs) < 1:
        raise ValueError("At least one fixed/moving metric pair is required.")

    print_multimetric_summary(metric_pairs)

    fixed_itk_list = [sitk_to_itk_float(img) for img in fixed_sitk_list]
    moving_itk_list = [sitk_to_itk_float(img) for img in moving_sitk_list]

    parameter_object = build_parameter_object(
        run_mode=run_mode,
        config=config,
        bspline_config=bspline_config,
        multimetric_pairs=metric_pairs,
    )

    elastix_object = itk.ElastixRegistrationMethod.New(
        fixed_itk_list[0],
        moving_itk_list[0],
    )

    for i in range(1, len(fixed_itk_list)):
        elastix_object.AddFixedImage(fixed_itk_list[i])
        elastix_object.AddMovingImage(moving_itk_list[i])

    elastix_object.SetParameterObject(parameter_object)
    elastix_object.SetOutputDirectory(output_dir)
    elastix_object.SetLogToConsole(config.log_to_console)
    elastix_object.SetLogToFile(config.log_to_file)
    elastix_object.UpdateLargestPossibleRegion()

    result_image_itk = elastix_object.GetOutput()
    result_transform_parameters = elastix_object.GetTransformParameterObject()

    result_sitk = itk_to_sitk(result_image_itk)
    result_sitk = copy_information_if_same_size(result_sitk, fixed_sitk_list[0])

    if final_output_name is None:
        final_output_name = f"registered_{run_mode}_multimetric_result.nii.gz"

    final_output_path = os.path.join(output_dir, final_output_name)
    sitk.WriteImage(result_sitk, final_output_path)

    return result_sitk, final_output_path, result_transform_parameters


def run_initial_rigid_overlap(
    fixed_path: str,
    moving_path: str,
    output_dir: str,
    sample_id: str,
    config: ElastixConfig,
) -> Dict[str, Any]:
    """
    Step 0:
    Initial rigid overlap.

    This is the first coarse registration used to bring the two objects
    into approximately the same physical position before PCA candidate search.
    """
    ensure_dir(output_dir)

    print("\n============================================================")
    print("Step 0: initial rigid overlap")
    print("============================================================")

    fixed_img = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
    moving_img = sitk.ReadImage(moving_path, sitk.sitkFloat32)

    print_image_info("Initial fixed", fixed_img)
    print_image_info("Initial moving", moving_img)

    result_img, result_path = run_elastix_registration(
        fixed_sitk=fixed_img,
        moving_sitk=moving_img,
        output_dir=output_dir,
        run_mode="rigid",
        config=config,
        final_output_name=f"{sample_id}_INITIAL_RIGID_result.nii.gz",
    )

    copy_transform_files(
        src_dir=output_dir,
        dst_dir=output_dir,
        prefix="INITIAL_RIGID",
    )

    summary = {
        "step": "initial rigid overlap",
        "run_mode": "rigid",
        "fixed_path": fixed_path,
        "moving_path": moving_path,
        "result_path": result_path,
        "output_dir": output_dir,
    }

    write_json(os.path.join(output_dir, "initial_rigid_summary.json"), summary)

    print("\nInitial rigid result saved to:")
    print(result_path)

    return summary


# ============================================================
# PCA candidate transform generation
# ============================================================

def image_points_to_physical(points_xyz: np.ndarray, img: sitk.Image) -> np.ndarray:
    """
    Convert N x 3 voxel indices in SimpleITK x,y,z order to physical coordinates.
    """
    points_xyz = np.asarray(points_xyz, dtype=np.float64)

    origin = np.asarray(img.GetOrigin(), dtype=np.float64)
    spacing = np.asarray(img.GetSpacing(), dtype=np.float64)
    direction = np.asarray(img.GetDirection(), dtype=np.float64).reshape(3, 3)

    continuous = points_xyz * spacing[None, :]
    physical = origin[None, :] + continuous @ direction.T

    return physical


def extract_foreground_points_physical(
    img: sitk.Image,
    percentile: float = 30.0,
    max_points: int = 300000,
    random_seed: int = 1234,
):
    """
    Extract foreground points for moving-image PCA.

    Foreground:
      non-zero finite voxels above percentile(non-zero values)
    """
    arr = sitk.GetArrayViewFromImage(img).astype(np.float32)  # z,y,x
    finite = np.isfinite(arr)

    values = arr[finite]
    nz = values[np.abs(values) > 0]

    if nz.size == 0:
        raise RuntimeError("PCA failed: moving image has no non-zero finite voxels.")

    thresh = float(np.percentile(nz, percentile))
    mask = finite & (arr > thresh)

    coords_zyx = np.argwhere(mask)

    if coords_zyx.shape[0] < 10:
        raise RuntimeError(
            f"PCA failed: too few foreground voxels after threshold. "
            f"threshold={thresh}, n={coords_zyx.shape[0]}"
        )

    if coords_zyx.shape[0] > max_points:
        rng = np.random.default_rng(random_seed)
        selected = rng.choice(coords_zyx.shape[0], size=max_points, replace=False)
        coords_zyx = coords_zyx[selected]

    coords_xyz = coords_zyx[:, [2, 1, 0]].astype(np.float64)
    points_physical = image_points_to_physical(coords_xyz, img)

    return points_physical, thresh, int(mask.sum())


def compute_moving_pca_long_axis(
    img: sitk.Image,
    config: CandidateConfig,
) -> Dict[str, Any]:
    """
    PCA on moving foreground physical coordinates.

    Returns:
      center, long_axis, eigenvalues, eigenvectors.
    """
    points, thresh, n_foreground = extract_foreground_points_physical(
        img,
        percentile=config.moving_pca_mask_percentile,
        max_points=config.pca_max_points,
        random_seed=config.random_seed,
    )

    center = points.mean(axis=0)
    centered = points - center[None, :]

    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    long_axis = eigenvectors[:, 0].astype(np.float64)
    long_axis = long_axis / (np.linalg.norm(long_axis) + 1e-12)

    return {
        "center": center,
        "long_axis": long_axis,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "threshold": thresh,
        "n_foreground_voxels": n_foreground,
        "n_points_used": int(points.shape[0]),
    }


def rotation_matrix_around_axis(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)

    x, y, z = axis
    angle = math.radians(angle_deg)

    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c

    R = np.array([
        [c + x * x * C,      x * y * C - z * s,  x * z * C + y * s],
        [y * x * C + z * s,  c + y * y * C,      y * z * C - x * s],
        [z * x * C - y * s,  z * y * C + x * s,  c + z * z * C],
    ], dtype=np.float64)

    return R


def flip_matrix_along_axis(axis: np.ndarray) -> np.ndarray:
    """
    Flip along the long axis:
      parallel component becomes negative,
      perpendicular plane stays unchanged.

    Matrix:
      F = I - 2 * a * a^T
    """
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)

    return np.eye(3, dtype=np.float64) - 2.0 * np.outer(axis, axis)


def create_centered_affine_from_matrix(center: np.ndarray, matrix: np.ndarray) -> sitk.AffineTransform:
    tx = sitk.AffineTransform(3)
    tx.SetCenter(tuple(np.asarray(center, dtype=np.float64).tolist()))
    tx.SetMatrix(np.asarray(matrix, dtype=np.float64).flatten().tolist())
    tx.SetTranslation((0.0, 0.0, 0.0))
    return tx


def compose_candidate_transform(
    center: np.ndarray,
    long_axis: np.ndarray,
    flip_long_axis: bool = False,
    roll_deg: float = 0.0,
) -> sitk.AffineTransform:
    """
    Candidate transform around the moving-image PCA centroid:
      1) optional flip along the moving long axis
      2) roll rotation around the same long axis
    """
    R = rotation_matrix_around_axis(long_axis, roll_deg)

    if flip_long_axis:
        F = flip_matrix_along_axis(long_axis)
        M = R @ F
    else:
        M = R

    return create_centered_affine_from_matrix(center, M)


def resample_candidate_to_fixed_space(
    moving_img: sitk.Image,
    fixed_img: sitk.Image,
    transform: sitk.Transform,
) -> sitk.Image:
    return sitk.Resample(
        moving_img,
        fixed_img,
        transform,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )


def get_roll_angles(config: CandidateConfig) -> List[int]:
    return list(range(0, 360, int(config.roll_step_deg)))


# ============================================================
# Metric functions
# ============================================================

def make_foreground_mask(img: sitk.Image, percentile: float = 30.0) -> sitk.Image:
    arr = sitk.GetArrayViewFromImage(img).astype(np.float32)

    positive = arr[np.isfinite(arr)]

    if positive.size == 0:
        return sitk.Image(img.GetSize(), sitk.sitkUInt8)

    nz = positive[np.abs(positive) > 0]

    if nz.size > 0:
        thresh = np.percentile(nz, percentile)
    else:
        thresh = np.percentile(positive, percentile)

    return sitk.Cast(img > float(thresh), sitk.sitkUInt8)


def compute_overlap_ratio(fixed_mask: sitk.Image, moving_mask: sitk.Image) -> float:
    overlap = sitk.And(fixed_mask, moving_mask)

    overlap_sum = float(sitk.GetArrayViewFromImage(overlap).sum())
    fixed_sum = float(sitk.GetArrayViewFromImage(fixed_mask).sum())

    if fixed_sum <= 0:
        return 0.0

    return overlap_sum / fixed_sum


def compute_ncc(fixed_img: sitk.Image, moving_img: sitk.Image, mask: Optional[sitk.Image] = None) -> float:
    f = sitk.GetArrayViewFromImage(fixed_img).astype(np.float64)
    m = sitk.GetArrayViewFromImage(moving_img).astype(np.float64)

    if mask is not None:
        mk = sitk.GetArrayViewFromImage(mask).astype(bool)
        f = f[mk]
        m = m[mk]

    valid = np.isfinite(f) & np.isfinite(m)

    f = f[valid]
    m = m[valid]

    if f.size < 10 or m.size < 10:
        return float("nan")

    f_mean = f.mean()
    m_mean = m.mean()

    f_std = f.std()
    m_std = m.std()

    if f_std < 1e-12 or m_std < 1e-12:
        return float("nan")

    ncc = np.mean(((f - f_mean) / f_std) * ((m - m_mean) / m_std))
    return float(ncc)


def compute_mse(fixed_img: sitk.Image, moving_img: sitk.Image, mask: Optional[sitk.Image] = None) -> float:
    f = sitk.GetArrayViewFromImage(fixed_img).astype(np.float64)
    m = sitk.GetArrayViewFromImage(moving_img).astype(np.float64)

    if mask is not None:
        mk = sitk.GetArrayViewFromImage(mask).astype(bool)
        f = f[mk]
        m = m[mk]

    valid = np.isfinite(f) & np.isfinite(m)

    f = f[valid]
    m = m[valid]

    if f.size < 10:
        return float("nan")

    return float(np.mean((f - m) ** 2))


def compute_mi_hist(
    fixed_img: sitk.Image,
    moving_img: sitk.Image,
    mask: Optional[sitk.Image] = None,
    bins: int = 64,
) -> float:
    f = sitk.GetArrayViewFromImage(fixed_img).astype(np.float64)
    m = sitk.GetArrayViewFromImage(moving_img).astype(np.float64)

    if mask is not None:
        mk = sitk.GetArrayViewFromImage(mask).astype(bool)
        f = f[mk]
        m = m[mk]

    valid = np.isfinite(f) & np.isfinite(m)

    f = f[valid]
    m = m[valid]

    if f.size < 100:
        return float("nan")

    f_min, f_max = np.percentile(f, [1, 99])
    m_min, m_max = np.percentile(m, [1, 99])

    if abs(f_max - f_min) < 1e-12 or abs(m_max - m_min) < 1e-12:
        return float("nan")

    f = np.clip(f, f_min, f_max)
    m = np.clip(m, m_min, m_max)

    joint_hist, _, _ = np.histogram2d(f, m, bins=bins)

    if np.sum(joint_hist) <= 0:
        return float("nan")

    pxy = joint_hist / np.sum(joint_hist)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = np.outer(px, py)

    nz = pxy > 0

    mi = np.sum(
        pxy[nz] *
        np.log((pxy[nz] + 1e-12) / (px_py[nz] + 1e-12))
    )

    return float(mi)


def compute_nmi_hist(
    fixed_img: sitk.Image,
    moving_img: sitk.Image,
    mask: Optional[sitk.Image] = None,
    bins: int = 64,
) -> float:
    """
    Histogram-based normalized mutual information:
      NMI = 2 * MI / (H_fixed + H_moving)

    Larger is better.
    """
    f = sitk.GetArrayViewFromImage(fixed_img).astype(np.float64)
    m = sitk.GetArrayViewFromImage(moving_img).astype(np.float64)

    if mask is not None:
        mk = sitk.GetArrayViewFromImage(mask).astype(bool)
        f = f[mk]
        m = m[mk]

    valid = np.isfinite(f) & np.isfinite(m)

    f = f[valid]
    m = m[valid]

    if f.size < 100:
        return float("nan")

    f_min, f_max = np.percentile(f, [1, 99])
    m_min, m_max = np.percentile(m, [1, 99])

    if abs(f_max - f_min) < 1e-12 or abs(m_max - m_min) < 1e-12:
        return float("nan")

    f = np.clip(f, f_min, f_max)
    m = np.clip(m, m_min, m_max)

    joint_hist, _, _ = np.histogram2d(f, m, bins=bins)

    if np.sum(joint_hist) <= 0:
        return float("nan")

    pxy = joint_hist / np.sum(joint_hist)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    nz_joint = pxy > 0
    nz_x = px > 0
    nz_y = py > 0

    h_fixed = -np.sum(px[nz_x] * np.log(px[nz_x] + 1e-12))
    h_moving = -np.sum(py[nz_y] * np.log(py[nz_y] + 1e-12))

    px_py = np.outer(px, py)

    mi = np.sum(
        pxy[nz_joint] *
        np.log((pxy[nz_joint] + 1e-12) / (px_py[nz_joint] + 1e-12))
    )

    denom = h_fixed + h_moving

    if denom < 1e-12:
        return float("nan")

    return float(2.0 * mi / denom)


def compute_metrics(
    fixed_img: sitk.Image,
    registered_img: sitk.Image,
    config: MetricConfig,
) -> Dict[str, float]:
    fixed_mask = make_foreground_mask(
        fixed_img,
        percentile=config.fixed_mask_percentile,
    )

    moving_mask = make_foreground_mask(
        registered_img,
        percentile=config.moving_mask_percentile,
    )

    score_mask = sitk.And(fixed_mask, moving_mask)
    overlap_ratio = compute_overlap_ratio(fixed_mask, moving_mask)

    if overlap_ratio < config.min_overlap_ratio:
        return {
            "mi": float("nan"),
            "nmi": float("nan"),
            "ncc": float("nan"),
            "mse": float("nan"),
            "overlap_ratio": overlap_ratio,
        }

    mi = compute_mi_hist(
        fixed_img,
        registered_img,
        mask=score_mask,
        bins=config.hist_bins,
    )

    nmi = compute_nmi_hist(
        fixed_img,
        registered_img,
        mask=score_mask,
        bins=config.hist_bins,
    )

    ncc = compute_ncc(
        fixed_img,
        registered_img,
        mask=score_mask,
    )

    mse = compute_mse(
        fixed_img,
        registered_img,
        mask=score_mask,
    )

    return {
        "mi": mi,
        "nmi": nmi,
        "ncc": ncc,
        "mse": mse,
        "overlap_ratio": overlap_ratio,
    }


# ============================================================
# Elastix transform parameter file helpers
# ============================================================

def read_transform_parameter_file(path: str) -> Dict[str, str]:
    params = {}

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("(") and line.endswith(")"):
                content = line[1:-1]
                parts = content.split(None, 1)
                if len(parts) == 2:
                    key, value = parts
                    params[key] = value

    return params


def write_transform_parameter_file(params: Dict[str, str], path: str) -> None:
    with open(path, "w") as f:
        for key, value in params.items():
            f.write(f"({key} {value})\n")


def update_transform_for_high_res(
    src_path: str,
    dst_path: str,
    high_res_fixed_img: sitk.Image,
) -> str:
    """
    Copy one elastix transform parameter file and change its output grid
    to the high-res fixed geometry.
    """
    params = read_transform_parameter_file(src_path)

    size = high_res_fixed_img.GetSize()
    spacing = high_res_fixed_img.GetSpacing()
    origin = high_res_fixed_img.GetOrigin()
    direction = high_res_fixed_img.GetDirection()

    params["Size"] = " ".join(str(int(s)) for s in size)
    params["Spacing"] = " ".join(str(float(s)) for s in spacing)
    params["Origin"] = " ".join(str(float(o)) for o in origin)
    params["Direction"] = " ".join(str(float(d)) for d in direction)

    params["ResultImagePixelType"] = '"float"'
    params["ResultImageFormat"] = '"nii"'

    write_transform_parameter_file(params, dst_path)

    return dst_path


def update_stage1_transform_chain_for_high_res(
    stage1_reg_dir: str,
    dst_dir: str,
    high_res_fixed_img: sitk.Image,
) -> str:
    """
    Copy the complete Stage 1 low-res rigid transform chain to high-res geometry.

    For run_mode="rigid", elastix writes:
      TransformParameters.0.txt = translation
      TransformParameters.1.txt = rigid

    The returned file is the final rigid transform file, which points to
    the translation file as its initial transform.
    """
    ensure_dir(dst_dir)

    src0 = os.path.join(stage1_reg_dir, "TransformParameters.0.txt")
    src1 = os.path.join(stage1_reg_dir, "TransformParameters.1.txt")

    if not os.path.exists(src0):
        raise RuntimeError(f"Missing Stage 1 translation transform: {src0}")

    if not os.path.exists(src1):
        raise RuntimeError(f"Missing Stage 1 rigid transform: {src1}")

    dst0 = os.path.join(dst_dir, "Stage1_highres_TransformParameters.0.txt")
    dst1 = os.path.join(dst_dir, "Stage1_highres_TransformParameters.1.txt")

    update_transform_for_high_res(src0, dst0, high_res_fixed_img)
    update_transform_for_high_res(src1, dst1, high_res_fixed_img)

    params0 = read_transform_parameter_file(dst0)
    params0["InitialTransformParametersFileName"] = '"NoInitialTransform"'
    write_transform_parameter_file(params0, dst0)

    params1 = read_transform_parameter_file(dst1)
    params1["InitialTransformParametersFileName"] = f'"{os.path.abspath(dst0)}"'
    write_transform_parameter_file(params1, dst1)

    return dst1


def apply_transform_with_transformix(
    moving_sitk: sitk.Image,
    transform_param_path: str,
    output_dir: str,
    config: ElastixConfig,
    reference_sitk: Optional[sitk.Image] = None,
) -> sitk.Image:
    """
    Apply an elastix transform parameter file to a moving image using transformix.
    """
    ensure_dir(output_dir)

    moving_itk = sitk_to_itk_float(moving_sitk)

    parameter_object = itk.ParameterObject.New()
    parameter_object.ReadParameterFile(transform_param_path)

    result_itk = itk.transformix_filter(
        moving_itk,
        parameter_object,
        output_directory=output_dir,
        log_to_console=config.log_to_console,
        log_to_file=config.log_to_file,
    )

    result_sitk = itk_to_sitk(result_itk)

    if reference_sitk is not None:
        result_sitk = copy_information_if_same_size(result_sitk, reference_sitk)

    return result_sitk


# ============================================================
# Apply full pipeline transform to extra images
# ============================================================

def get_last_transform_parameter_file(transform_dir: str) -> str:
    """
    Return the last TransformParameters.*.txt in an elastix output folder.

    For elastix transform chains, the last TransformParameters file normally
    points to the previous ones through InitialTransformParametersFileName.
    """
    if not os.path.exists(transform_dir):
        raise RuntimeError(f"Transform directory does not exist: {transform_dir}")

    transform_files = [
        fname for fname in os.listdir(transform_dir)
        if fname.startswith("TransformParameters.") and fname.endswith(".txt")
    ]

    if not transform_files:
        raise RuntimeError(f"No TransformParameters.*.txt found in: {transform_dir}")

    def _transform_index(fname: str) -> int:
        try:
            return int(fname.split(".")[1])
        except Exception:
            return 9999

    transform_files = sorted(transform_files, key=_transform_index)

    return os.path.join(transform_dir, transform_files[-1])


def make_nearest_neighbor_transform_file(
    src_transform_path: str,
    dst_transform_path: str,
) -> str:
    """
    Make a copy of an elastix TransformParameters file for mask/label resampling.

    Use this for:
      - segmentation masks
      - binary masks
      - label images

    Do not use this for scalar maps such as FA, MD, S0, density map.
    """
    params = read_transform_parameter_file(src_transform_path)

    params["ResampleInterpolator"] = '"FinalNearestNeighborInterpolator"'
    params["FinalBSplineInterpolationOrder"] = "0"

    write_transform_parameter_file(params, dst_transform_path)

    return dst_transform_path


def apply_full_pipeline_to_extra_image(
    raw_extra_image_path: str,
    output_dir: str,
    fixed_final_path: str,
    initial_rigid_transform_path: str,
    low_search_summary: Dict[str, Any],
    final_summary: Dict[str, Any],
    elastix_config: ElastixConfig,
    interpolation: str = "linear",
    output_name: Optional[str] = None,
    already_after_initial_rigid: bool = False,
) -> str:
    """
    Apply the same full registration pipeline to a new extra image.

    Full chain:
        raw extra image
        -> same Step 0 initial rigid
        -> same best PCA flip/roll candidate
        -> same Stage 1 high-res rigid transform
        -> same Stage 2 final transform
        -> fixed high-res microscopy space

    Parameters
    ----------
    raw_extra_image_path:
        Image that should be transformed, for example MD, S0, MO, mask, density map.

    output_dir:
        Output folder for intermediate and final results.

    fixed_final_path:
        Final fixed image path, usually case2_Topro3_high.nii.gz.

    initial_rigid_transform_path:
        Step 0 initial rigid transform.
        Usually:
            INITIAL_RIGID_OVERLAP/TransformParameters.1.txt

    low_search_summary:
        Dictionary loaded from best_candidate_LOWRES.json.

    final_summary:
        Dictionary loaded from final_summary_LOWSEARCH_HIGHFINAL.json.

    interpolation:
        "linear"  for scalar maps: FA, MD, S0, density map, probability map.
        "nearest" for masks / labels / segmentations.

    already_after_initial_rigid:
        Set True only if raw_extra_image_path has already been transformed by Step 0.

    Returns
    -------
    final_output_path:
        Path to the extra image transformed into fixed high-res microscopy space.
    """
    ensure_dir(output_dir)

    if interpolation not in ("linear", "nearest"):
        raise ValueError("interpolation must be 'linear' or 'nearest'.")

    fixed_final_img = sitk.ReadImage(fixed_final_path, sitk.sitkFloat32)
    extra_img = sitk.ReadImage(raw_extra_image_path, sitk.sitkFloat32)

    base = os.path.basename(raw_extra_image_path)
    base = base.replace(".nii.gz", "").replace(".nii", "")

    if output_name is None:
        output_name = f"{base}_FULL_PIPELINE_TO_FIXED.nii.gz"

    # ------------------------------------------------------------
    # Step 0: apply same initial rigid transform
    # ------------------------------------------------------------
    if not already_after_initial_rigid:
        if not initial_rigid_transform_path or not os.path.exists(initial_rigid_transform_path):
            raise RuntimeError(
                f"Initial rigid transform file not found: {initial_rigid_transform_path}"
            )

        step0_dir = os.path.join(output_dir, "01_after_initial_rigid")
        ensure_dir(step0_dir)

        transform_to_use = initial_rigid_transform_path

        if interpolation == "nearest":
            transform_to_use = make_nearest_neighbor_transform_file(
                src_transform_path=initial_rigid_transform_path,
                dst_transform_path=os.path.join(
                    step0_dir,
                    "InitialRigid_TransformParameters.1_NEAREST.txt",
                ),
            )

        extra_img = apply_transform_with_transformix(
            moving_sitk=extra_img,
            transform_param_path=transform_to_use,
            output_dir=step0_dir,
            config=elastix_config,
            reference_sitk=fixed_final_img,
        )

        step0_path = os.path.join(
            step0_dir,
            f"{base}_after_initial_rigid.nii.gz",
        )
        sitk.WriteImage(extra_img, step0_path)
        print(f"[apply full pipeline] Step 0 saved: {step0_path}")

    # ------------------------------------------------------------
    # Step 1: apply same best PCA flip/roll candidate
    # ------------------------------------------------------------
    pca_info = low_search_summary["moving_pca"]

    moving_pca_center = np.array(pca_info["center"], dtype=np.float64)
    moving_long_axis = np.array(pca_info["long_axis"], dtype=np.float64)

    best_low = low_search_summary["best_lowres_candidate"]

    best_flip = bool(best_low["flip_long_axis"])
    best_roll = float(best_low["roll_deg"])

    best_candidate_transform = compose_candidate_transform(
        center=moving_pca_center,
        long_axis=moving_long_axis,
        flip_long_axis=best_flip,
        roll_deg=best_roll,
    )

    if interpolation == "nearest":
        candidate_interpolator = sitk.sitkNearestNeighbor
        output_pixel_type = extra_img.GetPixelID()
    else:
        candidate_interpolator = sitk.sitkLinear
        output_pixel_type = sitk.sitkFloat32

    candidate_img = sitk.Resample(
        extra_img,
        fixed_final_img,
        best_candidate_transform,
        candidate_interpolator,
        0.0,
        output_pixel_type,
    )

    candidate_dir = os.path.join(output_dir, "02_after_best_candidate")
    ensure_dir(candidate_dir)

    candidate_path = os.path.join(
        candidate_dir,
        f"{base}_after_best_candidate_HIGHRES.nii.gz",
    )
    sitk.WriteImage(candidate_img, candidate_path)
    print(f"[apply full pipeline] candidate saved: {candidate_path}")

    moving_for_final = candidate_img

    # ------------------------------------------------------------
    # Step 2: apply same Stage 1 high-res rigid transform
    # ------------------------------------------------------------
    use_stage1 = bool(final_summary.get("use_stage1_rigid_as_highres_initialization", False))
    stage1_transform_path = final_summary.get("stage1_highres_transform_file", "")

    if use_stage1:
        if not stage1_transform_path or not os.path.exists(stage1_transform_path):
            raise RuntimeError(
                "Stage 1 high-res transform file is missing. "
                "Expected final_summary['stage1_highres_transform_file']."
            )

        stage1_dir = os.path.join(output_dir, "03_after_stage1_highres")
        ensure_dir(stage1_dir)

        transform_to_use = stage1_transform_path

        if interpolation == "nearest":
            transform_to_use = make_nearest_neighbor_transform_file(
                src_transform_path=stage1_transform_path,
                dst_transform_path=os.path.join(
                    stage1_dir,
                    "Stage1_highres_TransformParameters.1_NEAREST.txt",
                ),
            )

        moving_for_final = apply_transform_with_transformix(
            moving_sitk=moving_for_final,
            transform_param_path=transform_to_use,
            output_dir=stage1_dir,
            config=elastix_config,
            reference_sitk=fixed_final_img,
        )

        stage1_path = os.path.join(
            stage1_dir,
            f"{base}_after_best_candidate_plus_stage1_HIGHRES.nii.gz",
        )
        sitk.WriteImage(moving_for_final, stage1_path)
        print(f"[apply full pipeline] Stage 1 saved: {stage1_path}")

    # ------------------------------------------------------------
    # Step 3: apply same final Stage 2 transform
    # ------------------------------------------------------------
    final_registration_dir = final_summary.get("final_registration_dir", "")

    if not final_registration_dir:
        raise RuntimeError("final_summary does not contain 'final_registration_dir'.")

    final_transform_path = get_last_transform_parameter_file(final_registration_dir)

    final_dir = os.path.join(output_dir, "04_after_final_transform")
    ensure_dir(final_dir)

    transform_to_use = final_transform_path

    if interpolation == "nearest":
        transform_to_use = make_nearest_neighbor_transform_file(
            src_transform_path=final_transform_path,
            dst_transform_path=os.path.join(
                final_dir,
                os.path.basename(final_transform_path).replace(".txt", "_NEAREST.txt"),
            ),
        )

    final_img = apply_transform_with_transformix(
        moving_sitk=moving_for_final,
        transform_param_path=transform_to_use,
        output_dir=final_dir,
        config=elastix_config,
        reference_sitk=fixed_final_img,
    )

    final_output_path = os.path.join(output_dir, output_name)
    sitk.WriteImage(final_img, final_output_path)

    print(f"[apply full pipeline] final saved: {final_output_path}")

    return final_output_path
# ============================================================
# Stage 1: low-res PCA candidate search
# ============================================================

def run_lowres_candidate_search(
    moving_path: str,
    fixed_search_path: str,
    fixed_final_path: str,
    output_root: str,
    sample_id: str,
    candidate_config: CandidateConfig,
    elastix_config: ElastixConfig,
    metric_config: MetricConfig,
    use_highres_derived_low_fixed: bool = True,
    low_search_run_mode: str = "rigid",
) -> Dict[str, Any]:
    """
    Stage 1:
      1) PCA on the initially rigid-aligned moving image.
      2) Generate 24 candidates:
           flip=False/True x roll=0..330 step 30
      3) Resample each candidate to the low-res fixed grid.
      4) Run low-res elastix rigid registration.
      5) Compute MI/NMI/NCC/MSE/overlap and select by NMI.
    """
    if low_search_run_mode != "rigid":
        raise ValueError("low_search_run_mode must be 'rigid' for candidate search.")

    low_search_root = os.path.join(output_root, "LOW_RES_SEARCH")
    candidates_root = os.path.join(low_search_root, "candidates")
    registrations_root = os.path.join(low_search_root, "registrations")

    ensure_dir(low_search_root)
    ensure_dir(candidates_root)
    ensure_dir(registrations_root)

    print("\n============================================================")
    print("Stage 1: low-resolution PCA candidate search")
    print("============================================================")

    fixed_search_native_img = sitk.ReadImage(fixed_search_path, sitk.sitkFloat32)
    fixed_final_img = sitk.ReadImage(fixed_final_path, sitk.sitkFloat32)
    moving_img = sitk.ReadImage(moving_path, sitk.sitkFloat32)

    if use_highres_derived_low_fixed:
        print("\nGenerating low-res fixed from high-res fixed to keep orientation consistent...")
        fixed_search_img = make_lowres_fixed_from_highres(
            fixed_high_img=fixed_final_img,
            low_reference_img=fixed_search_native_img,
        )
        generated_low_path = os.path.join(
            low_search_root,
            "fixed_search_LOWRES_generated_from_highres.nii.gz",
        )
        sitk.WriteImage(fixed_search_img, generated_low_path)
        print(f"  generated low-res fixed: {generated_low_path}")
    else:
        fixed_search_img = fixed_search_native_img
        generated_low_path = ""

    print_image_info("Fixed search low-res native", fixed_search_native_img)
    print_image_info("Fixed search low-res used", fixed_search_img)
    print_image_info("Fixed final high-res", fixed_final_img)
    print_image_info("Moving after initial rigid", moving_img)

    warn_if_fixed_geometry_mismatch(fixed_search_img, fixed_final_img)

    pca_info = compute_moving_pca_long_axis(
        moving_img,
        config=candidate_config,
    )

    moving_pca_center = pca_info["center"]
    moving_long_axis = pca_info["long_axis"]

    print("\nMoving PCA long-axis information:")
    print(f"  PCA mask percentile      : {candidate_config.moving_pca_mask_percentile}")
    print(f"  PCA threshold            : {pca_info['threshold']}")
    print(f"  foreground voxels        : {pca_info['n_foreground_voxels']}")
    print(f"  points used for PCA      : {pca_info['n_points_used']}")
    print(f"  PCA center physical      : {moving_pca_center.tolist()}")
    print(f"  long axis physical vector: {moving_long_axis.tolist()}")
    print(f"  eigenvalues              : {pca_info['eigenvalues'].tolist()}")

    results = []
    roll_angles_deg = get_roll_angles(candidate_config)

    for flip_flag in candidate_config.flip_long_axis_options:
        for roll_deg in roll_angles_deg:
            candidate_name = f"flipLong{int(flip_flag)}_roll{int(roll_deg):03d}"

            print(f"\n=== Low-res candidate: {candidate_name} ===")

            candidate_dir = os.path.join(candidates_root, candidate_name)
            reg_dir = os.path.join(registrations_root, candidate_name)

            ensure_dir(candidate_dir)
            ensure_dir(reg_dir)

            candidate_transform = compose_candidate_transform(
                center=moving_pca_center,
                long_axis=moving_long_axis,
                flip_long_axis=flip_flag,
                roll_deg=roll_deg,
            )

            candidate_img_low = resample_candidate_to_fixed_space(
                moving_img=moving_img,
                fixed_img=fixed_search_img,
                transform=candidate_transform,
            )

            candidate_img_low_path = os.path.join(
                candidate_dir,
                f"{candidate_name}_LOWRES_resampled_to_fixed.nii.gz",
            )
            sitk.WriteImage(candidate_img_low, candidate_img_low_path)

            try:
                registered_img_low, registered_path_low = run_elastix_registration(
                    fixed_sitk=fixed_search_img,
                    moving_sitk=candidate_img_low,
                    output_dir=reg_dir,
                    run_mode=low_search_run_mode,
                    config=elastix_config,
                    final_output_name="registered_rigid_result.nii.gz",
                )

                metrics_low = compute_metrics(
                    fixed_search_img,
                    registered_img_low,
                    config=metric_config,
                )

                row = {
                    "candidate_name": candidate_name,
                    "flip_long_axis": int(flip_flag),
                    "roll_deg": float(roll_deg),

                    "pca_center_x": float(moving_pca_center[0]),
                    "pca_center_y": float(moving_pca_center[1]),
                    "pca_center_z": float(moving_pca_center[2]),

                    "long_axis_x": float(moving_long_axis[0]),
                    "long_axis_y": float(moving_long_axis[1]),
                    "long_axis_z": float(moving_long_axis[2]),

                    "candidate_image_lowres": candidate_img_low_path,
                    "registered_image_lowres": registered_path_low,
                    "low_search_run_mode": low_search_run_mode,

                    "mi": metrics_low["mi"],
                    "nmi": metrics_low["nmi"],
                    "ncc": metrics_low["ncc"],
                    "mse": metrics_low["mse"],
                    "overlap_ratio": metrics_low["overlap_ratio"],

                    "registration_dir_lowres": reg_dir,
                    "status": "success",
                }

            except Exception as e:
                row = {
                    "candidate_name": candidate_name,
                    "flip_long_axis": int(flip_flag),
                    "roll_deg": float(roll_deg),

                    "pca_center_x": float(moving_pca_center[0]),
                    "pca_center_y": float(moving_pca_center[1]),
                    "pca_center_z": float(moving_pca_center[2]),

                    "long_axis_x": float(moving_long_axis[0]),
                    "long_axis_y": float(moving_long_axis[1]),
                    "long_axis_z": float(moving_long_axis[2]),

                    "candidate_image_lowres": candidate_img_low_path,
                    "registered_image_lowres": "",
                    "low_search_run_mode": low_search_run_mode,

                    "mi": float("nan"),
                    "nmi": float("nan"),
                    "ncc": float("nan"),
                    "mse": float("nan"),
                    "overlap_ratio": 0.0,

                    "registration_dir_lowres": reg_dir,
                    "status": f"failed: {str(e)}",
                }

            results.append(row)

    valid_indices = []

    for i, r in enumerate(results):
        score_value = r.get(metric_config.score_method, float("nan"))
        if (
            r["status"] == "success"
            and np.isfinite(score_value)
            and r["overlap_ratio"] >= metric_config.min_overlap_ratio
        ):
            valid_indices.append(i)

    csv_path = os.path.join(output_root, "candidate_scores_LOWRES.csv")

    if len(valid_indices) == 0:
        write_csv(csv_path, results)
        raise RuntimeError("No valid low-resolution registration result found.")

    for i, r in enumerate(results):
        if i in valid_indices:
            results[i]["combined_score"] = float(r[metric_config.score_method])
            results[i]["score_method"] = metric_config.score_method
        else:
            results[i]["combined_score"] = float("-inf")
            results[i]["score_method"] = metric_config.score_method

    results_sorted = sorted(
        results,
        key=lambda x: x["combined_score"],
        reverse=True,
    )

    best_low = results_sorted[0]
    write_csv(csv_path, results_sorted)

    low_predicted_native_name = f"{sample_id}_{low_search_run_mode}_LOWRES_predicted_native_lowgrid.nii.gz"
    low_predicted_native_path = os.path.join(output_root, low_predicted_native_name)

    copy_if_exists(
        best_low["registered_image_lowres"],
        low_predicted_native_path,
    )

    copy_transform_files(
        src_dir=best_low["registration_dir_lowres"],
        dst_dir=output_root,
        prefix="LOWRES_predicted",
    )

    low_summary = {
        "sample_id": sample_id,
        "stage": "stage_1_low_resolution_candidate_search",
        "low_search_run_mode": low_search_run_mode,
        "use_highres_derived_low_fixed": use_highres_derived_low_fixed,

        "fixed_search_path": fixed_search_path,
        "fixed_search_generated_from_highres": generated_low_path,
        "fixed_final_path": fixed_final_path,
        "moving_after_initial_rigid_path": moving_path,

        "roll_step_deg": candidate_config.roll_step_deg,
        "roll_angles_deg": roll_angles_deg,
        "flip_long_axis_options": list(candidate_config.flip_long_axis_options),

        "moving_pca": {
            "mask_percentile": candidate_config.moving_pca_mask_percentile,
            "threshold": float(pca_info["threshold"]),
            "n_foreground_voxels": int(pca_info["n_foreground_voxels"]),
            "n_points_used": int(pca_info["n_points_used"]),
            "center": pca_info["center"].tolist(),
            "long_axis": pca_info["long_axis"].tolist(),
            "eigenvalues": pca_info["eigenvalues"].tolist(),
            "eigenvectors": pca_info["eigenvectors"].tolist(),
        },

        "score_method": metric_config.score_method,
        "score_definition": f"combined_score = {metric_config.score_method}",
        "candidate_scores_csv": csv_path,
        "lowres_predicted_native_lowgrid_image": low_predicted_native_path,
        "best_lowres_candidate": best_low,
        "fixed_search_img_used_size": fixed_search_img.GetSize(),
        "fixed_search_img_used_spacing": fixed_search_img.GetSpacing(),
    }

    low_json_path = os.path.join(output_root, "best_candidate_LOWRES.json")
    write_json(low_json_path, low_summary)

    print("\nBest low-resolution candidate:")
    print(f"  candidate_name : {best_low['candidate_name']}")
    print(f"  flip_long_axis : {best_low['flip_long_axis']}")
    print(f"  roll_deg       : {best_low['roll_deg']}")
    print(f"  NMI            : {best_low['nmi']}")
    print(f"  MI             : {best_low['mi']}")
    print(f"  NCC            : {best_low['ncc']}")
    print(f"  MSE            : {best_low['mse']}")
    print(f"  overlap_ratio  : {best_low['overlap_ratio']}")
    print(f"  combined_score : {best_low['combined_score']}")

    low_summary["best_candidate_json"] = low_json_path

    return low_summary


# ============================================================
# Stage 2: high-res final registration
# ============================================================

def run_highres_final_registration(
    moving_path: str,
    fixed_final_path: str,
    output_root: str,
    sample_id: str,
    low_search_summary: Dict[str, Any],
    candidate_config: CandidateConfig,
    elastix_config: ElastixConfig,
    metric_config: MetricConfig,
    final_run_mode: str = "rigid_affine",
    use_stage1_rigid_as_highres_initialization: bool = True,
    bspline_config: Optional[BSplineConfig] = None,
    final_multimetric_pairs: Optional[List[Dict[str, Any]]] = None,
    primary_metric: str = "AdvancedMattesMutualInformation",
    primary_weight: float = 1.0,
) -> Dict[str, Any]:
    """
    Stage 2:
      1) Use the best flip/roll angle selected from Stage 1.
      2) Apply that candidate transform to the initially rigid-aligned moving image
         directly on the high-res fixed grid.
      3) Optionally apply the complete Stage 1 low-res translation+rigid transform
         chain after converting its output geometry to the high-res grid.
      4) Run final high-res elastix registration.
         Supported nonlinear modes include rigid_bspline and rigid_affine_bspline.
    """
    final_high_root = os.path.join(output_root, "FINAL_HIGH_RES")
    ensure_dir(final_high_root)

    print("\n============================================================")
    print(f"Stage 2: high-resolution final registration ({final_run_mode})")
    print("============================================================")

    fixed_final_img = sitk.ReadImage(fixed_final_path, sitk.sitkFloat32)
    moving_img = sitk.ReadImage(moving_path, sitk.sitkFloat32)

    pca_info = low_search_summary["moving_pca"]
    moving_pca_center = np.array(pca_info["center"], dtype=np.float64)
    moving_long_axis = np.array(pca_info["long_axis"], dtype=np.float64)

    best_low = low_search_summary["best_lowres_candidate"]
    best_flip = bool(best_low["flip_long_axis"])
    best_roll = float(best_low["roll_deg"])

    best_candidate_transform = compose_candidate_transform(
        center=moving_pca_center,
        long_axis=moving_long_axis,
        flip_long_axis=best_flip,
        roll_deg=best_roll,
    )

    best_candidate_high = resample_candidate_to_fixed_space(
        moving_img=moving_img,
        fixed_img=fixed_final_img,
        transform=best_candidate_transform,
    )

    best_candidate_high_path = os.path.join(
        final_high_root,
        f"best_candidate_HIGHRES_flipLong{int(best_flip)}_roll{safe_int_angle(best_roll):03d}.nii.gz",
    )
    sitk.WriteImage(best_candidate_high, best_candidate_high_path)

    moving_for_final = best_candidate_high
    stage1_applied_path = ""
    highres_stage1_final_transform = ""

    if use_stage1_rigid_as_highres_initialization:
        stage1_reg_dir = best_low["registration_dir_lowres"]

        highres_stage1_final_transform = update_stage1_transform_chain_for_high_res(
            stage1_reg_dir=stage1_reg_dir,
            dst_dir=final_high_root,
            high_res_fixed_img=fixed_final_img,
        )

        print("\nStage 1 transform chain copied to high-res geometry:")
        print(f"  final Stage 1 high-res transform: {highres_stage1_final_transform}")

        moving_for_final = apply_transform_with_transformix(
            moving_sitk=best_candidate_high,
            transform_param_path=highres_stage1_final_transform,
            output_dir=final_high_root,
            config=elastix_config,
            reference_sitk=fixed_final_img,
        )

        stage1_applied_path = os.path.join(
            final_high_root,
            "moving_after_best_candidate_plus_stage1_rigid_HIGHRES.nii.gz",
        )
        sitk.WriteImage(moving_for_final, stage1_applied_path)

        low_predicted_highgrid_name = f"{sample_id}_rigid_LOWRES_predicted_highgrid.nii.gz"
        low_predicted_highgrid_path = os.path.join(output_root, low_predicted_highgrid_name)
        sitk.WriteImage(moving_for_final, low_predicted_highgrid_path)

        print(f"\nStage 1 best result applied in high-res space: {stage1_applied_path}")
        print(f"High-grid LOWRES predicted copy          : {low_predicted_highgrid_path}")

    else:
        low_predicted_highgrid_path = ""

    # --------------------------------------------------------
    # Final elastix registration: single metric or weighted multi-metric
    # --------------------------------------------------------
    multimetric_init_images = []

    if final_multimetric_pairs:
        multimetric_root = os.path.join(final_high_root, "MULTIMETRIC_INITIALIZED_IMAGES")
        ensure_dir(multimetric_root)

        metric_pairs = [
            {
                "name": "primary_stage2_pair",
                "metric": primary_metric,
                "weight": primary_weight,
            }
        ]
        fixed_sitk_list = [fixed_final_img]
        moving_sitk_list = [moving_for_final]

        for i, pair in enumerate(final_multimetric_pairs, start=1):
            pair_name = pair.get("name", f"extra_pair_{i}")
            pair_fixed_path = pair.get("fixed_path", fixed_final_path)
            pair_moving_path = pair.get("moving_path", "")

            if not pair_moving_path:
                raise ValueError(
                    f"final_multimetric_pairs[{i-1}] is missing 'moving_path'. "
                    "This should normally be the extra moving image after the same Step 0 initial rigid space."
                )

            pair_fixed_img = sitk.ReadImage(pair_fixed_path, sitk.sitkFloat32)
            pair_moving_img = sitk.ReadImage(pair_moving_path, sitk.sitkFloat32)

            # Apply the same selected PCA flip/roll candidate to every extra moving image.
            pair_candidate_high = resample_candidate_to_fixed_space(
                moving_img=pair_moving_img,
                fixed_img=fixed_final_img,
                transform=best_candidate_transform,
            )

            pair_candidate_path = os.path.join(
                multimetric_root,
                f"{pair_name}_best_candidate_HIGHRES.nii.gz",
            )
            sitk.WriteImage(pair_candidate_high, pair_candidate_path)

            pair_moving_for_final = pair_candidate_high
            pair_stage1_path = ""

            # Apply the same Stage 1 low-res rigid transform chain in high-res geometry.
            if use_stage1_rigid_as_highres_initialization:
                pair_moving_for_final = apply_transform_with_transformix(
                    moving_sitk=pair_candidate_high,
                    transform_param_path=highres_stage1_final_transform,
                    output_dir=multimetric_root,
                    config=elastix_config,
                    reference_sitk=fixed_final_img,
                )

                pair_stage1_path = os.path.join(
                    multimetric_root,
                    f"{pair_name}_after_best_candidate_plus_stage1_rigid_HIGHRES.nii.gz",
                )
                sitk.WriteImage(pair_moving_for_final, pair_stage1_path)

            metric_pairs.append(
                {
                    "name": pair_name,
                    "metric": pair.get("metric", "AdvancedMattesMutualInformation"),
                    "weight": float(pair.get("weight", 1.0)),
                    "fixed_path": pair_fixed_path,
                    "moving_path": pair_moving_path,
                    "candidate_highres_image": pair_candidate_path,
                    "stage1_highres_image": pair_stage1_path,
                }
            )
            fixed_sitk_list.append(pair_fixed_img)
            moving_sitk_list.append(pair_moving_for_final)

            multimetric_init_images.append(
                {
                    "name": pair_name,
                    "fixed_path": pair_fixed_path,
                    "moving_path": pair_moving_path,
                    "candidate_highres_image": pair_candidate_path,
                    "stage1_highres_image": pair_stage1_path,
                }
            )

        final_registered_img, final_registered_path, final_transform_parameters = run_elastix_registration_multimetric(
            fixed_sitk_list=fixed_sitk_list,
            moving_sitk_list=moving_sitk_list,
            metric_pairs=metric_pairs,
            output_dir=final_high_root,
            run_mode=final_run_mode,
            config=elastix_config,
            final_output_name=f"registered_{final_run_mode}_weighted_multimetric_result.nii.gz",
            bspline_config=bspline_config,
        )

        # Save registered versions of all moving images after the same final transform.
        registered_multi_dir = os.path.join(final_high_root, "registered_multimetric_images")
        ensure_dir(registered_multi_dir)
        for pair, moving_init_img in zip(metric_pairs, moving_sitk_list):
            registered_itk = itk.transformix_filter(
                sitk_to_itk_float(moving_init_img),
                final_transform_parameters,
            )
            registered_sitk = itk_to_sitk(registered_itk)
            registered_sitk = copy_information_if_same_size(registered_sitk, fixed_final_img)
            out_path = os.path.join(
                registered_multi_dir,
                f"{pair.get('name', 'pair')}_registered_by_final_transform.nii.gz",
            )
            sitk.WriteImage(registered_sitk, out_path)

    else:
        metric_pairs = []
        final_registered_img, final_registered_path = run_elastix_registration(
            fixed_sitk=fixed_final_img,
            moving_sitk=moving_for_final,
            output_dir=final_high_root,
            run_mode=final_run_mode,
            config=elastix_config,
            final_output_name=f"registered_{final_run_mode}_result.nii.gz",
            bspline_config=bspline_config,
        )

    final_metrics = compute_metrics(
        fixed_final_img,
        final_registered_img,
        config=metric_config,
    )

    final_predicted_name = f"{sample_id}_{final_run_mode}_FINAL_HIGHRES_predicted.nii.gz"
    final_predicted_path = os.path.join(output_root, final_predicted_name)

    copy_if_exists(
        final_registered_path,
        final_predicted_path,
    )

    copy_transform_files(
        src_dir=final_high_root,
        dst_dir=output_root,
        prefix="FINAL_HIGHRES_predicted",
    )

    final_summary = {
        "sample_id": sample_id,
        "stage": "stage_2_high_resolution_final_registration",
        "final_run_mode": final_run_mode,
        "use_weighted_multimetric_final": bool(final_multimetric_pairs),
        "final_multimetric_pairs": metric_pairs,
        "multimetric_initialized_images": multimetric_init_images,
        "bspline_config": (
            None if bspline_config is None else {
                "final_grid_spacing_in_physical_units": bspline_config.final_grid_spacing_in_physical_units,
                "number_of_resolutions": bspline_config.number_of_resolutions,
                "bspline_transform_spline_order": bspline_config.bspline_transform_spline_order,
            }
        ),
        "fixed_final_path": fixed_final_path,
        "moving_after_initial_rigid_path": moving_path,

        "best_flip_long_axis": int(best_flip),
        "best_roll_deg": float(best_roll),
        "best_candidate_highres_image": best_candidate_high_path,

        "use_stage1_rigid_as_highres_initialization": use_stage1_rigid_as_highres_initialization,
        "stage1_highres_transform_file": highres_stage1_final_transform,
        "moving_after_stage1_highres_image": stage1_applied_path,
        "lowres_predicted_highgrid_image": low_predicted_highgrid_path,

        "final_registered_image": final_registered_path,
        "final_predicted_image": final_predicted_path,
        "final_metrics": final_metrics,
        "final_registration_dir": final_high_root,

        "note": (
            "The low-resolution fixed image is used only for candidate angle selection. "
            "Stage 1 candidate search always uses rigid registration. "
            "Stage 2 applies the selected flip/roll to the moving image on the high-res fixed grid, "
            "then runs final high-resolution registration. Nonlinear final modes use elastix B-spline. If "
            "use_stage1_rigid_as_highres_initialization=True, the complete Stage 1 translation+rigid "
            "transform chain is also applied in high-res geometry before final registration."
        ),
    }

    final_json_path = os.path.join(output_root, "final_summary_LOWSEARCH_HIGHFINAL.json")
    final_summary["final_summary_json"] = final_json_path
    write_json(final_json_path, final_summary)

    print("\nSaved high-resolution final outputs:")
    print(f"  high-res candidate image       : {best_candidate_high_path}")
    if stage1_applied_path:
        print(f"  Stage 1 applied high-res image : {stage1_applied_path}")
    print(f"  final registered image         : {final_registered_path}")
    print(f"  final predicted image          : {final_predicted_path}")
    print(f"  final summary JSON             : {final_json_path}")

    print("\nFinal high-resolution metrics:")
    print(f"  MI            : {final_metrics['mi']}")
    print(f"  NMI           : {final_metrics['nmi']}")
    print(f"  NCC           : {final_metrics['ncc']}")
    print(f"  MSE           : {final_metrics['mse']}")
    print(f"  overlap_ratio : {final_metrics['overlap_ratio']}")

    return final_summary
