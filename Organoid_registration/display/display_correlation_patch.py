#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nibabel.processing import resample_from_to
from scipy.stats import pearsonr, spearmanr, t


# ============================================================
# User settings
# ============================================================

WORK_DIR = r"/autofs/arch11/DATA/HOMES/yuchen/Minor/Organoids/Nifti_data/case1_mature/data_final"

CENTER_CSV_PATH = os.path.join(WORK_DIR, "component_centers_organoid.csv")
MICROSCOPY_REFERENCE_PATH = os.path.join(WORK_DIR, "case1_Topro3_origin.nii")

MRI_METRICS = {
    # "FA": {
    #     "path": os.path.join(WORK_DIR, "registered_Addition_native_Bvalue/registered_masked/case2_FA_final_LAS_registered_native_masked.nii.gz"),
    #     "ylabel": "Mean registered FA",
    # },
    # "MD": {
    #     "path": os.path.join(WORK_DIR, "registered_Addition_native_Bvalue/registered_masked/case2_MD_final_LAS_registered_native_masked.nii.gz"),
    #     "ylabel": "Mean registered MD",
    # },
    # "S0": {
    #     "path": os.path.join(WORK_DIR, "registered_Addition_native_Bvalue/registered_masked/case2_S0_final_LAS_registered_native_masked.nii.gz"),
    #     "ylabel": "Mean registered S0",
    # },

    "ADC": {
        "path": os.path.join(WORK_DIR, "registered_Addition_native_Bvalue/registered_masked/case1_ADC_registered_native_masked.nii.gz"),
        "ylabel": "Mean registered ADC",
    },
    "S0": {
        "path": os.path.join(WORK_DIR, "registered_Addition_native_Bvalue/registered_masked/case1_S0_registered_native_masked.nii.gz"),
        "ylabel": "Mean registered S0",
    },
    "Kurtosis": {
        "path": os.path.join(WORK_DIR, "registered_Addition_native_Bvalue/registered_masked/case1_kurtosis_registered_native_masked.nii.gz"),
        "ylabel": "Mean registered Kurtosis",
    },
}

ENABLE_DMRI_RESAMPLING = False
TARGET_DMRI_SPACING_UM = (44.79, 44.79, 44.79)
DMRI_INTERPOLATION_ORDER = 1
SAVE_RESAMPLED_DMRI = True
SAVE_MAPPED_MICROSCOPY = True

DENSITY_UNIT_VOLUME_UM3 = 8000

CSV_WORLD_UNIT = "um"
MICROSCOPY_WORLD_UNIT = "um"
DMRI_WORLD_UNIT = "mm"

EXCLUDE_ZERO_DENSITY = False

ENABLE_VALUE_RANGE_EXCLUSION = False
X_EXCLUDE_RANGE = (0, 1)
Y_EXCLUDE_RANGES = {
    "ADC": None,
    "S0": None,
    "Kurtosis": None,
}

SAVE_FILTERED_ORIGINAL_VOLUMES = True
SAVE_LOW_DENSITY_VOLUME = True
FILTERED_VOXEL_VALUE = 0.0

HEXBIN_GRIDSIZE = 60

ENABLE_SLAB_ANALYSIS = True
SLAB_PLANE = "axial"  # "axial", "coronal", or "sagittal"
SLAB_RANGE_MM = (0.6, 1.1)
SAVE_ANALYSIS_SLAB_MASK = True

if ENABLE_DMRI_RESAMPLING:
    SPACING_TAG = "x".join(
        str(int(v)) if float(v).is_integer() else str(v)
        for v in TARGET_DMRI_SPACING_UM
    )
    OUTPUT_DIR = os.path.join(WORK_DIR, f"correlation_results_erosion2voxel__crop_resampled_{SPACING_TAG}um")
else:
    OUTPUT_DIR = os.path.join(WORK_DIR, "correlation_results_erosion2voxel_crop_native_voxel")

if ENABLE_SLAB_ANALYSIS:
    slab_min_mm, slab_max_mm = SLAB_RANGE_MM
    slab_tag = f"{SLAB_PLANE}_{slab_min_mm:g}-{slab_max_mm:g}mm"
    OUTPUT_DIR = OUTPUT_DIR + "_" + slab_tag

os.makedirs(OUTPUT_DIR, exist_ok=True)

RESAMPLED_DIR = os.path.join(OUTPUT_DIR, "resampled_dmri")
OUTPUT_STATS_CSV = os.path.join(OUTPUT_DIR, "density_correlation_summary.csv")


# ============================================================
# Helper functions
# ============================================================

def load_3d(path):
    nii = nib.load(path)
    data = np.asanyarray(nii.dataobj).astype(np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    return nii, data


def to_um_factor(unit):
    unit = unit.lower()
    if unit in ("um", "µm"):
        return 1.0
    if unit == "mm":
        return 1000.0
    raise ValueError(f"Unsupported unit: {unit}")


def affine_to_um(affine, unit):
    output = affine.copy().astype(np.float64)
    output[:3, :] *= to_um_factor(unit)
    return output


def get_spacing_um(nii, unit):
    return np.linalg.norm(affine_to_um(nii.affine, unit)[:3, :3], axis=0)


def get_voxel_volume_um3(nii, unit):
    affine_um = affine_to_um(nii.affine, unit)
    return float(abs(np.linalg.det(affine_um[:3, :3])))


def format_size(values):
    values = [
        f"{int(round(v))}" if abs(v - round(v)) < 1e-3 else f"{v:.2f}"
        for v in values
    ]
    return f"{values[0]} × {values[1]} × {values[2]} µm³"


def print_affine(name, nii):
    print(f"[INFO] {name} affine:")
    print(nii.affine)


def build_slab_mask(nii, plane, range_mm):
    plane = plane.lower()
    plane_codes = {"sagittal": ("L", "R"), "coronal": ("P", "A"), "axial": ("I", "S")}
    if plane not in plane_codes:
        raise ValueError('SLAB_PLANE must be "axial", "coronal", or "sagittal".')
    if len(range_mm) != 2:
        raise ValueError("SLAB_RANGE_MM must contain exactly two values.")

    start_mm, end_mm = sorted(float(v) for v in range_mm)
    if start_mm < 0 or end_mm <= start_mm:
        raise ValueError("SLAB_RANGE_MM must satisfy 0 <= start < end.")

    axcodes = nib.aff2axcodes(nii.affine)
    matching_axes = [i for i, code in enumerate(axcodes) if code in plane_codes[plane]]
    if len(matching_axes) != 1:
        raise ValueError(f"Could not identify a unique {plane} axis from orientation {axcodes}.")

    axis = matching_axes[0]
    spacing_mm = get_spacing_um(nii, DMRI_WORLD_UNIT)[axis] / 1000.0
    positions_mm = (np.arange(nii.shape[axis], dtype=np.float64) + 0.5) * spacing_mm
    keep_1d = (positions_mm >= start_mm) & (positions_mm <= end_mm)

    if not np.any(keep_1d):
        raise ValueError(
            f"SLAB_RANGE_MM={range_mm} selects no voxels. "
            f"Available {plane} distance is approximately 0-{nii.shape[axis] * spacing_mm:.3f} mm."
        )

    shape = [1, 1, 1]
    shape[axis] = nii.shape[axis]
    mask = np.broadcast_to(keep_1d.reshape(shape), nii.shape[:3]).copy()
    selected_indices = np.flatnonzero(keep_1d)

    print()
    print("[INFO] Slab analysis enabled.")
    print(f"[INFO] Plane: {plane}")
    print(f"[INFO] Image orientation: {axcodes}")
    print(f"[INFO] Slab voxel axis: {axis} ({axcodes[axis]})")
    print(f"[INFO] Requested slab range from first voxel edge: {start_mm:g}-{end_mm:g} mm")
    print(f"[INFO] Selected voxel indices: {selected_indices[0]}-{selected_indices[-1]}")
    print(f"[INFO] Selected slices: {len(selected_indices)}")
    print(f"[INFO] Selected slab voxels: {int(mask.sum())}")
    return mask


# ============================================================
# Load nuclei centres
# ============================================================

def load_centers_voxel(csv_path, microscopy_nii, csv_unit, microscopy_unit):
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        has_world = {"world_x", "world_y", "world_z"}.issubset(fieldnames)
        has_voxel = {"voxel_center_x", "voxel_center_y", "voxel_center_z"}.issubset(fieldnames)

        if not has_world and not has_voxel:
            raise ValueError(
                "CSV must contain either world_x/world_y/world_z or "
                f"voxel_center_x/voxel_center_y/voxel_center_z. Found: {fieldnames}"
            )

        if has_world:
            factor = to_um_factor(csv_unit)
            centers_world_um = []

            for row in reader:
                try:
                    centers_world_um.append([
                        float(row["world_x"]) * factor,
                        float(row["world_y"]) * factor,
                        float(row["world_z"]) * factor,
                    ])
                except (ValueError, TypeError):
                    continue

            centers_world_um = np.asarray(centers_world_um, dtype=np.float64)

            if len(centers_world_um) == 0:
                raise ValueError("No valid nuclei centres found.")

            microscopy_affine_um = affine_to_um(microscopy_nii.affine, microscopy_unit)
            centers_voxel = nib.affines.apply_affine(
                np.linalg.inv(microscopy_affine_um),
                centers_world_um,
            )

            print("[INFO] Center CSV coordinate type: world coordinates")
            print("[INFO] Converted world coordinates to microscopy voxel coordinates.")

        else:
            centers_voxel = []

            for row in reader:
                try:
                    centers_voxel.append([
                        float(row["voxel_center_x"]),
                        float(row["voxel_center_y"]),
                        float(row["voxel_center_z"]),
                    ])
                except (ValueError, TypeError):
                    continue

            centers_voxel = np.asarray(centers_voxel, dtype=np.float64)

            if len(centers_voxel) == 0:
                raise ValueError("No valid nuclei centres found.")

            print("[INFO] Center CSV coordinate type: microscopy voxel coordinates")
            print("[INFO] Using voxel centers directly in microscopy reference space.")

    print("[INFO] Number of nuclei centres:", len(centers_voxel))
    return centers_voxel


# ============================================================
# dMRI resampling
# ============================================================

def build_target_geometry(nii, target_spacing_um, spatial_unit):
    current_spacing_um = get_spacing_um(nii, spatial_unit)
    target_spacing_um = np.asarray(target_spacing_um, dtype=np.float64)

    if target_spacing_um.shape != (3,) or np.any(target_spacing_um <= 0):
        raise ValueError("TARGET_DMRI_SPACING_UM must contain three positive values.")

    current_affine = nii.affine.astype(np.float64)
    current_linear = current_affine[:3, :3]

    current_spacing_native = np.linalg.norm(current_linear, axis=0)
    target_spacing_native = target_spacing_um / to_um_factor(spatial_unit)

    direction = current_linear / current_spacing_native
    target_linear = direction * target_spacing_native

    current_shape = np.asarray(nii.shape[:3], dtype=np.float64)
    target_shape = np.maximum(
        1,
        np.round(current_shape * current_spacing_um / target_spacing_um).astype(int),
    )

    old_edge_origin = nib.affines.apply_affine(current_affine, [-0.5, -0.5, -0.5])
    target_translation = old_edge_origin + target_linear @ np.array([0.5, 0.5, 0.5])

    target_affine = np.eye(4, dtype=np.float64)
    target_affine[:3, :3] = target_linear
    target_affine[:3, 3] = target_translation

    old_fov_um = current_shape * current_spacing_um
    new_fov_um = target_shape.astype(np.float64) * target_spacing_um

    print("[INFO] Native dMRI shape:", tuple(nii.shape[:3]))
    print("[INFO] Native dMRI spacing (um):", current_spacing_um)
    print("[INFO] Native dMRI FOV (um):", old_fov_um)
    print("[INFO] Target dMRI shape:", tuple(target_shape))
    print("[INFO] Target dMRI spacing (um):", target_spacing_um)
    print("[INFO] Target dMRI FOV (um):", new_fov_um)
    print("[INFO] FOV difference after rounding (um):", new_fov_um - old_fov_um)

    return tuple(target_shape), target_affine


def make_nifti(data, affine, source_nii):
    header = source_nii.header.copy()
    output_nii = nib.Nifti1Image(data.astype(np.float32), affine, header)
    output_nii.set_data_dtype(np.float32)

    _, qcode = source_nii.get_qform(coded=True)
    _, scode = source_nii.get_sform(coded=True)

    qcode = int(qcode) if qcode is not None and int(qcode) > 0 else 1
    scode = int(scode) if scode is not None and int(scode) > 0 else 1

    output_nii.set_qform(affine, qcode)
    output_nii.set_sform(affine, scode)

    return output_nii


def resample_dmri_metric(nii, data, target_shape, target_affine, interpolation_order):
    if data.shape == target_shape and np.allclose(nii.affine, target_affine, atol=1e-7):
        return make_nifti(data, target_affine, nii), data.copy()

    valid = np.isfinite(data) & (data != 0)
    clean_data = np.where(valid, data, 0).astype(np.float32)

    data_nii = nib.Nifti1Image(clean_data, nii.affine)
    weight_nii = nib.Nifti1Image(valid.astype(np.float32), nii.affine)

    target = (target_shape, target_affine)

    resampled_data = resample_from_to(
        data_nii,
        target,
        order=interpolation_order,
        mode="constant",
        cval=0.0,
    ).get_fdata(dtype=np.float32)

    resampled_weight = resample_from_to(
        weight_nii,
        target,
        order=interpolation_order,
        mode="constant",
        cval=0.0,
    ).get_fdata(dtype=np.float32)

    resampled_support = resample_from_to(
        weight_nii,
        target,
        order=0,
        mode="constant",
        cval=0.0,
    ).get_fdata(dtype=np.float32) > 0.5

    output = np.zeros(target_shape, dtype=np.float32)
    valid_output = resampled_support & (resampled_weight > 1e-6)

    output[valid_output] = resampled_data[valid_output] / resampled_weight[valid_output]

    output_nii = make_nifti(output, target_affine, nii)

    return output_nii, output


def save_nifti(nii, output_path):
    nib.save(nii, output_path)
    print("[INFO] Saved:", output_path)


# ============================================================
# Map nuclei to dMRI voxels
# ============================================================

def microscopy_points_to_dmri_voxels(points_voxel, microscopy_nii, dmri_nii):
    microscopy_shape = np.asarray(microscopy_nii.shape[:3])
    dmri_shape = np.asarray(dmri_nii.shape[:3])

    microscopy_spacing_um = get_spacing_um(
        microscopy_nii,
        MICROSCOPY_WORLD_UNIT,
    )
    dmri_spacing_um = get_spacing_um(
        dmri_nii,
        DMRI_WORLD_UNIT,
    )

    microscopy_fov_um = microscopy_shape * microscopy_spacing_um
    dmri_fov_um = dmri_shape * dmri_spacing_um
    fov_difference_um = np.abs(microscopy_fov_um - dmri_fov_um)
    tolerance_um = np.maximum(microscopy_spacing_um, dmri_spacing_um)

    print("[INFO] Microscopy FOV (um):", microscopy_fov_um)
    print("[INFO] Analysis dMRI FOV (um):", dmri_fov_um)
    print("[INFO] FOV difference (um):", fov_difference_um)

    if np.any(fov_difference_um > tolerance_um):
        print(
            f"[WARNING] Microscopy and dMRI physical FOVs do not match. "
            f"Difference={fov_difference_um} um, tolerance={tolerance_um} um. "
            f"Continue anyway."
        )

    inside_microscopy = (
        (points_voxel[:, 0] >= -0.5) & (points_voxel[:, 0] < microscopy_shape[0] - 0.5) &
        (points_voxel[:, 1] >= -0.5) & (points_voxel[:, 1] < microscopy_shape[1] - 0.5) &
        (points_voxel[:, 2] >= -0.5) & (points_voxel[:, 2] < microscopy_shape[2] - 0.5)
    )

    points_um = (points_voxel + 0.5) * microscopy_spacing_um
    indices = np.floor(points_um / dmri_spacing_um).astype(np.int64)

    inside_dmri = (
        (indices[:, 0] >= 0) & (indices[:, 0] < dmri_shape[0]) &
        (indices[:, 1] >= 0) & (indices[:, 1] < dmri_shape[1]) &
        (indices[:, 2] >= 0) & (indices[:, 2] < dmri_shape[2])
    )

    valid = inside_microscopy & inside_dmri
    indices = indices[valid]

    flat_ids = np.ravel_multi_index(
        indices.T,
        tuple(dmri_shape),
    )

    print("[INFO] Nuclei inside microscopy image:", int(inside_microscopy.sum()))
    print("[INFO] Nuclei outside microscopy image:", int((~inside_microscopy).sum()))
    print("[INFO] Nuclei mapped inside analysis dMRI:", int(valid.sum()))
    print(
        "[INFO] Nuclei outside analysis dMRI FOV:",
        int((inside_microscopy & ~inside_dmri).sum()),
    )

    return flat_ids


def aggregate_nuclei(flat_ids, shape):
    counts = np.bincount(flat_ids, minlength=int(np.prod(shape)))
    return counts.reshape(shape).astype(np.int32)


# ============================================================
# Low-density mask back to native dMRI
# ============================================================

def resample_mask_to_native(mask, analysis_nii, native_nii):
    mask_nii = nib.Nifti1Image(mask.astype(np.uint8), analysis_nii.affine)

    native_mask = resample_from_to(
        mask_nii,
        (native_nii.shape[:3], native_nii.affine),
        order=0,
        mode="constant",
        cval=0,
    ).get_fdata(dtype=np.float32)

    return native_mask > 0.5


def save_filtered_original_volume(nii, data, remove_voxel_mask, output_path, fill_value):
    filtered = data.copy()
    filtered[remove_voxel_mask] = fill_value

    header = nii.header.copy()
    output_nii = nib.Nifti1Image(filtered.astype(np.float32), nii.affine, header)
    output_nii.set_data_dtype(np.float32)

    qform, qcode = nii.get_qform(coded=True)
    sform, scode = nii.get_sform(coded=True)

    if qform is not None:
        output_nii.set_qform(qform, int(qcode))
    if sform is not None:
        output_nii.set_sform(sform, int(scode))

    nib.save(output_nii, output_path)
    print("[INFO] Saved filtered original volume:", output_path)


# ============================================================
# Statistics
# ============================================================

def get_stats(x, y):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {
            "n": len(x),
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_r": np.nan,
            "spearman_p": np.nan,
            "slope": np.nan,
            "slope_ci_lower": np.nan,
            "slope_ci_upper": np.nan,
        }

    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)

    slope, intercept = np.polyfit(x, y, 1)
    y_fit = slope * x + intercept
    dof = len(x) - 2
    residual_se = np.sqrt(np.sum((y - y_fit) ** 2) / dof)
    sxx = np.sum((x - np.mean(x)) ** 2)
    slope_se = residual_se / np.sqrt(sxx)
    t_value = t.ppf(0.975, dof)
    slope_ci_lower = slope - t_value * slope_se
    slope_ci_upper = slope + t_value * slope_se

    return {
        "n": len(x),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "slope": slope,
        "slope_ci_lower": slope_ci_lower,
        "slope_ci_upper": slope_ci_upper,
    }


def add_regression_line(ax, x, y):
    if len(x) < 2 or np.ptp(x) == 0:
        return

    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 200)

    ax.plot(
        x_line,
        slope * x_line + intercept,
        color="red",
        linewidth=2,
        label="Linear regression",
    )
    ax.legend(loc="upper right", frameon=False)



# ============================================================
# Plotting
# ============================================================

def plot_hexbin(x, y, ylabel, output_path, voxel_size_um):
    stats = get_stats(x, y)

    fig, ax = plt.subplots(figsize=(7, 5))
    hb = ax.hexbin(
        x,
        y,
        gridsize=HEXBIN_GRIDSIZE,
        bins="log",
        mincnt=1,
        cmap="Blues",
    )

    add_regression_line(ax, x, y)

    ax.set_xlabel(
        rf"Nuclei density (nuclei / {DENSITY_UNIT_VOLUME_UM3:g} $\mu$m$^3$)"
    )
    ax.set_ylabel(ylabel)

    text = (
        f"r = {stats['pearson_r']:.3f}\n"
        f"ρ = {stats['spearman_r']:.3f}\n"
        f"Slope = {stats['slope']:.3g}\n"
    )
    ax.text(
        0.97,
        0.03,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
    )

    ax.text(
        0.03,
        0.97,
        "dMRI voxel: " + format_size(voxel_size_um),
        transform=ax.transAxes,
        ha="left",
        va="top",
    )

    cbar = fig.colorbar(hb, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("Voxel count")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("[INFO] Saved:", output_path)
    return stats
def plot_scatter(x, y, ylabel, output_path, voxel_size_um):
    stats = get_stats(x, y)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=12, alpha=0.35, edgecolors="none")

    add_regression_line(ax, x, y)

    ax.set_xlabel(
        rf"Nuclei density (nuclei / {DENSITY_UNIT_VOLUME_UM3:g} $\mu$m$^3$)"
    )
    ax.set_ylabel(ylabel)

    text = (
        f"r = {stats['pearson_r']:.3f}\n"
        f"ρ = {stats['spearman_r']:.3f}\n"
        f"Slope = {stats['slope']:.3g}\n"
    )
    ax.text(
        0.97,
        0.03,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
    )

    ax.text(
        0.03,
        0.97,
        "dMRI voxel: " + format_size(voxel_size_um),
        transform=ax.transAxes,
        ha="left",
        va="top",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("[INFO] Saved:", output_path)


def plot_density_groups(x, y, ylabel, output_path, voxel_size_um):
    q1, q2 = np.quantile(x, [1 / 3, 2 / 3])

    low = y[x <= q1]
    medium = y[(x > q1) & (x <= q2)]
    high = y[x > q2]

    groups = [low, medium, high]
    labels = ["Low", "Medium", "High"]
    medians = [np.median(g) if len(g) else np.nan for g in groups]

    valid_groups = [g for g in groups if len(g)]
    valid_labels = [label for label, g in zip(labels, groups) if len(g)]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.boxplot(
        valid_groups,
        tick_labels=valid_labels,
        showfliers=False,
        widths=0.6,
    )

    for i, group in enumerate(valid_groups, start=1):
        median = np.median(group)
        ax.text(i, median, f"  {median:.3g}", va="center")

    ax.set_xlabel("Nuclei density group")
    ax.set_ylabel(ylabel)

    ax.text(
        0.03,
        0.97,
        "dMRI voxel: " + format_size(voxel_size_um),
        transform=ax.transAxes,
        ha="left",
        va="top",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("[INFO] Saved:", output_path)

    return {
        "low_median": medians[0],
        "medium_median": medians[1],
        "high_median": medians[2],
        "low_n": len(low),
        "medium_n": len(medium),
        "high_n": len(high),
    }


# ============================================================
# Check input
# ============================================================

for path in (CENTER_CSV_PATH, MICROSCOPY_REFERENCE_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

if len(MRI_METRICS) == 0:
    raise ValueError("MRI_METRICS is empty.")

for metric_name, config in MRI_METRICS.items():
    if not os.path.exists(config["path"]):
        raise FileNotFoundError(f"{metric_name}: {config['path']}")


# ============================================================
# Load microscopy
# ============================================================

microscopy_nii = nib.load(MICROSCOPY_REFERENCE_PATH)

centers_voxel = load_centers_voxel(
    CENTER_CSV_PATH,
    microscopy_nii,
    CSV_WORLD_UNIT,
    MICROSCOPY_WORLD_UNIT,
)


# ============================================================
# Load native registered dMRI
# ============================================================

native_metric_data = {}

reference_metric_name = next(iter(MRI_METRICS))
native_reference_nii = None
native_reference_shape = None

for metric_name, config in MRI_METRICS.items():
    nii, data = load_3d(config["path"])

    if native_reference_nii is None:
        native_reference_nii = nii
        native_reference_shape = data.shape
    else:
        if data.shape != native_reference_shape:
            raise ValueError(
                f"{metric_name} shape differs from {reference_metric_name}: "
                f"{metric_name}={data.shape}, "
                f"{reference_metric_name}={native_reference_shape}"
            )

        if not np.allclose(nii.affine, native_reference_nii.affine, atol=1e-5):
            raise ValueError(
                f"{metric_name} affine differs from {reference_metric_name}."
            )

    native_metric_data[metric_name] = {
        "nii": nii,
        "data": data,
        "ylabel": config.get("ylabel", f"Mean registered {metric_name}"),
    }

    print()
    print(f"[INFO] Loaded native {metric_name}")
    print("[INFO] Shape:", data.shape)
    print("[INFO] Spacing (um):", get_spacing_um(nii, DMRI_WORLD_UNIT))
    print("[INFO] Range:", np.nanmin(data), np.nanmax(data))
    print_affine(metric_name, nii)


# ============================================================
# Build analysis dMRI
# ============================================================

print()
print("========================================")
print("Build analysis dMRI grid")
print("========================================")

print("[INFO] Microscopy shape:", microscopy_nii.shape[:3])
print("[INFO] Microscopy spacing (um):", get_spacing_um(microscopy_nii, MICROSCOPY_WORLD_UNIT))
print_affine("Microscopy", microscopy_nii)

if ENABLE_DMRI_RESAMPLING:
    target_shape, target_affine = build_target_geometry(
        native_reference_nii,
        TARGET_DMRI_SPACING_UM,
        DMRI_WORLD_UNIT,
    )
else:
    target_shape = native_reference_shape
    target_affine = native_reference_nii.affine.copy()

analysis_metric_data = {}

if SAVE_RESAMPLED_DMRI and ENABLE_DMRI_RESAMPLING:
    os.makedirs(RESAMPLED_DIR, exist_ok=True)

for metric_name, info in native_metric_data.items():
    if ENABLE_DMRI_RESAMPLING:
        analysis_nii, analysis_data = resample_dmri_metric(
            info["nii"],
            info["data"],
            target_shape,
            target_affine,
            DMRI_INTERPOLATION_ORDER,
        )
    else:
        analysis_nii = make_nifti(
            info["data"],
            info["nii"].affine,
            info["nii"],
        )
        analysis_data = info["data"].copy()

    analysis_metric_data[metric_name] = {
        "nii": analysis_nii,
        "data": analysis_data,
        "ylabel": info["ylabel"],
    }

    print()
    print(f"[INFO] Analysis {metric_name}")
    print("[INFO] Shape:", analysis_data.shape)
    print("[INFO] Spacing (um):", get_spacing_um(analysis_nii, DMRI_WORLD_UNIT))
    print("[INFO] Range:", np.nanmin(analysis_data), np.nanmax(analysis_data))
    print_affine(f"Analysis {metric_name}", analysis_nii)

    if SAVE_RESAMPLED_DMRI and ENABLE_DMRI_RESAMPLING:
        output_path = os.path.join(
            RESAMPLED_DIR,
            f"{metric_name}_resampled.nii.gz",
        )
        save_nifti(analysis_nii, output_path)


# ============================================================
# Check analysis metric geometry
# ============================================================

analysis_reference_nii = analysis_metric_data[reference_metric_name]["nii"]
analysis_reference_shape = analysis_metric_data[reference_metric_name]["data"].shape

for metric_name, info in analysis_metric_data.items():
    if info["data"].shape != analysis_reference_shape:
        raise ValueError(
            f"{metric_name} resampled shape differs from "
            f"{reference_metric_name}."
        )

    if not np.allclose(
        info["nii"].affine,
        analysis_reference_nii.affine,
        atol=1e-5,
    ):
        raise ValueError(
            f"{metric_name} resampled affine differs from "
            f"{reference_metric_name}."
        )


# ============================================================
# Nuclei density on analysis dMRI voxels
# ============================================================

voxel_size_um = get_spacing_um(
    analysis_reference_nii,
    DMRI_WORLD_UNIT,
)
voxel_volume_um3 = get_voxel_volume_um3(
    analysis_reference_nii,
    DMRI_WORLD_UNIT,
)

flat_ids = microscopy_points_to_dmri_voxels(
    centers_voxel,
    microscopy_nii,
    analysis_reference_nii,
)

nuclei_count = aggregate_nuclei(
    flat_ids,
    analysis_reference_shape,
)

density = (
    nuclei_count.astype(np.float64)
    / voxel_volume_um3
    * DENSITY_UNIT_VOLUME_UM3
)

if SAVE_MAPPED_MICROSCOPY:
    count_nii = make_nifti(
        nuclei_count.astype(np.float32),
        analysis_reference_nii.affine,
        analysis_reference_nii,
    )
    density_nii = make_nifti(
        density.astype(np.float32),
        analysis_reference_nii.affine,
        analysis_reference_nii,
    )

    save_nifti(
        count_nii,
        os.path.join(
            OUTPUT_DIR,
            "microscopy_nuclei_count_on_dmri_grid.nii.gz",
        ),
    )
    save_nifti(
        density_nii,
        os.path.join(
            OUTPUT_DIR,
            "microscopy_density_on_dmri_grid.nii.gz",
        ),
    )

unique_density = np.unique(density)

print()
print("[INFO] Analysis dMRI voxel size (um):", voxel_size_um)
print("[INFO] Analysis dMRI voxel volume (um^3):", voxel_volume_um3)
print("[INFO] Maximum nuclei in one dMRI voxel:", nuclei_count.max())
print(
    f"[INFO] Density range "
    f"(nuclei / {DENSITY_UNIT_VOLUME_UM3:g} um^3):",
    density.min(),
    density.max(),
)
print(
    "[INFO] Density step:",
    DENSITY_UNIT_VOLUME_UM3 / voxel_volume_um3,
)
print("[INFO] Unique density values:", len(unique_density))
print("[INFO] First unique density values:", unique_density[:20])

if ENABLE_SLAB_ANALYSIS:
    slab_mask = build_slab_mask(analysis_reference_nii, SLAB_PLANE, SLAB_RANGE_MM)
else:
    slab_mask = np.ones(analysis_reference_shape, dtype=bool)
    print()
    print("[INFO] Slab analysis disabled. Using the full 3D volume.")

if SAVE_ANALYSIS_SLAB_MASK and ENABLE_SLAB_ANALYSIS:
    slab_nii = make_nifti(slab_mask.astype(np.float32), analysis_reference_nii.affine, analysis_reference_nii)
    save_nifti(slab_nii, os.path.join(OUTPUT_DIR, "analysis_slab_mask.nii.gz"))


# ============================================================
# Shared low-density exclusion mask
# ============================================================

low_density_mask = None

if ENABLE_VALUE_RANGE_EXCLUSION and X_EXCLUDE_RANGE is not None:
    x_min, x_max = X_EXCLUDE_RANGE
    low_density_mask = (
        (density >= x_min)
        & (density <= x_max)
        & slab_mask
    )

    reference_data = analysis_metric_data[reference_metric_name]["data"]
    reference_valid = np.isfinite(reference_data) & (reference_data != 0)

    print()
    print(f"[INFO] Shared low-density range: [{x_min}, {x_max}]")
    print(
        "[INFO] Low-density analysis voxels:",
        int(low_density_mask.sum()),
    )
    print(
        "[INFO] Low-density voxels inside reference dMRI:",
        int((low_density_mask & reference_valid).sum()),
    )
else:
    print()
    print("[INFO] Shared low-density volume removal disabled.")

if SAVE_LOW_DENSITY_VOLUME and low_density_mask is not None:
    x_min, x_max = X_EXCLUDE_RANGE
    reference_data = analysis_metric_data[reference_metric_name]["data"]
    reference_valid = np.isfinite(reference_data) & (reference_data != 0)
    keep_mask = low_density_mask & reference_valid
    low_density_only = np.zeros(analysis_reference_shape, dtype=np.float32)

    if x_max > x_min:
        low_density_only[keep_mask] = (x_max - density[keep_mask]) / (x_max - x_min)
    else:
        low_density_only[keep_mask] = 1.0

    low_density_nii = make_nifti(low_density_only, analysis_reference_nii.affine, analysis_reference_nii)
    save_nifti(low_density_nii, os.path.join(OUTPUT_DIR, "microscopy_low_density_only_inverted_on_dmri_grid.nii.gz"))
    print("[INFO] Low-density-only voxels saved:", int(keep_mask.sum()))
    if np.any(keep_mask):
        print("[INFO] Inverted low-density intensity range:", float(low_density_only[keep_mask].min()), float(low_density_only[keep_mask].max()))


# ============================================================
# Analyze each dMRI metric
# ============================================================

rows = []

for metric_name, info in analysis_metric_data.items():
    print()
    print("========================================")
    print(f"Processing {metric_name}")
    print("========================================")

    data = info["data"]
    valid = np.isfinite(data) & (data != 0)

    print(
        f"[INFO] {metric_name} non-zero analysis voxels before slab/filtering:",
        int(valid.sum()),
    )

    valid &= slab_mask
    print(
        f"[INFO] {metric_name} non-zero voxels inside selected slab:",
        int(valid.sum()),
    )

    if SAVE_FILTERED_ORIGINAL_VOLUMES and low_density_mask is not None:
        native_info = native_metric_data[metric_name]

        if ENABLE_DMRI_RESAMPLING:
            remove_native_mask = resample_mask_to_native(
                low_density_mask,
                analysis_reference_nii,
                native_info["nii"],
            )
        else:
            remove_native_mask = low_density_mask.copy()

        remove_native_mask &= (
            np.isfinite(native_info["data"])
            & (native_info["data"] != 0)
        )

        print(
            f"[INFO] {metric_name} native voxels removed by low-density mask:",
            int(remove_native_mask.sum()),
        )

        output_filtered_volume = os.path.join(
            OUTPUT_DIR,
            f"{metric_name}_remove_low_density_native.nii.gz",
        )

        save_filtered_original_volume(
            native_info["nii"],
            native_info["data"],
            remove_native_mask,
            output_filtered_volume,
            FILTERED_VOXEL_VALUE,
        )

    if EXCLUDE_ZERO_DENSITY:
        valid &= density > 0
        print("[INFO] Zero-density voxels excluded.")

    if ENABLE_VALUE_RANGE_EXCLUSION:
        exclude = np.zeros(analysis_reference_shape, dtype=bool)

        if X_EXCLUDE_RANGE is not None:
            x_min, x_max = X_EXCLUDE_RANGE
            x_exclude = (
                (density >= x_min)
                & (density <= x_max)
            )
            exclude |= x_exclude

            print(
                f"[INFO] X range excluded: [{x_min}, {x_max}], voxels:",
                int((valid & x_exclude).sum()),
            )

        y_range = Y_EXCLUDE_RANGES.get(metric_name)

        if y_range is not None:
            y_min, y_max = y_range
            y_exclude = (
                (data >= y_min)
                & (data <= y_max)
            )
            exclude |= y_exclude

            print(
                f"[INFO] {metric_name} Y range excluded: "
                f"[{y_min}, {y_max}], voxels:",
                int((valid & y_exclude).sum()),
            )

        print(
            f"[INFO] {metric_name} total voxels excluded by value range:",
            int((valid & exclude).sum()),
        )

        valid &= ~exclude
    else:
        print("[INFO] Value-range exclusion disabled.")

    density_values = density[valid]
    metric_values = data[valid]

    if len(metric_values) < 2:
        print(
            f"[WARNING] Not enough valid voxels for {metric_name}. "
            "Skipping."
        )
        continue

    print(
        f"[INFO] {metric_name} valid analysis voxels after filtering:",
        len(metric_values),
    )
    print(
        f"[INFO] {metric_name} density range:",
        density_values.min(),
        density_values.max(),
    )
    print(
        f"[INFO] {metric_name} range:",
        metric_values.min(),
        metric_values.max(),
    )

    output_hexbin = os.path.join(
        OUTPUT_DIR,
        f"density_vs_{metric_name}_hexbin.png",
    )
    output_scatter = os.path.join(
        OUTPUT_DIR,
        f"density_vs_{metric_name}_scatter.png",
    )
    output_boxplot = os.path.join(
        OUTPUT_DIR,
        f"density_groups_{metric_name}_boxplot.png",
    )

    stats = plot_hexbin(
        density_values,
        metric_values,
        info["ylabel"],
        output_hexbin,
        voxel_size_um,
    )

    plot_scatter(
        density_values,
        metric_values,
        info["ylabel"],
        output_scatter,
        voxel_size_um,
    )

    groups = plot_density_groups(
        density_values,
        metric_values,
        info["ylabel"],
        output_boxplot,
        voxel_size_um,
    )

    rows.append({
        "metric": metric_name,
        "resampling_enabled": ENABLE_DMRI_RESAMPLING,
        "voxel_x_um": voxel_size_um[0],
        "voxel_y_um": voxel_size_um[1],
        "voxel_z_um": voxel_size_um[2],
        "voxel_volume_um3": voxel_volume_um3,
        "density_unit_volume_um3": DENSITY_UNIT_VOLUME_UM3,
        "slab_analysis_enabled": ENABLE_SLAB_ANALYSIS,
        "slab_plane": SLAB_PLANE if ENABLE_SLAB_ANALYSIS else None,
        "slab_range_mm": SLAB_RANGE_MM if ENABLE_SLAB_ANALYSIS else None,
        "exclude_zero_density": EXCLUDE_ZERO_DENSITY,
        "value_range_exclusion_enabled": ENABLE_VALUE_RANGE_EXCLUSION,
        "x_exclude_range": X_EXCLUDE_RANGE,
        "y_exclude_range": Y_EXCLUDE_RANGES.get(metric_name),
        **stats,
        **groups,
    })


# ============================================================
# Save summary
# ============================================================

if len(rows) > 0:
    with open(OUTPUT_STATS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("[INFO] Saved:", OUTPUT_STATS_CSV)
else:
    print("[WARNING] No metric results available for summary.")

print("[INFO] Output directory:", OUTPUT_DIR)