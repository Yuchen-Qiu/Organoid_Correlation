#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad


# ============================================================
# Paths
# ============================================================

work_path = "/autofs/arch11/DATA/HOMES/yuchen/Minor/Phantom/phantom_25um"

output_path = os.path.join(work_path, "SV_results_new")
os.makedirs(output_path, exist_ok=True)

centroid_csv = os.path.join(
    work_path,
    "watershed_centroids.csv"
)

# GRE / segmentation NIfTI used to generate centroid_x / centroid_y.
# Must be the same image space as the centroid CSV.
gre_path = "/autofs/arch11/DATA/HOMES/yuchen/Minor/Phantom/5mmIprobe_3mm_phantom.nii"

# Diffusion grid template.
diffusion_path = os.path.join(
    work_path,
    "tn28T_260428_4_5mmIprobe_3mm_phantom2_20C_56_1_30_adjusted.nii"
)

output_count_path = os.path.join(
    output_path,
    "fiber_area_weighted_count_on_diffusion_grid.nii.gz"
)

output_S_path = os.path.join(
    output_path,
    "gre_derived_S_um_on_diffusion_grid.nii.gz"
)

output_fiber_area_path = os.path.join(
    output_path,
    "gre_derived_fiber_area_um2_on_diffusion_grid.nii.gz"
)

output_V_path = os.path.join(
    output_path,
    "gre_derived_V_um2_on_diffusion_grid.nii.gz"
)

output_sv_path = os.path.join(
    output_path,
    "gre_derived_S_over_V_1_per_um_on_diffusion_grid.nii.gz"
)

output_csv = os.path.join(
    output_path,
    "fiber_area_S_V_SoverV_on_diffusion_grid.csv"
)

output_debug_csv = os.path.join(
    output_path,
    "fiber_assignment_debug.csv"
)

output_sv_hist_png = os.path.join(
    output_path,
    "gre_derived_S_over_V_histogram.png"
)


# ============================================================
# Parameters
# ============================================================

fiber_radius_um = 12.5
GEOM_TOL = 1e-10


# ============================================================
# Helper functions
# ============================================================

def get_xy_voxel_size_um_from_nifti(nii):
    header = nii.header
    zooms = header.get_zooms()
    spatial_unit, time_unit = header.get_xyzt_units()

    if len(zooms) < 2:
        raise ValueError(f"Cannot read x/y voxel size from header zooms: {zooms}")

    voxel_size_x = float(zooms[0])
    voxel_size_y = float(zooms[1])

    print("Header zooms:", zooms)
    print("Header spatial unit:", spatial_unit)
    print("Header time unit:", time_unit)

    if spatial_unit == "micron":
        voxel_size_x_um = voxel_size_x
        voxel_size_y_um = voxel_size_y
    elif spatial_unit == "mm":
        voxel_size_x_um = voxel_size_x * 1000.0
        voxel_size_y_um = voxel_size_y * 1000.0
    elif spatial_unit == "meter":
        voxel_size_x_um = voxel_size_x * 1_000_000.0
        voxel_size_y_um = voxel_size_y * 1_000_000.0
    elif spatial_unit == "unknown":
        print("\nWARNING:")
        print("NIfTI spatial unit is unknown.")
        print("Assuming the header spacing is in mm and converting to um.\n")
        voxel_size_x_um = voxel_size_x * 1000.0
        voxel_size_y_um = voxel_size_y * 1000.0
    else:
        raise ValueError(
            f"Unsupported spatial unit: {spatial_unit}. "
            "Please manually convert voxel size to um."
        )

    return voxel_size_x_um, voxel_size_y_um, spatial_unit


def gre_centroid_to_diffusion_index(x_gre, y_gre, gre_affine, inv_diff_affine):
    """
    CSV centroid:
        centroid_y = row coordinate
        centroid_x = column coordinate

    Nibabel voxel coordinate:
        [i, j, k] = [x, y, z]

    Therefore:
        gre_voxel = [centroid_x, centroid_y, 0]
    """
    gre_voxel_h = np.array([x_gre, y_gre, 0.0, 1.0], dtype=np.float64)
    world_h = gre_affine @ gre_voxel_h
    diff_voxel_h = inv_diff_affine @ world_h

    return (
        float(diff_voxel_h[0]),
        float(diff_voxel_h[1]),
        float(diff_voxel_h[2]),
        world_h,
    )


def make_3d_shape(shape):
    if len(shape) == 2:
        return (shape[0], shape[1], 1)
    if len(shape) == 3:
        return shape
    raise ValueError(f"Unsupported image dimension: {shape}")


def circle_rectangle_overlap_area(cx, cy, r, x0, x1, y0, y1):
    """
    Exact (numerically integrated) area of intersection between a circle
    and an axis-aligned rectangle.

    All coordinates are in micrometers in the diffusion voxel-axis system.
    """
    xa = max(x0, cx - r)
    xb = min(x1, cx + r)

    if xa >= xb:
        return 0.0

    def overlap_height(x):
        dx = x - cx
        inside = r * r - dx * dx
        if inside <= 0:
            return 0.0

        h = np.sqrt(inside)
        circle_y0 = cy - h
        circle_y1 = cy + h

        low = max(y0, circle_y0)
        high = min(y1, circle_y1)
        return max(0.0, high - low)

    # Add x positions where the circle intersects y0/y1 as integration
    # breakpoints. This improves numerical stability near corners.
    points = []
    for yb in (y0, y1):
        dy = yb - cy
        if abs(dy) < r:
            dx = np.sqrt(max(0.0, r * r - dy * dy))
            for xp in (cx - dx, cx + dx):
                if xa < xp < xb:
                    points.append(xp)

    area, _ = quad(
        overlap_height,
        xa,
        xb,
        points=sorted(set(points)) if points else None,
        epsabs=1e-9,
        epsrel=1e-9,
        limit=100,
    )

    return max(0.0, float(area))


def circle_rectangle_arc_length(cx, cy, r, x0, x1, y0, y1):
    """
    Exact arc length of the circle circumference lying inside one
    axis-aligned rectangle.
    """
    two_pi = 2.0 * np.pi
    angles = [0.0, two_pi]

    # Intersections with x = constant boundaries.
    for xb in (x0, x1):
        q = (xb - cx) / r
        if -1.0 <= q <= 1.0:
            q = np.clip(q, -1.0, 1.0)
            a = np.arccos(q)
            for angle in (a, two_pi - a):
                if GEOM_TOL < angle < two_pi - GEOM_TOL:
                    angles.append(float(angle))

    # Intersections with y = constant boundaries.
    for yb in (y0, y1):
        q = (yb - cy) / r
        if -1.0 <= q <= 1.0:
            q = np.clip(q, -1.0, 1.0)
            a = np.arcsin(q)
            for angle in (a % two_pi, (np.pi - a) % two_pi):
                if GEOM_TOL < angle < two_pi - GEOM_TOL:
                    angles.append(float(angle))

    angles = sorted(angles)

    # Remove nearly duplicated angles.
    unique_angles = [angles[0]]
    for angle in angles[1:]:
        if abs(angle - unique_angles[-1]) > GEOM_TOL:
            unique_angles.append(angle)

    if abs(unique_angles[-1] - two_pi) > GEOM_TOL:
        unique_angles.append(two_pi)

    total_angle = 0.0

    for a, b in zip(unique_angles[:-1], unique_angles[1:]):
        if b - a <= GEOM_TOL:
            continue

        mid = 0.5 * (a + b)
        x = cx + r * np.cos(mid)
        y = cy + r * np.sin(mid)

        if (
            x0 - GEOM_TOL <= x <= x1 + GEOM_TOL
            and y0 - GEOM_TOL <= y <= y1 + GEOM_TOL
        ):
            total_angle += b - a

    return float(r * total_angle)


def assign_fiber_exact(
    fiber_area_map,
    fiber_arc_map,
    x_center_voxel,
    y_center_voxel,
    z_center_voxel,
    voxel_size_x_um,
    voxel_size_y_um,
    fiber_radius_um,
):
    """
    Assign one circular fiber cross-section exactly to the diffusion voxels
    it overlaps.

    fiber_area_map:
        Accumulates circle-rectangle overlap area [um^2].

    fiber_arc_map:
        Accumulates circumference arc length inside each voxel [um].

    The through-plane depth is taken as 1, so:
        surface area numerically equals arc length,
        volume numerically equals cross-sectional area.
    """
    nx, ny, nz = fiber_area_map.shape

    if nz == 1:
        iz = 0
    else:
        iz = int(np.floor(z_center_voxel))

    ix_center = int(np.floor(x_center_voxel))
    iy_center = int(np.floor(y_center_voxel))

    if not (0 <= ix_center < nx and 0 <= iy_center < ny and 0 <= iz < nz):
        return {
            "inside": False,
            "corner_case": False,
            "assigned_area_um2": 0.0,
            "assigned_arc_um": 0.0,
            "area_fraction": 0.0,
            "arc_fraction": 0.0,
            "n_voxels": 0,
        }

    # Work in physical coordinates along the diffusion voxel axes.
    cx = x_center_voxel * voxel_size_x_um
    cy = y_center_voxel * voxel_size_y_um
    r = fiber_radius_um

    ix_min = int(np.floor((cx - r) / voxel_size_x_um))
    ix_max = int(np.floor((cx + r) / voxel_size_x_um))
    iy_min = int(np.floor((cy - r) / voxel_size_y_um))
    iy_max = int(np.floor((cy + r) / voxel_size_y_um))

    # Detect whether the circle crosses one x and one y boundary.
    local_x_um = (x_center_voxel - ix_center) * voxel_size_x_um
    local_y_um = (y_center_voxel - iy_center) * voxel_size_y_um

    crosses_x = (
        local_x_um < r
        or (voxel_size_x_um - local_x_um) < r
    )
    crosses_y = (
        local_y_um < r
        or (voxel_size_y_um - local_y_um) < r
    )
    corner_case = crosses_x and crosses_y

    assigned_area_um2 = 0.0
    assigned_arc_um = 0.0
    n_voxels = 0

    for ix in range(ix_min, ix_max + 1):
        if not (0 <= ix < nx):
            continue

        x0 = ix * voxel_size_x_um
        x1 = (ix + 1) * voxel_size_x_um

        for iy in range(iy_min, iy_max + 1):
            if not (0 <= iy < ny):
                continue

            y0 = iy * voxel_size_y_um
            y1 = (iy + 1) * voxel_size_y_um

            area = circle_rectangle_overlap_area(
                cx, cy, r, x0, x1, y0, y1
            )
            arc = circle_rectangle_arc_length(
                cx, cy, r, x0, x1, y0, y1
            )

            if area <= GEOM_TOL and arc <= GEOM_TOL:
                continue

            if area > GEOM_TOL:
                fiber_area_map[ix, iy, iz] += area
                assigned_area_um2 += area

            if arc > GEOM_TOL:
                fiber_arc_map[ix, iy, iz] += arc
                assigned_arc_um += arc

            n_voxels += 1

    full_area = np.pi * r * r
    full_arc = 2.0 * np.pi * r

    return {
        "inside": True,
        "corner_case": corner_case,
        "assigned_area_um2": assigned_area_um2,
        "assigned_arc_um": assigned_arc_um,
        "area_fraction": assigned_area_um2 / full_area,
        "arc_fraction": assigned_arc_um / full_arc,
        "n_voxels": n_voxels,
    }


# ============================================================
# Load files
# ============================================================

df = pd.read_csv(centroid_csv)

gre_nii = nib.load(gre_path)
diff_nii = nib.load(diffusion_path)

gre_affine = gre_nii.affine
diff_affine = diff_nii.affine
inv_diff_affine = np.linalg.inv(diff_affine)

gre_shape = gre_nii.shape
diff_shape = diff_nii.shape
count_shape = make_3d_shape(diff_shape)

print("============================================================")
print("Input information")
print("============================================================")
print("Centroid CSV:", centroid_csv)
print("GRE image:", gre_path)
print("Diffusion image:", diffusion_path)
print()
print("GRE shape:", gre_shape)
print("Diffusion shape:", diff_shape)
print("Output shape:", count_shape)
print()
print("GRE affine:")
print(gre_affine)
print()
print("Diffusion affine:")
print(diff_affine)
print()


# ============================================================
# Check CSV columns
# ============================================================

required_cols = ["centroid_y", "centroid_x"]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"CSV does not contain required column: {col}")

print("CSV columns:", list(df.columns))
print("Number of centroids:", len(df))
print()


# ============================================================
# Read diffusion voxel size
# ============================================================

print("============================================================")
print("Read diffusion voxel size from header")
print("============================================================")

voxel_size_x_um, voxel_size_y_um, spatial_unit = (
    get_xy_voxel_size_um_from_nifti(diff_nii)
)

voxel_area_um2 = voxel_size_x_um * voxel_size_y_um
circle_area_um2 = np.pi * fiber_radius_um ** 2
circle_arc_um = 2.0 * np.pi * fiber_radius_um

print()
print("Voxel size used for S/V calculation:")
print(f"voxel_size_x_um = {voxel_size_x_um}")
print(f"voxel_size_y_um = {voxel_size_y_um}")
print(f"voxel_area_um2 = {voxel_area_um2}")
print()
print("Fiber geometry:")
print(f"fiber_radius_um = {fiber_radius_um}")
print(f"fiber_area_um2 = {circle_area_um2}")
print(f"fiber_circumference_um = {circle_arc_um}")
print()

if not (1.0 <= voxel_size_x_um <= 500.0 and 1.0 <= voxel_size_y_um <= 500.0):
    print("WARNING:")
    print("Converted voxel size is outside the expected micrometer range.")
    print("Please check the NIfTI header unit and pixdim.")
    print()

if fiber_radius_um >= 0.5 * min(voxel_size_x_um, voxel_size_y_um):
    print("WARNING:")
    print("Fiber radius is >= half the diffusion voxel size.")
    print("A fiber may overlap more than four voxels; the exact method still works.")
    print()


# ============================================================
# Exact assignment of fiber area and circumference
# ============================================================

fiber_area_map_um2 = np.zeros(count_shape, dtype=np.float64)
fiber_arc_map_um = np.zeros(count_shape, dtype=np.float64)

outside_count = 0
corner_case_count = 0
debug_rows = []

sum_area_fraction = 0.0
sum_arc_fraction = 0.0

for idx, row in df.iterrows():
    y_gre = float(row["centroid_y"])
    x_gre = float(row["centroid_x"])

    x_diff_f, y_diff_f, z_diff_f, world_h = gre_centroid_to_diffusion_index(
        x_gre=x_gre,
        y_gre=y_gre,
        gre_affine=gre_affine,
        inv_diff_affine=inv_diff_affine,
    )

    info = assign_fiber_exact(
        fiber_area_map=fiber_area_map_um2,
        fiber_arc_map=fiber_arc_map_um,
        x_center_voxel=x_diff_f,
        y_center_voxel=y_diff_f,
        z_center_voxel=z_diff_f,
        voxel_size_x_um=voxel_size_x_um,
        voxel_size_y_um=voxel_size_y_um,
        fiber_radius_um=fiber_radius_um,
    )

    if not info["inside"]:
        outside_count += 1
    else:
        sum_area_fraction += info["area_fraction"]
        sum_arc_fraction += info["arc_fraction"]

    if info["corner_case"]:
        corner_case_count += 1

    if idx < 20:
        debug_rows.append({
            "label": row["label"] if "label" in row else idx + 1,
            "centroid_x_GRE": x_gre,
            "centroid_y_GRE": y_gre,
            "world_x": float(world_h[0]),
            "world_y": float(world_h[1]),
            "world_z": float(world_h[2]),
            "diff_x_float": x_diff_f,
            "diff_y_float": y_diff_f,
            "diff_z_float": z_diff_f,
            "inside": info["inside"],
            "corner_case": info["corner_case"],
            "n_overlapping_voxels": info["n_voxels"],
            "assigned_area_um2": info["assigned_area_um2"],
            "assigned_arc_um": info["assigned_arc_um"],
            "area_fraction": info["area_fraction"],
            "arc_fraction": info["arc_fraction"],
        })


# ============================================================
# Assignment summary
# ============================================================

inside_count = len(df) - outside_count

print("============================================================")
print("Exact geometry assignment summary")
print("============================================================")
print("Total centroids:", len(df))
print("Centroids inside diffusion grid:", inside_count)
print("Centroids outside diffusion grid:", outside_count)
print("Corner-crossing fibers:", corner_case_count)
print()

if inside_count > 0:
    print("Mean assigned area fraction per inside fiber:", sum_area_fraction / inside_count)
    print("Mean assigned arc fraction per inside fiber:", sum_arc_fraction / inside_count)
    print()
    print("For fibers not touching the OUTER image boundary, both values should be ~1.")
    print("Values below 1 can occur when part of a fiber lies outside the diffusion image.")
    print()

debug_df = pd.DataFrame(debug_rows)
debug_df.to_csv(output_debug_csv, index=False, float_format="%.8f")
print("Saved debug table:")
print(output_debug_csv)
print()


# ============================================================
# Compute maps
# ============================================================

# Area-weighted fiber count is retained only as a convenient diagnostic map.
# It is NOT used to calculate S.
fiber_area_weighted_count = fiber_area_map_um2 / circle_area_um2

# Unit depth h = 1:
# S [um^2] numerically equals circumference arc length [um] * 1.
# V [um^3] numerically equals free cross-sectional area [um^2] * 1.
# Therefore S/V has units 1/um.
S_map_um = fiber_arc_map_um
V_map_um2 = voxel_area_um2 - fiber_area_map_um2

SV_map_1_per_um = np.zeros(count_shape, dtype=np.float64)

has_fiber = fiber_area_map_um2 > GEOM_TOL
valid_sv = has_fiber & (V_map_um2 > GEOM_TOL)

SV_map_1_per_um[valid_sv] = S_map_um[valid_sv] / V_map_um2[valid_sv]

invalid_v_count = int(np.sum(has_fiber & (V_map_um2 <= GEOM_TOL)))
if invalid_v_count > 0:
    print("WARNING:")
    print(f"{invalid_v_count} non-empty voxels have V <= 0.")
    print("Their S/V values were left as 0. Check segmentation overlap or geometry.")
    print()

print("============================================================")
print("S/V calculation")
print("============================================================")
print("Non-empty voxels:", int(np.sum(has_fiber)))
print("Valid S/V voxels:", int(np.sum(valid_sv)))

if np.any(valid_sv):
    vals = SV_map_1_per_um[valid_sv]
    print(f"S/V min = {vals.min():.8f} um^-1")
    print(f"S/V max = {vals.max():.8f} um^-1")
    print(f"S/V mean = {vals.mean():.8f} um^-1")
    print(f"S/V median = {np.median(vals):.8f} um^-1")
print()


# ============================================================
# Save NIfTI maps
# ============================================================

out_header = diff_nii.header.copy()

def save_map(data, path):
    nii = nib.Nifti1Image(data.astype(np.float32), diff_affine, out_header)
    nii.set_data_dtype(np.float32)
    nib.save(nii, path)
    print("Saved:", path)

save_map(fiber_area_weighted_count, output_count_path)
save_map(S_map_um, output_S_path)
save_map(fiber_area_map_um2, output_fiber_area_path)
save_map(V_map_um2, output_V_path)
save_map(SV_map_1_per_um, output_sv_path)

print()


# ============================================================
# Save voxel-wise CSV
# ============================================================

rows = []
nonzero_indices = np.argwhere(has_fiber)

for x, y, z in nonzero_indices:
    rows.append({
        "diffusion_voxel_x": int(x),
        "diffusion_voxel_y": int(y),
        "diffusion_voxel_z": int(z),
        "area_weighted_fiber_count": float(fiber_area_weighted_count[x, y, z]),
        "fiber_area_um2": float(fiber_area_map_um2[x, y, z]),
        "S_um": float(S_map_um[x, y, z]),
        "V_um2": float(V_map_um2[x, y, z]),
        "S_over_V_um-1": float(SV_map_1_per_um[x, y, z]),
    })

count_table = pd.DataFrame(rows)
count_table.to_csv(output_csv, index=False, float_format="%.6f")

print("Saved voxel-wise table:")
print(output_csv)
print()


# ============================================================
# Plot S/V histograms
# ============================================================

if len(count_table) > 0:
    sv_values = count_table["S_over_V_um-1"].values
    sv_values = sv_values[np.isfinite(sv_values)]
    sv_values = sv_values[sv_values >= 0]

    if len(sv_values) == 0:
        print("No valid S/V values found, so histograms were not generated.")
    else:
        def save_sv_histogram(
            values,
            output_png,
            x_min,
            x_max,
            bin_width,
            x_tick_interval,
            tick_format,
            title_suffix,
        ):
            bins = np.arange(x_min, x_max + bin_width, bin_width)

            n_total = len(values)
            n_inside = np.sum((values >= x_min) & (values <= x_max))
            n_above = np.sum(values > x_max)

            print(f"Histogram range {x_min}-{x_max}:")
            print(f"values inside range: {n_inside}/{n_total}")
            print(f"values above {x_max}: {n_above}")
            print()

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(values, bins=bins, edgecolor="black")
            ax.set_xlabel("S/V ($\\mu$m$^{-1}$)")
            ax.set_ylabel("Voxel count")
            ax.set_title(f"Histogram of GRE-derived S/V {title_suffix}")

            x_ticks = np.arange(
                x_min,
                x_max + 0.5 * x_tick_interval,
                x_tick_interval
            )

            ax.set_xlim(x_min, x_max)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(
                [format(x, tick_format) for x in x_ticks],
                fontsize=9,
            )

            if n_above > 0:
                ax.text(
                    0.98,
                    0.95,
                    f"{n_above} values > {x_max}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=9,
                )

            plt.tight_layout()
            plt.savefig(output_png, dpi=300)
            plt.close()

            print("Saved histogram:")
            print(output_png)
            print()

        output_sv_hist_png_015 = output_sv_hist_png.replace(
            ".png",
            "_range_0_0p15.png"
        )

        output_sv_hist_png_15 = output_sv_hist_png.replace(
            ".png",
            "_range_0_1p5.png"
        )

        save_sv_histogram(
            values=sv_values,
            output_png=output_sv_hist_png_015,
            x_min=0.0,
            x_max=0.15,
            bin_width=0.0005,
            x_tick_interval=0.02,
            tick_format=".2f",
            title_suffix="(range 0-0.15)",
        )

        save_sv_histogram(
            values=sv_values,
            output_png=output_sv_hist_png_15,
            x_min=0.0,
            x_max=1.5,
            bin_width=0.005,
            x_tick_interval=0.2,
            tick_format=".1f",
            title_suffix="(range 0-1.5)",
        )

else:
    print("No non-empty voxels found, so histograms were not generated.")

print("Done.")
