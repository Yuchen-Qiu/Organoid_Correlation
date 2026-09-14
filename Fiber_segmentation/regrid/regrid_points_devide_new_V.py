import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

# ============================================================
# Paths
# ============================================================

# Main working folder
work_path = "/autofs/arch11/DATA/HOMES/yuchen/phantom/phantom_25um"

# Output folder
output_path = os.path.join(work_path, "SV_results_weighted_V_adjusted")
os.makedirs(output_path, exist_ok=True)

# Fiber centroid CSV
centroid_csv = os.path.join(
    work_path,
    "watershed_centroids.csv"
)

# GRE / segmentation NIfTI used to generate centroid_x / centroid_y
# This must be the same image space as the centroid CSV.
gre_path = "/autofs/arch11/DATA/HOMES/yuchen/phantom/5mmIprobe_3mm_phantom.nii"

# Diffusion grid template
# The goal is to count how much fiber cross-section belongs to each diffusion voxel.
diffusion_path = os.path.join(
    work_path,
    "tn28T_260428_4_5mmIprobe_3mm_phantom2_20C_56_1_30_adjusted.nii"
)

# Output maps
output_count_path = os.path.join(
    output_path,
    "fiber_weighted_count_on_diffusion_grid.nii.gz"
)

output_sv_path = os.path.join(
    output_path,
    "gre_derived_S_over_V_1_per_um_on_diffusion_grid.nii.gz"
)

output_S_path = os.path.join(
    output_path,
    "gre_derived_S_um_on_diffusion_grid.nii.gz"
)

output_V_path = os.path.join(
    output_path,
    "gre_derived_V_um2_on_diffusion_grid.nii.gz"
)

output_count_csv = os.path.join(
    output_path,
    "fiber_weighted_count_S_V_SoverV_on_diffusion_grid.csv"
)

output_sv_hist_png = os.path.join(
    output_path,
    "gre_derived_S_over_V_histogram.png"
)

# ============================================================
# Parameters for S/V
# ============================================================

# Fiber radius.
# IMPORTANT:
# 12.5 is in micrometers, not mm and not voxel units.
fiber_radius_um = 12.5


# ============================================================
# Helper functions
# ============================================================

def get_xy_voxel_size_um_from_nifti(nii):
    """
    Read x/y voxel size from NIfTI header and convert to micrometers.

    Returns:
        voxel_size_x_um
        voxel_size_y_um
        spatial_unit
    """

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

    # Convert header spacing to micrometers
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
        print()
        print("WARNING:")
        print("NIfTI spatial unit is unknown.")
        print("By default, this script assumes the header spacing is in mm.")
        print("Therefore spacing will be multiplied by 1000 to convert to um.")
        print("Please check this carefully.")
        print()

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
    Convert one GRE centroid from GRE pixel coordinates to diffusion voxel index.

    CSV centroid:
        centroid_y, centroid_x from skimage regionprops

        centroid_y = row coordinate
        centroid_x = column coordinate

    Nibabel voxel coordinate:
        [i, j, k] = [x, y, z]

    Therefore:
        gre_voxel = [centroid_x, centroid_y, 0]
    """

    # GRE voxel coordinate
    gre_voxel_h = np.array([x_gre, y_gre, 0.0, 1.0], dtype=np.float64)

    # GRE voxel -> world coordinate
    world_h = gre_affine @ gre_voxel_h

    # World coordinate -> diffusion continuous voxel coordinate
    diff_voxel_h = inv_diff_affine @ world_h

    x_diff_f = float(diff_voxel_h[0])
    y_diff_f = float(diff_voxel_h[1])
    z_diff_f = float(diff_voxel_h[2])

    return x_diff_f, y_diff_f, z_diff_f, world_h


def make_3d_shape(shape):
    """
    Convert 2D or 3D image shape to 3D output shape.
    """
    if len(shape) == 2:
        return (shape[0], shape[1], 1)
    elif len(shape) == 3:
        return shape
    else:
        raise ValueError(f"Unsupported image dimension: {shape}")


def circle_segment_fraction(distance_to_boundary_um, radius_um):
    """
    Compute the fraction of a circle area crossing one straight boundary.

    distance_to_boundary_um:
        Distance from the circle center to the voxel boundary, in micrometers.
        The center is assumed to be inside the current voxel, so distance >= 0.

    radius_um:
        Fiber radius in micrometers.

    Return:
        Fraction of the circle area located on the other side of that boundary.

    Cases:
        d >= r:
            no crossing, fraction = 0

        d = 0:
            circle center is exactly on the boundary,
            half of the circle is on the other side, fraction = 0.5

        0 < d < r:
            crossing fraction is calculated by circular segment area:
            A = r^2 arccos(d/r) - d sqrt(r^2 - d^2)
    """

    d = float(distance_to_boundary_um)
    r = float(radius_um)

    if d >= r:
        return 0.0

    if d <= 0:
        return 0.5

    segment_area = r * r * np.arccos(d / r) - d * np.sqrt(r * r - d * d)
    circle_area = np.pi * r * r

    return segment_area / circle_area


def assign_fiber_by_curved_boundary_fraction(
    fiber_count,
    x_center_voxel,
    y_center_voxel,
    z_center_voxel,
    voxel_size_x_um,
    voxel_size_y_um,
    fiber_radius_um,
):
    """
    Assign one fiber to diffusion voxels using circular-segment area fractions.

    The fiber is modeled as a circular cross-section with radius fiber_radius_um.

    Steps:
    1. Find the current diffusion voxel containing the fiber center.
    2. Compute the physical distance from the center to the four voxel boundaries:
       left, right, top, bottom.
    3. If distance < fiber_radius_um, the circular fiber crosses that boundary.
    4. Use the circular segment formula to calculate the fraction crossing the boundary.
    5. Add that fraction to the neighboring voxel and subtract it from the current voxel.

    Note:
    - This is exact for crossing one straight boundary.
    - If the fiber crosses two boundaries at the same time, near a voxel corner,
      this function treats the two boundary crossings independently.
      That is a practical approximation.
    """

    nx, ny, nz = fiber_count.shape

    ix = int(np.floor(x_center_voxel))
    iy = int(np.floor(y_center_voxel))

    if nz == 1:
        iz = 0
    else:
        iz = int(np.floor(z_center_voxel))

    # If the fiber center itself is outside the grid, skip this fiber.
    if not (0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz):
        return 0.0, {}, False

    # Local position within current voxel, in voxel coordinates.
    # Example:
    #   local_x = 0.2 means the center is 20% from the left boundary.
    #   local_x = 0.8 means the center is 20% from the right boundary.
    local_x = x_center_voxel - ix
    local_y = y_center_voxel - iy

    # Convert distances to four boundaries into micrometers.
    dist_left_um = local_x * voxel_size_x_um
    dist_right_um = (1.0 - local_x) * voxel_size_x_um
    dist_top_um = local_y * voxel_size_y_um
    dist_bottom_um = (1.0 - local_y) * voxel_size_y_um

    # Compute crossing fractions using circular segment area.
    frac_left = circle_segment_fraction(dist_left_um, fiber_radius_um)
    frac_right = circle_segment_fraction(dist_right_um, fiber_radius_um)
    frac_top = circle_segment_fraction(dist_top_um, fiber_radius_um)
    frac_bottom = circle_segment_fraction(dist_bottom_um, fiber_radius_um)

    weights = {}

    # Start from the assumption that the whole fiber belongs to the current voxel.
    weights[(ix, iy, iz)] = 1.0

    # Move fractions to neighboring voxels.
    # If the neighbor is outside the image grid, keep the fraction in the current voxel.
    # This avoids losing total weight at image borders.

    if frac_left > 0 and ix - 1 >= 0:
        weights[(ix, iy, iz)] -= frac_left
        weights[(ix - 1, iy, iz)] = weights.get((ix - 1, iy, iz), 0.0) + frac_left

    if frac_right > 0 and ix + 1 < nx:
        weights[(ix, iy, iz)] -= frac_right
        weights[(ix + 1, iy, iz)] = weights.get((ix + 1, iy, iz), 0.0) + frac_right

    if frac_top > 0 and iy - 1 >= 0:
        weights[(ix, iy, iz)] -= frac_top
        weights[(ix, iy - 1, iz)] = weights.get((ix, iy - 1, iz), 0.0) + frac_top

    if frac_bottom > 0 and iy + 1 < ny:
        weights[(ix, iy, iz)] -= frac_bottom
        weights[(ix, iy + 1, iz)] = weights.get((ix, iy + 1, iz), 0.0) + frac_bottom

    # Detect corner cases:
    # crossing one x boundary and one y boundary at the same time.
    # These are approximate in this simplified curved-boundary method.
    crosses_x = (frac_left > 0) or (frac_right > 0)
    crosses_y = (frac_top > 0) or (frac_bottom > 0)
    corner_case = crosses_x and crosses_y

    # Numerical cleanup.
    # Small negative values may happen due to floating point or corner approximation.
    for key in list(weights.keys()):
        if abs(weights[key]) < 1e-8:
            weights[key] = 0.0

    # If approximation creates a negative current weight, clip it.
    # This should be rare when fiber radius is smaller than half voxel size.
    for key in list(weights.keys()):
        if weights[key] < 0:
            weights[key] = 0.0

    # Normalize weights so each fiber contributes total weight 1.0.
    weight_sum = sum(weights.values())

    if weight_sum <= 0:
        return 0.0, {}, corner_case

    for key in list(weights.keys()):
        weights[key] = weights[key] / weight_sum

    assigned_weight = 0.0

    for (x_idx, y_idx, z_idx), weight in weights.items():
        if weight <= 0:
            continue

        fiber_count[x_idx, y_idx, z_idx] += weight
        assigned_weight += weight

    return assigned_weight, weights, corner_case


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
print("Output count shape:", count_shape)
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
# Read voxel size from diffusion header and convert to um
# ============================================================

print("============================================================")
print("Read diffusion voxel size from header")
print("============================================================")

voxel_size_x_um, voxel_size_y_um, spatial_unit = get_xy_voxel_size_um_from_nifti(diff_nii)

# Following Chantal's instruction:
# z is not included because it cancels out.
# Therefore this is effectively an area term, unit um^2.
V_value_um2 = voxel_size_x_um * voxel_size_y_um

print()
print("Voxel size used for S/V calculation:")
print(f"voxel_size_x_um = {voxel_size_x_um}")
print(f"voxel_size_y_um = {voxel_size_y_um}")
print(f"V_um2 = voxel_size_x_um * voxel_size_y_um = {V_value_um2}")
print()

print("Fiber radius:")
print(f"fiber_radius_um = {fiber_radius_um}")
print(f"fiber_radius / voxel_size_x = {fiber_radius_um / voxel_size_x_um}")
print(f"fiber_radius / voxel_size_y = {fiber_radius_um / voxel_size_y_um}")
print()

# Optional sanity warning
# For your 50 x 50 um2 diffusion grid, this should be around 50 and 50.
if not (1.0 <= voxel_size_x_um <= 500.0 and 1.0 <= voxel_size_y_um <= 500.0):
    print("WARNING:")
    print("The converted voxel size is not in the expected micrometer range.")
    print("Please check the NIfTI header unit and pixdim.")
    print()

if fiber_radius_um >= 0.5 * min(voxel_size_x_um, voxel_size_y_um):
    print("WARNING:")
    print("fiber_radius_um is larger than or equal to half of the diffusion voxel size.")
    print("The simple boundary-segment approximation may become less accurate.")
    print("Consider using exact circle-rectangle overlap or sampling for this case.")
    print()


# ============================================================
# Count fibers on diffusion grid using curved-boundary fractions
# ============================================================

# IMPORTANT:
# Use float32 because each fiber can contribute fractional weights.
fiber_count = np.zeros(count_shape, dtype=np.float32)

valid_weight_sum = 0.0
outside_count = 0
corner_case_count = 0

debug_rows = []

for idx, row in df.iterrows():

    y_gre = float(row["centroid_y"])
    x_gre = float(row["centroid_x"])

    x_diff_f, y_diff_f, z_diff_f, world_h = gre_centroid_to_diffusion_index(
        x_gre=x_gre,
        y_gre=y_gre,
        gre_affine=gre_affine,
        inv_diff_affine=inv_diff_affine,
    )

    assigned_weight, assigned_voxels, corner_case = assign_fiber_by_curved_boundary_fraction(
        fiber_count=fiber_count,
        x_center_voxel=x_diff_f,
        y_center_voxel=y_diff_f,
        z_center_voxel=z_diff_f,
        voxel_size_x_um=voxel_size_x_um,
        voxel_size_y_um=voxel_size_y_um,
        fiber_radius_um=fiber_radius_um,
    )

    if assigned_weight > 0:
        valid_weight_sum += assigned_weight
    else:
        outside_count += 1

    if corner_case:
        corner_case_count += 1

    # Save first few mappings for checking
    if idx < 10:
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
            "assigned_weight": assigned_weight,
            "corner_case": corner_case,
            "assigned_voxels": str(assigned_voxels),
            "inside": assigned_weight > 0,
        })


# ============================================================
# Print counting summary
# ============================================================

print("============================================================")
print("Counting summary")
print("============================================================")
print("Total centroids:", len(df))
print("Total assigned weight to diffusion voxels:", valid_weight_sum)
print("Outside diffusion grid:", outside_count)
print("Corner-crossing cases:", corner_case_count)
print("Max weighted fiber count in one diffusion voxel:", float(fiber_count.max()))
print("Number of non-empty diffusion voxels:", int(np.sum(fiber_count > 0)))
print()

print("Check:")
print("valid_weight_sum + outside_count =", valid_weight_sum + outside_count)
print("total centroids =", len(df))
print()

print("First 10 centroid mappings for checking:")
debug_df = pd.DataFrame(debug_rows)
print(debug_df.to_string(index=False))
print()


# ============================================================
# Save weighted fiber count map
# ============================================================

# Output map is defined on the diffusion grid,
# so it should use the diffusion affine and header.
out_header = diff_nii.header.copy()

count_nii = nib.Nifti1Image(fiber_count.astype(np.float32), diff_affine, out_header)
count_nii.set_data_dtype(np.float32)
nib.save(count_nii, output_count_path)

print("Saved weighted fiber count map:")
print(output_count_path)
print()


# ============================================================
# Compute GRE-derived S, V, and S/V maps
# ============================================================

# Units:
#
# fiber_radius_um: um
# voxel_size_x_um: um
# voxel_size_y_um: um
#
# S_um = 2 * pi * fiber_radius_um * weighted_fiber_count
#      = total circumference per diffusion voxel
#      unit: um
#
# V_um2 = voxel_size_x_um * voxel_size_y_um
#       unit: um^2
#
# S_over_V_1_per_um = S_um / V_um2
#                   unit: 1/um
#
# Following Chantal's instruction:
# z is not included because it cancels out.

S_map_um = (
    2.0
    * np.pi
    * fiber_radius_um
    * fiber_count.astype(np.float32)
)

fiber_area_um2 = np.pi * fiber_radius_um ** 2

V_map_um2 = V_value_um2 - fiber_area_um2 * fiber_count.astype(np.float32)


SV_map_1_per_um = S_map_um / V_map_um2


print("============================================================")
print("S/V calculation")
print("============================================================")
print(f"fiber_radius_um = {fiber_radius_um}")
print(f"voxel_size_x_um = {voxel_size_x_um}")
print(f"voxel_size_y_um = {voxel_size_y_um}")
print(f"V_value_um2 = {V_value_um2}")
print()


# ============================================================
# Save S, V, and S/V maps
# ============================================================

S_nii = nib.Nifti1Image(S_map_um.astype(np.float32), diff_affine, out_header)
S_nii.set_data_dtype(np.float32)
nib.save(S_nii, output_S_path)

V_nii = nib.Nifti1Image(V_map_um2.astype(np.float32), diff_affine, out_header)
V_nii.set_data_dtype(np.float32)
nib.save(V_nii, output_V_path)

SV_nii = nib.Nifti1Image(SV_map_1_per_um.astype(np.float32), diff_affine, out_header)
SV_nii.set_data_dtype(np.float32)
nib.save(SV_nii, output_sv_path)

print("Saved GRE-derived S map:")
print(output_S_path)
print()

print("Saved GRE-derived V map:")
print(output_V_path)
print()

print("Saved GRE-derived S/V map:")
print(output_sv_path)
print()


# ============================================================
# Save CSV table with voxel counts, S, V, and S/V
# Only save non-empty voxels
# ============================================================

rows = []
nonzero_indices = np.argwhere(fiber_count > 0)

for x, y, z in nonzero_indices:
    count = float(fiber_count[x, y, z])

    S_um = float(S_map_um[x, y, z])
    V_um2 = float(V_map_um2[x, y, z])
    SV_1_per_um = float(SV_map_1_per_um[x, y, z])

    rows.append({
        "diffusion_voxel_x": int(x),
        "diffusion_voxel_y": int(y),
        "weighted_fiber_count": count,
        "S_um": S_um,
        "V_um2": V_um2,
        "S_over_V_1_per_um": SV_1_per_um,
    })

count_table = pd.DataFrame(rows)

# Keep 4 decimal places for float values
count_table.to_csv(output_count_csv, index=False, float_format="%.4f")

print("Saved non-empty voxel count table:")
print(output_count_csv)
print()


# ============================================================
# Plot histogram: x = binned S/V values (um^-1), y = voxel count
# ============================================================

if len(count_table) > 0:

    # ============================================================
    # Get S/V values and clean invalid values
    # ============================================================
    sv_values = count_table["S_over_V_1_per_um"].values

    # Remove NaN / inf
    sv_values = sv_values[np.isfinite(sv_values)]

    # Keep only non-negative values
    sv_values = sv_values[sv_values >= 0]

    if len(sv_values) == 0:
        print("No valid S/V values found, so S/V histograms were not generated.")

    else:
        # ============================================================
        # Print statistics once
        # ============================================================
        print("S/V statistics:")
        print(f"min = {sv_values.min():.6f}")
        print(f"max = {sv_values.max():.6f}")
        print(f"mean = {sv_values.mean():.6f}")
        print(f"median = {np.median(sv_values):.6f}")
        print()

        # ============================================================
        # Function for saving histogram
        # ============================================================
        def save_sv_histogram(
            sv_values,
            output_png,
            x_min,
            x_max,
            bin_width,
            x_tick_interval,
            tick_format,
            title_suffix
        ):
            bins = np.arange(x_min, x_max + bin_width, bin_width)

            n_total = len(sv_values)
            n_inside = np.sum((sv_values >= x_min) & (sv_values <= x_max))
            n_above = np.sum(sv_values > x_max)

            print(f"Histogram range {x_min}-{x_max}:")
            print(f"values inside range: {n_inside}/{n_total}")
            print(f"values above {x_max}: {n_above}")
            print()

            fig, ax = plt.subplots(figsize=(10, 5))

            ax.hist(
                sv_values,
                bins=bins,
                edgecolor="black"
            )

            ax.set_xlabel("S/V ($\\mu$m$^{-1}$)")
            ax.set_ylabel("Voxel count")
            ax.set_title(f"Histogram of GRE-derived S/V {title_suffix}")

            x_ticks = np.arange(x_min, x_max + x_tick_interval, x_tick_interval)

            ax.set_xlim(x_min, x_max)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(
                [format(x, tick_format) for x in x_ticks],
                rotation=0,
                fontsize=9
            )

            if n_above > 0:
                ax.text(
                    0.98,
                    0.95,
                    f"{n_above} values > {x_max}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=9
                )

            plt.tight_layout()
            plt.savefig(output_png, dpi=300)
            plt.close()

            print("Saved S/V histogram:")
            print(output_png)
            print()

        # ============================================================
        # Output paths
        # ============================================================
        output_sv_hist_png_015 = output_sv_hist_png.replace(
            ".png",
            "_range_0_0p15.png"
        )

        output_sv_hist_png_15 = output_sv_hist_png.replace(
            ".png",
            "_range_0_1p5.png"
        )

        # ============================================================
        # Histogram 1: original range 0-0.15
        # ============================================================
        save_sv_histogram(
            sv_values=sv_values,
            output_png=output_sv_hist_png_015,
            x_min=0.0,
            x_max=0.15,
            bin_width=0.0005,
            x_tick_interval=0.02,
            tick_format=".2f",
            title_suffix="(range 0–0.15)"
        )

        # ============================================================
        # Histogram 2: wider range 0-1.5
        # ============================================================
        save_sv_histogram(
            sv_values=sv_values,
            output_png=output_sv_hist_png_15,
            x_min=0.0,
            x_max=1.5,
            bin_width=0.005,
            x_tick_interval=0.2,
            tick_format=".1f",
            title_suffix="(range 0–1.5)"
        )

else:
    print("No non-empty voxels found, so S/V histograms were not generated.")

print("Done.")