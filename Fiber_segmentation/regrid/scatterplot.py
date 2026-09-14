import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


# ============================================================
# Paths
# ============================================================

gre_csv_path = "/autofs/arch11/DATA/HOMES/yuchen/phantom/phantom_25um/sv_output/sv_values.csv"

diffusion_csv_path = (
    "/autofs/arch11/DATA/HOMES/yuchen/phantom/phantom_25um/"
    "fiber_count_S_V_SoverV_on_diffusion_grid.csv"
)

output_dir = "/autofs/arch11/DATA/HOMES/yuchen/phantom/scatter_output"
os.makedirs(output_dir, exist_ok=True)

merged_csv_out = os.path.join(output_dir, "merged_GRE_diffusion_SV_values.csv")
scatter_png_out = os.path.join(output_dir, "GRE_vs_diffusion_SV_scatter.png")


# ============================================================
# Load CSV files
# ============================================================

gre_df = pd.read_csv(gre_csv_path)
diff_df = pd.read_csv(diffusion_csv_path)

print("GRE table columns:")
print(gre_df.columns)

print("\nDiffusion table columns:")
print(diff_df.columns)


# ============================================================
# Standardize GRE columns
# ============================================================

gre_df = gre_df.rename(columns={
    "S_over_V_um-1": "GRE_S_over_V_1_per_um"
})

if "voxel_z" not in gre_df.columns:
    gre_df["voxel_z"] = 0


# ============================================================
# Standardize diffusion columns
# ============================================================

diff_df = diff_df.rename(columns={
    "diffusion_voxel_x": "voxel_x",
    "diffusion_voxel_y": "voxel_y",
    "S_over_V_1_per_um": "Diffusion_S_over_V_1_per_um"
})

# Your diffusion table is 2D, so add z = 0
if "voxel_z" not in diff_df.columns:
    diff_df["voxel_z"] = 0


# ============================================================
# Make voxel coordinates integer
# ============================================================

for col in ["voxel_x", "voxel_y", "voxel_z"]:
    gre_df[col] = gre_df[col].astype(int)
    diff_df[col] = diff_df[col].astype(int)


# ============================================================
# Merge by voxel coordinates
# ============================================================

merge_keys = ["voxel_x", "voxel_y", "voxel_z"]

merged_df = pd.merge(
    gre_df,
    diff_df,
    on=merge_keys,
    how="inner"
)

print("\nNumber of matched voxels:", len(merged_df))

if len(merged_df) == 0:
    raise ValueError(
        "No matched voxels found. Check whether the two tables use the same voxel coordinates."
    )


# ============================================================
# Select valid values
# ============================================================

x_col = "GRE_S_over_V_1_per_um"
y_col = "Diffusion_S_over_V_1_per_um"

valid_df = merged_df[
    merged_df[x_col].notna() &
    merged_df[y_col].notna() &
    (merged_df[x_col] > 0) &
    (merged_df[y_col] > 0)
].copy()

print("Number of valid voxels after filtering:", len(valid_df))

if len(valid_df) == 0:
    raise ValueError("No valid non-zero values found after filtering.")

x = valid_df[x_col].values
y = valid_df[y_col].values


# ============================================================
# Correlation
# ============================================================

if len(valid_df) >= 2:
    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)
else:
    pearson_r, pearson_p = float("nan"), float("nan")
    spearman_r, spearman_p = float("nan"), float("nan")

print(f"Pearson r = {pearson_r:.4f}, p = {pearson_p:.4e}")
print(f"Spearman r = {spearman_r:.4f}, p = {spearman_p:.4e}")


# ============================================================
# Save merged table
# ============================================================

valid_df.to_csv(merged_csv_out, index=False)

print("Saved merged CSV:")
print(merged_csv_out)


# ============================================================
# Plot scatter
# ============================================================

plt.figure(figsize=(7, 6))

plt.scatter(x, y, s=18, alpha=0.7)

plt.xlabel("GRE-derived S/V (µm⁻¹)")
plt.ylabel("Diffusion-derived S/V (µm⁻¹)")
plt.title("GRE-derived S/V vs Diffusion-derived S/V")

plt.grid(True, alpha=0.3)

text = (
    f"n = {len(valid_df)}\n"
    f"Pearson r = {pearson_r:.3f}\n"
    f"Spearman r = {spearman_r:.3f}"
)

plt.text(
    0.05,
    0.95,
    text,
    transform=plt.gca().transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", alpha=0.2)
)

plt.tight_layout()
plt.savefig(scatter_png_out, dpi=300)
plt.show()

print("Saved scatter plot:")
print(scatter_png_out)