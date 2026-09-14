import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import nibabel as nib
# ============================================================
# Paths
# ============================================================
work_path = "/autofs/arch11/DATA/HOMES/yuchen/phantom/phantom_25um/SV_results_weighted_V_adjusted_zero"
csv_file = os.path.join(work_path, "fiber_weighted_count_S_V_SoverV_on_diffusion_grid.csv")

df = pd.read_csv(csv_file)

# ============================================================
# Column names
# ============================================================
x_col = "diffusion_voxel_y"
y_col = "diffusion_voxel_x"
value_col = "S_over_V_1_per_um"

# ============================================================
# Build 2D image from voxel coordinates
# ============================================================
x = df[x_col].astype(int)
y = df[y_col].astype(int)
values = df[value_col].astype(float)

print("S/V min:", values.min())
print("S/V max:", values.max())
print("S/V mean:", values.mean())
print("S/V median:", values.median())

# one fiber expected S/V
one_fiber_sv = 2 * np.pi * 12.5 / (50 * 50)
print("Expected S/V for one fiber: {:.4f} um^-1".format(one_fiber_sv))

nx = x.max() + 1
ny = y.max() + 1

sv_map = np.full((ny, nx), np.nan)

for xi, yi, val in zip(x, y, values):
    sv_map[yi, xi] = val

# ============================================================
# Custom colormap
# Range is intended for 0 ~ 0.15 um^-1
#
# Approximate mapping:
# 0.00   -> light cyan
# 0.0225 -> cyan
# 0.0375 -> blue      (~ around 1 fiber: 0.0314)
# 0.0600 -> black
# 0.0900 -> red
# 0.1200 -> yellow
# 0.1500 -> white
# ============================================================
sv_cmap = LinearSegmentedColormap.from_list(
    "sv_cmap",
    [
        (0.00, "#c8ffff"),  # light cyan
        (0.15, "#66ffff"),  # cyan
        (0.25, "#0070ff"),  # blue
        (0.40, "#000000"),  # black
        (0.60, "#cc0000"),  # red
        (0.80, "#ffff00"),  # yellow
        (1.00, "#ffffff"),  # white
    ]
)

# background color for NaN
sv_cmap.set_bad(color="#c8ffff")

# ============================================================
# Plot
# ============================================================
sv_map_plot=sv_map
sv_map_plot = np.rot90(sv_map, k=-1)  # clockwise 90 degrees

fig, ax = plt.subplots(figsize=(8, 8))

im = ax.imshow(
    sv_map_plot,
    cmap=sv_cmap,
    origin="upper",
    interpolation="nearest",
    vmin=0,
    vmax=0.25
)

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r"$S/V,\ \mu m^{-1}$", fontsize=12)
cbar.set_ticks(np.arange(0, 0.26, 0.05))
cbar.ax.tick_params(labelsize=10)
ax.axis("off")
ax.set_aspect("equal")

plt.tight_layout()

output_path = os.path.join(work_path, "SV_map_0_to_0p150yx.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print("Saved figure to:", output_path)

plt.show(block=True)

# ============================================================
# 保存为 NIfTI 格式 (.nii.gz)
# ============================================================
# 1. 采用与画图一致的旋转矩阵 (sv_map_plot)
# 2. NIfTI 通常要求至少是 3D 的，所以通过增加一个切片维度将其从 (H, W) 拓展为 (H, W, 1)
nifti_data = np.expand_dims(sv_map_plot, axis=2)

# 【可选步骤】如果你使用的软件无法很好地显示 NaN 背景，可以将下面这行代码取消注释，把 NaN 替换为 0
# nifti_data = np.isnan(nifti_data, 0)

# 创建一个 4x4 的标准仿射矩阵 (单位矩阵)
# 如果你需要将该热图与某个特定的扩散核磁(dMRI)图像在空间上完美对齐，
# 建议通过 nib.load('dmri.nii.gz').affine 读取原图的仿射矩阵并替换下方的 np.eye(4)
affine = np.eye(4)

# 转换成 NIfTI 图像对象并保存
nifti_img = nib.Nifti1Image(nifti_data, affine)
nifti_output_path = os.path.join(work_path, "SV_map.nii.gz")
nib.save(nifti_img, nifti_output_path)
print("Saved NIfTI to:", nifti_output_path)

# ============================================================

plt.show(block=True)