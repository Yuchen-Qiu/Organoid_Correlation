import nibabel as nib
import numpy as np

# ============================================================
# 参数设置
# ============================================================

seg_path = "/autofs/arch11/DATA/HOMES/yuchen/phantom/phantom_25um/watershed_instance_mask.nii.gz"
output_sv_path = "/autofs/arch11/DATA/HOMES/yuchen/phantom/phantom_25um/SV_from_GRE.nii.gz"
output_fiber_count_path = "/autofs/arch11/DATA/HOMES/yuchen/phantom/phantom_25um/fiber_count_map.nii.gz"

# 扩散体素尺寸 (µm)
diff_voxel_size = (50.0, 50.0, 1.0)

# 纤维半径 (µm)
fiber_radius = 12.5

# ============================================================
# 1. 加载分割图
# ============================================================

print("Loading segmentation...")
seg_img = nib.load(seg_path)
seg_data = seg_img.get_fdata()
seg_affine = seg_img.affine

# 获取 GRE 体素尺寸 (mm -> µm)
seg_voxel_size_mm = seg_img.header.get_zooms()[:3]
seg_voxel_size = tuple(v * 1000 for v in seg_voxel_size_mm)  # 转换为 µm

print(f"Segmentation shape: {seg_data.shape}")
print(f"GRE voxel size (µm): {seg_voxel_size}")
print(f"Diffusion voxel size (µm): {diff_voxel_size}")

# ============================================================
# 2. 计算新网格尺寸
# ============================================================

# 每个扩散体素包含多少个 GRE 体素
downsample_factor = tuple(d / s for s, d in zip(seg_voxel_size, diff_voxel_size))
print(f"Downsample factor: {downsample_factor}")

# 新网格的 shape
new_shape = tuple(int(np.floor(seg_data.shape[i] / downsample_factor[i])) 
                  for i in range(3))
print(f"New grid shape: {new_shape}")

# ============================================================
# 3. 计算每个扩散体素的纤维数量和 S/V
# ============================================================

print("Computing fiber count and S/V for each voxel...")

fiber_count_map = np.zeros(new_shape, dtype=np.float32)
SV_map = np.zeros(new_shape, dtype=np.float32)

# V = voxelsize_x * voxelsize_y (µm²)
V = diff_voxel_size[0] * diff_voxel_size[1]

total_voxels = new_shape[0] * new_shape[1] * new_shape[2]
processed = 0

for iz in range(new_shape[2]):
    z_start = int(iz * downsample_factor[2])
    z_end = int((iz + 1) * downsample_factor[2])
    
    for iy in range(new_shape[1]):
        y_start = int(iy * downsample_factor[1])
        y_end = int((iy + 1) * downsample_factor[1])
        
        for ix in range(new_shape[0]):
            x_start = int(ix * downsample_factor[0])
            x_end = int((ix + 1) * downsample_factor[0])
            
            # 提取子块
            block = seg_data[x_start:x_end, y_start:y_end, z_start:z_end]
            
            # 统计唯一纤维数量（排除背景 0）
            unique_labels = np.unique(block)
            unique_labels = unique_labels[unique_labels != 0]
            n_fibers = len(unique_labels)
            
            fiber_count_map[ix, iy, iz] = n_fibers
            
            # S = 2π * r * n (总周长)
            S = 2 * np.pi * fiber_radius * n_fibers
            
            # S/V
            SV_map[ix, iy, iz] = S / V
            
            processed += 1
    
    # 进度显示
    progress = (iz + 1) / new_shape[2] * 100
    print(f"Progress: {progress:.1f}% (slice {iz + 1}/{new_shape[2]})")

print(f"Done. Processed {processed} voxels.")

# ============================================================
# 4. 统计信息
# ============================================================

print("\n--- Statistics ---")
print(f"Fiber count - min: {fiber_count_map.min()}, max: {fiber_count_map.max()}, "
      f"mean: {fiber_count_map.mean():.2f}")
print(f"S/V - min: {SV_map.min():.4f}, max: {SV_map.max():.4f}, "
      f"mean: {SV_map.mean():.4f}")

# 非零体素统计
nonzero_mask = fiber_count_map > 0
print(f"Voxels with fibers: {nonzero_mask.sum()} / {fiber_count_map.size} "
      f"({nonzero_mask.sum() / fiber_count_map.size * 100:.1f}%)")

# ============================================================
# 5. 保存结果
# ============================================================

# 创建新的 affine（更新体素尺寸）
new_affine = seg_affine.copy()
for i in range(3):
    # 缩放体素尺寸
    new_affine[i, i] = seg_affine[i, i] * downsample_factor[i]

# 保存 S/V 图
SV_img = nib.Nifti1Image(SV_map, affine=new_affine)
nib.save(SV_img, output_sv_path)
print(f"\nSaved: {output_sv_path}")

# 保存纤维计数图
fiber_count_img = nib.Nifti1Image(fiber_count_map, affine=new_affine)
nib.save(fiber_count_img, output_fiber_count_path)
print(f"Saved: {output_fiber_count_path}")

# ============================================================
# 6. 验证输出
# ============================================================

print("\n--- Output verification ---")
test_img = nib.load(output_sv_path)
print(f"S/V map shape: {test_img.shape}")
print(f"S/V map voxel size (mm): {test_img.header.get_zooms()[:3]}")
