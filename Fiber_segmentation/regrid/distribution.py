import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# ============================================================
# Paths
# ============================================================

nii_path = "/autofs/arch11/DATA/HOMES/yuchen/phantom/phantom_25um/D0_newNIFTI.nii"

output_dir = "/autofs/arch11/DATA/HOMES/yuchen/phantom/phantom_25um/output_histogram"
os.makedirs(output_dir, exist_ok=True)

hist_png_out = os.path.join(output_dir, "value_histogram.png")
hist_csv_out = os.path.join(output_dir, "value_distribution.csv")


# ============================================================
# Load NIfTI image
# ============================================================

nii = nib.load(nii_path)
data = nii.get_fdata()

print("Image shape:", data.shape)
print("Min:", np.nanmin(data))
print("Max:", np.nanmax(data))
print("Mean:", np.nanmean(data))
print("Median:", np.nanmedian(data))


# ============================================================
# Flatten values
# ============================================================

values = data.flatten()

# Remove NaN / inf
values = values[np.isfinite(values)]

# Optional: remove zero background
remove_zero = True

if remove_zero:
    values = values[values != 0]

print("Number of voxels used:", len(values))
print("Min after filtering:", values.min())
print("Max after filtering:", values.max())


# ============================================================
# Plot histogram
# ============================================================

plt.figure(figsize=(8, 5))

counts, bin_edges, _ = plt.hist(
    values,
    bins=50,
    edgecolor="black"
)

plt.xlabel("Voxel value")
plt.ylabel("Count")
plt.title("Histogram of Image Voxel Values")

plt.tight_layout()
plt.savefig(hist_png_out, dpi=300)
plt.close()

print("Saved histogram to:", hist_png_out)


# ============================================================
# Save histogram data to CSV
# ============================================================

bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

hist_data = np.column_stack([
    bin_edges[:-1],
    bin_edges[1:],
    bin_centers,
    counts
])

np.savetxt(
    hist_csv_out,
    hist_data,
    delimiter=",",
    header="bin_left,bin_right,bin_center,count",
    comments=""
)

print("Saved histogram CSV to:", hist_csv_out)