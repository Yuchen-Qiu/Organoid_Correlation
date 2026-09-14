import numpy as np
import nibabel as nib
from collections import deque
from scipy import ndimage as ndi
from skimage.measure import label


def save_nii(data, out_path, reference_nii, dtype=None):
    arr = np.asarray(data)

    if dtype is not None:
        arr = arr.astype(dtype)

    if arr.ndim == 2:
        arr = arr[:, :, None]

    affine = reference_nii.affine
    header = reference_nii.header.copy()

    out_nii = nib.Nifti1Image(arr, affine, header)

    if dtype is not None:
        out_nii.set_data_dtype(dtype)

    nib.save(out_nii, out_path)


def load_nii_as_2d(path):
    nii_obj = nib.load(path)
    data = nii_obj.get_fdata()

    if data.ndim == 3 and data.shape[2] == 1:
        data = data[:, :, 0]
    elif data.ndim == 2:
        pass
    else:
        raise ValueError(f"Expected 2D image or 3D image with one slice, got {data.shape}")

    return data, nii_obj


def read_and_normalize_image(input_path):
    nii = nib.load(input_path)
    img_raw = nii.get_fdata().astype(np.float32)

    print("Original shape:", img_raw.shape)

    if img_raw.ndim == 3 and img_raw.shape[2] == 1:
        img = img_raw[:, :, 0]
    elif img_raw.ndim == 2:
        img = img_raw
    else:
        raise ValueError(f"Expected 2D image or 3D image with one slice, got {img_raw.shape}")

    img = img - np.nanmin(img)
    img = img / (np.nanmax(img) + 1e-8)

    return img, nii


def prepare_manual_markers(marker_path):
    markers_data, _ = load_nii_as_2d(marker_path)
    markers_data = markers_data.astype(np.int32)

    unique_vals = np.unique(markers_data)
    unique_vals = unique_vals[unique_vals > 0]

    if len(unique_vals) <= 1:
        print("Manual markers look like binary seed mask. Relabeling...")
        markers = label(markers_data > 0).astype(np.int32)
    else:
        print("Manual markers look like labeled seed mask. Keeping labels...")
        markers = markers_data.astype(np.int32)

    print("Number of seed labels:", markers.max())
    return markers


def region_growing_from_markers(
    img,
    markers,
    mask,
    intensity_tolerance=0.12,
    max_radius=8,
    use_region_mean=True,
):
    # Important: initialize seed pixels first
    labels_rg = markers.astype(np.int32).copy()

    seed_ids = np.unique(markers)
    seed_ids = seed_ids[seed_ids > 0]

    h, w = img.shape

    for seed_id in seed_ids:
        seed_pos = np.argwhere(markers == seed_id)

        if len(seed_pos) == 0:
            continue

        sy = int(round(seed_pos[:, 0].mean()))
        sx = int(round(seed_pos[:, 1].mean()))

        seed_pixels_y = seed_pos[:, 0]
        seed_pixels_x = seed_pos[:, 1]
        seed_intensity = float(np.mean(img[seed_pixels_y, seed_pixels_x]))

        visited = np.zeros_like(mask, dtype=bool)
        queue = deque()

        # Start from this seed's own pixels
        region_sum = float(np.sum(img[seed_pixels_y, seed_pixels_x]))
        region_count = len(seed_pos)

        for py, px in seed_pos:
            py = int(py)
            px = int(px)

            if 0 <= py < h and 0 <= px < w:
                queue.append((py, px))
                visited[py, px] = True

        while queue:
            y, x = queue.popleft()

            if not mask[y, x]:
                continue

            dist_from_seed = np.sqrt((y - sy) ** 2 + (x - sx) ** 2)
            if dist_from_seed > max_radius:
                continue

            # Do not enter pixels already owned by another seed/object
            if labels_rg[y, x] != 0 and labels_rg[y, x] != seed_id:
                continue

            # Do not enter another seed's original marker pixel
            if markers[y, x] != 0 and markers[y, x] != seed_id:
                continue

            if use_region_mean and region_count > 0:
                reference_intensity = region_sum / region_count
            else:
                reference_intensity = seed_intensity

            if abs(float(img[y, x]) - reference_intensity) > intensity_tolerance:
                continue

            # Assign current pixel to this seed
            if labels_rg[y, x] == 0:
                labels_rg[y, x] = seed_id
                region_sum += float(img[y, x])
                region_count += 1

            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue

                    ny = y + dy
                    nx = x + dx

                    if 0 <= ny < h and 0 <= nx < w:
                        if not visited[ny, nx]:
                            # Do not even queue another seed's marker pixel
                            if markers[ny, nx] != 0 and markers[ny, nx] != seed_id:
                                continue

                            visited[ny, nx] = True
                            queue.append((ny, nx))

    return labels_rg