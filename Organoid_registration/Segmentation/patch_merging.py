#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge currently available Cellpose patch predictions.

This script does not require component_mapping.csv. It reconstructs the
local-to-global component mapping by matching labels in overlapping patches.

Outputs:
1. generated_component_mapping.csv
2. merged_binary_mask.nii.gz
3. merged_instance_mask.nii.gz
4. component_centers_mask.nii.gz
5. component_centers.csv

Only currently available patch files are processed. Regions without a patch
prediction remain background.
"""

import os
import re
import csv
import shutil
import tempfile
from itertools import product

import nibabel as nib
import numpy as np


# ============================================================
# User settings
# ============================================================

REFERENCE_IMAGE_PATH = r"/autofs/arch11/DATA/HOMES/yuchen/Minor/Organoids/Nifti_data/case1_mature/data_new/case1_Axon_origin.nii"
SEGMENTATION_OUTPUT_DIR = r"/autofs/arch11/DATA/HOMES/yuchen/Minor/Organoids/Nifti_data/seg/case1_axon_origin_192_192_60_no_mask"


PATCH_DIR = os.path.join(SEGMENTATION_OUTPUT_DIR, "patch_predictions")
OUTPUT_DIR = os.path.join(SEGMENTATION_OUTPUT_DIR, "merged")

GENERATED_MAPPING_PATH = os.path.join(OUTPUT_DIR, "generated_component_mapping.csv")
BINARY_MASK_PATH = os.path.join(OUTPUT_DIR, "merged_binary_mask.nii.gz")
INSTANCE_MASK_PATH = os.path.join(OUTPUT_DIR, "merged_instance_mask.nii.gz")
CENTER_MASK_PATH = os.path.join(OUTPUT_DIR, "component_centers_mask.nii.gz")
CENTER_CSV_PATH = os.path.join(OUTPUT_DIR, "component_centers.csv")

# Same matching settings as the original segmentation script.
MIN_OVERLAP_VOXELS = 10
IOMIN_THRESHOLD = 0.3

REMOVE_TEMPORARY_FILES = True

MIN_POINTS_PER_PATCH = 10
LOW_POINT_MASK_PATH = os.path.join(OUTPUT_DIR, "low_point_patch_mask.nii.gz")

# ============================================================
# Union-find
# ============================================================

class UnionFind:
    """Merge local component nodes that belong to the same global component."""

    def __init__(self):
        self.parent = []
        self.rank = []

    def add(self):
        node = len(self.parent)
        self.parent.append(node)
        self.rank.append(0)
        return node

    def find(self, node):
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, node_a, node_b):
        root_a = self.find(node_a)
        root_b = self.find(node_b)

        if root_a == root_b:
            return root_a

        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a

        self.parent[root_b] = root_a

        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1

        return root_a


# ============================================================
# General helper functions
# ============================================================

def choose_label_dtype(max_label):
    """Choose the smallest unsigned integer type that can store all labels."""
    if max_label <= np.iinfo(np.uint8).max:
        return np.uint8
    if max_label <= np.iinfo(np.uint16).max:
        return np.uint16
    return np.uint32


def save_nifti(data_xyz, reference_nii, output_path, dtype):
    """Save data while preserving the reference qform/sform state."""
    data_xyz = data_xyz.astype(dtype, copy=False)

    header = reference_nii.header.copy()
    header.set_data_shape(data_xyz.shape)
    header.set_data_dtype(dtype)

    qform, qform_code = reference_nii.get_qform(coded=True)
    sform, sform_code = reference_nii.get_sform(coded=True)

    output_nii = nib.Nifti1Image(data_xyz, None, header)

    if int(qform_code) > 0:
        output_nii.set_qform(qform, code=int(qform_code))
    else:
        output_nii.set_qform(None, code=0)

    if int(sform_code) > 0:
        output_nii.set_sform(sform, code=int(sform_code))
    else:
        output_nii.set_sform(None, code=0)

    nib.save(output_nii, output_path)


def calculate_ownership(starts, patch_sizes, full_size):
    """
    Assign every voxel along one axis to one patch.

    Boundaries are placed halfway between neighbouring patch centers.
    """
    starts = sorted(set(starts))

    if not starts:
        raise ValueError("No patch starts were found.")

    centers = [0.5 * (start + min(start + patch_sizes[start], full_size) - 1) for start in starts]

    boundaries = [0]

    for index in range(len(centers) - 1):
        boundary = int(np.floor(0.5 * (centers[index] + centers[index + 1]))) + 1
        boundaries.append(boundary)

    boundaries.append(full_size)

    return {start: (boundaries[index], boundaries[index + 1]) for index, start in enumerate(starts)}


def get_overlap_slices(patch_a, patch_b):
    """Return global and patch-local overlap slices for two patches."""
    z0 = max(patch_a["z0"], patch_b["z0"])
    z1 = min(patch_a["z1"], patch_b["z1"])
    y0 = max(patch_a["y0"], patch_b["y0"])
    y1 = min(patch_a["y1"], patch_b["y1"])
    x0 = max(patch_a["x0"], patch_b["x0"])
    x1 = min(patch_a["x1"], patch_b["x1"])

    if z0 >= z1 or y0 >= y1 or x0 >= x1:
        return None

    a_local = (
        slice(z0 - patch_a["z0"], z1 - patch_a["z0"]),
        slice(y0 - patch_a["y0"], y1 - patch_a["y0"]),
        slice(x0 - patch_a["x0"], x1 - patch_a["x0"]),
    )

    b_local = (
        slice(z0 - patch_b["z0"], z1 - patch_b["z0"]),
        slice(y0 - patch_b["y0"], y1 - patch_b["y0"]),
        slice(x0 - patch_b["x0"], x1 - patch_b["x0"]),
    )

    return a_local, b_local


# ============================================================
# Patch loading
# ============================================================

PATCH_PATTERN = re.compile(
    r"^patch_(\d+)_z(\d+)-(\d+)_y(\d+)-(\d+)_x(\d+)-(\d+)\.nii(?:\.gz)?$",
    re.IGNORECASE,
)


def load_patch_info(patch_dir):
    """Read patch indices and global coordinates from patch filenames."""
    patches = []

    for filename in os.listdir(patch_dir):
        match = PATCH_PATTERN.match(filename)

        if match is None:
            continue

        patch_index, z0, z1, y0, y1, x0, x1 = [int(value) for value in match.groups()]

        patches.append({
            "index": patch_index,
            "z0": z0,
            "z1": z1,
            "y0": y0,
            "y1": y1,
            "x0": x0,
            "x1": x1,
            "path": os.path.join(patch_dir, filename),
        })

    patches.sort(key=lambda item: item["index"])

    if not patches:
        raise RuntimeError(f"No valid patch masks found in: {patch_dir}")

    return patches


def load_patch_mask(patch):
    """Load one patch mask and convert NIfTI XYZ order to ZYX."""
    patch_nii = nib.load(patch["path"])
    patch_xyz = np.asanyarray(patch_nii.dataobj)
    patch_zyx = np.transpose(patch_xyz, (2, 1, 0)).astype(np.int32, copy=False)

    expected_shape = (
        patch["z1"] - patch["z0"],
        patch["y1"] - patch["y0"],
        patch["x1"] - patch["x0"],
    )

    if patch_zyx.shape != expected_shape:
        raise ValueError(
            f"Patch {patch['index']} shape mismatch: "
            f"{patch_zyx.shape} != {expected_shape}"
        )

    return patch_zyx


# ============================================================
# Overlap matching
# ============================================================

def match_overlap_labels(mask_a, mask_b, min_overlap_voxels, iomin_threshold):
    """
    Match component labels in an overlap region.

    IoMin is intersection divided by the size of the smaller component within
    the overlap comparison arrays.
    """
    valid = (mask_a > 0) & (mask_b > 0)

    if not np.any(valid):
        return []

    labels_a = mask_a[valid].astype(np.int64, copy=False)
    labels_b = mask_b[valid].astype(np.int64, copy=False)

    max_label_b = int(mask_b.max()) + 1
    pair_codes = labels_a * max_label_b + labels_b
    codes, intersections = np.unique(pair_codes, return_counts=True)

    counts_a = np.bincount(mask_a[mask_a > 0].astype(np.int64, copy=False))
    counts_b = np.bincount(mask_b[mask_b > 0].astype(np.int64, copy=False))

    candidates = []

    for code, intersection in zip(codes, intersections):
        label_a = int(code // max_label_b)
        label_b = int(code % max_label_b)

        if intersection < min_overlap_voxels:
            continue

        denominator = min(int(counts_a[label_a]), int(counts_b[label_b]))

        if denominator <= 0:
            continue

        iomin = float(intersection) / denominator

        if iomin >= iomin_threshold:
            candidates.append((iomin, int(intersection), label_a, label_b))

    # Use one-to-one greedy matching, as in the original script.
    candidates.sort(reverse=True)

    matches = []
    used_a = set()
    used_b = set()

    for iomin, intersection, label_a, label_b in candidates:
        if label_a in used_a or label_b in used_b:
            continue

        matches.append((label_a, label_b, intersection, iomin))
        used_a.add(label_a)
        used_b.add(label_b)

    return matches


def patches_overlap(patch_a, patch_b):
    """Return True when two patches overlap in all three dimensions."""
    return (
        patch_a["z0"] < patch_b["z1"] and patch_a["z1"] > patch_b["z0"] and
        patch_a["y0"] < patch_b["y1"] and patch_a["y1"] > patch_b["y0"] and
        patch_a["x0"] < patch_b["x1"] and patch_a["x1"] > patch_b["x0"]
    )


# ============================================================
# Generate component mapping
# ============================================================

def generate_component_mapping(patches):
    """
    Build local-to-global component mappings from the available patch masks.
    """
    union_find = UnionFind()
    patch_masks = {}
    label_to_node = {}
    low_point_patches = []

    print("[INFO] Loading patch masks and registering local components...")

    for number, patch in enumerate(patches, start=1):
        mask = load_patch_mask(patch)
        patch_masks[patch["index"]] = mask
        label_to_node[patch["index"]] = {}

        local_labels = np.unique(mask)
        local_labels = local_labels[local_labels > 0]
        point_count = len(local_labels)

        if point_count < MIN_POINTS_PER_PATCH:
            low_point_patches.append({
                "index": patch["index"],
                "point_count": point_count,
                "z0": patch["z0"],
                "z1": patch["z1"],
                "y0": patch["y0"],
                "y1": patch["y1"],
                "x0": patch["x0"],
                "x1": patch["x1"],
            })

        for local_label in local_labels:
            local_label = int(local_label)
            label_to_node[patch["index"]][local_label] = union_find.add()

        print(f"\r[INFO] Loaded patch {number}/{len(patches)}", end="", flush=True)

    print()

    if low_point_patches:
        print(f"[WARNING] Patches with fewer than {MIN_POINTS_PER_PATCH} points:")
        for item in low_point_patches:
            print(
                f"  Patch {item['index']}: {item['point_count']} points | "
                f"z={item['z0']}-{item['z1']}, "
                f"y={item['y0']}-{item['y1']}, "
                f"x={item['x0']}-{item['x1']}"
            )
    else:
        print(f"[INFO] All patches contain at least {MIN_POINTS_PER_PATCH} points.")

    matched_pairs = 0

    print("[INFO] Matching components in overlapping patches...")

    for current_position, current_patch in enumerate(patches):
        current_mask = patch_masks[current_patch["index"]]

        for previous_position in range(current_position):
            previous_patch = patches[previous_position]

            if not patches_overlap(current_patch, previous_patch):
                continue

            overlap = get_overlap_slices(current_patch, previous_patch)

            if overlap is None:
                continue

            current_local, previous_local = overlap
            previous_mask = patch_masks[previous_patch["index"]]

            matches = match_overlap_labels(
                current_mask[current_local],
                previous_mask[previous_local],
                MIN_OVERLAP_VOXELS,
                IOMIN_THRESHOLD,
            )

            for current_label, previous_label, _, _ in matches:
                current_node = label_to_node[current_patch["index"]].get(current_label)
                previous_node = label_to_node[previous_patch["index"]].get(previous_label)

                if current_node is None or previous_node is None:
                    continue

                union_find.union(current_node, previous_node)
                matched_pairs += 1

        print(
            f"\r[INFO] Matched patch {current_position + 1}/{len(patches)}",
            end="",
            flush=True,
        )

    print()

    roots = set()

    for patch_mapping in label_to_node.values():
        for node in patch_mapping.values():
            roots.add(union_find.find(node))

    root_to_global = {
        root: global_id
        for global_id, root in enumerate(sorted(roots), start=1)
    }

    mapping = {}
    mapping_rows = []

    for patch in patches:
        patch_index = patch["index"]
        mapping[patch_index] = {}

        for local_label, node in label_to_node[patch_index].items():
            root = union_find.find(node)
            global_id = root_to_global[root]

            mapping[patch_index][local_label] = global_id

            mapping_rows.append({
                "patch_index": patch_index,
                "local_label": local_label,
                "global_component_id": global_id,
            })

    print(f"[INFO] Local components: {sum(len(item) for item in label_to_node.values())}")
    print(f"[INFO] Matched component pairs: {matched_pairs}")
    print(f"[INFO] Global components: {len(root_to_global)}")

    low_point_patch_indices = {item["index"] for item in low_point_patches}

    return mapping, mapping_rows, patch_masks, len(root_to_global), low_point_patch_indices

def write_mapping_csv(path, rows):
    """Save the generated local-to-global component mapping."""
    fieldnames = ["patch_index", "local_label", "global_component_id"]

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Final mask generation
# ============================================================

def map_local_to_global(local_mask, label_mapping, dtype):
    """Replace local patch labels with global component IDs."""
    max_local_label = int(local_mask.max())

    if max_local_label == 0:
        return np.zeros(local_mask.shape, dtype=dtype)

    lookup = np.zeros(max_local_label + 1, dtype=dtype)

    for local_label, global_id in label_mapping.items():
        if 0 < local_label <= max_local_label:
            lookup[local_label] = global_id

    return lookup[local_mask]


def write_centers_csv(path, rows):
    """Save component centers and component voxel counts."""
    fieldnames = [
        "global_component_id",
        "voxel_count",
        "voxel_center_x",
        "voxel_center_y",
        "voxel_center_z",
        "voxel_center_x_rounded",
        "voxel_center_y_rounded",
        "voxel_center_z_rounded",
    ]

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def merge_available_patches():
    """Generate mapping and merge all currently available patch predictions."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(REFERENCE_IMAGE_PATH):
        raise FileNotFoundError(f"Reference image not found: {REFERENCE_IMAGE_PATH}")

    if not os.path.isdir(PATCH_DIR):
        raise NotADirectoryError(f"Patch directory not found: {PATCH_DIR}")

    reference_nii = nib.load(REFERENCE_IMAGE_PATH)
    shape_xyz = reference_nii.shape[:3]
    shape_zyx = (shape_xyz[2], shape_xyz[1], shape_xyz[0])

    patches = load_patch_info(PATCH_DIR)

    qform_code = int(reference_nii.header["qform_code"])
    sform_code = int(reference_nii.header["sform_code"])

    print(f"[INFO] Reference shape XYZ: {shape_xyz}")
    print(f"[INFO] Reference qform/sform: {qform_code}/{sform_code}")
    print(f"[INFO] Available patches: {len(patches)}")

    # Generate the component mapping directly from patch overlaps.
    mapping, mapping_rows, patch_masks, max_global_id, low_point_patch_indices = generate_component_mapping(patches)
    write_mapping_csv(GENERATED_MAPPING_PATH, mapping_rows)

    label_dtype = choose_label_dtype(max_global_id)

    # Reconstruct ownership regions from currently available patch coordinates.
    z_starts = [patch["z0"] for patch in patches]
    y_starts = [patch["y0"] for patch in patches]
    x_starts = [patch["x0"] for patch in patches]

    z_sizes = {patch["z0"]: patch["z1"] - patch["z0"] for patch in patches}
    y_sizes = {patch["y0"]: patch["y1"] - patch["y0"] for patch in patches}
    x_sizes = {patch["x0"]: patch["x1"] - patch["x0"] for patch in patches}

    ownership_z = calculate_ownership(z_starts, z_sizes, shape_zyx[0])
    ownership_y = calculate_ownership(y_starts, y_sizes, shape_zyx[1])
    ownership_x = calculate_ownership(x_starts, x_sizes, shape_zyx[2])

    temp_dir = tempfile.mkdtemp(prefix="merge_test_", dir=OUTPUT_DIR)
    instance_temp_path = os.path.join(temp_dir, "instance_mask.npy")

    instance_mask = np.lib.format.open_memmap(
        instance_temp_path,
        mode="w+",
        dtype=label_dtype,
        shape=shape_zyx,
    )
    instance_mask[:] = 0
    
    low_point_mask = np.zeros(shape_zyx, dtype=np.uint8)

    voxel_counts = np.zeros(max_global_id + 1, dtype=np.int64)
    sum_z = np.zeros(max_global_id + 1, dtype=np.float64)
    sum_y = np.zeros(max_global_id + 1, dtype=np.float64)
    sum_x = np.zeros(max_global_id + 1, dtype=np.float64)

    print("[INFO] Merging available patches...")

    for number, patch in enumerate(patches, start=1):
        patch_mask = patch_masks[patch["index"]]
        global_patch = map_local_to_global(patch_mask, mapping[patch["index"]], label_dtype)

        gz0, gz1 = ownership_z[patch["z0"]]
        gy0, gy1 = ownership_y[patch["y0"]]
        gx0, gx1 = ownership_x[patch["x0"]]

        # Restrict ownership to the actual patch boundaries.
        gz0, gz1 = max(gz0, patch["z0"]), min(gz1, patch["z1"])
        gy0, gy1 = max(gy0, patch["y0"]), min(gy1, patch["y1"])
        gx0, gx1 = max(gx0, patch["x0"]), min(gx1, patch["x1"])

        lz0, lz1 = gz0 - patch["z0"], gz1 - patch["z0"]
        ly0, ly1 = gy0 - patch["y0"], gy1 - patch["y0"]
        lx0, lx1 = gx0 - patch["x0"], gx1 - patch["x0"]

        owned_region = global_patch[lz0:lz1, ly0:ly1, lx0:lx1]
        instance_mask[gz0:gz1, gy0:gy1, gx0:gx1] = owned_region
        if patch["index"] in low_point_patch_indices:
            low_point_mask[gz0:gz1, gy0:gy1, gx0:gx1] = 1
            
        local_z, local_y, local_x = np.nonzero(owned_region)

        if local_z.size > 0:
            labels = owned_region[local_z, local_y, local_x].astype(np.int64, copy=False)

            global_z = local_z + gz0
            global_y = local_y + gy0
            global_x = local_x + gx0

            voxel_counts += np.bincount(labels, minlength=max_global_id + 1)
            sum_z += np.bincount(labels, weights=global_z, minlength=max_global_id + 1)
            sum_y += np.bincount(labels, weights=global_y, minlength=max_global_id + 1)
            sum_x += np.bincount(labels, weights=global_x, minlength=max_global_id + 1)

        print(f"\r[INFO] Merged patch {number}/{len(patches)}", end="", flush=True)

    print()
    instance_mask.flush()

    center_mask = np.zeros(shape_zyx, dtype=label_dtype)
    center_rows = []

    for global_id in range(1, max_global_id + 1):
        count = int(voxel_counts[global_id])

        if count == 0:
            continue

        center_z = sum_z[global_id] / count
        center_y = sum_y[global_id] / count
        center_x = sum_x[global_id] / count

        rounded_z = int(np.clip(np.rint(center_z), 0, shape_zyx[0] - 1))
        rounded_y = int(np.clip(np.rint(center_y), 0, shape_zyx[1] - 1))
        rounded_x = int(np.clip(np.rint(center_x), 0, shape_zyx[2] - 1))

        center_mask[rounded_z, rounded_y, rounded_x] = global_id

        center_rows.append({
            "global_component_id": global_id,
            "voxel_count": count,
            "voxel_center_x": float(center_x),
            "voxel_center_y": float(center_y),
            "voxel_center_z": float(center_z),
            "voxel_center_x_rounded": rounded_x,
            "voxel_center_y_rounded": rounded_y,
            "voxel_center_z_rounded": rounded_z,
        })

    binary_mask = (instance_mask > 0).astype(np.uint8)

    print("[INFO] Saving outputs...")

    save_nifti(np.transpose(instance_mask, (2, 1, 0)), reference_nii, INSTANCE_MASK_PATH, label_dtype)
    save_nifti(np.transpose(binary_mask, (2, 1, 0)), reference_nii, BINARY_MASK_PATH, np.uint8)
    save_nifti(np.transpose(center_mask, (2, 1, 0)), reference_nii, CENTER_MASK_PATH, label_dtype)
    save_nifti(np.transpose(low_point_mask, (2, 1, 0)), reference_nii, LOW_POINT_MASK_PATH, np.uint8)
    write_centers_csv(CENTER_CSV_PATH, center_rows)

    del instance_mask

    if REMOVE_TEMPORARY_FILES:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("\n" + "=" * 70)
    print("TEST MERGING FINISHED")
    print("=" * 70)
    print(f"Available patches: {len(patches)}")
    print(f"Components:       {len(center_rows)}")
    print(f"Generated mapping: {GENERATED_MAPPING_PATH}")
    print(f"Binary mask:       {BINARY_MASK_PATH}")
    print(f"Instance mask:     {INSTANCE_MASK_PATH}")
    print(f"Center mask:       {CENTER_MASK_PATH}")
    print(f"Center CSV:        {CENTER_CSV_PATH}")
    print(f"Low-point mask:    {LOW_POINT_MASK_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    merge_available_patches()