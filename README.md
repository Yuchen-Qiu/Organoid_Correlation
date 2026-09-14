# Organoid Imaging Analysis

This repository contains code for two organoid imaging analysis projects:

1. **Organoid Fiber Phantom Segmentation**
2. **Organoid Microscopy-dMRI Correlation**

The first project focuses on fiber segmentation from high-resolution GRE images and the calculation of GRE-derived surface-to-volume ratio (S/V) on the OGSE sampling grid. The second project focuses on multimodal registration between microscopy and diffusion MRI (dMRI), microscopy nuclei segmentation using Cellpose, registration evaluation, and quantitative nuclei-density analysis.

## Installation

The required Python environment is defined in `environment.yml`.

Create the environment using:

```bash
conda env create -f environment.yml
```

Then activate it:

```bash
conda activate organoid_final
```

Input and output paths are currently specified inside the individual scripts and should be adjusted according to the location of the data before running the analysis.

---

# 1. Organoid Fiber Phantom Segmentation

This project focuses on segmenting fibers from high-resolution GRE images of a fiber phantom and using the detected fiber locations to estimate a GRE-derived surface-to-volume ratio (S/V) on the OGSE sampling grid.

The fiber segmentation code is located in:

```text
Fiber_segmentation/
```

## 1.1 Fiber Segmentation

The main segmentation scripts are located in:

```text
Fiber_segmentation/segmentation/
```

Run the scripts in numerical order, starting with the script beginning with:

```text
01_
```

followed by:

```text
02_watershed
```

The segmentation procedure generates:

- a fiber segmentation mask
- a fiber centroid point map

The centroid point map contains the detected locations of the fibers and is used for the subsequent GRE-derived S/V calculation.

## 1.2 GRE-derived S/V Calculation

After fiber segmentation, run:

```text
regrid_points_devide_V.py
```

This script maps the detected fiber centroid points onto the OGSE sampling grid.

The number and spatial distribution of fibers within each OGSE voxel are then used to calculate the GRE-derived surface-to-volume ratio (S/V).

The resulting GRE-derived S/V measurements can subsequently be compared with the OGSE-derived S/V measurements.

---

# 2. Organoid Microscopy-dMRI Correlation

This project focuses on multimodal registration and quantitative comparison between microscopy and dMRI in organoids.

The workflow consists of microscopy-dMRI registration, application and evaluation of the registration transformations, Cellpose-based nuclei segmentation of the microscopy volume, reconstruction of the patch-level segmentation results, and nuclei-density analysis in relation to dMRI-derived measurements.

## 2.1 Microscopy-dMRI Registration

The registration code is located in:

```text
Organoid_registration/
```

Run:

```text
run_register.py
```

to perform the microscopy-dMRI registration.

The script performs the registration and saves the resulting transformation parameters for subsequent processing.

## 2.2 Applying the Final Transformation

After registration, use:

```text
apply_final_transform.py
```

to apply the final registration transformation to additional images or masks.

This allows other microscopy- or dMRI-derived images to be transformed into the corresponding registered space using the same transformation estimated during registration.

## 2.3 Registration Evaluation

Use:

```text
evaluate_registration.py
```

to evaluate the registration results.

The script calculates image similarity metrics including:

- Mutual Information (MI)
- Normalized Cross-Correlation (NCC)

The resulting evaluation files can then be combined using:

```text
evaluation_merging.py
```

This auxiliary script collects the registration evaluation results and organizes them into a summary table for easier comparison between different registration configurations.

---

# 2.4 Microscopy Nuclei Segmentation

The microscopy segmentation code is located in:

```text
Cellpose_Microscopy_Segmentation/
```

Run:

```text
run_cellpose_with_mask.py
```

to perform nuclei segmentation using Cellpose.

Because the original three-dimensional microscopy volume is too large to process directly, the microscopy image is first cropped into smaller patches. Cellpose segmentation is then performed independently on these patches.

## 2.5 Patch Merging

After all microscopy patches have been segmented, run the patch-merging script:

```text
patch_merging.py
```

The script combines the patch-level segmentation results back into the original microscopy coordinate space.

The merging procedure generates:

- a full-resolution microscopy nuclei segmentation mask
- a nuclei centroid point map

The centroid point map is used for the subsequent nuclei-density analysis.

## 2.6 Segmentation Visual Check

The original microscopy volume and the corresponding full-resolution segmentation are very large.

For easier visual inspection of the segmentation result, use:

```text
Downsampling_final.py
```

This script generates a downsampled version of the microscopy segmentation while preserving the spatial relationship of the data.

The downsampled result can then be used for visual quality control of the Cellpose segmentation and patch-merging results.

---

# 2.7 Nuclei Density and dMRI Correlation Analysis

After microscopy segmentation and registration have been completed, the microscopy-derived nuclei distribution can be compared with the dMRI-derived measurements.

The corresponding analysis scripts are located in the `display` directory.

## Full-volume Analysis

Use:

```text
display_correlation.py
```

to perform nuclei-density analysis using the full analysis volume.

The microscopy nuclei centroid locations are aggregated onto the dMRI analysis grid to calculate nuclei density. The resulting nuclei-density measurements can then be compared with dMRI-derived measurements for voxel-wise quantitative analysis.

## Selected-volume Analysis

Use:

```text
display_correlation_patch.py
```

to perform the same nuclei-density and dMRI analysis using only a selected range of the volume instead of the full volume.

This is useful when the analysis needs to be restricted to a specific region or to the part of the microscopy volume with reliable microscopy-dMRI correspondence.

## Low-density Region Analysis

Use:

```text
display_low_density.py
```

to inspect regions with low nuclei density.

For this visualization, nuclei density is displayed using an inverted intensity representation:

```text
lower nuclei density -> higher image intensity
```

This makes low-density areas easier to identify visually and can be used to inspect whether low-density regions correspond to particular structures or regions in the microscopy and dMRI images.

---

# Workflow Summary

The overall workflow for the fiber phantom project is:

```text
GRE image
    |
    v
01-02 segmentation
    |
    v
Fiber mask + centroid point map
    |
    v
regrid_points_devide_V.py
    |
    v
GRE-derived S/V on OGSE grid
```

The overall workflow for the microscopy-dMRI correlation project is:

```text
Microscopy + dMRI
        |
        v
run_register.py
        |
        v
Registration transformation
        |
        +------------------------+
        |                        |
        v                        v
apply_final_transform.py   evaluate_registration.py
                                 |
                                 v
                         evaluation_merging.py

Microscopy
    |
    v
run_cellpose_with_mask.py
    |
    v
Patch-level nuclei segmentation
    |
    v
patch_merging.py
    |
    +----------------------------+
    |                            |
    v                            v
Full-resolution mask       Nuclei centroid map
    |                            |
    v                            v
Downsampling_final.py      Density analysis
                                 |
              +------------------+------------------+
              |                  |                  |
              v                  v                  v
display_correlation.py  display_correlation_patch.py  display_low_density.py
```

---

# Data

Raw microscopy, GRE, OGSE, and dMRI data are not included in this repository.

Large imaging files and intermediate outputs are excluded from Git version control through `.gitignore`.

Input and output paths should be adjusted in the corresponding scripts before running the analysis.