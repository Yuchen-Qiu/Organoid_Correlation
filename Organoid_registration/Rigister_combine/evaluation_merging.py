#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from functions_weighted import merge_evaluation_metrics

# Root directory containing the individual registration evaluation runs
ROOT = "/autofs/arch11/DATA/HOMES/yuchen/Minor/Organoids/Nifti_data/case2_young/data_final/registration_metric_evaluation"

# Metric file generated for each evaluation run
CSV_NAME = "metrics_B9_MD_FA_S0.csv"

# Combined summary table across all registration runs
OUTPUT = os.path.join(ROOT, "metrics_summary.csv")


def main():
    # Collect and merge the per-run metric files into a single summary CSV
    rows = merge_evaluation_metrics(ROOT, CSV_NAME, OUTPUT)

    print(f"\nMerged rows: {len(rows)}")
    print(f"Saved to: {OUTPUT}")


if __name__ == "__main__":
    main()