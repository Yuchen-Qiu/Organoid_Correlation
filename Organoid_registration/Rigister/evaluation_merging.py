#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd

ROOT = "/autofs/arch11/DATA/HOMES/yuchen/Minor/Organoids/Nifti_data/case2_young/data_final/registration_metric_evaluation"
CSV_NAME = "metrics_B9_MD_FA_S0.csv"
OUTPUT = os.path.join(ROOT, "metrics_summary.csv")

rows = []

for folder in sorted(glob.glob(os.path.join(ROOT, "*"))):
    if not os.path.isdir(folder):
        continue

    experiment = os.path.basename(folder)
    csv_path = os.path.join(folder, "QC", CSV_NAME)

    if not os.path.exists(csv_path):
        print(f"[SKIP] {experiment}: CSV not found")
        continue

    df = pd.read_csv(csv_path)

    if len(df) == 1:
        row = df.iloc[0].to_dict()
        row = {"experiment": experiment, **row}
        rows.append(row)


    else:
        for _, r in df.iterrows():
            row = r.to_dict()
            row = {"experiment": experiment, **row}
            rows.append(row)

summary = pd.DataFrame(rows)
summary.to_csv(OUTPUT, index=False)

print("\nSummary:")
print(summary.to_string(index=False))
print(f"\nSaved to: {OUTPUT}")