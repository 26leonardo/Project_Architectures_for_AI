#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

parser = argparse.ArgumentParser(description="Generate strong scaling plot from multiple CSV files.")
parser.add_argument("--inputs", nargs="+", required=True, help="list of input CSV files")
parser.add_argument("--labels", nargs="+", required=True, help="labels for each CSV")
parser.add_argument("--outdir", default="img", help="output directory")
args = parser.parse_args()

if len(args.inputs) != len(args.labels):
    raise ValueError("Number of inputs must match number of labels")

os.makedirs(args.outdir, exist_ok=True)

plt.figure()
colors =[
    "#A7C7E7", "#6FA8DC",  # blue v1
    "#DCC6E0", "#B497BD",  # lavender v1 o3 500k . v1 o3 1M
    "#FBC4D9", "#F497B6",  # pink v2 o3 500k
    "#FFD6A5", "#FFB347",  # peach v2_OMP 500k v2_OMP  1M
    "#FFF3B0", "#FFE066",  # yellow
    "#B8F2E6", "#70D6C1",  # turquoise
    "#FFADAD", "#FF7F7F",  # red
    "#E0E0E0", "#B0B0B0",  # gray
    "#F5E6CC", "#E6CCB2"   # beige
]
# ---------- PROCESS EACH DATASET ----------
for i, (input_file, label) in enumerate(zip(args.inputs, args.labels)):

    if not Path(input_file).exists():
        raise FileNotFoundError(f"File not found: {input_file}")

    df = pd.read_csv(input_file)

    required_cols = ["threads", "elapsed"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{input_file} missing column: {col}")

    # ---------- AGGREGATE ----------
    grouped = df.groupby("threads").agg({
        "elapsed": ["mean", "std"]
    }).reset_index()

    grouped.columns = ["threads", "mean", "std"]
    grouped = grouped.sort_values("threads")

    # ---------- BASELINE ----------
    baseline = grouped[grouped["threads"] == 1]["mean"].values
    if len(baseline) == 0:
        raise ValueError(f"{input_file}: missing threads=1 baseline")
    baseline = baseline[0]

    # ---------- SPEEDUP ----------
    grouped["speedup_mean"] = baseline / grouped["mean"]
    grouped["speedup_std"] = (grouped["std"] / grouped["mean"]) * grouped["speedup_mean"]

    # ---------- PLOT ----------
    plt.errorbar(
        grouped["threads"],
        grouped["speedup_mean"],
        yerr=grouped["speedup_std"],
        marker='o',
        capsize=5,
        label=label,
        color=colors[i]
    )

# ---------- IDEAL LINE ----------
# usa l'ultimo dataset per recuperare i thread (assume siano uguali)
threads = grouped["threads"]
plt.plot(threads, threads, linestyle="--", label="Ideal")

# ---------- FORMAT ----------
plt.xlabel("Number of threads")
plt.ylabel("Speedup")
plt.legend()
plt.grid()

plot_path = os.path.join(args.outdir, "strong_scaling_1.png")
plt.savefig(plot_path, dpi=300)
print(f"\n=== {label} ===")
print(f"{'threads':>8} | {'mean':>10} | {'std':>10} ")
print("-"*60)

for _, row in grouped.iterrows():
    print(
        f"{int(row['threads']):8d} | "
        f"{row['mean']:10.6f} | "
        f"{row['std']:10.6f} | "
    )

print(f"Plot saved to {plot_path}")