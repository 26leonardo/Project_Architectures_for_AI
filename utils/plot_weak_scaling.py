#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

parser = argparse.ArgumentParser(description="Generate weak scaling plot from multiple CSV files.")
parser.add_argument("--inputs", nargs="+", required=True, help="list of input CSV files")
parser.add_argument("--labels", nargs="+", required=True, help="labels for each CSV")
parser.add_argument("--outdir", default="plots/weak_scaling", help="output directory")
parser.add_argument("--logy", action="store_true", help="log scale on Y axis")
args = parser.parse_args()

if len(args.inputs) != len(args.labels):
    raise ValueError("Number of inputs must match number of labels")

os.makedirs(args.outdir, exist_ok=True)

plt.figure()

colors = [
    "#A7C7E7", "#FFD6A5"
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

    # ---------- EFFICIENCY ----------
    grouped["efficiency_mean"] = baseline / grouped["mean"]
    grouped["efficiency_std"] = (grouped["std"] / grouped["mean"]) * grouped["efficiency_mean"]

    # ---------- PLOT ----------
    plt.errorbar(
        grouped["threads"],
        grouped["efficiency_mean"],
        yerr=grouped["efficiency_std"],
        marker='o',
        capsize=5,
        label=label,
        color=colors[i]
    )

# ---------- IDEAL ----------
threads = grouped["threads"]
plt.plot(threads, [1.0] * len(threads), linestyle="--", label="Ideal")

# ---------- SCALE ----------
if args.logy:
    plt.yscale("log", base=2)

# ---------- FORMAT ----------
plt.xlabel("Number of threads")
plt.ylabel("Efficiency (T1 / Tp)")
plt.title("Weak Scaling")
plt.legend()
plt.grid()

# ATTENZIONE: questo diventa discutibile con log-scale
if not args.logy:
    plt.ylim(0.0, 1.1)

plot_path = os.path.join(args.outdir, "weak_scaling.png")
plt.savefig(plot_path, dpi=300)

print(f"Plot saved to {plot_path}")