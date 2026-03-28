#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

parser = argparse.ArgumentParser(description="Throughput analysis from multiple CSV files.")
parser.add_argument("--inputs", nargs="+", required=True, help="list of input CSV files")
parser.add_argument("--labels", nargs="+", required=True, help="labels for each CSV")
parser.add_argument("--outdir", default="img", help="output directory")
parser.add_argument("--logy", action="store_true", help="log scale on Y axis")
args = parser.parse_args()

if len(args.inputs) != len(args.labels):
    raise ValueError("Number of inputs must match number of labels")

os.makedirs(args.outdir, exist_ok=True)

plt.figure()

# ---------- FUNCTION ----------
def compute_work(row):
    return row["N"] * row["K"] * row["D"] * row["iters"]

pastel_colors = ["#DCC6E0", "#FF7F7F","#FBC4D9",
    "#A7C7E7", "#6FA8DC",  # blue
    "#B7E4C7", "#76C893",  # green
    "#FBC4D9", "#F497B6",  # pink
    "#DCC6E0", "#B497BD",  # lavender
    "#FFD6A5", "#FFB347",  # peach
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

    required_cols = ["N", "K", "D", "iters", "elapsed"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{input_file} missing column: {col}")

    # ---------- COMPUTE WORK ----------
    df["work"] = df.apply(compute_work, axis=1)

    # ---------- GROUP ----------
    grouped = df.groupby(["N", "K", "D", "iters"]).agg({
        "elapsed": ["mean", "std"],
        "work": "first"
    }).reset_index()

    grouped.columns = ["N", "K", "D", "iters", "time_mean", "time_std", "work"]

    # ---------- THROUGHPUT ----------
    grouped["throughput_mean"] = grouped["work"] / grouped["time_mean"]
    grouped["throughput_std"] = (grouped["time_std"] / grouped["time_mean"]) * grouped["throughput_mean"]


    scale = 1e6
    grouped["throughput_mean"] /= scale
    grouped["throughput_std"] /= scale
    grouped["N"] /= scale

    # ---------- PRINT ----------
    print(f"\n=== {label} ===")
    print("N | K | D | iters | time_mean | time_std | throughput | thr_std")

    for _, row in grouped.iterrows():
        print(
            f"{int(row['N'])} | "
            f"{int(row['K'])} | "
            f"{int(row['D'])} | "
            f"{int(row['iters'])} | "
            f"{row['time_mean']:.6f} | "
            f"{row['time_std']:.6f} | "
            f"{row['throughput_mean']:.3e} | "
            f"{row['throughput_std']:.3e}"
        )

    # ---------- SORT FOR PLOT ----------
    grouped = grouped.sort_values("N")

    # ---------- PLOT ----------
    plt.errorbar(
        grouped["N"],
        grouped["throughput_mean"],
        yerr=grouped["throughput_std"],
        marker='o',
        capsize=5,
        label=label,
        color=pastel_colors[i]
    )

# ---------- SCALE ----------
if args.logy:
    plt.yscale("log")

# ---------- FORMAT ----------
plt.xlabel("N (×10⁶)")
plt.ylabel("Throughput (×10⁶ ops/s)", labelpad=1)
plt.ylim(bottom=0)
# plt.title("Throughput (axes SCALED BY 1M)")

plt.legend()
plt.grid()

plot_path = os.path.join(args.outdir, "throughput.png")
plt.savefig(plot_path, dpi=300)

print(f"\nPlot saved to {plot_path}")