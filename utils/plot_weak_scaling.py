#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

parser = argparse.ArgumentParser(description="Generate weak scaling efficiency plot and markdown from results CSV.")
parser.add_argument("name", help="base name (without extension) for results/<NAME>.csv and plots/<NAME>/")
parser.add_argument("--input", help="input CSV path (overrides default)", default=None)
parser.add_argument("--outdir", help="output directory base (overrides default 'plots')", default="plots")
args = parser.parse_args()

NAME = args.name
INPUT_FILE = args.input or f"results/{NAME}.csv"
OUTPUT_DIR = f"{args.outdir}/{NAME}"
OUTPUT_PLOT = "weak_scaling.png"
OUTPUT_MD = "weak_scaling.md"

# ---------- LOAD DATA ----------
if not Path(INPUT_FILE).exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)

# ---------- CHECK ----------
required_cols = ["experiment", "threads", "N", "K", "D", "iters", "run", "elapsed"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

# ---------- AGGREGATE ----------
grouped = df.groupby("threads").agg({
    "elapsed": ["mean", "std"],
    "N": "first", # N will scale up with threads in weak scaling
    "K": "first",
    "D": "first",
    "iters": "first"
}).reset_index()

# flatten columns
grouped.columns = ["threads", "mean", "std", "N", "K", "D", "iters"]
grouped = grouped.sort_values("threads")

# ---------- BASELINE ----------
baseline_data = grouped[grouped["threads"] == 1]["mean"].values
if len(baseline_data) == 0:
    raise ValueError("Missing threads=1 baseline")
baseline_time = baseline_data[0]

# ---------- EFFICIENCY ----------
# In weak scaling, ideal time is constant (T_1). Efficiency = T_1 / T_p.
grouped["efficiency_mean"] = baseline_time / grouped["mean"]
grouped["efficiency_std"] = (grouped["std"] / grouped["mean"]) * grouped["efficiency_mean"]

# ---------- FORMAT ----------
def fmt(x):
    return f"{x:.4f}"

# ---------- PRINT TABLE ----------
print("\nWEAK SCALING RESULTS:\n")
print("threads | N | K | D | iters | time_mean | time_std | efficiency | eff_std")
for _, row in grouped.iterrows():
    print(
        f"{int(row['threads']):7d} | "
        f"{int(row['N']):9d} | "
        f"{int(row['K']):1d} | "
        f"{int(row['D']):2d} | "
        f"{int(row['iters']):5d} | "
        f"{fmt(row['mean']):9s} | "
        f"{fmt(row['std']):8s} | "
        f"{fmt(row['efficiency_mean']):10s} | "
        f"{fmt(row['efficiency_std'])}"
    )

# ---------- SAVE MARKDOWN ----------
os.makedirs(OUTPUT_DIR, exist_ok=True)
md_path = os.path.join(OUTPUT_DIR, OUTPUT_MD)

with open(md_path, "w") as f_md:
    f_md.write("# Weak Scaling Results\n\n")
    f_md.write("| threads | N | K | D | iters | time_mean | time_std | efficiency | eff_std |\n")
    f_md.write("|---------|---|---|---|-------|-----------|----------|------------|---------|\n")
    for _, row in grouped.iterrows():
        f_md.write(
            f"| {int(row['threads'])} "
            f"| {int(row['N'])} "
            f"| {int(row['K'])} "
            f"| {int(row['D'])} "
            f"| {int(row['iters'])} "
            f"| {fmt(row['mean'])} "
            f"| {fmt(row['std'])} "
            f"| {fmt(row['efficiency_mean'])} "
            f"| {fmt(row['efficiency_std'])} |\n"
        )

print(f"\nMarkdown saved to {md_path}")

# ---------- PLOT ----------
plt.figure()

# Plot measured efficiency
plt.errorbar(
    grouped["threads"],
    grouped["efficiency_mean"],
    yerr=grouped["efficiency_std"],
    marker='o',
    capsize=5,
    label="Measured Efficiency"
)

# Plot ideal efficiency (flat line at 1.0)
threads = grouped["threads"]
plt.plot(threads, [1.0] * len(threads), linestyle="--", color="gray", label="Ideal Efficiency (1.0)")

plt.xlabel("Number of threads")
plt.ylabel("Efficiency ($T_1 / T_p$)")
plt.title("OpenMP K-Means Weak Scaling Efficiency")

# Adjust Y-axis to easily see drop-offs from 1.0
plt.ylim(0.0, 1.1) 
plt.legend()
plt.grid()

plot_path = os.path.join(OUTPUT_DIR, OUTPUT_PLOT)
plt.savefig(plot_path, dpi=300)
print(f"Plot saved to {plot_path}")