#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

parser = argparse.ArgumentParser(description="Generate speedup plot and markdown from results CSV.")
parser.add_argument("name", help="base name (without extension) for results/<NAME>.csv and plots/<NAME>/")
parser.add_argument("--input", help="input CSV path (overrides default)", default=None)
parser.add_argument("--outdir", help="output directory base (overrides default 'plots')", default="plots")
args = parser.parse_args()

NAME = args.name
INPUT_FILE = args.input or f"results/{NAME}.csv"
OUTPUT_DIR = f"{args.outdir}/{NAME}"
OUTPUT_PLOT = "speedup.png"
OUTPUT_MD = "speedup.md"

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
    "N": "first",
    "K": "first",
    "D": "first",
    "iters": "first"
}).reset_index()

# flatten columns
grouped.columns = ["threads", "mean", "std", "N", "K", "D", "iters"]
grouped = grouped.sort_values("threads")

# ---------- BASELINE ----------
baseline = grouped[grouped["threads"] == 1]["mean"].values
if len(baseline) == 0:
    raise ValueError("Missing threads=1 baseline")
baseline = baseline[0]

# ---------- SPEEDUP ----------
grouped["speedup_mean"] = baseline / grouped["mean"]
grouped["speedup_std"] = (grouped["std"] / grouped["mean"]) * grouped["speedup_mean"]

# ---------- FORMAT ----------
def fmt(x):
    return f"{x:.3f}"

# ---------- PRINT TABLE ----------
print("\nRESULTS:\n")
print("threads | N | K | D | iters | mean | std | speedup | speedup_std")
for _, row in grouped.iterrows():
    print(
        f"{int(row['threads'])} | "
        f"{int(row['N'])} | "
        f"{int(row['K'])} | "
        f"{int(row['D'])} | "
        f"{int(row['iters'])} | "
        f"{fmt(row['mean'])} | "
        f"{fmt(row['std'])} | "
        f"{fmt(row['speedup_mean'])} | "
        f"{fmt(row['speedup_std'])}"
    )

# ---------- SAVE MARKDOWN ----------
os.makedirs(OUTPUT_DIR, exist_ok=True)
md_path = os.path.join(OUTPUT_DIR, OUTPUT_MD)

with open(md_path, "w") as f_md:
    f_md.write("# Speedup Results\n\n")
    f_md.write("| threads | N | K | D | iters | mean | std | speedup | speedup_std |\n")
    f_md.write("|---------|---|---|---|-------|------|-----|----------|-------------|\n")
    for _, row in grouped.iterrows():
        f_md.write(
            f"| {int(row['threads'])} "
            f"| {int(row['N'])} "
            f"| {int(row['K'])} "
            f"| {int(row['D'])} "
            f"| {int(row['iters'])} "
            f"| {fmt(row['mean'])} "
            f"| {fmt(row['std'])} "
            f"| {fmt(row['speedup_mean'])} "
            f"| {fmt(row['speedup_std'])} |\n"
        )

print(f"\nMarkdown saved to {md_path}")

# ---------- PLOT ----------
plt.figure()
plt.errorbar(
    grouped["threads"],
    grouped["speedup_mean"],
    yerr=grouped["speedup_std"],
    marker='o',
    capsize=5
)

threads = grouped["threads"]
plt.plot(threads, threads, linestyle="--", label="Ideal speedup")

plt.xlabel("Number of threads")
plt.ylabel("Speedup")
plt.title("OpenMP K-Means Speedup")
plt.legend()
plt.grid()

plot_path = os.path.join(OUTPUT_DIR, OUTPUT_PLOT)
plt.savefig(plot_path, dpi=300)
print(f"Plot saved to {plot_path}")
