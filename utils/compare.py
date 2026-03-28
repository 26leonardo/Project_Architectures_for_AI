#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

parser = argparse.ArgumentParser(description="CPU vs GPU speedup (CPU/GPU)")
parser.add_argument("--cpu1", required=True)
parser.add_argument("--cpu2", required=True)
parser.add_argument("--gpu", required=True)
parser.add_argument("--label1", default="CPU1")
parser.add_argument("--label2", default="CPU2")
parser.add_argument("--outdir", default="plots/cpu_vs_gpu")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

required = ["N","K","D","iters","elapsed"]

def load_and_agg(path, prefix):
    if not Path(path).exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    for col in required:
        if col not in df.columns:
            raise ValueError(f"{path} missing column {col}")

    g = df.groupby(["N","K","D","iters"]).agg({
        "elapsed": ["mean","std"]
    }).reset_index()

    g.columns = ["N","K","D","iters", f"{prefix}_mean", f"{prefix}_std"]
    return g

# ---------- LOAD ----------
cpu1 = load_and_agg(args.cpu1, "cpu1")
cpu2 = load_and_agg(args.cpu2, "cpu2")
gpu  = load_and_agg(args.gpu,  "gpu")

# ---------- MERGE ----------
merged = cpu1.merge(cpu2, on=["N","K","D","iters"])
merged = merged.merge(gpu, on=["N","K","D","iters"])

if merged.empty:
    raise ValueError("No matching configurations")

# ---------- SPEEDUP ----------
merged["s1"] = merged["cpu1_mean"] / merged["gpu_mean"]
merged["s2"] = merged["cpu2_mean"] / merged["gpu_mean"]

# error propagation
merged["s1_std"] = merged["s1"] * np.sqrt(
    (merged["cpu1_std"]/merged["cpu1_mean"])**2 +
    (merged["gpu_std"]/merged["gpu_mean"])**2
)

merged["s2_std"] = merged["s2"] * np.sqrt(
    (merged["cpu2_std"]/merged["cpu2_mean"])**2 +
    (merged["gpu_std"]/merged["gpu_mean"])**2
)

# ---------- CHECK (importante) ----------
if merged[["K","D","iters"]].nunique().max() > 1:
    print("WARNING: stai mescolando configurazioni diverse → plot potenzialmente fuorviante")

# ---------- SORT ----------
merged = merged.sort_values("N")
scale = 1e6
merged["N"] /= scale

# ---------- PRINT ----------
print("\nN | CPU1 | CPU2 | GPU | S1 | S2")
for _, r in merged.iterrows():
    print(
        f"{int(r['N'])} | "
        f"{r['cpu1_mean']:.6f} | "
        f"{r['cpu2_mean']:.6f} | "
        f"{r['gpu_mean']:.6f} | "
        f"{r['s1']:.3f} | "
        f"{r['s2']:.3f}"
    )

# ---------- PLOT ----------
plt.figure()

plt.errorbar(
    merged["N"],
    merged["s1"],
    yerr=merged["s1_std"],
    marker='o',
    capsize=5,
    label=args.label1,
    color="#FF7F7F"
)

plt.errorbar(
    merged["N"],
    merged["s2"],
    yerr=merged["s2_std"],
    marker='s',
    capsize=5,
    label=args.label2,
    color="#6FA8DC"
)

plt.axhline(1.0, linestyle="--", color="gray", label="Parity (1.0)")

plt.xlabel("N (×10⁶)")
plt.ylabel("Speedup (CPU / GPU)")
# plt.title("CPU vs GPU Speedup")
plt.legend()
plt.grid()

plt.ylim(bottom=0)

plot_path = os.path.join(args.outdir, "cpu_vs_gpu.png")
plt.savefig(plot_path, dpi=300)

print(f"\nPlot saved to {plot_path}")