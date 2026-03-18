#!/usr/bin/env python3
"""
analysis.py -- Analysis and plotting of K-Means scaling experiments.

Reads CSVs produced by run_scaling.sh and generates:
  1. Strong scaling: speedup and efficiency vs threads
  2. Weak scaling: efficiency vs threads
  3. CUDA baseline: throughput vs N
  4. Cache study: time vs data size / cache level
  5. Hyperthreading: 8 vs 16 threads comparison

Run with:
    python3 analysis.py

Output:
    plots/strong_scaling.png
    plots/weak_scaling.png
    plots/cuda_baseline.png
    plots/cache_study.png
    plots/hyperthreading.png
    plots/omp_vs_cuda.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR = "results"
PLOTS_DIR   = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi":      150,
    "font.size":       11,
    "axes.titlesize":  13,
    "axes.labelsize":  11,
    "legend.fontsize": 10,
    "lines.linewidth": 2,
    "lines.markersize": 6,
})

# ---------------------------------------------------------------------------
# Helper: load CSV and compute mean ± std of elapsed time.
# Groups by the specified key column(s).
# ---------------------------------------------------------------------------
def load_stats(filename: str, group_cols) -> pd.DataFrame:
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        print(f"WARNING: {path} not found, skipping.")
        return None
    df = pd.read_csv(path)
    df["elapsed"] = pd.to_numeric(df["elapsed"], errors="coerce")
    df = df.dropna(subset=["elapsed"])
    stats = (
        df.groupby(group_cols)["elapsed"]
          .agg(mean="mean", std="std", count="count")
          .reset_index()
    )
    # Fill NaN std (single run) with 0
    stats["std"] = stats["std"].fillna(0.0)
    return stats

# ---------------------------------------------------------------------------
# 1. STRONG SCALING
#
# Speedup(p) = T(1) / T(p)   [T(1) = baseline with 1 thread]
# Efficiency(p) = Speedup(p) / p
# Ideal speedup = p (linear)
# ---------------------------------------------------------------------------
def plot_strong_scaling():
    stats = load_stats("strong_omp.csv", ["threads"])
    if stats is None:
        return

    t1 = stats.loc[stats["threads"] == 1, "mean"].values
    if len(t1) == 0:
        print("WARNING: no 1-thread entry in strong_omp.csv")
        return
    t1 = t1[0]

    stats["speedup"]    = t1 / stats["mean"]
    stats["efficiency"] = stats["speedup"] / stats["threads"]

    # Propagate error: sigma_speedup ≈ (T1/T^2) * sigma_T
    stats["speedup_err"] = (t1 / stats["mean"]**2) * stats["std"]

    threads = stats["threads"].values
    ideal   = threads.astype(float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Strong Scaling — OpenMP\n"
                 f"(N={stats['threads'].iloc[0] * 0:.0f} fixed, K=16, D=8, "
                 f"{len(stats)} thread configs, "
                 f"{stats['count'].iloc[0] if 'count' in stats.columns else '?'} runs each)")

    # Speedup
    ax1.plot(ideal, ideal, "k--", label="Ideal (linear)", linewidth=1.2)
    ax1.errorbar(threads, stats["speedup"], yerr=stats["speedup_err"],
                 marker="o", capsize=4, label="Measured speedup")
    ax1.set_xlabel("Threads (p)")
    ax1.set_ylabel("Speedup T(1)/T(p)")
    ax1.set_title("Speedup")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(threads)

    # Efficiency
    ax2.axhline(1.0, color="k", linestyle="--", linewidth=1.2, label="Ideal (100%)")
    ax2.plot(threads, stats["efficiency"], marker="s", color="tab:orange",
             label="Efficiency")
    ax2.set_xlabel("Threads (p)")
    ax2.set_ylabel("Efficiency Speedup/p")
    ax2.set_title("Parallel Efficiency")
    ax2.set_ylim(0, 1.2)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(threads)
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "strong_scaling.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    print(stats[["threads", "mean", "std", "speedup", "efficiency"]].to_string(index=False))


# ---------------------------------------------------------------------------
# 2. WEAK SCALING
#
# Ideal: T(p) = T(1) for all p (same per-thread work).
# Efficiency(p) = T(1) / T(p)  [should be 1 for perfect weak scaling]
# ---------------------------------------------------------------------------
def plot_weak_scaling():
    stats = load_stats("weak_omp.csv", ["threads", "N"])
    if stats is None:
        return

    t1 = stats.loc[stats["threads"] == 1, "mean"].values
    if len(t1) == 0:
        print("WARNING: no 1-thread entry in weak_omp.csv")
        return
    t1 = t1[0]

    stats_g = (
        load_stats("weak_omp.csv", ["threads"])
    )
    stats_g["efficiency"] = t1 / stats_g["mean"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Weak Scaling — OpenMP\n(N grows proportionally with threads, K=16, D=8)")

    threads = stats_g["threads"].values

    # Absolute time (should be flat for perfect scaling)
    ax1.errorbar(threads, stats_g["mean"], yerr=stats_g["std"],
                 marker="o", capsize=4, color="tab:blue", label="Measured time")
    ax1.axhline(t1, color="k", linestyle="--", linewidth=1.2, label=f"Ideal (T₁={t1:.3f}s)")
    ax1.set_xlabel("Threads (p)")
    ax1.set_ylabel("Elapsed time (s)")
    ax1.set_title("Wall-clock Time (should be flat)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(threads)

    # Efficiency
    ax2.axhline(1.0, color="k", linestyle="--", linewidth=1.2, label="Ideal (100%)")
    ax2.plot(threads, stats_g["efficiency"], marker="s", color="tab:orange",
             label="Efficiency T(1)/T(p)")
    ax2.set_xlabel("Threads (p)")
    ax2.set_ylabel("Weak scaling efficiency")
    ax2.set_title("Weak Scaling Efficiency")
    ax2.set_ylim(0, 1.3)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(threads)
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "weak_scaling.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    print(stats_g[["threads", "mean", "std", "efficiency"]].to_string(index=False))


# ---------------------------------------------------------------------------
# 3. CUDA BASELINE — throughput vs N
# Also compares CUDA vs OMP (1 thread, 8 threads, 16 threads) for same N.
# ---------------------------------------------------------------------------
def plot_cuda_baseline():
    cuda = load_stats("cuda_baseline.csv", ["N"])
    omp  = load_stats("strong_omp.csv",    ["threads", "N"])

    if cuda is None:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("CUDA Baseline — Throughput and Comparison (K=16, D=8)")

    # Throughput: points per second
    MAXITER = 50  # same as in run_scaling.sh
    cuda["throughput"] = (cuda["N"] * MAXITER) / cuda["mean"]

    ax1.plot(cuda["N"] / 1e6, cuda["throughput"] / 1e6,
             marker="o", color="tab:green", label="CUDA")
    ax1.fill_between(
        cuda["N"] / 1e6,
        (cuda["N"] * MAXITER) / (cuda["mean"] + cuda["std"]) / 1e6,
        (cuda["N"] * MAXITER) / np.maximum(cuda["mean"] - cuda["std"], 1e-9) / 1e6,
        alpha=0.2, color="tab:green"
    )
    ax1.set_xlabel("N (millions of points)")
    ax1.set_ylabel("Throughput (M point-iterations/s)")
    ax1.set_title("CUDA Throughput vs N")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Compare CUDA vs OMP for common N values
    if omp is not None:
        # Try to find OMP times at the largest N in strong scaling (may differ)
        # Use relative speedup: CUDA time / OMP time at same N for 1, 8, 16 threads
        common_n = set(cuda["N"].unique())
        for t_col, col, label in [
            (1,  "tab:blue",   "OMP 1 thread"),
            (8,  "tab:orange", "OMP 8 threads"),
            (16, "tab:red",    "OMP 16 threads"),
        ]:
            sub = omp[omp["threads"] == t_col]
            if sub.empty:
                continue
            # If OMP has only one N (strong scaling fixed N), replicate for all CUDA N
            # as a reference line
            omp_time = sub["mean"].values[0]
            ns = cuda["N"].values
            ratios = omp_time / cuda["mean"].values
            ax2.plot(ns / 1e6, ratios, marker="o", color=col, label=f"vs {label}")

        ax2.axhline(1.0, color="k", linestyle="--", linewidth=1.2, label="Break-even")
        ax2.set_xlabel("N (millions of points)")
        ax2.set_ylabel("Speedup (OMP time / CUDA time)")
        ax2.set_title("CUDA Speedup relative to OMP")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "cuda_baseline.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    print(cuda[["N", "mean", "std", "throughput"]].to_string(index=False))


# ---------------------------------------------------------------------------
# 4. CACHE STUDY
#
# Shows how elapsed time scales with data size relative to L2, L3, RAM.
# A clear jump at the L2→L3 boundary and L3→RAM boundary indicates
# memory-bandwidth bound behaviour.
# ---------------------------------------------------------------------------
def plot_cache_study():
    path = os.path.join(RESULTS_DIR, "cache_study_omp.csv")
    if not os.path.exists(path):
        print(f"WARNING: {path} not found, skipping.")
        return

    df = pd.read_csv(path)
    df["elapsed"] = pd.to_numeric(df["elapsed"], errors="coerce")
    df = df.dropna(subset=["elapsed"])

    stats = (
        df.groupby(["N", "cache_level"])["elapsed"]
          .agg(mean="mean", std="std")
          .reset_index()
          .sort_values("N")
    )
    stats["std"] = stats["std"].fillna(0.0)

    # Throughput: point-iters per second
    MAXITER = 50
    stats["throughput"] = (stats["N"] * MAXITER) / stats["mean"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Cache Study — OMP 8 threads (K=16, D=8)")

    colors = {"L2": "tab:green", "L3": "tab:orange", "RAM": "tab:red"}
    labels = stats["cache_level"].values

    for _, row in stats.iterrows():
        c = colors.get(row["cache_level"], "gray")
        ax1.bar(row["cache_level"], row["mean"], yerr=row["std"],
                color=c, capsize=5, alpha=0.8, label=row["cache_level"])
        ax2.bar(row["cache_level"], row["throughput"] / 1e6,
                color=c, alpha=0.8, label=row["cache_level"])

    ax1.set_xlabel("Data fits in")
    ax1.set_ylabel("Elapsed time (s)")
    ax1.set_title("Time vs Cache Level")
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.set_xlabel("Data fits in")
    ax2.set_ylabel("Throughput (M point-iters/s)")
    ax2.set_title("Throughput vs Cache Level")
    ax2.grid(True, axis="y", alpha=0.3)

    # Annotate with N values
    for ax in (ax1, ax2):
        for i, row in stats.reset_index(drop=True).iterrows():
            ax.text(i, 0.01, f"N={row['N']//1000}K", ha="center",
                    va="bottom", fontsize=8, color="black",
                    transform=ax.transData if False else ax.get_xaxis_transform())

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "cache_study.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    print(stats.to_string(index=False))


# ---------------------------------------------------------------------------
# 5. HYPERTHREADING STUDY
#
# Compare 8 physical cores vs 16 logical (SMT) threads on the same N.
# Expected: modest improvement from HT (10–20%) due to memory latency hiding,
# but not 2× because K-Means is memory-bandwidth bound (high arithmetic
# intensity per thread already saturates the FPU).
# ---------------------------------------------------------------------------
def plot_hyperthreading():
    stats = load_stats("hyperthreading_omp.csv", ["threads"])
    if stats is None:
        return

    t8  = stats.loc[stats["threads"] == 8,  "mean"].values
    t16 = stats.loc[stats["threads"] == 16, "mean"].values

    if len(t8) == 0 or len(t16) == 0:
        print("WARNING: missing 8 or 16 thread entries in hyperthreading_omp.csv")
        return

    t8,  e8  = t8[0],  stats.loc[stats["threads"] == 8,  "std"].values[0]
    t16, e16 = t16[0], stats.loc[stats["threads"] == 16, "std"].values[0]
    ht_gain = (t8 - t16) / t8 * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["8 cores\n(physical)", "16 threads\n(HT/SMT)"],
           [t8, t16], yerr=[e8, e16],
           color=["tab:blue", "tab:purple"], capsize=6, alpha=0.85)
    ax.set_ylabel("Elapsed time (s)")
    ax.set_title(f"Hyperthreading Study\n"
                 f"HT gain: {ht_gain:+.1f}% "
                 f"({'faster' if ht_gain > 0 else 'slower'} with HT)")
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate bars
    for x, (t, e) in enumerate([(t8, e8), (t16, e16)]):
        ax.text(x, t + e + 0.002 * max(t8, t16),
                f"{t:.3f}s\n±{e:.3f}s",
                ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "hyperthreading.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    print(f"  8  threads: {t8:.4f}s ± {e8:.4f}s")
    print(f"  16 threads: {t16:.4f}s ± {e16:.4f}s")
    print(f"  HT gain:    {ht_gain:+.1f}%")


# ---------------------------------------------------------------------------
# 6. OMP vs CUDA combined throughput summary
# ---------------------------------------------------------------------------
def plot_omp_vs_cuda():
    cuda = load_stats("cuda_baseline.csv", ["N"])
    omp  = load_stats("strong_omp.csv",    ["threads"])

    if cuda is None or omp is None:
        return

    MAXITER = 50

    # OMP throughput at various thread counts (fixed N from strong scaling)
    omp_N = None
    path = os.path.join(RESULTS_DIR, "strong_omp.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        omp_N = df["N"].iloc[0] if "N" in df.columns else None

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Throughput: OMP (fixed N) vs CUDA (varying N)\n(K=16, D=8)")

    if omp_N:
        omp["throughput"] = (omp_N * MAXITER) / omp["mean"]
        ax.plot(omp["threads"], omp["throughput"] / 1e6,
                marker="o", color="tab:blue", label="OMP (fixed N)")
        ax.set_xlabel("OMP Threads / CUDA N (M)")

    ax2 = ax.twiny()
    cuda["throughput"] = (cuda["N"] * MAXITER) / cuda["mean"]
    ax2.plot(cuda["N"] / 1e6, cuda["throughput"] / 1e6,
             marker="s", color="tab:green", linestyle="--", label="CUDA (varying N)")
    ax2.set_xlabel("CUDA N (millions of points)", color="tab:green")
    ax2.tick_params(axis="x", labelcolor="tab:green")

    ax.set_ylabel("Throughput (M point-iters/s)")
    ax.legend(loc="upper left")
    ax2.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "omp_vs_cuda.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Strong Scaling ===")
    plot_strong_scaling()
    print()

    print("=== Weak Scaling ===")
    plot_weak_scaling()
    print()

    print("=== CUDA Baseline ===")
    plot_cuda_baseline()
    print()

    print("=== Cache Study ===")
    plot_cache_study()
    print()

    print("=== Hyperthreading ===")
    plot_hyperthreading()
    print()

    print("=== OMP vs CUDA ===")
    plot_omp_vs_cuda()
    print()

    print(f"All plots saved to: {PLOTS_DIR}/")
