#!/usr/bin/env bash
# =============================================================================
# run_scaling.sh -- Automates strong and weak scaling experiments for
#                   omp-k-means and cuda-k-means.
#
# Usage:
#   chmod +x run_scaling.sh
#   ./run_scaling.sh
#
# Output:
#   results/strong_omp.csv
#   results/weak_omp.csv
#   results/cuda_baseline.csv
#   results/cache_study_omp.csv
#
# Each CSV has columns:
#   experiment, threads (or N), N, K, D, iters, run, elapsed
#
# The script runs each configuration NRUNS times (for mean ± std).
# Timing is printed by the program itself (Elapsed time line).
#
# Requirements: bash, omp-k-means, cuda-k-means, inputgen (already compiled).
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BINARY_OMP="./omp-k-means"
BINARY_OMP_V2="./omp-k-means-v2"
BINARY_OMP_V3="./omp-k-means-v3"
BINARY_OMP_V4="./omp-k-means-v4"
BINARY_OMP_O3="./omp-k-means-o3"
BINARY_OMP_O3_V4="./omp-k-means-o3-v4"
BINARY_CUDA="./cuda-k-means"
BINARY_CUDA_V2="./cuda-k-means-v2"
BINARY_CUDA_V3="./cuda-k-means-v3"
BINARY_CUDA_V4="./cuda-k-means-v4"
INPUTGEN="./inputgen"

RESULTS_DIR="results"
DATA_DIR="data/scaling"
mkdir -p "$RESULTS_DIR" "$DATA_DIR"

# Number of repeated runs per configuration (for mean ± std).
NRUNS=5

# Fixed K and D for all scaling experiments (K << N, D small as per spec).
K=8
D=40

# Fixed-iteration mode: always run exactly MAXITER iterations so that
# wall-clock time is not affected by convergence speed differences between
# datasets. Compile flags must include -DMAX_ITER_FIXED=MAXITER.
MAXITER=150

# Check that binaries exist.
# for bin in "$BINARY_OMP" "$BINARY_OMP_V2" "$BINARY_OMP_V3" "$BINARY_CUDA" "$INPUTGEN"; do
#     if [ ! -x "$bin" ]; then
#         echo "ERROR: $bin not found or not executable. Run 'make' first."
#         exit 1
#     fi
# done

# ---------------------------------------------------------------------------
# Helper: run a single timing experiment and extract elapsed seconds.
# Usage: run_timed <binary> <threads> <K> <input> <output>
# Returns: elapsed time (printed to stdout).
# ---------------------------------------------------------------------------
run_timed() {
    local bin="$1"
    local threads="$2"
    local k="$3"
    local input="$4"
    local output="$5"

    local elapsed
    elapsed=$( OMP_NUM_THREADS="$threads" "$bin" "$k" "$input" "$output" 2>/dev/null \
               | grep "Elapsed time" | awk '{print $3}' )
    echo "$elapsed"
}

run_timed_bind() {
    local bin="$1"
    local threads="$2"
    local k="$3"
    local input="$4"
    local output="$5"

    local run_output
    run_output=$( OMP_NUM_THREADS="$threads" \
               OMP_PROC_BIND=close \
               OMP_PLACES=threads \
               "$bin" "$k" "$input" "$output" 2>&1 )

    # Show the process binding and places information if the program prints it
    echo "$run_output" | grep -E "Proc bind|Places" || true

    local elapsed
    elapsed=$( echo "$run_output" | grep "Elapsed time" | awk '{print $3}' )
    echo "$elapsed"
}

# ---------------------------------------------------------------------------
# Helper: generate input if not already present.
# Usage: gen_input <points_per_cluster> <D> <K> <output_file>
# Total points = points_per_cluster * K
# ---------------------------------------------------------------------------
gen_input() {
    local ppc="$1"
    local d="$2"
    local k="$3"
    local outfile="$4"
    if [ ! -f "$outfile" ]; then
        echo "  Generating input: N=$(( ppc * k )), D=$d, K=$k → $outfile"
        "$INPUTGEN" "$ppc" "$d" "$k" > "$outfile"
    fi
}


# ---------------------------------------------------------------------------
#  find K and N (OpenMP)
# echo "=== initial test_v2s (OpenMP) ==="

# for SS_N in 20000 2000000 200000; do          # 2M points total
#     SS_PPC=$(( SS_N / K ))

#     SS_INPUT="$DATA_DIR/test_v2_N${SS_N}_D${D}_K${K}.txt"
#     gen_input "$SS_PPC" "$D" "$K" "$SS_INPUT"

#     SS_CSV="$RESULTS_DIR/test_v2_${SS_N}.csv"
#     echo "experiment,threads,N,K,D,iters,run,elapsed" > "$SS_CSV"

#     for THREADS in 1 2 4 6 8 10 12 14 16; do
#         echo "  Threads=$THREADS"
#         for RUN in $(seq 1 $NRUNS); do
#             OUT="$DATA_DIR/tmp_test_v2.out"
#             T=$( run_timed "$BINARY_OMP" "$THREADS" "$K" "$SS_INPUT" "$OUT" )
#             echo "test_v2_${SS_N}_omp,$THREADS,$SS_N,$K,$D,$MAXITER,$RUN,$T" >> "$SS_CSV"
#             rm -f "$OUT"
#         done
#     done

#     python3 utils/plot_speedup.py "test_v2_${SS_N}"

#     echo "  → $SS_CSV"
# done
#---------------------------------------------------------------------------
# O3 version
# ---------------------------------------------------------------------------
# echo "=== Controlla versus v4  ==="
# for version in "v4_enanced"; do
#     for SS_N in 500000; do          # 1000000  500K, 1M, 3M points total        
#         SS_PPC=$(( SS_N / K ))
#         SS_INPUT="$DATA_DIR/strong_O3_v4_bind_N${SS_N}_D${D}_K${K}.txt"
#         gen_input "$SS_PPC" "$D" "$K" "$SS_INPUT"
#         SS_CSV="$RESULTS_DIR/OMP/strong_enanced_${version}.csv"
#         echo "experiment,threads,N,K,D,iters,run,elapsed" > "$SS_CSV"
#         case "$version" in
#             "v1") BINARY="$BINARY_OMP" ;;
#             "v4_enanced") BINARY="$BINARY_OMP_V4" ;;
#             *) echo "Invalid version: $version"; exit 1 ;;
#         esac
#         for THREADS in 1 2 4 6 8 10 12 14 16; do
#             echo "  Threads=$THREADS"
#             for RUN in $(seq 1 $NRUNS); do
#                 OUT="$DATA_DIR/tmp_strong_enanced_${version}.out"
#                 if [ "$version" == "v4_bind" ]; then
#                     T=$( run_timed_bind "$BINARY" "$THREADS" "$K" "$SS_INPUT" "$OUT" )
#                 else
#                     T=$( run_timed "$BINARY" "$THREADS" "$K" "$SS_INPUT" "$OUT" )
#                 fi
#                 echo "strong_enanced_${version},$THREADS,$SS_N,$K,$D,$MAXITER,$RUN,$T" >> "$SS_CSV"
#                 rm -f "$OUT"
#             done
#         done
#         python3 utils/plot_speedup.py "strong_enanced_${version}_${SS_N}" --input "$SS_CSV"
#         echo "  → $SS_CSV"
#     done
# done

# ---------------------------------------------------------------------------
# 1. STRONG SCALING (OpenMP)
#
# Fixed problem size; vary number of threads from 1 to 16
# (includes hyperthreading study: 8 physical vs 16 logical cores).
# N is chosen large enough to avoid timing noise but to fit in RAM.
# ---------------------------------------------------------------------------
# echo "=== Strong Scaling (OpenMP) ==="
# SS_N=500000          # 500_000
# SS_PPC=$(( SS_N / K ))
# SS_INPUT="$DATA_DIR/strong_N${SS_N}_D${D}_K${K}.txt"
# gen_input "$SS_PPC" "$D" "$K" "$SS_INPUT"

# for version in "v4_bind"; do
#     echo "  Version: $version"
#     mkdir -p "$RESULTS_DIR/${version}"
#     SS_CSV="$RESULTS_DIR/${version}/strong_omp.csv"
#     echo "experiment,threads,N,K,D,iters,run,elapsed" > "$SS_CSV"

#     for THREADS in 1 2 4 6 8 10 12 14 16; do
#         echo "  Threads=$THREADS"
#         for RUN in $(seq 1 $NRUNS); do
#             OUT="$DATA_DIR/tmp_strong.out"
#             case "$version" in
#                 "v1") BINARY="$BINARY_OMP" ;;
#                 "v1_bind") BINARY="$BINARY_OMP" ;;
#                 "v2") BINARY="$BINARY_OMP_V2" ;;
#                 "v3") BINARY="$BINARY_OMP_V3" ;;
#                 "v4") BINARY="$BINARY_OMP_V4" ;;
#                 "v4_bind") BINARY="$BINARY_OMP_V4" ;;
#                 *) echo "Invalid version: $version"; exit 1 ;;
#             esac
#             T=$( run_timed_bind "$BINARY" "$THREADS" "$K" "$SS_INPUT" "$OUT" )
#             echo "strong_omp,$THREADS,$SS_N,$K,$D,$MAXITER,$RUN,$T" >> "$SS_CSV"
#             rm -f "$OUT"
#         done
#     done
    
#     python3 utils/plot_speedup.py "strong_omp_${version}_${SS_N}" --input "$SS_CSV"

#     echo "  → $SS_CSV"
# done
# ---------------------------------------------------------------------------
# 2. WEAK SCALING (OpenMP)
#
# Per-thread problem size is kept constant; total N grows with thread count.
# K and D are fixed (spec: "probably ok to keep K and D fixed").
# Per-thread work: N_per_thread = 250000 points.
# Total N = N_per_thread * threads.
# ---------------------------------------------------------------------------
# echo "=== Weak Scaling (OpenMP) ==="

# WS_PER_THREAD=100000 # 100_000 points per thread 

# for version in "v4_bind"; do
#     echo "  Version: $version"
#     WS_CSV="$RESULTS_DIR/${version}/weak_omp.csv"

#     echo "experiment,threads,N,K,D,iters,run,elapsed" > "$WS_CSV"
#     for THREADS in 1 2 4 6 8 10 12 14 16; do
#         WS_N=$(( WS_PER_THREAD * THREADS ))
#         WS_PPC=$(( WS_N / K ))
#         WS_INPUT="$DATA_DIR/weak_N${WS_N}_D${D}_K${K}.txt"
#         gen_input "$WS_PPC" "$D" "$K" "$WS_INPUT"

#         echo "  Threads=$THREADS, N=$WS_N"
#         for RUN in $(seq 1 $NRUNS); do
#             OUT="$DATA_DIR/tmp_weak.out"
#             case "$version" in
#                 "v1") BINARY="$BINARY_OMP" ;;
#                 "v1_bind") BINARY="$BINARY_OMP" ;;
#                 "v2") BINARY="$BINARY_OMP_V2" ;;
#                 "v3") BINARY="$BINARY_OMP_V3" ;;
#                 "v4") BINARY="$BINARY_OMP_V4" ;;
#                 "v4_bind") BINARY="$BINARY_OMP_V4" ;;
#                 *) echo "Invalid version: $version"; exit 1 ;;
#             esac
#             T=$( run_timed_bind "$BINARY" "$THREADS" "$K" "$WS_INPUT" "$OUT" )
#             echo "weak_omp,$THREADS,$WS_N,$K,$D,$MAXITER,$RUN,$T" >> "$WS_CSV"
#             rm -f "$OUT"
#         done
#     done

#     python3 utils/plot_weak_scaling.py "weak_omp_${version}_${WS_N}" --input "$WS_CSV"

#     echo "  → $WS_CSV"
# done


# ----------------------------------------------------------------------------
# NEL CASO v1_bind PERFORMI PEGGIO DI v4, FAI IL TEST CON 1M e verifica cache usando v1 senza bind
# echo "=== Strong Scaling (OpenMP CACHE) ==="
# SS_N=1000000          # 1_000_000
# SS_PPC=$(( SS_N / K ))
# SS_INPUT="$DATA_DIR/strong_N${SS_N}_D${D}_K${K}.txt"
# gen_input "$SS_PPC" "$D" "$K" "$SS_INPUT"

# for version in "v1" "v4" "v4_bind"; do
#     echo "  Version: $version"
#     mkdir -p "$RESULTS_DIR/${version}"
#     SS_CSV="$RESULTS_DIR/${version}/strong_omp_${SS_N}.csv"
#     echo "experiment,threads,N,K,D,iters,run,elapsed" > "$SS_CSV"

#     for THREADS in 1 2 4 6 8 10 12 14 16; do
#         echo "  Threads=$THREADS"
#         for RUN in $(seq 1 $NRUNS); do
#             OUT="$DATA_DIR/tmp_strong.out"
#             case "$version" in
#                 "v1") BINARY="$BINARY_OMP" ;;
#                 "v2") BINARY="$BINARY_OMP_V2" ;;
#                 "v3") BINARY="$BINARY_OMP_V3" ;;
#                 "v4") BINARY="$BINARY_OMP_V4" ;;
#                 "v4_bind") BINARY="$BINARY_OMP_V4" ;;
#                 *) echo "Invalid version: $version"; exit 1 ;;
#             esac
#             if [ "$version" == "v4_bind" ]; then
#                 T=$( run_timed_bind "$BINARY" "$THREADS" "$K" "$SS_INPUT" "$OUT" )
#             else
#                 T=$( run_timed "$BINARY" "$THREADS" "$K" "$SS_INPUT" "$OUT" )
#             fi
#             echo "strong_omp,$THREADS,$SS_N,$K,$D,$MAXITER,$RUN,$T" >> "$SS_CSV"
#             rm -f "$OUT"
#         done
#     done
    
#     python3 utils/plot_speedup.py "strong_omp_${version}_${SS_N}" --input "$SS_CSV"

#     echo "  → $SS_CSV"
# done


# ---------------------------------------------------------------------------
# 3. CUDA BASELINE
#
# Run cuda-k-means with the same inputs used in strong scaling,
# plus varying N to study GPU throughput.
# Compare against OMP with 1, 8, and 16 threads for the same N.
# ---------------------------------------------------------------------------
# Cuda v3 con N multipli di grid size N = 522_240 1_044_480 2_088_960
# ---------------------------------------------------------------------------
# echo "=== CUDA OPTIMUM ==="

# D=40
# for version in "v3"; do
#     echo "  Version: $version"
#     CUDA_CSV="$RESULTS_DIR/cuda/cuda_${version}_optimum.csv"
#     # echo "experiment,N,K,D,iters,run,elapsed" > "$CUDA_CSV"
#     case "$version" in
#         "v1") BINARY="$BINARY_CUDA" ;;
#         "v2") BINARY="$BINARY_CUDA_V2" ;;
#         "v3") BINARY="$BINARY_CUDA_V3" ;;
#         "v4") BINARY="$BINARY_CUDA_V4" ;;
#         *) echo "Invalid version: $version"; exit 1 ;;
#     esac
#     for CUDA_N in 522240 1044480 2088960 4177920 8355840; do
#         CUDA_PPC=$(( CUDA_N / K ))
#         CUDA_INPUT="$DATA_DIR/cuda_N${CUDA_N}_D${D}_K${K}.txt"
#         gen_input "$CUDA_PPC" "$D" "$K" "$CUDA_INPUT"

#         echo "  CUDA N=$CUDA_N"
#         for RUN in $(seq 1 $NRUNS); do
#             OUT="$DATA_DIR/tmp_cuda.out"
#             T=$( "$BINARY" "$K" "$CUDA_INPUT" "$OUT" 2>/dev/null \
#                 | grep "Elapsed time" | awk '{print $3}' )
#             echo "cuda_${version},$CUDA_N,$K,$D,$MAXITER,$RUN,$T" >> "$CUDA_CSV"
#             rm -f "$OUT"
#         done
#     done
#     echo "  → $CUDA_CSV"
# done

# VS cuda v2 40

for version in "omp_v4_o3" "omp_v4"; do
    echo "  Version: $version"
    CUDA_CSV="$RESULTS_DIR/cuda_vs_omp/${version}.csv"
    echo "experiment,N,K,D,iters,run,elapsed" > "$CUDA_CSV"
    case "$version" in
        "v1") BINARY="$BINARY_CUDA" ;;
        "v2") BINARY="$BINARY_CUDA_V2" ;;
        "v3") BINARY="$BINARY_CUDA_V3" ;;
        "v4") BINARY="$BINARY_CUDA_V4" ;;
        "omp_v4") BINARY="$BINARY_OMP_V4" ;;
        "omp_v4_o3") BINARY="$BINARY_OMP_O3_V4" ;;
        *) echo "Invalid version: $version"; exit 1 ;;
    esac
    for CUDA_N in 500000 1000000 2000000 4000000 8000000; do
        CUDA_PPC=$(( CUDA_N / K ))
        CUDA_INPUT="$DATA_DIR/cuda_N${CUDA_N}_D${D}_K${K}.txt"
        gen_input "$CUDA_PPC" "$D" "$K" "$CUDA_INPUT"

        echo "  N=$CUDA_N"
        for RUN in $(seq 1 $NRUNS); do
            OUT="$DATA_DIR/tmp_cuda.out"
            T=$( run_timed "$BINARY" "16" "$K" "$CUDA_INPUT" "$OUT")
            echo "cuda_${version},$CUDA_N,$K,$D,$MAXITER,$RUN,$T" >> "$CUDA_CSV"
            rm -f "$OUT"
        done
    done
    echo "  → $CUDA_CSV"
done

# ---------------------------------------------------------------------------
# 4. CACHE STUDY (OpenMP, 8 threads)
#
# test_v2s data sizes that fit in different cache levels of the CPU.
# Goal: observe cache effects on omp-k-means throughput.
#
# CPU cache sizes (Ryzen 7 7800X3D):
#   L2 per core: 1 MB → ~262K floats. N=200K fits if K and D are small.
#   L3 shared:  96 MB → ~25M floats. N*D floats: N=3M, D=8 → 24MB ≈ L3.
#   Overflow:  N=10M → 80MB > L3 → guaranteed cache miss pressure.
#
# data[] size = N * D * 4 bytes:
#   L2  (1 MB/core):  N=32768  → 32768*8*4 = 1.0 MB (single core L2)
#   L3  (96 MB):      N=3000000 → 3M*8*4 = 96 MB ≈ full L3
#   RAM overflow:     N=10000000 → 10M*8*4 = 320 MB > L3
# ---------------------------------------------------------------------------
# echo "=== Cache Study (OpenMP, 8 threads) ==="

# CACHE_CSV="$RESULTS_DIR/cache_study_omp.csv"
# echo "experiment,N,K,D,cache_level,iters,run,elapsed" > "$CACHE_CSV"

# declare -A CACHE_LABELS
# CACHE_LABELS[32768]="L2"
# CACHE_LABELS[3000000]="L3"
# CACHE_LABELS[10000000]="RAM"

# for CACHE_N in 32768 3000000 10000000; do
#     CACHE_PPC=$(( CACHE_N / K ))
#     # Ensure PPC >= 1
#     if [ "$CACHE_PPC" -lt 1 ]; then CACHE_PPC=1; fi
#     CACHE_INPUT="$DATA_DIR/cache_N${CACHE_N}_D${D}_K${K}.txt"
#     gen_input "$CACHE_PPC" "$D" "$K" "$CACHE_INPUT"

#     LABEL="${CACHE_LABELS[$CACHE_N]}"
#     echo "  Cache study N=$CACHE_N ($LABEL)"
#     for RUN in $(seq 1 $NRUNS); do
#         OUT="$DATA_DIR/tmp_cache.out"
#         T=$( run_timed "$BINARY_OMP" "8" "$K" "$CACHE_INPUT" "$OUT" )
#         echo "cache_omp,$CACHE_N,$K,$D,$LABEL,$MAXITER,$RUN,$T" >> "$CACHE_CSV"
#         rm -f "$OUT"
#     done
# done

# echo "  → $CACHE_CSV"

#----------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "All experiments complete. Results in: $RESULTS_DIR/"
ls -lh "$RESULTS_DIR/"
