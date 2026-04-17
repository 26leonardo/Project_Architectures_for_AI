# K-Means Clustering ( OpenMP & CUDA )

Develop of the parallel implementations of Lloyd's K-Means clustering algorithm using CUDA and OpenMP, with relative study of the performance and bottolneck given the architecture and code.


## Files

| File | Description |
|---|---|
| `report.pdf` | Report of the implementation/performance |
| `src/omp-k-means.c` | OpenMP parallelization |
| `src/cuda-k-means.cu` | CUDA parallelization |
| `utils/inputgen.c` | Synthetic input generator |
| `utils/hpc.h` | Timing utilities |

---

## Compile & Run

### OpenMP

```bash
gcc -DMAX_ITER_FIXED=150 -std=c99 -Wall -Wpedantic -fopenmp \
    -o omp-k-means omp-k-means.c 
```

```bash
OMP_NUM_THREADS=16 ./omp-k-means K input_file output_file
```

### CUDA

```bash
nvcc -DMAX_ITER_FIXED=150 -arch=sm_89 -o cuda-k-means cuda-k-means.cu 
```

```bash
./cuda-k-means K input_file output_file
```

---

## Flags

| Flag | Required | Description |
|---|---|---|
| `-DMAX_ITER_FIXED=N` | Recommended | Run exactly N iterations instead of waiting for convergence. Makes timing reproducible across datasets. Without it the algorithm stops when centroids stop moving. |
| `-O3` | Optional (OMP only) | Enables vectorization and heavy optimizations. |
| `-arch=sm_XX` | Recommended (CUDA) | Target GPU architecture. Use `sm_89` for RTX 40-series, `sm_86` for RTX 30-series, `sm_80` for A100. If omitted nvcc defaults to an older architecture and may miss optimizations. |

### Optional OpenMP environment variables

```bash
# Pin threads to physical cores (recommended for reproducibility)
OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=16 ./omp-k-means K in out
```

