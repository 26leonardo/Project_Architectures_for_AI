# Parallel optimization of Lloyd’s algorithm (K-Means)

This repository contains a working **serial** implementation of Lloyd’s K-Means (provided as `k-means.c`) plus a small input generator (`inputgen.c`) and helper scripts.   
There are also two **parallel** executables:

* `omp-k-means` — multicore version using **OpenMP** (gcc)
* `cuda-k-means` — GPU version using **CUDA** (nvcc)


## Requirements / dependencies

* GCC (for building OpenMP executable). 
* GNU `make`.
* NVIDIA CUDA Toolkit (nvcc) matching your GPU and driver.
* `gnuplot` (optional — for plotting).
* `ffmpeg` (optional — to encode frames into a movie).
* `time` (optional — use `/usr/bin/time -v` for detailed timing).
* Standard C math library (libm) is used by code; link with `-lm` when needed.


## How to build

These are minimal, explicit commands you can run from the project directory.

#### 1) Build the provided serial program (reference)

```bash
gcc -std=c99 -Wall -Wpedantic -O3 k-means.c -o k-means -lm
```

#### 2) Build the OpenMP version (recommended approach)

Compile command of `omp-k-means.c`:

```bash
gcc -std=c99 -Wall -Wpedantic -fopenmp -O3 omp-k-means.c -o omp-k-means -lm
```

>Notes:
>* `-fopenmp` enables OpenMP parallel regions.
>* `-O3` recommended for performance.
>* Use `-DNUM_ITER=100` (see fixed-iteration instructions).

#### 3) Build the CUDA version

Compile command of  `cuda-k-means.cu`:

```bash
nvcc -O3 -arch=sm_80 -o cuda-k-means cuda-k-means.cu -lm
```

>Notes:
>* Replace `sm_80` with the compute capability of your GPU (e.g., `sm_75`, `sm_86`, etc.). You can also use `-gencode` flags.
>* For debugging or profiling add `-G` (debug) or `-lineinfo` for profiling.

## How to run

Basic usage (same for serial / omp / cuda once built):

```bash
./<executable> K input_file output_file
# Example:
./omp-k-means 5 demo.txt demo.out
```

* `K` is the number of clusters.
* `input_file` must be plain text with `N` rows, each row contains `D` floating-point numbers (space/tab separated).
* `output_file` will contain a header block with centroid coordinates (commented lines starting with `#`) followed by `N` lines with the original point coordinates and the cluster id appended at the end of each line.

You can measure wall-clock time with:

```bash
/usr/bin/time -v ./omp-k-means 10 big_input.txt out.txt
```


## Generating input

The repository includes `inputgen.c`. Usage:

```bash
./inputgen points_per_cluster n_dims n_clusters > input.txt
```

Example used by the makefile demo:

```bash
./inputgen 20 2 50 > demo.txt
# this produces N = 20 * 50 = 1000 points in 2D
```

`inputgen` prints `N` rows to stdout; redirect to a file.


## Makefile and demo

A `Makefile` is included. It defines:

Build everything:

```
make
```

Build only OpenMP:

```
make omp-k-means
```

Build only CUDA:

```
make cuda-k-means
```

Generate demo movie:

```
make demo
```

Run serial:

```
./k-means 5 data/iris.txt out.txt
```

Run OpenMP with 8 threads:

```
OMP_NUM_THREADS=8 ./omp-k-means 5 data/test-N100000-D30.txt out.txt
```

Run CUDA:

```
./cuda-k-means 5 data/test-N150000-D20.txt out.txt
```

## Important technical params

1. Change `-arch=sm_80` to match your GPU.
   Example:

   * RTX 2060 → sm_75
   * RTX 3060 → sm_86
   * Check with: `nvidia-smi`

2. If CUDA fails to link math:
   You may remove `-lm` for CUDA target (nvcc usually handles it).

3. If you want deterministic benchmarking:
   Add `-DNUM_ITER=100` to `CFLAGS`.

---


