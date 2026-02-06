[] AGGIUNGI hardware specification: 
For the CPU– Type of processor (vendor, model name, ecc.)– Number of cores– Whether the CPU uses symmetric multithreading (e.gHyperThreading)– Clock frequency– Amount of RAM– Operating System– Compiler version and possibly compilation flags
For the GPU– Type of GPU (vendor, model name, ecc.)– Number of cores– Clock frequency– Amount of device memory– Compiler version and possibly compilation flags

The topic of this project is describing the design choices and illustrating the scalability and efficiency of the programs using the metrics described above. 
All the variables that cna be possible changed are:
- number of iteration
- number of klaster to find
- number of cores
- number of dimension and point when we ue other data (other then 
File
N. of points (N)
Dimensions (D)
avila.txt
iris.txt
letter-recognition.txt
demo.txt
test-N50000-D80.txt
test-N100000-D30.txt
test-N150000-D20.txt
10430
150
20000
1000
50000
100000
150000
10
4
16
2
80
30
20 ) 
[] USE AS SIRIAL TIME THE parallel program verison with p = 1 processors
[] Amdahl's Law and base speed-up calculation for scaling and speed up (calculate time serial  alpha and actual speed up)
[] Strong Scaling: increase the number of processors p keeping the total problem size fixed
– The total amount of work remains constant
– The amount of work for a single processor decreases as p increases
– Goal: reduce the total execution time by adding more processors
[] Weak Scaling: increase the number of processors p keeping the per-processor work fixed
– The total amount of work grows as p increases
– The amount of work for a single processor remains the same as p increases
– Goal: solve larger problems within the same amount of time
[] How to take in account the hyper treahs (not actual core but virtual?)


# plot.sh

`plot.sh` iterates over `out_*.txt` and `centroids_*.txt` files and uses `gnuplot` to produce `img_###.png` frames for each step. The makefile’s `demo` target uses it to generate frames for a movie.

If you compile `k-means.c` **with** `-DMAKE_MOVIE` defined, the C code will save intermediate `centroids_XXX.txt` and `out_XXX.txt` files each iteration. `plot.sh` reads those and creates one PNG per step.

After frames are produced you can make a movie:

```bash
ffmpeg -pattern_type glob -stream_loop 5 -y -r 1 -i "img_*.png" -vcodec mpeg4 -r 1 demo.avi
```

---

# Output format (what the program writes)

`save_results(FILE *f)` in the provided `k-means.c` produces:

* Header (commented) block listing centroids:

  ```
  # Centroids:
  #  0 : x0 x1 x2 ...
  #  1 : ...
  #
  ```
* Then `N` lines (one per input point) in the same order as input; for each point:

  ```
  <coord_0> <coord_1> ... <coord_D-1> <cluster_id>
  ```

So `cluster_id` is appended at the end of each line as an integer in `0 .. K-1`.

---

# How to force a fixed number of iterations (required for fair benchmarking)

The assignment requires you to modify the code so the program executes a predetermined number of iterations regardless of convergence. Minimal, low-effort approach: add a compile-time macro `NUM_ITER` and use it as the maximum iterations.

Patch suggestion (minimal; apply manually):

1. Open `k-means.c`.
2. Find the `main` function where:

```c
const int MAXITER = 100;
const float TOL = 1e-5;
```

3. Replace with:

```c
#ifndef NUM_ITER
#define NUM_ITER 100
#endif

const int MAXITER = NUM_ITER;
const float TOL = 1e-5;
```

Then compile with:

```bash
gcc -std=c99 -Wall -Wpedantic -O3 k-means.c -o k-means -lm
# or override at compile time:
gcc -std=c99 -Wall -Wpedantic -O3 -DNUM_ITER=500 k-means.c -o k-means -lm
```

For your OpenMP/CUDA ports, adopt the same pattern so you can benchmark reproducibly by changing `-DNUM_ITER=<value>` at compile time.

If you prefer a loop that ignores convergence and only runs `NUM_ITER` iterations, you can change the `do { ... } while` loop to a `for (iter = 0; iter < NUM_ITER; ++iter) { ... }` — but the macro approach above requires minimal edits and keeps the original logic for optional convergence checks.

---

# Notes / tips for parallel implementation

* **Classification step** (assign each point to nearest centroid): parallelize easily over points. Each thread/GPU thread computes the nearest centroid for one (or a batch of) points.

* **Update centroids**: you must accumulate coordinates of points per cluster. For OpenMP:

  * Use per-thread accumulators and combine (reduce) at the end, or use atomic updates. Per-thread accumulators usually give better performance for large `K`/`D`.
  * Example pattern: create `float local_sum[num_threads][K*D]` and `int local_count[num_threads][K]`. Each thread writes into its private arrays; at the end, reduce into global arrays.

* **For CUDA**:

  * The classification kernel is straightforward — each thread handles one point.
  * The update step requires parallel reduction by cluster — more involved: either perform reductions on the device (using per-block shared memory histograms and block-level partial sums then final global reduction) or do classification on GPU and transfer the cluster assignments to host and compute centroids on the CPU (less GPU-bound but simpler).
  * Keep memory transfers minimal: keep points and centroids resident on device memory across iterations.
  * Beware numeric differences due to float reductions; use `float` as in the serial code to keep consistent with reference.

* For timing/benchmarking: measure only the main loop (exclude data read/write) and report both total runtime and per-iteration average.

---

# Explanation of provided files

You supplied the following items — here's what each one does and how to use it:

* `k-means.c`
  The reference serial implementation of Lloyd’s algorithm. It:

  * Reads the input file and stores `N` points of dimension `D`.
  * Initializes `K` centroids (deterministic PRNG seed used in file).
  * Loops: classify points → compute new centroids → measure centroid shift → stop on convergence or `MAXITER`.
  * Writes results with `save_results()` (commented centroid header + `N` lines).
  * If compiled with `-DMAKE_MOVIE`, it writes `centroids_###.txt` and `out_###.txt` every iteration so you can create an animation with `plot.sh` and `ffmpeg`.

* `inputgen.c`
  Small program that generates synthetic input to the expected format (stdout). Usage:

  ```
  ./inputgen points_per_cluster n_dims n_clusters > demo.txt
  ```

  Example: `./inputgen 20 2 50 > demo.txt` produces 1000 2D points.

* `Makefile` (provided in your prompt)

  * Default target builds `k-means` and `inputgen` (serial).
  * `demo` target compiles `k-means.c` with `-DMAKE_MOVIE`, runs a demo dataset, generates `centroids_*.txt` and `out_*.txt`, runs `plot.sh` to create images and `ffmpeg` to produce `demo.avi`.
  * `clean` / `distclean` targets remove artifacts.

* `plot.sh` (provided in your prompt)
  Iterates existing `out_*.txt` and `centroids_*.txt` and uses `gnuplot` to create `img_XXX.png` for each step; intended to be used after compiling the k-means program with `-DMAKE_MOVIE` and producing the `out_*.txt` files.

---

# Example quick workflow (serial → OpenMP → CUDA)

1. Build and test serial:

   ```bash
   gcc -std=c99 -Wall -Wpedantic -O3 k-means.c -o k-means -lm
   ./inputgen 20 2 10 > demo.txt
   ./k-means 5 demo.txt demo.out
   ```

2. Make movie (optional):

   ```bash
   gcc -std=c99 -Wall -Wpedantic -O3 -DMAKE_MOVIE k-means.c -o k-means -lm
   ./inputgen 20 2 10 > demo.txt
   ./k-means 5 demo.txt demo.out
   ./plot.sh
   ffmpeg -pattern_type glob -stream_loop 5 -y -r 1 -i "img_*.png" -vcodec mpeg4 -r 1 demo.avi
   ```

3. Port to OpenMP: produce `omp-k-means.c` and compile with `-fopenmp`:

   ```bash
   gcc -std=c99 -Wall -Wpedantic -fopenmp -O3 omp-k-means.c -o omp-k-means -lm
   OMP_NUM_THREADS=8 ./omp-k-means 5 big_input.txt out.txt
   ```

4. Port to CUDA: produce `cuda-k-means.cu` and compile with `nvcc`. Example:

   ```bash
   nvcc -O3 -arch=sm_80 -o cuda-k-means cuda-k-means.cu -lm
   ./cuda-k-means 5 big_input.txt out_cuda.txt
   ```

---

# Notes about uploaded files / expired uploads

I can see `k-means.c` and `inputgen.c` in the working directory. If you uploaded additional files earlier and they no longer appear, some uploads may have expired; re-upload any missing files and I will incorporate them. If you want, I can produce the OpenMP and a skeleton CUDA port templates from the serial code now — tell me if you want that and I will generate them.

---

# Final checklist (what to produce for submission)

* `omp-k-means` executable (source and compile command in README)
* `cuda-k-means` executable (source and compile command in README)
* `README` (this file) with build/run instructions and notes about `NUM_ITER` for fixed iteration benchmarking
* `Makefile` updated if you want convenient targets for building the two versions (optional)
* A short `REPORT.md` listing measured runtimes for your test inputs and threads/block/grid configurations (optional but useful)

---

If you want, I will:

* produce an OpenMP skeleton (show where to put `#pragma omp parallel for` and per-thread reductions), and/or
* create a CUDA skeleton showing a classification kernel + suggested reduction approach.

Tell me which skeleton you want and I will generate the source file(s) now.
