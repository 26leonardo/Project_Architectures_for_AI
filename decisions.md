# K-Means Parallelization — Design Decisions

## 1. Parallel Pattern Choice

### 1.1 What was chosen: PARTITION + REDUCE

Lloyd's algorithm has two phases per iteration:

| Phase | Work | Parallel pattern |
|---|---|---|
| `classify()` | For each of N points, find nearest of K centroids (D ops each) | **PARTITION** over N |
| `update_centroids()` | Accumulate N points into K accumulators, then normalise | **REDUCE** (scatter → sum) |

#### Why PARTITION for classify?

The classify step is **embarrassingly parallel at the point level**: point `i` reads `data[i*D..(i+1)*D-1]` and the K centroids, and writes only `cluster_of[i]`. There is zero data dependency between different points. With N >> K, the N outer iterations dominate the total work, so splitting them evenly across P threads (or GPU thread blocks) gives near-ideal load balance.

**Static scheduling** (`schedule(static)`) is chosen over dynamic because every point performs exactly K*D floating-point operations regardless of its coordinates — there is no per-point work variance. Static avoids the overhead of the work queue that dynamic requires.

#### Why REDUCE for update_centroids?

The update step accumulates each point's coordinates into `new_centroids[cluster_of[i]]`. This is a **scatter-reduce**: the destination index `cluster_of[i]` is not monotone, so writes from different threads can collide on the same centroid. This is a textbook reduction pattern.

---

## 2. OpenMP Implementation Decisions

### 2.1 Thread-private accumulators (not atomic-per-point)

**Problem:** both `counts[j]` and `new_centroids[j*D + d]` are accumulated over N points with non-deterministic write order. Naive `#pragma omp atomic` inside the inner loop would fire N*K or N*K*D atomics per iteration, serialising the hot path.

**Solution:** each thread allocates a private copy of `counts` (`local_counts[K]`) and `new_centroids` (`local_nc[K*D]`). After the `omp for` loop, only K and K*D atomic additions are needed per thread — this is O(P*K) rather than O(N*K), which is negligible since K << N.

**Why not `reduction` clause?** OpenMP 4.5 supports user-defined reductions, but array reductions on dynamically-sized arrays require a manual reduction function. The private-copy approach is simpler, compiler-portable, and equally efficient.

### 2.2 Parallel region structure: one region per function

Rather than reusing the single `#pragma omp parallel` block from the original stub (which wrapped the entire `do-while`), each function creates and destroys its own parallel region. This is intentional:

- `classify()` and `update_centroids()` are called sequentially; there is an implicit barrier between them (update reads `cluster_of[]` written by classify).
- The implicit barrier at the end of each parallel region correctly enforces this dependency.
- Reusing one outer region would require careful use of `#pragma omp single` or `#pragma omp barrier` to prevent classify and update from running concurrently — more complex, same performance.

### 2.3 Serial sections: init, normalise, I/O

- `init_centroids()` uses `rand()`, which is **not thread-safe**. It is kept entirely serial.
- The normalisation (Phase B of update_centroids) is O(K*D) — negligible vs O(N) — and is kept serial after the parallel accumulation.
- I/O functions are kept serial as directed.

---

## 3. CUDA Implementation Decisions

### 3.1 Kernel decomposition: one thread per point

The dominant cost is the classify step: O(N*K*D) work distributed over N independent units. One thread per point maximises parallelism along the dominant axis.

**Alternative considered: one thread per (point × dimension)**  
This would give N*D threads. However, it requires a reduction over D dimensions to produce the per-(point,cluster) distance, which needs shared memory and `__syncthreads()`. When D is small (the assumed case), the overhead of the extra synchronisation and the reduced occupancy (larger blocks) outweighs the benefit. One-thread-per-point is simpler and sufficient.

### 3.2 Centroids in `__constant__` memory

During classify, all threads read the same K*D centroid values. `__constant__` memory is:
- **Broadcast**: all threads in a warp reading the same address get it in a single cycle (vs one L1 hit per thread for global memory).
- **Cached**: fits in the 64 KB constant cache on all modern NVIDIA architectures.
- **Size limit**: K*D ≤ 16384 floats = 64 KB. This covers K=256, D=64, which is well beyond the K << N regime assumed by the problem.

### 3.3 Memory access and coalescing

**data[] layout: row-major** (`data[i*D + d]`)

| Thread offset | Address accessed at d=0 |
|---|---|
| i   | i*D + 0 |
| i+1 | (i+1)*D + 0 |
| … | … |

Adjacent threads access addresses **D floats apart** — stride-D access, **not coalesced** for D > 1.

**The coalesced alternative** would be column-major storage (`data[d*N + i]`): at fixed d, thread i accesses `d*N + i`, thread i+1 accesses `d*N + i+1` — stride-1, perfectly coalesced. However:
- Transposing the data array requires either preprocessing (one-time cost) or maintaining a separate transposed copy.
- Column-major breaks the sequential access pattern within a single thread (now strided in d), trading one kind of inefficiency for another.
- For small D, the bandwidth wasted by strided accesses is proportional to D-1 wasted cache line slots — modest when D is small.

**Recommendation for further study**: transpose `data` to column-major before the main loop and re-run — expected speedup of 1.5–2× on the classify kernel for D ≥ 4.

### 3.4 atomicAdd in accumulate_kernel

Each thread i atomically adds D floats into `new_centroids[cluster_of[i]*D .. +D]`. With N points and K centroids, each centroid receives on average N/K updates. Since K << N, N/K is large — but these are spread over many warps in time, so atomics do not fully serialise. On RTX 4060 Ti (sm_89), `atomicAdd` on `float` is natively supported and efficient.

---

## 4. Memory Access Patterns — Row vs Column Summary

| Location | Layout | Inner loop | Cache behaviour |
|---|---|---|---|
| `data[]` (both OMP and CUDA) | Row-major | Iterates d (inner) | **OMP**: sequential → cache-friendly. **CUDA**: stride-D → not coalesced for D>1. |
| `centroids[]` | Row-major | Iterates d | Small (K*D << N); fits in L1/L2 for OMP; in constant cache for CUDA. |
| `new_centroids[]` | Row-major | Scatter write | OMP: private copy per thread (L1-resident, K small). CUDA: atomicAdd. |

---

## 5. Hyperthreading Analysis

The Ryzen 7 7800X3D has **8 physical cores** and **16 logical threads** via AMD SMT (Simultaneous Multithreading, equivalent to Intel HyperThreading).

### Expected behaviour for K-Means

K-Means classify is:
- **Compute-bound** (K*D FP multiplies and adds per point), but with D and K small, the working set per thread is tiny.
- **Memory-bandwidth bound** at large N: the data array does not fit in any single core's L2, so threads stall on cache misses from the L3 or DRAM.

When the bottleneck is **memory bandwidth**, SMT helps by having the second logical thread issue memory requests while the first is stalled — effectively hiding latency. Expected HT benefit: **10–20% speedup** (not 2×, because bandwidth is the actual limit and both logical threads share it).

When the bottleneck is **compute** (small N fitting in L2/L3), SMT may offer no benefit or even slight regression (contention for shared execution ports).

**Hypothesis to test**: for data fitting in L3 (N ≈ 3M, D=8 → 96 MB), 16 threads should offer a measurable improvement over 8. For data in L2 (N ≈ 32K), the benefit should be smaller or zero.

---

## 6. Fixed-Iteration Mode

The convergence check `(shift > TOL) && (iter <= MAXITER)` makes the number of iterations depend on the input data. Two datasets of the same size N can converge in 5 or 50 iterations, making wall-clock time not comparable for scaling studies.

**Solution**: compile with `-DMAX_ITER_FIXED=50` (or any constant). The code then runs exactly 50 iterations regardless of convergence. Since all configurations use the same MAXITER, the total work is proportional to N*K*D*MAXITER — making timing directly comparable across datasets of the same N.

The original convergence check is preserved and remains the default when `MAX_ITER_FIXED` is not defined.

---

## 7. Speedup Measurement Methodology

Following the theory:
- **Baseline**: `omp-k-means` compiled with `-fopenmp`, run with `OMP_NUM_THREADS=1`. This avoids comparing against a structurally different serial binary.
- **Speedup(p)** = T(1) / T(p).
- Each configuration is run **NRUNS=5 times**; mean ± standard deviation are reported.
- Input size N is chosen large (≥ 500K points) to ensure the computation time dominates over OpenMP thread-creation overhead.

---

## 8. Additional Experiments (Recommended)

| Experiment | What to study | How |
|---|---|---|
| **Transposed data layout** | Coalescing in CUDA | Transpose `data` to column-major before the main loop; compare classify kernel time. |
| **`-O3` flag** | Compiler vectorisation | Compile both versions with `-O3`, compare against baseline. Note: SIMD auto-vectorisation of the D-loop in `sqdist`. |
| **schedule(dynamic) vs static** | Load balance | For artificially unequal clusters (some have 10× more points), dynamic may outperform static. |
| **Vary K (fixed N, D)** | Atomic contention in CUDA | Larger K → less contention per centroid in `atomicAdd`; smaller K → more. |
| **Vary D (fixed N, K)** | Cache miss rate and FP intensity | Larger D increases both work and data volume; tests whether the code is compute- or bandwidth-bound. |
| **GPU L2 cache study** | GPU memory hierarchy | RTX 4060 Ti has 32 MB L2. N*D*4 = 32 MB → N=1M, D=8. Compare N=500K (fits) vs N=4M (overflow). |
| **Shared-memory histogram** | Reduce atomic contention | Replace `atomicAdd` in `count_kernel` with a block-level shared-memory histogram, then a single global atomic per block. Beneficial when K is very small (high contention). |
