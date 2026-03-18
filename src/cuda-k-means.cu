/****************************************************************************
 *
 * cuda-k-means.cu -- CUDA parallelization of the K-Means clustering algorithm.
 *
 * Based on k-means.c by Moreno Marzolla
 * <https://unibo.it/sitoweb/moreno.marzolla/>
 *
 * Parallelization by: [student]
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * --------------------------------------------------------------------------
 *
 * ## Parallel Pattern Choice: PARTITION + REDUCE
 *
 * ### Kernel decomposition: ONE THREAD PER POINT
 *
 * The dominant cost of Lloyd's algorithm is classify(): for each of the
 * N points, compute K distances of D operations each → O(N*K*D) total.
 * Since N >> K and N >> D, the natural parallelism axis is N.
 *
 * We assign one GPU thread per data point. This gives:
 *   - Maximum parallelism: N threads, N >> K (typically millions vs. tens).
 *   - Simple control flow: every thread does identical work → no warp divergence
 *     in the distance loop (all threads in a warp iterate the same j and d).
 *   - Alternative (one thread per (point × dimension)) would give N*D threads
 *     but requires an additional reduction over D per (point,cluster) pair,
 *     introducing shared-memory complexity that is not warranted when D is small.
 *
 * ### Memory layout and coalescing
 *
 * data[] is stored ROW-MAJOR: data[i*n_dims + d].
 * Thread i accesses data[i*n_dims + 0], data[i*n_dims + 1], ...,
 * data[i*n_dims + D-1] sequentially.
 *
 * In a warp of 32 threads (i, i+1, ..., i+31), at dimension d=0:
 *   thread i   → data[i*D + 0]
 *   thread i+1 → data[(i+1)*D + 0]
 *   ...
 * These addresses are D floats apart, NOT consecutive unless D=1.
 * For D > 1 the accesses are STRIDED → NOT coalesced.
 *
 * The alternative layout COLUMN-MAJOR (data[d*n_points + i]) would make
 * warp accesses at fixed d stride-1 (perfectly coalesced), but would
 * destroy the sequential access pattern within a single thread.
 * For large D it would be worth transposing; for small D (as assumed
 * here) the row-major layout is kept to preserve readability and match
 * the serial code. A transposed layout experiment is recommended.
 *
 * centroids[] (K*D floats, K<<N) is placed in GPU __constant__ memory:
 *   - Read-only during classify.
 *   - Broadcast to all threads in a warp in a single cycle (constant cache).
 *   - Fits easily: K*D floats × 4 bytes << 64 KB constant memory limit.
 *
 * ### update_centroids kernel
 *
 * Each thread i atomically adds data[i] into new_centroids[cluster_of[i]].
 * Because cluster_of[i] is not ordered, writes are scattered → atomicAdd
 * is required. This is acceptable because:
 *   - Only D atomic adds per point (D small).
 *   - K << N: contention per centroid is N/K on average, spread over
 *     many warps in time → atomics do not fully serialise.
 *
 * ### Fixed-iteration mode
 * Define MAX_ITER_FIXED at compile time (-DMAX_ITER_FIXED=200) to run a
 * predetermined number of iterations for reproducible timing.
 *
 * Compile with:
 *      nvcc -arch=sm_89 cuda-k-means.cu -o cuda-k-means -lm
 *      (adjust -arch for your GPU: RTX 4060 Ti → sm_89)
 *
 * Run with:
 *      ./cuda-k-means K input_file output_file
 *
 ****************************************************************************/

#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif

#include "../utils/hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

/* --------------------------------------------------------------------------
 * Constant memory for centroids.
 * During classify() centroids are read-only and accessed by every thread.
 * Constant memory is cached and broadcast efficiently to all threads in a warp.
 * Limit: 64 KB → max K*D = 16384 floats (e.g. K=256, D=64).
 * -------------------------------------------------------------------------- */
#define MAX_CENTROIDS_FLOATS 16384
__constant__ float d_centroids_const[MAX_CENTROIDS_FLOATS];

/* --------------------------------------------------------------------------
 * Global variables (host side; same structure as serial version).
 * -------------------------------------------------------------------------- */
int    n_dims;
int    n_points;
int    n_clusters;

float *data;            /* host: [n_points * n_dims]     */
float *centroids;       /* host: [n_clusters * n_dims]   */
float *new_centroids;   /* host: [n_clusters * n_dims]   */
int   *counts;          /* host: [n_clusters]            */
int   *cluster_of;      /* host: [n_points]              */

/* Device pointers. */
float *d_data;          /* device copy of data           */
float *d_centroids;     /* device copy of centroids (for update kernel) */
float *d_new_centroids; /* device accumulator            */
int   *d_counts;        /* device cluster sizes          */
int   *d_cluster_of;    /* device cluster assignments    */

/* --------------------------------------------------------------------------
 * Helpers
 * -------------------------------------------------------------------------- */
void *safe_malloc(size_t size)
{
    void *r = malloc(size);
    assert(r != NULL);
    return r;
}

/* Flattened row-major index: row i, column d, n_dims columns. */
__host__ __device__ int IDX(int i, int d, int dims)
{
    return i * dims + d;
}

/* --------------------------------------------------------------------------
 * KERNEL: classify_kernel
 *
 * One thread per data point. Each thread:
 *   1. Reads its point from d_data (row-major, stride-D → not coalesced for D>1).
 *   2. Computes K squared distances using centroids from constant memory
 *      (broadcast, no global memory traffic).
 *   3. Writes its nearest cluster ID to d_cluster_of (coalesced: consecutive
 *      threads write consecutive elements).
 *
 * No synchronisation needed: threads write to disjoint d_cluster_of[i].
 * counts[] is NOT updated here to avoid atomics in the hot path; a separate
 * kernel (count_kernel) handles that.
 * -------------------------------------------------------------------------- */
__global__ void classify_kernel(
    const float * __restrict__ d_data,
    int          * __restrict__ d_cluster_of,
    int n_points, int n_clusters, int n_dims)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points) return;  /* guard: last block may have extra threads */

    /* Find nearest centroid for point i. */
    int   nearest = 0;
    float mindist = 0.0f;

    /* First centroid distance. */
    for (int d = 0; d < n_dims; d++) {
        const float diff = d_data[IDX(i, d, n_dims)] - d_centroids_const[IDX(0, d, n_dims)];
        mindist += diff * diff;
    }

    for (int j = 1; j < n_clusters; j++) {
        float dist = 0.0f;
        for (int d = 0; d < n_dims; d++) {
            /* d_data access: thread i reads d_data[i*n_dims + d].
               Adjacent threads (i, i+1) read (i*n_dims+d, (i+1)*n_dims+d):
               offset = n_dims → stride-n_dims access pattern, not coalesced.
               Coalescing would require column-major layout. */
            const float diff = d_data[IDX(i, d, n_dims)] - d_centroids_const[IDX(j, d, n_dims)];
            dist += diff * diff;
        }
        if (dist < mindist) {
            mindist = dist;
            nearest = j;
        }
    }

    d_cluster_of[i] = nearest;
}

/* --------------------------------------------------------------------------
 * KERNEL: count_kernel
 *
 * Counts how many points belong to each cluster.
 * One thread per point; atomicAdd into d_counts[cluster_of[i]].
 * Contention: N/K threads compete per bucket on average; acceptable when K
 * is not too small. Could use shared-memory histogram for large N and small K.
 * -------------------------------------------------------------------------- */
__global__ void count_kernel(
    const int * __restrict__ d_cluster_of,
    int       * __restrict__ d_counts,
    int n_points)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points) return;
    atomicAdd(&d_counts[d_cluster_of[i]], 1);
}

/* --------------------------------------------------------------------------
 * KERNEL: accumulate_kernel
 *
 * For each point i, atomically add data[i] to new_centroids[cluster_of[i]].
 * This is a scatter reduction: D atomicAdd calls per point.
 *
 * atomicAdd on float is supported since compute capability 2.0.
 * Contention per centroid: ~N/K additions on average, which is manageable
 * because K<<N (many distinct centroid targets spread the atomics).
 *
 * Memory access:
 *   - d_data[i*n_dims + d]: stride-n_dims for adjacent threads → not coalesced.
 *   - d_new_centroids[cluster_of[i]*n_dims + d]: scattered write.
 * -------------------------------------------------------------------------- */
__global__ void accumulate_kernel(
    const float * __restrict__ d_data,
    const int   * __restrict__ d_cluster_of,
    float       * __restrict__ d_new_centroids,
    int n_points, int n_dims)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points) return;

    const int c = d_cluster_of[i];
    for (int d = 0; d < n_dims; d++) {
        atomicAdd(&d_new_centroids[IDX(c, d, n_dims)], d_data[IDX(i, d, n_dims)]);
    }
}

/* --------------------------------------------------------------------------
 * KERNEL: normalise_kernel
 *
 * Divides each centroid accumulator by its count to get the mean.
 * One thread per (centroid, dimension) pair.
 * K*D is small (K<<N, D small) → uses a small grid.
 * Also handles empty clusters (count == 0) by copying the old centroid.
 * -------------------------------------------------------------------------- */
__global__ void normalise_kernel(
    float       * __restrict__ d_new_centroids,
    const float * __restrict__ d_centroids_old,
    const int   * __restrict__ d_counts,
    int n_clusters, int n_dims)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_clusters) return;

    if (d_counts[j] == 0) {
        /* Empty cluster: preserve old centroid. */
        for (int d = 0; d < n_dims; d++)
            d_new_centroids[IDX(j, d, n_dims)] = d_centroids_old[IDX(j, d, n_dims)];
    } else {
        const float inv = 1.0f / (float)d_counts[j];
        for (int d = 0; d < n_dims; d++)
            d_new_centroids[IDX(j, d, n_dims)] *= inv;
    }
}

/* --------------------------------------------------------------------------
 * Host: compute max centroid shift (serial, K*D operations → negligible).
 * Called after copying new_centroids back to host.
 * -------------------------------------------------------------------------- */
float compute_shift(const float *old_c, const float *new_c, int n_clusters, int n_dims)
{
    float maxshift = 0.0f;
    for (int j = 0; j < n_clusters; j++) {
        float dist = 0.0f;
        for (int d = 0; d < n_dims; d++) {
            const float diff = old_c[IDX(j, d, n_dims)] - new_c[IDX(j, d, n_dims)];
            dist += diff * diff;
        }
        if (dist > maxshift) maxshift = dist;
    }
    return maxshift;
}

/* --------------------------------------------------------------------------
 * Vector utilities (host side, for init and I/O).
 * -------------------------------------------------------------------------- */
static void vzero_h( float *p, int n_dims )
{
    for (int d = 0; d < n_dims; d++) p[d] = 0.0f;
}

static void vcopy_h( float *p1, const float *p2, int n_dims )
{
    for (int d = 0; d < n_dims; d++) p1[d] = p2[d];
}

/* --------------------------------------------------------------------------
 * init_centroids (host, NOT parallelized: rand() is not thread-safe).
 * -------------------------------------------------------------------------- */
void init_centroids( void )
{
    int select    = n_clusters;
    int remaining = n_points;
    for (int i = 0; (i < n_points) && (select > 0); i++) {
        if ((rand() % remaining) < select) {
            select--;
            vcopy_h( &centroids[IDX(select, 0, n_dims)],
                     &data[IDX(i, 0, n_dims)], n_dims );
        }
        remaining--;
    }
}

/* --------------------------------------------------------------------------
 * Input/Output (DO NOT parallelize).
 * -------------------------------------------------------------------------- */
void read_input( FILE *f )
{
    const size_t BUFLEN = 1024;
    char buffer[BUFLEN];

    char *i_dont_care = fgets(buffer, BUFLEN, f);
    (void)i_dont_care;
    n_dims = -1;
    char *start, *end = buffer;
    do {
        start = end;
        strtof(start, &end);
        n_dims++;
    } while (end != start);

    assert(n_dims > 0);

    rewind(f);
    int n_items = 0;
    float dummy;
    while (1 == fscanf(f, "%f", &dummy))
        n_items++;

    n_points = n_items / n_dims;
    assert(n_points % n_dims == 0);

    data = (float*)safe_malloc(n_points * n_dims * sizeof(*data));

    rewind(f);
    for (int i = 0; i < n_points; i++) {
        for (int d = 0; d < n_dims; d++) {
            const int nread = fscanf(f, "%f", &data[IDX(i, d, n_dims)]);
            assert(nread == 1);
        }
    }
}

void save_results( FILE *f )
{
    fprintf(f, "# Centroids:\n#\n");
    for (int j = 0; j < n_clusters; j++) {
        fprintf(f, "# %3d :", j);
        for (int d = 0; d < n_dims; d++)
            fprintf(f, " %f", centroids[IDX(j, d, n_dims)]);
        fprintf(f, "\n");
    }
    fprintf(f, "#\n");
    for (int i = 0; i < n_points; i++) {
        for (int d = 0; d < n_dims; d++)
            fprintf(f, "%f ", data[IDX(i, d, n_dims)]);
        fprintf(f, "%d\n", cluster_of[i]);
    }
}

/* --------------------------------------------------------------------------
 * Main
 * -------------------------------------------------------------------------- */
int main( int argc, char *argv[] )
{
    FILE *inputf, *outputf;

#ifdef MAX_ITER_FIXED
    const int fixed_iters = MAX_ITER_FIXED;
#else
    const int   MAXITER = 100;
    const float TOL     = 1e-5f;
#endif

    /* Block size for the main kernels.
       256 threads/block: good occupancy on most NVIDIA architectures.
       RTX 4060 Ti (sm_89): 1024 max threads/block, 48 warps/SM.
       256 = 8 warps/block → up to 6 blocks per SM concurrently. */
    const int BLKDIM = 256;

    if (argc != 4) {
        fprintf(stderr, "Usage: %s K input_file output_file\n", argv[0]);
        return EXIT_FAILURE;
    }

    srand(123);
    n_clusters = atoi(argv[1]);

    if ((inputf = fopen(argv[2], "r")) == NULL) {
        fprintf(stderr, "FATAL: can not open input file \"%s\"\n", argv[2]);
        return EXIT_FAILURE;
    }
    read_input(inputf);
    fclose(inputf);

    assert(n_clusters < n_points);
    assert(n_clusters * n_dims <= MAX_CENTROIDS_FLOATS); /* constant memory limit */

    if ((outputf = fopen(argv[3], "w")) == NULL) {
        fprintf(stderr, "FATAL: can not create output file \"%s\"\n", argv[3]);
        return EXIT_FAILURE;
    }

    fprintf(outputf, "# Data points: %d\n",  n_points);
    fprintf(outputf, "# Dimensions: %d\n",   n_dims);
    fprintf(outputf, "# Clusters: %d\n",     n_clusters);

    printf("\nInput file....... %s\n", argv[2]);
    printf("Output file...... %s\n",  argv[3]);
    printf("Data points (N).. %d\n",  n_points);
    printf("Dimensions (D)... %d\n",  n_dims);
    printf("Clusters (K)..... %d\n\n", n_clusters);

    /* Allocate host arrays. */
    centroids     = (float*)safe_malloc(n_clusters * n_dims * sizeof(*centroids));
    new_centroids = (float*)safe_malloc(n_clusters * n_dims * sizeof(*new_centroids));
    cluster_of    = (int*)  safe_malloc(n_points            * sizeof(*cluster_of));
    counts        = (int*)  safe_malloc(n_clusters          * sizeof(*counts));

    /* DO NOT parallelize: rand() is not thread-safe. */
    init_centroids();

    /* Allocate device arrays.
       d_data and d_cluster_of persist across iterations (no realloc needed).
       d_new_centroids and d_counts are reset at the start of each iteration. */
    const size_t data_bytes      = (size_t)n_points   * n_dims    * sizeof(float);
    const size_t centroids_bytes = (size_t)n_clusters * n_dims    * sizeof(float);
    const size_t assign_bytes    = (size_t)n_points               * sizeof(int);
    const size_t counts_bytes    = (size_t)n_clusters             * sizeof(int);

    cudaSafeCall( cudaMalloc((void**)&d_data,          data_bytes) );
    cudaSafeCall( cudaMalloc((void**)&d_centroids,     centroids_bytes) );
    cudaSafeCall( cudaMalloc((void**)&d_new_centroids, centroids_bytes) );
    cudaSafeCall( cudaMalloc((void**)&d_counts,        counts_bytes) );
    cudaSafeCall( cudaMalloc((void**)&d_cluster_of,    assign_bytes) );

    /* Copy data to device once (data[] never changes during the algorithm). */
    cudaSafeCall( cudaMemcpy(d_data, data, data_bytes, cudaMemcpyHostToDevice) );

    /* Grid dimensions for the N-point kernels. */
    const int grid_points = (n_points + BLKDIM - 1) / BLKDIM;

    printf("Main loop starts\n\n");

    float  shift = 0.0f;
    int    iter  = 0;

    /* Start timing AFTER data transfer (we time only the algorithm itself,
       not the one-time H2D copy of data). */
    const double tstart = hpc_gettime();

    do {
        /* ---- Step 1: copy current centroids to constant memory ---- */
        /* Constant memory is used by classify_kernel for read-only broadcast.
           d_centroids (global) is kept for normalise_kernel which needs the
           OLD centroids to handle empty clusters. */
        cudaSafeCall( cudaMemcpyToSymbol(d_centroids_const, centroids,
                                         centroids_bytes) );
        /* Also copy to d_centroids (global) for the normalise kernel. */
        cudaSafeCall( cudaMemcpy(d_centroids, centroids, centroids_bytes,
                                  cudaMemcpyHostToDevice) );

        /* ---- Step 2: reset per-iteration accumulators ---- */
        cudaSafeCall( cudaMemset(d_new_centroids, 0, centroids_bytes) );
        cudaSafeCall( cudaMemset(d_counts,        0, counts_bytes) );

        /* ---- Step 3: classify (one thread per point) ---- */
        classify_kernel<<<grid_points, BLKDIM>>>(
            d_data, d_cluster_of, n_points, n_clusters, n_dims);
        cudaCheckError();

        /* ---- Step 4: count points per cluster ---- */
        count_kernel<<<grid_points, BLKDIM>>>(d_cluster_of, d_counts, n_points);
        cudaCheckError();

        /* ---- Step 5: accumulate new centroid positions ---- */
        accumulate_kernel<<<grid_points, BLKDIM>>>(
            d_data, d_cluster_of, d_new_centroids, n_points, n_dims);
        cudaCheckError();

        /* ---- Step 6: normalise (one thread per centroid) ---- */
        /* Grid for K centroids: small (K<<N). */
        const int grid_clusters = (n_clusters + BLKDIM - 1) / BLKDIM;
        normalise_kernel<<<grid_clusters, BLKDIM>>>(
            d_new_centroids, d_centroids, d_counts, n_clusters, n_dims);
        cudaCheckError();

        /* ---- Step 7: copy new centroids back to host ---- */
        /* We need centroids on the host to: (a) check convergence,
           (b) copy to constant memory on the next iteration.
           This D2H copy is small (K*D floats) → negligible latency. */
        cudaSafeCall( cudaMemcpy(new_centroids, d_new_centroids, centroids_bytes,
                                  cudaMemcpyDeviceToHost) );

        /* ---- Step 8: compute max shift (host, K*D ops → negligible) ---- */
        shift = compute_shift(centroids, new_centroids, n_clusters, n_dims);

        /* Update host centroids for next iteration. */
        memcpy(centroids, new_centroids, centroids_bytes);

        printf("Iteration %3d, shift = %f\n", iter, shift);
        iter++;

#ifdef MAX_ITER_FIXED
    } while (iter < fixed_iters);
#else
    } while ( (shift > TOL) && (iter <= MAXITER) );
#endif

    /* Synchronise GPU before stopping the timer. */
    cudaSafeCall( cudaDeviceSynchronize() );
    const double elapsed = hpc_gettime() - tstart;

    /* Copy final cluster assignments back to host for output. */
    cudaSafeCall( cudaMemcpy(cluster_of, d_cluster_of, assign_bytes,
                              cudaMemcpyDeviceToHost) );

    printf("\nMain loop completed\n");
    printf("Elapsed time %.6f\n\n", elapsed);

    save_results(outputf);
    fclose(outputf);

    /* Free device memory. */
    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_new_centroids);
    cudaFree(d_counts);
    cudaFree(d_cluster_of);

    /* Free host memory. */
    free(data);
    free(centroids);
    free(new_centroids);
    free(cluster_of);
    free(counts);

    return EXIT_SUCCESS;
}
