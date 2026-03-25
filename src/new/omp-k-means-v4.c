/****************************************************************************
 *
 * omp-k-means-v4.c -- Optimized OpenMP K-Means for use with:
 *   OMP_PROC_BIND=close OMP_PLACES=cores ./omp-k-means-v4 K in out
 *
 * Target: compiled with -O3 -fopenmp
 *
 * Optimizations over v1/v2/v3:
 *
 * 1. FUSED CLASSIFY + ACCUMULATE (main speedup)
 *    v1-v3 do two separate passes over data[]:
 *      Pass 1 (classify):  read data[i] to find nearest centroid
 *      Pass 2 (update):    read data[i] AGAIN to accumulate into new_centroids
 *    v4 does one pass:
 *      read data[i] once, find nearest centroid AND accumulate immediately.
 *    With -O3 and N=1M, data[] does not fit in L3 (160MB > 96MB), so every
 *    read is a DRAM access. Fusing halves the DRAM traffic per iteration.
 *    Expected gain: ~1.4-1.8x on N=1M (bandwidth-bound regime).
 *    On N=500K (fits in L3) the gain is smaller (~1.1x) since L3 reads are
 *    cheaper, but still meaningful.
 *
 * 2. PRE-ALLOCATED THREAD-PRIVATE BUFFERS
 *    local_counts and local_nc are allocated once per thread at the start
 *    of the parallel region and freed at the end. This removes 2*ITERS*P
 *    malloc/free calls from the hot loop. With -O3 malloc/free are fast
 *    but this also improves data locality: the buffers remain in the same
 *    physical cache lines across iterations (no TLB churn).
 *
 * 3. PERSISTENT PARALLEL REGION
 *    One omp parallel for all iterations. Thread-team creation cost
 *    is O(1) instead of O(2*ITERS). Already present in the last version
 *    but repeated here for completeness.
 *
 * 4. INLINE SIMD ACCUMULATION
 *    The D-loop inside the fused pass is decorated with #pragma omp simd.
 *    With -O3 gcc will vectorize it with AVX2 (D=40 = 5x8-wide float ops).
 *    This was not possible in v1/v3 because the accumulation was a function
 *    call (vadd) with a non-compile-time-known stride.
 *
 * Runtime environment (set BEFORE running):
 *   export OMP_PROC_BIND=close
 *   export OMP_PLACES=cores
 *   export OMP_NUM_THREADS=14     # optimal with -O3 (16 regresses due to HT)
 *
 * OMP_PROC_BIND=close:
 *   Pins thread 0 to core 0, thread 1 to core 1, etc. (closest available).
 *   Without this the OS scheduler may migrate threads mid-run, causing
 *   cache misses as a thread's working set is in the cache of a different core.
 *
 * OMP_PLACES=cores:
 *   Tells OpenMP that each "place" is one physical core (not a hardware thread).
 *   With close binding: thread k -> physical core k (no SMT sharing unless
 *   OMP_NUM_THREADS > 8). Prevents two threads sharing the same L1/L2.
 *   Combined with OMP_NUM_THREADS=14, you get 8 physical cores fully used
 *   plus 6 logical threads on the remaining SMT slots — the measured optimum.
 *
 ****************************************************************************/

#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif

#include "../../utils/hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <omp.h>

int n_dims;
int n_points;
int n_clusters;

float *data;
float *centroids;
float *new_centroids;
int   *counts;
int   *cluster_of;

void *safe_malloc(size_t size)
{
    void *r = malloc(size);
    assert(r != NULL);
    return r;
}

/* All vector utilities are kept for compatibility with init_centroids / I/O.
   In the hot loop (fused pass) they are NOT called: the D-loop is inlined
   so the compiler can vectorize across iterations and keep data in registers. */

void vzero(float *p) { memset(p, 0, n_dims * sizeof(float)); }

void vadd(float *p1, const float *p2)
{
    #pragma omp simd
    for (int d = 0; d < n_dims; d++) p1[d] += p2[d];
}

void vmul(float *p, float v)
{
    #pragma omp simd
    for (int d = 0; d < n_dims; d++) p[d] *= v;
}

void vcopy(float *p1, const float *p2)
{
    #pragma omp simd
    for (int d = 0; d < n_dims; d++) p1[d] = p2[d];
}

float sqdist(const float *p1, const float *p2)
{
    float r = 0.0f;
    #pragma omp simd reduction(+:r)
    for (int d = 0; d < n_dims; d++) {
        const float diff = p1[d] - p2[d];
        r += diff * diff;
    }
    return r;
}

int IDX(int i, int d) { return i * n_dims + d; }

void init_centroids(void)
{
    int select = n_clusters, remaining = n_points;
    for (int i = 0; (i < n_points) && (select > 0); i++) {
        if ((rand() % remaining) < select) {
            select--;
            vcopy(&centroids[IDX(select, 0)], &data[IDX(i, 0)]);
        }
        remaining--;
    }
}

void read_input(FILE *f)
{
    const size_t BUFLEN = 1024;
    char buffer[BUFLEN];
    char *ic = fgets(buffer, BUFLEN, f); (void)ic;
    n_dims = -1;
    char *start, *end = buffer;
    do { start = end; strtof(start, &end); n_dims++; } while (end != start);
    assert(n_dims > 0);

    rewind(f);
    int n_items = 0; float dummy;
    while (1 == fscanf(f, "%f", &dummy)) n_items++;
    assert(n_items % n_dims == 0);
    n_points = n_items / n_dims;

    data = (float*)safe_malloc(n_points * n_dims * sizeof(*data));
    rewind(f);
    for (int i = 0; i < n_points; i++)
        for (int d = 0; d < n_dims; d++) {
            const int nr = fscanf(f, "%f", &data[IDX(i, d)]);
            assert(nr == 1);
        }
}

void save_results(FILE *f)
{
    fprintf(f, "# Centroids:\n#\n");
    for (int j = 0; j < n_clusters; j++) {
        fprintf(f, "# %3d :", j);
        for (int d = 0; d < n_dims; d++)
            fprintf(f, " %f", centroids[IDX(j, d)]);
        fprintf(f, "\n");
    }
    fprintf(f, "#\n");
    for (int i = 0; i < n_points; i++) {
        for (int d = 0; d < n_dims; d++)
            fprintf(f, "%f ", data[IDX(i, d)]);
        fprintf(f, "%d\n", cluster_of[i]);
    }
}

int main(int argc, char *argv[])
{
    FILE *inputf, *outputf;

#ifdef MAX_ITER_FIXED
    const int fixed_iters = MAX_ITER_FIXED;
#else
    const int fixed_iters = 150;
#endif

    if (argc != 4) {
        fprintf(stderr, "Usage: %s K input_file output_file\n", argv[0]);
        return EXIT_FAILURE;
    }

    srand(123);
    n_clusters = atoi(argv[1]);

    if ((inputf = fopen(argv[2], "r")) == NULL) {
        fprintf(stderr, "FATAL: cannot open \"%s\"\n", argv[2]);
        return EXIT_FAILURE;
    }
    read_input(inputf);
    fclose(inputf);
    assert(n_clusters < n_points);

    if ((outputf = fopen(argv[3], "w")) == NULL) {
        fprintf(stderr, "FATAL: cannot create \"%s\"\n", argv[3]);
        return EXIT_FAILURE;
    }

    fprintf(outputf, "# Data points: %d\n", n_points);
    fprintf(outputf, "# Dimensions: %d\n",  n_dims);
    fprintf(outputf, "# Clusters: %d\n",    n_clusters);

    printf("\nInput file....... %s\n",  argv[2]);
    printf("Output file...... %s\n",   argv[3]);
    printf("Data points (N).. %d\n",   n_points);
    printf("Dimensions (D)... %d\n",   n_dims);
    printf("Clusters (K)..... %d\n\n", n_clusters);
    printf("Threads.......... %d\n",   omp_get_max_threads());
    printf("Proc bind........ %s\n\n", getenv("OMP_PROC_BIND") ? getenv("OMP_PROC_BIND") : "unset");

    centroids     = (float*)safe_malloc(n_clusters * n_dims * sizeof(*centroids));
    new_centroids = (float*)safe_malloc(n_clusters * n_dims * sizeof(*new_centroids));
    cluster_of    = (int*)  safe_malloc(n_points            * sizeof(*cluster_of));
    counts        = (int*)  safe_malloc(n_clusters          * sizeof(*counts));

    init_centroids();

    printf("Main loop starts\n\n");

    float  shift = 0.0f; /* written by omp single, read below */
    const double tstart = hpc_gettime();

    /* -----------------------------------------------------------------------
     * PERSISTENT PARALLEL REGION
     *
     * One team, created once. Thread-private buffers (local_counts,
     * local_nc) are allocated here and reused for all iterations.
     * This avoids 2 × fixed_iters × P heap operations inside the hot loop.
     *
     * The buffers stay hot in each core's L2 across iterations:
     *   local_nc size  = K * D * 4B = 8 * 40 * 4 = 1280 B  → always in L1
     *   local_counts   = K * 4B     = 32 B                  → always in L1
     *
     * Synchronisation per iteration:
     *   [omp for implicit barrier] after the fused loop
     *   [omp single + implicit barrier] for global reset
     *   [explicit omp barrier] after atomic merge
     *   [omp single + implicit barrier] for normalise (last sync of iter)
     * ----------------------------------------------------------------------- */
    #pragma omp parallel default(none) \
        shared(data, centroids, new_centroids, cluster_of, counts, \
               n_points, n_clusters, n_dims, fixed_iters, shift)
    {
        /* Allocate thread-private buffers once for all iterations. */
        const int nc_size = n_clusters * n_dims;
        int   *local_counts = (int*)  safe_malloc(n_clusters * sizeof(int));
        float *local_nc     = (float*)safe_malloc(nc_size    * sizeof(float));

        for (int iter = 0; iter < fixed_iters; iter++) {

            /* Reset private accumulators (tiny: K=8, K*D=320 floats). */
            memset(local_counts, 0, n_clusters * sizeof(int));
            memset(local_nc,     0, nc_size    * sizeof(float));

            /* -----------------------------------------------------------------
             * FUSED CLASSIFY + ACCUMULATE
             *
             * Each thread processes a contiguous block of N/P points.
             * For each point i:
             *   a) find nearest centroid (K * D multiply-adds via sqdist)
             *   b) write cluster_of[i]                  ← no race (disjoint)
             *   c) increment local_counts[nearest]      ← private, no race
             *   d) add data[i] to local_nc[nearest]     ← private, no race
             *
             * data[i*D .. i*D+D-1] is read ONCE and stays in L1/registers
             * for steps (a) and (d). In v1-v3 it was read twice: once in
             * classify() and once in update_centroids().
             *
             * With -O3 and N=1M (data > L3), halving DRAM reads directly
             * halves bandwidth pressure → expected ~1.5× speedup on N=1M.
             * With N=500K (data ≈ L3) the gain is smaller but still real.
             *
             * The inner D-loop for accumulation is decorated with simd so
             * the compiler can issue AVX2 stores into local_nc.
             * ----------------------------------------------------------------- */
            #pragma omp for schedule(static)
            for (int i = 0; i < n_points; i++) {

                /* --- step (a): find nearest centroid --- */
                const float *pi = &data[IDX(i, 0)];   /* pointer to point i */
                int   nearest = 0;
                float mindist = sqdist(pi, &centroids[IDX(0, 0)]);

                for (int j = 1; j < n_clusters; j++) {
                    const float dist = sqdist(pi, &centroids[IDX(j, 0)]);
                    if (dist < mindist) { mindist = dist; nearest = j; }
                }

                /* --- step (b): record assignment --- */
                cluster_of[i] = nearest;

                /* --- step (c): count --- */
                local_counts[nearest]++;

                /* --- step (d): accumulate into local_nc ---
                 * pi is already in L1 from step (a). No second DRAM read.
                 * #pragma omp simd lets gcc issue 8-wide AVX2 FP adds.
                 * local_nc[nearest*D .. nearest*D+D-1]: K*D = 320 floats
                 * → always fits in L1 (32 KB). No cache miss here. */
                float *nc_j = &local_nc[IDX(nearest, 0)];
                #pragma omp simd
                for (int d = 0; d < n_dims; d++)
                    nc_j[d] += pi[d];
            }
            /* Implicit barrier: all threads done with the fused loop. */

            /* Reset global accumulators. One thread; others wait at the
               implicit barrier that follows omp single. */
            #pragma omp single
            {
                memset(counts,        0, n_clusters * sizeof(int));
                memset(new_centroids, 0, nc_size    * sizeof(float));
            }

            /* Merge private → global.
             * K = 8 and K*D = 320: negligible vs the N-point fused loop.
             * Atomics here are correct and fast (K*D = 320 per thread). */
            for (int j = 0; j < n_clusters; j++) {
                #pragma omp atomic
                counts[j] += local_counts[j];

                for (int d = 0; d < n_dims; d++) {
                    #pragma omp atomic
                    new_centroids[IDX(j, d)] += local_nc[IDX(j, d)];
                }
            }

            /* Explicit barrier: all atomic merges complete before
               normalisation reads new_centroids[] and counts[]. */
            #pragma omp barrier

            /* Normalise + compute shift. O(K*D) = O(320) ops: serial.
               Implicit barrier after omp single: centroids[] is fully
               updated and visible to all threads before the next iteration's
               fused loop reads it. */
            #pragma omp single
            {
                float maxshift = 0.0f;
                for (int j = 0; j < n_clusters; j++) {
                    if (counts[j] == 0) {
                        vcopy(&new_centroids[IDX(j, 0)], &centroids[IDX(j, 0)]);
                    } else {
                        vmul(&new_centroids[IDX(j, 0)], 1.0f / counts[j]);
                    }
                    const float s = sqdist(&centroids[IDX(j, 0)],
                                           &new_centroids[IDX(j, 0)]);
                    if (s > maxshift) maxshift = s;
                    vcopy(&centroids[IDX(j, 0)], &new_centroids[IDX(j, 0)]);
                }
                shift = maxshift;
            }
            /* All threads see updated centroids[] here. */
        }

        /* Free thread-private buffers (runs once per thread). */
        free(local_counts);
        free(local_nc);
    }
    (void)shift; /* suppress unused-but-set: shift holds last maxshift */
    /* All threads join. */

    const double elapsed = hpc_gettime() - tstart;

    printf("\nMain loop completed\n");
    printf("Elapsed time %.6f\n\n", elapsed);

    save_results(outputf);
    fclose(outputf);

    free(data);
    free(centroids);
    free(new_centroids);
    free(cluster_of);
    free(counts);

    return EXIT_SUCCESS;
}
