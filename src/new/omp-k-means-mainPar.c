/****************************************************************************
 *
 * omp-k-means.c -- OpenMP parallelization of the K-Means clustering algorithm.
 *
 * Based on k-means.c by Moreno Marzolla
 * <https://unibo.it/sitoweb/moreno.marzolla/>
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
int *counts;
int *cluster_of;

void *safe_malloc(size_t size)
{
    void *result = malloc(size);
    assert(result != NULL);
    return result;
}

void vzero( float *p )
{
    memset(p, 0, n_dims * sizeof(float));
}

void vadd( float *p1, const float *p2 )
{
    #pragma omp simd
    for (int d=0; d<n_dims; d++)
        p1[d] += p2[d];
}

void vmul( float *p, float v )
{
    #pragma omp simd
    for (int d=0; d<n_dims; d++)
        p[d] *= v;
}

void vcopy( float *p1, const float *p2 )
{
    #pragma omp simd
    for (int d=0; d<n_dims; d++)
        p1[d] = p2[d];
}

float sqdist(const float *p1, const float *p2)
{
    float result = 0.0f;
    #pragma omp simd reduction(+:result)
    for (int d = 0; d < n_dims; d++) {
        const float diff = p1[d] - p2[d];
        result += diff * diff;
    }
    return result;
}

int IDX(int i, int d)
{
    return i*n_dims + d;
}

int randab(int a, int b)
{
    return a + rand() % (b-a+1);
}

void init_centroids( void )
{
    int select = n_clusters;
    int remaining = n_points;
    for (int i=0; (i < n_points) && (select > 0); i++) {
        if ((rand() % remaining) < select) {
            select--;
            vcopy( &centroids[IDX(select,0)], &data[IDX(i,0)] );
        }
        remaining--;
    }
}

/* --------------------------------------------------------------------------
 * classify() -- designed to be called from INSIDE an active omp parallel region.
 *
 * Since there is no `#pragma omp parallel` here, NO new thread team is created.
 * The calling threads (already spawned in main) execute this function directly.
 *
 * Synchronization structure:
 *
 *  [omp single]      Reset counts[] → only one thread, all others wait.
 *                    Implicit barrier at end of single ensures the zeroed
 *                    array is visible to all threads before the loop starts.
 *
 *  [omp for]         Distribute N points across the team.
 *                    Implicit barrier at end ensures all threads finish their
 *                    chunk before the atomic merge begins.
 *
 *  [atomic merge]    K atomic adds per thread (K << N → negligible).
 *
 *  [omp barrier]     EXPLICIT barrier after the merge: without it, a fast thread
 *                    could return from classify() and enter update_centroids()
 *                    reading counts[] before slower threads have finished merging.
 *
 * Memory access:
 *   - data[i*D .. i*D+D-1]:       sequential per point → cache-friendly.
 *   - centroids[j*D .. j*D+D-1]:  K*D = 8*50 = 1600 bytes → fits in L1.
 *     Every thread reads all centroids → shared read, no invalidation.
 * -------------------------------------------------------------------------- */
void classify(void)
{
    /* Reset counts: only one thread performs this, all others wait at the
       implicit barrier that follows omp single before proceeding. */
    #pragma omp single
    memset(counts, 0, n_clusters * sizeof(int));

    /* Each thread in the team gets its own local_counts on the heap.
       No sharing, no race: each thread calls safe_malloc independently. */
    int *local_counts = (int*)safe_malloc(n_clusters * sizeof(int));
    memset(local_counts, 0, n_clusters * sizeof(int));

    /* Partition N points across the team (static: equal work per point → perfect balance). */
    #pragma omp for schedule(static)
    for (int i = 0; i < n_points; i++) {
        int nearest = 0;
        float mindist = sqdist(&data[IDX(i, 0)], &centroids[IDX(0, 0)]);
        for (int j = 1; j < n_clusters; j++) {
            const float dist = sqdist(&data[IDX(i, 0)], &centroids[IDX(j, 0)]);
            if (dist < mindist) { mindist = dist; nearest = j; }
        }
        cluster_of[i] = nearest;
        local_counts[nearest]++;
    }
    /* Implicit barrier at end of omp for. */

    /* Merge: K atomics per thread (K=8 → fast). */
    for (int j = 0; j < n_clusters; j++) {
        #pragma omp atomic
        counts[j] += local_counts[j];
    }
    free(local_counts);

    /* Explicit barrier: all atomic merges must complete before update_centroids()
       reads counts[]. The implicit barrier from omp for is NOT enough here
       because the atomic merges happen AFTER that barrier. */
    #pragma omp barrier
}

/* --------------------------------------------------------------------------
 * update_centroids() -- designed to be called from INSIDE an active omp parallel region.
 *
 * Synchronization structure:
 *
 *  [omp single]      Zero new_centroids[]. Implicit barrier after.
 *
 *  [omp for]         Accumulate N points into thread-private local_nc.
 *                    Implicit barrier after.
 *
 *  [atomic merge]    K*D atomics per thread (K*D = 400 → small fraction of N=2M work).
 *
 *  [omp barrier]     All merges complete before normalisation reads new_centroids[].
 *
 *  [omp single]      Normalise + compute shift + copy centroids.
 *                    Implicit barrier after: ALL threads see updated centroids[]
 *                    before the next iteration's classify() reads them.
 *                    This barrier also eliminates the need for an additional
 *                    barrier between update_centroids() and the next classify().
 *
 * The float *p_maxshift output pointer allows the caller (main) to read the
 * convergence metric without breaking the persistent parallel region.
 * -------------------------------------------------------------------------- */
void update_centroids(float *p_maxshift)
{
    /* Zero new_centroids: one thread, implicit barrier after. */
    #pragma omp single
    {
        for (int j = 0; j < n_clusters; j++)
            vzero(&new_centroids[IDX(j, 0)]);
    }

    /* Each thread gets its own private accumulator (K*D = 1600 bytes → fits in L1). */
    const int nc_size = n_clusters * n_dims;
    float *local_nc = (float*)safe_malloc(nc_size * sizeof(float));
    memset(local_nc, 0, nc_size * sizeof(float));

    /* Scatter-accumulate: each thread accumulates over its block of points.
       Writes to local_nc[cluster_of[i]*D .. +D]: scattered destination but
       private array → no race, fits in L1 (K*D = 1600 bytes for K=8, D=50). */
    #pragma omp for schedule(static)
    for (int i = 0; i < n_points; i++) {
        vadd(&local_nc[IDX(cluster_of[i], 0)], &data[IDX(i, 0)]);
    }
    /* Implicit barrier after omp for. */

    /* Merge: K*D = 400 atomics per thread. */
    for (int j = 0; j < n_clusters; j++) {
        for (int d = 0; d < n_dims; d++) {
            #pragma omp atomic
            new_centroids[IDX(j, d)] += local_nc[IDX(j, d)];
        }
    }
    free(local_nc);

    /* Explicit barrier: all merges must be complete before normalisation. */
    #pragma omp barrier

    /* Normalise + shift: O(K*D) serial work, negligible vs O(N*K*D).
       Implicit barrier after single: centroids[] is fully updated and visible
       to all threads before classify() in the next iteration reads it.
       This is the last synchronization point of each iteration. */
    #pragma omp single
    {
        float maxshift = 0.0f;
        for (int j = 0; j < n_clusters; j++) {
            if (counts[j] == 0) {
                vcopy(&new_centroids[IDX(j, 0)], &centroids[IDX(j, 0)]);
            } else {
                vmul(&new_centroids[IDX(j, 0)], 1.0f / counts[j]);
            }
            const float s = sqdist(&centroids[IDX(j, 0)], &new_centroids[IDX(j, 0)]);
            if (s > maxshift) maxshift = s;
            vcopy(&centroids[IDX(j, 0)], &new_centroids[IDX(j, 0)]);
        }
        *p_maxshift = maxshift;
    }
}

/******************************************************************************
 ** Input/output functions. DO NOT parallelize them.
 ******************************************************************************/

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

    assert(n_items % n_dims == 0);
    n_points = n_items / n_dims;

    data = (float*)safe_malloc(n_points * n_dims * sizeof(*data));

    rewind(f);
    for (int i=0; i<n_points; i++) {
        for (int d=0; d<n_dims; d++) {
            const int nread = fscanf(f, "%f", &data[IDX(i, d)]);
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

/******************************************************************************
 ** Main program.
 ******************************************************************************/
int main( int argc, char *argv[] )
{
    FILE *inputf, *outputf;

#ifdef MAX_ITER_FIXED
    const int fixed_iters = MAX_ITER_FIXED;
#else
    const int   MAXITER = 100;
    const float TOL     = 1e-5f;
#endif

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

    if ((outputf = fopen(argv[3], "w")) == NULL) {
        fprintf(stderr, "FATAL: can not create output file \"%s\"\n", argv[3]);
        return EXIT_FAILURE;
    }

    fprintf(outputf, "# Data points: %d\n",  n_points);
    fprintf(outputf, "# Dimensions: %d\n",   n_dims);
    fprintf(outputf, "# Clusters: %d\n",     n_clusters);

    printf("\nInput file....... %s\n", argv[2]);
    printf("Output file...... %s\n",   argv[3]);
    printf("Data points (N).. %d\n",   n_points);
    printf("Dimensions (D)... %d\n",   n_dims);
    printf("Clusters (K)..... %d\n\n", n_clusters);
    printf("Threads.......... %d\n\n", omp_get_max_threads());

    centroids     = (float*)safe_malloc(n_clusters * n_dims * sizeof(*centroids));
    new_centroids = (float*)safe_malloc(n_clusters * n_dims * sizeof(*new_centroids));
    cluster_of    = (int*)  safe_malloc(n_points            * sizeof(*cluster_of));
    counts        = (int*)  safe_malloc(n_clusters          * sizeof(*counts));

    init_centroids();

    printf("Main loop starts\n\n");

    /* shift is written by update_centroids() via the pointer passed to it.
       It must be declared here (outside the parallel region) so it can be
       read after the region ends (for printing / convergence check). */
    float shift = 0.0f;

    const double tstart = hpc_gettime();

#ifdef MAX_ITER_FIXED
    /* -----------------------------------------------------------------------
     * PERSISTENT PARALLEL REGION — fixed-iteration mode.
     *
     * The thread team is created ONCE here and lives for all iterations.
     * Cost: O(1) team creation instead of O(2 × fixed_iters).
     * On 8 cores this saves ~100 × (create + destroy) overheads.
     *
     * All threads execute every iteration of the for loop. Work is
     * distributed inside classify() and update_centroids() via omp for.
     * Synchronisation between classify and update_centroids is handled
     * by the explicit omp barrier at the end of classify() and by the
     * implicit barrier at the end of update_centroids's final omp single.
     *
     * Globals (data, centroids, etc.) are automatically shared even
     * with default(none). Only main-local variables need to be listed:
     *   - fixed_iters: read-only, shared (all threads read the same value).
     *   - shift: written by one thread in omp single, read after the region.
     * ----------------------------------------------------------------------- */
    #pragma omp parallel default(none) shared(fixed_iters, shift)
    {
        for (int iter = 0; iter < fixed_iters; iter++) {
            classify();
            update_centroids(&shift);
        }
    }
    /* All threads join here. shift holds the last iteration's max shift. */

#else
    /* -----------------------------------------------------------------------
     * PERSISTENT PARALLEL REGION — convergence mode.
     *
     * The while loop runs until convergence. Loop-control variables
     * (keep_going, iter) are shared and updated by a single thread
     * inside an omp single block, whose implicit barrier ensures all
     * threads see the updated values before checking the while condition.
     *
     * MAXITER and TOL are main-local consts: firstprivate gives each thread
     * its own copy (same value), avoiding any false-sharing on the stack.
     * ----------------------------------------------------------------------- */
    int keep_going = 1;
    int iter       = 0;
    #pragma omp parallel default(none) \
        shared(shift, keep_going, iter) \
        firstprivate(MAXITER, TOL)
    {
        while (keep_going) {
            classify();
            update_centroids(&shift);

            /* Only one thread checks convergence and updates the shared flags.
               The implicit barrier after omp single ensures ALL threads see the
               updated keep_going before re-evaluating the while condition. */
            #pragma omp single
            {
                printf("Iteration %3d, shift = %f\n", iter, shift);
                iter++;
                keep_going = (shift > TOL) && (iter <= MAXITER);
            }
        }
    }
#endif

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