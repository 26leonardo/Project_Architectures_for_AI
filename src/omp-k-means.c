/****************************************************************************
 *
 * omp-k-means.c -- OpenMP parallelization of the K-Means clustering algorithm.
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
 * ## Parallel Pattern Choice: PARTITION (data parallelism) + REDUCE
 *
 * Lloyd's algorithm has two phases per iteration:
 *
 *   1. classify()      -- assign each of the N points to its nearest centroid
 *   2. update_centroids() -- recompute each centroid as the mean of its points
 *
 * ### Why PARTITION?
 * In classify(), each point i is independent of every other point: it reads
 * `data[i*n_dims .. (i+1)*n_dims - 1]` and `centroids[0..K*D-1]`, and writes
 * only `cluster_of[i]` and `counts[j]`. With N >> K, the work is dominated by
 * the N outer iterations, so we split them evenly across threads (PARTITION).
 *
 * ### Why REDUCE?
 * Both `counts[j]` and `new_centroids[j*n_dims + d]` are accumulated over all
 * N points. This is a classic reduction. We use OpenMP's array-reduction
 * (available from OpenMP 4.5 for user-defined arrays via lastprivate/atomic)
 * via thread-private copies + a final serial merge, which avoids the overhead
 * of `#pragma omp atomic` inside the inner loop (K*D atomics per iteration).
 *
 * ### Memory layout and cache behaviour
 * data[]         is stored ROW-MAJOR: data[i*n_dims + d].
 * centroids[]    is stored ROW-MAJOR: centroids[j*n_dims + d].
 * new_centroids[] idem.
 *
 * In classify() the innermost loop iterates over d (dimensions) for a fixed
 * point i → contiguous memory reads → CACHE-FRIENDLY (spatial locality).
 * centroids[] is small (K*D floats, K<<N) and fits in L1/L2 of every thread,
 * so the repeated reads are essentially free.
 *
 * In update_centroids() we accumulate into new_centroids[cluster_of[i]*n_dims + d].
 * Because cluster_of[i] is random, this is a SCATTER with non-contiguous writes.
 * We mitigate this with private copies per thread (see below).
 *
 * ### Fixed-iteration mode
 * Define MAX_ITER_FIXED at compile time (-DMAX_ITER_FIXED=200) to run a
 * predetermined number of iterations instead of waiting for convergence.
 * This makes timing reproducible across datasets (same N => same wall-clock).
 * The original convergence check is preserved and selectable at compile time.
 *
 * Compile with:
 *      gcc -std=c99 -Wall -Wpedantic -fopenmp omp-k-means.c -o omp-k-means
 *
 * Run with:
 *      OMP_NUM_THREADS=4 ./omp-k-means K input_file output_file
 *
 ****************************************************************************/

/* Enable POSIX extensions which are required for making function
   `clock_gettime()` (used in "hpc.h") visible. */
#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif

#include "../utils/hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>     /* memset, memcpy */
#include <assert.h>
#include <omp.h>

/* --------------------------------------------------------------------------
 * Global variables (same as serial version; shared across all threads).
 * -------------------------------------------------------------------------- */

int n_dims;             /* number of dimensions.                */
int n_points;           /* number of data points.               */
int n_clusters;         /* number of clusters.                  */

float *data;            /* [n_points * n_dims]
                           data[IDX(i,d)] is dimension d of point i. */

float *centroids;       /* [n_clusters * n_dims]
                           centroids[IDX(j,d)] is dimension d of centroid j. */

float *new_centroids;   /* [n_clusters * n_dims]
                           Accumulator for the next centroid positions. */

int *counts;            /* [n_clusters]
                           counts[j] = number of points assigned to cluster j. */

int *cluster_of;        /* [n_points]
                           cluster_of[i] = cluster ID of point i. */

/* --------------------------------------------------------------------------
 * Helpers
 * -------------------------------------------------------------------------- */

void *safe_malloc(size_t size)
{
    void *result = malloc(size);
    assert(result != NULL);
    return result;
}

/* --------------------------------------------------------------------------
 * Vector utilities (operate on arrays of n_dims floats).
 * These are called in hot loops. With OpenMP they run inside parallel
 * regions on private/local data, so no synchronisation is needed here.
 * -------------------------------------------------------------------------- */

void vzero( float *p )
{
    for (int d = 0; d < n_dims; d++)
        p[d] = 0.0f;
}

void vadd( float *p1, const float *p2 )
{
    for (int d = 0; d < n_dims; d++)
        p1[d] += p2[d];
}

void vmul( float *p, float v )
{
    for (int d = 0; d < n_dims; d++)
        p[d] *= v;
}

void vcopy( float *p1, const float *p2 )
{
    for (int d = 0; d < n_dims; d++)
        p1[d] = p2[d];
}

/* Squared Euclidean distance between p1 and p2.
   Inner loop over d: contiguous accesses if both are row-major → cache-friendly. */
float sqdist( const float *p1, const float *p2 )
{
    float result = 0.0f;
    for (int d = 0; d < n_dims; d++) {
        const float diff = p1[d] - p2[d];
        result += diff * diff;
    }
    return result;
}

/* --------------------------------------------------------------------------
 * Index helper (same semantics as serial version).
 * -------------------------------------------------------------------------- */
int IDX(int i, int d)
{
    return i * n_dims + d;
}

/* --------------------------------------------------------------------------
 * Random helpers (NOT parallelized: rand() is not thread-safe).
 * -------------------------------------------------------------------------- */
int randab(int a, int b)
{
    return a + rand() % (b - a + 1);
}

/* Centroid initialization (NOT parallelized: rand() is not thread-safe). */
void init_centroids( void )
{
    int select    = n_clusters;
    int remaining = n_points;
    for (int i = 0; (i < n_points) && (select > 0); i++) {
        if ((rand() % remaining) < select) {
            select--;
            vcopy( &centroids[IDX(select, 0)], &data[IDX(i, 0)] );
        }
        remaining--;
    }
}

/* --------------------------------------------------------------------------
 * classify() -- PARALLEL
 *
 * Pattern: PARTITION (embarrassingly parallel over points).
 *
 * Each thread handles a contiguous block of points (static schedule,
 * default chunk = N/nthreads). Static is chosen because every point
 * performs the same amount of work (K distance computations of D
 * operations each) → perfect load balance with static scheduling.
 *
 * `counts` and `cluster_of` have disjoint write ranges per iteration:
 *   - cluster_of[i] is written by exactly one thread (the one owning point i).
 *   - counts[j] can be incremented by multiple threads → RACE CONDITION.
 *
 * Resolution for counts: each thread keeps a PRIVATE copy of counts
 * (local_counts[j]), then they are merged with #pragma omp atomic after
 * the loop. This avoids serialising the inner loop with per-increment atomics.
 *
 * Memory access pattern:
 *   - data[i*n_dims .. i*n_dims+D-1]: sequential read per point → cache hit.
 *   - centroids[j*n_dims .. j*n_dims+D-1]: K*D floats total; small enough
 *     to reside in L1/L2 for all threads (K<<N, D small). Each thread
 *     reads the full centroid array → shared read, no invalidation.
 * -------------------------------------------------------------------------- */
void classify( void )
{
    /* Reset global counts; will be accumulated from private copies below. */
    for (int j = 0; j < n_clusters; j++)
        counts[j] = 0;

    #pragma omp parallel default(none) shared(data, centroids, cluster_of, counts, n_points, n_clusters, n_dims)
    {
        /* Private per-thread accumulator for cluster sizes.
           Avoids a critical section or atomic inside the inner loop. */
        int *local_counts = (int*)safe_malloc(n_clusters * sizeof(int));
        for (int j = 0; j < n_clusters; j++)
            local_counts[j] = 0;

        /* Partition the N points across threads (static = even chunks,
           good load balance since all points do the same work). */
        #pragma omp for schedule(static)
        for (int i = 0; i < n_points; i++) {
            /* Find the nearest centroid for point i.
               Accesses data[i*n_dims .. i*n_dims+D-1] sequentially → good locality. */
            int   nearest = 0;
            float mindist = sqdist( &data[IDX(i, 0)], &centroids[IDX(0, 0)] );

            for (int j = 1; j < n_clusters; j++) {
                const float dist = sqdist( &data[IDX(i, 0)], &centroids[IDX(j, 0)] );
                if (dist < mindist) {
                    mindist = dist;
                    nearest = j;
                }
            }
            cluster_of[i] = nearest;
            local_counts[nearest]++;   /* write to private copy: no race */
        }

        /* Merge private counts into the global array.
           Only K atomic increments per thread (K << N) → negligible overhead. */
        for (int j = 0; j < n_clusters; j++) {
            #pragma omp atomic
            counts[j] += local_counts[j];
        }

        free(local_counts);
    } /* implicit barrier here: all threads done before update_centroids() reads counts */
}

/* --------------------------------------------------------------------------
 * update_centroids() -- PARALLEL
 *
 * Pattern: PARTITION + REDUCE.
 *
 * Phase A: accumulate new centroid positions.
 *   new_centroids[cluster_of[i]] += data[i]  for all i.
 *   This is a SCATTER: cluster_of[i] is not monotone, so writes to
 *   new_centroids can conflict across threads.
 *
 *   Resolution: each thread keeps a PRIVATE copy of new_centroids
 *   (local_nc[j*n_dims + d]) and accumulates locally, then merges.
 *   Cost of merge: K*D atomics per thread (K*D << N*K*D).
 *
 *   Alternative (not used): #pragma omp atomic on each vadd() → K*D
 *   atomics PER POINT, which would serialise the inner loop.
 *
 *   Memory access in the accumulation:
 *   - data[i*n_dims .. i*n_dims+D-1]: sequential → cache-friendly.
 *   - local_nc[cluster_of[i]*n_dims .. +D-1]: scattered write, but
 *     local_nc is private per thread and fits in L1 (K*D floats, K<<N).
 *
 * Phase B: normalise and compute max shift.
 *   Only K*D operations → serial (negligible vs N*K*D work in Phase A).
 *   Only one thread performs it (omp single), others wait at the implicit barrier.
 * -------------------------------------------------------------------------- */
float update_centroids( void )
{
    /* Reset new_centroids accumulator. */
    for (int j = 0; j < n_clusters; j++)
        vzero( &new_centroids[IDX(j, 0)] );

    #pragma omp parallel default(none) shared(data, centroids, new_centroids, cluster_of, counts, n_points, n_clusters, n_dims)
    {
        const int nc_size = n_clusters * n_dims;

        /* Private per-thread accumulator for new centroid positions.
           Size = K*D floats; K and D are small, so this fits in L1/L2. */
        float *local_nc = (float*)safe_malloc(nc_size * sizeof(float));
        memset(local_nc, 0, nc_size * sizeof(float));

        /* Partition: each thread accumulates over its block of points. */
        #pragma omp for schedule(static)
        for (int i = 0; i < n_points; i++) {
            /* data[i*n_dims..] is sequential → spatial locality. */
            vadd( &local_nc[IDX(cluster_of[i], 0)], &data[IDX(i, 0)] );
        }

        /* Merge: K*D atomic additions per thread.
           Because K*D << N, this is a small fraction of total work. */
        for (int j = 0; j < n_clusters; j++) {
            for (int d = 0; d < n_dims; d++) {
                #pragma omp atomic
                new_centroids[IDX(j, d)] += local_nc[IDX(j, d)];
            }
        }

        free(local_nc);
    } /* implicit barrier: merge complete before normalisation */

    /* Phase B: normalise centroids and compute max shift.
       Serial section: only K*D work, negligible vs the N-point accumulation.
       Only the master thread executes this. */
    float maxshift = 0.0f;
    for (int j = 0; j < n_clusters; j++) {
        if (counts[j] == 0) {
            /* Empty cluster: keep the old centroid. */
            vcopy( &new_centroids[IDX(j, 0)], &centroids[IDX(j, 0)] );
        } else {
            vmul( &new_centroids[IDX(j, 0)], 1.0f / counts[j] );
        }
        const float shift = sqdist( &centroids[IDX(j, 0)], &new_centroids[IDX(j, 0)] );
        if (shift > maxshift)
            maxshift = shift;
        vcopy( &centroids[IDX(j, 0)], &new_centroids[IDX(j, 0)] );
    }

    return maxshift;
}

/* --------------------------------------------------------------------------
 * Input/Output functions. DO NOT parallelize them.
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
            const int nread = fscanf(f, "%f", &data[IDX(i, d)]);
            assert(nread == 1);
        }
    }
}

#ifdef MAKE_MOVIE

void save_centroids( int iter )
{
    char buf[1024];
    snprintf(buf, sizeof(buf), "temp/centroids_%03u.txt", (unsigned)iter);
    FILE *f = fopen(buf, "w"); assert(f != NULL);
    if (f == NULL) {
        fprintf(stderr, "FATAL: can not open file \"%s\" for writing\n", buf);
        exit(EXIT_FAILURE);
    }
    for (int j = 0; j < n_clusters; j++) {
        for (int d = 0; d < n_dims; d++)
            fprintf(f, "%f ", centroids[IDX(j, d)]);
        fprintf(f, "\n");
    }
    fclose(f);
}

void save_clusters( int iter )
{
    char buf[1024];
    snprintf(buf, sizeof(buf), "temp/out_%03u.txt", (unsigned)iter);
    FILE *f = fopen(buf, "w");
    if (f == NULL) {
        fprintf(stderr, "FATAL: can not open file \"%s\" for writing\n", buf);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n_points; i++) {
        for (int d = 0; d < n_dims; d++)
            fprintf(f, "%f ", data[IDX(i, d)]);
        fprintf(f, "%d\n", cluster_of[i]);
    }
    fclose(f);
}

#endif

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

/* --------------------------------------------------------------------------
 * Main
 * -------------------------------------------------------------------------- */
int main( int argc, char *argv[] )
{
    FILE *inputf, *outputf;

    /* MAXITER: hard upper bound on iterations (original convergence logic).
       If MAX_ITER_FIXED is defined at compile time (e.g. -DMAX_ITER_FIXED=200),
       we run exactly that many iterations regardless of convergence.
       This makes timing reproducible: same N always means same number of
       iterations, so wall-clock time is not contaminated by dataset variance. */
#ifdef MAX_ITER_FIXED
    const int fixed_iters = MAX_ITER_FIXED;
    /* Suppress "unused variable" warnings: MAXITER and TOL are only used
       in the convergence-check branch, which is compiled out here. */
    (void)0;
#else
    const int    MAXITER = 100;
    const float  TOL     = 1e-5f;  /* squared tolerance */
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
    printf("Output file...... %s\n",  argv[3]);
    printf("Data points (N).. %d\n",  n_points);
    printf("Dimensions (D)... %d\n",  n_dims);
    printf("Clusters (K)..... %d\n",  n_clusters);
    printf("Threads.......... %d\n\n", omp_get_max_threads());

    centroids     = (float*)safe_malloc(n_clusters * n_dims * sizeof(*centroids));
    new_centroids = (float*)safe_malloc(n_clusters * n_dims * sizeof(*new_centroids));
    cluster_of    = (int*)  safe_malloc(n_points            * sizeof(*cluster_of));
    counts        = (int*)  safe_malloc(n_clusters          * sizeof(*counts));

    /* DO NOT parallelize init_centroids: rand() is not thread-safe. */
    init_centroids();

    printf("Main loop starts\n\n");

    float  shift = 0.0f;
    int    iter  = 0;
    const double tstart = hpc_gettime();

    do {
        classify();

#ifdef MAKE_MOVIE
        save_centroids(iter);
        save_clusters(iter);
#endif

        shift = update_centroids();
        printf("Iteration %3d, shift = %f\n", iter, shift);
        iter++;

#ifdef MAX_ITER_FIXED
        /* Fixed-iteration mode: ignore convergence, run exactly fixed_iters. */
    } while (iter < fixed_iters);
#else
        /* Original convergence check. */
    } while ( (shift > TOL) && (iter <= MAXITER) );
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
