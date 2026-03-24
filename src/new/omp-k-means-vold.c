/****************************************************************************
 *
 * omp-k-means-opt.c -- OpenMP K-Means: optimised for large D (200-1000)
 *                      and large N (50 - 2,000,000).
 *
 * Based on k-means.c by Moreno Marzolla
 * <https://unibo.it/sitoweb/moreno.marzolla/>
 * Parallelization by: Leonardo Billi 
 *
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
 * ## What changes relative to omp-k-means.c (the standard version)
 *
 * ### Problem with large D in the standard version
 *
 * In the standard version, data[] is row-major: data[i*D + d].
 * In classify(), for each point i and each centroid j, sqdist reads:
 *
 *   data   at offsets: i*D+0, i*D+1, ..., i*D+D-1    (stride-1 ✅)
 *   centroids at offsets: j*D+0, j*D+1, ..., j*D+D-1  (stride-1 ✅)
 *
 * This is fine for a single point. BUT consider what happens with
 * MULTIPLE THREADS processing points i, i+1, i+2, ... simultaneously.
 * At dimension d=0, threads access:
 *
 *   thread 0: data[0*D + 0]
 *   thread 1: data[1*D + 0]
 *   ...
 *   thread P-1: data[(P-1)*D + 0]
 *
 * These are D floats apart in memory. With D=256, that is 1024 bytes
 * between consecutive thread accesses → CACHE LINE POLLUTION: each thread
 * loads its own cache line (64 bytes = 16 floats) but only uses 1 element
 * at position d=0. The other 15 elements on that cache line are wasted.
 *
 * ### Solution: COLUMN-MAJOR (TRANSPOSED) layout for data[]
 *
 * We store data in column-major order: data_T[d*N + i].
 * Now at dimension d=0, threads access:
 *
 *   thread 0: data_T[0*N + 0]       ← address: base + 0
 *   thread 1: data_T[0*N + 1]       ← address: base + 4 bytes
 *   ...
 *   thread P-1: data_T[0*N + P-1]   ← address: base + (P-1)*4 bytes
 *
 * These are STRIDE-1 → a single cache line serves 16 consecutive threads.
 * This is ROW-WISE access in the transposed array (iterating over i for
 * fixed d), which is COLUMN-WISE in the original point layout.
 *
 * Trade-off:
 *   - A single thread accessing its own point data_T[d*N + i] for d=0..D-1
 *     now has STRIDE-N (bad for serial inner loop).
 *   - But the PARALLELISM across threads is now stride-1 (excellent).
 *
 * For large D (>> cache line size / sizeof(float) = 16), the thread-parallel
 * benefit DOMINATES: we gain one cache miss per cache line (16 threads share
 * one line) instead of one cache miss per thread.
 *
 * Break-even point: when D*sizeof(float) > cache line size (64 bytes),
 * i.e., D > 16. For D=16 (standard case) the two layouts are equivalent.
 * For D=256 (16x larger), column-major is ~16x more cache-efficient for
 * the thread-parallel access pattern.
 *
 * ### What stays the same
 * - Parallel structure: PARTITION over points in classify(),
 *   PARTITION + REDUCE in update_centroids()
 * - Thread-private local accumulators for counts and new_centroids
 * - #pragma omp simd on inner loops
 * - Fixed-iteration mode via MAX_ITER_FIXED
 *
 * ### Memory access summary for this version
 *
 *   Array        Layout        Inner loop in classify     Access type
 *   data_T[]     Col-major     for i (d fixed)            ROW-WISE on data_T ✅
 *                              for d (i fixed, sqdist)    COL-WISE on data_T ⚠️
 *   centroids[]  Row-major     for d (j fixed)            ROW-WISE ✅
 *   local_nc[]   Row-major     for d (j fixed, vadd_T)    ROW-WISE ✅
 *
 * The COL-WISE access in sqdist (iterating d for fixed i) is unavoidable with
 * column-major layout and a single thread per point. It is the accepted
 * trade-off: serial access within a point is strided, but parallel access
 * across all threads at a fixed d is stride-1.
 *
 * Compile with:
 *      gcc -std=c99 -Wall -Wpedantic -fopenmp omp-k-means-opt.c -o omp-k-means-opt -lm
 *
 * Run with:
 *      OMP_NUM_THREADS=8 ./omp-k-means-opt K input_file output_file
 *
 ****************************************************************************/

/* Enable POSIX extensions which are required for making function
   `clock_gettime()` (used in "hpc.h") visible. The following
   `#define` must come at the very beginning, before including
   anything else.
*/
#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif

#include "../../utils/hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <omp.h>

/* --------------------------------------------------------------------------
 * Global variables.
 * data_T is the TRANSPOSED (column-major) layout of the input data.
 * All other arrays keep their original row-major layout.
 * -------------------------------------------------------------------------- */

int n_dims;             /* number of dimensions.        D        */

int n_points;           /* number of data points.       N        */

int n_clusters;         /* number of clusters.          K        */

float *data;            /* [array of length (n_points * n_dims)]
                           Original row-major layout. Used only for I/O.
                           data[i*n_dims + d] = dimension d of point i. */

float *data_T;          /* [array of length (n_dims * n_points)]
                           COLUMN-MAJOR (transposed) layout.
                           data_T[d*n_points + i] = dimension d of point i.
                           Used in the hot loops of classify() and
                           update_centroids() for better cache behaviour
                           when D is large and P threads run in parallel. */

float *centroids;       /* [array of length (n_clusters * n_dims)]
                           `&centroids[j*n_dims]` points to the
                           beginning of the j-th centroid, which is an
                           array of `n_dims` floating point
                           numbers.                             */

float *new_centroids;   /* [array of length (n_clusters * n_dims)] */

int *counts;            /* [array of length n_clusters] `counts[j]`
                           is the number of points that belong to
                           cluster j.                           */

int *cluster_of;        /* [array of length n_points] `clusters_of[i]`
                           is the ID of the cluster assigned to the
                           i-th data point; cluster IDs are integer in
                           0..(n_clusters-1).                   */


/* A safe version of `malloc()` that aborts if memory allocation
   fails. */
void *safe_malloc(size_t size)
{
    void *result = malloc(size);
    assert(result != NULL);
    return result;
}

/******************************************************************************
 **
 ** Utility functions that operate on arrays of `n_dims` elements.
 **
 ******************************************************************************/

/* Set all components of vector `p` of size `n_dims` equal to zero. */
void vzero( float *p )
{
    for (int d=0; d<n_dims; d++)
        p[d] = 0.0f;
}

/* Add vector `p2` to vector `p1`; store result in `p1`.
   Both p1 and p2 are contiguous row-major rows → stride-1 → effective SIMD. */
void vadd( float *p1, const float *p2 )
{
    #pragma omp simd
    for (int d=0; d<n_dims; d++)
        p1[d] += p2[d];
}

/* Multiply each element of vector `p` of size `n_dims` by `v`. */
void vmul( float *p, float v )
{
    #pragma omp simd
    for (int d=0; d<n_dims; d++)
        p[d] *= v;
}

/* Copy `p2` into `p1`. */
void vcopy( float *p1, const float *p2 )
{
    #pragma omp simd
    for (int d=0; d<n_dims; d++)
        p1[d] = p2[d];
}

/* Compute the Euclidean squared distance of `p1` and `p2`.
   Both p1 and p2 are row-major row pointers (stride-1) → effective SIMD.
   Used only for centroid shift computation (K*D, negligible). */
float sqdist( const float *p1, const float *p2 )
{
    float result = 0.0;
    #pragma omp simd reduction(+:result)
    for (int d=0; d<n_dims; d++) {
        result += (p1[d] - p2[d])*(p1[d] - p2[d]);
    }
    return result;
}

/* Compute the squared distance between point i (in column-major data_T)
   and centroid j (in row-major centroids).
   Access pattern:
     data_T[0*N+i], data_T[1*N+i], ..., data_T[(D-1)*N+i]  → stride-N (col-wise)
     centroids[j*D+0], centroids[j*D+1], ..., centroids[j*D+D-1] → stride-1
   The stride-N access on data_T is the accepted cost of column-major layout
   for the per-thread serial access. The benefit is that ACROSS threads at
   a fixed d, consecutive threads access consecutive memory (stride-1). */
static inline float sqdist_T( int i, int j )
{
    float result = 0.0f;
    #pragma omp simd reduction(+:result)
    for (int d = 0; d < n_dims; d++) {
        /* data_T[d*n_points + i]: COLUMN-WISE access within one thread,
           ROW-WISE across threads (for fixed d, consecutive i are adjacent). */
        const float diff = data_T[d * n_points + i] - centroids[j * n_dims + d];
        result += diff * diff;
    }
    return result;
}

/* Accumulate point i (column-major data_T) into row-major accumulator acc.
   Access: data_T[d*N + i] for d=0..D-1 → stride-N (col-wise within thread,
   row-wise across threads). */
static inline void vadd_T( float * restrict acc, int i )
{
    #pragma omp simd
    for (int d = 0; d < n_dims; d++) {
        acc[d] += data_T[d * n_points + i];
    }
}

/******************************************************************************
 **
 ** K-Means algorithm begins here.
 **
 ******************************************************************************/

/* This function can be used to access the arrays (actually, matrices)
   `data`, `centroids` and `new_centroids`. These are all matrices
   with `n_dims` columns. The function returns the linear index of row
   `i` and column `d`. Example: `data[IDX(i, d)]` is equivalent to
   `data[i*n_dims + d]`. */
int IDX(int i, int d)
{
    return i*n_dims + d;
}

/* Return a random integer in a..b. This function must not be
   parallelized, since `rand()` is not thread-safe. */
int randab(int a, int b)
{
    return a + rand() % (b-a+1);
}

/* Centroids are initialized by randomly selecting `n_clusters` data
   points. To select `n_clusters` out of `n_data` elements, we use
   Knuths' algorithm as reported in J. Bentley, "Programming Pearls",
   2nd ed., Addison-Wesley, 2000, p. 126.

   DO NOT PARALLELIZE THIS FUNCTION: `rand()` is not thread-safe. */
void init_centroids( void )
{
    int select = n_clusters;
    int remaining = n_points;
    for (int i=0; (i < n_points) && (select > 0); i++) {
        if ((rand() % remaining) < select) {
            select--;
            /* Select point `i` as one of the centroids. Uses row-major data[]. */
            vcopy( &centroids[IDX(select,0)], &data[IDX(i,0)] );
        }
        remaining--;
    }
}

/* Build data_T (column-major) from data (row-major).
   Called once before the main loop.
   This transpose is O(N*D) — done once, amortised over all iterations.
   For N=2M and D=256: 2M * 256 * 4B = 2 GB → done sequentially,
   takes a few seconds but only once.
   Memory cost: doubles the data footprint (data[] and data_T[] both present).
   Trade-off: acceptable when N*D*iter >> N*D (i.e., iter >> 1, which is
   always true for 150 iterations). */
void build_data_T( void )
{
    /* Parallelise the transpose: each thread handles a block of rows of
       data[] → writes to scattered columns of data_T[] but within each
       thread accesses are contiguous in d → row-wise on data[]. */
    #pragma omp parallel for schedule(static) default(none) \
        shared(data, data_T, n_points, n_dims)
    for (int i = 0; i < n_points; i++) {
        for (int d = 0; d < n_dims; d++) {
            /* Write: data_T[d*N + i] → stride-N across iterations of i,
               but this loop is serial in d for fixed i → acceptable. */
            data_T[d * n_points + i] = data[i * n_dims + d];
        }
    }
}

/* --------------------------------------------------------------------------
 * classify() -- PARALLEL (optimised for large D)
 *
 * Pattern: PARTITION (embarrassingly parallel over points).
 *
 * Same parallel structure as the standard version, but uses sqdist_T()
 * which accesses data_T (column-major) instead of data (row-major).
 *
 * Memory access analysis:
 *   sqdist_T(i, j) accesses data_T[d*N + i] for d=0..D-1.
 *   Within one thread (fixed i): stride-N → COLUMN-WISE on data_T → NOT
 *     cache-friendly for the sequential dimension loop.
 *   Across threads (fixed d, varying i): data_T[d*N+0], data_T[d*N+1], ...
 *     → STRIDE-1 → ROW-WISE on data_T → cache-friendly.
 *   Net effect: with P=8 threads, 8 consecutive points are fetched from
 *     the same cache line at each dimension d. For D=256 this means 256
 *     cache misses shared across 8 threads instead of 256*8 independent
 *     misses in the row-major version → 8x fewer L3 misses.
 * -------------------------------------------------------------------------- */
void classify( void )
{
    for (int j=0; j<n_clusters; j++) {
        counts[j] = 0;
    }

    #pragma omp parallel default(none) shared(data_T, centroids, cluster_of, counts, n_points, n_clusters, n_dims)
    {
        /* Private per-thread accumulator for cluster sizes.
           Avoids a critical section or atomic inside the inner loop.
           K ints → fits in a few cache lines. */
        int *local_counts = (int*)safe_malloc(n_clusters * sizeof(int));
        memset(local_counts, 0, n_clusters * sizeof(int));

        /* Partition the N points across threads (static = even chunks,
           good load balance since all points do the same work). */
        #pragma omp for schedule(static)
        for (int i=0; i<n_points; i++) {
            /* sqdist_T uses data_T (column-major):
               stride-N within a thread, stride-1 across threads. */
            int nearest = 0;
            float mindist = sqdist_T(i, 0);
            for (int j=1; j<n_clusters; j++) {
                const float dist = sqdist_T(i, j);
                if ( dist < mindist ) {
                    mindist = dist;
                    nearest = j;
                }
            }
            /* assign the point to the nearest centroid, and update the
               cluster size. */
            cluster_of[i] = nearest;
            local_counts[nearest]++;   /* private copy: no race */
        }

        /* Merge private counts into the global array.
           Only K atomic increments per thread (K << N) → negligible overhead. */
        for (int j = 0; j < n_clusters; j++) {
            #pragma omp atomic
            counts[j] += local_counts[j];
        }

        free(local_counts);
    } /* implicit barrier here: all threads done before update_centroids() */
}

/* --------------------------------------------------------------------------
 * update_centroids() -- PARALLEL (optimised for large D)
 *
 * Pattern: PARTITION + REDUCE.
 *
 * Phase A uses vadd_T() to accumulate from data_T into row-major local_nc.
 * vadd_T reads data_T[d*N + i] for d=0..D-1 (col-wise within one thread,
 * row-wise across threads) and writes to local_nc[cluster_of[i]*D + d]
 * (row-wise, private → fits in L1).
 *
 * Phase B is identical to the standard version (serial, K*D ops).
 * -------------------------------------------------------------------------- */
float update_centroids( void )
{
    for (int j=0; j<n_clusters; j++) {
        vzero( &new_centroids[IDX(j, 0)] );
    }

    #pragma omp parallel default(none) shared(data_T, centroids, new_centroids, cluster_of, counts, n_points, n_clusters, n_dims)
    {
        const int nc_size = n_clusters * n_dims;

        /* Private per-thread accumulator for new centroid positions.
           Row-major: local_nc[j*D + d].
           vadd_T writes to it row-wise (d=0..D-1 for fixed j = cluster_of[i]).
           For large D, local_nc may not fit in L1 (K*D > 8192 floats when
           K=8 and D>1024), but fits in L2 (1 MB per core). */
        float *local_nc = (float*)safe_malloc(nc_size * sizeof(float));
        memset(local_nc, 0, nc_size * sizeof(float));

        /* Partition: each thread accumulates over its block of points.
           vadd_T reads data_T[d*N+i] (col-wise within thread, row-wise
           across threads) and writes local_nc[cluster_of[i]*D+d] (row-wise). */
        #pragma omp for schedule(static)
        for (int i=0; i<n_points; i++) {
            vadd_T( &local_nc[IDX(cluster_of[i], 0)], i );
        }

        /* Merge: K*D atomic additions per thread. */
        for (int j = 0; j < n_clusters; j++) {
            for (int d = 0; d < n_dims; d++) {
                #pragma omp atomic
                new_centroids[IDX(j, d)] += local_nc[IDX(j, d)];
            }
        }

        free(local_nc);
    } /* implicit barrier: merge complete before normalisation */

    float maxshift = 0.0f;
    for (int j=0; j<n_clusters; j++) {
        /* If a cluster is empty, we simply copy the old centroid to
           the new one. */
        if (counts[j] == 0) {
            vcopy( &new_centroids[IDX(j,0)], &centroids[IDX(j,0)] );
        } else {
            vmul( &new_centroids[IDX(j, 0)], 1.0f/counts[j] );
        }
        const float shift = sqdist( &centroids[IDX(j, 0)], &new_centroids[IDX(j, 0)] );
        if (shift > maxshift)
            maxshift = shift;
        vcopy( &centroids[IDX(j, 0)], &new_centroids[IDX(j, 0)] );
    }

    return maxshift; // is quadratic
}

/******************************************************************************
 **
 ** Input/output functions. DO NOT parallelize them.
 **
 ******************************************************************************/

/* Read the input data from `f`. Each row must contain `n_dims`
   numbers. This function figures out how many numbers are in a row,
   and how many rows there are. Then, it initializes the variables
   `n_dims` and `n_points` accordingly. */
void read_input( FILE *f )
{
    /* BUFLEN must be large enough to hold an entire first line.
       Each float occupies ~11 chars (e.g. "123.456789 "), so for D=1000
       we need at least 11000 bytes. 65536 covers D up to ~5900.
       The original value of 1024 would silently truncate the first line
       for D > ~93, causing n_dims to be computed incorrectly. */
    const size_t BUFLEN = 65536;
    char buffer[BUFLEN];

    /* Get the first line of the input file, and count how many
       numbers are there. This function is not very robust: if the
       first line is empty, the number of dimensions will be zero; if
       the first line has more than `BUFLEN` characters, the number of
       fields will be computed incorrectly. */
    char *i_dont_care = fgets(buffer, BUFLEN, f);
    (void)i_dont_care; /* Avoid a compiler warning. */
    n_dims = -1;
    char *start, *end = buffer;
    do {
        start = end;
        strtof(start, &end);
        n_dims++;
    } while (end != start);

    assert(n_dims > 0); /* If this assertion fails, then the first
                           line of the input is empty. */

    /* Rewind the file and count how many data items are there. */
    rewind(f);
    int n_items = 0;
    float dummy;
    while (1 == fscanf(f, "%f", &dummy))
        n_items++;

    n_points = n_items / n_dims; // RESOLVED BUG: This has to be before the assertion below

    assert(n_items % n_dims == 0); /* If this assertion fails, then
                                      there is some line of the input
                                      file that has != n_dims items.
                                      NOTE: the original code had
                                      `n_points % n_dims == 0` here,
                                      which is wrong: it would fail
                                      whenever n_points is not a
                                      multiple of n_dims (e.g. N=4000,
                                      D=256 → 4000%256=160≠0). The
                                      correct check is on n_items. */

    data = (float*)safe_malloc(n_points * n_dims * sizeof(*data));

    /* Rewind and read the actual data. */
    rewind(f);
    for (int i=0; i<n_points; i++) {
        for (int d=0; d<n_dims; d++) {
            const int nread = fscanf(f, "%f", &data[IDX(i, d)]);
            assert(nread == 1);
        }
    }
}

#ifdef MAKE_MOVIE

/* Save the intermediate coordinates of the centroids into a
   file.

   This function is useful for generating a movie showing how the
   centroids get updated, otherwise it can be omitted.

   This function can be enable by defining the MAKE_MOVIE symbol at
   compilation time. */
void save_centroids( int iter )
{
    char buf[1024];

    snprintf(buf, sizeof(buf), "temp/centroids_%03u.txt", (unsigned)iter);
    FILE *f = fopen(buf, "w"); assert(f != NULL);
    if (f == NULL) {
        fprintf(stderr, "FATAL: can not open file \"%s\" for writing\n", buf);
        exit(EXIT_FAILURE);
    }
    for (int j=0; j<n_clusters; j++) {
        for (int d=0; d<n_dims; d++) {
            fprintf(f, "%f ", centroids[IDX(j, d)]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

/* Save the intermediate coordinates of the points and their clusters
   into a file.

   This function is useful for generating a movie showing how the
   centroids get updated, otherwise it can be omitted.

   This function can be enable by defining the MAKE_MOVIE symbol at
   compilation time.
*/
void save_clusters( int iter )
{
    char buf[1024];

    snprintf(buf, sizeof(buf), "temp/out_%03u.txt", (unsigned)iter);
    FILE *f = fopen(buf, "w");
    if (f == NULL) {
        fprintf(stderr, "FATAL: can not open file \"%s\" for writing\n", buf);
        exit(EXIT_FAILURE);
    }
    for (int i=0; i<n_points; i++) {
        for (int d=0; d<n_dims; d++) {
            fprintf(f, "%f ", data[IDX(i, d)]);
        }
        fprintf(f, "%d\n", cluster_of[i]);
    }
    fclose(f);
}

#endif

/* Print the final result of the computation, i.e, the coordinates of
   the centroids and the list of data points with the cluster id. */
void save_results( FILE *f )
{
    fprintf(f, "# Centroids:\n#\n");
    for (int j=0; j<n_clusters; j++) {
        fprintf(f, "# %3d :", j);
        for (int d=0; d<n_dims; d++) {
            fprintf(f, " %f", centroids[IDX(j, d)]);
        }
        fprintf(f, "\n");
    }
    fprintf(f, "#\n");
    for (int i=0; i<n_points; i++) {
        for (int d=0; d<n_dims; d++) {
            fprintf(f, "%f ", data[IDX(i, d)]);
        }
        fprintf(f, "%d\n", cluster_of[i]);
    }
}

/******************************************************************************
 **
 ** Main program.
 **
 ******************************************************************************/
int main( int argc, char *argv[] )
{
    FILE *inputf, *outputf;
    /* MAXITER: hard upper bound on iterations (original convergence logic).
       If MAX_ITER_FIXED is defined at compile time (e.g. -DMAX_ITER_FIXED=150),
       we run exactly that many iterations regardless of convergence.
       This makes timing reproducible: same N always means same number of
       iterations, so wall-clock time is not contaminated by dataset variance. */
#ifdef MAX_ITER_FIXED
    const int fixed_iters = MAX_ITER_FIXED;
#else
    const int MAXITER = 100;
    const float TOL = 1e-5; // is quadratic, so we need a small tolerance
#endif

    if (argc != 4) {
        fprintf(stderr, "Usage: %s K input_file output_file\n", argv[0]);
        return EXIT_FAILURE;
    }

    srand(123); /* Deterministic initialization of the PRNG. */

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

    fprintf(outputf, "# Data points: %d\n",     n_points);
    fprintf(outputf, "# Dimensions: %d\n",      n_dims);
    fprintf(outputf, "# Clusters: %d\n",        n_clusters);

    printf("\nInput file....... %s\n", argv[2]);
    printf("Output file...... %s\n", argv[3]);
    printf("Data points (N).. %d\n", n_points);
    printf("Dimensions (D)... %d\n", n_dims);
    printf("Clusters (K)..... %d\n\n", n_clusters);
    printf("Threads.......... %d\n\n", omp_get_max_threads());

    centroids = (float*)safe_malloc(n_clusters * n_dims * sizeof(*centroids));
    new_centroids = (float*)safe_malloc(n_clusters * n_dims * sizeof(*new_centroids));
    cluster_of = (int*)safe_malloc(n_points * sizeof(*cluster_of));
    counts = (int*)safe_malloc(n_clusters * sizeof(*counts));

    /* Allocate and build the transposed data array.
       This is a one-time O(N*D) cost, amortised over all iterations.
       data_T[d*N + i] = data[i*D + d]. */
    data_T = (float*)safe_malloc((size_t)n_points * n_dims * sizeof(*data_T));
    build_data_T();

    init_centroids();

    printf("Main loop starts\n\n");

    float shift;
    int iter = 0;
    const double tstart = hpc_gettime();

    do {
        classify();
        /* The following lines are useful only if you want to generate
           a movie of the evolution of the algorithm; if you are
           taking times for performance evaluation purposes, remove
           these lines, otherwise the time will be dominated by I/O
           operations. */
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
    printf("Elapsed time %.3f\n\n", elapsed);

    save_results(outputf);

    fclose(outputf);

    free(data);
    free(data_T);
    free(centroids);
    free(new_centroids); // Was missing in the original code
    free(cluster_of);
    free(counts);

    return EXIT_SUCCESS;
}
