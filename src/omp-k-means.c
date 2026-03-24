/****************************************************************************
 *
 * omp-k-means.c -- OpenMP parallelization of the K-Means clustering algorithm.
 *
 * Based on k-means.c by Moreno Marzolla
 * <https://unibo.it/sitoweb/moreno.marzolla/>
 *
 * Parallelization by: Leonardo Billi
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
 * Compile with:
 *      gcc -std=c99 -Wall -Wpedantic -fopenmp omp-k-means.c -o omp-k-means
 *
 * Run with:
 *      OMP_NUM_THREADS=4 ./omp-k-means K input_file output_file
 *
 ****************************************************************************/
#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif

#include "../utils/hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>     /* memset */
#include <assert.h>
#include <omp.h>

/* --------------------------------------------------------------------------
 * Global variables (same as serial version; shared across all threads).
 * -------------------------------------------------------------------------- 
*/

int n_dims;             /* number of dimensions.        D        */
int n_points;           /* number of data points.       N        */
int n_clusters;         /* number of clusters.          K        */

float *data;            /* [array of length (n_points * n_dims)]
                           `&data[i*n_dims]` points to the beginning
                           of the i-th data items, which is an array
                           of `n_dims` floating-point numbers.  */

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
    memset(p, 0, n_dims * sizeof(float));
}

/* Add vector `p1` to vector `p2`; store result in `p1`. Both vectors
   have size `n_dims`. */
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

/* Squared Euclidean distance between p1 and p2.*/
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

/* Centroid initialization (NOT parallelized: rand() is not thread-safe). */
void init_centroids( void )
{
    int select = n_clusters;
    int remaining = n_points;
    for (int i=0; (i < n_points) && (select > 0); i++) {
        if ((rand() % remaining) < select) {
            select--;
            /* Select point `i` as one of the centroids. */
            vcopy( &centroids[IDX(select,0)], &data[IDX(i,0)] );
        }
        remaining--;
    }
}

/* --------------------------------------------------------------------------
 * classify()
 *
 * Pattern: PARTITION (embarrassingly parallel over points).
 *
 * Each thread handles a contiguous block of points (static schedule,
 * default chunk = N/nthreads). Static is chosen because every point
 * performs the same amount of work (K distance computations of D
 * operations each) 
 * Perfect load balance with static scheduling.
 *
 * `counts` and `cluster_of` have disjoint write ranges per iteration:
 *   - cluster_of[i] is written by exactly one thread (the one owning point i).
 *   - counts[j] can be incremented by multiple threads (POSSIBLE RACE CONDITION).
 *
 * Resolution for counts: each thread keeps a PRIVATE copy of counts
 * (local_counts[j]), then they are merged with #pragma omp atomic after
 * the loop. This avoids serialising the inner loop with per-increment atomics.
 *
 * Memory access pattern:  ???????????????????
 *   - data[i*n_dims .. i*n_dims+D-1]: sequential read per point (optimizes cache hit).
 *   - centroids[j*n_dims .. j*n_dims+D-1]: K*D floats total; small enough
 *     to reside in L1/L2 for all threads (K<<N, D small). Each thread
 *     reads the full centroid array.
 * -------------------------------------------------------------------------- */
void classify( void )
{
    /* Reset global counts; will be accumulated from private copies below. */
    memset(counts, 0, n_clusters * sizeof(int));

    #pragma omp parallel default(none) shared(data, centroids, cluster_of, counts, n_points, n_clusters, n_dims)
    {
        /* Private per-thread accumulator for cluster sizes.
           Avoids a critical section or atomic inside the inner loop. */
        int *local_counts = (int*)safe_malloc(n_clusters * sizeof(int));
        memset(local_counts, 0, n_clusters * sizeof(int));

        /* Partition the N points across threads (static = even chunks,
           good load balance since all points do the same work). */
        #pragma omp for schedule(static)
        for (int i = 0; i < n_points; i++) {
            /* Find the nearest centroid for point i.
               Accesses data[i*n_dims .. i*n_dims+D-1] sequentially  good locality. */
            int nearest = 0;
            float mindist = sqdist( &data[IDX(i, 0)], &centroids[IDX(0, 0)] );

            for (int j = 1; j < n_clusters; j++) {
                const float dist = sqdist( &data[IDX(i, 0)], &centroids[IDX(j, 0)] );
                if (dist < mindist) {
                    mindist = dist;
                    nearest = j;
                }
            }
            cluster_of[i] = nearest;
            local_counts[nearest]++; 
        }

        /* Merge private counts into the global array.
           Only K atomic increments per thread (K << N)  negligible overhead. */
        for (int j = 0; j < n_clusters; j++) {
            #pragma omp atomic
            counts[j] += local_counts[j];
        }

        free(local_counts);
    } /* implicit barrier here*/
}

/* --------------------------------------------------------------------------
 * update_centroids()
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
 *   Alternative (not used): #pragma omp atomic on each vadd()  K*D
 *   atomics PER POINT, which would serialise the inner loop.
 *
 *   Memory access in the accumulation:
 *   - data[i*n_dims .. i*n_dims+D-1]: sequential  cache-friendly.
 *   - local_nc[cluster_of[i]*n_dims .. +D-1]: scattered write, but
 *     local_nc is private per thread and fits in L1 (K*D floats, K<<N).
 *
 * Phase B: normalise and compute max shift.
 *   Only K*D operations  serial (negligible vs N*K*D work in Phase A).
 *   Only one thread performs it (omp single), others wait at the implicit barrier.
 * -------------------------------------------------------------------------- */
void update_centroids( void )
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
        memset( local_nc, 0, nc_size * sizeof(float));

        /* Partition: each thread accumulates over its block of points. */
        #pragma omp for schedule(static)
        for (int i = 0; i < n_points; i++) {
            /* data[i*n_dims..] is sequential  spatial locality. */
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
    const size_t BUFLEN = 1024;
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

    assert(n_items % n_dims == 0);
    n_points = n_items / n_dims;

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
 **
 ** Main program.
 **
 ******************************************************************************/
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
    /* Suppress "unused variable" warnings: MAXITER */
    (void)0;
#else
    const int    MAXITER = 100;
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
    printf("Output file...... %s\n", argv[3]);
    printf("Data points (N).. %d\n", n_points);
    printf("Dimensions (D)... %d\n", n_dims);
    printf("Clusters (K)..... %d\n\n", n_clusters);
    printf("Threads.......... %d\n\n", omp_get_max_threads());

    centroids     = (float*)safe_malloc(n_clusters * n_dims * sizeof(*centroids));
    new_centroids = (float*)safe_malloc(n_clusters * n_dims * sizeof(*new_centroids));
    cluster_of    = (int*)  safe_malloc(n_points            * sizeof(*cluster_of));
    counts        = (int*)  safe_malloc(n_clusters          * sizeof(*counts));

    /* DO NOT parallelize init_centroids: rand() is not thread-safe. */
    init_centroids();

    printf("Main loop starts\n\n");

    int iter = 0;
    const double tstart = hpc_gettime();

    do {
        classify();
        update_centroids();
        // I delete the print of shift because input/output are the most compute-intensive part of the code, and printing to console is very slow, so it would affect the timing. If you want to print shift, you can uncomment the line below.
        // printf("Iteration %3d, shift = %f\n", iter, shift);
        iter++;

#ifdef MAX_ITER_FIXED
        /* Fixed-iteration mode: ignore convergence, run exactly fixed_iters. */
    } while (iter < fixed_iters);
#else
        /* Original convergence check. */
    } while ((iter <= MAXITER) );
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
