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
 ****************************************************************************
 * omp-k-means.c 
 * compile with:
 *   gcc -DMAX_ITER_FIXED=150 -std=c99 -Wall -Wpedantic -fopenmp -o omp-k-means omp-k-means.c (-O3 optional)
 * run with:
 *   ./omp-k-means K in out (OMP_PROC_BIND=close OMP_PLACES=cores/threads optional)
 * 
 * This actually the fourth version of the OpenMP parallelization.
 * Optimizations over v1/v2/v3:
 *
 * 1. FUSED CLASSIFY + ACCUMULATE (main speedup)
 *    v1-v3 did two separate passes over data[]:
 *      Pass 1 (classify):  read data[i] to find nearest centroid
 *      Pass 2 (update):    read data[i] AGAIN to accumulate into new_centroids
 *    v4 does one pass:
 *      read data[i] once, find nearest centroid AND accumulate immediately.
 *
 * 2. PRE-ALLOCATED THREAD-PRIVATE BUFFERS
 *    local_counts and local_nc are allocated once per thread at the start
 *    of the parallel region and freed at the end. This removes 2*ITERS*P
 *    malloc/free calls from the hot loop. 
 *
 * 3. PERSISTENT PARALLEL REGION
 *    One omp parallel for all iterations. Thread-team creation cost
 *    is O(1) instead of O(2*ITERS).
 *
 * In general we assume K and D are small, so we do parallelization mostly 
 * over N (the largest dimension).
 ****************************************************************************/

/*
I chose to comment only the part of the code that is different from the 
base version given in virtuale to avoid redundancy. 
*/

#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif

#include "../../utils/hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>    /* memset */
#include <assert.h>
#include <omp.h>       /* OpenMP support */

/**************************************************************************
 **  Global variables 
 **************************************************************************/

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

/**************************************************************************
 **  Utility functions
 **  These are the same as in the base version, except for the addition of:
 **  - the use of memset 
 **  - #pragma omp simd to allow vectorization.
 **************************************************************************/
void vmul(float *p, float v)
{
    #pragma omp simd
    for (int d = 0; d<n_dims; d++) p[d] *= v ;
}

void vcopy(float *p1, const float *p2)
{
    #pragma omp simd
    for (int d = 0; d<n_dims; d++) p1[d] = p2[d];
}

float sqdist(const float *p1, const float *p2)
{
    float result = 0.0;
    #pragma omp simd reduction(+:result)
    for (int d = 0; d<n_dims; d++) {
        const float diff = p1[d] - p2[d]; 
        // Differently from the base version, I reuse 
        // diff to compute the squared distance, to avoid 
        // recomputing p1[d] - p2[d] twice.
        result += diff*diff;
    }
    return result;
}

int IDX(int i, int d) { return i*n_dims + d; }

/******************************************************************
 **  Do not parallelize, rand() is not thread-safe.              
 ******************************************************************/
void init_centroids(void)
{
    int select = n_clusters; 
    int remaining = n_points;
    for (int i = 0; (i < n_points) && (select > 0); i++) {
        if ((rand() % remaining) < select) {
            select--;
            vcopy(&centroids[IDX(select, 0)], &data[IDX(i, 0)]);
        }
        remaining--;
    }
}

void read_input( FILE *f )
{
    const size_t BUFLEN = 1024;
    char buffer[BUFLEN];
    char *i_dont_care = fgets(buffer, BUFLEN, f);
    (void)i_dont_care; /* Avoid a compiler warning. */
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

int main(int argc, char *argv[])
{
    FILE *inputf, *outputf;

    /* Fix the number of iterations */
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
    printf("Proc bind........ %s\n", getenv("OMP_PROC_BIND") ? getenv("OMP_PROC_BIND") : "unset");
    printf("Places........... %s\n\n", getenv("OMP_PLACES") ? getenv("OMP_PLACES") : "unset");

    centroids = (float*)safe_malloc(n_clusters*n_dims*sizeof(*centroids));
    new_centroids = (float*)safe_malloc(n_clusters*n_dims*sizeof(*new_centroids));
    cluster_of = (int*)safe_malloc(n_points*sizeof(*cluster_of));
    counts= (int*)safe_malloc(n_clusters*sizeof(*counts));

    init_centroids();

    printf("Main loop starts\n\n");

    float  shift = 0.0f;
    const double tstart = hpc_gettime();

    /* -----------------------------------------------------------------------
     * PERSISTENT PARALLEL REGION
     *
     * One team, created once. Thread-private buffers (local_counts,
     * local_nc) are allocated here and reused for all iterations.
     * This avoids 2 * fixed_iters * P heap operations inside the hot loop.
     *
     * Given our assumption on K and D the buffers stay hot in each core's 
     * L2 across iterations. Especially in our target use case (K=8, D=40):
     *   local_nc size = K * D * 4B = 8 * 40 * 4 = 1280 B -> always in L1
     *   local_counts = K * 4B = 32 B -> always in L1
     *
     * Synchronisation per iteration:
     *   [omp for implicit barrier] after the fused loop
     *   [omp single + implicit barrier] for global reset
     *   [explicit omp barrier] after atomic merge
     *   [omp single + implicit barrier] for normalise (last sync of iter)
     * ----------------------------------------------------------------------- */
    #pragma omp parallel default(none) shared(data, centroids, new_centroids, cluster_of, counts, n_points, n_clusters, n_dims, fixed_iters, shift)
    {
        /* Allocate thread-private buffers once for all iterations. */
        const int nc_size = n_clusters * n_dims;                        /* number of floats for each centroid */
        int *local_counts = (int*)safe_malloc(n_clusters*sizeof(int));  /* [priavte array of counts]*/
        float *local_nc = (float*)safe_malloc(nc_size*sizeof(float));   /* [private array of accumulated coordinates] */

        for (int iter = 0; iter < fixed_iters; iter++) {
            /* Reset private accumulators */
            memset(local_counts, 0, n_clusters*sizeof(int));
            memset(local_nc,0, nc_size*sizeof(float));

            /* -----------------------------------------------------------------
             * FUSED CLASSIFY + ACCUMULATE
             *
             * Each thread processes a contiguous block of N/P points.
             * For each point i:
             *   a) find nearest centroid (K * D multiply-adds via sqdist)
             *   b) write cluster_of[i]                  <- no race (disjoint)
             *   c) increment local_counts[nearest]      <- private, no race
             *   d) add data[i] to local_nc[nearest]     <- private, no race
             *
             * data[i*D .. i*D+D-1] is read ONCE.
             *
             * Pattern: PARTITION (embarrassingly parallel over points) -> schedule(static)
             * Memory access: 
             * - data[] sequential read, 
             * - local_nc[] non-sequential write but fits in L1,
             * - cluster_of[] sequential write,
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

                /* --- step (d): accumulate into local_nc --- */
                float *nc_j = &local_nc[IDX(nearest, 0)];
                #pragma omp simd  /* ex vadd */
                for (int d = 0; d < n_dims; d++)
                    nc_j[d] += pi[d];
            }
            /* Implicit barrier: all threads done with the fused loop. */

            /* Reset global accumulators. One thread; others wait at the
               implicit barrier that follows omp single. */
            #pragma omp single
            {
                memset(counts, 0, n_clusters*sizeof(int));
                memset(new_centroids, 0, nc_size*sizeof(float));
            }

            /* Merge private -> global.
             * For this part of the code, I tried two approaches:
             * 1. A single critical section, 
             * 2. Tried to parallelise also this loop in case of 
             * large D or K (using omp parallel for collapse(2)
             *  schedule(static)).
             * Both approaches work, but the best performance 
             * was achieved with this approach:
             * Using atomic*/
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

            /* Normalise + compute shift. 
               Given our assumption on K and D,we compute this part in serial.
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

            /* If instead K is large, parallelising this loop would be 
               beneficial. A posssible approach would be the commented
               code below, where the maxshift reduction is performed in
               the same loop as normalisation.
               However, in our target use case K=8 is not large enough 
               to benefit from parallelisation (actually with 8 core 
               it is the minimum acceptable).*/
            // #pragma omp for reduction(max:shift) schedule(static)
            // for (int j = 0; j < n_clusters; j++) {
            //     if (counts[j] == 0) {
            //         vcopy(&new_centroids[IDX(j, 0)], &centroids[IDX(j, 0)]);
            //     } else {
            //         vmul(&new_centroids[IDX(j, 0)], 1.0f / counts[j]);
            //     }

            //     const float s = sqdist(&centroids[IDX(j, 0)],
            //                         &new_centroids[IDX(j, 0)]);
            //     if (s > shift)
            //         shift = s;

            //     vcopy(&centroids[IDX(j, 0)], &new_centroids[IDX(j, 0)]);
            // }


            /* All threads see updated centroids[] here. */
        }

        /* Free thread-private buffers (runs once per thread). */
        free(local_counts);
        free(local_nc);
    }
    /* suppress unused-but-set shift var (holds last maxshift) and 
       I commented the print of shift because input/output are the most 
       compute-intensive operations, so it would affect the timing. */
    // printf("Iteration %3d, shift = %f\n", iter, shift);
    (void)shift; /* no-warning */
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
