#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif

#include "../../utils/hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <omp.h>

/* Global variables shared across all threads */
int n_dims;     /* D */
int n_points;   /* N */
int n_clusters; /* K */

float *data;
float *centroids;
float *new_centroids;
int *counts;
int *cluster_of;

/* Per-thread reduction buffers to avoid race conditions and atomics overhead */
int n_threads;
int *thread_counts;
float *thread_new_centroids;

void *safe_malloc(size_t size) {
    void *result = malloc(size);
    assert(result != NULL);
    return result;
}

/* Vector utility functions with SIMD optimization */
void vzero( float *p ) {
    memset(p, 0, n_dims * sizeof(float));
}

void vadd( float *p1, const float *p2 ) {
    #pragma omp simd
    for (int d=0; d<n_dims; d++)
        p1[d] += p2[d];
}

void vmul( float *p, float v ) {
    #pragma omp simd
    for (int d=0; d<n_dims; d++)
        p[d] *= v;
}

void vcopy( float *p1, const float *p2 ) {
    #pragma omp simd
    for (int d=0; d<n_dims; d++)
        p1[d] = p2[d];
}

float sqdist(const float *p1, const float *p2) {
    float result = 0.0f;
    #pragma omp simd reduction(+:result)
    for (int d = 0; d < n_dims; d++) {
        const float diff = p1[d] - p2[d];
        result += diff * diff;
    }
    return result;
}

int IDX(int i, int d) {
    return i*n_dims + d;
}

/* Serial initialization: rand() is not thread-safe */
void init_centroids( void ) {
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

/**
 * Assign points to clusters. 
 * Parallelized over points (N) using a private count buffer per thread 
 * to ensure load balance and prevent race conditions.
 */
void classify( void ) {
    memset(counts, 0, n_clusters * sizeof(int));
    memset(thread_counts, 0, (size_t)n_threads * n_clusters * sizeof(int));

    #pragma omp parallel default(none) shared(data, centroids, cluster_of, counts, thread_counts, n_points, n_clusters, n_dims, n_threads)
    {
        const int tid = omp_get_thread_num();
        int *local_counts = &thread_counts[tid * n_clusters];

        #pragma omp for schedule(static)
        for (int i = 0; i < n_points; i++) {
            int nearest = 0;
            float mindist = sqdist(&data[IDX(i, 0)], &centroids[IDX(0, 0)]);

            for (int j = 1; j < n_clusters; j++) {
                const float dist = sqdist(&data[IDX(i, 0)], &centroids[IDX(j, 0)]);
                if (dist < mindist) {
                    mindist = dist;
                    nearest = j;
                }
            }
            cluster_of[i] = nearest;
            local_counts[nearest]++;
        }
    }

    /* Merge private thread counts into global count */
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < n_clusters; j++) {
        int sum = 0;
        for (int t = 0; t < n_threads; t++)
            sum += thread_counts[t * n_clusters + j];
        counts[j] = sum;
    }
}

/**
 * Update centroid positions.
 * Threads accumulate positions into private buffers (thread_new_centroids)
 * then merge results to minimize synchronization overhead.
 */
void update_centroids( void ) {
    memset(new_centroids, 0, (size_t)n_clusters * n_dims * sizeof(float));
    memset(thread_new_centroids, 0, (size_t)n_threads * n_clusters * n_dims * sizeof(float));

    const int nc_size = n_clusters * n_dims;

    #pragma omp parallel default(none) shared(data, centroids, new_centroids, cluster_of, counts, n_points, n_clusters, n_dims, n_threads, thread_new_centroids, nc_size)
    {
        const int tid = omp_get_thread_num();
        float *local_nc = &thread_new_centroids[tid * nc_size];

        #pragma omp for schedule(static)
        for (int i = 0; i < n_points; i++) {
            vadd(&local_nc[IDX(cluster_of[i], 0)], &data[IDX(i, 0)]);
        }
    }

    /* Merge private centroids into global buffer */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 0; j < n_clusters; j++) {
        for (int d = 0; d < n_dims; d++) {
            float sum = 0.0f;
            const int idx = IDX(j, d);
            for (int t = 0; t < n_threads; t++) {
                sum += thread_new_centroids[t * nc_size + idx];
            }
            new_centroids[idx] = sum;
        }
    }

    /* Final normalization and shift calculation (Serial) */
    float maxshift = 0.0f;
    for (int j = 0; j < n_clusters; j++) {
        if (counts[j] == 0) {
            vcopy(&new_centroids[IDX(j, 0)], &centroids[IDX(j, 0)]);
        } else {
            vmul(&new_centroids[IDX(j, 0)], 1.0f / counts[j]);
        }

        const float shift = sqdist(&centroids[IDX(j, 0)], &new_centroids[IDX(j, 0)]);
        if (shift > maxshift)
            maxshift = shift;

        vcopy(&centroids[IDX(j, 0)], &new_centroids[IDX(j, 0)]);
    }
}

void read_input( FILE *f ) {
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

void save_results( FILE *f ) {
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

int main( int argc, char *argv[] ) {
    FILE *inputf, *outputf;

#ifdef MAX_ITER_FIXED
    const int fixed_iters = MAX_ITER_FIXED;
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

    /* Setup thread environment and reduction buffers */
    omp_set_dynamic(0);
    n_threads = omp_get_max_threads();
    thread_counts = (int*)safe_malloc((size_t)n_threads * n_clusters * sizeof(*thread_counts));
    thread_new_centroids = (float*)safe_malloc((size_t)n_threads * n_clusters * n_dims * sizeof(*thread_new_centroids));

    if ((outputf = fopen(argv[3], "w")) == NULL) {
        fprintf(stderr, "FATAL: can not create output file \"%s\"\n", argv[3]);
        return EXIT_FAILURE;
    }

    centroids     = (float*)safe_malloc(n_clusters * n_dims * sizeof(*centroids));
    new_centroids = (float*)safe_malloc(n_clusters * n_dims * sizeof(*new_centroids));
    cluster_of    = (int*)  safe_malloc(n_points            * sizeof(*cluster_of));
    counts        = (int*)  safe_malloc(n_clusters          * sizeof(*counts));

    init_centroids();
    int iter = 0;
    const double tstart = hpc_gettime();

    do {
        classify();
        update_centroids();
        iter++;

#ifdef MAX_ITER_FIXED
    } while (iter < fixed_iters);
#else
    } while ((iter <= MAXITER) );
#endif

    const double elapsed = hpc_gettime() - tstart;
    printf("Elapsed time %.6f\n", elapsed);

    save_results(outputf);
    fclose(outputf);

    free(data);
    free(centroids);
    free(new_centroids);
    free(cluster_of);
    free(counts);
    free(thread_counts);
    free(thread_new_centroids);

    return EXIT_SUCCESS;
}