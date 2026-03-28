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

void *safe_malloc(size_t size) {
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

// The original classify() and update_centroids() functions have been removed
// since their logic has been moved and parallelized inside main.

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
    while (1 == fscanf(f, "%f", &dummy)) n_items++;

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

int main( int argc, char *argv[] ) {
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

    fprintf(outputf, "# Data points: %d\n", n_points);
    fprintf(outputf, "# Dimensions: %d\n", n_dims);
    fprintf(outputf, "# Clusters: %d\n", n_clusters);

    printf("\nInput file....... %s\n", argv[2]);
    printf("Output file...... %s\n", argv[3]);
    printf("Data points (N).. %d\n", n_points);
    printf("Dimensions (D)... %d\n", n_dims);
    printf("Clusters (K)..... %d\n\n", n_clusters);

    centroids = (float*)safe_malloc(n_clusters * n_dims * sizeof(*centroids));
    new_centroids = (float*)safe_malloc(n_clusters * n_dims * sizeof(*new_centroids));
    cluster_of = (int*)safe_malloc(n_points * sizeof(*cluster_of));
    counts = (int*)safe_malloc(n_clusters * sizeof(*counts));

    init_centroids();

    printf("Main loop starts (OpenMP Parallelized)\n\n");

    float shift = 0.0f;
    const double tstart = hpc_gettime(); 

    // START PARALLEL REGION
    #pragma omp parallel
    {
        // 1. Allocate local arrays for reduction (private to each thread)
        int *local_counts = (int*)safe_malloc(n_clusters * sizeof(*local_counts));
        float *local_new_centroids = (float*)safe_malloc(n_clusters * n_dims * sizeof(*local_new_centroids));

        for (int iter = 0; iter < fixed_iters; iter++) {
            
            // Reset local accumulators
            memset(local_counts, 0, n_clusters * sizeof(int));
            memset(local_new_centroids, 0, n_clusters * n_dims * sizeof(float));

            // --- PHASE 1: Classification (Parallel) ---
            #pragma omp for
            for (int i = 0; i < n_points; i++) {
                int nearest = 0;
                float mindist = sqdist(&data[IDX(i, 0)], &centroids[IDX(0, 0)]);
                
                for (int j = 1; j < n_clusters; j++) {
                    float dist = sqdist(&data[IDX(i, 0)], &centroids[IDX(j, 0)]);
                    if (dist < mindist) {
                        mindist = dist;
                        nearest = j;
                    }
                }
                
                cluster_of[i] = nearest;
                local_counts[nearest]++;
                
                // Accumulate data for the new centroids LOCALLY
                for (int d = 0; d < n_dims; d++) {
                    local_new_centroids[IDX(nearest, d)] += data[IDX(i, d)];
                }
            }

            // --- PHASE 2: Global Reduction ---
            // A single thread resets the global counters
            #pragma omp single
            {
                memset(counts, 0, n_clusters * sizeof(int));
                memset(new_centroids, 0, n_clusters * n_dims * sizeof(float));
            }

            // Each thread adds its local contribution to the global arrays
            #pragma omp critical
            {
                for (int j = 0; j < n_clusters; j++) {
                    counts[j] += local_counts[j];
                    for (int d = 0; d < n_dims; d++) {
                        new_centroids[IDX(j, d)] += local_new_centroids[IDX(j, d)];
                    }
                }
            }
            // Mandatory synchronization: wait for all threads to finish adding
            #pragma omp barrier

            // --- PHASE 3: Centroid and Shift Update ---
            #pragma omp single
            {
                shift = 0.0f;
                for (int j = 0; j < n_clusters; j++) {
                    if (counts[j] == 0) {
                        vcopy(&new_centroids[IDX(j, 0)], &centroids[IDX(j, 0)]);
                    } else {
                        vmul(&new_centroids[IDX(j, 0)], 1.0f / counts[j]);
                    }
                    float current_shift = sqdist(&centroids[IDX(j, 0)], &new_centroids[IDX(j, 0)]);
                    if (current_shift > shift) {
                        shift = current_shift;
                    }
                    vcopy(&centroids[IDX(j, 0)], &new_centroids[IDX(j, 0)]);
                }
                // printf("Iteration %3d, shift = %f\n", iter, shift);
            }
            // The single pragma has an implicit barrier at the end, 
            // so the threads will not start the next iteration before the update.
        }

        // Local memory deallocation
        free(local_counts);
        free(local_new_centroids);
    }
    // END PARALLEL REGION

    const double elapsed = hpc_gettime() - tstart;

    printf("\nMain loop completed\n");
    printf("Elapsed time %.3f\n\n", elapsed);

    save_results(outputf);
    fclose(outputf);

    free(data);
    free(centroids);
    free(new_centroids);
    free(cluster_of);
    free(counts);

    return EXIT_SUCCESS;
}