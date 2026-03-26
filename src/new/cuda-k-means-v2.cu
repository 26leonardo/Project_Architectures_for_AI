#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif

#include "../../utils/hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>

#define BLKDIM 256

int n_dims;
int n_points;
int n_clusters;

float *data;
float *centroids;
float *new_centroids;
int *counts;
int *cluster_of;

/* Utility function to check CUDA errors */
#define CHECK_CUDA(call)                                                \
{                                                                       \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error at %s:%d - %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(err));           \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
}

void *safe_malloc(size_t size) {
    void *result = malloc(size);
    assert(result != NULL);
    return result;
}

void vzero( float *p ) {
    for (int d=0; d<n_dims; d++) p[d] = 0.0f;
}

void vadd( float *p1, const float *p2 ) {
    for (int d=0; d<n_dims; d++) p1[d] += p2[d];
}

void vmul( float *p, float v ) {
    for (int d=0; d<n_dims; d++) p[d] *= v;
}

void vcopy( float *p1, const float *p2 ) {
    for (int d=0; d<n_dims; d++) p1[d] = p2[d];
}

float sqdist( float *p1, float *p2 ) {
    float result = 0.0;
    for (int d=0; d<n_dims; d++) {
        result += (p1[d] - p2[d])*(p1[d] - p2[d]);
    }
    return result;
}

int IDX(int i, int d) {
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

void read_input( FILE *f ) {
    const size_t BUFLEN = 1638400; // 1.6MB buffer to handle larger lines
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

// --------------------------------------------------------------------------
// CUDA KERNEL
// Computes classification and partial reduction in Shared Memory.
// Expects data_t to be transposed (D x N) for perfect coalesced reads.
// --------------------------------------------------------------------------
__global__ void kmeans_classify_and_reduce_kernel(
    const float* __restrict__ data_t, 
    const float* __restrict__ centroids,
    int* __restrict__ cluster_of,
    float* __restrict__ new_centroids,
    int* __restrict__ counts,
    int N, int K, int D)
{
    // Shared memory layout:
    // 1. centroids [K * D]
    // 2. local_new_centroids [K * D]
    // 3. local_counts [K]
    extern __shared__ float s_mem[];
    float* s_centroids = s_mem;
    float* s_new_centroids = (float*)&s_centroids[K * D];
    int* s_counts = (int*)&s_new_centroids[K * D];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // Phase 1: Collaborative initialization of shared memory
    for (int idx = tid; idx < K * D; idx += blockDim.x) {
        s_centroids[idx] = centroids[idx];
        s_new_centroids[idx] = 0.0f;
    }
    for (int idx = tid; idx < K; idx += blockDim.x) {
        s_counts[idx] = 0;
    }
    
    // Ensure all shared memory is initialized before proceeding
    __syncthreads();

    // Phase 2: Classification and Local Reduction
    if (i < N) {
        float min_dist = 1e30f;
        int nearest = 0;

        // Find nearest centroid
        for (int k = 0; k < K; k++) {
            float dist = 0.0f;
            for (int d = 0; d < D; d++) {
                // data_t is stored column-major: d * N + i
                float point_val = data_t[d * N + i]; 
                float diff = point_val - s_centroids[k * D + d];
                dist += diff * diff;
            }
            if (dist < min_dist) {
                min_dist = dist;
                nearest = k;
            }
        }

        // Save classification result
        cluster_of[i] = nearest;

        // Atomically update block-local shared memory
        atomicAdd(&s_counts[nearest], 1);
        for (int d = 0; d < D; d++) {
            float point_val = data_t[d * N + i];
            atomicAdd(&s_new_centroids[nearest * D + d], point_val);
        }
    }

    // Ensure all threads have finished local accumulation
    __syncthreads();

    // Phase 3: Commit block-local results to Global Memory
    for (int idx = tid; idx < K * D; idx += blockDim.x) {
        if (s_new_centroids[idx] != 0.0f) {
            atomicAdd(&new_centroids[idx], s_new_centroids[idx]);
        }
    }
    for (int idx = tid; idx < K; idx += blockDim.x) {
        if (s_counts[idx] > 0) {
            atomicAdd(&counts[idx], s_counts[idx]);
        }
    }
}

// --------------------------------------------------------------------------
// MAIN PROGRAM
// --------------------------------------------------------------------------
int main( int argc, char *argv[] )
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

    // --- CUDA OPTIMIZATION: Transpose Data to enable Coalesced Access ---
    float* data_t = (float*)safe_malloc(n_points * n_dims * sizeof(float));
    for(int i = 0; i < n_points; i++) {
        for(int d = 0; d < n_dims; d++) {
            data_t[d * n_points + i] = data[IDX(i, d)];
        }
    }

    fprintf(outputf, "# Data points: %d\n", n_points);
    fprintf(outputf, "# Dimensions: %d\n",  n_dims);
    fprintf(outputf, "# Clusters: %d\n",    n_clusters);

    printf("\nInput file....... %s\n", argv[2]);
    printf("Output file...... %s\n", argv[3]);
    printf("Data points (N).. %d\n", n_points);
    printf("Dimensions (D)... %d\n", n_dims);
    printf("Clusters (K)..... %d\n\n", n_clusters);

    // Host memory allocation
    centroids = (float*)safe_malloc(n_clusters * n_dims * sizeof(*centroids));
    new_centroids = (float*)safe_malloc(n_clusters * n_dims * sizeof(*new_centroids));
    cluster_of = (int*)safe_malloc(n_points * sizeof(*cluster_of));
    counts = (int*)safe_malloc(n_clusters * sizeof(*counts));

    init_centroids();

    // --- Device memory allocation ---
    float *d_data_t, *d_centroids, *d_new_centroids;
    int *d_counts, *d_cluster_of;

    CHECK_CUDA(cudaMalloc((void**)&d_data_t, n_points * n_dims * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_centroids, n_clusters * n_dims * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_new_centroids, n_clusters * n_dims * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_cluster_of, n_points * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_counts, n_clusters * sizeof(int)));

    // Copy static data to device
    CHECK_CUDA(cudaMemcpy(d_data_t, data_t, n_points * n_dims * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_centroids, centroids, n_clusters * n_dims * sizeof(float), cudaMemcpyHostToDevice));

    // Dynamic Shared Memory calculation
    // We need 2 arrays of size [K * D] (floats) and 1 array of size [K] (ints)
    size_t shared_mem_size = (2 * n_clusters * n_dims * sizeof(float)) + (n_clusters * sizeof(int));

    // Block and Grid setup
    dim3 block(BLKDIM);
    dim3 grid((n_points + block.x - 1) / block.x);

    printf("Grid Size: %d blocks, Block Size: %d threads\n", grid.x, block.x);
    printf("Shared Memory per block: %zu bytes\n", shared_mem_size);
    printf("Main loop starts (CUDA Accelerated)\n\n");

    float shift;
    int iter = 0;
    
    // Warm-up to exclude CUDA context initialization time from elapsed time
    cudaDeviceSynchronize();
    const double tstart = hpc_gettime();

    do {
        // Reset accumulators on device
        CHECK_CUDA(cudaMemset(d_new_centroids, 0, n_clusters * n_dims * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_counts, 0, n_clusters * sizeof(int)));

        // Launch Kernel
        kmeans_classify_and_reduce_kernel<<<grid, block, shared_mem_size>>>(
            d_data_t, d_centroids, d_cluster_of, d_new_centroids, d_counts, n_points, n_clusters, n_dims
        );
        CHECK_CUDA(cudaGetLastError());

        // Bring partial results back to Host to compute the shift and new centroids
        CHECK_CUDA(cudaMemcpy(new_centroids, d_new_centroids, n_clusters * n_dims * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(counts, d_counts, n_clusters * sizeof(int), cudaMemcpyDeviceToHost));

        // --- CPU Update Phase (Extremely fast for K=8, D=40) ---
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

        // Copy updated centroids back to Device for next iteration
        CHECK_CUDA(cudaMemcpy(d_centroids, centroids, n_clusters * n_dims * sizeof(float), cudaMemcpyHostToDevice));

        // printf("Iteration %3d, shift = %f\n", iter, shift);
        iter++;
    } while (iter < fixed_iters);

    // Final synchronization and time fetch
    cudaDeviceSynchronize();
    const double elapsed = hpc_gettime() - tstart;

    // Retrieve final classifications
    CHECK_CUDA(cudaMemcpy(cluster_of, d_cluster_of, n_points * sizeof(int), cudaMemcpyDeviceToHost));

    printf("\nMain loop completed\n");
    printf("Elapsed time %.6f\n\n", elapsed);

    save_results(outputf);
    fclose(outputf);

    // Cleanup Device
    cudaFree(d_data_t);
    cudaFree(d_centroids);
    cudaFree(d_new_centroids);
    cudaFree(d_cluster_of);
    cudaFree(d_counts);

    // Cleanup Host
    free(data);
    free(data_t);
    free(centroids);
    free(new_centroids);
    free(cluster_of);
    free(counts);

    return EXIT_SUCCESS;
}