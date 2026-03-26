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

void vcopy( float *p1, const float *p2 ) {
    for (int d=0; d<n_dims; d++) p1[d] = p2[d];
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
// KERNEL 1: Classification & Reduction (Transposed Data)
// --------------------------------------------------------------------------
__global__ void kmeans_classify_and_reduce_kernel(
    const float* __restrict__ data_t, 
    const float* __restrict__ centroids,
    int* __restrict__ cluster_of,
    float* __restrict__ new_centroids,
    int* __restrict__ counts,
    int N, int K, int D)
{
    extern __shared__ float s_mem[];
    float* s_centroids = s_mem;
    float* s_new_centroids = (float*)&s_centroids[K * D];
    int* s_counts = (int*)&s_new_centroids[K * D];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    for (int idx = tid; idx < K * D; idx += blockDim.x) {
        s_centroids[idx] = centroids[idx];
        s_new_centroids[idx] = 0.0f;
    }
    for (int idx = tid; idx < K; idx += blockDim.x) {
        s_counts[idx] = 0;
    }
    __syncthreads();

    if (i < N) {
        float min_dist = 1e30f;
        int nearest = 0;

        for (int k = 0; k < K; k++) {
            float dist = 0.0f;
            for (int d = 0; d < D; d++) {
                float point_val = data_t[d * N + i]; // Coalesced Memory Access!
                float diff = point_val - s_centroids[k * D + d];
                dist += diff * diff;
            }
            if (dist < min_dist) {
                min_dist = dist;
                nearest = k;
            }
        }
        cluster_of[i] = nearest;

        atomicAdd(&s_counts[nearest], 1);
        for (int d = 0; d < D; d++) {
            atomicAdd(&s_new_centroids[nearest * D + d], data_t[d * N + i]);
        }
    }
    __syncthreads();

    for (int idx = tid; idx < K * D; idx += blockDim.x) {
        atomicAdd(&new_centroids[idx], s_new_centroids[idx]);
    }
    for (int idx = tid; idx < K; idx += blockDim.x) {
        if (s_counts[idx] > 0) atomicAdd(&counts[idx], s_counts[idx]);
    }
}

// --------------------------------------------------------------------------
// KERNEL 2: Centroid Update & Shift Calculation
// Grid = K blocks (one per cluster). Block = 256 threads.
// Parallelized over the D dimension to handle huge D values efficiently.
// --------------------------------------------------------------------------
__global__ void update_centroids_kernel(
    const float* __restrict__ new_centroids,
    const int* __restrict__ counts,
    float* __restrict__ centroids,
    float* __restrict__ shifts,
    int K, int D)
{
    int j = blockIdx.x; // The cluster this block is responsible for
    int count = counts[j];
    
    // Shared memory for block reduction (summing squared differences)
    extern __shared__ float s_diff[]; 
    int tid = threadIdx.x;
    
    float diff_sq_sum = 0.0f;

    // Threads cooperatively process the D dimensions
    for (int d = tid; d < D; d += blockDim.x) {
        int idx = j * D + d;
        float old_val = centroids[idx];
        float new_val = new_centroids[idx];
        
        if (count > 0) {
            new_val = new_val / (float)count; // Average
        } else {
            new_val = old_val; // Keep old if cluster is empty
        }
        
        centroids[idx] = new_val; // Update globally
        
        float diff = old_val - new_val;
        diff_sq_sum += diff * diff;
    }
    
    s_diff[tid] = diff_sq_sum;
    __syncthreads();
    
    // Perform tree reduction in shared memory to find total shift for this cluster
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_diff[tid] += s_diff[tid + stride];
        }
        __syncthreads();
    }
    
    // Thread 0 writes the final shift of cluster j to global memory
    if (tid == 0) {
        shifts[j] = s_diff[0];
    }
}

int main( int argc, char *argv[] )
{
    FILE *inputf, *outputf;

#ifdef MAX_ITER_FIXED
    const int fixed_iters = MAX_ITER_FIXED;
#else    
    const int fixed_iters = 150; 
#endif

    if (argc != 4) return EXIT_FAILURE;

    srand(123); 
    n_clusters = atoi(argv[1]);

    if ((inputf = fopen(argv[2], "r")) == NULL) return EXIT_FAILURE;
    read_input(inputf);
    fclose(inputf);
    
    if ((outputf = fopen(argv[3], "w")) == NULL) return EXIT_FAILURE;

    // TRANSPOSE DATA
    float* data_t = (float*)safe_malloc(n_points * n_dims * sizeof(float));
    for(int i = 0; i < n_points; i++) {
        for(int d = 0; d < n_dims; d++) {
            data_t[d * n_points + i] = data[IDX(i, d)];
        }
    }

    centroids = (float*)safe_malloc(n_clusters * n_dims * sizeof(*centroids));
    cluster_of = (int*)safe_malloc(n_points * sizeof(*cluster_of));
    init_centroids();

    float *d_data_t, *d_centroids, *d_new_centroids, *d_shifts;
    int *d_counts, *d_cluster_of;

    CHECK_CUDA(cudaMalloc((void**)&d_data_t, n_points * n_dims * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_centroids, n_clusters * n_dims * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_new_centroids, n_clusters * n_dims * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_cluster_of, n_points * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_counts, n_clusters * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_shifts, n_clusters * sizeof(float))); // New!

    CHECK_CUDA(cudaMemcpy(d_data_t, data_t, n_points * n_dims * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_centroids, centroids, n_clusters * n_dims * sizeof(float), cudaMemcpyHostToDevice));

    size_t shared_mem_classify = (2 * n_clusters * n_dims * sizeof(float)) + (n_clusters * sizeof(int));
    size_t shared_mem_update   = BLKDIM * sizeof(float); // Used for reduction

    dim3 blockClassify(BLKDIM);
    dim3 gridClassify((n_points + blockClassify.x - 1) / blockClassify.x);
    
    dim3 blockUpdate(BLKDIM);
    dim3 gridUpdate(n_clusters); // One block per cluster

    float h_shifts[n_clusters]; // CPU array to fetch the 8 shifts
    int iter = 0;
    
    cudaDeviceSynchronize();
    const double tstart = hpc_gettime();

    do {
        CHECK_CUDA(cudaMemset(d_new_centroids, 0, n_clusters * n_dims * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_counts, 0, n_clusters * sizeof(int)));

        // Phase 1: Classification & partial sums
        kmeans_classify_and_reduce_kernel<<<gridClassify, blockClassify, shared_mem_classify>>>(
            d_data_t, d_centroids, d_cluster_of, d_new_centroids, d_counts, n_points, n_clusters, n_dims
        );
        
        // Phase 2: Compute average, apply new centroids, and get squared shifts
        update_centroids_kernel<<<gridUpdate, blockUpdate, shared_mem_update>>>(
            d_new_centroids, d_counts, d_centroids, d_shifts, n_clusters, n_dims
        );

        // Fetch ONLY the K shift values (K=8 floats) back to CPU
        CHECK_CUDA(cudaMemcpy(h_shifts, d_shifts, n_clusters * sizeof(float), cudaMemcpyDeviceToHost));
        
        float shift = 0.0f;
        for(int j=0; j<n_clusters; j++) {
            if(h_shifts[j] > shift) shift = h_shifts[j];
        }

        iter++;
    } while (iter < fixed_iters);

    cudaDeviceSynchronize();
    const double elapsed = hpc_gettime() - tstart;
    printf("Elapsed time %.6f\n\n", elapsed);

    // Fetch final results back to CPU for saving
    CHECK_CUDA(cudaMemcpy(centroids, d_centroids, n_clusters * n_dims * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(cluster_of, d_cluster_of, n_points * sizeof(int), cudaMemcpyDeviceToHost));

    save_results(outputf);
    fclose(outputf);

    cudaFree(d_data_t); cudaFree(d_centroids); cudaFree(d_new_centroids);
    cudaFree(d_cluster_of); cudaFree(d_counts); cudaFree(d_shifts);
    free(data); free(data_t); free(centroids); free(cluster_of);

    return EXIT_SUCCESS;
}