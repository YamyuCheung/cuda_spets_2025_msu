// Jacobi-3 program with GPU acceleration and OpenMP
#include <math.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>

#ifdef USE_FLOAT
typedef float real_t;
#define EPSILON 1e-6f
#else
typedef double real_t;
#define EPSILON 1e-12
#endif

typedef enum {
    MODE_CPU = 0,
    MODE_GPU = 1,
    MODE_COMPARE = 2
} RunMode;

#ifndef L
#define L 384
#endif

#define ITMAX 20
#define BLOCK_SIZE 256
#define Max(a, b) ((a) > (b) ? (a) : (b))
#define MAXEPS 0.5

#define CHECK_CUDA_ERROR(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void function(real_t *A, real_t *B){
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if(i > 0 && i < L-1){
        if(j >0 && j < L-1){
            if(k > 0 && k < L-1){
                int idx = i * L * L + j * L + k;
                B[idx] = (A[(i - 1) * L * L + j * L + k] +
                         A[i * L * L + (j - 1) * L + k] +
                         A[i * L * L + j * L + (k - 1)] +
                         A[i * L * L + j * L + (k + 1)] +
                         A[i * L * L + (j + 1) * L + k] +
                         A[(i + 1) * L * L + j * L + k]) / 6.0f;
            }
        }
    }
}

__global__ void difference(real_t *A, real_t *B, real_t *D){
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if(i > 0 && i < L-1){
        if(j >0 && j < L-1){
            if(k > 0 && k < L-1){
                int idx = i * L * L + j * L + k;
                D[idx] = fabs(B[idx]-A[idx]);
            }
        }
    }
}

__global__ void ab(real_t *A, real_t *B){
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if(i > 0 && i < L-1){
        if(j >0 && j < L-1){
            if(k > 0 && k < L-1){
                int idx = i * L * L + j * L + k;
                A[idx] = B[idx];
            }
        }
    }
}

__global__ void difference_reduce(real_t *A, real_t *B, real_t *block_max) {
    extern __shared__ real_t sdata[];
    // Linear thread index within the block
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

    // Compute global coordinates
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute difference value; set to 0 if out of valid range
    real_t diff_val = 0;
    if(i > 0 && i < L - 1 && j > 0 && j < L - 1 && k > 0 && k < L - 1) {
        int idx = i * L * L + j * L + k;
        diff_val = fabs(B[idx] - A[idx]);
    }
    sdata[tid] = diff_val;
    __syncthreads();

    // Perform reduction in shared memory: assume block size is 512 (8×8×8) and a power of 2
    for (unsigned int s = blockDim.x * blockDim.y * blockDim.z / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = sdata[tid] > sdata[tid + s] ? sdata[tid] : sdata[tid + s];
        }
        __syncthreads();
    }

    // First thread writes the maximum value of the block
    if (tid == 0) {
        int blockIndex = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
        block_max[blockIndex] = sdata[0];
    }
}

void print_gpu_info() {
    int deviceCount;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA-capable device found\n");
        return;
    }

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;
        CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, dev));

        printf("\nGPU Device %d: \"%s\"\n", dev, deviceProp.name);
        printf("  Total global memory: %.2f GB\n",
               (float)deviceProp.totalGlobalMem / 1048576.0f / 1024.0f);
        printf("  Compute capability: %d.%d\n",
               deviceProp.major, deviceProp.minor);
        printf("  Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Max thread dimensions: (%d, %d, %d)\n",
               deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("  Max grid dimensions: (%d, %d, %d)\n",
               deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    }

    size_t free, total;
    CHECK_CUDA_ERROR(cudaMemGetInfo(&free, &total));
    printf("\nMemory Info:\n");
    printf("  Total GPU memory: %.2f GB\n", (float)total / 1048576.0f / 1024.0f);
    printf("  Available GPU memory: %.2f GB\n", (float)free / 1048576.0f / 1024.0f);
}

int jacobi_cpu(real_t*** A, real_t*** B, int size, int max_iter, real_t max_eps) {
    int it;
    real_t eps;

    for (it = 1; it <= max_iter; it++) {
        eps = 0;

        #pragma omp parallel for collapse(3) reduction(max:eps)
        for (int i = 1; i < size - 1; i++) {
            for (int j = 1; j < size - 1; j++) {
                for (int k = 1; k < size - 1; k++) {
                    real_t tmp = fabs(B[i][j][k] - A[i][j][k]);
                    eps = Max(tmp, eps);
                    A[i][j][k] = B[i][j][k];
                }
            }
        }

        #pragma omp parallel for collapse(3)
        for (int i = 1; i < size - 1; i++) {
            for (int j = 1; j < size - 1; j++) {
                for (int k = 1; k < size - 1; k++) {
                    B[i][j][k] = (A[i-1][j][k] + A[i][j-1][k] + A[i][j][k-1] +
                                 A[i][j][k+1] + A[i][j+1][k] + A[i+1][j][k]) / 6.0;
                }
            }
        }

        printf(" IT = %4i   EPS = %14.7E\n", it, eps);
        if (eps < max_eps)
            break;
    }

    return it;
}

int jacobi_gpu(real_t* d_A, real_t* d_B, int size, int max_iter, real_t max_eps) {
    int it;
    real_t eps;

    // Use the originally defined thread configuration (8,8,8)
    dim3 threads(8, 8, 8);
    dim3 grid((L + threads.x - 1) / threads.x,
              (L + threads.y - 1) / threads.y,
              (L + threads.z - 1) / threads.z);

    // Calculate the total number of blocks in the grid
    int numBlocks = grid.x * grid.y * grid.z;
    // Allocate memory for block-wise results (much smaller than L³)
    real_t *d_blockDiff;
    CHECK_CUDA_ERROR(cudaMalloc(&d_blockDiff, numBlocks * sizeof(real_t)));

    // Calculate the required shared memory size
    size_t sharedMemSize = threads.x * threads.y * threads.z * sizeof(real_t);

    for (it = 1; it <= max_iter; it++) {
        eps = 0;

        // Use the new reduction kernel to compute the maximum difference per block
        difference_reduce<<<grid, threads, sharedMemSize>>>(d_A, d_B, d_blockDiff);
        // Use Thrust to reduce all block results into a single value
        thrust::device_ptr<real_t> d_blockDiff_ptr = thrust::device_pointer_cast(d_blockDiff);
        eps = thrust::reduce(d_blockDiff_ptr, d_blockDiff_ptr + numBlocks, 0.0, thrust::maximum<real_t>());

        // Then perform the original update steps
        ab<<<grid, threads>>>(d_A, d_B);
        function<<<grid, threads>>>(d_A, d_B);

        printf(" IT = %4i   EPS = %14.7E\n", it, eps);
        if (eps < max_eps)
            break;
    }

    CHECK_CUDA_ERROR(cudaFree(d_blockDiff));
    return it;
}


int main(int argc, char **argv) {
    int i, j, k;
    double start_time, end_time, cpu_time;
    int iterations;
    RunMode mode = MODE_CPU;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--cpu") == 0) {
            mode = MODE_CPU;
        } else if (strcmp(argv[i], "--gpu") == 0) {
            mode = MODE_GPU;
        } else if (strcmp(argv[i], "--compare") == 0) {
            mode = MODE_COMPARE;
        } else if (strncmp(argv[i], "--size=", 7) == 0) {
            int custom_size = atoi(argv[i] + 7);
            if (custom_size > 0) {
                printf("Warning: Custom size specified, but L is defined as %d at compile time\n", L);
                printf("Recompile with -DL=%d to use this size\n", custom_size);
            }
        }
    }

    printf("3D Jacobi Implementation\n");
    printf("Problem size: %d x %d x %d\n", L, L, L);
    printf("Data type: %s\n", sizeof(real_t) == sizeof(float) ? "float" : "double");

    if (mode == MODE_GPU || mode == MODE_COMPARE) {
        print_gpu_info();
    }

    if (mode == MODE_GPU || mode == MODE_COMPARE) {
        size_t free_mem, total_mem;
        CHECK_CUDA_ERROR(cudaMemGetInfo(&free_mem, &total_mem));

        size_t required_mem = 2 * L * L * L * sizeof(real_t);
        if (required_mem > free_mem) {
            printf("Error: Problem size is too large for available GPU memory.\n");
            printf("Required: %.2f GB, Available: %.2f GB\n",
                   (float)required_mem / 1048576.0f / 1024.0f,
                   (float)free_mem / 1048576.0f / 1024.0f);
            return 1;
        }
    }

    // Allocate and initialize CPU arrays
    real_t ***A = (real_t ***)malloc(L * sizeof(real_t **));
    real_t ***B = (real_t ***)malloc(L * sizeof(real_t **));
    real_t *A_data = (real_t *)malloc(L * L * L * sizeof(real_t));
    real_t *B_data = (real_t *)malloc(L * L * L * sizeof(real_t));

    if (A == NULL || B == NULL || A_data == NULL || B_data == NULL) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    for (i = 0; i < L; i++) {
        A[i] = (real_t **)malloc(L * sizeof(real_t *));
        B[i] = (real_t **)malloc(L * sizeof(real_t *));
        if (A[i] == NULL || B[i] == NULL) {
            printf("Memory allocation failed!\n");
            return 1;
        }
        for (j = 0; j < L; j++) {
            A[i][j] = &A_data[((i * L) + j) * L];
            B[i][j] = &B_data[((i * L) + j) * L];
        }
    }

    // Initialize arrays
    for (i = 0; i < L; i++) {
        for (j = 0; j < L; j++) {
            for (k = 0; k < L; k++) {
                A[i][j][k] = 0;
                if (i == 0 || j == 0 || k == 0 || i == L - 1 || j == L - 1 || k == L - 1)
                    B[i][j][k] = 0;
                else
                    B[i][j][k] = 4 + i + j + k;
            }
        }
    }

    // Create 1D copies for the GPU
    real_t *d_A = NULL;
    real_t *d_B = NULL;

    if (mode == MODE_GPU || mode == MODE_COMPARE) {
        // Allocate GPU memory
        CHECK_CUDA_ERROR(cudaMalloc(&d_A, L * L * L * sizeof(real_t)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_B, L * L * L * sizeof(real_t)));

        // Copy initialized data to GPU
        CHECK_CUDA_ERROR(cudaMemcpy(d_A, A_data, L * L * L * sizeof(real_t), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_B, B_data, L * L * L * sizeof(real_t), cudaMemcpyHostToDevice));
    }

    // Run selected mode
    if (mode == MODE_CPU || mode == MODE_COMPARE) {
        printf("\nRunning CPU version...\n");
        start_time = omp_get_wtime();
        iterations = jacobi_cpu(A, B, L, ITMAX, MAXEPS);
        end_time = omp_get_wtime();
        cpu_time = end_time - start_time;

        printf("\n Jacobi3D CPU benchmark completed.\n");
        printf(" Size             = %4d x %4d x %4d\n", L, L, L);
        printf(" Iterations       = %d\n", iterations - 1);
        printf(" Time (seconds)   = %.6f\n", cpu_time);
        printf(" Precision type   = %s precision\n", sizeof(real_t) == sizeof(double) ? "double" : "single");
    }

    if (mode == MODE_GPU || mode == MODE_COMPARE) {
        // If in compare mode, reset A and B on the GPU
        if (mode == MODE_COMPARE) {
            // Reset GPU arrays to match initial state
            for (i = 0; i < L; i++) {
                for (j = 0; j < L; j++) {
                    for (k = 0; k < L; k++) {
                        int idx = i * L * L + j * L + k;
                        A_data[idx] = 0;
                        if (i == 0 || j == 0 || k == 0 || i == L - 1 || j == L - 1 || k == L - 1)
                            B_data[idx] = 0;
                        else
                            B_data[idx] = 4 + i + j + k;
                    }
                }
            }

            CHECK_CUDA_ERROR(cudaMemcpy(d_A, A_data, L * L * L * sizeof(real_t), cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaMemcpy(d_B, B_data, L * L * L * sizeof(real_t), cudaMemcpyHostToDevice));
        }

        printf("\nRunning GPU version...\n");
        start_time = omp_get_wtime();
        iterations = jacobi_gpu(d_A, d_B, L, ITMAX, MAXEPS);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        end_time = omp_get_wtime();

        printf("\n Jacobi3D GPU benchmark completed.\n");
        printf(" Size             = %4d x %4d x %4d\n", L, L, L);
        printf(" Iterations       = %d\n", iterations - 1);
        printf(" Time (seconds)   = %.6f\n", end_time - start_time);
        printf(" Precision type   = %s precision\n", sizeof(real_t) == sizeof(double) ? "double" : "single");

        // If in compare mode, calculate speedup
        if (mode == MODE_COMPARE) {
            printf(" GPU speedup      = %.2f x\n",
                (end_time - start_time > 0.001) ?
                cpu_time / (end_time - start_time) : 0);
        }
    }



    // CLEAN
    for (i = 0; i < L; i++) {
        free(A[i]);
        free(B[i]);
    }
    free(A);
    free(B);
    free(A_data);
    free(B_data);

    if (mode == MODE_GPU || mode == MODE_COMPARE) {
        CHECK_CUDA_ERROR(cudaFree(d_A));
        CHECK_CUDA_ERROR(cudaFree(d_B));
    }

    return 0;
}

