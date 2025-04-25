/*
 * This program implements a dual-GPU Alternating Direction Implicit (ADI) solver
 * for a 3D Poisson equation. It partitions the domain along the x‑axis into slabs,
 * distributes them across two GPUs, and performs Gauss–Seidel sweeps in the i, j,
 * and k directions. Boundary slices are exchanged between GPUs each iteration,
 * and Thrust is used to compute the maximum error per block for convergence checking.
 */

#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#define nx 900
#define ny 900
#define nz 900
#define BLOCK_SIZE   32         // threads per block for compute sweeps
#define NUM_GPUS     2          // use 2 GPUs
#define ITMAX        10
#define CONV_THRESH  0.01

// Macro to check CUDA calls for errors and abort on failure.
#define CHECK_CUDA_ERROR(call)                                                   \
do {                                                                             \
    cudaError_t err = call;                                                      \
    if (err != cudaSuccess) {                                                    \
        fprintf(stderr, "CUDA error in %s:%d: %s\n",                             \
                __FILE__, __LINE__, cudaGetErrorString(err));                    \
        exit(EXIT_FAILURE);                                                      \
    }                                                                            \
} while (0)

// Helper for comparing values
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// Initialize the host grid: set boundary values to a ramp, interior to zero.
void init(double *grid) {
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j)
    for (int k = 0; k < nz; ++k) {
        size_t idx = (size_t)i * ny * nz + j * nz + k;
        if (i == 0 || i == nx - 1 ||
            j == 0 || j == ny - 1 ||
            k == 0 || k == nz - 1) {
            grid[idx] = 10.0 * i / (nx - 1)
                        + 10.0 * j / (ny - 1)
                        + 10.0 * k / (nz - 1);
        } else {
            grid[idx] = 0.0;
        }
    }
}

// Optimized sweep in the i‑direction
__global__ void sweep_i(double *devGrid, int slabWidth) {
    // Calculate global thread ID
    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + tid;

    // Total number of y-z planes
    int total_yz_lines = ny * nz;
    if (global_id >= total_yz_lines)
        return;

    // Calculate the y and z coordinates
    int z_coord = global_id % nz;
    int y_coord = global_id / nz;

    // Skip boundary points
    if (y_coord <= 0 || y_coord >= ny-1 || z_coord <= 0 || z_coord >= nz-1)
        return;

    // Precompute offsets for efficient memory access
    int yz_offset = y_coord * nz + z_coord;
    int stride_i = ny * nz;

    // Begin with boundary value at i=0
    double current_val = devGrid[yz_offset];

    // Process the line from left to right, respecting slab width
    #pragma unroll 4
    for (int i = 1; i < slabWidth-1; ++i) {
        // Calculate indices
        int right_idx = (i+1) * stride_i + yz_offset;
        int current_idx = i * stride_i + yz_offset;

        // Compute new value
        current_val = 0.5 * (current_val + devGrid[right_idx]);

        // Update grid point
        devGrid[current_idx] = current_val;
    }
}

// Optimized sweep in the j‑direction - fixed to include slabWidth parameter
__global__ void sweep_j(double *devGrid, int slabWidth) {
    // Calculate global thread ID
    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + tid;

    // Total number of x-z planes (using slabWidth)
    int total_xz_lines = slabWidth * nz;
    if (global_id >= total_xz_lines)
        return;

    // Calculate x and z coordinates
    int z_coord = global_id % nz;
    int x_coord = global_id / nz;

    // Skip boundary points and points outside our local slab
    if (x_coord <= 0 || x_coord >= slabWidth-1 || z_coord <= 0 || z_coord >= nz-1)
        return;

    // Precompute offsets for better memory access
    int stride_i = ny * nz;
    int stride_j = nz;
    int base_idx = x_coord * stride_i + z_coord;

    // Begin with the boundary value at j=0
    double current_val = devGrid[base_idx];

    // Process the vertical line from top to bottom
    #pragma unroll 4
    for (int j = 1; j < ny-1; ++j) {
        // Calculate indices
        int current_idx = base_idx + j * stride_j;
        int below_idx = base_idx + (j+1) * stride_j;

        // Compute new value
        current_val = 0.5 * (current_val + devGrid[below_idx]);

        // Update grid point
        devGrid[current_idx] = current_val;
    }
}

// Optimized sweep in the k‑direction with error calculation
__global__ void sweep_k(double *devGrid, double *devBlockMaxErr, int slabWidth) {
    // Calculate global thread ID
    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + tid;

    // Total number of x-y planes
    int total_xy_lines = slabWidth * ny;
    if (global_id >= total_xy_lines)
        return;

    // Calculate local x and y coordinates within slab
    int y_coord = global_id % ny;
    int x_local = global_id / ny;

    // Skip boundary points
    if (x_local <= 0 || x_local >= slabWidth-1 || y_coord <= 0 || y_coord >= ny-1) {
        // For boundary threads, don't contribute to error
        return;
    }

    // Precompute offsets
    int stride_i = ny * nz;
    int stride_j = nz;
    int base_idx = x_local * stride_i + y_coord * stride_j;

    // Get 1D index for error storage (interior points only)
    int idx1d = (y_coord - 1) * (slabWidth - 2) + (x_local - 1);

    // Track maximum error
    double max_err = 0.0;

    // First interior value and boundary value
    double below = devGrid[base_idx];
    double current = devGrid[base_idx + 1];
    double new_val;

    // Process the entire column
    #pragma unroll 4
    for (int k = 1; k < nz-1; ++k) {
        // Calculate indices
        int above_idx = base_idx + (k+1);
        int current_idx = base_idx + k;

        // Calculate new value
        new_val = 0.5 * (below + devGrid[above_idx]);

        // Track maximum error
        double err = fabs(new_val - current);
        max_err = MAX(max_err, err);

        // Update grid value
        devGrid[current_idx] = new_val;

        // Prepare for next iteration
        below = new_val;
        current = devGrid[above_idx];
    }

    // Store maximum error for this thread's column
    devBlockMaxErr[idx1d] = max_err;
}

int main() {
    // Query available GPUs and ensure at least NUM_GPUS present.
    int devCount = 0;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&devCount));
    if (devCount < NUM_GPUS) {
        fprintf(stderr, "Need %d GPUs but found %d\n", NUM_GPUS, devCount);
        return EXIT_FAILURE;
    }
    printf("Using %d GPUs\n", NUM_GPUS);

    // Check peer-to-peer access capability between GPUs.
    int can01 = 0, can10 = 0;
    cudaDeviceCanAccessPeer(&can01, 0, 1);
    cudaDeviceCanAccessPeer(&can10, 1, 0);

    // Allocate and initialize the host grid.
    size_t N = (size_t)nx * ny * nz;
    double *hostGrid = (double*)malloc(N * sizeof(double));
    if (!hostGrid) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }
    init(hostGrid);

    // Divide domain along x into slabs for each GPU.
    int interior = nx - 2;
    int localWidth = interior / NUM_GPUS;
    int slabWidth = localWidth + 2;
    size_t slabBytes = (size_t)slabWidth * ny * nz * sizeof(double);
    size_t sliceBytes = (size_t)ny * nz * sizeof(double);

    // Allocate device buffers for each GPU.
    double *devGrid[NUM_GPUS], *devBlockMaxErr[NUM_GPUS];
    for (int d = 0; d < NUM_GPUS; ++d) {
        CHECK_CUDA_ERROR(cudaSetDevice(d));
        if ((d == 0 && can01) || (d == 1 && can10))
            cudaDeviceEnablePeerAccess(1 - d, 0);
        cudaMalloc(&devGrid[d], slabBytes);
        cudaMalloc(&devBlockMaxErr[d],
                   localWidth * (ny - 2) * sizeof(double));
    }

    // Copy initial slabs to respective GPUs.
    CHECK_CUDA_ERROR(cudaSetDevice(0));
    cudaMemcpy(devGrid[0], hostGrid, slabBytes, cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR(cudaSetDevice(1));
    cudaMemcpy(devGrid[1],
               hostGrid + (size_t)localWidth * ny * nz,
               slabBytes, cudaMemcpyHostToDevice);

    // Allocate pinned host buffer for fallback slice exchange.
    double *pinnedSlice;
    cudaMallocHost(&pinnedSlice, sliceBytes);

    // Wrap block-error arrays with Thrust pointers for reduction.
    thrust::device_ptr<double> thrustPtr[NUM_GPUS];
    for (int d = 0; d < NUM_GPUS; ++d) {
        CHECK_CUDA_ERROR(cudaSetDevice(d));
        thrustPtr[d] = thrust::device_pointer_cast(devBlockMaxErr[d]);
    }

    // Calculate total lines for each direction in a slab
    int total_yz_lines = ny * nz;
    int total_xz_lines = slabWidth * nz;  // Using slabWidth instead of nx
    int total_xy_lines = slabWidth * ny;  // Using slabWidth instead of nx

    // Calculate blocks needed for each direction
    int blocks_i = (total_yz_lines + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blocks_j = (total_xz_lines + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blocks_k = (total_xy_lines + BLOCK_SIZE - 1) / BLOCK_SIZE;

    double epsGlobal = 0.0;
    double localMaxErr[NUM_GPUS];
    volatile int converged = 0;

    double tStart = omp_get_wtime();

    #pragma omp parallel num_threads(NUM_GPUS) shared(converged, epsGlobal, localMaxErr)
    {
        int dev = omp_get_thread_num();
        CHECK_CUDA_ERROR(cudaSetDevice(dev));

        for (int iter = 1; iter <= ITMAX; ++iter) {
            #pragma omp master
            converged = 0;
            #pragma omp barrier

            // Launch optimized i‑sweep with one thread per yz-line
            sweep_i<<<blocks_i, BLOCK_SIZE>>>(devGrid[dev], slabWidth);
            CHECK_CUDA_ERROR(cudaGetLastError());
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            // Launch optimized j‑sweep with one thread per xz-line - now passing slabWidth
            sweep_j<<<blocks_j, BLOCK_SIZE>>>(devGrid[dev], slabWidth);
            CHECK_CUDA_ERROR(cudaGetLastError());
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            // Launch optimized k‑sweep with one thread per xy-line, computing errors
            sweep_k<<<blocks_k, BLOCK_SIZE>>>(devGrid[dev],
                                             devBlockMaxErr[dev],
                                             slabWidth);
            CHECK_CUDA_ERROR(cudaGetLastError());
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            #pragma omp master
            {
                // Exchange ghost slices between GPUs via P2P or fallback.
                if (can01 && can10) {
                    cudaMemcpyPeer(
                        devGrid[1] + 0 * ny * nz, 1,
                        devGrid[0] + localWidth * ny * nz, 0,
                        sliceBytes);
                    cudaMemcpyPeer(
                        devGrid[0] + (slabWidth - 1) * ny * nz, 0,
                        devGrid[1] + 1 * ny * nz, 1,
                        sliceBytes);
                } else {
                    cudaSetDevice(0);
                    cudaMemcpy(pinnedSlice,
                               devGrid[0] + localWidth * ny * nz,
                               sliceBytes, cudaMemcpyDeviceToHost);
                    cudaSetDevice(1);
                    cudaMemcpy(devGrid[1] + 0 * ny * nz,
                               pinnedSlice, sliceBytes, cudaMemcpyHostToDevice);
                    cudaMemcpy(pinnedSlice,
                               devGrid[1] + 1 * ny * nz,
                               sliceBytes, cudaMemcpyDeviceToHost);
                    cudaSetDevice(0);
                    cudaMemcpy(devGrid[0] + (slabWidth - 1) * ny * nz,
                               pinnedSlice, sliceBytes, cudaMemcpyHostToDevice);
                }
            }
            #pragma omp barrier

            // Reduce to find local maximum error on this GPU.
            thrust::device_ptr<double> itMax = thrust::max_element(
                thrust::device,
                thrustPtr[dev],
                thrustPtr[dev] + localWidth * (ny - 2));
            CHECK_CUDA_ERROR(cudaMemcpy(&localMaxErr[dev],
                                        itMax.get(),
                                        sizeof(double),
                                        cudaMemcpyDeviceToHost));

            #pragma omp barrier
            #pragma omp master
            {
                // Compute global convergence criterion and print.
                epsGlobal = fmax(localMaxErr[0], localMaxErr[1]);
                printf(" IT = %4d   EPS = %14.7E\n", iter, epsGlobal);
                if (epsGlobal < CONV_THRESH) converged = 1;
            }
            #pragma omp barrier
            if (converged) break;
        }
    }

    double tEnd = omp_get_wtime();

    // Print benchmark results.
    printf(" ADI Benchmark Completed.\n");
    printf(" Size            = %4d x %4d x %4d\n", nx, ny, nz);
    printf(" Time in seconds =       %12.2lf\n", tEnd - tStart);
    printf(" Operation type  =   double precision\n");
    // printf(" Verification    =       %12s\n", (fabs(epsGlobal - 0.9089673E+00) < 1e-6) ? "SUCCESSFUL" : "UNSUCCESSFUL");
    printf(" END OF ADI Benchmark\n");

    // Cleanup host and device resources.
    free(hostGrid);
    cudaFreeHost(pinnedSlice);
    for (int d = 0; d < NUM_GPUS; ++d) {
        CHECK_CUDA_ERROR(cudaSetDevice(d));
        cudaFree(devGrid[d]);
        cudaFree(devBlockMaxErr[d]);
    }
    return 0;
}