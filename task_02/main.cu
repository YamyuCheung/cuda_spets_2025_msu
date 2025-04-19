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

#define nx 384
#define ny 384
#define nz 384
#define BLOCK_SIZE   32         // threads per block for compute sweeps
#define NUM_GPUS     2          // use 2 GPUs
#define ITMAX        100
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


// Perform Gauss–Seidel sweep in the i‑direction on each slab (with ghost planes).
__global__ void sweep_i(double *devGrid, int slabWidth) {
    int j = blockIdx.x + 1;
    int k = blockIdx.y * blockDim.x + threadIdx.x + 1;
    if (j >= ny - 1 || k >= nz - 1) return;
    int stride = ny * nz;
    double *row = devGrid + j * nz + k;
    double left = row[0];
    for (int i = 1; i < slabWidth - 1; ++i) {
        double right = row[(i + 1) * stride];
        double newv = 0.5 * (left + right);
        row[i * stride] = newv;
        left = newv;
    }
}


// Perform sweep in the j‑direction .
__global__ void sweep_j(double *devGrid) {
    int i = blockIdx.x + 1;
    int k = blockIdx.y * blockDim.x + threadIdx.x + 1;
    if (i >= nx - 1 || k >= nz - 1) return;
    int stride_i = ny * nz;
    int stride_j = nz;
    double *col = devGrid + i * stride_i + k;
    double up = col[0];
    for (int j = 1; j < ny - 1; ++j) {
        double down = col[(j + 1) * stride_j];
        double newv = 0.5 * (up + down);
        col[j * stride_j] = newv;
        up = newv;
    }
}


// Perform sweep in the k‑direction and record maximum local error per block.
__global__ void sweep_k(double *devGrid, double *devBlockMaxErr, int slabWidth) {
    if (threadIdx.x != 0) return;  // only one thread per (i,j) column does work
    int i_loc = blockIdx.x + 1;
    int j_loc = blockIdx.y + 1;
    if (i_loc > slabWidth || j_loc >= ny - 1) return;
    int idx1d = (j_loc - 1) * slabWidth + (i_loc - 1);
    int stride = ny * nz;
    double *col = devGrid + i_loc * stride + j_loc * nz;
    double maxErr = 0.0;
    for (int k = 1; k < nz - 1; ++k) {
        double oldv = col[k];
        double newv = 0.5 * (col[k - 1] + col[k + 1]);
        col[k] = newv;
        double err = fabs(newv - oldv);
        if (err > maxErr) maxErr = err;
    }
    devBlockMaxErr[idx1d] = maxErr;
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

    // Configure kernel launch dimensions.
    dim3 threads(BLOCK_SIZE);
    dim3 gridI(ny - 2, (nz - 2 + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 gridJ(localWidth, (nz - 2 + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 gridK(localWidth, ny - 2);

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

            // Launch i‑sweep.
            sweep_i<<<gridI, threads>>>(devGrid[dev], slabWidth);
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            // Launch j‑sweep.
            sweep_j<<<gridJ, threads>>>(devGrid[dev]);
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            // Launch k‑sweep and compute block-wise errors.
            sweep_k<<<gridK, threads>>>(devGrid[dev],
                                        devBlockMaxErr[dev],
                                        localWidth);
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
    printf(" Verification    =       %12s\n", (fabs(epsGlobal - 0.07249074) < 1e-6) ? "SUCCESSFUL" : "UNSUCCESSFUL");
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
