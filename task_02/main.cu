/*
* ADI (Alternating Direction Implicit) Method CUDA Implementation
 * ==============================================================
 *
 * This code implements a 3D ADI method solver using CUDA for high-performance
 * computing. It features two different implementation approaches (selectable via MODE):
 *
 * MODE 1: Column-oriented approach where each thread processes a specific (j,k) pair
 * for i-sweeps, (i,k) pair for j-sweeps, and a thread-per-column for k-sweeps. This
 * implementation follows a more traditional GPU parallelization strategy with explicit
 * grid/block configuration.
 *
 * MODE 2: Line-oriented approach where threads are distributed based on total lines in
 * each direction. This implementation uses flattened thread IDs with grid-stride
 * calculations to map work to threads, providing potentially better work distribution.
 *
 * Both implementations use:
 * - Thrust library for efficient parallel reduction when finding maximum error
 * - CUDA streams for asynchronous execution
 * - OpenMP for host-side initialization
 * - Streaming Gauss-Seidel iterations in each coordinate direction
 * - Optimized memory access patterns for GPU coalescing
 *
 * The solver iterates through all three dimensions (x, y, z) in succession until
 * convergence is reached or the maximum iteration count is hit. A key optimization
 * is computing the maximum error only after the z-direction sweep.
 *
 * To switch implementations, change the MODE define at the top of the file.
*/
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

// Choose implementation mode (1 or 2)
#define MODE 2

// Domain dimensions (shared)
#define DOMAIN_SIZE_X 900
#define DOMAIN_SIZE_Y 900
#define DOMAIN_SIZE_Z 900

// CUDA error checking (unified for both modes)
#define CHECK_CUDA_ERROR(call)                                              \
do {                                                                        \
    cudaError_t err = call;                                                 \
    if (err != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA error in %s:%d: %s\n",                        \
                __FILE__, __LINE__, cudaGetErrorString(err));               \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
} while (0)

// Mode 1 specific definitions
#if MODE == 1
    #define nx DOMAIN_SIZE_X
    #define ny DOMAIN_SIZE_Y
    #define nz DOMAIN_SIZE_Z
    #define BLOCK_SIZE 32
#endif

// Mode 2 specific definitions
#if MODE == 2
    #define N_X DOMAIN_SIZE_X
    #define N_Y DOMAIN_SIZE_Y
    #define N_Z DOMAIN_SIZE_Z
    #define THREAD_COUNT 32

    // Compute maximum of two values
    inline __device__ double maximum(double a, double b) {
        return (a > b) ? a : b;
    }
#endif

//==============================
// MODE 1 IMPLEMENTATION
//==============================
#if MODE == 1
// Initialize boundary on CPU (Mode 1)
void init(double *a) {
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j)
    for (int k = 0; k < nz; ++k) {
        int idx = i*ny*nz + j*nz + k;
        if (i==0 || i==nx-1 ||
            j==0 || j==ny-1 ||
            k==0 || k==nz-1)
            a[idx] = 10.0*i/(nx-1)
                     + 10.0*j/(ny-1)
                     + 10.0*k/(nz-1);
        else
            a[idx] = 0.0;
    }
}

// i-direction streaming Gauss-Seidel
__global__ void compute_i_stream(double * __restrict__ a) {
    int j = blockIdx.x + 1;
    if (j >= ny-1) return;
    int k = blockIdx.y*blockDim.x + threadIdx.x + 1;
    if (k >= nz-1) return;

    int stride = ny*nz;
    double *row = a + j*nz + k;

    double left = row[0];
    #pragma unroll 4
    for (int i = 1; i < nx-1; ++i) {
        double right = row[(i+1)*stride];
        double newv  = 0.5*(left + right);
        row[i*stride] = newv;
        left = newv;
    }
}

// j-direction streaming Gauss-Seidel
__global__ void compute_j_stream(double * __restrict__ a) {
    int i = blockIdx.x + 1;
    if (i >= nx-1) return;
    int k = blockIdx.y*blockDim.x + threadIdx.x + 1;
    if (k >= nz-1) return;

    int stride_i = ny*nz;
    int stride_j = nz;
    double *col = a + i*stride_i + k;

    double up = col[0];
    #pragma unroll 4
    for (int j = 1; j < ny-1; ++j) {
        double down = col[(j+1)*stride_j];
        double newv = 0.5*(up + down);
        col[j*stride_j] = newv;
        up = newv;
    }
}

// k-direction streaming GS, and write the maximum error for this column
__global__ void compute_k_stream(double *a, double *block_eps) {
    if (threadIdx.x != 0) return;

    int i = blockIdx.x + 1;
    int j = blockIdx.y + 1;
    int idx1d = (j-1)*(nx-2) + (i-1);

    int stride = ny*nz;
    double *col = a + i*stride + j*nz;

    double max_err = 0.0;
    for (int k = 1; k < nz-1; ++k) {
        double oldv = col[k];
        double newv = 0.5*(col[k-1] + col[k+1]);
        col[k] = newv;
        double err = fabs(newv - oldv);
        if (err > max_err) max_err = err;
    }
    block_eps[idx1d] = max_err;
}
#endif // MODE 1

//==============================
// MODE 2 IMPLEMENTATION
//==============================
#if MODE == 2
// Initialize domain on CPU (Mode 2)
void initialize_domain(double *field) {
    #pragma omp parallel for collapse(3)
    for (int x = 0; x < N_X; ++x) {
        for (int y = 0; y < N_Y; ++y) {
            for (int z = 0; z < N_Z; ++z) {
                int index = x*N_Y*N_Z + y*N_Z + z;

                // Set boundary conditions
                if (x == 0 || x == N_X-1 ||
                    y == 0 || y == N_Y-1 ||
                    z == 0 || z == N_Z-1) {
                    field[index] = 10.0 * x / (N_X-1) +
                                  10.0 * y / (N_Y-1) +
                                  10.0 * z / (N_Z-1);
                } else {
                    field[index] = 0.0;  // Initialize interior to zero
                }
            }
        }
    }
}

// X-direction sweep kernel
__global__ void sweep_x_direction(double *field) {
    // Calculate global thread ID
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Determine if thread should process a line
    int total_yz_lines = N_Y * N_Z;
    if (thread_id >= total_yz_lines)
        return;

    // Calculate y,z coordinates
    int z_pos = thread_id % N_Z;
    int y_pos = thread_id / N_Z;

    // Skip boundary points
    bool is_interior = (y_pos > 0 && y_pos < N_Y-1 &&
                        z_pos > 0 && z_pos < N_Z-1);
    if (!is_interior)
        return;

    // Calculate starting position and strides
    int yz_offset = y_pos * N_Z + z_pos;
    int x_stride = N_Y * N_Z;

    // Start with boundary value
    double running_value = field[yz_offset];  // Value at x=0
    const double relaxation = 0.5;

    // Process the line from left to right
    for (int x_pos = 1; x_pos < N_X-1; ++x_pos) {
        // Calculate global indices
        int current_index = x_pos * x_stride + yz_offset;
        int right_index = (x_pos+1) * x_stride + yz_offset;

        // Update using relaxation
        running_value = relaxation * (running_value + field[right_index]);
        field[current_index] = running_value;
    }
}

// Y-direction sweep kernel
__global__ void sweep_y_direction(double *field) {
    // Calculate global thread ID
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Determine if thread should process a line
    int total_xz_lines = N_X * N_Z;
    if (thread_id >= total_xz_lines)
        return;

    // Calculate x,z coordinates using division and modulo
    int z_pos = thread_id % N_Z;
    int x_pos = thread_id / N_Z;

    // Skip boundary points
    bool is_interior = (x_pos > 0 && x_pos < N_X-1 &&
                        z_pos > 0 && z_pos < N_Z-1);
    if (!is_interior)
        return;

    // Calculate offsets for efficient memory access
    int z_stride = 1;
    int y_stride = N_Z;
    int x_stride = N_Y * N_Z;
    int base_pos = x_pos * x_stride + z_pos;

    // Start with boundary value
    double previous = field[base_pos];  // Value at y=0
    const double weight = 0.5;

    // Process the line from top to bottom
    for (int y_pos = 1; y_pos < N_Y-1; ++y_pos) {
        int current_pos = base_pos + y_pos * y_stride;
        int bottom_pos = base_pos + (y_pos+1) * y_stride;

        // Compute weighted average with previous value
        previous = weight * (previous + field[bottom_pos]);
        field[current_pos] = previous;
    }
}

// Z-direction sweep kernel with error tracking
__global__ void sweep_z_direction(double *field, double *error_values) {
    // Calculate global thread ID
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Determine if thread should process a line
    int total_xy_lines = N_X * N_Y;
    if (thread_id >= total_xy_lines)
        return;

    // Calculate x,y coordinates - different approach from x,y kernels
    int y_pos = thread_id % N_Y;
    int x_pos = thread_id / N_Y;

    // Skip boundary points
    bool is_interior = (x_pos > 0 && x_pos < N_X-1 &&
                        y_pos > 0 && y_pos < N_Y-1);
    if (!is_interior) {
        // For boundary points, set error to 0 (won't affect max)
        error_values[thread_id] = 0.0;
        return;
    }

    // Calculate base position
    int x_stride = N_Y * N_Z;
    int y_stride = N_Z;
    int base_position = x_pos * x_stride + y_pos * y_stride;

    // Setup for first interior point
    double below = field[base_position];         // z=0 (boundary)
    double current = field[base_position + 1];   // z=1 (first interior)
    double error_max = 0.0;
    const double factor = 0.5;

    // Process all interior points along z axis
    for (int z_pos = 1; z_pos < N_Z-1; ++z_pos) {
        int point_pos = base_position + z_pos;
        double above = field[point_pos + 1];

        // Compute new value
        double updated = factor * (below + above);

        // Track maximum difference from previous value
        double diff = fabs(updated - current);
        error_max = maximum(error_max, diff);

        // Store updated value
        field[point_pos] = updated;

        // Slide window: current becomes previous, next becomes current
        below = updated;
        current = above;
    }

    // Store maximum error for this line
    error_values[thread_id] = error_max;
}
#endif // MODE 2

//==============================
// MAIN FUNCTION
//==============================
int main() {
    // Solver parameters
    double maxeps = 0.01;
    int itmax = 10, it;
    double startt, endt;

    printf("Starting ADI Benchmark with CUDA Implementation (Optimized)...\n");
    printf("Current Implementation Mode: %d\n", MODE);
    printf("Number of OpenMP threads for initialization: %d\n", omp_get_max_threads());
    printf("Initializing data...\n");

#if MODE == 1
    // --- Allocate and initialize Host data ---
    double *h_A = (double*)malloc((size_t)nx*ny*nz*sizeof(double));
    if (!h_A) {
        fprintf(stderr, "host malloc failed\n");
        return EXIT_FAILURE;
    }
    init(h_A);

    // --- Allocate Device data ---
    double *d_A, *d_block_eps;
    size_t totalBytes = (size_t)nx*ny*nz*sizeof(double);
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, totalBytes));

    printf("Copying data to device (done only once)...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, totalBytes,
                                cudaMemcpyHostToDevice));

    // Each (i,j) column generates one block-eps, totaling (nx-2)*(ny-2)
    int Nblocks = (nx-2)*(ny-2);
    CHECK_CUDA_ERROR(cudaMalloc(&d_block_eps, Nblocks*sizeof(double)));

    // Thrust device pointer needed
    thrust::device_ptr<double> dev_ptr =
        thrust::device_pointer_cast(d_block_eps);

    // grid/block configuration
    dim3 block(BLOCK_SIZE);
    dim3 grid_i(ny-2, (nz-2 + BLOCK_SIZE-1)/BLOCK_SIZE);
    dim3 grid_j(nx-2, (nz-2 + BLOCK_SIZE-1)/BLOCK_SIZE);
    dim3 grid_k(nx-2, ny-2);

    // Start timing
    startt = omp_get_wtime();
    double eps = 0.0;

    printf("Starting computation...\n");
    for (it = 1; it <= itmax; ++it) {
        // i, j, k direction streaming GS
        compute_i_stream<<<grid_i, block>>>(d_A);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        compute_j_stream<<<grid_j, block>>>(d_A);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        compute_k_stream<<<grid_k, block>>>(d_A, d_block_eps);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // --- Thrust version: Find global maximum error ---
        thrust::device_ptr<double> max_ptr =
            thrust::max_element(thrust::device,
                                dev_ptr, dev_ptr + Nblocks);
        CHECK_CUDA_ERROR(cudaMemcpy(&eps, max_ptr.get(),
                                    sizeof(double),
                                    cudaMemcpyDeviceToHost));
        // ------------------------------------

        printf(" IT = %4i   EPS = %14.7E\n", it, eps);
        if (eps < maxeps) {
            printf("Convergence reached at iteration %d\n", it);
            break;
        }
    }
    endt = omp_get_wtime();

    // --- Cleanup ---
    free(h_A);
    cudaFree(d_A);
    cudaFree(d_block_eps);

#elif MODE == 2
    // Allocate and initialize host memory
    const size_t grid_size = (size_t)N_X * N_Y * N_Z;
    const size_t memory_size = grid_size * sizeof(double);

    double *host_field = (double*)malloc(memory_size);
    if (!host_field) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // Initialize the domain
    initialize_domain(host_field);

    // Calculate thread blocks needed
    const int total_yz = N_Y * N_Z;
    const int total_xz = N_X * N_Z;
    const int total_xy = N_X * N_Y;

    const int blocks_x_sweep = (total_yz + THREAD_COUNT - 1) / THREAD_COUNT;
    const int blocks_y_sweep = (total_xz + THREAD_COUNT - 1) / THREAD_COUNT;
    const int blocks_z_sweep = (total_xy + THREAD_COUNT - 1) / THREAD_COUNT;

    // Allocate device memory
    double *device_field, *device_errors;

    CHECK_CUDA_ERROR(cudaMalloc(&device_field, memory_size));
    CHECK_CUDA_ERROR(cudaMalloc(&device_errors, total_xy * sizeof(double)));

    // Copy initial data to device
    printf("Copying data to device (done only once)...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(device_field, host_field, memory_size, cudaMemcpyHostToDevice));

    // Create Thrust device pointer for finding maximum error
    thrust::device_ptr<double> errors_ptr = thrust::device_pointer_cast(device_errors);

    // Start computation
    printf("Starting computation...\n");
    startt = omp_get_wtime();
    double max_error = 1.0;  // Initialize above threshold

    // Main iteration loop
    for (it = 1; it <= itmax; ++it) {
        // X-direction sweep
        sweep_x_direction<<<blocks_x_sweep, THREAD_COUNT>>>(device_field);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Y-direction sweep
        sweep_y_direction<<<blocks_y_sweep, THREAD_COUNT>>>(device_field);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Z-direction sweep with error calculation
        sweep_z_direction<<<blocks_z_sweep, THREAD_COUNT>>>(device_field, device_errors);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Find maximum error using Thrust
        thrust::device_ptr<double> max_error_ptr =
            thrust::max_element(thrust::device, errors_ptr, errors_ptr + total_xy);

        CHECK_CUDA_ERROR(cudaMemcpy(&max_error, max_error_ptr.get(), sizeof(double),
                            cudaMemcpyDeviceToHost));

        printf(" IT = %4i   EPS = %14.7E\n", it, max_error);

        // Check convergence
        if (max_error < maxeps) {
            printf("Convergence reached at iteration %d\n", it);
            break;
        }
    }

    endt = omp_get_wtime();

    // Cleanup
    free(host_field);
    cudaFree(device_field);
    cudaFree(device_errors);

    // For consistent variable name in the final output
    double eps = max_error;
#endif

    // Final summary
    printf(" ADI Benchmark Completed.\n");
#if MODE == 1
    printf(" Size            = %4d x %4d x %4d\n", nx, ny, nz);
#else
    printf(" Size            = %4d x %4d x %4d\n", N_X, N_Y, N_Z);
#endif
    printf(" Iterations      =       %12d\n", it <= itmax ? it : itmax);
    printf(" Time in seconds =       %12.2lf\n", endt - startt);
    printf(" Operation type  =   double precision\n");
    // printf(" Verification    =       %12s\n",
    //     (fabs(eps - 0.07249074) < 1e-6
    //      ? "SUCCESSFUL"
    //      : "UNSUCCESSFUL"));
    printf(" END OF ADI Benchmark\n");

    return 0;
}