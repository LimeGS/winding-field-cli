#include <cuda_runtime.h>
#include "predicates.h"

namespace {
double* device_rows = nullptr;
int8_t* device_output = nullptr;
int allocated_capacity = 0;

__global__ void numeric_probe(int8_t* result) { *result = wf_numeric_probe() ? 1 : 0; }
__global__ void filter_kernel(int mode, int n, const double* rows, int8_t* output) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n) output[i] = wf_interval_filter(mode, rows+WF_COLUMNS*i);
}
int fail(int error) {
    wf_cuda_close();
    return error;
}
}  // namespace

extern "C" int wf_cuda_memory(uint64_t* free_bytes, uint64_t* total_bytes) {
    if (!free_bytes || !total_bytes) return -1;
    size_t available, total;
    cudaError_t error = cudaMemGetInfo(&available, &total);
    if (error != cudaSuccess) return int(error);
    *free_bytes = available;
    *total_bytes = total;
    return 0;
}

extern "C" int wf_cuda_close() {
    cudaError_t first = cudaSuccess, second = cudaSuccess;
    if (device_rows) first = cudaFree(device_rows);
    if (device_output) second = cudaFree(device_output);
    device_rows = nullptr;
    device_output = nullptr;
    allocated_capacity = 0;
    return int(first != cudaSuccess ? first : second);
}

extern "C" int wf_cuda_init(int capacity, uint64_t reserve_bytes) {
    if (capacity < 1 || capacity > WF_MAX_ROWS || device_rows || device_output) return -1;
    if (!wf_host_environment()) return -3;
    uint64_t available, total;
    int status = wf_cuda_memory(&available, &total);
    if (status) return status;
    uint64_t needed = uint64_t(capacity)*(WF_COLUMNS*sizeof(double)+sizeof(int8_t));
    if (available < reserve_bytes || needed > available-reserve_bytes) return -2;
    cudaError_t error = cudaMalloc(&device_rows, size_t(capacity)*WF_COLUMNS*sizeof(double));
    if (error != cudaSuccess) return fail(int(error));
    error = cudaMalloc(&device_output, size_t(capacity)*sizeof(int8_t));
    if (error != cudaSuccess) return fail(int(error));
    numeric_probe<<<1, 1>>>(device_output);
    error = cudaGetLastError();
    if (error != cudaSuccess) return fail(int(error));
    int8_t numeric_ok = 0;
    error = cudaMemcpy(&numeric_ok, device_output, sizeof(numeric_ok), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) return fail(int(error));
    if (numeric_ok != 1) return fail(-3);
    status = wf_cuda_memory(&available, &total);
    if (status) return fail(status);
    if (available < reserve_bytes) return fail(-2);
    allocated_capacity = capacity;
    return 0;
}

extern "C" int wf_cuda_filter(int mode, int n, const double* rows, int8_t* output) {
    if (mode < 0 || mode > 1 || n < 1 || n > allocated_capacity || !rows || !output) return -1;
    cudaError_t error = cudaMemcpy(device_rows, rows, size_t(n)*WF_COLUMNS*sizeof(double), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) return fail(int(error));
    filter_kernel<<<(n+127)/128, 128>>>(mode, n, device_rows, device_output);
    error = cudaGetLastError();
    if (error != cudaSuccess) return fail(int(error));
    error = cudaMemcpy(output, device_output, size_t(n)*sizeof(int8_t), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) return fail(int(error));
    return 0;
}
