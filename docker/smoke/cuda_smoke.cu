#include <cuda_runtime.h>

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

__global__ void fill(int* output, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    output[i] = 3 * i + 7;
  }
}

}  // namespace

int main() {
  try {
    int device;
    cudaDeviceProp properties{};
    check(cudaGetDevice(&device), "cudaGetDevice");
    check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");

    constexpr int count = 257;
    constexpr int threads = 128;
    std::array<int, count> result{};
    constexpr auto bytes = count * sizeof(int);
    int* output = nullptr;
    check(cudaMalloc(&output, bytes), "cudaMalloc");
    fill<<<(count + threads - 1) / threads, threads>>>(output, count);
    check(cudaGetLastError(), "kernel launch");
    check(cudaMemcpy(result.data(), output, bytes, cudaMemcpyDeviceToHost),
          "cudaMemcpy");
    check(cudaFree(output), "cudaFree");

    for (int i = 0; i < count; ++i) {
      if (result[i] != 3 * i + 7) {
        throw std::runtime_error("Incorrect result at index " + std::to_string(i));
      }
    }
    std::cout << properties.name << " (compute capability " << properties.major
              << '.' << properties.minor << ") computed " << count << " values\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
