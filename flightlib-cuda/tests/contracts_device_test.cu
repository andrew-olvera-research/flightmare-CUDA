#include "flightlib_cuda/contracts.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using namespace flightlib_cuda::v1;

__global__ void writeLanes(Matrix<Scalar> state, BatchShape batch) {
  const Index n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= batch.lanes) return;
  state(static_cast<Index>(StateField::Position), n) = Scalar(batch.environment(n));
  state(static_cast<Index>(StateField::AccelerationBias) + 2, n) = Scalar(n);
}

#define CUDA_CHECK(call) do { const auto error = (call); if (error != cudaSuccess) { \
  std::cerr << #call << ": " << cudaGetErrorString(error) << '\n'; return 1; \
} } while (false)

int main() {
  const auto batch = makeBatchShape(7, 5);
  const auto bytes = checkedBytes<Scalar>(Offset{kStateSize} * batch.lanes);
  Scalar* data = nullptr;
  CUDA_CHECK(cudaMalloc(&data, bytes));
  CUDA_CHECK(cudaMemset(data, 0, bytes));
  writeLanes<<<1, 64>>>(Matrix<Scalar>{data, {25, batch.lanes}}, batch);
  CUDA_CHECK(cudaGetLastError());
  std::vector<Scalar> result(25 * batch.lanes);
  CUDA_CHECK(cudaMemcpy(result.data(), data, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(data));
  for (Index n = 0; n < batch.lanes; ++n) {
    if (result[n] != Scalar(n / 5) || result[24 * batch.lanes + n] != Scalar(n))
      return 1;
  }
  // Untouched rows detect addressing/stride mistakes.
  for (Index row = 1; row < 24; ++row)
    for (Index n = 0; n < batch.lanes; ++n)
      if (result[row * batch.lanes + n] != 0) return 1;
  return 0;
}
