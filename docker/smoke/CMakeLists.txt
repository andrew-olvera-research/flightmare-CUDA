cmake_minimum_required(VERSION 3.24)
project(flightmare_cuda_smoke LANGUAGES CUDA)

enable_testing()
add_executable(cuda_smoke cuda_smoke.cu)
target_compile_features(cuda_smoke PRIVATE cuda_std_17)
add_test(NAME cuda_smoke COMMAND cuda_smoke)
