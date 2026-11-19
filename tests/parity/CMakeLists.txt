cmake_minimum_required(VERSION 3.24)
project(flightmare_parity_tests LANGUAGES C CXX)

enable_testing()
find_package(Eigen3 3.3.4 CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(yaml-cpp CONFIG REQUIRED)

get_filename_component(FLIGHTMARE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(FLIGHTLIB_USE_SYSTEM_DEPS ON CACHE INTERNAL "Use dev image dependencies")
set(ENABLE_FAST OFF CACHE INTERNAL "Preserve the CPU oracle's NaN checks")
set(BUILD_TESTS OFF CACHE INTERNAL "Build parity tests in this tree")
set(BUILD_UNITY_BRIDGE_TESTS OFF CACHE INTERNAL "Run the oracle headless")
add_subdirectory("${FLIGHTMARE_ROOT}/flightlib" flightlib EXCLUDE_FROM_ALL)

# Eigen's fixed-size types have architecture-dependent alignment. Match the
# flags used by the CPU library for every target that calls its C++ API.
get_directory_property(FLIGHTLIB_ARCH_FLAGS DIRECTORY "${FLIGHTMARE_ROOT}/flightlib"
  DEFINITION CMAKE_CXX_ARCH_FLAGS)
separate_arguments(FLIGHTLIB_ARCH_OPTIONS NATIVE_COMMAND "${FLIGHTLIB_ARCH_FLAGS}")

add_library(flightmare_parity_oracle STATIC oracle.cpp)
target_compile_features(flightmare_parity_oracle PUBLIC cxx_std_17)
target_compile_definitions(flightmare_parity_oracle PUBLIC EIGEN_STACK_ALLOCATION_LIMIT=1048576)
target_compile_options(flightmare_parity_oracle PUBLIC
  "$<$<CONFIG:Release>:${FLIGHTLIB_ARCH_OPTIONS}>")
target_include_directories(flightmare_parity_oracle PUBLIC
  "${FLIGHTMARE_ROOT}/flightlib/include" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(flightmare_parity_oracle PUBLIC
  flightlib Eigen3::Eigen OpenMP::OpenMP_CXX yaml-cpp::yaml-cpp)

add_executable(flightmare_parity_tests dynamics_test.cpp quadrotor_test.cpp env_test.cpp)
target_link_libraries(flightmare_parity_tests PRIVATE flightmare_parity_oracle GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(flightmare_parity_tests
  DISCOVERY_MODE PRE_TEST
  PROPERTIES ENVIRONMENT "FLIGHTMARE_PATH=${FLIGHTMARE_ROOT}")

add_executable(flightmare_oracle_trace trace_main.cpp)
target_link_libraries(flightmare_oracle_trace PRIVATE flightmare_parity_oracle)
