cmake_minimum_required(VERSION 3.18)
project(netsentinel_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(NETSENTINEL_BUILD_PYTHON "Build the pybind11 module" ON)
option(NETSENTINEL_BUILD_TESTS "Build the GoogleTest unit tests" ON)
option(NETSENTINEL_WERROR "Treat compiler warnings as errors (used in CI)" OFF)

# Core library, no Python dependency
add_library(netsentinel_core STATIC src/baseline.cpp)
target_include_directories(netsentinel_core PUBLIC include)
set_target_properties(netsentinel_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(MSVC)
  target_compile_options(netsentinel_core PRIVATE /W4)
else()
  target_compile_options(netsentinel_core PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
endif()
if(NETSENTINEL_WERROR AND NOT MSVC)
  target_compile_options(netsentinel_core PRIVATE -Werror)
endif()

# Python module, written to the repository root so `import _netsentinel_cpp` works
if(NETSENTINEL_BUILD_PYTHON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_netsentinel_cpp src/bindings.cpp)
  target_link_libraries(_netsentinel_cpp PRIVATE netsentinel_core)
  set_target_properties(_netsentinel_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/.."
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${PROJECT_SOURCE_DIR}/.."
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${PROJECT_SOURCE_DIR}/..")
endif()

# Unit tests
if(NETSENTINEL_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  add_executable(baseline_test tests/test_baseline.cpp)
  target_link_libraries(baseline_test PRIVATE netsentinel_core GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(baseline_test)
endif()
