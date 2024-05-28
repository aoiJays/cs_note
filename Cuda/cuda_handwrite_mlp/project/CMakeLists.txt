cmake_minimum_required(VERSION 3.10)
project(MyCUDAProject CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)


file(GLOB_RECURSE CUDA_SOURCES src/*.cu)

include_directories(include)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2")

add_executable(app ${CUDA_SOURCES})
