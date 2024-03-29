cmake_minimum_required(VERSION 3.22)
project(CudaChatpter1 LANGUAGES CUDA)

# 添加CUDA源文件
set(SOURCE_FILES
    src/code_0.cu
)

# 添加可执行文件
add_executable(app ${SOURCE_FILES})

# 设置CUDA特定的编译选项
set_property(TARGET app PROPERTY CUDA_SEPARABLE_COMPILATION ON)
