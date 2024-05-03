cmake_minimum_required(VERSION 3.10)
project(MLP)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# 设置编译选项为-O2
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
# 添加include文件夹到头文件搜索路径
include_directories(include)

# 获取src文件夹中的所有源文件
file(GLOB SOURCES "src/*.cpp")

# 生成可执行文件app
add_executable(app ${SOURCES})
