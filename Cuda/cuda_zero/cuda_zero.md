# cuda编程 · 零

> [蒙特卡洛的树 - Cuda编程Bilibili](https://www.bilibili.com/video/BV17K411K76C/)

[TOC]

## 基本步骤

在进行运行之前，我们可以先查询一下设备中有多少块GPU

```cpp
    int gpuCount = -1;
    cudaGetDeviceCount(&gpuCount);
    printf("%d ", gpuCount);
```

然后可以设置成最后一块显卡的ID

`cudaGetDevice`可以得到当前正在使用的gpu

```cpp
    int gpuCount = -1;
    cudaGetDeviceCount(&gpuCount);
    printf("gpuCount = %d\n", gpuCount);

    // 1. 指定GPU设别
    // 单GPU设备其实可以省略此步骤
    cudaSetDevice(gpuCount - 1);

    int devideId = -1;
    cudaGetDevice(&devideId);
    printf("gpu = %d\n", devideId);
```

当设置不存在的设备编号时，默认启动0号gpu



基本步骤如下：

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>

__global__ void kernel(float *a) {
    a[threadIdx.x] = 1;
}

int main(int argc, char ** argv) {

    int gpuCount = -1;
    cudaGetDeviceCount(&gpuCount);
    printf("gpuCount = %d\n", gpuCount);

    // 1. 指定GPU设别
    // 单GPU设备其实可以省略此步骤
    cudaSetDevice(gpuCount - 1);

    int devideId = -1;
    cudaGetDevice(&devideId);
    printf("gpu = %d\n", devideId);

    // 2. 分配显存空间
    float *aGPU;
    // cudaError_t cudaMalloc(void **devPtr, size_t size);
    // void **devPtr 指向待分配内存空间指针的指针 
    // 		指针是通用的设备指针，可以指向任何类型的内存
    // size_t size 分配的内存大小
    cudaMalloc((void**)&aGPU, 16 * sizeof(float));

    // 3. 分配内存空间
    float a[16] = {0};

    // 4. 内存->显存
    // cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)
    // 目的地址, 源地址，需要复制的字节数量， 复制类型
        //  cudaMemcpyHostToHost：从主机到主机的内存复制。
        //  cudaMemcpyHostToDevice：从主机到设备的内存复制。
        //  cudaMemcpyDeviceToHost：从设备到主机的内存复制。
        //  cudaMemcpyDeviceToDevice：从设备到设备的内存复制。

    cudaMemcpy(aGPU, a, 16 * sizeof(float), cudaMemcpyHostToDevice);

    // 5. 设备代码
    kernel<<<1, 16>>>(aGPU);
    
    // 6. 显存->内存
    cudaMemcpy(a, aGPU, 16 * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i=0;i<16;++i)
        printf("%.2lf ", a[i]);

    // 7. 释放
    cudaFree(aGPU); // 释放申请的显存
    cudaDeviceReset(); // 重置设备
    // 如果主机内存也有申请 也需要释放
}
```

#### GPU详细信息

`cudaDeviceProp`是cuda封装的一个显卡信息结构体

我们可以通过这个结构体查看显卡信息

```cpp
    cudaDeviceProp prop;
    // 指定0号显卡
    cudaGetDeviceProperties(&prop, 0);

    printf("maxThreadsPerBLOCK: %d\n", prop.maxThreadsPerBlock);
    printf("maxThreadsDim: %d\n", prop.maxThreadsDim[0]);
    printf("maxGridSize: %d\n", prop.maxGridSize[0]);
    printf("totalConstMem: %d\n", prop.totalConstMem);
    printf("clockRate: %d\n", prop.clockRate);
    printf("integrated: %d\n", prop.integrated);    

```

还有一些别的东西

```cpp
// 程序可以在多个 CUDA 设备上运行时，可以使用这个函数来选择一个最合适的设备 device会变成被选中的设备编号 
// prop需要填写需求 自动匹配符合要求的设备
cudaError_t cudaChooseDevice(int* device, const cudaDeviceProp* prop)

// 传入一个编号数组和数组长度
// 只有编号在其中的设备会是有效设备
cudaError_t cudaSetValidDevices(int *device_arr, int len);
```

## Cuda项目建立

建立项目文件夹，新建`CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.22)
project(app LANGUAGES CUDA CXX)
find_package(CUDA REQUIRED)
CUDA_ADD_EXECUTABLE(app main.cu)
TARGET_LINK_LIBRARIES(app)
```

在同文件夹下建立一个`main.cu`

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>

__global__ void add(int *a, int *b, int *c, int num) {
	if ( threadIdx.x < num ) 
		c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

int main(int argc, char ** argv) {
	
	int num = 10;
	int a[num], b[num], c[num];
	
	for (int i=0;i<num;++i) a[i] = i;
	for (int i=0;i<num;++i) b[i] = i * i;

	int *agpu, *bgpu, *cgpu;
	
	cudaMalloc((void**)&agpu, num * sizeof(int));
	cudaMalloc((void**)&bgpu, num * sizeof(int));
	cudaMalloc((void**)&cgpu, num * sizeof(int));

	cudaMemcpy(agpu, a, num * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(bgpu, b, num * sizeof(int), cudaMemcpyHostToDevice);

	// 加法
	add<<<1, 10>>>(agpu, bgpu, cgpu, num);
	cudaMemcpy(c, cgpu, num * sizeof(int), cudaMemcpyDeviceToHost);
	
	printf("add:\n");
	for (int i=0;i<num;++i) printf("%d + %d = %d\n", a[i], b[i], c[i]);

	cudaFree(agpu); 
	cudaFree(bgpu); 
	cudaFree(cgpu); 
    cudaDeviceReset(); 
}
```

新建`build`文件夹

```bash
mkdir buid && cd build
cmake ..
make -j3
./app
```

## 手写卷积

什么是卷积？[【官方双语】那么……什么是卷积？](https://www.bilibili.com/video/BV1Vd4y1e7pj)

 首先需要添加一个新的东西：`CUDA_CHECK`

```cpp
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                __FILE__, __LINE__, err, cudaGetErrorString(err), #call); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// 后续我们使用Cuda函数时 用宏进行包装
// 即可及时报错
CUDA_CHECK(cudaMalloc(&devPtr, size));

```

code见`code/src/code_2.cu`
