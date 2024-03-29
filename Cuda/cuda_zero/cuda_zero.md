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

