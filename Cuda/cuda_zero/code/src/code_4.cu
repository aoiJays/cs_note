#include <stdio.h>
#include <sys/time.h>

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                __FILE__, __LINE__, err, cudaGetErrorString(err), #call); \
        exit(EXIT_FAILURE); \
    } \
} while (0)


__global__ void gpu_conv(float * img, float * kernel, float * res, int w, int h, int kernel_size) {
	
	// 边缘检测
	if ( threadIdx.x + blockIdx.x * blockDim.x >= w * h ) return;

	int x = (threadIdx.x + blockIdx.x * blockDim.x) / w;
	int y = (threadIdx.x + blockIdx.x * blockDim.x) % w;

	res[x * w + y] = 0;
	
	for (int i=0;i<kernel_size;++i)
		for (int j=0;j<kernel_size;++j) {
			int curx = x - kernel_size/2 + i;
			int cury = y - kernel_size/2 + j;
			if ( curx < 0 || curx >= h || cury < 0 || cury >= w ) continue;
			res[x * w + y] += kernel[i * kernel_size + j] * img[curx * w + cury]; 
		}	
}

void cpu_conv(float * img, float * kernel, float * res, int w, int h, int kernel_size) {
	for (int x=0;x<h;++x)
		for (int y=0;y<w;++y) {
	
			res[x * w + y] = 0;
			for (int i=0;i<kernel_size;++i)
				for (int j=0;j<kernel_size;++j) {
					int curx = x - kernel_size/2 + i;
					int cury = y - kernel_size/2 + j;
					if ( curx < 0 || curx >= h || cury < 0 || cury >= w ) continue;
					res[x * w + y] += kernel[i * kernel_size + j] * img[curx * w + cury]; 
				}	
		}
}

int main() {

	// 定义图像
	int width = 1920, height = 1080;
	float *img = new float[width * height];
	float *res = new float[width * height];
	
	for (int i=0;i<height;++i)
		for (int j=0;j<width;++j)
			img[i * width + j] = (i + j) % 256;
	
	// 定义卷积核
	int kernel_size = 3;
	float *kernel = new float[kernel_size * kernel_size];
	for (int i=0;i<kernel_size;++i)
		for (int j=0;j<kernel_size;++j)
			kernel[i * kernel_size + j] = j-1;

	// 申请显存
	float * img_gpu, *kernel_gpu, *res_gpu;
	CUDA_CHECK(cudaMalloc((void**)&img_gpu, width*height*sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&kernel_gpu, kernel_size*kernel_size*sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&res_gpu, width*height*sizeof(float)));

	// 内存->显存
	CUDA_CHECK(cudaMemcpy(img_gpu, img, width*height*sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(kernel_gpu, kernel, kernel_size*kernel_size*sizeof(float), cudaMemcpyHostToDevice));

	auto getMaxThreadCount = [&](int deviceID) {
		cudaDeviceProp prop;
		CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceID));
		printf("maxGridSize: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
		return prop.maxThreadsPerBlock;
	};

	int threadNum = getMaxThreadCount(0);	
	int blockNum = ( width * height + threadNum - 1 ) / threadNum; // 向上取整


	timeval startTime, endTime;
	auto print_Time = [&](const timeval &startTime, const timeval& endTime) {
    	long long elapsedTime = (endTime.tv_sec - startTime.tv_sec) * 1000000LL + (endTime.tv_usec - startTime.tv_usec);
    	printf("Elapsed time: %lld microseconds\n", elapsedTime); // 微秒
	};

	// // GPU 测速
	// puts("Pending GPU:");
	const int loopTime = 100;
	gettimeofday(&startTime, NULL);

	for (int i=0;i<loopTime;++i)
		gpu_conv<<<blockNum, threadNum>>>(img_gpu, kernel_gpu, res_gpu, width, height, kernel_size);

	gettimeofday(&endTime, NULL);
	print_Time(startTime, endTime);

	// CPU 测速
	puts("Pending CPU:");
	gettimeofday(&startTime, NULL);

	for (int i=0;i<loopTime;++i)
		cpu_conv(img, kernel, res, width, height, kernel_size);

	gettimeofday(&endTime, NULL);
	print_Time(startTime, endTime);

	// 显存->内存
	CUDA_CHECK(cudaMemcpy(res, res_gpu, width*height*sizeof(float), cudaMemcpyDeviceToHost));
	


	CUDA_CHECK(cudaFree(img_gpu));
	CUDA_CHECK(cudaFree(kernel_gpu));
	CUDA_CHECK(cudaFree(res_gpu));
	delete[] img; delete[] kernel; delete[] res;
    return 0;
}
