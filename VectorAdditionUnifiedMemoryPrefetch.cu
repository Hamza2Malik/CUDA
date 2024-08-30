#include <algorithm>
#include <assert.h>
#include <iostream>
#include <vector>



__global__ void vectorAdd(const int* __restrict a, const int *__restrict b, int* __restrict c, int N) {
	int tid = (blockDim.x * blockIdx.x) + threadIdx.x;

	if (tid < N)
		c[tid] = a[tid] + b[tid];
}

void allocateMemory(int *&a, int *&b, int *&c, int N){
    
    for(int i = 0; i<N; i++){
        assert(c[i]==a[i]+b[i]);
    }
}



int main() {
	constexpr int N = 1 << 16;
	constexpr size_t bytes = sizeof(int) * N;

	int *a, *b, *c;

    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    // Get the device ID for prefetching calls
    int id = cudaGetDevice(&id);

    // Set some hints about the data and do some prefetching
    cudaMemAdvise(a, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise(b, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemPrefetchAsync(c, bytes, id);




	for (int i = 0; i < N; i++) {
		a[i] = rand() % 100;
		b[i] = rand() % 100;
	}

    cudaMemAdvise(a, bytes, cudaMemAdviseSetReadMostly, id);
    cudaMemAdvise(b, bytes, cudaMemAdviseSetReadMostly, id);
    cudaMemPrefetchAsync(a, bytes, id);
    cudaMemPrefetchAsync(b, bytes, id);

	int NUM_THREAD = 1 << 10;
	int NUM_BLOCKS = (N + NUM_THREAD - 1) / NUM_THREAD;


	vectorAdd <<<NUM_BLOCKS, NUM_THREAD >>> (a, b, c, N);

    cudaDeviceSynchronize();

    cudaMemPrefetchAsync(a, bytes, cudaCpuDeviceId);
    cudaMemPrefetchAsync(b, bytes, cudaCpuDeviceId);
    cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);
	
    allocateMemory(a, b, c, N);

    // for(int i = 0; i<N; i++){
    //     assert(c[i]==a[i]+b[i]);
    // }
	


    for (int i = 0; i < 10; i++)
		std::cout << c[i] << " ";
	cudaFree(a);
	cudaFree(b);
	cudaFree(c);


	

	std::cout << "Completed Successfully";

	return 0;

}