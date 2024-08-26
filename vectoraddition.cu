#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <assert.h>
#include <iostream>
#include <vector>



__global__ void vectorAdd(const int* __restrict a, const int *__restrict b, int* __restrict c, int N) {
	int tid = (blockDim.x * blockIdx.x) + threadIdx.x;

	if (tid < N)
		c[tid] = a[tid] + b[tid];
}

void verify_result(std::vector<int>& a, std::vector<int>& b,
	std::vector<int>& c) {
	for (int i = 0; i < a.size(); i++) {
		assert(c[i] == a[i] + b[i]);
	}
}

int main() {
	constexpr int N = 1 << 16;
	constexpr size_t bytes = sizeof(int) * N;

	std::vector<int> a;
	//a.reserve(N);
	a.resize(N);
	std::vector<int> b;
	//b.reserve(N);
	b.resize(N);
	std::vector<int> c;
	//c.reserve(N);
	c.resize(N);


	for (int i = 0; i < N; i++) {
		a[i] = rand() % 100;
		b[i] = rand() % 100;
	}

	for (int i = 0; i < 10; i++) {
		std::cout << a[i] << " " << b[i] << std::endl;
	}

	// Allocate memory on device

	int* d_a, * d_b, *d_c;

	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	// copy data from host to device (CPU->GPU)
	cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

	int NUM_THREAD = 1 << 10;
	int NUM_BLOCKS = (N + NUM_THREAD - 1) / NUM_THREAD;


	vectorAdd <<<NUM_BLOCKS, NUM_THREAD >>> (d_a, d_b, d_c, N);

	cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
	verify_result(a, b, c);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);


	for (int i = 0; i < 10; i++)
		std::cout << c[i] << " ";

	std::cout << "Completed Successfully";

	return 0;

}