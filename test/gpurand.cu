#include "curand_kernel.h"

#define seed 42

__global__ void kernel(double* outdata)
{
	// curandStateXORWOW_t state;
	curandStateMRG32k3a_t state;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, idx, 0, &state);
	outdata[idx] = curand_uniform(&state);
}

int main()
{
	double* data;
	cudaMalloc((void**)data, sizeof(double));
	kernel<<<1,1>>>(data);
	cudaFree(data);
	return 0;
}