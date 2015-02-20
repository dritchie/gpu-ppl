#include "curand_kernel.h"


extern "C"
{
	__device__ void cu_rand_init(unsigned long long seed,
							  unsigned long long subsequence,
							  unsigned long long offset,
							  curandStateXORWOW* state)
	{
		curand_init(seed, subsequence, offset, state);
	}

	__device__ double cu_rand_uniform(curandStateXORWOW* state)
	{
		return curand_uniform_double(state);
	}
}