#include "curand_kernel.h"


// Which type of RNG should we expose?
// DON'T FORGET to change the RandomState type in platform/cuda.t, also
// typedef curandStateXORWOW rng_state;
typedef curandStateMRG32k3a rng_state;


extern "C"
{
	__device__ void cu_rand_init(unsigned long long seed,
							  unsigned long long subsequence,
							  unsigned long long offset,
							  rng_state* state)
	{
		curand_init(seed, subsequence, offset, state);
	}

	__device__ double cu_rand_uniform(rng_state* state)
	{
		return curand_uniform_double(state);
	}
}