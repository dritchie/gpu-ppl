#include <random>
#include <limits>

// Which type of RNG should we expose?
typedef std::default_random_engine rng_state;

extern "C"
{
	size_t std_rand_state_size() { return sizeof(rng_state); }

	void std_rand_init(unsigned int seed, rng_state* state)
	{
		*state = rng_state(seed);
	}

	double std_rand_uniform(rng_state* state)
	{
		return std::generate_canonical<double, std::numeric_limits<double>::digits>(*state);
	}
}