import numpy as np


def np_random(seed):
    if seed is not None and not (isinstance(seed, int) and 0 <= seed):
        raise Exception(f"Seed must be a non-negative integer or omitted, not {seed}")

    seed_seq = np.random.SeedSequence(seed)
    np_seed = seed_seq.entropy
    rng = RandomNumberGenerator(np.random.PCG64(seed_seq))  
    return rng, np_seed


RNG = RandomNumberGenerator = np.random.Generator
