__kernel void atomics(__global int *counters, __global uint *u, __global const int *in)
{
	size_t i = get_global_id(0);
	int v = in[i];
	atomic_add(&counters[0], v);
	atomic_sub(&counters[1], 1);
	atomic_max(&counters[2], v);
	atomic_min(&counters[3], v);
	atomic_inc(&u[0]);
	atomic_dec(&u[1]);
	atomic_xchg(&u[2], (uint)i);
	atomic_cmpxchg(&u[3], 0u, (uint)i);
	atomic_and(&u[4], ~(1u << (i & 31)));
	atomic_or(&u[5], 1u << (i & 31));
	atomic_xor(&u[6], (uint)v);
}

/* Contention: every item updates the same words many times, two in a row. */
__kernel void atomic_stress(__global int *c, __global uint *hist, __global const uint *in, int iters)
{
	size_t i = get_global_id(0);
	int k;

	for (k = 0; k < iters; k++) {
		atomic_add(&c[0], 1);
		atomic_add(&c[0], 2);
		atomic_inc(&hist[(in[i] + k) & 15]);
	}
	atomic_add(&c[1], (int)i);
	atomic_sub(&c[2], (int)i);
}
