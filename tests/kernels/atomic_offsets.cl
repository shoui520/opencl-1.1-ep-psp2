typedef struct { uint a; uint b[3]; int c; } S;
__kernel void atomic_offsets(__global uint *u, __global S *s, __global int *o)
{
	size_t i = get_global_id(0);
	o[i * 8 + 0] = atomic_inc(&u[3]);
	o[i * 8 + 1] = atomic_add(u + 5, 2u);
	o[i * 8 + 2] = atomic_dec(&s[1].b[2]);
	o[i * 8 + 3] = atomic_max(&s->c, (int)i - 4);
	o[i * 8 + 4] = atomic_cmpxchg(&u[(i & 3) + 7], 0u, (uint)i);
	o[i * 8 + 5] = atomic_xchg(&u[12], (uint)i);
	o[i * 8 + 6] = atomic_min(&s[2].c, (int)i - 4);
	o[i * 8 + 7] = atomic_sub(&u[13], 3u);
}
