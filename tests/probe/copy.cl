/* o[i] = in[i]: where loads and stores land relative to their address. */
__kernel void copy(__global uint *o, __global const uint *in, __global uint *count)
{
	uint i = get_global_id(0);

	o[i] = in[i];
	atomic_inc(count);
}
