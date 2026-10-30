__kernel void reduce(__global const int *in, __global int *out, __local int *scratch)
{
	size_t lid = get_local_id(0), n = get_local_size(0), s;
	scratch[lid] = in[get_global_id(0)];
	barrier(CLK_LOCAL_MEM_FENCE);
	for (s = n / 2; s > 0; s >>= 1) {
		if (lid < s)
			scratch[lid] += scratch[lid + s];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (lid == 0)
		out[get_group_id(0)] = scratch[0];
}

__kernel void local_atomic(__global int *out)
{
	__local int count;
	if (get_local_id(0) == 0)
		count = 0;
	barrier(CLK_LOCAL_MEM_FENCE);
	atomic_inc(&count);
	barrier(CLK_LOCAL_MEM_FENCE);
	if (get_local_id(0) == 0)
		out[get_group_id(0)] = count;
}
