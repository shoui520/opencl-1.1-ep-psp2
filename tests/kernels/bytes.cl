#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
__kernel void bytes(__global const uchar *in, __global uchar *out, __global short *s)
{
	size_t i = get_global_id(0);
	uchar c = in[i];
	out[i] = (uchar)(c * 7 + 3);
	s[i] = (short)(c - 200);
}

__kernel void char4ops(__global const char4 *in, __global char4 *out)
{
	size_t i = get_global_id(0);
	char4 v = in[i];
	out[i] = v.wzyx + (char4)(1, -1, 2, -2);
}
