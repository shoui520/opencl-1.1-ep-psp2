__kernel void convert(__global const float *f, __global int *i32, __global uchar *u8, __global float *back)
{
	size_t i = get_global_id(0);
	float v = f[i];
	i32[2 * i] = convert_int_sat(v);
	i32[2 * i + 1] = convert_int_rte(v);
	u8[i] = convert_uchar_sat(v);
	back[i] = convert_float((int)v) + as_float(as_int(v) ^ 0x80000000);
}
