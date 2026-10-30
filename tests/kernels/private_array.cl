__kernel void private_array(__global const int *in, __global int *out)
{
	int t[16];
	size_t i = get_global_id(0);
	int k;
	for (k = 0; k < 16; k++)
		t[k] = in[i * 16 + k] * k;
	out[i] = t[in[i] & 15] + t[(in[i] >> 4) & 15];
}
