__kernel void branches(__global const int *in, __global int *out, int n)
{
	size_t i = get_global_id(0);
	int v = in[i], acc = 0, k;

	if (v > 10)
		acc = v * 3;
	else if (v < -10)
		acc = -v;
	else
		acc = v + 100;
	for (k = 0; k < n; k++) {
		if (k == v)
			continue;
		acc += k ^ v;
		if (acc > 100000)
			break;
	}
	while (acc & 1)
		acc >>= 1;
	out[i] = acc;
}
