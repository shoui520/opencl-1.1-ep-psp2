__kernel void intops(__global const int *a, __global const int *b, __global int *o)
{
	size_t i = get_global_id(0);
	int x = a[i], y = b[i];
	o[6 * i + 0] = x / y;
	o[6 * i + 1] = x % y;
	o[6 * i + 2] = (int)((uint)x / (uint)y);
	o[6 * i + 3] = mul_hi(x, y);
	o[6 * i + 4] = rotate(x, y & 31);
	o[6 * i + 5] = clz(x) + (x >> (y & 31)) + ((uint)x >> (y & 31)) + abs(x) + max(x, y) + min(x, y);
}
