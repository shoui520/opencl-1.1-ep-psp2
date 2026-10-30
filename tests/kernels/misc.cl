typedef struct { int a; float b; short c; } params_t;

__constant int table[8] = { 3, 1, 4, 1, 5, 9, 2, 6 };

int helper(int x, __constant int *t)
{
	return t[x & 7] * x;
}

__kernel void misc(__global int *out, params_t p, __constant int *ct)
{
	size_t i = get_global_id(0);
	out[i] = helper((int)i, table) + p.a + (int)p.b + p.c + ct[i & 3];
}
