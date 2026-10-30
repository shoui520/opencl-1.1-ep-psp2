typedef struct { int a; int b[3]; int c; } S;

/* Pointers with constant offsets used as values, not only as load/store addresses. */
__kernel void pointers(__global int *a, __global S *s, __global int *o)
{
	size_t i = get_global_id(0);
	__global int *p = i & 1 ? &a[1] : &a[2];
	__global int *end = &a[6];
	__global int *r;
	__global S *t = &s[i & 3];
	__global int *f = i & 2 ? &t->b[2] : &t->c;
	int sum = 0, k = 0;

	o[i * 8 + 0] = p[i & 7];
	for (r = p; r < end && k < 16; r++, k++)	/* bounded even if end is wrong */
		sum += *r;
	o[i * 8 + 1] = sum * 16 + k;
	o[i * 8 + 2] = *f;
	o[i * 8 + 3] = (int)((size_t)f - (size_t)s);
	o[i * 8 + 4] = *(__global int *)((size_t)&s[2].b[1] | (get_work_dim() >> 8));
	o[i * 8 + 5] = (&s[1].c == f) + 2 * (f == &s[3].b[2]);
	o[i * 8 + 6] = (int)(&t->c - &s->b[0]);
	o[i * 8 + 7] = p[3];
}

/* Loads and stores through pointers the compiler can't tell apart (n is 0). */
__kernel void alias(__global int *a, __global int *o, int n)
{
	size_t i = get_global_id(0);
	__global int *b = a + n;
	int x = a[i];
	int y;

	b[i] = x + 1;
	y = a[i];
	b[i] = y * 2;
	o[i * 2] = x;
	o[i * 2 + 1] = a[i] + y;
}
