/* Kernels for the API tests (tests/device/cases.py). */
typedef struct { int a; float b; short c; } P;

__kernel void add_one(__global int *a)
{
	a[get_global_id(0)] += 1;
}

__kernel void scale(__global const int *a, __global int *b, int k)
{
	size_t i = get_global_id(0);

	b[i] = a[i] * k;
}

__kernel void null_arg(__global int *o, __global const int *p)
{
	o[get_global_id(0)] = p == 0 ? 1 : (p[0] == 7 ? 2 : 3);
}

/* Each by-value argument kind, written out as words. */
__kernel void values(__global int *o, char c, uchar uc, short s, ushort us, char4 c4, uchar2 uc2,
		     short2 s2, float2 f2, float3 f3, float4 f4, P p, uint u)
{
	o[0] = c; o[1] = uc; o[2] = s; o[3] = us;
	o[4] = c4.x; o[5] = c4.y; o[6] = c4.z; o[7] = c4.w;
	o[8] = uc2.x; o[9] = uc2.y; o[10] = s2.x; o[11] = s2.y;
	o[12] = as_int(f2.x); o[13] = as_int(f2.y);
	o[14] = as_int(f3.x); o[15] = as_int(f3.y); o[16] = as_int(f3.z);
	o[17] = as_int(f4.x); o[18] = as_int(f4.y); o[19] = as_int(f4.z); o[20] = as_int(f4.w);
	o[21] = p.a; o[22] = as_int(p.b); o[23] = p.c; o[24] = u;
}

__kernel void wide_values(__global int *o, int8 i8, float16 f16, uint after)
{
	o[0] = i8.s0; o[1] = i8.s1; o[2] = i8.s2; o[3] = i8.s3;
	o[4] = i8.s4; o[5] = i8.s5; o[6] = i8.s6; o[7] = i8.s7;
	o[8] = as_int(f16.s0); o[9] = as_int(f16.s1); o[10] = as_int(f16.s2); o[11] = as_int(f16.s3);
	o[12] = as_int(f16.s4); o[13] = as_int(f16.s5); o[14] = as_int(f16.s6); o[15] = as_int(f16.s7);
	o[16] = as_int(f16.s8); o[17] = as_int(f16.s9); o[18] = as_int(f16.sa); o[19] = as_int(f16.sb);
	o[20] = as_int(f16.sc); o[21] = as_int(f16.sd); o[22] = as_int(f16.se); o[23] = as_int(f16.sf);
	o[24] = after;
}

/* Private and __local memory per work-item, 512 bytes, over a 2D range. */
__kernel void scratch(__global const int *in, __global int *out, __local int *l)
{
	size_t i = get_global_id(1) * get_global_size(0) + get_global_id(0);
	int p[64];
	int k, s = 0;

	for (k = 0; k < 64; k++)
		p[k] = in[(i + k) & 1023] + k;
	for (k = 0; k < 64; k++)
		l[k] = p[(k * 7 + i) & 63];
	for (k = 0; k < 64; k++)
		s += l[k] * (k + 1) + p[(i + k * 3) & 63];
	out[i] = s;
}

/* Correctly rounded single precision arithmetic. */
__kernel void fp_ops(__global const float *a, __global const float *b, __global float *o)
{
	size_t i = get_global_id(0);

	o[4 * i] = a[i] + b[i];
	o[4 * i + 1] = a[i] - b[i];
	o[4 * i + 2] = a[i] * b[i];
	o[4 * i + 3] = (float)(int)a[i] + 0.5f * b[i];
}
