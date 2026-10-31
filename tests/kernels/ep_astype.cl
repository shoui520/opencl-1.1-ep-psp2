/* Section 6.2.4.2: as_type between every pair of types of one size (but
   the 3-component ones): the bytes are kept.  Each kernel takes one source
   type and stores it as each type of its size in turn (itself as is: the
   front end has no as_type to the same type). */
__kernel void a_ushort(__global const ushort *x, __global uchar *o)
{
	int i = get_global_id(0);
	ushort v = x[i];
	__global uchar *p = o + 8 * i;

	*(__global ushort *)(p + 0) = v;
	*(__global short *)(p + 2) = as_short(v);
	*(__global uchar2 *)(p + 4) = as_uchar2(v);
	*(__global char2 *)(p + 6) = as_char2(v);
}

__kernel void a_short(__global const short *x, __global uchar *o)
{
	int i = get_global_id(0);
	short v = x[i];
	__global uchar *p = o + 8 * i;

	*(__global ushort *)(p + 0) = as_ushort(v);
	*(__global short *)(p + 2) = v;
	*(__global uchar2 *)(p + 4) = as_uchar2(v);
	*(__global char2 *)(p + 6) = as_char2(v);
}

__kernel void a_uchar2(__global const uchar2 *x, __global uchar *o)
{
	int i = get_global_id(0);
	uchar2 v = x[i];
	__global uchar *p = o + 8 * i;

	*(__global ushort *)(p + 0) = as_ushort(v);
	*(__global short *)(p + 2) = as_short(v);
	*(__global uchar2 *)(p + 4) = v;
	*(__global char2 *)(p + 6) = as_char2(v);
}

__kernel void a_char2(__global const char2 *x, __global uchar *o)
{
	int i = get_global_id(0);
	char2 v = x[i];
	__global uchar *p = o + 8 * i;

	*(__global ushort *)(p + 0) = as_ushort(v);
	*(__global short *)(p + 2) = as_short(v);
	*(__global uchar2 *)(p + 4) = as_uchar2(v);
	*(__global char2 *)(p + 6) = v;
}

__kernel void a_uint(__global const uint *x, __global uchar *o)
{
	int i = get_global_id(0);
	uint v = x[i];
	__global uchar *p = o + 28 * i;

	*(__global uint *)(p + 0) = v;
	*(__global int *)(p + 4) = as_int(v);
	*(__global float *)(p + 8) = as_float(v);
	*(__global uchar4 *)(p + 12) = as_uchar4(v);
	*(__global char4 *)(p + 16) = as_char4(v);
	*(__global ushort2 *)(p + 20) = as_ushort2(v);
	*(__global short2 *)(p + 24) = as_short2(v);
}

__kernel void a_int(__global const int *x, __global uchar *o)
{
	int i = get_global_id(0);
	int v = x[i];
	__global uchar *p = o + 28 * i;

	*(__global uint *)(p + 0) = as_uint(v);
	*(__global int *)(p + 4) = v;
	*(__global float *)(p + 8) = as_float(v);
	*(__global uchar4 *)(p + 12) = as_uchar4(v);
	*(__global char4 *)(p + 16) = as_char4(v);
	*(__global ushort2 *)(p + 20) = as_ushort2(v);
	*(__global short2 *)(p + 24) = as_short2(v);
}

__kernel void a_float(__global const float *x, __global uchar *o)
{
	int i = get_global_id(0);
	float v = x[i];
	__global uchar *p = o + 28 * i;

	*(__global uint *)(p + 0) = as_uint(v);
	*(__global int *)(p + 4) = as_int(v);
	*(__global float *)(p + 8) = v;
	*(__global uchar4 *)(p + 12) = as_uchar4(v);
	*(__global char4 *)(p + 16) = as_char4(v);
	*(__global ushort2 *)(p + 20) = as_ushort2(v);
	*(__global short2 *)(p + 24) = as_short2(v);
}

__kernel void a_uchar4(__global const uchar4 *x, __global uchar *o)
{
	int i = get_global_id(0);
	uchar4 v = x[i];
	__global uchar *p = o + 28 * i;

	*(__global uint *)(p + 0) = as_uint(v);
	*(__global int *)(p + 4) = as_int(v);
	*(__global float *)(p + 8) = as_float(v);
	*(__global uchar4 *)(p + 12) = v;
	*(__global char4 *)(p + 16) = as_char4(v);
	*(__global ushort2 *)(p + 20) = as_ushort2(v);
	*(__global short2 *)(p + 24) = as_short2(v);
}

__kernel void a_char4(__global const char4 *x, __global uchar *o)
{
	int i = get_global_id(0);
	char4 v = x[i];
	__global uchar *p = o + 28 * i;

	*(__global uint *)(p + 0) = as_uint(v);
	*(__global int *)(p + 4) = as_int(v);
	*(__global float *)(p + 8) = as_float(v);
	*(__global uchar4 *)(p + 12) = as_uchar4(v);
	*(__global char4 *)(p + 16) = v;
	*(__global ushort2 *)(p + 20) = as_ushort2(v);
	*(__global short2 *)(p + 24) = as_short2(v);
}

__kernel void a_ushort2(__global const ushort2 *x, __global uchar *o)
{
	int i = get_global_id(0);
	ushort2 v = x[i];
	__global uchar *p = o + 28 * i;

	*(__global uint *)(p + 0) = as_uint(v);
	*(__global int *)(p + 4) = as_int(v);
	*(__global float *)(p + 8) = as_float(v);
	*(__global uchar4 *)(p + 12) = as_uchar4(v);
	*(__global char4 *)(p + 16) = as_char4(v);
	*(__global ushort2 *)(p + 20) = v;
	*(__global short2 *)(p + 24) = as_short2(v);
}

__kernel void a_short2(__global const short2 *x, __global uchar *o)
{
	int i = get_global_id(0);
	short2 v = x[i];
	__global uchar *p = o + 28 * i;

	*(__global uint *)(p + 0) = as_uint(v);
	*(__global int *)(p + 4) = as_int(v);
	*(__global float *)(p + 8) = as_float(v);
	*(__global uchar4 *)(p + 12) = as_uchar4(v);
	*(__global char4 *)(p + 16) = as_char4(v);
	*(__global ushort2 *)(p + 20) = as_ushort2(v);
	*(__global short2 *)(p + 24) = v;
}

__kernel void a_uint2(__global const uint2 *x, __global uchar *o)
{
	int i = get_global_id(0);
	uint2 v = x[i];
	__global uchar *p = o + 56 * i;

	*(__global uint2 *)(p + 0) = v;
	*(__global int2 *)(p + 8) = as_int2(v);
	*(__global float2 *)(p + 16) = as_float2(v);
	*(__global uchar8 *)(p + 24) = as_uchar8(v);
	*(__global char8 *)(p + 32) = as_char8(v);
	*(__global ushort4 *)(p + 40) = as_ushort4(v);
	*(__global short4 *)(p + 48) = as_short4(v);
}

__kernel void a_int2(__global const int2 *x, __global uchar *o)
{
	int i = get_global_id(0);
	int2 v = x[i];
	__global uchar *p = o + 56 * i;

	*(__global uint2 *)(p + 0) = as_uint2(v);
	*(__global int2 *)(p + 8) = v;
	*(__global float2 *)(p + 16) = as_float2(v);
	*(__global uchar8 *)(p + 24) = as_uchar8(v);
	*(__global char8 *)(p + 32) = as_char8(v);
	*(__global ushort4 *)(p + 40) = as_ushort4(v);
	*(__global short4 *)(p + 48) = as_short4(v);
}

__kernel void a_float2(__global const float2 *x, __global uchar *o)
{
	int i = get_global_id(0);
	float2 v = x[i];
	__global uchar *p = o + 56 * i;

	*(__global uint2 *)(p + 0) = as_uint2(v);
	*(__global int2 *)(p + 8) = as_int2(v);
	*(__global float2 *)(p + 16) = v;
	*(__global uchar8 *)(p + 24) = as_uchar8(v);
	*(__global char8 *)(p + 32) = as_char8(v);
	*(__global ushort4 *)(p + 40) = as_ushort4(v);
	*(__global short4 *)(p + 48) = as_short4(v);
}

__kernel void a_uchar8(__global const uchar8 *x, __global uchar *o)
{
	int i = get_global_id(0);
	uchar8 v = x[i];
	__global uchar *p = o + 56 * i;

	*(__global uint2 *)(p + 0) = as_uint2(v);
	*(__global int2 *)(p + 8) = as_int2(v);
	*(__global float2 *)(p + 16) = as_float2(v);
	*(__global uchar8 *)(p + 24) = v;
	*(__global char8 *)(p + 32) = as_char8(v);
	*(__global ushort4 *)(p + 40) = as_ushort4(v);
	*(__global short4 *)(p + 48) = as_short4(v);
}

__kernel void a_char8(__global const char8 *x, __global uchar *o)
{
	int i = get_global_id(0);
	char8 v = x[i];
	__global uchar *p = o + 56 * i;

	*(__global uint2 *)(p + 0) = as_uint2(v);
	*(__global int2 *)(p + 8) = as_int2(v);
	*(__global float2 *)(p + 16) = as_float2(v);
	*(__global uchar8 *)(p + 24) = as_uchar8(v);
	*(__global char8 *)(p + 32) = v;
	*(__global ushort4 *)(p + 40) = as_ushort4(v);
	*(__global short4 *)(p + 48) = as_short4(v);
}

__kernel void a_ushort4(__global const ushort4 *x, __global uchar *o)
{
	int i = get_global_id(0);
	ushort4 v = x[i];
	__global uchar *p = o + 56 * i;

	*(__global uint2 *)(p + 0) = as_uint2(v);
	*(__global int2 *)(p + 8) = as_int2(v);
	*(__global float2 *)(p + 16) = as_float2(v);
	*(__global uchar8 *)(p + 24) = as_uchar8(v);
	*(__global char8 *)(p + 32) = as_char8(v);
	*(__global ushort4 *)(p + 40) = v;
	*(__global short4 *)(p + 48) = as_short4(v);
}

__kernel void a_short4(__global const short4 *x, __global uchar *o)
{
	int i = get_global_id(0);
	short4 v = x[i];
	__global uchar *p = o + 56 * i;

	*(__global uint2 *)(p + 0) = as_uint2(v);
	*(__global int2 *)(p + 8) = as_int2(v);
	*(__global float2 *)(p + 16) = as_float2(v);
	*(__global uchar8 *)(p + 24) = as_uchar8(v);
	*(__global char8 *)(p + 32) = as_char8(v);
	*(__global ushort4 *)(p + 40) = as_ushort4(v);
	*(__global short4 *)(p + 48) = v;
}

__kernel void a_uint4(__global const uint4 *x, __global uchar *o)
{
	int i = get_global_id(0);
	uint4 v = x[i];
	__global uchar *p = o + 112 * i;

	*(__global uint4 *)(p + 0) = v;
	*(__global int4 *)(p + 16) = as_int4(v);
	*(__global float4 *)(p + 32) = as_float4(v);
	*(__global uchar16 *)(p + 48) = as_uchar16(v);
	*(__global char16 *)(p + 64) = as_char16(v);
	*(__global ushort8 *)(p + 80) = as_ushort8(v);
	*(__global short8 *)(p + 96) = as_short8(v);
}

__kernel void a_int4(__global const int4 *x, __global uchar *o)
{
	int i = get_global_id(0);
	int4 v = x[i];
	__global uchar *p = o + 112 * i;

	*(__global uint4 *)(p + 0) = as_uint4(v);
	*(__global int4 *)(p + 16) = v;
	*(__global float4 *)(p + 32) = as_float4(v);
	*(__global uchar16 *)(p + 48) = as_uchar16(v);
	*(__global char16 *)(p + 64) = as_char16(v);
	*(__global ushort8 *)(p + 80) = as_ushort8(v);
	*(__global short8 *)(p + 96) = as_short8(v);
}

__kernel void a_float4(__global const float4 *x, __global uchar *o)
{
	int i = get_global_id(0);
	float4 v = x[i];
	__global uchar *p = o + 112 * i;

	*(__global uint4 *)(p + 0) = as_uint4(v);
	*(__global int4 *)(p + 16) = as_int4(v);
	*(__global float4 *)(p + 32) = v;
	*(__global uchar16 *)(p + 48) = as_uchar16(v);
	*(__global char16 *)(p + 64) = as_char16(v);
	*(__global ushort8 *)(p + 80) = as_ushort8(v);
	*(__global short8 *)(p + 96) = as_short8(v);
}

__kernel void a_uchar16(__global const uchar16 *x, __global uchar *o)
{
	int i = get_global_id(0);
	uchar16 v = x[i];
	__global uchar *p = o + 112 * i;

	*(__global uint4 *)(p + 0) = as_uint4(v);
	*(__global int4 *)(p + 16) = as_int4(v);
	*(__global float4 *)(p + 32) = as_float4(v);
	*(__global uchar16 *)(p + 48) = v;
	*(__global char16 *)(p + 64) = as_char16(v);
	*(__global ushort8 *)(p + 80) = as_ushort8(v);
	*(__global short8 *)(p + 96) = as_short8(v);
}

__kernel void a_char16(__global const char16 *x, __global uchar *o)
{
	int i = get_global_id(0);
	char16 v = x[i];
	__global uchar *p = o + 112 * i;

	*(__global uint4 *)(p + 0) = as_uint4(v);
	*(__global int4 *)(p + 16) = as_int4(v);
	*(__global float4 *)(p + 32) = as_float4(v);
	*(__global uchar16 *)(p + 48) = as_uchar16(v);
	*(__global char16 *)(p + 64) = v;
	*(__global ushort8 *)(p + 80) = as_ushort8(v);
	*(__global short8 *)(p + 96) = as_short8(v);
}

__kernel void a_ushort8(__global const ushort8 *x, __global uchar *o)
{
	int i = get_global_id(0);
	ushort8 v = x[i];
	__global uchar *p = o + 112 * i;

	*(__global uint4 *)(p + 0) = as_uint4(v);
	*(__global int4 *)(p + 16) = as_int4(v);
	*(__global float4 *)(p + 32) = as_float4(v);
	*(__global uchar16 *)(p + 48) = as_uchar16(v);
	*(__global char16 *)(p + 64) = as_char16(v);
	*(__global ushort8 *)(p + 80) = v;
	*(__global short8 *)(p + 96) = as_short8(v);
}

__kernel void a_short8(__global const short8 *x, __global uchar *o)
{
	int i = get_global_id(0);
	short8 v = x[i];
	__global uchar *p = o + 112 * i;

	*(__global uint4 *)(p + 0) = as_uint4(v);
	*(__global int4 *)(p + 16) = as_int4(v);
	*(__global float4 *)(p + 32) = as_float4(v);
	*(__global uchar16 *)(p + 48) = as_uchar16(v);
	*(__global char16 *)(p + 64) = as_char16(v);
	*(__global ushort8 *)(p + 80) = as_ushort8(v);
	*(__global short8 *)(p + 96) = v;
}

__kernel void a_uint8(__global const uint8 *x, __global uchar *o)
{
	int i = get_global_id(0);
	uint8 v = x[i];
	__global uchar *p = o + 160 * i;

	*(__global uint8 *)(p + 0) = v;
	*(__global int8 *)(p + 32) = as_int8(v);
	*(__global float8 *)(p + 64) = as_float8(v);
	*(__global ushort16 *)(p + 96) = as_ushort16(v);
	*(__global short16 *)(p + 128) = as_short16(v);
}

__kernel void a_int8(__global const int8 *x, __global uchar *o)
{
	int i = get_global_id(0);
	int8 v = x[i];
	__global uchar *p = o + 160 * i;

	*(__global uint8 *)(p + 0) = as_uint8(v);
	*(__global int8 *)(p + 32) = v;
	*(__global float8 *)(p + 64) = as_float8(v);
	*(__global ushort16 *)(p + 96) = as_ushort16(v);
	*(__global short16 *)(p + 128) = as_short16(v);
}

__kernel void a_float8(__global const float8 *x, __global uchar *o)
{
	int i = get_global_id(0);
	float8 v = x[i];
	__global uchar *p = o + 160 * i;

	*(__global uint8 *)(p + 0) = as_uint8(v);
	*(__global int8 *)(p + 32) = as_int8(v);
	*(__global float8 *)(p + 64) = v;
	*(__global ushort16 *)(p + 96) = as_ushort16(v);
	*(__global short16 *)(p + 128) = as_short16(v);
}

__kernel void a_ushort16(__global const ushort16 *x, __global uchar *o)
{
	int i = get_global_id(0);
	ushort16 v = x[i];
	__global uchar *p = o + 160 * i;

	*(__global uint8 *)(p + 0) = as_uint8(v);
	*(__global int8 *)(p + 32) = as_int8(v);
	*(__global float8 *)(p + 64) = as_float8(v);
	*(__global ushort16 *)(p + 96) = v;
	*(__global short16 *)(p + 128) = as_short16(v);
}

__kernel void a_short16(__global const short16 *x, __global uchar *o)
{
	int i = get_global_id(0);
	short16 v = x[i];
	__global uchar *p = o + 160 * i;

	*(__global uint8 *)(p + 0) = as_uint8(v);
	*(__global int8 *)(p + 32) = as_int8(v);
	*(__global float8 *)(p + 64) = as_float8(v);
	*(__global ushort16 *)(p + 96) = as_ushort16(v);
	*(__global short16 *)(p + 128) = v;
}
