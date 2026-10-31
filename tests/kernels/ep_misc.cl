/* Sections 6.11.7 (vector data load and store), 6.11.9 and 6.11.10 (fences,
   async copies and prefetch) and 6.9 (the macros); 6.11.12's shuffles are
   ep_shuffle.cl. */

/* pocl, the reference, lacks vloada_half and vstorea_half (n = 1) */
#ifndef __EMBEDDED_PROFILE__
#define vloada_half vload_half
#define vstorea_half vstore_half
#endif

/* every vload_half form on 16 halves an item: rows of 16 floats */
__kernel void h_load(__global const half *h, __constant half *c, __global float *o,
		     __local half *l)
{
	int i = get_global_id(0), k;
	__global const half *p = h + 16 * i;
	__global float *r = o + 15 * 16 * i;
	ushort t[16];
	float3 v;

	for (k = 0; k < 16; k++)
		r[k] = vload_half(k, p);
	for (k = 0; k < 8; k++)
		vstore2(vload_half2(k, p), k, r + 16);
	for (k = 0; k < 5; k++)
		vstore3(vload_half3(k, p), k, r + 32);
	r[47] = vload_half(15, p);
	for (k = 0; k < 4; k++)
		vstore4(vload_half4(k, p), k, r + 48);
	for (k = 0; k < 2; k++)
		vstore8(vload_half8(k, p), k, r + 64);
	vstore16(vload_half16(0, p), 0, r + 80);

	for (k = 0; k < 16; k++)
		r[96 + k] = vloada_half(k, p);
	for (k = 0; k < 8; k++)
		vstore2(vloada_half2(k, p), k, r + 112);
	/* half3s are 4 halves apart */
	for (k = 0; k < 4; k++) {
		v = vloada_half3(k, p);
		r[128 + 4 * k] = v.x; r[129 + 4 * k] = v.y; r[130 + 4 * k] = v.z;
	}
	for (k = 0; k < 4; k++)
		vstore4(vloada_half4(k, p), k, r + 144);
	for (k = 0; k < 2; k++)
		vstore8(vloada_half8(k, p), k, r + 160);
	vstore16(vloada_half16(0, p), 0, r + 176);

	/* __constant (32 items' worth), __private and __local */
	vstore16(vload_half16(i & 31, c), 0, r + 192);
	for (k = 0; k < 16; k++)
		t[k] = ((__global const ushort *)p)[k];
	vstore16(vload_half16(0, (__private half *)t), 0, r + 208);
	for (k = 0; k < 16; k++)
		((__local ushort *)l)[k] = t[k];
	barrier(CLK_LOCAL_MEM_FENCE);
	vstore16(vloada_half16(0, l), 0, r + 224);
}

/* every vstore_half form on 16 floats an item: rows of 16 halves */
__kernel void h_store(__global const float *x, __global half *o, __local half *l)
{
	int i = get_global_id(0), k;
	__global const float *p = x + 16 * i;
	__global half *r = o + 23 * 16 * i;
	float16 v = vload16(0, p);
	ushort t[16];

	for (k = 0; k < 16; k++) {
		vstore_half(p[k], k, r);
		vstore_half_rte(p[k], k, r + 16);
		vstore_half_rtz(p[k], k, r + 32);
		vstore_half_rtp(p[k], k, r + 48);
		vstore_half_rtn(p[k], k, r + 64);
	}
	for (k = 0; k < 8; k++)
		vstore_half2(vload2(k, p), k, r + 80);
	for (k = 0; k < 5; k++)
		vstore_half3(vload3(k, p), k, r + 96);
	vstore_half(p[15], 15, r + 96);
	for (k = 0; k < 4; k++)
		vstore_half4(vload4(k, p), k, r + 112);
	for (k = 0; k < 2; k++)
		vstore_half8(vload8(k, p), k, r + 128);
	vstore_half16(v, 0, r + 144);

	for (k = 0; k < 8; k++)
		vstore_half2_rtz(vload2(k, p), k, r + 160);
	for (k = 0; k < 5; k++)
		vstore_half3_rtp(vload3(k, p), k, r + 176);
	vstore_half_rtp(p[15], 15, r + 176);
	for (k = 0; k < 4; k++)
		vstore_half4_rtn(vload4(k, p), k, r + 192);
	for (k = 0; k < 2; k++)
		vstore_half8_rte(vload8(k, p), k, r + 208);
	vstore_half16_rtz(v, 0, r + 224);

	for (k = 0; k < 16; k++)
		vstorea_half(p[k], k, r + 240);
	for (k = 0; k < 8; k++)
		vstorea_half2_rtp(vload2(k, p), k, r + 256);
	/* half3s 4 halves apart, from every 4 floats' first 3 */
	for (k = 0; k < 4; k++)
		vstorea_half3_rtn(vload3(0, p + 4 * k), k, r + 272);
	for (k = 0; k < 4; k++)
		vstorea_half4(vload4(k, p), k, r + 288);
	for (k = 0; k < 2; k++)
		vstorea_half8_rtz(vload8(k, p), k, r + 304);
	vstorea_half16_rte(v, 0, r + 320);

	/* __local and __private */
	vstore_half16_rtn(v, 0, l);
	barrier(CLK_LOCAL_MEM_FENCE);
	for (k = 0; k < 16; k++)
		((__global ushort *)r)[336 + k] = ((__local ushort *)l)[k];
	vstore_half8(v.lo, 0, (__private half *)t);
	vstorea_half8_rtp(v.hi, 1, (__private half *)t);
	for (k = 0; k < 16; k++)
		((__global ushort *)r)[352 + k] = t[k];
}

/* vloadn and vstoren of each gentype and n, at an offset, through
   __global, __constant, __local and __private pointers: 64 bytes an item
   in, 5 times that out */
#define VLS(T)										\
__kernel void v_##T(__global const T *a, __constant T *c, __global T *o, __local T *l)	\
{											\
	int i = get_global_id(0), k, n = 64 / sizeof(T);				\
	__global const T *p = a + n * i;						\
	__global T *r = o + 5 * n * i;							\
	T t[64];									\
											\
	for (k = 0; k < n / 2; k++)							\
		vstore2(vload2(k, p), k, r);						\
	for (k = 0; k < n / 3; k++)							\
		vstore3(vload3(k, p), k, r + n);					\
	for (k = 3 * (n / 3); k < n; k++)						\
		r[n + k] = p[k];							\
	for (k = 0; k < n / 8; k++)							\
		vstore4(vload4(k, p), k, r + 2 * n);					\
	for (k = n / 16; k < n / 8; k++)						\
		vstore8(vload8(k, p), k, r + 2 * n);					\
	for (k = 0; k < n / 16; k++)							\
		vstore16(vload16(k, p) + vload16(k, c + n * (i & 7)), k, r + 3 * n);	\
	for (k = 0; k < n / 4; k++) {							\
		vstore4(vload4(k, p), k, l);						\
		vstore4(vload4(k, p), k, t);						\
	}										\
	barrier(CLK_LOCAL_MEM_FENCE);							\
	for (k = 0; k < n / 2; k++)							\
		vstore2(vload2(k, l) ^ vload2(k, t), k, r + 4 * n);			\
}
VLS(char) VLS(uchar) VLS(short) VLS(ushort) VLS(int) VLS(uint)

/* float: no ^, so the __local and __private copies go to their own row */
__kernel void v_float(__global const float *a, __constant float *c, __global float *o,
		      __local float *l)
{
	int i = get_global_id(0), k;
	__global const float *p = a + 16 * i;
	__global float *r = o + 6 * 16 * i;
	float t[16];

	for (k = 0; k < 8; k++)
		vstore2(vload2(k, p), k, r);
	for (k = 0; k < 5; k++)
		vstore3(vload3(k, p), k, r + 16);
	r[31] = p[15];
	for (k = 0; k < 4; k++)
		vstore4(vload4(k, p), k, r + 32);
	for (k = 0; k < 2; k++)
		vstore8(vload8(k, p), k, r + 48);
	vstore16(vload16(0, p) * vload16(i & 7, c), 0, r + 64);
	vstore16(vload16(0, p), 0, l);
	vstore16(vload16(0, l), 0, t);
	barrier(CLK_LOCAL_MEM_FENCE);
	vstore16(vload16(0, t), 0, r + 80);
}

/* async_work_group_copy and the strided copies, both ways, with the
   fences, wait_group_events and prefetch: 64 ints an item */
__kernel void a_copy(__global const int *a, __global int *o, __local int *l)
{
	int i = get_global_id(0);
	__global const int *p = a + 64 * i;
	__global int *r = o + 4 * 64 * i;
	event_t e[4], f;

	prefetch(p, 64);
	e[0] = async_work_group_copy(l, p, 64, 0);
	wait_group_events(1, e);
	e[1] = async_work_group_copy(r, l, 64, 0);
	/* every 4th int, into 16 */
	e[2] = async_work_group_strided_copy(l, p + 1, 16, 4, 0);
	wait_group_events(2, e + 1);
	e[3] = async_work_group_copy(r + 64, l, 16, 0);
	wait_group_events(1, e + 3);
	/* 16 ints out to every 4th, those between untouched */
	e[0] = async_work_group_copy(l, p, 16, 0);
	wait_group_events(1, e);
	mem_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	e[1] = async_work_group_strided_copy(r + 128, l, 16, 4, 0);
	wait_group_events(1, e + 1);
	/* a vector type, and the event of one copy given to the next */
	f = async_work_group_copy((__local int4 *)l, (__global const int4 *)p, 16, 0);
	f = async_work_group_copy((__global int4 *)(r + 192), (__local const int4 *)l, 16, f);
	wait_group_events(1, &f);
	read_mem_fence(CLK_GLOBAL_MEM_FENCE);
	write_mem_fence(CLK_GLOBAL_MEM_FENCE);
}

/* section 6.9's pragmas: these are accepted, and an unknown one ignored */
#pragma OPENCL FP_CONTRACT ON
#pragma OPENCL EXTENSION all : disable
#pragma vita_unknown_pragma

/* section 6.9's macros, section 10's __EMBEDDED_PROFILE__, the limits of
   6.11.3, vec_step and sizeof (6.1, 6.3); -1 when one is undefined */
__kernel void m_macros(__global int *o)
{
	o[0] = __OPENCL_VERSION__; o[1] = CL_VERSION_1_0; o[2] = CL_VERSION_1_1;
#ifdef __ENDIAN_LITTLE__
	o[3] = __ENDIAN_LITTLE__;
#else
	o[3] = -1;
#endif
#ifdef __IMAGE_SUPPORT__
	o[4] = __IMAGE_SUPPORT__;
#else
	o[4] = -1;
#endif
#ifdef __EMBEDDED_PROFILE__
	o[5] = __EMBEDDED_PROFILE__;
#else
	o[5] = -1;
#endif
#ifdef __FAST_RELAXED_MATH__
	o[6] = __FAST_RELAXED_MATH__;
#else
	o[6] = -1;
#endif
	o[7] = __LINE__; o[8] = sizeof(__FILE__) > 1;
	o[9] = CHAR_BIT; o[10] = CHAR_MAX; o[11] = CHAR_MIN; o[12] = INT_MAX; o[13] = INT_MIN;
	o[14] = SCHAR_MAX; o[15] = SCHAR_MIN; o[16] = SHRT_MAX; o[17] = SHRT_MIN;
	o[18] = UCHAR_MAX; o[19] = USHRT_MAX; o[20] = (int)UINT_MAX;
	o[21] = vec_step(float); o[22] = vec_step(int2); o[23] = vec_step(char3);
	o[24] = vec_step(uchar4); o[25] = vec_step(short8); o[26] = vec_step(float16);
	o[27] = sizeof(char); o[28] = sizeof(short); o[29] = sizeof(int); o[30] = sizeof(float);
	o[31] = sizeof(float3); o[32] = sizeof(uchar3); o[33] = sizeof(short16);
	o[34] = sizeof(size_t); o[35] = sizeof(ptrdiff_t); o[36] = sizeof(intptr_t);
	o[37] = sizeof(uintptr_t); o[38] = sizeof(__global int *); o[39] = sizeof(bool);
	o[40] = sizeof(half); o[41] = sizeof(event_t) > 0;
#ifdef cl_khr_fp16
	o[42] = 1;
#else
	o[42] = 0;
#endif
#ifdef cl_khr_fp64
	o[43] = 1;
#else
	o[43] = 0;
#endif
#ifdef cles_khr_int64
	o[44] = 1;
#else
	o[44] = 0;
#endif
#ifdef cl_khr_global_int32_base_atomics
	o[45] = 1;
#else
	o[45] = 0;
#endif
#ifdef cl_khr_local_int32_base_atomics
	o[46] = 1;
#else
	o[46] = 0;
#endif
#ifdef cl_khr_byte_addressable_store
	o[47] = 1;
#else
	o[47] = 0;
#endif
}

/* section 6.10's attributes, and 6.9's __kernel_exec */
typedef struct __attribute__((packed)) { char c; int i; } packed_t;
typedef struct { char c; int i __attribute__((aligned(16))); } aligned_t;

__kernel_exec(1, float4) void m_exec(__global int *o)
{
	o[get_global_id(0)] = sizeof(packed_t) * 100 + sizeof(aligned_t);
}

__kernel __attribute__((reqd_work_group_size(1, 1, 1))) __attribute__((vec_type_hint(int)))
void m_reqd(__global int *o)
{
	o[get_global_id(0)] = get_local_size(0) + 10 * get_num_groups(0);
}
