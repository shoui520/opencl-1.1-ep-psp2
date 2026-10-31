/* Section 6.2.3: explicit conversions between every pair of the embedded
   profile's scalar types (no long: section 10 item 1), with every modifier;
   section 6.2.4: as_type. */

/* to an integer type: the default, _sat, the four rounding modes and both */
#define TO_INT(d, s) __kernel void cv_##d##_##s(__global const s *x, __global d *y) \
	{ \
		int i = get_global_id(0); \
		__global d *o = y + 10 * i; \
		o[0] = convert_##d(x[i]); o[1] = convert_##d##_sat(x[i]); \
		o[2] = convert_##d##_rte(x[i]); o[3] = convert_##d##_rtz(x[i]); \
		o[4] = convert_##d##_rtp(x[i]); o[5] = convert_##d##_rtn(x[i]); \
		o[6] = convert_##d##_sat_rte(x[i]); o[7] = convert_##d##_sat_rtz(x[i]); \
		o[8] = convert_##d##_sat_rtp(x[i]); o[9] = convert_##d##_sat_rtn(x[i]); \
	}
/* to float: _sat may not be used (6.2.3.3) */
#define TO_FLOAT(s) __kernel void cv_float_##s(__global const s *x, __global float *y) \
	{ \
		int i = get_global_id(0); \
		__global float *o = y + 5 * i; \
		o[0] = convert_float(x[i]); o[1] = convert_float_rte(x[i]); \
		o[2] = convert_float_rtz(x[i]); o[3] = convert_float_rtp(x[i]); \
		o[4] = convert_float_rtn(x[i]); \
	}
#define FROM(s) TO_INT(char, s) TO_INT(uchar, s) TO_INT(short, s) TO_INT(ushort, s) \
	TO_INT(int, s) TO_INT(uint, s) TO_FLOAT(s)

FROM(char) FROM(uchar) FROM(short) FROM(ushort) FROM(int) FROM(uint) FROM(float)

/* the vector forms, n = 2, 3, 4, 8 and 16 (6.2.3), each element of x once */
__kernel void cv_vectors(__global const float *x, __global const int *k, __global int *y,
			 __global float *f, __global uchar *u)
{
	int i = get_global_id(0);
	__global const float *p = x + 33 * i;
	__global const int *q = k + 33 * i;

	vstore2(convert_int2_sat_rte(vload2(0, p)), 0, y + 33 * i);
	vstore3(convert_int3(vload3(0, p + 2)), 0, y + 33 * i + 2);
	vstore4(convert_int4_rtp(vload4(0, p + 5)), 0, y + 33 * i + 5);
	vstore8(convert_float8(vload8(0, q + 9)), 0, f + 33 * i + 9);
	vstore16(convert_uchar16_sat(vload16(0, q + 17)), 0, u + 33 * i + 17);
	vstore8(convert_float8_rtn(vload8(0, q + 9)), 0, f + 33 * i + 25);
}

/* 6.2.4.2: as_type keeps the bits, between types of the same size */
__kernel void as_types(__global const uint4 *x, __global uint4 *y)
{
	int i = get_global_id(0);
	uint4 v = x[i];
	float4 f = as_float4(v);
	int4 n = as_int4(f);
	uchar16 c = as_uchar16(n);
	short8 s = as_short8(c);
	float2 h = as_float2(s.lo);

	y[4 * i] = as_uint4(f);
	y[4 * i + 1] = as_uint4(c);
	y[4 * i + 2] = (uint4)(as_uint(f.x), as_uint(as_float(v.y)), as_uint(h.x), as_uint(h.y));
	y[4 * i + 3] = (uint4)(as_uint(as_uchar4(v.z)), as_uint(as_short2(v.w)),
			       as_uint(as_char4(as_float(v.x))), as_uint(as_ushort2(f.w)));
}
