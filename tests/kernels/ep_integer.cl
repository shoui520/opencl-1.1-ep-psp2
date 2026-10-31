/* Section 6.11.3: the integer functions, and section 6.3's operators, for
   each of the embedded profile's integer types (no long: section 10 item 1).
   Every result goes out as a uint, 26 per work-item. */

/* x / y and x % y with y nonzero and not (MIN / -1) */
#define DIVISOR(T, UT) \
	T d = y == (T)0 ? (T)1 : y; \
	if (d == (T)-1 && x == (T)((UT)1 << (sizeof(T) * 8 - 1))) \
		d = (T)1;

#define BODY(T, UT) \
	int i = get_global_id(0); \
	T x = a[i], y = b[i], z = c[i]; \
	T lo = min(y, z), hi = max(y, z); \
	int bits = sizeof(T) * 8; \
	DIVISOR(T, UT) \
	__global uint *r = o + 26 * i; \
	r[0] = (uint)abs(x); r[1] = (uint)abs_diff(x, y); r[2] = (uint)add_sat(x, y); \
	r[3] = (uint)hadd(x, y); r[4] = (uint)rhadd(x, y); r[5] = (uint)clamp(x, lo, hi); \
	r[6] = (uint)clz(x); r[7] = (uint)mad_hi(x, y, z); r[8] = (uint)mad_sat(x, y, z); \
	r[9] = (uint)max(x, y); r[10] = (uint)min(x, y); r[11] = (uint)mul_hi(x, y); \
	r[12] = (uint)rotate(x, y); r[13] = (uint)sub_sat(x, y); \
	r[14] = (uint)(x + y); r[15] = (uint)(x - y); r[16] = (uint)(x * y); \
	r[17] = (uint)(x / d); r[18] = (uint)(x % d); \
	r[19] = (uint)(x << (y & (bits - 1))); r[20] = (uint)(x >> (y & (bits - 1))); \
	r[21] = (uint)(~x ^ y); r[22] = (uint)((x & y) | z); r[23] = (uint)(-x);

/* 8 and 16-bit: upsample (table 6.10) */
#define NARROW(T, UT) __kernel void i_##T(__global const T *a, __global const T *b, \
					  __global const T *c, __global uint *o) \
	{ \
		BODY(T, UT) \
		r[24] = (uint)upsample(x, (UT)y); r[25] = (uint)upsample(y, (UT)z); \
	}

/* 32-bit: mul24 and mad24 (table 6.11), of 24-bit operands */
#define WIDE(T, UT) __kernel void i_##T(__global const T *a, __global const T *b, \
					__global const T *c, __global uint *o) \
	{ \
		BODY(T, UT) \
		T x24 = (x << 8) >> 8, y24 = (y << 8) >> 8; \
		r[24] = (uint)mul24(x24, y24); r[25] = (uint)mad24(x24, y24, z); \
	}

NARROW(char, uchar) NARROW(uchar, uchar) NARROW(short, ushort) NARROW(ushort, ushort)
WIDE(int, uint) WIDE(uint, uint)

/* The vector forms, with the sgentype forms of clamp, max and min, and the
   shifts with the count taken modulo the element's bits (6.3.j). 16 per
   work-item, as uint4. */
#define VEC(T) __kernel void v_##T(__global const T##4 *a, __global const T##4 *b, \
				   __global const T##4 *c, __global uint4 *o) \
	{ \
		int i = get_global_id(0); \
		T##4 x = a[i], y = b[i], z = c[i]; \
		T s = min(y.x, z.x), t = max(y.x, z.x); \
		__global uint4 *r = o + 16 * i; \
		r[0] = convert_uint4(abs(x)); r[1] = convert_uint4(add_sat(x, y)); \
		r[2] = convert_uint4(hadd(x, y)); r[3] = convert_uint4(rhadd(x, y)); \
		r[4] = convert_uint4(clamp(x, s, t)); r[5] = convert_uint4(max(x, y.y)); \
		r[6] = convert_uint4(min(x, y.z)); r[7] = convert_uint4(mul_hi(x, y)); \
		r[8] = convert_uint4(rotate(x, y)); r[9] = convert_uint4(sub_sat(x, y)); \
		r[10] = convert_uint4(mad_sat(x, y, z)); r[11] = convert_uint4(x << y); \
		r[12] = convert_uint4(x >> y); r[13] = convert_uint4(clz(x)); \
		r[14] = convert_uint4(abs_diff(x, y)); r[15] = convert_uint4(mad_hi(x, y, z)); \
	}

VEC(char) VEC(uchar) VEC(short) VEC(ushort) VEC(int) VEC(uint)
