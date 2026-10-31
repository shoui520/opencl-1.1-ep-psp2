/* Section 6.11.12: shuffle and shuffle2. */

/* shuffle and shuffle2 of each n and m, masks from the input: the
   n-element vector (or mask) is the first n components.  One kernel for
   each: eight shuffles in one take minutes to compile and need more
   registers than the device has. */
#define P2(v) (v).s01
#define P4(v) (v).s0123
#define P8(v) (v).lo
#define P16(v) (v)
#define SHUF(T, U, n, m)								\
__kernel void s_##T##_##m##_##n(__global const T##16 *a, __global const T##16 *b,	\
				__global const U##16 *mask, __global T *r)		\
{											\
	int i = get_global_id(0);							\
	T##16 x = a[i], y = b[i];							\
	U##16 s = mask[i];								\
											\
	vstore##n(shuffle(P##m(x), P##n(s) & (U##n)(m - 1)), 0, r + 32 * i);		\
	vstore##n(shuffle2(P##m(x), P##m(y), P##n(s) & (U##n)(2 * m - 1)), 0,	\
		  r + 32 * i + 16);							\
}
#define SHUF_N(T, U, m) SHUF(T, U, 2, m) SHUF(T, U, 4, m) SHUF(T, U, 8, m) SHUF(T, U, 16, m)
#define SHUF_ALL(T, U) SHUF_N(T, U, 2) SHUF_N(T, U, 4) SHUF_N(T, U, 8) SHUF_N(T, U, 16)
#define SHUF_SOME(T, U) SHUF(T, U, 4, 4) SHUF(T, U, 16, 16)
SHUF_ALL(char, uchar) SHUF_ALL(int, uint) SHUF_ALL(float, uint)
SHUF_SOME(uchar, uchar) SHUF_SOME(short, ushort) SHUF_SOME(ushort, ushort) SHUF_SOME(uint, uint)
