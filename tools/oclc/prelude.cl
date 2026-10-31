/*
 * Prepended to every program by oclc to work around IMG's OpenCL front end.
 *
 * The front end runs each work-item as its own work-group (get_local_size()
 * is 1), so nothing else can see an item's local memory and the local-memory
 * atomics are done as plain read-modify-write.  This also avoids two front
 * end bugs: a scalar __local variable whose address is taken shares its
 * register with its value, and atomics drop constant address offsets
 * (atomic_inc(&p[3]) operates on p[0]).  Global atomics keep the hardware
 * lock but get an offset the front end cannot fold: get_work_dim() is at
 * most 3, so the shift is always zero.
 */
#define __VITA_NOFOLD (get_work_dim() >> 8)

/* The front end defines this, but the device has no image support. */
#undef __IMAGE_SUPPORT__

/* The front end miscompiles an event_t passed by value ("Bitcast requires
 * types of same width"), so a copy can't be given an earlier copy's event.
 * The library's copies are done before they return, so the event is
 * ignorable. */
#define async_work_group_copy(dst, src, n, event)					\
	async_work_group_copy(dst, src, n, 0)
#define async_work_group_strided_copy(dst, src, n, stride, event)			\
	async_work_group_strided_copy(dst, src, n, stride, 0)

#define __VITA_ATOMIC_1(name, op, T)							\
	T __attribute__((overloadable)) __vita_##name(__global volatile T *p)		\
	{ return name(p + __VITA_NOFOLD); }						\
	T __attribute__((overloadable)) __vita_##name(__local volatile T *vp)		\
	{ __local T *p = (__local T *)vp; T o = *p; *p = op; return o; }
#define __VITA_ATOMIC_2(name, op, T)							\
	T __attribute__((overloadable)) __vita_##name(__global volatile T *p, T v)	\
	{ return name(p + __VITA_NOFOLD, v); }						\
	T __attribute__((overloadable)) __vita_##name(__local volatile T *vp, T v)	\
	{ __local T *p = (__local T *)vp; T o = *p; *p = op; return o; }
#define __VITA_ATOMIC_INT(def, name, op)						\
	def(name, op, int)								\
	def(name, op, uint)

__VITA_ATOMIC_INT(__VITA_ATOMIC_2, atomic_add, o + v)
__VITA_ATOMIC_INT(__VITA_ATOMIC_2, atomic_sub, o - v)
__VITA_ATOMIC_INT(__VITA_ATOMIC_2, atomic_xchg, v)
__VITA_ATOMIC_2(atomic_xchg, v, float)
__VITA_ATOMIC_INT(__VITA_ATOMIC_1, atomic_inc, o + 1)
__VITA_ATOMIC_INT(__VITA_ATOMIC_1, atomic_dec, o - 1)
__VITA_ATOMIC_INT(__VITA_ATOMIC_2, atomic_min, min(o, v))
__VITA_ATOMIC_INT(__VITA_ATOMIC_2, atomic_max, max(o, v))
__VITA_ATOMIC_INT(__VITA_ATOMIC_2, atomic_and, o & v)
__VITA_ATOMIC_INT(__VITA_ATOMIC_2, atomic_or, o | v)
__VITA_ATOMIC_INT(__VITA_ATOMIC_2, atomic_xor, o ^ v)

#define __VITA_CMPXCHG(T)								\
	T __attribute__((overloadable)) __vita_atomic_cmpxchg(__global volatile T *p, T c, T v) \
	{ return atomic_cmpxchg(p + __VITA_NOFOLD, c, v); }				\
	T __attribute__((overloadable)) __vita_atomic_cmpxchg(__local volatile T *vp, T c, T v) \
	{ __local T *p = (__local T *)vp; T o = *p; *p = o == c ? v : o; return o; }
__VITA_CMPXCHG(int)
__VITA_CMPXCHG(uint)

#define atomic_add(p, v)	__vita_atomic_add(p, v)
#define atomic_sub(p, v)	__vita_atomic_sub(p, v)
#define atomic_xchg(p, v)	__vita_atomic_xchg(p, v)
#define atomic_inc(p)		__vita_atomic_inc(p)
#define atomic_dec(p)		__vita_atomic_dec(p)
#define atomic_cmpxchg(p, c, v)	__vita_atomic_cmpxchg(p, c, v)
#define atomic_min(p, v)	__vita_atomic_min(p, v)
#define atomic_max(p, v)	__vita_atomic_max(p, v)
#define atomic_and(p, v)	__vita_atomic_and(p, v)
#define atomic_or(p, v)		__vita_atomic_or(p, v)
#define atomic_xor(p, v)	__vita_atomic_xor(p, v)
#define atom_add(p, v)		__vita_atomic_add(p, v)
#define atom_sub(p, v)		__vita_atomic_sub(p, v)
#define atom_xchg(p, v)		__vita_atomic_xchg(p, v)
#define atom_inc(p)		__vita_atomic_inc(p)
#define atom_dec(p)		__vita_atomic_dec(p)
#define atom_cmpxchg(p, c, v)	__vita_atomic_cmpxchg(p, c, v)
#define atom_min(p, v)		__vita_atomic_min(p, v)
#define atom_max(p, v)		__vita_atomic_max(p, v)
#define atom_and(p, v)		__vita_atomic_and(p, v)
#define atom_or(p, v)		__vita_atomic_or(p, v)
#define atom_xor(p, v)		__vita_atomic_xor(p, v)

/*
 * The front end crashes on max and min of uchar and ushort and their
 * vectors, and USC fails on max, min, clamp, abs and clz of 8-bit integers;
 * a definition here replaces the built-in's overload, doing it in int.
 */
#define __VITA_OVL __attribute__((overloadable))
#define __VITA_MINMAX(T, n)								\
	T##n __VITA_OVL max(T##n a, T##n b)						\
	{ return convert_##T##n(max(convert_int##n(a), convert_int##n(b))); }		\
	T##n __VITA_OVL min(T##n a, T##n b)						\
	{ return convert_##T##n(min(convert_int##n(a), convert_int##n(b))); }
#define __VITA_CLAMP(T, n)								\
	T##n __VITA_OVL clamp(T##n x, T##n lo, T##n hi)					\
	{ return convert_##T##n(clamp(convert_int##n(x), convert_int##n(lo), convert_int##n(hi))); }
#define __VITA_SMINMAX(T, n)								\
	T##n __VITA_OVL max(T##n a, T b) { return max(a, (T##n)b); }			\
	T##n __VITA_OVL min(T##n a, T b) { return min(a, (T##n)b); }
#define __VITA_SCLAMP(T, n)								\
	T##n __VITA_OVL clamp(T##n x, T lo, T hi) { return clamp(x, (T##n)lo, (T##n)hi); }
#define __VITA_ABS(n)									\
	uchar##n __VITA_OVL abs(char##n x) { return convert_uchar##n(abs(convert_int##n(x))); }
#define __VITA_CLZ(n)									\
	uchar##n __VITA_OVL clz(uchar##n x)						\
	{ return convert_uchar##n(clz(convert_uint##n(x)) - 24); }			\
	char##n __VITA_OVL clz(char##n x) { return as_char##n(clz(as_uchar##n(x))); }
#define __VITA_EACH_N(def, ...)								\
	def(__VA_ARGS__, ) def(__VA_ARGS__, 2) def(__VA_ARGS__, 3) def(__VA_ARGS__, 4)	\
	def(__VA_ARGS__, 8) def(__VA_ARGS__, 16)
#define __VITA_VECS(def) def(2) def(3) def(4) def(8) def(16)
#define __VITA_EACH_VEC(def, ...)							\
	def(__VA_ARGS__, 2) def(__VA_ARGS__, 3) def(__VA_ARGS__, 4)			\
	def(__VA_ARGS__, 8) def(__VA_ARGS__, 16)

__VITA_EACH_N(__VITA_MINMAX, char) __VITA_EACH_N(__VITA_MINMAX, uchar)
__VITA_EACH_N(__VITA_MINMAX, ushort)
__VITA_EACH_VEC(__VITA_SMINMAX, char) __VITA_EACH_VEC(__VITA_SMINMAX, uchar)
__VITA_EACH_VEC(__VITA_SMINMAX, ushort)
__VITA_EACH_N(__VITA_CLAMP, char) __VITA_EACH_N(__VITA_CLAMP, uchar)
__VITA_EACH_VEC(__VITA_SCLAMP, char) __VITA_EACH_VEC(__VITA_SCLAMP, uchar)
__VITA_CLZ() __VITA_VECS(__VITA_CLZ)
__VITA_ABS() __VITA_VECS(__VITA_ABS)

/*
 * The library's 32-bit rhadd loses the carry of x + y + 1, clamp of uint,
 * mad_sat of int and uint and mul_hi and mad_hi of int vectors are wrong:
 * these are built from what is right (uint mul_hi, int scalar mul_hi, uint
 * max and min), with carries and overflow found by bit arithmetic so a
 * vector's comparisons (-1 for true) don't differ from a scalar's (1).
 */
#define __VITA_NZ(v)	(-(((v) | -(v)) >> 31))	/* of a uint: all ones if v != 0 */
#define __VITA_CARRY(a, b, s)	((((a) & (b)) | (((a) | (b)) & ~(s))) >> 31)
#define __VITA_RHADD(T, n)								\
	T##n __VITA_OVL rhadd(T##n x, T##n y) { return (x >> 1) + (y >> 1) + ((x | y) & 1); }
#define __VITA_UCLAMP(n)								\
	uint##n __VITA_OVL clamp(uint##n x, uint##n lo, uint##n hi) { return min(max(x, lo), hi); }
#define __VITA_SUCLAMP(n)								\
	uint##n __VITA_OVL clamp(uint##n x, uint lo, uint hi)				\
	{ return min(max(x, (uint##n)lo), (uint##n)hi); }
#define __VITA_MULHI(n)									\
	int##n __VITA_OVL mul_hi(int##n x, int##n y)					\
	{ return as_int##n(mul_hi(as_uint##n(x), as_uint##n(y))) - ((x >> 31) & y) - ((y >> 31) & x); } \
	int##n __VITA_OVL mad_hi(int##n x, int##n y, int##n z) { return mul_hi(x, y) + z; }
#define __VITA_MADSAT(n)								\
	uint##n __VITA_OVL mad_sat(uint##n x, uint##n y, uint##n z)			\
	{										\
		uint##n lo = x * y, s = lo + z;						\
		return s | __VITA_NZ(mul_hi(x, y) | __VITA_CARRY(lo, z, s));		\
	}										\
	int##n __VITA_OVL mad_sat(int##n x, int##n y, int##n z)				\
	{										\
		uint##n lo = as_uint##n(x * y), zu = as_uint##n(z), s = lo + zu;	\
		int##n hi = mul_hi(x, y) + (z >> 31) + as_int##n(__VITA_CARRY(lo, zu, s)); \
		int##n ov = as_int##n(__VITA_NZ(as_uint##n(hi ^ (as_int##n(s) >> 31))));	\
		return (as_int##n(s) & ~ov) | (((hi >> 31) ^ 0x7fffffff) & ov);		\
	}

__VITA_EACH_N(__VITA_RHADD, int) __VITA_EACH_N(__VITA_RHADD, uint)
__VITA_UCLAMP() __VITA_VECS(__VITA_UCLAMP) __VITA_VECS(__VITA_SUCLAMP)
__VITA_VECS(__VITA_MULHI)
__VITA_MADSAT() __VITA_VECS(__VITA_MADSAT)

/*
 * The front end's as_type from a vector of 8 or 16-bit integers to a type of
 * wider elements packs them big-endian (as_uint(ushort2) is x << 16 | y),
 * though the other way and memory are little-endian.  Those are assembled
 * here, the even components low; as_type between elements of one width is
 * right and gives the signed and float types.
 */
#define __VITA_AS16(n, m)								\
	ushort##m __VITA_OVL as_ushort##m(uchar##n v)					\
	{ return convert_ushort##m(v.even) | convert_ushort##m(v.odd) << (ushort##m)8; }		\
	ushort##m __VITA_OVL as_ushort##m(char##n v) { return as_ushort##m(as_uchar##n(v)); } \
	short##m __VITA_OVL as_short##m(uchar##n v) { return as_short##m(as_ushort##m(v)); } \
	short##m __VITA_OVL as_short##m(char##n v) { return as_short##m(as_ushort##m(as_uchar##n(v))); }
#define __VITA_AS32_FROM(S, n, m)							\
	int##m __VITA_OVL as_int##m(S##n v) { return as_int##m(as_uint##m(v)); }	\
	float##m __VITA_OVL as_float##m(S##n v) { return as_float##m(as_uint##m(v)); }
#define __VITA_AS32B(n, m)								\
	uint##m __VITA_OVL as_uint##m(uchar##n v)					\
	{										\
		return convert_uint##m(v.even.even) | convert_uint##m(v.odd.even) << (uint##m)8 |	\
		       convert_uint##m(v.even.odd) << (uint##m)16 |	\
		       convert_uint##m(v.odd.odd) << (uint##m)24; \
	}										\
	uint##m __VITA_OVL as_uint##m(char##n v) { return as_uint##m(as_uchar##n(v)); }	\
	__VITA_AS32_FROM(uchar, n, m) __VITA_AS32_FROM(char, n, m)
#define __VITA_AS32H(n, m)								\
	uint##m __VITA_OVL as_uint##m(ushort##n v)					\
	{ return convert_uint##m(v.even) | convert_uint##m(v.odd) << (uint##m)16; }		\
	uint##m __VITA_OVL as_uint##m(short##n v) { return as_uint##m(as_ushort##n(v)); } \
	__VITA_AS32_FROM(ushort, n, m) __VITA_AS32_FROM(short, n, m)

__VITA_AS16(2, ) __VITA_AS16(4, 2) __VITA_AS16(8, 4) __VITA_AS16(16, 8)
__VITA_AS32B(4, ) __VITA_AS32B(8, 2) __VITA_AS32B(16, 4)
__VITA_AS32H(2, ) __VITA_AS32H(4, 2) __VITA_AS32H(8, 4) __VITA_AS32H(16, 8)

/*
 * The library's int and uint to float conversions rounding to nearest, up
 * or down find the exponent from the hardware's log2, which is approximate
 * (16777215 becomes 33554430).  The conversion toward zero is right, and so
 * is the remainder it leaves, so these round from there: magnitude u
 * truncated to f, then the next float up where the mode says (mode 0: to
 * nearest even, 1: away from zero, 2: toward zero).
 */
#define __VITA_CVF(n)									\
	float##n __VITA_OVL __vita_u2f(uint##n u, int mode)				\
	{										\
		float##n f = convert_float##n##_rtz(u);					\
		uint##n d = u - convert_uint##n##_rtz(f);				\
		uint##n e = as_uint##n(f) >> (uint##n)23;				\
		/* half an ulp of f, where its ulp is 2 or more (e > 150) */		\
		uint##n hu = select((uint##n)0, (uint##n)1 << (e - (uint##n)151), e > (uint##n)150); \
		int##n up = d != (uint##n)0;						\
		if (mode == 0)								\
			up = (d > hu) | ((d == hu) & up & ((as_uint##n(f) & (uint##n)1) != (uint##n)0)); \
		else if (mode == 2)							\
			up = (int##n)0;							\
		return select(f, as_float##n(as_uint##n(f) + (uint##n)1), up);		\
	}										\
	float##n __VITA_OVL convert_float##n##_rte(uint##n u) { return __vita_u2f(u, 0); } \
	float##n __VITA_OVL convert_float##n##_rtp(uint##n u) { return __vita_u2f(u, 1); } \
	float##n __VITA_OVL convert_float##n##_rtn(uint##n u) { return __vita_u2f(u, 2); } \
	float##n __VITA_OVL convert_float##n##_rte(int##n x)				\
	{ float##n m = __vita_u2f(abs(x), 0); return select(m, -m, x < (int##n)0); }	\
	float##n __VITA_OVL convert_float##n##_rtp(int##n x)				\
	{ return select(__vita_u2f(abs(x), 1), -__vita_u2f(abs(x), 2), x < (int##n)0); } \
	float##n __VITA_OVL convert_float##n##_rtn(int##n x)				\
	{ return select(__vita_u2f(abs(x), 2), -__vita_u2f(abs(x), 1), x < (int##n)0); }

/* 8 and 16 from halves: a select of 8 floats is a vmovc repeated 8 times */
#define __VITA_CVF_HALVES(n, h, m)							\
	float##n __VITA_OVL convert_float##n##_##m(uint##n u)				\
	{ return (float##n)(convert_float##h##_##m(u.lo), convert_float##h##_##m(u.hi)); } \
	float##n __VITA_OVL convert_float##n##_##m(int##n x)				\
	{ return (float##n)(convert_float##h##_##m(x.lo), convert_float##h##_##m(x.hi)); }
#define __VITA_CVF_SPLIT(n, h)								\
	__VITA_CVF_HALVES(n, h, rte) __VITA_CVF_HALVES(n, h, rtp) __VITA_CVF_HALVES(n, h, rtn)

__VITA_CVF() __VITA_CVF(2) __VITA_CVF(3) __VITA_CVF(4)
__VITA_CVF_SPLIT(8, 4) __VITA_CVF_SPLIT(16, 8)

/*
 * Math functions the library gets wrong, from other built-ins that are right
 * (a definition here replaces the built-in, so it can't call it).
 *
 * ldexp overflows to infinity, where rounding toward zero (the device's
 * mode) gives FLT_MAX: here it is three multiplies by powers of two, exact
 * but for the last's rounding, which the device's multiply gets right.
 */
#define __VITA_LDEXP(n)									\
	float##n __VITA_OVL ldexp(float##n x, int##n k)					\
	{										\
		int##n c = clamp(k, -300, 300), a = c / (int##n)3, b = (c - a) / (int##n)2; \
		int##n d = c - a - b;							\
		return x * as_float##n((a + (int##n)127) << (int##n)23) *		\
		       as_float##n((b + (int##n)127) << (int##n)23) *			\
		       as_float##n((d + (int##n)127) << (int##n)23);			\
	}
#define __VITA_LDEXP_S(n)								\
	float##n __VITA_OVL ldexp(float##n x, int k) { return ldexp(x, (int##n)k); }
__VITA_LDEXP() __VITA_VECS(__VITA_LDEXP) __VITA_VECS(__VITA_LDEXP_S)

/*
 * nextafter(-0, +0) is +0 (x == y gives y), not FLT_MIN.  Flushing
 * sub-normals, this keeps to section 7.5.3's results for nextafter, as the
 * library does: from a zero, the smallest normal; to a sub-normal, a zero.
 */
float __VITA_OVL nextafter(float x, float y)
{
	int ix = as_int(x);
	float r;

	if (isnan(x) || isnan(y))
		return x + y;
	if (x == y)
		return y;
	if (x == 0.0f)		/* compared with x alike: y may be flushed */
		return y > x ? FLT_MIN : y < x ? -FLT_MIN : y;
	r = as_float(ix + ((x < y) == (x > 0.0f) ? 1 : -1));
	return fabs(r) < FLT_MIN ? copysign(0.0f, x) : r;
}
float2 __VITA_OVL nextafter(float2 x, float2 y)
{ return (float2)(nextafter(x.x, y.x), nextafter(x.y, y.y)); }
float3 __VITA_OVL nextafter(float3 x, float3 y)
{ return (float3)(nextafter(x.xy, y.xy), nextafter(x.z, y.z)); }
float4 __VITA_OVL nextafter(float4 x, float4 y)
{ return (float4)(nextafter(x.lo, y.lo), nextafter(x.hi, y.hi)); }
float8 __VITA_OVL nextafter(float8 x, float8 y)
{ return (float8)(nextafter(x.lo, y.lo), nextafter(x.hi, y.hi)); }
float16 __VITA_OVL nextafter(float16 x, float16 y)
{ return (float16)(nextafter(x.lo, y.lo), nextafter(x.hi, y.hi)); }

/*
 * asin of a normal below 2^-12 is x itself, but the library's gives 0 for
 * FLT_MIN.  Defined before the macro, so asin here is the library's.
 */
#define __VITA_ASIN(n)									\
	float##n __VITA_OVL __vita_asin(float##n x)					\
	{ return select(asin(x), x, fabs(x) < (float##n)0x1p-12f); }
float __VITA_OVL __vita_asin(float x) { return fabs(x) < 0x1p-12f ? x : asin(x); }
__VITA_VECS(__VITA_ASIN)
#define asin(x) __vita_asin(x)

/*
 * lgamma_r doesn't store the sign of gamma(x): the value is lgamma's, the
 * sign -1 between a negative odd and the even integer above it.
 */
#define __VITA_LGR(n, AS)								\
	float##n __VITA_OVL lgamma_r(float##n x, AS int##n *s)				\
	{										\
		int##n odd = (convert_int##n(floor(x)) & (int##n)1) != (int##n)0;	\
		*s = select((int##n)1, select((int##n)1, (int##n)-1, odd), x < (float##n)0); \
		return lgamma(x);							\
	}
#define __VITA_LGR_AS(n) __VITA_LGR(n, __global) __VITA_LGR(n, __local) __VITA_LGR(n, __private)
__VITA_LGR_AS() __VITA_VECS(__VITA_LGR_AS)

/*
 * sinpi and cospi: x reduced exactly to f in [0, 1/2) of a quarter turn
 * q, then sin or cos of pi times at most 1/4, so the exact zeros (7.5.1)
 * are exact and cospi(-0) is 1, not 0x1.fffffep-1.  The library's cos and sin
 * are right for such arguments; pi f is fma(f, pi, f * (pi - (float)pi)).
 */
float __vita_sincospi(float x, int cosine)
{
	float a = fabs(x), r, f, t, th, v;
	int q, swap;

	if (!isfinite(x))
		return x - x;
	r = a - 2.0f * floor(a * 0.5f);		/* in [0, 2), exact */
	q = (int)(r * 2.0f);
	f = r - (float)q * 0.5f;
	q += cosine;				/* cos(pi r) = sin(pi (r + 1/2)) */
	swap = f > 0.25f;
	t = swap ? 0.5f - f : f;
	th = fma(t, M_PI_F, t * -8.742278e-8f);
	/* q 0: sin(pi f), 1: cos(pi f), 2: -sin(pi f), 3: -cos(pi f) */
	if ((q & 1) ^ swap)
		v = t == 0.0f ? 1.0f : cos(th);
	else
		v = t == 0.0f ? 0.0f : sin(th);
	if (q & 2)
		v = -v;
	if (f == 0.0f && ((q & 1) == 0))
		v = 0.0f;			/* sinpi(n) and cospi(n + 1/2): +0 */
	return cosine || !signbit(x) ? v : -v;
}
#define __VITA_SINCOSPI_V(n, h)								\
	float##n __VITA_OVL sinpi(float##n x) { return (float##n)(sinpi(x.lo), sinpi(x.hi)); } \
	float##n __VITA_OVL cospi(float##n x) { return (float##n)(cospi(x.lo), cospi(x.hi)); }
float __VITA_OVL sinpi(float x) { return __vita_sincospi(x, 0); }
float __VITA_OVL cospi(float x) { return __vita_sincospi(x, 1); }
float2 __VITA_OVL sinpi(float2 x) { return (float2)(sinpi(x.x), sinpi(x.y)); }
float2 __VITA_OVL cospi(float2 x) { return (float2)(cospi(x.x), cospi(x.y)); }
float3 __VITA_OVL sinpi(float3 x) { return (float3)(sinpi(x.xy), sinpi(x.z)); }
float3 __VITA_OVL cospi(float3 x) { return (float3)(cospi(x.xy), cospi(x.z)); }
__VITA_SINCOSPI_V(4, 2) __VITA_SINCOSPI_V(8, 4) __VITA_SINCOSPI_V(16, 8)

/*
 * sin, cos, tan and sincos: the library's lose all accuracy past a few
 * thousand (sin(1e10) is far off) and half_cos, half_sin and half_tan are
 * past 8192 ulp.  x is reduced by pi/2 in integers (Payne and Hanek): its
 * 24-bit mantissa times 96 bits of 2/pi from the place its exponent needs
 * gives the quadrant and 64 bits of the fraction, times pi/2 in 64 bits,
 * as two floats.  On those, polynomials fit to [-pi/4, pi/4]; tan's divide
 * is corrected by its exact remainder (split in 12-bit halves) so the
 * rounding toward zero stays inside 5 ulp.
 */
/* 2/pi's bits after a zero word */
uint __vita_2bypi(int i)
{
	return i == 1 ? 0xa2f9836eu : i == 2 ? 0x4e441529u : i == 3 ? 0xfc2757d1u :
	       i == 4 ? 0xf534ddc0u : i == 5 ? 0xdb629599u : i == 6 ? 0x3c439041u :
	       i == 7 ? 0xfe5163abu : 0u;
}
uint __vita_bits(uint a, uint b, int sh)	/* a:b shifted left sh (0 to 31), the top word */
{
	return (a << sh) | ((b >> 1) >> (31 - sh));
}
float __vita_pow2(int e)			/* normal 2^e */
{
	return as_float((uint)(e + 127) << 23);
}
float __vita_hi12(float x)
{
	return as_float(as_uint(x) & 0xfffff000u);
}
/* n / d: the quotient's error by n - d q, exact from 12-bit halves */
float __vita_div(float n, float d)
{
	float sd = fabs(d) < 0x1p-100f ? 0x1p64f : 1.0f, q, dh, dl, qh, ql, e;

	d *= sd;			/* d's low half mustn't flush */
	q = n / d;
	dh = __vita_hi12(d);
	dl = d - dh;
	qh = __vita_hi12(q);
	ql = q - qh;
	e = (((n - dh * qh) - dh * ql) - dl * qh) - dl * ql;
	return (q + e / d) * sd;
}
/* fn 0 sin, 1 cos, 2 tan */
float __vita_trig(float x, int fn)
{
	uint ix = as_uint(x) & 0x7fffffffu, sign = as_uint(x) ^ ix;
	float rh = as_float(ix), rl = 0.0f, w, s, c, v;
	int q = 0;

	if (ix >= 0x7f800000u)
		return x - x;
	if (ix > 0x3f490fdau) {		/* past pi/4 */
		int o = (int)(ix >> 23) - 120, i = o >> 5, sh = o & 31, n, k, ex;
		uint m = (ix & 0x7fffffu) | 0x800000u;
		uint t0 = __vita_2bypi(i), t1 = __vita_2bypi(i + 1);
		uint t2 = __vita_2bypi(i + 2), t3 = __vita_2bypi(i + 3);
		uint w0 = __vita_bits(t0, t1, sh), w1 = __vita_bits(t1, t2, sh), w2 = __vita_bits(t2, t3, sh);
		/* m w is 2^94 times x 2/pi, less multiples of 4 */
		uint p0 = m * w2, cy = mul_hi(m, w2), p1 = m * w1 + cy;
		uint p2 = m * w0 + mul_hi(m, w1) + (p1 < cy ? 1u : 0u);
		uint fh = (p2 << 2) | (p1 >> 30), fl = (p1 << 2) | (p0 >> 30), neg = 0;
		uint hh, hl, l, a;

		q = (int)(p2 >> 30);
		if (fh & 0x80000000u) {		/* past a half: less than the next quadrant */
			fl = 0u - fl;
			fh = ~fh + (fl == 0u ? 1u : 0u);
			q++;
			neg = 0x80000000u;
		}
		ex = 0;
		if (fh == 0u) {
			fh = fl;
			fl = 0u;
			ex = 32;
		}
		n = (int)clz(fh);
		if (n) {
			fh = (fh << n) | (fl >> (32 - n));
			fl <<= n;
		}
		ex += n;
		/* times pi/4 in 64 bits */
		hh = mul_hi(fh, 0xc90fdaa2u);
		hl = fh * 0xc90fdaa2u;
		a = mul_hi(fh, 0x2168c235u);
		l = hl + a;
		hh += l < a ? 1u : 0u;
		a = mul_hi(fl, 0xc90fdaa2u);
		l += a;
		hh += l < a ? 1u : 0u;
		k = 0;
		if (!(hh & 0x80000000u)) {
			hh = (hh << 1) | (l >> 31);
			l <<= 1;
			k = 1;
		}
		rh = as_float(as_uint((float)(hh >> 8) * __vita_pow2(-23 - ex - k)) | neg);
		rl = as_float(as_uint((float)(((hh & 0xffu) << 16) | (l >> 16)) * __vita_pow2(-47 - ex - k)) | neg);
	}
	w = rh * rh;
	s = rh + (rh * w * (-0x1.555556p-3f + w * (0x1.11110ep-7f + w * (-0x1.a013a8p-13f +
	    w * 0x1.6dbe08p-19f))) + rl * (1.0f - 0.5f * w));
	c = 1.0f - (0.5f * w - (w * w * (0x1.555556p-5f + w * (-0x1.6c16cp-10f + w * (0x1.a015c4p-16f +
	    w * -0x1.25244ep-22f))) - rh * rl));
	q += fn == 1 ? 1 : 0;			/* cos(r) = sin(r + pi/2) */
	if (fn == 2)
		v = q & 1 ? -__vita_div(c, s) : __vita_div(s, c);
	else
		v = q & 1 ? c : s;
	if (fn != 2 && (q & 2))
		v = -v;
	return fn == 1 ? v : as_float(as_uint(v) ^ sign);
}
float __vita_sin1(float x) { return __vita_trig(x, 0); }
float __vita_cos1(float x) { return __vita_trig(x, 1); }
float __vita_tan1(float x) { return __vita_trig(x, 2); }

/*
 * erf and erfc: the library's are NaN at infinities and wrong past 1.5
 * (erf(2.5) is 6.77).  Below 0.84375 erf is x + x P(x^2); above, erfc is
 * exp(-x^2) G(1/x) / x, G fit on four spans about their middles, with
 * exp(-x^2) as exp(-z^2) exp((z - x)(z + x)) for z x's top 12 bits
 * (z^2 is exact).
 */
#define __VITA_PICK(p, a, b, c, d) (p == 0 ? a : p == 1 ? b : p == 2 ? c : d)
float __vita_erfc_big(float a)		/* erfc of a in [0.84375, 10.1) */
{
	int p = a < 1.25f ? 0 : a < 2.0f ? 1 : a < 3.5f ? 2 : 3;
	float t = 1.0f / a, z = __vita_hi12(a), g;
	float u = t - __VITA_PICK(p, 0x1.fc3518p-1f, 0x1.4cccccp-1f, 0x1.924924p-2f, 0x1.89f524p-3f);

	g = __VITA_PICK(p, 0x1.41e208p-8f, -0x1.2bbc7ap-6f, -0x1.dc17d2p-3f, 0x1.c2494p-2f);
	g = g * u + __VITA_PICK(p, -0x1.5566a2p-7f, -0x1.89f452p-7f, 0x1.5a1d76p-3f, 0x1.555bdp-2f);
	g = g * u + __VITA_PICK(p, 0x1.0ad2b2p-6f, 0x1.87d028p-5f, -0x1.2e880ep-6f, -0x1.a600acp-2f);
	g = g * u + __VITA_PICK(p, -0x1.36b1d2p-6f, -0x1.26c81p-4f, -0x1.e693bep-4f, 0x1.235c36p-4f);
	g = g * u + __VITA_PICK(p, 0x1.1daa76p-7f, 0x1.089efep-4f, 0x1.59d4e8p-3f, 0x1.b906bp-3f);
	g = g * u + __VITA_PICK(p, 0x1.2188c6p-5f, 0x1.f5c742p-9f, -0x1.53044ep-4f, -0x1.a54c7ap-3f);
	g = g * u + __VITA_PICK(p, -0x1.3d3a6ap-3f, -0x1.5f45dcp-3f, -0x1.3cd66p-3f, -0x1.910cd2p-4f);
	g = g * u + __VITA_PICK(p, 0x1.b704bep-2f, 0x1.f0e19p-2f, 0x1.0e666ap-1f, 0x1.1bcafap-1f);
	return exp(-z * z) * exp((z - a) * (z + a)) * (g * t);
}
float __vita_erf_small(float x)		/* x P(x^2) of |x| below 0.84375 */
{
	float w = x * x;

	return x * (0x1.06eba8p-3f + w * (-0x1.812746p-2f + w * (0x1.ce2eb6p-4f + w * (-0x1.b81e9ep-6f +
		w * (0x1.55735p-8f + w * (-0x1.b0ce5cp-11f + w * 0x1.74dbap-14f))))));
}
float __vita_erf1(float x)
{
	float a = fabs(x);

	if (isnan(x))
		return x;
	if (a < 0x1p-60f)
		return x * 0x1.20dd76p+0f;	/* 2/sqrt(pi): x P(x^2) would flush */
	if (a < 0.84375f)
		return x + __vita_erf_small(x);
	return copysign(a < 6.0f ? 1.0f - __vita_erfc_big(a) : 1.0f, x);
}
float __vita_erfc1(float x)
{
	float a = fabs(x);

	if (isnan(x))
		return x;
	if (a < 0.25f)
		return 1.0f - (x + __vita_erf_small(x));
	if (a < 0.84375f)
		return x < 0.0f ? 1.0f + (a + __vita_erf_small(a)) : 0.5f - ((x - 0.5f) + __vita_erf_small(x));
	if (x < 0.0f)
		return a < 6.0f ? 2.0f - __vita_erfc_big(a) : 2.0f;
	return x < 10.1f ? __vita_erfc_big(x) : 0.0f;
}

/*
 * tgamma: the library's is 0 everywhere.  x is reduced exactly to f in
 * [0, 1) and gamma(x) is gamma(1 + f) times, or over, the product of the
 * steps between, each step exact, kept in two floats (split in 12-bit
 * halves) and scaled by 2^-64 past 2^64; 1/gamma(1 + f), fit about f = 1/2,
 * has no poles.  Between -1 and 1, gamma(x) is 1/(x r(x)), r(x) as
 * (1 + x) r(1 + x) for negative x.
 */
float __vita_rgamma1(float f)		/* 1/gamma(1 + f), f in [0, 1] */
{
	float u = f - 0.5f;

	return 0x1.20dd76p+0f + u * (-0x1.514d3ep-5f + u * (-0x1.0da5a6p-1f + u * (0x1.669be4p-3f +
	       u * (0x1.a185fp-5f + u * (-0x1.595628p-5f + u * (0x1.b1254p-8f + u * (0x1.163788p-9f +
	       u * (-0x1.1ee024p-10f + u * 0x1.2d561p-13f))))))));
}
float __vita_tgamma1(float x)
{
	float a = fabs(x), n = floor(x), f, r, ph, pl, s, a1, a2, p1, p2, hi, lo, v;
	int k, m, sc, big;

	if (isnan(x) || x == -INFINITY || (x < 0.0f && x == n))
		return NAN;
	if (a < FLT_MIN)			/* a zero, or flushed */
		return copysign(INFINITY, x);
	if (a < 0x1p-100f)			/* 1/x - 0.577..: x r would flush */
		return __vita_div(1.0f, x);
	if (x > 36.0f)
		return INFINITY;
	if (x < -50.0f)		/* below FLT_MIN even beside the integers */
		return ((int)n & 1) ? -0.0f : 0.0f;
	if (a < 1.0f) {
		r = x < 0.0f ? (1.0f + x) * __vita_rgamma1(1.0f + x) : __vita_rgamma1(x);
		return __vita_div(1.0f, x * r);
	}
	f = x - n;
	r = __vita_rgamma1(f);
	/* x > 1: steps f + 1 to x - 1; x < -1: x to x + (-n - 1), and f */
	big = x > 0.0f;
	m = big ? (int)n - 1 : (int)-n;
	ph = big ? 1.0f : f;
	pl = 0.0f;
	sc = 0;
	for (k = 0; k < m; k++) {
		s = big ? f + (float)(k + 1) : x + (float)k;
		a1 = __vita_hi12(s);
		a2 = s - a1;
		p1 = __vita_hi12(ph);
		p2 = ph - p1;
		hi = p1 * a1;
		lo = (p1 * a2 + p2 * a1) + p2 * a2 + pl * s;
		ph = hi + lo;
		pl = lo - (ph - hi);
		if (fabs(ph) > 0x1p64f) {
			ph *= 0x1p-64f;
			pl *= 0x1p-64f;
			sc++;
		}
	}
	if (big)
		v = __vita_div(ph, r) + pl / r;
	else
		v = __vita_div(1.0f, ph * r + pl * r);
	for (k = 0; k < sc; k++)
		v *= big ? 0x1p64f : 0x1p-64f;
	return v;
}

/*
 * rootn: exp2(log2(x) / n) loses the low bits of log2(x)'s exponent part
 * (16 ulp at rootn(-6.4e-37, 3)).  x = m 2^e and e = q |n| + r: the root
 * is 2^(q sign(n)) exactly times exp2(sign(n) (r + log2(m)) / |n|), whose
 * argument is under 1.
 */
float __vita_rootn1(float x, int n)
{
	uint ix = as_uint(x) & 0x7fffffffu, sign = as_uint(x) ^ ix;
	int k = n == INT_MIN ? INT_MAX - 1 : n < 0 ? -n : n, e, q, r;	/* INT_MIN's root is INT_MAX - 1's */
	float t, v;

	if (n == 0 || ix > 0x7f800000u)
		return NAN;
	if (!(n & 1) && ix < 0x00800000u)
		sign = 0u;
	if (ix < 0x00800000u)			/* a zero, or flushed */
		return as_float((n < 0 ? 0x7f800000u : 0u) | sign);
	if (sign && !(n & 1))
		return NAN;
	if (ix == 0x7f800000u)
		return as_float((n > 0 ? 0x7f800000u : 0u) | sign);
	e = (int)(ix >> 23) - 127;
	q = e >= 0 ? e / k : -1 - (-e - 1) / k;
	r = e - q * k;				/* 0 to k - 1 */
	t = ((float)r + log2(as_float((ix & 0x7fffffu) | 0x3f800000u))) / (float)k;
	v = exp2(n < 0 ? -t : t);
	q = n < 0 ? -q : q;
	v *= __vita_pow2(q / 2);
	v *= __vita_pow2(q - q / 2);
	return as_float(as_uint(v) | sign);
}

/* vectors by halves; three as two and one */
#define __VITA_V1(f, g)									\
	float2 __VITA_OVL f(float2 x) { return (float2)(g(x.x), g(x.y)); }		\
	float3 __VITA_OVL f(float3 x) { return (float3)(f(x.xy), g(x.z)); }		\
	float4 __VITA_OVL f(float4 x) { return (float4)(f(x.lo), f(x.hi)); }		\
	float8 __VITA_OVL f(float8 x) { return (float8)(f(x.lo), f(x.hi)); }		\
	float16 __VITA_OVL f(float16 x) { return (float16)(f(x.lo), f(x.hi)); }
#define __VITA_F1(f, g) float __VITA_OVL f(float x) { return g(x); } __VITA_V1(f, g)
__VITA_F1(__vita_sin, __vita_sin1) __VITA_F1(__vita_cos, __vita_cos1)
__VITA_F1(__vita_tan, __vita_tan1) __VITA_F1(__vita_erf, __vita_erf1)
__VITA_F1(__vita_erfc, __vita_erfc1) __VITA_F1(__vita_tgamma, __vita_tgamma1)
#define __VITA_SINCOS(n, AS)								\
	float##n __VITA_OVL __vita_sincos(float##n x, AS float##n *c)			\
	{ *c = __vita_cos(x); return __vita_sin(x); }
#define __VITA_SINCOS_AS(n) __VITA_SINCOS(n, __global) __VITA_SINCOS(n, __local)	\
	__VITA_SINCOS(n, __private)
__VITA_SINCOS_AS() __VITA_VECS(__VITA_SINCOS_AS)
float __VITA_OVL __vita_rootn(float x, int n) { return __vita_rootn1(x, n); }
float2 __VITA_OVL __vita_rootn(float2 x, int2 n)
{ return (float2)(__vita_rootn1(x.x, n.x), __vita_rootn1(x.y, n.y)); }
float3 __VITA_OVL __vita_rootn(float3 x, int3 n)
{ return (float3)(__vita_rootn(x.xy, n.xy), __vita_rootn1(x.z, n.z)); }
float4 __VITA_OVL __vita_rootn(float4 x, int4 n)
{ return (float4)(__vita_rootn(x.lo, n.lo), __vita_rootn(x.hi, n.hi)); }
float8 __VITA_OVL __vita_rootn(float8 x, int8 n)
{ return (float8)(__vita_rootn(x.lo, n.lo), __vita_rootn(x.hi, n.hi)); }
float16 __VITA_OVL __vita_rootn(float16 x, int16 n)
{ return (float16)(__vita_rootn(x.lo, n.lo), __vita_rootn(x.hi, n.hi)); }
#define sin(x) __vita_sin(x)
#define cos(x) __vita_cos(x)
#define tan(x) __vita_tan(x)
#define sincos(x, c) __vita_sincos(x, c)
#define erf(x) __vita_erf(x)
#define erfc(x) __vita_erfc(x)
#define tgamma(x) __vita_tgamma(x)
#define rootn(x, n) __vita_rootn(x, n)
/* half_cos, half_sin and half_tan are past 8192 ulp */
#define __VITA_HALF_TRIG(n)								\
	float##n __VITA_OVL half_cos(float##n x) { return cos(x); }			\
	float##n __VITA_OVL half_sin(float##n x) { return sin(x); }			\
	float##n __VITA_OVL half_tan(float##n x) { return tan(x); }
__VITA_HALF_TRIG() __VITA_VECS(__VITA_HALF_TRIG)

/*
 * The scalar normalize loses the sign (normalize(-2.0f) is 1), fast_normalize
 * of a zero vector isn't p (6.11.5), the w of cross(float4) isn't 0, the
 * scalar smoothstep is far off its formula, and select of 8 and 16-bit
 * vectors picks wrongly: these from what is right.
 */
float __VITA_OVL normalize(float p) { return p == 0.0f || isnan(p) ? p : copysign(1.0f, p); }
#define __VITA_FASTNORM(n)								\
	float##n __VITA_OVL fast_normalize(float##n p)					\
	{										\
		float s = dot(p, p);							\
		return s < FLT_MIN ? p : p * half_rsqrt(s);				\
	}
__VITA_FASTNORM() __VITA_FASTNORM(2) __VITA_FASTNORM(3) __VITA_FASTNORM(4)
float4 __VITA_OVL cross(float4 p, float4 q) { return (float4)(cross(p.xyz, q.xyz), 0.0f); }
float __VITA_OVL smoothstep(float e0, float e1, float x)
{
	float t = clamp((x - e0) / (e1 - e0), 0.0f, 1.0f);
	return t * t * (3.0f - 2.0f * t);
}
/* int's select tests the top bit: c's, shifted there */
#define __VITA_SELECT(T, C, n, shift)							\
	T##n __VITA_OVL select(T##n a, T##n b, C##n c)					\
	{ return convert_##T##n(select(convert_int##n(a), convert_int##n(b),		\
				       convert_int##n(c) << (int##n)shift)); }
#define __VITA_SELECT8(n)								\
	__VITA_SELECT(char, char, n, 24) __VITA_SELECT(char, uchar, n, 24)		\
	__VITA_SELECT(uchar, char, n, 24) __VITA_SELECT(uchar, uchar, n, 24)
#define __VITA_SELECT16(n)								\
	__VITA_SELECT(short, short, n, 16) __VITA_SELECT(short, ushort, n, 16)		\
	__VITA_SELECT(ushort, short, n, 16) __VITA_SELECT(ushort, ushort, n, 16)
/* wider than 4 is too many for one vmovc: by halves */
#define __VITA_SELECT_SPLIT(T, C, n)							\
	T##n __VITA_OVL select(T##n a, T##n b, C##n c)					\
	{ return (T##n)(select(a.lo, b.lo, c.lo), select(a.hi, b.hi, c.hi)); }
#define __VITA_SELECT_SPLITS(n)								\
	__VITA_SELECT_SPLIT(char, char, n) __VITA_SELECT_SPLIT(char, uchar, n)		\
	__VITA_SELECT_SPLIT(uchar, char, n) __VITA_SELECT_SPLIT(uchar, uchar, n)	\
	__VITA_SELECT_SPLIT(short, short, n) __VITA_SELECT_SPLIT(short, ushort, n)	\
	__VITA_SELECT_SPLIT(ushort, short, n) __VITA_SELECT_SPLIT(ushort, ushort, n)
__VITA_SELECT8(2) __VITA_SELECT8(3) __VITA_SELECT8(4)
__VITA_SELECT16(2) __VITA_SELECT16(3) __VITA_SELECT16(4)
__VITA_SELECT_SPLITS(8) __VITA_SELECT_SPLITS(16)

/*
 * shuffle and shuffle2 (6.11.12): the front end reads the component at a
 * dynamic index with a temp register as the index, which both USCs ignore
 * (every result component is the first), and a private array in a function
 * crashes it, so each component is picked by selects, on ints (the 8 and
 * 16-bit selects above take minutes to compile this many of).
 */
/* i counts down to 0 at the component taken: comparing with 0 each time
   needs no constant for each component (a char16 shuffle of 16 runs out) */
#define __VITA_SH_X(n, k)								\
	i -= (int##n)1; r = select(r, (int##n)v.s##k, i == (int##n)0);
#define __VITA_SH_X2(n) __VITA_SH_X(n, 1)
#define __VITA_SH_X4(n) __VITA_SH_X2(n) __VITA_SH_X(n, 2) __VITA_SH_X(n, 3)
#define __VITA_SH_X8(n)									\
	__VITA_SH_X4(n) __VITA_SH_X(n, 4) __VITA_SH_X(n, 5) __VITA_SH_X(n, 6) __VITA_SH_X(n, 7)
#define __VITA_SH_X16(n)								\
	__VITA_SH_X8(n) __VITA_SH_X(n, 8) __VITA_SH_X(n, 9) __VITA_SH_X(n, a)		\
	__VITA_SH_X(n, b) __VITA_SH_X(n, c) __VITA_SH_X(n, d) __VITA_SH_X(n, e)	\
	__VITA_SH_X(n, f)
#define __VITA_SH_Y(n, m, k)								\
	i -= (int##n)1; r = select(r, (int##n)w.s##k, i == (int##n)0);
#define __VITA_SH_Y2(n, m) __VITA_SH_Y(n, m, 0) __VITA_SH_Y(n, m, 1)
#define __VITA_SH_Y4(n, m) __VITA_SH_Y2(n, m) __VITA_SH_Y(n, m, 2) __VITA_SH_Y(n, m, 3)
#define __VITA_SH_Y8(n, m)								\
	__VITA_SH_Y4(n, m) __VITA_SH_Y(n, m, 4) __VITA_SH_Y(n, m, 5)			\
	__VITA_SH_Y(n, m, 6) __VITA_SH_Y(n, m, 7)
#define __VITA_SH_Y16(n, m)								\
	__VITA_SH_Y8(n, m) __VITA_SH_Y(n, m, 8) __VITA_SH_Y(n, m, 9)			\
	__VITA_SH_Y(n, m, a) __VITA_SH_Y(n, m, b) __VITA_SH_Y(n, m, c)			\
	__VITA_SH_Y(n, m, d) __VITA_SH_Y(n, m, e) __VITA_SH_Y(n, m, f)
/* to and from ints: convert for the integer types, as_ for float */
#define __VITA_SHUF(T, U, n, m, to)							\
	T##n __VITA_OVL __vita_shuffle(T##m x, U##n mask)				\
	{										\
		int##m v = to##_int##m(x);						\
		int##n i = convert_int##n(mask & (U##n)(m - 1)), r = (int##n)v.s0;	\
		__VITA_SH_X##m(n)							\
		return to##_##T##n(r);							\
	}										\
	T##n __VITA_OVL __vita_shuffle2(T##m x, T##m y, U##n mask)			\
	{										\
		int##m v = to##_int##m(x), w = to##_int##m(y);				\
		int##n i = convert_int##n(mask & (U##n)(2 * m - 1)), r = (int##n)v.s0;	\
		__VITA_SH_X##m(n) __VITA_SH_Y##m(n, m)					\
		return to##_##T##n(r);							\
	}
/* 8 and 16 results by halves: all at once they spill to scratch memory */
#define __VITA_SHUF_HALVES(T, U, n, m)							\
	T##n __VITA_OVL __vita_shuffle(T##m x, U##n mask)				\
	{ return (T##n)(__vita_shuffle(x, mask.lo), __vita_shuffle(x, mask.hi)); }	\
	T##n __VITA_OVL __vita_shuffle2(T##m x, T##m y, U##n mask)			\
	{ return (T##n)(__vita_shuffle2(x, y, mask.lo), __vita_shuffle2(x, y, mask.hi)); }
#define __VITA_SHUF_M(T, U, m, to)							\
	__VITA_SHUF(T, U, 2, m, to) __VITA_SHUF(T, U, 4, m, to)				\
	__VITA_SHUF_HALVES(T, U, 8, m) __VITA_SHUF_HALVES(T, U, 16, m)
#define __VITA_SHUF_T(T, U, to)								\
	__VITA_SHUF_M(T, U, 2, to) __VITA_SHUF_M(T, U, 4, to)				\
	__VITA_SHUF_M(T, U, 8, to) __VITA_SHUF_M(T, U, 16, to)
__VITA_SHUF_T(char, uchar, convert) __VITA_SHUF_T(uchar, uchar, convert)
__VITA_SHUF_T(short, ushort, convert) __VITA_SHUF_T(ushort, ushort, convert)
__VITA_SHUF_T(int, uint, convert) __VITA_SHUF_T(uint, uint, convert)
__VITA_SHUF_T(float, uint, as)
#define shuffle(x, mask) __vita_shuffle(x, mask)
#define shuffle2(x, y, mask) __vita_shuffle2(x, y, mask)

/*
 * vload_half and vstore_half (6.11.7): USC can't load, store or convert f16
 * (ssa.c and icvt_core.c abort), so they're done on the bits, exactly and
 * with denormals.  Modes: 0 rte (the default), 1 rtz, 2 rtp, 3 rtn.
 */
/* h's exponent 0 is a denormal, m 2^-24s; 31 an infinity or NaN */
#define __VITA_H2F_DEF(n)								\
	float##n __VITA_OVL __vita_h2f(uint##n h)					\
	{										\
		uint##n s = (h & 0x8000) << 16, e = (h >> 10) & 31, v = (h & 0x7fff) << 13;	\
		uint##n d = as_uint##n(convert_float##n(h & 0x3ff) * 5.9604644775390625e-8f);	\
		return as_float##n(s | select(select(v + 0x38000000, d, e == 0),	\
					      v + 0x70000000, e == 31));		\
	}
/* m's top bits are the half truncated toward zero, t, and its low sh bits
   the part dropped, r, against half of the last place, hh.  Past 65520
   rounds as 65520 + a bit does; 25 puts any float below half of 2^-24. */
#define __VITA_F2H_DEF(n)								\
	uint##n __VITA_OVL __vita_f2h(float##n x, int mode)				\
	{										\
		uint##n u = as_uint##n(x), s = (u >> 16) & 0x8000, a = u & 0x7fffffff;	\
		uint##n c = min(a, (uint##n)0x477ff001), one = 1, zero = 0;		\
		uint##n m = select((c & 0x7fffff) | select(zero, (uint##n)0x800000, c >= 0x800000), \
				   c - 0x38000000, c >= 0x38800000);			\
		uint##n sh = select(min((uint##n)126 - (c >> 23), (uint##n)25),	\
				    (uint##n)13, c >= 0x38800000);			\
		uint##n t = m >> sh, r = m & ((one << sh) - 1), hh = one << (sh - 1);	\
		uint##n up = mode == 0 ? select(zero, one, (r > hh) | ((r == hh) & ((t & 1) != 0))) : \
			     mode == 2 ? select(zero, one, (r != 0) & (s == 0)) :		\
			     mode == 3 ? select(zero, one, (r != 0) & (s != 0)) : zero;	\
		uint##n h = select(s | (t + up), s | 0x7c00, a == 0x7f800000);		\
		return select(h, s | 0x7e00 | ((a >> 13) & 0x3ff), a > 0x7f800000);	\
	}
__VITA_H2F_DEF() __VITA_H2F_DEF(2) __VITA_H2F_DEF(3) __VITA_H2F_DEF(4)
__VITA_F2H_DEF() __VITA_F2H_DEF(2) __VITA_F2H_DEF(3) __VITA_F2H_DEF(4)

#define __VITA_H2F1(q) __vita_h2f((uint)(q)[0])
#define __VITA_H2F2(q) __vita_h2f((uint2)((q)[0], (q)[1]))
#define __VITA_H2F3(q) __vita_h2f((uint3)((q)[0], (q)[1], (q)[2]))
#define __VITA_H2F4(q) __vita_h2f((uint4)((q)[0], (q)[1], (q)[2], (q)[3]))
#define __VITA_H2F8(q) (float8)(__VITA_H2F4(q), __VITA_H2F4((q) + 4))
#define __VITA_H2F16(q) (float16)(__VITA_H2F8(q), __VITA_H2F8((q) + 8))
#define __VITA_F2H2(q, v, m) { uint2 h_ = __vita_f2h(v, m); (q)[0] = h_.x; (q)[1] = h_.y; }
#define __VITA_F2H3(q, v, m)								\
	{ uint3 h_ = __vita_f2h(v, m); (q)[0] = h_.x; (q)[1] = h_.y; (q)[2] = h_.z; }
#define __VITA_F2H4(q, v, m)								\
	{ uint4 h_ = __vita_f2h(v, m); (q)[0] = h_.x; (q)[1] = h_.y; (q)[2] = h_.z; (q)[3] = h_.w; }
#define __VITA_F2H8(q, v, m) __VITA_F2H4(q, (v).lo, m) __VITA_F2H4((q) + 4, (v).hi, m)
#define __VITA_F2H16(q, v, m) __VITA_F2H8(q, (v).lo, m) __VITA_F2H8((q) + 8, (v).hi, m)

/* vloada_half3 and vstorea_half3 step 4 halves */
#define __VITA_VLH(AS, n, a)								\
	float##n __VITA_OVL vload_half##n(size_t i, const AS half *p)			\
	{ return __VITA_H2F##n((const AS ushort *)p + i * n); }			\
	float##n __VITA_OVL vloada_half##n(size_t i, const AS half *p)			\
	{ return __VITA_H2F##n((const AS ushort *)p + i * a); }
#define __VITA_VLH_AS(AS)								\
	float __VITA_OVL vload_half(size_t i, const AS half *p)				\
	{ return __VITA_H2F1((const AS ushort *)p + i); }				\
	float __VITA_OVL vloada_half(size_t i, const AS half *p)			\
	{ return __VITA_H2F1((const AS ushort *)p + i); }				\
	__VITA_VLH(AS, 2, 2) __VITA_VLH(AS, 3, 4) __VITA_VLH(AS, 4, 4)			\
	__VITA_VLH(AS, 8, 8) __VITA_VLH(AS, 16, 16)
__VITA_VLH_AS(__global) __VITA_VLH_AS(__local) __VITA_VLH_AS(__constant)
__VITA_VLH_AS(__private)

#define __VITA_VSH(AS, n, a, sfx, m)							\
	void __VITA_OVL vstore_half##n##sfx(float##n v, size_t i, AS half *p)		\
	{ AS ushort *q = (AS ushort *)p + i * n; __VITA_F2H##n(q, v, m) }		\
	void __VITA_OVL vstorea_half##n##sfx(float##n v, size_t i, AS half *p)		\
	{ AS ushort *q = (AS ushort *)p + i * a; __VITA_F2H##n(q, v, m) }
#define __VITA_VSH_MODE(AS, sfx, m)							\
	void __VITA_OVL vstore_half##sfx(float v, size_t i, AS half *p)			\
	{ ((AS ushort *)p)[i] = __vita_f2h(v, m); }					\
	void __VITA_OVL vstorea_half##sfx(float v, size_t i, AS half *p)		\
	{ ((AS ushort *)p)[i] = __vita_f2h(v, m); }					\
	__VITA_VSH(AS, 2, 2, sfx, m) __VITA_VSH(AS, 3, 4, sfx, m)			\
	__VITA_VSH(AS, 4, 4, sfx, m) __VITA_VSH(AS, 8, 8, sfx, m)			\
	__VITA_VSH(AS, 16, 16, sfx, m)
#define __VITA_VSH_AS(AS)								\
	__VITA_VSH_MODE(AS, , 0) __VITA_VSH_MODE(AS, _rte, 0)				\
	__VITA_VSH_MODE(AS, _rtz, 1) __VITA_VSH_MODE(AS, _rtp, 2)			\
	__VITA_VSH_MODE(AS, _rtn, 3)
__VITA_VSH_AS(__global) __VITA_VSH_AS(__local) __VITA_VSH_AS(__private)
