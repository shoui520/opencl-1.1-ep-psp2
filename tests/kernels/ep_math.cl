/* Section 6.11.2: every built-in math function, one result per work-item. */

#define F1(f) __kernel void m_##f(__global const float *x, __global float *y) \
	{ int i = get_global_id(0); y[i] = f(x[i]); }
#define F2(f) __kernel void m_##f(__global const float *x, __global const float *z, __global float *y) \
	{ int i = get_global_id(0); y[i] = f(x[i], z[i]); }
#define F3(f) __kernel void m_##f(__global const float *x, __global const float *z, \
				  __global const float *w, __global float *y) \
	{ int i = get_global_id(0); y[i] = f(x[i], z[i], w[i]); }
#define FI(f) __kernel void m_##f(__global const float *x, __global const int *n, __global float *y) \
	{ int i = get_global_id(0); y[i] = f(x[i], n[i]); }
/* The float4 version on the same data, four work-items' worth per item. */
#define V1(f) __kernel void v_##f(__global const float4 *x, __global float4 *y) \
	{ int i = get_global_id(0); y[i] = f(x[i]); }
#define V2(f) __kernel void v_##f(__global const float4 *x, __global const float4 *z, __global float4 *y) \
	{ int i = get_global_id(0); y[i] = f(x[i], z[i]); }

F1(acos) F1(acosh) F1(acospi) F1(asin) F1(asinh) F1(asinpi) F1(atan) F2(atan2) F1(atanh)
F1(atanpi) F2(atan2pi) F1(cbrt) F1(ceil) F2(copysign) F1(cos) F1(cosh) F1(cospi) F1(erfc)
F1(erf) F1(exp) F1(exp2) F1(exp10) F1(expm1) F1(fabs) F2(fdim) F1(floor) F3(fma) F2(fmax)
F2(fmin) F2(fmod) F2(hypot) FI(ldexp) F1(lgamma) F1(log) F1(log2) F1(log10) F1(log1p) F1(logb)
F3(mad) F2(maxmag) F2(minmag) F2(nextafter) F2(pow) FI(pown) F2(powr) F2(remainder) F1(rint)
FI(rootn) F1(round) F1(rsqrt) F1(sin) F1(sinh) F1(sinpi) F1(sqrt) F1(tan) F1(tanh) F1(tanpi)
F1(tgamma) F1(trunc)

F1(half_cos) F2(half_divide) F1(half_exp) F1(half_exp2) F1(half_exp10) F1(half_log)
F1(half_log2) F1(half_log10) F2(half_powr) F1(half_recip) F1(half_rsqrt) F1(half_sin)
F1(half_sqrt) F1(half_tan)

F1(native_cos) F2(native_divide) F1(native_exp) F1(native_exp2) F1(native_exp10) F1(native_log)
F1(native_log2) F1(native_log10) F2(native_powr) F1(native_recip) F1(native_rsqrt) F1(native_sin)
F1(native_sqrt) F1(native_tan)

V1(sin) V1(exp) V1(log) V1(sqrt) V1(floor) V1(atan) V2(pow) V2(fmod) V2(atan2) V2(fmax)

/* Section 10 item 6: correctly rounded, RTZ being the default rounding mode
   (item 5) where round to nearest isn't supported. */
__kernel void m_add(__global const float *x, __global const float *z, __global float *y)
{
	int i = get_global_id(0);
	y[i] = x[i] + z[i];
}

__kernel void m_sub(__global const float *x, __global const float *z, __global float *y)
{
	int i = get_global_id(0);
	y[i] = x[i] - z[i];
}

__kernel void m_mul(__global const float *x, __global const float *z, __global float *y)
{
	int i = get_global_id(0);
	y[i] = x[i] * z[i];
}

/* x / y and 1 / x, table 10.1's first rows */
__kernel void m_div(__global const float *x, __global const float *z, __global float *y)
{
	int i = get_global_id(0);
	y[i] = x[i] / z[i];
}

__kernel void m_recip(__global const float *x, __global float *y)
{
	int i = get_global_id(0);
	y[i] = 1.0f / x[i];
}

/* The functions with a second result through a pointer. */
__kernel void m_fract(__global const float *x, __global float *y, __global float *ip)
{
	int i = get_global_id(0);
	float t;

	y[i] = fract(x[i], &t);
	ip[i] = t;
}

__kernel void m_frexp(__global const float *x, __global float *y, __global int *e)
{
	int i = get_global_id(0);
	int t;

	y[i] = frexp(x[i], &t);
	e[i] = t;
}

__kernel void m_modf(__global const float *x, __global float *y, __global float *ip)
{
	int i = get_global_id(0);
	float t;

	y[i] = modf(x[i], &t);
	ip[i] = t;
}

__kernel void m_remquo(__global const float *x, __global const float *z, __global float *y,
		       __global int *q)
{
	int i = get_global_id(0);
	int t;

	y[i] = remquo(x[i], z[i], &t);
	q[i] = t;
}

__kernel void m_sincos(__global const float *x, __global float *y, __global float *c)
{
	int i = get_global_id(0);
	float t;

	y[i] = sincos(x[i], &t);
	c[i] = t;
}

__kernel void m_lgamma_r(__global const float *x, __global float *y, __global int *s)
{
	int i = get_global_id(0);
	int t;

	y[i] = lgamma_r(x[i], &t);
	s[i] = t;
}

/* The pointer may be to __private, __local or __global memory (6.11.2). */
__kernel void m_fract_spaces(__global const float *x, __global float *y, __global float *ip,
			     __global float *lp, __local float *l)
{
	int i = get_global_id(0);

	y[i] = fract(x[i], &ip[i]) + fract(x[i], l);
	lp[i] = l[0];
}

__kernel void m_ilogb(__global const float *x, __global int *y)
{
	int i = get_global_id(0);
	y[i] = ilogb(x[i]);
}

__kernel void m_nan(__global const uint *x, __global float *y)
{
	int i = get_global_id(0);
	y[i] = nan(x[i]);
}

/* The gentype f(gentype, float) forms (table 6.8). */
__kernel void v_scalar_forms(__global const float4 *x, __global const float *s, __global float4 *y)
{
	int i = get_global_id(0);
	y[3 * i] = fmax(x[i], s[i]);
	y[3 * i + 1] = fmin(x[i], s[i]);
	y[3 * i + 2] = ldexp(x[i], (int)s[i]);
}

/* Section 6.11.2.1's constants and macros, and ilogb's. */
__kernel void m_constants(__global float *f, __global int *n)
{
	f[0] = MAXFLOAT; f[1] = HUGE_VALF; f[2] = INFINITY; f[3] = NAN;
	f[4] = M_E_F; f[5] = M_LOG2E_F; f[6] = M_LOG10E_F; f[7] = M_LN2_F; f[8] = M_LN10_F;
	f[9] = M_PI_F; f[10] = M_PI_2_F; f[11] = M_PI_4_F; f[12] = M_1_PI_F; f[13] = M_2_PI_F;
	f[14] = M_2_SQRTPI_F; f[15] = M_SQRT2_F; f[16] = M_SQRT1_2_F;
	f[17] = FLT_MAX; f[18] = FLT_MIN; f[19] = FLT_EPSILON;
	n[0] = FLT_DIG; n[1] = FLT_MANT_DIG; n[2] = FLT_MAX_10_EXP; n[3] = FLT_MAX_EXP;
	n[4] = FLT_MIN_10_EXP; n[5] = FLT_MIN_EXP; n[6] = FLT_RADIX; n[7] = FP_ILOGB0;
	n[8] = FP_ILOGBNAN;
#ifdef FP_FAST_FMAF
	n[9] = 1;
#else
	n[9] = 0;
#endif
}
