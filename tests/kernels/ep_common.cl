/* Sections 6.11.4 (common), 6.11.5 (geometric) and 6.11.6 (relational
   functions), with the sgentype forms. */

/* x, y, z any floats, t in [0, 1]; lo < hi for smoothstep's edges */
__kernel void c_float(__global const float *a, __global const float *b, __global const float *c,
		      __global const float *d, __global float *o)
{
	int i = get_global_id(0);
	float x = a[i], y = b[i], z = c[i], t = d[i];
	float lo = fmin(y, z), hi = fmax(y, z);
	__global float *r = o + 9 * i;

	r[0] = clamp(x, lo, hi); r[1] = degrees(x); r[2] = radians(x);
	r[3] = max(x, y); r[4] = min(x, y); r[5] = mix(x, y, t);
	r[6] = step(y, x); r[7] = smoothstep(lo, hi, x); r[8] = sign(x);
}

/* the float4 forms, and the forms with a float for some arguments */
__kernel void c_float4(__global const float4 *a, __global const float4 *b, __global const float4 *c,
		       __global const float4 *d, __global float4 *o)
{
	int i = get_global_id(0);
	float4 x = a[i], y = b[i], z = c[i], t = d[i];
	float4 lo = fmin(y, z), hi = fmax(y, z);
	float s = lo.x, u = hi.x, w = t.x;
	__global float4 *r = o + 15 * i;

	r[0] = clamp(x, lo, hi); r[1] = degrees(x); r[2] = radians(x);
	r[3] = max(x, y); r[4] = min(x, y); r[5] = mix(x, y, t);
	r[6] = step(y, x); r[7] = smoothstep(lo, hi, x); r[8] = sign(x);
	r[9] = clamp(x, s, u); r[10] = max(x, s); r[11] = min(x, s);
	r[12] = mix(x, y, w); r[13] = step(s, x); r[14] = smoothstep(s, u, x);
}

/* each of floatn's n = 1, 2, 3 and 4: dot, distance, length, fast_distance,
   fast_length, then normalize and fast_normalize; then cross */
__kernel void g_float(__global const float4 *a, __global const float4 *b, __global float *o)
{
	int i = get_global_id(0);
	float4 p = a[i], q = b[i];
	__global float *r = o + 48 * i;

	r[0] = dot(p.x, q.x); r[1] = distance(p.x, q.x); r[2] = length(p.x);
	r[3] = fast_distance(p.x, q.x); r[4] = fast_length(p.x);
	r[5] = dot(p.xy, q.xy); r[6] = distance(p.xy, q.xy); r[7] = length(p.xy);
	r[8] = fast_distance(p.xy, q.xy); r[9] = fast_length(p.xy);
	r[10] = dot(p.xyz, q.xyz); r[11] = distance(p.xyz, q.xyz); r[12] = length(p.xyz);
	r[13] = fast_distance(p.xyz, q.xyz); r[14] = fast_length(p.xyz);
	r[15] = dot(p, q); r[16] = distance(p, q); r[17] = length(p);
	r[18] = fast_distance(p, q); r[19] = fast_length(p);

	r[20] = normalize(p.x);
	vstore2(normalize(p.xy), 0, r + 21);
	vstore3(normalize(p.xyz), 0, r + 23);
	vstore4(normalize(p), 0, r + 26);
	r[30] = fast_normalize(p.x);
	vstore2(fast_normalize(p.xy), 0, r + 31);
	vstore3(fast_normalize(p.xyz), 0, r + 33);
	vstore4(fast_normalize(p), 0, r + 36);

	vstore4(cross(p, q), 0, r + 40);
	vstore3(cross(p.xyz, q.xyz), 0, r + 44);
	r[47] = 0.0f;
}

/* the scalar relational functions give 1 for true, the operators too */
__kernel void r_float(__global const float *a, __global const float *b, __global int *o)
{
	int i = get_global_id(0);
	float x = a[i], y = b[i];
	__global int *r = o + 20 * i;

	r[0] = isequal(x, y); r[1] = isnotequal(x, y); r[2] = isgreater(x, y);
	r[3] = isgreaterequal(x, y); r[4] = isless(x, y); r[5] = islessequal(x, y);
	r[6] = islessgreater(x, y); r[7] = isfinite(x); r[8] = isinf(x); r[9] = isnan(x);
	r[10] = isnormal(x); r[11] = isordered(x, y); r[12] = isunordered(x, y);
	r[13] = signbit(x);
	r[14] = x == y; r[15] = x != y; r[16] = x < y; r[17] = x > y; r[18] = x <= y; r[19] = x >= y;
}

/* a vector's give -1 (all bits set) for true */
__kernel void r_float4(__global const float4 *a, __global const float4 *b, __global int4 *o)
{
	int i = get_global_id(0);
	float4 x = a[i], y = b[i];
	__global int4 *r = o + 20 * i;

	r[0] = isequal(x, y); r[1] = isnotequal(x, y); r[2] = isgreater(x, y);
	r[3] = isgreaterequal(x, y); r[4] = isless(x, y); r[5] = islessequal(x, y);
	r[6] = islessgreater(x, y); r[7] = isfinite(x); r[8] = isinf(x); r[9] = isnan(x);
	r[10] = isnormal(x); r[11] = isordered(x, y); r[12] = isunordered(x, y);
	r[13] = signbit(x);
	r[14] = x == y; r[15] = x != y; r[16] = x < y; r[17] = x > y; r[18] = x <= y; r[19] = x >= y;
}

/* any and all (the top bit of each component), bitselect and select (the
   top bit of a vector's c, a scalar c nonzero), as uints */
__kernel void r_select(__global const uint4 *a, __global const uint4 *b, __global const uint4 *c,
		       __global uint *o)
{
	int i = get_global_id(0);
	uint4 x = a[i], y = b[i], m = c[i];
	__global uint *r = o + 40 * i;

	r[0] = any(as_int4(m)); r[1] = all(as_int4(m));
	r[2] = any(as_char16(m)); r[3] = all(as_char16(m));
	r[4] = any(as_short8(m)); r[5] = all(as_short8(m));
	r[6] = any(as_int(m.x)); r[7] = all(as_int(m.x));
	r[8] = any(as_int2(m.xy)); r[9] = all(as_int3(m.xyz));
	vstore4(bitselect(x, y, m), 0, r + 10);
	vstore4(as_uint4(bitselect(as_float4(x), as_float4(y), as_float4(m))), 0, r + 14);
	vstore4(select(x, y, as_int4(m)), 0, r + 18);
	vstore4(as_uint4(select(as_float4(x), as_float4(y), as_int4(m))), 0, r + 22);
	vstore4(as_uint4(select(as_uchar16(x), as_uchar16(y), as_char16(m))), 0, r + 26);
	vstore4(as_uint4(select(as_short8(x), as_short8(y), as_ushort8(m))), 0, r + 30);
	r[34] = select(x.x, y.x, m.x); r[35] = as_uint(select(as_float(x.y), as_float(y.y), as_int(m.y)));
	r[36] = select(x.z, y.z, (uint)(m.z & 1)); r[37] = bitselect(x.w, y.w, m.w);
	r[38] = (uint)select((int)x.x, (int)y.x, (int)(m.z & 0x80000000));
	r[39] = 0;
}

/* the operators as branch conditions, which compile differently */
__kernel void r_branch(__global const float *a, __global const float *b, __global int *o)
{
	int i = get_global_id(0);
	float x = a[i], y = b[i];
	__global int *r = o + 6 * i;

	if (x == y)
		r[0] = 1;
	if (x != y)
		r[1] = 1;
	if (x < y)
		r[2] = 1;
	if (x > y)
		r[3] = 1;
	if (x <= y)
		r[4] = 1;
	if (x >= y)
		r[5] = 1;
}

/* the negated operators, which are LLVM's unordered comparisons (true for a
   NaN), and an ordered != */
__kernel void r_negated(__global const float *a, __global const float *b, __global int *o)
{
	int i = get_global_id(0);
	float x = a[i], y = b[i];
	__global int *r = o + 6 * i;

	r[0] = !(x < y); r[1] = !(x <= y); r[2] = !(x > y); r[3] = !(x >= y);
	r[4] = x < y || x > y; r[5] = !(x < y || x > y);
}
