/* Whole programs of the kind OpenCL is used for, one work-item per output. */

/* C = A * B for row-major M x K and K x N matrices. */
__kernel void sgemm(__global const float *a, __global const float *b, __global float *c, int k)
{
	int col = get_global_id(0), row = get_global_id(1), n = get_global_size(0);
	float acc = 0.0f;
	int i;

	for (i = 0; i < k; i++)
		acc += a[row * k + i] * b[i * n + col];
	c[row * n + col] = acc;
}

/* The same with four columns per work-item. */
__kernel void sgemm4(__global const float *a, __global const float4 *b, __global float4 *c, int k)
{
	int col = get_global_id(0), row = get_global_id(1), n4 = get_global_size(0);
	float4 acc = (float4)(0.0f);
	int i;

	for (i = 0; i < k; i++)
		acc += a[row * k + i] * b[i * n4 + col];
	c[row * n4 + col] = acc;
}

/* 5x5 Gaussian blur of a float image, clamping at the edges. */
__constant float gauss5[5] = { 1.0f / 16, 4.0f / 16, 6.0f / 16, 4.0f / 16, 1.0f / 16 };

__kernel void blur5(__global const float *in, __global float *out)
{
	int x = get_global_id(0), y = get_global_id(1);
	int w = get_global_size(0), h = get_global_size(1);
	float acc = 0.0f;
	int dx, dy;

	for (dy = -2; dy <= 2; dy++)
		for (dx = -2; dx <= 2; dx++) {
			int sx = clamp(x + dx, 0, w - 1), sy = clamp(y + dy, 0, h - 1);
			acc += gauss5[dy + 2] * gauss5[dx + 2] * in[sy * w + sx];
		}
	out[y * w + x] = acc;
}

/* Sobel edge magnitude of an 8-bit image (|gx| + |gy|, saturated). */
__kernel void sobel(__global const uchar *in, __global uchar *out)
{
	int x = get_global_id(0), y = get_global_id(1);
	int w = get_global_size(0), h = get_global_size(1);
	int p[3][3], i, j, gx, gy;

	for (j = 0; j < 3; j++)
		for (i = 0; i < 3; i++)
			p[j][i] = in[clamp(y + j - 1, 0, h - 1) * w + clamp(x + i - 1, 0, w - 1)];
	gx = (p[0][2] + 2 * p[1][2] + p[2][2]) - (p[0][0] + 2 * p[1][0] + p[2][0]);
	gy = (p[2][0] + 2 * p[2][1] + p[2][2]) - (p[0][0] + 2 * p[0][1] + p[0][2]);
	out[y * w + x] = convert_uchar_sat(abs(gx) + abs(gy));
}

/* Mandelbrot iteration counts over [-2, 1] x [-1.2, 1.2]. */
__kernel void mandelbrot(__global int *out, int max_iter)
{
	int x = get_global_id(0), y = get_global_id(1);
	int w = get_global_size(0), h = get_global_size(1);
	float cr = -2.0f + 3.0f * x / w, ci = -1.2f + 2.4f * y / h;
	float zr = 0.0f, zi = 0.0f;
	int n = 0;

	while (n < max_iter && zr * zr + zi * zi <= 4.0f) {
		float t = zr * zr - zi * zi + cr;
		zi = 2.0f * zr * zi + ci;
		zr = t;
		n++;
	}
	out[y * w + x] = n;
}

/* One step of an all-pairs n-body simulation; pos.w is the mass. */
__kernel void nbody(__global const float4 *pos, __global const float4 *vel, __global float4 *pos_out,
		    __global float4 *vel_out, int n, float dt)
{
	int i = get_global_id(0), j;
	float4 p = pos[i], a = (float4)(0.0f);

	for (j = 0; j < n; j++) {
		float4 q = pos[j];
		float4 d = (float4)(q.xyz - p.xyz, 0.0f);
		float r2 = d.x * d.x + d.y * d.y + d.z * d.z + 0.01f;
		float inv = rsqrt(r2);
		a += d * (q.w * inv * inv * inv);
	}
	float4 v = vel[i] + a * dt;
	vel_out[i] = v;
	pos_out[i] = (float4)(p.xyz + v.xyz * dt, p.w);
}

/* y = A x for a CSR sparse matrix. */
__kernel void spmv_csr(__global const int *row_ptr, __global const int *cols,
		       __global const float *vals, __global const float *x, __global float *y)
{
	int row = get_global_id(0), k;
	float acc = 0.0f;

	for (k = row_ptr[row]; k < row_ptr[row + 1]; k++)
		acc += vals[k] * x[cols[k]];
	y[row] = acc;
}

/* Black-Scholes call and put prices. */
float cnd(float d)
{
	const float a1 = 0.31938153f, a2 = -0.356563782f, a3 = 1.781477937f;
	const float a4 = -1.821255978f, a5 = 1.330274429f;
	float k = 1.0f / (1.0f + 0.2316419f * fabs(d));
	float c = 0.3989422804f * exp(-0.5f * d * d) *
		  (k * (a1 + k * (a2 + k * (a3 + k * (a4 + k * a5)))));
	return d > 0.0f ? 1.0f - c : c;
}

__kernel void black_scholes(__global const float *spot, __global const float *strike,
			    __global const float *years, __global float *call, __global float *put,
			    float rate, float vol)
{
	int i = get_global_id(0);
	float s = spot[i], x = strike[i], t = years[i];
	float sq = vol * sqrt(t);
	float d1 = (log(s / x) + (rate + 0.5f * vol * vol) * t) / sq;
	float d2 = d1 - sq;
	float e = x * exp(-rate * t);

	call[i] = s * cnd(d1) - e * cnd(d2);
	put[i] = e * cnd(-d2) - s * cnd(-d1);
}

/* SHA-256 of one 55-byte message per work-item (a single padded block). */
__constant uint sha_k[64] = {
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

#define ROTR(x, n) rotate((x), (uint)(32 - (n)))

__kernel void sha256(__global const uchar *msg, __global uint *digest)
{
	int id = get_global_id(0), i;
	__global const uchar *m = msg + id * 55;
	uint w[64];
	uint h[8] = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
		      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
	uint a, b, c, d, e, f, g, hh;

	for (i = 0; i < 16; i++) {
		uint v = 0;
		int j;
		for (j = 0; j < 4; j++) {
			int k = 4 * i + j;
			uint byte = k < 55 ? m[k] : k == 55 ? 0x80 : 0;
			v = (v << 8) | byte;
		}
		w[i] = v;
	}
	w[15] = 55 * 8;
	for (i = 16; i < 64; i++) {
		uint s0 = ROTR(w[i - 15], 7) ^ ROTR(w[i - 15], 18) ^ (w[i - 15] >> 3);
		uint s1 = ROTR(w[i - 2], 17) ^ ROTR(w[i - 2], 19) ^ (w[i - 2] >> 10);
		w[i] = w[i - 16] + s0 + w[i - 7] + s1;
	}
	a = h[0]; b = h[1]; c = h[2]; d = h[3]; e = h[4]; f = h[5]; g = h[6]; hh = h[7];
	for (i = 0; i < 64; i++) {
		uint t1 = hh + (ROTR(e, 6) ^ ROTR(e, 11) ^ ROTR(e, 25)) + ((e & f) ^ (~e & g)) + sha_k[i] + w[i];
		uint t2 = (ROTR(a, 2) ^ ROTR(a, 13) ^ ROTR(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
		hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
	}
	digest[id * 8 + 0] = h[0] + a; digest[id * 8 + 1] = h[1] + b;
	digest[id * 8 + 2] = h[2] + c; digest[id * 8 + 3] = h[3] + d;
	digest[id * 8 + 4] = h[4] + e; digest[id * 8 + 5] = h[5] + f;
	digest[id * 8 + 6] = h[6] + g; digest[id * 8 + 7] = h[7] + hh;
}

/* 256-bin histogram of an 8-bit image, 16 pixels per work-item. */
__kernel void histogram(__global const uchar *in, __global uint *bins)
{
	int i, base = get_global_id(0) * 16;

	for (i = 0; i < 16; i++)
		atomic_inc(&bins[in[base + i]]);
}

/* One compare-exchange pass of a bitonic sort over n = 2 * global size keys. */
__kernel void bitonic_step(__global uint *keys, int stage, int pass)
{
	int i = get_global_id(0);
	int dist = 1 << (stage - pass);
	int lo = (i / dist) * 2 * dist + i % dist, hi = lo + dist;
	int up = ((i >> stage) & 1) == 0;
	uint a = keys[lo], b = keys[hi];

	if ((a > b) == up) {
		keys[lo] = b;
		keys[hi] = a;
	}
}

/* One pass of a Hillis-Steele inclusive prefix sum: out[i] = in[i] + in[i - d]. */
__kernel void scan_step(__global const int *in, __global int *out, int d)
{
	int i = get_global_id(0);

	out[i] = i >= d ? in[i] + in[i - d] : in[i];
}
