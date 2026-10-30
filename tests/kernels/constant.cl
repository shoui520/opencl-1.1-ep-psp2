/* Program-scope __constant arrays indexed at run time. */
__constant int tab[6] = { 11, 22, 33, 44, 55, 66 };
__constant float ftab[16] = { 0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f,
			      8.5f, 9.5f, 10.5f, 11.5f, 12.5f, 13.5f, 14.5f, 15.5f };

__kernel void const_tab(__global const int *idx, __global int *o)
{
	int i = get_global_id(0);
	o[i] = tab[idx[i]];
}

__kernel void const_ftab(__global const int *idx, __global float *o)
{
	int i = get_global_id(0);
	o[i] = ftab[idx[i]] * 2.0f;
}

/* sha256.cl's message schedule alone. */
#define ROTR(x, n) rotate((x), (uint)(32 - (n)))

__kernel void sha_schedule(__global const uchar *msg, __global uint *out)
{
	int id = get_global_id(0), i;
	__global const uchar *m = msg + id * 55;
	uint w[64];

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
	for (i = 0; i < 64; i++)
		out[id * 64 + i] = w[i];
}

/* sha256 with the round constants in a __global buffer. */
__kernel void sha256_k(__global const uchar *msg, __global uint *digest, __global const uint *sha_k)
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


/* sha256_k's round functions alone, on random words. */
__kernel void sha_funcs(__global const uint *in, __global uint *out)
{
	int id = get_global_id(0);
	uint a = in[id * 3], b = in[id * 3 + 1], c = in[id * 3 + 2];

	out[id * 4 + 0] = ROTR(a, 6) ^ ROTR(a, 11) ^ ROTR(a, 25);
	out[id * 4 + 1] = (a & b) ^ (~a & c);
	out[id * 4 + 2] = ROTR(a, 2) ^ ROTR(a, 13) ^ ROTR(a, 22);
	out[id * 4 + 3] = (a & b) ^ (a & c) ^ (b & c);
}

/* n rounds of sha256_k on a given schedule, without the final addition. */
__kernel void sha_rounds(__global const uint *w, __global uint *out, __global const uint *sha_k, int n)
{
	int id = get_global_id(0), i;
	__global const uint *wi = w + id * 64;
	uint a = 0x6a09e667, b = 0xbb67ae85, c = 0x3c6ef372, d = 0xa54ff53a;
	uint e = 0x510e527f, f = 0x9b05688c, g = 0x1f83d9ab, hh = 0x5be0cd19;

	for (i = 0; i < n; i++) {
		uint t1 = hh + (ROTR(e, 6) ^ ROTR(e, 11) ^ ROTR(e, 25)) + ((e & f) ^ (~e & g)) + sha_k[i] + wi[i];
		uint t2 = (ROTR(a, 2) ^ ROTR(a, 13) ^ ROTR(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
		hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
	}
	out[id * 8 + 0] = a; out[id * 8 + 1] = b; out[id * 8 + 2] = c; out[id * 8 + 3] = d;
	out[id * 8 + 4] = e; out[id * 8 + 5] = f; out[id * 8 + 6] = g; out[id * 8 + 7] = hh;
}
