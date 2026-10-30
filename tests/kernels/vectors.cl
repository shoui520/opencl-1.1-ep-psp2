__kernel void vec4(__global const float4 *a, __global const float4 *b, __global float4 *c, __global float *d)
{
	size_t i = get_global_id(0);
	float4 x = a[i], y = b[i];
	c[i] = x * y.yzwx + (float4)(1.0f, 2.0f, 3.0f, 4.0f);
	d[i] = dot(x, y);
}

__kernel void vload_store(__global const float *a, __global float *b)
{
	size_t i = get_global_id(0);
	float3 v = vload3(i, a);
	vstore3(v * 2.0f, i, b);
}
