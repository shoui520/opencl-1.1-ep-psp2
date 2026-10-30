__kernel void saxpy(float a, __global const float *x, __global float *y)
{
	size_t i = get_global_id(0);
	y[i] = mad(a, x[i], y[i]);
}

__kernel void fdiv_sqrt(__global const float *x, __global float *y)
{
	size_t i = get_global_id(0);
	float v = x[i];
	y[i] = sqrt(v) / (v + 1.0f) + rsqrt(v + 2.0f) - fabs(v) * 0.5f;
}
