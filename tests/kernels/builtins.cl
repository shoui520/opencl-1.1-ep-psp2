__kernel void transcendental(__global const float *x, __global float *y)
{
	size_t i = get_global_id(0);
	float v = x[i];
	y[8 * i + 0] = sin(v);
	y[8 * i + 1] = cos(v);
	y[8 * i + 2] = exp(v);
	y[8 * i + 3] = log(fabs(v) + 1.0f);
	y[8 * i + 4] = pow(fabs(v), 1.5f);
	y[8 * i + 5] = fmod(v, 3.0f);
	y[8 * i + 6] = floor(v) + ceil(v) + round(v) + trunc(v);
	y[8 * i + 7] = atan2(v, 2.0f) + tan(v * 0.1f);
}
