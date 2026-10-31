/* Kernels for the harness's own API checks (tests/harness/ep_api.c). */

__kernel void inc(__global int *a)
{
	a[get_global_id(0)] += 1;
}

/* For clEnqueueTask, which runs one work-item (section 5.8). */
__kernel void task(__global uint *o)
{
	o[0] = get_work_dim();
	o[1] = get_global_size(0);
	o[2] = get_local_size(0);
	o[3] = get_num_groups(0);
	o[4] = get_global_id(0);
	o[5] = 0x7a5c;
}

__kernel void args(__global int *o, int k, __local int *l, __constant int *c)
{
	l[0] = k;
	o[get_global_id(0)] = l[0] + c[0];
}
