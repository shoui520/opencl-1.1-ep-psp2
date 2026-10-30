__kernel void workitem(__global uint *o)
{
	size_t gx = get_global_id(0), gy = get_global_id(1);
	size_t idx = gy * get_global_size(0) + gx;
	o[8 * idx + 0] = get_global_id(0);
	o[8 * idx + 1] = get_global_id(1);
	o[8 * idx + 2] = get_local_id(0);
	o[8 * idx + 3] = get_local_id(1);
	o[8 * idx + 4] = get_group_id(0);
	o[8 * idx + 5] = get_num_groups(1);
	o[8 * idx + 6] = get_local_size(0);
	o[8 * idx + 7] = get_work_dim() + get_global_offset(0);
}
