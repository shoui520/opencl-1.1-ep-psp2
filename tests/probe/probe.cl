/* Which work-items run, and with what ids: hits[] counts each id (plain
 * read-modify-write), ids[] records the raw id, count[0] counts every item
 * with an atomic. */
__kernel void probe(__global uint *hits, __global uint *ids, __global uint *count)
{
	uint i = get_global_id(0);
	uint j = get_global_id(1);
	uint slot = (i + j * 64) & 1023;

	hits[slot] += 1;
	ids[slot] = i | (j << 16);
	atomic_inc(count);
}
