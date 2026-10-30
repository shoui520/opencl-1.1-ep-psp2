#include "gxc_kernel.h"

#include <string.h>

/* Work-items per draw: CHUNK_X vertices by up to CHUNK_POINTS / x instances. */
#define CHUNK_X		4096
#define CHUNK_POINTS	65536
#define SCENE_DRAWS	64		/* draws per scene at most */
#define SCRATCH_MAX	(8u << 20)	/* scratch memory for one scene */

#define ALIGN(x, a)	(((x) + ((a) - 1)) & ~((a) - 1))

#define MAP_MAGIC	0x314d4b56	/* "VKM1" */
#define MAP_LITERAL	0x80000000u	/* a channel naming a value folded in by the USP */

static int ensure_iota(gxc *g)
{
	uint32_t *idx;
	int err, i;

	if (g->iota.base)
		return 0;
	err = gxc_mem_alloc(&g->iota, SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE, CHUNK_X * 4,
			    GXC_MEM_GPU);
	if (err < 0)
		return err;
	idx = g->iota.base;
	for (i = 0; i < CHUNK_X; i++)
		idx[i] = i;
	return 0;
}

int gxc_kernel_init(gxc *g, gxc_kernel *k, const void *gxp, const void *map, size_t map_size)
{
	const uint32_t *m = map;
	int err;

	memset(k, 0, sizeof(*k));
	if (map_size < 16 || m[0] != MAP_MAGIC ||
	    map_size != 16 + 8 * (size_t)(m[2] + m[3]))
		return SCE_GXM_ERROR_INVALID_VALUE;
	k->sa_count = m[1];
	k->load_count = m[2];
	k->static_count = m[3];
	k->loads = m + 4;
	k->statics = k->loads + 2 * k->load_count;
	k->gxp = gxp;
	err = sceGxmProgramCheck(k->gxp);
	if (err < 0)
		return err;
	err = sceGxmShaderPatcherRegisterProgram(g->patcher, k->gxp, &k->id);
	if (err < 0)
		return err;
	err = sceGxmShaderPatcherCreateVertexProgram(g->patcher, k->id, NULL, 0, NULL, 0, &k->vp);
	if (err < 0)
		goto fail;
	err = sceGxmShaderPatcherCreateFragmentProgram(g->patcher, g->null_fragment,
						       SCE_GXM_OUTPUT_REGISTER_FORMAT_UCHAR4,
						       SCE_GXM_MULTISAMPLE_NONE, NULL, k->gxp, &k->fp);
	if (err < 0)
		goto fail;
	return 0;

fail:
	gxc_kernel_term(g, k);
	return err;
}

void gxc_kernel_term(gxc *g, gxc_kernel *k)
{
	if (k->fp)
		sceGxmShaderPatcherReleaseFragmentProgram(g->patcher, k->fp);
	if (k->vp)
		sceGxmShaderPatcherReleaseVertexProgram(g->patcher, k->vp);
	if (k->id)
		sceGxmShaderPatcherUnregisterProgram(g->patcher, k->id);
	memset(k, 0, sizeof(*k));
}

static uint32_t f2u(float f)
{
	uint32_t u;

	memcpy(&u, &f, 4);
	return u;
}

static uint32_t channel_value(const gxc_kernel *k, const uint32_t *consts, uint32_t channel)
{
	unsigned int i;

	if (!(channel & MAP_LITERAL))
		return channel < GXC_CONST_CHANNELS ? consts[channel] : 0;
	for (i = 0; i < k->static_count; i++)
		if (k->statics[2 * i] == channel)
			return k->statics[2 * i + 1];
	return 0;
}

static int draw(gxc *g, gxc_kernel *k, const uint32_t *consts, unsigned int nx, unsigned int ny)
{
	uint32_t *sa;
	void *fbuf;
	unsigned int i;
	int err;

	err = sceGxmReserveVertexDefaultUniformBuffer(g->context, (void **)&sa);
	if (err < 0)
		return err;
	if (sa) {
		memset(sa, 0, k->sa_count * 4);
		for (i = 0; i < k->load_count; i++)
			sa[k->loads[2 * i]] = channel_value(k, consts, k->loads[2 * i + 1]);
	}
	err = sceGxmReserveFragmentDefaultUniformBuffer(g->context, &fbuf);
	if (err < 0)
		return err;
	if (fbuf)
		memset(fbuf, 0, sceGxmProgramGetDefaultUniformBufferSize(
			       sceGxmFragmentProgramGetProgram(k->fp)));
	if (k->plain_draw)
		return sceGxmDraw(g->context, SCE_GXM_PRIMITIVE_POINTS, SCE_GXM_INDEX_FORMAT_U32,
				  g->iota.base, nx);
	return sceGxmDrawInstanced(g->context, SCE_GXM_PRIMITIVE_POINTS, SCE_GXM_INDEX_FORMAT_U32,
				   g->iota.base, nx * ny, nx);
}

static int ensure_scratch(gxc *g, unsigned int bytes)
{
	if (g->scratch.base && g->scratch.size >= bytes)
		return 0;
	gxc_mem_free(&g->scratch);
	return gxc_mem_alloc(&g->scratch, SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE, bytes, GXC_MEM_GPU);
}

/* Begin a scene for k's draws. */
static int begin(gxc *g, gxc_kernel *k)
{
	int err = gxc_begin(g);

	if (err < 0)
		return err;
	sceGxmSetVertexProgram(g->context, k->vp);
	sceGxmSetFragmentProgram(g->context, k->fp);
	return 0;
}

static int next_scene(gxc *g, gxc_kernel *k)
{
	int err = gxc_end_and_wait(g);

	return err < 0 ? err : begin(g, k);
}

int gxc_kernel_run(gxc *g, gxc_kernel *k, uint32_t *consts, unsigned int dims,
		   const size_t *global, const size_t *offset,
		   const gxc_scratch *scratch, unsigned int scratch_count)
{
	size_t size[3] = { 1, 1, 1 }, off[3] = { 0, 0, 0 };
	size_t x, y, z, max_points = CHUNK_POINTS, chunk_x;
	unsigned int d, i, per_item = 0, used = 0, draws = 0, bytes;
	int err, end;

	if (dims < 1 || dims > 3 || scratch_count > GXC_MAX_SCRATCH)
		return SCE_GXM_ERROR_INVALID_VALUE;
	for (d = 0; d < dims; d++) {
		size[d] = global[d];
		if (offset)
			off[d] = offset[d];
		if (size[d] == 0)
			return SCE_GXM_ERROR_INVALID_VALUE;
	}
	for (i = 0; i < scratch_count; i++) {
		const gxc_scratch *s = &scratch[i];

		if (s->channel + (s->local ? 1 : 0) >= GXC_CONST_CHANNELS || s->size == 0 ||
		    s->size % 4 || s->size > SCRATCH_MAX)
			return SCE_GXM_ERROR_INVALID_VALUE;
		per_item += s->size;
		if (per_item > SCRATCH_MAX)
			return SCE_GXM_ERROR_OUT_OF_MEMORY;
	}
	err = ensure_iota(g);
	if (err < 0)
		return err;
	if (per_item) {
		/* each slice is 16-byte aligned: leave room for the padding */
		max_points = (SCRATCH_MAX - 16 * scratch_count) / per_item;
		if (max_points == 0)
			return SCE_GXM_ERROR_OUT_OF_MEMORY;
		if (max_points > CHUNK_POINTS)
			max_points = CHUNK_POINTS;
		/* a draw has at most the run's work-items */
		if ((uint64_t)size[0] * size[1] * size[2] < max_points)
			max_points = size[0] * size[1] * size[2];
		/* room for a few draws a scene */
		bytes = 4 * (max_points * per_item + 16 * scratch_count);
		err = ensure_scratch(g, ALIGN(bytes < SCRATCH_MAX ? bytes : SCRATCH_MAX, 4096));
		if (err < 0)
			return err;
	}
	chunk_x = max_points < CHUNK_X ? max_points : CHUNK_X;

	for (i = 0; i < k->static_count; i++)
		if (!(k->statics[2 * i] & MAP_LITERAL) && k->statics[2 * i] < GXC_CONST_CHANNELS)
			consts[k->statics[2 * i]] = k->statics[2 * i + 1];
	consts[GXC_C_WORK_DIM * 4] = dims;
	for (d = 0; d < 3; d++) {
		consts[(GXC_C_GLOBAL_SIZE + d) * 4] = size[d];
		consts[(GXC_C_LOCAL_SIZE + d) * 4] = 1;
		consts[(GXC_C_NUM_GROUPS + d) * 4] = size[d];
		consts[(GXC_C_GLOBAL_OFFSET + d) * 4] = off[d];
	}
	consts[GXC_C_POSITION * 4 + 0] = f2u(-4.0f);
	consts[GXC_C_POSITION * 4 + 1] = f2u(-4.0f);
	consts[GXC_C_POSITION * 4 + 2] = f2u(-2.0f);
	consts[GXC_C_POSITION * 4 + 3] = f2u(1.0f);

	err = begin(g, k);
	if (err < 0)
		return err;
	for (z = 0; z < size[2] && err >= 0; z++) {
		consts[GXC_C_GLOBAL_ID2 * 4] = off[2] + z;
		consts[GXC_C_GROUP_ID2 * 4] = z;
		for (x = 0; x < size[0] && err >= 0; x += chunk_x) {
			size_t nx = size[0] - x < chunk_x ? size[0] - x : chunk_x;
			size_t chunk_y = k->plain_draw ? 1 : max_points / nx;

			/* with scratch, a draw's rows must be contiguous in the linear ids */
			if (per_item && nx < size[0])
				chunk_y = 1;
			for (y = 0; y < size[1] && err >= 0; y += chunk_y) {
				size_t ny = size[1] - y < chunk_y ? size[1] - y : chunk_y;
				/* the draw's first private index; the rest follow on */
				uint32_t first = (off[1] + y) * size[0] + off[0] + x;
				unsigned int need = 0;

				for (i = 0; i < scratch_count; i++)
					need += ALIGN(nx * ny * scratch[i].size, 16);
				if (draws == SCENE_DRAWS || used + need > g->scratch.size) {
					err = next_scene(g, k);
					if (err < 0)
						break;
					draws = 0;
					used = 0;
				}
				for (i = 0; i < scratch_count; i++) {
					const gxc_scratch *s = &scratch[i];
					uint32_t index = first;

					if (s->local)
						index += (off[2] + z) * size[1] * size[0];
					consts[s->channel] = (uint32_t)g->scratch.base + used -
							     GXC_POINTER_BIAS - index * s->size;
					if (s->local)
						consts[s->channel + 1] = s->size;
					used += ALIGN(nx * ny * s->size, 16);
				}
				consts[GXC_C_ID_BASE * 4 + 0] = off[0] + x;
				consts[GXC_C_ID_BASE * 4 + 1] = off[1] + y;
				err = draw(g, k, consts, nx, ny);
				draws++;
			}
		}
	}
	end = gxc_end_and_wait(g);
	return err < 0 ? err : end;
}
