#include "gxc.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define ALIGN(x, a)	(((x) + ((a) - 1)) & ~((a) - 1))

extern const uint8_t gxc_null_fragment_gxp[];

static void *patcher_host_alloc(void *user_data, unsigned int size)
{
	(void)user_data;
	return malloc(size);
}

static void patcher_host_free(void *user_data, void *mem)
{
	(void)user_data;
	free(mem);
}

static void display_callback(const void *data)
{
	(void)data;
}

int gxc_mem_alloc(gxc_mem *m, SceKernelMemBlockType type, unsigned int size, int kind)
{
	int err;
	unsigned int align = type == SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW ? 256 * 1024 : 4 * 1024;

	memset(m, 0, sizeof(*m));
	m->uid = -1;
	m->kind = kind;
	m->size = ALIGN(size, align);
	m->uid = sceKernelAllocMemBlock("gxc", type, m->size, NULL);
	if (m->uid < 0)
		return m->uid;
	err = sceKernelGetMemBlockBase(m->uid, &m->base);
	if (err < 0)
		goto fail;
	switch (kind) {
	case GXC_MEM_GPU:
		err = sceGxmMapMemory(m->base, m->size,
				      SCE_GXM_MEMORY_ATTRIB_READ | SCE_GXM_MEMORY_ATTRIB_WRITE);
		break;
	case GXC_MEM_VERTEX_USSE:
		err = sceGxmMapVertexUsseMemory(m->base, m->size, &m->usse_offset);
		break;
	case GXC_MEM_FRAGMENT_USSE:
		err = sceGxmMapFragmentUsseMemory(m->base, m->size, &m->usse_offset);
		break;
	default:
		err = -1;
	}
	if (err < 0)
		goto fail;
	return 0;

fail:
	sceKernelFreeMemBlock(m->uid);
	m->uid = -1;
	m->base = NULL;
	return err;
}

void gxc_mem_free(gxc_mem *m)
{
	if (m->uid < 0 || !m->base)
		return;
	switch (m->kind) {
	case GXC_MEM_GPU:
		sceGxmUnmapMemory(m->base);
		break;
	case GXC_MEM_VERTEX_USSE:
		sceGxmUnmapVertexUsseMemory(m->base);
		break;
	case GXC_MEM_FRAGMENT_USSE:
		sceGxmUnmapFragmentUsseMemory(m->base);
		break;
	}
	sceKernelFreeMemBlock(m->uid);
	m->uid = -1;
	m->base = NULL;
}

#define TRY(call, name) do { err = (call); if (err < 0) { *stage = (name); return err; } } while (0)

int gxc_init(gxc *g, const char **stage)
{
	int err;
	SceGxmInitializeParams ip;
	SceGxmContextParams cp;
	SceGxmRenderTargetParams rp;
	SceGxmShaderPatcherParams pp;
	const char *dummy;

	if (!stage)
		stage = &dummy;
	memset(g, 0, sizeof(*g));

	memset(&ip, 0, sizeof(ip));
	ip.flags = 0;
	ip.displayQueueMaxPendingCount = 1;
	ip.displayQueueCallback = display_callback;
	ip.displayQueueCallbackDataSize = 4;
	ip.parameterBufferSize = SCE_GXM_DEFAULT_PARAMETER_BUFFER_SIZE;
	TRY(sceGxmInitialize(&ip), "sceGxmInitialize");
	g->initialized = 1;

	TRY(gxc_mem_alloc(&g->vdm_ring, SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE,
			  SCE_GXM_DEFAULT_VDM_RING_BUFFER_SIZE, GXC_MEM_GPU), "vdm ring");
	TRY(gxc_mem_alloc(&g->vertex_ring, SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE,
			  SCE_GXM_DEFAULT_VERTEX_RING_BUFFER_SIZE, GXC_MEM_GPU), "vertex ring");
	TRY(gxc_mem_alloc(&g->fragment_ring, SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE,
			  SCE_GXM_DEFAULT_FRAGMENT_RING_BUFFER_SIZE, GXC_MEM_GPU), "fragment ring");
	TRY(gxc_mem_alloc(&g->fragment_usse_ring, SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE,
			  SCE_GXM_DEFAULT_FRAGMENT_USSE_RING_BUFFER_SIZE, GXC_MEM_FRAGMENT_USSE),
	    "fragment usse ring");

	g->context_host_mem = malloc(SCE_GXM_MINIMUM_CONTEXT_HOST_MEM_SIZE);
	memset(&cp, 0, sizeof(cp));
	cp.hostMem = g->context_host_mem;
	cp.hostMemSize = SCE_GXM_MINIMUM_CONTEXT_HOST_MEM_SIZE;
	cp.vdmRingBufferMem = g->vdm_ring.base;
	cp.vdmRingBufferMemSize = SCE_GXM_DEFAULT_VDM_RING_BUFFER_SIZE;
	cp.vertexRingBufferMem = g->vertex_ring.base;
	cp.vertexRingBufferMemSize = SCE_GXM_DEFAULT_VERTEX_RING_BUFFER_SIZE;
	cp.fragmentRingBufferMem = g->fragment_ring.base;
	cp.fragmentRingBufferMemSize = SCE_GXM_DEFAULT_FRAGMENT_RING_BUFFER_SIZE;
	cp.fragmentUsseRingBufferMem = g->fragment_usse_ring.base;
	cp.fragmentUsseRingBufferMemSize = SCE_GXM_DEFAULT_FRAGMENT_USSE_RING_BUFFER_SIZE;
	cp.fragmentUsseRingBufferOffset = g->fragment_usse_ring.usse_offset;
	TRY(sceGxmCreateContext(&cp, &g->context), "sceGxmCreateContext");

	memset(&rp, 0, sizeof(rp));
	rp.width = GXC_RT_WIDTH;
	rp.height = GXC_RT_HEIGHT;
	rp.scenesPerFrame = 1;
	rp.multisampleMode = SCE_GXM_MULTISAMPLE_NONE;
	rp.driverMemBlock = -1;
	TRY(sceGxmCreateRenderTarget(&rp, &g->render_target), "sceGxmCreateRenderTarget");

	TRY(gxc_mem_alloc(&g->color, SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE,
			  4 * GXC_RT_WIDTH * GXC_RT_HEIGHT, GXC_MEM_GPU), "color surface");
	memset(g->color.base, 0, g->color.size);
	TRY(sceGxmColorSurfaceInit(&g->color_surface, SCE_GXM_COLOR_FORMAT_A8B8G8R8,
				   SCE_GXM_COLOR_SURFACE_LINEAR, SCE_GXM_COLOR_SURFACE_SCALE_NONE,
				   SCE_GXM_OUTPUT_REGISTER_SIZE_32BIT, GXC_RT_WIDTH, GXC_RT_HEIGHT,
				   GXC_RT_WIDTH, g->color.base), "sceGxmColorSurfaceInit");

	{
		unsigned int w = ALIGN(GXC_RT_WIDTH, SCE_GXM_TILE_SIZEX);
		unsigned int h = ALIGN(GXC_RT_HEIGHT, SCE_GXM_TILE_SIZEY);

		TRY(gxc_mem_alloc(&g->depth, SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE,
				  4 * w * h, GXC_MEM_GPU), "depth");
		TRY(gxc_mem_alloc(&g->stencil, SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE,
				  4 * w * h, GXC_MEM_GPU), "stencil");
		TRY(sceGxmDepthStencilSurfaceInit(&g->depth_surface, SCE_GXM_DEPTH_STENCIL_FORMAT_S8D24,
						  SCE_GXM_DEPTH_STENCIL_SURFACE_TILED, w,
						  g->depth.base, g->stencil.base),
		    "sceGxmDepthStencilSurfaceInit");
	}

	TRY(gxc_mem_alloc(&g->patcher_buffer, SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE,
			  256 * 1024, GXC_MEM_GPU), "patcher buffer");
	TRY(gxc_mem_alloc(&g->patcher_vertex_usse, SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE,
			  1024 * 1024, GXC_MEM_VERTEX_USSE), "patcher vertex usse");
	TRY(gxc_mem_alloc(&g->patcher_fragment_usse, SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE,
			  64 * 1024, GXC_MEM_FRAGMENT_USSE), "patcher fragment usse");

	memset(&pp, 0, sizeof(pp));
	pp.hostAllocCallback = patcher_host_alloc;
	pp.hostFreeCallback = patcher_host_free;
	pp.bufferMem = g->patcher_buffer.base;
	pp.bufferMemSize = g->patcher_buffer.size;
	pp.vertexUsseMem = g->patcher_vertex_usse.base;
	pp.vertexUsseMemSize = g->patcher_vertex_usse.size;
	pp.vertexUsseOffset = g->patcher_vertex_usse.usse_offset;
	pp.fragmentUsseMem = g->patcher_fragment_usse.base;
	pp.fragmentUsseMemSize = g->patcher_fragment_usse.size;
	pp.fragmentUsseOffset = g->patcher_fragment_usse.usse_offset;
	TRY(sceGxmShaderPatcherCreate(&pp, &g->patcher), "sceGxmShaderPatcherCreate");
	TRY(sceGxmProgramCheck((const SceGxmProgram *)gxc_null_fragment_gxp), "null fragment");
	TRY(sceGxmShaderPatcherRegisterProgram(g->patcher, (const SceGxmProgram *)gxc_null_fragment_gxp,
					       &g->null_fragment), "null fragment");

	*stage = "ok";
	return 0;
}

void gxc_term(gxc *g)
{
	if (g->context)
		sceGxmFinish(g->context);
	gxc_mem_free(&g->scratch);
	gxc_mem_free(&g->iota);
	if (g->null_fragment)
		sceGxmShaderPatcherUnregisterProgram(g->patcher, g->null_fragment);
	if (g->patcher)
		sceGxmShaderPatcherDestroy(g->patcher);
	gxc_mem_free(&g->patcher_fragment_usse);
	gxc_mem_free(&g->patcher_vertex_usse);
	gxc_mem_free(&g->patcher_buffer);
	gxc_mem_free(&g->stencil);
	gxc_mem_free(&g->depth);
	gxc_mem_free(&g->color);
	if (g->render_target)
		sceGxmDestroyRenderTarget(g->render_target);
	if (g->context)
		sceGxmDestroyContext(g->context);
	gxc_mem_free(&g->fragment_usse_ring);
	gxc_mem_free(&g->fragment_ring);
	gxc_mem_free(&g->vertex_ring);
	gxc_mem_free(&g->vdm_ring);
	free(g->context_host_mem);
	if (g->initialized)
		sceGxmTerminate();
	memset(g, 0, sizeof(*g));
}

int gxc_begin(gxc *g)
{
	return sceGxmBeginScene(g->context, 0, g->render_target, NULL, NULL, NULL,
				&g->color_surface, &g->depth_surface);
}

int gxc_end_and_wait(gxc *g)
{
	int err = sceGxmEndScene(g->context, NULL, NULL);

	if (err < 0)
		return err;
	sceGxmFinish(g->context);
	return 0;
}
