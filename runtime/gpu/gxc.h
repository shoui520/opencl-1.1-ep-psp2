/*
 * gxc: the SceGxm context the OpenCL runtime dispatches work through.
 *
 * Work runs as vertex programs drawn as points: each vertex is one work-item,
 * the program writes its results to memory, and its position lies outside
 * the clip volume, so the render target is never touched.
 */
#ifndef GXC_H
#define GXC_H

#include <psp2/gxm.h>
#include <psp2/kernel/sysmem.h>

#define GXC_RT_WIDTH	64
#define GXC_RT_HEIGHT	64

typedef struct gxc_mem {
	SceUID uid;
	void *base;
	unsigned int size;
	int kind;		/* GXC_MEM_* */
	unsigned int usse_offset;
} gxc_mem;

enum {
	GXC_MEM_GPU,		/* mapped with sceGxmMapMemory */
	GXC_MEM_VERTEX_USSE,
	GXC_MEM_FRAGMENT_USSE,
};

typedef struct gxc {
	SceGxmContext *context;
	SceGxmRenderTarget *render_target;
	SceGxmColorSurface color_surface;
	SceGxmDepthStencilSurface depth_surface;
	SceGxmShaderPatcher *patcher;
	void *context_host_mem;
	gxc_mem vdm_ring, vertex_ring, fragment_ring, fragment_usse_ring;
	gxc_mem color, depth, stencil;
	gxc_mem patcher_buffer, patcher_vertex_usse, patcher_fragment_usse;
	gxc_mem iota;		/* 0, 1, 2, ...: the index buffer of kernel draws */
	gxc_mem scratch;	/* the work-items' private and __local memory */
	SceGxmShaderPatcherId null_fragment;	/* the fragment program of kernel draws */
	int initialized;	/* sceGxmInitialize succeeded */
} gxc;

/*
 * Returns 0 or a negative SceGxm/SceKernel error; *stage names the failing
 * call.  gxc_term undoes as much as was done, after a failure too.
 */
int gxc_init(gxc *g, const char **stage);
void gxc_term(gxc *g);

/*
 * GPU-visible memory.  type is a SceKernelMemBlockType; the block is mapped
 * read/write for the GPU at the same virtual address the CPU uses.
 */
int gxc_mem_alloc(gxc_mem *m, SceKernelMemBlockType type, unsigned int size, int kind);
void gxc_mem_free(gxc_mem *m);

/* Begin a scene on the private render target. */
int gxc_begin(gxc *g);
/* End the scene and wait for the GPU to finish it. */
int gxc_end_and_wait(gxc *g);

#endif
