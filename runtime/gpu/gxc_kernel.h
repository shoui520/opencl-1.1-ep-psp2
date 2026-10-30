/*
 * Kernel dispatch through SceGxm.
 *
 * A kernel is a vertex program packaged by tools/gxp/mkgxp.py, drawn as
 * indexed, instanced points: the vertex index is the work-item's id in
 * dimension 0 and the instance index its id in dimension 1, each less a
 * per-draw base the program adds from c18.xy.  The kernel's constants (work
 * sizes, arguments, the front end's literals) are register constants c0.x,
 * c0.y, ... which the program's constant map assigns to secondary attribute
 * registers; each draw writes them to the default uniform buffer, which the
 * program DMAs into those registers.
 */
#ifndef GXC_KERNEL_H
#define GXC_KERNEL_H

#include <stddef.h>
#include <stdint.h>

#include "gxc.h"

/* The front end's work-item constants (one vec4 register each). */
#define GXC_C_WORK_DIM		0	/* .x */
#define GXC_C_GLOBAL_SIZE	1	/* 1..3 .x */
#define GXC_C_LOCAL_SIZE	4	/* 4..6 .x */
#define GXC_C_NUM_GROUPS	7	/* 7..9 .x */
#define GXC_C_GLOBAL_OFFSET	10	/* 10..12 .x */
#define GXC_C_GLOBAL_ID2	13	/* .x: get_global_id(2) of the draw */
#define GXC_C_GROUP_ID2		14	/* .x */
#define GXC_C_ID_BASE		18	/* .xy: added to the vertex/instance index */
#define GXC_C_POSITION		19	/* the points' position, outside the clip volume */
#define GXC_C_ARGS		20	/* one register per kernel argument */

/*
 * The compiler's memory accesses (loads, stores, atomics) all address 4 bytes
 * past the pointer they are given, so a pointer argument is passed as the
 * buffer's address less this.
 */
#define GXC_POINTER_BIAS	4

#define GXC_CONST_REGS		256	/* tools/gxp/mkgxp.py CONST_CHANNELS / 4 */
#define GXC_CONST_CHANNELS	(GXC_CONST_REGS * 4)

/*
 * Memory a work-item needs for itself: a private array the front end keeps
 * in memory, or a __local argument (each work-item is a work-group of its
 * own).  Work-item i of the run uses size bytes at base + i * size, where
 * base is a constant channel and i is
 *
 *   private:  get_global_id(1) * get_global_size(0) + get_global_id(0)
 *   __local:  the linear global id, get_global_id(2) included (the argument
 *             is .x base, .y size)
 *
 * so gxc_kernel_run can give each draw its own slice of g->scratch, and the
 * memory only has to cover the draws of one scene.
 */
#define GXC_MAX_SCRATCH		16

typedef struct gxc_scratch {
	uint32_t channel;		/* of the base */
	uint32_t size;			/* bytes per work-item, a multiple of 4 */
	int local;			/* the __local form: size goes in channel + 1 */
} gxc_scratch;

typedef struct gxc_kernel {
	const SceGxmProgram *gxp;
	SceGxmShaderPatcherId id;
	SceGxmVertexProgram *vp;
	SceGxmFragmentProgram *fp;
	unsigned int sa_count, load_count, static_count;
	const uint32_t *loads;		/* (secondary register, channel) pairs */
	const uint32_t *statics;	/* (channel, value) pairs */
	int plain_draw;			/* testing: sceGxmDraw, one row per draw */
} gxc_kernel;

/*
 * Register a packaged kernel.  gxp and map (tools/gxp/mkgxp.py) must stay
 * where they are until gxc_kernel_term.
 */
int gxc_kernel_init(gxc *g, gxc_kernel *k, const void *gxp, const void *map, size_t map_size);
void gxc_kernel_term(gxc *g, gxc_kernel *k);

/*
 * Run k over global[0..dims-1] work-items from offset[] (NULL: 0).  consts
 * holds GXC_CONST_CHANNELS channels with the arguments filled in from c20;
 * the work-item constants and the scratch bases are filled in here.  Waits
 * for the GPU.  The global sizes must be at least 1.
 */
int gxc_kernel_run(gxc *g, gxc_kernel *k, uint32_t *consts, unsigned int dims,
		   const size_t *global, const size_t *offset,
		   const gxc_scratch *scratch, unsigned int scratch_count);

#endif
