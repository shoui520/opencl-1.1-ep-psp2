/*
 * GPU memory for buffers: arenas of uncached memory mapped for the GPU, each
 * holding blocks in address order.  A large allocation gets an arena of its
 * own; an arena goes back to the system when its last block is freed (but
 * one empty shared arena is kept).
 */
#include "cl_internal.h"

#include <stdlib.h>

#define ARENA_SIZE	(2u << 20)
#define OWN_ARENA	(512u << 10)	/* allocations above this get their own */

typedef struct arena {
	struct arena *next;
	gxc_mem mem;
	dev_block *blocks;
} arena;

struct dev_block {
	dev_block *next;
	arena *arena;
	size_t offset, size;
};

static arena *arenas;

static arena *new_arena(size_t size)
{
	arena *a = calloc(1, sizeof(*a));

	if (!a)
		return NULL;
	if (gxc_mem_alloc(&a->mem, SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE, size, GXC_MEM_GPU) < 0) {
		free(a);
		return NULL;
	}
	a->next = arenas;
	arenas = a;
	return a;
}

/* Put a block of size into a, if it fits. */
static dev_block *place(arena *a, size_t size)
{
	dev_block **link = &a->blocks, *b;
	size_t at = 0;

	for (;;) {
		size_t end = *link ? (*link)->offset : a->mem.size;

		if (end >= at && end - at >= size)
			break;
		if (!*link)
			return NULL;
		at = ALIGN((*link)->offset + (*link)->size, DEV_MEM_ALIGN);
		link = &(*link)->next;
	}
	b = malloc(sizeof(*b));
	if (!b)
		return NULL;
	b->arena = a;
	b->offset = at;
	b->size = size;
	b->next = *link;
	*link = b;
	return b;
}

void *dev_alloc(size_t size, dev_block **block)
{
	arena *a;
	dev_block *b = NULL;

	size = ALIGN(size ? size : 1, DEV_MEM_ALIGN);
	if (size <= OWN_ARENA)
		for (a = arenas; a && !b; a = a->next)
			if (a->mem.size == ARENA_SIZE)
				b = place(a, size);
	if (!b) {
		a = new_arena(size <= OWN_ARENA ? ARENA_SIZE : size);
		if (!a)
			return NULL;
		b = place(a, size);
		if (!b) {
			dev_free(NULL);
			return NULL;
		}
	}
	*block = b;
	return (uint8_t *)b->arena->mem.base + b->offset;
}

void dev_free(dev_block *block)
{
	arena **link;
	int kept = 0;

	if (block) {
		dev_block **b = &block->arena->blocks;

		while (*b != block)
			b = &(*b)->next;
		*b = block->next;
		free(block);
	}
	/* return the empty arenas */
	for (link = &arenas; *link;) {
		arena *a = *link;

		if (a->blocks || (a->mem.size == ARENA_SIZE && !kept++)) {
			link = &a->next;
			continue;
		}
		*link = a->next;
		gxc_mem_free(&a->mem);
		free(a);
	}
}
