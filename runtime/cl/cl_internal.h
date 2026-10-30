/*
 * The OpenCL 1.1 embedded profile runtime: one platform with one device, the
 * SGX543MP4+, which runs kernels compiled offline (tools/oclc/clbin.py) as
 * vertex programs through gxc.
 *
 * Every entry point takes one recursive lock.  Commands run in order, as
 * soon as what they wait for is complete (usually while they are enqueued);
 * a command waiting on a user event stays queued until it is set.
 */
#ifndef CL_INTERNAL_H
#define CL_INTERNAL_H

#define CL_TARGET_OPENCL_VERSION 110
#define CL_USE_DEPRECATED_OPENCL_1_0_APIS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.h>

#include <stddef.h>
#include <stdint.h>

#include "gxc.h"
#include "gxc_kernel.h"

#define ALIGN(x, a)	(((x) + ((a) - 1)) & ~((size_t)(a) - 1))

/* What the device reports (see device.c). */
#define DEV_MEM_ALIGN		128		/* CL_DEVICE_MEM_BASE_ADDR_ALIGN in bytes */
#define DEV_MAX_ALLOC		(32u << 20)
#define DEV_GLOBAL_MEM		(128u << 20)
#define DEV_LOCAL_MEM		(16u << 10)
#define DEV_MAX_CONSTANT	(64u << 10)
#define DEV_MAX_PARAMETERS	1024

enum {
	MAGIC_PLATFORM = 0x504c4154,
	MAGIC_DEVICE = 0x44455649,
	MAGIC_CONTEXT = 0x43545854,
	MAGIC_QUEUE = 0x51554555,
	MAGIC_MEM = 0x4d454d4f,
	MAGIC_PROGRAM = 0x50524f47,
	MAGIC_KERNEL = 0x4b45524e,
	MAGIC_EVENT = 0x4556454e,
};

struct _cl_platform_id {
	uint32_t magic;
};

struct _cl_device_id {
	uint32_t magic;
};

extern struct _cl_platform_id cl_the_platform;
extern struct _cl_device_id cl_the_device;

/* The GPU, set up by the first context. */
extern gxc cl_gpu;
cl_int cl_gpu_init(void);

void cl_lock(void);
void cl_unlock(void);

/* Copy an info value out, the way every clGet*Info does. */
cl_int cl_info(size_t size, void *value, size_t *size_ret, const void *src, size_t n);
cl_int cl_info_string(size_t size, void *value, size_t *size_ret, const char *s);

/* GPU memory (devmem.c): CPU and GPU addresses are the same. */
typedef struct dev_block dev_block;
void *dev_alloc(size_t size, dev_block **block);
void dev_free(dev_block *block);

typedef struct callback {
	struct callback *next;
	void (CL_CALLBACK *fn)();
	void *user_data;
} callback;

struct _cl_context {
	uint32_t magic;
	cl_uint refs;
	cl_context_properties props[3];
	size_t props_size;		/* in bytes, with the terminating 0 */
	void (CL_CALLBACK *notify)(const char *, const void *, size_t, void *);
	void *notify_data;
};

typedef struct mapping {
	struct mapping *next;
	void *ptr;
	size_t offset, size;
	cl_map_flags flags;
} mapping;

struct _cl_mem {
	uint32_t magic;
	cl_uint refs;
	cl_context ctx;
	cl_mem_flags flags;
	size_t size;
	void *host_ptr;			/* CL_MEM_USE_HOST_PTR's, for a sub-buffer too */
	cl_mem parent;			/* of a sub-buffer */
	size_t origin;			/* in the parent */
	uint8_t *dev;			/* the contents, which kernels use */
	dev_block *block;		/* NULL for a sub-buffer */
	cl_uint map_count;
	mapping *maps;
	callback *destructors;		/* the most recent first */
};

/* A kernel of a program binary. */
typedef struct kernel_arg {
	uint32_t space;			/* KARG_* */
	uint32_t creg;			/* its constant register */
	uint32_t elem, comps;		/* element bytes, components */
	uint32_t by_ref;		/* a struct the front end takes by reference */
} kernel_arg;

enum { KARG_VALUE, KARG_LOCAL, KARG_GLOBAL, KARG_CONSTANT };

typedef struct kernel_image {
	const char *name;
	const void *gxp, *map;
	size_t gxp_size, map_size;
	cl_uint arg_count;
	kernel_arg *args;
	cl_uint private_count;
	const uint32_t *privates;	/* (bytes per work-item, channel) pairs */
	uint32_t hidden_creg, hidden_size;	/* locals.py's __local struct; size 0: none */
	size_t reqd[3];			/* reqd_work_group_size, or 0s */
	gxc_kernel gk;
	int registered;
} kernel_image;

struct _cl_program {
	uint32_t magic;
	cl_uint refs;
	cl_context ctx;
	char *source;			/* from clCreateProgramWithSource */
	uint8_t *binary;		/* a copy of the binary; the images point into it */
	size_t binary_size;
	cl_build_status status;
	char *options, *log;
	cl_uint kernel_count;
	kernel_image *kernels;		/* once built */
	cl_uint kernel_objects;		/* live kernels: no rebuilding while there are any */
};

typedef struct kernel_value {
	int set;
	cl_mem mem;			/* a buffer argument, or NULL */
	size_t size;			/* a value's bytes, or a __local argument's */
	uint8_t *value;			/* size bytes */
} kernel_value;

struct _cl_kernel {
	uint32_t magic;
	cl_uint refs;
	cl_program program;
	kernel_image *image;
	kernel_value *args;
};

/* A command, with what it needs copied or retained when it was enqueued. */
typedef struct launch launch;

typedef struct command {
	cl_mem mem[2];			/* retained */
	size_t offset[2], size;
	void *ptr;
	size_t origin[2][3], region[3];	/* the rect commands: [0] the buffer's (or src) */
	size_t row_pitch[2], slice_pitch[2];
	mapping *map;			/* unmap: the mapping, now off the list */
	launch *run;			/* NDRange */
} command;

struct _cl_event {
	uint32_t magic;
	cl_uint refs;
	cl_context ctx;
	cl_command_queue queue;		/* NULL for a user event */
	cl_command_type type;
	cl_int status;
	cl_ulong times[4];		/* queued, submit, start, end in ns */
	callback *callbacks;
	cl_uint wait_count;
	cl_event *waits;		/* retained */
	struct _cl_event *next;		/* in the queue */
	command cmd;
};

struct _cl_command_queue {
	uint32_t magic;
	cl_uint refs;
	cl_context ctx;
	cl_command_queue_properties props;
	cl_event head, tail;		/* waiting to run; each holds a reference */
	struct _cl_command_queue *next;	/* all queues */
};

int cl_valid(const void *obj, uint32_t magic);
#define VALID(obj, m)	cl_valid((obj), (m))

void cl_context_retain(cl_context ctx);
void cl_context_release(cl_context ctx);
void cl_queue_retain(cl_command_queue q);
void cl_queue_release(cl_command_queue q);
void cl_mem_retain(cl_mem m);
void cl_mem_release(cl_mem m);
void cl_program_retain(cl_program p);
void cl_program_release(cl_program p);
void cl_kernel_release(cl_kernel k);
void cl_event_retain(cl_event e);
void cl_event_release(cl_event e);

cl_ulong cl_now(void);

/* queue.c */
cl_int cl_check_wait_list(cl_context ctx, cl_uint n, const cl_event *list);
/*
 * Make a command's event, waiting for n events, add it to q and run what can
 * run.  *out (if out) gets a reference to the event.  The command's retained
 * objects are released when it is done, whether it runs or not.
 */
cl_int cl_enqueue(cl_command_queue q, cl_command_type type, const command *cmd, cl_uint n,
		  const cl_event *list, cl_event *out, int blocking);
void cl_progress(void);
void cl_set_status(cl_event e, cl_int status);
/* Wait until e is no longer queued or running; returns its status. */
cl_int cl_wait(cl_event e);

/* Running commands (mem.c, kernel.c); returns CL_COMPLETE or a negative error. */
cl_int cl_run_mem_command(cl_event e);
cl_int cl_run_launch(launch *run);
void cl_free_launch(launch *run);
cl_mem cl_mem_root(cl_mem m, size_t *offset);

#endif
