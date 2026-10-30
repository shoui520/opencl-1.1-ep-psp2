/*
 * Programs, kernels and kernel launches.  There is no compiler on the
 * device: a program comes from a binary that tools/oclc/clbin.py made, and
 * a program made from source does not build.
 *
 * The binary ("VCLB", all words 32-bit little-endian, offsets from its
 * start):
 *   magic, version (1), kernel count, size
 *   per kernel, 20 words: name offset, name length, gxp offset, gxp size,
 *     map offset, map size, argument count, arguments offset, private count,
 *     privates offset, hidden __local register, hidden __local size,
 *     reqd_work_group_size[3], 0 * 5
 *   an argument, 8 words: address space (0 value, 1 local, 2 global,
 *     3 constant), constant register, element size, components, by
 *     reference, 0 * 3
 *   a private array, 2 words: bytes per work-item, constant channel
 * The gxp and map are tools/gxp/mkgxp.py's, the gxp 16-byte aligned.
 */
#include "cl_internal.h"

#include <malloc.h>
#include <stdlib.h>
#include <string.h>

#define BIN_MAGIC	0x424c4356	/* "VCLB" */
#define BIN_VERSION	1
#define KERNEL_WORDS	20
#define ARG_WORDS	8
#define MAX_ARGS	128

struct launch {
	cl_program program;		/* retained: it holds the image */
	kernel_image *image;
	uint32_t *consts;
	cl_uint dims;
	size_t global[3], offset[3];
	gxc_scratch scratch[GXC_MAX_SCRATCH];
	cl_uint scratch_count;
	cl_mem *mems;			/* retained */
	cl_uint mem_count;
	dev_block **blocks;		/* by-reference copies */
	cl_uint block_count;
};

static char *dup_string(const char *s)
{
	size_t n = strlen(s) + 1;
	char *d = malloc(n);

	if (d)
		memcpy(d, s, n);
	return d;
}

static int in_binary(size_t size, uint32_t off, uint32_t n)
{
	return off <= size && n <= size - off;
}

/* Check a binary and describe its kernels (pointing into it). */
static cl_int parse_binary(const uint8_t *bin, size_t size, kernel_image **out, cl_uint *count)
{
	const uint32_t *w = (const uint32_t *)bin;
	kernel_image *ks;
	uint32_t n, i, j;

	if (size < 16 || w[0] != BIN_MAGIC || w[1] != BIN_VERSION || w[3] != size)
		return CL_INVALID_BINARY;
	n = w[2];
	if (n == 0 || n > 1024 || !in_binary(size, 16, n * KERNEL_WORDS * 4))
		return CL_INVALID_BINARY;
	ks = calloc(n, sizeof(*ks));
	if (!ks)
		return CL_OUT_OF_HOST_MEMORY;
	for (i = 0; i < n; i++) {
		const uint32_t *r = w + 4 + i * KERNEL_WORDS;
		kernel_image *k = &ks[i];
		const uint32_t *a;

		if (!in_binary(size, r[0], r[1] + 1) || bin[r[0] + r[1]] ||
		    strlen((const char *)bin + r[0]) != r[1] ||
		    !in_binary(size, r[2], r[3]) || r[2] % 16 || r[3] < 16 ||
		    !in_binary(size, r[4], r[5]) || r[4] % 4 ||
		    r[6] > MAX_ARGS || !in_binary(size, r[7], r[6] * ARG_WORDS * 4) || r[7] % 4 ||
		    r[8] > GXC_MAX_SCRATCH || !in_binary(size, r[9], r[8] * 8) || r[9] % 4 ||
		    (r[11] && (r[10] >= GXC_CONST_REGS || r[11] > DEV_LOCAL_MEM)))
			goto bad;
		k->name = (const char *)bin + r[0];
		k->gxp = bin + r[2];
		k->gxp_size = r[3];
		k->map = bin + r[4];
		k->map_size = r[5];
		k->arg_count = r[6];
		k->private_count = r[8];
		k->privates = (const uint32_t *)(bin + r[9]);
		k->hidden_creg = r[10];
		k->hidden_size = r[11];
		for (j = 0; j < 3; j++)
			k->reqd[j] = r[12 + j];
		for (j = 0; j < k->private_count; j++)
			if (k->privates[2 * j] == 0 || k->privates[2 * j] > (1u << 20) ||
			    k->privates[2 * j + 1] >= GXC_CONST_CHANNELS)
				goto bad;
		k->args = calloc(k->arg_count ? k->arg_count : 1, sizeof(kernel_arg));
		if (!k->args)
			goto bad;
		a = (const uint32_t *)(bin + r[7]);
		for (j = 0; j < k->arg_count; j++, a += ARG_WORDS) {
			kernel_arg *ka = &k->args[j];

			ka->space = a[0];
			ka->creg = a[1];
			ka->elem = a[2];
			ka->comps = a[3];
			ka->by_ref = a[4];
			if (ka->space > KARG_CONSTANT || ka->elem == 0 || ka->comps == 0 ||
			    ka->comps > 16 || ka->creg * 4 + ka->comps > GXC_CONST_CHANNELS)
				goto bad;
			/* a value is one component per channel, of at most 32 bits */
			if (ka->space == KARG_VALUE && !ka->by_ref && ka->elem > 4)
				goto bad;
		}
	}
	*out = ks;
	*count = n;
	return CL_SUCCESS;

bad:
	for (i = 0; i < n; i++)
		free(ks[i].args);
	free(ks);
	return CL_INVALID_BINARY;
}

void cl_program_retain(cl_program p)
{
	p->refs++;
}

void cl_program_release(cl_program p)
{
	cl_uint i;

	if (--p->refs)
		return;
	for (i = 0; i < p->kernel_count; i++) {
		if (p->kernels[i].registered)
			gxc_kernel_term(&cl_gpu, &p->kernels[i].gk);
		free(p->kernels[i].args);
	}
	free(p->kernels);
	free(p->binary);
	free(p->source);
	free(p->options);
	free(p->log);
	cl_context_release(p->ctx);
	p->magic = 0;
	free(p);
}

static cl_program new_program(cl_context ctx)
{
	cl_program p = calloc(1, sizeof(*p));

	if (!p)
		return NULL;
	p->magic = MAGIC_PROGRAM;
	p->refs = 1;
	p->ctx = ctx;
	cl_context_retain(ctx);
	p->status = CL_BUILD_NONE;
	return p;
}

CL_API_ENTRY cl_program CL_API_CALL
clCreateProgramWithSource(cl_context context, cl_uint count, const char **strings,
			  const size_t *lengths, cl_int *errcode_ret)
{
	cl_program p = NULL;
	cl_int err = CL_SUCCESS;
	size_t total = 0, len;
	cl_uint i;

	cl_lock();
	if (!VALID(context, MAGIC_CONTEXT)) {
		err = CL_INVALID_CONTEXT;
		goto out;
	}
	if (!count || !strings) {
		err = CL_INVALID_VALUE;
		goto out;
	}
	for (i = 0; i < count; i++) {
		if (!strings[i]) {
			err = CL_INVALID_VALUE;
			goto out;
		}
		total += lengths && lengths[i] ? lengths[i] : strlen(strings[i]);
	}
	p = new_program(context);
	if (!p || !(p->source = malloc(total + 1))) {
		err = CL_OUT_OF_HOST_MEMORY;
		goto out;
	}
	for (total = 0, i = 0; i < count; i++) {
		len = lengths && lengths[i] ? lengths[i] : strlen(strings[i]);
		memcpy(p->source + total, strings[i], len);
		total += len;
	}
	p->source[total] = 0;
out:
	if (err != CL_SUCCESS && p) {
		cl_program_release(p);
		p = NULL;
	}
	cl_unlock();
	if (errcode_ret)
		*errcode_ret = err;
	return p;
}

CL_API_ENTRY cl_program CL_API_CALL
clCreateProgramWithBinary(cl_context context, cl_uint num_devices, const cl_device_id *device_list,
			  const size_t *lengths, const unsigned char **binaries,
			  cl_int *binary_status, cl_int *errcode_ret)
{
	cl_program p = NULL;
	cl_int err = CL_SUCCESS;

	cl_lock();
	if (!VALID(context, MAGIC_CONTEXT))
		err = CL_INVALID_CONTEXT;
	else if (!num_devices || !device_list)
		err = CL_INVALID_VALUE;
	else if (num_devices != 1 || device_list[0] != &cl_the_device)
		err = CL_INVALID_DEVICE;
	else if (!lengths || !binaries || !lengths[0] || !binaries[0])
		err = CL_INVALID_VALUE;
	if (err != CL_SUCCESS)
		goto out;
	p = new_program(context);
	if (!p || !(p->binary = memalign(16, lengths[0]))) {
		err = CL_OUT_OF_HOST_MEMORY;
		goto out;
	}
	memcpy(p->binary, binaries[0], lengths[0]);
	p->binary_size = lengths[0];
	err = parse_binary(p->binary, p->binary_size, &p->kernels, &p->kernel_count);
	if (binary_status)
		binary_status[0] = err == CL_INVALID_BINARY ? CL_INVALID_BINARY : CL_SUCCESS;
out:
	if (err != CL_SUCCESS && p) {
		cl_program_release(p);
		p = NULL;
	}
	cl_unlock();
	if (errcode_ret)
		*errcode_ret = err;
	return p;
}

CL_API_ENTRY cl_int CL_API_CALL
clRetainProgram(cl_program program)
{
	cl_int err = CL_INVALID_PROGRAM;

	cl_lock();
	if (VALID(program, MAGIC_PROGRAM)) {
		cl_program_retain(program);
		err = CL_SUCCESS;
	}
	cl_unlock();
	return err;
}

CL_API_ENTRY cl_int CL_API_CALL
clReleaseProgram(cl_program program)
{
	cl_int err = CL_INVALID_PROGRAM;

	cl_lock();
	if (VALID(program, MAGIC_PROGRAM)) {
		cl_program_release(program);
		err = CL_SUCCESS;
	}
	cl_unlock();
	return err;
}

CL_API_ENTRY cl_int CL_API_CALL
clBuildProgram(cl_program program, cl_uint num_devices, const cl_device_id *device_list,
	       const char *options, void (CL_CALLBACK *pfn_notify)(cl_program, void *),
	       void *user_data)
{
	cl_int err = CL_SUCCESS;
	cl_uint i;

	cl_lock();
	if (!VALID(program, MAGIC_PROGRAM)) {
		err = CL_INVALID_PROGRAM;
		goto out;
	}
	if (!device_list != !num_devices || (!pfn_notify && user_data)) {
		err = CL_INVALID_VALUE;
		goto out;
	}
	for (i = 0; i < num_devices; i++)
		if (device_list[i] != &cl_the_device) {
			err = CL_INVALID_DEVICE;
			goto out;
		}
	if (program->kernel_objects) {
		err = CL_INVALID_OPERATION;
		goto out;
	}
	free(program->options);
	free(program->log);
	program->options = dup_string(options ? options : "");
	if (program->source) {
		program->log = dup_string("This device has no compiler: build the program with "
					  "tools/oclc/clbin.py and use clCreateProgramWithBinary.\n");
		program->status = CL_BUILD_ERROR;
		err = CL_COMPILER_NOT_AVAILABLE;
	} else {
		program->log = dup_string("");
		program->status = CL_BUILD_SUCCESS;
	}
	if (!program->options || !program->log) {
		program->status = CL_BUILD_ERROR;
		err = CL_OUT_OF_HOST_MEMORY;
	}
	if (pfn_notify)
		pfn_notify(program, user_data);
out:
	cl_unlock();
	return err;
}

CL_API_ENTRY cl_int CL_API_CALL
clGetProgramInfo(cl_program program, cl_program_info param_name, size_t size, void *value,
		 size_t *size_ret)
{
	cl_device_id dev = &cl_the_device;
	cl_uint n;
	size_t bsize;
	cl_int err;

	cl_lock();
	if (!VALID(program, MAGIC_PROGRAM)) {
		err = CL_INVALID_PROGRAM;
		goto out;
	}
	switch (param_name) {
	case CL_PROGRAM_REFERENCE_COUNT:
		err = cl_info(size, value, size_ret, &program->refs, sizeof(cl_uint));
		break;
	case CL_PROGRAM_CONTEXT:
		err = cl_info(size, value, size_ret, &program->ctx, sizeof(cl_context));
		break;
	case CL_PROGRAM_NUM_DEVICES:
		n = 1;
		err = cl_info(size, value, size_ret, &n, sizeof(n));
		break;
	case CL_PROGRAM_DEVICES:
		err = cl_info(size, value, size_ret, &dev, sizeof(dev));
		break;
	case CL_PROGRAM_SOURCE:
		err = cl_info_string(size, value, size_ret, program->source ? program->source : "");
		break;
	case CL_PROGRAM_BINARY_SIZES:
		bsize = program->binary_size;
		err = cl_info(size, value, size_ret, &bsize, sizeof(bsize));
		break;
	case CL_PROGRAM_BINARIES:
		err = cl_info(size, NULL, size_ret, NULL, sizeof(unsigned char *));
		if (err == CL_SUCCESS && value) {
			unsigned char *dst = *(unsigned char **)value;

			if (size < sizeof(unsigned char *))
				err = CL_INVALID_VALUE;
			else if (dst && program->binary)
				memcpy(dst, program->binary, program->binary_size);
		}
		break;
	default:
		err = CL_INVALID_VALUE;
	}
out:
	cl_unlock();
	return err;
}

CL_API_ENTRY cl_int CL_API_CALL
clGetProgramBuildInfo(cl_program program, cl_device_id device, cl_program_build_info param_name,
		      size_t size, void *value, size_t *size_ret)
{
	cl_int err;

	cl_lock();
	if (!VALID(program, MAGIC_PROGRAM)) {
		err = CL_INVALID_PROGRAM;
		goto out;
	}
	if (device != &cl_the_device) {
		err = CL_INVALID_DEVICE;
		goto out;
	}
	switch (param_name) {
	case CL_PROGRAM_BUILD_STATUS:
		err = cl_info(size, value, size_ret, &program->status, sizeof(cl_build_status));
		break;
	case CL_PROGRAM_BUILD_OPTIONS:
		err = cl_info_string(size, value, size_ret, program->options ? program->options : "");
		break;
	case CL_PROGRAM_BUILD_LOG:
		err = cl_info_string(size, value, size_ret, program->log ? program->log : "");
		break;
	default:
		err = CL_INVALID_VALUE;
	}
out:
	cl_unlock();
	return err;
}

static cl_kernel make_kernel(cl_program p, kernel_image *img, cl_int *err)
{
	cl_kernel k;
	int e;

	if (!img->registered) {
		e = gxc_kernel_init(&cl_gpu, &img->gk, img->gxp, img->map, img->map_size);
		if (e < 0) {
			*err = CL_OUT_OF_RESOURCES;
			return NULL;
		}
		img->registered = 1;
	}
	k = calloc(1, sizeof(*k));
	if (k)
		k->args = calloc(img->arg_count ? img->arg_count : 1, sizeof(kernel_value));
	if (!k || !k->args) {
		free(k);
		*err = CL_OUT_OF_HOST_MEMORY;
		return NULL;
	}
	k->magic = MAGIC_KERNEL;
	k->refs = 1;
	k->program = p;
	k->image = img;
	cl_program_retain(p);
	p->kernel_objects++;
	*err = CL_SUCCESS;
	return k;
}

CL_API_ENTRY cl_kernel CL_API_CALL
clCreateKernel(cl_program program, const char *kernel_name, cl_int *errcode_ret)
{
	cl_kernel k = NULL;
	cl_int err = CL_INVALID_KERNEL_NAME;
	cl_uint i;

	cl_lock();
	if (!VALID(program, MAGIC_PROGRAM))
		err = CL_INVALID_PROGRAM;
	else if (program->status != CL_BUILD_SUCCESS)
		err = CL_INVALID_PROGRAM_EXECUTABLE;
	else if (!kernel_name)
		err = CL_INVALID_VALUE;
	else
		for (i = 0; i < program->kernel_count; i++)
			if (!strcmp(program->kernels[i].name, kernel_name)) {
				k = make_kernel(program, &program->kernels[i], &err);
				break;
			}
	cl_unlock();
	if (errcode_ret)
		*errcode_ret = err;
	return k;
}

CL_API_ENTRY cl_int CL_API_CALL
clCreateKernelsInProgram(cl_program program, cl_uint num_kernels, cl_kernel *kernels,
			 cl_uint *num_kernels_ret)
{
	cl_int err = CL_SUCCESS;
	cl_uint i, j;

	cl_lock();
	if (!VALID(program, MAGIC_PROGRAM)) {
		err = CL_INVALID_PROGRAM;
		goto out;
	}
	if (program->status != CL_BUILD_SUCCESS) {
		err = CL_INVALID_PROGRAM_EXECUTABLE;
		goto out;
	}
	if (kernels && num_kernels < program->kernel_count) {
		err = CL_INVALID_VALUE;
		goto out;
	}
	for (i = 0; kernels && i < program->kernel_count; i++) {
		kernels[i] = make_kernel(program, &program->kernels[i], &err);
		if (!kernels[i]) {
			for (j = 0; j < i; j++)
				cl_kernel_release(kernels[j]);
			goto out;
		}
	}
	if (num_kernels_ret)
		*num_kernels_ret = program->kernel_count;
out:
	cl_unlock();
	return err;
}

static void clear_arg(kernel_value *a)
{
	if (a->mem)
		cl_mem_release(a->mem);
	free(a->value);
	memset(a, 0, sizeof(*a));
}

void cl_kernel_release(cl_kernel k)
{
	cl_uint i;

	if (--k->refs)
		return;
	for (i = 0; i < k->image->arg_count; i++)
		clear_arg(&k->args[i]);
	free(k->args);
	k->program->kernel_objects--;
	cl_program_release(k->program);
	k->magic = 0;
	free(k);
}

CL_API_ENTRY cl_int CL_API_CALL
clRetainKernel(cl_kernel kernel)
{
	cl_int err = CL_INVALID_KERNEL;

	cl_lock();
	if (VALID(kernel, MAGIC_KERNEL)) {
		kernel->refs++;
		err = CL_SUCCESS;
	}
	cl_unlock();
	return err;
}

CL_API_ENTRY cl_int CL_API_CALL
clReleaseKernel(cl_kernel kernel)
{
	cl_int err = CL_INVALID_KERNEL;

	cl_lock();
	if (VALID(kernel, MAGIC_KERNEL)) {
		cl_kernel_release(kernel);
		err = CL_SUCCESS;
	}
	cl_unlock();
	return err;
}

/* The bytes a value argument takes on the host. */
static size_t value_size(const kernel_arg *ka)
{
	if (ka->by_ref)
		return ka->elem;
	return ka->elem * (ka->comps == 3 ? 4 : ka->comps);
}

CL_API_ENTRY cl_int CL_API_CALL
clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value)
{
	const kernel_arg *ka;
	kernel_value *a, na;
	cl_mem m = NULL;
	cl_int err = CL_SUCCESS;

	cl_lock();
	if (!VALID(kernel, MAGIC_KERNEL)) {
		err = CL_INVALID_KERNEL;
		goto out;
	}
	if (arg_index >= kernel->image->arg_count) {
		err = CL_INVALID_ARG_INDEX;
		goto out;
	}
	ka = &kernel->image->args[arg_index];
	a = &kernel->args[arg_index];
	memset(&na, 0, sizeof(na));
	na.set = 1;
	switch (ka->space) {
	case KARG_GLOBAL:
	case KARG_CONSTANT:
		if (arg_size != sizeof(cl_mem)) {
			err = CL_INVALID_ARG_SIZE;
			goto out;
		}
		if (arg_value)
			m = *(const cl_mem *)arg_value;
		if (m && (!VALID(m, MAGIC_MEM) || m->ctx != kernel->program->ctx)) {
			err = CL_INVALID_MEM_OBJECT;
			goto out;
		}
		if (m)
			cl_mem_retain(m);
		na.mem = m;
		break;
	case KARG_LOCAL:
		if (arg_value) {
			err = CL_INVALID_ARG_VALUE;
			goto out;
		}
		if (arg_size == 0 || arg_size > DEV_LOCAL_MEM) {
			err = CL_INVALID_ARG_SIZE;
			goto out;
		}
		na.size = arg_size;
		break;
	default:
		if (!arg_value) {
			err = CL_INVALID_ARG_VALUE;
			goto out;
		}
		if (arg_size != value_size(ka)) {
			err = CL_INVALID_ARG_SIZE;
			goto out;
		}
		na.value = malloc(arg_size);
		if (!na.value) {
			err = CL_OUT_OF_HOST_MEMORY;
			goto out;
		}
		memcpy(na.value, arg_value, arg_size);
		na.size = arg_size;
	}
	clear_arg(a);
	*a = na;
out:
	cl_unlock();
	return err;
}

CL_API_ENTRY cl_int CL_API_CALL
clGetKernelInfo(cl_kernel kernel, cl_kernel_info param_name, size_t size, void *value,
		size_t *size_ret)
{
	cl_int err;

	cl_lock();
	if (!VALID(kernel, MAGIC_KERNEL)) {
		err = CL_INVALID_KERNEL;
		goto out;
	}
	switch (param_name) {
	case CL_KERNEL_FUNCTION_NAME:
		err = cl_info_string(size, value, size_ret, kernel->image->name);
		break;
	case CL_KERNEL_NUM_ARGS:
		err = cl_info(size, value, size_ret, &kernel->image->arg_count, sizeof(cl_uint));
		break;
	case CL_KERNEL_REFERENCE_COUNT:
		err = cl_info(size, value, size_ret, &kernel->refs, sizeof(cl_uint));
		break;
	case CL_KERNEL_CONTEXT:
		err = cl_info(size, value, size_ret, &kernel->program->ctx, sizeof(cl_context));
		break;
	case CL_KERNEL_PROGRAM:
		err = cl_info(size, value, size_ret, &kernel->program, sizeof(cl_program));
		break;
	default:
		err = CL_INVALID_VALUE;
	}
out:
	cl_unlock();
	return err;
}

static cl_ulong local_mem_size(cl_kernel k)
{
	const kernel_image *img = k->image;
	cl_ulong n = img->hidden_size;
	cl_uint i;

	for (i = 0; i < img->arg_count; i++)
		if (img->args[i].space == KARG_LOCAL)
			n += k->args[i].size;
	return n;
}

CL_API_ENTRY cl_int CL_API_CALL
clGetKernelWorkGroupInfo(cl_kernel kernel, cl_device_id device,
			 cl_kernel_work_group_info param_name, size_t size, void *value,
			 size_t *size_ret)
{
	size_t one = 1;
	cl_ulong n;
	cl_uint i;
	cl_int err;

	cl_lock();
	if (!VALID(kernel, MAGIC_KERNEL)) {
		err = CL_INVALID_KERNEL;
		goto out;
	}
	if (device && device != &cl_the_device) {
		err = CL_INVALID_DEVICE;
		goto out;
	}
	switch (param_name) {
	case CL_KERNEL_WORK_GROUP_SIZE:
	case CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE:
		err = cl_info(size, value, size_ret, &one, sizeof(one));
		break;
	case CL_KERNEL_COMPILE_WORK_GROUP_SIZE:
		err = cl_info(size, value, size_ret, kernel->image->reqd, sizeof(kernel->image->reqd));
		break;
	case CL_KERNEL_LOCAL_MEM_SIZE:
		n = local_mem_size(kernel);
		err = cl_info(size, value, size_ret, &n, sizeof(n));
		break;
	case CL_KERNEL_PRIVATE_MEM_SIZE:
		for (n = 0, i = 0; i < kernel->image->private_count; i++)
			n += kernel->image->privates[2 * i];
		err = cl_info(size, value, size_ret, &n, sizeof(n));
		break;
	default:
		err = CL_INVALID_VALUE;
	}
out:
	cl_unlock();
	return err;
}

void cl_free_launch(launch *run)
{
	cl_uint i;

	for (i = 0; i < run->mem_count; i++)
		cl_mem_release(run->mems[i]);
	for (i = 0; i < run->block_count; i++)
		dev_free(run->blocks[i]);
	free(run->mems);
	free(run->blocks);
	free(run->consts);
	cl_program_release(run->program);
	free(run);
}

cl_int cl_run_launch(launch *run)
{
	int err = gxc_kernel_run(&cl_gpu, &run->image->gk, run->consts, run->dims, run->global,
				 run->offset, run->scratch, run->scratch_count);

	return err < 0 ? CL_OUT_OF_RESOURCES : CL_COMPLETE;
}

static cl_int add_scratch(launch *run, uint32_t channel, size_t size, int local)
{
	gxc_scratch *s;

	if (run->scratch_count == GXC_MAX_SCRATCH)
		return CL_OUT_OF_RESOURCES;
	s = &run->scratch[run->scratch_count++];
	s->channel = channel;
	s->size = ALIGN(size, local ? 16 : 4);
	s->local = local;
	return CL_SUCCESS;
}

/* Snapshot k's arguments into a launch. */
static cl_int make_launch(cl_kernel k, launch **out)
{
	const kernel_image *img = k->image;
	launch *run = calloc(1, sizeof(*run));
	cl_int err = CL_SUCCESS;
	cl_uint i, j;

	*out = run;
	if (!run)
		return CL_OUT_OF_HOST_MEMORY;
	run->program = k->program;
	cl_program_retain(k->program);
	run->image = k->image;
	run->consts = calloc(GXC_CONST_CHANNELS, 4);
	run->mems = calloc(img->arg_count + 1, sizeof(cl_mem));
	run->blocks = calloc(img->arg_count + 1, sizeof(dev_block *));
	if (!run->consts || !run->mems || !run->blocks)
		return CL_OUT_OF_HOST_MEMORY;
	for (i = 0; i < img->arg_count && err == CL_SUCCESS; i++) {
		const kernel_arg *ka = &img->args[i];
		const kernel_value *a = &k->args[i];
		uint32_t *c = &run->consts[ka->creg * 4];
		uint8_t *copy;

		switch (ka->space) {
		case KARG_GLOBAL:
		case KARG_CONSTANT:
			c[0] = a->mem ? (uint32_t)a->mem->dev - GXC_POINTER_BIAS : 0;
			if (a->mem) {
				run->mems[run->mem_count++] = a->mem;
				cl_mem_retain(a->mem);
			}
			break;
		case KARG_LOCAL:
			err = add_scratch(run, ka->creg * 4, a->size, 1);
			break;
		default:
			if (ka->by_ref) {
				/* the front end takes the struct's address: pass a copy's */
				copy = dev_alloc(a->size, &run->blocks[run->block_count]);
				if (!copy)
					return CL_OUT_OF_RESOURCES;
				run->block_count++;
				memcpy(copy, a->value, a->size);
				c[0] = (uint32_t)copy - GXC_POINTER_BIAS;
				break;
			}
			/* one component per channel, zero-extended */
			for (j = 0; j < ka->comps; j++) {
				uint32_t v = 0;

				memcpy(&v, a->value + j * ka->elem, ka->elem);
				c[j] = v;
			}
		}
	}
	if (err == CL_SUCCESS && img->hidden_size)
		err = add_scratch(run, img->hidden_creg * 4, img->hidden_size, 1);
	for (i = 0; i < img->private_count && err == CL_SUCCESS; i++)
		err = add_scratch(run, img->privates[2 * i + 1], img->privates[2 * i], 0);
	return err;
}

static cl_int enqueue_kernel(cl_command_queue q, cl_kernel k, cl_command_type type, cl_uint dims,
			     const size_t *offset, const size_t *global, const size_t *local,
			     cl_uint n, const cl_event *list, cl_event *event)
{
	const kernel_image *img;
	command cmd;
	launch *run;
	cl_int err;
	cl_uint i;

	if (!VALID(q, MAGIC_QUEUE))
		return CL_INVALID_COMMAND_QUEUE;
	if (!VALID(k, MAGIC_KERNEL))
		return CL_INVALID_KERNEL;
	if (k->program->ctx != q->ctx)
		return CL_INVALID_CONTEXT;
	err = cl_check_wait_list(q->ctx, n, list);
	if (err != CL_SUCCESS)
		return err;
	img = k->image;
	for (i = 0; i < img->arg_count; i++)
		if (!k->args[i].set)
			return CL_INVALID_KERNEL_ARGS;
	if (dims < 1 || dims > 3)
		return CL_INVALID_WORK_DIMENSION;
	if (!global)
		return CL_INVALID_GLOBAL_WORK_SIZE;
	for (i = 0; i < dims; i++) {
		if (global[i] == 0)
			return CL_INVALID_GLOBAL_WORK_SIZE;
		if (offset && offset[i] > (size_t)-1 - global[i])
			return CL_INVALID_GLOBAL_OFFSET;
	}
	if (img->reqd[0] && (!local || (img->reqd[0] != 1 || img->reqd[1] != 1 || img->reqd[2] != 1)))
		return CL_INVALID_WORK_GROUP_SIZE;
	if (local) {
		size_t items = 1;

		for (i = 0; i < dims; i++) {
			if (local[i] == 0)
				return CL_INVALID_WORK_GROUP_SIZE;
			items *= local[i];
		}
		if (items > 1)
			return CL_INVALID_WORK_GROUP_SIZE;
		for (i = 0; i < dims; i++)
			if (img->reqd[0] && local[i] != img->reqd[i])
				return CL_INVALID_WORK_GROUP_SIZE;
	}
	if (local_mem_size(k) > DEV_LOCAL_MEM)
		return CL_OUT_OF_RESOURCES;

	err = make_launch(k, &run);
	if (err != CL_SUCCESS) {
		if (run)
			cl_free_launch(run);
		return err;
	}
	run->dims = dims;
	for (i = 0; i < 3; i++) {
		run->global[i] = i < dims ? global[i] : 1;
		run->offset[i] = i < dims && offset ? offset[i] : 0;
	}
	memset(&cmd, 0, sizeof(cmd));
	cmd.run = run;
	return cl_enqueue(q, type, &cmd, n, list, event, 0);
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim,
		       const size_t *global_work_offset, const size_t *global_work_size,
		       const size_t *local_work_size, cl_uint num_events_in_wait_list,
		       const cl_event *event_wait_list, cl_event *event)
{
	cl_int err;

	cl_lock();
	err = enqueue_kernel(command_queue, kernel, CL_COMMAND_NDRANGE_KERNEL, work_dim,
			     global_work_offset, global_work_size, local_work_size,
			     num_events_in_wait_list, event_wait_list, event);
	cl_unlock();
	return err;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueTask(cl_command_queue command_queue, cl_kernel kernel, cl_uint num_events_in_wait_list,
	      const cl_event *event_wait_list, cl_event *event)
{
	static const size_t one[1] = { 1 };
	cl_int err;

	cl_lock();
	err = enqueue_kernel(command_queue, kernel, CL_COMMAND_TASK, 1, NULL, one, one,
			     num_events_in_wait_list, event_wait_list, event);
	cl_unlock();
	return err;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueNativeKernel(cl_command_queue command_queue, void (CL_CALLBACK *user_func)(void *),
		      void *args, size_t cb_args, cl_uint num_mem_objects, const cl_mem *mem_list,
		      const void **args_mem_loc, cl_uint num_events_in_wait_list,
		      const cl_event *event_wait_list, cl_event *event)
{
	(void)user_func, (void)args, (void)cb_args, (void)num_mem_objects, (void)mem_list;
	(void)args_mem_loc, (void)num_events_in_wait_list, (void)event_wait_list, (void)event;
	return VALID(command_queue, MAGIC_QUEUE) ? CL_INVALID_OPERATION : CL_INVALID_COMMAND_QUEUE;
}
