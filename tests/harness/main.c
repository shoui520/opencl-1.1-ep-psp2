/*
 * Hardware harness: runs the tests of ux0:data/ocl/tests.pak (written by
 * tests/device/devtest.py) through the OpenCL runtime and writes what they
 * read back to ux0:data/ocl/results.pak, which devtest.py checks against a
 * CPU OpenCL.  Progress goes to ux0:data/ocl/harness.log; its last line is
 * "HARNESS DONE".
 *
 * Each buffer a test makes is a sub-buffer of one with GUARD bytes either
 * side, which are checked at the end.
 */
#include <CL/cl.h>
#include <psp2/kernel/processmgr.h>
#include <psp2/kernel/sysmem.h>
#include <psp2/power.h>
#include <psp2/io/stat.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ep_api.h"

#define PACK_PATH	"ux0:data/ocl/tests.pak"
#define RESULTS_PATH	"ux0:data/ocl/results.pak"

#define PACK_MAGIC	0x32504356	/* "VCP2" */
#define TEST_MAGIC	0x32544356	/* "VCT2" */
#define RESULTS_MAGIC	0x32524356	/* "VCR2" */

#define OP_WORDS	24
#define NO_DATA		0xffffffffu
#define NULL_ID		0xffffffffu
#define MAX_BUFS	32
#define MAX_KERNELS	8
#define MAX_EVENTS	16
#define MAX_OPS		1024
#define GUARD		128	/* CL_DEVICE_MEM_BASE_ADDR_ALIGN: where a sub-buffer can start */
#define GUARD_BYTE	0xA5
#define BAD_PACK	((cl_int)0x80EC0001)
#define EP_API_REPORT	4096	/* OP_EP_API's output: failures, then their text */

#define ALIGN(x, a)	(((x) + ((a) - 1)) & ~((a) - 1))

enum {
	OP_BUFFER = 1, OP_SUBBUFFER, OP_KERNEL, OP_ARG_MEM, OP_ARG_VALUE, OP_ARG_LOCAL, OP_NDRANGE,
	OP_READ, OP_WRITE, OP_COPY, OP_READ_RECT, OP_WRITE_RECT, OP_COPY_RECT, OP_MAP,
	OP_USER_EVENT, OP_SET_STATUS, OP_STATUS, OP_FINISH, OP_WAIT, OP_RELEASE, OP_EP_API,
};

typedef struct pack_test {
	uint32_t magic, size;
	char name[32];
	uint32_t bin_off, bin_size, op_count, ops_off, data_off, data_size;
} pack_test;

typedef struct test_state {
	cl_context ctx;
	cl_command_queue q;
	cl_device_id dev;
	cl_program prog;
	const uint8_t *data;
	uint32_t data_size;
	cl_mem parent[MAX_BUFS], buf[MAX_BUFS];
	size_t size[MAX_BUFS];
	void *host[MAX_BUFS];		/* a CL_MEM_USE_HOST_PTR buffer's memory */
	cl_kernel k[MAX_KERNELS];
	cl_event ev[MAX_EVENTS];
	uint8_t *out[MAX_OPS];
	uint32_t out_size[MAX_OPS];
	cl_int codes[MAX_OPS];
	uint32_t kernel_us;
} test_state;

static FILE *log_file;

static void tlog(const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	vfprintf(log_file, fmt, ap);
	va_end(ap);
	fflush(log_file);
}

static const uint8_t *data_at(test_state *s, uint32_t off, uint32_t n)
{
	if (off == NO_DATA || off > s->data_size || n > s->data_size - off)
		return NULL;
	return s->data + off;
}

/* The event slot + 1 of a wait or event operand: 0 none, -1 bad. */
static int slot(uint32_t w)
{
	return w == 0 ? 0 : w <= MAX_EVENTS ? (int)w : -1;
}

static uint32_t output_size(const uint32_t *w)
{
	switch (w[0]) {
	case OP_READ:
		return w[5];
	case OP_READ_RECT:
		return w[17];
	case OP_MAP:
		return w[4] & CL_MAP_READ ? w[6] : 0;
	case OP_STATUS:
		return 4;
	case OP_EP_API:
		return 4 + EP_API_REPORT;
	}
	return 0;
}

#define BUF(i)	((i) < MAX_BUFS ? s->buf[i] : NULL)
#define KERN(i)	((i) < MAX_KERNELS ? s->k[i] : NULL)

static cl_int run_op(test_state *s, const uint32_t *w, uint8_t *out)
{
	int ws = slot(w[1]), es = slot(w[2]);
	cl_event *wait = ws > 0 ? &s->ev[ws - 1] : NULL;
	cl_event *event = es > 0 ? &s->ev[es - 1] : NULL;
	cl_uint nwait = wait ? 1 : 0;
	const uint8_t *d;
	cl_int err = CL_SUCCESS;
	size_t g[3], o[3], one[3] = { 1, 1, 1 };
	SceUInt64 t0;
	void *ptr;
	int32_t n;

	if (ws < 0 || es < 0 || (wait && !*wait))
		return BAD_PACK;
	if (event && *event) {
		clReleaseEvent(*event);
		*event = NULL;
	}
	switch (w[0]) {
	case OP_BUFFER: {
		uint32_t id = w[3], flags = w[4], size = w[5];
		size_t total = GUARD + ALIGN(size, GUARD) + GUARD;
		cl_buffer_region r = { GUARD, size };
		uint8_t *host;

		if (id >= MAX_BUFS || s->parent[id])
			return BAD_PACK;
		host = malloc(total);
		if (!host)
			return CL_OUT_OF_HOST_MEMORY;
		memset(host, GUARD_BYTE, total);
		memset(host + GUARD, 0, size);
		if (w[6] != NO_DATA) {
			if (!(d = data_at(s, w[6], size))) {
				free(host);
				return BAD_PACK;
			}
			memcpy(host + GUARD, d, size);
		}
		s->parent[id] = clCreateBuffer(s->ctx, (flags & ~(cl_mem_flags)CL_MEM_COPY_HOST_PTR) |
					       (flags & CL_MEM_USE_HOST_PTR ? 0 : CL_MEM_COPY_HOST_PTR),
					       total, host, &err);
		if (flags & CL_MEM_USE_HOST_PTR && s->parent[id])
			s->host[id] = host;
		else
			free(host);
		if (err != CL_SUCCESS)
			return err;
		s->size[id] = size;
		s->buf[id] = clCreateSubBuffer(s->parent[id], 0, CL_BUFFER_CREATE_TYPE_REGION, &r, &err);
		return err;
	}
	case OP_SUBBUFFER: {
		uint32_t id = w[3], pid = w[4];
		cl_buffer_region r = { GUARD + w[6], w[7] };

		if (id >= MAX_BUFS || s->buf[id] || pid >= MAX_BUFS || !s->parent[pid])
			return BAD_PACK;
		s->buf[id] = clCreateSubBuffer(s->parent[pid], w[5], CL_BUFFER_CREATE_TYPE_REGION, &r,
					       &err);
		return err;
	}
	case OP_KERNEL:
		if (w[3] >= MAX_KERNELS || !(d = data_at(s, w[4], w[5] + 1)) || d[w[5]])
			return BAD_PACK;
		if (s->k[w[3]])
			clReleaseKernel(s->k[w[3]]);
		s->k[w[3]] = clCreateKernel(s->prog, (const char *)d, &err);
		return err;
	case OP_ARG_MEM: {
		cl_mem m = w[5] == NULL_ID ? NULL : BUF(w[5]);

		return clSetKernelArg(KERN(w[3]), w[4], sizeof(m), &m);
	}
	case OP_ARG_VALUE:
		if (!(d = data_at(s, w[5], w[6])))
			return BAD_PACK;
		return clSetKernelArg(KERN(w[3]), w[4], w[6], d);
	case OP_ARG_LOCAL:
		return clSetKernelArg(KERN(w[3]), w[4], w[5], NULL);
	case OP_NDRANGE:
		g[0] = w[5], g[1] = w[6], g[2] = w[7];
		o[0] = w[8], o[1] = w[9], o[2] = w[10];
		/* where a GPU crash happened, since the log survives it */
		tlog("ndrange %u: %u x %u x %u\n", w[3], w[5], w[6], w[7]);
		t0 = sceKernelGetProcessTimeWide();
		err = clEnqueueNDRangeKernel(s->q, KERN(w[3]), w[4], w[12] ? o : NULL, g,
					     w[11] ? one : NULL, nwait, wait, event);
		s->kernel_us += sceKernelGetProcessTimeWide() - t0;
		return err;
	case OP_READ:
		return clEnqueueReadBuffer(s->q, BUF(w[3]), w[6], w[4], w[5], out, nwait, wait, event);
	case OP_WRITE:
		if (!(d = data_at(s, w[6], w[5])))
			return BAD_PACK;
		return clEnqueueWriteBuffer(s->q, BUF(w[3]), w[7], w[4], w[5], d, nwait, wait, event);
	case OP_COPY:
		return clEnqueueCopyBuffer(s->q, BUF(w[3]), BUF(w[4]), w[5], w[6], w[7], nwait, wait,
					   event);
	case OP_READ_RECT:
	case OP_WRITE_RECT: {
		size_t bo[3] = { w[4], w[5], w[6] }, ho[3] = { w[7], w[8], w[9] };
		size_t r[3] = { w[10], w[11], w[12] };
		uint8_t *host = out;

		if (w[0] == OP_WRITE_RECT && !(host = calloc(1, w[17] ? w[17] : 1)))
			return CL_OUT_OF_HOST_MEMORY;
		if (w[18] != NO_DATA) {
			if (!(d = data_at(s, w[18], w[17])))
				err = BAD_PACK;
			else
				memcpy(host, d, w[17]);
		}
		if (err == CL_SUCCESS && w[0] == OP_READ_RECT)
			err = clEnqueueReadBufferRect(s->q, BUF(w[3]), w[19], bo, ho, r, w[13], w[14],
						      w[15], w[16], host, nwait, wait, event);
		else if (err == CL_SUCCESS)
			/* blocking, so the host memory can go */
			err = clEnqueueWriteBufferRect(s->q, BUF(w[3]), CL_TRUE, bo, ho, r, w[13], w[14],
						       w[15], w[16], host, nwait, wait, event);
		if (w[0] == OP_WRITE_RECT)
			free(host);
		return err;
	}
	case OP_COPY_RECT: {
		size_t so[3] = { w[5], w[6], w[7] }, dst[3] = { w[8], w[9], w[10] };
		size_t r[3] = { w[11], w[12], w[13] };

		return clEnqueueCopyBufferRect(s->q, BUF(w[3]), BUF(w[4]), so, dst, r, w[14], w[15],
					       w[16], w[17], nwait, wait, event);
	}
	case OP_MAP: {
		cl_event e = NULL;
		cl_int uerr;

		ptr = clEnqueueMapBuffer(s->q, BUF(w[3]), w[8], w[4], w[5], w[6], nwait, wait, &e, &err);
		if (err != CL_SUCCESS)
			return err;
		if (!w[8])
			err = clWaitForEvents(1, &e);
		if (err == CL_SUCCESS && w[4] & CL_MAP_READ)
			memcpy(out, ptr, w[6]);
		if (err == CL_SUCCESS && w[4] & CL_MAP_WRITE && w[7] != NO_DATA) {
			if (!(d = data_at(s, w[7], w[6])))
				err = BAD_PACK;
			else
				memcpy(ptr, d, w[6]);
		}
		uerr = clEnqueueUnmapMemObject(s->q, BUF(w[3]), ptr, 0, NULL, NULL);
		if (event)
			*event = e;
		else
			clReleaseEvent(e);
		return err != CL_SUCCESS ? err : uerr;
	}
	case OP_USER_EVENT:
		if (!event)
			return BAD_PACK;
		*event = clCreateUserEvent(s->ctx, &err);
		return err;
	case OP_SET_STATUS:
		if (w[3] >= MAX_EVENTS)
			return BAD_PACK;
		return clSetUserEventStatus(s->ev[w[3]], (cl_int)w[4]);
	case OP_STATUS:
		if (w[3] >= MAX_EVENTS)
			return BAD_PACK;
		return clGetEventInfo(s->ev[w[3]], CL_EVENT_COMMAND_EXECUTION_STATUS, 4, out, NULL);
	case OP_FINISH:
		return clFinish(s->q);
	case OP_WAIT:
		if (w[3] >= MAX_EVENTS)
			return BAD_PACK;
		return clWaitForEvents(1, &s->ev[w[3]]);
	case OP_RELEASE:
		if (w[3] >= MAX_BUFS || !s->buf[w[3]])
			return BAD_PACK;
		err = clReleaseMemObject(s->buf[w[3]]);
		s->buf[w[3]] = NULL;
		return err;
	case OP_EP_API:
		if (!s->prog)
			return BAD_PACK;
		n = ep_api_run(s->ctx, s->q, s->dev, s->prog, (char *)out + 4, EP_API_REPORT, tlog);
		memcpy(out, &n, 4);
		return CL_SUCCESS;
	}
	return BAD_PACK;
}

static uint32_t guard_errors(test_state *s)
{
	uint32_t i, n, bad = 0;
	uint8_t *p;

	for (i = 0; i < MAX_BUFS; i++) {
		size_t total;

		if (!s->parent[i])
			continue;
		total = GUARD + ALIGN(s->size[i], GUARD) + GUARD;
		p = malloc(total);
		if (!p || clEnqueueReadBuffer(s->q, s->parent[i], CL_TRUE, 0, total, p, 0, NULL,
					      NULL) != CL_SUCCESS) {
			bad++;
		} else {
			for (n = 0; n < total; n++)
				bad += (n < GUARD || n >= GUARD + s->size[i]) && p[n] != GUARD_BYTE;
		}
		free(p);
	}
	return bad;
}

static void free_state(test_state *s)
{
	uint32_t i;

	for (i = 0; i < MAX_EVENTS; i++)
		if (s->ev[i])
			clReleaseEvent(s->ev[i]);
	for (i = 0; i < MAX_KERNELS; i++)
		if (s->k[i])
			clReleaseKernel(s->k[i]);
	for (i = 0; i < MAX_BUFS; i++) {
		if (s->buf[i])
			clReleaseMemObject(s->buf[i]);
		if (s->parent[i])
			clReleaseMemObject(s->parent[i]);
		free(s->host[i]);
	}
	for (i = 0; i < MAX_OPS; i++)
		free(s->out[i]);
	if (s->prog)
		clReleaseProgram(s->prog);
}

static const char *op_name(uint32_t op)
{
	static const char *names[] = {
		"?", "buffer", "subbuffer", "kernel", "arg_mem", "arg_value", "arg_local", "ndrange",
		"read", "write", "copy", "read_rect", "write_rect", "copy_rect", "map", "user_event",
		"set_status", "status", "finish", "wait", "release", "ep_api",
	};

	return op < sizeof(names) / sizeof(names[0]) ? names[op] : "?";
}

/* Run one test and append its result to out. */
static int run_test(cl_context ctx, cl_command_queue q, cl_device_id dev, const uint8_t *base,
		    uint32_t avail, FILE *out)
{
	static test_state st;
	test_state *s = &st;
	const pack_test *t = (const pack_test *)base;
	uint32_t head[5] = { 0, 0, 0, 0, 0 }, i, nops = 0;
	cl_int err = CL_SUCCESS;

	memset(s, 0, sizeof(*s));
	s->ctx = ctx;
	s->q = q;
	s->dev = dev;
	if (avail < sizeof(*t) || t->magic != TEST_MAGIC || t->size > avail ||
	    t->op_count > MAX_OPS || t->ops_off > t->size || t->ops_off % 4 ||
	    t->op_count * OP_WORDS * 4 > t->size - t->ops_off || t->bin_off > t->size ||
	    t->bin_size > t->size - t->bin_off || t->data_off > t->size ||
	    t->data_size > t->size - t->data_off) {
		err = BAD_PACK;
		goto done;
	}
	nops = t->op_count;
	s->data = base + t->data_off;
	s->data_size = t->data_size;
	if (t->bin_size) {
		const unsigned char *bin = base + t->bin_off;
		size_t size = t->bin_size;
		cl_int bstatus;

		s->prog = clCreateProgramWithBinary(ctx, 1, &dev, &size, &bin, &bstatus, &err);
		if (err == CL_SUCCESS)
			err = clBuildProgram(s->prog, 1, &dev, "", NULL, NULL);
		if (err != CL_SUCCESS) {
			tlog("program: %d\n", err);
			goto done;
		}
	}
	for (i = 0; i < nops; i++) {
		const uint32_t *w = (const uint32_t *)(base + t->ops_off) + i * OP_WORDS;

		s->out_size[i] = output_size(w);
		if (s->out_size[i] && !(s->out[i] = calloc(1, s->out_size[i]))) {
			err = CL_OUT_OF_HOST_MEMORY;
			break;
		}
		s->codes[i] = run_op(s, w, s->out[i]);
		if (s->codes[i] != CL_SUCCESS)
			tlog("op %u %s: %d\n", i, op_name(w[0]), s->codes[i]);
		if (s->codes[i] == BAD_PACK) {
			err = BAD_PACK;
			break;
		}
	}
	clFinish(q);
	head[2] = guard_errors(s);

done:
	head[0] = err;
	head[1] = s->kernel_us;
	head[3] = nops;
	for (i = 0; i < nops; i++)
		head[4] += s->out_size[i];
	fwrite(head, 4, 5, out);
	fwrite(s->codes, 4, nops, out);
	for (i = 0; i < nops; i++) {
		uint32_t n;

		if (s->out[i])
			fwrite(s->out[i], 1, s->out_size[i], out);
		else
			for (n = 0; n < s->out_size[i]; n++)
				fputc(0, out);
	}
	free_state(s);
	tlog("status %d, %u us in kernels, %u guard bytes overwritten\n", err, head[1], head[2]);
	return err;
}

static void log_device(cl_device_id dev)
{
	static const struct { cl_device_info what; const char *name; } strings[] = {
		{ CL_DEVICE_NAME, "name" }, { CL_DEVICE_VERSION, "version" },
		{ CL_DEVICE_PROFILE, "profile" }, { CL_DEVICE_EXTENSIONS, "extensions" },
	};
	char buf[512];
	cl_uint mhz = 0;
	cl_device_fp_config fp = 0;
	unsigned int i;

	for (i = 0; i < sizeof(strings) / sizeof(strings[0]); i++)
		if (clGetDeviceInfo(dev, strings[i].what, sizeof(buf), buf, NULL) == CL_SUCCESS)
			tlog("device %s: %s\n", strings[i].name, buf);
	clGetDeviceInfo(dev, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(mhz), &mhz, NULL);
	clGetDeviceInfo(dev, CL_DEVICE_SINGLE_FP_CONFIG, sizeof(fp), &fp, NULL);
	tlog("device clock: %u MHz (scePower gpu %d, xbar %d, arm %d), fp config 0x%x\n", mhz,
	     scePowerGetGpuClockFrequency(), scePowerGetGpuXbarClockFrequency(),
	     scePowerGetArmClockFrequency(), (unsigned int)fp);
}

static int run_pack(void)
{
	cl_platform_id plat;
	cl_device_id dev;
	cl_context ctx;
	cl_command_queue q;
	SceUID uid;
	FILE *f, *out;
	long size;
	uint8_t *pack;
	uint32_t count, i, done = 0, failed = 0;
	cl_int err;
	static const uint32_t header[2] = { RESULTS_MAGIC, 0 };

	f = fopen(PACK_PATH, "rb");
	if (!f) {
		tlog("no %s\n", PACK_PATH);
		return -1;
	}
	fseek(f, 0, SEEK_END);
	size = ftell(f);
	fseek(f, 0, SEEK_SET);
	uid = sceKernelAllocMemBlock("pack", SCE_KERNEL_MEMBLOCK_TYPE_USER_RW,
				     ALIGN(size + 1, 4096), NULL);
	if (uid < 0 || sceKernelGetMemBlockBase(uid, (void **)&pack) < 0 ||
	    fread(pack, 1, size, f) != (size_t)size) {
		tlog("cannot load %s (%ld bytes): 0x%08X\n", PACK_PATH, size, uid);
		fclose(f);
		return -1;
	}
	fclose(f);
	memcpy(&count, pack + 4, 4);
	if (size < 8 || *(uint32_t *)pack != PACK_MAGIC || 8 + 4 * (long)count > size) {
		tlog("%s is not a test pack\n", PACK_PATH);
		sceKernelFreeMemBlock(uid);
		return -1;
	}
	tlog("%u tests, %ld bytes\n", count, size);

	err = clGetPlatformIDs(1, &plat, NULL);
	if (err == CL_SUCCESS)
		err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
	if (err != CL_SUCCESS) {
		tlog("no device: %d\n", err);
		return -1;
	}
	log_device(dev);
	ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
	if (err != CL_SUCCESS) {
		tlog("clCreateContext: %d\n", err);
		return -1;
	}
	tlog("with a context: scePower gpu %d MHz\n", scePowerGetGpuClockFrequency());
	q = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err);
	if (err != CL_SUCCESS) {
		tlog("clCreateCommandQueue: %d\n", err);
		return -1;
	}
	out = fopen(RESULTS_PATH, "w+b");
	if (!out) {
		tlog("cannot write %s\n", RESULTS_PATH);
		return -1;
	}
	fwrite(header, 4, 2, out);
	for (i = 0; i < count; i++) {
		uint32_t off = ((uint32_t *)pack)[2 + i];
		const pack_test *t = (const pack_test *)(pack + off);
		char name[33];

		if (off > (uint32_t)size || (uint32_t)size - off < sizeof(*t)) {
			tlog("test %u: bad offset\n", i);
			break;
		}
		memcpy(name, t->name, 32);
		name[32] = 0;
		tlog("test %u %s\n", i, name);
		failed += run_test(ctx, q, dev, pack + off, size - off, out) != CL_SUCCESS;
		/* keep the count current, so a crash leaves the results so far */
		done = i + 1;
		fseek(out, 4, SEEK_SET);
		fwrite(&done, 4, 1, out);
		fseek(out, 0, SEEK_END);
		fflush(out);
	}
	fclose(out);
	tlog("%u tests run, %u failed to run\n", done, failed);
	clReleaseCommandQueue(q);
	clReleaseContext(ctx);
	sceKernelFreeMemBlock(uid);
	return failed ? -1 : 0;
}

int main(void)
{
	sceIoMkdir("ux0:data/ocl", 0777);
	log_file = fopen("ux0:data/ocl/harness.log", "w");
	if (!log_file)
		return 1;
	tlog("harness start\n");
	tlog("run_pack: %d\n", run_pack());
	tlog("HARNESS DONE\n");
	fclose(log_file);
	/* stay alive so the run coordinator sees the app in the foreground and closes it */
	for (;;)
		sceKernelDelayThread(1000000);
	return 0;
}
