/*
 * The runtime API against what the OpenCL 1.1 specification (revision 44)
 * requires of an embedded profile device: the platform and device queries
 * with the minimums of section 10's version of table 4.3, contexts, command
 * queues, buffers, the image and sampler calls of a device without images,
 * programs without a compiler, kernels, events, markers and barriers, and
 * profiling.  Each failed check is logged with the section it comes from;
 * ep_api_run returns how many failed.
 *
 * prog is tests/kernels/ep_api.cl, built.
 */
#include <CL/cl.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ep_api.h"

static char *report;
static size_t report_size, report_len;
static int checks, failures;
static ep_log_fn log_fn;

static void fail(const char *sect, int line, const char *fmt, ...)
{
	char msg[256];
	va_list ap;
	int n;

	va_start(ap, fmt);
	vsnprintf(msg, sizeof(msg), fmt, ap);
	va_end(ap);
	failures++;
	log_fn("ep_api: FAIL %s (line %d): %s\n", sect, line, msg);
	n = snprintf(report + report_len, report_size - report_len, "%s: %s\n", sect, msg);
	if (n > 0 && report_len + n < report_size)
		report_len += n;
}

#define CHECK(sect, cond, ...) do { \
	checks++; \
	if (!(cond)) \
		fail(sect, __LINE__, __VA_ARGS__); \
} while (0)
#define CHECK_ERR(sect, call, want) do { \
	cl_int e_ = (call); \
	CHECK(sect, e_ == (want), "%s returned %d, want %s (%d)", #call, e_, #want, want); \
} while (0)

static const char *str_info_platform(cl_platform_id p, cl_platform_info what, char *buf, size_t n)
{
	buf[0] = 0;
	clGetPlatformInfo(p, what, n, buf, NULL);
	return buf;
}

static const char *str_info_device(cl_device_id d, cl_device_info what, char *buf, size_t n)
{
	buf[0] = 0;
	clGetDeviceInfo(d, what, n, buf, NULL);
	return buf;
}

static cl_ulong dev_ulong(cl_device_id d, cl_device_info what, size_t want_size)
{
	union { cl_uint u; cl_ulong ul; size_t sz; unsigned char b[16]; } v;
	size_t size = 0;

	memset(&v, 0, sizeof(v));
	if (clGetDeviceInfo(d, what, sizeof(v), &v, &size) != CL_SUCCESS)
		return (cl_ulong)-1;
	CHECK("4.2 table 4.3", size == want_size, "device info 0x%x is %u bytes, want %u",
	      (unsigned)what, (unsigned)size, (unsigned)want_size);
	if (size == sizeof(cl_ulong))
		return v.ul;
	if (size == sizeof(cl_uint))
		return v.u;
	return v.sz;
}

static int has_word(const char *list, const char *word)
{
	size_t n = strlen(word);
	const char *p;

	for (p = list; (p = strstr(p, word)) != NULL; p += n)
		if ((p == list || p[-1] == ' ') && (p[n] == ' ' || p[n] == 0))
			return 1;
	return 0;
}

static void platform_checks(cl_platform_id plat)
{
	char buf[512];
	cl_platform_id plats[4];
	cl_uint n = 0;
	size_t size;

	CHECK_ERR("4.1", clGetPlatformIDs(0, NULL, &n), CL_SUCCESS);
	CHECK("4.1", n >= 1, "%u platforms", n);
	CHECK_ERR("4.1", clGetPlatformIDs(0, plats, NULL), CL_INVALID_VALUE);
	CHECK_ERR("4.1", clGetPlatformIDs(1, NULL, NULL), CL_INVALID_VALUE);

	/* section 10: CL_PLATFORM_PROFILE returns EMBEDDED_PROFILE if only that is supported */
	CHECK("10", !strcmp(str_info_platform(plat, CL_PLATFORM_PROFILE, buf, sizeof(buf)),
			    "EMBEDDED_PROFILE"), "CL_PLATFORM_PROFILE is \"%s\"", buf);
	/* table 4.1: "OpenCL<space><major.minor><space><platform-specific information>" */
	CHECK("4.1 table 4.1", !strncmp(str_info_platform(plat, CL_PLATFORM_VERSION, buf, sizeof(buf)),
					"OpenCL 1.1 ", 11), "CL_PLATFORM_VERSION is \"%s\"", buf);
	CHECK("4.1 table 4.1", str_info_platform(plat, CL_PLATFORM_NAME, buf, sizeof(buf))[0],
	      "empty CL_PLATFORM_NAME");
	CHECK("4.1 table 4.1", str_info_platform(plat, CL_PLATFORM_VENDOR, buf, sizeof(buf))[0],
	      "empty CL_PLATFORM_VENDOR");
	str_info_platform(plat, CL_PLATFORM_EXTENSIONS, buf, sizeof(buf));
	CHECK_ERR("4.1", clGetPlatformInfo(plat, CL_PLATFORM_PROFILE, 0, NULL, &size), CL_SUCCESS);
	CHECK("4.1", size == sizeof("EMBEDDED_PROFILE"), "CL_PLATFORM_PROFILE size %u", (unsigned)size);
	CHECK_ERR("4.1", clGetPlatformInfo(plat, CL_PLATFORM_PROFILE, 4, buf, NULL), CL_INVALID_VALUE);
	CHECK_ERR("4.1", clGetPlatformInfo(plat, 0x7fff, sizeof(buf), buf, NULL), CL_INVALID_VALUE);
}

static void device_checks(cl_platform_id plat, cl_device_id dev)
{
	char buf[1024], ext[1024];
	cl_device_id devs[4];
	cl_uint n = 0, u;
	cl_ulong global, alloc, v;
	size_t sizes[3], size;
	cl_device_fp_config fp;
	cl_command_queue_properties qp;
	cl_device_exec_capabilities exec;
	cl_bool b;
	cl_platform_id p;
	cl_device_type type;

	/* 4.2 */
	CHECK_ERR("4.2", clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 4, devs, &n), CL_SUCCESS);
	CHECK("4.2", n == 1 && devs[0] == dev, "%u devices", n);
	CHECK_ERR("4.2", clGetDeviceIDs(plat, CL_DEVICE_TYPE_DEFAULT, 1, devs, NULL), CL_SUCCESS);
	CHECK_ERR("4.2", clGetDeviceIDs(plat, CL_DEVICE_TYPE_CPU, 1, devs, &n), CL_DEVICE_NOT_FOUND);
	CHECK_ERR("4.2", clGetDeviceIDs(plat, CL_DEVICE_TYPE_ACCELERATOR, 1, devs, &n),
		  CL_DEVICE_NOT_FOUND);
	CHECK_ERR("4.2", clGetDeviceIDs(plat, 0, 1, devs, NULL), CL_INVALID_DEVICE_TYPE);
	CHECK_ERR("4.2", clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, devs, NULL), CL_INVALID_VALUE);
	CHECK_ERR("4.2", clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, NULL, NULL), CL_INVALID_VALUE);

	clGetDeviceInfo(dev, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
	CHECK("4.2 table 4.3", type & CL_DEVICE_TYPE_GPU, "CL_DEVICE_TYPE 0x%x", (unsigned)type);
	clGetDeviceInfo(dev, CL_DEVICE_PLATFORM, sizeof(p), &p, NULL);
	CHECK("4.2 table 4.3", p == plat, "CL_DEVICE_PLATFORM");

	/* the embedded profile minimums (section 10's table 4.3) */
	CHECK("10", dev_ulong(dev, CL_DEVICE_MAX_COMPUTE_UNITS, 4) >= 1, "CL_DEVICE_MAX_COMPUTE_UNITS");
	u = (cl_uint)dev_ulong(dev, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, 4);
	CHECK("10", u >= 3, "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS %u", u);
	CHECK_ERR("10", clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(sizes), sizes, &size),
		  CL_SUCCESS);
	CHECK("10", size == 3 * sizeof(size_t) && sizes[0] >= 1 && sizes[1] >= 1 && sizes[2] >= 1,
	      "CL_DEVICE_MAX_WORK_ITEM_SIZES");
	CHECK("10", dev_ulong(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t)) >= 1,
	      "CL_DEVICE_MAX_WORK_GROUP_SIZE");
	u = (cl_uint)dev_ulong(dev, CL_DEVICE_ADDRESS_BITS, 4);
	str_info_device(dev, CL_DEVICE_EXTENSIONS, ext, sizeof(ext));
	CHECK("10", u == 32 || (u == 64 && has_word(ext, "cles_khr_int64")),
	      "CL_DEVICE_ADDRESS_BITS %u without cles_khr_int64", u);
	global = dev_ulong(dev, CL_DEVICE_GLOBAL_MEM_SIZE, 8);
	alloc = dev_ulong(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, 8);
	CHECK("10", alloc >= global / 4 && alloc >= 1024 * 1024,
	      "CL_DEVICE_MAX_MEM_ALLOC_SIZE %llu < max(%llu / 4, 1 MB)",
	      (unsigned long long)alloc, (unsigned long long)global);
	CHECK("10", dev_ulong(dev, CL_DEVICE_MAX_PARAMETER_SIZE, sizeof(size_t)) >= 256,
	      "CL_DEVICE_MAX_PARAMETER_SIZE");
	/* the largest built-in type of the embedded profile without int64 is int16 (64 bytes) */
	CHECK("10", dev_ulong(dev, CL_DEVICE_MEM_BASE_ADDR_ALIGN, 4) >= 64 * 8,
	      "CL_DEVICE_MEM_BASE_ADDR_ALIGN under the bits of an int16");
	CHECK("10", dev_ulong(dev, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, 4) >= 64,
	      "CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE under the bytes of an int16");
	CHECK_ERR("10", clGetDeviceInfo(dev, CL_DEVICE_SINGLE_FP_CONFIG, sizeof(fp), &fp, NULL), CL_SUCCESS);
	CHECK("10", fp & (CL_FP_ROUND_TO_ZERO | CL_FP_ROUND_TO_NEAREST),
	      "CL_DEVICE_SINGLE_FP_CONFIG 0x%x has neither RTZ nor RTE", (unsigned)fp);
	CHECK("10", dev_ulong(dev, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, 8) >= 1024,
	      "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE");
	CHECK("10", dev_ulong(dev, CL_DEVICE_MAX_CONSTANT_ARGS, 4) >= 4, "CL_DEVICE_MAX_CONSTANT_ARGS");
	CHECK("10", dev_ulong(dev, CL_DEVICE_LOCAL_MEM_SIZE, 8) >= 1024, "CL_DEVICE_LOCAL_MEM_SIZE");
	clGetDeviceInfo(dev, CL_DEVICE_EXECUTION_CAPABILITIES, sizeof(exec), &exec, NULL);
	CHECK("10", exec & CL_EXEC_KERNEL, "CL_DEVICE_EXECUTION_CAPABILITIES lacks CL_EXEC_KERNEL");
	clGetDeviceInfo(dev, CL_DEVICE_QUEUE_PROPERTIES, sizeof(qp), &qp, NULL);
	CHECK("10", qp & CL_QUEUE_PROFILING_ENABLE, "CL_DEVICE_QUEUE_PROPERTIES lacks profiling");

	/* images are optional; without them the image limits needn't be met */
	clGetDeviceInfo(dev, CL_DEVICE_IMAGE_SUPPORT, sizeof(b), &b, NULL);
	if (b) {
		CHECK("10", dev_ulong(dev, CL_DEVICE_MAX_READ_IMAGE_ARGS, 4) >= 8, "read image args");
		CHECK("10", dev_ulong(dev, CL_DEVICE_MAX_WRITE_IMAGE_ARGS, 4) >= 1, "write image args");
		CHECK("10", dev_ulong(dev, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(size_t)) >= 2048, "2D width");
		CHECK("10", dev_ulong(dev, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(size_t)) >= 2048, "2D height");
		CHECK("10", dev_ulong(dev, CL_DEVICE_MAX_SAMPLERS, 4) >= 8, "samplers");
	}

	/* table 4.3: the version strings and the profile */
	CHECK("10", !strcmp(str_info_device(dev, CL_DEVICE_PROFILE, buf, sizeof(buf)), "EMBEDDED_PROFILE"),
	      "CL_DEVICE_PROFILE is \"%s\"", buf);
	CHECK("4.2 table 4.3", !strncmp(str_info_device(dev, CL_DEVICE_VERSION, buf, sizeof(buf)),
					"OpenCL 1.1 ", 11), "CL_DEVICE_VERSION is \"%s\"", buf);
	CHECK("4.2 table 4.3", !strncmp(str_info_device(dev, CL_DEVICE_OPENCL_C_VERSION, buf, sizeof(buf)),
					"OpenCL C 1.1 ", 13), "CL_DEVICE_OPENCL_C_VERSION is \"%s\"", buf);
	{
		unsigned major, minor;

		str_info_device(dev, CL_DRIVER_VERSION, buf, sizeof(buf));
		CHECK("4.2 table 4.3", sscanf(buf, "%u.%u", &major, &minor) == 2,
		      "CL_DRIVER_VERSION \"%s\" isn't <major>.<minor>", buf);
	}
	CHECK("4.2 table 4.3", str_info_device(dev, CL_DEVICE_NAME, buf, sizeof(buf))[0], "empty name");
	CHECK("4.2 table 4.3", str_info_device(dev, CL_DEVICE_VENDOR, buf, sizeof(buf))[0], "empty vendor");

	/* the extensions reported are ones section 10 allows the embedded profile; without
	 * cl_khr_fp64 or cl_khr_fp16 the preferred and native widths are 0 (table 4.3) */
	if (!has_word(ext, "cl_khr_fp64")) {
		CHECK("4.2 table 4.3", dev_ulong(dev, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, 4) == 0,
		      "preferred double width without cl_khr_fp64");
		CHECK("4.2 table 4.3", dev_ulong(dev, CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, 4) == 0,
		      "native double width without cl_khr_fp64");
	} else
		CHECK("10", has_word(ext, "cles_khr_int64"), "cl_khr_fp64 without cles_khr_int64");
	if (!has_word(ext, "cl_khr_fp16")) {
		CHECK("4.2 table 4.3", dev_ulong(dev, CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF, 4) == 0,
		      "preferred half width without cl_khr_fp16");
		CHECK("4.2 table 4.3", dev_ulong(dev, CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF, 4) == 0,
		      "native half width without cl_khr_fp16");
	}
	v = dev_ulong(dev, CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(size_t));
	CHECK("4.2 table 4.3", v > 0, "CL_DEVICE_PROFILING_TIMER_RESOLUTION 0");
	clGetDeviceInfo(dev, CL_DEVICE_AVAILABLE, sizeof(b), &b, NULL);
	CHECK("4.2 table 4.3", b, "CL_DEVICE_AVAILABLE false");
	CHECK_ERR("4.2", clGetDeviceInfo(dev, 0x7fff, sizeof(buf), buf, NULL), CL_INVALID_VALUE);
	CHECK_ERR("4.2", clGetDeviceInfo(dev, CL_DEVICE_NAME, 1, buf, NULL), CL_INVALID_VALUE);
	CHECK_ERR("4.2", clGetDeviceInfo(NULL, CL_DEVICE_NAME, sizeof(buf), buf, NULL), CL_INVALID_DEVICE);
}

static void CL_CALLBACK ctx_notify(const char *errinfo, const void *info, size_t cb, void *user)
{
}

static void context_checks(cl_platform_id plat, cl_device_id dev)
{
	cl_context_properties props[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)plat, 0 };
	cl_context_properties bad[] = { 0x7fff, 0, 0 };
	cl_context_properties got[8];
	cl_context c;
	cl_device_id d;
	cl_uint u;
	size_t size;
	cl_int err;
	int dummy;

	/* 4.3 */
	c = clCreateContext(props, 1, &dev, ctx_notify, NULL, &err);
	CHECK("4.3", c && err == CL_SUCCESS, "clCreateContext with CL_CONTEXT_PLATFORM: %d", err);
	if (c) {
		CHECK_ERR("4.3", clGetContextInfo(c, CL_CONTEXT_NUM_DEVICES, sizeof(u), &u, NULL), CL_SUCCESS);
		CHECK("4.3", u == 1, "CL_CONTEXT_NUM_DEVICES %u", u);
		CHECK_ERR("4.3", clGetContextInfo(c, CL_CONTEXT_DEVICES, sizeof(d), &d, NULL), CL_SUCCESS);
		CHECK("4.3", d == dev, "CL_CONTEXT_DEVICES");
		CHECK_ERR("4.3", clGetContextInfo(c, CL_CONTEXT_PROPERTIES, sizeof(got), got, &size),
			  CL_SUCCESS);
		CHECK("4.3", size == sizeof(props) && got[0] == CL_CONTEXT_PLATFORM &&
		      got[1] == (cl_context_properties)plat && got[2] == 0, "CL_CONTEXT_PROPERTIES");
		CHECK_ERR("4.3", clRetainContext(c), CL_SUCCESS);
		clGetContextInfo(c, CL_CONTEXT_REFERENCE_COUNT, sizeof(u), &u, NULL);
		CHECK("4.3", u == 2, "CL_CONTEXT_REFERENCE_COUNT %u after a retain", u);
		CHECK_ERR("4.3", clReleaseContext(c), CL_SUCCESS);
		CHECK_ERR("4.3", clGetContextInfo(c, 0x7fff, sizeof(u), &u, NULL), CL_INVALID_VALUE);
		CHECK_ERR("4.3", clReleaseContext(c), CL_SUCCESS);
	}
	c = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
	CHECK("4.3", c && err == CL_SUCCESS, "clCreateContext without properties: %d", err);
	if (c) {
		CHECK_ERR("4.3", clGetContextInfo(c, CL_CONTEXT_PROPERTIES, 0, NULL, &size), CL_SUCCESS);
		/* either nothing or a 0 terminator */
		CHECK("4.3", size == 0 || size == sizeof(cl_context_properties),
		      "CL_CONTEXT_PROPERTIES of a context without is %u bytes", (unsigned)size);
		clReleaseContext(c);
	}
	c = clCreateContext(bad, 1, &dev, NULL, NULL, &err);
	CHECK("4.3", !c && err == CL_INVALID_PROPERTY, "unknown context property: %d", err);
	c = clCreateContext(NULL, 0, &dev, NULL, NULL, &err);
	CHECK("4.3", !c && err == CL_INVALID_VALUE, "no devices: %d", err);
	c = clCreateContext(NULL, 1, NULL, NULL, NULL, &err);
	CHECK("4.3", !c && err == CL_INVALID_VALUE, "NULL devices: %d", err);
	c = clCreateContext(NULL, 1, &dev, NULL, &dummy, &err);
	CHECK("4.3", !c && err == CL_INVALID_VALUE, "user_data without pfn_notify: %d", err);
	c = clCreateContextFromType(props, CL_DEVICE_TYPE_GPU, NULL, NULL, &err);
	CHECK("4.3", c && err == CL_SUCCESS, "clCreateContextFromType(GPU): %d", err);
	if (c)
		clReleaseContext(c);
	c = clCreateContextFromType(props, CL_DEVICE_TYPE_DEFAULT, NULL, NULL, &err);
	CHECK("4.3", c && err == CL_SUCCESS, "clCreateContextFromType(DEFAULT): %d", err);
	if (c)
		clReleaseContext(c);
	c = clCreateContextFromType(props, CL_DEVICE_TYPE_CPU, NULL, NULL, &err);
	CHECK("4.3", !c && err == CL_DEVICE_NOT_FOUND, "clCreateContextFromType(CPU): %d", err);
}

static void queue_checks(cl_context ctx, cl_device_id dev)
{
	cl_command_queue q;
	cl_command_queue_properties p;
	cl_context c;
	cl_device_id d;
	cl_int err;

	/* 5.1 */
	q = clCreateCommandQueue(ctx, dev, 0, &err);
	CHECK("5.1", q && err == CL_SUCCESS, "in-order queue: %d", err);
	if (q) {
		clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(c), &c, NULL);
		clGetCommandQueueInfo(q, CL_QUEUE_DEVICE, sizeof(d), &d, NULL);
		clGetCommandQueueInfo(q, CL_QUEUE_PROPERTIES, sizeof(p), &p, NULL);
		CHECK("5.1", c == ctx && d == dev && p == 0, "queue info");
		CHECK_ERR("5.1", clFlush(q), CL_SUCCESS);
		CHECK_ERR("5.1", clFinish(q), CL_SUCCESS);
		clReleaseCommandQueue(q);
	}
	q = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE |
				 CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
	clGetDeviceInfo(dev, CL_DEVICE_QUEUE_PROPERTIES, sizeof(p), &p, NULL);
	if (p & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
		CHECK("5.1", q && err == CL_SUCCESS, "out-of-order profiling queue: %d", err);
	} else
		CHECK("5.1", !q && err == CL_INVALID_QUEUE_PROPERTIES, "unsupported out-of-order: %d", err);
	if (q)
		clReleaseCommandQueue(q);
	q = clCreateCommandQueue(ctx, dev, (cl_command_queue_properties)1 << 20, &err);
	CHECK("5.1", !q && err == CL_INVALID_VALUE, "unknown queue property: %d", err);
	q = clCreateCommandQueue(NULL, dev, 0, &err);
	CHECK("5.1", !q && err == CL_INVALID_CONTEXT, "NULL context: %d", err);
	q = clCreateCommandQueue(ctx, NULL, 0, &err);
	CHECK("5.1", !q && err == CL_INVALID_DEVICE, "NULL device: %d", err);
}

static int destructor_calls;

static void CL_CALLBACK mem_destructor(cl_mem m, void *user)
{
	destructor_calls += *(int *)user;
}

static void buffer_checks(cl_context ctx, cl_command_queue q, cl_device_id dev)
{
	static int host[256];
	cl_mem b, sub;
	cl_buffer_region region;
	cl_mem_object_type type;
	cl_mem_flags flags;
	size_t size, offset;
	void *ptr;
	cl_uint u, align;
	cl_ulong max_alloc;
	cl_mem assoc;
	cl_int err;
	int one = 1, i, *map;

	/* 5.2.1 */
	b = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 0, NULL, &err);
	CHECK("5.2.1", !b && err == CL_INVALID_BUFFER_SIZE, "size 0: %d", err);
	clGetDeviceInfo(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc), &max_alloc, NULL);
	b = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)max_alloc + 1, NULL, &err);
	CHECK("5.2.1", !b && err == CL_INVALID_BUFFER_SIZE, "over CL_DEVICE_MAX_MEM_ALLOC_SIZE: %d", err);
	b = clCreateBuffer(ctx, CL_MEM_USE_HOST_PTR, 64, NULL, &err);
	CHECK("5.2.1", !b && err == CL_INVALID_HOST_PTR, "CL_MEM_USE_HOST_PTR without a pointer: %d", err);
	b = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 64, host, &err);
	CHECK("5.2.1", !b && err == CL_INVALID_HOST_PTR, "a pointer without a host flag: %d", err);
	b = clCreateBuffer(ctx, CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR, 64, host, &err);
	CHECK("5.2.1", !b && err == CL_INVALID_VALUE, "USE_HOST_PTR | ALLOC_HOST_PTR: %d", err);
	b = clCreateBuffer(ctx, CL_MEM_USE_HOST_PTR | CL_MEM_COPY_HOST_PTR, 64, host, &err);
	CHECK("5.2.1", !b && err == CL_INVALID_VALUE, "USE_HOST_PTR | COPY_HOST_PTR: %d", err);
	b = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_WRITE_ONLY, 64, NULL, &err);
	CHECK("5.2.1", !b && err == CL_INVALID_VALUE, "READ_ONLY | WRITE_ONLY: %d", err);
	b = clCreateBuffer(ctx, (cl_mem_flags)1 << 20, 64, NULL, &err);
	CHECK("5.2.1", !b && err == CL_INVALID_VALUE, "unknown flag: %d", err);
	b = clCreateBuffer(NULL, CL_MEM_READ_WRITE, 64, NULL, &err);
	CHECK("5.2.1", !b && err == CL_INVALID_CONTEXT, "NULL context: %d", err);

	for (i = 0; i < 256; i++)
		host[i] = i;
	b = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(host), host, &err);
	CHECK("5.2.1", b && err == CL_SUCCESS, "CL_MEM_USE_HOST_PTR: %d", err);
	if (b) {
		/* 5.4.3 */
		clGetMemObjectInfo(b, CL_MEM_TYPE, sizeof(type), &type, NULL);
		clGetMemObjectInfo(b, CL_MEM_FLAGS, sizeof(flags), &flags, NULL);
		clGetMemObjectInfo(b, CL_MEM_SIZE, sizeof(size), &size, NULL);
		clGetMemObjectInfo(b, CL_MEM_HOST_PTR, sizeof(ptr), &ptr, NULL);
		CHECK("5.4.3", type == CL_MEM_OBJECT_BUFFER && flags == (CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR) &&
		      size == sizeof(host) && ptr == host, "buffer info");
		/* a USE_HOST_PTR buffer maps at host_ptr (5.4.2.1 is about access; 5.2.4 table) */
		map = clEnqueueMapBuffer(q, b, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 16, 64, 0, NULL, NULL, &err);
		CHECK("5.2.4", map && err == CL_SUCCESS && map[0] == 4, "map: %d", err);
		clGetMemObjectInfo(b, CL_MEM_MAP_COUNT, sizeof(u), &u, NULL);
		CHECK("5.4.3", u == 1, "CL_MEM_MAP_COUNT %u while mapped", u);
		if (map)
			CHECK_ERR("5.4.2", clEnqueueUnmapMemObject(q, b, map, 0, NULL, NULL), CL_SUCCESS);
		CHECK_ERR("5.4.2", clEnqueueUnmapMemObject(q, b, host + 100, 0, NULL, NULL), CL_INVALID_VALUE);
		clFinish(q);

		/* 5.2.1 clCreateSubBuffer */
		clGetDeviceInfo(dev, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(align), &align, NULL);
		region.origin = align / 8;
		region.size = 64;
		sub = clCreateSubBuffer(b, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
		CHECK("5.2.1", sub && err == CL_SUCCESS, "aligned sub-buffer: %d", err);
		if (sub) {
			clGetMemObjectInfo(sub, CL_MEM_ASSOCIATED_MEMOBJECT, sizeof(assoc), &assoc, NULL);
			clGetMemObjectInfo(sub, CL_MEM_OFFSET, sizeof(offset), &offset, NULL);
			clGetMemObjectInfo(sub, CL_MEM_HOST_PTR, sizeof(ptr), &ptr, NULL);
			CHECK("5.4.3", assoc == b && offset == region.origin &&
			      ptr == (char *)host + region.origin, "sub-buffer info");
			clGetMemObjectInfo(b, CL_MEM_REFERENCE_COUNT, sizeof(u), &u, NULL);
			clReleaseMemObject(sub);
		}
		region.origin = 4;
		sub = clCreateSubBuffer(b, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
		CHECK("5.2.1", !sub && err == CL_MISALIGNED_SUB_BUFFER_OFFSET, "misaligned sub-buffer: %d", err);
		region.origin = 0;
		region.size = sizeof(host) + 1;
		sub = clCreateSubBuffer(b, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
		CHECK("5.2.1", !sub && err == CL_INVALID_VALUE, "sub-buffer out of bounds: %d", err);
		region.size = 0;
		sub = clCreateSubBuffer(b, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
		CHECK("5.2.1", !sub && err == CL_INVALID_BUFFER_SIZE, "empty sub-buffer: %d", err);

		/* 5.4.1: destructor callbacks run when the memory object is deleted */
		destructor_calls = 0;
		CHECK_ERR("5.4.1", clSetMemObjectDestructorCallback(b, mem_destructor, &one), CL_SUCCESS);
		CHECK_ERR("5.4.1", clSetMemObjectDestructorCallback(b, NULL, NULL), CL_INVALID_VALUE);
		CHECK_ERR("5.4.1", clRetainMemObject(b), CL_SUCCESS);
		clGetMemObjectInfo(b, CL_MEM_REFERENCE_COUNT, sizeof(u), &u, NULL);
		CHECK("5.4.3", u == 2, "CL_MEM_REFERENCE_COUNT %u after a retain", u);
		clReleaseMemObject(b);
		CHECK("5.4.1", destructor_calls == 0, "destructor ran with a reference left");
		clReleaseMemObject(b);
		clFinish(q);
		CHECK("5.4.1", destructor_calls == 1, "destructor ran %d times", destructor_calls);
	}
	CHECK_ERR("5.4.1", clReleaseMemObject(NULL), CL_INVALID_MEM_OBJECT);

	/* 5.2.2: reads and writes are bounds checked */
	b = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 64, NULL, &err);
	if (b) {
		CHECK_ERR("5.2.2", clEnqueueReadBuffer(q, b, CL_TRUE, 32, 64, host, 0, NULL, NULL),
			  CL_INVALID_VALUE);
		CHECK_ERR("5.2.2", clEnqueueWriteBuffer(q, b, CL_TRUE, 0, 64, NULL, 0, NULL, NULL),
			  CL_INVALID_VALUE);
		CHECK_ERR("5.2.2", clEnqueueReadBuffer(q, b, CL_TRUE, 0, 4, host, 1, NULL, NULL),
			  CL_INVALID_EVENT_WAIT_LIST);
		clReleaseMemObject(b);
	}
}

static void image_checks(cl_context ctx, cl_command_queue q, cl_device_id dev)
{
	cl_image_format fmt = { CL_RGBA, CL_UNORM_INT8 }, fmts[16];
	cl_uint n = 99;
	cl_bool images;
	cl_sampler s;
	cl_mem m;
	cl_int err;

	clGetDeviceInfo(dev, CL_DEVICE_IMAGE_SUPPORT, sizeof(images), &images, NULL);
	if (images)
		return;
	/* 5.3.1: "CL_INVALID_OPERATION if there are no devices in context that support images" */
	m = clCreateImage2D(ctx, CL_MEM_READ_ONLY, &fmt, 16, 16, 0, NULL, &err);
	CHECK("5.3.1", !m && err == CL_INVALID_OPERATION, "clCreateImage2D without images: %d", err);
	m = clCreateImage3D(ctx, CL_MEM_READ_ONLY, &fmt, 16, 16, 2, 0, 0, NULL, &err);
	/* section 10 item 2: clCreateImage3D fails with CL_INVALID_OPERATION */
	CHECK("10, 5.3.1", !m && err == CL_INVALID_OPERATION, "clCreateImage3D without images: %d", err);
	/* 5.3.2: no formats to list */
	err = clGetSupportedImageFormats(ctx, CL_MEM_READ_ONLY, CL_MEM_OBJECT_IMAGE2D, 16, fmts, &n);
	CHECK("5.3.2", err == CL_SUCCESS && n == 0, "clGetSupportedImageFormats: %d, %u formats", err, n);
	/* 5.5.1 */
	s = clCreateSampler(ctx, CL_FALSE, CL_ADDRESS_CLAMP, CL_FILTER_NEAREST, &err);
	CHECK("5.5.1", !s && err == CL_INVALID_OPERATION, "clCreateSampler without images: %d", err);
}

static int build_notified;

static void CL_CALLBACK build_notify(cl_program p, void *user)
{
	build_notified++;
}

static void program_checks(cl_context ctx, cl_device_id dev, cl_program prog)
{
	const char *src = "__kernel void k(__global int *a) { a[0] = 1; }";
	static const unsigned char junk[64] = "not a program binary";
	const unsigned char *bin = junk;
	size_t size = sizeof(junk), sizes[2];
	cl_program p;
	cl_build_status st;
	cl_bool compiler;
	cl_uint u;
	cl_int err, bst = 0;
	char buf[256];

	clGetDeviceInfo(dev, CL_DEVICE_COMPILER_AVAILABLE, sizeof(compiler), &compiler, NULL);
	/* 5.6.1, 5.6.2 */
	p = clCreateProgramWithSource(ctx, 1, &src, NULL, &err);
	CHECK("5.6.1", p && err == CL_SUCCESS, "clCreateProgramWithSource: %d", err);
	if (p) {
		clGetProgramInfo(p, CL_PROGRAM_SOURCE, sizeof(buf), buf, NULL);
		CHECK("5.6.5", !strcmp(buf, src), "CL_PROGRAM_SOURCE");
		if (!compiler)
			/* "CL_COMPILER_NOT_AVAILABLE if program is created with clCreateProgramWithSource
			 * and a compiler is not available" */
			CHECK_ERR("5.6.2", clBuildProgram(p, 1, &dev, NULL, NULL, NULL), CL_COMPILER_NOT_AVAILABLE);
		clReleaseProgram(p);
	}
	CHECK_ERR("5.6.4", clUnloadCompiler(), CL_SUCCESS);
	p = clCreateProgramWithSource(ctx, 0, &src, NULL, &err);
	CHECK("5.6.1", !p && err == CL_INVALID_VALUE, "no strings: %d", err);

	p = clCreateProgramWithBinary(ctx, 1, &dev, &size, &bin, &bst, &err);
	CHECK("5.6.1", !p && err == CL_INVALID_BINARY && bst == CL_INVALID_BINARY,
	      "a bad binary: %d, status %d", err, bst);
	size = 0;
	p = clCreateProgramWithBinary(ctx, 1, &dev, &size, &bin, &bst, &err);
	CHECK("5.6.1", !p && err == CL_INVALID_VALUE, "a binary of 0 bytes: %d", err);

	/* 5.6.5 on the harness's program */
	clGetProgramInfo(prog, CL_PROGRAM_NUM_DEVICES, sizeof(u), &u, NULL);
	CHECK("5.6.5", u == 1, "CL_PROGRAM_NUM_DEVICES %u", u);
	CHECK_ERR("5.6.5", clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES, sizeof(sizes), sizes, &size),
		  CL_SUCCESS);
	CHECK("5.6.5", size == sizeof(size_t) && sizes[0] > 0, "CL_PROGRAM_BINARY_SIZES");
	if (sizes[0] > 0 && sizes[0] < (1u << 24)) {
		unsigned char *copy = malloc(sizes[0]);
		const unsigned char *cp = copy;

		if (copy) {
			CHECK_ERR("5.6.5", clGetProgramInfo(prog, CL_PROGRAM_BINARIES, sizeof(copy), &copy, NULL),
				  CL_SUCCESS);
			/* a program made from its own binary builds again */
			size = sizes[0];
			p = clCreateProgramWithBinary(ctx, 1, &dev, &size, &cp, &bst, &err);
			CHECK("5.6.1", p && err == CL_SUCCESS && bst == CL_SUCCESS,
			      "program from CL_PROGRAM_BINARIES: %d", err);
			if (p) {
				build_notified = 0;
				CHECK_ERR("5.6.2", clBuildProgram(p, 1, &dev, "-cl-fast-relaxed-math", build_notify,
								  NULL), CL_SUCCESS);
				CHECK("5.6.2", build_notified == 1, "build callback ran %d times", build_notified);
				clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_STATUS, sizeof(st), &st, NULL);
				CHECK("5.6.5", st == CL_BUILD_SUCCESS, "CL_PROGRAM_BUILD_STATUS %d", st);
				CHECK_ERR("5.6.5", clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, sizeof(buf), buf,
									 NULL), CL_SUCCESS);
				CHECK_ERR("5.6.2", clBuildProgram(p, 1, &dev, NULL, NULL, &bst), CL_INVALID_VALUE);
				clReleaseProgram(p);
			}
			free(copy);
		}
	}
}

static void kernel_checks(cl_context ctx, cl_command_queue q, cl_device_id dev, cl_program prog)
{
	cl_kernel k, ks[8];
	cl_uint n, u;
	size_t wg, cwg[3], size, g = 4, l = 2;
	cl_ulong lmem;
	cl_int err;
	cl_mem b;
	char buf[64];
	unsigned int out[6];

	/* 5.7.1 */
	k = clCreateKernel(prog, "no_such_kernel", &err);
	CHECK("5.7.1", !k && err == CL_INVALID_KERNEL_NAME, "unknown kernel: %d", err);
	k = clCreateKernel(prog, NULL, &err);
	CHECK("5.7.1", !k && err == CL_INVALID_VALUE, "NULL name: %d", err);
	CHECK_ERR("5.7.1", clCreateKernelsInProgram(prog, 0, NULL, &n), CL_SUCCESS);
	CHECK("5.7.1", n == 3, "clCreateKernelsInProgram counts %u kernels, want 3", n);
	CHECK_ERR("5.7.1", clCreateKernelsInProgram(prog, 1, ks, NULL), CL_INVALID_VALUE);
	if (clCreateKernelsInProgram(prog, 8, ks, &n) == CL_SUCCESS)
		while (n)
			clReleaseKernel(ks[--n]);

	k = clCreateKernel(prog, "args", &err);
	CHECK("5.7.1", k && err == CL_SUCCESS, "clCreateKernel: %d", err);
	if (!k)
		return;
	/* 5.7.3 */
	clGetKernelInfo(k, CL_KERNEL_FUNCTION_NAME, sizeof(buf), buf, NULL);
	clGetKernelInfo(k, CL_KERNEL_NUM_ARGS, sizeof(u), &u, NULL);
	CHECK("5.7.3", !strcmp(buf, "args") && u == 4, "kernel info: \"%s\", %u args", buf, u);
	CHECK_ERR("5.7.3", clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof(wg), &wg, NULL),
		  CL_SUCCESS);
	CHECK("5.7.3", wg >= 1, "CL_KERNEL_WORK_GROUP_SIZE %u", (unsigned)wg);
	CHECK_ERR("5.7.3", clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(cwg),
						   cwg, NULL), CL_SUCCESS);
	CHECK("5.7.3", cwg[0] == 0 && cwg[1] == 0 && cwg[2] == 0, "CL_KERNEL_COMPILE_WORK_GROUP_SIZE");
	CHECK_ERR("5.7.3", clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(lmem), &lmem,
						   NULL), CL_SUCCESS);
	CHECK_ERR("5.7.3", clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
						   sizeof(size), &size, NULL), CL_SUCCESS);
	CHECK_ERR("5.7.3", clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(lmem), &lmem,
						   NULL), CL_SUCCESS);
	/* 5.7.2 */
	CHECK_ERR("5.7.2", clSetKernelArg(k, 4, sizeof(cl_mem), &b), CL_INVALID_ARG_INDEX);
	CHECK_ERR("5.7.2", clSetKernelArg(k, 1, 2, &u), CL_INVALID_ARG_SIZE);
	CHECK_ERR("5.7.2", clSetKernelArg(k, 2, 16, &u), CL_INVALID_ARG_VALUE);
	CHECK_ERR("5.7.2", clSetKernelArg(k, 2, 0, NULL), CL_INVALID_ARG_SIZE);
	/* 5.8: arguments not all set */
	CHECK_ERR("5.8", clEnqueueNDRangeKernel(q, k, 1, NULL, &g, NULL, 0, NULL, NULL), CL_INVALID_KERNEL_ARGS);
	clReleaseKernel(k);

	/* 5.8: the work-group size limits */
	k = clCreateKernel(prog, "task", &err);
	b = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(out), NULL, &err);
	if (!k || !b)
		return;
	clSetKernelArg(k, 0, sizeof(b), &b);
	clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(wg), &wg, NULL);
	if (wg == 1)
		CHECK_ERR("5.8", clEnqueueNDRangeKernel(q, k, 1, NULL, &g, &l, 0, NULL, NULL),
			  CL_INVALID_WORK_GROUP_SIZE);
	l = 3;
	CHECK_ERR("5.8", clEnqueueNDRangeKernel(q, k, 1, NULL, &g, &l, 0, NULL, NULL),
		  CL_INVALID_WORK_GROUP_SIZE);
	CHECK_ERR("5.8", clEnqueueNDRangeKernel(q, k, 0, NULL, &g, NULL, 0, NULL, NULL), CL_INVALID_WORK_DIMENSION);
	CHECK_ERR("5.8", clEnqueueNDRangeKernel(q, k, 4, NULL, &g, NULL, 0, NULL, NULL), CL_INVALID_WORK_DIMENSION);
	CHECK_ERR("5.8", clEnqueueNDRangeKernel(q, k, 1, NULL, NULL, NULL, 0, NULL, NULL),
		  CL_INVALID_GLOBAL_WORK_SIZE);
	/* 5.8 clEnqueueTask: work_dim 1, global and local size 1 */
	memset(out, 0, sizeof(out));
	CHECK_ERR("5.8", clEnqueueTask(q, k, 0, NULL, NULL), CL_SUCCESS);
	CHECK_ERR("5.8", clEnqueueReadBuffer(q, b, CL_TRUE, 0, sizeof(out), out, 0, NULL, NULL), CL_SUCCESS);
	CHECK("5.8", out[0] == 1 && out[1] == 1 && out[2] == 1 && out[3] == 1 && out[4] == 0 &&
	      out[5] == 0x7a5c, "clEnqueueTask ran with dim %u, global %u, local %u, groups %u, id %u (%x)",
	      out[0], out[1], out[2], out[3], out[4], out[5]);
	/* 5.8 clEnqueueNativeKernel: "CL_INVALID_OPERATION if device cannot execute the native kernel" */
	{
		cl_device_exec_capabilities exec;

		clGetDeviceInfo(dev, CL_DEVICE_EXECUTION_CAPABILITIES, sizeof(exec), &exec, NULL);
		if (!(exec & CL_EXEC_NATIVE_KERNEL))
			CHECK_ERR("5.8", clEnqueueNativeKernel(q, (void (CL_CALLBACK *)(void *))build_notify, out,
							       sizeof(out), 0, NULL, NULL, 0, NULL, NULL),
				  CL_INVALID_OPERATION);
	}
	clReleaseMemObject(b);
	clReleaseKernel(k);
}

static int event_cb_status[3], event_cb_calls;

static void CL_CALLBACK event_cb(cl_event e, cl_int status, void *user)
{
	if (event_cb_calls < 3)
		event_cb_status[event_cb_calls] = status;
	event_cb_calls++;
}

static void event_checks(cl_context ctx, cl_command_queue hq, cl_device_id dev, cl_program prog)
{
	cl_event user, ev, marker, evs[2];
	cl_command_queue pq;
	cl_command_type type;
	cl_context c;
	cl_command_queue eq;
	cl_ulong t[4];
	cl_int err, st;
	cl_uint u;
	cl_kernel k;
	cl_mem b;
	size_t g = 16;
	int data[16], i;

	/* a queue without profiling (the harness's has it) */
	cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &err);

	CHECK("5.1", q && err == CL_SUCCESS, "in-order queue: %d", err);
	if (!q)
		return;
	/* 5.9 user events */
	user = clCreateUserEvent(ctx, &err);
	CHECK("5.9", user && err == CL_SUCCESS, "clCreateUserEvent: %d", err);
	if (!user)
		return;
	clGetEventInfo(user, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(st), &st, NULL);
	CHECK("5.9", st == CL_SUBMITTED, "a new user event's status is %d, want CL_SUBMITTED", st);
	clGetEventInfo(user, CL_EVENT_COMMAND_TYPE, sizeof(type), &type, NULL);
	clGetEventInfo(user, CL_EVENT_COMMAND_QUEUE, sizeof(eq), &eq, NULL);
	clGetEventInfo(user, CL_EVENT_CONTEXT, sizeof(c), &c, NULL);
	CHECK("5.9", type == CL_COMMAND_USER && eq == NULL && c == ctx, "user event info");
	CHECK_ERR("5.9", clSetUserEventStatus(user, CL_RUNNING), CL_INVALID_VALUE);
	event_cb_calls = 0;
	CHECK_ERR("5.9", clSetEventCallback(user, CL_COMPLETE, event_cb, NULL), CL_SUCCESS);
	CHECK_ERR("5.9", clSetEventCallback(user, CL_RUNNING, event_cb, NULL), CL_INVALID_VALUE);

	b = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(data),
			   memset(data, 0, sizeof(data)), &err);
	k = clCreateKernel(prog, "inc", &err);
	if (!b || !k)
		return;
	clSetKernelArg(k, 0, sizeof(b), &b);
	/* a kernel waiting on the user event doesn't run until it completes */
	CHECK_ERR("5.8", clEnqueueNDRangeKernel(q, k, 1, NULL, &g, NULL, 1, &user, &ev), CL_SUCCESS);
	clFlush(q);
	clGetEventInfo(ev, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(st), &st, NULL);
	CHECK("5.9", st != CL_COMPLETE, "a kernel waiting on a user event completed before it");
	CHECK_ERR("5.9", clSetUserEventStatus(user, CL_COMPLETE), CL_SUCCESS);
	CHECK_ERR("5.9", clSetUserEventStatus(user, CL_COMPLETE), CL_INVALID_OPERATION);
	CHECK_ERR("5.9", clWaitForEvents(1, &ev), CL_SUCCESS);
	CHECK("5.9", event_cb_calls == 1 && event_cb_status[0] == CL_COMPLETE,
	      "CL_COMPLETE callback ran %d times (status %d)", event_cb_calls, event_cb_status[0]);
	clGetEventInfo(ev, CL_EVENT_COMMAND_TYPE, sizeof(type), &type, NULL);
	clGetEventInfo(ev, CL_EVENT_COMMAND_QUEUE, sizeof(eq), &eq, NULL);
	clGetEventInfo(ev, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(st), &st, NULL);
	CHECK("5.9", type == CL_COMMAND_NDRANGE_KERNEL && eq == q && st == CL_COMPLETE, "kernel event info");
	/* a callback set on a complete event runs (5.9: "registered ... called") */
	event_cb_calls = 0;
	clSetEventCallback(ev, CL_COMPLETE, event_cb, NULL);
	clFinish(q);
	CHECK("5.9", event_cb_calls == 1, "callback on a complete event ran %d times", event_cb_calls);
	clRetainEvent(ev);
	clGetEventInfo(ev, CL_EVENT_REFERENCE_COUNT, sizeof(u), &u, NULL);
	CHECK("5.9", u == 2, "CL_EVENT_REFERENCE_COUNT %u after a retain", u);
	clReleaseEvent(ev);

	/* 5.12: profiling on a queue without it, and of user events */
	CHECK_ERR("5.12", clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t[0]), t, NULL),
		  CL_PROFILING_INFO_NOT_AVAILABLE);
	CHECK_ERR("5.12", clGetEventProfilingInfo(user, CL_PROFILING_COMMAND_START, sizeof(t[0]), t, NULL),
		  CL_PROFILING_INFO_NOT_AVAILABLE);
	clReleaseEvent(ev);

	/* a user event set to an error: commands waiting on it don't run (5.9) */
	clReleaseEvent(user);
	user = clCreateUserEvent(ctx, &err);
	clEnqueueNDRangeKernel(q, k, 1, NULL, &g, NULL, 1, &user, &ev);
	CHECK_ERR("5.9", clSetUserEventStatus(user, -1), CL_SUCCESS);
	clFinish(q);
	clGetEventInfo(ev, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(st), &st, NULL);
	CHECK("5.9", st < 0, "a kernel waiting on a failed user event has status %d", st);
	clReleaseEvent(ev);
	clReleaseEvent(user);

	/* 5.10 markers, barriers and waits */
	CHECK_ERR("5.10", clEnqueueNDRangeKernel(q, k, 1, NULL, &g, NULL, 0, NULL, &evs[0]), CL_SUCCESS);
	CHECK_ERR("5.10", clEnqueueMarker(q, &marker), CL_SUCCESS);
	CHECK_ERR("5.10", clEnqueueBarrier(q), CL_SUCCESS);
	CHECK_ERR("5.10", clEnqueueWaitForEvents(q, 1, &marker), CL_SUCCESS);
	CHECK_ERR("5.10", clEnqueueWaitForEvents(q, 0, NULL), CL_INVALID_VALUE);
	CHECK_ERR("5.10", clEnqueueMarker(q, NULL), CL_INVALID_VALUE);
	CHECK_ERR("5.10", clWaitForEvents(1, &marker), CL_SUCCESS);
	clGetEventInfo(evs[0], CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(st), &st, NULL);
	CHECK("5.10", st == CL_COMPLETE, "the marker completed before the kernel before it (%d)", st);
	clGetEventInfo(marker, CL_EVENT_COMMAND_TYPE, sizeof(type), &type, NULL);
	CHECK("5.10", type == CL_COMMAND_MARKER, "marker command type 0x%x", type);
	CHECK_ERR("5.9", clWaitForEvents(0, NULL), CL_INVALID_VALUE);
	clReleaseEvent(marker);
	clReleaseEvent(evs[0]);
	clFinish(q);
	clEnqueueReadBuffer(q, b, CL_TRUE, 0, sizeof(data), data, 0, NULL, NULL);
	for (i = 0; i < 16 && data[i] == 2; i++)
		;
	CHECK("5.8", i == 16, "after two increments a[%d] is %d", i, i < 16 ? data[i] : 0);

	/* 5.12 profiling */
	pq = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err);
	if (pq) {
		static const cl_profiling_info what[4] = { CL_PROFILING_COMMAND_QUEUED,
			CL_PROFILING_COMMAND_SUBMIT, CL_PROFILING_COMMAND_START, CL_PROFILING_COMMAND_END };
		size_t size;

		clEnqueueNDRangeKernel(pq, k, 1, NULL, &g, NULL, 0, NULL, &evs[0]);
		clEnqueueReadBuffer(pq, b, CL_FALSE, 0, sizeof(data), data, 0, NULL, &evs[1]);
		clFinish(pq);
		for (i = 0; i < 2; i++) {
			int j, ok = 1;

			for (j = 0; j < 4; j++) {
				err = clGetEventProfilingInfo(evs[i], what[j], sizeof(t[j]), &t[j], &size);
				ok &= err == CL_SUCCESS && size == sizeof(cl_ulong);
			}
			CHECK("5.12", ok, "profiling info of command %d", i);
			CHECK("5.12", t[0] <= t[1] && t[1] <= t[2] && t[2] <= t[3],
			      "command %d: queued %llu, submit %llu, start %llu, end %llu", i,
			      (unsigned long long)t[0], (unsigned long long)t[1], (unsigned long long)t[2],
			      (unsigned long long)t[3]);
			CHECK_ERR("5.12", clGetEventProfilingInfo(evs[i], 0x7fff, sizeof(t[0]), t, NULL),
				  CL_INVALID_VALUE);
		}
		CHECK("5.12", data[0] == 3, "profiled kernel's result %d", data[0]);
		clReleaseEvent(evs[0]);
		clReleaseEvent(evs[1]);
		clReleaseCommandQueue(pq);
	} else
		CHECK("5.12", 0, "profiling queue: %d", err);
	clReleaseKernel(k);
	clReleaseMemObject(b);
	clReleaseCommandQueue(q);
}

int ep_api_run(cl_context ctx, cl_command_queue q, cl_device_id dev, cl_program prog,
	       char *out, size_t out_size, ep_log_fn log)
{
	cl_platform_id plat;

	report = out;
	report_size = out_size;
	report_len = 0;
	if (out_size)
		out[0] = 0;
	checks = failures = 0;
	log_fn = log;
	clGetDeviceInfo(dev, CL_DEVICE_PLATFORM, sizeof(plat), &plat, NULL);
	platform_checks(plat);
	device_checks(plat, dev);
	context_checks(plat, dev);
	queue_checks(ctx, dev);
	buffer_checks(ctx, q, dev);
	image_checks(ctx, q, dev);
	program_checks(ctx, dev, prog);
	kernel_checks(ctx, q, dev, prog);
	event_checks(ctx, q, dev, prog);
	clFinish(q);
	log("ep_api: %d checks, %d failed\n", checks, failures);
	return failures;
}
