/* The lock, object checks, the platform and the device. */
#include "cl_internal.h"

#include <psp2/kernel/processmgr.h>
#include <psp2/kernel/threadmgr.h>
#include <psp2/power.h>
#include <string.h>

struct _cl_platform_id cl_the_platform = { MAGIC_PLATFORM };
struct _cl_device_id cl_the_device = { MAGIC_DEVICE };
gxc cl_gpu;

static SceKernelLwMutexWork lock_work;
static int lock_ready;
static int gpu_state;		/* 0 not tried, 1 ready, < 0 the error */

static void __attribute__((constructor)) lock_init(void)
{
	lock_ready = sceKernelCreateLwMutex(&lock_work, "opencl", SCE_KERNEL_MUTEX_ATTR_RECURSIVE,
					    0, NULL) >= 0;
}

void cl_lock(void)
{
	if (lock_ready)
		sceKernelLockLwMutex(&lock_work, 1, NULL);
}

void cl_unlock(void)
{
	if (lock_ready)
		sceKernelUnlockLwMutex(&lock_work, 1);
}

cl_int cl_gpu_init(void)
{
	const char *stage;
	int err;

	if (gpu_state == 0) {
		err = gxc_init(&cl_gpu, &stage);
		if (err < 0) {
			gxc_term(&cl_gpu);
			gpu_state = err;
		} else {
			gpu_state = 1;
		}
	}
	return gpu_state == 1 ? CL_SUCCESS : CL_OUT_OF_RESOURCES;
}

int cl_valid(const void *obj, uint32_t magic)
{
	return obj && *(const uint32_t *)obj == magic;
}

cl_ulong cl_now(void)
{
	return (cl_ulong)sceKernelGetProcessTimeWide() * 1000;
}

cl_int cl_info(size_t size, void *value, size_t *size_ret, const void *src, size_t n)
{
	if (value) {
		if (size < n)
			return CL_INVALID_VALUE;
		memcpy(value, src, n);
	}
	if (size_ret)
		*size_ret = n;
	return CL_SUCCESS;
}

cl_int cl_info_string(size_t size, void *value, size_t *size_ret, const char *s)
{
	return cl_info(size, value, size_ret, s, strlen(s) + 1);
}

#define PROFILE		"EMBEDDED_PROFILE"
#define VERSION		"OpenCL 1.1 vita-sgx"
#define EXTENSIONS	"cl_khr_byte_addressable_store " \
			"cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics " \
			"cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics"

CL_API_ENTRY cl_int CL_API_CALL
clGetPlatformIDs(cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms)
{
	if ((num_entries == 0 && platforms) || (!platforms && !num_platforms))
		return CL_INVALID_VALUE;
	if (platforms)
		platforms[0] = &cl_the_platform;
	if (num_platforms)
		*num_platforms = 1;
	return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL
clGetPlatformInfo(cl_platform_id platform, cl_platform_info param_name, size_t size,
		  void *value, size_t *size_ret)
{
	const char *s;

	if (platform && platform != &cl_the_platform)
		return CL_INVALID_PLATFORM;
	switch (param_name) {
	case CL_PLATFORM_PROFILE:
		s = PROFILE;
		break;
	case CL_PLATFORM_VERSION:
		s = VERSION;
		break;
	case CL_PLATFORM_NAME:
		s = "PS Vita SGX";
		break;
	case CL_PLATFORM_VENDOR:
		s = "opencl-1.1-psp2";
		break;
	case CL_PLATFORM_EXTENSIONS:
		s = "";
		break;
	default:
		return CL_INVALID_VALUE;
	}
	return cl_info_string(size, value, size_ret, s);
}

CL_API_ENTRY cl_int CL_API_CALL
clGetDeviceIDs(cl_platform_id platform, cl_device_type type, cl_uint num_entries,
	       cl_device_id *devices, cl_uint *num_devices)
{
	const cl_device_type known = CL_DEVICE_TYPE_DEFAULT | CL_DEVICE_TYPE_CPU |
				     CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR;

	if (platform && platform != &cl_the_platform)
		return CL_INVALID_PLATFORM;
	if (type != CL_DEVICE_TYPE_ALL && (type & ~known || !type))
		return CL_INVALID_DEVICE_TYPE;
	if ((num_entries == 0 && devices) || (!devices && !num_devices))
		return CL_INVALID_VALUE;
	if (!(type & (CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_DEFAULT))) {
		if (num_devices)
			*num_devices = 0;
		return CL_DEVICE_NOT_FOUND;
	}
	if (devices)
		devices[0] = &cl_the_device;
	if (num_devices)
		*num_devices = 1;
	return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL
clGetDeviceInfo(cl_device_id device, cl_device_info param_name, size_t size, void *value,
		size_t *size_ret)
{
	union {
		cl_uint u;
		cl_bool b;
		cl_ulong ul;
		size_t sz;
		size_t sizes[3];
		cl_device_type type;
		cl_device_fp_config fp;
		cl_device_mem_cache_type cache;
		cl_device_local_mem_type local;
		cl_device_exec_capabilities exec;
		cl_command_queue_properties queue;
		cl_platform_id platform;
	} v;
	size_t n;
	int mhz;

#define U(x)	do { v.u = (x); n = sizeof(cl_uint); } while (0)
#define B(x)	do { v.b = (x); n = sizeof(cl_bool); } while (0)
#define UL(x)	do { v.ul = (x); n = sizeof(cl_ulong); } while (0)
#define SZ(x)	do { v.sz = (x); n = sizeof(size_t); } while (0)
#define STR(x)	return cl_info_string(size, value, size_ret, (x))

	if (device != &cl_the_device)
		return CL_INVALID_DEVICE;
	switch (param_name) {
	case CL_DEVICE_TYPE:
		v.type = CL_DEVICE_TYPE_GPU;
		n = sizeof(v.type);
		break;
	case CL_DEVICE_VENDOR_ID:
		U(0x1010);		/* Imagination Technologies */
		break;
	case CL_DEVICE_MAX_COMPUTE_UNITS:
		U(4);
		break;
	case CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:
		U(3);
		break;
	case CL_DEVICE_MAX_WORK_ITEM_SIZES:
		v.sizes[0] = v.sizes[1] = v.sizes[2] = 1;
		n = sizeof(v.sizes);
		break;
	case CL_DEVICE_MAX_WORK_GROUP_SIZE:
		SZ(1);
		break;
	case CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR:
	case CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT:
	case CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT:
	case CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT:
	case CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR:
	case CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT:
	case CL_DEVICE_NATIVE_VECTOR_WIDTH_INT:
	case CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT:
		U(1);
		break;
	case CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG:
	case CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE:
	case CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF:
	case CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG:
	case CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE:
	case CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF:
		U(0);
		break;
	case CL_DEVICE_MAX_CLOCK_FREQUENCY:
		/* 0 while the GPU is unclocked (before sceGxmInitialize); applications
		 * get 111 MHz by default */
		mhz = scePowerGetGpuClockFrequency();
		U(mhz > 0 ? mhz : 111);
		break;
	case CL_DEVICE_ADDRESS_BITS:
		U(32);
		break;
	case CL_DEVICE_MAX_MEM_ALLOC_SIZE:
		UL(DEV_MAX_ALLOC);
		break;
	case CL_DEVICE_IMAGE_SUPPORT:
	case CL_DEVICE_ERROR_CORRECTION_SUPPORT:
	case CL_DEVICE_COMPILER_AVAILABLE:
		B(CL_FALSE);
		break;
	case CL_DEVICE_MAX_READ_IMAGE_ARGS:
	case CL_DEVICE_MAX_WRITE_IMAGE_ARGS:
	case CL_DEVICE_MAX_SAMPLERS:
	case CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE:
		U(0);
		break;
	case CL_DEVICE_IMAGE2D_MAX_WIDTH:
	case CL_DEVICE_IMAGE2D_MAX_HEIGHT:
	case CL_DEVICE_IMAGE3D_MAX_WIDTH:
	case CL_DEVICE_IMAGE3D_MAX_HEIGHT:
	case CL_DEVICE_IMAGE3D_MAX_DEPTH:
		SZ(0);
		break;
	case CL_DEVICE_MAX_PARAMETER_SIZE:
		SZ(DEV_MAX_PARAMETERS);
		break;
	case CL_DEVICE_MEM_BASE_ADDR_ALIGN:
		U(DEV_MEM_ALIGN * 8);
		break;
	case CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE:
		U(DEV_MEM_ALIGN);
		break;
	case CL_DEVICE_SINGLE_FP_CONFIG:
		/* USSE float add, subtract and multiply round toward zero (tests/device fp_ops) */
		v.fp = CL_FP_ROUND_TO_ZERO | CL_FP_INF_NAN;
		n = sizeof(v.fp);
		break;
	case CL_DEVICE_GLOBAL_MEM_CACHE_TYPE:
		v.cache = CL_NONE;
		n = sizeof(v.cache);
		break;
	case CL_DEVICE_GLOBAL_MEM_CACHE_SIZE:
		UL(0);
		break;
	case CL_DEVICE_GLOBAL_MEM_SIZE:
		UL(DEV_GLOBAL_MEM);
		break;
	case CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE:
		UL(DEV_MAX_CONSTANT);
		break;
	case CL_DEVICE_MAX_CONSTANT_ARGS:
		U(8);
		break;
	case CL_DEVICE_LOCAL_MEM_TYPE:
		v.local = CL_GLOBAL;
		n = sizeof(v.local);
		break;
	case CL_DEVICE_LOCAL_MEM_SIZE:
		UL(DEV_LOCAL_MEM);
		break;
	case CL_DEVICE_HOST_UNIFIED_MEMORY:
	case CL_DEVICE_ENDIAN_LITTLE:
	case CL_DEVICE_AVAILABLE:
		B(CL_TRUE);
		break;
	case CL_DEVICE_PROFILING_TIMER_RESOLUTION:
		SZ(1000);
		break;
	case CL_DEVICE_EXECUTION_CAPABILITIES:
		v.exec = CL_EXEC_KERNEL;
		n = sizeof(v.exec);
		break;
	case CL_DEVICE_QUEUE_PROPERTIES:
		v.queue = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE;
		n = sizeof(v.queue);
		break;
	case CL_DEVICE_PLATFORM:
		v.platform = &cl_the_platform;
		n = sizeof(v.platform);
		break;
	case CL_DEVICE_NAME:
		STR("PowerVR SGX543MP4+");
	case CL_DEVICE_VENDOR:
		STR("Imagination Technologies");
	case CL_DRIVER_VERSION:
		STR("0.1");
	case CL_DEVICE_PROFILE:
		STR(PROFILE);
	case CL_DEVICE_VERSION:
		STR(VERSION);
	case CL_DEVICE_OPENCL_C_VERSION:
		STR("OpenCL C 1.1 ");
	case CL_DEVICE_EXTENSIONS:
		STR(EXTENSIONS);
	default:
		return CL_INVALID_VALUE;
	}
	return cl_info(size, value, size_ret, &v, n);
#undef U
#undef B
#undef UL
#undef SZ
#undef STR
}

CL_API_ENTRY cl_int CL_API_CALL
clUnloadCompiler(void)
{
	return CL_SUCCESS;
}

CL_API_ENTRY void * CL_API_CALL
clGetExtensionFunctionAddress(const char *func_name)
{
	(void)func_name;
	return NULL;
}
