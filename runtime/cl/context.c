/* Contexts and command queues. */
#include "cl_internal.h"

#include <stdlib.h>
#include <string.h>

cl_command_queue cl_queues;	/* all queues, for cl_progress */

void cl_context_retain(cl_context ctx)
{
	ctx->refs++;
}

void cl_context_release(cl_context ctx)
{
	if (--ctx->refs)
		return;
	ctx->magic = 0;
	free(ctx);
}

static cl_context make_context(const cl_context_properties *props,
			       void (CL_CALLBACK *notify)(const char *, const void *, size_t, void *),
			       void *user_data, cl_int *err)
{
	cl_context ctx;
	size_t n = 0;
	int have_platform = 0;

	if (!notify && user_data) {
		*err = CL_INVALID_VALUE;
		return NULL;
	}
	for (; props && props[n]; n += 2) {
		if (props[n] != CL_CONTEXT_PLATFORM || have_platform) {
			*err = CL_INVALID_PROPERTY;
			return NULL;
		}
		if ((cl_platform_id)props[n + 1] != &cl_the_platform) {
			*err = CL_INVALID_PLATFORM;
			return NULL;
		}
		have_platform = 1;
	}
	*err = cl_gpu_init();
	if (*err != CL_SUCCESS)
		return NULL;
	ctx = calloc(1, sizeof(*ctx));
	if (!ctx) {
		*err = CL_OUT_OF_HOST_MEMORY;
		return NULL;
	}
	ctx->magic = MAGIC_CONTEXT;
	ctx->refs = 1;
	if (props) {
		memcpy(ctx->props, props, (n + 1) * sizeof(*props));
		ctx->props_size = (n + 1) * sizeof(*props);
	}
	ctx->notify = notify;
	ctx->notify_data = user_data;
	return ctx;
}

CL_API_ENTRY cl_context CL_API_CALL
clCreateContext(const cl_context_properties *properties, cl_uint num_devices,
		const cl_device_id *devices,
		void (CL_CALLBACK *pfn_notify)(const char *, const void *, size_t, void *),
		void *user_data, cl_int *errcode_ret)
{
	cl_context ctx = NULL;
	cl_int err = CL_SUCCESS;
	cl_uint i;

	cl_lock();
	if (!num_devices || !devices)
		err = CL_INVALID_VALUE;
	for (i = 0; err == CL_SUCCESS && i < num_devices; i++)
		if (devices[i] != &cl_the_device)
			err = CL_INVALID_DEVICE;
	if (err == CL_SUCCESS)
		ctx = make_context(properties, pfn_notify, user_data, &err);
	cl_unlock();
	if (errcode_ret)
		*errcode_ret = err;
	return ctx;
}

CL_API_ENTRY cl_context CL_API_CALL
clCreateContextFromType(const cl_context_properties *properties, cl_device_type device_type,
			void (CL_CALLBACK *pfn_notify)(const char *, const void *, size_t, void *),
			void *user_data, cl_int *errcode_ret)
{
	cl_device_id dev;
	cl_context ctx = NULL;
	cl_int err;

	cl_lock();
	err = clGetDeviceIDs(NULL, device_type, 1, &dev, NULL);
	if (err == CL_SUCCESS)
		ctx = make_context(properties, pfn_notify, user_data, &err);
	cl_unlock();
	if (errcode_ret)
		*errcode_ret = err;
	return ctx;
}

CL_API_ENTRY cl_int CL_API_CALL
clRetainContext(cl_context context)
{
	cl_int err = CL_INVALID_CONTEXT;

	cl_lock();
	if (VALID(context, MAGIC_CONTEXT)) {
		cl_context_retain(context);
		err = CL_SUCCESS;
	}
	cl_unlock();
	return err;
}

CL_API_ENTRY cl_int CL_API_CALL
clReleaseContext(cl_context context)
{
	cl_int err = CL_INVALID_CONTEXT;

	cl_lock();
	if (VALID(context, MAGIC_CONTEXT)) {
		cl_context_release(context);
		err = CL_SUCCESS;
	}
	cl_unlock();
	return err;
}

CL_API_ENTRY cl_int CL_API_CALL
clGetContextInfo(cl_context context, cl_context_info param_name, size_t size, void *value,
		 size_t *size_ret)
{
	cl_device_id dev = &cl_the_device;
	cl_uint n;
	cl_int err;

	cl_lock();
	if (!VALID(context, MAGIC_CONTEXT)) {
		err = CL_INVALID_CONTEXT;
		goto out;
	}
	switch (param_name) {
	case CL_CONTEXT_REFERENCE_COUNT:
		n = context->refs;
		err = cl_info(size, value, size_ret, &n, sizeof(n));
		break;
	case CL_CONTEXT_NUM_DEVICES:
		n = 1;
		err = cl_info(size, value, size_ret, &n, sizeof(n));
		break;
	case CL_CONTEXT_DEVICES:
		err = cl_info(size, value, size_ret, &dev, sizeof(dev));
		break;
	case CL_CONTEXT_PROPERTIES:
		err = cl_info(size, value, size_ret, context->props, context->props_size);
		break;
	default:
		err = CL_INVALID_VALUE;
	}
out:
	cl_unlock();
	return err;
}

void cl_queue_retain(cl_command_queue q)
{
	q->refs++;
}

void cl_queue_release(cl_command_queue q)
{
	cl_command_queue *link;

	if (--q->refs)
		return;
	for (link = &cl_queues; *link != q; link = &(*link)->next)
		;
	*link = q->next;
	q->magic = 0;
	cl_context_release(q->ctx);
	free(q);
}

CL_API_ENTRY cl_command_queue CL_API_CALL
clCreateCommandQueue(cl_context context, cl_device_id device,
		     cl_command_queue_properties properties, cl_int *errcode_ret)
{
	cl_command_queue q = NULL;
	cl_int err = CL_SUCCESS;

	cl_lock();
	if (!VALID(context, MAGIC_CONTEXT))
		err = CL_INVALID_CONTEXT;
	else if (device != &cl_the_device)
		err = CL_INVALID_DEVICE;
	else if (properties & ~(cl_command_queue_properties)(CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE |
							      CL_QUEUE_PROFILING_ENABLE))
		err = CL_INVALID_VALUE;
	else if (!(q = calloc(1, sizeof(*q))))
		err = CL_OUT_OF_HOST_MEMORY;
	if (q) {
		q->magic = MAGIC_QUEUE;
		q->refs = 1;
		q->ctx = context;
		q->props = properties;
		cl_context_retain(context);
		q->next = cl_queues;
		cl_queues = q;
	}
	cl_unlock();
	if (errcode_ret)
		*errcode_ret = err;
	return q;
}

CL_API_ENTRY cl_int CL_API_CALL
clRetainCommandQueue(cl_command_queue command_queue)
{
	cl_int err = CL_INVALID_COMMAND_QUEUE;

	cl_lock();
	if (VALID(command_queue, MAGIC_QUEUE)) {
		cl_queue_retain(command_queue);
		err = CL_SUCCESS;
	}
	cl_unlock();
	return err;
}

CL_API_ENTRY cl_int CL_API_CALL
clReleaseCommandQueue(cl_command_queue command_queue)
{
	cl_int err = CL_INVALID_COMMAND_QUEUE;

	cl_lock();
	if (VALID(command_queue, MAGIC_QUEUE)) {
		cl_progress();		/* the implicit flush */
		cl_queue_release(command_queue);
		err = CL_SUCCESS;
	}
	cl_unlock();
	return err;
}

CL_API_ENTRY cl_int CL_API_CALL
clGetCommandQueueInfo(cl_command_queue command_queue, cl_command_queue_info param_name,
		      size_t size, void *value, size_t *size_ret)
{
	cl_device_id dev = &cl_the_device;
	cl_int err;

	cl_lock();
	if (!VALID(command_queue, MAGIC_QUEUE)) {
		err = CL_INVALID_COMMAND_QUEUE;
		goto out;
	}
	switch (param_name) {
	case CL_QUEUE_CONTEXT:
		err = cl_info(size, value, size_ret, &command_queue->ctx, sizeof(cl_context));
		break;
	case CL_QUEUE_DEVICE:
		err = cl_info(size, value, size_ret, &dev, sizeof(dev));
		break;
	case CL_QUEUE_REFERENCE_COUNT:
		err = cl_info(size, value, size_ret, &command_queue->refs, sizeof(cl_uint));
		break;
	case CL_QUEUE_PROPERTIES:
		err = cl_info(size, value, size_ret, &command_queue->props,
			      sizeof(cl_command_queue_properties));
		break;
	default:
		err = CL_INVALID_VALUE;
	}
out:
	cl_unlock();
	return err;
}

CL_API_ENTRY cl_int CL_API_CALL
clSetCommandQueueProperty(cl_command_queue command_queue, cl_command_queue_properties properties,
			  cl_bool enable, cl_command_queue_properties *old_properties)
{
	cl_int err = CL_SUCCESS;

	cl_lock();
	if (!VALID(command_queue, MAGIC_QUEUE)) {
		err = CL_INVALID_COMMAND_QUEUE;
	} else if (properties & ~(cl_command_queue_properties)(CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE |
								 CL_QUEUE_PROFILING_ENABLE)) {
		err = CL_INVALID_VALUE;
	} else {
		if (old_properties)
			*old_properties = command_queue->props;
		if (enable)
			command_queue->props |= properties;
		else
			command_queue->props &= ~properties;
	}
	cl_unlock();
	return err;
}
