/* Events and running commands. */
#include "cl_internal.h"

#include <psp2/kernel/threadmgr.h>
#include <stdlib.h>
#include <string.h>

extern cl_command_queue cl_queues;

void cl_event_retain(cl_event e)
{
	e->refs++;
}

static void drop_command(cl_event e)
{
	command *c = &e->cmd;
	cl_uint i;

	for (i = 0; i < 2; i++)
		if (c->mem[i])
			cl_mem_release(c->mem[i]);
	free(c->map);
	if (c->run)
		cl_free_launch(c->run);
	memset(c, 0, sizeof(*c));
	for (i = 0; i < e->wait_count; i++)
		cl_event_release(e->waits[i]);
	free(e->waits);
	e->waits = NULL;
	e->wait_count = 0;
}

void cl_event_release(cl_event e)
{
	cl_context ctx = e->ctx;
	cl_command_queue q = e->queue;

	if (--e->refs)
		return;
	drop_command(e);
	while (e->callbacks) {
		callback *cb = e->callbacks;

		e->callbacks = cb->next;
		free(cb);
	}
	e->magic = 0;
	free(e);
	if (q)
		cl_queue_release(q);
	cl_context_release(ctx);
}

/* status: CL_COMPLETE or a negative error; then the callbacks run. */
void cl_set_status(cl_event e, cl_int status)
{
	e->status = status;
	if (status > CL_COMPLETE)
		return;
	drop_command(e);
	cl_event_retain(e);
	while (e->callbacks) {
		callback *cb = e->callbacks;

		e->callbacks = cb->next;
		((void (CL_CALLBACK *)(cl_event, cl_int, void *))cb->fn)(e, status, cb->user_data);
		free(cb);
	}
	cl_event_release(e);
}

static int ready(cl_event e)
{
	cl_uint i;

	for (i = 0; i < e->wait_count; i++)
		if (e->waits[i]->status > CL_COMPLETE)
			return 0;
	return 1;
}

static void execute(cl_event e)
{
	cl_int status = CL_COMPLETE;
	cl_uint i;

	for (i = 0; i < e->wait_count; i++)
		if (e->waits[i]->status < 0)
			status = CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST;
	e->times[1] = e->times[2] = cl_now();
	if (status == CL_COMPLETE) {
		e->status = CL_RUNNING;
		switch (e->type) {
		case CL_COMMAND_NDRANGE_KERNEL:
		case CL_COMMAND_TASK:
			status = cl_run_launch(e->cmd.run);
			break;
		case CL_COMMAND_MARKER:
			break;
		default:
			status = cl_run_mem_command(e);
		}
	}
	e->times[3] = cl_now();
	cl_set_status(e, status);
}

static int progressing;

/* Run every command that can run. */
void cl_progress(void)
{
	if (progressing)
		return;		/* from a callback: the loop below looks again */
	progressing = 1;
	for (;;) {
		cl_command_queue q;
		cl_event e = NULL;

		for (q = cl_queues; q && !e; q = q->next)
			if (q->head && ready(q->head))
				e = q->head;
		if (!e)
			break;
		q = e->queue;
		q->head = e->next;
		if (!q->head)
			q->tail = NULL;
		e->next = NULL;
		execute(e);
		cl_event_release(e);	/* the queue's */
	}
	progressing = 0;
}

cl_int cl_wait(cl_event e)
{
	cl_int status;

	cl_event_retain(e);
	for (;;) {
		cl_progress();
		if (e->status <= CL_COMPLETE)
			break;
		/* another thread may set the user event this waits for */
		cl_unlock();
		sceKernelDelayThread(100);
		cl_lock();
	}
	status = e->status;
	cl_event_release(e);
	return status;
}

cl_int cl_check_wait_list(cl_context ctx, cl_uint n, const cl_event *list)
{
	cl_uint i;

	if ((n && !list) || (!n && list))
		return CL_INVALID_EVENT_WAIT_LIST;
	for (i = 0; i < n; i++) {
		if (!VALID(list[i], MAGIC_EVENT))
			return CL_INVALID_EVENT_WAIT_LIST;
		if (list[i]->ctx != ctx)
			return CL_INVALID_CONTEXT;
	}
	return CL_SUCCESS;
}

static cl_event new_event(cl_context ctx, cl_command_queue q, cl_command_type type)
{
	cl_event e = calloc(1, sizeof(*e));

	if (!e)
		return NULL;
	e->magic = MAGIC_EVENT;
	e->refs = 1;
	e->ctx = ctx;
	cl_context_retain(ctx);
	e->queue = q;
	if (q)
		cl_queue_retain(q);
	e->type = type;
	e->status = q ? CL_QUEUED : CL_SUBMITTED;
	e->times[0] = cl_now();
	return e;
}

cl_int cl_enqueue(cl_command_queue q, cl_command_type type, const command *cmd, cl_uint n,
		  const cl_event *list, cl_event *out, int blocking)
{
	cl_event e = new_event(q->ctx, q, type);
	cl_int status;
	cl_uint i;

	if (out)
		*out = NULL;
	if (e && n && !(e->waits = malloc(n * sizeof(cl_event)))) {
		cl_event_release(e);
		e = NULL;
	}
	if (!e) {
		/* release what the command holds */
		struct _cl_event tmp;

		memset(&tmp, 0, sizeof(tmp));
		tmp.cmd = *cmd;
		drop_command(&tmp);
		return CL_OUT_OF_HOST_MEMORY;
	}
	e->cmd = *cmd;
	for (i = 0; i < n; i++) {
		e->waits[i] = list[i];
		cl_event_retain(list[i]);
	}
	e->wait_count = n;
	if (q->tail)
		q->tail->next = e;
	else
		q->head = e;
	q->tail = e;

	cl_event_retain(e);
	cl_progress();
	status = blocking ? cl_wait(e) : e->status;
	if (status < 0 && status != CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST) {
		/* it failed on its own, here: report it as the enqueue's error */
		cl_event_release(e);
		return status;
	}
	if (out)
		*out = e;
	else
		cl_event_release(e);
	return blocking && status < 0 ? CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST : CL_SUCCESS;
}

CL_API_ENTRY cl_event CL_API_CALL
clCreateUserEvent(cl_context context, cl_int *errcode_ret)
{
	cl_event e = NULL;
	cl_int err = CL_SUCCESS;

	cl_lock();
	if (!VALID(context, MAGIC_CONTEXT))
		err = CL_INVALID_CONTEXT;
	else if (!(e = new_event(context, NULL, CL_COMMAND_USER)))
		err = CL_OUT_OF_HOST_MEMORY;
	cl_unlock();
	if (errcode_ret)
		*errcode_ret = err;
	return e;
}

CL_API_ENTRY cl_int CL_API_CALL
clSetUserEventStatus(cl_event event, cl_int execution_status)
{
	cl_int err = CL_SUCCESS;

	cl_lock();
	if (!VALID(event, MAGIC_EVENT) || event->queue)
		err = CL_INVALID_EVENT;
	else if (execution_status > CL_COMPLETE)
		err = CL_INVALID_VALUE;
	else if (event->status <= CL_COMPLETE)
		err = CL_INVALID_OPERATION;
	if (err == CL_SUCCESS) {
		cl_set_status(event, execution_status);
		cl_progress();
	}
	cl_unlock();
	return err;
}

CL_API_ENTRY cl_int CL_API_CALL
clRetainEvent(cl_event event)
{
	cl_int err = CL_INVALID_EVENT;

	cl_lock();
	if (VALID(event, MAGIC_EVENT)) {
		cl_event_retain(event);
		err = CL_SUCCESS;
	}
	cl_unlock();
	return err;
}

CL_API_ENTRY cl_int CL_API_CALL
clReleaseEvent(cl_event event)
{
	cl_int err = CL_INVALID_EVENT;

	cl_lock();
	if (VALID(event, MAGIC_EVENT)) {
		cl_event_release(event);
		err = CL_SUCCESS;
	}
	cl_unlock();
	return err;
}

CL_API_ENTRY cl_int CL_API_CALL
clSetEventCallback(cl_event event, cl_int type,
		   void (CL_CALLBACK *pfn_notify)(cl_event, cl_int, void *), void *user_data)
{
	callback *cb, **link;
	cl_int err = CL_SUCCESS;

	cl_lock();
	if (!VALID(event, MAGIC_EVENT)) {
		err = CL_INVALID_EVENT;
	} else if (!pfn_notify || type != CL_COMPLETE) {
		err = CL_INVALID_VALUE;
	} else if (event->status <= CL_COMPLETE) {
		pfn_notify(event, event->status, user_data);
	} else if (!(cb = malloc(sizeof(*cb)))) {
		err = CL_OUT_OF_HOST_MEMORY;
	} else {
		cb->fn = (void (CL_CALLBACK *)())pfn_notify;
		cb->user_data = user_data;
		cb->next = NULL;
		for (link = &event->callbacks; *link; link = &(*link)->next)
			;
		*link = cb;
	}
	cl_unlock();
	return err;
}

CL_API_ENTRY cl_int CL_API_CALL
clGetEventInfo(cl_event event, cl_event_info param_name, size_t size, void *value,
	       size_t *size_ret)
{
	cl_int err;

	cl_lock();
	if (!VALID(event, MAGIC_EVENT)) {
		err = CL_INVALID_EVENT;
		goto out;
	}
	cl_progress();
	switch (param_name) {
	case CL_EVENT_COMMAND_QUEUE:
		err = cl_info(size, value, size_ret, &event->queue, sizeof(cl_command_queue));
		break;
	case CL_EVENT_CONTEXT:
		err = cl_info(size, value, size_ret, &event->ctx, sizeof(cl_context));
		break;
	case CL_EVENT_COMMAND_TYPE:
		err = cl_info(size, value, size_ret, &event->type, sizeof(cl_command_type));
		break;
	case CL_EVENT_COMMAND_EXECUTION_STATUS:
		err = cl_info(size, value, size_ret, &event->status, sizeof(cl_int));
		break;
	case CL_EVENT_REFERENCE_COUNT:
		err = cl_info(size, value, size_ret, &event->refs, sizeof(cl_uint));
		break;
	default:
		err = CL_INVALID_VALUE;
	}
out:
	cl_unlock();
	return err;
}

CL_API_ENTRY cl_int CL_API_CALL
clGetEventProfilingInfo(cl_event event, cl_profiling_info param_name, size_t size, void *value,
			size_t *size_ret)
{
	cl_int err;
	int i;

	cl_lock();
	if (!VALID(event, MAGIC_EVENT)) {
		err = CL_INVALID_EVENT;
		goto out;
	}
	switch (param_name) {
	case CL_PROFILING_COMMAND_QUEUED:
		i = 0;
		break;
	case CL_PROFILING_COMMAND_SUBMIT:
		i = 1;
		break;
	case CL_PROFILING_COMMAND_START:
		i = 2;
		break;
	case CL_PROFILING_COMMAND_END:
		i = 3;
		break;
	default:
		err = CL_INVALID_VALUE;
		goto out;
	}
	if (!event->queue || !(event->queue->props & CL_QUEUE_PROFILING_ENABLE) ||
	    event->status != CL_COMPLETE)
		err = CL_PROFILING_INFO_NOT_AVAILABLE;
	else
		err = cl_info(size, value, size_ret, &event->times[i], sizeof(cl_ulong));
out:
	cl_unlock();
	return err;
}

CL_API_ENTRY cl_int CL_API_CALL
clWaitForEvents(cl_uint num_events, const cl_event *event_list)
{
	cl_int err = CL_SUCCESS;
	cl_uint i;

	cl_lock();
	if (!num_events || !event_list)
		err = CL_INVALID_VALUE;
	for (i = 0; err == CL_SUCCESS && i < num_events; i++) {
		if (!VALID(event_list[i], MAGIC_EVENT))
			err = CL_INVALID_EVENT;
		else if (event_list[i]->ctx != event_list[0]->ctx)
			err = CL_INVALID_CONTEXT;
	}
	for (i = 0; err == CL_SUCCESS && i < num_events; i++)
		if (cl_wait(event_list[i]) < 0)
			err = CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST;
	cl_unlock();
	return err;
}

CL_API_ENTRY cl_int CL_API_CALL
clFlush(cl_command_queue command_queue)
{
	cl_int err = CL_INVALID_COMMAND_QUEUE;

	cl_lock();
	if (VALID(command_queue, MAGIC_QUEUE)) {
		cl_progress();
		err = CL_SUCCESS;
	}
	cl_unlock();
	return err;
}

CL_API_ENTRY cl_int CL_API_CALL
clFinish(cl_command_queue command_queue)
{
	cl_int err = CL_INVALID_COMMAND_QUEUE;

	cl_lock();
	if (VALID(command_queue, MAGIC_QUEUE)) {
		cl_queue_retain(command_queue);
		while (command_queue->tail)
			cl_wait(command_queue->tail);
		cl_queue_release(command_queue);
		err = CL_SUCCESS;
	}
	cl_unlock();
	return err;
}

static cl_int marker(cl_command_queue q, cl_uint n, const cl_event *list, cl_event *event)
{
	command cmd;

	memset(&cmd, 0, sizeof(cmd));
	return cl_enqueue(q, CL_COMMAND_MARKER, &cmd, n, list, event, 0);
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueMarker(cl_command_queue command_queue, cl_event *event)
{
	cl_int err;

	cl_lock();
	if (!VALID(command_queue, MAGIC_QUEUE))
		err = CL_INVALID_COMMAND_QUEUE;
	else if (!event)
		err = CL_INVALID_VALUE;
	else
		err = marker(command_queue, 0, NULL, event);
	cl_unlock();
	return err;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueBarrier(cl_command_queue command_queue)
{
	cl_int err;

	cl_lock();
	if (!VALID(command_queue, MAGIC_QUEUE))
		err = CL_INVALID_COMMAND_QUEUE;
	else
		err = marker(command_queue, 0, NULL, NULL);
	cl_unlock();
	return err;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueWaitForEvents(cl_command_queue command_queue, cl_uint num_events,
		       const cl_event *event_list)
{
	cl_int err = CL_SUCCESS;
	cl_uint i;

	cl_lock();
	if (!VALID(command_queue, MAGIC_QUEUE))
		err = CL_INVALID_COMMAND_QUEUE;
	else if (!num_events || !event_list)
		err = CL_INVALID_VALUE;
	for (i = 0; err == CL_SUCCESS && i < num_events; i++) {
		if (!VALID(event_list[i], MAGIC_EVENT))
			err = CL_INVALID_EVENT;
		else if (event_list[i]->ctx != command_queue->ctx)
			err = CL_INVALID_CONTEXT;
	}
	if (err == CL_SUCCESS)
		err = marker(command_queue, num_events, event_list, NULL);
	cl_unlock();
	return err;
}
