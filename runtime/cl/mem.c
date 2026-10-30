/*
 * Buffers and the commands that move their contents.  A buffer's contents
 * live in GPU memory, which the CPU maps at the same address, so a mapping
 * is a pointer into it; a CL_MEM_USE_HOST_PTR buffer keeps a copy there and
 * maps by copying to and from the host memory.  There is no image support.
 */
#include "cl_internal.h"

#include <stdlib.h>
#include <string.h>

#define ACCESS_FLAGS	(CL_MEM_READ_WRITE | CL_MEM_WRITE_ONLY | CL_MEM_READ_ONLY)
#define HOST_FLAGS	(CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR)

void cl_mem_retain(cl_mem m)
{
	m->refs++;
}

void cl_mem_release(cl_mem m)
{
	if (--m->refs)
		return;
	while (m->destructors) {
		callback *cb = m->destructors;

		m->destructors = cb->next;
		((void (CL_CALLBACK *)(cl_mem, void *))cb->fn)(m, cb->user_data);
		free(cb);
	}
	while (m->maps) {
		mapping *map = m->maps;

		m->maps = map->next;
		free(map);
	}
	if (m->block)
		dev_free(m->block);
	if (m->parent)
		cl_mem_release(m->parent);
	cl_context_release(m->ctx);
	m->magic = 0;
	free(m);
}

/* The buffer m is part of, and where m starts in it. */
cl_mem cl_mem_root(cl_mem m, size_t *offset)
{
	*offset = m->origin;
	return m->parent ? m->parent : m;
}

static int bad_access_flags(cl_mem_flags flags)
{
	cl_mem_flags a = flags & ACCESS_FLAGS;

	return a & (a - 1);	/* more than one */
}

CL_API_ENTRY cl_mem CL_API_CALL
clCreateBuffer(cl_context context, cl_mem_flags flags, size_t size, void *host_ptr,
	       cl_int *errcode_ret)
{
	cl_mem m = NULL;
	cl_int err = CL_SUCCESS;

	cl_lock();
	if (!flags)
		flags = CL_MEM_READ_WRITE;
	if (!VALID(context, MAGIC_CONTEXT))
		err = CL_INVALID_CONTEXT;
	else if (flags & ~(cl_mem_flags)(ACCESS_FLAGS | HOST_FLAGS) || bad_access_flags(flags) ||
		 (flags & CL_MEM_USE_HOST_PTR && flags & (CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR)))
		err = CL_INVALID_VALUE;
	else if (size == 0 || size > DEV_MAX_ALLOC)
		err = CL_INVALID_BUFFER_SIZE;
	else if (!host_ptr != !(flags & (CL_MEM_USE_HOST_PTR | CL_MEM_COPY_HOST_PTR)))
		err = CL_INVALID_HOST_PTR;
	else if (!(m = calloc(1, sizeof(*m))))
		err = CL_OUT_OF_HOST_MEMORY;
	else if (!(m->dev = dev_alloc(size, &m->block)))
		err = CL_MEM_OBJECT_ALLOCATION_FAILURE;
	if (err != CL_SUCCESS) {
		free(m);
		m = NULL;
	} else {
		m->magic = MAGIC_MEM;
		m->refs = 1;
		m->ctx = context;
		cl_context_retain(context);
		m->flags = flags;
		m->size = size;
		if (flags & CL_MEM_USE_HOST_PTR)
			m->host_ptr = host_ptr;
		if (host_ptr)
			memcpy(m->dev, host_ptr, size);
	}
	cl_unlock();
	if (errcode_ret)
		*errcode_ret = err;
	return m;
}

CL_API_ENTRY cl_mem CL_API_CALL
clCreateSubBuffer(cl_mem buffer, cl_mem_flags flags, cl_buffer_create_type type,
		  const void *info, cl_int *errcode_ret)
{
	const cl_buffer_region *r = info;
	cl_mem m = NULL;
	cl_int err = CL_SUCCESS;

	cl_lock();
	if (!VALID(buffer, MAGIC_MEM) || buffer->parent) {
		err = CL_INVALID_MEM_OBJECT;
	} else if (flags & ~(cl_mem_flags)ACCESS_FLAGS || bad_access_flags(flags) ||
		   (buffer->flags & CL_MEM_WRITE_ONLY && flags & (CL_MEM_READ_WRITE | CL_MEM_READ_ONLY)) ||
		   (buffer->flags & CL_MEM_READ_ONLY && flags & (CL_MEM_READ_WRITE | CL_MEM_WRITE_ONLY)) ||
		   type != CL_BUFFER_CREATE_TYPE_REGION || !r ||
		   r->origin > buffer->size || r->size > buffer->size - r->origin) {
		err = CL_INVALID_VALUE;
	} else if (r->size == 0) {
		err = CL_INVALID_BUFFER_SIZE;
	} else if (r->origin % DEV_MEM_ALIGN) {
		err = CL_MISALIGNED_SUB_BUFFER_OFFSET;
	} else if (!(m = calloc(1, sizeof(*m)))) {
		err = CL_OUT_OF_HOST_MEMORY;
	} else {
		m->magic = MAGIC_MEM;
		m->refs = 1;
		m->ctx = buffer->ctx;
		cl_context_retain(m->ctx);
		m->flags = (flags & ACCESS_FLAGS ? flags : buffer->flags & ACCESS_FLAGS) |
			   (buffer->flags & HOST_FLAGS);
		m->size = r->size;
		m->parent = buffer;
		cl_mem_retain(buffer);
		m->origin = r->origin;
		m->dev = buffer->dev + r->origin;
		if (buffer->host_ptr)
			m->host_ptr = (uint8_t *)buffer->host_ptr + r->origin;
	}
	cl_unlock();
	if (errcode_ret)
		*errcode_ret = err;
	return m;
}

CL_API_ENTRY cl_int CL_API_CALL
clRetainMemObject(cl_mem memobj)
{
	cl_int err = CL_INVALID_MEM_OBJECT;

	cl_lock();
	if (VALID(memobj, MAGIC_MEM)) {
		cl_mem_retain(memobj);
		err = CL_SUCCESS;
	}
	cl_unlock();
	return err;
}

CL_API_ENTRY cl_int CL_API_CALL
clReleaseMemObject(cl_mem memobj)
{
	cl_int err = CL_INVALID_MEM_OBJECT;

	cl_lock();
	if (VALID(memobj, MAGIC_MEM)) {
		cl_mem_release(memobj);
		err = CL_SUCCESS;
	}
	cl_unlock();
	return err;
}

CL_API_ENTRY cl_int CL_API_CALL
clSetMemObjectDestructorCallback(cl_mem memobj, void (CL_CALLBACK *pfn_notify)(cl_mem, void *),
				 void *user_data)
{
	callback *cb;
	cl_int err = CL_SUCCESS;

	cl_lock();
	if (!VALID(memobj, MAGIC_MEM)) {
		err = CL_INVALID_MEM_OBJECT;
	} else if (!pfn_notify) {
		err = CL_INVALID_VALUE;
	} else if (!(cb = malloc(sizeof(*cb)))) {
		err = CL_OUT_OF_HOST_MEMORY;
	} else {
		cb->fn = (void (CL_CALLBACK *)())pfn_notify;
		cb->user_data = user_data;
		cb->next = memobj->destructors;
		memobj->destructors = cb;
	}
	cl_unlock();
	return err;
}

CL_API_ENTRY cl_int CL_API_CALL
clGetMemObjectInfo(cl_mem memobj, cl_mem_info param_name, size_t size, void *value,
		   size_t *size_ret)
{
	union {
		cl_mem_object_type type;
		cl_mem_flags flags;
		size_t sz;
		void *ptr;
		cl_uint u;
		cl_context ctx;
		cl_mem mem;
	} v;
	size_t n;
	cl_int err;

	cl_lock();
	if (!VALID(memobj, MAGIC_MEM)) {
		err = CL_INVALID_MEM_OBJECT;
		goto out;
	}
	switch (param_name) {
	case CL_MEM_TYPE:
		v.type = CL_MEM_OBJECT_BUFFER;
		n = sizeof(v.type);
		break;
	case CL_MEM_FLAGS:
		v.flags = memobj->flags;
		n = sizeof(v.flags);
		break;
	case CL_MEM_SIZE:
		v.sz = memobj->size;
		n = sizeof(v.sz);
		break;
	case CL_MEM_HOST_PTR:
		v.ptr = memobj->host_ptr;
		n = sizeof(v.ptr);
		break;
	case CL_MEM_MAP_COUNT:
		v.u = memobj->map_count;
		n = sizeof(v.u);
		break;
	case CL_MEM_REFERENCE_COUNT:
		v.u = memobj->refs;
		n = sizeof(v.u);
		break;
	case CL_MEM_CONTEXT:
		v.ctx = memobj->ctx;
		n = sizeof(v.ctx);
		break;
	case CL_MEM_ASSOCIATED_MEMOBJECT:
		v.mem = memobj->parent;
		n = sizeof(v.mem);
		break;
	case CL_MEM_OFFSET:
		v.sz = memobj->origin;
		n = sizeof(v.sz);
		break;
	default:
		err = CL_INVALID_VALUE;
		goto out;
	}
	err = cl_info(size, value, size_ret, &v, n);
out:
	cl_unlock();
	return err;
}

static cl_int check_command(cl_command_queue q, cl_mem m, cl_uint n, const cl_event *list)
{
	if (!VALID(q, MAGIC_QUEUE))
		return CL_INVALID_COMMAND_QUEUE;
	if (!VALID(m, MAGIC_MEM))
		return CL_INVALID_MEM_OBJECT;
	if (m->ctx != q->ctx)
		return CL_INVALID_CONTEXT;
	return cl_check_wait_list(q->ctx, n, list);
}

static int out_of_range(cl_mem m, size_t offset, size_t size)
{
	return size == 0 || offset > m->size || size > m->size - offset;
}

static cl_int transfer(cl_command_queue q, cl_command_type type, cl_mem m, cl_bool blocking,
		       size_t offset, size_t size, void *ptr, cl_uint n, const cl_event *list,
		       cl_event *event)
{
	command cmd;
	cl_int err = check_command(q, m, n, list);

	if (err != CL_SUCCESS)
		return err;
	if (out_of_range(m, offset, size) || !ptr)
		return CL_INVALID_VALUE;
	memset(&cmd, 0, sizeof(cmd));
	cmd.mem[0] = m;
	cl_mem_retain(m);
	cmd.offset[0] = offset;
	cmd.size = size;
	cmd.ptr = ptr;
	return cl_enqueue(q, type, &cmd, n, list, event, blocking);
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueReadBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read,
		    size_t offset, size_t size, void *ptr, cl_uint num_events_in_wait_list,
		    const cl_event *event_wait_list, cl_event *event)
{
	cl_int err;

	cl_lock();
	err = transfer(command_queue, CL_COMMAND_READ_BUFFER, buffer, blocking_read, offset, size,
		       ptr, num_events_in_wait_list, event_wait_list, event);
	cl_unlock();
	return err;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueWriteBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write,
		     size_t offset, size_t size, const void *ptr, cl_uint num_events_in_wait_list,
		     const cl_event *event_wait_list, cl_event *event)
{
	cl_int err;

	cl_lock();
	err = transfer(command_queue, CL_COMMAND_WRITE_BUFFER, buffer, blocking_write, offset, size,
		       (void *)ptr, num_events_in_wait_list, event_wait_list, event);
	cl_unlock();
	return err;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueCopyBuffer(cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_buffer,
		    size_t src_offset, size_t dst_offset, size_t size,
		    cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
		    cl_event *event)
{
	command cmd;
	size_t sbase, dbase;
	cl_int err;

	cl_lock();
	err = check_command(command_queue, src_buffer, num_events_in_wait_list, event_wait_list);
	if (err == CL_SUCCESS && !VALID(dst_buffer, MAGIC_MEM))
		err = CL_INVALID_MEM_OBJECT;
	else if (err == CL_SUCCESS && dst_buffer->ctx != command_queue->ctx)
		err = CL_INVALID_CONTEXT;
	if (err != CL_SUCCESS)
		goto out;
	if (out_of_range(src_buffer, src_offset, size) || out_of_range(dst_buffer, dst_offset, size)) {
		err = CL_INVALID_VALUE;
		goto out;
	}
	if (cl_mem_root(src_buffer, &sbase) == cl_mem_root(dst_buffer, &dbase) &&
	    sbase + src_offset < dbase + dst_offset + size &&
	    dbase + dst_offset < sbase + src_offset + size) {
		err = CL_MEM_COPY_OVERLAP;
		goto out;
	}
	memset(&cmd, 0, sizeof(cmd));
	cmd.mem[0] = src_buffer;
	cmd.mem[1] = dst_buffer;
	cl_mem_retain(src_buffer);
	cl_mem_retain(dst_buffer);
	cmd.offset[0] = src_offset;
	cmd.offset[1] = dst_offset;
	cmd.size = size;
	err = cl_enqueue(command_queue, CL_COMMAND_COPY_BUFFER, &cmd, num_events_in_wait_list,
			 event_wait_list, event, 0);
out:
	cl_unlock();
	return err;
}

/*
 * Check a rect's origin and region, filling in the default pitches; limit is
 * the size of the memory it is in (0: the host's, unknown).
 */
static cl_int check_rect(const size_t *origin, const size_t *region, size_t *row, size_t *slice,
			 size_t limit)
{
	uint64_t end;

	if (!origin || !region || !region[0] || !region[1] || !region[2])
		return CL_INVALID_VALUE;
	if (!*row)
		*row = region[0];
	else if (*row < region[0])
		return CL_INVALID_VALUE;
	if (!*slice)
		*slice = region[1] * *row;
	else if (*slice < (uint64_t)region[1] * *row)
		return CL_INVALID_VALUE;
	end = (uint64_t)(origin[2] + region[2] - 1) * *slice +
	      (uint64_t)(origin[1] + region[1] - 1) * *row + origin[0] + region[0];
	if (limit && end > limit)
		return CL_INVALID_VALUE;
	return CL_SUCCESS;
}

static void copy_rect(uint8_t *dst, const size_t *dorigin, size_t drow, size_t dslice,
		      const uint8_t *src, const size_t *sorigin, size_t srow, size_t sslice,
		      const size_t *region)
{
	size_t y, z;

	for (z = 0; z < region[2]; z++)
		for (y = 0; y < region[1]; y++)
			memcpy(dst + (dorigin[2] + z) * dslice + (dorigin[1] + y) * drow + dorigin[0],
			       src + (sorigin[2] + z) * sslice + (sorigin[1] + y) * srow + sorigin[0],
			       region[0]);
}

/*
 * Whether two rects of region (at absolute offsets base + origin) share a
 * byte.  The rows of each are disjoint and in increasing order, so one pass
 * over both finds any overlap.
 */
static int rects_overlap(size_t abase, const size_t *aorigin, size_t arow, size_t aslice,
			 size_t bbase, const size_t *borigin, size_t brow, size_t bslice,
			 const size_t *region)
{
	size_t ai = 0, bi = 0, rows = region[1] * region[2];

	while (ai < rows && bi < rows) {
		size_t a = abase + (aorigin[2] + ai / region[1]) * aslice +
			   (aorigin[1] + ai % region[1]) * arow + aorigin[0];
		size_t b = bbase + (borigin[2] + bi / region[1]) * bslice +
			   (borigin[1] + bi % region[1]) * brow + borigin[0];

		if (a < b + region[0] && b < a + region[0])
			return 1;
		if (a < b)
			ai++;
		else
			bi++;
	}
	return 0;
}

static cl_int transfer_rect(cl_command_queue q, cl_command_type type, cl_mem m, cl_bool blocking,
			    const size_t *buffer_origin, const size_t *host_origin,
			    const size_t *region, size_t buffer_row, size_t buffer_slice,
			    size_t host_row, size_t host_slice, void *ptr, cl_uint n,
			    const cl_event *list, cl_event *event)
{
	command cmd;
	cl_int err = check_command(q, m, n, list);

	if (err == CL_SUCCESS)
		err = check_rect(buffer_origin, region, &buffer_row, &buffer_slice, m->size);
	if (err == CL_SUCCESS)
		err = check_rect(host_origin, region, &host_row, &host_slice, 0);
	if (err == CL_SUCCESS && !ptr)
		err = CL_INVALID_VALUE;
	if (err != CL_SUCCESS)
		return err;
	memset(&cmd, 0, sizeof(cmd));
	cmd.mem[0] = m;
	cl_mem_retain(m);
	memcpy(cmd.origin[0], buffer_origin, sizeof(cmd.origin[0]));
	memcpy(cmd.origin[1], host_origin, sizeof(cmd.origin[1]));
	memcpy(cmd.region, region, sizeof(cmd.region));
	cmd.row_pitch[0] = buffer_row;
	cmd.slice_pitch[0] = buffer_slice;
	cmd.row_pitch[1] = host_row;
	cmd.slice_pitch[1] = host_slice;
	cmd.ptr = ptr;
	return cl_enqueue(q, type, &cmd, n, list, event, blocking);
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueReadBufferRect(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read,
			const size_t *buffer_origin, const size_t *host_origin,
			const size_t *region, size_t buffer_row_pitch, size_t buffer_slice_pitch,
			size_t host_row_pitch, size_t host_slice_pitch, void *ptr,
			cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
			cl_event *event)
{
	cl_int err;

	cl_lock();
	err = transfer_rect(command_queue, CL_COMMAND_READ_BUFFER_RECT, buffer, blocking_read,
			    buffer_origin, host_origin, region, buffer_row_pitch,
			    buffer_slice_pitch, host_row_pitch, host_slice_pitch, ptr,
			    num_events_in_wait_list, event_wait_list, event);
	cl_unlock();
	return err;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueWriteBufferRect(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write,
			 const size_t *buffer_origin, const size_t *host_origin,
			 const size_t *region, size_t buffer_row_pitch, size_t buffer_slice_pitch,
			 size_t host_row_pitch, size_t host_slice_pitch, const void *ptr,
			 cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
			 cl_event *event)
{
	cl_int err;

	cl_lock();
	err = transfer_rect(command_queue, CL_COMMAND_WRITE_BUFFER_RECT, buffer, blocking_write,
			    buffer_origin, host_origin, region, buffer_row_pitch,
			    buffer_slice_pitch, host_row_pitch, host_slice_pitch, (void *)ptr,
			    num_events_in_wait_list, event_wait_list, event);
	cl_unlock();
	return err;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueCopyBufferRect(cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_buffer,
			const size_t *src_origin, const size_t *dst_origin, const size_t *region,
			size_t src_row_pitch, size_t src_slice_pitch, size_t dst_row_pitch,
			size_t dst_slice_pitch, cl_uint num_events_in_wait_list,
			const cl_event *event_wait_list, cl_event *event)
{
	command cmd;
	size_t sbase, dbase;
	cl_int err;

	cl_lock();
	err = check_command(command_queue, src_buffer, num_events_in_wait_list, event_wait_list);
	if (err == CL_SUCCESS && !VALID(dst_buffer, MAGIC_MEM))
		err = CL_INVALID_MEM_OBJECT;
	else if (err == CL_SUCCESS && dst_buffer->ctx != command_queue->ctx)
		err = CL_INVALID_CONTEXT;
	if (err == CL_SUCCESS)
		err = check_rect(src_origin, region, &src_row_pitch, &src_slice_pitch,
				 src_buffer->size);
	if (err == CL_SUCCESS)
		err = check_rect(dst_origin, region, &dst_row_pitch, &dst_slice_pitch,
				 dst_buffer->size);
	if (err == CL_SUCCESS && src_buffer == dst_buffer &&
	    (src_row_pitch != dst_row_pitch || src_slice_pitch != dst_slice_pitch))
		err = CL_INVALID_VALUE;
	if (err == CL_SUCCESS && cl_mem_root(src_buffer, &sbase) == cl_mem_root(dst_buffer, &dbase) &&
	    rects_overlap(sbase, src_origin, src_row_pitch, src_slice_pitch,
			  dbase, dst_origin, dst_row_pitch, dst_slice_pitch, region))
		err = CL_MEM_COPY_OVERLAP;
	if (err != CL_SUCCESS)
		goto out;
	memset(&cmd, 0, sizeof(cmd));
	cmd.mem[0] = src_buffer;
	cmd.mem[1] = dst_buffer;
	cl_mem_retain(src_buffer);
	cl_mem_retain(dst_buffer);
	memcpy(cmd.origin[0], src_origin, sizeof(cmd.origin[0]));
	memcpy(cmd.origin[1], dst_origin, sizeof(cmd.origin[1]));
	memcpy(cmd.region, region, sizeof(cmd.region));
	cmd.row_pitch[0] = src_row_pitch;
	cmd.slice_pitch[0] = src_slice_pitch;
	cmd.row_pitch[1] = dst_row_pitch;
	cmd.slice_pitch[1] = dst_slice_pitch;
	err = cl_enqueue(command_queue, CL_COMMAND_COPY_BUFFER_RECT, &cmd, num_events_in_wait_list,
			 event_wait_list, event, 0);
out:
	cl_unlock();
	return err;
}

CL_API_ENTRY void * CL_API_CALL
clEnqueueMapBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_map,
		   cl_map_flags map_flags, size_t offset, size_t size,
		   cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
		   cl_event *event, cl_int *errcode_ret)
{
	command cmd;
	mapping *map = NULL, **link;
	void *ptr = NULL;
	cl_int err;

	cl_lock();
	err = check_command(command_queue, buffer, num_events_in_wait_list, event_wait_list);
	if (err == CL_SUCCESS && (map_flags & ~(cl_map_flags)(CL_MAP_READ | CL_MAP_WRITE) ||
				  out_of_range(buffer, offset, size)))
		err = CL_INVALID_VALUE;
	if (err == CL_SUCCESS && !(map = malloc(sizeof(*map))))
		err = CL_OUT_OF_HOST_MEMORY;
	if (err != CL_SUCCESS)
		goto out;
	ptr = buffer->host_ptr ? (uint8_t *)buffer->host_ptr + offset : buffer->dev + offset;
	map->ptr = ptr;
	map->offset = offset;
	map->size = size;
	map->flags = map_flags;
	map->next = buffer->maps;
	buffer->maps = map;
	buffer->map_count++;
	memset(&cmd, 0, sizeof(cmd));
	cmd.mem[0] = buffer;
	cl_mem_retain(buffer);
	cmd.offset[0] = offset;
	cmd.size = size;
	cmd.ptr = ptr;
	err = cl_enqueue(command_queue, CL_COMMAND_MAP_BUFFER, &cmd, num_events_in_wait_list,
			 event_wait_list, event, blocking_map);
	if (err != CL_SUCCESS) {
		/* not mapped after all */
		for (link = &buffer->maps; *link != map; link = &(*link)->next)
			;
		*link = map->next;
		free(map);
		buffer->map_count--;
		ptr = NULL;
	}
out:
	cl_unlock();
	if (errcode_ret)
		*errcode_ret = err;
	return ptr;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueUnmapMemObject(cl_command_queue command_queue, cl_mem memobj, void *mapped_ptr,
			cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
			cl_event *event)
{
	command cmd;
	mapping **link;
	cl_int err;

	cl_lock();
	err = check_command(command_queue, memobj, num_events_in_wait_list, event_wait_list);
	if (err != CL_SUCCESS)
		goto out;
	for (link = &memobj->maps; *link && (*link)->ptr != mapped_ptr; link = &(*link)->next)
		;
	if (!*link) {
		err = CL_INVALID_VALUE;
		goto out;
	}
	memset(&cmd, 0, sizeof(cmd));
	cmd.map = *link;
	*link = cmd.map->next;
	memobj->map_count--;
	cmd.mem[0] = memobj;
	cl_mem_retain(memobj);
	err = cl_enqueue(command_queue, CL_COMMAND_UNMAP_MEM_OBJECT, &cmd, num_events_in_wait_list,
			 event_wait_list, event, 0);
out:
	cl_unlock();
	return err;
}

cl_int cl_run_mem_command(cl_event e)
{
	command *c = &e->cmd;
	cl_mem m = c->mem[0];

	switch (e->type) {
	case CL_COMMAND_READ_BUFFER:
		memcpy(c->ptr, m->dev + c->offset[0], c->size);
		break;
	case CL_COMMAND_WRITE_BUFFER:
		memcpy(m->dev + c->offset[0], c->ptr, c->size);
		break;
	case CL_COMMAND_COPY_BUFFER:
		memcpy(c->mem[1]->dev + c->offset[1], m->dev + c->offset[0], c->size);
		break;
	case CL_COMMAND_READ_BUFFER_RECT:
		copy_rect(c->ptr, c->origin[1], c->row_pitch[1], c->slice_pitch[1],
			  m->dev, c->origin[0], c->row_pitch[0], c->slice_pitch[0], c->region);
		break;
	case CL_COMMAND_WRITE_BUFFER_RECT:
		copy_rect(m->dev, c->origin[0], c->row_pitch[0], c->slice_pitch[0],
			  c->ptr, c->origin[1], c->row_pitch[1], c->slice_pitch[1], c->region);
		break;
	case CL_COMMAND_COPY_BUFFER_RECT:
		copy_rect(c->mem[1]->dev, c->origin[1], c->row_pitch[1], c->slice_pitch[1],
			  m->dev, c->origin[0], c->row_pitch[0], c->slice_pitch[0], c->region);
		break;
	case CL_COMMAND_MAP_BUFFER:
		if (m->host_ptr)
			memcpy(c->ptr, m->dev + c->offset[0], c->size);
		break;
	case CL_COMMAND_UNMAP_MEM_OBJECT:
		if (m->host_ptr && c->map->flags & CL_MAP_WRITE)
			memcpy(m->dev + c->map->offset, c->map->ptr, c->map->size);
		break;
	default:
		return CL_INVALID_OPERATION;
	}
	return CL_COMPLETE;
}

/* No images or samplers: no context can make one. */

CL_API_ENTRY cl_mem CL_API_CALL
clCreateImage2D(cl_context context, cl_mem_flags flags, const cl_image_format *image_format,
		size_t image_width, size_t image_height, size_t image_row_pitch, void *host_ptr,
		cl_int *errcode_ret)
{
	(void)flags, (void)image_format, (void)image_width, (void)image_height;
	(void)image_row_pitch, (void)host_ptr;
	if (errcode_ret)
		*errcode_ret = VALID(context, MAGIC_CONTEXT) ? CL_INVALID_OPERATION : CL_INVALID_CONTEXT;
	return NULL;
}

CL_API_ENTRY cl_mem CL_API_CALL
clCreateImage3D(cl_context context, cl_mem_flags flags, const cl_image_format *image_format,
		size_t image_width, size_t image_height, size_t image_depth,
		size_t image_row_pitch, size_t image_slice_pitch, void *host_ptr,
		cl_int *errcode_ret)
{
	(void)flags, (void)image_format, (void)image_width, (void)image_height, (void)image_depth;
	(void)image_row_pitch, (void)image_slice_pitch, (void)host_ptr;
	if (errcode_ret)
		*errcode_ret = VALID(context, MAGIC_CONTEXT) ? CL_INVALID_OPERATION : CL_INVALID_CONTEXT;
	return NULL;
}

CL_API_ENTRY cl_int CL_API_CALL
clGetSupportedImageFormats(cl_context context, cl_mem_flags flags, cl_mem_object_type image_type,
			   cl_uint num_entries, cl_image_format *image_formats,
			   cl_uint *num_image_formats)
{
	(void)flags, (void)num_entries, (void)image_formats;
	if (!VALID(context, MAGIC_CONTEXT))
		return CL_INVALID_CONTEXT;
	if (image_type != CL_MEM_OBJECT_IMAGE2D && image_type != CL_MEM_OBJECT_IMAGE3D)
		return CL_INVALID_VALUE;
	if (num_image_formats)
		*num_image_formats = 0;
	return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL
clGetImageInfo(cl_mem image, cl_image_info param_name, size_t size, void *value,
	       size_t *size_ret)
{
	(void)image, (void)param_name, (void)size, (void)value, (void)size_ret;
	return CL_INVALID_MEM_OBJECT;
}

static cl_int image_command(cl_command_queue q)
{
	return VALID(q, MAGIC_QUEUE) ? CL_INVALID_MEM_OBJECT : CL_INVALID_COMMAND_QUEUE;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueReadImage(cl_command_queue command_queue, cl_mem image, cl_bool blocking_read,
		   const size_t *origin, const size_t *region, size_t row_pitch, size_t slice_pitch,
		   void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
		   cl_event *event)
{
	(void)image, (void)blocking_read, (void)origin, (void)region, (void)row_pitch;
	(void)slice_pitch, (void)ptr, (void)num_events_in_wait_list, (void)event_wait_list, (void)event;
	return image_command(command_queue);
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueWriteImage(cl_command_queue command_queue, cl_mem image, cl_bool blocking_write,
		    const size_t *origin, const size_t *region, size_t input_row_pitch,
		    size_t input_slice_pitch, const void *ptr, cl_uint num_events_in_wait_list,
		    const cl_event *event_wait_list, cl_event *event)
{
	(void)image, (void)blocking_write, (void)origin, (void)region, (void)input_row_pitch;
	(void)input_slice_pitch, (void)ptr, (void)num_events_in_wait_list, (void)event_wait_list;
	(void)event;
	return image_command(command_queue);
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueCopyImage(cl_command_queue command_queue, cl_mem src_image, cl_mem dst_image,
		   const size_t *src_origin, const size_t *dst_origin, const size_t *region,
		   cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
		   cl_event *event)
{
	(void)src_image, (void)dst_image, (void)src_origin, (void)dst_origin, (void)region;
	(void)num_events_in_wait_list, (void)event_wait_list, (void)event;
	return image_command(command_queue);
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueCopyImageToBuffer(cl_command_queue command_queue, cl_mem src_image, cl_mem dst_buffer,
			   const size_t *src_origin, const size_t *region, size_t dst_offset,
			   cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
			   cl_event *event)
{
	(void)src_image, (void)dst_buffer, (void)src_origin, (void)region, (void)dst_offset;
	(void)num_events_in_wait_list, (void)event_wait_list, (void)event;
	return image_command(command_queue);
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueCopyBufferToImage(cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_image,
			   size_t src_offset, const size_t *dst_origin, const size_t *region,
			   cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
			   cl_event *event)
{
	(void)src_buffer, (void)dst_image, (void)src_offset, (void)dst_origin, (void)region;
	(void)num_events_in_wait_list, (void)event_wait_list, (void)event;
	return image_command(command_queue);
}

CL_API_ENTRY void * CL_API_CALL
clEnqueueMapImage(cl_command_queue command_queue, cl_mem image, cl_bool blocking_map,
		  cl_map_flags map_flags, const size_t *origin, const size_t *region,
		  size_t *image_row_pitch, size_t *image_slice_pitch,
		  cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
		  cl_event *event, cl_int *errcode_ret)
{
	(void)image, (void)blocking_map, (void)map_flags, (void)origin, (void)region;
	(void)image_row_pitch, (void)image_slice_pitch, (void)num_events_in_wait_list;
	(void)event_wait_list, (void)event;
	if (errcode_ret)
		*errcode_ret = image_command(command_queue);
	return NULL;
}

CL_API_ENTRY cl_sampler CL_API_CALL
clCreateSampler(cl_context context, cl_bool normalized_coords, cl_addressing_mode addressing_mode,
		cl_filter_mode filter_mode, cl_int *errcode_ret)
{
	(void)normalized_coords, (void)addressing_mode, (void)filter_mode;
	if (errcode_ret)
		*errcode_ret = VALID(context, MAGIC_CONTEXT) ? CL_INVALID_OPERATION : CL_INVALID_CONTEXT;
	return NULL;
}

CL_API_ENTRY cl_int CL_API_CALL
clRetainSampler(cl_sampler sampler)
{
	(void)sampler;
	return CL_INVALID_SAMPLER;
}

CL_API_ENTRY cl_int CL_API_CALL
clReleaseSampler(cl_sampler sampler)
{
	(void)sampler;
	return CL_INVALID_SAMPLER;
}

CL_API_ENTRY cl_int CL_API_CALL
clGetSamplerInfo(cl_sampler sampler, cl_sampler_info param_name, size_t size, void *value,
		 size_t *size_ret)
{
	(void)sampler, (void)param_name, (void)size, (void)value, (void)size_ret;
	return CL_INVALID_SAMPLER;
}
