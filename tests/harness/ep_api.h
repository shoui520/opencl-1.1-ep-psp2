#ifndef EP_API_H
#define EP_API_H

#include <CL/cl.h>
#include <stddef.h>

typedef void (*ep_log_fn)(const char *fmt, ...);

/* Check the API against the embedded profile; returns the failures, which are
 * also listed in out (one "section: what" line each). */
int ep_api_run(cl_context ctx, cl_command_queue q, cl_device_id dev, cl_program prog,
	       char *out, size_t out_size, ep_log_fn log);

#endif
