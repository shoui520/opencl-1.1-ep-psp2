/* Stand-in for the Sony SDK <kernel.h> used by PVR_PSP2's services.h. */
#ifndef PVRSC_COMPAT_KERNEL_H
#define PVRSC_COMPAT_KERNEL_H

#include <psp2/kernel/threadmgr.h>
#include <psp2/kernel/error.h>
#include <psp2/kernel/processmgr.h>

#ifndef SCE_NULL
#define SCE_NULL NULL
#endif
#ifndef SCE_OK
#define SCE_OK 0
#endif

#endif
