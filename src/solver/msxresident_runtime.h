#ifndef MSXRESIDENT_RUNTIME_H
#define MSXRESIDENT_RUNTIME_H

#include "msxresident_core.h"

/* Phase 3C-1 lifecycle boundary.  These hooks never alter segment topology. */
int MSXresidentRuntime_preHybridInit(void);
int MSXresidentRuntime_afterHybridInit(void);
int MSXresidentRuntime_flushPatches(void);
void MSXresidentRuntime_close(void);
int MSXresidentRuntime_residentNotReady(void);
int MSXresidentRuntime_isResident(void);
int MSXresidentRuntime_reactReady(void);
int MSXresidentRuntime_handoffReady(void);
int MSXresidentRuntime_reactCore(double dt);
int MSXresidentRuntime_beginStep(double dt);
int MSXresidentRuntime_completeHandoffs(void);
int MSXresidentRuntime_linkFallback(int linkIndex);
const char *MSXresidentRuntime_status(void);
MSXResidentStatus MSXresidentRuntime_lastStatus(void);
const char *MSXresidentRuntime_resolvedCapacityPath(void);
typedef struct { uint64_t wouldDescriptorPatches,wouldSlotPatches,stalePatches,fallbacks; int opened,resident; } MSXResidentRuntimeMetrics;
void MSXresidentRuntime_getMetrics(MSXResidentRuntimeMetrics *);

#endif
