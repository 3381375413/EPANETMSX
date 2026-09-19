#ifndef MSXRESIDENT_RUNTIME_H
#define MSXRESIDENT_RUNTIME_H

#include "msxresident_core.h"
#include "msxresident_core_cuda.h"
#include "msxgpu.h"

/* Resident close summaries are diagnostics, not part of the cache protocol.
   Keep this predicate pure so the CPU contract harness can cover the off and
   explicit-audit cases without linking the production runtime. */
static int MSXresidentRuntime_diagnosticSummaryGate(int diagnosticDetail,
                                                    int auditRequested)
{
    return diagnosticDetail || auditRequested;
}

/* Phase 3C-1 lifecycle boundary.  These hooks never alter segment topology. */
int MSXresidentRuntime_preHybridInit(void);
int MSXresidentRuntime_afterHybridInit(void);
int MSXresidentRuntime_flushPatches(void);
void MSXresidentRuntime_close(void);
int MSXresidentRuntime_residentNotReady(void);
int MSXresidentRuntime_isResident(void);
int MSXresidentRuntime_reactReady(void);
int MSXresidentRuntime_handoffReady(void);
/* Monotonic Resident CPU/GPU state epoch for versioned quality snapshots. */
uint64_t MSXresidentRuntime_stateVersion(void);
int MSXresidentRuntime_reactCore(double dt);
typedef struct {
    uint64_t magic;
    uint64_t stepId;
    uint64_t topologyVersion;
    uint64_t stateVersion;
    uint32_t batchCount;
    int inFlight;
    MSXgpuResidentCoreToken gpu;
} MSXResidentRuntimeToken;
int MSXresidentRuntime_submitCore(double dt, MSXResidentRuntimeToken *);
int MSXresidentRuntime_finishCore(MSXResidentRuntimeToken *);
int MSXresidentRuntime_abortCore(MSXResidentRuntimeToken *);
int MSXresidentRuntime_beginStep(double dt);
int MSXresidentRuntime_completeHandoffs(void);
/* Flush pending topology/slot patches, then return the cached or rebuilt
   Resident Core aggregate.  The caller owns the output buffers. */
MSXResidentStatus MSXresidentRuntime_reduce(double *massBySpecies,
                                            uint32_t massCount,
                                            MSXResidentGpuReduction *reduction);
/* Flush pending Resident patches and return one link's GPU aggregate.  The
   returned mass is c*volume in the Resident concentration/volume units; the
   caller combines it with CPU Boundary rows at the same state version. */
MSXResidentStatus MSXresidentRuntime_reduceLink(uint32_t linkIndex,
                                                double *massBySpecies,
                                                uint32_t massCount,
                                                double *volume,
                                                MSXResidentGpuReduction *reduction);
/* Fetch one selected Resident row from the authoritative GPU image.  c/lastc
   are caller-owned arrays of at least speciesStride entries. */
MSXResidentStatus MSXresidentRuntime_fetchSlot(uint32_t linkIndex,
                                               uint64_t parcelId,
                                               double *c, double *lastc,
                                               uint32_t stride,
                                               MSXResidentPayload *payload);
/* Fetch a caller-planned set of selected rows in one GPU gather.  c/lastc
   contain count packed rows and results receives the matching metadata. */
MSXResidentStatus MSXresidentRuntime_fetchBatch(
    const MSXResidentHandoffItem *items, uint32_t count,
    double *c, double *lastc, uint32_t stride,
    MSXResidentHandoffResult *results);
/* The quality loop calls this after post-transport Hybrid rebalance so the
   next pending patch batch is classified as demote/promote traffic. */
void MSXresidentRuntime_markRebalance(void);
int MSXresidentRuntime_linkFallback(int linkIndex);
const char *MSXresidentRuntime_status(void);
MSXResidentStatus MSXresidentRuntime_lastStatus(void);
const char *MSXresidentRuntime_resolvedCapacityPath(void);
typedef struct {
    uint64_t wouldDescriptorPatches,wouldSlotPatches,stalePatches,fallbacks;
    uint64_t activeIteratorPasses,activeRowsAppended,activeFullRowCopies,
             activeBuilderAborts;
    int opened,resident;
} MSXResidentRuntimeMetrics;
void MSXresidentRuntime_getMetrics(MSXResidentRuntimeMetrics *);

#endif
