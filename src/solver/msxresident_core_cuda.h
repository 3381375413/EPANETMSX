#ifndef MSXRESIDENT_CORE_CUDA_H
#define MSXRESIDENT_CORE_CUDA_H

/* Deliberately C-only: Phase 3 owns all runtime/configuration wiring. */
#include "msxresident_core.h"
#include <stdint.h>

/* C ABI row width for resident hydraulic values; MSX HydVar[0..9]. */
#define MSX_RESIDENT_HYD_STRIDE 10

/* Internal hydraulic-table layouts.  ACTIVE_MAJOR is retained only for the
   legacy non-Resident/contract wrapper; Resident dispatch uses PIPE_MAJOR. */
#define MSX_RESIDENT_HYD_ACTIVE_MAJOR 0u
#define MSX_RESIDENT_HYD_PIPE_MAJOR   1u

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MSXResidentGpu MSXResidentGpu;
typedef struct { uint32_t nLinks,totalSlots,speciesStride; const uint32_t *capacity,*base; } MSXResidentGpuOpen;
/* Fetch owns neither the metadata array nor the mutable concentration rows.
   cOut/lastcOut contain itemCount rows with this caller-provided stride. */
typedef struct { MSXResidentHandoffResult *meta; double *cOut,*lastcOut; uint32_t stride; } MSXResidentGpuFetchOutput;
typedef struct { uint64_t activeCount,checksumXor,checksumSum,staleGeneration,epochMismatch,descriptorReject,capacityOverflow,fallbacks,handoffCount,capacityReject,generationReject,epochStale,fetchStale,cudaErrors,cudaPoisons; double activeVolume; uint32_t nanCount,infCount,errorCount,firstErrorSlot; } MSXResidentGpuReduction;
/* Exact CUDA API copies made by a resident handoff fetch.  These counters are
   intentionally separate from logical rows/events and are used by the
   contract harness to prove that a multi-link selected batch is one H2D stage
   upload plus one metadata/concentration/last-concentration D2H triplet. */
typedef struct {
    uint64_t h2dBytes,h2dCalls,d2hBytes,d2hCalls;
    /* B2 resident pipe-Hyd traffic and exact-table comparison counters. */
    uint64_t hydH2DBytes,hydH2DCalls;
    uint64_t hydCandidateComparisons,hydUploads,hydSkips,hydBytes,hydApiCalls;
    int hydAppliedValid,hydPending;
} MSXResidentGpuTransferStats;
/* Phase 3B deliberately exposes no CUDA headers.  ``globalRow`` is the
   fixed row in the published resident span, never a Pseg or ring index. */
typedef struct { uint32_t linkIndex,globalRow,generation; uint64_t descriptorEpoch; double volume; const double *hyd; } MSXResidentActiveItem;
typedef struct { const MSXResidentActiveItem *item; uint32_t itemCount; } MSXResidentActiveBatch;
/* Read-only source table supplied for a Resident reaction.  The caller must
   initialize every (linkCount+1)*stride double before submit. */
typedef struct { const double *pipeHyd; uint32_t linkCount,hydStride,hydLayout; } MSXResidentHydView;
/* Non-destructive GPU-to-CPU mirror refresh for the just-completed active
   batch.  This never materializes a handoff or changes descriptor topology. */
typedef struct { uint32_t linkIndex,globalRow,generation; uint64_t descriptorEpoch;
    double hstep,hresponse,uresponse,dresponse; } MSXResidentGpuActiveSyncRow;
typedef struct { MSXResidentGpuActiveSyncRow *row; double *cOut,*lastcOut;
    uint32_t stride; } MSXResidentGpuActiveSyncOutput;
/* All addresses are plain integer values so this ABI is usable by C callers
   without importing CUDA headers.  They belong to the primary CUDA context. */
typedef struct { uint64_t segPipe,segRow,segVol,hstep,c,lastc,hyd,reacted,ros2Nfcn,ros2Njac,ros2Naccept,ros2Nreject,ros2LastHstep,ros2Err,streamHandle; uint32_t itemCount,speciesStride,hydStride,hydLayout; } MSXResidentGpuDeviceView;
/* reacted points at resident-owned host memory after finishActive succeeds.
   It has reactedLinkCount rows of reactedStride doubles and stays valid until
   the next prepareActive/close.  Phase 3C owns applying it to MSX.Link[]. */
typedef struct { double activeUploadMs,gatherMs,ros2Ms,equilMs,formulaMs,diagDownloadMs,syncMs; uint64_t ros2Nfcn,ros2Njac,ros2Naccept,ros2Nreject; double ros2LastHstep; int ros2Error; uint32_t nanCount,infCount; const double *reacted; uint32_t reactedStride,reactedLinkCount; } MSXResidentGpuReactResult;

MSXResidentStatus MSXresidentGpu_open(const MSXResidentGpuOpen *, MSXResidentGpu **);
MSXResidentStatus MSXresidentGpu_initialUpload(MSXResidentGpu *, const MSXResidentPatchBatch *);
MSXResidentStatus MSXresidentGpu_applyPatches(MSXResidentGpu *, const MSXResidentPatchBatch *);
MSXResidentStatus MSXresidentGpu_fetchHandoffs(MSXResidentGpu *, const MSXResidentHandoffPlan *, MSXResidentGpuFetchOutput *, uint32_t);
/* Fetch a flat set of handoff rows in one GPU gather and one D2H triplet.
   Items may belong to different links and boundary plans; the caller keeps
   the flat order when preparing per-link CPU transactions. */
MSXResidentStatus MSXresidentGpu_fetchHandoffBatch(MSXResidentGpu *, const MSXResidentHandoffItem *, uint32_t, MSXResidentGpuFetchOutput *);
MSXResidentStatus MSXresidentGpu_getTransferStats(const MSXResidentGpu *, MSXResidentGpuTransferStats *);
/* massBySpecies has speciesStride entries (including index zero) supplied by caller. */
MSXResidentStatus MSXresidentGpu_reduce(MSXResidentGpu *, double *massBySpecies, uint32_t massCount, MSXResidentGpuReduction *);
/* Return one link's aggregate c*volume and volume from the same cached
   reduction snapshot used by MSXresidentGpu_reduce. */
MSXResidentStatus MSXresidentGpu_reduceLink(MSXResidentGpu *, uint32_t linkIndex,
                                             double *massBySpecies,
                                             uint32_t massCount,
                                             double *volume,
                                             MSXResidentGpuReduction *);
/* Device 0 runtime primary context is established by open; Phase 3C calls
   MSXgpu_prepareResidentContext before open so driver and runtime share it.
   Preflight is all-or-nothing: no device write occurs on any reject.  The
   returned pointers stay valid until finish/close and share msxgpu's primary
   CUDA context.  msxgpu owns the ROS2/EQUIL/FORMULA launches. */
MSXResidentStatus MSXresidentGpu_prepareActive(MSXResidentGpu *, const MSXResidentActiveBatch *, MSXResidentGpuDeviceView *, MSXResidentGpuReactResult *);
/* Resident path: compare a complete read-only pipe table by double bit
   pattern, enqueue one full-table H2D only when the applied version changed,
   then enqueue active metadata on the same stream. */
MSXResidentStatus MSXresidentGpu_prepareActiveHyd(MSXResidentGpu *, const MSXResidentActiveBatch *, const MSXResidentHydView *, MSXResidentGpuDeviceView *, MSXResidentGpuReactResult *);
/* Selects whether finishActive returns optional per-active solver counters.
   Required hstep/error/quality state is retained in every mode. */
void MSXresidentGpu_setDiagnosticMode(MSXResidentGpu *, int enabled);
MSXResidentStatus MSXresidentGpu_getDeviceView(MSXResidentGpu *, MSXResidentGpuDeviceView *);
/* Enqueue the compact error reduction and hstep scatter on the same resident
   stream as prepareActive and the chemistry kernels.  It does not wait; the
   single completion boundary is finishActive. */
MSXResidentStatus MSXresidentGpu_enqueueActiveCompletion(MSXResidentGpu *);
MSXResidentStatus MSXresidentGpu_finishActive(MSXResidentGpu *, MSXResidentGpuReactResult *);
/* Driver-side finish has already waited on the ready CUevent.  This entry
   point consumes the same completion state without a second stream wait. */
MSXResidentStatus MSXresidentGpu_finishActiveAfterWait(MSXResidentGpu *, MSXResidentGpuReactResult *);
/* Explicit debug snapshot only: normal Resident reaction consumes device-owned
   Core state and uses selected handoff fetches instead of this full gather. */
MSXResidentStatus MSXresidentGpu_syncActive(MSXResidentGpu *,
                                             MSXResidentGpuActiveSyncOutput *,
                                             uint32_t);
/* Abort poisons the mirror and releases an in-flight active batch.  It is
   intentionally fail-closed: callers must reopen before another dispatch. */
MSXResidentStatus MSXresidentGpu_abortActive(MSXResidentGpu *);
/* Invalidate the applied pipe-Hyd version at an MSXinit/re-open boundary.
   It is only legal while no active GPU batch is in flight. */
MSXResidentStatus MSXresidentGpu_invalidateHyd(MSXResidentGpu *);
/* Mark the device mirror unusable after a completed Driver program reports a
   chemistry error.  All later read/query/dispatch operations fail closed. */
MSXResidentStatus MSXresidentGpu_poison(MSXResidentGpu *);
void MSXresidentGpu_close(MSXResidentGpu *);
int MSXresidentGpu_isEnabled(void);

#ifdef __cplusplus
}
#endif
#endif
