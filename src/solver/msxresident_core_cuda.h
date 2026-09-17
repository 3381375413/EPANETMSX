#ifndef MSXRESIDENT_CORE_CUDA_H
#define MSXRESIDENT_CORE_CUDA_H

/* Deliberately C-only: Phase 3 owns all runtime/configuration wiring. */
#include "msxresident_core.h"
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct MSXResidentGpu MSXResidentGpu;
typedef struct { uint32_t nLinks,totalSlots,speciesStride; const uint32_t *capacity,*base; } MSXResidentGpuOpen;
/* Fetch owns neither the metadata array nor the mutable concentration rows.
   cOut/lastcOut contain itemCount rows with this caller-provided stride. */
typedef struct { MSXResidentHandoffResult *meta; double *cOut,*lastcOut; uint32_t stride; } MSXResidentGpuFetchOutput;
typedef struct { uint64_t activeCount,checksumXor,checksumSum,staleGeneration,epochMismatch,descriptorReject,capacityOverflow,fallbacks,handoffCount,capacityReject,generationReject,epochStale,fetchStale,cudaErrors,cudaPoisons; double activeVolume; uint32_t nanCount,infCount,errorCount,firstErrorSlot; } MSXResidentGpuReduction;
/* Phase 3B deliberately exposes no CUDA headers.  ``globalRow`` is the
   fixed row in the published resident span, never a Pseg or ring index. */
typedef struct { uint32_t linkIndex,globalRow,generation; uint64_t descriptorEpoch; double volume; const double *hyd; } MSXResidentActiveItem;
typedef struct { const MSXResidentActiveItem *item; uint32_t itemCount; } MSXResidentActiveBatch;
/* All addresses are plain integer values so this ABI is usable by C callers
   without importing CUDA headers.  They belong to the primary CUDA context. */
typedef struct { uint64_t segPipe,segRow,segVol,hstep,c,lastc,hyd,reacted,ros2Nfcn,ros2Njac,ros2Naccept,ros2Nreject,ros2LastHstep,ros2Err; uint32_t itemCount,speciesStride,hydStride; } MSXResidentGpuDeviceView;
/* reacted points at resident-owned host memory after finishActive succeeds.
   It has reactedLinkCount rows of reactedStride doubles and stays valid until
   the next prepareActive/close.  Phase 3C owns applying it to MSX.Link[]. */
typedef struct { double activeUploadMs,gatherMs,ros2Ms,equilMs,formulaMs,diagDownloadMs,syncMs; uint64_t ros2Nfcn,ros2Njac,ros2Naccept,ros2Nreject; double ros2LastHstep; int ros2Error; uint32_t nanCount,infCount; const double *reacted; uint32_t reactedStride,reactedLinkCount; } MSXResidentGpuReactResult;

MSXResidentStatus MSXresidentGpu_open(const MSXResidentGpuOpen *, MSXResidentGpu **);
MSXResidentStatus MSXresidentGpu_initialUpload(MSXResidentGpu *, const MSXResidentPatchBatch *);
MSXResidentStatus MSXresidentGpu_applyPatches(MSXResidentGpu *, const MSXResidentPatchBatch *);
MSXResidentStatus MSXresidentGpu_fetchHandoffs(MSXResidentGpu *, const MSXResidentHandoffPlan *, MSXResidentGpuFetchOutput *, uint32_t);
/* massBySpecies has speciesStride entries (including index zero) supplied by caller. */
MSXResidentStatus MSXresidentGpu_reduce(MSXResidentGpu *, double *massBySpecies, uint32_t massCount, MSXResidentGpuReduction *);
/* Device 0 runtime primary context is established by open; Phase 3C calls
   MSXgpu_prepareResidentContext before open so driver and runtime share it.
   Preflight is all-or-nothing: no device write occurs on any reject.  The
   returned pointers stay valid until finish/close and share msxgpu's primary
   CUDA context.  msxgpu owns the ROS2/EQUIL/FORMULA launches. */
MSXResidentStatus MSXresidentGpu_prepareActive(MSXResidentGpu *, const MSXResidentActiveBatch *, MSXResidentGpuDeviceView *, MSXResidentGpuReactResult *);
MSXResidentStatus MSXresidentGpu_getDeviceView(MSXResidentGpu *, MSXResidentGpuDeviceView *);
MSXResidentStatus MSXresidentGpu_finishActive(MSXResidentGpu *, MSXResidentGpuReactResult *);
/* Abort poisons the mirror and releases an in-flight active batch.  It is
   intentionally fail-closed: callers must reopen before another dispatch. */
MSXResidentStatus MSXresidentGpu_abortActive(MSXResidentGpu *);
void MSXresidentGpu_close(MSXResidentGpu *);
int MSXresidentGpu_isEnabled(void);

#ifdef __cplusplus
}
#endif
#endif
