/******************************************************************************
**  MODULE:        MSXSEGMENT_STORAGE.H
**  PROJECT:       EPANET-MSX GPU modified
**  DESCRIPTION:   Optional pipe-local segment concentration storage.
******************************************************************************/

#ifndef MSXSEGMENT_STORAGE_H
#define MSXSEGMENT_STORAGE_H

#include <stddef.h>
#include "msxtypes.h"
#include "msxresident_core.h"

#ifdef __cplusplus
extern "C" {
#endif

int  MSXsegStorage_open(void);
void MSXsegStorage_close(void);
void MSXsegStorage_reset(void);

int  MSXsegStorage_preparePrivate(Pseg seg);
void MSXsegStorage_initPrivateValues(Pseg seg, const double c[]);
int  MSXsegStorage_bindPipeSegment(int k, Pseg seg);
void MSXsegStorage_unbindSegment(Pseg seg);

int  MSXsegStorage_isPipeRingEnabled(void);
int  MSXsegStorage_isPipeRingLink(int k);
int  MSXsegStorage_isPipeRingSegment(Pseg seg);

int  MSXsegStorage_isHybridEnabled(void);
int  MSXsegStorage_isHybridLink(int k);
int  MSXsegStorage_isHybridCoreSegment(Pseg seg);
/* Identity-only Core membership query for Resident active-row filtering.
   It reads the dense Core view and never follows or changes CPU topology. */
int  MSXsegStorage_isHybridCoreIdentity(int k, uint64_t hybridId);
/* O(1), fail-closed validation of a Resident active row. ``slot`` is the
   Resident row slot, not an assumed dense-Core slot. */
int  MSXsegStorage_isHybridCoreSlotIdentity(int k, int slot,
                                            uint64_t hybridId);
/* Explicit compatibility/debug snapshot helper.  Normal Resident reaction
   keeps Core concentrations device-owned and does not call this backfill. */
int  MSXsegStorage_hybridApplyResidentPayload(int k, int residentSlot,
                                              uint64_t hybridId,
                                              const MSXResidentPayload *payload);
void MSXsegStorage_hybridAssignIdentity(int k, Pseg seg);
MSXResidentStatus MSXsegStorage_residentLastStatus(void);
int  MSXsegStorage_hybridizeAll(void);
/* Reserve every dense Core row before hybridization.  Once reserved, Hybrid
   promotion is fixed-capacity and never allocates in the quality loop. */
int  MSXsegStorage_hybridReserve(const MSXResidentLayout *layout);
/* Resident startup transaction. Prepare records the prospective CPU Core
   without touching FirstSeg/LastSeg/prev/next/nsegs. Commit is a preflighted,
   allocation-free topology replacement. */
int  MSXsegStorage_hybridPrepareInitialImage(void);
int  MSXsegStorage_hybridStageInitialImage(void);
int  MSXsegStorage_hybridCommitInitialImage(void);
void MSXsegStorage_hybridAbortInitialImage(void);
/* Observer-only rescan used when a resident runtime opens after hybridizeAll. */
int  MSXsegStorage_hybridObserveAll(void);
int  MSXsegStorage_hybridRemoveHead(int k, Pseg seg);
/* Resident Rebalance captures one per-link FirstSeg -> prev topology image
   at round start.  Demote planning/commit and ordinary promote consume that
   image; the read-only audit seam below remains an independent reference
   rescan and is not part of the normal path. */
void MSXsegStorage_hybridRebalanceAll(void);
/* A0 Rebalance detail hooks.  They are inert unless the detail demote group
   is active, so runtime fetch/flush callers need no profile-mode branches. */
enum {
    MSX_REBALANCE_PHASE_PREFLUSH = 0,
    MSX_REBALANCE_PHASE_FETCH = 1
};
int MSXsegStorage_hybridRebalanceProfileActive(void);
void MSXsegStorage_hybridRebalanceAddPhase(int phase, double ms);
void MSXsegStorage_hybridRebalanceAddPatchCounts(uint64_t descriptors,
                                                 uint64_t rows);
/* Read-only audit seam for A1/A2.  ``rows`` are emitted in the authoritative
   CPU linked-list order (FirstSeg -> prev), and no Resident, mapping, or
   topology state is modified. */
typedef struct
{
    uint64_t parcel_id;
    int core;
    int core_slot;
    int resident_slot;
    uint32_t generation;
    uint64_t pipe_epoch;
    Pseg segment;
} MSXHybridAuditRow;
typedef struct
{
    int link_index;
    int total;
    int core_count;
    int downstream_boundary;
    int upstream_boundary;
    int first_core_slot;
    int last_core_slot;
    int orient;
    uint64_t first_core_id;
    uint64_t last_core_id;
} MSXHybridAuditSnapshot;
int MSXsegStorage_hybridAuditSnapshot(int k,
                                      MSXHybridAuditSnapshot *snapshot,
                                      MSXHybridAuditRow *rows,
                                      uint32_t row_capacity,
                                      uint32_t *row_count);
#if defined(MSX_RESIDENT_SCAN_OMP_TEST)
/* Test-only entry point for the production scan workshare.  It exposes no
   normal runtime configuration surface and is compiled only by the real
   OpenMP audit target. */
int MSXsegStorage_testResidentScanOMP(int fullSerial, int *teamSize,
                                      int *workerIds, int workerCapacity,
                                      MSXHybridAuditSnapshot *snapshots,
                                      int snapshotCapacity);
/* Test-only seam for the production scan -> publish -> empty-init ordering.
   It is compiled only by CPU/OpenMP contract harnesses. */
int MSXsegStorage_testResidentScanAndInitialize(int fullSerial, int *teamSize);
/* Test-only seam for the post-scan Resident demote pool failure. */
int MSXsegStorage_testResidentScanAndDemotePoolFailure(
    int fullSerial, int *teamSize, int failureLink);
#endif
/* Make the selected physical endpoint CPU-readable before transport.  This
   may stage a single Resident demotion and returns a mapped error before any
   endpoint consumer can observe a stale Core mirror. */
int  MSXsegStorage_hybridEnsureEndpointBoundary(int k, int boundarySide);
/* Resident Hybrid demote boundaries are reserved at Hybrid reserve time.
   A returned segment remains owned by the pool until the CPU list removes it;
   this prevents the general MSX.FreeSeg allocator from consuming the pool. */
int  MSXsegStorage_hybridAcquireBoundary(Pseg *segment);
int  MSXsegStorage_hybridReleaseBoundary(Pseg segment);
/* Explicit audit seam: poison only dense Hybrid Core c/lastc mirrors. */
int  MSXsegStorage_hybridAuditPoisonCoreMirrors(void);
int MSXsegStorage_hybridAfterListReorder(int k);
void MSXsegStorage_hybridClear(int k);
int  MSXsegStorage_hybridCoreCount(int k);
int  MSXsegStorage_hybridCoreCapacity(int k);
int  MSXsegStorage_hybridCoreSlotAt(int k, int pos);
int  MSXsegStorage_hybridOrientation(int k);
Pseg MSXsegStorage_hybridCoreSegFromHead(int k, int pos);
Pseg MSXsegStorage_hybridCoreSegAt(int k, int pos);
/* Returns one of at most two contiguous downstream-to-upstream Core spans.
   The returned Pseg views reference the authoritative dense slot arrays. */
int  MSXsegStorage_hybridCoreSpan(int k, int spanIndex, Pseg **segs,
                                  int *count);
void MSXsegStorage_hybridPrepareCore(int k);
void MSXsegStorage_hybridCommitCore(int k);
int  MSXsegStorage_hybridTimingEnabled(void);
void MSXsegStorage_hybridTimingAddCoreReact(double ms);
void MSXsegStorage_hybridTimingAddBoundaryReact(double ms);
void MSXsegStorage_hybridTimingAddHandoff(double ms);
void MSXsegStorage_hybridTimingAddPacking(double ms);
void MSXsegStorage_hybridTimingAddH2D(double ms);
void MSXsegStorage_hybridTimingAddKernel(double ms);
void MSXsegStorage_hybridTimingAddD2H(double ms);
void MSXsegStorage_hybridTimingAddUnpack(double ms);
void MSXsegStorage_hybridTimingStepBegin(void);
void MSXsegStorage_hybridTimingStepEnd(void);
void MSXsegStorage_hybridSyncSegmentScalars(Pseg seg);
/* Legacy non-Resident mirror maintenance; Resident normal stepping skips it. */
void MSXsegStorage_hybridSyncAllScalars(void);
/* Opaque prepared topology mapping.  Its fixed per-link arena is allocated
   during Hybrid reserve; preparation only fills and validates it. Commit is
   only the precomputed bounded list/dense stores. */
typedef struct { void *opaque; } MSXHybridResidentMaterialization;
/* Copies a complete Resident-owned row into a new CPU Boundary Pseg.  The
   caller owns the subsequent linked-list insertion/removal transaction. */
int  MSXsegStorage_hybridMaterializeResident(uint32_t linkIndex, uint32_t slot,
                                              uint32_t generation, Pseg *segment);
/* Internal no-fail-after-preflight half of a Resident handoff transaction. */
int  MSXsegStorage_hybridCommitResidentMaterialization(
    const MSXResidentHandoffPlan *plan, const MSXResidentHandoffResult *result,
    Pseg *boundary, uint32_t count);
int  MSXsegStorage_hybridValidateResidentMaterialization(
    const MSXResidentHandoffPlan *plan, const MSXResidentHandoffResult *result,
    Pseg *boundary, uint32_t count);
int  MSXsegStorage_hybridPrepareResidentMaterialization(
    const MSXResidentHandoffPlan *plan, const MSXResidentHandoffResult *result,
    Pseg *boundary, uint32_t count, MSXHybridResidentMaterialization *token);
int  MSXsegStorage_hybridValidatePreparedResidentMaterialization(
    const MSXHybridResidentMaterialization *token);
void MSXsegStorage_hybridCommitPreparedResidentMaterialization(
    MSXHybridResidentMaterialization *token);
void MSXsegStorage_hybridAbortPreparedResidentMaterialization(
    MSXHybridResidentMaterialization *token);

int  MSXsegStorage_pipeAppendTail(int k, Pseg seg);
Pseg MSXsegStorage_pipePeekHead(int k);
Pseg MSXsegStorage_pipePeekTail(int k);
int  MSXsegStorage_pipePopHead(int k);
int  MSXsegStorage_pipeConsumeHead(int k, double volume);
int  MSXsegStorage_pipeMergeTail(int k, const double upnodeQual[], double volume);
int  MSXsegStorage_pipeReverse(int k);
void MSXsegStorage_pipeClear(int k);
int  MSXsegStorage_pipeCount(int k);
Pseg MSXsegStorage_pipeSegFromHead(int k, int pos);
Pseg MSXsegStorage_pipeSegFromTail(int k, int pos);
int  MSXsegStorage_pipeSlotFromHead(int k, int pos);
int  MSXsegStorage_pipeSlotFromTail(int k, int pos);
double *MSXsegStorage_pipeC(int k, int slot);
double *MSXsegStorage_pipeLastC(int k, int slot);
double *MSXsegStorage_pipeVPtr(int k, int slot);
double *MSXsegStorage_pipeHstepPtr(int k, int slot);
double *MSXsegStorage_pipeHresponsePtr(int k, int slot);
double *MSXsegStorage_pipeUresponsePtr(int k, int slot);
double *MSXsegStorage_pipeDresponsePtr(int k, int slot);

void MSXsegStorage_syncPsegMirror(int k);
void MSXsegStorage_syncAllPsegMirrors(void);
void MSXsegStorage_syncScalarsFromPseg(int k);
void MSXsegStorage_syncAllScalarsFromPseg(void);
int  MSXsegStorage_validate(int k);
int  MSXsegStorage_validateAll(void);
int  MSXsegStorage_ringCapacity(void);
int  MSXsegStorage_ringStride(void);
int  MSXsegStorage_ringLinkCount(void);
size_t MSXsegStorage_ringSlotCount(void);
int  MSXsegStorage_pipeFlatIndex(int k, int slot);
double *MSXsegStorage_ringCData(void);
double *MSXsegStorage_ringLastCData(void);
double *MSXsegStorage_ringHstepData(void);

#ifdef __cplusplus
}
#endif

#endif
