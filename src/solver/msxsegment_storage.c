/******************************************************************************
**  MODULE:        MSXSEGMENT_STORAGE.C
**  PROJECT:       EPANET-MSX GPU modified
**  DESCRIPTION:   Optional pipe-local ring queue storage for pipe segments.
**
**  PIPE_RING mode is intentionally isolated from PSEG mode:
**  - PSEG mode leaves the original linked-list storage untouched.
**  - PIPE_RING mode makes each pipe's ring queue the authoritative source for
**    pipe segment order, volume, hstep, response terms, c, and lastc.
**  - Pseg objects are retained as mirror/view records for legacy modules.
******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "msxsegment_storage.h"
#include "msxresident_core.h"
#if defined(EPANETMSX_CUDA_ENABLED)
#include "msxresident_runtime.h"
#endif
#include "epanet2.h"
#include "msxgpu.h"

extern MSXproject MSX;

typedef struct HybridDemoteRequest HybridDemoteRequest;

/* One read-only topology image for a Resident Rebalance round.  The image is
   rebuilt from FirstSeg -> prev once per pipe and then carried through the
   demote plan/commit and ordinary promote phases.  Pseg pointers are valid
   until the corresponding phase commits; endpoint pointers are refreshed from
   the dense view after every successful topology mutation. */
typedef struct RebalanceSnapshot
{
    int linkIndex;
    int total;
    int coreCount;
    int downCount;
    int upCount;
    Pseg firstCore;
    Pseg lastCore;
    uint64_t firstCoreId;
    uint64_t lastCoreId;
    int head;
    int tail;
    int orient;
    int guard;
} RebalanceSnapshot;

typedef struct PipeRingStorage
{
    int opened;
    int cap;
    int nLinks;
    int nSpecies;
    int stride;

    double *c;
    double *lastc;
    double *v;
    double *hstep;
    double *hresponse;
    double *uresponse;
    double *dresponse;
    /* Resident observer staging is owned for the lifetime of the pipe. */
    uint64_t *observerId;
    MSXResidentPayload *observerPayload;

    unsigned char *used;
    Pseg *seg;
    int *head;
    int *tail;
    int *count;
    int *orient;
} PipeRingStorage;

static PipeRingStorage Ring = {0};

typedef struct HybridPipe
{
    int cap;
    int fixedCap;
    /* Fixed Resident links use the CSV's per-link guard.  Non-fixed legacy
       Hybrid links intentionally retain Hybrid.guard (currently two). */
    int guard;
    int count;
    int head;
    int tail;
    int orient;
    int downstreamBoundary;
    int upstreamBoundary;
    unsigned long long nextId;
    unsigned char *used;
    Pseg *view;
    double *c;
    double *lastc;
    double *v;
    double *hstep;
    double *hresponse;
    double *uresponse;
    double *dresponse;
    /* Maps Resident-owned row slots to the independent dense-Core slots.
       Both arrays are fixed at reserve/open time; they avoid assuming the
       two allocators happen to recycle slots in the same order. */
    int *residentSlot;
    int *coreSlotForResident;
    int *mapScratchResident;
    int *mapScratchCore;
    uint64_t *observerId;
    MSXResidentPayload *observerPayload;
    struct HybridResidentTx *residentTx;
    uint32_t initialCount;
    int initialPrepared;
} HybridPipe;

struct HybridDemoteRequest
{
    int linkIndex;
    int atHead;
    int coreSlot;
    uint64_t parcelId;
    Pseg core;
    Pseg boundary;
    MSXResidentHandoffItem item;
    MSXResidentHandoffResult result;
};

/* One token arena per link.  A quality step prepares no more than one
   materialization transaction for a link, while all links may be prepared
   concurrently as a batch. */
typedef struct HybridResidentTx
{
    HybridPipe *pipe;
    uint32_t linkIndex, count, capacity, boundarySide;
    int oldHead, oldTail, oldCount, nextEndpoint, active;
    Pseg *oldCore, *boundary, *down, *up;
    int *slot;
} HybridResidentTx;

typedef struct HybridStorage
{
    int opened;
    int nLinks;
    int nSpecies;
    int stride;
    int guard;
    int timingEnabled;
    HybridPipe *pipe;
    double totalMs;
    double coreReactMs;
    double boundaryReactMs;
    double handoffMs;
    double packingMs;
    double h2dMs;
    double kernelMs;
    double d2hMs;
    double unpackMs;
    double stepStartMs;
    /* Post-transport demote workspaces are reserved with the Hybrid layout.
       The quality loop only reuses these packed arrays; it never grows or
       frees them per rebalance. */
    HybridDemoteRequest *demoteRequest;
    MSXResidentHandoffResult *demoteResult;
    MSXResidentHandoffItem *demoteItem;
    double *demoteC;
    double *demoteLastC;
    uint32_t demoteCapacity;
    /* Dedicated CPU boundary arena.  Pseg objects in this pool are never
       handed to the general FreeSeg list while Hybrid is resident. */
    Pseg *demoteBoundary;
    unsigned char *demoteBoundaryState; /* 0=free, 1=leased, 2=in CPU list */
    uint32_t *demoteBoundaryNext;
    uint32_t demoteBoundaryFreeHead;
    uint32_t demoteBoundaryCapacity;
    RebalanceSnapshot *rebalanceSnapshot;
} HybridStorage;

static HybridStorage Hybrid = {0};
static MSXResidentStatus HybridResidentStatus = MSX_RESIDENT_OK;

enum {
    HYBRID_REBALANCE_NONE = 0,
    HYBRID_REBALANCE_SCAN,
    HYBRID_REBALANCE_EMPTY_INIT,
    HYBRID_REBALANCE_PLAN,
    HYBRID_REBALANCE_VALIDATE_COMMIT,
    HYBRID_REBALANCE_PROMOTE
};
static MSXRebalanceMetrics *HybridRebalanceMetrics = NULL;
static int HybridRebalanceStage = HYBRID_REBALANCE_NONE;
/* S00 audit-only seam.  It is disabled unless explicitly requested and is
   never part of a formal off run.  The disabled path pays one cached
   environment parse and a branch at the Resident rebalance entry. */
static int HybridResidentScanAuditState = -1;
static int HybridResidentScanAuditWritten = 0;

static int hybridResidentScanAuditEnabled(void)
{
    if (HybridResidentScanAuditState < 0)
    {
        const char *value = getenv("MSX_RESIDENT_SCAN_AUDIT");
        HybridResidentScanAuditState =
            value && (strcmp(value, "1") == 0 ||
                      _stricmp(value, "YES") == 0 ||
                      _stricmp(value, "ON") == 0);
    }
    return HybridResidentScanAuditState;
}

static void hybridResidentScanAuditRecord(int teamSize)
{
    FILE *f;
    if (HybridResidentScanAuditWritten || !hybridResidentScanAuditEnabled())
        return;
    f = fopen("resident_scan_team_size.txt", "wt");
    if (!f) return;
    fprintf(f, "phase=resident_rebalance_production_probe\n");
    fprintf(f, "team_size=%d\n", teamSize);
#if defined(_OPENMP)
    fprintf(f, "omp_max_threads=%d\n", omp_get_max_threads());
    fprintf(f, "omp_dynamic=%d\n", omp_get_dynamic());
    fprintf(f, "omp_nested=%d\n", omp_get_nested());
#else
    fprintf(f, "omp_max_threads=not_compiled\n");
    fprintf(f, "omp_dynamic=not_compiled\n");
    fprintf(f, "omp_nested=not_compiled\n");
#endif
    fclose(f);
    HybridResidentScanAuditWritten = 1;
}

static void hybridRebalanceAddPhaseInternal(int phase, double ms)
{
    if (!HybridRebalanceMetrics || ms < 0.0) return;
    switch (phase)
    {
    case MSX_REBALANCE_PHASE_PREFLUSH:
        HybridRebalanceMetrics->rb_preflush_ms += ms; break;
    case MSX_REBALANCE_PHASE_FETCH:
        HybridRebalanceMetrics->rb_fetch_ms += ms; break;
    default: break;
    }
}

static int hybridRebalanceDetailActive(void)
{
    return HybridRebalanceMetrics != NULL;
}

int MSXsegStorage_hybridRebalanceProfileActive(void)
{
    return hybridRebalanceDetailActive();
}

void MSXsegStorage_hybridRebalanceAddPhase(int phase, double ms)
{
    hybridRebalanceAddPhaseInternal(phase, ms);
}

void MSXsegStorage_hybridRebalanceAddPatchCounts(uint64_t descriptors,
                                                 uint64_t rows)
{
    if (!HybridRebalanceMetrics) return;
    HybridRebalanceMetrics->patch_descriptors += descriptors;
    HybridRebalanceMetrics->patch_rows += rows;
}

static int hybridCountSegmentsForRebalance(int k)
{
    Pseg seg;
    int total = 0;
    for (seg = MSX.FirstSeg[k]; seg; seg = seg->prev)
    {
        ++total;
        if (HybridRebalanceMetrics)
        {
            if (seg->inHybridCore) ++HybridRebalanceMetrics->core_visits;
            else ++HybridRebalanceMetrics->boundary_visits;
        }
    }
    if (HybridRebalanceMetrics) ++HybridRebalanceMetrics->scan_passes;
    return total;
}

static void hybridRebalanceAddPlan(double ms)
{
    if (HybridRebalanceMetrics && ms >= 0.0)
        HybridRebalanceMetrics->rb_plan_ms += ms;
}

static void hybridRebalanceAddValidateCommit(double ms)
{
    if (HybridRebalanceMetrics && ms >= 0.0)
        HybridRebalanceMetrics->rb_validate_commit_ms += ms;
}

static void hybridRebalanceFinish(MSXRebalanceMetrics *metrics,
                                  double parentStart)
{
    double children;
    if (!metrics) return;
    metrics->rb_parent_ms = MSXgpu_wallTimeMs() - parentStart;
    children = metrics->rb_scan_ms + metrics->rb_empty_init_ms +
               metrics->rb_plan_ms + metrics->rb_preflush_ms +
               metrics->rb_fetch_ms + metrics->rb_validate_commit_ms +
               metrics->rb_promote_ms;
    metrics->rb_residual_ms = metrics->rb_parent_ms - children;
    if (metrics->rb_parent_ms > 0.0 &&
        metrics->rb_residual_ms < -0.01 * metrics->rb_parent_ms)
        metrics->residual_negative_over_1pct = 1;
    MSXgpu_profileRecordRebalance(metrics);
}

static int hybridPipeGuard(const HybridPipe *p)
{
    return (p && p->fixedCap) ? p->guard : Hybrid.guard;
}
static int hybridNextSlot(const HybridPipe *p, int slot);
static void hybridReplaceInList(int k, Pseg oldseg, Pseg newseg);
static void hybridCopyBoundaryToSlot(int k, int slot, Pseg src);
static void hybridMarkBoundaryCommitted(Pseg seg);
extern void MSXqual_removeSeg(Pseg seg);
extern Pseg MSXqual_getFreeSeg(double v, double c[]);

/* Rebuild the fixed inverse mapping before publishing an observation.
   Resident retains rows for surviving IDs and assigns new IDs the lowest
   free row.  This bookkeeping never changes CPU topology. */
static int hybridRefreshResidentSlots(HybridPipe *p)
{
    int slot, pos, resident;
    if (!p || p->cap <= 0 || !p->residentSlot || !p->coreSlotForResident ||
        p->count < 0 || p->count > p->cap) return 0;
    for (resident = 0; resident < p->cap; ++resident)
        p->coreSlotForResident[resident] = -1;
    slot = p->head;
    for (pos = 0; pos < p->count; ++pos)
    {
        resident = p->residentSlot[slot];
        if (resident < 0 || resident >= p->cap ||
            p->coreSlotForResident[resident] >= 0)
            p->residentSlot[slot] = -1;
        else
            p->coreSlotForResident[resident] = slot;
        slot = hybridNextSlot(p, slot);
    }
    slot = p->head;
    for (pos = 0; pos < p->count; ++pos)
    {
        if (p->residentSlot[slot] < 0)
        {
            for (resident = 0; resident < p->cap; ++resident)
                if (p->coreSlotForResident[resident] < 0) break;
            if (resident == p->cap) return 0;
            p->residentSlot[slot] = resident;
            p->coreSlotForResident[resident] = slot;
        }
        slot = hybridNextSlot(p, slot);
    }
    return 1;
}

static int hybridResidentInsert(int k, int coreSlot, int atHead, Pseg source)
{
    HybridPipe *p = &Hybrid.pipe[k]; Pseg seg = source ? source : p->view[coreSlot];
    MSXResidentPayload x; uint32_t residentSlot = 0, generation = 0;
    MSXResidentStatus z;
    if (!MSXresident_isOpen() || MSXresident_mode() == MSX_RESIDENT_OFF) return 0;
    if (!seg) return ERR_PIPE_RING_CAPACITY;
    memset(&x, 0, sizeof(x)); x.volume = seg->v; x.hstep = seg->hstep;
    x.hresponse = seg->hresponse; x.uresponse = seg->uresponse;
    x.dresponse = seg->dresponse; x.parcelId = seg->hybridId;
    x.c = seg->c; x.lastc = seg->lastc;
    z = MSXresident_stageInsert((uint32_t)k, atHead ? 0 : 1, &x,
                                &residentSlot, &generation);
    if (z != MSX_RESIDENT_OK) { HybridResidentStatus = z; return ERR_PIPE_RING_CAPACITY; }
    p->residentSlot[coreSlot] = (int)residentSlot;
    p->coreSlotForResident[residentSlot] = coreSlot;
    (void)generation;
    return 0;
}

static int hybridResidentRemove(int k, int coreSlot)
{
    HybridPipe *p = &Hybrid.pipe[k]; int residentSlot; uint32_t generation;
    MSXResidentStatus z;
    if (!MSXresident_isOpen() || MSXresident_mode() == MSX_RESIDENT_OFF) return 0;
    if (coreSlot < 0 || coreSlot >= p->cap || !p->residentSlot) return ERR_PIPE_RING_CAPACITY;
    residentSlot = p->residentSlot[coreSlot];
    if (residentSlot < 0 || MSXresident_getSlotGeneration((uint32_t)k,
                                                            (uint32_t)residentSlot,
                                                            &generation) != MSX_RESIDENT_OK)
        return ERR_PIPE_RING_CAPACITY;
    z = MSXresident_stageRemove((uint32_t)k, (uint32_t)residentSlot,
                                (uint32_t)generation);
    if (z != MSX_RESIDENT_OK) { HybridResidentStatus = z; return ERR_PIPE_RING_CAPACITY; }
    p->residentSlot[coreSlot] = -1;
    p->coreSlotForResident[residentSlot] = -1;
    return 0;
}

static int hybridResidentReverse(int k)
{
    MSXResidentStatus z;
    if (!MSXresident_isOpen() || MSXresident_mode() == MSX_RESIDENT_OFF) return 0;
    z = MSXresident_stageReverse((uint32_t)k);
    if (z != MSX_RESIDENT_OK) { HybridResidentStatus = z; return ERR_PIPE_RING_CAPACITY; }
    return 0;
}

/* MSXqual_reversesegs() has already flipped the linked list when this hook
   runs.  Reversing it again is a bounded, non-fallible rollback used only if
   the Resident metadata commit rejects the orientation change. */
static void hybridRollbackListReverse(int k)
{
    Pseg seg, next, previous = NULL;
    if (k <= 0 || k > MSX.Nobjects[LINK]) return;
    seg = MSX.FirstSeg[k];
    MSX.FirstSeg[k] = MSX.LastSeg[k];
    MSX.LastSeg[k] = seg;
    while (seg)
    {
        next = seg->prev;
        seg->prev = previous;
        seg->next = next;
        previous = seg;
        seg = next;
    }
}

static int hybridSyncResidentMap(HybridPipe *p, int k)
{
    int coreSlot, residentSlot, remaining; uint32_t resident, generation; Pseg seg;
    if (!p || !p->residentSlot || !p->coreSlotForResident ||
        !p->mapScratchResident || !p->mapScratchCore || p->count < 0 ||
        p->count > p->cap) return 0;
    for (coreSlot = 0; coreSlot < p->cap; ++coreSlot)
        p->mapScratchResident[coreSlot] = -1;
    for (residentSlot = 0; residentSlot < p->cap; ++residentSlot)
        p->mapScratchCore[residentSlot] = -1;
    coreSlot = p->head; remaining = p->count;
    while (remaining > 0 && coreSlot >= 0)
    {
        seg = p->view[coreSlot];
        if (!seg || MSXresident_getSlotForParcel((uint32_t)k, seg->hybridId,
                                                  &resident,
                                                  &generation) != MSX_RESIDENT_OK)
            return 0;
        residentSlot = (int)resident;
        if (residentSlot < 0 || residentSlot >= p->cap ||
            p->mapScratchCore[residentSlot] >= 0)
            return 0;
        p->mapScratchResident[coreSlot] = residentSlot;
        p->mapScratchCore[residentSlot] = coreSlot;
        coreSlot = hybridNextSlot(p, coreSlot);
        --remaining;
    }
    if (remaining != 0) return 0;
    memcpy(p->residentSlot, p->mapScratchResident,
           (size_t)p->cap * sizeof(*p->residentSlot));
    memcpy(p->coreSlotForResident, p->mapScratchCore,
           (size_t)p->cap * sizeof(*p->coreSlotForResident));
    /* The observer is explicit/debug-only; restore the authoritative count. */
    return 1;
}

/* Observer only: MSX.FirstSeg/LastSeg remain the sole topology authority. */
static MSXResidentStatus hybridResidentObserve(int k)
{
    HybridPipe *p;
    MSXResidentPayload *payload;
    uint64_t *id;
    int pos, slot;
    if (!MSXresident_isOpen() || MSXresident_mode() == MSX_RESIDENT_OFF ||
        !MSXsegStorage_isHybridLink(k)) return MSX_RESIDENT_OK;
    p = &Hybrid.pipe[k];
    if (p->count <= 0)
    {
        HybridResidentStatus = MSXresident_observePipe((uint32_t)k, NULL,
                                                       NULL, 0, p->orient);
        if (HybridResidentStatus == MSX_RESIDENT_OK)
        {
            for (slot = 0; slot < p->cap; ++slot)
            {
                p->residentSlot[slot] = -1;
                p->coreSlotForResident[slot] = -1;
            }
        }
        return HybridResidentStatus;
    }
    if (p->count > p->cap || !p->observerPayload || !p->observerId)
        return MSX_RESIDENT_ERR_MEMORY;
    payload = p->observerPayload;
    id = p->observerId;
    slot = p->head;
    for (pos = 0; pos < p->count; pos++)
    {
        Pseg seg = p->view[slot];
        id[pos] = seg->hybridId;
        payload[pos].volume = seg->v;
        payload[pos].hstep = seg->hstep;
        payload[pos].hresponse = seg->hresponse;
        payload[pos].uresponse = seg->uresponse;
        payload[pos].dresponse = seg->dresponse;
        payload[pos].parcelId = seg->hybridId;
        payload[pos].c = seg->c;
        payload[pos].lastc = seg->lastc;
        slot = hybridNextSlot(p, slot);
    }
    HybridResidentStatus = MSXresident_observePipe((uint32_t)k, id, payload,
                                                    (uint32_t)p->count, p->orient);
    /* The Resident descriptor is published as a deterministic contiguous
       cyclic span (head 0).  Commit the matching CPU Core<->Resident inverse
       map only after that publication succeeds, so active-row filtering and
       payload writeback address the same identities as the GPU image. */
    if (HybridResidentStatus == MSX_RESIDENT_OK)
    {
        if (!hybridSyncResidentMap(p, k))
            HybridResidentStatus = MSX_RESIDENT_ERR_GENERATION;
    }
    return HybridResidentStatus;
}

static void hybridFreePipe(HybridPipe *p)
{
    int i;
    if (!p) return;
    if (p->view)
    {
        for (i = 0; i < p->cap; i++) FREE(p->view[i]);
    }
    FREE(p->view);
    FREE(p->c);
    FREE(p->lastc);
    FREE(p->v);
    FREE(p->hstep);
    FREE(p->hresponse);
    FREE(p->uresponse);
    FREE(p->dresponse);
    FREE(p->residentSlot);
    FREE(p->coreSlotForResident);
    FREE(p->mapScratchResident);
    FREE(p->mapScratchCore);
    FREE(p->observerId);
    FREE(p->observerPayload);
    if (p->residentTx)
    {
        FREE(p->residentTx->oldCore); FREE(p->residentTx->boundary);
        FREE(p->residentTx->down); FREE(p->residentTx->up);
        FREE(p->residentTx->slot); FREE(p->residentTx);
    }
    FREE(p->used);
    memset(p, 0, sizeof(*p));
    p->head = -1;
    p->tail = -1;
}

static void hybridClear(void)
{
    int k;
    uint32_t i;
    if (Hybrid.pipe)
        for (k = 1; k <= Hybrid.nLinks; k++) hybridFreePipe(&Hybrid.pipe[k]);
    /* Leased boundaries were never linked into CPU topology.  Return both
       free and leased pool objects through the normal segment release hook;
       committed objects remain in the caller-owned CPU list and are released
       there after Hybrid is closed. */
    if (Hybrid.demoteBoundary)
        for (i = 0; i < Hybrid.demoteBoundaryCapacity; ++i)
            if (Hybrid.demoteBoundary[i])
            {
                if (Hybrid.demoteBoundaryState[i] == 1)
                    (void)MSXsegStorage_hybridReleaseBoundary(
                        Hybrid.demoteBoundary[i]);
                if (Hybrid.demoteBoundaryState[i] == 0)
                    MSXqual_removeSeg(Hybrid.demoteBoundary[i]);
            }
    FREE(Hybrid.demoteBoundary);
    FREE(Hybrid.demoteBoundaryState);
    FREE(Hybrid.demoteBoundaryNext);
    FREE(Hybrid.demoteRequest);
    FREE(Hybrid.demoteResult);
    FREE(Hybrid.demoteItem);
    FREE(Hybrid.demoteC);
    FREE(Hybrid.demoteLastC);
    FREE(Hybrid.rebalanceSnapshot);
    FREE(Hybrid.pipe);
    memset(&Hybrid, 0, sizeof(Hybrid));
}

static void hybridWriteTiming(void)
{
    FILE *f;
    double transfer;
    double excluded;
    if (!Hybrid.opened) return;
    f = fopen("msx_hybrid_timing.csv", "wt");
    if (!f) return;
    transfer = Hybrid.packingMs + Hybrid.h2dMs + Hybrid.d2hMs + Hybrid.unpackMs;
    excluded = Hybrid.totalMs - transfer;
    if (excluded < 0.0) excluded = 0.0;
    fprintf(f, "component,total_ms,transfer_excluded_ms\n");
    fprintf(f, "total,%.9f,%.9f\n", Hybrid.totalMs, excluded);
    fprintf(f, "boundary_react,%.9f,%.9f\n", Hybrid.boundaryReactMs, Hybrid.boundaryReactMs);
    fprintf(f, "core_react,%.9f,%.9f\n", Hybrid.coreReactMs, Hybrid.coreReactMs);
    fprintf(f, "core_handoff,%.9f,%.9f\n", Hybrid.handoffMs, Hybrid.handoffMs);
    fprintf(f, "packing,%.9f,0.000000000\n", Hybrid.packingMs);
    fprintf(f, "h2d,%.9f,0.000000000\n", Hybrid.h2dMs);
    fprintf(f, "kernel,%.9f,%.9f\n", Hybrid.kernelMs, Hybrid.kernelMs);
    fprintf(f, "d2h,%.9f,0.000000000\n", Hybrid.d2hMs);
    fprintf(f, "unpack,%.9f,0.000000000\n", Hybrid.unpackMs);
    fclose(f);
    f = fopen("msx_hybrid_timing_mode.txt", "wt");
    if (f)
    {
        fprintf(f, "enabled=%d\n", Hybrid.timingEnabled ? 1 : 0);
        fprintf(f, "clock_scope=region_and_quality_step\n");
        fprintf(f, "per_segment_clock_reads=0\n");
        fclose(f);
    }
}

static int hybridAllocPipe(HybridPipe *p, int cap)
{
    int i;
    size_t slots, rows;
    if (!p || cap <= 0) return ERR_MEMORY;
    slots = (size_t)cap;
    rows = slots * (size_t)Hybrid.stride;
    p->cap = cap;
    p->head = -1;
    p->tail = -1;
    p->orient = 1;
    p->used = (unsigned char *)calloc(slots, sizeof(unsigned char));
    p->view = (Pseg *)calloc(slots, sizeof(Pseg));
    p->c = (double *)calloc(rows, sizeof(double));
    p->lastc = (double *)calloc(rows, sizeof(double));
    p->v = (double *)calloc(slots, sizeof(double));
    p->hstep = (double *)calloc(slots, sizeof(double));
    p->hresponse = (double *)calloc(slots, sizeof(double));
    p->uresponse = (double *)calloc(slots, sizeof(double));
    p->dresponse = (double *)calloc(slots, sizeof(double));
    p->residentSlot = (int *)malloc(slots * sizeof(int));
    p->coreSlotForResident = (int *)malloc(slots * sizeof(int));
    p->mapScratchResident = (int *)malloc(slots * sizeof(int));
    p->mapScratchCore = (int *)malloc(slots * sizeof(int));
    p->observerId = (uint64_t *)calloc(slots, sizeof(uint64_t));
    p->observerPayload = (MSXResidentPayload *)calloc(slots, sizeof(MSXResidentPayload));
    p->residentTx = (HybridResidentTx *)calloc(1, sizeof(HybridResidentTx));
    if (!p->used || !p->view || !p->c || !p->lastc || !p->v ||
        !p->hstep || !p->hresponse || !p->uresponse || !p->dresponse ||
        !p->residentSlot || !p->coreSlotForResident ||
        !p->mapScratchResident || !p->mapScratchCore ||
        !p->observerId || !p->observerPayload || !p->residentTx)
    {
        hybridFreePipe(p);
        return ERR_MEMORY;
    }
    p->residentTx->oldCore = (Pseg *)calloc(slots, sizeof(Pseg));
    p->residentTx->boundary = (Pseg *)calloc(slots, sizeof(Pseg));
    p->residentTx->down = (Pseg *)calloc(slots, sizeof(Pseg));
    p->residentTx->up = (Pseg *)calloc(slots, sizeof(Pseg));
    p->residentTx->slot = (int *)calloc(slots, sizeof(int));
    p->residentTx->capacity = (uint32_t)cap;
    if (!p->residentTx->oldCore || !p->residentTx->boundary ||
        !p->residentTx->down || !p->residentTx->up || !p->residentTx->slot)
    {
        hybridFreePipe(p);
        return ERR_MEMORY;
    }
    for (i = 0; i < cap; i++)
    {
        p->residentSlot[i] = -1;
        p->coreSlotForResident[i] = -1;
        p->view[i] = (Pseg)calloc(1, sizeof(struct Sseg));
        if (!p->view[i])
        {
            hybridFreePipe(p);
            return ERR_MEMORY;
        }
        p->view[i]->hybridSlot = i;
        p->view[i]->ringSlot = -1;
        p->view[i]->ringIndex = -1;
        p->view[i]->inHybridCore = FALSE;
        p->view[i]->ownerLink = 0;
        p->view[i]->c = p->c + (size_t)i * (size_t)Hybrid.stride;
        p->view[i]->lastc = p->lastc + (size_t)i * (size_t)Hybrid.stride;
    }
    return 0;
}

static int hybridOpen(void)
{
    int k, cap;
    const char *timing;
    memset(&Hybrid, 0, sizeof(Hybrid));
    Hybrid.nLinks = MSX.Nobjects[LINK];
    Hybrid.nSpecies = MSX.Nobjects[SPECIES];
    Hybrid.stride = Hybrid.nSpecies + 1;
    Hybrid.guard = 2;
    timing = getenv("MSX_HYBRID_TIMING");
    Hybrid.timingEnabled = !(timing &&
        (strcmp(timing, "0") == 0 || _stricmp(timing, "OFF") == 0 ||
         _stricmp(timing, "NO") == 0));
    Hybrid.pipe = (HybridPipe *)calloc((size_t)Hybrid.nLinks + 1, sizeof(HybridPipe));
    if (!Hybrid.pipe) return ERR_MEMORY;
    Hybrid.rebalanceSnapshot = (RebalanceSnapshot *)calloc(
        (size_t)Hybrid.nLinks + 1, sizeof(*Hybrid.rebalanceSnapshot));
    if (!Hybrid.rebalanceSnapshot)
    {
        FREE(Hybrid.pipe);
        memset(&Hybrid, 0, sizeof(Hybrid));
        return ERR_MEMORY;
    }
    cap = MAX(16, MIN(MAX(16, MSX.MaxSegments), 128));
    for (k = 1; k <= Hybrid.nLinks; k++)
    {
        Hybrid.pipe[k].head = -1;
        Hybrid.pipe[k].tail = -1;
        Hybrid.pipe[k].orient = 1;
        /* Allocate lazily per link.  A link receives storage at first promotion. */
        (void)cap;
    }
    Hybrid.opened = TRUE;
    return 0;
}

static int hybridEnsureCapacity(int k, int need)
{
    HybridPipe *p;
    int oldcap, cap, i, pos, oldslot;
    size_t newrow;
    double *newc, *newlastc, *newv, *newhstep;
    double *newhresponse, *newuresponse, *newdresponse;
    double *oldc, *oldlastc, *oldv, *oldhstep;
    double *oldhresponse, *olduresponse, *olddresponse;
    Pseg *oldview;
    unsigned char *oldused;
    int *oldResidentSlot, *oldCoreSlotForResident;
    int *oldMapScratchResident, *oldMapScratchCore;
    Pseg *newview;
    unsigned char *newused;
    int *newResidentSlot, *newCoreSlotForResident;
    int *newMapScratchResident, *newMapScratchCore;
    uint64_t *newObserverId;
    MSXResidentPayload *newObserverPayload;

    if (!Hybrid.opened || k <= 0 || k > Hybrid.nLinks) return ERR_MEMORY;
    p = &Hybrid.pipe[k];
    if (p->cap >= need) return 0;
    if (p->fixedCap) return ERR_PIPE_RING_CAPACITY;
    oldcap = p->cap;
    cap = oldcap > 0 ? oldcap : 16;
    while (cap < need)
    {
        if (cap >= MSX.MaxSegments) { cap = MSX.MaxSegments; break; }
        cap = MIN(MSX.MaxSegments, cap * 2);
    }
    if (cap < need) return ERR_PIPE_RING_CAPACITY;
    if (oldcap == 0) return hybridAllocPipe(p, cap);

    newrow = (size_t)cap * (size_t)Hybrid.stride;
    newc = (double *)calloc(newrow, sizeof(double));
    newlastc = (double *)calloc(newrow, sizeof(double));
    newv = (double *)calloc((size_t)cap, sizeof(double));
    newhstep = (double *)calloc((size_t)cap, sizeof(double));
    newhresponse = (double *)calloc((size_t)cap, sizeof(double));
    newuresponse = (double *)calloc((size_t)cap, sizeof(double));
    newdresponse = (double *)calloc((size_t)cap, sizeof(double));
    newview = (Pseg *)calloc((size_t)cap, sizeof(Pseg));
    newused = (unsigned char *)calloc((size_t)cap, sizeof(unsigned char));
    newResidentSlot = (int *)malloc((size_t)cap * sizeof(int));
    newCoreSlotForResident = (int *)malloc((size_t)cap * sizeof(int));
    newMapScratchResident = (int *)malloc((size_t)cap * sizeof(int));
    newMapScratchCore = (int *)malloc((size_t)cap * sizeof(int));
    newObserverId = (uint64_t *)calloc((size_t)cap, sizeof(uint64_t));
    newObserverPayload = (MSXResidentPayload *)calloc((size_t)cap, sizeof(MSXResidentPayload));
    if (!newc || !newlastc || !newv || !newhstep || !newhresponse ||
        !newuresponse || !newdresponse || !newview || !newused ||
        !newResidentSlot || !newCoreSlotForResident ||
        !newMapScratchResident || !newMapScratchCore ||
        !newObserverId || !newObserverPayload)
    {
        FREE(newc); FREE(newlastc); FREE(newv); FREE(newhstep);
        FREE(newhresponse); FREE(newuresponse); FREE(newdresponse);
        FREE(newview); FREE(newused); FREE(newResidentSlot); FREE(newCoreSlotForResident);
        FREE(newMapScratchResident); FREE(newMapScratchCore);
        FREE(newObserverId); FREE(newObserverPayload);
        return ERR_MEMORY;
    }
    oldc = p->c; oldlastc = p->lastc; oldv = p->v; oldhstep = p->hstep;
    oldhresponse = p->hresponse; olduresponse = p->uresponse;
    olddresponse = p->dresponse; oldview = p->view; oldused = p->used;
    oldResidentSlot = p->residentSlot;
    oldCoreSlotForResident = p->coreSlotForResident;
    oldMapScratchResident = p->mapScratchResident;
    oldMapScratchCore = p->mapScratchCore;
    for (i = 0; i < cap; ++i) newResidentSlot[i] = newCoreSlotForResident[i] = -1;
    for (i = 0; i < cap; ++i) newMapScratchResident[i] = newMapScratchCore[i] = -1;
    oldslot = p->count > 0 ? p->head : -1;
    for (pos = 0; pos < p->count; pos++)
    {
        Pseg seg = oldview[oldslot];
        size_t oldrow = (size_t)oldslot * (size_t)Hybrid.stride;
        size_t newrowpos = (size_t)pos * (size_t)Hybrid.stride;
        memcpy(newc + newrowpos, oldc + oldrow, (size_t)Hybrid.stride * sizeof(double));
        memcpy(newlastc + newrowpos, oldlastc + oldrow,
               (size_t)Hybrid.stride * sizeof(double));
        newv[pos] = oldview[oldslot]->v;
        newhstep[pos] = oldview[oldslot]->hstep;
        newhresponse[pos] = oldview[oldslot]->hresponse;
        newuresponse[pos] = oldview[oldslot]->uresponse;
        newdresponse[pos] = oldview[oldslot]->dresponse;
        newview[pos] = seg;
        newused[pos] = TRUE;
        newResidentSlot[pos] = oldResidentSlot ? oldResidentSlot[oldslot] : -1;
        if (newResidentSlot[pos] >= 0 && newResidentSlot[pos] < cap)
            newCoreSlotForResident[newResidentSlot[pos]] = pos;
        oldslot += p->orient;
        if (oldslot >= oldcap) oldslot = 0;
        if (oldslot < 0) oldslot = oldcap - 1;
    }
    /* Allocate every replacement view before changing the live pipe.  A
       failed allocation must leave its old rows, views and topology intact. */
    for (i = 0; i < cap; i++)
    {
        if (!newview[i]) newview[i] = (Pseg)calloc(1, sizeof(struct Sseg));
        if (!newview[i])
        {
            int j;
            for (j = 0; j < cap; j++)
            {
                int q, isOld = FALSE;
                for (q = 0; q < oldcap; q++)
                    if (newview[j] == oldview[q]) { isOld = TRUE; break; }
                if (newview[j] && !isOld) FREE(newview[j]);
            }
            FREE(newc); FREE(newlastc); FREE(newv); FREE(newhstep);
            FREE(newhresponse); FREE(newuresponse); FREE(newdresponse);
            FREE(newview); FREE(newused); FREE(newResidentSlot); FREE(newCoreSlotForResident);
            FREE(newMapScratchResident); FREE(newMapScratchCore);
            FREE(newObserverId); FREE(newObserverPayload);
            return ERR_MEMORY;
        }
    }
    for (i = 0; i < oldcap; i++)
        if (!oldused[i]) FREE(oldview[i]);
    FREE(oldc); FREE(oldlastc); FREE(oldv); FREE(oldhstep);
    FREE(oldhresponse); FREE(olduresponse); FREE(olddresponse);
    FREE(oldview); FREE(oldused);
    FREE(oldResidentSlot); FREE(oldCoreSlotForResident);
    FREE(oldMapScratchResident); FREE(oldMapScratchCore);
    FREE(p->observerId); FREE(p->observerPayload);
    p->c = newc; p->lastc = newlastc; p->v = newv; p->hstep = newhstep;
    p->hresponse = newhresponse; p->uresponse = newuresponse;
    p->dresponse = newdresponse; p->view = newview; p->used = newused;
    p->residentSlot = newResidentSlot;
    p->coreSlotForResident = newCoreSlotForResident;
    p->mapScratchResident = newMapScratchResident;
    p->mapScratchCore = newMapScratchCore;
    p->observerId = newObserverId; p->observerPayload = newObserverPayload;
    p->cap = cap;
    p->orient = 1;
    p->head = p->count > 0 ? 0 : -1;
    p->tail = p->count > 0 ? p->count - 1 : -1;
    for (i = 0; i < cap; i++)
    {
        p->view[i]->hybridSlot = i;
        p->view[i]->ringSlot = -1;
        p->view[i]->ringIndex = -1;
        p->view[i]->c = p->c + (size_t)i * (size_t)Hybrid.stride;
        p->view[i]->lastc = p->lastc + (size_t)i * (size_t)Hybrid.stride;
        if (p->used[i])
        {
            p->v[i] = p->view[i]->v;
            p->hstep[i] = p->view[i]->hstep;
            p->hresponse[i] = p->view[i]->hresponse;
            p->uresponse[i] = p->view[i]->uresponse;
            p->dresponse[i] = p->view[i]->dresponse;
        }
    }
    return 0;
}

int MSXsegStorage_hybridReserve(const MSXResidentLayout *layout)
{
    int k, err;
    if (!MSXsegStorage_isHybridEnabled() || !Hybrid.opened) return 0;
    if (!layout || layout->nLinks != (uint32_t)Hybrid.nLinks || !layout->capacity ||
        !layout->guard)
        return ERR_PIPE_RING_CAPACITY;
    /* A Resident row can become a CPU boundary through either handoff or
       post-transport demote.  The exact total row count is the bounded
       simultaneous demand; keeping one pool slot per row also makes a
       handoff followed by demote allocation-free. */
    uint32_t boundaryCapacity = layout->totalSlots;
    if (Hybrid.demoteCapacity && Hybrid.demoteCapacity != layout->totalSlots)
        return ERR_PIPE_RING_CAPACITY;
    if (Hybrid.demoteBoundaryCapacity &&
        Hybrid.demoteBoundaryCapacity != boundaryCapacity)
        return ERR_PIPE_RING_CAPACITY;
    if (!Hybrid.demoteCapacity)
    {
        size_t rows;
        if (layout->totalSlots == 0 ||
            layout->totalSlots > SIZE_MAX / sizeof(*Hybrid.demoteRequest) ||
            layout->totalSlots > SIZE_MAX / sizeof(*Hybrid.demoteResult) ||
            layout->totalSlots > SIZE_MAX / sizeof(*Hybrid.demoteItem) ||
            boundaryCapacity > SIZE_MAX / sizeof(*Hybrid.demoteBoundary) ||
            boundaryCapacity > SIZE_MAX / sizeof(*Hybrid.demoteBoundaryState) ||
            boundaryCapacity > SIZE_MAX / sizeof(*Hybrid.demoteBoundaryNext) ||
            (size_t)layout->totalSlots > SIZE_MAX / (size_t)Hybrid.stride)
            return ERR_MEMORY;
        rows = (size_t)layout->totalSlots * (size_t)Hybrid.stride;
        if (rows > SIZE_MAX / sizeof(*Hybrid.demoteC)) return ERR_MEMORY;
        Hybrid.demoteRequest = (HybridDemoteRequest *)calloc(
            layout->totalSlots, sizeof(*Hybrid.demoteRequest));
        Hybrid.demoteResult = (MSXResidentHandoffResult *)calloc(
            layout->totalSlots, sizeof(*Hybrid.demoteResult));
        Hybrid.demoteItem = (MSXResidentHandoffItem *)calloc(
            layout->totalSlots, sizeof(*Hybrid.demoteItem));
        Hybrid.demoteC = (double *)calloc(rows, sizeof(*Hybrid.demoteC));
        Hybrid.demoteLastC = (double *)calloc(rows, sizeof(*Hybrid.demoteLastC));
        Hybrid.demoteBoundary = (Pseg *)calloc(
            boundaryCapacity, sizeof(*Hybrid.demoteBoundary));
        Hybrid.demoteBoundaryState = (unsigned char *)calloc(
            boundaryCapacity, sizeof(*Hybrid.demoteBoundaryState));
        Hybrid.demoteBoundaryNext = (uint32_t *)malloc(
            (size_t)boundaryCapacity * sizeof(*Hybrid.demoteBoundaryNext));
        if (!Hybrid.demoteRequest || !Hybrid.demoteResult ||
            !Hybrid.demoteItem || !Hybrid.demoteC || !Hybrid.demoteLastC ||
            !Hybrid.demoteBoundary || !Hybrid.demoteBoundaryState ||
            !Hybrid.demoteBoundaryNext)
        {
            FREE(Hybrid.demoteRequest);
            FREE(Hybrid.demoteResult);
            FREE(Hybrid.demoteItem);
            FREE(Hybrid.demoteC);
            FREE(Hybrid.demoteLastC);
            FREE(Hybrid.demoteBoundary);
            FREE(Hybrid.demoteBoundaryState);
            FREE(Hybrid.demoteBoundaryNext);
            Hybrid.demoteRequest = NULL;
            Hybrid.demoteResult = NULL;
            Hybrid.demoteItem = NULL;
            Hybrid.demoteC = NULL;
            Hybrid.demoteLastC = NULL;
            Hybrid.demoteBoundary = NULL;
            Hybrid.demoteBoundaryState = NULL;
            Hybrid.demoteBoundaryNext = NULL;
            return ERR_MEMORY;
        }
        for (uint32_t i = 0; i < boundaryCapacity; ++i)
        {
            Hybrid.demoteBoundaryNext[i] =
                i + 1U < boundaryCapacity ? i + 1U : UINT32_MAX;
            Hybrid.demoteBoundary[i] = MSXqual_getFreeSeg(0.0, MSX.C1);
            if (!Hybrid.demoteBoundary[i])
            {
                while (i > 0)
                {
                    --i;
                    MSXqual_removeSeg(Hybrid.demoteBoundary[i]);
                    Hybrid.demoteBoundary[i] = NULL;
                }
                FREE(Hybrid.demoteRequest);
                FREE(Hybrid.demoteResult);
                FREE(Hybrid.demoteItem);
                FREE(Hybrid.demoteC);
                FREE(Hybrid.demoteLastC);
                FREE(Hybrid.demoteBoundary);
                FREE(Hybrid.demoteBoundaryState);
                FREE(Hybrid.demoteBoundaryNext);
                Hybrid.demoteRequest = NULL;
                Hybrid.demoteResult = NULL;
                Hybrid.demoteItem = NULL;
                Hybrid.demoteC = NULL;
                Hybrid.demoteLastC = NULL;
                Hybrid.demoteBoundary = NULL;
                Hybrid.demoteBoundaryState = NULL;
                Hybrid.demoteBoundaryNext = NULL;
                return ERR_MEMORY;
            }
            Hybrid.demoteBoundary[i]->hybridBoundaryPoolIndex = (int)i;
        }
        Hybrid.demoteBoundaryFreeHead = boundaryCapacity ? 0U : UINT32_MAX;
        Hybrid.demoteCapacity = layout->totalSlots;
        Hybrid.demoteBoundaryCapacity = boundaryCapacity;
    }
    for (k = 1; k <= Hybrid.nLinks; k++)
    {
        HybridPipe *p = &Hybrid.pipe[k];
        int cap = (int)layout->capacity[k];
        int guard = (int)layout->guard[k];
        if (cap <= 0 || (guard != 2 && guard != 4) || p->count > 0 ||
            (p->cap > 0 && p->cap != cap))
            return ERR_PIPE_RING_CAPACITY;
        /* Resident capacity is a contract, not a growth hint.  Reserve the
           exact CSV capacity before hybridization so preflight and GPU row
           ownership use the same bound. */
        err = p->cap == 0 ? hybridAllocPipe(p, cap) : 0;
        if (err) return err;
        p->fixedCap = TRUE;
        p->guard = guard;
    }
    return 0;
}

/* Build the startup Core image from Boundary Psegs without replacing a single
   list pointer.  residentTx->oldCore is a fixed arena reserved with the dense
   rows, so this prepare phase cannot grow storage. */
int MSXsegStorage_hybridPrepareInitialImage(void)
{
    int k;
    if (!MSXsegStorage_isHybridEnabled() || !Hybrid.opened) return 0;
    for (k = 1; k <= Hybrid.nLinks; ++k)
    {
        HybridPipe *p = &Hybrid.pipe[k]; Pseg seg; int total = 0, pos;
        for (seg = MSX.FirstSeg[k]; seg; seg = seg->prev) ++total;
        int guard = hybridPipeGuard(p);
        if (!p->fixedCap || p->count || !p->residentTx ||
            total > 2 * guard + p->cap) goto bad;
        p->initialCount = total > 2 * guard ?
                          (uint32_t)(total - 2 * guard) : 0;
        if (p->initialCount > (uint32_t)p->cap) goto bad;
        seg = MSX.FirstSeg[k];
        for (pos = 0; pos < guard && seg; ++pos) seg = seg->prev;
        for (pos = 0; pos < (int)p->initialCount; ++pos)
        {
            if (!seg || !seg->hybridId || !seg->c || !seg->lastc) goto bad;
            p->residentTx->oldCore[pos] = seg;
            p->observerId[pos] = seg->hybridId;
            p->observerPayload[pos].volume = seg->v;
            p->observerPayload[pos].hstep = seg->hstep;
            p->observerPayload[pos].hresponse = seg->hresponse;
            p->observerPayload[pos].uresponse = seg->uresponse;
            p->observerPayload[pos].dresponse = seg->dresponse;
            p->observerPayload[pos].parcelId = seg->hybridId;
            p->observerPayload[pos].c = seg->c;
            p->observerPayload[pos].lastc = seg->lastc;
            seg = seg->prev;
        }
        p->initialPrepared = TRUE;
    }
    return 0;
bad:
    MSXsegStorage_hybridAbortInitialImage();
    return ERR_PIPE_RING_CAPACITY;
}

/* Publish only to Resident's private metadata image; CPU topology remains
   entirely untouched until the runtime has completed every GPU fallible step. */
int MSXsegStorage_hybridStageInitialImage(void)
{
    int k;
    if (!MSXsegStorage_isHybridEnabled() || !Hybrid.opened) return 0;
    for (k = 1; k <= Hybrid.nLinks; ++k)
    {
        HybridPipe *p = &Hybrid.pipe[k];
        if (!p->initialPrepared) return ERR_PIPE_RING_CAPACITY;
        HybridResidentStatus = MSXresident_stageInitialPipe((uint32_t)k,
            p->initialCount ? p->observerId : NULL,
            p->initialCount ? p->observerPayload : NULL,
            p->initialCount, p->orient);
        if (HybridResidentStatus != MSX_RESIDENT_OK) return ERR_PIPE_RING_CAPACITY;
    }
    return 0;
}

/* Required precondition: prepare, metadata staging, initial GPU upload and
   program setup all succeeded.  Every target row and source Pseg was checked
   in prepare; this is deliberately only bounded stores/list rewiring/free. */
int MSXsegStorage_hybridCommitInitialImage(void)
{
    int k, i;
    if (!MSXsegStorage_isHybridEnabled() || !Hybrid.opened) return 0;
    /* Validate the entire batch before the first visible list mutation. */
    for (k = 1; k <= Hybrid.nLinks; ++k)
    {
        HybridPipe *p = &Hybrid.pipe[k]; Pseg seg = MSX.FirstSeg[k];
        if (!p->initialPrepared || p->count || p->initialCount > (uint32_t)p->cap)
            return ERR_PIPE_RING_CAPACITY;
        for (i = 0; i < hybridPipeGuard(p) && seg; ++i) seg = seg->prev;
        for (i = 0; i < (int)p->initialCount; ++i)
        {
            if (!seg || seg != p->residentTx->oldCore[i] || !seg->c ||
                !seg->lastc || !p->view[i] || p->used[i])
                return ERR_PIPE_RING_CAPACITY;
            seg = seg->prev;
        }
    }
    /* No checks, allocation, or fallible calls are permitted below here. */
    for (k = 1; k <= Hybrid.nLinks; ++k)
    {
        HybridPipe *p = &Hybrid.pipe[k];
        for (i = 0; i < (int)p->initialCount; ++i)
        {
            Pseg boundary = p->residentTx->oldCore[i];
            Pseg core = p->view[i];
            hybridCopyBoundaryToSlot(k, i, boundary);
            p->used[i] = TRUE;
            p->residentSlot[i] = i;
            p->coreSlotForResident[i] = i;
            p->count++;
            if (p->count == 1) p->head = p->tail = i;
            else p->tail = i;
            hybridReplaceInList(k, boundary, core);
            MSXqual_removeSeg(boundary);
        }
        p->initialCount = 0;
        p->initialPrepared = FALSE;
    }
    return 0;
}

void MSXsegStorage_hybridAbortInitialImage(void)
{
    int k;
    if (!Hybrid.opened) return;
    for (k = 1; k <= Hybrid.nLinks; ++k)
    {
        HybridPipe *p = &Hybrid.pipe[k];
        p->initialCount = 0;
        p->initialPrepared = FALSE;
    }
}

extern Pseg MSXqual_getFreeSeg(double v, double c[]);
extern void MSXqual_removeSeg(Pseg seg);

static int hybridNextSlot(const HybridPipe *p, int slot)
{
    int s = slot + p->orient;
    if (s >= p->cap) s = 0;
    if (s < 0) s = p->cap - 1;
    return s;
}

static int hybridPrevSlot(const HybridPipe *p, int slot)
{
    int s = slot - p->orient;
    if (s >= p->cap) s = 0;
    if (s < 0) s = p->cap - 1;
    return s;
}

static void hybridCopyBoundaryToSlot(int k, int slot, Pseg src)
{
    HybridPipe *p = &Hybrid.pipe[k];
    Pseg dst = p->view[slot];
    int m;
    size_t row = (size_t)slot * (size_t)Hybrid.stride;
    for (m = 1; m <= MSX.Nobjects[SPECIES]; m++)
    {
        p->c[row + m] = src->c[m];
        p->lastc[row + m] = src->lastc[m];
    }
    p->v[slot] = src->v;
    p->hstep[slot] = src->hstep;
    p->hresponse[slot] = src->hresponse;
    p->uresponse[slot] = src->uresponse;
    p->dresponse[slot] = src->dresponse;
    dst->ownerLink = k;
    dst->hybridSlot = slot;
    dst->hybridId = src->hybridId;
    dst->inHybridCore = TRUE;
    dst->inPipeRing = FALSE;
    dst->c = p->c + row;
    dst->lastc = p->lastc + row;
    dst->v = src->v;
    dst->hstep = src->hstep;
    dst->hresponse = src->hresponse;
    dst->uresponse = src->uresponse;
    dst->dresponse = src->dresponse;
}

static void hybridCopySlotToBoundary(int k, int slot, Pseg dst)
{
    HybridPipe *p = &Hybrid.pipe[k];
    Pseg src = p->view[slot];
    int m;
    size_t row = (size_t)slot * (size_t)Hybrid.stride;
    for (m = 1; m <= MSX.Nobjects[SPECIES]; m++)
    {
        dst->c[m] = p->c[row + m];
        dst->lastc[m] = p->lastc[row + m];
    }
    dst->v = src->v;
    dst->hstep = src->hstep;
    dst->hresponse = src->hresponse;
    dst->uresponse = src->uresponse;
    dst->dresponse = src->dresponse;
    dst->hybridId = src->hybridId;
}

int MSXsegStorage_hybridMaterializeResident(uint32_t linkIndex, uint32_t slot,
                                             uint32_t generation, Pseg *segment)
{
    MSXResidentPayload payload;
    Pseg boundary;
    int boundaryStatus;
    int m;
    if (!segment) return ERR_MEMORY;
    *segment = NULL;
    if (MSXresident_getSlotPayload(linkIndex, slot, generation, &payload) !=
        MSX_RESIDENT_OK) return ERR_PIPE_RING_CAPACITY;
    /* Resident handoff must consume a pre-reserved pool slot.  In particular,
       do not fall through to MSX.FreeSeg/Alloc: that would make handoff
       success depend on an unrelated general segment allocator. */
    boundaryStatus = MSXsegStorage_hybridAcquireBoundary(&boundary);
    if (boundaryStatus != 0) return boundaryStatus;
    boundary->v = payload.volume;
    boundary->hstep = payload.hstep;
    boundary->hresponse = payload.hresponse;
    boundary->uresponse = payload.uresponse;
    boundary->dresponse = payload.dresponse;
    boundary->hybridId = payload.parcelId;
    for (m = 1; m <= MSX.Nobjects[SPECIES]; m++)
    {
        boundary->c[m] = payload.c[m];
        boundary->lastc[m] = payload.lastc[m];
    }
    *segment = boundary;
    return 0;
}

/* Validate the exact dense-Core interval before any list pointer or Resident
   metadata changes.  This is deliberately separate from commit so the latter
   contains only bounded pointer/array stores. */
int MSXsegStorage_hybridValidateResidentMaterialization(const MSXResidentHandoffPlan *plan,
    const MSXResidentHandoffResult *result, Pseg *boundary, uint32_t count)
{
    HybridPipe *p;
    uint32_t i;
    int expected;
    if (!plan || !result || !boundary || !count || plan->boundarySide > 1 ||
        !MSXsegStorage_isHybridLink((int)plan->linkIndex)) return 0;
    p = &Hybrid.pipe[plan->linkIndex];
    if ((int)count > p->count) return 0;
    expected = plan->boundarySide ? p->tail : p->head;
    for (i = 0; i < count; i++)
    {
        MSXResidentPayload resident;
        Pseg core;
        if (!boundary[i] || result[i].linkIndex != plan->linkIndex ||
            result[i].boundarySide != plan->boundarySide ||
            result[i].pipeEpoch != plan->pipeEpoch || expected < 0 ||
            expected >= p->cap || !p->used[expected]) return 0;
        core = p->view[expected];
        if (!core || !core->inHybridCore ||
            MSXresident_getSlotPayload(result[i].linkIndex, result[i].slot,
                result[i].generation, &resident) != MSX_RESIDENT_OK ||
            resident.parcelId != core->hybridId ||
            result[i].payload.parcelId != resident.parcelId) return 0;
        expected = plan->boundarySide ? hybridPrevSlot(p, expected) :
                                        hybridNextSlot(p, expected);
    }
    return 1;
}

static int hybridResidentTxContains(const HybridResidentTx *t, Pseg s)
{ uint32_t i; for (i = 0; i < t->count; ++i) if (t->oldCore[i] == s) return 1; return 0; }

void MSXsegStorage_hybridAbortPreparedResidentMaterialization(
    MSXHybridResidentMaterialization *token)
{
    HybridResidentTx *t;
    uint32_t i;
    if (!token || !(t = (HybridResidentTx *)token->opaque)) return;
    /* A prepared boundary is still detached from CPU topology.  Return it to
       the fixed pool directly; calling MSXqual_removeSeg after this would
       expose the private pool object to the general FreeSeg list. */
    for (i = 0; i < t->count; ++i)
        if (t->boundary[i])
        {
            if (!MSXsegStorage_hybridReleaseBoundary(t->boundary[i]))
                MSXqual_removeSeg(t->boundary[i]);
            t->boundary[i] = NULL;
        }
    t->active = FALSE;
    t->count = 0;
    token->opaque = NULL;
}

int MSXsegStorage_hybridPrepareResidentMaterialization(
    const MSXResidentHandoffPlan *plan, const MSXResidentHandoffResult *result,
    Pseg *boundary, uint32_t count, MSXHybridResidentMaterialization *token)
{
    HybridResidentTx *t; uint32_t i; int slot;
    if (!token || token->opaque ||
        !MSXsegStorage_hybridValidateResidentMaterialization(plan, result, boundary, count)) return ERR_PIPE_RING_CAPACITY;
    t = Hybrid.pipe[plan->linkIndex].residentTx;
    if (!t || t->active || count > t->capacity) return ERR_PIPE_RING_CAPACITY;
    t->pipe = &Hybrid.pipe[plan->linkIndex]; t->linkIndex = plan->linkIndex;
    t->count = count; t->boundarySide = plan->boundarySide;
    t->active = TRUE;
    t->oldHead = t->pipe->head; t->oldTail = t->pipe->tail; t->oldCount = t->pipe->count;
    slot = plan->boundarySide ? t->oldTail : t->oldHead;
    for (i = 0; i < count; ++i)
    {
        t->slot[i] = slot; t->oldCore[i] = t->pipe->view[slot]; t->boundary[i] = boundary[i];
        t->down[i] = t->oldCore[i]->next; t->up[i] = t->oldCore[i]->prev;
        slot = plan->boundarySide ? hybridPrevSlot(t->pipe, slot) : hybridNextSlot(t->pipe, slot);
    }
    t->nextEndpoint = slot; token->opaque = t; return 0;
}

int MSXsegStorage_hybridValidatePreparedResidentMaterialization(
    const MSXHybridResidentMaterialization *token)
{
    HybridResidentTx *t; uint32_t i;
    if (!token || !(t = (HybridResidentTx *)token->opaque) || !t->pipe ||
        t->pipe->count != t->oldCount || t->pipe->head != t->oldHead || t->pipe->tail != t->oldTail) return 0;
    for (i = 0; i < t->count; ++i)
        if (!t->boundary[i] || t->slot[i] < 0 || t->slot[i] >= t->pipe->cap ||
            !t->pipe->used[t->slot[i]] || t->pipe->view[t->slot[i]] != t->oldCore[i] ||
            !t->oldCore[i]->inHybridCore || t->oldCore[i]->next != t->down[i] ||
            t->oldCore[i]->prev != t->up[i]) return 0;
    return 1;
}

void MSXsegStorage_hybridCommitPreparedResidentMaterialization(
    MSXHybridResidentMaterialization *token)
{
    HybridResidentTx *t = token ? (HybridResidentTx *)token->opaque : NULL;
    uint32_t i; Pseg bottom, top, down, up;
    if (!t) return; /* final validation is a required precondition. */
    bottom = t->boundarySide ? t->boundary[t->count - 1] : t->boundary[0];
    top = t->boundarySide ? t->boundary[0] : t->boundary[t->count - 1];
    down = t->boundarySide ? t->down[t->count - 1] : t->down[0];
    up = t->boundarySide ? t->up[0] : t->up[t->count - 1];
    for (i = 0; i < t->count; ++i)
    {
        Pseg b = t->boundary[i];
        b->next = t->boundarySide ? (i + 1 < t->count ? t->boundary[i + 1] : t->down[i])
                                  : (i ? t->boundary[i - 1] : t->down[i]);
        b->prev = t->boundarySide ? (i ? t->boundary[i - 1] : t->up[i])
                                  : (i + 1 < t->count ? t->boundary[i + 1] : t->up[i]);
        /* The boundary now belongs to the CPU list.  Keep it marked
           committed so a later MSXqual_removeSeg returns it to this pool. */
        hybridMarkBoundaryCommitted(b);
        t->oldCore[i]->next = NULL; t->oldCore[i]->prev = NULL;
        t->oldCore[i]->inHybridCore = FALSE; t->oldCore[i]->ownerLink = 0;
        t->oldCore[i]->hybridId = 0;
        /* Handoff commit removes the resident row as well as the CPU Core
           object.  Keep both directions of the slot map authoritative so a
           later promote/demote cannot address a stale resident slot. */
        if (t->slot[i] >= 0 && t->slot[i] < t->pipe->cap)
        {
            int residentSlot = t->pipe->residentSlot[t->slot[i]];
            if (residentSlot >= 0 && residentSlot < t->pipe->cap)
                t->pipe->coreSlotForResident[residentSlot] = -1;
            t->pipe->residentSlot[t->slot[i]] = -1;
            t->pipe->used[t->slot[i]] = FALSE;
        }
    }
    if (down) down->prev = bottom; else MSX.FirstSeg[t->linkIndex] = bottom;
    if (up) up->next = top; else MSX.LastSeg[t->linkIndex] = top;
    t->pipe->count -= (int)t->count;
    if (!t->pipe->count) t->pipe->head = t->pipe->tail = -1;
    else if (!t->boundarySide) t->pipe->head = t->nextEndpoint;
    else t->pipe->tail = t->nextEndpoint;
    for (i = 0; i < t->count; ++i) t->boundary[i] = NULL;
    MSXsegStorage_hybridAbortPreparedResidentMaterialization(token);
}

int MSXsegStorage_hybridCommitResidentMaterialization(
    const MSXResidentHandoffPlan *plan, const MSXResidentHandoffResult *result,
    Pseg *boundary, uint32_t count)
{
    MSXHybridResidentMaterialization t = {0}; int z = MSXsegStorage_hybridPrepareResidentMaterialization(plan, result, boundary, count, &t);
    if (z) return z;
    if (!MSXsegStorage_hybridValidatePreparedResidentMaterialization(&t)) { MSXsegStorage_hybridAbortPreparedResidentMaterialization(&t); return ERR_PIPE_RING_CAPACITY; }
    MSXsegStorage_hybridCommitPreparedResidentMaterialization(&t); return 0;
}

static void hybridReplaceInList(int k, Pseg oldseg, Pseg newseg)
{
    Pseg down, up;
    if (!oldseg || !newseg) return;
    down = oldseg->next;
    up = oldseg->prev;
    newseg->next = down;
    newseg->prev = up;
    if (down) down->prev = newseg;
    else MSX.FirstSeg[k] = newseg;
    if (up) up->next = newseg;
    else MSX.LastSeg[k] = newseg;
    oldseg->next = NULL;
    oldseg->prev = NULL;
}

static void hybridFindCore(HybridPipe *p, int k, Pseg *firstCore,
                           Pseg *lastCore, int *down, int *up, int *count)
{
    Pseg seg;
    int nDown = 0, nUp = 0, nCore = 0, seenCore = FALSE;
    *firstCore = NULL;
    *lastCore = NULL;
    for (seg = MSX.FirstSeg[k]; seg; seg = seg->prev)
    {
        if (seg->inHybridCore)
        {
            if (!*firstCore) *firstCore = seg;
            *lastCore = seg;
            nCore++;
            seenCore = TRUE;
        }
        else if (!seenCore) nDown++;
        else nUp++;
    }
    p->count = nCore;
    p->downstreamBoundary = nDown;
    p->upstreamBoundary = nUp;
    p->head = *firstCore ? (*firstCore)->hybridSlot : -1;
    p->tail = *lastCore ? (*lastCore)->hybridSlot : -1;
    if (down) *down = nDown;
    if (up) *up = nUp;
    if (count) *count = nCore;
    if (HybridRebalanceMetrics)
    {
        ++HybridRebalanceMetrics->scan_passes;
        HybridRebalanceMetrics->core_visits += (uint64_t)nCore;
        HybridRebalanceMetrics->boundary_visits +=
            (uint64_t)nDown + (uint64_t)nUp;
    }
}

/* Capture the authoritative CPU topology once for a Resident Rebalance
   round.  This deliberately does not call hybridFindCore: the latter remains
   the legacy/non-Resident and audit rescan seam. */
static void hybridCaptureRebalanceSnapshot(int k, RebalanceSnapshot *s)
{
    HybridPipe *p = &Hybrid.pipe[k];
    Pseg seg;
    int nDown = 0, nUp = 0, nCore = 0, total = 0, seenCore = FALSE;

    memset(s, 0, sizeof(*s));
    s->linkIndex = k;
    s->head = -1;
    s->tail = -1;
    s->orient = p->orient;
    s->guard = hybridPipeGuard(p);
    for (seg = MSX.FirstSeg[k]; seg; seg = seg->prev)
    {
        ++total;
        if (seg->inHybridCore)
        {
            if (!s->firstCore) s->firstCore = seg;
            s->lastCore = seg;
            ++nCore;
            seenCore = TRUE;
        }
        else if (!seenCore) ++nDown;
        else ++nUp;
    }
    s->total = total;
    s->coreCount = nCore;
    /* With no Core, all segments are the canonical downstream boundary. */
    s->downCount = nCore ? nDown : total;
    s->upCount = nCore ? nUp : 0;
    if (s->firstCore)
    {
        s->head = s->firstCore->hybridSlot;
        s->tail = s->lastCore->hybridSlot;
        s->firstCoreId = s->firstCore->hybridId;
        s->lastCoreId = s->lastCore->hybridId;
    }
    p->count = s->coreCount;
    p->downstreamBoundary = s->downCount;
    p->upstreamBoundary = s->upCount;
    p->head = s->head;
    p->tail = s->tail;
    if (HybridRebalanceMetrics)
    {
        ++HybridRebalanceMetrics->scan_passes;
        HybridRebalanceMetrics->core_visits += (uint64_t)s->coreCount;
        HybridRebalanceMetrics->boundary_visits +=
            (uint64_t)s->downCount + (uint64_t)s->upCount;
    }
}

static void hybridSnapshotRefreshEndpoints(RebalanceSnapshot *s,
                                            HybridPipe *p)
{
    s->head = p->head;
    s->tail = p->tail;
    s->coreCount = p->count;
    if (s->coreCount <= 0)
    {
        s->coreCount = 0;
        s->firstCore = NULL;
        s->lastCore = NULL;
        s->firstCoreId = 0;
        s->lastCoreId = 0;
        s->head = -1;
        s->tail = -1;
        s->downCount = s->total;
        s->upCount = 0;
    }
    else
    {
        s->firstCore = p->view[p->head];
        s->lastCore = p->view[p->tail];
        s->firstCoreId = s->firstCore ? s->firstCore->hybridId : 0;
        s->lastCoreId = s->lastCore ? s->lastCore->hybridId : 0;
    }
    p->count = s->coreCount;
    p->downstreamBoundary = s->downCount;
    p->upstreamBoundary = s->upCount;
}

static void hybridSnapshotApplyDemote(RebalanceSnapshot *s,
                                      HybridPipe *p, int atHead)
{
    if (s->coreCount > 0) --s->coreCount;
    if (atHead) ++s->downCount;
    else ++s->upCount;
    hybridSnapshotRefreshEndpoints(s, p);
}

static void hybridSnapshotApplyPromote(RebalanceSnapshot *s,
                                       HybridPipe *p, int atHead)
{
    ++s->coreCount;
    if (atHead)
    {
        if (s->downCount > 0) --s->downCount;
    }
    else if (s->upCount > 0) --s->upCount;
    hybridSnapshotRefreshEndpoints(s, p);
}

int MSXsegStorage_hybridAuditSnapshot(int k,
                                      MSXHybridAuditSnapshot *snapshot,
                                      MSXHybridAuditRow *rows,
                                      uint32_t row_capacity,
                                      uint32_t *row_count)
{
    HybridPipe *p;
    Pseg seg;
    int total = 0, core = 0, down = 0, up = 0, seenCore = FALSE;
    int firstCore = -1, lastCore = -1;
    uint32_t pos = 0;

    if (!snapshot || !row_count || k <= 0 || k > Hybrid.nLinks ||
        !Hybrid.pipe || !MSX.FirstSeg)
        return ERR_PIPE_RING_CAPACITY;
    p = &Hybrid.pipe[k];
    if (!MSXsegStorage_isHybridLink(k)) return ERR_PIPE_RING_CAPACITY;
    for (seg = MSX.FirstSeg[k]; seg; seg = seg->prev)
    {
        ++total;
        if (seg->inHybridCore)
        {
            if (!seenCore) seenCore = TRUE;
            ++core;
        }
        else if (!seenCore) ++down;
        else ++up;
    }
    if (total < 0 || (rows && (uint32_t)total > row_capacity) ||
        p->cap < 0 || (p->count < 0 || p->count > p->cap))
    {
        *row_count = (uint32_t)(total < 0 ? 0 : total);
        return ERR_PIPE_RING_CAPACITY;
    }
    memset(snapshot, 0, sizeof(*snapshot));
    snapshot->link_index = k;
    snapshot->total = total;
    snapshot->core_count = core;
    snapshot->downstream_boundary = core ? down : total;
    snapshot->upstream_boundary = core ? up : 0;
    snapshot->orient = p->orient;
    *row_count = (uint32_t)total;
    seenCore = FALSE;
    for (seg = MSX.FirstSeg[k]; seg; seg = seg->prev)
    {
        MSXHybridAuditRow *row = rows ? &rows[pos] : NULL;
        int coreSlot = -1, residentSlot = -1;
        uint32_t generation = 0;
        uint64_t epoch = 0;
        if (seg->inHybridCore)
        {
            coreSlot = seg->hybridSlot;
            if (coreSlot < 0 || coreSlot >= p->cap || !p->used ||
                !p->view || !p->used[coreSlot] ||
                p->view[coreSlot] != seg || seg->ownerLink != k)
                return ERR_PIPE_RING_CAPACITY;
            if (firstCore < 0) firstCore = coreSlot;
            lastCore = coreSlot;
            seenCore = TRUE;
            if (MSXresident_isOpen())
            {
                uint64_t residentParcelId = 0;
                if (!p->residentSlot || !p->coreSlotForResident)
                    return ERR_PIPE_RING_CAPACITY;
                residentSlot = p->residentSlot[coreSlot];
                if (residentSlot < 0 || residentSlot >= p->cap ||
                    p->coreSlotForResident[residentSlot] != coreSlot)
                    return ERR_PIPE_RING_CAPACITY;
                MSXResidentStatus z = MSXresident_getSlotIdentity(
                    (uint32_t)k, (uint32_t)residentSlot, &generation,
                    &residentParcelId, &epoch);
                if (z != MSX_RESIDENT_OK || residentParcelId != seg->hybridId)
                    return ERR_PIPE_RING_CAPACITY;
            }
        }
        if (row)
        {
            row->parcel_id = seg->hybridId;
            row->core = seg->inHybridCore ? 1 : 0;
            row->core_slot = coreSlot;
            row->resident_slot = residentSlot;
            row->generation = generation;
            row->pipe_epoch = epoch;
            row->segment = seg;
        }
        ++pos;
    }
    snapshot->first_core_slot = firstCore;
    snapshot->last_core_slot = lastCore;
    return 0;
}

static int hybridPromote(int k, Pseg boundary, int atHead)
{
    HybridPipe *p = &Hybrid.pipe[k];
    int slot, err;
    double timer = 0.0;
    int timing = MSXsegStorage_hybridTimingEnabled();
    int detail = MSXgpu_profileDetailGroupEnabled(MSX_PROFILE_DETAIL_DEMOTE);
    if (timing || detail) timer = MSXgpu_wallTimeMs();
    if (!boundary || boundary->inHybridCore) return 0;
    /* Fixed resident capacity is checked before the first mutation. */
    if (p->fixedCap && p->count + 1 > p->cap) return ERR_PIPE_RING_CAPACITY;
    err = hybridEnsureCapacity(k, p->count + 1);
    if (err) return err;
    if (p->count == 0)
        slot = 0;
    else if (atHead)
        slot = hybridPrevSlot(p, p->head);
    else
        slot = hybridNextSlot(p, p->tail);
    if (p->used[slot])
    {
        snprintf(MSX.Msg, MAXLINE,
                 "Error 525 - HYBRID slot collision: link=%d slot=%d head=%d tail=%d cap=%d count=%d orient=%d.",
                 k, slot, p->head, p->tail, p->cap, p->count, p->orient);
        ENwriteline(MSX.Msg);
        return ERR_PIPE_RING_CAPACITY;
    }
    /* Publish the fallible Resident row first.  The remaining CPU writes are
       bounded stores/list rewiring and cannot leave a half-promoted slot. */
    if (hybridResidentInsert(k, slot, atHead, boundary)) return ERR_PIPE_RING_CAPACITY;
    hybridCopyBoundaryToSlot(k, slot, boundary);
    p->used[slot] = TRUE;
    p->count++;
    if (p->count == 1) p->head = p->tail = slot;
    else if (atHead) p->head = slot;
    else p->tail = slot;
    hybridReplaceInList(k, boundary, p->view[slot]);
    MSXqual_removeSeg(boundary);
    if (HybridRebalanceMetrics)
    {
        if (HybridRebalanceStage == HYBRID_REBALANCE_EMPTY_INIT)
            ++HybridRebalanceMetrics->init_promotes;
        else if (HybridRebalanceStage == HYBRID_REBALANCE_PROMOTE)
            ++HybridRebalanceMetrics->promote_rows;
    }
    if (timing) MSXsegStorage_hybridTimingAddHandoff(MSXgpu_wallTimeMs() - timer);
    if (detail) MSXgpu_profileRecordPromote(1, MSXgpu_wallTimeMs() - timer);
    return 0;
}

static int hybridDemote(int k, int atHead)
{
    HybridPipe *p = &Hybrid.pipe[k];
    Pseg core, boundary;
    int slot, boundaryStatus;
    double timer = 0.0;
    int timing = MSXsegStorage_hybridTimingEnabled();
    int detail = MSXgpu_profileDetailGroupEnabled(MSX_PROFILE_DETAIL_DEMOTE);
    if (timing || detail) timer = MSXgpu_wallTimeMs();
    if (p->count <= 0) return 0;
    slot = atHead ? p->head : p->tail;
    core = p->view[slot];
    /* Resident Core concentration is not a CPU source of truth.  The seed is
       overwritten by selected GPU fetch (or by the legacy copy path below). */
    boundaryStatus = MSXsegStorage_hybridAcquireBoundary(&boundary);
    if (boundaryStatus == MSX_RESIDENT_DISABLED)
    {
#if defined(EPANETMSX_CUDA_ENABLED)
        if (MSXresidentRuntime_isResident()) return ERR_PIPE_RING_CAPACITY;
#endif
        boundary = MSXqual_getFreeSeg(0.0, MSX.C1);
        if (!boundary) return ERR_MEMORY;
    }
    else if (boundaryStatus != 0)
        return boundaryStatus;
#if defined(EPANETMSX_CUDA_ENABLED)
    if (MSXresidentRuntime_isResident())
    {
        MSXResidentPayload latest;
        MSXResidentStatus status = MSXresidentRuntime_fetchSlot(
            (uint32_t)k, core->hybridId, boundary->c, boundary->lastc,
            (uint32_t)Hybrid.stride, &latest);
        if (status != MSX_RESIDENT_OK)
        {
            MSXqual_removeSeg(boundary);
            MSX.ErrCode = ERR_GPU_KERNEL_RUNTIME_ERROR;
            return MSX.ErrCode;
        }
        boundary->v = latest.volume;
        boundary->hstep = latest.hstep;
        boundary->hresponse = latest.hresponse;
        boundary->uresponse = latest.uresponse;
        boundary->dresponse = latest.dresponse;
        boundary->hybridId = latest.parcelId;
    }
    else
#endif
        hybridCopySlotToBoundary(k, slot, boundary);
    if (hybridResidentRemove(k, slot))
    {
        MSXqual_removeSeg(boundary);
        return ERR_PIPE_RING_CAPACITY;
    }
    hybridMarkBoundaryCommitted(boundary);
    hybridReplaceInList(k, core, boundary);
    core->inHybridCore = FALSE;
    core->ownerLink = 0;
    core->hybridId = 0;
    p->used[slot] = FALSE;
    if (p->count == 1)
    {
        p->count = 0;
        p->head = p->tail = -1;
    }
    else
    {
        p->count--;
        if (atHead) p->head = hybridNextSlot(p, slot);
        else p->tail = hybridPrevSlot(p, slot);
    }
    if (timing) MSXsegStorage_hybridTimingAddHandoff(MSXgpu_wallTimeMs() - timer);
    if (detail) MSXgpu_profileRecordDemote(1, MSXgpu_wallTimeMs() - timer);
    return 0;
}

static int hybridInitializeLink(int k)
{
    HybridPipe *p = &Hybrid.pipe[k];
    Pseg seg, firstCore, lastCore;
    int total = 0, i, err, down, up, count, guard = hybridPipeGuard(p);
    total = hybridCountSegmentsForRebalance(k);
    if (total <= 2 * guard) return 0;
    firstCore = MSX.FirstSeg[k];
    for (i = 0; i < guard; i++) firstCore = firstCore->prev;
    err = hybridPromote(k, firstCore, TRUE);
    if (err) return err;
    hybridFindCore(p, k, &firstCore, &lastCore, &down, &up, &count);
    while (up > guard)
    {
        seg = lastCore->prev;
        err = hybridPromote(k, seg, FALSE);
        if (err) return err;
        hybridFindCore(p, k, &firstCore, &lastCore, &down, &up, &count);
    }
    return 0;
}

/* Resident-only empty-Core rebuild.  The legacy hybridInitializeLink above
   intentionally keeps its count/find rescans for non-Resident and audit
   callers.  Resident already owns one authoritative RebalanceSnapshot for
   this round, so every successful promote can advance the image directly.
   A failed promote leaves the previously committed prefix in both the CPU
   topology and this snapshot; there is deliberately no rollback here. */
static int hybridInitializeResidentLink(int k, RebalanceSnapshot *s)
{
    HybridPipe *p = &Hybrid.pipe[k];
    Pseg seg;
    int i, err, guard;

    if (!s || s->linkIndex != k || s->coreCount != 0 ||
        s->total <= 0 || s->total <= 2 * s->guard)
        return ERR_PIPE_RING_CAPACITY;
    guard = s->guard;
    seg = MSX.FirstSeg[k];
    for (i = 0; i < guard && seg; ++i) seg = seg->prev;
    if (!seg) return ERR_PIPE_RING_CAPACITY;

    err = hybridPromote(k, seg, TRUE);
    if (err) return err;
    s->coreCount = 1;
    s->downCount = guard;
    s->upCount = s->total - guard - 1;
    hybridSnapshotRefreshEndpoints(s, p);

    while (s->upCount > guard)
    {
        seg = s->lastCore ? s->lastCore->prev : NULL;
        if (!seg || seg->inHybridCore) return ERR_PIPE_RING_CAPACITY;
        err = hybridPromote(k, seg, FALSE);
        if (err) return err;
        ++s->coreCount;
        --s->upCount;
        /* p->tail is the newly published dense-Core endpoint.  Refreshing
           from the view preserves endpoint identity without a list rescan. */
        hybridSnapshotRefreshEndpoints(s, p);
    }
    return 0;
}

static int hybridRebalanceLink(int k)
{
    HybridPipe *p;
    Pseg firstCore, lastCore, seg;
    int down, up, count, total = 0, err, guard;
    if (!MSXsegStorage_isHybridLink(k)) return 0;
    p = &Hybrid.pipe[k];
    guard = hybridPipeGuard(p);
    for (seg = MSX.FirstSeg[k]; seg; seg = seg->prev) total++;
    hybridFindCore(p, k, &firstCore, &lastCore, &down, &up, &count);
    if (count == 0 && total > 2 * guard)
    {
        err = hybridInitializeLink(k);
        if (err) return err;
        hybridFindCore(p, k, &firstCore, &lastCore, &down, &up, &count);
    }
    while (count > 0 && down < guard)
    {
        err = hybridDemote(k, TRUE);
        if (err) return err;
        /* Demote mutates only the selected endpoint.  Carry the boundary
           counts forward instead of rescanning the complete linked list for
           every row.  This is the O(1) endpoint lookup used by S5b. */
        count--;
        down++;
    }
    while (count > 0 && up < guard)
    {
        err = hybridDemote(k, FALSE);
        if (err) return err;
        count--;
        up++;
    }
    if (!count) return 0;
    /* There has been at most one preflight scan for this link.  Refresh the
       endpoints once after the batched demotes; subsequent promotions update
       the endpoint directly from the ring head/tail. */
    hybridFindCore(p, k, &firstCore, &lastCore, &down, &up, &count);
    while (count > 0 && down > guard)
    {
        seg = firstCore->next;
        if (!seg || seg->inHybridCore) break;
        err = hybridPromote(k, seg, TRUE);
        if (err) return err;
        firstCore = p->view[p->head];
        count++;
        down--;
    }
    while (count > 0 && up > guard)
    {
        seg = lastCore->prev;
        if (!seg || seg->inHybridCore) break;
        err = hybridPromote(k, seg, FALSE);
        if (err) return err;
        lastCore = p->view[p->tail];
        count++;
        up--;
    }
    return 0;
}

#if defined(EPANETMSX_CUDA_ENABLED)
/* Once a post-transport demote batch has been fetched and validated, only
   bounded list/mapping stores remain.  Resident removal is prevalidated for
   every request before the first CPU topology mutation. */
static int hybridValidateDemoteRequest(const HybridDemoteRequest *r)
{
    HybridPipe *p;
    int residentSlot;
    uint32_t generation;
    uint64_t parcelId, epoch;
    if (!r || !r->core || !r->boundary || r->linkIndex <= 0 ||
        r->linkIndex > Hybrid.nLinks) return ERR_PIPE_RING_CAPACITY;
    p = &Hybrid.pipe[r->linkIndex];
    if (r->coreSlot < 0 || r->coreSlot >= p->cap ||
        !p->used[r->coreSlot] || p->view[r->coreSlot] != r->core ||
        !p->residentSlot || !p->coreSlotForResident)
        return ERR_PIPE_RING_CAPACITY;
    residentSlot = p->residentSlot[r->coreSlot];
    if (residentSlot < 0 || residentSlot >= p->cap ||
        p->coreSlotForResident[residentSlot] != r->coreSlot)
        return ERR_PIPE_RING_CAPACITY;
    if (MSXresident_getSlotIdentity((uint32_t)r->linkIndex,
                                    (uint32_t)residentSlot, &generation,
                                    &parcelId, &epoch) != MSX_RESIDENT_OK ||
        generation != r->item.generation || parcelId != r->parcelId ||
        epoch != r->item.pipeEpoch)
        return ERR_PIPE_RING_CAPACITY;
    if (r->result.linkIndex != (uint32_t)r->linkIndex ||
        r->result.slot != (uint32_t)residentSlot ||
        r->result.generation != r->item.generation ||
        r->result.pipeEpoch != r->item.pipeEpoch ||
        r->result.payload.parcelId != r->parcelId ||
        !r->result.payload.c || !r->result.payload.lastc)
        return ERR_PIPE_RING_CAPACITY;
    return 0;
}

static int hybridCommitDemoteRequest(HybridDemoteRequest *r)
{
    HybridPipe *p;
    int residentSlot;
    /* Resident removals are committed by one prevalidated batch per link
       before this CPU topology commit.  This function is intentionally only
       bounded pointer/list bookkeeping and has no fallible Resident call. */
    if (!r || !r->core || !r->boundary || r->linkIndex <= 0 ||
        r->linkIndex > Hybrid.nLinks) return ERR_PIPE_RING_CAPACITY;
    p = &Hybrid.pipe[r->linkIndex];
    residentSlot = p->residentSlot[r->coreSlot];
    r->boundary->v = r->result.payload.volume;
    r->boundary->hstep = r->result.payload.hstep;
    r->boundary->hresponse = r->result.payload.hresponse;
    r->boundary->uresponse = r->result.payload.uresponse;
    r->boundary->dresponse = r->result.payload.dresponse;
    memcpy(r->boundary->c, r->result.payload.c,
           (size_t)Hybrid.stride * sizeof(double));
    memcpy(r->boundary->lastc, r->result.payload.lastc,
           (size_t)Hybrid.stride * sizeof(double));
    r->boundary->hybridId = r->result.payload.parcelId;
    hybridMarkBoundaryCommitted(r->boundary);
    p->residentSlot[r->coreSlot] = -1;
    p->coreSlotForResident[residentSlot] = -1;
    hybridReplaceInList(r->linkIndex, r->core, r->boundary);
    r->core->inHybridCore = FALSE;
    r->core->ownerLink = 0;
    r->core->hybridId = 0;
    p->used[r->coreSlot] = FALSE;
    if (p->count == 1)
    {
        p->count = 0;
        p->head = p->tail = -1;
    }
    else
    {
        p->count--;
        if (r->atHead) p->head = hybridNextSlot(p, r->coreSlot);
        else p->tail = hybridPrevSlot(p, r->coreSlot);
    }
    return 0;
}

static void hybridFreeDemoteBatch(HybridDemoteRequest *request,
                                  uint32_t count, double *c, double *lastc,
                                  MSXResidentHandoffResult *result,
                                  MSXResidentHandoffItem *item)
{
    uint32_t i;
    (void)c; (void)lastc; (void)result; (void)item;
    if (request)
        for (i = 0; i < count; i++)
            if (request[i].boundary)
            {
                Pseg boundary = request[i].boundary;
                request[i].boundary = NULL;
                MSXqual_removeSeg(boundary);
            }
}

/* Plan all endpoint demotions without changing CPU topology, fetch every
   selected Resident row in one gather, then validate and commit the full
   batch.  This is deliberately separate from hybridDemote(), which remains
   the small pre-transport endpoint materializer. */
static int hybridDemoteResidentBatch(void)
{
    HybridDemoteRequest *request = Hybrid.demoteRequest;
    MSXResidentHandoffResult *result = Hybrid.demoteResult;
    MSXResidentHandoffItem *item = Hybrid.demoteItem;
    RebalanceSnapshot *snapshot = Hybrid.rebalanceSnapshot;
    double *c = Hybrid.demoteC, *lastc = Hybrid.demoteLastC;
    uint32_t capacity = Hybrid.demoteCapacity;
    uint32_t count = 0, pos = 0, i, begin, end;
    int link;
    MSXResidentStatus status;
    int detail = hybridRebalanceDetailActive();
    int timing = MSXsegStorage_hybridTimingEnabled();
    double timer = (timing || detail) ? MSXgpu_wallTimeMs() : 0.0;
    double planTimer = detail ? timer : 0.0;
    double validateTimer = 0.0;

    if (!snapshot) return ERR_MEMORY;
    for (i = 1; i <= (uint32_t)Hybrid.nLinks; i++)
    {
        RebalanceSnapshot *s = &snapshot[i];
        int downDemote = MIN(s->coreCount,
                             MAX(0, s->guard - s->downCount));
        int upDemote = MIN(s->coreCount - downDemote,
                           MAX(0, s->guard - s->upCount));
        count += (uint32_t)(downDemote + upDemote);
    }
    if (!count)
    {
        if (detail) hybridRebalanceAddPlan(MSXgpu_wallTimeMs() - planTimer);
        return 0;
    }
    if (!request || !result || !item || !c || !lastc || count > capacity)
    {
        if (detail) hybridRebalanceAddPlan(MSXgpu_wallTimeMs() - planTimer);
        return ERR_MEMORY;
    }
    /* Requests are reusable arena records.  Clear only this batch's range;
       cleanup below guarantees no boundary from an older batch remains live. */
    memset(request, 0, (size_t)count * sizeof(*request));

    for (i = 1; i <= (uint32_t)Hybrid.nLinks; i++)
    {
        HybridPipe *p = &Hybrid.pipe[i];
        RebalanceSnapshot *s = &snapshot[i];
        int downDemote, upDemote, remaining, head, tail;
        uint64_t parcelId;
        downDemote = MIN(s->coreCount,
                         MAX(0, s->guard - s->downCount));
        upDemote = MIN(s->coreCount - downDemote,
                       MAX(0, s->guard - s->upCount));
        remaining = downDemote + upDemote;
        head = s->head;
        tail = s->tail;
        while (remaining > 0 && downDemote > 0)
        {
            HybridDemoteRequest *r = &request[pos++];
            r->linkIndex = (int)i; r->atHead = TRUE; r->coreSlot = head;
            if (HybridRebalanceMetrics)
            {
                ++HybridRebalanceMetrics->demote_rows;
                ++HybridRebalanceMetrics->head_rows;
            }
            r->core = p->view[head];
            head = hybridNextSlot(p, head);
            remaining--; downDemote--;
            if (!r->core || !p->residentSlot ||
                p->residentSlot[r->coreSlot] < 0)
            { if (detail) hybridRebalanceAddPlan(MSXgpu_wallTimeMs() - planTimer);
              hybridFreeDemoteBatch(request, pos, c, lastc, result, item); return ERR_PIPE_RING_CAPACITY; }
            r->item.linkIndex = i;
            r->item.slot = (uint32_t)p->residentSlot[r->coreSlot];
            r->item.boundarySide = 0;
            if (HybridRebalanceMetrics) ++HybridRebalanceMetrics->id_lookup_probes;
            if (MSXresident_getSlotIdentity(i, r->item.slot,
                                            &r->item.generation,
                                            &parcelId,
                                            &r->item.pipeEpoch) != MSX_RESIDENT_OK)
            { if (detail) hybridRebalanceAddPlan(MSXgpu_wallTimeMs() - planTimer);
              hybridFreeDemoteBatch(request, pos, c, lastc, result, item); return ERR_PIPE_RING_CAPACITY; }
            r->parcelId = parcelId;
            r->item.requestedVolume = 0.0;
            if (MSXsegStorage_hybridAcquireBoundary(&r->boundary) != 0)
            { if (detail) hybridRebalanceAddPlan(MSXgpu_wallTimeMs() - planTimer);
              hybridFreeDemoteBatch(request, pos, c, lastc, result, item); return ERR_MEMORY; }
        }
        while (remaining > 0 && upDemote > 0)
        {
            HybridDemoteRequest *r = &request[pos++];
            r->linkIndex = (int)i; r->atHead = FALSE; r->coreSlot = tail;
            if (HybridRebalanceMetrics)
            {
                ++HybridRebalanceMetrics->demote_rows;
                ++HybridRebalanceMetrics->tail_rows;
            }
            r->core = p->view[tail];
            tail = hybridPrevSlot(p, tail);
            remaining--; upDemote--;
            if (!r->core || !p->residentSlot ||
                p->residentSlot[r->coreSlot] < 0)
            { if (detail) hybridRebalanceAddPlan(MSXgpu_wallTimeMs() - planTimer);
              hybridFreeDemoteBatch(request, pos, c, lastc, result, item); return ERR_PIPE_RING_CAPACITY; }
            r->item.linkIndex = i;
            r->item.slot = (uint32_t)p->residentSlot[r->coreSlot];
            r->item.boundarySide = 1;
            if (HybridRebalanceMetrics) ++HybridRebalanceMetrics->id_lookup_probes;
            if (MSXresident_getSlotIdentity(i, r->item.slot,
                                            &r->item.generation,
                                            &parcelId,
                                            &r->item.pipeEpoch) != MSX_RESIDENT_OK)
            { if (detail) hybridRebalanceAddPlan(MSXgpu_wallTimeMs() - planTimer);
              hybridFreeDemoteBatch(request, pos, c, lastc, result, item); return ERR_PIPE_RING_CAPACITY; }
            r->parcelId = parcelId;
            r->item.requestedVolume = 0.0;
            if (MSXsegStorage_hybridAcquireBoundary(&r->boundary) != 0)
            { if (detail) hybridRebalanceAddPlan(MSXgpu_wallTimeMs() - planTimer);
              hybridFreeDemoteBatch(request, pos, c, lastc, result, item); return ERR_MEMORY; }
        }
    }
    if (pos != count)
    { if (detail) hybridRebalanceAddPlan(MSXgpu_wallTimeMs() - planTimer);
      hybridFreeDemoteBatch(request, pos, c, lastc, result, item); return ERR_PIPE_RING_CAPACITY; }
    for (i = 0; i < count; i++) item[i] = request[i].item;
    if (detail) hybridRebalanceAddPlan(MSXgpu_wallTimeMs() - planTimer);
    status = MSXresidentRuntime_fetchBatch(item, count, c, lastc,
                                           (uint32_t)Hybrid.stride,
                                           result);
    if (status != MSX_RESIDENT_OK)
    {
        hybridFreeDemoteBatch(request, count, c, lastc, result, item);
        return ERR_GPU_KERNEL_RUNTIME_ERROR;
    }
    /* Fetch has its own mutually-exclusive bucket.  Start validation only
       after the runtime handoff returns so preflush/fetch is never counted
       again in rb_validate_commit_ms. */
    validateTimer = detail ? MSXgpu_wallTimeMs() : 0.0;
    for (i = 0; i < count; i++)
    {
        request[i].result = result[i];
        if (hybridValidateDemoteRequest(&request[i]))
        {
            if (detail) hybridRebalanceAddValidateCommit(
                MSXgpu_wallTimeMs() - validateTimer);
            hybridFreeDemoteBatch(request, count, c, lastc, result, item);
            return ERR_GPU_KERNEL_RUNTIME_ERROR;
        }
    }
    /* Every link is prevalidated against its complete projected endpoint
       sequence before any Resident row is cleared.  The packed request arena
       is built link-contiguously above, so no temporary index allocation is
       needed here. */
    for (begin = 0; begin < count; begin = end)
    {
        link = request[begin].linkIndex;
        end = begin + 1;
        while (end < count && request[end].linkIndex == link) ++end;
        if (HybridRebalanceMetrics)
            HybridRebalanceMetrics->validation_capacity_visits +=
                (uint64_t)(end - begin);
        status = MSXresident_validateRemoveBatch((uint32_t)link,
                                                  item + begin,
                                                  end - begin);
        if (status != MSX_RESIDENT_OK)
        {
            if (detail) hybridRebalanceAddValidateCommit(
                MSXgpu_wallTimeMs() - validateTimer);
            hybridFreeDemoteBatch(request, count, c, lastc, result, item);
            return ERR_GPU_KERNEL_RUNTIME_ERROR;
        }
    }
    /* The Core batch API repeats its cheap preflight and then performs only
       non-fallible bounded stores.  An unexpected failure poisons Resident;
       CPU topology is not advanced and the caller stops the quality step. */
    for (begin = 0; begin < count; begin = end)
    {
        link = request[begin].linkIndex;
        end = begin + 1;
        while (end < count && request[end].linkIndex == link) ++end;
        status = MSXresident_stageRemoveBatch((uint32_t)link,
                                               item + begin,
                                               end - begin);
        if (status != MSX_RESIDENT_OK)
        {
            MSXresident_poison();
            HybridResidentStatus = status;
            if (detail) hybridRebalanceAddValidateCommit(
                MSXgpu_wallTimeMs() - validateTimer);
            hybridFreeDemoteBatch(request, count, c, lastc, result, item);
            return ERR_GPU_KERNEL_RUNTIME_ERROR;
        }
    }
    for (i = 0; i < count; i++)
        if (hybridCommitDemoteRequest(&request[i]))
        {
            MSXresident_poison();
            HybridResidentStatus = MSX_RESIDENT_ERR_POISONED;
            if (detail) hybridRebalanceAddValidateCommit(
                MSXgpu_wallTimeMs() - validateTimer);
            hybridFreeDemoteBatch(request, count, c, lastc, result, item);
            return ERR_PIPE_RING_CAPACITY;
        }
        else
        {
            hybridSnapshotApplyDemote(&snapshot[request[i].linkIndex],
                                      &Hybrid.pipe[request[i].linkIndex],
                                      request[i].atHead);
            if (HybridRebalanceMetrics) ++HybridRebalanceMetrics->commit_rows;
            request[i].boundary = NULL;
        }
    hybridFreeDemoteBatch(request, count, c, lastc, result, item);
    if (detail) hybridRebalanceAddValidateCommit(
        MSXgpu_wallTimeMs() - validateTimer);
    if (MSXsegStorage_hybridTimingEnabled())
        MSXsegStorage_hybridTimingAddHandoff(MSXgpu_wallTimeMs() - timer);
    if (MSXgpu_profileDetailGroupEnabled(MSX_PROFILE_DETAIL_DEMOTE))
        MSXgpu_profileRecordDemote(count, MSXgpu_wallTimeMs() - timer);
    return 0;
}

static int hybridPromoteResidentExcess(int k, RebalanceSnapshot *s)
{
    HybridPipe *p = &Hybrid.pipe[k];
    Pseg seg;
    int guard, err;
    if (!s) return ERR_MEMORY;
    guard = s->guard;
    while (s->coreCount > 0 && s->downCount > guard)
    {
        seg = s->firstCore ? s->firstCore->next : NULL;
        if (!seg || seg->inHybridCore) break;
        err = hybridPromote(k, seg, TRUE);
        if (err) return err;
        hybridSnapshotApplyPromote(s, p, TRUE);
    }
    while (s->coreCount > 0 && s->upCount > guard)
    {
        seg = s->lastCore ? s->lastCore->prev : NULL;
        if (!seg || seg->inHybridCore) break;
        err = hybridPromote(k, seg, FALSE);
        if (err) return err;
        hybridSnapshotApplyPromote(s, p, FALSE);
    }
    return 0;
}
#endif

int MSXsegStorage_hybridizeAll(void)
{
    int k, err;
    if (!MSXsegStorage_isHybridEnabled() || !Hybrid.opened) return 0;
    /* Resident startup owns a separate prepare/commit transaction. */
    if (MSXresident_mode() == MSX_RESIDENT_RESIDENT && Hybrid.pipe[1].initialPrepared)
        return 0;
    /* Preflight all links before any promotion so a too-small CSV leaves
       every First/Last/nsegs and identity untouched. */
    for (k = 1; k <= Hybrid.nLinks; k++)
    {
        Pseg seg; int total = 0, projected;
        HybridPipe *p = &Hybrid.pipe[k];
        for (seg = MSX.FirstSeg[k]; seg; seg = seg->prev) total++;
        projected = total > 2 * hybridPipeGuard(p) ?
                    total - 2 * hybridPipeGuard(p) : 0;
        if ((p->fixedCap && projected > p->cap) ||
            (!p->fixedCap && projected > MSX.MaxSegments))
            return ERR_PIPE_RING_CAPACITY;
    }
    for (k = 1; k <= Hybrid.nLinks; k++)
    {
        err = hybridRebalanceLink(k);
        if (err)
        {
            MSX.ErrCode = err;
            return err;
        }
    }
    return 0;
}

int MSXsegStorage_hybridObserveAll(void)
{
    int k;
    if (!MSXsegStorage_isHybridEnabled() || !Hybrid.opened) return 0;
    for (k = 1; k <= Hybrid.nLinks; k++)
    {
        HybridResidentStatus = hybridResidentObserve(k);
        if (HybridResidentStatus != MSX_RESIDENT_OK) return ERR_PIPE_RING_CAPACITY;
    }
    return 0;
}

int MSXsegStorage_hybridRemoveHead(int k, Pseg seg)
{
    HybridPipe *p;
    int slot;
    Pseg next;
    double timer = 0.0;
    int timing = MSXsegStorage_hybridTimingEnabled();
    if (timing) timer = MSXgpu_wallTimeMs();
    if (!MSXsegStorage_isHybridLink(k) || !seg || !seg->inHybridCore)
        return 0;
    p = &Hybrid.pipe[k];
    slot = seg->hybridSlot;
    if (slot < 0 || slot >= p->cap || !p->used[slot] || slot != p->head)
    {
        snprintf(MSX.Msg, MAXLINE,
                 "Error 525 - HYBRID head mismatch: link=%d slot=%d head=%d cap=%d used=%d count=%d.",
                 k, slot, p->head, p->cap,
                 (slot >= 0 && slot < p->cap) ? p->used[slot] : 0, p->count);
        ENwriteline(MSX.Msg);
        return ERR_PIPE_RING_CAPACITY;
    }
    next = seg->prev;
    if (hybridResidentRemove(k, slot)) return ERR_PIPE_RING_CAPACITY;
    if (next) next->next = NULL;
    else MSX.LastSeg[k] = NULL;
    MSX.FirstSeg[k] = next;
    if (MSX.Link[k].nsegs > 0) MSX.Link[k].nsegs--;
    seg->next = NULL;
    seg->prev = NULL;
    seg->inHybridCore = FALSE;
    seg->ownerLink = 0;
    seg->hybridId = 0;
    p->used[slot] = FALSE;
    if (p->count == 1)
    {
        p->count = 0;
        p->head = p->tail = -1;
    }
    else
    {
        p->count--;
        p->head = hybridNextSlot(p, slot);
    }
    if (timing) MSXsegStorage_hybridTimingAddHandoff(MSXgpu_wallTimeMs() - timer);
    return 0;
}

void MSXsegStorage_hybridRebalanceAll(void)
{
    int k, err, audit = 0, promoteStarted = 0;
    MSXRebalanceMetrics metrics;
    double parentStart = 0.0, phaseStart;
    if (!MSXsegStorage_isHybridEnabled() || !Hybrid.opened) return;
#if defined(EPANETMSX_CUDA_ENABLED)
    if (MSXresidentRuntime_isResident())
    {
        audit = MSXgpu_profileDetailGroupEnabled(MSX_PROFILE_DETAIL_DEMOTE);
        if (hybridResidentScanAuditEnabled() && !HybridResidentScanAuditWritten)
        {
            int teamSize = 1;
#if defined(_OPENMP)
            /* Audit-only probe in the production Resident entry point.  The
               formal path remains free of the extra team and file operation. */
#pragma omp parallel
            {
#pragma omp single
                teamSize = omp_get_num_threads();
            }
#endif
            hybridResidentScanAuditRecord(teamSize);
        }
        if (audit)
        {
            memset(&metrics, 0, sizeof(metrics));
            HybridRebalanceMetrics = &metrics;
            HybridRebalanceStage = HYBRID_REBALANCE_SCAN;
            parentStart = MSXgpu_wallTimeMs();
        }
        /* Initialization contains only fallible promote commits.  Complete
           those first, then collect every post-transport demote before any
           CPU topology mutation or selected GPU fetch. */
        for (k = 1; k <= Hybrid.nLinks; k++)
        {
            RebalanceSnapshot *s = &Hybrid.rebalanceSnapshot[k];
            phaseStart = audit ? MSXgpu_wallTimeMs() : 0.0;
            if (audit) HybridRebalanceStage = HYBRID_REBALANCE_SCAN;
            hybridCaptureRebalanceSnapshot(k, s);
            if (audit) metrics.rb_scan_ms += MSXgpu_wallTimeMs() - phaseStart;
            if (s->coreCount == 0 && s->total > 2 * s->guard)
            {
                phaseStart = audit ? MSXgpu_wallTimeMs() : 0.0;
                if (audit) HybridRebalanceStage = HYBRID_REBALANCE_EMPTY_INIT;
                err = hybridInitializeResidentLink(k, s);
                if (audit) metrics.rb_empty_init_ms +=
                    MSXgpu_wallTimeMs() - phaseStart;
                if (err) { MSX.ErrCode = err; goto resident_done; }
                if (HybridRebalanceMetrics) ++HybridRebalanceMetrics->rebuilt_links;
            }
        }
        if (audit) HybridRebalanceStage = HYBRID_REBALANCE_PLAN;
        err = hybridDemoteResidentBatch();
        if (err) { MSX.ErrCode = err; goto resident_done; }
        phaseStart = audit ? MSXgpu_wallTimeMs() : 0.0;
        if (audit)
        {
            HybridRebalanceStage = HYBRID_REBALANCE_PROMOTE;
            promoteStarted = 1;
        }
        for (k = 1; k <= Hybrid.nLinks; k++)
        {
            err = hybridPromoteResidentExcess(k, &Hybrid.rebalanceSnapshot[k]);
            if (err) { MSX.ErrCode = err; goto resident_done; }
        }
resident_done:
        if (audit)
        {
            if (promoteStarted)
                metrics.rb_promote_ms += MSXgpu_wallTimeMs() - phaseStart;
            HybridRebalanceStage = HYBRID_REBALANCE_NONE;
            hybridRebalanceFinish(&metrics, parentStart);
            HybridRebalanceMetrics = NULL;
        }
        return;
    }
#endif
    for (k = 1; k <= Hybrid.nLinks; k++)
    {
        err = hybridRebalanceLink(k);
        if (err) { MSX.ErrCode = err; return; }
    }
}

int MSXsegStorage_hybridEnsureEndpointBoundary(int k, int boundarySide)
{
    if (!MSXsegStorage_isHybridLink(k) ||
        (boundarySide != 0 && boundarySide != 1)) return 0;
    if (boundarySide == 0)
    {
        while (MSX.FirstSeg[k] && MSX.FirstSeg[k]->inHybridCore)
        {
            int z = hybridDemote(k, TRUE);
            if (z) { MSX.ErrCode = z; return z; }
        }
    }
    else
    {
        while (MSX.LastSeg[k] && MSX.LastSeg[k]->inHybridCore)
        {
            int z = hybridDemote(k, FALSE);
            if (z) { MSX.ErrCode = z; return z; }
        }
    }
    return 0;
}

int MSXsegStorage_hybridAfterListReorder(int k)
{
    HybridPipe *p;
    Pseg firstCore, lastCore;
    int down, up, count;
    if (!MSXsegStorage_isHybridLink(k)) return 0;
    p = &Hybrid.pipe[k];
    if (p->count > 1)
    {
        /* Resident metadata is the fallible commit.  If it rejects (only
           epoch exhaustion or a poisoned state), restore the already-flipped
           CPU list before returning so the two owners cannot diverge. */
        if (hybridResidentReverse(k))
        {
            hybridRollbackListReverse(k);
            MSX.ErrCode = ERR_PIPE_RING_CAPACITY;
            return ERR_PIPE_RING_CAPACITY;
        }
        p->orient = -p->orient;
        hybridFindCore(p, k, &firstCore, &lastCore, &down, &up, &count);
    }
    return 0;
}

void MSXsegStorage_hybridClear(int k)
{
    HybridPipe *p;
    Pseg seg, next;
    MSXResidentStatus z;
    if (!MSXsegStorage_isHybridLink(k)) return;
    p = &Hybrid.pipe[k];
    /* Validate every CPU-side mapping before the single fallible Resident
       clear.  The commit loop below contains only bounded stores and pool
       returns, so it cannot strand a half-cleared ring. */
    for (seg = MSX.FirstSeg[k]; seg; seg = seg->prev)
        if (seg->inHybridCore)
        {
            int coreSlot = seg->hybridSlot;
            int residentSlot;
            if (coreSlot < 0 || coreSlot >= p->cap || !p->used[coreSlot] ||
                !p->residentSlot || !p->coreSlotForResident)
            {
                MSX.ErrCode = ERR_PIPE_RING_CAPACITY;
                return;
            }
            residentSlot = p->residentSlot[coreSlot];
            if (residentSlot < 0 || residentSlot >= p->cap ||
                p->coreSlotForResident[residentSlot] != coreSlot)
            {
                MSX.ErrCode = ERR_PIPE_RING_CAPACITY;
                return;
            }
        }
    z = MSXresident_stageClear((uint32_t)k);
    if (z != MSX_RESIDENT_OK)
    {
        HybridResidentStatus = z;
        MSX.ErrCode = ERR_PIPE_RING_CAPACITY;
        return;
    }
    seg = MSX.FirstSeg[k];
    while (seg)
    {
        next = seg->prev;
        if (seg->inHybridCore)
        {
            int coreSlot = seg->hybridSlot;
            int residentSlot = p->residentSlot[coreSlot];
            p->residentSlot[coreSlot] = -1;
            if (residentSlot >= 0 && residentSlot < p->cap)
                p->coreSlotForResident[residentSlot] = -1;
            seg->inHybridCore = FALSE;
            seg->ownerLink = 0;
            seg->hybridId = 0;
        }
        else
            MSXqual_removeSeg(seg);
        seg = next;
    }
    MSX.FirstSeg[k] = NULL;
    MSX.LastSeg[k] = NULL;
    MSX.Link[k].nsegs = 0;
    p->count = 0;
    p->head = p->tail = -1;
    memset(p->used, 0, (size_t)p->cap * sizeof(unsigned char));
    memset(p->residentSlot, 0xFF, (size_t)p->cap * sizeof(int));
    memset(p->coreSlotForResident, 0xFF, (size_t)p->cap * sizeof(int));
}

int MSXsegStorage_hybridCoreCount(int k)
{
    if (!MSXsegStorage_isHybridLink(k)) return 0;
    return Hybrid.pipe[k].count;
}

int MSXsegStorage_hybridCoreCapacity(int k)
{
    if (!MSXsegStorage_isHybridLink(k)) return 0;
    return Hybrid.pipe[k].cap;
}

int MSXsegStorage_hybridCoreSlotAt(int k, int pos)
{
    HybridPipe *p;
    int slot;
    if (!MSXsegStorage_isHybridLink(k) || pos < 0) return -1;
    p = &Hybrid.pipe[k];
    if (pos >= p->count || p->head < 0) return -1;
    slot = p->head + p->orient * pos;
    slot %= p->cap;
    if (slot < 0) slot += p->cap;
    return p->used[slot] ? slot : -1;
}

int MSXsegStorage_hybridOrientation(int k)
{
    if (!MSXsegStorage_isHybridLink(k)) return 0;
    return Hybrid.pipe[k].orient;
}

Pseg MSXsegStorage_hybridCoreSegFromHead(int k, int pos)
{
    /* Keep the legacy accessor on the same direct slot path as the React
       loop.  In particular, do not regress to a Pseg-list walk here. */
    HybridPipe *p;
    int slot;
    if (!MSXsegStorage_isHybridLink(k) || pos < 0) return NULL;
    p = &Hybrid.pipe[k];
    if (pos >= p->count || p->head < 0) return NULL;
    slot = p->head + p->orient * pos;
    slot %= p->cap;
    if (slot < 0) slot += p->cap;
    if (!p->used[slot]) return NULL;
    return p->view[slot];
}

Pseg MSXsegStorage_hybridCoreSegAt(int k, int pos)
{
    HybridPipe *p;
    int slot;
    if (!MSXsegStorage_isHybridLink(k) || pos < 0) return NULL;
    p = &Hybrid.pipe[k];
    if (pos >= p->count || p->head < 0) return NULL;
    slot = p->head + p->orient * pos;
    slot %= p->cap;
    if (slot < 0) slot += p->cap;
    if (!p->used[slot]) return NULL;
    return p->view[slot];
}

int MSXsegStorage_hybridCoreSpan(int k, int spanIndex, Pseg **segs,
                                 int *count)
{
    HybridPipe *p;
    int firstCount, secondCount;
    if (segs) *segs = NULL;
    if (count) *count = 0;
    if (!MSXsegStorage_isHybridLink(k) || spanIndex < 0 || spanIndex > 1)
        return 0;
    p = &Hybrid.pipe[k];
    if (p->count <= 0 || p->head < 0) return 0;

    /* Dense Core occupancy is a ring interval.  Expose that interval as
       one span, or two only when it wraps; never locate Core rows through
       the compatibility Pseg linked list. */
    if (p->orient > 0)
    {
        firstCount = p->cap - p->head;
        if (firstCount > p->count) firstCount = p->count;
        secondCount = p->count - firstCount;
        if (spanIndex == 0)
        {
            if (segs) *segs = p->view + p->head;
            if (count) *count = firstCount;
            return firstCount;
        }
        if (secondCount > 0)
        {
            if (segs) *segs = p->view;
            if (count) *count = secondCount;
        }
        return secondCount;
    }

    /* Reverse flow traverses decreasing slots.  The view array itself is
       ascending, so return spans in logical order but let the caller walk
       each span backwards.  A negative count denotes that direction. */
    firstCount = p->head + 1;
    if (firstCount > p->count) firstCount = p->count;
    secondCount = p->count - firstCount;
    if (spanIndex == 0)
    {
        if (segs) *segs = p->view + p->head;
        if (count) *count = -firstCount;
        return firstCount;
    }
    if (secondCount > 0)
    {
        if (segs) *segs = p->view + (p->cap - 1);
        if (count) *count = -secondCount;
    }
    return secondCount;
}

void MSXsegStorage_hybridPrepareCore(int k)
{
    HybridPipe *p;
    int pos, slot;
    if (!MSXsegStorage_isHybridLink(k)) return;
    p = &Hybrid.pipe[k];
    slot = p->head;
    for (pos = 0; pos < p->count; pos++)
    {
        if (p->used[slot])
        {
            p->view[slot]->v = p->v[slot];
            p->view[slot]->hstep = p->hstep[slot];
            p->view[slot]->hresponse = p->hresponse[slot];
            p->view[slot]->uresponse = p->uresponse[slot];
            p->view[slot]->dresponse = p->dresponse[slot];
        }
        slot = hybridNextSlot(p, slot);
    }
}

void MSXsegStorage_hybridCommitCore(int k)
{
    HybridPipe *p;
    int pos, slot;
    if (!MSXsegStorage_isHybridLink(k)) return;
    p = &Hybrid.pipe[k];
    slot = p->head;
    for (pos = 0; pos < p->count; pos++)
    {
        if (p->used[slot])
        {
            p->v[slot] = p->view[slot]->v;
            p->hstep[slot] = p->view[slot]->hstep;
            p->hresponse[slot] = p->view[slot]->hresponse;
            p->uresponse[slot] = p->view[slot]->uresponse;
            p->dresponse[slot] = p->view[slot]->dresponse;
        }
        slot = hybridNextSlot(p, slot);
    }
}

int MSXsegStorage_hybridTimingEnabled(void)
{
    return MSXsegStorage_isHybridEnabled() && Hybrid.timingEnabled &&
           MSXgpu_profileStageEnabled();
}

static void hybridTimingAdd(double *dst, double ms)
{
    if (dst && Hybrid.timingEnabled && MSXgpu_profileStageEnabled())
    {
        /* Core/boundary React regions run one link per OpenMP worker.  The
           transport/rebalance handoff calls remain sequential, but using the
           same atomic accumulator for every component keeps detailed timing
           correct if that scheduling changes later. */
#ifdef _OPENMP
#pragma omp atomic
#endif
        *dst += ms;
    }
}

void MSXsegStorage_hybridTimingAddCoreReact(double ms) { hybridTimingAdd(&Hybrid.coreReactMs, ms); }
void MSXsegStorage_hybridTimingAddBoundaryReact(double ms) { hybridTimingAdd(&Hybrid.boundaryReactMs, ms); }
void MSXsegStorage_hybridTimingAddHandoff(double ms) { hybridTimingAdd(&Hybrid.handoffMs, ms); }
void MSXsegStorage_hybridTimingAddPacking(double ms) { hybridTimingAdd(&Hybrid.packingMs, ms); }
void MSXsegStorage_hybridTimingAddH2D(double ms) { hybridTimingAdd(&Hybrid.h2dMs, ms); }
void MSXsegStorage_hybridTimingAddKernel(double ms) { hybridTimingAdd(&Hybrid.kernelMs, ms); }
void MSXsegStorage_hybridTimingAddD2H(double ms) { hybridTimingAdd(&Hybrid.d2hMs, ms); }
void MSXsegStorage_hybridTimingAddUnpack(double ms) { hybridTimingAdd(&Hybrid.unpackMs, ms); }

void MSXsegStorage_hybridTimingStepBegin(void)
{
    if (MSXsegStorage_hybridTimingEnabled()) Hybrid.stepStartMs = MSXgpu_wallTimeMs();
}

void MSXsegStorage_hybridTimingStepEnd(void)
{
    if (MSXsegStorage_hybridTimingEnabled() && Hybrid.stepStartMs > 0.0)
        Hybrid.totalMs += MSXgpu_wallTimeMs() - Hybrid.stepStartMs;
}

void MSXsegStorage_hybridSyncSegmentScalars(Pseg seg)
{
    HybridPipe *p;
    int slot;
    if (!seg || !seg->inHybridCore || !MSXsegStorage_isHybridLink(seg->ownerLink)) return;
    p = &Hybrid.pipe[seg->ownerLink];
    slot = seg->hybridSlot;
    if (slot < 0 || slot >= p->cap || !p->used[slot]) return;
    p->v[slot] = seg->v;
    p->hstep[slot] = seg->hstep;
    p->hresponse[slot] = seg->hresponse;
    p->uresponse[slot] = seg->uresponse;
    p->dresponse[slot] = seg->dresponse;
}

void MSXsegStorage_hybridSyncAllScalars(void)
{
    int k, pos, slot;
    HybridPipe *p;
    if (!MSXsegStorage_isHybridEnabled() || !Hybrid.opened) return;
    for (k = 1; k <= Hybrid.nLinks; k++)
    {
        p = &Hybrid.pipe[k];
        slot = p->head;
        for (pos = 0; pos < p->count; pos++)
        {
            if (p->used[slot])
                MSXsegStorage_hybridSyncSegmentScalars(p->view[slot]);
            slot = hybridNextSlot(p, slot);
        }
    }
}

int MSXsegStorage_hybridAuditPoisonCoreMirrors(void)
{
    int k, pos, slot, m;
    HybridPipe *p;
    if (!MSXsegStorage_isHybridEnabled() || !Hybrid.opened) return 0;
    /* Audit-only walk: dense Core rows are the only CPU concentrations that
       must be proven unused by Resident consumers.  Boundary rows are left
       intact because they remain an intentional CPU source of truth. */
    for (k = 1; k <= Hybrid.nLinks; ++k)
    {
        p = &Hybrid.pipe[k];
        slot = p->head;
        for (pos = 0; pos < p->count; ++pos)
        {
            if (p->used[slot])
            {
                size_t row = (size_t)slot * (size_t)Hybrid.stride;
                Pseg seg = p->view[slot];
                for (m = 1; m <= MSX.Nobjects[SPECIES]; ++m)
                {
                    p->c[row + m] = NAN;
                    p->lastc[row + m] = NAN;
                    if (seg && seg->c && seg->lastc)
                    {
                        seg->c[m] = NAN;
                        seg->lastc[m] = NAN;
                    }
                }
            }
            slot = hybridNextSlot(p, slot);
        }
    }
    return 0;
}

static size_t totalSlots(void)
{
    return (size_t)(Ring.nLinks + 1) * (size_t)Ring.cap;
}

static int flatSlotIndex(int k, int slot)
{
    return k * Ring.cap + slot;
}

static int nextSlot(int slot, int orient)
{
    int s = slot + orient;
    if (s >= Ring.cap) s = 0;
    else if (s < 0) s = Ring.cap - 1;
    return s;
}

static int slotFromHead(int k, int pos)
{
    int slot = Ring.head[k] + Ring.orient[k] * pos;
    while (slot >= Ring.cap) slot -= Ring.cap;
    while (slot < 0) slot += Ring.cap;
    return slot;
}

static int slotFromTail(int k, int pos)
{
    int slot = Ring.tail[k] - Ring.orient[k] * pos;
    while (slot >= Ring.cap) slot -= Ring.cap;
    while (slot < 0) slot += Ring.cap;
    return slot;
}

static void clearRing(void)
{
    FREE(Ring.c);
    FREE(Ring.lastc);
    FREE(Ring.v);
    FREE(Ring.hstep);
    FREE(Ring.hresponse);
    FREE(Ring.uresponse);
    FREE(Ring.dresponse);
    FREE(Ring.used);
    FREE(Ring.seg);
    FREE(Ring.head);
    FREE(Ring.tail);
    FREE(Ring.count);
    FREE(Ring.orient);
    memset(&Ring, 0, sizeof(Ring));
}

static void pointSegToPrivate(Pseg seg)
{
    if (!seg) return;
    seg->c = seg->privateC;
    seg->lastc = seg->privateLastC;
    seg->inPipeRing = FALSE;
    seg->ownerLink = 0;
    seg->ringSlot = -1;
    seg->ringIndex = -1;
}

int MSXsegStorage_hybridAcquireBoundary(Pseg *segment)
{
    uint32_t i;
    Pseg seg;
    if (!segment) return ERR_MEMORY;
    *segment = NULL;
    if (!Hybrid.opened || !Hybrid.demoteBoundaryCapacity ||
        !Hybrid.demoteBoundaryNext)
        return MSX_RESIDENT_DISABLED;
    i = Hybrid.demoteBoundaryFreeHead;
    if (i == UINT32_MAX) return ERR_PIPE_RING_CAPACITY;
    if (i >= Hybrid.demoteBoundaryCapacity ||
        Hybrid.demoteBoundaryState[i] != 0)
        return ERR_PIPE_RING_CAPACITY;
    seg = Hybrid.demoteBoundary[i];
    if (!seg || !seg->privateC || !seg->privateLastC) return ERR_MEMORY;
    Hybrid.demoteBoundaryFreeHead = Hybrid.demoteBoundaryNext[i];
    pointSegToPrivate(seg);
    memset(seg->privateC, 0, (size_t)Hybrid.stride * sizeof(double));
    memset(seg->privateLastC, 0, (size_t)Hybrid.stride * sizeof(double));
    seg->prev = seg->next = NULL;
    seg->hybridId = 0;
    seg->hybridBoundaryPoolIndex = (int)i;
    Hybrid.demoteBoundaryState[i] = 1;
    *segment = seg;
    return 0;
}

int MSXsegStorage_hybridReleaseBoundary(Pseg segment)
{
    int i;
    if (!segment || !Hybrid.demoteBoundary) return 0;
    i = segment->hybridBoundaryPoolIndex;
    if (i < 0 || (uint32_t)i >= Hybrid.demoteBoundaryCapacity ||
        Hybrid.demoteBoundary[i] != segment ||
        (Hybrid.demoteBoundaryState[i] != 1 &&
         Hybrid.demoteBoundaryState[i] != 2) ||
        !Hybrid.demoteBoundaryNext) return 0;
    Hybrid.demoteBoundaryState[i] = 0;
    Hybrid.demoteBoundaryNext[i] = Hybrid.demoteBoundaryFreeHead;
    Hybrid.demoteBoundaryFreeHead = (uint32_t)i;
    pointSegToPrivate(segment);
    segment->prev = segment->next = NULL;
    segment->hybridId = 0;
    segment->hybridBoundaryPoolIndex = -1;
    return 1;
}

static void hybridMarkBoundaryCommitted(Pseg seg)
{
    int i;
    if (!seg || !Hybrid.demoteBoundary) return;
    i = seg->hybridBoundaryPoolIndex;
    if (i < 0 || (uint32_t)i >= Hybrid.demoteBoundaryCapacity ||
        Hybrid.demoteBoundary[i] != seg) return;
    if (Hybrid.demoteBoundaryState[i] == 1)
        Hybrid.demoteBoundaryState[i] = 2;
}

static void copySegToSlot(int k, int slot, Pseg seg)
{
    int m;
    int index = flatSlotIndex(k, slot);
    size_t row = (size_t)index * (size_t)Ring.stride;

    for (m = 1; m <= MSX.Nobjects[SPECIES]; m++)
    {
        Ring.c[row + m] = seg->c[m];
        Ring.lastc[row + m] = seg->lastc[m];
    }
    Ring.v[index] = seg->v;
    Ring.hstep[index] = seg->hstep;
    Ring.hresponse[index] = seg->hresponse;
    Ring.uresponse[index] = seg->uresponse;
    Ring.dresponse[index] = seg->dresponse;
}

static void bindSegToSlot(int k, int slot, Pseg seg)
{
    int index = flatSlotIndex(k, slot);
    size_t row = (size_t)index * (size_t)Ring.stride;

    Ring.used[index] = TRUE;
    Ring.seg[index] = seg;
    seg->ownerLink = k;
    seg->ringSlot = slot;
    seg->ringIndex = index;
    seg->inPipeRing = TRUE;
    seg->c = Ring.c + row;
    seg->lastc = Ring.lastc + row;
}

static void syncSlotToSeg(int k, int slot)
{
    int index;
    Pseg seg;

    if (!Ring.opened || k <= 0 || k > Ring.nLinks) return;
    if (slot < 0 || slot >= Ring.cap) return;
    index = flatSlotIndex(k, slot);
    seg = Ring.seg[index];
    if (!seg || !Ring.used[index]) return;

    seg->v = Ring.v[index];
    seg->hstep = Ring.hstep[index];
    seg->hresponse = Ring.hresponse[index];
    seg->uresponse = Ring.uresponse[index];
    seg->dresponse = Ring.dresponse[index];
}

static void syncSegToSlot(Pseg seg)
{
    int index;
    if (!seg || !seg->inPipeRing || !Ring.opened) return;
    index = seg->ringIndex;
    if (index < 0 || index >= (int)totalSlots()) return;
    Ring.v[index] = seg->v;
    Ring.hstep[index] = seg->hstep;
    Ring.hresponse[index] = seg->hresponse;
    Ring.uresponse[index] = seg->uresponse;
    Ring.dresponse[index] = seg->dresponse;
}

static int validActiveSlot(int k, int slot)
{
    int index;
    if (!MSXsegStorage_isPipeRingLink(k)) return -1;
    if (slot < 0 || slot >= Ring.cap) return -1;
    index = flatSlotIndex(k, slot);
    if (!Ring.used[index]) return -1;
    return index;
}

static void releaseSlotToFreeSeg(int k, int slot)
{
    int index;
    Pseg seg;

    if (!Ring.opened || k <= 0 || k > Ring.nLinks) return;
    if (slot < 0 || slot >= Ring.cap) return;
    index = flatSlotIndex(k, slot);
    seg = Ring.seg[index];

    Ring.used[index] = FALSE;
    Ring.seg[index] = NULL;
    Ring.v[index] = 0.0;
    Ring.hstep[index] = 0.0;
    Ring.hresponse[index] = 0.0;
    Ring.uresponse[index] = 0.0;
    Ring.dresponse[index] = 0.0;
    memset(Ring.c + (size_t)index * (size_t)Ring.stride, 0,
           (size_t)Ring.stride * sizeof(double));
    memset(Ring.lastc + (size_t)index * (size_t)Ring.stride, 0,
           (size_t)Ring.stride * sizeof(double));

    if (seg)
    {
        pointSegToPrivate(seg);
        seg->prev = MSX.FreeSeg;
        seg->next = NULL;
        MSX.FreeSeg = seg;
    }
}

int MSXsegStorage_isPipeRingEnabled(void)
{
    return MSX.SegmentStorage == SEG_STORAGE_PIPE_RING;
}

int MSXsegStorage_isHybridEnabled(void)
{
    return MSX.SegmentStorage == SEG_STORAGE_HYBRID;
}

int MSXsegStorage_isHybridLink(int k)
{
    return MSXsegStorage_isHybridEnabled() && Hybrid.opened &&
           k > 0 && k <= Hybrid.nLinks;
}

int MSXsegStorage_isHybridCoreSegment(Pseg seg)
{
    return seg != NULL && seg->inHybridCore != FALSE;
}

int MSXsegStorage_isHybridCoreIdentity(int k, uint64_t hybridId)
{
    HybridPipe *p;
    int slot, pos;
    if (!hybridId || !MSXsegStorage_isHybridLink(k)) return 0;
    p = &Hybrid.pipe[k];
    slot = p->head;
    for (pos = 0; pos < p->count; pos++)
    {
        Pseg seg = p->view[slot];
        if (seg && seg->inHybridCore && seg->hybridId == hybridId) return 1;
        slot = hybridNextSlot(p, slot);
    }
    return 0;
}

int MSXsegStorage_isHybridCoreSlotIdentity(int k, int slot,
                                            uint64_t hybridId)
{
    HybridPipe *p;
    Pseg seg;
    int coreSlot;
    if (!hybridId || !MSXsegStorage_isHybridLink(k)) return 0;
    p = &Hybrid.pipe[k];
    if (slot < 0 || slot >= p->cap || !p->coreSlotForResident) return 0;
    coreSlot = p->coreSlotForResident[slot];
    if (coreSlot < 0 || coreSlot >= p->cap || !p->used[coreSlot]) return 0;
    seg = p->view[coreSlot];
    return seg && seg->inHybridCore && seg->ownerLink == k &&
           seg->hybridSlot == coreSlot && seg->hybridId == hybridId;
}

int MSXsegStorage_hybridApplyResidentPayload(int k, int residentSlot,
                                              uint64_t hybridId,
                                              const MSXResidentPayload *payload)
{
    HybridPipe *p;
    Pseg seg;
    int coreSlot, m;

    if (!payload || !payload->c || !payload->lastc ||
        !MSXsegStorage_isHybridCoreSlotIdentity(k, residentSlot, hybridId))
        return 0;
    p = &Hybrid.pipe[k];
    coreSlot = p->coreSlotForResident[residentSlot];
    seg = p->view[coreSlot];
    if (!seg || !seg->c || !seg->lastc) return 0;
    seg->v = payload->volume;
    seg->hstep = payload->hstep;
    seg->hresponse = payload->hresponse;
    seg->uresponse = payload->uresponse;
    seg->dresponse = payload->dresponse;
    for (m = 0; m <= Hybrid.nSpecies; m++)
    {
        seg->c[m] = payload->c[m];
        seg->lastc[m] = payload->lastc[m];
    }
    return 1;
}

void MSXsegStorage_hybridAssignIdentity(int k, Pseg seg)
{
    HybridPipe *p;
    if (!MSXsegStorage_isHybridLink(k) || !seg || seg->hybridId != 0) return;
    p = &Hybrid.pipe[k];
    if (p->nextId >= 0x0000FFFFFFFFFFFFULL)
    {
        HybridResidentStatus = MSX_RESIDENT_ERR_OVERFLOW;
        return;
    }
    p->nextId++;
    seg->hybridId = ((unsigned long long)(unsigned int)k << 48) |
                    (p->nextId & 0x0000FFFFFFFFFFFFULL);
}

MSXResidentStatus MSXsegStorage_residentLastStatus(void)
{
    return HybridResidentStatus;
}

int MSXsegStorage_isPipeRingLink(int k)
{
    return MSXsegStorage_isPipeRingEnabled() &&
           Ring.opened &&
           k > 0 &&
           k <= Ring.nLinks;
}

int MSXsegStorage_isPipeRingSegment(Pseg seg)
{
    return seg && seg->inPipeRing;
}

int MSXsegStorage_open(void)
{
    size_t slots;
    size_t rows;

    clearRing();
    hybridClear();
    if (MSXsegStorage_isHybridEnabled()) return hybridOpen();
    if (!MSXsegStorage_isPipeRingEnabled()) return 0;

    if (MSX.PipeRingCap <= 0) return ERR_PIPE_RING_CAPACITY;

    Ring.cap = MSX.PipeRingCap;
    Ring.nLinks = MSX.Nobjects[LINK];
    Ring.nSpecies = MSX.Nobjects[SPECIES];
    Ring.stride = Ring.nSpecies + 1;
    slots = totalSlots();
    rows = slots * (size_t)Ring.stride;

    Ring.c = (double *)calloc(rows, sizeof(double));
    Ring.lastc = (double *)calloc(rows, sizeof(double));
    Ring.v = (double *)calloc(slots, sizeof(double));
    Ring.hstep = (double *)calloc(slots, sizeof(double));
    Ring.hresponse = (double *)calloc(slots, sizeof(double));
    Ring.uresponse = (double *)calloc(slots, sizeof(double));
    Ring.dresponse = (double *)calloc(slots, sizeof(double));
    Ring.used = (unsigned char *)calloc(slots, sizeof(unsigned char));
    Ring.seg = (Pseg *)calloc(slots, sizeof(Pseg));
    Ring.head = (int *)calloc((size_t)Ring.nLinks + 1, sizeof(int));
    Ring.tail = (int *)calloc((size_t)Ring.nLinks + 1, sizeof(int));
    Ring.count = (int *)calloc((size_t)Ring.nLinks + 1, sizeof(int));
    Ring.orient = (int *)calloc((size_t)Ring.nLinks + 1, sizeof(int));

    if (!Ring.c || !Ring.lastc || !Ring.v || !Ring.hstep ||
        !Ring.hresponse || !Ring.uresponse || !Ring.dresponse ||
        !Ring.used || !Ring.seg || !Ring.head || !Ring.tail ||
        !Ring.count || !Ring.orient)
    {
        clearRing();
        return ERR_MEMORY;
    }

    Ring.opened = TRUE;
    MSXsegStorage_reset();
    return 0;
}

void MSXsegStorage_close(void)
{
    if (MSXsegStorage_isHybridEnabled()) hybridWriteTiming();
    clearRing();
    hybridClear();
}

void MSXsegStorage_reset(void)
{
    int k;

    if (MSXsegStorage_isHybridEnabled())
    {
        hybridClear();
        (void)hybridOpen();
        return;
    }

    if (!Ring.opened) return;

    memset(Ring.c, 0, totalSlots() * (size_t)Ring.stride * sizeof(double));
    memset(Ring.lastc, 0, totalSlots() * (size_t)Ring.stride * sizeof(double));
    memset(Ring.v, 0, totalSlots() * sizeof(double));
    memset(Ring.hstep, 0, totalSlots() * sizeof(double));
    memset(Ring.hresponse, 0, totalSlots() * sizeof(double));
    memset(Ring.uresponse, 0, totalSlots() * sizeof(double));
    memset(Ring.dresponse, 0, totalSlots() * sizeof(double));
    memset(Ring.used, 0, totalSlots() * sizeof(unsigned char));
    memset(Ring.seg, 0, totalSlots() * sizeof(Pseg));
    memset(Ring.head, 0, ((size_t)Ring.nLinks + 1) * sizeof(int));
    memset(Ring.tail, 0, ((size_t)Ring.nLinks + 1) * sizeof(int));
    memset(Ring.count, 0, ((size_t)Ring.nLinks + 1) * sizeof(int));
    for (k = 1; k <= Ring.nLinks; k++)
    {
        Ring.tail[k] = -1;
        Ring.orient[k] = 1;
    }
}

int MSXsegStorage_preparePrivate(Pseg seg)
{
    if (!seg) return ERR_MEMORY;

    if (!seg->privateC)
    {
        seg->privateC = (double *)Alloc((MSX.Nobjects[SPECIES] + 1) * sizeof(double));
        if (!seg->privateC)
        {
            MSX.OutOfMemory = TRUE;
            return ERR_MEMORY;
        }
    }
    if (!seg->privateLastC)
    {
        seg->privateLastC = (double *)Alloc((MSX.Nobjects[SPECIES] + 1) * sizeof(double));
        if (!seg->privateLastC)
        {
            MSX.OutOfMemory = TRUE;
            return ERR_MEMORY;
        }
    }

    pointSegToPrivate(seg);
    seg->hybridSlot = -1;
    seg->inHybridCore = FALSE;
    seg->hybridId = 0;
    seg->hybridBoundaryPoolIndex = -1;
    return 0;
}

void MSXsegStorage_initPrivateValues(Pseg seg, const double c[])
{
    int m;
    if (!seg || !seg->privateC || !seg->privateLastC) return;

    seg->c = seg->privateC;
    seg->lastc = seg->privateLastC;
    for (m = 1; m <= MSX.Nobjects[SPECIES]; m++)
    {
        seg->privateC[m] = c[m];
        seg->privateLastC[m] = c[m];
    }
}

int MSXsegStorage_bindPipeSegment(int k, Pseg seg)
{
    if (MSXsegStorage_isHybridLink(k)) return 0;
    return MSXsegStorage_pipeAppendTail(k, seg);
}

void MSXsegStorage_unbindSegment(Pseg seg)
{
    int k, slot, index;

    if (!seg || seg->inHybridCore) return;
    if (!seg->inPipeRing) return;

    k = seg->ownerLink;
    slot = seg->ringSlot;
    index = seg->ringIndex;
    if (Ring.opened && k > 0 && k <= Ring.nLinks &&
        slot >= 0 && slot < Ring.cap &&
        index == flatSlotIndex(k, slot) &&
        Ring.used[index])
    {
        Ring.used[index] = FALSE;
        Ring.seg[index] = NULL;
        Ring.v[index] = 0.0;
        if (Ring.count[k] > 0) Ring.count[k]--;
        MSX.Link[k].nsegs = Ring.count[k];
    }

    pointSegToPrivate(seg);
}

int MSXsegStorage_pipeAppendTail(int k, Pseg seg)
{
    int slot, index;

    if (!seg) return ERR_MEMORY;
    if (!MSXsegStorage_isPipeRingEnabled()) return 0;
    if (k <= 0 || k > MSX.Nobjects[LINK]) return 0;
    if (!Ring.opened) return ERR_PIPE_RING_CAPACITY;
    if (seg->inPipeRing) return 0;
    if (Ring.count[k] >= Ring.cap)
    {
        snprintf(MSX.Msg, MAXLINE,
                 "Error 525 - PIPE_RING capacity exceeded: pipe index=%d, cap=%d, active_segments=%d.",
                 k, Ring.cap, Ring.count[k]);
        ENwriteline(MSX.Msg);
        return ERR_PIPE_RING_CAPACITY;
    }

    if (Ring.count[k] == 0)
    {
        slot = Ring.head[k];
        Ring.tail[k] = slot;
    }
    else
    {
        slot = nextSlot(Ring.tail[k], Ring.orient[k]);
        Ring.tail[k] = slot;
    }

    index = flatSlotIndex(k, slot);
    if (Ring.used[index])
    {
        snprintf(MSX.Msg, MAXLINE,
                 "Error 525 - PIPE_RING slot collision: pipe index=%d, cap=%d, active_segments=%d.",
                 k, Ring.cap, Ring.count[k]);
        ENwriteline(MSX.Msg);
        return ERR_PIPE_RING_CAPACITY;
    }

    copySegToSlot(k, slot, seg);
    bindSegToSlot(k, slot, seg);
    Ring.count[k]++;
    MSX.Link[k].nsegs = Ring.count[k];
    return 0;
}

Pseg MSXsegStorage_pipePeekHead(int k)
{
    if (!MSXsegStorage_isPipeRingLink(k) || Ring.count[k] <= 0) return NULL;
    return MSXsegStorage_pipeSegFromHead(k, 0);
}

Pseg MSXsegStorage_pipePeekTail(int k)
{
    if (!MSXsegStorage_isPipeRingLink(k) || Ring.count[k] <= 0) return NULL;
    return MSXsegStorage_pipeSegFromTail(k, 0);
}

int MSXsegStorage_pipePopHead(int k)
{
    int oldHead;

    if (!MSXsegStorage_isPipeRingLink(k) || Ring.count[k] <= 0) return 0;

    oldHead = Ring.head[k];
    releaseSlotToFreeSeg(k, oldHead);
    Ring.count[k]--;
    if (Ring.count[k] == 0)
    {
        Ring.head[k] = 0;
        Ring.tail[k] = -1;
    }
    else
    {
        Ring.head[k] = nextSlot(oldHead, Ring.orient[k]);
    }
    MSX.Link[k].nsegs = Ring.count[k];
    return 0;
}

int MSXsegStorage_pipeConsumeHead(int k, double volume)
{
    int index;
    if (!MSXsegStorage_isPipeRingLink(k) || Ring.count[k] <= 0) return 0;
    index = flatSlotIndex(k, Ring.head[k]);
    Ring.v[index] -= volume;
    if (Ring.v[index] < 0.0) Ring.v[index] = 0.0;
    syncSlotToSeg(k, Ring.head[k]);
    return 0;
}

int MSXsegStorage_pipeMergeTail(int k, const double upnodeQual[], double volume)
{
    int m, index;
    double oldv, newv;
    Pseg seg;

    if (!MSXsegStorage_isPipeRingLink(k) || Ring.count[k] <= 0) return 0;
    index = flatSlotIndex(k, Ring.tail[k]);
    seg = Ring.seg[index];
    if (!seg || !Ring.used[index]) return 0;

    oldv = Ring.v[index];
    newv = oldv + volume;
    if (newv <= 0.0) return 0;
    for (m = 1; m <= MSX.Nobjects[SPECIES]; m++)
    {
        if (MSX.Species[m].type == BULK)
            seg->c[m] = (seg->c[m] * oldv + upnodeQual[m] * volume) / newv;
    }
    Ring.v[index] = newv;
    syncSlotToSeg(k, Ring.tail[k]);
    return 0;
}

int MSXsegStorage_pipeReverse(int k)
{
    int h;
    if (!MSXsegStorage_isPipeRingLink(k) || Ring.count[k] <= 1) return 0;
    h = Ring.head[k];
    Ring.head[k] = Ring.tail[k];
    Ring.tail[k] = h;
    Ring.orient[k] = -Ring.orient[k];
    return 0;
}

void MSXsegStorage_pipeClear(int k)
{
    int pos, slot;
    if (!MSXsegStorage_isPipeRingLink(k)) return;

    for (pos = 0; pos < Ring.count[k]; pos++)
    {
        slot = slotFromHead(k, pos);
        releaseSlotToFreeSeg(k, slot);
    }
    Ring.head[k] = 0;
    Ring.tail[k] = -1;
    Ring.count[k] = 0;
    Ring.orient[k] = 1;
    MSX.FirstSeg[k] = NULL;
    MSX.LastSeg[k] = NULL;
    MSX.Link[k].nsegs = 0;
}

int MSXsegStorage_pipeCount(int k)
{
    if (!MSXsegStorage_isPipeRingLink(k)) return 0;
    return Ring.count[k];
}

Pseg MSXsegStorage_pipeSegFromHead(int k, int pos)
{
    int slot, index;
    if (!MSXsegStorage_isPipeRingLink(k)) return NULL;
    if (pos < 0 || pos >= Ring.count[k]) return NULL;
    slot = slotFromHead(k, pos);
    index = flatSlotIndex(k, slot);
    syncSlotToSeg(k, slot);
    return Ring.seg[index];
}

Pseg MSXsegStorage_pipeSegFromTail(int k, int pos)
{
    int slot, index;
    if (!MSXsegStorage_isPipeRingLink(k)) return NULL;
    if (pos < 0 || pos >= Ring.count[k]) return NULL;
    slot = slotFromTail(k, pos);
    index = flatSlotIndex(k, slot);
    syncSlotToSeg(k, slot);
    return Ring.seg[index];
}

int MSXsegStorage_pipeSlotFromHead(int k, int pos)
{
    if (!MSXsegStorage_isPipeRingLink(k)) return -1;
    if (pos < 0 || pos >= Ring.count[k]) return -1;
    return slotFromHead(k, pos);
}

int MSXsegStorage_pipeSlotFromTail(int k, int pos)
{
    if (!MSXsegStorage_isPipeRingLink(k)) return -1;
    if (pos < 0 || pos >= Ring.count[k]) return -1;
    return slotFromTail(k, pos);
}

double *MSXsegStorage_pipeC(int k, int slot)
{
    int index = validActiveSlot(k, slot);
    if (index < 0) return NULL;
    return Ring.c + (size_t)index * (size_t)Ring.stride;
}

double *MSXsegStorage_pipeLastC(int k, int slot)
{
    int index = validActiveSlot(k, slot);
    if (index < 0) return NULL;
    return Ring.lastc + (size_t)index * (size_t)Ring.stride;
}

double *MSXsegStorage_pipeVPtr(int k, int slot)
{
    int index = validActiveSlot(k, slot);
    if (index < 0) return NULL;
    return Ring.v + index;
}

double *MSXsegStorage_pipeHstepPtr(int k, int slot)
{
    int index = validActiveSlot(k, slot);
    if (index < 0) return NULL;
    return Ring.hstep + index;
}

double *MSXsegStorage_pipeHresponsePtr(int k, int slot)
{
    int index = validActiveSlot(k, slot);
    if (index < 0) return NULL;
    return Ring.hresponse + index;
}

double *MSXsegStorage_pipeUresponsePtr(int k, int slot)
{
    int index = validActiveSlot(k, slot);
    if (index < 0) return NULL;
    return Ring.uresponse + index;
}

double *MSXsegStorage_pipeDresponsePtr(int k, int slot)
{
    int index = validActiveSlot(k, slot);
    if (index < 0) return NULL;
    return Ring.dresponse + index;
}

void MSXsegStorage_syncPsegMirror(int k)
{
    int pos;
    Pseg seg, prev;

    if (!MSXsegStorage_isPipeRingLink(k)) return;

    MSX.FirstSeg[k] = NULL;
    MSX.LastSeg[k] = NULL;
    prev = NULL;
    for (pos = 0; pos < Ring.count[k]; pos++)
    {
        seg = MSXsegStorage_pipeSegFromHead(k, pos);
        if (!seg) continue;
        seg->prev = NULL;
        seg->next = prev;
        if (prev) prev->prev = seg;
        else MSX.FirstSeg[k] = seg;
        prev = seg;
    }
    MSX.LastSeg[k] = prev;
    MSX.Link[k].nsegs = Ring.count[k];
}

void MSXsegStorage_syncAllPsegMirrors(void)
{
    int k;
    if (!Ring.opened) return;
    for (k = 1; k <= Ring.nLinks; k++)
        MSXsegStorage_syncPsegMirror(k);
}

void MSXsegStorage_syncScalarsFromPseg(int k)
{
    int pos, slot, index;
    Pseg seg;
    if (MSXsegStorage_isHybridLink(k))
        return;
    if (!MSXsegStorage_isPipeRingLink(k)) return;
    for (pos = 0; pos < Ring.count[k]; pos++)
    {
        slot = slotFromHead(k, pos);
        index = flatSlotIndex(k, slot);
        seg = Ring.seg[index];
        syncSegToSlot(seg);
    }
}

void MSXsegStorage_syncAllScalarsFromPseg(void)
{
    int k;
    /* GPU React unpacks hstep and other scalar state into the linked Pseg
       views.  Hybrid Core concentrations already alias dense rows, but its
       scalar arrays must be committed before the next transport/react step. */
    if (MSXsegStorage_isHybridEnabled())
    {
        MSXsegStorage_hybridSyncAllScalars();
        return;
    }
    if (!Ring.opened) return;
    for (k = 1; k <= Ring.nLinks; k++)
        MSXsegStorage_syncScalarsFromPseg(k);
}

int MSXsegStorage_validate(int k)
{
    int pos, slot, index, n = 0;
    if (!MSXsegStorage_isPipeRingLink(k)) return 0;
    for (pos = 0; pos < Ring.count[k]; pos++)
    {
        slot = slotFromHead(k, pos);
        index = flatSlotIndex(k, slot);
        if (!Ring.used[index] || !Ring.seg[index])
        {
            snprintf(MSX.Msg, MAXLINE,
                     "Error 525 - PIPE_RING validate failed: pipe index=%d, cap=%d, active_segments=%d.",
                     k, Ring.cap, Ring.count[k]);
            ENwriteline(MSX.Msg);
            return ERR_PIPE_RING_CAPACITY;
        }
        n++;
    }
    if (n != Ring.count[k] || MSX.Link[k].nsegs != Ring.count[k])
    {
        snprintf(MSX.Msg, MAXLINE,
                 "Error 525 - PIPE_RING count mismatch: pipe index=%d, cap=%d, active_segments=%d.",
                 k, Ring.cap, Ring.count[k]);
        ENwriteline(MSX.Msg);
        return ERR_PIPE_RING_CAPACITY;
    }
    return 0;
}

int MSXsegStorage_validateAll(void)
{
    int k, errcode;
    if (!Ring.opened) return 0;
    for (k = 1; k <= Ring.nLinks; k++)
    {
        errcode = MSXsegStorage_validate(k);
        if (errcode) return errcode;
    }
    return 0;
}

int MSXsegStorage_ringCapacity(void)
{
    return Ring.opened ? Ring.cap : 0;
}

int MSXsegStorage_ringStride(void)
{
    return Ring.opened ? Ring.stride : 0;
}

int MSXsegStorage_ringLinkCount(void)
{
    return Ring.opened ? Ring.nLinks : 0;
}

size_t MSXsegStorage_ringSlotCount(void)
{
    if (!Ring.opened) return 0;
    return totalSlots();
}

int MSXsegStorage_pipeFlatIndex(int k, int slot)
{
    if (validActiveSlot(k, slot) < 0) return -1;
    return flatSlotIndex(k, slot);
}

double *MSXsegStorage_ringCData(void)
{
    return Ring.opened ? Ring.c : NULL;
}

double *MSXsegStorage_ringLastCData(void)
{
    return Ring.opened ? Ring.lastc : NULL;
}

double *MSXsegStorage_ringHstepData(void)
{
    return Ring.opened ? Ring.hstep : NULL;
}
