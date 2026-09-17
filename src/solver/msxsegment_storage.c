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

#include "msxsegment_storage.h"
#include "msxresident_core.h"
#include "epanet2.h"
#include "msxgpu.h"

extern MSXproject MSX;

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
    uint64_t *observerId;
    MSXResidentPayload *observerPayload;
    struct HybridResidentTx *residentTx;
    uint32_t initialCount;
    int initialPrepared;
} HybridPipe;

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
} HybridStorage;

static HybridStorage Hybrid = {0};
static MSXResidentStatus HybridResidentStatus = MSX_RESIDENT_OK;
static int hybridPipeGuard(const HybridPipe *p)
{
    return (p && p->fixedCap) ? p->guard : Hybrid.guard;
}
static int hybridNextSlot(const HybridPipe *p, int slot);
static void hybridReplaceInList(int k, Pseg oldseg, Pseg newseg);
static void hybridCopyBoundaryToSlot(int k, int slot, Pseg src);
extern void MSXqual_removeSeg(Pseg seg);

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
        return MSXresident_observePipe((uint32_t)k, NULL, NULL, 0, p->orient);
    }
    if (p->count > p->cap || !p->observerPayload || !p->observerId ||
        !hybridRefreshResidentSlots(p))
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
    if (Hybrid.pipe)
        for (k = 1; k <= Hybrid.nLinks; k++) hybridFreePipe(&Hybrid.pipe[k]);
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
    p->observerId = (uint64_t *)calloc(slots, sizeof(uint64_t));
    p->observerPayload = (MSXResidentPayload *)calloc(slots, sizeof(MSXResidentPayload));
    p->residentTx = (HybridResidentTx *)calloc(1, sizeof(HybridResidentTx));
    if (!p->used || !p->view || !p->c || !p->lastc || !p->v ||
        !p->hstep || !p->hresponse || !p->uresponse || !p->dresponse ||
        !p->residentSlot || !p->coreSlotForResident ||
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
    cap = MAX(16, MIN(MAX(16, MSX.MaxSegments), 128));
    for (k = 1; k <= Hybrid.nLinks; k++)
    {
        Hybrid.pipe[k].head = -1;
        Hybrid.pipe[k].tail = -1;
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
    Pseg *newview;
    unsigned char *newused;
    int *newResidentSlot, *newCoreSlotForResident;
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
    newObserverId = (uint64_t *)calloc((size_t)cap, sizeof(uint64_t));
    newObserverPayload = (MSXResidentPayload *)calloc((size_t)cap, sizeof(MSXResidentPayload));
    if (!newc || !newlastc || !newv || !newhstep || !newhresponse ||
        !newuresponse || !newdresponse || !newview || !newused ||
        !newResidentSlot || !newCoreSlotForResident ||
        !newObserverId || !newObserverPayload)
    {
        FREE(newc); FREE(newlastc); FREE(newv); FREE(newhstep);
        FREE(newhresponse); FREE(newuresponse); FREE(newdresponse);
        FREE(newview); FREE(newused); FREE(newResidentSlot); FREE(newCoreSlotForResident);
        FREE(newObserverId); FREE(newObserverPayload);
        return ERR_MEMORY;
    }
    oldc = p->c; oldlastc = p->lastc; oldv = p->v; oldhstep = p->hstep;
    oldhresponse = p->hresponse; olduresponse = p->uresponse;
    olddresponse = p->dresponse; oldview = p->view; oldused = p->used;
    oldResidentSlot = p->residentSlot;
    oldCoreSlotForResident = p->coreSlotForResident;
    for (i = 0; i < cap; ++i) newResidentSlot[i] = newCoreSlotForResident[i] = -1;
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
    FREE(p->observerId); FREE(p->observerPayload);
    p->c = newc; p->lastc = newlastc; p->v = newv; p->hstep = newhstep;
    p->hresponse = newhresponse; p->uresponse = newuresponse;
    p->dresponse = newdresponse; p->view = newview; p->used = newused;
    p->residentSlot = newResidentSlot;
    p->coreSlotForResident = newCoreSlotForResident;
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
    int m;
    if (!segment) return ERR_MEMORY;
    *segment = NULL;
    if (MSXresident_getSlotPayload(linkIndex, slot, generation, &payload) !=
        MSX_RESIDENT_OK) return ERR_PIPE_RING_CAPACITY;
    boundary = MSXqual_getFreeSeg(0.0, (double *)payload.c);
    if (!boundary) return ERR_MEMORY;
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
    if (!token || !(t = (HybridResidentTx *)token->opaque)) return;
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
        t->oldCore[i]->next = NULL; t->oldCore[i]->prev = NULL;
        t->oldCore[i]->inHybridCore = FALSE; t->oldCore[i]->ownerLink = 0;
        t->oldCore[i]->hybridId = 0; t->pipe->used[t->slot[i]] = FALSE;
    }
    if (down) down->prev = bottom; else MSX.FirstSeg[t->linkIndex] = bottom;
    if (up) up->next = top; else MSX.LastSeg[t->linkIndex] = top;
    t->pipe->count -= (int)t->count;
    if (!t->pipe->count) t->pipe->head = t->pipe->tail = -1;
    else if (!t->boundarySide) t->pipe->head = t->nextEndpoint;
    else t->pipe->tail = t->nextEndpoint;
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
}

static int hybridPromote(int k, Pseg boundary, int atHead)
{
    HybridPipe *p = &Hybrid.pipe[k];
    int slot, err;
    double timer = 0.0;
    int timing = MSXsegStorage_hybridTimingEnabled();
    if (timing) timer = MSXgpu_wallTimeMs();
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
    hybridCopyBoundaryToSlot(k, slot, boundary);
    p->used[slot] = TRUE;
    if (p->residentSlot) p->residentSlot[slot] = -1;
    p->count++;
    if (p->count == 1) p->head = p->tail = slot;
    else if (atHead) p->head = slot;
    else p->tail = slot;
    hybridReplaceInList(k, boundary, p->view[slot]);
    MSXqual_removeSeg(boundary);
    if (timing) MSXsegStorage_hybridTimingAddHandoff(MSXgpu_wallTimeMs() - timer);
    return 0;
}

static int hybridDemote(int k, int atHead)
{
    HybridPipe *p = &Hybrid.pipe[k];
    Pseg core, boundary;
    int slot;
    double timer = 0.0;
    int timing = MSXsegStorage_hybridTimingEnabled();
    if (timing) timer = MSXgpu_wallTimeMs();
    if (p->count <= 0) return 0;
    slot = atHead ? p->head : p->tail;
    core = p->view[slot];
    boundary = MSXqual_getFreeSeg(0.0, core->c);
    if (!boundary) return ERR_MEMORY;
    hybridCopySlotToBoundary(k, slot, boundary);
    hybridReplaceInList(k, core, boundary);
    core->inHybridCore = FALSE;
    core->ownerLink = 0;
    core->hybridId = 0;
    p->used[slot] = FALSE;
    if (p->residentSlot) p->residentSlot[slot] = -1;
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
    return 0;
}

static int hybridInitializeLink(int k)
{
    HybridPipe *p = &Hybrid.pipe[k];
    Pseg seg, firstCore, lastCore;
    int total = 0, i, err, down, up, count, guard = hybridPipeGuard(p);
    for (seg = MSX.FirstSeg[k]; seg; seg = seg->prev) total++;
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
        hybridFindCore(p, k, &firstCore, &lastCore, &down, &up, &count);
    }
    while (count > 0 && up < guard)
    {
        err = hybridDemote(k, FALSE);
        if (err) return err;
        hybridFindCore(p, k, &firstCore, &lastCore, &down, &up, &count);
    }
    while (count > 0 && down > guard)
    {
        seg = firstCore->next;
        if (!seg || seg->inHybridCore) break;
        err = hybridPromote(k, seg, TRUE);
        if (err) return err;
        hybridFindCore(p, k, &firstCore, &lastCore, &down, &up, &count);
    }
    while (count > 0 && up > guard)
    {
        seg = lastCore->prev;
        if (!seg || seg->inHybridCore) break;
        err = hybridPromote(k, seg, FALSE);
        if (err) return err;
        hybridFindCore(p, k, &firstCore, &lastCore, &down, &up, &count);
    }
    return 0;
}

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
        HybridResidentStatus = hybridResidentObserve(k);
        if (HybridResidentStatus != MSX_RESIDENT_OK) return ERR_PIPE_RING_CAPACITY;
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
    if (p->residentSlot) p->residentSlot[slot] = -1;
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
    int k, err;
    if (!MSXsegStorage_isHybridEnabled() || !Hybrid.opened) return;
    for (k = 1; k <= Hybrid.nLinks; k++)
    {
        err = hybridRebalanceLink(k);
        if (err) { MSX.ErrCode = err; return; }
        HybridResidentStatus = hybridResidentObserve(k);
        if (HybridResidentStatus != MSX_RESIDENT_OK) { MSX.ErrCode = ERR_PIPE_RING_CAPACITY; return; }
    }
}

void MSXsegStorage_hybridAfterListReorder(int k)
{
    HybridPipe *p;
    Pseg firstCore, lastCore;
    int down, up, count;
    if (!MSXsegStorage_isHybridLink(k)) return;
    p = &Hybrid.pipe[k];
    hybridFindCore(p, k, &firstCore, &lastCore, &down, &up, &count);
    if (count > 1) p->orient = -p->orient;
    HybridResidentStatus = hybridResidentObserve(k);
    if (HybridResidentStatus != MSX_RESIDENT_OK) MSX.ErrCode = ERR_PIPE_RING_CAPACITY;
}

void MSXsegStorage_hybridClear(int k)
{
    HybridPipe *p;
    Pseg seg, next;
    if (!MSXsegStorage_isHybridLink(k)) return;
    p = &Hybrid.pipe[k];
    seg = MSX.FirstSeg[k];
    while (seg)
    {
        next = seg->prev;
        if (seg->inHybridCore)
        {
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
    return MSXsegStorage_isHybridEnabled() && Hybrid.timingEnabled;
}

static void hybridTimingAdd(double *dst, double ms)
{
    if (dst && Hybrid.timingEnabled)
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
