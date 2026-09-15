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
#include "epanet2.h"

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

    unsigned char *used;
    Pseg *seg;
    int *head;
    int *tail;
    int *count;
    int *orient;
} PipeRingStorage;

static PipeRingStorage Ring = {0};

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
    clearRing();
}

void MSXsegStorage_reset(void)
{
    int k;

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
    return MSXsegStorage_pipeAppendTail(k, seg);
}

void MSXsegStorage_unbindSegment(Pseg seg)
{
    int k, slot, index;

    if (!seg || !seg->inPipeRing) return;

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
