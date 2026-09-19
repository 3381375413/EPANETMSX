#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include "msxresident_core_cuda.h"
/* Keep the implementation body private so the exported wrapper can reject
   duplicate endpoint selections before any CUDA staging or synchronization. */
#define MSXresidentGpu_fetchHandoffBatch MSXresidentGpu_fetchHandoffBatch_impl
extern "C" void MSXgpu_profileRecordAggregate(uint64_t queries,uint64_t cacheHits,uint64_t rebuilds);
#if defined(EPANETMSX_CUDA_ENABLED)
extern "C" int MSXgpu_profileDetailEnabled(void);
extern "C" double MSXgpu_wallTimeMs(void);
extern "C" void MSXgpu_profileRecordTransfer(int direction,int scope,uint64_t bytes,double apiMs);
#else
/* The standalone Phase 2b contract harness links this CUDA core directly,
   without the production MSX timing object.  Keep profiling a no-op there;
   the production build gets the real implementations above. */
static int MSXgpu_profileDetailEnabled(void){return 0;}
static double MSXgpu_wallTimeMs(void){return 0.0;}
static void MSXgpu_profileRecordTransfer(int direction,int scope,uint64_t bytes,double apiMs)
{(void)direction;(void)scope;(void)bytes;(void)apiMs;}
#endif
#define MSX_PROFILE_TRANSFER_H2D 1
#define MSX_PROFILE_TRANSFER_D2H 2
#define MSX_PROFILE_TRANSFER_SCOPE_OTHER 0
#define MSX_PROFILE_TRANSFER_SCOPE_PATCH 1
#define MSX_PROFILE_TRANSFER_SCOPE_HANDOFF 2
#define MSX_PROFILE_TRANSFER_SCOPE_HYD 3
#define MSX_PROFILE_TRANSFER_SCOPE_DIAGNOSTIC 4
#define MSX_PROFILE_TRANSFER_SCOPE_REACTED 5
#define MSX_PROFILE_TRANSFER_SCOPE_PATCH_DESCRIPTOR 6
#define MSX_PROFILE_TRANSFER_SCOPE_PATCH_STAGE 7
#define MSX_PROFILE_TRANSFER_SCOPE_PATCH_PAYLOAD 8
#define MSX_PROFILE_TRANSFER_SCOPE_ACTIVE 9
#define MSX_PROFILE_TRANSFER_SCOPE_FULL_SYNC 10
#define MSX_PROFILE_TRANSFER_SCOPE_AGGREGATE 11
typedef struct { uint32_t link,capacity,head,tail,count; int32_t orient; uint64_t epoch; } DDesc;
typedef struct { uint32_t link; DDesc descriptor; } DDescStage;
typedef struct { uint32_t link,row,generation,used,kind,payloadRow; uint64_t id; double v,h,hr,ur,dr; } Stage;
struct MSXResidentGpu {
    uint32_t n,slots,owned,stride,activeCount;
    int device,uploaded,poisoned,activePrepared,completionEnqueued,diagnostic,aggregateValid;
    int activeBuilding,activeSealed;
    MSXResidentGpuActiveWriter *activeWriter;
    uint64_t activeBuildSequence;
    /* B2 Hyd cache lifecycle: submitted is immutable while activePrepared;
       applied is published only by a successful finish. */
    int hydAppliedValid,hydPending;
    uint64_t stateVersion,aggregateVersion;
    uint64_t handoffH2DBytes,handoffH2DCalls,handoffD2HBytes,handoffD2HCalls;
    uint64_t hydH2DBytes,hydH2DCalls,hydCandidateComparisons,hydUploads,hydSkips;
    cudaStream_t stream;
    uint64_t *fetchSeen; uint64_t fetchStamp;
    uint64_t *activeSeen;
    uint32_t *cap,*base,*hu,*hg,*haPipe,*haRow,*haGen,*haNf,*haNj,*haNa,*haNr;
    uint32_t *daPipe,*daRow,*daNf,*daNj,*daNa,*daNr,*dBase;
    int *haErr,*daErr,*dError;
    uint64_t *he,*hi,*haEpoch;
    DDesc *hd,*dd; DDescStage *hPatchDesc,*dPatchDesc;
    Stage *hPatch,*hGather,*dPatchStage,*dGatherStage;
    double *hPatchC,*hPatchL,*hGatherC,*hGatherL,*dPatchC,*dPatchL,*dGatherC,*dGatherL;
    double *haVol,*haHyd,*hCandidateHyd,*hSubmittedHyd,*haH,*haLast,*haReacted;
    double *daVol,*daHyd,*dPipeHyd,*daH,*daLast,*daReacted;
    uint32_t hydLayout;
    uint32_t *du,*dg; uint64_t *di;
    double *dv,*dh,*dhr,*dur,*ddr,*dc,*dl,*dmass;
    MSXResidentGpuReduction *dred,*hAggLink,*dAggLink,c,aggregate;
    double *hAggMass,*dAggMass,*aggregateMass;
};
static cudaError_t trackedCopyAsync(MSXResidentGpu *g,void *dst,const void *src,
                                    size_t bytes,cudaMemcpyKind kind,
                                    int direction,int scope)
{
    double start = 0.0;
    int timed = MSXgpu_profileDetailEnabled();
    if (timed) start = MSXgpu_wallTimeMs();
    cudaError_t e = cudaMemcpyAsync(dst,src,bytes,kind,g->stream);
    if (g && scope == MSX_PROFILE_TRANSFER_SCOPE_HANDOFF)
    {
        if (direction == MSX_PROFILE_TRANSFER_H2D)
        { g->handoffH2DBytes += (uint64_t)bytes; g->handoffH2DCalls++; }
        else if (direction == MSX_PROFILE_TRANSFER_D2H)
        { g->handoffD2HBytes += (uint64_t)bytes; g->handoffD2HCalls++; }
    }
    if (g && scope == MSX_PROFILE_TRANSFER_SCOPE_HYD &&
        direction == MSX_PROFILE_TRANSFER_H2D)
    { g->hydH2DBytes += (uint64_t)bytes; g->hydH2DCalls++; }
    if (timed)
        MSXgpu_profileRecordTransfer(direction,scope,(uint64_t)bytes,
                                     MSXgpu_wallTimeMs()-start);
    return e;
}
static cudaError_t trackedCopy(MSXResidentGpu *g,void *dst,const void *src,
                               size_t bytes,cudaMemcpyKind kind,int direction,
                               int scope)
{
    double start = 0.0;
    int timed = MSXgpu_profileDetailEnabled();
    if (timed) start = MSXgpu_wallTimeMs();
    cudaError_t e = cudaMemcpy(dst,src,bytes,kind);
    if (g && scope == MSX_PROFILE_TRANSFER_SCOPE_HANDOFF)
    {
        if (direction == MSX_PROFILE_TRANSFER_H2D)
        {
            g->handoffH2DBytes += (uint64_t)bytes;
            g->handoffH2DCalls++;
        }
        else if (direction == MSX_PROFILE_TRANSFER_D2H)
        {
            g->handoffD2HBytes += (uint64_t)bytes;
            g->handoffD2HCalls++;
        }
    }
    if (g && scope == MSX_PROFILE_TRANSFER_SCOPE_HYD &&
        direction == MSX_PROFILE_TRANSFER_H2D)
    { g->hydH2DBytes += (uint64_t)bytes; g->hydH2DCalls++; }
    if (timed)
        MSXgpu_profileRecordTransfer(direction,scope,(uint64_t)bytes,
                                     MSXgpu_wallTimeMs()-start);
    return e;
}
static int copyScope(MSXResidentGpu *g,const void *dst,const void *src,
                     cudaMemcpyKind kind)
{
    if (!g) return MSX_PROFILE_TRANSFER_SCOPE_OTHER;
    if (dst == (const void *)g->dPatchStage)
        return MSX_PROFILE_TRANSFER_SCOPE_PATCH_STAGE;
    if (dst == (const void *)g->dPatchC || dst == (const void *)g->dPatchL)
        return MSX_PROFILE_TRANSFER_SCOPE_PATCH_PAYLOAD;
    if (dst == (const void *)g->dPatchDesc)
        return MSX_PROFILE_TRANSFER_SCOPE_PATCH_DESCRIPTOR;
    if (dst == (const void *)g->dGatherStage ||
        dst == (const void *)g->hGather ||
        dst == (const void *)g->hGatherC || dst == (const void *)g->hGatherL)
        return g->activeCount ? MSX_PROFILE_TRANSFER_SCOPE_OTHER :
                                MSX_PROFILE_TRANSFER_SCOPE_HANDOFF;
    if (dst == (const void *)g->daHyd)
        return MSX_PROFILE_TRANSFER_SCOPE_HYD;
    if (dst == (const void *)g->dPipeHyd)
        return MSX_PROFILE_TRANSFER_SCOPE_HYD;
    if (dst == (const void *)g->daPipe || dst == (const void *)g->daRow ||
        dst == (const void *)g->daVol)
        return MSX_PROFILE_TRANSFER_SCOPE_ACTIVE;
    if (dst == (const void *)g->hAggLink || dst == (const void *)g->hAggMass)
        return MSX_PROFILE_TRANSFER_SCOPE_AGGREGATE;
    if (dst == (const void *)g->dError || src == (const void *)g->dError ||
        dst == (const void *)g->haLast ||
        dst == (const void *)g->haNf || dst == (const void *)g->haNj ||
        dst == (const void *)g->haNa || dst == (const void *)g->haNr)
        return MSX_PROFILE_TRANSFER_SCOPE_DIAGNOSTIC;
    if (dst == (const void *)g->haReacted || src == (const void *)g->daReacted)
        return MSX_PROFILE_TRANSFER_SCOPE_REACTED;
    (void)kind;
    return MSX_PROFILE_TRANSFER_SCOPE_OTHER;
}
#define cudaMemcpy(dst,src,bytes,kind) \
    trackedCopy((MSXResidentGpu *)g,(dst),(src),(bytes),(kind), \
        ((kind) == cudaMemcpyHostToDevice ? MSX_PROFILE_TRANSFER_H2D : MSX_PROFILE_TRANSFER_D2H), \
        copyScope((MSXResidentGpu *)g,(dst),(src),(kind)) )
typedef enum { V_OK,V_DESCRIPTOR,V_CAPACITY,V_GENERATION,V_EPOCH } VStatus;
static int ck(cudaError_t e){return e==cudaSuccess;} static int mul(size_t a,size_t b,size_t*r){if(a&&b>SIZE_MAX/a)return 0;*r=a*b;return 1;} static int da(void**p,size_t n){return ck(cudaMalloc(p,n));}
static int use(MSXResidentGpu*g){int now=-1;if(!g||!ck(cudaGetDevice(&now))||(!((now==g->device)||ck(cudaSetDevice(g->device)))))return 0;if(!g->stream&&!ck(cudaStreamCreateWithFlags(&g->stream,cudaStreamNonBlocking)))return 0;return 1;}
static void invalidateHydCache(MSXResidentGpu *g)
{
    if (!g) return;
    g->hydAppliedValid = 0;
    g->hydPending = 0;
}
static MSXResidentStatus badgpu(MSXResidentGpu*g,MSXResidentStatus s)
{
    if(g)
    {
        /* Failure is the only path allowed to pay an unconditional drain.
           It prevents close/abort from freeing pinned or device buffers still
           referenced by work already accepted by the resident stream. */
        if(g->stream) (void)cudaStreamSynchronize(g->stream);
        invalidateHydCache(g);
        g->c.cudaErrors++;
        if(!g->poisoned)g->c.cudaPoisons++;
        g->poisoned=1;g->activePrepared=0;g->completionEnqueued=0;
    }
    return s;
}
static void freePinned(void *p){if(p)(void)cudaFreeHost(p);}
static int hydTableBytes(uint32_t nLinks, size_t *bytes)
{
    size_t rows, values;
    if (!bytes || !mul((size_t)nLinks + 1u, (size_t)MSX_RESIDENT_HYD_STRIDE, &values) ||
        !mul(values, sizeof(double), &rows)) return 0;
    *bytes = rows;
    return 1;
}
static void gone(MSXResidentGpu*g)
{
    if(!g)return;
    if(g->stream)cudaStreamDestroy(g->stream);
    cudaFree(g->dd);cudaFree(g->dPatchDesc);cudaFree(g->dPatchStage);cudaFree(g->dGatherStage);
    cudaFree(g->dPatchC);cudaFree(g->dPatchL);cudaFree(g->dGatherC);cudaFree(g->dGatherL);
    cudaFree(g->du);cudaFree(g->dg);cudaFree(g->di);cudaFree(g->dv);cudaFree(g->dh);
    cudaFree(g->dhr);cudaFree(g->dur);cudaFree(g->ddr);cudaFree(g->dc);cudaFree(g->dl);
    cudaFree(g->dmass);cudaFree(g->dred);cudaFree(g->dAggLink);cudaFree(g->dAggMass);
    cudaFree(g->dBase);cudaFree(g->dError);cudaFree(g->daPipe);cudaFree(g->daRow);
    cudaFree(g->daVol);cudaFree(g->daHyd);cudaFree(g->dPipeHyd);cudaFree(g->daH);cudaFree(g->daNf);
    cudaFree(g->daNj);cudaFree(g->daNa);cudaFree(g->daNr);cudaFree(g->daErr);
    cudaFree(g->daLast);cudaFree(g->daReacted);
    freePinned(g->hd);freePinned(g->hPatchDesc);freePinned(g->hPatch);
    freePinned(g->hGather);freePinned(g->hPatchC);freePinned(g->hPatchL);
    freePinned(g->hGatherC);freePinned(g->hGatherL);
    freePinned(g->haPipe);freePinned(g->haRow);freePinned(g->haNf);
    freePinned(g->haNj);freePinned(g->haNa);freePinned(g->haNr);
    freePinned(g->haErr);freePinned(g->haVol);freePinned(g->haHyd);free(g->hCandidateHyd);freePinned(g->hSubmittedHyd);
    freePinned(g->haLast);freePinned(g->haReacted);
    free(g->cap);free(g->base);free(g->hu);free(g->hg);free(g->he);free(g->hi);
    free(g->fetchSeen);free(g->activeSeen);free(g->haGen);free(g->haEpoch);free(g->haH);
    freePinned(g->hAggLink);freePinned(g->hAggMass);free(g->aggregateMass);free(g);
}
extern "C" int MSXresidentGpu_isEnabled(void){return 1;}
extern "C" MSXResidentStatus MSXresidentGpu_open(const MSXResidentGpuOpen*o,MSXResidentGpu**out){MSXResidentGpu*g;size_t lb,rb,pipeHydBytes,owned=0;if(!out||*out||!o||!o->nLinks||!o->totalSlots||!o->speciesStride||!o->capacity||!o->base)return MSX_RESIDENT_ERR_ARGUMENT;if(!ck(cudaSetDevice(0))||!ck(cudaFree(0)))return MSX_RESIDENT_ERR_GPU;if(!mul((size_t)o->nLinks+1,sizeof(uint32_t),&lb)||!mul((size_t)o->totalSlots,o->speciesStride,&rb)||!mul(rb,sizeof(double),&rb)||!hydTableBytes(o->nLinks,&pipeHydBytes))return MSX_RESIDENT_ERR_OVERFLOW;for(uint32_t k=1;k<=o->nLinks;k++){if(!o->capacity[k]||o->base[k]>o->totalSlots||o->capacity[k]>o->totalSlots-o->base[k]||owned>o->totalSlots-o->capacity[k])return MSX_RESIDENT_ERR_ARGUMENT;owned+=o->capacity[k];for(uint32_t q=1;q<k;q++)if(o->base[k]<o->base[q]+o->capacity[q]&&o->base[q]<o->base[k]+o->capacity[k])return MSX_RESIDENT_ERR_ARGUMENT;}g=(MSXResidentGpu*)calloc(1,sizeof(*g));if(!g)return MSX_RESIDENT_ERR_MEMORY;g->device=0;g->n=o->nLinks;g->slots=o->totalSlots;g->owned=(uint32_t)owned;g->stride=o->speciesStride;g->cap=(uint32_t*)calloc(g->n+1,sizeof(uint32_t));g->base=(uint32_t*)calloc(g->n+1,sizeof(uint32_t));g->hu=(uint32_t*)calloc(g->slots,sizeof(uint32_t));g->hg=(uint32_t*)calloc(g->slots,sizeof(uint32_t));g->he=(uint64_t*)calloc(g->n+1,sizeof(uint64_t));g->hi=(uint64_t*)calloc(g->slots,sizeof(uint64_t));g->fetchSeen=(uint64_t*)calloc(g->slots,sizeof(uint64_t));g->activeSeen=(uint64_t*)calloc(g->slots,sizeof(uint64_t));if(!g->cap||!g->base||!g->hu||!g->hg||!g->he||!g->hi||!g->fetchSeen||!g->activeSeen)goto mem;memcpy(g->cap,o->capacity,lb);memcpy(g->base,o->base,lb);if(!ck(cudaHostAlloc(&g->hd,((size_t)g->n+1)*sizeof(DDesc),0))||!ck(cudaHostAlloc(&g->hPatch,(size_t)g->slots*sizeof(Stage),0))||!ck(cudaHostAlloc(&g->hGather,(size_t)g->slots*sizeof(Stage),0))||!ck(cudaHostAlloc(&g->hPatchC,rb,0))||!ck(cudaHostAlloc(&g->hPatchL,rb,0))||!ck(cudaHostAlloc(&g->hGatherC,rb,0))||!ck(cudaHostAlloc(&g->hGatherL,rb,0)))goto mem;
if(!ck(cudaHostAlloc(&g->hPatchDesc,(size_t)g->n*sizeof(DDescStage),0)))goto mem;
#define A(x,n) if(!da((void**)&g->x,n))goto mem
A(dPatchDesc,(size_t)g->n*sizeof(DDescStage));
A(dd,((size_t)g->n+1)*sizeof(DDesc));A(dPatchStage,(size_t)g->slots*sizeof(Stage));A(dGatherStage,(size_t)g->slots*sizeof(Stage));A(dPatchC,rb);A(dPatchL,rb);A(dGatherC,rb);A(dGatherL,rb);A(du,(size_t)g->slots*sizeof(uint32_t));A(dg,(size_t)g->slots*sizeof(uint32_t));A(di,(size_t)g->slots*sizeof(uint64_t));A(dv,(size_t)g->slots*sizeof(double));A(dh,(size_t)g->slots*sizeof(double));A(dhr,(size_t)g->slots*sizeof(double));A(dur,(size_t)g->slots*sizeof(double));A(ddr,(size_t)g->slots*sizeof(double));A(dc,rb);A(dl,rb);A(dmass,(size_t)g->stride*sizeof(double));A(dred,sizeof(*g->dred));
#undef A
g->haGen=(uint32_t*)calloc(g->owned,sizeof(uint32_t));
/* Validation addresses haNf by global slot and haNj by one-based link index.
   Active request/return arrays are pinned because their async transfers are
   part of the resident stream lifecycle and must complete before readyEvent. */
if(!ck(cudaHostAlloc(&g->haPipe,(size_t)g->owned*sizeof(uint32_t),0))||
   !ck(cudaHostAlloc(&g->haRow,(size_t)g->owned*sizeof(uint32_t),0))||
   !ck(cudaHostAlloc(&g->haNf,(size_t)g->slots*sizeof(uint32_t),0))||
   !ck(cudaHostAlloc(&g->haNj,(size_t)(g->owned>g->n?g->owned:g->n+1)*sizeof(uint32_t),0))||
   !ck(cudaHostAlloc(&g->haNa,(size_t)g->owned*sizeof(uint32_t),0))||
   !ck(cudaHostAlloc(&g->haNr,(size_t)g->owned*sizeof(uint32_t),0))||
   !ck(cudaHostAlloc(&g->haErr,sizeof(int),0))||
   !ck(cudaHostAlloc(&g->haVol,(size_t)g->owned*sizeof(double),0))||
   !ck(cudaHostAlloc(&g->haHyd,(size_t)g->owned*MSX_RESIDENT_HYD_STRIDE*sizeof(double),0))||
   !ck(cudaHostAlloc(&g->hSubmittedHyd,pipeHydBytes,0))||
   !ck(cudaHostAlloc(&g->haLast,(size_t)g->owned*sizeof(double),0))||
   !ck(cudaHostAlloc(&g->haReacted,(size_t)(g->n+1)*g->stride*sizeof(double),0)))
    goto mem;
g->hCandidateHyd=(double*)calloc((size_t)(g->n+1)*MSX_RESIDENT_HYD_STRIDE,sizeof(double));
if(!g->hCandidateHyd)goto mem;
/* Every compared byte is initialized.  The separate valid bit still forces
   the first reaction to upload even when the source table is all +0. */
memset(g->hSubmittedHyd,0,pipeHydBytes);
memset(g->haNf,0,(size_t)g->slots*sizeof(uint32_t));
memset(g->haNj,0,(size_t)(g->owned>g->n?g->owned:g->n+1)*sizeof(uint32_t));
g->haEpoch=(uint64_t*)calloc(g->owned,sizeof(uint64_t));
g->haH=(double*)calloc(g->owned,sizeof(double));
if(!ck(cudaHostAlloc(&g->hAggLink,((size_t)g->n+1)*sizeof(MSXResidentGpuReduction),0))||
   !ck(cudaHostAlloc(&g->hAggMass,(size_t)(g->n+1)*g->stride*sizeof(double),0)))
    goto mem;
g->aggregateMass=(double*)calloc(g->stride,sizeof(double));
if(!g->haGen||!g->haEpoch||!g->haH||!g->aggregateMass)goto mem;
if(!da((void**)&g->daPipe,(size_t)g->owned*sizeof(uint32_t))||!da((void**)&g->daRow,(size_t)g->owned*sizeof(uint32_t))||!da((void**)&g->daVol,(size_t)g->owned*sizeof(double))||!da((void**)&g->daHyd,(size_t)g->owned*MSX_RESIDENT_HYD_STRIDE*sizeof(double))||!da((void**)&g->dPipeHyd,pipeHydBytes)||!da((void**)&g->daH,(size_t)g->owned*sizeof(double))||!da((void**)&g->daNf,(size_t)g->owned*sizeof(uint32_t))||!da((void**)&g->daNj,(size_t)g->owned*sizeof(uint32_t))||!da((void**)&g->daNa,(size_t)g->owned*sizeof(uint32_t))||!da((void**)&g->daNr,(size_t)g->owned*sizeof(uint32_t))||!da((void**)&g->daErr,(size_t)g->owned*sizeof(int))||!da((void**)&g->dError,sizeof(int))||!da((void**)&g->dAggLink,((size_t)g->n+1)*sizeof(MSXResidentGpuReduction))||!da((void**)&g->dAggMass,((size_t)g->n+1)*g->stride*sizeof(double))||!da((void**)&g->dBase,((size_t)g->n+1)*sizeof(uint32_t))||!da((void**)&g->daLast,(size_t)g->owned*sizeof(double))||!da((void**)&g->daReacted,((size_t)g->n+1)*g->stride*sizeof(double)))goto mem;
if(!ck(cudaMemcpy(g->dBase,g->base,((size_t)g->n+1)*sizeof(uint32_t),cudaMemcpyHostToDevice)))goto mem;
*out=g;return MSX_RESIDENT_OK;mem:gone(g);return MSX_RESIDENT_ERR_MEMORY;}
/* haNf/haNj are preallocated active-result buffers.  They are only scratch
   while no active batch is prepared, so validation needs no steady-state
   allocation.  Each global row maps to one array element. */
static VStatus valid(MSXResidentGpu*g,const MSXResidentPatchBatch*b,int initial)
{
    VStatus z = V_OK;
    uint32_t i;
    if (!g || !b || (!b->descriptor && b->descriptorCount) ||
        (!b->slot && b->slotCount))
        return V_DESCRIPTOR;
    if (initial && (b->descriptorCount != g->n ||
                    b->slotCount != g->owned))
    {
        z = V_DESCRIPTOR;
        goto done;
    }
    for (i = 0; i < b->descriptorCount; ++i)
    {
        const MSXResidentDescriptorPatch *x = &b->descriptor[i];
        const MSXResidentPipeDesc *d = &x->descriptor;
        uint32_t k = x->linkIndex;
        if (!k || k > g->n || d->linkIndex != k ||
            d->capacity != g->cap[k] || d->orient < -1 || d->orient > 1 ||
            !d->orient || d->count > d->capacity ||
            (d->count && (d->head >= d->capacity || d->tail >= d->capacity)) ||
            (!d->count && (d->head != UINT32_MAX || d->tail != UINT32_MAX)))
        {
            z = V_DESCRIPTOR;
            goto done;
        }
        if (!initial && d->epoch < g->he[k])
        {
            z = V_EPOCH;
            goto done;
        }
        if (g->haNj[k])
        {
            z = V_DESCRIPTOR;
            goto done;
        }
        g->haNj[k] = i + 1;
    }
    for (i = 0; i < b->slotCount; ++i)
    {
        const MSXResidentSlotPatch *x = &b->slot[i];
        MSXResidentPatchKind kind = x->kind ? x->kind :
            (x->used ? MSX_RESIDENT_PATCH_IMPORT : MSX_RESIDENT_PATCH_INVALIDATE);
        uint32_t r;
        if (!x->linkIndex || x->linkIndex > g->n ||
            x->slot >= g->cap[x->linkIndex])
        {
            z = V_CAPACITY;
            goto done;
        }
        if (kind < MSX_RESIDENT_PATCH_IMPORT || kind > MSX_RESIDENT_PATCH_META ||
            (kind == MSX_RESIDENT_PATCH_INVALIDATE && x->used) ||
            (kind != MSX_RESIDENT_PATCH_INVALIDATE && !x->used) ||
            (kind != MSX_RESIDENT_PATCH_INVALIDATE && !x->generation) ||
            (kind == MSX_RESIDENT_PATCH_IMPORT &&
                                (!x->payload.c || !x->payload.lastc)))
        {
            z = V_DESCRIPTOR;
            goto done;
        }
        r = g->base[x->linkIndex] + x->slot;
        if (!initial && x->generation < g->hg[r])
        {
            z = V_GENERATION;
            goto done;
        }
        if (g->haNf[r])
        {
            z = V_DESCRIPTOR;
            goto done;
        }
        g->haNf[r] = i + 1;
    }
    for (i = 0; i < b->descriptorCount; ++i)
    {
        const MSXResidentPipeDesc *d = &b->descriptor[i].descriptor;
        uint32_t c = 0, k = d->linkIndex, s;
        for (s = 0; s < d->capacity; ++s)
        {
            uint32_t row = g->base[k] + s;
            uint32_t p = g->haNf[row];
            c += p ? b->slot[p - 1].used : g->hu[row];
        }
        if (c != d->count)
        {
            z = V_DESCRIPTOR;
            goto done;
        }
        if (d->count)
        {
            s = d->head;
            for (uint32_t j = 0; j < d->count; ++j)
            {
                uint32_t p = g->haNf[g->base[k] + s];
                if (!(p ? b->slot[p - 1].used :
                        g->hu[g->base[k] + s]))
                {
                    z = V_DESCRIPTOR;
                    goto done;
                }
                s = (uint32_t)(((int64_t)s + d->orient +
                                (int64_t)d->capacity) % d->capacity);
            }
            if (s != (uint32_t)(((int64_t)d->tail + d->orient +
                                 (int64_t)d->capacity) % d->capacity))
            {
                z = V_DESCRIPTOR;
                goto done;
            }
        }
    }
    if (initial)
        for (uint32_t k = 1; k <= g->n; ++k)
        {
            if (!g->haNj[k])
            {
                z = V_DESCRIPTOR;
                goto done;
            }
            for (uint32_t s = 0; s < g->cap[k]; ++s)
                if (!g->haNf[g->base[k] + s])
                {
                    z = V_DESCRIPTOR;
                    goto done;
                }
        }
done:
    for (i = 0; i < b->descriptorCount; ++i)
        if (b->descriptor[i].linkIndex &&
            b->descriptor[i].linkIndex <= g->n)
            g->haNj[b->descriptor[i].linkIndex] = 0;
    for (i = 0; i < b->slotCount; ++i)
        if (b->slot[i].linkIndex && b->slot[i].linkIndex <= g->n &&
            b->slot[i].slot < g->cap[b->slot[i].linkIndex])
            g->haNf[g->base[b->slot[i].linkIndex] + b->slot[i].slot] = 0;
    return z;
}
static MSXResidentStatus invalid(MSXResidentGpu*g,VStatus v){if(v==V_CAPACITY){g->c.capacityReject++;g->c.capacityOverflow++;return MSX_RESIDENT_ERR_CAPACITY;}if(v==V_GENERATION){g->c.generationReject++;g->c.staleGeneration++;return MSX_RESIDENT_ERR_GENERATION;}if(v==V_EPOCH){g->c.epochStale++;g->c.epochMismatch++;return MSX_RESIDENT_ERR_GENERATION;}g->c.descriptorReject++;return MSX_RESIDENT_ERR_ARGUMENT;}
__global__ static void scatter(const Stage*s,uint32_t n,uint32_t st,uint32_t*u,uint32_t*g,uint64_t*id,double*v,double*h,double*hr,double*ur,double*dr,const double*pc,const double*pl,double*c,double*l){uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){Stage x=s[i];uint32_t r=x.row;u[r]=x.used;g[r]=x.generation;id[r]=x.used?x.id:0;v[r]=x.used?x.v:0;h[r]=x.used?x.h:0;hr[r]=x.used?x.hr:0;ur[r]=x.used?x.ur:0;dr[r]=x.used?x.dr:0;if(x.kind==MSX_RESIDENT_PATCH_IMPORT)for(uint32_t m=0;m<st;m++){c[(size_t)r*st+m]=pc[(size_t)x.payloadRow*st+m];l[(size_t)r*st+m]=pl[(size_t)x.payloadRow*st+m];}}}
__global__ static void scatterDescriptors(const DDescStage*s,uint32_t n,DDesc*d){uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)d[s[i].link]=s[i].descriptor;}
__global__ static void gather(const Stage*q,uint32_t n,uint32_t st,const uint32_t*u,const uint32_t*g,const uint64_t*id,const double*v,const double*h,const double*hr,const double*ur,const double*dr,const double*c,const double*l,Stage*o,double*oc,double*ol){uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){uint32_t r=q[i].row;Stage x=q[i];x.used=u[r];x.generation=g[r];x.id=id[r];x.v=v[r];x.h=h[r];x.hr=hr[r];x.ur=ur[r];x.dr=dr[r];o[i]=x;for(uint32_t m=0;m<st;m++){oc[(size_t)i*st+m]=c[(size_t)r*st+m];ol[(size_t)i*st+m]=l[(size_t)r*st+m];}}}
#define MSX_RESIDENT_RED_THREADS 128
#define MSX_RESIDENT_MASS_TILE 8
/* One block owns one link.  Every thread walks a fixed strided subset of the
   descriptor, then pairwise shared-memory reductions produce a reproducible
   FP64 result.  Species are reduced in small tiles so the shared footprint
   stays bounded even when the MSX species count grows.  There are no global
   atomics and no per-link serial thread. */
__global__ static void redLink(uint32_t nLinks,uint32_t st,const DDesc*d,
    const uint32_t*base,const uint32_t*u,const uint64_t*id,const uint32_t*g,
    const double*v,const double*c,MSXResidentGpuReduction*out,double*m)
{
    const uint32_t k=blockIdx.x+1, tid=threadIdx.x;
    if(blockIdx.x>=nLinks || tid>=MSX_RESIDENT_RED_THREADS)return;
    extern __shared__ unsigned char raw[];
    MSXResidentGpuReduction *sr=(MSXResidentGpuReduction*)raw;
    double *sm=(double*)(sr+MSX_RESIDENT_RED_THREADS);
    const DDesc *p=&d[k];
    MSXResidentGpuReduction z={};
    z.firstErrorSlot=UINT_MAX;
    for(uint32_t j=tid;j<p->count;j+=MSX_RESIDENT_RED_THREADS){
        uint32_t s=p->orient>0?(p->head+j)%p->capacity:
            (p->head+p->capacity-(j%p->capacity))%p->capacity;
        uint32_t q=base[k]+s;
        if(!u[q])continue;
        z.activeCount++;
        z.activeVolume+=v[q];
        z.checksumXor^=id[q]^g[q];
        z.checksumSum+=id[q]+g[q];
        for(uint32_t x=0;x<st;x++){
            double y=c[(size_t)q*st+x];
            if(isnan(y)){z.nanCount++;z.errorCount++;if(z.firstErrorSlot==UINT_MAX)z.firstErrorSlot=q;}
            else if(isinf(y)){z.infCount++;z.errorCount++;if(z.firstErrorSlot==UINT_MAX)z.firstErrorSlot=q;}
        }
    }
    sr[tid]=z;
    __syncthreads();
    for(uint32_t span=MSX_RESIDENT_RED_THREADS/2;span;span>>=1){
        if(tid<span){
            MSXResidentGpuReduction *a=&sr[tid],*b=&sr[tid+span];
            a->activeCount+=b->activeCount;
            a->activeVolume+=b->activeVolume;
            a->checksumXor^=b->checksumXor;
            a->checksumSum+=b->checksumSum;
            a->nanCount+=b->nanCount;
            a->infCount+=b->infCount;
            a->errorCount+=b->errorCount;
            if(b->firstErrorSlot<a->firstErrorSlot)a->firstErrorSlot=b->firstErrorSlot;
        }
        __syncthreads();
    }
    if(tid==0)out[k]=sr[0];
    for(uint32_t first=0;first<st;first+=MSX_RESIDENT_MASS_TILE){
        double local[MSX_RESIDENT_MASS_TILE]={0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
        for(uint32_t j=tid;j<p->count;j+=MSX_RESIDENT_RED_THREADS){
            uint32_t s=p->orient>0?(p->head+j)%p->capacity:
                (p->head+p->capacity-(j%p->capacity))%p->capacity;
            uint32_t q=base[k]+s;
            if(!u[q])continue;
            for(uint32_t x=0;x<MSX_RESIDENT_MASS_TILE && first+x<st;x++)
                local[x]+=c[(size_t)q*st+first+x]*v[q];
        }
        for(uint32_t x=0;x<MSX_RESIDENT_MASS_TILE;x++)sm[tid*MSX_RESIDENT_MASS_TILE+x]=local[x];
        __syncthreads();
        for(uint32_t span=MSX_RESIDENT_RED_THREADS/2;span;span>>=1){
            if(tid<span)for(uint32_t x=0;x<MSX_RESIDENT_MASS_TILE;x++)
                sm[tid*MSX_RESIDENT_MASS_TILE+x]+=sm[(tid+span)*MSX_RESIDENT_MASS_TILE+x];
            __syncthreads();
        }
        if(tid==0)for(uint32_t x=0;x<MSX_RESIDENT_MASS_TILE && first+x<st;x++)
            m[(size_t)k*st+first+x]=sm[x];
        __syncthreads();
    }
}
static MSXResidentStatus patch(MSXResidentGpu*g,const MSXResidentPatchBatch*b,int init)
{
    uint32_t i, m, payloadCount = 0; VStatus v;
    if (!use(g)) return badgpu(g, MSX_RESIDENT_ERR_GPU);
    if (!g || g->poisoned) return MSX_RESIDENT_ERR_POISONED;
    v = valid(g, b, init); if (v != V_OK) return invalid(g, v);
    /* Pack every dirty descriptor before the first device mutation.  The
       staging record carries its link index, so one H2D API call can feed a
       device scatter instead of issuing one cudaMemcpy per descriptor. */
    for (i = 0; i < b->descriptorCount; ++i)
    {
        const MSXResidentPipeDesc *d = &b->descriptor[i].descriptor;
        g->hPatchDesc[i].link = d->linkIndex;
        g->hPatchDesc[i].descriptor = {d->linkIndex, d->capacity, d->head,
                                       d->tail, d->count, d->orient, d->epoch};
    }
    for (i = 0; i < b->slotCount; ++i)
    {
        const MSXResidentSlotPatch *x = &b->slot[i]; Stage *s = &g->hPatch[i];
        MSXResidentPatchKind kind = x->kind ? x->kind :
            (x->used ? MSX_RESIDENT_PATCH_IMPORT : MSX_RESIDENT_PATCH_INVALIDATE);
        if (kind < MSX_RESIDENT_PATCH_IMPORT || kind > MSX_RESIDENT_PATCH_META ||
            (kind == MSX_RESIDENT_PATCH_INVALIDATE && x->used) ||
            (kind != MSX_RESIDENT_PATCH_INVALIDATE && !x->used))
            return invalid(g, V_DESCRIPTOR);
        s->link=x->linkIndex; s->row=g->base[x->linkIndex]+x->slot;
        s->generation=x->generation; s->used=x->used; s->kind=(uint32_t)kind;
        s->payloadRow=payloadCount; s->id=x->payload.parcelId; s->v=x->payload.volume;
        s->h=x->payload.hstep; s->hr=x->payload.hresponse; s->ur=x->payload.uresponse;
        s->dr=x->payload.dresponse;
        if (kind == MSX_RESIDENT_PATCH_IMPORT)
        {
            if (!x->payload.c || !x->payload.lastc) return invalid(g, V_DESCRIPTOR);
            for (m=0;m<g->stride;m++)
            { g->hPatchC[(size_t)payloadCount*g->stride+m]=x->payload.c[m];
              g->hPatchL[(size_t)payloadCount*g->stride+m]=x->payload.lastc[m]; }
            ++payloadCount;
        }
    }
    if (b->slotCount &&
        !ck(trackedCopyAsync(g,g->dPatchStage,g->hPatch,(size_t)b->slotCount*sizeof(Stage),
                             cudaMemcpyHostToDevice,MSX_PROFILE_TRANSFER_H2D,
                             MSX_PROFILE_TRANSFER_SCOPE_PATCH_STAGE)))
        return badgpu(g, MSX_RESIDENT_ERR_TRANSFER);
    if (payloadCount &&
        (!ck(trackedCopyAsync(g,g->dPatchC,g->hPatchC,(size_t)payloadCount*g->stride*sizeof(double),
                              cudaMemcpyHostToDevice,MSX_PROFILE_TRANSFER_H2D,
                              MSX_PROFILE_TRANSFER_SCOPE_PATCH_PAYLOAD)) ||
         !ck(trackedCopyAsync(g,g->dPatchL,g->hPatchL,(size_t)payloadCount*g->stride*sizeof(double),
                              cudaMemcpyHostToDevice,MSX_PROFILE_TRANSFER_H2D,
                              MSX_PROFILE_TRANSFER_SCOPE_PATCH_PAYLOAD))))
        return badgpu(g, MSX_RESIDENT_ERR_TRANSFER);
    if (b->slotCount)
    {
        scatter<<<(b->slotCount+255)/256,256,0,g->stream>>>(g->dPatchStage,b->slotCount,g->stride,
            g->du,g->dg,g->di,g->dv,g->dh,g->dhr,g->dur,g->ddr,
            g->dPatchC,g->dPatchL,g->dc,g->dl);
        if (!ck(cudaGetLastError()))
            return badgpu(g, MSX_RESIDENT_ERR_GPU);
    }
    if (b->descriptorCount &&
        !ck(trackedCopyAsync(g,g->dPatchDesc, g->hPatchDesc,
                             (size_t)b->descriptorCount * sizeof(DDescStage),
                             cudaMemcpyHostToDevice, MSX_PROFILE_TRANSFER_H2D,
                             MSX_PROFILE_TRANSFER_SCOPE_PATCH_DESCRIPTOR)))
        return badgpu(g, MSX_RESIDENT_ERR_TRANSFER);
    if (b->descriptorCount)
    {
        scatterDescriptors<<<(b->descriptorCount + 255) / 256, 256, 0, g->stream>>>(
            g->dPatchDesc, b->descriptorCount, g->dd);
        if (!ck(cudaGetLastError()))
            return badgpu(g, MSX_RESIDENT_ERR_GPU);
    }
    for (i = 0; i < b->descriptorCount; ++i)
    {
        const MSXResidentPipeDesc *d = &b->descriptor[i].descriptor;
        g->hd[d->linkIndex] = {d->linkIndex, d->capacity, d->head, d->tail,
                               d->count, d->orient, d->epoch};
    }
    for (i=0;i<b->slotCount;i++){Stage*s=&g->hPatch[i];g->hu[s->row]=s->used;g->hg[s->row]=s->generation;g->hi[s->row]=s->used?s->id:0;}
    for (i=0;i<b->descriptorCount;i++)g->he[b->descriptor[i].linkIndex]=b->descriptor[i].descriptor.epoch;
    g->uploaded=1; return MSX_RESIDENT_OK;
}
extern "C" MSXResidentStatus MSXresidentGpu_initialUpload(MSXResidentGpu*g,const MSXResidentPatchBatch*b){MSXResidentStatus z;if(!g||g->uploaded)return MSX_RESIDENT_ERR_ARGUMENT;z=patch(g,b,1);if(z==MSX_RESIDENT_OK&&!ck(cudaStreamSynchronize(g->stream)))return badgpu(g,MSX_RESIDENT_ERR_TRANSFER);if(z==MSX_RESIDENT_OK){g->stateVersion++;g->aggregateValid=0;}return z;}extern "C" MSXResidentStatus MSXresidentGpu_applyPatches(MSXResidentGpu*g,const MSXResidentPatchBatch*b){MSXResidentStatus z;if(!g||!g->uploaded)return MSX_RESIDENT_ERR_ARGUMENT;z=patch(g,b,0);if(z==MSX_RESIDENT_OK&&!g->activePrepared&&!ck(cudaStreamSynchronize(g->stream)))return badgpu(g,MSX_RESIDENT_ERR_TRANSFER);if(z==MSX_RESIDENT_OK){g->stateVersion++;g->aggregateValid=0;}return z;}
extern "C" MSXResidentStatus MSXresidentGpu_fetchHandoffBatch(MSXResidentGpu*g,const MSXResidentHandoffItem*items,uint32_t n,MSXResidentGpuFetchOutput*out)
{
    if(!use(g))return badgpu(g,MSX_RESIDENT_ERR_GPU);
    if(!g||g->poisoned)return MSX_RESIDENT_ERR_POISONED;
    if(!out||(!items&&n)||n>g->slots||(!out->meta&&n)||(!out->cOut&&n)||
       (!out->lastcOut&&n)||out->stride<g->stride||!g->uploaded||g->activePrepared)
        return MSX_RESIDENT_ERR_ARGUMENT;
    for(uint32_t i=0;i<n;i++)
    {
        const MSXResidentHandoffItem*x=&items[i];
        if(x->boundarySide>1||x->linkIndex<1||x->linkIndex>g->n||
           x->slot>=g->cap[x->linkIndex])
        {g->c.fetchStale++;return MSX_RESIDENT_ERR_GENERATION;}
        if(x->pipeEpoch!=g->he[x->linkIndex])
        {g->c.epochStale++;g->c.epochMismatch++;return MSX_RESIDENT_ERR_GENERATION;}
        uint32_t r=g->base[x->linkIndex]+x->slot;
        if(!g->hu[r]||g->hg[r]!=x->generation)
        {g->c.fetchStale++;g->c.staleGeneration++;return MSX_RESIDENT_ERR_GENERATION;}
        g->hGather[i]={x->linkIndex,r,x->generation,1,0,0,0,0,0,0};
    }
    if(!n)return MSX_RESIDENT_OK;
    if(!ck(trackedCopyAsync(g,g->dGatherStage,g->hGather,(size_t)n*sizeof(Stage),
                            cudaMemcpyHostToDevice,MSX_PROFILE_TRANSFER_H2D,
                            MSX_PROFILE_TRANSFER_SCOPE_HANDOFF)))
        return badgpu(g,MSX_RESIDENT_ERR_TRANSFER);
    gather<<<(n+255)/256,256,0,g->stream>>>(g->dGatherStage,n,g->stride,g->du,g->dg,g->di,
        g->dv,g->dh,g->dhr,g->dur,g->ddr,g->dc,g->dl,g->dGatherStage,g->dGatherC,g->dGatherL);
    if(!ck(cudaGetLastError())||
       !ck(trackedCopyAsync(g,g->hGather,g->dGatherStage,(size_t)n*sizeof(Stage),
                            cudaMemcpyDeviceToHost,MSX_PROFILE_TRANSFER_D2H,
                            MSX_PROFILE_TRANSFER_SCOPE_HANDOFF))||
       !ck(trackedCopyAsync(g,g->hGatherC,g->dGatherC,(size_t)n*g->stride*sizeof(double),
                            cudaMemcpyDeviceToHost,MSX_PROFILE_TRANSFER_D2H,
                            MSX_PROFILE_TRANSFER_SCOPE_HANDOFF))||
       !ck(trackedCopyAsync(g,g->hGatherL,g->dGatherL,(size_t)n*g->stride*sizeof(double),
                            cudaMemcpyDeviceToHost,MSX_PROFILE_TRANSFER_D2H,
                            MSX_PROFILE_TRANSFER_SCOPE_HANDOFF))||
       !ck(cudaStreamSynchronize(g->stream)))
        return badgpu(g,MSX_RESIDENT_ERR_TRANSFER);
    for(uint32_t i=0;i<n;i++)
        if(!g->hGather[i].used||g->hGather[i].generation!=items[i].generation||!g->hGather[i].id)
        {g->c.fetchStale++;g->c.staleGeneration++;return MSX_RESIDENT_ERR_GENERATION;}
    for(uint32_t i=0;i<n;i++)
    {
        Stage*s=&g->hGather[i];MSXResidentHandoffResult*z=&out->meta[i];
        z->linkIndex=items[i].linkIndex;z->slot=items[i].slot;z->generation=s->generation;
        z->boundarySide=items[i].boundarySide;z->pipeEpoch=items[i].pipeEpoch;
        z->payload.volume=s->v;z->payload.hstep=s->h;z->payload.hresponse=s->hr;
        z->payload.uresponse=s->ur;z->payload.dresponse=s->dr;z->payload.parcelId=s->id;
        z->payload.generation=s->generation;z->payload.c=out->cOut+(size_t)i*out->stride;
        z->payload.lastc=out->lastcOut+(size_t)i*out->stride;
        for(uint32_t m=0;m<g->stride;m++)
        {out->cOut[(size_t)i*out->stride+m]=g->hGatherC[(size_t)i*g->stride+m];
         out->lastcOut[(size_t)i*out->stride+m]=g->hGatherL[(size_t)i*g->stride+m];}
    }
    g->c.handoffCount+=n;return MSX_RESIDENT_OK;
}
extern "C" MSXResidentStatus MSXresidentGpu_fetchHandoffs(MSXResidentGpu*g,const MSXResidentHandoffPlan*p,MSXResidentGpuFetchOutput*out,uint32_t n){if(!use(g))return badgpu(g,MSX_RESIDENT_ERR_GPU);if(!g||g->poisoned)return MSX_RESIDENT_ERR_POISONED;if(!p||!out||n!=p->itemCount||(!p->item&&n)||(!out->meta&&n)||(!out->cOut&&n)||(!out->lastcOut&&n)||out->stride<g->stride||!g->uploaded)return MSX_RESIDENT_ERR_ARGUMENT;for(uint32_t i=0;i<n;i++){const MSXResidentHandoffItem*x=&p->item[i];if(x->linkIndex!=p->linkIndex||x->boundarySide!=p->boundarySide||x->pipeEpoch!=p->pipeEpoch){g->c.fetchStale++;return MSX_RESIDENT_ERR_GENERATION;}}return MSXresidentGpu_fetchHandoffBatch(g,p->item,n,out);}
static MSXResidentStatus aggregateCurrent(MSXResidentGpu*g,double*m,uint32_t count,MSXResidentGpuReduction*r)
{
    MSXResidentGpuReduction z;uint32_t k,j;
    if(g->aggregateValid&&g->aggregateVersion==g->stateVersion)
    {
        memcpy(m,g->aggregateMass,(size_t)g->stride*sizeof(double));*r=g->aggregate;
        MSXgpu_profileRecordAggregate(1,1,0);return MSX_RESIDENT_OK;
    }
    memset(&z,0,sizeof(z));memset(g->aggregateMass,0,(size_t)g->stride*sizeof(double));
    z.firstErrorSlot=UINT_MAX;
    if(!ck(cudaMemsetAsync(g->dAggMass,0,((size_t)g->n+1)*g->stride*sizeof(double),g->stream)))
        return badgpu(g,MSX_RESIDENT_ERR_TRANSFER);
    redLink<<<g->n,MSX_RESIDENT_RED_THREADS,
        MSX_RESIDENT_RED_THREADS*(sizeof(MSXResidentGpuReduction)+MSX_RESIDENT_MASS_TILE*sizeof(double)),
        g->stream>>>(g->n,g->stride,g->dd,g->dBase,g->du,g->di,g->dg,g->dv,g->dc,
                     g->dAggLink,g->dAggMass);
    if(!ck(cudaGetLastError())||
       !ck(trackedCopyAsync(g,g->hAggLink,g->dAggLink,
                            ((size_t)g->n+1)*sizeof(MSXResidentGpuReduction),
                            cudaMemcpyDeviceToHost,MSX_PROFILE_TRANSFER_D2H,
                            MSX_PROFILE_TRANSFER_SCOPE_AGGREGATE))||
       !ck(trackedCopyAsync(g,g->hAggMass,g->dAggMass,
                            ((size_t)g->n+1)*g->stride*sizeof(double),
                            cudaMemcpyDeviceToHost,MSX_PROFILE_TRANSFER_D2H,
                            MSX_PROFILE_TRANSFER_SCOPE_AGGREGATE))||
       !ck(cudaStreamSynchronize(g->stream)))
        return badgpu(g,MSX_RESIDENT_ERR_TRANSFER);
    for(k=1;k<=g->n;k++)
    {
        MSXResidentGpuReduction*q=&g->hAggLink[k];
        z.activeCount+=q->activeCount;z.activeVolume+=q->activeVolume;
        z.checksumXor^=q->checksumXor;z.checksumSum+=q->checksumSum;
        z.nanCount+=q->nanCount;z.infCount+=q->infCount;z.errorCount+=q->errorCount;
        if(q->firstErrorSlot!=UINT_MAX&&(z.firstErrorSlot==UINT_MAX||
           q->firstErrorSlot<z.firstErrorSlot))z.firstErrorSlot=q->firstErrorSlot;
        for(j=0;j<g->stride;j++)
            g->aggregateMass[j]+=g->hAggMass[(size_t)k*g->stride+j];
    }
    z.staleGeneration=g->c.staleGeneration;z.epochMismatch=g->c.epochMismatch;
    z.descriptorReject=g->c.descriptorReject;z.capacityOverflow=g->c.capacityOverflow;
    z.fallbacks=g->c.fallbacks;z.handoffCount=g->c.handoffCount;
    z.capacityReject=g->c.capacityReject;z.generationReject=g->c.generationReject;
    z.epochStale=g->c.epochStale;z.fetchStale=g->c.fetchStale;
    z.cudaErrors=g->c.cudaErrors;z.cudaPoisons=g->c.cudaPoisons;
    g->aggregate=z;g->aggregateVersion=g->stateVersion;g->aggregateValid=1;
    memcpy(m,g->aggregateMass,(size_t)g->stride*sizeof(double));*r=z;
    MSXgpu_profileRecordAggregate(1,0,1);return MSX_RESIDENT_OK;
}
extern "C" MSXResidentStatus MSXresidentGpu_reduce(MSXResidentGpu*g,double*m,uint32_t count,MSXResidentGpuReduction*r){if(!use(g))return badgpu(g,MSX_RESIDENT_ERR_GPU);if(!g||g->poisoned)return MSX_RESIDENT_ERR_POISONED;if(!m||!r||count<g->stride||!g->uploaded||g->activePrepared)return MSX_RESIDENT_ERR_ARGUMENT;return aggregateCurrent(g,m,count,r);}extern "C" void MSXresidentGpu_close(MSXResidentGpu*g){if(g){cudaSetDevice(g->device);if(g->stream)cudaStreamSynchronize(g->stream);}gone(g);}
extern "C" MSXResidentStatus MSXresidentGpu_reduceLink(MSXResidentGpu*g,uint32_t link,double*m,uint32_t count,double*volume,MSXResidentGpuReduction*r){MSXResidentGpuReduction total;MSXResidentStatus status;if(!use(g))return badgpu(g,MSX_RESIDENT_ERR_GPU);if(!g||g->poisoned)return MSX_RESIDENT_ERR_POISONED;if(!m||!volume||!r||link<1||link>g->n||count<g->stride||!g->uploaded||g->activePrepared)return MSX_RESIDENT_ERR_ARGUMENT;status=aggregateCurrent(g,g->aggregateMass,g->stride,&total);if(status!=MSX_RESIDENT_OK)return status;memcpy(m,g->hAggMass+(size_t)link*g->stride,(size_t)g->stride*sizeof(double));*volume=g->hAggLink[link].activeVolume;*r=g->hAggLink[link];r->staleGeneration=g->c.staleGeneration;r->epochMismatch=g->c.epochMismatch;r->descriptorReject=g->c.descriptorReject;r->capacityOverflow=g->c.capacityOverflow;r->fallbacks=g->c.fallbacks;r->handoffCount=g->c.handoffCount;r->capacityReject=g->c.capacityReject;r->generationReject=g->c.generationReject;r->epochStale=g->c.epochStale;r->fetchStale=g->c.fetchStale;r->cudaErrors=g->c.cudaErrors;r->cudaPoisons=g->c.cudaPoisons;return MSX_RESIDENT_OK;}
__global__ static void activeGather(const uint32_t *row,uint32_t n,const double *src,double *dst){uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)dst[i]=src[row[i]];}
__global__ static void activeScatter(const uint32_t *row,uint32_t n,const double *src,double *dst){uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)dst[row[i]]=src[i];}
/* Error propagation is a required completion result, not optional
   diagnostics.  Reduce it to one stable active-index winner before the
   single scalar D2H read; this keeps off/stage free of per-row error copies. */
__global__ static void compactError(const int *src,uint32_t n,int *dst)
{
    __shared__ uint32_t first;
    uint32_t tid=threadIdx.x;
    if(tid==0)first=n;
    __syncthreads();
    for(uint32_t i=tid;i<n;i+=blockDim.x)
        if(src[i])atomicMin(&first,i);
    __syncthreads();
    if(tid==0)*dst=first<n?src[first]:0;
}
static int activeMember(const DDesc*d,uint32_t local){uint32_t delta;if(!d->count||local>=d->capacity)return 0;delta=d->orient>0?(local+d->capacity-d->head)%d->capacity:(d->head+d->capacity-local)%d->capacity;return delta<d->count;}
static MSXResidentStatus activeInvalid(MSXResidentGpu*g,uint32_t n,MSXResidentStatus z){for(uint32_t i=0;i<n;i++)g->haNf[g->haRow[i]]=0;return z;}
/* Build the complete candidate before touching the submitted pinned buffer.
   Both arrays contain only initialized double objects, so memcmp preserves
   signed zero and every NaN payload without a tolerance or padded struct. */
static int stageHydCandidate(MSXResidentGpu *g, const double *source,
                             size_t bytes)
{
    memcpy(g->hCandidateHyd, source, bytes);
    g->hydCandidateComparisons++;
    if (g->hydAppliedValid &&
        memcmp(g->hCandidateHyd, g->hSubmittedHyd, bytes) == 0)
    {
        g->hydSkips++;
        g->hydPending = 0;
        return 0;
    }
    /* This is the only write to the pinned submitted table.  It occurs before
       enqueue and never while the associated stream work is in flight. */
    memcpy(g->hSubmittedHyd, g->hCandidateHyd, bytes);
    return 1;
}
static void clearActiveSeen(MSXResidentGpu *g,
                            const MSXResidentGpuActiveWriter *w)
{
    uint32_t i;
    if (!g || !w || !g->activeSeen) return;
    for (i = 0; i < w->touchedCount; ++i)
        if (w->touchedRows[i] < g->slots &&
            g->activeSeen[w->touchedRows[i]] == w->buildSequence)
            g->activeSeen[w->touchedRows[i]] = 0;
}
static void resetActiveWriter(MSXResidentGpu *g,
                              MSXResidentGpuActiveWriter *w)
{
    if (g && g->activeWriter == w) {
        g->activeWriter = NULL;
        g->activeBuilding = 0;
        g->activeSealed = 0;
    }
    if (w) memset(w, 0, sizeof(*w));
}
extern "C" MSXResidentStatus MSXresidentGpu_beginActive(
    MSXResidentGpu *g, uint32_t expectedCount, uint64_t topologyVersion,
    MSXResidentGpuActiveWriter *w)
{
    uint64_t sequence;
    if (!g || !w || !g->uploaded || g->poisoned ||
        g->activePrepared || g->activeBuilding || g->activeSealed ||
        g->activeWriter) return g && g->poisoned ? MSX_RESIDENT_ERR_POISONED :
                                                   MSX_RESIDENT_ERR_ARGUMENT;
    if (expectedCount > g->owned) return MSX_RESIDENT_ERR_CAPACITY;
    sequence = ++g->activeBuildSequence;
    if (!sequence) {
        memset(g->activeSeen, 0, (size_t)g->slots * sizeof(*g->activeSeen));
        sequence = ++g->activeBuildSequence;
    }
    memset(w, 0, sizeof(*w));
    w->owner = g; w->buildSequence = sequence;
    w->topologyVersion = topologyVersion; w->expectedCount = expectedCount;
    w->touchedRows = g->haRow;
    w->building = 1; w->valid = 1;
    g->activeWriter = w; g->activeBuilding = 1; g->activeSealed = 0;
    return MSX_RESIDENT_OK;
}
extern "C" MSXResidentStatus MSXresidentGpu_appendActive(
    MSXResidentGpu *g, MSXResidentGpuActiveWriter *w,
    const MSXResidentActiveRow *x)
{
    uint32_t row, local, n;
    if (!g || !w || w->owner != g || !w->building || !w->valid || !x ||
        g->activeWriter != w) return MSX_RESIDENT_ERR_ARGUMENT;
    n = w->count;
    if (n >= w->expectedCount || n >= g->owned) goto capacity;
    if (!x->linkIndex || x->linkIndex > g->n) goto argument;
    if (x->slot >= g->cap[x->linkIndex] ||
        x->globalRow != g->base[x->linkIndex] + x->slot) goto capacity;
    row = x->globalRow; local = row - g->base[x->linkIndex];
    if (!isfinite(x->volume)) goto argument;
    if (!g->hu[row] || g->hg[row] != x->generation ||
        g->he[x->linkIndex] != x->descriptorEpoch ||
        !activeMember(&g->hd[x->linkIndex], local)) {
        g->c.staleGeneration++;
        w->valid = 0; clearActiveSeen(g, w); w->count = 0;
        resetActiveWriter(g, w);
        return MSX_RESIDENT_ERR_GENERATION;
    }
    if (g->activeSeen[row] == w->buildSequence) goto argument;
    g->activeSeen[row] = w->buildSequence;
    g->haPipe[n] = x->linkIndex; g->haRow[n] = row;
    g->haGen[n] = x->generation; g->haEpoch[n] = x->descriptorEpoch;
    g->haVol[n] = x->volume;
    w->count = n + 1; w->touchedCount = w->count;
    return MSX_RESIDENT_OK;
capacity:
    w->valid = 0; clearActiveSeen(g, w); w->count = 0;
    resetActiveWriter(g, w); return MSX_RESIDENT_ERR_CAPACITY;
argument:
    w->valid = 0; clearActiveSeen(g, w); w->count = 0;
    resetActiveWriter(g, w); return MSX_RESIDENT_ERR_ARGUMENT;
}
extern "C" MSXResidentStatus MSXresidentGpu_sealActive(
    MSXResidentGpu *g, MSXResidentGpuActiveWriter *w,
    uint64_t topologyVersion)
{
    if (!g || !w || w->owner != g || !w->building || !w->valid ||
        g->activeWriter != w) return MSX_RESIDENT_ERR_ARGUMENT;
    if (w->topologyVersion != topologyVersion) {
        w->valid = 0; clearActiveSeen(g, w); w->count = 0;
        resetActiveWriter(g, w); return MSX_RESIDENT_ERR_GENERATION;
    }
    if (w->count != w->expectedCount) {
        w->valid = 0; clearActiveSeen(g, w); w->count = 0;
        resetActiveWriter(g, w); return MSX_RESIDENT_ERR_CAPACITY;
    }
    clearActiveSeen(g, w);
    w->building = 0; w->sealed = 1;
    g->activeBuilding = 0; g->activeSealed = 1;
    return MSX_RESIDENT_OK;
}
extern "C" MSXResidentStatus MSXresidentGpu_abortActiveBuild(
    MSXResidentGpu *g, MSXResidentGpuActiveWriter *w)
{
    if (!g || !w || w->owner != g || g->activeWriter != w)
        return MSX_RESIDENT_ERR_ARGUMENT;
    if (g->activePrepared) return MSX_RESIDENT_ERR_ARGUMENT;
    clearActiveSeen(g, w); w->count = 0; resetActiveWriter(g, w);
    return MSX_RESIDENT_OK;
}
extern "C" MSXResidentStatus MSXresidentGpu_prepareSealedActiveHyd(
    MSXResidentGpu *g, MSXResidentGpuActiveWriter *w,
    const MSXResidentHydView *h, MSXResidentGpuDeviceView *v,
    MSXResidentGpuReactResult *r)
{
    uint32_t n; size_t hydBytes; int hydChanged;
    if (!use(g)) return badgpu(g, MSX_RESIDENT_ERR_GPU);
    if (v) memset(v, 0, sizeof(*v));
    if (r) memset(r, 0, sizeof(*r));
    if (!g || g->poisoned) return MSX_RESIDENT_ERR_POISONED;
    if (!w || w->owner != g || !w->sealed || !w->valid ||
        g->activeWriter != w || g->activePrepared || !h || !h->pipeHyd ||
        h->linkCount != g->n || h->hydStride != MSX_RESIDENT_HYD_STRIDE ||
        h->hydLayout != MSX_RESIDENT_HYD_PIPE_MAJOR || !v ||
        !hydTableBytes(g->n, &hydBytes)) {
        if (w && g->activeWriter == w && !g->activePrepared)
            (void)MSXresidentGpu_abortActiveBuild(g, w);
        return MSX_RESIDENT_ERR_ARGUMENT;
    }
    n = w->count;
    hydChanged = stageHydCandidate(g, h->pipeHyd, hydBytes);
    if (!ck(cudaMemsetAsync(g->dError, 0, sizeof(int), g->stream)) ||
        !ck(cudaMemsetAsync(g->daReacted, 0,
                            ((size_t)g->n + 1) * g->stride * sizeof(double),
                            g->stream)))
        return badgpu(g, MSX_RESIDENT_ERR_TRANSFER);
    if (hydChanged && !ck(trackedCopyAsync(g, g->dPipeHyd, g->hSubmittedHyd,
                                           hydBytes, cudaMemcpyHostToDevice,
                                           MSX_PROFILE_TRANSFER_H2D,
                                           MSX_PROFILE_TRANSFER_SCOPE_HYD)))
        return badgpu(g, MSX_RESIDENT_ERR_TRANSFER);
    if (hydChanged) { g->hydUploads++; g->hydPending = 1; }
    if (n && (!ck(trackedCopyAsync(g, g->daPipe, g->haPipe,
                                   (size_t)n * sizeof(uint32_t), cudaMemcpyHostToDevice,
                                   MSX_PROFILE_TRANSFER_H2D, MSX_PROFILE_TRANSFER_SCOPE_ACTIVE)) ||
               !ck(trackedCopyAsync(g, g->daRow, g->haRow,
                                   (size_t)n * sizeof(uint32_t), cudaMemcpyHostToDevice,
                                   MSX_PROFILE_TRANSFER_H2D, MSX_PROFILE_TRANSFER_SCOPE_ACTIVE)) ||
               !ck(trackedCopyAsync(g, g->daVol, g->haVol,
                                   (size_t)n * sizeof(double), cudaMemcpyHostToDevice,
                                   MSX_PROFILE_TRANSFER_H2D, MSX_PROFILE_TRANSFER_SCOPE_ACTIVE)) ||
               !ck(cudaMemsetAsync(g->daNf, 0, (size_t)n * sizeof(uint32_t), g->stream)) ||
               !ck(cudaMemsetAsync(g->daNj, 0, (size_t)n * sizeof(uint32_t), g->stream)) ||
               !ck(cudaMemsetAsync(g->daNa, 0, (size_t)n * sizeof(uint32_t), g->stream)) ||
               !ck(cudaMemsetAsync(g->daNr, 0, (size_t)n * sizeof(uint32_t), g->stream)) ||
               !ck(cudaMemsetAsync(g->daErr, 0, (size_t)n * sizeof(int), g->stream)) ||
               !ck(cudaMemsetAsync(g->daLast, 0, (size_t)n * sizeof(double), g->stream))))
        return badgpu(g, MSX_RESIDENT_ERR_TRANSFER);
    if (n) {
        activeGather<<<(n + 255) / 256, 256, 0, g->stream>>>(g->daRow, n,
                                                              g->dh, g->daH);
        if (!ck(cudaGetLastError())) return badgpu(g, MSX_RESIDENT_ERR_GPU);
    }
    v->segPipe=(uint64_t)(uintptr_t)g->daPipe; v->segRow=(uint64_t)(uintptr_t)g->daRow;
    v->segVol=(uint64_t)(uintptr_t)g->daVol; v->hstep=(uint64_t)(uintptr_t)g->daH;
    v->c=(uint64_t)(uintptr_t)g->dc; v->lastc=(uint64_t)(uintptr_t)g->dl;
    v->hyd=(uint64_t)(uintptr_t)g->dPipeHyd; v->reacted=(uint64_t)(uintptr_t)g->daReacted;
    v->ros2Nfcn=(uint64_t)(uintptr_t)g->daNf; v->ros2Njac=(uint64_t)(uintptr_t)g->daNj;
    v->ros2Naccept=(uint64_t)(uintptr_t)g->daNa; v->ros2Nreject=(uint64_t)(uintptr_t)g->daNr;
    v->ros2LastHstep=(uint64_t)(uintptr_t)g->daLast; v->ros2Err=(uint64_t)(uintptr_t)g->daErr;
    v->streamHandle=(uint64_t)(uintptr_t)g->stream; v->itemCount=n;
    v->speciesStride=g->stride; v->hydStride=MSX_RESIDENT_HYD_STRIDE;
    v->hydLayout=MSX_RESIDENT_HYD_PIPE_MAJOR; g->hydLayout=v->hydLayout;
    g->activeCount=n; g->activePrepared=1; g->activeSealed=0;
    w->sealed=0;
    return MSX_RESIDENT_OK;
}
extern "C" MSXResidentStatus MSXresidentGpu_prepareActive(MSXResidentGpu*g,const MSXResidentActiveBatch*b,MSXResidentGpuDeviceView*v,MSXResidentGpuReactResult*r)
{
    uint32_t i,n;
    if(!use(g))return badgpu(g,MSX_RESIDENT_ERR_GPU);
    if(v)memset(v,0,sizeof(*v));
    if(r)memset(r,0,sizeof(*r));
    if(!g||g->poisoned)return MSX_RESIDENT_ERR_POISONED;
    if(!b||(!b->item&&b->itemCount)||!v||!g->uploaded||g->activePrepared||
       g->activeBuilding||g->activeSealed)
        return MSX_RESIDENT_ERR_ARGUMENT;
    n=b->itemCount;
    if(n>g->owned)return MSX_RESIDENT_ERR_CAPACITY;
    for(i=0;i<n;i++)
    {
        const MSXResidentActiveItem*x=&b->item[i];
        uint32_t local,row;
        if(!x->linkIndex||x->linkIndex>g->n||!x->hyd||!isfinite(x->volume))
            return activeInvalid(g,i,MSX_RESIDENT_ERR_ARGUMENT);
        row=x->globalRow;
        if(row<g->base[x->linkIndex]||row>=g->base[x->linkIndex]+g->cap[x->linkIndex])
            return activeInvalid(g,i,MSX_RESIDENT_ERR_CAPACITY);
        local=row-g->base[x->linkIndex];
        if(!g->hu[row]||g->hg[row]!=x->generation||g->he[x->linkIndex]!=x->descriptorEpoch||
           !activeMember(&g->hd[x->linkIndex],local))
        { g->c.staleGeneration++; return activeInvalid(g,i,MSX_RESIDENT_ERR_GENERATION); }
        if(g->haNf[row])return activeInvalid(g,i,MSX_RESIDENT_ERR_ARGUMENT);
        g->haNf[row]=i+1;
        g->haPipe[i]=x->linkIndex; g->haRow[i]=row; g->haGen[i]=x->generation;
        g->haEpoch[i]=x->descriptorEpoch; g->haVol[i]=x->volume;
        memcpy(g->haHyd+(size_t)i*MSX_RESIDENT_HYD_STRIDE,x->hyd,
               MSX_RESIDENT_HYD_STRIDE*sizeof(double));
    }
    for(i=0;i<n;i++)g->haNf[g->haRow[i]]=0;
    /* Completion outputs have defined zero semantics even for an empty active
       set; per-row inputs and diagnostics are only touched when rows exist. */
    if(!ck(cudaMemsetAsync(g->dError,0,sizeof(int),g->stream)) ||
       !ck(cudaMemsetAsync(g->daReacted,0,((size_t)g->n+1)*g->stride*sizeof(double),g->stream)))
        return badgpu(g,MSX_RESIDENT_ERR_TRANSFER);
    if(n && (!ck(trackedCopyAsync(g,g->daPipe,g->haPipe,(size_t)n*sizeof(uint32_t),
                                  cudaMemcpyHostToDevice,MSX_PROFILE_TRANSFER_H2D,MSX_PROFILE_TRANSFER_SCOPE_ACTIVE)) ||
             !ck(trackedCopyAsync(g,g->daRow,g->haRow,(size_t)n*sizeof(uint32_t),
                                  cudaMemcpyHostToDevice,MSX_PROFILE_TRANSFER_H2D,MSX_PROFILE_TRANSFER_SCOPE_ACTIVE)) ||
             !ck(trackedCopyAsync(g,g->daVol,g->haVol,(size_t)n*sizeof(double),
                                  cudaMemcpyHostToDevice,MSX_PROFILE_TRANSFER_H2D,MSX_PROFILE_TRANSFER_SCOPE_ACTIVE)) ||
             !ck(trackedCopyAsync(g,g->daHyd,g->haHyd,(size_t)n*MSX_RESIDENT_HYD_STRIDE*sizeof(double),
                                  cudaMemcpyHostToDevice,MSX_PROFILE_TRANSFER_H2D,MSX_PROFILE_TRANSFER_SCOPE_HYD)) ||
             !ck(cudaMemsetAsync(g->daNf,0,(size_t)n*sizeof(uint32_t),g->stream)) ||
             !ck(cudaMemsetAsync(g->daNj,0,(size_t)n*sizeof(uint32_t),g->stream)) ||
             !ck(cudaMemsetAsync(g->daNa,0,(size_t)n*sizeof(uint32_t),g->stream)) ||
             !ck(cudaMemsetAsync(g->daNr,0,(size_t)n*sizeof(uint32_t),g->stream)) ||
             !ck(cudaMemsetAsync(g->daErr,0,(size_t)n*sizeof(int),g->stream)) ||
             !ck(cudaMemsetAsync(g->daLast,0,(size_t)n*sizeof(double),g->stream))))
        return badgpu(g,MSX_RESIDENT_ERR_TRANSFER);
    if(n)
    {
        activeGather<<<(n+255)/256,256,0,g->stream>>>(g->daRow,n,g->dh,g->daH);
        if(!ck(cudaGetLastError()))return badgpu(g,MSX_RESIDENT_ERR_GPU);
    }
    v->segPipe=(unsigned long long)g->daPipe; v->segRow=(unsigned long long)g->daRow;
    v->segVol=(unsigned long long)g->daVol; v->hstep=(unsigned long long)g->daH;
    v->c=(unsigned long long)g->dc; v->lastc=(unsigned long long)g->dl;
    v->hyd=(unsigned long long)g->daHyd; v->reacted=(unsigned long long)g->daReacted;
    v->ros2Nfcn=(unsigned long long)g->daNf; v->ros2Njac=(unsigned long long)g->daNj;
    v->ros2Naccept=(unsigned long long)g->daNa; v->ros2Nreject=(unsigned long long)g->daNr;
    v->ros2LastHstep=(unsigned long long)g->daLast; v->ros2Err=(unsigned long long)g->daErr;
    v->streamHandle=(uint64_t)(uintptr_t)g->stream;
    v->itemCount=n; v->speciesStride=g->stride; v->hydStride=MSX_RESIDENT_HYD_STRIDE;
    v->hydLayout=MSX_RESIDENT_HYD_ACTIVE_MAJOR;
    g->hydLayout=MSX_RESIDENT_HYD_ACTIVE_MAJOR; invalidateHydCache(g);
    g->activeCount=n; g->activePrepared=1; g->completionEnqueued=0;
    return MSX_RESIDENT_OK;
}
extern "C" MSXResidentStatus MSXresidentGpu_getDeviceView(MSXResidentGpu*g,MSXResidentGpuDeviceView*v){if(!use(g))return badgpu(g,MSX_RESIDENT_ERR_GPU);if(!v)return MSX_RESIDENT_ERR_ARGUMENT;memset(v,0,sizeof(*v));if(!g||g->poisoned)return MSX_RESIDENT_ERR_POISONED;if(!g->activePrepared)return MSX_RESIDENT_ERR_ARGUMENT;v->segPipe=(uint64_t)(uintptr_t)g->daPipe;v->segRow=(uint64_t)(uintptr_t)g->daRow;v->segVol=(uint64_t)(uintptr_t)g->daVol;v->hstep=(uint64_t)(uintptr_t)g->daH;v->c=(uint64_t)(uintptr_t)g->dc;v->lastc=(uint64_t)(uintptr_t)g->dl;v->hyd=(uint64_t)(uintptr_t)(g->hydLayout==MSX_RESIDENT_HYD_PIPE_MAJOR?g->dPipeHyd:g->daHyd);v->reacted=(uint64_t)(uintptr_t)g->daReacted;v->ros2Nfcn=(uint64_t)(uintptr_t)g->daNf;v->ros2Njac=(uint64_t)(uintptr_t)g->daNj;v->ros2Naccept=(uint64_t)(uintptr_t)g->daNa;v->ros2Nreject=(uint64_t)(uintptr_t)g->daNr;v->ros2LastHstep=(uint64_t)(uintptr_t)g->daLast;v->ros2Err=(uint64_t)(uintptr_t)g->daErr;v->streamHandle=(uint64_t)(uintptr_t)g->stream;v->itemCount=g->activeCount;v->speciesStride=g->stride;v->hydStride=MSX_RESIDENT_HYD_STRIDE;v->hydLayout=g->hydLayout;return MSX_RESIDENT_OK;}
extern "C" MSXResidentStatus MSXresidentGpu_prepareActiveHyd(MSXResidentGpu*g,const MSXResidentActiveBatch*b,const MSXResidentHydView*h,MSXResidentGpuDeviceView*v,MSXResidentGpuReactResult*r)
{
    uint32_t i,n; size_t hydBytes;
    if(!use(g))return badgpu(g,MSX_RESIDENT_ERR_GPU);
    if(v)memset(v,0,sizeof(*v));
    if(r)memset(r,0,sizeof(*r));
    if(!g||g->poisoned)return MSX_RESIDENT_ERR_POISONED;
    if(!b||(!b->item&&b->itemCount)||!v||!h||!h->pipeHyd||
       !g->uploaded||g->activePrepared||g->activeBuilding||g->activeSealed)
        return MSX_RESIDENT_ERR_ARGUMENT;
    if(h->linkCount!=g->n||h->hydStride!=MSX_RESIDENT_HYD_STRIDE||
       h->hydLayout!=MSX_RESIDENT_HYD_PIPE_MAJOR||!hydTableBytes(g->n,&hydBytes))
        return MSX_RESIDENT_ERR_ARGUMENT;
    n=b->itemCount;
    if(n>g->owned)return MSX_RESIDENT_ERR_CAPACITY;
    for(i=0;i<n;i++)
    {
        const MSXResidentActiveItem*x=&b->item[i];
        uint32_t local,row;
        if(!x->linkIndex||x->linkIndex>g->n||!isfinite(x->volume))
            return activeInvalid(g,i,MSX_RESIDENT_ERR_ARGUMENT);
        row=x->globalRow;
        if(row<g->base[x->linkIndex]||row>=g->base[x->linkIndex]+g->cap[x->linkIndex])
            return activeInvalid(g,i,MSX_RESIDENT_ERR_CAPACITY);
        local=row-g->base[x->linkIndex];
        if(!g->hu[row]||g->hg[row]!=x->generation||g->he[x->linkIndex]!=x->descriptorEpoch||
           !activeMember(&g->hd[x->linkIndex],local))
        { g->c.staleGeneration++; return activeInvalid(g,i,MSX_RESIDENT_ERR_GENERATION); }
        if(g->haNf[row])return activeInvalid(g,i,MSX_RESIDENT_ERR_ARGUMENT);
        g->haNf[row]=i+1;
        g->haPipe[i]=x->linkIndex; g->haRow[i]=row; g->haGen[i]=x->generation;
        g->haEpoch[i]=x->descriptorEpoch; g->haVol[i]=x->volume;
    }
    for(i=0;i<n;i++)g->haNf[g->haRow[i]]=0;
    /* Candidate comparison is host-only.  Only a changed candidate is copied
       into the pinned submitted table and sent to the device. */
    int hydChanged = stageHydCandidate(g, h->pipeHyd, hydBytes);
    if(!ck(cudaMemsetAsync(g->dError,0,sizeof(int),g->stream))||
       !ck(cudaMemsetAsync(g->daReacted,0,((size_t)g->n+1)*g->stride*sizeof(double),g->stream)))
        return badgpu(g,MSX_RESIDENT_ERR_TRANSFER);
    if (hydChanged)
    {
        if (!ck(trackedCopyAsync(g,g->dPipeHyd,g->hSubmittedHyd,hydBytes,
                                cudaMemcpyHostToDevice,MSX_PROFILE_TRANSFER_H2D,
                                MSX_PROFILE_TRANSFER_SCOPE_HYD)))
            return badgpu(g,MSX_RESIDENT_ERR_TRANSFER);
        g->hydUploads++;
        g->hydPending=1;
    }
    if(n && (!ck(trackedCopyAsync(g,g->daPipe,g->haPipe,(size_t)n*sizeof(uint32_t),cudaMemcpyHostToDevice,MSX_PROFILE_TRANSFER_H2D,MSX_PROFILE_TRANSFER_SCOPE_ACTIVE))||
             !ck(trackedCopyAsync(g,g->daRow,g->haRow,(size_t)n*sizeof(uint32_t),cudaMemcpyHostToDevice,MSX_PROFILE_TRANSFER_H2D,MSX_PROFILE_TRANSFER_SCOPE_ACTIVE))||
             !ck(trackedCopyAsync(g,g->daVol,g->haVol,(size_t)n*sizeof(double),cudaMemcpyHostToDevice,MSX_PROFILE_TRANSFER_H2D,MSX_PROFILE_TRANSFER_SCOPE_ACTIVE))||
             !ck(cudaMemsetAsync(g->daNf,0,(size_t)n*sizeof(uint32_t),g->stream))||
             !ck(cudaMemsetAsync(g->daNj,0,(size_t)n*sizeof(uint32_t),g->stream))||
             !ck(cudaMemsetAsync(g->daNa,0,(size_t)n*sizeof(uint32_t),g->stream))||
             !ck(cudaMemsetAsync(g->daNr,0,(size_t)n*sizeof(uint32_t),g->stream))||
             !ck(cudaMemsetAsync(g->daErr,0,(size_t)n*sizeof(int),g->stream))||
             !ck(cudaMemsetAsync(g->daLast,0,(size_t)n*sizeof(double),g->stream))))
        return badgpu(g,MSX_RESIDENT_ERR_TRANSFER);
    if(n){activeGather<<<(n+255)/256,256,0,g->stream>>>(g->daRow,n,g->dh,g->daH);if(!ck(cudaGetLastError()))return badgpu(g,MSX_RESIDENT_ERR_GPU);}
    v->segPipe=(uint64_t)(uintptr_t)g->daPipe; v->segRow=(uint64_t)(uintptr_t)g->daRow;
    v->segVol=(uint64_t)(uintptr_t)g->daVol; v->hstep=(uint64_t)(uintptr_t)g->daH;
    v->c=(uint64_t)(uintptr_t)g->dc; v->lastc=(uint64_t)(uintptr_t)g->dl;
    v->hyd=(uint64_t)(uintptr_t)g->dPipeHyd; v->reacted=(uint64_t)(uintptr_t)g->daReacted;
    v->ros2Nfcn=(uint64_t)(uintptr_t)g->daNf; v->ros2Njac=(uint64_t)(uintptr_t)g->daNj;
    v->ros2Naccept=(uint64_t)(uintptr_t)g->daNa; v->ros2Nreject=(uint64_t)(uintptr_t)g->daNr;
    v->ros2LastHstep=(uint64_t)(uintptr_t)g->daLast; v->ros2Err=(uint64_t)(uintptr_t)g->daErr;
    v->streamHandle=(uint64_t)(uintptr_t)g->stream; v->itemCount=n; v->speciesStride=g->stride;
    v->hydStride=MSX_RESIDENT_HYD_STRIDE; v->hydLayout=MSX_RESIDENT_HYD_PIPE_MAJOR;
    g->hydLayout=MSX_RESIDENT_HYD_PIPE_MAJOR; g->activeCount=n; g->activePrepared=1; g->completionEnqueued=0;
    return MSX_RESIDENT_OK;
}
extern "C" void MSXresidentGpu_setDiagnosticMode(MSXResidentGpu*g,int enabled){if(g){g->diagnostic=enabled?1:0;g->stateVersion++;g->aggregateValid=0;}}
extern "C" MSXResidentStatus MSXresidentGpu_enqueueActiveCompletion(MSXResidentGpu*g)
{
    uint32_t n;
    if(!use(g))return badgpu(g,MSX_RESIDENT_ERR_GPU);
    if(!g||g->poisoned)return MSX_RESIDENT_ERR_POISONED;
    if(!g->activePrepared)return MSX_RESIDENT_ERR_ARGUMENT;
    if(g->completionEnqueued)return MSX_RESIDENT_ERR_ARGUMENT;
    n=g->activeCount;
    if(!ck(cudaMemsetAsync(g->dError,0,sizeof(int),g->stream)))
        return badgpu(g,MSX_RESIDENT_ERR_TRANSFER);
    compactError<<<1,256,0,g->stream>>>(g->daErr,n,g->dError);
    if(!ck(cudaGetLastError()))return badgpu(g,MSX_RESIDENT_ERR_GPU);
    if(n)
    {
        activeScatter<<<(n+255)/256,256,0,g->stream>>>(g->daRow,n,g->daH,g->dh);
        if(!ck(cudaGetLastError()))return badgpu(g,MSX_RESIDENT_ERR_GPU);
    }
    if(!ck(trackedCopyAsync(g,g->haErr,g->dError,sizeof(int),cudaMemcpyDeviceToHost,
                            MSX_PROFILE_TRANSFER_D2H,MSX_PROFILE_TRANSFER_SCOPE_DIAGNOSTIC))||
       !ck(trackedCopyAsync(g,g->haReacted,g->daReacted,
                            ((size_t)g->n+1)*g->stride*sizeof(double),cudaMemcpyDeviceToHost,
                            MSX_PROFILE_TRANSFER_D2H,MSX_PROFILE_TRANSFER_SCOPE_REACTED)))
        return badgpu(g,MSX_RESIDENT_ERR_TRANSFER);
    if(g->diagnostic &&
       (!ck(trackedCopyAsync(g,g->haLast,g->daLast,(size_t)n*sizeof(double),cudaMemcpyDeviceToHost,
                             MSX_PROFILE_TRANSFER_D2H,MSX_PROFILE_TRANSFER_SCOPE_DIAGNOSTIC))||
        !ck(trackedCopyAsync(g,g->haNf,g->daNf,(size_t)n*sizeof(uint32_t),cudaMemcpyDeviceToHost,
                             MSX_PROFILE_TRANSFER_D2H,MSX_PROFILE_TRANSFER_SCOPE_DIAGNOSTIC))||
        !ck(trackedCopyAsync(g,g->haNj,g->daNj,(size_t)n*sizeof(uint32_t),cudaMemcpyDeviceToHost,
                             MSX_PROFILE_TRANSFER_D2H,MSX_PROFILE_TRANSFER_SCOPE_DIAGNOSTIC))||
        !ck(trackedCopyAsync(g,g->haNa,g->daNa,(size_t)n*sizeof(uint32_t),cudaMemcpyDeviceToHost,
                             MSX_PROFILE_TRANSFER_D2H,MSX_PROFILE_TRANSFER_SCOPE_DIAGNOSTIC))||
        !ck(trackedCopyAsync(g,g->haNr,g->daNr,(size_t)n*sizeof(uint32_t),cudaMemcpyDeviceToHost,
                             MSX_PROFILE_TRANSFER_D2H,MSX_PROFILE_TRANSFER_SCOPE_DIAGNOSTIC))))
        return badgpu(g,MSX_RESIDENT_ERR_TRANSFER);
    g->completionEnqueued=1;
    return MSX_RESIDENT_OK;
}
static MSXResidentStatus finishActiveImpl(MSXResidentGpu*g,MSXResidentGpuReactResult*r,int alreadyWaited)
{
    uint32_t n,i; uint64_t nf=0,nj=0,na=0,nr=0; int err=0;
    if(!use(g))return badgpu(g,MSX_RESIDENT_ERR_GPU);
    if(r)memset(r,0,sizeof(*r));
    if(!g||g->poisoned)return MSX_RESIDENT_ERR_POISONED;
    if(!g->activePrepared)return MSX_RESIDENT_ERR_ARGUMENT;
    n=g->activeCount;
    if(!g->completionEnqueued)
    {
        MSXResidentStatus z=MSXresidentGpu_enqueueActiveCompletion(g);
        if(z!=MSX_RESIDENT_OK)return z;
    }
    if(!alreadyWaited && !ck(cudaStreamSynchronize(g->stream)))
        return badgpu(g,MSX_RESIDENT_ERR_TRANSFER);
    err=*g->haErr;
    if(g->diagnostic)
    {
        for(i=0;i<n;i++){nf+=g->haNf[i];nj+=g->haNj[i];na+=g->haNa[i];nr+=g->haNr[i];}
    }
    for(i=0;i<n;i++){g->haNf[i]=0;g->haNj[i]=0;g->haNa[i]=0;g->haNr[i]=0;}
    if(r){r->ros2Nfcn=nf;r->ros2Njac=nj;r->ros2Naccept=na;r->ros2Nreject=nr;
        r->ros2Error=err;r->ros2LastHstep=(g->diagnostic&&n)?g->haLast[n-1]:0.0;
        r->reacted=g->haReacted;r->reactedStride=g->stride;r->reactedLinkCount=g->n+1;}
    g->completionEnqueued=0;
    if(err){invalidateHydCache(g);g->poisoned=1;g->activePrepared=0;g->activeCount=0;
        if (g->activeWriter) resetActiveWriter(g, g->activeWriter);
        return MSX_RESIDENT_ERR_GPU;}
    /* The stream has completed all kernels and completion copies.  Only now
       may the submitted table become the applied cache version. */
    if (g->hydPending)
    {
        g->hydAppliedValid=1;
        g->hydPending=0;
    }
    if(n){g->stateVersion++;g->aggregateValid=0;}
    g->activePrepared=0;g->activeCount=0;
    if (g->activeWriter) resetActiveWriter(g, g->activeWriter);
    return MSX_RESIDENT_OK;
}
extern "C" MSXResidentStatus MSXresidentGpu_finishActive(MSXResidentGpu*g,MSXResidentGpuReactResult*r){return finishActiveImpl(g,r,0);}
extern "C" MSXResidentStatus MSXresidentGpu_finishActiveAfterWait(MSXResidentGpu*g,MSXResidentGpuReactResult*r){return finishActiveImpl(g,r,1);}
extern "C" MSXResidentStatus MSXresidentGpu_syncActive(MSXResidentGpu*g,MSXResidentGpuActiveSyncOutput*out,uint32_t n)
{
    uint32_t i;
    if(!use(g))return badgpu(g,MSX_RESIDENT_ERR_GPU);
    if(!g||g->poisoned)return MSX_RESIDENT_ERR_POISONED;
    if(!out||n>g->owned||g->activePrepared||(!out->row&&n)||(!out->cOut&&n)||(!out->lastcOut&&n)||out->stride<g->stride)return MSX_RESIDENT_ERR_ARGUMENT;
    for(i=0;i<n;i++)
    {
        uint32_t row=g->haRow[i],k=g->haPipe[i];
        if(!k||k>g->n||row<g->base[k]||row>=g->base[k]+g->cap[k]||!g->hu[row]||g->hg[row]!=g->haGen[i]||g->he[k]!=g->haEpoch[i])
        {g->c.staleGeneration++;return MSX_RESIDENT_ERR_GENERATION;}
        g->hGather[i]={k,row,g->haGen[i],1,0,0,0,0,0,0};
    }
    if(n&&!ck(trackedCopyAsync(g,g->dGatherStage,g->hGather,(size_t)n*sizeof(Stage),cudaMemcpyHostToDevice,MSX_PROFILE_TRANSFER_H2D,MSX_PROFILE_TRANSFER_SCOPE_FULL_SYNC)))return badgpu(g,MSX_RESIDENT_ERR_TRANSFER);
    if(n)
    {
        gather<<<(n+255)/256,256,0,g->stream>>>(g->dGatherStage,n,g->stride,g->du,g->dg,g->di,g->dv,g->dh,g->dhr,g->dur,g->ddr,g->dc,g->dl,g->dGatherStage,g->dGatherC,g->dGatherL);
        if(!ck(cudaGetLastError())||
           !ck(trackedCopyAsync(g,g->hGather,g->dGatherStage,(size_t)n*sizeof(Stage),cudaMemcpyDeviceToHost,MSX_PROFILE_TRANSFER_D2H,MSX_PROFILE_TRANSFER_SCOPE_FULL_SYNC))||
           !ck(trackedCopyAsync(g,g->hGatherC,g->dGatherC,(size_t)n*g->stride*sizeof(double),cudaMemcpyDeviceToHost,MSX_PROFILE_TRANSFER_D2H,MSX_PROFILE_TRANSFER_SCOPE_FULL_SYNC))||
           !ck(trackedCopyAsync(g,g->hGatherL,g->dGatherL,(size_t)n*g->stride*sizeof(double),cudaMemcpyDeviceToHost,MSX_PROFILE_TRANSFER_D2H,MSX_PROFILE_TRANSFER_SCOPE_FULL_SYNC))||
           !ck(cudaStreamSynchronize(g->stream)))return badgpu(g,MSX_RESIDENT_ERR_TRANSFER);
    }
    for(i=0;i<n;i++)
    {
        Stage*s=&g->hGather[i];
        if(!s->used||s->generation!=g->haGen[i]||!s->id){g->c.staleGeneration++;return MSX_RESIDENT_ERR_GENERATION;}
        out->row[i]={s->link,s->row,s->generation,g->haEpoch[i],s->h,s->hr,s->ur,s->dr};
        memcpy(out->cOut+(size_t)i*out->stride,g->hGatherC+(size_t)i*g->stride,g->stride*sizeof(double));
        memcpy(out->lastcOut+(size_t)i*out->stride,g->hGatherL+(size_t)i*g->stride,g->stride*sizeof(double));
    }
    return MSX_RESIDENT_OK;
}
#undef MSXresidentGpu_fetchHandoffBatch
/* Validate duplicate global rows without comparing every pair.  The stamp
   table is sized once with the resident pool and re-used for each batch, so
   normal batches are O(n) and old marks never need a full-capacity clear. */
static int duplicateFetchItem(MSXResidentGpu *g,
                              const MSXResidentHandoffItem *items,
                              uint32_t n)
{
    uint64_t stamp;
    uint32_t i;
    if (!g || (!items && n) || !g->fetchSeen) return 0;
    stamp = ++g->fetchStamp;
    if (!stamp)
    {
        memset(g->fetchSeen, 0, (size_t)g->slots * sizeof(*g->fetchSeen));
        stamp = ++g->fetchStamp;
    }
    for (i = 0; i < n; ++i)
    {
        const MSXResidentHandoffItem *item = &items[i];
        uint32_t row;
        /* Leave bounds, side, epoch, generation, and payload validation to
           the implementation body so invalid input keeps its old status. */
        if (item->linkIndex < 1 || item->linkIndex > g->n ||
            item->slot >= g->cap[item->linkIndex])
            continue;
        row = g->base[item->linkIndex] + item->slot;
        if (g->fetchSeen[row] == stamp) return 1;
        g->fetchSeen[row] = stamp;
    }
    return 0;
}
extern "C" MSXResidentStatus MSXresidentGpu_fetchHandoffBatch(
    MSXResidentGpu *g, const MSXResidentHandoffItem *items, uint32_t n,
    MSXResidentGpuFetchOutput *out)
{
    if (g && duplicateFetchItem(g, items, n)) return MSX_RESIDENT_ERR_ARGUMENT;
    return MSXresidentGpu_fetchHandoffBatch_impl(g, items, n, out);
}
extern "C" MSXResidentStatus MSXresidentGpu_getTransferStats(const MSXResidentGpu *g,
                                                              MSXResidentGpuTransferStats *out)
{
    if (!g || !out) return MSX_RESIDENT_ERR_ARGUMENT;
    out->h2dBytes = g->handoffH2DBytes;
    out->h2dCalls = g->handoffH2DCalls;
    out->d2hBytes = g->handoffD2HBytes;
    out->d2hCalls = g->handoffD2HCalls;
    out->hydH2DBytes = g->hydH2DBytes;
    out->hydH2DCalls = g->hydH2DCalls;
    out->hydCandidateComparisons = g->hydCandidateComparisons;
    out->hydUploads = g->hydUploads;
    out->hydSkips = g->hydSkips;
    out->hydBytes = g->hydH2DBytes;
    out->hydApiCalls = g->hydH2DCalls;
    out->hydAppliedValid = g->hydAppliedValid;
    out->hydPending = g->hydPending;
    return MSX_RESIDENT_OK;
}
extern "C" MSXResidentStatus MSXresidentGpu_abortActive(MSXResidentGpu*g){if(!use(g))return badgpu(g,MSX_RESIDENT_ERR_GPU);if(!g)return MSX_RESIDENT_ERR_ARGUMENT;if(g->activeWriter&&!g->activePrepared){clearActiveSeen(g,g->activeWriter);g->activeWriter->count=0;resetActiveWriter(g,g->activeWriter);return MSX_RESIDENT_OK;}if(!g->activePrepared)return g->poisoned?MSX_RESIDENT_ERR_POISONED:MSX_RESIDENT_OK;if(!ck(cudaStreamSynchronize(g->stream)))return badgpu(g,MSX_RESIDENT_ERR_TRANSFER);invalidateHydCache(g);g->activePrepared=0;g->activeCount=0;g->completionEnqueued=0;if(g->activeWriter)resetActiveWriter(g,g->activeWriter);g->poisoned=1;g->c.cudaPoisons++;return MSX_RESIDENT_OK;}
extern "C" MSXResidentStatus MSXresidentGpu_invalidateHyd(MSXResidentGpu*g){if(!g)return MSX_RESIDENT_ERR_ARGUMENT;if(g->activePrepared)return MSX_RESIDENT_ERR_ARGUMENT;invalidateHydCache(g);return MSX_RESIDENT_OK;}
extern "C" MSXResidentStatus MSXresidentGpu_poison(MSXResidentGpu*g){if(!g)return MSX_RESIDENT_ERR_ARGUMENT;if(g->activePrepared&&!ck(cudaStreamSynchronize(g->stream)))return badgpu(g,MSX_RESIDENT_ERR_TRANSFER);invalidateHydCache(g);g->activePrepared=0;g->activeCount=0;g->completionEnqueued=0;if(!g->poisoned){g->poisoned=1;g->c.cudaPoisons++;}g->aggregateValid=0;return MSX_RESIDENT_OK;}
