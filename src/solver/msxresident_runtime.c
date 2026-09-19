#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "msxresident_runtime.h"
#include "msxresident_core_cuda.h"
#include "msxresident_hash.h"
#include "msxsegment_storage.h"
#include "msxgpu.h"
#include "msxtypes.h"

extern MSXproject MSX;

typedef struct { int opened, resident, dispatchReady, handoffReady, patchClass; MSXResidentGpu *gpu;
    uint64_t wouldDescriptors, wouldSlots, stale, fallbacks; MSXResidentStatus lastStatus;
    MSXResidentActiveRow *rows; MSXResidentActiveItem *active;
    uint32_t activeCap, stride;
    MSXResidentHandoffPlan *plan; MSXResidentHandoffItem *item; MSXResidentHandoffResult *out;
    double *c,*lastc; MSXResidentHandoffTransaction *tx; unsigned char *fallback;
    uint32_t *offset; uint32_t itemCap, itemCount;
    double handoffPlanMs;
    int auditPoisonCpuMirrors, auditAggregateFail, inFlight;
    uint64_t sequence, topologyVersion, stateVersion;
    MSXResidentRuntimeToken flight;
    char status[96], resolvedCapacity[MAXFNAME]; } Runtime;
static Runtime R;

#define MSX_RESIDENT_RUNTIME_TOKEN_MAGIC UINT64_C(0x52544f4b454e5033)

static int map(MSXResidentStatus s)
{
    switch (s) {
    case MSX_RESIDENT_ERR_GPU: case MSX_RESIDENT_ERR_TRANSFER:
    case MSX_RESIDENT_ERR_POISONED: return ERR_GPU_KERNEL_RUNTIME_ERROR;
    case MSX_RESIDENT_ERR_MEMORY: return ERR_GPU_MEMORY_ALLOCATION_FAILED;
    case MSX_RESIDENT_ERR_CAPACITY: case MSX_RESIDENT_ERR_OVERFLOW:
        return ERR_GPU_SEGMENT_PACK_FAILED;
    case MSX_RESIDENT_ERR_GENERATION: return ERR_GPU_KERNEL_RUNTIME_ERROR;
    default: return ERR_GPU_UNSUPPORTED_FEATURE;
    }
}
static int fail(MSXResidentStatus s, const char *where)
{
    int e = map(s);
    snprintf(R.status, sizeof(R.status), "%s:%d", where, (int)s);
    fprintf(stderr, "RESIDENT_ERROR,stage=%s,status=%d,mapped=%d\n",
            where ? where : "unknown", (int)s, e);
    fflush(stderr);
    R.lastStatus = s;
    MSX.GpuError.code = e;
    MSX.GpuError.value = (double)s;
    MSX.ErrCode = e;
    return e;
}
/* Deterministic startup fault seam for the Phase 3C atomicity harness.  It is
   inert unless explicitly set by a test process. */
static int initialFault(const char *point)
{ const char *v=getenv("MSX_RESIDENT_INIT_FAIL"); return v && point && !_stricmp(v,point); }
/* The explicit audit seams are resolved once at Resident startup.  With no
   switch, neither poison walks nor aggregate-failure branches are entered. */
static int auditPoisonRequested(void)
{ const char *v = getenv("MSX_RESIDENT_AUDIT_POISON_CPU_MIRRORS"); return v && !strcmp(v, "1"); }
static int auditAggregateMode(void)
{
    const char *v = getenv("MSX_RESIDENT_AUDIT_FAIL_AGGREGATE");
    if (!v) return 0;
    if (!_stricmp(v, "link")) return 1;
    if (!_stricmp(v, "total")) return 2;
    return 0;
}
static int auditPoisonMirrors(const char *stage)
{
    MSXResidentStatus s = MSXresident_auditPoisonCpuMirrors();
    if (s != MSX_RESIDENT_OK) return fail(s, stage);
    if (MSXsegStorage_hybridAuditPoisonCoreMirrors())
        return fail(MSX_RESIDENT_ERR_TRANSFER, stage);
    return 0;
}
static int hasWall(void)
{ int m; for (m=1;m<=MSX.Nobjects[SPECIES];m++) if (MSX.Species[m].type == WALL) return 1; return 0; }
static int absolutePath(const char *p)
{ return p && ((p[0]=='/' || p[0]=='\\') || (p[0] && p[1]==':' && (p[2]=='/' || p[2]=='\\')) || (p[0]=='\\' && p[1]=='\\')); }
/* Capacity files are relative to the MSX file, never the runner CWD. */
static int resolveCapacityPath(char out[MAXFNAME], const char *capacity, const char *msx)
{
    char full[MAXFNAME], base[MAXFNAME], *slash;
    int n;
    if (!out || !capacity || !capacity[0] || !msx || !msx[0]) return 0;
    if (absolutePath(capacity)) { n=snprintf(out,MAXFNAME,"%s",capacity); return n>0 && n<MAXFNAME; }
    if (absolutePath(msx)) n=snprintf(full,MAXFNAME,"%s",msx);
    else { if (!_fullpath(full,msx,MAXFNAME)) return 0; n=(int)strlen(full); }
    if (n>=MAXFNAME) return 0;
    n=snprintf(base,MAXFNAME,"%s",full); if(n<=0 || n>=MAXFNAME)return 0;
    slash=strrchr(base,'\\'); { char *s=strrchr(base,'/'); if(!slash || (s&&s>slash)) slash=s; }
    if (!slash) return 0; slash[1]=0;
    n=snprintf(out,MAXFNAME,"%s%s",base,capacity);
    return n>0 && n<MAXFNAME;
}

static void runtimeFreeBuffers(void)
{
    free(R.rows); free(R.active); free(R.plan); free(R.item); free(R.out);
    free(R.c); free(R.lastc); free(R.tx); free(R.fallback); free(R.offset);
    R.rows=0; R.active=0; R.plan=0; R.item=0; R.out=0; R.c=0; R.lastc=0;
    R.tx=0; R.fallback=0; R.offset=0;
}
static int runtimeAllocateBuffers(const MSXResidentLayout *l)
{
    R.rows=(MSXResidentActiveRow*)calloc(l->totalSlots,sizeof(*R.rows));
    R.active=(MSXResidentActiveItem*)calloc(l->totalSlots,sizeof(*R.active));
    R.plan=(MSXResidentHandoffPlan*)calloc((size_t)l->nLinks+1,sizeof(*R.plan));
    R.item=(MSXResidentHandoffItem*)calloc(l->totalSlots,sizeof(*R.item));
    R.out=(MSXResidentHandoffResult*)calloc(l->totalSlots,sizeof(*R.out));
    R.c=(double*)calloc((size_t)l->totalSlots*l->speciesStride,sizeof(*R.c));
    R.lastc=(double*)calloc((size_t)l->totalSlots*l->speciesStride,sizeof(*R.lastc));
    R.tx=(MSXResidentHandoffTransaction*)calloc((size_t)l->nLinks+1,sizeof(*R.tx));
    R.fallback=(unsigned char*)calloc((size_t)l->nLinks+1,1);
    R.offset=(uint32_t*)calloc((size_t)l->nLinks+1,sizeof(*R.offset));
    if(!R.rows||!R.active||!R.plan||!R.item||!R.out||!R.c||!R.lastc||!R.tx||!R.fallback||!R.offset)
    { runtimeFreeBuffers(); return fail(MSX_RESIDENT_ERR_MEMORY,"runtime_buffers"); }
    R.activeCap=R.itemCap=l->totalSlots; R.stride=l->speciesStride;
    return 0;
}
static void makeConfig(MSXResidentConfig *c, const char *hash)
{
    memset(c,0,sizeof(*c)); c->mode=(MSXResidentMode)MSX.GpuCoreMode;c->strict=MSX.GpuStrict;
    c->guard=MSX.GpuCoreGuard;c->segmentStorage=MSX.SegmentStorage;c->solver=MSX.Solver;
    c->gpuSolver=MSX.GpuSolver;c->gpuScope=MSX.GpuReactScope;c->gpuReact=MSX.GpuReact;
    c->gpuOde=MSX.GpuOde;c->gpuEquil=MSX.GpuEquil;c->gpuFormula=MSX.GpuFormula;
    c->hasWall=hasWall();c->hasDispersion=MSX.DispersionFlag;c->capacityFile=R.resolvedCapacity;c->caseHash=hash;
}
/* All fallible Resident/Shadow setup happens before HYBRID changes Pseg
   topology.  The initial GPU image deliberately contains every owned slot
   with used=0; afterHybridInit only publishes real Core patches. */
int MSXresidentRuntime_preHybridInit(void)
{
    MSXResidentConfig c; MSXResidentStatus s; MSXResidentLayout l;
    MSXResidentPatchBatch b; MSXResidentGpuOpen o; char actual[65];
    if (MSX.GpuCoreMode == MSX_RESIDENT_OFF) return 0;
    if (R.opened) return 0;
    R.auditPoisonCpuMirrors = auditPoisonRequested();
    R.auditAggregateFail = auditAggregateMode();
    if (!MSX.InpFileName[0]) return fail(MSX_RESIDENT_ERR_PATH,"missing_inp_path");
    if (!resolveCapacityPath(R.resolvedCapacity,MSX.GpuCoreCapacityFile,MSX.MsxFile.name))
        return fail(MSX_RESIDENT_ERR_PATH,"capacity_path");
    if (!MSXresident_caseHashFiles(MSX.InpFileName,MSX.MsxFile.name,actual))
        return fail(MSX_RESIDENT_ERR_CASE_HASH,"case_hash");
    makeConfig(&c,actual);
    s=MSXresident_validateConfig(&c); if(s!=MSX_RESIDENT_OK)return fail(s,"pre_validate");
    s=MSXresident_open(c.capacityFile); if(s!=MSX_RESIDENT_OK)return fail(s,"pre_open");
    R.opened=1; MSXresident_setMode(c.mode,c.strict);
    if((s=MSXresident_verifyCaseHash(actual))!=MSX_RESIDENT_OK) { MSXresidentRuntime_close(); return fail(s,"pre_verify"); }
    if((s=MSXresident_getLayout(&l))!=MSX_RESIDENT_OK) { MSXresidentRuntime_close(); return fail(s,"pre_layout"); }
    if(MSXsegStorage_hybridReserve(&l)) { MSXresidentRuntime_close(); return fail(MSX_RESIDENT_ERR_CAPACITY,"pre_reserve"); }
    if(runtimeAllocateBuffers(&l)) { MSXresidentRuntime_close(); return MSX.ErrCode; }
    if(c.mode==MSX_RESIDENT_SHADOW) { R.lastStatus=MSX_RESIDENT_OK; strcpy(R.status,"shadow-preopened"); return 0; }
    if((s=MSXresident_beginInitialImage())!=MSX_RESIDENT_OK) { MSXresidentRuntime_close(); return fail(s,"pre_initial_begin"); }
    if(MSXsegStorage_hybridPrepareInitialImage()) { MSXresidentRuntime_close(); return fail(MSX_RESIDENT_ERR_CAPACITY,"pre_initial_prepare"); }
    if(MSXsegStorage_hybridStageInitialImage()) { MSXresidentRuntime_close(); return fail(MSX_RESIDENT_ERR_CAPACITY,"pre_initial_stage"); }
    if((s=MSXresident_getInitialBatch(&b))!=MSX_RESIDENT_OK) { MSXresidentRuntime_close(); return fail(s,"pre_snapshot"); }
    if(!l.capacity || !l.base || b.descriptorCount!=l.nLinks || b.slotCount!=l.totalSlots)
    { MSXresidentRuntime_close(); return fail(MSX_RESIDENT_ERR_ARGUMENT,"pre_layout"); }
    o.nLinks=l.nLinks;o.totalSlots=l.totalSlots;o.speciesStride=l.speciesStride;o.capacity=l.capacity;o.base=l.base;
    if(MSXgpu_prepareResidentContext()){MSXresidentRuntime_close();return fail(MSX_RESIDENT_ERR_GPU,"pre_context");}
    s=MSXresidentGpu_open(&o,&R.gpu);
    if(s==MSX_RESIDENT_OK && initialFault("initial_upload")) s=MSX_RESIDENT_ERR_TRANSFER;
    if(s==MSX_RESIDENT_OK)
    {
        double t=0.0, elapsed=0.0;
        if (MSXgpu_profileStageEnabled()) t=MSXgpu_wallTimeMs();
        s=MSXresidentGpu_initialUpload(R.gpu,&b);
        if (MSXgpu_profileStageEnabled())
        {
            elapsed=MSXgpu_wallTimeMs()-t;
            MSXgpu_recordInitTime(MSX_INIT_INITIAL_UPLOAD,elapsed);
            MSX.GpuTimingRecord.resident_initial_upload_ms+=elapsed;
        }
    }
    if(s!=MSX_RESIDENT_OK){MSXresidentRuntime_close();return fail(s,"pre_initial_upload");}
    /* This seam represents a failed post-upload initial-image apply/sync.
       No CPU topology is published on either side of this boundary. */
    if(initialFault("initial_apply")){MSXresidentRuntime_close();return fail(MSX_RESIDENT_ERR_TRANSFER,"pre_initial_apply");}
    if(initialFault("program")){MSXresidentRuntime_close();return fail(MSX_RESIDENT_ERR_GPU,"pre_program_open");}
    if(MSXgpu_openResidentPrograms()){MSXresidentRuntime_close();return fail(MSX_RESIDENT_ERR_GPU,"pre_program_open");}
    R.lastStatus=MSX_RESIDENT_OK; strcpy(R.status,"resident-preopened-image-staged");
    return 0;
}

int MSXresidentRuntime_afterHybridInit(void)
{
    MSXResidentStatus s; MSXResidentPatchBatch b;
    if (MSX.GpuCoreMode == MSX_RESIDENT_OFF) return 0; /* absolutely no I/O */
    if (!R.opened) return fail(MSX_RESIDENT_ERR_POISONED,"after_not_preopened");
    if (MSX.GpuCoreMode == MSX_RESIDENT_RESIDENT)
    {
        /* All GPU operations that can fail completed in preHybridInit.  The
           following CPU topology + metadata publication has no allocator or
           GPU edge and therefore cannot leave a half-published Resident. */
        if (MSXsegStorage_hybridCommitInitialImage())
            return fail(MSX_RESIDENT_ERR_CAPACITY,"initial_cpu_commit");
        if ((s=MSXresident_commitInitialImage())!=MSX_RESIDENT_OK)
            return fail(s,"initial_metadata_commit");
        if (R.auditPoisonCpuMirrors && auditPoisonMirrors("audit_poison_initial"))
            return MSX.ErrCode;
        R.resident=1; R.dispatchReady=1; R.lastStatus=MSX_RESIDENT_OK;
        strcpy(R.status,"resident-react-ready-handoff-pending");
        return 0;
    }
    {
        int profileStage = MSXgpu_profileStageEnabled();
        double observeStart = profileStage ? MSXgpu_wallTimeMs() : 0.0;
        int observeError = MSXsegStorage_hybridObserveAll();
        if (profileStage)
            MSXgpu_profileRunPhase(MSX_PROFILE_RUN_TRANSPORT_RESIDENT_OBSERVE,
                                   MSXgpu_wallTimeMs() - observeStart);
        if (observeError) return fail(MSX_RESIDENT_ERR_CAPACITY,"observe");
    }
    if ((s=MSXresident_getPatches(&b))!=MSX_RESIDENT_OK) return fail(s,"patch_read");
    R.wouldDescriptors+=b.descriptorCount; R.wouldSlots+=b.slotCount;
    if (MSX.GpuCoreMode==MSX_RESIDENT_RESIDENT && (s=MSXresidentGpu_applyPatches(R.gpu,&b))!=MSX_RESIDENT_OK)
        return fail(s,"initial_patch_apply");
    MSXresident_clearPatches(); R.resident=(MSX.GpuCoreMode==MSX_RESIDENT_RESIDENT); R.dispatchReady=R.resident;
    R.lastStatus=MSX_RESIDENT_OK; strcpy(R.status,R.resident?"resident-react-ready-handoff-pending":"shadow");
    return 0;
}

int MSXresidentRuntime_flushPatches(void)
{
    MSXResidentPatchBatch b; MSXResidentStatus s;
    if (!R.opened) return 0;
    if (R.inFlight) return fail(MSX_RESIDENT_ERR_POISONED,"patch_inflight");
    if ((s=MSXresident_getPatches(&b))!=MSX_RESIDENT_OK) return fail(s,"patch_read");
    R.wouldDescriptors+=b.descriptorCount; R.wouldSlots+=b.slotCount;
    if (!b.descriptorCount && !b.slotCount) return 0;
    if (!R.resident) { MSXresident_clearPatches(); return 0; }
    { double t=0.0; if (MSXgpu_profileStageEnabled()) t=MSXgpu_wallTimeMs();
      s=MSXresidentGpu_applyPatches(R.gpu,&b);
      if (MSXgpu_profileStageEnabled()) MSX.GpuTimingRecord.resident_patch_h2d_ms+=MSXgpu_wallTimeMs()-t; }
    if (s!=MSX_RESIDENT_OK) { R.stale++; return fail(s,"patch_apply"); }
    /* Logical descriptor/slot counts are recorded here.  API bytes/calls are
       emitted by the CUDA driver wrapper from each actual cudaMemcpy call;
       passing a pointer-sized C aggregate here would be an estimate and would
       double count the real transfer. */
    MSXgpu_profileRecordPatch(b.descriptorCount, b.slotCount, 0, 0,
                              R.patchClass);
    MSXresident_clearPatches(); R.topologyVersion++; R.stateVersion++; return 0;
}

MSXResidentStatus MSXresidentRuntime_reduce(double *massBySpecies,
                                            uint32_t massCount,
                                            MSXResidentGpuReduction *reduction)
{
    MSXResidentStatus s;
    if (!R.resident || !R.gpu) return MSX_RESIDENT_DISABLED;
    if (!massBySpecies || !reduction || massCount < R.stride)
        return MSX_RESIDENT_ERR_ARGUMENT;
    if (R.auditAggregateFail == 2)
    {
        fail(MSX_RESIDENT_ERR_TRANSFER, "audit_aggregate_total");
        return MSX_RESIDENT_ERR_TRANSFER;
    }
    /* A transport step may have published a new Core/boundary split without
       reaching the outer flush point (notably the final quality step).  The
       consumer must never read a pre-patch aggregate. */
    if (MSXresidentRuntime_flushPatches()) return R.lastStatus;
    s = MSXresidentGpu_reduce(R.gpu, massBySpecies, massCount, reduction);
    if (s != MSX_RESIDENT_OK) {
        fail(s, "aggregate_reduce");
        return s;
    }
    return MSX_RESIDENT_OK;
}

MSXResidentStatus MSXresidentRuntime_reduceLink(uint32_t linkIndex,
                                                double *massBySpecies,
                                                uint32_t massCount,
                                                double *volume,
                                                MSXResidentGpuReduction *reduction)
{
    MSXResidentStatus s;
    if (!R.resident || !R.gpu) return MSX_RESIDENT_DISABLED;
    if (!massBySpecies || !volume || !reduction || massCount < R.stride ||
        linkIndex == 0 || linkIndex > (uint32_t)MSX.Nobjects[LINK])
        return MSX_RESIDENT_ERR_ARGUMENT;
    if (R.auditAggregateFail == 1)
    {
        fail(MSX_RESIDENT_ERR_TRANSFER, "audit_aggregate_link");
        return MSX_RESIDENT_ERR_TRANSFER;
    }
    if (MSXresidentRuntime_flushPatches()) return R.lastStatus;
    s = MSXresidentGpu_reduceLink(R.gpu, linkIndex, massBySpecies,
                                  massCount, volume, reduction);
    if (s != MSX_RESIDENT_OK)
    {
        fail(s, "aggregate_link");
        return s;
    }
    return MSX_RESIDENT_OK;
}

MSXResidentStatus MSXresidentRuntime_fetchSlot(uint32_t linkIndex,
                                               uint64_t parcelId,
                                               double *c, double *lastc,
                                               uint32_t stride,
                                               MSXResidentPayload *payload)
{
    MSXResidentStatus s;
    MSXResidentHandoffResult result;
    MSXResidentHandoffItem item;
    uint32_t slot, generation;
    uint64_t epoch;

    if (!R.resident || !R.gpu) return MSX_RESIDENT_DISABLED;
    if (!parcelId || !c || !lastc || !payload || stride < R.stride ||
        linkIndex == 0 || linkIndex > (uint32_t)MSX.Nobjects[LINK])
        return MSX_RESIDENT_ERR_ARGUMENT;
    if (MSXresidentRuntime_flushPatches()) return R.lastStatus;
    s = MSXresident_getSlotForParcel(linkIndex, parcelId, &slot, &generation);
    if (s != MSX_RESIDENT_OK) return s;
    s = MSXresident_getSlotIdentity(linkIndex, slot, &generation, NULL, &epoch);
    if (s != MSX_RESIDENT_OK) return s;
    memset(&item, 0, sizeof(item));
    item.linkIndex = linkIndex;
    item.slot = slot;
    item.generation = generation;
    item.pipeEpoch = epoch;
    memset(&result, 0, sizeof(result));
    s = MSXresidentRuntime_fetchBatch(&item, 1, c, lastc, stride, &result);
    if (s != MSX_RESIDENT_OK) return s;
    *payload = result.payload;
    payload->c = c;
    payload->lastc = lastc;
    return MSX_RESIDENT_OK;
}

MSXResidentStatus MSXresidentRuntime_fetchBatch(
    const MSXResidentHandoffItem *items, uint32_t count,
    double *c, double *lastc, uint32_t stride,
    MSXResidentHandoffResult *results)
{
    MSXResidentStatus s;
    MSXResidentGpuFetchOutput f;

    if (!R.resident || !R.gpu) return MSX_RESIDENT_DISABLED;
    if ((!items && count) || (!c && count) || (!lastc && count) ||
        (!results && count) || stride < R.stride || count > R.itemCap)
        return MSX_RESIDENT_ERR_ARGUMENT;
    if (MSXresidentRuntime_flushPatches()) return R.lastStatus;
    memset(&f, 0, sizeof(f));
    f.meta = results;
    f.cOut = c;
    f.lastcOut = lastc;
    f.stride = stride;
    s = MSXresidentGpu_fetchHandoffBatch(R.gpu, items, count, &f);
    if (s != MSX_RESIDENT_OK)
    {
        fail(s, count > 1 ? "selected_fetch_batch" : "selected_fetch");
        return s;
    }
    return MSX_RESIDENT_OK;
}

void MSXresidentRuntime_close(void)
{
    if (R.inFlight)
    {
        /* Close is a lifecycle boundary: never release the CUDA/core buffers
           while a token still owns an active stream submission. */
        (void)MSXgpu_abortResidentCore(R.gpu,&R.flight.gpu);
        R.inFlight=0; R.dispatchReady=0; R.flight.inFlight=0;
    }
    /* Same CUDA context: destroy dependent program objects before its mirror. */
    MSXgpu_closeResidentPrograms();
    if (R.gpu) { MSXresidentGpu_close(R.gpu); R.gpu=0; }
    MSXsegStorage_hybridAbortInitialImage();
    MSXresident_abortInitialImage();
    if (R.opened || MSXresident_isOpen()) MSXresident_close();
    free(R.rows); free(R.active); free(R.plan); free(R.item); free(R.out); free(R.c); free(R.lastc); free(R.tx); free(R.fallback); free(R.offset); memset(&R,0,sizeof(R));
}
int MSXresidentRuntime_residentNotReady(void)
{ return R.resident && !R.dispatchReady; }
int MSXresidentRuntime_isResident(void) { return R.resident; }
int MSXresidentRuntime_reactReady(void) { return R.resident && R.dispatchReady; }
int MSXresidentRuntime_handoffReady(void) { return R.resident && R.handoffReady; }
uint64_t MSXresidentRuntime_stateVersion(void)
{
    if (!R.resident) return 0;
    /* Every committed topology patch advances stateVersion together with the
       topology epoch; the monotonic runtime sequence is the stable key. */
    return R.stateVersion;
}
int MSXresidentRuntime_linkFallback(int k) { return R.resident && k>0 && R.fallback && R.fallback[k]; }
void MSXresidentRuntime_markRebalance(void) { R.patchClass = 1; }
/* Keep Resident transport displacement identical to CPU Advect: one quality
   step can displace no more than a pipe's physical volume.  Compare before
   multiplying when possible so a pathological high flow cannot overflow
   before it is capped. */
static double handoffDisplacement(int k, double q, double dt, int *wholePipe)
{
    double v = 0.785398 * MSX.Link[k].len * SQR(MSX.Link[k].diam);
    double aq = ABS(q);

    if (wholePipe) *wholePipe = 0;
    if (dt == 0.0 || aq == 0.0 || v <= 0.0) return 0.0;
    if (dt > 0.0 && aq >= v / dt)
    {
        if (wholePipe) *wholePipe = 1;
        return v;
    }
    return aq * dt;
}
static double boundaryVolume(int k, uint32_t side)
{
    double v = 0.0;
    Pseg s = side ? MSX.LastSeg[k] : MSX.FirstSeg[k];
    while (s && !MSXsegStorage_isHybridCoreSegment(s))
    {
        v += s->v;
        s = side ? s->next : s->prev;
    }
    return v;
}
static int materializePlan(int k, MSXResidentHandoffPlan *p)
{
    MSXResidentGpuFetchOutput f;
    MSXResidentStatus z;
    uint32_t o=R.offset[k];
    double t=0.0, fetchStart=0.0, prepareStart=0.0;
    double fetchMs=0.0, prepareMs=0.0;
    uint64_t fetchBytes=0, fetchCalls=0;
    uint64_t fetchBytesBefore=0, fetchCallsBefore=0;
    int handoffDetail = MSXgpu_profileDetailGroupEnabled(MSX_PROFILE_DETAIL_HANDOFF);
    memset(&f,0,sizeof(f));
    f.meta=R.out+o; f.cOut=R.c+(size_t)o*R.stride;
    f.lastcOut=R.lastc+(size_t)o*R.stride; f.stride=R.stride;
    if (MSXgpu_profileStageEnabled()) t=MSXgpu_wallTimeMs();
    if (handoffDetail)
    {
        const MSXProfileRunTotals *totals=MSXgpu_getProfileRunTotals();
        fetchBytesBefore=totals->handoff_d2h_bytes;
        fetchCallsBefore=totals->handoff_d2h_calls;
        fetchStart=MSXgpu_wallTimeMs();
    }
    z=MSXresidentGpu_fetchHandoffs(R.gpu,p,&f,p->itemCount);
    if (handoffDetail)
    {
        const MSXProfileRunTotals *totals=MSXgpu_getProfileRunTotals();
        fetchMs=MSXgpu_wallTimeMs()-fetchStart;
        fetchBytes=totals->handoff_d2h_bytes-fetchBytesBefore;
        fetchCalls=totals->handoff_d2h_calls-fetchCallsBefore;
    }
    if (handoffDetail)
        MSXgpu_profileRunPhase(MSX_PROFILE_RUN_TRANSPORT_HANDOFF_GPU_FETCH,
                               fetchMs);
    if(z!=MSX_RESIDENT_OK)return fail(z,"fallback_fetch");
    if (handoffDetail) prepareStart=MSXgpu_wallTimeMs();
    z=MSXresident_prepareHandoffTransaction(p,f.meta,p->itemCount,&R.tx[k]);
    if (handoffDetail) prepareMs=MSXgpu_wallTimeMs()-prepareStart;
    if (handoffDetail)
        MSXgpu_profileRunPhase(MSX_PROFILE_RUN_TRANSPORT_HANDOFF_CPU_PREPARE,
                               prepareMs);
    if(z!=MSX_RESIDENT_OK)return fail(z,"fallback_prepare");
    z=MSXresident_validateHandoffTransactions(&R.tx[k],1);
    if(z!=MSX_RESIDENT_OK){MSXresident_abortHandoffTransaction(&R.tx[k]);return fail(z,"fallback_final_validate");}
    MSXresident_commitHandoffTransaction(&R.tx[k]);
    R.patchClass = 0;
    if(MSXresidentRuntime_flushPatches()){R.dispatchReady=0;return MSX.ErrCode;}
    if (MSXgpu_profileStageEnabled())
        MSX.GpuTimingRecord.resident_handoff_ms+=MSXgpu_wallTimeMs()-t;
    if (MSXgpu_profileDetailGroupEnabled(MSX_PROFILE_DETAIL_HANDOFF))
    {
        double elapsed = MSXgpu_wallTimeMs() - t;
        MSXgpu_profileRecordFallbackHandoff(0.0, fetchMs, prepareMs, 0.0, 0.0,
            p->itemCount, fetchBytes, fetchCalls);
    }
    return 0;
}
int MSXresidentRuntime_beginStep(double dt)
{
    uint32_t k, off = 0;
    MSXResidentStatus z;
    int handoffDetail;
    double planTimer = 0.0;
    if (!R.resident) return 0;
    if (dt < 0.0) return fail(MSX_RESIDENT_ERR_ARGUMENT, "handoff_plan");
    R.handoffReady = 0;
    R.itemCount = 0;
    R.handoffPlanMs = 0.0;
    memset(R.plan, 0, ((size_t)MSX.Nobjects[LINK] + 1) * sizeof(*R.plan));
    memset(R.fallback, 0, (size_t)MSX.Nobjects[LINK] + 1);
    memset(R.tx, 0, ((size_t)MSX.Nobjects[LINK] + 1) * sizeof(*R.tx));
    if (MSXresidentRuntime_flushPatches()) return MSX.ErrCode;
    /* The transport consumers require both physical endpoints to be CPU
       Boundary-readable.  Resolve short-Core/zero-flow/reverse cases before
       building the handoff plan, then publish any staged demotions once. */
    for (k = 1; k <= (uint32_t)MSX.Nobjects[LINK]; k++)
    {
        if (!MSXsegStorage_isHybridLink((int)k)) continue;
        if (MSXsegStorage_hybridEnsureEndpointBoundary((int)k, 0) ||
            MSXsegStorage_hybridEnsureEndpointBoundary((int)k, 1))
            return MSX.ErrCode ? MSX.ErrCode :
                   ERR_GPU_KERNEL_RUNTIME_ERROR;
    }
    if (MSXresidentRuntime_flushPatches()) return MSX.ErrCode;
    handoffDetail = MSXgpu_profileDetailGroupEnabled(MSX_PROFILE_DETAIL_HANDOFF);
    if (handoffDetail) planTimer = MSXgpu_wallTimeMs();
    for (k = 1; k <= (uint32_t)MSX.Nobjects[LINK]; k++)
    {
        int wholePipe;
        /* The segment list is re-oriented by flowdirchanged() before this
           hook, so FirstSeg is always the transport-leading endpoint.  The
           resident ring follows that same physical order; choosing by the
           sign of Q here would hand off the trailing endpoint on reverse
           flow and leave a Core row exposed to evalnodeinflow(). */
        uint32_t side = 0u;
        double displacement = handoffDisplacement((int)k, MSX.Q[k], dt, &wholePipe);
        R.offset[k] = off;
        z = wholePipe ?
            MSXresident_planAllHandoffInto(k, side, R.item + off, R.itemCap - off, &R.plan[k]) :
            MSXresident_planHandoffInto(k, displacement, 1.0, side,
                                        boundaryVolume((int)k, side), MSX.GpuStrict,
                                        R.item + off, R.itemCap - off, &R.plan[k]);
        /* A request can consume the complete resident Core before it reaches
           LINKVOL because the two CPU boundary bands are not resident. */
        if (z == MSX_RESIDENT_ERR_CAPACITY)
            z = MSXresident_planAllHandoffInto(k, side, R.item + off,
                                               R.itemCap - off, &R.plan[k]);
        if (z == MSX_RESIDENT_FALLBACK_WHOLE_LINK)
        {
            z = MSXresident_planAllHandoffInto(k, side, R.item + off,
                                               R.itemCap - off, &R.plan[k]);
            if (z != MSX_RESIDENT_OK) return fail(z, "fallback_plan_all");
            R.fallback[k] = 1;
            if (materializePlan((int)k, &R.plan[k])) return MSX.ErrCode;
            R.fallbacks++;
            R.plan[k].itemCount = 0;
        }
        else if (z != MSX_RESIDENT_OK)
            return fail(z, "handoff_plan");
        off += R.plan[k].itemCount;
    }
    R.itemCount = off;
    if (handoffDetail) R.handoffPlanMs = MSXgpu_wallTimeMs() - planTimer;
    return 0;
}
int MSXresidentRuntime_completeHandoffs(void)
{
    uint32_t k, fetchPerformed = 0; MSXResidentStatus z;
    uint64_t fetchBytes=0, fetchCalls=0;
    uint64_t fetchBytesBefore=0, fetchCallsBefore=0;
    double t = 0.0, fetchTimer = 0.0, prepareTimer = 0.0;
    double validateTimer = 0.0, commitTimer = 0.0;
    double fetchMs = 0.0, prepareMs = 0.0, validateMs = 0.0, commitMs = 0.0;
    int handoffDetail;
    if(!R.resident)return 0;
    handoffDetail = MSXgpu_profileDetailGroupEnabled(MSX_PROFILE_DETAIL_HANDOFF);
    if (MSXgpu_profileStageEnabled()) t=MSXgpu_wallTimeMs();
    if (handoffDetail)
    {
        const MSXProfileRunTotals *totals=MSXgpu_getProfileRunTotals();
        fetchBytesBefore=totals->handoff_d2h_bytes;
        fetchCallsBefore=totals->handoff_d2h_calls;
        fetchTimer=MSXgpu_wallTimeMs();
    }
    /* Normal plans are packed in R.item/R.out order. Fetch the whole flat
       set once; CPU transaction preparation remains per-link but performs no
       additional GPU or D2H operation. The actual bytes/calls are recorded by
       the CUDA API wrapper; logical handoff counters below stay separate. */
    if (R.itemCount)
    {
        MSXResidentGpuFetchOutput f;
        memset(&f,0,sizeof(f)); f.meta=R.out; f.cOut=R.c;
        f.lastcOut=R.lastc; f.stride=R.stride;
        z=MSXresidentGpu_fetchHandoffBatch(R.gpu,R.item,R.itemCount,&f);
        if(z!=MSX_RESIDENT_OK){
            if (handoffDetail)
                MSXgpu_profileRunPhase(MSX_PROFILE_RUN_TRANSPORT_HANDOFF_GPU_FETCH,
                                       MSXgpu_wallTimeMs() - fetchTimer);
            for(uint32_t j=1;j<=(uint32_t)MSX.Nobjects[LINK];j++)MSXresident_abortHandoffTransaction(&R.tx[j]);
            return fail(z,"handoff_fetch");
        }
        fetchPerformed=1;
    }
    if (handoffDetail)
    {
        const MSXProfileRunTotals *totals=MSXgpu_getProfileRunTotals();
        fetchMs=MSXgpu_wallTimeMs()-fetchTimer;
        fetchBytes=totals->handoff_d2h_bytes-fetchBytesBefore;
        fetchCalls=totals->handoff_d2h_calls-fetchCallsBefore;
    }
    if (handoffDetail)
        MSXgpu_profileRunPhase(MSX_PROFILE_RUN_TRANSPORT_HANDOFF_GPU_FETCH,
                               fetchMs);
    if (handoffDetail) prepareTimer = MSXgpu_wallTimeMs();
    for(k=1;k<=(uint32_t)MSX.Nobjects[LINK];k++) if(R.plan[k].itemCount)
    {
        uint32_t o=R.offset[k];
        z=MSXresident_prepareHandoffTransaction(&R.plan[k],R.out+o,R.plan[k].itemCount,&R.tx[k]);
        if(z!=MSX_RESIDENT_OK){
            if (handoffDetail)
                MSXgpu_profileRunPhase(MSX_PROFILE_RUN_TRANSPORT_HANDOFF_CPU_PREPARE,
                                       MSXgpu_wallTimeMs() - prepareTimer);
            for(uint32_t j=1;j<=(uint32_t)MSX.Nobjects[LINK];j++)MSXresident_abortHandoffTransaction(&R.tx[j]);
            return fail(z,"handoff_prepare");
        }
    }
    if (handoffDetail) prepareMs = MSXgpu_wallTimeMs() - prepareTimer;
    if (handoffDetail)
        MSXgpu_profileRunPhase(MSX_PROFILE_RUN_TRANSPORT_HANDOFF_CPU_PREPARE,
                               prepareMs);
    if (handoffDetail) validateTimer = MSXgpu_wallTimeMs();
    z=MSXresident_validateHandoffTransactions(R.tx,(uint32_t)MSX.Nobjects[LINK]+1);
    if(z!=MSX_RESIDENT_OK){for(k=1;k<=(uint32_t)MSX.Nobjects[LINK];k++)MSXresident_abortHandoffTransaction(&R.tx[k]);return fail(z,"handoff_final_validate");}
    if (handoffDetail) validateMs = MSXgpu_wallTimeMs() - validateTimer;
    if (handoffDetail) commitTimer = MSXgpu_wallTimeMs();
    for(k=1;k<=(uint32_t)MSX.Nobjects[LINK];k++)if(R.tx[k].opaque)MSXresident_commitHandoffTransaction(&R.tx[k]);
    if (handoffDetail) commitMs = MSXgpu_wallTimeMs() - commitTimer;
    /* CPU transactions are committed atomically as a batch. A GPU flush
       failure is fail-stop; it never claims CPU topology was rolled back. */
    R.patchClass = 0;
    if(MSXresidentRuntime_flushPatches()){R.dispatchReady=0;return MSX.ErrCode;}
    R.handoffReady=1;
    if (MSXgpu_profileStageEnabled()) MSX.GpuTimingRecord.resident_handoff_ms+=MSXgpu_wallTimeMs()-t;
    if (handoffDetail && fetchPerformed)
        MSXgpu_profileRecordNormalHandoff(R.handoffPlanMs,
                                          fetchMs, prepareMs, validateMs,
                                          commitMs, R.itemCount, fetchBytes,
                                          fetchCalls);
    return 0;
}
static MSXResidentStatus runtimeBuildActive(uint32_t *activeOut)
{
    MSXResidentStatus s; uint32_t n=0,i,active=0;
    double timer=0.0, phaseStart=0.0;
    int stage=MSXgpu_profileStageEnabled();
    if (!activeOut) return MSX_RESIDENT_ERR_ARGUMENT;
    if (stage) timer=MSXgpu_wallTimeMs();
    s=MSXresident_enumerateActive(R.rows,R.activeCap,&n);
    if (stage)
    {
        double elapsed=MSXgpu_wallTimeMs()-timer;
        MSXgpu_profileRunPhase(MSX_PROFILE_RUN_REACT_ENUMERATE,elapsed);
        MSX.GpuTimingRecord.resident_enumerate_filter_ms+=elapsed;
    }
    if(s!=MSX_RESIDENT_OK)return s;
    if(stage)phaseStart=MSXgpu_wallTimeMs();
    for(i=0;i<n;i++)
        if(MSXsegStorage_isHybridCoreSlotIdentity((int)R.rows[i].linkIndex,
                                                   (int)R.rows[i].slot,
                                                   R.rows[i].parcelId))
        {
            MSXResidentActiveItem *a=&R.active[active];
            a->linkIndex=R.rows[i].linkIndex; a->globalRow=R.rows[i].globalRow;
            a->generation=R.rows[i].generation; a->descriptorEpoch=R.rows[i].descriptorEpoch;
            a->volume=R.rows[i].volume; a->hyd=MSX.Link[a->linkIndex].HydVar;
            active++;
        }
    if(stage)
    {
        double elapsed=MSXgpu_wallTimeMs()-phaseStart;
        MSXgpu_profileRunPhase(MSX_PROFILE_RUN_REACT_IDENTITY_FILTER,elapsed);
        MSX.GpuTimingRecord.resident_enumerate_filter_ms+=elapsed;
    }
    if(active!=n)return MSX_RESIDENT_ERR_GENERATION;
    if(stage)MSX.GpuTimingRecord.resident_active_rows+=active;
    *activeOut=active; return MSX_RESIDENT_OK;
}

/* A stale caller token must never leave the device submission live: its
   buffers are still owned by the runtime, and continuing would make the next
   step reuse those buffers concurrently.  Drain the authoritative flight
   copy, then fail closed. */
static void runtimeDrainFlight(void)
{
    if (R.inFlight && R.gpu && R.flight.gpu.inFlight)
        (void)MSXgpu_abortResidentCore(R.gpu, &R.flight.gpu);
    R.inFlight = 0;
    R.flight.inFlight = 0;
    R.flight.magic = 0;
    R.dispatchReady = 0;
}

int MSXresidentRuntime_submitCore(double dt, MSXResidentRuntimeToken *token)
{
    MSXResidentStatus s; MSXResidentActiveBatch batch; uint32_t active=0; int err;
    double flushStart=0.0; int stage=MSXgpu_profileStageEnabled();
    if (!token || token->magic || token->inFlight) return fail(MSX_RESIDENT_ERR_ARGUMENT,"submit_token");
    if (!R.resident || !R.dispatchReady || !R.gpu) return fail(MSX_RESIDENT_ERR_POISONED,"submit_not_ready");
    if (R.inFlight) return fail(MSX_RESIDENT_ERR_POISONED,"submit_inflight");
    if (dt < 0.0) return fail(MSX_RESIDENT_ERR_ARGUMENT,"submit_dt");
    if(stage)flushStart=MSXgpu_wallTimeMs();
    err=MSXresidentRuntime_flushPatches();
    if(stage)MSXgpu_profileRunPhase(MSX_PROFILE_RUN_REACT_FLUSH,
                                    MSXgpu_wallTimeMs()-flushStart);
    if(err)return MSX.ErrCode;
    s=runtimeBuildActive(&active);
    if(s!=MSX_RESIDENT_OK){R.dispatchReady=0;return fail(s,"active_identity_filter");}
    memset(token,0,sizeof(*token));
    batch.item=R.active; batch.itemCount=active;
    err=MSXgpu_submitResidentCore(R.gpu,&batch,dt,&token->gpu);
    if(err){R.dispatchReady=0;(void)fail(MSX_RESIDENT_ERR_GPU,"submit_gpu");return err;}
    R.sequence++; if(!R.sequence)R.sequence++;
    R.inFlight=1; R.stateVersion++;
    token->magic=MSX_RESIDENT_RUNTIME_TOKEN_MAGIC; token->stepId=R.sequence;
    token->topologyVersion=R.topologyVersion; token->stateVersion=R.stateVersion;
    token->gpu.batchId=R.sequence;
    token->batchCount=active; token->inFlight=1; R.flight=*token;
    return 0;
}

int MSXresidentRuntime_finishCore(MSXResidentRuntimeToken *token)
{
    MSXResidentGpuReactResult out; int err, i, m;
    double phaseStart=0.0; int stage=MSXgpu_profileStageEnabled();
    if(!token || token->magic!=MSX_RESIDENT_RUNTIME_TOKEN_MAGIC || !token->inFlight ||
       !R.inFlight || token->stepId!=R.flight.stepId ||
       token->gpu.batchId!=R.flight.gpu.batchId ||
       token->topologyVersion!=R.topologyVersion || token->stateVersion!=R.flight.stateVersion)
    {
        runtimeDrainFlight();
        if(token)memset(token,0,sizeof(*token));
        return fail(MSX_RESIDENT_ERR_POISONED,"finish_stale_token");
    }
    memset(&out,0,sizeof(out)); if(stage)phaseStart=MSXgpu_wallTimeMs();
    err=MSXgpu_finishResidentCore(R.gpu,&token->gpu,&out);
    if(err)
    {
        R.inFlight=0; R.dispatchReady=0; token->inFlight=0; token->magic=0;
        R.flight.inFlight=0; R.flight.magic=0;
        return fail(MSX_RESIDENT_ERR_GPU,"finish_gpu");
    }
    R.inFlight=0; token->inFlight=0; token->magic=0; R.flight.inFlight=0;
    R.stateVersion++;
    if(!out.reacted || out.reactedStride<(uint32_t)MSX.Nobjects[SPECIES]+1 ||
       out.reactedLinkCount<(uint32_t)MSX.Nobjects[LINK]+1)
    { R.dispatchReady=0; return fail(MSX_RESIDENT_ERR_TRANSFER,"finish_reacted"); }
    if(stage)phaseStart=MSXgpu_wallTimeMs();
    for(i=1;i<=MSX.Nobjects[LINK];i++)
        for(m=1;m<=MSX.Nobjects[SPECIES];m++)
            MSX.Link[i].reacted[m]+=out.reacted[(size_t)i*out.reactedStride+m];
    if(stage)MSXgpu_profileRunPhase(MSX_PROFILE_RUN_REACT_REACTED_MERGE,
                                    MSXgpu_wallTimeMs()-phaseStart);
    if (R.auditPoisonCpuMirrors && auditPoisonMirrors("audit_poison_after_react"))
        return MSX.ErrCode;
    return 0;
}

int MSXresidentRuntime_abortCore(MSXResidentRuntimeToken *token)
{
    int err;
    if(!token || token->magic!=MSX_RESIDENT_RUNTIME_TOKEN_MAGIC || !token->inFlight ||
       !R.inFlight || token->stepId!=R.flight.stepId ||
       token->gpu.batchId!=R.flight.gpu.batchId)
    { runtimeDrainFlight(); if(token)memset(token,0,sizeof(*token));
      return fail(MSX_RESIDENT_ERR_POISONED,"abort_stale_token"); }
    err=MSXgpu_abortResidentCore(R.gpu,&token->gpu);
    R.inFlight=0; R.dispatchReady=0; token->inFlight=0; token->magic=0; R.flight.inFlight=0;
    if(err) return fail(MSX_RESIDENT_ERR_GPU,"abort_gpu");
    R.lastStatus=MSX_RESIDENT_ERR_POISONED; strcpy(R.status,"resident-aborted");
    return 0;
}

int MSXresidentRuntime_reactCore(double dt)
{
    MSXResidentRuntimeToken token; int err;
    memset(&token,0,sizeof(token)); err=MSXresidentRuntime_submitCore(dt,&token);
    if(err)return err; return MSXresidentRuntime_finishCore(&token);
}
const char *MSXresidentRuntime_status(void) { return R.status; }
MSXResidentStatus MSXresidentRuntime_lastStatus(void) { return R.lastStatus; }
const char *MSXresidentRuntime_resolvedCapacityPath(void) { return R.resolvedCapacity; }
void MSXresidentRuntime_getMetrics(MSXResidentRuntimeMetrics *m)
{ if(!m)return; m->wouldDescriptorPatches=R.wouldDescriptors; m->wouldSlotPatches=R.wouldSlots; m->stalePatches=R.stale; m->fallbacks=R.fallbacks; m->opened=R.opened; m->resident=R.resident; }
