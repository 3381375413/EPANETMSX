#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "msxresident_runtime.h"
#include "msxresident_core_cuda.h"
#include "msxresident_hash.h"
#include "msxsegment_storage.h"
#include "msxgpu.h"
#include "msxtypes.h"
#include "msxresident_capacity.h"
#include "msxresident_alloc.h"
#include "msxresident_upload.h"
#define free(p) MSXresidentAlloc_free(p)

extern MSXproject MSX;

typedef struct { int opened, resident, dispatchReady, handoffReady, patchClass; MSXResidentGpu *gpu;
    int directInitial;
    int auditMemory;
    uint32_t peakPatchRows, peakImportRows, peakDescriptors;
    uint32_t peakHandoffRows, peakMaterializeRows, peakSelectedRows;
    uint64_t wouldDescriptors, wouldSlots, stale, fallbacks; MSXResidentStatus lastStatus;
    MSXResidentActivePipeView *activePipe; double *pipeHyd;
    uint32_t activeCap,activeLinks, stride;
    MSXResidentHandoffPlan *plan; MSXResidentHandoffItem *item; MSXResidentHandoffResult *out;
    MSXResidentHandoffTarget *targets; MSXResidentHandoffTransaction *tx; unsigned char *fallback;
    uint32_t *offset; uint32_t itemCap, itemCount;
    double handoffPlanMs;
    int auditPoisonCpuMirrors, auditAggregateFail, auditHyd, inFlight;
    uint64_t topologyVersion;
    uint64_t readWindow;
    MSXResidentTransaction transaction;
    uint64_t activeIteratorPasses, activeRowsAppended, activeFullRowCopies,
             activeBuilderAborts;
    MSXResidentGpuActiveWriter activeWriter;
    MSXResidentRuntimeToken flight;
    char status[96], resolvedCapacity[MAXFNAME]; } Runtime;
static Runtime R;
static int runtimeEndReadWindow(void)
{
    int error;
    if(!R.readWindow)return 0;
    error=MSXsegStorage_hybridEndReadWindow(R.readWindow);
    if(!error)R.readWindow=0;
    return error;
}
static int DirectInitialPlanning;
int MSXresidentRuntime_directInitialPlanning(void){return DirectInitialPlanning;}
uint64_t MSXresidentRuntime_initialScratchBytes(uint32_t links,uint32_t tanks,uint32_t stride)
{
    uint64_t rows=(uint64_t)links+tanks+1,bytes;
    if(!stride||rows>UINT64_MAX/stride/sizeof(double))return UINT64_MAX;
    bytes=rows*stride*sizeof(double);
    if(bytes>UINT64_MAX-2*MSXresidentAlloc_headerBytes()||
       (uint64_t)tanks+1>(UINT64_MAX-bytes-2*MSXresidentAlloc_headerBytes())/sizeof(MSXResidentInitialTankDraft))return UINT64_MAX;
    return bytes+((uint64_t)tanks+1)*sizeof(MSXResidentInitialTankDraft)+2*MSXresidentAlloc_headerBytes();
}
static void auditPatchDemand(const MSXResidentPatchBatch *b)
{
    uint32_t i,imports=0;
    if(!R.auditMemory)return;
    for(i=0;i<b->slotCount;++i)
        if(b->slot[i].used && b->slot[i].kind==MSX_RESIDENT_PATCH_IMPORT)++imports;
    if(b->slotCount>R.peakPatchRows)R.peakPatchRows=b->slotCount;
    if(imports>R.peakImportRows)R.peakImportRows=imports;
    if(b->descriptorCount>R.peakDescriptors)R.peakDescriptors=b->descriptorCount;
}
static void writeMemoryDemand(void)
{
    FILE *f;
    if(!R.auditMemory)return;
    f=fopen("resident_transaction_demand.csv","a+");if(!f)return;
    fseek(f,0,SEEK_END);
    if(ftell(f)==0)fprintf(f,"peak_patch_rows,peak_import_rows,peak_descriptors,peak_handoff_rows,peak_materialize_rows,peak_selected_rows\n");
    fprintf(f,"%u,%u,%u,%u,%u,%u\n",R.peakPatchRows,R.peakImportRows,R.peakDescriptors,
        R.peakHandoffRows,R.peakMaterializeRows,R.peakSelectedRows);
    fclose(f);
}
uint64_t MSXresidentRuntime_fixedHostBytes(uint32_t links,uint32_t slots,uint32_t stride)
{
 uint64_t l=(uint64_t)links+1,s=slots;
 return 9*MSXresidentAlloc_headerBytes()+
   l*(sizeof(MSXResidentActivePipeView)+MSX_RESIDENT_HYD_STRIDE*sizeof(double)+sizeof(MSXResidentHandoffPlan)+
   sizeof(MSXResidentHandoffTransaction)+1+sizeof(uint32_t))+
   s*(sizeof(MSXResidentHandoffItem)+
   sizeof(MSXResidentHandoffResult)+sizeof(MSXResidentHandoffTarget));
}

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
    if (!(s==MSX_RESIDENT_ERR_MEMORY && MSX.GpuCoreCapacityMode &&
          MSXresidentCapacity_startupAttempt()==0))
    fprintf(stderr, "RESIDENT_ERROR,stage=%s,status=%d,mapped=%d\n",
            where ? where : "unknown", (int)s, e);
    fflush(stderr);
    R.lastStatus = s;
    MSXtransaction_fail(&R.transaction,s==MSX_RESIDENT_ERR_GPU||s==MSX_RESIDENT_ERR_TRANSFER||s==MSX_RESIDENT_ERR_POISONED);
    R.dispatchReady=0;
    MSX.GpuError.code = e;
    MSX.GpuError.value = (double)s;
    MSX.ErrCode = e;
    return e;
}
static int runtimeAuditFixedLayout(const char *where)
{
    MSXResidentStatus status;
    if(!R.gpu||!MSXgpu_profileDetailGroupEnabled(MSX_PROFILE_DETAIL_DIAGNOSTIC))return 0;
    status=MSXresidentGpu_auditFixedLayout(R.gpu);
    if(status==MSX_RESIDENT_OK)return 0;
    (void)MSXresidentGpu_poison(R.gpu);
    MSXresident_poison();MSXtransaction_fail(&R.transaction,1);
    return fail(status,where);
}
/* Deterministic startup fault seam for the Phase 3C atomicity harness.  It is
   inert unless explicitly set by a test process. */
static int initialFault(const char *point)
{ const char *v=getenv("MSX_RESIDENT_INIT_FAIL"); return v && point && !_stricmp(v,point); }
/* The explicit audit seams are resolved once at Resident startup.  With no
   switch, neither poison walks nor aggregate-failure branches are entered. */
static int auditPoisonRequested(void)
{ const char *v = getenv("MSX_RESIDENT_AUDIT_POISON_CPU_MIRRORS"); return v && !strcmp(v, "1"); }
static int auditHydRequested(void)
{ const char *v = getenv("MSX_RESIDENT_AUDIT_HYD"); return v && !strcmp(v, "1"); }
/* Compare the source values that the legacy active-major path consumed with
   the corresponding pipe-major rows.  This is intentionally opt-in: the
   production path does not walk all active HydVar values a second time. */
static int auditHydTable(const MSXResidentActivePipeView *pipes, uint32_t count,
                         const MSXResidentHydView *hyd)
{
    uint32_t k,observed=0,mismatches = 0;
    if ((!pipes && count) || !hyd || !hyd->pipeHyd) return -1;
    /* Each row in one pipe consumes the identical hydraulic source. Compare
       once per pipe, preserving the old per-active mismatch count. No rows
       or temporary totalSlots allocation is needed for this read-only audit. */
    for (k = 1; k <= hyd->linkCount; ++k)
    {
        if(!pipes[k].count)continue;
        observed+=pipes[k].count;
        const double *oldHyd = MSX.Link[k].HydVar;
        const double *pipeHyd = hyd->pipeHyd +
            (size_t)k * MSX_RESIDENT_HYD_STRIDE;
        if (memcmp(oldHyd, pipeHyd,
                   MSX_RESIDENT_HYD_STRIDE * sizeof(double)) != 0)
            mismatches+=pipes[k].count;
    }
    fprintf(stderr, "RESIDENT_HYD_AUDIT,active=%u,mismatches=%u\n",
            count, mismatches);
    fflush(stderr);
    return mismatches||observed!=count ? -1 : 0;
}
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
    free(R.activePipe); free(R.pipeHyd); free(R.plan); free(R.item); free(R.out);
    free(R.targets); free(R.tx); free(R.fallback); free(R.offset);
    R.activePipe=0; R.pipeHyd=0; R.plan=0; R.item=0; R.out=0; R.targets=0;
    R.tx=0; R.fallback=0; R.offset=0;
}
static uint64_t RuntimeAllocationBytes;
static void *runtime_calloc_impl(size_t n,size_t size,unsigned site)
{if(n&&size>SIZE_MAX/n)return NULL;RuntimeAllocationBytes+=(uint64_t)n*size+MSXresidentAlloc_headerBytes();return MSXresidentAlloc_calloc(n,size,__FILE__,site);}
#define runtime_calloc(n,w) runtime_calloc_impl(n,w,__LINE__)
static int runtimeAllocateBuffers(const MSXResidentLayout *l)
{
    size_t hydValues;RuntimeAllocationBytes=0;
    if (!l || (size_t)l->nLinks + 1u > SIZE_MAX / MSX_RESIDENT_HYD_STRIDE)
        return fail(MSX_RESIDENT_ERR_OVERFLOW,"runtime_hyd_size");
    hydValues=((size_t)l->nLinks+1u)*MSX_RESIDENT_HYD_STRIDE;
    if (hydValues > SIZE_MAX / sizeof(double))
        return fail(MSX_RESIDENT_ERR_OVERFLOW,"runtime_hyd_bytes");
    R.activePipe=(MSXResidentActivePipeView*)runtime_calloc((size_t)l->nLinks+1,sizeof(*R.activePipe));
    R.pipeHyd=(double*)runtime_calloc(hydValues,sizeof(*R.pipeHyd));
    R.plan=(MSXResidentHandoffPlan*)runtime_calloc((size_t)l->nLinks+1,sizeof(*R.plan));
    R.item=(MSXResidentHandoffItem*)runtime_calloc(l->totalSlots,sizeof(*R.item));
    R.out=(MSXResidentHandoffResult*)runtime_calloc(l->totalSlots,sizeof(*R.out));
    R.targets=(MSXResidentHandoffTarget*)runtime_calloc(l->totalSlots,sizeof(*R.targets));
    R.tx=(MSXResidentHandoffTransaction*)runtime_calloc((size_t)l->nLinks+1,sizeof(*R.tx));
    R.fallback=(unsigned char*)runtime_calloc((size_t)l->nLinks+1,1);
    R.offset=(uint32_t*)runtime_calloc((size_t)l->nLinks+1,sizeof(*R.offset));
    if(!R.activePipe||!R.pipeHyd||!R.plan||!R.item||!R.out||!R.targets||!R.tx||!R.fallback||!R.offset)
    { runtimeFreeBuffers(); return fail(MSX_RESIDENT_ERR_MEMORY,"runtime_buffers"); }
    if(RuntimeAllocationBytes!=MSXresidentRuntime_fixedHostBytes(l->nLinks,l->totalSlots,l->speciesStride))
    {runtimeFreeBuffers();return fail(MSX_RESIDENT_ERR_OVERFLOW,"runtime_size_ledger");}
    R.activeCap=R.itemCap=l->totalSlots;R.activeLinks=l->nLinks; R.stride=l->speciesStride;
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
static MSXResidentStatus stageUploads(void *owner,const MSXResidentSlotPatch *patch,uint32_t count)
{return MSXresidentGpu_stageImports((MSXResidentGpu *)owner,patch,count);}
static int finishInitialImage(void);
static int preHybridAttempt(int direct)
{
    MSXResidentConfig c; MSXResidentStatus s; MSXResidentLayout l;
    MSXResidentPatchBatch b; MSXResidentGpuOpen o; MSXResidentMemoryConfig memoryConfig; char actual[65];
    if (MSX.GpuCoreMode == MSX_RESIDENT_OFF) return 0;
    if(MSXsegStorage_hybridWriteAllowed())return ERR_PIPE_RING_CAPACITY;
    if (R.opened) return 0;
    {const char *a=getenv("MSX_RESIDENT_MEMORY_AUDIT");R.auditMemory=a&&!strcmp(a,"1");}
    R.auditPoisonCpuMirrors = auditPoisonRequested();
    R.auditHyd = auditHydRequested();
    R.auditAggregateFail = auditAggregateMode();
    if (!MSX.InpFileName[0]) return fail(MSX_RESIDENT_ERR_PATH,"missing_inp_path");
    if (!MSX.GpuCoreCapacityMode && !resolveCapacityPath(R.resolvedCapacity,MSX.GpuCoreCapacityFile,MSX.MsxFile.name))
        return fail(MSX_RESIDENT_ERR_PATH,"capacity_path");
    if (!MSXresident_caseHashFiles(MSX.InpFileName,MSX.MsxFile.name,actual))
        return fail(MSX_RESIDENT_ERR_CASE_HASH,"case_hash");
    makeConfig(&c,actual);
    s=MSXresident_validateConfig(&c); if(s!=MSX_RESIDENT_OK)return fail(s,"pre_validate");
    if(!MSXtransaction_begin(&R.transaction))return fail(MSX_RESIDENT_ERR_POISONED,"initial_reentry");
    s=MSX.GpuCoreCapacityMode ? MSXresidentCapacity_openPlan(actual) : MSXresident_open(c.capacityFile);
    if(s!=MSX_RESIDENT_OK)return fail(s,"pre_open");
    R.opened=1;R.directInitial=direct; MSXresident_setMode(c.mode,c.strict);
    if((s=MSXresident_verifyCaseHash(actual))!=MSX_RESIDENT_OK) { MSXresidentRuntime_close(); return fail(s,"pre_verify"); }
    if((s=MSXresident_getLayout(&l))!=MSX_RESIDENT_OK) { MSXresidentRuntime_close(); return fail(s,"pre_layout"); }
    if(c.mode==MSX_RESIDENT_RESIDENT&&!MSX.GpuCoreCapacityMode){
        s=MSXresidentCapacity_fileBudget(&l);
        if(s!=MSX_RESIDENT_OK){MSXresidentRuntime_close();return fail(s,"file_budget");}
    }
    { int reserveError=MSXsegStorage_hybridReserve(&l);
      if(reserveError) { MSXresidentRuntime_close(); return fail(reserveError==ERR_MEMORY?MSX_RESIDENT_ERR_MEMORY:MSX_RESIDENT_ERR_CAPACITY,"pre_reserve"); } }
    if(runtimeAllocateBuffers(&l)) { MSXresidentRuntime_close(); return MSX.ErrCode; }
    if(!MSXtransaction_advance(&R.transaction,MSX_TX_PREPARED)||
       !MSXtransaction_advance(&R.transaction,MSX_TX_STAGING))return fail(MSX_RESIDENT_ERR_POISONED,"initial_state");
    if(c.mode==MSX_RESIDENT_SHADOW) { R.lastStatus=MSX_RESIDENT_OK; strcpy(R.status,"shadow-preopened"); return 0; }
    o.nLinks=l.nLinks;o.totalSlots=l.totalSlots;o.speciesStride=l.speciesStride;o.capacity=l.capacity;o.base=l.base;
    if(MSXgpu_prepareResidentContext()){MSXresidentRuntime_close();return fail(MSX_RESIDENT_ERR_GPU,"pre_context");}
    {uint64_t freeDevice;
     s=MSXresidentGpu_availableMemory(&freeDevice);
     if(s!=MSX_RESIDENT_OK){MSXresidentRuntime_close();return fail(s,"pre_memory_query");}
     MSXresidentCapacity_setDeviceBaseline(freeDevice);}
    s=MSXresidentCapacity_memoryConfig(&memoryConfig);
    if(s==MSX_RESIDENT_OK)s=MSXresidentGpu_openConfigured(&o,&memoryConfig,&R.gpu);
    if(s==MSX_RESIDENT_OK){const char *audit=getenv("MSX_RESIDENT_MEMORY_AUDIT");
        if(audit&&!strcmp(audit,"1"))s=MSXresidentGpu_writeAllocationManifest(R.gpu,"resident_cuda_allocations.csv");}
    if(s==MSX_RESIDENT_OK && (initialFault("allocation_oom") ||
       (initialFault("allocation_oom_once") && MSXresidentCapacity_startupAttempt()==0)))
        s=MSX_RESIDENT_ERR_MEMORY;
    if(s==MSX_RESIDENT_OK && initialFault("initial_upload")) s=MSX_RESIDENT_ERR_TRANSFER;
    if(s==MSX_RESIDENT_OK)s=MSXupload_open(memoryConfig.transferBatchRows<l.totalSlots?
        memoryConfig.transferBatchRows:l.totalSlots,l.speciesStride,stageUploads,R.gpu);
    if(s==MSX_RESIDENT_OK)s=MSXresident_beginInitialImage();
    if(s==MSX_RESIDENT_OK&&direct){R.lastStatus=MSX_RESIDENT_OK;strcpy(R.status,"resident-direct-prepared");return 0;}
    if(s==MSX_RESIDENT_OK&&MSXsegStorage_hybridPrepareInitialImage())s=MSX_RESIDENT_ERR_CAPACITY;
    if(s==MSX_RESIDENT_OK&&MSXsegStorage_hybridStageInitialImage())s=MSX_RESIDENT_ERR_CAPACITY;
    if(s!=MSX_RESIDENT_OK){MSXresidentRuntime_close();return fail(s,"pre_initial_stage");}
    return finishInitialImage();
}

static int finishInitialImage(void)
{
    MSXResidentStatus s;MSXResidentLayout l;MSXResidentPatchBatch b;
    s=MSXresident_getLayout(&l);
    if(s==MSX_RESIDENT_OK)s=MSXresident_getInitialBatch(&b);
    if(s==MSX_RESIDENT_OK&&(!l.capacity||!l.base||b.descriptorCount!=l.nLinks||b.slotCount!=l.totalSlots))s=MSX_RESIDENT_ERR_ARGUMENT;
    if(s==MSX_RESIDENT_OK)s=MSXupload_flush();
    if(s==MSX_RESIDENT_OK)
    {
        double t=0.0, elapsed=0.0;
        if (MSXgpu_profileStageEnabled()) t=MSXgpu_wallTimeMs();
        auditPatchDemand(&b);
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
    if(R.directInitial&&(s=MSXresidentGpu_invalidateHyd(R.gpu))!=MSX_RESIDENT_OK){
        MSXresidentRuntime_close();return fail(s,"direct_initial_hyd_invalidate");
    }
    R.lastStatus=MSX_RESIDENT_OK; strcpy(R.status,"resident-preopened-image-staged");
    return 0;
}

int MSXresidentRuntime_preHybridInit(void)
{
    int error=preHybridAttempt(0);
    if(error && MSX.GpuCoreCapacityMode &&
       (error==ERR_MEMORY || error==ERR_GPU_MEMORY_ALLOCATION_FAILED) &&
       MSXresidentCapacity_retryBudget())
    {
        const char *reason=getenv("MSX_RESIDENT_INIT_FAIL");
        MSXresidentCapacity_noteStartupRetry(reason?reason:"initial_memory_allocation");
        /* No topology has been committed. Destroy all staging arenas before
           the one permitted smaller fixed-layout attempt. */
        MSXresidentRuntime_close();
        error=MSXsegStorage_hybridResetUncommitted();
        if(error)return fail(error==ERR_MEMORY?MSX_RESIDENT_ERR_MEMORY:MSX_RESIDENT_ERR_POISONED,"retry_staging_reset");
        MSX.ErrCode=0; MSX.OutOfMemory=0;
        memset(&MSX.GpuError,0,sizeof(MSX.GpuError));
        error=preHybridAttempt(0);
    }
    return error;
}
int MSXresidentRuntime_prepareDirectInitial(void)
{
    int error;if(MSXsegStorage_hybridWriteAllowed())return ERR_PIPE_RING_CAPACITY;DirectInitialPlanning=1;error=preHybridAttempt(1);DirectInitialPlanning=0;return error;
}
int MSXresidentRuntime_finishDirectInitial(void)
{
    if(MSXsegStorage_hybridWriteAllowed())return ERR_PIPE_RING_CAPACITY;
    if(!R.opened||!R.directInitial)return fail(MSX_RESIDENT_ERR_POISONED,"direct_not_prepared");
    return finishInitialImage();
}

int MSXresidentRuntime_afterHybridInit(void)
{
    if(MSXsegStorage_hybridWriteAllowed())return ERR_PIPE_RING_CAPACITY;
    MSXResidentStatus s; MSXResidentPatchBatch b;
    if (MSX.GpuCoreMode == MSX_RESIDENT_OFF) return 0; /* absolutely no I/O */
    if (!R.opened) return fail(MSX_RESIDENT_ERR_POISONED,"after_not_preopened");
    if(!MSXtransaction_advance(&R.transaction,MSX_TX_APPLYING))return fail(MSX_RESIDENT_ERR_POISONED,"initial_publish_state");
    R.transaction.cpuChanged=1;
    if (MSX.GpuCoreMode == MSX_RESIDENT_RESIDENT)
    {
        /* All GPU operations that can fail completed in preHybridInit.  The
           following CPU topology + metadata publication has no allocator or
           GPU edge and therefore cannot leave a half-published Resident. */
        if (MSXsegStorage_hybridCommitInitialImage())
            return fail(MSX_RESIDENT_ERR_CAPACITY,"initial_cpu_commit");
        if ((s=MSXresident_commitInitialImage())!=MSX_RESIDENT_OK)
            return fail(s,"initial_metadata_commit");
        if(runtimeAuditFixedLayout("initial_fixed_layout"))return MSX.ErrCode;
        /* Direct initialization publishes tank lists in the same APPLYING
           interval. Its visible version is completed only after those stores. */
        if(R.directInitial)return 0;
        /* MSXinit is a real model/reaction lifecycle boundary.  The device
           Hyd table is not valid until the first post-init reaction submits
           and finishes a complete pipe-major candidate. */
        if (!R.directInitial&&(s=MSXresidentGpu_invalidateHyd(R.gpu))!=MSX_RESIDENT_OK)
            return fail(s,"initial_hyd_invalidate");
        if (R.auditPoisonCpuMirrors && auditPoisonMirrors("audit_poison_initial"))
            return MSX.ErrCode;
        if(!MSXtransaction_complete(&R.transaction)||!MSXtransaction_commit(&R.transaction))
            return fail(MSX_RESIDENT_ERR_POISONED,"initial_publish");
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
    auditPatchDemand(&b);
    if (MSX.GpuCoreMode==MSX_RESIDENT_RESIDENT && (s=MSXresidentGpu_applyPatches(R.gpu,&b))!=MSX_RESIDENT_OK)
        return fail(s,"initial_patch_apply");
    MSXresident_clearPatches(); R.resident=(MSX.GpuCoreMode==MSX_RESIDENT_RESIDENT); R.dispatchReady=R.resident;
    if(!MSXtransaction_complete(&R.transaction)||!MSXtransaction_commit(&R.transaction))
        return fail(MSX_RESIDENT_ERR_POISONED,"shadow_publish");
    R.lastStatus=MSX_RESIDENT_OK; strcpy(R.status,R.resident?"resident-react-ready-handoff-pending":"shadow");
    return 0;
}
int MSXresidentRuntime_completeDirectInitial(void)
{
    if(MSXsegStorage_hybridWriteAllowed())return ERR_PIPE_RING_CAPACITY;
    if(!R.opened||!R.directInitial||R.transaction.state!=MSX_TX_APPLYING)
        return fail(MSX_RESIDENT_ERR_POISONED,"direct_completion_state");
    if(R.auditPoisonCpuMirrors&&auditPoisonMirrors("audit_poison_initial"))return MSX.ErrCode;
    if(!MSXtransaction_complete(&R.transaction)||!MSXtransaction_commit(&R.transaction))
        return fail(MSX_RESIDENT_ERR_POISONED,"direct_initial_publish");
    R.resident=1;R.dispatchReady=1;R.lastStatus=MSX_RESIDENT_OK;
    strcpy(R.status,"resident-react-ready-handoff-pending");
    return 0;
}

static int runtimeFlushPatches(int continuing)
{
    MSXResidentPatchBatch b; MSXResidentStatus s;
    if (!R.opened) return 0;
    if(MSXsegStorage_hybridWriteAllowed())return ERR_PIPE_RING_CAPACITY;
    if (R.inFlight || (!continuing&&!MSXtransaction_readable(&R.transaction)))
        return fail(MSX_RESIDENT_ERR_POISONED,"patch_inflight");
    if((s=MSXupload_flush())!=MSX_RESIDENT_OK)return fail(s,"upload_stage");
    if ((s=MSXresident_getPatches(&b))!=MSX_RESIDENT_OK) return fail(s,"patch_read");
    R.wouldDescriptors+=b.descriptorCount; R.wouldSlots+=b.slotCount;
    if (!b.descriptorCount && !b.slotCount) return 0;
    auditPatchDemand(&b);
    if (!R.resident) { MSXresident_clearPatches(); return 0; }
    if(!continuing){
        if(!MSXtransaction_begin(&R.transaction)||!MSXtransaction_advance(&R.transaction,MSX_TX_PREPARED)||
           !MSXtransaction_advance(&R.transaction,MSX_TX_STAGING)||!MSXtransaction_advance(&R.transaction,MSX_TX_APPLYING))
            return fail(MSX_RESIDENT_ERR_POISONED,"patch_state");
    }else if(R.transaction.state!=MSX_TX_APPLYING)return fail(MSX_RESIDENT_ERR_POISONED,"handoff_publish_state");
    R.transaction.cpuChanged=1;
    MSXsegStorage_hybridRebalanceAddPatchCounts(b.descriptorCount,
                                                b.slotCount);
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
    if(R.topologyVersion==UINT64_MAX)return fail(MSX_RESIDENT_ERR_OVERFLOW,"topology_version");
    MSXresident_clearPatches(); R.topologyVersion++;
    if(!MSXtransaction_complete(&R.transaction)||!MSXtransaction_commit(&R.transaction))
        return fail(MSX_RESIDENT_ERR_POISONED,"patch_publish");
    return 0;
}
int MSXresidentRuntime_flushPatches(void){return runtimeFlushPatches(0);}

/* Reads retain the transaction fail-stop contract even though the public
   flush write entry must reject an owned read window without mutation. */
static MSXResidentStatus runtimeFlushForRead(void)
{
    int error;
    if (R.inFlight || !MSXtransaction_readable(&R.transaction))
    {
        (void)fail(MSX_RESIDENT_ERR_POISONED,"patch_inflight");
        return MSX_RESIDENT_ERR_POISONED;
    }
    error=MSXresidentRuntime_flushPatches();
    if(error==ERR_PIPE_RING_CAPACITY)return MSX_RESIDENT_ERR_ARGUMENT;
    return error ? R.lastStatus : MSX_RESIDENT_OK;
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
    if ((s=runtimeFlushForRead())!=MSX_RESIDENT_OK) return s;
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
    if ((s=runtimeFlushForRead())!=MSX_RESIDENT_OK) return s;
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
    if ((s=runtimeFlushForRead())!=MSX_RESIDENT_OK) return s;
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
    int rebalanceProfile = MSXsegStorage_hybridRebalanceProfileActive();
    double preflushStart = rebalanceProfile ? MSXgpu_wallTimeMs() : 0.0;
    double fetchStart;

    if (!R.resident || !R.gpu) return MSX_RESIDENT_DISABLED;
    if ((!items && count) || (!c && count) || (!lastc && count) ||
        (!results && count) || stride < R.stride || count > R.itemCap)
        return MSX_RESIDENT_ERR_ARGUMENT;
    s = runtimeFlushForRead();
    if (rebalanceProfile)
        MSXsegStorage_hybridRebalanceAddPhase(
            MSX_REBALANCE_PHASE_PREFLUSH,
            MSXgpu_wallTimeMs() - preflushStart);
    if (s!=MSX_RESIDENT_OK) return s;
    memset(&f, 0, sizeof(f));
    f.meta = results;
    f.cOut = c;
    f.lastcOut = lastc;
    f.stride = stride;
    fetchStart = rebalanceProfile ? MSXgpu_wallTimeMs() : 0.0;
    if(R.auditMemory&&count>R.peakSelectedRows)R.peakSelectedRows=count;
    s = MSXresidentGpu_fetchHandoffBatch(R.gpu, items, count, &f);
    if (rebalanceProfile)
        MSXsegStorage_hybridRebalanceAddPhase(
            MSX_REBALANCE_PHASE_FETCH,
            MSXgpu_wallTimeMs() - fetchStart);
    if (s != MSX_RESIDENT_OK)
    {
        fail(s, count > 1 ? "selected_fetch_batch" : "selected_fetch");
        return s;
    }
    return MSX_RESIDENT_OK;
}

MSXResidentStatus MSXresidentRuntime_fetchTargets(const MSXResidentHandoffItem *items,uint32_t count,
 MSXResidentHandoffResult *r,const MSXResidentHandoffTarget *t)
{
 MSXResidentStatus s;int detail=MSXsegStorage_hybridRebalanceProfileActive();
 double start=detail?MSXgpu_wallTimeMs():0.0;
 if(!R.resident||!R.gpu)return MSX_RESIDENT_DISABLED;
 if(count>R.itemCap)return MSX_RESIDENT_ERR_CAPACITY;
 if((s=runtimeFlushForRead())!=MSX_RESIDENT_OK)return s;
 if(detail){MSXsegStorage_hybridRebalanceAddPhase(MSX_REBALANCE_PHASE_PREFLUSH,MSXgpu_wallTimeMs()-start);start=MSXgpu_wallTimeMs();}
 if(R.auditMemory&&count>R.peakSelectedRows)R.peakSelectedRows=count;
 s=MSXresidentGpu_fetchHandoffTargets(R.gpu,items,count,r,t);
 if(detail)MSXsegStorage_hybridRebalanceAddPhase(MSX_REBALANCE_PHASE_FETCH,MSXgpu_wallTimeMs()-start);
 if(s!=MSX_RESIDENT_OK)fail(s,"selected_fetch_targets");return s;
}
void MSXresidentRuntime_close(void)
{
    writeMemoryDemand();
    if (R.resident && MSX.GpuCoreOverflow) MSXresidentCapacity_writeUsage();
    int emitDiagnosticSummary = 0;
    if (R.auditPoisonCpuMirrors || R.auditHyd || R.auditAggregateFail)
        emitDiagnosticSummary = 1;
    else if (R.opened || R.gpu)
        emitDiagnosticSummary =
            MSXresidentRuntime_diagnosticSummaryGate(
                MSXgpu_profileDetailGroupEnabled(MSX_PROFILE_DETAIL_DIAGNOSTIC),
                0);
    if (R.inFlight)
    {
        /* Close is a lifecycle boundary: never release the CUDA/core buffers
           while a token still owns an active stream submission. */
        (void)MSXgpu_abortResidentCore(R.gpu,&R.flight.gpu);
        R.inFlight=0; R.dispatchReady=0; R.flight.inFlight=0;
    }
    if(runtimeEndReadWindow())(void)fail(MSX_RESIDENT_ERR_POISONED,"close_read_window");
    if(R.gpu)(void)runtimeAuditFixedLayout("close_fixed_layout");
    /* Emit the B2 cache ledger before close/re-open invalidates the applied
       version.  These are counters only; no diagnostic timing is introduced.
       A close boundary must leave no pending submission. */
    if (R.gpu)
    {
        MSXResidentGpuTransferStats hydStats;
        if(MSXresidentGpu_getTransferStats(R.gpu,&hydStats)==MSX_RESIDENT_OK)
            MSXresidentCapacity_writeTransferStats(&hydStats);
        if (emitDiagnosticSummary &&
            MSXresidentGpu_getTransferStats(R.gpu, &hydStats) == MSX_RESIDENT_OK)
            fprintf(stderr,
                    "RESIDENT_HYD_CACHE,candidate_comparisons=%llu,uploads=%llu,skips=%llu,bytes=%llu,api_calls=%llu,applied_valid=%d,pending=%d\n",
                    (unsigned long long)hydStats.hydCandidateComparisons,
                    (unsigned long long)hydStats.hydUploads,
                    (unsigned long long)hydStats.hydSkips,
                    (unsigned long long)hydStats.hydBytes,
                    (unsigned long long)hydStats.hydApiCalls,
                    hydStats.hydAppliedValid, hydStats.hydPending);
        /* Close/re-open is the remaining model-change boundary; invalidate
           before releasing the resident buffers so no stale Hyd version can
           be reused. */
        (void)MSXresidentGpu_invalidateHyd(R.gpu);
    }
    if (emitDiagnosticSummary)
        fprintf(stderr,
                "RESIDENT_ACTIVE_BUILDER,iterator_passes=%llu,rows_appended=%llu,full_row_copies=%llu,builder_aborts=%llu\n",
                (unsigned long long)R.activeIteratorPasses,
                (unsigned long long)R.activeRowsAppended,
                (unsigned long long)R.activeFullRowCopies,
                (unsigned long long)R.activeBuilderAborts);
    /* Same CUDA context: destroy dependent program objects before its mirror. */
    MSXgpu_closeResidentPrograms();
    if (R.gpu) { MSXresidentGpu_close(R.gpu); R.gpu=0; }
    MSXupload_close();
    MSXsegStorage_hybridAbortInitialImage();
    MSXresident_abortInitialImage();
    if (R.opened || MSXresident_isOpen()) MSXresident_close();
    runtimeFreeBuffers(); memset(&R,0,sizeof(R));
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
    return R.transaction.published;
}
void MSXresidentRuntime_transactionSnapshot(MSXResidentTransaction *out)
{if(out)*out=R.transaction;}
int MSXresidentRuntime_linkFallback(int k) { return R.resident && k>0 && R.fallback && R.fallback[k]; }
void MSXresidentRuntime_markRebalance(void) { if(MSXsegStorage_hybridWriteAllowed())return;R.patchClass = 1; }
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
    f.meta=R.out+o;f.stride=R.stride;
    z=MSXresident_prepareHandoffLeases(p,f.meta,R.targets+o,&R.tx[k]);
    if(z!=MSX_RESIDENT_OK)return fail(z,"materialize_leases");
    if (MSXgpu_profileStageEnabled()) t=MSXgpu_wallTimeMs();
    if (handoffDetail)
    {
        const MSXProfileRunTotals *totals=MSXgpu_getProfileRunTotals();
        fetchBytesBefore=totals->handoff_d2h_bytes;
        fetchCallsBefore=totals->handoff_d2h_calls;
        fetchStart=MSXgpu_wallTimeMs();
    }
    if(R.auditMemory&&p->itemCount>R.peakMaterializeRows)R.peakMaterializeRows=p->itemCount;
    z=MSXresidentGpu_fetchHandoffTargets(R.gpu,p->item,p->itemCount,f.meta,R.targets+o);
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
    if(z!=MSX_RESIDENT_OK){MSXresident_abortHandoffTransaction(&R.tx[k]);return fail(z,"fallback_fetch");}
    if (handoffDetail) prepareStart=MSXgpu_wallTimeMs();
    z=MSXresident_finishHandoffLeases(&R.tx[k],f.meta);
    if (handoffDetail) prepareMs=MSXgpu_wallTimeMs()-prepareStart;
    if (handoffDetail)
        MSXgpu_profileRunPhase(MSX_PROFILE_RUN_TRANSPORT_HANDOFF_CPU_PREPARE,
                               prepareMs);
    if(z!=MSX_RESIDENT_OK){MSXresident_abortHandoffTransaction(&R.tx[k]);return fail(z,"fallback_prepare");}
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
    if(MSXsegStorage_hybridWriteAllowed())return ERR_PIPE_RING_CAPACITY;
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
    MSXresidentCapacity_recordOwnership();
    if (handoffDetail) R.handoffPlanMs = MSXgpu_wallTimeMs() - planTimer;
    return MSX.ErrCode;
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
    if(MSXsegStorage_hybridWriteAllowed())return ERR_PIPE_RING_CAPACITY;
    if(R.itemCount && !MSXtransaction_begin(&R.transaction))return fail(MSX_RESIDENT_ERR_POISONED,"handoff_reentry");
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
        if(MSXsegStorage_hybridEnsureBoundaryPoolFree(R.itemCount))return fail(MSX_RESIDENT_ERR_MEMORY,"handoff_pool");
        for(k=1;k<=(uint32_t)MSX.Nobjects[LINK];++k)if(R.plan[k].itemCount){uint32_t o=R.offset[k];
            z=MSXresident_prepareHandoffLeases(&R.plan[k],R.out+o,R.targets+o,&R.tx[k]);
            if(z){for(uint32_t j=1;j<=(uint32_t)MSX.Nobjects[LINK];++j)MSXresident_abortHandoffTransaction(&R.tx[j]);
                return fail(z,"handoff_leases");}}
        if(R.auditMemory&&R.itemCount>R.peakHandoffRows)R.peakHandoffRows=R.itemCount;
        if(!MSXtransaction_advance(&R.transaction,MSX_TX_PREPARED)||!MSXtransaction_advance(&R.transaction,MSX_TX_STAGING))
            return fail(MSX_RESIDENT_ERR_POISONED,"handoff_stage_state");
        z=MSXresidentGpu_fetchHandoffTargets(R.gpu,R.item,R.itemCount,R.out,R.targets);
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
        z=MSXresident_finishHandoffLeases(&R.tx[k],R.out+o);
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
    if(R.itemCount){
        if(!MSXtransaction_advance(&R.transaction,MSX_TX_APPLYING))return fail(MSX_RESIDENT_ERR_POISONED,"handoff_apply_state");
        R.transaction.cpuChanged=1;
    }
    for(k=1;k<=(uint32_t)MSX.Nobjects[LINK];k++)if(R.tx[k].opaque)MSXresident_commitHandoffTransaction(&R.tx[k]);
    if (handoffDetail) commitMs = MSXgpu_wallTimeMs() - commitTimer;
    /* CPU transactions are committed atomically as a batch. A GPU flush
       failure is fail-stop; it never claims CPU topology was rolled back. */
    R.patchClass = 0;
    if(runtimeFlushPatches(R.itemCount!=0)){R.dispatchReady=0;return MSX.ErrCode;}
    if(R.itemCount && R.transaction.state!=MSX_TX_COMMITTED){
        if(!MSXtransaction_complete(&R.transaction)||!MSXtransaction_commit(&R.transaction))
            return fail(MSX_RESIDENT_ERR_POISONED,"handoff_empty_publish");
    }
    R.handoffReady=1;
    if (MSXgpu_profileStageEnabled()) MSX.GpuTimingRecord.resident_handoff_ms+=MSXgpu_wallTimeMs()-t;
    if (handoffDetail && fetchPerformed)
        MSXgpu_profileRecordNormalHandoff(R.handoffPlanMs,
                                          fetchMs, prepareMs, validateMs,
                                          commitMs, R.itemCount, fetchBytes,
                                          fetchCalls);
    return 0;
}
static MSXResidentStatus runtimeActiveIdentity(const MSXResidentActiveRow *row,void *context)
{
    (void)context;
    return MSXsegStorage_isHybridCoreSlotIdentity((int)row->linkIndex,
               (int)row->slot,row->parcelId)?MSX_RESIDENT_OK:MSX_RESIDENT_ERR_GENERATION;
}
static MSXResidentStatus runtimeActiveAudit(const MSXResidentActiveRow *row,void *context)
{
    (void)context;
    if(MSXResidentCapacityAuditEnabled&&
       MSXresidentCapacity_auditReact(row->linkIndex,row->parcelId,1))
        return MSX_RESIDENT_ERR_GENERATION;
    R.activeRowsAppended++;
    return MSX_RESIDENT_OK;
}
static MSXResidentStatus runtimeActiveAuditDiagnostic(const MSXResidentActiveRow *row,void *context)
{
    ++*(uint64_t *)context; /* Writer succeeded even when the audit rejects. */
    return runtimeActiveAudit(row,NULL);
}
static MSXResidentStatus runtimeBuildActive(MSXResidentGpuActiveWriter *writer,
                                            uint32_t *activeOut)
{
    MSXResidentStatus s; uint32_t n=0,i;
    double timer=0.0, phaseStart=0.0;
    double enumMs=0.0,identityMs=0.0,sealMs=0.0,diagnosticStart=0.0;
    int stage=MSXgpu_profileStageEnabled();
    int diagnostic=MSXgpu_profileResidentDiagnosticEnabled();
    uint64_t physicalWrites=0;
    MSXResidentActiveRowCheck after=diagnostic?runtimeActiveAuditDiagnostic:runtimeActiveAudit;
    if (!writer || !activeOut) return MSX_RESIDENT_ERR_ARGUMENT;
    if (stage) timer=MSXgpu_wallTimeMs();
    if(diagnostic)diagnosticStart=MSXgpu_wallTimeMs();
    s=MSXresident_preflightActive(R.activePipe,R.activeLinks+1,R.activeCap,&n);
    if(diagnostic)enumMs=MSXgpu_wallTimeMs()-diagnosticStart;
    R.activeIteratorPasses++;
    if (stage)
    {
        double elapsed=MSXgpu_wallTimeMs()-timer;
        MSXgpu_profileRunPhase(MSX_PROFILE_RUN_REACT_ENUMERATE,elapsed);
        MSX.GpuTimingRecord.resident_enumerate_filter_ms+=elapsed;
    }
    if(s!=MSX_RESIDENT_OK){if(diagnostic)MSXgpu_profileRecordActiveStages(enumMs,0,0,0,0,1);return s;}
    s=MSXresidentGpu_beginActive(R.gpu,n,R.topologyVersion,writer);
    if(s!=MSX_RESIDENT_OK){if(diagnostic)MSXgpu_profileRecordActiveStages(enumMs,0,0,n,0,2);return s;}
    if(stage)phaseStart=MSXgpu_wallTimeMs();
    if(diagnostic)diagnosticStart=MSXgpu_wallTimeMs();
    R.activeIteratorPasses++;
    for(i=1;i<=R.activeLinks;i++)
    {
        R.activePipe[i].readWindow=&R.readWindow;
        R.activePipe[i].windowToken=R.readWindow;
        /* One exclusive read window owns both passes and the borrowed
           pointers. Empty pipes preserve the old no-callback semantics. */
        s=MSXresidentGpu_appendActivePipeView(R.gpu,writer,&R.activePipe[i],
                                        runtimeActiveIdentity,after,diagnostic?&physicalWrites:NULL);
        if(s!=MSX_RESIDENT_OK)
        {
            R.activeBuilderAborts++;
            (void)MSXresidentGpu_abortActiveBuild(R.gpu,writer);
            if(diagnostic){identityMs=MSXgpu_wallTimeMs()-diagnosticStart;MSXgpu_profileRecordActiveStages(enumMs,identityMs,0,n,physicalWrites,2);}
            return s;
        }
    }
    if(diagnostic)identityMs=MSXgpu_wallTimeMs()-diagnosticStart;
    if(stage)
    {
        double elapsed=MSXgpu_wallTimeMs()-phaseStart;
        MSXgpu_profileRunPhase(MSX_PROFILE_RUN_REACT_IDENTITY_FILTER,elapsed);
        MSX.GpuTimingRecord.resident_enumerate_filter_ms+=elapsed;
    }
    if (R.topologyVersion != writer->topologyVersion)
    {
        R.activeBuilderAborts++;
        (void)MSXresidentGpu_abortActiveBuild(R.gpu,writer);
        if(diagnostic)MSXgpu_profileRecordActiveStages(enumMs,identityMs,0,n,physicalWrites,2);
        return MSX_RESIDENT_ERR_GENERATION;
    }
    if(diagnostic)diagnosticStart=MSXgpu_wallTimeMs();
    s=MSXresidentGpu_sealActive(R.gpu,writer,R.topologyVersion);
    if(diagnostic)sealMs=MSXgpu_wallTimeMs()-diagnosticStart;
    if(s!=MSX_RESIDENT_OK)
    {
        R.activeBuilderAborts++;
        (void)MSXresidentGpu_abortActiveBuild(R.gpu,writer);
        if(diagnostic)MSXgpu_profileRecordActiveStages(enumMs,identityMs,sealMs,n,physicalWrites,3);
        return s;
    }
    if(stage)MSX.GpuTimingRecord.resident_active_rows+=n;
    if(diagnostic)MSXgpu_profileRecordActiveStages(enumMs,identityMs,sealMs,n,physicalWrites,0);
    *activeOut=n; return MSX_RESIDENT_OK;
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
    if(runtimeEndReadWindow())(void)fail(MSX_RESIDENT_ERR_POISONED,"drain_read_window");
}

int MSXresidentRuntime_submitCore(double dt, MSXResidentRuntimeToken *token)
{
    MSXResidentStatus s; MSXResidentHydView hyd;
    uint32_t active=0,k,m; int err; size_t hydValues;
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
    if(runtimeAuditFixedLayout("step_fixed_layout"))return MSX.ErrCode;
    if(!MSXtransaction_begin(&R.transaction))return fail(MSX_RESIDENT_ERR_POISONED,"react_reentry");
    if(MSXsegStorage_hybridBeginReadWindow(&R.readWindow))
        return fail(MSX_RESIDENT_ERR_POISONED,"react_read_window");
    s=runtimeBuildActive(&R.activeWriter,&active);
    if(s!=MSX_RESIDENT_OK){R.dispatchReady=0;(void)runtimeEndReadWindow();return fail(s,"active_identity_filter");}
    memset(token,0,sizeof(*token));
    hydValues=((size_t)MSX.Nobjects[LINK]+1u)*MSX_RESIDENT_HYD_STRIDE;
    memset(R.pipeHyd,0,hydValues*sizeof(double));
    for(k=1;k<=(uint32_t)MSX.Nobjects[LINK];k++)
        for(m=0;m<MSX_RESIDENT_HYD_STRIDE;m++)
            R.pipeHyd[(size_t)k*MSX_RESIDENT_HYD_STRIDE+m]=MSX.Link[k].HydVar[m];
    hyd.pipeHyd=R.pipeHyd; hyd.linkCount=(uint32_t)MSX.Nobjects[LINK];
    hyd.hydStride=MSX_RESIDENT_HYD_STRIDE; hyd.hydLayout=MSX_RESIDENT_HYD_PIPE_MAJOR;
    if (R.auditHyd && auditHydTable(R.activePipe,active,&hyd) != 0)
    { R.dispatchReady=0; R.activeBuilderAborts++;
      (void)MSXresidentGpu_abortActive(R.gpu);
      (void)runtimeEndReadWindow();
      return fail(MSX_RESIDENT_ERR_ARGUMENT,"audit_hyd"); }
    if(MSXsegStorage_hybridValidateReadWindow(R.readWindow))
    { (void)MSXresidentGpu_abortActiveBuild(R.gpu,&R.activeWriter);
      (void)runtimeEndReadWindow();
      return fail(MSX_RESIDENT_ERR_GENERATION,"react_partition_window"); }
    if(!MSXtransaction_advance(&R.transaction,MSX_TX_PREPARED)||!MSXtransaction_advance(&R.transaction,MSX_TX_STAGING)||
       !MSXtransaction_advance(&R.transaction,MSX_TX_APPLYING)){
        (void)MSXresidentGpu_abortActiveBuild(R.gpu,&R.activeWriter);
        (void)runtimeEndReadWindow();return fail(MSX_RESIDENT_ERR_POISONED,"react_stage_state");}
    err=MSXgpu_submitResidentCoreHydPrepared(R.gpu,&R.activeWriter,&hyd,dt,&token->gpu);
    if(err){R.dispatchReady=0; R.activeBuilderAborts++;
        (void)MSXresidentGpu_abortActive(R.gpu);
        (void)runtimeEndReadWindow();
        (void)fail(MSX_RESIDENT_ERR_GPU,"submit_gpu");return err;}
    R.transaction.gpuAccepted=1;R.inFlight=1;
    token->magic=MSX_RESIDENT_RUNTIME_TOKEN_MAGIC; token->stepId=R.transaction.sequence;
    token->topologyVersion=R.topologyVersion; token->stateVersion=R.transaction.working;
    token->gpu.batchId=R.transaction.sequence;
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
       token->topologyVersion!=R.topologyVersion || token->stateVersion!=R.flight.stateVersion ||
       token->stateVersion!=R.transaction.working || R.transaction.state!=MSX_TX_APPLYING)
    {
        runtimeDrainFlight();
        if(token)memset(token,0,sizeof(*token));
        return fail(MSX_RESIDENT_ERR_POISONED,"finish_stale_token");
    }
    if(runtimeEndReadWindow()){runtimeDrainFlight();return fail(MSX_RESIDENT_ERR_POISONED,"finish_read_window");}
    memset(&out,0,sizeof(out)); if(stage)phaseStart=MSXgpu_wallTimeMs();
    err=MSXgpu_finishResidentCore(R.gpu,&token->gpu,&out);
    if(err)
    {
        R.inFlight=0; R.dispatchReady=0; token->inFlight=0; token->magic=0;
        R.flight.inFlight=0; R.flight.magic=0;
        return fail(MSX_RESIDENT_ERR_GPU,"finish_gpu");
    }
    R.inFlight=0; token->inFlight=0; token->magic=0; R.flight.inFlight=0;
    if(!out.reacted || out.reactedStride<(uint32_t)MSX.Nobjects[SPECIES]+1 ||
       out.reactedLinkCount<(uint32_t)MSX.Nobjects[LINK]+1)
    { R.dispatchReady=0; return fail(MSX_RESIDENT_ERR_TRANSFER,"finish_reacted"); }
    if(stage)phaseStart=MSXgpu_wallTimeMs();
    for(i=1;i<=MSX.Nobjects[LINK];i++)
        for(m=1;m<=MSX.Nobjects[SPECIES];m++)
            MSX.Link[i].reacted[m]+=out.reacted[(size_t)i*out.reactedStride+m];
    if(stage)MSXgpu_profileRunPhase(MSX_PROFILE_RUN_REACT_REACTED_MERGE,
                                    MSXgpu_wallTimeMs()-phaseStart);
    MSXresidentCapacity_recordDeviceMemory();
    if (R.auditPoisonCpuMirrors && auditPoisonMirrors("audit_poison_after_react"))
        return MSX.ErrCode;
    if(!MSXtransaction_complete(&R.transaction)||!MSXtransaction_commit(&R.transaction))
        return fail(MSX_RESIDENT_ERR_POISONED,"react_publish");
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
    if(runtimeEndReadWindow()){runtimeDrainFlight();return fail(MSX_RESIDENT_ERR_POISONED,"abort_read_window");}
    err=MSXgpu_abortResidentCore(R.gpu,&token->gpu);
    R.inFlight=0; R.dispatchReady=0; token->inFlight=0; token->magic=0; R.flight.inFlight=0;
    if(err) return fail(MSX_RESIDENT_ERR_GPU,"abort_gpu");
    R.lastStatus=MSX_RESIDENT_ERR_POISONED; strcpy(R.status,"resident-aborted");
    MSXtransaction_fail(&R.transaction,1);
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
{ if(!m)return; m->wouldDescriptorPatches=R.wouldDescriptors; m->wouldSlotPatches=R.wouldSlots; m->stalePatches=R.stale; m->fallbacks=R.fallbacks; m->activeIteratorPasses=R.activeIteratorPasses; m->activeRowsAppended=R.activeRowsAppended; m->activeFullRowCopies=R.activeFullRowCopies; m->activeBuilderAborts=R.activeBuilderAborts; m->opened=R.opened; m->resident=R.resident; }

#ifdef MSX_RESIDENT_TEST_API
/* Exported only in an isolated validation build. Production ABI is unchanged. */
static MSXResidentRuntimeToken TestFlight;
__declspec(dllexport) int MSXTESTresidentReadWindow(int begin)
{
    int error;
    if(R.inFlight)return ERR_PIPE_RING_CAPACITY;
    if(begin)return MSXsegStorage_hybridBeginReadWindow(&R.readWindow);
    error=runtimeEndReadWindow();if(!error)MSX.ErrCode=0;
    return error;
}
__declspec(dllexport) int MSXTESTresidentLifecycleState(uint64_t *values)
{
    int k;uint64_t count=0;if(!values)return 1;
    if(MSX.Link)for(k=1;k<=MSX.Nobjects[LINK];k++)count+=MSX.Link[k].nsegs;
    values[0]=(uint64_t)(uintptr_t)MSX.FirstSeg;values[1]=(uint64_t)(uintptr_t)MSX.LastSeg;
    values[2]=(uint64_t)(uintptr_t)MSX.Link;values[3]=count;
    values[4]=MSX.Saveflag;values[5]=MSX.ProjectOpened;values[6]=R.inFlight;
    return 0;
}
__declspec(dllexport) int MSXTESTresidentSubmit(double dt,int reenter)
{
    MSXResidentRuntimeToken second={0};
    if(reenter)return MSXresidentRuntime_submitCore(dt,&second);
    memset(&TestFlight,0,sizeof(TestFlight));return MSXresidentRuntime_submitCore(dt,&TestFlight);
}
__declspec(dllexport) int MSXTESTresidentFinish(void)
{return MSXresidentRuntime_finishCore(&TestFlight);}
__declspec(dllexport) int MSXTESTresidentRead(void)
{
    double mass[65];MSXResidentGpuReduction result;
    return (int)MSXresidentRuntime_reduce(mass,65,&result);
}
static MSXResidentStatus runtimeTestReadIdentity(MSXResidentHandoffItem *item,uint64_t *parcelId,int needTwo)
{
    uint32_t k,slot,found;
    for(k=1;k<=R.activeLinks;++k){
        found=0;
        for(slot=0;slot<R.activePipe[k].capacity;++slot){
            uint32_t generation;uint64_t id,epoch;
            if(MSXresident_getSlotIdentity(k,slot,&generation,&id,&epoch)!=MSX_RESIDENT_OK)continue;
            if(!found){item->linkIndex=k;item->slot=slot;item->generation=generation;
                item->pipeEpoch=epoch;*parcelId=id;}
            if(++found>=(needTwo?2u:1u))return MSX_RESIDENT_OK;
        }
    }
    return MSX_RESIDENT_ERR_ARGUMENT;
}
__declspec(dllexport) int MSXTESTresidentReadKind(int kind,int invalid)
{
    double mass[65],c[65],lastc[65],volume;
    MSXResidentGpuReduction reduction;MSXResidentPayload payload;
    MSXResidentHandoffItem item={0};MSXResidentHandoffResult result;
    MSXResidentHandoffTarget target={0};
    uint64_t parcelId=1;
    item.linkIndex=1;item.slot=0;item.generation=1;
    if(!invalid&&runtimeTestReadIdentity(&item,&parcelId,0)!=MSX_RESIDENT_OK)return MSX_RESIDENT_ERR_ARGUMENT;
    target.c=c;target.lastc=lastc;
    switch(kind){
    case 0:return MSXresidentRuntime_reduce(invalid?NULL:mass,65,&reduction);
    case 1:return MSXresidentRuntime_reduceLink(item.linkIndex,invalid?NULL:mass,65,&volume,&reduction);
    case 2:return MSXresidentRuntime_fetchSlot(item.linkIndex,parcelId,invalid?NULL:c,lastc,65,&payload);
    case 3:return MSXresidentRuntime_fetchBatch(&item,1,invalid?NULL:c,lastc,65,&result);
    case 4:return MSXresidentRuntime_fetchTargets(invalid>=2?NULL:&item,invalid==1?R.itemCap+1:1,&result,&target);
    default:return MSX_RESIDENT_ERR_ARGUMENT;
    }
}
__declspec(dllexport) int MSXTESTresidentReadPendingPatch(int kind,uint64_t *values)
{
    MSXResidentHandoffItem item={0};uint64_t parcelId;
    MSXResidentPayload before,after;MSXResidentPatchBatch batch;
    MSXResidentStatus s;double c[65],lastc[65];
    if(!values||R.stride>65)return MSX_RESIDENT_ERR_ARGUMENT;
    s=runtimeTestReadIdentity(&item,&parcelId,1);if(s)return s;
    s=MSXresident_getSlotMetadata(item.linkIndex,item.slot,item.generation,&before);if(s)return s;
    before.hresponse+=1.0;
    s=MSXresident_stageMeta(item.linkIndex,item.slot,item.generation,&before);if(s)return s;
    s=MSXresident_stageReverse(item.linkIndex);if(s)return s;
    s=MSXresident_stageReverse(item.linkIndex);if(s)return s;
    s=MSXresident_getPatches(&batch);if(s)return s;
    values[0]=batch.slotCount;values[1]=batch.descriptorCount;
    values[2]=R.transaction.published;values[3]=R.topologyVersion;
    s=(MSXResidentStatus)MSXTESTresidentReadKind(kind==5?4:kind,kind==5?2:0);
    if(kind!=5){
        if(s)return s;
        s=MSXresidentRuntime_fetchSlot(item.linkIndex,parcelId,c,lastc,65,&after);if(s)return s;
        if(memcmp(&before.volume,&after.volume,sizeof(double))||
           memcmp(&before.hresponse,&after.hresponse,sizeof(double)))return MSX_RESIDENT_ERR_ARGUMENT;
    }else if(s!=MSX_RESIDENT_ERR_ARGUMENT)return s?s:MSX_RESIDENT_ERR_GPU;
    s=MSXresident_getPatches(&batch);if(s)return s;
    values[4]=batch.slotCount;values[5]=batch.descriptorCount;
    values[6]=R.transaction.published;values[7]=R.topologyVersion;
    return kind==5?MSX_RESIDENT_ERR_ARGUMENT:MSX_RESIDENT_OK;
}
__declspec(dllexport) int MSXTESTresidentReadState(uint64_t *values)
{
    if(!values)return 1;
    values[0]=R.topologyVersion;values[1]=R.readWindow;
    values[2]=R.inFlight;values[3]=(uint64_t)(int64_t)R.lastStatus;
    values[4]=(uint64_t)(int64_t)MSX.ErrCode;return 0;
}
__declspec(dllexport) int MSXTESTresidentFlushWrite(void)
{return MSXresidentRuntime_flushPatches();}
__declspec(dllexport) int MSXTESTresidentState(uint64_t *values)
{
    if(!values)return 1;
    values[0]=R.transaction.sequence;values[1]=R.transaction.working;
    values[2]=R.transaction.published;values[3]=R.transaction.completed;
    values[4]=R.transaction.state;return 0;
}
__declspec(dllexport) int MSXTESTresidentActiveStorage(uint64_t *values)
{
    void *owned[9]={R.activePipe,R.pipeHyd,R.plan,R.item,R.out,R.targets,R.tx,R.fallback,R.offset};
    uint64_t actual=0;unsigned i;
    if(!values)return 1;
    for(i=0;i<9;i++)if(owned[i])actual+=MSXresidentAlloc_chargedBytes(owned[i]);
    values[0]=R.activeLinks;values[1]=R.activeCap;values[2]=R.stride;
    values[3]=sizeof(MSXResidentActivePipeView);
    values[4]=R.activePipe?MSXresidentAlloc_payloadBytes(R.activePipe):0;
    values[5]=R.activePipe?MSXresidentAlloc_chargedBytes(R.activePipe):0;
    values[6]=actual;
    values[7]=R.activePipe?MSXresidentRuntime_fixedHostBytes(R.activeLinks,R.activeCap,R.stride):0;
    values[8]=MSXresidentAlloc_headerBytes();
    values[9]=R.activePipe?RuntimeAllocationBytes:0;
    values[10]=sizeof(MSXResidentActiveRow);
    return 0;
}
__declspec(dllexport) int MSXTESTresidentBudget(uint64_t *values)
{
    MSXResidentBudget snapshot;if(!values)return 1;
    MSXresidentBudget_snapshot(MSXresidentBudget_global(),&snapshot);
    values[0]=snapshot.allocated[0]+snapshot.allocated[1]+snapshot.reserved[0]+snapshot.reserved[1];
    values[1]=snapshot.allocated[2]+snapshot.reserved[2];values[2]=snapshot.hostLimit;
    values[3]=snapshot.deviceLimit;return 0;
}
#endif

#ifdef MSX_RESIDENT_TEST_API
#include "msxresident_inventory.h"
void MSXinv_Runtime(MSXInventory *s)
{
 INV_HEAP(s,R.activePipe);
 INV_HEAP(s,R.pipeHyd);
 INV_HEAP(s,R.plan);
 INV_HEAP(s,R.item);
 INV_HEAP(s,R.out);
 INV_HEAP(s,R.targets);
 INV_HEAP(s,R.tx);
 INV_HEAP(s,R.fallback);
 INV_HEAP(s,R.offset);
 MSXinv_tag(s,"Runtime.fixedHostModelBytes",R.activePipe?MSXresidentRuntime_fixedHostBytes(R.activeLinks,R.activeCap,R.stride):0);MSXinv_tag(s,"Runtime.startupAllocationBytes",R.activePipe?RuntimeAllocationBytes:0);MSXinv_tag(s,"Runtime.activeLinks",R.activeLinks);MSXinv_tag(s,"Runtime.activeCap",R.activeCap);MSXinv_tag(s,"Runtime.stride",R.stride);MSXinv_tag(s,"Runtime.published",R.transaction.published);MSXinv_tag(s,"Runtime.completed",R.transaction.completed);MSXinv_tag(s,"Runtime.inFlight",R.inFlight);
 MSXinv_CUDA(s,R.gpu);
}
#endif

#ifdef MSX_RESIDENT_TEST_API
int MSXinv_quiescent(void){return !R.inFlight&&!R.readWindow;}
#endif
