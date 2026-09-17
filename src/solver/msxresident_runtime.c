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

typedef struct { int opened, resident, dispatchReady, handoffReady; MSXResidentGpu *gpu;
    uint64_t wouldDescriptors, wouldSlots, stale, fallbacks; MSXResidentStatus lastStatus;
    MSXResidentActiveRow *rows; MSXResidentActiveItem *active; uint32_t activeCap, stride;
    MSXResidentHandoffPlan *plan; MSXResidentHandoffItem *item; MSXResidentHandoffResult *out;
    double *c,*lastc; MSXResidentHandoffTransaction *tx; unsigned char *fallback;
    uint32_t *offset; uint32_t itemCap, itemCount;
    char status[96], resolvedCapacity[MAXFNAME]; } Runtime;
static Runtime R;

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
    if(s==MSX_RESIDENT_OK){double t=MSXgpu_wallTimeMs();s=MSXresidentGpu_initialUpload(R.gpu,&b);MSX.GpuTimingRecord.resident_initial_upload_ms+=MSXgpu_wallTimeMs()-t;}
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
        R.resident=1; R.dispatchReady=1; R.lastStatus=MSX_RESIDENT_OK;
        strcpy(R.status,"resident-react-ready-handoff-pending");
        return 0;
    }
    if (MSXsegStorage_hybridObserveAll()) return fail(MSX_RESIDENT_ERR_CAPACITY,"observe");
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
    if ((s=MSXresident_getPatches(&b))!=MSX_RESIDENT_OK) return fail(s,"patch_read");
    R.wouldDescriptors+=b.descriptorCount; R.wouldSlots+=b.slotCount;
    if (!b.descriptorCount && !b.slotCount) return 0;
    if (!R.resident) { MSXresident_clearPatches(); return 0; }
    s=MSXresidentGpu_applyPatches(R.gpu,&b);
    if (s!=MSX_RESIDENT_OK) { R.stale++; return fail(s,"patch_apply"); }
    MSXresident_clearPatches(); return 0;
}
void MSXresidentRuntime_close(void)
{
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
int MSXresidentRuntime_linkFallback(int k) { return R.resident && k>0 && R.fallback && R.fallback[k]; }
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
static double boundaryVolume(int k){double v=0;Pseg s=MSX.FirstSeg[k];while(s&&!MSXsegStorage_isHybridCoreSegment(s)){v+=s->v;s=s->prev;}return v;}
static int materializePlan(int k, MSXResidentHandoffPlan *p){MSXResidentGpuFetchOutput f;MSXResidentStatus z;uint32_t o=R.offset[k];double t=MSXgpu_wallTimeMs();memset(&f,0,sizeof(f));f.meta=R.out+o;f.cOut=R.c+(size_t)o*R.stride;f.lastcOut=R.lastc+(size_t)o*R.stride;f.stride=R.stride;z=MSXresidentGpu_fetchHandoffs(R.gpu,p,&f,p->itemCount);if(z!=MSX_RESIDENT_OK)return fail(z,"fallback_fetch");z=MSXresident_prepareHandoffTransaction(p,f.meta,p->itemCount,&R.tx[k]);if(z!=MSX_RESIDENT_OK)return fail(z,"fallback_prepare");z=MSXresident_validateHandoffTransactions(&R.tx[k],1);if(z!=MSX_RESIDENT_OK){MSXresident_abortHandoffTransaction(&R.tx[k]);return fail(z,"fallback_final_validate");}MSXresident_commitHandoffTransaction(&R.tx[k]);if(MSXresidentRuntime_flushPatches()){R.dispatchReady=0;return MSX.ErrCode;}MSX.GpuTimingRecord.resident_handoff_ms+=MSXgpu_wallTimeMs()-t;return 0;}
int MSXresidentRuntime_beginStep(double dt){uint32_t k,off=0;MSXResidentStatus z;if(!R.resident)return 0;if(dt<0.0)return fail(MSX_RESIDENT_ERR_ARGUMENT,"handoff_plan");R.handoffReady=0;R.itemCount=0;memset(R.plan,0,((size_t)MSX.Nobjects[LINK]+1)*sizeof(*R.plan));memset(R.fallback,0,(size_t)MSX.Nobjects[LINK]+1);memset(R.tx,0,((size_t)MSX.Nobjects[LINK]+1)*sizeof(*R.tx));if(MSXresidentRuntime_flushPatches())return MSX.ErrCode;for(k=1;k<=(uint32_t)MSX.Nobjects[LINK];k++){int wholePipe;double displacement=handoffDisplacement((int)k,MSX.Q[k],dt,&wholePipe);R.offset[k]=off;z=wholePipe?MSXresident_planAllHandoffInto(k,0,R.item+off,R.itemCap-off,&R.plan[k]):MSXresident_planHandoffInto(k,displacement,1.0,0,boundaryVolume(k),MSX.GpuStrict,R.item+off,R.itemCap-off,&R.plan[k]);/* A request can consume the complete resident Core before it reaches
           LINKVOL because the two CPU boundary bands are not resident.  This
           is a whole-Core handoff, not a capacity violation: plan every slot
           once and retain strict failure only if that fixed range cannot fit. */if(z==MSX_RESIDENT_ERR_CAPACITY)z=MSXresident_planAllHandoffInto(k,0,R.item+off,R.itemCap-off,&R.plan[k]);if(z==MSX_RESIDENT_FALLBACK_WHOLE_LINK){z=MSXresident_planAllHandoffInto(k,0,R.item+off,R.itemCap-off,&R.plan[k]);if(z!=MSX_RESIDENT_OK)return fail(z,"fallback_plan_all");R.fallback[k]=1;if(materializePlan((int)k,&R.plan[k]))return MSX.ErrCode;R.fallbacks++;R.plan[k].itemCount=0;}else if(z!=MSX_RESIDENT_OK)return fail(z,"handoff_plan");off+=R.plan[k].itemCount;}R.itemCount=off;return 0;}
int MSXresidentRuntime_completeHandoffs(void){uint32_t k;MSXResidentStatus z;double t;if(!R.resident)return 0;t=MSXgpu_wallTimeMs();for(k=1;k<=(uint32_t)MSX.Nobjects[LINK];k++)if(R.plan[k].itemCount){MSXResidentGpuFetchOutput f;uint32_t o=R.offset[k];memset(&f,0,sizeof(f));f.meta=R.out+o;f.cOut=R.c+(size_t)o*R.stride;f.lastcOut=R.lastc+(size_t)o*R.stride;f.stride=R.stride;z=MSXresidentGpu_fetchHandoffs(R.gpu,&R.plan[k],&f,R.plan[k].itemCount);if(z!=MSX_RESIDENT_OK){for(uint32_t j=1;j<=k;j++)MSXresident_abortHandoffTransaction(&R.tx[j]);return fail(z,"handoff_fetch");}z=MSXresident_prepareHandoffTransaction(&R.plan[k],f.meta,R.plan[k].itemCount,&R.tx[k]);if(z!=MSX_RESIDENT_OK){for(uint32_t j=1;j<=k;j++)MSXresident_abortHandoffTransaction(&R.tx[j]);return fail(z,"handoff_prepare");}}z=MSXresident_validateHandoffTransactions(R.tx,(uint32_t)MSX.Nobjects[LINK]+1);if(z!=MSX_RESIDENT_OK){for(k=1;k<=(uint32_t)MSX.Nobjects[LINK];k++)MSXresident_abortHandoffTransaction(&R.tx[k]);return fail(z,"handoff_final_validate");}for(k=1;k<=(uint32_t)MSX.Nobjects[LINK];k++)if(R.tx[k].opaque)MSXresident_commitHandoffTransaction(&R.tx[k]);/* CPU transactions are committed atomically as a batch. A GPU flush failure is fail-stop; it never rolls CPU topology back. */if(MSXresidentRuntime_flushPatches()){R.dispatchReady=0;return MSX.ErrCode;}R.handoffReady=1;MSX.GpuTimingRecord.resident_handoff_ms+=MSXgpu_wallTimeMs()-t;return 0;}
int MSXresidentRuntime_reactCore(double dt)
{
    MSXResidentStatus s; MSXResidentGpuReactResult out; uint32_t n=0,i,active=0,m;
    int err; double timer;
    if(!R.resident || !R.dispatchReady || !R.gpu) return fail(MSX_RESIDENT_ERR_POISONED,"react_not_ready");
    if((err=MSXresidentRuntime_flushPatches())) return err;
    timer=MSXgpu_wallTimeMs(); s=MSXresident_enumerateActive(R.rows,R.activeCap,&n);
    MSX.GpuTimingRecord.resident_enumerate_filter_ms += MSXgpu_wallTimeMs()-timer;
    if(s!=MSX_RESIDENT_OK) return fail(s,"active_enumerate");
    for(i=0;i<n;i++) if(MSXsegStorage_isHybridCoreSlotIdentity((int)R.rows[i].linkIndex,(int)R.rows[i].slot,R.rows[i].parcelId)){
        MSXResidentActiveItem *a=&R.active[active++]; a->linkIndex=R.rows[i].linkIndex; a->globalRow=R.rows[i].globalRow; a->generation=R.rows[i].generation; a->descriptorEpoch=R.rows[i].descriptorEpoch; a->volume=R.rows[i].volume; a->hyd=MSX.Link[a->linkIndex].HydVar;
    }
    memset(&out,0,sizeof(out));
    MSX.GpuTimingRecord.resident_active_rows += active;
    { MSXResidentActiveBatch b; b.item=R.active; b.itemCount=active; err=MSXgpu_reactResidentCore(R.gpu,&b,dt,&out); }
    if(err){R.dispatchReady=0;R.lastStatus=MSX_RESIDENT_ERR_POISONED;return fail(MSX_RESIDENT_ERR_POISONED,"react_dispatch");}
    if(!out.reacted || out.reactedStride<(uint32_t)MSX.Nobjects[SPECIES]+1 || out.reactedLinkCount<(uint32_t)MSX.Nobjects[LINK]+1){R.dispatchReady=0;return fail(MSX_RESIDENT_ERR_TRANSFER,"react_result");}
    for(i=1;i<=MSX.Nobjects[LINK];i++) for(m=1;m<=MSX.Nobjects[SPECIES];m++) MSX.Link[i].reacted[m]+=out.reacted[(size_t)i*out.reactedStride+m];
    return 0;
}
const char *MSXresidentRuntime_status(void) { return R.status; }
MSXResidentStatus MSXresidentRuntime_lastStatus(void) { return R.lastStatus; }
const char *MSXresidentRuntime_resolvedCapacityPath(void) { return R.resolvedCapacity; }
void MSXresidentRuntime_getMetrics(MSXResidentRuntimeMetrics *m)
{ if(!m)return; m->wouldDescriptorPatches=R.wouldDescriptors; m->wouldSlotPatches=R.wouldSlots; m->stalePatches=R.stale; m->fallbacks=R.fallbacks; m->opened=R.opened; m->resident=R.resident; }
