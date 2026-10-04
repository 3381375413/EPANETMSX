#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#define WIN32_LEAN_AND_MEAN
#define NOGDI
#include <windows.h>
#include <psapi.h>
#pragma comment(lib,"psapi.lib")
#include "msxresident_capacity.h"
#include "msxresident_capacity_search.h"
#include "msxresident_pool.h"
#include "msxresident_runtime.h"
#include "msxresident_hash.h"
#include "msxresident_memory_config.h"
#include "msxsegment_storage.h"
#include "msxqual_shared.h"
#include "epanet2.h"
#include "msxresident_alloc.h"
#include "msxresident_budget.h"
#include "msxgpu.h"
#define calloc(n,w) MSXresidentAlloc_calloc(n,w,__FILE__,__LINE__)
#define free(p) MSXresidentAlloc_free(p)
extern MSXproject MSX;
int MSXResidentCapacityAuditEnabled;
typedef struct { uint64_t id; unsigned char owner,seen; } AuditParcel;
typedef struct { AuditParcel *parcel;uint32_t count; } AuditPipe;
typedef struct {
 uint32_t corePeak,cpuPeak,spillPeak,demandPeak;
 uint64_t spillSteps,cpuSteps,gpuSteps;
 uint64_t spillRows,demandRows;
 double cpuReactMs;
 int64_t firstSpill;
} Usage;
static struct {
 uint32_t links,stride,slots;
 MSXResidentCapacityPipe *pipe;
 uint32_t *limit,*guard,*trial;
 Usage *usage;
 uint64_t hostAvailable,deviceBudget,hostBudget,initialBytes;
 uint64_t hostAvailableCommit,sampleManagedHost,scratchReserve,temporaryReserve,uploadReserve,cpuQuota;
 uint64_t admissionHostBudget,admissionDeviceBudget,freeObjects;
 uint64_t baselineScratch,baselineTemporary,baselineUpload;
 uint64_t baselineHybrid;
 MSXResidentBudget baseline;
 MSXResidentResolvedOptions resolved;
 MSXResidentMemoryEstimate memory;
 MSXResidentMemoryConfig memoryConfig;
 MSXResidentMemoryEstimate programMemory;
 char hydHash[65],inpHash[65],msxHash[65];
 double scanMs,budgetMs;
 int retries;
 char retryReason[96];
 AuditPipe *audit;
 uint64_t auditSteps,auditParcels;
 uint64_t expectedSteps;
 uint64_t initialFreeDevice,maxDeviceFreeDelta,deviceSamples;
} Plan;

int MSXresidentCapacity_interval(int64_t a,int64_t b,int64_t step,
                                 uint64_t *count,int64_t *minimum)
{
 int64_t first,last,length;
 if(!count||!minimum||a<0||b<a||step<=0)return 0;
 *count=0;*minimum=0;if(a==b)return 1;
 *count=(uint64_t)(b/step+(b%step!=0)-a/step);
 length=b-a;first=step-a%step;last=b%step;
 *minimum=length<first?length:first;
 if(last&&last<*minimum)*minimum=last;
 return *count>0&&*minimum>0;
}
int MSXresidentCapacity_predict(MSXResidentCapacityPipe *p,uint32_t maximum)
{
 uint64_t upper,estimate,core,margin,coreUpper;
 if(!p||!maximum||!isfinite(p->volume)||p->volume<0||p->initial>maximum||
    (p->guard!=2&&p->guard!=4))return 0;
 upper=p->positiveSteps>maximum-p->initial?maximum:p->initial+p->positiveSteps;
 estimate=p->initial;
 if(p->positiveSteps){
  if(!isfinite(p->vmin)||p->vmin<=0)return 0;
  /* Compare before division and conversion: no Inf-to-integer conversion. */
  if(upper<=1||p->volume>=p->vmin*(double)(upper-1))estimate=upper;
  else {double ratio=ceil(p->volume/p->vmin);estimate=(uint64_t)ratio+1;}
  if(estimate<p->initial)estimate=p->initial;if(estimate>upper)estimate=upper;
 }
 p->upper=(uint32_t)upper;p->predicted=(uint32_t)estimate;
 core=estimate>2*p->guard?estimate-2*p->guard:0;
 coreUpper=upper>2*p->guard?upper-2*p->guard:0;
 margin=(core+19)/20;if(margin<4)margin=4;
 p->requested=(uint32_t)(core?(core+margin<coreUpper?core+margin:coreUpper):0);
 return 1;
}
int MSXresidentCapacity_getPrediction(uint32_t link,MSXResidentCapacityPipe *p)
{if(!p||!Plan.pipe||!link||link>Plan.links)return 0;*p=Plan.pipe[link];return 1;}
static int readBytes(FILE *f,void *p,size_t n,MSXResidentSha256 *hash)
{if(fread(p,1,n,f)!=n)return 0;MSXresident_sha256Add(hash,p,n);return 1;}
static int skipFinite(FILE *f,uint64_t count,MSXResidentSha256 *hash)
{
 REAL4 values[256];uint32_t i,n;
 while(count){n=count>256?256:(uint32_t)count;
  if(!readBytes(f,values,(size_t)n*sizeof(*values),hash))return 0;
  for(i=0;i<n;++i)if(!isfinite(values[i]))return 0;
  count-=n;
 }return 1;
}
void MSXresidentCapacity_close(void)
{uint32_t k;if(Plan.audit){for(k=1;k<=Plan.links;++k)free(Plan.audit[k].parcel);free(Plan.audit);}
 free(Plan.pipe);free(Plan.limit);free(Plan.guard);free(Plan.trial);free(Plan.usage);memset(&Plan,0,sizeof(Plan));MSXResidentCapacityAuditEnabled=0;}
int MSXresidentCapacity_prepare(void)
{
 FILE *f=NULL;INT4 header[8],time,step;REAL4 *q=NULL;
 MSXResidentSha256 hash;int64_t expected=0;uint32_t k;
 double start=MSXgpu_wallTimeMs();MEMORYSTATUSEX host;
 MSXresidentCapacity_close();
 if(!MSX.GpuCoreCapacityMode)return 0;
 if(MSX.GpuCoreCapacityMode!=1||MSX.GpuCoreMode!=MSX_RESIDENT_RESIDENT||
    !MSX.GpuCoreOverflow||MSX.GpuCoreCapacityFile[0]||MSX.Qstep<=0||MSX.MaxSegments<=0)
  {MSX.ErrCode=ERR_GPU_UNSUPPORTED_FEATURE;return MSX.ErrCode;}
 Plan.links=(uint32_t)MSX.Nobjects[LINK];Plan.stride=(uint32_t)MSX.Nobjects[SPECIES]+1;
 Plan.pipe=(MSXResidentCapacityPipe*)calloc((size_t)Plan.links+1,sizeof(*Plan.pipe));
 Plan.limit=(uint32_t*)calloc((size_t)Plan.links+1,sizeof(*Plan.limit));
 Plan.guard=(uint32_t*)calloc((size_t)Plan.links+1,sizeof(*Plan.guard));
 Plan.trial=(uint32_t*)calloc((size_t)Plan.links+1,sizeof(*Plan.trial));
 Plan.usage=(Usage*)calloc((size_t)Plan.links+1,sizeof(*Plan.usage));
 q=(REAL4*)calloc((size_t)Plan.links+1,sizeof(*q));
 if(!Plan.pipe||!Plan.limit||!Plan.guard||!Plan.trial||!Plan.usage||!q)goto memory;
 memset(&host,0,sizeof(host));host.dwLength=sizeof(host);
 if(!GlobalMemoryStatusEx(&host))goto memory;Plan.hostAvailable=host.ullAvailPhys;
 Plan.hostAvailableCommit=host.ullAvailPageFile;
 {MSXResidentBudget snapshot;MSXresidentBudget_snapshot(MSXresidentBudget_global(),&snapshot);
  Plan.sampleManagedHost=snapshot.allocated[0]+snapshot.allocated[1]+snapshot.reserved[0]+snapshot.reserved[1];}
 for(k=1;k<=Plan.links;++k){
  MSXResidentCapacityPipe *p=&Plan.pipe[k];
  p->volume=MSXqual_pipeVolume(&MSX.Link[k]);p->guard=MSX.GpuCoreGuard;
  if(!isfinite(p->volume)||p->volume<0)goto bad;
  p->initial=p->volume>0?(uint32_t)MIN(100,MSX.MaxSegments):0;
  Plan.initialBytes+=(uint64_t)p->initial*(sizeof(struct Sseg)+(uint64_t)2*Plan.stride*sizeof(double));
  Plan.guard[k]=p->guard;Plan.usage[k].firstSpill=-1;
 }
 f=fopen(MSX.HydFile.name,"rb");if(!f)goto bad;MSXresident_sha256Init(&hash);
 if(!readBytes(f,header,sizeof(header),&hash)||header[0]!=MAGICNUMBER||
    header[2]!=MSX.Nobjects[NODE]||header[3]!=(INT4)Plan.links||
    header[7]<0||(int64_t)header[7]*1000!=MSX.Dur||MSX.HydOffset!=(long)sizeof(header))goto bad;
 for(;;){
  uint64_t count;int64_t minimum,a,b;
  if(!readBytes(f,&time,sizeof(time),&hash)||time<0||(int64_t)time*1000!=expected||
     !skipFinite(f,(uint64_t)2*MSX.Nobjects[NODE],&hash)||
     !readBytes(f,q+1,(size_t)Plan.links*sizeof(*q),&hash)||
     !skipFinite(f,(uint64_t)2*Plan.links,&hash)||!readBytes(f,&step,sizeof(step),&hash)||step<0)goto bad;
  a=expected;b=a+(int64_t)step*1000;
  /* EPANET can save the final hydraulic interval beyond Duration (for
     example Duration=10min, hydraulic step=30min). Quality consumes only
     its overlap with Duration; the terminal record retains hydraulic time. */
  if(a>=MSX.Dur&&step)goto bad;
  if(!MSXresidentCapacity_interval(a,a<MSX.Dur?MIN(b,MSX.Dur):a,MSX.Qstep,&count,&minimum))goto bad;
  if(UINT64_MAX-Plan.expectedSteps<count)goto bad;Plan.expectedSteps+=count;
  for(k=1;k<=Plan.links;++k){
   MSXResidentCapacityPipe *p=&Plan.pipe[k];double flow,v;int direction;
   if(!isfinite(q[k]))goto bad;flow=MSXqual_effectiveFlow(q[k]);
   if(!count)continue;
   direction=flow>0?1:flow<0?-1:0;
   if(!direction){p->zeroMs+=(uint64_t)(b-a);continue;}
   flow=fabs(flow);if(!p->qmin||flow<p->qmin)p->qmin=flow;if(flow>p->qmax)p->qmax=flow;
   v=flow*((double)minimum/1000.0);if(!isfinite(v)||v<=0)goto bad;
   if(!p->vmin||v<p->vmin)p->vmin=v;
   if(UINT64_MAX-p->positiveSteps<count)goto bad;p->positiveSteps+=count;
   if(p->lastDirection&&p->lastDirection!=direction)++p->reversals;
   p->lastDirection=direction;
  }
  if(!step){unsigned char eof;
   if(a<MSX.Dur||!readBytes(f,&eof,1,&hash)||eof!=0x1a||fgetc(f)!=EOF||ferror(f))goto bad;
   break;
  }expected=b;
 }
 MSXresident_sha256Finish(&hash,Plan.hydHash);fclose(f);f=NULL;free(q);q=NULL;
 for(k=1;k<=Plan.links;++k)if(!MSXresidentCapacity_predict(&Plan.pipe[k],(uint32_t)MSX.MaxSegments))goto bad;
 if(!MSXresident_sha256File(MSX.InpFileName,Plan.inpHash)||!MSXresident_sha256File(MSX.MsxFile.name,Plan.msxHash))goto bad;
 Plan.scanMs=MSXgpu_wallTimeMs()-start;return 0;
memory:MSX.ErrCode=ERR_MEMORY;goto done;
bad:MSX.ErrCode=ERR_READ_HYD_FILE;
done:if(f)fclose(f);free(q);MSXresidentCapacity_close();return MSX.ErrCode;
}
MSXResidentStatus MSXresidentCapacity_memoryConfig(MSXResidentMemoryConfig *c)
{
 if(!c)return MSX_RESIDENT_ERR_ARGUMENT;
 if(Plan.memoryConfig.version){*c=Plan.memoryConfig;return MSX_RESIDENT_OK;}
 {MSXResidentResolvedOptions resolved;
  if(!MSXmemory_resolve(&MSX.ResidentMemoryOptions,0,&resolved))return MSX_RESIDENT_ERR_CONFIG;
  c->version=1;c->hydLayout=MSX_RESIDENT_HYD_PIPE_MAJOR;c->transferBatchRows=resolved.values.batchRows;
  return MSX_RESIDENT_OK;}
}
typedef struct {
 uint64_t slots,initialPool,poolBacking,cpuUsed,baselinePageable,baselinePinned,baselineDevice;
 uint64_t gpuHost,gpuPinned,gpuDevice,coreHost,hybridHost,runtimeHost,poolIndexSaving,host;
 uint64_t initialCpu,initialTank,initialCacheBytes,initialTankDraftBytes;
 uint64_t scratchHeadroom,temporaryHeadroom,uploadHeadroom;
 uint64_t hybridFull,hybridExisting;
 int valid,cpuFits,hostFits,deviceFits;
} CapacityCost;
static int evaluateCost(const uint32_t *admission,const uint32_t *guards,const uint32_t *capacity,
                        const MSXResidentMemoryConfig *config,
                        MSXResidentMemoryEstimate *memory,CapacityCost *cost)
{
 uint64_t host,extra,available=0,pattern[3],initialPool,slots=0,objects;
 uint32_t k;int direct=MSXresidentRuntime_directInitialPlanning();Pseg cursor;AllocForecast forecast;
 memset(cost,0,sizeof(*cost));
 if(!admission||!guards)return 0;
 for(k=1;k<=Plan.links;++k){uint32_t physical=capacity?capacity[k]:(admission[k]?admission[k]:1);
  uint32_t initial=Plan.pipe?Plan.pipe[k].initial:
   (MSXqual_pipeVolume(&MSX.Link[k])>0?(uint32_t)MIN(100,MSX.MaxSegments):0);
  uint32_t core=initial>2*guards[k]?initial-2*guards[k]:0;
  if(!physical||admission[k]>physical||(guards[k]!=2&&guards[k]!=4))return 0;
  slots+=physical;if(slots>UINT32_MAX)return 0;
  if(direct)cost->initialCpu+=initial-MIN(core,admission[k]);
 }
 if(direct){
  for(k=1;k<=(uint32_t)MSX.Nobjects[TANK];++k)if(MSX.Tank[k].a!=0.0)
   cost->initialTank+=MSX.Tank[k].mixModel==MIX2?2:1;
  cost->initialCpu+=cost->initialTank;
  cost->initialCacheBytes=((uint64_t)Plan.links+MSX.Nobjects[TANK]+1)*Plan.stride*sizeof(double)+MSXresidentAlloc_headerBytes();
  cost->initialTankDraftBytes=((uint64_t)MSX.Nobjects[TANK]+1)*sizeof(MSXResidentInitialTankDraft)+MSXresidentAlloc_headerBytes();
  /* Cache and tank drafts are real allocations before this frozen baseline.
     Record their exact compiled sizes, but never charge them a second time. */
  if(cost->initialCacheBytes+cost->initialTankDraftBytes!=
     MSXresidentRuntime_initialScratchBytes(Plan.links,(uint32_t)MSX.Nobjects[TANK],Plan.stride))return 0;
 }
 initialPool=MSXresidentPool_initial((uint32_t)slots);
 cost->slots=slots;cost->initialPool=initialPool;
 if(slots>UINT32_MAX||MSXresidentGpu_estimateConfigured(Plan.links,(uint32_t)slots,Plan.stride,config,memory)!=MSX_RESIDENT_OK)return 0;
 cost->gpuHost=memory->hostBytes;cost->gpuPinned=memory->pinnedBytes;cost->gpuDevice=memory->deviceBytes;
 pattern[0]=sizeof(struct Sseg);pattern[1]=pattern[2]=(uint64_t)Plan.stride*sizeof(double);
 if(!AllocForecastBegin(MSX.QualPool,&forecast))return 0;
 /* Actual direct order is reserve the bounded pool, then pipe CPU parcels,
    then tank parcels. Every new object has the same Sseg/C/lastC pattern,
    so a single continuous forecast preserves that order and block cursor.
    Recycled incomplete objects are repaired in their actual LIFO order. */
 objects=initialPool+cost->initialCpu;
 for(cursor=MSX.FreeSeg;cursor&&available<objects;cursor=cursor->prev,++available){
  if(!cursor->privateC&&!AllocForecastAppend(&forecast,pattern+1,1,1))return 0;
  if(!cursor->privateLastC&&!AllocForecastAppend(&forecast,pattern+2,1,1))return 0;
 }
 if(!AllocForecastAppend(&forecast,pattern,3,objects-available))return 0;
 extra=forecast.bytes;
 {MSXResidentBudget cpu;MSXresidentBudget_snapshot(MSXresidentBudget_domain(MSX_BUDGET_CPU_POOL),&cpu);
  cost->cpuUsed=cpu.allocated[0]+cpu.reserved[0];
  cost->cpuFits=extra<=Plan.cpuQuota&&cost->cpuUsed<=Plan.cpuQuota-extra;}
 cost->poolBacking=extra;
 cost->baselinePageable=Plan.baseline.allocated[0]+Plan.baseline.reserved[0];
 cost->baselinePinned=Plan.baseline.allocated[1]+Plan.baseline.reserved[1];
 cost->baselineDevice=Plan.baseline.allocated[2]+Plan.baseline.reserved[2];
 cost->scratchHeadroom=MSXcapacity_headroom(Plan.scratchReserve,Plan.baselineScratch);
 cost->temporaryHeadroom=MSXcapacity_headroom(Plan.temporaryReserve,Plan.baselineTemporary);
 cost->uploadHeadroom=MSXcapacity_headroom(Plan.uploadReserve,Plan.baselineUpload);
 cost->coreHost=MSXresident_fixedHostBytes(Plan.links,(uint32_t)slots,Plan.stride);
 cost->hybridFull=MSXsegStorage_hybridFixedHostBytes(Plan.links,(uint32_t)slots,Plan.stride)-
        slots*(sizeof(struct Sseg)+(uint64_t)2*Plan.stride*sizeof(double));
 cost->hybridExisting=Plan.baselineHybrid;
 if(cost->hybridExisting>cost->hybridFull)return 0;
 /* The live Hybrid.pipe and rebalanceSnapshot remain charged in the actual
    frozen global baseline. Only still-unallocated Hybrid storage is future
    fixed cost; FILE's separate existing Core deduction is unchanged. */
 cost->hybridHost=cost->hybridFull-cost->hybridExisting;
 cost->runtimeHost=MSXresidentRuntime_fixedHostBytes(Plan.links,(uint32_t)slots,Plan.stride);
 memory->deviceBytes+=Plan.baseline.allocated[2]+Plan.baseline.reserved[2];
 memory->pinnedBytes+=Plan.baseline.allocated[1]+Plan.baseline.reserved[1];
 host=memory->hostBytes+memory->pinnedBytes+extra+
      Plan.baseline.allocated[0]+Plan.baseline.reserved[0]+cost->scratchHeadroom+cost->temporaryHeadroom+cost->uploadHeadroom+
      cost->coreHost+cost->hybridHost+cost->runtimeHost;
 cost->poolIndexSaving=(slots-initialPool)*(sizeof(Pseg)+sizeof(uint32_t)+sizeof(unsigned char));
 host-=cost->poolIndexSaving;
 memory->hostBytes=host-memory->pinnedBytes;
 cost->host=host;cost->valid=1;
 cost->deviceFits=memory->deviceBytes<=(Plan.admissionDeviceBudget?Plan.admissionDeviceBudget:Plan.deviceBudget);
 cost->hostFits=host<=(Plan.admissionHostBudget?Plan.admissionHostBudget:Plan.hostBudget);
 return cost->cpuFits&&cost->hostFits&&cost->deviceFits;
}
static int fits(const uint32_t *admission,const uint32_t *guards,const uint32_t *capacity,
                MSXResidentMemoryEstimate *memory)
{CapacityCost cost;return evaluateCost(admission,guards,capacity,&Plan.memoryConfig,memory,&cost);}
/* Host reason names identify contributing costs of one shared envelope, not
   independent quotas. CPU pool and device failures are independent checks. */
static void costReasons(const CapacityCost *c,char *out)
{
 out[0]=0;
 if(!c->valid){strcpy(out,"MODEL_ERROR");return;}
 if(!c->hostFits){strcat(out,"HOST_FIXED");
  if(c->poolBacking)strcat(out,"|HOST_POOL_RESERVE");
  if(c->uploadHeadroom)strcat(out,"|HOST_UPLOAD_RESERVE");
  if(c->scratchHeadroom||c->temporaryHeadroom)strcat(out,"|HOST_TEMPORARY");}
 if(!c->cpuFits){if(out[0])strcat(out,"|");strcat(out,"CPU_POOL_QUOTA");}
 if(!c->deviceFits){if(out[0])strcat(out,"|");strcat(out,"DEVICE");}
 if(!out[0])strcpy(out,"none");
}
static MSXResidentStatus writeCosts(void)
{
 FILE *f=fopen("resident_capacity_cost_model.csv","wb");uint32_t b,i,k,bump=1;int failed;
 const uint32_t batches[3]={32768,65536,131072};
 if(!f)return MSX_RESIDENT_ERR_PATH;
 fputs("probe,batch_rows,slots,initial_pool_objects,gpu_pageable_bytes,gpu_pinned_bytes,gpu_device_bytes,core_pageable_bytes,hybrid_metadata_pageable_bytes,runtime_pageable_bytes,pool_index_saving_bytes,additional_pool_backing_bytes,baseline_pageable_bytes,baseline_pinned_bytes,baseline_device_bytes,scratch_reserve_bytes,temporary_reserve_bytes,upload_reserve_bytes,cpu_pool_used_bytes,cpu_pool_quota_bytes,host_model_bytes,device_model_bytes,host_admission_budget_bytes,device_admission_budget_bytes,pageable_allocation_header_bytes,fits,host_envelope_fits,cpu_pool_quota_fits,device_fits,contributing_constraints,pinned_cap_status,initial_cpu_objects,initial_tank_objects,initial_equil_cache_bytes,initial_tank_draft_bytes,initial_scratch_in_baseline,candidate_link_index,configured_scratch_quota_bytes,baseline_scratch_bytes,configured_temporary_quota_bytes,baseline_temporary_bytes,configured_upload_quota_bytes,baseline_upload_bytes,hybrid_full_metadata_pageable_bytes,hybrid_existing_baseline_pageable_bytes\n",f);
 for(k=1;k<=Plan.links;++k)if(Plan.limit[k]<Plan.pipe[k].requested){bump=k;break;}
 for(b=0;b<4;++b)for(i=0;i<3;++i){MSXResidentMemoryConfig config=Plan.memoryConfig;
  MSXResidentMemoryEstimate memory;CapacityCost c;char reason[192];int ok;
  if(b<3)config.transferBatchRows=batches[b];
  for(k=1;k<=Plan.links;++k)Plan.trial[k]=i==2?Plan.pipe[k].requested:Plan.limit[k];
  if(i==1)++Plan.trial[bump];
  ok=evaluateCost(Plan.trial,Plan.guard,NULL,&config,&memory,&c);costReasons(&c,reason);
  fprintf(f,"%s,%u,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%d,%d,%d,%d,%s,not_configured,%llu,%llu,%llu,%llu,%s,%u,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n",
   i==0?"selected":i==1?"selected_plus_one":"requested",config.transferBatchRows,
   (unsigned long long)c.slots,(unsigned long long)c.initialPool,(unsigned long long)c.gpuHost,(unsigned long long)c.gpuPinned,(unsigned long long)c.gpuDevice,
   (unsigned long long)c.coreHost,(unsigned long long)c.hybridHost,(unsigned long long)c.runtimeHost,(unsigned long long)c.poolIndexSaving,
   (unsigned long long)c.poolBacking,(unsigned long long)c.baselinePageable,(unsigned long long)c.baselinePinned,(unsigned long long)c.baselineDevice,
   (unsigned long long)c.scratchHeadroom,(unsigned long long)c.temporaryHeadroom,(unsigned long long)c.uploadHeadroom,
   (unsigned long long)c.cpuUsed,(unsigned long long)Plan.cpuQuota,(unsigned long long)c.host,
   (unsigned long long)(c.gpuDevice+c.baselineDevice),(unsigned long long)(Plan.admissionHostBudget?Plan.admissionHostBudget:Plan.hostBudget),
   (unsigned long long)(Plan.admissionDeviceBudget?Plan.admissionDeviceBudget:Plan.deviceBudget),
   (unsigned long long)MSXresidentAlloc_headerBytes(),ok,c.hostFits,c.cpuFits,c.deviceFits,reason,
   (unsigned long long)c.initialCpu,(unsigned long long)c.initialTank,(unsigned long long)c.initialCacheBytes,
   (unsigned long long)c.initialTankDraftBytes,MSXresidentRuntime_directInitialPlanning()?"true":"false",i==1?bump:0,
   (unsigned long long)Plan.scratchReserve,(unsigned long long)Plan.baselineScratch,
   (unsigned long long)Plan.temporaryReserve,(unsigned long long)Plan.baselineTemporary,
   (unsigned long long)Plan.uploadReserve,(unsigned long long)Plan.baselineUpload,
   (unsigned long long)c.hybridFull,(unsigned long long)c.hybridExisting);
 }
 failed=ferror(f);if(fclose(f))failed=1;return failed?MSX_RESIDENT_ERR_PATH:MSX_RESIDENT_OK;
}
static uint64_t capped(uint32_t ceiling,int extras,uint32_t *admission)
{
 uint32_t k;uint64_t slots=0;
 for(k=1;k<=Plan.links;++k){MSXResidentCapacityPipe *p=&Plan.pipe[k];
  uint32_t n=MSXcapacity_cappedLimit(p->initial,p->guard,p->requested,ceiling,extras);
  slots+=n?n:1;admission[k]=n;
 }return slots;
}
static int ceilingFits(uint32_t ceiling,int extras,void *context)
{
 MSXResidentMemoryEstimate *memory=(MSXResidentMemoryEstimate*)context;
 capped(ceiling,extras,Plan.trial);return fits(Plan.trial,Plan.guard,NULL,memory);
}
static void freezeBaseline(void)
{
 MSXResidentBudget domain;
 MSXresidentBudget_snapshot(MSXresidentBudget_global(),&Plan.baseline);
 Plan.baselineHybrid=MSXsegStorage_hybridExistingHostBytes();
 MSXresidentBudget_snapshot(MSXresidentBudget_domain(MSX_BUDGET_SCRATCH),&domain);
 Plan.baselineScratch=domain.allocated[0]+domain.allocated[1]+domain.reserved[0]+domain.reserved[1];
 MSXresidentBudget_snapshot(MSXresidentBudget_domain(MSX_BUDGET_TEMPORARY),&domain);
 Plan.baselineTemporary=domain.allocated[0]+domain.allocated[1]+domain.reserved[0]+domain.reserved[1];
 MSXresidentBudget_snapshot(MSXresidentBudget_domain(MSX_BUDGET_UPLOAD),&domain);
 Plan.baselineUpload=domain.allocated[0]+domain.allocated[1]+domain.reserved[0]+domain.reserved[1];
}
static MSXResidentStatus configureBudget(void)
{
 if(MSXresidentBudget_configure(MSXresidentBudget_global(),Plan.hostBudget,Plan.deviceBudget)||
    MSXresidentBudget_configure(MSXresidentBudget_domain(MSX_BUDGET_CPU_POOL),Plan.cpuQuota,UINT64_MAX)||
    MSXresidentBudget_configure(MSXresidentBudget_domain(MSX_BUDGET_UPLOAD),Plan.uploadReserve,UINT64_MAX)||
    MSXresidentBudget_configure(MSXresidentBudget_domain(MSX_BUDGET_SCRATCH),Plan.scratchReserve,UINT64_MAX)||
    MSXresidentBudget_configure(MSXresidentBudget_domain(MSX_BUDGET_TEMPORARY),Plan.temporaryReserve,UINT64_MAX))
  return MSX_RESIDENT_ERR_MEMORY;
 return MSX_RESIDENT_OK;
}
static MSXResidentStatus sampleFreeObjects(void)
{
 Pseg p;Plan.freeObjects=0;
 for(p=MSX.FreeSeg;p;p=p->prev){if(++Plan.freeObjects>INT_MAX)return MSX_RESIDENT_ERR_OVERFLOW;}
 return MSX_RESIDENT_OK;
}
static MSXResidentStatus reserves(void)
{
 Plan.scratchReserve=MAX(UINT64_C(16777216),((uint64_t)Plan.links+1)*Plan.stride*sizeof(double)+
    ((uint64_t)MAX(MSX.Nobjects[NODE],MSX.Nobjects[LINK])+1)*(sizeof(REAL4)+2*sizeof(double))+
    Plan.stride*sizeof(double)+3*MSXresidentAlloc_headerBytes());
 Plan.temporaryReserve=UINT64_C(33554432);Plan.uploadReserve=Plan.resolved.values.uploadBytes?
    Plan.resolved.values.uploadBytes:UINT64_C(33554432);
 if(Plan.scratchReserve>Plan.hostBudget||Plan.temporaryReserve>Plan.hostBudget-Plan.scratchReserve||
    Plan.uploadReserve>Plan.hostBudget-Plan.scratchReserve-Plan.temporaryReserve)return MSX_RESIDENT_ERR_MEMORY;
 Plan.cpuQuota=Plan.resolved.values.cpuPoolBytes?Plan.resolved.values.cpuPoolBytes:
    Plan.hostBudget-Plan.scratchReserve-Plan.temporaryReserve-Plan.uploadReserve;
 if(Plan.cpuQuota>Plan.hostBudget-Plan.scratchReserve-Plan.temporaryReserve-Plan.uploadReserve)return MSX_RESIDENT_ERR_CONFIG;
 if(Plan.baselineScratch>Plan.scratchReserve||Plan.baselineTemporary>Plan.temporaryReserve||
    Plan.baselineUpload>Plan.uploadReserve)return MSX_RESIDENT_ERR_MEMORY;
 return MSX_RESIDENT_OK;
}
static MSXResidentStatus writeResolved(void)
{
 FILE *f=fopen("resolved_memory_config.json","wb");int failed;
 if(!f)return MSX_RESIDENT_ERR_PATH;
 fprintf(f,"{\n\"version\":1,\n\"host_budget_bytes\":%llu,\n\"device_budget_bytes\":%llu,\n"
  "\"cpu_pool_quota_bytes\":%llu,\n\"upload_reserve_bytes\":%llu,\n\"scratch_reserve_bytes\":%llu,\n"
  "\"temporary_reserve_bytes\":%llu,\n\"transfer_batch_rows\":%u,\n\"host_source\":\"%s\",\n"
  "\"cpu_source\":\"%s\",\n\"upload_source\":\"%s\",\n\"batch_source\":\"%s\",\n"
  "\"available_physical_at_sample_bytes\":%llu,\n\"available_commit_at_sample_bytes\":%llu,\n"
  "\"managed_host_at_sample_bytes\":%llu,\n\"sampling_point\":\"before_capacity_search\",\n"
  "\"scope\":\"MSX-managed heap/backing requests; OS, opaque CUDA/CRT, static globals and thread stacks separately reported\"\n}\n",
  (unsigned long long)Plan.hostBudget,(unsigned long long)Plan.deviceBudget,(unsigned long long)Plan.cpuQuota,
  (unsigned long long)Plan.uploadReserve,(unsigned long long)Plan.scratchReserve,(unsigned long long)Plan.temporaryReserve,
  Plan.memoryConfig.transferBatchRows,Plan.resolved.hostSource,Plan.resolved.cpuSource,Plan.resolved.uploadSource,Plan.resolved.batchSource,
  (unsigned long long)Plan.hostAvailable,(unsigned long long)Plan.hostAvailableCommit,(unsigned long long)Plan.sampleManagedHost);
 failed=ferror(f);if(fclose(f))failed=1;return failed?MSX_RESIDENT_ERR_PATH:MSX_RESIDENT_OK;
}
static MSXResidentStatus writeAttempt(uint32_t slots)
{
 FILE *f=fopen("resident_capacity_attempt.json","wb");int failed;
 if(!f)return MSX_RESIDENT_ERR_PATH;
 fprintf(f,"{\"capacity_mode\":\"%s\",\"startup_attempt\":%d,\"startup_retries\":%d,\"planned_gpu_slots\":%u,\"fixed_gpu_slots\":%u,\"retry_allowed\":%s,\"retry_reason\":\"%s\"}\n",
  MSX.GpuCoreCapacityMode?"AUTO":"FILE",Plan.retries,Plan.retries,slots,MSX.GpuCoreCapacityMode?0:slots,
  MSX.GpuCoreCapacityMode&&!Plan.retries?"true":"false",Plan.retryReason[0]?Plan.retryReason:"none");
 failed=ferror(f);if(fclose(f))failed=1;return failed?MSX_RESIDENT_ERR_PATH:MSX_RESIDENT_OK;
}
MSXResidentStatus MSXresidentCapacity_fileBudget(const MSXResidentLayout *layout)
{
 MEMORYSTATUSEX host;uint64_t freeDevice,managed,compatibility;MSXResidentStatus s;
 if(!layout)return MSX_RESIDENT_ERR_ARGUMENT;
 Plan.links=layout->nLinks;Plan.stride=layout->speciesStride;
 memset(&host,0,sizeof(host));host.dwLength=sizeof(host);
 if(!GlobalMemoryStatusEx(&host))return MSX_RESIDENT_ERR_MEMORY;
 Plan.hostAvailable=host.ullAvailPhys;Plan.hostAvailableCommit=host.ullAvailPageFile;
 if(MSXgpu_prepareResidentContext())return MSX_RESIDENT_ERR_GPU;
 MSXresidentBudget_snapshot(MSXresidentBudget_global(),&Plan.baseline);
 managed=Plan.baseline.allocated[0]+Plan.baseline.allocated[1]+Plan.baseline.reserved[0]+Plan.baseline.reserved[1];
 Plan.sampleManagedHost=managed;compatibility=host.ullAvailPhys/2;
 if(compatibility>UINT64_MAX-managed)return MSX_RESIDENT_ERR_OVERFLOW;
 if(!MSXmemory_resolve(&MSX.ResidentMemoryOptions,compatibility+managed,&Plan.resolved))return MSX_RESIDENT_ERR_CONFIG;
 Plan.hostBudget=Plan.resolved.values.hostBytes;
 s=MSXresidentGpu_availableMemory(&freeDevice);if(s)return s;
 Plan.deviceBudget=((freeDevice+Plan.baseline.allocated[2]+Plan.baseline.reserved[2])/5)*4;
 if(MSX.GpuCoreMemoryMB>0){double bytes=MSX.GpuCoreMemoryMB*1048576.0;
  if(!isfinite(bytes)||bytes>=(double)UINT64_MAX)return MSX_RESIDENT_ERR_CONFIG;
  if(bytes<(double)Plan.deviceBudget)Plan.deviceBudget=(uint64_t)bytes;}
 if(MSXresidentBudget_configure(MSXresidentBudget_global(),Plan.hostBudget,Plan.deviceBudget))return MSX_RESIDENT_ERR_MEMORY;
 {int error=MSXgpu_openResidentPrograms();if(error)return error==ERR_MEMORY||error==ERR_GPU_MEMORY_ALLOCATION_FAILED?
    MSX_RESIDENT_ERR_MEMORY:MSX_RESIDENT_ERR_GPU;}
 freezeBaseline();
 /* The FILE parser has already allocated Core; subtract that heap exactly
    once, as fits includes it among the upcoming fixed-layout allocations. */
 {uint64_t core=MSXresident_fixedHostBytes(Plan.links,layout->totalSlots,Plan.stride);
  if(Plan.baseline.allocated[0]<core)return MSX_RESIDENT_ERR_CONFIG;Plan.baseline.allocated[0]-=core;}
 Plan.memoryConfig.version=1;
 Plan.memoryConfig.hydLayout=MSX_RESIDENT_HYD_PIPE_MAJOR;Plan.memoryConfig.transferBatchRows=Plan.resolved.values.batchRows;
 s=reserves();if(s)return s;
 s=sampleFreeObjects();if(s)return s;
 /* Usage arrays are materialized after initialization for FILE mode. */
 Plan.baseline.allocated[0]+=((uint64_t)Plan.links+1)*(sizeof(Usage)+2*sizeof(uint32_t))+3*MSXresidentAlloc_headerBytes();
 if(!fits(layout->admissionLimit,layout->guard,layout->capacity,&Plan.memory))return MSX_RESIDENT_ERR_MEMORY;
 s=configureBudget();if(s)return s;s=writeResolved();if(s)return s;return writeAttempt(layout->totalSlots);
}
MSXResidentStatus MSXresidentCapacity_openPlan(const char *caseHash)
{
 uint64_t freeDevice,slots=0;uint32_t k,lo=0,hi=0,initialUpper=0;int extras;
 MSXResidentMemoryEstimate memory;MSXResidentStatus status;double start=MSXgpu_wallTimeMs();
 if(!Plan.pipe||!caseHash)return MSX_RESIDENT_ERR_CONFIG;
 if(!Plan.retries){
  uint64_t compatibility=Plan.hostAvailable/2;
  if(compatibility>UINT64_MAX-Plan.sampleManagedHost)return MSX_RESIDENT_ERR_OVERFLOW;
  if(!MSXmemory_resolve(&MSX.ResidentMemoryOptions,compatibility+Plan.sampleManagedHost,&Plan.resolved))
   return MSX_RESIDENT_ERR_CONFIG;
  Plan.memoryConfig.version=1;Plan.memoryConfig.hydLayout=MSX_RESIDENT_HYD_PIPE_MAJOR;
  Plan.memoryConfig.transferBatchRows=Plan.resolved.values.batchRows;
 }
 if(MSXgpu_prepareResidentContext())return MSX_RESIDENT_ERR_GPU;
 /* Freeze and enforce the device envelope before program device allocations. */
 status=MSXresidentGpu_availableMemory(&freeDevice);if(status)return status;
 if(!Plan.retries){MSXResidentBudget snapshot;MSXresidentBudget_snapshot(MSXresidentBudget_global(),&snapshot);
  Plan.deviceBudget=((freeDevice+snapshot.allocated[2]+snapshot.reserved[2])/5)*4;
  if(MSX.GpuCoreMemoryMB>0){double bytes=MSX.GpuCoreMemoryMB*1048576.0;
   if(!isfinite(bytes)||bytes>=(double)UINT64_MAX)return MSX_RESIDENT_ERR_CONFIG;
   if(bytes<(double)Plan.deviceBudget)Plan.deviceBudget=(uint64_t)bytes;}}
 if(MSXresidentBudget_configure(MSXresidentBudget_global(),Plan.resolved.values.hostBytes,Plan.deviceBudget))
  return MSX_RESIDENT_ERR_MEMORY;
 {int error=MSXgpu_openResidentPrograms();
  if(error)return error==ERR_MEMORY||error==ERR_GPU_MEMORY_ALLOCATION_FAILED?MSX_RESIDENT_ERR_MEMORY:MSX_RESIDENT_ERR_GPU;}
 if(MSXgpu_getResidentProgramMemory(&Plan.programMemory))return MSX_RESIDENT_ERR_GPU;
 start=MSXgpu_wallTimeMs(); /* Context startup is separate from prediction. */
 freezeBaseline();
 if(!Plan.retries){
  Plan.hostBudget=Plan.resolved.values.hostBytes;
  status=reserves();if(status)return status;
 }
 status=sampleFreeObjects();if(status)return status;
 for(k=1;k<=Plan.links;++k){Plan.trial[k]=Plan.pipe[k].requested;
  slots+=Plan.trial[k]?Plan.trial[k]:1;hi=MAX(hi,Plan.trial[k]);
  initialUpper=MAX(initialUpper,MSXcapacity_cappedLimit(Plan.pipe[k].initial,Plan.guard[k],Plan.trial[k],0,1));}
 if(fits(Plan.trial,Plan.guard,NULL,&memory))memcpy(Plan.limit,Plan.trial,((size_t)Plan.links+1)*sizeof(*Plan.limit));
 else {
  extras=ceilingFits(0,1,&memory);
  /* Only the constant-initial-CPU extras region has a monotone upper
     boundary. Below it, enumerate every <=97 candidate; a CPU quota failure
     at ceiling 0 does not imply no feasible larger initial admission. */
  if(!MSXcapacity_searchCeiling(extras?hi:initialUpper,extras,ceilingFits,&memory,&lo))
   return MSX_RESIDENT_ERR_MEMORY;
  slots=capped(lo,extras,Plan.limit);
  memcpy(Plan.trial,Plan.limit,((size_t)Plan.links+1)*sizeof(*Plan.trial));
  for(k=1;k<=Plan.links;++k){uint32_t target=extras?Plan.pipe[k].requested:
     MIN(Plan.pipe[k].requested,Plan.pipe[k].initial>2*Plan.pipe[k].guard?Plan.pipe[k].initial-2*Plan.pipe[k].guard:0);
   if(Plan.limit[k]<target){uint64_t more=Plan.limit[k]?1:0;
    ++Plan.trial[k];
    if(fits(Plan.trial,Plan.guard,NULL,&memory)){++Plan.limit[k];slots+=more;}
    else --Plan.trial[k];}
  }
 }
 if(!fits(Plan.limit,Plan.guard,NULL,&Plan.memory))return MSX_RESIDENT_ERR_MEMORY;
 status=configureBudget();if(status)return status;
 status=writeResolved();if(status)return status;
 Plan.slots=(uint32_t)slots;for(k=1;k<=Plan.links;++k)Plan.pipe[k].admission=Plan.limit[k];
 status=writeAttempt(Plan.slots);if(status)return status;
 Plan.budgetMs+=MSXgpu_wallTimeMs()-start;
 status=MSXresident_openPlan(Plan.links,Plan.limit,Plan.guard,caseHash);
 if(status==MSX_RESIDENT_OK){
  FILE *f;char id[EN_MAXID+1],requestedReason[192],nextReason[192],nextReasons[4][192];
  unsigned char nextSeen[4]={0};CapacityCost cost;
  status=writeCosts();if(status)return status;
  for(k=1;k<=Plan.links;++k)Plan.trial[k]=Plan.pipe[k].requested;
  evaluateCost(Plan.trial,Plan.guard,NULL,&Plan.memoryConfig,&memory,&cost);costReasons(&cost,requestedReason);
  memcpy(Plan.trial,Plan.limit,((size_t)Plan.links+1)*sizeof(*Plan.trial));
  f=fopen("resident_capacity_prediction.csv","wb");
  if(!f)return MSX_RESIDENT_ERR_PATH;
  fputs("link_index,link_id,volume,N0,guard,Qmin,Qmax,vmin,K_positive,zero_ms,reversals,U,predicted,requested,admissionLimit,capacity,clipped,initial_core_required,initial_core_admitted,initial_shortfall,required,admitted,shortfall,physical_capacity,requested_plan_constraints,next_slot_constraints,pinned_cap_status,constraint_semantics\n",f);
  for(k=1;k<=Plan.links;++k){MSXResidentCapacityPipe *p=&Plan.pipe[k];uint32_t initial=p->initial>2*p->guard?p->initial-2*p->guard:0;ENgetlinkid((int)k,id);
   strcpy(nextReason,"none");
   if(p->admission<p->requested){unsigned group=(p->admission?2:0)+(p->admission<initial?1:0);
    /* An exact +1 vector changes only physical slots (0/1) and initial CPU
       objects (0/-1). The model has no other per-link allocation term; all
       members of one group therefore have exactly the same backing/cost.
       Evaluate a representative full vector, avoiding a quadratic report. */
    if(!nextSeen[group]){++Plan.trial[k];
     evaluateCost(Plan.trial,Plan.guard,NULL,&Plan.memoryConfig,&memory,&cost);
     costReasons(&cost,nextReasons[group]);--Plan.trial[k];nextSeen[group]=1;}
    strcpy(nextReason,nextReasons[group]);}
   fprintf(f,"%u,%s,%.17g,%u,%u,%.17g,%.17g,%.17g,%llu,%llu,%llu,%u,%u,%u,%u,%u,%s,%u,%u,%u,%u,%u,%u,%u,%s,%s,not_configured,host_terms_contribute_to_shared_envelope_cpu_and_device_independent\n",k,id,p->volume,p->initial,p->guard,p->qmin,p->qmax,p->vmin,(unsigned long long)p->positiveSteps,(unsigned long long)p->zeroMs,(unsigned long long)p->reversals,p->upper,p->predicted,p->requested,p->admission,p->admission?p->admission:1,p->admission<p->requested?"budget":"none",initial,MIN(initial,p->admission),initial-MIN(initial,p->admission),initial,MIN(initial,p->admission),initial-MIN(initial,p->admission),p->admission?p->admission:1,p->admission<p->requested?requestedReason:"none",nextReason);
  }if(ferror(f)){fclose(f);return MSX_RESIDENT_ERR_PATH;}fclose(f);
  f=fopen("resident_capacity_summary.json","wb");if(!f)return MSX_RESIDENT_ERR_PATH;
  fprintf(f,"{\n \"prediction_version\":\"hyd_minflow_v1\",\n \"inp_sha256\":\"%s\",\n \"msx_sha256\":\"%s\",\n \"hyd_sha256\":\"%s\",\n \"runtime_hash_reference\":\"runtime_manifest.csv\",\n \"Q_STAGNANT_cfs\":%.17g,\n \"quality_step_ms\":%lld,\n \"duration_ms\":%lld,\n \"device_budget_bytes\":%llu,\n \"host_budget_bytes\":%llu,\n \"device_array_bytes\":%llu,\n \"pinned_array_bytes\":%llu,\n \"host_array_bytes_including_initial_parcels\":%llu,\n \"hyd_statistics_ms\":%.9g,\n \"prediction_budget_ms\":%.9g,\n \"gpu_slots\":%u,\n \"gpu_capacity_growth_count\":0,\n \"startup_retries\":%d,\n \"retry_reason\":\"%s\",\n \"ownership_policy\":\"contiguous_core_cpu_overflow\"\n}\n",Plan.inpHash,Plan.msxHash,Plan.hydHash,MSX_Q_STAGNANT,(long long)MSX.Qstep,(long long)MSX.Dur,(unsigned long long)Plan.deviceBudget,(unsigned long long)Plan.hostBudget,(unsigned long long)Plan.memory.deviceBytes,(unsigned long long)Plan.memory.pinnedBytes,(unsigned long long)Plan.memory.hostBytes,Plan.scanMs,Plan.budgetMs,Plan.slots,Plan.retries,Plan.retryReason[0]?Plan.retryReason:"none");
  if(ferror(f)){fclose(f);return MSX_RESIDENT_ERR_PATH;}fclose(f);
 }return status;
}
int MSXresidentCapacity_retryBudget(void)
{
 if(!Plan.pipe||Plan.retries)return 0;
 ++Plan.retries;Plan.admissionDeviceBudget=(Plan.deviceBudget/5)*4;
 Plan.admissionHostBudget=(Plan.hostBudget/5)*4;return 1;
}
void MSXresidentCapacity_noteStartupRetry(const char *reason)
{
 size_t n=0;if(!reason||!*reason)reason="initial_memory_allocation";
 /* Store a bounded diagnostic identifier that remains valid JSON. */
 for(;reason[n]&&n+1<sizeof(Plan.retryReason);++n){char c=reason[n];
  Plan.retryReason[n]=(c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='_'||c=='-'?c:'_';}
 Plan.retryReason[n]=0;
}
int MSXresidentCapacity_startupAttempt(void)
{ return Plan.pipe ? Plan.retries : -1; }
static int compareParcel(const void *a,const void *b)
{uint64_t x=((const AuditParcel*)a)->id,y=((const AuditParcel*)b)->id;return x<y?-1:x>y?1:0;}
int MSXresidentCapacity_auditReact(uint32_t link,uint64_t id,int owner)
{
 uint32_t lo=0,hi;AuditPipe *p;
 if(!MSXResidentCapacityAuditEnabled)return 0;
 if(!Plan.audit||!link||link>Plan.links)return ERR_GPU_SEGMENT_PACK_FAILED;
 p=&Plan.audit[link];hi=p->count;
 while(lo<hi){uint32_t m=lo+(hi-lo)/2;if(p->parcel[m].id<id)lo=m+1;else hi=m;}
 if(lo==p->count||p->parcel[lo].id!=id||p->parcel[lo].owner!=owner||p->parcel[lo].seen)
  return ERR_GPU_SEGMENT_PACK_FAILED;
 p->parcel[lo].seen=1;return 0;
}
int MSXresidentCapacity_auditFinish(void)
{
 uint32_t k,i;if(!MSXResidentCapacityAuditEnabled)return 0;
 for(k=1;k<=Plan.links;++k)for(i=0;i<Plan.audit[k].count;++i)
  if(!Plan.audit[k].parcel[i].seen)return ERR_GPU_SEGMENT_PACK_FAILED;
 ++Plan.auditSteps;return 0;
}
void MSXresidentCapacity_recordOwnership(void)
{
 uint32_t k;uint64_t base=0;MSXResidentLayout layout;
 if(!MSX.GpuCoreOverflow||!MSXresidentRuntime_isResident())return;
 if(MSXresident_getLayout(&layout)!=MSX_RESIDENT_OK)return;
 if(!Plan.usage){Plan.links=layout.nLinks;Plan.usage=(Usage*)calloc((size_t)Plan.links+1,sizeof(*Plan.usage));
  if(!Plan.usage){MSX.ErrCode=ERR_MEMORY;return;}
  for(k=1;k<=Plan.links;++k)Plan.usage[k].firstSpill=-1;
  Plan.limit=(uint32_t*)calloc((size_t)Plan.links+1,sizeof(*Plan.limit));
  Plan.guard=(uint32_t*)calloc((size_t)Plan.links+1,sizeof(*Plan.guard));
  if(!Plan.limit||!Plan.guard){MSX.ErrCode=ERR_MEMORY;return;}
  Plan.slots=layout.totalSlots;
  memcpy(Plan.limit,layout.admissionLimit,((size_t)Plan.links+1)*sizeof(*Plan.limit));
  memcpy(Plan.guard,layout.guard,((size_t)Plan.links+1)*sizeof(*Plan.guard));
 }
 if(layout.totalSlots!=Plan.slots){MSX.ErrCode=ERR_GPU_SEGMENT_PACK_FAILED;return;}
 {const char *audit=getenv("MSX_RESIDENT_AUDIT_CPU_SPILL");
  MSXResidentCapacityAuditEnabled=audit&&!strcmp(audit,"1");}
 if(MSXResidentCapacityAuditEnabled&&!Plan.audit){Plan.audit=(AuditPipe*)calloc((size_t)Plan.links+1,sizeof(*Plan.audit));
  if(!Plan.audit){MSX.ErrCode=ERR_MEMORY;return;}}
 for(k=1;k<=Plan.links;++k){Usage *u=&Plan.usage[k];uint32_t n=(uint32_t)MSX.Link[k].nsegs;
  uint32_t core=(uint32_t)MSXsegStorage_hybridCoreCount((int)k),cpu=n-core,guard=layout.guard[k];
  uint32_t spill=cpu>2*guard?cpu-2*guard:0,limit=layout.admissionLimit[k];
  uint32_t demand=n>2*guard+limit?n-2*guard-limit:0;
  if(limit!=Plan.limit[k]||guard!=Plan.guard[k]||layout.capacity[k]!=(limit?limit:1)||layout.base[k]!=base)
   {MSX.ErrCode=ERR_GPU_SEGMENT_PACK_FAILED;return;}
  base+=layout.capacity[k];
  if(core>limit||core>n){MSX.ErrCode=ERR_GPU_SEGMENT_PACK_FAILED;return;}
  u->corePeak=MAX(u->corePeak,core);u->cpuPeak=MAX(u->cpuPeak,cpu);
  u->spillPeak=MAX(u->spillPeak,spill);u->demandPeak=MAX(u->demandPeak,demand);
  u->cpuSteps+=cpu;u->gpuSteps+=core;if(spill){++u->spillSteps;if(u->firstSpill<0)u->firstSpill=MSX.Qtime;}
  u->spillRows+=spill;u->demandRows+=demand;
  if(MSXResidentCapacityAuditEnabled){Pseg seg;AuditPipe *p=&Plan.audit[k];uint32_t pos=0;
   free(p->parcel);p->count=0;p->parcel=(AuditParcel*)calloc(n?n:1,sizeof(*p->parcel));
   if(!p->parcel){MSX.ErrCode=ERR_MEMORY;return;}
   for(seg=MSX.FirstSeg[k];seg;seg=seg->prev){
    if(pos>=n||!seg->hybridId){MSX.ErrCode=ERR_GPU_SEGMENT_PACK_FAILED;return;}
    p->parcel[pos].id=seg->hybridId;p->parcel[pos].owner=seg->inHybridCore?1:0;
    /* Chemistry deliberately skips zero-length links in the existing path. */
    p->parcel[pos].seen=MSX.Link[k].len==0.0;++pos;
   }
   if(pos!=n){MSX.ErrCode=ERR_GPU_SEGMENT_PACK_FAILED;return;}
   p->count=n;qsort(p->parcel,n,sizeof(*p->parcel),compareParcel);
   for(pos=1;pos<n;++pos)if(p->parcel[pos].id==p->parcel[pos-1].id){MSX.ErrCode=ERR_GPU_SEGMENT_PACK_FAILED;return;}
   Plan.auditParcels+=n;
  }
 }
}
void MSXresidentCapacity_addCpuReactMs(uint32_t link,double ms)
{if(Plan.usage&&link&&link<=Plan.links&&ms>=0)Plan.usage[link].cpuReactMs+=ms;}
void MSXresidentCapacity_setDeviceBaseline(uint64_t bytes)
{Plan.initialFreeDevice=bytes;Plan.maxDeviceFreeDelta=Plan.deviceSamples=0;}
void MSXresidentCapacity_recordDeviceMemory(void)
{
 uint64_t current;
 if(Plan.initialFreeDevice&&MSXresidentGpu_availableMemory(&current)==MSX_RESIDENT_OK){
  uint64_t delta=current<Plan.initialFreeDevice?Plan.initialFreeDevice-current:0;
  if(delta>Plan.maxDeviceFreeDelta)Plan.maxDeviceFreeDelta=delta;++Plan.deviceSamples;
 }
}
void MSXresidentCapacity_writeTransferStats(const MSXResidentGpuTransferStats *stats)
{
 FILE *f;if(!stats)return;f=fopen("resident_transfer_summary.json","wb");if(!f)return;
 fprintf(f,"{\"handoff_h2d_bytes\":%llu,\"handoff_d2h_bytes\":%llu,\"handoff_h2d_calls\":%llu,\"handoff_d2h_calls\":%llu,\"hyd_h2d_bytes\":%llu,\"hyd_h2d_calls\":%llu,\"patch_reuse_waits\":%llu,\"patch_reuse_wait_ms\":%.17g,\"patch_reuse_wait_timing_collected\":%s,\"max_global_device_free_delta_after_react_bytes\":%llu,\"device_memory_samples\":%llu}\n",(unsigned long long)stats->h2dBytes,(unsigned long long)stats->d2hBytes,(unsigned long long)stats->h2dCalls,(unsigned long long)stats->d2hCalls,(unsigned long long)stats->hydH2DBytes,(unsigned long long)stats->hydH2DCalls,(unsigned long long)stats->patchReuseWaits,stats->patchReuseWaitMs,MSXgpu_profileDetailEnabled()?"true":"false",(unsigned long long)Plan.maxDeviceFreeDelta,(unsigned long long)Plan.deviceSamples);fclose(f);
}
void MSXresidentCapacity_writeProcessMetrics(void)
{
 PROCESS_MEMORY_COUNTERS_EX memory;FILE *f;
 memset(&memory,0,sizeof(memory));memory.cb=sizeof(memory);
 if(!GetProcessMemoryInfo(GetCurrentProcess(),(PROCESS_MEMORY_COUNTERS*)&memory,sizeof(memory)))return;
 f=fopen("resident_process_memory.json","wb");if(!f)return;
 fprintf(f,"{\"peak_working_set_bytes\":%llu,\"peak_pagefile_bytes\":%llu,\"private_usage_at_cleanup_bytes\":%llu}\n",(unsigned long long)memory.PeakWorkingSetSize,(unsigned long long)memory.PeakPagefileUsage,(unsigned long long)memory.PrivateUsage);fclose(f);
}
void MSXresidentCapacity_writeUsage(void)
{
 FILE *f;uint32_t k;MSXResidentLayout layout;
 if(!Plan.usage||MSXresident_getLayout(&layout)!=MSX_RESIDENT_OK)return;
 f=fopen("resident_capacity_usage.csv","wb");if(!f)return;
 fputs("link_index,admissionLimit,capacity,core_peak,cpu_peak,cpu_extra_peak,capacity_demand_peak,first_spill_ms,spill_steps,cpu_row_steps,gpu_row_steps,cpu_spill_row_steps,capacity_demand_row_steps,cpu_react_ms,final_core,cpu_pool_growth_count,gpu_capacity_growth_count\n",f);
 for(k=1;k<=Plan.links;++k){Usage *u=&Plan.usage[k];fprintf(f,"%u,%u,%u,%u,%u,%u,%u,%lld,%llu,%llu,%llu,%llu,%llu,%.9g,%d,%llu,0\n",k,layout.admissionLimit[k],layout.capacity[k],u->corePeak,u->cpuPeak,u->spillPeak,u->demandPeak,(long long)u->firstSpill,(unsigned long long)u->spillSteps,(unsigned long long)u->cpuSteps,(unsigned long long)u->gpuSteps,(unsigned long long)u->spillRows,(unsigned long long)u->demandRows,u->cpuReactMs,MSXsegStorage_hybridCoreCount((int)k),(unsigned long long)MSXsegStorage_hybridCpuPoolGrowthCount());}
 fclose(f);
 if(MSXResidentCapacityAuditEnabled){f=fopen("resident_ownership_audit.json","wb");if(f){fprintf(f,"{\"steps\":%llu,\"hyd_grid_expected_steps\":%llu,\"parcel_row_steps\":%llu,\"status\":\"%s\"}\n",(unsigned long long)Plan.auditSteps,(unsigned long long)Plan.expectedSteps,(unsigned long long)Plan.auditParcels,MSX.ErrCode||(Plan.pipe&&Plan.auditSteps!=Plan.expectedSteps)?"FAIL":"PASS");fclose(f);}}
}
