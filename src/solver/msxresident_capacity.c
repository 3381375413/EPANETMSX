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
#include "msxresident_runtime.h"
#include "msxresident_hash.h"
#include "msxsegment_storage.h"
#include "msxqual_shared.h"
#include "epanet2.h"
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
 uint32_t *limit,*guard;
 Usage *usage;
 uint64_t hostAvailable,deviceBudget,hostBudget,initialBytes;
 MSXResidentMemoryEstimate memory;
 MSXResidentMemoryEstimate programMemory;
 char hydHash[65],inpHash[65],msxHash[65];
 double scanMs,budgetMs;
 int retries;
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
 free(Plan.pipe);free(Plan.limit);free(Plan.guard);free(Plan.usage);memset(&Plan,0,sizeof(Plan));MSXResidentCapacityAuditEnabled=0;}
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
 Plan.usage=(Usage*)calloc((size_t)Plan.links+1,sizeof(*Plan.usage));
 q=(REAL4*)calloc((size_t)Plan.links+1,sizeof(*q));
 if(!Plan.pipe||!Plan.limit||!Plan.guard||!Plan.usage||!q)goto memory;
 memset(&host,0,sizeof(host));host.dwLength=sizeof(host);
 if(!GlobalMemoryStatusEx(&host))goto memory;Plan.hostAvailable=host.ullAvailPhys;
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
static int fits(uint64_t slots,MSXResidentMemoryEstimate *memory)
{
 uint64_t host;
 if(slots>UINT32_MAX||MSXresidentGpu_estimateMemory(Plan.links,(uint32_t)slots,Plan.stride,memory)!=MSX_RESIDENT_OK)return 0;
 memory->deviceBytes+=Plan.programMemory.deviceBytes;
 memory->pinnedBytes+=Plan.programMemory.pinnedBytes;
 memory->hostBytes+=Plan.programMemory.hostBytes;
 host=memory->hostBytes+memory->pinnedBytes+Plan.initialBytes+
      MSXresident_fixedHostBytes(Plan.links,(uint32_t)slots,Plan.stride)+
      MSXsegStorage_hybridFixedHostBytes(Plan.links,(uint32_t)slots,Plan.stride)+
      MSXresidentRuntime_fixedHostBytes(Plan.links,(uint32_t)slots,Plan.stride);
 memory->hostBytes=host-memory->pinnedBytes;
 return memory->deviceBytes<=Plan.deviceBudget&&host<=Plan.hostBudget;
}
static uint64_t capped(uint32_t ceiling,int extras,int publish)
{
 uint32_t k;uint64_t slots=0;
 for(k=1;k<=Plan.links;++k){MSXResidentCapacityPipe *p=&Plan.pipe[k];
  uint32_t initial=p->initial>2*p->guard?p->initial-2*p->guard:0;
  uint32_t base=MIN(p->requested,initial),n=extras?base+MIN(p->requested-base,ceiling):MIN(base,ceiling);
  slots+=n?n:1;if(publish)Plan.limit[k]=n;
 }return slots;
}
MSXResidentStatus MSXresidentCapacity_openPlan(const char *caseHash)
{
 uint64_t freeDevice,slots=0;uint32_t k,lo=0,hi=0;int extras;
 MSXResidentMemoryEstimate memory;MSXResidentStatus status;double start=MSXgpu_wallTimeMs();
 if(!Plan.pipe||!caseHash)return MSX_RESIDENT_ERR_CONFIG;
 if(MSXgpu_prepareResidentContext())return MSX_RESIDENT_ERR_GPU;
 {int error=MSXgpu_openResidentPrograms();
  if(error)return error==ERR_MEMORY||error==ERR_GPU_MEMORY_ALLOCATION_FAILED?MSX_RESIDENT_ERR_MEMORY:MSX_RESIDENT_ERR_GPU;}
 if(MSXgpu_getResidentProgramMemory(&Plan.programMemory))return MSX_RESIDENT_ERR_GPU;
 status=MSXresidentGpu_availableMemory(&freeDevice);if(status)return status;
 start=MSXgpu_wallTimeMs(); /* Context startup is separate from prediction. */
 if(!Plan.retries){Plan.deviceBudget=((freeDevice+Plan.programMemory.deviceBytes)/5)*4;Plan.hostBudget=Plan.hostAvailable/2;}
 if(!Plan.retries&&MSX.GpuCoreMemoryMB>0){double bytes=MSX.GpuCoreMemoryMB*1048576.0;
  if(bytes<(double)Plan.deviceBudget)Plan.deviceBudget=(uint64_t)bytes;}
 for(k=1;k<=Plan.links;++k){slots+=Plan.pipe[k].requested?Plan.pipe[k].requested:1;hi=MAX(hi,Plan.pipe[k].requested);}
 if(fits(slots,&memory))for(k=1;k<=Plan.links;++k)Plan.limit[k]=Plan.pipe[k].requested;
 else {
  extras=fits(capped(0,1,0),&memory);
  if(!fits(Plan.links,&memory))return MSX_RESIDENT_ERR_MEMORY;
  while(lo<hi){uint32_t middle=lo+(hi-lo+1)/2;
   if(fits(capped(middle,extras,0),&memory))lo=middle;else hi=middle-1;}
  slots=capped(lo,extras,1);
  for(k=1;k<=Plan.links;++k){uint32_t target=extras?Plan.pipe[k].requested:
     MIN(Plan.pipe[k].requested,Plan.pipe[k].initial>2*Plan.pipe[k].guard?Plan.pipe[k].initial-2*Plan.pipe[k].guard:0);
   if(Plan.limit[k]<target){uint64_t more=Plan.limit[k]?1:0;
    if(fits(slots+more,&memory)){++Plan.limit[k];slots+=more;}}
  }
 }
 if(!fits(slots,&Plan.memory))return MSX_RESIDENT_ERR_MEMORY;
 Plan.slots=(uint32_t)slots;for(k=1;k<=Plan.links;++k)Plan.pipe[k].admission=Plan.limit[k];
 Plan.budgetMs+=MSXgpu_wallTimeMs()-start;
 status=MSXresident_openPlan(Plan.links,Plan.limit,Plan.guard,caseHash);
 if(status==MSX_RESIDENT_OK){
  FILE *f=fopen("resident_capacity_prediction.csv","wb");char id[EN_MAXID+1];
  if(!f)return MSX_RESIDENT_ERR_PATH;
  fputs("link_index,link_id,volume,N0,guard,Qmin,Qmax,vmin,K_positive,zero_ms,reversals,U,predicted,requested,admissionLimit,capacity,clipped\n",f);
  for(k=1;k<=Plan.links;++k){MSXResidentCapacityPipe *p=&Plan.pipe[k];ENgetlinkid((int)k,id);
   fprintf(f,"%u,%s,%.17g,%u,%u,%.17g,%.17g,%.17g,%llu,%llu,%llu,%u,%u,%u,%u,%u,%s\n",k,id,p->volume,p->initial,p->guard,p->qmin,p->qmax,p->vmin,(unsigned long long)p->positiveSteps,(unsigned long long)p->zeroMs,(unsigned long long)p->reversals,p->upper,p->predicted,p->requested,p->admission,p->admission?p->admission:1,p->admission<p->requested?"budget":"none");
  }if(ferror(f)){fclose(f);return MSX_RESIDENT_ERR_PATH;}fclose(f);
  f=fopen("resident_capacity_summary.json","wb");if(!f)return MSX_RESIDENT_ERR_PATH;
  fprintf(f,"{\n \"prediction_version\":\"hyd_minflow_v1\",\n \"inp_sha256\":\"%s\",\n \"msx_sha256\":\"%s\",\n \"hyd_sha256\":\"%s\",\n \"runtime_hash_reference\":\"runtime_manifest.csv\",\n \"Q_STAGNANT_cfs\":%.17g,\n \"quality_step_ms\":%lld,\n \"duration_ms\":%lld,\n \"device_budget_bytes\":%llu,\n \"host_budget_bytes\":%llu,\n \"device_array_bytes\":%llu,\n \"pinned_array_bytes\":%llu,\n \"host_array_bytes_including_initial_parcels\":%llu,\n \"hyd_statistics_ms\":%.9g,\n \"prediction_budget_ms\":%.9g,\n \"gpu_slots\":%u,\n \"gpu_capacity_growth_count\":0,\n \"startup_retries\":%d,\n \"ownership_policy\":\"contiguous_core_cpu_overflow\"\n}\n",Plan.inpHash,Plan.msxHash,Plan.hydHash,MSX_Q_STAGNANT,(long long)MSX.Qstep,(long long)MSX.Dur,(unsigned long long)Plan.deviceBudget,(unsigned long long)Plan.hostBudget,(unsigned long long)Plan.memory.deviceBytes,(unsigned long long)Plan.memory.pinnedBytes,(unsigned long long)Plan.memory.hostBytes,Plan.scanMs,Plan.budgetMs,Plan.slots,Plan.retries);
  if(ferror(f)){fclose(f);return MSX_RESIDENT_ERR_PATH;}fclose(f);
 }return status;
}
int MSXresidentCapacity_retryBudget(void)
{
 if(!Plan.pipe||Plan.retries)return 0;
 ++Plan.retries;Plan.deviceBudget=(Plan.deviceBudget/5)*4;
 Plan.hostBudget=(Plan.hostBudget/5)*4;return 1;
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
 fprintf(f,"{\"handoff_h2d_bytes\":%llu,\"handoff_d2h_bytes\":%llu,\"handoff_h2d_calls\":%llu,\"handoff_d2h_calls\":%llu,\"hyd_h2d_bytes\":%llu,\"hyd_h2d_calls\":%llu,\"max_global_device_free_delta_after_react_bytes\":%llu,\"device_memory_samples\":%llu}\n",(unsigned long long)stats->h2dBytes,(unsigned long long)stats->d2hBytes,(unsigned long long)stats->h2dCalls,(unsigned long long)stats->d2hCalls,(unsigned long long)stats->hydH2DBytes,(unsigned long long)stats->hydH2DCalls,(unsigned long long)Plan.maxDeviceFreeDelta,(unsigned long long)Plan.deviceSamples);fclose(f);
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
