#include "msxresident_core_cuda.h"
#include <string.h>
/* Fail closed: a non-CUDA build must never claim a resident GPU mirror. */
int MSXresidentGpu_isEnabled(void){return 0;}
MSXResidentStatus MSXresidentGpu_open(const MSXResidentGpuOpen*a,MSXResidentGpu**b){(void)a;if(b)*b=0;return MSX_RESIDENT_DISABLED;}
MSXResidentStatus MSXresidentGpu_initialUpload(MSXResidentGpu*a,const MSXResidentPatchBatch*b){(void)a;(void)b;return MSX_RESIDENT_DISABLED;}
MSXResidentStatus MSXresidentGpu_applyPatches(MSXResidentGpu*a,const MSXResidentPatchBatch*b){(void)a;(void)b;return MSX_RESIDENT_DISABLED;}
MSXResidentStatus MSXresidentGpu_fetchHandoffs(MSXResidentGpu*a,const MSXResidentHandoffPlan*b,MSXResidentGpuFetchOutput*c,uint32_t d){(void)a;(void)b;(void)c;(void)d;return MSX_RESIDENT_DISABLED;}
MSXResidentStatus MSXresidentGpu_fetchHandoffBatch(MSXResidentGpu*a,const MSXResidentHandoffItem*b,uint32_t c,MSXResidentGpuFetchOutput*d){(void)a;(void)b;(void)c;(void)d;return MSX_RESIDENT_DISABLED;}
MSXResidentStatus MSXresidentGpu_getTransferStats(const MSXResidentGpu*a,MSXResidentGpuTransferStats*b){(void)a;if(b)memset(b,0,sizeof(*b));return MSX_RESIDENT_DISABLED;}
MSXResidentStatus MSXresidentGpu_reduce(MSXResidentGpu*a,double*b,uint32_t c,MSXResidentGpuReduction*d){(void)a;(void)b;(void)c;(void)d;return MSX_RESIDENT_DISABLED;}
MSXResidentStatus MSXresidentGpu_reduceLink(MSXResidentGpu*a,uint32_t b,double*c,uint32_t d,double*e,MSXResidentGpuReduction*f){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;return MSX_RESIDENT_DISABLED;}
MSXResidentStatus MSXresidentGpu_prepareActive(MSXResidentGpu*a,const MSXResidentActiveBatch*b,MSXResidentGpuDeviceView*c,MSXResidentGpuReactResult*d){(void)a;(void)b;if(c)memset(c,0,sizeof(*c));if(d)memset(d,0,sizeof(*d));return MSX_RESIDENT_DISABLED;}
MSXResidentStatus MSXresidentGpu_prepareActiveHyd(MSXResidentGpu*a,const MSXResidentActiveBatch*b,const MSXResidentHydView*h,MSXResidentGpuDeviceView*c,MSXResidentGpuReactResult*d){(void)a;(void)b;(void)h;if(c)memset(c,0,sizeof(*c));if(d)memset(d,0,sizeof(*d));return MSX_RESIDENT_DISABLED;}
MSXResidentStatus MSXresidentGpu_beginActive(MSXResidentGpu*a,uint32_t n,uint64_t t,MSXResidentGpuActiveWriter*w){(void)a;(void)n;(void)t;if(w)memset(w,0,sizeof(*w));return MSX_RESIDENT_DISABLED;}
MSXResidentStatus MSXresidentGpu_appendActive(MSXResidentGpu*a,MSXResidentGpuActiveWriter*w,const MSXResidentActiveRow*x){(void)a;(void)w;(void)x;return MSX_RESIDENT_DISABLED;}
MSXResidentStatus MSXresidentGpu_sealActive(MSXResidentGpu*a,MSXResidentGpuActiveWriter*w,uint64_t t){(void)a;(void)w;(void)t;return MSX_RESIDENT_DISABLED;}
MSXResidentStatus MSXresidentGpu_abortActiveBuild(MSXResidentGpu*a,MSXResidentGpuActiveWriter*w){(void)a;(void)w;return MSX_RESIDENT_DISABLED;}
MSXResidentStatus MSXresidentGpu_prepareSealedActiveHyd(MSXResidentGpu*a,MSXResidentGpuActiveWriter*w,const MSXResidentHydView*h,MSXResidentGpuDeviceView*c,MSXResidentGpuReactResult*d){(void)a;(void)w;(void)h;if(c)memset(c,0,sizeof(*c));if(d)memset(d,0,sizeof(*d));return MSX_RESIDENT_DISABLED;}
MSXResidentStatus MSXresidentGpu_getDeviceView(MSXResidentGpu*a,MSXResidentGpuDeviceView*b){(void)a;if(b)memset(b,0,sizeof(*b));return MSX_RESIDENT_DISABLED;}
MSXResidentStatus MSXresidentGpu_finishActive(MSXResidentGpu*a,MSXResidentGpuReactResult*b){(void)a;if(b)memset(b,0,sizeof(*b));return MSX_RESIDENT_DISABLED;}
MSXResidentStatus MSXresidentGpu_finishActiveAfterWait(MSXResidentGpu*a,MSXResidentGpuReactResult*b){(void)a;if(b)memset(b,0,sizeof(*b));return MSX_RESIDENT_DISABLED;}
MSXResidentStatus MSXresidentGpu_enqueueActiveCompletion(MSXResidentGpu*a){(void)a;return MSX_RESIDENT_DISABLED;}
MSXResidentStatus MSXresidentGpu_syncActive(MSXResidentGpu*a,MSXResidentGpuActiveSyncOutput*b,uint32_t c){(void)a;(void)b;(void)c;return MSX_RESIDENT_DISABLED;}
MSXResidentStatus MSXresidentGpu_abortActive(MSXResidentGpu*a){(void)a;return MSX_RESIDENT_DISABLED;}
MSXResidentStatus MSXresidentGpu_invalidateHyd(MSXResidentGpu*a){(void)a;return MSX_RESIDENT_DISABLED;}
MSXResidentStatus MSXresidentGpu_poison(MSXResidentGpu*a){(void)a;return MSX_RESIDENT_DISABLED;}
void MSXresidentGpu_close(MSXResidentGpu*a){(void)a;}
