/* Phase 2b CUDA-core contract harness: production .cu, no stub or simulation. */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <cuda_runtime_api.h>
#include "msxresident_core_cuda.h"
#include "msxresident_active_stream.h"

static int pass, fail, aggregateQueries, aggregateHits, aggregateRebuilds;
#define OK(x) do { if (x) ++pass; else { ++fail; fprintf(stderr,"FAIL:%d: %s\n",__LINE__,#x); } } while (0)
void MSXgpu_profileRecordAggregate(uint64_t q,uint64_t h,uint64_t r) { aggregateQueries+=(int)q; aggregateHits+=(int)h; aggregateRebuilds+=(int)r; }
#define S 3
static cudaError_t vcopy(const MSXResidentGpuDeviceView *v,void *dst,const void *src,size_t bytes,enum cudaMemcpyKind kind)
{
    cudaStream_t stream=(cudaStream_t)(uintptr_t)v->streamHandle;
    cudaError_t e=cudaMemcpyAsync(dst,src,bytes,kind,stream);
    return e==cudaSuccess?cudaStreamSynchronize(stream):e;
}
#define LARGE 2048
static void desc(MSXResidentDescriptorPatch *d,uint32_t link,uint32_t cap,uint32_t count,uint64_t epoch,int orient) { memset(d,0,sizeof(*d)); d->linkIndex=link; d->descriptor.linkIndex=link; d->descriptor.capacity=cap; d->descriptor.count=count; d->descriptor.head=count?0:UINT32_MAX; d->descriptor.tail=count?count-1:UINT32_MAX; d->descriptor.epoch=epoch; d->descriptor.orient=orient; }
static void slot(MSXResidentSlotPatch *x,uint32_t link,uint32_t local,uint32_t gen,uint64_t id,double v,double base) { static double c[4][S],l[4][S]; uint32_t q=(local+(link-1)*2)&3; memset(x,0,sizeof(*x)); c[q][0]=base;c[q][1]=base+1;c[q][2]=base+2;l[q][0]=base+10;l[q][1]=base+11;l[q][2]=base+12; x->linkIndex=link;x->slot=local;x->generation=gen;x->used=1;x->payload.parcelId=id;x->payload.volume=v;x->payload.hstep=base+3;x->payload.hresponse=base+4;x->payload.uresponse=base+5;x->payload.dresponse=base+6;x->payload.c=c[q];x->payload.lastc=l[q]; }
static MSXResidentGpu *open4(void) { uint32_t cap[3]={0,2,2},base[3]={0,0,2}; MSXResidentGpuOpen o={2,4,S,cap,base}; MSXResidentGpu*g=0; OK(MSXresidentGpu_open(&o,&g)==0&&g); return g; }
static MSXResidentGpu *initial4(void) { MSXResidentGpu*g=open4(); MSXResidentDescriptorPatch d[2]; MSXResidentSlotPatch x[4]; MSXResidentPatchBatch b; desc(&d[0],1,2,2,5,1);desc(&d[1],2,2,2,7,-1);slot(&x[0],1,0,1,101,2,10);slot(&x[1],1,1,1,102,3,20);slot(&x[2],2,0,1,201,4,30);slot(&x[3],2,1,1,202,5,40);b=(MSXResidentPatchBatch){d,2,x,4};OK(MSXresidentGpu_initialUpload(g,&b)==0);MSXresidentGpu_setDiagnosticMode(g,1);return g; }
static int snap(MSXResidentGpu*g,MSXResidentGpuReduction*r,double*m) { return MSXresidentGpu_reduce(g,m,S,r)==0; }
static int same(const MSXResidentGpuReduction*a,const MSXResidentGpuReduction*b,const double*x,const double*y) { return a->activeCount==b->activeCount&&a->checksumXor==b->checksumXor&&a->checksumSum==b->checksumSum&&a->activeVolume==b->activeVolume&&!memcmp(x,y,sizeof(double)*S); }
static void invalid_unchanged(MSXResidentGpu*g,MSXResidentPatchBatch*b,MSXResidentStatus want) { MSXResidentGpuReduction a,z;double x[S],y[S];OK(snap(g,&a,x));OK(MSXresidentGpu_applyPatches(g,b)==want);OK(snap(g,&z,y)&&same(&a,&z,x,y)); }
static void t_open(void) { uint32_t c[3]={0,2,2},b[3]={0,0,4};MSXResidentGpuOpen o={2,6,S,c,b};MSXResidentGpu*g=0; OK(MSXresidentGpu_open(&o,&g)==0);MSXresidentGpu_close(g);g=0;b[2]=1;OK(MSXresidentGpu_open(&o,&g)<0);b[2]=4;c[2]=3;OK(MSXresidentGpu_open(&o,&g)<0); {uint32_t zc[2]={0,1},zb[2]={0,0};MSXResidentGpuOpen z={1,1,S,zc,zb};MSXResidentDescriptorPatch d;MSXResidentSlotPatch x;MSXResidentPatchBatch q;MSXResidentStatus os,is;desc(&d,1,1,0,1,1);memset(&x,0,sizeof(x));x.linkIndex=1;x.slot=0;x.generation=1;q=(MSXResidentPatchBatch){&d,1,&x,1};os=MSXresidentGpu_open(&z,&g);is=os==0?MSXresidentGpu_initialUpload(g,&q):os;if(is)fprintf(stderr,"t_open initial=%d\n",(int)is);OK(os==0);OK(is==0);MSXresidentGpu_close(g);} }
static void t_initial(void) { MSXResidentGpu*g=open4();MSXResidentDescriptorPatch d[2];MSXResidentSlotPatch x[4];MSXResidentPatchBatch b;desc(&d[0],1,2,2,5,1);desc(&d[1],2,2,2,7,-1);slot(&x[0],1,0,1,1,1,1);slot(&x[1],1,1,1,2,1,2);slot(&x[2],2,0,1,3,1,3);slot(&x[3],2,1,1,4,1,4);b=(MSXResidentPatchBatch){d,2,x,4};OK(MSXresidentGpu_initialUpload(g,&b)==0);OK(MSXresidentGpu_initialUpload(g,&b)<0);MSXresidentGpu_close(g); g=open4();b.descriptorCount=1;OK(MSXresidentGpu_initialUpload(g,&b)<0);b.descriptorCount=2;b.slotCount=3;OK(MSXresidentGpu_initialUpload(g,&b)<0);b.slotCount=4;x[3].slot=0;OK(MSXresidentGpu_initialUpload(g,&b)<0);x[3].slot=1;b.slotCount=3;OK(MSXresidentGpu_initialUpload(g,&b)<0);MSXresidentGpu_close(g); }
static void t_scatter(void) { MSXResidentGpu*g=initial4();MSXResidentDescriptorPatch d;MSXResidentSlotPatch x[2];MSXResidentPatchBatch b;MSXResidentHandoffItem it[2];MSXResidentHandoffPlan p;MSXResidentHandoffResult r[2];double c[8],l[8];MSXResidentGpuFetchOutput out={r,c,l,4};desc(&d,2,2,2,8,-1);slot(&x[0],2,1,2,222,9,70);slot(&x[1],2,0,2,211,8,60);b=(MSXResidentPatchBatch){&d,1,x,2};OK(MSXresidentGpu_applyPatches(g,&b)==0);memset(it,0,sizeof(it));it[0]=(MSXResidentHandoffItem){2,1,2,0,8,0};it[1]=(MSXResidentHandoffItem){2,0,2,0,8,0};p=(MSXResidentHandoffPlan){2,0,2,8,0,it};OK(MSXresidentGpu_fetchHandoffs(g,&p,&out,2)==0&&r[0].payload.parcelId==222&&r[1].payload.parcelId==211&&c[0]==70&&c[4]==60&&r[0].payload.hstep==73);MSXresidentGpu_close(g); }
static void t_descriptor_generation(void) { MSXResidentGpu*g=initial4();MSXResidentDescriptorPatch d;MSXResidentPatchBatch b;desc(&d,1,2,2,6,0);b=(MSXResidentPatchBatch){&d,1,0,0};invalid_unchanged(g,&b,MSX_RESIDENT_ERR_ARGUMENT);desc(&d,1,2,2,4,1);invalid_unchanged(g,&b,MSX_RESIDENT_ERR_GENERATION);desc(&d,1,2,3,6,1);invalid_unchanged(g,&b,MSX_RESIDENT_ERR_ARGUMENT); {MSXResidentSlotPatch x;desc(&d,1,2,2,6,1);slot(&x,1,0,2,301,2,50);b=(MSXResidentPatchBatch){&d,1,&x,1};OK(MSXresidentGpu_applyPatches(g,&b)==0);x.generation=1;invalid_unchanged(g,&b,MSX_RESIDENT_ERR_GENERATION);} MSXresidentGpu_close(g); }
static void t_fetch(void) { MSXResidentGpu*g=initial4();MSXResidentHandoffItem a[2],b[1];MSXResidentHandoffPlan p;MSXResidentHandoffResult r[2];double c[8],l[8];MSXResidentGpuFetchOutput o={r,c,l,4};memset(a,0,sizeof(a));a[0]=(MSXResidentHandoffItem){1,1,1,0,5,0};a[1]=(MSXResidentHandoffItem){1,0,1,0,5,0};p=(MSXResidentHandoffPlan){1,0,2,5,0,a};OK(MSXresidentGpu_fetchHandoffs(g,&p,&o,2)==0&&r[0].payload.parcelId==102&&r[1].payload.parcelId==101&&c[0]==20&&c[4]==10);c[0]=777;o.stride=2;OK(MSXresidentGpu_fetchHandoffs(g,&p,&o,2)<0&&c[0]==777);o.stride=4;o.cOut=0;OK(MSXresidentGpu_fetchHandoffs(g,&p,&o,2)<0);o.cOut=c;b[0]=(MSXResidentHandoffItem){2,0,1,0,7,0};p=(MSXResidentHandoffPlan){2,0,1,7,0,b};OK(MSXresidentGpu_fetchHandoffs(g,&p,&o,1)==0&&r[0].payload.parcelId==201);b[0].generation=9;OK(MSXresidentGpu_fetchHandoffs(g,&p,&o,1)==MSX_RESIDENT_ERR_GENERATION);b[0].generation=1;b[0].pipeEpoch=6;p.pipeEpoch=6;OK(MSXresidentGpu_fetchHandoffs(g,&p,&o,1)==MSX_RESIDENT_ERR_GENERATION); {MSXResidentGpuReduction z;double m[S];OK(snap(g,&z,m)&&z.fetchStale>=1&&z.epochStale>=1);}MSXresidentGpu_close(g); }
/* A selected demote batch is a flat multi-link gather: one staging H2D and
   exactly three D2H APIs (metadata, c, lastc), independent of row count.
   The counters are API calls, not logical handoff rows. */
static void t_fetch_batch_contract(void) { MSXResidentGpu*g=initial4();MSXResidentHandoffItem a[2];MSXResidentHandoffResult r[2];MSXResidentGpuFetchOutput o;MSXResidentGpuTransferStats before,after;double c[8]={0},l[8]={0};a[0]=(MSXResidentHandoffItem){1,0,1,0,5,0};a[1]=(MSXResidentHandoffItem){2,1,1,0,7,1};o=(MSXResidentGpuFetchOutput){r,c,l,4};OK(MSXresidentGpu_getTransferStats(g,&before)==0&&before.h2dCalls==0&&before.d2hCalls==0);OK(MSXresidentGpu_fetchHandoffBatch(g,a,2,&o)==0&&r[0].payload.parcelId==101&&r[1].payload.parcelId==202&&c[0]==10&&c[4]==40);OK(MSXresidentGpu_getTransferStats(g,&after)==0&&after.h2dCalls==1&&after.d2hCalls==3&&after.h2dBytes>0&&after.d2hBytes>0);a[1].pipeEpoch=8;OK(MSXresidentGpu_fetchHandoffBatch(g,a,2,&o)==MSX_RESIDENT_ERR_GENERATION);OK(MSXresidentGpu_getTransferStats(g,&before)==0&&before.h2dCalls==after.h2dCalls&&before.d2hCalls==after.d2hCalls);a[1].pipeEpoch=7;a[1].linkIndex=1;a[1].slot=0;OK(MSXresidentGpu_fetchHandoffBatch(g,a,2,&o)==MSX_RESIDENT_ERR_ARGUMENT);OK(MSXresidentGpu_getTransferStats(g,&before)==0&&before.h2dCalls==after.h2dCalls&&before.d2hCalls==after.d2hCalls);MSXresidentGpu_close(g); }
static void t_reduce_lifecycle(void) { MSXResidentGpu*g=initial4();MSXResidentGpuReduction a,b;double x[S],y[S],hyd[MSX_RESIDENT_HYD_STRIDE]={0},changed[S]={123,21,22};aggregateQueries=aggregateHits=aggregateRebuilds=0;OK(snap(g,&a,x)&&a.activeCount==4&&a.activeVolume==14&&x[0]==(10*2+20*3+30*4+40*5));OK(snap(g,&b,y)&&same(&a,&b,x,y)); {MSXResidentActiveItem ai={1,0,1,5,2,hyd};MSXResidentActiveBatch ab={&ai,1};MSXResidentGpuDeviceView v;MSXResidentGpuReactResult rr;OK(MSXresidentGpu_prepareActive(g,&ab,&v,&rr)==0&&vcopy(&v,(void*)(uintptr_t)v.c,changed,sizeof(changed),cudaMemcpyHostToDevice)==cudaSuccess&&MSXresidentGpu_finishActive(g,&rr)==0&&snap(g,&a,x)&&x[0]==(123*2+20*3+30*4+40*5));} {MSXResidentDescriptorPatch d;MSXResidentSlotPatch q;MSXResidentPatchBatch z;desc(&d,1,2,2,6,1);slot(&q,1,0,2,101,2,NAN);((double*)q.payload.c)[1]=1;((double*)q.payload.c)[2]=2;z=(MSXResidentPatchBatch){&d,1,&q,1};OK(MSXresidentGpu_applyPatches(g,&z)==0&&snap(g,&a,x)&&a.nanCount==1&&a.errorCount==1&&a.firstErrorSlot==0);slot(&q,1,1,2,102,3,INFINITY);((double*)q.payload.c)[1]=1;((double*)q.payload.c)[2]=2;OK(MSXresidentGpu_applyPatches(g,&z)==0&&snap(g,&a,x)&&a.infCount==1&&a.nanCount==1&&a.errorCount==2&&a.firstErrorSlot==0);}OK(aggregateQueries==5&&aggregateHits==1&&aggregateRebuilds==4);MSXresidentGpu_close(g);MSXresidentGpu_close(0);for(int i=0;i<3;i++){g=open4();MSXresidentGpu_close(g);} }
static void t_hole_initial_fetch(void) { uint32_t cap[3]={0,2,2},base[3]={0,0,4};MSXResidentGpuOpen o={2,6,S,cap,base};MSXResidentGpu*g=0;MSXResidentDescriptorPatch d[2];MSXResidentSlotPatch x[4];MSXResidentPatchBatch q;MSXResidentGpuReduction z;MSXResidentStatus os,is;double m[S],c[S],l[S];MSXResidentHandoffItem i={2,0,1,0,7,0};MSXResidentHandoffPlan p={2,0,1,7,0,&i};MSXResidentHandoffResult r;MSXResidentGpuFetchOutput out={&r,c,l,S};desc(&d[0],1,2,2,5,1);desc(&d[1],2,2,2,7,1);slot(&x[0],1,0,1,101,2,10);slot(&x[1],1,1,1,102,3,20);slot(&x[2],2,0,1,201,4,30);slot(&x[3],2,1,1,202,5,40);q=(MSXResidentPatchBatch){d,2,x,4};os=MSXresidentGpu_open(&o,&g);is=os==0?MSXresidentGpu_initialUpload(g,&q):os;if(is)fprintf(stderr,"t_hole initial=%d\n",(int)is);OK(os==0);OK(is==0);OK(snap(g,&z,m)&&z.activeCount==4&&z.activeVolume==14);OK(MSXresidentGpu_fetchHandoffs(g,&p,&out,1)==0&&r.payload.parcelId==201&&c[0]==30&&l[2]==42);MSXresidentGpu_close(g); }
static void t_ring_span_reject(void) { uint32_t cap[2]={0,3},base[2]={0,0};MSXResidentGpuOpen o={1,3,S,cap,base};MSXResidentGpu*g=0;MSXResidentDescriptorPatch d;MSXResidentSlotPatch x[3],off;MSXResidentPatchBatch q,b;desc(&d,1,3,3,1,1);slot(&x[0],1,0,1,1,1,1);slot(&x[1],1,1,1,2,1,2);slot(&x[2],1,2,1,3,1,3);q=(MSXResidentPatchBatch){&d,1,x,3};OK(MSXresidentGpu_open(&o,&g)==0&&MSXresidentGpu_initialUpload(g,&q)==0);desc(&d,1,3,2,2,1);d.descriptor.tail=2;memset(&off,0,sizeof(off));off.linkIndex=1;off.slot=1;off.generation=2;off.used=0;b=(MSXResidentPatchBatch){&d,1,&off,1};invalid_unchanged(g,&b,MSX_RESIDENT_ERR_ARGUMENT);MSXresidentGpu_close(g); }
static void t_wrap_active(void) { uint32_t cap[2]={0,4},base[2]={0,0};MSXResidentGpuOpen o={1,4,S,cap,base};MSXResidentGpu*g=0;MSXResidentDescriptorPatch d;MSXResidentSlotPatch x[4];MSXResidentPatchBatch q;double hyd[MSX_RESIDENT_HYD_STRIDE]={0};MSXResidentActiveItem a[2];MSXResidentActiveBatch b;MSXResidentGpuDeviceView v;MSXResidentGpuReactResult r;desc(&d,1,4,2,5,-1);d.descriptor.tail=3;slot(&x[0],1,0,1,101,1,1);memset(&x[1],0,sizeof(x[1]));x[1].linkIndex=1;x[1].slot=1;memset(&x[2],0,sizeof(x[2]));x[2].linkIndex=1;x[2].slot=2;slot(&x[3],1,3,1,102,1,2);q=(MSXResidentPatchBatch){&d,1,x,4};a[0]=(MSXResidentActiveItem){1,0,1,5,1,hyd};a[1]=(MSXResidentActiveItem){1,3,1,5,1,hyd};b=(MSXResidentActiveBatch){a,2};OK(MSXresidentGpu_open(&o,&g)==0&&MSXresidentGpu_initialUpload(g,&q)==0);OK(MSXresidentGpu_prepareActive(g,&b,&v,&r)==0&&v.itemCount==2&&v.hydStride==MSX_RESIDENT_HYD_STRIDE);OK(MSXresidentGpu_finishActive(g,&r)==0);MSXresidentGpu_close(g); }
static void t_stream_builder(void)
{
    MSXResidentGpu *g=initial4();
    MSXResidentGpuActiveWriter w;
    MSXResidentActiveRow row[2];
    MSXResidentActiveItem legacy;
    MSXResidentActiveBatch oldBatch;
    MSXResidentHydView hyd;
    MSXResidentGpuDeviceView v;
    MSXResidentGpuReactResult r;
    MSXResidentGpuTransferStats s0,s1;
    double pipe[(size_t)3*MSX_RESIDENT_HYD_STRIDE]={0};
    memset(row,0,sizeof(row));
    row[0]=(MSXResidentActiveRow){1,0,0,1,0,2,1,5,101,2.0};
    row[1]=(MSXResidentActiveRow){2,1,3,1,0,2,-1,7,202,5.0};
    hyd=(MSXResidentHydView){pipe,2,MSX_RESIDENT_HYD_STRIDE,MSX_RESIDENT_HYD_PIPE_MAJOR};
    legacy=(MSXResidentActiveItem){1,0,1,5,2.0,0};
    oldBatch=(MSXResidentActiveBatch){&legacy,1};
    OK(MSXresidentGpu_beginActive(g,2,19,&w)==0&&w.building&&w.count==0&&w.expectedCount==2);
    /* BUILDING is host-only: the legacy path cannot consume the pinned rows. */
    OK(MSXresidentGpu_prepareActiveHyd(g,&oldBatch,&hyd,&v,&r)==MSX_RESIDENT_ERR_ARGUMENT);
    OK(MSXresidentGpu_appendActive(g,&w,&row[0])==0&&w.count==1);
    OK(MSXresidentGpu_appendActive(g,&w,&row[1])==0&&w.count==2);
    OK(w.touchedRows&&w.touchedCount==2&&w.touchedRows[0]==0&&w.touchedRows[1]==3);
    OK(MSXresidentGpu_sealActive(g,&w,19)==0&&w.sealed&&!w.building);
    OK(MSXresidentGpu_getTransferStats(g,&s0)==0&&s0.hydH2DCalls==0);
    OK(MSXresidentGpu_prepareSealedActiveHyd(g,&w,&hyd,&v,&r)==0&&v.itemCount==2);
    OK(MSXresidentGpu_getTransferStats(g,&s1)==0&&s1.hydH2DCalls==1&&s1.hydUploads==1);
    OK(MSXresidentGpu_finishActive(g,&r)==0);
    /* The pinned writer is not reusable until the previous finish. */
    OK(MSXresidentGpu_beginActive(g,0,20,&w)==0&&MSXresidentGpu_sealActive(g,&w,20)==0);
    OK(MSXresidentGpu_prepareSealedActiveHyd(g,&w,&hyd,&v,&r)==0&&v.itemCount==0);
    OK(MSXresidentGpu_finishActive(g,&r)==0);
    OK(MSXresidentGpu_beginActive(g,0,21,&w)==0&&MSXresidentGpu_sealActive(g,&w,21)==0);
    OK(MSXresidentGpu_abortActive(g)==0);
    /* Duplicate and stale rows clear only the touched prefix and invalidate
       the writer; a fresh begin remains possible on the untouched mirror. */
    OK(MSXresidentGpu_beginActive(g,2,22,&w)==0&&MSXresidentGpu_appendActive(g,&w,&row[0])==0);
    OK(MSXresidentGpu_appendActive(g,&w,&row[0])==MSX_RESIDENT_ERR_ARGUMENT);
    OK(MSXresidentGpu_beginActive(g,2,23,&w)==0);
    row[0].generation=9;
    OK(MSXresidentGpu_appendActive(g,&w,&row[0])==MSX_RESIDENT_ERR_GENERATION);
    row[0].generation=1;
    MSXresidentGpu_close(g);
}
static void t_active_view(void) { MSXResidentGpu*g=initial4(); double hyd[MSX_RESIDENT_HYD_STRIDE]={0},h; uint32_t u; MSXResidentActiveItem a={1,0,1,5,2.0,hyd}, dup[2]={{1,0,1,5,2.0,hyd},{1,0,1,5,2.0,hyd}}, stale={1,0,9,5,2.0,hyd}; MSXResidentActiveBatch b={&a,1}, d={dup,2}, z={&stale,1}; MSXResidentGpuDeviceView v,w; MSXResidentGpuReactResult r; OK(MSXresidentGpu_prepareActive(g,&b,&v,&r)==0&&v.itemCount==1&&v.segRow&&v.c&&v.lastc&&v.hyd&&v.ros2Err&&v.hydStride==MSX_RESIDENT_HYD_STRIDE); OK(MSXresidentGpu_getDeviceView(g,&w)==0&&w.hstep==v.hstep&&w.hydStride==MSX_RESIDENT_HYD_STRIDE); h=91.25;u=11;OK(vcopy(&v,(void*)(uintptr_t)v.hstep,&h,sizeof(h),cudaMemcpyHostToDevice)==cudaSuccess);OK(vcopy(&v,(void*)(uintptr_t)v.ros2LastHstep,&h,sizeof(h),cudaMemcpyHostToDevice)==cudaSuccess);OK(vcopy(&v,(void*)(uintptr_t)v.ros2Nfcn,&u,sizeof(u),cudaMemcpyHostToDevice)==cudaSuccess);u=12;OK(vcopy(&v,(void*)(uintptr_t)v.ros2Njac,&u,sizeof(u),cudaMemcpyHostToDevice)==cudaSuccess);u=13;OK(vcopy(&v,(void*)(uintptr_t)v.ros2Naccept,&u,sizeof(u),cudaMemcpyHostToDevice)==cudaSuccess);u=14;OK(vcopy(&v,(void*)(uintptr_t)v.ros2Nreject,&u,sizeof(u),cudaMemcpyHostToDevice)==cudaSuccess);OK(MSXresidentGpu_finishActive(g,&r)==0&&r.ros2Nfcn==11&&r.ros2Njac==12&&r.ros2Naccept==13&&r.ros2Nreject==14&&r.ros2LastHstep==91.25); OK(MSXresidentGpu_prepareActive(g,&b,&v,&r)==0);OK(vcopy(&v,&h,(void*)(uintptr_t)v.hstep,sizeof(h),cudaMemcpyDeviceToHost)==cudaSuccess&&h==91.25);OK(MSXresidentGpu_finishActive(g,&r)==0); OK(MSXresidentGpu_prepareActive(g,&d,&v,&r)==MSX_RESIDENT_ERR_ARGUMENT); OK(MSXresidentGpu_prepareActive(g,&z,&v,&r)==MSX_RESIDENT_ERR_GENERATION); MSXresidentGpu_close(g); }
static void t_pipe_hyd_contract(void) { MSXResidentGpu*g=initial4(); double pipe[(size_t)3*MSX_RESIDENT_HYD_STRIDE],back[(size_t)3*MSX_RESIDENT_HYD_STRIDE]; MSXResidentActiveItem a[2]={{1,0,1,5,2.0,0},{2,3,1,7,3.0,0}}; MSXResidentActiveBatch b={a,2},empty={0,0}; MSXResidentHydView h={pipe,2,MSX_RESIDENT_HYD_STRIDE,MSX_RESIDENT_HYD_PIPE_MAJOR},bad; MSXResidentGpuDeviceView v; MSXResidentGpuReactResult r; MSXResidentGpuTransferStats before,after; size_t bytes=sizeof(pipe); for(size_t i=0;i<sizeof(pipe)/sizeof(pipe[0]);++i) pipe[i]=900.0+(double)i; for(uint32_t m=0;m<MSX_RESIDENT_HYD_STRIDE;++m){pipe[MSX_RESIDENT_HYD_STRIDE+m]=100.0+m;pipe[2*MSX_RESIDENT_HYD_STRIDE+m]=200.0+m;} memset(back,0,sizeof(back)); bad=h;bad.hydLayout=MSX_RESIDENT_HYD_ACTIVE_MAJOR;OK(MSXresidentGpu_prepareActiveHyd(g,&b,&bad,&v,&r)==MSX_RESIDENT_ERR_ARGUMENT);bad=h;bad.hydStride=MSX_RESIDENT_HYD_STRIDE-1;OK(MSXresidentGpu_prepareActiveHyd(g,&b,&bad,&v,&r)==MSX_RESIDENT_ERR_ARGUMENT);bad=h;bad.linkCount=1;OK(MSXresidentGpu_prepareActiveHyd(g,&b,&bad,&v,&r)==MSX_RESIDENT_ERR_ARGUMENT); OK(MSXresidentGpu_getTransferStats(g,&before)==0&&before.hydH2DCalls==0&&before.hydH2DBytes==0); OK(MSXresidentGpu_prepareActiveHyd(g,&b,&h,&v,&r)==0&&v.itemCount==2&&v.hydLayout==MSX_RESIDENT_HYD_PIPE_MAJOR&&v.hydStride==MSX_RESIDENT_HYD_STRIDE); OK(MSXresidentGpu_getTransferStats(g,&after)==0&&after.hydH2DCalls==1&&after.hydH2DBytes==bytes&&after.hydCandidateComparisons==1&&after.hydUploads==1&&after.hydSkips==0); OK(vcopy(&v,back,(const void*)(uintptr_t)v.hyd,bytes,cudaMemcpyDeviceToHost)==cudaSuccess&&memcmp(back,pipe,bytes)==0); OK(MSXresidentGpu_finishActive(g,&r)==0); OK(MSXresidentGpu_prepareActiveHyd(g,&empty,&h,&v,&r)==0&&v.itemCount==0&&v.hydLayout==MSX_RESIDENT_HYD_PIPE_MAJOR); OK(MSXresidentGpu_getTransferStats(g,&after)==0&&after.hydH2DCalls==1&&after.hydH2DBytes==bytes&&after.hydCandidateComparisons==2&&after.hydUploads==1&&after.hydSkips==1); OK(MSXresidentGpu_finishActive(g,&r)==0); MSXresidentGpu_close(g); }
static void t_pipe_hyd_exact_bits(void)
{
    MSXResidentGpu*g=initial4();
    double pipe[(size_t)3*MSX_RESIDENT_HYD_STRIDE]={0};
    MSXResidentHydView h={pipe,2,MSX_RESIDENT_HYD_STRIDE,MSX_RESIDENT_HYD_PIPE_MAJOR};
    MSXResidentActiveBatch empty={0,0}; MSXResidentGpuDeviceView v;
    MSXResidentGpuReactResult r; MSXResidentGpuTransferStats s;
    uint64_t bits;
    pipe[0]=+0.0;
    OK(MSXresidentGpu_prepareActiveHyd(g,&empty,&h,&v,&r)==0&&MSXresidentGpu_finishActive(g,&r)==0);
    pipe[0]=-0.0;
    OK(MSXresidentGpu_prepareActiveHyd(g,&empty,&h,&v,&r)==0&&MSXresidentGpu_finishActive(g,&r)==0);
    bits=UINT64_C(0x7ff8000000000001); memcpy(&pipe[1],&bits,sizeof(bits));
    OK(MSXresidentGpu_prepareActiveHyd(g,&empty,&h,&v,&r)==0&&MSXresidentGpu_finishActive(g,&r)==0);
    bits=UINT64_C(0x7ff8000000000002); memcpy(&pipe[1],&bits,sizeof(bits));
    OK(MSXresidentGpu_prepareActiveHyd(g,&empty,&h,&v,&r)==0&&MSXresidentGpu_finishActive(g,&r)==0);
    OK(MSXresidentGpu_getTransferStats(g,&s)==0&&s.hydCandidateComparisons==4&&
       s.hydUploads==4&&s.hydSkips==0&&s.hydApiCalls==4&&s.hydBytes==
       (uint64_t)4*sizeof(pipe));
    OK(MSXresidentGpu_invalidateHyd(g)==0);
    OK(MSXresidentGpu_prepareActiveHyd(g,&empty,&h,&v,&r)==0&&MSXresidentGpu_finishActive(g,&r)==0);
    OK(MSXresidentGpu_getTransferStats(g,&s)==0&&s.hydUploads==5);
    MSXresidentGpu_close(g);
}
static void t_pipe_hyd_failure_invalidates(void)
{
    double pipe[(size_t)3*MSX_RESIDENT_HYD_STRIDE]={0};
    double hyd[MSX_RESIDENT_HYD_STRIDE]={0};
    MSXResidentHydView h={pipe,2,MSX_RESIDENT_HYD_STRIDE,MSX_RESIDENT_HYD_PIPE_MAJOR};
    MSXResidentActiveItem item={1,0,1,5,2.0,hyd};
    MSXResidentActiveBatch batch={&item,1};
    MSXResidentGpuDeviceView v; MSXResidentGpuReactResult r;
    MSXResidentGpuTransferStats s; int e=-7;
    MSXResidentGpu *g=initial4();
    OK(MSXresidentGpu_prepareActiveHyd(g,&batch,&h,&v,&r)==0);
    OK(MSXresidentGpu_getTransferStats(g,&s)==0&&s.hydAppliedValid==0&&s.hydPending==1);
    OK(vcopy(&v,(void*)(uintptr_t)v.ros2Err,&e,sizeof(e),cudaMemcpyHostToDevice)==cudaSuccess);
    OK(MSXresidentGpu_finishActive(g,&r)==MSX_RESIDENT_ERR_GPU&&r.ros2Error==-7);
    OK(MSXresidentGpu_getTransferStats(g,&s)==0&&s.hydAppliedValid==0&&s.hydPending==0);
    MSXresidentGpu_close(g);
    /* A failed batch cannot make the next open/reaction skip its first upload. */
    g=initial4();
    OK(MSXresidentGpu_prepareActiveHyd(g,&batch,&h,&v,&r)==0);
    OK(MSXresidentGpu_getTransferStats(g,&s)==0&&s.hydUploads==1&&s.hydApiCalls==1);
    OK(MSXresidentGpu_finishActive(g,&r)==0);
    MSXresidentGpu_close(g);
    /* Abort also invalidates a pending submission, and invalidation is
       rejected while its stream token is still in flight. */
    g=initial4();
    OK(MSXresidentGpu_prepareActiveHyd(g,&batch,&h,&v,&r)==0);
    OK(MSXresidentGpu_invalidateHyd(g)==MSX_RESIDENT_ERR_ARGUMENT);
    OK(MSXresidentGpu_abortActive(g)==0);
    OK(MSXresidentGpu_getTransferStats(g,&s)==0&&s.hydAppliedValid==0&&s.hydPending==0);
    MSXresidentGpu_close(g);
}
static void t_active_error(void) { MSXResidentGpu*g=initial4();double hyd[MSX_RESIDENT_HYD_STRIDE]={0};int e=-2;MSXResidentActiveItem a={1,0,1,5,2,hyd};MSXResidentActiveBatch b={&a,1};MSXResidentGpuDeviceView v;MSXResidentGpuReactResult r;OK(MSXresidentGpu_prepareActive(g,&b,&v,&r)==0);OK(vcopy(&v,(void*)(uintptr_t)v.ros2Err,&e,sizeof(e),cudaMemcpyHostToDevice)==cudaSuccess);OK(MSXresidentGpu_finishActive(g,&r)==MSX_RESIDENT_ERR_GPU&&r.ros2Error==-2);OK(MSXresidentGpu_prepareActive(g,&b,&v,&r)==MSX_RESIDENT_ERR_POISONED);MSXresidentGpu_close(g);}
static void t_active_empty(void) { MSXResidentGpu*g=initial4();MSXResidentActiveBatch b={0,0};MSXResidentGpuDeviceView v;MSXResidentGpuReactResult r;OK(MSXresidentGpu_prepareActive(g,&b,&v,&r)==0&&v.itemCount==0&&v.streamHandle!=0);OK(MSXresidentGpu_finishActive(g,&r)==0&&r.ros2Error==0&&r.reacted&&r.reactedStride==S&&r.reactedLinkCount==3);for(uint32_t i=0;i<r.reactedLinkCount*r.reactedStride;i++)OK(r.reacted[i]==0.0);OK(MSXresidentGpu_finishActive(g,&r)==MSX_RESIDENT_ERR_ARGUMENT);MSXresidentGpu_close(g);}
static void t_active_abort_and_query_guard(void) { MSXResidentGpu*g=initial4();double hyd[MSX_RESIDENT_HYD_STRIDE]={0},mass[S],c[S],l[S];MSXResidentActiveItem a={1,0,1,5,2,hyd};MSXResidentActiveBatch b={&a,1};MSXResidentGpuDeviceView v;MSXResidentGpuReactResult r;MSXResidentGpuReduction q;MSXResidentHandoffItem item={1,0,1,0,5,0};MSXResidentHandoffResult meta;MSXResidentGpuFetchOutput out={&meta,c,l,S};OK(MSXresidentGpu_prepareActive(g,&b,&v,&r)==0);OK(MSXresidentGpu_reduce(g,mass,S,&q)==MSX_RESIDENT_ERR_ARGUMENT);OK(MSXresidentGpu_fetchHandoffBatch(g,&item,1,&out)==MSX_RESIDENT_ERR_ARGUMENT);OK(MSXresidentGpu_abortActive(g)==0);OK(MSXresidentGpu_prepareActive(g,&b,&v,&r)==MSX_RESIDENT_ERR_POISONED);MSXresidentGpu_close(g);}
static void t_explicit_poison_query_guard(void) { MSXResidentGpu*g=initial4();double mass[S],c[S],l[S];MSXResidentGpuReduction q;MSXResidentHandoffItem item={1,0,1,0,5,0};MSXResidentHandoffResult meta;MSXResidentGpuFetchOutput out={&meta,c,l,S};OK(MSXresidentGpu_poison(g)==MSX_RESIDENT_OK);OK(MSXresidentGpu_reduce(g,mass,S,&q)==MSX_RESIDENT_ERR_POISONED);OK(MSXresidentGpu_fetchHandoffBatch(g,&item,1,&out)==MSX_RESIDENT_ERR_POISONED);OK(MSXresidentGpu_poison(g)==MSX_RESIDENT_OK);MSXresidentGpu_close(g);}
static void t_large_batch(void) { static MSXResidentSlotPatch x[LARGE]; static MSXResidentActiveItem a[LARGE]; static MSXResidentHandoffItem hi[LARGE]; static MSXResidentHandoffResult hr[LARGE]; static double hc[(size_t)LARGE*S],hl[(size_t)LARGE*S]; MSXResidentDescriptorPatch d,dd[2]; MSXResidentPatchBatch b; MSXResidentGpu*g=0; MSXResidentGpuDeviceView v; MSXResidentGpuReactResult r; MSXResidentGpuReduction z; MSXResidentGpuFetchOutput fo; MSXResidentGpuTransferStats ts0,ts1; double m[S],hyd[MSX_RESIDENT_HYD_STRIDE]={0}; uint32_t cap[2]={0,LARGE},base[2]={0,0}; MSXResidentGpuOpen o={1,LARGE,S,cap,base}; for(uint32_t i=0;i<LARGE;i++){slot(&x[i],1,i,1,1000+i,1.0,(double)i);a[i]=(MSXResidentActiveItem){1,i,1,1,1.0,hyd};hi[i]=(MSXResidentHandoffItem){1,i,1,0,1};}desc(&d,1,LARGE,LARGE,1,1);b=(MSXResidentPatchBatch){&d,1,x,LARGE};OK(MSXresidentGpu_open(&o,&g)==0&&MSXresidentGpu_initialUpload(g,&b)==0);fo=(MSXResidentGpuFetchOutput){hr,hc,hl,S};OK(MSXresidentGpu_getTransferStats(g,&ts0)==0&&ts0.h2dCalls==0&&ts0.d2hCalls==0);OK(MSXresidentGpu_fetchHandoffBatch(g,hi,LARGE,&fo)==0&&hr[LARGE-1].payload.parcelId==1000+LARGE-1);OK(MSXresidentGpu_getTransferStats(g,&ts1)==0&&ts1.h2dCalls==1&&ts1.d2hCalls==3&&ts1.h2dBytes>0&&ts1.d2hBytes>0);OK(snap(g,&z,m)&&z.activeCount==LARGE);dd[0]=d;dd[1]=d;b=(MSXResidentPatchBatch){dd,2,0,0};invalid_unchanged(g,&b,MSX_RESIDENT_ERR_ARGUMENT);b=(MSXResidentPatchBatch){0,0,x,1};x[0].slot=LARGE;invalid_unchanged(g,&b,MSX_RESIDENT_ERR_CAPACITY);x[0].slot=0;{MSXResidentActiveBatch q={a,LARGE};OK(MSXresidentGpu_prepareActive(g,&q,&v,&r)==0&&v.itemCount==LARGE&&v.hydStride==MSX_RESIDENT_HYD_STRIDE);OK(MSXresidentGpu_finishActive(g,&r)==0);}MSXresidentGpu_close(g);}
/* S06a streaming writer contract: the host writer accepts the second chunk
   after 256 rows, while duplicate/generation/epoch failures stay host-only.
   Transfer counters prove those failures cannot submit a new GPU batch. */
static void t_stream_writer_large(void)
{
    enum { CAP = 300, ACTIVE = 257 };
    static MSXResidentSlotPatch slots[CAP];
    static MSXResidentActiveRow rows[ACTIVE];
    uint32_t cap[2] = {0, CAP}, base[2] = {0, 0};
    MSXResidentGpuOpen open = {1, CAP, S, cap, base};
    MSXResidentDescriptorPatch d;
    MSXResidentPatchBatch batch;
    MSXResidentGpu *g = 0;
    MSXResidentGpuActiveWriter w;
    MSXResidentGpuTransferStats before, after;
    MSXResidentHydView hyd;
    MSXResidentGpuDeviceView view;
    MSXResidentGpuReactResult result;
    double pipe[(size_t)2 * MSX_RESIDENT_HYD_STRIDE] = {0};
    uint32_t i;

    desc(&d, 1, CAP, ACTIVE, 11, 1);
    d.descriptor.tail = ACTIVE - 1;
    for (i = 0; i < ACTIVE; ++i)
    {
        slot(&slots[i], 1, i, 1, UINT64_C(5000) + i, 1.0, (double)i);
        rows[i] = (MSXResidentActiveRow){1, i, i, 1, 0, ACTIVE, 1,
                                         11, UINT64_C(5000) + i, 1.0};
    }
    for (; i < CAP; ++i)
    {
        memset(&slots[i], 0, sizeof(slots[i]));
        slots[i].linkIndex = 1;
        slots[i].slot = i;
    }
    batch = (MSXResidentPatchBatch){&d, 1, slots, CAP};
    open.totalSlots = CAP;
    OK(MSXresidentGpu_open(&open, &g) == MSX_RESIDENT_OK &&
       MSXresidentGpu_initialUpload(g, &batch) == MSX_RESIDENT_OK);
    hyd = (MSXResidentHydView){pipe, 1, MSX_RESIDENT_HYD_STRIDE,
                               MSX_RESIDENT_HYD_PIPE_MAJOR};

    OK(MSXresidentGpu_beginActive(g, ACTIVE, 30, &w) == MSX_RESIDENT_OK);
    for (i = 0; i < ACTIVE; ++i)
        OK(MSXresidentGpu_appendActive(g, &w, &rows[i]) == MSX_RESIDENT_OK);
    OK(w.count == ACTIVE && w.touchedCount == ACTIVE &&
       w.touchedRows[255] == 255 && w.touchedRows[256] == 256);
    OK(MSXresidentGpu_sealActive(g, &w, 30) == MSX_RESIDENT_OK && w.sealed);
    OK(MSXresidentGpu_getTransferStats(g, &before) == MSX_RESIDENT_OK &&
       before.hydH2DCalls == 0);
    OK(MSXresidentGpu_prepareSealedActiveHyd(g, &w, &hyd, &view, &result) ==
           MSX_RESIDENT_OK && view.itemCount == ACTIVE);
    OK(MSXresidentGpu_finishActive(g, &result) == MSX_RESIDENT_OK);
    OK(MSXresidentGpu_getTransferStats(g, &before) == MSX_RESIDENT_OK &&
       before.hydH2DCalls == 1);

    /* Duplicate clears the writer once; no transfer or launch is added. */
    OK(MSXresidentGpu_beginActive(g, ACTIVE, 31, &w) == MSX_RESIDENT_OK);
    OK(MSXresidentGpu_appendActive(g, &w, &rows[0]) == MSX_RESIDENT_OK);
    OK(MSXresidentGpu_appendActive(g, &w, &rows[0]) ==
           MSX_RESIDENT_ERR_ARGUMENT);
    OK(MSXresidentGpu_getTransferStats(g, &after) == MSX_RESIDENT_OK &&
       after.hydH2DCalls == before.hydH2DCalls && after.h2dCalls == before.h2dCalls);

    rows[0].generation = 9;
    OK(MSXresidentGpu_beginActive(g, ACTIVE, 32, &w) == MSX_RESIDENT_OK);
    OK(MSXresidentGpu_appendActive(g, &w, &rows[0]) ==
           MSX_RESIDENT_ERR_GENERATION);
    rows[0].generation = 1;
    rows[0].descriptorEpoch = 12;
    OK(MSXresidentGpu_beginActive(g, ACTIVE, 33, &w) == MSX_RESIDENT_OK);
    OK(MSXresidentGpu_appendActive(g, &w, &rows[0]) ==
           MSX_RESIDENT_ERR_GENERATION);
    rows[0].descriptorEpoch = 11;
    OK(MSXresidentGpu_getTransferStats(g, &after) == MSX_RESIDENT_OK &&
       after.hydH2DCalls == before.hydH2DCalls && after.h2dCalls == before.h2dCalls);

    /* abortActiveBuild is idempotence-guarded: a second call cannot submit or
       mutate a new writer, and a fresh begin remains possible afterward. */
    OK(MSXresidentGpu_beginActive(g, ACTIVE, 34, &w) == MSX_RESIDENT_OK &&
       MSXresidentGpu_appendActive(g, &w, &rows[0]) == MSX_RESIDENT_OK);
    OK(MSXresidentGpu_abortActiveBuild(g, &w) == MSX_RESIDENT_OK);
    OK(MSXresidentGpu_abortActiveBuild(g, &w) == MSX_RESIDENT_ERR_ARGUMENT);
    OK(MSXresidentGpu_getTransferStats(g, &after) == MSX_RESIDENT_OK &&
       after.hydH2DCalls == before.hydH2DCalls && after.h2dCalls == before.h2dCalls);
    OK(MSXresidentGpu_beginActive(g, 0, 35, &w) == MSX_RESIDENT_OK &&
       MSXresidentGpu_sealActive(g, &w, 35) == MSX_RESIDENT_OK);
    MSXresidentGpu_close(g);
}
static MSXResidentGpu *openStreamFixture(void);
static int captureSealedWriter(MSXResidentGpu *g,
                               MSXResidentGpuActiveWriter *w,
                               uint32_t count,
                               uint32_t *pipeOut,
                               uint32_t *rowOut,
                               double *volOut)
{
    double hyd[(size_t)2 * MSX_RESIDENT_HYD_STRIDE] = {0};
    MSXResidentHydView h = {hyd, 1, MSX_RESIDENT_HYD_STRIDE,
                            MSX_RESIDENT_HYD_PIPE_MAJOR};
    MSXResidentGpuDeviceView v;
    MSXResidentGpuReactResult r;
    if (MSXresidentGpu_prepareSealedActiveHyd(g, w, &h, &v, &r) !=
        MSX_RESIDENT_OK || v.itemCount != count)
        return 0;
    if (count &&
        (vcopy(&v, pipeOut, (const void *)(uintptr_t)v.segPipe,
               (size_t)count * sizeof(*pipeOut), cudaMemcpyDeviceToHost) !=
             cudaSuccess ||
         vcopy(&v, rowOut, (const void *)(uintptr_t)v.segRow,
               (size_t)count * sizeof(*rowOut), cudaMemcpyDeviceToHost) !=
             cudaSuccess ||
         vcopy(&v, volOut, (const void *)(uintptr_t)v.segVol,
               (size_t)count * sizeof(*volOut), cudaMemcpyDeviceToHost) !=
             cudaSuccess))
        return 0;
    return MSXresidentGpu_finishActive(g, &r) == MSX_RESIDENT_OK;
}

/* S06b batch writer contract.  The batch path must produce the same pinned
   active payload as the legacy one-row path, while preserving seen state
   across a 256/1 split and clearing it on every failed batch. */
static void t_batch_writer_contract(void)
{
    enum { CAP = 300, ACTIVE = 257 };
    static MSXResidentActiveRow rows[ACTIVE];
    uint32_t i, accepted, pipeBatch[ACTIVE], rowBatch[ACTIVE];
    uint32_t pipeOracle[ACTIVE], rowOracle[ACTIVE];
    double volBatch[ACTIVE], volOracle[ACTIVE];
    MSXResidentGpu *batchGpu = openStreamFixture();
    MSXResidentGpu *oracleGpu = openStreamFixture();
    MSXResidentGpuActiveWriter batchWriter, oracleWriter;
    MSXResidentGpuTransferStats before, after;
    MSXResidentStatus z;
    MSXResidentActiveRow duplicateRows[2];

    (void)CAP;
    for (i = 0; i < ACTIVE; ++i)
        rows[i] = (MSXResidentActiveRow){1, i, i, 1, 0, ACTIVE, 1,
                                         11, UINT64_C(5000) + i, 1.0};

    OK(MSXresidentGpu_beginActive(batchGpu, ACTIVE, 41, &batchWriter) ==
       MSX_RESIDENT_OK);
    OK(MSXresidentGpu_appendActiveBatch(batchGpu, &batchWriter, rows, 256,
                                         &accepted) == MSX_RESIDENT_OK &&
       accepted == 256 && batchWriter.count == 256);
    OK(MSXresidentGpu_appendActiveBatch(batchGpu, &batchWriter, rows + 256, 1,
                                         &accepted) == MSX_RESIDENT_OK &&
       accepted == 1 && batchWriter.count == ACTIVE);
    OK(MSXresidentGpu_sealActive(batchGpu, &batchWriter, 41) ==
       MSX_RESIDENT_OK && batchWriter.sealed);

    OK(MSXresidentGpu_beginActive(oracleGpu, ACTIVE, 41, &oracleWriter) ==
       MSX_RESIDENT_OK);
    for (i = 0; i < ACTIVE; ++i)
        OK(MSXresidentGpu_appendActive(oracleGpu, &oracleWriter, &rows[i]) ==
           MSX_RESIDENT_OK);
    OK(MSXresidentGpu_sealActive(oracleGpu, &oracleWriter, 41) ==
       MSX_RESIDENT_OK && oracleWriter.sealed);
    OK(captureSealedWriter(batchGpu, &batchWriter, ACTIVE,
                           pipeBatch, rowBatch, volBatch));
    OK(captureSealedWriter(oracleGpu, &oracleWriter, ACTIVE,
                           pipeOracle, rowOracle, volOracle));
    OK(!memcmp(pipeBatch, pipeOracle, sizeof(pipeBatch)) &&
       !memcmp(rowBatch, rowOracle, sizeof(rowBatch)) &&
       !memcmp(volBatch, volOracle, sizeof(volBatch)));
    MSXresidentGpu_close(batchGpu);
    MSXresidentGpu_close(oracleGpu);

    /* A zero-capacity batch is a valid no-op, while NULL with a non-zero
       capacity is a fail-closed argument error and must not transfer. */
    batchGpu = openStreamFixture();
    accepted = UINT32_MAX;
    OK(MSXresidentGpu_beginActive(batchGpu, 0, 40, &batchWriter) ==
       MSX_RESIDENT_OK &&
       MSXresidentGpu_appendActiveBatch(batchGpu, &batchWriter, NULL, 0,
                                         &accepted) == MSX_RESIDENT_OK &&
       accepted == 0 &&
       MSXresidentGpu_sealActive(batchGpu, &batchWriter, 40) ==
           MSX_RESIDENT_OK);
    MSXresidentGpu_close(batchGpu);
    batchGpu = openStreamFixture();
    OK(MSXresidentGpu_beginActive(batchGpu, ACTIVE, 40, &batchWriter) ==
       MSX_RESIDENT_OK);
    OK(MSXresidentGpu_getTransferStats(batchGpu, &before) == MSX_RESIDENT_OK);
    OK(MSXresidentGpu_appendActiveBatch(batchGpu, &batchWriter, NULL, 1,
                                         &accepted) ==
           MSX_RESIDENT_ERR_ARGUMENT && accepted == 0 && !batchWriter.owner);
    OK(MSXresidentGpu_getTransferStats(batchGpu, &after) == MSX_RESIDENT_OK &&
       sameTransfer(&before, &after));
    MSXresidentGpu_close(batchGpu);

    /* A duplicate split across batches clears the whole writer and reports
       the valid prefix accepted by the failed batch. */
    batchGpu = openStreamFixture();
    duplicateRows[0] = rows[256];
    duplicateRows[1] = rows[0];
    OK(MSXresidentGpu_beginActive(batchGpu, 258, 42, &batchWriter) ==
       MSX_RESIDENT_OK);
    OK(MSXresidentGpu_appendActiveBatch(batchGpu, &batchWriter, rows, 256,
                                         &accepted) == MSX_RESIDENT_OK);
    OK(MSXresidentGpu_appendActiveBatch(batchGpu, &batchWriter, duplicateRows, 2,
                                         &accepted) == MSX_RESIDENT_ERR_ARGUMENT &&
       accepted == 1 && !batchWriter.owner && batchWriter.count == 0);
    OK(MSXresidentGpu_beginActive(batchGpu, 0, 43, &batchWriter) ==
       MSX_RESIDENT_OK &&
       MSXresidentGpu_sealActive(batchGpu, &batchWriter, 43) ==
       MSX_RESIDENT_OK);
    MSXresidentGpu_close(batchGpu);

    /* Cross-pipe duplicate: a valid pipe-2 row follows pipe 1, then the same
       pipe-2 row is repeated in a later batch. */
    {
        MSXResidentGpu *g = initial4();
        MSXResidentActiveRow cross[2] = {
            {1, 0, 0, 1, 0, 2, 1, 5, 101, 2.0},
            {2, 1, 3, 1, 0, 2, -1, 7, 202, 5.0}};
        OK(MSXresidentGpu_beginActive(g, 3, 44, &batchWriter) ==
           MSX_RESIDENT_OK);
        OK(MSXresidentGpu_appendActiveBatch(g, &batchWriter, cross, 2,
                                            &accepted) == MSX_RESIDENT_OK &&
           accepted == 2);
        OK(MSXresidentGpu_appendActiveBatch(g, &batchWriter, &cross[1], 1,
                                            &accepted) ==
               MSX_RESIDENT_ERR_ARGUMENT && accepted == 0 &&
           !batchWriter.owner && batchWriter.count == 0);
        OK(MSXresidentGpu_beginActive(g, 0, 45, &batchWriter) ==
           MSX_RESIDENT_OK &&
           MSXresidentGpu_sealActive(g, &batchWriter, 45) ==
               MSX_RESIDENT_OK);
        MSXresidentGpu_close(g);
    }

    /* Each row-level validation failure is fail-closed and does not add a
       transfer.  A fresh begin proves activeSeen/touchedRows were cleared. */
    batchGpu = openStreamFixture();
    OK(MSXresidentGpu_getTransferStats(batchGpu, &before) == MSX_RESIDENT_OK);
    rows[0].generation = 9;
    z = MSXresidentGpu_beginActive(batchGpu, ACTIVE, 46, &batchWriter);
    OK(z == MSX_RESIDENT_OK);
    OK(MSXresidentGpu_appendActiveBatch(batchGpu, &batchWriter, rows, 1,
                                         &accepted) == MSX_RESIDENT_ERR_GENERATION &&
       accepted == 0 && !batchWriter.owner);
    rows[0].generation = 1;
    rows[0].descriptorEpoch = 12;
    OK(MSXresidentGpu_beginActive(batchGpu, ACTIVE, 47, &batchWriter) ==
       MSX_RESIDENT_OK);
    OK(MSXresidentGpu_appendActiveBatch(batchGpu, &batchWriter, rows, 1,
                                         &accepted) == MSX_RESIDENT_ERR_GENERATION &&
       accepted == 0 && !batchWriter.owner);
    rows[0].descriptorEpoch = 11;
    rows[0].volume = NAN;
    OK(MSXresidentGpu_beginActive(batchGpu, ACTIVE, 48, &batchWriter) ==
       MSX_RESIDENT_OK);
    OK(MSXresidentGpu_appendActiveBatch(batchGpu, &batchWriter, rows, 1,
                                         &accepted) == MSX_RESIDENT_ERR_ARGUMENT &&
       accepted == 0 && !batchWriter.owner);
    rows[0].volume = 1.0;
    rows[0].globalRow = 1;
    OK(MSXresidentGpu_beginActive(batchGpu, ACTIVE, 49, &batchWriter) ==
       MSX_RESIDENT_OK);
    OK(MSXresidentGpu_appendActiveBatch(batchGpu, &batchWriter, rows, 1,
                                         &accepted) == MSX_RESIDENT_ERR_CAPACITY &&
       accepted == 0 && !batchWriter.owner);
    rows[0].globalRow = 0;
    rows[0].slot = CAP;
    OK(MSXresidentGpu_beginActive(batchGpu, ACTIVE, 50, &batchWriter) ==
       MSX_RESIDENT_OK);
    OK(MSXresidentGpu_appendActiveBatch(batchGpu, &batchWriter, rows, 1,
                                         &accepted) == MSX_RESIDENT_ERR_CAPACITY &&
       accepted == 0 && !batchWriter.owner);
    rows[0].slot = 0;
    OK(MSXresidentGpu_getTransferStats(batchGpu, &after) == MSX_RESIDENT_OK &&
       sameTransfer(&before, &after));
    /* An unused mirror slot must fail generation even when its generation
       field is made to look current.  A used but non-member slot must take
       the same fail-closed path through activeMember. */
    rows[0].slot = 257;
    rows[0].globalRow = 257;
    rows[0].generation = 1;
    OK(MSXresidentGpu_beginActive(batchGpu, ACTIVE, 52, &batchWriter) ==
       MSX_RESIDENT_OK);
    OK(MSXresidentGpu_appendActiveBatch(batchGpu, &batchWriter, rows, 1,
                                         &accepted) == MSX_RESIDENT_ERR_GENERATION &&
       accepted == 0 && !batchWriter.owner);
    rows[0].slot = 0;
    rows[0].globalRow = 0;
    rows[0].generation = 1;
    /* Restore the original row before exercising expectedCount exhaustion on
       a fresh mirror. */
    rows[0].slot = 0;
    rows[0].globalRow = 0;
    rows[0].descriptorEpoch = 11;
    /* The previous descriptor fixture is intentionally closed before the
       expected-count case, so no stale writer or descriptor state is reused. */
    MSXresidentGpu_close(batchGpu);
    batchGpu = openStreamFixture();
    OK(MSXresidentGpu_beginActive(batchGpu, 1, 54, &batchWriter) ==
       MSX_RESIDENT_OK);
    OK(MSXresidentGpu_appendActiveBatch(batchGpu, &batchWriter, rows, 1,
                                         &accepted) == MSX_RESIDENT_OK &&
       accepted == 1 && batchWriter.count == 1);
    OK(MSXresidentGpu_appendActiveBatch(batchGpu, &batchWriter, rows, 1,
                                         &accepted) == MSX_RESIDENT_ERR_CAPACITY &&
       accepted == 0 && !batchWriter.owner && batchWriter.count == 0);
    OK(MSXresidentGpu_beginActive(batchGpu, 0, 55, &batchWriter) ==
       MSX_RESIDENT_OK &&
       MSXresidentGpu_sealActive(batchGpu, &batchWriter, 55) ==
           MSX_RESIDENT_OK);
    MSXresidentGpu_close(batchGpu);
}
typedef struct {
    MSXResidentActiveRow row[301];
    uint32_t count, emitCount, position, nextCalls, identityCalls, auditCalls;
    int nextFailAt, identityFailAt, duplicateAt, auditEvery, topologyChangeAt;
    MSXResidentStatus nextStatus, rawStatus, beginStatus, validateStatus,
                      countStatus;
    uint64_t topologyVersion;
} StreamSourceTest;

static void streamSourceReset(StreamSourceTest *s)
{
    uint32_t i;
    memset(s, 0, sizeof(*s));
    s->count = s->emitCount = 257;
    s->nextFailAt = s->identityFailAt = s->duplicateAt = -1;
    s->topologyChangeAt = -1;
    s->nextStatus = MSX_RESIDENT_ERR_OVERFLOW;
    s->rawStatus = s->beginStatus = s->validateStatus = s->countStatus =
        MSX_RESIDENT_OK;
    s->topologyVersion = 30;
    for (i = 0; i < s->count; ++i)
    {
        s->row[i] = (MSXResidentActiveRow){1, i, i, 1, 0, 257, 1,
                                           11, UINT64_C(5000) + i, 1.0};
    }
}
static MSXResidentStatus streamSourceBegin(void *p)
{ StreamSourceTest *s = (StreamSourceTest *)p; s->position = 0; return s->beginStatus; }
static MSXResidentStatus streamSourceValidate(void *p)
{ return ((StreamSourceTest *)p)->validateStatus; }
static MSXResidentStatus streamSourceCount(void *p, uint32_t *n)
{ StreamSourceTest *s = (StreamSourceTest *)p; *n = s->count; return s->countStatus; }
static MSXResidentStatus streamSourceRawCount(void *p, uint64_t *n,
                                              uint32_t *links)
{
    StreamSourceTest *s = (StreamSourceTest *)p;
    if (n) *n = s->count;
    if (links) *links = 1;
    return s->rawStatus;
}
static MSXResidentStatus streamSourceNext(void *p, MSXResidentActiveRow *row)
{
    StreamSourceTest *s = (StreamSourceTest *)p;
    uint32_t i = s->position++;
    ++s->nextCalls;
    if (s->nextFailAt >= 0 && (int)i == s->nextFailAt) return s->nextStatus;
    if (i >= s->emitCount) return MSX_RESIDENT_ITER_END;
    *row = s->row[i];
    if (s->duplicateAt >= 0 && (int)i == s->duplicateAt) *row = s->row[0];
    return MSX_RESIDENT_OK;
}
static MSXResidentStatus streamSourceNextBatch(void *p,
                                               MSXResidentActiveRow *rows,
                                               uint32_t cap, uint32_t *count)
{
    StreamSourceTest *s = (StreamSourceTest *)p;
    MSXResidentStatus z;
    uint32_t n = 0;
    if (!rows || !count || !cap) return MSX_RESIDENT_ERR_ARGUMENT;
    while (n < cap)
    {
        z = streamSourceNext(p, &rows[n]);
        if (z == MSX_RESIDENT_OK) { ++n; continue; }
        *count = n;
        return z;
    }
    *count = n;
    return MSX_RESIDENT_OK;
}
static int streamSourceIdentity(void *p, const MSXResidentActiveRow *row)
{
    StreamSourceTest *s = (StreamSourceTest *)p;
    (void)row;
    return s->identityFailAt < 0 || (int)s->identityCalls++ != s->identityFailAt;
}
static int streamSourceAudit(void *p, const MSXResidentActiveRow *row)
{
    StreamSourceTest *s = (StreamSourceTest *)p;
    (void)row;
    ++s->auditCalls;
    return s->auditEvery > 0 && (s->auditCalls - 1) % s->auditEvery == 0;
}
static uint64_t streamSourceTopology(void *p)
{
    StreamSourceTest *s = (StreamSourceTest *)p;
    return s->topologyChangeAt >= 0 &&
           (int)s->nextCalls >= s->topologyChangeAt ? s->topologyVersion + 1 :
                                                        s->topologyVersion;
}
static MSXResidentActiveStreamSource streamSource(StreamSourceTest *s, int audit)
{
    MSXResidentActiveStreamSource source;
    memset(&source, 0, sizeof(source));
    source.context = s;
    source.begin = streamSourceBegin;
    source.validate = streamSourceValidate;
    source.count = streamSourceCount;
    source.rawCount = streamSourceRawCount;
    source.next = streamSourceNext;
    source.nextBatch = streamSourceNextBatch;
    source.identity = streamSourceIdentity;
    source.audit = audit ? streamSourceAudit : NULL;
    source.topology = streamSourceTopology;
    return source;
}
static MSXResidentGpu *openStreamFixture(void)
{
    static MSXResidentSlotPatch slots[300];
    uint32_t cap[2] = {0, 300}, base[2] = {0, 0}, i;
    MSXResidentGpuOpen open = {1, 300, S, cap, base};
    MSXResidentDescriptorPatch d;
    MSXResidentPatchBatch batch;
    MSXResidentGpu *g = 0;
    desc(&d, 1, 300, 257, 11, 1);
    for (i = 0; i < 300; ++i)
    {
        if (i < 257) slot(&slots[i], 1, i, 1, UINT64_C(5000) + i, 1.0, i);
        else { memset(&slots[i], 0, sizeof(slots[i])); slots[i].linkIndex = 1; slots[i].slot = i; }
    }
    batch = (MSXResidentPatchBatch){&d, 1, slots, 300};
    OK(MSXresidentGpu_open(&open, &g) == MSX_RESIDENT_OK && g &&
       MSXresidentGpu_initialUpload(g, &batch) == MSX_RESIDENT_OK);
    return g;
}
static int sameTransfer(const MSXResidentGpuTransferStats *a,
                        const MSXResidentGpuTransferStats *b)
{
    return a->h2dCalls == b->h2dCalls && a->h2dBytes == b->h2dBytes &&
        a->d2hCalls == b->d2hCalls && a->d2hBytes == b->d2hBytes &&
        a->hydH2DCalls == b->hydH2DCalls && a->hydH2DBytes == b->hydH2DBytes &&
        a->hydApiCalls == b->hydApiCalls;
}
/* The shared production stream helper is exercised through injected source
   callbacks.  This covers the real writer lifecycle without duplicating its
   append/seal/prepare algorithm in the harness. */
static void t_runtime_stream_contract(void)
{
    StreamSourceTest sourceState;
    MSXResidentActiveStreamSource source;
    MSXResidentActiveStreamReport report;
    MSXResidentGpu *g = openStreamFixture();
    MSXResidentGpuActiveWriter w;
    MSXResidentGpuTransferStats before, after;
    MSXResidentGpuDeviceView view;
    MSXResidentGpuReactResult result;
    MSXResidentHydView hyd;
    double pipe[(size_t)2 * MSX_RESIDENT_HYD_STRIDE] = {0};
    MSXResidentHydView badHyd;
    uint32_t active;

    hyd = (MSXResidentHydView){pipe, 1, MSX_RESIDENT_HYD_STRIDE,
                               MSX_RESIDENT_HYD_PIPE_MAJOR};
    streamSourceReset(&sourceState);
    source = streamSource(&sourceState, 0);
    memset(&w, 0, sizeof(w));
    OK(MSXresidentGpu_getTransferStats(g, &before) == MSX_RESIDENT_OK);
    OK(MSXresident_activeStreamBuild(g, &w, 30, &source, &hyd, &active,
                                     &report) == MSX_RESIDENT_OK);
    OK(active == 257 && report.expectedCount == 257 &&
       report.iteratorPasses == 1 && report.rawCountPasses == 1 &&
       report.rawLinks == 1 && report.rowsAppended == 257 &&
       report.chunkHighWater == 256 && report.builderAborts == 0 && w.sealed);
    OK(MSXresidentGpu_prepareSealedActiveHyd(g, &w, &hyd, &view, &result) ==
       MSX_RESIDENT_OK && view.itemCount == 257 &&
       MSXresidentGpu_finishActive(g, &result) == MSX_RESIDENT_OK);
    OK(MSXresidentGpu_getTransferStats(g, &after) == MSX_RESIDENT_OK &&
       after.hydH2DCalls == before.hydH2DCalls + 1);

    /* Invalid prepare is a pre-submit failure and clears the sealed writer. */
    streamSourceReset(&sourceState); source = streamSource(&sourceState, 0);
    memset(&w, 0, sizeof(w)); badHyd = hyd; badHyd.hydLayout = MSX_RESIDENT_HYD_ACTIVE_MAJOR;
    OK(MSXresident_activeStreamBuild(g, &w, 30, &source, &hyd, &active, &report) == 0);
    OK(MSXresidentGpu_getTransferStats(g, &before) == MSX_RESIDENT_OK &&
       MSXresidentGpu_prepareSealedActiveHyd(g, &w, &badHyd, &view, &result) ==
       MSX_RESIDENT_ERR_ARGUMENT && !w.owner &&
       MSXresidentGpu_getTransferStats(g, &after) == MSX_RESIDENT_OK &&
       sameTransfer(&before, &after));

    /* A next() error after an append/duplicate error keeps next-error
       priority, but the already-reset writer is not aborted twice. */
    streamSourceReset(&sourceState); sourceState.duplicateAt = 1;
    sourceState.nextFailAt = 257; source = streamSource(&sourceState, 0);
    memset(&w, 0, sizeof(w));
    OK(MSXresidentGpu_getTransferStats(g, &before) == MSX_RESIDENT_OK);
    OK(MSXresident_activeStreamBuild(g, &w, 30, &source, &hyd, &active, &report) ==
       MSX_RESIDENT_ERR_OVERFLOW && report.builderAborts == 1 &&
       sourceState.nextCalls == 258 && !w.owner);
    OK(MSXresidentGpu_getTransferStats(g, &after) == MSX_RESIDENT_OK &&
       sameTransfer(&before, &after));

    streamSourceReset(&sourceState); sourceState.identityFailAt = 256;
    source = streamSource(&sourceState, 0); memset(&w, 0, sizeof(w));
    OK(MSXresidentGpu_getTransferStats(g, &before) == MSX_RESIDENT_OK);
    OK(MSXresident_activeStreamBuild(g, &w, 30, &source, &hyd, &active, &report) ==
       MSX_RESIDENT_ERR_GENERATION && report.builderAborts == 1 &&
       report.rowsAppended == 256 && sourceState.nextCalls == 258 && !w.owner);
    OK(MSXresidentGpu_getTransferStats(g, &after) == MSX_RESIDENT_OK &&
       sameTransfer(&before, &after));

    /* Identity must win when the same row is also malformed for append;
       an earlier append failure wins over a later identity failure. */
    streamSourceReset(&sourceState);
    sourceState.identityFailAt = 0;
    sourceState.row[0].volume = NAN;
    source = streamSource(&sourceState, 0); memset(&w, 0, sizeof(w));
    OK(MSXresidentGpu_getTransferStats(g, &before) == MSX_RESIDENT_OK);
    OK(MSXresident_activeStreamBuild(g, &w, 30, &source, &hyd, &active,
                                     &report) == MSX_RESIDENT_ERR_GENERATION &&
       report.appendFailures == 0 && report.builderAborts == 1 && !w.owner);
    OK(MSXresidentGpu_getTransferStats(g, &after) == MSX_RESIDENT_OK &&
       sameTransfer(&before, &after));
    streamSourceReset(&sourceState);
    sourceState.identityFailAt = 1;
    sourceState.row[0].volume = NAN;
    source = streamSource(&sourceState, 0); memset(&w, 0, sizeof(w));
    OK(MSXresident_activeStreamBuild(g, &w, 30, &source, &hyd, &active,
                                     &report) == MSX_RESIDENT_ERR_ARGUMENT &&
       report.appendFailures == 1 && report.builderAborts == 1 && !w.owner);
    OK(MSXresidentGpu_getTransferStats(g, &after) == MSX_RESIDENT_OK &&
       sameTransfer(&before, &after));

    /* A raw-count failure is deferred until the single iterator pass has
       completed; no writer or GPU transfer may be created. */
    streamSourceReset(&sourceState); sourceState.rawStatus =
        MSX_RESIDENT_ERR_CAPACITY;
    source = streamSource(&sourceState, 0); memset(&w, 0, sizeof(w));
    OK(MSXresidentGpu_getTransferStats(g, &before) == MSX_RESIDENT_OK);
    OK(MSXresident_activeStreamBuild(g, &w, 30, &source, &hyd, &active,
                                     &report) == MSX_RESIDENT_ERR_CAPACITY &&
       report.rawCountPasses == 1 && report.iteratorPasses == 1 &&
       report.iteratorFailures == 0 && sourceState.nextCalls > 0 &&
       !w.owner);
    OK(MSXresidentGpu_getTransferStats(g, &after) == MSX_RESIDENT_OK &&
       sameTransfer(&before, &after));

    /* The audit continues after its first mismatch; every even row is
       counted, while append/submit remains stopped after row zero. */
    streamSourceReset(&sourceState); sourceState.auditEvery = 2;
    source = streamSource(&sourceState, 1); memset(&w, 0, sizeof(w));
    OK(MSXresidentGpu_getTransferStats(g, &before) == MSX_RESIDENT_OK);
    OK(MSXresident_activeStreamBuild(g, &w, 30, &source, &hyd, &active, &report) ==
       MSX_RESIDENT_ERR_ARGUMENT && report.expectedCount == 257 &&
       report.hydMismatches == 129 && sourceState.auditCalls == 257 &&
       sourceState.nextCalls == 258 && report.builderAborts == 1 && !w.owner);
    OK(MSXresidentGpu_getTransferStats(g, &after) == MSX_RESIDENT_OK &&
       sameTransfer(&before, &after));

    streamSourceReset(&sourceState); sourceState.emitCount = 256;
    source = streamSource(&sourceState, 0); memset(&w, 0, sizeof(w));
    OK(MSXresident_activeStreamBuild(g, &w, 30, &source, &hyd, &active, &report) ==
       MSX_RESIDENT_ERR_CAPACITY && report.builderAborts == 1 && !w.owner);

    streamSourceReset(&sourceState); sourceState.topologyChangeAt = 258;
    source = streamSource(&sourceState, 0); memset(&w, 0, sizeof(w));
    OK(MSXresident_activeStreamBuild(g, &w, 30, &source, &hyd, &active, &report) ==
       MSX_RESIDENT_ERR_GENERATION && report.builderAborts == 1 && !w.owner);

    streamSourceReset(&sourceState); sourceState.count = 301;
    source = streamSource(&sourceState, 0); memset(&w, 0, sizeof(w));
    OK(MSXresidentGpu_getTransferStats(g, &before) == MSX_RESIDENT_OK);
    OK(MSXresident_activeStreamBuild(g, &w, 30, &source, &hyd, &active, &report) ==
       MSX_RESIDENT_ERR_CAPACITY && report.expectedCount == 301 &&
       report.beginFailures == 1 && sourceState.nextCalls > 0);
    OK(MSXresidentGpu_getTransferStats(g, &after) == MSX_RESIDENT_OK &&
       sameTransfer(&before, &after));
    MSXresidentGpu_close(g);
}
/* P1 typed patches: META leaves the GPU concentration vectors untouched,
   INVALIDATE carries no concentration payload, and a later IMPORT reuses the
   slot with a new generation.  The descriptor patch is intentionally sparse. */
static void t_typed_patches(void) { MSXResidentGpu*g=initial4();MSXResidentGpuReduction a,z;MSXResidentDescriptorPatch d,multi[2];MSXResidentSlotPatch x,typed[2];MSXResidentPatchBatch b;double m[S],n[S];slot(&x,1,0,1,101,2,90);x.kind=MSX_RESIDENT_PATCH_META;x.payload.c=0;x.payload.lastc=0;b=(MSXResidentPatchBatch){0,0,&x,1};OK(snap(g,&a,m)&&MSXresidentGpu_applyPatches(g,&b)==0&&snap(g,&z,n)&&same(&a,&z,m,n));desc(&d,1,2,1,6,1);d.descriptor.head=1;d.descriptor.tail=1;memset(&x,0,sizeof(x));x.linkIndex=1;x.slot=0;x.generation=1;x.used=0;x.kind=MSX_RESIDENT_PATCH_INVALIDATE;x.payload.c=0;x.payload.lastc=0;b=(MSXResidentPatchBatch){&d,1,&x,1};OK(MSXresidentGpu_applyPatches(g,&b)==0&&snap(g,&z,n)&&z.activeCount==3);slot(&x,1,0,2,301,2,55);x.kind=MSX_RESIDENT_PATCH_IMPORT;desc(&d,1,2,2,7,1);b=(MSXResidentPatchBatch){&d,1,&x,1};OK(MSXresidentGpu_applyPatches(g,&b)==0&&snap(g,&z,n)&&z.activeCount==4);
/* One sparse batch may carry multiple descriptors and mixed typed events.  META
   must preserve both GPU payloads; INVALIDATE carries no concentration arrays. */
desc(&multi[0],1,2,2,8,1);desc(&multi[1],2,2,2,8,-1);slot(&typed[0],1,1,1,102,3,20);typed[0].kind=MSX_RESIDENT_PATCH_META;slot(&typed[1],2,0,1,201,4,30);typed[1].kind=MSX_RESIDENT_PATCH_META;b=(MSXResidentPatchBatch){multi,2,typed,2};OK(snap(g,&a,m)&&MSXresidentGpu_applyPatches(g,&b)==0&&snap(g,&z,n)&&same(&a,&z,m,n));desc(&multi[0],1,2,2,9,1);desc(&multi[1],2,2,1,9,-1);multi[1].descriptor.head=1;multi[1].descriptor.tail=1;slot(&typed[0],1,1,1,102,3,20);typed[0].kind=MSX_RESIDENT_PATCH_META;memset(&typed[1],0,sizeof(typed[1]));typed[1].linkIndex=2;typed[1].slot=0;typed[1].generation=1;typed[1].kind=MSX_RESIDENT_PATCH_INVALIDATE;b=(MSXResidentPatchBatch){multi,2,typed,2};OK(MSXresidentGpu_applyPatches(g,&b)==0&&snap(g,&z,n)&&z.activeCount==3);MSXresidentGpu_close(g);}
static void t_active_sync(void) { MSXResidentGpu*g=initial4();MSXResidentGpuReduction z;double mass[S],hyd[MSX_RESIDENT_HYD_STRIDE]={0},c[S],l[S],h=77;MSXResidentActiveItem a={1,0,1,5,2,hyd};MSXResidentActiveBatch b={&a,1};MSXResidentGpuDeviceView v;MSXResidentGpuReactResult r;MSXResidentGpuActiveSyncRow row;MSXResidentGpuActiveSyncOutput o={&row,c,l,S};MSXResidentDescriptorPatch d;MSXResidentPatchBatch q;OK(MSXresidentGpu_prepareActive(g,&b,&v,&r)==0);c[0]=321;l[0]=654;OK(vcopy(&v,(void*)(uintptr_t)v.c,c,sizeof(c),cudaMemcpyHostToDevice)==cudaSuccess);OK(vcopy(&v,(void*)(uintptr_t)v.lastc,l,sizeof(l),cudaMemcpyHostToDevice)==cudaSuccess);OK(vcopy(&v,(void*)(uintptr_t)v.hstep,&h,sizeof(h),cudaMemcpyHostToDevice)==cudaSuccess);OK(MSXresidentGpu_finishActive(g,&r)==0);OK(MSXresidentGpu_reduce(g,mass,S,&z)==0&&mass[0]==321*2+20*3+30*4+40*5);memset(c,0,sizeof(c));memset(l,0,sizeof(l));OK(MSXresidentGpu_syncActive(g,&o,1)==0&&row.linkIndex==1&&row.globalRow==0&&row.generation==1&&row.descriptorEpoch==5&&row.hstep==77&&c[0]==321&&l[0]==654);OK(MSXresidentGpu_prepareActive(g,&b,&v,&r)==0&&MSXresidentGpu_finishActive(g,&r)==0);desc(&d,1,2,2,6,1);q=(MSXResidentPatchBatch){&d,1,0,0};OK(MSXresidentGpu_applyPatches(g,&q)==0);OK(MSXresidentGpu_syncActive(g,&o,1)==MSX_RESIDENT_ERR_GENERATION);MSXresidentGpu_close(g);}
int main(void) { struct cudaDeviceProp p;int n=0;cudaError_t e=cudaGetDeviceCount(&n);printf("cuda_required=true\n");if(e!=cudaSuccess||n<1){printf("gpu_name=unavailable\nassertions_passed=0\nassertions_failed=1\n");return 2;}cudaGetDeviceProperties(&p,0);printf("gpu_name=%s\n",p.name);OK(MSXresidentGpu_isEnabled()==1);t_open();t_initial();t_scatter();t_descriptor_generation();t_fetch();t_fetch_batch_contract();t_reduce_lifecycle();t_hole_initial_fetch();t_ring_span_reject();t_wrap_active();t_stream_builder();t_stream_writer_large();t_batch_writer_contract();t_runtime_stream_contract();t_active_view();t_pipe_hyd_contract();t_pipe_hyd_exact_bits();t_pipe_hyd_failure_invalidates();t_active_sync();t_active_error();t_active_empty();t_active_abort_and_query_guard();t_explicit_poison_query_guard();t_large_batch();t_typed_patches();printf("assertions_passed=%d\nassertions_failed=%d\n",pass,fail);return fail?1:0; }
