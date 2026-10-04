/* Full logical initial-state diagnostic. This file is deliberately present
   only in the isolated M7 source copy. No export exists in ordinary builds.
   Bounded CRT scratch belongs to this test exporter, not production planning. */
#ifdef MSX_RESIDENT_TEST_API
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "msxtypes.h"
#include "msxqual_shared.h"
#include "msxresident_initial_golden.h"
#include "msxresident_runtime.h"
#include "msxresident_capacity.h"

extern MSXproject MSX;
#if defined(_WIN32)
#define GOLDEN_EXPORT __declspec(dllexport)
#else
#define GOLDEN_EXPORT __attribute__((visibility("default")))
#endif

static int GoldenArmed, GoldenReady;
static uint64_t GoldenVersion, GoldenSequence;

GOLDEN_EXPORT int MSXTESTinitialArm(void)
{
    if(MSX.Qtime!=0 || MSX.ErrCode)return 1;
    GoldenArmed=1;GoldenReady=0;return 0;
}

int MSXinitialGolden_checkpoint(void)
{
    if(!GoldenArmed)return 0;
    if(MSX.Qtime!=0 || MSX.ErrCode)return 0;
    GoldenArmed=0;GoldenReady=1;
    GoldenVersion=MSXresidentRuntime_stateVersion();++GoldenSequence;
    return 1;
}

GOLDEN_EXPORT int MSXTESTinitialStatus(uint64_t *values)
{
    if(!values)return 1;
    values[0]=GoldenReady;values[1]=GoldenSequence;
    values[2]=(uint64_t)MSX.Qtime;values[3]=GoldenVersion;
    values[4]=MSXresidentRuntime_stateVersion();values[5]=MSX.ErrCode;
    return 0;
}

typedef struct {
    uint32_t entityKind,entityIndex,ordinal,owner,generation;
    int32_t ownerLink;
    uint64_t parcelId;
    double scalar[5];
} GoldenRow;

typedef struct {
    uint32_t capacity,count,gpuCount,stride;
    GoldenRow *rows;
    double *c,*lastc;
    MSXResidentHandoffItem *item;
    MSXResidentHandoffResult *result;
    MSXResidentHandoffTarget *target;
    uint32_t *gpuRow;
} GoldenBatch;

static int put(FILE *f,const void *p,size_t n)
{return fwrite(p,1,n,f)==n;}
static int put32(FILE *f,uint32_t v){return put(f,&v,sizeof(v));}
static int put64(FILE *f,uint64_t v){return put(f,&v,sizeof(v));}

static void batchFree(GoldenBatch *b)
{
    free(b->rows);free(b->c);free(b->lastc);free(b->item);
    free(b->result);free(b->target);free(b->gpuRow);memset(b,0,sizeof(*b));
}

static int batchOpen(GoldenBatch *b,uint32_t capacity,uint32_t stride)
{
    size_t rows;
    memset(b,0,sizeof(*b));
    if(!capacity||!stride||capacity>SIZE_MAX/stride)return 0;
    rows=(size_t)capacity*stride;
    if(rows>SIZE_MAX/sizeof(double))return 0;
    b->capacity=capacity;b->stride=stride;
    b->rows=(GoldenRow*)calloc(capacity,sizeof(*b->rows));
    b->c=(double*)calloc(rows,sizeof(double));b->lastc=(double*)calloc(rows,sizeof(double));
    b->item=(MSXResidentHandoffItem*)calloc(capacity,sizeof(*b->item));
    b->result=(MSXResidentHandoffResult*)calloc(capacity,sizeof(*b->result));
    b->target=(MSXResidentHandoffTarget*)calloc(capacity,sizeof(*b->target));
    b->gpuRow=(uint32_t*)calloc(capacity,sizeof(*b->gpuRow));
    if(!b->rows||!b->c||!b->lastc||!b->item||!b->result||!b->target||!b->gpuRow){batchFree(b);return 0;}
    return 1;
}

static int batchFlush(FILE *f,GoldenBatch *b)
{
    uint32_t i,m;
    if(b->gpuCount){
        MSXResidentStatus z=MSXresidentRuntime_fetchTargets(b->item,b->gpuCount,b->result,b->target);
        if(z!=MSX_RESIDENT_OK)return 0;
        for(i=0;i<b->gpuCount;++i){
            GoldenRow *r=&b->rows[b->gpuRow[i]];
            MSXResidentHandoffResult *got=&b->result[i];
            const MSXResidentHandoffItem *wanted=&b->item[i];
            double scalar[5];
            scalar[0]=got->payload.volume;scalar[1]=got->payload.hstep;
            scalar[2]=got->payload.hresponse;scalar[3]=got->payload.uresponse;scalar[4]=got->payload.dresponse;
            if(got->linkIndex!=wanted->linkIndex||got->slot!=wanted->slot||
               got->generation!=wanted->generation||got->pipeEpoch!=wanted->pipeEpoch||
               got->payload.parcelId!=r->parcelId||memcmp(scalar,r->scalar,sizeof(scalar)))return 0;
        }
    }
    for(i=0;i<b->count;++i){
        GoldenRow *r=&b->rows[i];
        if(!put32(f,r->entityKind)||!put32(f,r->entityIndex)||!put32(f,r->ordinal)||
           !put32(f,r->owner)||!put32(f,r->generation)||!put32(f,(uint32_t)r->ownerLink)||
           !put64(f,r->parcelId)||!put(f,r->scalar,sizeof(r->scalar)))return 0;
        for(m=1;m<b->stride;++m)if(!isfinite(b->c[(size_t)i*b->stride+m])||
                                     !isfinite(b->lastc[(size_t)i*b->stride+m]))return 0;
        if(!put(f,b->c+(size_t)i*b->stride+1,(b->stride-1)*sizeof(double))||
           !put(f,b->lastc+(size_t)i*b->stride+1,(b->stride-1)*sizeof(double)))return 0;
    }
    b->count=b->gpuCount=0;return 1;
}

static int batchAppend(GoldenBatch *b,uint32_t kind,uint32_t index,uint32_t ordinal,Pseg seg)
{
    GoldenRow *r;uint32_t row=b->count,m;
    if(!seg||row>=b->capacity)return 0;
    r=&b->rows[row];memset(r,0,sizeof(*r));
    r->entityKind=kind;r->entityIndex=index;r->ordinal=ordinal;
    r->owner=seg->inHybridCore?1:0;r->ownerLink=seg->ownerLink;r->parcelId=seg->hybridId;
    r->scalar[0]=seg->v;r->scalar[1]=seg->hstep;r->scalar[2]=seg->hresponse;
    r->scalar[3]=seg->uresponse;r->scalar[4]=seg->dresponse;
    if(r->owner){
        uint32_t q=b->gpuCount,slot,generation;uint64_t id,epoch;
        if(kind!=0||!MSXresidentRuntime_isResident()||
           MSXresident_getSlotForParcel(index,r->parcelId,&slot,&generation)!=MSX_RESIDENT_OK||
           MSXresident_getSlotIdentity(index,slot,&generation,&id,&epoch)!=MSX_RESIDENT_OK||id!=r->parcelId)return 0;
        r->generation=generation;
        memset(&b->item[q],0,sizeof(b->item[q]));
        b->item[q].linkIndex=index;b->item[q].slot=slot;
        b->item[q].generation=generation;b->item[q].pipeEpoch=epoch;
        b->target[q].c=b->c+(size_t)row*b->stride;
        b->target[q].lastc=b->lastc+(size_t)row*b->stride;
        b->gpuRow[q]=row;++b->gpuCount;
    }else{
        if(!seg->c||!seg->lastc)return 0;
        for(m=1;m<b->stride;++m){b->c[(size_t)row*b->stride+m]=seg->c[m];
                              b->lastc[(size_t)row*b->stride+m]=seg->lastc[m];}
    }
    ++b->count;return 1;
}

/* Encoding: little-endian header, initial mass IEEE64, entity headers then
   canonical FirstSeg->prev rows. Explicit fields avoid struct padding bytes.
   Species index zero is outside the logical model and is not serialized. */
GOLDEN_EXPORT int MSXTESTinitialExport(const char *path)
{
    FILE *f=NULL;GoldenBatch b;MSXResidentLayout layout;MSXResidentMemoryConfig config;
    uint32_t capacity=4096,k,m,nlinks=(uint32_t)MSX.Nobjects[LINK];
    uint32_t ntanks=(uint32_t)MSX.Nobjects[TANK],species=(uint32_t)MSX.Nobjects[SPECIES];
    int okay=0,hasLayout=0;memset(&b,0,sizeof(b));memset(&layout,0,sizeof(layout));
    if(!path||!GoldenReady||MSX.Qtime||MSX.ErrCode||
       GoldenVersion!=MSXresidentRuntime_stateVersion()||sizeof(double)!=8||!species)return 1;
    if(MSXresident_isOpen()){
        if(MSXresident_getLayout(&layout)!=MSX_RESIDENT_OK)return 1;
        hasLayout=1;
        if(MSXresidentCapacity_memoryConfig(&config)!=MSX_RESIDENT_OK)return 1;
        if(capacity>config.transferBatchRows)capacity=config.transferBatchRows;
        if(capacity>layout.totalSlots)capacity=layout.totalSlots;
    }
    if(!batchOpen(&b,capacity,species+1))goto done;
    f=fopen(path,"wb");if(!f)goto done;
    if(!put(f,"MSXIG001",8)||!put32(f,3)||!put32(f,0x01020304)||!put32(f,nlinks)||
       !put32(f,ntanks)||!put32(f,species)||!put32(f,(uint32_t)MSX.MaxSegments)||
       !put32(f,(uint32_t)MSX.GpuCoreMode)||!put32(f,capacity)||
       !put32(f,(uint32_t)MSX.Nobjects[NODE]))goto done;
    if(!put64(f,(uint64_t)MSX.Qtime)||!put64(f,(uint64_t)MSX.Htime)||
       !put64(f,(uint64_t)MSX.Qstep)||!put64(f,(uint64_t)MSX.Dur)||
       !put64(f,(uint64_t)MSX.Rtime))goto done;
    for(m=1;m<=species;++m){if(!isfinite(MSX.MassBalance.initial[m])||
        !put(f,&MSX.MassBalance.initial[m],sizeof(double)))goto done;}
    if(!put(f,MSX.C1+1,species*sizeof(double))||
       !put(f,MSX.MassBalance.inflow+1,species*sizeof(double))||
       !put(f,MSX.MassBalance.indisperse+1,species*sizeof(double))||
       !put(f,MSX.MassBalance.outflow+1,species*sizeof(double))||
       !put(f,MSX.MassBalance.reacted+1,species*sizeof(double))||
       !put(f,MSX.MassBalance.final+1,species*sizeof(double))||
       !put(f,MSX.MassBalance.ratio+1,species*sizeof(double)))goto done;
    for(k=1;k<=(uint32_t)MSX.Nobjects[NODE];++k){
        uint32_t sources=0;Psource source;
        for(source=MSX.Node[k].sources;source;source=source->next){if(sources==UINT32_MAX)goto done;++sources;}
        if(!put32(f,k)||!put32(f,(uint32_t)MSX.Node[k].tank)||!put32(f,sources)||
           !put(f,MSX.Node[k].c+1,species*sizeof(double))||
           !put(f,MSX.Node[k].c0+1,species*sizeof(double))||
           !put(f,&MSX.D[k],sizeof(REAL4))||!put(f,&MSX.H[k],sizeof(REAL4)))goto done;
        for(source=MSX.Node[k].sources;source;source=source->next){
            /* Schema 3 excludes legacy massRate: readSource never initializes
               this unused member. Do not read its indeterminate storage. */
            if(!put32(f,(uint32_t)source->type)||!put32(f,(uint32_t)source->species)||
               !put(f,&source->c0,sizeof(double))||!put32(f,(uint32_t)source->pat))goto done;
        }
    }
    for(k=1;k<=ntanks;++k){
        Stank *tank=&MSX.Tank[k];double scalar[5];
        scalar[0]=tank->hstep;scalar[1]=tank->a;scalar[2]=tank->v0;
        scalar[3]=tank->v;scalar[4]=tank->vMix;
        if(!put32(f,k)||!put32(f,(uint32_t)tank->node)||!put32(f,(uint32_t)tank->mixModel)||
           !put(f,scalar,sizeof(scalar))||!put(f,tank->c+1,species*sizeof(double))||
           !put(f,tank->reacted+1,species*sizeof(double)))goto done;
    }
    for(k=1;k<=nlinks;++k)if(!put32(f,k)||
       !put(f,MSX.Link[k].reacted+1,species*sizeof(double)))goto done;
    if(!put32(f,hasLayout?layout.nLinks:0))goto done;
    if(hasLayout)for(k=1;k<=layout.nLinks;++k){
        MSXResidentPipeDesc d;uint32_t slot;
        if(MSXresident_testInitialDescriptor(k,&d)!=MSX_RESIDENT_OK||
           !put32(f,d.linkIndex)||!put32(f,d.capacity)||!put32(f,d.head)||!put32(f,d.tail)||
           !put32(f,d.count)||!put32(f,(uint32_t)d.orient)||!put64(f,d.epoch))goto done;
        for(slot=0;slot<d.capacity;++slot){uint32_t used,generation;uint64_t id;
            if(MSXresident_testInitialSlot(k,slot,&used,&generation,&id)!=MSX_RESIDENT_OK||
               !put32(f,used)||!put32(f,generation)||!put64(f,id))goto done;}
    }
    for(k=1;k<=nlinks+ntanks;++k){
        uint32_t kind=k>nlinks,index=kind?k-nlinks:k,actual=0,ordinal=0,expected;
        Pseg seg,previous=NULL;
        if(kind){Stank *tank=&MSX.Tank[index];expected=tank->a==0.0?0:(tank->mixModel==MIX2?2:1);}
        else expected=MSXqual_pipeVolume(&MSX.Link[k])>0.0?(uint32_t)MIN(100,MSX.MaxSegments):0;
        for(seg=MSX.FirstSeg[k];seg;seg=seg->prev){
            if(actual>=expected||seg->next!=previous||
               (previous&&previous->prev!=seg))goto done;
            previous=seg;++actual;
        }
        if(previous!=MSX.LastSeg[k]||(previous&&previous->prev)||MSX.NewSeg[k])goto done;
        if(actual!=expected)goto done;
        if(!kind&&(MSX.Link[k].nsegs<0||(uint32_t)MSX.Link[k].nsegs!=actual))goto done;
        if(!put32(f,kind)||!put32(f,index)||!put32(f,expected)||!put32(f,actual)||
           !put32(f,kind?0:(uint32_t)MSX.FlowDir[k])||
           !put32(f,!kind&&hasLayout?layout.guard[k]:0)||
           !put32(f,!kind&&hasLayout?layout.admissionLimit[k]:0)||
           !put32(f,!kind&&hasLayout?layout.capacity[k]:0)||
           !put32(f,kind?UINT32_MAX:(uint32_t)MSX.Link[k].nsegs)||
           !put32(f,MSX.FirstSeg[k]!=NULL)||!put32(f,MSX.LastSeg[k]!=NULL)||
           !put32(f,MSX.NewSeg[k]!=NULL)||!put32(f,UINT32_MAX)||
           !put64(f,MSX.FirstSeg[k]?MSX.FirstSeg[k]->hybridId:0)||
           !put64(f,MSX.LastSeg[k]?MSX.LastSeg[k]->hybridId:0)||
           !put64(f,MSX.NewSeg[k]?MSX.NewSeg[k]->hybridId:0))goto done;
    }
    /* Rows form one canonical stream after all entity headers. Batches may
       cross pipe boundaries without creating a full-slot concentration copy. */
    for(k=1;k<=nlinks+ntanks;++k){
        uint32_t kind=k>nlinks,index=kind?k-nlinks:k,ordinal=0;Pseg seg;
        for(seg=MSX.FirstSeg[k];seg;seg=seg->prev,++ordinal){
            if(!batchAppend(&b,kind,index,ordinal,seg))goto done;
            if(b.count==b.capacity&&!batchFlush(f,&b))goto done;
        }
    }
    if(!batchFlush(f,&b))goto done;
    if(MSX.Qtime||GoldenVersion!=MSXresidentRuntime_stateVersion()||MSX.ErrCode)goto done;
    okay=1;
done:
    if(f){int error=ferror(f);if(fclose(f)||error)okay=0;}
    batchFree(&b);return okay?0:1;
}
#endif
