#include <stdio.h>
#include <string.h>
#include "msxresident_upload.h"
#include "msxresident_alloc.h"
static int failures,failStage,calls;static uint64_t received;
#define CHECK(x) do{if(!(x)){fprintf(stderr,"line %d: %s\n",__LINE__,#x);++failures;}}while(0)
static MSXResidentStatus stage(void *owner,const MSXResidentSlotPatch *rows,uint32_t n)
{
    uint32_t i;uint64_t revision;CHECK(owner==(void *)1);++calls;
    CHECK(MSXupload_capture(&rows[0],&revision)==MSX_RESIDENT_ERR_POISONED);
    if(failStage)return MSX_RESIDENT_ERR_TRANSFER;
    for(i=0;i<n;++i){CHECK(rows[i].uploadRevision==received+1);
        CHECK(rows[i].payload.c[0]==(double)received&&rows[i].payload.lastc[1]==-(double)received);
        ++received;}return MSX_RESIDENT_OK;
}
int main(void)
{
    MSXResidentBudget *b=MSXresidentBudget_domain(MSX_BUDGET_UPLOAD),snapshot;
    MSXResidentSlotPatch p={0};double c[2],lastc[2];uint64_t revision;int i;
    uint64_t size=3*(sizeof(p)+4*sizeof(double))+3*MSXresidentAlloc_headerBytes();
    CHECK(!MSXresidentBudget_configure(b,size,UINT64_MAX));
    CHECK(!MSXupload_open(100,2,stage,(void *)1));
    p.used=1;p.generation=1;p.linkIndex=1;p.slot=0;p.payload.parcelId=1;p.payload.c=c;p.payload.lastc=lastc;
    for(i=0;i<100;++i){c[0]=i;c[1]=i+1;lastc[0]=i;lastc[1]=-i;
        CHECK(!MSXupload_capture(&p,&revision)&&revision==(uint64_t)i+1);
        c[0]=lastc[1]=9999;} /* Original Pseg storage may be recycled at once. */
    CHECK(!MSXupload_flush()&&received==100&&calls==34);
    MSXresidentBudget_snapshot(b,&snapshot);CHECK(snapshot.allocated[0]==size);
    MSXupload_close();MSXupload_close();MSXresidentBudget_snapshot(b,&snapshot);CHECK(!snapshot.allocated[0]&&!snapshot.reserved[0]);
    CHECK(!MSXupload_open(3,2,stage,(void *)1));c[0]=100;lastc[1]=-100;
    CHECK(!MSXupload_capture(&p,&revision));failStage=1;
    CHECK(MSXupload_flush()==MSX_RESIDENT_ERR_TRANSFER);
    CHECK(MSXupload_flush()==MSX_RESIDENT_ERR_POISONED);
    CHECK(MSXupload_capture(&p,&revision)==MSX_RESIDENT_ERR_POISONED);
    MSXupload_close();MSXresidentBudget_snapshot(b,&snapshot);CHECK(!snapshot.allocated[0]&&!snapshot.reserved[0]);
    printf("resident_upload_tests failures=%d\n",failures);return failures?1:0;
}
