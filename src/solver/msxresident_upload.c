#include <stdio.h>
#include <string.h>
#include "msxresident_upload.h"
#include "msxresident_alloc.h"
static struct {MSXResidentSlotPatch *lease;double *c,*lastc;uint32_t capacity,stride,count,peak;
    uint64_t sequence,completed,staged,flushes,cancelled;MSXUploadStage stage;void *owner;int poisoned,staging;} U;
int MSXupload_enabled(void){return U.stage!=NULL;}
MSXResidentStatus MSXupload_open(uint32_t rows,uint32_t stride,MSXUploadStage stage,void *owner)
{
    MSXResidentBudget budget;uint64_t available,perRow;
    if(U.stage||!rows||!stride||!stage||!owner)return MSX_RESIDENT_ERR_ARGUMENT;
    MSXresidentBudget_snapshot(MSXresidentBudget_domain(MSX_BUDGET_UPLOAD),&budget);
    available=budget.hostLimit-budget.allocated[0]-budget.reserved[0];
    perRow=sizeof(*U.lease)+(uint64_t)2*stride*sizeof(double);
    if(available<=3*MSXresidentAlloc_headerBytes())return MSX_RESIDENT_ERR_MEMORY;
    available=(available-3*MSXresidentAlloc_headerBytes())/perRow;
    if(rows>available)rows=(uint32_t)available;if(!rows)return MSX_RESIDENT_ERR_MEMORY;
    U.capacity=rows;U.stride=stride;
    U.lease=MSXresidentAlloc_callocBudget(MSXresidentBudget_domain(MSX_BUDGET_UPLOAD),rows,sizeof(*U.lease),__FILE__,__LINE__);
    U.c=MSXresidentAlloc_callocBudget(MSXresidentBudget_domain(MSX_BUDGET_UPLOAD),rows,(size_t)stride*sizeof(double),__FILE__,__LINE__);
    U.lastc=MSXresidentAlloc_callocBudget(MSXresidentBudget_domain(MSX_BUDGET_UPLOAD),rows,(size_t)stride*sizeof(double),__FILE__,__LINE__);
    if(!U.lease||!U.c||!U.lastc){MSXupload_close();return MSX_RESIDENT_ERR_MEMORY;}
    U.stage=stage;U.owner=owner;return MSX_RESIDENT_OK;
}
MSXResidentStatus MSXupload_flush(void)
{
    MSXResidentStatus status;if(!U.stage||!U.count)return MSX_RESIDENT_OK;
    if(U.poisoned||U.staging)return MSX_RESIDENT_ERR_POISONED;
    U.staging=1;status=U.stage(U.owner,U.lease,U.count);U.staging=0;
    if(status){U.poisoned=1;return status;}
    /* The callback acknowledges completed H2D to independent IMPORT storage.
       Only now may snapshot and pinned rows be reused; no descriptor applied. */
    U.completed=U.lease[U.count-1].uploadRevision;U.staged+=U.count;++U.flushes;U.count=0;return MSX_RESIDENT_OK;
}
MSXResidentStatus MSXupload_capture(const MSXResidentSlotPatch *patch,uint64_t *revision)
{
    MSXResidentStatus status;uint32_t row;MSXResidentSlotPatch *lease;
    if(!patch||!revision||!U.stage||!patch->used||!patch->payload.c||!patch->payload.lastc||
       !patch->payload.parcelId||!patch->generation)return MSX_RESIDENT_ERR_ARGUMENT;
    if(U.poisoned||U.staging||U.sequence==UINT64_MAX)return MSX_RESIDENT_ERR_POISONED;
    if(U.count==U.capacity){status=MSXupload_flush();if(status)return status;}
    row=U.count++;lease=&U.lease[row];*lease=*patch;
    lease->uploadRevision=++U.sequence;lease->payload.c=U.c+(size_t)row*U.stride;
    lease->payload.lastc=U.lastc+(size_t)row*U.stride;
    memcpy((double *)lease->payload.c,patch->payload.c,(size_t)U.stride*sizeof(double));
    memcpy((double *)lease->payload.lastc,patch->payload.lastc,(size_t)U.stride*sizeof(double));
    if(U.count>U.peak)U.peak=U.count;*revision=lease->uploadRevision;return MSX_RESIDENT_OK;
}
void MSXupload_close(void)
{
    if(U.stage){FILE *f=fopen("upload_lease_summary.json","wb");U.cancelled+=U.count;
        if(f){fprintf(f,"{\"capacity\":%u,\"peak_leased\":%u,\"captured\":%llu,\"staged\":%llu,\"cancelled_on_close\":%llu,\"completion_sequence\":%llu,\"stage_batches\":%llu,\"poisoned\":%d,\"remaining_leases\":0}\n",
            U.capacity,U.peak,(unsigned long long)U.sequence,(unsigned long long)U.staged,
            (unsigned long long)U.cancelled,(unsigned long long)U.completed,(unsigned long long)U.flushes,U.poisoned);fclose(f);}}
    MSXresidentAlloc_free(U.lease);MSXresidentAlloc_free(U.c);MSXresidentAlloc_free(U.lastc);memset(&U,0,sizeof(U));
}

#ifdef MSX_RESIDENT_TEST_API
#include "msxresident_inventory.h"
void MSXinv_Upload(MSXInventory *s)
{
 INV_HEAP(s,U.lease);
 INV_HEAP(s,U.c);
 INV_HEAP(s,U.lastc);
 MSXinv_tag(s,"upload.capacity",U.capacity);MSXinv_tag(s,"upload.pending",U.count);MSXinv_tag(s,"upload.completed",U.completed);
}
#endif
