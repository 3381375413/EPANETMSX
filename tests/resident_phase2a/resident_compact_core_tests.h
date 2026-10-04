#include "msxresident_upload.h"
static MSXResidentStatus compactStage(void *owner,const MSXResidentSlotPatch *rows,uint32_t n)
{uint32_t i;(void)owner;for(i=0;i<n;++i)OK(rows[i].payload.c&&rows[i].payload.lastc&&rows[i].uploadRevision);return MSX_RESIDENT_OK;}
static void t_compact_core(void)
{
    char csvRows[512];double c[2]={-0.0,2},lastc[2]={0.0,3};uint64_t id=99;
    MSXResidentPayload value,q;MSXResidentPatchBatch patches;
    setup(1,1);MSX.GpuCoreMode=2;onecsv(csvRows,sizeof(csvRows),UP,2);csv("resident_phase2a.csv",csvRows);
    OK(MSXresident_open("resident_phase2a.csv")==0);MSXresident_setMode(MSX_RESIDENT_RESIDENT,1);
    OK(MSXupload_open(1,2,compactStage,(void *)1)==0);value=payload(c,lastc,5,id);
    OK(MSXresident_beginInitialImage()==0);
    OK(MSXresident_stageInitialPipe(1,&id,&value,1,1)==0);
    c[1]=lastc[1]=999; /* Core owns no borrowed Pseg concentration. */
    OK(MSXresident_getSlotMetadata(1,0,1,&q)==0&&q.parcelId==id&&q.volume==5&&!q.c&&!q.lastc);
    OK(MSXresident_getSlotPayload(1,0,1,&q)==MSX_RESIDENT_ERR_ARGUMENT&&!q.c&&!q.lastc);
    OK(MSXresident_getInitialBatch(&patches)==0&&patches.slot[0].uploadRevision==1&&!patches.slot[0].payload.c);
    OK(MSXupload_flush()==0&&MSXresident_commitInitialImage()==0);
    OK(MSXresident_observePipe(1,&id,&value,1,1)==MSX_RESIDENT_ERR_ARGUMENT);
    MSXupload_close();MSXresident_close();MSX.GpuCoreMode=0;
}
