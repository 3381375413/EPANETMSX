static uint64_t sharedCapturedId[32];static double sharedCapturedC[32];
static unsigned sharedCapturedRows;
static MSXResidentStatus sharedCapture(void *owner,const MSXResidentSlotPatch *rows,uint32_t n)
{
    uint32_t i;(void)owner;
    for(i=0;i<n;++i){
        OK(sharedCapturedRows<32);
        if(sharedCapturedRows<32){
            sharedCapturedId[sharedCapturedRows]=rows[i].payload.parcelId;
            sharedCapturedC[sharedCapturedRows++]=rows[i].payload.c[1];
        }
        OK(signbit(rows[i].payload.c[0])&&signbit(rows[i].payload.lastc[0]));
    }
    return MSX_RESIDENT_OK;
}
static int sharedFixture(MSXResidentLayout *layout)
{
    uint32_t limits[4]={0,3,5,0},guards[4]={0,2,4,2};int k;
    setup(1,3);MSX.SegmentStorage=SEG_STORAGE_HYBRID;MSX.GpuCoreMode=2;MSX.GpuCoreOverflow=1;
    for(k=1;k<=3;++k){MSX.Link[k].len=k<3?1:0;MSX.Link[k].diam=1;}
    if(MSXresident_openPlan(3,limits,guards,UP)||MSXresident_getLayout(layout))return 0;
    MSXresident_setMode(MSX_RESIDENT_RESIDENT,1);
    return MSXsegStorage_open()==0;
}
static void t_shared_initial_observer(void)
{
    MSXResidentLayout layout;MSXResidentBudget before,after;uint64_t bytes;
    double c1[2]={-0.0,10},c2[2]={-0.0,20};unsigned i;int attempt;
    testDirectInitialPlanning=1;OK(sharedFixture(&layout));
    MSXresidentBudget_snapshot(MSXresidentBudget_global(),&before);
    OK(MSXsegStorage_hybridReserve(&layout)==0);
    MSXresidentBudget_snapshot(MSXresidentBudget_global(),&after);bytes=after.allocated[0]-before.allocated[0];
    OK(MSXsegStorage_hybridInitialScratchCapacity()==5);
    OK(bytes+MSXsegStorage_hybridExistingHostBytes()+
       layout.totalSlots*(sizeof(struct Sseg)+4*sizeof(double))==
       MSXsegStorage_hybridFixedHostBytesWithInitial(3,layout.totalSlots,2,1,5));
    OK(MSXsegStorage_hybridFixedHostBytes(3,layout.totalSlots,2)>
       MSXsegStorage_hybridFixedHostBytesWithInitial(3,layout.totalSlots,2,1,5));
    sharedCapturedRows=0;
    OK(MSXupload_open(2,2,sharedCapture,(void *)1)==MSX_RESIDENT_OK);
    OK(MSXresident_beginInitialImage()==MSX_RESIDENT_OK);
    OK(MSXsegStorage_hybridPrepareInitialImage()==ERR_PIPE_RING_CAPACITY);
    OK(MSXsegStorage_hybridStageDirectInitialPipe(1,16,1,c1)==0);c1[1]=999;
    OK(MSXsegStorage_hybridStageDirectInitialPipe(2,16,2,c2)==0);c2[1]=999;
    OK(MSXsegStorage_hybridStageDirectInitialPipe(3,0,0,c1)==0);
    OK(MSXupload_flush()==MSX_RESIDENT_OK&&sharedCapturedRows==8);
    for(i=0;i<8;++i){
        OK(sharedCapturedC[i]==(i<3?10:20));
        OK((sharedCapturedId[i]>>48)==(i<3?1:2));
    }
    OK(MSXsegStorage_hybridValidateDirectInitial()==0);
    MSXsegStorage_hybridAbortInitialImage();MSXsegStorage_hybridAbortInitialImage();
    MSXresident_abortInitialImage();MSXupload_close();
    testDirectInitialPlanning=0;MSX.GpuCoreMode=0;
    MSXsegStorage_close();MSXsegStorage_close();
    for(attempt=0;attempt<2;++attempt){
        testDirectInitialPlanning=1;OK(sharedFixture(&layout));
        MSXresidentBudget_snapshot(MSXresidentBudget_global(),&before);
        OK(MSXresidentBudget_configure(MSXresidentBudget_global(),before.allocated[0]+bytes-(attempt==0),UINT64_MAX)==MSX_BUDGET_OK);
        OK(MSXsegStorage_hybridReserve(&layout)==(attempt==0?ERR_MEMORY:0));
        OK(MSXsegStorage_hybridInitialScratchCapacity()==(attempt==0?0:5));
        testDirectInitialPlanning=0;MSX.GpuCoreMode=0;
        MSXsegStorage_close();MSXsegStorage_close();
        MSXresidentBudget_snapshot(MSXresidentBudget_global(),&after);
        OK(after.allocated[0]+MSXsegStorage_hybridExistingHostBytes()<before.allocated[0]&&!after.reserved[0]);
        OK(MSXresidentBudget_configure(MSXresidentBudget_global(),UINT64_MAX,UINT64_MAX)==MSX_BUDGET_OK);
    }
    testDirectInitialPlanning=1;OK(sharedFixture(&layout));
    for(i=1;i<=3;++i)MSX.Link[i].len=0;
    MSXresidentBudget_snapshot(MSXresidentBudget_global(),&before);
    OK(MSXsegStorage_hybridReserve(&layout)==0&&MSXsegStorage_hybridInitialScratchCapacity()==0);
    MSXresidentBudget_snapshot(MSXresidentBudget_global(),&after);
    OK(after.allocated[0]-before.allocated[0]+MSXsegStorage_hybridExistingHostBytes()+
       layout.totalSlots*(sizeof(struct Sseg)+4*sizeof(double))==
       MSXsegStorage_hybridFixedHostBytesWithInitial(3,layout.totalSlots,2,1,0));
    OK(MSXresident_beginInitialImage()==MSX_RESIDENT_OK);
    for(i=1;i<=3;++i)OK(MSXsegStorage_hybridStageDirectInitialPipe((int)i,0,0,c1)==0);
    OK(MSXsegStorage_hybridValidateDirectInitial()==0);
    MSXsegStorage_hybridAbortInitialImage();MSXresident_abortInitialImage();
    testDirectInitialPlanning=0;
    OK(fixture());OK(MSXsegStorage_hybridObserveAll()==0);
}
