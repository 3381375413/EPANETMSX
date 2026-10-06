/* C02 private seams are compiled only into this harness's CUDA object. */
MSXResidentStatus MSXresidentGpu_testActiveStamp(MSXResidentGpu *,uint32_t,uint64_t *);
MSXResidentStatus MSXresidentGpu_testActiveSequence(MSXResidentGpu *,uint64_t);

static uint64_t active_stamp(MSXResidentGpu *g,uint32_t row)
{
    uint64_t stamp=UINT64_MAX;
    OK(MSXresidentGpu_testActiveStamp(g,row,&stamp)==MSX_RESIDENT_OK);
    return stamp;
}
static void t_active_stamps(void)
{
    int passed=pass,failed=fail;
    MSXResidentGpu *g=initial4(),*other=initial4();
    MSXResidentGpuActiveWriter w,old;
    MSXResidentActiveRow row={1,0,0,1,0,2,1,5,101,2.0},bad;
    MSXResidentGpuDeviceView v;
    MSXResidentGpuReactResult r;
    MSXResidentGpuTransferStats before,after;
    double pipe[3*MSX_RESIDENT_HYD_STRIDE]={0};
    MSXResidentHydView hyd={pipe,2,MSX_RESIDENT_HYD_STRIDE,MSX_RESIDENT_HYD_PIPE_MAJOR},invalid;
    uint64_t sequence;
    /* Successful seal/finish leaves stamps intact; cross-batch same row is legal. */
    for(int i=0;i<3;++i) {
        OK(MSXresidentGpu_beginActive(g,1,10+i,&w)==0);
        sequence=w.buildSequence;
        OK(MSXresidentGpu_appendActive(g,&w,&row)==0);
        old=w;
        OK(MSXresidentGpu_appendActive(g,&old,&row)==MSX_RESIDENT_ERR_ARGUMENT);
        OK(MSXresidentGpu_sealActive(other,&w,10+i)==MSX_RESIDENT_ERR_ARGUMENT);
        OK(MSXresidentGpu_getTransferStats(g,&before)==0);
        OK(MSXresidentGpu_sealActive(g,&w,10+i)==0);
        OK(active_stamp(g,0)==sequence);
        OK(MSXresidentGpu_getTransferStats(g,&after)==0&&
           before.h2dCalls==after.h2dCalls&&before.d2hCalls==after.d2hCalls);
        OK(MSXresidentGpu_sealActive(g,&w,10+i)==MSX_RESIDENT_ERR_ARGUMENT);
        OK(MSXresidentGpu_beginActive(g,0,20,&old)==MSX_RESIDENT_ERR_ARGUMENT);
        OK(MSXresidentGpu_testActiveSequence(g,UINT64_MAX)==MSX_RESIDENT_ERR_ARGUMENT);
        OK(MSXresidentGpu_prepareSealedActiveHyd(g,&w,&hyd,&v,&r)==0);
        OK(MSXresidentGpu_finishActive(g,&r)==0&&!w.owner&&!w.valid);
        OK(active_stamp(g,0)==sequence);
        OK(MSXresidentGpu_appendActive(g,&old,&row)==MSX_RESIDENT_ERR_ARGUMENT);
        OK(MSXresidentGpu_finishActive(g,&r)==MSX_RESIDENT_ERR_ARGUMENT);
    }
    /* Same-batch duplicate and partial stale append clear only current stamps. */
    OK(MSXresidentGpu_beginActive(g,2,30,&w)==0&&MSXresidentGpu_appendActive(g,&w,&row)==0);
    old=w;
    OK(MSXresidentGpu_appendActive(g,&w,&row)==MSX_RESIDENT_ERR_ARGUMENT);
    OK(active_stamp(g,0)==0&&!w.owner&&!w.count);
    OK(MSXresidentGpu_abortActiveBuild(g,&old)==MSX_RESIDENT_ERR_ARGUMENT);
    OK(MSXresidentGpu_beginActive(g,2,31,&w)==0&&MSXresidentGpu_appendActive(g,&w,&row)==0);
    bad=row;bad.slot=1;bad.globalRow=1;bad.generation=9;
    OK(MSXresidentGpu_appendActive(g,&w,&bad)==MSX_RESIDENT_ERR_GENERATION);
    OK(active_stamp(g,0)==0&&!w.owner);
    /* Failed seal count/topology, explicit abort, invalid prepare all clean up. */
    for(int mode=0;mode<5;++mode) {
        OK(MSXresidentGpu_beginActive(g,mode==0?2:1,40,&w)==0);
        OK(MSXresidentGpu_appendActive(g,&w,&row)==0);
        if(mode==0) OK(MSXresidentGpu_sealActive(g,&w,40)==MSX_RESIDENT_ERR_CAPACITY);
        else if(mode==1) OK(MSXresidentGpu_sealActive(g,&w,41)==MSX_RESIDENT_ERR_GENERATION);
        else {
            OK(MSXresidentGpu_sealActive(g,&w,40)==0);
            OK(active_stamp(g,0)==w.buildSequence);
            if(mode==2) OK(MSXresidentGpu_abortActiveBuild(g,&w)==0);
            else if(mode==3) OK(MSXresidentGpu_abortActive(g)==0);
            else {
                invalid=hyd;invalid.linkCount=1;
                OK(MSXresidentGpu_prepareSealedActiveHyd(g,&w,&invalid,&v,&r)==MSX_RESIDENT_ERR_ARGUMENT);
            }
        }
        OK(active_stamp(g,0)==0&&!w.owner&&!w.count);
    }
    /* A stamp of 1 from an earlier successful batch must vanish on wrap. */
    OK(MSXresidentGpu_testActiveSequence(g,0)==0);
    OK(MSXresidentGpu_beginActive(g,1,50,&w)==0&&w.buildSequence==1);
    OK(MSXresidentGpu_appendActive(g,&w,&row)==0&&MSXresidentGpu_sealActive(g,&w,50)==0);
    OK(MSXresidentGpu_prepareSealedActiveHyd(g,&w,&hyd,&v,&r)==0&&MSXresidentGpu_finishActive(g,&r)==0);
    OK(active_stamp(g,0)==1);
    OK(MSXresidentGpu_testActiveSequence(g,UINT64_MAX)==0);
    OK(MSXresidentGpu_beginActive(g,1,51,&w)==0&&w.buildSequence==1);
    OK(active_stamp(g,0)==0);
    OK(MSXresidentGpu_appendActive(g,&w,&row)==0&&MSXresidentGpu_sealActive(g,&w,51)==0);
    OK(MSXresidentGpu_prepareSealedActiveHyd(g,&w,&hyd,&v,&r)==0&&MSXresidentGpu_finishActive(g,&r)==0);
    /* Empty seal doesn't clear unrelated historical stamps. */
    OK(MSXresidentGpu_beginActive(g,0,52,&w)==0&&MSXresidentGpu_sealActive(g,&w,52)==0);
    OK(active_stamp(g,0)==1);
    OK(MSXresidentGpu_prepareSealedActiveHyd(g,&w,&hyd,&v,&r)==0&&v.itemCount==0);
    OK(MSXresidentGpu_finishActive(g,&r)==0);
    OK(MSXresidentGpu_beginActive(g,1,53,&w)==0&&MSXresidentGpu_appendActive(g,&w,&row)==0);
    old=w;
    MSXresidentGpu_close(g);g=initial4();
    OK(active_stamp(g,0)==0);
    OK(MSXresidentGpu_appendActive(g,&old,&row)==MSX_RESIDENT_ERR_ARGUMENT);
    OK(MSXresidentGpu_sealActive(g,&old,53)==MSX_RESIDENT_ERR_ARGUMENT);
    OK(MSXresidentGpu_abortActiveBuild(g,&old)==MSX_RESIDENT_ERR_ARGUMENT);
    OK(MSXresidentGpu_beginActive(g,1,54,&w)==0&&w.buildSequence==1);
    OK(MSXresidentGpu_appendActive(g,&w,&row)==0&&MSXresidentGpu_abortActiveBuild(g,&w)==0);
    MSXresidentGpu_close(g);MSXresidentGpu_close(other);
    printf("c02_assertions_passed=%d\nc02_assertions_failed=%d\n",pass-passed,fail-failed);
}
