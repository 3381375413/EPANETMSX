/* C01 independent oracle: never calls a production endpoint accessor,
   snapshot, cached count, or span generator to construct expected output. */
static void partition_oracle_compare(void)
{
    Pseg q, previous=NULL, first=NULL, last=NULL, seen[64];
    int count=0,core=0,phase=0,i;
    const MSXHybridPartitionView *v=NULL;
    uint64_t token=0;
    for(q=MSX.FirstSeg[1];q;q=q->prev) {
        OK(count<64);
        if(count>=64)return;
        for(i=0;i<count;i++)OK(seen[i]!=q);
        seen[count++]=q;
        OK(q->next==previous);
        if(q->inHybridCore) {
            OK(phase!=2&&q->ownerLink==1&&q->hybridId!=0);
            if(!first)first=q;
            last=q;core++;phase=1;
        } else if(phase==1)phase=2;
        previous=q;
    }
    OK(previous==MSX.LastSeg[1]&&count==MSX.Link[1].nsegs);
    OK(MSXsegStorage_hybridBeginReadWindow(&token)==0);
    OK(MSXsegStorage_hybridValidateReadWindow(token)==0);
    OK(MSXsegStorage_hybridGetPartitionView(1,&v)==0&&v!=NULL);
    if(v) {
        OK(v->valid&&v->continuityProven&&v->link==1&&v->lifecycle!=0);
        OK(v->coreCount==core&&v->firstCore==first&&v->lastCore==last);
        OK(v->firstCoreId==(first?first->hybridId:0)&&v->lastCoreId==(last?last->hybridId:0));
        OK(v->head==(first?first->hybridSlot:-1)&&v->tail==(last?last->hybridSlot:-1));
    }
    OK(MSXsegStorage_hybridEndReadWindow(token)==0);
}
static void partition_fixture(int count)
{
    uint32_t caps[2]={0,32},guards[2]={0,2};MSXResidentLayout layout;
    int i;Pseg q,previous=NULL;
    setup(1,1);MSX.SegmentStorage=SEG_STORAGE_HYBRID;MSX.GpuCoreMode=0;
    MSX.MaxSegments=64;
    OK(MSXresident_openPlan(1,caps,guards,UP)==0);
    MSXresident_setMode(MSX_RESIDENT_SHADOW,0);
    OK(MSXsegStorage_open()==0&&MSXresident_getLayout(&layout)==0);
    OK(MSXsegStorage_hybridReserve(&layout)==0);
    for(i=0;i<count;i++) {
        q=fseg(i+1,(uint64_t)(0x1000+i));OK(q!=NULL);
        q->next=previous;
        if(previous)previous->prev=q;else MSX.FirstSeg[1]=q;
        previous=q;MSX.LastSeg[1]=q;MSX.Link[1].nsegs++;
    }
    OK(MSXsegStorage_hybridizeAll()==0);
}
static int partition_expect_bad_window(uint64_t *token)
{
    int error;
    OK(MSXsegStorage_hybridBeginReadWindow(token)==0);
    error=MSXsegStorage_hybridValidateReadWindow(*token);
    OK(MSXsegStorage_hybridEndReadWindow(*token)==0);
    *token=0;
    return error;
}
static void t_partition_view_contract(void)
{
    const MSXHybridPartitionView *v;Pseg first,last,q,prev,next,middle;
    uint64_t token=0,oldtoken,revision,lifecycle;unsigned rng=0x5317u;
    int i,operation,count;
    /* Random public promote/demote/reverse/head-remove commits, interleaved
       with the exact Qual CPU append/remove rewiring and C01 hook contract. */
    partition_fixture(12);partition_oracle_compare();
    for(i=0;i<512;i++) {
        rng=rng*1664525u+1013904223u;operation=(int)((rng>>16)%4);
        count=MSX.Link[1].nsegs;
        if(operation==0&&count<24) {
            OK(MSXsegStorage_hybridPrepareTopologyChange(1,1)==0);
            q=fseg((double)i+1,(uint64_t)(0x10000+i));OK(q!=NULL);
            q->next=MSX.LastSeg[1];
            if(q->next)q->next->prev=q;else MSX.FirstSeg[1]=q;
            MSX.LastSeg[1]=q;MSX.Link[1].nsegs++;
            MSXsegStorage_hybridCommitTopologyChange(1);
        } else if(operation==1&&count) {
            q=MSX.FirstSeg[1];
            if(q->inHybridCore)OK(MSXsegStorage_hybridRemoveHead(1,q)==0);
            else {
                OK(MSXsegStorage_hybridPrepareTopologyChange(1,1)==0);
                MSX.FirstSeg[1]=q->prev;
                if(q->prev)q->prev->next=NULL;else MSX.LastSeg[1]=NULL;
                MSX.Link[1].nsegs--;
                MSXsegStorage_hybridCommitTopologyChange(1);MSXqual_removeSeg(q);
            }
        } else if(operation==2) {
            OK(MSXsegStorage_hybridPrepareTopologyChange(1,1)==0);
            q=MSX.FirstSeg[1];MSX.FirstSeg[1]=MSX.LastSeg[1];MSX.LastSeg[1]=q;prev=NULL;
            while(q){next=q->prev;q->prev=prev;q->next=next;prev=q;q=next;}
            MSXsegStorage_hybridCommitTopologyChange(1);
            OK(MSXsegStorage_hybridAfterListReorder(1)==0);
        } else MSXsegStorage_hybridRebalanceAll();
        OK(MSX.ErrCode==0);partition_oracle_compare();
    }
    partition_fixture(12);partition_oracle_compare();
    OK(MSXsegStorage_hybridGetPartitionView(1,&v)==0);
    revision=v->revision;lifecycle=v->lifecycle;first=MSX.FirstSeg[1];last=MSX.LastSeg[1];
    OK(MSXsegStorage_hybridBeginReadWindow(&token)==0);oldtoken=token;
    OK(MSXsegStorage_hybridPrepareTopologyChange(1,1)==ERR_PIPE_RING_CAPACITY);
    MSX.ErrCode=0;
    MSXsegStorage_hybridClear(1);OK(MSX.ErrCode==ERR_PIPE_RING_CAPACITY);MSX.ErrCode=0;
    OK(MSXresident_stageClear(1)==MSX_RESIDENT_ERR_ARGUMENT);MSX.ErrCode=0;
    OK(MSXresident_stageReverse(1)==MSX_RESIDENT_ERR_ARGUMENT);MSX.ErrCode=0;
    MSXsegStorage_close();OK(MSX.ErrCode==ERR_PIPE_RING_CAPACITY);MSX.ErrCode=0;
    MSXsegStorage_reset();OK(MSX.ErrCode==ERR_PIPE_RING_CAPACITY);MSX.ErrCode=0;
    OK(MSXsegStorage_hybridGetPartitionView(1,&v)==0&&v->valid&&v->revision==revision);
    OK(MSX.FirstSeg[1]==first&&MSX.LastSeg[1]==last);
    OK(MSXsegStorage_hybridEndReadWindow(token)==0);token=0;
    OK(MSXsegStorage_hybridBeginReadWindow(&token)==0&&token!=oldtoken);
    OK(MSXsegStorage_hybridEndReadWindow(oldtoken)==ERR_PIPE_RING_CAPACITY);MSX.ErrCode=0;
    OK(MSXsegStorage_hybridWriteAllowed()==ERR_PIPE_RING_CAPACITY);MSX.ErrCode=0;
    OK(MSXsegStorage_hybridEndReadWindow(token)==0);token=0;
    /* Cached endpoints are both unchanged: an interior CPU island is a hard
       error, not a legal invalidation/rebuild. */
    middle=v->firstCore->prev;OK(middle&&middle!=v->lastCore);
    middle->inHybridCore=FALSE;MSXsegStorage_hybridInvalidatePartitionView(1);
    OK(partition_expect_bad_window(&token)==ERR_PIPE_RING_CAPACITY&&token==0);
    OK(MSXsegStorage_residentLastStatus()==MSX_RESIDENT_ERR_GENERATION);
    middle->inHybridCore=TRUE;MSX.ErrCode=0;partition_oracle_compare();
    q=MSX.FirstSeg[1]->prev;prev=q->next;q->next=NULL;
    MSXsegStorage_hybridInvalidatePartitionView(1);
    OK(partition_expect_bad_window(&token)==ERR_PIPE_RING_CAPACITY&&token==0);
    q->next=prev;MSX.ErrCode=0;partition_oracle_compare();
    /* Parcel identity mismatch remains fatal even with correct endpoints. */
    middle=v->firstCore;revision=middle->hybridId;middle->hybridId++;
    MSXsegStorage_hybridInvalidatePartitionView(1);
    OK(partition_expect_bad_window(&token)==ERR_PIPE_RING_CAPACITY&&token==0);
    middle->hybridId=revision;MSX.ErrCode=0;partition_oracle_compare();
#if defined(MSX_RESIDENT_TEST_API) || defined(MSX_RESIDENT_ACTIVE_TEST_API)
    {
        MSXResidentPipeDesc original,bad;MSXResidentActiveRow rows[64];uint32_t rowsCount=0;
        OK(MSXresident_testInitialDescriptor(1,&original)==0);bad=original;
        bad.count=bad.capacity+1;
        OK(MSXresident_testActiveDescriptor(1,&bad)==0);
        middle=v->firstCore;revision=middle->hybridId;middle->hybridId++;
        MSXsegStorage_hybridInvalidatePartitionView(1);
        /* Acquiring the lock cannot replace the old enumerator's CAPACITY
           with the partition validator's separate GENERATION failure. */
        OK(MSXsegStorage_hybridBeginReadWindow(&token)==0);
        OK(MSXresident_enumerateActiveSpans(rows,64,&rowsCount)==MSX_RESIDENT_ERR_CAPACITY);
        OK(MSXsegStorage_hybridValidateReadWindow(token)==ERR_PIPE_RING_CAPACITY);
        OK(MSXsegStorage_residentLastStatus()==MSX_RESIDENT_ERR_GENERATION);
        OK(MSXsegStorage_hybridEndReadWindow(token)==0);token=0;
        middle->hybridId=revision;MSX.ErrCode=0;
        OK(MSXresident_testActiveDescriptor(1,&original)==0);
        partition_oracle_compare();
    }
#endif
#if defined(MSX_RESIDENT_TEST_API) || defined(MSX_RESIDENT_PARTITION_TEST_API)
    OK(MSXsegStorage_testPartitionCounters(1,lifecycle,UINT64_MAX,100)==0);
    OK(MSXsegStorage_hybridPrepareTopologyChange(1,1)==ERR_PIPE_RING_CAPACITY);
    OK(MSX.FirstSeg[1]==first&&MSX.LastSeg[1]==last);MSX.ErrCode=0;
    OK(MSXsegStorage_testPartitionCounters(1,lifecycle,UINT64_MAX-1,100)==0);
    OK(MSXsegStorage_hybridPrepareTopologyChange(1,2)==ERR_PIPE_RING_CAPACITY);MSX.ErrCode=0;
    OK(MSXsegStorage_hybridPrepareTopologyChange(1,1)==0);
    MSXsegStorage_hybridCommitTopologyChange(1);
    OK(MSXsegStorage_hybridGetPartitionView(1,&v)==0&&v->revision==UINT64_MAX&&!v->valid);
    OK(MSXsegStorage_testPartitionCounters(1,UINT64_MAX,0,100)==0);
    OK(MSXsegStorage_open()==ERR_PIPE_RING_CAPACITY&&MSX.FirstSeg[1]==first);MSX.ErrCode=0;
    OK(MSXsegStorage_testPartitionCounters(1,lifecycle,0,UINT64_MAX)==0);
    OK(MSXsegStorage_hybridBeginReadWindow(&token)==ERR_PIPE_RING_CAPACITY&&token==0);MSX.ErrCode=0;
    OK(MSXsegStorage_testPartitionCounters(1,lifecycle,0,100)==0);
#endif
    partition_oracle_compare();
    cleanup();partition_fixture(12);partition_oracle_compare();
    OK(MSXsegStorage_hybridGetPartitionView(1,&v)==0&&v->lifecycle!=lifecycle);
}
