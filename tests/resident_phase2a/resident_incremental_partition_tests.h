static uint64_t TestIncrementalCoreAvoided;
static void incremental_oracle(void)
{
    MSXHybridRebalanceSnapshot actual;Pseg q,previous=NULL,first=NULL,last=NULL,seen[64];
    int total=0,core=0,down=0,up=0,phase=0,i;uint64_t token=0;
    uint64_t beforePublished,beforeFallback,afterPublished,afterFallback;
    const MSXHybridPartitionView *actualView;int captureTrusted=0;
    /* The reference never calls an accessor/span/snapshot/view. */
    for(q=MSX.FirstSeg[1];q;q=q->prev) {
        OK(total<64);if(total>=64)return;
        for(i=0;i<total;i++)OK(seen[i]!=q);
        seen[total++]=q;OK(q->next==previous);
        if(q->inHybridCore){OK(phase!=2);if(!first)first=q;last=q;core++;phase=1;}
        else if(!phase)down++;else{up++;phase=2;}
        previous=q;
    }
    OK(total==MSX.Link[1].nsegs&&previous==MSX.LastSeg[1]);
    /* Compare the consumer's actual pre-refresh valid/fallback choice. A
       later explicit full validation seeds the next mutation's proof. */
    OK(MSXsegStorage_hybridGetPartitionView(1,&actualView)==0);
    captureTrusted=actualView->valid&&actualView->continuityProven;
    MSXsegStorage_hybridIncrementalMetrics(&beforePublished,&beforeFallback);
    OK(MSXsegStorage_hybridCaptureIncrementalSnapshotReadOnly(1,&actual)==0);
    MSXsegStorage_hybridIncrementalMetrics(&afterPublished,&afterFallback);
    OK(beforePublished==afterPublished&&beforeFallback==afterFallback);
    if(captureTrusted)TestIncrementalCoreAvoided+=(uint64_t)actual.coreCount;
    OK(actual.total==total&&actual.coreCount==core&&actual.downCount==down&&actual.upCount==up);
    OK(actual.firstCore==first&&actual.lastCore==last);
    OK(actual.firstCoreId==(first?first->hybridId:0)&&actual.lastCoreId==(last?last->hybridId:0));
    OK(MSXsegStorage_hybridBeginReadWindow(&token)==0);
    OK(MSXsegStorage_hybridValidateReadWindow(token)==0);
    OK(MSXsegStorage_hybridEndReadWindow(token)==0);
}
static void t_incremental_partition_development(void)
{
    unsigned seeds[3]={1,7,42};int seed,i,j,op,count;unsigned rng;
    Pseg q,next,previous,first,last;uint64_t token,revision,published,fallback,avoided;
    uint64_t basePublished,baseFallback,baseAvoided;
    const MSXHybridPartitionView *v;MSXHybridRebalanceSnapshot scratch;
    for(seed=0;seed<3;seed++) {
        rng=seeds[seed];partition_fixture(12);incremental_oracle();
        MSXsegStorage_hybridIncrementalMetrics(&basePublished,&baseFallback);
        baseAvoided=TestIncrementalCoreAvoided;
        for(i=0;i<1000;i++) {
            if(i&&i%137==0) {
                MSXsegStorage_hybridClear(1);incremental_oracle();
                for(j=0;j<12;j++) {
                    q=fseg((double)j+1,((uint64_t)seeds[seed]<<40)+((uint64_t)i<<8)+(uint64_t)j);
                    OK(MSXsegStorage_hybridAppendCpuBoundary(1,q)==0);
                }
                MSXsegStorage_hybridRebalanceAll();incremental_oracle();
            }
            rng=rng*1664525u+1013904223u;op=(int)((rng>>16)%5);count=MSX.Link[1].nsegs;
            if(i%19==0) {
                q=fseg(1,((uint64_t)seeds[seed]<<32)+(uint64_t)i+1000);
                OK(MSXsegStorage_hybridGetPartitionView(1,&v)==0);
                revision=v->revision;first=MSX.FirstSeg[1];last=MSX.LastSeg[1];token=0;
                OK(MSXsegStorage_hybridBeginReadWindow(&token)==0);
                OK(MSXsegStorage_hybridAppendCpuBoundary(1,q)==ERR_PIPE_RING_CAPACITY);
                OK(MSX.FirstSeg[1]==first&&MSX.LastSeg[1]==last&&v->revision==revision&&MSX.Link[1].nsegs==count);
                OK(MSXsegStorage_hybridEndReadWindow(token)==0);MSX.ErrCode=0;MSXqual_removeSeg(q);
            }
            if(i%23==0&&count) {
                /* Passing an already-linked object cannot manufacture an
                   owned append proof or change the committed revision. */
                OK(MSXsegStorage_hybridGetPartitionView(1,&v)==0);
                revision=v->revision;first=MSX.FirstSeg[1];last=MSX.LastSeg[1];
                OK(MSXsegStorage_hybridAppendCpuBoundary(1,first)==ERR_PIPE_RING_CAPACITY);
                OK(MSX.FirstSeg[1]==first&&MSX.LastSeg[1]==last&&v->revision==revision);
                MSX.ErrCode=0;
            }
            if(op==0&&count<24) {
                q=fseg((double)i+1,((uint64_t)seeds[seed]<<32)+(uint64_t)i+0x10000);
                OK(MSXsegStorage_hybridAppendCpuBoundary(1,q)==0);
            } else if(op==1&&count) {
                q=MSX.FirstSeg[1];
                if(q->inHybridCore)OK(MSXsegStorage_hybridRemoveHead(1,q)==0);
                else {OK(MSXsegStorage_hybridRemoveCpuHead(1,q)==0);MSXqual_removeSeg(q);}
            } else if(op==2) {
                OK(MSXsegStorage_hybridPrepareTopologyChange(1,1)==0);
                q=MSX.FirstSeg[1];MSX.FirstSeg[1]=MSX.LastSeg[1];MSX.LastSeg[1]=q;previous=NULL;
                while(q){next=q->prev;q->prev=previous;q->next=next;previous=q;q=next;}
                MSXsegStorage_hybridCommitTopologyChange(1);
                OK(MSXsegStorage_hybridAfterListReorder(1)==0);
            } else if(op==3)MSXsegStorage_hybridRebalanceAll();
            else if(count) {
                /* Scalar-only CPU tail merge is deliberately not a new
                   topology event or a cached Core-volume operation. */
                q=MSX.LastSeg[1];if(!q->inHybridCore){q->v+=0.125;q->hstep+=0.25;}
            }
            if(MSX.ErrCode) {
                MSXResidentPipeDesc descriptor;
                MSXResidentPatchBatch patches;
                (void)MSXresident_getPatches(&patches);
                (void)MSXresident_testInitialDescriptor(1,&descriptor);
                printf("c08_first_fault seed=%u i=%d op=%d count=%d error=%d status=%d\n",
                    seeds[seed],i,op,count,MSX.ErrCode,(int)MSXsegStorage_residentLastStatus());
                printf("c08_fault_descriptor count=%u head=%u tail=%u cap=%u orient=%d cpuCore=%d cap=%d\n",
                    descriptor.count,descriptor.head,descriptor.tail,descriptor.capacity,descriptor.orient,
                    MSXsegStorage_hybridCoreCount(1),MSXsegStorage_hybridCoreCapacity(1));
                printf("c08_fault_patch_slots=%u\n",patches.slotCount);
                OK(MSX.ErrCode==0);break;
            }
            incremental_oracle();
            /* Each randomized operation is one completed quality-step
               publication; production flush consumes this patch arena. */
            MSXresident_clearPatches();
        }
        MSXsegStorage_hybridIncrementalMetrics(&published,&fallback);
        avoided=TestIncrementalCoreAvoided;
        printf("c08_seed=%u operations=1000 typed_published=%llu complex_fallback=%llu core_rows_avoided=%llu\n",
               seeds[seed],(unsigned long long)(published-basePublished),
               (unsigned long long)(fallback-baseFallback),(unsigned long long)(avoided-baseAvoided));
        OK(published>basePublished&&fallback>baseFallback&&avoided>baseAvoided);
    }
    partition_fixture(12);incremental_oracle();
    OK(MSXsegStorage_hybridGetPartitionView(1,&v)==0);
    q=v->firstCore->prev;q->inHybridCore=FALSE;
    MSXsegStorage_hybridCommitTopologyChange(1);
    OK(!v->valid&&!v->continuityProven);
    OK(MSXsegStorage_hybridCaptureIncrementalSnapshotReadOnly(1,&scratch)==ERR_PIPE_RING_CAPACITY);
    token=0;OK(MSXsegStorage_hybridBeginReadWindow(&token)==0);
    OK(MSXsegStorage_hybridValidateReadWindow(token)==ERR_PIPE_RING_CAPACITY);
    OK(MSXsegStorage_hybridEndReadWindow(token)==0);
    q->inHybridCore=TRUE;MSX.ErrCode=0;incremental_oracle();
}

static void t_incremental_capacity_reject(void)
{
    MSXResidentPipeDesc before,after;MSXResidentPayload payload;
    MSXResidentPatchBatch patchesBefore,patchesAfter;
    const MSXHybridPartitionView *view;Pseg first,last;
    uint32_t slot=0,generation=0,target;uint64_t revision;
    partition_fixture(12);incremental_oracle();MSXresident_clearPatches();
    OK(MSXresident_testInitialDescriptor(1,&before)==0);
    target=before.orient>0?(before.tail+1)%before.capacity:
        (before.tail+before.capacity-1)%before.capacity;
    first=MSX.FirstSeg[1];last=MSX.LastSeg[1];
    OK(MSXsegStorage_hybridGetPartitionView(1,&view)==0);revision=view->revision;
    memset(&payload,0,sizeof(payload));payload.parcelId=0xfeed;
    payload.c=last->c;payload.lastc=last->lastc;payload.volume=1;
    OK(MSXresident_getPatches(&patchesBefore)==0);
    /* Occupied-adjacent-slot fault is injected independently of view
       accessors; rejection must precede descriptor/patch/CPU writes. */
    OK(MSXresident_testActiveUsed(1,target,1)==0);
    OK(MSXresident_stageInsert(1,1,&payload,&slot,&generation)==MSX_RESIDENT_ERR_CAPACITY);
    OK(MSXresident_testInitialDescriptor(1,&after)==0);
    OK(memcmp(&before,&after,sizeof(before))==0&&slot==0&&generation==0);
    OK(MSXresident_getPatches(&patchesAfter)==0);
    OK(patchesBefore.slotCount==patchesAfter.slotCount&&patchesBefore.descriptorCount==patchesAfter.descriptorCount);
    OK(MSX.FirstSeg[1]==first&&MSX.LastSeg[1]==last&&view->revision==revision&&view->valid);
    OK(MSXresident_testActiveUsed(1,target,0)==0);incremental_oracle();
}
