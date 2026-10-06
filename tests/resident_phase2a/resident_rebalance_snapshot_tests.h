/* Reference deliberately walks FirstSeg/prev and never calls partition or
   production snapshot helpers. Capture must not publish any Hybrid fields. */
#ifdef _OPENMP
#include <omp.h>
#endif
static void t_rebalance_snapshot_readonly(void)
{
    int count, i, coreBefore, total, down, up, core, seen;
    Pseg previous, q, first, last, firstCore, lastCore;
    MSXHybridRebalanceSnapshot draft;
    for (count=0; count<=12; ++count) {
        setup(1,1); MSX.SegmentStorage=SEG_STORAGE_HYBRID;
        OK(MSXsegStorage_open()==0);
        previous=NULL;
        for(i=0;i<count;++i) {
            q=fseg(i+1,(uint64_t)(3100+i)); OK(q!=NULL);
            q->next=previous;
            if(previous) previous->prev=q; else MSX.FirstSeg[1]=q;
            previous=q; MSX.LastSeg[1]=q; MSX.Link[1].nsegs++;
        }
        OK(MSXsegStorage_hybridizeAll()==0);
        coreBefore=MSXsegStorage_hybridCoreCount(1);
        first=MSX.FirstSeg[1]; last=MSX.LastSeg[1];
        total=down=up=core=seen=0; firstCore=lastCore=NULL;
        for(q=first;q;q=q->prev) {
            total++;
            if(q->inHybridCore) {
                if(!firstCore) firstCore=q;
                lastCore=q; core++; seen=1;
            } else if(!seen) down++; else up++;
        }
        memset(&draft,0xa5,sizeof(draft));
        OK(MSXsegStorage_hybridCaptureSnapshotReadOnly(1,&draft)==0);
        OK(draft.linkIndex==1 && draft.total==total && draft.coreCount==core &&
           draft.downCount==(core?down:total) && draft.upCount==(core?up:0));
        OK(draft.firstCore==firstCore && draft.lastCore==lastCore &&
           draft.firstCoreId==(firstCore?firstCore->hybridId:0) &&
           draft.lastCoreId==(lastCore?lastCore->hybridId:0));
        OK(draft.head==(firstCore?firstCore->hybridSlot:-1) &&
           draft.tail==(lastCore?lastCore->hybridSlot:-1));
        OK(MSX.FirstSeg[1]==first && MSX.LastSeg[1]==last &&
           MSX.Link[1].nsegs==count && MSXsegStorage_hybridCoreCount(1)==coreBefore);
        OK(MSXsegStorage_hybridCaptureSnapshotReadOnly(0,&draft)==ERR_PIPE_RING_CAPACITY);
        OK(MSXsegStorage_hybridCaptureSnapshotReadOnly(1,NULL)==ERR_PIPE_RING_CAPACITY);
        {
            MSXHybridRebalanceSnapshot candidate;
            int backend;
            /* Refresh is explicit fixture setup, never part of pure capture. */
            OK(MSXsegStorage_hybridRefreshPartitionView(1)==0);
            for(backend=MSX_SNAPSHOT_SERIAL_FULL;backend<=MSX_SNAPSHOT_ENDPOINT_BOUNDARY;++backend) {
                OK(MSXsegStorage_hybridCaptureSnapshotBackend(1,backend,&candidate)==0);
                OK(candidate.linkIndex==1 && candidate.total==total && candidate.coreCount==core &&
                   candidate.downCount==(core?down:total) && candidate.upCount==(core?up:0));
                OK(candidate.firstCore==firstCore && candidate.lastCore==lastCore &&
                   candidate.firstCoreId==(firstCore?firstCore->hybridId:0) &&
                   candidate.lastCoreId==(lastCore?lastCore->hybridId:0));
                OK(candidate.head==(firstCore?firstCore->hybridSlot:-1) &&
                   candidate.tail==(lastCore?lastCore->hybridSlot:-1) &&
                   candidate.orient==draft.orient && candidate.guard==draft.guard);
                OK(candidate.scanBoundaryVisits==(uint64_t)(total-core));
                OK(candidate.scanCoreVisits==0);
        OK(candidate.captureCoreRows==(uint64_t)core);
        OK(candidate.captureTrusted==(core?1:0));
                OK(MSX.FirstSeg[1]==first && MSX.LastSeg[1]==last &&
                   MSX.Link[1].nsegs==count && MSXsegStorage_hybridCoreCount(1)==coreBefore);
            }
            MSXsegStorage_hybridInvalidatePartitionView(1);
            for(backend=MSX_SNAPSHOT_SERIAL_FULL;backend<=MSX_SNAPSHOT_ENDPOINT_BOUNDARY;++backend) {
                OK(MSXsegStorage_hybridCaptureSnapshotBackend(1,backend,&candidate)==0);
                OK(candidate.scanCoreVisits==(uint64_t)core && !candidate.captureTrusted);
                OK(candidate.captureCoreRows==(uint64_t)core && candidate.scanBoundaryVisits==(uint64_t)(total-core));
                OK(candidate.firstCore==firstCore && candidate.lastCore==lastCore && candidate.total==total);
            }
            OK(MSXsegStorage_hybridCaptureSnapshotBackend(1,MSX_SNAPSHOT_ENDPOINT_BOUNDARY,&candidate)==0);
            OK(candidate.captureBackend==MSX_SNAPSHOT_SERIAL_FULL && candidate.scanCoreVisits==(uint64_t)core);
            OK(MSXsegStorage_hybridCaptureSnapshotBackend(1,-1,&candidate)==ERR_PIPE_RING_CAPACITY);
            if(first) {
                Pseg savedNext=first->next;
                first->next=first; /* hard contradiction before any publish */
                for(backend=MSX_SNAPSHOT_SERIAL_FULL;backend<=MSX_SNAPSHOT_ENDPOINT_BOUNDARY;++backend)
                    OK(MSXsegStorage_hybridCaptureSnapshotBackend(1,backend,&candidate)==ERR_PIPE_RING_CAPACITY);
                first->next=savedNext;
            }
        }
    }
}

static void t_rebalance_snapshot_parallel_batch(void)
{
    MSXHybridRebalanceSnapshot reference[34], draft[34];
    Pseg first[34], last[34], q, previous;
    int k, i, backend;
#ifdef _OPENMP
    omp_set_dynamic(0); omp_set_num_threads(8);
#endif
    setup(1,33); MSX.SegmentStorage=SEG_STORAGE_HYBRID;
    OK(MSXsegStorage_open()==0);
    for(k=1;k<=33;++k) {
        previous=NULL;
        for(i=0;i<(k%16);++i) {
            q=fseg(i+1,(uint64_t)(100000+k*100+i)); OK(q!=NULL);
            q->next=previous;
            if(previous) previous->prev=q; else MSX.FirstSeg[k]=q;
            previous=q; MSX.LastSeg[k]=q; ++MSX.Link[k].nsegs;
        }
    }
    OK(MSXsegStorage_hybridizeAll()==0);
    for(k=1;k<=33;++k) {
        first[k]=MSX.FirstSeg[k]; last[k]=MSX.LastSeg[k];
        OK(MSXsegStorage_hybridCaptureSnapshotReadOnly(k,&reference[k])==0);
        OK(MSXsegStorage_hybridRefreshPartitionView(k)==0);
    }
    for(backend=MSX_SNAPSHOT_SERIAL_FULL;backend<=MSX_SNAPSHOT_ENDPOINT_BOUNDARY;++backend) {
        int savedError=MSX.ErrCode;
        OK(MSXsegStorage_hybridCaptureSnapshotsReadOnly(backend,draft,34)==0);
        for(k=1;k<=33;++k) {
            MSXHybridRebalanceSnapshot *a=&reference[k], *b=&draft[k];
            uint64_t oracleIds[32], selectedIds[32];
            unsigned oracleN=0, selectedN=0;
            OK(b->captureStatus==0 && a->linkIndex==b->linkIndex && a->total==b->total &&
               a->coreCount==b->coreCount && a->downCount==b->downCount && a->upCount==b->upCount);
            OK(a->head==b->head && a->tail==b->tail && a->orient==b->orient && a->guard==b->guard &&
               a->firstCore==b->firstCore && a->lastCore==b->lastCore &&
               a->firstCoreId==b->firstCoreId && a->lastCoreId==b->lastCoreId);
            /* Independent CPU object order vs draft-selected migration ends. */
            for(q=first[k];q;q=q->prev) if(!q->inHybridCore) oracleIds[oracleN++]=q->hybridId;
            for(q=first[k];q&&q!=b->firstCore;q=q->prev) selectedIds[selectedN++]=q->hybridId;
            if(b->lastCore) for(q=b->lastCore->prev;q;q=q->prev) selectedIds[selectedN++]=q->hybridId;
            OK(selectedN==oracleN && !memcmp(selectedIds,oracleIds,oracleN*sizeof(uint64_t)));
            OK(MSX.FirstSeg[k]==first[k] && MSX.LastSeg[k]==last[k]);
        }
        OK(MSX.ErrCode==savedError);
    }
    OK(MSXsegStorage_hybridCaptureSnapshotsReadOnly(MSX_SNAPSHOT_OMP_STATIC,draft,33)==ERR_PIPE_RING_CAPACITY);
    {
        Pseg saved3=first[3]->next, saved9=first[9]->next;
        int savedError=MSX.ErrCode;
        first[3]->next=first[3]; first[9]->next=first[9];
        for(backend=MSX_SNAPSHOT_SERIAL_FULL;backend<=MSX_SNAPSHOT_ENDPOINT_BOUNDARY;++backend) {
            MSXsegStorage_hybridInvalidatePartitionView(3);
            MSXsegStorage_hybridInvalidatePartitionView(9);
            OK(MSXsegStorage_hybridCaptureSnapshotsReadOnly(backend,draft,34)==0);
            OK(draft[3].captureStatus==ERR_PIPE_RING_CAPACITY && draft[9].captureStatus==ERR_PIPE_RING_CAPACITY);
            OK(draft[1].captureStatus==0 && draft[2].captureStatus==0 && draft[4].captureStatus==0);
            OK(MSX.ErrCode==savedError);
        }
        first[3]->next=saved3; first[9]->next=saved9;
    }
}
