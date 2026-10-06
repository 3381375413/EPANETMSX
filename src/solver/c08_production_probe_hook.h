/* Include at the END of isolated msxsegment_storage.c, never in main.
   Test ABI only. Raw reference export never calls a view/span/snapshot. */
#ifdef MSX_RESIDENT_TEST_API
extern void MSXqual_reversesegs(int k);
__declspec(dllexport) int MSXTESTc08Raw(const char *path,int selected)
{
    FILE *f;int k,ordinal,start=selected?selected:1,end=selected?selected:MSX.Nobjects[LINK];
    Pseg q;
    if(!path||start<1||end>MSX.Nobjects[LINK]||!MSX.FirstSeg||!MSX.LastSeg)return 525;
    f=fopen(path,"w");if(!f)return 1;fprintf(f,"H,1\n");
    for(k=start;k<=end;k++) {
        fprintf(f,"P,%d,%d,%llu,%llu\n",k,MSX.Link[k].nsegs,
            (unsigned long long)(uintptr_t)MSX.FirstSeg[k],(unsigned long long)(uintptr_t)MSX.LastSeg[k]);
        for(q=MSX.FirstSeg[k],ordinal=0;q;q=q->prev,ordinal++) {
            if(ordinal>MSX.Link[k].nsegs||MSX.Link[k].nsegs<0){fclose(f);return 525;}
            fprintf(f,"R,%d,%d,%llu,%llu,%llu,%d,%d,%llu,%d\n",k,ordinal,
                (unsigned long long)(uintptr_t)q,(unsigned long long)(uintptr_t)q->next,
                (unsigned long long)(uintptr_t)q->prev,q->inHybridCore,q->ownerLink,
                (unsigned long long)q->hybridId,q->hybridSlot);
        }
    }
    return fclose(f)?1:0;
}
__declspec(dllexport) int MSXTESTc08Views(const char *path,int selected)
{
    FILE *f;int k,start=selected?selected:1,end=selected?selected:MSX.Nobjects[LINK];
    MSXResidentPipeDesc descriptor;uint32_t slot,generation,used;uint64_t id;MSXResidentStatus status;
    const MSXHybridPartitionView *v;
    if(!path||start<1||end>MSX.Nobjects[LINK])return 525;
    f=fopen(path,"w");if(!f)return 1;fprintf(f,"H,1\n");
    for(k=start;k<=end;k++) {
        if(!MSXsegStorage_isHybridLink(k))continue;
        if(MSXsegStorage_hybridGetPartitionView(k,&v)){fclose(f);return 525;}
        fprintf(f,"V,%d,%d,%d,%llu,%llu,%d,%llu,%llu,%llu,%llu,%d,%d,%d\n",k,
            v->valid,v->continuityProven,(unsigned long long)v->lifecycle,(unsigned long long)v->revision,
            v->coreCount,(unsigned long long)(uintptr_t)v->firstCore,(unsigned long long)(uintptr_t)v->lastCore,
            (unsigned long long)v->firstCoreId,(unsigned long long)v->lastCoreId,v->head,v->tail,v->orient);
        status=MSXresident_testInitialDescriptor((uint32_t)k,&descriptor);
        if(status){fprintf(f,"E,%d,%d\n",k,(int)status);continue;}
        fprintf(f,"D,%d,%u,%u,%u,%u,%d,%llu\n",k,descriptor.capacity,descriptor.count,
            descriptor.head,descriptor.tail,descriptor.orient,(unsigned long long)descriptor.epoch);
        for(slot=0;slot<descriptor.capacity;slot++) {
            status=MSXresident_testInitialSlot((uint32_t)k,slot,&used,&generation,&id);
            if(status){fclose(f);return 525;}
            if(used)fprintf(f,"I,%d,%u,%u,%llu\n",k,slot,generation,(unsigned long long)id);
        }
    }
    return fclose(f)?1:0;
}
/* op1 CPU append; 2 physical head remove; 3 adjacent promote; 4 endpoint
   demote; 5 Qual reverse; 6 clear; 7 full proof rebuild; 8 real production
   RebalanceAll round. side0=down/head. */
__declspec(dllexport) int MSXTESTc08Operation(int k,int op,int side)
{
    HybridPipe *p;Pseg q;int error,atHead=side==0;
    if(!MSXsegStorage_isHybridLink(k)||op<1||op>8||side<0||side>1)return 525;
    if(MSXsegStorage_hybridWriteAllowed())return 525;
    if(MSX.ErrCode)return MSX.ErrCode;
    p=&Hybrid.pipe[k];
    if(op==1) {
        q=MSX.LastSeg[k];if(!q||q->inHybridCore)return 525;
        q=MSXqual_getFreeSeg(1.0,q->c);if(!q)return ERR_MEMORY;
        MSXsegStorage_hybridAssignIdentity(k,q);
        if(MSX.ErrCode){error=MSX.ErrCode;MSXqual_removeSeg(q);return error;}
        error=MSXsegStorage_hybridAppendCpuBoundary(k,q);
        if(error)MSXqual_removeSeg(q);return error;
    }
    if(op==2) {
        q=MSX.FirstSeg[k];if(!q)return 525;
        if(q->inHybridCore)return MSXsegStorage_hybridRemoveHead(k,q);
        error=MSXsegStorage_hybridRemoveCpuHead(k,q);
        if(!error)MSXqual_removeSeg(q);return error;
    }
    if(op==3) {
        if(!p->count)return 525;
        q=atHead?p->view[p->head]->next:p->view[p->tail]->prev;
        if(!q||q->inHybridCore)return 525;
        return hybridPromote(k,q,atHead);
    }
    if(op==4){if(!p->count)return 525;return hybridDemote(k,atHead);}
    if(op==5){MSXqual_reversesegs(k);if(MSX.ErrCode)return MSX.ErrCode;
        return MSXsegStorage_hybridAfterListReorder(k);}
    if(op==6){MSXsegStorage_hybridClear(k);return MSX.ErrCode;}
    if(op==8){MSXsegStorage_hybridRebalanceAll();return MSX.ErrCode;}
    return MSXsegStorage_hybridRefreshPartitionView(k);
}
__declspec(dllexport) int MSXTESTc08Metrics(uint64_t *values)
{
    if(!values)return 1;
    int k;
    MSXsegStorage_hybridIncrementalMetrics(&values[0],&values[1]);
    /* Last actual production capture outputs, not a diagnostic recapture.
       Capture row counts survive later serial migration of snapshot counts. */
    values[2]=values[3]=values[4]=values[5]=values[6]=values[7]=0;
    if(Hybrid.rebalanceSnapshot)for(k=1;k<=Hybrid.nLinks;++k) {
        const RebalanceSnapshot *s=&Hybrid.rebalanceSnapshot[k];
        if(s->linkIndex!=k||s->captureStatus)continue;
        values[2]+=s->scanCoreVisits;values[3]+=s->scanBoundaryVisits;
        values[4]+=s->captureTrusted?1:0;values[5]+=s->captureTrusted?0:1;
        values[6]+=s->captureCoreRows;
        if(s->captureTrusted)values[7]+=s->captureCoreRows;
    }
    return 0;
}
#endif
