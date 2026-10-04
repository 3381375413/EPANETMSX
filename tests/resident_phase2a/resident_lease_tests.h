static void t_handoff_leases(void)
{
    MSXResidentHandoffPlan p;MSXResidentHandoffItem item;
    MSXResidentHandoffResult r;MSXResidentHandoffTarget target;
    MSXResidentHandoffTransaction tx={0};MSXResidentPayload payload;
    Pseg first,last;int n;uint64_t calls;
    if(!initialFixture()){OK(0);return;}
    OK(MSXsegStorage_hybridPrepareInitialImage()==0&&MSXsegStorage_hybridStageInitialImage()==0);
    OK(MSXsegStorage_hybridCommitInitialImage()==0&&MSXresident_commitInitialImage()==0);
    OK(MSXresident_planAllHandoffInto(1,0,&item,1,&p)==0&&p.itemCount==1);
    OK(MSXresident_getSlotPayload(1,item.slot,item.generation,&payload)==0);
    first=MSX.FirstSeg[1];last=MSX.LastSeg[1];n=MSX.Link[1].nsegs;
    calls=MSXresident_testAllocationCount();
    OK(MSXresident_prepareHandoffLeases(&p,&r,&target,&tx)==0&&target.c&&target.lastc);
    OK(MSXresident_validateHandoffTransactions(&tx,1)==MSX_RESIDENT_ERR_GENERATION);
    /* Simulate a partial download then abort twice. No topology is published. */
    target.c[0]=123;
    MSXresident_abortHandoffTransaction(&tx);MSXresident_abortHandoffTransaction(&tx);
    OK(MSX.FirstSeg[1]==first&&MSX.LastSeg[1]==last&&MSX.Link[1].nsegs==n&&MSXsegStorage_hybridCoreCount(1)==1);
    OK(MSXresident_prepareHandoffLeases(&p,&r,&target,&tx)==0);
    memcpy(target.c,payload.c,2*sizeof(double));memcpy(target.lastc,payload.lastc,2*sizeof(double));
    r.payload=payload;r.payload.c=target.c;r.payload.lastc=target.lastc;
    OK(MSXresident_finishHandoffLeases(&tx,&r)==0&&MSXresident_validateHandoffTransactions(&tx,1)==0);
    OK(MSXresident_testAllocationCount()==calls);
    MSXresident_commitHandoffTransaction(&tx);
    OK(MSXsegStorage_hybridCoreCount(1)==0&&MSX.Link[1].nsegs==n&&MSX.FirstSeg[1]==first&&MSX.LastSeg[1]==last);
    MSXresident_abortHandoffTransaction(&tx);
}
