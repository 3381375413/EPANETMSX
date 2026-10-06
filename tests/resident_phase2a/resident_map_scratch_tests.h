/* C09a: the explicit observer must reject metadata-only mode before it can
   consume its omitted map scratch. Reopen must reconstruct legacy scratch. */
static void t_map_scratch_modes(void)
{
    uint32_t limits[3]={0,3,5},guards[3]={0,2,2};MSXResidentLayout layout;
    MSXResidentBudget before,after;uint64_t allocation,existing;int species;
    for(species=1;species<=3;species+=2){
        setup(species,2);MSX.SegmentStorage=SEG_STORAGE_HYBRID;MSX.GpuCoreMode=2;
        OK(MSXresident_openPlan(2,limits,guards,UP)==0&&MSXresident_getLayout(&layout)==0);
        MSXresident_setMode(MSX_RESIDENT_RESIDENT,1);
        allocation=MSXsegStorage_testAllocationBytes();
        OK(MSXsegStorage_open()==0);
        existing=MSXsegStorage_hybridExistingHostBytes();
        OK(MSXsegStorage_hybridReserve(&layout)==0);
        OK(MSXsegStorage_testAllocationBytes()-allocation+
           8*(sizeof(struct Sseg)+(uint64_t)2*(species+1)*sizeof(double))==
           MSXsegStorage_hybridFixedHostBytes(2,8,(uint32_t)species+1));
        MSXresidentBudget_snapshot(MSXresidentBudget_global(),&before);
        OK(MSXsegStorage_hybridObserveAll()==ERR_PIPE_RING_CAPACITY);
        OK(MSXsegStorage_residentLastStatus()==MSX_RESIDENT_ERR_ARGUMENT);
        OK(MSXsegStorage_hybridCoreCount(1)==0&&MSXsegStorage_hybridCoreCount(2)==0);
        MSX.GpuCoreMode=0;
        OK(MSXsegStorage_hybridObserveAll()==ERR_PIPE_RING_CAPACITY);
        MSXresidentBudget_snapshot(MSXresidentBudget_global(),&after);
        OK(after.allocated[0]==before.allocated[0]&&after.reserved[0]==before.reserved[0]);
        MSXsegStorage_close();MSXsegStorage_close();
        OK(existing>0);
        OK(fixture());
        OK(MSXsegStorage_hybridObserveAll()==0);
        {Pseg s=MSXsegStorage_hybridCoreSegFromHead(1,0);
         OK(s&&MSXsegStorage_isHybridCoreIdentity(1,s->hybridId));}
    }
}
