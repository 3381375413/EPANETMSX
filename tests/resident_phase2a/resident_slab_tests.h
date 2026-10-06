#include "msxresident_alloc.h"
/* Exercise budget rejection positions. CPU fixture pool objects use its CRT
   allocator; the storage ledger is compared separately. */
static void t_core_slab(void)
{
    uint32_t cap[4]={0,4,4,4},guard[4]={0,2,2,2};
    MSXResidentLayout layout;MSXResidentBudget before,after;
    uint64_t bytes=0,pipeBytes=0,header=MSXresidentAlloc_headerBytes();
    int species,iteration,k;Pseg base;uint32_t n;
    for(species=1;species<=5;species+=2){
        setup(species,3);MSX.SegmentStorage=SEG_STORAGE_HYBRID;MSX.GpuCoreMode=2;
        OK(MSXresident_openPlan(3,cap,guard,UP)==0&&MSXresident_getLayout(&layout)==0);
        OK(MSXsegStorage_open()==0);
        MSXresidentBudget_snapshot(MSXresidentBudget_global(),&before);
        {uint64_t ledger=MSXsegStorage_testAllocationBytes();
         OK(MSXsegStorage_hybridReserve(&layout)==0);
         bytes=MSXsegStorage_testAllocationBytes()-ledger;
         OK(bytes+MSXsegStorage_hybridExistingHostBytes()+
            12*(sizeof(struct Sseg)+(uint64_t)2*(species+1)*sizeof(double))==
            MSXsegStorage_hybridFixedHostBytes(3,12,(uint32_t)species+1));}
        /* Recover the compiled Tx/control contribution from the cost probe;
           charge the actual slot arrays, including their slab header. */
        pipeBytes=MSXsegStorage_hybridFixedHostBytes(1,4,(uint32_t)species+1)-
                  MSXsegStorage_hybridFixedHostBytes(0,4,(uint32_t)species+1)-
                  (MSXsegStorage_hybridExistingHostBytes()-2*header)/4+
                  4*(sizeof(struct Sseg)+5*sizeof(Pseg)+1+3*sizeof(int)+
                     5*sizeof(double)+sizeof(uint64_t)+sizeof(MSXResidentPayload));
        for(k=1;k<=3;k++){
            OK(MSXsegStorage_hybridAuditCoreSlab(k,&base,&n)&&n==4);
            OK(MSXresidentAlloc_chargedBytes(base)==4*sizeof(*base)+header);
            for(iteration=0;iteration<4;iteration++)
                OK(base[iteration].hybridSlot==iteration&&base[iteration].ringSlot==-1&&
                   !base[iteration].inHybridCore&&!base[iteration].c&&!base[iteration].lastc);
        }
        MSXsegStorage_close();MSXsegStorage_close();
        MSXresidentBudget_snapshot(MSXresidentBudget_global(),&after);
        OK(after.allocated[0]+MSXsegStorage_hybridExistingHostBytes()<before.allocated[0]);
    }
    /* Sweep the envelope; every failed partial pipe is closed twice and all
       storage allocations/reservations must return to the pre-open baseline. */
    cleanup();
    for(iteration=0;iteration<=44;iteration++){
        uint64_t allowance=iteration==41?bytes-1:bytes*(uint64_t)iteration/40;
        if(iteration>=42)
            allowance=bytes-(uint64_t)(44-iteration)*pipeBytes-1;
        setup(1,3);MSX.SegmentStorage=SEG_STORAGE_HYBRID;MSX.GpuCoreMode=2;
        OK(MSXresident_openPlan(3,cap,guard,UP)==0&&MSXresident_getLayout(&layout)==0);
        MSXresidentBudget_snapshot(MSXresidentBudget_global(),&before);
        OK(MSXsegStorage_open()==0);
        MSXresidentBudget_snapshot(MSXresidentBudget_global(),&after);
        OK(MSXresidentBudget_configure(MSXresidentBudget_global(),after.allocated[0]+allowance,UINT64_MAX)==MSX_BUDGET_OK);
        {int result=MSXsegStorage_hybridReserve(&layout);
         OK(result==(allowance>=bytes?0:ERR_MEMORY));}
        if(iteration>=42){
            for(k=1;k<=3;k++)
                OK(MSXsegStorage_hybridAuditCoreSlab(k,&base,&n)==(k<iteration-41));
        }
        MSX.GpuCoreMode=0; /* free must use actual ownership, not current mode */
        MSXsegStorage_close();MSXsegStorage_close();
        MSXresidentBudget_snapshot(MSXresidentBudget_global(),&after);
        OK(after.allocated[0]==before.allocated[0]&&after.reserved[0]==before.reserved[0]);
        OK(MSXresidentBudget_configure(MSXresidentBudget_global(),UINT64_MAX,UINT64_MAX)==MSX_BUDGET_OK);
    }
}
