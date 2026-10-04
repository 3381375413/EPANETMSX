#include "msxresident_budget.h"
static void t_budget_backing(void)
{
    uint32_t cap[3]={0,2,2},base[3]={0,0,2};
    MSXResidentGpuOpen o={2,4,S,cap,base};
    MSXResidentMemoryEstimate m;
    MSXResidentBudget *b=MSXresidentBudget_global(),state;
    MSXResidentGpu *g=NULL;
    MSXBudgetTicket preexisting={0};
    uint64_t host,device;
    OK(MSXresidentGpu_estimateMemory(2,4,S,&m)==MSX_RESIDENT_OK);
    host=m.hostBytes+m.pinnedBytes;device=m.deviceBytes;
    OK(MSXresidentBudget_configure(b,host,device)==MSX_BUDGET_OK);
    OK(MSXresidentGpu_open(&o,&g)==MSX_RESIDENT_OK);
    MSXresidentBudget_snapshot(b,&state);
    OK(state.allocated[0]==m.hostBytes && state.allocated[1]==m.pinnedBytes && state.allocated[2]==device);
    MSXresidentGpu_close(g);g=NULL;MSXresidentBudget_snapshot(b,&state);
    OK(!state.allocated[0]&&!state.allocated[1]&&!state.allocated[2]);
    /* Refusal in pageable/pinned/device midway must release the already
       acquired arena and cancel only the unsuccessful reservation. */
    for(int i=0;i<4;++i){
        uint64_t h=i==0?host/2:i==1?host-1:host;
        uint64_t d=i==2?device/2:i==3?device-1:device;
        OK(MSXresidentBudget_configure(b,h,d)==MSX_BUDGET_OK);
        OK(MSXresidentGpu_open(&o,&g)==MSX_RESIDENT_ERR_MEMORY&&g==NULL);
        MSXresidentBudget_snapshot(b,&state);
        OK(!state.allocated[0]&&!state.allocated[1]&&!state.allocated[2]&&
           !state.reserved[0]&&!state.reserved[1]&&!state.reserved[2]);
    }
    OK(MSXresidentBudget_configure(b,host,device)==MSX_BUDGET_OK);
    OK(MSXresidentBudget_reserve(b,MSX_MEMORY_PAGEABLE,1,&preexisting)==MSX_BUDGET_OK);
    OK(MSXresidentGpu_open(&o,&g)==MSX_RESIDENT_ERR_MEMORY&&g==NULL);
    MSXresidentBudget_snapshot(b,&state);
    OK(state.reserved[0]==1&&!state.allocated[0]&&!state.allocated[1]&&!state.allocated[2]);
    OK(MSXresidentBudget_cancelReservation(b,&preexisting)==MSX_BUDGET_OK);
    OK(MSXresidentBudget_configure(b,UINT64_MAX,UINT64_MAX)==MSX_BUDGET_OK);
}
