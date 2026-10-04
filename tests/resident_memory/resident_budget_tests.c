#include <stdio.h>
#include <string.h>
#include <windows.h>
#include "msxresident_budget.h"
static int failures;
#define CHECK(x) do { if(!(x)) {fprintf(stderr,"line %d: %s\n",__LINE__,#x);++failures;} } while(0)
typedef struct { MSXResidentBudget *b; LONG errors; } Worker;
static DWORD WINAPI exercise(void *arg)
{
    Worker *w=(Worker*)arg;
    int k;
    for(k=0;k<10000;++k)
    {
        MSXBudgetTicket t={0}; MSXBudgetAllocation a={0};
        if(MSXresidentBudget_reserve(w->b,MSX_MEMORY_PAGEABLE,1,&t)||
           MSXresidentBudget_commitAllocation(w->b,&t,1,&a)||
           MSXresidentBudget_releaseAllocation(w->b,&a))InterlockedIncrement(&w->errors);
    }
    return 0;
}
int main(void)
{
    MSXResidentBudget b, other, snap;
    MSXBudgetTicket t={0}, p={0}, d={0}; MSXBudgetAllocation a={0};
    Worker worker; HANDLE threads[8]; int k;
    MSXresidentBudget_init(&b); MSXresidentBudget_init(&other);
    CHECK(!MSXresidentBudget_configure(&b,100,20));
    CHECK(!MSXresidentBudget_reserve(&b,MSX_MEMORY_PAGEABLE,60,&t));
    CHECK(!MSXresidentBudget_reserve(&b,MSX_MEMORY_PINNED,40,&p));
    CHECK(MSXresidentBudget_reserve(&b,MSX_MEMORY_PAGEABLE,1,&d)==MSX_BUDGET_LIMIT);
    CHECK(MSXresidentBudget_reserve(&b,MSX_MEMORY_PINNED,1,&p)==MSX_BUDGET_STATE);
    CHECK(MSXresidentBudget_configure(&b,99,20)==MSX_BUDGET_LIMIT);
    CHECK(MSXresidentBudget_cancelReservation(&other,&t)==MSX_BUDGET_STATE);
    CHECK(MSXresidentBudget_commitAllocation(&b,&t,61,&a)==MSX_BUDGET_STATE);
    CHECK(!MSXresidentBudget_commitAllocation(&b,&t,59,&a));
    MSXresidentBudget_snapshot(&b,&snap);
    CHECK(snap.allocated[0]==59 && snap.reserved[0]==0 && snap.reserved[1]==40);
    CHECK(snap.peakHostBytes==100);
    CHECK(!MSXresidentBudget_reserve(&b,MSX_MEMORY_DEVICE,20,&d));
    CHECK(MSXresidentBudget_releaseAllocation(&other,&a)==MSX_BUDGET_STATE);
    CHECK(!MSXresidentBudget_releaseAllocation(&b,&a));
    CHECK(MSXresidentBudget_releaseAllocation(&b,&a)==MSX_BUDGET_STATE);
    CHECK(!MSXresidentBudget_cancelReservation(&b,&p));
    CHECK(MSXresidentBudget_cancelReservation(&b,&p)==MSX_BUDGET_STATE);
    CHECK(!MSXresidentBudget_cancelReservation(&b,&d));
    CHECK(!MSXresidentBudget_configure(&b,UINT64_MAX,UINT64_MAX));
    CHECK(!MSXresidentBudget_reserve(&b,MSX_MEMORY_PAGEABLE,UINT64_MAX,&t));
    CHECK(MSXresidentBudget_reserve(&b,MSX_MEMORY_PINNED,1,&p)==MSX_BUDGET_LIMIT);
    CHECK(!MSXresidentBudget_cancelReservation(&b,&t));
    CHECK(MSXresidentBudget_reserve(&b,(MSXMemoryClass)99,1,&t)==MSX_BUDGET_INVALID);
    CHECK(MSXresidentBudget_reserve(&b,MSX_MEMORY_PAGEABLE,0,&t)==MSX_BUDGET_INVALID);
    CHECK(!MSXresidentBudget_configure(&b,8,1)); worker.b=&b;worker.errors=0;
    for(k=0;k<8;++k) {threads[k]=CreateThread(NULL,0,exercise,&worker,0,NULL);CHECK(threads[k]!=NULL);}
    CHECK(WaitForMultipleObjects(8,threads,TRUE,INFINITE)==WAIT_OBJECT_0);
    for(k=0;k<8;++k)CloseHandle(threads[k]);
    CHECK(worker.errors==0); MSXresidentBudget_snapshot(&b,&snap);
    CHECK(snap.allocated[0]==0 && snap.reserved[0]==0);
    printf("resident_budget_tests failures=%d concurrent_operations=80000\n",failures);
    return failures?1:0;
}
