#include <stdio.h>
#include <string.h>
#include <windows.h>
#include "msxresident_budget.h"
static int failures;
#define CHECK(x) do { if(!(x)) {fprintf(stderr,"line %d: %s\n",__LINE__,#x);++failures;} } while(0)
static void checkSummary(const char *path,const char *status,const char *limits,const char *peak)
{
    char text[8192];FILE *f=fopen(path,"rb");size_t size=0;
    CHECK(f!=NULL);if(!f)return;size=fread(text,1,sizeof(text)-1,f);text[size]=0;fclose(f);
    CHECK(strstr(text,status)!=NULL);CHECK(strstr(text,limits)!=NULL);CHECK(strstr(text,peak)!=NULL);
    CHECK(strstr(text,"\"memory_class_order\":[\"pageable\",\"pinned\",\"device\"]")!=NULL);
    CHECK(strstr(text,"\"domains\":[{\"name\":\"cpu_pool\"")!=NULL);
}
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
    {
        MSXResidentBudget child,sibling;
        MSXresidentBudget_init(&b);CHECK(!MSXresidentBudget_configure(&b,100,20));
        CHECK(!MSXresidentBudget_initChild(&child,&b));
        CHECK(!MSXresidentBudget_initChild(&sibling,&b));
        CHECK(MSXresidentBudget_initChild(&other,&child)==MSX_BUDGET_INVALID);
        CHECK(!MSXresidentBudget_configure(&child,60,20));
        CHECK(!MSXresidentBudget_reserve(&child,MSX_MEMORY_PAGEABLE,60,&t));
        CHECK(MSXresidentBudget_reserve(&child,MSX_MEMORY_PINNED,1,&p)==MSX_BUDGET_LIMIT);
        CHECK(!MSXresidentBudget_reserve(&sibling,MSX_MEMORY_PINNED,40,&p));
        CHECK(MSXresidentBudget_reserve(&b,MSX_MEMORY_PAGEABLE,1,&d)==MSX_BUDGET_LIMIT);
        MSXresidentBudget_snapshot(&b,&snap);CHECK(snap.reserved[0]+snap.reserved[1]==100);
        CHECK(!MSXresidentBudget_commitAllocation(&child,&t,59,&a));
        MSXresidentBudget_snapshot(&b,&snap);CHECK(snap.allocated[0]==59 && snap.reserved[0]==0);
        CHECK(!MSXresidentBudget_releaseAllocation(&child,&a));
        CHECK(!MSXresidentBudget_cancelReservation(&sibling,&p));
        MSXresidentBudget_snapshot(&b,&snap);CHECK(!snap.allocated[0]&&!snap.reserved[0]&&!snap.reserved[1]);
        CHECK(!MSXresidentBudget_configure(&b,8,1));worker.b=&child;worker.errors=0;
        for(k=0;k<8;++k)threads[k]=CreateThread(NULL,0,exercise,&worker,0,NULL);
        CHECK(WaitForMultipleObjects(8,threads,TRUE,INFINITE)==WAIT_OBJECT_0);
        for(k=0;k<8;++k)CloseHandle(threads[k]);
        CHECK(worker.errors==0);MSXresidentBudget_snapshot(&b,&snap);
        CHECK(!snap.allocated[0]&&!snap.reserved[0]);
    }
    printf("resident_budget_tests failures=%d concurrent_operations=80000\n",failures);
    {
        MSXResidentBudget child;MSXBudgetTicket whole={0},part={0};MSXBudgetAllocation allocation={0};
        MSXresidentBudget_init(&b);MSXresidentBudget_initChild(&child,&b);
        CHECK(!MSXresidentBudget_configure(&b,100,100));CHECK(!MSXresidentBudget_configure(&child,60,100));
        CHECK(!MSXresidentBudget_reserve(&b,MSX_MEMORY_PAGEABLE,100,&whole));
        CHECK(MSXresidentBudget_split(&whole,&child,61,&part)==MSX_BUDGET_LIMIT);
        CHECK(whole.bytes==100&&!part.live);
        CHECK(!MSXresidentBudget_split(&whole,&child,60,&part));
        MSXresidentBudget_snapshot(&b,&snap);CHECK(snap.reserved[0]==100&&whole.bytes==40);
        CHECK(!MSXresidentBudget_commitAllocation(&child,&part,60,&allocation));
        CHECK(!MSXresidentBudget_split(&whole,&b,40,&part));CHECK(!whole.live);
        CHECK(!MSXresidentBudget_cancelReservation(&b,&part));CHECK(!MSXresidentBudget_releaseAllocation(&child,&allocation));
        MSXresidentBudget_snapshot(&b,&snap);CHECK(!snap.allocated[0]&&!snap.reserved[0]&&snap.peakHostBytes==100);
    }
    {
        MSXResidentBudget *root=MSXresidentBudget_global(),*child=MSXresidentBudget_domain(MSX_BUDGET_CPU_POOL);
        MSXBudgetTicket ticket={0},pin={0},device={0},rejected={0};MSXBudgetAllocation held={0};uint64_t sequence;
        CHECK(!MSXresidentBudget_beginSession());
        CHECK(!MSXresidentBudget_configure(root,100,20));CHECK(!MSXresidentBudget_configure(child,60,20));
        CHECK(!MSXresidentBudget_reserve(child,MSX_MEMORY_PAGEABLE,60,&ticket));
        CHECK(!MSXresidentBudget_reserve(root,MSX_MEMORY_PINNED,40,&pin));
        CHECK(!MSXresidentBudget_reserve(root,MSX_MEMORY_DEVICE,20,&device));
        CHECK(MSXresidentBudget_reserve(child,MSX_MEMORY_PAGEABLE,1,&rejected)==MSX_BUDGET_LIMIT);
        CHECK(MSXresidentBudget_reserve(root,MSX_MEMORY_PAGEABLE,1,&rejected)==MSX_BUDGET_LIMIT);
        CHECK(MSXresidentBudget_beginSession()==MSX_BUDGET_STATE);
        CHECK(!MSXresidentBudget_cancelReservation(child,&ticket));
        CHECK(!MSXresidentBudget_cancelReservation(root,&pin));CHECK(!MSXresidentBudget_cancelReservation(root,&device));
        MSXresidentBudget_snapshot(root,&snap);sequence=snap.nextSequence;
        CHECK(MSXresidentBudget_endSession("budget_session_one.json"));
        checkSummary("budget_session_one.json","\"status\":\"PASS\"","\"host_limit_bytes\":100,\"device_limit_bytes\":20","\"peak_host_including_reservations\":100");
        CHECK(MSXresidentBudget_endSession("budget_session_one.json"));
        CHECK(!MSXresidentBudget_beginSession());CHECK(!MSXresidentBudget_configure(root,10,2));
        CHECK(!MSXresidentBudget_configure(child,8,2));
        MSXresidentBudget_snapshot(root,&snap);CHECK(!snap.peakHostBytes&&!snap.peakDeviceBytes&&!snap.rejectedReservations&&snap.nextSequence==sequence);
        MSXresidentBudget_snapshot(child,&snap);CHECK(!snap.peakHostBytes&&!snap.rejectedReservations);
        CHECK(!MSXresidentBudget_reserve(child,MSX_MEMORY_PAGEABLE,7,&ticket));CHECK(ticket.sequence>sequence);
        CHECK(!MSXresidentBudget_commitAllocation(child,&ticket,7,&held));
        CHECK(MSXresidentBudget_endSession("budget_session_residue.json"));
        checkSummary("budget_session_residue.json","\"status\":\"FAIL\"","\"host_limit_bytes\":10,\"device_limit_bytes\":2","\"final_allocated\":[7,0,0]");
        CHECK(MSXresidentBudget_beginSession()==MSX_BUDGET_STATE);
        MSXresidentBudget_snapshot(root,&snap);CHECK(snap.allocated[0]==7&&snap.peakHostBytes==7);
        CHECK(!MSXresidentBudget_releaseAllocation(child,&held));CHECK(!MSXresidentBudget_beginSession());
        CHECK(!MSXresidentBudget_reserve(root,MSX_MEMORY_PINNED,3,&pin));CHECK(!MSXresidentBudget_cancelReservation(root,&pin));
        CHECK(MSXresidentBudget_endSession("budget_session_two.json"));
        checkSummary("budget_session_two.json","\"status\":\"PASS\"","\"host_limit_bytes\":10,\"device_limit_bytes\":2","\"peak_host_including_reservations\":3");
        /* A lower final limit must not hide an earlier high water mark. */
        CHECK(!MSXresidentBudget_beginSession());CHECK(!MSXresidentBudget_reserve(root,MSX_MEMORY_PAGEABLE,8,&ticket));
        CHECK(!MSXresidentBudget_cancelReservation(root,&ticket));CHECK(!MSXresidentBudget_configure(root,7,2));
        CHECK(MSXresidentBudget_endSession("budget_session_peak_fail.json"));
        checkSummary("budget_session_peak_fail.json","\"status\":\"FAIL\"","\"host_limit_bytes\":7,\"device_limit_bytes\":2","\"peak_exceeds_final_limit\":true");
        CHECK(!MSXresidentBudget_configure(root,UINT64_MAX,UINT64_MAX));
        CHECK(!MSXresidentBudget_configure(child,UINT64_MAX,UINT64_MAX));
    }
    printf("resident_budget_session_tests failures=%d\n",failures);
    return failures?1:0;
}
