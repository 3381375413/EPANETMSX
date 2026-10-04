#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#define WIN32_LEAN_AND_MEAN
#define NOGDI
#include <windows.h>
#include "msxresident_budget.h"
static MSXResidentBudget globalBudget = { UINT64_MAX, UINT64_MAX };
static FILE *lifetimeAudit, *allocationAudit;
static int memoryAuditMode=-1;
static void enter(MSXResidentBudget *b)
{ while (InterlockedCompareExchange((volatile LONG *)&b->lock,1,0)) SwitchToThread(); }
static void leave(MSXResidentBudget *b)
{ InterlockedExchange((volatile LONG *)&b->lock,0); }
/* All changes maintain nonoverflowing allocated+reserved totals. */
static uint64_t total(MSXResidentBudget *b, unsigned c)
{ return b->allocated[c]+b->reserved[c]; }
static uint64_t hostTotal(MSXResidentBudget *b)
{ return total(b,MSX_MEMORY_PAGEABLE)+total(b,MSX_MEMORY_PINNED); }
/* Diagnostic-only I/O.  Events are serialized with the budget transition,
   including pool reservations, so peaks describe simultaneously live bytes. */
static int auditEnabled(void)
{ if(memoryAuditMode<0){const char *v=getenv("MSX_RESIDENT_MEMORY_AUDIT");memoryAuditMode=v && !strcmp(v,"1");}return memoryAuditMode; }
static void audit(MSXResidentBudget *b,const char *event,uint64_t sequence,unsigned c,uint64_t bytes)
{
    FILE *f;
    if(b!=&globalBudget || !auditEnabled())return;
    if(!lifetimeAudit){lifetimeAudit=fopen("resident_budget_lifetime.csv","a+");
        if(lifetimeAudit){fseek(lifetimeAudit,0,SEEK_END);
            if(!ftell(lifetimeAudit))fprintf(lifetimeAudit,"event,sequence,memory_class,bytes,pageable_allocated,pinned_allocated,device_allocated,pageable_reserved,pinned_reserved,device_reserved,host_total,device_total,host_limit,device_limit\n");}}
    f=lifetimeAudit;if(!f)return;
    fprintf(f,"%s,%llu,%u,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n",event,
        (unsigned long long)sequence,c,(unsigned long long)bytes,
        (unsigned long long)b->allocated[0],(unsigned long long)b->allocated[1],(unsigned long long)b->allocated[2],
        (unsigned long long)b->reserved[0],(unsigned long long)b->reserved[1],(unsigned long long)b->reserved[2],
        (unsigned long long)hostTotal(b),(unsigned long long)total(b,MSX_MEMORY_DEVICE),
        (unsigned long long)b->hostLimit,(unsigned long long)b->deviceLimit);
    if(!hostTotal(b)&&!total(b,MSX_MEMORY_DEVICE)){
        fclose(lifetimeAudit);lifetimeAudit=NULL;
        if(allocationAudit){fclose(allocationAudit);allocationAudit=NULL;}}
}
static void peak(MSXResidentBudget *b)
{
    uint64_t h=hostTotal(b), d=total(b,MSX_MEMORY_DEVICE);
    if(h>b->peakHostBytes)b->peakHostBytes=h;
    if(d>b->peakDeviceBytes)b->peakDeviceBytes=d;
}
void MSXresidentBudget_init(MSXResidentBudget *b)
{ memset(b,0,sizeof(*b)); b->hostLimit=b->deviceLimit=UINT64_MAX; }
MSXResidentBudget *MSXresidentBudget_global(void) { return &globalBudget; }
MSXBudgetStatus MSXresidentBudget_configure(MSXResidentBudget *b,uint64_t host,uint64_t device)
{
    MSXBudgetStatus s=MSX_BUDGET_OK;
    if(!b)return MSX_BUDGET_INVALID;
    enter(b);
    if(host<hostTotal(b)||device<total(b,MSX_MEMORY_DEVICE))s=MSX_BUDGET_LIMIT;
    else { b->hostLimit=host; b->deviceLimit=device; }
    leave(b); return s;
}
MSXBudgetStatus MSXresidentBudget_reserve(MSXResidentBudget *b,MSXMemoryClass c,uint64_t bytes,MSXBudgetTicket *t)
{
    MSXBudgetStatus s=MSX_BUDGET_OK;
    if(!b||!t||c<0||c>=MSX_MEMORY_CLASS_COUNT||!bytes)return MSX_BUDGET_INVALID;
    enter(b);
    if(t->live)s=MSX_BUDGET_STATE;
    else if(b->nextSequence==UINT64_MAX)s=MSX_BUDGET_STATE;
    else if(bytes>(c==MSX_MEMORY_DEVICE?b->deviceLimit-total(b,c):b->hostLimit-hostTotal(b)))
    { ++b->rejectedReservations; s=MSX_BUDGET_LIMIT; }
    else
    {
        t->sequence=++b->nextSequence; t->memoryClass=(unsigned)c;
        t->bytes=bytes; t->budgetOwner=(uintptr_t)b; t->live=1; b->reserved[c]+=bytes; peak(b);
        audit(b,"reserve",t->sequence,c,bytes);
    }
    leave(b); return s;
}
MSXBudgetStatus MSXresidentBudget_commitAllocation(MSXResidentBudget *b,MSXBudgetTicket *t,uint64_t actual,MSXBudgetAllocation *a)
{
    MSXBudgetStatus s=MSX_BUDGET_OK;
    if(!b||!t||!a||!actual)return MSX_BUDGET_INVALID;
    enter(b);
    if(!t->live||t->budgetOwner!=(uintptr_t)b||a->live||t->memoryClass>=MSX_MEMORY_CLASS_COUNT||actual>t->bytes||
       b->reserved[t->memoryClass]<t->bytes)s=MSX_BUDGET_STATE;
    else
    {
        b->reserved[t->memoryClass]-=t->bytes; b->allocated[t->memoryClass]+=actual;
        a->sequence=t->sequence; a->bytes=actual; a->budgetOwner=(uintptr_t)b; a->memoryClass=t->memoryClass; a->live=1;
        t->live=0;
        audit(b,"commit",a->sequence,a->memoryClass,actual);
    }
    leave(b);return s;
}
MSXBudgetStatus MSXresidentBudget_cancelReservation(MSXResidentBudget *b,MSXBudgetTicket *t)
{
    MSXBudgetStatus s=MSX_BUDGET_OK;
    if(!b||!t)return MSX_BUDGET_INVALID;
    enter(b);
    if(!t->live||t->budgetOwner!=(uintptr_t)b||t->memoryClass>=MSX_MEMORY_CLASS_COUNT||b->reserved[t->memoryClass]<t->bytes)s=MSX_BUDGET_STATE;
    else {b->reserved[t->memoryClass]-=t->bytes;t->live=0;audit(b,"cancel",t->sequence,t->memoryClass,t->bytes);}
    leave(b);return s;
}
MSXBudgetStatus MSXresidentBudget_releaseAllocation(MSXResidentBudget *b,MSXBudgetAllocation *a)
{
    MSXBudgetStatus s=MSX_BUDGET_OK;
    if(!b||!a)return MSX_BUDGET_INVALID;
    enter(b);
    if(!a->live||a->budgetOwner!=(uintptr_t)b||a->memoryClass>=MSX_MEMORY_CLASS_COUNT||b->allocated[a->memoryClass]<a->bytes)s=MSX_BUDGET_STATE;
    else {b->allocated[a->memoryClass]-=a->bytes;a->live=0;audit(b,"release",a->sequence,a->memoryClass,a->bytes);}
    leave(b);return s;
}
void MSXresidentBudget_snapshot(MSXResidentBudget *b,MSXResidentBudget *out)
{ if(!b||!out||b==out)return;enter(b);memcpy(out,b,sizeof(*out));out->lock=0;leave(b); }
void MSXresidentBudget_auditAllocation(MSXResidentBudget *b,const MSXBudgetAllocation *a,
    const char *source,unsigned line,uint64_t payloadBytes)
{
    FILE *f;
    if(!b||!a||b!=&globalBudget)return;
    enter(b);
    if(!auditEnabled()){leave(b);return;}
    if(!allocationAudit){allocationAudit=fopen("resident_allocation_manifest.csv","a+");
        if(allocationAudit){fseek(allocationAudit,0,SEEK_END);
            if(!ftell(allocationAudit))fprintf(allocationAudit,"sequence,source,line,memory_class,payload_bytes,charged_bytes\n");}}
    f=allocationAudit;
    if(f){
        fprintf(f,"%llu,%s,%u,%u,%llu,%llu\n",(unsigned long long)a->sequence,
            source?source:"unknown",line,a->memoryClass,(unsigned long long)payloadBytes,
            (unsigned long long)a->bytes);}
    leave(b);
}
