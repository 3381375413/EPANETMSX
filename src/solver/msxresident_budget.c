#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#define WIN32_LEAN_AND_MEAN
#define NOGDI
#include <windows.h>
#include "msxresident_budget.h"
static MSXResidentBudget globalBudget = { UINT64_MAX, UINT64_MAX };
static MSXResidentBudget domains[MSX_BUDGET_DOMAIN_COUNT];
static volatile LONG domainsReady;
static uint64_t sessionSequence;
static int sessionActive;
static FILE *lifetimeAudit, *allocationAudit;
static int memoryAuditMode=-1;
static void enter(MSXResidentBudget *b)
{ if(b->parent)b=b->parent;while (InterlockedCompareExchange((volatile LONG *)&b->lock,1,0)) SwitchToThread(); }
static void leave(MSXResidentBudget *b)
{ if(b->parent)b=b->parent;InterlockedExchange((volatile LONG *)&b->lock,0); }
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
    if(b->parent)b=b->parent;
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
MSXBudgetStatus MSXresidentBudget_initChild(MSXResidentBudget *b,MSXResidentBudget *parent)
{
    if(!b||!parent||b==parent||parent->parent)return MSX_BUDGET_INVALID;
    MSXresidentBudget_init(b);b->parent=parent;return MSX_BUDGET_OK;
}
MSXResidentBudget *MSXresidentBudget_domain(MSXBudgetDomain d)
{
    unsigned k;if(d<0||d>=MSX_BUDGET_DOMAIN_COUNT)return NULL;
    if(InterlockedCompareExchange(&domainsReady,1,0)==0){
        for(k=0;k<MSX_BUDGET_DOMAIN_COUNT;++k)MSXresidentBudget_initChild(&domains[k],&globalBudget);
        InterlockedExchange(&domainsReady,2);
    }else while(InterlockedCompareExchange(&domainsReady,2,2)!=2)SwitchToThread();
    return &domains[d];
}
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
    else if((b->parent?b->parent:b)->nextSequence==UINT64_MAX)s=MSX_BUDGET_STATE;
    else if(bytes>(c==MSX_MEMORY_DEVICE?b->deviceLimit-total(b,c):b->hostLimit-hostTotal(b)))
    { ++b->rejectedReservations; s=MSX_BUDGET_LIMIT; }
    else if(b->parent && bytes>(c==MSX_MEMORY_DEVICE?b->parent->deviceLimit-total(b->parent,c):
                b->parent->hostLimit-hostTotal(b->parent)))
    {++b->rejectedReservations;++b->parent->rejectedReservations;s=MSX_BUDGET_LIMIT;}
    else
    {
        t->sequence=++(b->parent?b->parent:b)->nextSequence; t->memoryClass=(unsigned)c;
        t->bytes=bytes; t->budgetOwner=(uintptr_t)b; t->live=1; b->reserved[c]+=bytes; peak(b);
        if(b->parent){b->parent->reserved[c]+=bytes;peak(b->parent);}
        audit(b,"reserve",t->sequence,c,bytes);
    }
    leave(b); return s;
}
MSXBudgetStatus MSXresidentBudget_split(MSXBudgetTicket *source,MSXResidentBudget *dest,uint64_t bytes,MSXBudgetTicket *out)
{
    MSXResidentBudget *from,*rootBudget;MSXBudgetStatus status=MSX_BUDGET_OK;char event[64];
    if(!source||!dest||!out||source==out||!bytes||!source->live||!source->budgetOwner||
       source->memoryClass>=MSX_MEMORY_CLASS_COUNT)return MSX_BUDGET_INVALID;
    from=(MSXResidentBudget *)source->budgetOwner;rootBudget=from->parent?from->parent:from;
    if((dest->parent?dest->parent:dest)!=rootBudget)return MSX_BUDGET_INVALID;
    enter(from);
    if(out->live||!source->live||bytes>source->bytes||from->reserved[source->memoryClass]<source->bytes||
       rootBudget->nextSequence==UINT64_MAX)status=MSX_BUDGET_STATE;
    else if(dest!=from&&dest!=rootBudget&&bytes>(source->memoryClass==MSX_MEMORY_DEVICE?
            dest->deviceLimit-total(dest,source->memoryClass):dest->hostLimit-hostTotal(dest)))status=MSX_BUDGET_LIMIT;
    else {
        if(from!=rootBudget)from->reserved[source->memoryClass]-=bytes;
        if(dest!=rootBudget)dest->reserved[source->memoryClass]+=bytes;
        *out=*source;out->sequence=++rootBudget->nextSequence;out->bytes=bytes;out->budgetOwner=(uintptr_t)dest;
        source->bytes-=bytes;if(!source->bytes)source->live=0;
        peak(dest);sprintf(event,"split:%llu",(unsigned long long)source->sequence);
        audit(rootBudget,event,out->sequence,out->memoryClass,bytes);
    }
    leave(from);return status;
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
        if(b->parent){b->parent->reserved[t->memoryClass]-=t->bytes;b->parent->allocated[t->memoryClass]+=actual;}
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
    else {b->reserved[t->memoryClass]-=t->bytes;if(b->parent)b->parent->reserved[t->memoryClass]-=t->bytes;
        t->live=0;audit(b,"cancel",t->sequence,t->memoryClass,t->bytes);}
    leave(b);return s;
}
MSXBudgetStatus MSXresidentBudget_releaseAllocation(MSXResidentBudget *b,MSXBudgetAllocation *a)
{
    MSXBudgetStatus s=MSX_BUDGET_OK;
    if(!b||!a)return MSX_BUDGET_INVALID;
    enter(b);
    if(!a->live||a->budgetOwner!=(uintptr_t)b||a->memoryClass>=MSX_MEMORY_CLASS_COUNT||b->allocated[a->memoryClass]<a->bytes)s=MSX_BUDGET_STATE;
    else {b->allocated[a->memoryClass]-=a->bytes;if(b->parent)b->parent->allocated[a->memoryClass]-=a->bytes;
        a->live=0;audit(b,"release",a->sequence,a->memoryClass,a->bytes);}
    leave(b);return s;
}
void MSXresidentBudget_snapshot(MSXResidentBudget *b,MSXResidentBudget *out)
{ if(!b||!out||b==out)return;enter(b);memcpy(out,b,sizeof(*out));out->lock=0;leave(b); }
MSXBudgetStatus MSXresidentBudget_beginSession(void)
{
    unsigned k,c;MSXBudgetStatus status=MSX_BUDGET_OK;
    /* Materialize children before taking their shared root lock. */
    (void)MSXresidentBudget_domain(MSX_BUDGET_CPU_POOL);
    enter(&globalBudget);
    if(sessionActive||sessionSequence==UINT64_MAX)status=MSX_BUDGET_STATE;
    for(k=0;k<=MSX_BUDGET_DOMAIN_COUNT&&status==MSX_BUDGET_OK;++k){
        MSXResidentBudget *b=k?&domains[k-1]:&globalBudget;
        for(c=0;c<MSX_MEMORY_CLASS_COUNT;++c)
            if(b->allocated[c]||b->reserved[c])status=MSX_BUDGET_STATE;
    }
    if(status==MSX_BUDGET_OK){
        for(k=0;k<=MSX_BUDGET_DOMAIN_COUNT;++k){
            MSXResidentBudget *b=k?&domains[k-1]:&globalBudget;
            b->peakHostBytes=b->peakDeviceBytes=b->rejectedReservations=0;
        }
        ++sessionSequence;sessionActive=1;
    }
    leave(&globalBudget);return status;
}
static void summaryBudget(FILE *f,const MSXResidentBudget *b)
{
    fprintf(f,"\"host_limit_bytes\":%llu,\"device_limit_bytes\":%llu,"
        "\"host_limit_unbounded\":%s,\"device_limit_unbounded\":%s,"
        "\"peak_host_including_reservations\":%llu,\"peak_device_including_reservations\":%llu,"
        "\"final_allocated\":[%llu,%llu,%llu],"
        "\"final_reserved\":[%llu,%llu,%llu],"
        "\"rejected_reservations\":%llu",
        (unsigned long long)b->hostLimit,(unsigned long long)b->deviceLimit,
        b->hostLimit==UINT64_MAX?"true":"false",b->deviceLimit==UINT64_MAX?"true":"false",
        (unsigned long long)b->peakHostBytes,(unsigned long long)b->peakDeviceBytes,
        (unsigned long long)b->allocated[0],(unsigned long long)b->allocated[1],(unsigned long long)b->allocated[2],
        (unsigned long long)b->reserved[0],(unsigned long long)b->reserved[1],(unsigned long long)b->reserved[2],
        (unsigned long long)b->rejectedReservations);
}
int MSXresidentBudget_endSession(const char *path)
{
    static const char *names[]={"cpu_pool","upload","scratch","temporary"};
    MSXResidentBudget snapshot[MSX_BUDGET_DOMAIN_COUNT+1];uint64_t sequence;
    unsigned k,c;int remaining=0,exceeded=0,failed;FILE *f;
    if(!path)return 0;
    enter(&globalBudget);
    if(!sessionActive){leave(&globalBudget);return 1;}
    snapshot[0]=globalBudget;
    for(k=0;k<MSX_BUDGET_DOMAIN_COUNT;++k)snapshot[k+1]=domains[k];
    sequence=sessionSequence;sessionActive=0;
    leave(&globalBudget);
    for(k=0;k<=MSX_BUDGET_DOMAIN_COUNT;++k)for(c=0;c<MSX_MEMORY_CLASS_COUNT;++c)
        if(snapshot[k].allocated[c]||snapshot[k].reserved[c])remaining=1;
    for(k=0;k<=MSX_BUDGET_DOMAIN_COUNT;++k)
        if(snapshot[k].peakHostBytes>snapshot[k].hostLimit||snapshot[k].peakDeviceBytes>snapshot[k].deviceLimit)exceeded=1;
    f=fopen(path,"wb");if(!f)return 0;
    fprintf(f,"{\"schema_version\":1,\"session\":%llu,\"status\":\"%s\",\"remaining_nonzero\":%s,\"peak_exceeds_final_limit\":%s,"
        "\"scope\":\"MSX managed requested heap: pageable plus pinned host; device separate; allocation and budget metadata included\","
        "\"excluded_scope\":\"EPANET, CRT, CUDA internal allocations and OS process overhead\","
        "\"memory_class_order\":[\"pageable\",\"pinned\",\"device\"],"
        "\"child_accounting\":\"included in root totals; do not sum children with root\",",
        (unsigned long long)sequence,remaining||exceeded?"FAIL":"PASS",remaining?"true":"false",exceeded?"true":"false");
    summaryBudget(f,&snapshot[0]);fprintf(f,",\"domains\":[");
    for(k=0;k<MSX_BUDGET_DOMAIN_COUNT;++k){
        const MSXResidentBudget *b=&snapshot[k+1];int live=0;
        for(c=0;c<MSX_MEMORY_CLASS_COUNT;++c)if(b->allocated[c]||b->reserved[c])live=1;
        if(k)fputc(',',f);fprintf(f,"{\"name\":\"%s\",\"status\":\"%s\",",names[k],
            live||b->peakHostBytes>b->hostLimit||b->peakDeviceBytes>b->deviceLimit?"FAIL":"PASS");
        summaryBudget(f,b);fputc('}',f);
    }
    fprintf(f,"]}\n");failed=ferror(f);if(fclose(f))failed=1;return !failed;
}
void MSXresidentBudget_auditAllocation(MSXResidentBudget *b,const MSXBudgetAllocation *a,
    const char *source,unsigned line,uint64_t payloadBytes)
{
    FILE *f;
    if(!b||!a||(b->parent?b->parent:b)!=&globalBudget)return;
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
