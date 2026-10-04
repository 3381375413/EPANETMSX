#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#define WIN32_LEAN_AND_MEAN
#define NOGDI
#include <windows.h>
#include "msxresident_alloc.h"
#include "msxresident_budget.h"
/* Preserve malloc's alignment while charging the embedded allocation token.
   No pointer registry, uncharged heap bookkeeping, or per-object pool charge. */
typedef __declspec(align(16)) struct {
    MSXBudgetAllocation allocation;
    uint64_t payloadBytes;
    uint64_t magic;
} AllocationHeader;
#define ALLOCATION_MAGIC UINT64_C(0x4d5358414c4c4f43)
size_t MSXresidentAlloc_headerBytes(void) { return sizeof(AllocationHeader); }
void *MSXresidentAlloc_malloc(size_t bytes, const char *source, unsigned line)
{return MSXresidentAlloc_mallocBudget(MSXresidentBudget_global(),bytes,source,line);}
static void *mallocWithReservation(MSXBudgetTicket *peakTicket,MSXResidentBudget *b,size_t bytes,const char *source,unsigned line)
{
    MSXBudgetTicket ticket = {0};
    AllocationHeader *h;
    size_t charged;
    if (!bytes || bytes > SIZE_MAX - sizeof(*h)) return NULL;
    charged = bytes + sizeof(*h);
    if ((peakTicket?MSXresidentBudget_split(peakTicket,b,charged,&ticket):
         MSXresidentBudget_reserve(b, MSX_MEMORY_PAGEABLE, charged, &ticket)) != MSX_BUDGET_OK)
        return NULL;
    h = (AllocationHeader *)malloc(charged);
    if (!h) { MSXresidentBudget_cancelReservation(b, &ticket); return NULL; }
    memset(h, 0, sizeof(*h));
    if (MSXresidentBudget_commitAllocation(b, &ticket, charged, &h->allocation) != MSX_BUDGET_OK)
    { free(h); MSXresidentBudget_cancelReservation(b, &ticket); return NULL; }
    h->payloadBytes = bytes; h->magic = ALLOCATION_MAGIC;
    MSXresidentBudget_auditAllocation(b, &h->allocation, source, line, bytes);
    return h + 1;
}
void *MSXresidentAlloc_mallocBudget(MSXResidentBudget *b,size_t bytes,const char *source,unsigned line)
{return mallocWithReservation(NULL,b,bytes,source,line);}
void *MSXresidentAlloc_mallocReserved(MSXBudgetTicket *ticket,MSXResidentBudget *b,size_t bytes,const char *source,unsigned line)
{return mallocWithReservation(ticket,b,bytes,source,line);}
void *MSXresidentAlloc_calloc(size_t count, size_t width, const char *source, unsigned line)
{return MSXresidentAlloc_callocBudget(MSXresidentBudget_global(),count,width,source,line);}
void *MSXresidentAlloc_callocBudget(MSXResidentBudget *b,size_t count,size_t width,const char *source,unsigned line)
{
    void *p; size_t bytes;
    if (!count || !width || width > SIZE_MAX / count) return NULL;
    bytes = count * width;
    p = MSXresidentAlloc_mallocBudget(b,bytes, source, line);
    if (p) memset(p, 0, bytes);
    return p;
}
void MSXresidentAlloc_free(void *ptr)
{
    AllocationHeader *h; MSXBudgetAllocation allocation;
    if (!ptr) return;
    h = (AllocationHeader *)ptr - 1;
    /* A mixed allocator is a programming error; do not silently undercharge. */
    if (h->magic != ALLOCATION_MAGIC) abort();
    allocation = h->allocation; h->magic = 0;
    free(h);
    if (MSXresidentBudget_releaseAllocation((MSXResidentBudget *)allocation.budgetOwner, &allocation) != MSX_BUDGET_OK)
        abort();
}
size_t MSXresidentAlloc_payloadBytes(const void *ptr)
{
    const AllocationHeader *h;
    if (!ptr) return 0;
    h = (const AllocationHeader *)ptr - 1;
    if (h->magic != ALLOCATION_MAGIC) abort();
    return (size_t)h->payloadBytes;
}
size_t MSXresidentAlloc_chargedBytes(const void *ptr)
{ return ptr ? MSXresidentAlloc_payloadBytes(ptr) + sizeof(AllocationHeader) : 0; }
void *MSXresidentAlloc_realloc(void *ptr, size_t bytes, const char *source, unsigned line)
{
    void *next; size_t old;
    if (!bytes) { MSXresidentAlloc_free(ptr); return NULL; }
    old = MSXresidentAlloc_payloadBytes(ptr);
    next = MSXresidentAlloc_mallocBudget(ptr?(MSXResidentBudget *)((AllocationHeader *)ptr-1)->allocation.budgetOwner:
        MSXresidentBudget_global(),bytes, source, line);
    if (!next) return NULL; /* Old pointer and its charge remain valid. */
    if (ptr) memcpy(next, ptr, old < bytes ? old : bytes);
    MSXresidentAlloc_free(ptr);
    return next;
}
struct MSXExternalAllocation {
    MSXBudgetTicket ticket;
    MSXBudgetAllocation allocation;
    uintptr_t ptr;
    const char *source;
    unsigned line;
    struct MSXExternalAllocation *next;
};
static MSXExternalAllocation *externalHead;
static volatile LONG externalLock;
static void externalEnter(void)
{ while (InterlockedCompareExchange(&externalLock,1,0)) SwitchToThread(); }
static void externalLeave(void) { InterlockedExchange(&externalLock,0); }
size_t MSXresidentExternal_recordBytes(void)
{ return sizeof(MSXExternalAllocation) + sizeof(AllocationHeader); }
MSXExternalAllocation *MSXresidentExternal_reserve(size_t bytes, MSXMemoryClass kind,
                                                 const char *source, unsigned line)
{
    MSXExternalAllocation *r = (MSXExternalAllocation *)MSXresidentAlloc_calloc(
        1,sizeof(*r),source,line);
    if (!r) return NULL;
    if (MSXresidentBudget_reserve(MSXresidentBudget_global(),kind,bytes,&r->ticket)!=MSX_BUDGET_OK)
    { MSXresidentAlloc_free(r); return NULL; }
    r->source=source;r->line=line;return r;
}
void MSXresidentExternal_commit(MSXExternalAllocation *r,uintptr_t ptr)
{
    if (!r || !ptr) abort();
    if (MSXresidentBudget_commitAllocation(MSXresidentBudget_global(),&r->ticket,
            r->ticket.bytes,&r->allocation)!=MSX_BUDGET_OK) abort();
    r->ptr=ptr;
    MSXresidentBudget_auditAllocation(MSXresidentBudget_global(),&r->allocation,
        r->source,r->line,r->allocation.bytes);
    externalEnter();r->next=externalHead;externalHead=r;externalLeave();
}
void MSXresidentExternal_cancel(MSXExternalAllocation *r)
{
    if (!r) return;
    if (MSXresidentBudget_cancelReservation(MSXresidentBudget_global(),&r->ticket)!=MSX_BUDGET_OK) abort();
    MSXresidentAlloc_free(r);
}
void MSXresidentExternal_release(uintptr_t ptr,MSXMemoryClass kind)
{
    MSXExternalAllocation **p,*r=NULL;
    externalEnter();
    for(p=&externalHead;*p;p=&(*p)->next)
        if((*p)->ptr==ptr && (*p)->allocation.memoryClass==(unsigned)kind)
        {r=*p;*p=r->next;break;}
    externalLeave();
    if(!r)abort();
    if(MSXresidentBudget_releaseAllocation(MSXresidentBudget_global(),&r->allocation)!=MSX_BUDGET_OK)abort();
    MSXresidentAlloc_free(r);
}
