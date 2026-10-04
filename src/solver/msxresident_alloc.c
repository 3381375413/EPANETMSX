#include <stdlib.h>
#include <string.h>
#include <stdint.h>
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
{
    MSXResidentBudget *b = MSXresidentBudget_global();
    MSXBudgetTicket ticket = {0};
    AllocationHeader *h;
    size_t charged;
    if (!bytes || bytes > SIZE_MAX - sizeof(*h)) return NULL;
    charged = bytes + sizeof(*h);
    if (MSXresidentBudget_reserve(b, MSX_MEMORY_PAGEABLE, charged, &ticket) != MSX_BUDGET_OK)
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
void *MSXresidentAlloc_calloc(size_t count, size_t width, const char *source, unsigned line)
{
    void *p; size_t bytes;
    if (!count || !width || width > SIZE_MAX / count) return NULL;
    bytes = count * width;
    p = MSXresidentAlloc_malloc(bytes, source, line);
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
    if (MSXresidentBudget_releaseAllocation(MSXresidentBudget_global(), &allocation) != MSX_BUDGET_OK)
        abort();
}
