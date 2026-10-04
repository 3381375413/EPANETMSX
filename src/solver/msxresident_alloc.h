#ifndef MSX_RESIDENT_ALLOC_H
#define MSX_RESIDENT_ALLOC_H
#include <stddef.h>
#include "msxresident_budget.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Only pointers created by this allocator may be passed to free.  QualPool
   objects are interior pointers and continue to use the pool release hooks. */
void *MSXresidentAlloc_malloc(size_t bytes, const char *source, unsigned line);
void *MSXresidentAlloc_mallocBudget(MSXResidentBudget *,size_t,const char *,unsigned);
void *MSXresidentAlloc_mallocReserved(MSXBudgetTicket *,MSXResidentBudget *,size_t,const char *,unsigned);
void *MSXresidentAlloc_callocBudget(MSXResidentBudget *,size_t,size_t,const char *,unsigned);
void *MSXresidentAlloc_calloc(size_t count, size_t width, const char *source, unsigned line);
void *MSXresidentAlloc_realloc(void *ptr, size_t bytes, const char *source, unsigned line);
void MSXresidentAlloc_free(void *ptr);
size_t MSXresidentAlloc_headerBytes(void);
size_t MSXresidentAlloc_payloadBytes(const void *ptr);
size_t MSXresidentAlloc_chargedBytes(const void *ptr);
typedef struct MSXExternalAllocation MSXExternalAllocation;
/* External allocation records are themselves charged pageable allocations.
   Release is called ONLY after the driver's actual free succeeds. */
MSXExternalAllocation *MSXresidentExternal_reserve(size_t, MSXMemoryClass, const char *, unsigned);
void MSXresidentExternal_commit(MSXExternalAllocation *, uintptr_t ptr);
void MSXresidentExternal_cancel(MSXExternalAllocation *);
void MSXresidentExternal_release(uintptr_t ptr, MSXMemoryClass);
size_t MSXresidentExternal_recordBytes(void);
#ifdef __cplusplus
}
#endif
#endif
