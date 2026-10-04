#ifndef MSX_RESIDENT_ALLOC_REDIRECT_H
#define MSX_RESIDENT_ALLOC_REDIRECT_H
/* Include after system headers. All MSX-owned heap producers and consumers
   use this allocator; QualPool interior pointers keep pool release hooks.
   Zero-sized CRT workspace requests receive one owned byte for compatibility. */
#include "msxresident_alloc.h"
#ifdef MSX_RESIDENT_WORKSPACE_DOMAIN
#define MSX_WORKSPACE_BUDGET MSXresidentBudget_domain(MSX_RESIDENT_WORKSPACE_DOMAIN)
#else
#define MSX_WORKSPACE_BUDGET MSXresidentBudget_global()
#endif
static inline void *MSXresidentWorkspace_malloc(size_t n,const char *s,unsigned line)
{return MSXresidentAlloc_mallocBudget(MSX_WORKSPACE_BUDGET,n?n:1,s,line);}
static inline void *MSXresidentWorkspace_calloc(size_t n,size_t w,const char *s,unsigned line)
{return n&&w?MSXresidentAlloc_callocBudget(MSX_WORKSPACE_BUDGET,n,w,s,line):
    MSXresidentAlloc_mallocBudget(MSX_WORKSPACE_BUDGET,1,s,line);}
#define malloc(n) MSXresidentWorkspace_malloc(n,__FILE__,__LINE__)
#define calloc(n,w) MSXresidentWorkspace_calloc(n,w,__FILE__,__LINE__)
#define realloc(p,n) MSXresidentAlloc_realloc(p,n,__FILE__,__LINE__)
#define free(p) MSXresidentAlloc_free(p)
#endif
