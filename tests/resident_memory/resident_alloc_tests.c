#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <windows.h>
#include "msxresident_budget.h"
static int failures, heapLive, failNext;
static void *heapMalloc(size_t n)
{void *p;if(failNext){failNext=0;return NULL;}p=malloc(n);if(p)++heapLive;return p;}
static void heapFree(void *p){if(p)--heapLive;free(p);}
#define malloc heapMalloc
#define free(p) heapFree(p)
#include "msxresident_alloc.c"
#undef malloc
#undef free
#define CHECK(x) do {if(!(x)){fprintf(stderr,"line %d: %s\n",__LINE__,#x);++failures;}}while(0)
int main(void)
{
    MSXResidentBudget *b=MSXresidentBudget_global(), s;
    size_t header=MSXresidentAlloc_headerBytes(); unsigned char *a,*p; size_t i;
    CHECK(header%16==0);
    CHECK(!MSXresidentBudget_configure(b,header+17,UINT64_MAX));
    a=(unsigned char*)MSXresidentAlloc_malloc(17,__FILE__,__LINE__);CHECK(a!=NULL);
    CHECK((uintptr_t)a%16==0);
    CHECK(MSXresidentAlloc_malloc(1,__FILE__,__LINE__)==NULL);
    CHECK(MSXresidentBudget_configure(b,header+16,UINT64_MAX)==MSX_BUDGET_LIMIT);
    MSXresidentAlloc_free(a);MSXresidentAlloc_free(NULL);
    CHECK(!MSXresidentBudget_configure(b,header+16,UINT64_MAX));
    CHECK(MSXresidentAlloc_malloc(17,__FILE__,__LINE__)==NULL);
    CHECK(!MSXresidentBudget_configure(b,UINT64_MAX,UINT64_MAX));
    failNext=1;CHECK(MSXresidentAlloc_malloc(17,__FILE__,__LINE__)==NULL);
    MSXresidentBudget_snapshot(b,&s);CHECK(!s.allocated[0]&&!s.reserved[0]&&heapLive==0);
    CHECK(MSXresidentAlloc_malloc(SIZE_MAX,__FILE__,__LINE__)==NULL);
    CHECK(MSXresidentAlloc_calloc(SIZE_MAX,2,__FILE__,__LINE__)==NULL);
    CHECK(MSXresidentAlloc_calloc(0,1,__FILE__,__LINE__)==NULL);
    p=(unsigned char*)MSXresidentAlloc_calloc(17,3,__FILE__,__LINE__);CHECK(p!=NULL);
    if(p)for(i=0;i<51;++i)CHECK(p[i]==0);
    MSXresidentBudget_snapshot(b,&s);CHECK(s.allocated[0]==header+51);
    MSXresidentAlloc_free(p);
    /* Realloc reserves the complete replacement while the old block lives. */
    a=(unsigned char*)MSXresidentAlloc_malloc(17,__FILE__,__LINE__);
    memset(a,0x37,17);
    CHECK(!MSXresidentBudget_configure(b,2*header+17+33-1,UINT64_MAX));
    CHECK(MSXresidentAlloc_realloc(a,33,__FILE__,__LINE__)==NULL);
    CHECK(a[0]==0x37 && a[16]==0x37);
    CHECK(!MSXresidentBudget_configure(b,2*header+17+33,UINT64_MAX));
    p=(unsigned char*)MSXresidentAlloc_realloc(a,33,__FILE__,__LINE__);CHECK(p!=NULL);
    CHECK(p[0]==0x37 && p[16]==0x37 && MSXresidentAlloc_payloadBytes(p)==33);
    CHECK(MSXresidentAlloc_chargedBytes(p)==header+33);
    CHECK(MSXresidentAlloc_realloc(p,0,__FILE__,__LINE__)==NULL);
    CHECK(!MSXresidentBudget_configure(b,UINT64_MAX,UINT64_MAX));
    {
        MSXExternalAllocation *r=MSXresidentExternal_reserve(100,MSX_MEMORY_DEVICE,__FILE__,__LINE__);
        CHECK(r!=NULL);MSXresidentBudget_snapshot(b,&s);
        CHECK(s.reserved[2]==100 && s.allocated[0]==MSXresidentExternal_recordBytes());
        MSXresidentExternal_cancel(r);MSXresidentBudget_snapshot(b,&s);
        CHECK(!s.reserved[2] && !s.allocated[0]);
        r=MSXresidentExternal_reserve(100,MSX_MEMORY_PINNED,__FILE__,__LINE__);
        MSXresidentExternal_commit(r,12345);MSXresidentBudget_snapshot(b,&s);
        CHECK(s.allocated[1]==100 && !s.reserved[1]);
        MSXresidentExternal_release(12345,MSX_MEMORY_PINNED);
        CHECK(!MSXresidentBudget_configure(b,UINT64_MAX,99));
        CHECK(MSXresidentExternal_reserve(100,MSX_MEMORY_DEVICE,__FILE__,__LINE__)==NULL);
        MSXresidentBudget_snapshot(b,&s);CHECK(!s.allocated[0]&&!s.reserved[2]);
        CHECK(!MSXresidentBudget_configure(b,UINT64_MAX,UINT64_MAX));
    }
    /* A second allocation failure must not cancel the first allocation. */
    a=(unsigned char*)MSXresidentAlloc_malloc(17,__FILE__,__LINE__);failNext=1;
    CHECK(MSXresidentAlloc_malloc(17,__FILE__,__LINE__)==NULL);
    MSXresidentBudget_snapshot(b,&s);CHECK(s.allocated[0]==header+17&&!s.reserved[0]);
    MSXresidentAlloc_free(a);MSXresidentBudget_snapshot(b,&s);
    CHECK(!s.allocated[0]&&!s.reserved[0]&&heapLive==0);
    printf("resident_alloc_tests failures=%d header_bytes=%llu live_heap=%d\n",failures,(unsigned long long)header,heapLive);
    return failures?1:0;
}
