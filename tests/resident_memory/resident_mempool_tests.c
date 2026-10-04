#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int failures, allocations, live, failAt;
static void *testMalloc(size_t n)
{
    void *p;
    if (++allocations == failAt) return NULL;
    p = malloc(n); if (p) ++live;
    return p;
}
static void *testCalloc(size_t n, size_t s)
{
    void *p;
    if (++allocations == failAt) return NULL;
    p = calloc(n,s); if (p) ++live;
    return p;
}
static void testFree(void *p) { if (p) --live; free(p); }
#define malloc testMalloc
#define calloc testCalloc
#define free(p) testFree(p)
#include "mempool.c"
#undef malloc
#undef calloc
#undef free
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"line %d: %s\n",__LINE__,#x); ++failures; } } while(0)
int main(void)
{
    alloc_handle_t *a, *b;
    AllocPoolStats before, after;
    char *first, *second;
    int k;
    /* Failure at every allocation in initial pool creation leaks nothing. */
    for (k=1;k<=3;++k)
    {
        failAt=allocations+k;
        CHECK(AllocInit()==NULL); CHECK(live==0); CHECK(root==NULL);
    }
    failAt=0; a=AllocInit(); CHECK(a!=NULL);
    CHECK(AllocGetPoolStats(a,&before)); CHECK(before.blockCount==1);
    CHECK(before.backingBytes==sizeof(alloc_root_t)+sizeof(alloc_hdr_t)+64000);
    first=Alloc(32000); CHECK(first!=NULL); memset(first,0x37,32000);
    second=Alloc(32000); CHECK(second!=NULL); CHECK(second!=first);
    CHECK(second==first+32000);
    CHECK(AllocGetPoolStats(a,&before)); CHECK(before.blockCount==1);
    CHECK(before.allocationCalls==2); CHECK(before.requestedBytes==64000);
    CHECK(first[0]==0x37 && first[31999]==0x37);
    AllocReset(); CHECK(AllocGetPoolStats(a,&after));
    CHECK(after.backingBytes==before.backingBytes); CHECK(after.resetCount==1);
    CHECK(Alloc(32000)==first); CHECK(Alloc(32000)==second);
    CHECK(AllocGetPoolStats(a,&after)); CHECK(after.blockCount==1);
    CHECK(Alloc(64001)==NULL); CHECK(Alloc(0)==NULL); CHECK(Alloc(-1)==NULL);
    /* Failed growth leaves current pointer usable and frees partial mallocs. */
    for(k=1;k<=2;++k)
    {
        char *saved=root->current->free;
        int savedLive=live;
        failAt=allocations+k;
        CHECK(Alloc(40000)==NULL); CHECK(live==savedLive);
        CHECK(root->current->free==saved);
    }
    failAt=0; CHECK(Alloc(40000)!=NULL);
    b=AllocInit(); CHECK(b!=NULL);
    CHECK(AllocSetPool(a)==b); CHECK(AllocGetPoolStats(a,&before));
    CHECK(AllocGetPoolStats(b,&after)); CHECK(before.poolId!=after.poolId);
    /* Failed new pool must not replace the previously active pool. */
    failAt=allocations+2; CHECK(AllocInit()==NULL); CHECK(root==(alloc_root_t*)a);
    failAt=0; AllocFreePool(); CHECK(root==NULL);
    AllocSetPool(b); AllocFreePool(); CHECK(live==0);
    AllocReset(); AllocFreePool(); CHECK(Alloc(1)==NULL);
    CHECK(!AllocGetPoolStats(NULL,&before));
    {
        MSXResidentBudget *budget=MSXresidentBudget_global(), snap;
        uint64_t minimum=sizeof(alloc_root_t)+sizeof(alloc_hdr_t)+64000;
        CHECK(!MSXresidentBudget_configure(budget,minimum-1,UINT64_MAX));
        CHECK(AllocInit()==NULL); CHECK(live==0);
        MSXresidentBudget_snapshot(budget,&snap);
        CHECK(snap.allocated[0]==0 && snap.reserved[0]==0);
        CHECK(!MSXresidentBudget_configure(budget,minimum,UINT64_MAX));
        a=AllocInit(); CHECK(a!=NULL); CHECK(Alloc(32000)!=NULL);
        CHECK(Alloc(32000)!=NULL); /* Fill the existing block exactly. */
        CHECK(Alloc(1)==NULL); /* Only actual growth needs another block. */
        AllocReset(); CHECK(Alloc(32000)!=NULL);
        CHECK(Alloc(32000)!=NULL);
        AllocReset(); CHECK(Alloc(64000)!=NULL);
        CHECK(Alloc(1)==NULL);
        MSXresidentBudget_snapshot(budget,&snap);
        CHECK(snap.allocated[0]==minimum && snap.reserved[0]==0);
        AllocFreePool(); CHECK(live==0);
        MSXresidentBudget_snapshot(budget,&snap);
        CHECK(snap.allocated[0]==0 && snap.reserved[0]==0);
        CHECK(!MSXresidentBudget_configure(budget,UINT64_MAX,UINT64_MAX));
    }
    printf("resident_mempool_tests failures=%d live=%d root_bytes=%llu header_bytes=%llu block_bytes=64000\n",
        failures,live,(unsigned long long)sizeof(alloc_root_t),(unsigned long long)sizeof(alloc_hdr_t));
    return failures?1:0;
}
