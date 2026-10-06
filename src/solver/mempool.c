/*  mempool.c
**
**  A simple fast memory allocation package.
**
**  By Steve Hill in Graphics Gems III, David Kirk (ed.),
**    Academic Press, Boston, MA, 1992
**
**  Modified by Lew Rossman, 8/13/94.
**
**  AllocInit()     - create an alloc pool, returns the old pool handle
**  Alloc()         - allocate memory
**  AllocReset()    - reset the current pool
**  AllocSetPool()  - set the current pool
**  AllocFree()     - free the memory used by the current pool.
**
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>

#include "mempool.h"
#include "msxresident_budget.h"

/*
**  ALLOC_BLOCK_SIZE - adjust this size to suit your installation - it
**  should be reasonably large otherwise you will be mallocing a lot.
*/

#define ALLOC_BLOCK_SIZE   64000       /*(62*1024)*/

/*
**  alloc_hdr_t - Header for each block of memory.
*/

typedef struct alloc_hdr_s
{
    struct alloc_hdr_s *next;   /* Next Block          */
    char               *block,  /* Start of block      */
                       *free,   /* Next free in block  */
                       *end;    /* block + block size  */
    MSXBudgetAllocation backing;
}  alloc_hdr_t;

/*
**  alloc_root_t - Header for the whole pool.
*/

typedef struct alloc_root_s
{
    alloc_hdr_t *first,    /* First header in pool */
                *current;  /* Current header       */
    AllocPoolStats stats;
    FILE *audit;
    MSXBudgetAllocation backing;
    char owner[24];
    alloc_hdr_t *forecastCurrent;
    char *forecastFree;
    uint64_t forecastRetainedNext;
    uint64_t forecastBlockCount;
    MSXBudgetTicket *peakReservation;
}  alloc_root_t;

/*
**  root - Pointer to the current pool.
*/

static alloc_root_t *root;
static uint64_t nextPoolId;
#ifdef MSX_RESIDENT_TEST_API
static unsigned testFailBlock;
void AllocTestFailBlock(unsigned n) { testFailBlock=n; }
#endif

int AllocBindReservation(alloc_handle_t *handle,MSXBudgetTicket *ticket)
{
    alloc_root_t *pool=(alloc_root_t *)handle;if(!pool)return 0;
    if(ticket&&pool->peakReservation)return 0;
    pool->peakReservation=ticket;return 1;
}

static void auditPool(alloc_root_t *pool, const char *event)
{
    AllocPoolStats *s = &pool->stats;
    if (!pool->audit) return;
    fprintf(pool->audit, "%llu,%s,%s,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n",
        (unsigned long long)s->poolId, pool->owner, event,
        (unsigned long long)s->blockCount, (unsigned long long)s->backingBytes,
        (unsigned long long)s->peakBackingBytes,
        (unsigned long long)s->allocationCalls, (unsigned long long)s->requestedBytes,
        (unsigned long long)s->alignedBytes, (unsigned long long)s->resetCount,
        (unsigned long long)s->failedAllocations);
    /* Multiple pools share the audit file; flush before another handle opens
       it so the header and event ordering remain unambiguous. Audit only. */
    fflush(pool->audit);
}


/*
**  AllocHdr()
**
**  Private routine to allocate a header and memory block.
*/

static alloc_hdr_t *AllocHdr(alloc_root_t *pool);
                
static alloc_hdr_t * AllocHdr(alloc_root_t *pool)
{
    alloc_hdr_t     *hdr;
    char            *block;
    MSXBudgetTicket ticket = {0};
    MSXResidentBudget *budget = (MSXResidentBudget *)pool->backing.budgetOwner;
#ifdef MSX_RESIDENT_TEST_API
    if(testFailBlock && !--testFailBlock){
        ++pool->stats.failedAllocations;
        return NULL;
    }
#endif

    if ((pool->peakReservation?MSXresidentBudget_split(pool->peakReservation,budget,
            ALLOC_BLOCK_SIZE+sizeof(*hdr),&ticket):MSXresidentBudget_reserve(budget,MSX_MEMORY_PAGEABLE,
            ALLOC_BLOCK_SIZE + sizeof(*hdr), &ticket)) != MSX_BUDGET_OK)
    {
        ++pool->stats.failedAllocations;
        auditPool(pool, "block_budget_rejected");
        return NULL;
    }

    block = (char *) malloc(ALLOC_BLOCK_SIZE);
    hdr   = (alloc_hdr_t *) malloc(sizeof(alloc_hdr_t));

    if (hdr == NULL || block == NULL)
    {
        free(hdr);
        free(block);
        MSXresidentBudget_cancelReservation(budget, &ticket);
        ++pool->stats.failedAllocations;
        auditPool(pool, "block_failed");
        return(NULL);
    }
    hdr->block = block;
    hdr->free  = block;
    hdr->next  = NULL;
    hdr->end   = block + ALLOC_BLOCK_SIZE;
    memset(&hdr->backing, 0, sizeof(hdr->backing));
    MSXresidentBudget_commitAllocation(budget, &ticket,
        ALLOC_BLOCK_SIZE + sizeof(*hdr), &hdr->backing);
    MSXresidentBudget_auditAllocation(budget, &hdr->backing, pool->owner, __LINE__, ALLOC_BLOCK_SIZE);

    ++pool->stats.blockCount;
    pool->stats.backingBytes += ALLOC_BLOCK_SIZE + sizeof(alloc_hdr_t);
    pool->stats.peakBackingBytes = pool->stats.backingBytes;
    auditPool(pool, "block_allocated");

    return(hdr);
}


/*
**  AllocInit()
**
**  Create a new memory pool with one block.
**  Returns pointer to the new pool.
*/

alloc_handle_t * AllocInit()
{ return AllocInitNamed("unspecified"); }

alloc_handle_t * AllocInitNamed(const char *owner)
{
    alloc_root_t *pool;
    MSXBudgetTicket ticket = {0};
    MSXResidentBudget *budget = owner && !strcmp(owner,"QualPool")?
        MSXresidentBudget_domain(MSX_BUDGET_CPU_POOL):MSXresidentBudget_global();
    const char *audit = getenv("MSX_RESIDENT_MEMORY_AUDIT");
    if (MSXresidentBudget_reserve(budget, MSX_MEMORY_PAGEABLE,
            sizeof(*pool), &ticket) != MSX_BUDGET_OK) return NULL;
    pool = (alloc_root_t *)calloc(1, sizeof(*pool));
    if (!pool)
    {
        MSXresidentBudget_cancelReservation(budget, &ticket);
        return NULL;
    }
    MSXresidentBudget_commitAllocation(budget, &ticket, sizeof(*pool), &pool->backing);
    /* Labels are fixed identifiers, not arbitrary CSV text. */
    if (owner && strlen(owner)<sizeof(pool->owner) && !strpbrk(owner,",\r\n"))
        strcpy(pool->owner,owner);
    else strcpy(pool->owner,"unspecified");
    MSXresidentBudget_auditAllocation(budget, &pool->backing, pool->owner, __LINE__, sizeof(*pool));
    pool->stats.poolId = ++nextPoolId;
    pool->stats.backingBytes = pool->stats.peakBackingBytes = sizeof(*pool);
    if (audit && !strcmp(audit, "1"))
    {
        long length;
        pool->audit = fopen("resident_pool_backing.csv", "a+");
        if (pool->audit)
        {
            fseek(pool->audit, 0, SEEK_END);
            length = ftell(pool->audit);
            if (length == 0) fprintf(pool->audit,
                "pool_id,owner,event,block_count,backing_bytes,peak_backing_bytes,allocation_calls,requested_bytes,aligned_bytes,reset_count,failed_allocations\n");
        }
    }
    auditPool(pool, "pool_created");
    if ((pool->first = AllocHdr(pool)) == NULL)
    {
        MSXBudgetAllocation backing = pool->backing;
        pool->stats.backingBytes = 0;
        auditPool(pool, "pool_released");
        if (pool->audit) fclose(pool->audit);
        free(pool);
        MSXresidentBudget_releaseAllocation(budget, &backing);
        return NULL;
    }
    pool->current = pool->first;
    root = pool;
    return (alloc_handle_t *)pool;
}


/*
**  Alloc()
**
**  Use as a direct replacement for malloc().  Allocates
**  memory from the current pool.
*/

char * Alloc(long size)
{
    alloc_hdr_t  *hdr;
    char         *ptr;
    long requested = size;
    if (!root) return NULL;
    /* Reject invalid/oversize requests before pointer arithmetic. This pool
       uses fixed-size blocks and cannot safely serve a larger object. */
    if (size <= 0 || size > ALLOC_BLOCK_SIZE || size > LONG_MAX - 3)
    {
        ++root->stats.failedAllocations;
        auditPool(root, "request_rejected");
        return NULL;
    }
    hdr = root->current;

    /*
    **  Align to 4 byte boundary - should be ok for most machines.
    **  Change this if your machine has weird alignment requirements.
    */
    size = (size + 3) & ~3L;

    ptr = hdr->free;

    /* Check if the current block is exhausted. */

    if ((size_t)size > (size_t)(hdr->end - hdr->free))
    {
        /* Is the next block already allocated? */

        if (hdr->next != NULL)
        {
            /* re-use block */
            hdr->next->free = hdr->next->block;
            root->current = hdr->next;
        }
        else
        {
            /* extend the pool with a new block */
            if ( (hdr->next = AllocHdr(root)) == NULL) return(NULL);
            root->current = hdr->next;
        }

        /* set ptr to the first location in the next block */
        ptr = root->current->free;
        root->current->free += size;
    }
    else hdr->free += size;

    ++root->stats.allocationCalls;
    root->stats.requestedBytes += (uint64_t)requested;
    root->stats.alignedBytes += (uint64_t)size;

    /* Return pointer to allocated memory. */

    return(ptr);
}


/*
**  AllocSetPool()
**
**  Change the current pool.  Return the old pool.
*/

alloc_handle_t * AllocSetPool(alloc_handle_t *newpool)
{
    alloc_handle_t *old = (alloc_handle_t *) root;
    root = (alloc_root_t *) newpool;
    return(old);
}


/*
**  AllocReset()
**
**  Reset the current pool for re-use.  No memory is freed,
**  so this is very fast.
*/

void  AllocReset()
{
    if (!root) return;
    root->current = root->first;
    root->current->free = root->current->block;
    ++root->stats.resetCount;
    auditPool(root, "pool_reset_retained");
}

int AllocGetPoolStats(alloc_handle_t *pool, AllocPoolStats *stats)
{
    if (!pool || !stats) return 0;
    *stats = ((alloc_root_t *)pool)->stats;
    return 1;
}
int AllocForecastBegin(alloc_handle_t *handle,AllocForecast *forecast)
{
    alloc_root_t *pool=(alloc_root_t *)handle;alloc_hdr_t *hdr;
    if(!pool||!forecast)return 0;
    hdr=pool->current;
    if(pool->forecastCurrent!=hdr || pool->forecastFree!=hdr->free ||
       pool->forecastBlockCount!=pool->stats.blockCount){
        pool->forecastRetainedNext=0;
        for(alloc_hdr_t *p=hdr->next;p;p=p->next)++pool->forecastRetainedNext;
        pool->forecastCurrent=hdr;pool->forecastFree=hdr->free;
        pool->forecastBlockCount=pool->stats.blockCount;
    }
    forecast->remaining=(uint64_t)(hdr->end-hdr->free);
    forecast->retained=pool->forecastRetainedNext;forecast->bytes=0;
    return 1;
}
int AllocForecastAppend(AllocForecast *forecast,const uint64_t *pattern,
                        unsigned count,uint64_t repeats)
{
    uint64_t r,remaining,extra,retained,visited=0,seenRepeat[16]={0},seenBlocks[16]={0};
    unsigned k;unsigned char seen[16]={0};
    if(!forecast||!pattern||!count||count>16)return 0;
    for(k=0;k<count;++k)if(!pattern[k]||pattern[k]>ALLOC_BLOCK_SIZE)return 0;
    remaining=forecast->remaining;retained=forecast->retained;extra=forecast->bytes;
    for(r=0;r<repeats;++r)for(k=0;k<count;++k){
        uint64_t size=(pattern[k]+3)&~UINT64_C(3);
        if(size>remaining){
            /* A fresh block at the same pattern index has the same future.
               Skip whole cycles, charging only blocks beyond retained ones. */
            if(seen[k] && r>seenRepeat[k]){
                uint64_t period=r-seenRepeat[k],blocks=visited-seenBlocks[k];
                uint64_t cycles=(repeats-1-r)/period,skip;
                if(cycles && blocks){
                    if(cycles>UINT64_MAX/blocks)return 0;skip=cycles*blocks;
                    uint64_t fresh=skip>retained?skip-retained:0;
                    if(fresh>(UINT64_MAX-extra)/(ALLOC_BLOCK_SIZE+sizeof(alloc_hdr_t)))return 0;
                    extra+=fresh*(ALLOC_BLOCK_SIZE+sizeof(alloc_hdr_t));
                    retained=skip<retained?retained-skip:0;r+=cycles*period;visited+=skip;
                }
            }else {seen[k]=1;seenRepeat[k]=r;seenBlocks[k]=visited;}
            ++visited;
            if(retained)--retained;
            else {if(extra>UINT64_MAX-ALLOC_BLOCK_SIZE-sizeof(alloc_hdr_t))return 0;
                extra+=ALLOC_BLOCK_SIZE+sizeof(alloc_hdr_t);}
            remaining=ALLOC_BLOCK_SIZE;
        }
        remaining-=size;
    }
    forecast->remaining=remaining;forecast->retained=retained;forecast->bytes=extra;return 1;
}
int AllocAdditionalBacking(alloc_handle_t *handle,const uint64_t *pattern,
                          unsigned count,uint64_t repeats,uint64_t *bytes)
{
    AllocForecast forecast;
    if(!bytes||!AllocForecastBegin(handle,&forecast)||
       !AllocForecastAppend(&forecast,pattern,count,repeats))return 0;
    *bytes=forecast.bytes;return 1;
}


/*
**  AllocFreePool()
**
**  Free the memory used by the current pool.
**  Don't use where AllocReset() could be used.
*/

void  AllocFreePool()
{
    alloc_hdr_t  *tmp, *hdr;
    if (!root) return;
    hdr = root->first;
    auditPool(root, "pool_release_begin");

    while (hdr != NULL)
    {
        MSXBudgetAllocation backing = hdr->backing;
        tmp = hdr->next;
        free((char *) hdr->block);
        free((char *) hdr);
        MSXresidentBudget_releaseAllocation((MSXResidentBudget *)backing.budgetOwner, &backing);
        hdr = tmp;
    }
    root->stats.blockCount = root->stats.backingBytes = 0;
    auditPool(root, "pool_released");
    if (root->audit) fclose(root->audit);
    {
        MSXBudgetAllocation backing = root->backing;
        free((char *) root);
        MSXresidentBudget_releaseAllocation((MSXResidentBudget *)backing.budgetOwner, &backing);
    }
    root = NULL;
}

#ifdef MSX_RESIDENT_TEST_API
#include "msxresident_inventory.h"
void MSXinv_Pool(MSXInventory *s,void *handle,const char *name)
{ alloc_root_t *p=(alloc_root_t *)handle;alloc_hdr_t *b;char key[128];uint32_t i=0;if(!p)return;snprintf(key,sizeof(key),"%s.root",name);MSXinv_ticket(s,key,p,p->backing.bytes,&p->backing);for(b=p->first;b;b=b->next){snprintf(key,sizeof(key),"%s.block[%u]",name,i++);MSXinv_ticket(s,key,b->block,(uint64_t)(b->end-b->block),&b->backing);} }
void MSXinv_poolIndexed(MSXInventory *s,const char *module,uint32_t k,uint32_t j,const char *field,const void *ptr,void *handle)
{ alloc_root_t *p=(alloc_root_t *)handle;alloc_hdr_t *b;const void *owner=NULL;char key[256];uintptr_t address=(uintptr_t)ptr;if(s->closed){s->error=1;return;}if(ptr&&p)for(b=p->first;b;b=b->next)if(address>=(uintptr_t)b->block&&address<(uintptr_t)b->free){owner=b->block;break;}if(ptr&&!owner)s->error=1;snprintf(key,sizeof(key),"%s[%u,%u].%s",module,k,j,field);MSXinv_alias(s,key,ptr,owner); }
#endif
