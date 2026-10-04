/*
**  mempool.h
**
**  Header for mempool.c
**
**  The type alloc_handle_t provides an opaque reference to the
**  alloc pool - only the alloc routines know its structure.
*/

#ifndef MEMPOOL_H
#define MEMPOOL_H

#include <stdint.h>
#include "msxresident_budget.h"

/* Requested backing bytes, not OS resident/commit bytes. Reset retains blocks. */
typedef struct
{
    uint64_t poolId, blockCount, backingBytes, peakBackingBytes;
    uint64_t allocationCalls, requestedBytes, alignedBytes, resetCount;
    uint64_t failedAllocations;
} AllocPoolStats;

typedef struct
{
   long  dummy;
}  alloc_handle_t;

alloc_handle_t *AllocInit(void);
alloc_handle_t *AllocInitNamed(const char *owner);
char           *Alloc(long);
alloc_handle_t *AllocSetPool(alloc_handle_t *);
void            AllocReset(void);
void            AllocFreePool(void);
int             AllocGetPoolStats(alloc_handle_t *, AllocPoolStats *);
/* Dry-run the exact fixed-block cursor, including retained blocks. */
typedef struct
{
    uint64_t remaining, retained, bytes;
} AllocForecast;
int             AllocForecastBegin(alloc_handle_t *, AllocForecast *);
int             AllocForecastAppend(AllocForecast *, const uint64_t *pattern,
                                   unsigned patternCount, uint64_t repeats);
int             AllocAdditionalBacking(alloc_handle_t *,const uint64_t *pattern,
                                      unsigned patternCount,uint64_t repeats,uint64_t *bytes);
int             AllocBindReservation(alloc_handle_t *,MSXBudgetTicket *);
#ifdef MSX_RESIDENT_TEST_API
/* Fail the Nth subsequent block creation; zero disables the test fault. */
void            AllocTestFailBlock(unsigned);
#endif

#endif
