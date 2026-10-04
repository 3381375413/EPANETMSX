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

#endif
