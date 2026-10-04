#ifndef MSX_RESIDENT_BUDGET_H
#define MSX_RESIDENT_BUDGET_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef enum { MSX_MEMORY_PAGEABLE, MSX_MEMORY_PINNED, MSX_MEMORY_DEVICE,
    MSX_MEMORY_CLASS_COUNT } MSXMemoryClass;
typedef enum { MSX_BUDGET_OK, MSX_BUDGET_LIMIT, MSX_BUDGET_STATE,
    MSX_BUDGET_INVALID } MSXBudgetStatus;
/* Tokens are owned by one caller and must not be copied while live. */
typedef struct { uint64_t sequence, bytes; uintptr_t budgetOwner; unsigned memoryClass, live; } MSXBudgetTicket;
typedef struct { uint64_t sequence, bytes; uintptr_t budgetOwner; unsigned memoryClass, live; } MSXBudgetAllocation;
typedef struct {
    uint64_t hostLimit, deviceLimit;
    uint64_t allocated[MSX_MEMORY_CLASS_COUNT], reserved[MSX_MEMORY_CLASS_COUNT];
    uint64_t peakHostBytes, peakDeviceBytes, rejectedReservations;
    uint64_t nextSequence;
    volatile long lock;
} MSXResidentBudget;
void MSXresidentBudget_init(MSXResidentBudget *budget);
MSXResidentBudget *MSXresidentBudget_global(void);
MSXBudgetStatus MSXresidentBudget_configure(MSXResidentBudget *, uint64_t host, uint64_t device);
MSXBudgetStatus MSXresidentBudget_reserve(MSXResidentBudget *, MSXMemoryClass, uint64_t, MSXBudgetTicket *);
MSXBudgetStatus MSXresidentBudget_commitAllocation(MSXResidentBudget *, MSXBudgetTicket *, uint64_t actual, MSXBudgetAllocation *);
MSXBudgetStatus MSXresidentBudget_cancelReservation(MSXResidentBudget *, MSXBudgetTicket *);
MSXBudgetStatus MSXresidentBudget_releaseAllocation(MSXResidentBudget *, MSXBudgetAllocation *);
void MSXresidentBudget_snapshot(MSXResidentBudget *, MSXResidentBudget *snapshot);
void MSXresidentBudget_auditAllocation(MSXResidentBudget *, const MSXBudgetAllocation *,
    const char *source, unsigned line, uint64_t payloadBytes);
#ifdef __cplusplus
}
#endif
#endif
