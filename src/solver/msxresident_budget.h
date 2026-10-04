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
typedef struct MSXResidentBudget {
    uint64_t hostLimit, deviceLimit;
    uint64_t allocated[MSX_MEMORY_CLASS_COUNT], reserved[MSX_MEMORY_CLASS_COUNT];
    uint64_t peakHostBytes, peakDeviceBytes, rejectedReservations;
    uint64_t nextSequence;
    volatile long lock;
    struct MSXResidentBudget *parent;
} MSXResidentBudget;
typedef enum {MSX_BUDGET_CPU_POOL,MSX_BUDGET_UPLOAD,MSX_BUDGET_SCRATCH,
              MSX_BUDGET_TEMPORARY,MSX_BUDGET_DOMAIN_COUNT} MSXBudgetDomain;
void MSXresidentBudget_init(MSXResidentBudget *budget);
MSXBudgetStatus MSXresidentBudget_initChild(MSXResidentBudget *,MSXResidentBudget *parent);
MSXResidentBudget *MSXresidentBudget_domain(MSXBudgetDomain);
MSXResidentBudget *MSXresidentBudget_global(void);
/* Project observations only; begin refuses live backing/reservations and
   preserves ticket sequences. End emits one summary, including real residue. */
MSXBudgetStatus MSXresidentBudget_beginSession(void);
int MSXresidentBudget_endSession(const char *path);
MSXBudgetStatus MSXresidentBudget_configure(MSXResidentBudget *, uint64_t host, uint64_t device);
MSXBudgetStatus MSXresidentBudget_reserve(MSXResidentBudget *, MSXMemoryClass, uint64_t, MSXBudgetTicket *);
/* Partition a reserved peak without increasing the root total. */
MSXBudgetStatus MSXresidentBudget_split(MSXBudgetTicket *,MSXResidentBudget *,uint64_t,MSXBudgetTicket *);
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
