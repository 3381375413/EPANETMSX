#ifndef MSXRESIDENT_CAPACITY_H
#define MSXRESIDENT_CAPACITY_H
#include "msxresident_core_cuda.h"
MSXResidentStatus MSXresidentCapacity_memoryConfig(MSXResidentMemoryConfig *);
MSXResidentStatus MSXresidentCapacity_fileBudget(const MSXResidentLayout *);
typedef struct {
 double volume,qmin,qmax,vmin;
 uint64_t positiveSteps,zeroMs,reversals;
 uint32_t initial,upper,predicted,requested,admission,guard;
 int lastDirection;
} MSXResidentCapacityPipe;
/* Pure integer interval and capped prediction helpers, shared with tests. */
int MSXresidentCapacity_interval(int64_t,int64_t,int64_t,uint64_t *,int64_t *);
int MSXresidentCapacity_predict(MSXResidentCapacityPipe *,uint32_t);
int MSXresidentCapacity_getPrediction(uint32_t,MSXResidentCapacityPipe *);
int MSXresidentCapacity_prepare(void);
MSXResidentStatus MSXresidentCapacity_openPlan(const char *);
int MSXresidentCapacity_retryBudget(void);
void MSXresidentCapacity_noteStartupRetry(const char *reason);
int MSXresidentCapacity_startupAttempt(void);
void MSXresidentCapacity_close(void);
void MSXresidentCapacity_recordOwnership(void);
void MSXresidentCapacity_writeUsage(void);
void MSXresidentCapacity_addCpuReactMs(uint32_t,double);
void MSXresidentCapacity_writeProcessMetrics(void);
void MSXresidentCapacity_setDeviceBaseline(uint64_t);
void MSXresidentCapacity_recordDeviceMemory(void);
void MSXresidentCapacity_writeTransferStats(const MSXResidentGpuTransferStats *);
extern int MSXResidentCapacityAuditEnabled;
int MSXresidentCapacity_auditReact(uint32_t,uint64_t,int);
int MSXresidentCapacity_auditFinish(void);
#endif
