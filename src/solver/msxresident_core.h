#ifndef MSXRESIDENT_CORE_H
#define MSXRESIDENT_CORE_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef enum { MSX_RESIDENT_OFF, MSX_RESIDENT_SHADOW, MSX_RESIDENT_RESIDENT } MSXResidentMode;
typedef enum { MSX_RESIDENT_OK=0, MSX_RESIDENT_DISABLED=1, MSX_RESIDENT_FALLBACK_WHOLE_LINK=2,
 MSX_RESIDENT_ERR_ARGUMENT=-1, MSX_RESIDENT_ERR_CAPACITY=-2, MSX_RESIDENT_ERR_GENERATION=-3,
 MSX_RESIDENT_ERR_UNSUPPORTED_WALL=-4, MSX_RESIDENT_ERR_UNSUPPORTED_DISPERSION=-5,
 MSX_RESIDENT_ERR_CSV=-6, MSX_RESIDENT_ERR_MEMORY=-7, MSX_RESIDENT_ERR_OVERFLOW=-8,
 MSX_RESIDENT_ERR_CASE_HASH=-9, MSX_RESIDENT_ERR_GPU=-10,
 MSX_RESIDENT_ERR_TRANSFER=-11, MSX_RESIDENT_ERR_POISONED=-12,
 MSX_RESIDENT_ERR_CONFIG=-13, MSX_RESIDENT_ERR_PATH=-14,
 MSX_RESIDENT_ERR_GUARD=-15, MSX_RESIDENT_ERR_SOLVER=-16,
 MSX_RESIDENT_ERR_SCOPE=-17 } MSXResidentStatus;
typedef struct {
 MSXResidentMode mode; int strict, guard, segmentStorage, solver, gpuSolver;
 int gpuScope, gpuReact, gpuOde, gpuEquil, gpuFormula, hasWall, hasDispersion;
 const char *capacityFile, *caseHash;
} MSXResidentConfig;
typedef struct { uint32_t linkIndex,capacity,head,tail,count; int32_t orient; uint64_t epoch; } MSXResidentPipeDesc;
typedef struct { double volume,hstep,hresponse,uresponse,dresponse; uint64_t parcelId; uint32_t generation; const double *c,*lastc; } MSXResidentPayload;
typedef struct { uint32_t linkIndex,slot,generation,used; MSXResidentPayload payload; } MSXResidentSlotPatch;
typedef struct { uint32_t linkIndex,reserved; MSXResidentPipeDesc descriptor; } MSXResidentDescriptorPatch;
typedef struct { const MSXResidentDescriptorPatch *descriptor; uint32_t descriptorCount; const MSXResidentSlotPatch *slot; uint32_t slotCount; } MSXResidentPatchBatch;
typedef struct { uint32_t linkIndex,slot,generation,boundarySide; uint64_t pipeEpoch; double requestedVolume; } MSXResidentHandoffItem;
typedef struct { uint32_t linkIndex,boundarySide,itemCount; uint64_t pipeEpoch; double flowVolume; const MSXResidentHandoffItem *item; } MSXResidentHandoffPlan;
typedef struct { uint32_t linkIndex,slot,generation,boundarySide; uint64_t pipeEpoch; MSXResidentPayload payload; } MSXResidentHandoffResult;
/* Opaque, caller-owned handle for a prepared CPU materialization batch. */
typedef struct { void *opaque; } MSXResidentHandoffTransaction;
typedef struct { uint64_t descriptorPatches,slotUploads,slotInvalidates,reverses,handoffRequests,fallbacks,generationFailures; } MSXResidentStats;
/* Indexed by link like capacity/base; parsed from the authoritative CSV. */
typedef struct { uint32_t nLinks,totalSlots,speciesStride; const uint32_t *capacity,*base,*guard; } MSXResidentLayout;
/* Read-only snapshot of one published resident row.  It deliberately exposes
   no CPU topology pointer: callers may only build GPU active batches. */
typedef struct { uint32_t linkIndex,slot,globalRow,generation,descriptorHead,descriptorCount; int32_t descriptorOrient; uint64_t descriptorEpoch,parcelId; double volume; } MSXResidentActiveRow;
MSXResidentStatus MSXresident_open(const char*); void MSXresident_close(void); int MSXresident_isOpen(void);
void MSXresident_setMode(MSXResidentMode,int); MSXResidentMode MSXresident_mode(void);
MSXResidentStatus MSXresident_validateFeatures(int,int); MSXResidentStatus MSXresident_verifyCaseHash(const char*);
/* Pure configuration check: no CUDA allocation, topology mutation, or fallback. */
MSXResidentStatus MSXresident_validateConfig(const MSXResidentConfig *config);
MSXResidentStatus MSXresident_observePipe(uint32_t,const uint64_t*,const MSXResidentPayload*,uint32_t,int32_t);
MSXResidentStatus MSXresident_getSlotPayload(uint32_t,uint32_t,uint32_t,MSXResidentPayload*);
MSXResidentStatus MSXresident_getPatches(MSXResidentPatchBatch*); void MSXresident_clearPatches(void);
MSXResidentStatus MSXresident_getLayout(MSXResidentLayout *);
/* Enumerates exactly the used rows in published descriptor order.  ``rows``
   is caller-owned/preallocated; duplicate rows or a descriptor/count mismatch
   are rejected fail-closed. */
MSXResidentStatus MSXresident_enumerateActive(MSXResidentActiveRow *rows,
                                              uint32_t capacity,
                                              uint32_t *count);
/* Complete fixed image: one descriptor per link and one patch per owned slot;
   unused rows are included with used=0. Valid until close/reset. */
MSXResidentStatus MSXresident_getInitialBatch(MSXResidentPatchBatch *);
/* Initial-image staging is intentionally metadata-only.  It lets the runtime
   construct and upload a complete GPU image before CPU Hybrid topology is
   committed.  Commit contains no allocation or GPU work. */
MSXResidentStatus MSXresident_beginInitialImage(void);
MSXResidentStatus MSXresident_stageInitialPipe(uint32_t, const uint64_t *,
                                               const MSXResidentPayload *,
                                               uint32_t, int32_t);
MSXResidentStatus MSXresident_commitInitialImage(void);
void MSXresident_abortInitialImage(void);
MSXResidentStatus MSXresident_planHandoff(uint32_t,double,double,uint32_t,double,int,MSXResidentHandoffPlan*);
/* Caller-owned plan storage: unlike planHandoff this never exposes State's
   reusable scratch array, so plans for several links may coexist. */
MSXResidentStatus MSXresident_planHandoffInto(uint32_t,double,double,uint32_t,double,int,
                                              MSXResidentHandoffItem*,uint32_t,
                                              MSXResidentHandoffPlan*);
MSXResidentStatus MSXresident_planAllHandoffInto(uint32_t,uint32_t,
                                                 MSXResidentHandoffItem*,uint32_t,
                                                 MSXResidentHandoffPlan*);
/* This validates and copies returned rows only; it never changes topology. */
MSXResidentStatus MSXresident_applyHandoff(const MSXResidentHandoffResult*,uint32_t);
/* The transaction/token workspaces are allocated by open/preHybrid. Prepare
   only fills those fixed arenas (Boundary Pseg pool ownership is separate),
   and does not change topology, dense Core, or Resident metadata. Commit is
   allocation-free; abort is always safe. */
MSXResidentStatus MSXresident_prepareHandoffTransaction(const MSXResidentHandoffPlan*,const MSXResidentHandoffResult*,uint32_t,MSXResidentHandoffTransaction*);
/* Revalidates an entire prepared batch without allocation or mutation. */
MSXResidentStatus MSXresident_validateHandoffTransactions(const MSXResidentHandoffTransaction*,uint32_t);
/* No-fail primitive: callers must final-validate the whole batch first. */
void MSXresident_commitHandoffTransaction(MSXResidentHandoffTransaction*);
void MSXresident_abortHandoffTransaction(MSXResidentHandoffTransaction*);
const MSXResidentStats *MSXresident_stats(void);
/* Test seam: counts allocations made by this module.  The fault index is
   relative to the next module allocation; -1 disables injection. */
uint64_t MSXresident_testAllocationCount(void);
void MSXresident_testResetAllocationCount(void);
void MSXresident_testFailAllocationAfter(int64_t allocationIndex);
#ifdef __cplusplus
}
#endif
#endif
