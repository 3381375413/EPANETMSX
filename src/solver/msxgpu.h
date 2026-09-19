#ifndef MSXGPU_H
#define MSXGPU_H

#include "msxtypes.h"
#include "msxresident_core_cuda.h"

double MSXgpu_wallTimeMs(void);
void MSXgpu_profileInit(void);
MSXProfileMode MSXgpu_profileMode(void);
int MSXgpu_profileStageEnabled(void);
int MSXgpu_profileDetailEnabled(void);
int MSXgpu_profileDetailGroupEnabled(MSXProfileDetailGroup group);
int MSXgpu_cpuChemistryTimingEnabled(void);
const char *MSXgpu_profileModeName(void);
const char *MSXgpu_profileDetailName(void);
void MSXgpu_recordInitTime(MSXInitPhase phase, double ms);

/* Detail-only transfer counters.  They are intentionally opaque to the
   solver; the process summary owns their representation and coverage. */
void MSXgpu_profileRecordNormalHandoff(double planMs, double fetchMs,
                                       double prepareMs, double validateMs,
                                       double commitMs,
                                       uint64_t rows, uint64_t d2hBytes,
                                       uint64_t d2hCalls);
void MSXgpu_profileRecordFallbackHandoff(double planMs, double fetchMs,
                                         double prepareMs, double validateMs,
                                         double commitMs,
                                         uint64_t rows, uint64_t d2hBytes,
                                         uint64_t d2hCalls);
/* Records logical row transitions; JSON keeps these separate from the
   underlying patch/API call count. */
void MSXgpu_profileRecordDemote(uint64_t rows, double ms);
void MSXgpu_profileRecordPromote(uint64_t rows, double ms);
void MSXgpu_profileRecordPatch(uint64_t descriptors, uint64_t slots,
                               uint64_t h2dBytes, uint64_t apiCalls,
                               int demoteClass);
void MSXgpu_profileRecordHyd(uint64_t bytes, uint64_t apiCalls);
void MSXgpu_profileRecordDiagnostic(uint64_t bytes, uint64_t apiCalls);
void MSXgpu_profileRecordReacted(uint64_t bytes, uint64_t apiCalls);
void MSXgpu_profileRecordAggregate(uint64_t queries, uint64_t cacheHits,
                                   uint64_t rebuilds);
/* Resident Rebalance A0 accounting.  The record is emitted only when the
   detail demote group is selected; callers may therefore keep the same
   instrumentation seam in normal/off runs without taking diagnostic clocks. */
typedef struct
{
    double rb_parent_ms;
    double rb_scan_ms;
    double rb_empty_init_ms;
    double rb_plan_ms;
    double rb_preflush_ms;
    double rb_fetch_ms;
    double rb_validate_commit_ms;
    double rb_promote_ms;
    double rb_residual_ms;
    uint64_t scan_passes;
    uint64_t core_visits;
    uint64_t boundary_visits;
    uint64_t rebuilt_links;
    uint64_t init_promotes;
    uint64_t demote_rows;
    uint64_t head_rows;
    uint64_t tail_rows;
    uint64_t promote_rows;
    uint64_t patch_descriptors;
    uint64_t patch_rows;
    uint64_t validation_capacity_visits;
    uint64_t commit_rows;
    uint64_t id_lookup_probes;
    uint64_t residual_negative_over_1pct;
} MSXRebalanceMetrics;
void MSXgpu_profileRecordRebalance(const MSXRebalanceMetrics *metrics);
/* Runtime/driver phase accounting.  These helpers are no-ops in profile
   off mode and never introduce a CUDA synchronization. */
void MSXgpu_profileRunPhase(MSXProfileRunPhase phase, double ms);
void MSXgpu_profileRunCount(MSXProfileRunPhase phase, uint64_t count);
const MSXProfileRunTotals *MSXgpu_getProfileRunTotals(void);
void MSXgpu_profileRecordTransfer(int direction, int scope, uint64_t bytes,
                                  double apiMs);
#define MSX_PROFILE_TRANSFER_H2D 1
#define MSX_PROFILE_TRANSFER_D2H 2
#define MSX_PROFILE_TRANSFER_SCOPE_OTHER 0
#define MSX_PROFILE_TRANSFER_SCOPE_PATCH 1
#define MSX_PROFILE_TRANSFER_SCOPE_HANDOFF 2
#define MSX_PROFILE_TRANSFER_SCOPE_HYD 3
#define MSX_PROFILE_TRANSFER_SCOPE_DIAGNOSTIC 4
#define MSX_PROFILE_TRANSFER_SCOPE_REACTED 5
#define MSX_PROFILE_TRANSFER_SCOPE_PATCH_DESCRIPTOR 6
#define MSX_PROFILE_TRANSFER_SCOPE_PATCH_STAGE 7
#define MSX_PROFILE_TRANSFER_SCOPE_PATCH_PAYLOAD 8
#define MSX_PROFILE_TRANSFER_SCOPE_ACTIVE 9
#define MSX_PROFILE_TRANSFER_SCOPE_FULL_SYNC 10
#define MSX_PROFILE_TRANSFER_SCOPE_AGGREGATE 11
int MSXgpu_openTiming(void);
void MSXgpu_closeTiming(void);
int MSXcpu_openTiming(void);
void MSXcpu_closeTiming(void);
int MSXgpu_validateStrict(void);
void MSXgpu_beginStep(double simTimeSec);
void MSXgpu_endStep(int errorCode);
void MSXgpu_reactBegin(void);
void MSXgpu_reactEnd(void);
int MSXgpu_reactPipeSegments(double dt);
int MSXgpu_prepareResidentContext(void);
int MSXgpu_openResidentPrograms(void);
void MSXgpu_closeResidentPrograms(void);
typedef struct {
    int code;
    int stage;
    int sid;
    int pipe;
    int species;
    int expr;
    int iter;
    double value;
} MSXgpuResidentError;
typedef struct {
    uint64_t magic;
    uint64_t batchId;
    uint32_t itemCount;
    int inFlight;
    MSXResidentGpu *gpu;
    MSXResidentGpuDeviceView view;
    uint64_t topologyVersion;
    MSXgpuResidentError gpuError;
    double submitMs;
} MSXgpuResidentCoreToken;
int MSXgpu_submitResidentCore(MSXResidentGpu *, const MSXResidentActiveBatch *,
                              double dt, MSXgpuResidentCoreToken *);
int MSXgpu_submitResidentCoreHyd(MSXResidentGpu *, const MSXResidentActiveBatch *,
                                 const MSXResidentHydView *, double dt,
                                 MSXgpuResidentCoreToken *);
int MSXgpu_submitResidentCoreHydPrepared(MSXResidentGpu *,
                                         MSXResidentGpuActiveWriter *,
                                         const MSXResidentHydView *, double dt,
                                         MSXgpuResidentCoreToken *);
int MSXgpu_finishResidentCore(MSXResidentGpu *, MSXgpuResidentCoreToken *,
                              MSXResidentGpuReactResult *);
int MSXgpu_abortResidentCore(MSXResidentGpu *, MSXgpuResidentCoreToken *);
int MSXgpu_reactResidentCore(MSXResidentGpu *, const MSXResidentActiveBatch *,
                             double dt, MSXResidentGpuReactResult *);
void MSXgpu_addOdeTime(double ms);
void MSXgpu_addEquilTime(double ms);
void MSXgpu_addFormulaTime(double ms);

#endif
