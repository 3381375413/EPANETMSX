#define _CRT_SECURE_NO_WARNINGS

#include <math.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#include <ctype.h>
#include <sys/stat.h>
#include <time.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOGDI
#include <direct.h>
#include <io.h>
#include <sys/utime.h>
#include <windows.h>
#else
#include <dirent.h>
#include <unistd.h>
#include <utime.h>
#endif

#include "msxgpu.h"

typedef char MSXresidentHydStrideAbiAssert[
    (MSX_RESIDENT_HYD_STRIDE == MAX_HYD_VARS) ? 1 : -1];

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef EPANETMSX_CUDA_ENABLED
#include <cuda.h>
#include <nvrtc.h>
#include "msxreact_transfer_cuda.h"
#endif
#include "msxsegment_storage.h"

extern MSXproject MSX;

#define GPU_MAX_SPECIES 64
#define GPU_MAX_EQUIL   16
#define GPU_MAX_STACK   128
#define GPU_CACHE_VERSION "v24_ros2_dims_hydpipe_abi1"
#define GPU_ROS2_MAX_RATE_SPECIES 16
#define RK5_FAST_BUCKETS 6
#define GPU_CACHE_MAX_BYTES (200LL * 1024LL * 1024LL)
#define GPU_CACHE_MAX_AGE_SECONDS (30LL * 24LL * 60LL * 60LL)
#define GPU_PATH_MAX 1024

typedef struct
{
    int nSpecies;
    int nRate;
    int nEquil;
    int nFormula;
} GpuSpecializedDimensions;

typedef struct
{
    int valid;
    int compiler;
    uint64_t cacheKey;
    GpuSpecializedDimensions dims;
} GpuModelBinding;

static GpuModelBinding GpuModel;

typedef struct
{
    int opcode;
    int ivar;
    double fvalue;
} GpuInstrHost;

typedef struct
{
    int first;
    int count;
} GpuProgramHost;

typedef struct
{
    int code;
    int stage;
    int sid;
    int pipe;
    int species;
    int expr;
    int iter;
    double value;
} GpuErrorHost;

static FILE *TimingFile = NULL;
static FILE *CpuTimingFile = NULL;
static long long StepIndex = 0;
static int ReactActive = 0;
static double StepStartMs = 0.0;
static MSXGpuTiming TotalTiming;
static MSXProfileRunTotals RunTotals;
static MSXGpuTiming CpuTotalTiming;
static double InitTiming[MSX_INIT_PHASE_COUNT];
static double InitTimingTotal = 0.0;
static MSXProfileMode ProfileMode = (MSXProfileMode)-1;
static char ProfileDetail[128] = "all";
static unsigned int ProfileDetailMask = MSX_PROFILE_DETAIL_ALL;
static int ProfileAnnounced = 0;
static int ProfileExplicit = 0;
static int GpuTimingOutputEnabled = 0;
static int Ros2RawErrorReason = 0;

typedef struct
{
    uint64_t normalHandoffRows, normalHandoffBatches;
    uint64_t normalHandoffD2hBytes, normalHandoffD2hCalls;
    double normalPlanMs, normalFetchMs, normalPrepareMs;
    double normalValidateMs, normalCommitMs;
    uint64_t fallbackHandoffRows, fallbackHandoffBatches;
    uint64_t fallbackHandoffD2hBytes, fallbackHandoffD2hCalls;
    double fallbackPlanMs, fallbackFetchMs, fallbackPrepareMs;
    double fallbackValidateMs, fallbackCommitMs;
    uint64_t demoteRows, demoteLogicalEvents, demoteApiCalls;
    uint64_t promoteRows, promoteLogicalEvents, promoteApiCalls;
    double demoteMs, promoteMs;
    uint64_t patchDescriptors, patchSlots, patchH2dBytes, patchH2dCalls;
    uint64_t demotePatchDescriptors, demotePatchSlots;
    uint64_t hydH2dBytes, hydH2dCalls;
    uint64_t diagnosticD2hBytes, diagnosticD2hCalls;
    uint64_t reactedD2hBytes, reactedD2hCalls;
    uint64_t aggregateQueries, aggregateCacheHits, aggregateRebuilds;
    MSXRebalanceMetrics rebalance;
} MSXProfileCounters;
static MSXProfileCounters ProfileCounters;
static int Ros2RawErrorSid = -1;
#ifdef EPANETMSX_CUDA_ENABLED
static int ReactTransferReady = 0;
#endif

static int cacheGpuModelBinding(void);
static void clearGpuModelBinding(void);

double MSXgpu_wallTimeMs(void)
{
#ifdef _WIN32
    LARGE_INTEGER counter, frequency;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return 1000.0 * (double)counter.QuadPart / (double)frequency.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1000.0 * (double)ts.tv_sec + (double)ts.tv_nsec / 1000000.0;
#endif
}

static int lastIndex(int objectType)
{
    int result = 0;
    if (objectType >= SPECIES) result += MSX.Nobjects[SPECIES];
    if (objectType >= TERM) result += MSX.Nobjects[TERM];
    if (objectType >= PARAMETER) result += MSX.Nobjects[PARAMETER];
    if (objectType >= CONSTANT) result += MSX.Nobjects[CONSTANT];
    return result;
}

static void setGpuError(int code, int stage, int sid, int pipe, int species,
                        int expr, int iter, double value)
{
    if (!MSX.GpuError.code)
    {
        FILE *f;
        MSX.GpuError.code = code;
        MSX.GpuError.stage = stage;
        MSX.GpuError.sid = sid;
        MSX.GpuError.pipe = pipe;
        MSX.GpuError.species = species;
        MSX.GpuError.expr = expr;
        MSX.GpuError.iter = iter;
        MSX.GpuError.value = value;
        f = fopen("msx_gpu_error.csv", "at");
        if (f)
        {
            long pos = ftell(f);
            if (pos == 0)
                fprintf(f, "code,stage,sid,pipe,species,expr,iter,value,ros2_raw_reason,ros2_raw_sid\n");
            fprintf(f, "%d,%d,%d,%d,%d,%d,%d,%.17g,%d,%d\n",
                    code, stage, sid, pipe, species, expr, iter, value,
                    Ros2RawErrorReason, Ros2RawErrorSid);
            fclose(f);
        }
    }
}

static void writeGpuErrorDetail(const GpuErrorHost *gpuErr, int nSpecies, const double *c)
{
    FILE *f;
    int m;
    if (!gpuErr || !gpuErr->code || gpuErr->sid < 0 || !c) return;
    f = fopen("msx_gpu_error_detail.csv", "wt");
    if (!f) return;
    fprintf(f, "code,stage,sid,pipe,species,expr,iter,value\n");
    fprintf(f, "%d,%d,%d,%d,%d,%d,%d,%.17g\n",
            gpuErr->code, gpuErr->stage, gpuErr->sid, gpuErr->pipe,
            gpuErr->species, gpuErr->expr, gpuErr->iter, gpuErr->value);
    fprintf(f, "\nspecies_index,species_id,concentration\n");
    for (m = 1; m <= nSpecies; m++)
    {
        fprintf(f, "%d,%s,%.17g\n", m, MSX.Species[m].id,
                c[gpuErr->sid * (nSpecies + 1) + m]);
    }
    fclose(f);
}

static int countExprNodes(MathExpr *expr)
{
    int n = 0;
    while (expr)
    {
        n++;
        expr = expr->next;
    }
    return n;
}

static int exprHasUnsupportedReference(MathExpr *expr, int allowFormulaSpecies,
                                       int allowTermReference)
{
    const int lastSpecies = lastIndex(SPECIES);
    const int lastTerm = lastIndex(TERM);
    while (expr)
    {
        if (expr->opcode < 3 || (expr->opcode > 28 && expr->opcode != 31))
            return 1;
        if (expr->opcode == 8)
        {
            int ivar = expr->ivar;
            if (!allowFormulaSpecies && ivar >= 1 && ivar <= lastSpecies &&
                MSX.Species[ivar].pipeExprType == FORMULA)
                return 1;
            if (!allowTermReference && ivar > lastSpecies && ivar <= lastTerm)
                return 1;
        }
        expr = expr->next;
    }
    return 0;
}

static int validateModelForGpu(void)
{
    int m;
    int hasRate = 0;
    int hasEquil = 0;
    int hasFormula = 0;
    int nEquil = 0;
    int nRate = 0;

    if (MSX.GpuReactScope != GPU_PIPE_SEGMENT) return ERR_GPU_UNSUPPORTED_FEATURE;
    if (MSX.GpuSolver != EUL && MSX.GpuSolver != RK5 && MSX.GpuSolver != ROS2) return ERR_GPU_SOLVER_UNSUPPORTED;
    if (MSX.Solver != EUL && MSX.Solver != RK5 && MSX.Solver != ROS2) return ERR_GPU_SOLVER_UNSUPPORTED;
    if (MSX.GpuSolver != MSX.Solver) return ERR_GPU_SOLVER_UNSUPPORTED;
    if (MSXsegStorage_isPipeRingEnabled() &&
        MSX.GpuSolver == RK5 &&
        MSX.GpuRk5Mode == GPU_RK5_FAST_BUCKET)
        return ERR_GPU_UNSUPPORTED_FEATURE;
    if (MSX.Coupling == FULL_COUPLING) return ERR_GPU_FULL_COUPLING_UNSUPPORTED;
    if (MSX.Nobjects[SPECIES] > GPU_MAX_SPECIES) return ERR_GPU_UNSUPPORTED_FEATURE;

    for (m = 1; m <= MSX.Nobjects[TERM]; m++)
    {
        if (exprHasUnsupportedReference(MSX.Term[m].expr, 0, 1))
            return ERR_GPU_UNSUPPORTED_FEATURE;
        if (countExprNodes(MSX.Term[m].expr) > GPU_MAX_STACK)
            return ERR_GPU_UNSUPPORTED_FEATURE;
    }

    for (m = 1; m <= MSX.Nobjects[SPECIES]; m++)
    {
        if (MSX.Species[m].pipeExprType == RATE)
        {
            hasRate = 1;
            nRate++;
            if (exprHasUnsupportedReference(MSX.Species[m].pipeExpr, 0, 1))
                return ERR_GPU_UNSUPPORTED_FEATURE;
        }
        else if (MSX.Species[m].pipeExprType == EQUIL)
        {
            hasEquil = 1;
            nEquil++;
            if (exprHasUnsupportedReference(MSX.Species[m].pipeExpr, 0, 1))
                return ERR_GPU_UNSUPPORTED_FEATURE;
        }
        else if (MSX.Species[m].pipeExprType == FORMULA)
        {
            hasFormula = 1;
            if (exprHasUnsupportedReference(MSX.Species[m].pipeExpr, 0, 1))
                return ERR_GPU_UNSUPPORTED_FEATURE;
        }
        if (countExprNodes(MSX.Species[m].pipeExpr) > GPU_MAX_STACK)
            return ERR_GPU_UNSUPPORTED_FEATURE;
    }

    if (MSX.GpuSolver == ROS2 &&
        (nRate < 1 || nRate > GPU_ROS2_MAX_RATE_SPECIES))
        return ERR_GPU_UNSUPPORTED_FEATURE;

    if (nEquil > GPU_MAX_EQUIL) return ERR_GPU_EQUIL_UNSUPPORTED;
    if (hasRate && !MSX.GpuOde) return ERR_GPU_UNSUPPORTED_FEATURE;
    if (hasEquil && !MSX.GpuEquil) return ERR_GPU_UNSUPPORTED_FEATURE;
    if (hasFormula && !MSX.GpuFormula) return ERR_GPU_UNSUPPORTED_FEATURE;
    return 0;
}

int MSXgpu_validateStrict(void)
{
    int gpuWorkRequested = MSX.GpuReact || MSX.GpuOde || MSX.GpuEquil || MSX.GpuFormula;
    int err;
    if (!gpuWorkRequested) return 0;
    err = validateModelForGpu();
    if (err) setGpuError(err, GPU_STAGE_NONE, -1, -1, -1, -1, -1, 0.0);
    return err;
}

int MSXgpu_openTiming(void)
{
    int err;
    MSXgpu_profileInit();
    MSXgpu_closeTiming();
    err = cacheGpuModelBinding();
    if (err) return err;
    StepIndex = 0;
    memset(&TotalTiming, 0, sizeof(TotalTiming));
    memset(&RunTotals, 0, sizeof(RunTotals));
    memset(&ProfileCounters, 0, sizeof(ProfileCounters));
    memset(&InitTiming, 0, sizeof(InitTiming));
    InitTimingTotal = 0.0;
    GpuTimingOutputEnabled = MSXgpu_profileStageEnabled() &&
        (ProfileExplicit || MSX.GpuTiming);
    if (!GpuTimingOutputEnabled) return 0;
    TimingFile = fopen("msx_gpu_timing.csv", "wt");
    if (!TimingFile) return ERR_IO_OUT_FILE;
    fprintf(TimingFile,
        "record,step_count,sim_time_sec,gpu_enabled,gpu_strict,react_gpu,advect_gpu,mix_gpu,release_gpu,disperse_gpu,"
        "ode_gpu,equil_gpu,formula_gpu,full_coupling,equil_time_embedded,total_ms,react_ms,advect_ms,"
        "mix_ms,release_ms,disperse_ms,react_pack_ms,h2d_ms,react_ode_ms,react_equil_ms,react_formula_ms,"
        "d2h_ms,react_unpack_ms,jit_ms,nvrtc_compile_ms,ptx_cache_read_ms,cu_module_load_ms,kernel_lookup_ms,"
        "rk5_nfcn,rk5_naccpt,rk5_nrejct,rk5_last_hstep,rk5_bucket_count,rk5_bucket_launches,"
        "rk5_bucket_max_size,rk5_bucket_reorder_ms,react_count_parallel_ms,react_count_prefix_ms,"
        "react_pack_segment_ms,react_host_alloc_ms,react_device_alloc_ms,react_scatter_ms,"
        "ros2_nfcn,ros2_njac,ros2_naccept,ros2_nreject,ros2_last_hstep,"
        "rk5_fast_mode,rk5_error_code,ros2_error_code,error_code,"
        "resident_boundary_cpu_ms,resident_enumerate_filter_ms,resident_patch_h2d_ms,"
        "resident_active_h2d_ms,resident_diag_d2h_hstep_ms,resident_sync_ms,"
        "resident_initial_upload_ms,resident_handoff_ms,resident_fallback_ms,resident_active_rows\n");
    fflush(TimingFile);
    return 0;
}

void MSXgpu_profileRunPhase(MSXProfileRunPhase phase, double ms)
{
    double *dst = NULL;
    if (!MSXgpu_profileStageEnabled() || phase < 0 ||
        phase >= MSX_PROFILE_RUN_PHASE_COUNT)
        return;
    if (ms < 0.0) ms = 0.0;
    switch (phase)
    {
    case MSX_PROFILE_RUN_QUALITY_API: dst = &RunTotals.quality_api_ms; break;
    case MSX_PROFILE_RUN_HYD_READ: dst = &RunTotals.hyd_read_ms; break;
    case MSX_PROFILE_RUN_HYD_EVAL: dst = &RunTotals.hyd_eval_ms; break;
    case MSX_PROFILE_RUN_FLOW_REORIENT: dst = &RunTotals.flow_reorient_ms; break;
    case MSX_PROFILE_RUN_NODE_SORT: dst = &RunTotals.node_sort_ms; break;
    case MSX_PROFILE_RUN_SEGMENT_INIT: dst = &RunTotals.segment_init_ms; break;
    case MSX_PROFILE_RUN_TRANSPORT_CALL: dst = &RunTotals.transport_call_ms; break;
    case MSX_PROFILE_RUN_OUTER_PATCH: dst = &RunTotals.outer_patch_ms; break;
    case MSX_PROFILE_RUN_REPORT: dst = &RunTotals.report_ms; break;
    case MSX_PROFILE_RUN_REPORT_SNAPSHOT_WAIT: dst = &RunTotals.report_snapshot_wait_ms; break;
    case MSX_PROFILE_RUN_REPORT_CORE_REDUCE: dst = &RunTotals.report_core_reduce_ms; break;
    case MSX_PROFILE_RUN_REPORT_BOUNDARY_SUM: dst = &RunTotals.report_boundary_sum_ms; break;
    case MSX_PROFILE_RUN_REPORT_PACK: dst = &RunTotals.report_pack_ms; break;
    case MSX_PROFILE_RUN_REPORT_WRITE: dst = &RunTotals.report_write_ms; break;
    case MSX_PROFILE_RUN_FINAL_MASS: dst = &RunTotals.final_mass_ms; break;
    case MSX_PROFILE_RUN_OUTER_OTHER: dst = &RunTotals.outer_other_ms; break;
    case MSX_PROFILE_RUN_TRANSPORT_PRE_STEP_SYNC: dst = &RunTotals.transport_pre_step_sync_ms; break;
    case MSX_PROFILE_RUN_TRANSPORT_HANDOFF_PLAN: dst = &RunTotals.transport_handoff_plan_ms; break;
    case MSX_PROFILE_RUN_TRANSPORT_HANDOFF_GPU_FETCH: dst = &RunTotals.transport_handoff_gpu_fetch_ms; break;
    case MSX_PROFILE_RUN_TRANSPORT_HANDOFF_CPU_PREPARE: dst = &RunTotals.transport_handoff_cpu_prepare_ms; break;
    case MSX_PROFILE_RUN_TRANSPORT_REACT: dst = &RunTotals.transport_react_ms; break;
    case MSX_PROFILE_RUN_TRANSPORT_HANDOFF_COMPLETE: dst = &RunTotals.transport_handoff_complete_ms; break;
    case MSX_PROFILE_RUN_TRANSPORT_SCALAR_SYNC: dst = &RunTotals.transport_scalar_sync_ms; break;
    case MSX_PROFILE_RUN_TRANSPORT_ADVECT: dst = &RunTotals.transport_advect_ms; break;
    case MSX_PROFILE_RUN_TRANSPORT_TOPOLOGICAL: dst = &RunTotals.transport_topological_ms; break;
    case MSX_PROFILE_RUN_TRANSPORT_REBALANCE: dst = &RunTotals.transport_rebalance_ms; break;
    case MSX_PROFILE_RUN_TRANSPORT_RESIDENT_OBSERVE: dst = &RunTotals.transport_resident_observe_ms; break;
    case MSX_PROFILE_RUN_TRANSPORT_POST_STEP: dst = &RunTotals.transport_post_step_ms; break;
    case MSX_PROFILE_RUN_TRANSPORT_OTHER: dst = &RunTotals.transport_other_ms; break;
    case MSX_PROFILE_RUN_REACT_CPU_BOUNDARY: dst = &RunTotals.react_cpu_boundary_ms; break;
    case MSX_PROFILE_RUN_REACT_CPU_TANK: dst = &RunTotals.react_cpu_tank_ms; break;
    case MSX_PROFILE_RUN_REACT_FLUSH: dst = &RunTotals.react_flush_ms; break;
    case MSX_PROFILE_RUN_REACT_ENUMERATE: dst = &RunTotals.react_enumerate_ms; break;
    case MSX_PROFILE_RUN_REACT_IDENTITY_FILTER: dst = &RunTotals.react_identity_filter_ms; break;
    case MSX_PROFILE_RUN_REACT_PREPARE: dst = &RunTotals.react_prepare_ms; break;
    case MSX_PROFILE_RUN_REACT_GPU_SUBMIT: dst = &RunTotals.react_gpu_submit_ms; break;
    case MSX_PROFILE_RUN_REACT_GPU_WAIT: dst = &RunTotals.react_gpu_wait_ms; break;
    case MSX_PROFILE_RUN_REACT_GPU_FINISH: dst = &RunTotals.react_gpu_finish_ms; break;
    case MSX_PROFILE_RUN_REACT_FULL_SYNC_GATHER: dst = &RunTotals.react_full_sync_gather_ms; break;
    case MSX_PROFILE_RUN_REACT_SYNC_VALIDATE: dst = &RunTotals.react_sync_validate_ms; break;
    case MSX_PROFILE_RUN_REACT_PAYLOAD_APPLY: dst = &RunTotals.react_payload_apply_ms; break;
    case MSX_PROFILE_RUN_REACT_DENSE_APPLY: dst = &RunTotals.react_dense_apply_ms; break;
    case MSX_PROFILE_RUN_REACT_REACTED_MERGE: dst = &RunTotals.react_reacted_merge_ms; break;
    default: break;
    }
    if (dst) *dst += ms;
}

void MSXgpu_profileRunCount(MSXProfileRunPhase phase, uint64_t count)
{
    if (!MSXgpu_profileStageEnabled()) return;
    switch (phase)
    {
    case MSX_PROFILE_RUN_QUALITY_API: RunTotals.quality_api_calls += count; break;
    case MSX_PROFILE_RUN_HYD_READ: RunTotals.hyd_event_count += count; break;
    case MSX_PROFILE_RUN_TRANSPORT_CALL: RunTotals.transport_call_count += count; break;
    case MSX_PROFILE_RUN_TRANSPORT_PRE_STEP_SYNC: RunTotals.quality_substep_count += count; break;
    case MSX_PROFILE_RUN_REPORT: RunTotals.report_count += count; break;
    default: break;
    }
}

const MSXProfileRunTotals *MSXgpu_getProfileRunTotals(void)
{
    return &RunTotals;
}

void MSXgpu_profileRecordTransfer(int direction, int scope, uint64_t bytes,
                                  double apiMs)
{
    int collect = 0;
    if (!MSXgpu_profileDetailEnabled()) return;
    if (scope == MSX_PROFILE_TRANSFER_SCOPE_PATCH ||
        scope == MSX_PROFILE_TRANSFER_SCOPE_PATCH_DESCRIPTOR ||
        scope == MSX_PROFILE_TRANSFER_SCOPE_PATCH_STAGE ||
        scope == MSX_PROFILE_TRANSFER_SCOPE_PATCH_PAYLOAD ||
        scope == MSX_PROFILE_TRANSFER_SCOPE_HANDOFF)
        collect = MSXgpu_profileDetailGroupEnabled(MSX_PROFILE_DETAIL_HANDOFF) ||
                  MSXgpu_profileDetailGroupEnabled(MSX_PROFILE_DETAIL_DEMOTE);
    else if (scope == MSX_PROFILE_TRANSFER_SCOPE_HYD ||
             scope == MSX_PROFILE_TRANSFER_SCOPE_DIAGNOSTIC ||
             scope == MSX_PROFILE_TRANSFER_SCOPE_REACTED)
        collect = MSXgpu_profileDetailGroupEnabled(MSX_PROFILE_DETAIL_DIAGNOSTIC);
    else if (scope == MSX_PROFILE_TRANSFER_SCOPE_AGGREGATE)
        collect = MSXgpu_profileDetailGroupEnabled(MSX_PROFILE_DETAIL_AGGREGATE);
    else if (scope == MSX_PROFILE_TRANSFER_SCOPE_ACTIVE ||
             scope == MSX_PROFILE_TRANSFER_SCOPE_FULL_SYNC)
        collect = MSXgpu_profileDetailGroupEnabled(MSX_PROFILE_DETAIL_CHEM);
    else collect = 1;
    if (!collect) return;
    if (apiMs < 0.0) apiMs = 0.0;
    if (direction == MSX_PROFILE_TRANSFER_H2D)
    {
        RunTotals.transfer_h2d_bytes += bytes;
        RunTotals.transfer_h2d_calls++;
        RunTotals.transfer_h2d_api_ms += apiMs;
        if (scope == MSX_PROFILE_TRANSFER_SCOPE_PATCH ||
            scope == MSX_PROFILE_TRANSFER_SCOPE_PATCH_DESCRIPTOR ||
            scope == MSX_PROFILE_TRANSFER_SCOPE_PATCH_STAGE ||
            scope == MSX_PROFILE_TRANSFER_SCOPE_PATCH_PAYLOAD)
        {
            RunTotals.patch_h2d_bytes += bytes;
            RunTotals.patch_h2d_calls++;
            ProfileCounters.patchH2dBytes += bytes;
            ProfileCounters.patchH2dCalls++;
            if (scope == MSX_PROFILE_TRANSFER_SCOPE_PATCH_DESCRIPTOR)
            {
                RunTotals.patch_descriptor_h2d_bytes += bytes;
                RunTotals.patch_descriptor_h2d_calls++;
            }
            else if (scope == MSX_PROFILE_TRANSFER_SCOPE_PATCH_STAGE)
            {
                RunTotals.patch_stage_h2d_bytes += bytes;
                RunTotals.patch_stage_h2d_calls++;
            }
            else if (scope == MSX_PROFILE_TRANSFER_SCOPE_PATCH_PAYLOAD)
            {
                RunTotals.patch_payload_h2d_bytes += bytes;
                RunTotals.patch_payload_h2d_calls++;
            }
        }
        else if (scope == MSX_PROFILE_TRANSFER_SCOPE_HANDOFF)
        {
            RunTotals.handoff_h2d_bytes += bytes;
            RunTotals.handoff_h2d_calls++;
        }
        else if (scope == MSX_PROFILE_TRANSFER_SCOPE_HYD)
        {
            RunTotals.hyd_h2d_bytes += bytes;
            RunTotals.hyd_h2d_calls++;
            ProfileCounters.hydH2dBytes += bytes;
            ProfileCounters.hydH2dCalls++;
        }
        else if (scope == MSX_PROFILE_TRANSFER_SCOPE_ACTIVE)
        {
            RunTotals.active_h2d_bytes += bytes;
            RunTotals.active_h2d_calls++;
        }
        else if (scope == MSX_PROFILE_TRANSFER_SCOPE_FULL_SYNC)
        {
            RunTotals.full_sync_h2d_bytes += bytes;
            RunTotals.full_sync_h2d_calls++;
        }
        else
        {
            RunTotals.other_h2d_bytes += bytes;
            RunTotals.other_h2d_calls++;
        }
    }
    else if (direction == MSX_PROFILE_TRANSFER_D2H)
    {
        RunTotals.transfer_d2h_bytes += bytes;
        RunTotals.transfer_d2h_calls++;
        RunTotals.transfer_d2h_api_ms += apiMs;
        if (scope == MSX_PROFILE_TRANSFER_SCOPE_HANDOFF)
        {
            RunTotals.handoff_d2h_bytes += bytes;
            RunTotals.handoff_d2h_calls++;
        }
        else if (scope == MSX_PROFILE_TRANSFER_SCOPE_DIAGNOSTIC)
        {
            RunTotals.diagnostic_d2h_bytes += bytes;
            RunTotals.diagnostic_d2h_calls++;
            ProfileCounters.diagnosticD2hBytes += bytes;
            ProfileCounters.diagnosticD2hCalls++;
        }
        else if (scope == MSX_PROFILE_TRANSFER_SCOPE_REACTED)
        {
            RunTotals.reacted_d2h_bytes += bytes;
            RunTotals.reacted_d2h_calls++;
            ProfileCounters.reactedD2hBytes += bytes;
            ProfileCounters.reactedD2hCalls++;
        }
        else if (scope == MSX_PROFILE_TRANSFER_SCOPE_FULL_SYNC)
        {
            RunTotals.full_sync_d2h_bytes += bytes;
            RunTotals.full_sync_d2h_calls++;
        }
        else if (scope == MSX_PROFILE_TRANSFER_SCOPE_AGGREGATE)
        {
            RunTotals.aggregate_d2h_bytes += bytes;
            RunTotals.aggregate_d2h_calls++;
        }
        else
        {
            RunTotals.other_d2h_bytes += bytes;
            RunTotals.other_d2h_calls++;
        }
    }
}

#ifdef EPANETMSX_CUDA_ENABLED
static CUresult profileCuMemcpyHtoD(CUdeviceptr dst, const void *src,
                                    size_t bytes, int scope)
{
    int timed = MSXgpu_profileDetailEnabled();
    double start = timed ? MSXgpu_wallTimeMs() : 0.0;
    CUresult result = cuMemcpyHtoD(dst, src, bytes);
    if (timed)
        MSXgpu_profileRecordTransfer(MSX_PROFILE_TRANSFER_H2D, scope,
                                     (uint64_t)bytes,
                                     MSXgpu_wallTimeMs() - start);
    return result;
}

static CUresult profileCuMemcpyDtoH(void *dst, CUdeviceptr src,
                                    size_t bytes, int scope)
{
    int timed = MSXgpu_profileDetailEnabled();
    double start = timed ? MSXgpu_wallTimeMs() : 0.0;
    CUresult result = cuMemcpyDtoH(dst, src, bytes);
    if (timed)
        MSXgpu_profileRecordTransfer(MSX_PROFILE_TRANSFER_D2H, scope,
                                     (uint64_t)bytes,
                                     MSXgpu_wallTimeMs() - start);
    return result;
}

static CUresult profileCuMemcpyDtoHAsync(void *dst, CUdeviceptr src,
                                         size_t bytes, CUstream stream,
                                         int scope)
{
    int timed = MSXgpu_profileDetailEnabled();
    double start = timed ? MSXgpu_wallTimeMs() : 0.0;
    CUresult result = cuMemcpyDtoHAsync(dst, src, bytes, stream);
    if (timed)
        MSXgpu_profileRecordTransfer(MSX_PROFILE_TRANSFER_D2H, scope,
                                     (uint64_t)bytes,
                                     MSXgpu_wallTimeMs() - start);
    return result;
}
#endif

static int profileEquals(const char *value, const char *word)
{
    unsigned char a, b;
    if (!value || !word) return 0;
    while (*value && *word)
    {
        a = (unsigned char)*value++;
        b = (unsigned char)*word++;
        if (tolower(a) != tolower(b)) return 0;
    }
    return *value == '\0' && *word == '\0';
}

static unsigned int profileDetailToken(const char *token)
{
    if (profileEquals(token, "all")) return MSX_PROFILE_DETAIL_ALL;
    if (profileEquals(token, "chem")) return MSX_PROFILE_DETAIL_CHEM;
    if (profileEquals(token, "handoff") || profileEquals(token, "normal_handoff"))
        return MSX_PROFILE_DETAIL_HANDOFF;
    if (profileEquals(token, "demote") || profileEquals(token, "rebalance"))
        return MSX_PROFILE_DETAIL_DEMOTE;
    if (profileEquals(token, "aggregate")) return MSX_PROFILE_DETAIL_AGGREGATE;
    if (profileEquals(token, "init") || profileEquals(token, "startup"))
        return MSX_PROFILE_DETAIL_INIT;
    if (profileEquals(token, "diagnostic") || profileEquals(token, "diag"))
        return MSX_PROFILE_DETAIL_DIAGNOSTIC;
    return 0;
}

static void profileDetailNameFromMask(unsigned int mask)
{
    static const struct { unsigned int bit; const char *name; } groups[] = {
        { MSX_PROFILE_DETAIL_CHEM, "chem" },
        { MSX_PROFILE_DETAIL_HANDOFF, "handoff" },
        { MSX_PROFILE_DETAIL_DEMOTE, "demote" },
        { MSX_PROFILE_DETAIL_AGGREGATE, "aggregate" },
        { MSX_PROFILE_DETAIL_INIT, "init" },
        { MSX_PROFILE_DETAIL_DIAGNOSTIC, "diagnostic" }
    };
    size_t i;
    size_t used = 0;
    ProfileDetail[0] = '\0';
    if (mask == MSX_PROFILE_DETAIL_ALL)
    {
        strcpy(ProfileDetail, "all");
        return;
    }
    for (i = 0; i < sizeof(groups) / sizeof(groups[0]); i++)
    {
        if (mask & groups[i].bit)
        {
            int n = snprintf(ProfileDetail + used, sizeof(ProfileDetail) - used,
                             "%s%s", used ? "|" : "", groups[i].name);
            if (n < 0 || (size_t)n >= sizeof(ProfileDetail) - used) break;
            used += (size_t)n;
        }
    }
    if (!used) strcpy(ProfileDetail, "none");
}

static void resolveProfileDetail(const char *value)
{
    char token[32];
    size_t n = 0;
    unsigned int mask = 0;
    int sawToken = 0;
    const unsigned char *p = (const unsigned char *)value;

    if (!value || !value[0])
    {
        ProfileDetailMask = MSX_PROFILE_DETAIL_ALL;
        profileDetailNameFromMask(ProfileDetailMask);
        return;
    }
    while (*p)
    {
        unsigned char c = *p++;
        int separator = isspace(c) || c == '|' || c == ',' || c == '+' || c == ';';
        if (!separator && n + 1 < sizeof(token))
            token[n++] = (char)tolower(c);
        if (separator && n)
        {
            unsigned int bit;
            token[n] = '\0';
            bit = profileDetailToken(token);
            if (!bit)
                fprintf(stderr, "WARNING: unknown MSX_PROFILE_DETAIL group '%s'.\n", token);
            else if (bit == MSX_PROFILE_DETAIL_ALL)
                mask = MSX_PROFILE_DETAIL_ALL;
            else
                mask |= bit;
            sawToken = 1;
            n = 0;
        }
    }
    if (n)
    {
        unsigned int bit;
        token[n] = '\0';
        bit = profileDetailToken(token);
        if (!bit)
            fprintf(stderr, "WARNING: unknown MSX_PROFILE_DETAIL group '%s'.\n", token);
        else if (bit == MSX_PROFILE_DETAIL_ALL)
            mask = MSX_PROFILE_DETAIL_ALL;
        else
            mask |= bit;
        sawToken = 1;
    }
    ProfileDetailMask = sawToken ? mask : MSX_PROFILE_DETAIL_ALL;
    profileDetailNameFromMask(ProfileDetailMask);
}

static const char *profileModeName(MSXProfileMode mode)
{
    if (mode == MSX_PROFILE_STAGE) return "stage";
    if (mode == MSX_PROFILE_DETAIL) return "detail";
    return "off";
}

void MSXgpu_profileInit(void)
{
    const char *value;
    const char *detail;
    int explicitMode = 0;

    if (ProfileMode >= MSX_PROFILE_OFF) return;
    value = getenv("MSX_PROFILE");
    detail = getenv("MSX_PROFILE_DETAIL");
    resolveProfileDetail(detail);
    if (value && value[0])
    {
        explicitMode = 1;
        ProfileExplicit = 1;
        if (profileEquals(value, "stage")) ProfileMode = MSX_PROFILE_STAGE;
        else if (profileEquals(value, "detail")) ProfileMode = MSX_PROFILE_DETAIL;
        else if (profileEquals(value, "off") || profileEquals(value, "0") ||
                 profileEquals(value, "false") || profileEquals(value, "no"))
            ProfileMode = MSX_PROFILE_OFF;
        else
        {
            ProfileMode = MSX_PROFILE_OFF;
            fprintf(stderr, "WARNING: invalid MSX_PROFILE='%s'; using off.\n", value);
        }
    }
    else if (MSX.GpuTimingConfigured)
    {
        /* GPU_TIMING remains the legacy opt-in gate.  In particular,
           GPU_TIMING NO must not become a detail profile merely because the
           detail option was present or CPU timing was requested. */
        if (!MSX.GpuTiming)
            ProfileMode = (MSX.CpuTimingConfigured && MSX.CpuTiming)
                        ? MSX_PROFILE_DETAIL : MSX_PROFILE_OFF;
        else
            /* Before MSX_PROFILE, GpuTimingDetail defaulted on.  Preserve
               that behavior for an explicit legacy GPU_TIMING YES, while an
               explicit DETAIL NO selects the cheaper stage profile. */
            ProfileMode = (!MSX.GpuTimingDetailConfigured || MSX.GpuTimingDetail)
                        ? MSX_PROFILE_DETAIL : MSX_PROFILE_STAGE;
    }
    else if (MSX.CpuTimingConfigured && MSX.CpuTiming)
        /* Legacy CPU_TIMING enabled the detailed chemistry clocks even when
           GPU_TIMING itself was not requested.  Retain that behavior without
           opening a GPU timing CSV (handled separately in openTiming). */
        ProfileMode = MSX_PROFILE_DETAIL;
    else
        ProfileMode = MSX_PROFILE_OFF;

    /* An explicit new mode owns the precedence.  Do not let an old option
       reopen detail timing after MSX_PROFILE=off. */
    if (explicitMode && ProfileMode == MSX_PROFILE_OFF)
    {
        MSX.GpuTiming = FALSE;
        MSX.GpuTimingDetail = FALSE;
        MSX.CpuTiming = FALSE;
    }
    if (!ProfileAnnounced)
    {
        printf("\nTIMING,profile_mode=%s,profile_detail=%s\n",
               profileModeName(ProfileMode), ProfileDetail);
        ProfileAnnounced = 1;
    }
}

MSXProfileMode MSXgpu_profileMode(void)
{
    MSXgpu_profileInit();
    return ProfileMode;
}

int MSXgpu_profileStageEnabled(void)
{
    return MSXgpu_profileMode() >= MSX_PROFILE_STAGE;
}

int MSXgpu_profileDetailEnabled(void)
{
    return MSXgpu_profileMode() == MSX_PROFILE_DETAIL;
}

int MSXgpu_profileDetailGroupEnabled(MSXProfileDetailGroup group)
{
    MSXgpu_profileInit();
    return ProfileMode == MSX_PROFILE_DETAIL &&
           (ProfileDetailMask & (unsigned int)group) != 0;
}

const char *MSXgpu_profileModeName(void)
{
    return profileModeName(MSXgpu_profileMode());
}

const char *MSXgpu_profileDetailName(void)
{
    MSXgpu_profileInit();
    return ProfileDetail;
}

void MSXgpu_recordInitTime(MSXInitPhase phase, double ms)
{
    if (!MSXgpu_profileStageEnabled() || phase < 0 || phase >= MSX_INIT_PHASE_COUNT)
        return;
    if (MSXgpu_profileDetailEnabled() &&
        !MSXgpu_profileDetailGroupEnabled(MSX_PROFILE_DETAIL_INIT))
        return;
    if (ms < 0.0) ms = 0.0;
    InitTiming[phase] += ms;
    InitTimingTotal += ms;
}

void MSXgpu_profileRecordNormalHandoff(double planMs, double fetchMs,
                                       double prepareMs, double validateMs,
                                       double commitMs,
                                       uint64_t rows, uint64_t d2hBytes,
                                       uint64_t d2hCalls)
{
    if (!MSXgpu_profileDetailGroupEnabled(MSX_PROFILE_DETAIL_HANDOFF)) return;
    ProfileCounters.normalHandoffBatches++;
    ProfileCounters.normalHandoffRows += rows;
    ProfileCounters.normalHandoffD2hBytes += d2hBytes;
    ProfileCounters.normalHandoffD2hCalls += d2hCalls;
    ProfileCounters.normalPlanMs += planMs;
    ProfileCounters.normalFetchMs += fetchMs;
    ProfileCounters.normalPrepareMs += prepareMs;
    ProfileCounters.normalValidateMs += validateMs;
    ProfileCounters.normalCommitMs += commitMs;
}

void MSXgpu_profileRecordFallbackHandoff(double planMs, double fetchMs,
                                         double prepareMs, double validateMs,
                                         double commitMs,
                                         uint64_t rows, uint64_t d2hBytes,
                                         uint64_t d2hCalls)
{
    if (!MSXgpu_profileDetailGroupEnabled(MSX_PROFILE_DETAIL_HANDOFF)) return;
    ProfileCounters.fallbackHandoffBatches++;
    ProfileCounters.fallbackHandoffRows += rows;
    ProfileCounters.fallbackHandoffD2hBytes += d2hBytes;
    ProfileCounters.fallbackHandoffD2hCalls += d2hCalls;
    ProfileCounters.fallbackPlanMs += planMs;
    ProfileCounters.fallbackFetchMs += fetchMs;
    ProfileCounters.fallbackPrepareMs += prepareMs;
    ProfileCounters.fallbackValidateMs += validateMs;
    ProfileCounters.fallbackCommitMs += commitMs;
}

void MSXgpu_profileRecordDemote(uint64_t rows, double ms)
{
    if (!MSXgpu_profileDetailGroupEnabled(MSX_PROFILE_DETAIL_DEMOTE)) return;
    ProfileCounters.demoteRows += rows;
    /* This hook is emitted once per logical row transition.  It is not a
       CUDA/API call counter; those belong to patch.api_calls below. */
    ProfileCounters.demoteLogicalEvents++;
    ProfileCounters.demoteMs += ms;
}

void MSXgpu_profileRecordPromote(uint64_t rows, double ms)
{
    if (!MSXgpu_profileDetailGroupEnabled(MSX_PROFILE_DETAIL_DEMOTE)) return;
    ProfileCounters.promoteRows += rows;
    ProfileCounters.promoteLogicalEvents++;
    ProfileCounters.promoteMs += ms;
}

void MSXgpu_profileRecordPatch(uint64_t descriptors, uint64_t slots,
                               uint64_t h2dBytes, uint64_t apiCalls,
                               int demoteClass)
{
    MSXProfileDetailGroup group = demoteClass ? MSX_PROFILE_DETAIL_DEMOTE :
                                                MSX_PROFILE_DETAIL_HANDOFF;
    if (!MSXgpu_profileDetailGroupEnabled(group)) return;
    ProfileCounters.patchDescriptors += descriptors;
    ProfileCounters.patchSlots += slots;
    ProfileCounters.patchH2dBytes += h2dBytes;
    ProfileCounters.patchH2dCalls += apiCalls;
    if (demoteClass)
    {
        ProfileCounters.demotePatchDescriptors += descriptors;
        ProfileCounters.demotePatchSlots += slots;
    }
}

void MSXgpu_profileRecordHyd(uint64_t bytes, uint64_t apiCalls)
{
    if (!MSXgpu_profileDetailGroupEnabled(MSX_PROFILE_DETAIL_DIAGNOSTIC)) return;
    ProfileCounters.hydH2dBytes += bytes;
    ProfileCounters.hydH2dCalls += apiCalls;
}

void MSXgpu_profileRecordDiagnostic(uint64_t bytes, uint64_t apiCalls)
{
    if (!MSXgpu_profileDetailGroupEnabled(MSX_PROFILE_DETAIL_DIAGNOSTIC)) return;
    ProfileCounters.diagnosticD2hBytes += bytes;
    ProfileCounters.diagnosticD2hCalls += apiCalls;
}

void MSXgpu_profileRecordReacted(uint64_t bytes, uint64_t apiCalls)
{
    if (!MSXgpu_profileDetailGroupEnabled(MSX_PROFILE_DETAIL_DIAGNOSTIC)) return;
    ProfileCounters.reactedD2hBytes += bytes;
    ProfileCounters.reactedD2hCalls += apiCalls;
}

void MSXgpu_profileRecordAggregate(uint64_t queries, uint64_t cacheHits,
                                   uint64_t rebuilds)
{
    if (!MSXgpu_profileDetailGroupEnabled(MSX_PROFILE_DETAIL_AGGREGATE)) return;
    ProfileCounters.aggregateQueries += queries;
    ProfileCounters.aggregateCacheHits += cacheHits;
    ProfileCounters.aggregateRebuilds += rebuilds;
}

void MSXgpu_profileRecordRebalance(const MSXRebalanceMetrics *m)
{
    MSXRebalanceMetrics *d;
    if (!m || !MSXgpu_profileDetailGroupEnabled(MSX_PROFILE_DETAIL_DEMOTE))
        return;
    d = &ProfileCounters.rebalance;
    d->rb_parent_ms += m->rb_parent_ms;
    d->rb_scan_ms += m->rb_scan_ms;
    d->rb_empty_init_ms += m->rb_empty_init_ms;
    d->rb_plan_ms += m->rb_plan_ms;
    d->rb_preflush_ms += m->rb_preflush_ms;
    d->rb_fetch_ms += m->rb_fetch_ms;
    d->rb_validate_commit_ms += m->rb_validate_commit_ms;
    d->rb_promote_ms += m->rb_promote_ms;
    d->rb_residual_ms += m->rb_residual_ms;
    d->scan_passes += m->scan_passes;
    d->core_visits += m->core_visits;
    d->boundary_visits += m->boundary_visits;
    d->rebuilt_links += m->rebuilt_links;
    d->init_promotes += m->init_promotes;
    d->demote_rows += m->demote_rows;
    d->head_rows += m->head_rows;
    d->tail_rows += m->tail_rows;
    d->promote_rows += m->promote_rows;
    d->patch_descriptors += m->patch_descriptors;
    d->patch_rows += m->patch_rows;
    d->validation_capacity_visits += m->validation_capacity_visits;
    d->commit_rows += m->commit_rows;
    d->id_lookup_probes += m->id_lookup_probes;
    d->residual_negative_over_1pct += m->residual_negative_over_1pct;
}

int MSXcpu_openTiming(void)
{
    MSXcpu_closeTiming();
    memset(&CpuTotalTiming, 0, sizeof(CpuTotalTiming));
    if (!MSX.CpuTiming) return 0;
    CpuTimingFile = fopen("msx_cpu_timing.csv", "wt");
    if (!CpuTimingFile) return ERR_IO_OUT_FILE;
    fprintf(CpuTimingFile,
        "record,step_count,sim_time_sec,full_coupling,total_ms,react_ms,advect_ms,mix_ms,release_ms,disperse_ms,other_ms,react_rateode_ms,react_equil_ms,react_formula_ms,react_other_ms,error_code\n");
    return 0;
}

static void writeCpuTimingRecord(const char *record, const MSXGpuTiming *t)
{
    char line[512];
    int lineLength;
    double otherMs = t->total_ms - t->react_ms - t->advect_ms - t->mix_ms -
        t->release_ms - t->disperse_ms;
    double reactOtherMs = t->react_ms - t->react_ode_ms - t->react_equil_ms -
        t->react_formula_ms;
    if (otherMs < 0.0) otherMs = 0.0;
    if (reactOtherMs < 0.0) reactOtherMs = 0.0;
    lineLength = snprintf(line, sizeof(line),
        "%s,%lld,%.6f,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d",
        record, t->step_index, t->sim_time_sec, t->full_coupling, t->total_ms,
        t->react_ms, t->advect_ms, t->mix_ms, t->release_ms, t->disperse_ms,
        otherMs, t->react_ode_ms, t->react_equil_ms, t->react_formula_ms,
        reactOtherMs, t->error_code);
    if (lineLength < 0 || lineLength >= (int)sizeof(line)) return;
    fprintf(CpuTimingFile, "%s\n", line);
}

void MSXcpu_closeTiming(void)
{
    if (CpuTimingFile)
    {
        writeCpuTimingRecord("TOTAL", &CpuTotalTiming);
        fclose(CpuTimingFile);
    }
    CpuTimingFile = NULL;
}

static void summaryUncollected(FILE *f, int *first, int collected,
                               const char *name)
{
    if (collected) return;
    fprintf(f, "%s\n    \"%s\"", *first ? "" : ",", name);
    *first = 0;
}

static void summaryNumberOrNull(FILE *f, int collected, const char *name,
                                double value, int last)
{
    if (collected)
        fprintf(f, "  \"%s\": %.6f%s\n", name, value, last ? "" : ",");
    else
        fprintf(f, "  \"%s\": null%s\n", name, last ? "" : ",");
}

static void writeProfileSummary(void)
{
    FILE *f;
    int detail = MSXgpu_profileDetailEnabled();
    int initCollected = MSXgpu_profileStageEnabled() &&
        (!detail || MSXgpu_profileDetailGroupEnabled(MSX_PROFILE_DETAIL_INIT));
    int chemCollected = detail && MSXgpu_profileDetailGroupEnabled(MSX_PROFILE_DETAIL_CHEM);
    int handoffCollected = detail && MSXgpu_profileDetailGroupEnabled(MSX_PROFILE_DETAIL_HANDOFF);
    int demoteCollected = detail && MSXgpu_profileDetailGroupEnabled(MSX_PROFILE_DETAIL_DEMOTE);
    int aggregateCollected = detail && MSXgpu_profileDetailGroupEnabled(MSX_PROFILE_DETAIL_AGGREGATE);
    int diagnosticCollected = detail && MSXgpu_profileDetailGroupEnabled(MSX_PROFILE_DETAIL_DIAGNOSTIC);
    int first = 1;
    if (!GpuTimingOutputEnabled) return;
    f = fopen("msx_profile_summary.json", "wt");
    if (!f) return;
    fprintf(f, "{\n");
    fprintf(f, "  \"schema_version\": 2,\n");
    fprintf(f, "  \"mode\": \"%s\",\n", MSXgpu_profileModeName());
    fprintf(f, "  \"detail\": \"%s\",\n", MSXgpu_profileDetailName());
    fprintf(f, "  \"coverage\": \"gpu_driver+resident_runtime+quality_stages+cpu_chemistry\",\n");
    fprintf(f, "  \"collected\": {\"init\": %s, \"chem\": %s, \"handoff\": %s, \"demote\": %s, \"aggregate\": %s, \"diagnostic\": %s},\n",
            initCollected ? "true" : "false", chemCollected ? "true" : "false",
            handoffCollected ? "true" : "false", demoteCollected ? "true" : "false",
            aggregateCollected ? "true" : "false", diagnosticCollected ? "true" : "false");
    fprintf(f, "  \"uncollected\": [");
    summaryUncollected(f, &first, initCollected, "init");
    summaryUncollected(f, &first, chemCollected, "chem");
    summaryUncollected(f, &first, handoffCollected, "handoff");
    summaryUncollected(f, &first, demoteCollected, "demote");
    summaryUncollected(f, &first, aggregateCollected, "aggregate");
    summaryUncollected(f, &first, diagnosticCollected, "diagnostic");
    fprintf(f, "\n  ],\n");
    summaryNumberOrNull(f, initCollected, "init_total_ms", InitTimingTotal, 0);
    summaryNumberOrNull(f, initCollected, "init_context_ms", InitTiming[MSX_INIT_CONTEXT], 0);
    summaryNumberOrNull(f, initCollected, "init_compile_ms", InitTiming[MSX_INIT_COMPILE], 0);
    summaryNumberOrNull(f, initCollected, "init_cache_read_ms", InitTiming[MSX_INIT_CACHE_READ], 0);
    summaryNumberOrNull(f, initCollected, "init_module_ms", InitTiming[MSX_INIT_MODULE], 0);
    summaryNumberOrNull(f, initCollected, "init_alloc_ms", InitTiming[MSX_INIT_ALLOC], 0);
    summaryNumberOrNull(f, initCollected, "init_initial_upload_ms", InitTiming[MSX_INIT_INITIAL_UPLOAD], 0);
    summaryNumberOrNull(f, initCollected, "init_other_ms", InitTiming[MSX_INIT_OTHER], 0);
    fprintf(f, "  \"step_total_ms\": %.6f,\n", TotalTiming.total_ms);
    fprintf(f, "  \"step_react_ms\": %.6f,\n", TotalTiming.react_ms);
    fprintf(f, "  \"step_advect_ms\": %.6f,\n", TotalTiming.advect_ms);
    fprintf(f, "  \"resident_handoff_ms\": %.6f,\n", TotalTiming.resident_handoff_ms);
    fprintf(f, "  \"resident_patch_h2d_ms\": %.6f,\n", TotalTiming.resident_patch_h2d_ms);
    fprintf(f, "  \"resident_sync_ms\": %.6f,\n", TotalTiming.resident_sync_ms);
    fprintf(f, "  \"resident_active_rows\": %.0f,\n", TotalTiming.resident_active_rows);
    {
        const MSXProfileRunTotals *r = &RunTotals;
        fprintf(f, "  \"run_totals\": {\n");
        fprintf(f, "    \"quality_api_ms\": %.6f, \"hyd_read_ms\": %.6f, \"hyd_eval_ms\": %.6f, \"flow_reorient_ms\": %.6f, \"node_sort_ms\": %.6f, \"segment_init_ms\": %.6f,\n",
                r->quality_api_ms, r->hyd_read_ms, r->hyd_eval_ms,
                r->flow_reorient_ms, r->node_sort_ms, r->segment_init_ms);
        fprintf(f, "    \"transport_call_ms\": %.6f, \"outer_patch_ms\": %.6f, \"report_ms\": %.6f, \"report_snapshot_wait_ms\": %.6f, \"report_core_reduce_ms\": %.6f, \"report_boundary_sum_ms\": %.6f, \"report_pack_ms\": %.6f, \"report_write_ms\": %.6f, \"final_mass_ms\": %.6f, \"outer_other_ms\": %.6f,\n",
                r->transport_call_ms, r->outer_patch_ms, r->report_ms,
                r->report_snapshot_wait_ms, r->report_core_reduce_ms,
                r->report_boundary_sum_ms, r->report_pack_ms,
                r->report_write_ms,
                r->final_mass_ms, r->outer_other_ms);
        fprintf(f, "    \"quality_api_calls\": %llu, \"hyd_event_count\": %llu, \"quality_substep_count\": %llu, \"report_count\": %llu, \"transport_call_count\": %llu,\n",
                (unsigned long long)r->quality_api_calls,
                (unsigned long long)r->hyd_event_count,
                (unsigned long long)r->quality_substep_count,
                (unsigned long long)r->report_count,
                (unsigned long long)r->transport_call_count);
        fprintf(f, "    \"transport\": {\"pre_step_sync_ms\": %.6f, \"handoff_plan_ms\": %.6f, \"handoff_gpu_fetch_ms\": %.6f, \"handoff_cpu_prepare_ms\": %.6f, \"react_ms\": %.6f, \"handoff_complete_ms\": %.6f, \"scalar_sync_ms\": %.6f, \"advect_ms\": %.6f, \"topological_transport_ms\": %.6f, \"rebalance_ms\": %.6f, \"resident_observe_ms\": %.6f, \"post_step_ms\": %.6f, \"other_ms\": %.6f},\n",
                r->transport_pre_step_sync_ms, r->transport_handoff_plan_ms,
                r->transport_handoff_gpu_fetch_ms,
                r->transport_handoff_cpu_prepare_ms,
                r->transport_react_ms, r->transport_handoff_complete_ms,
                r->transport_scalar_sync_ms, r->transport_advect_ms,
                r->transport_topological_ms, r->transport_rebalance_ms,
                r->transport_resident_observe_ms, r->transport_post_step_ms,
                r->transport_other_ms);
        fprintf(f, "    \"react\": {\"cpu_boundary_ms\": %.6f, \"cpu_tank_ms\": %.6f, \"flush_ms\": %.6f, \"enumerate_ms\": %.6f, \"identity_filter_ms\": %.6f, \"prepare_ms\": %.6f, \"gpu_submit_ms\": %.6f, \"gpu_wait_ms\": %.6f, \"gpu_finish_ms\": %.6f, \"full_sync_gather_ms\": %.6f, \"sync_validate_ms\": %.6f, \"payload_apply_ms\": %.6f, \"dense_apply_ms\": %.6f, \"reacted_merge_ms\": %.6f}\n",
                r->react_cpu_boundary_ms, r->react_cpu_tank_ms,
                r->react_flush_ms, r->react_enumerate_ms,
                r->react_identity_filter_ms, r->react_prepare_ms,
                r->react_gpu_submit_ms, r->react_gpu_wait_ms,
                r->react_gpu_finish_ms, r->react_full_sync_gather_ms,
                r->react_sync_validate_ms, r->react_payload_apply_ms,
                r->react_dense_apply_ms, r->react_reacted_merge_ms);
        fprintf(f, "  },\n");
        fprintf(f, "  \"api_transfers\": {\"h2d_bytes\": %llu, \"h2d_calls\": %llu, \"h2d_api_ms\": %.6f, \"d2h_bytes\": %llu, \"d2h_calls\": %llu, \"d2h_api_ms\": %.6f, \"patch_h2d_bytes\": %llu, \"patch_h2d_calls\": %llu, \"patch_descriptor_h2d_bytes\": %llu, \"patch_descriptor_h2d_calls\": %llu, \"patch_stage_h2d_bytes\": %llu, \"patch_stage_h2d_calls\": %llu, \"patch_payload_h2d_bytes\": %llu, \"patch_payload_h2d_calls\": %llu, \"handoff_h2d_bytes\": %llu, \"handoff_h2d_calls\": %llu, \"handoff_d2h_bytes\": %llu, \"handoff_d2h_calls\": %llu, \"hyd_h2d_bytes\": %llu, \"hyd_h2d_calls\": %llu, \"diagnostic_d2h_bytes\": %llu, \"diagnostic_d2h_calls\": %llu, \"reacted_d2h_bytes\": %llu, \"reacted_d2h_calls\": %llu, \"active_h2d_bytes\": %llu, \"active_h2d_calls\": %llu, \"full_sync_h2d_bytes\": %llu, \"full_sync_h2d_calls\": %llu, \"full_sync_d2h_bytes\": %llu, \"full_sync_d2h_calls\": %llu, \"aggregate_d2h_bytes\": %llu, \"aggregate_d2h_calls\": %llu, \"other_h2d_bytes\": %llu, \"other_h2d_calls\": %llu, \"other_d2h_bytes\": %llu, \"other_d2h_calls\": %llu},\n",
                (unsigned long long)r->transfer_h2d_bytes,
                (unsigned long long)r->transfer_h2d_calls,
                r->transfer_h2d_api_ms,
                (unsigned long long)r->transfer_d2h_bytes,
                (unsigned long long)r->transfer_d2h_calls,
                r->transfer_d2h_api_ms,
                (unsigned long long)r->patch_h2d_bytes,
                (unsigned long long)r->patch_h2d_calls,
                (unsigned long long)r->patch_descriptor_h2d_bytes,
                (unsigned long long)r->patch_descriptor_h2d_calls,
                (unsigned long long)r->patch_stage_h2d_bytes,
                (unsigned long long)r->patch_stage_h2d_calls,
                (unsigned long long)r->patch_payload_h2d_bytes,
                (unsigned long long)r->patch_payload_h2d_calls,
                (unsigned long long)r->handoff_h2d_bytes,
                (unsigned long long)r->handoff_h2d_calls,
                (unsigned long long)r->handoff_d2h_bytes,
                (unsigned long long)r->handoff_d2h_calls,
                (unsigned long long)r->hyd_h2d_bytes,
                (unsigned long long)r->hyd_h2d_calls,
                (unsigned long long)r->diagnostic_d2h_bytes,
                (unsigned long long)r->diagnostic_d2h_calls,
                (unsigned long long)r->reacted_d2h_bytes,
                (unsigned long long)r->reacted_d2h_calls,
                (unsigned long long)r->active_h2d_bytes,
                (unsigned long long)r->active_h2d_calls,
                (unsigned long long)r->full_sync_h2d_bytes,
                (unsigned long long)r->full_sync_h2d_calls,
                (unsigned long long)r->full_sync_d2h_bytes,
                (unsigned long long)r->full_sync_d2h_calls,
                (unsigned long long)r->aggregate_d2h_bytes,
                (unsigned long long)r->aggregate_d2h_calls,
                (unsigned long long)r->other_h2d_bytes,
                (unsigned long long)r->other_h2d_calls,
                (unsigned long long)r->other_d2h_bytes,
                (unsigned long long)r->other_d2h_calls);
    }
    fprintf(f, "  \"transfers\": {\n");
    if (handoffCollected)
        fprintf(f, "    \"normal_handoff\": {\n      \"batches\": %llu, \"rows\": %llu, \"d2h_bytes\": %llu, \"d2h_calls\": %llu,\n      \"plan_ms\": %.6f, \"fetch_ms\": %.6f, \"prepare_ms\": %.6f, \"validate_ms\": %.6f, \"commit_ms\": %.6f\n    },\n",
                (unsigned long long)ProfileCounters.normalHandoffBatches,
                (unsigned long long)ProfileCounters.normalHandoffRows,
                (unsigned long long)ProfileCounters.normalHandoffD2hBytes,
                (unsigned long long)ProfileCounters.normalHandoffD2hCalls,
                ProfileCounters.normalPlanMs, ProfileCounters.normalFetchMs,
                ProfileCounters.normalPrepareMs,
                ProfileCounters.normalValidateMs, ProfileCounters.normalCommitMs);
    else
        fprintf(f, "    \"normal_handoff\": null,\n");
    if (handoffCollected)
        fprintf(f, "    \"fallback_handoff\": {\n      \"batches\": %llu, \"rows\": %llu, \"d2h_bytes\": %llu, \"d2h_calls\": %llu,\n      \"plan_ms\": %.6f, \"fetch_ms\": %.6f, \"prepare_ms\": %.6f, \"validate_ms\": %.6f, \"commit_ms\": %.6f\n    },\n",
                (unsigned long long)ProfileCounters.fallbackHandoffBatches,
                (unsigned long long)ProfileCounters.fallbackHandoffRows,
                (unsigned long long)ProfileCounters.fallbackHandoffD2hBytes,
                (unsigned long long)ProfileCounters.fallbackHandoffD2hCalls,
                ProfileCounters.fallbackPlanMs, ProfileCounters.fallbackFetchMs,
                ProfileCounters.fallbackPrepareMs,
                ProfileCounters.fallbackValidateMs, ProfileCounters.fallbackCommitMs);
    else
        fprintf(f, "    \"fallback_handoff\": null,\n");
    if (demoteCollected)
        fprintf(f, "    \"demote\": {\n      \"rows\": %llu, \"logical_events\": %llu, \"api_calls\": %llu, \"ms\": %.6f,\n      \"promote_rows\": %llu, \"promote_logical_events\": %llu, \"promote_api_calls\": %llu, \"promote_ms\": %.6f\n    },\n",
                (unsigned long long)ProfileCounters.demoteRows,
                (unsigned long long)ProfileCounters.demoteLogicalEvents,
                (unsigned long long)ProfileCounters.demoteApiCalls, ProfileCounters.demoteMs,
                (unsigned long long)ProfileCounters.promoteRows,
                (unsigned long long)ProfileCounters.promoteLogicalEvents,
                (unsigned long long)ProfileCounters.promoteApiCalls, ProfileCounters.promoteMs);
    else
        fprintf(f, "    \"demote\": null,\n");
    if (handoffCollected || demoteCollected)
        fprintf(f, "    \"patch\": {\n      \"descriptors\": %llu, \"slots\": %llu, \"h2d_bytes\": %llu, \"api_calls\": %llu,\n      \"demote_descriptors\": %llu, \"demote_slots\": %llu\n    },\n",
                (unsigned long long)ProfileCounters.patchDescriptors,
                (unsigned long long)ProfileCounters.patchSlots,
                (unsigned long long)ProfileCounters.patchH2dBytes,
                (unsigned long long)ProfileCounters.patchH2dCalls,
                (unsigned long long)ProfileCounters.demotePatchDescriptors,
                (unsigned long long)ProfileCounters.demotePatchSlots);
    else
        fprintf(f, "    \"patch\": null,\n");
    if (diagnosticCollected)
    {
        fprintf(f, "    \"hyd\": {\"h2d_bytes\": %llu, \"api_calls\": %llu},\n",
                (unsigned long long)ProfileCounters.hydH2dBytes,
                (unsigned long long)ProfileCounters.hydH2dCalls);
        fprintf(f, "    \"diagnostic\": {\"d2h_bytes\": %llu, \"api_calls\": %llu},\n",
                (unsigned long long)ProfileCounters.diagnosticD2hBytes,
                (unsigned long long)ProfileCounters.diagnosticD2hCalls);
        fprintf(f, "    \"reacted\": {\"d2h_bytes\": %llu, \"api_calls\": %llu},\n",
                (unsigned long long)ProfileCounters.reactedD2hBytes,
                (unsigned long long)ProfileCounters.reactedD2hCalls);
    }
    else
    {
        fprintf(f, "    \"hyd\": null,\n");
        fprintf(f, "    \"diagnostic\": null,\n");
        fprintf(f, "    \"reacted\": null,\n");
    }
    if (aggregateCollected)
        fprintf(f, "    \"aggregate\": {\"queries\": %llu, \"cache_hits\": %llu, \"rebuilds\": %llu, \"d2h_bytes\": %llu, \"d2h_calls\": %llu}%s\n",
                (unsigned long long)ProfileCounters.aggregateQueries,
                (unsigned long long)ProfileCounters.aggregateCacheHits,
                (unsigned long long)ProfileCounters.aggregateRebuilds,
                (unsigned long long)RunTotals.aggregate_d2h_bytes,
                (unsigned long long)RunTotals.aggregate_d2h_calls,
                demoteCollected ? "," : "");
    else
        fprintf(f, "    \"aggregate\": null%s\n", demoteCollected ? "," : "");
    if (demoteCollected)
    {
        const MSXRebalanceMetrics *r = &ProfileCounters.rebalance;
        fprintf(f, "    \"rebalance\": {\n"
                   "      \"parent_ms\": %.6f, \"rb_scan_ms\": %.6f, \"rb_empty_init_ms\": %.6f, \"rb_plan_ms\": %.6f,\n"
                   "      \"rb_preflush_ms\": %.6f, \"rb_fetch_ms\": %.6f, \"rb_validate_commit_ms\": %.6f, \"rb_promote_ms\": %.6f, \"rb_residual_ms\": %.6f,\n"
                   "      \"scan_passes\": %llu, \"core_visits\": %llu, \"boundary_visits\": %llu,\n"
                   "      \"rebuilt_links\": %llu, \"init_promotes\": %llu, \"demote_rows\": %llu, \"head_rows\": %llu, \"tail_rows\": %llu, \"promote_rows\": %llu,\n"
                   "      \"patch_descriptors\": %llu, \"patch_rows\": %llu, \"validation_capacity_visits\": %llu, \"commit_rows\": %llu, \"id_lookup_probes\": %llu,\n"
                   "      \"residual_negative_over_1pct\": %llu\n"
                   "    }\n",
                r->rb_parent_ms, r->rb_scan_ms, r->rb_empty_init_ms,
                r->rb_plan_ms, r->rb_preflush_ms, r->rb_fetch_ms,
                r->rb_validate_commit_ms, r->rb_promote_ms,
                r->rb_residual_ms,
                (unsigned long long)r->scan_passes,
                (unsigned long long)r->core_visits,
                (unsigned long long)r->boundary_visits,
                (unsigned long long)r->rebuilt_links,
                (unsigned long long)r->init_promotes,
                (unsigned long long)r->demote_rows,
                (unsigned long long)r->head_rows,
                (unsigned long long)r->tail_rows,
                (unsigned long long)r->promote_rows,
                (unsigned long long)r->patch_descriptors,
                (unsigned long long)r->patch_rows,
                (unsigned long long)r->validation_capacity_visits,
                (unsigned long long)r->commit_rows,
                (unsigned long long)r->id_lookup_probes,
                (unsigned long long)r->residual_negative_over_1pct);
    }
    fprintf(f, "  }\n}\n");
    fclose(f);
}

void MSXgpu_closeTiming(void)
{
    if (TimingFile)
    {
        fprintf(TimingFile,
            "TOTAL,%lld,%.6f,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,"
            "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
            "%.6f,%.6f,%.6f,%.17g,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
            "%.6f,%.6f,%.6f,%.6f,%.6f,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
            TotalTiming.step_index, TotalTiming.sim_time_sec,
            TotalTiming.gpu_enabled, TotalTiming.gpu_strict,
            TotalTiming.react_gpu, TotalTiming.advect_gpu, TotalTiming.mix_gpu,
            TotalTiming.release_gpu, TotalTiming.disperse_gpu, TotalTiming.ode_gpu,
            TotalTiming.equil_gpu, TotalTiming.formula_gpu, TotalTiming.full_coupling,
            TotalTiming.equil_time_embedded, TotalTiming.total_ms, TotalTiming.react_ms,
            TotalTiming.advect_ms, TotalTiming.mix_ms, TotalTiming.release_ms,
            TotalTiming.disperse_ms, TotalTiming.react_pack_ms, TotalTiming.h2d_ms,
            TotalTiming.react_ode_ms, TotalTiming.react_equil_ms,
            TotalTiming.react_formula_ms, TotalTiming.d2h_ms,
            TotalTiming.react_unpack_ms, TotalTiming.jit_ms,
            TotalTiming.nvrtc_compile_ms, TotalTiming.ptx_cache_read_ms,
            TotalTiming.cu_module_load_ms, TotalTiming.kernel_lookup_ms,
            TotalTiming.rk5_nfcn, TotalTiming.rk5_naccpt,
            TotalTiming.rk5_nrejct, TotalTiming.rk5_last_hstep,
            TotalTiming.rk5_bucket_count, TotalTiming.rk5_bucket_launches,
            TotalTiming.rk5_bucket_max_size, TotalTiming.rk5_bucket_reorder_ms,
            TotalTiming.react_count_parallel_ms, TotalTiming.react_count_prefix_ms,
            TotalTiming.react_pack_segment_ms, TotalTiming.react_host_alloc_ms,
            TotalTiming.react_device_alloc_ms, TotalTiming.react_scatter_ms,
            TotalTiming.ros2_nfcn, TotalTiming.ros2_njac,
            TotalTiming.ros2_naccept, TotalTiming.ros2_nreject,
            TotalTiming.ros2_last_hstep,
            TotalTiming.rk5_fast_mode,
            TotalTiming.rk5_error_code, TotalTiming.ros2_error_code,
            TotalTiming.error_code,
            TotalTiming.resident_boundary_cpu_ms, TotalTiming.resident_enumerate_filter_ms,
            TotalTiming.resident_patch_h2d_ms, TotalTiming.resident_active_h2d_ms,
            TotalTiming.resident_diag_d2h_hstep_ms, TotalTiming.resident_sync_ms,
            TotalTiming.resident_initial_upload_ms, TotalTiming.resident_handoff_ms,
            TotalTiming.resident_fallback_ms, TotalTiming.resident_active_rows);
        fclose(TimingFile);
    }
    TimingFile = NULL;
    writeProfileSummary();
#ifdef EPANETMSX_CUDA_ENABLED
    if (ReactTransferReady)
    {
        MSXreactTransfer_close();
        ReactTransferReady = 0;
    }
#endif
    GpuTimingOutputEnabled = 0;
}

void MSXgpu_beginStep(double simTimeSec)
{
    MSXGpuTiming *t = &MSX.GpuTimingRecord;
    memset(t, 0, sizeof(*t));
    Ros2RawErrorReason = 0;
    Ros2RawErrorSid = -1;
    t->step_index = StepIndex++;
    t->sim_time_sec = simTimeSec;
    t->gpu_enabled = (MSX.GpuReact || MSX.GpuCompiler);
    t->gpu_strict = MSX.GpuStrict;
    t->full_coupling = (MSX.Coupling == FULL_COUPLING);
    t->equil_time_embedded = t->full_coupling;
    StepStartMs = MSXgpu_profileStageEnabled() ? MSXgpu_wallTimeMs() : 0.0;
}

void MSXgpu_endStep(int errorCode)
{
    MSXGpuTiming *t = &MSX.GpuTimingRecord;
    if (!MSXgpu_profileStageEnabled())
    {
        t->error_code = errorCode;
        return;
    }
    t->total_ms = MSXgpu_wallTimeMs() - StepStartMs;
    t->error_code = errorCode;
    TotalTiming.step_index = StepIndex;
    TotalTiming.sim_time_sec = t->sim_time_sec;
    TotalTiming.gpu_enabled |= t->gpu_enabled;
    TotalTiming.gpu_strict |= t->gpu_strict;
    TotalTiming.react_gpu |= t->react_gpu;
    TotalTiming.advect_gpu |= t->advect_gpu;
    TotalTiming.mix_gpu |= t->mix_gpu;
    TotalTiming.release_gpu |= t->release_gpu;
    TotalTiming.disperse_gpu |= t->disperse_gpu;
    TotalTiming.ode_gpu |= t->ode_gpu;
    TotalTiming.equil_gpu |= t->equil_gpu;
    TotalTiming.formula_gpu |= t->formula_gpu;
    TotalTiming.full_coupling |= t->full_coupling;
    TotalTiming.equil_time_embedded |= t->equil_time_embedded;
    TotalTiming.total_ms += t->total_ms;
    TotalTiming.react_ms += t->react_ms;
    TotalTiming.advect_ms += t->advect_ms;
    TotalTiming.mix_ms += t->mix_ms;
    TotalTiming.release_ms += t->release_ms;
    TotalTiming.disperse_ms += t->disperse_ms;
    TotalTiming.react_pack_ms += t->react_pack_ms;
    TotalTiming.h2d_ms += t->h2d_ms;
    TotalTiming.react_ode_ms += t->react_ode_ms;
    TotalTiming.react_equil_ms += t->react_equil_ms;
    TotalTiming.react_formula_ms += t->react_formula_ms;
    TotalTiming.d2h_ms += t->d2h_ms;
    TotalTiming.react_unpack_ms += t->react_unpack_ms;
    TotalTiming.jit_ms += t->jit_ms;
    TotalTiming.nvrtc_compile_ms += t->nvrtc_compile_ms;
    TotalTiming.ptx_cache_read_ms += t->ptx_cache_read_ms;
    TotalTiming.cu_module_load_ms += t->cu_module_load_ms;
    TotalTiming.kernel_lookup_ms += t->kernel_lookup_ms;
    TotalTiming.rk5_nfcn += t->rk5_nfcn;
    TotalTiming.rk5_naccpt += t->rk5_naccpt;
    TotalTiming.rk5_nrejct += t->rk5_nrejct;
    if (t->rk5_last_hstep != 0.0) TotalTiming.rk5_last_hstep = t->rk5_last_hstep;
    TotalTiming.rk5_bucket_count += t->rk5_bucket_count;
    TotalTiming.rk5_bucket_launches += t->rk5_bucket_launches;
    if (t->rk5_bucket_max_size > TotalTiming.rk5_bucket_max_size)
        TotalTiming.rk5_bucket_max_size = t->rk5_bucket_max_size;
    TotalTiming.rk5_bucket_reorder_ms += t->rk5_bucket_reorder_ms;
    TotalTiming.react_count_parallel_ms += t->react_count_parallel_ms;
    TotalTiming.react_count_prefix_ms += t->react_count_prefix_ms;
    TotalTiming.react_pack_segment_ms += t->react_pack_segment_ms;
    TotalTiming.react_host_alloc_ms += t->react_host_alloc_ms;
    TotalTiming.react_device_alloc_ms += t->react_device_alloc_ms;
    TotalTiming.react_scatter_ms += t->react_scatter_ms;
    TotalTiming.resident_boundary_cpu_ms += t->resident_boundary_cpu_ms;
    TotalTiming.resident_enumerate_filter_ms += t->resident_enumerate_filter_ms;
    TotalTiming.resident_patch_h2d_ms += t->resident_patch_h2d_ms;
    TotalTiming.resident_active_h2d_ms += t->resident_active_h2d_ms;
    TotalTiming.resident_diag_d2h_hstep_ms += t->resident_diag_d2h_hstep_ms;
    TotalTiming.resident_sync_ms += t->resident_sync_ms;
    TotalTiming.resident_initial_upload_ms += t->resident_initial_upload_ms;
    TotalTiming.resident_handoff_ms += t->resident_handoff_ms;
    TotalTiming.resident_fallback_ms += t->resident_fallback_ms;
    TotalTiming.resident_active_rows += t->resident_active_rows;
    TotalTiming.ros2_nfcn += t->ros2_nfcn;
    TotalTiming.ros2_njac += t->ros2_njac;
    TotalTiming.ros2_naccept += t->ros2_naccept;
    TotalTiming.ros2_nreject += t->ros2_nreject;
    if (t->ros2_last_hstep != 0.0) TotalTiming.ros2_last_hstep = t->ros2_last_hstep;
    TotalTiming.rk5_fast_mode |= t->rk5_fast_mode;
    if (!TotalTiming.rk5_error_code && t->rk5_error_code) TotalTiming.rk5_error_code = t->rk5_error_code;
    if (!TotalTiming.ros2_error_code && t->ros2_error_code) TotalTiming.ros2_error_code = t->ros2_error_code;
    if (!TotalTiming.error_code && errorCode) TotalTiming.error_code = errorCode;
    if (CpuTimingFile)
    {
        CpuTotalTiming.step_index = StepIndex;
        CpuTotalTiming.sim_time_sec = t->sim_time_sec;
        CpuTotalTiming.full_coupling |= t->full_coupling;
        CpuTotalTiming.total_ms += t->total_ms;
        CpuTotalTiming.react_ms += t->react_ms;
        CpuTotalTiming.advect_ms += t->advect_ms;
        CpuTotalTiming.mix_ms += t->mix_ms;
        CpuTotalTiming.release_ms += t->release_ms;
        CpuTotalTiming.disperse_ms += t->disperse_ms;
        CpuTotalTiming.react_ode_ms += t->react_ode_ms;
        CpuTotalTiming.react_equil_ms += t->react_equil_ms;
        CpuTotalTiming.react_formula_ms += t->react_formula_ms;
        if (!CpuTotalTiming.error_code && errorCode) CpuTotalTiming.error_code = errorCode;
    }
}

void MSXgpu_reactBegin(void) { ReactActive = 1; }
void MSXgpu_reactEnd(void) { ReactActive = 0; }

static int legacyCpuChemistryTimingEnabled(void)
{
    /* Preserve the old CPU_TIMING YES chemistry clocks when no new profile
       mode overrides the legacy switches.  This matters for the combination
       GPU_TIMING YES + GPU_TIMING_DETAIL NO + CPU_TIMING YES: GPU detail is
       off, but the explicitly requested CPU timing must remain observable. */
    return !ProfileExplicit && MSX.CpuTimingConfigured && MSX.CpuTiming;
}

int MSXgpu_cpuChemistryTimingEnabled(void)
{
    MSXgpu_profileInit();
    return MSXgpu_profileDetailGroupEnabled(MSX_PROFILE_DETAIL_CHEM) ||
           legacyCpuChemistryTimingEnabled();
}

static void addTime(double *field, double ms)
{
    if (!ReactActive) return;
    if (!MSXgpu_cpuChemistryTimingEnabled()) return;
#ifdef _OPENMP
    if (omp_in_parallel()) ms /= (double)omp_get_num_threads();
#pragma omp atomic
#endif
    *field += ms;
}

void MSXgpu_addOdeTime(double ms) { addTime(&MSX.GpuTimingRecord.react_ode_ms, ms); }
void MSXgpu_addEquilTime(double ms) { addTime(&MSX.GpuTimingRecord.react_equil_ms, ms); }
void MSXgpu_addFormulaTime(double ms) { addTime(&MSX.GpuTimingRecord.react_formula_ms, ms); }

#ifndef EPANETMSX_CUDA_ENABLED
int MSXgpu_prepareResidentContext(void)
{
    setGpuError(ERR_GPU_NOT_ENABLED, GPU_STAGE_NONE, -1, -1, -1, -1, -1, 0.0);
    return ERR_GPU_NOT_ENABLED;
}

int MSXgpu_openResidentPrograms(void)
{
    setGpuError(ERR_GPU_NOT_ENABLED, GPU_STAGE_NONE, -1, -1, -1, -1, -1, 0.0);
    return ERR_GPU_NOT_ENABLED;
}

void MSXgpu_closeResidentPrograms(void)
{
    /* Non-CUDA builds have no program cache. */
}

static int mapResidentStatus(MSXResidentStatus status)
{
    switch (status)
    {
    case MSX_RESIDENT_ERR_ARGUMENT:
    case MSX_RESIDENT_ERR_CONFIG:
    case MSX_RESIDENT_ERR_SCOPE:
    case MSX_RESIDENT_ERR_SOLVER:
        return ERR_GPU_UNSUPPORTED_FEATURE;
    case MSX_RESIDENT_ERR_CAPACITY:
    case MSX_RESIDENT_ERR_GENERATION:
    case MSX_RESIDENT_ERR_OVERFLOW:
        return ERR_GPU_SEGMENT_PACK_FAILED;
    case MSX_RESIDENT_ERR_TRANSFER:
    case MSX_RESIDENT_ERR_GPU:
    case MSX_RESIDENT_ERR_POISONED:
        return ERR_GPU_KERNEL_RUNTIME_ERROR;
    default:
        return ERR_GPU_NOT_ENABLED;
    }
}
int MSXgpu_reactResidentCore(MSXResidentGpu *gpu, const MSXResidentActiveBatch *batch,
                             double dt, MSXResidentGpuReactResult *result)
{
    (void)gpu; (void)batch; (void)dt;
    if (result) memset(result, 0, sizeof(*result));
    return MSXgpu_prepareResidentContext();
}
int MSXgpu_submitResidentCore(MSXResidentGpu *gpu, const MSXResidentActiveBatch *batch,
                              double dt, MSXgpuResidentCoreToken *token)
{ (void)gpu; (void)batch; (void)dt; if(token) memset(token,0,sizeof(*token)); return ERR_GPU_NOT_ENABLED; }
int MSXgpu_submitResidentCoreHyd(MSXResidentGpu *gpu, const MSXResidentActiveBatch *batch,
                                 const MSXResidentHydView *hyd, double dt,
                                 MSXgpuResidentCoreToken *token)
{ (void)gpu; (void)batch; (void)hyd; (void)dt; if(token) memset(token,0,sizeof(*token)); return ERR_GPU_NOT_ENABLED; }
int MSXgpu_submitResidentCoreHydPrepared(MSXResidentGpu *gpu,
                                         MSXResidentGpuActiveWriter *writer,
                                         const MSXResidentHydView *hyd, double dt,
                                         MSXgpuResidentCoreToken *token)
{ (void)gpu; (void)writer; (void)hyd; (void)dt; if(token) memset(token,0,sizeof(*token)); return ERR_GPU_NOT_ENABLED; }
int MSXgpu_finishResidentCore(MSXResidentGpu *gpu, MSXgpuResidentCoreToken *token,
                              MSXResidentGpuReactResult *result)
{ (void)gpu; (void)token; if(result) memset(result,0,sizeof(*result)); return ERR_GPU_NOT_ENABLED; }
int MSXgpu_abortResidentCore(MSXResidentGpu *gpu, MSXgpuResidentCoreToken *token)
{ (void)gpu; (void)token; return ERR_GPU_NOT_ENABLED; }
int MSXgpu_reactPipeSegments(double dt)
{
    (void)dt;
    setGpuError(ERR_GPU_NOT_ENABLED, GPU_STAGE_NONE, -1, -1, -1, -1, -1, 0.0);
    return ERR_GPU_NOT_ENABLED;
}
#else

static int mapResidentStatus(MSXResidentStatus status)
{
    switch (status)
    {
    case MSX_RESIDENT_ERR_ARGUMENT: case MSX_RESIDENT_ERR_CONFIG:
    case MSX_RESIDENT_ERR_SCOPE: case MSX_RESIDENT_ERR_SOLVER:
        return ERR_GPU_UNSUPPORTED_FEATURE;
    case MSX_RESIDENT_ERR_CAPACITY: case MSX_RESIDENT_ERR_GENERATION:
    case MSX_RESIDENT_ERR_OVERFLOW:
        return ERR_GPU_SEGMENT_PACK_FAILED;
    case MSX_RESIDENT_ERR_TRANSFER: case MSX_RESIDENT_ERR_GPU:
    case MSX_RESIDENT_ERR_POISONED:
        return ERR_GPU_KERNEL_RUNTIME_ERROR;
    default: return ERR_GPU_NOT_ENABLED;
    }
}

#ifndef EPANETMSX_REACT_TRANSFER_STATIC_CUDA
int MSXreactTransfer_init(char *errmsg, int errmsgLen)
{
    if (errmsg && errmsgLen > 0)
        snprintf(errmsg, (size_t)errmsgLen, "static CUDA react transfer module is not built");
    return 1;
}
void MSXreactTransfer_close(void) {}
int MSXreactTransfer_countAndPack(double dt, int nSpecies, MSXReactTransferView *view)
{
    (void)dt; (void)nSpecies; (void)view;
    return ERR_GPU_NOT_ENABLED;
}
int MSXreactTransfer_ringView(double dt, int nSpecies, MSXReactTransferView *view)
{
    (void)dt; (void)nSpecies; (void)view;
    return ERR_GPU_NOT_ENABLED;
}
int MSXreactTransfer_upload(MSXReactTransferView *view, const void *zeroErr, size_t errSize)
{
    (void)view; (void)zeroErr; (void)errSize;
    return ERR_GPU_NOT_ENABLED;
}
int MSXreactTransfer_download(MSXReactTransferView *view, void *hostErr, size_t errSize)
{
    (void)view; (void)hostErr; (void)errSize;
    return ERR_GPU_NOT_ENABLED;
}
int MSXreactTransfer_unpack(MSXReactTransferView *view)
{
    (void)view;
    return ERR_GPU_NOT_ENABLED;
}
#endif

static int ensureReactTransfer(void)
{
    char errText[512];
    if (ReactTransferReady) return 0;
    if (MSXreactTransfer_init(errText, (int)sizeof(errText)))
    {
        setGpuError(ERR_GPU_NOT_ENABLED, GPU_STAGE_NONE, -1, -1, -1, -1, -1, 0.0);
        return ERR_GPU_NOT_ENABLED;
    }
    ReactTransferReady = 1;
    return 0;
}

static const char *GpuReactCudaSource =
"extern \"C\" {\n"
"typedef struct { int opcode; int ivar; double fvalue; } Instr;\n"
"typedef struct { int first; int count; } Prog;\n"
"typedef struct { int code; int stage; int sid; int pipe; int species; int expr; int iter; double value; } GpuErr;\n"
"__device__ void set_err(GpuErr* e,int code,int stage,int sid,int pipe,int species,int expr,int iter,double value){ if(atomicCAS(&e->code,0,code)==0){ e->stage=stage; e->sid=sid; e->pipe=pipe; e->species=species; e->expr=expr; e->iter=iter; e->value=value; }}\n"
"__device__ double d_step(double x){ return x<=0.0?0.0:1.0; }\n"
"__device__ double d_sgn(double x){ return (x>0.0)-(x<0.0); }\n"
"__device__ double d_pow(double a,double b){ return a<=0.0?0.0:pow(a,b); }\n"
"__device__ double d_safe_mul(double a,double b){ return (a==0.0||b==0.0)?0.0:a*b; }\n"
"__device__ double term_value_flat(int tid,double* c,int pipe,const double* params,int pStride,const double* constants,const double* hyd,int hBase,const Prog* termProg,const Instr* instr,int lastSpecies,int lastTerm,int lastParam,int lastConst){ const Prog p=termProg[tid]; double st[128]; int sp=0; st[0]=0.0; for(int n=0;n<p.count;n++){ Instr in=instr[p.first+n]; double a,b; switch(in.opcode){ case 3:b=st[sp--];a=st[sp];st[sp]=a+b;break; case 4:b=st[sp--];a=st[sp];st[sp]=a-b;break; case 5:b=st[sp--];a=st[sp];st[sp]=a*b;break; case 6:b=st[sp--];a=st[sp];st[sp]=a/b;break; case 7:st[++sp]=in.fvalue;break; case 8: if(in.ivar<=lastSpecies) st[++sp]=c[in.ivar]; else if(in.ivar<=lastTerm) st[++sp]=0.0; else if(in.ivar<=lastParam){ int pid=in.ivar-lastTerm; st[++sp]=params[pipe*pStride+pid]; } else if(in.ivar<=lastConst) st[++sp]=constants[in.ivar-lastParam]; else st[++sp]=hyd[hBase+in.ivar-lastConst]; break; case 9:st[sp]=-st[sp];break; case 10:st[sp]=cos(st[sp]);break; case 11:st[sp]=sin(st[sp]);break; case 12:st[sp]=tan(st[sp]);break; case 13:st[sp]=1.0/tan(st[sp]);break; case 14:st[sp]=fabs(st[sp]);break; case 15:st[sp]=d_sgn(st[sp]);break; case 16:st[sp]=sqrt(st[sp]);break; case 17:st[sp]=log(st[sp]);break; case 18:st[sp]=exp(st[sp]);break; case 19:st[sp]=asin(st[sp]);break; case 20:st[sp]=acos(st[sp]);break; case 21:st[sp]=atan(st[sp]);break; case 22:st[sp]=1.57079632679489661923-atan(st[sp]);break; case 23:st[sp]=sinh(st[sp]);break; case 24:st[sp]=cosh(st[sp]);break; case 25:st[sp]=tanh(st[sp]);break; case 26:st[sp]=1.0/tanh(st[sp]);break; case 27:st[sp]=log10(st[sp]);break; case 28:st[sp]=d_step(st[sp]);break; case 31:b=st[sp--];a=st[sp];st[sp]=d_pow(a,b);break; default: return 0.0; }} return (st[sp]==st[sp]?st[sp]:0.0); }\n"
"__device__ double var_value(int ivar,double* c,int pipe,const double* params,int pStride,const double* constants,const double* hyd,int hBase,const Prog* termProg,const Instr* instr,int lastSpecies,int lastTerm,int lastParam,int lastConst){ if(ivar<=lastSpecies) return c[ivar]; if(ivar<=lastTerm){ int tid=ivar-lastSpecies; const Prog p=termProg[tid]; double st[128]; int sp=0; st[0]=0.0; for(int n=0;n<p.count;n++){ Instr in=instr[p.first+n]; double a,b; switch(in.opcode){ case 3:b=st[sp--];a=st[sp];st[sp]=a+b;break; case 4:b=st[sp--];a=st[sp];st[sp]=a-b;break; case 5:b=st[sp--];a=st[sp];st[sp]=a*b;break; case 6:b=st[sp--];a=st[sp];st[sp]=a/b;break; case 7:st[++sp]=in.fvalue;break; case 8: if(in.ivar<=lastSpecies) st[++sp]=c[in.ivar]; else if(in.ivar<=lastTerm) st[++sp]=term_value_flat(in.ivar-lastSpecies,c,pipe,params,pStride,constants,hyd,hBase,termProg,instr,lastSpecies,lastTerm,lastParam,lastConst); else if(in.ivar<=lastParam){ int pid=in.ivar-lastTerm; st[++sp]=params[pipe*pStride+pid]; } else if(in.ivar<=lastConst) st[++sp]=constants[in.ivar-lastParam]; else st[++sp]=hyd[hBase+in.ivar-lastConst]; break; case 9:st[sp]=-st[sp];break; case 10:st[sp]=cos(st[sp]);break; case 11:st[sp]=sin(st[sp]);break; case 12:st[sp]=tan(st[sp]);break; case 13:st[sp]=1.0/tan(st[sp]);break; case 14:st[sp]=fabs(st[sp]);break; case 15:st[sp]=d_sgn(st[sp]);break; case 16:st[sp]=sqrt(st[sp]);break; case 17:st[sp]=log(st[sp]);break; case 18:st[sp]=exp(st[sp]);break; case 19:st[sp]=asin(st[sp]);break; case 20:st[sp]=acos(st[sp]);break; case 21:st[sp]=atan(st[sp]);break; case 22:st[sp]=1.57079632679489661923-atan(st[sp]);break; case 23:st[sp]=sinh(st[sp]);break; case 24:st[sp]=cosh(st[sp]);break; case 25:st[sp]=tanh(st[sp]);break; case 26:st[sp]=1.0/tanh(st[sp]);break; case 27:st[sp]=log10(st[sp]);break; case 28:st[sp]=d_step(st[sp]);break; case 31:b=st[sp--];a=st[sp];st[sp]=d_pow(a,b);break; default: return 0.0; }} return (st[sp]==st[sp]?st[sp]:0.0); } if(ivar<=lastParam){ int pid=ivar-lastTerm; return params[pipe*pStride+pid]; } if(ivar<=lastConst) return constants[ivar-lastParam]; return hyd[hBase+ivar-lastConst]; }\n"
"__device__ double eval_prog(int progId,const Prog* prog,const Instr* instr,double* c,int pipe,const double* params,int pStride,const double* constants,const double* hyd,int hBase,const Prog* termProg,int lastSpecies,int lastTerm,int lastParam,int lastConst){ Prog p=prog[progId]; double st[128]; int sp=0; st[0]=0.0; for(int n=0;n<p.count;n++){ Instr in=instr[p.first+n]; double a,b; switch(in.opcode){ case 3:b=st[sp--];a=st[sp];st[sp]=a+b;break; case 4:b=st[sp--];a=st[sp];st[sp]=a-b;break; case 5:b=st[sp--];a=st[sp];st[sp]=a*b;break; case 6:b=st[sp--];a=st[sp];st[sp]=a/b;break; case 7:st[++sp]=in.fvalue;break; case 8: if(in.ivar<=lastParam) st[++sp]=var_value(in.ivar,c,pipe,params,pStride,constants,hyd,hBase,termProg,instr,lastSpecies,lastTerm,lastParam,lastConst); else if(in.ivar<=lastConst) st[++sp]=constants[in.ivar-lastParam]; else st[++sp]=hyd[hBase+in.ivar-lastConst]; break; case 9:st[sp]=-st[sp];break; case 10:st[sp]=cos(st[sp]);break; case 11:st[sp]=sin(st[sp]);break; case 12:st[sp]=tan(st[sp]);break; case 13:st[sp]=1.0/tan(st[sp]);break; case 14:st[sp]=fabs(st[sp]);break; case 15:st[sp]=d_sgn(st[sp]);break; case 16:st[sp]=sqrt(st[sp]);break; case 17:st[sp]=log(st[sp]);break; case 18:st[sp]=exp(st[sp]);break; case 19:st[sp]=asin(st[sp]);break; case 20:st[sp]=acos(st[sp]);break; case 21:st[sp]=atan(st[sp]);break; case 22:st[sp]=1.57079632679489661923-atan(st[sp]);break; case 23:st[sp]=sinh(st[sp]);break; case 24:st[sp]=cosh(st[sp]);break; case 25:st[sp]=tanh(st[sp]);break; case 26:st[sp]=1.0/tanh(st[sp]);break; case 27:st[sp]=log10(st[sp]);break; case 28:st[sp]=d_step(st[sp]);break; case 31:b=st[sp--];a=st[sp];st[sp]=d_pow(a,b);break; default:return 0.0; }} return (st[sp]==st[sp]?st[sp]:0.0); }\n"
"__device__ double eval_prog_deriv(int progId,int wrt,const Prog* prog,const Instr* instr,double* c,int pipe,const double* params,int pStride,const double* constants,const double* hyd,int hBase,const Prog* termProg,int lastSpecies,int lastTerm,int lastParam,int lastConst){ Prog p=prog[progId]; double v[128]; double d[128]; int sp=0; v[0]=0.0; d[0]=0.0; for(int n=0;n<p.count;n++){ Instr in=instr[p.first+n]; double a,b,da,db,t; switch(in.opcode){ case 3:b=v[sp];db=d[sp--];a=v[sp];da=d[sp];v[sp]=a+b;d[sp]=da+db;break; case 4:b=v[sp];db=d[sp--];a=v[sp];da=d[sp];v[sp]=a-b;d[sp]=da-db;break; case 5:b=v[sp];db=d[sp--];a=v[sp];da=d[sp];v[sp]=a*b;d[sp]=d_safe_mul(da,b)+d_safe_mul(a,db);break; case 6:b=v[sp];db=d[sp--];a=v[sp];da=d[sp];v[sp]=a/b;d[sp]=(d_safe_mul(da,b)-d_safe_mul(a,db))/(b*b);break; case 7:sp++;v[sp]=in.fvalue;d[sp]=0.0;break; case 8:sp++;v[sp]=var_value(in.ivar,c,pipe,params,pStride,constants,hyd,hBase,termProg,instr,lastSpecies,lastTerm,lastParam,lastConst);d[sp]=(in.ivar==wrt && in.ivar<=lastSpecies)?1.0:0.0;break; case 9:v[sp]=-v[sp];d[sp]=-d[sp];break; case 10:d[sp]=d_safe_mul(-sin(v[sp]),d[sp]);v[sp]=cos(v[sp]);break; case 11:d[sp]=d_safe_mul(cos(v[sp]),d[sp]);v[sp]=sin(v[sp]);break; case 12:t=tan(v[sp]);d[sp]=d_safe_mul(1.0+t*t,d[sp]);v[sp]=t;break; case 13:t=tan(v[sp]);d[sp]=d_safe_mul(-(1.0+t*t)/(t*t),d[sp]);v[sp]=1.0/t;break; case 14:d[sp]=d_safe_mul(d_sgn(v[sp]),d[sp]);v[sp]=fabs(v[sp]);break; case 15:v[sp]=d_sgn(v[sp]);d[sp]=0.0;break; case 16:d[sp]=(v[sp]>0.0?d_safe_mul(0.5/sqrt(v[sp]),d[sp]):0.0);v[sp]=sqrt(v[sp]);break; case 17:d[sp]=d[sp]/v[sp];v[sp]=log(v[sp]);break; case 18:t=exp(v[sp]);d[sp]=d_safe_mul(t,d[sp]);v[sp]=t;break; case 19:d[sp]=d[sp]/sqrt(1.0-v[sp]*v[sp]);v[sp]=asin(v[sp]);break; case 20:d[sp]=-d[sp]/sqrt(1.0-v[sp]*v[sp]);v[sp]=acos(v[sp]);break; case 21:d[sp]=d[sp]/(1.0+v[sp]*v[sp]);v[sp]=atan(v[sp]);break; case 22:d[sp]=-d[sp]/(1.0+v[sp]*v[sp]);v[sp]=1.57079632679489661923-atan(v[sp]);break; case 23:d[sp]=d_safe_mul(cosh(v[sp]),d[sp]);v[sp]=sinh(v[sp]);break; case 24:d[sp]=d_safe_mul(sinh(v[sp]),d[sp]);v[sp]=cosh(v[sp]);break; case 25:t=tanh(v[sp]);d[sp]=d_safe_mul(1.0-t*t,d[sp]);v[sp]=t;break; case 26:t=tanh(v[sp]);d[sp]=d_safe_mul(-(1.0-t*t)/(t*t),d[sp]);v[sp]=1.0/t;break; case 27:d[sp]=d[sp]/(v[sp]*2.30258509299404568402);v[sp]=log10(v[sp]);break; case 28:v[sp]=d_step(v[sp]);d[sp]=0.0;break; case 31:b=v[sp];db=d[sp--];a=v[sp];da=d[sp]; if(a>0.0){ t=pow(a,b); d[sp]=d_safe_mul(t,(d_safe_mul(db,log(a))+d_safe_mul(b,da/a))); v[sp]=t; } else { v[sp]=0.0; d[sp]=0.0; } break; default:return 0.0; }} return (d[sp]==d[sp]?d[sp]:0.0); }\n"
"#if defined(MSX_GPU_SOLVER_RK5)\n"
"__device__ void eval_rates(int nRate,const int* rateSpecies,double* state,int pipe,const double* params,int pStride,const double* constants,const double* hyd,int hBase,const Prog* prog,const Prog* termProg,const Instr* instr,int lastSpecies,int lastTerm,int lastParam,int lastConst,double* out){ for(int i=1;i<=nRate;i++){ int m=rateSpecies[i]; double v=eval_prog(m,prog,instr,state,pipe,params,pStride,constants,hyd,hBase,termProg,lastSpecies,lastTerm,lastParam,lastConst); out[i]=(v==v?v:0.0); }}\n"
"#endif\n"
"#if defined(MSX_GPU_SOLVER_EUL)\n"
"__global__ void ode_kernel(int nSeg,int nSpecies,int nRate,double tstep,const int* segPipe,const int* segRow,const double* segVol,const int* rateSpecies,const int* speciesType,const double* linkDiam,double areaUcf,double lperFt3,double* c,double* cOde,double* reacted,const double* params,int pStride,const double* constants,const double* hyd,int hStride,const Prog* prog,const Prog* termProg,const Instr* instr,int lastSpecies,int lastTerm,int lastParam,int lastConst,GpuErr* err){ int sid=blockIdx.x*blockDim.x+threadIdx.x; if(sid>=nSeg) return; int pipe=segPipe[sid]; double vol=segVol[sid]; int base=segRow[sid]*(nSpecies+1); int hBase=sid*hStride; double old[65]; double deriv[65]; for(int m=1;m<=nSpecies;m++) old[m]=c[base+m]; if(tstep>0.0){ for(int i=1;i<=nRate;i++){ int m=rateSpecies[i]; deriv[i]=eval_prog(m,prog,instr,&c[base],pipe,params,pStride,constants,hyd,hBase,termProg,lastSpecies,lastTerm,lastParam,lastConst); if(!(deriv[i]==deriv[i])) deriv[i]=0.0; } for(int i=1;i<=nRate;i++){ int m=rateSpecies[i]; double v=c[base+m]+deriv[i]*tstep; c[base+m]=(v>=0.0?v:0.0); }} for(int m=1;m<=nSpecies;m++){ cOde[base+m]=c[base+m]; double delta=c[base+m]-old[m]; double add = speciesType[m]==0 ? vol*delta*lperFt3 : (linkDiam[pipe]>0.0 ? vol*4.0/linkDiam[pipe]*areaUcf*delta : 0.0); atomicAdd(&reacted[pipe*(nSpecies+1)+m], add); if(!(c[base+m]==c[base+m])) c[base+m]=0.0; }}\n"
"#endif\n"
"#if defined(MSX_GPU_SOLVER_RK5)\n"
"#if !defined(MSX_GPU_RK5_FAST_BUCKET)\n"
"__global__ void rk5_kernel(int nSeg,int nSpecies,int nRate,double tstep,const int* segPipe,const int* segRow,const double* segVol,double* hstep,const int* rateSpecies,const int* speciesType,const double* linkDiam,double areaUcf,double lperFt3,double* c,double* cOde,double* reacted,const double* params,int pStride,const double* constants,const double* hyd,int hStride,const Prog* prog,const Prog* termProg,const Instr* instr,int lastSpecies,int lastTerm,int lastParam,int lastConst,GpuErr* err){ int sid=blockIdx.x*blockDim.x+threadIdx.x; if(sid>=nSeg) return; int pipe=segPipe[sid]; double vol=segVol[sid]; int base=segRow[sid]*(nSpecies+1); int hBase=sid*hStride; double old[65],state[65],tmp[65],k1[65],k2[65],k3[65],k4[65],k5[65],k6[65],k7[65]; const double a21=0.20,a31=3.0/40.0,a32=9.0/40.0,a41=44.0/45.0,a42=-56.0/15.0,a43=32.0/9.0,a51=19372.0/6561.0,a52=-25360.0/2187.0,a53=64448.0/6561.0,a54=-212.0/729.0,a61=9017.0/3168.0,a62=-355.0/33.0,a63=46732.0/5247.0,a64=49.0/176.0,a65=-5103.0/18656.0,a71=35.0/384.0,a73=500.0/1113.0,a74=125.0/192.0,a75=-2187.0/6784.0,a76=11.0/84.0; for(int m=1;m<=nSpecies;m++){ old[m]=c[base+m]; state[m]=c[base+m]; } if(tstep>0.0&&nRate>0){ double h=hstep[sid]; if(!(h>0.0)||h>tstep) h=tstep; int nSub=(int)ceil(tstep/h); if(nSub<1) nSub=1; if(nSub>1000){ set_err(err,9012,1,sid,pipe,rateSpecies[1],rateSpecies[1],0,(double)nSub); return; } h=tstep/(double)nSub; for(int step=0;step<nSub;step++){ eval_rates(nRate,rateSpecies,state,pipe,params,pStride,constants,hyd,hBase,prog,termProg,instr,lastSpecies,lastTerm,lastParam,lastConst,k1); for(int m=1;m<=nSpecies;m++) tmp[m]=state[m]; for(int i=1;i<=nRate;i++){ int m=rateSpecies[i]; tmp[m]=state[m]+h*a21*k1[i]; } eval_rates(nRate,rateSpecies,tmp,pipe,params,pStride,constants,hyd,hBase,prog,termProg,instr,lastSpecies,lastTerm,lastParam,lastConst,k2); for(int m=1;m<=nSpecies;m++) tmp[m]=state[m]; for(int i=1;i<=nRate;i++){ int m=rateSpecies[i]; tmp[m]=state[m]+h*(a31*k1[i]+a32*k2[i]); } eval_rates(nRate,rateSpecies,tmp,pipe,params,pStride,constants,hyd,hBase,prog,termProg,instr,lastSpecies,lastTerm,lastParam,lastConst,k3); for(int m=1;m<=nSpecies;m++) tmp[m]=state[m]; for(int i=1;i<=nRate;i++){ int m=rateSpecies[i]; tmp[m]=state[m]+h*(a41*k1[i]+a42*k2[i]+a43*k3[i]); } eval_rates(nRate,rateSpecies,tmp,pipe,params,pStride,constants,hyd,hBase,prog,termProg,instr,lastSpecies,lastTerm,lastParam,lastConst,k4); for(int m=1;m<=nSpecies;m++) tmp[m]=state[m]; for(int i=1;i<=nRate;i++){ int m=rateSpecies[i]; tmp[m]=state[m]+h*(a51*k1[i]+a52*k2[i]+a53*k3[i]+a54*k4[i]); } eval_rates(nRate,rateSpecies,tmp,pipe,params,pStride,constants,hyd,hBase,prog,termProg,instr,lastSpecies,lastTerm,lastParam,lastConst,k5); for(int m=1;m<=nSpecies;m++) tmp[m]=state[m]; for(int i=1;i<=nRate;i++){ int m=rateSpecies[i]; tmp[m]=state[m]+h*(a61*k1[i]+a62*k2[i]+a63*k3[i]+a64*k4[i]+a65*k5[i]); } eval_rates(nRate,rateSpecies,tmp,pipe,params,pStride,constants,hyd,hBase,prog,termProg,instr,lastSpecies,lastTerm,lastParam,lastConst,k6); for(int m=1;m<=nSpecies;m++) tmp[m]=state[m]; for(int i=1;i<=nRate;i++){ int m=rateSpecies[i]; tmp[m]=state[m]+h*(a71*k1[i]+a73*k3[i]+a74*k4[i]+a75*k5[i]+a76*k6[i]); } eval_rates(nRate,rateSpecies,tmp,pipe,params,pStride,constants,hyd,hBase,prog,termProg,instr,lastSpecies,lastTerm,lastParam,lastConst,k7); for(int i=1;i<=nRate;i++){ int m=rateSpecies[i]; double v=tmp[m]; state[m]=(v>=0.0?v:0.0); } } hstep[sid]=h; for(int m=1;m<=nSpecies;m++) c[base+m]=state[m]; } for(int m=1;m<=nSpecies;m++){ cOde[base+m]=c[base+m]; double delta=c[base+m]-old[m]; double add=speciesType[m]==0?vol*delta*lperFt3:(linkDiam[pipe]>0.0?vol*4.0/linkDiam[pipe]*areaUcf*delta:0.0); atomicAdd(&reacted[pipe*(nSpecies+1)+m],add); if(!(c[base+m]==c[base+m])) c[base+m]=0.0; }}\n"
"#endif\n"
"__device__ void rk5_align_one(int sid,int nSeg,int nSpecies,int nRate,double tstep,const int* segPipe,const int* segRow,const double* segVol,double* hstep,const double* rateAtol,const double* rateRtol,const int* rateSpecies,const int* speciesType,const double* linkDiam,double areaUcf,double lperFt3,double* c,double* cOde,double* reacted,const double* params,int pStride,const double* constants,const double* hyd,int hStride,const Prog* prog,const Prog* termProg,const Instr* instr,int lastSpecies,int lastTerm,int lastParam,int lastConst,int* rk5Nfcn,int* rk5Naccpt,int* rk5Nrejct,double* rk5LastHstep,int* rk5Err,GpuErr* err){ int pipe=segPipe[sid]; double vol=segVol[sid]; int base=segRow[sid]*(nSpecies+1); int hBase=sid*hStride; double old[65],state[65],ynew[65],tmp[65],k1[65],k2[65],k3[65],k4[65],k5[65],k6[65],k7[65]; const double c2=0.20,c3=0.30,c4c=0.80,c5c=8.0/9.0; const double a21=0.20,a31=3.0/40.0,a32=9.0/40.0,a41=44.0/45.0,a42=-56.0/15.0,a43=32.0/9.0,a51=19372.0/6561.0,a52=-25360.0/2187.0,a53=64448.0/6561.0,a54=-212.0/729.0,a61=9017.0/3168.0,a62=-355.0/33.0,a63=46732.0/5247.0,a64=49.0/176.0,a65=-5103.0/18656.0,a71=35.0/384.0,a73=500.0/1113.0,a74=125.0/192.0,a75=-2187.0/6784.0,a76=11.0/84.0; const double e1=71.0/57600.0,e3=-71.0/16695.0,e4=71.0/1920.0,e5=-17253.0/339200.0,e6=22.0/525.0,e7=-1.0/40.0; const double UROUND=2.3e-16,SAFE=0.90,fac1=0.2,fac2=10.0,beta=0.04,expo1=0.2-0.04*0.75,facc1=1.0/fac1,facc2=1.0/fac2; int nfcn=0,naccpt=0,nrejct=0,reject=0,nstep=1; for(int m=1;m<=nSpecies;m++){ old[m]=c[base+m]; state[m]=c[base+m]; } if(tstep>0.0&&nRate>0){ double t=0.0,h=hstep[sid],hmax=tstep,facold=1.0e-4; eval_rates(nRate,rateSpecies,state,pipe,params,pStride,constants,hyd,hBase,prog,termProg,instr,lastSpecies,lastTerm,lastParam,lastConst,k1); nfcn++; if(h==0.0){ h=tstep; for(int i=1;i<=nRate;i++){ int m=rateSpecies[i]; double ytol=rateAtol[i]+rateRtol[i]*fabs(state[m]); if(k1[i]!=0.0) h=fmin(h,ytol/fabs(k1[i])); }} h=fmax(1.0e-8,h); while(t<tstep){ double tnew,hnew,errest=0.0,fac11=1.0,fac; if(0.10*fabs(h)<=fabs(t)*UROUND){ rk5Err[sid]=9009; set_err(err,9009,1,sid,pipe,rateSpecies[1],rateSpecies[1],nstep,h); return; } if((t+1.01*h-tstep)>0.0) h=tstep-t; tnew=t+c2*h; for(int m=1;m<=nSpecies;m++) tmp[m]=state[m]; for(int i=1;i<=nRate;i++){ int m=rateSpecies[i]; tmp[m]=state[m]+h*a21*k1[i]; } eval_rates(nRate,rateSpecies,tmp,pipe,params,pStride,constants,hyd,hBase,prog,termProg,instr,lastSpecies,lastTerm,lastParam,lastConst,k2); tnew=t+c3*h; for(int m=1;m<=nSpecies;m++) tmp[m]=state[m]; for(int i=1;i<=nRate;i++){ int m=rateSpecies[i]; tmp[m]=state[m]+h*(a31*k1[i]+a32*k2[i]); } eval_rates(nRate,rateSpecies,tmp,pipe,params,pStride,constants,hyd,hBase,prog,termProg,instr,lastSpecies,lastTerm,lastParam,lastConst,k3); tnew=t+c4c*h; for(int m=1;m<=nSpecies;m++) tmp[m]=state[m]; for(int i=1;i<=nRate;i++){ int m=rateSpecies[i]; tmp[m]=state[m]+h*(a41*k1[i]+a42*k2[i]+a43*k3[i]); } eval_rates(nRate,rateSpecies,tmp,pipe,params,pStride,constants,hyd,hBase,prog,termProg,instr,lastSpecies,lastTerm,lastParam,lastConst,k4); tnew=t+c5c*h; for(int m=1;m<=nSpecies;m++) tmp[m]=state[m]; for(int i=1;i<=nRate;i++){ int m=rateSpecies[i]; tmp[m]=state[m]+h*(a51*k1[i]+a52*k2[i]+a53*k3[i]+a54*k4[i]); } eval_rates(nRate,rateSpecies,tmp,pipe,params,pStride,constants,hyd,hBase,prog,termProg,instr,lastSpecies,lastTerm,lastParam,lastConst,k5); tnew=t+h; for(int m=1;m<=nSpecies;m++) tmp[m]=state[m]; for(int i=1;i<=nRate;i++){ int m=rateSpecies[i]; tmp[m]=state[m]+h*(a61*k1[i]+a62*k2[i]+a63*k3[i]+a64*k4[i]+a65*k5[i]); } eval_rates(nRate,rateSpecies,tmp,pipe,params,pStride,constants,hyd,hBase,prog,termProg,instr,lastSpecies,lastTerm,lastParam,lastConst,k6); for(int m=1;m<=nSpecies;m++) ynew[m]=state[m]; for(int i=1;i<=nRate;i++){ int m=rateSpecies[i]; ynew[m]=state[m]+h*(a71*k1[i]+a73*k3[i]+a74*k4[i]+a75*k5[i]+a76*k6[i]); tmp[m]=ynew[m]; } eval_rates(nRate,rateSpecies,tmp,pipe,params,pStride,constants,hyd,hBase,prog,termProg,instr,lastSpecies,lastTerm,lastParam,lastConst,k7); nfcn+=6; hnew=h; for(int i=1;i<=nRate;i++){ int m=rateSpecies[i]; double ee=(e1*k1[i]+e3*k3[i]+e4*k4[i]+e5*k5[i]+e6*k6[i]+e7*k7[i])*h; double sk=rateAtol[i]+rateRtol[i]*fmax(fabs(state[m]),fabs(ynew[m])); if(sk!=0.0){ double q=ee/sk; errest+=q*q; }} errest=sqrt(errest/(double)nRate); fac11=pow(errest,expo1); fac=fac11/pow(facold,beta); fac=fmax(facc2,fmin(facc1,fac/SAFE)); hnew=h/fac; if(errest<=1.0){ facold=fmax(errest,1.0e-4); naccpt++; for(int i=1;i<=nRate;i++){ int m=rateSpecies[i]; k1[i]=k7[i]; state[m]=ynew[m]; } t=t+h; hstep[sid]=h; if(fabs(hnew)>hmax) hnew=hmax; if(reject) hnew=fmin(fabs(hnew),fabs(h)); reject=0; } else { hnew=h/fmin(facc1,fac11/SAFE); reject=1; if(naccpt>=1) nrejct++; } h=hnew; hstep[sid]=h; nstep++; if(nstep>=1000){ rk5Err[sid]=9009; set_err(err,9009,1,sid,pipe,rateSpecies[1],rateSpecies[1],nstep,h); return; }} for(int i=1;i<=nRate;i++){ int m=rateSpecies[i]; double v=state[m]; c[base+m]=(v>=0.0?v:0.0); }} for(int m=1;m<=nSpecies;m++){ cOde[base+m]=c[base+m]; double delta=c[base+m]-old[m]; double add=speciesType[m]==0?vol*delta*lperFt3:(linkDiam[pipe]>0.0?vol*4.0/linkDiam[pipe]*areaUcf*delta:0.0); atomicAdd(&reacted[pipe*(nSpecies+1)+m],add); if(!(c[base+m]==c[base+m])) c[base+m]=0.0; } rk5Nfcn[sid]=nfcn; rk5Naccpt[sid]=naccpt; rk5Nrejct[sid]=nrejct; rk5LastHstep[sid]=hstep[sid]; rk5Err[sid]=0; }\n"
"__global__ void rk5_align_kernel(int nSeg,int nSpecies,int nRate,double tstep,const int* segPipe,const int* segRow,const double* segVol,double* hstep,const double* rateAtol,const double* rateRtol,const int* rateSpecies,const int* speciesType,const double* linkDiam,double areaUcf,double lperFt3,double* c,double* cOde,double* reacted,const double* params,int pStride,const double* constants,const double* hyd,int hStride,const Prog* prog,const Prog* termProg,const Instr* instr,int lastSpecies,int lastTerm,int lastParam,int lastConst,int* rk5Nfcn,int* rk5Naccpt,int* rk5Nrejct,double* rk5LastHstep,int* rk5Err,GpuErr* err){ int sid=blockIdx.x*blockDim.x+threadIdx.x; if(sid>=nSeg) return; rk5_align_one(sid,nSeg,nSpecies,nRate,tstep,segPipe,segRow,segVol,hstep,rateAtol,rateRtol,rateSpecies,speciesType,linkDiam,areaUcf,lperFt3,c,cOde,reacted,params,pStride,constants,hyd,hStride,prog,termProg,instr,lastSpecies,lastTerm,lastParam,lastConst,rk5Nfcn,rk5Naccpt,rk5Nrejct,rk5LastHstep,rk5Err,err);}\n"
"__global__ void rk5_align_pipe_warp_kernel(int nActiveLinks,const int* activeLink,const int* pipeSegOffset,const int* pipeSegCount,int nSeg,int nSpecies,int nRate,double tstep,const int* segPipe,const int* segRow,const double* segVol,double* hstep,const double* rateAtol,const double* rateRtol,const int* rateSpecies,const int* speciesType,const double* linkDiam,double areaUcf,double lperFt3,double* c,double* cOde,double* reacted,const double* params,int pStride,const double* constants,const double* hyd,int hStride,const Prog* prog,const Prog* termProg,const Instr* instr,int lastSpecies,int lastTerm,int lastParam,int lastConst,int* rk5Nfcn,int* rk5Naccpt,int* rk5Nrejct,double* rk5LastHstep,int* rk5Err,GpuErr* err){ int globalThread=blockIdx.x*blockDim.x+threadIdx.x; int warpIndex=globalThread>>5; int lane=threadIdx.x&31; if(warpIndex>=nActiveLinks) return; int start=pipeSegOffset[warpIndex]; int end=start+pipeSegCount[warpIndex]; (void)activeLink[warpIndex]; for(int sid=start+lane;sid<end;sid+=32){ rk5_align_one(sid,nSeg,nSpecies,nRate,tstep,segPipe,segRow,segVol,hstep,rateAtol,rateRtol,rateSpecies,speciesType,linkDiam,areaUcf,lperFt3,c,cOde,reacted,params,pStride,constants,hyd,hStride,prog,termProg,instr,lastSpecies,lastTerm,lastParam,lastConst,rk5Nfcn,rk5Naccpt,rk5Nrejct,rk5LastHstep,rk5Err,err); }}\n"
"#endif\n"
"__device__ void eval_equil(double* x,int nEq,const int* eqSpecies,const Prog* prog,const Prog* termProg,const Instr* instr,double* c,int pipe,const double* params,int pStride,const double* constants,const double* hyd,int hBase,int lastSpecies,int lastTerm,int lastParam,int lastConst,double* f){ for(int i=1;i<=nEq;i++) c[eqSpecies[i]]=x[i]; for(int i=1;i<=nEq;i++){ int m=eqSpecies[i]; f[i]=eval_prog(m,prog,instr,c,pipe,params,pStride,constants,hyd,hBase,termProg,lastSpecies,lastTerm,lastParam,lastConst); }}\n"
"__device__ int lu_factor(double a[17][17],int n,double* w,int* indx){ for(int i=1;i<=n;i++){ double big=0.0; for(int j=1;j<=n;j++){ double temp=fabs(a[i][j]); if(temp>big) big=temp; } if(big==0.0 || !(big==big)) return -i; w[i]=1.0/big; } for(int j=1;j<=n;j++){ for(int i=1;i<j;i++){ double sum=a[i][j]; for(int k=1;k<i;k++) sum-=a[i][k]*a[k][j]; a[i][j]=sum; } double big=0.0; int imax=j; for(int i=j;i<=n;i++){ double sum=a[i][j]; for(int k=1;k<j;k++) sum-=a[i][k]*a[k][j]; a[i][j]=sum; double dum=w[i]*fabs(sum); if(dum>=big){ big=dum; imax=i; }} if(j!=imax){ for(int k=1;k<=n;k++){ double dum=a[imax][k]; a[imax][k]=a[j][k]; a[j][k]=dum; } w[imax]=w[j]; } indx[j]=imax; if(a[j][j]==0.0) a[j][j]=1.0e-20; if(j!=n){ double dum=1.0/a[j][j]; for(int i=j+1;i<=n;i++) a[i][j]*=dum; }} return 1; }\n"
"__device__ void lu_solve(double a[17][17],int n,int* indx,double* b){ int ii=0; for(int i=1;i<=n;i++){ int ip=indx[i]; double sum=b[ip]; b[ip]=b[i]; if(ii) for(int j=ii;j<=i-1;j++) sum-=a[i][j]*b[j]; else if(sum) ii=i; b[i]=sum; } for(int i=n;i>=1;i--){ double sum=b[i]; for(int j=i+1;j<=n;j++) sum-=a[i][j]*b[j]; b[i]=sum/a[i][i]; }}\n"
"__global__ void equil_kernel(int nSeg,int nSpecies,int nEq,const int* segPipe,const int* segRow,const int* eqSpecies,double* c,const double* params,int pStride,const double* constants,const double* hyd,int hStride,const Prog* prog,const Prog* termProg,const Instr* instr,int lastSpecies,int lastTerm,int lastParam,int lastConst,GpuErr* err){ int sid=blockIdx.x*blockDim.x+threadIdx.x; if(sid>=nSeg||nEq==0) return; int pipe=segPipe[sid]; int base=segRow[sid]*(nSpecies+1); int hBase=sid*hStride; double x[17],f[17],w[17],jmat[17][17]; int indx[17]; double rel=1.0e-3; double lastErr=0.0; int lastIdx=1; for(int i=1;i<=nEq;i++) x[i]=c[base+eqSpecies[i]]; for(int iter=1;iter<=20;iter++){ eval_equil(x,nEq,eqSpecies,prog,termProg,instr,&c[base],pipe,params,pStride,constants,hyd,hBase,lastSpecies,lastTerm,lastParam,lastConst,f); for(int row=1;row<=nEq;row++){ int m=eqSpecies[row]; for(int col=1;col<=nEq;col++){ int wrt=eqSpecies[col]; jmat[row][col]=eval_prog_deriv(m,wrt,prog,instr,&c[base],pipe,params,pStride,constants,hyd,hBase,termProg,lastSpecies,lastTerm,lastParam,lastConst); }} int lures=lu_factor(jmat,nEq,w,indx); if(lures<0){ int bad=-lures; set_err(err,9010,2,sid,pipe,eqSpecies[bad],eqSpecies[bad],iter,(double)bad); return; } if(!lures){ set_err(err,9010,2,sid,pipe,eqSpecies[1],eqSpecies[1],iter,0.0); return; } for(int i=1;i<=nEq;i++) f[i]=-f[i]; lu_solve(jmat,nEq,indx,f); double errmax=0.0; for(int i=1;i<=nEq;i++){ double cscal=x[i]; if(cscal<rel) cscal=rel; x[i]+=f[i]; double ex=fabs(f[i]/cscal); if(ex>errmax){ errmax=ex; lastIdx=i; } } lastErr=errmax; if(errmax<=rel){ for(int i=1;i<=nEq;i++) c[base+eqSpecies[i]]=x[i]; return; }} set_err(err,9010,2,sid,pipe,eqSpecies[lastIdx],eqSpecies[lastIdx],20,lastErr); }\n"
"__global__ void formula_kernel(int nSeg,int nSpecies,int nFormula,const int* segPipe,const int* segRow,const int* formulaSpecies,double* c,const double* params,int pStride,const double* constants,const double* hyd,int hStride,const Prog* prog,const Prog* termProg,const Instr* instr,int lastSpecies,int lastTerm,int lastParam,int lastConst,GpuErr* err){ int sid=blockIdx.x*blockDim.x+threadIdx.x; if(sid>=nSeg) return; int pipe=segPipe[sid]; int base=segRow[sid]*(nSpecies+1); int hBase=sid*hStride; for(int i=1;i<=nFormula;i++){ int m=formulaSpecies[i]; double v=eval_prog(m,prog,instr,&c[base],pipe,params,pStride,constants,hyd,hBase,termProg,lastSpecies,lastTerm,lastParam,lastConst); if(!(v==v)) v=0.0; c[base+m]=v; }}\n"
"}\n";

/* ROS2 is kept in a separate source fragment so its kernel and diagnostics
   remain independent from the existing EUL/RK5 source branches. */
static const char *GpuRos2CudaSource =
"extern \"C\" {\n"
"__device__ void ros2_eval_rates(int n,const int* rs,double* y,int pipe,const double* p,int ps,const double* co,const double* hyd,int hb,const Prog* pr,const Prog* tr,const Instr* in,int ls,int lt,int lp,int lc,double* out){for(int i=1;i<=n;i++){int m=rs[i];double v=eval_prog(m,pr,in,y,pipe,p,ps,co,hyd,hb,tr,ls,lt,lp,lc);out[i]=(v==v?v:0.0);}}\n"
"__device__ int ros2_lu_factor(double a[17][17],int n,double* w,int* ix){for(int i=1;i<=n;i++){double big=0.0;for(int j=1;j<=n;j++){double z=fabs(a[i][j]);if(z>big)big=z;}if(big==0.0||!(big==big))return 0;w[i]=1.0/big;}for(int j=1;j<=n;j++){for(int i=1;i<j;i++){double s=a[i][j];for(int k=1;k<i;k++)s-=a[i][k]*a[k][j];a[i][j]=s;}double big=0.0;int im=j;for(int i=j;i<=n;i++){double s=a[i][j];for(int k=1;k<j;k++)s-=a[i][k]*a[k][j];a[i][j]=s;double z=w[i]*fabs(s);if(z>=big){big=z;im=i;}}if(j!=im){for(int k=1;k<=n;k++){double z=a[im][k];a[im][k]=a[j][k];a[j][k]=z;}w[im]=w[j];}ix[j]=im;if(a[j][j]==0.0)a[j][j]=1.0e-20;if(j!=n){double z=1.0/a[j][j];for(int i=j+1;i<=n;i++)a[i][j]*=z;}}return 1;}\n"
"__device__ void ros2_lu_solve(double a[17][17],int n,int* ix,double* b){int ii=0;for(int i=1;i<=n;i++){int ip=ix[i];double s=b[ip];b[ip]=b[i];if(ii)for(int j=ii;j<=i-1;j++)s-=a[i][j]*b[j];else if(s)ii=i;b[i]=s;}for(int i=n;i>=1;i--){double s=b[i];for(int j=i+1;j<=n;j++)s-=a[i][j]*b[j];b[i]=s/a[i][i];}}\n"
"__global__ void ros2_kernel(int ns,int nsp,int nr,double step,const int* pipe,const int* row,const double* vol,double* hs,const double* at,const double* rt,const int* rs,const int* st,const double* diam,double au,double lpf,double* c,double* co,double* reacted,const double* par,int pstride,const double* con,const double* hyd,int hstride,const Prog* prog,const Prog* term,const Instr* instr,int ls,int lt,int lp,int lc,int* nf,int* nj,int* na,int* nn,double* lh,int* re,GpuErr* ge){int sid=blockIdx.x*blockDim.x+threadIdx.x;if(sid>=ns)return;int pk=pipe[sid],base=row[sid]*(nsp+1),hb=sid*hstride;double ov=vol[sid],old[65],y[65],yn[65],k1[17],k2[17],fp[17],fm[17],w[17],a[17][17];int ix[17];int f=0,jc=0,ac=0,rc=0,reject=0;const double ur=2.3e-16,g=1.7071067811865475,eps=1.0e-7;double t=0.0,gh0=0.0,hmax=step,hmin=1.0e-8,h=hs[sid];for(int m=1;m<=nsp;m++){old[m]=c[base+m];y[m]=c[base+m];}if(nr>16){re[sid]=-3;set_err(ge,9001,1,sid,pk,rs[1],rs[1],0,(double)nr);return;}if(step>0.0&&nr>0){if(h==0.0){ros2_eval_rates(nr,rs,y,pk,par,pstride,con,hyd,hb,prog,term,instr,ls,lt,lp,lc,k1);f++;h=step;for(int i=1;i<=nr;i++){int m=rs[i];double tol=at[i]+rt[i]*fabs(y[m]);if(k1[i]!=0.0)h=fmin(h,tol/fabs(k1[i]));}}h=fmax(hmin,h);h=fmin(hmax,h);while(t<step){if(0.10*fabs(h)<=fabs(t)*ur){re[sid]=-2;set_err(ge,513,1,sid,pk,rs[1],rs[1],ac,h);return;}double tplus=t+h;if(tplus>step){h=step-t;tplus=step;}if(!reject){for(int col=1;col<=nr;col++){int m=rs[col],i;double tmp=y[m];y[m]=tmp+eps;ros2_eval_rates(nr,rs,y,pk,par,pstride,con,hyd,hb,prog,term,instr,ls,lt,lp,lc,fp);y[m]=tmp==0.0?tmp:tmp-eps;double e2=tmp==0.0?eps:2.0*eps;ros2_eval_rates(nr,rs,y,pk,par,pstride,con,hyd,hb,prog,term,instr,ls,lt,lp,lc,fm);for(i=1;i<=nr;i++)a[i][col]=(fp[i]-fm[i])/e2;y[m]=tmp;}jc++;f+=2*nr;gh0=0.0;}double gh=-1.0/(g*h),dgh=gh-gh0;for(int i=1;i<=nr;i++)a[i][i]+=dgh;gh0=gh;if(!ros2_lu_factor(a,nr,w,ix)){re[sid]=-1;set_err(ge,513,1,sid,pk,rs[1],rs[1],ac,0.0);return;}ros2_eval_rates(nr,rs,y,pk,par,pstride,con,hyd,hb,prog,term,instr,ls,lt,lp,lc,k1);f++;for(int i=1;i<=nr;i++)k1[i]*=gh;ros2_lu_solve(a,nr,ix,k1);for(int m=1;m<=nsp;m++)yn[m]=y[m];for(int i=1;i<=nr;i++){int m=rs[i];yn[m]=y[m]+h*k1[i];}ros2_eval_rates(nr,rs,yn,pk,par,pstride,con,hyd,hb,prog,term,instr,ls,lt,lp,lc,k2);f++;for(int i=1;i<=nr;i++)k2[i]=(k2[i]-2.0*k1[i])*gh;ros2_lu_solve(a,nr,ix,k2);for(int m=1;m<=nsp;m++)yn[m]=y[m];double er=0.0;for(int i=1;i<=nr;i++){int m=rs[i];yn[m]=y[m]+1.5*h*k1[i]+0.5*h*k2[i];double tol=at[i]+rt[i]*fabs(yn[m]);double q=fabs((yn[m]-y[m]-h*k1[i])/tol);er+=q*q;}er=fmax(ur,sqrt(er/(double)nr));double fac=0.9/sqrt(er);fac=fmin(fac,reject?1.0:10.0);fac=fmax(fac,0.1);h=fmin(hmax,fac*h);if(er>1.0){reject=1;rc++;h=0.5*h;}else{reject=0;for(int i=1;i<=nr;i++){int m=rs[i];y[m]=yn[m];if(y[m]<=ur)y[m]=0.0;}hs[sid]=h;t=tplus;ac++;}}}for(int m=1;m<=nsp;m++){c[base+m]=y[m];co[base+m]=c[base+m];double d=c[base+m]-old[m];double add=st[m]==0?ov*d*lpf:(diam[pk]>0.0?ov*4.0/diam[pk]*au*d:0.0);atomicAdd(&reacted[pk*(nsp+1)+m],add);if(!(c[base+m]==c[base+m]))c[base+m]=0.0;}nf[sid]=f;nj[sid]=jc;na[sid]=ac;nn[sid]=rc;lh[sid]=hs[sid];re[sid]=0;}\n"
"}\n";

typedef struct
{
    CUmodule module;
    CUfunction odeKernel;
    CUfunction ros2Kernel;
    CUfunction rk5Kernel;
    CUfunction rk5PipeWarpKernel;
    CUfunction equilKernel;
    CUfunction formulaKernel;
    int ready;
    int solver;
    int rk5Mode;
    int compiler;
    uint64_t cacheKey;
} GpuModuleState;

static GpuModuleState GpuModule;
static CUcontext GpuContext = NULL;

/* Immutable NH2CL chemistry/program tables.  Parameter changes after open
   are unsupported in this phase; Phase 3 uses a fixed compiled case. */
typedef struct { int ready,nLinks,nSpecies,nParams,nConsts,rateCount,eqCount,formulaCount,nInstr; int *rateSpecies,*eqSpecies,*formulaSpecies,*speciesType; double *rateAtol,*rateRtol,*params,*consts,*linkDiam; GpuInstrHost *instr; GpuProgramHost *speciesProg,*termProg; CUdeviceptr d_rateAtol,d_rateRtol,d_rateSpecies,d_eqSpecies,d_formulaSpecies,d_speciesType,d_params,d_consts,d_linkDiam,d_instr,d_speciesProg,d_termProg,d_err; GpuErrorHost *h_err; CUevent evStart[3],evStop[3],readyEvent; unsigned long long allocCount; } ResidentPrograms;
static ResidentPrograms ResidentProgram;

static int checkCu(CUresult r, int code)
{
    if (r != CUDA_SUCCESS)
    {
        const char *name = NULL;
        const char *text = NULL;
        FILE *logFile;
        cuGetErrorName(r, &name);
        cuGetErrorString(r, &text);
        logFile = fopen("msx_gpu_cuda_error.log", "at");
        if (logFile)
        {
            fprintf(logFile, "code=%d cu=%d name=%s text=%s\n",
                    code, (int)r, name ? name : "", text ? text : "");
            fclose(logFile);
        }
        setGpuError(code, GPU_STAGE_NONE, -1, -1, -1, -1, -1, (double)r);
        return code;
    }
    return 0;
}

static int checkNvrtc(nvrtcResult r)
{
    if (r != NVRTC_SUCCESS)
    {
        setGpuError(ERR_GPU_NVRTC_COMPILE_FAILED, GPU_STAGE_NONE, -1, -1, -1, -1, -1, (double)r);
        return ERR_GPU_NVRTC_COMPILE_FAILED;
    }
    return 0;
}

typedef struct
{
    char path[GPU_PATH_MAX];
    long long size;
    long long mtime;
} GpuCacheEntry;

static void normalizePathSeparators(char *path)
{
    while (*path)
    {
        if (*path == '\\') *path = '/';
        path++;
    }
}

static int getGpuCacheDir(char *dir, size_t size)
{
    char filePath[GPU_PATH_MAX];
    char *needle;
    size_t n;
    strncpy(filePath, __FILE__, sizeof(filePath) - 1);
    filePath[sizeof(filePath) - 1] = '\0';
    normalizePathSeparators(filePath);
    needle = strstr(filePath, "/src/solver/msxgpu.c");
    if (needle)
    {
        *needle = '\0';
        n = strlen(filePath);
        if (n + strlen("/gpu_cache") + 1 > size) return ERR_MEMORY;
        sprintf(dir, "%s/gpu_cache", filePath);
    }
    else
    {
        if (strlen("gpu_cache") + 1 > size) return ERR_MEMORY;
        strcpy(dir, "gpu_cache");
    }
    return 0;
}

static int ensureDirectory(const char *dir)
{
#ifdef _WIN32
    if (_mkdir(dir) == 0 || errno == EEXIST) return 0;
#else
    if (mkdir(dir, 0777) == 0 || errno == EEXIST) return 0;
#endif
    return ERR_IO_OUT_FILE;
}

static int isPtxFile(const char *name)
{
    size_t n = strlen(name);
    return n > 4 && strcmp(name + n - 4, ".ptx") == 0;
}

static int statCacheFile(const char *path, long long *size, long long *mtime)
{
#ifdef _WIN32
    struct _stat64 st;
    if (_stat64(path, &st) != 0) return 0;
    *size = (long long)st.st_size;
    *mtime = (long long)st.st_mtime;
#else
    struct stat st;
    if (stat(path, &st) != 0) return 0;
    *size = (long long)st.st_size;
    *mtime = (long long)st.st_mtime;
#endif
    return 1;
}

static void touchCacheFile(const char *path)
{
#ifdef _WIN32
    struct _utimbuf tb;
    tb.actime = (time_t)time(NULL);
    tb.modtime = tb.actime;
    _utime(path, &tb);
#else
    struct utimbuf tb;
    tb.actime = time(NULL);
    tb.modtime = tb.actime;
    utime(path, &tb);
#endif
}

static int compareCacheEntryMtime(const void *a, const void *b)
{
    const GpuCacheEntry *ea = (const GpuCacheEntry*)a;
    const GpuCacheEntry *eb = (const GpuCacheEntry*)b;
    if (ea->mtime < eb->mtime) return -1;
    if (ea->mtime > eb->mtime) return 1;
    return 0;
}

static void cleanupGpuCache(const char *dir)
{
    const int maxEntries = 4096;
    GpuCacheEntry *entries = (GpuCacheEntry*)calloc(maxEntries, sizeof(GpuCacheEntry));
    int nEntries = 0;
    long long total = 0;
    long long now = (long long)time(NULL);
    int i;
    if (!entries) return;
#ifdef _WIN32
    WIN32_FIND_DATAA fd;
    HANDLE h;
    char pattern[GPU_PATH_MAX];
    snprintf(pattern, sizeof(pattern), "%s/*.ptx", dir);
    h = FindFirstFileA(pattern, &fd);
    if (h == INVALID_HANDLE_VALUE)
    {
        free(entries);
        return;
    }
    do
    {
        char path[GPU_PATH_MAX];
        long long size, mtime;
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
        if (!isPtxFile(fd.cFileName)) continue;
        snprintf(path, sizeof(path), "%s/%s", dir, fd.cFileName);
        if (!statCacheFile(path, &size, &mtime)) continue;
        if (now - mtime > GPU_CACHE_MAX_AGE_SECONDS)
        {
            remove(path);
            continue;
        }
        if (nEntries < maxEntries)
        {
            strncpy(entries[nEntries].path, path, sizeof(entries[nEntries].path) - 1);
            entries[nEntries].path[sizeof(entries[nEntries].path) - 1] = '\0';
            entries[nEntries].size = size;
            entries[nEntries].mtime = mtime;
            total += size;
            nEntries++;
        }
    } while (FindNextFileA(h, &fd));
    FindClose(h);
#else
    DIR *d = opendir(dir);
    struct dirent *de;
    if (!d)
    {
        free(entries);
        return;
    }
    while ((de = readdir(d)) != NULL)
    {
        char path[GPU_PATH_MAX];
        long long size, mtime;
        if (!isPtxFile(de->d_name)) continue;
        snprintf(path, sizeof(path), "%s/%s", dir, de->d_name);
        if (!statCacheFile(path, &size, &mtime)) continue;
        if (now - mtime > GPU_CACHE_MAX_AGE_SECONDS)
        {
            remove(path);
            continue;
        }
        if (nEntries < maxEntries)
        {
            strncpy(entries[nEntries].path, path, sizeof(entries[nEntries].path) - 1);
            entries[nEntries].path[sizeof(entries[nEntries].path) - 1] = '\0';
            entries[nEntries].size = size;
            entries[nEntries].mtime = mtime;
            total += size;
            nEntries++;
        }
    }
    closedir(d);
#endif
    if (total <= GPU_CACHE_MAX_BYTES)
    {
        free(entries);
        return;
    }
    qsort(entries, nEntries, sizeof(entries[0]), compareCacheEntryMtime);
    for (i = 0; i < nEntries && total > GPU_CACHE_MAX_BYTES; i++)
    {
        if (remove(entries[i].path) == 0) total -= entries[i].size;
    }
    free(entries);
}

static int readFileBytes(const char *path, char **data, size_t *size)
{
    FILE *f = fopen(path, "rb");
    long len;
    char *buf;
    *data = NULL;
    *size = 0;
    if (!f) return 0;
    if (fseek(f, 0, SEEK_END) != 0)
    {
        fclose(f);
        return 0;
    }
    len = ftell(f);
    if (len <= 0)
    {
        fclose(f);
        return 0;
    }
    rewind(f);
    buf = (char*)malloc((size_t)len + 1);
    if (!buf)
    {
        fclose(f);
        return 0;
    }
    if (fread(buf, 1, (size_t)len, f) != (size_t)len)
    {
        free(buf);
        fclose(f);
        return 0;
    }
    fclose(f);
    buf[len] = '\0';
    *data = buf;
    *size = (size_t)len;
    touchCacheFile(path);
    return 1;
}

static int writeFileBytes(const char *path, const char *data, size_t size)
{
    FILE *f = fopen(path, "wb");
    if (!f) return ERR_IO_OUT_FILE;
    if (fwrite(data, 1, size, f) != size)
    {
        fclose(f);
        return ERR_IO_OUT_FILE;
    }
    fclose(f);
    touchCacheFile(path);
    return 0;
}

static uint64_t fnv1aBytes(uint64_t h, const void *data, size_t size)
{
    const unsigned char *p = (const unsigned char*)data;
    size_t i;
    for (i = 0; i < size; i++)
    {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static uint64_t hashExpr(uint64_t h, MathExpr *expr)
{
    while (expr)
    {
        h = fnv1aBytes(h, &expr->opcode, sizeof(expr->opcode));
        h = fnv1aBytes(h, &expr->ivar, sizeof(expr->ivar));
        h = fnv1aBytes(h, &expr->fvalue, sizeof(expr->fvalue));
        expr = expr->next;
    }
    return h;
}

static int collectSpecializedDimensions(GpuSpecializedDimensions *dims)
{
    int m;

    if (!dims) return ERR_GPU_UNSUPPORTED_FEATURE;
    memset(dims, 0, sizeof(*dims));
    dims->nSpecies = MSX.Nobjects[SPECIES];
    if (dims->nSpecies < 1 || dims->nSpecies > GPU_MAX_SPECIES)
        return ERR_GPU_UNSUPPORTED_FEATURE;
    for (m = 1; m <= dims->nSpecies; m++)
    {
        if (MSX.Species[m].pipeExprType == RATE) dims->nRate++;
        else if (MSX.Species[m].pipeExprType == EQUIL) dims->nEquil++;
        else if (MSX.Species[m].pipeExprType == FORMULA) dims->nFormula++;
    }
    if (dims->nEquil > GPU_MAX_EQUIL)
        return ERR_GPU_EQUIL_UNSUPPORTED;
    if (dims->nRate > GPU_MAX_SPECIES)
        return ERR_GPU_UNSUPPORTED_FEATURE;
    if (MSX.GpuSolver == ROS2 &&
        (dims->nRate < 1 || dims->nRate > GPU_ROS2_MAX_RATE_SPECIES))
        return ERR_GPU_UNSUPPORTED_FEATURE;
    return 0;
}

static uint64_t hashModelExpressions(void)
{
    uint64_t h = 1469598103934665603ULL;
    static const char generatorAbi[] = "specialized-model-dims-v1-kernel-abi1";
    GpuSpecializedDimensions dims;
    int m;
    int rateOrdinal = 0;
    int equilOrdinal = 0;
    int formulaOrdinal = 0;

    h = fnv1aBytes(h, generatorAbi, sizeof(generatorAbi) - 1);
    if (collectSpecializedDimensions(&dims))
    {
        const int invalid = -1;
        h = fnv1aBytes(h, &invalid, sizeof(invalid));
        return h;
    }
    h = fnv1aBytes(h, &dims.nSpecies, sizeof(dims.nSpecies));
    h = fnv1aBytes(h, &dims.nRate, sizeof(dims.nRate));
    h = fnv1aBytes(h, &dims.nEquil, sizeof(dims.nEquil));
    h = fnv1aBytes(h, &dims.nFormula, sizeof(dims.nFormula));
    h = fnv1aBytes(h, &MSX.Nobjects[TERM], sizeof(MSX.Nobjects[TERM]));
    for (m = 1; m <= MSX.Nobjects[TERM]; m++)
        h = hashExpr(h, MSX.Term[m].expr);
    for (m = 1; m <= dims.nSpecies; m++)
    {
        h = fnv1aBytes(h, &MSX.Species[m].pipeExprType, sizeof(MSX.Species[m].pipeExprType));
        h = fnv1aBytes(h, &m, sizeof(m));
        if (MSX.Species[m].pipeExprType == RATE)
        {
            h = fnv1aBytes(h, &rateOrdinal, sizeof(rateOrdinal));
            rateOrdinal++;
        }
        else if (MSX.Species[m].pipeExprType == EQUIL)
        {
            h = fnv1aBytes(h, &equilOrdinal, sizeof(equilOrdinal));
            equilOrdinal++;
        }
        else if (MSX.Species[m].pipeExprType == FORMULA)
        {
            h = fnv1aBytes(h, &formulaOrdinal, sizeof(formulaOrdinal));
            formulaOrdinal++;
        }
        h = hashExpr(h, MSX.Species[m].pipeExpr);
    }
    return h;
}

static int cacheGpuModelBinding(void)
{
    GpuSpecializedDimensions dims;
    int compiler = MSX.GpuCompiler ? 1 : 0;
    int err;

    memset(&dims, 0, sizeof(dims));
    if (compiler)
    {
        err = collectSpecializedDimensions(&dims);
        if (err)
        {
            clearGpuModelBinding();
            return err;
        }
    }
    memset(&GpuModel, 0, sizeof(GpuModel));
    GpuModel.compiler = compiler;
    GpuModel.dims = dims;
    GpuModel.cacheKey = compiler ? hashModelExpressions() : 0;
    GpuModel.valid = 1;
    return 0;
}

static void clearGpuModelBinding(void)
{
    memset(&GpuModel, 0, sizeof(GpuModel));
}

static int gpuRk5PipeWarpEnabled(void)
{
    const char *value = getenv("MSX_GPU_RK5_PIPE_WARP");
    if (!value) return 0;
    return strcmp(value, "1") == 0 || strcmp(value, "YES") == 0 || strcmp(value, "yes") == 0;
}

typedef struct
{
    char *data;
    size_t len;
    size_t cap;
} GpuSourceBuilder;

typedef struct
{
    char *value;
    char *deriv;
} GpuExprPair;

static void sbFree(GpuSourceBuilder *sb)
{
    if (!sb) return;
    free(sb->data);
    sb->data = NULL;
    sb->len = 0;
    sb->cap = 0;
}

static int sbReserve(GpuSourceBuilder *sb, size_t extra)
{
    size_t need;
    char *tmp;
    if (!sb) return ERR_MEMORY;
    need = sb->len + extra + 1;
    if (need <= sb->cap) return 0;
    if (sb->cap == 0) sb->cap = 4096;
    while (sb->cap < need) sb->cap *= 2;
    tmp = (char*)realloc(sb->data, sb->cap);
    if (!tmp) return ERR_MEMORY;
    sb->data = tmp;
    return 0;
}

static int sbAppendN(GpuSourceBuilder *sb, const char *text, size_t n)
{
    int err = sbReserve(sb, n);
    if (err) return err;
    memcpy(sb->data + sb->len, text, n);
    sb->len += n;
    sb->data[sb->len] = '\0';
    return 0;
}

static int sbAppend(GpuSourceBuilder *sb, const char *text)
{
    return sbAppendN(sb, text, strlen(text));
}

static int sbAppendFmt(GpuSourceBuilder *sb, const char *fmt, ...)
{
    va_list ap;
    va_list ap2;
    int n;
    int err;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0)
    {
        va_end(ap2);
        return ERR_MEMORY;
    }
    err = sbReserve(sb, (size_t)n);
    if (err)
    {
        va_end(ap2);
        return err;
    }
    vsnprintf(sb->data + sb->len, sb->cap - sb->len, fmt, ap2);
    va_end(ap2);
    sb->len += (size_t)n;
    return 0;
}

static char *gpuStrDup(const char *s)
{
    size_t n = strlen(s);
    char *r = (char*)malloc(n + 1);
    if (!r) return NULL;
    memcpy(r, s, n + 1);
    return r;
}

static char *gpuStrFmt(const char *fmt, ...)
{
    va_list ap;
    va_list ap2;
    int n;
    char *r;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0)
    {
        va_end(ap2);
        return NULL;
    }
    r = (char*)malloc((size_t)n + 1);
    if (!r)
    {
        va_end(ap2);
        return NULL;
    }
    vsnprintf(r, (size_t)n + 1, fmt, ap2);
    va_end(ap2);
    return r;
}

static void freeExprStack(char **stack, int n)
{
    int i;
    for (i = 0; i < n; i++) free(stack[i]);
}

static void freeExprPairStack(GpuExprPair *stack, int n)
{
    int i;
    for (i = 0; i < n; i++)
    {
        free(stack[i].value);
        free(stack[i].deriv);
    }
}

static char *emitCudaExprValue(MathExpr *expr, int depth);
static int emitCudaExprPair(MathExpr *expr, int wrt, int depth, GpuExprPair *out);

static char *emitCudaVarValue(int ivar, int depth)
{
    int lastSpecies = lastIndex(SPECIES);
    int lastTerm = lastIndex(TERM);
    int lastParam = lastIndex(PARAMETER);
    int lastConst = lastIndex(CONSTANT);
    if (ivar <= lastSpecies) return gpuStrFmt("c[%d]", ivar);
    if (ivar <= lastTerm)
    {
        int tid = ivar - lastSpecies;
        if (depth > MSX.Nobjects[TERM] + 1) return NULL;
        return emitCudaExprValue(MSX.Term[tid].expr, depth + 1);
    }
    if (ivar <= lastParam) return gpuStrFmt("params[pipe*pStride+%d]", ivar - lastTerm);
    if (ivar <= lastConst) return gpuStrFmt("constants[%d]", ivar - lastParam);
    return gpuStrFmt("hyd[hBase+%d]", ivar - lastConst);
}

static int emitCudaVarPair(int ivar, int wrt, int depth, GpuExprPair *out)
{
    int lastSpecies = lastIndex(SPECIES);
    int lastTerm = lastIndex(TERM);
    int lastParam = lastIndex(PARAMETER);
    int lastConst = lastIndex(CONSTANT);
    memset(out, 0, sizeof(*out));
    if (ivar <= lastSpecies)
    {
        out->value = gpuStrFmt("c[%d]", ivar);
        out->deriv = gpuStrDup(ivar == wrt ? "1.0" : "0.0");
    }
    else if (ivar <= lastTerm)
    {
        int tid = ivar - lastSpecies;
        if (depth > MSX.Nobjects[TERM] + 1) return ERR_GPU_UNSUPPORTED_FEATURE;
        return emitCudaExprPair(MSX.Term[tid].expr, wrt, depth + 1, out);
    }
    else if (ivar <= lastParam)
    {
        out->value = gpuStrFmt("params[pipe*pStride+%d]", ivar - lastTerm);
        out->deriv = gpuStrDup("0.0");
    }
    else if (ivar <= lastConst)
    {
        out->value = gpuStrFmt("constants[%d]", ivar - lastParam);
        out->deriv = gpuStrDup("0.0");
    }
    else
    {
        out->value = gpuStrFmt("hyd[hBase+%d]", ivar - lastConst);
        out->deriv = gpuStrDup("0.0");
    }
    return (out->value && out->deriv) ? 0 : ERR_MEMORY;
}

static char *emitUnaryValue(int opcode, const char *a)
{
    switch (opcode)
    {
    case 9:  return gpuStrFmt("-(%s)", a);
    case 10: return gpuStrFmt("cos(%s)", a);
    case 11: return gpuStrFmt("sin(%s)", a);
    case 12: return gpuStrFmt("tan(%s)", a);
    case 13: return gpuStrFmt("(1.0/tan(%s))", a);
    case 14: return gpuStrFmt("fabs(%s)", a);
    case 15: return gpuStrFmt("d_sgn(%s)", a);
    case 16: return gpuStrFmt("sqrt(%s)", a);
    case 17: return gpuStrFmt("log(%s)", a);
    case 18: return gpuStrFmt("exp(%s)", a);
    case 19: return gpuStrFmt("asin(%s)", a);
    case 20: return gpuStrFmt("acos(%s)", a);
    case 21: return gpuStrFmt("atan(%s)", a);
    case 22: return gpuStrFmt("(1.57079632679489661923-atan(%s))", a);
    case 23: return gpuStrFmt("sinh(%s)", a);
    case 24: return gpuStrFmt("cosh(%s)", a);
    case 25: return gpuStrFmt("tanh(%s)", a);
    case 26: return gpuStrFmt("(1.0/tanh(%s))", a);
    case 27: return gpuStrFmt("log10(%s)", a);
    case 28: return gpuStrFmt("d_step(%s)", a);
    default: return NULL;
    }
}

static int emitUnaryPair(int opcode, GpuExprPair a, GpuExprPair *out)
{
    char *tv = NULL;
    memset(out, 0, sizeof(*out));
    out->value = emitUnaryValue(opcode, a.value);
    switch (opcode)
    {
    case 9:
        out->deriv = gpuStrFmt("-(%s)", a.deriv);
        break;
    case 10:
        out->deriv = gpuStrFmt("d_safe_mul(-sin(%s),%s)", a.value, a.deriv);
        break;
    case 11:
        out->deriv = gpuStrFmt("d_safe_mul(cos(%s),%s)", a.value, a.deriv);
        break;
    case 12:
        tv = gpuStrFmt("tan(%s)", a.value);
        if (tv) out->deriv = gpuStrFmt("d_safe_mul((1.0+(%s)*(%s)),%s)", tv, tv, a.deriv);
        break;
    case 13:
        tv = gpuStrFmt("tan(%s)", a.value);
        if (tv) out->deriv = gpuStrFmt("d_safe_mul(-((1.0+(%s)*(%s))/((%s)*(%s))),%s)", tv, tv, tv, tv, a.deriv);
        break;
    case 14:
        out->deriv = gpuStrFmt("d_safe_mul(d_sgn(%s),%s)", a.value, a.deriv);
        break;
    case 15:
        out->deriv = gpuStrDup("0.0");
        break;
    case 16:
        out->deriv = gpuStrFmt("((%s)>0.0?d_safe_mul(0.5/sqrt(%s),%s):0.0)", a.value, a.value, a.deriv);
        break;
    case 17:
        out->deriv = gpuStrFmt("((%s)/(%s))", a.deriv, a.value);
        break;
    case 18:
        out->deriv = gpuStrFmt("d_safe_mul(exp(%s),%s)", a.value, a.deriv);
        break;
    case 19:
        out->deriv = gpuStrFmt("((%s)/sqrt(1.0-(%s)*(%s)))", a.deriv, a.value, a.value);
        break;
    case 20:
        out->deriv = gpuStrFmt("(-(%s)/sqrt(1.0-(%s)*(%s)))", a.deriv, a.value, a.value);
        break;
    case 21:
        out->deriv = gpuStrFmt("((%s)/(1.0+(%s)*(%s)))", a.deriv, a.value, a.value);
        break;
    case 22:
        out->deriv = gpuStrFmt("(-(%s)/(1.0+(%s)*(%s)))", a.deriv, a.value, a.value);
        break;
    case 23:
        out->deriv = gpuStrFmt("d_safe_mul(cosh(%s),%s)", a.value, a.deriv);
        break;
    case 24:
        out->deriv = gpuStrFmt("d_safe_mul(sinh(%s),%s)", a.value, a.deriv);
        break;
    case 25:
        tv = gpuStrFmt("tanh(%s)", a.value);
        if (tv) out->deriv = gpuStrFmt("d_safe_mul((1.0-(%s)*(%s)),%s)", tv, tv, a.deriv);
        break;
    case 26:
        tv = gpuStrFmt("tanh(%s)", a.value);
        if (tv) out->deriv = gpuStrFmt("d_safe_mul(-((1.0-(%s)*(%s))/((%s)*(%s))),%s)", tv, tv, tv, tv, a.deriv);
        break;
    case 27:
        out->deriv = gpuStrFmt("((%s)/((%s)*2.30258509299404568402))", a.deriv, a.value);
        break;
    case 28:
        out->deriv = gpuStrDup("0.0");
        break;
    default:
        break;
    }
    free(tv);
    free(a.value);
    free(a.deriv);
    return (out->value && out->deriv) ? 0 : ERR_MEMORY;
}

static char *emitCudaExprValue(MathExpr *expr, int depth)
{
    char *stack[GPU_MAX_STACK + 2];
    int sp = 0;
    MathExpr *node = expr;
    memset(stack, 0, sizeof(stack));
    if (!expr) return gpuStrDup("0.0");
    while (node)
    {
        char *v = NULL;
        char *a, *b;
        if (sp >= GPU_MAX_STACK) goto fail;
        switch (node->opcode)
        {
        case 3: case 4: case 5: case 6: case 31:
            if (sp < 2) goto fail;
            a = stack[--sp];
            b = stack[--sp];
            if (node->opcode == 3) v = gpuStrFmt("((%s)+(%s))", b, a);
            else if (node->opcode == 4) v = gpuStrFmt("((%s)-(%s))", b, a);
            else if (node->opcode == 5) v = gpuStrFmt("((%s)*(%s))", b, a);
            else if (node->opcode == 6) v = gpuStrFmt("((%s)/(%s))", b, a);
            else v = gpuStrFmt("d_pow(%s,%s)", b, a);
            free(a);
            free(b);
            if (!v) goto fail;
            stack[sp++] = v;
            break;
        case 7:
            v = gpuStrFmt("%.17g", node->fvalue);
            if (!v) goto fail;
            stack[sp++] = v;
            break;
        case 8:
            v = emitCudaVarValue(node->ivar, depth);
            if (!v) goto fail;
            stack[sp++] = v;
            break;
        case 9: case 10: case 11: case 12: case 13: case 14: case 15:
        case 16: case 17: case 18: case 19: case 20: case 21: case 22:
        case 23: case 24: case 25: case 26: case 27: case 28:
            if (sp < 1) goto fail;
            a = stack[--sp];
            v = emitUnaryValue(node->opcode, a);
            free(a);
            if (!v) goto fail;
            stack[sp++] = v;
            break;
        default:
            goto fail;
        }
        node = node->next;
    }
    if (sp != 1) goto fail;
    return stack[0];
fail:
    freeExprStack(stack, sp);
    return NULL;
}

static int emitCudaExprPair(MathExpr *expr, int wrt, int depth, GpuExprPair *out)
{
    GpuExprPair stack[GPU_MAX_STACK + 2];
    int sp = 0;
    MathExpr *node = expr;
    memset(stack, 0, sizeof(stack));
    memset(out, 0, sizeof(*out));
    if (!expr)
    {
        out->value = gpuStrDup("0.0");
        out->deriv = gpuStrDup("0.0");
        return (out->value && out->deriv) ? 0 : ERR_MEMORY;
    }
    while (node)
    {
        GpuExprPair a, b, r;
        memset(&r, 0, sizeof(r));
        if (sp >= GPU_MAX_STACK) goto fail;
        switch (node->opcode)
        {
        case 3: case 4: case 5: case 6: case 31:
            if (sp < 2) goto fail;
            a = stack[--sp];
            b = stack[--sp];
            if (node->opcode == 3)
            {
                r.value = gpuStrFmt("((%s)+(%s))", b.value, a.value);
                r.deriv = gpuStrFmt("((%s)+(%s))", b.deriv, a.deriv);
            }
            else if (node->opcode == 4)
            {
                r.value = gpuStrFmt("((%s)-(%s))", b.value, a.value);
                r.deriv = gpuStrFmt("((%s)-(%s))", b.deriv, a.deriv);
            }
            else if (node->opcode == 5)
            {
                r.value = gpuStrFmt("((%s)*(%s))", b.value, a.value);
                r.deriv = gpuStrFmt("(d_safe_mul(%s,%s)+d_safe_mul(%s,%s))", b.deriv, a.value, b.value, a.deriv);
            }
            else if (node->opcode == 6)
            {
                r.value = gpuStrFmt("((%s)/(%s))", b.value, a.value);
                r.deriv = gpuStrFmt("((d_safe_mul(%s,%s)-d_safe_mul(%s,%s))/((%s)*(%s)))",
                                    b.deriv, a.value, b.value, a.deriv, a.value, a.value);
            }
            else
            {
                r.value = gpuStrFmt("d_pow(%s,%s)", b.value, a.value);
                r.deriv = gpuStrFmt("((%s)>0.0?d_safe_mul(pow(%s,%s),(d_safe_mul(%s,log(%s))+d_safe_mul(%s,(%s)/(%s)))):0.0)",
                                    b.value, b.value, a.value, a.deriv, b.value, a.value, b.deriv, b.value);
            }
            free(b.value); free(b.deriv);
            free(a.value); free(a.deriv);
            if (!r.value || !r.deriv) goto fail;
            stack[sp++] = r;
            break;
        case 7:
            r.value = gpuStrFmt("%.17g", node->fvalue);
            r.deriv = gpuStrDup("0.0");
            if (!r.value || !r.deriv) goto fail;
            stack[sp++] = r;
            break;
        case 8:
            if (emitCudaVarPair(node->ivar, wrt, depth, &r)) goto fail;
            stack[sp++] = r;
            break;
        case 9: case 10: case 11: case 12: case 13: case 14: case 15:
        case 16: case 17: case 18: case 19: case 20: case 21: case 22:
        case 23: case 24: case 25: case 26: case 27: case 28:
            if (sp < 1) goto fail;
            a = stack[--sp];
            if (emitUnaryPair(node->opcode, a, &r)) goto fail;
            stack[sp++] = r;
            break;
        default:
            goto fail;
        }
        node = node->next;
    }
    if (sp != 1) goto fail;
    *out = stack[0];
    return 0;
fail:
    freeExprPairStack(stack, sp);
    return ERR_GPU_UNSUPPORTED_FEATURE;
}

static int appendSpecializedEvalProg(GpuSourceBuilder *sb)
{
    int m;
    int err;
    err = sbAppend(sb, "__device__ double eval_prog(int progId,const Prog* prog,const Instr* instr,double* c,int pipe,const double* params,int pStride,const double* constants,const double* hyd,int hBase,const Prog* termProg,int lastSpecies,int lastTerm,int lastParam,int lastConst){ (void)prog;(void)instr;(void)termProg;(void)lastSpecies;(void)lastTerm;(void)lastParam;(void)lastConst; switch(progId){\n");
    if (err) return err;
    for (m = 1; m <= MSX.Nobjects[SPECIES]; m++)
    {
        char *expr = emitCudaExprValue(MSX.Species[m].pipeExpr, 0);
        if (!expr) return ERR_GPU_UNSUPPORTED_FEATURE;
        err = sbAppendFmt(sb, "case %d:{ double v=(%s); return (v==v?v:0.0); }\n", m, expr);
        free(expr);
        if (err) return err;
    }
    err = sbAppend(sb, "default:return 0.0; }}\n");
    return err;
}

static int appendSpecializedEvalDeriv(GpuSourceBuilder *sb)
{
    int m, w;
    int err;
    err = sbAppend(sb, "__device__ double eval_prog_deriv(int progId,int wrt,const Prog* prog,const Instr* instr,double* c,int pipe,const double* params,int pStride,const double* constants,const double* hyd,int hBase,const Prog* termProg,int lastSpecies,int lastTerm,int lastParam,int lastConst){ (void)prog;(void)instr;(void)termProg;(void)lastSpecies;(void)lastTerm;(void)lastParam;(void)lastConst; switch(progId){\n");
    if (err) return err;
    for (m = 1; m <= MSX.Nobjects[SPECIES]; m++)
    {
        if (MSX.Species[m].pipeExprType != EQUIL) continue;
        err = sbAppendFmt(sb, "case %d: switch(wrt){\n", m);
        if (err) return err;
        for (w = 1; w <= MSX.Nobjects[SPECIES]; w++)
        {
            GpuExprPair pair;
            err = emitCudaExprPair(MSX.Species[m].pipeExpr, w, 0, &pair);
            if (err) return err;
            err = sbAppendFmt(sb, "case %d:{ double d=(%s); return (d==d?d:0.0); }\n", w, pair.deriv);
            free(pair.value);
            free(pair.deriv);
            if (err) return err;
        }
        err = sbAppend(sb, "default:return 0.0;}\n");
        if (err) return err;
    }
    err = sbAppend(sb, "default:return 0.0; }}\n");
    return err;
}

static int buildSpecializedCudaSource(char **source)
{
    const char *start;
    const char *end;
    GpuSourceBuilder generated = {0};
    GpuSourceBuilder result = {0};
    int err;
    *source = NULL;
    start = strstr(GpuReactCudaSource, "__device__ double eval_prog(");
    if (!start) return ERR_GPU_UNSUPPORTED_FEATURE;
    end = strstr(start, "#if defined(MSX_GPU_SOLVER_RK5)");
    if (!end) return ERR_GPU_UNSUPPORTED_FEATURE;
    err = appendSpecializedEvalProg(&generated);
    if (!err) err = appendSpecializedEvalDeriv(&generated);
    if (err)
    {
        sbFree(&generated);
        return err;
    }
    err = sbAppendN(&result, GpuReactCudaSource, (size_t)(start - GpuReactCudaSource));
    if (!err) err = sbAppend(&result, generated.data);
    if (!err) err = sbAppend(&result, end);
    sbFree(&generated);
    if (err)
    {
        sbFree(&result);
        return err;
    }
    *source = result.data;
    return 0;
}

/* The interpreter and specialized source share one generated kernel ABI.
   Keep the long literal fragments readable/unchanged, then apply the ABI
   splice once to the final source.  This also covers the separately appended
   ROS2 fragment and the specialized evaluator body. */
static int sourceReplaceAll(const char *source, const char *needle,
                            const char *replacement, char **out)
{
    const char *p, *q;
    GpuSourceBuilder sb = {0};
    size_t nlen;
    if (!source || !needle || !replacement || !out) return ERR_GPU_UNSUPPORTED_FEATURE;
    *out = NULL; nlen = strlen(needle); p = source;
    while ((q = strstr(p, needle)) != NULL)
    {
        if (sbAppendN(&sb, p, (size_t)(q - p)) || sbAppend(&sb, replacement))
        { sbFree(&sb); return ERR_MEMORY; }
        p = q + nlen;
    }
    if (sbAppend(&sb, p)) { sbFree(&sb); return ERR_MEMORY; }
    *out = sb.data;
    return 0;
}

/* Resize only numeric array extents in generated CUDA declarations.  The
   source fragments retain their legacy interpreter text; this helper is
   called exclusively for GPU_COMPILER YES after the model dimensions have
   been validated. */
static int replaceNamedArrayExtents(const char *source,
                                    const char *const *names,
                                    size_t nameCount,
                                    int dimension,
                                    int matrix,
                                    char **out)
{
    GpuSourceBuilder sb = {0};
    const char *p = source;

    if (!source || !names || !nameCount || dimension < 1 || !out)
        return ERR_GPU_UNSUPPORTED_FEATURE;
    *out = NULL;
    while (*p)
    {
        const char *name = NULL;
        const char *close = NULL;
        size_t nameLen = 0;
        size_t i;
        int value;

        for (i = 0; i < nameCount; i++)
        {
            size_t len = strlen(names[i]);
            const char *open;
            const char *q;
            char *end;
            if (strncmp(p, names[i], len) != 0) continue;
            open = p + len;
            if (*open != '[' || !isdigit((unsigned char)open[1])) continue;
            value = (int)strtol(open + 1, &end, 10);
            if (value < 1 || end == open + 1 || *end != ']') continue;
            q = end + 1;
            if (matrix)
            {
                char *matrixEnd;
                if (*q != '[' || !isdigit((unsigned char)q[1])) continue;
                (void)strtol(q + 1, &matrixEnd, 10);
                if (matrixEnd == q + 1 || *matrixEnd != ']') continue;
                close = matrixEnd + 1;
            }
            else close = q;
            name = names[i];
            nameLen = len;
            break;
        }
        if (!name)
        {
            if (sbAppendN(&sb, p, 1)) { sbFree(&sb); return ERR_MEMORY; }
            p++;
            continue;
        }
        if (sbAppendN(&sb, name, nameLen) ||
            sbAppendFmt(&sb, "[%d]", dimension))
        {
            sbFree(&sb);
            return ERR_MEMORY;
        }
        if (matrix && sbAppendFmt(&sb, "[%d]", dimension))
        {
            sbFree(&sb);
            return ERR_MEMORY;
        }
        p = close;
    }
    *out = sb.data;
    return 0;
}

static int specializeRos2CudaSource(const char *source,
                                    const GpuSpecializedDimensions *dims,
                                    char **out)
{
    static const char *stateArrays[] = { "old", "y", "yn" };
    static const char *rateArrays[] = { "k1", "k2", "fp", "fm", "w", "ix" };
    static const char *matrixArrays[] = { "a" };
    char *current = NULL;
    char *next = NULL;
    int err;

    if (!dims || dims->nRate < 1) return ERR_GPU_UNSUPPORTED_FEATURE;
    err = replaceNamedArrayExtents(source, stateArrays,
                                   sizeof(stateArrays) / sizeof(stateArrays[0]),
                                   dims->nSpecies + 1, 0, &current);
    if (err) return err;
    err = replaceNamedArrayExtents(current, rateArrays,
                                   sizeof(rateArrays) / sizeof(rateArrays[0]),
                                   dims->nRate + 1, 0, &next);
    free(current); current = NULL;
    if (err) return err;
    current = next; next = NULL;
    err = replaceNamedArrayExtents(current, matrixArrays,
                                   sizeof(matrixArrays) / sizeof(matrixArrays[0]),
                                   dims->nRate + 1, 1, &next);
    free(current);
    if (err) return err;
    *out = next;
    return 0;
}

static int specializeEquilCudaSource(const char *source,
                                     const GpuSpecializedDimensions *dims,
                                     char **out)
{
    static const char *vectorArrays[] = { "x", "f", "w", "indx" };
    static const char *matrixArrays[] = { "jmat", "a" };
    char *current = NULL;
    char *next = NULL;
    int dimension;
    int err;

    if (!dims || dims->nEquil < 0) return ERR_GPU_UNSUPPORTED_FEATURE;
    dimension = dims->nEquil + 1;
    err = replaceNamedArrayExtents(source, vectorArrays,
                                   sizeof(vectorArrays) / sizeof(vectorArrays[0]),
                                   dimension, 0, &current);
    if (err) return err;
    err = replaceNamedArrayExtents(current, matrixArrays,
                                   sizeof(matrixArrays) / sizeof(matrixArrays[0]),
                                   dimension, 1, &next);
    free(current);
    if (err) return err;
    *out = next;
    return 0;
}

static void writeSpecializedFunctionAttributes(FILE *f, const char *name,
                                               CUfunction function)
{
    int numRegs = -1;
    int localSize = -1;
    int sharedSize = -1;
    int maxThreads = -1;
    CUresult rRegs = cuFuncGetAttribute(&numRegs, CU_FUNC_ATTRIBUTE_NUM_REGS, function);
    CUresult rLocal = cuFuncGetAttribute(&localSize, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, function);
    CUresult rShared = cuFuncGetAttribute(&sharedSize, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function);
    CUresult rThreads = cuFuncGetAttribute(&maxThreads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function);
    fprintf(f, "function=%s,num_regs=%d,local_size_bytes=%d,shared_size_bytes=%d,max_threads_per_block=%d,status=%d/%d/%d/%d\n",
            name, numRegs, localSize, sharedSize, maxThreads,
            (int)rRegs, (int)rLocal, (int)rShared, (int)rThreads);
}

static void writeSpecializedMetadata(const char *cachePath, uint64_t cacheKey,
                                     const GpuSpecializedDimensions *dims)
{
    char metadataPath[GPU_PATH_MAX + 16];
    FILE *f;
    int m;
    int rateOrdinal = 0;
    int equilOrdinal = 0;
    int formulaOrdinal = 0;

    if (!MSXgpu_profileDetailEnabled() || !cachePath || !dims) return;
    if (snprintf(metadataPath, sizeof(metadataPath), "%s.meta", cachePath) < 0)
        return;
    f = fopen(metadataPath, "wt");
    if (!f) return;
    fprintf(f, "cache_version=%s\ncache_key=%016llx\nsolver=%d\n",
            GPU_CACHE_VERSION, (unsigned long long)cacheKey, MSX.GpuSolver);
    fprintf(f, "n_species=%d,n_rate=%d,n_equil=%d,n_formula=%d\n",
            dims->nSpecies, dims->nRate, dims->nEquil, dims->nFormula);
    fprintf(f, "species_index,type,ordinal\n");
    for (m = 1; m <= dims->nSpecies; m++)
    {
        int ordinal = 0;
        if (MSX.Species[m].pipeExprType == RATE) ordinal = ++rateOrdinal;
        else if (MSX.Species[m].pipeExprType == EQUIL) ordinal = ++equilOrdinal;
        else if (MSX.Species[m].pipeExprType == FORMULA) ordinal = ++formulaOrdinal;
        fprintf(f, "%d,%d,%d\n", m, MSX.Species[m].pipeExprType, ordinal);
    }
    fprintf(f, "attributes\n");
    if (MSX.GpuSolver == ROS2)
        writeSpecializedFunctionAttributes(f, "ros2_kernel", GpuModule.ros2Kernel);
    else if (MSX.GpuSolver == RK5)
        writeSpecializedFunctionAttributes(f, "rk5_align_kernel", GpuModule.rk5Kernel);
    else
        writeSpecializedFunctionAttributes(f, "ode_kernel", GpuModule.odeKernel);
    writeSpecializedFunctionAttributes(f, "equil_kernel", GpuModule.equilKernel);
    writeSpecializedFunctionAttributes(f, "formula_kernel", GpuModule.formulaKernel);
    fclose(f);
}

static int applyHydLayoutAbi(const char *source, char **out)
{
    static const struct { const char *from, *to; } edits[] = {
        {"const double* hyd,int hStride,const Prog* prog",
         "const double* hyd,int hStride,int hydLayout,const Prog* prog"},
        {"const double* hyd,int hstride,const Prog* prog",
         "const double* hyd,int hstride,int hydLayout,const Prog* prog"},
        {",hyd,hStride,prog", ",hyd,hStride,hydLayout,prog"},
        {"int hBase=sid*hStride;", "int hBase=(hydLayout==1?pipe:sid)*hStride;"},
        {"hb=sid*hstride", "hb=(hydLayout==1?pk:sid)*hstride"}
    };
    char *current = NULL, *next = NULL;
    size_t i;
    int err;
    if (!source || !out) return ERR_GPU_UNSUPPORTED_FEATURE;
    current = gpuStrDup(source);
    if (!current) return ERR_MEMORY;
    for (i = 0; i < sizeof(edits)/sizeof(edits[0]); ++i)
    {
        err = sourceReplaceAll(current, edits[i].from, edits[i].to, &next);
        free(current); current = NULL;
        if (err) return err;
        current = next; next = NULL;
    }
    *out = current;
    return 0;
}

static int compileSourceToPtx(const char *source, const char *arch, const char *solverMacro,
                              const char *modeMacro, const char *cachePath, char **ptx, size_t *ptxSize)
{
    const char *opts[4];
    int nOpts = 2;
    nvrtcProgram prog;
    nvrtcResult cr;
    int err;
    FILE *logFile;

    *ptx = NULL;
    *ptxSize = 0;
    opts[0] = arch;
    opts[1] = "--std=c++11";
    if (solverMacro && solverMacro[0])
        opts[nOpts++] = solverMacro;
    if (modeMacro && modeMacro[0])
        opts[nOpts++] = modeMacro;

    cr = nvrtcCreateProgram(&prog, source, "msx_gpu_react.cu", 0, NULL, NULL);
    err = checkNvrtc(cr);
    if (err) return err;
    cr = nvrtcCompileProgram(prog, nOpts, opts);
    if (cr != NVRTC_SUCCESS)
    {
        size_t logSize = 0;
        char *logText = NULL;
        nvrtcGetProgramLogSize(prog, &logSize);
        if (logSize > 1)
        {
            logText = (char*)malloc(logSize);
            if (logText)
            {
                nvrtcGetProgramLog(prog, logText);
                logFile = fopen("msx_gpu_nvrtc.log", "wt");
                if (logFile)
                {
                    fputs(logText, logFile);
                    fclose(logFile);
                }
                free(logText);
            }
        }
        nvrtcDestroyProgram(&prog);
        setGpuError(ERR_GPU_NVRTC_COMPILE_FAILED, GPU_STAGE_NONE, -1, -1, -1, -1, -1, (double)cr);
        return ERR_GPU_NVRTC_COMPILE_FAILED;
    }
    err = checkNvrtc(nvrtcGetPTXSize(prog, ptxSize));
    if (err)
    {
        nvrtcDestroyProgram(&prog);
        return err;
    }
    *ptx = (char*)malloc(*ptxSize);
    if (!*ptx)
    {
        nvrtcDestroyProgram(&prog);
        return ERR_MEMORY;
    }
    err = checkNvrtc(nvrtcGetPTX(prog, *ptx));
    nvrtcDestroyProgram(&prog);
    if (err)
    {
        free(*ptx);
        *ptx = NULL;
        return err;
    }
    if (cachePath && cachePath[0])
        writeFileBytes(cachePath, *ptx, *ptxSize);
    return 0;
}

static int ensureModule(void)
{
    int err;
    CUdevice dev;
    int major = 0, minor = 0;
    char arch[64];
    char cacheDir[GPU_PATH_MAX];
    char cachePath[GPU_PATH_MAX];
    const char *solverName;
    const char *solverMacro;
    const char *modeMacro = NULL;
    char *ptx = NULL;
    size_t ptxSize = 0;
    double startMs = 0.0, timer = 0.0, phaseStartMs;
    int timing = MSXgpu_profileStageEnabled();
    GpuSpecializedDimensions specializedDims;
    uint64_t cacheKey = 0;
    int haveSpecializedDims = 0;
    int requestedCompiler;
    int liveCompiler = MSX.GpuCompiler ? 1 : 0;

    memset(&specializedDims, 0, sizeof(specializedDims));
    if (!GpuModel.valid || GpuModel.compiler != liveCompiler)
    {
        err = cacheGpuModelBinding();
        if (err) return err;
    }
    requestedCompiler = GpuModel.compiler;
    cacheKey = GpuModel.cacheKey;
    if (requestedCompiler)
    {
        specializedDims = GpuModel.dims;
        haveSpecializedDims = 1;
    }
    if (GpuModule.ready && GpuModule.solver == MSX.GpuSolver &&
        (MSX.GpuSolver != RK5 || GpuModule.rk5Mode == MSX.GpuRk5Mode) &&
        GpuModule.compiler == requestedCompiler &&
        GpuModule.cacheKey == cacheKey) return 0;
    if (GpuModule.ready)
    {
        cuModuleUnload(GpuModule.module);
        memset(&GpuModule, 0, sizeof(GpuModule));
    }

    if (timing) startMs = MSXgpu_wallTimeMs();
    if (MSX.GpuSolver == RK5)
        solverName = (MSX.GpuRk5Mode == GPU_RK5_FAST_BUCKET) ? "rk5_fast_bucket" : "rk5_align";
    else if (MSX.GpuSolver == ROS2)
        solverName = "ros2";
    else
        solverName = "eul";
    solverMacro = (MSX.GpuSolver == RK5) ? "-DMSX_GPU_SOLVER_RK5" :
                  (MSX.GpuSolver == ROS2 ? "-DMSX_GPU_SOLVER_ROS2" : "-DMSX_GPU_SOLVER_EUL");
    if (MSX.GpuSolver == RK5 && MSX.GpuRk5Mode == GPU_RK5_FAST_BUCKET)
        modeMacro = "-DMSX_GPU_RK5_FAST_BUCKET";
    err = checkCu(cuInit(0), ERR_GPU_NOT_ENABLED);
    if (err) return err;
    err = checkCu(cuDeviceGet(&dev, 0), ERR_GPU_NOT_ENABLED);
    if (err) return err;
    err = checkCu(cuDevicePrimaryCtxRetain(&GpuContext, dev), ERR_GPU_NOT_ENABLED);
    if (err) return err;
    err = checkCu(cuCtxSetCurrent(GpuContext), ERR_GPU_NOT_ENABLED);
    if (err) return err;
    err = checkCu(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev), ERR_GPU_NOT_ENABLED);
    if (err) return err;
    err = checkCu(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev), ERR_GPU_NOT_ENABLED);
    if (err) return err;
    if (timing)
        MSXgpu_recordInitTime(MSX_INIT_CONTEXT, MSXgpu_wallTimeMs() - startMs);

    sprintf(arch, "--gpu-architecture=compute_%d%d", major, minor);

    err = getGpuCacheDir(cacheDir, sizeof(cacheDir));
    if (err) return err;
    err = ensureDirectory(cacheDir);
    if (err) return err;
    cleanupGpuCache(cacheDir);

    if (requestedCompiler)
    {
        snprintf(cachePath, sizeof(cachePath), "%s/specialized_%s_%016llx_sm%d%d_%s.ptx",
                 cacheDir, solverName, (unsigned long long)cacheKey, major, minor, GPU_CACHE_VERSION);
    }
    else
    {
        snprintf(cachePath, sizeof(cachePath), "%s/interp_%s_sm%d%d_%s.ptx",
                 cacheDir, solverName, major, minor, GPU_CACHE_VERSION);
    }

    if (timing) timer = MSXgpu_wallTimeMs();
    if (readFileBytes(cachePath, &ptx, &ptxSize))
    {
        if (timing)
        {
            phaseStartMs = MSXgpu_wallTimeMs() - timer;
            MSXgpu_recordInitTime(MSX_INIT_CACHE_READ, phaseStartMs);
            MSX.GpuTimingRecord.ptx_cache_read_ms += phaseStartMs;
        }
    }
    else
    {
        char *specializedSource = NULL;
        char *dimensionedSource = NULL;
        char *specializedRos2 = NULL;
        char *ros2Source = NULL;
        char *hydSource = NULL;
        const char *compileSource = GpuReactCudaSource;
        if (timing) timer = MSXgpu_wallTimeMs();
        if (requestedCompiler)
        {
            err = buildSpecializedCudaSource(&specializedSource);
            if (err) return err;
            err = specializeEquilCudaSource(specializedSource, &specializedDims,
                                            &dimensionedSource);
            if (err)
            {
                free(specializedSource);
                return err;
            }
            free(specializedSource);
            specializedSource = dimensionedSource;
            dimensionedSource = NULL;
            compileSource = specializedSource;
        }
        if (MSX.GpuSolver == ROS2)
        {
            const char *ros2Fragment = GpuRos2CudaSource;
            size_t baseLen = strlen(compileSource);
            size_t ros2Len;
            if (requestedCompiler)
            {
                err = specializeRos2CudaSource(GpuRos2CudaSource, &specializedDims,
                                               &specializedRos2);
                if (err)
                {
                    free(specializedSource);
                    free(dimensionedSource);
                    return err;
                }
                ros2Fragment = specializedRos2;
            }
            ros2Len = strlen(ros2Fragment);
            ros2Source = (char*)malloc(baseLen + ros2Len + 1);
            if (!ros2Source)
            {
                free(specializedSource);
                free(dimensionedSource);
                free(specializedRos2);
                return ERR_MEMORY;
            }
            memcpy(ros2Source, compileSource, baseLen);
            memcpy(ros2Source + baseLen, ros2Fragment, ros2Len + 1);
            compileSource = ros2Source;
        }
        err = applyHydLayoutAbi(compileSource, &hydSource);
        if (!err) err = compileSourceToPtx(hydSource, arch, solverMacro, modeMacro, cachePath, &ptx, &ptxSize);
        free(hydSource);
        free(ros2Source);
        free(specializedSource);
        free(dimensionedSource);
        free(specializedRos2);
        if (timing)
        {
            phaseStartMs = MSXgpu_wallTimeMs() - timer;
            MSXgpu_recordInitTime(MSX_INIT_COMPILE, phaseStartMs);
            MSX.GpuTimingRecord.nvrtc_compile_ms += phaseStartMs;
        }
        if (err) return err;
    }

    if (timing) timer = MSXgpu_wallTimeMs();
    err = checkCu(cuModuleLoadData(&GpuModule.module, ptx), ERR_GPU_NVRTC_COMPILE_FAILED);
    if (timing)
    {
        phaseStartMs = MSXgpu_wallTimeMs() - timer;
        MSXgpu_recordInitTime(MSX_INIT_MODULE, phaseStartMs);
        MSX.GpuTimingRecord.cu_module_load_ms += phaseStartMs;
    }
    free(ptx);
    if (err) return err;

    if (timing) timer = MSXgpu_wallTimeMs();
    if (MSX.GpuSolver == RK5)
    {
        const char *rk5KernelName = "rk5_align_kernel";
        err = checkCu(cuModuleGetFunction(&GpuModule.rk5Kernel, GpuModule.module, rk5KernelName), ERR_GPU_KERNEL_LAUNCH_FAILED);
        if (err) return err;
        err = checkCu(cuModuleGetFunction(&GpuModule.rk5PipeWarpKernel, GpuModule.module,
                                          "rk5_align_pipe_warp_kernel"), ERR_GPU_KERNEL_LAUNCH_FAILED);
        if (err) return err;
    }
    else if (MSX.GpuSolver == ROS2)
    {
        err = checkCu(cuModuleGetFunction(&GpuModule.ros2Kernel, GpuModule.module, "ros2_kernel"), ERR_GPU_KERNEL_LAUNCH_FAILED);
        if (err) return err;
    }
    else
    {
        err = checkCu(cuModuleGetFunction(&GpuModule.odeKernel, GpuModule.module, "ode_kernel"), ERR_GPU_KERNEL_LAUNCH_FAILED);
        if (err) return err;
    }
    err = checkCu(cuModuleGetFunction(&GpuModule.equilKernel, GpuModule.module, "equil_kernel"), ERR_GPU_KERNEL_LAUNCH_FAILED);
    if (err) return err;
    err = checkCu(cuModuleGetFunction(&GpuModule.formulaKernel, GpuModule.module, "formula_kernel"), ERR_GPU_KERNEL_LAUNCH_FAILED);
    if (err) return err;
    if (timing)
    {
        phaseStartMs = MSXgpu_wallTimeMs() - timer;
        MSXgpu_recordInitTime(MSX_INIT_MODULE, phaseStartMs);
        MSX.GpuTimingRecord.kernel_lookup_ms += phaseStartMs;
    }
    GpuModule.ready = 1;
    GpuModule.solver = MSX.GpuSolver;
    GpuModule.rk5Mode = MSX.GpuRk5Mode;
    GpuModule.compiler = requestedCompiler;
    GpuModule.cacheKey = cacheKey;
    if (haveSpecializedDims)
        writeSpecializedMetadata(cachePath, cacheKey, &specializedDims);
    if (timing)
    {
        phaseStartMs = MSXgpu_wallTimeMs() - startMs;
        MSX.GpuTimingRecord.jit_ms += phaseStartMs;
    }
    return 0;
}

static int appendExpr(MathExpr *expr, GpuInstrHost **instr, int *nInstr, int *cap,
                      GpuProgramHost *prog);

static int appendOneInstr(MathExpr *expr, GpuInstrHost **instr, int *nInstr, int *cap)
{
    if (*nInstr >= *cap)
    {
        int newCap = (*cap == 0) ? 256 : (*cap * 2);
        GpuInstrHost *tmp = (GpuInstrHost*)realloc(*instr, newCap * sizeof(GpuInstrHost));
        if (!tmp) return ERR_MEMORY;
        *instr = tmp;
        *cap = newCap;
    }
    (*instr)[*nInstr].opcode = expr->opcode;
    (*instr)[*nInstr].ivar = expr->ivar;
    (*instr)[*nInstr].fvalue = expr->fvalue;
    (*nInstr)++;
    return 0;
}

static int appendExprExpanded(MathExpr *expr, GpuInstrHost **instr, int *nInstr,
                              int *cap, int *count, int depth)
{
    const int lastSpecies = lastIndex(SPECIES);
    const int lastTerm = lastIndex(TERM);
    int err;
    if (depth > MSX.Nobjects[TERM] + 1) return ERR_GPU_UNSUPPORTED_FEATURE;
    while (expr)
    {
        if (expr->opcode == 8 && expr->ivar > lastSpecies && expr->ivar <= lastTerm)
        {
            int termIndex = expr->ivar - lastSpecies;
            err = appendExprExpanded(MSX.Term[termIndex].expr, instr, nInstr, cap, count, depth + 1);
            if (err) return err;
        }
        else
        {
            err = appendOneInstr(expr, instr, nInstr, cap);
            if (err) return err;
            (*count)++;
        }
        expr = expr->next;
    }
    return 0;
}

static int appendExpr(MathExpr *expr, GpuInstrHost **instr, int *nInstr, int *cap,
                      GpuProgramHost *prog)
{
    int first = *nInstr;
    int count = 0;
    int err = appendExprExpanded(expr, instr, nInstr, cap, &count, 0);
    if (err) return err;
    prog->first = first;
    prog->count = count;
    return 0;
}

static int buildPrograms(GpuInstrHost **instr, int *nInstr,
                         GpuProgramHost **speciesProg, GpuProgramHost **termProg)
{
    int cap = 0;
    int m;
    int err = 0;
    *nInstr = 0;
    *instr = NULL;
    *speciesProg = (GpuProgramHost*)calloc(MSX.Nobjects[SPECIES] + 1, sizeof(GpuProgramHost));
    *termProg = (GpuProgramHost*)calloc(MSX.Nobjects[TERM] + 1, sizeof(GpuProgramHost));
    if (!*speciesProg || !*termProg) return ERR_MEMORY;
    for (m = 1; m <= MSX.Nobjects[TERM]; m++)
    {
        err = appendExpr(MSX.Term[m].expr, instr, nInstr, &cap, &((*termProg)[m]));
        if (err) return err;
    }
    for (m = 1; m <= MSX.Nobjects[SPECIES]; m++)
    {
        err = appendExpr(MSX.Species[m].pipeExpr, instr, nInstr, &cap, &((*speciesProg)[m]));
        if (err) return err;
    }
    return 0;
}

/* Read a completed event pair without introducing another wait.  Detail-mode
   event pairs are consumed only after the dispatch's existing completion
   synchronize, so timing cannot serialize the kernels it measures. */
static double eventReadElapsedMs(CUevent start, CUevent stop)
{
    float ms = 0.0f;
    cuEventElapsedTime(&ms, start, stop);
    return (double)ms;
}

int MSXgpu_prepareResidentContext(void)
{
    int err = validateModelForGpu();
    if (err) return err;
    err = ensureModule();
    if (err) return err;
    return checkCu(cuCtxSetCurrent(GpuContext), ERR_GPU_NOT_ENABLED);
}

void MSXgpu_closeResidentPrograms(void)
{
    if (GpuContext)
        (void)checkCu(cuCtxSetCurrent(GpuContext), ERR_GPU_NOT_ENABLED);
    if (GpuModule.ready)
    {
        (void)checkCu(cuModuleUnload(GpuModule.module),
                      ERR_GPU_KERNEL_RUNTIME_ERROR);
        memset(&GpuModule, 0, sizeof(GpuModule));
    }
    clearGpuModelBinding();
#ifdef EPANETMSX_CUDA_ENABLED
    if (ResidentProgram.h_err) cuMemFreeHost(ResidentProgram.h_err);
#endif
#define RP_FREE_D(x) do { if (ResidentProgram.x) cuMemFree(ResidentProgram.x); } while (0)
    RP_FREE_D(d_rateAtol); RP_FREE_D(d_rateRtol); RP_FREE_D(d_rateSpecies);
    RP_FREE_D(d_eqSpecies); RP_FREE_D(d_formulaSpecies); RP_FREE_D(d_speciesType);
    RP_FREE_D(d_params); RP_FREE_D(d_consts); RP_FREE_D(d_linkDiam); RP_FREE_D(d_instr);
    RP_FREE_D(d_speciesProg); RP_FREE_D(d_termProg); RP_FREE_D(d_err);
#undef RP_FREE_D
    {
        int i;
        for (i = 0; i < 3; i++)
        {
            if (ResidentProgram.evStart[i]) cuEventDestroy(ResidentProgram.evStart[i]);
            if (ResidentProgram.evStop[i]) cuEventDestroy(ResidentProgram.evStop[i]);
        }
        if (ResidentProgram.readyEvent) cuEventDestroy(ResidentProgram.readyEvent);
    }
    free(ResidentProgram.rateSpecies); free(ResidentProgram.eqSpecies);
    free(ResidentProgram.formulaSpecies); free(ResidentProgram.speciesType);
    free(ResidentProgram.rateAtol); free(ResidentProgram.rateRtol); free(ResidentProgram.params);
    free(ResidentProgram.consts); free(ResidentProgram.linkDiam); free(ResidentProgram.instr);
    free(ResidentProgram.speciesProg); free(ResidentProgram.termProg);
    memset(&ResidentProgram, 0, sizeof(ResidentProgram));
}

int MSXgpu_openResidentPrograms(void)
{
    int err, nLinks, nSpecies, nParams, nConsts, rateCount = 0, eqCount = 0, formulaCount = 0;
    int m, k, nInstr = 0;
    GpuErrorHost zeroErr;
    err = MSXgpu_prepareResidentContext(); if (err) return err;
    if (ResidentProgram.ready) return 0;
    nLinks=MSX.Nobjects[LINK]; nSpecies=MSX.Nobjects[SPECIES]; nParams=MSX.Nobjects[PARAMETER]; nConsts=MSX.Nobjects[CONSTANT];
    for(m=1;m<=nSpecies;m++) { if(MSX.Species[m].pipeExprType==RATE)rateCount++; else if(MSX.Species[m].pipeExprType==EQUIL)eqCount++; else if(MSX.Species[m].pipeExprType==FORMULA)formulaCount++; }
    ResidentProgram.params=(double*)calloc((size_t)(nLinks+1)*(nParams+1),sizeof(double)); ResidentProgram.consts=(double*)calloc(nConsts+1,sizeof(double)); ResidentProgram.linkDiam=(double*)calloc(nLinks+1,sizeof(double));
    ResidentProgram.rateSpecies=(int*)calloc(rateCount+1,sizeof(int)); ResidentProgram.rateAtol=(double*)calloc(rateCount+1,sizeof(double)); ResidentProgram.rateRtol=(double*)calloc(rateCount+1,sizeof(double)); ResidentProgram.eqSpecies=(int*)calloc(eqCount+1,sizeof(int)); ResidentProgram.formulaSpecies=(int*)calloc(formulaCount+1,sizeof(int)); ResidentProgram.speciesType=(int*)calloc(nSpecies+1,sizeof(int));
    if(!ResidentProgram.params||!ResidentProgram.consts||!ResidentProgram.linkDiam||!ResidentProgram.rateSpecies||!ResidentProgram.rateAtol||!ResidentProgram.rateRtol||!ResidentProgram.eqSpecies||!ResidentProgram.formulaSpecies||!ResidentProgram.speciesType) { err=ERR_MEMORY; goto fail; }
    ResidentProgram.rateCount=ResidentProgram.eqCount=ResidentProgram.formulaCount=0;
    for(m=1;m<=nSpecies;m++){ ResidentProgram.speciesType[m]=MSX.Species[m].type; if(MSX.Species[m].pipeExprType==RATE){int i=++ResidentProgram.rateCount;ResidentProgram.rateSpecies[i]=m;ResidentProgram.rateAtol[i]=MSX.Species[m].aTol;ResidentProgram.rateRtol[i]=MSX.Species[m].rTol;}else if(MSX.Species[m].pipeExprType==EQUIL)ResidentProgram.eqSpecies[++ResidentProgram.eqCount]=m;else if(MSX.Species[m].pipeExprType==FORMULA)ResidentProgram.formulaSpecies[++ResidentProgram.formulaCount]=m; }
    for(m=1;m<=nConsts;m++)ResidentProgram.consts[m]=MSX.Const[m].value;
    for(k=1;k<=nLinks;k++){ResidentProgram.linkDiam[k]=MSX.Link[k].diam;for(m=1;m<=nParams;m++)ResidentProgram.params[k*(nParams+1)+m]=MSX.Link[k].param[m];}
    err=buildPrograms(&ResidentProgram.instr,&nInstr,&ResidentProgram.speciesProg,&ResidentProgram.termProg); if(err)goto fail; ResidentProgram.nInstr=nInstr; memset(&zeroErr,0,sizeof(zeroErr));
#define RP_ALLOC_COPY(x,h,n) do { err=checkCu(cuMemAlloc(&ResidentProgram.x,(n)),ERR_GPU_MEMORY_ALLOCATION_FAILED);if(err)goto fail;err=checkCu(profileCuMemcpyHtoD(ResidentProgram.x,(h),(n),MSX_PROFILE_TRANSFER_SCOPE_OTHER),ERR_GPU_MEMORY_ALLOCATION_FAILED);if(err)goto fail;} while(0)
    RP_ALLOC_COPY(d_rateAtol,ResidentProgram.rateAtol,(rateCount+1)*sizeof(double)); RP_ALLOC_COPY(d_rateRtol,ResidentProgram.rateRtol,(rateCount+1)*sizeof(double)); RP_ALLOC_COPY(d_rateSpecies,ResidentProgram.rateSpecies,(rateCount+1)*sizeof(int)); RP_ALLOC_COPY(d_eqSpecies,ResidentProgram.eqSpecies,(eqCount+1)*sizeof(int)); RP_ALLOC_COPY(d_formulaSpecies,ResidentProgram.formulaSpecies,(formulaCount+1)*sizeof(int)); RP_ALLOC_COPY(d_speciesType,ResidentProgram.speciesType,(nSpecies+1)*sizeof(int)); RP_ALLOC_COPY(d_params,ResidentProgram.params,(size_t)(nLinks+1)*(nParams+1)*sizeof(double)); RP_ALLOC_COPY(d_consts,ResidentProgram.consts,(nConsts+1)*sizeof(double)); RP_ALLOC_COPY(d_linkDiam,ResidentProgram.linkDiam,(nLinks+1)*sizeof(double)); RP_ALLOC_COPY(d_instr,ResidentProgram.instr,(nInstr?nInstr:1)*sizeof(GpuInstrHost)); RP_ALLOC_COPY(d_speciesProg,ResidentProgram.speciesProg,(nSpecies+1)*sizeof(GpuProgramHost)); RP_ALLOC_COPY(d_termProg,ResidentProgram.termProg,(MSX.Nobjects[TERM]+1)*sizeof(GpuProgramHost));
#undef RP_ALLOC_COPY
    err=checkCu(cuMemAlloc(&ResidentProgram.d_err,sizeof(zeroErr)),ERR_GPU_MEMORY_ALLOCATION_FAILED);if(err)goto fail;err=checkCu(profileCuMemcpyHtoD(ResidentProgram.d_err,&zeroErr,sizeof(zeroErr),MSX_PROFILE_TRANSFER_SCOPE_OTHER),ERR_GPU_MEMORY_ALLOCATION_FAILED);if(err)goto fail;
    /* The error copy is enqueued before the ready event.  Keep its host
       destination pinned for the complete resident-program lifetime: the
       caller's lifecycle token is commonly stack storage and is not valid as
       an asynchronous Driver-API destination. */
    err=checkCu(cuMemHostAlloc((void **)&ResidentProgram.h_err,sizeof(zeroErr),CU_MEMHOSTALLOC_PORTABLE),ERR_GPU_MEMORY_ALLOCATION_FAILED);if(err)goto fail;
    memset(ResidentProgram.h_err,0,sizeof(zeroErr));
    /* The resident CUDA core owns the Runtime stream.  This Driver event is
       recorded on the exact stream handle returned by prepareActive; it is
       a dependency token, not a second stream. */
    err=checkCu(cuEventCreate(&ResidentProgram.readyEvent,CU_EVENT_DISABLE_TIMING),ERR_GPU_MEMORY_ALLOCATION_FAILED);if(err)goto fail;
    /* Completion is provided by the stream/context synchronize below.  These
       events exist only for detail-mode device timing. */
    if (MSXgpu_profileDetailGroupEnabled(MSX_PROFILE_DETAIL_CHEM))
    {
        int i;
        for (i = 0; i < 3; i++)
        {
            err=checkCu(cuEventCreate(&ResidentProgram.evStart[i],CU_EVENT_DEFAULT),ERR_GPU_MEMORY_ALLOCATION_FAILED);if(err)goto fail;
            err=checkCu(cuEventCreate(&ResidentProgram.evStop[i],CU_EVENT_DEFAULT),ERR_GPU_MEMORY_ALLOCATION_FAILED);if(err)goto fail;
        }
    }
    ResidentProgram.ready=1; ResidentProgram.allocCount++; return 0;
fail: MSXgpu_closeResidentPrograms(); return err;
}

/* P3 lifecycle: prepare and launch on the Runtime-owned stream, then return
   before any device wait.  The Driver API consumes the same underlying stream
   handle exposed by the CUDA Runtime core. */
static int submitResidentCoreHydInternal(MSXResidentGpu *resident,
                                         const MSXResidentActiveBatch *batch,
                                         MSXResidentGpuActiveWriter *writer,
                                         const MSXResidentHydView *hyd,
                                         double dt, MSXgpuResidentCoreToken *token)
{
    MSXResidentGpuDeviceView view;
    MSXResidentStatus residentStatus;
    CUstream stream;
    int err = 0, nSpecies = MSX.Nobjects[SPECIES], nParams = MSX.Nobjects[PARAMETER];
    int nSeg, block = 128, grid, rateCount, eqCount, formulaCount;
    int lastSpecies, lastTerm, lastParam, lastConst, hStride, hydLayout, paramStride;
    double tstep, areaUcf, lperFt3;
    void *ros2Args[40], *equilArgs[30], *formulaArgs[30];
    double prepareStart = 0.0, launchStart = 0.0;
    int stage = MSXgpu_profileStageEnabled();
    int detail = MSXgpu_profileDetailGroupEnabled(MSX_PROFILE_DETAIL_CHEM);
    if (!token) return ERR_GPU_UNSUPPORTED_FEATURE;
    if (token->inFlight || token->magic) return ERR_GPU_KERNEL_RUNTIME_ERROR;
    memset(token, 0, sizeof(*token));
    if (!resident || (!writer && (!batch || (!batch->item && batch->itemCount))))
        return ERR_GPU_UNSUPPORTED_FEATURE;
    err = MSXgpu_prepareResidentContext();
    if (err) return err;
    if (!ResidentProgram.ready || MSX.GpuSolver != ROS2 || MSX.Solver != ROS2)
        return ERR_GPU_SOLVER_UNSUPPORTED;
    MSX.GpuTimingRecord.react_gpu = 1;
    MSX.GpuTimingRecord.ode_gpu = 1;
    MSX.GpuTimingRecord.equil_gpu = (MSX.GpuEquil != 0);
    MSX.GpuTimingRecord.formula_gpu = (MSX.GpuFormula != 0);
    MSXresidentGpu_setDiagnosticMode(resident,
        MSXgpu_profileDetailGroupEnabled(MSX_PROFILE_DETAIL_DIAGNOSTIC));
    if (stage) prepareStart = MSXgpu_wallTimeMs();
    residentStatus = writer ? MSXresidentGpu_prepareSealedActiveHyd(resident, writer, hyd, &view, NULL) :
                              (hyd ? MSXresidentGpu_prepareActiveHyd(resident, batch, hyd, &view, NULL) :
                                     MSXresidentGpu_prepareActive(resident, batch, &view, NULL));
    if (stage) MSXgpu_profileRunPhase(MSX_PROFILE_RUN_REACT_PREPARE,
                                      MSXgpu_wallTimeMs() - prepareStart);
    if (residentStatus != MSX_RESIDENT_OK)
        return mapResidentStatus(residentStatus);
    if (stage) launchStart = MSXgpu_wallTimeMs();
    stream = (CUstream)(uintptr_t)view.streamHandle;
    if (!stream) { err=ERR_GPU_KERNEL_RUNTIME_ERROR; goto submit_fail; }
    rateCount = ResidentProgram.rateCount; eqCount = ResidentProgram.eqCount;
    formulaCount = ResidentProgram.formulaCount;
    if (view.speciesStride != (uint32_t)(nSpecies + 1) ||
        view.hydStride != MAX_HYD_VARS || view.hydLayout > MSX_RESIDENT_HYD_PIPE_MAJOR)
    { err=ERR_GPU_UNSUPPORTED_FEATURE; goto submit_fail; }
    err=checkCu(cuMemsetD8Async(ResidentProgram.d_err,0,sizeof(GpuErrorHost),stream),ERR_GPU_KERNEL_LAUNCH_FAILED);
    if (err) goto submit_fail;
    if (view.itemCount)
    {
        CUdeviceptr d_segPipe=(CUdeviceptr)view.segPipe,d_segRow=(CUdeviceptr)view.segRow;
        CUdeviceptr d_segVol=(CUdeviceptr)view.segVol,d_hstep=(CUdeviceptr)view.hstep;
        CUdeviceptr d_c=(CUdeviceptr)view.c,d_cOde=(CUdeviceptr)view.lastc;
        CUdeviceptr d_hyd=(CUdeviceptr)view.hyd,d_reacted=(CUdeviceptr)view.reacted;
        CUdeviceptr d_nf=(CUdeviceptr)view.ros2Nfcn,d_nj=(CUdeviceptr)view.ros2Njac;
        CUdeviceptr d_na=(CUdeviceptr)view.ros2Naccept,d_nr=(CUdeviceptr)view.ros2Nreject;
        CUdeviceptr d_lh=(CUdeviceptr)view.ros2LastHstep,d_re=(CUdeviceptr)view.ros2Err;
        nSeg=(int)view.itemCount; hStride=(int)view.hydStride; hydLayout=(int)view.hydLayout; paramStride=nParams+1;
        grid=(nSeg+block-1)/block; tstep=dt/MSX.Ucf[RATE_UNITS];
        areaUcf=MSX.Ucf[AREA_UNITS]; lperFt3=LperFT3;
        lastSpecies=lastIndex(SPECIES); lastTerm=lastIndex(TERM);
        lastParam=lastIndex(PARAMETER); lastConst=lastIndex(CONSTANT);
        ros2Args[0]=&nSeg; ros2Args[1]=&nSpecies; ros2Args[2]=&rateCount; ros2Args[3]=&tstep;
        ros2Args[4]=&d_segPipe; ros2Args[5]=&d_segRow; ros2Args[6]=&d_segVol; ros2Args[7]=&d_hstep;
        ros2Args[8]=&ResidentProgram.d_rateAtol; ros2Args[9]=&ResidentProgram.d_rateRtol;
        ros2Args[10]=&ResidentProgram.d_rateSpecies; ros2Args[11]=&ResidentProgram.d_speciesType;
        ros2Args[12]=&ResidentProgram.d_linkDiam; ros2Args[13]=&areaUcf; ros2Args[14]=&lperFt3;
        ros2Args[15]=&d_c; ros2Args[16]=&d_cOde; ros2Args[17]=&d_reacted;
        ros2Args[18]=&ResidentProgram.d_params; ros2Args[19]=&paramStride;
        ros2Args[20]=&ResidentProgram.d_consts; ros2Args[21]=&d_hyd; ros2Args[22]=&hStride; ros2Args[23]=&hydLayout;
        ros2Args[24]=&ResidentProgram.d_speciesProg; ros2Args[25]=&ResidentProgram.d_termProg;
        ros2Args[26]=&ResidentProgram.d_instr; ros2Args[27]=&lastSpecies; ros2Args[28]=&lastTerm;
        ros2Args[29]=&lastParam; ros2Args[30]=&lastConst; ros2Args[31]=&d_nf; ros2Args[32]=&d_nj;
        ros2Args[33]=&d_na; ros2Args[34]=&d_nr; ros2Args[35]=&d_lh; ros2Args[36]=&d_re;
        ros2Args[37]=&ResidentProgram.d_err;
        equilArgs[0]=&nSeg; equilArgs[1]=&nSpecies; equilArgs[2]=&eqCount;
        equilArgs[3]=&d_segPipe; equilArgs[4]=&d_segRow; equilArgs[5]=&ResidentProgram.d_eqSpecies;
        equilArgs[6]=&d_c; equilArgs[7]=&ResidentProgram.d_params; equilArgs[8]=&paramStride;
        equilArgs[9]=&ResidentProgram.d_consts; equilArgs[10]=&d_hyd; equilArgs[11]=&hStride; equilArgs[12]=&hydLayout;
        equilArgs[13]=&ResidentProgram.d_speciesProg; equilArgs[14]=&ResidentProgram.d_termProg;
        equilArgs[15]=&ResidentProgram.d_instr; equilArgs[16]=&lastSpecies; equilArgs[17]=&lastTerm;
        equilArgs[18]=&lastParam; equilArgs[19]=&lastConst; equilArgs[20]=&ResidentProgram.d_err;
        formulaArgs[0]=&nSeg; formulaArgs[1]=&nSpecies; formulaArgs[2]=&formulaCount;
        formulaArgs[3]=&d_segPipe; formulaArgs[4]=&d_segRow; formulaArgs[5]=&ResidentProgram.d_formulaSpecies;
        formulaArgs[6]=&d_c; formulaArgs[7]=&ResidentProgram.d_params; formulaArgs[8]=&paramStride;
        formulaArgs[9]=&ResidentProgram.d_consts; formulaArgs[10]=&d_hyd; formulaArgs[11]=&hStride; formulaArgs[12]=&hydLayout;
        formulaArgs[13]=&ResidentProgram.d_speciesProg; formulaArgs[14]=&ResidentProgram.d_termProg;
        formulaArgs[15]=&ResidentProgram.d_instr; formulaArgs[16]=&lastSpecies; formulaArgs[17]=&lastTerm;
        formulaArgs[18]=&lastParam; formulaArgs[19]=&lastConst; formulaArgs[20]=&ResidentProgram.d_err;
        if (detail) cuEventRecord(ResidentProgram.evStart[0], stream);
        err=checkCu(cuLaunchKernel(GpuModule.ros2Kernel,grid,1,1,block,1,1,0,stream,ros2Args,NULL),ERR_GPU_KERNEL_LAUNCH_FAILED);
        if (err) goto submit_fail;
        if (detail) cuEventRecord(ResidentProgram.evStop[0], stream);
        if (eqCount) {
            if (detail) cuEventRecord(ResidentProgram.evStart[1], stream);
            err=checkCu(cuLaunchKernel(GpuModule.equilKernel,grid,1,1,block,1,1,0,stream,equilArgs,NULL),ERR_GPU_KERNEL_LAUNCH_FAILED);
            if (err) goto submit_fail;
            if (detail) cuEventRecord(ResidentProgram.evStop[1], stream);
        }
        if (formulaCount) {
            if (detail) cuEventRecord(ResidentProgram.evStart[2], stream);
            err=checkCu(cuLaunchKernel(GpuModule.formulaKernel,grid,1,1,block,1,1,0,stream,formulaArgs,NULL),ERR_GPU_KERNEL_LAUNCH_FAILED);
            if (err) goto submit_fail;
            if (detail) cuEventRecord(ResidentProgram.evStop[2], stream);
        }
    }
    residentStatus = MSXresidentGpu_enqueueActiveCompletion(resident);
    if (residentStatus != MSX_RESIDENT_OK) { err=mapResidentStatus(residentStatus); goto submit_fail; }
    err=checkCu(profileCuMemcpyDtoHAsync(ResidentProgram.h_err,ResidentProgram.d_err,
                                  sizeof(GpuErrorHost),stream,
                                  MSX_PROFILE_TRANSFER_SCOPE_DIAGNOSTIC),ERR_GPU_KERNEL_LAUNCH_FAILED);
    if (err) goto submit_fail;
    err=checkCu(cuEventRecord(ResidentProgram.readyEvent,stream),ERR_GPU_KERNEL_LAUNCH_FAILED);
    if (err) goto submit_fail;
    if (stage) MSXgpu_profileRunPhase(MSX_PROFILE_RUN_REACT_GPU_SUBMIT,
                                      MSXgpu_wallTimeMs() - launchStart);
    token->magic=UINT64_C(0x52534944454e5433); token->batchId=(uint64_t)view.itemCount;
    token->itemCount=view.itemCount; token->inFlight=1; token->gpu=resident;
    token->view=view; token->topologyVersion=0; token->submitMs=stage ? MSXgpu_wallTimeMs()-prepareStart : 0.0;
    return 0;
submit_fail:
    if (stage) MSXgpu_profileRunPhase(MSX_PROFILE_RUN_REACT_GPU_SUBMIT,
                                      MSXgpu_wallTimeMs() - launchStart);
    MSXresidentGpu_abortActive(resident);
    setGpuError(err ? err : ERR_GPU_KERNEL_RUNTIME_ERROR,GPU_STAGE_NONE,-1,-1,-1,-1,-1,0.0);
    return err ? err : ERR_GPU_KERNEL_RUNTIME_ERROR;
}

int MSXgpu_submitResidentCoreHyd(MSXResidentGpu *resident,
                                 const MSXResidentActiveBatch *batch,
                                 const MSXResidentHydView *hyd,
                                 double dt, MSXgpuResidentCoreToken *token)
{ return submitResidentCoreHydInternal(resident, batch, NULL, hyd, dt, token); }

int MSXgpu_submitResidentCoreHydPrepared(MSXResidentGpu *resident,
                                         MSXResidentGpuActiveWriter *writer,
                                         const MSXResidentHydView *hyd,
                                         double dt, MSXgpuResidentCoreToken *token)
{ return submitResidentCoreHydInternal(resident, NULL, writer, hyd, dt, token); }

int MSXgpu_submitResidentCore(MSXResidentGpu *resident, const MSXResidentActiveBatch *batch,
                              double dt, MSXgpuResidentCoreToken *token)
{
    return MSXgpu_submitResidentCoreHyd(resident, batch, NULL, dt, token);
}

int MSXgpu_finishResidentCore(MSXResidentGpu *resident, MSXgpuResidentCoreToken *token,
                              MSXResidentGpuReactResult *result)
{
    MSXResidentGpuReactResult finished; MSXResidentStatus status;
    double waitStart=0.0, finishStart=0.0, odeMs=0.0, eqMs=0.0, formulaMs=0.0;
    int stage=MSXgpu_profileStageEnabled(), detail=MSXgpu_profileDetailGroupEnabled(MSX_PROFILE_DETAIL_CHEM);
    if (result) memset(result,0,sizeof(*result));
    if (!resident || !token || token->magic != UINT64_C(0x52534944454e5433) ||
        !token->inFlight || token->gpu != resident)
        return ERR_GPU_KERNEL_RUNTIME_ERROR;
    if (stage) waitStart=MSXgpu_wallTimeMs();
    if (checkCu(cuEventSynchronize(ResidentProgram.readyEvent),ERR_GPU_KERNEL_RUNTIME_ERROR))
    {
        if (stage) MSXgpu_profileRunPhase(MSX_PROFILE_RUN_REACT_GPU_WAIT,MSXgpu_wallTimeMs()-waitStart);
        MSXresidentGpu_abortActive(resident); token->inFlight=0; token->magic=0;
        return ERR_GPU_KERNEL_RUNTIME_ERROR;
    }
    if (stage) MSXgpu_profileRunPhase(MSX_PROFILE_RUN_REACT_GPU_WAIT,MSXgpu_wallTimeMs()-waitStart);
    memset(&finished,0,sizeof(finished)); if (stage) finishStart=MSXgpu_wallTimeMs();
    status=MSXresidentGpu_finishActiveAfterWait(resident,&finished);
    if (stage) MSXgpu_profileRunPhase(MSX_PROFILE_RUN_REACT_GPU_FINISH,MSXgpu_wallTimeMs()-finishStart);
    if (status != MSX_RESIDENT_OK)
    {
        setGpuError(mapResidentStatus(status),GPU_STAGE_NONE,-1,-1,-1,-1,-1,(double)status);
        token->inFlight=0; token->magic=0; return mapResidentStatus(status);
    }
    if (ResidentProgram.h_err)
        memcpy(&token->gpuError, ResidentProgram.h_err,
               sizeof(token->gpuError));
    if (token->gpuError.code)
    {
        setGpuError(ERR_GPU_KERNEL_RUNTIME_ERROR, token->gpuError.stage,
                    token->gpuError.sid, token->gpuError.pipe,
                    token->gpuError.species, token->gpuError.expr,
                    token->gpuError.iter, token->gpuError.value);
        /* The program may have updated only a prefix of the active rows before
           reporting its compact error.  Never expose that partial state to a
           later aggregate or selected fetch. */
        (void)MSXresidentGpu_poison(resident);
        token->inFlight=0; token->magic=0;
        return ERR_GPU_KERNEL_RUNTIME_ERROR;
    }
    if (detail && token->itemCount) {
        odeMs=eventReadElapsedMs(ResidentProgram.evStart[0],ResidentProgram.evStop[0]);
        if (ResidentProgram.eqCount) eqMs=eventReadElapsedMs(ResidentProgram.evStart[1],ResidentProgram.evStop[1]);
        if (ResidentProgram.formulaCount) formulaMs=eventReadElapsedMs(ResidentProgram.evStart[2],ResidentProgram.evStop[2]);
        MSX.GpuTimingRecord.ros2_nfcn+=finished.ros2Nfcn; MSX.GpuTimingRecord.ros2_njac+=finished.ros2Njac;
        MSX.GpuTimingRecord.ros2_naccept+=finished.ros2Naccept; MSX.GpuTimingRecord.ros2_nreject+=finished.ros2Nreject;
        if (finished.ros2LastHstep!=0.0) MSX.GpuTimingRecord.ros2_last_hstep=finished.ros2LastHstep;
    }
    MSX.GpuTimingRecord.react_ode_ms+=odeMs; MSX.GpuTimingRecord.react_equil_ms+=eqMs; MSX.GpuTimingRecord.react_formula_ms+=formulaMs;
    if (finished.ros2Error && !MSX.GpuTimingRecord.ros2_error_code) MSX.GpuTimingRecord.ros2_error_code=finished.ros2Error;
    finished.ros2Ms=odeMs; finished.equilMs=eqMs; finished.formulaMs=formulaMs;
    if (result) *result=finished;
    token->inFlight=0; token->magic=0;
    return finished.ros2Error ? ERR_GPU_KERNEL_RUNTIME_ERROR : 0;
}

int MSXgpu_abortResidentCore(MSXResidentGpu *resident, MSXgpuResidentCoreToken *token)
{
    MSXResidentStatus status;
    if (!resident || !token || token->magic != UINT64_C(0x52534944454e5433) ||
        !token->inFlight || token->gpu != resident)
        return ERR_GPU_KERNEL_RUNTIME_ERROR;
    status=MSXresidentGpu_abortActive(resident);
    token->inFlight=0; token->magic=0;
    return status==MSX_RESIDENT_OK ? 0 : mapResidentStatus(status);
}

/* Compatibility wrapper for non-overlapped callers. */
int MSXgpu_reactResidentCore(MSXResidentGpu *resident, const MSXResidentActiveBatch *batch,
                             double dt, MSXResidentGpuReactResult *result)
{
    MSXgpuResidentCoreToken token; int err;
    memset(&token,0,sizeof(token)); err=MSXgpu_submitResidentCore(resident,batch,dt,&token);
    if (err) return err; return MSXgpu_finishResidentCore(resident,&token,result);
}


static int rk5FastBucket(double hstep, double tstep)
{
    int nSub;
    if (!(hstep > 0.0) || hstep > tstep) hstep = tstep;
    if (!(hstep > 0.0)) return 0;
    nSub = (int)ceil(tstep / hstep);
    if (nSub <= 1) return 0;
    if (nSub <= 2) return 1;
    if (nSub <= 4) return 2;
    if (nSub <= 8) return 3;
    if (nSub <= 16) return 4;
    return 5;
}

static int reorderSegmentsForRk5Fast(int nSeg, int nSpecies, double tstep,
                                     int *segPipe, double *segVol, double *hstep,
                                     Pseg *segPtrs, int *unpackOrder,
                                     double *c, double *cOde, double *hyd,
                                     int *bucketOffset, int *bucketCount, int *bucketMaxSize,
                                     int *didReorder)
{
    int *bucket = NULL, *count = NULL, *offset = NULL, *cursor = NULL, *order = NULL;
    int *tmpSegPipe = NULL;
    double *tmpSegVol = NULL, *tmpHstep = NULL, *tmpC = NULL, *tmpCOde = NULL, *tmpHyd = NULL;
    Pseg *tmpSegPtrs = NULL;
    int sid, b;
    size_t cRow = (size_t)(nSpecies + 1) * sizeof(double);
    size_t hRow = (size_t)MAX_HYD_VARS * sizeof(double);

    *didReorder = 0;
    if (nSeg <= 1 || !(tstep > 0.0))
    {
        bucketOffset[0] = 0;
        bucketCount[0] = nSeg;
        *bucketMaxSize = nSeg;
        return 0;
    }
    for (b = 0; b < RK5_FAST_BUCKETS; b++)
    {
        bucketOffset[b] = 0;
        bucketCount[b] = 0;
    }
    *bucketMaxSize = 0;
    bucket = (int*)calloc(nSeg, sizeof(int));
    count = (int*)calloc(RK5_FAST_BUCKETS, sizeof(int));
    offset = (int*)calloc(RK5_FAST_BUCKETS, sizeof(int));
    cursor = (int*)calloc(RK5_FAST_BUCKETS, sizeof(int));
    if (!bucket || !count || !offset || !cursor)
    {
        free(bucket); free(count); free(offset); free(cursor);
        return ERR_MEMORY;
    }
    for (sid = 0; sid < nSeg; sid++)
    {
        b = rk5FastBucket(hstep[sid], tstep);
        bucket[sid] = b;
        count[b]++;
    }
    for (b = 1; b < RK5_FAST_BUCKETS; b++) offset[b] = offset[b - 1] + count[b - 1];
    for (b = 0; b < RK5_FAST_BUCKETS; b++)
    {
        bucketOffset[b] = offset[b];
        bucketCount[b] = count[b];
        if (count[b] > *bucketMaxSize) *bucketMaxSize = count[b];
    }
    if (*bucketMaxSize >= (int)(0.80 * (double)nSeg))
    {
        free(bucket); free(count); free(offset); free(cursor);
        return 0;
    }
    order = (int*)calloc(nSeg, sizeof(int));
    tmpSegPipe = (int*)calloc(nSeg, sizeof(int));
    tmpSegVol = (double*)calloc(nSeg, sizeof(double));
    tmpHstep = (double*)calloc(nSeg, sizeof(double));
    tmpSegPtrs = (Pseg*)calloc(nSeg, sizeof(Pseg));
    tmpC = (double*)calloc((size_t)nSeg * (nSpecies + 1), sizeof(double));
    tmpCOde = (double*)calloc((size_t)nSeg * (nSpecies + 1), sizeof(double));
    tmpHyd = (double*)calloc((size_t)nSeg * MAX_HYD_VARS, sizeof(double));
    if (!order || !tmpSegPipe || !tmpSegVol || !tmpHstep || !tmpSegPtrs ||
        !tmpC || !tmpCOde || !tmpHyd)
    {
        free(bucket); free(count); free(offset); free(cursor); free(order);
        free(tmpSegPipe); free(tmpSegVol); free(tmpHstep); free(tmpSegPtrs);
        free(tmpC); free(tmpCOde); free(tmpHyd);
        return ERR_MEMORY;
    }
    memcpy(cursor, offset, (size_t)RK5_FAST_BUCKETS * sizeof(int));
    for (sid = 0; sid < nSeg; sid++) order[cursor[bucket[sid]]++] = sid;
    for (sid = 0; sid < nSeg; sid++)
    {
        int oldSid = order[sid];
        if (unpackOrder) unpackOrder[oldSid] = sid;
        tmpSegPipe[sid] = segPipe[oldSid];
        tmpSegVol[sid] = segVol[oldSid];
        tmpHstep[sid] = hstep[oldSid];
        tmpSegPtrs[sid] = segPtrs[oldSid];
        memcpy(&tmpC[(size_t)sid * (nSpecies + 1)], &c[(size_t)oldSid * (nSpecies + 1)], cRow);
        memcpy(&tmpCOde[(size_t)sid * (nSpecies + 1)], &cOde[(size_t)oldSid * (nSpecies + 1)], cRow);
        memcpy(&tmpHyd[(size_t)sid * MAX_HYD_VARS], &hyd[(size_t)oldSid * MAX_HYD_VARS], hRow);
    }
    memcpy(segPipe, tmpSegPipe, (size_t)nSeg * sizeof(int));
    memcpy(segVol, tmpSegVol, (size_t)nSeg * sizeof(double));
    memcpy(hstep, tmpHstep, (size_t)nSeg * sizeof(double));
    memcpy(segPtrs, tmpSegPtrs, (size_t)nSeg * sizeof(Pseg));
    memcpy(c, tmpC, (size_t)nSeg * (nSpecies + 1) * sizeof(double));
    memcpy(cOde, tmpCOde, (size_t)nSeg * (nSpecies + 1) * sizeof(double));
    memcpy(hyd, tmpHyd, (size_t)nSeg * MAX_HYD_VARS * sizeof(double));
    *didReorder = 1;
    free(bucket); free(count); free(offset); free(cursor); free(order);
    free(tmpSegPipe); free(tmpSegVol); free(tmpHstep); free(tmpSegPtrs);
    free(tmpC); free(tmpCOde); free(tmpHyd);
    return 0;
}

int MSXgpu_reactPipeSegments(double dt)
{
    int err = 0;
    int nLinks = MSX.Nobjects[LINK];
    int nSpecies = MSX.Nobjects[SPECIES];
    int nParams = MSX.Nobjects[PARAMETER];
    int nConsts = MSX.Nobjects[CONSTANT];
    int nSeg = 0, sid = 0;
    int rateCount = 0, eqCount = 0, formulaCount = 0;
    int rk5BucketOffset[RK5_FAST_BUCKETS] = {0};
    int rk5BucketCount[RK5_FAST_BUCKETS] = {0};
    int rk5BucketMaxSize = 0;
    int rk5DidReorder = 0;
    int *rateSpecies = NULL, *eqSpecies = NULL, *formulaSpecies = NULL, *speciesType = NULL;
    double *params = NULL, *consts = NULL, *linkDiam = NULL;
    double *rateAtol = NULL, *rateRtol = NULL;
    GpuInstrHost *instr = NULL;
    GpuProgramHost *speciesProg = NULL, *termProg = NULL;
    int nInstr = 0;
    double timer = 0.0;
    int k, m;
    int timing = MSXgpu_profileStageEnabled();
    int detail = MSXgpu_profileDetailGroupEnabled(MSX_PROFILE_DETAIL_CHEM);
    CUdeviceptr d_segPipe = 0, d_segRow = 0, d_segVol = 0, d_hstep = 0, d_rateAtol = 0, d_rateRtol = 0, d_rateSpecies = 0, d_eqSpecies = 0, d_formulaSpecies = 0, d_speciesType = 0;
    CUdeviceptr d_activeLink = 0, d_pipeSegOffset = 0, d_pipeSegCount = 0;
    CUdeviceptr d_c = 0, d_cOde = 0, d_hyd = 0, d_params = 0, d_consts = 0, d_linkDiam = 0, d_reacted = 0;
    CUdeviceptr d_instr = 0, d_speciesProg = 0, d_termProg = 0, d_err = 0;
    CUdeviceptr d_rk5Nfcn = 0, d_rk5Naccpt = 0, d_rk5Nrejct = 0, d_rk5LastHstep = 0, d_rk5Err = 0;
    CUdeviceptr d_ros2Nfcn = 0, d_ros2Njac = 0, d_ros2Naccept = 0, d_ros2Nreject = 0, d_ros2LastHstep = 0, d_ros2Err = 0;
    CUevent evStart[3] = { NULL, NULL, NULL }, evStop[3] = { NULL, NULL, NULL };
    GpuErrorHost gpuErr;
    MSXReactTransferView transfer;

    MSX.GpuTimingRecord.react_gpu = 1;
    MSX.GpuTimingRecord.ode_gpu = (MSX.GpuOde != 0);
    MSX.GpuTimingRecord.equil_gpu = (MSX.GpuEquil != 0);
    MSX.GpuTimingRecord.formula_gpu = (MSX.GpuFormula != 0);

    err = validateModelForGpu();
    if (err) return err;
    err = ensureModule();
    if (err) return err;
    err = checkCu(cuCtxSetCurrent(GpuContext), ERR_GPU_NOT_ENABLED);
    if (err) return err;
    err = ensureReactTransfer();
    if (err) return err;

    if (timing) timer = MSXgpu_wallTimeMs();
    for (m = 1; m <= nSpecies; m++)
    {
        if (MSX.Species[m].pipeExprType == RATE) rateCount++;
        else if (MSX.Species[m].pipeExprType == EQUIL) eqCount++;
        else if (MSX.Species[m].pipeExprType == FORMULA) formulaCount++;
    }

    params = (double*)calloc((nLinks + 1) * (nParams + 1), sizeof(double));
    consts = (double*)calloc(nConsts + 1, sizeof(double));
    linkDiam = (double*)calloc(nLinks + 1, sizeof(double));
    rateSpecies = (int*)calloc(rateCount + 1, sizeof(int));
    rateAtol = (double*)calloc(rateCount + 1, sizeof(double));
    rateRtol = (double*)calloc(rateCount + 1, sizeof(double));
    eqSpecies = (int*)calloc(eqCount + 1, sizeof(int));
    formulaSpecies = (int*)calloc(formulaCount + 1, sizeof(int));
    speciesType = (int*)calloc(nSpecies + 1, sizeof(int));
    if (!params || !consts || !linkDiam ||
        !rateSpecies || !rateAtol || !rateRtol || !eqSpecies || !formulaSpecies || !speciesType)
    {
        err = ERR_MEMORY;
        goto cleanup;
    }

    rateCount = eqCount = formulaCount = 0;
    for (m = 1; m <= nSpecies; m++)
    {
        speciesType[m] = MSX.Species[m].type;
        if (MSX.Species[m].pipeExprType == RATE)
        {
            rateSpecies[++rateCount] = m;
            rateAtol[rateCount] = MSX.Species[m].aTol;
            rateRtol[rateCount] = MSX.Species[m].rTol;
        }
        else if (MSX.Species[m].pipeExprType == EQUIL) eqSpecies[++eqCount] = m;
        else if (MSX.Species[m].pipeExprType == FORMULA) formulaSpecies[++formulaCount] = m;
    }
    for (m = 1; m <= nConsts; m++) consts[m] = MSX.Const[m].value;
    for (k = 1; k <= nLinks; k++)
    {
        linkDiam[k] = MSX.Link[k].diam;
        for (m = 1; m <= nParams; m++) params[k * (nParams + 1) + m] = MSX.Link[k].param[m];
    }
    if (MSXsegStorage_isPipeRingEnabled())
        err = MSXreactTransfer_ringView(dt, nSpecies, &transfer);
    else
        err = MSXreactTransfer_countAndPack(dt, nSpecies, &transfer);
    if (err)
    {
        if (err == ERR_GPU_SEGMENT_PACK_FAILED)
            setGpuError(err, GPU_STAGE_NONE, -1, MSX.GpuError.pipe, -1, -1, -1, MSX.GpuError.value);
        goto cleanup;
    }
    nSeg = transfer.nSeg;
    if (!transfer.ringResident && MSX.GpuSolver == RK5 && MSX.GpuRk5Mode == GPU_RK5_FAST_BUCKET)
    {
        double reorderStart = detail ? MSXgpu_wallTimeMs() : 0.0;
        err = reorderSegmentsForRk5Fast(nSeg, nSpecies, dt / MSX.Ucf[RATE_UNITS],
                                        transfer.segPipe, transfer.segVol, transfer.hstep, transfer.segPtrs,
                                        transfer.unpackOrder, transfer.c, transfer.cOde, transfer.hyd,
                                        rk5BucketOffset, rk5BucketCount, &rk5BucketMaxSize,
                                        &rk5DidReorder);
        if (detail)
        {
            MSX.GpuTimingRecord.rk5_fast_mode = 1;
            MSX.GpuTimingRecord.rk5_bucket_count += RK5_FAST_BUCKETS;
            MSX.GpuTimingRecord.rk5_bucket_max_size =
                (double)MAX((int)MSX.GpuTimingRecord.rk5_bucket_max_size, rk5BucketMaxSize);
        }
        if (detail && rk5DidReorder)
            MSX.GpuTimingRecord.rk5_bucket_reorder_ms += MSXgpu_wallTimeMs() - reorderStart;
        if (err) goto cleanup;
    }
    err = buildPrograms(&instr, &nInstr, &speciesProg, &termProg);
    if (err) goto cleanup;
    if (timing) MSX.GpuTimingRecord.react_pack_ms += MSXgpu_wallTimeMs() - timer;

    memset(&gpuErr, 0, sizeof(gpuErr));
    err = MSXreactTransfer_upload(&transfer, &gpuErr, sizeof(gpuErr));
    if (err) goto cleanup;
    d_segPipe = (CUdeviceptr)transfer.d_segPipe;
    d_segRow = (CUdeviceptr)transfer.d_segRow;
    d_activeLink = (CUdeviceptr)transfer.d_activeLink;
    d_pipeSegOffset = (CUdeviceptr)transfer.d_pipeSegOffset;
    d_pipeSegCount = (CUdeviceptr)transfer.d_pipeSegCount;
    d_segVol = (CUdeviceptr)transfer.d_segVol;
    d_hstep = (CUdeviceptr)transfer.d_hstep;
    d_c = (CUdeviceptr)transfer.d_c;
    d_cOde = (CUdeviceptr)transfer.d_cOde;
    d_hyd = (CUdeviceptr)transfer.d_hyd;
    d_reacted = (CUdeviceptr)transfer.d_reacted;
    d_rk5Nfcn = (CUdeviceptr)transfer.d_rk5Nfcn;
    d_rk5Naccpt = (CUdeviceptr)transfer.d_rk5Naccpt;
    d_rk5Nrejct = (CUdeviceptr)transfer.d_rk5Nrejct;
    d_rk5Err = (CUdeviceptr)transfer.d_rk5Err;
    d_rk5LastHstep = (CUdeviceptr)transfer.d_rk5LastHstep;
    d_ros2Nfcn = (CUdeviceptr)transfer.d_ros2Nfcn;
    d_ros2Njac = (CUdeviceptr)transfer.d_ros2Njac;
    d_ros2Naccept = (CUdeviceptr)transfer.d_ros2Naccept;
    d_ros2Nreject = (CUdeviceptr)transfer.d_ros2Nreject;
    d_ros2Err = (CUdeviceptr)transfer.d_ros2Err;
    d_ros2LastHstep = (CUdeviceptr)transfer.d_ros2LastHstep;
    d_err = (CUdeviceptr)transfer.d_err;

    if (timing) timer = MSXgpu_wallTimeMs();
#define GPU_ALLOC_COPY(ptr, host, bytes) do { err = checkCu(cuMemAlloc(&(ptr), (bytes)), ERR_GPU_MEMORY_ALLOCATION_FAILED); if (err) goto cleanup; err = checkCu(profileCuMemcpyHtoD((ptr), (host), (bytes), MSX_PROFILE_TRANSFER_SCOPE_OTHER), ERR_GPU_MEMORY_ALLOCATION_FAILED); if (err) goto cleanup; } while (0)
    GPU_ALLOC_COPY(d_rateAtol, rateAtol, (rateCount + 1) * sizeof(double));
    GPU_ALLOC_COPY(d_rateRtol, rateRtol, (rateCount + 1) * sizeof(double));
    GPU_ALLOC_COPY(d_rateSpecies, rateSpecies, (rateCount + 1) * sizeof(int));
    GPU_ALLOC_COPY(d_eqSpecies, eqSpecies, (eqCount + 1) * sizeof(int));
    GPU_ALLOC_COPY(d_formulaSpecies, formulaSpecies, (formulaCount + 1) * sizeof(int));
    GPU_ALLOC_COPY(d_speciesType, speciesType, (nSpecies + 1) * sizeof(int));
    GPU_ALLOC_COPY(d_params, params, (nLinks + 1) * (nParams + 1) * sizeof(double));
    GPU_ALLOC_COPY(d_consts, consts, (nConsts + 1) * sizeof(double));
    GPU_ALLOC_COPY(d_linkDiam, linkDiam, (nLinks + 1) * sizeof(double));
    GPU_ALLOC_COPY(d_instr, instr, (nInstr > 0 ? nInstr : 1) * sizeof(GpuInstrHost));
    GPU_ALLOC_COPY(d_speciesProg, speciesProg, (nSpecies + 1) * sizeof(GpuProgramHost));
    GPU_ALLOC_COPY(d_termProg, termProg, (MSX.Nobjects[TERM] + 1) * sizeof(GpuProgramHost));
#undef GPU_ALLOC_COPY
    if (timing) MSX.GpuTimingRecord.h2d_ms += MSXgpu_wallTimeMs() - timer;

    if (nSeg > 0)
    {
        int block = 128;
        int grid = (nSeg + block - 1) / block;
        int nActiveLinks = transfer.nActiveLinks;
        int warpsPerBlock = 4;
        int warpBlock = warpsPerBlock * 32;
        int warpGrid = (nActiveLinks + warpsPerBlock - 1) / warpsPerBlock;
        int usePipeWarp = !transfer.ringResident && MSX.GpuSolver == RK5 &&
                          MSX.GpuRk5Mode == GPU_RK5_CPU_ALIGN && nActiveLinks > 0 &&
                          gpuRk5PipeWarpEnabled();
        double tstep = dt / MSX.Ucf[RATE_UNITS];
        double areaUcf = MSX.Ucf[AREA_UNITS];
        double lperFt3 = LperFT3;
        int lastSpecies = lastIndex(SPECIES);
        int lastTerm = lastIndex(TERM);
        int lastParam = lastIndex(PARAMETER);
        int lastConst = lastIndex(CONSTANT);
        int hStride = MAX_HYD_VARS;
        int hydLayout = MSX_RESIDENT_HYD_ACTIVE_MAJOR;
        int paramStride = nParams + 1;
        void *odeArgs[] = { &nSeg, &nSpecies, &rateCount, &tstep, &d_segPipe, &d_segRow, &d_segVol,
            &d_rateSpecies, &d_speciesType, &d_linkDiam, &areaUcf, &lperFt3, &d_c, &d_cOde, &d_reacted,
            &d_params, &paramStride, &d_consts, &d_hyd, &hStride, &hydLayout, &d_speciesProg, &d_termProg,
            &d_instr, &lastSpecies, &lastTerm, &lastParam, &lastConst, &d_err };
        void *rk5AlignArgs[] = { &nSeg, &nSpecies, &rateCount, &tstep, &d_segPipe, &d_segRow, &d_segVol, &d_hstep,
            &d_rateAtol, &d_rateRtol, &d_rateSpecies, &d_speciesType, &d_linkDiam, &areaUcf, &lperFt3,
            &d_c, &d_cOde, &d_reacted, &d_params, &paramStride, &d_consts, &d_hyd, &hStride, &hydLayout,
            &d_speciesProg, &d_termProg, &d_instr, &lastSpecies, &lastTerm, &lastParam, &lastConst,
            &d_rk5Nfcn, &d_rk5Naccpt, &d_rk5Nrejct, &d_rk5LastHstep, &d_rk5Err, &d_err };
        void *rk5PipeWarpArgs[] = { &nActiveLinks, &d_activeLink, &d_pipeSegOffset, &d_pipeSegCount,
            &nSeg, &nSpecies, &rateCount, &tstep, &d_segPipe, &d_segRow, &d_segVol, &d_hstep,
            &d_rateAtol, &d_rateRtol, &d_rateSpecies, &d_speciesType, &d_linkDiam, &areaUcf, &lperFt3,
            &d_c, &d_cOde, &d_reacted, &d_params, &paramStride, &d_consts, &d_hyd, &hStride, &hydLayout,
            &d_speciesProg, &d_termProg, &d_instr, &lastSpecies, &lastTerm, &lastParam, &lastConst,
            &d_rk5Nfcn, &d_rk5Naccpt, &d_rk5Nrejct, &d_rk5LastHstep, &d_rk5Err, &d_err };
        void *ros2Args[] = { &nSeg, &nSpecies, &rateCount, &tstep, &d_segPipe, &d_segRow, &d_segVol, &d_hstep,
            &d_rateAtol, &d_rateRtol, &d_rateSpecies, &d_speciesType, &d_linkDiam, &areaUcf, &lperFt3,
            &d_c, &d_cOde, &d_reacted, &d_params, &paramStride, &d_consts, &d_hyd, &hStride, &hydLayout,
            &d_speciesProg, &d_termProg, &d_instr, &lastSpecies, &lastTerm, &lastParam, &lastConst,
            &d_ros2Nfcn, &d_ros2Njac, &d_ros2Naccept, &d_ros2Nreject, &d_ros2LastHstep, &d_ros2Err, &d_err };
        void *equilArgs[] = { &nSeg, &nSpecies, &eqCount, &d_segPipe, &d_segRow, &d_eqSpecies, &d_c, &d_params,
            &paramStride, &d_consts, &d_hyd, &hStride, &hydLayout, &d_speciesProg, &d_termProg,
            &d_instr, &lastSpecies, &lastTerm, &lastParam, &lastConst, &d_err };
        void *formulaArgs[] = { &nSeg, &nSpecies, &formulaCount, &d_segPipe, &d_segRow, &d_formulaSpecies, &d_c,
            &d_params, &paramStride, &d_consts, &d_hyd, &hStride, &hydLayout, &d_speciesProg, &d_termProg,
            &d_instr, &lastSpecies, &lastTerm, &lastParam, &lastConst, &d_err };

        if (detail)
        {
            int i;
            for (i = 0; i < 3; i++)
            {
                cuEventCreate(&evStart[i], CU_EVENT_DEFAULT);
                cuEventCreate(&evStop[i], CU_EVENT_DEFAULT);
            }
            cuEventRecord(evStart[0], 0);
        }
        if (!transfer.ringResident && MSX.GpuSolver == RK5 && MSX.GpuRk5Mode == GPU_RK5_FAST_BUCKET &&
            rk5BucketMaxSize < (int)(0.80 * (double)nSeg))
        {
            int b;
            for (b = 0; b < RK5_FAST_BUCKETS; b++)
            {
                int start = rk5BucketOffset[b];
                int count = rk5BucketCount[b];
                int bucketGrid;
                CUdeviceptr b_segPipe, b_segVol, b_hstep, b_c, b_cOde, b_hyd;
                CUdeviceptr b_rk5Nfcn, b_rk5Naccpt, b_rk5Nrejct, b_rk5LastHstep, b_rk5Err;
                CUdeviceptr b_segRow;
                void *rk5BucketArgs[37];
                if (count <= 0) continue;
                bucketGrid = (count + block - 1) / block;
                b_segPipe = d_segPipe + (size_t)start * sizeof(int);
                b_segRow = d_segRow + (size_t)start * sizeof(int);
                b_segVol = d_segVol + (size_t)start * sizeof(double);
                b_hstep = d_hstep + (size_t)start * sizeof(double);
                b_c = d_c + (size_t)start * (nSpecies + 1) * sizeof(double);
                b_cOde = d_cOde + (size_t)start * (nSpecies + 1) * sizeof(double);
                b_hyd = d_hyd + (size_t)start * MAX_HYD_VARS * sizeof(double);
                b_rk5Nfcn = d_rk5Nfcn + (size_t)start * sizeof(int);
                b_rk5Naccpt = d_rk5Naccpt + (size_t)start * sizeof(int);
                b_rk5Nrejct = d_rk5Nrejct + (size_t)start * sizeof(int);
                b_rk5LastHstep = d_rk5LastHstep + (size_t)start * sizeof(double);
                b_rk5Err = d_rk5Err + (size_t)start * sizeof(int);
                rk5BucketArgs[0] = &count;
                rk5BucketArgs[1] = &nSpecies;
                rk5BucketArgs[2] = &rateCount;
                rk5BucketArgs[3] = &tstep;
                rk5BucketArgs[4] = &b_segPipe;
                rk5BucketArgs[5] = &b_segRow;
                rk5BucketArgs[6] = &b_segVol;
                rk5BucketArgs[7] = &b_hstep;
                rk5BucketArgs[8] = &d_rateAtol;
                rk5BucketArgs[9] = &d_rateRtol;
                rk5BucketArgs[10] = &d_rateSpecies;
                rk5BucketArgs[11] = &d_speciesType;
                rk5BucketArgs[12] = &d_linkDiam;
                rk5BucketArgs[13] = &areaUcf;
                rk5BucketArgs[14] = &lperFt3;
                rk5BucketArgs[15] = &b_c;
                rk5BucketArgs[16] = &b_cOde;
                rk5BucketArgs[17] = &d_reacted;
                rk5BucketArgs[18] = &d_params;
                rk5BucketArgs[19] = &paramStride;
                rk5BucketArgs[20] = &d_consts;
                rk5BucketArgs[21] = &b_hyd;
                rk5BucketArgs[22] = &hStride;
                rk5BucketArgs[23] = &hydLayout;
                rk5BucketArgs[24] = &d_speciesProg;
                rk5BucketArgs[25] = &d_termProg;
                rk5BucketArgs[26] = &d_instr;
                rk5BucketArgs[27] = &lastSpecies;
                rk5BucketArgs[28] = &lastTerm;
                rk5BucketArgs[29] = &lastParam;
                rk5BucketArgs[30] = &lastConst;
                rk5BucketArgs[31] = &b_rk5Nfcn;
                rk5BucketArgs[32] = &b_rk5Naccpt;
                rk5BucketArgs[33] = &b_rk5Nrejct;
                rk5BucketArgs[34] = &b_rk5LastHstep;
                rk5BucketArgs[35] = &b_rk5Err;
                rk5BucketArgs[36] = &d_err;
                err = checkCu(cuLaunchKernel(GpuModule.rk5Kernel, bucketGrid, 1, 1, block, 1, 1, 0, 0,
                                             rk5BucketArgs, NULL),
                              ERR_GPU_KERNEL_LAUNCH_FAILED);
                if (err) goto cleanup;
                if (detail) MSX.GpuTimingRecord.rk5_bucket_launches += 1.0;
            }
        }
        else if (usePipeWarp)
        {
            err = checkCu(cuLaunchKernel(GpuModule.rk5PipeWarpKernel,
                                         warpGrid, 1, 1, warpBlock, 1, 1, 0, 0,
                                         rk5PipeWarpArgs, NULL),
                          ERR_GPU_KERNEL_LAUNCH_FAILED);
            if (err) goto cleanup;
        }
        else if (MSX.GpuSolver == ROS2)
        {
            err = checkCu(cuLaunchKernel(GpuModule.ros2Kernel, grid, 1, 1, block, 1, 1, 0, 0,
                                         ros2Args, NULL),
                          ERR_GPU_KERNEL_LAUNCH_FAILED);
            if (err) goto cleanup;
        }
        else
        {
            err = checkCu(cuLaunchKernel((MSX.GpuSolver == RK5) ? GpuModule.rk5Kernel : GpuModule.odeKernel,
                                         grid, 1, 1, block, 1, 1, 0, 0,
                                         (MSX.GpuSolver == RK5) ? rk5AlignArgs : odeArgs, NULL),
                          ERR_GPU_KERNEL_LAUNCH_FAILED);
            if (err) goto cleanup;
            if (MSX.GpuSolver == RK5 && MSX.GpuRk5Mode == GPU_RK5_FAST_BUCKET)
                if (detail) MSX.GpuTimingRecord.rk5_bucket_launches += 1.0;
        }
        if (detail)
        {
            cuEventRecord(evStop[0], 0);
        }

        if (eqCount > 0)
        {
            if (detail) cuEventRecord(evStart[1], 0);
            err = checkCu(cuLaunchKernel(GpuModule.equilKernel, grid, 1, 1, block, 1, 1, 0, 0, equilArgs, NULL),
                          ERR_GPU_KERNEL_LAUNCH_FAILED);
            if (err) goto cleanup;
            if (detail)
            {
                cuEventRecord(evStop[1], 0);
            }
        }

        if (formulaCount > 0)
        {
            if (detail) cuEventRecord(evStart[2], 0);
            err = checkCu(cuLaunchKernel(GpuModule.formulaKernel, grid, 1, 1, block, 1, 1, 0, 0, formulaArgs, NULL),
                          ERR_GPU_KERNEL_LAUNCH_FAILED);
            if (err) goto cleanup;
            if (detail)
            {
                cuEventRecord(evStop[2], 0);
            }
        }
        err = checkCu(cuCtxSynchronize(), ERR_GPU_KERNEL_RUNTIME_ERROR);
        if (err) goto cleanup;
        if (detail)
        {
            MSX.GpuTimingRecord.react_ode_ms += eventReadElapsedMs(evStart[0], evStop[0]);
            if (eqCount > 0)
                MSX.GpuTimingRecord.react_equil_ms += eventReadElapsedMs(evStart[1], evStop[1]);
            if (formulaCount > 0)
                MSX.GpuTimingRecord.react_formula_ms += eventReadElapsedMs(evStart[2], evStop[2]);
        }
    }

    err = MSXreactTransfer_download(&transfer, &gpuErr, sizeof(gpuErr));
    if (err) goto cleanup;

    if (MSX.GpuSolver == RK5 && detail)
    {
        for (sid = 0; sid < nSeg; sid++)
        {
            MSX.GpuTimingRecord.rk5_nfcn += transfer.rk5Nfcn[sid];
            MSX.GpuTimingRecord.rk5_naccpt += transfer.rk5Naccpt[sid];
            MSX.GpuTimingRecord.rk5_nrejct += transfer.rk5Nrejct[sid];
            if (transfer.rk5LastHstep[sid] != 0.0) MSX.GpuTimingRecord.rk5_last_hstep = transfer.rk5LastHstep[sid];
            if (!MSX.GpuTimingRecord.rk5_error_code && transfer.rk5Err[sid]) MSX.GpuTimingRecord.rk5_error_code = transfer.rk5Err[sid];
        }
    }

    if (MSX.GpuSolver == ROS2)
    {
        for (sid = 0; sid < nSeg; sid++)
        {
            if (detail)
            {
                MSX.GpuTimingRecord.ros2_nfcn += transfer.ros2Nfcn[sid];
                MSX.GpuTimingRecord.ros2_njac += transfer.ros2Njac[sid];
                MSX.GpuTimingRecord.ros2_naccept += transfer.ros2Naccept[sid];
                MSX.GpuTimingRecord.ros2_nreject += transfer.ros2Nreject[sid];
                if (transfer.ros2LastHstep[sid] != 0.0)
                    MSX.GpuTimingRecord.ros2_last_hstep = transfer.ros2LastHstep[sid];
            }
            if (transfer.ros2Err[sid])
            {
                if (!MSX.GpuTimingRecord.ros2_error_code)
                    MSX.GpuTimingRecord.ros2_error_code = transfer.ros2Err[sid];
                if (!Ros2RawErrorReason)
                {
                    Ros2RawErrorReason = transfer.ros2Err[sid];
                    Ros2RawErrorSid = sid;
                }
            }
        }
    }

    if (gpuErr.code)
    {
        writeGpuErrorDetail(&gpuErr, nSpecies, transfer.c);
        setGpuError(gpuErr.code, gpuErr.stage, gpuErr.sid, gpuErr.pipe, gpuErr.species,
                    gpuErr.expr, gpuErr.iter, gpuErr.value);
        err = gpuErr.code;
        goto cleanup;
    }

    err = MSXreactTransfer_unpack(&transfer);
    if (err) goto cleanup;

cleanup:
    {
        int i;
        for (i = 0; i < 3; i++)
        {
            if (evStart[i]) cuEventDestroy(evStart[i]);
            if (evStop[i]) cuEventDestroy(evStop[i]);
        }
    }
    if (d_rateAtol) cuMemFree(d_rateAtol);
    if (d_rateRtol) cuMemFree(d_rateRtol);
    if (d_rateSpecies) cuMemFree(d_rateSpecies);
    if (d_eqSpecies) cuMemFree(d_eqSpecies);
    if (d_formulaSpecies) cuMemFree(d_formulaSpecies);
    if (d_speciesType) cuMemFree(d_speciesType);
    if (d_params) cuMemFree(d_params);
    if (d_consts) cuMemFree(d_consts);
    if (d_linkDiam) cuMemFree(d_linkDiam);
    if (d_instr) cuMemFree(d_instr);
    if (d_speciesProg) cuMemFree(d_speciesProg);
    if (d_termProg) cuMemFree(d_termProg);
    free(params);
    free(consts);
    free(linkDiam);
    free(rateSpecies);
    free(rateAtol);
    free(rateRtol);
    free(eqSpecies);
    free(formulaSpecies);
    free(speciesType);
    free(instr);
    free(speciesProg);
    free(termProg);
    if (err) setGpuError(err, GPU_STAGE_NONE, -1, -1, -1, -1, -1, 0.0);
    return err;
}
#endif
