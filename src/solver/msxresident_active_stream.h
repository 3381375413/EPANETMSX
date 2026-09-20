#ifndef MSXRESIDENT_ACTIVE_STREAM_H
#define MSXRESIDENT_ACTIVE_STREAM_H

/* Shared C1 active-row lifecycle.  The production Resident runtime supplies
   callbacks backed by MSXresident's allocation-free iterator and Hybrid
   identity map.  The Phase2b contract harness supplies fault-injectable
   callbacks to exercise the same begin/chunk/append/abort/seal path. */
#include "msxresident_core_cuda.h"
#include <string.h>

#define MSX_RESIDENT_ACTIVE_CHUNK_ROWS 256u

typedef struct {
    void *context;
    MSXResidentStatus (*begin)(void *);
    MSXResidentStatus (*validate)(void *);
    MSXResidentStatus (*count)(void *, uint32_t *);
    MSXResidentStatus (*next)(void *, MSXResidentActiveRow *);
    int (*identity)(void *, const MSXResidentActiveRow *);
    int (*audit)(void *, const MSXResidentActiveRow *);
    uint64_t (*topology)(void *);
    void (*validationEnd)(void *);
    void (*streamBegin)(void *);
    void (*streamEnd)(void *);
} MSXResidentActiveStreamSource;

typedef struct {
    uint32_t expectedCount;
    uint64_t iteratorPasses;
    uint64_t rowsAppended;
    uint64_t builderAborts;
    uint64_t chunkHighWater;
    uint64_t hydMismatches;
} MSXResidentActiveStreamReport;

static MSXResidentStatus MSXresident_activeStreamBuild(
    MSXResidentGpu *gpu, MSXResidentGpuActiveWriter *writer,
    uint64_t topologyVersion, MSXResidentActiveStreamSource *source,
    const MSXResidentHydView *hyd, uint32_t *activeOut,
    MSXResidentActiveStreamReport *report)
{
    MSXResidentActiveRow chunk[MSX_RESIDENT_ACTIVE_CHUNK_ROWS];
    MSXResidentStatus s, deferred = MSX_RESIDENT_OK;
    MSXResidentStatus iterError = MSX_RESIDENT_OK;
    uint32_t n = 0, chunkCount, i;
    int done = 0;

    if (!gpu || !writer || !source || !source->begin || !source->validate ||
        !source->count || !source->next || !source->identity || !activeOut)
        return MSX_RESIDENT_ERR_ARGUMENT;
    *activeOut = 0;
    if (report) memset(report, 0, sizeof(*report));

    s = source->begin(source->context);
    if (s == MSX_RESIDENT_OK) s = source->validate(source->context);
    if (report) ++report->iteratorPasses;
    if (s == MSX_RESIDENT_OK) s = source->count(source->context, &n);
    if (report) report->expectedCount = n;
    if (source->validationEnd) source->validationEnd(source->context);
    if (s != MSX_RESIDENT_OK) return s;

    s = source->begin(source->context);
    if (s != MSX_RESIDENT_OK) return s;
    if (report) ++report->iteratorPasses;
    s = MSXresidentGpu_beginActive(gpu, n, topologyVersion, writer);
    if (s != MSX_RESIDENT_OK) return s;
    if (source->streamBegin) source->streamBegin(source->context);

    while (!done)
    {
        chunkCount = 0;
        while (chunkCount < MSX_RESIDENT_ACTIVE_CHUNK_ROWS)
        {
            s = source->next(source->context, &chunk[chunkCount]);
            if (s == MSX_RESIDENT_OK) { ++chunkCount; continue; }
            if (s == MSX_RESIDENT_ITER_END) { done = 1; break; }
            iterError = s;
            done = 1;
            break;
        }
        if (report && chunkCount > report->chunkHighWater)
            report->chunkHighWater = chunkCount;
        if (iterError != MSX_RESIDENT_OK)
        {
            if (source->streamEnd) source->streamEnd(source->context);
            /* A prior identity/audit abort or append failure owns/reset the
               writer already.  Do not issue a second abort when a later
               iterator error wins the error-priority rule. */
            if (writer->owner == gpu && writer->building)
            {
                if (report) ++report->builderAborts;
                (void)MSXresidentGpu_abortActiveBuild(gpu, writer);
            }
            return iterError;
        }
        for (i = 0; i < chunkCount; ++i)
        {
            MSXResidentActiveRow *row = &chunk[i];
            if (deferred == MSX_RESIDENT_OK)
            {
                if (!source->identity(source->context, row))
                {
                    deferred = MSX_RESIDENT_ERR_GENERATION;
                    if (report) ++report->builderAborts;
                    (void)MSXresidentGpu_abortActiveBuild(gpu, writer);
                }
                else
                {
                    s = MSXresidentGpu_appendActive(gpu, writer, row);
                    if (s != MSX_RESIDENT_OK)
                    {
                        /* appendActive owns/reset its writer on failure. */
                        deferred = s;
                        if (report) ++report->builderAborts;
                    }
                    else if (report)
                        ++report->rowsAppended;
                }
            }
            /* Audit is diagnostic coverage.  Continue calling it after the
               first failure so the mismatch count covers the full stream. */
            if (source->audit && source->audit(source->context, row))
            {
                if (report) ++report->hydMismatches;
                if (deferred == MSX_RESIDENT_OK)
                {
                    deferred = MSX_RESIDENT_ERR_ARGUMENT;
                    if (report) ++report->builderAborts;
                    (void)MSXresidentGpu_abortActiveBuild(gpu, writer);
                }
            }
        }
    }
    if (source->streamEnd) source->streamEnd(source->context);
    if (deferred != MSX_RESIDENT_OK) return deferred;
    if (source->topology && source->topology(source->context) != topologyVersion)
    {
        if (report) ++report->builderAborts;
        (void)MSXresidentGpu_abortActiveBuild(gpu, writer);
        return MSX_RESIDENT_ERR_GENERATION;
    }
    s = MSXresidentGpu_sealActive(gpu, writer, topologyVersion);
    if (s != MSX_RESIDENT_OK)
    {
        if (report) ++report->builderAborts;
        return s;
    }
    *activeOut = n;
    (void)hyd;
    return MSX_RESIDENT_OK;
}

#endif
