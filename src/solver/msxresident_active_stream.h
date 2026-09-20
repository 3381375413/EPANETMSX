#ifndef MSXRESIDENT_ACTIVE_STREAM_H
#define MSXRESIDENT_ACTIVE_STREAM_H

/* Production runtime and CUDA contract harness include this same state
   machine. Production adapters below expand to direct core/GPU APIs; only
   the harness build supplies callbacks for fault injection. */
#include "msxresident_core_cuda.h"
#include <string.h>

#define MSX_RESIDENT_ACTIVE_CHUNK_ROWS 256u

typedef struct {
    uint64_t rawCountPasses, rawLinks, rawExpectedRows;
    uint64_t iteratorPasses, iteratorChunkCalls;
    uint64_t batchCalls, batchRowsAttempted;
    uint64_t rowsAppended, builderAborts, chunkHighWater;
    uint64_t hydMismatches;
    uint64_t beginFailures, appendFailures, iteratorFailures, sealFailures;
    uint32_t expectedCount;
} MSXResidentActiveStreamReport;

#if defined(MSX_RESIDENT_ACTIVE_STREAM_TEST)

typedef struct {
    void *context;
    MSXResidentActiveIterator iterator;
    MSXResidentStatus (*begin)(void *);
    MSXResidentStatus (*validate)(void *);
    MSXResidentStatus (*count)(void *, uint32_t *);
    MSXResidentStatus (*rawCount)(void *, uint64_t *, uint32_t *);
    MSXResidentStatus (*next)(void *, MSXResidentActiveRow *);
    MSXResidentStatus (*nextBatch)(void *, MSXResidentActiveRow *,
                                   uint32_t, uint32_t *);
    int (*identity)(void *, const MSXResidentActiveRow *);
    int (*audit)(void *, const MSXResidentActiveRow *);
    uint64_t (*topology)(void *);
} MSXResidentActiveStreamSource;

static MSXResidentStatus MSXresident_activeStreamRawCount(
    MSXResidentActiveStreamSource *source, uint64_t *raw, uint32_t *links)
{
    MSXResidentStatus s;
    uint32_t n = 0;
    if (source->rawCount)
        return source->rawCount(source->context, raw, links);
    if (!source->begin || !source->validate || !source->count)
        return MSX_RESIDENT_ERR_ARGUMENT;
    s = source->begin(source->context);
    if (s == MSX_RESIDENT_OK) s = source->validate(source->context);
    if (s == MSX_RESIDENT_OK) s = source->count(source->context, &n);
    if (s == MSX_RESIDENT_OK) { *raw = n; if (links) *links = 0; }
    return s;
}
static MSXResidentStatus MSXresident_activeStreamBegin(
    MSXResidentActiveStreamSource *source)
{ return source->begin ? source->begin(source->context) : MSX_RESIDENT_ERR_ARGUMENT; }
static MSXResidentStatus MSXresident_activeStreamNextBatch(
    MSXResidentActiveStreamSource *source, MSXResidentActiveRow *rows,
    uint32_t cap, uint32_t *count)
{
    MSXResidentStatus s;
    uint32_t n = 0;
    if (source->nextBatch)
        return source->nextBatch(source->context, rows, cap, count);
    if (!source->next || !count || !cap) return MSX_RESIDENT_ERR_ARGUMENT;
    while (n < cap)
    {
        s = source->next(source->context, &rows[n]);
        if (s == MSX_RESIDENT_OK) { ++n; continue; }
        *count = n;
        return s;
    }
    *count = n;
    return n ? MSX_RESIDENT_OK : MSX_RESIDENT_ITER_END;
}
static int MSXresident_activeStreamIdentity(
    MSXResidentActiveStreamSource *source, const MSXResidentActiveRow *row)
{ return source->identity && source->identity(source->context, row); }
static int MSXresident_activeStreamAudit(
    MSXResidentActiveStreamSource *source, const MSXResidentActiveRow *row)
{ return source->audit && source->audit(source->context, row); }
static uint64_t MSXresident_activeStreamTopology(
    MSXResidentActiveStreamSource *source)
{ return source->topology ? source->topology(source->context) : UINT64_MAX; }

#else

/* Production source: no function pointers and no callback dispatch. */
typedef struct {
    MSXResidentActiveIterator iterator;
    const MSXResidentHydView *hyd;
    int auditHyd;
    uint64_t topologyVersion;
    int stage;
    double enumerateStart, streamStart;
} MSXResidentActiveStreamSource;

#ifndef MSX_RESIDENT_ACTIVE_IDENTITY
#define MSX_RESIDENT_ACTIVE_IDENTITY(source, row) 1
#endif
#ifndef MSX_RESIDENT_ACTIVE_AUDIT
#define MSX_RESIDENT_ACTIVE_AUDIT(source, row) 0
#endif
#ifndef MSX_RESIDENT_ACTIVE_PROFILE_ENUM_END
#define MSX_RESIDENT_ACTIVE_PROFILE_ENUM_END(source) ((void)0)
#endif
#ifndef MSX_RESIDENT_ACTIVE_PROFILE_STREAM_BEGIN
#define MSX_RESIDENT_ACTIVE_PROFILE_STREAM_BEGIN(source) ((void)0)
#endif
#ifndef MSX_RESIDENT_ACTIVE_PROFILE_STREAM_END
#define MSX_RESIDENT_ACTIVE_PROFILE_STREAM_END(source) ((void)0)
#endif

static MSXResidentStatus MSXresident_activeStreamRawCount(
    MSXResidentActiveStreamSource *source, uint64_t *raw, uint32_t *links)
{ (void)source; return MSXresident_rawActiveCount(raw, links); }
static MSXResidentStatus MSXresident_activeStreamBegin(
    MSXResidentActiveStreamSource *source)
{ return MSXresident_beginActiveIterator(&source->iterator); }
static MSXResidentStatus MSXresident_activeStreamNextBatch(
    MSXResidentActiveStreamSource *source, MSXResidentActiveRow *rows,
    uint32_t cap, uint32_t *count)
{ return MSXresident_nextActiveBatch(&source->iterator, rows, cap, count); }
static int MSXresident_activeStreamIdentity(
    MSXResidentActiveStreamSource *source, const MSXResidentActiveRow *row)
{ return MSX_RESIDENT_ACTIVE_IDENTITY(source, row); }
static int MSXresident_activeStreamAudit(
    MSXResidentActiveStreamSource *source, const MSXResidentActiveRow *row)
{ return source->auditHyd ? MSX_RESIDENT_ACTIVE_AUDIT(source, row) : 0; }
static uint64_t MSXresident_activeStreamTopology(
    MSXResidentActiveStreamSource *source)
{ return source->topologyVersion; }

#endif

static MSXResidentStatus MSXresident_activeStreamBuild(
    MSXResidentGpu *gpu, MSXResidentGpuActiveWriter *writer,
    uint64_t topologyVersion, MSXResidentActiveStreamSource *source,
    const MSXResidentHydView *hyd, uint32_t *activeOut,
    MSXResidentActiveStreamReport *report)
{
    MSXResidentActiveRow rows[MSX_RESIDENT_ACTIVE_CHUNK_ROWS];
    MSXResidentStatus s, nextStatus;
    MSXResidentStatus deferredBegin = MSX_RESIDENT_OK;
    MSXResidentStatus deferred = MSX_RESIDENT_OK;
    MSXResidentStatus iterError = MSX_RESIDENT_OK;
    uint64_t rawCount = 0;
    uint32_t rawLinkCount = 0;
    uint32_t expected = 0, batchCount = 0, i;
    int writerLive = 0, done = 0;

    (void)hyd;
    if (!gpu || !writer || !source || !activeOut)
        return MSX_RESIDENT_ERR_ARGUMENT;
    *activeOut = 0;
    if (report) memset(report, 0, sizeof(*report));
#if !defined(MSX_RESIDENT_ACTIVE_STREAM_TEST)
    if (source->stage) source->enumerateStart = MSXgpu_wallTimeMs();
#endif

    if (report) ++report->rawCountPasses;
    s = MSXresident_activeStreamRawCount(source, &rawCount, &rawLinkCount);
    if (report)
    {
        report->rawLinks += rawLinkCount;
        report->rawExpectedRows += rawCount;
    }
    if (s != MSX_RESIDENT_OK)
        deferredBegin = s;
    else if (rawCount > UINT32_MAX)
        deferredBegin = MSX_RESIDENT_ERR_OVERFLOW;
    else
        expected = (uint32_t)rawCount;
    if (report) report->expectedCount = expected;

    if (report) ++report->iteratorPasses;
    s = MSXresident_activeStreamBegin(source);
#if !defined(MSX_RESIDENT_ACTIVE_STREAM_TEST)
    MSX_RESIDENT_ACTIVE_PROFILE_ENUM_END(source);
#endif
    if (s != MSX_RESIDENT_OK)
    {
        if (report) ++report->iteratorFailures;
        return s;
    }

    if (deferredBegin == MSX_RESIDENT_OK)
    {
        s = MSXresidentGpu_beginActive(gpu, expected, topologyVersion, writer);
        if (s != MSX_RESIDENT_OK)
        {
            deferredBegin = s;
            if (report) ++report->beginFailures;
        }
        else
            writerLive = 1;
    }

#if !defined(MSX_RESIDENT_ACTIVE_STREAM_TEST)
    MSX_RESIDENT_ACTIVE_PROFILE_STREAM_BEGIN(source);
#endif
    while (!done)
    {
        uint32_t identityFail = 0, hydFail = 0, accepted = 0;
        uint32_t appendCount = 0;
        MSXResidentStatus terminal = MSX_RESIDENT_OK;
        int haveIdentityFail = 0, haveHydFail = 0;

        s = MSXresident_activeStreamNextBatch(source, rows,
                                               MSX_RESIDENT_ACTIVE_CHUNK_ROWS,
                                               &batchCount);
        nextStatus = s;
        if (batchCount && report)
        {
            ++report->iteratorChunkCalls;
            if (batchCount > report->chunkHighWater)
                report->chunkHighWater = batchCount;
        }
        if (deferred == MSX_RESIDENT_OK && writerLive)
        {
            for (i = 0; i < batchCount; ++i)
                if (!MSXresident_activeStreamIdentity(source, &rows[i]))
                { identityFail = i; haveIdentityFail = 1; break; }
        }
        for (i = 0; i < batchCount; ++i)
            if (MSXresident_activeStreamAudit(source, &rows[i]))
            { if (!haveHydFail) hydFail = i; haveHydFail = 1;
              if (report) ++report->hydMismatches; }

        if (deferred == MSX_RESIDENT_OK && writerLive)
        {
            if (haveIdentityFail && (!haveHydFail || identityFail <= hydFail))
            { appendCount = identityFail; terminal = MSX_RESIDENT_ERR_GENERATION; }
            else if (haveHydFail)
            { appendCount = hydFail + 1u; terminal = MSX_RESIDENT_ERR_ARGUMENT; }
            else appendCount = batchCount;
            if (appendCount)
            {
                if (report) { ++report->batchCalls; report->batchRowsAttempted += appendCount; }
                s = MSXresidentGpu_appendActiveBatch(gpu, writer, rows,
                                                      appendCount, &accepted);
                if (report) report->rowsAppended += accepted;
                if (s != MSX_RESIDENT_OK)
                {
                    deferred = s; writerLive = 0;
                    if (report) { ++report->appendFailures; ++report->builderAborts; }
                }
                else if (terminal != MSX_RESIDENT_OK)
                {
                    deferred = terminal;
                    (void)MSXresidentGpu_abortActiveBuild(gpu, writer);
                    writerLive = 0;
                    if (report) ++report->builderAborts;
                }
            }
            else if (terminal != MSX_RESIDENT_OK)
            {
                deferred = terminal;
                (void)MSXresidentGpu_abortActiveBuild(gpu, writer);
                writerLive = 0;
                if (report) ++report->builderAborts;
            }
        }
        if (nextStatus == MSX_RESIDENT_ITER_END) done = 1;
        else if (nextStatus != MSX_RESIDENT_OK)
        { iterError = nextStatus; done = 1; }
    }
#if !defined(MSX_RESIDENT_ACTIVE_STREAM_TEST)
    MSX_RESIDENT_ACTIVE_PROFILE_STREAM_END(source);
#endif
    if (iterError != MSX_RESIDENT_OK)
    {
        if (report) ++report->iteratorFailures;
        if (writerLive)
        {
            (void)MSXresidentGpu_abortActiveBuild(gpu, writer);
            writerLive = 0;
            if (report) ++report->builderAborts;
        }
        return iterError;
    }
    if (deferredBegin != MSX_RESIDENT_OK) return deferredBegin;
    if (deferred != MSX_RESIDENT_OK) return deferred;
    if (MSXresident_activeStreamTopology(source) != topologyVersion)
    {
        (void)MSXresidentGpu_abortActiveBuild(gpu, writer);
        if (report) ++report->builderAborts;
        return MSX_RESIDENT_ERR_GENERATION;
    }
    s = MSXresidentGpu_sealActive(gpu, writer, topologyVersion);
    if (s != MSX_RESIDENT_OK)
    {
        if (report) { ++report->sealFailures; ++report->builderAborts; }
        return s;
    }
    *activeOut = expected;
    return MSX_RESIDENT_OK;
}

#endif
