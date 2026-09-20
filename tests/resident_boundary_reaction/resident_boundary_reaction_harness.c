/* Boundary-reaction view contract: compare the current FULL filtered walk
   with the current certified CoreSpan boundary walk.  This target exercises
   the same endpoint API used by msxchem without retaining any topology
   pointer across a mutation. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "msxsegment_storage.h"
#include "msxgpu.h"
#include "msxtypes.h"

#define LINK_COUNT 3
#define SPECIES_COUNT 1
#define MAX_IDS 32
#define ERR_EXPECTED ERR_PIPE_RING_CAPACITY

MSXproject MSX;
static int passed, failed;

#define CHECK(expr) do { if (expr) ++passed; else { ++failed; \
    fprintf(stderr, "FAIL:%d: %s\n", __LINE__, #expr); } } while (0)

int ENgetlinkid(int index, char *id) { sprintf(id, "L%d", index); return 0; }
int ENwriteline(char *line) { (void)line; return 0; }
char *Alloc(long n) { return (char *)calloc(1, (size_t)n); }
double MSXgpu_wallTimeMs(void) { return 0.0; }
int MSXgpu_profileStageEnabled(void) { return 0; }
int MSXgpu_profileDetailGroupEnabled(MSXProfileDetailGroup group)
{ (void)group; return 0; }
void MSXgpu_profileRecordDemote(uint64_t rows, double ms)
{ (void)rows; (void)ms; }
void MSXgpu_profileRecordPromote(uint64_t rows, double ms)
{ (void)rows; (void)ms; }
void MSXgpu_profileRecordRebalance(const MSXRebalanceMetrics *metrics)
{ (void)metrics; }

Pseg MSXqual_getFreeSeg(double volume, double *c)
{
    Pseg seg = (Pseg)calloc(1, sizeof(*seg));
    int m;
    if (!seg) return NULL;
    seg->c = (double *)calloc(SPECIES_COUNT + 1, sizeof(double));
    seg->lastc = (double *)calloc(SPECIES_COUNT + 1, sizeof(double));
    if (!seg->c || !seg->lastc)
    {
        free(seg->c); free(seg->lastc); free(seg); return NULL;
    }
    seg->privateC = seg->c;
    seg->privateLastC = seg->lastc;
    seg->v = volume;
    if (c) for (m = 0; m <= SPECIES_COUNT; ++m) seg->c[m] = c[m];
    return seg;
}

void MSXqual_removeSeg(Pseg seg)
{
    if (!seg) return;
    if (seg->privateLastC != seg->privateC) free(seg->privateLastC);
    free(seg->privateC);
    free(seg);
}

static Pseg makeSegment(int link, int ordinal)
{
    Pseg seg = (Pseg)calloc(1, sizeof(*seg));
    if (!seg) return NULL;
    seg->c = (double *)calloc(SPECIES_COUNT + 1, sizeof(double));
    seg->lastc = (double *)calloc(SPECIES_COUNT + 1, sizeof(double));
    if (!seg->c || !seg->lastc)
    {
        free(seg->c); free(seg->lastc); free(seg); return NULL;
    }
    seg->privateC = seg->c;
    seg->privateLastC = seg->lastc;
    seg->v = 1.0 + ordinal;
    seg->hybridId = (uint64_t)link * 1000u + (uint64_t)ordinal + 1u;
    return seg;
}

static int buildFixture(void)
{
    static const int nseg[LINK_COUNT + 1] = {0, 3, 7, 5};
    MSXResidentLayout layout;
    uint32_t capacity[LINK_COUNT + 1] = {0, 1, 5, 3};
    uint32_t base[LINK_COUNT + 1] = {0, 0, 1, 6};
    uint32_t guard[LINK_COUNT + 1] = {0, 2, 2, 2};
    int k, i;
    memset(&MSX, 0, sizeof(MSX));
    MSX.Nobjects[LINK] = LINK_COUNT;
    MSX.Nobjects[SPECIES] = SPECIES_COUNT;
    MSX.MaxSegments = 16;
    MSX.SegmentStorage = SEG_STORAGE_HYBRID;
    MSX.Link = (Slink *)calloc(LINK_COUNT + 1, sizeof(Slink));
    MSX.FirstSeg = (Pseg *)calloc(LINK_COUNT + 1, sizeof(Pseg));
    MSX.LastSeg = (Pseg *)calloc(LINK_COUNT + 1, sizeof(Pseg));
    if (!MSX.Link || !MSX.FirstSeg || !MSX.LastSeg) return 0;
    for (k = 1; k <= LINK_COUNT; ++k)
    {
        Pseg previous = NULL;
        for (i = 0; i < nseg[k]; ++i)
        {
            Pseg seg = makeSegment(k, i);
            if (!seg) return 0;
            seg->next = previous;
            if (previous) previous->prev = seg;
            else MSX.FirstSeg[k] = seg;
            previous = seg;
            MSX.LastSeg[k] = seg;
            ++MSX.Link[k].nsegs;
        }
    }
    if (MSXsegStorage_open()) return 0;
    layout.nLinks = LINK_COUNT;
    layout.totalSlots = 9;
    layout.speciesStride = SPECIES_COUNT + 1;
    layout.capacity = capacity;
    layout.base = base;
    layout.guard = guard;
    if (MSXsegStorage_hybridReserve(&layout)) return 0;
    return MSXsegStorage_hybridizeAll() == 0;
}

static void destroyFixture(void)
{
    int k;
    Pseg seg, next;
    for (k = 1; k <= LINK_COUNT; ++k)
    {
        for (seg = MSX.FirstSeg[k]; seg; seg = next)
        {
            next = seg->prev;
            if (!seg->inHybridCore) MSXqual_removeSeg(seg);
        }
        MSX.FirstSeg[k] = MSX.LastSeg[k] = NULL;
        MSX.Link[k].nsegs = 0;
    }
    MSXsegStorage_close();
    free(MSX.FirstSeg); free(MSX.LastSeg); free(MSX.Link);
    memset(&MSX, 0, sizeof(MSX));
}

static void reverseList(int k)
{
    Pseg seg = MSX.FirstSeg[k], next;
    Pseg oldFirst = MSX.FirstSeg[k];
    while (seg)
    {
        next = seg->prev;
        seg->prev = seg->next;
        seg->next = next;
        seg = next;
    }
    MSX.FirstSeg[k] = MSX.LastSeg[k];
    MSX.LastSeg[k] = oldFirst;
}

typedef struct
{
    uint64_t ids[MAX_IDS];
    int count;
    int coreSeen;
    int attempted;
    int failAt;
} BoundaryTrace;

static int traceBoundaryVisitor(int k, double dt, Pseg seg, void *context)
{
    BoundaryTrace *trace = (BoundaryTrace *)context;
    (void)k;
    (void)dt;
    ++trace->attempted;
    if (seg->inHybridCore) ++trace->coreSeen;
    if (trace->failAt > 0 && trace->attempted == trace->failAt)
        return ERR_EXPECTED;
    if (trace->count >= MAX_IDS) return ERR_EXPECTED;
    trace->ids[trace->count++] = seg->hybridId;
    return 0;
}

static int collectWithProductionTraversal(int k, int mode, int failAt,
                                          BoundaryTrace *trace,
                                          MSXResidentBoundaryScanResult *scan)
{
    memset(trace, 0, sizeof(*trace));
    trace->failAt = failAt;
    return MSXsegStorage_visitHybridBoundarySegments(
        k, mode, 0.0, traceBoundaryVisitor, trace, scan);
}

static void compareBoundaryOrder(int k)
{
    BoundaryTrace full, span;
    MSXResidentBoundaryScanResult fullScan, spanScan;
    CHECK(collectWithProductionTraversal(
        k, MSX_RESIDENT_BOUNDARY_SCAN_FULL, 0, &full, &fullScan) == 0);
    CHECK(collectWithProductionTraversal(
        k, MSX_RESIDENT_BOUNDARY_SCAN_SPAN, 0, &span, &spanScan) == 0);
    CHECK(fullScan.skippedCore == (uint64_t)MSXsegStorage_hybridCoreCount(k));
    CHECK(full.coreSeen == 0 && span.coreSeen == 0);
    CHECK(!spanScan.spanHit || spanScan.skippedCore == fullScan.skippedCore);
    CHECK(full.count == span.count);
    CHECK(full.count == 0 || memcmp(full.ids, span.ids,
                                    (size_t)full.count * sizeof(full.ids[0])) == 0);
}

static void checkErrorPrefix(int k)
{
    BoundaryTrace full, span;
    MSXResidentBoundaryScanResult fullScan, spanScan;
    int fullError, spanError;
    fullError = collectWithProductionTraversal(
        k, MSX_RESIDENT_BOUNDARY_SCAN_FULL, 2, &full, &fullScan);
    spanError = collectWithProductionTraversal(
        k, MSX_RESIDENT_BOUNDARY_SCAN_SPAN, 2, &span, &spanScan);
    CHECK(fullError == ERR_EXPECTED && spanError == ERR_EXPECTED);
    CHECK(full.attempted == span.attempted && full.count == span.count);
    CHECK(full.count == 0 || memcmp(full.ids, span.ids,
                                    (size_t)full.count * sizeof(full.ids[0])) == 0);
    CHECK(fullScan.visits == (uint64_t)full.count &&
          spanScan.visits == (uint64_t)span.count);
}

static void checkEndpoints(int k)
{
    Pseg first = NULL, last = NULL;
    int count = 0, verified = 0;
    CHECK(MSXsegStorage_hybridBoundaryView(
        k, &first, &last, &count, &verified) == 0);
    CHECK(verified == 1);
    CHECK(count == MSXsegStorage_hybridCoreCount(k));
    if (count == 0)
        CHECK(first == NULL && last == NULL);
    else
        CHECK(first != NULL && last != NULL && first->inHybridCore &&
              last->inHybridCore);
}

static void checkInvalidAndFallback(void)
{
    Pseg core = MSXsegStorage_hybridCoreSegAt(2, 0);
    Pseg *span = NULL;
    int count = 0;
    CHECK(core != NULL);
    if (core) core->ownerLink = 0;
    CHECK(MSXsegStorage_hybridBoundaryView(2, NULL, NULL, NULL, NULL) ==
          ERR_EXPECTED);
    CHECK(MSXsegStorage_hybridAuditCoreSpan(2) == ERR_EXPECTED);
    CHECK(MSXsegStorage_hybridCoreSpan(2, 0, &span, &count) == ERR_EXPECTED);
    if (core) core->ownerLink = 2;
    CHECK(MSXsegStorage_hybridAuditCoreSpan(2) == 0);
    compareBoundaryOrder(2);

    CHECK(MSXsegStorage_testResidentSpanCommitFailure(2) == ERR_EXPECTED);
    CHECK(MSXsegStorage_hybridBoundaryView(2, NULL, NULL, NULL, NULL) == 0);
    compareBoundaryOrder(2); /* legal unverified state falls back to FULL */
    checkErrorPrefix(2);
    CHECK(MSXsegStorage_hybridAuditCoreSpan(2) == 0);
    compareBoundaryOrder(2);
}

int main(void)
{
    CHECK(buildFixture());
    checkEndpoints(1); /* no Core */
    checkEndpoints(2); /* multi-Core, non-power-of-two capacity */
    checkEndpoints(3); /* small ring exercises the wrapped-view fixture */
    compareBoundaryOrder(1);
    compareBoundaryOrder(2);
    compareBoundaryOrder(3);
    checkErrorPrefix(2);

    reverseList(2);
    CHECK(MSXsegStorage_hybridAfterListReorder(2) == 0);
    compareBoundaryOrder(2);
    reverseList(2);
    CHECK(MSXsegStorage_hybridAfterListReorder(2) == 0);
    compareBoundaryOrder(2);
    checkInvalidAndFallback();

    destroyFixture();
    printf("assertions_passed=%d\nassertions_failed=%d\n", passed, failed);
    return failed ? 1 : 0;
}
