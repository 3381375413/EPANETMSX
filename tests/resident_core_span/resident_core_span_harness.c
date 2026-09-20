/* CoreSpan invariant audit: mixed empty/non-empty links, non-power-of-two
   fixed capacities, reverse traversal, reuse, and controlled corruption. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "msxsegment_storage.h"
#include "msxgpu.h"
#include "msxtypes.h"

#define LINK_COUNT 3
#define SPECIES_COUNT 1
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
    /* Core view segments are owned by Hybrid and are released by close;
       boundary segments remain caller-owned CPU objects. */
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

static void checkGoodSpans(void)
{
    Pseg *first = NULL, *second = NULL;
    int firstCount = 0, secondCount = 0, pos;
    CHECK(MSXsegStorage_hybridCoreCount(1) == 0);
    CHECK(MSXsegStorage_hybridCoreSpan(1, 0, &first, &firstCount) == 0 &&
          first == NULL && firstCount == 0);
    CHECK(MSXsegStorage_hybridCoreCount(2) == 3);
    CHECK(MSXsegStorage_hybridCoreSpan(2, 0, &first, &firstCount) > 0 &&
          MSXsegStorage_hybridCoreSpan(2, 1, &second, &secondCount) >= 0 &&
          firstCount + secondCount == 3);
    for (pos = 0; pos < 3; ++pos)
        CHECK(MSXsegStorage_hybridCoreSegAt(2, pos) != NULL);
    CHECK(MSXsegStorage_hybridCoreSpan(3, 0, &first, &firstCount) > 0 &&
          firstCount == 1);
}

static void checkReverseAndReuse(void)
{
    Pseg *first = NULL, *second = NULL;
    int firstCount = 0, secondCount = 0;
    reverseList(2);
    CHECK(MSXsegStorage_hybridAfterListReorder(2) == 0);
    CHECK(MSXsegStorage_hybridOrientation(2) == -1);
    CHECK(MSXsegStorage_hybridCoreSpan(2, 0, &first, &firstCount) > 0 &&
          MSXsegStorage_hybridCoreSpan(2, 1, &second, &secondCount) >= 0 &&
          -firstCount - secondCount == 3);
}

static void checkCorruption(int kind)
{
    Pseg core, seg, candidate = NULL;
    Pseg *view = NULL;
    int count = 0, seen = 0;
    CHECK(buildFixture());
    core = MSXsegStorage_hybridCoreSegAt(2, 0);
    if (kind == 1)
    {
        for (seg = MSX.FirstSeg[2]; seg; seg = seg->prev)
        {
            if (seg == core) seen = 1;
            else if (seen && !seg->inHybridCore) { candidate = seg; break; }
        }
        CHECK(candidate != NULL);
        if (candidate) candidate->inHybridCore = TRUE;
    }
    else if (kind == 2)
    {
        seg = MSXsegStorage_hybridCoreSegAt(2, 1);
        CHECK(seg != NULL);
        if (seg) seg->inHybridCore = FALSE;
    }
    else
    {
        CHECK(core != NULL);
        if (core) core->ownerLink = 0;
    }
    CHECK(MSXsegStorage_hybridAuditCoreSpan(2) == ERR_EXPECTED);
    if (kind == 1 && candidate) candidate->inHybridCore = FALSE;
    if (kind == 2 && seg) seg->inHybridCore = TRUE;
    if (kind == 3 && core) core->ownerLink = 2;
    destroyFixture();
}

int main(void)
{
    CHECK(buildFixture());
    checkGoodSpans();
    checkReverseAndReuse();
    destroyFixture();

    CHECK(buildFixture());
    checkGoodSpans();
    destroyFixture();

    checkCorruption(1); /* hidden second Core */
    checkCorruption(2); /* internal hole */
    checkCorruption(3); /* owner/mapping mismatch */

    printf("assertions_passed=%d\nassertions_failed=%d\n", passed, failed);
    return failed ? 1 : 0;
}
