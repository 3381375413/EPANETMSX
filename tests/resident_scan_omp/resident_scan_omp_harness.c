/* Real OpenMP Resident scan contract harness.  The test invokes the same
   production scan helper/workshare used by the solver, with 64 non-empty
   links, and compares it with FULL_SERIAL. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "msxsegment_storage.h"
#include "msxgpu.h"
#include "msxtypes.h"

#if !defined(_OPENMP)
#error "resident_scan_omp_harness must be compiled with OpenMP"
#endif

#define LINK_COUNT 64
#define SEGMENTS_PER_LINK 3
#define CHECK(expr) do { if (expr) ++passed; else { ++failed; \
    fprintf(stderr, "FAIL:%d: %s\n", __LINE__, #expr); } } while (0)

MSXproject MSX;
static int passed, failed;

int ENgetlinkid(int index, char *id)
{
    sprintf(id, "L%d", index);
    return 0;
}
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
/* The failure seam stops before fetch; this test-only stub keeps the storage
   object linkable without pulling in the production CUDA runtime. */
MSXResidentStatus MSXresidentRuntime_fetchBatch(
    const MSXResidentHandoffItem *items, uint32_t count, double *c,
    double *lastc, uint32_t stride, MSXResidentHandoffResult *results)
{ (void)items; (void)count; (void)c; (void)lastc; (void)stride;
  (void)results; return MSX_RESIDENT_ERR_GPU; }

Pseg MSXqual_getFreeSeg(double volume, double *c)
{
    Pseg seg = (Pseg)calloc(1, sizeof(*seg));
    int n = MSX.Nobjects[SPECIES] + 1;
    if (!seg) return NULL;
    seg->c = (double *)calloc((size_t)n, sizeof(double));
    seg->lastc = (double *)calloc((size_t)n, sizeof(double));
    seg->privateC = seg->c;
    seg->privateLastC = seg->lastc;
    seg->v = volume;
    if (c && seg->c) memcpy(seg->c, c, (size_t)n * sizeof(double));
    return seg;
}
void MSXqual_removeSeg(Pseg seg)
{
    if (!seg) return;
    free(seg->privateC);
    if (seg->privateLastC != seg->privateC) free(seg->privateLastC);
    free(seg);
}

static void freeTopology(void)
{
    int k;
    for (k = 1; k <= LINK_COUNT; ++k)
    {
        Pseg seg = MSX.FirstSeg[k];
        while (seg)
        {
            Pseg next = seg->prev;
            free(seg);
            seg = next;
        }
    }
    free(MSX.FirstSeg);
    free(MSX.LastSeg);
    free(MSX.Link);
    memset(&MSX, 0, sizeof(MSX));
}

static void buildTopology(void)
{
    int k, i;
    memset(&MSX, 0, sizeof(MSX));
    MSX.Nobjects[LINK] = LINK_COUNT;
    MSX.Nobjects[SPECIES] = 1;
    MSX.MaxSegments = 16;
    MSX.SegmentStorage = SEG_STORAGE_HYBRID;
    MSX.Link = (Slink *)calloc(LINK_COUNT + 1, sizeof(Slink));
    MSX.FirstSeg = (Pseg *)calloc(LINK_COUNT + 1, sizeof(Pseg));
    MSX.LastSeg = (Pseg *)calloc(LINK_COUNT + 1, sizeof(Pseg));
    for (k = 1; k <= LINK_COUNT; ++k)
    {
        Pseg previous = NULL;
        for (i = 0; i < SEGMENTS_PER_LINK; ++i)
        {
            Pseg seg = (Pseg)calloc(1, sizeof(*seg));
            seg->hybridId = (uint64_t)k * 1000u + (uint64_t)i;
            seg->c = (double *)calloc((size_t)MSX.Nobjects[SPECIES] + 1,
                                      sizeof(double));
            seg->lastc = (double *)calloc((size_t)MSX.Nobjects[SPECIES] + 1,
                                          sizeof(double));
            seg->privateC = seg->c;
            seg->privateLastC = seg->lastc;
            seg->next = previous;
            if (previous) previous->prev = seg;
            else MSX.FirstSeg[k] = seg;
            previous = seg;
            MSX.LastSeg[k] = seg;
            ++MSX.Link[k].nsegs;
        }
    }
}

static void appendSegment(int k, int ordinal)
{
    Pseg seg = (Pseg)calloc(1, sizeof(*seg));
    if (!seg) return;
    seg->hybridId = (uint64_t)k * 1000u + (uint64_t)ordinal;
    seg->c = (double *)calloc((size_t)MSX.Nobjects[SPECIES] + 1,
                              sizeof(double));
    seg->lastc = (double *)calloc((size_t)MSX.Nobjects[SPECIES] + 1,
                                  sizeof(double));
    seg->privateC = seg->c;
    seg->privateLastC = seg->lastc;
    seg->next = MSX.LastSeg[k];
    if (MSX.LastSeg[k]) MSX.LastSeg[k]->prev = seg;
    else MSX.FirstSeg[k] = seg;
    MSX.LastSeg[k] = seg;
    ++MSX.Link[k].nsegs;
}

static void reverseTopology(int k)
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

static int uniqueWorkers(const int *workerIds)
{
    int seen[LINK_COUNT] = {0};
    int i, count = 0, worker;
    for (i = 1; i <= LINK_COUNT; ++i)
    {
        worker = workerIds[i];
        if (worker < 0 || worker >= (int)(sizeof(seen) / sizeof(seen[0])))
            return -1;
        if (!seen[worker])
        {
            seen[worker] = 1;
            ++count;
        }
    }
    return count;
}

static int fileContains(const char *path, const char *needle)
{
    FILE *f = fopen(path, "rt");
    char line[256];
    int found = 0;
    if (!f) return 0;
    while (fgets(line, sizeof(line), f))
        if (strstr(line, needle)) { found = 1; break; }
    fclose(f);
    return found;
}

int main(void)
{
    int parallelTeam = 0, serialTeam = 0, i;
    int invalidWorkers[LINK_COUNT + 1] = {0};
    int parallelWorkers[LINK_COUNT + 1], serialWorkers[LINK_COUNT + 1];
    MSXHybridAuditSnapshot parallel[LINK_COUNT], serial[LINK_COUNT];
    MSXHybridAuditSnapshot fullCore[LINK_COUNT], spanOmp[LINK_COUNT];
    MSXHybridAuditSnapshot spanSerial[LINK_COUNT], reverseFull[LINK_COUNT];
    MSXHybridAuditSnapshot reverseSpan[LINK_COUNT];
    int spanWorkers[LINK_COUNT + 1], reverseWorkers[LINK_COUNT + 1];
    Pseg first[LINK_COUNT + 1], last[LINK_COUNT + 1];

    _putenv_s("MSX_RESIDENT_SCAN_AUDIT", "1");
    _putenv_s("MSX_RESIDENT_SCAN_MODE", "FULL_OMP8");
    remove("resident_scan_team_size.txt");
    remove("resident_scan_omp_team_size.txt");
    buildTopology();
    for (i = 1; i <= LINK_COUNT; ++i)
    {
        first[i] = MSX.FirstSeg[i];
        last[i] = MSX.LastSeg[i];
    }
    CHECK(MSXsegStorage_open() == 0);
    CHECK(MSXsegStorage_testResidentScanOMP(
        0, &parallelTeam, parallelWorkers, LINK_COUNT + 1,
        parallel, LINK_COUNT) == 0);
    CHECK(parallelTeam == 8);
    CHECK(uniqueWorkers(parallelWorkers) >= 2);
    invalidWorkers[1] = -1;
    CHECK(uniqueWorkers(invalidWorkers) == -1);
    invalidWorkers[1] = LINK_COUNT;
    CHECK(uniqueWorkers(invalidWorkers) == -1);
    for (i = 0; i < LINK_COUNT; ++i)
        CHECK(parallel[i].link_index == i + 1 &&
              parallel[i].total == SEGMENTS_PER_LINK &&
              parallel[i].core_count == 0 &&
              parallel[i].downstream_boundary == SEGMENTS_PER_LINK &&
              parallel[i].upstream_boundary == 0 &&
              parallel[i].first_core_slot == -1 &&
              parallel[i].last_core_slot == -1);
    for (i = 1; i <= LINK_COUNT; ++i)
        CHECK(MSX.FirstSeg[i] == first[i] && MSX.LastSeg[i] == last[i] &&
              MSX.Link[i].nsegs == SEGMENTS_PER_LINK);

    CHECK(MSXsegStorage_testResidentScanOMP(
        1, &serialTeam, serialWorkers, LINK_COUNT + 1,
        serial, LINK_COUNT) == 0);
    CHECK(serialTeam == 1);
    for (i = 1; i <= LINK_COUNT; ++i)
        CHECK(serialWorkers[i] == 0);
    for (i = 0; i < LINK_COUNT; ++i)
        CHECK(serial[i].link_index == parallel[i].link_index &&
              serial[i].total == parallel[i].total &&
              serial[i].core_count == parallel[i].core_count &&
              serial[i].downstream_boundary == parallel[i].downstream_boundary &&
              serial[i].upstream_boundary == parallel[i].upstream_boundary &&
              serial[i].first_core_slot == parallel[i].first_core_slot &&
              serial[i].last_core_slot == parallel[i].last_core_slot &&
              serial[i].orient == parallel[i].orient &&
              serial[i].first_core_id == parallel[i].first_core_id &&
              serial[i].last_core_id == parallel[i].last_core_id);

    /* close/reopen resets both cached environment decisions and write-once
       guards.  Switch the environment between lifetimes to catch stale
       audit state in a single process. */
    CHECK(fileContains("resident_scan_omp_team_size.txt",
                      "scan_mode=FULL_OMP8"));
    MSXsegStorage_close();
    remove("resident_scan_omp_team_size.txt");
    _putenv_s("MSX_RESIDENT_SCAN_AUDIT", "0");
    CHECK(MSXsegStorage_open() == 0);
    CHECK(MSXsegStorage_testResidentScanOMP(
        0, &parallelTeam, parallelWorkers, LINK_COUNT + 1,
        parallel, LINK_COUNT) == 0);
    CHECK(!fileContains("resident_scan_omp_team_size.txt", "scan_mode="));
    MSXsegStorage_close();
    _putenv_s("MSX_RESIDENT_SCAN_AUDIT", "1");
    _putenv_s("MSX_RESIDENT_SCAN_MODE", "FULL_SERIAL");
    CHECK(MSXsegStorage_open() == 0);
    CHECK(MSXsegStorage_testResidentScanOMP(
        -1, &serialTeam, serialWorkers, LINK_COUNT + 1,
        serial, LINK_COUNT) == 0);
    CHECK(serialTeam == 1);
    CHECK(fileContains("resident_scan_omp_team_size.txt",
                      "scan_mode=FULL_SERIAL"));

    /* The same seam now exercises all four modes against an already
       certified, non-empty CoreSpan.  The initial 3-segment fixture is
       extended through the ordinary CPU list API; no pointer is fabricated. */
    {
        uint32_t capacity[LINK_COUNT + 1], base[LINK_COUNT + 1];
        uint32_t guard[LINK_COUNT + 1];
        MSXResidentLayout layout;
        int k, i, spanTeam = 0, reverseTeam = 0;
        for (k = 1; k <= LINK_COUNT; ++k)
        {
            for (i = 3; i < 7; ++i) appendSegment(k, i);
            capacity[k] = 16;
            base[k] = 0;
            guard[k] = 2;
        }
        layout.nLinks = LINK_COUNT;
        layout.totalSlots = LINK_COUNT;
        layout.speciesStride = MSX.Nobjects[SPECIES] + 1;
        layout.capacity = capacity;
        layout.base = base;
        layout.guard = guard;
        CHECK(MSXsegStorage_hybridReserve(&layout) == 0);
        CHECK(MSXsegStorage_hybridizeAll() == 0);
        CHECK(MSXsegStorage_hybridCoreCount(1) == 3);

        CHECK(MSXsegStorage_testResidentScanOMP(
            MSX_RESIDENT_SCAN_FULL_SERIAL, &serialTeam, serialWorkers,
            LINK_COUNT + 1, fullCore, LINK_COUNT) == 0);
        CHECK(serialTeam == 1);
        CHECK(MSXsegStorage_testResidentScanOMP(
            MSX_RESIDENT_SCAN_SPAN_OMP8, &spanTeam, spanWorkers,
            LINK_COUNT + 1, spanOmp, LINK_COUNT) == 0);
        CHECK(spanTeam == 8 && uniqueWorkers(spanWorkers) >= 2);
        CHECK(MSXsegStorage_testResidentScanOMP(
            MSX_RESIDENT_SCAN_SPAN_SERIAL, &serialTeam, serialWorkers,
            LINK_COUNT + 1, spanSerial, LINK_COUNT) == 0);
        CHECK(serialTeam == 1 && uniqueWorkers(serialWorkers) == 1);
        for (i = 0; i < LINK_COUNT; ++i)
        {
            CHECK(spanOmp[i].total == fullCore[i].total &&
                  spanOmp[i].core_count == fullCore[i].core_count &&
                  spanOmp[i].downstream_boundary ==
                      fullCore[i].downstream_boundary &&
                  spanOmp[i].upstream_boundary ==
                      fullCore[i].upstream_boundary &&
                  spanOmp[i].first_core_slot ==
                      fullCore[i].first_core_slot &&
                  spanOmp[i].last_core_slot ==
                      fullCore[i].last_core_slot &&
                  spanOmp[i].orient == fullCore[i].orient &&
                  spanOmp[i].first_core_id == fullCore[i].first_core_id &&
                  spanOmp[i].last_core_id == fullCore[i].last_core_id);
            CHECK(spanSerial[i].total == fullCore[i].total &&
                  spanSerial[i].core_count == fullCore[i].core_count &&
                  spanSerial[i].downstream_boundary ==
                      fullCore[i].downstream_boundary &&
                  spanSerial[i].upstream_boundary ==
                      fullCore[i].upstream_boundary &&
                  spanSerial[i].first_core_id == fullCore[i].first_core_id &&
                  spanSerial[i].last_core_id == fullCore[i].last_core_id);
        }

        /* A stale but otherwise legal certification falls back to FULL and
           must preserve the same snapshot. */
        CHECK(MSXsegStorage_testResidentPublishFailure(1) ==
              ERR_PIPE_RING_CAPACITY);
        CHECK(MSXsegStorage_testResidentScanOMP(
            MSX_RESIDENT_SCAN_SPAN_SERIAL, &serialTeam, serialWorkers,
            LINK_COUNT + 1, spanSerial, LINK_COUNT) == 0);
        CHECK(spanSerial[0].total == fullCore[0].total &&
              spanSerial[0].core_count == fullCore[0].core_count &&
              spanSerial[0].first_core_id == fullCore[0].first_core_id &&
              spanSerial[0].last_core_id == fullCore[0].last_core_id);
        CHECK(MSXsegStorage_testResidentScanAndInitialize(
            MSX_RESIDENT_SCAN_SPAN_SERIAL, &serialTeam) == 0 &&
              serialTeam == 1);
        CHECK(MSXsegStorage_hybridAuditCoreSpan(1) == 0);

        /* A certified endpoint contradiction is not a silent fallback. */
        {
            Pseg corrupt = MSXsegStorage_hybridCoreSegAt(1, 0);
            CHECK(corrupt != NULL);
            if (corrupt) corrupt->inHybridCore = FALSE;
            MSX.ErrCode = 0;
            CHECK(MSXsegStorage_testResidentScanAndInitialize(
                MSX_RESIDENT_SCAN_SPAN_SERIAL, &serialTeam) ==
                  ERR_PIPE_RING_CAPACITY &&
                  MSX.ErrCode == ERR_PIPE_RING_CAPACITY);
            if (corrupt) corrupt->inHybridCore = TRUE;
            CHECK(MSXsegStorage_hybridAuditCoreSpan(1) == 0);
        }

        /* Reverse orientation must preserve all snapshot identity fields. */
        for (k = 1; k <= LINK_COUNT; ++k)
            reverseTopology(k);
        for (k = 1; k <= LINK_COUNT; ++k)
            CHECK(MSXsegStorage_hybridAfterListReorder(k) == 0);
        CHECK(MSXsegStorage_testResidentScanOMP(
            MSX_RESIDENT_SCAN_FULL_SERIAL, &reverseTeam, serialWorkers,
            LINK_COUNT + 1, reverseFull, LINK_COUNT) == 0 &&
              reverseTeam == 1);
    CHECK(MSXsegStorage_testResidentScanOMP(
            MSX_RESIDENT_SCAN_SPAN_OMP8, &reverseTeam, reverseWorkers,
            LINK_COUNT + 1, reverseSpan, LINK_COUNT) == 0 &&
              reverseTeam == 8 && uniqueWorkers(reverseWorkers) >= 2);
        for (i = 0; i < LINK_COUNT; ++i)
            CHECK(reverseSpan[i].total == reverseFull[i].total &&
                  reverseSpan[i].core_count == reverseFull[i].core_count &&
                  reverseSpan[i].downstream_boundary ==
                      reverseFull[i].downstream_boundary &&
                  reverseSpan[i].upstream_boundary ==
                      reverseFull[i].upstream_boundary &&
                  reverseSpan[i].orient == -1 &&
                  reverseSpan[i].first_core_id == reverseFull[i].first_core_id &&
                  reverseSpan[i].last_core_id == reverseFull[i].last_core_id);
    }

    /* Drop boundary ownership and detach the Core views before storage close.
       The no-Resident harness deliberately has no metadata rows for
       hybridClear's Resident mapping preflight; storage close owns the Core
       view objects. */
    for (i = 1; i <= LINK_COUNT; ++i)
    {
        Pseg seg = MSX.FirstSeg[i], next;
        while (seg)
        {
            next = seg->prev;
            if (!seg->inHybridCore) MSXqual_removeSeg(seg);
            seg = next;
        }
        MSX.FirstSeg[i] = MSX.LastSeg[i] = NULL;
        MSX.Link[i].nsegs = 0;
    }

    printf("team_size=%d\nworkers=%d\nnonempty_links=%d\n"
           "serial_team_size=%d\nassertions_passed=%d\n"
           "assertions_failed=%d\n", parallelTeam,
           uniqueWorkers(parallelWorkers), LINK_COUNT, serialTeam,
           passed, failed);
    MSXsegStorage_close();
    _putenv_s("MSX_RESIDENT_SCAN_AUDIT", "");
    _putenv_s("MSX_RESIDENT_SCAN_MODE", "");
    remove("resident_scan_team_size.txt");
    remove("resident_scan_omp_team_size.txt");
    freeTopology();
    return failed ? 1 : 0;
}
