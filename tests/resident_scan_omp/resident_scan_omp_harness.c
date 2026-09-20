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
            seg->next = previous;
            if (previous) previous->prev = seg;
            else MSX.FirstSeg[k] = seg;
            previous = seg;
            MSX.LastSeg[k] = seg;
            ++MSX.Link[k].nsegs;
        }
    }
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
