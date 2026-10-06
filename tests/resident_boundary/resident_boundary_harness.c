#include <stdio.h>
#include <string.h>
#include "msxboundary_visit.h"

static int passed, failed;
#define CHECK(x) do { if (x) ++passed; else { ++failed; fprintf(stderr,"FAIL line %d: %s\n", __LINE__, #x); } } while (0)
typedef struct { uint64_t ids[32]; unsigned n, failAt; } Output;
static int react(Pseg seg, void *ctx)
{
    Output *o = (Output *)ctx;
    if (o->failAt && o->n + 1 == o->failAt) return 777;
    o->ids[o->n++] = seg->hybridId;
    return 0;
}

/* Independent reference: direct FirstSeg/prev, no span accessor/count/view. */
static unsigned oracle(Pseg first, uint64_t *ids)
{
    unsigned n = 0;
    while (first) {
        if (!first->inHybridCore) ids[n++] = first->hybridId;
        first = first->prev;
    }
    return n;
}

static void sequence_case(unsigned n, unsigned start, unsigned count, int reverse, unsigned failAt)
{
    struct Sseg pool[32];
    Pseg ordered[32], first;
    MSXBoundaryCoreInterval interval = {0};
    Output full = {{0},0,failAt}, fast = {{0},0,failAt};
    uint64_t reference[32], fullCpu = 0, fullCore = 0, fastCpu = 0, fastCore = 0;
    unsigned i, referenceN;
    int fullErr, fastErr;
    memset(pool, 0, sizeof(pool));
    /* Nonlinear storage order models ring wrap and reused addresses. */
    for (i = 0; i < n; ++i) {
        ordered[i] = &pool[(i * 7) % 32];
        ordered[i]->hybridId = UINT64_C(1000) + (reverse ? n - i : i);
        ordered[i]->inHybridCore = i >= start && i < start + count;
        ordered[i]->v = i == 0 ? 0.0 : (double)i;
    }
    for (i = 0; i < n; ++i) {
        ordered[i]->prev = i + 1 < n ? ordered[i+1] : NULL;
        ordered[i]->next = i ? ordered[i-1] : NULL;
    }
    first = n ? ordered[0] : NULL;
    if (count) { interval.head = ordered[start]; interval.tail = ordered[start+count-1]; interval.count = count; }
    referenceN = oracle(first, reference);
    fullErr = MSXboundary_visit(first,NULL,react,&full,&fullCpu,&fullCore);
    fastErr = MSXboundary_visit(first,&interval,react,&fast,&fastCpu,&fastCore);
    CHECK(fullErr == fastErr);
    CHECK(full.n == fast.n && !memcmp(full.ids,fast.ids,full.n*sizeof(full.ids[0])));
    CHECK(fastCpu == fast.n && fullCpu == full.n);
    CHECK(fastCore == 0);
    CHECK(full.n <= referenceN && !memcmp(full.ids,reference,full.n*sizeof(reference[0])));
    if (!fullErr) { CHECK(full.n == referenceN); CHECK(fullCore == count); }
}

int main(void)
{
    unsigned n, start, count, failure;
    int direction;
    {
        struct Sseg cpu = {0}, core = {0};
        Output output = {{0},0,0};
        MSXBoundaryCoreInterval bad = {0};
        uint64_t rows = 0, walked = 0;
        cpu.hybridId = 11; core.hybridId = 22; core.inHybridCore = 1;
        CHECK(MSXboundary_visit(NULL,NULL,NULL,NULL,NULL,NULL) == ERR_GPU_SEGMENT_PACK_FAILED);
        bad.count = 1;
        CHECK(MSXboundary_visit(&cpu,&bad,react,&output,&rows,&walked) == ERR_GPU_SEGMENT_PACK_FAILED);
        bad.head = bad.tail = &cpu;
        CHECK(MSXboundary_visit(&cpu,&bad,react,&output,&rows,&walked) == ERR_GPU_SEGMENT_PACK_FAILED);
        bad.head = bad.tail = &core;
        CHECK(MSXboundary_visit(&cpu,&bad,react,&output,&rows,&walked) == ERR_GPU_SEGMENT_PACK_FAILED);
        CHECK(output.n == 1 && output.ids[0] == 11); /* matched successful prefix */
        bad.count = 0;
        CHECK(MSXboundary_visit(&cpu,&bad,react,&output,&rows,&walked) == ERR_GPU_SEGMENT_PACK_FAILED);
        bad.head = bad.tail = NULL;
        CHECK(MSXboundary_visit(&core,&bad,react,&output,&rows,&walked) == ERR_GPU_SEGMENT_PACK_FAILED);
    }
    sequence_case(0,0,0,0,0);
    for (n = 1; n <= 16; ++n)
        for (start = 0; start <= n; ++start)
            for (count = 0; count <= n-start; ++count)
                for (direction = 0; direction < 2; ++direction)
                    for (failure = 0; failure <= n-count; ++failure)
                        sequence_case(n,start,count,direction,failure);
    printf("BOUNDARY_RESULT passed=%d failed=%d\n",passed,failed);
    return failed ? 1 : 0;
}
