#ifndef MSX_BOUNDARY_VISIT_H
#define MSX_BOUNDARY_VISIT_H
#include "msxtypes.h"
#include <stdint.h>

/* The caller obtains head/tail/count from a certified partition borrowed
   inside its read window. This routine does not establish that certificate.
   NULL certificate selects the original FirstSeg/prev traversal. */
typedef struct {
    Pseg head, tail;
    uint64_t count;
} MSXBoundaryCoreInterval;
typedef int (*MSXBoundaryReaction)(Pseg, void *);

static inline int MSXboundary_visit(Pseg first,
    const MSXBoundaryCoreInterval *certified, MSXBoundaryReaction reaction,
    void *context, uint64_t *cpuRows, uint64_t *coreWalked)
{
    Pseg seg = first;
    int err;
    if (!reaction) return ERR_GPU_SEGMENT_PACK_FAILED;
    if (!certified) {
        while (seg) {
            if (seg->inHybridCore) {
                if (coreWalked) ++*coreWalked;
            } else {
                err = reaction(seg, context);
                if (err) return err;
                if (cpuRows) ++*cpuRows;
            }
            seg = seg->prev;
        }
        return 0;
    }
    if ((certified->count && (!certified->head || !certified->tail)) ||
        (!certified->count && (certified->head || certified->tail)))
        return ERR_GPU_SEGMENT_PACK_FAILED;
    if (certified->count && (!certified->head->inHybridCore || !certified->tail->inHybridCore))
        return ERR_GPU_SEGMENT_PACK_FAILED;
    while (seg && seg != certified->head) {
        if (seg->inHybridCore) return ERR_GPU_SEGMENT_PACK_FAILED;
        err = reaction(seg, context);
        if (err) return err;
        if (cpuRows) ++*cpuRows;
        seg = seg->prev;
    }
    if (certified->count) {
        if (!seg) return ERR_GPU_SEGMENT_PACK_FAILED;
        seg = certified->tail->prev;
    }
    while (seg) {
        if (seg->inHybridCore) return ERR_GPU_SEGMENT_PACK_FAILED;
        err = reaction(seg, context);
        if (err) return err;
        if (cpuRows) ++*cpuRows;
        seg = seg->prev;
    }
    return 0;
}
#endif
