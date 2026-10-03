#ifndef MSXQUAL_SHARED_H
#define MSXQUAL_SHARED_H
#include <math.h>
#include "msxtypes.h"
#define MSX_Q_STAGNANT (0.005 / GPMperCFS)
static double MSXqual_effectiveFlow(REAL4 q)
{ return fabs((double)q) < MSX_Q_STAGNANT ? 0.0 : (double)q; }
static double MSXqual_pipeVolume(const Slink *link)
{ return 0.785398 * link->len * (link->diam * link->diam); }
#endif
