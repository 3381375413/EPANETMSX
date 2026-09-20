#ifndef MSXRESIDENT_BOUNDARY_MODE_H
#define MSXRESIDENT_BOUNDARY_MODE_H

#include "msxsegment_storage.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Internal chemistry selector; it is reset at each MSXchem open lifetime. */
void MSXresidentBoundaryModeReset(void);
int  MSXresidentBoundaryMode(void);
const char *MSXresidentBoundaryModeName(int mode);

#ifdef __cplusplus
}
#endif

#endif
