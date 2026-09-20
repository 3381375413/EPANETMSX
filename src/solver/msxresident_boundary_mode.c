#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "msxresident_boundary_mode.h"

static int ResidentBoundaryModeState = -1;

void MSXresidentBoundaryModeReset(void)
{
    ResidentBoundaryModeState = -1;
}

int MSXresidentBoundaryMode(void)
{
    const char *value;
    if (ResidentBoundaryModeState >= 0)
        return ResidentBoundaryModeState;
    ResidentBoundaryModeState = MSX_RESIDENT_BOUNDARY_SCAN_FULL;
    value = getenv("MSX_RESIDENT_BOUNDARY_SCAN_MODE");
    if (value && _stricmp(value, "SPAN") == 0)
        ResidentBoundaryModeState = MSX_RESIDENT_BOUNDARY_SCAN_SPAN;
    return ResidentBoundaryModeState;
}

const char *MSXresidentBoundaryModeName(int mode)
{
    return mode == MSX_RESIDENT_BOUNDARY_SCAN_SPAN ? "SPAN" : "FULL";
}
