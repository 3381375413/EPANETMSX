/******************************************************************************
**  MODULE:        MSXSEGMENT_PROFILE.H
**  DESCRIPTION:   Read-only segment topology profiler for PSEG validation.
******************************************************************************/

#ifndef MSXSEGMENT_PROFILE_H
#define MSXSEGMENT_PROFILE_H

#include "msxtypes.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Event kinds are intentionally small and stable because they are also used
   by the CSV summary produced by this module. */
enum
{
    MSX_PROFILE_DOWNSTREAM_COMPLETE_DELETE = 1,
    MSX_PROFILE_DOWNSTREAM_PARTIAL_CONSUME = 2,
    MSX_PROFILE_UPSTREAM_NEW_SEGMENT       = 3,
    MSX_PROFILE_UPSTREAM_MERGE             = 4,
    MSX_PROFILE_FLOW_REVERSAL              = 5
};

int  MSXsegProfile_open(void);
void MSXsegProfile_reset(void);
void MSXsegProfile_close(void);
int  MSXsegProfile_enabled(void);
void MSXsegProfile_beginStep(double sim_time_sec);
void MSXsegProfile_endStep(double sim_time_sec);
void MSXsegProfile_reactVisitsForLink(int k, int count);
void MSXsegProfile_event(int k, int event_kind);
void MSXsegProfile_segmentAllocated(Pseg seg);

#ifdef __cplusplus
}
#endif

#endif
