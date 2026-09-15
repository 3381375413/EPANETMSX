/******************************************************************************
**  MODULE:        MSXSEGMENT_STORAGE.H
**  PROJECT:       EPANET-MSX GPU modified
**  DESCRIPTION:   Optional pipe-local segment concentration storage.
******************************************************************************/

#ifndef MSXSEGMENT_STORAGE_H
#define MSXSEGMENT_STORAGE_H

#include <stddef.h>
#include "msxtypes.h"

#ifdef __cplusplus
extern "C" {
#endif

int  MSXsegStorage_open(void);
void MSXsegStorage_close(void);
void MSXsegStorage_reset(void);

int  MSXsegStorage_preparePrivate(Pseg seg);
void MSXsegStorage_initPrivateValues(Pseg seg, const double c[]);
int  MSXsegStorage_bindPipeSegment(int k, Pseg seg);
void MSXsegStorage_unbindSegment(Pseg seg);

int  MSXsegStorage_isPipeRingEnabled(void);
int  MSXsegStorage_isPipeRingLink(int k);
int  MSXsegStorage_isPipeRingSegment(Pseg seg);

int  MSXsegStorage_pipeAppendTail(int k, Pseg seg);
Pseg MSXsegStorage_pipePeekHead(int k);
Pseg MSXsegStorage_pipePeekTail(int k);
int  MSXsegStorage_pipePopHead(int k);
int  MSXsegStorage_pipeConsumeHead(int k, double volume);
int  MSXsegStorage_pipeMergeTail(int k, const double upnodeQual[], double volume);
int  MSXsegStorage_pipeReverse(int k);
void MSXsegStorage_pipeClear(int k);
int  MSXsegStorage_pipeCount(int k);
Pseg MSXsegStorage_pipeSegFromHead(int k, int pos);
Pseg MSXsegStorage_pipeSegFromTail(int k, int pos);
int  MSXsegStorage_pipeSlotFromHead(int k, int pos);
int  MSXsegStorage_pipeSlotFromTail(int k, int pos);
double *MSXsegStorage_pipeC(int k, int slot);
double *MSXsegStorage_pipeLastC(int k, int slot);
double *MSXsegStorage_pipeVPtr(int k, int slot);
double *MSXsegStorage_pipeHstepPtr(int k, int slot);
double *MSXsegStorage_pipeHresponsePtr(int k, int slot);
double *MSXsegStorage_pipeUresponsePtr(int k, int slot);
double *MSXsegStorage_pipeDresponsePtr(int k, int slot);

void MSXsegStorage_syncPsegMirror(int k);
void MSXsegStorage_syncAllPsegMirrors(void);
void MSXsegStorage_syncScalarsFromPseg(int k);
void MSXsegStorage_syncAllScalarsFromPseg(void);
int  MSXsegStorage_validate(int k);
int  MSXsegStorage_validateAll(void);
int  MSXsegStorage_ringCapacity(void);
int  MSXsegStorage_ringStride(void);
int  MSXsegStorage_ringLinkCount(void);
size_t MSXsegStorage_ringSlotCount(void);
int  MSXsegStorage_pipeFlatIndex(int k, int slot);
double *MSXsegStorage_ringCData(void);
double *MSXsegStorage_ringLastCData(void);
double *MSXsegStorage_ringHstepData(void);

#ifdef __cplusplus
}
#endif

#endif
