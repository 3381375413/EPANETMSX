#ifndef MSXREACT_TRANSFER_CUDA_H
#define MSXREACT_TRANSFER_CUDA_H

#include <stddef.h>

#include "msxtypes.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned long long MSXGpuDevicePtr;

typedef struct
{
    int nSeg;
    int nLinks;
    int nSpecies;
    int nActiveLinks;
    int ringResident;
    int ringSlotCount;
    int ringStride;

    int *segRow;
    int *segPipe;
    int *activeLink;
    int *pipeSegOffset;
    int *pipeSegCount;
    double *segVol;
    double *hstep;
    Pseg *segPtrs;
    int *unpackOrder;
    double *c;
    double *cOde;
    double *hyd;
    double *reacted;
    int *rk5Nfcn;
    int *rk5Naccpt;
    int *rk5Nrejct;
    int *rk5Err;
    double *rk5LastHstep;
    int *ros2Nfcn;
    int *ros2Njac;
    int *ros2Naccept;
    int *ros2Nreject;
    int *ros2Err;
    double *ros2LastHstep;

    MSXGpuDevicePtr d_segPipe;
    MSXGpuDevicePtr d_segRow;
    MSXGpuDevicePtr d_activeLink;
    MSXGpuDevicePtr d_pipeSegOffset;
    MSXGpuDevicePtr d_pipeSegCount;
    MSXGpuDevicePtr d_segVol;
    MSXGpuDevicePtr d_hstep;
    MSXGpuDevicePtr d_c;
    MSXGpuDevicePtr d_cOde;
    MSXGpuDevicePtr d_hyd;
    MSXGpuDevicePtr d_reacted;
    MSXGpuDevicePtr d_rk5Nfcn;
    MSXGpuDevicePtr d_rk5Naccpt;
    MSXGpuDevicePtr d_rk5Nrejct;
    MSXGpuDevicePtr d_rk5Err;
    MSXGpuDevicePtr d_rk5LastHstep;
    MSXGpuDevicePtr d_ros2Nfcn;
    MSXGpuDevicePtr d_ros2Njac;
    MSXGpuDevicePtr d_ros2Naccept;
    MSXGpuDevicePtr d_ros2Nreject;
    MSXGpuDevicePtr d_ros2Err;
    MSXGpuDevicePtr d_ros2LastHstep;
    MSXGpuDevicePtr d_err;
} MSXReactTransferView;

int MSXreactTransfer_init(char *errmsg, int errmsgLen);
void MSXreactTransfer_close(void);
int MSXreactTransfer_countAndPack(double dt, int nSpecies, MSXReactTransferView *view);
int MSXreactTransfer_ringView(double dt, int nSpecies, MSXReactTransferView *view);
int MSXreactTransfer_upload(MSXReactTransferView *view, const void *zeroErr, size_t errSize);
int MSXreactTransfer_download(MSXReactTransferView *view, void *hostErr, size_t errSize);
int MSXreactTransfer_unpack(MSXReactTransferView *view);

#ifdef __cplusplus
}
#endif

#endif
