#include <cuda_runtime.h>

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOGDI
#include <windows.h>
#else
#include <time.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include "msxreact_transfer_cuda.h"
#include "msxsegment_storage.h"

extern "C" MSXproject MSX;

#define LINKVOL_GPU(k) (0.785398 * MSX.Link[(k)].len * SQR(MSX.Link[(k)].diam))

typedef struct
{
    int nLinksCap;
    int activeCap;
    int nSegCap;
    int cValueCap;
    int hydValueCap;
    int reactedValueCap;

    int *activeFlagByLink;
    int *countByLink;
    int *activeLink;
    int *segOffset;
    int *segCount;

    int *segRow;
    int *segPipe;
    double *segVol;
    double *hstep;
    Pseg *segPtrs;
    int *unpackOrder;
    double *c;
    double *cOde;
    double *hyd;
    double *reacted;
    int *unpackSpecies;
    int unpackSpeciesCap;
    int unpackSpeciesCount;
    int unpackSpeciesVersion;
    int *rk5Nfcn;
    int *rk5Naccpt;
    int *rk5Nrejct;
    int *rk5Err;
    double *rk5LastHstep;

    int deviceNSegCap;
    int deviceActiveCap;
    int deviceCValueCap;
    int deviceHydValueCap;
    int deviceReactedValueCap;
    int deviceLinksSpeciesCap;

    int *d_segPipe;
    int *d_segRow;
    int *d_activeLink;
    int *d_pipeSegOffset;
    int *d_pipeSegCount;
    double *d_segVol;
    double *d_hstep;
    double *d_c;
    double *d_cOde;
    double *d_hyd;
    double *d_reacted;
    int *d_rk5Nfcn;
    int *d_rk5Naccpt;
    int *d_rk5Nrejct;
    int *d_rk5Err;
    double *d_rk5LastHstep;
    void *d_err;
    size_t errCap;
} ReactTransferWorkspace;

static ReactTransferWorkspace Ws;

static int buildUnpackSpeciesList(int nSpecies)
{
    int m;
    int count = 0;
    int version = 0;
    int *next;

    if (Ws.unpackSpeciesVersion == nSpecies && Ws.unpackSpecies && Ws.unpackSpeciesCap >= nSpecies)
        return 0;

    if (nSpecies > Ws.unpackSpeciesCap)
    {
        next = (int *)realloc(Ws.unpackSpecies, (size_t)nSpecies * sizeof(int));
        if (!next) return 1;
        Ws.unpackSpecies = next;
        Ws.unpackSpeciesCap = nSpecies;
    }

    for (m = 1; m <= nSpecies; m++)
    {
        int exprType = MSX.Species[m].pipeExprType;
        if (exprType == RATE || exprType == EQUIL || exprType == FORMULA)
            Ws.unpackSpecies[count++] = m;
    }

    version = nSpecies;
    Ws.unpackSpeciesCount = count;
    Ws.unpackSpeciesVersion = version;
    return 0;
}

#ifdef _OPENMP
static int unpackOpenmpEnabled(void)
{
    const char *value = getenv("MSX_GPU_UNPACK_OPENMP");
    if (!value) return 0;
    return strcmp(value, "1") == 0 || strcmp(value, "YES") == 0 || strcmp(value, "yes") == 0 ||
           strcmp(value, "TRUE") == 0 || strcmp(value, "true") == 0;
}
#endif

static double hostWallTimeMs(void)
{
#ifdef _WIN32
    LARGE_INTEGER counter, frequency;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return 1000.0 * (double)counter.QuadPart / (double)frequency.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1000.0 * (double)ts.tv_sec + (double)ts.tv_nsec / 1000000.0;
#endif
}

static int nextCapacity(int current, int required)
{
    int capacity = current > 0 ? current : 1024;
    while (capacity < required)
    {
        if (capacity > INT_MAX / 2) return required;
        capacity *= 2;
    }
    return capacity;
}

static int growPinned(void **ptr, size_t oldBytes, size_t newBytes)
{
    void *next = NULL;
    cudaError_t err;
    if (newBytes == 0) return 0;
    err = cudaMallocHost(&next, newBytes);
    if (err != cudaSuccess) return 1;
    if (*ptr)
    {
        size_t copyBytes = oldBytes < newBytes ? oldBytes : newBytes;
        if (copyBytes > 0) memcpy(next, *ptr, copyBytes);
        cudaFreeHost(*ptr);
    }
    *ptr = next;
    return 0;
}

static int growPinnedInt(int **ptr, int oldCap, int newCap)
{
    return growPinned((void **)ptr, (size_t)oldCap * sizeof(int), (size_t)newCap * sizeof(int));
}

static int growPinnedDouble(double **ptr, int oldCap, int newCap)
{
    return growPinned((void **)ptr, (size_t)oldCap * sizeof(double), (size_t)newCap * sizeof(double));
}

static int ensureLinkCapacity(int nLinks)
{
    int oldCap = Ws.nLinksCap;
    int cap;
    int *activeNext;
    int *countNext;
    if (nLinks + 1 <= oldCap) return 0;
    cap = nextCapacity(oldCap, nLinks + 1);
    activeNext = (int *)realloc(Ws.activeFlagByLink, (size_t)cap * sizeof(int));
    if (!activeNext) return 1;
    Ws.activeFlagByLink = activeNext;
    countNext = (int *)realloc(Ws.countByLink, (size_t)cap * sizeof(int));
    if (!countNext) return 1;
    Ws.countByLink = countNext;
    Ws.nLinksCap = cap;
    return 0;
}

static int ensureActiveCapacity(int activeCount)
{
    int newCap;
    if (activeCount <= Ws.activeCap) return 0;
    newCap = nextCapacity(Ws.activeCap, activeCount);
    if (growPinnedInt(&Ws.activeLink, Ws.activeCap, newCap)) return 1;
    if (growPinnedInt(&Ws.segOffset, Ws.activeCap, newCap)) return 1;
    if (growPinnedInt(&Ws.segCount, Ws.activeCap, newCap)) return 1;
    Ws.activeCap = newCap;
    return 0;
}

static int ensureSegmentCapacity(int nSeg, int nSpecies)
{
    int newSegCap = Ws.nSegCap;
    int cValues;
    int hydValues;
    if (nSeg > newSegCap)
    {
        int oldSegCap = Ws.nSegCap;
        newSegCap = nextCapacity(Ws.nSegCap, nSeg);
        Pseg *nextSegPtrs;
        if (growPinnedInt(&Ws.segRow, oldSegCap, newSegCap)) return 1;
        if (growPinnedInt(&Ws.segPipe, oldSegCap, newSegCap)) return 1;
        if (growPinnedDouble(&Ws.segVol, oldSegCap, newSegCap)) return 1;
        if (growPinnedDouble(&Ws.hstep, oldSegCap, newSegCap)) return 1;
        nextSegPtrs = (Pseg *)realloc(Ws.segPtrs, (size_t)newSegCap * sizeof(Pseg));
        if (!nextSegPtrs) return 1;
        Ws.segPtrs = nextSegPtrs;
        if (growPinnedInt(&Ws.unpackOrder, oldSegCap, newSegCap)) return 1;
        if (growPinnedInt(&Ws.rk5Nfcn, oldSegCap, newSegCap)) return 1;
        if (growPinnedInt(&Ws.rk5Naccpt, oldSegCap, newSegCap)) return 1;
        if (growPinnedInt(&Ws.rk5Nrejct, oldSegCap, newSegCap)) return 1;
        if (growPinnedInt(&Ws.rk5Err, oldSegCap, newSegCap)) return 1;
        if (growPinnedDouble(&Ws.rk5LastHstep, oldSegCap, newSegCap)) return 1;
        Ws.nSegCap = newSegCap;
    }
    cValues = Ws.nSegCap * (nSpecies + 1);
    hydValues = Ws.nSegCap * MAX_HYD_VARS;
    if (cValues > Ws.cValueCap)
    {
        if (growPinnedDouble(&Ws.c, Ws.cValueCap, cValues)) return 1;
        if (growPinnedDouble(&Ws.cOde, Ws.cValueCap, cValues)) return 1;
        Ws.cValueCap = cValues;
    }
    if (hydValues > Ws.hydValueCap)
    {
        if (growPinnedDouble(&Ws.hyd, Ws.hydValueCap, hydValues)) return 1;
        Ws.hydValueCap = hydValues;
    }
    return 0;
}

static int ensureReactedCapacity(int nLinks, int nSpecies)
{
    int values = (nLinks + 1) * (nSpecies + 1);
    if (values <= Ws.reactedValueCap) return 0;
    if (growPinnedDouble(&Ws.reacted, Ws.reactedValueCap, values)) return 1;
    Ws.reactedValueCap = values;
    return 0;
}

static int ensureDevice(MSXReactTransferView *view, size_t errSize)
{
    int segValues = view->nSeg > 0 ? view->nSeg : 1;
    int cValues = view->ringResident ? view->ringSlotCount * view->ringStride : segValues * (view->nSpecies + 1);
    int hydValues = segValues * MAX_HYD_VARS;
    int reactedValues = (view->nLinks + 1) * (view->nSpecies + 1);
    double timer = hostWallTimeMs();
    if (segValues > Ws.deviceNSegCap)
    {
        int newCap = nextCapacity(Ws.deviceNSegCap, segValues);
        if (Ws.d_segRow) cudaFree(Ws.d_segRow);
        if (Ws.d_segPipe) cudaFree(Ws.d_segPipe);
        if (Ws.d_segVol) cudaFree(Ws.d_segVol);
        if (Ws.d_hstep) cudaFree(Ws.d_hstep);
        if (Ws.d_rk5Nfcn) cudaFree(Ws.d_rk5Nfcn);
        if (Ws.d_rk5Naccpt) cudaFree(Ws.d_rk5Naccpt);
        if (Ws.d_rk5Nrejct) cudaFree(Ws.d_rk5Nrejct);
        if (Ws.d_rk5Err) cudaFree(Ws.d_rk5Err);
        if (Ws.d_rk5LastHstep) cudaFree(Ws.d_rk5LastHstep);
        Ws.d_segRow = NULL; Ws.d_segPipe = NULL; Ws.d_segVol = NULL; Ws.d_hstep = NULL;
        Ws.d_rk5Nfcn = NULL; Ws.d_rk5Naccpt = NULL; Ws.d_rk5Nrejct = NULL;
        Ws.d_rk5Err = NULL; Ws.d_rk5LastHstep = NULL;
        if (cudaMalloc((void **)&Ws.d_segRow, (size_t)newCap * sizeof(int)) != cudaSuccess) return 1;
        if (cudaMalloc((void **)&Ws.d_segPipe, (size_t)newCap * sizeof(int)) != cudaSuccess) return 1;
        if (cudaMalloc((void **)&Ws.d_segVol, (size_t)newCap * sizeof(double)) != cudaSuccess) return 1;
        if (cudaMalloc((void **)&Ws.d_hstep, (size_t)newCap * sizeof(double)) != cudaSuccess) return 1;
        if (cudaMalloc((void **)&Ws.d_rk5Nfcn, (size_t)newCap * sizeof(int)) != cudaSuccess) return 1;
        if (cudaMalloc((void **)&Ws.d_rk5Naccpt, (size_t)newCap * sizeof(int)) != cudaSuccess) return 1;
        if (cudaMalloc((void **)&Ws.d_rk5Nrejct, (size_t)newCap * sizeof(int)) != cudaSuccess) return 1;
        if (cudaMalloc((void **)&Ws.d_rk5Err, (size_t)newCap * sizeof(int)) != cudaSuccess) return 1;
        if (cudaMalloc((void **)&Ws.d_rk5LastHstep, (size_t)newCap * sizeof(double)) != cudaSuccess) return 1;
        Ws.deviceNSegCap = newCap;
    }
    if (view->nActiveLinks > Ws.deviceActiveCap)
    {
        int newCap = nextCapacity(Ws.deviceActiveCap, view->nActiveLinks);
        if (Ws.d_activeLink) cudaFree(Ws.d_activeLink);
        if (Ws.d_pipeSegOffset) cudaFree(Ws.d_pipeSegOffset);
        if (Ws.d_pipeSegCount) cudaFree(Ws.d_pipeSegCount);
        Ws.d_activeLink = NULL;
        Ws.d_pipeSegOffset = NULL;
        Ws.d_pipeSegCount = NULL;
        if (cudaMalloc((void **)&Ws.d_activeLink, (size_t)newCap * sizeof(int)) != cudaSuccess) return 1;
        if (cudaMalloc((void **)&Ws.d_pipeSegOffset, (size_t)newCap * sizeof(int)) != cudaSuccess) return 1;
        if (cudaMalloc((void **)&Ws.d_pipeSegCount, (size_t)newCap * sizeof(int)) != cudaSuccess) return 1;
        Ws.deviceActiveCap = newCap;
    }
    if (cValues > Ws.deviceCValueCap)
    {
        int newCap = nextCapacity(Ws.deviceCValueCap, cValues);
        if (Ws.d_c) cudaFree(Ws.d_c);
        if (Ws.d_cOde) cudaFree(Ws.d_cOde);
        Ws.d_c = NULL; Ws.d_cOde = NULL;
        if (cudaMalloc((void **)&Ws.d_c, (size_t)newCap * sizeof(double)) != cudaSuccess) return 1;
        if (cudaMalloc((void **)&Ws.d_cOde, (size_t)newCap * sizeof(double)) != cudaSuccess) return 1;
        Ws.deviceCValueCap = newCap;
    }
    if (hydValues > Ws.deviceHydValueCap)
    {
        int newCap = nextCapacity(Ws.deviceHydValueCap, hydValues);
        if (Ws.d_hyd) cudaFree(Ws.d_hyd);
        Ws.d_hyd = NULL;
        if (cudaMalloc((void **)&Ws.d_hyd, (size_t)newCap * sizeof(double)) != cudaSuccess) return 1;
        Ws.deviceHydValueCap = newCap;
    }
    if (reactedValues > Ws.deviceReactedValueCap)
    {
        int newCap = nextCapacity(Ws.deviceReactedValueCap, reactedValues);
        if (Ws.d_reacted) cudaFree(Ws.d_reacted);
        Ws.d_reacted = NULL;
        if (cudaMalloc((void **)&Ws.d_reacted, (size_t)newCap * sizeof(double)) != cudaSuccess) return 1;
        Ws.deviceReactedValueCap = newCap;
    }
    if (errSize > Ws.errCap)
    {
        if (Ws.d_err) cudaFree(Ws.d_err);
        if (cudaMalloc(&Ws.d_err, errSize) != cudaSuccess)
        {
            Ws.d_err = NULL;
            Ws.errCap = 0;
            return 1;
        }
        Ws.errCap = errSize;
    }
    MSX.GpuTimingRecord.react_device_alloc_ms += hostWallTimeMs() - timer;
    view->d_segRow = (MSXGpuDevicePtr)(size_t)Ws.d_segRow;
    view->d_segPipe = (MSXGpuDevicePtr)(size_t)Ws.d_segPipe;
    view->d_activeLink = (MSXGpuDevicePtr)(size_t)Ws.d_activeLink;
    view->d_pipeSegOffset = (MSXGpuDevicePtr)(size_t)Ws.d_pipeSegOffset;
    view->d_pipeSegCount = (MSXGpuDevicePtr)(size_t)Ws.d_pipeSegCount;
    view->d_segVol = (MSXGpuDevicePtr)(size_t)Ws.d_segVol;
    view->d_hstep = (MSXGpuDevicePtr)(size_t)Ws.d_hstep;
    view->d_c = (MSXGpuDevicePtr)(size_t)Ws.d_c;
    view->d_cOde = (MSXGpuDevicePtr)(size_t)Ws.d_cOde;
    view->d_hyd = (MSXGpuDevicePtr)(size_t)Ws.d_hyd;
    view->d_reacted = (MSXGpuDevicePtr)(size_t)Ws.d_reacted;
    view->d_rk5Nfcn = (MSXGpuDevicePtr)(size_t)Ws.d_rk5Nfcn;
    view->d_rk5Naccpt = (MSXGpuDevicePtr)(size_t)Ws.d_rk5Naccpt;
    view->d_rk5Nrejct = (MSXGpuDevicePtr)(size_t)Ws.d_rk5Nrejct;
    view->d_rk5Err = (MSXGpuDevicePtr)(size_t)Ws.d_rk5Err;
    view->d_rk5LastHstep = (MSXGpuDevicePtr)(size_t)Ws.d_rk5LastHstep;
    view->d_err = (MSXGpuDevicePtr)(size_t)Ws.d_err;
    return 0;
}

extern "C" int MSXreactTransfer_init(char *errmsg, int errmsgLen)
{
    int deviceCount = 0;
    cudaError_t err;
    memset(&Ws, 0, sizeof(Ws));
    err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount <= 0)
    {
        if (errmsg && errmsgLen > 0)
        {
            snprintf(errmsg, (size_t)errmsgLen, "CUDA react transfer init failed: %s",
                     err == cudaSuccess ? "no CUDA device found" : cudaGetErrorString(err));
        }
        return 1;
    }
    err = cudaSetDevice(0);
    if (err != cudaSuccess)
    {
        if (errmsg && errmsgLen > 0) snprintf(errmsg, (size_t)errmsgLen, "CUDA react transfer cudaSetDevice failed: %s", cudaGetErrorString(err));
        return 1;
    }
    err = cudaFree(0);
    if (err != cudaSuccess)
    {
        if (errmsg && errmsgLen > 0) snprintf(errmsg, (size_t)errmsgLen, "CUDA react transfer cudaFree(0) failed: %s", cudaGetErrorString(err));
        return 1;
    }
    if (errmsg && errmsgLen > 0) errmsg[0] = '\0';
    return 0;
}

extern "C" void MSXreactTransfer_close(void)
{
    free(Ws.activeFlagByLink);
    free(Ws.countByLink);
    if (Ws.activeLink) cudaFreeHost(Ws.activeLink);
    if (Ws.segOffset) cudaFreeHost(Ws.segOffset);
    if (Ws.segCount) cudaFreeHost(Ws.segCount);
    if (Ws.segRow) cudaFreeHost(Ws.segRow);
    if (Ws.segPipe) cudaFreeHost(Ws.segPipe);
    if (Ws.segVol) cudaFreeHost(Ws.segVol);
    if (Ws.hstep) cudaFreeHost(Ws.hstep);
    free(Ws.segPtrs);
    if (Ws.unpackOrder) cudaFreeHost(Ws.unpackOrder);
    if (Ws.c) cudaFreeHost(Ws.c);
    if (Ws.cOde) cudaFreeHost(Ws.cOde);
    if (Ws.hyd) cudaFreeHost(Ws.hyd);
    if (Ws.reacted) cudaFreeHost(Ws.reacted);
    free(Ws.unpackSpecies);
    if (Ws.rk5Nfcn) cudaFreeHost(Ws.rk5Nfcn);
    if (Ws.rk5Naccpt) cudaFreeHost(Ws.rk5Naccpt);
    if (Ws.rk5Nrejct) cudaFreeHost(Ws.rk5Nrejct);
    if (Ws.rk5Err) cudaFreeHost(Ws.rk5Err);
    if (Ws.rk5LastHstep) cudaFreeHost(Ws.rk5LastHstep);
    if (Ws.d_segRow) cudaFree(Ws.d_segRow);
    if (Ws.d_segPipe) cudaFree(Ws.d_segPipe);
    if (Ws.d_activeLink) cudaFree(Ws.d_activeLink);
    if (Ws.d_pipeSegOffset) cudaFree(Ws.d_pipeSegOffset);
    if (Ws.d_pipeSegCount) cudaFree(Ws.d_pipeSegCount);
    if (Ws.d_segVol) cudaFree(Ws.d_segVol);
    if (Ws.d_hstep) cudaFree(Ws.d_hstep);
    if (Ws.d_c) cudaFree(Ws.d_c);
    if (Ws.d_cOde) cudaFree(Ws.d_cOde);
    if (Ws.d_hyd) cudaFree(Ws.d_hyd);
    if (Ws.d_reacted) cudaFree(Ws.d_reacted);
    if (Ws.d_rk5Nfcn) cudaFree(Ws.d_rk5Nfcn);
    if (Ws.d_rk5Naccpt) cudaFree(Ws.d_rk5Naccpt);
    if (Ws.d_rk5Nrejct) cudaFree(Ws.d_rk5Nrejct);
    if (Ws.d_rk5Err) cudaFree(Ws.d_rk5Err);
    if (Ws.d_rk5LastHstep) cudaFree(Ws.d_rk5LastHstep);
    if (Ws.d_err) cudaFree(Ws.d_err);
    memset(&Ws, 0, sizeof(Ws));
}

extern "C" int MSXreactTransfer_countAndPack(double dt, int nSpecies, MSXReactTransferView *view)
{
    int k;
    int nLinks = MSX.Nobjects[LINK];
    int activeCount = 0;
    int nSeg = 0;
    int badLink = 0;
    int badCount = 0;
    double timer = hostWallTimeMs();
    double packTimer;

    if (!view) return ERR_GPU_SEGMENT_PACK_FAILED;
    memset(view, 0, sizeof(*view));
    if (ensureLinkCapacity(nLinks)) return ERR_GPU_MEMORY_ALLOCATION_FAILED;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 64)
#endif
    for (k = 1; k <= nLinks; k++)
    {
        Pseg seg;
        int count = 0;
        Ws.activeFlagByLink[k] = 0;
        Ws.countByLink[k] = 0;
        if (MSX.Link[k].len == 0.0) continue;
        for (seg = MSX.FirstSeg[k]; seg; seg = seg->prev) count++;
        Ws.countByLink[k] = count;
        if (count != MSX.Link[k].nsegs)
        {
#ifdef _OPENMP
#pragma omp critical
#endif
            {
                if (badLink == 0 || k < badLink)
                {
                    badLink = k;
                    badCount = count;
                }
            }
        }
        if (count > 0) Ws.activeFlagByLink[k] = 1;
    }
    MSX.GpuTimingRecord.react_count_parallel_ms += hostWallTimeMs() - timer;
    if (badLink)
    {
        MSX.GpuError.code = ERR_GPU_SEGMENT_PACK_FAILED;
        MSX.GpuError.pipe = badLink;
        MSX.GpuError.value = (double)badCount;
        return ERR_GPU_SEGMENT_PACK_FAILED;
    }

    timer = hostWallTimeMs();
    for (k = 1; k <= nLinks; k++)
    {
        if (!Ws.activeFlagByLink[k]) continue;
        activeCount++;
        nSeg += Ws.countByLink[k];
    }
    {
        double allocTimer = hostWallTimeMs();
        if (ensureActiveCapacity(activeCount) ||
            ensureSegmentCapacity(nSeg > 0 ? nSeg : 1, nSpecies) ||
            ensureReactedCapacity(nLinks, nSpecies))
        {
            return ERR_GPU_MEMORY_ALLOCATION_FAILED;
        }
        MSX.GpuTimingRecord.react_host_alloc_ms += hostWallTimeMs() - allocTimer;
    }

    timer = hostWallTimeMs();
    activeCount = 0;
    nSeg = 0;
    for (k = 1; k <= nLinks; k++)
    {
        if (!Ws.activeFlagByLink[k]) continue;
        Ws.activeLink[activeCount] = k;
        Ws.segOffset[activeCount] = nSeg;
        Ws.segCount[activeCount] = Ws.countByLink[k];
        nSeg += Ws.countByLink[k];
        activeCount++;
    }
    MSX.GpuTimingRecord.react_count_prefix_ms += hostWallTimeMs() - timer;

    packTimer = hostWallTimeMs();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 64)
#endif
    for (int a = 0; a < activeCount; a++)
    {
        int sid = Ws.segOffset[a];
        int link = Ws.activeLink[a];
        Pseg seg;
        for (seg = MSX.FirstSeg[link]; seg; seg = seg->prev)
        {
            int m;
            Ws.segRow[sid] = sid;
            Ws.segPipe[sid] = link;
            Ws.segVol[sid] = seg->v;
            Ws.hstep[sid] = seg->hstep;
            Ws.segPtrs[sid] = seg;
            Ws.unpackOrder[sid] = sid;
            for (m = 1; m <= nSpecies; m++)
            {
                Ws.c[sid * (nSpecies + 1) + m] = seg->c[m];
                Ws.cOde[sid * (nSpecies + 1) + m] = seg->c[m];
            }
            for (m = 1; m < MAX_HYD_VARS; m++)
                Ws.hyd[sid * MAX_HYD_VARS + m] = MSX.Link[link].HydVar[m];
            sid++;
        }
    }
    memset(Ws.reacted, 0, (size_t)(nLinks + 1) * (nSpecies + 1) * sizeof(double));
    memset(Ws.rk5Nfcn, 0, (size_t)(nSeg > 0 ? nSeg : 1) * sizeof(int));
    memset(Ws.rk5Naccpt, 0, (size_t)(nSeg > 0 ? nSeg : 1) * sizeof(int));
    memset(Ws.rk5Nrejct, 0, (size_t)(nSeg > 0 ? nSeg : 1) * sizeof(int));
    memset(Ws.rk5Err, 0, (size_t)(nSeg > 0 ? nSeg : 1) * sizeof(int));
    memset(Ws.rk5LastHstep, 0, (size_t)(nSeg > 0 ? nSeg : 1) * sizeof(double));
    MSX.GpuTimingRecord.react_pack_segment_ms += hostWallTimeMs() - packTimer;

    view->nSeg = nSeg;
    view->nLinks = nLinks;
    view->nSpecies = nSpecies;
    view->nActiveLinks = activeCount;
    view->ringResident = 0;
    view->ringSlotCount = 0;
    view->ringStride = nSpecies + 1;
    view->segRow = Ws.segRow;
    view->segPipe = Ws.segPipe;
    view->activeLink = Ws.activeLink;
    view->pipeSegOffset = Ws.segOffset;
    view->pipeSegCount = Ws.segCount;
    view->segVol = Ws.segVol;
    view->hstep = Ws.hstep;
    view->segPtrs = Ws.segPtrs;
    view->unpackOrder = Ws.unpackOrder;
    view->c = Ws.c;
    view->cOde = Ws.cOde;
    view->hyd = Ws.hyd;
    view->reacted = Ws.reacted;
    view->rk5Nfcn = Ws.rk5Nfcn;
    view->rk5Naccpt = Ws.rk5Naccpt;
    view->rk5Nrejct = Ws.rk5Nrejct;
    view->rk5Err = Ws.rk5Err;
    view->rk5LastHstep = Ws.rk5LastHstep;
    return 0;
}

extern "C" int MSXreactTransfer_ringView(double dt, int nSpecies, MSXReactTransferView *view)
{
    int k, pos;
    int nLinks = MSX.Nobjects[LINK];
    int nSeg = 0;
    int sid = 0;
    double timer = hostWallTimeMs();
    double packTimer;
    (void)dt;

    if (!view) return ERR_GPU_SEGMENT_PACK_FAILED;
    memset(view, 0, sizeof(*view));
    if (!MSXsegStorage_isPipeRingEnabled()) return ERR_GPU_SEGMENT_PACK_FAILED;
    if (ensureLinkCapacity(nLinks)) return ERR_GPU_MEMORY_ALLOCATION_FAILED;

    for (k = 1; k <= nLinks; k++)
    {
        int count = MSXsegStorage_pipeCount(k);
        Ws.activeFlagByLink[k] = 0;
        Ws.countByLink[k] = count;
        if (count > 0)
        {
            Ws.activeFlagByLink[k] = 1;
            nSeg += count;
        }
    }
    MSX.GpuTimingRecord.react_count_parallel_ms += hostWallTimeMs() - timer;

    {
        double allocTimer = hostWallTimeMs();
        if (ensureSegmentCapacity(nSeg > 0 ? nSeg : 1, nSpecies) ||
            ensureReactedCapacity(nLinks, nSpecies))
        {
            return ERR_GPU_MEMORY_ALLOCATION_FAILED;
        }
        MSX.GpuTimingRecord.react_host_alloc_ms += hostWallTimeMs() - allocTimer;
    }

    timer = hostWallTimeMs();
    MSX.GpuTimingRecord.react_count_prefix_ms += hostWallTimeMs() - timer;

    packTimer = hostWallTimeMs();
    for (k = 1; k <= nLinks; k++)
    {
        int count = Ws.countByLink[k];
        for (pos = 0; pos < count; pos++)
        {
            int slot = MSXsegStorage_pipeSlotFromHead(k, pos);
            int row = MSXsegStorage_pipeFlatIndex(k, slot);
            double *v = MSXsegStorage_pipeVPtr(k, slot);
            double *hstep = MSXsegStorage_pipeHstepPtr(k, slot);
            if (row < 0 || !v || !hstep) return ERR_GPU_SEGMENT_PACK_FAILED;
            Ws.segRow[sid] = row;
            Ws.segPipe[sid] = k;
            Ws.segVol[sid] = *v;
            Ws.hstep[sid] = *hstep;
            Ws.segPtrs[sid] = NULL;
            Ws.unpackOrder[sid] = sid;
            for (int m = 1; m < MAX_HYD_VARS; m++)
                Ws.hyd[sid * MAX_HYD_VARS + m] = MSX.Link[k].HydVar[m];
            sid++;
        }
    }
    memset(Ws.reacted, 0, (size_t)(nLinks + 1) * (nSpecies + 1) * sizeof(double));
    memset(Ws.rk5Nfcn, 0, (size_t)(nSeg > 0 ? nSeg : 1) * sizeof(int));
    memset(Ws.rk5Naccpt, 0, (size_t)(nSeg > 0 ? nSeg : 1) * sizeof(int));
    memset(Ws.rk5Nrejct, 0, (size_t)(nSeg > 0 ? nSeg : 1) * sizeof(int));
    memset(Ws.rk5Err, 0, (size_t)(nSeg > 0 ? nSeg : 1) * sizeof(int));
    memset(Ws.rk5LastHstep, 0, (size_t)(nSeg > 0 ? nSeg : 1) * sizeof(double));
    MSX.GpuTimingRecord.react_pack_segment_ms += hostWallTimeMs() - packTimer;

    view->nSeg = nSeg;
    view->nLinks = nLinks;
    view->nSpecies = nSpecies;
    view->nActiveLinks = 0;
    view->ringResident = 1;
    view->ringSlotCount = (int)MSXsegStorage_ringSlotCount();
    view->ringStride = MSXsegStorage_ringStride();
    view->segRow = Ws.segRow;
    view->segPipe = Ws.segPipe;
    view->activeLink = NULL;
    view->pipeSegOffset = NULL;
    view->pipeSegCount = NULL;
    view->segVol = Ws.segVol;
    view->hstep = Ws.hstep;
    view->segPtrs = Ws.segPtrs;
    view->unpackOrder = Ws.unpackOrder;
    view->c = MSXsegStorage_ringCData();
    view->cOde = MSXsegStorage_ringLastCData();
    view->hyd = Ws.hyd;
    view->reacted = Ws.reacted;
    view->rk5Nfcn = Ws.rk5Nfcn;
    view->rk5Naccpt = Ws.rk5Naccpt;
    view->rk5Nrejct = Ws.rk5Nrejct;
    view->rk5Err = Ws.rk5Err;
    view->rk5LastHstep = Ws.rk5LastHstep;
    if (!view->c || !view->cOde || view->ringSlotCount <= 0 || view->ringStride != nSpecies + 1)
        return ERR_GPU_SEGMENT_PACK_FAILED;
    return 0;
}

extern "C" int MSXreactTransfer_upload(MSXReactTransferView *view, const void *zeroErr, size_t errSize)
{
    int nSeg;
    int nSpecies;
    int nLinks;
    size_t cBytes;
    size_t hydBytes;
    size_t reactedBytes;
    double timer;
    cudaError_t err;
    if (!view) return ERR_GPU_MEMORY_ALLOCATION_FAILED;
    if (ensureDevice(view, errSize)) return ERR_GPU_MEMORY_ALLOCATION_FAILED;
    nSeg = view->nSeg > 0 ? view->nSeg : 1;
    nSpecies = view->nSpecies;
    nLinks = view->nLinks;
    cBytes = view->ringResident ? (size_t)view->ringSlotCount * (size_t)view->ringStride * sizeof(double) :
             (size_t)nSeg * (nSpecies + 1) * sizeof(double);
    hydBytes = (size_t)nSeg * MAX_HYD_VARS * sizeof(double);
    reactedBytes = (size_t)(nLinks + 1) * (nSpecies + 1) * sizeof(double);
    timer = hostWallTimeMs();
#define CUDA_COPY(expr) do { err = (expr); if (err != cudaSuccess) return ERR_GPU_MEMORY_ALLOCATION_FAILED; } while (0)
    CUDA_COPY(cudaMemcpy(Ws.d_segRow, view->segRow, (size_t)nSeg * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_COPY(cudaMemcpy(Ws.d_segPipe, view->segPipe, (size_t)nSeg * sizeof(int), cudaMemcpyHostToDevice));
    if (view->nActiveLinks > 0)
    {
        CUDA_COPY(cudaMemcpy(Ws.d_activeLink, view->activeLink,
                             (size_t)view->nActiveLinks * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_COPY(cudaMemcpy(Ws.d_pipeSegOffset, view->pipeSegOffset,
                             (size_t)view->nActiveLinks * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_COPY(cudaMemcpy(Ws.d_pipeSegCount, view->pipeSegCount,
                             (size_t)view->nActiveLinks * sizeof(int), cudaMemcpyHostToDevice));
    }
    CUDA_COPY(cudaMemcpy(Ws.d_segVol, view->segVol, (size_t)nSeg * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_COPY(cudaMemcpy(Ws.d_hstep, view->hstep, (size_t)nSeg * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_COPY(cudaMemcpy(Ws.d_c, view->c, cBytes, cudaMemcpyHostToDevice));
    CUDA_COPY(cudaMemcpy(Ws.d_cOde, view->cOde, cBytes, cudaMemcpyHostToDevice));
    CUDA_COPY(cudaMemcpy(Ws.d_hyd, view->hyd, hydBytes, cudaMemcpyHostToDevice));
    CUDA_COPY(cudaMemset(Ws.d_reacted, 0, reactedBytes));
    CUDA_COPY(cudaMemset(Ws.d_rk5Nfcn, 0, (size_t)nSeg * sizeof(int)));
    CUDA_COPY(cudaMemset(Ws.d_rk5Naccpt, 0, (size_t)nSeg * sizeof(int)));
    CUDA_COPY(cudaMemset(Ws.d_rk5Nrejct, 0, (size_t)nSeg * sizeof(int)));
    CUDA_COPY(cudaMemset(Ws.d_rk5Err, 0, (size_t)nSeg * sizeof(int)));
    CUDA_COPY(cudaMemset(Ws.d_rk5LastHstep, 0, (size_t)nSeg * sizeof(double)));
    if (zeroErr && errSize > 0) CUDA_COPY(cudaMemcpy(Ws.d_err, zeroErr, errSize, cudaMemcpyHostToDevice));
#undef CUDA_COPY
    MSX.GpuTimingRecord.h2d_ms += hostWallTimeMs() - timer;
    return 0;
}

extern "C" int MSXreactTransfer_download(MSXReactTransferView *view, void *hostErr, size_t errSize)
{
    int nSeg;
    int nSpecies;
    int nLinks;
    size_t cBytes;
    size_t reactedBytes;
    double timer;
    cudaError_t err;
    if (!view) return ERR_GPU_KERNEL_RUNTIME_ERROR;
    nSeg = view->nSeg > 0 ? view->nSeg : 1;
    nSpecies = view->nSpecies;
    nLinks = view->nLinks;
    cBytes = view->ringResident ? (size_t)view->ringSlotCount * (size_t)view->ringStride * sizeof(double) :
             (size_t)nSeg * (nSpecies + 1) * sizeof(double);
    reactedBytes = (size_t)(nLinks + 1) * (nSpecies + 1) * sizeof(double);
    timer = hostWallTimeMs();
#define CUDA_COPY(expr) do { err = (expr); if (err != cudaSuccess) return ERR_GPU_KERNEL_RUNTIME_ERROR; } while (0)
    CUDA_COPY(cudaMemcpy(view->c, Ws.d_c, cBytes, cudaMemcpyDeviceToHost));
    CUDA_COPY(cudaMemcpy(view->cOde, Ws.d_cOde, cBytes, cudaMemcpyDeviceToHost));
    CUDA_COPY(cudaMemcpy(view->hstep, Ws.d_hstep, (size_t)nSeg * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_COPY(cudaMemcpy(view->rk5Nfcn, Ws.d_rk5Nfcn, (size_t)nSeg * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_COPY(cudaMemcpy(view->rk5Naccpt, Ws.d_rk5Naccpt, (size_t)nSeg * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_COPY(cudaMemcpy(view->rk5Nrejct, Ws.d_rk5Nrejct, (size_t)nSeg * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_COPY(cudaMemcpy(view->rk5Err, Ws.d_rk5Err, (size_t)nSeg * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_COPY(cudaMemcpy(view->rk5LastHstep, Ws.d_rk5LastHstep, (size_t)nSeg * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_COPY(cudaMemcpy(view->reacted, Ws.d_reacted, reactedBytes, cudaMemcpyDeviceToHost));
    if (hostErr && errSize > 0) CUDA_COPY(cudaMemcpy(hostErr, Ws.d_err, errSize, cudaMemcpyDeviceToHost));
#undef CUDA_COPY
    MSX.GpuTimingRecord.d2h_ms += hostWallTimeMs() - timer;
    return 0;
}

extern "C" int MSXreactTransfer_unpack(MSXReactTransferView *view)
{
    int sid, i, k;
    int stride;
    int nUnpackSpecies;
    int *unpackSpecies;
    double timer;
    if (!view) return ERR_GPU_SEGMENT_PACK_FAILED;
    if (buildUnpackSpeciesList(view->nSpecies)) return ERR_GPU_MEMORY_ALLOCATION_FAILED;
    stride = view->nSpecies + 1;
    nUnpackSpecies = Ws.unpackSpeciesCount;
    unpackSpecies = Ws.unpackSpecies;
    timer = hostWallTimeMs();
#ifdef _OPENMP
    if (unpackOpenmpEnabled())
    {
#pragma omp parallel for schedule(static)
        for (int pos = 0; pos < view->nSeg; pos++)
        {
            sid = view->unpackOrder ? view->unpackOrder[pos] : pos;
            Pseg seg = view->segPtrs[sid];
            size_t base = (size_t)sid * (size_t)stride;
            if (view->ringResident) continue;
            seg->hstep = view->hstep[sid];
            for (i = 0; i < nUnpackSpecies; i++)
            {
                int m = unpackSpecies[i];
                seg->c[m] = view->c[base + m];
            }
        }
        for (int pos = 0; pos < view->nSeg; pos++)
        {
            sid = view->unpackOrder ? view->unpackOrder[pos] : pos;
            Pseg seg = view->segPtrs[sid];
            size_t base = (size_t)sid * (size_t)stride;
            if (view->ringResident) continue;
            for (i = 0; i < nUnpackSpecies; i++)
            {
                int m = unpackSpecies[i];
                seg->lastc[m] = view->cOde[base + m];
            }
        }
    }
    else
#endif
    {
        for (int pos = 0; pos < view->nSeg; pos++)
        {
            sid = view->unpackOrder ? view->unpackOrder[pos] : pos;
            Pseg seg = view->segPtrs[sid];
            size_t base = view->ringResident ? (size_t)view->segRow[sid] * (size_t)view->ringStride :
                          (size_t)sid * (size_t)stride;
            if (view->ringResident)
            {
                double *hstep = MSXsegStorage_ringHstepData();
                if (hstep) hstep[view->segRow[sid]] = view->hstep[sid];
                continue;
            }
            seg->hstep = view->hstep[sid];
            for (i = 0; i < nUnpackSpecies; i++)
            {
                int m = unpackSpecies[i];
                seg->c[m] = view->c[base + m];
                seg->lastc[m] = view->cOde[base + m];
            }
        }
    }
    for (k = 1; k <= view->nLinks; k++)
    {
        size_t base = (size_t)k * (size_t)stride;
        for (i = 0; i < nUnpackSpecies; i++)
        {
            int m = unpackSpecies[i];
            MSX.Link[k].reacted[m] += view->reacted[base + m];
        }
    }
    {
        double elapsed = hostWallTimeMs() - timer;
        MSX.GpuTimingRecord.react_scatter_ms += elapsed;
        MSX.GpuTimingRecord.react_unpack_ms += elapsed;
    }
    return 0;
}
