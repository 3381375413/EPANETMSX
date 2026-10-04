#ifndef MSX_RESIDENT_DRIVER_ALLOC_H
#define MSX_RESIDENT_DRIVER_ALLOC_H
#include "msxresident_alloc.h"
#ifdef EPANETMSX_CUDA_ENABLED
static CUresult managedCuMemAlloc(CUdeviceptr *ptr,size_t bytes)
{
    CUresult result;MSXExternalAllocation *r;
    *ptr=0;
    r=MSXresidentExternal_reserve(bytes,MSX_MEMORY_DEVICE,__FILE__,__LINE__);
    if(!r)return CUDA_ERROR_OUT_OF_MEMORY;
    result=cuMemAlloc(ptr,bytes);
    if(result!=CUDA_SUCCESS){MSXresidentExternal_cancel(r);*ptr=0;return result;}
    MSXresidentExternal_commit(r,(uintptr_t)*ptr);return result;
}
static CUresult managedCuMemHostAlloc(void **ptr,size_t bytes,unsigned flags)
{
    CUresult result;MSXExternalAllocation *r;
    *ptr=NULL;
    r=MSXresidentExternal_reserve(bytes,MSX_MEMORY_PINNED,__FILE__,__LINE__);
    if(!r)return CUDA_ERROR_OUT_OF_MEMORY;
    result=cuMemHostAlloc(ptr,bytes,flags);
    if(result!=CUDA_SUCCESS){MSXresidentExternal_cancel(r);*ptr=NULL;return result;}
    MSXresidentExternal_commit(r,(uintptr_t)*ptr);return result;
}
static CUresult managedCuMemFree(CUdeviceptr ptr)
{
    CUresult result=cuMemFree(ptr);
    if(result==CUDA_SUCCESS && ptr)MSXresidentExternal_release((uintptr_t)ptr,MSX_MEMORY_DEVICE);
    return result;
}
static CUresult managedCuMemFreeHost(void *ptr)
{
    CUresult result=cuMemFreeHost(ptr);
    if(result==CUDA_SUCCESS && ptr)MSXresidentExternal_release((uintptr_t)ptr,MSX_MEMORY_PINNED);
    return result;
}
#endif
#endif
