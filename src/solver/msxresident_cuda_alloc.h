#ifndef MSX_RESIDENT_CUDA_ALLOC_H
#define MSX_RESIDENT_CUDA_ALLOC_H
#include "msxresident_alloc.h"
static cudaError_t managedCudaMalloc(void **ptr,size_t bytes)
{
    *ptr=NULL;
    MSXExternalAllocation *r=MSXresidentExternal_reserve(bytes,MSX_MEMORY_DEVICE,__FILE__,__LINE__);
    if(!r)return cudaErrorMemoryAllocation;
    cudaError_t result=cudaMalloc(ptr,bytes);
    if(result!=cudaSuccess){MSXresidentExternal_cancel(r);*ptr=NULL;return result;}
    MSXresidentExternal_commit(r,(uintptr_t)*ptr);return result;
}
static cudaError_t managedCudaMallocHost(void **ptr,size_t bytes)
{
    *ptr=NULL;
    MSXExternalAllocation *r=MSXresidentExternal_reserve(bytes,MSX_MEMORY_PINNED,__FILE__,__LINE__);
    if(!r)return cudaErrorMemoryAllocation;
    cudaError_t result=cudaMallocHost(ptr,bytes);
    if(result!=cudaSuccess){MSXresidentExternal_cancel(r);*ptr=NULL;return result;}
    MSXresidentExternal_commit(r,(uintptr_t)*ptr);return result;
}
static cudaError_t managedCudaFree(void *ptr)
{
    cudaError_t result=cudaFree(ptr);
    if(result==cudaSuccess && ptr)MSXresidentExternal_release((uintptr_t)ptr,MSX_MEMORY_DEVICE);
    return result;
}
static cudaError_t managedCudaFreeHost(void *ptr)
{
    cudaError_t result=cudaFreeHost(ptr);
    if(result==cudaSuccess && ptr)MSXresidentExternal_release((uintptr_t)ptr,MSX_MEMORY_PINNED);
    return result;
}
#endif
