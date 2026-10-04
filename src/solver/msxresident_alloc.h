#ifndef MSX_RESIDENT_ALLOC_H
#define MSX_RESIDENT_ALLOC_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Only pointers created by this allocator may be passed to free.  QualPool
   objects are interior pointers and continue to use the pool release hooks. */
void *MSXresidentAlloc_malloc(size_t bytes, const char *source, unsigned line);
void *MSXresidentAlloc_calloc(size_t count, size_t width, const char *source, unsigned line);
void MSXresidentAlloc_free(void *ptr);
size_t MSXresidentAlloc_headerBytes(void);
#ifdef __cplusplus
}
#endif
#endif
