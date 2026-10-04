#ifndef MSX_RESIDENT_UPLOAD_H
#define MSX_RESIDENT_UPLOAD_H
#include "msxresident_core.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef MSXResidentStatus (*MSXUploadStage)(void *,const MSXResidentSlotPatch *,uint32_t);
MSXResidentStatus MSXupload_open(uint32_t rows,uint32_t stride,MSXUploadStage,void *);
MSXResidentStatus MSXupload_capture(const MSXResidentSlotPatch *,uint64_t *revision);
MSXResidentStatus MSXupload_flush(void);
void MSXupload_close(void); /* GPU must already have stopped. */
int MSXupload_enabled(void);
#ifdef __cplusplus
}
#endif
#endif
