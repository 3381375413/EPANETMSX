#ifndef MSXRESIDENT_MEMORY_CONFIG_H
#define MSXRESIDENT_MEMORY_CONFIG_H
#include <stdlib.h>
#include <errno.h>
#include "msxresident_core_cuda.h"
/* Integer-only, reject signs, whitespace, overflow and trailing characters.
   Configuration is resolved before planning; allocation uses that snapshot. */
static int MSXresident_parseBytes(const char *s,uint64_t *out)
{
    char *end; unsigned long long value;
    if(!s||!*s||*s<'0'||*s>'9')return 0;
    errno=0;value=strtoull(s,&end,10);
    if(errno||*end)return 0;
    *out=(uint64_t)value;return 1;
}
static MSXResidentStatus MSXresident_resolveMemoryConfig(MSXResidentMemoryConfig *c)
{
    uint64_t rows;const char *s=getenv("MSX_RESIDENT_TRANSFER_BATCH_ROWS");
    if(!c)return MSX_RESIDENT_ERR_ARGUMENT;
    c->version=1;c->hydLayout=MSX_RESIDENT_HYD_PIPE_MAJOR;
    c->transferBatchRows=65536;
    if(s){if(!MSXresident_parseBytes(s,&rows)||!rows||rows>UINT32_MAX)
        return MSX_RESIDENT_ERR_CONFIG;c->transferBatchRows=(uint32_t)rows;}
    return MSX_RESIDENT_OK;
}
#endif
