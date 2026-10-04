#ifndef MSX_RESIDENT_OPTIONS_H
#define MSX_RESIDENT_OPTIONS_H
#include <stdint.h>
#include <stdlib.h>
enum { MSX_MEMORY_OPTION_HOST=1,MSX_MEMORY_OPTION_BATCH=2,
       MSX_MEMORY_OPTION_CPU=4,MSX_MEMORY_OPTION_UPLOAD=8 };
typedef struct {
    uint64_t hostBytes,cpuPoolBytes,uploadBytes;
    uint32_t batchRows,explicitMask;
} MSXResidentMemoryOptions;
typedef struct {
    MSXResidentMemoryOptions values;
    const char *hostSource,*cpuSource,*uploadSource,*batchSource;
} MSXResidentResolvedOptions;
static int MSXmemory_parseInteger(const char *s,uint64_t *out)
{
    uint64_t value=0;const unsigned char *p=(const unsigned char *)s;
    if(!s||!*s||!out)return 0;
    for(;*p;++p){unsigned digit=(unsigned)(*p-'0');
        if(digit>9||value>(UINT64_MAX-digit)/10)return 0;value=value*10+digit;}
    *out=value;return 1;
}
/* Public MB options are positive integer MiB, converted without a double. */
static int MSXmemory_parseMB(const char *s,uint64_t *out)
{
    uint64_t value;
    if(!MSXmemory_parseInteger(s,&value)||!value||value>UINT64_MAX/1048576)return 0;
    *out=value*1048576;return 1;
}
static int MSXmemory_resolveOne(uint64_t option,int present,const char *bytesEnv,
                               const char *mbEnv,uint64_t fallback,uint64_t *out,const char **source)
{
    const char *s;uint64_t value=option;int set=present;
    *source=present?"MSX_OPTION":"COMPATIBILITY_DEFAULT";
    if(bytesEnv && (s=getenv(bytesEnv))!=NULL){uint64_t bytes;
        if(!MSXmemory_parseInteger(s,&bytes)||!bytes||(set&&value!=bytes))return 0;
        value=bytes;set=1;*source=present?"MSX_OPTION+ENV_BYTES":bytesEnv;}
    if(mbEnv && (s=getenv(mbEnv))!=NULL){uint64_t bytes;
        if(!MSXmemory_parseMB(s,&bytes)||(set&&value!=bytes))return 0;
        value=bytes;set=1;*source=present?"MSX_OPTION+ENV_MB":mbEnv;}
    *out=set?value:fallback;return 1;
}
static int MSXmemory_resolve(const MSXResidentMemoryOptions *o,uint64_t defaultHost,
                             MSXResidentResolvedOptions *out)
{
    uint64_t rows;unsigned m=o->explicitMask;
    if(!MSXmemory_resolveOne(o->hostBytes,m&MSX_MEMORY_OPTION_HOST,"MSX_RESIDENT_HOST_BUDGET_BYTES",
         "MSX_RESIDENT_HOST_MEMORY_MB",defaultHost,&out->values.hostBytes,&out->hostSource)||
       !MSXmemory_resolveOne(o->cpuPoolBytes,m&MSX_MEMORY_OPTION_CPU,"MSX_RESIDENT_CPU_POOL_BUDGET_BYTES",
         "MSX_RESIDENT_CPU_POOL_BUDGET_MB",0,&out->values.cpuPoolBytes,&out->cpuSource)||
       !MSXmemory_resolveOne(o->uploadBytes,m&MSX_MEMORY_OPTION_UPLOAD,"MSX_RESIDENT_UPLOAD_BUDGET_BYTES",
         "MSX_RESIDENT_UPLOAD_BUDGET_MB",0,&out->values.uploadBytes,&out->uploadSource)||
       !MSXmemory_resolveOne(o->batchRows,m&MSX_MEMORY_OPTION_BATCH,"MSX_RESIDENT_TRANSFER_BATCH_ROWS",
         NULL,65536,&rows,&out->batchSource)||!rows||rows>UINT32_MAX)return 0;
    out->values.batchRows=(uint32_t)rows;out->values.explicitMask=m;return 1;
}
#endif
