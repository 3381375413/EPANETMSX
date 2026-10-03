#ifndef MSXRESIDENT_HASH_H
#define MSXRESIDENT_HASH_H
#include <stdint.h>
#include <stddef.h>
typedef struct { uint32_t h[8]; uint64_t n; unsigned char b[64]; size_t used; } MSXResidentSha256;
void MSXresident_sha256Init(MSXResidentSha256 *);
void MSXresident_sha256Add(MSXResidentSha256 *,const void *,size_t);
void MSXresident_sha256Finish(MSXResidentSha256 *,char [65]);

/* Hashes raw bytes only; output is NUL-terminated lowercase hex. */
int MSXresident_sha256File(const char *path, char output[65]);
int MSXresident_caseHashFiles(const char *inpPath, const char *msxPath, char output[65]);

#endif
