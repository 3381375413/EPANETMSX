#ifndef MSXRESIDENT_HASH_H
#define MSXRESIDENT_HASH_H

/* Hashes raw bytes only; output is NUL-terminated lowercase hex. */
int MSXresident_sha256File(const char *path, char output[65]);
int MSXresident_caseHashFiles(const char *inpPath, const char *msxPath, char output[65]);

#endif
