#ifndef MSX_RESIDENT_POOL_H
#define MSX_RESIDENT_POOL_H
#include <stdint.h>
/* 252210 objects was the audited whole-batch high-water mark; round upward. */
static uint32_t MSXresidentPool_initial(uint32_t limit)
{return limit<262144U?limit:262144U;}
#endif
