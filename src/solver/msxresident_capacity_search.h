#ifndef MSXRESIDENT_CAPACITY_SEARCH_H
#define MSXRESIDENT_CAPACITY_SEARCH_H
#include <stdint.h>
typedef int (*MSXCapacityCeilingFits)(uint32_t ceiling,int extras,void *context);
static uint64_t MSXcapacity_headroom(uint64_t quota,uint64_t occupied)
{return quota>occupied?quota-occupied:0;}
static uint32_t MSXcapacity_cappedLimit(uint32_t initial,uint32_t guard,
                                      uint32_t requested,uint32_t ceiling,int extras)
{
 uint32_t core=initial>2*guard?initial-2*guard:0;
 uint32_t base=requested<core?requested:core;
 uint32_t remaining=requested-base;
 if(extras)return base+(remaining<ceiling?remaining:ceiling);
 return base<ceiling?base:ceiling;
}
/* In the extras region the initial CPU count is constant: admissions never
   fall below min(initial Core demand, requested). Fixed arrays and the bounded
   initial pool are nondecreasing, so exact fits has a monotone upper boundary.
   Below that base, CPU backing can fall while arrays grow. Enumerate every
   discrete ceiling (at most 96+1, because N0<=100 and guard>=2), including
   disconnected feasible islands and actual pool backing block staircases. */
static int MSXcapacity_searchCeiling(uint32_t upper,int extras,
                                   MSXCapacityCeilingFits fits,void *context,uint32_t *chosen)
{
 uint32_t lo=0,hi=upper,m;int found=0;
 if(!fits||!chosen)return 0;
 if(extras){
  if(!fits(0,1,context))return 0;
  while(lo<hi){m=lo+(hi-lo+1)/2;
   if(fits(m,1,context))lo=m;else hi=m-1;}
  *chosen=lo;return 1;
 }
 for(m=0;;++m){if(fits(m,0,context)){*chosen=m;found=1;}if(m==upper)break;}
 return found;
}
#endif
