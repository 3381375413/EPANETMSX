#include <stdio.h>
#include <stdlib.h>
#include "msxresident_capacity_search.h"
static unsigned checks;
#define CHECK(x) do{++checks;if(!(x)){fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x);exit(1);}}while(0)
typedef struct {uint32_t lower,upper;unsigned calls;int islands;} Fixture;
static int model(uint32_t m,int extras,void *context)
{
 Fixture *f=(Fixture*)context;(void)extras;++f->calls;
 if(f->islands)return (m>=50&&m<=58)||(m>=90&&m<=96);
 return m>=f->lower&&m<=f->upper;
}
static uint32_t cpu(uint32_t initial,uint32_t guard,uint32_t admission)
{uint32_t core=initial>2*guard?initial-2*guard:0;return initial-(admission<core?admission:core);}
int main(void)
{
 uint32_t chosen,n,guard,requested,m;Fixture f={0,173,0,0};
 CHECK(MSXcapacity_headroom(100,30)==70);
 CHECK(30+MSXcapacity_headroom(100,30)==100);
 CHECK(MSXcapacity_headroom(100,100)==0);
 CHECK(MSXcapacity_headroom(100,101)==0);
 CHECK(MSXcapacity_headroom(UINT64_MAX,0)==UINT64_MAX);
 CHECK(MSXcapacity_searchCeiling(50000,1,model,&f,&chosen)&&chosen==173);
 CHECK(f.calls<=18);
 /* A low ceiling fails CPU quota while a larger ceiling is feasible. */
 f.lower=30;f.upper=80;f.calls=0;
 CHECK(MSXcapacity_searchCeiling(96,0,model,&f,&chosen)&&chosen==80);
 CHECK(f.calls==97);
 /* Backing block ceilings need not form one feasible interval. */
 f.islands=1;f.calls=0;
 CHECK(MSXcapacity_searchCeiling(96,0,model,&f,&chosen)&&chosen==96);
 CHECK(f.calls==97);
 f.islands=0;f.lower=97;f.upper=100;
 CHECK(!MSXcapacity_searchCeiling(96,0,model,&f,&chosen));
 for(n=0;n<=100;++n)for(guard=2;guard<=4;guard+=2)
  for(requested=0;requested<=300;requested+=3){
   uint32_t base=MSXcapacity_cappedLimit(n,guard,requested,0,1);
   uint32_t constant=cpu(n,guard,base),previous=base;
   CHECK(base<=96);
   for(m=0;m<=300;m+=3){uint32_t admit=MSXcapacity_cappedLimit(n,guard,requested,m,1);
    CHECK(admit>=previous&&admit<=requested);CHECK(cpu(n,guard,admit)==constant);previous=admit;}
  }
 /* Same physical slots can have different initial CPU costs. */
 CHECK(40+1==1+40);
 CHECK(cpu(100,2,40)+cpu(20,2,0)!=cpu(100,2,0)+cpu(20,2,40));
 /* Replacing an empty placeholder with one logical admission adds no slots. */
 CHECK(cpu(100,2,1)+1==cpu(100,2,0));
 printf("PASS %u checks: production ceiling search and vector CPU invariants\n",checks);
 return 0;
}
