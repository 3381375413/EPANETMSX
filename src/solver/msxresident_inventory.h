#ifndef MSX_RESIDENT_INVENTORY_H
#define MSX_RESIDENT_INVENTORY_H
#include <stdio.h>
#include <stdint.h>
#include "msxresident_budget.h"
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#endif
#ifdef __cplusplus
extern "C" {
#endif
typedef struct { FILE *file; uint64_t total[3],domains[4][3],rows; int error,closed; } MSXInventory;
int MSXinv_describeHeap(const void *,MSXBudgetAllocation *,uint64_t *);
int MSXinv_begin(MSXInventory *,const char *,int);
int MSXinv_end(MSXInventory *);
void MSXinv_heap(MSXInventory *,const char *,const void *,uint64_t);
void MSXinv_indexed(MSXInventory *,const char *,uint32_t,uint32_t,const char *,const void *,uint64_t);
void MSXinv_alias(MSXInventory *,const char *,const void *,const void *);
void MSXinv_ticket(MSXInventory *,const char *,const void *,uint64_t,const MSXBudgetAllocation *);
void MSXinv_tag(MSXInventory *,const char *,uint64_t);
void MSXinv_matrix(MSXInventory *,const char *,double **);
void MSXinv_external(MSXInventory *);
void MSXinv_Core(MSXInventory *);
void MSXinv_Storage(MSXInventory *);
void MSXinv_Runtime(MSXInventory *);
void MSXinv_Upload(MSXInventory *);
void MSXinv_Capacity(MSXInventory *);
void MSXinv_Project(MSXInventory *);
void MSXinv_Quality(MSXInventory *);
void MSXinv_Chemistry(MSXInventory *);
void MSXinv_GPUPrograms(MSXInventory *);
void MSXinv_Profile(MSXInventory *);
void MSXinv_Pool(MSXInventory *,void *,const char *);
void MSXinv_poolIndexed(MSXInventory *,const char *,uint32_t,uint32_t,const char *,const void *,void *);
void MSXinv_Hash(MSXInventory *,void *,uint32_t);
void MSXinv_Expr(MSXInventory *,void *,const char *,uint32_t);
void MSXinv_TankExpr(MSXInventory *,void *,void *,const char *,uint32_t);
void MSXinv_Newton(MSXInventory *);
void MSXinv_RK5(MSXInventory *);
void MSXinv_ROS2(MSXInventory *);
void MSXinv_CUDA(MSXInventory *,void *);
int MSXinv_quiescent(void);
void MSXinv_checkpoint(const void *,const void *);
#define INV_HEAP(s,p) MSXinv_heap((s),#p,(p),sizeof(*(p)))
#define INV_INDEX(s,m,k,j,p) MSXinv_indexed((s),(m),(k),(j),#p,(p),sizeof(*(p)))
#define INV_POOL_INDEX(s,m,k,j,p,h) MSXinv_poolIndexed((s),(m),(k),(j),#p,(p),(h))
#ifdef __cplusplus
}
#endif
#endif
