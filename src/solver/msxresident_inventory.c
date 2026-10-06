/* Test-only streaming inspection. No allocation registry or payload copy. */
#ifdef MSX_RESIDENT_TEST_API
#include <string.h>
#include "msxresident_inventory.h"
#include "msxresident_alloc.h"
static char InitialPath[1024];
static void quoted(FILE *f,const char *s)
{ fputc('"',f);while(*s){if(*s=='"'||*s=='\\')fputc('\\',f);fputc((unsigned char)*s++,f);}fputc('"',f); }
static int domain(uintptr_t owner)
{ int i;for(i=0;i<4;i++)if(owner==(uintptr_t)MSXresidentBudget_domain((MSXBudgetDomain)i))return i;return -1; }
static void add(MSXInventory *s,uint64_t *total,uint64_t value)
{ if(value>UINT64_MAX-*total)s->error=1;else *total+=value; }
void MSXinv_ticket(MSXInventory *s,const char *key,const void *ptr,uint64_t payload,const MSXBudgetAllocation *a)
{
 unsigned c=a?a->memoryClass:0;uint64_t charge=a?a->bytes:0;int d=a?domain(a->budgetOwner):-1;
 if(s->closed){s->error=1;return;}
 if(ptr&&(!a||!a->live||c>2||payload>charge))s->error=1;
 if(!ptr&&(payload||charge))s->error=1;
 if(c<3){add(s,&s->total[c],charge);if(d>=0)add(s,&s->domains[d][c],charge);}
 fprintf(s->file,"{\"record\":\"OWNER\",\"key\":");quoted(s->file,key);
 fprintf(s->file,",\"pointer\":%llu,\"requested_bytes\":%llu,\"charged_bytes\":%llu,\"class\":%u,\"domain\":%d,\"ticket\":%llu,\"live\":%u}\n",
  (unsigned long long)(uintptr_t)ptr,(unsigned long long)payload,(unsigned long long)charge,c,d,
  (unsigned long long)(a?a->sequence:0),a?a->live:0);++s->rows;
}
void MSXinv_heap(MSXInventory *s,const char *key,const void *ptr,uint64_t width)
{
 MSXBudgetAllocation a={0};uint64_t payload=0;
 if(s->closed){s->error=1;return;}
 if(ptr&&!MSXinv_describeHeap(ptr,&a,&payload)){s->error=1;return;}
 MSXinv_ticket(s,key,ptr,payload,ptr?&a:NULL);
 fprintf(s->file,"{\"record\":\"SHAPE\",\"key\":");quoted(s->file,key);
 fprintf(s->file,",\"element_width\":%llu,\"max_elements\":%llu,\"header_bytes\":%llu}\n",
  (unsigned long long)width,(unsigned long long)(width?payload/width:0),
  (unsigned long long)(ptr?a.bytes-payload:0));
}
void MSXinv_indexed(MSXInventory *s,const char *module,uint32_t k,uint32_t j,const char *field,const void *ptr,uint64_t width)
{ char key[256];snprintf(key,sizeof(key),"%s[%u,%u].%s",module,k,j,field);MSXinv_heap(s,key,ptr,width); }
void MSXinv_alias(MSXInventory *s,const char *key,const void *ptr,const void *backing)
{ fprintf(s->file,"{\"record\":\"ALIAS\",\"key\":");quoted(s->file,key);fprintf(s->file,",\"pointer\":%llu,\"backing_pointer\":%llu,\"charged_bytes\":0}\n",(unsigned long long)(uintptr_t)ptr,(unsigned long long)(uintptr_t)backing); }
void MSXinv_tag(MSXInventory *s,const char *key,uint64_t value)
{ fprintf(s->file,"{\"record\":\"TAG\",\"key\":");quoted(s->file,key);fprintf(s->file,",\"value\":%llu}\n",(unsigned long long)value); }
void MSXinv_matrix(MSXInventory *s,const char *key,double **p)
{ char name[256];MSXinv_heap(s,key,p,sizeof(*p));snprintf(name,sizeof(name),"%s.data",key);MSXinv_heap(s,name,p?p[0]:NULL,sizeof(double)); }
int MSXinv_begin(MSXInventory *s,const char *path,int closed)
{ memset(s,0,sizeof(*s));if(!path||!path[0])return 1;s->closed=closed;s->file=fopen(path,"wb");if(!s->file)return 1;fprintf(s->file,"{\"record\":\"BEGIN\",\"schema\":1,\"closed\":%d,\"scope\":\"actual managed allocation owners; EPANET/CRT/CUDA internal/module handles excluded\"}\n",closed);return 0; }
int MSXinv_end(MSXInventory *s)
{
 MSXResidentBudget b;int d,c;uint64_t root[3];
 MSXresidentBudget_snapshot(MSXresidentBudget_global(),&b);
 for(c=0;c<3;c++){root[c]=b.allocated[c];if(s->total[c]!=root[c]|| (s->closed&&b.reserved[c]))s->error=1;}
 fprintf(s->file,"{\"record\":\"LEDGER\",\"domain\":-1,\"allocated\":[%llu,%llu,%llu],\"reserved\":[%llu,%llu,%llu]}\n",(unsigned long long)b.allocated[0],(unsigned long long)b.allocated[1],(unsigned long long)b.allocated[2],(unsigned long long)b.reserved[0],(unsigned long long)b.reserved[1],(unsigned long long)b.reserved[2]);
 for(d=0;d<4;d++){MSXresidentBudget_snapshot(MSXresidentBudget_domain((MSXBudgetDomain)d),&b);for(c=0;c<3;c++)if(s->domains[d][c]!=b.allocated[c]||(s->closed&&b.reserved[c]))s->error=1;
 fprintf(s->file,"{\"record\":\"LEDGER\",\"domain\":%d,\"allocated\":[%llu,%llu,%llu],\"reserved\":[%llu,%llu,%llu]}\n",d,(unsigned long long)b.allocated[0],(unsigned long long)b.allocated[1],(unsigned long long)b.allocated[2],(unsigned long long)b.reserved[0],(unsigned long long)b.reserved[1],(unsigned long long)b.reserved[2]);}
 fprintf(s->file,"{\"record\":\"END\",\"status\":\"%s\",\"rows\":%llu,\"catalogue_charged\":[%llu,%llu,%llu],\"root_allocated\":[%llu,%llu,%llu]}\n",s->error?"FAIL":"PASS",(unsigned long long)s->rows,(unsigned long long)s->total[0],(unsigned long long)s->total[1],(unsigned long long)s->total[2],(unsigned long long)root[0],(unsigned long long)root[1],(unsigned long long)root[2]);
 if(ferror(s->file))s->error=1;if(fclose(s->file))s->error=1;s->file=NULL;return s->error;
}
#ifndef MSX_INVENTORY_STANDALONE
#include "msxtypes.h"
#include "msxresident_runtime.h"
extern MSXproject MSX;
static void all(MSXInventory *s)
{ MSXinv_external(s);MSXinv_Project(s);MSXinv_Quality(s);MSXinv_Core(s);MSXinv_Storage(s);MSXinv_Runtime(s);MSXinv_Upload(s);MSXinv_Capacity(s);MSXinv_Chemistry(s);MSXinv_GPUPrograms(s);MSXinv_Profile(s); }
__declspec(dllexport) int MSXTESTresidentAllocationInventory(const char *path,int closed)
{ MSXInventory s;if(!closed&&!MSXinv_quiescent())return 525;if(MSXinv_begin(&s,path,closed))return 501;MSXinv_tag(&s,"checkpoint.phase",closed?2:1);if(!closed){MSXinv_tag(&s,"checkpoint.qualityTimeMs",(uint64_t)MSX.Qtime);all(&s);}return MSXinv_end(&s)?525:0; }
__declspec(dllexport) int MSXTESTresidentAllocationInventoryArm(const char *path)
{ if(!path||!path[0]||strlen(path)>=sizeof(InitialPath)||InitialPath[0])return 525;strcpy(InitialPath,path);return 0; }
void MSXinv_checkpoint(const void *cache,const void *draft)
{ MSXInventory s;char path[sizeof(InitialPath)];uint32_t k,j;Pseg seg;const MSXResidentInitialTankDraft *d=(const MSXResidentInitialTankDraft *)draft;
 if(!InitialPath[0])return;strcpy(path,InitialPath);InitialPath[0]=0;if(MSXinv_begin(&s,path,0))return;MSXinv_tag(&s,"checkpoint.phase",0);all(&s);MSXinv_heap(&s,"initial.cache",cache,sizeof(double));MSXinv_heap(&s,"initial.tankDraft",draft,sizeof(*d));
 if(d)for(k=1;k<=(uint32_t)MSX.Nobjects[TANK];k++){j=0;for(seg=d[k].first;seg;seg=seg->prev){if(seg->privateC)INV_POOL_INDEX(&s,"Initial.tankPrivate",k,j,seg->privateC,MSX.QualPool);if(seg->privateLastC)INV_POOL_INDEX(&s,"Initial.tankPrivate",k,j,seg->privateLastC,MSX.QualPool);++j;}}
 MSXinv_end(&s); }
#endif
#endif
