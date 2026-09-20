#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include "msxresident_core.h"
#include "msxresident_hash.h"
#include "msxtypes.h"
#include "msxsegment_storage.h"
#include "epanet2.h"
extern MSXproject MSX;
extern void MSXqual_removeSeg(Pseg seg);
#define NONE UINT32_MAX
static uint64_t ResidentAllocationCount;
static int64_t ResidentAllocationFailAfter = -1;
static void *resident_malloc(size_t n)
{ if (ResidentAllocationFailAfter == 0) return NULL; if (ResidentAllocationFailAfter > 0) ResidentAllocationFailAfter--; ResidentAllocationCount++; return malloc(n); }
static void *resident_calloc(size_t n, size_t z)
{ if (ResidentAllocationFailAfter == 0) return NULL; if (ResidentAllocationFailAfter > 0) ResidentAllocationFailAfter--; ResidentAllocationCount++; return calloc(n,z); }
#define malloc resident_malloc
#define calloc resident_calloc
typedef struct { MSXResidentPipeDesc d; uint32_t guard; unsigned char *used; uint32_t *gen,*patch,*obsI2s,*obsKeep; uint64_t *id; double *row; } Pipe;
/* One reusable transaction arena is owned by each resident link.  Runtime
   prepares at most one transaction per link before the all-link validation. */
typedef struct { MSXResidentHandoffPlan plan; MSXResidentHandoffItem *item; MSXResidentHandoffResult *result; Pseg *boundary; double *c,*lastc; uint32_t count,capacity,patchReserve; int active; MSXHybridResidentMaterialization storage; } Tx;
typedef struct { int open,strict,initialStaging,initialCommitted,poisoned; MSXResidentMode mode; uint32_t nlinks,stride,rowWidth; size_t slots; char hash[65]; Pipe *p; uint32_t *dirty,*capacity,*base,*guard; MSXResidentDescriptorPatch *desc,*initialDesc; MSXResidentSlotPatch *slot,*initialSlot; MSXResidentHandoffItem *hand; Tx *tx; uint32_t ndesc,nslot,nh; MSXResidentStats stat; } State;
static State S={0};
static int mul(size_t a,size_t b,size_t*o){if(a&&b>SIZE_MAX/a)return 0;*o=a*b;return 1;}
static void txreset(Tx *t);
static void freestate(void){uint32_t k;if(S.tx){for(k=1;k<=S.nlinks;k++){txreset(&S.tx[k]);free(S.tx[k].item);free(S.tx[k].result);free(S.tx[k].boundary);free(S.tx[k].c);free(S.tx[k].lastc);}free(S.tx);}if(S.p)for(k=1;k<=S.nlinks;k++){free(S.p[k].used);free(S.p[k].gen);free(S.p[k].patch);free(S.p[k].obsI2s);free(S.p[k].obsKeep);free(S.p[k].id);free(S.p[k].row);}free(S.p);free(S.dirty);free(S.capacity);free(S.base);free(S.guard);free(S.desc);free(S.initialDesc);free(S.slot);free(S.initialSlot);free(S.hand);memset(&S,0,sizeof(S));}
static int hashok(const char*s){int i;if(!s)return 0;for(i=0;i<64;i++)if(!((s[i]>='0'&&s[i]<='9')||(s[i]>='A'&&s[i]<='F')||(s[i]>='a'&&s[i]<='f')))return 0;return s[64]==0;}
static void hashupper(char*s){int i;for(i=0;i<64;i++)if(s[i]>='a'&&s[i]<='f')s[i]=(char)(s[i]-'a'+'A');}
static int hashcmp(const char*a,const char*b){char x[65],y[65];if(!hashok(a)||!hashok(b))return 1;memcpy(x,a,65);memcpy(y,b,65);hashupper(x);hashupper(y);return strcmp(x,y);}
static int unum(const char*s,uint32_t*x){char*e;unsigned long v;if(!*s)return 0;errno=0;v=strtoul(s,&e,10);if(errno||e==s||*e||v>UINT32_MAX)return 0;*x=(uint32_t)v;return 1;}
static int csv(char*s,uint32_t*k,char*id,uint32_t*cap,uint32_t*core,uint32_t*burst,uint32_t*guard,char*h){char*f[7],*p=s;int i;for(i=0;i<7;i++){f[i]=p;p=strchr(p,',');if(i<6){if(!p)return 0;*p++=0;}else if(p)return 0;}if(!unum(f[0],k)||!unum(f[2],cap)||!unum(f[3],core)||!unum(f[4],burst)||!unum(f[5],guard)||!f[1][0]||strlen(f[1])>EN_MAXID||!hashok(f[6]))return 0;strcpy(id,f[1]);strcpy(h,f[6]);hashupper(h);/* Only the open parser owns Pipe metadata; validation must be read-only. */if(S.p&&!S.open&&*k&&*k<=S.nlinks)S.p[*k].guard=*guard;return 1;}
static MSXResidentStatus validatecsv(const char *path,int guard,const char *hash)
{ FILE*f;char line[1024],id[EN_MAXID+1],want[EN_MAXID+1],h[65],first[65],*e;uint32_t row=0,k,cap,core,burst,g,n=(uint32_t)MSX.Nobjects[LINK];
  f=fopen(path,"rb");if(!f)return MSX_RESIDENT_ERR_PATH;
  if(!fgets(line,sizeof(line),f)||strcmp(line,"link_index,link_id,capacity,max_core_count,combined_burst_p99,guard,case_hash\n")&&strcmp(line,"link_index,link_id,capacity,max_core_count,combined_burst_p99,guard,case_hash\r\n")){fclose(f);return MSX_RESIDENT_ERR_CSV;}
  while(fgets(line,sizeof(line),f)){e=strpbrk(line,"\r\n");if(!e||(*e=='\r'&&e[1]!='\n')||e[1+(*e=='\r')]){fclose(f);return MSX_RESIDENT_ERR_CSV;}*e=0;if(!csv(line,&k,id,&cap,&core,&burst,&g,h)||k!=row+1||!k||k>n||!cap||core>cap||(g!=2&&g!=4)||cap>(uint32_t)MSX.MaxSegments){fclose(f);return MSX_RESIDENT_ERR_CSV;}if(guard&&g<(uint32_t)guard){fclose(f);return MSX_RESIDENT_ERR_GUARD;}want[0]=0;if(ENgetlinkid((int)k,want)||strcmp(id,want)){fclose(f);return MSX_RESIDENT_ERR_CSV;}if(!row)strcpy(first,h);else if(strcmp(first,h)){fclose(f);return MSX_RESIDENT_ERR_CSV;}if(hash&&hashcmp(h,hash)){fclose(f);return MSX_RESIDENT_ERR_CSV;}row++;}
  if(ferror(f)||row!=n){fclose(f);return MSX_RESIDENT_ERR_CSV;}fclose(f);return MSX_RESIDENT_OK; }
static MSXResidentStatus pipeok(uint32_t k,Pipe**p){if(!S.open)return MSX_RESIDENT_DISABLED;if(S.poisoned)return MSX_RESIDENT_ERR_POISONED;if(!k||k>S.nlinks)return MSX_RESIDENT_ERR_ARGUMENT;*p=&S.p[k];return MSX_RESIDENT_OK;}
/* slot layout is [5 scalar + stride c][5 scalar + stride lastc]. */
static double*crow(Pipe*p,uint32_t s){return p->row+(size_t)s*2*S.rowWidth;}
static double*lrow(Pipe*p,uint32_t s){return crow(p,s)+S.rowWidth;}
static void getpayload(Pipe*p,uint32_t s,MSXResidentPayload*x){double*r=crow(p,s);x->volume=r[0];x->hstep=r[1];x->hresponse=r[2];x->uresponse=r[3];x->dresponse=r[4];x->parcelId=p->id[s];x->generation=p->gen[s];x->c=r+5;x->lastc=lrow(p,s)+5;}
static int find(Pipe*p,uint64_t id){uint32_t s;for(s=0;s<p->d.capacity;s++)if(p->used[s]&&p->id[s]==id)return(int)s;return-1;}
static uint32_t nextslot(const Pipe *p,uint32_t slot,int dir)
{ int64_t q=(int64_t)slot+(dir<0?-1:1); if(q<0)q+=p->d.capacity; if(q>=(int64_t)p->d.capacity)q-=p->d.capacity; return(uint32_t)q; }
static uint32_t prevslot(const Pipe *p,uint32_t slot,int dir)
{ return nextslot(p,slot,-dir); }
static MSXResidentPatchKind defaultKind(uint32_t used)
{ return used?MSX_RESIDENT_PATCH_IMPORT:MSX_RESIDENT_PATCH_INVALIDATE; }
static void markdesc(uint32_t k){uint32_t n=S.dirty[k];if(n==NONE){n=S.ndesc++;S.dirty[k]=n;S.desc[n].linkIndex=k;}S.desc[n].descriptor=S.p[k].d;S.stat.descriptorPatches++;}
static void markslot(uint32_t k,uint32_t s,uint32_t used,MSXResidentPatchKind kind){Pipe*p=&S.p[k];uint32_t n=p->patch[s];int hadPatch=n!=NONE;MSXResidentSlotPatch*q;MSXResidentPatchKind finalKind=kind?kind:defaultKind(used);if(!hadPatch){n=S.nslot++;p->patch[s]=n;}q=&S.slot[n];if(hadPatch&&q->kind==MSX_RESIDENT_PATCH_IMPORT&&finalKind==MSX_RESIDENT_PATCH_META)finalKind=MSX_RESIDENT_PATCH_IMPORT;q->linkIndex=k;q->slot=s;q->generation=p->gen[s];q->used=used;q->kind=finalKind;if(used)getpayload(p,s,&q->payload);else{memset(&q->payload,0,sizeof(q->payload));q->payload.generation=p->gen[s];}if(used)S.stat.slotUploads++;else S.stat.slotInvalidates++;}
static int payloadMetaEqual(const Pipe *p,uint32_t s,const MSXResidentPayload *x)
{ const double *r=crow((Pipe *)p,s); return r[0]==x->volume&&r[1]==x->hstep&&r[2]==x->hresponse&&r[3]==x->uresponse&&r[4]==x->dresponse&&p->id[s]==x->parcelId; }
static int payloadConcentrationEqual(const Pipe *p,uint32_t s,const MSXResidentPayload *x)
{ const double *r=crow((Pipe *)p,s);uint32_t m;for(m=0;m<S.stride;m++)if(r[5+m]!=x->c[m]||lrow((Pipe *)p,s)[5+m]!=x->lastc[m])return 0;return 1; }
static void writePayload(Pipe *p,uint32_t s,const MSXResidentPayload *x)
{ double*r=crow(p,s);uint32_t m;r[0]=x->volume;r[1]=x->hstep;r[2]=x->hresponse;r[3]=x->uresponse;r[4]=x->dresponse;p->id[s]=x->parcelId;for(m=0;m<S.stride;m++){r[5+m]=x->c[m];lrow(p,s)[5+m]=x->lastc[m];} }

MSXResidentStatus MSXresident_open(const char*path){FILE*f;char line[1024],id[EN_MAXID+1],want[EN_MAXID+1],h[65],*e;uint32_t row=0,k,cap,core,burst,guard,base=0;size_t b,n;Pipe*p;if(!path)return MSX_RESIDENT_ERR_ARGUMENT;freestate();f=fopen(path,"rb");if(!f)return MSX_RESIDENT_ERR_CSV;if(!fgets(line,sizeof(line),f))goto bad;e=strpbrk(line,"\r\n");if(!e||strcmp(line,"link_index,link_id,capacity,max_core_count,combined_burst_p99,guard,case_hash\n")&&strcmp(line,"link_index,link_id,capacity,max_core_count,combined_burst_p99,guard,case_hash\r\n"))goto bad;S.nlinks=(uint32_t)MSX.Nobjects[LINK];if(!mul((size_t)S.nlinks+1,sizeof(Pipe),&b))goto ov;S.p=(Pipe*)calloc(1,b);if(!S.p)goto mem;while(fgets(line,sizeof(line),f)){e=strpbrk(line,"\r\n");if(!e||(*e=='\r'&&e[1]!='\n')||e[1+( *e=='\r')])goto bad;*e=0;if(!csv(line,&k,id,&cap,&core,&burst,&guard,h)||k!=row+1||!k||k>S.nlinks||!cap||core>cap||(guard!=2&&guard!=4)||cap>(uint32_t)MSX.MaxSegments)goto bad;want[0]=0;if(ENgetlinkid((int)k,want)||strcmp(id,want))goto bad;if(!row)strcpy(S.hash,h);else if(strcmp(S.hash,h))goto bad;if(S.slots>(size_t)UINT32_MAX-cap)goto ov;S.slots+=(size_t)cap;p=&S.p[k];p->d.linkIndex=k;p->d.capacity=cap;p->d.head=p->d.tail=NONE;p->d.orient=1;row++;}fclose(f);if(row!=S.nlinks)goto clean;S.stride=(uint32_t)MSX.Nobjects[SPECIES]+1;if(S.stride>UINT32_MAX-5)goto ov2;S.rowWidth=5+S.stride;if(!mul((size_t)S.nlinks+1,sizeof(uint32_t),&b))goto ov2;S.dirty=(uint32_t*)malloc(b);S.capacity=(uint32_t*)calloc(1,b);S.base=(uint32_t*)calloc(1,b);if(!S.dirty||!S.capacity||!S.base)goto mem2;for(k=0;k<=S.nlinks;k++)S.dirty[k]=NONE;for(k=1;k<=S.nlinks;k++){S.capacity[k]=S.p[k].d.capacity;S.base[k]=base;base+=S.capacity[k];}if(!mul(S.nlinks,sizeof(*S.desc),&b))goto ov2;S.desc=(MSXResidentDescriptorPatch*)calloc(1,b);if(!mul(S.slots,sizeof(*S.slot),&b))goto ov2;S.slot=(MSXResidentSlotPatch*)calloc(1,b);if(!mul(S.slots,sizeof(*S.hand),&b))goto ov2;S.hand=(MSXResidentHandoffItem*)calloc(1,b);if(!S.desc||!S.slot||!S.hand)goto mem2;S.tx=(Tx*)calloc((size_t)S.nlinks+1,sizeof(*S.tx));if(!S.tx)goto mem2;for(k=1;k<=S.nlinks;k++){Tx*t=&S.tx[k];p=&S.p[k];t->capacity=p->d.capacity;t->item=(MSXResidentHandoffItem*)calloc(t->capacity,sizeof(*t->item));t->result=(MSXResidentHandoffResult*)calloc(t->capacity,sizeof(*t->result));t->boundary=(Pseg*)calloc(t->capacity,sizeof(*t->boundary));t->c=(double*)calloc((size_t)t->capacity*S.stride,sizeof(*t->c));t->lastc=(double*)calloc((size_t)t->capacity*S.stride,sizeof(*t->lastc));if(!t->item||!t->result||!t->boundary||!t->c||!t->lastc)goto mem2;if(!mul(p->d.capacity,2*(size_t)S.rowWidth,&n)||!mul(n,sizeof(double),&b))goto ov2;p->row=(double*)calloc(1,b);p->used=(unsigned char*)calloc(p->d.capacity,1);p->gen=(uint32_t*)calloc(p->d.capacity,sizeof(uint32_t));p->id=(uint64_t*)calloc(p->d.capacity,sizeof(uint64_t));p->patch=(uint32_t*)malloc((size_t)p->d.capacity*sizeof(uint32_t));p->obsI2s=(uint32_t*)malloc((size_t)p->d.capacity*sizeof(uint32_t));p->obsKeep=(uint32_t*)malloc((size_t)p->d.capacity*sizeof(uint32_t));if(!p->row||!p->used||!p->gen||!p->id||!p->patch||!p->obsI2s||!p->obsKeep)goto mem2;for(n=0;n<p->d.capacity;n++)p->patch[n]=NONE;}S.open=1;S.mode=MSX_RESIDENT_OFF;return MSX_RESIDENT_OK;bad:fclose(f);clean:freestate();return MSX_RESIDENT_ERR_CSV;ov:fclose(f);freestate();return MSX_RESIDENT_ERR_OVERFLOW;mem:fclose(f);freestate();return MSX_RESIDENT_ERR_MEMORY;ov2:freestate();return MSX_RESIDENT_ERR_OVERFLOW;mem2:freestate();return MSX_RESIDENT_ERR_MEMORY;}
void MSXresident_close(void){freestate();}int MSXresident_isOpen(void){return S.open;}void MSXresident_setMode(MSXResidentMode m,int strict){S.mode=m<=MSX_RESIDENT_RESIDENT?m:MSX_RESIDENT_OFF;S.strict=!!strict;}MSXResidentMode MSXresident_mode(void){return S.mode;}
MSXResidentStatus MSXresident_validateFeatures(int w,int d){if(!S.open)return MSX_RESIDENT_DISABLED;return w?MSX_RESIDENT_ERR_UNSUPPORTED_WALL:d?MSX_RESIDENT_ERR_UNSUPPORTED_DISPERSION:MSX_RESIDENT_OK;}MSXResidentStatus MSXresident_verifyCaseHash(const char*h){char want[65];if(!S.open||!h||!hashok(h))return MSX_RESIDENT_ERR_CASE_HASH;memcpy(want,h,65);hashupper(want);return!strcmp(want,S.hash)?MSX_RESIDENT_OK:MSX_RESIDENT_ERR_CASE_HASH;}
static MSXResidentStatus checkcsv(const char *path,int guard,const char *want)
{ return validatecsv(path,guard,want); }
MSXResidentStatus MSXresident_validateConfig(const MSXResidentConfig *c)
{ char want[65]; MSXResidentStatus z;
  if(!c || c->mode<MSX_RESIDENT_OFF || c->mode>MSX_RESIDENT_RESIDENT)return MSX_RESIDENT_ERR_CONFIG;
  if(c->mode==MSX_RESIDENT_OFF)return MSX_RESIDENT_OK;
  if(!c->capacityFile || !c->capacityFile[0])return MSX_RESIDENT_ERR_PATH;
  if(c->guard!=2 && c->guard!=4)return MSX_RESIDENT_ERR_GUARD;
  if(c->segmentStorage!=SEG_STORAGE_HYBRID)return MSX_RESIDENT_ERR_CONFIG;
  if(c->solver!=ROS2 || c->gpuSolver!=ROS2)return MSX_RESIDENT_ERR_SOLVER;
  if(c->gpuScope!=GPU_PIPE_SEGMENT)return MSX_RESIDENT_ERR_SCOPE;
  if(!c->gpuReact || !c->gpuOde || !c->gpuEquil || !c->gpuFormula)return MSX_RESIDENT_ERR_CONFIG;
  if(c->hasWall)return MSX_RESIDENT_ERR_UNSUPPORTED_WALL;
  if(c->hasDispersion)return MSX_RESIDENT_ERR_UNSUPPORTED_DISPERSION;
  if(!c->caseHash || !hashok(c->caseHash))return MSX_RESIDENT_ERR_CASE_HASH;
  memcpy(want,c->caseHash,65);hashupper(want);
  z=checkcsv(c->capacityFile,c->guard,want);if(z)return z;
  if(S.open)return MSXresident_verifyCaseHash(want);
  return MSX_RESIDENT_OK; }
MSXResidentStatus MSXresident_observePipe(uint32_t k,const uint64_t*id,const MSXResidentPayload*in,uint32_t count,int32_t orient)
{
    Pipe *p; MSXResidentStatus z; uint32_t *i2s, *keep, i, s, j, needed = 0;
    int topologyChanged = 0;
    if (S.mode == MSX_RESIDENT_OFF) return MSX_RESIDENT_OK;
    z = pipeok(k, &p); if (z) return z;
    if ((count && (!id || !in)) || count > p->d.capacity ||
        (orient != 1 && orient != -1)) return MSX_RESIDENT_ERR_ARGUMENT;
    i2s = p->obsI2s; keep = p->obsKeep;
    memset(keep, 0, (size_t)p->d.capacity * sizeof(*keep));
    for (i = 0; i < count; ++i)
    {
        if (!id[i] || !in[i].c || !in[i].lastc) return MSX_RESIDENT_ERR_ARGUMENT;
        for (j = 0; j < i; ++j) if (id[j] == id[i]) return MSX_RESIDENT_ERR_ARGUMENT;
        i2s[i] = NONE;
        s = (uint32_t)find(p, id[i]);
        if (s != NONE) { i2s[i] = s; keep[s] = 1; }
    }
    /* Preserve existing rows.  New rows are allocated only at a free cyclic
       neighbour; an unrepresentable middle insertion is rejected instead of
       silently moving surviving slots. */
    for (i = 0; i < count; ++i) if (i2s[i] == NONE)
    {
        uint32_t candidate = NONE;
        for (j = i; j > 0; --j) if (i2s[j - 1] != NONE)
        {
            candidate = nextslot(p, i2s[j - 1], orient); break;
        }
        if (candidate == NONE) for (j = i + 1; j < count; ++j)
            if (i2s[j] != NONE) { candidate = prevslot(p, i2s[j], orient); break; }
        if (candidate == NONE)
            for (s = 0; s < p->d.capacity; ++s)
                if (!p->used[s] || !keep[s]) { candidate = s; break; }
        if (candidate == NONE || (p->used[candidate] && keep[candidate]))
            return MSX_RESIDENT_ERR_CAPACITY;
        for (j = 0; j < i; ++j) if (i2s[j] == candidate)
            return MSX_RESIDENT_ERR_CAPACITY;
        i2s[i] = candidate; keep[candidate] = 1;
    }
    if (count)
    {
        for (i = 1; i < count; ++i)
            if (i2s[i] != nextslot(p, i2s[i - 1], orient))
                return MSX_RESIDENT_ERR_ARGUMENT;
        if (p->d.count != count || p->d.head != i2s[0] ||
            p->d.tail != i2s[count - 1] || p->d.orient != orient)
            topologyChanged = 1;
    }
    else if (p->d.count || p->d.head != NONE || p->d.tail != NONE)
        topologyChanged = 1;
    for (s = 0; s < p->d.capacity; ++s)
        if (p->used[s] != keep[s]) { topologyChanged = 1; break; }
    if (topologyChanged && p->d.epoch == UINT64_MAX) return MSX_RESIDENT_ERR_OVERFLOW;
    if (topologyChanged && S.dirty[k] == NONE && S.ndesc >= S.nlinks)
        return MSX_RESIDENT_ERR_CAPACITY;
    for (s = 0; s < p->d.capacity; ++s)
        if (p->used[s] && !keep[s] && p->patch[s] == NONE) ++needed;
    for (i = 0; i < count; ++i)
    {
        s = i2s[i];
        if (!p->used[s] || p->id[s] != id[i] ||
            !payloadMetaEqual(p, s, &in[i]) ||
            !payloadConcentrationEqual(p, s, &in[i]))
            if (p->patch[s] == NONE) ++needed;
        if ((!p->used[s] || p->id[s] != id[i]) && p->gen[s] == UINT32_MAX)
            return MSX_RESIDENT_ERR_GENERATION;
    }
    if (needed > S.slots - S.nslot) return MSX_RESIDENT_ERR_CAPACITY;
    /* Commit is allocation-free after the complete preflight above. */
    for (s = 0; s < p->d.capacity; ++s) if (p->used[s] && !keep[s])
    {
        p->used[s] = 0; p->id[s] = 0;
        memset(crow(p, s), 0, 2 * (size_t)S.rowWidth * sizeof(double));
        markslot(k, s, 0, MSX_RESIDENT_PATCH_INVALIDATE);
    }
    for (i = 0; i < count; ++i)
    {
        MSXResidentPatchKind kind = MSX_RESIDENT_PATCH_META;
        s = i2s[i];
        if (!p->used[s] || p->id[s] != id[i])
        {
            p->gen[s]++; p->used[s] = 1; kind = MSX_RESIDENT_PATCH_IMPORT;
        }
        else if (!payloadConcentrationEqual(p, s, &in[i])) kind = MSX_RESIDENT_PATCH_IMPORT;
        else if (!payloadMetaEqual(p, s, &in[i])) kind = MSX_RESIDENT_PATCH_META;
        if (kind != MSX_RESIDENT_PATCH_META || !payloadMetaEqual(p, s, &in[i]) ||
            !payloadConcentrationEqual(p, s, &in[i]))
        {
            writePayload(p, s, &in[i]);
            markslot(k, s, 1, kind);
        }
    }
    if (topologyChanged)
    {
        p->d.count = count; p->d.orient = orient;
        p->d.head = count ? i2s[0] : NONE;
        p->d.tail = count ? i2s[count - 1] : NONE;
        p->d.epoch++; markdesc(k);
    }
    return MSX_RESIDENT_OK;
}
MSXResidentStatus MSXresident_stageInsert(uint32_t k,int side,const MSXResidentPayload *x,
                                          uint32_t *slotOut,uint32_t *generationOut)
{
    Pipe *p; MSXResidentStatus z; uint32_t s;
    if (!x || side < 0 || side > 1 || !x->parcelId || !x->c || !x->lastc)
        return MSX_RESIDENT_ERR_ARGUMENT;
    if (S.mode == MSX_RESIDENT_OFF) return MSX_RESIDENT_OK;
    z = pipeok(k, &p); if (z) return z;
    if (find(p, x->parcelId) >= 0) return MSX_RESIDENT_ERR_ARGUMENT;
    if (p->d.count >= p->d.capacity || p->d.epoch == UINT64_MAX)
        return p->d.epoch == UINT64_MAX ? MSX_RESIDENT_ERR_OVERFLOW : MSX_RESIDENT_ERR_CAPACITY;
    if (p->d.count == 0) s = 0;
    else s = side ? nextslot(p, p->d.tail, p->d.orient) : prevslot(p, p->d.head, p->d.orient);
    if (p->used[s] || (p->patch[s] != NONE && S.nslot >= S.slots))
        return MSX_RESIDENT_ERR_CAPACITY;
    if (p->gen[s] == UINT32_MAX) return MSX_RESIDENT_ERR_GENERATION;
    if (p->patch[s] == NONE && S.nslot >= S.slots) return MSX_RESIDENT_ERR_CAPACITY;
    p->gen[s]++; p->used[s] = 1; writePayload(p, s, x);
    if (!p->d.count) p->d.head = p->d.tail = s;
    else if (side) p->d.tail = s;
    else p->d.head = s;
    p->d.count++; p->d.epoch++;
    markslot(k, s, 1, MSX_RESIDENT_PATCH_IMPORT); markdesc(k);
    if (slotOut) *slotOut = s;
    if (generationOut) *generationOut = p->gen[s];
    return MSX_RESIDENT_OK;
}
MSXResidentStatus MSXresident_stageRemove(uint32_t k,uint32_t s,uint32_t generation)
{
    Pipe *p; MSXResidentStatus z; int side;
    if (S.mode == MSX_RESIDENT_OFF) return MSX_RESIDENT_OK;
    z = pipeok(k, &p); if (z) return z;
    if (s >= p->d.capacity || !p->used[s] || p->gen[s] != generation)
        return MSX_RESIDENT_ERR_GENERATION;
    if (s == p->d.head) side = 0;
    else if (s == p->d.tail) side = 1;
    else return MSX_RESIDENT_ERR_ARGUMENT;
    if (p->d.epoch == UINT64_MAX) return MSX_RESIDENT_ERR_OVERFLOW;
    p->used[s] = 0; p->id[s] = 0;
    memset(crow(p, s), 0, 2 * (size_t)S.rowWidth * sizeof(double));
    if (p->d.count == 1) p->d.head = p->d.tail = NONE;
    else if (!side) p->d.head = nextslot(p, p->d.head, p->d.orient);
    else p->d.tail = prevslot(p, p->d.tail, p->d.orient);
    p->d.count--; p->d.epoch++;
    markslot(k, s, 0, MSX_RESIDENT_PATCH_INVALIDATE); markdesc(k);
    return MSX_RESIDENT_OK;
}
static MSXResidentStatus validateRemoveBatchInternal(uint32_t k,
                                                     const MSXResidentHandoffItem *items,
                                                     uint32_t count,
                                                     Pipe **out)
{
    Pipe *p; MSXResidentStatus z; uint32_t i, needed = 0;
    uint32_t head, tail, remaining;
    if (S.mode == MSX_RESIDENT_OFF) return MSX_RESIDENT_OK;
    if (count && !items) return MSX_RESIDENT_ERR_ARGUMENT;
    z = pipeok(k, &p); if (z) return z;
    if (out) *out = p;
    if (!count) return MSX_RESIDENT_OK;
    if (count > p->d.count || S.nslot > S.slots)
        return MSX_RESIDENT_ERR_CAPACITY;
    if ((uint64_t)count > UINT64_MAX - p->d.epoch)
        return MSX_RESIDENT_ERR_OVERFLOW;
    if (S.dirty[k] == NONE && S.ndesc >= S.nlinks)
        return MSX_RESIDENT_ERR_CAPACITY;
    head = p->d.head; tail = p->d.tail; remaining = p->d.count;
    for (i = 0; i < count; ++i)
    {
        uint32_t s = items[i].slot;
        int side;
        if (items[i].linkIndex != k || s >= p->d.capacity ||
            !p->used[s] || p->gen[s] != items[i].generation ||
            items[i].pipeEpoch != p->d.epoch)
            return MSX_RESIDENT_ERR_GENERATION;
        if (!remaining) return MSX_RESIDENT_ERR_ARGUMENT;
        if (s == head) side = 0;
        else if (s == tail) side = 1;
        else return MSX_RESIDENT_ERR_ARGUMENT;
        if (p->patch[s] == NONE) ++needed;
        if (remaining == 1) head = tail = NONE;
        else if (!side) head = nextslot(p, head, p->d.orient);
        else tail = prevslot(p, tail, p->d.orient);
        --remaining;
    }
    if (needed > S.slots - S.nslot) return MSX_RESIDENT_ERR_CAPACITY;
    return MSX_RESIDENT_OK;
}
MSXResidentStatus MSXresident_validateRemoveBatch(
    uint32_t k, const MSXResidentHandoffItem *items, uint32_t count)
{ return validateRemoveBatchInternal(k, items, count, NULL); }
MSXResidentStatus MSXresident_stageRemoveBatch(
    uint32_t k, const MSXResidentHandoffItem *items, uint32_t count)
{
    Pipe *p; MSXResidentStatus z; uint32_t i, head, tail, remaining;
    z = validateRemoveBatchInternal(k, items, count, &p);
    if (z || S.mode == MSX_RESIDENT_OFF || !count) return z;
    head = p->d.head; tail = p->d.tail; remaining = p->d.count;
    for (i = 0; i < count; ++i)
    {
        uint32_t s = items[i].slot;
        int side;
        if (s == head) side = 0;
        else if (s == tail) side = 1;
        else return MSX_RESIDENT_ERR_ARGUMENT;
        p->used[s] = 0; p->id[s] = 0;
        memset(crow(p, s), 0, 2 * (size_t)S.rowWidth * sizeof(double));
        if (remaining == 1) head = tail = NONE;
        else if (!side) head = nextslot(p, head, p->d.orient);
        else tail = prevslot(p, tail, p->d.orient);
        --remaining;
        markslot(k, s, 0, MSX_RESIDENT_PATCH_INVALIDATE);
    }
    p->d.head = head; p->d.tail = tail; p->d.count = remaining;
    p->d.epoch += count; markdesc(k);
    return MSX_RESIDENT_OK;
}
void MSXresident_poison(void) { if (S.open) S.poisoned = 1; }

MSXResidentStatus MSXresident_auditPoisonCpuMirrors(void)
{
    uint32_t k, s, m;
    if (!S.open) return MSX_RESIDENT_DISABLED;
    /* This is intentionally an explicit audit-only walk.  Production runs
       never call it unless the runtime audit environment switch is enabled. */
    for (k = 1; k <= S.nlinks; ++k)
        for (s = 0; s < S.p[k].d.capacity; ++s)
            if (S.p[k].used[s])
            {
                double *c = crow(&S.p[k], s) + 5;
                double *lastc = lrow(&S.p[k], s) + 5;
                for (m = 0; m < S.stride; ++m)
                    c[m] = lastc[m] = NAN;
            }
    return MSX_RESIDENT_OK;
}

MSXResidentStatus MSXresident_stageClear(uint32_t k)
{
    Pipe *p; MSXResidentStatus z; uint32_t s, used = 0, reserve = 0;
    if (S.mode == MSX_RESIDENT_OFF) return MSX_RESIDENT_OK;
    z = pipeok(k, &p); if (z) return z;
    for (s = 0; s < p->d.capacity; ++s)
        if (p->used[s])
        {
            ++used;
            if (p->patch[s] == NONE) ++reserve;
        }
    if (!used && !p->d.count) return MSX_RESIDENT_OK;
    if (used != p->d.count) return MSX_RESIDENT_ERR_GENERATION;
    if (p->d.epoch == UINT64_MAX || reserve > S.slots - S.nslot)
        return p->d.epoch == UINT64_MAX ? MSX_RESIDENT_ERR_OVERFLOW :
                                          MSX_RESIDENT_ERR_CAPACITY;
    /* All fallible checks are complete before the first invalidation. */
    for (s = 0; s < p->d.capacity; ++s)
        if (p->used[s])
        {
            p->used[s] = 0; p->id[s] = 0;
            memset(crow(p, s), 0, 2 * (size_t)S.rowWidth * sizeof(double));
            markslot(k, s, 0, MSX_RESIDENT_PATCH_INVALIDATE);
        }
    p->d.count = 0; p->d.head = p->d.tail = NONE;
    p->d.epoch++; markdesc(k);
    return MSX_RESIDENT_OK;
}
MSXResidentStatus MSXresident_stageMeta(uint32_t k,uint32_t s,uint32_t generation,
                                         const MSXResidentPayload *x)
{
    Pipe *p; MSXResidentStatus z; double *r;
    if (!x) return MSX_RESIDENT_ERR_ARGUMENT;
    if (S.mode == MSX_RESIDENT_OFF) return MSX_RESIDENT_OK;
    z = pipeok(k, &p); if (z) return z;
    if (s >= p->d.capacity || !p->used[s] || p->gen[s] != generation)
        return MSX_RESIDENT_ERR_GENERATION;
    if (x->parcelId && x->parcelId != p->id[s]) return MSX_RESIDENT_ERR_GENERATION;
    r = crow(p, s);
    if (r[0] == x->volume && r[1] == x->hstep && r[2] == x->hresponse &&
        r[3] == x->uresponse && r[4] == x->dresponse)
        return MSX_RESIDENT_OK;
    r[0] = x->volume; r[1] = x->hstep; r[2] = x->hresponse;
    r[3] = x->uresponse; r[4] = x->dresponse;
    markslot(k, s, 1, MSX_RESIDENT_PATCH_META);
    return MSX_RESIDENT_OK;
}
MSXResidentStatus MSXresident_stageReverse(uint32_t k)
{
    Pipe *p; MSXResidentStatus z;
    if (S.mode == MSX_RESIDENT_OFF) return MSX_RESIDENT_OK;
    z = pipeok(k, &p); if (z) return z;
    if (p->d.count <= 1) return MSX_RESIDENT_OK;
    if (p->d.epoch == UINT64_MAX) return MSX_RESIDENT_ERR_OVERFLOW;
    { uint32_t s = p->d.head; p->d.head = p->d.tail; p->d.tail = s; }
    p->d.orient = -p->d.orient; p->d.epoch++; markdesc(k);
    return MSX_RESIDENT_OK;
}
MSXResidentStatus MSXresident_getSlotForParcel(uint32_t k,uint64_t parcelId,
                                                uint32_t *slotOut,uint32_t *generationOut)
{
    Pipe *p; int s; MSXResidentStatus z;
    if (!parcelId) return MSX_RESIDENT_ERR_ARGUMENT;
    z = pipeok(k, &p); if (z) return z; s = find(p, parcelId);
    if (s < 0) return MSX_RESIDENT_ERR_GENERATION;
    if (slotOut) *slotOut = (uint32_t)s;
    if (generationOut) *generationOut = p->gen[s];
    return MSX_RESIDENT_OK;
}
MSXResidentStatus MSXresident_getSlotGeneration(uint32_t k,uint32_t s,uint32_t *generationOut)
{
    Pipe *p; MSXResidentStatus z;
    if (!generationOut) return MSX_RESIDENT_ERR_ARGUMENT;
    z = pipeok(k, &p); if (z) return z;
    if (s >= p->d.capacity || !p->used[s]) return MSX_RESIDENT_ERR_GENERATION;
    *generationOut = p->gen[s]; return MSX_RESIDENT_OK;
}
MSXResidentStatus MSXresident_getSlotIdentity(uint32_t k,uint32_t s,
                                               uint32_t *generationOut,
                                               uint64_t *parcelIdOut,
                                               uint64_t *epochOut)
{
    Pipe *p; MSXResidentStatus z;
    z = pipeok(k, &p); if (z) return z;
    if (s >= p->d.capacity || !p->used[s])
    {
        S.stat.generationFailures++;
        return MSX_RESIDENT_ERR_GENERATION;
    }
    if (generationOut) *generationOut = p->gen[s];
    if (parcelIdOut) *parcelIdOut = p->id[s];
    if (epochOut) *epochOut = p->d.epoch;
    return MSX_RESIDENT_OK;
}
MSXResidentStatus MSXresident_getSlotPayload(uint32_t k,uint32_t s,uint32_t g,MSXResidentPayload*x){Pipe*p;MSXResidentStatus z;if(!x)return MSX_RESIDENT_ERR_ARGUMENT;z=pipeok(k,&p);if(z)return z;if(s>=p->d.capacity||!p->used[s]||p->gen[s]!=g){S.stat.generationFailures++;return MSX_RESIDENT_ERR_GENERATION;}getpayload(p,s,x);return MSX_RESIDENT_OK;}
MSXResidentStatus MSXresident_applyActivePayload(const MSXResidentActiveRow*r,const MSXResidentPayload*x,uint32_t n){uint32_t i,m;Pipe*p;MSXResidentStatus z;if((!r||!x)&&n)return MSX_RESIDENT_ERR_ARGUMENT;for(i=0;i<n;i++){z=pipeok(r[i].linkIndex,&p);if(z)return z;if(r[i].globalRow!=S.base[r[i].linkIndex]+r[i].slot||r[i].slot>=p->d.capacity||!p->used[r[i].slot]||p->gen[r[i].slot]!=r[i].generation||p->d.epoch!=r[i].descriptorEpoch||p->id[r[i].slot]!=r[i].parcelId||!x[i].c||!x[i].lastc){S.stat.generationFailures++;return MSX_RESIDENT_ERR_GENERATION;}}for(i=0;i<n;i++){p=&S.p[r[i].linkIndex];{double*q=crow(p,r[i].slot);q[0]=x[i].volume;q[1]=x[i].hstep;q[2]=x[i].hresponse;q[3]=x[i].uresponse;q[4]=x[i].dresponse;for(m=0;m<S.stride;m++){q[5+m]=x[i].c[m];lrow(p,r[i].slot)[5+m]=x[i].lastc[m];}}}return MSX_RESIDENT_OK;}
uint64_t MSXresident_testAllocationCount(void){return ResidentAllocationCount;}
void MSXresident_testResetAllocationCount(void){ResidentAllocationCount=0;}
void MSXresident_testFailAllocationAfter(int64_t allocationIndex){ResidentAllocationFailAfter=allocationIndex;}
MSXResidentStatus MSXresident_getPatches(MSXResidentPatchBatch*b){if(!b)return MSX_RESIDENT_ERR_ARGUMENT;memset(b,0,sizeof(*b));if(!S.open)return MSX_RESIDENT_DISABLED;b->descriptor=S.desc;b->descriptorCount=S.ndesc;b->slot=S.slot;b->slotCount=S.nslot;return MSX_RESIDENT_OK;}
void MSXresident_clearPatches(void)
{ uint32_t i,k,s; if(!S.open)return; for(i=0;i<S.ndesc;i++){k=S.desc[i].linkIndex;if(k&&k<=S.nlinks)S.dirty[k]=NONE;} for(i=0;i<S.nslot;i++){k=S.slot[i].linkIndex;s=S.slot[i].slot;if(k&&k<=S.nlinks&&s<S.p[k].d.capacity)S.p[k].patch[s]=NONE;} S.ndesc=S.nslot=0; }
MSXResidentStatus MSXresident_getLayout(MSXResidentLayout *o){uint32_t k;if(!o)return MSX_RESIDENT_ERR_ARGUMENT;memset(o,0,sizeof(*o));if(!S.open||!S.capacity||!S.base)return MSX_RESIDENT_DISABLED;if(!S.guard){S.guard=(uint32_t*)resident_calloc((size_t)S.nlinks+1,sizeof(*S.guard));if(!S.guard)return MSX_RESIDENT_ERR_MEMORY;for(k=1;k<=S.nlinks;k++)S.guard[k]=S.p[k].guard;}o->nLinks=S.nlinks;o->totalSlots=(uint32_t)S.slots;o->speciesStride=S.stride;o->capacity=S.capacity;o->base=S.base;o->guard=S.guard;return MSX_RESIDENT_OK;}
static MSXResidentStatus activeIteratorStep(MSXResidentActiveIterator *it,
                                            MSXResidentActiveRow *row)
{
    Pipe *p;
    uint32_t s;
    if (!it || !row) return MSX_RESIDENT_ERR_ARGUMENT;
    if (!it->active || !S.open) return MSX_RESIDENT_DISABLED;
    if (it->done) return MSX_RESIDENT_ITER_END;
    while (it->linkIndex <= it->nlinks)
    {
        p = &S.p[it->linkIndex];
        if (it->seen == 0)
        {
            if (p->d.count > p->d.capacity ||
                (p->d.count && p->d.head >= p->d.capacity))
                return MSX_RESIDENT_ERR_CAPACITY;
            it->slot = p->d.head;
            it->orient = p->d.orient;
        }
        if (it->seen < p->d.count)
        {
            s = it->slot;
            if (s >= p->d.capacity || !p->used[s])
                return MSX_RESIDENT_ERR_GENERATION;
            row->linkIndex = it->linkIndex;
            row->slot = s;
            row->globalRow = it->base + s;
            row->generation = p->gen[s];
            row->descriptorHead = p->d.head;
            row->descriptorCount = p->d.count;
            row->descriptorOrient = p->d.orient;
            row->descriptorEpoch = p->d.epoch;
            row->parcelId = p->id[s];
            row->volume = crow(p, s)[0];
            ++it->seen;
            ++it->count;
            ++it->total;
            it->slot = (uint32_t)(((int64_t)s + it->orient +
                                   (int64_t)p->d.capacity) %
                                  (int64_t)p->d.capacity);
            return MSX_RESIDENT_OK;
        }
        if (it->base > UINT32_MAX - p->d.capacity)
            return MSX_RESIDENT_ERR_OVERFLOW;
        it->base += p->d.capacity;
        ++it->linkIndex;
        it->seen = 0;
        it->slot = 0;
    }
    if (it->total != it->count || it->count > S.slots)
        return MSX_RESIDENT_ERR_OVERFLOW;
    it->done = 1;
    return MSX_RESIDENT_ITER_END;
}

MSXResidentStatus MSXresident_beginActiveIterator(MSXResidentActiveIterator *it)
{
    if (!it) return MSX_RESIDENT_ERR_ARGUMENT;
    memset(it, 0, sizeof(*it));
    if (!S.open) return MSX_RESIDENT_DISABLED;
    it->active = 1;
    it->linkIndex = 1;
    it->nlinks = S.nlinks;
    return MSX_RESIDENT_OK;
}

MSXResidentStatus MSXresident_validateActiveIterator(MSXResidentActiveIterator *it)
{
    MSXResidentActiveIterator q;
    MSXResidentActiveRow row;
    MSXResidentStatus z;
    if (!it) return MSX_RESIDENT_ERR_ARGUMENT;
    if (!it->active || !S.open) return MSX_RESIDENT_DISABLED;
    q = *it;
    q.linkIndex = 1;
    q.slot = q.seen = q.base = q.count = q.total = 0;
    q.done = 0;
    while ((z = activeIteratorStep(&q, &row)) == MSX_RESIDENT_OK) { }
    if (z != MSX_RESIDENT_ITER_END) return z;
    /* Keep validation's total separate from the cursor counters.  The caller
       may immediately count() and then consume the same iterator with next;
       next must start at the first row, not after the validation pass. */
    it->linkIndex = 1;
    it->slot = it->seen = it->base = it->count = it->total = 0;
    it->done = 0;
    it->expectedCount = q.count;
    it->validated = 1;
    return MSX_RESIDENT_OK;
}

MSXResidentStatus MSXresident_countActiveIterator(
    const MSXResidentActiveIterator *it, uint32_t *count)
{
    if (!it || !count) return MSX_RESIDENT_ERR_ARGUMENT;
    if (!it->active || !S.open) return MSX_RESIDENT_DISABLED;
    if (!it->validated) return MSX_RESIDENT_ERR_ARGUMENT;
    *count = it->expectedCount;
    return MSX_RESIDENT_OK;
}

MSXResidentStatus MSXresident_nextActive(MSXResidentActiveIterator *it,
                                         MSXResidentActiveRow *row)
{
    return activeIteratorStep(it, row);
}

MSXResidentStatus MSXresident_rawActiveCount(uint64_t *count, uint32_t *links)
{
    uint32_t k;
    uint64_t total = 0, n;
    if (!count) return MSX_RESIDENT_ERR_ARGUMENT;
    *count = 0;
    if (links) *links = 0;
    if (!S.open) return MSX_RESIDENT_DISABLED;
    for (k = 1; k <= S.nlinks; ++k)
    {
        n = (uint64_t)S.p[k].d.count;
        if (UINT64_MAX - total < n) total = UINT64_MAX;
        else total += n;
    }
    *count = total;
    if (links) *links = S.nlinks;
    return MSX_RESIDENT_OK;
}

MSXResidentStatus MSXresident_nextActiveBatch(MSXResidentActiveIterator *it,
                                               MSXResidentActiveRow *rows,
                                               uint32_t cap, uint32_t *count)
{
    MSXResidentStatus z;
    uint32_t n = 0;
    if (!count || (!rows && cap)) return MSX_RESIDENT_ERR_ARGUMENT;
    *count = 0;
    if (!it || !it->active || !S.open) return MSX_RESIDENT_DISABLED;
    if (!cap) return MSX_RESIDENT_ERR_ARGUMENT;
    while (n < cap)
    {
        /* Once the current pipe has been exhausted, leave its transition to
           activeIteratorStep on the next call.  This guarantees no batch
           crosses a link boundary while preserving the old cursor order. */
        if (n && it->linkIndex <= it->nlinks &&
            it->seen >= S.p[it->linkIndex].d.count)
            break;
        z = activeIteratorStep(it, &rows[n]);
        if (z == MSX_RESIDENT_OK) { ++n; continue; }
        *count = n;
        return z;
    }
    *count = n;
    return n ? MSX_RESIDENT_OK : MSX_RESIDENT_ITER_END;
}

MSXResidentStatus MSXresident_enumerateActive(MSXResidentActiveRow *rows,
                                               uint32_t cap,
                                               uint32_t *count)
{
    MSXResidentActiveIterator it;
    MSXResidentActiveRow row;
    MSXResidentStatus z;
    uint32_t n = 0;
    if (!count || (!rows && cap)) return MSX_RESIDENT_ERR_ARGUMENT;
    *count = 0;
    z = MSXresident_beginActiveIterator(&it);
    if (z != MSX_RESIDENT_OK) return z;
    while ((z = MSXresident_nextActive(&it, &row)) == MSX_RESIDENT_OK)
    {
        if (n >= cap) return MSX_RESIDENT_ERR_CAPACITY;
        if (rows) rows[n] = row;
        ++n;
    }
    if (z != MSX_RESIDENT_ITER_END) return z;
    *count = n;
    return MSX_RESIDENT_OK;
}
MSXResidentStatus MSXresident_getInitialBatch(MSXResidentPatchBatch *b){uint32_t k,s,n=0;if(!b)return MSX_RESIDENT_ERR_ARGUMENT;memset(b,0,sizeof(*b));if(!S.open)return MSX_RESIDENT_DISABLED;if(!S.initialStaging||S.initialCommitted)return MSX_RESIDENT_ERR_ARGUMENT;/* Initial image construction reuses the patch arenas; discard the staging
   indices before filling the dense all-slot image so later incremental
   markslot() calls cannot address overwritten records. */MSXresident_clearPatches();for(k=1;k<=S.nlinks;k++){S.desc[k-1].linkIndex=k;S.desc[k-1].descriptor=S.p[k].d;for(s=0;s<S.p[k].d.capacity;s++){MSXResidentSlotPatch*q=&S.slot[n++];q->linkIndex=k;q->slot=s;q->generation=S.p[k].gen[s];q->used=S.p[k].used[s];q->kind=q->used?MSX_RESIDENT_PATCH_IMPORT:MSX_RESIDENT_PATCH_INVALIDATE;getpayload(&S.p[k],s,&q->payload);}}b->descriptor=S.desc;b->descriptorCount=S.nlinks;b->slot=S.slot;b->slotCount=n;return MSX_RESIDENT_OK;}
MSXResidentStatus MSXresident_beginInitialImage(void)
{ if(!S.open||S.initialStaging||S.initialCommitted)return MSX_RESIDENT_ERR_ARGUMENT; S.initialStaging=1; return MSX_RESIDENT_OK; }
MSXResidentStatus MSXresident_stageInitialPipe(uint32_t k,const uint64_t *id,
    const MSXResidentPayload *payload,uint32_t count,int32_t orient)
{ if(!S.initialStaging||S.initialCommitted)return MSX_RESIDENT_ERR_ARGUMENT; return MSXresident_observePipe(k,id,payload,count,orient); }
MSXResidentStatus MSXresident_commitInitialImage(void)
{ if(!S.initialStaging||S.initialCommitted)return MSX_RESIDENT_ERR_ARGUMENT; MSXresident_clearPatches(); S.initialStaging=0; S.initialCommitted=1; return MSX_RESIDENT_OK; }
void MSXresident_abortInitialImage(void)
{ if(!S.open)return; S.initialStaging=0; S.initialCommitted=0; MSXresident_clearPatches(); }
static uint32_t nth(Pipe*p,uint32_t base,int dir,uint32_t n){int64_t q=(int64_t)base+(int64_t)dir*n;while(q<0)q+=p->d.capacity;return(uint32_t)(q%p->d.capacity);}
MSXResidentStatus MSXresident_planHandoff(uint32_t k,double flow,double dt,uint32_t side,double bv,int strict,MSXResidentHandoffPlan*o){Pipe*p;MSXResidentStatus z;double need;if(!o||dt<0||side>1)return MSX_RESIDENT_ERR_ARGUMENT;memset(o,0,sizeof(*o));if(S.mode==MSX_RESIDENT_OFF)return MSX_RESIDENT_OK;z=pipeok(k,&p);if(z)return z;need=fabs(flow)*dt;o->linkIndex=k;o->boundarySide=side;o->pipeEpoch=p->d.epoch;o->flowVolume=need;if(need<=bv)return MSX_RESIDENT_OK;need-=bv;S.nh=0;while(need>0&&S.nh<p->d.count){uint32_t s=nth(p,side?p->d.tail:p->d.head,side?-p->d.orient:p->d.orient,S.nh);if(!p->used[s])return MSX_RESIDENT_ERR_GENERATION;S.hand[S.nh].linkIndex=k;S.hand[S.nh].slot=s;S.hand[S.nh].generation=p->gen[s];S.hand[S.nh].boundarySide=side;S.hand[S.nh].pipeEpoch=p->d.epoch;S.hand[S.nh].requestedVolume=crow(p,s)[0];need-=crow(p,s)[0];S.nh++;}S.stat.handoffRequests++;if(need>0){S.stat.fallbacks++;return strict||S.strict?MSX_RESIDENT_ERR_CAPACITY:MSX_RESIDENT_FALLBACK_WHOLE_LINK;}o->itemCount=S.nh;o->item=S.hand;return MSX_RESIDENT_OK;}
MSXResidentStatus MSXresident_planHandoffInto(uint32_t k,double flow,double dt,uint32_t side,double bv,int strict,MSXResidentHandoffItem *item,uint32_t cap,MSXResidentHandoffPlan *o){MSXResidentStatus z;MSXResidentHandoffPlan q;if(!o)return MSX_RESIDENT_ERR_ARGUMENT;z=MSXresident_planHandoff(k,flow,dt,side,bv,strict,&q);if(z!=MSX_RESIDENT_OK)return z;if(q.itemCount>cap||(!item&&q.itemCount))return MSX_RESIDENT_ERR_CAPACITY;*o=q;o->item=item;if(q.itemCount)memcpy(item,q.item,(size_t)q.itemCount*sizeof(*item));return MSX_RESIDENT_OK;}
MSXResidentStatus MSXresident_planAllHandoffInto(uint32_t k,uint32_t side,MSXResidentHandoffItem *item,uint32_t cap,MSXResidentHandoffPlan *o){Pipe*p;MSXResidentStatus z;uint32_t i;if(!o||side>1)return MSX_RESIDENT_ERR_ARGUMENT;memset(o,0,sizeof(*o));z=pipeok(k,&p);if(z)return z;if(p->d.count>cap||(!item&&p->d.count))return MSX_RESIDENT_ERR_CAPACITY;o->linkIndex=k;o->boundarySide=side;o->pipeEpoch=p->d.epoch;o->itemCount=p->d.count;o->item=item;for(i=0;i<p->d.count;i++){uint32_t s=nth(p,side?p->d.tail:p->d.head,side?-p->d.orient:p->d.orient,i);if(!p->used[s])return MSX_RESIDENT_ERR_GENERATION;item[i].linkIndex=k;item[i].slot=s;item[i].generation=p->gen[s];item[i].boundarySide=side;item[i].pipeEpoch=p->d.epoch;item[i].requestedVolume=crow(p,s)[0];}return MSX_RESIDENT_OK;}
MSXResidentStatus MSXresident_applyHandoff(const MSXResidentHandoffResult*r,uint32_t n){uint32_t i,m;Pipe*p;MSXResidentStatus z;if(!r&&n)return MSX_RESIDENT_ERR_ARGUMENT;for(i=0;i<n;i++){z=pipeok(r[i].linkIndex,&p);if(z)return z;if(r[i].slot>=p->d.capacity||!p->used[r[i].slot]||p->gen[r[i].slot]!=r[i].generation||p->d.epoch!=r[i].pipeEpoch||r[i].boundarySide>1||!r[i].payload.c||!r[i].payload.lastc){S.stat.generationFailures++;return MSX_RESIDENT_ERR_GENERATION;}}for(i=0;i<n;i++){p=&S.p[r[i].linkIndex];{double*q=crow(p,r[i].slot);q[0]=r[i].payload.volume;q[1]=r[i].payload.hstep;q[2]=r[i].payload.hresponse;q[3]=r[i].payload.uresponse;q[4]=r[i].payload.dresponse;for(m=0;m<S.stride;m++){q[5+m]=r[i].payload.c[m];lrow(p,r[i].slot)[5+m]=r[i].payload.lastc[m];}}}return MSX_RESIDENT_OK;}const MSXResidentStats*MSXresident_stats(void){return&S.stat;}

static void txreset(Tx *t)
{
    uint32_t i;
    if (!t) return;
    /* Once storage preparation published its token, storage owns abort
       cleanup and returns detached boundaries to the fixed pool.  Before
       that point (for example, a later materialize row failed), the Core
       transaction still owns the partially acquired boundaries. */
    if (t->storage.opaque)
        MSXsegStorage_hybridAbortPreparedResidentMaterialization(&t->storage);
    else if (t->boundary) for (i = 0; i < t->count; i++)
        if (t->boundary[i]) MSXqual_removeSeg(t->boundary[i]);
    if (t->boundary) memset(t->boundary, 0, (size_t)t->capacity * sizeof(*t->boundary));
    t->count = t->patchReserve = 0;
    t->active = 0;
    memset(&t->plan, 0, sizeof(t->plan));
}

static MSXResidentStatus txvalidate(const MSXResidentHandoffPlan *plan,
    const MSXResidentHandoffResult *r, uint32_t n)
{
    uint32_t i, j;
    Pipe *p;
    if (!plan || !r || !n || n != plan->itemCount || !plan->item ||
        plan->boundarySide > 1) return MSX_RESIDENT_ERR_ARGUMENT;
    for (i = 0; i < n; i++)
    {
        const MSXResidentHandoffItem *q = &plan->item[i];
        if (q->linkIndex != plan->linkIndex || q->boundarySide != plan->boundarySide ||
            q->pipeEpoch != plan->pipeEpoch || r[i].linkIndex != q->linkIndex ||
            r[i].slot != q->slot || r[i].generation != q->generation ||
            r[i].boundarySide != q->boundarySide || r[i].pipeEpoch != q->pipeEpoch ||
            !r[i].payload.c || !r[i].payload.lastc) return MSX_RESIDENT_ERR_ARGUMENT;
        if (pipeok(q->linkIndex, &p) != MSX_RESIDENT_OK || p->d.epoch != q->pipeEpoch ||
            q->slot >= p->d.capacity || !p->used[q->slot] || p->gen[q->slot] != q->generation)
            return MSX_RESIDENT_ERR_GENERATION;
        for (j = 0; j < i; j++) if (plan->item[j].slot == q->slot)
            return MSX_RESIDENT_ERR_ARGUMENT;
    }
    return MSX_RESIDENT_OK;
}

MSXResidentStatus MSXresident_prepareHandoffTransaction(const MSXResidentHandoffPlan *plan,
    const MSXResidentHandoffResult *r, uint32_t n, MSXResidentHandoffTransaction *out)
{
    Tx *t; MSXResidentStatus z; uint32_t i, m, reserve = 0;
    if (!out || out->opaque) return MSX_RESIDENT_ERR_ARGUMENT;
    z = txvalidate(plan, r, n); if (z) return z;
    t = &S.tx[plan->linkIndex];
    if (t->active || n > t->capacity) return MSX_RESIDENT_ERR_CAPACITY;
    /* All mutable transaction rows were allocated in MSXresident_open(). */
    t->active = 1;
    t->plan = *plan; t->plan.item = t->item;
    memcpy(t->item, plan->item, (size_t)n * sizeof(*t->item));
    memcpy(t->result, r, (size_t)n * sizeof(*r));
    t->count = n;
    /* Deep-copy D2H rows now: their runtime buffers may be reused before commit. */
    for (i = 0; i < n; ++i) {
        t->result[i].payload.c = t->c + (size_t)i * S.stride;
        t->result[i].payload.lastc = t->lastc + (size_t)i * S.stride;
        for (m = 0; m < S.stride; ++m) {
            t->c[(size_t)i * S.stride + m] = r[i].payload.c[m];
            t->lastc[(size_t)i * S.stride + m] = r[i].payload.lastc[m];
        }
    }
    /* Every invalidation has a bounded preallocated patch slot.  Reserve the
       exact number that is not already published; no State field changes here. */
    for (i = 0; i < n; ++i) if (S.p[plan->linkIndex].patch[t->result[i].slot] == NONE) ++reserve;
    if (reserve > S.slots - S.nslot) { txreset(t); return MSX_RESIDENT_ERR_CAPACITY; }
    t->patchReserve = reserve;
    for (i = 0; i < n; i++)
    {
        if (MSXsegStorage_hybridMaterializeResident(r[i].linkIndex, r[i].slot,
            r[i].generation, &t->boundary[i])) { txreset(t); return MSX_RESIDENT_ERR_MEMORY; }
        /* Use returned D2H payload, not a second resident row read. */
        t->boundary[i]->v = r[i].payload.volume;
        t->boundary[i]->hstep = r[i].payload.hstep;
        t->boundary[i]->hresponse = r[i].payload.hresponse;
        t->boundary[i]->uresponse = r[i].payload.uresponse;
        t->boundary[i]->dresponse = r[i].payload.dresponse;
        for (m = 0; m < S.stride; m++)
        {
            t->boundary[i]->c[m] = r[i].payload.c[m];
            t->boundary[i]->lastc[m] = r[i].payload.lastc[m];
        }
    }
    if (MSXsegStorage_hybridPrepareResidentMaterialization(&t->plan,
        t->result, t->boundary, t->count, &t->storage)) { txreset(t); return MSX_RESIDENT_ERR_GENERATION; }
    out->opaque = t; return MSX_RESIDENT_OK;
}

MSXResidentStatus MSXresident_validateHandoffTransactions(const MSXResidentHandoffTransaction *x,
    uint32_t n)
{
    uint32_t j;
    if (!x && n) return MSX_RESIDENT_ERR_ARGUMENT;
    for (j = 0; j < n; ++j) if (x[j].opaque) {
        Tx *t = (Tx *)x[j].opaque;
        if (!t->active || txvalidate(&t->plan, t->result, t->count) != MSX_RESIDENT_OK ||
            t->patchReserve > S.slots - S.nslot ||
            !MSXsegStorage_hybridValidatePreparedResidentMaterialization(&t->storage))
            return MSX_RESIDENT_ERR_GENERATION;
    }
    return MSX_RESIDENT_OK;
}

void MSXresident_commitHandoffTransaction(MSXResidentHandoffTransaction *x)
{
    Tx *t; Pipe *p; uint32_t i;
    if (!x || !(t = (Tx *)x->opaque)) return;
    /* Required precondition: MSXresident_validateHandoffTransactions() has
       accepted the whole batch.  No validation, allocation, lookup, or error path. */
    MSXsegStorage_hybridCommitPreparedResidentMaterialization(&t->storage);
    p = &S.p[t->plan.linkIndex];
    for (i = 0; i < t->count; i++)
    {
        uint32_t s = t->result[i].slot;
        p->used[s] = 0; p->id[s] = 0;
        memset(crow(p, s), 0, 2 * (size_t)S.rowWidth * sizeof(double));
        markslot(t->plan.linkIndex, s, 0, MSX_RESIDENT_PATCH_INVALIDATE);
    }
    p->d.count -= t->count;
    if (!p->d.count) p->d.head = p->d.tail = NONE;
    else if (!t->plan.boundarySide) p->d.head = nth(p, p->d.head, p->d.orient, t->count);
    else p->d.tail = nth(p, p->d.tail, -p->d.orient, t->count);
    p->d.epoch++; markdesc(t->plan.linkIndex);
    for (i = 0; i < t->count; i++) t->boundary[i] = NULL;
    txreset(t); x->opaque = NULL;
}

void MSXresident_abortHandoffTransaction(MSXResidentHandoffTransaction *x)
{ if (!x) return; txreset((Tx *)x->opaque); x->opaque = NULL; }
