/* Real CUDA boundary coverage; deliberately unique concentrations per row. */
#include <stdlib.h>
static void t_chunks(uint32_t batch,uint32_t n,uint32_t stride)
{
    uint32_t cap[3]={0,n/2,n-n/2},base[3]={0,0,n/2};
    if(n==1){cap[1]=1;cap[2]=0;}
    uint32_t links=n==1?1:2;
    MSXResidentGpuOpen o={links,n,stride,cap,base};
    MSXResidentMemoryConfig cfg={1,MSX_RESIDENT_HYD_PIPE_MAJOR,batch};
    MSXResidentGpu *g=NULL;MSXResidentMemoryEstimate expected,actual,legacy;
    MSXResidentSlotPatch *x=calloc(n,sizeof(*x));
    MSXResidentHandoffItem *q=calloc(n,sizeof(*q));
    MSXResidentHandoffResult *z=calloc(n,sizeof(*z));
    MSXResidentActiveItem *active=calloc(n,sizeof(*active));
    MSXResidentGpuActiveSyncRow *sync=calloc(n,sizeof(*sync));
    double *c=calloc((size_t)n*stride,sizeof(double)),*l=calloc((size_t)n*stride,sizeof(double));
    double *oc=calloc((size_t)n*stride,sizeof(double)),*ol=calloc((size_t)n*stride,sizeof(double));
    double hyd[3*MSX_RESIDENT_HYD_STRIDE]={0};MSXResidentDescriptorPatch d[2];
    MSXResidentPatchBatch p;MSXResidentGpuDeviceView view;MSXResidentGpuReactResult r;
    MSXResidentGpuFetchOutput out={z,oc,ol,stride};
    MSXResidentGpuActiveSyncOutput so={sync,oc,ol,stride};
    MSXResidentHydView hv={hyd,links,MSX_RESIDENT_HYD_STRIDE,MSX_RESIDENT_HYD_PIPE_MAJOR};
    for(uint32_t k=1;k<=links;++k)desc(&d[k-1],k,cap[k],cap[k],1,1);
    for(uint32_t i=0;i<n;++i){uint32_t k=links==1||i<base[2]?1:2;
        for(uint32_t m=0;m<stride;++m){c[(size_t)i*stride+m]=i*100.0+m;l[(size_t)i*stride+m]=-i*100.0-m;}
        x[i].linkIndex=k;x[i].slot=i-base[k];x[i].generation=1;x[i].used=1;
        x[i].kind=MSX_RESIDENT_PATCH_IMPORT;
        x[i].payload=(MSXResidentPayload){0};x[i].payload.parcelId=1000+i;
        x[i].payload.generation=1;x[i].payload.volume=1;x[i].payload.hstep=2+i;
        x[i].payload.hresponse=3+i;x[i].payload.uresponse=4+i;x[i].payload.dresponse=5+i;
        x[i].payload.c=c+(size_t)i*stride;x[i].payload.lastc=l+(size_t)i*stride;
        q[i]=(MSXResidentHandoffItem){k,i-base[k],1,0,1,1};
        active[i]=(MSXResidentActiveItem){k,i,1,1,1,NULL};
    }
    OK(MSXresidentGpu_estimateConfigured(links,n,stride,&cfg,&expected)==0);
    OK(MSXresidentGpu_openConfigured(&o,&cfg,&g)==0);
    if(!g)goto done;
    OK(MSXresidentGpu_getFixedMemory(g,&actual)==0&&!memcmp(&actual,&expected,sizeof(actual)));
    OK(MSXresidentGpu_estimateMemory(links,n,stride,&legacy)==0&&legacy.pinnedBytes>actual.pinnedBytes);
    p=(MSXResidentPatchBatch){d,links,x,n};
    OK(MSXresidentGpu_initialUpload(g,&p)==0);
    uint32_t counts[6]={0,1,batch-1,batch,batch+1,2*batch+1};
    for(uint32_t a=0;a<6;++a){uint32_t count=counts[a];if(count>n)continue;
        MSXResidentGpuTransferStats before,after;
        OK(MSXresidentGpu_getTransferStats(g,&before)==0);
        OK(MSXresidentGpu_fetchHandoffBatch(g,q,count,&out)==0);
        OK(!memcmp(oc,c,(size_t)count*stride*sizeof(double))&&!memcmp(ol,l,(size_t)count*stride*sizeof(double)));
        OK(MSXresidentGpu_getTransferStats(g,&after)==0&&after.h2dCalls-before.h2dCalls==(count+batch-1)/batch&&
            after.d2hCalls-before.d2hCalls==3*((count+batch-1)/batch));
        for(uint32_t i=0;i<count;++i)if(z[i].payload.parcelId!=1000+i||z[i].payload.hstep!=2+i||
            z[i].payload.hresponse!=3+i||z[i].payload.uresponse!=4+i||z[i].payload.dresponse!=5+i){OK(0);break;}
    }
    if(n>batch){MSXResidentHandoffItem saved=q[batch];q[batch]=q[0];
        OK(MSXresidentGpu_fetchHandoffBatch(g,q,n,&out)==MSX_RESIDENT_ERR_ARGUMENT);q[batch]=saved;
        MSXResidentSlotPatch savedSlot=x[batch];x[batch]=x[0];
        OK(MSXresidentGpu_applyPatches(g,&p)==MSX_RESIDENT_ERR_ARGUMENT);x[batch]=savedSlot;}
    /* Mixed META/IMPORT forces payload and logical offsets to diverge. */
    for(uint32_t i=0;i<n;++i){if(i%2)x[i].kind=MSX_RESIDENT_PATCH_META;
        else for(uint32_t m=0;m<stride;++m)c[(size_t)i*stride+m]+=7;}
    OK(MSXresidentGpu_applyPatches(g,&p)==0);
    OK(MSXresidentGpu_fetchHandoffBatch(g,q,n,&out)==0&&!memcmp(oc,c,(size_t)n*stride*sizeof(double)));
    MSXResidentHandoffTarget *targets=calloc(n,sizeof(*targets));
    for(uint32_t i=0;i<n;++i){targets[i].c=oc+(size_t)(n-1-i)*stride;targets[i].lastc=ol+(size_t)(n-1-i)*stride;}
    OK(MSXresidentGpu_fetchHandoffTargets(g,q,n,z,targets)==0);
    for(uint32_t i=0;i<n;++i)if(memcmp(targets[i].c,c+(size_t)i*stride,stride*sizeof(double))||
        memcmp(targets[i].lastc,l+(size_t)i*stride,stride*sizeof(double))||z[i].payload.c!=targets[i].c){OK(0);break;}
    double *savedTarget=targets[n-1].c;targets[n-1].c=NULL;
    OK(MSXresidentGpu_fetchHandoffTargets(g,q,n,z,targets)==MSX_RESIDENT_ERR_ARGUMENT);
    targets[n-1].c=savedTarget;free(targets);
    MSXResidentActiveBatch ab={active,n};
    OK(MSXresidentGpu_prepareActiveHyd(g,&ab,&hv,&view,&r)==0);
    OK(MSXresidentGpu_finishActive(g,&r)==0);
    MSXresidentGpu_setDiagnosticMode(g,1);
    OK(MSXresidentGpu_prepareActiveHyd(g,&ab,&hv,&view,&r)==0&&MSXresidentGpu_finishActive(g,&r)==0);
    OK(MSXresidentGpu_syncActive(g,&so,n)==0&&!memcmp(oc,c,(size_t)n*stride*sizeof(double))&&
        !memcmp(ol,l,(size_t)n*stride*sizeof(double))&&sync[n-1].generation==1);
done:
    MSXresidentGpu_close(g);free(x);free(q);free(z);free(active);free(sync);free(c);free(l);free(oc);free(ol);
}
static void t_chunk_boundaries(void)
{
    t_chunks(4,9,3);t_chunks(4,1,17);t_chunks(65536,131073,3);
}
static void t_chunk_failures(void)
{
    uint32_t cap[2]={0,9},base[2]={0,0};MSXResidentGpuOpen o={1,9,S,cap,base};
    MSXResidentMemoryConfig cfg={1,MSX_RESIDENT_HYD_PIPE_MAJOR,4};
    MSXResidentDescriptorPatch d;MSXResidentSlotPatch x[9];MSXResidentHandoffItem q[9];
    MSXResidentHandoffResult r[9];double c[9*S],l[9*S],mass[S];
    MSXResidentGpuFetchOutput out={r,c,l,S};MSXResidentGpuReduction reduction;
    desc(&d,1,9,9,1,1);
    for(uint32_t i=0;i<9;++i){slot(&x[i],1,i,1,100+i,1,i);q[i]=(MSXResidentHandoffItem){1,i,1,0,1,1};}
    MSXResidentPatchBatch b={&d,1,x,9};
    for(uint32_t chunk=1;chunk<=3;++chunk){char value[8];MSXResidentGpu*g=NULL;snprintf(value,sizeof(value),"%u",chunk);
        _putenv_s("MSX_RESIDENT_TEST_PATCH_FAIL_CHUNK",value);
        OK(MSXresidentGpu_openConfigured(&o,&cfg,&g)==0);
        OK(MSXresidentGpu_initialUpload(g,&b)==MSX_RESIDENT_ERR_TRANSFER);
        OK(MSXresidentGpu_reduce(g,mass,S,&reduction)==MSX_RESIDENT_ERR_POISONED);MSXresidentGpu_close(g);
        _putenv_s("MSX_RESIDENT_TEST_PATCH_FAIL_CHUNK","");g=NULL;
        _putenv_s("MSX_RESIDENT_TEST_FETCH_FAIL_CHUNK",value);
        OK(MSXresidentGpu_openConfigured(&o,&cfg,&g)==0&&MSXresidentGpu_initialUpload(g,&b)==0);
        memset(r,0x77,sizeof(r));OK(MSXresidentGpu_fetchHandoffBatch(g,q,9,&out)==MSX_RESIDENT_ERR_TRANSFER);
        for(uint32_t i=0;i<9;++i)OK(r[i].payload.c==NULL&&r[i].payload.lastc==NULL);
        OK(MSXresidentGpu_reduce(g,mass,S,&reduction)==MSX_RESIDENT_ERR_POISONED);MSXresidentGpu_close(g);
        _putenv_s("MSX_RESIDENT_TEST_FETCH_FAIL_CHUNK","");
    }
}
