extern int MSXresidentGpu_testPinnedPayloadOwners(MSXResidentGpu *,uint64_t *,uint32_t *);
static void t_pinned_payload_arenas(void)
{
    MSXResidentGpu *g=initial4();uint64_t address[4]={0};uint32_t tickets=0;
    MSXResidentMemoryEstimate m;MSXResidentBudget before,after;
    OK(g!=NULL);if(!g)return;
    OK(MSXresidentGpu_testPinnedPayloadOwners(g,address,&tickets));
    OK(address[0]&&address[0]==address[1]&&address[2]&&address[2]==address[3]);
    OK(address[0]!=address[2]&&tickets==2);
    /* Alternate H2D import and D2H gather through the same owners. Exact
       signs of zero and independent C/lastC vectors must survive reuse. */
    for(int cycle=0;cycle<2;++cycle){
        double c[S],l[S],outC[S],outL[S];MSXResidentSlotPatch patch;
        MSXResidentHandoffItem item={1,0,1,0,5,1};MSXResidentHandoffResult result;
        MSXResidentGpuFetchOutput output={&result,outC,outL,S};
        MSXResidentPatchBatch batch={0,0,&patch,1};
        for(uint32_t m=0;m<S;++m){c[m]=100.0+cycle*10+m;l[m]=-200.0-cycle*10-m;}
        c[0]=cycle?0.0:-0.0;l[0]=cycle?-0.0:0.0;
        slot(&patch,1,0,1,101,2,0);patch.kind=MSX_RESIDENT_PATCH_IMPORT;
        patch.payload.c=c;patch.payload.lastc=l;
        OK(MSXresidentGpu_applyPatches(g,&batch)==MSX_RESIDENT_OK);
        OK(MSXresidentGpu_fetchHandoffBatch(g,&item,1,&output)==MSX_RESIDENT_OK);
        OK(!memcmp(c,outC,sizeof(c))&&!memcmp(l,outL,sizeof(l)));
    }
    OK(MSXresidentGpu_getFixedMemory(g,&m)==MSX_RESIDENT_OK);
    MSXresidentBudget_snapshot(MSXresidentBudget_global(),&before);
    OK(before.allocated[1]==m.pinnedBytes);
    OK(MSXresidentGpu_writeAllocationManifest(g,"resident_shared_pinned_manifest.csv")==MSX_RESIDENT_OK);
    {FILE *f=fopen("resident_shared_pinned_manifest.csv","rb");char line[1024];int aliases=0;
     OK(f!=NULL);
     if(f){while(fgets(line,sizeof(line),f)){
        if(strstr(line,"&g->hGatherC,CUDA_CORE")||strstr(line,"&g->hGatherL,CUDA_CORE")){
            OK(strstr(line,",0,&g->hPatch")!=NULL);++aliases;}}
        fclose(f);OK(aliases==2);}}
    MSXresidentGpu_close(g);g=NULL;MSXresidentGpu_close(g);
    MSXresidentBudget_snapshot(MSXresidentBudget_global(),&after);
    OK(!after.allocated[0]&&!after.allocated[1]&&!after.allocated[2]);
    OK(!after.reserved[0]&&!after.reserved[1]&&!after.reserved[2]);
}
