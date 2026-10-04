static void t_upload_revisions(void)
{
    MSXResidentGpu *g=initial4();MSXResidentSlotPatch uploads[2],latest;
    MSXResidentDescriptorPatch d;MSXResidentPatchBatch batch;
    MSXResidentGpuReduction before,after;double a[S],b[S],c[2][S]={{41,42,43},{51,52,53}},l[2][S]={{61,62,63},{71,72,73}};
    slot(&uploads[0],1,0,2,301,2,0);uploads[0].uploadRevision=1;
    uploads[0].payload.c=c[0];uploads[0].payload.lastc=l[0];uploads[1]=uploads[0];
    uploads[1].uploadRevision=2;uploads[1].payload.c=c[1];uploads[1].payload.lastc=l[1];
    OK(snap(g,&before,a));OK(MSXresidentGpu_stageImports(g,uploads,2)==0);
    OK(snap(g,&after,b)&&same(&before,&after,a,b)); /* No live row or descriptor publication. */
    latest=uploads[1];latest.payload.c=latest.payload.lastc=NULL;
    desc(&d,1,2,2,6,1);batch=(MSXResidentPatchBatch){&d,1,&latest,1};
    memset(c,0,sizeof(c));memset(l,0,sizeof(l)); /* Source can now be reused. */
    OK(MSXresidentGpu_applyPatches(g,&batch)==0);
    MSXResidentHandoffItem item={1,0,2,0,6,0};MSXResidentHandoffResult result;
    double rc[S],rl[S];MSXResidentGpuFetchOutput output={&result,rc,rl,S};
    OK(MSXresidentGpu_fetchHandoffBatch(g,&item,1,&output)==0&&rc[0]==51&&rl[2]==73);
    latest.uploadRevision=1;invalid_unchanged(g,&batch,MSX_RESIDENT_ERR_GENERATION);
    OK(MSXresidentGpu_stageImports(g,uploads,1)==MSX_RESIDENT_ERR_GENERATION);
    /* A late old upload cannot replace a new generation/revision. */
    uploads[0].generation=3;uploads[0].payload.parcelId=401;uploads[0].uploadRevision=3;
    c[0][0]=81;c[0][1]=82;c[0][2]=83;l[0][0]=91;l[0][1]=92;l[0][2]=93;
    OK(MSXresidentGpu_stageImports(g,uploads,1)==0);
    latest=uploads[0];latest.payload.c=latest.payload.lastc=NULL;desc(&d,1,2,2,7,1);
    OK(MSXresidentGpu_applyPatches(g,&batch)==0);
    OK(MSXresidentGpu_stageImports(g,&uploads[1],1)==MSX_RESIDENT_ERR_GENERATION);
    item.generation=3;item.pipeEpoch=7;
    OK(MSXresidentGpu_fetchHandoffBatch(g,&item,1,&output)==0&&rc[0]==81&&rl[2]==93);
    /* IMPORT followed by META retains the staged vectors. */
    latest.kind=MSX_RESIDENT_PATCH_META;latest.uploadRevision=0;
    latest.payload.hstep=111;OK(MSXresidentGpu_applyPatches(g,&batch)==0);
    OK(MSXresidentGpu_fetchHandoffBatch(g,&item,1,&output)==0&&rc[0]==81&&result.payload.hstep==111);
    MSXresidentGpu_close(g);
}
