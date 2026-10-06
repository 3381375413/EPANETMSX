typedef struct { uint32_t beforeCount,afterCount,failBefore,failAfter; } ActivePipeChecks;
static MSXResidentStatus active_pipe_before(const MSXResidentActiveRow *row,void *ctx)
{
    ActivePipeChecks *c=(ActivePipeChecks*)ctx;(void)row;
    return ++c->beforeCount==c->failBefore?MSX_RESIDENT_ERR_ARGUMENT:MSX_RESIDENT_OK;
}
static MSXResidentStatus active_pipe_after(const MSXResidentActiveRow *row,void *ctx)
{
    ActivePipeChecks *c=(ActivePipeChecks*)ctx;(void)row;
    return ++c->afterCount==c->failAfter?MSX_RESIDENT_ERR_CAPACITY:MSX_RESIDENT_OK;
}
static void t_active_pipe_writer(void)
{
    int passed=pass,failed=fail;
    MSXResidentGpu *g=initial4();MSXResidentGpuActiveWriter w;
    MSXResidentActiveRow rows[2]={{1,0,0,1,0,2,1,5,101,2.0},{1,1,1,1,1,2,1,5,102,3.0}};
    ActivePipeChecks c;
    /* The empty loop never checks a writer or calls either callback. */
    memset(&c,0,sizeof(c));
    OK(MSXresidentGpu_appendActivePipe(NULL,NULL,NULL,0,active_pipe_before,active_pipe_after,&c)==0);
    OK(c.beforeCount==0&&c.afterCount==0);
    for(int mode=0;mode<6;++mode) {
        MSXResidentActiveRow r[2];memcpy(r,rows,sizeof(r));memset(&c,0,sizeof(c));
        OK(MSXresidentGpu_beginActive(g,2,100+mode,&w)==0);
        if(mode==1){c.failBefore=1;r[0].generation=9;r[1].globalRow=4;}
        if(mode==2){c.failBefore=2;r[1].generation=9;}
        if(mode==3){r[1].generation=9;r[1].descriptorEpoch=1;}
        if(mode==4){c.failAfter=1;r[1].generation=9;}
        if(mode==5){c.failAfter=2;}
        MSXResidentStatus z=MSXresidentGpu_appendActivePipe(g,&w,r,2,active_pipe_before,active_pipe_after,&c);
        if(mode==0) OK(z==0&&w.count==2&&c.beforeCount==2&&c.afterCount==2);
        if(mode==1) OK(z==MSX_RESIDENT_ERR_ARGUMENT&&w.count==0&&w.valid&&c.beforeCount==1&&c.afterCount==0);
        if(mode==2) OK(z==MSX_RESIDENT_ERR_ARGUMENT&&w.count==1&&w.valid&&c.beforeCount==2&&c.afterCount==1);
        if(mode==3) OK(z==MSX_RESIDENT_ERR_GENERATION&&!w.owner&&c.beforeCount==2&&c.afterCount==1);
        if(mode==4) OK(z==MSX_RESIDENT_ERR_CAPACITY&&w.count==1&&w.valid&&c.beforeCount==1&&c.afterCount==1);
        if(mode==5) OK(z==MSX_RESIDENT_ERR_CAPACITY&&w.count==2&&w.valid&&c.beforeCount==2&&c.afterCount==2);
        if(w.owner) OK(MSXresidentGpu_abortActiveBuild(g,&w)==0);
        OK(active_stamp(g,0)==0&&active_stamp(g,1)==0);
    }
    OK(MSXresidentGpu_beginActive(g,2,110,&w)==0);
    memset(&c,0,sizeof(c));
    OK(MSXresidentGpu_appendActivePipe(g,&w,NULL,1,active_pipe_before,active_pipe_after,&c)==MSX_RESIDENT_ERR_ARGUMENT);
    OK(w.valid&&w.count==0&&c.beforeCount==0&&c.afterCount==0);
    OK(MSXresidentGpu_appendActivePipe(g,&w,rows,2,NULL,NULL,NULL)==0&&w.count==2);
    OK(MSXresidentGpu_sealActive(g,&w,110)==0);
    OK(MSXresidentGpu_abortActiveBuild(g,&w)==0);
    MSXresidentGpu_close(g);
    printf("c05_writer_assertions_passed=%d\nc05_writer_assertions_failed=%d\n",pass-passed,fail-failed);
}
