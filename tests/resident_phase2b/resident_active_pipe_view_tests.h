typedef struct {
    ActivePipeChecks checks;
    MSXResidentActiveRow observed[4];
} ActiveViewChecks;
static MSXResidentStatus active_view_before(const MSXResidentActiveRow *row,void *context)
{
    ActiveViewChecks *c=(ActiveViewChecks*)context;
    c->observed[c->checks.beforeCount]=*row;
    return active_pipe_before(row,&c->checks);
}
static MSXResidentStatus active_view_after(const MSXResidentActiveRow *row,void *context)
{return active_pipe_after(row,&((ActiveViewChecks*)context)->checks);}
static int active_view_row_equal(const MSXResidentActiveRow *a,const MSXResidentActiveRow *b)
{
    return a->linkIndex==b->linkIndex&&a->slot==b->slot&&a->globalRow==b->globalRow&&
        a->generation==b->generation&&a->descriptorHead==b->descriptorHead&&
        a->descriptorCount==b->descriptorCount&&a->descriptorOrient==b->descriptorOrient&&
        a->descriptorEpoch==b->descriptorEpoch&&a->parcelId==b->parcelId&&
        memcmp(&a->volume,&b->volume,sizeof(double))==0;
}
static void t_active_pipe_view_writer(void)
{
    int passed=pass,failed=fail;
    MSXResidentGpu *g=initial4();MSXResidentGpuActiveWriter writer;
    uint64_t readWindow=123;
    uint32_t gen[2];uint64_t id[2]={101,102};double scalar[6];
    MSXResidentActivePipeView pipe;
    for(int orient=-2;orient<=2;++orient)for(uint32_t head=0;head<2;++head)
     for(uint32_t count=0;count<=2;++count)for(int fault=0;fault<16;++fault)
    {
        MSXResidentActiveRow rows[2];ActiveViewChecks oldChecks,newChecks;
        MSXResidentStatus oldStatus,newStatus;
        uint32_t oldCount,oldTouched;int oldValid,oldOwned;
        memset(&pipe,0,sizeof(pipe));memset(&oldChecks,0,sizeof(oldChecks));
        gen[0]=gen[1]=1;scalar[0]=-0.0;scalar[3]=3.0;
        pipe.linkIndex=1;pipe.capacity=2;pipe.head=head;pipe.count=count;
        pipe.orient=orient;pipe.epoch=5;pipe.generation=gen;
        pipe.parcelId=id;pipe.scalar=scalar;pipe.scalarStride=3;
        pipe.readWindow=&readWindow;pipe.windowToken=readWindow;
        if(fault==1)oldChecks.checks.failBefore=1;
        if(fault==2)oldChecks.checks.failBefore=2;
        if(fault==3)oldChecks.checks.failAfter=1;
        if(fault==4)oldChecks.checks.failAfter=2;
        if(fault==5)gen[head]=9;
        if(fault==6)gen[1-head]=9;
        if(fault==7)scalar[3*head]=NAN;
        if(fault==8)scalar[3*(1-head)]=NAN;
        if(fault==9)pipe.epoch=4;
        if(fault==10)pipe.base=4;
        if(fault==11){oldChecks.checks.failBefore=1;gen[head]=9;scalar[3*head]=NAN;gen[1-head]=9;}
        if(fault==12){oldChecks.checks.failAfter=1;scalar[3*(1-head)]=NAN;}
        if(fault==13){oldChecks.checks.failBefore=2;gen[1-head]=9;scalar[3*head]=NAN;}
        if(fault==14){pipe.base=4;scalar[3*head]=NAN;}
        if(fault==15)oldChecks.checks.failBefore=2;
        newChecks=oldChecks;
        uint32_t slotIndex=head;
        /* Independent scalar traversal, matching old enumerate row layout. */
        for(uint32_t i=0;i<count;++i){
            rows[i]=(MSXResidentActiveRow){1,slotIndex,pipe.base+slotIndex,
                gen[slotIndex],head,count,orient,pipe.epoch,id[slotIndex],scalar[3*slotIndex]};
            slotIndex=(uint32_t)(((int64_t)slotIndex+orient+2)%2);
        }
        uint32_t expected=fault==15&&count==2?1:count;
        OK(MSXresidentGpu_beginActive(g,expected,700,&writer)==0);
        oldStatus=MSXresidentGpu_appendActivePipe(g,&writer,rows,count,
            active_view_before,active_view_after,&oldChecks);
        oldCount=writer.count;oldTouched=writer.touchedCount;
        oldValid=writer.valid;oldOwned=writer.owner!=NULL;
        if(writer.owner)OK(MSXresidentGpu_abortActiveBuild(g,&writer)==0);
        OK(MSXresidentGpu_beginActive(g,expected,700,&writer)==0);
        newStatus=MSXresidentGpu_appendActivePipeView(g,&writer,&pipe,
            active_view_before,active_view_after,&newChecks);
        OK(oldStatus==newStatus&&oldCount==writer.count&&oldTouched==writer.touchedCount&&
            oldValid==writer.valid&&oldOwned==(writer.owner!=NULL));
        OK(oldChecks.checks.beforeCount==newChecks.checks.beforeCount&&
            oldChecks.checks.afterCount==newChecks.checks.afterCount);
        for(uint32_t i=0;i<oldChecks.checks.beforeCount;i++)
            OK(active_view_row_equal(&oldChecks.observed[i],&newChecks.observed[i]));
        if(writer.owner)OK(MSXresidentGpu_abortActiveBuild(g,&writer)==0);
    }
    /* META/current scalar source is borrowed, never copied by epoch. */
    memset(&pipe,0,sizeof(pipe));pipe.linkIndex=1;pipe.capacity=2;pipe.count=1;
    pipe.orient=1;pipe.epoch=5;pipe.generation=gen;gen[0]=1;
    pipe.parcelId=id;pipe.scalar=scalar;pipe.scalarStride=3;
    pipe.readWindow=&readWindow;pipe.windowToken=readWindow;
    scalar[0]=19.25;ActiveViewChecks current;memset(&current,0,sizeof(current));
    OK(MSXresidentGpu_beginActive(g,1,701,&writer)==0);
    OK(MSXresidentGpu_appendActivePipeView(g,&writer,&pipe,active_view_before,NULL,&current)==0);
    OK(current.observed[0].volume==19.25&&current.observed[0].descriptorEpoch==5);
    OK(MSXresidentGpu_abortActiveBuild(g,&writer)==0);
    /* Ending or reusing the owner window rejects before reading expired
       source arrays. Deliberately invalid pointers must never be touched. */
    pipe.generation=(const uint32_t *)(uintptr_t)1;
    pipe.parcelId=(const uint64_t *)(uintptr_t)1;
    pipe.scalar=(const double *)(uintptr_t)1;
    readWindow=0;
    OK(MSXresidentGpu_beginActive(g,1,702,&writer)==0);
    OK(MSXresidentGpu_appendActivePipeView(g,&writer,&pipe,NULL,NULL,NULL)==MSX_RESIDENT_ERR_GENERATION);
    OK(writer.count==0&&writer.valid&&MSXresidentGpu_abortActiveBuild(g,&writer)==0);
    readWindow=124;
    OK(MSXresidentGpu_beginActive(g,1,703,&writer)==0);
    OK(MSXresidentGpu_appendActivePipeView(g,&writer,&pipe,NULL,NULL,NULL)==MSX_RESIDENT_ERR_GENERATION);
    OK(writer.count==0&&writer.valid&&MSXresidentGpu_abortActiveBuild(g,&writer)==0);
    MSXresidentGpu_close(g);
    printf("c06_view_writer_assertions_passed=%d\nc06_view_writer_assertions_failed=%d\n",pass-passed,fail-failed);
}
