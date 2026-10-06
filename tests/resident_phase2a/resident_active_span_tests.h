/* C05 independent scalar oracle, including partial output on rejection. */
static MSXResidentStatus active_span_compare(uint32_t capacity)
{
    MSXResidentActiveRow reference[32],candidate[32];uint32_t nr=123,nc=456;
    MSXResidentStatus sr,sc,sf;MSXResidentActivePipeView pipes[3];uint32_t nf=789;
    memset(reference,0xa5,sizeof(reference));memset(candidate,0xa5,sizeof(candidate));
    sr=MSXresident_enumerateActive(reference,capacity,&nr);
    sc=MSXresident_enumerateActiveSpans(candidate,capacity,&nc);
    sf=MSXresident_preflightActive(pipes,3,capacity,&nf);
    OK(sr==sc&&nr==nc);
    OK(sr==sf&&nr==nf);
    OK(memcmp(reference,candidate,sizeof(reference))==0);
    if(sf==MSX_RESIDENT_OK)
    {
        uint32_t k,n=0;
        memset(candidate,0xa5,sizeof(candidate));
        /* Independent scalar reconstruction: no production span generator. */
        for(k=1;k<=2;k++){
            uint32_t i,slot=pipes[k].head;
            for(i=0;i<pipes[k].count;i++){
                MSXResidentActiveRow *r=&candidate[n++];
                r->linkIndex=k;r->slot=slot;r->globalRow=pipes[k].base+slot;
                r->generation=pipes[k].generation[slot];r->descriptorHead=pipes[k].head;
                r->descriptorCount=pipes[k].count;r->descriptorOrient=pipes[k].orient;
                r->descriptorEpoch=pipes[k].epoch;r->parcelId=pipes[k].parcelId[slot];
                r->volume=pipes[k].scalar[(size_t)slot*pipes[k].scalarStride];
                slot=(uint32_t)(((int64_t)slot+pipes[k].orient+pipes[k].capacity)%(int64_t)pipes[k].capacity);
            }
            OK(pipes[k].activeOffset+pipes[k].count==n);
        }
        OK(n==nr&&memcmp(reference,candidate,sizeof(reference))==0);
    }
    return sr;
}
static void t_active_spans(void)
{
    char table[768];double c[2]={0,1},last[2]={0,2};
    uint64_t ids[8];MSXResidentPayload values[8];uint32_t i,n;
    MSXResidentActiveRow rows[32];
    setup(1,2);
    snprintf(table,sizeof(table),"link_index,link_id,capacity,max_core_count,combined_burst_p99,guard,case_hash\n1,L1,8,8,1,2,%s\n2,L2,8,8,1,2,%s\n",UP,UP);
    csv("resident_phase2a.csv",table);OK(MSXresident_open("resident_phase2a.csv")==0);
    MSXresident_setMode(MSX_RESIDENT_SHADOW,0);
    for(i=0;i<8;i++){ids[i]=100+i;values[i]=payload(c,last,(double)i+1,ids[i]);}
    OK(MSXresident_observePipe(1,ids,values,8,1)==0);
    OK(MSXresident_observePipe(2,ids,values,8,1)==0);
    OK(active_span_compare(32)==MSX_RESIDENT_OK);
    OK(MSXresident_stageReverse(1)==0&&active_span_compare(32)==MSX_RESIDENT_OK);
    OK(MSXresident_enumerateActiveSpans(rows,32,&n)==0&&n==16);
    /* META must read current volume without an epoch change. */
    {MSXResidentPayload changed=values[0];uint64_t epoch=rows[0].descriptorEpoch;
     changed.parcelId=rows[0].parcelId;changed.volume=17.25;
     OK(MSXresident_stageMeta(1,rows[0].slot,rows[0].generation,&changed)==0);
     OK(MSXresident_enumerateActiveSpans(rows,32,&n)==0&&rows[0].volume==17.25&&rows[0].descriptorEpoch==epoch);
     OK(active_span_compare(32)==MSX_RESIDENT_OK);}
    OK(active_span_compare(0)==MSX_RESIDENT_ERR_CAPACITY);
    OK(active_span_compare(7)==MSX_RESIDENT_ERR_CAPACITY);
    OK(MSXresident_enumerateActiveSpans(NULL,1,&n)==MSX_RESIDENT_ERR_ARGUMENT);
    OK(MSXresident_enumerateActiveSpans(rows,32,NULL)==MSX_RESIDENT_ERR_ARGUMENT);
#if defined(MSX_RESIDENT_TEST_API) || defined(MSX_RESIDENT_ACTIVE_TEST_API)
    {MSXResidentPipeDesc original,other,d;uint32_t head,count,limit;int orientation;
     OK(MSXresident_testInitialDescriptor(1,&original)==0);
     OK(MSXresident_testInitialDescriptor(2,&other)==0);
     /* Every legal ring interval including empty, full and both wraps. */
     for(orientation=-1;orientation<=1;orientation+=2)
      for(head=0;head<8;head++)for(count=0;count<=8;count++){
       d=original;d.head=head;d.count=count;d.orient=orientation;
       OK(MSXresident_testActiveDescriptor(1,&d)==0);
       for(limit=0;limit<=16;limit++)active_span_compare(limit);
      }
     /* Defined malformed orientation behavior stays scalar, including repeats. */
     for(orientation=-3;orientation<=3;orientation++){
      d=original;d.head=4;d.count=8;d.orient=orientation;
      OK(MSXresident_testActiveDescriptor(1,&d)==0);
      OK(active_span_compare(32)==MSX_RESIDENT_OK);
     }
     d=original;d.count=9;
     OK(MSXresident_testActiveDescriptor(1,&d)==0);
     OK(active_span_compare(0)==MSX_RESIDENT_ERR_CAPACITY);
     d=original;d.head=8;
     OK(MSXresident_testActiveDescriptor(1,&d)==0);
     OK(active_span_compare(32)==MSX_RESIDENT_ERR_CAPACITY);
     /* Row used check precedes output capacity, and first-pipe row rejection
        precedes a later pipe's descriptor capacity rejection. */
     d=original;d.head=0;d.orient=1;
     OK(MSXresident_testActiveDescriptor(1,&d)==0);
     OK(MSXresident_testActiveUsed(1,0,0)==0);
     OK(active_span_compare(0)==MSX_RESIDENT_ERR_GENERATION);
     other.count=9;OK(MSXresident_testActiveDescriptor(2,&other)==0);
     OK(active_span_compare(32)==MSX_RESIDENT_ERR_GENERATION);
     OK(MSXresident_testActiveUsed(1,0,1)==0);
     OK(active_span_compare(32)==MSX_RESIDENT_ERR_CAPACITY);
     other.count=8;OK(MSXresident_testActiveDescriptor(2,&other)==0);
     /* Later enumerate-used failure still wins over early nonfinite volume;
        volume rejection belongs only to writer after the full preflight. */
     {MSXResidentPayload changed=values[0];uint32_t generation;
      OK(MSXresident_getSlotGeneration(1,0,&generation)==0);
      changed.volume=NAN;
      OK(MSXresident_stageMeta(1,0,generation,&changed)==0);
      OK(MSXresident_testActiveUsed(2,0,0)==0);
      OK(active_span_compare(32)==MSX_RESIDENT_ERR_GENERATION);
      OK(MSXresident_testActiveUsed(2,0,1)==0);
      OK(active_span_compare(32)==MSX_RESIDENT_OK);
      changed.volume=-0.0;
      OK(MSXresident_stageMeta(1,0,generation,&changed)==0);
      OK(active_span_compare(32)==MSX_RESIDENT_OK);
     }
     /* Base overflow occurs only after the affected pipe's row traversal. */
     d=original;d.count=0;d.capacity=UINT32_MAX;
     OK(MSXresident_testActiveDescriptor(1,&d)==0);
     OK(active_span_compare(32)==MSX_RESIDENT_ERR_OVERFLOW);
     OK(active_span_compare(0)==MSX_RESIDENT_ERR_CAPACITY);
     OK(MSXresident_testActiveDescriptor(1,&original)==0);
    }
#endif
    OK(MSXresident_stageClear(1)==0&&MSXresident_stageClear(2)==0);
    OK(active_span_compare(32)==MSX_RESIDENT_OK);
    printf("c06_active_row_bytes=%zu\nc06_pipe_view_bytes=%zu\n",
        sizeof(MSXResidentActiveRow),sizeof(MSXResidentActivePipeView));
}
