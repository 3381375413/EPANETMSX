#include <stdio.h>
#include <string.h>
#include "msxresident_transaction.h"
static int failures;
#define CHECK(x) do{if(!(x)){fprintf(stderr,"line %d: %s\n",__LINE__,#x);++failures;}}while(0)
int main(void)
{
    MSXResidentTransaction t={0};unsigned k;
    CHECK(MSXtransaction_readable(&t));CHECK(!MSXtransaction_commit(&t));
    for(k=0;k<100;++k){
        CHECK(MSXtransaction_begin(&t));CHECK(t.working==t.published+1);
        CHECK(!MSXtransaction_readable(&t));CHECK(!MSXtransaction_begin(&t));
        CHECK(!MSXtransaction_advance(&t,MSX_TX_APPLYING));
        CHECK(MSXtransaction_advance(&t,MSX_TX_PREPARED));
        CHECK(MSXtransaction_advance(&t,MSX_TX_STAGING));
        CHECK(!MSXtransaction_commit(&t));CHECK(!MSXtransaction_complete(&t));
        CHECK(MSXtransaction_advance(&t,MSX_TX_APPLYING));
        CHECK(!MSXtransaction_commit(&t));CHECK(MSXtransaction_complete(&t));
        CHECK(!MSXtransaction_readable(&t));CHECK(MSXtransaction_commit(&t));
        CHECK(MSXtransaction_readable(&t)&&t.published==k+1&&t.completed==t.published);
    }
    CHECK(MSXtransaction_begin(&t));CHECK(MSXtransaction_abort(&t));
    CHECK(MSXtransaction_abort(&t));CHECK(t.published==100&&t.completed==100);
    CHECK(MSXtransaction_begin(&t));t.cpuChanged=1;
    CHECK(!MSXtransaction_abort(&t)&&t.state==MSX_TX_POISONED);CHECK(!MSXtransaction_begin(&t));
    memset(&t,0,sizeof(t));CHECK(MSXtransaction_begin(&t));t.gpuAccepted=1;
    CHECK(!MSXtransaction_abort(&t)&&t.state==MSX_TX_POISONED);
    memset(&t,0,sizeof(t));CHECK(MSXtransaction_begin(&t));MSXtransaction_fail(&t,0);
    CHECK(t.state==MSX_TX_FAILED&&!MSXtransaction_readable(&t)&&!MSXtransaction_begin(&t));
    memset(&t,0,sizeof(t));t.sequence=UINT64_MAX;CHECK(!MSXtransaction_begin(&t));
    t.sequence=0;t.published=t.completed=UINT64_MAX;CHECK(!MSXtransaction_begin(&t));
    printf("resident_transaction_tests failures=%d\n",failures);return failures?1:0;
}
