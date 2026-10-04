#include <stdio.h>
#include <string.h>
#include "msxresident_options.h"
static int failures;
#define CHECK(x) do{if(!(x)){fprintf(stderr,"line %d: %s\n",__LINE__,#x);++failures;}}while(0)
int main(void)
{
    uint64_t bytes;MSXResidentMemoryOptions o={0};MSXResidentResolvedOptions r;
    const char *names[]={"MSX_RESIDENT_HOST_BUDGET_BYTES","MSX_RESIDENT_HOST_MEMORY_MB",
        "MSX_RESIDENT_CPU_POOL_BUDGET_BYTES","MSX_RESIDENT_CPU_POOL_BUDGET_MB",
        "MSX_RESIDENT_UPLOAD_BUDGET_BYTES","MSX_RESIDENT_UPLOAD_BUDGET_MB","MSX_RESIDENT_TRANSFER_BATCH_ROWS"};
    unsigned k;for(k=0;k<sizeof(names)/sizeof(*names);++k)_putenv_s(names[k],"");
    CHECK(MSXmemory_parseInteger("18446744073709551615",&bytes)&&bytes==UINT64_MAX);
    CHECK(!MSXmemory_parseInteger("18446744073709551616",&bytes));
    CHECK(!MSXmemory_parseInteger("-1",&bytes));CHECK(!MSXmemory_parseInteger("+1",&bytes));
    CHECK(!MSXmemory_parseInteger(" 1",&bytes));CHECK(!MSXmemory_parseInteger("1x",&bytes));
    CHECK(!MSXmemory_parseMB("0",&bytes));CHECK(!MSXmemory_parseMB("1.5",&bytes));
    CHECK(!MSXmemory_parseMB("17592186044416",&bytes));
    CHECK(MSXmemory_parseMB("17592186044415",&bytes)&&bytes==UINT64_C(18446744073708503040));
    CHECK(MSXmemory_resolve(&o,123,&r)&&r.values.hostBytes==123&&r.values.batchRows==65536);
    o.hostBytes=1048576;o.explicitMask=MSX_MEMORY_OPTION_HOST;
    _putenv_s(names[0],"1048576");CHECK(MSXmemory_resolve(&o,123,&r));
    _putenv_s(names[1],"1");CHECK(MSXmemory_resolve(&o,123,&r));
    _putenv_s(names[1],"2");CHECK(!MSXmemory_resolve(&o,123,&r));
    _putenv_s(names[1],"");_putenv_s(names[0],"1048575");CHECK(!MSXmemory_resolve(&o,123,&r));
    _putenv_s(names[0],"");_putenv_s(names[6],"0");CHECK(!MSXmemory_resolve(&o,123,&r));
    _putenv_s(names[6],"4294967296");CHECK(!MSXmemory_resolve(&o,123,&r));
    _putenv_s(names[6],"32768");CHECK(MSXmemory_resolve(&o,123,&r)&&r.values.batchRows==32768);
    o.batchRows=65536;o.explicitMask|=MSX_MEMORY_OPTION_BATCH;CHECK(!MSXmemory_resolve(&o,123,&r));
    printf("resident_options_tests failures=%d\n",failures);return failures?1:0;
}
