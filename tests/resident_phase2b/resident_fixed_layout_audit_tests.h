MSXResidentStatus MSXresidentGpu_testAuditFlip(MSXResidentGpu *,uint32_t);
static void t_fixed_layout_audit(void)
{
    int passed=pass,failed=fail;
    MSXResidentGpu *g=initial4();MSXResidentBudget a,b;
    MSXResidentBudget *budget=MSXresidentBudget_global();uint64_t charged;
    MSXresidentBudget_snapshot(budget,&a);
    OK(MSXresidentGpu_auditFixedLayout(NULL)==MSX_RESIDENT_ERR_ARGUMENT);
    OK(MSXresidentGpu_auditFixedLayout(g)==0);
    MSXresidentBudget_snapshot(budget,&b);charged=b.allocated[0]-a.allocated[0];
    OK(charged>37*sizeof(uintptr_t));
    for(uint32_t i=0;i<39;++i){
        OK(MSXresidentGpu_testAuditFlip(g,i)==0);
        OK(MSXresidentGpu_auditFixedLayout(g)==MSX_RESIDENT_ERR_GENERATION);
        OK(MSXresidentGpu_testAuditFlip(g,i)==0);
        OK(MSXresidentGpu_auditFixedLayout(g)==0);
    }
    MSXresidentBudget_snapshot(budget,&a);
    OK(a.allocated[0]==b.allocated[0]&&a.allocated[1]==b.allocated[1]&&a.allocated[2]==b.allocated[2]);
    /* Close/reopen capture is object-local; exact and one-byte-short audit budgets. */
    MSXresidentGpu_close(g);MSXresidentBudget_snapshot(budget,&a);
    OK(!a.allocated[0]&&!a.allocated[1]&&!a.allocated[2]&&!a.reserved[0]);
    g=initial4();MSXresidentBudget_snapshot(budget,&a);
    uint64_t host=a.allocated[0]+a.allocated[1];
    OK(MSXresidentBudget_configure(budget,host+charged-1,UINT64_MAX)==MSX_BUDGET_OK);
    OK(MSXresidentGpu_auditFixedLayout(g)==MSX_RESIDENT_ERR_MEMORY);
    MSXresidentBudget_snapshot(budget,&b);
    OK(a.allocated[0]==b.allocated[0]&&!b.reserved[0]);
    OK(MSXresidentBudget_configure(budget,host+charged,UINT64_MAX)==MSX_BUDGET_OK);
    OK(MSXresidentGpu_auditFixedLayout(g)==0);
    OK(MSXresidentGpu_auditFixedLayout(g)==0);
    MSXresidentGpu_close(g);MSXresidentBudget_snapshot(budget,&b);
    OK(!b.allocated[0]&&!b.allocated[1]&&!b.allocated[2]&&!b.reserved[0]);
    OK(MSXresidentBudget_configure(budget,UINT64_MAX,UINT64_MAX)==MSX_BUDGET_OK);
    g=open4();OK(MSXresidentGpu_auditFixedLayout(g)==MSX_RESIDENT_ERR_ARGUMENT);MSXresidentGpu_close(g);
    printf("c00_audit_assertions_passed=%d\nc00_audit_assertions_failed=%d\n",pass-passed,fail-failed);
}
