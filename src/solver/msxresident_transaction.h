#ifndef MSX_RESIDENT_TRANSACTION_H
#define MSX_RESIDENT_TRANSACTION_H
#include <stdint.h>
typedef enum {MSX_TX_IDLE,MSX_TX_PLANNING,MSX_TX_PREPARED,MSX_TX_STAGING,
    MSX_TX_APPLYING,MSX_TX_COMMITTED,MSX_TX_ABORTED,MSX_TX_FAILED,MSX_TX_POISONED} MSXTransactionState;
typedef struct {
    uint64_t sequence,working,published,completed;
    MSXTransactionState state;
    unsigned cpuChanged,gpuAccepted;
} MSXResidentTransaction;
static int MSXtransaction_readable(const MSXResidentTransaction *t)
{return (t->state==MSX_TX_IDLE||t->state==MSX_TX_COMMITTED||t->state==MSX_TX_ABORTED)&&
        t->published==t->completed;}
static int MSXtransaction_begin(MSXResidentTransaction *t)
{
    if(!MSXtransaction_readable(t)||t->sequence==UINT64_MAX||t->published==UINT64_MAX)return 0;
    ++t->sequence;t->working=t->published+1;t->cpuChanged=t->gpuAccepted=0;t->state=MSX_TX_PLANNING;return 1;
}
static int MSXtransaction_advance(MSXResidentTransaction *t,MSXTransactionState state)
{
    if((t->state==MSX_TX_PLANNING&&state==MSX_TX_PREPARED)||
       (t->state==MSX_TX_PREPARED&&state==MSX_TX_STAGING)||
       (t->state==MSX_TX_STAGING&&state==MSX_TX_APPLYING))
    {t->state=state;return 1;}return 0;
}
static int MSXtransaction_complete(MSXResidentTransaction *t)
{if(t->state!=MSX_TX_APPLYING)return 0;t->completed=t->working;return 1;}
static int MSXtransaction_commit(MSXResidentTransaction *t)
{
    if(t->state!=MSX_TX_APPLYING||t->completed!=t->working)return 0;
    t->published=t->working;t->state=MSX_TX_COMMITTED;return 1;
}
static void MSXtransaction_fail(MSXResidentTransaction *t,int gpuFault)
{t->state=(gpuFault||t->cpuChanged||t->gpuAccepted)?MSX_TX_POISONED:MSX_TX_FAILED;}
static int MSXtransaction_abort(MSXResidentTransaction *t)
{
    if(t->state==MSX_TX_ABORTED)return 1;
    if(t->cpuChanged||t->gpuAccepted){t->state=MSX_TX_POISONED;return 0;}
    if(t->state<MSX_TX_PLANNING||t->state>MSX_TX_APPLYING)return 0;
    t->working=t->published;t->state=MSX_TX_ABORTED;return 1;
}
#endif
