#ifndef MSXRESIDENT_INITIAL_GOLDEN_H
#define MSXRESIDENT_INITIAL_GOLDEN_H
/* Isolated TEST_API only. Call the checkpoint immediately after successful
   time-zero initSegs, before any reaction or transport mutation. Schema 3
   exports defined state only: species index 0, CPU addresses and the unused,
   uninitialized Source.massRate member are excluded. */
#ifdef MSX_RESIDENT_TEST_API
int MSXinitialGolden_checkpoint(void);
#endif
#endif
