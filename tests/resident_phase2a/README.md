# Phase 2a resident CPU harness

This CPU-only harness links the real `msxresident_core.c` and
`msxsegment_storage.c`. It covers ten Phase 2a classes: CSV/hash
canonicalization, row layout, sparse patches, atomic observer preflight,
generation reuse, handoff planning, stale results, transaction abort, real
Hybrid materialization/commit, fail-closed transaction preflight, and the
steady-state transaction arena allocation regression.

The fixture supplies only EPANET/pool boundary stubs. Transaction classes use
the production Hybrid Dense-Core validation and materialization routines; they
do not reproduce storage algorithms in the harness.
