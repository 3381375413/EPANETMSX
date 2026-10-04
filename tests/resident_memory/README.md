# Resident memory foundation tests

Configure this directory with CMake and build Release. Run `ctest -C Release`.

`resident_mempool_tests` injects failure at every initial/growth heap allocation,
checks retained backing after reset, stable reuse, separate pool handles, exact
budget and one-byte-short admission, rejection without cursor mutation, and final
zero live allocations. It prints compiled root/header sizes for the real audit.

`resident_budget_tests` checks shared pageable+pinned limits, separate device
limits, reserve/commit/cancel/release, invalid ownership, overcommit, duplicate
operations, UINT64_MAX boundaries, and 80,000 operations on eight Windows threads.

`resident_alloc_tests` checks aligned, zeroed pageable storage, the charged
allocation header, exact/one-byte-short budgets, size overflow, real malloc
failure cancellation, preservation of an earlier allocation after partial OOM,
and final zero heap/reservation accounting. CUDA Phase2b additionally compares
dry-run costs with actual pageable/pinned/device charges and tests partial-arena
budget refusal and preexisting reservations.

Production integration covers QualPool/HashPool backing, Core, Capacity,
Runtime, Hybrid/ring heap arrays, and Resident CUDA pageable/pinned/device arrays.
Embedded ledger/header overhead is charged, and array cost estimators include it.
The global limits default to UINT64_MAX. Existing AUTO
hostBudget is still a planning budget and is not yet connected to this allocator
budget. Chemistry/project/GPU-program workspaces and partition quotas must be
integrated before claiming a production managed hard budget or enabling M5.

Audit is opt-in via `MSX_RESIDENT_MEMORY_AUDIT=1`. It writes pool backing,
CPU boundary occupancy, and logical transaction demand separately. Requested
backing bytes include embedded budget bookkeeping; they are not OS memory usage.
`resident_budget_lifetime.csv` records serialized reserve/commit/cancel/release
totals. `resident_allocation_manifest.csv` maps committed tokens to source or
buffer names. The audit option is frozen on first use; one buffered handle per
file is closed at zero live backing to avoid per-event fopen overhead.
