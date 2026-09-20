# Resident scan OpenMP contract test

This target compiles and calls the production `hybridScanResidentSnapshots`
workshare through its test-only seam. It does not copy the scan algorithm. The
harness requires `_OPENMP`, uses 64 non-empty links, checks that an 8-thread
run assigns work to at least two workers, compares FULL_SERIAL fields, checks
worker-id bounds, and verifies close/reopen audit-state reset and environment
switching. `resident_phase2a` contains the Resident empty-init, reverse,
fixed-capacity, pool-failure, and error-prefix tests; its CMake target enables
the same production seam.

## Reproducible build and run

From the `EPANETMSX_under100_8t` worktree:

```powershell
cmake -S tests/resident_scan_omp -B build/resident_scan_omp -DCMAKE_BUILD_TYPE=Release
cmake --build build/resident_scan_omp --config Release --target resident_scan_omp_harness
$env:OMP_NUM_THREADS = '8'
$env:OMP_DYNAMIC = 'FALSE'
$env:OMP_NESTED = 'FALSE'
& build/resident_scan_omp/resident_scan_omp_harness.exe
```

The exact executable path is generator-dependent; for a Visual Studio
generator use `build/resident_scan_omp/Release/resident_scan_omp_harness.exe`.

The formal production audit is separate from this CPU harness. After a real
Resident CUDA audit run has exited, both sidecars are mandatory:

```powershell
powershell -ExecutionPolicy Bypass -File tests/resident_scan_omp/ResidentScanAuditGate.ps1 `
  -ResultDir <audit-result-directory> -ExpectedMode FULL_OMP8
```

The gate fails if either sidecar is missing, malformed, has a write-truncated
key set, or does not prove `scan_mode=FULL_OMP8`, `team_size=8`,
`omp_dynamic=0`, and `omp_nested=0`. A `FULL_SERIAL` smoke audit can be checked
with `-ExpectedMode FULL_SERIAL`; it must report `team_size=1` in both files.
Sidecar write failure is therefore a test/gate failure after the run, not a new
production chemistry error.
