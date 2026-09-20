param(
    [Parameter(Mandatory = $true)][string]$RuntimeDir,
    [Parameter(Mandatory = $true)][string]$InputMsx,
    [Parameter(Mandatory = $true)][string]$InputInp,
    [Parameter(Mandatory = $true)][string]$CapacityFile,
    [Parameter(Mandatory = $true)][string]$RunVcBat,
    [Parameter(Mandatory = $true)][string]$GpuCacheDir,
    [Parameter(Mandatory = $true)][string]$WorkDir,
    [int]$DurationHours = 48
)

$ErrorActionPreference = 'Stop'
$RuntimeDir = [IO.Path]::GetFullPath($RuntimeDir)
$InputMsx = [IO.Path]::GetFullPath($InputMsx)
$InputInp = [IO.Path]::GetFullPath($InputInp)
$CapacityFile = [IO.Path]::GetFullPath($CapacityFile)
$RunVcBat = [IO.Path]::GetFullPath($RunVcBat)
$GpuCacheDir = [IO.Path]::GetFullPath($GpuCacheDir)
$WorkDir = [IO.Path]::GetFullPath($WorkDir)
foreach ($path in @($RuntimeDir, $InputMsx, $InputInp, $CapacityFile, $RunVcBat, $GpuCacheDir)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing FD fallback input: $path" }
}
$runtimeExe = Join-Path $RuntimeDir 'runepanetmsx.exe'
foreach ($name in @('runepanetmsx.exe', 'epanetmsx.dll', 'epanet2.dll')) {
    if (-not (Test-Path -LiteralPath (Join-Path $RuntimeDir $name))) {
        throw "Runtime is not self-contained: $name"
    }
}
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
Copy-Item -LiteralPath $RunVcBat -Destination (Join-Path $WorkDir 'runvc.bat') -Force
Copy-Item -LiteralPath $InputInp -Destination (Join-Path $WorkDir 'input.inp') -Force
Copy-Item -LiteralPath $CapacityFile -Destination (Join-Path $WorkDir 'resident_capacity.csv') -Force
$fixtureMsx = Join-Path $WorkDir 'fd_ineligible.msx'
$text = Get-Content -LiteralPath $InputMsx -Raw
$replacement = 'RATE   HOCL    SIN(H*0) -a1 + a2 - a3 + a4 + a8 - a12'
$fixture = [regex]::Replace($text, '(?m)^RATE\s+HOCL\s+.*$', $replacement, 1)
if ($fixture -eq $text) { throw 'RATE fixture splice marker missing' }
# The modified expression changes the case hash, so keep the Resident capacity
# artifact but disable Resident mode for this specialized GPU-only FD gate.
$fixture = [regex]::Replace($fixture, '(?mi)^GPU_CORE_MODE\s+RESIDENT\s*$', 'GPU_CORE_MODE OFF', 1)
[IO.File]::WriteAllText($fixtureMsx, $fixture, [Text.UTF8Encoding]::new($false))
$shortInp = Join-Path $WorkDir 'input.inp'
if ($DurationHours -ne 48) {
    $inpText = Get-Content -LiteralPath $shortInp -Raw
    $inpText = [regex]::Replace($inpText, '(?mi)^\s*Duration\s+\S+', ('Duration {0}:00' -f $DurationHours), 1)
    [IO.File]::WriteAllText($shortInp, $inpText, [Text.UTF8Encoding]::new($false))
}
$stdout = Join-Path $WorkDir 'stdout.txt'
$stderr = Join-Path $WorkDir 'stderr.txt'
$oldPath = $env:Path
$oldProfile = $env:MSX_PROFILE
$oldThreads = $env:OMP_NUM_THREADS
$oldDynamic = $env:OMP_DYNAMIC
$oldNested = $env:OMP_NESTED
$env:Path = $RuntimeDir + ';' + $env:Path
$env:MSX_PROFILE = 'detail'
$env:OMP_NUM_THREADS = '8'
$env:OMP_DYNAMIC = 'FALSE'
$env:OMP_NESTED = 'FALSE'
try {
    $p = Start-Process -FilePath $runtimeExe `
        -ArgumentList $shortInp,$fixtureMsx,(Join-Path $WorkDir 'fd_ineligible.rpt') `
        -WorkingDirectory $WorkDir -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
        -NoNewWindow -PassThru -Wait
} finally {
    $env:Path = $oldPath
    $env:MSX_PROFILE = $oldProfile
    $env:OMP_NUM_THREADS = $oldThreads
    $env:OMP_DYNAMIC = $oldDynamic
    $env:OMP_NESTED = $oldNested
}
if ($p.ExitCode -ne 0) { throw "FD-ineligible fixture failed with exit=$($p.ExitCode)" }
$quality = Join-Path $WorkDir 'net3_nh2cl_ros2_resident_quality.rpt'
if (-not (Test-Path -LiteralPath $quality) -or (Get-Item $quality).Length -eq 0) {
    throw 'FD-ineligible quality report is missing or empty'
}
$timing = Import-Csv -LiteralPath (Join-Path $WorkDir 'msx_gpu_timing.csv') |
    Where-Object { $_.record -eq 'TOTAL' } | Select-Object -First 1
if (-not $timing -or [double]$timing.sim_time_sec -ne 172800) {
    throw "Expected hour48 timing row, got sim_time=$($timing.sim_time_sec)"
}
if ([double]$timing.ros2_nfcn -le 200000000) {
    throw "Expected finite-difference nfcn > 200M, got $($timing.ros2_nfcn)"
}
$metadata = Get-ChildItem -LiteralPath $GpuCacheDir -Filter '*v26*.meta' -File |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $metadata) { throw 'FD-ineligible metadata was not generated' }
$metaText = Get-Content -LiteralPath $metadata.FullName -Raw
if ($metaText -notmatch '(?m)^rate_jacobian_mode=finite_difference\r?$') {
    throw "Expected finite_difference mode: $($metadata.FullName)"
}
if ($metaText -notmatch '(?m)^rate_jacobian_eligibility=fd_ineligible\r?$') {
    throw "Expected fd_ineligible metadata: $($metadata.FullName)"
}
$qualityHash = (Get-FileHash -LiteralPath $quality -Algorithm SHA256).Hash
@"
# X01 FD fallback fixture

- exit: $($p.ExitCode)
- simulation_seconds: $($timing.sim_time_sec)
- ros2_nfcn: $($timing.ros2_nfcn)
- ros2_njac: $($timing.ros2_njac)
- ros2_naccept: $($timing.ros2_naccept)
- ros2_nreject: $($timing.ros2_nreject)
- quality_sha256: $qualityHash
- metadata: $($metadata.FullName)
- mode: finite_difference
- eligibility: fd_ineligible
"@ | Set-Content -LiteralPath (Join-Path $WorkDir 'acceptance.md') -Encoding UTF8
Write-Output 'fd_ineligible_fixture_passed=1'
Write-Output ('ros2_nfcn=' + $timing.ros2_nfcn)
Write-Output ('ros2_njac=' + $timing.ros2_njac)
Write-Output ('quality_sha256=' + $qualityHash)
Write-Output ('metadata=' + $metadata.FullName)
