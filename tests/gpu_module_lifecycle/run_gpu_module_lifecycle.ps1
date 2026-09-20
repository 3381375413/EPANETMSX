param(
    [Parameter(Mandatory = $true)]
    [string]$RuntimeDir,
    [Parameter(Mandatory = $true)]
    [string]$InpFile,
    [Parameter(Mandatory = $true)]
    [string]$NoMsxFile,
    [Parameter(Mandatory = $true)]
    [string]$YesMsxFile,
    [Parameter(Mandatory = $true)]
    [string]$ResultDir,
    [string[]]$ExplicitExpectedModes,
    [switch]$Reverse
)

$ErrorActionPreference = "Stop"
$RuntimeDir = [IO.Path]::GetFullPath($RuntimeDir)
$InpFile = [IO.Path]::GetFullPath($InpFile)
$NoMsxFile = [IO.Path]::GetFullPath($NoMsxFile)
$YesMsxFile = [IO.Path]::GetFullPath($YesMsxFile)
$ResultDir = [IO.Path]::GetFullPath($ResultDir)
foreach ($path in @($RuntimeDir, $InpFile, $NoMsxFile, $YesMsxFile)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing lifecycle gate input: $path"
    }
}
$resolvedNoMsx = [IO.Path]::GetFullPath($NoMsxFile)
$resolvedYesMsx = [IO.Path]::GetFullPath($YesMsxFile)
if ([StringComparer]::OrdinalIgnoreCase.Equals($resolvedNoMsx, $resolvedYesMsx)) {
    throw "NoMsxFile and YesMsxFile must be different files"
}

$expectedModes = @('NO', 'YES')
if ($ExplicitExpectedModes) {
    if ($ExplicitExpectedModes.Count -ne 2) {
        throw "ExplicitExpectedModes must contain exactly two entries: NO/YES roles"
    }
    $expectedModes = @(
        $ExplicitExpectedModes[0].ToUpperInvariant(),
        $ExplicitExpectedModes[1].ToUpperInvariant()
    )
}
foreach ($mode in $expectedModes) {
    if ($mode -notin @('NO', 'YES')) {
        throw "Unsupported expected GPU_COMPILER mode '$mode'; use NO or YES"
    }
}
function Read-GpuCompilerMode([string]$msxPath) {
    foreach ($line in Get-Content -LiteralPath $msxPath) {
        if ($line -match '^\s*[;#]') {
            continue
        }
        if ($line -match '^\s*GPU_COMPILER\s+(NO|YES)\s*$') {
            return $Matches[1].ToUpperInvariant()
        }
    }
    throw "Missing non-comment GPU_COMPILER NO/YES directive in $msxPath"
}
$actualNoMode = Read-GpuCompilerMode $NoMsxFile
$actualYesMode = Read-GpuCompilerMode $YesMsxFile
if ($actualNoMode -ne $expectedModes[0] -or $actualYesMode -ne $expectedModes[1]) {
    throw "Lifecycle role mismatch: NoMsxFile=$actualNoMode (expected $($expectedModes[0])), YesMsxFile=$actualYesMode (expected $($expectedModes[1]))"
}
New-Item -ItemType Directory -Force -Path $ResultDir | Out-Null

$env:Path = $RuntimeDir + ";" + $env:Path
$env:OMP_NUM_THREADS = "8"
$env:OMP_DYNAMIC = "FALSE"
$env:OMP_NESTED = "FALSE"
$env:MSX_PROFILE = "off"
Remove-Item Env:MSX_PROFILE_DETAIL -ErrorAction SilentlyContinue
Set-Location $RuntimeDir

Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;

public static class MsxGpuModuleLifecycleNative {
    [DllImport("epanetmsx.dll", CallingConvention=CallingConvention.Cdecl, CharSet=CharSet.Ansi)]
    public static extern int MSXENopen(string inp, string rpt, string bin);
    [DllImport("epanetmsx.dll", CallingConvention=CallingConvention.Cdecl)]
    public static extern int MSXENclose();
    [DllImport("epanetmsx.dll", CallingConvention=CallingConvention.Cdecl, CharSet=CharSet.Ansi)]
    public static extern int MSXopen(string msx);
    [DllImport("epanetmsx.dll", CallingConvention=CallingConvention.Cdecl)]
    public static extern int MSXclose();
    [DllImport("epanetmsx.dll", CallingConvention=CallingConvention.Cdecl)]
    public static extern int MSXsolveH();
    [DllImport("epanetmsx.dll", CallingConvention=CallingConvention.Cdecl)]
    public static extern int MSXinit(int saveFlag);
    [DllImport("epanetmsx.dll", CallingConvention=CallingConvention.Cdecl)]
    public static extern int MSXstep(ref double time, ref double timeLeft);
    [DllImport("epanetmsx.dll", CallingConvention=CallingConvention.Cdecl)]
    public static extern int MSXreport();

    public static int Run(string inp, string msx, string report) {
        int err = MSXENopen(inp, report, "");
        if (err != 0) return 1000 + err;
        err = MSXopen(msx);
        if (err != 0) { MSXENclose(); return 2000 + err; }
        err = MSXsolveH();
        if (err != 0) { MSXclose(); MSXENclose(); return 3000 + err; }
        err = MSXinit(1);
        if (err != 0) { MSXclose(); MSXENclose(); return 4000 + err; }
        double time = 0.0;
        double timeLeft = 0.0;
        do {
            err = MSXstep(ref time, ref timeLeft);
        } while (err == 0 && timeLeft > 0.0);
        if (err == 0) err = MSXreport();
        int closeErr = MSXclose();
        int encloseErr = MSXENclose();
        if (err != 0) return 5000 + err;
        if (closeErr != 0) return 6000 + closeErr;
        if (encloseErr != 0) return 7000 + encloseErr;
        return 0;
    }
}
'@

$first = if ($Reverse) {
    @(@{ Name = "YES"; Msx = $YesMsxFile }, @{ Name = "NO"; Msx = $NoMsxFile })
} else {
    @(@{ Name = "NO"; Msx = $NoMsxFile }, @{ Name = "YES"; Msx = $YesMsxFile })
}
$rows = @()
$total = [Diagnostics.Stopwatch]::StartNew()
foreach ($case in $first) {
    $report = Join-Path $ResultDir ("same_process_{0}.rpt" -f $case.Name.ToLowerInvariant())
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $code = [MsxGpuModuleLifecycleNative]::Run($InpFile, $case.Msx, $report)
    $watch.Stop()
    $rows += [pscustomobject]@{
        order = if ($Reverse) { "YES_NO" } else { "NO_YES" }
        mode = $case.Name
        exit_code = $code
        elapsed_ms = [math]::Round($watch.Elapsed.TotalMilliseconds, 3)
        report = $report
    }
    if ($code -ne 0) {
        $rows | Export-Csv -LiteralPath (Join-Path $ResultDir "lifecycle_results.csv") -NoTypeInformation
        throw "Lifecycle run failed for $($case.Name): $code"
    }
}
$total.Stop()
$rows | Export-Csv -LiteralPath (Join-Path $ResultDir "lifecycle_results.csv") -NoTypeInformation
@(
    "order=$(if ($Reverse) { 'YES_NO' } else { 'NO_YES' })"
    "total_elapsed_ms=$([math]::Round($total.Elapsed.TotalMilliseconds, 3))"
    "all_exit_zero=True"
) | Set-Content -LiteralPath (Join-Path $ResultDir "acceptance.txt") -Encoding UTF8
$rows | Format-Table -AutoSize
