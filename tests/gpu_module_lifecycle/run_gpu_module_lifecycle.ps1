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
