[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ResultDir,
    [ValidateSet('FULL_OMP8', 'FULL_SERIAL')]
    [string]$ExpectedMode = 'FULL_OMP8'
)

$ErrorActionPreference = 'Stop'

function Read-Sidecar([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing Resident scan audit sidecar: $Path"
    }
    $values = @{}
    foreach ($line in (Get-Content -LiteralPath $Path)) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        $parts = $line -split '=', 2
        if ($parts.Count -ne 2 -or [string]::IsNullOrWhiteSpace($parts[0])) {
            throw "Malformed Resident scan audit line in $Path`: $line"
        }
        if ($values.ContainsKey($parts[0])) {
            throw "Duplicate Resident scan audit key '$($parts[0])' in $Path"
        }
        $values[$parts[0]] = $parts[1].Trim()
    }
    foreach ($key in @('scan_mode', 'team_size', 'omp_dynamic', 'omp_nested')) {
        if (-not $values.ContainsKey($key)) {
            throw "Missing '$key' in Resident scan audit sidecar: $Path"
        }
    }
    foreach ($key in @('team_size', 'omp_dynamic', 'omp_nested')) {
        $number = 0
        if (-not [int]::TryParse($values[$key], [ref]$number)) {
            throw "Non-integer '$key' in Resident scan audit sidecar: $Path"
        }
        $values[$key] = $number
    }
    return $values
}

$production = Read-Sidecar (Join-Path $ResultDir 'resident_scan_team_size.txt')
$omp = Read-Sidecar (Join-Path $ResultDir 'resident_scan_omp_team_size.txt')
foreach ($sidecar in @(@{Name = 'resident_scan_team_size.txt'; Data = $production},
                       @{Name = 'resident_scan_omp_team_size.txt'; Data = $omp})) {
    if ($sidecar.Data.scan_mode -ne $ExpectedMode) {
        throw "$($sidecar.Name) has scan_mode=$($sidecar.Data.scan_mode), expected $ExpectedMode"
    }
}

if ($ExpectedMode -eq 'FULL_OMP8') {
    foreach ($sidecar in @(@{Name = 'resident_scan_team_size.txt'; Data = $production},
                           @{Name = 'resident_scan_omp_team_size.txt'; Data = $omp})) {
        if ($sidecar.Data.team_size -ne 8 -or
            $sidecar.Data.omp_dynamic -ne 0 -or
            $sidecar.Data.omp_nested -ne 0) {
            throw "$($sidecar.Name) does not prove team_size=8, omp_dynamic=0, omp_nested=0"
        }
    }
}
elseif ($production.team_size -ne 1 -or $omp.team_size -ne 1) {
    throw "FULL_SERIAL audit must report team_size=1 in both sidecars"
}

Write-Output "Resident scan audit gate PASS: mode=$ExpectedMode; production/team=$($production.team_size); omp/team=$($omp.team_size); dynamic=$($omp.omp_dynamic); nested=$($omp.omp_nested)"
