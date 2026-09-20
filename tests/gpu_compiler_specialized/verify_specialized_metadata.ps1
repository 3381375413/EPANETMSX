param(
    [Parameter(Mandatory = $true)]
    [string]$MetadataFile,
    [Parameter(Mandatory = $true)]
    [string]$SourceFile,
    [int[]]$ExpectedRateSpecies = @(1, 2, 3, 4, 5, 8, 9, 14)
)

$ErrorActionPreference = 'Stop'
$MetadataFile = [IO.Path]::GetFullPath($MetadataFile)
$SourceFile = [IO.Path]::GetFullPath($SourceFile)
foreach ($path in @($MetadataFile, $SourceFile)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing specialized evaluator test input: $path"
    }
}

function Require-Text([string]$text, [string]$pattern, [string]$label) {
    if ($text -notmatch $pattern) {
        throw "Missing $label"
    }
}

$metadata = Get-Content -LiteralPath $MetadataFile -Raw
$source = Get-Content -LiteralPath $SourceFile -Raw
Require-Text $metadata '^cache_version=v26_ros2_analytic_rate_jacobian_hydpipe_abi3' 'v26 analytic Jacobian cache ABI'
Require-Text $metadata '(?m)^rate_evaluator=joint_fixed_outputs\r?$' 'joint RATE evaluator metadata'
Require-Text $metadata '(?m)^rate_jacobian_mode=analytic\r?$' 'analytic Jacobian mode metadata'
Require-Text $metadata '(?m)^rate_jacobian_eligibility=eligible\r?$' 'analytic Jacobian eligibility metadata'
Require-Text $source 'specialized-model-dims-v3-analytic-rate-jacobian-kernel-abi3' 'generator ABI v3'
Require-Text $source 'dims->nRate < 1' 'zero-RATE admission guard'
Require-Text $source 'const Prog\* prog,const Prog\* termProg,const Instr\* instr' 'ROS2 helper ABI'
Require-Text $source '\(void\)n;\(void\)rs;\(void\)prog;' 'unused helper ABI arguments'
Require-Text $source 'out\[%d\]=\(v==v\?v:0\.0\)' 'per-RATE NaN clamp generation'
Require-Text $source 'ros2_eval_rates_jacobian' 'joint analytic Jacobian helper'
Require-Text $source 'jac\[%d\*jacStride\+%d\]' 'row/ordinal Jacobian mapping'
Require-Text $source 'if\(!isfinite\(d\)\)\*analyticOk=0' 'fail-closed derivative check'
Require-Text $source 'jc\+\+;gh0=0\.0' 'analytic Jacobian does not inflate nfcn'
Require-Text $source 'if \(!start\) return ERR_GPU_UNSUPPORTED_FEATURE' 'missing helper start failure'
Require-Text $source 'if \(!end\) return ERR_GPU_UNSUPPORTED_FEATURE' 'missing helper end failure'

$mapping = @()
$inMapping = $false
foreach ($line in (Get-Content -LiteralPath $MetadataFile)) {
    if ($line -eq 'rate_ordinal,species_index') {
        $inMapping = $true
        continue
    }
    if ($inMapping -and $line -eq 'attributes') { break }
    if ($inMapping -and $line -match '^([0-9]+),([0-9]+)$') {
        $mapping += [int]$Matches[2]
    }
}
if (($mapping -join ',') -ne ($ExpectedRateSpecies -join ',')) {
    throw "RATE ordinal mapping mismatch: actual=$($mapping -join ',') expected=$($ExpectedRateSpecies -join ',')"
}

Write-Output ('specialized_metadata_passed=1')
Write-Output ('cache_version=v26_ros2_analytic_rate_jacobian_hydpipe_abi3')
Write-Output ('rate_species=' + ($mapping -join ','))
