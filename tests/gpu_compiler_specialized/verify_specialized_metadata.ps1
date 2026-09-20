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
Require-Text $metadata '^cache_version=v25_ros2_joint_rate_hydpipe_abi2' 'v25 cache ABI'
Require-Text $metadata '(?m)^rate_evaluator=joint_fixed_outputs\r?$' 'joint RATE evaluator metadata'
Require-Text $source 'specialized-model-dims-v2-joint-rate-kernel-abi2' 'generator ABI v2'
Require-Text $source 'dims->nRate < 1' 'zero-RATE admission guard'
Require-Text $source 'const Prog\* prog,const Prog\* termProg,const Instr\* instr' 'ROS2 helper ABI'
Require-Text $source '\(void\)n;\(void\)rs;\(void\)prog;' 'unused helper ABI arguments'
Require-Text $source 'out\[%d\]=\(v==v\?v:0\.0\)' 'per-RATE NaN clamp generation'
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
Write-Output ('cache_version=v25_ros2_joint_rate_hydpipe_abi2')
Write-Output ('rate_species=' + ($mapping -join ','))
