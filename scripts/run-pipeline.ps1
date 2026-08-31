param(
    [Parameter(Mandatory = $true)]
    [string]$RunId,
    [switch]$Resume,
    [switch]$FailFast,
    [switch]$SkipChecks
)

$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$RunsRoot = Join-Path $ProjectRoot "output\runs"

if (-not $SkipChecks) {
    Write-Host "[1/3] Running offline validation and tests"
    & (Join-Path $PSScriptRoot "run-tests.ps1")
    Write-Host "[2/3] Generating the active prompt file"
    & (Join-Path $PSScriptRoot "run-prompts.ps1")
}
else {
    Write-Host "[1/3] Offline checks skipped by -SkipChecks"
    Write-Host "[2/3] Prompt generation skipped by -SkipChecks"
}

Write-Host "[3/3] Fixed five-sample run: $RunId"
$Parameters = @{ RunId = $RunId }
if ($Resume) {
    $Parameters.Resume = $true
}
if ($FailFast) {
    $Parameters.FailFast = $true
}
& (Join-Path $PSScriptRoot "run-base.ps1") @Parameters

$RunRoot = Join-Path $RunsRoot $RunId
Write-Host "Pipeline completed"
Write-Host "Run:          $RunRoot"
Write-Host "Final output: $(Join-Path $RunRoot 'selected\annotations.json')"
