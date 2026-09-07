param(
    [Parameter(Mandatory = $true)] [string]$RunId,
    [string]$Input,
    [Nullable[int]]$Limit = $null,
    [switch]$Resume,
    [switch]$FailFast,
    [string]$ReaggregateFrom,
    [string]$ConfigPath,
    [string]$PythonExecutable = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$Config = if ($ConfigPath) { $ConfigPath } else { Join-Path $ProjectRoot "config\self_annotator.json" }

if ($ReaggregateFrom) {
    $Arguments = @(
        (Join-Path $ProjectRoot "code\self_annotator\reaggregate.py"),
        "--config", $Config, "--source-run-root", $ReaggregateFrom, "--run-id", $RunId
    )
} else {
    $Arguments = @(
        (Join-Path $ProjectRoot "code\self_annotator\runner.py"),
        "--config", $Config, "--run-id", $RunId
    )
    if ($Input) { $Arguments += @("--input", $Input) }
    if ($null -ne $Limit) { $Arguments += @("--limit", $Limit) }
    if ($Resume) { $Arguments += "--resume" }
    if ($FailFast) { $Arguments += "--fail-fast" }
}

& $PythonExecutable @Arguments
if ($LASTEXITCODE -ne 0) { throw "Self annotator failed (exit code $LASTEXITCODE)" }
