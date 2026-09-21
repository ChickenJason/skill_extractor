param(
    [Parameter(Mandatory = $true)] [string]$RunId,
    [Parameter(Mandatory = $true)] [string]$BaseConcurrentRunId,
    [Nullable[int]]$Limit = $null,
    [switch]$PrepareOnly,
    [switch]$AllowNetwork,
    [switch]$ConfirmFullRun,
    [switch]$Resume,
    [switch]$RetryFailed,
    [string]$ConfigPath,
    [string]$PythonExecutable = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$Config = if ($ConfigPath) { $ConfigPath } else { Join-Path $ProjectRoot "config\second_layer_reflection.json" }
$Arguments = @(
    (Join-Path $ProjectRoot "code\second_layer\collaborative_reflection\runner.py"),
    "--config", $Config,
    "--run-id", $RunId,
    "--base-concurrent-run-id", $BaseConcurrentRunId
)
if ($null -ne $Limit) { $Arguments += @("--limit", $Limit) }
if ($PrepareOnly) { $Arguments += "--prepare-only" }
if ($AllowNetwork) { $Arguments += "--allow-network" }
if ($ConfirmFullRun) { $Arguments += "--confirm-full-run" }
if ($Resume) { $Arguments += "--resume" }
if ($RetryFailed) { $Arguments += "--retry-failed" }

& $PythonExecutable @Arguments
if ($LASTEXITCODE -ne 0) { throw "Collaborative reflection scheduling failed (exit code $LASTEXITCODE)" }
