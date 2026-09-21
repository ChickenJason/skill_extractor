param(
    [Parameter(Mandatory = $true)] [ValidateSet("EvidenceOnly", "WithExpertResults")] [string]$Mode,
    [Parameter(Mandatory = $true)] [string]$RunId,
    [Parameter(Mandatory = $true)] [string]$ContextManifest,
    [Parameter(Mandatory = $true)] [string]$ContextRecords,
    [string]$TrfPredictions,
    [string]$ExemplarPredictions,
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
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..") )
$Config = if ($ConfigPath) { $ConfigPath } else { Join-Path $ProjectRoot "config\aggregator.json" }
$Arguments = @(
    (Join-Path $ProjectRoot "code\aggregator\runner.py"),
    "--config", $Config, "--mode", $Mode, "--run-id", $RunId,
    "--context-manifest", $ContextManifest, "--context-records", $ContextRecords
)
if ($TrfPredictions) { $Arguments += @("--trf-predictions", $TrfPredictions) }
if ($ExemplarPredictions) { $Arguments += @("--exemplar-predictions", $ExemplarPredictions) }
if ($null -ne $Limit) { $Arguments += @("--limit", $Limit) }
if ($PrepareOnly) { $Arguments += "--prepare-only" }
if ($AllowNetwork) { $Arguments += "--allow-network" }
if ($ConfirmFullRun) { $Arguments += "--confirm-full-run" }
if ($Resume) { $Arguments += "--resume" }
if ($RetryFailed) { $Arguments += "--retry-failed" }

& $PythonExecutable @Arguments
if ($LASTEXITCODE -ne 0) { throw "Aggregator failed (exit code $LASTEXITCODE)" }
