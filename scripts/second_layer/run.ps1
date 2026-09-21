param(
    [Parameter(Mandatory = $true)] [string]$RunId,
    [ValidateSet("independent", "leave-one-out")]
    [string]$TargetMode = "independent",
    [string]$Input,
    [Nullable[int]]$Limit = $null,
    [switch]$PrepareOnly,
    [switch]$AllowModelDownload,
    [switch]$AllowNetwork,
    [switch]$ConfirmFullRun,
    [switch]$Resume,
    [switch]$RetryFailed,
    [string]$ReuseEmbeddingsFrom,
    [string]$ConfigPath,
    [string]$PythonExecutable = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$Config = if ($ConfigPath) { $ConfigPath } else { Join-Path $ProjectRoot "config\second_layer.json" }
$Arguments = @(
    (Join-Path $ProjectRoot "code\second_layer\runner.py"),
    "--config", $Config, "--run-id", $RunId, "--target-mode", $TargetMode
)
if ($Input) { $Arguments += @("--input", $Input) }
if ($null -ne $Limit) { $Arguments += @("--limit", $Limit) }
if ($PrepareOnly) { $Arguments += "--prepare-only" }
if ($AllowModelDownload) { $Arguments += "--allow-model-download" }
if ($AllowNetwork) { $Arguments += "--allow-network" }
if ($ConfirmFullRun) { $Arguments += "--confirm-full-run" }
if ($Resume) { $Arguments += "--resume" }
if ($RetryFailed) { $Arguments += "--retry-failed" }
if ($ReuseEmbeddingsFrom) { $Arguments += @("--reuse-embeddings-from", $ReuseEmbeddingsFrom) }

& $PythonExecutable @Arguments
if ($LASTEXITCODE -ne 0) { throw "Second-layer parallel run failed (exit code $LASTEXITCODE)" }
