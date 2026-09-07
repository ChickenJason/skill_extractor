param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Offline", "Target", "Full")]
    [string]$Mode,
    [Parameter(Mandatory = $true)] [string]$RunId,
    [string]$Demonstrations,
    [string]$Input,
    [string]$Feedback,
    [ValidateSet("independent", "leave-one-out")]
    [string]$TargetMode = "independent",
    [string]$GeneratedConfigPath,
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
$Config = if ($ConfigPath) { $ConfigPath } else { Join-Path $ProjectRoot "config\trf.json" }

if ($Mode -eq "Offline") {
    if ($Input -or $Feedback -or $null -ne $Limit -or $PrepareOnly -or $AllowNetwork -or
        $ConfirmFullRun -or $Resume -or $RetryFailed -or $ReuseEmbeddingsFrom) {
        throw "Offline mode accepts only -Demonstrations and -AllowModelDownload"
    }
    $Arguments = @(
        (Join-Path $ProjectRoot "code\trf\offline\runner.py"),
        "--config", $Config, "--run-id", $RunId
    )
    if ($Demonstrations) { $Arguments += @("--demonstrations", $Demonstrations) }
    if ($AllowModelDownload) { $Arguments += "--allow-model-download" }
} elseif ($Mode -eq "Target") {
    if (-not $GeneratedConfigPath) { throw "Target mode requires -GeneratedConfigPath" }
    if ($Demonstrations -or $AllowModelDownload) {
        throw "Target mode reads demonstrations from -GeneratedConfigPath"
    }
    $Arguments = @(
        (Join-Path $ProjectRoot "code\trf\target\runner.py"),
        "--config", $GeneratedConfigPath, "--run-id", $RunId, "--mode", $TargetMode
    )
    if ($Input) { $Arguments += @("--input", $Input) }
    if ($Feedback) { $Arguments += @("--feedback", $Feedback) }
    if ($null -ne $Limit) { $Arguments += @("--limit", $Limit) }
    if ($PrepareOnly) { $Arguments += "--prepare-only" }
    if ($AllowNetwork) { $Arguments += "--allow-network" }
    if ($ConfirmFullRun) { $Arguments += "--confirm-full-run" }
    if ($Resume) { $Arguments += "--resume" }
    if ($RetryFailed) { $Arguments += "--retry-failed" }
    if ($ReuseEmbeddingsFrom) {
        $Arguments += @("--reuse-embeddings-from", $ReuseEmbeddingsFrom)
    }
} else {
    $Arguments = @(
        (Join-Path $ProjectRoot "code\trf\orchestration\runner.py"),
        "--config", $Config, "--run-id", $RunId, "--mode", $TargetMode
    )
    if ($Demonstrations) { $Arguments += @("--demonstrations", $Demonstrations) }
    if ($AllowModelDownload) { $Arguments += "--allow-model-download" }
    if ($Input) { $Arguments += @("--input", $Input) }
    if ($Feedback) { $Arguments += @("--feedback", $Feedback) }
    if ($null -ne $Limit) { $Arguments += @("--limit", $Limit) }
    if ($PrepareOnly) { $Arguments += "--prepare-only" }
    if ($AllowNetwork) { $Arguments += "--allow-network" }
    if ($ConfirmFullRun) { $Arguments += "--confirm-full-run" }
    if ($Resume) { $Arguments += "--resume" }
    if ($RetryFailed) { $Arguments += "--retry-failed" }
    if ($ReuseEmbeddingsFrom) {
        $Arguments += @("--reuse-embeddings-from", $ReuseEmbeddingsFrom)
    }
}

& $PythonExecutable @Arguments
if ($LASTEXITCODE -ne 0) { throw "TRF $Mode run failed (exit code $LASTEXITCODE)" }
