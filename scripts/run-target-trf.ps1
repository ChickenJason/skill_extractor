param(
    [Parameter(Mandatory = $true)]
    [string]$RunId,
    [Parameter(Mandatory = $true)]
    [ValidateSet("independent", "leave-one-out")]
    [string]$Mode,
    [Alias("Input")]
    [string]$InputPath,
    [int]$Limit,
    [switch]$PrepareOnly,
    [switch]$AllowNetwork,
    [switch]$Resume,
    [switch]$RetryFailed,
    [string]$ReuseEmbeddingsFrom,
    [switch]$ConfirmFullRun,
    [string]$ConfigPath,
    [string]$PythonExecutable = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$Script = Join-Path $ProjectRoot "code\trf_target\RunTargetTRF.py"
$Config = if ($ConfigPath) {
    [System.IO.Path]::GetFullPath($ConfigPath)
} else {
    Join-Path $ProjectRoot "config\trf_target.json"
}
$Arguments = @($Script, "--config", $Config, "--run-id", $RunId, "--mode", $Mode)

if ($InputPath) { $Arguments += @("--input", $InputPath) }
if ($PSBoundParameters.ContainsKey("Limit")) { $Arguments += @("--limit", $Limit) }
if ($PrepareOnly) { $Arguments += "--prepare-only" }
if ($AllowNetwork) { $Arguments += "--allow-network" }
if ($Resume) { $Arguments += "--resume" }
if ($RetryFailed) { $Arguments += "--retry-failed" }
if ($ReuseEmbeddingsFrom) {
    $Arguments += @("--reuse-embeddings-from", $ReuseEmbeddingsFrom)
}
if ($ConfirmFullRun) { $Arguments += "--confirm-full-run" }

& $PythonExecutable @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Target TRF pipeline failed (exit code $LASTEXITCODE)"
}
