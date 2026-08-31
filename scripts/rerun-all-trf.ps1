param(
    [Parameter(Mandatory = $true)]
    [string]$RunId,
    [Parameter(Mandatory = $true)]
    [Alias("SelfAnnotatorId", "SourceRunId")]
    [string]$SelfAnnotatorRunId,
    [Parameter(Mandatory = $true)]
    [ValidateSet("independent", "leave-one-out")]
    [string]$Mode,
    [Alias("Input")]
    [string]$InputPath,
    [int]$Limit,
    [switch]$PrepareOnly,
    [switch]$AllowModelDownload,
    [switch]$AllowNetwork,
    [switch]$ConfirmFullRun,
    [switch]$Resume,
    [switch]$RetryFailed,
    [string]$ReuseEmbeddingsFrom,
    [string]$PythonExecutable = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$Script = Join-Path $ProjectRoot "code\trf_full\RunFullTRF.py"
$Arguments = @(
    $Script,
    "--run-id", $RunId,
    "--self-annotator-run-id", $SelfAnnotatorRunId,
    "--mode", $Mode
)

if ($InputPath) { $Arguments += @("--input", $InputPath) }
if ($PSBoundParameters.ContainsKey("Limit")) { $Arguments += @("--limit", $Limit) }
if ($PrepareOnly) { $Arguments += "--prepare-only" }
if ($AllowModelDownload) { $Arguments += "--allow-model-download" }
if ($AllowNetwork) { $Arguments += "--allow-network" }
if ($ConfirmFullRun) { $Arguments += "--confirm-full-run" }
if ($Resume) { $Arguments += "--resume" }
if ($RetryFailed) { $Arguments += "--retry-failed" }
if ($ReuseEmbeddingsFrom) {
    $Arguments += @("--reuse-embeddings-from", $ReuseEmbeddingsFrom)
}

& $PythonExecutable @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Complete TRF rerun failed (exit code $LASTEXITCODE)"
}
