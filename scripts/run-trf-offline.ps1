param(
    [Parameter(Mandatory = $true)]
    [string]$RunId,
    [Parameter(Mandatory = $true)]
    [Alias("SelfAnnotatorId", "SourceRunId")]
    [string]$SelfAnnotatorRunId,
    [switch]$AllowModelDownload,
    [string]$PythonExecutable = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$Script = Join-Path $ProjectRoot "code\trf\RunTRFOffline.py"
$Config = Join-Path $ProjectRoot "config\trf.json"
$Arguments = @(
    $Script,
    "--config", $Config,
    "--run-id", $RunId,
    "--self-annotator-run-id", $SelfAnnotatorRunId
)
if ($AllowModelDownload) {
    $Arguments += "--allow-model-download"
}

& $PythonExecutable @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Offline TRF pipeline failed (exit code $LASTEXITCODE)"
}
