param(
    [Parameter(Mandatory = $true)]
    [string]$SourceRunRoot,
    [string]$RunId = "span-xmlc-base5-replay-majority3-anchor-v1",
    [string]$PythonExecutable = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$Script = Join-Path $ProjectRoot "code\self_consistent_annotation\ReaggregateRun.py"
$PipelineConfig = Join-Path $ProjectRoot "config\pipeline.json"

& $PythonExecutable $Script `
    --config $PipelineConfig `
    --source-run-root $SourceRunRoot `
    --run-id $RunId
if ($LASTEXITCODE -ne 0) {
    throw "Offline reaggregation failed (exit code $LASTEXITCODE)"
}
