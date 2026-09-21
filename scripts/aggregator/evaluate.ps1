param(
    [Parameter(Mandatory = $true)] [string]$EvidenceRunId,
    [Parameter(Mandatory = $true)] [string]$ExpertRunId,
    [Parameter(Mandatory = $true)] [string]$Gold,
    [Parameter(Mandatory = $true)] [string]$TrfPredictions,
    [Parameter(Mandatory = $true)] [string]$ExemplarPredictions,
    [string]$Output,
    [int]$BootstrapSamples = 2000,
    [int]$Seed = 42,
    [string]$ConfigPath,
    [string]$PythonExecutable = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..") )
$Config = if ($ConfigPath) { $ConfigPath } else { Join-Path $ProjectRoot "config\aggregator.json" }
$Arguments = @(
    (Join-Path $ProjectRoot "code\aggregator\evaluation.py"),
    "--config", $Config,
    "--evidence-run-id", $EvidenceRunId,
    "--expert-run-id", $ExpertRunId,
    "--gold", $Gold,
    "--trf-predictions", $TrfPredictions,
    "--exemplar-predictions", $ExemplarPredictions,
    "--bootstrap-samples", $BootstrapSamples,
    "--seed", $Seed
)
if ($Output) { $Arguments += @("--output", $Output) }
& $PythonExecutable @Arguments
if ($LASTEXITCODE -ne 0) { throw "Aggregator evaluation failed (exit code $LASTEXITCODE)" }
