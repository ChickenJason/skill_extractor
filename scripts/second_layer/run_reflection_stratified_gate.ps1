param(
    [Parameter(Mandatory = $true)]
    [string]$RunId,
    [Parameter(Mandatory = $true)]
    [string]$BaseConcurrentRunId,
    [switch]$PrepareOnly,
    [switch]$AllowNetwork,
    [switch]$Resume,
    [switch]$RetryFailed,
    [string]$GateSpecPath = "config/gates/collaborative_reflection_stratified12_v1.json",
    [string]$ConfigPath = "config/second_layer_reflection_json_schema.json",
    [string]$PythonExecutable = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$runner = Join-Path $projectRoot "code\second_layer\collaborative_reflection_stratified_gate\runner.py"
$arguments = @(
    $runner,
    "--config", $ConfigPath,
    "--gate-spec", $GateSpecPath,
    "--run-id", $RunId,
    "--base-concurrent-run-id", $BaseConcurrentRunId
)
if ($PrepareOnly) { $arguments += "--prepare-only" }
if ($AllowNetwork) { $arguments += "--allow-network" }
if ($Resume) { $arguments += "--resume" }
if ($RetryFailed) { $arguments += "--retry-failed" }

Push-Location $projectRoot
try {
    & $PythonExecutable @arguments
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}
