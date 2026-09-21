param(
    [Parameter(Mandatory = $true)]
    [string]$RunId,
    [Parameter(Mandatory = $true)]
    [string]$BaseConcurrentRunId,
    [int]$Limit,
    [switch]$PrepareOnly,
    [switch]$AllowNetwork,
    [switch]$ConfirmFullRun,
    [switch]$Resume,
    [switch]$RetryFailed,
    [string]$ConfigPath = "config/second_layer_reflection_json_schema.json",
    [string]$PythonExecutable = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$runner = Join-Path $projectRoot "code\second_layer\collaborative_reflection_json_schema\runner.py"
$arguments = @(
    $runner,
    "--config", $ConfigPath,
    "--run-id", $RunId,
    "--base-concurrent-run-id", $BaseConcurrentRunId
)
if ($PSBoundParameters.ContainsKey("Limit")) { $arguments += @("--limit", $Limit) }
if ($PrepareOnly) { $arguments += "--prepare-only" }
if ($AllowNetwork) { $arguments += "--allow-network" }
if ($ConfirmFullRun) { $arguments += "--confirm-full-run" }
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
