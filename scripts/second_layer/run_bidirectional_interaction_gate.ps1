param(
    [Parameter(Mandatory = $true)]
    [string]$RunId,
    [Parameter(Mandatory = $true)]
    [string]$GateSpec,
    [ValidateSet("none", "trf_to_exemplar", "exemplar_to_trf", "bidirectional")]
    [string]$Mode = "bidirectional",
    [ValidateSet(1, 2)]
    [int]$MaxRounds = 2,
    [switch]$PrepareOnly,
    [switch]$Resume,
    [switch]$RetryFailed,
    [switch]$AllowNetwork,
    [string]$ConfigPath = "config/second_layer_bidirectional_interaction.json",
    [string]$PythonExecutable = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$runner = Join-Path $projectRoot "code\second_layer\bidirectional_interaction\gate_runner.py"
$arguments = @(
    $runner,
    "--config", $ConfigPath,
    "--run-id", $RunId,
    "--gate-spec", $GateSpec,
    "--mode", $Mode,
    "--max-rounds", $MaxRounds
)
if ($PrepareOnly) { $arguments += "--prepare-only" }
if ($Resume) { $arguments += "--resume" }
if ($RetryFailed) { $arguments += "--retry-failed" }
if ($AllowNetwork) { $arguments += "--allow-network" }

Push-Location $projectRoot
try {
    & $PythonExecutable @arguments
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}
