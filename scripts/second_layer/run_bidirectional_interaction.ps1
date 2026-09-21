param(
    [Parameter(Mandatory = $true)]
    [string]$RunId,
    [Parameter(Mandatory = $true)]
    [string]$BaseConcurrentRunId,
    [ValidateSet("none", "trf_to_exemplar", "exemplar_to_trf", "bidirectional")]
    [string]$Mode = "bidirectional",
    [ValidateSet(1, 2)]
    [int]$MaxRounds = 2,
    [int]$Limit,
    [switch]$PrepareOnly,
    [switch]$Resume,
    [switch]$RetryFailed,
    [switch]$AllowNetwork,
    [switch]$ConfirmFullRun,
    [string]$ConfigPath = "config/second_layer_bidirectional_interaction.json",
    [string]$PythonExecutable = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$runner = Join-Path $projectRoot "code\second_layer\bidirectional_interaction\runner.py"
$arguments = @(
    $runner,
    "--config", $ConfigPath,
    "--run-id", $RunId,
    "--base-concurrent-run-id", $BaseConcurrentRunId,
    "--mode", $Mode,
    "--max-rounds", $MaxRounds
)
if ($PSBoundParameters.ContainsKey("Limit")) { $arguments += @("--limit", $Limit) }
if ($PrepareOnly) { $arguments += "--prepare-only" }
if ($Resume) { $arguments += "--resume" }
if ($RetryFailed) { $arguments += "--retry-failed" }
if ($AllowNetwork) { $arguments += "--allow-network" }
if ($ConfirmFullRun) { $arguments += "--confirm-full-run" }

Push-Location $projectRoot
try {
    & $PythonExecutable @arguments
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}
