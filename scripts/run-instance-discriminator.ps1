param(
    [Parameter(Mandatory = $true)]
    [string]$TargetTRFRunId,
    [Parameter(Mandatory = $true)]
    [string]$RunId,
    [int]$Limit,
    [switch]$PrepareOnly,
    [switch]$AllowNetwork,
    [switch]$Resume,
    [switch]$RetryFailed,
    [switch]$ConfirmFullRun,
    [string]$ConfigPath,
    [string]$PythonExecutable = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$Script = Join-Path $ProjectRoot "code\instance_discriminator\RunInstanceDiscriminator.py"
$Config = if ($ConfigPath) {
    [System.IO.Path]::GetFullPath($ConfigPath)
} else {
    Join-Path $ProjectRoot "config\instance_discriminator.json"
}
$Arguments = @(
    $Script,
    "--config", $Config,
    "--target-trf-run-id", $TargetTRFRunId,
    "--run-id", $RunId
)

if ($PSBoundParameters.ContainsKey("Limit")) { $Arguments += @("--limit", $Limit) }
if ($PrepareOnly) { $Arguments += "--prepare-only" }
if ($AllowNetwork) { $Arguments += "--allow-network" }
if ($Resume) { $Arguments += "--resume" }
if ($RetryFailed) { $Arguments += "--retry-failed" }
if ($ConfirmFullRun) { $Arguments += "--confirm-full-run" }

& $PythonExecutable @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Instance discriminator pipeline failed (exit code $LASTEXITCODE)"
}
