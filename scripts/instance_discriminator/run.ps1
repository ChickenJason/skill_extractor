param(
    [Parameter(Mandatory = $true)] [string]$RunId,
    [Parameter(Mandatory = $true)] [string]$Targets,
    [Parameter(Mandatory = $true)] [string]$Candidates,
    [string]$Features,
    [Nullable[int]]$Limit = $null,
    [switch]$PrepareOnly,
    [switch]$AllowNetwork,
    [switch]$ConfirmFullRun,
    [switch]$Resume,
    [switch]$RetryFailed,
    [string]$ConfigPath,
    [string]$PythonExecutable = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$Config = if ($ConfigPath) { $ConfigPath } else { Join-Path $ProjectRoot "config\instance_discriminator.json" }
$Arguments = @(
    (Join-Path $ProjectRoot "code\instance_discriminator\runner.py"),
    "--config", $Config, "--run-id", $RunId,
    "--targets", $Targets, "--candidates", $Candidates
)
if ($Features) { $Arguments += @("--features", $Features) }
if ($null -ne $Limit) { $Arguments += @("--limit", $Limit) }
if ($PrepareOnly) { $Arguments += "--prepare-only" }
if ($AllowNetwork) { $Arguments += "--allow-network" }
if ($ConfirmFullRun) { $Arguments += "--confirm-full-run" }
if ($Resume) { $Arguments += "--resume" }
if ($RetryFailed) { $Arguments += "--retry-failed" }

& $PythonExecutable @Arguments
if ($LASTEXITCODE -ne 0) { throw "Instance discriminator failed (exit code $LASTEXITCODE)" }
