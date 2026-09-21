param(
    [Parameter(Mandatory = $true)] [string]$RunId,
    [string]$ConfigPath,
    [string]$PythonExecutable = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..") )
$Config = if ($ConfigPath) { $ConfigPath } else { Join-Path $ProjectRoot "config\aggregator.json" }
& $PythonExecutable (Join-Path $ProjectRoot "code\aggregator\validator.py") --config $Config --run-id $RunId
if ($LASTEXITCODE -ne 0) { throw "Aggregator validation failed (exit code $LASTEXITCODE)" }
