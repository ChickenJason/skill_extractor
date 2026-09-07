param(
    [string]$PythonExecutable
)

$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$Config = Join-Path $ProjectRoot "config\self_annotator.json"
$Validator = Join-Path $ProjectRoot "code\self_annotator\runner.py"
$BundledPython = Join-Path $ProjectRoot ".venv-trf\Scripts\python.exe"
$Python = if ($PythonExecutable) {
    $PythonExecutable
} elseif (Test-Path -LiteralPath $BundledPython) {
    $BundledPython
} else {
    "python"
}
$env:PYTHONDONTWRITEBYTECODE = "1"

& $Python $Validator --config $Config --run-id validation-only --validate-only
if ($LASTEXITCODE -ne 0) { throw "Dataset or prompt validation failed (exit code $LASTEXITCODE)" }

& $Python -m unittest discover -s (Join-Path $ProjectRoot "tests") -t $ProjectRoot -v
if ($LASTEXITCODE -ne 0) { throw "Offline tests failed (exit code $LASTEXITCODE)" }
