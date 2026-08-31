$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$GenerateScript = Join-Path $ProjectRoot "code\self_consistent_annotation\GeneratePrompts.py"
$PromptConfig = Join-Path $ProjectRoot "config\prompt.json"
$TestsRoot = Join-Path $ProjectRoot "tests"
$env:PYTHONDONTWRITEBYTECODE = "1"

& python $GenerateScript --config $PromptConfig --validate-only
if ($LASTEXITCODE -ne 0) {
    throw "Dataset or prompt validation failed (exit code $LASTEXITCODE)"
}

& python -m unittest discover -s $TestsRoot -v
if ($LASTEXITCODE -ne 0) {
    throw "Offline tests failed (exit code $LASTEXITCODE)"
}
