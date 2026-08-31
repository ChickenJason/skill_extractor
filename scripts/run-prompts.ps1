param(
    [Nullable[int]]$Limit = $null
)

$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$GenerateScript = Join-Path $ProjectRoot "code\self_consistent_annotation\GeneratePrompts.py"
$PromptConfig = Join-Path $ProjectRoot "config\prompt.json"
$Arguments = @($GenerateScript, "--config", $PromptConfig)
if ($null -ne $Limit) {
    $Arguments += @("--limit", $Limit)
}

& python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Prompt generation failed (exit code $LASTEXITCODE)"
}
