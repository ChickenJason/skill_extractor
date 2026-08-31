param(
    [Parameter(Mandatory = $true)]
    [string]$RunId,
    [Nullable[int]]$Limit = $null,
    [switch]$Resume,
    [switch]$FailFast
)

$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$CodeRoot = Join-Path $ProjectRoot "code\self_consistent_annotation"
$PipelineConfig = Join-Path $ProjectRoot "config\pipeline.json"

$AskArguments = @(
    (Join-Path $CodeRoot "AskQwen.py"),
    "--config", $PipelineConfig,
    "--run-id", $RunId
)
if ($null -ne $Limit) {
    $AskArguments += @("--limit", $Limit)
}
if ($Resume) {
    $AskArguments += "--resume"
}
if ($FailFast) {
    $AskArguments += "--fail-fast"
}

& python @AskArguments
if ($LASTEXITCODE -ne 0) {
    throw "Qwen sampling failed (exit code $LASTEXITCODE)"
}

$Stages = @(
    @{ Script = "ParseAnswers.py"; Name = "Answer parsing" },
    @{ Script = "SpanXMLCConsensus.py"; Name = "Span XMLC consensus" },
    @{ Script = "SelectAnnotations.py"; Name = "Annotation selection" }
)
foreach ($Stage in $Stages) {
    $StageScript = Join-Path $CodeRoot $Stage.Script
    & python $StageScript --config $PipelineConfig --run-id $RunId
    if ($LASTEXITCODE -ne 0) {
        throw "$($Stage.Name) failed (exit code $LASTEXITCODE)"
    }
}

Write-Host "Base five-sample run completed: $RunId"
