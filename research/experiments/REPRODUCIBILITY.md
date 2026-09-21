# Reproducibility guide

## Environment

- Repository root: `E:\research\extractor\SkillSentence-Processor-0822`
- Python contract: `>=3.10,<3.11` from `pyproject.toml`
- Provider environment: `DASHSCOPE_API_KEY` and `DASHSCOPE_BASE_URL`
- Frozen model settings are in `config/aggregator.json`.

## Tests

```powershell
python -m compileall -q code\aggregator tests\aggregator
python -m unittest tests.aggregator.test_pipeline tests.common.test_architecture -v
python -m unittest tests.common.test_skill_prediction tests.trf.test_target `
  tests.instance_discriminator.test_pipeline tests.aggregator.test_pipeline -v
```

## Prepare the two matched runs

```powershell
.\scripts\aggregator\run.ps1 `
  -Mode EvidenceOnly `
  -RunId synthetic100-agg-evidence-v1 `
  -ContextManifest <second-layer-manifest> `
  -ContextRecords <second-layer-context> `
  -PrepareOnly

.\scripts\aggregator\run.ps1 `
  -Mode WithExpertResults `
  -RunId synthetic100-agg-experts-v1 `
  -ContextManifest <second-layer-manifest> `
  -ContextRecords <second-layer-context> `
  -TrfPredictions <trf-prediction-results> `
  -ExemplarPredictions <exemplar-prediction-results> `
  -PrepareOnly
```

Inspect the immutable inputs and prompts, then resume each run with
`-Resume -AllowNetwork -ConfirmFullRun`. A retry of terminal failures additionally requires
`-RetryFailed`.

## Validate and evaluate

```powershell
.\scripts\aggregator\validate.ps1 -RunId synthetic100-agg-evidence-v1
.\scripts\aggregator\validate.ps1 -RunId synthetic100-agg-experts-v1

.\scripts\aggregator\evaluate.ps1 `
  -EvidenceRunId synthetic100-agg-evidence-v1 `
  -ExpertRunId synthetic100-agg-experts-v1 `
  -Gold data\raw\synthetic_jd_sentences_100_gold.json `
  -TrfPredictions <trf-prediction-results> `
  -ExemplarPredictions <exemplar-prediction-results> `
  -Output research\experiments\aggregator-synthetic100-results.json
```

Gold is accepted only by the evaluator. The runner has no Gold argument, and the validator rebuilds
all runtime artifacts from locked non-Gold sources.

## Reproduce the one-record cleanup smoke

Set `DASHSCOPE_API_KEY` and `DASHSCOPE_BASE_URL` in the current process without writing them to a
file, then run with the project Python 3.10 environment:

```powershell
.\scripts\self_annotator\run.ps1 `
  -RunId cleanup-smoke-self-20260921 `
  -Input data\raw\synthetic_jd_sentences_100.json -Limit 1 `
  -PythonExecutable .\.venv-trf\Scripts\python.exe

.\scripts\second_layer\run.ps1 `
  -RunId cleanup-smoke-second-20260921-py310 `
  -TargetMode independent -Input data\raw\synthetic_jd_sentences_100.json `
  -Limit 1 -AllowNetwork -PythonExecutable .\.venv-trf\Scripts\python.exe
```

Use the resulting second-layer manifest/context and the two child
`prediction/results.jsonl` files as inputs to `scripts\aggregator\run.ps1`, once per mode with
`-Limit 1 -AllowNetwork`. Validate the four completed run IDs with the corresponding validators.
Do not reuse the run IDs unless using the explicit resume semantics.
