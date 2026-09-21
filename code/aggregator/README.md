# Final aggregator

`aggregator` is one implementation with two frozen modes. Both modes use the same model,
strict inline-tag parser, two-repair policy, audited recovery, review rules, and public prediction
contract. `prediction/records.jsonl` remains exact target-coordinate output, while
`prediction/results.jsonl` retains one outcome per target. After repair exhaustion, whitelisted
one-to-one CP1252 punctuation normalization may be projected back to exact source offsets; other
structurally valid extractions remain available in `model_sentence` coordinates as provisional
results.

- `EvidenceOnly` sees target sentence, normalized target TRFs, and at most eight hard-gated
  examples with exact labels and `helpfulness_score`.
- `WithExpertResults` sees the same evidence plus `skill-prediction-result-v1` proposals from
  the independent TRF and exemplar experts. Legacy `skill-prediction-v1` inputs remain readable.

The evidence-only CLI rejects expert paths. The expert-aware CLI requires both artifact paths, but
an individual target may lack one or both usable expert proposals. In that case prediction continues
from target TRFs and examples, with `expert_prediction_missing:<branch>` review reasons. Expert
disagreement is audited and is not itself a review trigger.

Terminal model-output validation issues are retained in `prediction/failures.jsonl`,
`prediction/results.jsonl`, and `audit/validation_issues.jsonl`. They are deduplicated by target;
at most `floor(N × 3%)` affected targets produces `completed`, while a larger rate produces
`partial`. Provider/runtime, missing-result, identity, source, and hash failures are never covered
by this allowance.

Prepare without a model call:

```powershell
.\scripts\aggregator\run.ps1 -Mode EvidenceOnly -RunId check-evidence `
  -ContextManifest <manifest.json> -ContextRecords <records.jsonl> -PrepareOnly
```

Run the expert-aware version:

```powershell
.\scripts\aggregator\run.ps1 -Mode WithExpertResults -RunId final-experts `
  -ContextManifest <manifest.json> -ContextRecords <records.jsonl> `
  -TrfPredictions <trf-records.jsonl> -ExemplarPredictions <exemplar-records.jsonl> `
  -AllowNetwork -ConfirmFullRun
```

Validate a prepared or completed run with
`.\scripts\aggregator\validate.ps1 -RunId <run-id>`. Gold is accepted only by the separate
`evaluate.ps1` command and never by the runner or prompt builder.
