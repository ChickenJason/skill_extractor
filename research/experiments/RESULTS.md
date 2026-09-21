# Final aggregator implementation results

## Verified implementation results

- Targeted aggregator and architecture suite: 13/13 passed.
- Current cross-consumer suite covering the shared result contract, TRF target pipeline, exemplar
  discriminator, and aggregator: 56/56 passed after the 3% policy change.
- Gold loader: 100 sentences and 213 unique exact spans parsed successfully.
- Existing real second-layer branch artifacts can build 326 evidence-only inputs containing 1833
  selected examples.
- Expert-aware construction now keeps targets for which one or both formal expert predictions are
  unavailable. Structurally parseable expert outputs remain visible with model-sentence coordinates;
  otherwise the branch continues from the same base evidence with missing-expert review reasons.
- A fake-client integration run showed that the expert-aware aggregator can reject a proposed
  `Python` span and publish a new `services` span, while retaining the missing-expert review reason.
- Validator reconstruction detects modified prompt records.
- Shared prediction recovery tests pass for TRF, exemplar, and aggregator consumers. Existing
  `idx=216` raw responses from both experts recover exact source spans across two audited
  `U+0097 → U+2014` substitutions.
- Deterministic threshold tests confirm that 3/100 and 9/326 affected targets complete, while
  4/100 and 10/326 are partial; provider/runtime outcomes remain blocking at any rate.
- Fake-client integration confirms that judgment/TRF extraction validation failures produce empty
  `needs_review` contexts, continue to final prediction, and remain in the validation-issue ledger.

## Accuracy results

Not measured. No network-backed 100-item aggregation runs were performed, so this document makes no
claim that either mode improves accuracy. Use `scripts/aggregator/evaluate.ps1` only after both runs
are completed and independently validated.

## Full-suite context

The post-change full repository suite ran 181 tests: 178 passed, 1 skipped, and two pre-existing
`collaborative_reflection_stratified_gate` selection tests failed because their current base artifact
does not contain `idx=13`. The aggregator tests and architecture gate passed; the unrelated second-
layer data issue was not modified.

## 2026-09-21 live smoke results

### Observed

- Self-annotation: 1 record, 5 live samples, 1 formal result, validator `valid`.
- Second layer: 1 target ready; TRF produced 2 target TRFs; the exemplar branch selected 3 examples;
  both expert predictions had outcome `exact`; no repair or validation issue occurred.
- Evidence-only aggregator: 1/1 exact result, 4 spans, zero repairs, zero validation issues, failure
  rate 0%.
- Expert-aware aggregator: 1/1 exact result, 4 spans, both experts available and in exact agreement,
  zero repairs, zero validation issues, failure rate 0%.
- The two aggregator modes returned the same final span set for the smoke sentence. Both independently
  split the experts' two longer proposals into four narrower spans.
- Every read-only validator returned `valid`; the provider key was not found in persisted artifacts.

### Interpretation

The cleanup preserved the executable three-layer path and both aggregator interfaces for this
one-record smoke. The matching result is not evidence of accuracy parity or expert contribution;
those claims still require the frozen 100-item Gold evaluation.
