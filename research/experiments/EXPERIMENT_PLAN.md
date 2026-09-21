# Final aggregator experiment plan

## Question and hypothesis

The experiment asks whether exposing the parsed outputs of the two independent second-layer
experts improves final exact skill-span prediction beyond using the underlying TRF and exemplar
evidence alone.

The independent variable is a single prompt field: parsed expert proposals are absent in
`evidence_only` and present in `with_expert_results`. Model, temperature, output budget, strict
parser, repair budget, target contexts, example order, review rules, and evaluation data remain
fixed.

The hypothesis is accepted only if the expert-aware version improves exact-span micro F1 or
sentence exact-set match, does not reduce the other metric, and does not increase the false-positive
rate on negative sentences.

## Inputs and leakage boundary

- Runtime input: a completed `second-layer-context-v1` manifest and records.
- Expert-aware-only input: `skill-prediction-result-v1` records from branches `trf` and `exemplar`;
  legacy exact `skill-prediction-v1` remains compatible.
- Never supplied to the runner or prompt: Gold labels, expert raw responses, expert prompts,
  reasoning, exemplar `pseudo_trfs`, similarity, existence score, role, or reason codes.
- At most eight examples are used, in `gate_rank` order. The hard gate is
  `helpfulness_score > 3` (integer scores 4–5).
- A missing single expert creates a review reason. If both expert proposals are unavailable, the
  expert-aware branch continues from the same base TRF and exemplar evidence.

## Fixed settings

- Chat model: `qwen3.7-plus-2026-05-26`
- Temperature: 0
- Maximum output: 512 tokens
- Strict repair attempts: 2
- Terminal model-output validation issue tolerance: 3%, floored by target count and deduplicated by
  target across all model stages
- Post-repair retention: audited one-to-one CP1252 punctuation projection; all other structurally
  valid outputs retained downstream in explicit `model_sentence` coordinates
- Paired bootstrap: 2,000 samples, seed 42

## Evaluation

Run both modes on the same 100 Gold sentences and compare them with each single expert. Report:

- exact-span micro precision, recall, and F1;
- sentence-level exact-set match;
- skill-presence F1;
- negative-sentence false-positive rate;
- formal coverage, result coverage, repair rate, validation-issue rate, and provisional rate;
- per-sentence changes;
- paired bootstrap 95% intervals for exact-span F1 and exact-set-match deltas;
- expert-aware results on all targets and on the subset where both experts are available.

No online experiment was authorized or executed during implementation. The Gold file is reserved for
the offline evaluator.

## 2026-09-21 maintenance-smoke addendum

The behavior-preserving cleanup removes only legacy helpers with no production references and the
tests that existed solely for those superseded helpers. Acceptance requires the focused main-chain
suite to remain green, the repository-wide result to have no new failures, and one live record to
complete through self-annotation, both second-layer experts, both aggregator modes, and every
read-only validator. This smoke run is operational verification only; it is not an accuracy
experiment and does not use Gold labels.
