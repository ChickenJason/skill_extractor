# Failure and edge-case record

## Implementation-time failures

1. A plain `unittest discover` invocation used the wrong top-level directory and allowed
   `tests/common` to shadow `code/common`. The supported repository command uses `-t .`.
2. Ruff is not installed in the active Python environment. Syntax was verified with `compileall`,
   the PowerShell parser, targeted tests, and the repository test script.
3. A Windows `py_compile` command did not expand `*.py`; `compileall` was used instead.
4. Two existing full-suite tests fail because a stratified-gate base artifact lacks `idx=13`.
   This is outside the aggregator scope and remains unchanged.

## Deliberate fail-closed cases

- Evidence-only receives either expert path.
- Context manifest is incomplete or its records hash/byte count differs.
- Context identity, sentence, source hash, schema, status, selected count, gate order, helpfulness,
  or exact spans are invalid.
- Expert schema, branch, identity, sentence, source hash, status, has-skill value, or exact spans are
  invalid.
- Prompt size exceeds the configured limit.
- Resume changes sources, configuration, target selection, or implementation hashes.
- Output, prompt, prediction, failure, summary, review, compatibility, or SHA256 ledger is tampered.

Only terminal model-output validation issues that exhaust all strict repairs are eligible for the
fixed 3% tolerance. Events are retained individually but counted once per target across stages.
Missing raw responses, provider/runtime failures, and invalid failure audits are not eligible.

`idx=216` exposed two misdecoded CP1252 em-dash bytes (`U+0097`). Both experts normalized them to
`U+2014` on all three attempts. The new recovery contract retains those exact-span extractions as
`needs_review`; arbitrary spelling, whitespace, insertion, deletion, or length-changing edits are
not projected into target offsets. When their tag structure is parseable, they remain downstream
results with `coordinate_space=model_sentence`; exact-span metrics treat them as missing formal
predictions.

## 2026-09-21 cleanup-smoke failures

1. A broad ad-hoc AST scan encountered a non-UTF Python source outside the active main-chain scope.
   The scan was narrowed to decodable `code/` and `tests/` files; no additional safe deletion was
   inferred from that scan.
2. `cleanup-smoke-second-20260921` used the shell's Python 3.12.7 and was rejected by the intentional
   TRF Python 3.10 runtime gate before any network call. The failed run and log were retained. A new
   run, `cleanup-smoke-second-20260921-py310`, used `.venv-trf` Python 3.10.14 and completed.
3. The full offline suite still has the same two unrelated stratified-gate failures because its base
   artifact is empty/missing `idx=13`; this cleanup did not alter those experimental fixtures.
