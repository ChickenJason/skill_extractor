# Run log

All commands ran from `E:\research\extractor\SkillSentence-Processor-0822` on 2026-09-20.
No network-backed model run was executed.

| Command or check | Status | Result |
|---|---:|---|
| Read common IO/contracts, strict prediction, discriminator runner/diagnostics, second-layer assembler/validator, scripts, and architecture tests | 0 | Established existing contracts and lifecycle patterns |
| Inspect representative TRF, selected-exemplar, and prediction records | 0 | Confirmed real public artifact shapes |
| `python -m compileall -q code\aggregator` | 0 | New module compiled |
| `python -m unittest discover -s tests -p "test_*.py"` | 1 | Incorrect discovery root shadowed `code/common`; replaced with repository-supported `-t .` invocation |
| Inspect Gold, `pyproject.toml`, README test command, and test script | 0 | Confirmed 100-item Gold and correct test entry point |
| `python -m ruff check ...; python -m py_compile code\aggregator\*.py` | 1 | Ruff unavailable and Windows did not expand the wildcard; compileall used instead |
| `.venv-trf\Scripts\python.exe -B -m unittest tests.aggregator.test_pipeline tests.common.test_architecture -v` | 0 | Python 3.10.14; final targeted run: 13 tests passed without generating bytecode caches |
| Parse all three PowerShell entry points with the PowerShell AST parser | 0 | Syntax valid |
| Parse `synthetic_jd_sentences_100_gold.json` with the evaluator | 0 | 100 records and 213 exact spans |
| `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run-tests.ps1` | 1 | 174 tests ran; 172 passed, 2 unrelated existing stratified-gate tests failed because base artifacts lack idx 13 |
| Read-only normalization of real 326-item TRF and discriminator outputs | 0 | Evidence: 326 inputs/1833 examples; expert-aware: 325 executable; both missing at idx 216; one additional missing TRF expert |
| Contract audit for peer imports, forbidden exemplar evidence fields, Gold isolation, and generated caches | 0 | No peer imports, no forbidden exemplar evidence fields, Gold only in diagnostics/evaluation, no remaining aggregator caches |
| Inspect `idx=216` TRF/exemplar raw, parsed, selected, and failure records | 0 | Both branches completed upstream evidence but exhausted strict repairs after `U+0097` was normalized to `U+2014` |
| Read `aris-experiment-bridge/SKILL.md` and shared prediction/consumer contracts | 0 | Defined strict-plus-retention implementation boundary |
| `.venv-trf\Scripts\python.exe -B -m unittest tests.common.test_skill_prediction -v` | 0 | 6 shared prediction contract tests passed |
| Offline recovery of both existing `idx=216` terminal responses | 0 | Both produced exact source spans and audited substitutions at positions 91 and 123 |
| `.venv-trf\Scripts\python.exe -B -m unittest tests.common.test_skill_prediction tests.trf.test_target tests.instance_discriminator.test_pipeline tests.aggregator.test_pipeline tests.common.test_architecture -v` | 0 | 55 cross-consumer tests passed |
| `.venv-trf\Scripts\python.exe -B -m unittest tests.common.test_skill_prediction -v; git diff --check` | 0 | 6 shared-contract tests passed; no whitespace errors, only existing line-ending warnings |
| Python 3.10 in-memory compile of seven changed Python files and scoped Git status | 0 | Syntax valid; no generated aggregator caches or unexpected task files |
| `git diff --check` | 0 | No whitespace errors; only repository line-ending warnings |

The integration tests used temporary directories and fake deterministic clients. They exercised
strict repair, resume, retry-failed, expert replacement, and validator tamper detection without
leaving experiment output in the repository.

## 2026-09-21 — 3% retention policy implementation

All commands ran from `E:\research\extractor\SkillSentence-Processor-0822`; no network-backed
model call was made.

| Command or check | Status | Result |
|---|---:|---|
| Read `aris-experiment-bridge/SKILL.md`; inspect Git status, configs, runners, validators, diagnostics, tests, and current failure artifacts with `Get-Content`/`rg` | 0 | Confirmed dirty user worktree was preserved and located all 1% and fail-closed paths |
| `python -m pytest tests/common/test_skill_prediction.py tests/aggregator/test_pipeline.py tests/instance_discriminator/test_pipeline.py tests/trf/test_target.py -q` | 0 | Baseline before edits: 54 passed |
| `python -m py_compile` over the changed shared, aggregator, TRF, and discriminator Python modules | 0 | Syntax valid after the first implementation pass |
| Same four-suite pytest command after the first pass | 1 | 46 passed, 8 failed; exposed stale final-prediction prompts after an upstream retry plus expected legacy assertions |
| Same three consumer suites after prompt-hash invalidation fix | 1 | 45 passed, 3 legacy expectation failures remained |
| Same four-suite pytest command after updating contracts and tests | 0 | 56 passed |
| `python -m pytest -q` | 1 | 178 passed, 1 skipped, 2 pre-existing stratified-gate failures because base artifacts are empty/missing `idx=13` |
| Re-run the four-suite pytest command after audit hardening | 0 | 56 passed |
| Re-run `python -m pytest -q` | 1 | Stable result: 178 passed, 1 skipped, the same 2 unrelated stratified-gate fixture failures |
| `python -m compileall -q code/common code/aggregator code/trf/target code/instance_discriminator tests/common tests/aggregator tests/trf tests/instance_discriminator` | 0 | Changed Python modules and tests compile successfully |
| `git diff --check` | 0 | No whitespace errors; only existing LF-to-CRLF warnings |

Observed implementation behavior: result envelopes cover every target; recovered and provisional
terminal validation issues are retained and counted; validation events from multiple stages are
deduplicated by target; 3/100 and 9/326 pass while 4/100 and 10/326 are partial; runtime failures
remain blocking. Upstream validation failures create empty review contexts and continue to final
prediction. No accuracy result was produced.

## 2026-09-21 — behavior-preserving cleanup and live smoke

All commands ran from `E:\research\extractor\SkillSentence-Processor-0822`. Provider credentials
were supplied only to transient processes and were not written to repository artifacts.

| Command or check | Status | Result |
|---|---:|---|
| Static reference search with `rg` | 0 | Identified five superseded, production-unreferenced helpers and two helper-only tests |
| Broad ad-hoc Python AST scan | 1 | Encountered a non-UTF Python file outside the active code scope; rerun was limited to decodable `code/` and `tests/` sources |
| `python -m pytest tests/common/test_skill_prediction.py tests/aggregator/test_pipeline.py tests/instance_discriminator/test_pipeline.py tests/trf/test_target.py -q` | 0 | 54 passed after cleanup |
| `python -m pytest -q` | 1 | 176 passed, 1 skipped, same 2 pre-existing stratified-gate fixture failures (`idx=13` missing) |
| `python -m compileall -q code` plus dependency import check | 0 | Syntax valid; PyTorch and Transformers import successfully |
| Self-annotator run `cleanup-smoke-self-20260921`, limit 1 | 0 | Five live responses aggregated; 1/1 formal; validator `valid` |
| Second-layer run `cleanup-smoke-second-20260921`, limit 1, default Python | 1 | Preserved failure: TRF correctly rejected Python 3.12.7 before any network call |
| Second-layer run `cleanup-smoke-second-20260921-py310`, limit 1, `.venv-trf` | 0 | Both experts completed, 1/1 context ready, both child predictions exact |
| Aggregator runs `cleanup-smoke-agg-evidence-20260921` and `cleanup-smoke-agg-experts-20260921`, limit 1 | 0 | Both completed with one exact result, zero repairs, zero validation issues |
| Read-only validators for the self-annotator, second layer, and both aggregators | 0 | All four returned `valid` |
| Credential-prefix scan over output, research, config, code, and scripts | 0 | No provider key prefix found |
