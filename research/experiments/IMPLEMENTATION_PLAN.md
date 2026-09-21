# Final aggregator implementation plan

## Implemented components

1. `code/aggregator/common.py`: frozen configuration, source snapshots, run paths, compatibility,
   implementation hashes, and manifest lifecycle.
2. `code/aggregator/pipeline.py`: strict context normalization, expert alignment, review rules, and
   prompt construction shared by both modes.
3. `code/aggregator/runner.py`: prepare, online prediction, resume, failed retry, diagnostics, and
   terminal status handling.
4. `code/aggregator/validator.py`: read-only reconstruction of inputs, prompts, parser output,
   failures, review samples, compatibility, and the complete SHA256 ledger.
5. `code/aggregator/evaluation.py`: Gold-only offline metrics, single-expert baselines, per-sentence
   changes, contribution criterion, and paired bootstrap intervals.
6. `scripts/aggregator/*.ps1`, `config/aggregator.json`, and module documentation.
7. Unit and integration tests in `tests/aggregator/`, plus the peer-import architecture gate.

## Completion checks

- [x] Evidence-only mode rejects all expert paths.
- [x] Both modes share identical normalized base evidence.
- [x] Expert results are advisory and may be ignored or replaced.
- [x] One or both unavailable experts continue from base evidence with branch-specific review reasons.
- [x] Expert conflict and expert `needs_review` do not independently force final review.
- [x] Strict source reconstruction and two bounded repairs are reused.
- [x] Safe post-repair character recovery publishes exact source spans as `needs_review`.
- [x] Unsafe but structurally valid terminal outputs remain downstream-visible in model coordinates.
- [x] Every target has a result envelope and validation issues are merged by target across stages.
- [x] TRF/exemplar upstream validation failures produce reviewable empty contexts and continue.
- [x] Prepare, resume, retry-failed, 3% policy, and tamper detection are covered.
- [x] Offline evaluator parses the 100-item Gold set.
- [ ] Execute the two network-backed 100-item runs after a completed second-layer context exists.
- [ ] Publish Gold metrics only after both completed runs validate.

## 2026-09-21 behavior-preserving cleanup

- [x] Remove the superseded shared completion helper; retain the unified result-envelope policy.
- [x] Remove the old TRF and exemplar success-only raw parsers; retain terminal-result rebuilding.
- [x] Remove duplicate local failure-quota helpers; retain the shared 3% calculation and boundary
  coverage.
- [x] Remove two tests that covered only deleted compatibility helpers.
- [x] Run focused and repository-wide offline tests, compile the code, and execute a one-record live
  smoke through all three layers and both aggregator modes.
