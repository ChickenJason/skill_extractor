# TRF--Exemplar Synchronous Interaction

This package creates an interaction context for a future aggregator. It never predicts or emits target Skill spans and it does not import any `collaborative_reflection*` package.

The parent runner accepts a completed `second-layer-parallel-v1` run and supports four modes: `none`, `trf_to_exemplar`, `exemplar_to_trf`, and `bidirectional`. Every active branch in a round reads the same frozen `R^(t-1), H^(t-1)`. Proposals are applied only after the complete round validates.

```powershell
.\scripts\second_layer\run_bidirectional_interaction.ps1 `
  -RunId interaction-none-smoke `
  -BaseConcurrentRunId skill-concurrent-v1 `
  -Mode none

.\scripts\second_layer\run_bidirectional_interaction.ps1 `
  -RunId interaction-prepared-smoke `
  -BaseConcurrentRunId skill-concurrent-v1 `
  -Mode bidirectional -Limit 3 -PrepareOnly
```

Online work requires `DASHSCOPE_API_KEY`, `DASHSCOPE_BASE_URL`, and `-AllowNetwork`. An unlimited online run additionally requires `-ConfirmFullRun`.

A recoverable credentials, transport, parse, or target failure leaves the parent `partial` and publishes no `context/records.jsonl`. Resume the same code/config with `-Resume`; add `-RetryFailed` to retry only the latest failed raw record. A source, identity, configuration, or hash violation marks the run `failed` and requires a new run ID.

Gate runs are separate and do not accept `Limit`:

```powershell
.\scripts\second_layer\run_bidirectional_interaction_gate.ps1 `
  -RunId interaction-gate-smoke3 `
  -GateSpec config/gates/bidirectional_interaction_smoke3_v1.json `
  -AllowNetwork
```

For the low-cost daily smoke test, run two representative targets for one round:

```powershell
.\scripts\second_layer\run_bidirectional_interaction_gate.ps1 `
  -RunId interaction-smoke-lite `
  -GateSpec config/gates/bidirectional_interaction_smoke_lite_v1.json `
  -MaxRounds 1 -AllowNetwork
```

The lite gate checks structural completion, ready contexts, an active-TRF result,
a selected-exemplar result, and bounded calls. It intentionally omits frozen
score, role, selection, and change-from-baseline semantic oracles.

Run artifacts live below `output/second_layer/<run-id>/source|rounds|context|audit|logs`. Raw branch logs are append-only. Prompts, schemas, raw responses, parsed proposals, states, contexts, gates, and resources are covered by SHA256/byte ledgers.
