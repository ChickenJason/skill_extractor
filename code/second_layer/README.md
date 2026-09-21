# Concurrent Scheduling（第二层并行调度）

`second_layer` 是 TRF Full 与示例判别器上方的独立调度层，不改变两个业务模块的代码、配置或数据合同。

`second-layer-parallel-v1` 是这一 Concurrent Scheduling 实现的兼容 pipeline 名；语义上的 `schedule_type` 为 `concurrent`。新命令可使用 `scripts/second_layer/run_concurrent.ps1`，原 `run.ps1` 继续保留并保持兼容。

它先调用 TRF Full 的 PrepareOnly，随后把目标记录和 16 候选检索记录冻结到父运行目录。TRF Resume 与示例判别器再作为两个独立子进程同时运行。示例判别器只接收冻结的 targets 和 candidates，命令中不传 `--features`，因此其 `feature_context` 必须是 `absent`。

两个子运行均完成并通过原模块 validator 后，调度层才发布 `context/records.jsonl`。该文件是一份 `second-layer-context-v1` 上下文包，供未来聚合器读取；本层不输出最终技能跨度预测。

## 运行

先冻结共享输入：

```powershell
.\scripts\second_layer\run.ps1 `
  -RunId synthetic100-layer2-v1 `
  -Input .\data\raw\synthetic_jd_sentences_100.json `
  -Limit 3 `
  -PrepareOnly `
  -AllowNetwork `
  -PythonExecutable .\.venv-trf\Scripts\python.exe
```

继续并行在线阶段：

```powershell
.\scripts\second_layer\run.ps1 `
  -RunId synthetic100-layer2-v1 `
  -Input .\data\raw\synthetic_jd_sentences_100.json `
  -Limit 3 `
  -AllowNetwork `
  -Resume `
  -PythonExecutable .\.venv-trf\Scripts\python.exe
```

若某个子模块以 partial 状态结束，可在恢复命令中增加 `-RetryFailed`。完整验证命令：

```powershell
.\.venv-trf\Scripts\python.exe .\code\second_layer\validator.py `
  --config .\config\second_layer.json `
  --run-id synthetic100-layer2-v1
```

## 输出

- `shared/targets.jsonl`、`shared/candidates.jsonl`：不可变共享输入快照；
- `logs/`：准备阶段和两个并行子进程的独立日志；
- `child-validation.json`：两个官方 validator 的结果；
- `context/records.jsonl`：仅在两路完整且验证通过后发布；
- `context/summary.json`、`manifest.json`：父运行摘要、阶段、进程、网络和 SHA256 审计。

派生子运行 ID 固定为 `<run-id>-trf` 与 `<run-id>-examples`。

基于已完成 concurrent 运行执行单轮协作反思，请见 [collaborative_reflection/README.md](collaborative_reflection/README.md)。
