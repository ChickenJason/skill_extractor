# trf

TRF 是独立业务模块，内部提供 `Offline`、`Target` 和 `Full` 三种运行模式。这些是同一模块的运行方式，不表示项目中三个业务模块之间存在固定流水线。

## 功能

- `offline/`：从中立 demonstration 集构建 313 条正式语料，计算主候选与 context-only 候选，并使用固定 BERT 为 demonstration 分配伪 TRF；
- `target/`：对独立目标句或 leave-one-out 目标生成 embedding，检索 16 个候选实例，执行两轮开放 TRF 抽取，再以目标原文和推断 TRF 预测精确 Skill 字符跨度；
- `orchestration/`：在一个 run-id 下组织本模块的 offline 与 target 模式并联合校验。

TRF 可接收一个可选反馈 JSON/JSONL。当前实现只验证 `schema_version`、`dataset_id`、`record_id`、`source_sha256` 并锁定文件哈希，不把反馈用于候选或特征计算。

## 输入与输出

默认 demonstration manifest 是 `data/processed/demonstrations/v1/manifest.json`。独立目标输入是 JSON 数组，每项至少含 `idx` 和 `sentence`；目标会被规范化为中立记录身份。

`Full` 输出位于 `output/trf/<run-id>/`：

- `offline/corpus/records.jsonl`：正式 TRF 语料；
- `offline/candidates/selected_trfs.json`：全局候选 TRF；
- `offline/demonstrations/pseudo_trfs.jsonl`：demonstration 伪 TRF；
- `target/targets/records.jsonl`：规范化目标；
- `target/retrieval/records.jsonl`：通用候选实例记录，可作为判别器 candidates；
- `target/parsed/records.jsonl`：通用特征记录，可作为判别器可选 features；
- `target/prediction/prompts.jsonl`、`raw.jsonl`、`records.jsonl`：TRF 专家的最终目标 Skill 预测及严格审计链；
- `target/prediction/failures.jsonl`：终态预测失败，包含原句、模型输出、结构化 provisional extraction、每次请求哈希和校验错误；
- `target/prediction/results.jsonl`：每个目标一条结果信封，明确 exact、recovered、provisional、validation_failed 或阻断状态及坐标空间；
- `target/audit/validation_issues.jsonl`：TRF 抽取与最终 Skill 预测按目标合并后的校验问题账本；
- 父目录和两个子目录各自的 `manifest.json`：来源与输出哈希合同。

## 运行

只运行离线部分：

```powershell
.\scripts\trf\run.ps1 `
  -Mode Offline `
  -RunId trf-offline-v1 `
  -PythonExecutable .\.venv-trf\Scripts\python.exe
```

如本地没有固定 BERT 快照，可显式添加 `-AllowModelDownload`。

对 100 条句子运行本模块全部模式：

```powershell
.\scripts\trf\run.ps1 `
  -Mode Full `
  -RunId synthetic100-trf-v1 `
  -TargetMode independent `
  -Input .\data\raw\synthetic_jd_sentences_100.json `
  -AllowNetwork `
  -ConfirmFullRun `
  -PythonExecutable .\.venv-trf\Scripts\python.exe
```

`Target` 模式要求显式传入由本模块锁定的 generated config：

```powershell
.\scripts\trf\run.ps1 `
  -Mode Target `
  -RunId synthetic100-target-v2 `
  -GeneratedConfigPath .\output\trf\synthetic100-trf-v1\target-config.generated.json `
  -TargetMode independent `
  -Input .\data\raw\synthetic_jd_sentences_100.json `
  -AllowNetwork `
  -ConfirmFullRun `
  -PythonExecutable .\.venv-trf\Scripts\python.exe
```

Target/Full 中断后用 `-Resume`；只重试失败在线记录时同时使用 `-RetryFailed`；可用 `-ReuseEmbeddingsFrom <run-id>` 复用严格兼容的 embedding。严格校验和两次 repair 均失败后，仅当模型句与源句等长且所有差异都属于白名单 CP1252 标点归一化时，系统才把标签投影回源句，并以 `prediction_recovered_after_validation_failure` 标记为 `needs_review`。无法安全投影但标签结构有效的结果通过结果信封按模型原句坐标保留。TRF v4 把 TRF 抽取和最终跨度预测的终态校验问题按目标去重；不超过 `floor(N × 3%)` 时整体完成，超过时为 `partial`。TRF 抽取校验失败会生成 `needs_review` 空 TRF 上下文并继续预测；网络、缺失响应或运行错误仍使运行保持 partial。Offline 运行不可覆盖，失败后应换一个新 run-id 重跑。

只读联合验证：

```powershell
.\.venv-trf\Scripts\python.exe .\code\trf\orchestration\validator.py `
  --config .\config\trf.json `
  --run-id synthetic100-trf-v1
```

## 非目标

当前不实现反馈驱动的迭代更新，不调用实例判别器，也不融合另一专家的预测。`candidates`、`features` 与独立的 `skill-prediction-v1` 都是中立接口，消费者由调用方选择。
