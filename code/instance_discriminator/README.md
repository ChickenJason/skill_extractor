# instance_discriminator

实例判别器是独立业务模块。它读取目标、候选实例和可选特征，要求模型为每个候选给出 helpfulness、supporting/contrastive/irrelevant 角色和审计原因，再通过固定阈值与数量上限做确定性筛选。

## 输入合同

入口显式要求：

- `targets`：目标 JSON/JSONL；
- `candidates`：每个目标恰好 16 个候选实例；
- `features`：可选的外部特征 JSON/JSONL。

三个文件通过 `(dataset_id, record_id)` 对齐，记录同时携带 `schema_version`、`source_sha256` 和兼容字段 `idx`。候选泄漏检查也使用完整身份，因此两个不同数据集使用相同 `idx` 不会被误判。

有 features 时，Prompt 使用其中的 entity types 和 TRF 等特征；无 features 时，只使用目标文本以及候选文本、跨度、相似度和存在性可靠度。缺失特征不会被伪造，manifest 明确记录 `feature_context`。

## 运行

有外部特征：

```powershell
.\scripts\instance_discriminator\run.ps1 `
  -RunId synthetic100-disc-v1 `
  -Targets .\path\to\targets.jsonl `
  -Candidates .\path\to\candidates.jsonl `
  -Features .\path\to\features.jsonl `
  -AllowNetwork `
  -ConfirmFullRun `
  -PythonExecutable .\.venv-trf\Scripts\python.exe
```

无外部特征时省略 `-Features`。若只想检查输入、候选合同和 Prompt，不调用聊天模型，可添加 `-PrepareOnly`。

中断后使用 `-Resume`；只重试失败记录时同时添加 `-RetryFailed`。

## 输出

运行目录为 `output/instance_discriminator/<run-id>/`，主要包含：

- `source_snapshot.json`：输入路径、SHA256 和 feature context；
- `candidates/records.jsonl`：规范化候选；
- `prompts/records.jsonl`：泄漏受控 Prompt；
- `raw/responses.jsonl`：原始判别响应；
- `parsed/judgments.jsonl`：严格解析的逐候选判别；
- `selected/records.jsonl`：硬门控后的候选及 needs_review 状态；
- `audit/summary.json`、`audit/manual_review.jsonl`：诊断与复核样本；
- `manifest.json`：运行状态及完整哈希账本。

只读验证：

```powershell
.\.venv-trf\Scripts\python.exe .\code\instance_discriminator\validator.py `
  --config .\config\instance_discriminator.json `
  --run-id synthetic100-disc-v1
```

## 非目标

本模块不定位或猜测某个 TRF 运行目录，不要求 features 必须由 TRF 产生，也不实现最终 Skill 跨度预测器。它只处理调用方显式提供的中立数据。
