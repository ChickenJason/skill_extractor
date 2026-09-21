# instance_discriminator

实例判别器是独立业务模块。它读取目标、候选实例和可选特征，要求模型为每个候选给出 helpfulness、supporting/contrastive/irrelevant 角色和受控原因代码，不生成自由文本 `reason`。随后只做 Eligible 筛选：保留 helpfulness 大于 3 且角色不是 `irrelevant` 的全部候选，并维持候选原始顺序；不再排序或执行数量、角色上限截断。最后依据保留的示例预测目标 Skill 精确字符跨度。

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

`instance-discriminator-v5` 允许终态模型输出校验问题涉及的目标数不超过 `floor(目标数 × 3%)`。候选判别与最终 Skill 预测的问题按目标合并使用同一个配额；例如 100 个目标最多允许 3 个，326 个最多允许 9 个。判别校验失败会生成 `needs_review` 空示例上下文并继续最终预测。安全的一对一异常标点归一化可投影回原句；无法安全对齐但结构可解析的输出按模型原句坐标保留在 `prediction/results.jsonl`。网络、缺失响应、运行时和完整性错误不属于可容忍校验问题。

v4 及更早版本的 completed/partial 运行目录保持不变，不能用 v5 配置恢复；应用新容差时必须使用新的 run-id。

候选判别响应只允许 `demo_idx`、`helpfulness_score`、`role` 和 `reason_codes`；`reason` 属于未知字段并会触发严格修复。若聊天模型返回重复/遗漏的 `demo_idx`、非法角色、额外 JSON 字段或其他可解析但不合规的结构化结果，运行器会在当前记录内自动执行最多两次受约束 repair。repair prompt 会携带原始校验错误、16 个必需 ID 及每个候选允许的角色，但不会放宽严格解析规则，也不会静默改写模型判断。原始失败响应、repair 请求哈希和最终响应保存在同一条 raw 记录的 `repair` 审计链中。

从旧实现创建的 partial 运行可在第一次 `-Resume -RetryFailed` 时进行一次仅限 repair 相关文件的实现升级；升级前后哈希和原因记录在 manifest 的 `implementation_upgrades` 中。其他实现变化仍会拒绝恢复。

## 输出

运行目录为 `output/instance_discriminator/<run-id>/`，主要包含：

- `source_snapshot.json`：输入路径、SHA256 和 feature context；
- `candidates/records.jsonl`：规范化候选；
- `prompts/records.jsonl`：泄漏受控 Prompt；
- `raw/responses.jsonl`：原始判别响应；
- `parsed/judgments.jsonl`：严格解析的逐候选判别；
- `selected/records.jsonl`：硬门控后的候选及 needs_review 状态；
- `prediction/prompts.jsonl`、`raw.jsonl`、`records.jsonl`：只使用所选示例、示例技能跨度、helpfulness 和角色产生的目标 Skill 预测；
- `prediction/failures.jsonl`：无法安全发布的终态失败及其结构化 provisional extraction；
- `prediction/results.jsonl`：覆盖每个目标的结果信封及明确坐标空间；
- `audit/validation_issues.jsonl`：判别与最终预测按目标去重后的校验问题账本；
- `audit/summary.json`、`audit/manual_review.jsonl`：诊断与复核样本；
- `manifest.json`：运行状态及完整哈希账本。

当问题目标数在上述 3% 配额内时，Manifest 为 `completed`；超过配额时保留全部产物并标记 `partial`。正式 `prediction/records.jsonl` 可以少于目标总数，但 `prediction/results.jsonl` 必须每目标一条。具体配额、结果类型和实际问题数记录在 `audit/summary.json`。

只读验证：

```powershell
.\.venv-trf\Scripts\python.exe .\code\instance_discriminator\validator.py `
  --config .\config\instance_discriminator.json `
  --run-id synthetic100-disc-v1
```

## 非目标

本模块不定位或猜测某个 TRF 运行目录，不要求 features 必须由 TRF 产生，也不融合 TRF 专家的独立 Skill 预测。它只处理调用方显式提供的中立数据。
