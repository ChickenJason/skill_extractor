# Skill Sentence Processor

本项目包含三个底层业务模块和一个最终聚合器。它们共享 `code/common/` 中的 I/O、哈希、Qwen 客户端和数据合同；业务包之间不互相 import，聚合器只读取第二层发布的中立文件。

| 模块 | 主要职责 | 配置 | 正式入口 | 输出根目录 |
|---|---|---|---|---|
| `self_annotator` | 对句子进行五次独立自我标注，解析 Skill 字符跨度并聚合共识 | `config/self_annotator.json` | `scripts/self_annotator/run.ps1` | `output/self_annotator/<run-id>/` |
| `trf` | 构建离线 TRF、提取目标开放 TRF，并据此预测目标 Skill 精确跨度 | `config/trf.json` | `scripts/trf/run.ps1` | `output/trf/<run-id>/` |
| `instance_discriminator` | 对候选实例评分并做 Eligible 筛选，再据保留示例预测目标 Skill 精确跨度 | `config/instance_discriminator.json` | `scripts/instance_discriminator/run.ps1` | `output/instance_discriminator/<run-id>/` |
| `aggregator` | 以相同 pipeline 运行证据版和专家结果版，发布最终 Skill 精确跨度 | `config/aggregator.json` | `scripts/aggregator/run.ps1` | `output/aggregator/<run-id>/` |

各模块的详细说明分别位于：

- [自我标注器](code/self_annotator/README.md)
- [TRF](code/trf/README.md)
- [实例判别器](code/instance_discriminator/README.md)
- [最终聚合器](code/aggregator/README.md)

此外，[第二层并行调度](code/second_layer/README.md)可在不改变底层业务模块的前提下，冻结 TRF 检索结果，并行运行 TRF 与不含目标 Features 的示例判别器，最终生成供聚合器读取的上下文包。

## 环境

项目固定使用 Python 3.10。推荐使用已有环境：

```powershell
cd E:\research\extractor\SkillSentence-Processor-0822
$python = ".\.venv-trf\Scripts\python.exe"
& $python --version
```

需要调用 Qwen 时设置：

```powershell
$env:DASHSCOPE_API_KEY = "你的密钥"
$env:DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
```

## 100 条目标句

`data/raw/synthetic_jd_sentences_100.json` 是待识别的 100 条目标句，不替换已经固化的 326 条 demonstration。

对这 100 条句子运行自我标注：

```powershell
.\scripts\self_annotator\run.ps1 `
  -RunId synthetic100-self-v1 `
  -Input .\data\raw\synthetic_jd_sentences_100.json `
  -PythonExecutable .\.venv-trf\Scripts\python.exe
```

独立运行完整 TRF 模式：

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

TRF 可额外接收中立反馈文件；当前版本只校验并锁定它，不让反馈改变算法：

```powershell
.\scripts\trf\run.ps1 `
  -Mode Full `
  -RunId synthetic100-trf-feedback-v1 `
  -TargetMode independent `
  -Input .\data\raw\synthetic_jd_sentences_100.json `
  -Feedback .\path\to\feedback.jsonl `
  -AllowNetwork `
  -ConfirmFullRun `
  -PythonExecutable .\.venv-trf\Scripts\python.exe
```

实例判别器不推断上游目录。显式传入 targets、candidates，并按需传入 features：

```powershell
.\scripts\instance_discriminator\run.ps1 `
  -RunId synthetic100-disc-v1 `
  -Targets .\output\trf\synthetic100-trf-v1\target\targets\records.jsonl `
  -Candidates .\output\trf\synthetic100-trf-v1\target\retrieval\records.jsonl `
  -Features .\output\trf\synthetic100-trf-v1\target\parsed\records.jsonl `
  -AllowNetwork `
  -ConfirmFullRun `
  -PythonExecutable .\.venv-trf\Scripts\python.exe
```

若不传 `-Features`，判别器只使用目标文本以及候选实例中的文本、跨度和可靠度信息；manifest 会记录 `feature_context: absent`，不会伪造 TRF。

两个模块分别在自己的 `prediction/records.jsonl` 发布目标原句坐标下的 `skill-prediction-v1`，并在 `prediction/results.jsonl` 为每个目标发布 `skill-prediction-result-v1`。预测严格使用 `<skill>` 内联标签还原 end-exclusive 字符跨度；非法结构最多进行两次受约束修复。修复耗尽后，安全的一对一 CP1252 异常标点归一化仍可投影回原句；其他结构可解析的输出以 `model_sentence` 坐标作为 provisional 结果进入下游，不会伪装成正式目标坐标。TRF v4 和示例判别器 v5 将各自所有模型阶段的终态校验问题按目标去重：比例不超过 3% 时运行完成，超过 3% 时标记 `partial`。网络、缺失响应、运行时和完整性错误不使用该额度。

并行运行第二层（示例判别器固定不接收 Features）：

```powershell
.\scripts\second_layer\run.ps1 `
  -RunId synthetic100-layer2-v1 `
  -TargetMode independent `
  -Input .\data\raw\synthetic_jd_sentences_100.json `
  -Limit 3 `
  -AllowNetwork `
  -PythonExecutable .\.venv-trf\Scripts\python.exe
```

父运行位于 `output/second_layer/synthetic100-layer2-v1/`，两个子运行分别为 `synthetic100-layer2-v1-trf` 和 `synthetic100-layer2-v1-examples`。只有两个子 validator 均通过时，才会发布 `context/records.jsonl`；它提供双专家上下文，但不生成最终技能跨度预测。

最终聚合器提供严格匹配的两种模式。证据版禁止传入专家路径；专家结果版优先读取两路 `prediction/results.jsonl`，并兼容旧的 `skill-prediction-v1`：

```powershell
.\scripts\aggregator\run.ps1 `
  -Mode EvidenceOnly `
  -RunId synthetic100-agg-evidence-v1 `
  -ContextManifest .\output\second_layer\synthetic100-layer2-v1\manifest.json `
  -ContextRecords .\output\second_layer\synthetic100-layer2-v1\context\records.jsonl `
  -PrepareOnly

.\scripts\aggregator\run.ps1 `
  -Mode WithExpertResults `
  -RunId synthetic100-agg-experts-v1 `
  -ContextManifest .\output\second_layer\synthetic100-layer2-v1\manifest.json `
  -ContextRecords .\output\second_layer\synthetic100-layer2-v1\context\records.jsonl `
  -TrfPredictions <trf-prediction-results> `
  -ExemplarPredictions <exemplar-prediction-results> `
  -PrepareOnly
```

确认 `inputs/records.jsonl` 与 `prompts/records.jsonl` 后，用同一个 run-id 加 `-Resume -AllowNetwork -ConfirmFullRun` 继续。两个版本的 Gold 对比只通过 `scripts/aggregator/evaluate.ps1` 离线执行，Gold 不进入运行输入或 Prompt。

## 中立数据合同

可交换记录携带以下字段：

- `schema_version`：记录类型和版本；
- `dataset_id`、`record_id`：跨数据集主键；
- `source_sha256`：该记录的来源哈希；
- `idx`：只用于稳定排序和兼容旧数据。

leave-one-out 和泄漏检查使用 `(dataset_id, record_id)`。完整输入文件及输出文件的 SHA256 由各次运行的 `manifest.json` 锁定。

正式 demonstration 位于 `data/processed/demonstrations/v1/`，共 326 条：225 accepted、88 negative、13 unsolved、0 abstained；其 manifest 同时锁定 `records.jsonl` 和 `audit.jsonl`。

## 输出与恢复

运行目录不可覆盖。中断或可恢复失败后，应使用同一个 run-id 并添加 `-Resume`；仅重试失败的在线记录时再添加 `-RetryFailed`。已经 completed 的运行保持不可变，需要重跑时使用新的 run-id。

每个模块都提供只读 validator。常用命令见对应模块 README。

## 离线验证

```powershell
.\scripts\run-tests.ps1
```

等价的 unittest 命令：

```powershell
python -m unittest discover -s tests -t . -v
```

测试不访问网络。固定 BERT 权重或 Python 3.10 不可用时，真实模型 smoke test 会明确跳过；其余 fixture 测试仍需通过。

## 历史记录

旧版设计文档和实验日志保留在 `research/`。其中出现的旧 `output/*_runs`、`archive/*`、旧包名和旧脚本仅描述当时实验；这些运行产物已经清理，不再是可执行接口。
