# Skill Sentence Processor

本项目包含三个并列的业务模块。它们共享 `code/common/` 中的 I/O、哈希、Qwen 客户端和数据合同，但业务包之间不互相 import，也没有固定的跨模块执行顺序。

| 模块 | 主要职责 | 配置 | 正式入口 | 输出根目录 |
|---|---|---|---|---|
| `self_annotator` | 对句子进行五次独立自我标注，解析 Skill 字符跨度并聚合共识 | `config/self_annotator.json` | `scripts/self_annotator/run.ps1` | `output/self_annotator/<run-id>/` |
| `trf` | 构建离线 TRF、为目标句检索候选实例并提取开放 TRF | `config/trf.json` | `scripts/trf/run.ps1` | `output/trf/<run-id>/` |
| `instance_discriminator` | 对显式传入的候选实例逐个评分、分配角色并执行确定性硬门控 | `config/instance_discriminator.json` | `scripts/instance_discriminator/run.ps1` | `output/instance_discriminator/<run-id>/` |

三个模块的详细说明分别位于：

- [自我标注器](code/self_annotator/README.md)
- [TRF](code/trf/README.md)
- [实例判别器](code/instance_discriminator/README.md)

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
