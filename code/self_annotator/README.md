# self_annotator

自我标注器是独立业务模块。它对每条句子请求五个独立结构化响应，解析 Skill 字符跨度，执行存在性投票、跨度家族聚合和 hard span match，最后导出正式标注与完整决策。

## 输入与输出

输入是 JSON 数组，每项至少包含唯一整数 `idx` 和非空 `sentence`。运行时根据输入哈希生成 `dataset_id`，输出记录包含 `schema_version`、`dataset_id`、`record_id`、`source_sha256` 和兼容字段 `idx`。

运行目录为 `output/self_annotator/<run-id>/`，主要产物包括：

- `prompts/records.json`：本次 Prompt 快照；
- `raw/responses.jsonl`：五样本原始响应；
- `parsed/samples.jsonl`：严格解析结果；
- `aggregated/consensus.jsonl`：跨度共识；
- `selected/annotations.json`：accepted/negative 正式标注；
- `selected/decisions.jsonl`：包含 accepted、negative、unsolved、abstained 的完整决策；
- `manifest.json`：配置、输入、实现和状态合同。

## 运行

```powershell
.\scripts\self_annotator\run.ps1 `
  -RunId synthetic100-self-v1 `
  -Input .\data\raw\synthetic_jd_sentences_100.json `
  -PythonExecutable .\.venv-trf\Scripts\python.exe
```

小规模检查可添加 `-Limit 3`。中断后使用 `-Resume`；如需遇到首个失败即停止，可添加 `-FailFast`。

只读验证：

```powershell
.\.venv-trf\Scripts\python.exe .\code\self_annotator\validator.py `
  --config .\config\self_annotator.json `
  --run-id synthetic100-self-v1
```

## 零 API 重聚合

仅当某次旧运行仍保留 `manifest.json` 和 `raw/responses.jsonl` 时，才可以重新解析和聚合：

```powershell
.\scripts\self_annotator\run.ps1 `
  -RunId replay-v1 `
  -ReaggregateFrom .\path\to\source-run `
  -PythonExecutable .\.venv-trf\Scripts\python.exe
```

本轮已清理的历史 raw responses 无法再进行零 API 重聚合。

## 非目标

本模块不运行 TRF、不筛选候选实例，也不生成所谓“最终预测器”结果。它导出的中立数据可以由任何模块显式读取，但代码中不绑定消费者。
