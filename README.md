# SkillSentence Processor v0.6.0

本项目从招聘语句中提取原文技能片段。Qwen 仍使用原提示词，以 `<skill>...</skill>`
做内联标记；活动聚合链路已经改为字符跨度级 XMLC，不再使用 Token/BIO，也不再追加
自适应采样。

```text
固定五次采样
→ Has-skill 三票多数门控
→ Exact-span 三票接受
→ Residual overlap-family
→ 最高频span锚点覆盖硬匹配
→ accepted / unsolved / negative / abstained
```

自我标注器不生成标准化技能名称或 taxonomy。正式 span 输出仍使用原文字符串跨度，
偏移、票数和聚合证据保存在 decisions 与 audit 文件中。独立的 TRF 离线流水线会使用
固定 BERT 生成论文式伪 TRF，但不会修改自我标注结果。

## 目录

```text
code/       核心 Python 流程
config/     Prompt 与固定五采样配置
data/raw/   326 条原始语句
output/     Prompt 和按 run-id 隔离的运行结果
scripts/    PowerShell 入口
tests/      离线测试
research/   实验计划、运行日志与结果
archive/    历史运行产物，只读保留
```

## 环境

```powershell
Set-Location E:\research\extractor\SkillSentence-Processor-0822
conda activate extractor
$python = (Get-Command python).Source
```

真实调用前设置阿里云 Model Studio 凭据：

```powershell
$env:DASHSCOPE_API_KEY = "<API Key>"
$env:DASHSCOPE_BASE_URL = "https://<WorkspaceId>.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
```

## 运行

```powershell
# Prompt 校验与全部离线测试
.\scripts\run-tests.ps1

# 生成完整 Prompt；-Limit 3 时写入 preview.json
.\scripts\run-prompts.ps1
.\scripts\run-prompts.ps1 -Limit 3

# 固定五次采样，然后 Parse → Span XMLC → Select
.\scripts\run-base.ps1 -RunId <run-id>

# 测试、Prompt 与固定五采样流水线
.\scripts\run-pipeline.ps1 -RunId <run-id>

# 从历史基础响应重聚合；不读取补采样响应，也不访问网络
.\scripts\run-reaggregate.ps1 `
  -SourceRunRoot .\archive\self-annotation-history\inline-v4-2-full-20260821-001 `
  -RunId span-xmlc-base5-replay-majority3-anchor-v1 `
  -PythonExecutable $python
```

真实采样入口中断后可用相同 run-id 增加 `-Resume`。离线重聚合要求目标目录为空，
避免覆盖已有结果。

## TRF 离线流水线

TRF 三阶段通过 `SelfAnnotatorRunId` 读取 `output/runs/<SelfAnnotatorRunId>`：构建313条正式语料、提取
20 个全局候选及 20 个 context-only 候选，并为 225 条正类分配两组 BERT Top-5；
88 条负类的两组 TRF 均为空。

输入与输出 ID 严格分离：`SelfAnnotatorRunId` 只选择自我标注输入；`RunId` 只命名
TRF 输出目录。`config/trf.json` 仅保存 `output/runs` 根路径，不再写死任何自我标注 run-id。

首次缓存固定 revision 的 `bert-base-cased` 时显式允许下载：

```powershell
.\scripts\run-trf-offline.ps1 `
  -SelfAnnotatorRunId <自我标注-run-id> `
  -RunId <新的-run-id> `
  -AllowModelDownload
```

缓存完成后的普通运行不访问网络：

```powershell
.\scripts\run-trf-offline.ps1 `
  -SelfAnnotatorRunId <自我标注-run-id> `
  -RunId <新的-run-id>

& $python .\code\trf\ValidateTRFRun.py `
  --self-annotator-run-id <自我标注-run-id> `
  --run-id <run-id>
```

每个 run-id 只能创建一次，失败后也必须换新 id。TRF 运行保存在
`output/trf_runs/<run-id>/`；人工语义检查样本位于
`audit/semantic_review_sample.jsonl`。算法验收完成不等于语义验收通过。

## 目标句 TRF 提取器

`config/trf_target.json` 只是完整 runner 使用的模板，不能直接执行。完整 runner 会在父 run
中生成 `target-config.generated.json`，以本次离线子 run 的实际路径和 SHA256 替换模板来源。
目标阶段使用 `qwen3.7-text-embedding` 执行 K=50 → k=16 检索，再用同一两轮消息历史先判断
`Skill` 类型、后开放生成目标 TRF。正式 Prompt 只使用主伪 TRF。

如需单独恢复或调试已经生成的目标配置，必须显式传入它：

```powershell
.\scripts\run-target-trf.ps1 `
  -ConfigPath .\output\trf_full_runs\<parent-id>\target-config.generated.json `
  -RunId <parent-id>-target `
  -Mode leave-one-out `
  -Resume `
  -AllowNetwork `
  -PythonExecutable $python
```

解析失败只有同时增加 `-RetryFailed` 才会重新请求，成功目标不会重调。

运行产物保存在 `output/trf_target_runs/<run-id>/`。`audit/manual_review.jsonl` 的语义状态
在人工检查前固定为 `pending_manual_review`；leave-one-out 指标只是伪标签一致性，不是
Gold 准确率。

## 一键重跑完整 TRF

`rerun-all-trf.ps1` 会按顺序执行：离线 corpus → 全局候选 → BERT Top-5 → 目标
Embedding 检索 → 两轮目标 TRF → 联合验证。父 run 自动创建：

- `<RunId>-offline`：本次重新生成的离线 TRF；
- `<RunId>-target`：只引用上述离线子 run 实际 SHA256 的目标抽取；
- `output/trf_full_runs/<RunId>/`：父 manifest、动态锁定配置和联合验证结果。

从空 `output/` 开始，先执行零 API 自标注重聚合，再运行完整 leave-one-out：

```powershell
Set-Location "E:\research\extractor\SkillSentence-Processor-0822"

$env:DASHSCOPE_API_KEY = "<你的 API Key>"
$env:DASHSCOPE_BASE_URL = "https://<WorkspaceId>.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"

conda activate extractor
$python = (Get-Command python).Source
$selfRun = "span-xmlc-base5-replay-majority3-anchor-v1"
$trfRun = "trf-complete-majority3-anchor-v1"

.\scripts\run-reaggregate.ps1 `
  -SourceRunRoot ".\archive\self-annotation-history\inline-v4-2-full-20260821-001" `
  -RunId $selfRun `
  -PythonExecutable $python

.\scripts\rerun-all-trf.ps1 `
  -SelfAnnotatorRunId $selfRun `
  -RunId $trfRun `
  -Mode leave-one-out `
  -AllowNetwork `
  -ConfirmFullRun `
  -PythonExecutable $python

& $python .\code\trf_full\ValidateFullTRFRun.py --run-id $trfRun
```

当前机器已经缓存固定 BERT 权重。新机器没有缓存时，再增加 `-AllowModelDownload`。
更稳妥的方式是先准备全部向量、
检索和 Prompt，再恢复执行 652 次 chat：

```powershell
.\scripts\rerun-all-trf.ps1 `
  -SelfAnnotatorRunId $selfRun `
  -RunId trf-complete-majority3-anchor-v1 `
  -Mode leave-one-out `
  -PrepareOnly `
  -AllowNetwork `
  -PythonExecutable $python

.\scripts\rerun-all-trf.ps1 `
  -SelfAnnotatorRunId $selfRun `
  -RunId trf-complete-majority3-anchor-v1 `
  -Mode leave-one-out `
  -Resume `
  -AllowNetwork `
  -ConfirmFullRun `
  -PythonExecutable $python
```

最终联合验证：

```powershell
& $python .\code\trf_full\ValidateFullTRFRun.py `
  --run-id trf-complete-majority3-anchor-v1
```

父 run、两个派生子 run 都不可覆盖。离线子 run 失败时必须换新的父 `RunId`；目标子 run
中断时用父 `RunId` 增加 `-Resume`，解析失败还需显式增加 `-RetryFailed`。

## 状态

- `accepted`：存在性门控通过，所有正式跨度均已解决且互不冲突；
- `negative`：至少四张有效 `no_skill` 票；
- `unsolved`：存在性未达三票且未判负，或跨度族仍无法稳定裁决；
- `abstained`：五次结构完整，但有效解析样本少于三次。

只有 `accepted` 与 `negative` 进入 `selected/annotations.json`。

Residual family support 为3至5时进入硬匹配。系统先选择最高频跨度作为锚点，再按锚点
字符覆盖率和锚点 `word_units` 动态阈值逐一核对其他成员。最高票并列时依次使用最小
覆盖率、平均覆盖率、跨度紧凑度和字符位置选择确定性 winner。

## 运行输出

每个运行保存在 `output/runs/<run-id>/`：

```text
manifest.json
raw/
parsed/samples.jsonl
aggregated/consensus.jsonl
aggregated/aggregation_audit.jsonl
aggregated/uncertainty_audit.jsonl
selected/annotations.json
selected/decisions.jsonl
```

离线派生运行的 `raw/` 只保存源路径和 SHA256，不复制历史响应。完整技术规则见
`SELF_ANNOTATOR.md`，设计与迁移依据见 `SELF_ANNOTATOR_XMLC_REFACTOR_PLAN.md`。
