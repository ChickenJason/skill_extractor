# 可复现性说明

## 环境

- 操作系统：Windows 11 `10.0.26200`。
- Python：3.12.7。
- Python 可执行文件：`D:\python\anaconda\python.exe`。
- 合同复验环境：项目内 `.venv-trf`，Python 3.10.14；自我标注器核心36项通过。
- 项目版本：0.6.0。
- 项目不是 Git 工作树，因此用输入、源运行和派生输出 SHA256 固定实验状态。

## 关键 SHA256

| 文件 | SHA256 |
|---|---|
| `data/raw/skill_sentences.json` | `0c426247d64c23db7228edb20400565161e671f715abe1351fe609ed813d27a8` |
| `output/prompts/prompts.json` | `2220fffba8d55d93f4ae97fa6132e9ad723b7502d2254a4349e5245dc9c246ba` |
| 归档 `raw/responses.jsonl` | `eef358cb9bda4228d1955ca37b529c0b4fccc0793caa0e6cfedacf5ef62dcd88` |
| 归档 `manifest.json` | `49d9058d2220b7f7c9dad2ec783ee111dbe98b67ffc0e42da2c3e7adf3cfb04b` |
| v0.5.0 派生 `aggregated/consensus.jsonl` | `2e070aead1eb521e2386b69279303b3bc1529b2b3190a79feed7cef3023217ba` |
| v0.5.0 派生 `selected/annotations.json` | `c1b60b02ac5d95e11b66066466ea6f484a008788d64f4a8bc0906c6da42ee8ac` |
| v0.5.1 派生 `aggregated/consensus.jsonl` | `d7f99c800e3b7b416195d21115319108cfcc2f70d7144afbe15e290c6be8bf6b` |
| v0.5.1 派生 `selected/annotations.json` | `e89ab84c8ee840186fc000529405ddea867dccc95e87190a5288818e073cb44c` |
| v0.6.0 `config/pipeline.json` | `7bd3beeadAE82f0f7b9c39cc020bd8f685bb8f556f5d7eaaa8dc26b5f274c42e` |
| v0.6.0 `hard_span_matcher.py` | `3dee2201b823d9d88fb6ee76922782e0cbd99959be2a68664c0d8f330b94a0d5` |
| v0.6.0 派生 `aggregated/consensus.jsonl` | `09e6a02333b2ced02533d57fdf59d1323f27d00e74100a97bf10e25721b86668` |
| v0.6.0 派生 `aggregation_audit.jsonl` | `9ffa2ff8f82a8f7905451efda40687c9935d530806d7ba2475c5511a69bd9763` |
| v0.6.0 派生 `selected/annotations.json` | `2f4c1c9281d564a43f7af612e0ca9601c1b9703f2e10f10df476f711563a2c9b` |

## 复现命令

在项目根目录运行：

```powershell
.\scripts\run-tests.ps1

.\scripts\run-reaggregate.ps1 `
  -SourceRunRoot .\archive\self-annotation-history\inline-v4-2-full-20260821-001 `
  -RunId span-xmlc-base5-replay-majority3-anchor-v1
```

第二条命令不会访问网络。不同 run-id 会改变 manifest 时间戳，但在相同 Python/JSON
序列化环境和输入下，`consensus.jsonl` 与 `annotations.json` 应保持相同 SHA256。

## 必检不变量

1. 派生 manifest 的 `run_kind` 为 `offline_reaggregation`。
2. manifest 顶层和 compatibility 内均为 `network_called:false`。
3. `source_unchanged:true`。
4. parsed 样本数为 1630，四个句级 JSONL 输出各 326 行。
5. 状态计数为 accepted 225、negative 88、unsolved 13、abstained 0。
6. 接受来源为 exact 491、hard match 6；正式 annotations 为313条。
7. 共尝试9个 hard-match family：6个接受、3个锚点覆盖失败、4个最高票并列。
8. 相较 v0.5.1，状态迁移必须为192个 accepted 保持、88个 negative 保持、33个
   unsolved→accepted、13个 unsolved 保持。
9. `config/trf.json` 引用 v0.6.0 多数票重放；`config/trf_target.json` 仅作为完整 runner 模板，
   实际来源由父 run 动态写入并锁定 SHA256。

## 从空输出重建 Self-Annotator 与完整 TRF（2026-08-31）

```powershell
Set-Location "E:\research\extractor\SkillSentence-Processor-0822"
$env:DASHSCOPE_API_KEY = "<API Key>"
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

固定 BERT 已在当前机器缓存；其他机器缺少缓存时向完整 runner 增加
`-AllowModelDownload`。自标注重放不访问网络；目标阶段对326条数据执行 embedding 与
两轮 chat，实际云端响应不承诺字节复现。

本轮清理后的 `output/` 只含五个 `.gitkeep`。历史输出被送入 Windows 回收站而非覆盖，
需要恢复旧实验时应从回收站恢复；正常重建应使用新的不可变 run-id 或上面的空目录固定 ID。

---

# TRF 可复现性（最终 v4 / 见证 v5）

## 环境与模型

- Python 3.10.14；NumPy 1.26.4；scikit-learn 1.6.1；Torch 2.5.1+cpu；
  Transformers 4.57.1。
- BERT revision：`cd5ef92a9fb2f889e972770a36d4ed042daf221e`。
- `model.safetensors`：435,755,784 bytes；SHA256
  `1d8bdcee6021e2c25f0325e84889b61c2eb26b843eef5659c247af138d64f050`。
- 最终 v4 为纯离线运行，manifest 中 `network_called:false`、`source_unchanged:true`。

## 核心产物 SHA256

| 文件 | SHA256 |
|---|---|
| `corpus/records.jsonl` | `893a6fe3b54d01431d7771f7fdf4b52dcb650ac4f9172b9be52684989065358b` |
| `corpus/excluded.jsonl` | `7cb26496daee8fdf0acdf2f8b25ca97a480cd25aca3833a5ebeced139f49d1fe` |
| `candidates/selected_trfs.json` | `c556081658f6dba8c34011ef59aec564cc78415e4043f4d11fcb7f0c18f96e3f` |
| `demonstrations/pseudo_trfs.jsonl` | `f4e84d42f5692d6685141a1f8d5dc0e3f5a4b762bed26b6778e6575176322292` |
| `audit/semantic_review_sample.jsonl` | `726d8ba47dca5fcfee0302677866ba52b937ef211f87f7c535b28f5466a3ccc3` |

源 decisions SHA256 为
`d7f99c800e3b7b416195d21115319108cfcc2f70d7144afbe15e290c6be8bf6b`；源 manifest 和
aggregation audit 也由每个 TRF manifest 独立固定和复核。

## 复现命令

首次下载只能显式授权，并须使用空 run-id：

```powershell
.\scripts\run-trf-offline.ps1 `
  -SelfAnnotatorRunId <self-annotator-run-id> `
  -RunId <new-run-id> `
  -AllowModelDownload `
  -PythonExecutable $python
```

缓存完成后离线复现与验证：

```powershell
.\scripts\run-trf-offline.ps1 `
  -SelfAnnotatorRunId <self-annotator-run-id> `
  -RunId <new-offline-run-id> `
  -PythonExecutable $python

.\.venv-trf\Scripts\python.exe .\code\trf\ValidateTRFRun.py `
  --config .\config\trf.json `
  --run-id <new-offline-run-id>
```

v4/v5 已证明 manifest 列出的全部 19 个非 manifest 产物 SHA256 可逐项相等。manifest
自身包含时间和 run-id，因此不要求字节相同。

---

# 目标句 TRF 提取器可复现性

## 固定输入与模型合同

- Top-20 TRF manifest：`3267d314bb96a8102dd4780f5604f72239f18d13f6c365eeeec9fed53228bd95`。
- corpus：`893a6fe3b54d01431d7771f7fdf4b52dcb650ac4f9172b9be52684989065358b`。
- 主候选：`356b993a1b900265be5470a7341e54203d8da57373222a80174c753bb94492b3`（20 项）。
- 伪 TRF：`8f4ae0d191a8655bfd877be6b47c7b1d9d814bc518ff2db0ed2b0acaa7430f47`。
- decisions：`d7f99c800e3b7b416195d21115319108cfcc2f70d7144afbe15e290c6be8bf6b`。
- Embedding：`qwen3.7-text-embedding`、1024 维、batch 20。
- Chat：`qwen3.7-plus-2026-05-26`、thinking false、temperature 0、JSON object。

## 建议复现顺序

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
.\.venv-trf\Scripts\python.exe -m compileall -q `
  .\code\trf_target .\code\common\qwen_client.py
.\.venv-trf\Scripts\python.exe -m unittest discover -s .\tests -v

.\scripts\run-target-trf.ps1 `
  -RunId <new-smoke-id> `
  -Mode leave-one-out `
  -Limit 10 `
  -AllowNetwork `
  -PythonExecutable .\.venv-trf\Scripts\python.exe

.\.venv-trf\Scripts\python.exe .\code\trf_target\ValidateTargetTRFRun.py `
  --config .\config\trf_target.json `
  --run-id <completed-smoke-id>
```

## 可重复性边界

同一缓存向量和 raw response 快照应产生完全相同的 targets、retrieval、prompts、parsed、
summary 与人工审计样本。Embedding/chat 的真实云端重新调用没有本地 revision，因此不承诺
字节一致；manifest 用 model ID、调用时间、response id、usage、向量/响应文件 SHA256
固定实际证据。所有凭据只从环境变量解析，不写入 manifest、日志或产物。

---

# 完整 TRF 一键重跑可复现性

## 一条命令

```powershell
Set-Location E:\research\extractor\SkillSentence-Processor-0822
conda activate extractor
$python = (Get-Command python).Source
$env:DASHSCOPE_API_KEY = "<API Key>"
$env:DASHSCOPE_BASE_URL = "<OpenAI-compatible endpoint>"

.\scripts\rerun-all-trf.ps1 `
  -SelfAnnotatorRunId <self-annotator-run-id> `
  -RunId <new-parent-run-id> `
  -Mode leave-one-out `
  -AllowNetwork `
  -ConfirmFullRun `
  -PythonExecutable $python
```

没有固定 BERT revision 缓存时增加 `-AllowModelDownload`。完成后执行：

```powershell
& $python .\code\trf_full\ValidateFullTRFRun.py `
  --run-id <parent-run-id>
```

## 输出映射

- `output/trf_full_runs/<id>/manifest.json`：父运行合同。
- `output/trf_full_runs/<id>/target-config.generated.json`：锁定本次离线产物的目标配置。
- `output/trf_runs/<id>-offline/`：本次离线三阶段。
- `output/trf_target_runs/<id>-target/`：本次目标检索与两轮抽取。

离线结果在固定环境/权重下要求字节级确定；真实 Qwen 响应不承诺重新调用后字节相同，
但缓存后的检索、Prompt、解析与 summary 必须可确定性重建。父验证器以两个子 manifest
SHA256 固定一次实际运行。
