# TRF 流水线结构、输入输出与完整示例

## 1. 路径约定

本文档对应当前 **Top-20 TRF** 实现。

项目绝对根目录为：

```text
E:\research\extractor\SkillSentence-Processor-0822
```

下文使用 `<ROOT>` 表示该目录。例如：

```text
<ROOT>\config\trf.json
```

对应的绝对路径是：

```text
E:\research\extractor\SkillSentence-Processor-0822\config\trf.json
```

本文档自身位于：

```text
<ROOT>\TRF_PIPELINE_STRUCTURE.md
```

## 2. 当前 TRF 总体流程

```text
<ROOT>\output\runs\span-xmlc-base5-replay-support4-v1\selected\decisions.jsonl
                              │
                              ▼
                    离线 TRF 三阶段
          ① BuildTRFCorpus：构建 280 条正式语料
          ② ExtractTRFCandidates：提取全局 Top-20
          ③ AssignPseudoTRFs：为 demonstration 分配 Top-5
                              │
                              ▼
                      目标句 TRF 阶段
          ④ 构建 326 条 leave-one-out 目标句
          ⑤ Qwen Embedding 检索 K=50 → k=16
          ⑥ Qwen 第一轮判断 Skill 类型
          ⑦ Qwen 第二轮开放生成目标 TRF
          ⑧ 解析、诊断、人工审计与只读验证
```

四个不同的数量不要混淆：

| 名称                  | 当前数量 | 作用                                      |
| --------------------- | -------: | ----------------------------------------- |
| 全局 TRF 候选库       |       20 | 整个数据集共享的候选词库                  |
| demonstration 伪 TRF  |        5 | 每条 accepted 示例从 20 项中选择 5 项     |
| 初始检索邻居          |       50 | 目标句按余弦相似度取出的初始邻居          |
| Prompt demonstrations |       16 | 从 50 条邻居中最终选入 Qwen Prompt 的示例 |

最终 Qwen 生成的目标 TRF 是开放式结果，不限制在全局 Top-20 中，数量也不固定。

## 3. 代码、配置和脚本路径

### 3.1 配置文件

| 根目录相对路径             | 绝对路径                          | 用途                                                     |
| -------------------------- | --------------------------------- | -------------------------------------------------------- |
| `config\trf.json`        |  `<ROOT>\config\trf.json`       | 自标注 runs 根目录、候选 Top-20、BERT revision、Top-5、seed 等配置 |
| `config\trf_target.json` | `<ROOT>\config\trf_target.json` | 目标来源、Embedding、K=50/k=16、Qwen Chat、输出目录配置  |
| `environment.yml`        | `<ROOT>\environment.yml`        | Python 和核心依赖版本合同                                |
| `pyproject.toml`         | `<ROOT>\pyproject.toml`         | 项目 Python 配置                                         |

当前关键配置为：

```text
config\trf.json:
  source.runs_root = output/runs
  candidates.max_trfs = 20
  model.top_k = 5

config\trf_target.json:
  candidates.expected_main_trfs = 20
  retrieval.nearest_neighbors = 50
  retrieval.demonstrations = 16
```

### 3.2 离线 TRF 代码

所有路径均从 `<ROOT>` 开始：

| 根目录相对路径                       | 用途                                        |
| ------------------------------------ | ------------------------------------------- |
| `code\trf\RunTRFOffline.py`        | 顺序执行三个离线阶段                        |
| `code\trf\BuildTRFCorpus.py`       | 构建正式 corpus 和 excluded 审计            |
| `code\trf\ExtractTRFCandidates.py` | unigram、MI、rho、方向门控和 Top-20         |
| `code\trf\AssignPseudoTRFs.py`     | 固定 BERT、token offset、欧氏距离和 Top-5   |
| `code\trf\ValidateTRFRun.py`       | 只读验证一个已完成离线 run                  |
| `code\trf\common.py`               | 离线 run 路径、manifest、hash、不可覆盖合同 |
| `code\trf\__init__.py`             | Python 子包声明                             |

例如离线入口的绝对路径为：

```text
E:\research\extractor\SkillSentence-Processor-0822\code\trf\RunTRFOffline.py
```

### 3.3 目标句 TRF 代码

| 根目录相对路径                              | 用途                                      |
| ------------------------------------------- | ----------------------------------------- |
| `code\trf_target\RunTargetTRF.py`         | 目标阶段入口、状态机、Resume、RetryFailed |
| `code\trf_target\pipeline.py`             | 目标输入、检索、Prompt、解析的纯逻辑      |
| `code\trf_target\online.py`               | Embedding 缓存和两轮 Qwen 调用            |
| `code\trf_target\diagnostics.py`          | summary、一致性指标和人工样本             |
| `code\trf_target\ValidateTargetTRFRun.py` | 只读重算检索、Prompt、解析和 hash         |
| `code\trf_target\common.py`               | 来源锁定、manifest、输出路径和 SHA256     |
| `code\trf_target\__init__.py`             | Python 子包声明                           |

### 3.4 完整流水线编排代码

| 根目录相对路径                          | 用途                               |
| --------------------------------------- | ---------------------------------- |
| `code\trf_full\RunFullTRF.py`         | 创建父 run，串联离线和目标阶段     |
| `code\trf_full\ValidateFullTRFRun.py` | 联合验证父 run 和两个子 run        |
| `code\trf_full\common.py`             | 父 run 的 manifest、阶段和输出合同 |
| `code\trf_full\__init__.py`           | Python 子包声明                    |

### 3.5 公共代码

| 根目录相对路径                 | 用途                                             |
| ------------------------------ | ------------------------------------------------ |
| `code\common\io_utils.py`    | JSON/JSONL、原子写入、SHA256、路径解析           |
| `code\common\qwen_client.py` | Qwen Chat 和 Embedding 客户端、重试和 usage 审计 |
| `code\common\__init__.py`    | Python 子包声明                                  |

### 3.6 PowerShell 入口

| 根目录相对路径                  | 用途                                   |
| ------------------------------- | -------------------------------------- |
| `scripts\run-trf-offline.ps1` | 只运行离线三个阶段                     |
| `scripts\run-target-trf.ps1`  | 只运行目标 Embedding、检索和 Qwen 阶段 |
| `scripts\rerun-all-trf.ps1`   | 从离线 corpus 到目标 TRF 的完整重跑    |

### 3.7 测试文件

| 根目录相对路径                        | 用途                                               |
| ------------------------------------- | -------------------------------------------------- |
| `tests\test_trf_pipeline.py`        | corpus、tokenizer、MI、Top-20、BERT 和离线集成测试 |
| `tests\test_trf_target_pipeline.py` | 输入、检索、Prompt、解析、恢复和目标集成测试       |
| `tests\test_trf_full_pipeline.py`   | 父 run、动态配置和联合验证测试                     |
| `tests\test_qwen_client.py`         | Qwen Chat/Embedding 客户端合同测试                 |

## 4. 权威输入路径

### 4.1 原始数据

```text
<ROOT>\data\raw\skill_sentences.json
```

绝对路径：

```text
E:\research\extractor\SkillSentence-Processor-0822\data\raw\skill_sentences.json
```

### 4.2 自我标注来源 run

当前权威来源目录：

```text
<ROOT>\output\runs\span-xmlc-base5-replay-support4-v1
```

离线 TRF 会锁定以下三个源文件：

```text
<ROOT>\output\runs\span-xmlc-base5-replay-support4-v1\manifest.json
<ROOT>\output\runs\span-xmlc-base5-replay-support4-v1\selected\decisions.jsonl
<ROOT>\output\runs\span-xmlc-base5-replay-support4-v1\aggregated\aggregation_audit.jsonl
```

其中正式数据输入是：

```text
<ROOT>\output\runs\span-xmlc-base5-replay-support4-v1\selected\decisions.jsonl
```

共 326 条：

```text
accepted   192
negative    88
unsolved    46
abstained    0
```

## 5. 离线阶段一：构建 TRF corpus

实现文件：

```text
<ROOT>\code\trf\BuildTRFCorpus.py
```

### 输入

```text
<ROOT>\output\runs\span-xmlc-base5-replay-support4-v1\selected\decisions.jsonl
```

### 处理规则

- `accepted` 进入正类 corpus，`pseudo_types=["Skill"]`。
- `negative` 进入负类 corpus，`pseudo_types=[]`。
- `unsolved/abstained` 写入 excluded 审计，不进入 demonstration pool。
- accepted 的 `existence_score = has_skill_votes / valid_votes`。
- negative 的 `existence_score = no_skill_votes / valid_votes`。
- 每个 span 必须满足 `sentence[start:end] == text`，否则阶段失败。

### 输出

当前 Top-20 离线 run 根目录：

```text
<ROOT>\output\trf_runs\trf-offline-bert-base-cased-top20-v1
```

阶段一输出：

```text
<ROOT>\output\trf_runs\trf-offline-bert-base-cased-top20-v1\corpus\records.jsonl
<ROOT>\output\trf_runs\trf-offline-bert-base-cased-top20-v1\corpus\excluded.jsonl
<ROOT>\output\trf_runs\trf-offline-bert-base-cased-top20-v1\corpus\summary.json
<ROOT>\output\trf_runs\trf-offline-bert-base-cased-top20-v1\audit\manual_review_sample.jsonl
```

正式 corpus 为 280 条：192 accepted + 88 negative。excluded 为 46 条。

### 正类示例：idx=98

```json
{
  "idx": 98,
  "sentence": "Ideally you should have undergone training in IT support and have at least one year of experience working as an IT Support Assistant .",
  "status": "accepted",
  "pseudo_types": ["Skill"],
  "skill_spans": [
    {
      "text": "IT support",
      "start": 46,
      "end": 56,
      "exact_votes": 5,
      "accepted_by": "exact_vote"
    }
  ],
  "existence_score": 1.0
}
```

### 负类示例：idx=14

```json
{
  "idx": 14,
  "sentence": "You can walk on stairs",
  "status": "negative",
  "pseudo_types": [],
  "skill_spans": [],
  "existence_score": 1.0
}
```

## 6. 离线阶段二：提取全局 Top-20

实现文件：

```text
<ROOT>\code\trf\ExtractTRFCandidates.py
```

### 输入

```text
<ROOT>\output\trf_runs\trf-offline-bert-base-cased-top20-v1\corpus\records.jsonl
```

### 处理过程

1. 对文本执行 NFKC 规范化。
2. 删除 `<ORGANIZATION>`、`<LOCATION>`、`<NAME>`、`<CONTACT>` 等占位符。
3. unigram 分词，并保留 `C++`、`C#`、`CI/CD` 等技术形式。
4. 删除 scikit-learn 英文 stopwords 和包含数字的候选。
5. 统计正类、负类 raw count 和 document frequency。
6. 计算词与 Skill 类型之间的二值互信息。
7. 应用论文频率比：

$$
\frac{C_{negative}(w)}{C_{positive}(w)} \le 3
$$

8. 应用方向门控：

$$
\frac{df_{positive}(w)}{192} > \frac{df_{negative}(w)}{88}
$$

9. 按 MI 降序、正类 document frequency 降序、字符串字典序取前 20 项。

### 当前主 Top-20

```text
English, skills, management, projects, sales,
communication, process, product, software, tools,
leadership, technical, years, basic, design,
independently, strategic, taking, user, development
```

### 候选指标示例：communication

```json
{
  "text": "communication",
  "raw_count": {"positive": 8, "negative": 0},
  "document_frequency": {"positive": 8, "negative": 0},
  "mi": 0.010971410091421128,
  "frequency_ratio_negative_to_positive": 0.0,
  "normalized_document_rate": {
    "positive": 0.041666666666666664,
    "negative": 0.0
  },
  "direction_gate_passed": true,
  "rank": 6
}
```

### 输出路径

```text
<ROOT>\output\trf_runs\trf-offline-bert-base-cased-top20-v1\candidates\selected_trfs.json
<ROOT>\output\trf_runs\trf-offline-bert-base-cased-top20-v1\candidates\trfs.json
<ROOT>\output\trf_runs\trf-offline-bert-base-cased-top20-v1\candidates\context_only_trfs.json
<ROOT>\output\trf_runs\trf-offline-bert-base-cased-top20-v1\candidates\main_all_metrics.jsonl
<ROOT>\output\trf_runs\trf-offline-bert-base-cased-top20-v1\candidates\main_paper_formula.jsonl
<ROOT>\output\trf_runs\trf-offline-bert-base-cased-top20-v1\candidates\main_directional.jsonl
<ROOT>\output\trf_runs\trf-offline-bert-base-cased-top20-v1\candidates\context_only_all_metrics.jsonl
<ROOT>\output\trf_runs\trf-offline-bert-base-cased-top20-v1\candidates\context_only_paper_formula.jsonl
<ROOT>\output\trf_runs\trf-offline-bert-base-cased-top20-v1\candidates\context_only_directional.jsonl
<ROOT>\output\trf_runs\trf-offline-bert-base-cased-top20-v1\candidates\summary.json
```

`context_only` 会先把 accepted 技能跨度替换为等长空格再统计，仅用于诊断。正式 Prompt 不使用 context-only TRF。

## 7. 离线阶段三：BERT demonstration Top-5

实现文件：

```text
<ROOT>\code\trf\AssignPseudoTRFs.py
```

### 输入

```text
<ROOT>\output\trf_runs\trf-offline-bert-base-cased-top20-v1\corpus\records.jsonl
<ROOT>\output\trf_runs\trf-offline-bert-base-cased-top20-v1\candidates\selected_trfs.json
```

模型合同：

```text
bert-base-cased
revision: cd5ef92a9fb2f889e972770a36d4ed042daf221e
hidden size: 768
device: CPU
```

### 处理过程

1. 每个候选单独输入 BERT。
2. 对非特殊 subword 的最后一层向量 mean pooling。
3. 整句输入 BERT，通过 offset mapping 得到词级表示。
4. 计算候选与句中最近词的欧氏距离：

$$
d(w,x_i)=\min_j\|\mathbf e_w-\mathbf e_{ij}\|_2
$$

5. 按距离、候选全局 rank、候选字符串排序。
6. accepted 选择 Top-5；negative 输出空集。

### idx=98 示例

```json
{
  "idx": 98,
  "status": "accepted",
  "trfs": [
    {"text": "product", "rank": 1, "distance": 9.24436760},
    {"text": "process", "rank": 2, "distance": 9.35550308},
    {"text": "taking", "rank": 3, "distance": 9.81190586},
    {"text": "communication", "rank": 4, "distance": 10.04402542},
    {"text": "projects", "rank": 5, "distance": 10.49428654}
  ]
}
```

该例的最近 token 都落在 `Ideally`，说明算法和 offset 合同正确，但语义最近邻仍可能偏向通用词，需要人工审阅。

### idx=14 负类示例

```json
{
  "idx": 14,
  "status": "negative",
  "trfs": [],
  "context_only_trfs": []
}
```

### 输出路径

```text
<ROOT>\output\trf_runs\trf-offline-bert-base-cased-top20-v1\demonstrations\pseudo_trfs.jsonl
<ROOT>\output\trf_runs\trf-offline-bert-base-cased-top20-v1\demonstrations\summary.json
<ROOT>\output\trf_runs\trf-offline-bert-base-cased-top20-v1\demonstrations\model_files.json
<ROOT>\output\trf_runs\trf-offline-bert-base-cased-top20-v1\audit\semantic_review_sample.jsonl
```

离线 run 的总审计文件：

```text
<ROOT>\output\trf_runs\trf-offline-bert-base-cased-top20-v1\manifest.json
<ROOT>\output\trf_runs\trf-offline-bert-base-cased-top20-v1\source\snapshot.json
```

## 8. 目标阶段一：构建目标集

实现文件：

```text
<ROOT>\code\trf_target\pipeline.py
```

当前目标 run 根目录：

```text
<ROOT>\output\trf_target_runs\trf-top20-target-v1
```

### leave-one-out 模式

输入：

```text
<ROOT>\output\runs\span-xmlc-base5-replay-support4-v1\selected\decisions.jsonl
```

输出 326 条目标句：

```text
<ROOT>\output\trf_target_runs\trf-top20-target-v1\targets\records.jsonl
```

- accepted/negative 目标位于 280 条 demonstration pool 中，检索前排除自身。
- 46 条 excluded 也作为目标推理，但它们不在 demonstration pool 中，不需要排除自身。

### independent 模式

默认输入路径：

```text
<ROOT>\data\raw\target_sentences.json
```

严格格式：

```json
[
  {"idx": 1, "sentence": "..."}
]
```

## 9. 目标阶段二：Qwen Embedding 和 K=50→k=16 检索

相关实现文件：

```text
<ROOT>\code\trf_target\online.py
<ROOT>\code\trf_target\pipeline.py
<ROOT>\code\common\qwen_client.py
```

模型：

```text
qwen3.7-text-embedding
dimensions: 1024
batch size: 20
```

### 输入

```text
280 条 demonstration 句子
326 条目标句
每条 demonstration 的主 Top-5 伪 TRF
```

### Embedding 输出

```text
<ROOT>\output\trf_target_runs\trf-top20-target-v1\embeddings\demonstrations.jsonl
<ROOT>\output\trf_target_runs\trf-top20-target-v1\embeddings\targets.jsonl
```

每条记录保存句子 SHA256、模型、1024 维向量、批次审计和复用来源。本次 Top-20 run 的向量复用自：

```text
<ROOT>\output\trf_target_runs\trf-top50-complete-v1-target
```

### 检索规则

1. 所有向量 L2 归一化。
2. 按未舍入余弦相似度降序、demo idx 升序取 K=50。
3. 在 K=50 内按 existence score 降序、相似度降序、idx 升序取 k=16。
4. accepted 和 negative 均可进入 Prompt，不设置类别配额。

### idx=98 示例

`idx=98` 本身属于 demonstration pool，因此先排除自己：

```text
leave_one_out = true
self_excluded = true
```

最终选择 16 条，其中 13 accepted、3 negative。第一条是：

```json
{
  "selected_rank": 1,
  "demo_idx": 99,
  "status": "accepted",
  "existence_score": 1.0,
  "similarity": 0.6396655008,
  "trfs": ["years", "tools", "development", "strategic", "basic"]
}
```

检索完整输出：

```text
<ROOT>\output\trf_target_runs\trf-top20-target-v1\retrieval\records.jsonl
```

每条记录同时保存 K=50 完整邻居、最终 16 条和 self-exclusion 审计。

## 10. 目标阶段三：两轮 Qwen Chat

Prompt 构建：

```text
<ROOT>\code\trf_target\pipeline.py
```

在线执行：

```text
<ROOT>\code\trf_target\online.py
```

Chat 模型合同：

```text
qwen3.7-plus-2026-05-26
thinking = false
temperature = 0
structured JSON = true
```

### 第一轮：判断类型

输入由 label set 和目标句构成：

```text
Given entity label set: ["Skill"] ...
Target sentence: "Ideally you should have undergone training in IT support ..."
```

idx=98 的响应：

```json
{"entity_types": ["Skill"]}
```

### 第二轮：开放生成 TRF

同一消息历史依次包含：

```text
user：第一轮类型问题
assistant：第一轮 JSON 响应
user：16 条 demonstrations + 再次给出目标句
```

idx=98 的响应：

```json
{
  "trfs": ["tools", "technical", "communication", "years", "basic"]
}
```

Prompt 与原始响应路径：

```text
<ROOT>\output\trf_target_runs\trf-top20-target-v1\prompts\records.jsonl
<ROOT>\output\trf_target_runs\trf-top20-target-v1\raw\responses.jsonl
```

即使第一轮返回空类型，第二轮仍会执行。因此可能出现“类型为空但 TRF 非空”的冲突。

## 11. 目标阶段四：解析、诊断和最终输出

解析实现：

```text
<ROOT>\code\trf_target\pipeline.py
```

诊断实现：

```text
<ROOT>\code\trf_target\diagnostics.py
```

### 解析规则

- 第一轮只能返回 `[]` 或 `["Skill"]`。
- 第二轮必须返回字符串列表。
- TRF 执行 NFKC、首尾空白清理和稳定去重。
- 候选库外 TRF 不会被删除。
- 每项保存候选库命中和目标句出现情况。
- 类型为空但 TRF 非空时标记 `needs_review/type_absent_but_trfs_nonempty`。

### idx=98 最终结果

```json
{
  "idx": 98,
  "status": "complete",
  "entity_types": ["Skill"],
  "trfs": [
    {"normalized_text": "tools", "in_main_bank_exact": true},
    {"normalized_text": "technical", "in_main_bank_exact": true},
    {"normalized_text": "communication", "in_main_bank_exact": true},
    {"normalized_text": "years", "in_main_bank_exact": true},
    {"normalized_text": "basic", "in_main_bank_exact": true}
  ],
  "retrieval_count": 16,
  "review_reasons": []
}
```

### idx=14 完整负类闭环

```text
原句：You can walk on stairs
自我标注：negative
离线伪 TRF：[]
Qwen 第一轮：{"entity_types":[]}
Qwen 第二轮：{"trfs":[]}
最终：complete，entity_types=[]，trfs=[]
```

### idx=35 excluded 冲突示例

`idx=35` 在自我标注中是 `unsolved`，所以不进入 280 条 demonstration，但仍是 leave-one-out 的目标句：

```text
self_excluded = false
```

Qwen 返回：

```json
第一轮：{"entity_types":[]}
第二轮：{
  "trfs":[
    "communication",
    "development",
    "independently",
    "leadership",
    "projects",
    "sales"
  ]
}
```

最终标记：

```json
{
  "status": "needs_review",
  "review_reasons": ["type_absent_but_trfs_nonempty"]
}
```

### 最终目标输出路径

```text
<ROOT>\output\trf_target_runs\trf-top20-target-v1\parsed\records.jsonl
<ROOT>\output\trf_target_runs\trf-top20-target-v1\audit\summary.json
<ROOT>\output\trf_target_runs\trf-top20-target-v1\audit\manual_review.jsonl
<ROOT>\output\trf_target_runs\trf-top20-target-v1\manifest.json
<ROOT>\output\trf_target_runs\trf-top20-target-v1\source_snapshot.json
```

## 12. 当前 Top-20 输出目录完整清单

### 12.1 离线 run

根目录：

```text
<ROOT>\output\trf_runs\trf-offline-bert-base-cased-top20-v1
```

完整文件：

```text
manifest.json
source\snapshot.json
corpus\records.jsonl
corpus\excluded.jsonl
corpus\summary.json
candidates\selected_trfs.json
candidates\trfs.json
candidates\context_only_trfs.json
candidates\main_all_metrics.jsonl
candidates\main_paper_formula.jsonl
candidates\main_directional.jsonl
candidates\context_only_all_metrics.jsonl
candidates\context_only_paper_formula.jsonl
candidates\context_only_directional.jsonl
candidates\summary.json
demonstrations\pseudo_trfs.jsonl
demonstrations\summary.json
demonstrations\model_files.json
audit\manual_review_sample.jsonl
audit\semantic_review_sample.jsonl
```

以上每个文件的完整路径都是：

```text
E:\research\extractor\SkillSentence-Processor-0822\output\trf_runs\trf-offline-bert-base-cased-top20-v1\<对应文件>
```

### 12.2 目标 run

根目录：

```text
<ROOT>\output\trf_target_runs\trf-top20-target-v1
```

完整文件：

```text
manifest.json
source_snapshot.json
targets\records.jsonl
embeddings\demonstrations.jsonl
embeddings\targets.jsonl
retrieval\records.jsonl
prompts\records.jsonl
raw\responses.jsonl
parsed\records.jsonl
audit\summary.json
audit\manual_review.jsonl
```

以上每个文件的完整路径都是：

```text
E:\research\extractor\SkillSentence-Processor-0822\output\trf_target_runs\trf-top20-target-v1\<对应文件>
```

## 13. 完整父 run 的路径结构

完整入口：

```text
<ROOT>\scripts\rerun-all-trf.ps1
```

如果父 run-id 为 `<RUN_ID>`，会生成：

```text
<ROOT>\output\trf_full_runs\<RUN_ID>\manifest.json
<ROOT>\output\trf_full_runs\<RUN_ID>\target-config.generated.json
<ROOT>\output\trf_full_runs\<RUN_ID>\validation.json

<ROOT>\output\trf_runs\<RUN_ID>-offline\...
<ROOT>\output\trf_target_runs\<RUN_ID>-target\...
```

父 runner 会：

1. 创建不可覆盖的父 run。
2. 执行离线三个阶段。
3. 把本次离线 corpus、候选和伪 TRF 的 SHA256 写入动态目标配置。
4. 执行目标 Embedding、检索和两轮 Qwen。
5. 分别执行离线与目标只读验证器。
6. 生成父级 `validation.json`。

当前 Top-20 是分别执行的两个已验证 run：

```text
<ROOT>\output\trf_runs\trf-offline-bert-base-cased-top20-v1
<ROOT>\output\trf_target_runs\trf-top20-target-v1
```

当前没有与它们对应的 Top-20 父级 full run；这不影响两个子阶段的算法结果，只是没有额外父 manifest。

## 14. 运行命令

### 14.1 只运行离线 Top-20

```powershell
Set-Location E:\research\extractor\SkillSentence-Processor-0822
conda activate extractor
$python = (Get-Command python).Source

.\scripts\run-trf-offline.ps1 `
  -SelfAnnotatorRunId <self-annotator-run-id> `
  -RunId <new-offline-run-id> `
  -PythonExecutable $python
```

### 14.2 只运行目标 Qwen 阶段

```powershell
Set-Location E:\research\extractor\SkillSentence-Processor-0822
conda activate extractor
$python = (Get-Command python).Source

$env:DASHSCOPE_API_KEY = "<API Key>"
$env:DASHSCOPE_BASE_URL = "<OpenAI-compatible endpoint>"

.\scripts\run-target-trf.ps1 `
  -RunId <new-target-run-id> `
  -Mode leave-one-out `
  -AllowNetwork `
  -ConfirmFullRun `
  -ReuseEmbeddingsFrom trf-top20-target-v1 `
  -PythonExecutable $python
```

### 14.3 完整重跑

```powershell
Set-Location E:\research\extractor\SkillSentence-Processor-0822
conda activate extractor
$python = (Get-Command python).Source

.\scripts\rerun-all-trf.ps1 `
  -SelfAnnotatorRunId <self-annotator-run-id> `
  -RunId <new-parent-run-id> `
  -Mode leave-one-out `
  -AllowNetwork `
  -ConfirmFullRun `
  -ReuseEmbeddingsFrom trf-top20-target-v1 `
  -PythonExecutable $python
```

## 15. 实验文档路径

所有实验记录均位于项目根目录下：

```text
<ROOT>\research\experiments\EXPERIMENT_PLAN.md
<ROOT>\research\experiments\IMPLEMENTATION_PLAN.md
<ROOT>\research\experiments\RUN_LOG.md
<ROOT>\research\experiments\RESULTS.md
<ROOT>\research\experiments\FAILURES.md
<ROOT>\research\experiments\REPRODUCIBILITY.md
```

## 16. 一句话总结

当前 TRF 的核心是：先从自我标注结果构建 192 正类和 88 负类 demonstration，通过全局统计得到 20 个 Skill 候选词，再用固定 BERT 为每条正类选择 5 个伪 TRF；对于目标句，使用 Qwen Embedding 检索 16 条可靠示例，最后让 Qwen 在同一两轮会话中先判断 Skill 类型，再开放生成目标 TRF，并把所有冲突、来源和 SHA256 完整保存用于审计。

## 17. 目标 TRF 后的实例判别器

新增的 `code/instance_discriminator/` 是目标 TRF 的单向下游，不修改本文件前述两类 TRF
run，也不接入 `RunFullTRF.py`：

```text
target parsed record + target retrieval.selected[16]
  → join self-annotation decisions（只补 demonstration 跨度）
  → 一次 Qwen structured-JSON helpfulness 判别
  → 分数至少 4、角色非 irrelevant 的程序化 hard gate
  → 0～8 条 selected demonstrations
```

来源由 `--target-trf-run-id` 指定。runner 会从已完成目标 run 的 manifest 和
source snapshot 自动锁定 `retrieval/records.jsonl`、`parsed/records.jsonl` 与上游
`selected/decisions.jsonl` 的路径和 SHA256。排序使用分数、existence score、目标检索阶段
已经保存的 10 位 similarity、demo idx，不尝试恢复未保存的原始浮点数。

该阶段的设计、状态原因、目录合同和命令见 `INSTANCE_DISCRIMINATOR_DESIGN.md`；当前只报告
门控和审计指标，不把 leave-one-out 伪标签当作人工 Gold，也不宣称准确率提升。
