# 实例判别器设计：TRF-aware Demonstration Discriminator v1

## 1. 设计结论

实例判别器位于目标句 TRF 提取器之后、未来的技能跨度预测器之前：

```text
Span XMLC 自我标注器
        ↓
离线 TRF：全局候选 + demonstration 伪 TRF
        ↓
目标句 TRF：K=50 → k=16 + 目标类型 + 目标 TRF
        ↓
实例判别器：16 条候选逐条评分 + 程序化 hard gate
        ↓
0～8 条高帮助 supporting / contrastive demonstrations
        ↓
未来的技能跨度预测器与聚合器
```

v1 采用单向协作：`目标 TRF → 实例判别器`。判别器不重新生成 TRF，也不修改自我标注、
离线 TRF 或目标 TRF 产物。

该设计模仿 CMAS 的三项核心结构：

1. embedding 先召回候选示例；
2. LLM 根据目标信息判断每条示例的 helpfulness；
3. 下游预测器使用判别结果。

但不复制 CMAS 的纯软提示。模型评分必须被解析，低分示例必须被程序化排除。这与本项目
`research/ideas/SELECTED_IDEA.md` 中的既有约束一致。

## 2. 目标与非目标

### 2.1 目标

- 判断候选示例是否有助于目标句的精确 Skill 字符跨度识别，而不只是主题相似。
- 同时允许正类示例作为 `supporting` 证据、负类示例作为 `contrastive` 证据。
- 使用目标实体类型、目标 TRF、示例伪 TRF、自标注可靠度和语义相似度。
- 对 LLM 输出进行严格结构校验和确定性 hard gate。
- 保存 source hash、prompt、raw response、解析结果、门控决定和人工审计样本。
- 支持 `prepare-only`、断点续跑和显式 `retry-failed`。

### 2.2 非目标

- 不在判别器内重新标注目标句的技能跨度。
- 不把判别器分数当作人工 Gold 或准确率。
- 不覆盖目标 TRF 阶段的 K=50 → k=16 检索产物。
- 不允许低分示例因为数量不足而静默回填。
- v1 不实现 TRF 与判别器的循环反馈。

## 3. 权威输入

实例判别器只读取锁定的上游产物。

### 3.1 目标 TRF run

```text
output/trf_target_runs/<TargetTRFRunId>/manifest.json
output/trf_target_runs/<TargetTRFRunId>/source_snapshot.json
output/trf_target_runs/<TargetTRFRunId>/retrieval/records.jsonl
output/trf_target_runs/<TargetTRFRunId>/parsed/records.jsonl
```

每个目标提供：

- `idx`、`sentence`；
- `entity_types`，只能是 `[]` 或 `["Skill"]`；
- 开放生成的 `trfs`；
- K=50 邻居和已经选出的 16 条候选示例；
- 每条候选的 `similarity`、`status`、`existence_score` 和伪 TRF。

### 3.2 自我标注 decision

从目标 TRF run 的 source snapshot 解析并锁定：

```text
output/runs/<SelfAnnotatorRunId>/selected/decisions.jsonl
```

只为 demonstration 补充：

- accepted 示例的正式 `spans`；
- negative 示例的空跨度；
- `accepted_span_details` 和存在性共识，用于审计可靠度。

禁止把 leave-one-out 目标自身的伪标签、状态或跨度放入判别 prompt，避免标签泄漏。

## 4. 候选示例记录

对目标 TRF 检索产物中的 16 条 `selected` 示例做 join：

```json
{
  "demo_idx": 103,
  "sentence": "...",
  "status": "accepted",
  "skill_spans": ["project management"],
  "pseudo_trfs": ["projects", "management", "skills", "tools", "process"],
  "similarity": 0.5650409234,
  "existence_score": 1.0,
  "retrieval_rank": 1
}
```

negative 示例保持：

```json
{
  "status": "negative",
  "skill_spans": [],
  "pseudo_trfs": []
}
```

negative 不能因为 TRF 为空而自动判为无帮助。它可能是判断“相似招聘表述为什么不构成技能”的
重要对照证据。

## 5. 模块结构

```text
code/instance_discriminator/
├── __init__.py
├── common.py
├── pipeline.py
├── online.py
├── diagnostics.py
├── RunInstanceDiscriminator.py
└── ValidateInstanceDiscriminatorRun.py

config/instance_discriminator.json
scripts/run-instance-discriminator.ps1
tests/test_instance_discriminator_pipeline.py
```

### 5.1 `common.py`

- 定义 `InstanceDiscriminatorError`。
- 校验配置和 source snapshot。
- 创建不可覆盖的 run 目录。
- 固定上游文件的 SHA256。
- 管理 manifest、阶段状态、依赖版本和输出 hash。
- 提供 `latest_records()` 支持 append-only raw response 续跑。

### 5.2 `pipeline.py`

建议提供纯函数：

```python
load_source_bundle(config, target_trf_run_id, limit)
build_candidate_record(target, retrieval, decisions_by_idx, expected_count)
build_discriminator_prompt(candidate_record, max_characters)
parse_judgments(content, candidates, max_reason_characters=240)
apply_hard_gate(candidates, judgments, gate_config)
build_selected_record(candidate_record, judgments, gate_config, chat_model)
```

所有排序、解析和门控逻辑必须可在无网络测试中复算。

### 5.3 `online.py`

- 复用 `code/common/qwen_client.py`。
- 每个目标一次批量判别调用，不对 16 条示例分别请求。
- temperature 固定为 0，启用 structured JSON。
- raw response append-only 保存。
- 已成功目标在 resume 时不重调。
- 失败目标只有显式 `-RetryFailed` 才重调。

### 5.4 `diagnostics.py`

统计：

- complete / needs_review / failed 数量；
- 评分 1～5 分布；
- supporting / contrastive / irrelevant 分布；
- hard gate 保留率与每目标保留数量；
- accepted / negative 示例保留率；
- 评分与 embedding similarity、existence score 的相关性；
- 无示例和少于两条示例的目标比例；
- TRF 为空目标与 TRF 非空目标的评分差异；
- 固定随机种子的人工审计样本。

### 5.5 Validator

`ValidateInstanceDiscriminatorRun.py` 必须只读复算：

- source hash；
- 候选 join；
- prompt hash；
- raw response 解析；
- hard gate；
- selected records；
- summary 和所有正式输出 hash。

## 6. LLM 判别任务

### 6.1 判别目标

模型判断的是：

> 如果把当前 demonstration 提供给后续技能跨度预测器，它能否帮助预测器准确判断目标句中
> 是否存在 Skill，并精确确定原文字符跨度？

必须明确提示“主题相似不等于有帮助”。

### 6.2 Prompt 输入

```json
{
  "target": {
    "sentence": "Work in a way that is patient-centred and inclusive .",
    "entity_types": ["Skill"],
    "target_trfs": ["communication", "skills"]
  },
  "demonstrations": [
    {
      "demo_idx": 103,
      "sentence": "...",
      "status": "accepted",
      "skill_spans": ["..."],
      "pseudo_trfs": ["..."],
      "similarity": 0.5650409234,
      "existence_score": 1.0
    }
  ]
}
```

目标句只能携带目标 TRF 阶段产生的 `entity_types` 和 `target_trfs`，不能携带其上游
self-annotation label。

### 6.3 输出 schema

```json
{
  "judgments": [
    {
      "demo_idx": 103,
      "helpfulness_score": 5,
      "role": "supporting",
      "reason_codes": ["TRF_ALIGNED", "BOUNDARY_TRANSFERABLE"],
      "reason": "The example contains a transferable skill-span boundary pattern."
    },
    {
      "demo_idx": 2,
      "helpfulness_score": 4,
      "role": "contrastive",
      "reason_codes": ["NEGATIVE_CONTRAST"],
      "reason": "It is a useful no-skill contrast for similar wording."
    }
  ]
}
```

严格约束：

- 16 个 `demo_idx` 必须各出现一次；
- 不得出现未知或重复 ID；
- `helpfulness_score` 必须是 1～5 的整数；
- role 只能是 `supporting`、`contrastive`、`irrelevant`；
- accepted 示例只能是 supporting 或 irrelevant；
- negative 示例只能是 contrastive 或 irrelevant；
- reason codes 只能来自固定枚举；
- reason 限制长度，只用于审计，不参与门控。

建议原因码：

```text
TYPE_ALIGNED
TRF_ALIGNED
BOUNDARY_TRANSFERABLE
NEGATIVE_CONTRAST
SEMANTIC_ONLY
LABEL_UNCERTAIN
TASK_MISMATCH
REDUNDANT
```

## 7. 程序化 hard gate

### 7.1 基本门槛

```text
keep iff:
  helpfulness_score >= 4
  and role in {supporting, contrastive}
```

任何 1～3 分示例都不得进入正式 selected 输出。

### 7.2 确定性排序

通过门槛的示例按以下顺序排序：

1. `helpfulness_score` 降序；
2. `existence_score` 降序；
3. 目标 TRF 检索产物中已经保存的 10 位 embedding similarity 降序；
4. `demo_idx` 升序。

建议最多保留 8 条：

```text
max_selected = 8
max_supporting = 6
max_contrastive = 3
```

这些是上限，不是必须填满的配额。不得为了填满 8 条而回填低分示例。

### 7.3 少样本处理

- 保留 2～8 条：`complete`；
- 保留 1 条：`needs_review / insufficient_helpful_examples`；
- 保留 0 条：`needs_review / no_helpful_examples`；
- 未来的下游预测器在 0 条时应执行明确标记的 zero-shot，而不是恢复使用原来的 16 条；
  该预测器不属于 v1 的实现范围。

### 7.4 TRF 缺失处理

当前真实运行中可能出现 `entity_types=["Skill"]` 但 `target_trfs=[]`。此时判别器仍可根据：

- 目标句与示例句；
- accepted/negative 状态；
- 示例技能跨度；
- embedding similarity；
- self-annotation existence score；

进行判断，同时为结果添加 `skill_type_without_target_trfs` 复核原因。不得伪造 TRF。

## 8. 正式输出

建议每个目标输出：

```json
{
  "idx": 1,
  "sentence": "...",
  "status": "complete",
  "target_evidence": {
    "entity_types": ["Skill"],
    "trfs": ["communication", "skills"]
  },
  "candidate_count": 16,
  "selected_count": 5,
  "selected": [
    {
      "demo_idx": 103,
      "role": "supporting",
      "helpfulness_score": 5,
      "sentence": "...",
      "skill_spans": ["..."],
      "pseudo_trfs": ["..."],
      "similarity": 0.5650409234,
      "existence_score": 1.0,
      "reason_codes": ["TRF_ALIGNED", "BOUNDARY_TRANSFERABLE"]
    }
  ],
  "rejected_demo_ids": [2, 75],
  "review_reasons": [],
  "models": {
    "chat": "qwen3.7-plus-2026-05-26"
  }
}
```

正式 selected 记录保留判别证据，但下游 prompt 不一定需要携带 reason 文本。

## 9. Run 目录

```text
output/instance_discriminator_runs/<RunId>/
├── manifest.json
├── source_snapshot.json
├── candidates/records.jsonl
├── prompts/records.jsonl
├── raw/responses.jsonl
├── parsed/judgments.jsonl
├── selected/records.jsonl
└── audit/
    ├── summary.json
    └── manual_review.jsonl
```

run-id 不可覆盖。manifest 至少记录：

- 配置和实现 SHA256；
- 目标 TRF run-id；
- self-annotator 来源及 SHA256；
- 模型与 structured-output 参数；
- hard-gate 阈值和排序合同；
- network_called；
- 每阶段状态和最终输出 hash。

## 10. 配置草案

```json
{
  "schema_version": 1,
  "pipeline_version": "instance-discriminator-v1",
  "source": {
    "target_runs_root": "output/trf_target_runs",
    "candidate_count": 16
  },
  "provider": {
    "name": "aliyun-model-studio",
    "api_key": "${DASHSCOPE_API_KEY}",
    "base_url": "${DASHSCOPE_BASE_URL}"
  },
  "chat": {
    "model": "qwen3.7-plus-2026-05-26",
    "enable_thinking": false,
    "structured_output": true,
    "temperature": 0.0,
    "max_tokens": 4096,
    "timeout_seconds": 120,
    "max_retries": 5,
    "max_prompt_characters": 60000,
    "max_reason_characters": 240
  },
  "gate": {
    "minimum_helpfulness": 4,
    "max_selected": 8,
    "max_supporting": 6,
    "max_contrastive": 3,
    "minimum_for_complete": 2
  },
  "diagnostics": {
    "review_sample_size": 20
  },
  "output": {
    "runs_root": "output/instance_discriminator_runs"
  }
}
```

## 11. PowerShell 入口

```powershell
.\scripts\run-instance-discriminator.ps1 `
  -TargetTRFRunId <target-trf-run-id> `
  -RunId <new-discriminator-run-id> `
  -PrepareOnly

.\scripts\run-instance-discriminator.ps1 `
  -TargetTRFRunId <target-trf-run-id> `
  -RunId <discriminator-run-id> `
  -Resume `
  -AllowNetwork `
  -ConfirmFullRun
```

`PrepareOnly` 应创建并验证 candidates、prompts 和 manifest，但不实例化网络客户端。

Python runner 的实际接口为：

```text
--config --target-trf-run-id --run-id --limit --prepare-only
--allow-network --resume --retry-failed --confirm-full-run
```

只读验证器只需要 `--run-id`（以及可选 `--config`），它会从判别器 manifest 找回并复核
目标 TRF run-id。prepare-only run 返回 `valid_prepared`，完整 run 返回 `valid_completed`。

## 12. 测试与验收

### 12.1 单元测试

1. target TRF、retrieval 和 decision 的 idx 不一致时拒绝运行。
2. leave-one-out prompt 不得包含目标自身的 decision、status 或 spans。
3. 每个目标必须恰好有 16 条候选。
4. 16 个 judgment 必须完整、唯一且 ID 匹配。
5. 分数、role 和 reason code 非法时解析失败。
6. 1～3 分示例永远不会进入 selected。
7. 高分 negative 可以作为 contrastive 示例进入 selected。
8. 排序和上限规则在重复运行中完全确定。
9. 0/1 条通过门控时正确产生 review reason。
10. resume 不重调成功目标，retry-failed 只重调失败目标。

### 12.2 集成测试

- 使用 fake Qwen client 完成 independent 与 leave-one-out 小样本运行。
- Validator 能从 raw response 完整复算 parsed、selected 和 summary。
- prepare-only 保证 `network_called=false`。
- 上游 source 文件运行前后 SHA256 不变。

### 12.3 消融实验

在获得人工 Gold 或实现最终跨度预测器后，至少比较：

1. KNN-16：不使用判别器；
2. CMAS-soft：保留 16 条，只把评分放进上下文；
3. Hard-gate：本设计，低分示例程序化删除；
4. No-TRF：判别器不读取目标/示例 TRF；
5. No-reliability：不提供 existence score。

在没有人工 Gold 前，只报告门控、稳定性和审计指标，不宣称提升技能抽取准确率。

## 13. 双向协作扩展

v1 验证后可以增加一次受控反馈：

```text
目标 TRF₀
  → 判别器评分并 hard gate
  → 使用保留示例重新生成目标 TRF₁
  → 判别器复评一次
  → 下游预测器
```

必须固定 `feedback_rounds=1`，分别保存 round-0 和 round-1 证据，并与单向 v1 做独立消融。
在没有人工 Gold 和最终预测器基线之前，不应默认启用双向模式。
