# 自我标注器技术说明：Span XMLC v0.6.0

## 1. 活动流水线

自我标注器保留 `skill-inline-annotation-v4.1` 提示词、`skill-inline-span-v1` 标注接口和
Qwen 请求协议。活动流程为：

```text
AskQwen（固定五次采样）
→ ParseAnswers（严格字符重建）
→ Has-skill 3/5 多数门控
→ Exact-span 3票接受
→ Residual overlap-family
→ 最高频span锚点覆盖硬匹配
→ SelectAnnotations
```

活动代码不使用 Token/BIO，也不追加自适应采样。历史 `archive/` 和既有 run 只读保留。

## 2. 固定采样与严格解析

`AskQwen.py` 强制 `generation.samples == 5`。每句必须具有且仅具有
`sample_index=0..4`；缺失、重复或额外索引是结构错误。

`ParseAnswers.py` 从唯一 JSON 对象读取 `annotated_sentence`，校验 `<skill>` 标签，删除
标签后逐字符重建原句，并为每个实际模型跨度写入：

```text
text
start
end
label_id = span:{idx}:{start}:{end}
```

安全投影只允许唯一、单调的 exact tag-payload 映射。解析失败是无效证据，不能作为
`no_skill` 票。

## 3. Has-skill 多数门控

对有效样本重新从 `annotation.spans` 是否为空派生 `has_skill`。设有效票、正票和空标签票
分别为 $V_{valid}$、$V_{has}$、$V_{no}$：

| 条件 | 结果 | 是否聚合跨度 |
|---|---|---|
| $V_{valid}<3$ | `abstained` | 否 |
| $V_{no}\ge4$ | `negative` | 否 |
| $V_{has}<3$ 且未判负 | `unsolved` | 否 |
| $V_{has}\ge3$ | 正门控通过 | 是 |

五次都有效时，$V_{has}=0/1$ 为负样本，$V_{has}=2$ 为 `unsolved`，
$V_{has}=3/4/5$ 进入跨度聚合。

## 4. Exact-span 多标签投票

每个 `(start,end)` 是当前句子动态标签空间中的一个标签。相同文本位于不同位置时仍是
不同标签；每个采样对同一标签最多贡献一票。

- exact 票数至少3：以 `exact_vote` 接受；
- 低票标签若与一个或多个 exact-accepted 标签重叠：记录为 `explained_variant`；
- explained variant 保存全部 `explained_by_label_ids`，不再进入 residual family。

正式字符串接口可以出现相同文本两次，位置差异保存在 audit 中。

## 5. Residual overlap-family

未被 exact 标签解释的跨度按正长度字符重叠建图，连通分量构成 family。Family support 是
贡献过至少一个成员的不同采样数量；同一采样最多贡献一票。

| Family support | 处理 |
|---|---|
| 1–2 | 作为噪声审计，不阻塞已有稳定标签 |
| 3–5 | 进入锚点覆盖硬匹配 |

正门控通过但没有 exact 或 hard-match winner 时，整句为 `unsolved`。

## 6. 最高频span锚点覆盖硬匹配

硬匹配要求至少三个有效样本、family support 至少3、至少两个唯一跨度，且 family 中没有
已达到三票的 exact 标签。Winner 必须是模型实际输出过的跨度。

### 6.1 锚点与覆盖率

首先选择 exact 票数最高的跨度作为锚点 $a$。对每个其他唯一跨度 $q$，计算字符区间的
锚点覆盖率：

$R(a,q)=\dfrac{|[a_s,a_e)\cap[q_s,q_e)|}{a_e-a_s}$

该指标是非对称的：较长跨度完整包含锚点时，$R(a,q)=1$；反向比较则会按较长锚点的
长度折减。

### 6.2 长度动态阈值

锚点执行 NFKC、小写化和空白合并，仅用于计算 `word_units`。技术词内部的
`. + # / -` 保留；连续 CJK 每两个字符计一个单位并向上取整。

设锚点的 `word_units` 为 $n$：

$T(n)=\min(0.85,\ 0.45+0.05(n-1))$

| $n$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | $\ge9$ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| $T(n)$ | .45 | .50 | .55 | .60 | .65 | .70 | .75 | .80 | .85 |

Winner 必须对所有其他唯一成员满足 $R(a,q)\ge T(n)$。比较使用 $10^{-12}$ 容差，
审计值保留六位小数。任一比较失败时，family 原因码为
`family_anchor_overlap_failed`。

系统不再要求 family 成员两两重叠。重叠链中不覆盖锚点的成员覆盖率为0，因此会自然
fail closed。

### 6.3 最高票并列

多个 span 并列最高票时，对每个并列候选分别作为锚点，依次按以下规则决胜：

1. 对其他成员的最小锚点覆盖率降序；
2. 平均锚点覆盖率降序；
3. `word_units` 升序；
4. 字符长度升序；
5. `start/end` 升序。

通过后仍以 `family_hard_match` 作为接受来源，保持正式接口兼容。

## 7. 严格句级裁决

任一正式 family 未解决或 accepted spans 发生重叠冲突时，整句输出 `unsolved`，不会把
部分稳定跨度写入正式文件。预接受证据仍保存在 audit。

最终状态只有：

- `accepted`：存在性多数通过，所有跨度已解决且无冲突；
- `negative`：至少四张有效空标签票；
- `unsolved`：存在性或跨度无法正式裁决；
- `abstained`：有效解析少于三次。

## 8. 输出与审计

`aggregated/consensus.jsonl` 保存正式字符串跨度、存在性共识、接受详情、候选跨度和未解决
family。`aggregation_audit.jsonl` 额外保存：

```text
anchor_label_id
anchor_exact_votes
top_exact_vote_tie
minimum_anchor_coverage
average_anchor_coverage
anchor_word_units
overlap_characters
anchor_characters
anchor_coverage
required_threshold
passed
```

`uncertainty_audit.jsonl` 保存 exact 标签集合 Jaccard、标签熵和 cardinality，不含 BIO。

只有 `accepted` 与 `negative` 进入 `selected/annotations.json`：

```json
{
  "idx": 1,
  "sentence": "Use Python.",
  "has_skill": 1,
  "spans": ["Python"]
}
```

四类完整决策保存在 `selected/decisions.jsonl`。

## 9. 活动配置与 schema

```text
min_valid_samples = 3
min_has_skill_votes = 3
min_no_skill_votes = 4
min_exact_candidate_votes = 3
min_exact_accept_votes = 3
min_family_candidate_votes = 3
hard_match_family_votes = 3
```

活动 consensus schema 为 `span-xmlc-majority-anchor-overlap-v2`，项目版本为 `0.6.0`。

## 10. 离线重聚合

`ReaggregateRun.py` 只读取历史基础 run 的 `raw/responses.jsonl`，不实例化网络客户端。
派生 manifest 保存源绝对路径、SHA256、`network_called:false`，并在结束时复核源文件未变。

```powershell
.\scripts\run-reaggregate.ps1 `
  -SourceRunRoot .\archive\self-annotation-history\inline-v4-2-full-20260821-001 `
  -RunId span-xmlc-base5-replay-majority3-anchor-v1
```

当前326句重放结果为225个 `accepted`、88个 `negative`、13个 `unsolved`、0个
`abstained`；共接受497个跨度，其中491个来自 exact vote、6个来自 hard match。

该 run 不覆盖历史结果。TRF 不再固定引用某个自我标注 run；运行时通过
`SelfAnnotatorRunId` 解析 `output/runs/<SelfAnnotatorRunId>`。
