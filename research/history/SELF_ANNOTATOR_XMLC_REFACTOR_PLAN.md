# 自我标注器重构方案：基于动态跨度 XMLC 的三级流程

## 1. 文档状态

- 目标项目：`SkillSentence-Processor-0822`
- 设计对象：自我标注器（Self-Annotator）
- 当前阶段：v0.6.0 已实施；本节修订优先于后续 v0.5.x 历史设计正文
- 核心约束：保留原提示词；固定 5 次采样；移除 BIO、Token 级共识和追加补采样
- 参考方法：CMAS 的分层多数投票、标签级置信度与样本级置信度

实施验证：v0.5.0 的 `span-xmlc-base5-replay-v1` 作为 support-5 对照保留；v0.5.1 的
`span-xmlc-base5-replay-support4-v1` 已对同一批 326 句基础响应完成零 API 重聚合，得到
accepted 192、negative 88、unsolved 46、abstained 0；其中 15 个跨度由硬匹配接受。

### v0.6.0 多数票与锚点覆盖修订

v0.6.0 将存在门控、exact 接受和 family 硬匹配统一为3/5多数票：

```text
min_valid_samples = 3
min_has_skill_votes = 3
min_exact_accept_votes = 3
hard_match_family_votes = 3
```

`min_no_skill_votes` 保持4。硬匹配先选择 exact 票数最高的模型原始跨度作为锚点，使用
$R(a,q)=|a\cap q|/|a|$ 比较所有其他唯一成员；阈值为
$T(n)=\min(0.85,0.45+0.05(n-1))$，其中 $n$ 只取锚点自身的 `word_units`。并列锚点按
最小覆盖率、平均覆盖率、`word_units`、字符长度和位置依次决胜。至少三个有效响应即可
运行；重叠链不再单独前置拒绝，而会因某个成员对锚点覆盖率为0自然失败。

活动 schema 为 `span-xmlc-majority-anchor-overlap-v2`。后续正文记录 v0.5.x 的设计演进，
凡涉及四票门控、加权相似度、两两重叠或五个响应全有效的规则，均已被本修订取代。

## 2. 重构目标

将当前基于 Token/BIO 的自我标注流程改造为：

> **Instance-wise Span XMLC（实例内动态跨度极端多标签分类）+ 存在性、精确跨度、残余跨度族三级流程**

模型不再预测每个 token 的 `B/I/O` 标签，而是让每次采样直接产生一个无序的精确字符跨度集合。聚合器以完整跨度为标签进行投票，并使用重叠跨度族检测“技能存在但边界尚未达成共识”的情况。

预期达到以下目标：

1. 正式接受的技能跨度必须被足够多的模型采样完整输出过。
2. 消除由不同采样的 token 票和边界票拼接出的虚构跨度。
3. 保留当前提示词、模型调用、原始响应和恢复运行能力。
4. 每个句子固定采样 5 次，不再对不确定样本追加采样。
5. 首先进行句级 `has_skill` 存在性投票，只有至少 4 票认为存在技能时才进入跨度判断。
6. 对五次采样一致认为“存在技能”但边界不同的 family，使用确定性硬匹配脚本裁决。
7. 存在性不确定或硬匹配失败的候选不进入正式标注，统一标记为 `unsolved`。

## 3. 当前 BIO 聚合的问题

当前流程大致为：

```text
LLM inline 输出
    → ParseAnswers
    → 字符跨度映射到 token/BIO
    → TokenBIOConsensus
    → token 覆盖票 + 左右边界票
    → accepted/review/negative/abstained
```

这种方法可能将来自不同模型采样的证据拼在一起。例如：

- 采样 A 支持技能内部的一部分 token；
- 采样 B 支持左边界；
- 采样 C 支持右边界；
- 没有任何采样完整输出最终跨度；
- BIO 聚合仍可能将其组合成一个正式跨度。

对当前结果的离线审计显示：476 个 accepted candidates 中，有 16 个候选（涉及 13 个句子）没有达到相应的完整跨度支持阈值。这说明 Token 级覆盖票和边界票可能形成从未被任何单次采样完整提出过的 “Frankenstein span”。

## 4. 任务形式化

### 4.1 动态标签空间

对于长度为 \(m\) 的输入句子 \(x\)，定义该句子的动态标签空间：

$
\mathcal L_x=\{(s,e)\mid 0\le s<e\le m\}
$

其中，\((s,e)\) 表示原句中的字符区间 `[start, end)`。

第 \(k\) 次模型采样输出一个无序标签集合：

$
Y_k=\{(s_1,e_1),(s_2,e_2),\ldots,(s_n,e_n)\}
$

标签身份定义为：

```text
(sentence_idx, start, end)
```

同一个技能词如果出现在句子的不同位置，应视为不同标签。`text` 只是由 `sentence[start:end]` 重建出的可读字段，不作为唯一标识。

### 4.2 与传统 XMLC 的区别

本方案属于**实例内动态标签空间的 span-set XMLC**，而不是使用固定技能 taxonomy 的传统全局极端多标签分类。

- 不提前建立全局技能词表；
- 不枚举 \(O(m^2)\) 个字符跨度；
- 继续让 LLM 通过原有 inline `<skill>` 格式稀疏提出候选；
- 聚合器只处理采样实际提出的跨度。

技能规范化、同义词合并和 taxonomy 映射应放在自标注之后，不与当前跨度识别任务混合。

## 5. 总体架构

```text
原提示词与固定 5 次独立采样
            │
            ▼
严格解析与原文重建
            │
            ▼
每次采样得到一个 exact-span label set
            │
            ▼
阶段一：Has-skill 存在性投票
            │
            ├── no_skill ≥ 4 ──────────────→ negative
            ├── has_skill 不足 4 且未判负 ─→ unsolved
            │
            ▼ has_skill ≥ 4
阶段二：Exact-span 标签投票
            │
            ▼
阶段三：Residual overlap-family 检测与硬匹配
            │
            ▼
集合冲突检查与句级状态机
            │
            ├── accepted
            ├── unsolved
            ├── negative
            └── abstained
```

## 6. 阶段零：严格解析

保留 `ParseAnswers` 中以下能力：

1. JSON 和 inline 标记格式验证；
2. 删除标记后必须与原句完全一致；
3. 将 `<skill>...</skill>` 唯一投影回原句；
4. 生成精确的字符级 `[start, end)`；
5. 同一次采样内的跨度去重；
6. 保存解析状态、恢复信息和原始响应索引。

移除：

- tokenization；
- Token 对齐；
- BIO 序列构造；
- 以 token 边界为依据的候选拒绝；
- 旧版辅助字段和处理逻辑。

建议单次采样结构：

```json
{
  "sentence_idx": 12,
  "sample_idx": 3,
  "parse_status": "ok",
  "has_skill": true,
  "labels": [
    {
      "label_id": "12:18:35",
      "start": 18,
      "end": 35,
      "text": "project management"
    }
  ]
}
```

`has_skill` 不需要修改原提示词，而是由严格解析结果确定：有效响应的 `labels` 非空时为 `true`，为空时为 `false`；解析失败时为 `null`，不得将其当作 `false`。

## 7. 阶段一：Has-skill 存在性投票

### 7.1 单次采样的存在性票

对第 \(k\) 个有效采样定义：

$
h_k=\mathbf 1[|Y_k|>0]
$

其中：

- `h_k = 1`：该采样至少输出一个有效技能跨度；
- `h_k = 0`：该采样成功解析，但输出空跨度集合；
- `h_k = null`：解析失败，不参与任一方向的投票。

分别统计：

$
V_{has}=\sum_k\mathbf 1[h_k=1]
$

$
V_{no}=\sum_k\mathbf 1[h_k=0]
$

并满足：

$
V_{valid}=V_{has}+V_{no}
$

### 7.2 存在性门控规则

当五次采样全部有效时：

| `V_has` | 含义 | 处理方式 |
|---:|---|---|
| 4 或 5 | 高置信存在技能 | 进入 Exact-span 与 family 判断 |
| 2 或 3 | 是否存在技能仍有分歧 | `unsolved` |
| 0 或 1 | 高置信不存在技能 | `negative` |

因此标准路径的核心规则是：

```text
V_has >= 4  → 继续跨度级投票
V_has <= 1  → negative
V_has ∈ {2, 3} → unsolved
```

当存在解析失败时，失败响应不视为 `no_skill`。此时使用等价的双边门控：

```text
V_valid < 3 → abstained
V_has >= 4  → 进入跨度级投票
V_no  >= 4  → negative
其他情况    → unsolved
```

例如，三个有效空响应和两个解析失败只能得到 `V_no=3`，不能因为 `V_has=0` 就判为负样本；该情况证据不足，进入 `unsolved`。这样可以防止解析错误被误当作模型的无技能判断。

### 7.3 存在性门控的审计输出

```json
{
  "valid_sample_count": 5,
  "has_skill_votes": 4,
  "no_skill_votes": 1,
  "invalid_sample_count": 0,
  "existence_status": "positive",
  "proceed_to_span_consensus": true
}
```

如果 `existence_status` 为 `negative`、`unsolved` 或 `abstained`，后续 exact-span、family 和 hard matcher 均不得运行。

## 8. 阶段二：跨度级共识（Exact-span → Overlap-family）

只有 `V_has >= 4` 时才能进入本阶段。

### 8.1 Exact-span 标签投票

对句子中的每个精确跨度标签 \(l=(s,e)\)，定义：

$
v(l)=\sum_{k=1}^{5}\mathbf 1[l\in Y_k]
$

其中，每次采样对同一标签最多贡献一票。低票 exact span 仍需保留，以便后续构建 overlap family。

### 8.2 Exact-span 固定阈值

| 判断 | 阈值 |
|---|---:|
| 候选 exact label | 3/5 |
| 正式接受 exact label | 4/5 |

只有完整 `[start,end)` 达到正式阈值时，该跨度才能进入高置信标签集合：

$
H=\{l\mid v(l)\ge4\}
$

不得再使用 token 票、独立左边界票或独立右边界票拼接正式跨度。

### 8.3 Overlap-family 存在性检测与硬匹配

朴素 exact voting 虽然能避免虚构跨度，但可能静默遗漏边界分裂的真实技能。例如三个语义相近的跨度分别获得 2、2、1 票，没有单个跨度达到 3 票，但五次采样都认为该位置存在技能。

因此需要第二阶段检测。

### 8.4 解释已接受标签的边界变体

对于未达到正式阈值的候选跨度，如果它与某个已接受标签 $h\in H$ 重叠，则将其记录为该标签的 `explained_variant`：

```text
accepted:  project management
variants:  management
           project-management
```

这些变体用于审计，但不会成为正式标签，也不再触发独立 review family。

### 8.5 构建剩余跨度族

对尚未被解释的跨度建立区间重叠图：

- 每个跨度是一个节点；
- 两个跨度在原文上发生字符重叠时连边；
- 每个连通分量构成一个 residual overlap family。

对 family \(F\) 定义支持度：

$
v(F)=\left|\{k\mid Y_k\cap F\neq\varnothing\}\right|
$

同一次采样即使在 family 中提出多个跨度，也只贡献一次存在性票。

### 8.6 Family 基础判定

- `v(F) < 3/5`：证据不足，不作为正式 family；
- `v(F) = 3/5`：证据未达到硬匹配准入条件，标记为 `unsolved`；
- `v(F) = 4/5` 或 `5/5`：至少四次采样认为该处存在技能，允许进入硬匹配脚本；
- 硬匹配通过：从 family 中选择票数最高的真实跨度，状态为 `accepted`；
- 硬匹配失败：不得猜测边界，状态为 `unsolved`。

因此，正式标签有两条互斥的接受路径：

1. `exact_vote`：某个完整跨度直接达到 `4/5`；
2. `family_hard_match`：没有跨度达到 `4/5`，但 family 至少达到 `4/5`，并通过确定性硬匹配。

硬匹配只能选择五次采样中真实出现过的跨度，禁止生成 union、intersection、平均边界或其他新跨度。

如果 `V_has >= 4`，但所有 residual family 的支持度都小于 `3/5`，并且不存在 `4/5` exact label，说明模型对“有技能”形成了共识，却无法对技能位置形成稳定共识。该句必须标记为 `unsolved`，不能退回 `abstained`。

### 8.7 硬匹配脚本的输入条件

建议新增独立脚本：

```text
hard_span_matcher.py
```

只有同时满足以下条件时才运行：

1. 固定采样数为 5；
2. 五次响应均有效；
3. family support 为 `4/5` 或 `5/5`；
4. family 中没有 exact span 达到 `4/5`；
5. support 中列出的每个采样至少为该 family 提出一个跨度；
6. family 内所有**唯一跨度两两发生正长度字符重叠**，避免由重叠链把两个不同技能错误连接。

任一前置条件不满足，脚本直接返回 `unsolved`。

### 8.8 文本规范化与长度单位

硬匹配只做确定性的表层匹配，不调用 LLM、embedding 或语义模型。

规范化步骤：

1. Unicode NFKC；
2. 英文转小写；
3. 合并连续空白；
4. 去除首尾普通标点，但保留 `C++`、`C#`、`.NET`、`Node.js` 等技术词内部符号；
5. 不删除停用词，因为 `of`、`for`、`and` 等词可能属于技能边界。

`word_units(span)` 默认取规范化后的词项数量。对于无法按空白切词的连续 CJK 文本，使用 `ceil(非空白字符数 / 2)` 作为回退长度单位。

比较两个跨度 `a`、`b` 时，使用较长者决定阈值：

$
n=\max(word\_units(a),word\_units(b))
$

这样，一词或两词技能可以容忍较低的重叠比例，而长技能短语必须具有更高的一致性。

### 8.9 硬匹配相似度

对两个来自同一句子的重叠跨度，计算：

1. **词项包含度**

$
C_{tok}=\frac{|T_a\cap T_b|}{\min(|T_a|,|T_b|)}
$

2. **词项 IoU**

$
J_{tok}=\frac{|T_a\cap T_b|}{|T_a\cup T_b|}
$

3. **规范化字符二元组 IoU**

$
J_{char}=\frac{|B_a\cap B_b|}{|B_a\cup B_b|}
$

其中 $B_a$、$B_b$ 是空白合并后规范化字符串的连续字符二元组。词项与字符二元组交集均采用 multiset 计数，最终硬匹配分数为：

$
S(a,b)=0.55C_{tok}+0.30J_{tok}+0.15J_{char}
$

较高的包含度权重允许 `management` 与 `project management` 这类短边界变体获得合理分数；词项 IoU 和字符 IoU 则抑制只有一个泛化词相同、但整体含义不同的跨度。

### 8.10 随跨度长度变化的阈值

动态阈值定义为：

$
T(n)=\min(0.86,\ 0.42+0.08(n-1))
$

对应阈值如下：

| 较长跨度的 `word_units` | 最低匹配分数 |
|---:|---:|
| 1 | 0.42 |
| 2 | 0.50 |
| 3 | 0.58 |
| 4 | 0.66 |
| 5 | 0.74 |
| 6 | 0.82 |
| 7 及以上 | 0.86 |

一对跨度只有在以下条件同时满足时才算硬匹配：

```text
字符交集长度 > 0
且
S(a, b) >= T(max(word_units(a), word_units(b)))
```

### 8.11 Winner 选择和整族裁决

首先从 family 中选择候选 winner：

1. exact vote 数最多；
2. 票数相同时，选择与 family 其他跨度平均相似度最高者；
3. 仍相同时，选择长度最接近 family 中位长度者；
4. 仍相同时，依次选择 `word_units` 更少、字符长度更短的跨度，避免边界过度扩张；
5. 仍相同时，按 `(start, end)` 升序确定，保证结果可重放。

因此，当两个成员获得相同最高 exact 票数时，不直接将 family 判为失败，也不随机选择，
而是把它们视为候选 medoid，使用第 2–5 项依次确定代表性、紧凑且可重放的 winner。

然后将 winner 与 family 中每个其他唯一跨度比较。只有**所有比较均达到各自的动态阈值**时，family 才通过硬匹配：

```text
family_support >= 4
AND all_unique_spans_pairwise_overlap
AND for every span q in family: score(winner, q) >= threshold(winner, q)
```

通过后：

```json
{
  "status": "accepted",
  "accept_method": "family_hard_match",
  "winner": "票数最多的原始跨度"
}
```

失败后：

```json
{
  "status": "unsolved",
  "reason": "family_hard_match_failed"
}
```

### 8.12 脚本核心伪代码

```python
def dynamic_threshold(a, b):
    n = max(word_units(a.text), word_units(b.text))
    return min(0.86, 0.42 + 0.08 * (n - 1))


def span_similarity(a, b):
    token_containment = multiset_intersection(a.tokens, b.tokens) / min(
        len(a.tokens), len(b.tokens)
    )
    token_iou = multiset_intersection(a.tokens, b.tokens) / multiset_union_size(
        a.tokens, b.tokens
    )
    char_iou = interval_iou((a.start, a.end), (b.start, b.end))
    return 0.55 * token_containment + 0.30 * token_iou + 0.15 * char_iou


def hard_match_family(family, valid_sample_count):
    if valid_sample_count != 5 or family.support != 5:
        return unsolved("family_not_unanimous")
    if family.max_exact_vote >= 4:
        return accepted(family.max_vote_span, method="exact_vote")
    if not all_unique_spans_pairwise_overlap(family.spans):
        return unsolved("family_is_overlap_chain")

    winner = choose_by_vote_centrality_and_median_length(family.spans)
    for span in family.unique_spans:
        if span_similarity(winner, span) < dynamic_threshold(winner, span):
            return unsolved("family_hard_match_failed")

    return accepted(winner, method="family_hard_match")
```

实际实现时必须处理空 token、重复词项和零长度区间；任何异常采用 fail-closed 策略返回 `unsolved`，不得默认接受。

## 9. CMAS 方法的保留与修正

CMAS 的两阶段思路可抽象为：

```text
mention 是否存在 → mention 属于什么类型
```

本方案映射为：

```text
句子是否存在技能
    → 哪些位置形成稳定的技能跨度或 overlap family
    → 哪个 exact span 是最终标签
```

保留的部分：

1. 多次独立采样与自一致性；
2. 先进行句级 `has_skill` 存在性投票；
3. 再进行标签级投票和置信度判断；
4. 保留样本级集合置信度；
5. 对 `4/5` 或 `5/5` family 使用第三阶段硬匹配；
6. 保留完整投票证据和硬匹配分项分数用于审计。

需要修正的部分：

1. family 必须至少达到 `4/5` 才允许启用最高票跨度裁决；
2. 最高票跨度仍须与 family 所有成员通过长度自适应硬匹配；
3. helpfulness 或示例相似度不作为 MVP 的门控；
4. `3/5` family，以及硬匹配失败的 `4/5` 或 `5/5` family，一律标记为 `unsolved`。

## 10. 多标签集合约束

同一次模型采样产生的跨度默认应互不重叠。聚合后也需要检查接受标签集合：

1. 接受标签之间不得重复；
2. 接受标签原则上不得相互重叠；
3. 如果不同 family 的正式标签相互重叠，不通过全局优化器强行选择，直接进入 `unsolved`；
4. 不在 MVP 阶段引入 weighted interval scheduling 或其他全局解码器。

由于正式接受阈值大于一半，且单次采样内跨度不重叠，两个互相重叠的标签通常无法同时过半。因此出现这种情况时，更可能意味着解析异常或采样约束被破坏，应优先审计。

## 11. 句级状态机

推荐按以下优先级决定最终状态：

1. **abstained**：`V_valid < 3`；
2. **negative**：`V_no >= 4`；五次响应均有效时等价于 `V_has <= 1`；
3. **unsolved**：存在性门控既未达到 `V_has >= 4`，也未达到 `V_no >= 4`；
4. 只有 `V_has >= 4` 时，才执行 exact-span 投票；
5. 对没有被 exact 标签解释的 residual family 计算支持度；
6. `4/5` 或 `5/5` family 运行硬匹配，通过后把 winner 加入正式标签集合；
7. **unsolved**：存在 `3/5` family，或 `4/5`、`5/5` family 硬匹配失败；
8. **unsolved**：`V_has >= 4`，但没有形成任何可接受 exact label 或稳定 family；
9. **unsolved**：正式标签之间存在重复、重叠或集合冲突；
10. **accepted**：存在非空、无冲突的正式标签集合，且所有 family 均已解决。

继续使用严格句级门控：如果句子中任何一个潜在技能 family 为 `unsolved`，则整句状态必须为 `unsolved`，不得只发布该句中其他已经接受的跨度。高置信跨度仍可保存在审计字段中，供人工处理时参考。

## 12. 建议输出结构

### 12.1 标签级审计

```json
{
  "label_id": "12:18:35",
  "start": 18,
  "end": 35,
  "text": "project management",
  "vote_count": 4,
  "valid_sample_count": 5,
  "confidence": 0.8,
  "sample_indices": [0, 1, 3, 4],
  "status": "accepted",
  "explained_variants": [
    {
      "start": 26,
      "end": 35,
      "text": "management",
      "vote_count": 1
    }
  ]
}
```

### 12.2 Family 级审计

```json
{
  "family_id": "12:f2",
  "member_labels": ["12:41:48", "12:41:55"],
  "family_vote_count": 5,
  "valid_sample_count": 5,
  "sample_indices": [0, 1, 2, 3, 4],
  "winner": "12:41:55",
  "winner_vote_count": 3,
  "accept_method": "family_hard_match",
  "pairwise_overlap": true,
  "minimum_pair_score": 0.78,
  "maximum_required_threshold": 0.74,
  "status": "accepted"
}
```

硬匹配失败时必须记录失败原因和最弱匹配对：

```json
{
  "family_id": "12:f3",
  "family_vote_count": 5,
  "winner": "12:70:83",
  "failed_pair": ["12:70:83", "12:70:102"],
  "pair_score": 0.61,
  "required_threshold": 0.82,
  "status": "unsolved",
  "reason": "family_hard_match_failed"
}
```

### 12.3 集合级审计

建议记录：

- 每次有效采样的 `has_skill` 布尔值；
- `V_has`、`V_no`、`V_valid` 和无效响应索引；
- `positive/negative/unsolved/abstained` 存在性门控结果；
- 是否进入跨度级共识；
- 每次采样的 exact label set；
- 两两集合 Jaccard；
- 完整集合完全一致率；
- 各采样预测的标签数量分布；
- 每个标签的 Bernoulli entropy；
- `3/5`、`4/5` 和 `5/5` family 数量；
- 硬匹配通过率与失败率；
- `exact_vote` 与 `family_hard_match` 的接受数量；
- 每个 hard-matched family 的 winner、动态阈值、最小匹配分数和失败原因；
- `unsolved` family 数量。

这些指标在 MVP 中只用于审计和分析，不直接代替正式阈值。

## 13. 代码重构边界

### 13.1 保持不变

- 原提示词和 inline `<skill>` 输出格式；
- `GeneratePrompts`；
- `AskQwen` 和 Qwen 客户端；
- 原始响应保存；
- resume manifest；
- 每个句子固定 5 次独立采样；
- `SelectAnnotations` 对外输出的基本句级结构。

### 13.2 修改 `ParseAnswers`

- 保留严格解析和字符跨度；
- 删除 token/BIO 生成；
- 删除 token-boundary rejection；
- 添加稳定的 `label_id`；
- 明确输出每次采样的无序跨度集合。

### 13.3 新增 `SpanXMLCConsensus.py`

职责：

1. 从有效 label set 派生 `has_skill`；
2. 统计 `V_has`、`V_no` 与 `V_valid`；
3. 执行存在性门控，未通过时提前返回；
4. exact-span 计票；
5. explained variants 匹配；
6. residual overlap graph 构建；
7. family 计票；
8. 调用 `hard_span_matcher.py`；
9. 集合冲突检查；
10. 句级状态判定；
11. 输出存在性、标签、family 和集合四级审计信息。

该模块替代活动调用链中的 `TokenBIOConsensus.py`。

### 13.4 新增 `hard_span_matcher.py`

职责：

1. 验证 `valid_sample_count == 5` 与 `family_support >= 4`；
2. 验证 family 内唯一跨度两两发生正长度字符重叠；
3. 完成 NFKC、技术词保护和词项切分；
4. 计算词项包含度、词项 IoU 和字符 IoU；
5. 根据较长跨度的 `word_units` 计算动态阈值；
6. 按票数、中心性、中位长度、紧凑度和字符位置确定 winner；
7. 对 winner 与所有成员执行 fail-closed 硬匹配；
8. 返回 `accepted/family_hard_match` 或 `unsolved` 以及完整审计证据。

该脚本必须是无随机性的纯函数。同一输入在不同运行中必须得到相同结果。

### 13.5 修改 `SelectAnnotations`

- `accepted` 只读取正式 exact-span 标签；
- `accepted` 同时允许 `exact_vote` 和 `family_hard_match` 两种来源；
- `unsolved` 输出高票标签、失败 family、动态阈值、相似度和原始证据；
- 对外的 `text/start/end` 格式尽量保持兼容；
- 不再读取 BIO 或 token consensus 字段。

## 14. Schema 与迁移策略

建议使用新的 schema 标识：

```text
span-xmlc-has-skill-hardmatch-v1
```

迁移原则：

1. 不覆盖当前 v4.2 输出；
2. 新结果写入独立目录；
3. 保留旧结果作为对照基线；
4. 先使用现有 parsed char spans 离线重聚合；
5. 在确认解析字段足够后再删除活动调用链中的 BIO 代码；
6. 旧 BIO 文件可归档，但新模块不得再依赖它们；
7. 所有离线重放只使用每句固定的前 5 次采样；
8. `has_skill` 从已有 parsed label set 离线派生，不需要重新请求模型。

已有模型响应和解析跨度应能够直接重放，因此 E1/E2 实验原则上不需要新增 API 调用。

## 15. 最小可行实验

### E0：冻结当前基线

记录当前 BIO v4.2 固定 5 次采样结果：

| 状态 | 数量 |
|---|---:|
| accepted | 188 |
| review | 40 |
| abstained | 10 |
| negative | 88 |

正式自动处理覆盖为 `276/326 = 84.66%`，人工处理量为 `50/326 = 15.34%`。该表仅作为旧 BIO 基线，重构后的新状态使用 `unsolved`，不再使用 `review`。

### E1：朴素 Exact Baseline

仅以精确跨度投票，不构建 overlap family。

必须仅使用每个句子的固定 5 次采样重新计算。该结果用于确认 exact voting 消除虚构跨度的收益，以及它静默遗漏边界分裂技能的风险。

### E2：主方案离线重聚合

运行 has-skill gate + exact-span voting + explained variants + residual overlap-family + dynamic hard matcher：

- 不重新调用 Qwen；
- 核查 `V_has >= 4` 是否严格作为所有跨度判断的前置条件；
- 核查 `V_has <= 1` 的五票有效样本是否全部直接判为 `negative`；
- 核查 `V_has` 为 2 或 3 的样本是否全部进入 `unsolved`；
- 比较与 BIO 和朴素 exact baseline 的状态迁移；
- 分别核查 `exact_vote` 和 `family_hard_match` 接受的跨度；
- 核查 hard matcher 选择的 winner 是否均来自原始采样；
- 核查 hard matcher 失败的 family 是否全部进入 `unsolved`；
- 统计不同词数下的阈值、通过率与错误率。

### E3：建立人工 Gold

抽取 50–100 条句子进行精确字符跨度标注，至少覆盖：

- 明确正例；
- 明确负例；
- `V_has` 分别为 0、1、2、3、4、5 的样本；
- 存在解析失败、但 `V_has/V_no` 尚未达到四票的样本；
- 多技能句；
- 相邻技能；
- 重复技能词；
- 长短边界争议；
- `3/5`、`4/5` family；
- `4/5` 或 `5/5` 且硬匹配通过的 family；
- `4/5` 或 `5/5` 但硬匹配失败的 family；
- 一词/两词短技能和七词以上长技能。

### E4：阈值消融

比较：

- 是否启用 has-skill 前置门控；
- 将 `V_has >= 4` 与 `V_has >= 3` 作为对照，但正式方案固定使用 4；
- exact accept：`3/5`、`4/5`、`5/5`；
- 动态阈值起点：`0.38`、`0.42`、`0.46`；
- 每增加一个 word unit 的增量：`0.06`、`0.08`、`0.10`；
- 阈值上限：`0.82`、`0.86`、`0.90`；
- 相似度权重组合；
- 是否要求所有唯一跨度两两重叠；
- winner 只比较其余成员，或要求 family 内所有跨度两两通过相似度阈值；
- 是否启用严格句级门控；
- 是否将 explained variants 从 residual family 中移除。

## 16. 必要基线

1. 当前 Token/BIO v4.2；
2. 不使用 has-skill gate 的朴素 exact-span majority；
3. has-skill gate + 朴素 exact-span majority；
4. 采样集合 medoid；
5. has-skill gate + overlap-family，但不启用硬匹配；
6. has-skill gate + 固定阈值硬匹配；
7. 主方案：has-skill gate + overlap-family + 长度自适应硬匹配。

暂不将以下方法作为 MVP：

- 固定全局技能 taxonomy XMLC；
- Proposal–Verifier 多代理架构；
- weighted interval scheduling；
- 独立 start/end 分类；
- EM 或概率弱监督聚合；
- 依赖 helpfulness 的动态示例选择。

## 17. 评价指标

### 跨度级

- exact-span micro precision；
- exact-span micro recall；
- exact-span micro F1；
- accepted-only exact precision；
- 边界偏移分布。

### 句子级

- has-skill precision、recall 和 F1；
- `V_has` 五个票数档位的 gold 分布；
- existence gate 的 positive/negative/unsolved/abstained 数量；
- 标签集合 exact match；
- accepted / unsolved / abstained / negative 数量；
- 自动覆盖率；
- 人工处理率；
- negative precision/recall。

### 安全性

- Frankenstein span 数量；
- `family_hard_match` 接受跨度的人工正确率；
- hard matcher 错误接受和错误拒绝数量；
- 硬匹配失败却被错误放行的句子数量；
- 接受标签集合内部的重叠冲突数量。

### 成本

- 每句模型采样数，必须固定为 5；
- hard matcher 触发率；
- 每个正式接受句的 API 成本；
- 每个正确跨度的 API 成本。

## 18. 验收标准

重构方案至少应满足：

1. 任何 `accepted` 句子必须满足 `V_has >= 4`；
2. 五次响应均有效时，`V_has <= 1` 必须直接得到 `negative`；
3. 存在无效响应时，只有 `V_no >= 4` 才能判为 `negative`，无效响应不得充当无技能票；
4. 未达到正向或负向四票门控的样本必须进入 `unsolved`，不得运行跨度级聚合；
5. 每个 accepted span 必须明确记录 `exact_vote` 或 `family_hard_match`；
6. `exact_vote` 接受的跨度必须达到 `4/5`；
7. `family_hard_match` 接受的跨度必须来自 support 至少为 `4/5` 的 family，且必须是模型实际输出过的跨度；最高票并列时按平均相似度、长度中位数距离、紧凑度和位置依次决胜；
8. hard matcher 中所有成员比较必须达到各自的长度动态阈值；
9. hard matcher 失败或异常的 family 必须 100% 进入 `unsolved`；
10. Frankenstein span 数量为 0；
11. accepted 句子中 `unsolved` family 数量为 0；
12. 人工 gold 上 accepted-only exact precision 不低于固定 5 次采样的 BIO 基线；
13. 长度自适应硬匹配相较固定阈值硬匹配，在不降低 accepted-only precision 的前提下降低 `unsolved` 比例；
14. E1/E2 可以完全使用现有 5 次采样响应离线重放；
15. 原提示词和模型请求协议无需修改；
16. 每个句子只允许 5 次模型采样，不存在追加请求；
17. 新运行链路不存在旧版辅助逻辑、token consensus 或 BIO 依赖。

## 19. 推荐实施顺序

1. 冻结 v4.2 基线和当前产物；
2. 为现有 parsed 数据补充稳定 `label_id`；
3. 从现有 parsed label set 派生 `has_skill`，实现四票存在性门控；
4. 为 `V_has=0/1/2/3/4/5` 和无效响应组合编写状态机测试；
5. 实现纯函数式 exact-span 计票；
6. 实现 explained variants 与 overlap-family；
7. 实现 `hard_span_matcher.py` 的规范化、相似度和动态阈值；
8. 为硬匹配脚本编写短跨度、长跨度、重叠链和并列场景单元测试；
9. 实现 `accepted/unsolved/negative/abstained` 句级状态机和审计输出；
10. 只使用每句前 5 次采样离线重放现有 326 条数据；
11. 对所有 `family_hard_match` 和 `unsolved` 状态迁移样本进行人工核查；
12. 建立 50–100 条人工 gold；
13. 完成阈值消融和基线比较；
14. 指标达标后切换活动调用链；
15. 最后删除或归档 BIO、Token consensus 和额外补采样遗留代码。

## 20. 结论

推荐采用 **Has-skill 门控 → Exact-span 投票 → Overlap-family 硬匹配** 的三级结构：

- 首先用句级 `has_skill` 四票门控决定是否允许进行跨度判断；
- `V_has <= 1` 的五票有效样本直接判为 `negative`，`V_has` 为 2 或 3 时判为 `unsolved`；
- 用 exact-span 投票决定可以正式发布的完整技能边界；
- 用 overlap-family 检测“技能存在但边界不同”的情况；
- 对 support 为 `4/5` 或 `5/5` 的 family 使用长度自适应硬匹配，并通过确定性并列规则选择真实跨度；
- 将 hard matcher 无法可靠裁决的 family 标记为 `unsolved`；
- 用严格句级门控阻止不完整多标签集合被静默接受；
- 每句固定采样 5 次，不追加模型请求；
- 不改变原提示词，离线验证不需要重新调用模型。

该方案保留了 CMAS 的多次采样和分层置信度思想，并将判断进一步拆成句级存在性、精确跨度与 family 边界裁决三级，更适合作为当前 BIO 自标注器的替代架构。

## 21. 相关工作定位

无 BIO 的 span classification、跨度组合、无序集合预测以及 self-consistency 已有相关研究。因此，本项目应把方法贡献定位为“动态跨度 XMLC、CMAS 启发的三级流程、长度自适应硬匹配和严格句级门控的组合”，而不是将“不使用 BIO”本身作为新颖性声明。

- GNNer: <https://aclanthology.org/2022.acl-srw.9/>
- SpanNER: <https://aclanthology.org/2021.acl-long.558/>
- Global Span Selection: <https://aclanthology.org/2022.umios-1.2/>
- Set Generation Networks: <https://aclanthology.org/2021.emnlp-main.760/>
- Extreme Classification from Aggregated Labels: <https://proceedings.mlr.press/v119/shen20f.html>
- Self-Consistency: <https://arxiv.org/abs/2203.11171>
