# 选定方案：Instance-wise Span XMLC with Two-Stage Consensus

## 一句话定义

保留当前 LLM 内联标注提示词，把每次响应解析成一个稀疏的 exact-span 标签集合；先用重叠区间族汇总
“技能是否存在”的证据，再用完整 `[start,end)` 标签票决定边界，彻底移除 Token/BIO 聚合。

## 精确研究假设

在相同 prompt、模型、采样次数和解析证据下，**overlap-family presence → exact-span boundary** 的
两阶段多标签共识，相比 Token/BIO 共识能够：

1. 将所有正式 accepted span 限制为获得完整 exact 多数支持的真实模型提案；
2. 阻止朴素 exact vote 因边界票分散而产生的部分集合假接受；
3. 在不增加 API 成本的前提下，提高 accepted-only exact-span precision；
4. 将补采样后的人工复核率保持在可接受范围。

## 任务形式化

给定字符长度为 `m` 的原句 `x`：

```text
L_x = { span(s,e) | 0 <= s < e <= m }
```

`L_x` 是该句动态生成的极端标签空间。第 `k` 次模型采样经严格解析得到：

```text
Y_k ⊂ L_x
```

每个标签身份由 `(idx,start,end)` 唯一确定，表面文本始终等于 `sentence[start:end]`。标签集合无序；
同一表面词出现在不同位置时是不同标签。

不显式枚举全部 `L_x`，只保存模型提出过的稀疏候选：

```text
C_x = union(Y_1, ..., Y_N)
v(l) = sum_k 1[l in Y_k]
p(l) = v(l) / N_valid
```

这不是带固定技能 taxonomy 的传统全局 XMLC，而是 **动态标签空间的 span-set XMLC**。技能规范化应留在
后续独立模块。

## 两阶段共识

### 阶段 0：严格解析

保留现有 JSON、inline-tag、逐字符重构、唯一 exact projection 和字符偏移验证。移除：

- Token 化；
- BIO 序列；
- “span 必须贴合 Token 边界”的限制。

继续拒绝空、嵌套、重叠、重复和无法唯一映射回原句的单样本输出。

### 阶段 1：Exact label 投票

每个样本对同一 `[start,end)` 最多贡献一票。基础 5 样本使用：

- `min_valid_samples = 3`
- `exact_candidate_votes = 3`
- `exact_accept_votes = 4`
- `negative_votes = 4`

Adaptive 10 样本使用：

- `min_valid_samples = 6`
- `exact_candidate_votes = 6`
- `exact_accept_votes = 7`
- `negative_votes = 7`

正式 span 只能来自 `v(l) >= exact_accept_votes` 的完整标签，禁止由不同样本的覆盖票和边界票拼接。

### 阶段 2：Overlap-family presence

只做 exact vote 会遗漏“多个样本都认为这里有技能，但边界各不相同”的情况。因此对模型提出的区间构造
冲突图：两个不同标签的字符区间只要有非空交集，就连接一条边；连通分量形成 overlap family。

为避免链式重叠把已确定技能和相邻争议技能一起吞掉，采用两步残余检查：

1. 先得到 exact accepted 标签集合 `H`；
2. 与任一 `H` 重叠的低票标签视为该 accepted span 的边界变体，只用于审计；
3. 对其余未解释标签重新构建 residual overlap families；
4. family support 是至少提出该 family 一个成员的不同样本数，每个样本最多贡献一票；
5. residual family support 达到 3/5 或 6/10，但没有 exact accepted winner 时，整句必须进入 review。

这对应 CMAS 的两阶段思想：

```text
CMAS: mention 是否存在 → mention 的 type
本方案: span family 是否存在 → exact boundary label
```

### 多标签集合约束

- `H` 是无序集合。
- 基础 4/5 或 adaptive 7/10 多数阈值，加上单样本不允许 overlapping spans，意味着两个互相重叠的
  exact 标签不可能同时达到正式阈值。
- MVP 不使用优化器强行消解冲突；任何异常冲突进入 review。
- Weighted interval scheduling 只作为后续消融，不作为正式自动决策。

## 句级状态机

按以下顺序判定：

1. 有效样本不足：`abstained / insufficient_valid_samples`。
2. 空标签集票数达到 negative 阈值：`negative / high_confidence_negative`。
3. 存在 residual majority-supported family：`review / unresolved_span_family`。
4. 存在 exact accepted labels，且没有冲突或未解释 family：`accepted`。
5. 存在 exact candidate 或 majority-supported family，但没有 exact winner：`review`。
6. 其他情况：`abstained / no_consensus`。

继续沿用当前严格句级政策：一句中只要有一个技能族未解决，就不把部分高置信标签写入正式
`annotations.json`；但在审计中保留这些高置信标签。

## CMAS 方法的保留与修正

保留：

- 多次随机采样；
- label-level 置信度；
- sentence/sample-level 置信筛选；
- 不确定样本追加采样；
- 全量原始响应与每票审计。

修正：

- 不采用 CMAS 中“mention 过半后 type 只取最高票、无需再次过半”的宽松规则；exact 边界必须独立达到
  4/5 或 7/10。
- 不把空标签集的平均 label score 设为 0 后直接排除；negative 是正式、多数支持的标签集合状态。
- 不采用 CMAS helpfulness 的纯软提示作为 MVP 门控；若以后引入示例，低帮助示例必须程序化 hard gate。

## 建议数据结构

### Parsed sample

```json
{
  "idx": 1,
  "sample_index": 0,
  "parse_status": "ok",
  "annotation": {
    "has_skill": 1,
    "spans": [
      {
        "label_id": "span:14:30",
        "text": "project planning",
        "start": 14,
        "end": 30
      }
    ]
  }
}
```

### Aggregated label audit

每个 observed label 保存：

- `label_id/text/start/end`
- `exact_votes/exact_probability`
- `sample_indexes`
- `family_id`
- `relation_to_accepted`: `accepted/explained_variant/residual_candidate`
- `status`: `accepted/review/unsupported`

每个 family 保存：

- `family_id`
- `member_label_ids`
- `sample_support/sample_indexes`
- `winner_label_id/winner_votes`
- `resolved`

### Set-level audit

- 每个样本的 exact label set；
- pairwise exact-set Jaccard；
- 与最终候选集的 precision/recall/Jaccard；
- full-set exact match rate；
- label cardinality 分布；
- exact label Bernoulli entropy；
- unresolved family 数量。

这些指标第一版只用于审计和阈值分析，不直接改变状态机。

## 代码重构边界

### 保持不变

- `GeneratePrompts.py`
- `prompt_template/skill_annotation.txt`
- `AskQwen.py`
- `code/common/qwen_client.py`
- raw run、Resume、manifest 和 API 调用合同

### 修改

- `ParseAnswers.py`
  - 保留严格字符对齐；
  - 输出 `label_id`；
  - 停止生成 `bio_tags`；
  - 删除 Token 边界作为 parse 成功条件。
- 新增 `SpanXMLCConsensus.py`
  - 替代活动调用链中的 `TokenBIOConsensus.py`；
  - exact label vote；
  - explained variants；
  - residual overlap families；
  - set-level audit。
- `SelectAnnotations.py`
  - 消费 `span-xmlc-consensus-v1`；
  - 对外 `annotations.json` 结构可保持不变。
- `AdaptiveResample.py`
  - 保留采样和来源验证；
  - 将 6/7/6/7 BIO 阈值改成 6/7 exact/family 阈值。
- scripts/tests/docs/config
  - 活动 consensus schema 改为 `span-xmlc-consensus-v1`。

不建议直接覆盖 v4.2 文件和历史 run。应使用新 schema/version，并允许对已有
`parsed/samples.jsonl` 做离线 re-aggregation。

## 最小可行实验

### E0：冻结基线

- 当前 BIO v4.2 adaptive 结果：214/15/8/89。
- 记录 476 个 accepted candidate、16 个 exact 支持不足候选、13 个受影响 accepted 句。

### E1：朴素 exact baseline

- 不建 family，只按 exact 3/4、6/7 阈值。
- 已有探针结果：204/21/12/89。
- 用它确认“防 Frankenstein”与“静默漏标”两个方向的 trade-off。

### E2：主方案离线重聚合

- 使用同一份 parsed samples；零 API 调用。
- 验证 13 个受影响 accepted 句进入 review 或得到真实 exact winner。
- 验证朴素 exact 错误升为 accepted 的 3 句被 residual family 拦截。

### E3：人工 gold

从 326 句中抽取 50–100 句，覆盖：

- 当前 accepted/negative/review/abstained；
- 长句、多技能、协调结构；
- `C++`、斜杠、连字符等非传统 Token 边界；
- 16 个 exact 支持不足候选；
- 朴素 exact 的全部状态转换句。

人工 gold 必须标 exact `[start,end)`，并独立于两种聚合结果。

### E4：Adaptive 重放与阈值消融

- 3/4 vs 3/5 exact；
- family detector on/off；
- strict sentence gate on/off；
- 5 样本 vs 10 样本；
- 可选 set-Jaccard gate。

## 必要基线

1. 当前 Token/BIO v4.2。
2. 朴素 exact-span majority。
3. sample-set medoid。
4. 主方案 family → exact。
5. 主方案 + adaptive 10。

## 指标

- exact-span micro precision/recall/F1；
- sentence-level exact set match；
- accepted-only precision；
- formal coverage 与 risk–coverage 曲线；
- negative precision/recall；
- review/abstained rate；
- Frankenstein accepted count；
- unresolved-family false-release count；
- API calls、tokens 和 wall time。

## 决策标准

主方案进入实现替换需同时满足：

1. 正式 accepted span 的完整 exact 支持达标率为 100%；
2. residual majority family 未解决却被正式 accepted 的句子为 0；
3. 人工 gold 上 accepted-only exact precision 不低于 BIO 基线，目标至少提升 2 个百分点；
4. adaptive 后 formal coverage 至少 88%，人工复核率不高于 12%；
5. exact-span micro F1 不低于 BIO 基线；
6. MVP 不增加任何 API 调用；
7. 所有历史 raw/parsed 证据可被新聚合器离线重放。

若 precision 提升但 coverage 低于 88%，优先调 family/residual review 规则，不降低 exact 4/5、7/10
正式接受阈值。

