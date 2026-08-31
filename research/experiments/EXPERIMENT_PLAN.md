# Span XMLC v0.6.0 多数票与锚点覆盖实验计划

## Self-Annotator → TRF 清洁重建实验（2026-08-31）

本轮验证从空 `output/` 出发，只使用归档五次响应重建 v0.6.0 自我标注，再以固定
`bert-base-cased` 生成离线 TRF，并为后续 326 条 leave-one-out 在线目标抽取生成动态来源配置。

验收条件为：归档 SHA256 不变；自标注得到 225/88/13/0 和 313 条正式样本；离线 TRF
得到 313 条 corpus、13 条排除记录、两组各20个候选和313条伪 TRF；目标阶段不再依赖
硬编码的280条示例或历史 `output/`。本轮不执行付费的 embedding/chat 请求。

## 目标

验证自我标注器能在不调用 API 的前提下，将存在性、exact span 和 residual family 统一为
3/5多数票，并以最高频 span 的长度动态锚点覆盖率替代旧复合相似度。v0.5.1
`span-xmlc-base5-replay-support4-v1` 作为不可变对照。

## 输入

- 数据集：`data/raw/skill_sentences.json`，326 句。
- Prompt：`skill-inline-annotation-v4.1`，保持不变。
- 原始响应：`archive/self-annotation-history/inline-v4-2-full-20260821-001/raw/responses.jsonl`。
- 每句固定五个响应，共 1630 个样本；不读取任何追加采样响应。

## 实验条件

1. 有效样本少于3时 `abstained`，`no_skill` 至少4票时 `negative`。
2. `has_skill` 至少3票后进入跨度聚合，exact span 三票直接接受。
3. residual family support 1–2 只审计，support 3–5 进入硬匹配。
4. 以最高 exact 票 span 为锚点，计算 $R(a,q)=|a\cap q|/|a|$。
5. 阈值为 $T(n)=\min(0.85,0.45+0.05(n-1))$，$n$ 只取锚点 `word_units`。
6. 并列时依次使用最小覆盖率、平均覆盖率、紧凑度和字符位置。
7. 任一成员未达阈值、正式 family 未解决或标签冲突都触发句级 `unsolved`。

## 回归警戒线

v0.5.1 对照与 v0.6.0 只读原型给出的预期是：

| 状态/来源 | v0.5.1 对照 | v0.6.0 预期 |
|---|---:|---:|
| accepted | 192 | 225 |
| negative | 88 | 88 |
| unsolved | 46 | 13 |
| abstained | 0 | 0 |
| exact_vote spans | 381 | 491 |
| family_hard_match spans | 15 | 6 |
| accepted spans | 396 | 497 |

若正式结果不同，必须逐项定位到句子、family 和比较值并在 `RESULTS.md` 解释。

## 验证范围

- 配置、解析、门控、exact、family、hard matcher、selection 和 manifest 单元测试。
- Prompt 及 326 条输入数据校验。
- 源 raw/manifest SHA256 前后不变。
- 派生 manifest 明确为 `network_called:false`。
- 三类 aggregation 输出和全部 decision 均各 326 行。
- 所有 support 3–5 family 的锚点、并列候选、覆盖率、阈值和失败原因。
- v0.5.1 到 v0.6.0 的逐句状态迁移；预期33句从 `unsolved` 转为 `accepted`。

人工 Gold 标注、论文级 precision/recall/F1 和线上新采样不在本实验范围。

---

# TRF 前三阶段实验计划（2026-08-28）

## 目标与边界

在不修改源 self-annotation run 的前提下，实现并验证三个离线阶段：TRF corpus、
unigram/MI 全局候选库、固定 revision BERT Top-5 伪 TRF。目标句检索、TRF Prompt、
Qwen 调用、helpfulness 和最终技能预测均不在本实验中。

## 固定条件

- 权威输入：`output/runs/span-xmlc-base5-replay-support4-v1/selected/decisions.jsonl`。
- 固定计数：326 = 192 accepted + 88 negative + 46 unsolved + 0 abstained；正式语料 280。
- 候选：大小写敏感 unigram、NFKC、scikit-learn stopwords、排除数字和匿名占位符，
  `rho=3`，主结果增加正向文档率门控，并保留论文原式审计。
- 模型：`bert-base-cased@cd5ef92a9fb2f889e972770a36d4ed042daf221e`，CPU，
  token-level 欧氏距离，Top-5，seed 42，禁止截断。
- 环境：Python 3.10、NumPy 1.26.4、scikit-learn 1.6.1、Torch 2.5.1+cpu、
  Transformers 4.57.1。

## 验收标准

1. 三个源文件执行前后 SHA256 不变，全部 span 可精确回指原句。
2. 当前活动主榜单、context-only 榜单各 20 项；正式候选不得含匿名占位符。
3. 192 条正类各有两组 5 项，88 条负类两组均为空，全部最近 token offset 有效。
4. manifest 固定配置、实现、依赖、模型 revision、权重和全部输出 SHA256。
5. 同配置两次离线运行的全部 19 个产物 SHA256 一致。
6. 10 exact + 5 hard-match + 5 negative 的语义样本在人工确认前保持
   `pending_manual_review`，不得宣称语义准确率。

---

# 目标句 TRF 提取器实验计划（2026-08-28）

## 目标与边界

在固定 Top-20 离线产物之上实现论文公式（3）的目标句路径：Qwen Embedding 检索
K=50 → k=16 示例，再在一条两轮消息历史中执行 Skill 类型判断和开放 TRF 抽取。
本实验不实现 helpfulness、最终技能预测或最终 self-consistency。

## 固定条件

- 源 run：`trf-offline-bert-base-cased-top20-v1`；正式 Prompt 只读取主伪 TRF。
- 模式：严格两字段 independent 输入，或从 326 条 decisions 派生 leave-one-out 目标。
- Embedding：`qwen3.7-text-embedding`、1024 维、batch 20、原句直编码。
- 检索：未舍入余弦降序取 50，再按 existence score、余弦、idx 取 16；不设类别配额。
- Chat：`qwen3.7-plus-2026-05-26`、thinking false、temperature 0、结构化 JSON。
- 成功结果断点续跑时不重调；失败解析只在显式 `RetryFailed` 时重调。

## 验收标准

1. 源 manifest、corpus、候选、伪 TRF、decisions 的 SHA256 前后不变。
2. 每个目标保存 50 邻居、16 示例和自身排除审计；Prompt 只含主 TRF。
3. Embedding 缓存校验句子 hash、维度、有限值、顺序和批次证据，并按批原子提交。
4. 第二轮保留第一轮 user/assistant 消息；开放 OOV 不过滤，非法 JSON fail closed。
5. partial/resume/retry/reuse 不重复调用成功目标；完整在线 run 必须显式确认。
6. mock 集成覆盖两种模式；真实运行先限 10 条 smoke，再决定是否运行 326 条。
7. 20 条人工样本确认前保持 `pending_manual_review`；一致性指标不得解释为 Gold 性能。

---

# 完整 TRF 一键重跑实验计划（2026-08-28）

## 最小实验

将离线三阶段和目标句阶段放入同一父运行合同。父 run 必须生成两个新子 run，并把本次
离线 corpus、候选和伪 TRF 的实际 SHA256 写入动态目标配置，禁止目标阶段暗中引用历史
run。最小测试使用已验证 Top-20 run 的只读副本代替耗时离线执行，目标阶段使用 mock provider，
覆盖 PrepareOnly → Resume → completed → joint validation。

## 全量合同

- 输入仍是 `span-xmlc-base5-replay-support4-v1` 的 326 条 decisions。
- 离线子 run 完整执行 corpus、MI 候选和固定 BERT Top-5。
- 目标子 run 在 leave-one-out 下处理 326 条目标，每条检索 50→16 并执行两轮 chat。
- 无限制在线运行必须同时提供 AllowNetwork 和 ConfirmFullRun。
- 父、离线子、目标子三层 run-id 均不可覆盖；父 manifest 固定子 manifest SHA256。
- 联合验证分别通过离线验证器和目标验证器，并确认动态配置指向本次离线子 run。

真实全量在线调用仍需凭据；在此之前只声明代码与 mock 贯通完成，不声明语义结果。

---

# TRF 全局候选 Top-50 扩展实验（2026-08-28）

## 变更假设

将全局主候选和 context-only 诊断候选由 10 扩展为 50，可以为每条 demonstration 的
BERT 最近距离选择提供更宽的候选池。该变更不调整 MI、rho、方向门控、BERT revision、
句级 Top-5、目标检索 K=50/k=16 或两轮 Qwen Prompt。

## 验收标准

1. 两个候选库均恰好 50 项、无重复，全部通过既有方向门控；原 Top-10 保持为新榜单前缀。
2. 192 条 accepted 仍各输出两组 Top-5，88 条 negative 仍输出空集。
3. 新 run 使用独立 run-id，旧 v4/v5 与 `trf-complete-v1` 保持不可变。
4. 目标配置同时锁定候选数量和新 run 的逐文件 SHA256，旧 Top-10 来源 fail closed。
5. 本轮不调用 Qwen；历史在线结果不得表述为已用 Top-50 重算。

---

# TRF 全局候选 Top-20 折中实验（2026-08-28）

## 变更假设

Top-50 在线结果表现出候选词过泛和伪标签一致性下降，因此将活动候选池收缩为 Top-20，
在保留 `leadership`、`technical`、`design`、`independently`、`strategic`、`development`
等扩展词的同时，排除 Top-20 之后较弱或领域偶然性更强的候选。MI、rho、方向门控、
BERT revision、句级 Top-5 和目标检索 K=50/k=16 均不改变。

## 验收标准

1. 主候选和 context-only 候选均恰好20项、无重复并通过方向门控。
2. 原 Top-10 保持为主 Top-20 的前缀；192条 accepted 仍各输出两组 Top-5。
3. 新离线 run 独立生成并锁定 SHA256；Top-10 和 Top-50 来源均被活动目标配置拒绝。
4. 本轮只生成离线 Top-20，不自动调用 Qwen；在线语义优劣需新的完整 run 才能判断。
