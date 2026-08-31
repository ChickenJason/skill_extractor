# Span XMLC 重放结果

## Self-Annotator → TRF 清洁重建迁移结果（2026-08-31）

### 观察结果

- 326 条 Prompt 校验和全项目60项测试通过，其中包括真实缓存 BERT smoke。
- `run-reaggregate.ps1 -PythonExecutable` 实际零 API 重放完成326句，manifest 保持
  `network_called:false`。
- 实际离线 TRF 验证得到 225 accepted、88 negative、13 unsolved、0 abstained；正式
  corpus 和 pseudo TRF 均为313条，excluded为13条。
- 主候选与 context-only 候选各20项；固定 BERT revision 和权重 SHA256 验证通过。
- 目标阶段的缺失 formal index 测试按预期 fail closed；模板配置不能被目标入口直接运行。
- 最终从项目 `output/` 清理329个文件、126,760,787 bytes；五个输出根目录均只保留
  `.gitkeep`。归档 raw/manifest SHA256、Python 3.10 虚拟环境和固定 BERT 缓存保持不变。

### 解释

这些结果证明 v0.6.0 自我标注可以作为当前 TRF 的确定性输入，且清空项目输出后，测试与
配置不再暗中依赖历史 run。它们不证明在线 Qwen 目标 TRF 的语义质量；326 条在线运行仍需
用户提供凭据并承担 embedding 与约652次 chat 调用。

## v0.6.0 观察结果（2026-08-31）

### 状态与跨度

| 状态 | v0.5.1 对照 | v0.6.0 |
|---|---:|---:|
| accepted | 192 | 225 |
| negative | 88 | 88 |
| unsolved | 46 | 13 |
| abstained | 0 | 0 |

| 接受来源 | v0.5.1 | v0.6.0 |
|---|---:|---:|
| exact_vote | 381 | 491 |
| family_hard_match | 15 | 6 |
| accepted spans | 396 | 497 |

正式 `selected/annotations.json` 包含313句，其中225句为正、88句为负。四个句级 JSONL
均为326行，parsed 样本仍为1630条。派生 manifest 为 `offline_reaggregation`、
`network_called:false`、`source_unchanged:true`。

### 状态迁移与硬匹配

从 v0.5.1 到 v0.6.0 的逐句迁移为：

| 迁移 | 数量 |
|---|---:|
| accepted → accepted | 192 |
| negative → negative | 88 |
| unsolved → accepted | 33 |
| unsolved → unsolved | 13 |

新聚合器实际尝试9个 residual family：6个通过、3个以
`family_anchor_overlap_failed` 保持未解决，4个 family 出现最高 exact 票并列。全部6个
winner 都是 family 中的模型实际跨度，且 winner 与其他成员的每项覆盖率比较均通过。

### 与警戒线比较

正式结果逐项命中只读原型警戒线：225/88/13/0、491个 exact spans、6个 hard-match
spans、497个 accepted spans。预期的33个 `unsolved → accepted` 迁移全部出现，没有需要
解释的偏差。

## v0.6.0 解释边界

新增 accepted 主要来自 exact 阈值由4票降为3票；因此 hard matcher 的使用量从15个跨度
降为6个，而不是因为锚点覆盖脚本更频繁地放行。锚点覆盖只处理三票多数下仍然边界分散
的 residual family。

这些结果证明实现、状态机、审计和离线可重放性符合方案，但没有人工 Gold，不能据此
声称 precision、recall 或技能语义准确率提高。正式语料从280句扩大到313句会改变未来
下游实验分布；本次按约束没有切换 TRF 输入。

---

## v0.5.x 历史结果

## 观察结果

### 解析

| 项目 | 数量 |
|---|---:|
| 句子 | 326 |
| 五次响应样本 | 1630 |
| ok | 1622 |
| recovered | 8 |
| failed | 0 |

8 个恢复样本均使用 `exact_tag_payload_projection`。

### v0.5.0 support-5 对照

| 状态 | 数量 |
|---|---:|
| accepted | 187 |
| negative | 88 |
| unsolved | 51 |
| abstained | 0 |

| 接受来源 | 跨度数 |
|---|---:|
| exact_vote | 377 |
| family_hard_match | 10 |
| 总 accepted spans | 387 |

正式输出共 275 句，其中正样本 187 句、负样本 88 句。

### v0.5.1 support-4 当前结果

| 状态 | 数量 |
|---|---:|
| accepted | 192 |
| negative | 88 |
| unsolved | 46 |
| abstained | 0 |

| 接受来源 | 跨度数 |
|---|---:|
| exact_vote | 381 |
| family_hard_match | 15 |
| 总 accepted spans | 396 |

正式输出共 280 句，其中正样本 192 句、负样本 88 句。

### 输出完整性

- `consensus.jsonl`：326 行。
- `aggregation_audit.jsonl`：326 行。
- `uncertainty_audit.jsonl`：326 行。
- `selected/decisions.jsonl`：326 行。
- v0.5.0 `selected/annotations.json`：275 条。
- v0.5.1 `selected/annotations.json`：280 条。
- 派生 run 不含 `raw/responses.jsonl`，只保存 `raw/source_reference.json`。

## 与警戒线比较

v0.5.1 正式结果与只读原型逐项一致：192/88/46/0，hard matcher 接受 15 个跨度。
相较 v0.5.0，以下 5 句从 `unsolved` 转为 `accepted`：

| idx | support-4 winner | 是否最高票并列 |
|---:|---|---|
| 41 | `multiple data sources` | 否，3:1 |
| 42 | `insights communication` | 是，2:2 |
| 153 | `recruitment` | 否，3:1 |
| 160 | `message or event based architectures` | 是，2:2 |
| 200 | `independently` | 是，2:2 |

idx 80 和 320 的 support-4 family 仍因成员并非两两重叠而失败，未因阈值放宽越过
`family_overlap_chain` 安全门控。

## 解释

Has-skill 门控先排除了 88 个高置信负样本，并阻止存在性不足四票的句子进入跨度聚合。
当前 exact 投票提供 381 个正式跨度；support-4/5 硬匹配提供 15 个跨度。共尝试 24 个
hard-match family：support-5 接受 10 个，support-4 接受 5 个；另外 6 个重叠链和 3 个
相似度不足 family 保持未解决。最高票并列发生于 idx 42、160、200，均由紧凑度规则
选择较短的模型原始提案。

`unsolved` 从 51 降至 46，变化完全来自 5 个 support-4 family；负样本和存在性门控结果
没有变化。这说明本次修改只影响预定的 family 裁决层。

以上解释是对观察结果的机制性说明，不是基于人工 Gold 的准确率结论。

---

# TRF 前三阶段结果

## 观察结果

### 语料与候选

| 项目 | 数量/结果 |
|---|---:|
| 源 decisions | 326 |
| 正类 / 负类 / 排除 | 192 / 88 / 46 |
| 正式 corpus | 280 |
| accepted spans | 396（381 exact，15 hard-match） |
| 主候选统计词项 | 1408 |
| 主候选论文公式通过 / 方向门控通过 | 1133 / 936 |
| context-only 统计词项 | 1101 |
| context-only 论文公式通过 / 方向门控通过 | 800 / 607 |

主 Top-10：`English, skills, management, projects, sales, communication, process, product,
software, tools`。

context-only Top-10：`projects, product, years, taking, skills, Use, advantage, basic, global,
include`。

### BERT 分配与审计

- 正类主分配 960 项，context-only 分配 960 项；每句各 5 项。
- 88 条负类的两个列表全部为空。
- 模型维度 768；revision 为
  `cd5ef92a9fb2f889e972770a36d4ed042daf221e`。
- `model.safetensors` SHA256：
  `1d8bdcee6021e2c25f0325e84889b61c2eb26b843eef5659c247af138d64f050`。
- v4 与 v5 的 19 个产物文件 hash 完全相同。
- 固定 20 条语义样本已生成，状态为 `pending_manual_review`、`passed:false`。

## 解释

观察结果证明数据合同、MI/门控、BERT offset 对齐、Top-5 数量、负类空集、源安全和
字节级可重复性均满足计划。主候选整体具有明显技能语义；context-only 榜单中存在
`taking`、`Use` 等较弱项，这是从技能跨度外上下文统计得到的真实诊断结果，不应在没有
下游 Gold 的情况下人为替换。

本实验没有证明 TRF 语义准确率或最终技能预测性能。尤其是 token-level 最近距离可能
偏向通用上下文词，必须先人工审阅 `audit/semantic_review_sample.jsonl`，再通过后续 Gold
或下游任务决定是否调整候选、池化或检索阶段。

---

# 目标句 TRF 提取器实现结果

## 观察结果

- 固定源合同读取 280 个 demonstration（192 accepted、88 negative）和 326 个
  leave-one-out 目标；该次历史实现验收使用主候选库 10 项。
- mock independent 与 leave-one-out 集成均完成 K=50、k=16、两轮消息和只读验证；
  leave-one-out 正式池目标均排除了自身。
- Embedding mock 按 20 条分批，维度/NaN/Inf/零范数/响应索引错误均有 fail-closed 门控。
- partial → Resume 保留成功目标；一次解析失败在普通 Resume 中没有重调，增加
  RetryFailed 后完成。
- 相同目标集可复用 demonstration 与 target vectors；目标集变化时只复用固定
  demonstration vectors。
- 加入完整 runner 后，全项目 54 项测试通过，既有自我标注和 BERT TRF 结果未回归。

## 解释

这些观察支持的是运行合同、检索排序、消息顺序、解析、恢复和审计实现正确。mock 输出
不能证明 `qwen3.7-text-embedding` 的真实邻居质量，也不能证明开放 TRF 的语义质量。
由于当前没有云端凭据，本轮没有生成正式目标 TRF、调用成本或真实一致性统计；这些结果
必须在 Limit 10 真实 smoke 后再记录。

---

# 完整 TRF 重跑入口结果

## 观察结果

- 父级入口能从本次离线子 run 读取四个必要产物并生成逐文件 SHA256 锁定配置。
- PrepareOnly 后父状态为 partial，离线阶段 completed、目标阶段 partial；同一父 ID Resume
  后完成目标抽取与联合验证。
- 联合验证同时重跑离线/目标只读验证器，并核对父输出、两个子 manifest、网络标志和
  动态 source run-id。
- 无 Limit 的在线调用缺少 ConfirmFullRun 时在创建父目录前失败。

## 解释

这些结果证明完整重跑的阶段衔接、来源绑定、恢复和联合审计正确。测试没有重新请求真实
Qwen，也没有生成正式 326 条目标结果；因此不提供云端调用成功率、TRF 语义质量或最终
一致性指标。

---

# TRF Top-50 扩展结果

## 观察结果

- 新 run：`output/trf_runs/trf-offline-bert-base-cased-top50-v1/`，状态 `completed`，
  只读验证状态 `validated`，`network_called:false`。
- 主候选和 context-only 候选均为 50 项且无重复；原主 Top-10 保持为新榜单前缀。
- 主 Top-50 为：`English, skills, management, projects, sales, communication, process,
  product, software, tools, leadership, technical, years, basic, design, independently,
  strategic, taking, user, development, Engineering, Use, advantage, agile, analysis,
  evaluation, global, include, like, microscopy, set, speak, supply, data, marketing,
  Business, Excellent, Experience, Lead, Manager, Product, SAP, SCR, Teamcenter, algorithm,
  algorithms, analytics, applications, architecture, areas`。
- 正式 corpus 仍为 280 条；192 条 accepted 各有 5 个主伪 TRF 和 5 个 context-only
  伪 TRF，88 条 negative 两组均为空。
- 与历史 v4 Top-10 相比，192/192 条 accepted 的主 Top-5 和 context-only Top-5 均发生
  变化；新主伪 TRF 实际使用 34 个不同候选，context-only 实际使用 35 个不同候选。
- corpus SHA256 未变化；新 candidate SHA256 为
  `cff436daebcec1b5e59fce0a1c938cde9f87c13cedef2bcd9b87945453f6d199`，新 pseudo TRF
  SHA256 为 `8725d66a6a57aeba6b7c3e8c6bfea16d59433b6f148513277cfce841b3c45cd0`。

## 解释边界

扩大候选池已改变句级 Top-5 的可选范围，但没有改变每句输出数量。它只证明新的结构、
数量、来源锁定和离线算法通过验收；本轮没有重新调用 Qwen，因此既有
`trf-complete-v1` 仍是历史 Top-10 在线结果，不是 Top-50 目标句结果。

---

# TRF Top-20 折中结果

## 观察结果

- 新离线 run：`output/trf_runs/trf-offline-bert-base-cased-top20-v1/`，状态
  `completed`，只读验证为 `validated`，且 `network_called:false`、`source_unchanged:true`。
- 主 Top-20：`English, skills, management, projects, sales, communication, process,
  product, software, tools, leadership, technical, years, basic, design, independently,
  strategic, taking, user, development`。
- context-only Top-20：`projects, product, years, taking, skills, Use, advantage, basic,
  global, include, like, process, speak, strategic, supply, English, Excellent, Experience,
  Lead, Manager`。
- 192条 accepted 各有两组 Top-5，88条 negative 两组为空；主伪 TRF 实际使用18个
  不同候选，context-only 也使用18个。
- 与历史 Top-10、Top-50 比较，192/192条 accepted 的两组 Top-5 均发生变化，说明候选池
  大小会实质改变 BERT 最近距离分配，不只是扩充未使用的尾部候选。
- manifest、候选、伪 TRF SHA256 分别为
  `3267d314bb96a8102dd4780f5604f72239f18d13f6c365eeeec9fed53228bd95`、
  `356b993a1b900265be5470a7341e54203d8da57373222a80174c753bb94492b3`、
  `8f4ae0d191a8655bfd877be6b47c7b1d9d814bc518ff2db0ed2b0acaa7430f47`。

## 解释

结果证明 Top-20 活动合同、离线分配、来源锁定和下游 mock 集成正确。它尚未证明 Top-20
在线目标 TRF 优于 Top-10 或 Top-50；必须以新 full run-id 重新执行 Qwen 阶段并进行同样
的人工审阅和伪标签一致性对比后才能判断。
