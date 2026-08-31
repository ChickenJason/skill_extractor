# Span XMLC v0.6.0 实施记录

## Self-Annotator → TRF 清洁重建迁移（2026-08-31）

1. `config/trf.json` 已迁移到 v0.6.0 多数票重放，固定 225 accepted、88 negative、
   13 unsolved、313 formal。
2. 目标 TRF 从 decisions 派生 `accepted/negative` 正式索引，并要求 corpus、pseudo TRF
   与正式索引集合完全一致；embedding 验证按实际 demonstration 数量执行。
3. `config/trf_target.json` 改为不可直接运行的完整流水线模板；父 runner 生成带实际路径和
   SHA256 的 `generated-run-config`。单独目标入口新增 `-ConfigPath`。
4. `run-reaggregate.ps1` 新增 `-PythonExecutable`，Self-Annotator 与 TRF 可统一使用项目
   Python 3.10 环境。
5. TRF 单元和集成测试在临时目录从归档响应重放，并构造 validator-compatible 离线夹具，
   不再复制或读取项目 `output/` 历史 run。
6. TRF v2 路径合同取消配置中的固定源 ID：`SelfAnnotatorRunId` 绑定
   `output/runs/<id>` 输入，`RunId` 仅命名 TRF 父输出及 `-offline`、`-target` 子运行。

## 已完成模块

1. `SpanXMLCConsensus.py` 与配置
   - 保持固定五样本结构、解析和严格句级裁决。
   - 将 Has-skill、exact 接受和 hard-match family 统一为3票，负样本保持4票。
2. `hard_span_matcher.py`
   - 删除 token containment、token IoU 和字符二元组 IoU 加权分数。
   - 按最高 exact 票选锚点，使用字符交集/锚点长度和锚点 `word_units` 动态阈值。
   - 并列时按最小覆盖率、平均覆盖率、紧凑度和位置确定模型原始 winner。
   - 允许三个有效响应运行，成员对锚点覆盖不足时 fail closed。
3. `AskQwen.py` 与 manifest
   - 继续强制五次采样，活动 schema/version 升级到 v2/v0.6.0。
   - 删除旧权重校验，严格校验锚点覆盖配置字段并保存完整 aggregation。
4. `SelectAnnotations.py`
   - 状态改为 `accepted/unsolved/abstained/negative`，保持字符串跨度正式接口。
5. `ReaggregateRun.py`
   - 新增零 API 重放；记录源绝对路径、SHA256、`network_called:false` 并复核源不可变。
6. 脚本、测试和文档
   - 活动入口改为 Parse → SpanXMLCConsensus → Select。
   - 删除活动 BIO/Adaptive 模块、入口和测试，历史产物保留。

## 验收顺序

1. 编译检查。
2. Prompt validation 与单元测试。
3. 326 句离线重放。
4. 校验状态计数、接受来源、行数、正式接口和源 SHA256。
5. 扫描活动代码与脚本中的遗留入口。
6. 将 v0.6.0 派生 run 与不可变的 v0.5.1 run 做逐句差异比较；TRF 输入保持旧 run。

---

# TRF 前三阶段实施记录

## 已完成模块

1. `config/trf.json`：固定源 run、状态计数、tokenizer、MI/rho、20 项候选库、BERT
   revision、句级 Top-5 和 seed。
2. `code/trf/BuildTRFCorpus.py`：从 decisions 构建 280 条正式 corpus 和 46 条排除审计；
   fail-closed 校验全部字符跨度，并生成 10/5/5 人工检查样本。
3. `code/trf/ExtractTRFCandidates.py`：技术词 unigram、匿名占位符删除、二值文档 MI、
   论文频率比、正向率门控、context-only 等长字符屏蔽和确定性排序。
4. `code/trf/AssignPseudoTRFs.py`：固定 safetensors、独立候选池化、fast-tokenizer offset
   对齐、词级 subword mean、最小欧氏距离、Top-5 决胜和负类空集。
5. `code/trf/common.py`：不可覆盖 run、源前后 SHA256、阶段状态、依赖/实现/权重/输出审计。
6. `RunTRFOffline.py`、`ValidateTRFRun.py` 与 `run-trf-offline.ps1`：统一执行和只读验收。
7. `tests/test_trf_pipeline.py`：语料、跨度、tokenizer、MI、mask、BERT 假模型、真实模型、
   离线模型解析和阶段一/二集成测试。

## 发布规则

只有三个阶段均成功、源文件复核不变且全部输出 hash 写入 manifest 后，run 才能标记
`completed`。失败 run 保留审计且禁止续写；必须使用新 run-id。算法通过后仍将语义验收
保持为 pending，直至人工检查 20 条固定样本。

---

# 目标句 TRF 提取器实施记录

## 已完成模块

1. `config/trf_target.json`：固定 Top-20 离线源 SHA256 与候选数合同、两个 Qwen model ID、1024/20、
   K=50/k=16、temperature 0 和人工样本数。
2. `code/trf_target/pipeline.py`：严格目标输入、源 demonstration join、L2/余弦检索、
   leave-one-out、自包含 Prompt、严格 JSON/NFKC 解析和开放 TRF 审计。
3. `code/trf_target/online.py`：逐批原子 embedding checkpoint、句子 hash 缓存、精确复用、
   两轮消息历史、成功跳过与显式失败重试。
4. `code/trf_target/RunTargetTRF.py`：不可覆盖 run、partial/resume 状态机、网络/完整运行
   双门控、source/implementation/output hash、脱敏错误和诊断发布。
5. `code/trf_target/ValidateTargetTRFRun.py`：从固定向量重新计算检索和 Prompt，并复核
   raw/parsed/summary/review 与 manifest hash ledger。
6. `code/common/qwen_client.py`：在不破坏 `embeddings()` 返回接口的前提下增加向量、
   attempts、response id、usage、批次顺序与索引审计。
7. `scripts/run-target-trf.ps1` 与 `tests/test_trf_target_pipeline.py`：完整 CLI 参数、两模式
   mock 集成、恢复/失败重试、缓存复用和 fail-closed 合同测试。

## 发布规则

结构验收完成只代表代码、缓存、检索、Prompt、解析和审计链正确。当前没有真实云端响应
产物；取得凭据后必须先运行 Limit 10 smoke，检查 Prompt、调用量和 20 条人工样本，再
决定是否执行完整 leave-one-out。云端重新调用不要求字节复现；相同向量/响应快照的派生
检索、Prompt、解析和 summary 必须一致。

---

# 完整 TRF 重跑实施记录

1. `code/trf_full/RunFullTRF.py`：统一执行离线三阶段、生成动态目标配置、调用目标抽取，
   支持 PrepareOnly、Resume、RetryFailed、Embedding 复用和完整运行确认。
2. `code/trf_full/common.py`：父 run 的不可覆盖合同、实现 hash、三阶段状态、网络汇总、
   子 manifest hash 和输出 ledger。
3. `code/trf_full/ValidateFullTRFRun.py`：联合复核父输出、配置/输入 hash、两个子验证器、
   动态 source 绑定、网络审计和联合 summary。
4. `scripts/rerun-all-trf.ps1`：提供单条 PowerShell 完整重跑入口。
5. 目标配置加载器允许由完整 runner 生成新的、哈希锁定的离线 source，并同步写入候选
   数量合同；默认独立运行配置固定引用 `trf-offline-bert-base-cased-top20-v1`。跨 run
   Embedding 复用按模型、维度和 corpus SHA256 门控。
6. `tests/test_trf_full_pipeline.py`：覆盖父级 PrepareOnly/Resume、动态配置、两轮 mock、
   子 run 追踪、联合验证及无限制在线确认门控。

---

# Top-50 扩展实施记录

1. `config/trf.json` 将 `candidates.max_trfs` 从 10 改为 50；候选统计与排序算法不变。
2. `trf_target` 新增 `candidates.expected_main_trfs`，同时校验主榜单、context-only 榜单和
   `max_trfs`，防止误接旧 Top-10 产物。
3. `trf_full` 从本次离线配置自动传播候选数到动态目标配置，不再依赖模板中的静态假设。
4. 新离线 run 为 `trf-offline-bert-base-cased-top50-v1`；不联网、不调用 Qwen。
5. 测试覆盖 50 项数量、唯一性、方向门控、原 Top-10 前缀与下游 source contract。

---

# Top-20 折中实施记录

1. `config/trf.json` 将活动 `candidates.max_trfs` 从50收缩为20，统计与排序算法不变。
2. `config/trf_target.json` 的来源、候选数与四个产物 SHA256 锁定到
   `trf-offline-bert-base-cased-top20-v1`。
3. 完整 runner 的测试副本切换到 Top-20；动态配置仍从离线配置传播候选数。
4. 目标合同测试同时证明历史 Top-10 与 Top-50 来源会 fail closed。
5. 本次只重建离线 Top-20 产物，不覆盖历史 run，也不调用 Qwen。
