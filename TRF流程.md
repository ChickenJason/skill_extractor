# 3 Trf-based prediction（TRF 特征提取与 Skill 精确跨度预测）

## 3.0 总体结构

TRF 模块分为三种运行模式：

- Offline：根据自标注示例构建正式语料、全局 TRF 候选库，并为示例分配伪 TRF
- Target：为目标句检索示例、提取开放词表 TRF，并预测精确 Skill 字符跨度
- Full：在同一个 run-id 下依次执行 Offline、Target 和联合校验

统一入口为 scripts/trf/run.ps1，核心配置为 config/trf.json。

代码结构：

```text
code/trf/
├── offline/
│   ├── runner.py
│   ├── build_corpus.py
│   ├── extract_candidates.py
│   ├── assign_pseudo_trfs.py
│   ├── common.py
│   └── validator.py
├── target/
│   ├── runner.py
│   ├── pipeline.py
│   ├── online.py
│   ├── diagnostics.py
│   ├── common.py
│   └── validator.py
└── orchestration/
    ├── runner.py
    ├── common.py
    └── validator.py
```

---

## 3.1 scripts/trf/run.ps1（TRF 模块调度文件）

### Input

- 运行模式：Offline、Target 或 Full
- 运行标识：RunId
- TRF 配置文件：默认使用 config/trf.json
- 可选目标句文件：Input
- 可选 Demonstration manifest：Demonstrations
- Target 模式所需的生成配置：GeneratedConfigPath
- 可选反馈文件：Feedback
- 目标模式：
  - independent：处理独立目标句
  - leave-one-out：以 Demonstration 为目标，并排除自身
- 运行控制参数：
  - AllowModelDownload
  - AllowNetwork
  - PrepareOnly
  - Resume
  - RetryFailed
  - ReuseEmbeddingsFrom
  - ConfirmFullRun

### Process

- 当 Mode=Offline 时，调用 code/trf/offline/runner.py
- 当 Mode=Target 时，调用 code/trf/target/runner.py
- 当 Mode=Full 时，调用 code/trf/orchestration/runner.py
- 根据运行模式检查参数是否合法
- 未指定 AllowNetwork 时，禁止发起 Qwen 网络请求
- 对无限制的完整在线运行，要求显式设置 ConfirmFullRun
- Target 模式不能直接使用模板配置，必须使用 Offline 完成后生成并锁定的 Target 配置

### Output

所有 TRF 运行统一写入：

```text
output/trf/<run-id>/
```

根据模式不同，其下可能包含：

```text
offline/
target/
manifest.json
target-config.generated.json
validation.json
```

---

# 3.2 Offline TRF（构建全局 TRF 与 Demonstration 伪 TRF）

## 3.2.1 code/trf/offline/runner.py

### Input

- TRF 配置：config/trf.json
- Demonstration manifest：
  data/processed/demonstrations/v1/manifest.json
- 可选参数：--allow-model-download

### Process

依次执行三个确定性阶段：

1. build_corpus
2. extract_candidates
3. assign_pseudo_trfs

每个阶段开始和结束时都会更新 manifest.json。任一阶段失败后，运行会被标记为 failed，并记录失败阶段和错误信息。

### Output

```text
output/trf/<run-id>/offline/
├── manifest.json
├── source/
├── corpus/
├── candidates/
├── demonstrations/
└── audit/
```

---

## 3.2.2 code/trf/offline/build_corpus.py

### Input

- Demonstration manifest
- 自标注决策记录
- 固定的记录数量合同

每条输入记录必须包含：

- dataset_id
- record_id
- source_sha256
- idx
- sentence
- status
- accepted 记录的精确 Skill 跨度

### Process

- 检查记录身份、原句哈希、索引唯一性和状态合法性
- 校验 accepted 记录中的 Skill 字符跨度
- 将记录转化成为正式的语料与排除的预料：
  - 将 accepted 和 negative 纳入正式语料
    - 为 accepted 记录设置：
      - pseudo_types = ["Skill"]
      - eligible_for_statistics = true
      - eligible_as_demonstration = true
    - 为 negative 记录设置：
      - pseudo_types = []
      - skill_spans = []
  - 将 unsolved （无法判决的样本——无法判别是否有技能以及无法确定精准的技能跨度）和 abstained （有效回答不足）排除
- 构造固定的 20 条人工审核样本，用来检查自我标注器的生成质量：
  - 11 条精确投票 accepted
  - 4 条 hard-match accepted
  - 5 条 negative

### Output

```text
output/trf/<run-id>/offline/corpus/
├── records.jsonl
├── excluded.jsonl
└── summary.json
```

- records.jsonl：313 条正式 TRF 语料
- excluded.jsonl：13 条 unsolved/abstained 记录
- summary.json：状态数量、跨度数量和接受方式统计

人工审核样本：

```text
output/trf/<run-id>/offline/audit/manual_review_sample.jsonl
```

---

## 3.2.3 code/trf/offline/extract_candidates.py

### Input

- 正式语料：offline/corpus/records.jsonl
- Tokenization 配置：
  - Unicode NFKC 规范化
  - 保留 . + # / - 等技术字符
  - 排除包含数字的 token
  - 排除匿名占位符
  - 使用 scikit-learn 英文停用词
- TRF 筛选参数：
  - rho = 3.0
  - 每类最多 20 个 TRF
  - 启用方向门控

### Process

分别构建两个候选库：

- Main TRF：
  - 直接从完整句子提取 token
  - 允许 Skill 实体自身参与统计
- Context-only TRF：
  - 先屏蔽已标注 Skill 跨度
  - 只从剩余上下文中提取 token

针对每个 token：

- 统计 accepted 和 negative 中的词频及文档频率
- 计算二元互信息 MI
- 应用论文公式筛选：
  - token 必须在正类中出现
  - negative_count / positive_count <= rho
- 应用方向门控：
  - accepted 中的文档出现率必须高于 negative
- 按以下顺序排序：
  - MI 降序
  - accepted 文档频率降序
  - token 字典序
- 分别选出 Top-20 Main TRF 和 Top-20 Context-only TRF

### Output

完整统计文件：

```text
offline/candidates/
├── main_all_metrics.jsonl
├── main_paper_formula.jsonl
├── main_directional.jsonl
├── context_only_all_metrics.jsonl
├── context_only_paper_formula.jsonl
└── context_only_directional.jsonl
```

正式候选库：

```text
offline/candidates/
├── trfs.json
├── context_only_trfs.json
├── selected_trfs.json
└── summary.json
```

其中 selected_trfs.json 同时保存：

- 最终 20 个 Main TRF
- 最终 20 个 Context-only TRF
- 每个候选的 MI、词频、文档频率、方向门控结果和排名

---

## 3.2.4 code/trf/offline/assign_pseudo_trfs.py

### Input

- 正式语料：offline/corpus/records.jsonl
- TRF 候选：offline/candidates/selected_trfs.json
- 固定 BERT 模型：
  - bert-base-cased
  - 固定完整 commit revision
  - hidden size：768
  - CPU 运行
  - Top-K：5

### Process

- 检查或下载固定版本的 BERT 快照
- 对模型权重计算 SHA256
- 为每个 TRF 候选生成 BERT 向量
- 对 accepted 示例：
  - 对句子进行词法切分
  - 使用 BERT offset mapping 对齐原句字符位置和 subword
  - 对每个词的 subword 向量取平均
  - 计算每个 TRF 候选与句子中最近词向量的欧氏距离
  - Main 候选选出 Top-5
  - Context-only 候选选出 Top-5
- 对 negative 示例：
  - trfs = []
  - context_only_trfs = []
- 全过程固定随机种子、CPU 线程数和确定性算法

### Output

```text
offline/demonstrations/
├── pseudo_trfs.jsonl
├── model_files.json
└── summary.json
```

- pseudo_trfs.jsonl：每条 Demonstration 对应的 Top-5 Main TRF 和 Top-5 Context-only TRF
- model_files.json：模型版本、快照路径和权重哈希
- summary.json：正负样本数量、TRF 分配数量及人工语义审核状态

语义审核样本：

```text
offline/audit/semantic_review_sample.jsonl
```

---

## 3.2.5 code/trf/offline/common.py

### Process

- 解析和验证 Offline 配置
- 创建运行目录
- 冻结源文件路径、大小和 SHA256
- 管理三个 Offline 阶段的状态
- 锁定实现文件与依赖版本
- 防止覆盖已有运行
- 汇总所有输出文件的 SHA256
- 在源文件发生变化时 fail closed

### Output

```text
offline/manifest.json
offline/source/snapshot.json
```

---

## 3.2.6 code/trf/offline/validator.py

### Input

- 已完成的 Offline 运行目录
- 原始 TRF 配置
- Offline manifest 和全部输出文件

### Process

- 检查运行状态是否为 completed
- 检查三个阶段是否全部完成
- 重新计算输出文件哈希
- 检查源文件是否保持不变
- 检查正式语料、候选 TRF 和伪 TRF 数量
- 检查 BERT revision、权重文件和环境版本

### Output

- 在终端输出 Offline 校验摘要
- 不修改已有运行文件

---

# 3.3 Target TRF（目标 TRF 抽取与 Skill 预测）

## 3.3.1 code/trf/target/runner.py

### Input

- Offline 生成的 target-config.generated.json
- 目标运行模式：
  - independent
  - leave-one-out
- 独立目标句 JSON
- Offline 正式语料、候选 TRF 和伪 TRF
- Qwen API 配置
- 可选 embedding 复用运行
- 可选反馈 JSON/JSONL

反馈文件目前只会进行身份校验和哈希锁定，不参与 TRF 计算。

### Process

按以下顺序执行：

1. 规范化目标记录
2. 生成或复用 Demonstration embedding
3. 生成或复用目标 embedding
4. 检索候选 Demonstration
5. 构造两轮 TRF Prompt
6. 调用模型提取目标开放 TRF
7. 构造独立的 Skill 精确跨度 Prompt
8. 调用模型预测最终 Skill 跨度
9. 汇总失败、审核事件和诊断结果
10. 更新 manifest

使用 PrepareOnly 时，只执行目标准备、embedding、检索和 Prompt 构造，不调用 TRF 抽取及 Skill 预测。

### Output

```text
output/trf/<run-id>/target/
├── manifest.json
├── source_snapshot.json
├── targets/
├── embeddings/
├── retrieval/
├── prompts/
├── raw/
├── parsed/
├── prediction/
└── audit/
```

---

## 3.3.2 code/trf/target/pipeline.py

### Input

- 规范化目标句
- 326 条 Demonstration
- Demonstration 和目标的 1024 维 Qwen embedding
- Demonstration 伪 TRF
- Main TRF 候选库

### Process

#### 目标构造

- independent：从外部 JSON 构造目标记录
- leave-one-out：将 Demonstration 自身作为目标

#### 示例检索

- 使用余弦相似度检索 Top-50 邻居
- Leave-one-out 模式通过完整记录身份排除自身
- 从 Top-50 中按以下顺序选出 16 条：
  - existence_score 降序
  - 未舍入的余弦相似度降序
  - demo_idx 升序
- 不设置正负样本比例配额

#### Prompt 构造

- Stage 1：
  - 判断目标句中是否存在 Skill
  - 严格返回 {"entity_types":[]} 或 {"entity_types":["Skill"]}
- Stage 2：
  - 展示 16 条示例句及其伪 TRF
  - 结合 Stage 1 判断提取目标句的开放词表 TRF
  - 严格返回 {"trfs":[...]}
- Final Skill Prediction：
  - 只向最终预测器提供目标原句和推断出的 TRF
  - 不提供 Demonstration 标签
  - 要求用 <skill>...</skill> 标注原句中的精确跨度

### Output

```text
target/targets/records.jsonl
target/retrieval/records.jsonl
target/prompts/records.jsonl
target/parsed/records.jsonl
target/prediction/prompts.jsonl
```

其中：

- retrieval/records.jsonl 使用中立的 candidate-instances-v1 格式，可供实例判别器复用
- parsed/records.jsonl 使用中立的 feature-records-v1 格式，可作为其他模块的可选特征

---

## 3.3.3 code/trf/target/online.py

### Input

- Demonstration 和目标句
- Qwen embedding 配置：
  - 模型：qwen3.7-text-embedding
  - 维度：1024
  - Batch size：20
- 两轮 TRF Prompt
- Qwen Chat 配置：
  - 模型：qwen3.7-plus-2026-05-26
  - Temperature：0
  - Structured output：开启

### Process

- 优先读取当前运行已有的 embedding
- 如果提供兼容运行，则复用其 embedding
- 相同句子也可直接复用当前运行中的 Demonstration embedding
- 对缺失 embedding 分批调用 Qwen API
- 对每条目标执行：
  - Stage 1 实体类型判断
  - Stage 2 开放 TRF 提取
- 严格解析 JSON key 和字段类型
- TRF 可以不出现在 Main TRF 候选库，也不要求逐字出现在目标句中
- 验证失败时：
  - 生成空 TRF 上下文
  - 将记录标记为 needs_review
  - 继续执行最终 Skill 预测
- Provider、网络或运行时错误则作为阻断错误保留

### Output

```text
target/embeddings/demonstrations.jsonl
target/embeddings/targets.jsonl
target/raw/responses.jsonl
target/parsed/records.jsonl
```

---

## 3.3.4 最终 Skill 精确跨度预测

### Input

- 目标原句
- Stage 2 推断出的开放 TRF
- 严格 Skill 标注说明

### Process

- 要求模型返回：

```json
{
  "annotated_sentence": "包含可选 <skill>...</skill> 标签的完整原句"
}
```

- 移除标签后的文本必须与原句完全一致
- 如果首次输出非法，最多执行两次 repair
- 如果只有安全的 CP1252 标点差异，可映射回原句坐标
- 无法安全映射但标签结构有效时，保留 model-coordinate provisional extraction
- 按目标去重统计 TRF 抽取和跨度预测的验证失败
- 允许的验证失败数量为：

```text
floor(目标数量 × 3%)
```

- 网络错误、运行时错误和缺失响应不进入该容错额度，仍会阻止完整完成

### Output

```text
target/prediction/
├── prompts.jsonl
├── raw.jsonl
├── records.jsonl
├── failures.jsonl
└── results.jsonl
```

- records.jsonl：成功解析的精确跨度
- failures.jsonl：终态校验失败、模型输出和 repair 记录
- results.jsonl：每个目标一条结果信封，状态包括：
  - exact
  - recovered
  - provisional
  - validation_failed
  - runtime_failed
  - missing

---

## 3.3.5 code/trf/target/diagnostics.py

### Input

- 目标记录
- 检索结果
- 解析后的 TRF
- 最终 Skill 预测
- 失败记录和验证事件

### Process

- 汇总检索、TRF 抽取和 Skill 预测状态
- 计算 TRF 与参考伪 TRF 的 Jaccard 和 Recall 诊断值
- 统计 exact、recovered、provisional 和失败数量
- 生成固定数量的人工审核样本
- 不利用 Gold 标签计算最终任务准确率

### Output

```text
target/audit/
├── summary.json
├── manual_review.jsonl
└── validation_issues.jsonl
```

---

## 3.3.6 code/trf/target/common.py 与 validator.py

### Process

common.py 负责：

- 验证生成配置
- 管理运行目录和 manifest
- 锁定 Offline 来源文件
- 锁定目标输入和反馈文件
- 记录代码、依赖和输出哈希
- 检查 Resume 兼容性

validator.py 负责：

- 从源数据重新构造目标、检索、Prompt 和解析结果
- 检查全部记录身份和原句哈希
- 检查 embedding、模型版本和输出文件哈希
- 检查失败额度和最终运行状态
- 确认运行结果没有被事后修改

### Output

- Target 验证摘要
- 校验过程为只读，不改写运行产物

---

# 3.4 Full TRF（Offline 与 Target 联合运行）

## 3.4.1 code/trf/orchestration/runner.py

### Input

- 模块配置：config/trf.json
- 统一 run-id
- Demonstration manifest
- 独立目标文件或 leave-one-out 模式
- Offline 模型下载权限
- Target 网络调用权限
- Resume、Retry 和 embedding 复用参数

### Process

- 执行或读取已完成的 Offline 子运行
- 调用 Offline validator
- 根据 Offline 产物生成 target-config.generated.json
- 在生成配置中锁定：
  - Offline manifest
  - 正式语料
  - TRF 候选
  - 伪 TRF
  - Demonstration manifest 和 records
  - 每个文件的 SHA256
- 执行 Target 子运行
- 调用 Target validator
- 汇总 Offline 和 Target 的联合验证结果
- Offline 子运行失败后不可恢复，必须更换 run-id
- Target 子运行为 partial 时，可以使用 Resume

### Output

```text
output/trf/<run-id>/
├── manifest.json
├── target-config.generated.json
├── validation.json
├── offline/
└── target/
```

---

## 3.4.2 code/trf/orchestration/common.py

### Process

- 管理 Full 运行 manifest
- 记录以下阶段：
  - offline_trf
  - target_trf
  - joint_validation
- 锁定入口脚本、环境文件和 orchestration 实现哈希
- 检查 Resume 时的配置、输入和实现兼容性
- 汇总子运行状态、网络调用情况和全部输出哈希

### Output

```text
output/trf/<run-id>/manifest.json
```

---

## 3.4.3 code/trf/orchestration/validator.py

### Input

- Full manifest
- Offline manifest
- Target manifest
- Generated Target 配置
- 联合验证结果

### Process

- 分别调用 Offline 和 Target validator
- 检查父子 run-id 和运行状态
- 检查 Generated Target 配置是否仍与 Offline 产物一致
- 检查父级 manifest 中记录的全部文件哈希
- 确认 Offline 和 Target 均通过验证

### Output

```text
output/trf/<run-id>/validation.json
```

该文件包含 Offline 与 Target 的联合校验摘要。

---

# 3.5 TRF 模块的边界

当前 TRF 模块：

- 负责生成全局 TRF、示例伪 TRF、目标开放 TRF和 TRF 专家的 Skill 精确跨度预测
- 不调用实例判别器
- 不融合实例判别器或 Aggregator 的预测
- 不使用反馈文件更新 TRF
- 只将检索候选和 TRF 特征发布为中立接口，供第二层或其他模块消费
