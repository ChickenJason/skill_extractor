# 新颖性与方法边界核查

## 快速检索结论

本轮只进行了面向方案定位的快速核查，不能把“未检索到”当作新颖性证明。

1. **无 BIO 的 span classification 已是成熟范式。** GNNer 明确将 NER 分为 sequence labeling 与
   枚举/分类所有 span 两类，并指出 unconstrained span classification 的重叠预测问题。
   - https://aclanthology.org/2022.acl-srw.9/
2. **Span prediction 作为基础系统和系统组合器已有系统研究。**
   - https://aclanthology.org/2021.acl-long.558/
3. **对 span 集合做全局最优选择也不是新概念。** Global Span Selection 使用全局分数和动态规划输出
   segmentation。
   - https://aclanthology.org/2022.umios-1.2/
4. **无序集合直接预测及匹配损失已有相关工作。** Set Generation Networks 强调输出集合而非人为序列，
   并使用 bipartite matching 处理排列不变性。
   - https://aclanthology.org/2021.emnlp-main.760/
5. **XMLC 的核心定义是从巨大标签全集选择相关子集；候选短名单/label filtering 也是标准可扩展策略。**
   - https://proceedings.mlr.press/v119/shen20f.html
   - https://proceedings.mlr.press/v54/niculescu-mizil17a.html
6. **多次采样后选择一致答案的 self-consistency 已有明确先例。**
   - https://arxiv.org/abs/2203.11171

## 不能宣称的新颖点

- “不用 BIO，改为 span classification”。
- “把输出看作无序多标签集合”。
- “枚举或筛选大量 span 标签”。
- “多次调用 LLM 后做多数投票”。
- “用全局优化去除 overlapping spans”。

## 可能形成贡献、但仍需完整文献核查的组合

**LLM 稀疏提案驱动的 instance-wise span XMLC + CMAS 式两阶段共识 + 风险受控自标注。**

具体组合是：

1. 不显式枚举 `O(m^2)` 标签，而由保持不变的 inline prompt 稀疏提出正 span 标签；
2. 第一阶段对重叠 span family 投票，判断某一区域是否存在稳定技能提及；
3. 第二阶段只允许得到完整 exact `[start,end)` 多数支持的标签进入正式标注；
4. 任何 majority-supported family 没有 exact winner 时进入 review，防止静默漏标；
5. 所有概率、冲突与样本集合差异保留为审计证据，而不是由 BIO token 拼接边界。

这个组合更适合称为项目方法假设，而不是当前已证明的新算法。若要用于论文贡献，需要进一步检索：

- LLM-based span aggregation / annotation aggregation；
- open-set extreme multi-label extraction；
- interval graph consensus；
- weak supervision for set-valued structured outputs；
- risk-coverage calibration for LLM pseudo-labeling。

## 本地证据对方法定位的支持

- 当前 BIO 方案存在 16 个 accepted candidate 缺少相应 exact 完整支持的问题。
- 朴素 exact vote 能消除这些候选，却会因为丢弃分散边界提案而将 3 个当前 review 句子错误升为
  accepted。
- 因此本项目真正需要验证的不是“span classification 是否优于 BIO”，而是：
  **两阶段 family/exact 共识能否同时阻止拼接假阳性和部分集合假接受。**

