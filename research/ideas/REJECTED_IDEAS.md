# 暂不采用或延后的方案

## 直接采用朴素 Exact-Span 多数投票

保留为基线，不作为最终方案。它能消除 BIO 拼接，但本地重放会把 3 个当前 review 句子升级为
accepted，因为另一个被边界变体分散支持的技能没有任何 exact 标签达到 candidate 阈值，从而被静默丢弃。

## 全量枚举所有字符 span 后逐标签询问 LLM

拒绝。动态标签空间为 `O(m^2)`，显式展开会造成 prompt 和调用成本爆炸。当前 inline prompt 已经是更高效的
稀疏 proposal generator。

## Start/End 独立投票

拒绝作为正式决策。它仍允许把不同采样中的 start 和 end 重新组合，保留当前 Frankenstein span 的根因。
start/end 统计最多作为审计信息存在。

## 立即采用全局 Surface-Form 标签库

延后。项目当前目标仍是开放式原文技能 mention 抽取；全局短语库会造成 OOV 漏召回、同义词碎片化和
表面词标签爆炸。可在高质量 mention 自标注稳定后，作为独立标准化层研究。

## 立即采用 Canonical Skill Taxonomy XMLC

延后。当前没有冻结 taxonomy、label descriptions、grounding gold 或 taxonomy 外标签政策，而且原提示词不输出
canonical label。现在引入会把“自标注器重构”和“技能规范化”两个研究问题混在一起。

## 用全局优化器自动解决所有重叠

拒绝作为 MVP 正式门。优化器必然返回一个集合，容易掩盖真实不确定性。主方案优先 review；全局 set packing
只做消融或候选建议。

## 直接采用 Dawid–Skene/EM

延后。同一模型同一 prompt 的重复采样不是相互独立的标注器，当前样本规模也不足以可靠估计复杂误差模型。
先建立人工 gold 和硬阈值基线，再判断概率聚合是否有价值。

## 第一版就加入 Helpfulness demonstrations

延后。CMAS 的 helpfulness 分支主要把评分留在会话上下文，没有程序化过滤低分示例。当前 prompt 冻结，
第一版应先隔离“聚合范式变化”的效果。以后若引入 demonstrations，必须做 hard gate，并单独消融。

