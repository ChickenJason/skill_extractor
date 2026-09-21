# 协作反思调度

本包在已经 completed 且通过官方 validator 的 Concurrent Scheduling 运行之上执行一次双向交叉反思。它不会重跑 TRF、候选检索或示例判别器，也不会修改这些模块的代码和产物。

两个 reflector 从同一份冻结 R0 读取数据并由独立子进程同时运行：

- TRF reflector 使用完整 16 候选及其原判断，对基线 TRF 执行 `keep/drop/add`；
- Exemplar reflector 只使用基线 TRF，重新判断全部 16 候选；程序再调用原有硬门控生成修订选择；
- 两侧都不读取对方的 R1 输出，因此这是单轮、无顺序偏置的调度；
- 纯代码 interaction 阶段整理冲突、未复核新增 TRF 和 review 状态，不调用第三个模型；
- 最终只发布聚合器上下文，不生成技能跨度预测。

新增 TRF 还受 `config/trf_reflection_semantics.json` 的严格语义合同约束。新增项必须声明
`domain_feature` 或 `skill_type_cue`；品质/特征、描述性修饰语、任务/对象以及技能跨度或
其改写都不是 TRF。人工标注的反例会直接导致解析失败并进入受约束 repair，而不会被程序
静默删除；尚未有人工标签的新词保留为 `added_trf_semantics_unverified` 供审阅。
若基线原 TRF 命中人工标注的排除样例，反思输出也必须选择 `drop`。

## 运行

先完成不联网的基线验证、最小快照冻结和 prompt 准备：

```powershell
.\scripts\second_layer\run_reflection.ps1 `
  -RunId skill-reflection-limit3-v1 `
  -BaseConcurrentRunId skill-concurrent-v1 `
  -Limit 3 `
  -PrepareOnly `
  -PythonExecutable .\.venv-trf\Scripts\python.exe
```

恢复并同时启动两个在线 reflector：

```powershell
.\scripts\second_layer\run_reflection.ps1 `
  -RunId skill-reflection-limit3-v1 `
  -BaseConcurrentRunId skill-concurrent-v1 `
  -Limit 3 `
  -AllowNetwork `
  -Resume `
  -PythonExecutable .\.venv-trf\Scripts\python.exe
```

某条结构化输出会在当前请求内最多 repair 两次。若三次仍失败，运行保持 partial；修复外部原因后使用 `-Resume -RetryFailed` 只重试最新失败记录。全量在线运行不传 `-Limit`，并且必须显式增加 `-ConfirmFullRun`。

验证 completed 运行：

```powershell
.\.venv-trf\Scripts\python.exe `
  .\code\second_layer\collaborative_reflection\validator.py `
  --config .\config\second_layer_reflection.json `
  --run-id skill-reflection-v1
```

## 合同与目录

`source/` 仅冻结 base context、16 候选和原 judgments，并生成 `collaborative-reflection-input-v1`。`trf-reflection/` 与 `exemplar-reflection/` 分别保存 prompt、append-only raw/repair 链和严格解析结果。只有两侧 completed 且各自 validator 通过时，才原子发布：

- `interaction/records.jsonl`：确定性的 `aligned/mixed/conflicted` 整理；
- `context/records.jsonl`：`collaborative-reflection-context-v1`，同时保留 R0 与 R1；
- `context/summary.json`：变化、review、调用、token 和耗时统计；
- `audit/manual_review.jsonl`：所有非 aligned 记录；
- `manifest.json`：base/子 manifest、三个冻结文件、配置、实现及父级产物 SHA256。

旧的 `second-layer-parallel-v1` 是 Concurrent Scheduling 的兼容 pipeline 名。新入口 `.\scripts\second_layer\run_concurrent.ps1` 完整透传原 `.\scripts\second_layer\run.ps1`；已经完成的 concurrent manifest 不迁移、不回写。
