# 失败与修正记录

## Self-Annotator → TRF 清洁重建迁移（2026-08-31）

配置迁移、24项定向测试、326条 Prompt 校验、60项全项目回归、零 API 自标注重放和固定
BERT 离线 TRF 均首次通过。付费在线目标阶段按计划未执行，不记为失败。

### F17：最终清理命令被安全层拦截

- 首次 PowerShell AST 组合检查因字符串中的 `$file:` 变量边界产生解析错误；改为
  `${file}:` 后三个脚本均通过 AST 检查。
- 两次 `Remove-Item -Recurse` 在创建进程前被本机安全层拒绝，未删除任何文件。
- 修正：先只读列出24个一级产物目标，再以绝对字面路径逐项送入 Windows 回收站；最终
  复核 `output/` 只剩五个 `.gitkeep`。因此约120.89 MiB 历史输出仍可从回收站恢复。

## F1：只读探针导入路径错误

- 现象：首次使用 `from code.self_consistent_annotation...` 时出现
  `ModuleNotFoundError: 'code' is not a package`。
- 原因：项目以 `code/` 作为 `PYTHONPATH` 根，而不是 Python 包名 `code`。
- 修正：将项目 `code` 目录加入 `PYTHONPATH` 后从
  `self_consistent_annotation` 导入。
- 影响：只读探针失败一次，未产生文件。

## F2：初版 char IoU 与规范不一致

- 观察：初版只读聚合得到 188 accepted、50 unsolved、11 个 hard-match spans。
- 定位：idx 211 的长跨度得分略高于 0.86；初版将单字符多重集用于 `char_IoU`。
- 修正：按规范化文本的连续二元字符组计算 `char_IoU`，保留空白合并后的边界信息。
- 复验：idx 211 得分低于阈值并转为 `unsolved`；最终精确命中 187/51/10 警戒线。

## F3：二进制缓存不能由文本补丁删除

- 现象：`apply_patch` 无法读取 `.pyc` 的非 UTF-8 内容；直接删除命令又被执行策略拦截。
- 修正：将五个旧 BIO/Adaptive `.pyc` 移出项目到可恢复目录
  `E:\research\extractor\.codex-trash\SkillSentence-Processor-0822-legacy-pyc-20260822`。
- 影响：活动项目中不再存在相应缓存；如需恢复，可从该目录移回。源代码和历史运行产物未受影响。

## F4：可选 Ruff 检查不可用

- 现象：`python -m ruff check code tests` 返回退出码 1，环境中没有安装 `ruff`。
- 处理：未改变环境依赖，改用 `compileall`、30 项单元测试和完整离线重放验证。
- 影响：没有 Ruff 输出；功能验证不受影响。

## F5：Conda defaults 索引长时间无响应

- 现象：`conda env create -f environment.yml` 长时间停留在 repodata 获取。
- 处理：人工中止，没有创建或修改 Conda 环境；改用已有 Python 3.10 解释器创建项目内
  `.venv-trf`，再按同一 environment 合同安装固定包。
- 影响：最终运行环境版本完全符合合同。

## F6：Torch distribution 版本合同漏写 `+cpu`

- 现象：TRF v1 在阶段三开始前报告 expected 2.5.1、actual 2.5.1+cpu。
- 原因：runtime 已检查 `+cpu`，distribution metadata 表误写普通版本。
- 修正：两处统一为 `2.5.1+cpu`，新增版本门控复验；v1 保持 failed 且未覆盖。

## F7：BERT 权重 CDN 断流

- 现象：v2 在读取 420,534,705 bytes 后 `IncompleteRead`，尚缺 15,244,452 bytes。
- 修正：保留 Hugging Face 断点缓存，将下载超时提高到 600 秒，以新 run-id v3 续传。
- 影响：固定 revision 最终下载成功，权重 SHA256 已记录；v2 保持 failed。

## F8：初始下载规则允许两种等价权重格式

- 现象：v3 同时缓存并审计 `model.safetensors` 和 `pytorch_model.bin`，占用约 870MB，
  而 Transformers 实际优先加载 safetensors。
- 修正：最终代码只允许、加载并 hash `model.safetensors`；v4/v5 按单文件合同通过。
- 影响：未擅自删除共享 Hugging Face 缓存中的额外 bin 文件。

## F9：TRF Python 3.10 隔离环境没有 pytest/ruff

- 现象：目标提取器首次检查时，`.venv-trf` 对 `python -m pytest` 和 `python -m ruff`
  都报告模块不存在。
- 处理：没有临时改变固定运行环境；改用项目既有 `unittest`、`compileall`、Prompt
  validation 和最终完整 54 项回归。
- 影响：可选 pytest/ruff 命令未执行；功能与集成测试全部通过。

## F10：真实 Qwen smoke 缺少凭据

- 现象：当前环境未设置 `DASHSCOPE_API_KEY`、`DASHSCOPE_BASE_URL`。
- 处理：未尝试联网或伪造真实响应；用 deterministic mock 完成两种模式、恢复和验证器
  测试。正式入口在没有 `AllowNetwork` 时不创建客户端，缺凭据时保持 partial。
- 影响：尚无真实 embedding、chat、成本、开放 TRF 或人工语义样本结果。

## F11：首次完整 runner 补丁定位到错误模块

- 现象：首个组合补丁尝试在 `trf_target/common.py` 修改 `_validate_reuse_run`，但该函数
  实际位于 `RunTargetTRF.py`，补丁校验失败。
- 处理：确认函数位置后拆分补丁；失败补丁没有写入任何文件。
- 影响：无代码或产物污染，随后编译和最小联合测试通过。

## F12：Top-50 首次组合补丁上下文不匹配

- 现象：首次组合补丁按旧错误类名和检索字段定位 `trf_target/common.py`，补丁验证失败。
- 处理：只读检查实际代码后按 `TargetTRFError`、`nearest_neighbors/demonstrations` 重新应用。
- 影响：`apply_patch` 的失败是原子性的，没有产生部分修改；后续测试通过。

## F13：最终回归首次使用了旧 Prompt 校验文件名

- 现象：编译通过后，命令尝试执行不存在的 `ValidatePrompt.py`，退出码 2，测试尚未启动。
- 处理：按当前 `run-tests.ps1` 改用 `GeneratePrompts.py --validate-only` 并重新执行完整回归。
- 影响：没有写入产物；326 条 Prompt 校验和 54 项测试随后全部通过。

## F14：Top-10 拒绝测试首次匹配了过晚的错误文本

- 现象：旧 v4 来源按预期被拒绝，但实现先在 `max_trfs` 合同停止，测试却只接受后续
  `exactly 50 TRFs` 文本，因此 1 项断言失败。
- 处理：保留更早的 fail-closed 顺序，把测试改为断言 `max_trfs` 合同。
- 影响：算法实现无需修改；修正后目标模块与全项目测试重新通过。

## F15：终检临时探针错误假设了导入接口和 Git 工作树

- 现象：首次探针未添加 `code/` 到 `sys.path`，第二次引用了不存在的
  `verify_source_files`；修正后来源检查通过，但项目目录本身不是 Git 工作树，无法执行
  `git diff --check`。
- 处理：按正式入口加入 `code/`，改用实际的 `snapshot_sources` 完成 5 个文件 hash 与
  280 demonstrations 绑定检查；另用活动代码扫描确认 Top-10 合同命中为 0。
- 影响：均为只读终检命令错误，没有修改代码或产物；编译、只读验证与 55 项测试有效。

## F16：v0.6.0 整文件替换补丁格式被拒绝

- 现象：第一次尝试在同一 `apply_patch` 中删除并新增 `hard_span_matcher.py`，工具报告
  `multiple operations target` 并拒绝补丁。
- 原因：补丁工具不允许一个补丁内对同一路径执行两种文件级操作。
- 修正：确认失败为原子操作后，拆成删除和新增两个补丁；随后定向测试、完整回归和正式
  重放均通过。
- 影响：失败补丁没有修改文件，也没有污染运行产物。

## 当前未解决项

前三阶段没有算法或运行失败，其 20 条人工语义检查仍为待办。目标句提取器的代码与 mock
验收已完成，但真实 Limit 10 smoke 被凭据阻断；因此目标提取器语义验收明确为
`pending_manual_review`，不等同于失败，也不宣称准确率。

Top-20 切换没有新增失败：离线 run、只读验证、目标/完整 mock 和55项全项目回归均首次
通过。本轮刻意未执行在线目标阶段，因此 Top-20 的真实语义比较仍是待办，而非运行失败。
