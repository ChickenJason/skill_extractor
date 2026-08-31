# 运行日志

## 2026-09-01：TRF 输入/输出 ID 路径合同

| 项目 | 命令或检查 | 退出码 | 结果 |
|---|---|---:|---|
| Python 语法 | `python -m compileall -q code tests` | 0 | v2 路径绑定相关代码全部可编译 |
| TRF 定向回归 | `python -m unittest tests.test_trf_full_pipeline tests.test_trf_target_pipeline tests.test_trf_pipeline -v` | 0 | 24/24 通过 |
| PowerShell AST | 解析 `rerun-all-trf.ps1`、`run-trf-offline.ps1` | 0 | 两个入口均无语法错误 |
| Prompt 校验 | `GeneratePrompts.py --config config/prompt.json --validate-only` | 0 | 326/326 通过 |
| 完整离线回归 | `python -m unittest discover -s tests -v` | 0 | 62/62 通过 |

观察结果：`config/trf.json` 不再包含具体 `source.run_id` 或 `source.run_root`；调用者必须
提供 `SelfAnnotatorRunId`，系统据此绑定 `source.runs_root/<id>`。TRF `RunId` 只用于创建
`trf_full_runs/<RunId>`、`trf_runs/<RunId>-offline` 和
`trf_target_runs/<RunId>-target`。全部验证均使用临时测试目录，未调用网络。

## Self-Annotator → TRF 清洁重建迁移（2026-08-31，Asia/Shanghai）

工作目录：`E:\research\extractor\SkillSentence-Processor-0822`；解释器：项目
`.venv-trf` Python 3.10.14。

| 阶段 | 命令/操作 | 退出码 | 观察结果 |
|---|---|---:|---|
| 现状与依赖核查 | `rg`、`Get-Content` 检查配置、入口、验证器、测试和输出目录 | 0 | 定位旧 v0.5.1 来源、目标代码硬编码280和测试复制历史离线 run |
| 静态编译 | `.venv-trf\Scripts\python.exe -m compileall -q code tests` | 0 | 修改后的 Python 模块全部可编译 |
| TRF 定向回归 | `.venv-trf\Scripts\python.exe -m unittest tests.test_trf_pipeline tests.test_trf_target_pipeline tests.test_trf_full_pipeline -v` | 0 | 24项通过，测试来源全部位于系统临时目录 |
| Prompt 与全项目回归 | `GeneratePrompts.py --validate-only`；`python -m unittest discover -s tests -v` | 0 | 326条输入和60项测试全部通过 |
| 零 API 入口复验 | `run-reaggregate.ps1 ... -RunId span-xmlc-cleanup-validation-v1 -PythonExecutable .venv-trf` | 0 | 326句完成，不调用网络 |
| 实际离线 TRF | `run-trf-offline.ps1 -RunId trf-cleanup-validation-v1 -PythonExecutable .venv-trf` | 0 | 固定缓存 BERT 完成313条语料与伪 TRF |
| 离线只读验证 | `ValidateTRFRun.py --run-id trf-cleanup-validation-v1` | 0 | 225/88/13/0、313 formal、13 excluded、20/20候选全部通过 |
| PowerShell AST 首次组合检查 | 三个脚本 AST 检查，错误使用 `$file:` | 1 | PowerShell 在执行检查前报变量边界语法错误，无文件变化 |
| PowerShell AST 复验 | `Parser.ParseFile` 检查三个 PowerShell 入口 | 0 | `run-reaggregate`、`run-target-trf`、`rerun-all-trf` 全部通过 |
| 输出清理预检 | 锁定项目直属 `output`，统计文件并复核归档/模型 | 0 | 329个文件、126,760,787 bytes；归档 SHA256 与固定 BERT 均有效 |
| `Remove-Item` 清理尝试 | 动态目标和绝对字面目标各一次 | 创建进程前拒绝 | 安全层拦截，两次均无删除 |
| 可恢复清理 | 24个绝对一级产物目标逐项送入 Windows 回收站 | 0 | 历史与验收输出全部离开项目；五个输出根只保留 `.gitkeep` |
| 清空后全项目回归 | Prompt validate；`.venv-trf` 执行 `unittest discover -s tests -v` | 0 | 空 `output/` 状态下326条输入、60项测试和真实缓存 BERT smoke 全部通过 |
| 最终不变量 | 输出扫描、归档 SHA256、环境/模型缓存、活动旧合同扫描 | 0 | 仅5个 `.gitkeep`；归档哈希不变；Python/BERT存在；活动旧合同0命中 |

真实 Qwen 全量目标阶段未执行；当前进程没有可用凭据，本轮也未授权产生付费调用。

## Span XMLC v0.6.0（2026-08-31，Asia/Shanghai）

工作目录：`E:\research\extractor\SkillSentence-Processor-0822`。

| 阶段 | 命令/操作 | 退出码 | 观察结果 |
|---|---|---:|---|
| 修改前核心基线 | `python -m unittest tests.test_hard_span_matcher tests.test_span_xmlc_pipeline -v` | 0 | v0.5.1 核心26项通过 |
| 仓库与实验输入核查 | `rg`、`Get-Content` 检查配置、聚合器、测试、历史实验文件和归档输入 | 0 | 确认固定五采样、旧4票阈值、v0.5.1对照与归档源路径 |
| 修改后定向测试 | `python -m unittest tests.test_hard_span_matcher tests.test_span_xmlc_pipeline tests.test_qwen_client -v` | 0 | 新锚点覆盖、三票门控和配置合同36项通过 |
| 编译与全项目回归 | `python -m compileall -q code\self_consistent_annotation`；`.\scripts\run-tests.ps1` | 0 | Prompt校验326句；59项通过，固定BERT smoke因当前解释器条件跳过1项 |
| Python 3.10核心复验 | `.\.venv-trf\Scripts\python.exe -m unittest tests.test_hard_span_matcher tests.test_span_xmlc_pipeline tests.test_qwen_client -v` | 0 | 合同解释器 Python 3.10.14 下36项全部通过 |
| 正式零API重放 | `.\scripts\run-reaggregate.ps1 -SourceRunRoot 'archive\self-annotation-history\inline-v4-2-full-20260821-001' -RunId 'span-xmlc-base5-replay-majority3-anchor-v1'` | 0 | 326句完成，`network_called:false`、`source_unchanged:true` |
| 结果与迁移不变量 | Python只读核验 manifest、四类输出、锚点比较及v0.5.1逐句迁移 | 0 | 225/88/13/0；491 exact、6 hard；33句 unsolved→accepted |
| 最终来源与遗留扫描 | `Get-FileHash`、`rg`、行数核验 | 0 | 归档和v0.5.1 hash未变；TRF仍引用旧run；活动代码无旧复合相似度 |

v0.6.0 正式产物：

`output/runs/span-xmlc-base5-replay-majority3-anchor-v1/`

运行创建并完成于 `2026-08-31T14:58:05+00:00`。运行只读取归档五次响应，没有读取
Qwen 凭据、调用网络或覆盖任何历史 run。

---

环境日期：2026-08-22；时区：Asia/Shanghai。

| 阶段 | 命令 | 退出码 | 观察结果 |
|---|---|---:|---|
| 修改前基线 | `.\scripts\run-tests.ps1` | 0 | Prompt 校验 326 句；旧测试 37 项通过 |
| 编译检查 | `python -m compileall -q code/self_consistent_annotation` | 0 | 新模块可编译 |
| 修改后测试 | `.\scripts\run-tests.ps1` | 0 | Prompt 校验 326 句；新测试 29 项通过 |
| 可选 lint | `python -m ruff check code tests` | 1 | 当前环境未安装 Ruff；记录于 `FAILURES.md` |
| 正式重放 | `.\scripts\run-reaggregate.ps1 -SourceRunRoot 'archive/self-annotation-history/inline-v4-2-full-20260821-001' -RunId 'span-xmlc-base5-replay-v1'` | 0 | 326 句完成，未调用网络 |
| 输出核对 | 读取 derived manifest、parsed/aggregated/selected summaries 与 SHA256 | 0 | 1630 样本，326 decisions，275 formal annotations，源文件未变化 |
| 遗留扫描 | 扫描活动 code/config/scripts/docs 与数学分隔符 | 0 | 无活动 BIO/Adaptive 引用；无旧式方括号数学分隔符 |
| v0.5.1 编译与测试 | `python -m compileall -q code/self_consistent_annotation`；`.\scripts\run-tests.ps1` | 0 | Prompt 校验 326 句；30 项测试通过 |
| support-4 只读探针 | 使用当前 parsed 样本调用 `aggregate_records`，并与 v0.5.0 consensus 逐句比较 | 0 | 预期 192/88/46/0；5 句由 unsolved 转 accepted |
| support-4 正式重放 | `.\scripts\run-reaggregate.ps1 -SourceRunRoot 'archive/self-annotation-history/inline-v4-2-full-20260821-001' -RunId 'span-xmlc-base5-replay-support4-v1'` | 0 | 326 句完成，未调用网络 |
| support-4 输出核对 | 读取新 manifest、aggregated/selected summaries 和 SHA256 | 0 | 280 formal annotations；源文件未变化 |
| support-4 最终不变量 | 校验状态、接受来源、5 个状态迁移、24 个硬匹配 family、3 个并列 winner 与字符偏移 | 0 | 全部通过 |

## 重放产物

`output/runs/span-xmlc-base5-replay-v1/`

运行开始于派生 manifest 的 `2026-08-22T14:14:55+00:00`，完成于
`2026-08-22T14:14:56+00:00`。

v0.5.1 当前产物：

`output/runs/span-xmlc-base5-replay-support4-v1/`

运行开始于 `2026-08-22T14:42:18+00:00`，完成于 `2026-08-22T14:42:19+00:00`。

## 网络与外部状态

- `network_called:false`。
- 未设置或读取 Qwen API 凭据。
- 未写入 `archive/`。
- 未读取已有 adaptive 运行目录。

---

# TRF 运行日志（2026-08-28，Asia/Shanghai）

| 阶段 | 命令/操作 | 退出码 | 观察结果 |
|---|---|---:|---|
| 初始 TRF 单测 | `python -m unittest tests.test_trf_pipeline -v` | 0 | 10 通过，真实 BERT smoke 因 Python 3.12/缺权重跳过 |
| Conda 建环境 | `conda env create -f environment.yml` | 1（人工中止） | defaults repodata 长时间无响应；未创建新 Conda 环境 |
| 隔离环境 | `D:\Anaconda\envs\extractor\python.exe -m venv .venv-trf` + 固定 pip 安装 | 0 | Python 3.10.14；全部指定依赖精确匹配 |
| 正式 v1 | 带 `-AllowModelDownload` 运行 | 1 | torch distribution 合同误写为 2.5.1，被 `2.5.1+cpu` 门控阻断；未下载模型 |
| 正式 v2 | 带下载许可运行 | 1 | CDN 在已读 420,534,705 bytes 后 `IncompleteRead`，缺 15,244,452 bytes；未发布 |
| 下载续传 v3 | `HF_HUB_DOWNLOAD_TIMEOUT=600` 后带下载许可运行 | 0 | 固定 revision 下载并完成三阶段；用于发现双格式权重审计问题 |
| 最终代码全测 | compileall、Prompt validate、`unittest discover -s tests -v` | 0 | 326 条输入通过；41 项测试全部通过，真实 BERT smoke 通过 |
| 最终离线 v4 | 不带下载许可运行 `trf-offline-bert-base-cased-v4` | 0 | 24 秒完成，`network_called:false`，19 个产物写入 manifest |
| v4 只读验收 | `ValidateTRFRun.py --run-id trf-offline-bert-base-cased-v4` | 0 | 状态、计数、候选、伪标签、source/output/weight hash 全部通过 |
| 确定性 v5 | 不带下载许可运行并只读验收 | 0 | v4/v5 的 19 个产物 SHA256 全部一致，0 mismatch |

最终算法产物采用 `output/trf_runs/trf-offline-bert-base-cased-v4/`；v5 仅作为确定性见证。
v1、v2 是不可覆盖的失败审计，v3 是修正单一 safetensors 审计前的成功中间 run。

---

# 目标句 TRF 提取器运行日志（2026-08-28，Asia/Shanghai）

| 阶段 | 命令/操作 | 退出码 | 观察结果 |
|---|---|---:|---|
| 静态编译 | `.venv-trf\Scripts\python.exe -m compileall -q code\trf_target code\common\qwen_client.py` | 0 | 新模块与客户端扩展可编译 |
| 可选工具探测 | `.venv-trf\Scripts\python.exe -m pytest ...`、`-m ruff ...` | 1 | 该隔离环境未安装 pytest/ruff；未进入测试执行 |
| 目标模块测试 | `.venv-trf\Scripts\python.exe -m unittest tests.test_qwen_client tests.test_trf_target_pipeline -v` | 0 | 15 项通过，含 independent/LOO、partial/resume/retry/reuse 与验证器 |
| 全项目回归 | Prompt validate + `.venv-trf\Scripts\python.exe -m unittest discover -s tests -v` | 0 | 326 条输入通过；52 项测试全部通过；真实本地 BERT smoke 通过 |
| 入口与源终检 | PowerShell AST parse + 固定源 SHA256 复核 + 输出目录扫描 | 0 | 脚本语法通过；5 个源 hash 全匹配；目标输出目录只有 `.gitkeep` |

## 网络与外部状态

- 当前进程未设置 `DASHSCOPE_API_KEY` 或 `DASHSCOPE_BASE_URL`，没有执行真实 Qwen smoke。
- 所有目标提取器集成结果来自 deterministic mock provider，保存在测试临时目录并已清理。
- 没有创建正式 `output/trf_target_runs/<run-id>`；没有修改 TRF v4 或 self-annotation 源。

---

# 完整 TRF 重跑入口日志（2026-08-28，Asia/Shanghai）

| 阶段 | 命令/操作 | 退出码 | 观察结果 |
|---|---|---:|---|
| 入口静态检查 | `compileall code/trf_full code/trf_target` + PowerShell AST parse + 两个 CLI `--help` | 0 | Python 与 PowerShell 入口可解析，参数完整 |
| 最小联合测试 | `.venv-trf\Scripts\python.exe -m unittest tests.test_trf_full_pipeline tests.test_trf_target_pipeline -v` | 0 | 11 项通过；父 partial/resume/completed 和联合验证通过 |
| 最新全项目回归 | `.venv-trf\Scripts\python.exe -m unittest discover -s tests` | 0 | 54 项全部通过，包含真实 BERT smoke 与完整 runner 测试 |
| 最终安全审计 | 三个 PowerShell AST parse + 默认目标源 SHA256 + 输出目录/凭据检查 | 0 | 5 个源 hash 不变；full/target 目录仅 `.gitkeep`；未设置 Qwen 凭据 |

最小联合测试的离线 executor 复制已验证 v4 到临时目录，仅用于隔离测试父级编排；目标
Embedding/chat 使用 deterministic mock。临时目录退出后清理，没有发布正式 full run，
也没有调用 Qwen 网络。

---

# TRF Top-50 扩展运行日志（2026-08-28，Asia/Shanghai）

| 阶段 | 命令/操作 | 退出码 | 观察结果 |
|---|---|---:|---|
| 离线候选单测 | `.venv-trf\Scripts\python.exe -m unittest tests.test_trf_pipeline -v` | 0 | 11 项通过；两个候选库均为 50，原 Top-10 前缀不变 |
| 新离线 run | `run-trf-offline.ps1 -RunId trf-offline-bert-base-cased-top50-v1` | 0 | 27 秒完成；复用本地固定 BERT，未联网 |
| 新 run 只读验证 | `ValidateTRFRun.py --run-id trf-offline-bert-base-cased-top50-v1` | 0 | 192/88/46/0、50/50 候选、280 条伪标签及权重 hash 全部通过 |
| 目标/完整模块测试 | `python -m unittest tests.test_trf_target_pipeline tests.test_trf_full_pipeline -v` | 0 | 11 项通过；动态候选数与 Top-50 source contract 通过 |
| 首次最终回归 | compileall 后调用旧 `ValidatePrompt.py` | 2 | 文件不存在；测试未启动，未产生输出修改 |
| 最终全项目回归 | 当前 Prompt validate + `python -m unittest discover -s tests -v` | 0 | 326 条输入通过；55 项测试全部通过，含旧 Top-10 来源拒绝测试 |

本轮没有设置 `AllowNetwork`，未实例化 Qwen 客户端。既有 `trf-complete-v1` 保持历史
Top-10 在线结果；若要获得 Top-50 目标句 TRF，需使用新 full run-id 重新执行在线阶段。

---

# TRF Top-20 折中运行日志（2026-08-28，Asia/Shanghai）

工作目录：`E:\research\extractor\SkillSentence-Processor-0822`；解释器为项目
`.venv-trf` Python 3.10，固定 BERT 已缓存，所有命令均未允许网络。

| 阶段 | 命令/操作 | 退出码 | 观察结果 |
|---|---|---:|---|
| 离线单测 | `python -m unittest tests.test_trf_pipeline -v` | 0 | 11项通过；主/context均为20项 |
| Top-20离线 run | `run-trf-offline.ps1 -RunId trf-offline-bert-base-cased-top20-v1` | 0 | 约28秒完成，`network_called:false` |
| 只读验证 | `ValidateTRFRun.py --run-id trf-offline-bert-base-cased-top20-v1` | 0 | 192/88/46/0、20/20候选、280伪标签全部通过 |
| 目标/完整测试 | `python -m unittest tests.test_trf_target_pipeline tests.test_trf_full_pipeline -v` | 0 | 12项通过；Top-10/Top-50来源均被拒绝 |
| 全项目回归 | compileall、Prompt validate、`unittest discover -s tests -v` | 0 | 326条输入、55项测试全部通过 |

本轮只发布离线 Top-20 基线，没有创建新的完整父 run，没有发生 Qwen embedding/chat 调用。
