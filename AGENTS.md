# Agent Operating Contract

本文件适用于整个 `xh-202628-agent` 仓库，是后续工作 Agent 的强制规约。

## 1. 启动门禁

每个新会话开始时，按顺序完成：

1. 完整阅读 `COMPETITION_CONTEXT.md`、本文件和 `state/PROJECT_STATE.md`。
2. 只读取与当前任务相关的 Skill、reference 和 runbook。
3. 检查 `git status`、最近提交、实验记录和 `state/submission-state.json`。
4. 明确自己是 Main Agent、Candidate Producer 还是 Independent Auditor；Producer/Auditor 必须核对任务信封中的 worktree、candidate commit、`worktree_base_commit` 和 `performance_baseline_commit`。
5. 说明当前算子、接口契约版本、显式基线 commit、可用设备及未解除门禁。

不得仅凭历史对话摘要继续优化。事实变化时，先更新长期上下文和状态，再编码。

## 2. 可信仓库与隔离工作树

- 本地 `E:\XH-202628\xh-202628-agent` 是唯一 primary 工作树；它的 `main` 和 Git 历史是正式源码源头。
- 并行候选只能位于 Main Agent 通过 `scripts/parallel_worktree.py` 创建的 sibling linked worktree。每个 Subagent 独占一个 branch/worktree。
- GitHub 的认证、pull/push 只由 Main Agent 在 primary 工作树完成。
- NVIDIA 服务器只是执行镜像：接收某个已提交 commit 的归档，运行测试，再返回结果。
- 远端镜像不包含 `.git`，不直接访问 GitHub，也不得成为手工改源码的第二工作树。
- XPU-OJ 操作使用本地控制端中用户已登录的浏览器会话。

本地与远端如有差异，一律以本地已提交 commit 为准。不得从远端反向覆盖本地源码；只能回收测试结果。

### 2.1 并行所有权

- Main Agent 独占 `main`、实验 ID 分配、baseline 选择、候选集成、正式状态、controller token、GitHub、浏览器、XPU-OJ 和用户提交报告。
- Candidate Producer 只能实现被分配的一个可证伪假设，提交自己的 candidate branch，写 `handoffs/<experiment-id>.md`，并执行唯一允许的共享写操作 `submission_controller.py candidate-enqueue`。
- Independent Auditor 只在 Main Agent 创建的 detached audit worktree 中复核精确 candidate commit；禁止编辑、commit、入队或修改 producer handoff。
- Producer/Auditor 禁止修改 `state/PROJECT_STATE.md`、`state/experiments.jsonl`、`state/submission-state.json`，禁止切换 `main`、push GitHub、访问浏览器/OJ、执行 controller 特权命令或索取 controller/recovery token。
- 每条 lane 必须有唯一 batch/experiment/candidate、当前含完整工具的 `worktree_base_commit`、显式已测的 `performance_baseline_commit`、机制 family 和验收门槛。创建器必须确认两种 baseline 的 submission source blob 相同；历史性能 commit 不能直接作为缺少新工具的 worktree。
- 详细流程、角色模板和崩溃恢复见 `runbooks/parallel-orchestration.md`。

## 3. 权限与远端门禁

当前允许在本地仓库内创建和改进工作流、源码、测试、实验记录并管理 Git。用户也要求在条件满足后由工作 Agent 提交 XPU-OJ，并在每次评测结束后报告。

当前授权状态：

- **远端执行镜像已创建。** `/home/user/lynsdu2/xh-202628-agent` 已核对；不得再次初始化、覆盖或扩展到其他路径。
- **GPU 策略已授权。** 可用 GPU 0-7，最多 4 个并行 run；任何选中卡存在计算进程时必须拒绝启动。
- GitHub `origin` 已配置为 `git@github.com:RibbonFish-SHU/xh-202628.git`，使用本机已验证的 SSH 身份。
- 未提供 C500 本地算力；NVIDIA 结果不得声称为 C500 结果。

机器门禁位于 `state/remote-execution.json`。许可、创建时间、精确路径和 GPU 策略已经记录。不得为绕过脚本而擅自修改门禁。

远端只允许触及 `/home/user/lynsdu2/xh-202628-agent`。不得递归读取、索引、修改、移动、清理或借用服务器上的其他项目、环境和凭据。

## 4. 远端执行协议

每次 NVIDIA 测试严格遵守：

1. 在本地完成代码和一个位于 `remote-jobs/` 的可执行入口脚本。
2. 本地测试能够执行的部分，并记录一个明确、可证伪的实验假设。
3. 提交全部待测源码；要求工作树完全干净。
4. 使用 `scripts/invoke-remote-gpu-run.ps1`。该脚本只通过 `git archive HEAD` 传输已提交文件。
5. 远端在唯一的 `runs/<experiment-id>-<short-commit>-aNN/` 中解包并运行；不手工编辑源码，保留原始 `source.tar` 作为来源证据。纯 GPU/slot 启动竞态使用下一 attempt，不复用或另烧 experiment ID。
6. 回收 `results/` 到本地忽略目录 `artifacts/raw/remote-runs/`，核对退出码和原始日志。
7. 用实验账本记录 commit、设备、命令、结果路径和结论；只把经过检查的小型证据提交 Git。

禁止传输未提交文件、`.git`、秘密、构建树和无关文件。远端 run 默认永久保留，未取得保留/清理策略前不得自动删除或覆盖。

## 5. 优化纪律

1. 从实时 OJ 题面保存题目、语言、签名、shape、dtype、容差、限制和提交配额。
2. 建立独立参考实现、随机/边界正确性测试和最小可运行 baseline。
3. 记录环境、精确命令、原始输出、源代码 commit 和基线结果。
4. 每轮只验证一个性能假设，只做一组可归因修改。
5. 每轮执行 `build -> test -> benchmark -> regression`；失败实验也记录原因。
6. 提交 OJ 前确认源码 commit 等于已测试 commit。
7. 在 C500 可用前，所有 NVIDIA 性能判断只写成 `proxy/NVIDIA` 结论。

严禁修改官方测试标准、硬编码测试点、跳过计算或利用评测系统缺陷。

## 6. Git 纪律

- 默认分支为 `main`，除非目标 GitHub 仓库已有不同约定。
- 一个实验 commit 对应一个清晰假设；消息包含算子和实验 ID。
- 先 commit，再远端测试；如果测试导致后续修改，必须新建 commit 和实验 ID。
- 提交 OJ 的代码必须来自一个已记录且已测试的 commit。
- Main Agent 只能把已审阅 candidate 集成到 `main`；Producer/Auditor 不得 rebase、合并或清理其他 lane。
- Producer 只提交 handoff 和 candidate；只有 Main Agent 串行追加 `state/experiments.jsonl`，避免并行 branch 冲突。
- 不强推，不重写已经用于测试、OJ 或报告的历史。
- 不提交密码、token、Cookie、SSH 私钥、个人信息、构建产物和大型 profiler 文件。
- 不覆盖用户或其他 Agent 的未提交变更；对外 push 前检查状态和秘密。

建议 commit 格式：

```text
operator(scope): exp-YYYYMMDD-NNN concise hypothesis
```

## 7. XPU-OJ 提交门禁

XPU-OJ 采用“并行候选、单通道提交”。共享控制数据库位于 Git common directory 的 `.git/xh-202628/submission-control.sqlite3`，不进入 Git且不允许替换路径；primary `state/submission-state.json` 是由 Main Agent 导出的正式历史镜像。SQLite 是真源，dirty/digest 任一异常都会阻止下一次操作。

每次提交只由 Main Agent执行：

1. 在干净 primary `main` 获取 controller token，运行 `submission_controller.py check`，并把一个已集成候选提升为 `ready`。
2. 使用稳定 request ID 执行 `claim`，核对实时题目、语言、提交文件、commit、源码 hash 与全部正确性/回归证据。
3. 在最终 submit click 前执行 `arm`；通过用户已登录的本地浏览器只点击一次，得到 numeric submission ID 后立即 `bind`。
4. 等待评测终态，保存时间、状态、分数、排名/测试点和页面证据，再执行 `finalize`。
5. 立即向用户发送一次独立可见的报告；报告发出后才执行 `report` 释放全局槽位。
6. 更新正式状态和 tracked mirror，commit/push 后再次 `check`，之后才能 claim 下一份候选。

若结果未成为正式最佳，Main Agent 必须从明确的 formal-best commit 恢复 submission source，核对 exact source hash，提交并 push 恢复后才能集成下一候选。禁止在 losing candidate 上隐式叠加新改动。

旧 `submission_ledger.py record/report` 在中心数据库存在后禁止使用。编译错误、Wrong Answer、运行错误、超时、取消和零分都算一次提交，必须登记并报告。`armed` 或 `judging` 不能超时自动清除；崩溃后必须核对 OJ 历史。若当前回合无法在两次提交间向用户报告，则该回合最多提交一次。

OJ 评测等待期间，其他 Subagent 可以继续生产、测试和入队候选，但任何 Agent 都不能启动第二个 OJ 提交。

## 8. 证据与状态

每个实验至少记录实验 ID、时间、算子、设备、基线/候选 commit、假设、精确命令、正确性结果、原始性能数据、失败原因和保留/回滚决定。

只有 Main Agent 在以下事件后更新 `state/PROJECT_STATE.md` 和相关机器状态：用户批准远端目录或 GPU 策略、远端目录创建、GitHub 配置、正式最佳变化、每次 OJ 提交及报告、取得 C500 算力、正式规则变化。Subagent 把候选状态写入 handoff 和 candidate queue，不写正式状态。

会话结束前，确保一个没有对话上下文的新 Agent 仅靠仓库即可安全接手。
