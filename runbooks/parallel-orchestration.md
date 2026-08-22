# Parallel Optimization Orchestration

本手册把算子优化拆成“并行生产候选、串行提交 OJ”。目标是让 Subagent、NVIDIA proxy 和代码审计持续并行，同时保证 XPU-OJ 永远只有一个提交者和一个在途提交。

## 不变量

1. Main Agent 是唯一协调者，独占主工作树的 `main`、GitHub push、正式状态文件、浏览器和 XPU-OJ。
2. 每个 Subagent 只在 Main Agent 创建的独立 branch/worktree 中工作，不能切换或修改 `main`。
3. Main Agent 为每条 lane 分配唯一 batch/experiment/candidate、当前 `worktree_base_commit`、显式 `performance_baseline_commit` 和一个可证伪假设。任何 Agent 都不能把两种 baseline 混为一谈。
4. Subagent 可以并行构建、测试并入队候选；只有 Main Agent 能把候选提升为 `ready`、claim OJ 槽位或改变提交记录。
5. OJ 同时最多一个 active claim。终态结果必须先登记、真实报告给用户，再释放槽位。
6. 等待 OJ 或一个慢 lane 时，其他 lane 继续生产候选。第一个达到晋级门槛的候选可立即进入串行提交，不等待整批结束。
7. NVIDIA 仅提供 `proxy/NVIDIA` 证据。远端最多 4 个并行 run，Main Agent 的 GPU 分配加 runner 的项目槽/GPU 锁是最终门禁；纯启动竞态使用下一 attempt，不烧掉 experiment 身份。

中心控制数据库位于 Git common directory 的 `.git/xh-202628/submission-control.sqlite3`。所有 linked worktree 共享它，但它不进入 Git。控制器拒绝自定义数据库或 linked-worktree mirror 路径；`state/submission-state.json` 始终指向 primary 工作树，只是 Main Agent 导出的受跟踪历史镜像。SQLite 是真源，镜像使用 digest 和 dirty 双门禁。

## 角色边界

| 能力 | Main Agent | Candidate Producer | Independent Auditor |
| --- | --- | --- | --- |
| 分配 batch / experiment / candidate | 唯一负责 | 禁止自行分配 | 禁止 |
| 创建 branch / worktree | 唯一负责 | 只使用已分配目录 | 只使用 detached audit worktree |
| 修改候选源码并 commit | 可做 | 只在自己的 branch | 禁止 |
| NVIDIA proxy run | 调度或执行 | 可在分配 lane 内执行 | 仅在任务明确要求时复核 |
| 写候选队列 | 可 | 仅 `candidate-enqueue` | 禁止 |
| 写实验账本 | 唯一串行导入 | 禁止 | 禁止 |
| 集成到 `main` / 更新正式状态 | 唯一负责 | 禁止 | 禁止 |
| GitHub push | 唯一负责 | 禁止 | 禁止 |
| controller/recovery token | 唯一持有 | 禁止索取、读取或转发 | 禁止索取、读取或转发 |
| 浏览器 / XPU-OJ | 唯一负责 | 禁止打开、提交或轮询 | 禁止打开、提交或轮询 |
| 向用户报告 OJ 结果 | 唯一负责 | 只向 Main Agent 交付技术结果 | 只向 Main Agent 交付审计发现 |

Producer 和 Auditor 都不得修改 `state/PROJECT_STATE.md`、`state/experiments.jsonl`、`state/submission-state.json` 或任何正式提交账本。Producer 的持久交付物是候选 commit 和 `handoffs/<experiment-id>.md`；Auditor 直接向 Main Agent 返回只读 findings，不修改 producer handoff。

## Main Agent 启动

必须从主工作树 `E:\XH-202628\xh-202628-agent` 的干净 `main` 启动：

```powershell
git status --short --branch
python skills/xpuoj-operator-optimizer/scripts/submission_controller.py init
python skills/xpuoj-operator-optimizer/scripts/submission_controller.py controller-status
python skills/xpuoj-operator-optimizer/scripts/submission_controller.py doctor
```

`init` 输出 recovery key 的 primary `.git` 路径。该文件和 controller token 一样只能由 Main Agent读取，不能复制到任务消息、日志、Git 或 Subagent 环境。

若没有 controller：

```powershell
$lease = python skills/xpuoj-operator-optimizer/scripts/submission_controller.py `
  controller-acquire --owner main-<session-id> | ConvertFrom-Json
$env:XPUOJ_CONTROLLER_TOKEN = $lease.controller_token
```

不要把 token 写入文件、Git、任务消息或日志。token/epoch 是同一操作系统用户下防止 Agent 误并发的协作式 fencing，不是安全隔离；能够读取该用户进程环境或 `.git` 的进程仍可能取得凭据。若已有 controller，先确认旧 Main Agent 已停止；不能仅因拿不到旧 token 就盲目接管。takeover 必须提供 `controller-status` 中刚核对的旧 owner/epoch，并通过环境变量提供 recovery token：

```powershell
$status = python skills/xpuoj-operator-optimizer/scripts/submission_controller.py `
  controller-status | ConvertFrom-Json
$env:XPUOJ_RECOVERY_TOKEN = (Get-Content -Raw .git/xh-202628/controller-recovery.key).Trim()
$lease = python skills/xpuoj-operator-optimizer/scripts/submission_controller.py `
  controller-takeover `
  --owner main-<new-session-id> `
  --reason "confirmed previous coordinator stopped" `
  --expected-owner $status.controller.owner `
  --expected-epoch $status.controller.epoch | ConvertFrom-Json
$env:XPUOJ_CONTROLLER_TOKEN = $lease.controller_token
Remove-Item Env:XPUOJ_RECOVERY_TOKEN
```

takeover 允许 primary mirror 因上一轮 finalize 而处于预期 dirty 状态，但不会授权后续命令绕过 digest/dirty 门禁，也不会删除 active claim。若 `doctor` 显示 drift，必须先检查 Git diff，再按 submission protocol 显式 reconcile/export。

## 规划一个并行批次

Main Agent 先建立机制矩阵。每条 lane 至少写明：

- 唯一 lane 名、experiment ID 和 candidate ID。
- 完整 40 位 `worktree_base_commit`：必须是创建时干净 primary `main` 的 HEAD，包含最新 workflow 工具。
- 完整 40 位 `performance_baseline_commit`：正式最佳或明确选择的已测源码基线；其 submission source blob 必须与 worktree base 中的 blob 完全相同。
- 一个目标 case、一个代码区域、一个机制 family 和一个可证伪假设。
- 预计改变的 bytes、barrier、CTA、LDS、thread、occupancy 或指令路径。
- 必做测试、proxy 证据边界、历史 denylist 和停止条件。

两条 lane 的 `target-case + code-region + mechanism-family` 不得有两个维度相同。同一家族连续两次目标无收益后关闭，除非获得新的硬件证据。不能把改名、无收益的指令挪动或已否决旋钮包装成新机制。

典型批次同时启动 2-3 条正交 lane，并保留 Main Agent 的协调容量。创建 worktree：

```powershell
python scripts/parallel_worktree.py create `
  --batch batch-20260822-01 `
  --lane g2s-pipeline `
  --experiment exp-20260822-026 `
  --worktree-base-commit <current-main-full-commit> `
  --performance-baseline-commit <measured-baseline-full-commit> `
  --source operators/fused_moe_i8_tn/cuda_maca/submission.cu
```

脚本只接受主工作树的干净 `main`，并创建：

```text
../xh-202628-agent-worktrees/<batch>/<lane>-<experiment>/
candidate/<batch>/<lane>-<experiment>
```

创建前脚本先验证 worktree base 等于当前 `main`、包含 controller/runbook/templates，且它的 source blob 与 performance baseline 完全一致；因此历史正式最佳 commit 只用于性能归因，不直接作为 worktree checkout。随后使用 `refs/xh-202628/experiments/<experiment>` 原子预留实验 ID并固定到 worktree base。并发分配同一 ID 时恰好一个成功；失败或已使用的 reservation 永不删除和复用。GPU/slot 的纯启动竞态使用同一 experiment/commit 的下一 `-Attempt`，不另烧实验 ID。

Main Agent 使用 `templates/subagent-task.md` 填充任务，并把精确 worktree、branch、base commit、GPU 分配和验收门槛交给一个 Subagent。不要让多个 Subagent 共享工作树，也不要让它们自行寻找“最新 baseline”。

## Subagent 交付

Subagent 在自己的 worktree 中完成一个候选：

1. 核对当前 `HEAD` 与分配的 `worktree_base_commit` 完全相同；不要 checkout 历史 performance baseline，它的精确 source 已由创建器验证并带入。
2. 只实现分配的单一假设；发现需要扩大机制或改变 baseline 时停止该 lane 并回报。
3. 创建唯一 `remote-jobs/<experiment-id>.sh`，完成可用的 build、correctness、benchmark 和 regression。
4. 需要 NVIDIA 时从自己的干净 worktree 调用 `scripts/invoke-remote-gpu-run.ps1`。只能使用 Main Agent 分配且实时空闲的 GPU；原始结果会集中写入 primary 的共享 ignored artifact root。
5. 记录 `handoffs/<experiment-id>.md`，格式见 `templates/subagent-handoff.md`。
6. 把源码、测试、remote job 和 handoff 一起 commit；保持工作树干净。
7. 将候选写入共享队列：

```powershell
python skills/xpuoj-operator-optimizer/scripts/submission_controller.py `
  candidate-enqueue `
  --candidate cand-exp-20260822-026-g2s `
  --producer <subagent-name> `
  --experiment exp-20260822-026 `
  --batch batch-20260822-01 `
  --lane g2s-pipeline `
  --worktree-base-commit <full-worktree-base-commit> `
  --performance-baseline-commit <full-performance-baseline-commit> `
  --commit <full-candidate-commit> `
  --source operators/fused_moe_i8_tn/cuda_maca/submission.cu `
  --language "CUDA Maca" `
  --hypothesis "<one falsifiable sentence>" `
  --evidence handoffs/exp-20260822-026.md
```

`candidate-enqueue` 会再次验证 reservation、batch/lane 对应的 branch tip、worktree-base ancestry、worktree/performance baseline 的起始 source 相等、精确 `remote-jobs/<experiment>.sh`、精确 `handoffs/<experiment>.md` 和候选源码 SHA-256。队列位于共享数据库，因此不会弄脏任何 worktree。Subagent 随后只向 Main Agent 发送候选 commit、结论和风险，不写 tracked 实验账本，也不做 OJ 操作。

## 独立 Auditor

涉及 barrier、异步 intrinsic、LDS 复用或线程映射时，Main Agent 为已提交 candidate 创建 detached audit worktree：

```powershell
python scripts/parallel_worktree.py audit-create `
  --batch batch-20260822-01 `
  --lane g2s-pipeline `
  --experiment exp-20260822-026 `
  --candidate-commit <full-candidate-commit> `
  --performance-baseline-commit <measured-baseline-full-commit> `
  --source operators/fused_moe_i8_tn/cuda_maca/submission.cu
```

脚本验证 audit commit 是已分配 candidate branch 的精确 tip，且派生自 reservation base。Main Agent 使用 `templates/auditor-task.md`；Auditor 不编辑 tracked 文件、不 commit、不 enqueue，也不接触 token、浏览器或 OJ，只返回有证据的 findings 和 `approve | reject | needs-fix`。

## Main Agent 晋级

候选到达后，Main Agent 不按完成顺序盲目提交，而按证据和预期收益排序：

1. 阅读 handoff 和 diff，确认 candidate 直接派生自分配 baseline，且只有一个归因变量。
2. Main Agent 根据 handoff 和共享 raw artifact 串行执行 `record_experiment.py`，避免多个 branch 同时追加 `state/experiments.jsonl`。
3. 检查历史 denylist、接口合同、地址覆盖、只读输入、同步、源码哈希和完整回归。
4. 涉及 barrier、异步 intrinsic、LDS 复用或线程映射的候选，必须由另一个 Agent 独立审计后才能晋级。
5. 要求结构性收益模型。一般应预计约 1% 或足以跨一个测试点分档；仅为获取必要目标反馈的诊断候选可以例外，但必须明确预算。
6. NVIDIA 未执行 MACA 分支时，只把结果用于来源与 fallback 回归，不能据此判断目标性能。

通过后，Main Agent 把候选 commit 集成到最新 `main`。若 cherry-pick 无冲突且提交后的 source blob 与候选完全相同，可继续使用原 candidate；若解决冲突改变了提交源码，必须分配新 experiment/candidate 并重新测试，不能伪装成原候选。

主工作树干净且候选 commit 已在 `main` 历史后执行：

```powershell
python skills/xpuoj-operator-optimizer/scripts/submission_controller.py `
  candidate-promote `
  --candidate cand-exp-20260822-026-g2s `
  --commit <full-integrated-main-commit>
```

控制器会再次核对 `main` 中的源码 blob。Main Agent 在 OJ 前 push 已验证的 `main`，但不能 push Subagent 临时分支或秘密。

## 串行 OJ 与持续生产

提交步骤严格使用 `skills/xpuoj-operator-optimizer/references/submission-protocol.md`。核心状态机是：

```text
queued -> ready -> claimed -> armed -> judging -> awaiting_report -> reported
```

`armed` 必须发生在最终 submit click 之前；`bind` 必须在点击后得到提交 ID 时立即执行。`awaiting_report` 在用户报告真正发出前一直占用全局槽位。

OJ 在 `judging` 时：

- Main Agent 继续收集和审阅其他 lane，不启动第二次 OJ 提交。
- Subagent 继续下一条已经分配的正交假设或完成当前交付。
- 候选可以保持 `queued`，但不要提前把多个会互相覆盖的 source blob 集成到 `main`。

若某个候选成为新最佳，Main Agent 更新正式 best。以旧 baseline 为基础且改动区域重叠的候选作废或重新分配；正交候选在重新核对后仍可排队。两个改动只有在分别取得目标收益后才能组合，组合必须使用新的 experiment/candidate。

若 OJ 候选没有成为正式最佳，在集成下一候选前必须完成显式恢复：

```powershell
git restore --source=<formal-best-commit> -- <submission-source>
git diff --exit-code <formal-best-commit> -- <submission-source>
git add -- <submission-source>
git commit -m "chore(operator): restore formal best after <submission-id>"
```

随后从 commit 重新读取 source 并核对它与正式 best 的源码 SHA-256 完全一致，再更新状态并 push。不能让 losing candidate 留在 `main` 后直接叠加下一候选；组合改动必须另建 experiment/candidate 并重新测试。

## 崩溃恢复

controller takeover 会 fence 旧 token，但不会删除 active claim。新 Main Agent 根据 `controller-status` 的 phase 恢复：

| Phase | 恢复动作 |
| --- | --- |
| `claimed` | 确认尚未打开提交动作后，可 `abandon` 并写原因 |
| `armed` | 必须检查 OJ 提交历史；找到记录就 `bind`，确认没有提交才可 `abandon --confirmed-no-submit`，并提供历史证据文件、核对时间和 claim source hash |
| `judging` | 按 submission ID 继续等待或抓取终态，然后 `finalize` |
| `awaiting_report` | 先向用户发送或保守地重复发送报告，再执行 `report` |

绝不能因超时自动清除 `armed` 或 `judging`。点击后尚未得到 ID 时，使用时间、题目、语言和源码哈希对账；在确认前保持全局阻塞，不能重提。

确认没有提交时，将历史页截图或文本保存到 `artifacts/raw/xpuoj/<reconciliation>/...`，再执行：

```powershell
python skills/xpuoj-operator-optimizer/scripts/submission_controller.py abandon `
  --claim <claim-id> `
  --reason "OJ history confirms no submit occurred" `
  --confirmed-no-submit `
  --oj-history-evidence artifacts/raw/xpuoj/<reconciliation>/history.txt `
  --checked-at <ISO-8601-with-timezone> `
  --expected-source-sha256 <armed-claim-source-sha256>
```

控制器会校验文件存在、核对时间不早于 `arm`，并把证据文件哈希、题目、语言和 claim source hash 写入审计事件。

会话正常结束且没有 active claim 或待报告结果时，可执行 `controller-release`。候选 worktree 和远端 run 不自动删除。Main Agent 只有在 candidate commit/branch 仍可定位、handoff 已提交、队列状态已终结、且 raw evidence 已确认位于 primary 共享 artifact root 后，才可退役本地 worktree；不得删除 reservation ref 或远端 run，其他清理仍需遵守现有保留策略。
