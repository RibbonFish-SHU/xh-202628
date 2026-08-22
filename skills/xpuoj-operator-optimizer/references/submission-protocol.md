# XPU-OJ Submission Protocol

每个最终 submit click 都是外部写操作。只有 primary `main` 工作树中的 Main Agent 可以执行本协议；Subagent 禁止访问浏览器或 OJ。

控制器：

```text
skills/xpuoj-operator-optimizer/scripts/submission_controller.py
```

它在所有 linked worktree 共享的 Git common directory 中维护 SQLite 状态，并强制使用 canonical `.git/xh-202628/submission-control.sqlite3` 与 primary `state/submission-state.json`；`--db`/`--mirror` 不能创建第二套控制面。旧 `submission_ledger.py record/report` 在中心数据库存在后会拒绝任何路径的写入，不能作为旁路。

## 1. Controller 与 Preflight

Main Agent 必须持有本会话的 controller token，并仅通过环境变量传给命令。token/epoch 只是在同一操作系统用户下避免多个协作 Agent 误操作的 fencing，不构成针对同用户进程的安全边界：

```powershell
$env:XPUOJ_CONTROLLER_TOKEN = <token-from-controller-acquire-or-takeover>
python skills/xpuoj-operator-optimizer/scripts/submission_controller.py check
python skills/xpuoj-operator-optimizer/scripts/submission_controller.py doctor
```

继续前确认：

1. 当前是 primary worktree 的干净 `main`，没有 active claim、mirror dirty 或 digest drift；延迟用户报告不是 blocker。
2. 候选状态为 `ready`，且其完整 integrated commit 已经测试、记录并位于 `main` 历史。
3. 实时比赛、题目 URL、可见标题、语言、配额/冷却与本地合同一致。
4. commit 中待提交 source 的 blob、长度和 SHA-256 与候选记录一致。
5. build、correctness、regression 和必要的独立同步审计通过；C500 未验证部分明确标注。

为本次命令生成一个稳定且唯一的 request ID。同一个 claim 命令若输出丢失，只能使用原 request ID 重试；新尝试使用新 ID：

```powershell
$claim = python skills/xpuoj-operator-optimizer/scripts/submission_controller.py `
  claim `
  --candidate cand-exp-20260822-026-g2s `
  --request-id req-exp-20260822-026-attempt01 | ConvertFrom-Json
```

保存输出中的 `claim_id`、commit、source 和 SHA-256。request ID 永久绑定该 candidate；活动 claim 的相同请求可幂等重试，完成后的 request ID 永远不能复用。claim 成功后全局 OJ 槽位已占用，但尚未表示发生了点击。

## 2. 持久化点击意图

填充页面之前使用用户已登录的浏览器会话。按 `$kimi-webbridge` 要求先做 health check；登录失效时只请用户手动登录，不读取、保存或传输密码、Cookie 或 session token。

在最终核对题目与语言后、点击之前，先持久化本次点击意图。`arm` 会再次确认 clean `main` 的 source 仍与 claim hash 完全一致：

```powershell
python skills/xpuoj-operator-optimizer/scripts/submission_controller.py `
  arm `
  --claim $claim.claim_id `
  --problem-url https://xpuoj.com/contest/12/problem/1 `
  --language "CUDA Maca"
```

然后：

1. 在专用 browser session/tab 打开精确题目，不接管或关闭用户无关页面。
2. 使用 snapshot 和语义元素引用，核对标题与语言。
3. 从 controller 返回的 commit 读取精确 source，不提交工作树中的未提交文本。
4. 填充后尽量核对长度和 SHA-256。
5. 只点击一次。不能因页面迟缓双击，也不能在查 OJ 历史前重试。

一旦提交列表出现 numeric submission ID，立即绑定：

```powershell
python skills/xpuoj-operator-optimizer/scripts/submission_controller.py `
  bind --claim $claim.claim_id --submission-id <numeric-id>
```

`armed` 和 `judging` 都会持续阻塞下一次 claim。

## 3. 结果捕获与 Finalize

等待 submission ID 到达终态。排队、编译中或 judging 不是终态，不能据此宣称成功或再次提交。捕获：

- 提交 ID、提交/终态时间、commit 和 experiment。
- status、样例/正确性、score、rank、runtime、memory 和各 case 结果。
- 上一正式最佳、分数与时间差异。
- 页面文本或截图证据。

原始页面证据放在唯一忽略路径 `artifacts/raw/xpuoj/<submission-id>/`，派生可提交摘要前检查个人信息。

无论 Accepted、Compilation Error、Wrong Answer、Runtime Error、Timeout、Canceled 或零分，都执行。状态必须来自控制器的显式终态集合；Waiting、Queued、Compiling、Judging、Running 或未知文本不能 finalize：

```powershell
python skills/xpuoj-operator-optimizer/scripts/submission_controller.py `
  finalize `
  --claim $claim.claim_id `
  --operator "Fused MoE i8 tn" `
  --status "<terminal-status>" `
  --score "<score-or-n/a>" `
  --previous-score "<previous-best-or-n/a>" `
  --rank "<rank-or-n/a>" `
  --url "<safe-result-url>" `
  --evidence "artifacts/raw/xpuoj/<submission-id>" `
  --notes "<short-technical-note>"
```

`finalize` 原子写入终态、把 candidate 标记为 `finalized`、删除 active claim，再导出 tracked mirror。若进程在数据库提交与镜像导出之间中断，SQLite 保留真相且 `mirror_dirty=1` 会阻止后续操作；使用相同 claim 和终态参数重试 `finalize`，或由 Main Agent 执行 `export-ledger` 修复。镜像一致后 OJ 槽位已经释放，`reported_to_user=false` 只表示等待汇总，不阻止下一次 claim。

## 4. 静默连续运行与延迟汇报

在用户人为打断之前，不主动发送逐次打榜报告，也不等待用户回复。每次 `finalize` 后仍须完成：

1. 根据结果更新正式 best 或恢复 formal-best source。
2. 更新 `state/PROJECT_STATE.md`、实验状态和 tracked mirror。
3. commit/push，保持 primary `main` 干净，然后继续下一候选。

用户主动打断、询问状态、任务结束或出现必须由用户处理的阻塞时，先读取全部延迟结果：

```powershell
python skills/xpuoj-operator-optimizer/scripts/submission_controller.py unreported-list
```

使用 `templates/submission-report.md` 发送一次汇总，每个提交至少包含：

- 提交 ID 与时间、完整 Git commit、语言和实现方案。
- 正确性/终态、总分、排名或各测试点结果。
- 相对上一版与正式最佳的变化。
- NVIDIA proxy 证据及其不能代表 C500 的边界。
- 失败原因、keep/reject/investigate 决策和下一假设。

只有汇总消息实际发送后，才逐条标记 reported：

```powershell
python skills/xpuoj-operator-optimizer/scripts/submission_controller.py `
  report `
  --submission-id <numeric-id> `
  --summary "<same-report-summary>" `
  --message-ref "user-update-<timestamp-or-stable-reference>"
```

`report` 只记录该结果已经向用户送达，不负责释放 OJ 槽位，也不影响正在进行的其他 claim。它会导出 mirror；汇总完成后把报告元数据 commit/push。

若提交没有成为正式最佳，下一候选集成前必须从明确的 formal-best commit 恢复 submission source，验证 exact source hash，commit/push 恢复。不得在 losing candidate 上叠加未经独立验证的组合。

## 5. Crash Reconciliation

新 Main Agent takeover 前先运行 `controller-status`，确认旧协调者已停止；takeover 必须提供 `.git/xh-202628/controller-recovery.key` 的环境变量、刚读取的 expected owner 和 expected epoch。它会 fence 旧 token，但不能清空未知状态：

| Phase | 必须动作 |
| --- | --- |
| `claimed` | 明确确认没有点击后，`abandon --claim ... --reason ...` |
| `armed` | 检查 OJ 历史；找到记录就 `bind`，确认不存在才执行有证据的 `abandon --confirmed-no-submit` |
| `judging` | 继续按已绑定 ID 等待/抓取终态，然后 `finalize` |
| legacy `awaiting_report` | 使用完全相同的终态参数重试 `finalize`，释放槽位并保留延迟报告 |

点击后崩溃且尚未 bind 时，以时间、题目、语言、源码长度/hash 对照 OJ 历史。不能确认时保持 `armed`，报告阻塞状态；绝不能假定未提交并重复点击。

确认没有提交后，把历史页文本或截图保存为 `artifacts/raw/xpuoj/` 下的文件，并同时提供核对时间和 `controller-status` 返回的 armed source hash：

```powershell
python skills/xpuoj-operator-optimizer/scripts/submission_controller.py abandon `
  --claim <claim-id> `
  --reason "OJ history confirms no submit occurred" `
  --confirmed-no-submit `
  --oj-history-evidence artifacts/raw/xpuoj/<reconciliation>/history.txt `
  --checked-at <ISO-8601-with-timezone> `
  --expected-source-sha256 <armed-claim-source-sha256>
```

控制器拒绝不存在或越界的证据文件、早于 `arm` 的核对时间和不匹配的 source hash，并把证据 SHA-256 写入审计事件。

claim、arm、bind、finalize 和 report 对相同参数可安全重试；任何冲突参数都会失败。`armed/judging` 不会因 controller takeover 或时间流逝自动解除。已 `finalized` 的 claim 即使 active row 已删除，也可用原参数幂等重试。

若 `doctor` 报告 tracked mirror digest drift，先检查 Git diff。只有确认 SQLite 应覆盖该文件时，才使用 `export-ledger --reconcile-drift --expected-current-sha256 <doctor-actual>`；摘要变化或缺少显式确认都会拒绝覆盖。
