# C500 Native Execution

本手册适用于已经初始化的 C500 执行镜像。它取代 NVIDIA proxy 作为新实验的默认性能测试链；旧 NVIDIA 脚本只用于读取历史结果，不得再启动新 run。

## 不变量

- 本地 Git commit 是唯一源码真源。远端不含 `.git`，不访问 GitHub，不手工编辑源码。
- 只有 Main Agent 能调用 C500。Producer 只交付 commit；不得访问 SSH alias、远端镜像或 C500 槽。
- `WorkflowCommit` 是当前 lane 的受信 `worktree_base_commit`。调用器中的 invoke/stage 必须与它同 blob；远端 runner、harness、汇总器全部从该 commit 归档，不从 candidate 分支取得。
- `refs/xh-202628/experiments/<id>` 与 `refs/xh-202628/baselines/<id>` 分别不可变地固定 `WorkflowCommit` 与 `BaselineCommit`；调用器同时核对两者，不能由调用参数自签名控制面或基线。
- 仓库只记录 SSH alias `xh-c500`；不记录登录地址、用户名、密码、私钥或控制 socket。
- 每次只运行一个 C500 任务。runner 在项目锁之外还要求 `mx-smi --show-process` 无设备进程，并核对 25% compute / 16000 MiB slice。
- 远端 run 永久保留，除非用户另行批准清理。不得覆盖、复用或删除既有 run。
- 本地结果必须标记为 `c500-local`。OJ slice 未确认前，本地绝对时间不能写成 OJ 时间或分数。

机器可读门禁位于 `state/c500-execution.json`。远端只允许触及 `/root/xh-202628-agent`。

## SSH 连接

入口网关当前只接受 password，不能完成端到端公钥认证。WSL 的 `xh-c500` alias 使用一条持久 OpenSSH control connection；密码不落盘。先检查：

```powershell
wsl ssh -o BatchMode=yes xh-c500 true
```

若该命令失败，说明控制连接在重启或断网后消失。此时执行一次交互式 `wsl ssh xh-c500 true`，由用户在终端输入密码；不得把密码写进脚本、环境变量、仓库或日志。连接恢复后再使用 BatchMode。

## 候选任务

Producer 从 `templates/remote-job.sh` 创建 `remote-jobs/<experiment-id>.sh`，两者 Git blob 必须完全相同。Main Agent 不执行 candidate 自定义控制脚本；调用器只接受上述受信 wrapper。Fused MoE driver 会：

1. 用同一 MACA/CUCC 参数编译 candidate 和 paired baseline。
2. 对 candidate 执行 correctness 与 regression。
3. 对两者各做一次完整预热。
4. 按 baseline A、candidate A、candidate B、baseline B 顺序运行基准。
5. 保留四份原始日志，并生成 `paired-benchmark.json`。

paired baseline 不是把工作树 checkout 到历史 commit，也不是复制 candidate 的 harness。调用器把 `WorkflowCommit` 归档为两棵受信树，将 `CandidateCommit` 的 submission source/模板 job 叠加到 candidate 树和 baseline 树，再用 `BaselineCommit` 的 submission source 覆盖 baseline 树。因此 runner、harness、编译脚本和 job 两臂完全相同，归因变量只有两份提交源码。

远端 stage 在解包前核对 workflow/candidate/baseline archive、stage 和两份源码 SHA-256，拒绝绝对路径、`..`、symlink、hardlink、device 与非预期 overlay 文件。staging 成功或失败均以原子 rename 固化 attempt；runner 状态也以临时文件原子替换。每个 run 只有一次持久 runner claim，重复启动不会改写已有状态。candidate 进程前后重新验证两份源码和共同 workflow 文件 manifest。

runner 用 `env -i` 和显式 allowlist 构造编译/执行环境，避免把 SSH agent、临时 token 或调用者环境变量传给 candidate；HOME/TMPDIR 也限制在 run 内。但当前入口以 root 运行且没有容器、mount namespace 或非特权用户，candidate host code 仍可访问项目目录外的 root 文件系统。哈希门禁能检测受监控树被修改，不能构成文件系统沙箱；只运行已审阅的提交源码，不得在文档中把这一边界描述成技术隔离。

## 启动

Producer 的 candidate、模板 job 和初版 handoff 已提交后，把 exact commit 交给 Main Agent。Main Agent 从干净、workflow-bearing 的控制工作树运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\invoke-c500-run.ps1 `
  -ExperimentId exp-YYYYMMDD-NNN `
  -EntryPoint remote-jobs/exp-YYYYMMDD-NNN.sh `
  -CandidateCommit <full-40-char-candidate> `
  -BaselineCommit <full-40-char-measured-baseline> `
  -WorkflowCommit <full-40-char-worktree-base> `
  -SubmissionSource operators/fused_moe_i8_tn/cuda_maca/submission.cu `
  -Attempt 1
```

`CandidateCommit` 必须是 `WorkflowCommit` 的后代，且 experiment job 的 blob 必须等于 `WorkflowCommit:templates/remote-job.sh`。两个 reservation ref 必须已由 `parallel_worktree.py create` 原子创建并与传入 workflow/baseline commit 完全一致。纯启动竞态使用同一 experiment/commit 的下一 `-Attempt`。源码变化必须使用新 commit 和新 experiment。不要用 SCP 手工传目录，也不要把未提交文件塞进归档。

若 stage 的 SSH 连接以 255 中断，但远端已经完整写成 `staged`，调用器会重新核对全部 metadata/hash、run/状态文件类型、持久 runner claim、项目锁、启动/结束标记和 runner 进程；全部通过后才在本次调用继续。若调用进程已经退出，使用同一组参数并增加：

```powershell
-ResumeStagedRun exp-YYYYMMDD-NNN-<12-char-commit>-aNN
```

`ResumeStagedRun` 只接受尚未启动的精确 `staged/0`，与 `RetrieveExistingRun` 互斥；出现 runner claim、锁、启动标记、运行进程或终态时都会失败关闭。不要通过新 attempt 覆盖状态不明的旧 run。

若远端已到终态但本地回收中断，使用同一参数并增加：

```powershell
-RetrieveExistingRun exp-YYYYMMDD-NNN-<12-char-commit>-aNN
```

恢复仍须提供完全相同的 candidate/baseline/workflow commit、entrypoint 和 source。调用器会重新生成三份 Git archive 和 source SHA-256，再与远端 terminal status 逐项比较；run ID 的 12 位 commit 不能由远端自报替代。

原始结果先回收到 primary 忽略目录中的唯一 `.partial-<guid>`。只有 `status.json` 的 state/exit code、全部 commit/hash、四 case `paired-benchmark.json` 和 `result-manifest.sha256` 验证通过后才原子改为 `artifacts/raw/c500-runs/<run-id>/`；网络中断留下的 partial 不阻塞下一次 recovery。正式目录已存在时会重新完整验证并幂等复用。runner SSH 若返回 255，而远端终态和回收 manifest 已全部验证，则以终态证据为准并保留 warning；其他退出码不一致会在正式 rename 前拒绝。

## 结果判定

- 任何 correctness/regression 失败都直接拒绝。
- `succeeded` 必须对应 exit code 0、完整 build/warmup/ABBA/mx-smi 证据和有效 manifest；`failed`、`preflight-failed`、`staging-failed`、`interrupted` 都是可诊断 terminal state，不得改写为成功。
- 每份 ABBA 日志必须恰好包含公开四 case，每 case 五个正有限 raw samples、可重算 median 与 `sampled_correctness=PASS`。
- slice 配额、设备进程或利用率门禁变化时，不得绕过 runner；先解释环境变化并重新建立 baseline。
- 检查 ABBA 两端 drift。接近噪声的收益必须用新的 attempt 重复；不能只挑最快一次。
- profiler 是按需诊断，不替代 paired wall-time。当前镜像提供 `mcProfiler`；先查看 `mcProfiler help` 和可用 metrics，再把大文件留在远端/ignored artifacts。当前未发现 `mcTracer`，不得假定它存在。
