# Historical NVIDIA Execution

**已停用：不得启动新 NVIDIA run。** 本文件只用于恢复和解释已有历史结果；新实验读取 `c500-execution.md` 并使用 `scripts/invoke-c500-run.ps1`。`state/remote-execution.json` 的执行开关保持关闭。

## 前置条件

- `state/remote-execution.json` 中目录许可和执行开关均已启用。
- 用户已授权积极完成任务：GPU 0-7 可用，最多 4 个并行 run；选中卡启动前仍不得存在计算进程。
- 本地 Git 至少有一个 commit，工作树完全干净。
- 待运行入口位于 `remote-jobs/<experiment-id>.sh`，已经提交且能够从仓库根目录执行。
- 实验 ID 使用 `exp-YYYYMMDD-NNN`，同一 ID 不复用。

并行批次中，每个 Subagent 只从自己的干净 linked worktree 运行 Main Agent 分配的 experiment/GPU。Main Agent 负责避免 lane 间重复分配；远端 run-slot/GPU 锁负责处理启动竞态。所有 linked worktree 的原始结果统一回收到 primary 工作树的 `artifacts/raw/remote-runs/`，不会因候选 worktree 退役而丢失。

入口脚本应自行完成 build、correctness、benchmark 和 regression，失败时返回非零退出码。不要在入口脚本中写密码、token、固定 GPU ID、远端绝对 run 路径或清理命令。

可从 [`../templates/remote-job.sh`](../templates/remote-job.sh) 创建入口脚本，然后先提交：

```powershell
git status --short
git commit -m "operator(scope): exp-YYYYMMDD-NNN hypothesis"
```

## 运行

先只读查询 GPU 状态，不要固定假定 GPU 0 空闲：

```powershell
ssh -o ClearAllForwardings=yes lynsdu2@10.0.33.75 `
  "nvidia-smi --query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits"
```

从 `state/remote-execution.json` 的白名单内选择需要的卡。只有利用率和显存低于当前门禁、且该 GPU UUID 没有计算进程时才可选择；显示服务等图形进程不等同于计算进程。不要为了凑满 4 个并行任务而使用忙卡。状态查询与任务启动之间存在竞态，因此 runner 会在取得项目锁后再次检查；二次检查失败时保留该 attempt 的记录，重新核对后用同一 experiment/commit 的下一 attempt 重试。

然后从仓库根目录执行，例如（GPU ID 仅为示例，必须替换为刚核验的空闲卡）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\invoke-remote-gpu-run.ps1 `
  -ExperimentId exp-20260818-001 `
  -EntryPoint remote-jobs/exp-20260818-001.sh `
  -GpuIds 1 `
  -Attempt 1
```

脚本会：

1. 检查门禁、GPU 白名单、入口文件、commit 和干净工作树。
2. 用 `git archive HEAD` 生成临时 tar；不会包含 `.git`、未提交文件或仓库外文件。
3. 上传到 `incoming/`，并在唯一的 `runs/<experiment-id>-<short-commit>-aNN/` 解包。
4. 使用 `/usr/local/cuda`、指定 GPU 和动态空闲检查运行入口；利用率阈值 50%、显存阈值 2048 MiB、单次上限 21600 秒。
5. 保存环境、stdout、stderr、时间和退出状态。
6. 把 `results/` 回收到 primary 工作树的 `artifacts/raw/remote-runs/<run-id>/`。

所有项目 SSH/SCP 调用显式使用 `ClearAllForwardings=yes`，不会继承本机 SSH 配置中的端口转发。

入口失败时仍尝试回收结果，并以非零状态结束。run ID 或远端目录冲突时拒绝覆盖。若源码/假设不变且只因 GPU/slot 启动竞态失败，使用下一 `-Attempt`；若源码或假设变化，必须创建新实验 ID。任何情况都不清理旧目录。

## 结果处理

检查本地结果中的：

- `status.json`
- `environment.txt`
- `gpu-before.csv`
- `stdout.log` 与 `stderr.log`
- `started-at.txt`、`finished-at.txt` 和 `duration-ns.txt`

原始目录默认由 `.gitignore` 排除。使用实验账本记录 commit、命令、设备、关键数字、原始相对路径和结论；只提交审阅过的小型摘要或 CSV/JSON。

所有结果标为 `proxy/NVIDIA`。不得根据这些结果直接声称 C500/MACA 兼容或榜单提升。

## 保留策略

远端空间门禁为 100 GiB。run、源码归档和结果一律保留，达到上限时停止新任务并报告；不得在入口脚本或本地编排脚本中递归删除、覆盖同步或自动清理。
