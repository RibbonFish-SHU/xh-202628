# XH-202628 Agent Workflow

这是赛题二的本地主控仓库。Main Agent 在 primary `main` 维护唯一正式 Git 历史；多个 Subagent 在隔离 linked worktree 中并行生产候选。Main Agent 用受信 workflow commit 固定 C500 runner/harness，只叠加已提交的 candidate/baseline submission source，完成目标编译、正确性、回归和同机配对基准。C500 和 XPU-OJ 都由 Main Agent 各自单通道操作。

## 当前状态

| 项目 | 状态 |
| --- | --- |
| Primary 工作树 | `E:\XH-202628\xh-202628-agent`，Main Agent 独占 `main` |
| 并行候选 | sibling linked worktree；每个 Subagent 独立 branch/worktree |
| 本地 Git | `main` 跟踪 `origin/main`；正式变更只由 Main Agent 提交和推送 |
| C500 执行目录 | `/root/xh-202628-agent` 已创建并核对 |
| C500 使用策略 | device 0、最多 1 个空闲设备任务；25% compute / 16000 MiB slice 门禁 |
| GitHub | `git@github.com:RibbonFish-SHU/xh-202628.git`，本机 SSH 已验证 |
| C500 执行链路 | 受信 workflow + 双 source overlay；原子 status/recovery、严格四 case ABBA 与结果 manifest |
| NVIDIA 执行链路 | 已停用；旧脚本和结果只作历史来源恢复 |
| XPU-OJ 历史 | 以 `state/submission-state.json` 的最新 tracked mirror 和中心控制器为准 |
| 提交控制 | 共享 SQLite controller；并行候选、单 active claim、延迟汇总报告 |

## 架构

```text
primary xh-202628-agent/main（Main Agent）
  |-- shared candidate/controller DB <- isolated Subagent worktrees
  |-- reviewed commits -> GitHub
  |-- one claimed commit -> 已登录浏览器 -> XPU-OJ/C500
  |-- workflow commit + candidate/baseline source -> WSL SSH/SCP -> C500（单 run）
  `-- shared raw evidence <- primary artifacts/raw/c500-runs/
```

远端不保存 `.git`、GitHub 凭据或手工修改的源码，也不信任 candidate 自带的 runner/harness。C500 数据标记为 `c500-local`；本地与 OJ 都是 C500/MACA，但 OJ slice 未确认与当前 25% slice 相同，因此本地 absolute timing 不等同于 OJ 分数。

## 新 Agent 启动顺序

1. 阅读 [`COMPETITION_CONTEXT.md`](COMPETITION_CONTEXT.md)。
2. 阅读 [`AGENTS.md`](AGENTS.md)。
3. 阅读 [`state/PROJECT_STATE.md`](state/PROJECT_STATE.md)。
4. 使用 [`skills/xpuoj-operator-optimizer/SKILL.md`](skills/xpuoj-operator-optimizer/SKILL.md)。
5. Main Agent 阅读 [`runbooks/parallel-orchestration.md`](runbooks/parallel-orchestration.md)，获取 controller 并创建隔离 lane。
6. 需要 C500、GitHub 或 OJ 时，读取对应 runbook/reference 并检查门禁。

## 目录

```text
xh-202628-agent/
  AGENTS.md                     强制操作规约
  COMPETITION_CONTEXT.md        长期比赛事实与解释
  state/                        当前状态与实验/提交账本
  runbooks/                     远端初始化与执行手册
  scripts/                      worktree、提交控制、本地传输和远端执行工具
  templates/                    Producer/Auditor 任务、候选交付、实验、远端任务和提交报告模板
  skills/xpuoj-operator-optimizer/
```

核心原则是：一条 lane 一个显式 baseline、一个假设、一个候选 commit 和一份可复现 handoff；候选并行生产，OJ 串行提交，终态落盘后立即释放槽位，用户主动打断时再汇总未报告结果。
