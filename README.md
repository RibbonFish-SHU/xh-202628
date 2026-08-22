# XH-202628 Agent Workflow

这是赛题二的本地主控仓库。Main Agent 在 primary `main` 维护唯一正式 Git 历史；多个 Subagent 在隔离 linked worktree 中并行生产候选。需要 NVIDIA 算力时，只把各候选已提交 commit 的源码快照传到远端执行镜像，再取回测试结果。XPU-OJ 始终由 Main Agent 单通道提交。

## 当前状态

| 项目 | 状态 |
| --- | --- |
| Primary 工作树 | `E:\XH-202628\xh-202628-agent`，Main Agent 独占 `main` |
| 并行候选 | sibling linked worktree；每个 Subagent 独立 branch/worktree |
| 本地 Git | `main` 跟踪 `origin/main`；正式变更只由 Main Agent 提交和推送 |
| 远端只读探测 | 已完成 |
| 远端执行目录 | `/home/user/lynsdu2/xh-202628-agent` 已创建并核对 |
| GPU 使用策略 | GPU 0-7、最多 4 个并行空闲卡任务，已授权 |
| GitHub | `git@github.com:RibbonFish-SHU/xh-202628.git`，本机 SSH 已验证 |
| NVIDIA 执行链路 | 已验证：干净 commit 归档、远端 CUDA 12.2 编译执行、结果回收均成功 |
| XPU-OJ 历史 | 以 `state/submission-state.json` 的最新 tracked mirror 和中心控制器为准 |
| 提交控制 | 共享 SQLite controller；并行候选、单 active claim、逐次报告 |

## 架构

```text
primary xh-202628-agent/main（Main Agent）
  |-- shared candidate/controller DB <- isolated Subagent worktrees
  |-- reviewed commits -> GitHub
  |-- one claimed commit -> 已登录浏览器 -> XPU-OJ/C500
  |-- committed snapshots -> SSH/SCP -> NVIDIA 执行镜像（最多 4 run）
  `-- shared raw evidence <- primary artifacts/raw/remote-runs/
```

远端不保存 `.git`、GitHub 凭据或手工修改的源码。NVIDIA 数据只属于 `proxy/NVIDIA` 证据，不能代表 C500/MACA 性能。

## 新 Agent 启动顺序

1. 阅读 [`COMPETITION_CONTEXT.md`](COMPETITION_CONTEXT.md)。
2. 阅读 [`AGENTS.md`](AGENTS.md)。
3. 阅读 [`state/PROJECT_STATE.md`](state/PROJECT_STATE.md)。
4. 使用 [`skills/xpuoj-operator-optimizer/SKILL.md`](skills/xpuoj-operator-optimizer/SKILL.md)。
5. Main Agent 阅读 [`runbooks/parallel-orchestration.md`](runbooks/parallel-orchestration.md)，获取 controller 并创建隔离 lane。
6. 需要远端、GitHub 或 OJ 时，读取对应 runbook/reference 并检查门禁。

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

核心原则是：一条 lane 一个显式 baseline、一个假设、一个候选 commit 和一份可复现 handoff；候选并行生产，OJ 串行提交，每次终态先向用户报告再释放提交槽位。
