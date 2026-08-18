# XH-202628 Agent Workflow

这是赛题二的本地主控工作区。后续工作 Agent 在这里维护唯一 Git 历史；需要 NVIDIA 算力时，只把已提交 commit 的源码快照传到远端执行镜像，再取回测试结果。

## 当前状态

| 项目 | 状态 |
| --- | --- |
| 本地工作区 | `E:\XH-202628\xh-202628-agent`，已创建 |
| 本地 Git | `main` 跟踪 `origin/main`；待提交本工作流 |
| 远端只读探测 | 已完成 |
| 远端执行目录 | `/home/user/lynsdu2/xh-202628-agent` 已获准，尚未创建 |
| GPU 使用策略 | GPU 0-7、最多 4 个并行空闲卡任务，已授权 |
| GitHub | `git@github.com:RibbonFish-SHU/xh-202628.git`，本机 SSH 已验证 |
| Agent 发起的 XPU-OJ 提交 | 0 次 |

## 架构

```text
本地 xh-202628-agent（唯一 Git 工作区）
  |-- GitHub CLI/MCP 或 git -> GitHub
  |-- 已登录浏览器 -> XPU-OJ/C500
  `-- git archive HEAD -> SSH/SCP -> NVIDIA 执行镜像
                                      `-> 原始结果 -> 本地 artifacts/raw/
```

远端不保存 `.git`、GitHub 凭据或手工修改的源码。NVIDIA 数据只属于 `proxy/NVIDIA` 证据，不能代表 C500/MACA 性能。

## 新 Agent 启动顺序

1. 阅读 [`COMPETITION_CONTEXT.md`](COMPETITION_CONTEXT.md)。
2. 阅读 [`AGENTS.md`](AGENTS.md)。
3. 阅读 [`state/PROJECT_STATE.md`](state/PROJECT_STATE.md)。
4. 使用 [`skills/xpuoj-operator-optimizer/SKILL.md`](skills/xpuoj-operator-optimizer/SKILL.md)。
5. 需要远端、GitHub 或 OJ 时，读取对应 runbook/reference 并检查门禁。

## 目录

```text
xh-202628-agent/
  AGENTS.md                     强制操作规约
  COMPETITION_CONTEXT.md        长期比赛事实与解释
  state/                        当前状态与实验/提交账本
  runbooks/                     远端初始化与执行手册
  scripts/                      本地传输和远端执行工具
  templates/                    实验、远端任务和提交报告模板
  skills/xpuoj-operator-optimizer/
```

核心原则是：一个假设、一个 commit、一个可复现测试记录；每次 OJ 评测结束后先向用户报告，再进行下一次提交。
