# Project State

更新时间：2026-08-18（Asia/Shanghai）

## 门禁状态

| 门禁 | 状态 | 说明 |
| --- | --- | --- |
| 报名与审核 | complete | 用户已报名并审核通过 |
| XPU-OJ 账号 | complete | 用户已取得账号并登录 |
| 目标 OJ 入口 | verified | `https://xpuoj.com/contest/12/problem/1`，2026-08-18 已登录页面只读核对 |
| 目标算子 | verified | 页面标题确认 `Agent 推理算子库优化 - Fused MoE i8 tn` |
| 实时合同快照 | partial | CUDA Maca 签名、4 组 shape、容差和环境已记录；其他语言/配额待实际提交前复核 |
| 本地工作区 | complete | `E:\XH-202628\xh-202628-agent` 是唯一可信源码工作区 |
| 本地 Git 仓库 | connected | `main` 跟踪 `origin/main`；作者沿用远端提交 `Yuyang Dai <dyyshanghai@shu.edu.cn>` |
| 远端 SSH 只读探测 | complete | 已检查 `lynsdu2@10.0.33.75`，未写入 |
| 远端目录创建许可 | complete | 用户回复“1、允许”；许可和创建结果均已记录 |
| 远端执行目录 | complete | `/home/user/lynsdu2/xh-202628-agent` 于 2026-08-18T21:12:09+08:00 创建并核对 |
| 远端角色 | decided | 仅执行已提交 commit 的镜像，不作为 Git 工作树 |
| GPU 使用策略 | authorized | GPU 0-7；最多 4 个并行 run；不抢占已有计算进程；其余阈值见 JSON |
| GitHub 仓库 | connected | `git@github.com:RibbonFish-SHU/xh-202628.git`，现有 `main` 初始提交已 fetch |
| GitHub 认证 | complete | 本机 SSH 身份为 `RibbonFish-SHU`；GitHub 主机键按官方 Ed25519 指纹核验 |
| C500 本地算力 | unavailable | 当前未提供 |
| Agent OJ 提交次数 | 0 | 无历史提交 |
| 待向用户报告的提交 | none | 账本门禁清空 |

机器可读远端门禁见 `state/remote-execution.json`。

## 远端只读探测快照

- Home：`/home/user/lynsdu2`。
- OS：Ubuntu 20.04.6，kernel 5.15。
- CPU/RAM：2 x Xeon Silver 4216，64 threads，约 251 GiB RAM。
- GPU：8 x RTX A5000，compute capability 8.6，每卡约 24 GiB。
- Driver：535.261.03；CUDA driver capability 12.2。
- 编译器：`/usr/bin/nvcc` 是 CUDA 10.1；必须显式使用 `/usr/local/cuda/bin/nvcc`（CUDA 12.2）。
- 基础工具：Git 2.25.1、G++ 9.4、CMake 3.16、Make、Python 3.8。
- 缺少：pip/ensurepip、Conda、Micromamba、uv、Ninja、`rg`、`gh`。
- Docker daemon 对该用户不可用；远端 Codex 因 Node.js v10 不可用。
- `/home/user` 约 578 GiB 可用但总体已使用 91%，因此必须确定磁盘和保留策略。
- 远端无法直连 GitHub、PyPI、GitLink 或 XPU-OJ；新架构不依赖这些连接。

## 远端创建许可记录

- 状态：`CREATED`
- 用户许可原文：`1、允许`
- 批准记录时间：`2026-08-18T21:06:19+08:00`
- 批准绝对路径：`/home/user/lynsdu2/xh-202628-agent`
- 创建时间：`2026-08-18T21:12:09+08:00`
- 创建结果：base/incoming/locks/runs 均为 `lynsdu2:lynsdu2`、权限 `700`；marker 权限 `600`；初始子目录为空，总占用 20 KiB。

初始化已经完成，不得再次运行创建脚本。

## GitHub 配置

- Origin：`git@github.com:RibbonFish-SHU/xh-202628.git`。
- 默认分支：`main`；远端初始提交 `9ce0330e53ad2314f96afb690a9956eae80810d0`。
- 仓库级作者：`Yuyang Dai <dyyshanghai@shu.edu.cn>`，取自现有远端提交。
- 本机 SSH 已成功认证；服务器不保存 GitHub 凭据。
- 官方 `gh` 未安装，但当前同步只需 Git 2.54.0 和现有 SSH key。

## GPU 与磁盘策略

- 允许 GPU ID：0-7；最多 4 个并行 run，每张卡使用独立锁。
- 启动门禁：不得存在计算进程；利用率不超过 50%，显存占用不超过 2048 MiB。
- 单次 run 最长 21600 秒；远端目录上限 100 GiB。
- 用户要求以完成任务为主；仍不得抢占其他项目。
- 结果全部保留且不自动删除；达到空间门禁时停止并报告。

## 当前技术阶段与下一步

1. 提交并推送远端创建结果和 SSH 隔离修正。
2. 基于已冻结的实时 OJ 合同建立最小正确版本和 NVIDIA proxy baseline；提交前再次复核页面。
