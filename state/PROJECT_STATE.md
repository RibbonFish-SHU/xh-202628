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
| NVIDIA 执行链路 | verified | `exp-20260818-001` 在空闲 GPU 1 成功完成归档、隔离 staging、CUDA 12.2 编译、内核运行和结果回收；`exp-20260818-000` 的预检拒绝也已留档 |
| Fused MoE NVIDIA 候选 | verified | 基线 `exp-20260818-002` / `ccabad8ab154`；packed INT8 dot 候选 `exp-20260818-003` / `3b7f02efb795` 在全部 4 个 proxy workload 提速且回归通过 |
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

1. 当前最小正确基线源码为 `ccabad8ab1541a2ec066795877f555fff5fe4e47`，实验为 `exp-20260818-002`。
2. 当前待目标验证候选为 `3b7f02efb7950fa5106ea7cd166ccc40e06a8bdd`，实验为 `exp-20260818-003`；相对基线的 4 个 proxy/NVIDIA speedup 为 1.1766x、1.2280x、1.2467x、1.2366x。
3. 首次 OJ 提交前复核实时题面、语言、签名、限制和提交配额，并确保提交源码精确来自 `3b7f02efb795`。目标首测需要验证 MXMACA 是否支持 `__dp4a` 以及 OJ 分配器能否暴露精确 allocation range。

## NVIDIA 执行链路验证

- `exp-20260818-000` / `6020d29b591c`：请求 GPU 0 时检测到已有负载，预检以 125 拒绝；没有启动 CUDA 任务，项目锁已释放。
- `exp-20260818-001` / `e59225b509a5`：物理 GPU 1（RTX A5000，compute 8.6）空闲；使用 CUDA 12.2 编译并运行最小内核，返回 `PASS` 和数值 42，runner 退出 0。
- `exp-20260818-002` / `ccabad8ab154`：物理 GPU 1 上建立 Fused MoE CUDA 基线。随机、tile 边界、INT8 极值和零值回归均以 `matched_ratio=1.0`、`max_abs=0` 通过；4 个公开 shape 的 allocation-range 推断全部通过；只读输入未被修改。四个 shape 的 proxy/NVIDIA median 分别为 45.584、341.346、21.856、173.180 ms（1 次预热、5 次记录）。
- `exp-20260818-003` / `3b7f02efb795`：只把逐字节标量 dot4 替换为 packed signed INT8 `__dp4a`。相同测试和方法下，四个 shape 的 proxy/NVIDIA median 分别为 38.742、277.980、17.531、140.046 ms；相对基线降低 15.01%-19.79% 延迟。全部正确性与回归继续以零差异通过。
- 上述原始结果均位于本地忽略目录 `artifacts/raw/remote-runs/`，远端唯一 run 目录继续保留。

`exp-20260818-002` 和 `exp-20260818-003` 的性能数据只属于 proxy/NVIDIA，不证明 CUDA Maca 可编译、C500 性能或 OJ 得分。CUDA 原始指针接口不携带 shape；当前源码仅使用官方 starter 的公开 allocation-size 推断，并移除了依赖生成数据值的 fallback。XPU-OJ 内部分配器能否暴露精确 allocation range、MXMACA 是否接受 `__dp4a`，仍是首次目标评测需要验证的风险。
