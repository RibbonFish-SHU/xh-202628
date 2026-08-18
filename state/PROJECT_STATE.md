# Project State

更新时间：2026-08-19（Asia/Shanghai）

## 门禁状态

| 门禁 | 状态 | 说明 |
| --- | --- | --- |
| 报名与审核 | complete | 用户已报名并审核通过 |
| XPU-OJ 账号 | complete | 用户已取得账号并登录 |
| 目标 OJ 入口 | verified | `https://xpuoj.com/contest/12/problem/1`，2026-08-19 已登录页面复核 |
| 目标算子 | verified | 页面标题确认 `Agent 推理算子库优化 - Fused MoE i8 tn` |
| 实时合同快照 | partial | CUDA Maca 与 Triton 签名、4 组 shape、容差和环境已复核；TileLang/配额待实际使用前复核 |
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
| Fused MoE 正确基线 | target-verified | `exp-20260818-004` / `b6e4272e9d8f` 已经 OJ `#116973` 验证 MXMACA 编译和 4/4 正确性；C500 得分仅 11，不能作为性能候选 |
| Fused MoE MMA 基线 | target-verified | `exp-20260819-007` / `83fd3d26e95f` 已由 OJ `#117056` 验证 MXMACA 编译和 4/4 正确性；恢复官方 `grid_x=1` 调度后得分 81.75、排名 23，仅测试点 1 增加 1 分 |
| Fused MoE wide-MMA 候选 | target-regressed | `exp-20260819-008` / `d24ad933e174` 经 OJ `#117079` 验证 4/4 正确但仅 79.25 分；prefill gate-up 从 73 降至 63，已由 `c2d5bcd` 恢复 81.75 分源码 |
| Fused MoE A-first inference 候选 | proxy-verified | `exp-20260819-009` / `7d4eef642c15` 保留目标验证的 128x128 MMA 核心，只把 shape inference 热路径从两次 allocation-range 查询减为一次；完整代理门禁通过，目标计时收益待 OJ 验证 |
| Fused MoE exact-row A-load 候选 | proxy-verified | `exp-20260819-010` / `8c519e6c1bb5` 在 128x128 流水核心中移除合同保证永远为真的 A-load 行谓词；完整代理门禁与逐线程地址范围证明通过，是下一次 OJ 目标验证候选 |
| C500 本地算力 | unavailable | 当前未提供 |
| Agent OJ 提交次数 | 6 | `#116962` CE/0 分；`#116973` Accepted/11 分；`#117017` CE/0 分；`#117034` Accepted/81.5 分；`#117056` Accepted/81.75 分/排名 23；`#117079` Accepted/79.25 分；均已报告 |
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

1. 当前目标正确性与性能基线为 `83fd3d26e95f7d3739a9f2522afe866ed76f59cb` / `exp-20260819-007`；OJ `#117056` 已验证其 MXMACA 编译、allocation-range shape 推断和 4/4 wave-MMA 正确性，C500 得分 81.75、排名 23。活动源码已由 revert commit `c2d5bcd` 恢复到该算子实现；`b6e4272e9d8f` 仅保留为标量兼容回退。
2. `exp-20260818-003` / `3b7f02efb795` 的 `__dp4a` 优化已被 OJ `#116962` 目标编译结果否决；不得再次提交该 intrinsic。
3. `#116973` 四组 C500 用户核时间为 41.947、333.856、21.359、170.268 ms，仅为官方 baseline 的 0.122x、0.068x、0.209x、0.127x；标量字节展开是主瓶颈。
4. `exp-20260819-005` / `067e38fdd6f3` 已按上述设计实现：MACA 分支的 128 次 MMA 调用及 A/B LDS/LDG 序列与官方 standalone kernel 逐项一致，A/scale_a 改为直接 routed-row 索引；NVIDIA 分支保持标量基线。
5. OJ `#117034` 已验证 `exp-20260819-006` / `cba47c3418a5` 的 MXMACA 编译与 4/4 wave-MMA 正确性，得分 81.5、排名 23。`exp-20260819-007` / `83fd3d26e95f` 只把固定最多 8 个 M tile 交错的网格恢复为官方 `grid_x=1` 等价公式；OJ `#117056` 得分 81.75、排名仍为 23，仅测试点 1 从 82 提至 83，prefill gate-up 仍为 73。该调度假设仅获边际收益，下一步分析测试点 2 的 MMA 流水和访存并行度。
6. `exp-20260819-008` / `d24ad933e174` 只为 prefill gate-up 增加官方 raw-array 风格的 128x256x128、256 线程 G2S MMA 路径。OJ `#117079` 虽 4/4 正确，但总分降至 79.25，目标测试点 2 从 73 降至 63、用户核时间从 8 ms 增至 13 ms。该假设已被否决：每 K tile 同步、寄存器压力和 48 KiB LDS 的代价超过 A 复用收益，后续保留已验证的 128x128 流水核心。
7. `exp-20260819-009` / `7d4eef642c15` 已验证固定启动开销假设：四个公开 shape 的 A allocation size 唯一，因此 shape 推断热路径先由 A 成功返回，只在 A 查询失败时回退到 out 查询；MMA 内核、launch geometry 和算术均保持不变。proxy/NVIDIA 同进程微基准从 188.6 ns 降至 93.9 ns（2.008x），完整 correctness、benchmark 和 regression 通过；但绝对只减少约 94.7 ns，相对最快的 558 us OJ 测试点约 0.017%，不单独消耗一次 OJ 提交。
8. `exp-20260819-010` / `8c519e6c1bb5` 只把 MACA MMA 的 A global load 从行谓词版本改为基线已用于 B 的无谓词 `__builtin_mxc_ldg_b128`，并删除死的 `row_a_mask`。合同四个 shape 的 EM 和 K 都精确按 128 对齐；逐线程回归证明每个 load 行都在当前 tile 内，gate-up/down 每线程分别移除 224/64 次谓词。完整代理门禁通过，下一步只对该 commit 做一次 OJ 目标编译、正确性和性能验证；本轮不再提交 OJ。

## NVIDIA 执行链路验证

- `exp-20260818-000` / `6020d29b591c`：请求 GPU 0 时检测到已有负载，预检以 125 拒绝；没有启动 CUDA 任务，项目锁已释放。
- `exp-20260818-001` / `e59225b509a5`：物理 GPU 1（RTX A5000，compute 8.6）空闲；使用 CUDA 12.2 编译并运行最小内核，返回 `PASS` 和数值 42，runner 退出 0。
- `exp-20260818-002` / `ccabad8ab154`：物理 GPU 1 上建立 Fused MoE CUDA 基线。随机、tile 边界、INT8 极值和零值回归均以 `matched_ratio=1.0`、`max_abs=0` 通过；4 个公开 shape 的 allocation-range 推断全部通过；只读输入未被修改。四个 shape 的 proxy/NVIDIA median 分别为 45.584、341.346、21.856、173.180 ms（1 次预热、5 次记录）。
- `exp-20260818-003` / `3b7f02efb795`：只把逐字节标量 dot4 替换为 packed signed INT8 `__dp4a`。相同测试和方法下，四个 shape 的 proxy/NVIDIA median 分别为 38.742、277.980、17.531、140.046 ms；相对基线降低 15.01%-19.79% 延迟。全部正确性与回归继续以零差异通过。
- `exp-20260818-004` / `b6e4272e9d8f`：根据 OJ `#116962` 的 CE 恢复官方兼容标量 dot4。全部正确性和回归零差异通过；四个 shape 的 proxy/NVIDIA median 为 43.577、341.350、21.775、173.166 ms。该轮是目标兼容性回退，不宣称保留 `__dp4a` 的 NVIDIA 提速。
- `exp-20260819-005` / `067e38fdd6f3`：只在 MACA 编译分支加入官方 `__builtin_mxc_mma_16x16x16i8` 的 128x128x128 pipeline，并将旧 token gather 语义改为实时 OJ 的 routed-row 直读；NVIDIA 标量兼容分支未改。随机/边界/INT8 极值/零值/只读输入/四公开 shape 推断/公开 shape 抽样全部通过，128x128 输出映射 16,384 个元素恰好覆盖一次。proxy/NVIDIA 四组 median 为 45.529、338.711、21.707、171.755 ms；目标 MACA 分支未在 NVIDIA 上执行，决定为 `investigate`，等待 OJ 目标验证。
- `exp-20260819-006` / `cba47c3418a5`：只把 MMA 内核 `a_base` 的声明与 reset 改为通过 `const_cast<int8_t*>(a_ptr)` 派生，适配 OJ `#117017` 显示的 MXMACA load builtin `void*` 形参；地址、数据与计算均未改变。完整 correctness、benchmark 和 regression 通过；proxy/NVIDIA 四组 median 为 45.533、338.750、21.550、172.007 ms，与上一版处于同一噪声范围。随后 OJ `#117034` 以 4/4 Accepted、81.5 分完成目标验证。
- `exp-20260819-007` / `83fd3d26e95f`：只恢复官方按平均每专家行数计算的 M-grid 分组，四个合同 shape 均从固定最多 8 个交错 tile 改为 `grid_x=1`；M tile 集合和算术不变。新增回归确认四个 shape 的每个 M tile 恰好覆盖一次。完整 correctness、benchmark 和 regression 通过；proxy/NVIDIA 四组 median 为 45.541、338.749、21.697、173.208 ms。OJ `#117056` 随后以 4/4 Accepted 验证目标兼容性，得分从 81.5 增至 81.75；只有测试点 1 增加 1 分，未改善测试点 2。
- `exp-20260819-008` / `d24ad933e174`：仅对 prefill gate-up 增加 128x256x128 wide-MMA 路径，采用官方 256x256x128 raw-array 内核的 G2S、LDS 与 MMA 布局，同时把 M tile 保持为 128 以维持单专家语义；其余 shape 的目标路径不变。新增回归确认 32,768 个输出元素恰好覆盖一次且只有目标 shape 分派到新内核。完整代理门禁通过；proxy/NVIDIA 四组 median 为 45.530、338.708、21.548、171.745 ms。OJ `#117079` 以 4/4 Accepted 证明目标兼容性，但得分从 81.75 降至 79.25，prefill gate-up 从 73 降至 63；该候选已由 `c2d5bcd` 回退。
- `exp-20260819-009` / `7d4eef642c15`：只让 `infer_public_config` 在 A allocation size 成功匹配后立即返回，out allocation query 保留为 A 查询失败时的兼容回退；目标验证的 128x128 MMA 内核与 NVIDIA fallback 均未改变。物理 GPU 1 上 build、三组 correctness、四公开 shape 的 A 主路径/out 回退、MMA 输出映射、M-grid、只读输入和公开 shape 抽样均通过。proxy/NVIDIA shape inference 中位数为 93.9 ns，对照两次查询为 188.6 ns（2.008x）；四组核心 median 为 45.534、338.755、21.528、171.751 ms，处于既有噪声范围。OJ 是否计入驱动查询固定开销待目标验证。
- `exp-20260819-010` / `8c519e6c1bb5`：只移除 MACA 128x128 MMA 核心的 A-load 行谓词，B load、双缓冲指令序列、网格、输出和 NVIDIA fallback 均未改变。物理 GPU 1 上 source check、build、完整 correctness/benchmark/regression 通过；逐线程范围验证确认 gate-up/down 分别安全移除 224/64 次谓词。proxy/NVIDIA 四组 core median 为 45.375、338.715、21.527、171.780 ms，目标改动未在 NVIDIA 分支执行。无谓词 `__builtin_mxc_ldg_b128` 已被 OJ 基线的 B load 验证可编译，但 A 地址正确性和 C500 性能仍需 OJ 验证。
- 上述原始结果均位于本地忽略目录 `artifacts/raw/remote-runs/`，远端唯一 run 目录继续保留。

`exp-20260818-002` 到 `exp-20260819-010` 的 NVIDIA 性能数据只属于 proxy/NVIDIA，不证明 C500 性能或 OJ 得分。OJ `#116962` 已确认 MXMACA `xcore1000` 不声明 `__dp4a`；OJ `#116973` 已确认 allocation-range shape 推断在评测分配器中可用且标量实现 4/4 正确；OJ `#117034` 和 `#117056` 已确认当前 MMA 后端及 `grid_x=1` 调度能够在目标上编译并通过 4/4 正确性；OJ `#117079` 已否决 128x256 G2S 性能假设。

## XPU-OJ 提交记录

- `#116962` / `2026-08-18 23:22:21 +08:00` / `3b7f02efb795` / CUDA Maca：`Compilation Error`，0 分，0 us，0 B。错误为 `/sandbox/source/main.cu:84:12: use of undeclared identifier '__dp4a'`，编译目标 `xcore1000`。证据位于忽略目录 `artifacts/raw/xpuoj/116962/`；submission ledger 已登记、报告并清空门禁。
- `#116973` / `2026-08-18 23:46:20 +08:00` / `b6e4272e9d8f` / CUDA Maca：`Accepted`，4/4 正确，11 分，页面汇总 567 ms / 22.6 G，2026-08-19 榜单排名 31。四组用户核时间为 41.947、333.856、21.359、170.268 ms；证据位于忽略目录 `artifacts/raw/xpuoj/116973/`，submission ledger 已登记、报告并清空门禁。
- `#117017` / `2026-08-19 00:38:15 +08:00` / `067e38fdd6f3` / CUDA Maca：`Compilation Error`，0 分，排名仍为 31。CUTE/MACA 头文件解析成功；五处错误均为 A load 的 `const int8_t*` 不能转换为 MXMACA builtin 所需的 `void*`。证据位于忽略目录 `artifacts/raw/xpuoj/117017/`；submission ledger 已登记、向用户报告并清空门禁。
- `#117034` / `2026-08-19 01:06:38 +08:00` / `cba47c3418a5` / CUDA Maca：`Accepted`，4/4 正确，81.5 分，页面汇总 14 ms / 22.6 G，榜单排名 23。四组分数为 82、73、88、83，用户核时间为 1053 us、8 ms、567 us、4286 us；证据位于忽略目录 `artifacts/raw/xpuoj/117034/`，submission ledger 已登记、向用户报告并清空门禁。
- `#117056` / `2026-08-19 01:41:15 +08:00` / `83fd3d26e95f` / CUDA Maca：`Accepted`，4/4 正确，81.75 分，榜单排名 23。四组分数为 83、73、88、83，用户核时间为 1024 us、8 ms、558 us、4228 us；相对 `#117034` 仅测试点 1 增加 1 分。证据位于忽略目录 `artifacts/raw/xpuoj/117056/`，submission ledger 已登记、向用户报告并清空门禁。
- `#117079` / `2026-08-19 02:12:20 +08:00` / `d24ad933e174` / CUDA Maca：`Accepted`，4/4 正确，79.25 分，榜单最佳仍为 81.75、排名 23。四组分数为 83、63、88、83，用户核时间为 1014 us、13 ms、558 us、4196 us；相对最佳总分下降 2.5，测试点 2 下降 10。证据位于忽略目录 `artifacts/raw/xpuoj/117079/`，submission ledger 已登记、向用户报告并清空门禁；源码已通过 `c2d5bcd` 回退。
