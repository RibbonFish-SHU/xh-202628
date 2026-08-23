# C500 Native Optimization

当任务涉及 C500 本地测试、候选晋级或目标硬件优化时读取本 reference。

## 已观测环境

| 项目 | 当前值 |
| --- | --- |
| 物理设备 | MetaX C500，xcore1000 |
| MACA / MXCC | 3.7.1.5 / 1.0.0 (`d9102a1572`) |
| Driver / mx-smi | 3.8.30 / 2.3.1 |
| 当前可见 slice | 25% compute，16000 MiB VRAM，device 0 |
| 物理显存 | 64 GiB HBM2e |
| 执行模型 | wave64，104 AP（物理卡） |
| Cache | 128-byte cache line；每 AP 32 KiB L1；8 MiB L2 |
| 最高 xcore clock | 1600 MHz；空闲时会降频 |

这些值必须由每次 run 的 `environment.txt` 和 `mx-smi-*` 再确认。OJ 使用 C500/MACA，但尚未确认其 compute/VRAM slice 与本地完全一致。因此：编译器、ISA、wave、同步和缓存方向具有很高目标相关性；本地绝对时间、吞吐和预期 OJ 分档仍不是等价量。

## 晋级实验协议

1. Main Agent 用显式 `WorkflowCommit` 固定两臂的受信 harness、runner 和编译参数，只分别叠加 candidate/baseline submission source；Producer 不连接 C500。
2. experiment 与 baseline reservation ref 必须分别等于 `WorkflowCommit` 和 `BaselineCommit`；不要接受调用者自行声明的控制面或配对基线。
3. 先 build，再跑 candidate correctness 与 regression。失败不进入性能比较。
4. 先做完整 workload 预热，保存预热后 clock；不要在冷启动数据上作结论。
5. 使用 ABBA 顺序，并保留每次 harness 的全部原始 samples。每份日志必须有公开四 case、每 case 五个正有限 sample、可重算 median 和 sampled correctness PASS；比较 paired median、两端 drift 和 source hash。
6. 基线漂移明显或候选收益接近噪声时，用下一 attempt 重复。小于约 1% 的结果至少需要两个独立 ABBA run 同方向，且不能靠选取单次最小值成立。
7. slice 配额、MACA/MXCC、driver 或编译参数变化后，历史绝对时间失效；在新环境重测 paired baseline。
8. 只接受已验证 terminal status、workflow/candidate/baseline archive hash、两份 source hash 和 result manifest 一致的回收结果；`.partial-*` 不是正式证据。
9. `env -i` 只清理继承环境；当前 root candidate 没有文件系统沙箱。不要让未审阅 host code 上机，也不要把树前后哈希写成路径/秘密隔离保证。

写结论时使用 `c500-local`。可写“同机配对快 X%”；除非确认 OJ slice 和计时合同相同，不写“预计 OJ 快 X%/增加 Y 分”。

## 利用硬件一致性的优化方向

### 先看目标 codegen

- 对 candidate 与 baseline 比较 MXCC 生成物、寄存器、LDS、barrier 和目标指令数量；source-level 简化没有形成 codegen 差异时，不消耗 OJ 提交。
- 使用 `mcProfiler` 定位 compute、HBM/L2、LDS、stall 或 occupancy 瓶颈；只选能回答当前假设的 metrics。Profiler 多 pass 会扰动 wall-time，profile run 与 timing run 分开。
- 记录编译器版本、完整命令、commit、source SHA-256 和 profiler 输出路径。不要根据 NVIDIA PTX/SASS 推断 xcore1000 lowering。

### Wave64 与并行映射

- 以 64 lanes 为真实 wave 边界证明 shuffle、lane ownership、收敛和跨 lane 数据交换；不能沿用 warp32 的隐含假设。
- 当前 256-thread CTA 对应 4 个 wave。改变 threads/CTA、tile ownership 或 producer/consumer 分工时，同时验证有效 AP occupancy、寄存器和 LDS 驻留，不只看线程数。
- barrier、BSM、异步 copy 和 LDS 复用必须有目标语义证明和独立审计。OJ 已经否决过多个“能编译但 completion/visibility 错误”的 BSM 假设；本地 correctness 现在应先捕获这类错误。

### 内存层级

- 围绕 128-byte cache line 检查全局访问对齐、连续性和每个 transaction 的有效字节。优先消除跨 line、重复 transaction 和低利用率 gather，而不是只减少源码 load 数。
- 32 KiB L1 很小；大 expert 权重不能假定驻留。利用真实 expert 分布测试 L2 reuse、CTA 调度和 B/scale-B locality，避免只用单 expert或全 1 数据得出缓存结论。
- LDS 缩减只有在提高实际 occupancy 或减少关键 barrier/stall 时才有价值。K64 实验已经表明减半 LDS 但翻倍 barrier 会显著回退。
- 对 vector load/store 同时证明地址对齐、边界覆盖、只读输入和尾部行为。128-byte transaction 目标不等于每线程都应做最宽 load。

### Fused MoE 当前优先级

- 当前正式最佳主要短板是 `32768x4096x7168` prefill gate-up（case 2）。优先让 profiler 判断它是 MMA、B/scale-B traffic、LDS/barrier、occupancy 还是调度受限。
- 公开 shape 可做 xcore1000 常量特化，但只有生成更好的目标代码且 paired case 收益稳定时才保留。避免增加无收益的实例体积或 instruction-cache 压力。
- 对 expert locality、row metadata、epilogue scale、A/B tile 生命周期建立字节和指令模型；一次只验证一个机制。参考 `state/PROJECT_STATE.md` 的 target denylist，不能重复已经由 C500/OJ 否决的 BSM、K64、直接 global-A 等机制，除非有新的硬件证据改变前提。

## C500 与 OJ 的分工

C500 本地环境现在是候选生产的首要门禁：用于目标编译、正确性、回归、paired timing 和 profiler。XPU-OJ 只用于最终计分合同、未公开环境差异和排行榜确认。除首个路径验证或明确诊断外，不再把未经 C500 paired 证明的参数 sweep 送到 OJ。

同机本地提升仍可能因 OJ slice、系统负载、编译封装或计时方式不同而不涨分。OJ 结果是最终裁决；掉分候选仍按正式恢复协议回到 formal best。
