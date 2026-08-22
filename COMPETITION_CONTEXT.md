# XH-202628 比赛长期上下文

最后核对日期：2026-08-22（Asia/Shanghai）

## 1. 当前参赛状态

- 题目编号：`XH-202628`。
- 题目名称：基于 AI Agent 开发范式的国产 GPU 大模型推理算子库优化。
- 本项目参加的是官方“赛题二”，不是“基于国产软件栈大模型推理前沿算子优化”的赛题一。
- 用户已完成报名、审核通过，已取得赛题二专用 XPU-OJ 账号并成功登录。
- 用户给出的当前打榜入口：<https://xpuoj.com/contest/12/problem/1>。
- 已于 2026-08-22 通过用户已登录的 OJ 页面再次只读确认题目为 `1. Agent 推理算子库优化 - Fused MoE i8 tn`，支持 CUDA Maca、Triton、TileLang，限制仍为 10000 ms / 4096 MiB，目标硬件为 C500。详细实时合同快照见 Skill reference；编码和每次提交前仍须复核页面变化。
- 已完成 19 次工作流内 OJ 提交；完整逐次记录见 `state/PROJECT_STATE.md` 和 `state/submission-state.json`。当前保留版本仍是 `#117114` / `8c519e6c1bb5` 的 exact-row A-load CUDA Maca 核心，4/4 Accepted、82.25 分。最新提交 `#122083` / `3584f6802448` 的 case-2 蛇形 N 调度候选虽 4/4 Accepted，但仅 82.00 分，已否决并恢复最佳源码。
- 2026-08-22 16:40 核对 contest 12 榜单：当前账号 `muxi2026C2047` 排名 24、最佳分 82.25；榜首 87 分。最佳版本四个测试点分数为 83、73、89、84；prefill gate-up 仍是主要短板。
- NVIDIA 服务器：`lynsdu2@10.0.33.75`。
- 已于 2026-08-18 完成该服务器的只读环境探测，并在用户明确许可后创建专用执行目录；后续远端 run 均只使用该隔离目录。
- 用户明确要求：在服务器创建任何工作目录之前，必须先取得他的明确许可。
- 用户已于 2026-08-18 回复“1、允许”，明确批准创建 `/home/user/lynsdu2/xh-202628-agent`；该目录已于 21:12:09 +08:00 创建并核对。
- 用户授权 GPU 策略可以更激进、以完成任务为主；落实为 GPU 0-7、最多 4 个并行 run，同时不抢占已有计算进程。
- 用户决定本地 `xh-202628-agent` 是唯一 Git 工作区；需要算力时，把已提交 commit 的项目快照传到服务器同名专用目录执行，再取回结果。
- GitHub 仓库为 `git@github.com:RibbonFish-SHU/xh-202628.git`，本机 SSH 身份已验证为 `RibbonFish-SHU`。

## 2. 这场比赛到底交什么

这不是只交一个“通用 Agent 平台”的比赛，也不是只交一段快 Kernel 的普通算子赛。它有两个相互关联的产出：

1. 在 XPU-OJ 提交符合题面 `run_kernel(...)` 契约的算子代码，先通过正确性和稳定性，再以 C500/MACA 上的性能打榜。
2. 提交能够复现优化过程的 Agent/Skill 工作流，以及源码、测试框架、性能脚本、性能报告、README、PPT、演示视频/答辩材料等完整作品。

因此，OJ 代码是性能成绩的直接载体；Agent/Skill 是作品的一部分，必须真实参与源码理解、候选实现、测试、性能分析、调优和多轮迭代。只有 prompt 包装、文档生成或简单补全，不符合赛题核心要求。

## 3. 可选任务与当前方向

赛题二有三个独立任务：

| 任务 | 主要目标 | 官方允许语言 | 关键数据类型/范围 |
| --- | --- | --- | --- |
| FlashInfer | Ragged/Paged Prefill、Paged Decode、MLA Paged Attention 等 | MACA C++、Triton、TileLang | BF16，长序列，多个 head dimension/page size |
| FlashAttention | `flash_attn_with_kvcache` | MACA C++、Triton、TileLang | BF16，head dimension 32-512，page size 16 |
| MCTLASS Fused MoE | `Fused MoE i8 tn` | MACA C++、Triton、TileLang | INT8 W8A8，真实模型 shape |

各任务独立打榜；可以做一个或多个，但性能评分只取成绩最好的任务，不叠加。当前先聚焦用户给出的 OJ 单题，不在没有证据的情况下同时铺开多个任务。

### Fused MoE 当前已知接口语义

官方教程中的 CUDA Maca 冒烟接口是：

```cpp
extern "C" void run_kernel(
    const int8_t* a,
    const int8_t* b_col_major,
    const float* scale_a,
    const float* scale_b,
    const float* moe_weights,
    const int32_t* token_ids,
    const int32_t* expert_ids,
    int64_t topk,
    __nv_bfloat16* out
);
```

官方教程还说明：`topk=8`，专家数为 256，`EM=num_tokens*8` 且为 128 的倍数，
`b_col_major` 为 `[expert, n, k]`，Gate-up 典型 `(N,K)=(4096,7168)`，
Down 典型 `(7168,2048)`。这些只是当前资料快照；提交前必须逐字核对实时题面。

不同官方文档对 Fused MoE 容差的摘要存在差异，因此不得把本文件中的任何容差当作最终规则。以实时 OJ 题面和评测报告为准，并把快照存入实验记录。

## 4. 评分与硬门槛

总分 100 分：

- 性能提升效果：60%。按各任务 XPU-OJ 榜单排名计分。官方资料说明 baseline 对应 OJ 约 50 分；只有高于 50 分且进入前 30 名才获得该项排名分，第 1 名 60 分，第 30 名 2 分。
- Agent/Skill 可复现性：20%。功能可复现为 5 分；性能复现达到标称成绩的 60%/80%/90% 时分别为 10/15/20 分。
- 文档说明与演示报告：20%。包括技术报告、README、运行说明、性能报告、Agent/Skill 文档、演示视频和答辩材料的完整性与工程可复现性。

正确性和基础稳定性是硬门槛。未通过时不进入性能排名，性能项为 0。

禁止识别或硬编码测试样例、跳过计算、牺牲正确性、利用未授权接口或评测漏洞。异常成绩可能被复测或取消。

## 5. 时间与提交渠道

- 官方方案和“选手入口”写明：初赛完整作品材料须在 **2026-09-05 前**发送至 `opensource@metax-tech.com`。
- 2026-09-20 前完成初审；10 月完善晋级作品；11 月终审擂台赛。
- 即使 XPU-OJ 页面显示更晚的比赛结束日期（既有检查曾看到 2026-10-31），也不能据此推翻 9 月 5 日的初赛材料截止时间。截止规则如有变化，只接受组委会正式通知。
- 邮件包需包含可复现源码、测试/测试框架、性能脚本、性能报告、Agent/Skill、PPT 和文档；还需同步报名系统审核通过的报名表。除报名表外，作品材料不得携带学校、老师和学生个人信息。

## 6. NVIDIA 与 C500 的边界

NVIDIA 服务器可以做：

- 建立 PyTorch/CPU 高精度参考实现和随机差分测试。
- 理解 shape、布局、量化、路由和数值误差。
- 开发 CUDA/Triton/TileLang 的可迁移算法原型。
- 验证编译流程、内存越界、竞态、边界条件和回归测试。
- 做 NVIDIA 代理 Benchmark，筛掉明显较差的实现。
- 沉淀 Agent 工作流、实验账本、Git 历史和报告材料。

NVIDIA 服务器不能证明：

- MACA 编译器能够编译同一份源码。
- C500 上的 wave/寄存器/LDS/缓存/带宽行为与 NVIDIA 相同。
- NVIDIA 上的相对快慢在 C500 上仍成立。
- 代码已达到最终榜单性能或可复现性要求。

所有 NVIDIA 数据必须标为 `proxy/NVIDIA`，不得写成 C500 结果。所有涉及榜单性能的结论只能来自 XPU-OJ 或之后获得的 C500 环境。没有 C500 本地算力时，OJ 是目标平台反馈通道，但不能靠频繁盲提交代替工程测试。

## 7. 正确工作流理解

最小闭环是：

```text
读取实时题面与接口
  -> 建立可运行正确性基线
  -> 记录基线和提交源码 commit
  -> 提出一个可证伪的性能假设
  -> 每轮只做一组可归因的修改
  -> Git commit
  -> git archive HEAD -> NVIDIA 执行镜像
  -> build -> test -> benchmark -> 回归
  -> 回收原始结果并记录 proxy/NVIDIA 证据
  -> XPU-OJ 提交并等待终态
  -> 保存结果和页面证据
  -> 向用户报告本次提交
  -> 再决定保留、回滚或进入下一轮
```

每次 OJ 提交必须对应一个可定位的 Git commit。不得提交只存在于编辑器或浏览器中的临时代码。

每次提交完成评测后，必须先向用户报告，才能进行下一次提交。如果当前交互环境无法在长任务中发送中间报告，则每个回合最多提交一次。

## 8. 浏览器与凭据

- XPU-OJ 提交需要用户的已登录浏览器会话。未来 Agent 可以在授权范围内通过浏览器自动化完成提交和读取结果。
- 不把 XPU-OJ 密码、Cookie、GitHub token、SSH 私钥写入仓库、日志、prompt 或聊天。
- 若登录失效，只要求用户在浏览器中重新登录；不要索取明文密码。
- 浏览器自动化在点击最终“提交”前，必须核对题目、语言和源码 commit。

## 9. Git 与远程计算架构

确定架构：

```text
本地 xh-202628-agent（唯一可信 Git 工作区）
  |-- git/gh/GitHub MCP -> GitHub（用户决定公开或私有）
  |-- 已登录浏览器 -> XPU-OJ / C500 目标评测
  `-- git archive HEAD -> SSH/SCP
                              |
                              v
NVIDIA 执行镜像 /home/user/lynsdu2/xh-202628-agent
  唯一 run 目录 + 无 .git 源码快照 + 测试/Benchmark 原始产物
                              |
                              `-> SCP 回收 results 到本地
```

批准的服务器绝对目录 `/home/user/lynsdu2/xh-202628-agent` 已创建。不得再次初始化，不得移动、删除或修改服务器上的其他项目。

远端不是 GitHub 同步端：不安装 `gh`，不保存 GitHub token/SSH key，不执行 clone/pull/push，也不手工修改快照源码。每次远端测试必须来自干净工作树的已提交 commit，通过 `git archive HEAD` 传输，并在唯一 run 目录执行。结果取回后在本地记录；未确定保留策略前不自动删除远端 run。

## 10. 尚未解决、不得擅自假设的问题

- GitHub 仓库可见性尚未单独确认，但 SSH 读写身份和现有 `main` 已确认。
- 远端自动清理仍未授权；达到 100 GiB 门禁时必须停止并报告。
- 当前 OJ 标题、CUDA Maca 接口、shape 和正确性口径已保存为合同快照；2026-08-19 提交前页面未显示配额或冷却限制，且四台 C500 评测机在线。TileLang 的完整签名仍需在实际选型前保存，提交限制和榜单状态仍须在每次提交前实时复核。
- 9 月 5 日之后 OJ 成绩是否冻结、统一复测细节、Agent 性能复现口径、终审版本规则等尚无正式答案。

GitLink issue #46 是“8 月 20 日 15:00-17:00 直播答疑”的问题征集帖。2026-08-18 核对时，该帖已关闭征集，评论主要是选手问题，不是官方答复。不得把评论中的推测写成比赛规则；应关注直播答疑或后续正式纪要并更新本文件。

## 11. 信息源与优先级

按具体事项分别判断权威来源：

- 实时接口、语言、测试范围、提交结果：XPU-OJ 当前题面与评测报告。
- 截止日期、材料和赛制：官方比赛方案、组委会正式通知、官方仓库当前版本。
- 本地官方资料克隆：`../official-op_optimization/`（本地工作区中的兄弟目录；默认不传到远端执行镜像）。
- 本地官方克隆核对 commit：`4f2aa14e92353e382e59bae98abe2c19e652ebd7`（commit 时间 2026-08-07T16:35:06+08:00）。
- 官方 GitLink 仓库：<https://www.gitlink.org.cn/metax-maca/op_optimization/>。
- 直播答疑问题征集：<https://www.gitlink.org.cn/metax-maca/op_optimization/issues/46>。
- XPU-OJ：<https://xpuoj.com/>。

如果来源冲突：保留原始证据，记录核对时间，不静默覆盖。接口问题以 OJ 实时题面为准；日期和材料问题向组委会确认。
