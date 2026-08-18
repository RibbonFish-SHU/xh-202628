# Agent Operating Contract

本文件适用于整个 `xh-202628-agent` 仓库，是后续工作 Agent 的强制规约。

## 1. 启动门禁

每个新会话开始时，按顺序完成：

1. 完整阅读 `COMPETITION_CONTEXT.md`、本文件和 `state/PROJECT_STATE.md`。
2. 只读取与当前任务相关的 Skill、reference 和 runbook。
3. 检查 `git status`、最近提交、实验记录和 `state/submission-state.json`。
4. 说明当前算子、接口契约版本、基线 commit、可用设备及未解除门禁。

不得仅凭历史对话摘要继续优化。事实变化时，先更新长期上下文和状态，再编码。

## 2. 唯一可信工作区

- 本地 `E:\XH-202628\xh-202628-agent` 是唯一可信 Git 工作区和源码源头。
- GitHub 的认证、建仓、pull/push 均在本地控制端完成。
- NVIDIA 服务器只是执行镜像：接收某个已提交 commit 的归档，运行测试，再返回结果。
- 远端镜像不包含 `.git`，不直接访问 GitHub，也不得成为手工改源码的第二工作树。
- XPU-OJ 操作使用本地控制端中用户已登录的浏览器会话。

本地与远端如有差异，一律以本地已提交 commit 为准。不得从远端反向覆盖本地源码；只能回收测试结果。

## 3. 权限与远端门禁

当前允许在本地仓库内创建和改进工作流、源码、测试、实验记录并管理 Git。用户也要求在条件满足后由工作 Agent 提交 XPU-OJ，并在每次评测结束后报告。

当前授权状态：

- **远端目录创建已授权但尚未执行。** 只允许初始化 `/home/user/lynsdu2/xh-202628-agent`；创建完成并记录前不得上传 run。
- **GPU 策略已授权。** 可用 GPU 0-7，最多 4 个并行 run；任何选中卡存在计算进程时必须拒绝启动。
- GitHub `origin` 已配置为 `git@github.com:RibbonFish-SHU/xh-202628.git`，使用本机已验证的 SSH 身份。
- 未提供 C500 本地算力；NVIDIA 结果不得声称为 C500 结果。

机器门禁位于 `state/remote-execution.json`。许可原文、时间、精确路径和 GPU 策略已经记录；必须先提交该状态，再初始化远端。不得为绕过脚本而擅自修改门禁。

远端只允许触及 `/home/user/lynsdu2/xh-202628-agent`。不得递归读取、索引、修改、移动、清理或借用服务器上的其他项目、环境和凭据。

## 4. 远端执行协议

每次 NVIDIA 测试严格遵守：

1. 在本地完成代码和一个位于 `remote-jobs/` 的可执行入口脚本。
2. 本地测试能够执行的部分，并记录一个明确、可证伪的实验假设。
3. 提交全部待测源码；要求工作树完全干净。
4. 使用 `scripts/invoke-remote-gpu-run.ps1`。该脚本只通过 `git archive HEAD` 传输已提交文件。
5. 远端在唯一的 `runs/<experiment-id>-<short-commit>/` 中解包并运行；不手工编辑源码，保留原始 `source.tar` 作为来源证据。
6. 回收 `results/` 到本地忽略目录 `artifacts/raw/remote-runs/`，核对退出码和原始日志。
7. 用实验账本记录 commit、设备、命令、结果路径和结论；只把经过检查的小型证据提交 Git。

禁止传输未提交文件、`.git`、秘密、构建树和无关文件。远端 run 默认永久保留，未取得保留/清理策略前不得自动删除或覆盖。

## 5. 优化纪律

1. 从实时 OJ 题面保存题目、语言、签名、shape、dtype、容差、限制和提交配额。
2. 建立独立参考实现、随机/边界正确性测试和最小可运行 baseline。
3. 记录环境、精确命令、原始输出、源代码 commit 和基线结果。
4. 每轮只验证一个性能假设，只做一组可归因修改。
5. 每轮执行 `build -> test -> benchmark -> regression`；失败实验也记录原因。
6. 提交 OJ 前确认源码 commit 等于已测试 commit。
7. 在 C500 可用前，所有 NVIDIA 性能判断只写成 `proxy/NVIDIA` 结论。

严禁修改官方测试标准、硬编码测试点、跳过计算或利用评测系统缺陷。

## 6. Git 纪律

- 默认分支为 `main`，除非目标 GitHub 仓库已有不同约定。
- 一个实验 commit 对应一个清晰假设；消息包含算子和实验 ID。
- 先 commit，再远端测试；如果测试导致后续修改，必须新建 commit 和实验 ID。
- 提交 OJ 的代码必须来自一个已记录且已测试的 commit。
- 不强推，不重写已经用于测试、OJ 或报告的历史。
- 不提交密码、token、Cookie、SSH 私钥、个人信息、构建产物和大型 profiler 文件。
- 不覆盖用户或其他 Agent 的未提交变更；对外 push 前检查状态和秘密。

建议 commit 格式：

```text
operator(scope): exp-YYYYMMDD-NNN concise hypothesis
```

## 7. XPU-OJ 提交门禁

每次提交执行：

1. 运行 `submission_ledger.py check`；存在待报告提交时立即停止。
2. 核对实时题目、语言、待提交文件、干净工作树和 commit。
3. 核对所有可用正确性/回归证据，并明确记录尚不能验证的 C500 缺口。
4. 通过用户已登录的本地浏览器提交；登录失效时只请求用户重新登录。
5. 等待评测终态，保存提交 ID、时间、状态、分数、排名/测试点和页面证据。
6. 用 `submission_ledger.py record` 记录结果，立即向用户报告。
7. 报告发出后才运行 `submission_ledger.py report`，之后才能再次提交。

编译错误、Wrong Answer、运行错误、超时和零分都算一次提交，必须记录并报告。若当前回合无法在两次提交间向用户报告，则该回合最多提交一次。

## 8. 证据与状态

每个实验至少记录实验 ID、时间、算子、设备、基线/候选 commit、假设、精确命令、正确性结果、原始性能数据、失败原因和保留/回滚决定。

以下事件后立即更新 `state/PROJECT_STATE.md` 和相关机器状态：用户批准远端目录或 GPU 策略、远端目录创建、GitHub 配置、基线变化、每次 OJ 提交及报告、取得 C500 算力、正式规则变化。

会话结束前，确保一个没有对话上下文的新 Agent 仅靠仓库即可安全接手。
