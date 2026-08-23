# State And Ledgers

- `PROJECT_STATE.md`：人类和 Agent 都应首先阅读的当前状态。
- `c500-execution.json`：当前 C500 alias、执行镜像、单槽/slice 门禁、工具链和已观测硬件指纹；不含凭据。
- `remote-execution.json`：已停用的 NVIDIA 历史执行配置，只用于解释/恢复旧结果。
- `experiments.jsonl`：只由 Main Agent 根据 committed handoff 串行调用 `record_experiment.py` 追加；Producer/Auditor 不修改。
- `submission-state.json`：由 Main Agent 的 `submission_controller.py` 导出的受跟踪 OJ 历史镜像；不作为并发控制真源。

共享运行时数据库位于 Git common directory 的 `.git/xh-202628/submission-control.sqlite3`。它在所有 linked worktree 之间原子协调 candidate queue、controller epoch、唯一 active claim 和延迟用户报告队列，不进入 Git。终态 `finalize` 后槽位立即释放；`reported_to_user=false` 只表示等待用户打断后的汇总，不阻止后续提交。旧 `submission_ledger.py` 在该数据库存在后只保留只读兼容，不能写提交历史。

Producer/Auditor 不修改本目录中的正式状态。Producer 将候选 handoff 提交到 `handoffs/`，并只通过 `candidate-enqueue` 写共享运行时队列；Auditor 不写队列。

JSON/JSONL 中只保存可公开的技术元数据和相对产物路径。不得保存密码、token、Cookie、SSH 私钥或含个人信息的页面内容。
