# State And Ledgers

- `PROJECT_STATE.md`：人类和 Agent 都应首先阅读的当前状态。
- `experiments.jsonl`：由 `record_experiment.py` 逐行追加的实验记录，首次记录时自动创建。
- `submission-state.json`：由 `submission_ledger.py` 管理的 OJ 提交和报告门禁。

JSON/JSONL 中只保存可公开的技术元数据和相对产物路径。不得保存密码、token、Cookie、SSH 私钥或含个人信息的页面内容。
