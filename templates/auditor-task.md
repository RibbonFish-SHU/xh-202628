# Independent Auditor Task Envelope

由 Main Agent 填写后交给一个没有参与该候选实现的 Auditor。Auditor 只做独立复核，不生产候选，不写共享状态。

```text
role: independent-auditor
batch: batch-YYYYMMDD-NN
lane: <producer-lane>
experiment_id: exp-YYYYMMDD-NNN
candidate_id: cand-...
audit_worktree: <absolute-detached-worktree-path>
candidate_branch: candidate/<batch>/<lane>-<experiment>
worktree_base_commit: <full-40-char workflow commit>
performance_baseline_commit: <full-40-char measured source baseline>
candidate_commit: <full-40-char-commit>
source_path: <repository-relative-submission-source>
source_sha256: <64-char-sha256>
hypothesis: <producer hypothesis>
audit_scope: <synchronization/addressing/contracts/etc.>
required_read_only_checks: <exact commands or proofs>
known_risks: <specific suspected failure modes>
deadline_or_stop_condition: <explicit condition>
```

## Mandatory Instructions

- Confirm the worktree is detached at the exact candidate commit, the worktree base is its ancestor, and the worktree-base source exactly matches the explicit performance-baseline source.
- Do not edit tracked files, create commits or branches, enqueue/reject/promote candidates, modify formal state, access GitHub, use the browser, or operate XPU-OJ.
- Build or test only when the task explicitly requests it. Generated build output stays disposable and must not become a commit or handoff mutation.
- Review the complete candidate diff against the assigned base. Check contract, coverage/addressing, read-only inputs, synchronization, boundary cases, launch geometry and claimed performance mechanism.
- Treat producer notes as claims to verify, not as evidence by themselves. Cite exact file/line or command evidence for every blocking finding.
- Return findings ordered by severity, then residual risks and a clear `approve | reject | needs-fix` recommendation directly to Main Agent.
