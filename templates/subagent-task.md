# Subagent Task Envelope

由 Main Agent 填写后原样交给一个 Subagent。每个字段都必须明确，不能让 Subagent自行猜测。

```text
role: candidate-producer
batch: batch-YYYYMMDD-NN
lane: <lane-name>
candidate_id: cand-...
experiment_id: exp-YYYYMMDD-NNN
reservation_ref: refs/xh-202628/experiments/<experiment-id>
worktree: <absolute-linked-worktree-path>
branch: candidate/<batch>/<lane>-<experiment>
worktree_base_commit: <current-main full-40-char-commit containing workflow tools>
performance_baseline_commit: <full-40-char measured source baseline>
performance_baseline_source_sha256: <64-char-sha256>
source_path: <repository-relative-submission-source>
language: <exact-XPU-OJ-language>
target_cases: <case IDs or shapes>
code_region: <one owned region>
mechanism_family: <one mechanism family>
hypothesis: <one falsifiable sentence>
expected_structural_change: <bytes/barriers/CTA/LDS/threads/instructions>
required_checks: <exact commands or named gates>
assigned_gpu_ids: <IDs or none>
historical_denylist: <mechanisms that must not be repeated>
stop_conditions: <conditions that end this lane>
```

## Mandatory Instructions

- Work only in the assigned worktree and branch. Confirm `HEAD == worktree_base_commit` and do not check out the historical performance baseline; its exact source is already present in the worktree base.
- Do not switch to or modify `main`; do not pull, push, rebase shared history or operate the browser/OJ.
- Do not modify `state/PROJECT_STATE.md`, `state/experiments.jsonl`, `state/submission-state.json` or the controller database directly. The only allowed controller write is `candidate-enqueue`.
- Change only the assigned mechanism. If the hypothesis needs expansion, report it instead of silently widening scope.
- Use the assigned experiment ID exactly once. Commit all tested source, tests, remote job and handoff.
- Run the required correctness and regression gates. Label NVIDIA evidence only as `proxy/NVIDIA`.
- Write `handoffs/<experiment-id>.md` from `templates/subagent-handoff.md`.
- Enqueue exactly one immutable committed candidate with `submission_controller.py candidate-enqueue`.
- Return the full base commit, candidate commit, handoff path, concise result, risk and keep/reject/investigate recommendation to Main Agent.
- Do not act as an independent auditor for your own candidate. The Main Agent uses `templates/auditor-task.md` when a separate audit is required.
