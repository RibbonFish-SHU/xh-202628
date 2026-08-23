---
name: xpuoj-operator-optimizer
description: Optimize XH-202628 GPU operators with parallel isolated candidate agents, native MetaX C500 build/correctness/paired benchmarks, target-aware review, and one centralized XPU-OJ submission lane. Use when starting or resuming this competition workspace, producing or auditing operator candidates, running C500 tests, preparing an XPU-OJ submission, or analyzing target scores.
---

# XPU-OJ Operator Optimizer

## Start

1. Locate the repository root containing `AGENTS.md`.
2. Read `COMPETITION_CONTEXT.md`, `AGENTS.md`, and `state/PROJECT_STATE.md` completely.
3. Inspect Git status, recent commits, experiment records, `state/submission-state.json`, and `state/c500-execution.json`.
4. Determine the assigned role: Main Agent/coordinator, isolated Candidate Producer, or independent read-only Auditor. Never infer the role from `HEAD` alone.
5. State the target problem, live-contract status, language, explicit baseline commit, C500 availability, and unresolved gates.
6. Treat local committed Git as authoritative. Never edit source in a remote execution mirror.

## Load Needed Guidance

- For a C500 run, read `../../runbooks/c500-execution.md` and `references/c500-native-optimization.md`.
- For a parallel batch, all roles read `../../runbooks/parallel-orchestration.md`, then follow only their assigned role and template.
- For Fused MoE, read `references/fused-moe-contract.md`, then capture the authenticated live OJ contract.
- Before browser automation or an OJ submission, read `references/submission-protocol.md`.
- For GitHub setup, read `references/github-sync.md`.
- Read `references/nvidia-maca-boundary.md` only when interpreting historical NVIDIA evidence. New NVIDIA runs are disabled.
- For another operator, use `references/official-source-map.md`, then create an operator-specific contract snapshot.

Do not load unrelated references merely because they exist.

## Freeze The Contract

Capture from the authenticated live problem page:

- Contest/problem title and URL.
- Supported language and exact `run_kernel(...)` signature.
- Shapes, layouts, dtypes, semantics, tolerance, stability and forbidden techniques.
- Time/memory limits, submission quota, capture time, and evidence path.

Follow the live interface when local documents differ. Escalate deadline or eligibility conflicts instead of guessing.

## Establish Correctness

- Build an independent CPU/PyTorch reference where feasible.
- Cover representative, random, boundary, adversarial and read-only-input cases without hidden-test assumptions.
- Reproduce the starter/formal baseline before changing performance-sensitive code.
- Save exact commands, seeds, environment, raw output and source commit.
- Never relax official tolerances or modify the evaluator to manufacture success.

## Produce C500 Candidates

The Main Agent owns Agent delegation, experiment allocation, explicit baselines, isolated worktrees, C500 scheduling, candidate integration, formal state and submission priority. Producers work only in assigned candidate worktrees. Auditors work only in detached audit worktrees and never edit or enqueue. Neither role may spawn another Agent or touch `main`, GitHub, controller credentials, browser or OJ.

For each lane:

1. Main Agent assigns one falsifiable hypothesis, unique IDs, current workflow-bearing `worktree_base_commit`, and measured `performance_baseline_commit`. The two base commits must start from the same submission-source blob and are atomically fixed by the experiment and baseline reservation refs.
2. Producer makes one attributable source change, copies `templates/remote-job.sh` byte-for-byte to `remote-jobs/<experiment-id>.sh`, and commits candidate plus initial handoff. Producer does not connect to C500.
3. Main Agent runs `scripts/invoke-c500-run.ps1` from a clean trusted control worktree with explicit candidate, performance-baseline and workflow-base commits. The caller verifies both reservation refs; the remote trees come from the workflow commit and only the two committed submission sources differ.
4. Require C500 build, correctness, regression, warmup and strict four-case ABBA paired benchmark. Preserve raw samples, `mx-smi` snapshots, toolchain/slice fingerprint, archive/source hashes and the result manifest.
5. Interpret results with `references/c500-native-optimization.md`. A local paired delta is target-relevant evidence, but not an OJ score prediction while slice equivalence is unknown.
6. Producer records the returned evidence in a metadata-only handoff commit without changing the tested source/job blobs, then enqueues exactly one immutable candidate with `submission_controller.py candidate-enqueue`. Producer never appends tracked `state/experiments.jsonl`.
7. Main Agent serially imports the handoff, verifies the queued source blob equals the tested source blob, reviews target codegen/evidence, requests an independent audit for synchronization or mapping risk, and integrates/promotes only candidates meeting the documented gate.

Do not delete C500 runs until the user approves a retention policy.

## Target-Aware Decisions

Use C500 hardware evidence instead of NVIDIA analogies. Check wave64 mapping, 128-byte transaction efficiency, 32 KiB L1 / 8 MiB L2 behavior, LDS/barrier cost, register pressure, occupancy and xcore1000 lowering. Use `mcProfiler` when it answers a specific bottleneck question; keep profile runs separate from timing runs.

Submit only when one condition holds:

- A first minimal-correctness submission is needed to validate the OJ path.
- A candidate passed C500 correctness/regression and shows stable paired improvement with a documented target mechanism.
- A failure fix or diagnostic requires OJ feedback unavailable on the local C500.

Do not submit blind parameter sweeps. Waiting for one OJ result must not stop unrelated Subagents from producing candidates.

## Submit And Report

Follow `references/submission-protocol.md` exactly:

1. Main Agent acquires the centralized controller and runs `submission_controller.py check`.
2. Promote one integrated candidate, claim the global slot, and persist `arm` before the final click.
3. Submit the exact committed source through the user's authenticated local browser, then immediately `bind` the numeric submission ID.
4. Wait for terminal evidence and run `finalize`; this records the result and releases the slot.
5. Persist formal state, restore the formal-best source after a losing candidate, commit/push, and continue without waiting for a user report.
6. When the user interrupts, asks for status, the task ends, or user action is required, run `unreported-list`, send one consolidated report, then mark delivered results with `report`.

Compile errors, Wrong Answer, crashes, timeouts, cancellations and zero scores are terminal submissions and must be recorded. Deferred reports do not block the next submission.

If a target result is not the formal best, restore from the explicit formal-best commit, verify its exact source hash, commit/push that restoration, and only then integrate another candidate. Never accumulate an untested combination on a losing candidate.

## Finish

An iteration is complete only when code, exact commands, raw evidence, experiment record, Git commit and decision agree. Main Agent updates `state/PROJECT_STATE.md` before ending so a new coordinator can resume without conversation history. Producers finish their committed handoff and queue entry; Auditors return read-only findings directly to Main Agent.
