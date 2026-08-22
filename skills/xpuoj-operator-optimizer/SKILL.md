---
name: xpuoj-operator-optimizer
description: Drive reproducible XH-202628 GPU operator optimization with parallel isolated candidate agents and one centralized XPU-OJ coordinator, including live contract capture, committed NVIDIA proxy runs, candidate review, crash-safe target submissions, and deferred consolidated reports on user interruption. Use when onboarding or resuming this competition workspace, orchestrating Subagents, optimizing FlashInfer, FlashAttention, or Fused MoE kernels, running remote NVIDIA tests, preparing an XPU-OJ submission, or analyzing an OJ score.
---

# XPU-OJ Operator Optimizer

## Start

1. Locate the local repository root containing `AGENTS.md`.
2. Read `COMPETITION_CONTEXT.md`, `AGENTS.md`, and `state/PROJECT_STATE.md` completely.
3. Inspect Git status, recent commits, experiment records, `state/submission-state.json`, and `state/remote-execution.json`.
4. Determine the assigned role: Main Agent/coordinator, isolated candidate Producer, or independent read-only Auditor. Never infer the role from `HEAD` alone.
5. State the target problem, live-contract status, language, explicit baseline commit, available device class, and unresolved gates.
6. Treat the local Git repository as authoritative. Never edit source in the remote execution mirror.

## Load Only Needed Guidance

- For first-time remote directory creation, read `../../runbooks/remote-bootstrap.md` and stop while permission is pending.
- For a remote NVIDIA run, read `../../runbooks/remote-execution.md` and `references/nvidia-maca-boundary.md`.
- For a parallel batch, all roles must read `../../runbooks/parallel-orchestration.md`; follow only the section for the assigned role and use the matching task template.
- For local GitHub setup, read `references/github-sync.md`.
- For Fused MoE, read `references/fused-moe-contract.md`, then capture the authenticated live OJ contract.
- Before browser automation or an OJ submission, read `references/submission-protocol.md`.
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
- Cover representative, random, boundary, and adversarial layouts without hidden-test assumptions.
- Reproduce the starter baseline before changing performance-sensitive code.
- Save exact commands, seeds, environment, raw output, and source commit.
- Never relax official tolerances or modify the evaluator to manufacture success.

## Produce And Review Candidates

The Main Agent owns experiment allocation, explicit baselines, isolated worktrees, candidate integration, formal state and submission priority. Producers work only in assigned candidate worktrees. Auditors work only in detached audit worktrees and never edit or enqueue. Neither role may touch `main`, GitHub, formal state, controller credentials, browser or OJ.

For each lane:

1. Main Agent assigns one falsifiable hypothesis, `exp-YYYYMMDD-NNN`, candidate ID, current workflow-bearing `worktree_base_commit`, and explicit measured `performance_baseline_commit`. The two commits must contain the same submission-source blob; never check out an old best commit that lacks current tools.
2. Subagent makes one attributable change and runs all locally available checks.
3. Create a deterministic `remote-jobs/<experiment-id>.sh` when NVIDIA testing is required.
4. Commit the candidate and `handoffs/<experiment-id>.md`; require a clean worktree.
5. Run `scripts/invoke-remote-gpu-run.ps1`; never transfer files by ad hoc recursive copy.
6. Producer inspects raw evidence, completes the committed handoff, and enqueues the immutable candidate with `submission_controller.py candidate-enqueue`. Producer must not append tracked `state/experiments.jsonl`.
7. Main Agent serially imports the handoff into the experiment ledger, requests an independent audit when required, and integrates/promotes only candidates that meet the documented gate. Use a new experiment and candidate ID whenever submitted source changes.

Label NVIDIA measurements `nvidia`/`proxy`. Never convert NVIDIA speedup into a claimed C500 speedup. Do not delete remote runs until the user approves a retention policy.

## Decide Whether To Submit

Submit only when one condition holds:

- A first minimal-correctness submission is needed to validate the OJ path.
- A candidate passed all available regression checks and has a documented reason to improve C500 behavior.
- A failure fix requires target feedback unavailable locally.

Check live quotas and the centralized candidate/claim state. Do not submit parameter sweeps without an explicit experiment design. Waiting for one OJ result must not stop unrelated Subagents from producing candidates.

## Submit And Report

Follow `references/submission-protocol.md` exactly:

1. Main Agent acquires the centralized controller and runs `submission_controller.py check`.
2. Promote one integrated candidate, claim the global slot, and persist `arm` before the final click.
3. Submit the exact committed source through the user's authenticated local browser without exposing credentials, then immediately `bind` the numeric submission ID.
4. Wait for a terminal result, save evidence, and run `finalize`; this records the result and releases the slot.
5. Persist formal state, restore the formal-best source after a losing candidate, commit/push, and continue without waiting for a user report.
6. When the user interrupts, asks for status, the task ends, or user action is required, run `unreported-list`, send one consolidated report based on `../../templates/submission-report.md`, then mark each delivered result with `report`.

Treat compile errors, Wrong Answer, crashes, timeouts, and zero scores as terminal submissions. Record all of them; deferred user reports do not block the next submission.

If a target result does not become the formal best, restore the submission source from the explicit formal-best commit, verify its exact source hash, commit/push that restoration, and only then integrate another candidate. Never accumulate an untested combination on top of a losing candidate.

## Finish

Complete an iteration only when code, exact commands, raw evidence, experiment record, Git commit, and decision agree. An OJ iteration is internally complete after terminal evidence is finalized; its user-visible report may remain deferred until interruption.

Main Agent updates `state/PROJECT_STATE.md` before ending so a new coordinator can resume without conversation history. Producers finish their committed handoff and queue entry. Auditors return read-only findings directly to Main Agent.
