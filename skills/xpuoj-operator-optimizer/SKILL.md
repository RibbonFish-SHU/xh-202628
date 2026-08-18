---
name: xpuoj-operator-optimizer
description: Drive reproducible XH-202628 GPU operator optimization from an authoritative local Git worktree, including live XPU-OJ contract capture, committed-snapshot NVIDIA proxy runs over SSH, correctness and benchmark evidence, C500 target submissions, and mandatory per-submission user reports. Use when onboarding or resuming this competition workspace, optimizing FlashInfer, FlashAttention, or Fused MoE kernels, running remote NVIDIA tests, preparing an XPU-OJ submission, or analyzing an OJ score.
---

# XPU-OJ Operator Optimizer

## Start

1. Locate the local repository root containing `AGENTS.md`.
2. Read `COMPETITION_CONTEXT.md`, `AGENTS.md`, and `state/PROJECT_STATE.md` completely.
3. Inspect Git status, recent commits, experiment records, `state/submission-state.json`, and `state/remote-execution.json`.
4. State the target problem, live-contract status, language, baseline commit, available device class, and unresolved gates.
5. Treat the local repository as authoritative. Never edit source in the remote execution mirror.

## Load Only Needed Guidance

- For first-time remote directory creation, read `../../runbooks/remote-bootstrap.md` and stop while permission is pending.
- For a remote NVIDIA run, read `../../runbooks/remote-execution.md` and `references/nvidia-maca-boundary.md`.
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

## Run One Measured Iteration

1. Write one falsifiable bottleneck hypothesis and assign `exp-YYYYMMDD-NNN`.
2. Make one attributable change and run all locally available checks.
3. Create a deterministic `remote-jobs/<experiment-id>.sh` when NVIDIA testing is required.
4. Commit the candidate and require a clean worktree.
5. Run `scripts/invoke-remote-gpu-run.ps1`; never transfer files by ad hoc recursive copy.
6. Inspect returned raw evidence and record the experiment with `scripts/record_experiment.py`.
7. Keep, supersede, or revert based on evidence. Use a new commit and experiment ID for every changed candidate.

Label NVIDIA measurements `nvidia`/`proxy`. Never convert NVIDIA speedup into a claimed C500 speedup. Do not delete remote runs until the user approves a retention policy.

## Decide Whether To Submit

Submit only when one condition holds:

- A first minimal-correctness submission is needed to validate the OJ path.
- A candidate passed all available regression checks and has a documented reason to improve C500 behavior.
- A failure fix requires target feedback unavailable locally.

Check live quotas and queue state. Do not submit parameter sweeps without an explicit experiment design.

## Submit And Report

Follow `references/submission-protocol.md` exactly:

1. Run `python skills/xpuoj-operator-optimizer/scripts/submission_ledger.py check`.
2. Verify clean source provenance and the tested commit hash.
3. Submit through the user's authenticated local browser without exposing credentials.
4. Wait for a terminal result and save evidence.
5. Record the result with `submission_ledger.py record`.
6. Send the user a report based on `../../templates/submission-report.md`.
7. Only after sending that message, run `submission_ledger.py report`.

Treat compile errors, Wrong Answer, crashes, timeouts, and zero scores as reportable submissions. Never make another submission while a report is pending.

## Finish

Complete an iteration only when code, exact commands, raw evidence, experiment record, Git commit, and decision agree. Complete an OJ iteration only after terminal OJ evidence and the user report exist.

Before ending, update `state/PROJECT_STATE.md` so a new Agent can resume without conversation history.
