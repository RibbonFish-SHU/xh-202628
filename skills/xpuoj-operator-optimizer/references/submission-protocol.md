# XPU-OJ Submission Protocol

Every final submit click is an external action and must be traceable.

## Preflight

1. Read `state/PROJECT_STATE.md` and run:

   ```bash
   python skills/xpuoj-operator-optimizer/scripts/submission_ledger.py check
   ```

2. Confirm there is no pending report and no judging submission from this Agent.
3. Record contest/problem URL, visible title, selected language and any quota/cooldown.
4. Require a Git commit containing the exact submitted source.
5. Verify `git diff <commit> -- <source>` is empty for the submitted file.
6. Run available correctness/regression gates and reference the experiment ID.
7. Hash or archive the exact submission text in a non-secret artifact.

## Browser Action

- Use `$kimi-webbridge` with the user's existing authenticated browser profile. Run its required health check first; if unhealthy, follow that Skill's operations guide rather than improvising.
- Open the problem in a new tab under a dedicated session such as `xpuoj-submit-<experiment-id>`. Do not attach to and later close an unrelated user tab.
- Use `snapshot` and semantic element references before CSS or JavaScript selectors. If authentication is missing, stop and ask the user to log in manually.
- Never read, store or transmit passwords, Cookies or session tokens.
- Recheck title and language immediately before submit.
- Fill the exact committed source; compare a hash/length after filling where practical.
- Click submit once. Avoid double clicks and do not retry until the submission list is checked.
- The user's standing instruction authorizes a gated submission; do not request redundant confirmation when every preflight condition passes. Stop for any page, language, quota or source mismatch.

## Result Capture

Wait until the submission reaches a terminal state. Capture:

- Submission ID and timestamps.
- Commit and experiment ID.
- Status, correctness/stability, score, rank, runtime/memory and per-case details available.
- Screenshot or textual page evidence path.
- Previous best score and delta.

Do not claim success from a queue/compiling status. If the platform stalls, report it as pending without making a duplicate submission.

Store raw screenshots/text under a unique ignored path such as `artifacts/raw/xpuoj/<submission-id>/`. Review for account or personal data before deriving any small committable evidence. Close the dedicated WebBridge session when the browser task ends.

## Ledger And User Report

Record the terminal result:

```bash
python skills/xpuoj-operator-optimizer/scripts/submission_ledger.py record \
  --id <submission-id> \
  --operator <operator> \
  --language <language> \
  --commit <full-commit> \
  --status <terminal-status> \
  --score <score-or-na> \
  --rank <rank-or-na> \
  --url <public-or-safe-result-url> \
  --evidence <relative-evidence-path> \
  --experiment <experiment-id>
```

Immediately send the user a report using `templates/submission-report.md`. Include failures as candidly as successes.

After the report message has actually been sent:

```bash
python skills/xpuoj-operator-optimizer/scripts/submission_ledger.py report \
  --id <submission-id> \
  --summary <short-summary>
```

Then run `check` again. The next submission is prohibited until it passes.

If the interaction environment cannot send an intermediate message, stop after one submission and report in the final response.
