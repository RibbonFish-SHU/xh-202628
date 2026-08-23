# Experiment <EXP-ID>

- Timestamp:
- Operator/problem:
- Language:
- Device class: `CPU | c500-local | XPU-OJ/C500`
- Environment fingerprint:
- MACA/MXCC and slice quota:
- Baseline commit:
- Tested candidate commit:
- Workflow commit:
- Workflow / baseline reservation refs:
- Final queue commit (if metadata-only handoff follows test):
- Workflow/candidate/baseline archive SHA-256:
- Baseline source SHA-256:
- Candidate source SHA-256:

## Hypothesis

一句话写出可证伪的性能假设。

## Change

列出唯一或紧密相关的一组改动，以及预期影响的瓶颈。

## Commands

```text
build command
test command
benchmark command
```

## Correctness

- Test set/seed:
- Result:
- Numeric tolerance source:
- Raw log:

## Performance

- Baseline metric:
- Candidate metric:
- Repetitions/statistic:
- ABBA order and baseline/candidate drift:
- Delta:
- Raw artifact:

## Decision

`keep | revert | investigate`

说明证据、限制和下一条假设。本地结果写 `c500-local`；OJ slice 未确认前，不得用本地绝对时间推断分数。
