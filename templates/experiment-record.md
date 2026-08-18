# Experiment <EXP-ID>

- Timestamp:
- Operator/problem:
- Language:
- Device class: `CPU | NVIDIA proxy | C500 local | XPU-OJ/C500`
- Environment fingerprint:
- Baseline commit:
- Candidate commit:

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
- Delta:
- Raw artifact:

## Decision

`keep | revert | investigate`

说明证据、限制和下一条假设。NVIDIA 数据必须明确写 `proxy`，不得推断 C500 排名。
