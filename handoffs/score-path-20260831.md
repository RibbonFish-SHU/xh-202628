# Fused MoE 85.50 Score Path

## Evidence

- Formal submission: `#130490`
- Formal commit: `8341aa38e55659285673111df90e786c1fba08df`
- Formal displayed cases: `86/77/90/86` (`339` raw points, `84.75` total)
- Formal user times: `0.804/6.425/0.493/3.473 ms`
- SPJ baselines: `5.097/22.728/4.458/21.651 ms`
- Target: `342` raw points, or `85.50` total

The verified SPJ formula is:

```text
ratio = baseline_ms / (baseline_ms + user_ms)
displayed_score = floor(100 * ratio)
```

For a target displayed score `s`, the exact upper bound on user time is:

```text
user_ms <= baseline_ms * (100 - s) / s
```

## Per-Case Thresholds

| Case | Tier | Maximum user time (ms) | Reduction from formal |
|---|---:|---:|---:|
| C1 | 87 | 0.761621 | 5.2711% |
| C1 | 88 | 0.695045 | 13.5516% |
| C1 | 89 | 0.629966 | 21.6460% |
| C2 | 78 | 6.410462 | 0.2263% |
| C2 | 79 | 6.041620 | 5.9670% |
| C2 | 80 | 5.682000 | 11.5642% |
| C3 | 91 | 0.440901 | 10.5677% |
| C3 | 92 | 0.387652 | 21.3687% |
| C3 | 93 | 0.335548 | 31.9374% |
| C4 | 87 | 3.235207 | 6.8469% |
| C4 | 88 | 2.952409 | 14.9897% |
| C4 | 89 | 2.675966 | 22.9494% |

## Complete Three-Point Combinations

Each tuple is `(C1, C2, C3, C4)` and contains the raw points gained in each
case. `Max reduction` is the largest individual case reduction required by the
combination. `Sum reduction` is shown only as a secondary comparison; reductions
from different cases are not interchangeable.

| Gain tuple | Max reduction | Sum reduction |
|---|---:|---:|
| `(0,0,0,3)` | 22.9494% | 22.9494% |
| `(0,0,1,2)` | 14.9897% | 25.5574% |
| `(0,0,2,1)` | 21.3687% | 28.2156% |
| `(0,0,3,0)` | 31.9374% | 31.9374% |
| `(0,1,0,2)` | 14.9897% | 15.2159% |
| `(0,1,1,1)` | 10.5677% | 17.6409% |
| `(0,1,2,0)` | 21.3687% | 21.5950% |
| `(0,2,0,1)` | 6.8469% | 12.8139% |
| `(0,2,1,0)` | 10.5677% | 16.5347% |
| `(0,3,0,0)` | 11.5642% | 11.5642% |
| `(1,0,0,2)` | 14.9897% | 20.2607% |
| `(1,0,1,1)` | 10.5677% | 22.6857% |
| `(1,0,2,0)` | 21.3687% | 26.6398% |
| `(1,1,0,1)` | 6.8469% | 12.3442% |
| `(1,1,1,0)` | 10.5677% | 16.0651% |
| `(1,2,0,0)` | 5.9670% | 11.2381% |
| `(2,0,0,1)` | 13.5516% | 20.3985% |
| `(2,0,1,0)` | 13.5516% | 24.1193% |
| `(2,1,0,0)` | 13.5516% | 13.7778% |
| `(3,0,0,0)` | 21.6460% | 21.6460% |

## Allocation Gate

The lowest maximum-reduction paths are:

1. `(1,2,0,0)`: C1 `5.2711%` and C2 `5.9670%`.
2. `(1,1,0,1)`: C1 `5.2711%`, C2 `0.2263%`, and C4 `6.8469%`.
3. `(0,2,0,1)`: C2 `5.9670%` and C4 `6.8469%`.

Allocate a new executable experiment only when its mechanism and composition
account for every case threshold in at least one complete tuple. A case-2-only
candidate must reach `11.5642%`, not `2.724%`, to provide all three points.
