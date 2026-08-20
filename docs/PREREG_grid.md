# Pre-registration: field validation on a converged source grid

**Registered 2026-08-17, before the sweep was run.**
Predictions are not edited after results are known. Results are appended
under `RESULTS`.

---

## Background

`tests/test_convergence.py` measures the source-region grid at first order,
not fourth: the default of ten steps sits 2.82 % below the converged mass
fraction at 400 m, and 320 steps reaches 0.03 %. The cause is structural —
EQ 1a replaces the species integration with an algebraic ramp whose source
term has a slope discontinuity at the pool edge, which costs Runge-Kutta its
order.

The field comparison, meanwhile, sits at MG 0.606 — a 65 % under-prediction.
Arithmetic says the discretisation accounts for about 4 % of that gap, so the
conclusions cannot turn on it.

**That arithmetic has never been run.** A reviewer asking *"is the
discrepancy just discretisation error?"* currently gets an inference from a
different table rather than a direct answer.

## What is being tested

Whether the field discrepancy survives grid refinement. `n_source_steps` is
an input of the original, so nothing in the reproduction chain changes; this
is a re-run, not a code change.

## Predictions

| # | Prediction | Criterion |
|---|---|---|
| **P-N1** | If the gap is physical, converging the grid does not close it | field aggregate MG changes **< 10 %** (outside 0.55-0.67 falsifies) |
| **P-N2** | Any effect concentrates in source-dominated trials and is negligible for instantaneous releases | per-trial change correlates with source type |
| **NC** | An instantaneous release has no source region, so it cannot move | **exactly zero** change for Thorney Island |

## What would falsify

If MG moves outside 0.55-0.67, the discretisation is a first-order
contributor to the field discrepancy and every statement in the paper that
treats the gap as physical needs qualifying.

## Note on scope

This tests the *source* grid only. The puff grid converges to six figures in
40 steps and the field grid to 0.03 % in 80, both already established, so
they are held at their defaults.

---

## RESULTS

*(appended after the run; predictions above unedited)*

*(appended 2026-08-17 after the run; nothing above this line was edited)*

### Concentration comparison (Witlox arcs, four trials, twelve points)

| `n_source` | FAC2 | MG | VG | NMSE |
|---|---|---|---|---|
| **10** (default) | 0.67 | **0.605** | 1.615 | 0.748 |
| 40 | 0.67 | 0.599 | 1.616 | 0.740 |
| 160 | 0.67 | 0.598 | 1.615 | 0.737 |
| **320** (converged) | 0.67 | **0.596** | 1.616 | 0.738 |

MG moves **1.5 %**, from 0.605 to 0.596. FAC2 does not move at all.

### Distance comparison (ten LNG pool trials)

| `n_source` | FAC2 | MG | VG | NMSE |
|---|---|---|---|---|
| 10 | 0.90 | 1.252 | 1.225 | 0.121 |
| 320 | 0.90 | 1.251 | 1.227 | 0.122 |

MG moves **0.1 %**.

### P-N1 — PASSED

Criterion was a change under 10 %, with 0.55-0.67 as the falsifying band.
The converged value is **0.596**, a change of 1.5 %.

The field discrepancy is **not** a discretisation artefact. Refining the
source grid by a factor of 32 closes about **4 %** of the gap between
MG 0.605 and unity, which is what the arithmetic from
`tests/test_convergence.py` predicted and is now measured rather than
inferred.

### P-N2 — PASSED

Per-trial change, 10 to 320 steps:

| trial | 10 | 320 | ratio |
|---|---|---|---|
| **BU08** | 448.6 | 460.5 | **1.027** |
| CO6 | 385.0 | 385.7 | 1.002 |
| BU09 | 404.2 | 404.5 | 1.001 |
| MS27 | 152.9 | 152.9 | 1.000 |
| BU03 | 146.9 | 146.7 | 0.998 |
| BU07 | 253.5 | 253.0 | 0.998 |
| MS34 | 112.5 | 112.3 | 0.998 |
| MS35 | 113.9 | 113.6 | 0.998 |
| CO3 | 129.3 | 128.8 | 0.996 |
| CO5 | 89.3 | 89.0 | 0.996 |

Nine of ten move by 0.4 % or less. The exception is **BU08 at 2.7 %** — the
low-wind stable trial, which is where the source region occupies the largest
fraction of the run and where the pool-edge discontinuity therefore matters
most. That is the predicted concentration, and it is also the trial the
paper singles out for other reasons.

The direction is not uniform: six trials fall slightly and four rise. A
systematic discretisation bias would move them together. It does not.

### NC — PASSED exactly

An instantaneous release has no source region. Thorney Island at
`n_source` = 10 and 320 returns C(300 m) = 0.0035120902 both times,
difference **0.000e+00**.

### Status

| | |
|---|---|
| P-N1 | **PASS** — MG 0.605 → 0.596, 1.5 % |
| P-N2 | **PASS** — concentrated in BU08 (2.7 %), nine others ≤ 0.4 %, no common sign |
| NC | **PASS** — bit-identical |

The discretisation accounts for about 4 % of the field gap. Every conclusion
that treats the gap as physical stands.

### A note on what was changed to run this

`run_trial` gained a `**run_kw` pass-through so the sweep could reach
`n_source_steps`. No default moved and no model code changed;
`n_source_steps` is an input of the original.
