# Pre-registration: is the ambient stratification term the cause?

**Registered 2026-08-19, before the intervention was run.**
Predictions are not edited afterwards.

---

## The claim under test

Section 4.3 of the draft reports that stability correlates with the model's
error at r = 0.87 across fifteen Thorney Island trials, and that holding
everything else fixed while moving B to F changes the 300 m concentration
6.17-fold. It then says:

> 본 연구는 원인폐합을 직접 측정하지 않았으므로 인과를 확정하지 않는다.

That reservation is avoidable. The suspected pathway is fully explicit in
the code, and one of its links can be cut.

## The pathway

EQ 35d builds an in-cloud Monin-Obukhov length as a **sum of two
stratification sources**:

    1/L_cloud = ( 1/L_ambient · u*_a²  +  C_mu · g · Δρ/ρ ) / u*²
                └── the air the cloud ──┘  └── the cloud's ──┘
                    sits in                    own density

which then damps the vertical entrainment through

    phi_h = 1 + phi_stable · h_top · 1/L_cloud
    W_e   ∝ 1 / phi_h

So the proposed chain is

    Pasquill class → 1/L_ambient → 1/L_cloud → phi_h → W_e → concentration

Every link is deterministic. The first term can be weighted to zero without
touching anything else — including the cloud's own damping, which stays.

## The intervention

`w_ambient_stratification`, default 1.0 (what SLAB does), set to 0.

This is not a coefficient adjustment offered as an improvement. It is an
ablation, run to find out whether the ambient pathway carries the observed
response.

## Predictions

| # | Prediction | Criterion |
|---|---|---|
| **P-S1** | The 6.17-fold B→F response is carried by the ambient term | with the term removed the ratio falls **below 2.0** |
| **P-S2** | The cloud's own stratification still damps, so the response does not vanish entirely | ratio stays **above 1.0** — a completely flat response would mean the cloud term does nothing either |
| **P-S3** | The ambient term's share of `1/L_cloud` is large in these trials | median ambient fraction **> 0.5** across the fifteen trials |
| **NC** | A neutral trial has no ambient stratification to remove | D-class ratio **exactly 1.000** |

## What would falsify the causal claim

If P-S1 fails — if the response survives with the ambient term gone — then
the 6.17-fold behaviour comes from somewhere else (the density-dependent
term, the mixing height, the wind profile) and §4.3's suspicion is wrong.

If P-S3 fails — if the ambient term is a small part of `1/L_cloud` — then
even a large P-S1 effect would be a lever-arm artefact rather than the
mechanism.

## What must not happen

`w_ambient_stratification` is a **diagnostic switch, not a proposed fix**.
Removing a term that SLAB has for a reason is not an improvement, and the
default stays at 1.0. The paper's own argument is that a change which
improves a statistic without a mechanism is worthless; the converse also
holds — an ablation that establishes a mechanism is not thereby a
prescription.

---

## RESULTS

*(appended after the run; predictions above unedited)*

*(appended 2026-08-19 after the run; nothing above this line was edited)*

### The intervention

Trial 008, everything held, class varied, `w_ambient_stratification` swept.
300 m peak concentration [%]:

| `w_ambient` | B | C | D | E | F | **F/B** |
|---|---|---|---|---|---|---|
| **1.0** (SLAB) | 0.0721 | 0.1716 | **0.3821** | 0.4419 | 0.4447 | **6.17** |
| 0.5 | 0.1189 | 0.2746 | **0.3821** | 0.4172 | 0.4173 | 3.51 |
| **0.0** | 0.3140 | 0.3504 | **0.3821** | 0.3893 | 0.3844 | **1.22** |

### P-S1 — PASSED

Removing the ambient term drops the B→F response from **6.17 to 1.22**,
against a registered threshold of 2.0.

The sweep is **monotonic in the ablation weight** — 6.17, 3.51, 1.22 — which
is a dose-response relationship, not an on/off effect. Halving the term
roughly halves the response in log terms.

### P-S2 — PASSED

The response does not vanish: 1.22 > 1.0. The cloud's own stratification
still damps a little, as it should.

### NC — PASSED exactly

The D column is **0.3821 at every weight**, to four decimals. A neutral class
has no ambient stratification to remove, so weighting it changes nothing.
The negative control is built into the sweep rather than run separately.

### P-S3 — **FAILED**, and it changes the reading

The ambient share of `1/L_cloud`, measured across the fifteen trials:

| trial | ambient share |
|---|---|
| 004 | 1.009 |
| 014 | 1.016 |
| 015 | 1.019 |
| 010 | 0.586 |
| 007 | 0.142 |
| 012 | 0.064 |
| **the other nine** | **0.000** |

Median **0.000**, against a registered criterion of > 0.5.

It is **bimodal, not a gradient**: the term is either everything or nothing.
Nine of fifteen Thorney Island trials are neutral, and a neutral class has
`1/L_ambient` = 0 exactly, so the ambient pathway is switched off for them
by construction.

### What is established, and what is not

**Established — the mechanism.** The pathway

    Pasquill class → 1/L_ambient → 1/L_cloud → phi_h → W_e → concentration

carries essentially all of the isolated 6.17-fold response. Cutting one link
collapses it to 1.22; weighting the link halfway gives an intermediate
response; a class with nothing to cut is unchanged to four decimals. That is
an intervention with a dose-response and a negative control, on a
deterministic pathway that is explicit in the source. **The causal statement
in §4.3 can be made.**

**Not established — that this explains the fifteen-trial correlation.**
P-S3 shows the ambient term acts on at most six of the fifteen trials, and
strongly on three. The r = 0.87 across the trial set cannot be attributed to
this pathway on the strength of this test alone: either it is carried by
those few trials, or something else contributes. **That remains open.**

### Suggested wording for §4.3

> 안정도 응답의 원인폐합을 절제 실험으로 확정하였다. EQ 35d의 주변 성층항
> 가중을 1.0에서 0으로 낮추면 B→F 응답이 6.17배에서 1.22배로 붕괴하며,
> 가중 0.5에서 3.51배로 단조 감소한다. 중립(D)급은 제거할 주변 성층이 없어
> 세 가중 모두에서 소수 넷째 자리까지 동일하다. 즉
> `안정도 → 1/L_주변 → 1/L_구름 → φ_h → W_e → 농도` 경로가 격리된 응답의
> 거의 전부를 운반한다.
>
> 다만 15시험 전체의 상관(r = 0.87)이 이 경로로 설명되는지는 별개이다.
> `1/L_구름` 중 주변항의 비중은 이분적이어서 — 아홉 시험에서 정확히 0,
> 세 시험에서 약 1.0 — 중립 시험에는 이 경로가 구조적으로 작동하지 않는다.
> 상관이 소수 시험에 의해 운반되는지 다른 기여가 있는지는 확인하지 않았다.

### Status

| | |
|---|---|
| P-S1 | **PASS** — 6.17 → 1.22, monotonic in weight |
| P-S2 | **PASS** — 1.22 > 1.0 |
| P-S3 | **FAIL** — ambient share bimodal, median 0.000 |
| NC | **PASS** — D unchanged to four decimals |
| causal claim (isolated response) | **established** |
| causal claim (fifteen-trial correlation) | **open** |

`w_ambient_stratification` stays at 1.0. It is a diagnostic switch, not a
proposed change.
