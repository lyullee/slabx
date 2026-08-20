# Validation

Every comparison, and the command that reproduces it.

Two questions are kept apart:

* **Does it reproduce SLAB?** — against Ermak's Fortran (§1)
* **Is SLAB right?** — against measurements (§2)

The first has to be settled before the second means anything: a 6 % change
from real-fluid properties is only interpretable if the reproduction error is
0.5 %.

---

## 1. Against the original Fortran

The oracle is `SLAB.FOR` compiled with `gfortran`. It is not redistributable
here; `golden/fortran/README.md` says how to obtain and build it. Without it
these tests skip and everything else runs.

```bash
python3 examples/fortran_reference.py            # the five manual decks
python3 examples/fuzz_fortran.py 60 7            # random differential test
python3 -m pytest tests/test_fortran_oracle.py   # 9 tests
```

### The manual's five examples

Median relative difference over the established plume:

| example | IDSPL | h | bb | b | **cv** | rho | T | u |
|---|---|---|---|---|---|---|---|---|
| §4.1 Burro 8 | 1 | 0.30 % | 0.31 % | 0.35 % | **0.55 %** | 0.21 % | 0.07 % | 0.11 % |
| §4.1b neutral | 1 | 0.15 % | 0.13 % | 0.05 % | **0.44 %** | 0.27 % | 0.06 % | 0.05 % |
| §4.2 DT4 jet | 2 | 0.28 % | 0.14 % | 1.48 % | **0.50 %** | 0.14 % | 0.06 % | 0.07 % |
| §4.3 LNG instant | 4 | 0.11 % | 0.21 % | 0.07 % | **0.58 %** | 0.24 % | 0.08 % | 0.11 % |
| §4.4 Cl2 stack | 3 | 0.49 % | 1.13 % | 0.87 % | **5.64 %** | 0.18 % | 0.00 % | 0.12 % |

The tolerance is the **original's own grid sensitivity** (0.03 – 0.52 %),
estimated by running it on a coarser grid. Matching more closely than the
original resolves its own equations would be meaningless.

Source expansion agrees too: 12.82 m geometric becomes 31.55 m here against
the original's 31.50 m.

The chlorine near field is the one open item — the ratio is 0.77 at 3 m and
1.01 by 125 m. It is recorded, not tuned.

### Random differential test

108 decks over three seeds. Four source types, five real materials, stability
sampled over the full range the original accepts.

| variable | median | p90 |
|---|---|---|
| centre height | 0.02 – 0.15 % | 0.16 – 1.51 % |
| temperature | 0.09 – 0.12 % | 0.14 – 1.41 % |
| cloud speed | 0.16 – 0.20 % | 1.15 – 2.80 % |
| half width | 0.17 – 0.25 % | 2.75 – 9.61 % |
| density | 0.20 – 0.22 % | 0.38 – 2.12 % |
| cloud height | 0.26 – 0.29 % | 1.66 – 7.35 % |
| **volume fraction** | **0.61 – 0.91 %** | 8.41 – 18.31 % |

The oracle failed on **none** of them, including 51 decks at `|1/L| > 0.15`.

### What the fuzzing found

Two defects in `slabx`, both fixed, neither reachable from the manual's decks:

* the plume-to-puff transition row did not carry the plume-rise duration, so
  time stepped backwards — **four of ten** sampled vertical jets;
* the output grid could place a point upwind of the release when the rise
  exceeded the domain.

One difference remains, with its cause located but not fixed: see
`LIMITATIONS.md` §2.

### A correction worth reading

Early work compared against a JavaScript transcription and recorded six
differences as defects of SLAB. Against the Fortran, **only one was.** Three
came from a mechanical `min` → `Math.min` substitution that also corrupted
words inside comments, one was a single-character typo (`all` for `aal`,
appearing once against ten), and one was a deliberate sentinel.

The typo made the JavaScript throw for `|1/L| > 0.15`, which was recorded as a
limit of SLAB and used to bound the whole reproduction claim. It was
withdrawn. `docs/02_VALIDATION_REFERENCE.md` §2.5 has the evidence.

---

## 2. Against measurements

38 trials, four source types.

```bash
python3 examples/validate_burro.py       # pre-registered verdicts
python3 examples/diagnose_burro.py       # how hypotheses were excluded
python3 examples/entrainment_limits.py   # canonical-flow limits
python3 -m pytest tests/test_lng_pools.py tests/test_thorney_island.py \
                 tests/test_prairie_grass.py tests/test_desert_tortoise.py \
                 tests/test_fladis.py tests/test_wind_tunnel.py
```

### LNG pools — distance to LFL (10 trials)

Burro, Coyote and Maplin Sands. Observed distances and the Phast comparison
are from PHMSA's *Final Environmental Assessment* Table 9.

| | FAC2 | MG | VG | FB | NMSE |
|---|---|---|---|---|---|
| **slabx** | 0.90 | 1.252 | 1.225 | +0.117 | **0.121** |
| Phast 8.4 | 0.90 | 0.996 | 1.206 | −0.038 | 0.204 |

All five Chang–Hanna criteria pass. Scatter is better here, bias is better in
Phast.

Burro 8 is worth singling out — low wind, stable, the condition PHMSA calls
*"most directly relevant to current federal regulations"*. Phast gets 0.42 of
the observed distance; `slabx` gets **0.98**. That is the original SLAB doing
well, not an improvement of ours.

### Prairie Grass — the passive limit (2 trials)

Every other trial is a dense gas, so this limit had never been exposed.

| | FAC2 | MG | VG | FB | NMSE |
|---|---|---|---|---|---|
| concentration | **1.00** | 1.218 | 1.143 | +0.173 | 0.178 |
| plume width | **1.00** | 0.874 | 1.087 | −0.123 | 0.100 |

All ten points within a factor of two, on both. These trials also report
measured `u*`, which becomes a prediction: 0.371 against 0.31, and 0.218
against 0.21.

### Two-phase jets — Desert Tortoise and FLADIS (6 trials)

Thermodynamics first, against HSE's independent recalculation of the source
conditions (Jack Rabbit III Modelers Working Group, Tables 3–7). Different
group, different method, different property source:

| trial | jet temperature | mole fraction | density |
|---|---|---|---|
| DT1 | 1.011 | 1.012 | 0.988 |
| DT2 | 1.010 | 1.012 | 0.988 |
| DT4 | 1.011 | 1.013 | 0.988 |

All within 1.3 %.

Arc-maximum concentrations against the JRIII observations, with two other
models for scale:

| | FAC2 | MG | VG |
|---|---|---|---|
| **Desert Tortoise** | | | |
| slabx | 0.67 | 0.733 | 1.483 |
| DRIFT 3.7 | 0.67 | 0.559 | 1.573 |
| Phast 8.61 | 1.00 | 1.025 | 1.222 |
| **FLADIS** (1000× smaller release) | | | |
| slabx | **0.67** | **0.946** | 1.914 |
| DRIFT 3.7 | 0.56 | 1.152 | 1.508 |
| Phast 8.61 | 0.56 | 0.722 | 1.518 |

All three models under-predict the near arcs and over-predict the far ones, at
both scales. That points at source treatment rather than at scale.

FLADIS also gives 161 sensors with concentration standard deviations, so the
fluctuation intensity is measured directly: 0.72 – 0.86, falling from 1.10
near the source to 0.72 beyond 200 m.

### Instantaneous releases — Thorney Island and Lathen (17 trials)

Fifteen Thorney Island Phase I trials reproduce the reported insensitivity to
wind speed (r = +0.18, p = 0.53) but **not** to stability — see
`LIMITATIONS.md` §3.

Trial 008, arrival time by distance band:

| | FAC2 | MG | VG | NMSE |
|---|---|---|---|---|
| arrival time | **1.00** | 0.777 | 1.108 | **0.026** |
| dose | 0.40 | 1.844 | 1.977 | 0.765 |

Arrival time is the sharpest test available for an instantaneous release: it
depends only on the front speed, with no thermodynamics or profile shape. The
ratio converges from 1.81 at 65 m to **1.03** at 480 m.

Lathen 49 is 1/500 of the mass. Arrival time again has FAC2 1.00; dose is 10×
low, diagnosed as the `2Bx/u` passage-time estimate rather than the model — a
puff this size keeps spreading while it passes.

### Wind tunnel (3 trials)

The model runs at 7 mm reference height and 1.7e-4 kg/s. It is not
Froude-similar; see `LIMITATIONS.md` §1.

### Trials excluded (19)

Every one with a recorded reason, in
`src/slabx/validation/data/smedis_inventory.csv`:

| reason | count |
|---|---|
| obstacles — walls, trenches, canyons, box arrays, fences | 11 |
| surface slope 4 – 11.6 % | 3 |
| coordinate frame not recoverable | 4 |
| roughness 5 m | 1 |

The obstacle exclusions are supported by measurement, not assumption — see
`LIMITATIONS.md` §1.

### An independent evaluation, and what it shows about evaluations

EPA's own comparison of seven dense-gas models (Zapert et al. 1991,
EPA-450/4-90-018) reported SLAB under-predicting Desert Tortoise by a factor
of **2.6 at 100 m and 2.74 at 800 m**, with none of the three trials inside a
factor of two — while on Burro at 140 m the same model had the **smallest bias
and RMS error of the five** compared.

The report also records, in its methods section, that SLAB's developer had
identified the cause:

> the cross sectional area of the fully expanded jet should be used as a
> source area, as opposed to the orifice area which was used [...] This
> modification would greatly reduce the source velocity and could
> significantly alter model results. This change was not implemented, since
> it was identified only after performance results had been obtained.

`slabx` uses the post-flash two-phase conditions — the correction that was
asked for — and gets MG 0.733 on the same trials, an under-prediction of 1.36
rather than 2.6.

Two of the report's five conclusions are worth reading alongside its numbers:
that models sharing the same dispersion treatment and meteorology differ by
**more than an order of magnitude** through source characterisation alone, and
that it was **"not feasible to attribute model performance to any specific
algorithms or design features."** The second is the reason this project
decomposes rather than aggregates.

### An independent check on the whole picture

SMEDIS reports group statistics by model class. For shallow-layer models on
arcwise comparisons without complex effects: FAC2 0.65, MG 0.613. Running
`slabx` in its reference-faithful configuration at the standard roughness
gives FAC2 **0.67**, MG **0.605** — a different implementation, different
trials and different modellers, reproducing not just the code but its
documented performance.

---

## 3. Pre-registration

Thirteen hypotheses were registered with explicit predictions before the
comparison was run, and judged on whether the mechanism acted *where it was
predicted to*, not on whether an aggregate statistic improved.

**Four of seven coefficient changes were rejected.** Three times an aggregate
statistic said "better" while the decomposition said the mechanism had
failed — see `docs/03_VALIDATION_FIELD.md` §3.5. Three of our own hypotheses
were falsified outright, including the structural one that motivated the
height closure.

The registered wording was not edited afterwards. A failed prediction that has
been rewritten is worth nothing.

```bash
python3 examples/validate_burro.py       # prints each verdict
python3 -m pytest tests/test_height_closure.py    # the falsified one
```

---

## 4. Numerical

```bash
python3 -m pytest tests/test_convergence.py
```

Puff grid: six figures in 40 steps. Field grid: 0.03 % in 80. Source grid:
**first order**, and 2.8 % low at the default. See `LIMITATIONS.md` §7.

---

## 5. Input validation

```bash
python3 -m pytest tests/test_scope.py
```

Four inputs used to run to completion and return plausible numbers — a
negative wind speed produced `u* = −0.178`, a reference height below the
roughness produced `u* = 33.4 m/s`. All now raise with a reason.

Three further ranges warn rather than refuse: see `LIMITATIONS.md` §5.
