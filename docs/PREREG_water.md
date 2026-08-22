# Pre-registration: water below its triple point

**Registered 2026-08-21, before the correction was run against any
validation set.** Predictions are not edited afterwards.

---

## The defect

`CoolPropThermo._clamp` raises every saturation-line query to the triple
point before it reaches CoolProp:

```python
lo = self._T_triple + 1e-6
return min(max(T, lo), hi)
```

That is correct as a solver guard, and the docstring says so: the
equilibrium solver brackets its root over a wide interval and will
evaluate unphysical temperatures on the way, where the values need only
be finite and monotone.

It is wrong as physics for **water alone**, because water is the only
substance here whose triple point (273.16 K) lies above the temperatures
the clouds reach. Released materials sit below theirs — methane at
90.7 K, hydrogen at 13.96 K — so the clamp never engages for them.

Measured saturation pressure through the water backend:

| T [K] | CoolProp | IAPWS sublimation | ratio |
|---|---|---|---|
| 273.16 | 6.34e2 | 6.12e2 | 1.04 |
| 260 | 6.34e2 | 1.96e2 | 3.2 |
| 240 | 6.34e2 | 2.73e1 | 23 |
| 200 | 6.34e2 | 1.63e-1 | **3,901** |
| 150 | 6.34e2 | 6.10e-6 | **1.0e8** |

The Burro clouds reach 197-221 K, so a 200 K cloud is being told it can
hold as much vapour as air at 0 °C. Water does not condense when it
should.

## The correction

Keep `_clamp` — it does its job. Below the triple point, take the
saturation pressure from the IAPWS sublimation curve (Wagner,
Riethmann, Feistel and Harvey, 2011) and add the enthalpy of fusion to
the latent heat, since the transition is now solid-vapour.

**No free parameters.** The curve is fixed by the reference correlation.

## What is being tested

Whether the correction leaves the dense-gas validation untouched, and
what it does to the three variants that use the real-fluid backend.

## Predictions

| # | Prediction | Criterion |
|---|---|---|
| **P-W1** | The dense-gas validation cannot move, because its path never calls the CoolProp water backend | LFL statistics **bit-identical** |
| **P-W2** | The real-fluid variants move, but only slightly, because the water mass fraction is small | MG change **< 5 %** for `coolprop`, `composition`, `canonical+coolprop` |
| **P-W3** | Real properties alone do not improve on the baseline | `coolprop` FAC2 **at or below** the `slab90` baseline of 0.67 |
| **NC** | A trial that stays above the triple point is unaffected | Prairie Grass (minimum 297.7 K) **bit-identical** |

## What would falsify

**P-W1 is the one that matters.** The LFL path calls `water_backend()`,
which returns `LegacyThermo`, so the correction cannot reach it. If the
LFL statistics move at all, the reading of the code is wrong and
everything below it needs re-examining.

If P-W2 fails — if a variant moves by more than 5 % — then water is
carrying more of the density than the small mass fraction suggests, and
the earlier statement that the correction is a tidy-up rather than a
physics change does not hold.

## What must not happen

The correction is not offered as an improvement to field performance.
It is a defect fix, and the expected outcome is that the published
results do not move. Should a statistic happen to improve, that is not
evidence for the correction; the evidence is the reference correlation.

---

## RESULTS

*(appended after the run; predictions above unedited)*

*(appended 2026-08-21 after the run; nothing above this line was edited)*

### The curve reproduces the reference

| T [K] | legacy | corrected | IAPWS | ratio |
|---|---|---|---|---|
| 260 | 2.211e2 | **1.958e2** | 1.958e2 | **1.000** |
| 240 | 3.775e1 | **2.727e1** | 2.727e1 | **1.000** |
| 200 | 3.814e-1 | **1.626e-1** | 1.626e-1 | **1.000** |
| 150 | 3.892e-5 | **6.096e-6** | 6.096e-6 | **1.000** |

Above the triple point nothing changes.

### P-W1 — PASSED, bit-identical

```
LFL 10 trials:  n=10  FAC2=0.90  MG=1.252  VG=1.225  FB=+0.117  NMSE=0.121
before      :  n=10  FAC2=0.90  MG=1.252  VG=1.225  FB=+0.117  NMSE=0.121
```

Not merely close — **identical to every reported digit**, as it must be:
the path calls `water_backend()`, which returns `LegacyThermo`, and the
correction is inside `CoolPropThermo`.

### P-W2 — PASSED

| variant | FAC2 | MG | change |
|---|---|---|---|
| `slab90` (baseline) | 0.67 → 0.67 | 0.605 → 0.605 | **0.0 %** |
| `coolprop` | **0.67 → 0.58** | 0.591 → 0.580 | −1.8 % |
| `composition` | 0.75 → 0.75 | 0.685 → 0.674 | −1.6 % |
| `canonical+coolprop` | 0.50 → 0.50 | 0.506 → 0.492 | −2.7 % |

All within the registered 5 %. The water mass fraction is small enough
that a factor of 3900 in its saturation pressure moves the aggregate by
under three per cent — which is why the defect survived this long.

### P-W3 — PASSED

`coolprop` falls to **FAC2 0.58**, below the 0.67 baseline. Real
properties alone do not improve on the legacy correlations; the
improvement reported for `composition` (0.75) comes from the
multicomponent treatment, not from the equation of state.

**The correction sharpens that separation rather than softening it.**
Before, real properties alone matched the baseline and it was possible to
read the improvement as coming from either change.

### NC — PASSED

Prairie Grass stays above the triple point throughout (minimum 297.7 K)
and is unchanged. 111 tests pass across the thermodynamics,
multicomponent, Prairie Grass and LNG pool modules.

### Status

| | |
|---|---|
| P-W1 | **PASS** — bit-identical |
| P-W2 | **PASS** — largest change 2.7 % |
| P-W3 | **PASS** — `coolprop` FAC2 0.58 < 0.67 |
| NC | **PASS** |

### What this changes in the paper

Only the real-fluid numbers in the optional-modules section. The
statement that multicomponent composition is the one addition improving
all four indices is **strengthened**: real properties alone now fall
below the baseline instead of matching it.

Nothing in the verification, the field comparison, the stability
ablation, the Froude scaling or the added-mass test moves.

### An unregistered consequence — worth recording

`tests/test_rainout.py` had a test asserting that at the point where the
liquid runs out, velocity and concentration matched the SMEDIS DT1
equivalent-source targets while the distance did not, and reading that as
evidence the thermodynamics were right and only the droplet lifetime
wrong.

The correction moved it. Against the four DT1 targets:

| | x [m] | u [m/s] | C [%] | T [K] |
|---|---|---|---|---|
| SMEDIS DT1 | 51.0 | 7.50 | 13.0 | 205 |
| before | 77.1 (1.51) | 7.30 (0.97) | 13.46 (**1.04**) | 228 (1.11) |
| **after** | 65.2 (**1.28**) | 7.53 (**1.00**) | 16.32 (1.26) | 218 (**1.06**) |

**Three of four improved.** The droplet lifetime, which is what the
correction physically acts on, went from 51 % long to 28 % long.
Velocity became near-exact. Temperature improved.

Concentration moved away, and the reason is mechanical: with water
condensing as it should, the latent heat evaporates the ammonia droplets
sooner, so the liquid runs out at 65 m instead of 77 m and the
concentration is read closer to the source, where it is higher.

**The earlier agreement on concentration was therefore not independent
evidence.** It held while the droplet lifetime was half again too long,
and it went away once that error was reduced. One quantity was
compensating another.

This was not registered — it was found by a test failing — and it is
recorded here rather than in the results because it concerns a
diagnostic, not a validation statistic. The test now pins the corrected
state and says what changed.
