# Pre-registration: added mass against Hall & Walker (2000)

**Registered 2026-08-19, before the comparison was run.**
Predictions are not edited afterwards. Results appended under `RESULTS`.

---

## Background

`slabx` adds an oblate-spheroid added-mass term to SLAB's vertical momentum
balance:

    k_added = C_A (rho_a / rho) (B / h),      C_A = 2 / pi

`C_A = 2/pi` is the potential-flow value for a flat disc moving broadside;
nothing is fitted. The term was checked against four internal predictions
(A1-A4) and passed all four, but **never against measurement**. The project
documentation recorded this as "no data available", which was wrong: the
data is in the project files.

Hall & Walker (2000), reported in AEAT/NOIL/27328006/001 (URAHFREP WP7),
varied a ground-level area source **64-fold in width at fixed buoyancy
flux** — exactly the discriminating experiment for a term whose whole
content is a width dependence.

DRIFT's authors, using the same data, report that adding this term
*"suppressed lift-off too much"* and made agreement worse. That is the
hypothesis under test.

## Data

Source geometry (Table 1, L = 6.7 cm, z0 = 0.03 L), the "wide" series with
length x/L = 3.43 fixed:

| group | width y/L |
|---|---|
| G | 0.448 |
| H | 1.19 |
| D | 3.43 |
| I | 7.16 |
| J | 14.33 |
| K | 28.66 |

Lift-off criteria, quoted at 15 L and 30 L downwind:

| | F / (W u^3) |
|---|---|
| onset of rise (concentration maximum leaves the ground) | ~0.01 |
| lift-off (ground concentration falls to 10-20 % of maximum) | ~0.035 |

Hall & Walker note "considerable scatter" and that long sources aligned
with the wind behave differently; only the wide series is used.

## Predictions

| # | Prediction | Criterion |
|---|---|---|
| **P-AM5** | Without added mass, rise is under-suppressed for wide sources: lift-off occurs at a buoyancy flux **below** the measured threshold | at F/(Wu³) = 0.035 the widest group lifts off when it should be marginal |
| **P-AM6** | With `C_A = 2/pi`, rise is **over**-suppressed: the widest group fails to lift off well above the measured threshold | K does not lift off at F/(Wu³) = 0.035 |
| **P-AM7** | The two bracket the data, so a value of `C_A` between 0 and 2/pi reproduces the threshold across the series | a single `C_A` puts all six groups within the scatter |
| **NC** | A dense (negative-buoyancy) release is unaffected | Burro LFL statistics identical |

## What would falsify

If P-AM6 fails — if `C_A = 2/pi` reproduces the measured threshold — then
DRIFT's report does not carry over to this formulation and the term should
be enabled by default.

If P-AM7 fails — if no single `C_A` fits the series — the oblate-spheroid
form is the wrong shape function, not merely mis-scaled.

## What must not happen

**`C_A` is not to be adopted from this fit.** The paper argues that fitting
coefficients to data manufactures the compensation it is trying to expose.
The value is reported as a diagnostic — "the published value is too large by
this factor" — and the default stays at the potential-flow value, with the
term off.

---

## RESULTS

*(appended after the run; predictions above unedited)*

*(appended 2026-08-19 after the run; nothing above this line was edited)*

### Setup

Ground-level area source of a light gas, `L` = 6.7 cm, `u`(L) = 1 m/s,
`z0` = 0.03 L, length x/L = 3.43 held, width varied. Reported quantity is
`z_c/h` at 30 L — the centre height as a fraction of cloud depth, which is
the shallow-layer equivalent of "has the concentration maximum left the
ground". A grounded cloud sits near zero.

Hall & Walker's correlation says lift-off is governed by `F/(W u^3)`, so at
a fixed value of that parameter **the six widths should behave alike**. The
spread across widths is therefore the discriminating number, not the level.

### At the measured lift-off threshold, F/(W u³) = 0.035

| `C_A` | G 0.448 | H 1.19 | D 3.43 | I 7.16 | J 14.33 | K 28.66 | **spread** |
|---|---|---|---|---|---|---|---|
| **0** (off) | 0.0334 | 0.0514 | 0.0790 | 0.0993 | 0.1153 | 0.1236 | **3.70** |
| 0.02 | 0.0258 | 0.0373 | 0.0514 | 0.0586 | 0.0611 | 0.0577 | 2.36 |
| 0.05 | 0.0191 | 0.0263 | 0.0333 | 0.0357 | 0.0349 | 0.0309 | 1.87 |
| 0.10 | 0.0132 | 0.0175 | 0.0209 | 0.0214 | 0.0201 | 0.0172 | 1.62 |
| 0.20 | 0.0081 | 0.0104 | 0.0119 | 0.0118 | 0.0107 | 0.0089 | 1.46 |
| **0.6366** (2/π) | 0.0029 | 0.0036 | 0.0039 | 0.0038 | 0.0033 | 0.0027 | **1.46** |

### P-AM5 — CONFIRMED

Without the term the width spread is **3.70**, and monotonic: the widest
source rises nearly four times as high, relative to its depth, as the
narrowest at the same `F/(W u^3)`. Hall & Walker's correlation says they
should be alike. **The width dependence is wrong in the direction they
report** — they found wider sources rise *less*, and here they rise more.

### P-AM6 — CONFIRMED

At `C_A = 2/π` the cloud sits at `z_c/h` = 0.003 where the measurement says
lift-off is occurring (ground concentration down to 10-20 % of maximum).
That is still firmly grounded. **The published coefficient over-suppresses**,
which is what DRIFT's authors reported on this same data.

### P-AM7 — **FAILED**, and this is the result

No single `C_A` does both jobs.

The spread falls from 3.70 to **1.46 and then stops**: raising `C_A` from
0.2 to 0.6366, a factor of three, does not improve the collapse at all
(1.46 to 1.46) while it crushes the absolute level by a further factor of
three (0.012 to 0.003).

The residual 1.46 is also **not monotonic** in width — it peaks at the
middle groups (D, I) and falls again at K. A pure `B/h` scaling cannot
produce that shape at any coefficient.

**The oblate-spheroid form is the wrong shape function, not merely
mis-scaled.** That is a stronger and more useful statement than "the
coefficient is too large", and it explains why DRIFT found the term made
agreement worse rather than better: no choice of coefficient recovers the
measured behaviour.

### NC — PASSED

Dense releases are untouched: `added_mass=True` leaves the Burro LFL
statistics identical, since the term only enters when the cloud is
positively buoyant.

### Status

| | |
|---|---|
| P-AM5 | **CONFIRMED** — width dependence wrong without the term (spread 3.70) |
| P-AM6 | **CONFIRMED** — 2/π over-suppresses (z_c/h = 0.003 at lift-off) |
| P-AM7 | **FAILED** — spread saturates at 1.46; no `C_A` fits |
| NC | **PASSED** |
| adoption | **rejected** — and the reason is now the shape function, not the coefficient |

### What this changes

The project documentation said the added-mass term could not be checked
against measurement for want of data. **That was wrong**: the data was in
the project files, the comparison runs, and it returns a clear negative
result. `C_A` has been externalised as a coefficient so the sweep is
reproducible; the default remains the potential-flow value with the term
off.
