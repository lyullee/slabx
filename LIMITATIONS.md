# Limitations

What `slabx` cannot do, and what it does badly. Each entry says how it is
known, so you can judge the evidence rather than take the claim.

Most of these are inherited from SLAB. A few are ours, and they are marked.

---

## 1. Outside the model entirely

These have no representation in the equations. The model will still return
numbers; they will be wrong.

### Urban and suburban terrain

Two separate objections, and the second is the one that matters.

The validated roughness reaches **0.04 m** (FLADIS, over farmland) — above
the 0.03 m that 49 CFR 193.2059 prescribes, and in fact the trial where the
model does best of the three compared. So the regulatory value is covered.
An earlier version of this document said the trials stopped at 3e-3 m; that
was wrong.

Urban roughness is 0.1 – 2 m, an order of magnitude further. But the reason
not to run an urban case is not that the number is untested. **Roughness of
that size is produced by buildings**, and turning `z0` up does not stand in
for them: the measurement below shows a single fence moving concentration by
a factor of three, which no surface-roughness parameter can reproduce,
because the mechanism is a wake and a recirculation rather than a shear
stress.

### Obstacles, dikes, tanks, buildings, urban arrays

**Measured, not assumed.** The SMEDIS EEC propane trials were run in matched
pairs — same release, with and without an upwind fence. At sensors common to
both:

| pair | common sensors | median concentration ratio | range |
|---|---|---|---|
| EEC170 / 171 | 25 | **0.32** | 0.19 – 1.05 |
| EEC360 / 361 | 17 | 2.06 | 0.42 – **98** |
| EEC550 / 551 | 24 | 0.99 | 0.20 – **35** |

A single fence moves the median by a factor of three and individual sensors by
two orders of magnitude. Pinned in `tests/test_eec.py`.

### Sloped or undulating terrain

Three SMEDIS trials with gradients of 4 – 11.6 % are excluded on this ground.
The model has one horizontal plane.

### Transfer between wind-tunnel and field scale

The model is not Froude-similar. Scaling a trial by a factor of 1000 in
length — with velocity as √s and release rate as s^2.5, which should leave
concentration unchanged — changes it by **2.39×**.

The cause is located: EQ 35a evaluates the entrainment velocity ratio at a
**hard-coded 4 m** (SLAB.FOR line 466, `hrf = 4.0`), an absolute length inside
a similarity closure. Scaling that reference height with the geometry brings
the factor down to 1.57.

Run a wind-tunnel case at tunnel scale, or a field case at field scale. Do not
convert between them.

### Buoyant rise of light gases

Not implemented. `VerticalJet` raises `BuoyantRiseNotImplemented` rather than
silently approximating.

---

## 2. Known failure

### Dense, slow releases — resolved, with a caveat

About one deck in ten of the random differential test disagrees with the
original by more than half at the **default grid**, and the tail is a single
family: a small source area, a high release rate, a heavy material.

**This was recorded as an unresolved structural defect. It is not.** Both
reproducing decks agree with the original once the grid resolves the
transition:

| deck | remedy | result | original |
|---|---|---|---|
| chlorine pool | `n_source` >= 160 | source half-width 1.372 m, h(20 m) = 1.18 m | 1.270 m, 1.24 m |
| two-phase jet | `substeps` >= 8 | h(10 m) = 0.819 m | 0.808 m |

The two respond to **different** grids — a jet has no source region, so
`n_source` does nothing for it. An earlier version of this document called
them the same failure reached through different source types; that was wrong.

**It is a marginal instability, not a convergence error.** The response is
not monotonic in resolution: at `substeps` = 1 the jet deck is correct at
`n_field` = 40 and 240 but diverges at 60, 80 and 120. A systematic
discretisation error would improve monotonically. What this shows is a
delicate transition — the release of accumulated cross-wind gravity momentum
at grounding, the same structure that leaves `e_v` at 0.9997 — that
particular step sizes excite.

**The defaults are not changed.** `n_source` = 10 and `substeps` = 3 are
Ermak's, and they are part of what this project reproduces. Raising them to
resolve a deck costs nothing on the field comparison (LFL MG 1.252 to 1.249,
FAC2 unchanged), so the practical advice is to raise them when a run looks
implausible, not to change the default.

`tests/test_fuzz_fortran.py::test_dense_low_momentum_jet` remains `xfail`
because it pins the behaviour *at the defaults*, which is what a user sees.

**What is left is not one class.** After correcting a fault in the comparison
harness — it did not exclude the region where the original leaves sentinels,
which counted two correctly-behaving chlorine jets as 222 % disagreements —
the rate falls from 14 % to **9 %**, and the three remaining decks are an
instantaneous release, a narrow high-flux pool, and a very wide pool. No
dimensionless criterion separates them from the decks that resolve, and the
density ratio runs the wrong way: the deck that the finer grid fixes is the
densest of them all.

## 3. Where the physics is known to be wrong

These are not crashes. The model runs and returns plausible numbers.

### The stability response is too strong

**Four independent datasets point at the same place.**

1. **Burro.** The surface roughness each trial needs, fitted individually,
   spans **200×** — 0.0006 m for the stable trial, 0.117 m for the unstable
   one — and the ordering follows the Pasquill class.

2. **The damping coefficient.** Sweeping `phi_stable × c_mu` over a decade,
   the value implied by the canonical laboratory flows (Kato & Phillips 1969
   for the stratified-shear limit, Sutton 1953 for the passive limit) is the
   **worst** of the range on every metric. Field data prefer damping about
   four times weaker than Ermak's, which is itself far weaker than the
   laboratory value.

3. **Thorney Island.** McQuaid & Roebuck (1985 §16.5.1) report that peak
   concentration is *"quite insensitive"* to stability. Holding everything
   else fixed and moving B → F changes the model's prediction at 300 m by
   **6.2×**; the correlation across fifteen trials is r = +0.87 (p < 0.001),
   with no confounding by wind speed or density.

4. **Prairie Grass.** Cloud width over-grows with distance in the stable trial
   only (0.95 → 1.76 from 50 m to 800 m); the unstable trial is flat.

The reading is that SLAB applies ambient stability damping even where the
cloud's own gravity dominates mixing. Independently, PHMSA's review of a Phast
submission records an *"unexpected trend of dispersion distance vs stability"*
and mandates an uncertainty factor of 2.5 for stable, low-wind conditions
(PHMSA-2021-0041-0011).

**Practical consequence:** treat stable, low-wind cases as the least reliable.

### Cross-wind gravity momentum is barely damped

Measured from the trajectories, the damping factor `e_v` is **0.9997** — an
effective drag coefficient of 0.006. Gravity supplies more than half the cloud
width even at 900 m. This follows from EQ 5a as written; it is not a porting
error.

### The near field of a vertical jet

Concentration differs from the original by up to a factor 0.77 at 3 m,
recovering to 1.01 by 125 m. Same family as a 1.5 % shape-parameter difference
on the Desert Tortoise jet. Recorded, not tuned away.

---

## 4. The measurements themselves disagree

Not a limitation of the model, but it bounds what any validation of it means.

An earlier instance is on record. EPA's 1991 evaluation reported SLAB
under-predicting Desert Tortoise by 2.6 to 2.74 while noting, in its methods
section, that the developer had identified the source area as
mis-specified and that the correction was not applied
(Zapert et al., EPA-450/4-90-018). The published statistic and the known
cause sat in the same document.

### Two published datasets for the same trials

Two published datasets for the same four Burro trials imply LFL distances that
differ by **1.4 – 2.1×**. Which one you use reverses the recommended surface
roughness:

| z0 | against concentrations (Witlox 2013) | against distances (PHMSA) |
|---|---|---|
| 0.0002 | fails, MG 0.61 | **passes, MG 1.25** |
| 0.03 | **passes, MG 1.12** | fails, MG 1.80 |

For these trials the choice of dataset moves the answer more than any model
change examined in this project. Note also that 49 CFR 193.2059 specifies a
roughness of 0.03 m while the validation database uses 0.0002 m.

The two comparisons are kept in separate modules so that averaging cannot hide
the contradiction.

---

## 5. Validated envelope

`slabx.scope.describe_scope()` prints this and the model warns — but does not
refuse — outside it.

| | range | basis |
|---|---|---|
| `1/L` | −2 to 2 | a physical bound on the Monin-Obukhov length. An earlier 0.15 limit was **withdrawn**: it came from a typo in a JavaScript transcription, not from SLAB |
| `u_ref` | 0.5 – 12 m/s | the span of the validated trials (1.7 – 10.5 m/s); below ~0.5 m/s the plume formulation degenerates |
| `z0` | 1e-5 – 0.1 m | the validated trials span 2e-4 m (Burro, over water) to **0.04 m** (FLADIS, over Danish farmland), with Prairie Grass at 6e-3, Thorney Island at 5e-3 and Desert Tortoise at 3e-3 between. This **covers the 0.03 m** of 49 CFR 193.2059. Suburban and urban roughness (0.1 – 2 m) is untested |

Warnings rather than refusals, because someone modelling an urban release may
have no better option and should get the answer along with its weight.

---

## 6. Not tested at all

| | why |
|---|---|
| ~~Added mass against measurements~~ **(tested)** | **the data existed and has now been used.** Hall & Walker (2000), reported in AEAT/NOIL/27328006/001 (URAHFREP WP7), varied source width 64-fold at fixed buoyancy flux and give lift-off criteria of F/(Wu³) ≈ 0.01 (onset) and ≈ 0.035 (lift-off). Without the term the six widths spread 3.7-fold where the measured correlation says they should collapse, and in the opposite sense. With it the spread falls to 1.46 and **saturates** — tripling `C_A` does not improve it further while crushing the level threefold. **No coefficient fits**, so the oblate-spheroid `B/h` scaling is the wrong shape function rather than a mis-scaled one, which is why DRIFT's authors found it made agreement worse. See `docs/PREREG_added_mass.md` |
| Ground heat transfer | a resistance argument bounds the effect below 2 %: ground conduction supplies 37 000 W/m² after a second and 2 100 after five minutes, while the cloud's own convective film draws only 500-600, so the film is the bottleneck until they cross at about 300 s. Verifying that argument needs a long land spill; none of the validated trials is one |
| Dry deposition | not implemented; none of the ten pool trials used a depositing material |
| Statistical significance of most submodel effects | the samples are small. "Direction confirmed, magnitude not" is the accurate statement for the added physics |
| Fractionation of multicomponent releases | the pseudo-component has a fixed composition |

---

## 7. Numerical

The source-region grid converges at **first order**, not fourth, and the
default of ten steps sits **2.8 %** below the converged answer.

```
n_source     m @ 400 m     vs converged
      10     0.0300757        -2.82 %     <- default
      40     0.0306843        -0.85 %
     320     0.0309373        -0.03 %
```

The cause is structural: EQ 1a replaces the species integration with an
algebraic ramp, and its source term has a slope discontinuity at the pool
edge, which costs Runge–Kutta its order. The default is kept because it is
what the original uses; the cost is measured and pinned in
`tests/test_convergence.py`.

`n_source_steps` is a separate argument from `n_field_steps`, so raising the
field resolution alone leaves this untouched.

The puff grid converges to six figures in 40 steps; the field grid to 0.03 %
in 80.
