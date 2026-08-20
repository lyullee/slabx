"""
Does the shallow-layer height closure cause the bias?
=====================================================

SMEDIS evaluated four classes of dense-gas model against the same field data
with the same arc-wise statistic and found, on the trials with no complex
effects, that integral models were unbiased (ln(MG) = -0.01, FAC2 = 0.74)
while shallow-layer models over-predicted by a factor 1.6 (ln(MG) = -0.49,
FAC2 = 0.65).  `slabx` reproduces the shallow-layer number to two figures
against the Burro trials.

That is a correlation between a model *class* and a bias.  It is not yet a
cause.  The obvious candidate is the one structural thing that distinguishes
the two classes: an integral model carries cloud height as a dependent
variable of its own, while a shallow-layer model recovers it algebraically
from mass conservation,

    h = R / (rho U B)                                            EQ 2b

so any error in the width propagates directly into the height with the
opposite sign, and the cross-section — which is what sets the concentration
— absorbs both.

This module tests that by integrating the height from the vertical
entrainment and recovering the *width* from mass conservation instead —
swapping which of the two is the dependent variable, and changing nothing
else.

A correction made while designing the test
-------------------------------------------
The first version of this prediction assumed that integrating the height
would change the dilution.  It does not.  Mass conservation fixes the
cross-section,

    B h = R / (rho U)

whichever of B and h is algebraic, so the *product* is identical either way
and the cross-wind-averaged concentration cannot move.  What does move is the
centreline value, because the profile carries the width and the shape
parameter separately,

    C(x,0,z) ~ B (h/sigma) Cbar / b,   sigma = h/sqrt3 for a grounded cloud

so ``h/sigma`` is constant and ``C ~ B/b``.  Integrating the height gives a
cloud 1.3 to 2.3 times taller on these trials, hence a width smaller by the
same factor, hence a lower centreline concentration — with ``b`` unchanged,
since it is driven by gravity spreading and not by entrainment.

The test is therefore about the *partition* between width and height, not
about the total dilution.  That is a sharper hypothesis than the one it
replaces: it predicts a specific ratio, not just a direction.

The prediction, registered before the comparison
------------------------------------------------
If the height closure is the cause:

    P1  the over-prediction largely disappears: MG moves from 0.61 towards
        the integral-model value of 0.99, i.e. into 0.85-1.15
    P2  FAC2 rises from 0.67 towards the integral-model 0.74
    P3  the effect is *largest* where the width error is largest, which the
        diagnosis puts in the neutral, windy trials (BU03, BU07, BU09) and
        not in the stable one (BU08), where the model is already unbiased
    P4  the reproduction of the *reference* degrades, because this is a
        deliberate change to the formulation rather than a defect fix — if
        it did not, the change would be doing nothing

If instead the bias comes from somewhere else — the entrainment closure, the
source term, the profile shapes — then P1 and P2 fail while P4 still holds,
and the shallow-layer explanation is wrong.

P3 is the discriminating one.  A change that improves every trial by a
similar amount is behaving like a bias correction, not like a mechanism.

What is deliberately *not* changed
----------------------------------
Entrainment, thermodynamics, the source terms, the profile functions and the
concentration post-processing are untouched, so anything that moves is
attributable to the height closure.  The width equation EQ 7 is also
untouched; only the route from mass to height changes.

Outcome: the hypothesis is falsified
------------------------------------
Implemented as `height_closure="constrained"` in `plume.integrate_plume`,
and run on the four Burro trials against the same observations:

    P1   MG   0.606 -> 0.546        target 0.85-1.15        FAILED
    P1b  Cbar changed by 13.3 %     target < 1 %            FAILED
    P2   FAC2 0.67  -> 0.67         target >= 0.72          FAILED
    P3   effect 5.8 % in the neutral trials, 21.0 % in the
         stable one — the wrong way round                   FAILED

The implementation is sound: mass conservation holds to 1e-3 in both
closures, ``m = q/(2R)`` to 1e-5, and the height does come out 1.1-1.3 times
larger with the width smaller to match.  The bias simply gets *worse*.

Where the reasoning was wrong
------------------------------
P1b was supposed to hold by construction and did not, which is the tell.  The
argument was that mass conservation fixes ``B h``, so the partition cannot
change the dilution.  But EQ 2a entrains through

    dR/dx = rho_a sqrt3 (V_e h + W_e B) + source

which depends on h and B *separately*, not on their product.  A taller,
narrower cloud presents more side area and less top area, and for these
trials the trade is unfavourable: the cloud entrains less overall, so it
stays more concentrated.  The cross-section is therefore not conserved
between closures, and the premise of the test was wrong.

What this does and does not settle
-----------------------------------
It rules out the simplest version of the shallow-layer explanation: the bias
is not caused merely by which of the two lengths is the dependent variable.
Something about the shallow-layer formulation may still be responsible — the
flat-slab geometry, the profile shapes, the single vertical scale — but the
algebraic height recovery on its own is not it.

It does not rescue SLAB either.  The correlation SMEDIS reports between model
class and bias stands; only this particular mechanism for it is eliminated.

The registered predictions are kept above exactly as written.  Rewriting them
after the fact to match what happened would destroy the only thing that makes
the negative result worth anything.

A second candidate, also eliminated
------------------------------------
If the height closure is not responsible, the next structural suspect is the
concentration profile: SLAB reports a flat-topped cross-wind shape with
Gaussian edges, and a rigid ``sigma_z = h/sqrt3``, where an integral model
would carry the vertical scale independently.

That can be tested without changing anything, by asking whether SLAB's own
reported centreline concentration agrees with what its own spreads imply for
a plain Gaussian plume,

    C(x,0,0) = q / (pi u sigma_y sigma_z rho)

On the Burro trials:

    BU03, BU07, BU09 (neutral, windy)   C_slab / C_gauss = 0.76 - 1.03
    BU08 (stable, dense)                C_slab / C_gauss = 0.36 - 0.51

In the neutral trials SLAB *is* a consistent Gaussian plume built on its own
sigmas, so neither the profile shape nor the normalisation can be responsible
for the over-prediction there — it is entirely in the spreads.  Only in the
stable, dense trial does the flat-topped profile do real work, and that is
the one trial the model already gets right.

Three candidates have now been eliminated: the height closure, the profile
shape, and the normalisation.  What remains is the dispersion rates
themselves, which is where the per-trial roughness analysis in
`validation/field_trials.py` also points — the required roughness spans a
factor 200 ordered by stability class, so the deficiency varies with
stability rather than being a constant offset.

What twelve points can no longer separate
------------------------------------------
Distinguishing which stability-dependent term is short — the damping
function, the mixing height, the ambient-stability correction S(La), or the
friction-velocity mapping — needs several trials per stability class.  Burro
gives one class C, two class D and one class E.  The remaining candidates
cannot be told apart on this data, and adding coefficients until the
statistics improve is exactly the failure mode the pre-registration is there
to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Registered:
    """A prediction fixed before the comparison that tests it."""

    name: str
    statement: str
    passes_if: str


#: Registered before any run of the integrated-height variant.
PREDICTIONS = (
    Registered(
        "P1", "the geometric mean bias moves from the shallow-layer value "
              "towards the integral-model value",
        "0.85 <= MG <= 1.15 (from 0.61)",
    ),
    Registered(
        "P1b", "the cross-wind-averaged concentration does NOT move, because "
               "mass conservation fixes B h regardless of which is integrated",
        "median |dCbar| < 1 % — if this fails the change is not what it claims",
    ),
    Registered(
        "P2", "the fraction within a factor of two rises towards the "
              "integral-model value",
        "FAC2 >= 0.72 (from 0.67)",
    ),
    Registered(
        "P3", "the effect concentrates in the trials where the width error "
              "is largest, not uniformly",
        "|dMG| in BU03/BU07/BU09 exceeds |dMG| in BU08 by at least 2x",
    ),
    Registered(
        "P4", "agreement with the reference degrades, since the formulation "
              "has deliberately changed",
        "median |A-C| on cloud height exceeds 5 % (from 0.19 %)",
    ),
)


#: The candidate closures, and what separates them.
#:
#: EQ 2a already carries the entrained mass through ``rho_a sqrt3 (V_e h +
#: W_e B)``.  A height equation that simply integrates the vertical
#: entrainment velocity would count the same air twice — once as mass, once
#: as volume — so the closure has to be stated carefully.
HEIGHT_CLOSURES = {
    "algebraic": (
        "h = R / (rho U B).  Ermak's, and the definition of a shallow-layer "
        "model.  Mass is conserved exactly; height carries whatever error "
        "the width has, inverted."
    ),
    "entrainment": (
        "dh/dx = sqrt3 W_e / U, with the density stretch applied to h as "
        "well as B.  Height then follows the turbulence directly, and mass "
        "conservation has to be restored by letting the *density* or the "
        "concentration absorb the residual — which is what an integral model "
        "does.  This is the variant the prediction above is about."
    ),
    "constrained": (
        "Integrate h from the vertical entrainment, then recover B from "
        "``B = R/(rho U h)``.  Mass stays exact and the roles of the two "
        "lengths are swapped.  This is the variant the predictions above "
        "describe: it is minimal, it leaves every other closure alone, and "
        "the only thing it can change is the width-to-height partition."
    ),
}
