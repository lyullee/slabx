"""
Field-trial validation
======================

Runs `slabx` against measured concentrations and reports the standard
performance measures.  This is the first point in the project where the
question stops being "does it reproduce Ermak?" and becomes "is it right?".

Pre-registration
----------------
Every model variant carries a written prediction of *where its effect should
appear and where it should not*, recorded in `PREDICTIONS` before the
comparison is run.  A variant that improves agreement in the trials it was
supposed to and leaves the others alone is evidence about the physics; one
that improves everything by a similar amount is evidence about a free
parameter, and one that improves the wrong trials is evidence against the
mechanism regardless of the aggregate statistics.

None of the three physics changes on offer was calibrated on dispersion
data:

    canonical  c_mu_strat from Sutton (1953) and Kato & Phillips (1969)
    coolprop   reference equations of state
    cf_tno     C_f from TNO 4.127

so this comparison is a test, not a fit.  `frontal` is the exception — its
coefficient is a placeholder — and it is labelled accordingly.

Data
----
`data/burro_observed_vs_phast.csv` holds the observed arc-wise maximum
concentrations at 1 m from Witlox (2013), together with that paper's Phast
predictions for context.  Note the averaging times: the trials were
processed at 50-140 s, not at SLAB's 10 s default, and the input table
carries the value per trial.

Independent corroboration of the baseline
-----------------------------------------
SMEDIS (EU Scientific Model Evaluation of Dense Gas Dispersion Models, final
report Table 4.8) evaluated four classes of model against dense-gas field
data using the same arc-wise maximum-concentration comparison.  For the
datasets with no complex effects — the group the Burro trials fall in — it
reports, for the shallow-layer class that SLAB belongs to:

    FAC2 = 0.65,   ln(MG) = -0.49  (MG = 0.61)

Against the same measure this module gives, for the unmodified model at the
MVD base roughness:

    FAC2 = 0.67,   MG = 0.605

That is a match on both, from an independent multi-model exercise using
different trials and different modellers.  Two things follow.

First, the port is validated against SLAB's documented *performance*, not
just against its code.

Second, the 1.6-fold over-prediction is a real and previously documented
property of shallow-layer models at standard roughness — not an artefact of
the source area assumed here or of the roughness chosen.  Raising the
roughness to FERC's value therefore masks a known model bias rather than
correcting a data error, which is why it cannot be justified on the strength
of the improved statistics alone.

SMEDIS also reports the *integral* model class at ln(MG) = -0.01 and
FAC2 = 0.74 on the same data, so the bias is specific to the shallow-layer
formulation — the flat slab with height recovered algebraically from
``h = R/(rho U B)`` — and not to integral dense-gas modelling in general.

Caveats that limit what can be concluded
----------------------------------------
* Twelve points across four trials.  Wide confidence intervals are expected
  and the paired bootstrap is what makes the *differences* meaningful even
  so.
* LNG is modelled as pure methane while the trials were 87-93 % methane with
  ethane and propane; `data/burro_lng_composition.csv` records the real
  composition, which the single-component thermodynamics cannot use.
* Source rates and averaging times differ between published sources.  The
  Witlox values are used throughout so that the Phast column stays
  comparable.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..coefficients import COEFFS, Coefficients, preset
from ..core.plume import run_dispersion
from ..core.source import EvaporatingPool
from ..post.concentration import concentration_field
from ..submodels.atmosphere import Atmosphere
from ..thermo.base import LegacyThermo, Substance, water_backend
from .metrics import check_acceptance, compare, metrics

__all__ = [
    "BURRO_TRIALS", "PREDICTIONS", "VARIANTS", "SLAB90", "Variant",
    "ROUGHNESS", "METHANE", "load_composition",
    "load_observations", "load_conditions", "run_trial", "predictions_for",
    "validate",
]

DATA = Path(__file__).parent / "data"

from ._data_access import ObservationsUnavailable, require  # noqa: E402

#: SLAB's own LNG property set (manual example 1, example 3 latent heat).
METHANE = Substance(
    name="LNG", mw=0.016043, cp_vapour=2238.0, cp_liquid=3348.5,
    dh_vap=509900.0, T_boil=111.7, rho_liquid=424.1,
)

#: Surface roughness cases.  This turns out to matter more than any model
#: change tested here, so it is named rather than buried in the input file.
#:
#: The MVD (03903-RP-002, section 2.2.5.4) records that Stewart et al. and
#: the MDA give 0.0002 m for the Burro site, that FERC proposed 0.01 m as
#: "the upper range of soils and short grass", and that the MVD rejected the
#: latter as "excessive, given the strong effect this parameter can have on
#: atmospheric dispersion", capping its own sensitivity at 0.002 m.
#:
#: The rejected value is the one at which the *aggregate* works.  At 0.0002 m
#: the original SLAB over-predicts by 1.65 and fails Chang-Hanna on three
#: measures; at 0.01 m it passes on all five, unchanged.  That is a larger
#: effect than any physics change examined here.
#:
#: But it passes by cancellation, not by being right.  Solving each trial
#: separately for the roughness that would make it unbiased gives
#:
#:     BU08  (E, u = 1.9 m/s)   0.0006
#:     BU07  (D, u = 8.8 m/s)   0.022
#:     BU09  (D, u = 5.9 m/s)   0.025
#:     BU03  (C, u = 5.6 m/s)   0.117
#:
#: — a spread of 200, ordered by stability class.  A single roughness cannot
#: describe all four, so the residual is condition-dependent and roughness is
#: standing in for something that varies with stability.  SLAB's entrainment
#: does respond to stability (W_e rises 6.8-fold from class E to C at fixed
#: wind), just not by enough.
#:
#: Dropping BU09, which PHMSA excludes from its own MVD because a rapid phase
#: transition disturbed the 57 m arc, does not change any of this.
ROUGHNESS = {"mvd_base": 0.0002, "mvd_upper": 0.002, "ferc": 0.01}

#: Evaporation flux of LNG on water [kg/(m^2 s)], back-figured from the one
#: documented case: SLAB's manual gives 657 m^2 for BU08 at 116.93 kg/s.
#: The result, 0.178, sits inside the 0.085-0.2 range reported for LNG on
#: water, so the rule is at least not contradicted by the literature.
LNG_ON_WATER_FLUX = 116.93 / 657.0

#: Source areas [m^2].  Only BU08's is documented; the rest follow from
#: ``A = q / E`` with the flux above.
#:
#: This is the weakest input in the whole comparison and it should be read
#: that way.  A pool spreading on water is set by the balance between spill
#: and evaporation, and a constant flux is a crude closure for it.  The
#: paired design is what makes the variant comparisons survive this: the
#: area error is common to baseline and variant and cancels in the
#: difference, even though it dominates the absolute agreement.
BURRO_AREA = {t: round(q / LNG_ON_WATER_FLUX, 1) for t, q in
              (("BU03", 87.98), ("BU07", 99.46), ("BU08", 116.93),
               ("BU09", 135.98))}

BURRO_TRIALS = ("BU03", "BU07", "BU08", "BU09")


# ===========================================================================
# pre-registered predictions
# ===========================================================================
@dataclass(frozen=True)
class Variant:
    """
    A fully specified model configuration.

    Every switch that distinguishes one run from another lives here, so that
    "which model was this?" has a single answer.  An earlier version carried
    only the coefficient preset and left the two atmosphere flags at their
    defaults, which meant the nominal baseline was **not** the original SLAB:
    it silently had the stability-mapping typo and the unstable wind-profile
    transcription error already corrected.  A baseline that is not the thing
    it claims to be invalidates every difference measured against it.
    """

    coefficients: str
    real_properties: bool = False
    composition: bool = False
    """Model the release as its measured mixture rather than pure methane."""
    legacy_stability_bug: bool = False
    legacy_unstable_profile: bool = False

    def atmosphere(self, cond: dict) -> Atmosphere:
        return Atmosphere(
            u_ref=cond["u_ref"], z_ref=cond["z_ref"], T=cond["T"],
            rh=cond["rh"], z0=cond["z0"], stability=cond["stability"],
            p=cond["p"],
            legacy_stability_bug=self.legacy_stability_bug,
            legacy_unstable_profile=self.legacy_unstable_profile,
        )


#: The reference point.
#:
#: Named `SLAB90` for the year of Ermak's release, but what the flags
#: reproduce is the **JavaScript transcription** that this project first
#: compared against, not the Fortran: `legacy_stability_bug` recreates a
#: single-character typo (`all` for `aal`) that exists only in that copy.
#: The name is kept because it appears in the pre-registered predictions,
#: which are not edited after the fact.
#:
#: On these trials it makes no difference — `slab90`, a variant with the
#: typo off, and `slabx` return identical statistics to four figures
#: (FAC2 0.67, MG 0.605), because the Burro decks sit at |1/L| = 0.0665 and
#: the faulty branch never fires. The distinction matters for *attribution*,
#: not for these numbers.
SLAB90 = Variant(coefficients="ermak90", real_properties=False,
                 legacy_stability_bug=True, legacy_unstable_profile=True)

#: What Ermak's Fortran actually does: `aal` in the stability mapping
#: (SLAB.FOR L236) and his own unstable wind profile (L1860-1862).
ERMAK90 = Variant(coefficients="ermak90", real_properties=False,
                  legacy_stability_bug=False, legacy_unstable_profile=True)

#: Named configurations.
#:
#: `slab90` reproduces the JavaScript transcription; `ermak90` reproduces the
#: Fortran; `slabx` additionally replaces Ermak's unstable wind profile with
#: the exact integral. All three agree to four figures on these trials, so
#: any of them serves as the baseline for judging a *physics* change — but
#: `ermak90` is the one to cite when the claim is about SLAB.
VARIANTS: dict[str, Variant] = {
    "slab90": SLAB90,
    "ermak90": ERMAK90,
    "slabx": Variant(coefficients="ermak90"),
    "canonical": Variant(coefficients="canonical"),
    "coolprop": Variant(coefficients="ermak90", real_properties=True),
    "composition": Variant(coefficients="ermak90", real_properties=True,
                           composition=True),
    "canonical+coolprop": Variant(coefficients="canonical",
                                  real_properties=True),
    "frontal": Variant(coefficients="frontal"),
}


@dataclass(frozen=True)
class Prediction:
    """What a variant is expected to do, written down before the run."""

    variant: str
    baseline: str
    rationale: str
    expect_effect_in: tuple[str, ...]
    expect_no_effect_in: tuple[str, ...]


#: Registered before any comparison was run.  Each names its own baseline,
#: because the right comparison differs: a defect correction is judged
#: against the original, a physics change against the corrected port.
PREDICTIONS: dict[str, Prediction] = {
    "slabx": Prediction(
        variant="slabx", baseline="slab90",
        rationale=(
            "The three corrections are a stability-mapping typo that only "
            "fires for |1/L| below about 0.15, a wind-profile transcription "
            "error confined to unstable classes, and a 2.6 % rounding of one "
            "drag coefficient.  Burro 3 is the only unstable trial, so it is "
            "the only one where the profile correction can act at all; the "
            "rest should barely move."
        ),
        expect_effect_in=(),
        expect_no_effect_in=("BU07", "BU08", "BU09"),
    ),
    "canonical": Prediction(
        variant="canonical", baseline="slabx",
        rationale=(
            "c_mu_strat sets how fast entrainment is damped by in-cloud "
            "stratification, so the change must grow with the density excess "
            "the cloud sustains and with ambient stability.  BU08 is the "
            "stable, low-wind trial and should move most; BU07 and BU09 are "
            "neutral with strong winds and should move least."
        ),
        expect_effect_in=("BU08",),
        expect_no_effect_in=("BU07", "BU09"),
    ),
    "coolprop": Prediction(
        variant="coolprop", baseline="slabx",
        rationale=(
            "Real properties act through the phase split and the equation of "
            "state, so the effect must concentrate where the cloud is cold "
            "and partly condensed — near the source and in the trial that "
            "stays coldest longest.  Far downwind, where the cloud is "
            "essentially ambient air, it should vanish."
        ),
        expect_effect_in=("BU08",),
        expect_no_effect_in=(),
    ),
    "composition": Prediction(
        variant="composition", baseline="coolprop",
        rationale=(
            "Burro was 87-93 % methane with ethane and propane, and is "
            "modelled here as pure methane.  The measured mixture is 8-14 % "
            "heavier, which lowers the predicted *mole* fraction — what the "
            "sensors report — for a given mass.  The effect must therefore "
            "scale with the non-methane content: BU03 at 92.5 % methane "
            "should move least, and the three trials near 87 % should move "
            "more and by similar amounts."
        ),
        expect_effect_in=("BU07", "BU08", "BU09"),
        expect_no_effect_in=(),
    ),
    "canonical+coolprop": Prediction(
        variant="canonical+coolprop", baseline="slabx",
        rationale=(
            "The two physics changes act through different routes — one on "
            "the entrainment damping, one on the phase split — so if neither "
            "is absorbing the other's error their effects should combine "
            "roughly additively.  A combined effect much smaller than the "
            "sum would mean they are compensating, which is a warning that "
            "one of them is standing in for something else."
        ),
        expect_effect_in=("BU08",),
        expect_no_effect_in=(),
    ),
    "frontal": Prediction(
        variant="frontal", baseline="slabx",
        rationale=(
            "Gravity-front entrainment scales with V_g, which is largest "
            "where the cloud stays dense and the wind is weak.  BU08 is that "
            "trial.  Unlike the other variants this coefficient is a "
            "placeholder, not a derived value, so the result says only "
            "whether the mechanism acts where it should."
        ),
        expect_effect_in=("BU08",),
        expect_no_effect_in=(),
    ),
}


# ===========================================================================
# data
# ===========================================================================
def _read(name: str) -> list[dict]:
    with open(require(name), newline="") as fh:
        return list(csv.DictReader(fh))


def load_observations(trials=BURRO_TRIALS) -> list[dict]:
    """Observed arc-wise maximum concentrations [mol %] at 1 m."""
    out = []
    for r in _read("burro_observed_vs_phast.csv"):
        if r["trial"] not in trials or not r["c_obs_molpct"]:
            continue
        out.append(dict(
            trial=r["trial"], x=float(r["x_m"]), z=float(r["z_m"]),
            c_obs=float(r["c_obs_molpct"]) / 100.0,
            c_phast=float(r["c_phast_molpct"]) / 100.0
            if r["c_phast_molpct"] else math.nan,
        ))
    return out


def load_conditions(source: str = "Witlox2013_T2") -> dict[str, dict]:
    """Release and meteorological conditions, one entry per trial."""
    out = {}
    for r in _read("burro_inputs_multisource.csv"):
        if r["source"] != source:
            continue
        out[r["trial"]] = dict(
            rate=float(r["rate_kg_s"]), duration=float(r["duration_s"]),
            u_ref=float(r["u_ref_m_s"]), z_ref=float(r["z_ref"] if "z_ref" in r
                                                    else r["z_ref_m"]),
            stability=r["stability"], T=float(r["T_K"]), p=float(r["p_Pa"]),
            rh=float(r["RH_pct"]), z0=float(r["z0_m"]),
            t_avg=float(r["t_avg_s"]),
        )
    return out


# ===========================================================================
# running
# ===========================================================================
def load_composition() -> dict[str, dict[str, float]]:
    """Measured LNG composition per trial, as CoolProp mole fractions."""
    out = {}
    for r in _read("burro_lng_composition.csv"):
        out[r["trial"]] = {"Methane": float(r["CH4_pct"]) / 100.0,
                           "Ethane": float(r["C2H6_pct"]) / 100.0,
                           "n-Propane": float(r["C3H8_pct"]) / 100.0}
    # BU09's composition was not reported; BU08 was spilled from the same
    # batch and is used in its place, which is recorded rather than hidden.
    out.setdefault("BU09", out["BU08"])
    return out


def run_trial(trial: str, cond: dict, variant: str | Variant = "slabx",
              *, x_max: float = 1000.0, **run_kw):
    """
    One Burro trial under one fully specified configuration.

    Extra keywords go to `run_dispersion`, which is how the grid-convergence
    check reaches `n_source_steps` without any change to the model.
    """
    v = VARIANTS[variant] if isinstance(variant, str) else variant
    atm = v.atmosphere(cond)

    sub, fluid = METHANE, "Methane"
    if v.composition:
        from ..thermo.coolprop import pseudo_component
        sub = pseudo_component("LNG", load_composition()[trial])
        fluid = None
    src = EvaporatingPool(substance=sub, rate=cond["rate"],
                          area=BURRO_AREA[trial], duration=cond["duration"])
    if v.real_properties:
        from ..thermo.coolprop import CoolPropThermo, coolprop_water
        em = CoolPropThermo(sub, fluid=fluid or "Methane")
        wt = coolprop_water()
    else:
        em, wt = LegacyThermo(sub), water_backend()

    traj, _ = run_dispersion(src, atm, em, wt, x_max=x_max, n_puff_steps=60,
                             coeffs=preset(v.coefficients), **run_kw)
    field = concentration_field(traj, atm, z=1.0, t_avg=cond["t_avg"],
                                t_release=cond["duration"])
    return traj, field


def _at(field, x: float) -> float:
    v = np.maximum(field.peak, 1e-30)
    return float(np.exp(np.interp(x, field.x, np.log(v))))


def predictions_for(variant: str, obs, conditions, **kw) -> np.ndarray:
    """Model concentration at every observation point, for one variant."""
    fields = {t: run_trial(t, conditions[t], variant, **kw)[1]
              for t in sorted({o["trial"] for o in obs})}
    return np.array([_at(fields[o["trial"]], o["x"]) for o in obs])


# ===========================================================================
def validate(trials=BURRO_TRIALS, n_boot: int = 20000,
             roughness: str | float = "mvd_base", **kw) -> dict:
    """
    Every registered prediction, each against the baseline it declares.

    Returns predictions per variant, the `Metrics` for each, and the paired
    `Comparison` for each prediction.
    """
    obs = list(load_observations(trials))
    z0 = ROUGHNESS[roughness] if isinstance(roughness, str) else roughness
    conditions = {t: {**c, "z0": z0} for t, c in load_conditions().items()}
    y = np.array([o["c_obs"] for o in obs])
    tid = np.array([o["trial"] for o in obs])

    pred = {v: predictions_for(v, obs, conditions, **kw) for v in VARIANTS}
    out = {
        "observations": obs, "y": y, "trial": tid, "predictions": pred,
        "z0": z0,
        "metrics": {v: metrics(y, p) for v, p in pred.items()},
        "comparisons": {
            name: compare(y, pred[p.baseline], pred[p.variant], trial=tid,
                          label_a=p.baseline, label_b=p.variant,
                          n_boot=n_boot)
            for name, p in PREDICTIONS.items()
        },
    }
    return out
