"""
Field-trial validation harness.

These tests check the *harness*, not the model: that the data load, that the
pre-registered predictions are well formed, and that the negative control
behaves like a negative control.  Whether the model agrees with the
observations is a result, not an assertion, and is reported by
`examples/validate_burro.py` rather than pinned here.
"""

import math

import numpy as np
import pytest

from slabx.validation.field_trials import (
    BURRO_AREA,
    BURRO_TRIALS,
    LNG_ON_WATER_FLUX,
    PREDICTIONS,
    SLAB90,
    VARIANTS,
    load_conditions,
    load_observations,
    predictions_for,
    run_trial,
)
from slabx.validation.metrics import compare, metrics


@pytest.fixture(scope="module")
def data():
    return load_observations(), load_conditions()


# ===========================================================================
# data
# ===========================================================================
def test_observations_load(data):
    obs, _ = data
    assert len(obs) == 12
    assert {o["trial"] for o in obs} == set(BURRO_TRIALS)
    for o in obs:
        assert 0.0 < o["c_obs"] < 1.0          # a volume fraction, not %
        assert o["z"] == 1.0                    # all reported at 1 m


def test_conditions_cover_every_trial(data):
    _, cond = data
    assert set(cond) == set(BURRO_TRIALS)
    for c in cond.values():
        assert c["rate"] > 0 and c["duration"] > 0
        assert c["stability"] in "ABCDEF"
        assert c["t_avg"] >= 50.0               # long averaging, not SLAB's 10 s


def test_source_area_rule_is_anchored_on_the_documented_case():
    """
    Only BU08's source area is documented (657 m^2 in SLAB's own manual).
    The others follow from A = q/E with E back-figured from it, and E must
    land inside the range reported for LNG boiling on water.
    """
    assert BURRO_AREA["BU08"] == pytest.approx(657.0, rel=1e-3)
    assert 0.085 <= LNG_ON_WATER_FLUX <= 0.2
    assert BURRO_AREA["BU03"] < BURRO_AREA["BU08"] < BURRO_AREA["BU09"]


# ===========================================================================
# pre-registration
# ===========================================================================
def test_every_variant_carries_a_prediction():
    for name, p in PREDICTIONS.items():
        assert p.rationale.strip()
        assert p.variant in VARIANTS and p.baseline in VARIANTS
        assert set(p.expect_effect_in) <= set(BURRO_TRIALS)
        assert set(p.expect_no_effect_in) <= set(BURRO_TRIALS)
        assert not (set(p.expect_effect_in) & set(p.expect_no_effect_in))


def test_the_original_model_is_the_original_model():
    """
    Regression guard for a defect that quietly invalidated everything
    measured against the baseline: the validation used to build its
    atmosphere with the default flags, so its nominal "SLAB" already had the
    stability-mapping typo and the unstable-profile transcription error
    corrected.  A baseline that is not what it claims to be makes every
    difference meaningless.
    """
    assert SLAB90.coefficients == "ermak90"
    assert SLAB90.legacy_stability_bug is True
    assert SLAB90.legacy_unstable_profile is True
    assert SLAB90.real_properties is False
    assert VARIANTS["slab90"] is SLAB90


def test_defect_corrections_are_judged_against_the_original():
    """
    A physics change must not take credit for the transcription fixes, and a
    transcription fix must be measured against the code that had the defect.
    """
    assert PREDICTIONS["slabx"].baseline == "slab90"
    for name in ("canonical", "coolprop", "frontal"):
        assert PREDICTIONS[name].baseline == "slabx"


def test_variants_differ_only_where_declared():
    """Each variant must change exactly the switch its name implies."""
    base = VARIANTS["slabx"]
    assert VARIANTS["canonical"].coefficients == "canonical"
    assert VARIANTS["canonical"].real_properties == base.real_properties
    assert VARIANTS["coolprop"].real_properties is True
    assert VARIANTS["coolprop"].coefficients == base.coefficients
    combo = VARIANTS["canonical+coolprop"]
    assert combo.coefficients == "canonical" and combo.real_properties is True


# ===========================================================================
# the harness runs
# ===========================================================================
def test_a_trial_runs_and_gives_physical_concentrations(data):
    _, cond = data
    traj, field = run_trial("BU08", cond["BU08"])
    assert len(traj) > 20
    assert np.all(np.isfinite(field.peak))
    assert np.all((field.peak >= 0.0) & (field.peak <= 1.0))
    assert np.all(np.diff(field.peak[field.x > 60.0]) < 0)


def test_defect_corrections_are_negligible_on_these_trials(data):
    """
    The three corrections act only in conditions the Burro trials barely
    visit: the stability typo needs |1/L| below about 0.15, the profile error
    needs an unstable class, and the drag rounding is 2.6 % of one
    coefficient.  So `slabx` must sit essentially on top of `slab90` here —
    which is also what makes it a usable baseline for the physics changes.
    """
    obs, cond = data
    a = predictions_for("slab90", obs, cond)
    b = predictions_for("slabx", obs, cond)
    assert np.allclose(a, b, rtol=0.02)


def test_paired_comparison_is_tighter_than_the_absolute_skill(data):
    """
    The reason the source-area uncertainty does not sink the variant
    comparisons: it is common to both runs and cancels in the difference.
    """
    obs, cond = data
    y = np.array([o["c_obs"] for o in obs])
    tid = np.array([o["trial"] for o in obs])
    base = predictions_for("slabx", obs, cond)
    alt = predictions_for("coolprop", obs, cond)

    c = compare(y, base, alt, trial=tid, n_boot=4000)
    width_delta = c.ci["MG"][1] - c.ci["MG"][0]

    rng = np.random.default_rng(0)
    groups = [np.flatnonzero(tid == t) for t in BURRO_TRIALS]
    draws = [metrics(y[p], base[p]).MG for p in
             (np.concatenate([groups[i] for i in rng.integers(0, 4, 4)])
              for _ in range(4000))]
    width_abs = np.percentile(draws, 97.5) - np.percentile(draws, 2.5)
    assert width_delta < 0.5 * width_abs


# ===========================================================================
# surface roughness
# ===========================================================================
def test_roughness_cases_match_the_validation_database():
    """
    03903-RP-002 section 2.2.5.4: base 0.0002 m, MVD upper bound 0.002 m,
    and FERC's proposed 0.01 m which the MVD declined to adopt.
    """
    from slabx.validation.field_trials import ROUGHNESS

    assert ROUGHNESS["mvd_base"] == 0.0002
    assert ROUGHNESS["mvd_upper"] == 0.002
    assert ROUGHNESS["ferc"] == 0.01


def test_roughness_dominates_every_model_change(data):
    """
    The result that reframes the whole comparison.

    Moving the roughness from the MVD base value to FERC's changes the
    geometric mean bias by more than any physics variant does, and flips the
    unchanged original model from failing three Chang-Hanna criteria to
    passing all five.  Any claim about a model change has to be read against
    that.
    """
    from slabx.validation.field_trials import ROUGHNESS

    obs, base = data
    y = np.array([o["c_obs"] for o in obs])

    out = {}
    for name in ("mvd_base", "ferc"):
        cond = {t: {**c, "z0": ROUGHNESS[name]} for t, c in base.items()}
        out[name] = metrics(y, predictions_for("slab90", obs, cond))

    assert out["mvd_base"].MG < 0.7          # over-predicts by ~1.65
    assert 0.7 < out["ferc"].MG < 1.3        # unbiased
    assert out["ferc"].VG < out["mvd_base"].VG
    assert out["ferc"].NMSE < 0.5 * out["mvd_base"].NMSE

    from slabx.validation.metrics import check_acceptance
    assert not check_acceptance(out["mvd_base"])["all"]
    assert check_acceptance(out["ferc"])["all"]


def test_real_properties_are_the_only_variant_that_helps_at_ferc(data):
    """
    At the roughness where the model is unbiased, only the property change
    improves it — and it improves FAC2, VG and NMSE together, which a bias
    shift cannot do.  The entrainment-damping variant makes VG much worse.
    """
    from slabx.validation.field_trials import ROUGHNESS

    obs, base = data
    y = np.array([o["c_obs"] for o in obs])
    cond = {t: {**c, "z0": ROUGHNESS["ferc"]} for t, c in base.items()}

    ref = metrics(y, predictions_for("slabx", obs, cond))
    cool = metrics(y, predictions_for("coolprop", obs, cond))
    canon = metrics(y, predictions_for("canonical", obs, cond))

    assert cool.FAC2 > ref.FAC2
    assert cool.VG < ref.VG and cool.NMSE < ref.NMSE
    assert canon.VG > 2.0 * ref.VG


@pytest.mark.filterwarnings("ignore::slabx.scope.ScopeWarning")
def test_no_single_roughness_fits_all_four_trials(data):
    """
    The finding that stops "just use FERC's roughness" from being the answer.

    Solving each trial separately for the roughness that would make it
    unbiased gives values spanning two orders of magnitude, ordered by
    stability class.  The aggregate passes at 0.01 m by over-correcting the
    stable trial and under-correcting the unstable one, so the residual is
    condition-dependent and roughness is standing in for something else.
    """
    from scipy.optimize import brentq

    obs, base = data
    y = np.array([o["c_obs"] for o in obs])
    tid = np.array([o["trial"] for o in obs])

    def bias(z0, mask):
        cond = {t: {**c, "z0": z0} for t, c in base.items()}
        return metrics(y[mask], predictions_for("slabx", obs, cond)[mask]).MG

    needed = {}
    for t in ("BU03", "BU08"):
        m = tid == t
        needed[t] = 10 ** brentq(lambda lz: bias(10 ** lz, m) - 1.0,
                                 -4.0, -0.5, xtol=2e-3)

    # the stable trial wants the MVD base value, the unstable one wants
    # two orders of magnitude more
    assert needed["BU08"] < 0.002
    assert needed["BU03"] > 0.05
    assert needed["BU03"] / needed["BU08"] > 50.0


def test_excluding_the_rpt_affected_trial_changes_nothing(data):
    """
    PHMSA drops BU09 from its own MVD because a rapid phase transition
    disturbed the 57 m arc.  The conclusions must not depend on it.
    """
    from slabx.validation.field_trials import ROUGHNESS
    from slabx.validation.metrics import check_acceptance

    obs, base = data
    y = np.array([o["c_obs"] for o in obs])
    keep = np.array([o["trial"] != "BU09" for o in obs])
    cond = {t: {**c, "z0": ROUGHNESS["ferc"]} for t, c in base.items()}

    p = predictions_for("slab90", obs, cond)
    assert check_acceptance(metrics(y, p))["all"]
    assert check_acceptance(metrics(y[keep], p[keep]))["all"]


# ===========================================================================
# independent corroboration
# ===========================================================================
#: SMEDIS final report, Table 4.8 (arc-wise comparison), shallow-layer model
#: class, datasets with no complex effects.  ln(MG) = -0.49, FAC2 = 0.65.
SMEDIS_SHALLOW_LAYER = {"MG": math.exp(-0.49), "FAC2": 0.65}


def test_baseline_matches_the_independent_smedis_evaluation(data):
    """
    The strongest external check available on the baseline.

    SMEDIS evaluated four model classes against dense-gas field data with the
    same arc-wise maximum-concentration statistic.  For the shallow-layer
    class — SLAB's — on datasets with no complex effects it reports FAC2 =
    0.65 and MG = 0.61.  Reproducing both from a different implementation,
    different trials and different modellers means the port carries SLAB's
    documented performance and not merely its code.

    It also means the 1.6-fold over-prediction is a known property of the
    model class, so raising the roughness until the statistics improve is
    masking it rather than fixing it.
    """
    from slabx.validation.field_trials import ROUGHNESS

    obs, base = data
    y = np.array([o["c_obs"] for o in obs])
    cond = {t: {**c, "z0": ROUGHNESS["mvd_base"]} for t, c in base.items()}
    m = metrics(y, predictions_for("slab90", obs, cond))

    assert m.MG == pytest.approx(SMEDIS_SHALLOW_LAYER["MG"], rel=0.10)
    assert m.FAC2 == pytest.approx(SMEDIS_SHALLOW_LAYER["FAC2"], abs=0.10)


def test_slab90_reproduces_the_transcription_not_the_fortran():
    """
    A labelling correction, pinned so it cannot drift back.

    `SLAB90` carries `legacy_stability_bug=True`, which recreates a
    single-character typo (`all` for `aal`) present only in the JavaScript
    copy this project first compared against.  Ermak's Fortran reads `aal`
    (SLAB.FOR L236).  The docstring used to call this variant "SLAB exactly
    as Ermak released it", which was wrong.

    On these trials the distinction is invisible: the Burro decks sit at
    |1/L| = 0.0665 and the faulty branch never fires.  It matters for
    attribution, not for the numbers, and this test states both halves.
    """
    from slabx.validation.field_trials import ERMAK90, SLAB90

    assert SLAB90.legacy_stability_bug is True
    assert ERMAK90.legacy_stability_bug is False
    # Ermak's own unstable profile is genuine, so both keep it
    assert SLAB90.legacy_unstable_profile is True
    assert ERMAK90.legacy_unstable_profile is True

    obs, cond = load_observations(), load_conditions()
    y = np.array([o["c_obs"] for o in obs])
    a = metrics(y, predictions_for(SLAB90, obs, cond))
    b = metrics(y, predictions_for(ERMAK90, obs, cond))
    assert a.MG == pytest.approx(b.MG, rel=1e-4)
    assert a.FAC2 == pytest.approx(b.FAC2)
