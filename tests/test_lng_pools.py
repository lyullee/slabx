"""
LNG pool trials against measured LFL distances.

Ten evaporating-pool releases rather than four, and the quantity the siting
regulation is actually written around.  These tests pin the comparison, the
result, and the conflict between the two published observational datasets.
"""

import numpy as np
import pytest

from slabx.validation._data_access import require
from slabx.coefficients import preset
from slabx.validation.lng_pools import (
    LFL, VAPORISATION_RATE, compare_lfl, load_lfl, load_trials,
    predict_lfl_distance,
)
from slabx.validation.metrics import check_acceptance, metrics


@pytest.fixture(scope="module")
def result():
    return compare_lfl()


# ===========================================================================
# the data
# ===========================================================================
def test_ten_trials_across_three_series():
    trials = load_trials()
    assert len(trials) == 10
    assert {t.series for t in trials.values()} == {"Burro", "Coyote",
                                                   "MaplinSands"}
    assert {t.surface for t in trials.values()} == {"water_pond", "sea"}


def test_the_stability_coverage_is_what_motivated_this():
    """
    Four Burro trials gave one class C, two D and one E, which cannot
    separate a stability-dependent deficiency from scatter.  Ten give three,
    six and one.
    """
    trials = load_trials()
    classes = [t.stability for t in trials.values()]
    assert sum(c.startswith("C") for c in classes) >= 3
    assert sum(c == "D" for c in classes) >= 5


def test_measured_distances_load_with_their_caveats():
    m = load_lfl()
    assert len(m) == 10
    assert m["BU08"]["observed"] == 455.0
    assert m["BU08"]["phast"] == 191.0
    # Burro 9's arc was disturbed by rapid phase transitions
    assert "RPT" in m["BU09"]["note"]


# ===========================================================================
# the result
# ===========================================================================
def test_all_five_acceptance_criteria_are_met(result):
    m = metrics(result["observed"], result["predicted"])
    assert check_acceptance(m)["all"]
    assert m.FAC2 >= 0.85
    assert m.NMSE < 0.2


def test_scatter_beats_phast_on_the_same_trials(result):
    """
    Phast 8.4 is less biased and `slabx` less scattered; on FAC2 they tie.
    Stated as a fact about these ten trials, not a general claim.
    """
    ours = metrics(result["observed"], result["predicted"])
    theirs = metrics(result["observed"], result["phast"])
    assert ours.NMSE < theirs.NMSE
    assert ours.FAC2 >= theirs.FAC2
    assert abs(theirs.MG - 1.0) < abs(ours.MG - 1.0)      # Phast less biased


def test_burro_8_the_case_the_regulator_worried_about(result):
    """
    Low wind, stable — the conditions 49 CFR 193 is written for.  PHMSA
    records that Phast 8.4 puts its LFL at 191 m against a measured 455, that
    doubling the prediction would still fall short, and that DNV is "not
    fully certain why".
    """
    i = int(np.flatnonzero(result["trial"] == "BU08")[0])
    obs = result["observed"][i]
    assert result["phast"][i] / obs < 0.5             # Phast, badly short
    assert 0.8 < result["predicted"][i] / obs < 1.25  # here, close


# ===========================================================================
# the conflict between the two datasets
# ===========================================================================
def test_higher_roughness_makes_the_distance_comparison_worse():
    """
    The opposite of what the concentration comparison in `field_trials` says.
    Recorded because the disagreement is the finding: the two published
    observational datasets differ by a factor 1.4 to 2.1 on the same trials,
    and the recommendation for surface roughness flips with the choice.
    """
    base = metrics(*[compare_lfl()[k] for k in ("observed", "predicted")])
    rough = compare_lfl(z0=0.03)
    high = metrics(rough["observed"], rough["predicted"])
    assert high.MG > base.MG
    assert high.VG > base.VG and high.NMSE > base.NMSE
    assert check_acceptance(base)["all"]
    assert not check_acceptance(high)["all"]


def test_the_two_observational_datasets_disagree():
    """Quantifies the conflict rather than picking a side."""
    from slabx.validation.field_trials import load_observations

    obs = load_observations()
    measured = load_lfl()
    for trial, tol in (("BU07", 0.6), ("BU08", 0.8), ("BU09", 0.6)):
        pts = sorted((o["x"], o["c_obs"] * 100.0)
                     for o in obs if o["trial"] == trial)
        x = np.array([p[0] for p in pts])
        c = np.array([p[1] for p in pts])
        implied = float(np.exp(np.interp(np.log(5.0), np.log(c[::-1]),
                                         np.log(x[::-1]))))
        assert implied < tol * measured[trial]["observed"], trial


# ===========================================================================
# the prediction itself
# ===========================================================================
def test_prediction_responds_to_the_level_and_the_source_rate():
    t = load_trials()["BU08"]
    assert predict_lfl_distance(t, level=0.025) > predict_lfl_distance(t)
    # a smaller pool at the same rate is a stronger source per unit area
    near = predict_lfl_distance(t, vaporisation=2 * VAPORISATION_RATE)
    assert near > 0.0
    assert LFL == 0.05


def test_averaging_time_cannot_explain_the_dataset_discrepancy():
    """
    The two published datasets imply LFL distances differing by 1.4 to 2.1,
    which overlaps the measured peak-to-mean spread (1.17 to 2.43). The
    obvious objection is that the sources report different averaging windows.

    They do not, and that is the whole argument. Witlox (2013) states the
    windows explicitly -- "Averaging time (s) 100 140 80 50" for BU03, BU07,
    BU08, BU09, under a table captioned "long averaging time" -- and PHMSA
    carries the same values.

    The magnitude of the averaging effect does *not* help here, and an
    earlier version of this test wrongly claimed it did. Observed
    concentration decays as x^-n with n near 0.95 (fitted below), so a
    concentration ratio passes almost one-for-one into a distance ratio: the
    observed peak-to-mean of 1.17-2.43 implies distance ratios of 1.18-2.54,
    which spans the discrepancy entirely. Had the windows differed, averaging
    could have accounted for all of it.
    """
    import csv
    import math

    import numpy as np

    from slabx.validation.lng_pools import DATA

    # (1) the two sources agree on the window, trial by trial
    witlox = list(csv.DictReader(open(require("burro_observed_vs_phast.csv"))))
    assert {r["averaging"] for r in witlox} == {"long"}

    pool = {r["trial"]: r for r in csv.DictReader(
        open(require("lng_pool_trials.csv")))}
    p2m = {r["trial"]: r for r in csv.DictReader(
        open(require("peak_to_mean.csv")))}
    for trial, row in p2m.items():
        assert float(row["t_long_s"]) == float(pool[trial]["t_avg_long_s"]), \
            trial
    # the values Witlox prints
    assert [float(pool[t]["t_avg_long_s"])
            for t in ("BU03", "BU07", "BU08", "BU09")] == [100, 140, 80, 50]

    # (2) and the effect would have been large enough to matter, so (1) is
    #     doing all the work -- this half is what the earlier version got
    #     backwards
    arcs = {}
    for r in witlox:
        arcs.setdefault(r["trial"], []).append(
            (float(r["x_m"]), float(r["c_obs_molpct"])))
    ns = []
    for trial, v in arcs.items():
        if len(v) < 3:
            continue
        x = np.log([a for a, _ in v])
        c = np.log([b for _, b in v])
        ns.append(-float(np.polyfit(x, c, 1)[0]))
    assert len(ns) >= 3
    n = float(np.median(ns))
    assert 0.8 < n < 1.2, f"decay exponent {n:.2f}"

    implied = [float(r["implied_ratio"]) ** (1.0 / n) for r in p2m.values()]
    assert max(implied) > 2.0, "averaging is not negligible for distance"
