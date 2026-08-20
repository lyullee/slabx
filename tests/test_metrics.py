"""Tier-0 verification of the validation metrics."""

import numpy as np
import pytest

from slabx.validation.metrics import (
    ACCEPTANCE,
    check_acceptance,
    compare,
    metrics,
    vg_min,
)


# ===========================================================================
# analytic limits
# ===========================================================================
def test_perfect_model():
    o = np.array([1.0, 10.0, 100.0, 3.3])
    m = metrics(o, o)
    assert m.FB == pytest.approx(0.0)
    assert m.MG == pytest.approx(1.0)
    assert m.NMSE == pytest.approx(0.0)
    assert m.VG == pytest.approx(1.0)
    assert m.FAC2 == pytest.approx(1.0)
    assert m.FAC5 == pytest.approx(1.0)
    assert m.MRB == pytest.approx(0.0)
    assert m.MRSE == pytest.approx(0.0)
    assert m.R == pytest.approx(1.0)
    assert check_acceptance(m)["all"]


@pytest.mark.parametrize("f", [0.5, 2.0, 4.0])
def test_uniform_scaling(f):
    """pred = f*obs everywhere => MG = 1/f exactly, VG = (ln f)^2 exponent."""
    o = np.array([1.0, 5.0, 50.0])
    m = metrics(o, f * o)
    assert m.MG == pytest.approx(1.0 / f)
    assert m.VG == pytest.approx(np.exp(np.log(f) ** 2))
    assert m.VG == pytest.approx(vg_min(m.MG))       # pure bias, no scatter
    assert m.R == pytest.approx(1.0)


def test_under_prediction_gives_MG_above_one():
    """Sign convention: model too low => MG > 1 and FB > 0."""
    o = np.array([10.0, 20.0])
    m = metrics(o, 0.5 * o)
    assert m.MG > 1.0
    assert m.FB > 0.0


def test_fac2_boundaries_are_inclusive():
    o = np.array([1.0, 1.0, 1.0, 1.0])
    p = np.array([0.5, 2.0, 0.499, 2.001])
    assert metrics(o, p).FAC2 == pytest.approx(0.5)


def test_vg_min_parabola_is_a_lower_bound():
    """No sample may fall below VG_min(MG)."""
    rng = np.random.default_rng(3)
    for _ in range(200):
        o = np.exp(rng.normal(0, 1.5, 40))
        p = o * np.exp(rng.normal(rng.normal(0, 0.5), 0.8, 40))
        m = metrics(o, p)
        assert m.VG >= vg_min(m.MG) - 1e-9


def test_cancelling_errors_fool_the_log_measures():
    """
    The documented failure mode: equal over- and under-prediction in the log
    sense gives a perfect MG (and MRB) while VG, NMSE and FAC2 correctly
    report the scatter.  FB is arithmetic and so does not cancel here — which
    is exactly why both a log and a linear bias measure must be reported.
    """
    o = np.array([10.0, 10.0])
    m = metrics(o, np.array([40.0, 2.5]))       # x4 and x1/4
    assert m.MG == pytest.approx(1.0)           # fooled
    assert m.MRB == pytest.approx(0.0)          # fooled
    assert m.VG > 5.0                           # not fooled
    assert m.NMSE > 1.0                         # not fooled
    assert m.FAC2 == 0.0                        # not fooled
    assert not check_acceptance(m)["all"]


# ===========================================================================
# screening
# ===========================================================================
def test_floor_excludes_and_counts():
    o = np.array([1.0, 0.0, 5.0, -1.0])
    p = np.array([1.0, 3.0, 5.0, 2.0])
    m = metrics(o, p, floor=1e-9)
    assert m.n == 2
    assert m.n_excluded == 2
    assert m.MG == pytest.approx(1.0)


def test_nan_is_screened():
    o = np.array([1.0, np.nan, 4.0])
    p = np.array([1.0, 2.0, 4.0])
    assert metrics(o, p).n == 2


def test_all_screened_raises():
    with pytest.raises(ValueError):
        metrics([1.0, 2.0], [1.0, 2.0], floor=1e6)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        metrics([1.0, 2.0], [1.0])


def test_floor_changes_result():
    """Sensitivity to the screening threshold must be visible, not hidden."""
    o = np.array([1e-4, 1.0, 10.0])
    p = np.array([1.0, 1.0, 10.0])
    assert metrics(o, p, floor=1e-9).MG < metrics(o, p, floor=1e-2).MG


# ===========================================================================
# acceptance criteria
# ===========================================================================
def test_criteria_tables_match_publications():
    ch = ACCEPTANCE["chang_hanna_2004"]
    assert ch["FB"] == (-0.3, 0.3)
    assert ch["MG"] == (0.7, 1.3)
    assert ch["NMSE"] == (None, 4.0)
    assert ch["VG"] == (None, 1.6)
    assert ch["FAC2"] == (0.5, None)
    urban = ACCEPTANCE["hanna_chang_2012_urban"]
    assert urban["FAC2"] == (0.3, None)


def test_urban_criteria_are_looser():
    """A model that fails the rural test may pass the urban one."""
    rng = np.random.default_rng(11)
    o = np.exp(rng.normal(0, 1.0, 400))
    p = o * np.exp(rng.normal(0.0, 1.1, 400))
    m = metrics(o, p)
    assert not check_acceptance(m, "chang_hanna_2004")["all"]
    assert check_acceptance(m, "hanna_chang_2012_urban")["all"]


def test_custom_criteria_accepted():
    m = metrics([1.0, 2.0], [1.0, 2.0])
    assert check_acceptance(m, {"FAC2": (0.9, None)})["all"]


# ===========================================================================
# paired bootstrap
# ===========================================================================
def _synthetic(n_trials=25, per_trial=4, bias_a=1.6, bias_b=1.15, seed=0):
    """
    Two variants sharing a large common error (trial-level), differing only
    in a modest systematic bias.  This is the structure of the real problem.
    """
    rng = np.random.default_rng(seed)
    obs, pa, pb, tid = [], [], [], []
    for t in range(n_trials):
        truth = np.exp(rng.normal(0, 1.2, per_trial))
        common = np.exp(rng.normal(0, 0.55))          # shared source-term error
        obs.append(truth)
        pa.append(truth * common / bias_a * np.exp(rng.normal(0, 0.10, per_trial)))
        pb.append(truth * common / bias_b * np.exp(rng.normal(0, 0.10, per_trial)))
        tid.append(np.full(per_trial, t))
    return (np.concatenate(obs), np.concatenate(pa),
            np.concatenate(pb), np.concatenate(tid))


def test_paired_bootstrap_detects_a_real_improvement():
    o, a, b, t = _synthetic()
    c = compare(o, a, b, trial=t, label_a="S0", label_b="S3", n_boot=2000)
    assert c.n_trials == 25
    assert c.delta["MG"] < 0                  # B is closer to MG = 1
    assert c.significant["MG"]
    assert c.metrics_b.FAC2 > c.metrics_a.FAC2
    assert "S0" in c.summary() and "S3" in c.summary()


def test_paired_bootstrap_reports_no_difference_when_there_is_none():
    o, a, _, t = _synthetic()
    c = compare(o, a, a, trial=t, n_boot=2000)
    for k in ("FAC2", "MG", "VG", "FB", "NMSE"):
        assert c.delta[k] == pytest.approx(0.0, abs=1e-12)
        assert not c.significant[k]


def test_common_error_cancels_in_the_difference():
    """
    The point of the design: absolute skill is poorly determined while the
    difference is well determined.  The CI on the difference must be much
    narrower than the spread of the absolute metric.
    """
    o, a, b, t = _synthetic(seed=5)
    c = compare(o, a, b, trial=t, n_boot=3000)
    width_delta = c.ci["MG"][1] - c.ci["MG"][0]

    rng = np.random.default_rng(1)
    groups = [np.flatnonzero(t == k) for k in np.unique(t)]
    abs_mg = []
    for _ in range(3000):
        pick = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        abs_mg.append(metrics(o[pick], a[pick]).MG)
    width_abs = np.percentile(abs_mg, 97.5) - np.percentile(abs_mg, 2.5)

    assert width_delta < 0.5 * width_abs


def test_trial_level_resampling_is_wider_than_point_level():
    """Ignoring within-trial correlation must not be allowed to look better."""
    o, a, b, t = _synthetic(per_trial=8, seed=7)
    wide = compare(o, a, b, trial=t, n_boot=2000)
    narrow = compare(o, a, b, trial=None, n_boot=2000)
    w = lambda c: c.ci["MG"][1] - c.ci["MG"][0]
    assert w(wide) > w(narrow)


def test_bootstrap_is_reproducible():
    o, a, b, t = _synthetic()
    kw = dict(trial=t, n_boot=500)
    assert compare(o, a, b, seed=42, **kw).ci == compare(o, a, b, seed=42, **kw).ci


def test_compare_rejects_length_mismatch():
    with pytest.raises(ValueError):
        compare([1.0, 2.0], [1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        compare([1.0, 2.0], [1.0, 2.0], [1.0, 2.0], trial=[1])
