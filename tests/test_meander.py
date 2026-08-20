"""
Plume meander and the peak-to-mean ratio.

SLAB's EQ 50b scales its own trace-gas plume spread by an averaging-time
factor.  The alternative closure here uses the measured wind-direction
fluctuation instead, on the reasoning that a plume meanders because the wind
does.  Both are validated against peak-to-mean ratios extracted from PHMSA's
assessment of Phast 8.4.
"""

import math

import numpy as np
import pytest

from slabx.coefficients import preset
from slabx.core.plume import run_dispersion
from slabx.post.concentration import (
    SIGMA_THETA, concentration_field, meander_width, sigma_theta,
)
from slabx.thermo.base import LegacyThermo, water_backend
from slabx.validation.field_trials import METHANE
from slabx.validation.lng_pools import load_trials

#: Measured peak-to-mean, from PHMSA FEA Table 3 (see `peak_to_mean.csv`).
OBSERVED = {"BU03": 1.52, "BU08": 1.21, "BU09": 1.35,
            "CO3": 2.43, "CO5": 2.39, "CO6": 1.17}


def _peak_to_mean(trial, closure):
    atm, src = trial.atmosphere(), trial.source()
    traj, _ = run_dispersion(src, atm, LegacyThermo(METHANE), water_backend(),
                             x_max=2000.0, n_puff_steps=40,
                             coeffs=preset("ermak90"))
    out = []
    for t_avg in (1.0, trial.t_avg):
        f = concentration_field(traj, atm, z=1.0, t_avg=t_avg,
                                t_release=trial.duration,
                                meander_closure=closure)
        out.append(float(np.exp(np.interp(200.0, f.x,
                                          np.log(np.maximum(f.peak, 1e-30))))))
    return out[0] / out[1]


@pytest.fixture(scope="module")
def ratios():
    trials = load_trials()
    return {c: {t: _peak_to_mean(trials[t], c) for t in OBSERVED}
            for c in ("ermak", "sigma_theta")}


# ===========================================================================
# the direction fluctuation
# ===========================================================================
def test_sigma_theta_spans_a_factor_of_ten():
    """A to F, from wind-vane records — not from dispersion statistics."""
    assert SIGMA_THETA[1] / SIGMA_THETA[6] == 10.0
    assert sigma_theta(4.0) == pytest.approx(math.radians(10.0))
    assert sigma_theta(3.5) == pytest.approx(math.radians(12.5))
    assert sigma_theta(0.0) == sigma_theta(1.0)          # clamped
    assert sigma_theta(9.0) == sigma_theta(6.0)


def test_unknown_closure_is_rejected():
    trials = load_trials()
    t = trials["BU08"]
    atm, src = t.atmosphere(), t.source()
    traj, _ = run_dispersion(src, atm, LegacyThermo(METHANE), water_backend(),
                             x_max=500.0, n_puff_steps=20,
                             coeffs=preset("ermak90"))
    with pytest.raises(ValueError, match="unknown meander closure"):
        meander_width(traj, atm, t_avg=60.0, t_release=t.duration,
                      closure="nonsense")


def test_both_closures_vanish_at_zero_averaging():
    """The consistency check the reference leaves implicit."""
    trials = load_trials()
    t = trials["BU08"]
    atm, src = t.atmosphere(), t.source()
    traj, _ = run_dispersion(src, atm, LegacyThermo(METHANE), water_backend(),
                             x_max=500.0, n_puff_steps=20,
                             coeffs=preset("ermak90"))
    for closure in ("ermak", "sigma_theta"):
        B, beta = meander_width(traj, atm, t_avg=0.0, t_release=t.duration,
                                closure=closure)
        assert np.allclose(beta, traj.beta)
        assert np.allclose(B, traj.b_half)


# ===========================================================================
# against the measured peak-to-mean
# ===========================================================================
def test_ermak_compresses_the_variation(ratios):
    """
    The finding.  The measured peak-to-mean spans a factor 2.08 across these
    six trials, ordered by stability class; EQ 50b spans 1.17 — it has the
    right sign and almost none of the range.
    """
    v = np.array(list(ratios["ermak"].values()))
    assert v.max() / v.min() < 1.3
    obs = np.array(list(OBSERVED.values()))
    assert obs.max() / obs.min() > 2.0


def test_sigma_theta_reproduces_the_dynamic_range(ratios):
    """
    And overshoots it.  For a fluctuation model the range is the property
    that matters, but the bias goes the other way — 1.25 against Ermak's
    0.69 — and six trials cannot settle which is better overall.
    """
    v = np.array([ratios["sigma_theta"][t] for t in OBSERVED])
    o = np.array([OBSERVED[t] for t in OBSERVED])
    assert v.max() / v.min() > 2.0
    bias = float(np.exp(np.mean(np.log(v / o))))
    assert 1.0 < bias < 1.6


def test_both_closures_follow_the_stability_ordering(ratios):
    """
    Weakly, and equally: both correlate at about 0.75 in the log, which at
    n = 6 is not significant.  Pinned so that a future closure has something
    to beat.
    """
    o = np.array([OBSERVED[t] for t in OBSERVED])
    for closure in ("ermak", "sigma_theta"):
        v = np.array([ratios[closure][t] for t in OBSERVED])
        r = np.corrcoef(np.log(o), np.log(v))[0, 1]
        assert r > 0.6, closure


def test_the_stable_trial_fluctuates_least(ratios):
    """Physically the whole point: convection makes the plume wander."""
    assert OBSERVED["BU08"] < OBSERVED["CO3"]           # class E below C
    assert ratios["sigma_theta"]["BU08"] < ratios["sigma_theta"]["CO3"]
