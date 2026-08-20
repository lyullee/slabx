"""
Thorney Island: the instantaneous source type against field data.

Three of the four source types had been compared with measurements; the
instantaneous release had not.  These fifteen trials close that gap, and the
result is a specific disagreement rather than a pass or a fail.
"""

import math

import numpy as np
import pytest

from slabx.coefficients import PHYS
from slabx.validation.thorney_island import (
    CONTAINER_RADIUS, load_trials, peak_concentration, sensitivity_to,
)


@pytest.fixture(scope="module")
def sens():
    return {d: sensitivity_to(d) for d in (100.0, 300.0)}


# ===========================================================================
# the trials
# ===========================================================================
def test_fifteen_usable_trials_spanning_the_conditions():
    t = load_trials()
    assert len(t) == 15
    assert min(x.u10 for x in t) < 2.0 and max(x.u10 for x in t) > 7.0
    assert min(x.rel_density for x in t) < 1.0     # trial 004, neutral
    assert max(x.rel_density for x in t) > 4.0     # trial 017, pure R-12
    assert {x.stability_class for x in t} >= {"B", "C", "D", "E"}


def test_the_unusable_trial_is_excluded():
    """
    Trial 005: the bag hung up during the drop and gas escaped, so the
    released mass is unknown.
    """
    assert "005" not in {t.trial for t in load_trials()}
    assert "005" in {t.trial for t in load_trials(usable_only=False)}


def test_the_mixture_is_built_from_its_relative_density():
    t = next(x for x in load_trials() if x.trial == "008")
    assert t.substance().mw == pytest.approx(1.63 * PHYS.MW_AIR)
    atm = t.atmosphere()
    src = t.source(atm)
    assert src.area == pytest.approx(math.pi * CONTAINER_RADIUS**2)
    assert src.mass == pytest.approx(t.volume * t.rel_density * atm.rho)


# ===========================================================================
# the model runs on every one of them
# ===========================================================================
def test_every_trial_runs_and_decays():
    for t in load_trials():
        c = peak_concentration(t, (50.0, 150.0, 400.0))
        assert np.all(np.isfinite(c))
        assert np.all((c > 0) & (c < 1))
        assert c[0] > c[1] > c[2], t.trial


# ===========================================================================
# the report's statement
# ===========================================================================
def test_wind_insensitivity_is_reproduced(sens):
    """
    McQuaid & Roebuck: the peak concentration at a given distance is "quite
    insensitive to wind speed".  It is — the correlation across fifteen
    trials is weak and not significant.
    """
    for d in (100.0, 300.0):
        assert abs(sens[d]["wind"]) < 0.4


def test_stability_dependence_is_not(sens):
    """
    The same sentence says insensitive to "atmospheric stability", and here
    the model disagrees: the correlation is 0.87 at 300 m, far stronger than
    the wind correlation it does reproduce.

    Recorded as a failure to match a stated experimental result, not as a
    bug — SLAB applies its ambient-stability damping in full even where the
    cloud's own gravity dominates the mixing.
    """
    for d in (100.0, 300.0):
        assert sens[d]["stability"] > 2.0 * abs(sens[d]["wind"])
    assert sens[300.0]["stability"] > 0.7


def test_the_stability_effect_isolated_is_sixfold():
    """
    With every other input held fixed, moving only the class from B to F.
    The trial set does not confound this: stability correlates with wind at
    -0.14 and with density at +0.38, neither significant.
    """
    t = next(x for x in load_trials() if x.trial == "008")
    c = {s: peak_concentration(t, (300.0,), stability=s)[0]
         for s in ("B", "D", "F")}
    assert c["F"] / c["B"] > 4.0
    assert c["D"] > c["B"]


def test_stability_is_not_confounded_in_the_trial_set():
    trials = load_trials()
    cls = np.array([{"B": 2, "C": 3, "D": 4, "E": 5}[t.stability_class]
                    for t in trials], dtype=float)
    u = np.array([t.u10 for t in trials])
    d = np.array([t.rel_density for t in trials])
    assert abs(float(np.corrcoef(cls, u)[0, 1])) < 0.4
    assert abs(float(np.corrcoef(cls, d)[0, 1])) < 0.5


def test_density_dependence_is_expected_and_present(sens):
    """
    Unlike wind and stability, the initial density *should* matter — the
    report attributes the dispersion to gravity-driven motion, which is
    driven by exactly that.
    """
    for d in (100.0, 300.0):
        assert sens[d]["density"] > 0.4


def test_the_ambient_stratification_term_carries_the_stability_response():
    """
    The intervention that turns §4.3's correlation into a mechanism.

    EQ 35d builds the in-cloud Monin-Obukhov length as a sum of two
    stratification sources — the ambient one and the cloud's own density
    excess — and `w_ambient_stratification` weights only the first. Setting
    it to zero cuts one link of

        class -> 1/L_ambient -> 1/L_cloud -> phi_h -> W_e -> concentration

    while leaving the cloud's own damping intact.

    The isolated B-to-F response falls from 6.17 to 1.22, and halfway
    weighting gives 3.51 — a dose-response, not an on/off effect. The
    neutral class is unchanged to four decimals at every weight, because it
    has no ambient stratification to remove; the negative control is built
    into the sweep.

    What this does *not* establish is that the same pathway explains the
    correlation across the fifteen trials: the ambient share of `1/L_cloud`
    is bimodal, zero for the nine neutral trials. See
    `docs/PREREG_stability.md`.
    """
    from slabx.coefficients import COEFFS

    t = next(x for x in load_trials() if x.trial == "008")

    def sweep(w):
        c = COEFFS.perturb(w_ambient_stratification=w, name="w")
        return {s: peak_concentration(t, (300.0,), stability=s, coeffs=c)[0]
                for s in "BDF"}

    full, half, none = sweep(1.0), sweep(0.5), sweep(0.0)

    # the response collapses when the link is cut
    assert full["F"] / full["B"] > 4.0
    assert none["F"] / none["B"] < 2.0

    # and responds in proportion to how much of the link is left
    r_full = full["F"] / full["B"]
    r_half = half["F"] / half["B"]
    r_none = none["F"] / none["B"]
    assert r_full > r_half > r_none

    # negative control: a neutral class has nothing to remove
    assert full["D"] == pytest.approx(none["D"], rel=1e-6)
    assert half["D"] == pytest.approx(none["D"], rel=1e-6)

    # the switch is diagnostic only
    assert COEFFS.w_ambient_stratification == 1.0
