"""
The height-closure hypothesis, and its falsification.

`core/height_closure.py` registered four predictions before the comparison
that tests them.  All four failed.  These tests pin both halves of that: the
alternative closure is implemented and sound, and it does not do what was
predicted.

Keeping a falsified hypothesis under test is deliberate.  The closure is a
supported option, someone will be tempted to try it again, and the result
should be one command away rather than one re-derivation away.
"""

import numpy as np
import pytest

from slabx.coefficients import preset
from slabx.core.height_closure import HEIGHT_CLOSURES, PREDICTIONS
from slabx.core.plume import run_dispersion
from slabx.core.source import EvaporatingPool
from slabx.post.concentration import concentration_field
from slabx.thermo.base import LegacyThermo, water_backend
from slabx.validation.field_trials import (
    BURRO_AREA, BURRO_TRIALS, METHANE, VARIANTS,
    load_conditions, load_observations,
)
from slabx.validation.metrics import metrics


def _run(trial, cond, closure):
    atm = VARIANTS["slabx"].atmosphere(cond)
    src = EvaporatingPool(substance=METHANE, rate=cond["rate"],
                          area=BURRO_AREA[trial], duration=cond["duration"])
    traj, _ = run_dispersion(src, atm, LegacyThermo(METHANE), water_backend(),
                             x_max=1000.0, n_puff_steps=40,
                             coeffs=preset("ermak90"), height_closure=closure)
    field = concentration_field(traj, atm, z=1.0, t_avg=cond["t_avg"],
                                t_release=cond["duration"])
    return traj, field


@pytest.fixture(scope="module")
def burro():
    obs = load_observations()
    cond = load_conditions()
    out = {}
    for closure in ("algebraic", "constrained"):
        runs = {t: _run(t, cond[t], closure) for t in BURRO_TRIALS}
        pred = np.array([
            float(np.exp(np.interp(o["x"], runs[o["trial"]][1].x,
                                   np.log(np.maximum(runs[o["trial"]][1].peak,
                                                     1e-30)))))
            for o in obs])
        out[closure] = (runs, pred)
    y = np.array([o["c_obs"] for o in obs])
    tid = np.array([o["trial"] for o in obs])
    return out, y, tid


# ===========================================================================
# the registration itself
# ===========================================================================
def test_predictions_are_recorded_with_pass_criteria():
    assert len(PREDICTIONS) >= 4
    for p in PREDICTIONS:
        assert p.statement.strip() and p.passes_if.strip()
    assert set(HEIGHT_CLOSURES) >= {"algebraic", "constrained"}


# ===========================================================================
# the implementation is sound, so the failure is about the physics
# ===========================================================================
def test_both_closures_conserve_mass(burro):
    """
    Without this the negative result would be meaningless — a broken
    alternative would fail the predictions for the wrong reason.
    """
    out, _, _ = burro
    for closure in ("algebraic", "constrained"):
        traj, _ = out[closure][0]["BU07"]
        p = traj.is_plume & (traj.R_flux > 0)
        err = np.abs(traj.R_flux[p]
                     / (traj.rho[p] * traj.u[p] * traj.b_half[p] * traj.h[p])
                     - 1.0)
        assert err.max() < 5e-3, closure


def test_the_alternative_closure_actually_changes_the_partition(burro):
    """A taller cloud with a narrower width — otherwise nothing was tested."""
    out, _, _ = burro
    a = out["algebraic"][0]["BU07"][0]
    c = out["constrained"][0]["BU07"][0]
    for x in (50.0, 200.0, 800.0):
        i = int(np.argmin(np.abs(a.x - x)))
        j = int(np.argmin(np.abs(c.x - x)))
        assert c.h[j] > a.h[i]
        assert c.b_half[j] < a.b_half[i] * 1.05


# ===========================================================================
# the predictions, as registered
# ===========================================================================
def test_P1_bias_does_not_move_towards_the_integral_models(burro):
    out, y, _ = burro
    mg = metrics(y, out["constrained"][1]).MG
    assert not (0.85 <= mg <= 1.15)
    assert mg < metrics(y, out["algebraic"][1]).MG      # it gets worse


def test_P1b_the_cross_section_is_not_preserved(burro):
    """
    The premise that failed.  EQ 2a entrains through ``V_e h + W_e B``, which
    depends on the two lengths separately, so changing the partition changes
    the dilution and ``B h`` is not invariant.
    """
    out, _, _ = burro
    a = out["algebraic"][0]["BU07"][0]
    c = out["constrained"][0]["BU07"][0]
    i = int(np.argmin(np.abs(a.x - 200.0)))
    j = int(np.argmin(np.abs(c.x - 200.0)))
    assert abs(c.b_half[j] * c.h[j] / (a.b_half[i] * a.h[i]) - 1.0) > 0.05


def test_P3_the_effect_lands_on_the_wrong_trials(burro):
    """
    The discriminating prediction: the effect was to concentrate in the
    neutral, windy trials where the width error is largest.  It does the
    opposite, landing hardest on the stable trial that was already unbiased.
    """
    out, y, tid = burro
    eff = {}
    for t in BURRO_TRIALS:
        m = tid == t
        a = metrics(y[m], out["algebraic"][1][m]).MG
        c = metrics(y[m], out["constrained"][1][m]).MG
        eff[t] = abs(c / a - 1.0)
    neutral = np.mean([eff["BU03"], eff["BU07"], eff["BU09"]])
    assert eff["BU08"] > 2.0 * neutral


# ===========================================================================
# the profile shape, eliminated without changing anything
# ===========================================================================
def test_neutral_trials_are_a_consistent_gaussian_plume(burro):
    """
    Second structural candidate, ruled out.

    If the flat-topped profile or the rigid ``sigma_z = h/sqrt3`` were
    responsible for the over-prediction, SLAB's reported centreline value
    would not agree with what its *own* spreads imply for a plain Gaussian
    plume.  In the neutral trials it agrees to within a quarter, so the
    profile and the normalisation are not the problem there — the spreads
    are.

    The stable trial is different, and correctly so: that is the one where
    the cloud really is a flat dense slab, and it is also the one the model
    already gets right.
    """
    import math

    out, _, _ = burro
    cond = load_conditions()
    obs = load_observations()

    ratios = {"neutral": [], "stable": []}
    for t in BURRO_TRIALS:
        traj, field = out["algebraic"][0][t]
        atm = VARIANTS["slabx"].atmosphere(cond[t])
        q = cond[t]["rate"]
        for o in obs:
            if o["trial"] != t:
                continue
            i = int(np.argmin(np.abs(traj.x - o["x"])))
            sy = math.sqrt(traj.b_shape[i] ** 2 / 3.0 + traj.beta[i] ** 2)
            sz = traj.sigma_z[i]
            cm = q / (math.pi * traj.u[i] * sy * sz * atm.rho)
            cv = (atm.mw_moist * cm
                  / (METHANE.mw + (atm.mw_moist - METHANE.mw) * cm))
            cs = float(np.exp(np.interp(o["x"], field.x,
                                        np.log(np.maximum(field.peak, 1e-30)))))
            ratios["stable" if t == "BU08" else "neutral"].append(cs / cv)

    neutral = np.array(ratios["neutral"])
    stable = np.array(ratios["stable"])
    assert 0.7 < np.median(neutral) < 1.15      # a Gaussian plume, in effect
    assert np.median(stable) < 0.6              # the dense slab is not
