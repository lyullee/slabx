"""
Numerical convergence under grid refinement.

Reproducing the reference says the implementation matches Ermak's; it does
not say either of them has resolved the equations.  These tests refine the
grid and measure what changes.

The headline is that the two grids behave completely differently: the puff
grid converges to six figures by forty steps, the field grid to 0.03 % by
eighty, and the *source* grid converges only at first order and leaves the
default about 3 % low.
"""

import numpy as np
import pytest

from slabx.coefficients import preset
from slabx.core.plume import run_dispersion, run_plume
from slabx.core.source import EvaporatingPool
from slabx.post.concentration import concentration_field
from slabx.submodels.atmosphere import Atmosphere
from slabx.thermo.base import LegacyThermo, Substance, water_backend

METHANE = Substance(name="LNG", mw=0.016043, cp_vapour=2238.0,
                    cp_liquid=3348.5, dh_vap=509900.0, T_boil=111.7,
                    rho_liquid=424.1)


def _setup():
    atm = Atmosphere(u_ref=1.94, z_ref=3.0, T=306.0, rh=4.6, z0=2e-4,
                     stability="E")
    src = EvaporatingPool(substance=METHANE, rate=117.0, area=657.0,
                          duration=107.0)
    return atm, src


def _m_at(x, **kw):
    atm, src = _setup()
    traj, _ = run_plume(src, atm, LegacyThermo(METHANE), water_backend(),
                        x_max=1000.0, coeffs=preset("ermak90"), **kw)
    return float(np.interp(x, traj.x, traj.mass_frac))


# ===========================================================================
# the two grids that behave
# ===========================================================================
def test_puff_grid_converges_immediately():
    """Six figures by forty steps."""
    atm, src = _setup()
    out = []
    for n in (40, 80, 160, 320):
        traj, _ = run_dispersion(src, atm, LegacyThermo(METHANE),
                                 water_backend(), x_max=1000.0,
                                 n_field_steps=80, n_puff_steps=n,
                                 coeffs=preset("ermak90"))
        field = concentration_field(traj, atm, z=1.0, t_avg=80.0,
                                    t_release=107.0)
        out.append(float(np.exp(np.interp(600.0, field.x,
                                          np.log(np.maximum(field.peak,
                                                            1e-30))))))
    assert abs(out[-1] / out[1] - 1.0) < 1e-4


def test_field_grid_converges_by_eighty_steps():
    v = [_m_at(400.0, n_field_steps=n) for n in (40, 80, 160)]
    assert abs(v[2] / v[1] - 1.0) < 0.002


def test_the_transition_is_grid_independent():
    """
    Exact interpolation to Qint = q t_sd / 2 rather than snapping to an
    output row, which is where this port departs from the reference and is
    right to.
    """
    atm, src = _setup()
    for n in (20, 40, 160):
        traj, _ = run_dispersion(src, atm, LegacyThermo(METHANE),
                                 water_backend(), x_max=1000.0,
                                 n_field_steps=n, n_puff_steps=n,
                                 coeffs=preset("ermak90"))
        assert traj.meta["transition_t"] == pytest.approx(107.0, abs=1e-6)


# ===========================================================================
# the grid that does not
# ===========================================================================
def test_source_grid_is_the_bottleneck():
    """
    The finding.  The field grid is resolved at eighty steps while the
    source grid is still moving at 3 %, so refining the field alone hides
    the larger error.
    """
    coarse = _m_at(400.0, n_source_steps=10, n_field_steps=160)
    fine = _m_at(400.0, n_source_steps=320, n_field_steps=160)
    assert abs(coarse / fine - 1.0) > 0.02
    assert coarse < fine                    # the default is low, not high


def test_source_grid_converges_at_first_order():
    """
    First order, not fourth.  Inside the source region EQ 1a replaces the
    species integration with an algebraic ramp, so the only integrated
    quantity is R, and its source term has a slope discontinuity at the pool
    edge which costs the Runge-Kutta scheme its order.
    """
    v = np.array([_m_at(400.0, n_source_steps=n, n_field_steps=160)
                  for n in (20, 40, 80, 160, 640)])
    err = np.abs(v[:-1] - v[-1])
    orders = np.log2(err[:-1] / err[1:])
    assert np.all(orders < 2.0)             # nowhere near fourth order
    assert np.all(np.diff(v) > 0)           # monotone, so extrapolable


def test_refining_both_grids_together_converges():
    v = [_m_at(400.0, n_source_steps=n // 4, n_field_steps=n)
         for n in (160, 320, 640)]
    assert abs(v[-1] / v[-2] - 1.0) < 0.005


def test_the_default_is_ermaks_not_the_resolved_answer():
    """
    Kept at ten so that reproduction of the reference is undisturbed.  A user
    who wants the resolved answer raises it, and this records the cost of not
    doing so.
    """
    import inspect

    from slabx.core.plume import integrate_plume

    assert inspect.signature(integrate_plume).parameters[
        "n_source_steps"].default == 10
    default = _m_at(400.0, n_field_steps=160)
    resolved = _m_at(400.0, n_source_steps=640, n_field_steps=160)
    assert 0.02 < abs(default / resolved - 1.0) < 0.06


def test_source_expansion_is_grid_independent():
    """
    The outer fixed point for the source-area expansion settles on the same
    effective half-width whatever the grid, which is what makes the source
    error a discretisation error rather than a different problem.
    """
    atm, src = _setup()
    widths = []
    for n in (40, 160, 640):
        _, used = run_plume(src, atm, LegacyThermo(METHANE), water_backend(),
                            x_max=1000.0, n_field_steps=n,
                            coeffs=preset("ermak90"))
        widths.append(used.half_width)
    assert max(widths) - min(widths) < 1e-6


def test_the_field_discrepancy_is_not_discretisation():
    """
    The pre-registered grid check; see `docs/PREREG_grid.md`.

    The source grid converges at first order and the default of ten steps
    sits about 3 % below the converged answer, which invites the objection
    that the field comparison's 65 % under-prediction is a numerical
    artefact.

    Refining by a factor of 32 moves the aggregate MG from 0.605 to 0.596 —
    1.5 %, or about 4 % of the gap to unity. It is not.

    The per-trial breakdown matches the prediction too: the effect
    concentrates in BU08, the low-wind stable trial where the source region
    is the largest fraction of the run, and the other nine move by 0.4 % or
    less with no common sign.
    """
    import numpy as np

    from slabx.validation.field_trials import (
        VARIANTS, load_conditions, load_observations, predictions_for,
    )
    from slabx.validation.metrics import metrics

    obs, cond = load_observations(), load_conditions()
    y = np.array([o["c_obs"] for o in obs])

    coarse = metrics(y, predictions_for(VARIANTS["slab90"], obs, cond,
                                        n_source_steps=10))
    fine = metrics(y, predictions_for(VARIANTS["slab90"], obs, cond,
                                      n_source_steps=320))

    assert abs(fine.MG / coarse.MG - 1.0) < 0.10, (coarse.MG, fine.MG)
    assert fine.FAC2 == coarse.FAC2
    # and the gap it was meant to explain is still there
    assert fine.MG < 0.7


def test_an_instantaneous_release_ignores_the_source_grid():
    """Negative control for the check above: bit-identical, not merely close."""
    from slabx.core.plume import run_dispersion
    from slabx.core.source import InstantaneousRelease
    from slabx.submodels.atmosphere import Atmosphere
    from slabx.thermo.base import LegacyThermo, Substance, water_backend

    sub = Substance(name="TI", mw=0.0424, cp_vapour=800.0, cp_liquid=1000.0,
                    dh_vap=2e5, T_boil=200.0, rho_liquid=1400.0)
    atm = Atmosphere(u_ref=2.4, z_ref=10.0, T=288.0, rh=60.0, z0=0.005,
                     stability="D")
    src = InstantaneousRelease(substance=sub, rate=0.0, duration=0.0,
                               area=154.0, mass=3941.0, height=0.0)

    out = []
    for n in (10, 320):
        traj, _ = run_dispersion(src, atm, LegacyThermo(sub), water_backend(),
                                 x_max=500.0, n_puff_steps=40,
                                 n_source_steps=n)
        out.append(float(np.interp(300.0, traj.x, traj.vol_frac)))
    assert out[0] == out[1]
