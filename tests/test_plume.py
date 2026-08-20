"""
Tier-0/1 verification of the plume integrator.

Conservation, limits and internal consistency.  Comparison against the
manual's printed table is deliberately confined to the plume region — see
`test_manual_table_switches_to_puff_mode` for why.
"""

import math

import numpy as np
import pytest

from slabx.core.plume import (
    run_dispersion,
    SourceExpansionRequired,
    _solve_velocity_cubic,
    integrate_plume,
    run_plume,
)
from slabx.core.source import EvaporatingPool
from slabx.core.trajectory import Mode
from slabx.coefficients import COEFFS
from slabx.submodels.atmosphere import Atmosphere
from slabx.thermo.base import LegacyThermo, Substance, water_backend

METHANE = Substance(
    name="LNG", mw=0.016043, cp_vapour=2238.0, cp_liquid=3348.5,
    dh_vap=509900.0, T_boil=111.7, rho_liquid=424.1,
)


def burro8(x_max=200.0, **kw):
    atm = Atmosphere(u_ref=1.92, z_ref=2.88, T=306.0, rh=4.6,
                     z0=2e-4, inv_L=0.0665)
    src = EvaporatingPool(substance=METHANE, rate=117.0, area=657.0,
                          duration=107.0)
    tr, used = run_plume(src, atm, LegacyThermo(METHANE), water_backend(),
                         x_max=x_max, **kw)
    return tr, used, atm, src


@pytest.fixture(scope="module")
def run():
    return burro8()


# ===========================================================================
# the velocity cubic, EQ 4b-4e
# ===========================================================================
def test_cubic_root_satisfies_the_equation():
    u_e, u_g3 = 2.0, 0.05
    u, needs, _ = _solve_velocity_cubic(u_e, u_g3, 1.0, COEFFS)
    assert not needs
    assert u**3 - u_e * u**2 + u_g3 == pytest.approx(0.0, abs=1e-10)


def test_cubic_returns_the_largest_root():
    """EQ 4e: the physical branch approaches U_e as gravity flow vanishes."""
    u_e = 2.0
    us = [_solve_velocity_cubic(u_e, g, 1.0, COEFFS)[0]
          for g in (1e-6, 0.01, 0.05, 0.1)]
    assert us[0] == pytest.approx(u_e, rel=1e-3)
    assert all(b < a for a, b in zip(us, us[1:]))
    assert all(u > u_e / 2 for u in us)


def test_cubic_no_gravity_flow_gives_the_ambient_velocity():
    assert _solve_velocity_cubic(3.0, 0.0, 1.0, COEFFS)[0] == pytest.approx(3.0)
    assert _solve_velocity_cubic(3.0, -1.0, 1.0, COEFFS)[0] == pytest.approx(3.0)


def test_cubic_detects_the_no_solution_regime():
    """EQ 4d: no positive real root once U_g^3 > (4/27) U_e^3."""
    u_e = 2.0
    limit = (4.0 / 27.0) * u_e**3
    assert not _solve_velocity_cubic(u_e, 0.99 * limit, 1.0, COEFFS)[1]
    u, needs, ratio = _solve_velocity_cubic(u_e, 1.5 * limit, 1.0, COEFFS)
    assert needs and ratio == pytest.approx(1.5, rel=1e-9)


# ===========================================================================
# the run completes and is well formed
# ===========================================================================
def test_produces_a_valid_trajectory(run):
    tr, *_ = run
    assert len(tr) > 20
    assert np.all(tr.mode == Mode.PLUME)
    assert np.all(np.isfinite(tr.vol_frac))
    assert tr.meta["source"] == "evaporating_pool"
    assert tr.meta["coefficients"] == "ermak90"


def test_state_is_physical_everywhere(run):
    tr, *_ = run
    # The first row is the upwind edge of the pool, where the cloud does not
    # yet exist: h = 0 and R = 0 by construction (see EvaporatingPool).
    assert tr.h[0] == 0.0 and tr.R_flux[0] == 0.0
    assert np.all(tr.h[1:] > 0) and np.all(tr.b_half > 0)
    assert np.all(tr.u > 0) and np.all(tr.rho > 0)
    assert np.all((tr.mass_frac >= 0) & (tr.mass_frac <= 1))
    assert np.all((tr.vol_frac >= 0) & (tr.vol_frac <= 1))
    assert np.all(tr.T > METHANE.T_boil * 0.9)


def test_shape_relation_holds_along_the_path(run):
    """EQ 13: B^2 = b^2 + 3 beta^2."""
    tr, *_ = run
    assert np.allclose(tr.b_half**2, tr.b_shape**2 + 3.0 * tr.beta**2, rtol=1e-9)


def test_master_variable_is_consistent(run):
    """EQ 2b: h = R / (rho U B) must hold at every stored point."""
    tr, *_ = run
    assert np.allclose(tr.R_flux, tr.rho * tr.u * tr.b_half * tr.h, rtol=1e-6)


def test_concentration_follows_the_analytic_solution(run):
    """EQ 1a: m = q / (2R) beyond the source region."""
    tr, _, _, src = run
    used = tr.meta.get("effective_half_width", None)
    far = tr.x > 60.0
    assert np.allclose(tr.mass_frac[far], src.rate / (2.0 * tr.R_flux[far]),
                       rtol=1e-6)


def test_volume_and_mass_fraction_are_consistent(run):
    """EQ 12."""
    tr, _, atm, _ = run
    mw_a, mw_s = atm.mw_moist, METHANE.mw
    expected = mw_a * tr.mass_frac / (mw_s + (mw_a - mw_s) * tr.mass_frac)
    assert np.allclose(tr.vol_frac, expected, rtol=1e-9)


# ===========================================================================
# monotonicity and limits
# ===========================================================================
def test_cloud_dilutes_and_warms_monotonically_downwind(run):
    tr, *_ = run
    far = tr.x > 0.0
    m, T = tr.mass_frac[far], tr.T[far]
    assert np.all(np.diff(m) < 0)
    assert np.all(np.diff(T) > 0)


def test_cloud_widens_and_accelerates_downwind(run):
    tr, *_ = run
    far = tr.x > 0.0
    assert np.all(np.diff(tr.b_half[far]) > 0)
    assert np.all(np.diff(tr.u[far]) > 0)


def test_cloud_approaches_ambient(run):
    tr, _, atm, _ = run
    assert tr.T[-1] < atm.T
    assert tr.rho[-1] > atm.rho
    assert tr.u[-1] < atm.wind_speed(max(tr.h[-1], 1e-3)) * 1.05


def test_travel_time_is_monotonic_and_starts_at_zero(run):
    tr, *_ = run
    assert tr.t[0] == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.diff(tr.t) > 0)


def test_dense_cloud_stays_grounded(run):
    tr, *_ = run
    assert np.all(tr.z_c == 0.0)


# ===========================================================================
# source-area expansion
# ===========================================================================
def test_burro8_requires_source_expansion(run):
    """
    A 117 kg/s LNG release into a 1.9 m/s wind cannot be carried through its
    geometric footprint; EQ 4d forces the source to spread upwind.
    """
    _, used, _, src = run
    assert used.half_width > src.half_width_geometric * 2.0
    assert used.rate == src.rate                       # mass rate preserved


def test_expansion_search_drives_the_widening_to_zero(run):
    """
    The converged footprint is the one that needs no further help: a smaller
    one still triggers in-place widening (EQ 4d), a larger one does not.
    The integration never fails outright — since the reference widens in
    place rather than abandoning the step, so does this implementation.
    """
    tr, used, atm, src = run
    assert tr.meta["source_widening"] <= 1e-3 * used.half_width

    smaller = src.expanded(used.half_width * 0.85)
    tr_small = integrate_plume(smaller, atm, LegacyThermo(METHANE),
                               water_backend(), x_max=200.0)
    assert tr_small.meta["source_widening"] > 0.05 * smaller.half_width


def test_weak_release_needs_no_expansion():
    """A small, warm, low-rate release fits through its own footprint."""
    atm = Atmosphere(u_ref=5.0, z_ref=10.0, T=290.0, rh=50.0, z0=0.01,
                     stability="D")
    src = EvaporatingPool(substance=METHANE, rate=0.5, area=100.0, duration=600.0)
    tr = integrate_plume(src, atm, LegacyThermo(METHANE), water_backend(),
                         x_max=500.0)
    assert len(tr) > 10
    assert tr.mass_frac[-1] < 1e-3


# ===========================================================================
# comparison with the manual, plume region only
# ===========================================================================
def test_manual_table_switches_to_puff_mode():
    """
    The manual's example-1 table cannot be compared beyond about 50 m.

    Its own CM column stops satisfying m = q/(2 rho U B h) there — the ratio
    grows 1.00, 1.13, 1.62, 2.11 — because the release lasts 107 s and the
    cloud has by then left plume mode.  EQ 15a replaces EQ 1a and R becomes a
    volume rather than a flux.  Comparing a plume-only run past that point is
    a category error, not a discrepancy.
    """
    q, Bs = 117.0, 31.1
    table = [   # x, h, BB, U, rho, CM
        (0.0, 1.89, 46.3, 0.808, 1.31, 0.316),
        (31.1, 2.08, 78.6, 1.00, 1.28, 0.280),
        (59.1, 1.69, 102.0, 1.19, 1.24, 0.204),
        (260.0, 3.44, 165.0, 1.65, 1.16, 0.0255),
    ]
    ratios = []
    for x, h, B, U, rho, cm in table:
        R = rho * U * B * h
        m = q / (2 * R) if x > Bs else q * (x + Bs) / (4 * Bs * R)
        ratios.append(m / cm)
    assert ratios[0] == pytest.approx(1.0, abs=0.02)     # plume: consistent
    assert ratios[1] == pytest.approx(1.0, abs=0.02)
    assert ratios[2] > 1.1                               # past transition
    assert ratios[3] > 2.0


def test_agrees_with_the_manual_inside_the_plume_region(run):
    """
    Order-of-magnitude agreement at the source, which is as much as a
    plume-only implementation can be asked for.  A tighter comparison needs
    the golden reference driven with the same effective source width; the
    scanned table is not accurate enough on its own.
    """
    tr, *_ = run
    i = int(np.argmin(np.abs(tr.x)))
    assert tr.T[i] == pytest.approx(216.0, rel=0.10)
    assert tr.rho[i] == pytest.approx(1.31, rel=0.05)
    assert tr.mass_frac[i] == pytest.approx(0.316, rel=0.20)
    assert tr.vol_frac[i] == pytest.approx(0.454, rel=0.20)
    assert tr.u[i] == pytest.approx(0.808, rel=0.15)


# ===========================================================================
# numerical settings
# ===========================================================================
def test_result_converges_with_substeps():
    tr3, _, _, _ = burro8(x_max=150.0, substeps=3)
    tr9, _, _, _ = burro8(x_max=150.0, substeps=9)
    i3 = int(np.argmin(np.abs(tr3.x - 100.0)))
    i9 = int(np.argmin(np.abs(tr9.x - 100.0)))
    assert tr9.vol_frac[i9] == pytest.approx(tr3.vol_frac[i3], rel=0.15)


# ===========================================================================
# gravity-front entrainment, end to end
# ===========================================================================
def _burro8_with(alpha_front):
    from slabx.post.concentration import concentration_field

    atm = Atmosphere(u_ref=1.92, z_ref=2.88, T=306.0, rh=4.6, z0=2e-4,
                     inv_L=0.0665)
    src = EvaporatingPool(substance=METHANE, rate=117.0, area=657.0,
                          duration=107.0)
    tr, _ = run_dispersion(src, atm, LegacyThermo(METHANE), water_backend(),
                           x_max=300.0, n_puff_steps=40,
                           coeffs=COEFFS.perturb(alpha_front=alpha_front,
                                                 name="f"))
    field = concentration_field(tr, atm, z="max", t_avg=10.0, t_release=107.0)
    return tr, field


def test_frontal_entrainment_dilutes_without_reshaping_the_profile():
    """
    The effect must be a dilution: more mass entrained, lower concentration,
    shorter hazard distance — with the cross-wind *shape* B/b essentially
    unchanged, since the term does not widen the cloud.
    """
    off, f_off = _burro8_with(0.0)
    on, f_on = _burro8_with(0.2)
    i = int(np.argmin(np.abs(off.x - 200.0)))
    j = int(np.argmin(np.abs(on.x - 200.0)))

    assert on.R_flux[j] > off.R_flux[i]                  # more air entrained
    assert on.mass_frac[j] < off.mass_frac[i]            # more dilute
    shape_off = off.b_half[i] / off.b_shape[i]
    shape_on = on.b_half[j] / on.b_shape[j]
    assert shape_on == pytest.approx(shape_off, rel=0.02)
    assert f_on.distance_to(0.05) < f_off.distance_to(0.05)


def test_frontal_entrainment_effect_grows_monotonically():
    d = [_burro8_with(a)[1].distance_to(0.05) for a in (0.0, 0.05, 0.1, 0.2)]
    assert all(b <= a for a, b in zip(d, d[1:]))
    assert d[-1] < d[0]
