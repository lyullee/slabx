"""Tier-0/1 verification of the source models."""

import math

import pytest

from slabx.coefficients import PHYS
from slabx.core.source import (
    EvaporatingPool,
    SourceModel,
    initial_cloud_height,
)
from slabx.core.trajectory import Mode
from slabx.submodels.atmosphere import Atmosphere
from slabx.thermo.base import Substance

METHANE = Substance(
    name="LNG", mw=0.016043, cp_vapour=2238.0, cp_liquid=3348.5,
    dh_vap=509900.0, T_boil=111.7, rho_liquid=424.1,
)


def burro8_atm():
    return Atmosphere(u_ref=1.92, z_ref=2.88, T=306.0, rh=4.6,
                      z0=2e-4, inv_L=0.0665)


def burro8_pool(**kw):
    d = dict(substance=METHANE, rate=117.0, area=657.0, duration=107.0)
    d.update(kw)
    return EvaporatingPool(**d)


# ===========================================================================
# protocol and validation
# ===========================================================================
def test_satisfies_protocol():
    assert isinstance(burro8_pool(), SourceModel)


@pytest.mark.parametrize("bad", [dict(rate=0.0), dict(area=-1.0), dict(duration=0.0)])
def test_rejects_invalid_input(bad):
    with pytest.raises(ValueError):
        burro8_pool(**bad)


# ===========================================================================
# source properties against the manual's printed values (example 1)
# ===========================================================================
def test_source_density_matches_manual():
    """Manual example 1 prints RHOS = 1.7503 kg/m^3."""
    assert burro8_pool().rho_source == pytest.approx(1.7503, rel=1e-3)


def test_half_width_matches_manual():
    """B_s = 0.5 sqrt(A_s); manual prints BS = 1.2816E+01 for AS = 657 m^2."""
    assert burro8_pool().half_width == pytest.approx(12.816, rel=1e-3)


def test_vertical_velocity_matches_manual():
    """W_s = q / (rho_s A_s); manual prints WS = 1.0174E-01."""
    assert burro8_pool().w_source == pytest.approx(0.10174, rel=1e-3)


def test_total_mass_matches_manual():
    """QTCS = q t_sd; manual prints 1.2519E+04 kg."""
    assert burro8_pool().total_mass == pytest.approx(12519.0)


def test_continuous_release_has_infinite_mass():
    assert math.isinf(burro8_pool(duration=math.inf).total_mass)


def test_source_temperature_is_the_boiling_point():
    assert burro8_pool().T == pytest.approx(METHANE.T_boil)
    assert burro8_pool(T_source=90.0).T == pytest.approx(METHANE.T_boil)
    assert burro8_pool(T_source=150.0).T == pytest.approx(150.0)


# ===========================================================================
# source flux
# ===========================================================================
def test_flux_is_confined_to_the_source_region():
    p = burro8_pool()
    B = p.half_width
    assert p.flux(0.0, 0.0).is_active
    assert p.flux(-B, 0.0).is_active
    assert p.flux(B, 0.0).is_active
    assert not p.flux(B * 1.001, 0.0).is_active
    assert not p.flux(-B * 1.001, 0.0).is_active


def test_flux_stops_after_the_release():
    p = burro8_pool()
    assert p.flux(0.0, 106.0).is_active
    assert not p.flux(0.0, 108.0).is_active


def test_integrated_flux_recovers_the_source_rate():
    """
    EQ 2a integrates ``rho_s W_s B_s`` across the source region:

        int_{-Bs}^{Bs} rho_s W_s B_s dx = 2 rho_s W_s B_s^2 = q / 2

    The factor 1/2 is not an error: B is a *half*-width, so R = rho U B h
    describes half the plume.  EQ 1a is consistent — at the downwind edge it
    gives m = q / (2R).
    """
    p = burro8_pool()
    f = p.flux(0.0, 0.0)
    integral = 2.0 * p.half_width * f.mass
    assert integral == pytest.approx(0.5 * p.rate, rel=1e-9)


def test_enthalpy_flux_is_consistent():
    p = burro8_pool()
    f = p.flux(0.0, 0.0)
    assert f.enthalpy == pytest.approx(f.mass * METHANE.cp_vapour * p.T)


# ===========================================================================
# source-area expansion
# ===========================================================================
def test_expansion_preserves_mass_rate_and_lowers_velocity():
    p = burro8_pool()
    big = p.expanded(3.0 * p.half_width_geometric)
    assert big.rate == p.rate
    assert big.half_width == pytest.approx(3.0 * p.half_width_geometric)
    assert big.w_source_effective == pytest.approx(p.w_source / 9.0)
    # enlarging the footprint must not change the total mass released
    integral = 2.0 * big.half_width * big.flux(0.0, 0.0).mass
    assert integral == pytest.approx(0.5 * p.rate, rel=1e-9)


def test_expansion_cannot_shrink_the_source():
    p = burro8_pool()
    with pytest.raises(ValueError):
        p.expanded(0.5 * p.half_width_geometric)


def test_unexpanded_effective_velocity_equals_geometric():
    p = burro8_pool()
    assert p.w_source_effective == pytest.approx(p.w_source)


# ===========================================================================
# initial cloud height
# ===========================================================================
def test_initial_height_satisfies_its_defining_equation():
    atm = burro8_atm()
    dx = 0.5
    h = initial_cloud_height(atm, dx)
    z0 = atm.z0
    z1 = (math.sqrt(3.0) * 1.5 * PHYS.VON_KARMAN**3
          * atm.dimensionless_profile(4.0) * dx / atm.u_star)
    F = math.log1p(h / z0)
    residual = (h + z0) * F * F - 2.0 * (h + z0) * F + 2.0 * h - z1
    assert residual == pytest.approx(0.0, abs=1e-8 * max(z1, 1.0))


def test_initial_height_grows_with_step_length():
    atm = burro8_atm()
    hs = [initial_cloud_height(atm, dx) for dx in (0.1, 0.5, 2.0, 10.0)]
    assert all(b > a for a, b in zip(hs, hs[1:]))
    assert all(0.0 < h < atm.h_mix for h in hs)


@pytest.mark.parametrize("z0", [2e-4, 0.003, 0.02, 0.1])
@pytest.mark.parametrize("cls", ["A", "D", "F"])
def test_initial_height_converges_everywhere(z0, cls):
    atm = Atmosphere(u_ref=4.0, z_ref=10.0, T=290.0, z0=z0, stability=cls)
    h = initial_cloud_height(atm, 1.0)
    assert math.isfinite(h) and h > 0.0


def test_initial_height_rejects_bad_step():
    with pytest.raises(ValueError):
        initial_cloud_height(burro8_atm(), 0.0)


# ===========================================================================
# initial state
# ===========================================================================
def test_initial_state_starts_as_ambient_air():
    """
    The cloud is created empty at the upwind edge; material enters only
    through the source flux.  A non-zero initial concentration here would
    double-count the source.
    """
    atm = burro8_atm()
    s = burro8_pool().initial_state(atm, dx=0.5)
    assert s.m_emission == 0.0 and s.m_ev == 0.0
    assert s.T == pytest.approx(atm.T)
    assert s.rho == pytest.approx(atm.rho)
    assert s.cp == pytest.approx(atm.cp_moist)
    assert s.m_water == pytest.approx(atm.mass_frac_water)
    assert s.h == 0.0 and s.R_flux == 0.0


def test_initial_state_geometry():
    p = burro8_pool()
    s = p.initial_state(burro8_atm(), dx=0.5)
    assert s.x_start == pytest.approx(-p.half_width)
    assert s.b_half == pytest.approx(p.half_width)
    assert s.b_shape == pytest.approx(0.9 * p.half_width)
    assert s.mode is Mode.PLUME
    assert s.b_half_x == 1.0                     # plume mode marker


def test_initial_beta_from_the_shape_relation():
    """EQ 13: B^2 = b^2 + 3 beta^2."""
    s = burro8_pool().initial_state(burro8_atm(), dx=0.5)
    assert s.b_half**2 == pytest.approx(
        s.b_shape**2 + 3.0 * s.beta**2, rel=1e-12
    )


def test_initial_velocity_is_the_depth_averaged_wind():
    atm = burro8_atm()
    s = burro8_pool().initial_state(atm, dx=0.5)
    he = initial_cloud_height(atm, 0.5)
    assert s.u == pytest.approx(s.u_ambient_mean)
    assert 0.0 < s.u < atm.wind_speed(he)        # mean is below the top value


def test_initial_entrainment_produces_the_first_step_height():
    """
    w is seeded so that the first step grows the cloud to `he`:
    sqrt(3) w dx / u == he.
    """
    atm = burro8_atm()
    dx = 0.5
    s = burro8_pool().initial_state(atm, dx=dx)
    he = initial_cloud_height(atm, dx)
    assert math.sqrt(3.0) * s.w_entrain * dx / s.u == pytest.approx(he, rel=1e-12)


def test_expanded_source_shifts_the_start_upwind():
    p = burro8_pool()
    big = p.expanded(4.0 * p.half_width_geometric)
    s0 = p.initial_state(burro8_atm(), dx=0.5)
    s1 = big.initial_state(burro8_atm(), dx=0.5)
    assert s1.x_start < s0.x_start
    assert s1.b_half == pytest.approx(4.0 * s0.b_half)


@pytest.mark.parametrize("cls", ["A", "B", "C", "D", "E", "F"])
def test_initial_state_valid_in_every_stability(cls):
    atm = Atmosphere(u_ref=3.0, z_ref=10.0, T=290.0, rh=40.0, z0=0.01,
                     stability=cls)
    s = burro8_pool().initial_state(atm, dx=1.0)
    assert s.u > 0 and s.w_entrain > 0 and s.v_entrain > 0
    assert 0.0 <= s.m_water < 1.0


# ===========================================================================
# scaled vs unscaled source velocity
# ===========================================================================
def test_flux_reports_the_geometric_velocity_for_the_shear_term():
    """
    Regression guard.  `SourceFlux.w_source` feeds the shear term of the
    in-cloud friction velocity (EQ 35e) and must stay at the geometric value
    even after the footprint is enlarged; only the *mass* flux is reduced.
    The reference keeps the two apart as `wss = ws` and
    `wse = ws bs^2 / bse^2` (SLAB.FOR L770-771).

    Confusing them costs nothing until the source expands — which is exactly
    why it survived every test that did not need expansion.
    """
    p = burro8_pool()
    big = p.expanded(3.0 * p.half_width_geometric)

    assert big.flux(0.0, 0.0).w_source == pytest.approx(p.w_source)
    assert big.w_source_effective == pytest.approx(p.w_source / 9.0)
    # mass rate still conserved
    integral = 2.0 * big.half_width * big.flux(0.0, 0.0).mass
    assert integral == pytest.approx(0.5 * p.rate, rel=1e-9)
