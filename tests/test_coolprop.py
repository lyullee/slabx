"""
Real-fluid backend.

Skipped when CoolProp is absent, so the suite still runs without it.  The
point of these tests is that the backend is a *drop-in*: it satisfies the same
protocol, produces physically sensible properties over the whole range a
cloud traverses, and leaves the legacy path bit-for-bit unchanged.
"""

import math

import numpy as np
import pytest

pytest.importorskip("CoolProp")

from slabx.coefficients import PHYS, preset                       # noqa: E402
from slabx.core.plume import run_dispersion                       # noqa: E402
from slabx.core.source import EvaporatingPool                     # noqa: E402
from slabx.submodels.atmosphere import Atmosphere                 # noqa: E402
from slabx.thermo.base import (                                   # noqa: E402
    WATER, LegacyThermo, Substance, ThermoBackend, water_backend,
)
from slabx.thermo.coolprop import CoolPropThermo, coolprop_water  # noqa: E402
from slabx.thermo.equilibrium import Mixture, solve_equilibrium   # noqa: E402

METHANE = Substance(name="LNG", mw=0.016043, cp_vapour=2238.0, cp_liquid=3348.5,
                    dh_vap=509900.0, T_boil=111.7, rho_liquid=424.1)
AMMONIA = Substance(name="NH3", mw=0.017031, cp_vapour=2045.9, cp_liquid=4611.8,
                    dh_vap=1170000.0, T_boil=239.57, rho_liquid=603.0,
                    sat_B=2976.01)
CHLORINE = Substance(name="Cl2", mw=0.070906, cp_vapour=498.1, cp_liquid=926.3,
                     dh_vap=287840.0, T_boil=239.1, rho_liquid=1574.0,
                     sat_B=1978.34, sat_C=-27.01)
ALL = [METHANE, AMMONIA, CHLORINE]


# ===========================================================================
# protocol conformance
# ===========================================================================
@pytest.mark.parametrize("sub", ALL)
def test_satisfies_the_backend_protocol(sub):
    assert isinstance(CoolPropThermo(sub), ThermoBackend)


def test_unknown_substance_is_reported():
    odd = Substance(name="Unobtainium", mw=0.1, cp_vapour=500.0,
                    cp_liquid=900.0, dh_vap=2e5, T_boil=200.0, rho_liquid=1000.0)
    with pytest.raises(KeyError, match="Unobtainium"):
        CoolPropThermo(odd)
    with pytest.raises(ValueError):
        CoolPropThermo(odd, fluid="NotAFluid")


def test_explicit_fluid_overrides_the_name():
    assert CoolPropThermo(METHANE, fluid="Ethane").name == "coolprop:Ethane"


# ===========================================================================
# properties are physical over the whole range a cloud traverses
# ===========================================================================
@pytest.mark.parametrize("sub", ALL)
def test_properties_finite_and_positive_everywhere(sub):
    """
    The equilibrium solver brackets its root over [1, 2000] K, so every
    property must return a usable number far outside the two-phase range —
    including above the critical point, which for methane is 190.6 K and so
    squarely inside an LNG cloud's trajectory.
    """
    th = CoolPropThermo(sub)
    for T in np.geomspace(50.0, 1500.0, 40):
        for f in (th.saturation_ratio, th.cp_vapour, th.cp_liquid,
                  th.rho_liquid, th.dh_vap, th.dh_vap_datum):
            v = f(float(T))
            assert math.isfinite(v), (f.__name__, T)
            assert v >= 0.0, (f.__name__, T)
        assert th.cp_vapour(float(T)) < 1e5      # not the critical divergence


@pytest.mark.parametrize("sub", ALL)
def test_saturation_matches_the_boiling_point(sub):
    """P_sat(T_boil) = P_atm to within CoolProp's own accuracy."""
    th = CoolPropThermo(sub)
    import CoolProp.CoolProp as CP
    T_b = CP.PropsSI("T", "P", PHYS.P_ATM, "Q", 0, th._fluid)
    assert th.saturation_ratio(T_b) == pytest.approx(1.0, rel=1e-3)


@pytest.mark.parametrize("sub", ALL)
def test_saturation_increases_with_temperature(sub):
    th = CoolPropThermo(sub)
    f = [th.saturation_ratio(T) for T in np.linspace(0.6 * sub.T_boil,
                                                     1.2 * sub.T_boil, 12)]
    assert all(b >= a for a, b in zip(f, f[1:]))


@pytest.mark.parametrize("sub", ALL)
def test_saturation_derivative_matches_finite_difference(sub):
    th = CoolPropThermo(sub)
    T = 0.95 * sub.T_boil
    h = 0.05
    fd = (th.saturation_ratio(T + h) - th.saturation_ratio(T - h)) / (2 * h)
    assert th.d_saturation_ratio(T) == pytest.approx(fd, rel=0.05)


@pytest.mark.parametrize("sub", ALL)
def test_latent_heat_falls_towards_the_critical_point(sub):
    th = CoolPropThermo(sub)
    lo, hi = th.dh_vap(sub.T_boil), th.dh_vap(0.98 * th._T_crit)
    assert 0.0 <= hi < lo


def test_latent_heat_is_zero_above_the_critical_point():
    th = CoolPropThermo(METHANE)
    assert th.dh_vap(th._T_crit + 50.0) == 0.0


@pytest.mark.parametrize("sub", ALL)
def test_datum_latent_heat_is_well_conditioned(sub):
    """
    Regression guard.  ``dH - T (cp_v - cp_l)`` diverges near the critical
    point, where the latent heat collapses but the cp difference does not.
    Freezing the evaluation at the boiling point keeps it bounded; without
    that, methane at 250 K returns 1.3e8 J/kg instead of 6.7e5.
    """
    th = CoolPropThermo(sub)
    vals = [th.dh_vap_datum(T) for T in (0.7 * sub.T_boil, sub.T_boil,
                                         2.0 * sub.T_boil, 1500.0)]
    assert all(0.0 < v < 5e6 for v in vals)
    assert max(vals) / min(vals) < 1.2              # nearly constant
    # and above the boiling point it must not vary at all
    assert vals[2] == pytest.approx(vals[3], rel=1e-12)


def test_liquid_density_falls_with_temperature():
    th = CoolPropThermo(AMMONIA)
    rho = [th.rho_liquid(T) for T in (200.0, 250.0, 300.0)]
    assert all(b < a for a, b in zip(rho, rho[1:]))


# ===========================================================================
# it disagrees with the legacy model where it should
# ===========================================================================
def test_ammonia_properties_differ_substantially():
    """
    Ammonia is the case where SLAB's inputs are furthest from reality: its
    latent heat is 15 % low and its liquid density 12 % low at the boiling
    point.  If this stops being true the comparison table needs revisiting.
    """
    th = CoolPropThermo(AMMONIA)
    c = th.compare_with_legacy(AMMONIA.T_boil)
    assert c["dh_vap"][1] / c["dh_vap"][0] > 1.10
    assert c["rho_liquid"][1] / c["rho_liquid"][0] > 1.10


def test_chlorine_properties_are_close():
    """Chlorine's inputs are good, so the backend should barely move it."""
    th = CoolPropThermo(CHLORINE)
    c = th.compare_with_legacy(CHLORINE.T_boil)
    for k in ("dh_vap", "rho_liquid", "cp_liquid"):
        assert c[k][1] / c[k][0] == pytest.approx(1.0, abs=0.05)


# ===========================================================================
# it drops into the solver unchanged
# ===========================================================================
def test_equilibrium_solves_with_real_properties():
    atm = dict(T_ambient=306.0, rho_ambient=1.1523, mw_ambient_moist=0.028933)
    m = 0.31
    m_w = (1.0 - m) * 1.52e-3
    E = (m * 2238.0 * 111.7 + (1.0 - m) * 1007.14 * 306.0)
    mix = Mixture(m_emission=m, m_water=m_w,
                  m_dry_air=(1.0 - m) * (1.0 - 1.52e-3),
                  m_ev_transported=m, m_wv_transported=m_w, enthalpy=E)
    real = solve_equilibrium(mix, CoolPropThermo(METHANE), coolprop_water(),
                             **atm)
    old = solve_equilibrium(mix, LegacyThermo(METHANE), water_backend(), **atm)
    for eq in (real, old):
        assert 100.0 < eq.T < 306.0
        assert eq.rho > atm["rho_ambient"]
        assert eq.m_ev + eq.m_el == pytest.approx(m, rel=1e-12)
    assert real.T == pytest.approx(old.T, rel=0.05)   # same ballpark


def test_full_run_completes_and_stays_physical():
    atm = Atmosphere(u_ref=1.92, z_ref=2.88, T=306.0, rh=4.6, z0=2e-4,
                     inv_L=0.0665)
    src = EvaporatingPool(substance=METHANE, rate=117.0, area=657.0,
                          duration=107.0)
    tr, _ = run_dispersion(src, atm, CoolPropThermo(METHANE), coolprop_water(),
                           x_max=500.0, n_puff_steps=30,
                           coeffs=preset("ermak90"))
    assert len(tr) > 20
    assert np.all(np.isfinite(tr.vol_frac))
    assert np.all((tr.vol_frac >= 0) & (tr.vol_frac <= 1))
    assert np.all(np.diff(tr.mass_frac[tr.x > 0]) < 0)
    assert tr.T[-1] == pytest.approx(atm.T, rel=0.02)
    assert tr.meta["thermo"].startswith("coolprop:")


def test_the_change_is_larger_than_the_reproduction_error():
    """
    The whole point of the reproduction work: an effect of this size is
    attributable to the property model, because the baseline tracks the
    original implementation to under 1 %.
    """
    atm = Atmosphere(u_ref=1.92, z_ref=2.88, T=306.0, rh=4.6, z0=2e-4,
                     inv_L=0.0665)
    src = EvaporatingPool(substance=METHANE, rate=117.0, area=657.0,
                          duration=107.0)
    kw = dict(x_max=500.0, n_puff_steps=30, coeffs=preset("ermak90"))
    old, _ = run_dispersion(src, atm, LegacyThermo(METHANE), water_backend(), **kw)
    new, _ = run_dispersion(src, atm, CoolPropThermo(METHANE),
                            coolprop_water(), **kw)
    xs = np.geomspace(10.0, 400.0, 10)
    a = np.interp(xs, old.x, old.vol_frac)
    b = np.interp(xs, new.x, new.vol_frac)
    assert np.median(np.abs(b / a - 1.0)) > 0.01      # well above 0.93 %


# ===========================================================================
# the legacy path must be untouched
# ===========================================================================
@pytest.mark.parametrize("sub", ALL)
def test_legacy_datum_is_exactly_the_old_constant(sub):
    """
    `dh_vap_datum` generalised `Substance.dh_vap_ref` to be temperature
    dependent.  For constant specific heats the T-dependence cancels
    identically, so the legacy backend must return the old constant at every
    temperature — otherwise the reproduction baseline has moved.
    """
    th = LegacyThermo(sub)
    for T in (50.0, sub.T_boil, 300.0, 1000.0):
        assert th.dh_vap_datum(T) == sub.dh_vap_ref
    assert WATER.dh_vap_ref == pytest.approx(3136402.4, rel=1e-9)
