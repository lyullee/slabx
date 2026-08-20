"""
The three source types beyond the evaporating pool.

Each exercises a path the pool never touches, and each turned up a defect
that no pool test could have found:

* a jet starts as *pure material* rather than as ambient air, so the
  enthalpy carry term of EQ 41 needs m0 = 1;
* a pure saturated substance has no diluent, so the phase split is fixed by
  enthalpy rather than by temperature;
* an elevated jet has alpha_g = 0, which switches off both the source-area
  expansion *and* the h >= 2 z_c clamp;
* an instantaneous release is in puff mode from t = 0 and has no rate.
"""

import math

import numpy as np
import pytest

from slabx.coefficients import COEFFS, preset
from slabx.core.plume import run_dispersion
from slabx.core.source import (
    EvaporatingPool,
    HorizontalJet,
    InstantaneousRelease,
    SourceModel,
)
from slabx.core.trajectory import Mode
from slabx.submodels.atmosphere import Atmosphere
from slabx.thermo.base import LegacyThermo, Substance, water_backend

NH3 = Substance(name="NH3", mw=0.017031, cp_vapour=2045.9, cp_liquid=4611.8,
                dh_vap=1170000.0, T_boil=239.57, rho_liquid=603.0,
                sat_B=2976.01, sat_C=0.0)
LNG = Substance(name="LNG", mw=0.016043, cp_vapour=2238.0, cp_liquid=3348.5,
                dh_vap=509900.0, T_boil=111.7, rho_liquid=424.1)


def dt4_atm():
    return Atmosphere(u_ref=4.5, z_ref=2.0, T=306.2, rh=21.3, z0=0.003,
                      inv_L=0.0221, legacy_stability_bug=True,
                      legacy_unstable_profile=True)


def dt4_jet(**kw):
    d = dict(substance=NH3, rate=107.87, area=0.93, duration=381.0,
             liquid_fraction=0.81, height=1.0)
    d.update(kw)
    return HorizontalJet(**d)


def burro8_atm():
    return Atmosphere(u_ref=1.92, z_ref=2.88, T=306.0, rh=4.6, z0=2e-4,
                      inv_L=0.0665, legacy_stability_bug=True,
                      legacy_unstable_profile=True)


@pytest.fixture(scope="module")
def dt4():
    atm = dt4_atm()
    src = dt4_jet()
    tr, _ = run_dispersion(src, atm, LegacyThermo(NH3), water_backend(),
                           x_max=2800.0, n_puff_steps=50,
                           coeffs=preset("ermak90"))
    return tr, atm, src


@pytest.fixture(scope="module")
def instant():
    atm = burro8_atm()
    src = InstantaneousRelease(substance=LNG, rate=0.0, area=900.0,
                               duration=1e-9, mass=5000.0, height=0.0)
    tr, _ = run_dispersion(src, atm, LegacyThermo(LNG), water_backend(),
                           x_max=1000.0, n_puff_steps=50,
                           coeffs=preset("ermak90"))
    return tr, atm, src


# ===========================================================================
# protocol
# ===========================================================================
@pytest.mark.parametrize("src", [
    dt4_jet(),
    InstantaneousRelease(substance=LNG, rate=0.0, area=900.0, duration=1e-9,
                         mass=5000.0),
    EvaporatingPool(substance=LNG, rate=117.0, area=657.0, duration=107.0),
])
def test_all_types_satisfy_the_protocol(src):
    assert isinstance(src, SourceModel)
    assert src.released_mass(0.0) >= 0.0
    assert src.released_mass(1e9) == pytest.approx(src.total_mass)


def test_instantaneous_releases_everything_at_once():
    s = InstantaneousRelease(substance=LNG, rate=0.0, area=900.0,
                             duration=1e-9, mass=5000.0)
    assert s.released_mass(0.0) == 5000.0
    assert s.half_width == pytest.approx(15.0)      # 0.5 sqrt(900)


# ===========================================================================
# two-phase source state
# ===========================================================================
def test_two_phase_source_is_denser_than_its_own_vapour():
    """
    81 % droplets make an ammonia release five times denser than the vapour
    alone — the reason a light-molecule cloud disperses as a dense gas.
    """
    atm, jet = dt4_atm(), dt4_jet()
    rho_vapour = NH3.mw * 101325.0 / (8.31431 * jet.T)
    assert jet.rho_source(atm) > 4.0 * rho_vapour
    assert jet.rho_source(atm) > 3.0 * atm.rho


def test_dry_jet_reduces_to_its_vapour_density():
    atm = dt4_atm()
    dry = dt4_jet(liquid_fraction=0.0)
    rho_vapour = NH3.mw * 101325.0 / (8.31431 * dry.T)
    assert dry.rho_source(atm) == pytest.approx(rho_vapour, rel=0.02)


def test_jet_starts_as_pure_material_not_ambient_air(dt4):
    """
    Regression guard.  Initialising m0 = 0 (the pool convention) poisons the
    carry term of EQ 41 and sends the cloud above ambient temperature.
    """
    tr, atm, _ = dt4
    assert tr.mass_frac[0] == pytest.approx(1.0)
    assert tr.vol_frac[0] == pytest.approx(1.0)
    assert tr.T[0] == pytest.approx(NH3.T_boil)
    assert np.all(tr.T <= atm.T + 1e-6)          # never hotter than ambient


def test_droplets_cool_the_cloud_below_the_boiling_point(dt4):
    """Evaporative cooling must undershoot T_boil before dilution warms it."""
    tr, atm, _ = dt4
    assert tr.T.min() < NH3.T_boil - 5.0
    assert tr.mass_frac_emission_liquid[0] == pytest.approx(0.81, rel=1e-9)
    assert tr.mass_frac_emission_liquid[-1] == pytest.approx(0.0, abs=1e-12)
    assert tr.rho.max() > 3.0 * atm.rho


# ===========================================================================
# alpha_g and the two things it switches off
# ===========================================================================
def test_gravity_coefficient_follows_the_cloud_state():
    """
    alpha_g is a *state* variable, not a source attribute: the reference
    recomputes it every step from whether the cloud is lofted and whether it
    is denser than air (SLAB.FOR L2446-2637).  An elevated jet starts at
    zero and switches on once it settles.
    """
    from slabx.core.plume import gravity_coefficient
    rho_a = 1.2
    assert gravity_coefficient(True, 3.0, rho_a) == 0.0      # lofted
    assert gravity_coefficient(False, 3.0, rho_a) == COEFFS.alpha_g
    assert gravity_coefficient(False, 0.8, rho_a) == 0.0     # buoyant


def test_elevated_jet_starts_lofted(dt4):
    """
    Regression guard.  The initial state must report h_top = z_c + h/2 > h.
    Setting h_top = h classifies the jet as grounded from the first step,
    which switches on the gravity current and the h >= 2 z_c clamp together
    and collapses the jet within one step.
    """
    tr, _, src = dt4
    assert tr.z_c[0] > 0.5 * tr.h[0]
    assert tr.h[1] < 1.5 * tr.h[0]


def test_jet_keeps_its_momentum_through_the_first_steps(dt4):
    """
    Regression guard for the subtlest defect found so far.

    SLAB_eval writes the height clamp as

        if (h < 2 z_c && alfg*rho > alfg*rhoa) { h = 2 z_c; ... }

    The `alfg` on both sides looks like a no-op but is not: for an elevated
    jet alpha_g = 0, the test is false, and the clamp is disabled.  Dropping
    the factor pins h at 2 z_c from the first step, which forces
    U = R/(rho B h) down by a factor of four and destroys the jet — 33 %
    error against the reference, from an expression that reads as an
    identity.
    """
    tr, *_ = dt4
    near = tr.x < 3.0
    assert tr.u[0] > 20.0
    assert tr.u[near].min() > 0.7 * tr.u[0]      # decelerates gently
    assert np.all(tr.h[near] < 2.0)              # not pinned at 2 z_c


def test_jet_needs_no_source_expansion(dt4):
    """While the jet is lofted, EQ 4b has no gravity term and cannot fail."""
    tr, _, src = dt4
    assert tr.meta["source_widening"] == 0.0
    assert src.expanded(99.0) is src


# ===========================================================================
# instantaneous release
# ===========================================================================
def test_instantaneous_is_puff_from_the_start(instant):
    tr, *_ = instant
    assert np.all(tr.mode == Mode.PUFF)
    assert tr.t[0] == 0.0 and tr.u[0] == 0.0


def test_instantaneous_conserves_its_inventory(instant):
    tr, _, src = instant
    inv = tr.mass_frac * tr.R_flux
    assert np.allclose(inv, 0.25 * src.mass, rtol=1e-9)
    assert tr.mass_in_cloud[-1] == pytest.approx(0.5 * src.mass, rel=1e-9)


def test_instantaneous_slumps_then_grows(instant):
    """A dense instantaneous cloud collapses vertically before it rises."""
    tr, _, _ = instant
    i = int(np.argmin(tr.h))
    assert 0 < i < len(tr) - 1
    assert tr.h[i] < 0.5 * tr.h[0]
    assert tr.h[-1] > 3.0 * tr.h[0]


def test_instantaneous_dilutes_and_reaches_ambient(instant):
    tr, atm, _ = instant
    assert tr.mass_frac[0] == pytest.approx(1.0)
    assert np.all(np.diff(tr.mass_frac) < 0)
    assert tr.T[-1] == pytest.approx(atm.T, rel=0.01)
    assert tr.rho[-1] == pytest.approx(atm.rho, rel=0.01)


def test_elevated_source_averages_the_wind_to_the_cloud_top():
    """
    Regression guard.  EQ 4c averages the ambient wind from the ground to the
    cloud *top*, not over the cloud's thickness.  For a grounded release the
    two are the same; for a jet released at 1 m with a 1 m thick cloud the
    top is 50 % higher and the mean wind nearly 10 % larger.  Getting it
    wrong under-drives the cloud and, through the density stretch in the
    reconstruction, inflates the shape parameter downwind.
    """
    from slabx.core.source import _mean_wind

    atm, jet = dt4_atm(), dt4_jet()
    s = jet.initial_state(atm, dx=1.0)
    h_top = s.z_c + 0.5 * s.h
    assert h_top > s.h                                   # genuinely elevated
    assert s.u_ambient_mean == pytest.approx(_mean_wind(atm, h_top), rel=1e-9)
    assert s.u_ambient_mean > 1.05 * _mean_wind(atm, s.h)


def test_pool_is_unaffected_by_the_cloud_top_rule():
    """A grounded release has h_top = h, so nothing changes for it."""
    atm = burro8_atm()
    pool = EvaporatingPool(substance=LNG, rate=117.0, area=657.0,
                           duration=107.0)
    s = pool.initial_state(atm, dx=0.5)
    assert s.z_c == 0.0
    assert s.u_ambient_mean > 0.0
