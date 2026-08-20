"""
Tier-0 verification of the atmosphere submodel.

These tests check internal consistency and known analytic limits.  They do
not touch experimental data — a failure here is a bug, not a modelling
error.
"""

import math
import warnings

import pytest
from scipy.integrate import quad

from slabx.coefficients import PHYS
from slabx.submodels.atmosphere import Atmosphere, StabilityMap


ALL_CLASSES = ["A", "B", "C", "D", "E", "F"]


def burro8(**kw):
    """Burro 8 conditions (SLAB manual example 1)."""
    d = dict(u_ref=1.92, z_ref=2.88, T=306.0, rh=4.6, z0=2e-4, inv_L=0.0665)
    d.update(kw)
    return Atmosphere(**d)


# ===========================================================================
# stability mapping
# ===========================================================================
@pytest.mark.parametrize("z0", [2e-4, 0.003, 0.01, 0.02, 0.1, 0.5])
@pytest.mark.parametrize("s", [2.0, 3.0, 3.5, 4.5, 5.0, 5.5, 6.0])
def test_stability_roundtrip(z0, s):
    """
    s -> 1/L -> s is the identity where |s-4| <= 2 or at the anchor |s-4| = 3.
    Ermak's forward map interpolates the exponent in `aal` space and the
    reverse map in `astb` space, so they coincide only at those points.
    """
    m = StabilityMap(z0)
    inv_L = m.inv_L_from_stability(s)
    assert m.stability_from_inv_L(inv_L) == pytest.approx(s, rel=1e-9)


@pytest.mark.parametrize("z0", [2e-4, 0.02, 0.5])
@pytest.mark.parametrize("s", [1.2, 1.5, 1.8, 6.2, 6.5, 6.8])
def test_stability_roundtrip_known_inconsistency(z0, s):
    """
    Inside 2 < |s-4| < 3.5 the mapping is not invertible.  Document the size
    of the discrepancy so a future change cannot silently make it worse.
    """
    m = StabilityMap(z0)
    back = m.stability_from_inv_L(m.inv_L_from_stability(s))
    assert back == pytest.approx(s, abs=0.35)
    assert (back - 4.0) * (s - 4.0) > 0          # sign of stability preserved


@pytest.mark.parametrize("z0", [2e-4, 0.02, 0.1])
def test_stability_monotone(z0):
    """|1/L| must increase monotonically away from neutral."""
    m = StabilityMap(z0)
    stable = [m.inv_L_from_stability(4.0 + d) for d in (0.5, 1.0, 1.5, 2.0, 3.0)]
    assert all(b > a for a, b in zip(stable, stable[1:]))
    unstable = [m.inv_L_from_stability(4.0 - d) for d in (0.5, 1.0, 1.5, 2.0, 3.0)]
    assert all(b < a for a, b in zip(unstable, unstable[1:]))


def test_neutral_is_zero():
    assert StabilityMap(2e-4).inv_L_from_stability(4.0) == pytest.approx(0.0)


def test_legacy_bug_is_reproducible_and_off_by_default():
    """
    The JavaScript transcription compares an undeclared `all`, so every
    |1/L| >= al2 falls
    through to the most stable class.  We must be able to reproduce that,
    and must not do so by default.
    """
    m = StabilityMap(0.1)
    inv_L = 0.5 * (m.al2 + m.al3)          # lands in the broken branch
    assert m.stability_from_inv_L(inv_L, legacy_bug=True) == pytest.approx(7.5)
    good = m.stability_from_inv_L(inv_L, legacy_bug=False)
    assert 6.0 < good < 7.5
    atm = Atmosphere(u_ref=3.0, z_ref=10.0, T=290.0, z0=0.1, inv_L=inv_L)
    assert atm.s == pytest.approx(good)     # default = intended behaviour


# ===========================================================================
# wind profile: closed form must be the exact integral of the gradient
# ===========================================================================
@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_profile_is_integral_of_gradient(cls):
    """u_a(z) from the closed form == numerical integral of EQ 32."""
    atm = Atmosphere(u_ref=4.0, z_ref=10.0, T=290.0, z0=0.02, stability=cls)
    for z in (5.0, 20.0, 50.0, 100.0):
        if z >= atm.h_mix:
            continue
        num, err = quad(atm.wind_gradient, atm._zt, z, limit=200)
        closed = atm.wind_speed(z) - atm.wind_speed(atm._zt)
        assert closed == pytest.approx(num, rel=1e-6, abs=10 * err)


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_profile_continuous_at_transition(cls):
    """Value and slope must match at z_t = e*z0 (EQ 34c)."""
    atm = Atmosphere(u_ref=4.0, z_ref=10.0, T=290.0, z0=0.02, stability=cls)
    zt, eps = atm._zt, 1e-9
    lo, hi = atm.dimensionless_profile(zt - eps), atm.dimensionless_profile(zt + eps)
    assert lo == pytest.approx(hi, rel=1e-6, abs=1e-9)
    d_lo = (atm.dimensionless_profile(zt - eps) -
            atm.dimensionless_profile(zt - 3 * eps)) / (2 * eps)
    d_hi = (atm.dimensionless_profile(zt + 3 * eps) -
            atm.dimensionless_profile(zt + eps)) / (2 * eps)
    assert d_lo == pytest.approx(d_hi, rel=1e-4)


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_reference_speed_recovered(cls):
    """By construction u_a(z_ref) must equal the input wind speed."""
    atm = Atmosphere(u_ref=4.0, z_ref=10.0, T=290.0, z0=0.02, stability=cls)
    assert atm.wind_speed(10.0) == pytest.approx(4.0, rel=1e-12)


def test_neutral_reduces_to_log_law():
    """With 1/L = 0 and H -> inf the profile is the plain log law."""
    atm = Atmosphere(u_ref=5.0, z_ref=10.0, T=290.0, z0=0.03, inv_L=0.0)
    assert atm.inv_L_eff == 0.0
    for z in (1.0, 5.0, 10.0):
        log_law = atm.u_star / PHYS.VON_KARMAN * math.log(z / atm.z0)
        corr = atm.u_star / PHYS.VON_KARMAN * (z - atm.z0) / atm.h_mix
        assert atm.wind_speed(z) == pytest.approx(log_law - corr, rel=1e-12)


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_profile_increases_with_height(cls):
    atm = Atmosphere(u_ref=4.0, z_ref=10.0, T=290.0, z0=0.02, stability=cls)
    zs = [0.5, 1, 2, 5, 10, 20, 40]
    us = [atm.wind_speed(z) for z in zs if z < atm.h_mix]
    assert all(b > a for a, b in zip(us, us[1:]))


def test_stable_profile_steeper_than_unstable():
    """Shear must be larger under stable stratification."""
    kw = dict(u_ref=4.0, z_ref=10.0, T=290.0, z0=0.02)
    ratios = {}
    for cls in ("B", "D", "F"):
        a = Atmosphere(stability=cls, **kw)
        ratios[cls] = a.wind_speed(20.0) / a.wind_speed(2.0)
    assert ratios["F"] > ratios["D"] > ratios["B"]


# ===========================================================================
# mixing height
# ===========================================================================
@pytest.mark.parametrize(
    "cls,expected", [("A", 130 * 2**6), ("B", 130 * 2**5), ("C", 130 * 2**4),
                     ("D", 130 * 2**3), ("E", 130 * 2**2), ("F", 130 * 2**1)]
)
def test_mixing_height_table(cls, expected):
    """H = 130 * 2^(7-s), section 2.5.1."""
    atm = Atmosphere(u_ref=4.0, z_ref=10.0, T=290.0, z0=0.02, stability=cls)
    assert atm.h_mix == pytest.approx(expected)


# ===========================================================================
# depth-averaged wind
# ===========================================================================
@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_mean_wind_converges(cls):
    """Simpson 3-point (SLAB) vs a refined rule: same to a few percent."""
    atm = Atmosphere(u_ref=4.0, z_ref=10.0, T=290.0, z0=0.02, stability=cls)
    fine = atm.mean_wind_over(10.0, n=400)
    finer = atm.mean_wind_over(10.0, n=1600)
    assert fine == pytest.approx(finer, rel=2e-3)          # rule converges
    assert atm.wind_speed(1e-9) < fine < atm.wind_speed(10.0)
    # SLAB itself uses the 3-point rule; on a log profile that is coarse.
    # Record the bias rather than pretend it is small.
    coarse = atm.mean_wind_over(10.0, n=2)
    assert coarse == pytest.approx(fine, rel=0.25)


# ===========================================================================
# moist air
# ===========================================================================
def test_dry_air_limits():
    atm = Atmosphere(u_ref=4.0, z_ref=10.0, T=290.0, rh=0.0, z0=0.02, stability="D")
    assert atm.mass_frac_water == pytest.approx(0.0)
    assert atm.mw_moist == pytest.approx(PHYS.MW_AIR)
    assert atm.cp_moist == pytest.approx(PHYS.CP_AIR)


def test_humidity_lowers_molecular_weight():
    kw = dict(u_ref=4.0, z_ref=10.0, T=300.0, z0=0.02, stability="D")
    dry = Atmosphere(rh=0.0, **kw)
    wet = Atmosphere(rh=80.0, **kw)
    assert wet.mass_frac_water > 0
    assert wet.mw_moist < dry.mw_moist        # water is lighter than air
    assert wet.cp_moist > dry.cp_moist
    assert wet.rho < dry.rho


def test_burro8_reference_values():
    """Sanity-check the manual's example-1 conditions."""
    atm = burro8()
    assert atm.s == pytest.approx(4.5457, abs=5e-3)   # manual: STAB = 4.5457
    assert atm.h_mix == pytest.approx(712.44, rel=2e-3)  # manual: HMX = 7.1244E+02
    assert atm.u_star == pytest.approx(0.070342, rel=2e-3)  # manual: 7.0342E-02
    # The manual prints RHOA = 1.1623, which is inconsistent with its own
    # equation of state and with the printed WMAE/PA/TA.  Those give 1.1523;
    # 6 <-> 5 confusion is pervasive in the scan.
    assert atm.rho == pytest.approx(
        atm.mw_moist * atm.p / (PHYS.R_GAS * atm.T), rel=1e-12
    )
    assert atm.rho == pytest.approx(1.1523, rel=2e-3)
    assert atm.mw_moist == pytest.approx(0.028933, rel=2e-3)
    assert atm.cp_moist == pytest.approx(1007.1, rel=2e-3)


# ===========================================================================
# input validation
# ===========================================================================
def test_rejects_missing_stability():
    with pytest.raises(ValueError):
        Atmosphere(u_ref=4.0, z_ref=10.0, T=290.0, z0=0.02)


def test_rejects_out_of_range_class():
    with pytest.raises(ValueError):
        Atmosphere(u_ref=4.0, z_ref=10.0, T=290.0, z0=0.02, stability=9.0)


def test_rejects_bad_roughness():
    with pytest.raises(ValueError):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Atmosphere(u_ref=4.0, z_ref=10.0, T=290.0, z0=0.0, stability="D")


# ===========================================================================
# legacy reference-implementation defects
# ===========================================================================
@pytest.mark.parametrize("cls", ["A", "B", "C"])
def test_legacy_unstable_profile_differs_slightly(cls):
    """
    The ported closed form has a sign flip and a spurious square root.  The
    error must stay small (it is divided by gu*H) but be non-zero, and only
    the exact form may integrate the gradient.
    """
    kw = dict(u_ref=4.0, z_ref=10.0, T=290.0, z0=0.02, stability=cls)
    exact = Atmosphere(**kw)
    legacy = Atmosphere(legacy_unstable_profile=True, **kw)
    for z in (5.0, 50.0, 200.0):
        rel = abs(legacy.wind_speed(z) / exact.wind_speed(z) - 1.0)
        assert 0.0 < rel < 1e-3

    num, err = quad(exact.wind_gradient, exact._zt, 100.0, limit=300)
    closed = exact.wind_speed(100.0) - exact.wind_speed(exact._zt)
    assert closed == pytest.approx(num, rel=1e-8, abs=10 * err)


def test_legacy_flags_off_by_default():
    atm = Atmosphere(u_ref=4.0, z_ref=10.0, T=290.0, z0=0.02, stability="B")
    assert atm.legacy_unstable_profile is False
    assert atm.legacy_stability_bug is False
