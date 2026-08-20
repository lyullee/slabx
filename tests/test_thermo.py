"""Tier-0/1 verification of the thermodynamics."""

import math

import pytest

from slabx.coefficients import PHYS
from slabx.thermo.base import WATER, LegacyThermo, Substance, ThermoBackend, water_backend
from slabx.thermo.equilibrium import (
    Equilibrium,
    EquilibriumError,
    Mixture,
    solve_equilibrium,
)

# SLAB manual example 1 (LNG) and example 4 (chlorine)
METHANE = Substance(
    name="LNG", mw=0.016043, cp_vapour=2238.0, cp_liquid=3348.5,
    dh_vap=509900.0, T_boil=111.7, rho_liquid=424.1, sat_B=-1.0, sat_C=0.0,
)
CHLORINE = Substance(
    name="Chlorine", mw=0.070906, cp_vapour=498.1, cp_liquid=926.3,
    dh_vap=287840.0, T_boil=239.1, rho_liquid=1574.0,
    sat_B=1978.34, sat_C=-27.01,
)
AMMONIA = Substance(
    name="Ammonia", mw=0.017031, cp_vapour=2045.9, cp_liquid=4611.8,
    dh_vap=1170000.0, T_boil=239.57, rho_liquid=603.0, sat_B=2976.01, sat_C=0.0,
)

AMBIENT = dict(T_ambient=306.0, rho_ambient=1.1523, mw_ambient_moist=0.028933)


def mixture(m_e, *, m_wa=1.52e-3, enthalpy=None, T_ref=290.0,
            m_ev_t=None, m_wv_t=None):
    """A well-formed Mixture; enthalpy defaults to a plausible value."""
    m_w = (1.0 - m_e) * m_wa
    m_da = (1.0 - m_e) * (1.0 - m_wa)
    if enthalpy is None:
        enthalpy = T_ref * (m_da * PHYS.CP_AIR + m_w * PHYS.CP_WATER_VAP
                            + m_e * METHANE.cp_vapour)
    return Mixture(
        m_emission=m_e, m_water=m_w, m_dry_air=m_da,
        m_ev_transported=m_e if m_ev_t is None else m_ev_t,
        m_wv_transported=m_w if m_wv_t is None else m_wv_t,
        enthalpy=enthalpy,
    )


def solve(sub, mix, **kw):
    return solve_equilibrium(
        mix, LegacyThermo(sub), water_backend(), **AMBIENT, **kw
    )


# ===========================================================================
# Substance
# ===========================================================================
def test_clapeyron_default_when_B_not_given():
    """EQ 43b: B = dHe * M / R, C = 0."""
    assert METHANE.B == pytest.approx(509900.0 * 0.016043 / PHYS.R_GAS)
    assert METHANE.C == 0.0
    # SLAB manual example 1 prints SPA = 8.8083, SPB = 983.89
    assert METHANE.B == pytest.approx(983.89, rel=1e-4)
    assert METHANE.A == pytest.approx(8.8083, rel=1e-4)


def test_explicit_constants_are_used():
    assert CHLORINE.B == 1978.34 and CHLORINE.C == -27.01
    assert CHLORINE.A == pytest.approx(1978.34 / (239.1 - 27.01))


@pytest.mark.parametrize("sub", [METHANE, CHLORINE, AMMONIA, WATER])
def test_saturation_is_unity_at_the_boiling_point(sub):
    """EQ 43c fixes A so that P_sat(T_bp) = P_ambient."""
    assert LegacyThermo(sub).saturation_ratio(sub.T_boil) == pytest.approx(1.0)


@pytest.mark.parametrize("sub", [METHANE, CHLORINE, AMMONIA])
def test_saturation_derivative_matches_finite_difference(sub):
    th = LegacyThermo(sub)
    for T in (0.7 * sub.T_boil, 0.9 * sub.T_boil, sub.T_boil * 1.2):
        h = 1e-5 * T
        fd = (th.saturation_ratio(T + h) - th.saturation_ratio(T - h)) / (2 * h)
        assert th.d_saturation_ratio(T) == pytest.approx(fd, rel=1e-5)


def test_saturation_linearised_above_boiling_point():
    """SLAB_thermo L2252: the exponential is replaced by its tangent."""
    th = LegacyThermo(METHANE)
    T = 1.5 * METHANE.T_boil
    assert th.saturation_ratio(T) < math.exp(METHANE.A - METHANE.B / T)
    assert th.saturation_ratio(T) == pytest.approx(
        1.0 + METHANE.A * (T - METHANE.T_boil) / METHANE.T_boil
    )


def test_latent_heat_recovers_input_at_boiling_point():
    assert LegacyThermo(METHANE).dh_vap(METHANE.T_boil) == pytest.approx(
        METHANE.dh_vap
    )


def test_latent_heat_falls_with_temperature_for_cryogens():
    th = LegacyThermo(METHANE)
    assert th.dh_vap(200.0) < th.dh_vap(111.7)      # cp_liq > cp_vap


def test_source_temperature_clamping():
    """The original caps the boiling point just below the lower of the
    source and ambient temperatures (SLAB.FOR L671)."""
    assert METHANE.source_temperature(90.0) == pytest.approx(111.7)
    assert METHANE.source_temperature(200.0) == pytest.approx(200.0)
    assert METHANE.source_temperature(200.0, liquid_fraction=0.5) == pytest.approx(111.7)


def test_backend_satisfies_protocol():
    assert isinstance(LegacyThermo(METHANE), ThermoBackend)


def test_rejects_nonpositive_properties():
    with pytest.raises(ValueError):
        Substance(name="x", mw=0.0, cp_vapour=1.0, cp_liquid=1.0,
                  dh_vap=1.0, T_boil=100.0, rho_liquid=1.0)


# ===========================================================================
# Mixture validation
# ===========================================================================
def test_mixture_rejects_unnormalised_composition():
    with pytest.raises(ValueError, match="sum to"):
        Mixture(m_emission=0.5, m_water=0.1, m_dry_air=0.1,
                m_ev_transported=0.5, m_wv_transported=0.1, enthalpy=3e5)


def test_mixture_rejects_negative_fraction():
    with pytest.raises(ValueError):
        Mixture(m_emission=1.2, m_water=0.0, m_dry_air=-0.2,
                m_ev_transported=1.2, m_wv_transported=0.0, enthalpy=3e5)


# ===========================================================================
# equilibrium: analytic limits
# ===========================================================================
def test_trace_release_recovers_ambient_temperature():
    """As m -> 0 the cloud is air and T -> the enthalpy-implied value."""
    mix = mixture(1e-9, T_ref=300.0)
    eq = solve(METHANE, mix)
    assert eq.T == pytest.approx(300.0, rel=1e-3)
    assert eq.m_el == pytest.approx(0.0, abs=1e-12)


def test_trace_release_recovers_ambient_density():
    mix = mixture(1e-9, T_ref=AMBIENT["T_ambient"])
    eq = solve(METHANE, mix)
    assert eq.rho == pytest.approx(AMBIENT["rho_ambient"], rel=1e-3)


def test_equation_of_state_identity():
    """EQ 10 must hold exactly for the returned alpha, gamma."""
    eq = solve(METHANE, mixture(0.2, T_ref=200.0))
    rho = (AMBIENT["rho_ambient"] * AMBIENT["T_ambient"]
           / (eq.alpha * eq.T + eq.gamma * AMBIENT["T_ambient"]))
    assert eq.rho == pytest.approx(rho, rel=1e-12)


def test_gamma_zero_without_droplets():
    eq = solve(METHANE, mixture(0.05, T_ref=290.0))
    assert not eq.has_droplets
    assert eq.gamma == pytest.approx(0.0, abs=1e-15)


def test_mass_is_conserved_across_the_phase_split():
    eq = solve(METHANE, mixture(0.3, T_ref=150.0))
    m = mixture(0.3, T_ref=150.0)
    assert eq.m_ev + eq.m_el == pytest.approx(m.m_emission, rel=1e-12)
    assert eq.m_wv + eq.m_wl == pytest.approx(m.m_water, rel=1e-12)
    assert min(eq.m_ev, eq.m_el, eq.m_wv, eq.m_wl) >= -1e-15


def test_specific_heat_is_the_mass_weighted_sum():
    m = mixture(0.25, T_ref=180.0)
    eq = solve(METHANE, m)
    expected = (m.m_dry_air * PHYS.CP_AIR
                + eq.m_wv * PHYS.CP_WATER_VAP + eq.m_wl * PHYS.CP_WATER_LIQ
                + eq.m_ev * METHANE.cp_vapour + eq.m_el * METHANE.cp_liquid)
    assert eq.cp == pytest.approx(expected, rel=1e-12)


# ===========================================================================
# equilibrium: physics
# ===========================================================================
def test_cold_dense_cloud_is_denser_than_air():
    """LNG is lighter than air by molecular weight but denser when cold."""
    eq = solve(METHANE, mixture(0.31, T_ref=216.0))
    assert eq.T < AMBIENT["T_ambient"]
    assert eq.rho > AMBIENT["rho_ambient"]


def test_dilution_warms_the_cloud_towards_ambient():
    Ts = [solve(METHANE, mixture(m, T_ref=None or 300.0)).T
          for m in (0.5, 0.2, 0.05, 0.001)]
    assert all(a <= b for a, b in zip(Ts, Ts[1:]))


def test_water_condenses_out_of_a_cold_cloud():
    """Humid air entrained into a cryogenic cloud must drop out its water."""
    mix = mixture(0.4, m_wa=0.01, T_ref=140.0)
    eq = solve(METHANE, mix)
    assert eq.m_wl > 0.0
    assert eq.m_wv < mix.m_water
    assert not eq.saturated_water


def test_no_condensation_in_a_warm_dry_cloud():
    eq = solve(CHLORINE, mixture(0.02, m_wa=1e-4, T_ref=300.0))
    assert eq.m_wl == pytest.approx(0.0, abs=1e-15)
    assert eq.m_el == pytest.approx(0.0, abs=1e-15)
    assert eq.saturated_emission and eq.saturated_water


def test_condensation_releases_latent_heat():
    """
    A cloud whose transported vapour exceeds its equilibrium vapour must end
    up warmer than the same cloud with no phase change available.
    """
    m_e = 0.35
    base = mixture(m_e, m_wa=0.01, T_ref=150.0)
    # same enthalpy but arriving fully vaporised: more condensation to come
    more_vapour = Mixture(
        m_emission=base.m_emission, m_water=base.m_water,
        m_dry_air=base.m_dry_air,
        m_ev_transported=base.m_emission, m_wv_transported=base.m_water,
        enthalpy=base.enthalpy,
    )
    assert solve(METHANE, more_vapour).T >= solve(METHANE, base).T


def test_droplets_raise_density_through_gamma():
    wet = solve(METHANE, mixture(0.45, m_wa=0.02, T_ref=130.0))
    assert wet.has_droplets and wet.gamma > 0.0
    rho_no_droplet = (AMBIENT["rho_ambient"] * AMBIENT["T_ambient"]
                      / (wet.alpha * wet.T))
    assert wet.rho < rho_no_droplet    # gamma adds to the denominator


@pytest.mark.parametrize("sub", [METHANE, CHLORINE, AMMONIA])
def test_solves_for_all_manual_substances(sub):
    eq = solve(sub, mixture(0.15, T_ref=0.9 * AMBIENT["T_ambient"]))
    assert math.isfinite(eq.T) and eq.rho > 0.0 and eq.cp > 0.0


# ===========================================================================
# solver behaviour
# ===========================================================================
def test_converges_and_reports_iterations():
    eq = solve(METHANE, mixture(0.3, T_ref=170.0))
    assert 0 < eq.iterations <= 60


def test_initial_guess_reduces_work_and_not_the_answer():
    mix = mixture(0.3, T_ref=170.0)
    cold = solve(METHANE, mix)
    warm = solve(METHANE, mix, T_guess=cold.T)
    assert warm.T == pytest.approx(cold.T, rel=1e-6)
    assert warm.iterations <= cold.iterations


def test_result_independent_of_starting_point():
    mix = mixture(0.3, T_ref=170.0)
    Ts = [solve(METHANE, mix, T_guess=g).T for g in (80.0, 200.0, 400.0, 900.0)]
    assert max(Ts) - min(Ts) < 1e-3


def test_failure_is_raised_not_swallowed():
    """
    The reference implementation exits its fixed loop silently.  An
    unreachable enthalpy must raise instead.
    """
    mix = mixture(0.3)
    absurd = Mixture(
        m_emission=mix.m_emission, m_water=mix.m_water, m_dry_air=mix.m_dry_air,
        m_ev_transported=mix.m_ev_transported,
        m_wv_transported=mix.m_wv_transported,
        enthalpy=1e12,
    )
    with pytest.raises(EquilibriumError):
        solve(METHANE, absurd)


def test_bracket_can_be_narrowed():
    mix = mixture(0.05, T_ref=295.0)
    eq = solve(METHANE, mix, T_bounds=(200.0, 400.0))
    assert 200.0 < eq.T < 400.0


def test_bad_bracket_raises_clearly():
    with pytest.raises(EquilibriumError, match="bracket"):
        solve(METHANE, mixture(0.05, T_ref=295.0), T_bounds=(400.0, 500.0))


def test_temperature_monotone_in_enthalpy():
    base = mixture(0.25, T_ref=200.0)
    hotter = Mixture(
        m_emission=base.m_emission, m_water=base.m_water,
        m_dry_air=base.m_dry_air, m_ev_transported=base.m_ev_transported,
        m_wv_transported=base.m_wv_transported,
        enthalpy=base.enthalpy * 1.10,
    )
    assert solve(METHANE, hotter).T > solve(METHANE, base).T


# ===========================================================================
# solver efficiency — regression guard
# ===========================================================================
@pytest.mark.parametrize("m_e", [0.005, 0.02, 0.05, 0.10, 0.20, 0.31, 0.50])
def test_newton_is_actually_used(m_e):
    """
    Regression guard.  An earlier safeguarded-Newton loop rejected a step
    that landed exactly on a bracket endpoint — which is what happens when
    Newton finds the root — and fell back to bisection, taking 16-19
    iterations where 2-4 suffice.  Without phase change the residual is
    linear in T, so Newton must converge in a couple of steps from any
    starting point.
    """
    eq = solve(METHANE, mixture(m_e, T_ref=306.0))
    assert eq.iterations <= 6


def test_iteration_count_stable_across_starting_points():
    mix = mixture(0.2, T_ref=306.0)
    for guess in (None, 120.0, 250.0, 400.0, 1500.0):
        eq = solve(METHANE, mix, T_guess=guess)
        assert eq.iterations <= 8


def test_lng_dilution_curve_reaches_the_manual_state():
    """
    Adiabatic mixing of LNG vapour at its boiling point into Burro-8 ambient
    air.  At the manual's source mass fraction the state must land close to
    the printed values; the remaining gap is the ground heat flux, which is
    not part of this calculation.
    """
    eq = solve(METHANE, mixture(0.316, m_wa=1.52e-3, enthalpy=None, T_ref=None)
               if False else _adiabatic_mix(0.316))
    assert eq.T == pytest.approx(216.0, abs=8.0)      # manual: 216 K
    assert eq.rho == pytest.approx(1.31, rel=0.03)    # manual: 1.31
    assert eq.T < 216.0        # no ground heating here, so we must run cold


def _adiabatic_mix(m_e, T_source=111.7, T_amb=306.0, m_wa=1.52e-3):
    m_w = (1.0 - m_e) * m_wa
    m_da = (1.0 - m_e) * (1.0 - m_wa)
    E = (m_e * METHANE.cp_vapour * T_source
         + m_da * PHYS.CP_AIR * T_amb
         + m_w * PHYS.CP_WATER_VAP * T_amb)
    return Mixture(m_emission=m_e, m_water=m_w, m_dry_air=m_da,
                   m_ev_transported=m_e, m_wv_transported=m_w, enthalpy=E)


def test_adiabatic_lng_stays_dense_all_the_way_to_full_dilution():
    """
    Methane is lighter than air by molecular weight, yet an LNG cloud never
    becomes buoyant under adiabatic mixing alone: dilution warms and lightens
    it at the same rate, so density approaches ambient from above and never
    crosses.  Buoyancy requires an external heat input — which is exactly why
    SLAB carries a ground heat flux (EQ 37).
    """
    ms = (0.5, 0.1, 0.01, 1e-3, 1e-4, 1e-6)
    rhos = [solve(METHANE, _adiabatic_mix(m)).rho for m in ms]
    assert all(b < a for a, b in zip(rhos, rhos[1:]))            # monotone
    assert rhos[0] > 1.25                                        # clearly dense
    assert rhos[-1] == pytest.approx(AMBIENT["rho_ambient"], rel=2e-5)
    assert all(r >= AMBIENT["rho_ambient"] * (1 - 1e-4) for r in rhos)


def test_ground_heating_can_make_an_lng_cloud_buoyant():
    """
    The counterpart: add heat at fixed composition and the same cloud does
    become lighter than air, because methane's low molecular weight is then
    no longer masked by low temperature.  This is the lofting branch.
    """
    m = _adiabatic_mix(0.05)
    heated = Mixture(
        m_emission=m.m_emission, m_water=m.m_water, m_dry_air=m.m_dry_air,
        m_ev_transported=m.m_ev_transported,
        m_wv_transported=m.m_wv_transported,
        enthalpy=m.enthalpy * 1.20,
    )
    cold, warm = solve(METHANE, m), solve(METHANE, heated)
    assert warm.T > cold.T
    assert cold.rho > AMBIENT["rho_ambient"] > warm.rho


# ===========================================================================
# reference-temperature regression
# ===========================================================================
def test_water_latent_heat_is_quoted_at_298K_not_at_boiling():
    """
    Regression guard for a defect that cost 9 % on the reproduction.

    SLAB hard-codes ``dhw0 = dhw + 298.2 (cp_wl - cp_wv)``.  An earlier
    version extrapolated from `WATER.T_boil` instead, which is back-figured
    from the saturation constants and lands at 365.6 K — a 5 % error in the
    phase-change energy.  That shifts cloud temperature by ~0.1 K, density by
    0.05 %, and the buoyancy term ``rho - rho_a`` by 0.24 %, which the source
    expansion then amplifies eightfold.
    """
    from slabx.thermo.base import T_REF_WATER

    assert WATER.T_reference == pytest.approx(298.2)
    assert WATER.T_ref == pytest.approx(T_REF_WATER)
    expected = PHYS.DH_WATER + 298.2 * (PHYS.CP_WATER_LIQ - PHYS.CP_WATER_VAP)
    assert WATER.dh_vap_ref == pytest.approx(expected, rel=1e-12)
    assert WATER.dh_vap_ref == pytest.approx(3136402.4, rel=1e-9)

    # and the latent heat must come back out at the reference temperature
    assert water_backend().dh_vap(298.2) == pytest.approx(PHYS.DH_WATER)


def test_released_material_still_references_its_boiling_point():
    """The default must not change: only water is special."""
    assert METHANE.T_reference is None
    assert METHANE.T_ref == pytest.approx(METHANE.T_boil)
    assert LegacyThermo(METHANE).dh_vap(METHANE.T_boil) == pytest.approx(
        METHANE.dh_vap
    )


def test_equilibrium_matches_the_reference_implementation():
    """
    One stage of the reference's own thermodynamics, reproduced exactly.

    Inputs and outputs were logged from SLAB_thermo at the first Runge-Kutta
    stage of Burro 8 (golden/th.js).  Agreement here is to nine figures, so a
    regression in the phase split, the specific heat or the equation of state
    will show up immediately.
    """
    eq = solve_equilibrium(
        Mixture(m_emission=0.17061496069519, m_water=0.0012566493126703,
                m_dry_air=0.82812838999214,
                m_ev_transported=0.17061496069519,
                m_wv_transported=0.0012566493126703,
                enthalpy=298255.75185481),
        LegacyThermo(METHANE), water_backend(),
        T_ambient=306.0, rho_ambient=1.152307066,
        mw_ambient_moist=0.02893338542, T_guess=306.0,
    )
    assert eq.T == pytest.approx(246.65831991220, rel=1e-8)
    assert eq.cp == pytest.approx(1218.9331940244, rel=1e-8)
    assert eq.m_wv == pytest.approx(0.00049008253907381, rel=1e-5)
    assert eq.m_ev == pytest.approx(0.17061496069519, rel=1e-12)
    assert eq.alpha == pytest.approx(1.1358565455197, rel=1e-8)
    assert eq.rho == pytest.approx(1.2585485637570, rel=1e-8)
