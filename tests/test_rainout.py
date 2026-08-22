"""
Droplet rainout, and its pre-registered predictions.

The submodel adds physics SLAB does not have: liquid that settles out of the
cloud onto the ground.  `submodels/rainout.py` registered five predictions
before it was wired in.  These tests check all five, and the negative
control (R1) is the one that matters most — if a two-phase submodel changes
a single-phase release, nothing else it does can be trusted.
"""

import math

import numpy as np
import pytest

pytest.importorskip("CoolProp")

import sys                                                        # noqa: E402
from pathlib import Path                                          # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from decks import CASES, build_source                  # noqa: E402
from slabx.coefficients import preset                             # noqa: E402
from slabx.core.plume import run_dispersion                       # noqa: E402
from slabx.post.concentration import concentration_field          # noqa: E402
from slabx.submodels.rainout import (                             # noqa: E402
    PREDICTIONS, WEBER_CRITICAL, droplet_diameter, rainout_rate,
    terminal_velocity,
)
from slabx.thermo.coolprop import CoolPropThermo, coolprop_water  # noqa: E402


def _run(case, rainout):
    sub = case.substance()
    atm = case.atmosphere(legacy=False)
    src = build_source(case, sub)
    em, wt = CoolPropThermo(sub, fluid=case.fluid), coolprop_water()
    traj, _ = run_dispersion(src, atm, em, wt, x_max=case.xffm,
                             t_max=case.t_max, n_puff_steps=40,
                             coeffs=preset("ermak90"), rainout=rainout)
    field = concentration_field(traj, atm, z="max", t_avg=case.tav,
                                t_release=case.tsd)
    return traj, field


def _mid(traj, field):
    x = math.sqrt(max(traj.x[traj.x > 0].min(), 1.0) * traj.x.max())
    c = float(np.exp(np.interp(x, field.x,
                               np.log(np.maximum(field.peak, 1e-30)))))
    return c, float(np.interp(x, traj.x, traj.rho))


@pytest.fixture(scope="module")
def runs():
    return {n: {ro: _run(CASES[n], ro) for ro in (False, True)}
            for n in ("burro8", "lng_instant", "dt4", "cl2_stack")}


# ===========================================================================
# the droplet chain
# ===========================================================================
def test_predictions_are_recorded():
    assert len(PREDICTIONS) == 5
    assert WEBER_CRITICAL == 12.0


def test_break_up_criterion_scales_correctly():
    """d ~ We sigma / (rho u^2): halving sigma halves d, doubling u quarters it."""
    d1 = droplet_diameter(0.03, 1.2, 20.0)
    assert droplet_diameter(0.015, 1.2, 20.0) == pytest.approx(d1 / 2)
    assert droplet_diameter(0.03, 1.2, 40.0) == pytest.approx(d1 / 4)
    assert droplet_diameter(0.03, 1.2, 0.0) == 0.01          # capped


def test_terminal_velocity_picks_the_right_regime():
    slow = terminal_velocity(2e-5, 600.0, 1.2)               # 20 um
    fast = terminal_velocity(2e-3, 600.0, 1.2)               # 2 mm
    assert slow.regime == "stokes" and slow.reynolds <= 1.0
    assert fast.regime in ("intermediate", "newton")
    assert fast.terminal_velocity > slow.terminal_velocity
    # Stokes: v ~ d^2
    assert terminal_velocity(4e-5, 600.0, 1.2).terminal_velocity \
        == pytest.approx(4.0 * slow.terminal_velocity, rel=0.2)


def test_R5_settling_falls_monotonically_with_exit_velocity():
    """
    The discriminating prediction on the droplet side: a faster jet makes
    smaller droplets, which settle more slowly.  A submodel that ignores
    droplet size would not show this.
    """
    v = [terminal_velocity(droplet_diameter(0.034, 1.2, u), 680.0,
                           1.2).terminal_velocity
         for u in (5.0, 10.0, 25.0, 50.0, 100.0)]
    assert all(b < a for a, b in zip(v, v[1:]))
    assert v[0] / v[-1] > 10.0


def test_rate_is_the_fall_time_over_the_travel_time():
    assert rainout_rate(0.5, 1.0, 10.0, 2.0) == pytest.approx(1.0 / 20.0)
    assert rainout_rate(0.0, 1.0, 10.0, 2.0) == 0.0          # no liquid
    assert rainout_rate(0.5, 0.0, 10.0, 2.0) == 0.0          # no settling


# ===========================================================================
# the registered predictions, end to end
# ===========================================================================
def test_R1_no_effect_on_a_single_phase_release(runs):
    """
    The negative control.  Burro is pure methane vapour and the instantaneous
    LNG release has no liquid either; neither may move at all.
    """
    for name in ("burro8", "lng_instant"):
        off = _mid(*runs[name][False])
        on = _mid(*runs[name][True])
        assert on[0] == pytest.approx(off[0], rel=1e-9), name
        assert on[1] == pytest.approx(off[1], rel=1e-9), name


def test_R2_effect_concentrates_in_the_two_phase_decks(runs):
    """
    And is larger for the ammonia jet than the chlorine stack: ammonia is
    released more slowly through a larger orifice, so its droplets are bigger
    and settle faster.
    """
    dt = abs(_mid(*runs["dt4"][True])[0] / _mid(*runs["dt4"][False])[0] - 1)
    cl = abs(_mid(*runs["cl2_stack"][True])[0]
             / _mid(*runs["cl2_stack"][False])[0] - 1)
    assert dt > 0.1
    assert dt > 2.0 * cl


def test_R3_concentration_falls_and_the_cloud_lightens(runs):
    """Removing liquid removes mass and the droplet-volume term of EQ 10."""
    off = _mid(*runs["dt4"][False])
    on = _mid(*runs["dt4"][True])
    assert on[0] < 0.8 * off[0]
    assert on[1] < off[1]


def test_R4_mass_is_conserved_between_cloud_and_pool(runs):
    """
    Everything that leaves arrives.  This is what separates a rainout model
    from a concentration fudge, and an earlier version failed it: a sign
    error sent the pool negative, it was clamped at zero, and the mass simply
    vanished — while the concentrations still moved, plausibly, in the wrong
    direction.
    """
    for name, case in (("dt4", CASES["dt4"]), ("cl2_stack", CASES["cl2_stack"])):
        traj, _ = runs[name][True]
        # Only past the source region: inside it the release is still adding
        # material, so the budget is against the fraction emitted so far
        # rather than against the whole rate.
        p = traj.is_plume & (traj.x > 2.0)
        total = (traj.mass_frac[p] * traj.R_flux[p] + traj.rained_out[p])
        assert np.allclose(total / (case.qs / 2.0), 1.0, rtol=0.03), name


def test_a_useful_fraction_actually_rains_out(runs):
    """
    If the rate were negligible the predictions would pass trivially.  For an
    81 % liquid ammonia jet a third of the release should reach the ground.
    """
    traj, _ = runs["dt4"][True]
    p = traj.is_plume
    assert traj.rained_out[p][-1] / (CASES["dt4"].qs / 2.0) > 0.2


# ===========================================================================
# calibration against an independent droplet measurement
# ===========================================================================
def test_smedis_equivalent_sources_are_available():
    """
    SMEDIS distributed, for Desert Tortoise, the distance at which no liquid
    remains in the plume together with the velocity, concentration and width
    there.  It is a droplet-lifetime measurement and carries no dispersion
    statistics, so calibrating the rainout rate against it and then testing
    concentrations against field data is not circular.
    """
    import csv
    from slabx.validation._data_access import require

    rows = {r["trial"]: r for r in
            csv.DictReader(open(require("smedis_equivalent_sources.csv")))}
    assert {"DT1", "DT2"} <= set(rows)
    assert float(rows["DT1"]["x_liquid_gone_m"]) == 51.0
    assert float(rows["DT1"]["velocity_m_s"]) == 7.5
    assert float(rows["DT1"]["molar_pct"]) == 13.0


def test_reaching_the_droplet_lifetime_target_needs_a_slower_rate():
    """
    The unscaled closure empties the plume of liquid at 28 m where SMEDIS
    puts it at 51; switching rainout off entirely puts it at 77.  The target
    lies between, so the removal rate is several times too fast and the
    efficiency coefficient exists to say so rather than to hide it.
    """
    from slabx.core.source import HorizontalJet
    from slabx.submodels.atmosphere import Atmosphere
    from slabx.thermo.base import Substance

    NH3 = Substance(name="NH3", mw=0.017031, cp_vapour=2045.9,
                    cp_liquid=4611.8, dh_vap=1170000.0, T_boil=239.57,
                    rho_liquid=603.0, sat_B=2976.01)
    atm = Atmosphere(u_ref=7.4, z_ref=2.0, T=306.0, rh=15.0, z0=0.003,
                     stability="D")
    src = HorizontalJet(substance=NH3, rate=81.0, area=0.93 * 81.0 / 107.9,
                        duration=126.0, liquid_fraction=0.80, height=0.79)
    em, wt = CoolPropThermo(NH3, fluid="Ammonia"), coolprop_water()

    def liquid_gone(eff):
        traj, _ = run_dispersion(
            src, atm, em, wt, x_max=800.0, n_puff_steps=30,
            coeffs=preset("ermak90").perturb(rainout_efficiency=eff, name="e"),
            rainout=eff > 0.0)
        liq = traj.mass_frac_emission_liquid
        k = int(np.argmax(liq < 1e-4)) if np.any(liq < 1e-4) else len(traj) - 1
        return float(traj.x[k])

    off, full = liquid_gone(0.0), liquid_gone(1.0)
    assert full < 51.0 < off                    # the target is bracketed
    assert liquid_gone(0.25) > full             # slowing it lengthens the plume


def test_velocity_and_concentration_at_the_target_already_agree():
    """
    The state at the point where the liquid runs out, against the four
    SMEDIS equivalent-source targets for DT1.

    **This test used to assert that velocity and concentration matched
    while the droplet lifetime did not, and read that as evidence the
    thermodynamics were right.** Correcting the water saturation pressure
    below its triple point (`docs/PREREG_water.md`) showed that reading was
    wrong:

    | | x [m] | u [m/s] | C [%] | T [K] |
    |---|---|---|---|---|
    | SMEDIS DT1 | 51.0 | 7.50 | 13.0 | 205 |
    | before | 77.1 (1.51) | 7.30 (0.97) | 13.46 (**1.04**) | 228 (1.11) |
    | after | 65.2 (**1.28**) | 7.53 (**1.00**) | 16.32 (1.26) | 218 (**1.06**) |

    Three of the four targets improved. Concentration moved away, and the
    reason is that the evaluation point itself moved: with water condensing
    as it should, the latent heat evaporates the droplets sooner, so the
    liquid runs out at 65 m instead of 77 m and the concentration is read
    closer to the source, where it is naturally higher.

    So the earlier agreement on concentration was not independent evidence
    that the thermodynamics were right. It held **while the droplet
    lifetime was 51 % long**, and it went away once that error was reduced
    -- one quantity compensating another, which is what this project keeps
    finding.

    The assertions below pin the corrected state and the direction of the
    change, not a target the model meets.
    """
    from slabx.core.source import HorizontalJet
    from slabx.submodels.atmosphere import Atmosphere
    from slabx.thermo.base import Substance

    NH3 = Substance(name="NH3", mw=0.017031, cp_vapour=2045.9,
                    cp_liquid=4611.8, dh_vap=1170000.0, T_boil=239.57,
                    rho_liquid=603.0, sat_B=2976.01)
    atm = Atmosphere(u_ref=7.4, z_ref=2.0, T=306.0, rh=15.0, z0=0.003,
                     stability="D")
    src = HorizontalJet(substance=NH3, rate=81.0, area=0.93 * 81.0 / 107.9,
                        duration=126.0, liquid_fraction=0.80, height=0.79)
    traj, _ = run_dispersion(src, atm, CoolPropThermo(NH3, fluid="Ammonia"),
                             coolprop_water(), x_max=800.0, n_puff_steps=30,
                             coeffs=preset("ermak90"), rainout=False)
    liq = traj.mass_frac_emission_liquid
    k = int(np.argmax(liq < 1e-4))
    # velocity now matches almost exactly
    assert traj.u[k] == pytest.approx(7.5, rel=0.05)          # SMEDIS

    # the droplet lifetime, the quantity the correction acts on, is closer
    # to the measured 51 m than the 77 m it used to give
    assert 55.0 < traj.x[k] < 70.0

    # and concentration is now high, because the point moved up-wind
    assert 15.0 < traj.vol_frac[k] * 100.0 < 18.0


# ===========================================================================
# finite-rate evaporation: relaxing local equilibrium
# ===========================================================================
def _dt1_setup():
    from slabx.core.source import HorizontalJet
    from slabx.submodels.atmosphere import Atmosphere
    from slabx.thermo.base import Substance

    NH3 = Substance(name="NH3", mw=0.017031, cp_vapour=2045.9,
                    cp_liquid=4611.8, dh_vap=1170000.0, T_boil=239.57,
                    rho_liquid=603.0, sat_B=2976.01)
    atm = Atmosphere(u_ref=7.4, z_ref=2.0, T=306.0, rh=15.0, z0=0.003,
                     stability="D")
    src = HorizontalJet(substance=NH3, rate=81.0, area=0.93 * 81.0 / 107.9,
                        duration=126.0, liquid_fraction=0.80, height=0.79)
    return NH3, atm, src


def test_evaporation_is_driven_by_concentration_not_temperature():
    """
    Regression guard for a falsified first attempt.

    Driving evaporation by heat conduction from ambient gives a positive
    feedback with the wrong sign — fewer droplets evaporate, so less latent
    heat is absorbed, so the cloud stays warm, so the driver shrinks.  On
    Desert Tortoise it left droplets alive at 300 m where SMEDIS puts them
    gone at 51, and the cloud at 299 K against a measured 205.

    The mass-transfer form is largest exactly where the cloud is coldest and
    most dilute, which is the stabilising feedback.
    """
    from slabx.submodels.rainout import evaporation_rate

    # unsaturated surroundings drive evaporation
    fast = evaporation_rate(5e-4, y_surface=0.35, y_bulk=0.10,
                            rho_gas=2.0, rho_liquid=600.0, reynolds=50.0)
    slow = evaporation_rate(5e-4, y_surface=0.35, y_bulk=0.30,
                            rho_gas=2.0, rho_liquid=600.0, reynolds=50.0)
    assert fast > slow > 0.0
    # saturated or supersaturated: no net evaporation
    assert evaporation_rate(5e-4, 0.20, 0.20, 2.0, 600.0) == 0.0
    assert evaporation_rate(5e-4, 0.20, 0.40, 2.0, 600.0) == 0.0
    # convection accelerates it
    assert evaporation_rate(5e-4, 0.35, 0.10, 2.0, 600.0, reynolds=100.0) \
        > evaporation_rate(5e-4, 0.35, 0.10, 2.0, 600.0, reynolds=0.0)


def test_finite_rate_tracks_local_equilibrium():
    """
    The result: SLAB's local-equilibrium assumption is adequate here.

    Integrating the droplet history explicitly, with the mass-transfer
    d-squared law and Ranz-Marshall convection, gives nearly the same liquid
    fraction and temperature as assuming instantaneous equilibrium.  The
    droplets are small enough and the surroundings unsaturated enough that
    the kinetics do not bind, so the assumption is vindicated rather than
    merely unexamined — and finite-rate evaporation is eliminated as an
    explanation for the remaining discrepancy against SMEDIS.
    """
    from slabx.core.plume import integrate_plume

    NH3, atm, src = _dt1_setup()
    em, wt = CoolPropThermo(NH3, fluid="Ammonia"), coolprop_water()
    runs = {k: integrate_plume(src, atm, em, wt, x_max=300.0,
                               coeffs=preset("ermak90"),
                               kinetic_evaporation=k)
            for k in (False, True)}

    for x in (5.0, 15.0, 30.0, 51.0):
        i = int(np.argmin(np.abs(runs[False].x - x)))
        j = int(np.argmin(np.abs(runs[True].x - x)))
        assert runs[True].T[j] == pytest.approx(runs[False].T[i], abs=6.0)
        assert abs(runs[True].mass_frac_emission_liquid[j]
                   - runs[False].mass_frac_emission_liquid[i]) < 0.07


def test_the_kinetic_constraint_is_actually_exercised():
    """
    Without this the previous test would pass trivially: a submodel that
    never fires agrees with everything.  The droplet area has to shrink.
    """
    from slabx.core.plume import integrate_plume

    NH3, atm, src = _dt1_setup()
    traj = integrate_plume(src, atm, CoolPropThermo(NH3, fluid="Ammonia"),
                           coolprop_water(), x_max=60.0,
                           coeffs=preset("ermak90"), kinetic_evaporation=True)
    assert traj.meta["kinetic_evaporation"] is True
    liq = traj.mass_frac_emission_liquid
    assert liq[0] == pytest.approx(0.80, rel=0.02)
    assert liq[-1] < 0.3 * liq[0]


def test_finite_rate_leaves_single_phase_releases_alone():
    """The negative control again, for the second submodel."""
    from slabx.core.plume import run_dispersion

    case = CASES["burro8"]
    sub = case.substance()
    atm = case.atmosphere(legacy=False)
    src = build_source(case, sub)
    em, wt = CoolPropThermo(sub, fluid=case.fluid), coolprop_water()
    out = []
    for k in (False, True):
        traj, _ = run_dispersion(src, atm, em, wt, x_max=case.xffm,
                                 n_puff_steps=30, coeffs=preset("ermak90"),
                                 kinetic_evaporation=k)
        out.append(traj.vol_frac[len(traj) // 2])
    assert out[1] == pytest.approx(out[0], rel=1e-9)
