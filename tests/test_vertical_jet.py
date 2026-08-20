"""
Vertical jet and the plume-rise submodel (EQ 45-47).

The only release type whose cloud is *elevated* when dispersion starts, so
it is the only one that exercises the height-of-maximum-concentration logic
and the reciprocal combination of momentum and density rise.
"""

import math

import numpy as np
import pytest

from slabx.coefficients import preset
from slabx.core.plume import run_dispersion
from slabx.core.trajectory import Mode
from slabx.core.vertical_jet import (
    BuoyantRiseNotImplemented,
    VerticalJet,
)
from slabx.post.concentration import concentration_field, height_of_maximum
from slabx.submodels.atmosphere import Atmosphere
from slabx.thermo.base import LegacyThermo, Substance, water_backend

CL2 = Substance(name="Cl2", mw=0.070906, cp_vapour=498.1, cp_liquid=926.3,
                dh_vap=287840.0, T_boil=239.1, rho_liquid=1574.0,
                sat_B=1978.34, sat_C=-27.01)


def stack_atm():
    return Atmosphere(u_ref=1.0, z_ref=10.0, T=276.0, rh=30.0, z0=0.1,
                      stability=4.0, legacy_stability_bug=True,
                      legacy_unstable_profile=True)


def cl2_jet(**kw):
    d = dict(substance=CL2, rate=3.33, area=0.02, duration=300.0,
             liquid_fraction=0.88, height=1.0)
    d.update(kw)
    return VerticalJet(**d)


@pytest.fixture(scope="module")
def stack_run():
    atm = stack_atm()
    src = cl2_jet()
    tr, _ = run_dispersion(src, atm, LegacyThermo(CL2), water_backend(),
                           x_max=10000.0, n_puff_steps=50,
                           coeffs=preset("ermak90"))
    return tr, atm, src


# ===========================================================================
# exit conditions and rise
# ===========================================================================
def test_exit_velocity_matches_the_reference():
    """The original prints ws = 5.6215 m/s for this deck (SLAB.FOR L594)."""
    assert cl2_jet().w_source(stack_atm()) == pytest.approx(5.6215, rel=1e-4)


def test_rise_height_matches_the_manual():
    """
    Manual section 4.4: the plume rises from z_c = 1.00 m at x = 1.00 m to a
    maximum of z_c = 2.24 m at x = 1.01 m.
    """
    r = cl2_jet().rise(stack_atm())
    assert r.z_c == pytest.approx(2.24, abs=0.02)
    assert r.x_rise == pytest.approx(0.01, abs=0.005)


def test_dense_rise_is_limited_by_the_smaller_of_the_two():
    """
    EQ 46a combines the two reciprocally, so the result is below both and
    dominated by whichever is smaller — here the density limit.
    """
    r = cl2_jet().rise(stack_atm())
    assert r.h_rise < min(r.h_momentum, r.h_dense)
    assert r.h_dense < r.h_momentum
    assert r.h_rise == pytest.approx(
        r.h_momentum * r.h_dense / math.hypot(r.h_momentum, r.h_dense), rel=1e-12
    )


def test_faster_jet_rises_higher():
    atm = stack_atm()
    rises = [cl2_jet(area=a).rise(atm).h_rise for a in (0.08, 0.04, 0.02, 0.01)]
    assert all(b > a for a, b in zip(rises, rises[1:]))


def test_rise_is_capped_by_the_mixing_layer():
    atm = stack_atm()
    tall = cl2_jet(height=atm.h_mix - 0.5)
    assert tall.rise(atm).z_c <= atm.h_mix + 1e-9


def test_buoyant_branch_is_reported_not_approximated():
    """
    A release lighter than air needs the Briggs branch of EQ 46a, which is
    not implemented.  It must raise rather than silently take the dense path.
    """
    light = Substance(name="H2", mw=0.002016, cp_vapour=14300.0,
                      cp_liquid=9700.0, dh_vap=451900.0, T_boil=20.3,
                      rho_liquid=71.0)
    jet = VerticalJet(substance=light, rate=0.5, area=0.02, duration=300.0,
                      T_source=290.0, height=1.0)
    with pytest.raises(BuoyantRiseNotImplemented):
        jet.rise(stack_atm())


# ===========================================================================
# the rise region, EQ 47
# ===========================================================================
def test_rise_profile_spans_the_rise_and_ends_at_the_initial_state():
    atm = stack_atm()
    jet = cl2_jet()
    rows = jet.rise_profile(atm)
    end = jet.initial_state(atm, dx=1.0)
    assert rows[0]["x"] == pytest.approx(1.0)
    assert rows[-1]["x"] == pytest.approx(end.x_start, rel=1e-9)
    assert rows[-1]["h"] == pytest.approx(end.h, rel=1e-9)
    assert rows[-1]["b_half"] == pytest.approx(end.b_half, rel=1e-9)
    assert rows[0]["mass_frac"] == pytest.approx(1.0)


def test_rise_profile_reaches_the_maximum_height_at_the_top():
    atm = stack_atm()
    jet = cl2_jet()
    r = jet.rise(atm)
    z = [row["z_c"] for row in jet.rise_profile(atm)]
    assert z[0] == pytest.approx(jet.height, rel=0.02)
    assert max(z) == pytest.approx(r.z_c, rel=0.02)


# ===========================================================================
# height of maximum concentration
# ===========================================================================
def test_grounded_cloud_peaks_at_the_ground():
    """z_c <= sigma has only the trivial root."""
    assert height_of_maximum(np.array([0.0, 0.1, 0.9]),
                             np.array([1.0, 1.0, 1.0])).max() == 0.0


def test_lifted_cloud_peaks_between_ground_and_centre():
    z_c = np.array([3.0, 10.0])
    sigma = np.array([1.0, 1.0])
    zm = height_of_maximum(z_c, sigma)
    assert np.all(zm > 0) and np.all(zm <= z_c)
    # the fixed point must be satisfied
    E = np.exp(-2.0 * z_c * zm / sigma**2)
    assert np.allclose(zm, z_c * (1.0 - E) / (1.0 + E), rtol=1e-9)


def test_higher_cloud_peaks_higher():
    sigma = np.full(4, 1.0)
    zm = height_of_maximum(np.array([1.5, 3.0, 6.0, 12.0]), sigma)
    assert np.all(np.diff(zm) > 0)


def test_max_evaluation_never_below_a_fixed_plane(stack_run):
    """C(x,0,z_m) is by definition the largest value over height."""
    tr, atm, src = stack_run
    kw = dict(t_avg=1.0, t_release=src.duration)
    top = concentration_field(tr, atm, z="max", **kw).peak
    for z in (0.0, 1.0, 5.0):
        assert np.all(top >= concentration_field(tr, atm, z=z, **kw).peak - 1e-15)


# ===========================================================================
# the run
# ===========================================================================
def test_trajectory_starts_at_the_stack_and_is_valid(stack_run):
    tr, _, src = stack_run
    assert tr.x[0] == pytest.approx(1.0)
    assert tr.z_c[0] == pytest.approx(src.height, rel=0.02)
    assert tr.mass_frac[0] == pytest.approx(1.0)
    assert np.all(np.isfinite(tr.vol_frac))


def test_dense_cloud_slumps_back_to_the_ground(stack_run):
    """
    Chlorine is denser than air at every concentration, so whatever the jet
    lifts must come back down; SLAB's own output shows z_c decaying to
    millimetres within a few tens of metres.
    """
    tr, *_ = stack_run
    assert tr.z_c.max() > 1.5 * tr.z_c[0]
    assert tr.z_c[-1] < 0.05 * tr.z_c.max()


def test_transition_at_the_release_duration(stack_run):
    """
    EQ 29b puts the plume-to-puff switch at t_sd, measured from the start of
    the release.  The trajectory's clock starts when the jet leaves the
    stack, so for a vertical jet the reported time carries the rise duration
    on top — the same offset every other row has.

    The invariant tested is consistency: the reported transition time must be
    the trajectory's own time at the transition distance, and must exceed the
    release duration by the rise.  Comparing against `src.duration` alone
    held only while the transition row was the one row that had not been
    offset (see `test_the_transition_row_carries_the_rise_duration`).
    """
    import numpy as np

    tr, _, src = stack_run
    t_switch = tr.meta["transition_t"]
    x_switch = tr.meta["transition_x"]

    assert t_switch >= src.duration
    assert t_switch == pytest.approx(float(np.interp(x_switch, tr.x, tr.t)),
                                     rel=1e-6, abs=1e-6)
    # the rise is short next to a 300 s release, so the offset is small
    assert t_switch - src.duration < 0.05 * src.duration


def test_no_source_expansion_for_a_jet(stack_run):
    """
    The cloud leaves the rise lofted, so alpha_g starts at zero and EQ 4d
    cannot fail; it only switches on once the cloud settles, by which point
    the source region is long behind.
    """
    from slabx.core.plume import gravity_coefficient

    tr, atm, _ = stack_run
    assert gravity_coefficient(True, tr.rho[0], atm.rho) == 0.0
    assert tr.meta["source_widening"] == 0.0


def test_risen_cloud_uses_the_point_wind_not_a_depth_average():
    """
    The vertical jet is the exception to the cloud-top averaging rule: by the
    top of its rise the cloud is already travelling with the wind at that
    level, so SLAB takes the point value U_a(z_cr) rather than an average
    from the ground (SLAB.FOR L1154).  Averaging instead understates the
    wind and slows the cloud.
    """
    atm = stack_atm()
    jet = cl2_jet()
    s = jet.initial_state(atm, dx=1.0)
    assert s.u_ambient_mean == pytest.approx(
        atm.wind_speed(jet.rise(atm).z_c), rel=1e-9
    )


def test_thermodynamics_is_solved_at_the_top_of_the_rise():
    """
    Regression guard.  The jet is roughly half diluted by the time it tops
    out, so it is no longer at its exit temperature: droplets have evaporated
    and it has cooled far below the boiling point.  Reusing the exit values
    inflates the density tenfold and shrinks the cloud cross-section by the
    same factor.
    """
    atm = stack_atm()
    jet = cl2_jet()
    bare = jet.initial_state(atm, dx=1.0)
    full = jet.initial_state(atm, dx=1.0, emission=LegacyThermo(CL2),
                             water=water_backend())

    assert full.T < CL2.T_boil - 20.0            # cooled by evaporation
    assert full.rho < 0.2 * bare.rho             # and much lighter
    assert full.b_half > 2.5 * bare.b_half       # so the cloud is far bigger
    # and it must still be denser than air, or it would not slump
    assert full.rho > 2.0 * atm.rho


def test_the_transition_row_carries_the_rise_duration():
    """
    Regression: a vertical jet whose release duration is comparable to its
    rise time used to raise "t must be non-decreasing".

    The plume-to-puff transition row takes its time from EQ 29b, which is in
    the plume's own clock.  Every other row had the rise duration added, so
    that one row stepped backwards.  Cases where the rise was negligible, or
    where t_sd was far larger than it, happened to stay monotonic — which is
    why the five manual decks and the hand-written tests all passed.

    Found by fuzzing against the compiled Fortran: four of ten sampled
    vertical jets hit it.
    """
    import numpy as np

    sub = Substance(name="Propane", mw=0.044096, cp_vapour=1670.0,
                    cp_liquid=2520.0, dh_vap=425700.0, T_boil=231.1,
                    rho_liquid=580.0)
    atm = Atmosphere(u_ref=4.0, z_ref=3.0, T=290.0, rh=50.0, z0=1e-3,
                     stability="D")
    src = VerticalJet(substance=sub, rate=5.0, area=0.01, duration=30.0,
                      liquid_fraction=0.5, height=1.5)

    traj, _ = run_dispersion(src, atm, LegacyThermo(sub), water_backend(),
                             x_max=500.0, n_puff_steps=40)

    assert np.all(np.diff(traj.t) >= 0.0), "time steps backwards"
    assert np.all(np.diff(traj.x) >= 0.0)
    # the rise is genuinely there, so the clock does not start at zero
    assert traj.t[-1] > 0.0


def test_a_rise_longer_than_the_domain_does_not_break_the_grid():
    """
    Regression: a jet whose plume rise carries it past the requested domain
    used to raise "x must be non-decreasing".

    The output grid returned `x_max` when the domain was already behind the
    start, which put a row upwind of the release.  Two separate paths could
    do it — a source region narrower than the start distance, and a rise
    longer than `x_max` — and both are reachable with a small orifice.

    Found by fuzzing against the Fortran, on an ethane jet through a
    0.0029 m2 orifice whose rise was 1465 m against a 405 m domain.  The
    deck is physically extreme, but a scope warning is the right answer to
    that, not a broken trajectory.
    """
    import numpy as np

    sub = Substance(name="Ethane", mw=0.030069, cp_vapour=1750.0,
                    cp_liquid=2440.0, dh_vap=489000.0, T_boil=184.6,
                    rho_liquid=544.0)
    atm = Atmosphere(u_ref=8.965, z_ref=3.0, T=299.70, rh=37.89,
                     z0=0.000114, stability="C")
    src = VerticalJet(substance=sub, rate=113.3557, area=0.00290,
                      duration=324.38, liquid_fraction=0.3030, height=0.3875)

    import warnings

    from slabx.scope import ScopeWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ScopeWarning)
        traj, _ = run_dispersion(src, atm, LegacyThermo(sub), water_backend(),
                                 x_max=405.36, n_puff_steps=40)

    assert np.all(np.diff(traj.x) >= 0.0), "distance steps backwards"
    assert np.all(np.diff(traj.t) >= 0.0)
    assert traj.x[0] > 0.0
