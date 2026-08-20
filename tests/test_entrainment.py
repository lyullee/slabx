"""
Tier-0/1 verification of the entrainment and flux submodel.

Checks structural identities, physical limits and monotonicity.  Nothing
here compares against experimental data.
"""

import math

import pytest

from slabx.coefficients import COEFFS, PHYS, preset
from slabx.submodels.atmosphere import Atmosphere
from slabx.submodels.entrainment import (
    SQRT3,
    CloudLocal,
    entrainment,
    fluxes,
    friction,
)


def atm(stability="D", **kw):
    d = dict(u_ref=4.0, z_ref=10.0, T=290.0, rh=50.0, z0=0.02, stability=stability)
    d.update(kw)
    return Atmosphere(**d)


def cloud(**kw):
    """A grounded, denser-than-air, slower-than-ambient plume."""
    d = dict(h=3.0, b_half=25.0, b_half_x=1.0, z_c=0.0, u=2.0,
             T=270.0, rho=1.6, cp=1000.0, v_g=0.4)
    d.update(kw)
    return CloudLocal(**d)


def _fr(a, cl, ubar=None, **kw):
    return friction(cl, a, u_bar_ambient=ubar or a.wind_speed(cl.h), **kw)


# ===========================================================================
# geometry helpers
# ===========================================================================
def test_grounded_vs_lofted_classification():
    assert not cloud(z_c=0.0).is_lofted
    assert cloud(z_c=0.0).h_top == pytest.approx(3.0)
    hi = cloud(z_c=10.0)
    assert hi.is_lofted
    assert hi.h_top == pytest.approx(11.5)      # z_c + h/2


# ===========================================================================
# friction velocity structure, EQ 35e
# ===========================================================================
def test_ustar_is_sum_of_three_components():
    a = atm()
    f = _fr(a, cloud())
    assert f.u_star**2 == pytest.approx(
        f.u_mg_star**2 + f.u_mh_star_sq + f.u_t_sq, rel=1e-12
    )


def test_all_components_positive_for_dense_cold_cloud():
    f = _fr(atm(), cloud())
    assert f.u_mg_star > 0 and f.u_mh_star_sq > 0 and f.u_t_sq > 0


def test_no_thermal_term_when_cloud_warmer_than_ground():
    a = atm(T=280.0)
    assert _fr(a, cloud(T=300.0)).u_t_sq == 0.0


def test_no_thermal_term_when_ground_decoupled():
    """EQ 46d: the effective source of a buoyant jet has ground heating off."""
    f = _fr(atm(), cloud(), ground_coupled=False)
    assert f.u_t_sq == 0.0
    assert f.T_ground == pytest.approx(cloud().T)


def test_no_thermal_term_when_lofted():
    assert _fr(atm(), cloud(z_c=20.0)).u_t_sq == 0.0


def test_lofted_ground_friction_reduces_to_ambient():
    """A lofted cloud feels no ground friction; U_mg*^2 -> u_a*^2 + U_mh*^2."""
    a = atm()
    cl = cloud(z_c=20.0)
    f = _fr(a, cl)
    assert f.u_mg_star**2 == pytest.approx(a.u_star**2 + f.u_mh_star_sq, rel=1e-12)


def test_slip_velocity_sign_and_density_weighting():
    a = atm()
    ubar = a.wind_speed(3.0)
    slow = _fr(a, cloud(u=0.5 * ubar), ubar)
    fast = _fr(a, cloud(u=1.5 * ubar), ubar)
    assert slow.delta_u > 0 and fast.delta_u < 0
    light = _fr(a, cloud(u=0.5 * ubar, rho=a.rho), ubar)
    assert abs(light.delta_u) > abs(slow.delta_u)      # denser cloud -> smaller dU


def test_source_relaxation_blends_and_decays():
    a, cl = atm(), cloud()
    base = _fr(a, cl)
    enhanced = 25.0 * base.u_mg_star**2
    near = _fr(a, cl, u_star_source_sq=enhanced, relax=0.0)
    mid = _fr(a, cl, u_star_source_sq=enhanced, relax=1.0)
    far = _fr(a, cl, u_star_source_sq=enhanced, relax=25.0)
    assert near.u_mg_star == pytest.approx(base.u_mg_star)   # relax=0 -> disabled
    assert mid.u_mg_star > base.u_mg_star
    assert far.u_mg_star == pytest.approx(base.u_mg_star, rel=1e-9)


def test_source_jet_shear_raises_friction():
    a = atm()
    assert _fr(a, cloud(w_source=3.0)).u_mg_star > _fr(a, cloud(w_source=0.0)).u_mg_star


# ===========================================================================
# in-cloud stability, EQ 35d
# ===========================================================================
def test_dense_cloud_is_more_stable_than_neutral_density():
    """Stratification term must raise 1/L, i.e. damp mixing."""
    a = atm()
    dense = _fr(a, cloud(rho=1.6))
    neutral = _fr(a, cloud(rho=a.rho))
    assert dense.inv_L_cloud > neutral.inv_L_cloud
    assert dense.phi_h > neutral.phi_h


def test_buoyant_cloud_does_not_destabilise():
    """A lighter-than-air cloud is clamped, not treated as unstable."""
    a = atm()
    light = _fr(a, cloud(rho=0.5 * a.rho))
    neutral = _fr(a, cloud(rho=a.rho))
    assert light.inv_L_cloud == pytest.approx(neutral.inv_L_cloud, rel=1e-9)


def test_phi_h_limits():
    a = atm()
    f = _fr(a, cloud(rho=a.rho))
    assert f.phi_h >= 1.0 or f.inv_L_cloud < 0.0


# ===========================================================================
# entrainment
# ===========================================================================
def test_entrainment_all_positive():
    a, cl = atm(), cloud()
    e = entrainment(cl, a, _fr(a, cl), u_bar_ambient=a.wind_speed(cl.h))
    assert e.w > 0 and e.v > 0 and e.v_x > 0


def test_crosswind_entrainment_is_hypot_of_parts():
    a, cl = atm(), cloud()
    e = entrainment(cl, a, _fr(a, cl), u_bar_ambient=a.wind_speed(cl.h))
    assert e.v == pytest.approx(math.hypot(e.v_ambient, e.v_shear))


def test_the_code_and_the_tno_documentation_disagree_slightly():
    """
    A discrepancy between Ermak's code and TNO's write-up of it, not a defect
    in either.

    TNO's Yellow Book gives the cross-wind shear entrainment as
    ``0.209 * kappa * du``.  With the entrainment constant a = 1.5 that
    implies ``sqrt(Cf) * a = 0.209``, hence Cf = 0.0195.  Ermak's Fortran uses
    0.02 (SLAB.FOR line 381), giving sqrt(0.02) * 1.5 = 0.2121.

    The two differ by 1.3 %.  The default follows the code; ``preset("tno")``
    carries the documented value.

    This was originally recorded as a rounding defect in the code, on the
    assumption that TNO reported the true constant.  Comparing against the
    Fortran showed the code has always used 0.02, so the direction of the
    claim was wrong.
    """
    a, cl = atm(), cloud()
    f = _fr(a, cl)

    code = entrainment(cl, a, f, u_bar_ambient=a.wind_speed(cl.h)).v_shear
    documented = entrainment(cl, a, f, u_bar_ambient=a.wind_speed(cl.h),
                             coeffs=preset("tno")).v_shear
    tno = 0.209 * PHYS.VON_KARMAN * f.delta_u

    # the documented value reproduces TNO almost exactly
    assert documented == pytest.approx(tno, rel=0.005)
    # the code's value is close but visibly different
    assert code == pytest.approx(tno, rel=0.02)
    assert abs(code / tno - 1.0) > abs(documented / tno - 1.0)


def test_the_default_is_ermaks_own_constant():
    """
    Regression guard.  Every coefficient in `Coefficients` must be the value
    in SLAB.FOR, not a value inferred from a secondary description of it.
    """
    assert COEFFS.c_drag_top == 0.02        # SLAB.FOR L381
    assert COEFFS.a_entrain == 1.50         #           L378  aa = 1.50
    assert COEFFS.c_10 == 0.086             #           L379  cf00 = .086
    assert COEFFS.a2_horiz == 0.0004        #           L380  sigb = .0004
    assert COEFFS.c_mu_strat == 0.025       #           L382  cri = .025
    assert COEFFS.c_thermal == 0.14         #           L383  cth = .14
    assert COEFFS.alpha_g == 0.25           #           L385  alfg = .25
    assert COEFFS.alpha_gv == 0.75          #           L386  alfgv = .75
    assert COEFFS.c_source_shear == 0.05    #           L387  cws = .05
    assert COEFFS.t_ref_avg == 900.0        #           L389  tav0 = 900.
    assert COEFFS.tau_min == 10.0           #           L406  tau0 = 10.
    assert COEFFS.h_entrain_ref == 4.0      #           L466  hrf = 4.0


def test_vertical_entrainment_damped_by_stable_stratification():
    a = atm()
    kw = lambda cl: dict(u_bar_ambient=a.wind_speed(cl.h))
    dense, neutral = cloud(rho=2.5), cloud(rho=a.rho)
    w_dense = entrainment(dense, a, _fr(a, dense), **kw(dense)).w
    w_neut = entrainment(neutral, a, _fr(a, neutral), **kw(neutral)).w
    assert w_dense / _fr(a, dense).u_star < w_neut / _fr(a, neutral).u_star


def test_vertical_entrainment_vanishes_at_mixing_layer_top():
    a = atm()
    cl = cloud(h=a.h_mix, z_c=0.0)
    e = entrainment(cl, a, _fr(a, cl), u_bar_ambient=a.wind_speed(10.0))
    assert e.w == pytest.approx(0.0, abs=1e-12)


def test_wide_cloud_entrains_less_per_unit_area():
    """EQ 36a: V_a decreases with B through the a2 term."""
    a = atm()
    kw = lambda cl: dict(u_bar_ambient=a.wind_speed(cl.h))
    narrow, wide = cloud(b_half=5.0), cloud(b_half=500.0)
    assert (entrainment(wide, a, _fr(a, wide), **kw(wide)).v_ambient
            < entrainment(narrow, a, _fr(a, narrow), **kw(narrow)).v_ambient)


def test_meander_function_is_unity_at_reference_time():
    """F_theta(t0) = 1 by construction, so a1 = 0.08 * S(La) there."""
    a, cl = atm(), cloud()
    e = entrainment(cl, a, _fr(a, cl), u_bar_ambient=a.wind_speed(cl.h),
                    t_avg=COEFFS.t_ref_avg)
    e0 = entrainment(cl, a, _fr(a, cl), u_bar_ambient=a.wind_speed(cl.h), t_avg=0.0)
    assert e.a_total / e0.a_total == pytest.approx(
        1.0 / (COEFFS.tau_min / COEFFS.t_ref_avg) ** COEFFS.p_meander
    )


def test_longer_averaging_time_increases_horizontal_entrainment():
    a, cl = atm(), cloud()
    f = _fr(a, cl)
    kw = dict(u_bar_ambient=a.wind_speed(cl.h))
    vs = [entrainment(cl, a, f, t_avg=t, **kw).v for t in (0.0, 60.0, 900.0)]
    assert vs[0] < vs[1] < vs[2]


def test_stable_ambient_suppresses_horizontal_entrainment():
    ratios = {}
    for s in ("B", "D", "F"):
        a = atm(s)
        cl = cloud()
        ratios[s] = entrainment(cl, a, _fr(a, cl),
                                u_bar_ambient=a.wind_speed(cl.h)).a_total
    assert ratios["B"] > ratios["D"] > ratios["F"]


def test_downwind_entrainment_uses_ambient_shear_not_slip():
    """
    EQ 36b's shear term comes from the ambient profile, so it survives even
    when the cloud moves with the wind (dU = 0), unlike the cross-wind term.
    """
    a = atm()
    ubar = a.wind_speed(3.0)
    cl = cloud(u=ubar, v_g=0.0)
    f = _fr(a, cl, ubar)
    e = entrainment(cl, a, f, u_bar_ambient=ubar)
    assert f.delta_u == pytest.approx(0.0, abs=1e-12)
    assert e.v_shear == pytest.approx(0.0, abs=1e-12)
    assert e.v_x > e.v_ambient


# ===========================================================================
# fluxes, EQ 37-39
# ===========================================================================
def test_grounded_fluxes_oppose_motion():
    a, cl = atm(), cloud()
    fl = fluxes(cl, a, _fr(a, cl))
    assert fl.f_v < 0.0            # gravity spreading is retarded
    assert fl.f_w == 0.0
    assert fl.f_t > 0.0            # ground warms the cold cloud


def test_heat_flux_sign_follows_temperature_difference():
    a = atm(T=290.0)
    assert fluxes(cloud(T=250.0), a, _fr(a, cloud(T=250.0))).f_t > 0
    assert fluxes(cloud(T=320.0), a, _fr(a, cloud(T=320.0))).f_t < 0


def test_fu_vanishes_when_cloud_moves_with_the_wind():
    """
    EQ 38 subtracts the ambient friction so that f_u = 0 when U = <ua>;
    otherwise a passive cloud would be decelerated forever.
    """
    a = atm()
    ubar = a.wind_speed(3.0)
    cl = cloud(u=ubar, v_g=0.0, rho=a.rho)
    fl = fluxes(cl, a, _fr(a, cl, ubar))
    scale = cl.rho * cl.b_half * a.u_star**2
    assert abs(fl.f_u) < 1e-6 * scale


def test_lofted_fluxes_switch_branch():
    a, cl = atm(), cloud(z_c=20.0, w_c=0.3)
    fl = fluxes(cl, a, _fr(a, cl))
    assert fl.f_t == 0.0 and fl.f_v == 0.0 and fl.f_ug == 0.0
    assert fl.f_w < 0.0            # drag opposes the rise


def test_puff_fluxes_scale_with_length():
    a = atm()
    kw = dict(is_puff=True, u_g=0.5, v_g=0.4)
    short = cloud(b_half_x=10.0, **kw)
    long_ = cloud(b_half_x=30.0, **kw)
    fs, fl = fluxes(short, a, _fr(a, short)), fluxes(long_, a, _fr(a, long_))
    assert fl.f_t / fs.f_t == pytest.approx(3.0, rel=1e-12)
    assert fl.f_ug / fs.f_ug == pytest.approx(3.0, rel=1e-12)


def test_plume_mode_fluxes_independent_of_bx():
    a = atm()
    f1 = fluxes(cloud(b_half_x=1.0), a, _fr(a, cloud(b_half_x=1.0)))
    f2 = fluxes(cloud(b_half_x=99.0), a, _fr(a, cloud(b_half_x=99.0)))
    assert f1.f_t == pytest.approx(f2.f_t)


def test_drag_is_quadratic_and_signed():
    """Cf|x|x, not Cf x^2 — the drag must reverse with the velocity."""
    a = atm()
    ubar = a.wind_speed(3.0)
    up = fluxes(cloud(z_c=20.0, w_c=+0.5), a, _fr(a, cloud(z_c=20.0, w_c=+0.5)))
    dn = fluxes(cloud(z_c=20.0, w_c=-0.5), a, _fr(a, cloud(z_c=20.0, w_c=-0.5)))
    assert up.f_w < 0 < dn.f_w
    assert up.f_w == pytest.approx(-dn.f_w, rel=1e-12)


# ===========================================================================
# sqrt(3) convention
# ===========================================================================
def test_reduced_velocities_are_documented_convention():
    """
    The module returns reduced velocities; the conservation equations apply
    sqrt(3).  Verify the factor is the documented one and not folded in.
    """
    a, cl = atm(), cloud()
    e = entrainment(cl, a, _fr(a, cl), u_bar_ambient=a.wind_speed(cl.h))
    W_e = SQRT3 * e.w
    assert SQRT3 == pytest.approx(math.sqrt(3.0))
    assert W_e > e.w


# ===========================================================================
# the central physics: stratification damping dominates
# ===========================================================================
def test_stratification_damping_dominates_for_a_dense_cloud():
    """
    SLAB's whole reason for existing: a denser-than-air cloud is stably
    stratified, phi_h grows, and entrainment collapses.  Under Burro-8-like
    conditions the damping factor must span an order of magnitude across a
    realistic density range, and w must fall monotonically with density.
    """
    a = Atmosphere(u_ref=1.92, z_ref=2.88, T=306.0, rh=4.6, z0=2e-4, inv_L=0.0665)
    ws, phis = [], []
    for rho in (1.16, 1.25, 1.40, 1.60):
        cl = cloud(h=1.89, b_half=46.3, u=0.81, T=216.0, rho=rho, cp=1400.0, v_g=0.74)
        ubar = a.mean_wind_over(cl.h, n=200)
        f = friction(cl, a, u_bar_ambient=ubar)
        ws.append(entrainment(cl, a, f, u_bar_ambient=ubar).w)
        phis.append(f.phi_h)
    assert all(b < a_ for a_, b in zip(ws, ws[1:]))       # w falls with density
    assert all(b > a_ for a_, b in zip(phis, phis[1:]))   # phi_h rises
    assert phis[-1] / phis[0] > 10.0                      # order of magnitude


# ===========================================================================
# gravity-front entrainment (not in Ermak)
# ===========================================================================
def test_frontal_entrainment_is_off_by_default():
    """
    The default must reproduce Ermak exactly, or the golden-reference
    baseline moves and every reproduction number in the project is invalid.
    """
    assert COEFFS.alpha_front == 0.0
    a, cl = atm(), cloud(v_g=0.8)
    e = entrainment(cl, a, _fr(a, cl), u_bar_ambient=a.wind_speed(cl.h))
    assert e.v_front == 0.0
    assert e.v == pytest.approx(math.hypot(e.v_ambient, e.v_shear))


def test_frontal_entrainment_scales_with_the_front_velocity():
    a = atm()
    c = COEFFS.perturb(alpha_front=0.05, name="f")
    for vg in (0.0, 0.4, 0.8):
        cl = cloud(v_g=vg)
        e = entrainment(cl, a, _fr(a, cl), u_bar_ambient=a.wind_speed(cl.h),
                        coeffs=c)
        assert e.v_front == pytest.approx(0.05 * vg)


def test_frontal_entrainment_switches_itself_off():
    """
    A lofted or buoyant cloud has no gravity current, and V_g is already zero
    there, so the term needs no gating of its own.
    """
    a = atm()
    c = COEFFS.perturb(alpha_front=0.5, name="f")
    lofted = cloud(z_c=20.0, v_g=0.0)
    e = entrainment(lofted, a, _fr(a, lofted),
                    u_bar_ambient=a.wind_speed(lofted.h), coeffs=c)
    assert e.v_front == 0.0


def test_frontal_entrainment_is_kept_out_of_the_width_velocity():
    """
    Regression guard for a modelling error that reverses the sign of the
    effect.

    In SLAB the cross-wind entrainment velocity does double duty: it dilutes
    (EQ 2a) and it widens (EQ 7).  The width equation already carries V_g in
    its own right, so folding the frontal term into `v` double-counts the
    front's advance — B inflates while the flat-core width b, driven only by
    V_g, does not follow.  The cross-wind profile then sharpens and the
    centreline concentration *rises* even as the cloud dilutes, which is the
    opposite of what the term is for.
    """
    a, cl = atm(), cloud(v_g=0.8)
    c = COEFFS.perturb(alpha_front=0.5, name="f")
    e0 = entrainment(cl, a, _fr(a, cl), u_bar_ambient=a.wind_speed(cl.h))
    e1 = entrainment(cl, a, _fr(a, cl), u_bar_ambient=a.wind_speed(cl.h),
                     coeffs=c)
    assert e1.v_front > 3.0 * e1.v_ambient       # a large frontal term
    assert e1.v == pytest.approx(e0.v)           # yet `v` is untouched


def test_front_velocity_dwarfs_the_existing_horizontal_entrainment():
    """
    Why the term matters at all.  Under stable, low-wind conditions — the
    ones SLAB is used for in conservative hazard assessment — the gravity
    front advances an order of magnitude faster than SLAB's entire
    horizontal entrainment velocity, so even a small coefficient is not a
    small correction.
    """
    a = Atmosphere(u_ref=1.92, z_ref=2.88, T=306.0, rh=4.6, z0=2e-4,
                   inv_L=0.0665)
    cl = cloud(h=2.5, b_half=150.0, u=1.3, T=290.0, rho=1.20, v_g=0.6)
    e = entrainment(cl, a, _fr(a, cl), u_bar_ambient=a.mean_wind_over(cl.h))
    assert abs(cl.v_g) > 5.0 * e.v


# ===========================================================================
# canonical-flow limits of the entrainment function
# ===========================================================================
def _shear_limit(coeffs, rho_ratio, h=2.0):
    """u_e/u_* and Ri for an isothermal cloud with no slip and no gravity flow."""
    a = Atmosphere(u_ref=5.0, z_ref=10.0, T=290.0, rh=0.0, z0=0.01,
                   stability="D")
    ub = a.mean_wind_over(h, n=200)
    cl = CloudLocal(h=h, b_half=200.0, b_half_x=1.0, z_c=0.0, u=ub,
                    T=a.T, rho=rho_ratio * a.rho, cp=1010.0)
    fr = friction(cl, a, u_bar_ambient=ub, coeffs=coeffs)
    en = entrainment(cl, a, fr, u_bar_ambient=ub, coeffs=coeffs)
    Ri = (PHYS.GRAVITY * (cl.rho - a.rho) * h / (cl.rho * fr.u_star**2))
    return SQRT3 * en.w / fr.u_star, Ri


@pytest.mark.parametrize("phi_s,c_mu", [(5.0, 0.025), (5.0, 0.05),
                                        (5.0, 0.06), (4.7, 0.025)])
def test_damping_shape_is_one_over_phi_times_cmu(phi_s, c_mu):
    """
    ``lambda_2/lambda_1 = 1/(phi_stable c_mu_strat)``.

    This ratio is what makes the comparison with laboratory data possible:
    both limits are normalised by the same u_*, so it is independent of the
    box-height convention and of how the turbulence scale is defined.
    """
    c = COEFFS.perturb(phi_stable=phi_s, c_mu_strat=c_mu, name="x")
    lam1, _ = _shear_limit(c, 1.0 + 1e-9)
    ratio, Ri = _shear_limit(c, 200.0)
    lam2 = ratio * Ri
    assert lam2 / lam1 == pytest.approx(1.0 / (phi_s * c_mu), rel=0.08)


def test_ermak_damping_is_weaker_than_the_canonical_flows_require():
    """
    Kato & Phillips (1969) and Sutton (1953) give lambda_2/lambda_1 = 3.33.
    Ermak's coefficients give 8.0, so SLAB damps stratified entrainment 2.4
    times too weakly.  Neither reference value comes from a dispersion trial.
    """
    lam1, _ = _shear_limit(COEFFS, 1.0 + 1e-9)
    ratio, Ri = _shear_limit(COEFFS, 200.0)
    assert (ratio * Ri) / lam1 == pytest.approx(8.0, rel=0.05)

    canonical = preset("canonical")
    lam1c, _ = _shear_limit(canonical, 1.0 + 1e-9)
    ratioc, Ric = _shear_limit(canonical, 200.0)
    assert (ratioc * Ric) / lam1c == pytest.approx(2.5 / 0.75, rel=0.05)


def test_canonical_preset_damps_more_at_high_richardson():
    """Stronger damping must mean less entrainment for a dense cloud."""
    dense = 2.0
    r_ermak, _ = _shear_limit(COEFFS, dense)
    r_canon, _ = _shear_limit(preset("canonical"), dense)
    assert r_canon < r_ermak
    # and the two must agree in the passive limit, which fixes lambda_1
    p_ermak, _ = _shear_limit(COEFFS, 1.0 + 1e-9)
    p_canon, _ = _shear_limit(preset("canonical"), 1.0 + 1e-9)
    assert p_canon == pytest.approx(p_ermak, rel=1e-6)


def test_entrainment_carries_an_absolute_height_factor():
    """
    A similarity closure should see the cloud only through dimensionless
    groups, but EQ 35a multiplies by ``U_r/U_a(h)`` with the reference height
    fixed at 4 m.  Two clouds at the same Richardson number therefore entrain
    at different rates purely because one is thinner — worth recording as a
    departure, not a bug.
    """
    vals = [_shear_limit(COEFFS, 1.0 + 1e-9, h=h)[0] for h in (0.5, 4.0, 16.0)]
    assert vals[0] > vals[1] > vals[2]
    assert vals[0] / vals[2] > 1.5


def _convective_limit(coeffs, rho_ratio, h=2.0, dT=60.0, u_ref=1.0):
    """
    u_e/w_* and Ri_w for a cold cloud over warm ground.

    SLAB's thermal velocity is *not* the convective velocity scale.  From
    EQ 35e, ``U_t^3 = C_t g dT V_H h / T``; the standard scale is
    ``w_*^3 = g h phi/(rho cp T)`` with ``phi = rho V_H cp dT``, which gives
    ``w_*^3 = g h V_H dT / T``.  The two differ by exactly C_t, so
    ``U_t = C_t^(1/3) w_*``.  Comparing u_e/U_t against a measured u_e/w_*
    would overstate SLAB's convective entrainment by 1/0.519 = 1.93.
    """
    a = Atmosphere(u_ref=u_ref, z_ref=10.0, T=290.0, rh=0.0, z0=0.01,
                   stability="D")
    ub = a.mean_wind_over(h, n=200)
    cl = CloudLocal(h=h, b_half=200.0, b_half_x=1.0, z_c=0.0, u=ub,
                    T=a.T - dT, rho=rho_ratio * a.rho, cp=1010.0)
    fr = friction(cl, a, u_bar_ambient=ub, coeffs=coeffs)
    en = entrainment(cl, a, fr, u_bar_ambient=ub, coeffs=coeffs)
    w_star = math.sqrt(fr.u_t_sq) * coeffs.c_thermal ** (-1.0 / 3.0)
    Ri_w = PHYS.GRAVITY * (rho_ratio - 1.0) / rho_ratio * h / w_star**2
    return SQRT3 * en.w / w_star, Ri_w


def test_thermal_velocity_is_ct_cubed_root_times_w_star():
    """
    The identity that makes the convective comparison possible at all.
    Getting it wrong inflates SLAB's apparent convective entrainment by a
    factor of two.
    """
    a = Atmosphere(u_ref=5.0, z_ref=10.0, T=290.0, rh=0.0, z0=0.01,
                   stability="D")
    h, ub = 2.0, None
    ub = a.mean_wind_over(h, n=200)
    for dT in (5.0, 20.0, 60.0):
        cl = CloudLocal(h=h, b_half=200.0, b_half_x=1.0, z_c=0.0, u=ub,
                        T=a.T - dT, rho=1.05 * a.rho, cp=1010.0)
        fr = friction(cl, a, u_bar_ambient=ub)
        fx = fluxes(cl, a, fr)
        phi = fx.f_t / cl.b_half                       # W/m^2
        w_star = (PHYS.GRAVITY * h * phi / (cl.rho * cl.cp * cl.T)) ** (1 / 3)
        assert math.sqrt(fr.u_t_sq) / w_star == pytest.approx(
            COEFFS.c_thermal ** (1 / 3), rel=0.10
        )


def test_one_coefficient_moves_both_branches():
    """
    The consistency check that makes `c_mu_strat = 0.060` more than a fit.

    It was set from the *shear* limits alone (Sutton, Kato & Phillips).  The
    convective limits (Farmer/Bo Pedersen, Deardorff) enter through the same
    damping function but a different velocity scale, so if the closure has
    the right shape one coefficient must improve both.  It does: the
    convective knee moves from 2.5 to 1.0 against a reference of 0.72,
    without having been fitted to it.
    """
    def knee(coeffs):
        l1, _ = _shear_limit(coeffs, 1.0 + 1e-9)
        r2, Ri2 = _shear_limit(coeffs, 400.0)
        l3, _ = _convective_limit(coeffs, 1.0 + 1e-9)
        r4, Ri4 = _convective_limit(coeffs, 400.0)
        return (r2 * Ri2) / l1, (r4 * Ri4) / l3

    shear_e, conv_e = knee(COEFFS)
    shear_c, conv_c = knee(preset("canonical"))

    assert shear_e == pytest.approx(7.7, rel=0.10)
    assert shear_c == pytest.approx(2.5 / 0.75, rel=0.10)      # the target
    # the convective branch follows without having been used to set it
    assert conv_e > 2.0
    assert conv_c < 0.5 * conv_e
    assert abs(conv_c - 0.72) < abs(conv_e - 0.72)


def test_canonical_reduces_entrainment_of_a_dense_convecting_cloud():
    r_e, _ = _convective_limit(COEFFS, 2.0)
    r_c, _ = _convective_limit(preset("canonical"), 2.0)
    assert r_c < r_e


def test_field_and_laboratory_damping_pull_in_opposite_directions():
    """
    The sharpest result of the validation work, pinned so it cannot be
    quietly lost.

    ``PRESETS["canonical"]`` sets the in-cloud stratification damping from
    laboratory gravity-current and stratified-flow experiments; it is a
    stronger damping than Ermak's.  ``PRESETS["field_damping"]`` sets it from
    the between-trial scatter on the Burro measurements; it is weaker.  Both
    are defensible, they disagree by a factor five, and the disagreement is
    the finding.
    """
    lab = preset("canonical")
    field = preset("field_damping")
    ermak = COEFFS

    D = lambda c: c.phi_stable * c.c_mu_strat        # noqa: E731
    assert D(field) < D(ermak) < D(lab)
    assert D(lab) / D(field) > 4.0

    # and the shape number each implies
    assert 1.0 / D(lab) == pytest.approx(2.5 / 0.75, rel=0.05)   # canonical
    assert 1.0 / D(field) == pytest.approx(16.0, rel=0.05)
