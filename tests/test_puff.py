"""
Tier-0/1 verification of the puff mode, the transition, and the full run.

The final test in this file is the first end-to-end comparison against the
manual: Burro 8 across the whole domain, plume and puff.
"""

import math

import numpy as np
import pytest

from slabx.core.plume import run_dispersion, run_plume
from slabx.core.puff import (
    PuffState,
    slopes_puff,
    transition_half_length,
    transition_state,
)
from slabx.core.source import EvaporatingPool
from slabx.core.trajectory import Mode
from slabx.submodels.atmosphere import Atmosphere
from slabx.thermo.base import LegacyThermo, Substance, water_backend
from slabx.validation.metrics import check_acceptance, metrics

METHANE = Substance(
    name="LNG", mw=0.016043, cp_vapour=2238.0, cp_liquid=3348.5,
    dh_vap=509900.0, T_boil=111.7, rho_liquid=424.1,
)

#: Manual example 1 (Burro 8), "instantaneous spatially averaged cloud
#: parameters" table: x [m], h [m], BB [m], U [m/s], rho [kg/m3], CV [-].
MANUAL = [
    (99.9, 1.60, 127.0, 1.32, 1.19, 0.187),
    (164.0, 2.23, 148.0, 1.49, 1.17, 0.0928),
    (260.0, 3.44, 165.0, 1.65, 1.16, 0.0460),
    (427.0, 5.68, 183.0, 1.82, 1.15, 0.0201),
    (630.0, 8.36, 197.0, 1.96, 1.15, 0.0106),
    (938.0, 12.2, 213.0, 2.09, 1.15, 0.00546),
]


def burro8_setup(duration=107.0):
    atm = Atmosphere(u_ref=1.92, z_ref=2.88, T=306.0, rh=4.6,
                     z0=2e-4, inv_L=0.0665)
    src = EvaporatingPool(substance=METHANE, rate=117.0, area=657.0,
                          duration=duration)
    return src, atm, LegacyThermo(METHANE), water_backend()


@pytest.fixture(scope="module")
def full_run():
    src, atm, em, wt = burro8_setup()
    tr, used = run_dispersion(src, atm, em, wt, x_max=200.0, n_puff_steps=60)
    return tr, used, atm


def _log_interp(tr, name, x):
    v = np.maximum(getattr(tr, name), 1e-30)
    return float(np.exp(np.interp(x, tr.x, np.log(v))))


# ===========================================================================
# transition, section 2.3.3
# ===========================================================================
def test_transition_occurs_near_the_release_duration(full_run):
    """
    EQ 29a puts the switch where half the released mass is in the cloud,
    which is when the centre of mass passes — close to t = t_sd.
    """
    tr, used, _ = full_run
    assert 0.8 * used.duration < tr.meta["transition_t"] < 1.5 * used.duration
    assert tr.meta["transition_x"] > 0.0


def test_both_modes_present_and_ordered(full_run):
    tr, *_ = full_run
    assert tr.is_plume.sum() > 3 and tr.is_puff.sum() > 10
    k = tr.transition_index
    assert k is not None
    assert np.all(tr.mode[:k] == Mode.PLUME)
    assert np.all(tr.mode[k:] == Mode.PUFF)


def test_state_is_continuous_across_the_transition(full_run):
    """
    Section 2.3.3 requires every spatially averaged property to be continuous;
    only the bookkeeping changes.  One step either side must agree closely.
    """
    tr, *_ = full_run
    k = tr.transition_index
    for name, rtol in (("h", 0.05), ("b_half", 0.05), ("u", 0.05),
                       ("T", 0.02), ("rho", 0.02), ("mass_frac", 0.08)):
        a, b = getattr(tr, name)[k - 1], getattr(tr, name)[k]
        assert b == pytest.approx(a, rel=rtol), name


def test_transition_half_length_matches_the_explicit_form():
    """
    EQ 30 can be written two ways.  ``Bx = R/(rho By h)`` with
    ``R = q t_sd/(4m)`` and ``Bx = U t_sd / 2`` agree identically because the
    plume relation ``m = q/(2 rho U By h)`` holds at the switch.
    """
    src, atm, em, wt = burro8_setup()
    tr, used = run_plume(src, atm, em, wt, x_max=120.0)
    i = len(tr) - 1
    from slabx.core.plume import PlumeState

    st = PlumeState(
        R=tr.R_flux[i], Bv=0, bv=0, z_c=tr.z_c[i], Qint=tr.mass_in_cloud[i],
        h=tr.h[i], b_half=tr.b_half[i], b_shape=tr.b_shape[i], u=tr.u[i],
        T=tr.T[i], rho=tr.rho[i], cp=tr.cp[i], m=tr.mass_frac[i],
        m_ev=tr.mass_frac_emission_vapour[i], m_water=tr.mass_frac_water[i],
        m_wv=tr.mass_frac_water_vapour[i], u_bar=tr.u_ambient_mean[i],
        h_top=tr.h[i],
    )
    ps = transition_state(st, atm, used, float(tr.t[i]))
    assert ps.b_half_x == pytest.approx(
        transition_half_length(st.u, used.duration), rel=1e-6
    )


def test_continuous_release_never_transitions():
    src, atm, em, wt = burro8_setup(duration=math.inf)
    tr, _ = run_dispersion(src, atm, em, wt, x_max=300.0)
    assert np.all(tr.mode == Mode.PLUME)


# ===========================================================================
# puff equations
# ===========================================================================
def test_puff_slopes_have_the_right_shape(full_run):
    tr, used, atm = full_run
    k = tr.transition_index
    st = PuffState(
        R=tr.R_flux[k], x_c=tr.x[k], Bv=0, bv=0, Bvx=0, bvx=0,
        z_c=tr.z_c[k], t=tr.t[k], h=tr.h[k], b_half=tr.b_half[k],
        b_shape=tr.b_shape[k], b_half_x=tr.b_half_x[k],
        b_shape_x=tr.b_shape_x[k], u=tr.u[k], T=tr.T[k], rho=tr.rho[k],
        cp=tr.cp[k], m=tr.mass_frac[k], m_ev=0.0, m_water=0.0, m_wv=0.0,
        w_e=tr.w_entrain[k], v_e=tr.v_entrain[k], v_ex=tr.v_x_entrain[k],
    )
    f = slopes_puff(st, atm, used)
    assert len(f) == 15
    assert f[0] > 0.0                 # cloud gains mass by entrainment
    assert f[1] == pytest.approx(st.u)   # dXc/dt = U once the source is off


def test_puff_inventory_is_conserved_after_the_release(full_run):
    """
    EQ 15a: once ``t > t_sd`` the released mass in the cloud is fixed, so
    m R must be constant even though both grow and shrink.
    """
    tr, used, _ = full_run
    late = tr.is_puff & (tr.t > used.duration * 1.5)
    inventory = tr.mass_frac[late] * tr.R_flux[late]
    assert inventory.std() / inventory.mean() < 1e-6


def test_puff_spreads_horizontally_throughout(full_run):
    tr, *_ = full_run
    p = tr.is_puff
    assert np.all(np.diff(tr.b_half[p]) > 0)
    assert np.all(np.diff(tr.b_half_x[p]) > 0)


def test_puff_slumps_before_it_grows(full_run):
    """
    The height is *not* monotonic.  Just after the release stops the cloud is
    still dense and stably stratified (phi_h is several), so vertical
    entrainment is suppressed while gravity keeps spreading it sideways —
    R/(rho Bx By) therefore falls.  Only once the cloud has diluted towards
    ambient does the height start to grow.  This slumping-then-growth shape
    is the signature of dense-gas dispersion and a monotone height would mean
    the stratification damping had been lost.
    """
    tr, _, atm = full_run
    p = tr.is_puff
    h, rho, phi = tr.h[p], tr.rho[p], tr.phi_h[p]
    i = int(np.argmin(h))

    assert 0 < i < len(h) - 1                       # a genuine interior minimum
    assert h[i] < h[0]                              # slumps first
    assert 0.02 < 1.0 - h[i] / h[0] < 0.5           # by a physical amount
    assert np.all(np.diff(h[:i + 1]) < 0)           # monotone down, then
    assert np.all(np.diff(h[i:]) > 0)               # monotone up
    assert h[-1] > 5.0 * h[0]

    # the turning point must coincide with the loss of dense-gas damping
    assert rho[i] / atm.rho > 1.02                  # still denser than air
    assert phi[i] > 2.0                             # still damped
    assert phi[-1] < phi[i]                         # damping fades downwind


def test_puff_dilutes_and_warms(full_run):
    tr, _, atm = full_run
    p = tr.is_puff
    assert np.all(np.diff(tr.mass_frac[p]) < 0)
    assert np.all(np.diff(tr.T[p]) > 0)
    assert tr.T[-1] == pytest.approx(atm.T, rel=0.01)
    assert tr.rho[-1] == pytest.approx(atm.rho, rel=0.01)


def test_puff_accelerates_towards_the_ambient_wind(full_run):
    tr, _, atm = full_run
    p = tr.is_puff
    assert np.all(np.diff(tr.u[p]) > 0)
    assert tr.u[-1] < atm.wind_speed(tr.h[-1]) * 1.05


def test_shape_relations_hold_in_puff_mode(full_run):
    tr, *_ = full_run
    p = tr.is_puff
    assert np.allclose(tr.b_half[p] ** 2,
                       tr.b_shape[p] ** 2 + 3.0 * tr.beta[p] ** 2, rtol=1e-9)
    assert np.allclose(tr.b_half_x[p] ** 2,
                       tr.b_shape_x[p] ** 2 + 3.0 * tr.beta_x[p] ** 2, rtol=1e-9)


def test_puff_master_variable_is_a_volume_integral(full_run):
    """R = rho Bx By h in puff mode, unlike the plume's rho U By h."""
    tr, *_ = full_run
    p = tr.is_puff
    assert np.allclose(tr.R_flux[p],
                       tr.rho[p] * tr.b_half_x[p] * tr.b_half[p] * tr.h[p],
                       rtol=1e-6)


# ===========================================================================
# end-to-end comparison with the manual
# ===========================================================================
def test_reproduces_manual_example_1(full_run):
    """
    Burro 8 across the full domain, 100 m to 938 m, plume and puff.

    This is the first end-to-end check of the model against a published
    result.  The tolerance is set by what the scanned table can support, not
    by the model: the printed values carry three significant figures and the
    OCR of the source parameters was verified separately.
    """
    tr, *_ = full_run
    obs, pred = [], []
    for x, h, B, u, rho, cv in MANUAL:
        assert _log_interp(tr, "h", x) == pytest.approx(h, rel=0.10)
        assert _log_interp(tr, "b_half", x) == pytest.approx(B, rel=0.05)
        assert _log_interp(tr, "u", x) == pytest.approx(u, rel=0.05)
        assert _log_interp(tr, "rho", x) == pytest.approx(rho, rel=0.02)
        obs.append(cv)
        pred.append(_log_interp(tr, "vol_frac", x))

    m = metrics(obs, pred)
    assert m.FAC2 == 1.0
    assert m.MG == pytest.approx(1.0, abs=0.10)
    assert m.VG < 1.05
    assert check_acceptance(m, "chang_hanna_2004")["all"]


def test_concentration_falls_monotonically_over_three_decades(full_run):
    tr, *_ = full_run
    cv = tr.vol_frac[tr.x > 0]
    assert np.all(np.diff(cv) < 0)
    assert cv[0] / cv[-1] > 100.0


# ===========================================================================
# transition bookkeeping — regression guards
# ===========================================================================
def test_transition_time_is_exactly_the_release_duration(full_run):
    """
    EQ 29a and 29b together fix the switch at ``t = t_sd`` exactly, since
    ``Qint = q t / 2``.  An earlier version snapped the transition to the
    nearest output row, which put it anywhere from t_sd to t_sd + 3 s
    depending on where the logarithmic grid fell — a 2 % error that then
    propagated into every downstream travel time and half-length.
    """
    tr, used, _ = full_run
    assert tr.meta["transition_t"] == pytest.approx(used.duration, rel=1e-9)
    k = tr.transition_index
    assert tr.t[k - 1] == pytest.approx(used.duration, rel=1e-9)


@pytest.mark.parametrize("n_field", [30, 60, 120])
def test_transition_is_independent_of_the_output_grid(n_field):
    src, atm, em, wt = burro8_setup()
    tr, _ = run_dispersion(src, atm, em, wt, x_max=400.0,
                           n_field_steps=n_field, n_puff_steps=30)
    assert tr.meta["transition_t"] == pytest.approx(107.0, rel=1e-9)
    assert tr.meta["transition_x"] == pytest.approx(51.0, rel=0.02)


def test_mass_in_cloud_is_half_the_release(full_run):
    """
    Regression guard.  ``Qint`` is the released mass inside the cloud,
    ``0.5 q t`` (EQ 29b); the factor 1/4 in EQ 15a belongs to the
    *concentration*, because R describes a quarter of the cloud.  Confusing
    them made the puff inventory half its correct value while leaving every
    concentration correct — invisible except against the reference.
    """
    tr, used, _ = full_run
    assert tr.mass_in_cloud[-1] == pytest.approx(0.5 * used.total_mass, rel=1e-9)
    k = tr.transition_index
    assert tr.mass_in_cloud[k - 1] == pytest.approx(0.5 * used.total_mass,
                                                    rel=1e-9)
    # and the plume relation t = 2 Qint / q must hold throughout the plume
    pl = tr.is_plume
    assert np.allclose(tr.t[pl], 2.0 * tr.mass_in_cloud[pl] / used.rate,
                       rtol=1e-9)


def test_interpolated_transition_row_is_self_consistent(full_run):
    """
    The transition row is produced by interpolation, and the shape
    parameters are not independent (EQ 13).  Blending B, b and beta
    separately leaves the row violating B^2 = b^2 + 3 beta^2 by a few parts
    in 10^7 — small, but enough to make the meander round trip fail and to
    put a kink in any profile built from that row.
    """
    tr, *_ = full_run
    k = tr.transition_index
    for i in (k - 1, k):
        assert tr.b_half[i] ** 2 == pytest.approx(
            tr.b_shape[i] ** 2 + 3.0 * tr.beta[i] ** 2, rel=1e-12)
        assert tr.b_half_x[i] ** 2 == pytest.approx(
            tr.b_shape_x[i] ** 2 + 3.0 * tr.beta_x[i] ** 2, rel=1e-12)
