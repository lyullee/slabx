"""
Tier-0/1 verification of the concentration post-processing.

The last group is the model's first quantitative reproduction of a published
*concentration* result, as opposed to the internal cloud parameters.
"""

import math

import numpy as np
import pytest

from slabx.core.plume import run_dispersion
from slabx.core.source import EvaporatingPool
from slabx.post.concentration import (
    cloud_duration,
    concentration_field,
    meander_width,
)
from slabx.submodels.atmosphere import Atmosphere
from slabx.thermo.base import LegacyThermo, Substance, water_backend
from slabx.validation.metrics import check_acceptance, metrics

METHANE = Substance(
    name="LNG", mw=0.016043, cp_vapour=2238.0, cp_liquid=3348.5,
    dh_vap=509900.0, T_boil=111.7, rho_liquid=424.1,
)

#: Manual example 1, "time averaged volume concentration: maximum
#: concentration along centerline", TAV = 10.5 s, z = 0.
#: x [m], C(x,0,0) [-], cloud duration [s]
MANUAL_CENTRELINE = [
    (183.0, 0.165, 209.0),
    (205.0, 0.140, 215.0),
    (230.0, 0.119, 221.0),
    (260.0, 0.0996, 227.0),
    (293.0, 0.0831, 233.0),
    (332.0, 0.0690, 240.0),
    (376.0, 0.0570, 248.0),
    (427.0, 0.0469, 256.0),
    (485.0, 0.0384, 265.0),
]


@pytest.fixture(scope="module")
def burro8():
    atm = Atmosphere(u_ref=1.92, z_ref=2.88, T=306.0, rh=4.6,
                     z0=2e-4, inv_L=0.0665)
    src = EvaporatingPool(substance=METHANE, rate=117.0, area=657.0,
                          duration=107.0)
    tr, used = run_dispersion(src, atm, LegacyThermo(METHANE), water_backend(),
                              x_max=200.0, n_puff_steps=60)
    return tr, atm, used


def _at(field, x):
    return float(np.exp(np.interp(x, field.x, np.log(np.maximum(field.peak, 1e-30)))))


# ===========================================================================
# cloud duration
# ===========================================================================
def test_cloud_duration_never_shorter_than_the_release(burro8):
    tr, _, used = burro8
    t_cld = cloud_duration(tr, used.duration)
    assert np.all(t_cld >= used.duration - 1e-9)


def test_cloud_duration_is_non_decreasing_downwind(burro8):
    """A distant receptor cannot see a shorter cloud than a near one."""
    tr, _, used = burro8
    assert np.all(np.diff(cloud_duration(tr, used.duration)) >= -1e-9)


def test_cloud_duration_matches_the_manual(burro8):
    tr, _, used = burro8
    t_cld = cloud_duration(tr, used.duration)
    for x, _, t_man in MANUAL_CENTRELINE:
        got = float(np.interp(x, tr.x, t_cld))
        assert got == pytest.approx(t_man, rel=0.06)


# ===========================================================================
# meander, EQ 48-53
# ===========================================================================
def test_no_meander_at_zero_averaging_time(burro8):
    """EQ 50b must vanish at t_av = 0, leaving the instantaneous width."""
    tr, atm, used = burro8
    B_c, beta_c = meander_width(tr, atm, t_avg=0.0, t_release=used.duration)
    assert np.allclose(B_c, tr.b_half, rtol=1e-9)
    assert np.allclose(beta_c, tr.beta, rtol=1e-9)


def test_meander_widens_monotonically_with_averaging_time(burro8):
    tr, atm, used = burro8
    widths = [meander_width(tr, atm, t_avg=t, t_release=used.duration)[0]
              for t in (0.0, 60.0, 300.0, 900.0)]
    for a, b in zip(widths, widths[1:]):
        assert np.all(b >= a - 1e-12)
        assert np.any(b > a)


def test_meander_is_capped_by_the_cloud_duration(burro8):
    """
    A puff cannot meander for longer than it lasts (section 2.6.2), so the
    width must stop growing once t_av exceeds the local cloud duration.
    """
    tr, atm, used = burro8
    t_cld = cloud_duration(tr, used.duration).max()
    w1 = meander_width(tr, atm, t_avg=t_cld, t_release=used.duration)[0]
    w2 = meander_width(tr, atm, t_avg=10.0 * t_cld, t_release=used.duration)[0]
    assert np.allclose(w1, w2, rtol=1e-9)


def test_meander_preserves_the_shape_relation(burro8):
    """EQ 53: B_c^2 = b^2 + 3 beta_c^2 must still hold."""
    tr, atm, used = burro8
    B_c, beta_c = meander_width(tr, atm, t_avg=600.0, t_release=used.duration)
    assert np.allclose(B_c**2, tr.b_shape**2 + 3.0 * beta_c**2, rtol=1e-9)


# ===========================================================================
# concentration field
# ===========================================================================
def test_peak_is_a_valid_volume_fraction(burro8):
    tr, atm, used = burro8
    f = concentration_field(tr, atm, z=0.0, t_avg=10.5, t_release=used.duration)
    assert np.all((f.peak >= 0.0) & (f.peak <= 1.0))
    assert np.all(np.isfinite(f.peak))


def test_peak_exceeds_the_spatial_average(burro8):
    """
    The centreline value must be above the cross-wind and vertical average;
    otherwise the profile functions are not normalised correctly.
    """
    tr, atm, used = burro8
    f = concentration_field(tr, atm, z=0.0, t_avg=10.5, t_release=used.duration)
    far = tr.x > 100.0
    assert np.all(f.peak[far] > tr.vol_frac[far])


def test_concentration_falls_off_crosswind(burro8):
    tr, atm, used = burro8
    f = concentration_field(tr, atm, z=0.0, t_avg=10.5, t_release=used.duration)
    i = int(np.argmin(np.abs(tr.x - 300.0)))
    ys = np.array([0.0, 0.5, 1.0, 2.0]) * f.B_c[i]
    vals = [f.at_y(y)[i] for y in ys]
    assert vals[0] == pytest.approx(f.peak[i], rel=1e-9)
    assert all(b < a for a, b in zip(vals, vals[1:]))
    assert vals[-1] < 0.2 * vals[0]


def test_concentration_falls_off_with_height(burro8):
    tr, atm, used = burro8
    peaks = [concentration_field(tr, atm, z=z, t_avg=10.5,
                                 t_release=used.duration).peak
             for z in (0.0, 1.0, 5.0, 20.0)]
    i = int(np.argmin(np.abs(tr.x - 300.0)))
    vals = [p[i] for p in peaks]
    assert all(b < a for a, b in zip(vals, vals[1:]))


def test_longer_averaging_lowers_the_peak(burro8):
    tr, atm, used = burro8
    i_at = lambda t: concentration_field(   # noqa: E731
        tr, atm, z=0.0, t_avg=t, t_release=used.duration).peak
    p10, p300, p900 = i_at(10.5), i_at(300.0), i_at(900.0)
    j = int(np.argmin(np.abs(tr.x - 300.0)))
    assert p900[j] < p300[j] < p10[j]


def test_distance_to_a_level_is_consistent(burro8):
    tr, atm, used = burro8
    f = concentration_field(tr, atm, z=0.0, t_avg=10.5, t_release=used.duration)
    for level in (0.15, 0.05, 0.01):
        d = f.distance_to(level)
        assert math.isfinite(d)
        assert _at(f, d) == pytest.approx(level, rel=0.05)
    # a level never reached must not silently return a distance
    assert math.isnan(f.distance_to(10.0))


def test_lfl_distance_is_physically_reasonable(burro8):
    """
    Burro 8's measured LFL (5 % v/v) distance was of order 200-300 m; the
    manual reports SLAB reproducing the LNG-series LFL distances to within
    about 15 %.
    """
    tr, atm, used = burro8
    f = concentration_field(tr, atm, z=0.0, t_avg=10.5, t_release=used.duration)
    assert 150.0 < f.distance_to(0.05) < 450.0


# ===========================================================================
# reproduction of the manual's concentration table
# ===========================================================================
def test_reproduces_manual_centreline_concentration(burro8):
    """
    Manual example 1, centreline maximum concentration, 183 m to 485 m.

    This closes the chain: source parameters, cloud parameters, and now the
    concentration a receptor would record.  The residual bias is uniform
    (every point within 4-5 %), which points at a single constant rather than
    a defect in the physics — most likely the source half-width, which the
    expansion search puts 17 % above the manual's.
    """
    tr, atm, used = burro8
    f = concentration_field(tr, atm, z=0.0, t_avg=10.5, t_release=used.duration)

    obs = [c for _, c, _ in MANUAL_CENTRELINE]
    pred = [_at(f, x) for x, _, _ in MANUAL_CENTRELINE]

    m = metrics(obs, pred)
    assert m.FAC2 == 1.0
    assert m.MG == pytest.approx(1.0, abs=0.08)
    assert m.VG < 1.01
    assert check_acceptance(m, "chang_hanna_2004")["all"]

    ratios = np.array(pred) / np.array(obs)
    assert ratios.std() / ratios.mean() < 0.01     # bias is uniform, not scatter
