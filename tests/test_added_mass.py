"""
Added mass in the vertical momentum balance.

SLAB's EQ 6 accelerates only the cloud's own mass, so a lofting cloud rises
as if the air above it were not there.  `submodels/added_mass.py` registered
four predictions before the term was wired in; all four are checked here.
"""

import numpy as np
import pytest

from slabx.coefficients import preset
from slabx.core.plume import run_dispersion
from slabx.core.source import EvaporatingPool
from slabx.post.concentration import concentration_field
from slabx.submodels.added_mass import (
    C_ADDED_OBLATE, PREDICTIONS, added_mass_factor,
)
from slabx.submodels.atmosphere import Atmosphere
from slabx.thermo.base import LegacyThermo, Substance, water_backend
from slabx.validation.field_trials import (
    BURRO_AREA, METHANE, VARIANTS, load_conditions, load_observations,
)
from slabx.validation.metrics import metrics

HYDROGEN = Substance(name="H2", mw=0.002016, cp_vapour=14300.0,
                     cp_liquid=9800.0, dh_vap=445000.0, T_boil=20.3,
                     rho_liquid=70.8)


def _rise(area, added_mass, x=50.0):
    atm = Atmosphere(u_ref=2.0, z_ref=10.0, T=290.0, rh=50.0, z0=0.01,
                     stability="F")
    src = EvaporatingPool(substance=HYDROGEN, rate=10.0, area=area,
                          duration=200.0)
    traj, _ = run_dispersion(src, atm, LegacyThermo(HYDROGEN), water_backend(),
                             x_max=400.0, n_puff_steps=20,
                             coeffs=preset("ermak90"), added_mass=added_mass)
    i = int(np.argmin(np.abs(traj.x - x)))
    return (float(np.interp(x, traj.x, traj.z_c)), float(traj.w_c.max()),
            traj.b_half[i] / max(traj.h[i], 1e-9))


# ===========================================================================
# the factor
# ===========================================================================
def test_predictions_are_recorded():
    assert len(PREDICTIONS) == 4
    assert C_ADDED_OBLATE == pytest.approx(2.0 / np.pi)


def test_A3_compact_limit_is_order_one():
    """
    The term must not quietly rescale every buoyant release.  At B = h the
    effective inertia is 1.6 times the cloud's own — a correction, not a
    transformation — and it grows from there.
    """
    assert 1.0 + added_mass_factor(1.0, 1.0, 1.0, 1.0) == pytest.approx(1.64,
                                                                        rel=0.02)
    assert 1.0 + added_mass_factor(0.5, 1.0, 1.0, 1.0) < 1.4


def test_factor_scales_with_aspect_ratio_and_density():
    assert added_mass_factor(20.0, 1.0, 1.0, 1.0) == pytest.approx(
        10.0 * added_mass_factor(2.0, 1.0, 1.0, 1.0))
    # a lighter cloud has more air to shift, relatively
    assert (added_mass_factor(5.0, 1.0, 0.5, 1.2)
            > added_mass_factor(5.0, 1.0, 1.2, 1.2))
    assert added_mass_factor(5.0, 0.0, 1.0, 1.0) == 0.0
    assert added_mass_factor(1e6, 1.0, 1.0, 1.0) == 50.0        # capped


# ===========================================================================
# the registered predictions
# ===========================================================================
def test_A1_no_effect_on_a_cloud_that_never_lofts():
    """
    The negative control.  Every Burro trial stays on the ground, so the
    field statistics must be identical to the last digit.
    """
    obs = load_observations()
    cond = load_conditions()
    y = np.array([o["c_obs"] for o in obs])
    out = []
    for am in (False, True):
        pred = []
        for o in obs:
            t = o["trial"]
            c = cond[t]
            atm = VARIANTS["slabx"].atmosphere(c)
            src = EvaporatingPool(substance=METHANE, rate=c["rate"],
                                  area=BURRO_AREA[t], duration=c["duration"])
            traj, _ = run_dispersion(src, atm, LegacyThermo(METHANE),
                                     water_backend(), x_max=1000.0,
                                     n_puff_steps=40, coeffs=preset("ermak90"),
                                     added_mass=am)
            field = concentration_field(traj, atm, z=1.0, t_avg=c["t_avg"],
                                        t_release=c["duration"])
            pred.append(float(np.exp(np.interp(
                o["x"], field.x, np.log(np.maximum(field.peak, 1e-30))))))
        out.append(metrics(y, np.array(pred)))
    assert out[1].MG == pytest.approx(out[0].MG, rel=1e-12)
    assert out[1].VG == pytest.approx(out[0].VG, rel=1e-12)


def test_A4_rise_is_suppressed_never_enhanced():
    for area in (4.0, 100.0, 2500.0):
        off, w_off, _ = _rise(area, False)
        on, w_on, _ = _rise(area, True)
        assert on <= off
        assert w_on < w_off


def test_A2_suppression_scales_with_the_aspect_ratio():
    """
    The discriminating prediction.  A term that held every buoyant cloud down
    equally would be a fudge on the buoyancy; added mass must bite hardest on
    the flattest clouds.

    Measured against the *achieved* aspect ratio rather than the source
    width, because the cloud reshapes as it travels and the two are not in
    the same order.
    """
    rows = []
    for area in (25.0, 100.0, 400.0, 2500.0):
        off, _, ratio = _rise(area, False)
        on, _, _ = _rise(area, True)
        if off > 1.0:                     # only clouds that actually lofted
            rows.append((ratio, on / off))
    rows.sort()
    suppression = [r[1] for r in rows]
    assert len(rows) >= 3
    assert all(b <= a for a, b in zip(suppression, suppression[1:]))
    assert suppression[0] / suppression[-1] > 3.0


def test_without_it_rise_height_is_blind_to_width():
    """
    Why the term is missing rather than merely small: unmodified, a fiftyfold
    change in source width moves the rise height by under a percent.
    """
    heights = [_rise(a, False, x=150.0)[0] for a in (4.0, 100.0, 2500.0)]
    assert max(heights) / min(heights) < 1.02


def test_no_added_mass_coefficient_fits_hall_and_walker():
    """
    The external check the documentation said could not be done.

    Hall & Walker (2000), reported in AEAT/NOIL/27328006/001, varied a
    ground-level area source 64-fold in width at fixed buoyancy flux and
    found lift-off governed by F/(W u^3) — so at a fixed value of that
    parameter the widths should behave alike.

    They do not. Without the term the spread across widths is 3.7 and
    monotonic, in the opposite sense to the measurement. With the term it
    falls to 1.46 and then **saturates**: tripling `C_A` from 0.2 to 2/pi
    does not improve the collapse at all while it crushes the absolute level
    threefold. No coefficient does both jobs, so the oblate-spheroid `B/h`
    scaling is the wrong shape function rather than a mis-scaled one.

    This is why DRIFT's authors found the term made agreement worse on the
    same data. Slow: runs twelve dispersion cases.
    """
    import math
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    from added_mass_liftoff import WIDTHS, lift_off_metrics

    from slabx.coefficients import COEFFS

    def spread(c_added):
        v = [lift_off_metrics(g, 0.035, added_mass=c_added is not None,
                              c_added=c_added)["30L"] for g in WIDTHS]
        return max(v) / min(v), sum(v) / len(v)

    off_spread, off_level = spread(None)
    mid_spread, mid_level = spread(0.2)
    pot_spread, pot_level = spread(2.0 / math.pi)

    # without it the widths do not collapse, and the wide ones rise most
    assert off_spread > 3.0, off_spread

    # the term helps, but stops helping
    assert mid_spread < 1.7 < off_spread
    assert pot_spread == pytest.approx(mid_spread, abs=0.05), \
        (mid_spread, pot_spread)

    # while still crushing the level
    assert pot_level < mid_level / 2.0

    # and the default is unchanged
    assert COEFFS.c_added_mass == pytest.approx(2.0 / math.pi)
