"""
Ground cooling under a cold cloud.

SLAB holds the ground at ambient for the whole release.  This submodel lets
it cool, and the result is that for releases of the length of the field
trials it makes almost no difference — which is a finding about Ermak's
assumption rather than a defect in it.
"""

import math

import numpy as np
import pytest

from slabx.coefficients import preset
from slabx.core.plume import run_dispersion
from slabx.core.source import EvaporatingPool
from slabx.post.concentration import concentration_field
from slabx.submodels.ground import (
    CONCRETE, DRY_SOIL, WATER_SUBSTRATE, Substrate, surface_cooling,
)
from slabx.thermo.base import LegacyThermo, water_backend
from slabx.validation.field_trials import (
    BURRO_AREA, METHANE, VARIANTS, load_conditions, load_observations,
)
from slabx.validation.metrics import metrics


# ===========================================================================
# the substrate model
# ===========================================================================
def test_effusivity_orders_the_substrates():
    """
    Effusivity, not conductivity, decides how fast a surface cools under a
    given flux — which is why an LNG pool on water boils at a nearly steady
    rate and one on dry land slows down.
    """
    assert WATER_SUBSTRATE.effusivity > 2.0 * DRY_SOIL.effusivity
    assert CONCRETE.effusivity > DRY_SOIL.effusivity
    assert DRY_SOIL.diffusivity == pytest.approx(0.30 / (1600.0 * 800.0))


def test_cooling_follows_the_square_root_of_time():
    """dT ~ sqrt(t): quadrupling the time doubles the drop."""
    a = surface_cooling(550.0, 100.0, DRY_SOIL)
    b = surface_cooling(550.0, 400.0, DRY_SOIL)
    assert b == pytest.approx(2.0 * a, rel=1e-9)
    assert surface_cooling(1100.0, 100.0, DRY_SOIL) == pytest.approx(2.0 * a)


def test_cooling_is_zero_without_a_flux_and_bounded_in_time():
    assert surface_cooling(0.0, 100.0) == 0.0
    assert surface_cooling(550.0, 0.0) == 0.0
    assert surface_cooling(550.0, 1e9, DRY_SOIL) == 200.0     # capped


def test_water_cools_least():
    for t in (10.0, 107.0, 400.0):
        assert (surface_cooling(550.0, t, WATER_SUBSTRATE)
                < surface_cooling(550.0, t, DRY_SOIL))


def test_the_convective_film_is_the_bottleneck_early_on():
    """
    Why the effect turns out small.  A step change at the surface would drive
    tens of kW/m^2 in the first seconds, against the 500-600 W/m^2 the cloud
    can actually take away, so for the first minutes it is the cloud's own
    convective transfer that limits the heat flow and not the ground.
    """
    k, alpha, dT = DRY_SOIL.conductivity, DRY_SOIL.diffusivity, 100.0
    conduction = lambda t: k * dT / math.sqrt(math.pi * alpha * t)  # noqa: E731
    assert conduction(1.0) > 50.0 * 600.0
    assert conduction(300.0) > 600.0            # still not limiting at 5 min


# ===========================================================================
# effect on the field comparison
# ===========================================================================
@pytest.fixture(scope="module")
def burro():
    obs = load_observations()
    cond = load_conditions()
    out = {}
    for name, sub in (("none", None), ("soil", DRY_SOIL)):
        pred = []
        for o in obs:
            t = o["trial"]
            c = cond[t]
            atm = VARIANTS["slabx"].atmosphere(c)
            src = EvaporatingPool(substance=METHANE, rate=c["rate"],
                                  area=BURRO_AREA[t], duration=c["duration"])
            traj, _ = run_dispersion(src, atm, LegacyThermo(METHANE),
                                     water_backend(), x_max=1000.0,
                                     n_puff_steps=40,
                                     coeffs=preset("ermak90"), substrate=sub)
            field = concentration_field(traj, atm, z=1.0, t_avg=c["t_avg"],
                                        t_release=c["duration"])
            pred.append(float(np.exp(np.interp(
                o["x"], field.x, np.log(np.maximum(field.peak, 1e-30))))))
        out[name] = np.array(pred)
    return out, np.array([o["c_obs"] for o in obs])


def test_ground_cooling_changes_the_answer_but_barely(burro):
    """
    The result.  Ermak's fixed ground temperature costs nothing on releases
    of this length: the effect is real, it is in the direction the physics
    says, and it is an order of magnitude smaller than the residual scatter.
    """
    p, y = burro
    assert not np.allclose(p["soil"], p["none"])          # it does something
    a, b = metrics(y, p["none"]), metrics(y, p["soil"])
    assert abs(b.MG / a.MG - 1.0) < 0.02                 # under 2 %


def test_it_is_recorded_in_the_trajectory_metadata():
    cond = load_conditions()["BU08"]
    atm = VARIANTS["slabx"].atmosphere(cond)
    src = EvaporatingPool(substance=METHANE, rate=cond["rate"],
                          area=BURRO_AREA["BU08"], duration=cond["duration"])
    traj, _ = run_dispersion(src, atm, LegacyThermo(METHANE), water_backend(),
                             x_max=300.0, n_puff_steps=20,
                             coeffs=preset("ermak90"), substrate=DRY_SOIL)
    assert traj.meta["substrate"] == "dry soil"
