"""
Multicomponent releases, as a pseudo-component.

SLAB carries one condensable species plus water, so a real LNG — 87 to 93 %
methane with ethane and propane — cannot be represented directly.  The
mixture is collapsed to a single `Substance` evaluated at its own bubble
point, which captures the molecular weight and leaves out fractionation.

PHMSA lists pure methane as a sensitivity case against the measured
composition and calls it "generally conservative"; these tests check that the
model reproduces that, and by how much.
"""

import numpy as np
import pytest

pytest.importorskip("CoolProp")

from slabx.thermo.coolprop import pseudo_component            # noqa: E402
from slabx.validation.field_trials import (                   # noqa: E402
    BURRO_TRIALS, load_composition, load_conditions,
    load_observations, predictions_for,
)
from slabx.validation.metrics import metrics                  # noqa: E402

BU08 = {"Methane": 0.874, "Ethane": 0.103, "n-Propane": 0.023}


# ===========================================================================
# the pseudo-component itself
# ===========================================================================
def test_pure_methane_round_trips():
    s = pseudo_component("LNG", {"Methane": 1.0})
    assert s.mw == pytest.approx(0.016043, rel=1e-3)
    assert s.T_boil == pytest.approx(111.67, abs=0.05)


def test_mixture_is_heavier_and_boils_higher():
    pure = pseudo_component("LNG", {"Methane": 1.0})
    mix = pseudo_component("LNG", BU08)
    assert mix.mw / pure.mw == pytest.approx(1.13, rel=0.02)
    assert mix.T_boil > pure.T_boil
    assert mix.rho_liquid > pure.rho_liquid


def test_composition_is_normalised():
    a = pseudo_component("LNG", BU08)
    b = pseudo_component("LNG", {k: 100 * v for k, v in BU08.items()})
    assert a.mw == pytest.approx(b.mw, rel=1e-9)
    with pytest.raises(ValueError):
        pseudo_component("LNG", {"Methane": 0.0})


def test_heavier_mixture_lowers_the_mole_fraction():
    """
    Half the effect is a unit conversion, not dispersion.  EQ 12 converts
    mass to mole fraction through the molecular weight, and the sensors
    reported mole fraction, so a 13 % heavier molecule reports about 11 %
    less at the same mass loading.
    """
    mw_a = 0.028933
    pure = pseudo_component("LNG", {"Methane": 1.0}).mw
    mix = pseudo_component("LNG", BU08).mw
    cv = lambda ms, m: mw_a * m / (ms + (mw_a - ms) * m)   # noqa: E731
    assert cv(mix, 0.05) / cv(pure, 0.05) == pytest.approx(0.89, abs=0.02)


# ===========================================================================
# the composition data
# ===========================================================================
def test_composition_loads_for_every_trial():
    comp = load_composition()
    assert set(comp) >= set(BURRO_TRIALS)
    for t, c in comp.items():
        assert sum(c.values()) == pytest.approx(1.0, abs=0.01), t
        assert 0.85 <= c["Methane"] <= 0.95, t
    # BU09 was not reported and stands in from BU08 — recorded, not hidden
    assert comp["BU09"] == comp["BU08"]


# ===========================================================================
# effect on the field comparison
# ===========================================================================
@pytest.fixture(scope="module")
def runs():
    obs = load_observations()
    cond = load_conditions()
    return (obs, np.array([o["c_obs"] for o in obs]),
            np.array([o["trial"] for o in obs]),
            {v: predictions_for(v, obs, cond) for v in ("coolprop", "composition")})


def test_pure_methane_is_conservative(runs):
    """
    PHMSA's word for it.  Modelling the mixture as pure methane must
    over-predict the reported mole fraction, so switching to the measured
    composition moves MG towards one from below on every measure.
    """
    _, y, _, p = runs
    a, b = metrics(y, p["coolprop"]), metrics(y, p["composition"])
    assert b.MG > a.MG
    assert b.FAC2 >= a.FAC2
    assert b.VG < a.VG and b.NMSE < a.NMSE
    assert abs(b.FB) < abs(a.FB)


def test_effect_is_ordered_by_non_methane_content(runs):
    """
    The mechanism signature.  A change that moved every trial by the same
    amount would be a bias correction; this one must scale with how much
    ethane and propane each release actually contained, and BU03 — the
    leanest at 92.5 % methane — must move least.
    """
    _, y, tid, p = runs
    comp = load_composition()
    eff = {}
    for t in BURRO_TRIALS:
        m = tid == t
        a = metrics(y[m], p["coolprop"][m]).MG
        b = metrics(y[m], p["composition"][m]).MG
        eff[t] = abs(b / a - 1.0)
    assert eff["BU03"] == min(eff.values())
    assert 1.0 - comp["BU03"]["Methane"] < 1.0 - comp["BU07"]["Methane"]


def test_effect_is_monotone_in_composition():
    """
    Swept directly, with everything else held fixed: more ethane and propane
    must mean a lower predicted mole fraction, without exception.
    """
    from slabx.coefficients import preset
    from slabx.core.plume import run_dispersion
    from slabx.core.source import EvaporatingPool
    from slabx.post.concentration import concentration_field
    from slabx.thermo.base import LegacyThermo, water_backend
    from slabx.validation.field_trials import BURRO_AREA, VARIANTS

    cond = load_conditions()["BU08"]
    atm = VARIANTS["slabx"].atmosphere(cond)
    out = []
    for ch4 in (1.0, 0.95, 0.90, 0.85, 0.80):
        rest = 1.0 - ch4
        sub = pseudo_component("LNG", {"Methane": ch4, "Ethane": 0.8 * rest,
                                       "n-Propane": 0.2 * rest})
        src = EvaporatingPool(substance=sub, rate=cond["rate"],
                              area=BURRO_AREA["BU08"],
                              duration=cond["duration"])
        traj, _ = run_dispersion(src, atm, LegacyThermo(sub), water_backend(),
                                 x_max=600.0, n_puff_steps=30,
                                 coeffs=preset("ermak90"))
        field = concentration_field(traj, atm, z=1.0, t_avg=cond["t_avg"],
                                    t_release=cond["duration"])
        out.append(float(np.exp(np.interp(140.0, field.x,
                                          np.log(np.maximum(field.peak, 1e-30))))))
    assert all(b < a for a, b in zip(out, out[1:]))
    assert out[-1] / out[0] < 0.75            # 20 % non-methane, >25 % lower
