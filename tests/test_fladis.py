"""
FLADIS: two-phase ammonia jets, and measured concentration fluctuation.

Three trials from the Risoe/STEP campaign — pressurised ammonia through a
4-6 mm nozzle at 1.5 m, releasing 0.27-0.46 kg/s for 10-19 minutes.  Two
things make them worth having.

First, they double the two-phase jet evidence.  Desert Tortoise gave three
trials at 80-117 kg/s; FLADIS is three orders of magnitude smaller, so
together they span a range no single campaign covers.

Second, and rarer: the SMEDIS files record the **standard deviation** of the
concentration at every sensor alongside the mean.  Concentration fluctuation
has been inferred indirectly everywhere else in this package — from the ratio
of short- to long-averaged statistics — and here it is measured.

What the fluctuation data says
------------------------------
Fluctuation intensity sigma_C / C_mean, over 150 sensors:

    FL9  (D)    median 0.86    FL16 (D-E)  median 0.86    FL24 (C)  median 0.72

and it falls with distance in every trial — 1.10 within 30 m against 0.72
beyond 200 m for FL9.  Both are physical: a jet is most intermittent close to
the source, and the least stable trial fluctuates least.  The values sit
close to unity, which is the well-known result that in a dispersing plume the
concentration standard deviation is of the same order as the mean.
"""

import csv
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("CoolProp")

from slabx.coefficients import preset                            # noqa: E402
from slabx.core.plume import run_dispersion                      # noqa: E402
from slabx.core.source import HorizontalJet                      # noqa: E402
from slabx.post.concentration import concentration_field         # noqa: E402
from slabx.submodels.atmosphere import Atmosphere                # noqa: E402
from slabx.thermo.base import Substance                          # noqa: E402
from slabx.thermo.coolprop import CoolPropThermo, coolprop_water  # noqa: E402
from slabx.validation.metrics import metrics                     # noqa: E402
from slabx.validation._data_access import require

AMMONIA = Substance(name="NH3", mw=0.017031, cp_vapour=2272.0,
                    cp_liquid=4611.8, dh_vap=1370000.0, T_boil=239.6,
                    rho_liquid=684.0, sat_B=2976.01)

DATA = (Path(__file__).resolve().parents[1] / "src" / "slabx" / "validation"
        / "data")

#: Post-flash expansion, as an area ratio on the nozzle.  SLAB has no
#: notional-nozzle model, so the expanded jet area is supplied; 25 is the
#: value consistent with the Desert Tortoise flash calculation scaled to
#: these pressures, and it is a modelling choice rather than a measurement.
EXPANSION = 25.0


def _read(name):
    with open(require(name), newline="") as fh:
        return list(csv.DictReader(fh))


def _conditions():
    return {r["trial"]: r for r in _read("fladis_conditions.csv")}


def _relative_humidity(raw):
    """The files record RH as a percentage in one trial and a fraction in
    the others; 0.62 means 62 %, not 0.62 %."""
    rh = float(raw)
    return min(rh * 100.0 if rh < 1.5 else rh, 99.0)


def _predict(trial):
    c = _conditions()[trial]
    atm = Atmosphere(u_ref=float(c["u10_m_s"]), z_ref=10.0,
                     T=float(c["T_amb_C"]) + 273.15,
                     rh=_relative_humidity(c["rh_pct"]),
                     z0=float(c["z0_m"]), inv_L=1.0 / float(c["L_m"]))
    d = float(c["nozzle_d_m"])
    jet = HorizontalJet(substance=AMMONIA, rate=float(c["rate_kg_s"]),
                        area=math.pi * d * d / 4.0 * EXPANSION,
                        duration=float(c["duration_s"]), liquid_fraction=0.80,
                        height=float(c["release_z_m"]), T_source=239.6)
    traj, _ = run_dispersion(jet, atm, CoolPropThermo(AMMONIA, fluid="Ammonia"),
                             coolprop_water(), x_max=400.0, n_puff_steps=40,
                             coeffs=preset("ermak90"))
    return concentration_field(traj, atm, z=1.5, t_avg=float(c["t_avg_s"]),
                               t_release=float(c["duration_s"]))


# ===========================================================================
# the data
# ===========================================================================
def test_three_trials_across_stability():
    c = _conditions()
    assert set(c) == {"FL9", "FL16", "FL24"}
    assert float(c["FL24"]["L_m"]) < 0          # unstable
    assert float(c["FL9"]["L_m"]) > 0           # stable side of neutral


def test_the_rates_are_three_orders_below_desert_tortoise():
    """Together the two campaigns span 0.27 to 117 kg/s."""
    c = _conditions()
    assert all(0.2 < float(r["rate_kg_s"]) < 0.5 for r in c.values())


def test_sensor_observations_carry_a_standard_deviation():
    rows = _read("fladis_sensors.csv")
    assert len(rows) > 150
    assert {r["trial"] for r in rows} == {"FL9", "FL16", "FL24"}
    for r in rows:
        assert float(r["std_C_pct"]) >= 0.0


# ===========================================================================
# measured concentration fluctuation
# ===========================================================================
def _intensity(trial, lo=0.0, hi=1e9):
    rows = [r for r in _read("fladis_sensors.csv")
            if r["trial"] == trial and float(r["mean_C_pct"]) > 1e-4
            and lo <= float(r["x_m"]) < hi]
    return np.array([float(r["std_C_pct"]) / float(r["mean_C_pct"])
                     for r in rows])


def test_fluctuation_intensity_is_of_order_one():
    """
    The classic result for a dispersing plume, here measured rather than
    inferred: the concentration standard deviation is comparable with the
    mean.
    """
    for trial in ("FL9", "FL16", "FL24"):
        median = float(np.median(_intensity(trial)))
        assert 0.5 < median < 1.2, trial


def test_fluctuation_falls_with_distance():
    """Most intermittent near the source, as a jet should be."""
    for trial in ("FL9", "FL16", "FL24"):
        near = np.median(_intensity(trial, 0.0, 30.0))
        far = np.median(_intensity(trial, 200.0, 1e9))
        assert far < near, trial


def test_the_unstable_trial_fluctuates_least():
    """
    FL24 is the only unstable trial and has the lowest intensity, which is
    the opposite of the peak-to-mean ordering inferred from the Burro and
    Coyote long/short averaging pairs.  Recorded because the two measures are
    not the same quantity: intensity is an instantaneous property at a fixed
    point, peak-to-mean compares two averaging windows.
    """
    assert np.median(_intensity("FL24")) < np.median(_intensity("FL9"))


# ===========================================================================
# arc-max concentrations, against the other integral models
# ===========================================================================
@pytest.fixture(scope="module")
def arcs():
    rows = [r for r in _read("jriii_arcmax.csv") if r["trial"].startswith("FL")]
    fields = {t: _predict(t) for t in ("FL9", "FL16", "FL24")}
    obs, pred = [], []
    for r in rows:
        f = fields[r["trial"]]
        x = float(r["arc_m"])
        obs.append(float(r["c_obs_ppm"]))
        pred.append(float(np.exp(np.interp(
            x, f.x, np.log(np.maximum(f.peak, 1e-30))))) * 1e6)
    return np.array(obs), np.array(pred), rows


def test_arc_max_agreement(arcs):
    """
    Nine arcs across three trials.  Two thirds fall within a factor of two
    and the geometric mean bias is 0.95 — better on both counts than DRIFT
    or Phast on the same points, though the scatter is worse.
    """
    obs, pred, _ = arcs
    m = metrics(obs, pred)
    assert m.FAC2 >= 0.6
    assert 0.7 < m.MG < 1.4


def test_bias_beats_the_other_integral_models(arcs):
    """A fact about these nine points, not a general claim."""
    obs, pred, rows = arcs
    ours = metrics(obs, pred)
    for column in ("c_drift", "c_phast"):
        theirs = metrics(obs, np.array([float(r[column]) for r in rows]))
        assert abs(ours.MG - 1.0) <= abs(theirs.MG - 1.0), column


def test_the_near_arc_is_where_it_is_hardest(arcs):
    """
    Every model under-predicts the 20 m arc and over-predicts the far one —
    the same pattern Desert Tortoise shows at 100 m and 800 m, at a
    thousandth of the release rate.  That is a property of the source
    treatment, not of the scale.
    """
    obs, pred, rows = arcs
    near = np.array([float(r["arc_m"]) == 20.0 for r in rows])
    far = np.array([float(r["arc_m"]) > 200.0 for r in rows])
    assert np.median(pred[near] / obs[near]) < 1.0
    assert np.median(pred[far] / obs[far]) > 1.0
