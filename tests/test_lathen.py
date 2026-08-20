"""
Lathen trial 49: a small instantaneous propane release.

Seven kilograms of propane-air mixture released in one second, measured over
a 45 m grid — three orders of magnitude smaller than Thorney Island and at a
tenth the distance.  It is the only unobstructed, level trial left in the
SMEDIS set after the others are excluded, and it is included here because
what it shows is a limit rather than a success.

What agrees
-----------
Cloud arrival time, to within a factor 1.5 at both bands, with FAC2 = 1.0.
The same quantity that Thorney Island gets right at 480 m is right here at
15 m and a thousandth of the mass — the front speed closure travels.

What does not
-------------
Dose, by a factor of ten, and not for want of a source assumption: sweeping
the release area from 1 to 80 square metres moves the answer by less than 20
per cent and never within a factor of five.

Tracing it, the model has the cloud passing a sensor at 15 m in about 10
seconds while the measured arrival and departure times imply 15 to 60.  The
passage duration is estimated as ``2 Bx / u``, which assumes the cloud is
advected past the sensor as a rigid body of half-length Bx; a real puff of
this size is still spreading longitudinally while it passes, and at these
distances it has had no time to reach the self-similar state that assumption
describes.

Where this leaves the scale
----------------------------
The instantaneous source type is validated at Thorney Island's scale — 2000
cubic metres, arcs from 50 to 500 m — and not at this one.  The failure is in
the dose estimate rather than the dispersion: the concentrations and the
front speed are both reasonable, and it is only their product over an assumed
passage time that is wrong.

That is a bound on the *dose* post-processing, not on the model, and it is
recorded as such.
"""

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from slabx.coefficients import preset
from slabx.core.plume import run_dispersion
from slabx.core.source import InstantaneousRelease
from slabx.submodels.atmosphere import Atmosphere
from slabx.thermo.base import LegacyThermo, Substance, water_backend
from slabx.validation.metrics import metrics
from slabx.validation._data_access import require

DATA = (Path(__file__).resolve().parents[1] / "src" / "slabx" / "validation"
        / "data")

BANDS = ((10, 20), (20, 30))


def _read(name):
    with open(require(name), newline="") as fh:
        return list(csv.DictReader(fh))


def _conditions():
    return _read("lathen_conditions.csv")[0]


def _run(area=4.0):
    c = _conditions()
    sub = Substance(name="C3/air", mw=float(c["mw_g"]) / 1000.0,
                    cp_vapour=1200.0, cp_liquid=2500.0, dh_vap=4.26e5,
                    T_boil=231.1, rho_liquid=580.0)
    atm = Atmosphere(u_ref=float(c["u_ref_m_s"]), z_ref=float(c["z_ref_m"]),
                     T=float(c["T_amb_C"]) + 273.15, rh=float(c["rh_pct"]),
                     z0=float(c["z0_m"]), inv_L=1.0 / float(c["L_m"]))
    src = InstantaneousRelease(
        substance=sub, rate=0.0, duration=0.0, area=area,
        mass=float(c["rate_kg_s"]) * float(c["duration_s"]), height=0.0)
    traj, _ = run_dispersion(src, atm, LegacyThermo(sub), water_backend(),
                             x_max=120.0, t_max=400.0, n_puff_steps=50,
                             coeffs=preset("ermak90"))
    return traj


@pytest.fixture(scope="module")
def banded():
    traj = _run()
    rows = [r for r in _read("lathen_sensors.csv")
            if float(r["dose_pct_s"]) > 0 and float(r["z_m"]) <= 0.5]
    out = []
    for lo, hi in BANDS:
        band = [(float(r["dose_pct_s"]),
                 float(r["t_arrival_s"]) if r["t_arrival_s"] else np.nan)
                for r in rows
                if lo <= math.hypot(float(r["x_m"]), float(r["y_m"])) < hi]
        if not band:
            continue
        mid = (lo + hi) / 2.0
        i = int(np.argmin(np.abs(traj.x - mid)))
        duration = 2.0 * traj.b_half_x[i] / max(traj.u[i], 1e-6)
        out.append(dict(
            distance=mid,
            dose_obs=max(d for d, _ in band),
            dose_pred=traj.vol_frac[i] * 100.0 * duration,
            arrival_obs=float(np.nanmedian([t for _, t in band])),
            arrival_pred=float(np.interp(mid, traj.x, traj.t)),
            passage_pred=duration,
        ))
    return out, rows


def test_it_is_the_only_level_unobstructed_trial_left():
    c = _conditions()
    assert c["obstacle"] == "none"
    assert c["substance"] == "Propane"
    assert float(c["duration_s"]) <= 1.0


def test_the_release_is_a_thousandth_of_thorney_island():
    c = _conditions()
    mass = float(c["rate_kg_s"]) * float(c["duration_s"])
    assert mass < 10.0                        # against ~3900 kg


def test_arrival_time_still_works_at_this_scale(banded):
    """
    The front-speed closure travels across three orders of magnitude in mass
    and one in distance: FAC2 = 1.0 here as at Thorney Island.
    """
    out, _ = banded
    obs = np.array([b["arrival_obs"] for b in out])
    pred = np.array([b["arrival_pred"] for b in out])
    m = metrics(obs, pred)
    assert m.FAC2 == 1.0
    assert m.VG < 1.4


def test_dose_fails_by_an_order_of_magnitude(banded):
    """Recorded, not tuned away."""
    out, _ = banded
    obs = np.array([b["dose_obs"] for b in out])
    pred = np.array([b["dose_pred"] for b in out])
    assert metrics(obs, pred).MG > 4.0


def test_the_source_area_cannot_explain_it():
    """
    Eighty-fold in area moves the dose by under 20 %, so the discrepancy is
    not a source assumption.
    """
    doses = []
    for area in (1.0, 4.0, 12.0, 30.0, 80.0):
        traj = _run(area)
        i = int(np.argmin(np.abs(traj.x - 15.0)))
        duration = 2.0 * traj.b_half_x[i] / max(traj.u[i], 1e-6)
        doses.append(traj.vol_frac[i] * 100.0 * duration)
    assert max(doses) / min(doses) < 2.0
    assert max(doses) < 0.25 * 98.81          # observed at 15 m


def test_the_passage_duration_is_what_is_wrong(banded):
    """
    The diagnosis.  Measured arrival and departure times imply the cloud
    takes 15 to 60 s to pass; the model's ``2 Bx / u`` gives about 10.  A puff
    this small is still spreading longitudinally as it passes and has not
    reached the self-similar state that estimate assumes.
    """
    out, rows = banded
    assert all(b["passage_pred"] < 13.0 for b in out)
    measured = [float(r["t_departure_s"]) - float(r["t_arrival_s"])
                for r in rows if r["t_arrival_s"] and r["t_departure_s"]]
    assert np.median(measured) > 20.0
