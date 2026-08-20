"""
Thorney Island trial 008: dose and cloud arrival time.

The fifteen Phase I trials tested in `test_thorney_island.py` are compared
against a *statement* — McQuaid & Roebuck's finding that peak concentration
is insensitive to wind speed and stability.  This trial has numbers: 118
sensors with integrated dose, arrival time and departure time, and a release
point given in the same coordinate frame.

Why arrival time is the sharper test
-------------------------------------
Dose is the concentration integrated over the cloud's passage, so it carries
every error the concentration does plus the duration.  Arrival time carries
only the cloud front's speed, which for an instantaneous dense release is the
sum of advection and gravity spreading — exactly the balance the model is
built around, with no thermodynamics and no profile shape in the way.

It comes out at FAC2 = 1.00 and VG = 1.11 across five distance bands, and
converges with distance: 1.81 at 65 m, 1.10 at 330 m, 1.03 at 480 m.  The
near-field excess is the source region, where the cloud is still collapsing
out of its 14 m container and the model starts it as a slab.

Dose is weaker — FAC2 = 0.40, under-predicting beyond 90 m by a factor of
two to three, which is consistent with the cloud passing more slowly in
reality than ``2 Bx / u`` implies.

Why the comparison is banded
-----------------------------
The sensor grid is a site frame, and the best-fit cloud path leaves a
dose-weighted cross-wind offset of 41 m — the cloud did not travel exactly
along the grid axis.  Comparing individual sensors would therefore be
comparing positions the model cannot know.  Taking the maximum dose in each
distance band is the arc-max convention used everywhere else in this package
and is insensitive to that misalignment.

Trial 021 is registered but excluded: it has a curved wall.
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

#: Released volume [m^3]; the container was 14 m across and 13 m tall.
VOLUME = 2000.0
BANDS = ((40, 90), (90, 160), (160, 260), (260, 400), (400, 560))


def _read(name):
    with open(require(name), newline="") as fh:
        return list(csv.DictReader(fh))


def _conditions():
    return {r["trial"]: r for r in _read("thorney_dose_conditions.csv")}


@pytest.fixture(scope="module")
def banded():
    c = _conditions()["TI008"]
    x0, y0 = float(c["release_x"]), float(c["release_y"])
    sub = Substance(name="R12/N2", mw=float(c["mw_g"]) / 1000.0,
                    cp_vapour=600.0, cp_liquid=900.0, dh_vap=1e5,
                    T_boil=50.0, rho_liquid=1300.0)
    atm = Atmosphere(u_ref=float(c["u_ref_m_s"]), z_ref=10.0,
                     T=float(c["T_amb_C"]) + 273.15, rh=70.0,
                     z0=float(c["z0_m"]), stability="D")
    mass = VOLUME * float(c["mw_g"]) / 28.964 * atm.rho
    src = InstantaneousRelease(substance=sub, rate=0.0, duration=0.0,
                               area=math.pi * 7.0 ** 2, mass=mass, height=0.0)
    traj, _ = run_dispersion(src, atm, LegacyThermo(sub), water_backend(),
                             x_max=900.0, t_max=1500.0, n_puff_steps=60,
                             coeffs=preset("ermak90"))

    rows = [r for r in _read("thorney_dose_sensors.csv")
            if r["trial"] == "TI008" and float(r["dose_pct_s"]) > 0]
    out = []
    for lo, hi in BANDS:
        band = [(float(r["dose_pct_s"]),
                 float(r["t_arrival_s"]) if r["t_arrival_s"] else np.nan)
                for r in rows
                if lo <= abs(float(r["y_m"]) - y0) < hi
                and float(r["z_m"]) <= 0.5]
        if not band:
            continue
        mid = (lo + hi) / 2.0
        i = int(np.argmin(np.abs(traj.x - mid)))
        duration = 2.0 * traj.b_half_x[i] / max(traj.u[i], 1e-6)
        out.append(dict(
            distance=mid, n=len(band),
            dose_obs=max(d for d, _ in band),
            dose_pred=traj.vol_frac[i] * 100.0 * duration,
            arrival_obs=float(np.nanmedian([t for _, t in band])),
            arrival_pred=float(np.interp(mid, traj.x, traj.t)),
        ))
    return out


# ===========================================================================
# the data
# ===========================================================================
def test_trial_008_is_unobstructed_and_021_is_not():
    c = _conditions()
    assert c["TI008"]["obstacle"] == "none"
    assert "wall" in c["TI021"]["obstacle"]


def test_sensors_carry_dose_and_arrival_time():
    rows = [r for r in _read("thorney_dose_sensors.csv")
            if r["trial"] == "TI008"]
    assert len(rows) > 100
    with_arrival = [r for r in rows if r["t_arrival_s"]]
    assert len(with_arrival) > 30
    # -999 sentinels are dropped rather than carried through as numbers
    assert all(float(r["t_arrival_s"]) > 0 for r in with_arrival)


def test_the_release_is_instantaneous_and_dense():
    c = _conditions()["TI008"]
    assert float(c["duration_s"]) <= 1.0
    assert float(c["mw_g"]) / 28.964 > 1.5        # 1.63 times air


# ===========================================================================
# arrival time
# ===========================================================================
def test_arrival_time_agrees_at_every_band(banded):
    """
    The sharpest quantitative test of the instantaneous source type: cloud
    front speed, with no thermodynamics or profile shape involved.
    """
    obs = np.array([b["arrival_obs"] for b in banded])
    pred = np.array([b["arrival_pred"] for b in banded])
    m = metrics(obs, pred)
    assert m.FAC2 == 1.0
    assert m.VG < 1.3
    assert m.NMSE < 0.1


def test_arrival_time_converges_with_distance(banded):
    """
    1.81 at 65 m falling to 1.03 at 480 m.  The near-field excess is the
    source region, where the cloud is still collapsing out of its container
    and the model starts it as a slab.
    """
    ratio = [b["arrival_pred"] / b["arrival_obs"] for b in banded]
    assert ratio[0] > 1.4
    assert abs(ratio[-1] - 1.0) < 0.15
    assert ratio[0] > ratio[-1]


# ===========================================================================
# dose
# ===========================================================================
def test_dose_is_the_weaker_of_the_two(banded):
    """
    Dose carries the concentration error and the passage duration together,
    and comes out two to three times low beyond 90 m — consistent with the
    cloud passing more slowly in reality than ``2 Bx / u`` implies.  Recorded
    rather than tuned.
    """
    obs = np.array([b["dose_obs"] for b in banded])
    pred = np.array([b["dose_pred"] for b in banded])
    m = metrics(obs, pred)
    assert m.FAC2 <= 0.6
    assert 1.2 < m.MG < 3.0                       # under-predicts
    assert m.NMSE < 1.5


def test_the_near_field_band_is_over_predicted(banded):
    """Both measures err the same way close in, which points at the source
    region rather than at the dispersion."""
    assert banded[0]["dose_pred"] > banded[0]["dose_obs"]
    assert banded[0]["arrival_pred"] > banded[0]["arrival_obs"]
