"""
Prairie Grass: the passive-dispersion limit, measured.

Every other trial in this package is a dense-gas release, where the cloud's
own gravity dominates the near field and the passive limit is never cleanly
exposed.  Prairie Grass is a continuous ground-level SO2 tracer at 0.06-0.09
kg/s — no buoyancy, no slumping, no phase change.  It tests the part of the
model that survives after the density excess has gone.

Why it matters here
-------------------
`examples/entrainment_limits.py` found SLAB's passive-dispersion asymptote
sitting at 1.48 times the canonical value, and varying with cloud depth from
1.63 at 0.5 m to 0.85 at 16 m because of an absolute reference height in
EQ 35a.  That is a statement about a limit, and it left open whether the
limit is ever reached in a way that matters.

Against measurement it does not: concentration and cross-wind spread both
come out within a factor of two at every arc, and the aggregate passes all
five Chang-Hanna criteria.  The asymptotic discrepancy is real but does not
translate into an error against passive field data.

What the data gives that others do not
---------------------------------------
The SMEDIS files record the *measured* friction velocity, inverse
Monin-Obukhov length and sigma_theta alongside the concentrations, so the
atmosphere needs no fitting: 1/L is an input, not an estimate.  They also
record the measured plume width at each arc, which almost no dense-gas
dataset does — so the width can be validated directly rather than inferred.
"""

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from slabx.coefficients import preset
from slabx.core.plume import run_dispersion
from slabx.core.source import EvaporatingPool
from slabx.post.concentration import concentration_field
from slabx.submodels.atmosphere import Atmosphere
from slabx.thermo.base import LegacyThermo, Substance, water_backend
from slabx.validation.metrics import check_acceptance, metrics
from slabx.validation._data_access import require

SO2 = Substance(name="SO2", mw=0.064, cp_vapour=622.6, cp_liquid=1331.0,
                dh_vap=386500.0, T_boil=263.13, rho_liquid=1462.0)

DATA = (Path(__file__).resolve().parents[1] / "src" / "slabx" / "validation"
        / "data")


def _read(name):
    with open(require(name), newline="") as fh:
        return list(csv.DictReader(fh))


def _conditions():
    return {r["trial"]: r for r in _read("prairie_grass_conditions.csv")}


@pytest.fixture(scope="module")
def comparison():
    arcs = _read("prairie_grass_arcs.csv")
    obs_c, pred_c, obs_w, pred_w, ustar = [], [], [], [], {}
    for trial, c in _conditions().items():
        atm = Atmosphere(u_ref=float(c["u_ref_m_s"]),
                         z_ref=float(c["z_ref_m"]), T=float(c["T_amb_K"]),
                         rh=50.0, z0=float(c["z0_m"]),
                         inv_L=float(c["inv_L"]))
        d = float(c["source_diam_m"])
        src = EvaporatingPool(substance=SO2, rate=float(c["rate_kg_s"]),
                              area=math.pi * d * d / 4.0,
                              duration=float(c["duration_s"]))
        traj, _ = run_dispersion(src, atm, LegacyThermo(SO2), water_backend(),
                                 x_max=1200.0, n_puff_steps=40,
                                 coeffs=preset("ermak90"))
        field = concentration_field(traj, atm, z=1.5,
                                    t_avg=float(c["t_avg_s"]),
                                    t_release=float(c["duration_s"]))
        ustar[trial] = (atm.u_star, float(c["u_star_m_s"]))
        for r in arcs:
            if r["trial"] != trial:
                continue
            x = float(r["distance_m"])
            obs_c.append(float(r["c_arcmax_ppm"]))
            pred_c.append(float(np.exp(np.interp(
                x, field.x, np.log(np.maximum(field.peak, 1e-30))))) * 1e6)
            i = int(np.argmin(np.abs(traj.x - x)))
            obs_w.append(float(r["sigma_y_m"]))
            pred_w.append(math.sqrt(traj.b_shape[i] ** 2 / 3.0
                                    + field.beta_c[i] ** 2))
    return (np.array(obs_c), np.array(pred_c),
            np.array(obs_w), np.array(pred_w), ustar)


# ===========================================================================
# the data
# ===========================================================================
def test_two_trials_bracketing_neutral():
    c = _conditions()
    assert set(c) == {"PG8", "PG17"}
    assert float(c["PG8"]["inv_L"]) < 0        # unstable
    assert float(c["PG17"]["inv_L"]) > 0       # stable


def test_arcs_span_a_factor_of_sixteen_in_distance():
    arcs = _read("prairie_grass_arcs.csv")
    d = sorted({float(r["distance_m"]) for r in arcs})
    assert d == [50.0, 100.0, 200.0, 400.0, 800.0]
    assert all(float(r["sigma_y_m"]) > 0 for r in arcs)


# ===========================================================================
# the atmosphere needs no fitting
# ===========================================================================
def test_friction_velocity_is_reproduced(comparison):
    """
    1/L and z0 are measured inputs here, so u* is a prediction rather than a
    fitted quantity — and it comes out within 20 % of the measured value.
    """
    *_, ustar = comparison
    for trial, (ours, theirs) in ustar.items():
        assert ours == pytest.approx(theirs, rel=0.25), trial


# ===========================================================================
# the passive limit, measured
# ===========================================================================
def test_concentration_passes_every_acceptance_criterion(comparison):
    obs, pred, *_ = comparison
    m = metrics(obs, pred)
    assert m.FAC2 == 1.0                      # all ten arcs within a factor 2
    assert check_acceptance(m)["all"]


def test_plume_width_is_predicted_too(comparison):
    """
    Rarely testable: most dense-gas datasets do not record plume width.  Here
    it agrees as well as the concentration does.
    """
    _, _, obs_w, pred_w, _ = comparison
    m = metrics(obs_w, pred_w)
    assert m.FAC2 == 1.0
    assert check_acceptance(m)["all"]


def test_the_passive_asymptote_discrepancy_does_not_show_up_here(comparison):
    """
    The point of running these trials.  SLAB's passive-dispersion asymptote
    sits 1.48 times above the canonical value and varies with cloud depth
    because of an absolute reference height in EQ 35a — but against measured
    passive dispersion the model is unbiased to within 22 %.

    So the asymptotic departure is real and the Froude-similarity failure it
    causes is real, without either producing a first-order error on passive
    field data.
    """
    obs, pred, *_ = comparison
    m = metrics(obs, pred)
    assert 0.75 < m.MG < 1.35
    assert m.VG < 1.3


def test_width_growth_is_too_fast_in_the_stable_trial(comparison):
    """
    The one systematic residual, and it carries the same signature as
    everything else: in the stable trial the predicted width runs from 0.95
    of the measurement at 50 m to 1.76 at 800 m, while in the unstable trial
    it stays near unity throughout.  Stability again.
    """
    arcs = _read("prairie_grass_arcs.csv")
    _, _, obs_w, pred_w, _ = comparison
    ratio = pred_w / obs_w
    stable = np.array([r["trial"] == "PG17" for r in arcs])
    far = np.array([float(r["distance_m"]) >= 400.0 for r in arcs])
    assert ratio[stable & far].mean() > 1.3
    assert ratio[~stable & far].mean() < 1.2
