"""
LNG pool trials: distance to the lower flammable limit
======================================================

Ten evaporating-pool releases — Burro, Coyote and Maplin Sands — against the
measured distance at which the arc-wise maximum concentration falls to the
lower flammable limit.

Why this measure
----------------
The arc-wise concentration comparison in `field_trials` rests on four Burro
trials and twelve points.  That was never enough to separate a
condition-dependent deficiency from scatter: it gave one trial at stability
class C, two at D and one at E.  The LFL distance is reported for all ten
trials in PHMSA's final environmental assessment of Phast 8.4 (Table 9),
which lifts the set to three C, six D and one E, and adds a second surface —
Maplin Sands was spilled at sea.

It is also the quantity the regulation is written around: 49 CFR 193.2059
defines the exclusion zone by the distance to half the lower flammable limit.

Why Burro 8 matters more than the others
-----------------------------------------
It is the low-wind, stable trial, and PHMSA singles it out: Phast 8.4 puts
its LFL at 191 m against a measured 455, and the assessment notes that "even
if the predicted LFL distances were doubled ... the distance would still be
below that based on experimental data", with DNV "not fully certain why".
Those are the conditions 49 CFR 193 is most concerned with, and any model
used for siting has to handle them.

What the same document says about roughness
--------------------------------------------
Independently of anything here, PHMSA reports surface roughness as "the most
impactful parameter for all the cases", and that the values the validation
database specifies "are generally low and result in higher concentrations and
longer dispersion distances to the LFL, which may cause the model to appear
more conservative than it is".  It then records that 49 CFR 193.2059 itself
prescribes 0.03 m or higher — two orders of magnitude above the 0.0002 m the
database uses for these trials.

Two published datasets disagree, and the answer depends on which
----------------------------------------------------------------
Interpolating the arc-wise concentrations in Witlox (2013) down to 5 % gives
LFL distances of 127 m for Burro 7, 314 m for Burro 8 and 194 m for Burro 9,
against PHMSA's 264, 455 and 406 — a factor of 1.4 to 2.1 apart on the same
four trials.  The two must be processed differently; both are published, and
neither is obviously wrong.

The consequence is not academic.  Against the Witlox concentrations the model
over-predicts and raising the surface roughness improves every measure;
against the PHMSA distances it under-predicts and raising the roughness makes
every measure worse, monotonically:

    z0 [m]     FAC2      MG      VG    NMSE
    0.0002     0.90   1.252   1.225   0.121     passes all five
    0.003      0.90   1.431   1.248   0.154
    0.03       0.70   1.797   1.530   0.382

So the same model, the same trials and the same parameter give opposite
recommendations depending on which published observations are used.  That is
worth stating plainly: for these trials the choice of observational dataset
is a larger source of disagreement than any model change examined in this
project.  The two comparisons are therefore kept separate — this module for
distances, `field_trials` for concentrations — rather than merged into one
number that would hide the conflict.

Caveats
-------
* The wind speeds for the Coyote trials differ between the validation
  database (6.0, 9.7, 4.6 m/s) and the Phast assessment (6.77, 10.47,
  5.04).  The latter are used here so the Phast column stays comparable.
* Burro 9's measured value is disturbed by rapid phase transitions at the
  400 m arc; removing those spikes gives 270 m rather than 406.  Both are
  recorded and the unmodified value is used.
* Maplin Sands had large variations in wind speed and direction, worst in
  trial 35 — which is where every model does worst.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..coefficients import Coefficients, preset
from ..core.plume import run_dispersion
from ..core.source import EvaporatingPool
from ..post.concentration import concentration_field
from ..submodels.atmosphere import Atmosphere
from ..thermo.base import LegacyThermo, water_backend
from .field_trials import METHANE

__all__ = [
    "LFL", "VAPORISATION_RATE", "load_trials", "load_lfl",
    "predict_lfl_distance", "compare_lfl",
]

DATA = Path(__file__).parent / "data"

from ._data_access import ObservationsUnavailable, require  # noqa: E402

#: Methane's lower flammable limit as a volume fraction.
LFL = 0.05

#: Steady-pool vaporisation rate [kg/(m^2 s)].  FERC's higher of two values,
#: adopted by the MVD for every LNG spill series; the lower is 0.085.
VAPORISATION_RATE = 0.167


@dataclass(frozen=True)
class Trial:
    """One evaporating-pool release."""

    series: str
    trial: str
    rate: float
    duration: float
    u_ref: float
    z_ref: float
    stability: str
    z0: float
    t_avg: float
    surface: str

    def atmosphere(self, *, z0: float | None = None,
                   T: float = 290.0, rh: float = 50.0) -> Atmosphere:
        # "C/D" appears for Maplin Sands 27; the MVD treats it as D with C as
        # a sensitivity case.
        cls = "D" if self.stability == "C/D" else self.stability
        return Atmosphere(u_ref=self.u_ref, z_ref=self.z_ref, T=T, rh=rh,
                          z0=self.z0 if z0 is None else z0, stability=cls)

    def source(self, *, vaporisation: float = VAPORISATION_RATE,
               substance=METHANE) -> EvaporatingPool:
        return EvaporatingPool(substance=substance, rate=self.rate,
                               area=self.rate / vaporisation,
                               duration=self.duration)


def _read(name: str) -> list[dict]:
    with open(require(name), newline="") as fh:
        return list(csv.DictReader(fh))


def load_trials() -> dict[str, Trial]:
    return {r["trial"]: Trial(
        series=r["series"], trial=r["trial"], rate=float(r["rate_kg_s"]),
        duration=float(r["duration_s"]), u_ref=float(r["u_ref_m_s"]),
        z_ref=float(r["z_ref_m"]), stability=r["stability"],
        z0=float(r["z0_m"]), t_avg=float(r["t_avg_long_s"]),
        surface=r["surface"],
    ) for r in _read("lng_pool_trials.csv")}


def load_lfl() -> dict[str, dict]:
    """Measured and Phast-predicted LFL distances [m]."""
    return {r["trial"]: {"observed": float(r["lfl_obs_m"]),
                         "phast": float(r["lfl_phast_m"]),
                         "phast_half": float(r["half_lfl_phast_m"]),
                         "note": r["note"]}
            for r in _read("lfl_distances.csv")}


def predict_lfl_distance(trial: Trial, *, level: float = LFL,
                         coeffs: Coefficients | None = None,
                         z0: float | None = None,
                         vaporisation: float = VAPORISATION_RATE,
                         substance=METHANE, x_max: float = 2000.0,
                         **kw) -> float:
    """Distance at which the arc-wise maximum falls to `level`."""
    atm = trial.atmosphere(z0=z0)
    src = trial.source(vaporisation=vaporisation, substance=substance)
    traj, _ = run_dispersion(src, atm, LegacyThermo(substance), water_backend(),
                             x_max=x_max, n_puff_steps=40,
                             coeffs=coeffs or preset("ermak90"), **kw)
    field = concentration_field(traj, atm, z=1.0, t_avg=trial.t_avg,
                                t_release=trial.duration)
    return field.distance_to(level)


def compare_lfl(**kw) -> dict:
    """Observed, predicted and Phast LFL distances for every trial."""
    trials, measured = load_trials(), load_lfl()
    names = [t for t in trials if t in measured]
    return {
        "trial": np.array(names),
        "stability": np.array([trials[t].stability for t in names]),
        "observed": np.array([measured[t]["observed"] for t in names]),
        "predicted": np.array([predict_lfl_distance(trials[t], **kw)
                               for t in names]),
        "phast": np.array([measured[t]["phast"] for t in names]),
    }
