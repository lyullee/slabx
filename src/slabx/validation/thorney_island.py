"""
Thorney Island: instantaneous dense releases
============================================

Fifteen usable trials from Phase I — a Freon-12/nitrogen mixture held in a
14 m cylinder whose walls were dropped, letting the cloud slump and disperse
over flat ground.  Relative densities from 0.99 to 4.2, wind speeds from 1.7
to 7.5 m/s, stability classes B to E.

Why these trials
----------------
`slabx` has four source types and until now only three had been compared with
field data at all.  The instantaneous release (IDSPL = 4) had been checked
against the manual's deck and against the reference implementation, but never
against a measurement.  These are the canonical instantaneous dense-gas
trials.

The testable statement
----------------------
The final report does not tabulate concentrations — the results are in
figures — but it states a result that is sharper than a table would be:

    "the peak concentration at a given distance along the path of the cloud
    centroid is quite insensitive to wind speed and atmospheric stability.
    This conclusion suggests that the dispersion ... is dominated by the
    gravity-driven motion of the cloud over most of the dispersion field
    covered by the trials."
                            — McQuaid & Roebuck (1985), section 16.5.1

That is a prediction any model of these releases has to reproduce, and it
needs no concentration data to test.

Result
------
Across the fifteen trials, the predicted peak concentration is:

    insensitive to wind speed        r = +0.18, p = 0.53 at 300 m   agrees
    strongly dependent on stability  r = +0.90, p < 0.001 at 300 m  disagrees

Holding everything else fixed and moving only the stability class from B to F
changes the predicted concentration at 300 m by a factor 6.2.  The trial set
does not confound the two: stability correlates with wind speed at r = -0.14
(p = 0.62) and with relative density at r = +0.38 (p = 0.16).

So SLAB applies its ambient-stability damping in full even where the cloud's
own gravity dominates the mixing — which is exactly where the measurements
say it should not matter.  This is the third independent line pointing at the
same place:

* the roughness each Burro trial needs spans a factor 200, ordered by
  stability class (`field_trials`);
* the in-cloud damping strength the field prefers is four times weaker than
  Ermak's and twelve times weaker than the laboratory limits require
  (`coefficients.c_mu_strat`);
* here, a sixfold stability response where the measurements show none.

Caveats
-------
* Trial 005 is excluded: the bag hung up during the drop and part of the gas
  escaped, so the released mass is not known.
* Trial 004 was neutrally buoyant (rho_r = 0.99) and is kept because it
  anchors the low-density end, but it is not a dense-gas release.
* Trials 014, 015 and 018 had air in the mixture; the report gives correction
  factors, which are recorded in the data file.
* The mixture is modelled as a non-condensing pseudo-component at the
  measured relative density.  The releases were isothermal, so no phase
  behaviour is involved.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..coefficients import PHYS, Coefficients, preset
from ..core.plume import run_dispersion
from ..core.source import InstantaneousRelease
from ..submodels.atmosphere import Atmosphere
from ..thermo.base import LegacyThermo, Substance, water_backend

__all__ = ["CONTAINER_RADIUS", "Trial", "load_trials", "peak_concentration",
           "sensitivity_to"]

DATA = Path(__file__).parent / "data"

from ._data_access import ObservationsUnavailable, require  # noqa: E402

#: Radius of the release container [m].  14 m diameter, 13 m tall.
CONTAINER_RADIUS = 7.0


@dataclass(frozen=True)
class Trial:
    """One instantaneous release."""

    trial: str
    u10: float
    stability: str
    volume: float
    rel_density: float
    note: str

    @property
    def stability_class(self) -> str:
        """Single letter; '?' becomes D and 'D/E' becomes D."""
        s = self.stability.replace("?", "D").replace("/", "")
        return s[0]

    def substance(self) -> Substance:
        """
        The mixture as a non-condensing pseudo-component.

        Only the molecular weight matters: the releases were isothermal and
        at ambient pressure, so nothing condenses and the boiling point is
        set far below any temperature reached.
        """
        return Substance(name="R12/N2", mw=self.rel_density * PHYS.MW_AIR,
                         cp_vapour=600.0, cp_liquid=900.0, dh_vap=1e5,
                         T_boil=50.0, rho_liquid=1300.0)

    def atmosphere(self, *, stability: str | None = None, z0: float = 0.005,
                   T: float = 288.0, rh: float = 70.0) -> Atmosphere:
        return Atmosphere(u_ref=self.u10, z_ref=10.0, T=T, rh=rh, z0=z0,
                          stability=stability or self.stability_class)

    def source(self, atm: Atmosphere) -> InstantaneousRelease:
        return InstantaneousRelease(
            substance=self.substance(), rate=0.0, duration=0.0,
            area=math.pi * CONTAINER_RADIUS**2,
            mass=self.volume * self.rel_density * atm.rho, height=0.0,
        )


def load_trials(*, usable_only: bool = True) -> list[Trial]:
    with open(require("thorney_island_phase1_conditions.csv"), newline="") as fh:
        rows = list(csv.DictReader(fh))
    return [Trial(trial=r["trial"], u10=float(r["u10_m_s"]),
                  stability=r["stability"], volume=float(r["volume_m3"]),
                  rel_density=float(r["rel_density"]), note=r["note"])
            for r in rows if not usable_only or r["usable"] == "True"]


def peak_concentration(trial: Trial, distances=(100.0, 300.0), *,
                       stability: str | None = None,
                       coeffs: Coefficients | None = None,
                       **kw) -> np.ndarray:
    """Predicted peak volume fraction at each distance."""
    atm = trial.atmosphere(stability=stability)
    src = trial.source(atm)
    sub = trial.substance()
    traj, _ = run_dispersion(src, atm, LegacyThermo(sub), water_backend(),
                             x_max=max(distances) * 2.5, t_max=2000.0,
                             n_puff_steps=60,
                             coeffs=coeffs or preset("ermak90"), **kw)
    logs = np.log(np.maximum(traj.vol_frac, 1e-30))
    return np.array([float(np.exp(np.interp(x, traj.x, logs)))
                     for x in distances])


#: Stability class as a number, for correlation.
_CLASS = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6}


def sensitivity_to(distance: float = 300.0, **kw) -> dict:
    """
    Correlation of the predicted peak with wind speed, stability and density.

    The report says the first two should show none.
    """
    trials = load_trials()
    c = np.array([peak_concentration(t, (distance,), **kw)[0] for t in trials])
    u = np.array([t.u10 for t in trials])
    s = np.array([_CLASS[t.stability_class] for t in trials], dtype=float)
    d = np.array([t.rel_density for t in trials])

    def r(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    return {"concentration": c, "wind": r(np.log(u), np.log(c)),
            "stability": r(s, np.log(c)),
            "density": r(np.log(d), np.log(c)),
            "spread": float(c.max() / c.min())}
