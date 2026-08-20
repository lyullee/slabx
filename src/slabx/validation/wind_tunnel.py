"""
Wind-tunnel trials, and whether the model is scale-similar
==========================================================

Three unobstructed wind-tunnel releases from the flammable MVD — two SF6 at
BA-Hamburg, one CO2 at CHRC — at rates of 1.7e-4 to 1.1e-2 kg/s and reference
heights of 7 to 67 mm.  Every deck the model has been run on until now was
field scale, four to six orders of magnitude larger.

Two things are being asked
--------------------------
1. Does the model run at all down there?  It does: all three complete and
   produce monotonically decaying, physical concentrations.

2. Is it *scale-similar*?  It is not, and that is the finding.

The similarity test
-------------------
Densimetric Froude similarity is what lets a tunnel result stand for a field
release: scale lengths by s, velocities by sqrt(s) and the release rate by
s^2.5, and the dimensionless concentration field should be unchanged.
Starting from BA-Hamburg DAT223 and scaling up to s = 1000:

    s        C at x = 10 L    relative
    1           0.463 %        1.000
    10          0.544 %        1.177
    100         0.699 %        1.511
    1000        1.106 %        2.391

A factor 2.4 where similarity requires 1.0.

Where it comes from
-------------------
EQ 35a scales the vertical entrainment by ``U_r / U_a(h_top)``, with ``U_r``
the dimensionless wind profile evaluated at a *fixed* height — 4 m, hard-coded
in the reference.  An absolute length in an otherwise similarity-based
closure cannot survive a change of scale.  Scaling it along with the geometry
takes the 2.4 down to 1.6:

    everything fixed          2.391
    + h_entrain_ref scaled    1.571
    + mixing height scaled    1.571   (no further effect)

So the reference height accounts for most of it and something else — the
source-region treatment and the roughness-dependent profile are the
candidates — for the rest.

Why it matters
--------------
It means tunnel and field results cannot be transferred through this model,
which is why the MEP requires wind-tunnel trials to be run at tunnel scale
"to avoid uncertainties associated with scaling experimental data".  It is
also consistent with what PHMSA reports for Phast: unbiased on the field
trials, under-predicting the scaled wind-tunnel trials by a factor of two to
three.

This is the same absolute-height factor already noted in
`examples/entrainment_limits.py`, where it makes the passive-dispersion limit
vary from 1.63 to 0.85 as the cloud depth goes from 0.5 m to 16 m.  Here it
is measured as a consequence rather than as a property.

Caveats
-------
* The obstructed and sloped trials are recorded but marked inapplicable: the
  model has no representation of fences, dikes or terrain.
* Source areas are not given in the MVD extract used here and are assumed;
  the similarity test does not depend on the assumption, since it scales
  whatever area is chosen.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..coefficients import COEFFS, Coefficients
from ..core.plume import run_dispersion
from ..core.source import EvaporatingPool
from ..submodels.atmosphere import Atmosphere
from ..thermo.base import LegacyThermo, Substance, water_backend

__all__ = ["SF6", "CO2", "Tunnel", "load_tunnel_trials", "froude_scaling"]

DATA = Path(__file__).parent / "data"

from ._data_access import ObservationsUnavailable, require  # noqa: E402

SF6 = Substance(name="SF6", mw=0.14606, cp_vapour=665.0, cp_liquid=900.0,
                dh_vap=1e5, T_boil=209.3, rho_liquid=1880.0)
CO2 = Substance(name="CO2", mw=0.044009, cp_vapour=846.0, cp_liquid=1900.0,
                dh_vap=5.7e5, T_boil=194.7, rho_liquid=1180.0)


@dataclass(frozen=True)
class Tunnel:
    facility: str
    trial: str
    gas: str
    rate: float
    u_ref: float
    z_ref: float
    z0: float
    obstruction: str
    applicable: bool

    def substance(self) -> Substance:
        return SF6 if self.gas == "SF6" else CO2


def load_tunnel_trials(*, applicable_only: bool = True) -> list[Tunnel]:
    with open(require("wind_tunnel_trials.csv"), newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = [Tunnel(facility=r["facility"], trial=r["trial"], gas=r["gas"],
                  rate=float(r["rate_kg_s"]), u_ref=float(r["u_ref_m_s"]),
                  z_ref=float(r["z_ref_m"]), z0=float(r["z0_m"]),
                  obstruction=r["obstruction"],
                  applicable=r["slab_applicable"] == "True")
           for r in rows]
    return [t for t in out if t.applicable] if applicable_only else out


def froude_scaling(scales=(1.0, 10.0, 100.0, 1000.0), *,
                   length: float = 0.12, u_ref: float = 0.74,
                   rate: float = 8.72e-4, z_ref: float = 0.0072,
                   z0: float = 3.3e-5, area: float = 0.0113,
                   scale_reference_height: bool = False,
                   coeffs: Coefficients | None = None) -> np.ndarray:
    """
    Concentration at ``x = 10 L`` under Froude-similar scaling.

    Perfect similarity would return a constant array.  With
    `scale_reference_height` the absolute 4 m in EQ 35a is scaled too, which
    isolates how much of the departure it accounts for.
    """
    base = coeffs or COEFFS
    out = []
    for s in scales:
        c = base.perturb(name="froude",
                         h_entrain_ref=base.h_entrain_ref
                         * (s if scale_reference_height else 1.0))
        atm = Atmosphere(u_ref=u_ref * np.sqrt(s), z_ref=z_ref * s, T=293.0,
                         rh=50.0, z0=z0 * s, stability="D")
        src = EvaporatingPool(substance=SF6, rate=rate * s**2.5,
                              area=area * s * s, duration=1e5)
        traj, _ = run_dispersion(src, atm, LegacyThermo(SF6), water_backend(),
                                 x_max=20.0 * length * s, n_field_steps=60,
                                 n_puff_steps=20, coeffs=c)
        logs = np.log(np.maximum(traj.vol_frac, 1e-30))
        out.append(float(np.exp(np.interp(10.0 * length * s, traj.x, logs))))
    return np.array(out)
