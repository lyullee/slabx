"""
Ground cooling under a cold cloud
=================================

SLAB holds the ground at ambient temperature for the entire release (EQ 39,
``T_g = T_a``).  Real ground beneath a cryogenic cloud cools, and the heat it
can supply falls with it.

The estimate
------------
Treating the ground as a semi-infinite solid drawn on at a roughly constant
surface flux q, the surface temperature falls as

    dT_s(t) = 2 q sqrt(alpha t / pi) / k,     alpha = k / (rho cp)

Which resistance dominates
--------------------------
Not the ground's, at first.  A step change at the surface would drive
37 000 W/m^2 after one second and 2 100 after five minutes, while the cloud's
own convective transfer draws only 500-600 W/m^2.  So for the first minutes
the *convective* film is the bottleneck and Ermak's fixed ground temperature
is a fair approximation; the two cross over at around 300 s, after which
conduction begins to limit.

Measured effect
---------------
On the Burro trials, against the observations:

    substrate      FAC2     MG      VG     NMSE
    none (Ermak)   0.67   0.606   1.616   0.750
    water          0.67   0.603   1.625   0.758
    dry soil       0.58   0.601   1.638   0.770
    concrete       0.67   0.604   1.625   0.758

Negligible, and slightly the wrong way.  The stable trial improves a little
(MG 0.952 to 0.961 on dry soil) and the others move the other way by about
as much, which is what a change smaller than the residual scatter looks like.

That is the expected answer given the resistance argument above, and it is
worth having established rather than assumed: Ermak's fixed ground
temperature is not an approximation that costs anything on releases of this
length.  On a release lasting tens of minutes, or over a substrate with a
lower effusivity than these, it would.

Burro was spilled onto a water pond, and water's effusivity is 2.6 times dry
soil's, so the surface there falls about 4 K rather than 10 — the smallest
case of an already small effect.

Properties
----------
Defaults are for the dry desert soil of the Nevada Test Site and China Lake.
Water is included because spills onto water are the other common case, and
its far larger effusivity is why a pool on water keeps boiling steadily while
one on land slows down.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["Substrate", "DRY_SOIL", "CONCRETE", "WATER_SUBSTRATE",
           "surface_cooling"]


@dataclass(frozen=True)
class Substrate:
    """Thermal properties of the surface under the cloud."""

    name: str
    conductivity: float      #: k [W/(m K)]
    density: float           #: rho [kg/m^3]
    heat_capacity: float     #: cp [J/(kg K)]

    @property
    def diffusivity(self) -> float:
        """alpha = k / (rho cp) [m^2/s]."""
        return self.conductivity / (self.density * self.heat_capacity)

    @property
    def effusivity(self) -> float:
        """
        sqrt(k rho cp) [J/(m^2 K s^1/2)].

        The property that decides how fast a surface cools under a given
        flux — water's is about six times dry soil's, which is why an LNG
        pool on water boils at a nearly steady rate and one on land does not.
        """
        return math.sqrt(self.conductivity * self.density * self.heat_capacity)


DRY_SOIL = Substrate("dry soil", 0.30, 1600.0, 800.0)
CONCRETE = Substrate("concrete", 1.40, 2300.0, 880.0)
WATER_SUBSTRATE = Substrate("water", 0.60, 1000.0, 4180.0)


def surface_cooling(flux: float, elapsed: float,
                    substrate: Substrate = DRY_SOIL,
                    *, max_drop: float = 200.0) -> float:
    """
    Drop in surface temperature after `elapsed` seconds at heat flux `flux`.

    The constant-flux solution for a semi-infinite solid.  Capped, because
    the expression grows without bound in time while the real surface cannot
    fall below the cloud's own temperature — beyond that point the flux would
    reverse, which this simple form does not represent.
    """
    if flux <= 0.0 or elapsed <= 0.0:
        return 0.0
    drop = (2.0 * flux * math.sqrt(substrate.diffusivity * elapsed / math.pi)
            / substrate.conductivity)
    return min(drop, max_drop)
