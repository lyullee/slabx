"""
Input decks
===========

The five worked examples of the manual, and the material set the random
differential test draws from.

Kept apart from any harness so that neither the Fortran comparison nor the
fuzzer depends on the other. They were originally defined inside a script
that drove a JavaScript transcription of SLAB, which is no longer used;
see `golden/fortran/oracle.py` for why.

Field order follows the original's read sequence (SLAB.FOR L195-230), so a
`Case` maps onto an input deck without reordering.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slabx.submodels.atmosphere import Atmosphere          # noqa: E402
from slabx.thermo.base import Substance                    # noqa: E402

__all__ = ["Case", "CASES", "Fluid", "FLUIDS", "build_source"]


@dataclass(frozen=True)
class Case:
    """One SLAB input deck, in the order the original reads it (SLAB.FOR L195-230)."""

    name: str
    idspl: int
    ncalc: int
    wms: float
    cps: float
    tbp: float
    cmed0: float
    dhe: float
    cpsl: float
    rhosl: float
    spb: float
    spc: float
    ts: float
    qs: float
    as_: float
    tsd: float
    qtis: float
    hs: float
    tav: float
    xffm: float
    zp: float
    z0: float
    za: float
    ua: float
    ta: float
    rh: float
    stab: float
    ala: float
    n: int = 241
    fluid: str | None = None
    """CoolProp fluid name for the released material.  The case name is a
    label (`LNG-neutral`), not a substance, so it cannot be inferred."""
    t_max: float | None = None
    """Puff integration limit [s]; defaults to a generous multiple of x_max,
    which is far too long for a slow release over a 10 km domain."""

    def argv(self) -> list[str]:
        return [str(v) for v in (
            self.idspl, self.ncalc, self.wms, self.cps, self.tbp, self.cmed0,
            self.dhe, self.cpsl, self.rhosl, self.spb, self.spc, self.ts,
            self.qs, self.as_, self.tsd, self.qtis, self.hs, self.tav,
            self.xffm, self.zp, self.z0, self.za, self.ua, self.ta, self.rh,
            self.stab, self.ala, self.n)]

    def substance(self) -> Substance:
        return Substance(
            name=self.name, mw=self.wms, cp_vapour=self.cps,
            cp_liquid=self.cpsl, dh_vap=self.dhe, T_boil=self.tbp,
            rho_liquid=self.rhosl, sat_B=self.spb, sat_C=self.spc,
        )

    def atmosphere(self, legacy: bool) -> Atmosphere:
        kw = dict(u_ref=self.ua, z_ref=self.za, T=self.ta, rh=self.rh,
                  z0=self.z0)
        if self.stab == 0.0:
            kw["inv_L"] = self.ala
        else:
            kw["stability"] = self.stab
        return Atmosphere(**kw, legacy_stability_bug=legacy,
                          legacy_unstable_profile=legacy)


#: Manual example 1.  The published deck; only `dhe` differs from the value
#: usually quoted for LNG because the manual lists 609900 J/kg.
BURRO8 = Case(
    name="LNG", idspl=1, ncalc=1, wms=0.016043, cps=2238.0, tbp=111.70,
    cmed0=0.0, dhe=609900.0, cpsl=3348.50, rhosl=424.10, spb=-1.0, spc=0.0, fluid="Methane",
    ts=111.70, qs=117.00, as_=657.00, tsd=107.0, qtis=0.0, hs=0.0,
    tav=10.0, xffm=1000.0, zp=0.0, z0=0.000200, za=2.88, ua=1.92,
    ta=306.00, rh=4.60, stab=0.0, ala=0.0665,
)

#: Same release, neutral stability and a stronger wind (manual's second run).
BURRO8_NEUTRAL = Case(**{**BURRO8.__dict__, "name": "LNG-neutral",
                         "stab": 4.0, "ala": 0.0, "ua": 5.0})

#: Manual example 2 — two-phase horizontal ammonia jet, Desert Tortoise 4.
#: 81 % of the release is liquid at the orifice, so the droplet evaporation
#: makes an ammonia cloud denser than air despite MW 17 < 29.
DT4 = Case(
    name="Ammonia-jet", idspl=2, ncalc=1, wms=0.017031, cps=2045.9,
    tbp=239.57, cmed0=0.81, dhe=1170000.0, cpsl=4611.8, rhosl=603.0, fluid="Ammonia",
    spb=2976.01, spc=0.0, ts=239.57, qs=107.87, as_=0.93, tsd=381.0,
    qtis=0.0, hs=1.0, tav=10.0, xffm=2800.0, zp=0.0, z0=0.003, za=2.00,
    ua=4.50, ta=306.2, rh=21.3, stab=0.0, ala=0.0221,
)

#: Manual example 3 — instantaneous LNG release, same material as example 1.
LNG_INSTANT = Case(
    name="LNG-instant", idspl=4, ncalc=1, wms=0.016043, cps=2238.0,
    tbp=111.70, cmed0=0.0, dhe=509900.0, cpsl=3348.50, rhosl=424.10, fluid="Methane",
    spb=-1.0, spc=0.0, ts=111.70, qs=0.0, as_=900.0, tsd=0.0,
    qtis=5000.0, hs=0.0, tav=10.0, xffm=1000.0, zp=0.0, z0=0.000200,
    za=2.88, ua=1.92, ta=306.00, rh=4.60, stab=0.0, ala=0.0665,
)

#: Manual example 4 — vertical chlorine jet.  Denser than air at every
#: concentration, and the only deck that exercises the plume-rise submodel.
CL2_STACK = Case(
    name="Chlorine-stack", idspl=3, ncalc=1, wms=0.070906, cps=498.1,
    tbp=239.1, cmed0=0.88, dhe=287840.0, cpsl=926.3, rhosl=1574.0, fluid="Chlorine",
    spb=1978.34, spc=-27.01, ts=239.1, qs=3.33, as_=0.02, tsd=300.0,
    qtis=0.0, hs=1.0, tav=1.0, xffm=10000.0, zp=1.0, z0=0.1, za=10.0,
    ua=1.0, ta=276.0, rh=30.0, stab=4.0, ala=-1.0,
)

CASES = {
    "burro8": BURRO8,
    "burro8_neutral": BURRO8_NEUTRAL,
    "dt4": DT4,
    "lng_instant": LNG_INSTANT,
    "cl2_stack": CL2_STACK,
}


@dataclass(frozen=True)
class Fluid:
    """A substance with self-consistent properties."""

    name: str
    fluid: str
    wms: float
    cps: float
    cpsl: float
    dhe: float
    tbp: float
    rhosl: float
    spb: float = -1.0
    spc: float = 0.0


#: Materials with real property sets, spanning light-and-cold to
#: heavy-and-toxic.  Sampling molecular weight and boiling point
#: independently would produce fluids that do not exist.
FLUIDS = [
    Fluid("LNG", "Methane", 0.016043, 2238.0, 3348.5, 509900.0, 111.70, 424.1),
    Fluid("Ammonia", "Ammonia", 0.017031, 2045.9, 4611.8, 1170000.0, 239.57,
          603.0, 2976.01, 0.0),
    Fluid("Chlorine", "Chlorine", 0.070906, 498.1, 926.3, 287840.0, 239.1,
          1574.0, 1978.34, -27.01),
    Fluid("Propane", "n-Propane", 0.044096, 1670.0, 2520.0, 425700.0, 231.1,
          580.0),
    Fluid("Ethane", "Ethane", 0.030069, 1750.0, 2440.0, 489000.0, 184.6, 544.0),
]


# ===========================================================================
# Case -> slabx objects
# ===========================================================================
def build_source(case: "Case", substance=None):
    """The deck's source model.  `idspl` selects among the four types."""
    from slabx.core.source import (
        EvaporatingPool, HorizontalJet, InstantaneousRelease,
    )
    from slabx.core.vertical_jet import VerticalJet

    sub = substance if substance is not None else case.substance()
    common = dict(substance=sub, rate=case.qs, area=case.as_,
                  duration=case.tsd)

    if case.idspl == 1:
        return EvaporatingPool(**common)
    if case.idspl == 2:
        return HorizontalJet(liquid_fraction=case.cmed0, height=case.hs,
                             T_source=case.ts, **common)
    if case.idspl == 3:
        return VerticalJet(liquid_fraction=case.cmed0, height=case.hs,
                           T_source=case.ts, **common)
    if case.idspl == 4:
        return InstantaneousRelease(substance=sub, rate=0.0, duration=0.0,
                                    area=case.as_, mass=case.qtis,
                                    height=case.hs)
    raise ValueError(f"idspl must be 1-4, got {case.idspl}")
