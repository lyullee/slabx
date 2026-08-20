"""
Multicomponent releases
=======================

SLAB carries one released species plus ambient water.  Real LNG is not one
species: the Burro trials were 87-93 % methane with the balance ethane and
propane, and the validation database says so — *"Goldwire et al. provided the
actual composition for the LNG mixture used in each trial, which is therefore
used to define the base case.  Given the common (and generally conservative)
practice of modelling LNG releases as pure methane, a sensitivity case with
100 % methane is included"* (03903-RP-002, section 2.4.5.6).

Every run in this project so far has been the sensitivity case.

Why it matters for a dense gas
------------------------------
A 87.4/10.3/2.3 methane/ethane/propane mixture, against pure methane:

    molecular weight    18.13 vs 16.04 g/mol      +13 %
    latent heat         633 vs 511 kJ/kg          +24 %
    bubble point        113.2 vs 111.7 K          +1.5 K
    liquid density      461 vs 422 kg/m3          +9 %

Density scales with molecular weight at fixed temperature, so the cloud is an
eighth heavier than the pure-methane model says before any thermodynamics is
done at all.  For a model whose entire behaviour turns on the density excess
over ambient, that is not a detail.

What is and is not implemented
-------------------------------
The mixture enters as a **pseudo-component**: CoolProp evaluates the real
mixture equation of state and the result is used in the existing
single-species framework.  That captures the molecular weight, the caloric
properties, the bubble point and the saturation curve of the actual mixture.

It does *not* capture differential condensation — propane condensing out
first and leaving a lighter vapour behind — because the conservation
equations carry one emission mass fraction.  That is a genuine limitation
and the reason this is called a pseudo-component rather than a
multicomponent model.

The predictions, registered before the comparison
--------------------------------------------------
M1  **Concentrations rise.**  A heavier, colder cloud is denser, entrains
    less per unit distance, and stays more concentrated.  Against the Burro
    observations, which the model already over-predicts, agreement must get
    *worse*: MG falls further below 1.

M2  **The effect scales with the heavy fraction.**  BU03 is 92.5 % methane,
    BU07 and BU08 about 87 %.  BU03 must move least.

M3  **No effect on a genuinely pure release.**  Passing a single-component
    mixture must reproduce the pure-substance result exactly — the negative
    control.

M1 is the uncomfortable one and it is registered deliberately.  The
validation database calls pure methane "generally conservative", which means
it expects the real composition to give *lower* hazard distances, not higher.
If M1 holds, then either the conventional wisdom is about a different
quantity than arc-wise peak concentration, or the pseudo-component treatment
misses the mechanism that makes the real mixture disperse better — most
likely the differential condensation it cannot represent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["Composition", "BURRO_COMPOSITIONS", "PREDICTIONS",
           "substance_from"]


@dataclass(frozen=True)
class Composition:
    """A mole-fraction mixture, as a CoolProp fluid string."""

    name: str
    components: tuple[tuple[str, float], ...]

    def __post_init__(self):
        total = sum(x for _, x in self.components)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"mole fractions sum to {total}, not 1")

    @property
    def is_pure(self) -> bool:
        return len(self.components) == 1

    def coolprop(self) -> str:
        """CoolProp fluid string, e.g. ``Methane[0.874]&Ethane[0.103]``."""
        if self.is_pure:
            return self.components[0][0]
        return "&".join(f"{c}[{x:.6f}]" for c, x in self.components)

    @classmethod
    def from_percentages(cls, name: str, **pct: float) -> "Composition":
        """
        Build from mole percentages, e.g. ``Methane=87.4, Ethane=10.3``.

        Renormalised, since published compositions rarely sum to exactly 100.
        """
        total = sum(pct.values())
        if total <= 0:
            raise ValueError("no components")
        return cls(name, tuple((k, v / total) for k, v in pct.items() if v > 0))


#: Trial compositions from Goldwire et al., as tabulated in 03903-RP-002
#: Table 2-8.  BU09 is not reported there; the BU08 composition is the
#: closest documented and is used for it, which is recorded rather than
#: hidden because it is an assumption.
BURRO_COMPOSITIONS = {
    "BU03": Composition.from_percentages("BU03 LNG", Methane=92.5, Ethane=6.2,
                                         Propane=1.3),
    "BU07": Composition.from_percentages("BU07 LNG", Methane=87.0, Ethane=10.4,
                                         Propane=2.6),
    "BU08": Composition.from_percentages("BU08 LNG", Methane=87.4, Ethane=10.3,
                                         Propane=2.3),
    "BU09": Composition.from_percentages("BU09 LNG", Methane=87.4, Ethane=10.3,
                                         Propane=2.3),
    "pure": Composition("pure methane", (("Methane", 1.0),)),
}

PREDICTIONS = (
    ("M1", "concentrations rise; agreement with Burro gets worse"),
    ("M2", "effect scales with the heavy fraction; BU03 moves least"),
    ("M3", "a single-component mixture reproduces the pure result exactly"),
)


def substance_from(comp: Composition, *, p_ambient: float = 101325.0):
    """
    Build a `Substance` whose properties are those of the real mixture.

    The input block SLAB expects — molecular weight, specific heats, latent
    heat, boiling point, liquid density, saturation constants — is filled
    from the mixture equation of state at its bubble point, so a
    pseudo-component run needs no hand-entered numbers and no guessing at
    what "the" boiling point of an LNG blend is.

    The saturation constants are fitted rather than taken from a table: two
    points on the mixture's own saturation curve, at the bubble point and
    10 K below, determine the Antoine pair ``B`` and ``C`` that the legacy
    backend needs.  A `CoolPropThermo` built on the same composition ignores
    them and uses the curve directly.
    """
    import CoolProp.CoolProp as CP

    from .base import Substance

    fluid = comp.coolprop()
    T_b = float(CP.PropsSI("T", "P", p_ambient, "Q", 0, fluid))
    mw = float(CP.PropsSI("M", fluid))

    h_g = float(CP.PropsSI("H", "P", p_ambient, "Q", 1, fluid))
    h_l = float(CP.PropsSI("H", "P", p_ambient, "Q", 0, fluid))
    cp_l = float(CP.PropsSI("CPMASS", "P", p_ambient, "Q", 0, fluid))
    rho_l = float(CP.PropsSI("D", "P", p_ambient, "Q", 0, fluid))
    # Ideal-gas cp, evaluated just off the saturation line: for a pure fluid
    # the bubble point *is* the saturation temperature at this pressure and
    # CoolProp refuses the query as ambiguous.
    cp_v = float(CP.PropsSI("Cp0mass", "T", T_b + 0.5, "P", p_ambient, fluid))

    # Two-point Antoine fit on the mixture's own saturation line.
    T2 = T_b - 10.0
    p2 = float(CP.PropsSI("P", "T", T2, "Q", 0, fluid))
    # ln(p/p_a) = A - B/T with A = B/T_b (EQ 43a with C = 0), so one extra
    # point fixes B.
    B = -math.log(p2 / p_ambient) * T_b * T2 / (T_b - T2)

    return Substance(
        name=comp.name, mw=mw, cp_vapour=cp_v, cp_liquid=cp_l,
        dh_vap=h_g - h_l, T_boil=T_b, rho_liquid=rho_l,
        sat_B=B, sat_C=0.0,
    )
