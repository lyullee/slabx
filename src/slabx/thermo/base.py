"""
Thermodynamic property backends
===============================

SLAB's thermodynamics rests on four assumptions:

1. ideal gas for the vapour phase,
2. constant specific heats,
3. an Antoine-form saturation pressure, and
4. local thermodynamic equilibrium between vapour and droplets.

Only (4) is structural — it is woven through the conservation equations.
Assumptions (1)-(3) are *property* assumptions and are isolated behind the
`ThermoBackend` protocol so they can be replaced without touching the
dispersion core.

    LegacyThermo    Ermak (1990) exactly.  Reference for ablation.
    CoolPropThermo  real property data (not yet implemented).

A backend supplies properties; it does not solve phase equilibrium.  That
lives in `equilibrium.py` and is shared by all backends.

Saturation pressure
-------------------
EQ 43a expresses the saturation pressure as a fraction of ambient:

    P_s(T) / P_a = exp(A - B / (T + C))

`B` and `C` are user inputs.  When `B` is not supplied SLAB falls back to
the Clapeyron relation (EQ 43b) and sets `C = 0`:

    B = dHe * M_s / R,      C = 0

`A` is never an input: it is fixed by requiring P_s(T_bp) = P_a (EQ 43c),

    A = B / (T_bp + C)

so the boiling point is honoured by construction.

Reference-state latent heats
----------------------------
Because the specific heats of vapour and liquid differ, the latent heat is
temperature dependent even under assumption (2).  SLAB carries the value
extrapolated to 0 K,

    dHe0 = dHe + T_bp (cp_liq - cp_vap)        (JS `dhe0`)
    dHw0 = dHw + 298.2 (cp_wl - cp_wv)         (JS `dhw0`)

so that ``dH(T) = dH0 + T (cp_vap - cp_liq)`` recovers the input value at
the reference temperature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..coefficients import PHYS

__all__ = [
    "T_REF_WATER",
    "Substance",
    "ThermoBackend",
    "LegacyThermo",
    "WATER",
    "water_backend",
]


# ===========================================================================
# substance description (the SLAB input block)
# ===========================================================================
@dataclass(frozen=True)
class Substance:
    """
    Released material, described by SLAB's nine input properties.

    Parameters
    ----------
    mw : molecular weight [kg/mol]                       (SLAB `WMS`)
    cp_vapour : vapour specific heat [J/(kg K)]          (SLAB `CPS`)
    cp_liquid : liquid specific heat [J/(kg K)]          (SLAB `CPSL`)
    dh_vap : latent heat at the boiling point [J/kg]     (SLAB `DHE`)
    T_boil : normal boiling point [K]                    (SLAB `TBP`)
    rho_liquid : liquid density [kg/m^3]                 (SLAB `RHOSL`)
    sat_B, sat_C : saturation-pressure constants [K]     (SLAB `SPB`, `SPC`)
        `sat_B <= 0` selects the Clapeyron default (EQ 43b).
    """

    name: str
    mw: float
    cp_vapour: float
    cp_liquid: float
    dh_vap: float
    T_boil: float
    rho_liquid: float
    sat_B: float = -1.0
    sat_C: float = 0.0
    T_reference: float | None = None
    """
    Temperature at which `dh_vap` is quoted [K].  Defaults to the boiling
    point, which is the convention for the released material.  Water is the
    exception: SLAB quotes its latent heat at 298.2 K, not at 373 K
    (SLAB_thermo, ``dhw0 = dhw + 298.2 (cpwl - cpwv)``).
    """

    # derived
    A: float = field(init=False)
    B: float = field(init=False)
    C: float = field(init=False)

    def __post_init__(self):
        for f, v in (("mw", self.mw), ("T_boil", self.T_boil),
                     ("rho_liquid", self.rho_liquid), ("dh_vap", self.dh_vap),
                     ("cp_vapour", self.cp_vapour), ("cp_liquid", self.cp_liquid)):
            if v <= 0:
                raise ValueError(f"{f} must be > 0, got {v}")

        if self.sat_B > 0.0:
            B, C = self.sat_B, self.sat_C
        else:                                   # EQ 43b, Clapeyron default
            B, C = self.dh_vap * self.mw / PHYS.R_GAS, 0.0
        if self.T_boil + C <= 0:
            raise ValueError("T_boil + sat_C must be > 0")
        object.__setattr__(self, "B", B)
        object.__setattr__(self, "C", C)
        object.__setattr__(self, "A", B / (self.T_boil + C))   # EQ 43c

    # ------------------------------------------------------------------
    @property
    def T_ref(self) -> float:
        """Temperature at which `dh_vap` is quoted."""
        return self.T_boil if self.T_reference is None else self.T_reference

    @property
    def dh_vap_ref(self) -> float:
        """
        Latent heat extrapolated to 0 K, dH0 (JS `dhe0` / `dhw0`).

        ``dH(T) = dH0 + T (cp_vap - cp_liq)`` recovers `dh_vap` at `T_ref`.
        Getting `T_ref` wrong is a silent 5 % error in the phase-change energy
        for water, which shifts cloud temperature by tenths of a kelvin — a
        difference that the buoyancy term ``rho - rho_a`` then amplifies
        roughly eightfold.
        """
        return self.dh_vap + self.T_ref * (self.cp_liquid - self.cp_vapour)

    def source_temperature(self, T: float, liquid_fraction: float = 0.0) -> float:
        """
        SLAB clamps the source temperature (SLAB.FOR L671):
        a release cannot be below its boiling point, and a two-phase release
        must be *at* it.
        """
        if liquid_fraction > 0.0:
            return self.T_boil
        return max(T, self.T_boil)


#: Reference temperature of SLAB's built-in water latent heat [K].
T_REF_WATER = 298.2

#: Water, with the constants SLAB hard-codes (SLAB.FOR L330-337).
#: `T_boil` is back-figured from the saturation constants so that
#: `saturation_ratio(T_boil) == 1`, but the latent heat is quoted at 298.2 K,
#: so `T_reference` must be set explicitly.
WATER = Substance(
    name="Water",
    mw=PHYS.MW_WATER,
    cp_vapour=PHYS.CP_WATER_VAP,
    cp_liquid=PHYS.CP_WATER_LIQ,
    dh_vap=PHYS.DH_WATER,
    T_boil=PHYS.SAT_B_WATER / PHYS.SAT_A_WATER,
    rho_liquid=PHYS.RHO_WATER_LIQ,
    sat_B=PHYS.SAT_B_WATER,
    sat_C=0.0,
    T_reference=T_REF_WATER,
)


# ===========================================================================
# backend protocol
# ===========================================================================
@runtime_checkable
class ThermoBackend(Protocol):
    """
    Temperature-dependent properties of one condensable species.

    All methods take temperature in K.  `saturation_ratio` is dimensionless
    (P_sat / P_ambient); everything else is SI.  Implementations must be pure
    functions of T so that results are reproducible and cacheable.
    """

    substance: Substance
    name: str

    def saturation_ratio(self, T: float) -> float:
        """P_sat(T) / P_ambient, EQ 43a."""

    def d_saturation_ratio(self, T: float) -> float:
        """d/dT of `saturation_ratio`; needed by the Newton solver."""

    def dh_vap(self, T: float) -> float:
        """Latent heat of vaporisation [J/kg] at T."""

    def dh_vap_datum(self, T: float) -> float:
        """
        Latent heat extrapolated to the 0 K datum, evaluated near `T`.

        SLAB's energy equation (EQ 41) splits the mixture enthalpy into a
        sensible part ``sum m_i cp_i T`` and a latent part ``dH0 m_v``, so the
        latent heat it needs is the one referred to the same 0 K datum as the
        sensible terms:

            dH0(T) = dH(T) - T (cp_vap(T) - cp_liq(T))

        With constant specific heats this is a constant and the split is
        exact.  With real properties it drifts slowly, so it is evaluated at
        the current temperature; that keeps the decomposition consistent
        without restructuring the conservation equations.
        """

    def cp_vapour(self, T: float) -> float:
        """Vapour specific heat [J/(kg K)] at T."""

    def cp_liquid(self, T: float) -> float:
        """Liquid specific heat [J/(kg K)] at T."""

    def rho_liquid(self, T: float) -> float:
        """Liquid density [kg/m^3] at T."""

    def surface_tension(self, T: float) -> float:
        """
        Liquid surface tension [N/m] at T, for the droplet break-up criterion.

        Needed only by the rainout submodel.  Backends without a measured
        curve should estimate rather than fail: a missing value would remove
        rainout entirely, which is a larger error than an approximate one.
        """


# ===========================================================================
# legacy backend
# ===========================================================================
@dataclass(frozen=True)
class LegacyThermo:
    """
    Ermak (1990) property model: Antoine saturation pressure, constant
    specific heats, constant liquid density.

    Linearisation above the boiling point
    -------------------------------------
    The exponential form diverges above T_boil, where P_sat > P_ambient has
    no physical meaning for an open cloud.  SLAB_thermo (L2252-2256) replaces
    it there with the tangent at the boiling point,

        f(T) = 1 + A (T - T_bp) / (T_bp + C),    f'(T) = A / (T_bp + C)

    which keeps the Newton iteration well behaved while leaving the
    sub-boiling branch untouched.  Reproduced here because it changes results
    for cryogenic releases warming through their boiling point.
    """

    substance: Substance

    @property
    def name(self) -> str:
        return f"legacy:{self.substance.name}"

    # -- saturation ------------------------------------------------------
    def saturation_ratio(self, T: float) -> float:
        s = self.substance
        denom = T + s.C
        if denom <= 0.0:
            return 0.0
        f = math.exp(s.A - s.B / denom)
        if f > 1.0:                                   # above boiling point
            return 1.0 + s.A * (T - s.T_boil) / (s.T_boil + s.C)
        return f

    def d_saturation_ratio(self, T: float) -> float:
        s = self.substance
        denom = T + s.C
        if denom <= 0.0:
            return 0.0
        f = math.exp(s.A - s.B / denom)
        if f > 1.0:
            return s.A / (s.T_boil + s.C)
        return s.B * f / (denom * denom)

    # -- caloric ---------------------------------------------------------
    def dh_vap(self, T: float) -> float:
        s = self.substance
        return s.dh_vap_ref + T * (s.cp_vapour - s.cp_liquid)

    def cp_vapour(self, T: float) -> float:
        return self.substance.cp_vapour

    def cp_liquid(self, T: float) -> float:
        return self.substance.cp_liquid

    def rho_liquid(self, T: float) -> float:
        return self.substance.rho_liquid

    def surface_tension(self, T: float) -> float:
        """
        Estimated, since Ermak's input block carries no surface tension.

        Uses the Brock-Bird corresponding-states correlation with the
        critical temperature taken as ``T_boil / 0.6`` (Guldberg's rule) and
        a representative critical pressure.

        Accuracy is respectable for non-polar fluids (methane: 12.3 against a
        measured 12.9 mN/m) and poor for polar ones (ammonia: 15.8 against
        34.2), because corresponding states does not know about hydrogen
        bonding.  `CoolPropThermo` uses the measured curve wherever CoolProp
        has one and falls back to this only when it does not — chlorine being
        the case that arises in practice.
        """
        T_c = self.substance.T_boil / 0.6
        if T >= T_c:
            return 0.0
        T_r = T / T_c
        T_br = self.substance.T_boil / T_c
        # Brock & Bird (1955), with P_c in bar
        p_c = 45.0
        Q = 0.1196 * (1.0 + T_br * math.log(p_c / 1.01325)
                      / (1.0 - T_br)) - 0.279
        sigma = (p_c ** (2.0 / 3.0) * T_c ** (1.0 / 3.0) * Q
                 * (1.0 - T_r) ** (11.0 / 9.0)) * 1e-3
        return max(sigma, 0.0)

    def dh_vap_datum(self, T: float) -> float:
        """Constant by construction; the T-dependence cancels exactly."""
        return self.substance.dh_vap_ref


def water_backend() -> LegacyThermo:
    """Water backend with SLAB's built-in constants."""
    return LegacyThermo(WATER)
