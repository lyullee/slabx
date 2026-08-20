"""
Ambient atmosphere submodel
===========================

Pasquill class <-> inverse Monin-Obukhov length, mixing-layer height, the
ambient wind profile and the friction velocity.

Equation numbers refer to UCRL-MA-105607 (Ermak 1990) section 2.5.1;
`TNO` refers to CPR 14E chapter 4.  Where the scanned manual is illegible
the original Fortran (SLAB.FOR; a JavaScript transliteration was used
Fortran) was used and is cited by line number.

Wind profile
------------
The profile derives from the assumed gradient (EQ 32 / TNO 4.119)

    du_a/dz = (u* / (k z)) * Phi_m(z/L) * g(z/H),     g = 1 - z/H

which SLAB integrates in closed form.  `dimensionless_profile` returns the
integral, i.e. u_a(z) = (u*/k) * dimensionless_profile(z).  Below
z_t = e*z0 the profile is replaced by a quadratic matched in value and
slope at z_t (EQ 34c), because the log form is meaningless there.

Known defects in the reference implementation
---------------------------------------------
Two defects were found in that JavaScript while porting.  Both
are reproducible on demand so that a bit-level comparison against the
reference remains possible; both are OFF by default.

1. ``legacy_stability_bug`` — line ~123 reads ``else if (all < alm)`` where
   a misspelling of it is an undeclared identifier; the intended variable is the intended variable.
   In JavaScript the comparison is always false, so every case with
   ``aal >= al2`` collapses to the most stable class (astb = 3.5).

   **This is a defect of the JavaScript transcription, not of SLAB.**
   Ermak's Fortran reads the correctly spelled variable at that branch (SLAB.FOR line 236)
   and uses the intended variable in all ten places.  The flag is kept because the
   golden-reference harness compares against the JavaScript, and reproducing
   its behaviour is what makes that comparison meaningful — but the default
   (flag off) is the original.

2. ``legacy_unstable_profile`` — **confirmed present in Ermak's Fortran**
   (SLAB.FOR lines 1860-1862), so unlike the stability typo this one is
   original.  SLAB_uafn's unstable branch reads

       ... - 2(1-phi)*( log((1+u)/(1+u0)) - (sqrt(u)-sqrt(u0))/(gu*H) )

   whereas the exact integral of Phi_m*(1 - z/H)/z is

       ... - 2(1-phi)*( log((1+u)/(1+u0)) + (u - u0)/(gu*H) )

   i.e. the sign is flipped and an extra square root has crept in.  Because
   gu*H is large the resulting error is only 0.002-0.02 % over 1-200 m, but
   it exceeds the tolerance used for golden-reference comparison.

A third, non-code issue: the manual's printed ambient density for example 1
(1.1623 kg/m^3) is inconsistent with its own equation of state and with the
other printed values, which give 1.1523.  This is one of the 5/6 OCR
confusions visible throughout the scan.

Known model inconsistency (not a bug)
-------------------------------------
``StabilityMap`` is not exactly invertible for 2 < |s-4| < 3.5.  The forward
map interpolates the exponent in the intended variable space, the reverse map in ``astb``
space; the two agree only at the anchor points |s-4| = 2 and 3.  The
resulting round-trip error is a few percent in s.  This is inherent to
Ermak (1990) and is reproduced faithfully.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..coefficients import PHYS, Coefficients, COEFFS

__all__ = ["StabilityMap", "Atmosphere"]


# ===========================================================================
# Pasquill class  <->  inverse Monin-Obukhov length
# ===========================================================================
@dataclass(frozen=True)
class StabilityMap:
    """
    Golder (1976)-type mapping between Pasquill class and 1/L, as fitted in
    SLAB (SLAB.FOR L228-268).

    The class index used internally is ``s`` with A=1 ... F=6; SLAB accepts
    0.5 <= s <= 7.5.  The signed offset ``stb = s - 4`` is negative for
    unstable and positive for stable conditions.
    """

    z0: float

    al1: float = field(init=False)
    al2: float = field(init=False)
    al3: float = field(init=False)
    alm: float = field(init=False)
    en2: float = field(init=False)
    eni: float = field(init=False)
    dln: float = field(init=False)

    def __post_init__(self):
        if self.z0 <= 0:
            raise ValueError("surface roughness z0 must be > 0")
        z0 = self.z0
        al1 = 0.0081 / z0**0.3044
        if z0 > 0.0111:
            al2 = 0.0385 / z0**0.1715
            al3 = 0.0875 / z0**0.1028
        else:
            al2 = al1 + 0.0137 / z0**0.1715 + 0.0218
            al3 = al2 + 0.0557

        en2 = math.log(al2 / al1) / math.log(2.0)
        en3 = math.log(al3 / al1) / math.log(3.0)
        eni = en3 + en3 - en2
        dln = en2 - eni
        alm = al1 * 3.5 ** (eni + dln / 3.25)

        for k, v in dict(al1=al1, al2=al2, al3=al3, alm=alm,
                         en2=en2, eni=eni, dln=dln).items():
            object.__setattr__(self, k, v)

    # ------------------------------------------------------------------
    def stability_from_inv_L(self, inv_L: float, *, legacy_bug: bool = False) -> float:
        """Pasquill index s (A=1..F=6) from inverse MO length [1/m]."""
        aal = abs(inv_L)
        if aal < self.al2:
            astb = (aal / self.al1) ** (1.0 / self.en2)
        elif (not legacy_bug) and aal < self.alm:
            ral = (aal - self.al2) / (self.al3 - self.al2)
            en = self.eni + self.dln / (1.0 + ral * ral)
            astb = (aal / self.al1) ** (1.0 / en)
        else:
            astb = 3.5
        stb = astb if inv_L >= 0 else -astb
        return 4.0 + stb

    def inv_L_from_stability(self, s: float) -> float:
        """Inverse MO length [1/m] from Pasquill index s (A=1..F=6)."""
        stb = s - 4.0
        astb = abs(stb)
        if astb < 2.0:
            aal = self.al1 * astb**self.en2 if astb > 0 else 0.0
        elif astb < 3.5:
            en = self.eni + self.dln / (1.0 + (astb - 2.0) ** 2)
            aal = self.al1 * astb**en
        else:
            aal = self.alm
        return aal if stb >= 0 else -aal


# ===========================================================================
# Atmosphere
# ===========================================================================
@dataclass
class Atmosphere:
    """
    Ambient conditions and the derived wind profile.

    Parameters
    ----------
    u_ref, z_ref : wind speed [m/s] measured at height [m]
    T : ambient temperature [K]
    rh : relative humidity [%]
    z0 : surface roughness length [m]
    stability : Pasquill class as a letter ('A'..'F'), an index 1..6, or None
    inv_L : inverse Monin-Obukhov length [1/m]; required if stability is None
    p : ambient pressure [Pa]
    """

    u_ref: float
    z_ref: float
    T: float
    rh: float = 0.0
    z0: float = 2e-4
    stability: str | float | None = None
    inv_L: float | None = None
    p: float = PHYS.P_ATM
    coeffs: Coefficients = COEFFS
    legacy_stability_bug: bool = False
    legacy_unstable_profile: bool = False

    # derived
    smap: StabilityMap = field(init=False, repr=False)
    s: float = field(init=False)
    inv_L_ground: float = field(init=False)
    inv_L_eff: float = field(init=False)
    z_L: float = field(init=False)
    h_mix: float = field(init=False)
    u_star: float = field(init=False)
    _zt: float = field(init=False, repr=False)
    _cu1: float = field(init=False, repr=False)
    _cu2: float = field(init=False, repr=False)
    _phi_inf: float = field(init=False, repr=False)
    _gu: float = field(init=False, repr=False)

    _CLASS = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0, "F": 6.0}

    # ------------------------------------------------------------------
    def __post_init__(self):
        c = self.coeffs
        # Input checks.  These exist because the failures they catch are
        # silent: a negative wind speed gives a negative friction velocity and
        # the run continues to a plausible-looking answer, and a reference
        # height below the roughness length inverts the logarithm and returns
        # u* = 33 m/s.  Neither raises anything on its own, and a user who
        # mistypes an input deck deserves an error rather than a number.
        if self.u_ref < 0:
            raise ValueError(
                f"wind speed must be >= 0, got {self.u_ref}; a negative value "
                "yields a negative friction velocity and a plausible but "
                "meaningless answer"
            )
        if self.z_ref <= self.z0:
            raise ValueError(
                f"reference height {self.z_ref} m must exceed the roughness "
                f"length {self.z0} m; below it the log profile inverts"
            )
        if self.T <= 0:
            raise ValueError(f"absolute temperature must be > 0, got {self.T}")
        if not 0.0 <= self.rh <= 100.0:
            raise ValueError(
                f"relative humidity must be a percentage in [0, 100], got "
                f"{self.rh}"
            )
        if self.p <= 0:
            raise ValueError(f"ambient pressure must be > 0, got {self.p}")
        from ..scope import check_scope
        check_scope(inv_L=self.inv_L if self.stability is None else None,
                    u_ref=self.u_ref, z0=self.z0, stacklevel=4)

        self.smap = StabilityMap(self.z0)

        # --- stability <-> 1/L -----------------------------------------
        if self.stability is None:
            if self.inv_L is None:
                raise ValueError("give either `stability` or `inv_L`")
            self.inv_L_ground = float(self.inv_L)
            self.s = self.smap.stability_from_inv_L(
                self.inv_L_ground, legacy_bug=self.legacy_stability_bug
            )
        else:
            s = self._CLASS.get(str(self.stability).strip().upper()) \
                if isinstance(self.stability, str) else float(self.stability)
            if s is None:
                raise ValueError(f"bad stability {self.stability!r}")
            if not 0.5 <= s <= 7.5:
                raise ValueError(f"stability index {s} outside SLAB range [0.5, 7.5]")
            self.s = s
            self.inv_L_ground = self.smap.inv_L_from_stability(s)

        stb = self.s - 4.0

        # --- limiter z_L (section 2.5.1) --------------------------------
        # stable   : z_L = 1 + 0.8 (s - 4)          [JS L278]
        # unstable : z_L = exp(3.2 - 0.8 s)         [JS L279, EQ 33b]
        self.z_L = (1.0 + 0.8 * stb) if self.s >= 4.0 else math.exp(3.2 - 0.8 * self.s)

        # --- mixing layer height, H = 130 * 2^(7-s) = 130 * 2^(3-stb) ---
        self.h_mix = c.mix_height_ref * 2.0 ** (3.0 - stb)

        # --- effective 1/L, evaluated once at the measurement height ----
        # JS L285: ala0 = ala * (1 + z/zl) with z = max(z_ref, 3)
        z_eval = max(self.z_ref, 3.0)
        self.inv_L_eff = self.inv_L_ground * (1.0 + z_eval / self.z_L)

        # --- unstable profile parameters (EQ 33b) -----------------------
        if self.inv_L_eff < 0.0:
            self._phi_inf = 1.0 / math.sqrt(
                math.sqrt(1.0 - c.phi_unstable * self.z_L * self.inv_L_eff)
            )
            self._gu = -8.0 * self.inv_L_eff / (1.0 - self._phi_inf)
        else:
            self._phi_inf = 1.0
            self._gu = 0.0

        # --- near-ground quadratic match at z_t = e*z0 (EQ 34c) ---------
        self._zt = c.z_transition * self.z0
        self._cu1, self._cu2 = 1.0 / self._zt, 0.0   # placeholders; unused at z=zt
        uf = self.dimensionless_profile(self._zt)
        phm = self.phi_m(self._zt)
        ufp = phm * (1.0 - self._zt / self.h_mix) / self._zt
        self._cu1 = (2.0 * uf - self._zt * ufp) / self._zt
        self._cu2 = (self._zt * ufp - uf) / (self._zt * self._zt)

        # --- friction velocity ------------------------------------------
        denom = self.dimensionless_profile(self.z_ref)
        if denom <= 0:
            raise ValueError(
                f"non-positive wind profile integral at z_ref={self.z_ref}; "
                "check z0 and z_ref"
            )
        self.u_star = PHYS.VON_KARMAN * self.u_ref / denom

    # ------------------------------------------------------------------
    # profile functions
    # ------------------------------------------------------------------
    def phi_m(self, z: float) -> float:
        """Momentum Monin-Obukhov function Phi_m, EQ 33a/33b."""
        c = self.coeffs
        if self.inv_L_eff < 0.0:
            return self._phi_inf + (1.0 - self._phi_inf) / math.sqrt(1.0 + self._gu * z)
        return 1.0 + c.phi_stable * self.inv_L_eff * z / (1.0 + z / self.z_L)

    def dimensionless_profile(self, z: float) -> float:
        """
        Integral of the wind gradient: u_a(z) = (u*/k) * this.

        Mirrors SLAB_uafn (SLAB.FOR L1848-1870).
        """
        if z < self._zt:
            return self._cu1 * z + self._cu2 * z * z

        z0, H = self.z0, self.h_mix
        if self.inv_L_eff < 0.0:                       # unstable, EQ 34b
            phi, gu = self._phi_inf, self._gu
            u = math.sqrt(1.0 + gu * z)
            u0 = math.sqrt(1.0 + gu * z0)
            if self.legacy_unstable_profile:
                tail = -(math.sqrt(u) - math.sqrt(u0)) / (gu * H)   # SLAB_uafn
            else:
                tail = (u - u0) / (gu * H)                          # exact
            return (
                math.log(z / z0)
                - phi * (z - z0) / H
                - 2.0 * (1.0 - phi) * (math.log((1.0 + u) / (1.0 + u0)) + tail)
            )
        # stable / neutral
        zL = self.z_L
        return (
            math.log(z / z0)
            - (z - z0) / H
            + self.coeffs.phi_stable * self.inv_L_eff * zL * (
                (1.0 + zL / H) * math.log((z + zL) / (z0 + zL)) - (z - z0) / H
            )
        )

    def wind_speed(self, z: float) -> float:
        """Ambient wind speed [m/s] at height z [m]."""
        return self.u_star / PHYS.VON_KARMAN * self.dimensionless_profile(z)

    def wind_gradient(self, z: float) -> float:
        """du_a/dz [1/s] from EQ 32 — used to verify the closed form."""
        z = max(z, self._zt)
        return (self.u_star / (PHYS.VON_KARMAN * z)) * self.phi_m(z) * (1.0 - z / self.h_mix)

    def mean_wind_over(self, h: float, n: int = 2) -> float:
        """
        Depth-averaged ambient wind over [0, h] by Simpson's rule (EQ 4c).
        SLAB uses the 3-point rule (n=2); higher even n available for testing.
        """
        if h <= 0:
            raise ValueError("h must be > 0")
        if n % 2:
            raise ValueError("Simpson's rule needs an even number of intervals")
        dz = h / n
        total = self.wind_speed(1e-12) + self.wind_speed(h)
        for i in range(1, n):
            total += (4.0 if i % 2 else 2.0) * self.wind_speed(i * dz)
        return total * dz / 3.0 / h

    # ------------------------------------------------------------------
    @property
    def rho(self) -> float:
        """Moist-air density [kg/m^3] using SLAB's effective molecular weight."""
        return self.mw_moist * self.p / (PHYS.R_GAS * self.T)

    @property
    def mass_frac_water(self) -> float:
        """Ambient water-vapour mass fraction (SLAB.FOR L338-345)."""
        pw = 0.01 * self.rh * math.exp(PHYS.SAT_A_WATER - PHYS.SAT_B_WATER / self.T)
        return PHYS.MW_WATER * pw / (PHYS.MW_AIR + (PHYS.MW_WATER - PHYS.MW_AIR) * pw)

    @property
    def mw_moist(self) -> float:
        """Effective molecular weight of moist air [kg/mol], EQ 10."""
        cw = self.mass_frac_water
        return PHYS.MW_AIR * PHYS.MW_WATER / (
            PHYS.MW_WATER + (PHYS.MW_AIR - PHYS.MW_WATER) * cw
        )

    @property
    def cp_moist(self) -> float:
        """Specific heat of moist air [J/(kg*K)], EQ 41."""
        cw = self.mass_frac_water
        return (1.0 - cw) * PHYS.CP_AIR + cw * PHYS.CP_WATER_VAP
