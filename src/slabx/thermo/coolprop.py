"""
Real-fluid thermodynamic backend
================================

Replaces Ermak's three property assumptions — ideal gas, constant specific
heats, two-parameter Antoine saturation — with reference equations of state
via `CoolProp <http://coolprop.org>`_, while leaving the dispersion core and
the phase-equilibrium solver untouched.

What actually changes
---------------------
Four things, in rough order of how much they matter for a dense-gas release:

1. **Saturation pressure.**  SLAB fits a two-parameter Antoine curve through a
   single point — it forces ``P_sat(T_boil) = P_atm`` and takes the slope from
   Clausius-Clapeyron or a user constant.  That is accurate at the boiling
   point and drifts away from it, and the phase split is exactly where a
   cryogenic cloud spends its life.

2. **Latent heat.**  A constant, evaluated at the boiling point.  Real latent
   heat falls towards zero at the critical point; for ammonia SLAB's input is
   15 % below the true value at its own boiling point.

3. **Specific heats.**  Constants.  The vapour cp of a polar molecule like
   ammonia varies by tens of percent over the range a cloud traverses.

4. **Liquid density.**  A constant, which enters the equation of state
   through the droplet-volume term of EQ 10.  SLAB's ammonia value is 12 %
   below reality.

What does *not* change
----------------------
The vapour phase is still treated as an ideal-gas mixture by EQ 10, and local
equilibrium is still assumed.  Both are structural: they are woven through
the conservation equations, not isolated behind this protocol.  A backend
cannot fix them, and claiming otherwise would overstate what this is.

Numerical cost
--------------
`PropsSI` is roughly a thousand times slower than evaluating a constant, and
the equilibrium solver calls into it a few times per Newton step, four times
per Runge-Kutta stage.  Results are cached on temperature, quantised to
`cache_resolution` kelvin, which makes the cost negligible while staying far
below the tolerance the solver converges to.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

from ..coefficients import PHYS
from .base import Substance

__all__ = ["CoolPropThermo", "TabulatedThermo", "tabulated",
           "coolprop_water", "COOLPROP_NAMES"]


#: Map from the substance names used in this project to CoolProp fluid names.
#: Extend as needed; `CoolPropThermo` also accepts an explicit `fluid`.
COOLPROP_NAMES: dict[str, str] = {
    "LNG": "Methane",
    "Methane": "Methane",
    "NH3": "Ammonia",
    "Ammonia": "Ammonia",
    "Cl2": "Chlorine",
    "Chlorine": "Chlorine",
    "Water": "Water",
    "Propane": "n-Propane",
    "Butane": "n-Butane",
    "Ethane": "Ethane",
    "H2": "Hydrogen",
    "Hydrogen": "Hydrogen",
}


def _import_coolprop():
    try:
        import CoolProp.CoolProp as CP
    except ImportError as exc:                       # pragma: no cover
        raise ImportError(
            "CoolPropThermo needs the CoolProp package: pip install CoolProp"
        ) from exc
    return CP


@dataclass(frozen=True)
class CoolPropThermo:
    """
    `ThermoBackend` backed by CoolProp reference equations of state.

    Parameters
    ----------
    substance : the same `Substance` the legacy backend uses.  Its molecular
        weight and boiling point are still used — the first by the equation of
        state, the second to identify the fluid and to anchor extrapolation —
        but its `cp_*`, `dh_vap`, `rho_liquid` and saturation constants are
        superseded by CoolProp.
    fluid : CoolProp fluid name.  Inferred from ``substance.name`` when
        omitted.
    cache_resolution : temperature quantisation for the property cache [K].
        1 mK is three orders below the solver's tolerance.
    p_ambient : pressure at which saturation and properties are evaluated [Pa].
    """

    substance: Substance
    fluid: str | None = None
    cache_resolution: float = 1e-3
    """
    Temperature quantisation for the property cache [K].

    1 mK is three orders below the equilibrium solver's tolerance, so it
    costs nothing in accuracy.  A mixture flash is roughly a hundred times
    more expensive than a pure-fluid one and its retry path can call CoolProp
    several times, so `for_mixture` loosens this to 0.05 K, still well inside
    the solver's tolerance.
    """
    p_ambient: float = PHYS.P_ATM

    _fluid: str = field(init=False, repr=False)
    _T_triple: float = field(init=False, repr=False)
    _T_crit: float = field(init=False, repr=False)

    def __post_init__(self):
        CP = _import_coolprop()
        fluid = self.fluid or COOLPROP_NAMES.get(self.substance.name)
        if fluid is None:
            raise KeyError(
                f"no CoolProp fluid known for {self.substance.name!r}; "
                "pass `fluid=` explicitly or add it to COOLPROP_NAMES"
            )
        try:
            T_t, T_c = _limits(fluid)
        except Exception as exc:                     # pragma: no cover
            raise ValueError(f"CoolProp rejected fluid {fluid!r}: {exc}") from exc
        if not (T_t > 0.0 and T_c > T_t):
            raise ValueError(
                f"CoolProp gave an unusable range for {fluid!r}: "
                f"triple {T_t}, critical {T_c}"
            )
        object.__setattr__(self, "_fluid", fluid)
        object.__setattr__(self, "_T_triple", float(T_t))
        object.__setattr__(self, "_T_crit", float(T_c))

    @classmethod
    def for_mixture(cls, substance: Substance, fluid: str, **kw):
        """A backend tuned for the cost of mixture property evaluation."""
        return cls(substance, fluid=fluid,
                   cache_resolution=kw.pop("cache_resolution", 0.05), **kw)

    @property
    def name(self) -> str:
        return f"coolprop:{self._fluid}"

    # ------------------------------------------------------------------
    def _clamp(self, T: float) -> float:
        """
        Keep two-phase queries inside the range CoolProp can answer.

        The saturation line only exists between the triple and critical
        points, and the equilibrium solver brackets its root over a much
        wider interval — it *will* evaluate unphysical temperatures on the
        way.  Those evaluations only need to be finite and monotone.

        The critical point is approached no closer than 0.1 K: saturated
        properties diverge there, and at 1e-3 K methane's saturated-vapour cp
        comes back as 1.5e8 J/(kg K).
        """
        lo = self._T_triple + 1e-6
        hi = self._T_crit - 0.1
        return min(max(T, lo), hi)

    def _q(self, T: float) -> float:
        """Quantised temperature for two-phase (saturation-line) queries."""
        r = self.cache_resolution
        return round(self._clamp(T) / r) * r

    def _q_free(self, T: float) -> float:
        """Quantised temperature for single-phase queries; no upper clamp."""
        r = self.cache_resolution
        return round(max(T, 1.0) / r) * r

    # -- saturation ------------------------------------------------------
    def saturation_ratio(self, T: float) -> float:
        """
        P_sat(T) / P_ambient from the reference EOS.

        Above the critical point the ratio is capped rather than undefined,
        and above the boiling point it is linearised exactly as the legacy
        backend does, so the Newton iteration behaves identically there.
        """
        if T >= self._T_crit:
            return _sat_ratio(self._fluid, self._q(self._T_crit),
                              self.p_ambient)
        return _sat_ratio(self._fluid, self._q(T), self.p_ambient)

    def d_saturation_ratio(self, T: float) -> float:
        """Central difference on the cached saturation curve."""
        h = max(1e-3 * abs(T), 1e-3)
        return (self.saturation_ratio(T + h) - self.saturation_ratio(T - h)) / (2 * h)

    # -- caloric ---------------------------------------------------------
    def dh_vap(self, T: float) -> float:
        """Zero above the critical point, where there is nothing to condense."""
        if T >= self._T_crit:
            return 0.0
        return _dh_vap(self._fluid, self._q(T))

    def cp_vapour(self, T: float) -> float:
        """
        Ideal-gas specific heat, not the saturated-vapour value.

        The released material in the cloud is a dilute component at a small
        partial pressure, so the ideal-gas limit is the physically right
        model — and unlike the saturation line it is defined at every
        temperature, including above the critical point.  Methane's critical
        temperature is 190.6 K, well inside the range an LNG cloud passes
        through, so this is not an edge case.
        """
        return _cp0(self._fluid, self._q_free(T))

    def cp_liquid(self, T: float) -> float:
        return _cp_liquid(self._fluid, self._q(T))

    def rho_liquid(self, T: float) -> float:
        return _rho_liquid(self._fluid, self._q(T))

    def surface_tension(self, T: float) -> float:
        """
        Measured curve where CoolProp has one, correlation where it does not.

        Chlorine is the case in point: CoolProp carries its equation of state
        but not its surface tension, and returning zero would silently switch
        rainout off for exactly the substance that most needs it.
        """
        v = _surface_tension(self._fluid, self._q(T))
        if v > 0.0:
            return v
        from .base import LegacyThermo
        return LegacyThermo(self.substance).surface_tension(T)

    def dh_vap_datum(self, T: float) -> float:
        """
        0 K-datum latent heat (see `ThermoBackend`), frozen above the boiling
        point.

        The decomposition ``dH(T) - T (cp_v - cp_l)`` presumes a liquid phase.
        Above the boiling point there is nothing condensed for it to describe,
        and near the critical point it diverges — the real latent heat
        collapses to zero while the cp difference does not, which for methane
        at 250 K returns 1.3e8 J/kg instead of 6.7e5.

        Evaluating it at ``min(T, T_boil)`` keeps it well conditioned and
        loses nothing: the phase split only varies with temperature where the
        species is actually partly condensed, and above the boiling point the
        term multiplies a difference set by the transported composition
        rather than by T.
        """
        Tq = self._clamp(min(T, self.substance.T_boil))
        return max(self.dh_vap(Tq) - Tq * (self.cp_vapour(Tq) - self.cp_liquid(Tq)),
                   0.0)

    # -- diagnostics ------------------------------------------------------
    def compare_with_legacy(self, T: float) -> dict[str, tuple[float, float]]:
        """
        Property-by-property ``(legacy, real)`` at one temperature.

        Useful for deciding *before* a run whether a substance is one where
        the property assumptions matter.
        """
        from .base import LegacyThermo

        old = LegacyThermo(self.substance)
        return {
            "saturation_ratio": (old.saturation_ratio(T), self.saturation_ratio(T)),
            "dh_vap": (old.dh_vap(T), self.dh_vap(T)),
            "cp_vapour": (old.cp_vapour(T), self.cp_vapour(T)),
            "cp_liquid": (old.cp_liquid(T), self.cp_liquid(T)),
            "rho_liquid": (old.rho_liquid(T), self.rho_liquid(T)),
        }


# ---------------------------------------------------------------------------
# cached CoolProp calls.  Module-level so the cache is shared between
# backend instances for the same fluid.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=200_000)
def _sat_ratio(fluid: str, T: float, p_ambient: float) -> float:
    try:
        return _on_saturation(fluid, T, "P", 0) / p_ambient
    except Exception:                                # pragma: no cover
        return 0.0


@lru_cache(maxsize=200_000)
def _dh_vap(fluid: str, T: float) -> float:
    try:
        return (_on_saturation(fluid, T, "H", 1)
                - _on_saturation(fluid, T, "H", 0))
    except Exception:                                # pragma: no cover
        return 0.0


@lru_cache(maxsize=1024)
def _limits(fluid: str) -> tuple[float, float]:
    """
    Triple and critical temperature, cached.

    For a mixture CoolProp computes the critical point by solving for it, and
    the call takes about 0.6 s — six hundred times a property evaluation.
    It sits on the failure path of `_on_saturation`, which is taken at every
    temperature outside the two-phase range, so without this a single
    trajectory spends minutes rediscovering a constant.
    """
    CP = _import_coolprop()
    T_t = float(CP.PropsSI("Ttriple", fluid))
    try:
        return T_t, float(CP.PropsSI("Tcrit", fluid))
    except Exception:
        pass
    # For a mixture CoolProp solves for the critical point, and the solve
    # fails for some compositions.  Kay's rule — the mole-fraction average of
    # the pure critical temperatures — is close enough: the value is used
    # only to bound queries away from the critical region, not as a property.
    if "[" not in fluid:
        raise
    total = crit = 0.0
    for part in fluid.split("&"):
        name, _, frac = part.partition("[")
        x = float(frac.rstrip("]"))
        crit += x * float(CP.PropsSI("Tcrit", name))
        total += x
    return T_t, crit / total


def _on_saturation(fluid: str, T: float, prop: str, quality: int) -> float:
    """
    A saturated-phase property, retried at nearby temperatures on failure.

    A pure fluid's saturation line is a curve and CoolProp follows it
    reliably; a mixture's is a two-dimensional band, and its flash solver
    diverges at some temperatures inside the range it nominally covers.
    Falling over would remove the property entirely, so the query is retried
    at successively safer temperatures and only then gives up.
    """
    CP = _import_coolprop()
    T_t, T_c = _limits(fluid)
    lo, hi = T_t + 0.5, T_c - 0.5
    for T_try in (T, min(max(T, lo), hi), 0.5 * (lo + hi), lo):
        try:
            return float(CP.PropsSI(prop, "T", min(max(T_try, lo), hi),
                                    "Q", quality, fluid))
        except Exception:
            continue
    raise ValueError(f"no {prop} available for {fluid!r} near {T} K")


@lru_cache(maxsize=200_000)
def _cp_liquid(fluid: str, T: float) -> float:
    return _on_saturation(fluid, T, "CPMASS", 0)


@lru_cache(maxsize=200_000)
def _cp0(fluid: str, T: float) -> float:
    """
    Ideal-gas cp.

    `Cp0mass` fails in two ways.  For pure fluids CoolProp refuses it below
    the melting line (water below 273 K).  For mixtures it can be asked at a
    temperature where its flash solver diverges, and the fallback to
    saturated vapour then fails too because a mixture's saturation line is a
    band rather than a curve.  Both cases end at the value on the saturation
    line at the nearest temperature CoolProp will answer for, which for a
    dilute vapour is immaterial.
    """
    CP = _import_coolprop()
    for query in (("Cp0mass", "T", T, "P", 101325.0),):
        try:
            return float(CP.PropsSI(*query, fluid))
        except Exception:
            pass
    T_t, T_c = _limits(fluid)
    for T_try in (T, 0.5 * (T_t + T_c), T_c - 1.0):
        try:
            return float(CP.PropsSI(
                "CPMASS", "T", min(max(T_try, T_t + 0.5), T_c - 0.5), "Q", 1,
                fluid))
        except Exception:
            continue
    raise ValueError(f"no ideal-gas cp available for {fluid!r} at {T} K")


@lru_cache(maxsize=200_000)
def _surface_tension(fluid: str, T: float) -> float:
    CP = _import_coolprop()
    try:
        return float(CP.PropsSI("SURFACE_TENSION", "T", T, "Q", 0, fluid))
    except Exception:
        return 0.0


@lru_cache(maxsize=200_000)
def _rho_liquid(fluid: str, T: float) -> float:
    return _on_saturation(fluid, T, "D", 0)


def pseudo_component(name: str, composition: dict[str, float], *,
                     p_ambient: float = PHYS.P_ATM) -> Substance:
    """
    Collapse a real mixture into the single condensable species SLAB carries.

    SLAB's thermodynamics handles one released material plus water, so a real
    LNG — 87 to 93 % methane with ethane and propane — cannot be represented
    directly.  This evaluates the mixture's properties at its own bubble
    point and returns them as a `Substance`, which captures the part of the
    composition effect that matters most and leaves out the part that cannot
    be represented at all.

    What it captures
    ----------------
    The molecular weight, which is 8-14 % higher than methane's for the Burro
    compositions.  That enters twice: through the equation of state, where a
    heavier vapour makes a denser cloud, and through EQ 12, where the same
    *mass* fraction converts to a smaller *mole* fraction.  Field data is
    reported as mole fraction, so modelling LNG as pure methane over-predicts
    it systematically — which is why the practice is described as
    conservative.  Of a 21 % reduction on the Burro-8 composition, about half
    is the mole-fraction conversion and about half is dispersion.

    What it does not capture
    ------------------------
    Fractionation.  Propane condenses preferentially and the vapour
    composition drifts downwind; a pseudo-component has a fixed composition
    by construction.  For LNG at these concentrations the effect is second
    order, but for a mixture with widely separated boiling points it would
    not be.

    Parameters
    ----------
    composition : mole fractions keyed by CoolProp fluid name, e.g.
        ``{"Methane": 0.874, "Ethane": 0.103, "n-Propane": 0.023}``.
        Normalised internally.
    """
    CP = _import_coolprop()
    total = sum(composition.values())
    if total <= 0:
        raise ValueError("composition sums to zero")
    frac = {k: v / total for k, v in composition.items() if v > 0}
    if len(frac) == 1:
        fluid = next(iter(frac))
    else:
        fluid = "&".join(f"{k}[{v}]" for k, v in frac.items())

    T_b = float(CP.PropsSI("T", "P", p_ambient, "Q", 0, fluid))
    at = lambda k, q: float(CP.PropsSI(k, "T", T_b, "Q", q, fluid))  # noqa: E731
    # Ideal-gas cp at a low pressure: CoolProp refuses `Cp0mass` on the
    # saturation line itself.
    cp_v = float(CP.PropsSI("Cp0mass", "T", T_b, "P", 5000.0, fluid))

    return Substance(
        name=name,
        mw=at("M", 0),
        cp_vapour=cp_v,
        cp_liquid=at("CPMASS", 0),
        dh_vap=at("H", 1) - at("H", 0),
        T_boil=T_b,
        rho_liquid=at("D", 0),
    )


def coolprop_water() -> CoolPropThermo:
    """Water backend on real properties, to pair with a real-fluid emission."""
    from .base import WATER

    return CoolPropThermo(WATER, fluid="Water")


# ===========================================================================
@dataclass(frozen=True)
class TabulatedThermo:
    """
    Any backend, precomputed onto a temperature grid and interpolated.

    A mixture flash costs about twenty times a pure-fluid one, and when it
    fails — which it does at temperatures the equilibrium solver's bracket
    routinely visits — the retry path costs several more.  Across a run that
    is tens of thousands of calls and turns a two-second case into minutes.

    Building a table once removes the problem entirely: the properties are
    smooth functions of temperature, so linear interpolation on a grid fine
    enough to resolve them is indistinguishable from the real thing, and the
    failures are absorbed at build time where they can be handled once.

    Outside the grid the end values are held, matching what the underlying
    backends already do at the ends of their own validity ranges.
    """

    inner: CoolPropThermo
    T_min: float = 50.0
    T_max: float = 1500.0
    n_points: int = 600

    _grid: tuple = field(init=False, repr=False)

    def __post_init__(self):
        import numpy as np

        T = np.geomspace(self.T_min, self.T_max, self.n_points)
        cols = {}
        for name, fn in (("sat", self.inner.saturation_ratio),
                         ("cpv", self.inner.cp_vapour),
                         ("cpl", self.inner.cp_liquid),
                         ("rhol", self.inner.rho_liquid),
                         ("dh", self.inner.dh_vap),
                         ("datum", self.inner.dh_vap_datum),
                         ("sigma", self.inner.surface_tension)):
            vals = np.empty_like(T)
            for i, t in enumerate(T):
                try:
                    vals[i] = fn(float(t))
                except Exception:                    # pragma: no cover
                    vals[i] = np.nan
            # Fill any gap the backend left, so a single awkward temperature
            # does not put a hole in the middle of the table.
            bad = ~np.isfinite(vals)
            if bad.any() and (~bad).any():
                vals[bad] = np.interp(T[bad], T[~bad], vals[~bad])
            cols[name] = np.nan_to_num(vals)
        object.__setattr__(self, "_grid", (T, cols))

    @property
    def substance(self) -> Substance:
        return self.inner.substance

    @property
    def name(self) -> str:
        return f"tabulated:{self.inner.name}"

    def _at(self, key: str, T: float) -> float:
        import numpy as np

        grid, cols = self._grid
        return float(np.interp(T, grid, cols[key]))

    def saturation_ratio(self, T: float) -> float:
        return self._at("sat", T)

    def d_saturation_ratio(self, T: float) -> float:
        h = max(1e-3 * abs(T), 1e-3)
        return (self.saturation_ratio(T + h) - self.saturation_ratio(T - h)) / (2 * h)

    def cp_vapour(self, T: float) -> float:
        return self._at("cpv", T)

    def cp_liquid(self, T: float) -> float:
        return self._at("cpl", T)

    def rho_liquid(self, T: float) -> float:
        return self._at("rhol", T)

    def dh_vap(self, T: float) -> float:
        return self._at("dh", T)

    def dh_vap_datum(self, T: float) -> float:
        return self._at("datum", T)

    def surface_tension(self, T: float) -> float:
        return self._at("sigma", T)


def tabulated(substance: Substance, fluid: str, **kw) -> TabulatedThermo:
    """`CoolPropThermo` for `fluid`, precomputed.  The way to run mixtures."""
    return TabulatedThermo(CoolPropThermo(substance, fluid=fluid), **kw)
