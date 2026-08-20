"""
Source models
=============

A source does three things and nothing else:

1. produces the initial cloud state handed to the integrator,
2. supplies the source flux term over the source region, and
3. declares which dispersion mode the run starts in.

Everything downstream — the conservation equations, entrainment,
thermodynamics — is identical for all four release types.  This is the one
place where they differ, which is why the reference implementation's habit
of testing ``idspl`` in five separate places (initialisation, the
integration loop, ``SLAB_entran``, ``SLAB_eval``, ``SLAB_thermo``) is worth
undoing: adding a release type there means finding and editing all five.

Initial cloud height
--------------------
At the first step the cloud has no height yet, so ``h`` cannot be recovered
from ``R = rho U B h``.  SLAB instead integrates the vertical entrainment
over the first step analytically.  With a logarithmic wind profile and the
neutral entrainment law this gives an implicit equation for the height
``he`` reached after one step of length ``dx`` (SLAB.FOR L502-520):

    (he + z0) F^2 - 2 (he + z0) F + 2 he = Z1,     F = ln(1 + he/z0)

    Z1 = sqrt(3) a k^3 U_rf dx / u_a*

Solved here by the same Newton iteration, but with a convergence test the
reference lacks (it runs a fixed five passes).

Source-area expansion
---------------------
For a strong, dense, low-wind release the momentum equation (EQ 4b) has no
positive real root inside the source region: the cloud cannot be carried
downwind through an area that small.  Physically the cloud spreads upwind
until the area is large enough.  SLAB models this by enlarging the source
half-width at constant mass rate and retrying, which is an outer loop over
the whole source-region integration.  `EvaporatingPool.expanded()` produces
the enlarged source; the driver owns the loop because only it knows whether
the integration succeeded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

from ..coefficients import PHYS, Coefficients, COEFFS
from ..core.trajectory import Mode
from ..submodels.atmosphere import Atmosphere
from ..thermo.base import Substance

__all__ = [
    "SourceFlux",
    "SourceState",
    "SourceModel",
    "EvaporatingPool",
    "HorizontalJet",
    "InstantaneousRelease",
    "initial_cloud_height",
]

SQRT3 = math.sqrt(3.0)


# ===========================================================================
@dataclass(frozen=True)
class SourceFlux:
    """Source terms added to the conservation equations at one location."""

    mass: float = 0.0        #: rho_s w_s B_s [kg/(m s)], EQ 1/2
    enthalpy: float = 0.0    #: rho_s w_s B_s cp_s T_s [J/(m s)], EQ 3
    w_source: float = 0.0
    """
    Vertical injection velocity entering the shear term of the in-cloud
    friction velocity, EQ 35e.

    This is the **geometric** velocity, not the one reduced by source-area
    expansion.  The original keeps the two apart explicitly (SLAB.FOR
    L770-771: ``wse = ws bs^2 / bse^2`` for the mass flux, ``wss = ws`` for
    the shear), and the distinction is defensible: enlarging the footprint is
    a device for representing upwind gravity flow, and it should not weaken
    the jet-like shear that the real pool surface produces.

    The two coincide until the source expands, so a run that needs no
    expansion cannot reveal a confusion between them.
    """

    @property
    def is_active(self) -> bool:
        return self.mass != 0.0


@dataclass(frozen=True)
class SourceState:
    """
    Initial condition handed to the integrator.

    ``x_start`` is negative for an area source, whose upwind edge lies at
    ``-B_s``; the source region is then ``[-B_s, +B_s]``.
    """

    x_start: float
    t_start: float
    mode: Mode

    h: float                  #: cloud height [m]
    z_c: float                #: centre-height parameter [m]
    b_half: float             #: half-width B [m]
    b_shape: float            #: shape parameter b [m]
    b_half_x: float           #: half-length Bx [m] (1.0 in plume mode)
    b_shape_x: float

    u: float                  #: cloud velocity [m/s]
    T: float                  #: temperature [K]
    rho: float                #: density [kg/m^3]
    cp: float                 #: specific heat [J/(kg K)]

    m_emission: float         #: mass fraction of released material
    m_ev: float               #: of which vapour
    m_water: float
    m_wv: float

    v_g: float = 0.0
    u_g: float = 0.0
    w_c: float = 0.0
    w_entrain: float = 0.0    #: initial vertical entrainment, EQ 35a
    v_entrain: float = 0.0
    u_ambient_mean: float = 0.0
    R_flux: float = 0.0       #: R = rho U B h; zero when h = 0

    @property
    def beta(self) -> float:
        """EQ 13: B^2 = b^2 + 3 beta^2."""
        return math.sqrt(max(self.b_half**2 - self.b_shape**2, 0.0)) / SQRT3


@runtime_checkable
class SourceModel(Protocol):
    """Contract every release type implements."""

    name: str
    substance: Substance
    duration: float
    """Release duration [s]; math.inf for a continuous release."""

    def initial_state(self, atm: Atmosphere, *, dx: float, **kw) -> SourceState:
        """
        Initial condition.  Accepts optional `emission` and `water` thermo
        backends; a source whose starting state is already at equilibrium
        (a pool, an undiluted jet) ignores them, but one that dilutes before
        dispersion begins — a vertical jet, which entrains through its rise —
        needs them to find the temperature and density it starts from.
        """

    def flux(self, x: float, t: float) -> SourceFlux: ...

    @property
    def half_width(self) -> float: ...



    @property
    def total_mass(self) -> float: ...

    def released_mass(self, t: float) -> float:
        """
        Material released by time `t` [kg].

        Continuous sources ramp up and then plateau; an instantaneous release
        delivers everything at t = 0.  The puff equations need this rather
        than ``rate * t``, which is zero for a source that has no rate.
        """


# ===========================================================================
def initial_cloud_height(
    atm: Atmosphere,
    dx: float,
    *,
    coeffs: Coefficients = COEFFS,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> float:
    """
    Cloud height after the first integration step, SLAB.FOR L502-520.

    Solves ``(h + z0) F^2 - 2 (h + z0) F + 2h = Z1`` with ``F = ln(1 + h/z0)``.
    The reference runs exactly five Newton passes with no convergence test;
    here the iteration is checked and the initial bracket is the reference's.
    """
    if dx <= 0:
        raise ValueError("dx must be > 0")
    z0 = atm.z0
    k = PHYS.VON_KARMAN
    z1 = (SQRT3 * coeffs.a_entrain * k**3
          * atm.dimensionless_profile(4.0) * dx / atm.u_star)

    z2 = (3.0 * z1 * z0 * z0) ** (1.0 / 3.0)
    h = 3.0 * z1 / math.log(1.0 + z2 / z0) ** 2

    for _ in range(max_iter):
        F = math.log1p(h / z0)
        s = h + z0
        f = s * F * F - 2.0 * s * F + 2.0 * h - z1
        step = f / (F * F)                       # reference's approximation
        h -= step
        if h <= 0.0:                             # guard the reference lacks
            h = 0.5 * (h + step)
        if abs(step) < tol * max(h, 1.0):
            return h
    raise RuntimeError(f"initial cloud height did not converge (h = {h})")


def _mean_wind(atm: Atmosphere, h: float, n: int = 5) -> float:
    """
    Depth-averaged ambient wind from the ground to `h`, EQ 4c.

    The upper limit is the cloud **top** `h_top`, not its thickness: for an
    elevated release the two differ, and averaging only over the thickness
    understates the wind the cloud actually sees by nearly 10 %.

    Composite Simpson with `n` panels, matching SLAB.FOR L709-770 at the
    default n = 5.  Kept as a free function because the source needs it
    before a cloud exists.
    """
    if h <= 0:
        return 0.0
    dh = h / n
    total = 0.0
    for i in range(n):
        a, m, b = i * dh, (i + 0.5) * dh, (i + 1) * dh
        total += (atm.dimensionless_profile(max(a, 1e-12))
                  + 4.0 * atm.dimensionless_profile(m)
                  + atm.dimensionless_profile(b))
    return atm.u_star / PHYS.VON_KARMAN * total / (6.0 * n)


# ===========================================================================
@dataclass(frozen=True)
class EvaporatingPool:
    """
    Ground-level evaporating pool, ``IDSPL = 1``.

    The released material is pure vapour at its boiling point, injected
    vertically over a square source of area ``area`` at a constant rate for
    ``duration`` seconds.  Droplets may still form later in the cloud, but
    not at the source.

    Parameters
    ----------
    substance : released material.
    rate : mass source rate [kg/s]                       (SLAB `QS`)
    area : source area [m^2]                             (SLAB `AS`)
    duration : release duration [s]; ``inf`` for continuous  (SLAB `TSD`)
    T_source : source temperature [K]; clamped to the boiling point.
    half_width_effective : enlarged half-width produced by
        `expanded()`; ``None`` means the geometric value ``0.5 sqrt(area)``.
    """

    substance: Substance
    rate: float
    area: float
    duration: float = math.inf
    T_source: float | None = None
    half_width_effective: float | None = None

    name: str = "evaporating_pool"

    def __post_init__(self):
        if self.rate <= 0:
            raise ValueError("rate must be > 0")
        if self.area <= 0:
            raise ValueError("area must be > 0")
        if self.duration <= 0:
            raise ValueError("duration must be > 0")

    # -- geometry -------------------------------------------------------
    @property
    def half_width_geometric(self) -> float:
        """B_s = 0.5 sqrt(A_s) (SLAB.FOR L424)."""
        return 0.5 * math.sqrt(self.area)

    @property
    def half_width(self) -> float:
        """Effective half-width, enlarged if gravity flow required it."""
        return self.half_width_effective or self.half_width_geometric

    @property
    def T(self) -> float:
        T = self.T_source if self.T_source is not None else self.substance.T_boil
        return self.substance.source_temperature(T)

    @property
    def rho_source(self) -> float:
        """Ideal-gas source density at the source temperature."""
        return self.substance.mw * PHYS.P_ATM / (PHYS.R_GAS * self.T)

    @property
    def w_source(self) -> float:
        """
        Vertical vapour velocity, ``W_s = q / (rho_s A_s)``.

        Held at the *geometric* area: enlarging the source spreads the same
        mass over more area, so the injection velocity falls (SLAB.FOR
        L770, ``wse = ws bs^2 / bse^2``).
        """
        return self.rate / (self.rho_source * self.area)

    @property
    def w_source_effective(self) -> float:
        r = self.half_width_geometric / self.half_width
        return self.w_source * r * r

    @property
    def total_mass(self) -> float:
        return math.inf if math.isinf(self.duration) else self.rate * self.duration

    def released_mass(self, t: float) -> float:
        return self.rate * min(max(t, 0.0), self.duration)

    def expanded(self, half_width: float) -> "EvaporatingPool":
        """Same release through a larger footprint, at constant mass rate."""
        if half_width < self.half_width_geometric:
            raise ValueError("effective half-width cannot shrink below geometric")
        return replace(self, half_width_effective=half_width)

    # -- source term ----------------------------------------------------
    def flux(self, x: float, t: float) -> SourceFlux:
        """
        Source flux, non-zero only inside ``-B_se <= x <= B_se`` while the
        release is active (EQ 1/2, ``rho_s w_s B_s``).
        """
        B = self.half_width
        if not (-B <= x <= B) or t > self.duration:
            return SourceFlux()
        m = self.rho_source * self.w_source_effective * B
        return SourceFlux(
            mass=m,
            enthalpy=m * self.substance.cp_vapour * self.T,
            w_source=self.w_source,          # geometric, see SourceFlux
        )

    # -- initial condition ----------------------------------------------
    def initial_state(
        self, atm: Atmosphere, *, dx: float, coeffs: Coefficients = COEFFS,
        **kw,
    ) -> SourceState:
        """
        State at the upwind edge of the pool, ``x = -B_se``.

        The cloud starts as pure ambient air: zero height, zero concentration,
        ambient density and temperature.  Material enters through the source
        flux as the integration crosses the pool.  Only the *width* is
        prescribed, at the pool footprint (SLAB.FOR L555-590).
        """
        B = self.half_width
        he = initial_cloud_height(atm, dx, coeffs=coeffs)
        u_bar = _mean_wind(atm, he)

        return SourceState(
            x_start=-B,
            t_start=0.0,
            mode=Mode.PLUME,
            h=0.0,
            z_c=0.0,
            b_half=B,
            b_shape=0.9 * B,
            b_half_x=1.0,
            b_shape_x=1.0,
            u=u_bar,
            T=atm.T,
            rho=atm.rho,
            cp=atm.cp_moist,
            m_emission=0.0,
            m_ev=0.0,
            m_water=atm.mass_frac_water,
            m_wv=atm.mass_frac_water,
            w_entrain=u_bar * he / (SQRT3 * dx),
            v_entrain=_horizontal_entrainment_seed(atm, coeffs),
            u_ambient_mean=u_bar,
            R_flux=0.0,
        )


def _horizontal_entrainment_seed(atm: Atmosphere, coeffs: Coefficients) -> float:
    """Initial cross-wind entrainment, ``a1 S(La) Ua`` (SLAB.FOR L770)."""
    r_cf = math.sqrt((atm.u_star / atm.wind_speed(2.0)) / coeffs.c_10)
    inv_La = atm.inv_L_ground
    s = (1.0 - r_cf * coeffs.L_a_ref * inv_La if inv_La < 0.0
         else 1.0 / (1.0 + r_cf * coeffs.L_a_ref * inv_La))
    f_theta = (coeffs.tau_min / coeffs.t_ref_avg) ** coeffs.p_meander
    return 0.08 * s * f_theta * atm.u_ref


# ===========================================================================
@dataclass(frozen=True)
class _PressurisedRelease:
    """
    Shared behaviour of the two pressurised source types.

    Unlike a pool, a pressurised release enters the domain as *pure material*
    rather than as ambient air that material is added to: the initial mass
    fraction is 1, not 0.  It may be two-phase, in which case the droplet
    fraction is an input (`liquid_fraction`) and the temperature is pinned to
    the boiling point.  The initial density follows from EQ 10 applied to the
    undiluted mixture,

        rho_s = rho_a T_a / (alpha T_s + gamma T_a)
        alpha = (M_ae / M_s)(1 - m_l),   gamma = (rho_a / rho_sl) m_l

    which is heavier than the vapour alone whenever droplets are present —
    the reason a released ammonia cloud is dense despite MW 17 < 29.
    """

    substance: Substance
    rate: float
    area: float
    duration: float
    liquid_fraction: float = 0.0
    T_source: float | None = None
    height: float = 0.0

    def __post_init__(self):
        if self.rate < 0:
            raise ValueError("rate must be >= 0")
        if self.area <= 0:
            raise ValueError("area must be > 0")
        if not 0.0 <= self.liquid_fraction < 1.0:
            raise ValueError("liquid_fraction must be in [0, 1)")

    @property
    def T(self) -> float:
        T = self.T_source if self.T_source is not None else self.substance.T_boil
        return self.substance.source_temperature(T, self.liquid_fraction)

    @property
    def half_width_geometric(self) -> float:
        return 0.5 * math.sqrt(self.area)

    @property
    def vapour_fraction(self) -> float:
        return 1.0 - self.liquid_fraction

    def exit_velocity(self, atm: Atmosphere | None = None) -> float:
        """
        Velocity at the orifice [m/s].

        Sets the droplet size through the break-up criterion, so it is the
        velocity *at formation* rather than the local slip further downwind —
        droplets do not re-form once the jet has slowed.
        """
        rho = self.rho_source(atm) if atm is not None else self.substance.mw \
            * PHYS.P_ATM / (PHYS.R_GAS * self.T)
        return self.rate / (rho * self.area) if self.area > 0 else 0.0

    def rho_source(self, atm: Atmosphere) -> float:
        """EQ 10 for the undiluted two-phase mixture."""
        alpha = (atm.mw_moist / self.substance.mw) * self.vapour_fraction
        gamma = (atm.rho / self.substance.rho_liquid) * self.liquid_fraction
        return atm.rho * atm.T / (alpha * self.T + gamma * atm.T)

    def cp_source(self) -> float:
        return (self.vapour_fraction * self.substance.cp_vapour
                + self.liquid_fraction * self.substance.cp_liquid)


@dataclass(frozen=True)
class HorizontalJet(_PressurisedRelease):
    """
    Elevated horizontal jet, ``IDSPL = 2``.

    The jet is treated as an area source facing downwind, centred at
    ``x = 1 m`` and height `height` (SLAB.FOR L594-660).  There is no
    vertical injection, so ``W_s = 0`` and the source contributes no mass
    flux to the conservation equations: all of the material is present from
    the first step, and dilution happens purely by entrainment.

    The initial geometry depends on whether the jet clears the ground.  An
    elevated jet keeps its own square cross-section (``h = 2 B_s``, gravity
    spreading off, ``alpha_g = 0``); one whose lower edge touches the ground
    is flattened into a grounded slab of the same area with gravity spreading
    switched back on.
    """

    name: str = "horizontal_jet"

    @property
    def half_width(self) -> float:
        return 1.0                          # EQ 29b table: B_xs = 0, X_s = 1 m

    @property
    def total_mass(self) -> float:
        return math.inf if math.isinf(self.duration) else self.rate * self.duration

    def released_mass(self, t: float) -> float:
        return self.rate * min(max(t, 0.0), self.duration)

    def expanded(self, half_width: float) -> "HorizontalJet":
        return self                          # EQ 4d cannot fail: alpha_g = 0

    def flux(self, x: float, t: float) -> SourceFlux:
        return SourceFlux()                  # W_s = 0 everywhere

    def initial_state(self, atm: Atmosphere, *, dx: float,
                      coeffs: Coefficients = COEFFS, **kw) -> SourceState:
        rho = self.rho_source(atm)
        u = self.rate / (rho * self.area) if self.area > 0 else 0.0
        B = self.half_width_geometric
        h = 2.0 * B
        z_c = self.height

        if self.height <= 0.5 * h:           # jet touches down at the source
            B = 0.5 * (math.sqrt(self.height**2 + 2.0 * self.area)
                       - self.height)
            h = B + self.height
        if h > atm.h_mix:                    # clipped by the mixing layer
            B, h = h * B / atm.h_mix, atm.h_mix
        z_c = min(z_c, atm.h_mix - 0.5 * h)
        h_top = z_c + 0.5 * h if z_c > 0.5 * h else h

        return SourceState(
            x_start=1.0, t_start=0.0, mode=Mode.PLUME,
            h=h, z_c=z_c, b_half=B, b_shape=0.9 * B,
            b_half_x=1.0, b_shape_x=1.0,
            u=u, T=self.T, rho=rho, cp=self.cp_source(),
            m_emission=1.0, m_ev=self.vapour_fraction,
            m_water=0.0, m_wv=0.0,
            # No entrainment seed: a jet already has a cloud, so the rates
            # follow from the entrainment submodel evaluated at this state.
            # The pool's `w = U h_e / (sqrt3 dx)` is a device for growing a
            # cloud out of nothing in the first step and does not apply here
            # — using it makes the jet entrain several times its own mass
            # over the first half metre and destroys its momentum.
            u_ambient_mean=_mean_wind(atm, h_top),
            R_flux=rho * u * B * h,
        )


@dataclass(frozen=True)
class InstantaneousRelease(_PressurisedRelease):
    """
    Instantaneous or short-duration ground-level release, ``IDSPL = 4``.

    The whole inventory `mass` is present at ``t = 0`` in a slab of the given
    area, so the run is in puff mode from the first step (EQ 15-26) and never
    needs the plume equations.  `rate` and `duration` are retained only so
    that the short-duration variant, in which SLAB switches an evaporating
    pool to this branch once the centre of mass has passed, shares the type.
    """

    mass: float = 0.0
    name: str = "instantaneous"

    def __post_init__(self):
        super().__post_init__()
        if self.mass <= 0:
            raise ValueError("mass must be > 0")

    @property
    def half_width(self) -> float:
        return self.half_width_geometric

    @property
    def total_mass(self) -> float:
        return self.mass

    def released_mass(self, t: float) -> float:
        return self.mass                     # all of it, at t = 0

    def expanded(self, half_width: float) -> "InstantaneousRelease":
        return self

    def flux(self, x: float, t: float) -> SourceFlux:
        return SourceFlux()

    def initial_state(self, atm: Atmosphere, *, dx: float,
                      coeffs: Coefficients = COEFFS, **kw) -> SourceState:
        rho = self.rho_source(atm)
        B = self.half_width_geometric
        # Slab of the given footprint holding the whole inventory.
        #
        # The height is deliberately *not* clipped to the mixing layer.  The
        # reference limits only the centre height (SLAB.FOR L1157,
        # ``zcmx = hmx - h/2``) and lets h stand; clipping h without widening
        # B to compensate would silently discard mass.  A deck that produces
        # a cloud taller than the boundary layer — a large inventory in a
        # tiny footprint — is unphysical at the input, and is better left
        # visibly so than quietly truncated.
        h = self.mass / (rho * 4.0 * B * B)
        return SourceState(
            x_start=0.0, t_start=0.0, mode=Mode.PUFF,
            h=h, z_c=self.height, b_half=B, b_shape=0.9 * B,
            b_half_x=B, b_shape_x=0.9999 * B,
            u=0.0, T=self.T, rho=rho, cp=self.cp_source(),
            m_emission=1.0, m_ev=self.vapour_fraction,
            m_water=0.0, m_wv=0.0,
            u_ambient_mean=_mean_wind(
                atm, min(self.height + 0.5 * h if self.height > 0.5 * h else h,
                         atm.h_mix)),
            R_flux=rho * B * B * h,
        )
