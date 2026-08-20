"""
Steady-state plume mode
=======================

Solves EQ 1-10 for a continuous or finite-duration release.

Why this is not `solve_ivp(rhs, ...)`
-------------------------------------
SLAB does not integrate the conservation equations directly.  Section 2.4
rearranges them so that only eleven *flux-like* quantities are integrated,

    R    = rho U B h                       EQ 2a
    Bv, bv                                 EQ 7, 8   (width growth)
    G, Sfy                                 EQ 5ab    (gravity flow)
    Sft, Sfu, Sfz, Gw                      EQ 3a, 4a (accumulated fluxes)
    z_c                                    EQ 9
    Qint                                   EQ 29b    (mass in cloud)

while the physical state — concentration, temperature, density, velocity and
height — is recovered *algebraically* at every stage:

    m    = q (x + B_s) / (4 B_s R)         EQ 1a     analytic
    T    from the energy integral          EQ 41     Newton, with the phase split
    rho  from the equation of state        EQ 10
    U    from the cubic                    EQ 4b/4e  closed form
    h    = R / (rho U B)                   EQ 2b     algebraic

Ermak's stated reason is accuracy: integrating fluxes and reconstructing the
state keeps the conserved quantities conserved by construction.  The cost is
that the right-hand side is not a function of the state alone, so a generic
adaptive integrator cannot be dropped in.  The fixed-step RK4 of the
reference is kept here; replacing it is deliberately a separate step (`S1` in
the ablation plan) so that the change can be measured on its own.

The reconstruction is itself iterative
--------------------------------------
`reconstruct` (SLAB_eval, SLAB.FOR L2446 (subroutine eval)) solves a small fixed-point
problem: `U` needs `h` and the depth-averaged wind, `h` needs `U`, and the
averaged wind needs the cloud top.  The reference damps the update
geometrically, ``U <- (U U_new^2)^(1/3)``, and stops when the relative change
falls below 1e-3.  Both are reproduced; the damping matters, as plain
substitution oscillates for dense low-wind releases.

Source-region gravity flow (EQ 4d)
----------------------------------
The cubic has a positive real root only while ``U_g^3 < (4/27) U_e^3``.  When
it does not, the release cannot be carried downwind through a footprint that
small; physically the cloud spreads upwind until it can.  `reconstruct`
reports this through `needs_expansion`, and `integrate_plume` raises
`SourceExpansionRequired` so the driver can enlarge the source and retry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from ..coefficients import PHYS, Coefficients, COEFFS
from ..submodels.atmosphere import Atmosphere
from ..submodels.entrainment import CloudLocal, entrainment, fluxes, friction
from ..thermo.base import ThermoBackend
from ..thermo.equilibrium import Mixture, solve_equilibrium
import numpy as np

from .source import SourceModel
from .trajectory import ARRAY_FIELDS, Mode, Trajectory

__all__ = [
    "run_dispersion",
    "gravity_coefficient",
    "PlumeState",
    "SourceExpansionRequired",
    "reconstruct",
    "slopes",
    "integrate_plume",
]

SQRT3 = math.sqrt(3.0)
_TWO_27 = 2.0 / 27.0


def gravity_coefficient(lofted: bool, rho: float, rho_ambient: float,
                        coeffs: Coefficients = COEFFS) -> float:
    """
    Gravity-flow pressure coefficient alpha_g for EQ 4, from the cloud state.

    This is **not** a property of the source.  The reference recomputes it at
    the end of every step (SLAB.FOR L2446-2637 (eval)):

        lofted            -> 0      no ground to spread along
        grounded, denser  -> 0.25   gravity current
        grounded, lighter -> 0      nothing to drive it

    Two things hang off it: the gravity term of the velocity cubic (EQ 4b,
    hence whether EQ 4d can fail at all) and the ``h >= 2 z_c`` clamp, which
    the reference gates on ``alfg*rho > alfg*rhoa`` precisely so that it
    switches off together with alpha_g.  Treating it as a fixed source
    attribute makes an elevated release that later settles behave as if it
    had never touched down.
    """
    if lofted or rho <= rho_ambient:
        return 0.0
    return coeffs.alpha_g


class SourceExpansionRequired(RuntimeError):
    """EQ 4d has no positive real root: the source footprint is too small."""

    def __init__(self, x: float, suggested_half_width: float):
        super().__init__(
            f"no real velocity solution at x = {x:.3f} m; "
            f"source half-width must grow to about {suggested_half_width:.3f} m"
        )
        self.x = x
        self.suggested_half_width = suggested_half_width


# ===========================================================================
# integrated state
# ===========================================================================
@dataclass
class PlumeState:
    """The eleven integrated quantities plus the reconstructed cloud state."""

    # integrated
    R: float          #: rho U B h [kg/(m s)]
    Bv: float         #: half-width accumulator (EQ 7)
    bv: float         #: shape-parameter accumulator (EQ 8)
    z_c: float        #: centre height [m] (EQ 9)
    Qint: float       #: released mass in the cloud [kg] (EQ 29b)

    # reconstructed
    h: float
    b_half: float
    b_shape: float
    u: float
    T: float
    rho: float
    cp: float
    m: float          #: mass fraction of released material
    m_ev: float
    m_water: float
    m_wv: float
    v_g: float = 0.0
    w_c: float = 0.0
    u_bar: float = 0.0    #: depth-averaged ambient wind (EQ 4c)
    h_top: float = 0.0

    # entrainment / fluxes at this point
    w_e: float = 0.0
    v_e: float = 0.0
    v_ex: float = 0.0
    v_front: float = 0.0
    """Gravity-front entrainment; dilutes but does not widen (not in Ermak)."""
    f_u: float = 0.0
    f_v: float = 0.0
    f_w: float = 0.0
    f_t: float = 0.0
    u_star: float = 0.0
    phi_h: float = 0.0
    inv_L: float = 0.0
    evap_rate: float = 0.0
    """-d(d^2)/dt for the droplets [m^2/s]; zero without finite-rate evaporation."""
    rainout_flux: float = 0.0
    """Liquid leaving the cloud per unit down-wind distance [kg/(m s)/m]."""
    pool: float = 0.0
    """Cumulative mass rained out [kg/(m s)]; zero without a droplet model."""
    d2: float = 0.0
    """Droplet area [m^2]; zero unless finite-rate evaporation is active."""
    hv: float = 0.0
    """
    Cloud height integrated from the vertical entrainment.

    Always computed, used only when `height_closure` is "constrained".  With
    the default closure the height is algebraic (EQ 2b) and this is carried
    along unused, which keeps the two code paths identical everywhere else.
    """

    @property
    def beta(self) -> float:
        return math.sqrt(max(self.b_half**2 - self.b_shape**2, 0.0)) / SQRT3

    @property
    def is_lofted(self) -> bool:
        return self.h_top > self.h

    def vol_frac(self, mw_emission: float, mw_ambient: float) -> float:
        """EQ 12: volume from mass concentration."""
        return mw_ambient * self.m / (mw_emission + (mw_ambient - mw_emission) * self.m)


# ===========================================================================
# slopes, EQ 1-10 as rearranged in section 2.4
# ===========================================================================
def slopes(
    st: PlumeState,
    atm: Atmosphere,
    src: SourceModel,
    x: float,
    *,
    coeffs: Coefficients = COEFFS,
) -> list[float]:
    """
    The eleven derivatives, mirroring `SLAB_slope` (SLAB.FOR L1897 (subroutine slope)).

    Index map (kept from the reference so the two can be compared directly):

        0  R      1  Bv     2  bv     3  G      4  Sft    5  Sfu
        6  Gw     7  z_c    8  Qint   9  Sfy   10  Sfz   11  hv   12  pool
       13  d2 (droplet area, finite-rate evaporation)

    Slope 11 is not Ermak's.  It integrates the cloud height from the
    vertical entrainment so that `height_closure` can swap which of height
    and width is the dependent variable; with the default closure it is
    computed but unused, which costs nothing and keeps the two paths
    identical everywhere else.
    """
    c = coeffs
    f = [0.0] * 14
    B, h = st.b_half, st.h
    d_rho = st.rho - atm.rho

    src_flux = src.flux(x, 0.0)

    # EQ 2a, with the gravity-front entrainment of `v_front` added to the
    # dilution only — see `entrainment()` for why it must not enter EQ 7.
    f[0] = (atm.rho * SQRT3 * ((st.v_e + st.v_front) * h + st.w_e * B)
            + src_flux.mass)
    f[1] = (SQRT3 * (atm.rho / st.rho) * st.v_e + st.v_g) / st.u
    f[2] = (st.v_g * st.b_shape) / (st.u * B) if B > 0 else 0.0
    f[3] = c.alpha_gv * PHYS.GRAVITY * d_rho * h * h
    f[4] = st.f_t
    f[5] = st.f_u
    f[6] = -PHYS.GRAVITY * d_rho * B * h
    f[7] = st.w_c / st.u
    f[8] = 2.0 * st.rho * B * h * st.m
    f[9] = st.f_v
    f[10] = st.f_w
    f[11] = SQRT3 * st.w_e / st.u if st.u > 0 else 0.0
    # Rainout: liquid lost to the ground per unit distance.  Not Ermak's;
    # zero unless a droplet model is supplied, so the default path is
    # untouched.  The mass leaves the cloud, so it leaves EQ 2a as well.
    # The liquid leaves the cloud and arrives in the pool: R loses it,
    # `pool` gains it, and the two must sum to the released mass.
    f[12] = st.rainout_flux
    f[0] -= st.rainout_flux
    # Droplet area shrinks by the d-squared law; zero unless finite-rate
    # evaporation is switched on, so the default path is Ermak's.
    f[13] = -st.evap_rate / st.u if st.u > 0 else 0.0
    return f


# ===========================================================================
# algebraic reconstruction, EQ 4b-4e + 2b  (SLAB_eval)
# ===========================================================================
@dataclass
class _Reconstruction:
    u: float
    h: float
    b_half: float
    b_shape: float
    u_bar: float
    h_top: float
    v_g: float
    w_c: float
    needs_expansion: bool
    expansion_ratio: float
    iterations: int
    d_b_half: float = 0.0


def _solve_velocity_cubic(
    u_e: float, u_g3: float, u_prev: float, coeffs: Coefficients
) -> tuple[float, bool, float]:
    """
    Largest positive root of ``U^3 - U_e U^2 + U_g^3 = 0`` (EQ 4b, 4e).

    Returns ``(U, needs_expansion, expansion_ratio)``.  When EQ 4d is
    violated the returned velocity is the degenerate value ``2 U_e / 3`` and
    `expansion_ratio` is the factor by which the source area must grow,
    ``(27/4) U_g^3 / U_e^3`` (SLAB.FOR L2600-2637).
    """
    if u_g3 <= 0.0:                      # neutrally or positively buoyant
        return u_e, False, 1.0

    u_m3 = _TWO_27 * u_e**3              # (2/27) U_e^3; EQ 4d limit is twice this
    if u_g3 - u_m3 > u_m3:               # no positive real root
        ratio = (27.0 / 4.0) * u_g3 / u_e**3 if u_e > 0 else 1.0
        return min(2.0 * u_e / 3.0, u_prev), True, ratio

    phi = math.acos(max(-1.0, min(1.0, -(u_g3 - u_m3) / u_m3)))
    return (u_e + 2.0 * u_e * math.cos(phi / 3.0)) / 3.0, False, 1.0


def reconstruct(
    *,
    R: float,
    rho: float,
    m: float,
    z_c: float,
    atm: Atmosphere,
    prev: PlumeState,
    Bv: float,
    bv: float,
    Bv0: float,
    bv0: float,
    G: float,
    Gw: float,
    Sfu: float,
    Sfy: float,
    Sfz: float,
    R0: float,
    Sru0: float,
    inside_source: bool,
    alpha_g: float,
    hv: float = 0.0,
    height_closure: str = "algebraic",
    added_mass: bool = False,
    apply_widening: bool = True,
    coeffs: Coefficients = COEFFS,
) -> _Reconstruction:
    """
    Recover U, h, B, b, the depth-averaged wind and the gravity velocities.

    This is `SLAB_eval`.  The coupled fixed point is

        B  <- (B0 + Bv - Bv0) * rex(rho0/rho, h, B)
        h  <- R / (rho U B)
        U  <- cubic(U_e(u_bar), U_g(rho, B, R))
        u_bar <- Simpson over [h_top0, h_top]

    with `rex` the density-ratio stretch that keeps the cloud cross-section
    consistent as it warms or cools.
    """
    c = coeffs
    B0, b0, rho0, u0 = prev.b_half, prev.b_shape, prev.rho, prev.u
    htp0, u_bar0, h0 = prev.h_top, prev.u_bar, prev.h

    # Which of height and width is the dependent variable.  Mass conservation
    # fixes their product, ``B h = R/(rho U)``, so the choice cannot change
    # the dilution — only the partition, and through it the centreline
    # concentration, which carries B and the shape parameter b separately.
    integrate_h = height_closure == "constrained"

    u1 = atm.dimensionless_profile(max(htp0, 1e-9))
    u = u0
    if integrate_h:
        h = max(hv, 1e-9)
        B = R / (rho * u * h) if rho * u * h > 0 else B0
    else:
        B = B0 + Bv - Bv0
        h = R / (rho * u * B) if rho * u * B > 0 else 0.0
    rex = (rho0 / rho) ** (h / (h + 2.0 * B)) if (h + 2.0 * B) > 0 else 1.0

    needs_expansion, ratio, htp, u_bar = False, 1.0, prev.h_top, u_bar0
    n = 0
    for n in range(1, c.eval_max_iter + 1):
        gex = h / (h + 2.0 * B) if (h + 2.0 * B) > 0 else 0.0
        rex0 = (rho0 * u0 / (rho * u)) if u0 > u_bar0 else (rho0 / rho)
        rex = math.sqrt(rex * rex0**gex)

        b = (b0 + bv - bv0) * rex
        if integrate_h:
            h = max(hv, 1e-9)
            B = R / (rho * u * h)
        else:
            B = (B0 + Bv - Bv0) * rex
            h = R / (rho * u * B)

            if h > 2.0 * B:                   # cloud taller than wide
                B = math.sqrt(R / (2.0 * rho * u))
                h = 2.0 * B
                rex = math.sqrt(rex0)

        htp = z_c + 0.5 * h if z_c > 0.5 * h else h
        u_bar = _mean_ambient(atm, htp0, htp, u_bar0, u1)

        # EQ 4b coefficients
        sfue = (0.0 if Sfu == 0.0 else
                -R0 * (u0 - u_bar0) * abs(Sfu)
                / (abs(Sfu) + R0 * abs(u0 - u_bar0)))
        u_e = u_bar * (1.0 - m) + (Sru0 + sfue) / R
        u_g3 = (alpha_g * PHYS.GRAVITY * (rho - atm.rho) * R
                / (2.0 * B * rho * rho))

        u_new, needs_expansion, ratio = _solve_velocity_cubic(u_e, u_g3, u0, c)
        u_new = (u * u_new * u_new) ** (1.0 / 3.0)     # geometric damping

        if u <= 0.0 or u_new <= 0.0:
            raise RuntimeError(
                f"non-positive cloud velocity in the reconstruction "
                f"(U = {u:.3e} -> {u_new:.3e}); check the source momentum"
            )
        converged = abs(u - u_new) / math.sqrt(u * u_new) < c.eval_tol
        u = u_new
        if integrate_h:
            B = R / (rho * u * h)
        else:
            h = R / (rho * u * B)
        if converged:
            break

    # A grounded dense cloud cannot be thinner than twice its centre height.
    # The reference writes the density test as `alfg*rho > alfg*rhoa`
    # (SLAB_eval, SLAB.FOR L2638 (subroutine evalpf)), which looks like a no-op but is not:
    # with alpha_g = 0 — an elevated jet, where gravity spreading is switched
    # off — it is false, so the clamp is disabled.  That is the intent: z_c is
    # the release height for a jet, not a centre-of-mass constraint on its
    # own thickness.  Dropping the factor pins h at 2 z_c from the first step
    # and destroys the jet's momentum.
    if h < 2.0 * z_c and alpha_g * rho > alpha_g * atm.rho and not integrate_h:
        h = 2.0 * z_c
        u = R / (rho * B * h)
    h = min(h, atm.h_mix)
    z_c = min(z_c, atm.h_mix - 0.5 * h)
    htp = z_c + 0.5 * h if z_c > 0.5 * h else h
    u_bar = _mean_ambient(atm, htp0, htp, u_bar0, u1)

    dB = 0.0
    if needs_expansion and apply_widening:
        # EQ 4d has no root: widen the cloud in place by the factor that
        # restores one, and hand the increment back so that the width
        # accumulator stays consistent (SLAB_eval L2633, SLAB.FOR L930).
        B, b_new = B * ratio, b * ratio
        dB = B * (ratio - 1.0) / ratio
        b = b_new

    # -- gravity-flow velocities, EQ 5a / 6a ---------------------------
    #
    # The cross-wind gravity velocity is recovered from an accumulated
    # momentum integral, ``V_g = e_v (G + R0 V_g0) / R``, damped by the
    # friction integral through ``e_v = R0 V_g0 / (R0 V_g0 + |Sfy|)``.
    #
    # Measured on the Burro trials, ``|Sfy| / (R0 V_g0)`` is of order 1e-4,
    # so ``e_v`` sits at 0.999 and the damping does essentially nothing: the
    # gravity current keeps spreading long after the buoyancy driving it has
    # gone.  For BU09 the density excess falls by a factor 211 between 11 m
    # and 921 m while V_g falls only by 3.2, and gravity still supplies more
    # than half the cloud width at the far end.
    #
    # The friction is weak by construction.  ``f_v = -rho B (V_mg^2 + V_mh^2)``
    # with ``V_mg = c_f V_g/2`` and ``c_f = u_a*/<ua> ~ 0.05``, giving an
    # effective drag coefficient of about 0.006 on V_g.
    #
    # This is not a porting error — it is what EQ 5a says — but it is the
    # single largest contributor to the over-wide, over-shallow clouds seen
    # against the Burro measurements.  See `examples/diagnose_burro.py`.
    lofted = htp > h
    v_g, w_c = 0.0, 0.0
    if inside_source:
        if rho > atm.rho:
            ev = 1.0 if prev.v_g == 0.0 else R0 * prev.v_g / (R0 * prev.v_g + abs(Sfy))
            v_g = ev * (G + R0 * prev.v_g) / R
    elif lofted or rho <= atm.rho:
        ew = (1.0 if prev.w_c == 0.0 else
              R0 * abs(prev.w_c) / (R0 * abs(prev.w_c) + abs(Sfz)))
        w_c = ew * (Gw + R0 * prev.w_c) / R
        if added_mass:
            # A rising cloud must accelerate the air it displaces as well as
            # itself.  EQ 6 carries only the cloud's own mass, which for the
            # flat pancake a dense release becomes understates the inertia by
            # an order of magnitude.  See `submodels/added_mass`.
            from ..submodels.added_mass import added_mass_factor
            w_c /= 1.0 + added_mass_factor(B, h, rho, atm.rho,
                                           coefficient=c.c_added_mass)
    else:
        vg0 = prev.v_g
        if vg0 == 0.0:
            ev = 1.0
            vg0 = -2.0 * prev.w_c * B0 / h0 if h0 > 0 else 0.0
        else:
            ev = R0 * vg0 / (R0 * vg0 + abs(Sfy))
        v_g = ev * (G + R0 * vg0) / R
        w_c = -v_g * z_c / B if B > 0 else 0.0

    return _Reconstruction(
        u=u, h=h, b_half=B, b_shape=b, u_bar=u_bar, h_top=htp,
        v_g=v_g, w_c=w_c, needs_expansion=needs_expansion,
        expansion_ratio=ratio, iterations=n, d_b_half=dB,
    )


def _mean_ambient(
    atm: Atmosphere, htp0: float, htp: float, u_bar0: float, u1: float
) -> float:
    """
    Running depth average of the ambient wind, EQ 4c.

    Extends the previous average from `htp0` to `htp` with a three-point
    Simpson rule rather than re-integrating from the ground, which is what
    SLAB_eval does and keeps the average continuous across steps.
    """
    if htp <= 0.0:
        return 0.0
    k = PHYS.VON_KARMAN
    u12 = atm.dimensionless_profile(max(0.5 * (htp0 + htp), 1e-9))
    u2 = atm.dimensionless_profile(max(htp, 1e-9))
    u_bar = (htp0 * u_bar0 + (atm.u_star / k)
             * (u1 + 4.0 * u12 + u2) * (htp - htp0) / 6.0) / htp
    return min(u_bar, atm.u_star / k * u2)


# ===========================================================================
# driver
# ===========================================================================
def integrate_plume(
    src: SourceModel,
    atm: Atmosphere,
    emission: ThermoBackend,
    water: ThermoBackend,
    *,
    x_max: float,
    # `n_source_steps` is deliberately separate from `n_field_steps`, and it
    # is the accuracy bottleneck.  Refining the field grid alone converges to
    # about 0.03 % by 80 steps; the source grid converges only at *first*
    # order and the default of 10 sits about 2.8 % below the resolved answer.
    #
    # The reason is structural.  Inside the source region EQ 1a replaces the
    # integration of the species equation with the algebraic ``m = s_q / R``,
    # where ``s_q`` ramps linearly across the pool.  The only integrated
    # quantity there is R, and its source term has a slope discontinuity at
    # the pool edge, which costs the Runge-Kutta scheme its order.
    #
    # Raising it is cheap — the source region is a few metres — and
    # ``n_source_steps = n_field_steps // 4`` reaches about 0.3 % by 640.
    # The default stays at Ermak's so that reproduction of the reference is
    # undisturbed; the cost of that choice is measured in
    # `tests/test_convergence.py`.
    n_source_steps: int = 10,
    n_field_steps: int = 40,
    substeps: int = 3,
    coeffs: Coefficients = COEFFS,
    height_closure: str = "algebraic",
    rainout: bool = False,
    kinetic_evaporation: bool = False,
    substrate=None,
    added_mass: bool = False,
) -> Trajectory:
    """
    Integrate the steady-state plume equations from the source to `x_max`.

    The grid is uniform across the source region and logarithmic beyond it,
    matching the reference's two-stage layout.  `substeps` is the reference's
    `NSSM`; unlike `NCALC` it controls only accuracy, since the output grid is
    separate.

    Raises
    ------
    SourceExpansionRequired
        If EQ 4d fails inside the source region.  The caller should call
        ``src.expanded(...)`` and retry; see `run_plume` for a driver that
        does this automatically.
    """
    B_s = src.half_width
    x0 = src.initial_state(atm, dx=1.0, coeffs=coeffs,
                           emission=emission, water=water).x_start
    has_source_region = src.flux(0.5 * (x0 + B_s), 0.0).is_active
    dx_source = ((B_s - x0) / (n_source_steps * substeps)
                 if has_source_region else None)
    st, rows = _initial_state(src, atm, dx_source or 1.0, coeffs,
                              emission, water)

    # Droplets are formed at the orifice and carry their size from there, so
    # the reference area is set once rather than recomputed downwind.
    d2_ref, m_el_source = 0.0, 0.0
    if kinetic_evaporation and st.m > st.m_ev:
        from ..submodels.rainout import droplet_diameter

        fn = getattr(src, "exit_velocity", None)
        u_jet = fn(atm) if callable(fn) else st.u
        d0 = droplet_diameter(emission.surface_tension(st.T), atm.rho,
                              max(u_jet, 1e-3), weber=coeffs.weber_critical)
        d2_ref, m_el_source = d0 * d0, st.m - st.m_ev
        st.d2 = d2_ref
    x = x0

    grid = _build_grid(x0, B_s, x_max, n_source_steps, n_field_steps,
                       has_source_region)
    # Momentum accumulator of EQ 4b.  A pool starts with R = 0 so this is
    # zero, but a jet arrives with its own momentum and starting from zero
    # would make U_e vanish at the first step.
    alpha_g0 = gravity_coefficient(st.h_top > st.h, st.rho, atm.rho, coeffs)
    srug0 = (0.5 * alpha_g0 * PHYS.GRAVITY * (st.rho - atm.rho)
             * st.b_half * st.h * st.h)
    sru0 = st.R * st.u - st.R * (1.0 - st.m) * st.u_bar + srug0
    prev_flux = _StageMemory(st, R0=st.R, Sru0=sru0, alpha_g=alpha_g0)
    total_widening = 0.0

    for x_target in grid:
        n_sub = substeps
        dx = (x_target - x) / n_sub
        for _ in range(n_sub):
            st, prev_flux, dB = _rk4_step(
                st, prev_flux, x, dx, atm, src, emission, water, coeffs,
                height_closure, rainout, kinetic_evaporation,
                d2_ref, m_el_source, substrate, added_mass,
            )
            # Only widening inside the source region drives the expansion
            # search.  SLAB accumulates `dbt` in the source-region loop
            # alone (SLAB.FOR L933); the near-field loop keeps its own
            # copy that never feeds back.  Counting downwind widening as
            # well makes the search overshoot and stall.
            if x < B_s:
                total_widening += dB
            x += dx
        rows.append(_row(st, x, atm, src, emission))

    return Trajectory.from_rows(
        rows,
        meta={
            # Where the source region ends, so the expansion search can find
            # the reference station the original uses (`msfm`, SLAB.FOR
            # line 347: 11 of the 11 source-region stations for an area
            # source).  Index 0 is the upwind edge at x = -B.
            "source_station": (n_source_steps if has_source_region else 0),
            "source": src.name,
            "substance": emission.substance.name,
            "coefficients": coeffs.name,
            "thermo": emission.name,
            "mode": "plume",
            "x_max": x_max,
            "substeps": substeps,
            "legacy_unstable_profile": atm.legacy_unstable_profile,
            "legacy_stability_bug": atm.legacy_stability_bug,
            "height_closure": height_closure,
            "rainout": rainout,
            "kinetic_evaporation": kinetic_evaporation,
            "substrate": getattr(substrate, "name", None),
            "added_mass": added_mass,
            "source_widening": total_widening,
            "effective_half_width": B_s,
        },
    )


@dataclass
class _StageMemory:
    """Values from the end of the previous step that the stages need."""

    prev: PlumeState
    R0: float
    Sru0: float
    alpha_g: float = 0.0
    """Gravity-flow coefficient carried from the end of the previous step."""
    Bv0: float = 0.0
    bv0: float = 0.0
    z_c0: float = 0.0
    Qint0: float = 0.0


def _build_grid(x0: float, B_s: float, x_max: float, n_source: int,
                n_field: int, has_source_region: bool) -> list[float]:
    """
    Uniform through the source region, logarithmic beyond.

    A jet has no source region to traverse — its material is all present at
    the first step — so the grid is logarithmic from the release point.
    """
    grid = []
    start = x0
    # `B_s > x0` is what "there is a source region to traverse" means.  A
    # vertical jet starts at the top of its rise, which can be several metres
    # downwind, while its orifice half-width is centimetres; the difference
    # is then negative and the uniform leg runs backwards, so the trajectory
    # rejects it with "x must be non-decreasing".  Found by fuzzing against
    # the Fortran, on a two-phase jet through a 0.0029 m2 orifice.
    if has_source_region and B_s > x0:
        grid = [x0 + (B_s - x0) * (i + 1) / n_source for i in range(n_source)]
        start = B_s
    if x_max <= start:
        # Nothing left to integrate.  Returning `x_max` here would place a
        # row behind the start whenever the release begins downwind of it —
        # a vertical jet whose rise carries it past the requested domain,
        # for instance — and the trajectory would then be rejected for going
        # backwards.  The last point is the start itself.
        return grid or [start]
    lo, hi = math.log(max(start, 1e-6)), math.log(x_max)
    return grid + [
        math.exp(lo + (hi - lo) * (i + 1) / n_field) for i in range(n_field)
    ]


def _initial_state(src, atm, dx, coeffs, emission=None, water=None):
    s0 = src.initial_state(atm, dx=dx, coeffs=coeffs,
                           emission=emission, water=water)
    st = PlumeState(
        R=s0.R_flux, Bv=s0.b_half, bv=s0.b_shape, z_c=s0.z_c, Qint=0.0,
        h=s0.h, b_half=s0.b_half, b_shape=s0.b_shape, u=s0.u,
        T=s0.T, rho=s0.rho, cp=s0.cp,
        # A pool starts as ambient air (m = 0) and gains material through the
        # source flux; a pressurised release starts as pure material (m = 1),
        # possibly part liquid.  Getting this wrong poisons the enthalpy carry
        # term of EQ 41 and sends the cloud hotter than ambient.
        m=s0.m_emission, m_ev=s0.m_ev,
        m_water=s0.m_water, m_wv=s0.m_wv,
        v_g=s0.v_g, w_c=s0.w_c, u_bar=s0.u_ambient_mean,
        # EQ 13.  An elevated release has h_top = z_c + h/2 > h and is
        # therefore *lofted*; setting h_top = h unconditionally classifies a
        # jet as grounded from the first step, which switches on both the
        # gravity current and the h >= 2 z_c clamp and collapses it.
        h_top=(s0.z_c + 0.5 * s0.h if s0.z_c > 0.5 * s0.h else s0.h),
        w_e=s0.w_entrain, v_e=s0.v_entrain,
        hv=s0.h if s0.h > 0 else 1e-9,
    )
    if st.w_e == 0.0 and st.v_e == 0.0:
        # Source supplied no seed: evaluate the closure at the initial state.
        cl = CloudLocal(
            h=st.h, b_half=st.b_half, b_half_x=1.0, z_c=st.z_c, u=st.u,
            T=st.T, rho=st.rho, cp=st.cp, v_g=st.v_g, w_c=st.w_c,
            w_source=src.flux(s0.x_start, 0.0).w_source,
        )
        u_bar = max(st.u_bar, 1e-6)
        fr = friction(cl, atm, u_bar_ambient=u_bar, coeffs=coeffs)
        en = entrainment(cl, atm, fr, u_bar_ambient=u_bar, coeffs=coeffs)
        fx = fluxes(cl, atm, fr, coeffs=coeffs)
        st.w_e, st.v_e, st.v_ex = en.w, en.v, en.v_x
        st.v_front = en.v_front
        st.f_u, st.f_v, st.f_w, st.f_t = fx.f_u, fx.f_v, fx.f_w, fx.f_t
        st.u_star, st.phi_h, st.inv_L = fr.u_star, fr.phi_h, fr.inv_L_cloud

    rows = [_row(st, s0.x_start, atm, src, None)]
    return st, rows


def _row(st: PlumeState, x: float, atm, src, emission) -> dict:
    mw_e = emission.substance.mw if emission is not None else src.substance.mw
    cv = st.vol_frac(mw_e, atm.mw_moist)
    sigma = (0.5 * st.h / SQRT3 if st.z_c > 0.5 * st.h
             else max(st.h - st.z_c, 0.0) / SQRT3)
    # EQ 29b: the travel time of the centre of mass is the released mass
    # already in the cloud divided by the source rate, not x/U.  Qint
    # integrates 2 rho B h m, hence the factor 2.
    t = 2.0 * st.Qint / src.rate if src.rate > 0 else 0.0
    return dict(
        x=x, t=t, mode=int(Mode.PLUME),
        h=st.h, z_c=st.z_c, b_half=st.b_half, b_shape=st.b_shape,
        beta=st.beta, b_half_x=1.0, b_shape_x=1.0, beta_x=0.0, sigma_z=sigma,
        u=st.u, T=st.T, rho=st.rho, cp=st.cp,
        mass_frac=st.m, vol_frac=cv,
        v_g=st.v_g, u_g=0.0, w_c=st.w_c, u_ambient_mean=st.u_bar,
        mass_frac_dry_air=(1.0 - st.m) * (1.0 - atm.mass_frac_water),
        mass_frac_water=st.m_water, mass_frac_water_vapour=st.m_wv,
        mass_frac_emission_vapour=st.m_ev,
        mass_frac_water_liquid=st.m_water - st.m_wv,
        mass_frac_emission_liquid=st.m - st.m_ev,
        w_entrain=st.w_e, v_entrain=st.v_e, v_x_entrain=st.v_ex,
        u_star_cloud=st.u_star, phi_h=st.phi_h, inv_L_cloud=st.inv_L,
        f_u=st.f_u, f_v=st.f_v, f_w=st.f_w, f_t=st.f_t,
        mass_in_cloud=st.Qint, R_flux=st.R, rained_out=st.pool,
    )


def _run_puff_only(src, atm, emission, water, *, x_max, t_max,
                   n_puff_steps, coeffs):
    """Instantaneous release: puff mode from the first step (EQ 15-26)."""
    from .puff import PuffState, integrate_puff, row_puff

    s0 = src.initial_state(atm, dx=1.0, coeffs=coeffs)
    st = PuffState(
        R=s0.R_flux, x_c=s0.x_start, Bv=s0.b_half, bv=s0.b_shape,
        Bvx=s0.b_half_x, bvx=s0.b_shape_x, z_c=s0.z_c, t=0.0,
        h=s0.h, b_half=s0.b_half, b_shape=s0.b_shape,
        b_half_x=s0.b_half_x, b_shape_x=s0.b_shape_x,
        u=s0.u, T=s0.T, rho=s0.rho, cp=s0.cp,
        m=s0.m_emission, m_ev=s0.m_ev, m_water=s0.m_water, m_wv=s0.m_wv,
        u_bar=s0.u_ambient_mean, h_top=s0.h,
        w_e=s0.w_entrain, v_e=s0.v_entrain,
        mass_in_cloud=src.total_mass,
    )
    if t_max is None:
        t_max = 3.0 * x_max / max(atm.wind_speed(max(st.h, 1.0)), 0.1)
    rows = [row_puff(st, atm, emission)]
    rows += integrate_puff(st, atm, src, emission, water, t_max=t_max,
                           n_steps=n_puff_steps, coeffs=coeffs)
    return Trajectory.from_rows(rows, meta={
        "source": src.name, "substance": emission.substance.name,
        "coefficients": coeffs.name, "thermo": emission.name,
        "mode": "puff", "x_max": x_max,
        "effective_half_width": src.half_width,
        "legacy_unstable_profile": atm.legacy_unstable_profile,
        "legacy_stability_bug": atm.legacy_stability_bug,
    })


# ===========================================================================
# one RK4 step
# ===========================================================================
_RK_WEIGHTS = (1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0)
_RK_OFFSETS = (0.5, 0.5, 1.0)          # stage offsets for the trial states


def _rk4_step(
    st: PlumeState,
    mem: _StageMemory,
    x: float,
    dx: float,
    atm: Atmosphere,
    src: SourceModel,
    emission: ThermoBackend,
    water: ThermoBackend,
    coeffs: Coefficients,
    height_closure: str = "algebraic",
    rainout: bool = False,
    kinetic: bool = False,
    d2_ref: float = 0.0,
    m_el_source: float = 0.0,
    substrate=None,
    added_mass: bool = False,
) -> tuple[PlumeState, _StageMemory]:
    """
    Advance one step, mirroring SLAB.FOR L763-940.

    Classical RK4 on the eleven integrated quantities.  At every stage the
    algebraic state is rebuilt from them, so the "slope" of stage k is
    evaluated at a fully consistent thermodynamic and dynamic state — which
    is what makes this scheme more accurate than integrating the primitive
    variables, and also what stops a generic integrator being usable.
    """
    prev = st
    R0, Bv0, bv0, zc0, Q0 = st.R, st.Bv, st.bv, st.z_c, st.Qint
    hv0, pool0, d2_0 = st.hv, st.pool, st.d2
    Sru0, alpha_g = mem.Sru0, mem.alpha_g

    total = [0.0] * 14
    stage = st
    widened = 0.0

    for k in range(4):
        x_stage = x + (dx * _RK_OFFSETS[k] if k < 3 else dx)
        f = slopes(stage, atm, src, x_stage if k < 3 else x, coeffs=coeffs)

        for j in range(14):
            total[j] += _RK_WEIGHTS[k] * dx * f[j]
        dy = (total if k == 3
              else [dx * _RK_OFFSETS[k] * f[j] for j in range(14)])

        stage, dB = _build_stage(
            dy=dy, x=x_stage, R0=R0, Bv0=Bv0, bv0=bv0, zc0=zc0, Q0=Q0,
            hv0=hv0, pool0=pool0, d2_0=d2_0, Sru0=Sru0, prev=prev,
            atm=atm, src=src,
            emission=emission, water=water, coeffs=coeffs,
            alpha_g=alpha_g, height_closure=height_closure,
            rainout=rainout, kinetic=kinetic, d2_ref=d2_ref,
            m_el_source=m_el_source, substrate=substrate,
            added_mass=added_mass, final_stage=(k == 3),
        )
        total[1] += dB
        widened += dB

    alpha_g_next = gravity_coefficient(stage.is_lofted, stage.rho, atm.rho,
                                       coeffs)
    srug = 0.5 * alpha_g_next * PHYS.GRAVITY * (stage.rho - atm.rho) \
        * stage.b_half * stage.h * stage.h
    Sru = stage.R * stage.u - stage.R * (1.0 - stage.m) * stage.u_bar + srug
    return (stage,
            _StageMemory(prev=stage, R0=stage.R, Sru0=Sru,
                         alpha_g=alpha_g_next),
            widened)


def _build_stage(
    *, dy, x, R0, Bv0, bv0, zc0, Q0, hv0, pool0, d2_0, Sru0, prev, atm, src,
    emission, water, coeffs, alpha_g, height_closure="algebraic",
    rainout=False, kinetic=False, d2_ref=0.0, m_el_source=0.0,
    substrate=None, added_mass=False, final_stage=False,
) -> tuple[PlumeState, float]:
    """Rebuild the full physical state from the integrated increments."""
    # --- SLAB_solve: apply the increments -----------------------------
    R = R0 + dy[0]
    Bv, bv = Bv0 + dy[1], bv0 + dy[2]
    G, Sft, Sfu, Gw = dy[3], dy[4], dy[5], dy[6]
    z_c = min(max(zc0 + dy[7], 0.0), atm.h_mix - 0.5 * prev.h)
    Qint = Q0 + dy[8]
    Sfy, Sfz = dy[9], dy[10]
    hv = hv0 + dy[11]
    pool = pool0 + dy[12]
    d2 = max(d2_0 + dy[13], 0.0)

    if R <= 0.0:
        raise RuntimeError(f"non-positive mass flux R = {R:.3e} at x = {x:.3f}")
    pool = max(pool, 0.0)

    B_s = src.half_width
    inside = x <= B_s

    # --- EQ 1a: concentration, analytic -------------------------------
    q = src.rate
    s_q = 0.5 * q if x > B_s else 0.25 * q * (B_s + x) / B_s
    # EQ 1a assumes every gram released is still airborne.  With rainout it
    # is not: the pooled mass has to come off the numerator as well as out of
    # R, or the model reports material it has already put on the ground.
    airborne = max(s_q - pool, 0.0)
    m = min(max(airborne / R, 0.0), 1.0) if q > 0 else 0.0

    # --- EQ 40: species -----------------------------------------------
    m_wa = atm.mass_frac_water
    m_water = (1.0 - m) * m_wa
    m_dry = (1.0 - m) * (1.0 - m_wa)
    ratio = R0 / R
    m_wv_t = m_water + ratio * (prev.m_wv - prev.m_water)
    # Transported vapour, EQ 40d.  The term ``prev.m_ev - prev.m`` is minus
    # the previous liquid load, so with rainout it has to be reduced by the
    # liquid that has since reached the ground — otherwise the equilibrium
    # solver is handed droplets that are no longer in the cloud, absorbs
    # their latent heat again, and returns a colder and denser cloud.
    m_el_prev = max(prev.m - prev.m_ev, 0.0)
    if pool > pool0 and R0 > 0.0:
        m_el_prev = max(m_el_prev - (pool - pool0) / R0, 0.0)
    m_ev_t = m - ratio * m_el_prev

    # --- EQ 41: enthalpy, with the damped ground-heat term ------------
    T_s = src.T
    cp_s = emission.substance.cp_vapour
    dT0 = prev.T - atm.T
    sfte = (0.0 if Sft == 0.0 else
            -prev.cp * dT0 * abs(Sft) / (abs(Sft) + R0 * prev.cp * abs(dT0)))
    E = ((1.0 - m) * atm.cp_moist * atm.T + m * cp_s * T_s
         + ratio * (prev.cp * prev.T
                    - (1.0 - prev.m) * atm.cp_moist * atm.T
                    - prev.m * cp_s * T_s + sfte))

    # Finite-rate evaporation: the droplets carry their own history, so the
    # phase split follows the d-squared law rather than local equilibrium.
    # The constraint is one-sided — evaporation is rate-limited, condensation
    # is not — so the liquid load may exceed equilibrium but never fall below
    # it, and the equilibrium answer is recovered as soon as the kinetics
    # stop binding.
    m_ev_kin = None
    if kinetic and d2_ref > 0.0 and d2 > 0.0:
        surviving = (d2 / d2_ref) ** 1.5           # mass goes as d^3
        m_el_kin = m_el_source * surviving * (airborne / max(s_q, 1e-30))
        m_ev_kin = max(m - m_el_kin, 0.0)

    eq = solve_equilibrium(
        Mixture(m_emission=m, m_water=m_water, m_dry_air=m_dry,
                m_ev_transported=m_ev_t, m_wv_transported=m_wv_t, enthalpy=E,
                m_ev_prescribed=m_ev_kin),
        emission, water,
        T_ambient=atm.T, rho_ambient=atm.rho,
        mw_ambient_moist=atm.mw_moist,
        T_guess=prev.T, coeffs=coeffs,
    )

    # --- EQ 4b/4e + 2b: velocity, height, width -----------------------
    rec = reconstruct(
        R=R, rho=eq.rho, m=m, z_c=z_c, atm=atm, prev=prev,
        Bv=Bv, bv=bv, Bv0=Bv0, bv0=bv0,
        G=G, Gw=Gw, Sfu=Sfu, Sfy=Sfy, Sfz=Sfz,
        R0=R0, Sru0=Sru0, inside_source=inside, alpha_g=alpha_g,
        hv=hv, height_closure=height_closure, added_mass=added_mass,
        apply_widening=final_stage, coeffs=coeffs,
    )
    st = PlumeState(
        R=R, Bv=Bv, bv=bv, z_c=z_c, Qint=Qint,
        h=rec.h, b_half=rec.b_half, b_shape=rec.b_shape, u=rec.u,
        T=eq.T, rho=eq.rho, cp=eq.cp, m=m, m_ev=eq.m_ev,
        m_water=m_water, m_wv=eq.m_wv,
        v_g=rec.v_g, w_c=rec.w_c, u_bar=rec.u_bar, h_top=rec.h_top,
        hv=hv,
    )

    # --- entrainment and fluxes for the next slope --------------------
    cl = CloudLocal(
        h=st.h, b_half=st.b_half, b_half_x=1.0, z_c=st.z_c, u=st.u,
        T=st.T, rho=st.rho, cp=st.cp, v_g=st.v_g, w_c=st.w_c,
        w_source=src.flux(x, 0.0).w_source,
    )
    # Ground cooling: the surface under a cold cloud is not held at ambient,
    # as EQ 39 assumes.  The drop depends on elapsed time, which `friction`
    # cannot see, so it is evaluated here from the flux the previous step
    # actually drew.
    cooling = 0.0
    if substrate is not None and prev.b_half > 0.0 and src.rate > 0.0:
        from ..submodels.ground import surface_cooling
        cooling = surface_cooling(abs(prev.f_t) / prev.b_half,
                                  2.0 * Qint / src.rate, substrate)

    fr = friction(cl, atm, u_bar_ambient=max(st.u_bar, 1e-6),
                  ground_cooling=cooling, coeffs=coeffs)
    en = entrainment(cl, atm, fr, u_bar_ambient=max(st.u_bar, 1e-6), coeffs=coeffs)
    fx = fluxes(cl, atm, fr, coeffs=coeffs)

    st.w_e, st.v_e, st.v_ex = en.w, en.v, en.v_x
    st.v_front = en.v_front
    st.f_u, st.f_v, st.f_w, st.f_t = fx.f_u, fx.f_v, fx.f_w, fx.f_t
    st.u_star, st.phi_h, st.inv_L = fr.u_star, fr.phi_h, fr.inv_L_cloud
    st.pool = pool
    st.d2 = d2
    st.rainout_flux = _rainout_flux(st, atm, src, emission, rainout, coeffs)
    st.evap_rate = _evap_rate(st, atm, emission, kinetic, coeffs)
    return st, rec.d_b_half


def _evap_rate(st: PlumeState, atm: Atmosphere, emission: ThermoBackend,
               enabled: bool, coeffs: Coefficients) -> float:
    """``-d(d^2)/dt`` for the droplets, zero unless finite-rate is enabled."""
    if not enabled or st.d2 <= 0.0 or st.m <= st.m_ev:
        return 0.0
    from ..submodels.rainout import evaporation_rate, terminal_velocity

    d = math.sqrt(st.d2)
    drop = terminal_velocity(d, emission.rho_liquid(st.T), st.rho)
    # Surface mass fraction: saturated at the droplet's temperature, which
    # for a flashing release sits at its boiling point.  Bulk: whatever the
    # cloud already carries as vapour.
    mw_s = emission.substance.mw
    x_s = min(emission.saturation_ratio(st.T), 1.0)
    y_s = x_s * mw_s / (x_s * mw_s + (1.0 - x_s) * PHYS.MW_AIR)
    return evaporation_rate(d, y_s, st.m_ev, st.rho,
                            emission.rho_liquid(st.T),
                            reynolds=drop.reynolds)


def _rainout_flux(st: PlumeState, atm: Atmosphere, src: SourceModel,
                  emission: ThermoBackend, enabled: bool,
                  coeffs: Coefficients = COEFFS) -> float:
    """
    Liquid mass leaving the cloud per unit down-wind distance.

    Zero unless rainout is switched on, so the default path is Ermak's.  The
    droplet diameter is set once at the release, by the exit velocity that
    formed the droplets, rather than by the local slip — the droplets do not
    re-form downwind.
    """
    m_el = st.m - st.m_ev                  # released material as droplets
    if not enabled or m_el <= 0.0 or st.h <= 0.0 or st.u <= 0.0:
        return 0.0
    from ..submodels.rainout import (
        droplet_diameter, rainout_rate, terminal_velocity,
    )

    fn = getattr(src, "exit_velocity", None)
    u_jet = fn(atm) if callable(fn) else st.u
    sigma = emission.surface_tension(st.T)
    d = droplet_diameter(sigma, atm.rho, max(u_jet, 1e-3),
                         weber=coeffs.weber_critical)
    drop = terminal_velocity(d, emission.rho_liquid(st.T), st.rho)
    rate = coeffs.rainout_efficiency * rainout_rate(
        m_el, drop.terminal_velocity, st.u, st.h)
    # Cap the fractional loss per metre.  The rate is a first-order decay
    # constant, and near a slow, shallow source it can exceed 1/m, at which
    # point a Runge-Kutta stage can remove more liquid than the cloud holds
    # and drive R negative.  The cap bounds the step, not the physics: the
    # liquid still leaves, just not faster than the cloud can supply it.
    return min(rate, 1.0) * m_el * st.R


# ===========================================================================
def run_plume(
    src: SourceModel,
    atm: Atmosphere,
    emission: ThermoBackend,
    water: ThermoBackend,
    *,
    x_max: float,
    max_expansions: int = 30,
    rtol: float = 5e-4,
    coeffs: Coefficients = COEFFS,
    **kw,
) -> tuple[Trajectory, SourceModel]:
    """
    `integrate_plume` with the source-area expansion search of section 2.4.

    Two mechanisms act together, as in the reference:

    * **in place** — whenever EQ 4d has no root at the final Runge-Kutta
      stage the cloud is widened by the factor that restores one and the
      increment is added to the width integral.  The step is not abandoned.
    * **outer fixed point** — the source half-width is advanced to the
      *geometric* mean of its current value and the value that would absorb
      all of the in-place widening,

          B_new = sqrt(B (B + sum dB))                  SLAB.FOR L981

      and the integration repeated.  The geometric mean is what makes this
      converge: the widening is distributed over many steps, so the
      arithmetic sum badly overshoots.

    Converged when the update moves the width by less than `rtol`.  For the
    manual's example 1 this reproduces the reference's 31.148 m from a
    geometric footprint of 12.816 m in about six passes.

    Two earlier attempts are recorded here because both converge to the wrong
    footprint: bisecting on "does the integration succeed unaided" lands
    about 17 % high, and using the arithmetic sum ``B + sum dB`` lands more
    than twice as high.

    A known gap, and what it is not
    -------------------------------
    Only the widening branch is implemented.  The original has a second one:
    when a pass produces *no* in-place widening, that brackets the answer
    from above (``bmax = bse``) and it estimates the width from the
    gravity-spreading balance instead of stopping (SLAB.FOR lines 987-999)::

        bsen = 3.375 alfg g (rho - rho_a) R / (U_a^3 rho^2 (1 - m)^3)
               - bb(msfm) + bb(1)
        bsen = 0.5 (bse + bsen)

    then bisects within ``[bmin, bmax]``.  ``msfm`` is the last of the eleven
    source-region stations for an area source (SLAB.FOR line 347), which is
    the row at ``x = +B``; the plume-rise-dependent form at line 1152 applies
    only to a vertical jet.

    **This branch was implemented faithfully and it does not fix the dense
    release failure.**  The estimate cannot fire there, because by its own
    reference station the cloud has already been destroyed: on the chlorine
    deck below the density excess at ``x = +B`` is 0.0002 kg/m3, the cloud
    being 99.97 % air, so the balance has no positive root.

    Tracing the rows shows why.  The blow-up is not a slow runaway that an
    outer search could catch — it happens in **one step**::

        i        x        bb         h        m
        0  -0.3171     0.317     0.000  0.00000
        1  -0.2536    33.059    66.091  0.00003     <- one Runge-Kutta step

    So the cause is upstream of this search: a first-step instability in the
    source region when the geometric footprint is tiny next to the release
    rate (10.4 kg/s of chlorine through 0.402 m2).  The original reaches
    1.270 m and a cloud 1.4 m tall; here the ``h = 2B`` clamp returns 66 m.

    Recorded rather than half-implemented — the branch is real and missing,
    but adding it changes nothing here and costs three regression tests that
    assert the search drives the widening to zero.  See
    `tests/test_fuzz_fortran.py::test_dense_low_momentum_jet`.
    """
    trial = src
    traj = integrate_plume(trial, atm, emission, water, x_max=x_max,
                           coeffs=coeffs, **kw)
    for _ in range(max_expansions):
        widen = traj.meta["source_widening"]
        if widen <= 0.0:
            break
        B = trial.half_width
        B_new = math.sqrt(B * (B + widen))
        if abs(B_new - B) <= rtol * B_new:
            break
        trial = src.expanded(B_new)
        traj = integrate_plume(trial, atm, emission, water, x_max=x_max,
                               coeffs=coeffs, **kw)
    return traj, trial


# ===========================================================================
# combined plume + puff run
# ===========================================================================
def run_dispersion(
    src: SourceModel,
    atm: Atmosphere,
    emission: ThermoBackend,
    water: ThermoBackend,
    *,
    x_max: float,
    t_max: float | None = None,
    n_puff_steps: int = 40,
    coeffs: Coefficients = COEFFS,
    **kw,
) -> tuple[Trajectory, SourceModel]:
    """
    Full simulation: steady-state plume, then transient puff (section 2.3.3).

    The plume integration stops as soon as EQ 29a is satisfied — the released
    material inside the cloud reaches half the total, meaning the centre of
    mass has passed — and the puff takes over from that state.  A continuous
    release never satisfies EQ 29a and stays a plume all the way to `x_max`.
    """
    from .puff import PuffState, integrate_puff, transition_state

    # An instantaneous release is a puff from t = 0: there is no steady state
    # to establish, so the plume equations are never used (section 2.3.2).
    if src.initial_state(atm, dx=1.0, coeffs=coeffs).mode is Mode.PUFF:
        return _run_puff_only(src, atm, emission, water, x_max=x_max,
                              t_max=t_max, n_puff_steps=n_puff_steps,
                              coeffs=coeffs), src

    # A vertical jet spends its first metre or two rising; SLAB interpolates
    # that region (EQ 47) rather than integrating it, then starts the plume
    # from the top of the rise.
    rise_rows: list[dict] = []
    t_rise = 0.0
    if hasattr(src, "rise_profile"):
        full = src.rise_profile(atm, coeffs=coeffs,
                                emission=emission, water=water)
        rise_rows = full[:-1]
        t_rise = full[-1]["t"]

    traj, used = run_plume(src, atm, emission, water, x_max=x_max,
                           coeffs=coeffs, **kw)
    if rise_rows:
        # The plume restarts its clock at the top of the rise, so the rise
        # duration has to be carried over or the trajectory's time goes
        # backwards at the join.
        rows = []
        for i in range(len(traj)):
            r = {n: float(getattr(traj, n)[i]) for n in ARRAY_FIELDS}
            r["t"] += t_rise
            r["mode"] = int(traj.mode[i])
            rows.append(r)
        traj = Trajectory.from_rows(rise_rows + rows, meta=traj.meta)
    if math.isinf(used.duration):
        return traj, used

    # EQ 29a: the switch is where the released mass inside the cloud reaches
    # half the total, which by EQ 29b is exactly t = t_sd.  Snapping that to
    # the nearest output row would put the transition anywhere from t_sd to
    # t_sd + 3 s depending on where the logarithmic grid happens to fall, so
    # the crossing is interpolated instead.
    half_mass = 0.5 * used.total_mass
    Q = traj.mass_in_cloud
    above = np.flatnonzero(Q >= half_mass)
    if above.size == 0 or above[0] == 0:
        return traj, used                       # never reached the transition
    hi = int(above[0])
    lo = hi - 1
    span = Q[hi] - Q[lo]
    f = float((half_mass - Q[lo]) / span) if span > 0 else 0.0
    f = min(max(f, 0.0), 1.0)

    def blend(name):
        a, b = getattr(traj, name)[lo], getattr(traj, name)[hi]
        return float(a + f * (b - a))

    rows = [
        {name: float(getattr(traj, name)[i]) for name in ARRAY_FIELDS}
        | {"mode": int(traj.mode[i])}
        for i in range(lo + 1)
    ]
    cut = {name: blend(name) for name in ARRAY_FIELDS}
    cut["mass_in_cloud"] = half_mass
    # The shape parameters are not independent (EQ 13: B^2 = b^2 + 3 beta^2),
    # so blending all three separately would leave the interpolated row
    # inconsistent.  Recover the dependent ones instead.
    cut["beta"] = math.sqrt(
        max(cut["b_half"] ** 2 - cut["b_shape"] ** 2, 0.0)) / SQRT3
    cut["beta_x"] = math.sqrt(
        max(cut["b_half_x"] ** 2 - cut["b_shape_x"] ** 2, 0.0)) / SQRT3
    cut["sigma_z"] = (0.5 * cut["h"] / SQRT3 if cut["z_c"] > 0.5 * cut["h"]
                      else max(cut["h"] - cut["z_c"], 0.0) / SQRT3)
    # == t_sd by EQ 29b, but in the plume's own clock.  For a vertical jet
    # the rows before this one already carry the rise duration, so it has to
    # be added here too or the trajectory's time steps backwards at exactly
    # this row.  Found by fuzzing against the Fortran: four of ten sampled
    # vertical jets raised "t must be non-decreasing", while the cases with
    # no rise, or with t_sd far larger than the rise, happened to be safe.
    cut["t"] = 2.0 * half_mass / used.rate + t_rise
    cut["mode"] = int(Mode.PLUME)
    rows.append(cut)

    st = PlumeState(
        R=cut["R_flux"], Bv=cut["b_half"], bv=cut["b_shape"],
        z_c=cut["z_c"], Qint=half_mass,
        h=cut["h"], b_half=cut["b_half"], b_shape=cut["b_shape"],
        u=cut["u"], T=cut["T"], rho=cut["rho"], cp=cut["cp"],
        m=cut["mass_frac"], m_ev=cut["mass_frac_emission_vapour"],
        m_water=cut["mass_frac_water"], m_wv=cut["mass_frac_water_vapour"],
        v_g=cut["v_g"], w_c=cut["w_c"],
        u_bar=cut["u_ambient_mean"], h_top=cut["h"],
        w_e=cut["w_entrain"], v_e=cut["v_entrain"], v_ex=cut["v_x_entrain"],
        f_u=cut["f_u"], f_v=cut["f_v"], f_w=cut["f_w"], f_t=cut["f_t"],
    )
    puff0 = transition_state(st, atm, used, cut["t"])
    puff0.x_c = cut["x"]

    if t_max is None:
        # Generous by design: `x_max` bounds the *plume* grid, and the puff
        # is normally followed well beyond it.  The scale is the cloud's own
        # velocity, so a slow release integrates for a long time — pass
        # `t_max` explicitly when that is not wanted.
        t_max = cut["t"] + 4.0 * x_max / max(st.u, 0.1)
    rows += integrate_puff(
        puff0, atm, used, emission, water,
        t_max=t_max, n_steps=n_puff_steps, coeffs=coeffs,
    )

    meta = dict(traj.meta)
    meta.update(mode="plume+puff", transition_x=cut["x"],
                transition_t=cut["t"],
                effective_half_width=used.half_width)
    return Trajectory.from_rows(rows, meta=meta), used
