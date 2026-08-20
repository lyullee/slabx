"""
Entrainment and flux submodel
=============================

The turbulent-mixing closure of SLAB: the rate at which ambient air is drawn
into the cloud through its top and its sides, and the momentum and heat
fluxes exchanged with the ground.

Equation numbers refer to UCRL-MA-105607 section 2.5.2-2.5.3; ``TNO`` to
CPR 14E chapter 4.  Where the scan is illegible the reference implementation
``SLAB_entran`` (SLAB.FOR L2764-2940) was used and is cited by line.

Reduced entrainment velocities
------------------------------
``w``, ``v`` and ``vx`` returned here are **reduced** velocities: the factor
sqrt(3) that appears in EQ 35a/36a/36b is *not* applied here.  It is applied
where the entrainment enters the conservation equations, e.g.

    SLAB_slope L2811:  vhwb = sqrt(3) * (v*h + w*bb)

Keeping the convention of the reference avoids a double factor.  Multiply by
``math.sqrt(3)`` to obtain W_e, V_e, V_ex as printed in the manual.

Structure of the in-cloud friction velocity (EQ 35e)
----------------------------------------------------
    U*^2 = U_mg*^2 + U_mh*^2 + U_t^2

    U_mg*^2  ground friction        : (u*/<ua>)^2 (U^2 + V_g^2/4) + U_ss^2
    U_mh*^2  top-of-cloud drag      : Cf (dU^2 + (rho_a/rho)^2 V_g^2/4)
    U_t^3    thermal convection     : Ct g (Tg - T) V_H h / Tmean

with dU = (rho_a/rho)(<ua> - U) the density-weighted slip velocity.  Only
U_mg* carries the source-jet term U_ss^2 and only the grounded branch has a
thermal contribution.

Sign of the stratification term (EQ 35d)
-----------------------------------------
The scan reads ``L^-1 = [L_a^-1 U_a*^2 - C_mu g (rho-rho_a)/rho] / U*^2``.
The reference implementation adds the term, and physics requires it: a cloud
denser than air is stably stratified, which must *increase* 1/L and so damp
mixing.  We follow the implementation and clamp the term at zero (a lighter-
than-air cloud is not treated as unstable).

Known deviations of the reference implementation
------------------------------------------------
0.02 for that coefficient where the manual, TNO 4.132-4.133 and TNO 4.127 all imply
0.0195.  See ``Coefficients.c_drag_top``.  Select with
``coefficients.preset("tno")``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..coefficients import PHYS, Coefficients, COEFFS
from .atmosphere import Atmosphere

__all__ = ["CloudLocal", "Friction", "Entrainment", "Fluxes",
           "friction", "entrainment", "fluxes"]

SQRT3 = math.sqrt(3.0)


# ===========================================================================
# input state
# ===========================================================================
@dataclass(frozen=True)
class CloudLocal:
    """
    The subset of the cloud state that the mixing closure needs.

    Kept deliberately small and independent of the (not yet frozen)
    Trajectory schema so that this module can be tested on its own.
    """

    h: float            # cloud height [m]
    b_half: float       # cloud half-width B [m]  (JS `bb`)
    b_half_x: float     # cloud half-length Bx [m]; = 1.0 in plume mode
    z_c: float          # cloud centre-height parameter [m]
    u: float            # cloud down-wind velocity [m/s]
    T: float            # cloud temperature [K]
    rho: float          # cloud density [kg/m^3]
    cp: float           # cloud specific heat [J/(kg K)]
    v_g: float = 0.0    # cross-wind gravity-flow velocity [m/s] (JS `vg`)
    u_g: float = 0.0    # down-wind gravity-flow velocity [m/s], puff only
    w_c: float = 0.0    # centre-height rate [m/s] (JS `wc`)
    w_source: float = 0.0   # vertical source velocity W_s [m/s]
    is_puff: bool = False

    @property
    def h_top(self) -> float:
        """Height of the cloud top, EQ 13 (JS `htp`)."""
        return self.z_c + 0.5 * self.h if self.z_c > 0.5 * self.h else self.h

    @property
    def is_lofted(self) -> bool:
        """True when the cloud has lifted clear of the ground."""
        return self.h_top > self.h


# ===========================================================================
# results
# ===========================================================================
@dataclass(frozen=True)
class Friction:
    """In-cloud friction velocity and its components, EQ 35e."""

    u_star: float        # U*   [m/s]
    u_mg_star: float     # U_mg* [m/s], ground friction (JS sqrt(ubs2))
    u_mh_star_sq: float  # U_mh*^2 [m^2/s^2], top drag (JS uhs2)
    u_t_sq: float        # U_t^2 [m^2/s^2], thermal (JS uts2)
    v_H: float           # heat/mass transfer velocity V_H [m/s] (JS vh)
    delta_u: float       # density-weighted slip dU [m/s] (JS delu)
    c_f: float           # u_a*/<ua> (JS cf)
    inv_L_cloud: float   # in-cloud 1/L [1/m], EQ 35d
    phi_h: float         # stability function at cloud top, EQ 35c
    T_ground: float      # effective ground temperature [K]


@dataclass(frozen=True)
class Entrainment:
    """Reduced entrainment velocities (multiply by sqrt(3) for W_e, V_e)."""

    w: float    # vertical, EQ 35a
    v: float    # horizontal cross-wind, EQ 36a
    v_x: float  # horizontal down-wind, EQ 36b
    v_ambient: float   # ambient-turbulence part of v (JS va)
    v_shear: float     # shear part of v (JS vjp * sqrt(Cf))
    v_front: float     # gravity-front part of v (not in Ermak)
    a_total: float     # a1 = 0.08 S(La) F_theta (JS atot)


@dataclass(frozen=True)
class Fluxes:
    """Momentum and heat fluxes, EQ 37-39."""

    f_u: float    # down-wind momentum [N/m]
    f_ug: float   # down-wind gravity-flow momentum (puff only)
    f_v: float    # cross-wind momentum
    f_w: float    # vertical momentum (lofted only)
    f_t: float    # ground heat flux [J/(m s)]


# ===========================================================================
# friction velocity
# ===========================================================================
def friction(
    cl: CloudLocal,
    atm: Atmosphere,
    *,
    u_bar_ambient: float,
    u_star_source_sq: float | None = None,
    relax: float = 0.0,
    ground_coupled: bool = True,
    ground_cooling: float = 0.0,
    coeffs: Coefficients = COEFFS,
) -> Friction:
    """
    In-cloud friction velocity, EQ 35c-35e.

    Parameters
    ----------
    u_bar_ambient : depth-averaged ambient wind over the cloud, <ua> (JS `uab`).
    u_star_source_sq : U_bs^2 retained from the source region.  When given
        together with ``relax``, the ground-friction term is blended as
        ``U_bs^2 = U_bsi^2 + (U_bs0^2 - U_bsi^2) * exp(-relax)`` (JS L2841,
        L2845).  This keeps the enhanced shear of the source region from
        vanishing discontinuously downwind.
    relax : non-negative relaxation exponent; 0 means no blending.
    ground_coupled : False switches off the ground heat flux (used for the
        neutrally buoyant effective source of a momentum/buoyant jet, EQ 46d).
    """
    c = coeffs
    rho_ratio = atm.rho / cl.rho

    u_bar_g = 0.5 * cl.u_g
    v_bar_g = 0.5 * cl.v_g
    u_rho = u_bar_g * rho_ratio
    v_rho = v_bar_g * rho_ratio
    delta_u = (u_bar_ambient - cl.u) * rho_ratio

    c_f = atm.u_star / u_bar_ambient

    # top-of-cloud drag, U_mh*^2
    u_mh_sq = c.c_drag_top * (delta_u * delta_u + v_rho * v_rho)

    # ground friction, U_mg*^2
    if cl.is_lofted:
        u_bs_sq_inst = atm.u_star**2 + u_mh_sq
    else:
        u_bs_sq_inst = (
            (c_f * cl.u) ** 2
            + (c_f * v_bar_g) ** 2
            + c.c_source_shear * cl.w_source * u_bar_ambient
        )

    if u_star_source_sq is not None and relax > 0.0:
        u_bs_sq = u_bs_sq_inst + (u_star_source_sq - u_bs_sq_inst) * math.exp(-relax)
    else:
        u_bs_sq = u_bs_sq_inst
    u_bs_sq = max(u_bs_sq, 0.0)
    u_bs = math.sqrt(u_bs_sq)

    v_H = c_f * u_bs

    # thermal convection, U_t^2
    #
    # Ermak holds the ground at ambient temperature for the whole release.
    # Real ground under a cryogenic cloud cools: treating it as a
    # semi-infinite solid under a constant surface flux q gives
    #
    #     dT_s = 2 q sqrt(alpha t / pi) / k
    #
    # which for dry desert soil (k = 0.3 W/m/K, alpha = 2.3e-7 m^2/s) and the
    # 500-600 W/m^2 the cloud actually draws is about 10 K over a 107 s
    # release and 20 K over 381 s.  The offset is supplied by the caller
    # because it depends on elapsed time, which `friction` does not see.
    T_g = (atm.T - ground_cooling) if ground_coupled else cl.T
    if cl.is_lofted or T_g <= cl.T:
        u_t_sq = 0.0
    else:
        T_mean = 0.5 * (T_g + cl.T)
        u_t_sq = (
            c.c_thermal * PHYS.GRAVITY * (T_g - cl.T) * v_H * cl.h / T_mean
        ) ** (2.0 / 3.0)

    u_star_sq = u_bs_sq + u_mh_sq + u_t_sq
    u_star = math.sqrt(u_star_sq)

    # in-cloud Monin-Obukhov length, EQ 35d
    g_stratify = max(
        c.c_mu_strat * PHYS.GRAVITY * (cl.rho - atm.rho) / cl.rho, 0.0
    )
    inv_L_a = atm.inv_L_eff / (1.0 + cl.h_top / atm.z_L)
    # EQ 35d is a *sum* of two stratification sources: the ambient one the
    # cloud sits in, and the cloud's own density excess.  `w_ambient` weights
    # only the first, so setting it to zero removes the ambient pathway while
    # leaving everything else — including the cloud's own damping — intact.
    # That is the intervention that separates correlation from cause.
    inv_L = (
        (c.w_ambient_stratification * inv_L_a * atm.u_star**2 + g_stratify)
        / u_star_sq
        if u_star_sq > 0.0 else 0.0
    )

    return Friction(
        u_star=u_star,
        u_mg_star=u_bs,
        u_mh_star_sq=u_mh_sq,
        u_t_sq=u_t_sq,
        v_H=v_H,
        delta_u=delta_u,
        c_f=c_f,
        inv_L_cloud=inv_L,
        phi_h=_phi_h(cl.h_top, inv_L, coeffs=c),
        T_ground=T_g,
    )


def _phi_h(z: float, inv_L: float, *, coeffs: Coefficients = COEFFS) -> float:
    """Heat Monin-Obukhov function phi_h, EQ 35c."""
    if inv_L < 0.0:
        return 1.0 / math.sqrt(1.0 - coeffs.phi_unstable * z * inv_L)
    return 1.0 + coeffs.phi_stable * z * inv_L


# ===========================================================================
# entrainment velocities
# ===========================================================================
def entrainment(
    cl: CloudLocal,
    atm: Atmosphere,
    fr: Friction,
    *,
    u_bar_ambient: float,
    t_avg: float = 0.0,
    coeffs: Coefficients = COEFFS,
) -> Entrainment:
    """
    Reduced entrainment velocities, EQ 35a (vertical) and 36a/36b (horizontal).

    Parameters
    ----------
    t_avg : averaging time entering the meander function F_theta.  The
        instantaneous, meander-free solution of the conservation equations
        uses ``t_avg = 0``; meander is added afterwards in the concentration
        post-processing (section 2.6.2).  Passing a non-zero value here
        double-counts meander.
    """
    c = coeffs

    # ---- vertical, EQ 35a --------------------------------------------
    h_top = cl.h_top
    f_mix = 1.0 - h_top / atm.h_mix
    phi = fr.phi_h

    # A lofted cloud also entrains across the gap between ground and cloud
    # base; SLAB_entran L2882 adds a second, independent contribution.
    f_mix_b, phi_b = 0.0, 1.0
    if h_top > 1.01 * cl.h:
        h_below = h_top - cl.h
        f_mix_b = 1.0 - h_below / atm.h_mix
        phi_b = _phi_h(h_below, fr.inv_L_cloud, coeffs=c)

    if c.entrainment_closure == "slab":
        # EQ 35a as written.  `urf` is evaluated at a fixed 4 m
        # (SLAB.FOR L466-467), an absolute length inside a similarity
        # closure; see `docs/PREREG_froude.md`.
        u_ref_dimensionless = atm.dimensionless_profile(c.h_entrain_ref)
        u_top_dimensionless = atm.dimensionless_profile(h_top)
        shape = u_ref_dimensionless / u_top_dimensionless
        w = (shape * c.a_entrain * PHYS.VON_KARMAN * fr.u_star
             * (f_mix / phi + f_mix_b / phi_b))
    else:
        w = _richardson_entrainment(cl, atm, fr, h_top, f_mix, f_mix_b,
                                    coeffs=c)

    # ---- horizontal, EQ 36a ------------------------------------------
    # S(La): ambient-stability correction, referenced to the 2 m wind
    # A second absolute length, in EQ 36a this time: the ambient-stability
    # correction is referenced to the wind at a fixed 2 m.  Externalised as
    # `h_horiz_ref` so the Froude test can scale it; the default is 2.0,
    # which is what the original does.
    r_cf = math.sqrt(
        (atm.u_star / atm.wind_speed(c.h_horiz_ref)) / c.c_10
    )
    inv_La = atm.inv_L_ground
    if inv_La < 0.0:
        s_stab = 1.0 - r_cf * c.L_a_ref * inv_La
    else:
        s_stab = 1.0 / (1.0 + r_cf * c.L_a_ref * inv_La)

    f_theta = ((t_avg + c.tau_min * math.exp(-t_avg / c.tau_min))
               / c.t_ref_avg) ** c.p_meander
    a_total = 0.08 * s_stab * f_theta

    r_ab = 0.5 * c.a2_horiz / a_total
    v_ambient = a_total * u_bar_ambient / (1.0 + r_ab * cl.b_half / SQRT3)
    v_shear = math.sqrt(c.c_drag_top) * c.a_entrain * PHYS.VON_KARMAN * fr.delta_u

    # Gravity-front entrainment.  Not in Ermak: his EQ 36a has only ambient
    # turbulence and jet shear, so the advancing front entrains nothing.  A
    # gravity current mixes most vigorously at its head, and the other
    # integral models of the period all carry a term for it.
    #
    # It is returned separately from `v` because in SLAB the cross-wind
    # entrainment velocity does double duty — it dilutes (EQ 2a) *and* it
    # widens (EQ 7):
    #
    #     dR/dx  = rho_a sqrt3 (V_e h + W_e B)
    #     dBv/dx = (sqrt3 (rho_a/rho) V_e + V_g) / u
    #
    # Air drawn into the head of an existing gravity current dilutes the
    # cloud, but the position of the front is already set by V_g, which the
    # width equation carries in its own right.  Folding the frontal term into
    # V_e therefore double-counts the front's advance: it inflates B while
    # leaving the flat-core width b (driven only by V_g) behind, which makes
    # the cross-wind profile more sharply peaked and *raises* the centreline
    # concentration even as the cloud dilutes.  The term belongs in the mass
    # equation alone.
    #
    # It switches itself off exactly when it should: V_g is zero for a lofted
    # or a buoyant cloud, so no gating is needed.
    v_front = c.alpha_front * abs(cl.v_g)
    v = math.hypot(v_ambient, v_shear)

    # ---- horizontal down-wind, EQ 36b --------------------------------
    # shear here comes from the vertical gradient of the *ambient* wind
    sigma = (0.5 * cl.h / SQRT3 if cl.z_c > 0.5 * cl.h
             else (cl.h - cl.z_c) / SQRT3)
    z_r = cl.z_c + 0.5 * sigma
    v_x_shear = (
        c.c_shear_x * atm.u_star / PHYS.VON_KARMAN
        * (z_r / (z_r + atm.z0))
        * atm.phi_m(z_r)
        * (1.0 - z_r / atm.h_mix)
    )
    v_x_ambient = a_total * u_bar_ambient / (1.0 + r_ab * cl.b_half_x / SQRT3)
    v_x = math.hypot(v_x_ambient, v_x_shear)

    return Entrainment(
        w=w, v=v, v_x=v_x,
        v_ambient=v_ambient, v_shear=v_shear, v_front=v_front,
        a_total=a_total,
    )


# ===========================================================================
# fluxes
# ===========================================================================
def fluxes(
    cl: CloudLocal,
    atm: Atmosphere,
    fr: Friction,
    *,
    coeffs: Coefficients = COEFFS,
) -> Fluxes:
    """
    Momentum and heat fluxes, EQ 37-39.

    The quadratic drag terms are written ``Cf |x| x`` rather than ``Cf x^2``
    so that they always oppose the motion.
    """
    c = coeffs
    rho_ratio = atm.rho / cl.rho
    drag = lambda x: c.c_drag_top * abs(x) * x          # noqa: E731

    u_mh_sq = drag(fr.delta_u)
    ug_mh_sq = drag(0.5 * cl.u_g * rho_ratio)
    v_mh_sq = drag(0.5 * cl.v_g * rho_ratio)
    w_mh_sq = drag(cl.w_c)

    # In puff mode the fluxes are per unit length, so they scale with Bx.
    rho_eff = cl.rho * cl.b_half_x if cl.is_puff else cl.rho
    B, h = cl.b_half, cl.h

    if cl.is_lofted:
        return Fluxes(
            f_u=rho_eff * (2.0 * B + h) * u_mh_sq,
            f_ug=0.0,
            f_v=0.0,
            f_w=-rho_eff * h * w_mh_sq,
            f_t=0.0,
        )

    u_mg = fr.c_f * cl.u
    ug_mg = fr.c_f * 0.5 * cl.u_g
    v_mg = fr.c_f * 0.5 * cl.v_g
    return Fluxes(
        f_u=-rho_eff * (B * (u_mg**2 - atm.u_star**2) - (B + h) * u_mh_sq),
        f_ug=-rho_eff * B * (ug_mg**2 + ug_mh_sq),
        f_v=-rho_eff * B * (v_mg**2 + v_mh_sq),
        f_w=0.0,
        f_t=rho_eff * B * fr.v_H * cl.cp * (fr.T_ground - cl.T),
    )


# ===========================================================================
# Richardson-number closures -- variants, off by default
# ===========================================================================
def _bulk_richardson(cl, atm, fr) -> float:
    """
    The cloud's own bulk Richardson number, ``g' h / u*^2``.

    Dimensionless and built only from cloud-scale quantities, so unlike the
    fixed 4 m of EQ 35a it carries no absolute length.
    """
    d_rho = cl.rho - atm.rho
    if d_rho <= 0.0:
        return 0.0
    g_prime = PHYS.GRAVITY * d_rho / atm.rho
    u_star = max(fr.u_star, 1e-9)
    return g_prime * max(cl.h, 0.0) / (u_star * u_star)


def _richardson_entrainment(cl, atm, fr, h_top, f_mix, f_mix_b, *,
                            coeffs) -> float:
    """
    Vertical entrainment from a published Richardson-number function, in
    place of EQ 35a's ``urf/uah`` factor.

    Two forms, both taken as published and neither refitted:

    ``"robins"``
        Robins, Carruthers & Britter (2001), Atmos. Environ. 35, 2243-2252::

            W_E / u* = 0.65 / (1 + 0.2 Ri*)      (Ri* < 15)

        Measured directly from the down-wind development of the
        concentration field of a two-dimensional dense plume over a rough
        surface, rather than inferred from a model.

    ``"nielsen"``
        Nielsen (1998), Risoe-R-1030(EN)::

            u_e / e = 0.25 / (3.3 + Ri)
            e = (u*^3 + 0.1 w*^3)^(2/3)

        Constructed to satisfy the passive, stratified-shear, weak- and
        strong-convective limits simultaneously.  ``w*`` here uses the
        cloud's own buoyancy flux against the ground rather than a surface
        heat flux, since a dense cloud's convection is driven by its own
        density excess.

    The mixing-height taper ``f_mix`` is kept: it is a separate physical
    constraint (the cloud cannot entrain from above the inversion) and is
    not what the Froude test is about.  The stability damping ``phi`` is
    *not* kept -- the Richardson number already carries it, and applying
    both would damp twice.
    """
    c = coeffs
    ri = _bulk_richardson(cl, atm, fr)
    u_star = max(fr.u_star, 1e-9)
    taper = f_mix + f_mix_b

    if c.entrainment_closure == "robins":
        return c.c_robins * u_star * taper / (1.0 + c.b_robins * ri)

    if c.entrainment_closure == "nielsen":
        # Atmospheric convective scale: zero unless the surface layer is
        # unstable.  w* = (g H_f h_mix / (rho cp T))^(1/3), written here
        # through the Monin-Obukhov length as w* = u* (-h_mix/(kappa L))^(1/3).
        w_star = 0.0
        inv_L = atm.inv_L_ground
        if inv_L < 0.0:
            w_star = u_star * (-atm.h_mix * inv_L / PHYS.VON_KARMAN) ** (1 / 3)
        # (1/3), not the (2/3) the form is sometimes quoted with: the
        # latter gives e the dimensions of velocity squared, so `u_e / e`
        # would be an inverse velocity, and under Froude scaling the
        # entrainment velocity would go as s rather than the required
        # sqrt(s).  A dimensional check settles it without needing the
        # source.
        e = (u_star ** 3 + 0.1 * w_star ** 3) ** (1 / 3)
        return c.c_nielsen * e * taper / (c.a_nielsen + ri)

    raise ValueError(
        f"entrainment_closure must be slab, robins or nielsen; "
        f"got {c.entrainment_closure!r}"
    )
