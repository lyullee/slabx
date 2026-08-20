"""
Transient puff mode
===================

Solves EQ 15-26 for a cloud that is no longer being fed by its source.

Differences from the plume mode
-------------------------------
* The independent variable is travel time, not distance.  Down-wind position
  follows from the centre-of-mass equation (EQ 22).
* `R = rho Bx By h` is a *mass*, not a mass flux, so the cloud carries a fixed
  inventory once the release has stopped.
* There is no velocity cubic.  EQ 18 is a plain translation equation,

      U = (1 - m) <ua> + (Sru0 + Sfue) / R                      EQ 18

  because gravity flow is carried by the separate equations EQ 19a (down-wind)
  and EQ 20a (cross-wind) rather than appearing as a pressure gradient in the
  translation equation.  Puff mode therefore cannot fail the way EQ 4d fails,
  and needs no source-area expansion.
* Two extra length scales, `Bx` and `bx`, evolve alongside `By` and `by`, and
  the density stretch is applied anisotropically (SLAB_evalpf L2718).

Transition
----------
Section 2.3.3 puts the switch at the moment the centre of mass has passed,
i.e. when the released material inside the cloud reaches half the total,

    Qint >= q t_sd / 2                                          EQ 29a

Continuity of the spatially-averaged properties then fixes the half-length,

    Bx = R_puff / (rho By h) = U t_sd / 2                        EQ 30

with ``R_puff = q t_sd / (4 m)``.  The two forms agree identically because the
plume relation ``m = q / (2 rho U By h)`` holds at the transition point; the
implementation uses the first and `transition_half_length` checks the second.

The per-unit-length fluxes of the plume equations become totals, so they are
multiplied by `Bx`, and the down-wind gravity velocity is seeded from the
cross-wind one through the aspect ratio, ``U_g = (By / Bx) V_g``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..coefficients import PHYS, Coefficients, COEFFS
from ..submodels.atmosphere import Atmosphere
from ..submodels.entrainment import CloudLocal, entrainment, fluxes, friction
from ..thermo.base import ThermoBackend
from ..thermo.equilibrium import Mixture, solve_equilibrium
from .plume import PlumeState, _mean_ambient
from .source import SourceModel
from .trajectory import Mode

__all__ = ["PuffState", "transition_state", "slopes_puff", "integrate_puff"]

SQRT3 = math.sqrt(3.0)


# ===========================================================================
@dataclass
class PuffState:
    """The fifteen integrated quantities plus the reconstructed cloud state."""

    # integrated
    R: float          #: rho Bx By h [kg]  — an inventory, not a flux
    x_c: float        #: centre-of-mass position [m] (EQ 22)
    Bv: float
    bv: float
    Bvx: float
    bvx: float
    z_c: float
    t: float

    # reconstructed
    h: float
    b_half: float
    b_shape: float
    b_half_x: float
    b_shape_x: float
    u: float
    T: float
    rho: float
    cp: float
    m: float
    m_ev: float
    m_water: float
    m_wv: float
    v_g: float = 0.0
    u_g: float = 0.0
    w_c: float = 0.0
    u_bar: float = 0.0
    h_top: float = 0.0
    pool: float = 0.0
    """Material already rained out, carried over from the plume phase."""

    w_e: float = 0.0
    v_e: float = 0.0
    v_ex: float = 0.0
    v_front: float = 0.0
    f_u: float = 0.0
    f_ug: float = 0.0
    f_v: float = 0.0
    f_w: float = 0.0
    f_t: float = 0.0
    u_star: float = 0.0
    phi_h: float = 0.0
    inv_L: float = 0.0
    mass_in_cloud: float = 0.0

    @property
    def beta(self) -> float:
        return math.sqrt(max(self.b_half**2 - self.b_shape**2, 0.0)) / SQRT3

    @property
    def beta_x(self) -> float:
        return math.sqrt(max(self.b_half_x**2 - self.b_shape_x**2, 0.0)) / SQRT3

    @property
    def is_lofted(self) -> bool:
        return self.h_top > self.h

    def vol_frac(self, mw_emission: float, mw_ambient: float) -> float:
        return mw_ambient * self.m / (mw_emission + (mw_ambient - mw_emission) * self.m)


# ===========================================================================
# transition, section 2.3.3
# ===========================================================================
def transition_half_length(u: float, duration: float) -> float:
    """EQ 30 in its explicit form, ``Bx = U t_sd / 2``."""
    return 0.5 * u * duration


def transition_state(
    st: PlumeState, atm: Atmosphere, src: SourceModel, t: float
) -> PuffState:
    """
    Convert the plume state at the transition into a puff state (EQ 30).

    Everything spatially averaged is continuous across the switch by
    construction; only the bookkeeping changes.
    """
    if st.m <= 0.0:
        raise ValueError("cannot transition with zero concentration")

    R = 0.25 * src.rate * src.duration / st.m
    Bx = R / (st.rho * st.b_half * st.h)

    fx = st.b_half_x if False else Bx      # fluxes become totals
    return PuffState(
        R=R, x_c=st.Qint * 0.0, Bv=st.b_half, bv=st.b_shape,
        Bvx=Bx, bvx=0.9999 * Bx, z_c=st.z_c, t=t,
        h=st.h, b_half=st.b_half, b_shape=st.b_shape,
        b_half_x=Bx, b_shape_x=0.9999 * Bx,
        u=st.u, T=st.T, rho=st.rho, cp=st.cp, m=st.m, m_ev=st.m_ev,
        m_water=st.m_water, m_wv=st.m_wv,
        v_g=st.v_g,
        u_g=(st.b_half / Bx) * st.v_g,     # seed from the cross-wind flow
        w_c=st.w_c, u_bar=st.u_bar, h_top=st.h_top,
        w_e=st.w_e, v_e=st.v_e, v_ex=st.v_ex,
        f_u=fx * st.f_u, f_v=fx * st.f_v, f_w=fx * st.f_w, f_t=fx * st.f_t,
        u_star=st.u_star, phi_h=st.phi_h, inv_L=st.inv_L,
        mass_in_cloud=st.Qint, pool=getattr(st, "pool", 0.0),
    )


# ===========================================================================
# slopes, EQ 15-26
# ===========================================================================
def slopes_puff(
    st: PuffState,
    atm: Atmosphere,
    src: SourceModel,
    *,
    coeffs: Coefficients = COEFFS,
) -> list[float]:
    """
    The fifteen derivatives, mirroring `SLAB_slopepf` (SLAB.FOR L1984, subroutine slopepf).

        0  R      1  x_c    2  Bv     3  bv     4  Bvx    5  bvx
        6  G      7  Gx     8  Sft    9  Sfu   10  Gw    11  z_c
       12  Sfy   13  Sfx   14  Sfz
    """
    c = coeffs
    f = [0.0] * 15
    B, Bx, h = st.b_half, st.b_half_x, st.h
    d_rho = st.rho - atm.rho

    if st.t >= src.duration or src.rate <= 0.0:
        rqs, atau = 0.0, 0.0
    else:
        rqs = 0.25 * src.rate
        atau = rqs / st.R if st.R > 0 else 0.0

    # EQ 15a, with gravity-front entrainment added to the dilution only.
    f[0] = (atm.rho * SQRT3 * (((st.v_ex + st.v_front) * B
                                + (st.v_e + st.v_front) * Bx) * h
                               + st.w_e * B * Bx) + rqs)
    f[1] = st.u - atau * st.x_c
    f[2] = SQRT3 * (atm.rho / st.rho) * st.v_e + st.v_g
    f[3] = (st.v_g * st.b_shape) / B if B > 0 else 0.0
    f[4] = SQRT3 * (atm.rho / st.rho) * st.v_ex + st.u_g
    f[5] = (st.u_g * st.b_shape_x) / Bx if Bx > 0 else 0.0

    f_grav = c.alpha_gv * PHYS.GRAVITY * d_rho * h * h
    f[6] = f_grav * Bx
    f[7] = f_grav * B
    f[8] = st.f_t
    f[9] = st.f_u
    f[10] = -PHYS.GRAVITY * d_rho * B * Bx * h
    f[11] = st.w_c
    f[12] = st.f_v
    f[13] = st.f_ug
    f[14] = st.f_w
    return f


# ===========================================================================
# reconstruction, EQ 18 + anisotropic stretch  (SLAB_evalpf)
# ===========================================================================
def _reconstruct_puff(
    *, R, rho, m, z_c, atm, prev: PuffState,
    Bv, bv, Bvx, bvx, Bv0, bv0, Bvx0, bvx0,
    G, Gx, Gw, Sfu, Sfy, Sfx, Sfz, R0, Sru0, coeffs,
):
    """
    Recover the four length scales, the height and the velocities.

    Unlike the plume the system is explicit — no fixed point — because EQ 18
    gives U directly.  The density stretch is split between the two horizontal
    directions in proportion to the cloud's aspect ratio (SLAB_evalpf L2718),
    so that a long thin cloud stretches mostly along its short axis.
    """
    B0, b0, Bx0, bx0 = prev.b_half, prev.b_shape, prev.b_half_x, prev.b_shape_x
    rho0, h0, htp0, u_bar0 = prev.rho, prev.h, prev.h_top, prev.u_bar

    denom = h0 * (Bx0 + B0) + 2.0 * B0 * Bx0
    g_y = Bx0 * h0 / denom if denom > 0 else 0.0
    g_x = (B0 / Bx0) * g_y if Bx0 > 0 else 0.0
    rey, rex = (rho0 / rho) ** g_y, (rho0 / rho) ** g_x

    B = (B0 + Bv - Bv0) * rey
    b = (b0 + bv - bv0) * rey
    Bx = (Bx0 + Bvx - Bvx0) * rex
    bx = (bx0 + bvx - bvx0) * rex

    h = min(R / (rho * Bx * B), atm.h_mix)
    z_c = min(max(z_c, 0.0), atm.h_mix - 0.5 * h)
    htp = z_c + 0.5 * h if z_c > 0.5 * h else h

    u1 = atm.dimensionless_profile(max(htp0, 1e-9))
    u_bar = _mean_ambient(atm, htp0, htp, u_bar0, u1)

    sfue = (0.0 if Sfu == 0.0 else
            -R0 * (prev.u - u_bar0) * abs(Sfu)
            / (abs(Sfu) + R0 * abs(prev.u - u_bar0)))
    u = (1.0 - m) * u_bar + (Sru0 + sfue) / R          # EQ 18

    # -- gravity velocities, EQ 19a / 20a / 21 --------------------------
    v_g = u_g = w_c = 0.0
    if htp > h or rho <= atm.rho:                      # lofted or buoyant
        ew = (1.0 if prev.w_c == 0.0 else
              R0 * abs(prev.w_c) / (R0 * abs(prev.w_c) + abs(Sfz)))
        w_c = ew * (Gw + R0 * prev.w_c) / R
    else:
        rbxy = B0 * Bx0 / (B0 * B0 + Bx0 * Bx0) if (B0 or Bx0) else 0.0
        vg0, ug0 = prev.v_g, prev.u_g
        if vg0 == 0.0:
            ev = 1.0
            vg0 = -2.0 * rbxy * Bx0 * prev.w_c / h0 if h0 > 0 else 0.0
        else:
            ev = R0 * vg0 / (R0 * vg0 + abs(Sfy))
        v_g = ev * (G + R0 * vg0) / R
        if ug0 == 0.0:
            eu = 1.0
            ug0 = -2.0 * rbxy * B0 * prev.w_c / h0 if h0 > 0 else 0.0
        else:
            eu = R0 * ug0 / (R0 * ug0 + abs(Sfx))
        u_g = eu * (Gx + R0 * ug0) / R
        w_c = -(v_g / B + u_g / Bx) * z_c if B > 0 and Bx > 0 else 0.0

    return dict(B=B, b=b, Bx=Bx, bx=bx, h=h, z_c=z_c, htp=htp,
                u=u, u_bar=u_bar, v_g=v_g, u_g=u_g, w_c=w_c)


# ===========================================================================
# integration
# ===========================================================================
_RK_W = (1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0)
_RK_O = (0.5, 0.5, 1.0)


def integrate_puff(
    st: PuffState,
    atm: Atmosphere,
    src: SourceModel,
    emission: ThermoBackend,
    water: ThermoBackend,
    *,
    t_max: float,
    n_steps: int = 40,
    substeps: int = 3,
    growth: float = 1.12,
    coeffs: Coefficients = COEFFS,
) -> list[dict]:
    """
    Integrate the puff equations from `st` to travel time `t_max`.

    Time steps grow geometrically, which is what the reference does (its
    `gam` factor, SLAB.FOR L1561): the cloud changes fast just after the
    release ends and slowly thereafter.

    Returns rows in `Trajectory.from_rows` format so that the caller can
    concatenate them with the plume rows.
    """
    if t_max <= st.t:
        return []
    span = t_max - st.t
    dt0 = span * (growth - 1.0) / (growth**n_steps - 1.0) if growth > 1.0 \
        else span / n_steps

    rows: list[dict] = []
    Sru = st.R * (st.u - (1.0 - st.m) * st.u_bar)
    dt = dt0
    for _ in range(n_steps):
        for _ in range(substeps):
            st, Sru = _rk4_step_puff(
                st, Sru, dt / substeps, atm, src, emission, water, coeffs
            )
        rows.append(row_puff(st, atm, emission))
        dt *= growth
    return rows


def _rk4_step_puff(st, Sru0, dt, atm, src, emission, water, coeffs):
    prev = st
    R0, xc0 = st.R, st.x_c
    Bv0, bv0, Bvx0, bvx0, zc0 = st.Bv, st.bv, st.Bvx, st.bvx, st.z_c
    t0 = st.t

    total = [0.0] * 15
    stage = st
    for k in range(4):
        f = slopes_puff(stage, atm, src, coeffs=coeffs)
        for j in range(15):
            total[j] += _RK_W[k] * dt * f[j]
        dy = total if k == 3 else [dt * _RK_O[k] * f[j] for j in range(15)]
        t_stage = t0 + (dt * _RK_O[k] if k < 3 else dt)
        stage = _build_stage_puff(
            dy=dy, t=t_stage, R0=R0, xc0=xc0, Bv0=Bv0, bv0=bv0,
            Bvx0=Bvx0, bvx0=bvx0, zc0=zc0, Sru0=Sru0, prev=prev,
            atm=atm, src=src, emission=emission, water=water, coeffs=coeffs,
        )
    Sru = stage.R * (stage.u - (1.0 - stage.m) * stage.u_bar)
    return stage, Sru


def _build_stage_puff(*, dy, t, R0, xc0, Bv0, bv0, Bvx0, bvx0, zc0, Sru0,
                      prev, atm, src, emission, water, coeffs):
    R = R0 + dy[0]
    if R <= 0.0:
        raise RuntimeError(f"non-positive puff mass R = {R:.3e} at t = {t:.3f}")
    x_c = xc0 + dy[1]
    Bv, bv, Bvx, bvx = Bv0 + dy[2], bv0 + dy[3], Bvx0 + dy[4], bvx0 + dy[5]
    G, Gx, Sft, Sfu, Gw = dy[6], dy[7], dy[8], dy[9], dy[10]
    z_c = zc0 + dy[11]
    Sfy, Sfx, Sfz = dy[12], dy[13], dy[14]

    # EQ 15a: the inventory is fixed once the release has stopped.  The
    # factor 1/4 belongs to the *concentration* (R describes a quarter of the
    # cloud, as B and Bx are half-widths); the mass actually carried is half
    # the release, matching the plume-mode Qint = 0.5 q t of EQ 29b.
    released = src.released_mass(t)
    m = min(max(0.25 * released / R, 0.0), 1.0)
    mass_in_cloud = 0.5 * released

    m_wa = atm.mass_frac_water
    m_water, m_dry = (1.0 - m) * m_wa, (1.0 - m) * (1.0 - m_wa)
    ratio = R0 / R
    m_wv_t = m_water + ratio * (prev.m_wv - prev.m_water)
    m_ev_t = m + ratio * (prev.m_ev - prev.m)

    cp_s, T_s = emission.substance.cp_vapour, src.T
    dT0 = prev.T - atm.T
    sfte = (0.0 if Sft == 0.0 else
            -prev.cp * dT0 * abs(Sft) / (abs(Sft) + R0 * prev.cp * abs(dT0)))
    E = ((1.0 - m) * atm.cp_moist * atm.T + m * cp_s * T_s
         + ratio * (prev.cp * prev.T
                    - (1.0 - prev.m) * atm.cp_moist * atm.T
                    - prev.m * cp_s * T_s + sfte))

    eq = solve_equilibrium(
        Mixture(m_emission=m, m_water=m_water, m_dry_air=m_dry,
                m_ev_transported=m_ev_t, m_wv_transported=m_wv_t, enthalpy=E),
        emission, water, T_ambient=atm.T, rho_ambient=atm.rho,
        mw_ambient_moist=atm.mw_moist, T_guess=prev.T, coeffs=coeffs,
    )

    rec = _reconstruct_puff(
        R=R, rho=eq.rho, m=m, z_c=z_c, atm=atm, prev=prev,
        Bv=Bv, bv=bv, Bvx=Bvx, bvx=bvx,
        Bv0=Bv0, bv0=bv0, Bvx0=Bvx0, bvx0=bvx0,
        G=G, Gx=Gx, Gw=Gw, Sfu=Sfu, Sfy=Sfy, Sfx=Sfx, Sfz=Sfz,
        R0=R0, Sru0=Sru0, coeffs=coeffs,
    )

    st = PuffState(
        R=R, x_c=x_c, Bv=Bv, bv=bv, Bvx=Bvx, bvx=bvx, z_c=rec["z_c"], t=t,
        h=rec["h"], b_half=rec["B"], b_shape=rec["b"],
        b_half_x=rec["Bx"], b_shape_x=rec["bx"],
        u=rec["u"], T=eq.T, rho=eq.rho, cp=eq.cp, m=m, m_ev=eq.m_ev,
        m_water=m_water, m_wv=eq.m_wv,
        v_g=rec["v_g"], u_g=rec["u_g"], w_c=rec["w_c"],
        u_bar=rec["u_bar"], h_top=rec["htp"],
        mass_in_cloud=mass_in_cloud,
    )

    cl = CloudLocal(
        h=st.h, b_half=st.b_half, b_half_x=st.b_half_x, z_c=st.z_c, u=st.u,
        T=st.T, rho=st.rho, cp=st.cp, v_g=st.v_g, u_g=st.u_g, w_c=st.w_c,
        is_puff=True,
    )
    fr = friction(cl, atm, u_bar_ambient=max(st.u_bar, 1e-6), coeffs=coeffs)
    en = entrainment(cl, atm, fr, u_bar_ambient=max(st.u_bar, 1e-6), coeffs=coeffs)
    fx = fluxes(cl, atm, fr, coeffs=coeffs)
    st.w_e, st.v_e, st.v_ex = en.w, en.v, en.v_x
    st.v_front = en.v_front
    st.f_u, st.f_v, st.f_w, st.f_t = fx.f_u, fx.f_v, fx.f_w, fx.f_t
    st.f_ug = fx.f_ug
    st.u_star, st.phi_h, st.inv_L = fr.u_star, fr.phi_h, fr.inv_L_cloud
    return st


def row_puff(st: PuffState, atm: Atmosphere, emission: ThermoBackend) -> dict:
    cv = st.vol_frac(emission.substance.mw, atm.mw_moist)
    sigma = (0.5 * st.h / SQRT3 if st.z_c > 0.5 * st.h
             else max(st.h - st.z_c, 0.0) / SQRT3)
    return dict(
        x=st.x_c, t=st.t, mode=int(Mode.PUFF),
        h=st.h, z_c=st.z_c, b_half=st.b_half, b_shape=st.b_shape,
        beta=st.beta, b_half_x=st.b_half_x, b_shape_x=st.b_shape_x,
        beta_x=st.beta_x, sigma_z=sigma,
        u=st.u, T=st.T, rho=st.rho, cp=st.cp,
        mass_frac=st.m, vol_frac=cv,
        v_g=st.v_g, u_g=st.u_g, w_c=st.w_c, u_ambient_mean=st.u_bar,
        mass_frac_dry_air=(1.0 - st.m) * (1.0 - atm.mass_frac_water),
        mass_frac_water=st.m_water, mass_frac_water_vapour=st.m_wv,
        mass_frac_emission_vapour=st.m_ev,
        mass_frac_water_liquid=st.m_water - st.m_wv,
        mass_frac_emission_liquid=st.m - st.m_ev,
        w_entrain=st.w_e, v_entrain=st.v_e, v_x_entrain=st.v_ex,
        u_star_cloud=st.u_star, phi_h=st.phi_h, inv_L_cloud=st.inv_L,
        f_u=st.f_u, f_v=st.f_v, f_w=st.f_w, f_t=st.f_t,
        mass_in_cloud=st.mass_in_cloud, R_flux=st.R, rained_out=st.pool,
    )
