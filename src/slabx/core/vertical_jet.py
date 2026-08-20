"""
Vertical jet and plume rise
===========================

`IDSPL = 3`: material leaves a stack vertically, rises, bends over, and only
then disperses as a plume.  The rise happens over a short down-wind distance
— a metre or two for the manual's chlorine example — but it sets the height,
width and dilution the plume starts from, so it cannot be skipped.

Two rise mechanisms (section 2.5.5, EQ 45-47)
---------------------------------------------
**Momentum**, always present.  The jet's vertical momentum carries it up
against the cross-wind until entrainment has replaced it with ambient
momentum.  EQ 46b gives, after a three-pass fixed point,

    h_pm = 1.0363 W_s B_s / (beta_u sqrt(<ua> u_a*)),
    beta_u = 0.4 + 1.2 <ua> / W_s

**Density**, whose sign decides everything else:

* *denser than air* — the jet cannot rise indefinitely; it reaches a maximum
  determined by its densimetric Froude number (Hoot, Meroney & Peterka 1973)
  and falls back.  The two rises combine reciprocally,
  ``h_pr = h_pm h_pd / sqrt(h_pm^2 + h_pd^2)``, so the smaller dominates.
  The cloud is still dense at the top of the rise and disperses from there.

* *lighter than air* — Briggs buoyant rise adds in quadrature,
  ``h_pr = sqrt(h_pm^2 + h_pb^2)``.  By the top of the rise the cloud has
  entrained enough air to be neutrally buoyant, so SLAB hands it to the
  horizontal-jet equations at the risen height with ambient density.  That
  branch is `HorizontalJet` with an elevated release point.

Only the dense branch is implemented here; the buoyant branch is reported
rather than silently approximated.

The rise region (EQ 47)
-----------------------
Rather than integrating through the rise, SLAB interpolates across it.  With
``xi = (x - x_0)/x_pr`` running 0 to 1 over the rise length,

    U(xi)   = U_1 sqrt(xi)
    z_c(xi) = h_s + h_pr sqrt(1 - (xi - 1)^2)
    h(xi)   = h_0 + (h_1 - h_0) xi
    B(xi)   = B_s + (B_1 - B_s) xi
    m(xi)   = 1 / (1 + xi^2 (1 - m_1)/m_1)

with subscript 1 the fully-risen state.  `rise_profile` returns these rows so
the driver can prepend them to the trajectory; `initial_state` returns the
state at ``xi = 1``, where the plume integration starts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..coefficients import COEFFS, PHYS, Coefficients
from ..submodels.atmosphere import Atmosphere
from ..core.trajectory import Mode
from .source import SourceFlux, SourceState, _PressurisedRelease, _mean_wind

__all__ = ["VerticalJet", "PlumeRise", "BuoyantRiseNotImplemented"]

SQRT3 = math.sqrt(3.0)


class BuoyantRiseNotImplemented(NotImplementedError):
    """The release is lighter than air at the stack; see module docstring."""


@dataclass(frozen=True)
class PlumeRise:
    """Geometry and dilution at the top of the rise."""

    h_momentum: float   #: momentum-only rise [m] (EQ 46b)
    h_dense: float      #: maximum rise of a dense jet [m] (HMP 1973)
    h_rise: float       #: combined rise actually used [m] (EQ 46a)
    x_rise: float       #: down-wind distance over which it happens [m]
    vol_frac: float     #: volume fraction at the top of the rise
    mass_frac: float    #: mass fraction at the top of the rise
    z_c: float          #: centre height at the top of the rise [m]


@dataclass(frozen=True)
class VerticalJet(_PressurisedRelease):
    """
    Vertical jet from a stack, ``IDSPL = 3``.

    Parameters follow `_PressurisedRelease`; `height` is the stack height and
    `area` the exit area.  Unlike a horizontal jet the exit velocity is
    vertical, so `rate` sets `W_s` rather than `U_s`.
    """

    name: str = "vertical_jet"

    # -- exit conditions -------------------------------------------------
    def w_source(self, atm: Atmosphere) -> float:
        """Vertical exit velocity, ``W_s = q / (rho_s A_s)``."""
        return self.rate / (self.rho_source(atm) * self.area)

    @property
    def half_width(self) -> float:
        return 1.0                       # EQ 29b table: X_s = 1 m, B_xs = 0

    @property
    def total_mass(self) -> float:
        return math.inf if math.isinf(self.duration) else self.rate * self.duration

    def released_mass(self, t: float) -> float:
        return self.rate * min(max(t, 0.0), self.duration)

    def expanded(self, half_width: float) -> "VerticalJet":
        return self

    def flux(self, x: float, t: float) -> SourceFlux:
        return SourceFlux()

    # -- plume rise ------------------------------------------------------
    def rise(self, atm: Atmosphere, *, coeffs: Coefficients = COEFFS) -> PlumeRise:
        """
        Solve EQ 45-46 for the rise height and the state at the top.

        Raises
        ------
        BuoyantRiseNotImplemented
            If the release is lighter than air at the stack.
        """
        rho = self.rho_source(atm)
        w_s = self.w_source(atm)
        B_s = self.half_width_geometric
        z_s = max(self.height, 1.0)                 # SLAB.FOR L594-601
        u_a = atm.wind_speed(z_s)

        # -- momentum rise, EQ 46b ---------------------------------------
        beta_u = 0.4 + 1.2 * u_a / w_s
        h_pm = 1.0363 * w_s * B_s / (beta_u * math.sqrt(u_a * atm.u_star))
        p = 6.0 / 7.0
        h67 = h_pm**p
        for _ in range(3):                          # SLAB's three passes
            h_t = h67 * (self.height + h_pm) ** (1.0 - p)
            h_pm = ((self.height + p * h_pm) * h_t
                    / (self.height + h_pm - (1.0 - p) * h_t))

        if rho <= atm.rho:
            raise BuoyantRiseNotImplemented(
                f"stack density {rho:.3f} <= ambient {atm.rho:.3f}; the "
                "Briggs buoyant branch of EQ 46a is not implemented"
            )

        # -- dense-gas maximum rise, Hoot/Meroney/Peterka (1973) ---------
        d_s = 4.0 * B_s / math.sqrt(math.pi)        # equivalent stack diameter
        froude2 = w_s * w_s / (PHYS.GRAVITY * d_s * (rho - atm.rho) / atm.rho)
        sg = rho / atm.rho
        r_wu = w_s / u_a
        h_pd = 1.32 * d_s * (r_wu * sg * froude2) ** (1.0 / 3.0)

        # EQ 46a: the two rises combine reciprocally, so the smaller wins
        h_pr = h_pm * h_pd / math.hypot(h_pd, h_pm)
        h_pr = min(h_pr, atm.h_mix - self.height)

        x_pr = 0.435 * h_pr**3 / (r_wu * r_wu * sg * d_s * d_s)

        # -- dilution at the top of the rise -----------------------------
        cv_peak = min(1.69 * r_wu / (h_pr / d_s) ** 1.85, 0.99)
        cv = (0.5236 * (1.0 - cv_peak) + cv_peak) * cv_peak
        cm = (self.substance.mw * cv
              / (atm.mw_moist + (self.substance.mw - atm.mw_moist) * cv))

        return PlumeRise(
            h_momentum=h_pm, h_dense=h_pd, h_rise=h_pr, x_rise=x_pr,
            vol_frac=cv, mass_frac=cm, z_c=self.height + h_pr,
        )

    # -- state at the top of the rise ------------------------------------
    def initial_state(self, atm: Atmosphere, *, dx: float,
                      coeffs: Coefficients = COEFFS,
                      emission=None, water=None, **kw) -> SourceState:
        """
        State at ``x = 1 + x_pr``, where the plume integration begins.

        By the top of the rise the jet has already entrained enough air to be
        roughly half diluted, so it is **not** at its exit temperature any
        more: droplets have evaporated and the mixture has cooled well below
        the boiling point.  `emission` and `water` are therefore required to
        get T and rho right — using the exit values instead inflates the
        density tenfold and shrinks the cloud cross-section by the same
        factor (SLAB.FOR L1157 calls SLAB_thermo here for exactly this).

        The velocity comes from a quadratic balance between the entrained
        ambient momentum and the drag accumulated over the rise
        (SLAB.FOR L1157-1161), and the cloud is then shaped to hold the
        implied cross-section with an aspect ratio that tightens as the rise
        gets longer relative to the stack (`h_ft`).
        """
        c = coeffs
        pr = self.rise(atm, coeffs=c)
        B_s = self.half_width_geometric
        rho = self.rho_source(atm)
        m = pr.mass_frac

        u_bar = atm.wind_speed(pr.z_c)
        cf_pr = 3.0 * (2.0 + pr.x_rise / B_s) * c.c_drag_top
        cm_pr = math.sqrt((1.0 - m) ** 2 + 4.0 * m * cf_pr)
        u = 0.5 * u_bar / (1.0 - cf_pr) * (1.0 - m - 2.0 * cf_pr + cm_pr)

        T, cp, m_ev, m_wv = self.T, self.cp_source(), m * self.vapour_fraction, 0.0
        if emission is not None and water is not None:
            eq = self._equilibrium_at_rise(atm, m, emission, water, coeffs)
            rho, T, cp = eq.rho, eq.T, eq.cp
            m_ev, m_wv = eq.m_ev, eq.m_wv

        area_total = self.rate / (rho * u * m)      # cross-section of the cloud
        area_mat = self.rate / (rho * u)            # of the material alone
        h_ft = 2.4 + 1.6 / (1.0 + pr.x_rise**2 / (100.0 * B_s * B_s))

        B = max(math.sqrt(area_total / h_ft), B_s)
        b = max(0.9 * math.sqrt(area_mat / h_ft), 0.9 * B_s)
        h = area_total / (2.0 * B)
        z_c = pr.z_c

        if 0.5 * h > z_c:                           # cloud reaches the ground
            h_top = z_c + 0.5 * h
            B, b, h = h * B / h_top, h * b / h_top, h_top
        if h > atm.h_mix:
            B, b, h = h * B / atm.h_mix, h * b / atm.h_mix, atm.h_mix
        z_c = min(z_c, atm.h_mix - 0.5 * h)
        h_top = z_c + 0.5 * h if z_c > 0.5 * h else h

        return SourceState(
            x_start=1.0 + pr.x_rise, t_start=0.0, mode=Mode.PLUME,
            h=h, z_c=z_c, b_half=B, b_shape=b,
            b_half_x=1.0, b_shape_x=1.0,
            u=u, T=T, rho=rho, cp=cp,
            m_emission=m, m_ev=m_ev,
            m_water=(1.0 - m) * atm.mass_frac_water, m_wv=m_wv,
            # Point value at the risen centre height, not a depth
            # average: the cloud arrives there already moving with the wind
            # at that level (SLAB.FOR L1154).
            u_ambient_mean=u_bar,
            R_flux=rho * u * B * h,
        )

    def _equilibrium_at_rise(self, atm, m, emission, water, coeffs):
        """
        Phase equilibrium at the top of the rise.

        The enthalpy is simply the source stream mixed with the entrained
        air, because the carry term of EQ 41 collapses when the previous
        state *is* the source: with ``m0 = 1`` and ``R0/R = m`` it reduces to

            E = (1 - m) cp_a T_a + m cp_s T_s
        """
        from ..thermo.equilibrium import Mixture, solve_equilibrium

        m_water = (1.0 - m) * atm.mass_frac_water
        E = (1.0 - m) * atm.cp_moist * atm.T + m * self.cp_source() * self.T
        return solve_equilibrium(
            Mixture(m_emission=m, m_water=m_water,
                    m_dry_air=(1.0 - m) * (1.0 - atm.mass_frac_water),
                    m_ev_transported=m * self.vapour_fraction,
                    m_wv_transported=m_water, enthalpy=E),
            emission, water, T_ambient=atm.T, rho_ambient=atm.rho,
            mw_ambient_moist=atm.mw_moist, T_guess=self.T, coeffs=coeffs,
        )

    # -- the rise region itself, EQ 47 ------------------------------------
    def rise_profile(self, atm: Atmosphere, *, n: int | None = None,
                     coeffs: Coefficients = COEFFS,
                     emission=None, water=None) -> list[dict]:
        """
        Interpolated rows across the rise, ``x = 1`` to ``x = 1 + x_pr``.

        Returned in `Trajectory.from_rows` format.  The last row coincides
        with `initial_state`, so the driver drops it to avoid a duplicate.
        """
        pr = self.rise(atm, coeffs=coeffs)
        end = self.initial_state(atm, dx=1.0, coeffs=coeffs,
                                 emission=emission, water=water)
        B_s = self.half_width_geometric
        rho_s = self.rho_source(atm)
        w_s = self.w_source(atm)

        if n is None:                               # SLAB.FOR L1152-1153
            n = min(int(2.0 + pr.x_rise / (2.0 * B_s)), 11)
        n = max(n, 2)

        h0 = (2.0 * B_s if w_s > end.u
              else 2.0 * B_s * w_s / max(end.u, 1e-12))
        rows = []
        for i in range(n):
            xi = i / (n - 1)
            x = 1.0 + pr.x_rise * xi
            u = end.u * math.sqrt(xi) if xi > 0 else 1e-6
            m = 1.0 / (1.0 + xi * xi * (1.0 - end.m_emission) / end.m_emission)
            h = h0 + (end.h - h0) * xi
            z_c = min(self.height + pr.h_rise * math.sqrt(max(1.0 - (xi - 1.0) ** 2, 0.0)),
                      atm.h_mix - 0.5 * h)
            B = B_s + (end.b_half - B_s) * xi
            b = 0.9 * B_s + (end.b_shape - 0.9 * B_s) * xi
            cv = (atm.mw_moist * m
                  / (self.substance.mw + (atm.mw_moist - self.substance.mw) * m))
            rows.append(dict(
                x=x, t=2.0 * (x - 1.0) / max(end.u, 1e-12), mode=int(Mode.PLUME),
                h=h, z_c=z_c, b_half=B, b_shape=b,
                beta=math.sqrt(max(B * B - b * b, 0.0)) / SQRT3,
                b_half_x=1.0, b_shape_x=1.0, beta_x=0.0,
                sigma_z=(0.5 * h / SQRT3 if z_c > 0.5 * h
                         else max(h - z_c, 0.0) / SQRT3),
                u=u, T=self.T, rho=rho_s, cp=self.cp_source(),
                mass_frac=m, vol_frac=cv,
                v_g=0.0, u_g=0.0, w_c=w_s * (end.u - u) / max(end.u, 1e-12),
                u_ambient_mean=atm.wind_speed(max(z_c, 1e-3)),
                mass_frac_dry_air=(1.0 - m) * (1.0 - atm.mass_frac_water),
                mass_frac_water=(1.0 - m) * atm.mass_frac_water,
                mass_frac_water_vapour=(1.0 - m) * atm.mass_frac_water,
                mass_frac_emission_vapour=m * self.vapour_fraction,
                mass_frac_water_liquid=0.0,
                mass_frac_emission_liquid=m * self.liquid_fraction,
                w_entrain=math.nan, v_entrain=math.nan, v_x_entrain=math.nan,
                u_star_cloud=math.nan, phi_h=math.nan, inv_L_cloud=math.nan,
                f_u=0.0, f_v=0.0, f_w=0.0, f_t=0.0,
                mass_in_cloud=0.5 * self.rate * (2.0 * (x - 1.0)
                                                 / max(end.u, 1e-12)),
                R_flux=rho_s * u * B * h,
            ))
        return rows
