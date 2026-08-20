"""
Two-species local thermodynamic equilibrium
===========================================

Solves the coupled temperature-density-phase problem at one integration step
(section 2.5.4, EQ 40-44).  Two species may condense: the released material
and the ambient water vapour drawn into the cloud.

The problem
-----------
Given the cloud's total composition and its enthalpy content, find the
temperature `T` such that

    T = [ E + dHw0 (m_wvt - m_wv(T)) + dHe0 (m_evt - m_ev(T)) ] / cp(T)

where `m_wv(T)` and `m_ev(T)` are the equilibrium vapour fractions and
`m_wvt`, `m_evt` are the fractions transported from the previous step under
frozen composition.  Every term on the right depends on T, and the latent
heats are large, so the coupling is strong: condensing 1 % of a cryogen can
move the temperature by tens of kelvin.

Equilibrium condition (EQ 42-44)
---------------------------------
The vapour mole fraction of a species equals its saturation ratio, capped by
the amount actually present:

    x_v = min( P_sat(T) / P_a ,  x_total )

In mass terms, with `alpha_s = M_s (m_da/M_a + m_wv/M_w)` the "other gases"
term, the unsaturated branch is `m_ev = alpha_s f / (1 - f)`.  As `f -> 1`
this diverges, so the implementation switches to the saturated branch — all
of the species is vapour — before that happens, and linearises about the
switch-over temperature to keep the derivative finite.

Why not the reference implementation's loop
--------------------------------------------
`SLAB_thermo` runs a fixed 15 Newton steps inside a 2-pass outer loop that
toggles the all-vapour flags `ie`/`iw`.  It has three weaknesses:

* no convergence test on exit — a non-converged step is silently accepted;
* the flag toggling can cycle, and the outer loop simply gives up after two
  passes, again silently;
* Newton is unguarded, so a bad derivative near the saturation switch can
  throw the iterate to a non-physical temperature.

Here the same equations are solved with a **safeguarded Newton**: the root is
bracketed first, Newton steps are accepted only while they stay inside the
bracket, and bisection is used otherwise.  Convergence is guaranteed and
failure is raised rather than ignored.  The flags are not toggled at all —
capping the vapour fraction at the total makes them unnecessary, which also
removes the cycling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..coefficients import PHYS, Coefficients, COEFFS
from .base import ThermoBackend

__all__ = ["Mixture", "Equilibrium", "EquilibriumError", "solve_equilibrium"]


class EquilibriumError(RuntimeError):
    """The equilibrium solve did not converge."""


# ===========================================================================
@dataclass(frozen=True)
class Mixture:
    """Bulk composition of the cloud at one step, before phase splitting."""

    m_emission: float       #: total released material (EQ 1a/15a)
    m_water: float          #: total water (EQ 40a)
    m_dry_air: float        #: dry air (EQ 40b)
    m_ev_transported: float #: emission vapour carried over (EQ 40d, `cmevt`)
    m_wv_transported: float #: water vapour carried over (EQ 40c, `cmwvt`)
    enthalpy: float         #: E in EQ 41 [J/kg] (JS `etrn`)
    m_ev_prescribed: float | None = None
    """
    Vapour fraction imposed instead of solved from saturation.

    Set by the finite-rate evaporation model, which determines the phase
    split from droplet kinetics rather than from local equilibrium.  When
    given, the solver still finds T from the energy balance but takes the
    split as data — evaporation is rate-limited, so equilibrium is an upper
    bound on the vapour, not the answer.
    """

    def __post_init__(self):
        tot = self.m_emission + self.m_water + self.m_dry_air
        if not math.isclose(tot, 1.0, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError(f"mass fractions sum to {tot}, not 1")
        for n in ("m_emission", "m_water", "m_dry_air"):
            if getattr(self, n) < -1e-12:
                raise ValueError(f"{n} is negative")


@dataclass(frozen=True)
class Equilibrium:
    """Result of the phase-equilibrium solve at one step."""

    T: float                #: temperature [K]
    rho: float              #: density [kg/m^3], EQ 10
    cp: float               #: mixture specific heat [J/(kg K)], EQ 41
    m_ev: float             #: emission as vapour (EQ 40d)
    m_el: float             #: emission as droplets (EQ 40f)
    m_wv: float             #: water as vapour (EQ 40c)
    m_wl: float             #: water as droplets (EQ 40e)
    alpha: float            #: EQ 10 molar term
    gamma: float            #: EQ 10 droplet-volume term
    iterations: int
    saturated_emission: bool  #: True if all emission is vapour
    saturated_water: bool

    @property
    def has_droplets(self) -> bool:
        return self.m_el > 0.0 or self.m_wl > 0.0


# ===========================================================================
def _vapour_fraction(
    f_sat: float, df_sat: float, m_total: float, m_other_molar: float,
    mw: float, m_carried: float = 0.0,
) -> tuple[float, float, bool]:
    """
    Equilibrium vapour mass fraction of one species and its dT-derivative.

    Parameters
    ----------
    f_sat, df_sat : saturation ratio P_sat/P_a and its temperature derivative.
    m_total : total mass fraction of the species present.
    m_other_molar : sum over the *other* gaseous species of m_i / M_i.
    mw : molecular weight of this species [kg/mol].

    Returns
    -------
    (m_vapour, d m_vapour / dT, saturated)
    """
    if m_total <= 0.0:
        return 0.0, 0.0, True

    alpha_s = mw * m_other_molar                 # JS `alfs` / `alfw`
    if alpha_s <= 0.0:
        # No diluent: the species is pure and saturated, so its partial
        # pressure equals the ambient pressure at the boiling point and the
        # vapour/liquid split is fixed by the enthalpy rather than by T.
        # Declaring it all vapour here would make the solver absorb the full
        # latent heat of the droplets and drive T negative.  Returning the
        # amount already in the vapour phase leaves the split to the carry
        # term of EQ 40d, which is what physically determines it.
        return m_carried, 0.0, False

    # mole fraction if everything were vapour
    x_total = m_total / (m_total + alpha_s)

    if f_sat >= x_total:                         # unsaturated: all vapour
        return m_total, 0.0, True

    m_v = alpha_s * f_sat / (1.0 - f_sat)
    dm_v = alpha_s * df_sat / (1.0 - f_sat) ** 2
    if m_v >= m_total:                           # numerical safety
        return m_total, 0.0, True
    return m_v, dm_v, False


def _state(
    T: float,
    mix: Mixture,
    emission: ThermoBackend,
    water: ThermoBackend,
    mw_emission: float,
) -> dict:
    """Phase split, specific heat and residual temperature at a trial T."""
    fe = emission.saturation_ratio(T)
    dfe = emission.d_saturation_ratio(T)
    fw = water.saturation_ratio(T)
    dfw = water.d_saturation_ratio(T)

    # The two species are coupled through the diluent term; one pass of
    # substitution is what the reference does and is enough because water is
    # a trace component whenever the emission is not, and vice versa.
    if mix.m_ev_prescribed is None:
        m_ev, dm_ev, sat_e = _vapour_fraction(
            fe, dfe, mix.m_emission,
            mix.m_dry_air / PHYS.MW_AIR + mix.m_wv_transported / PHYS.MW_WATER,
            mw_emission, m_carried=mix.m_ev_transported,
        )
    else:
        m_ev = min(max(mix.m_ev_prescribed, 0.0), mix.m_emission)
        dm_ev, sat_e = 0.0, m_ev >= mix.m_emission - 1e-15
    m_wv, dm_wv, sat_w = _vapour_fraction(
        fw, dfw, mix.m_water,
        mix.m_dry_air / PHYS.MW_AIR + m_ev / mw_emission,
        PHYS.MW_WATER, m_carried=mix.m_wv_transported,
    )

    m_el = mix.m_emission - m_ev
    m_wl = mix.m_water - m_wv

    cp = (
        mix.m_dry_air * PHYS.CP_AIR
        + m_wv * water.cp_vapour(T) + m_wl * water.cp_liquid(T)
        + m_ev * emission.cp_vapour(T) + m_el * emission.cp_liquid(T)
    )

    dhe0 = emission.dh_vap_datum(T)
    dhw0 = water.dh_vap_datum(T)
    T_new = (
        mix.enthalpy
        + dhw0 * (mix.m_wv_transported - m_wv)
        + dhe0 * (mix.m_ev_transported - m_ev)
    ) / cp

    # d(residual)/dT for Newton, JS `gfncp`
    dprime = 1.0 + (
        water.dh_vap(T_new) * dm_wv + emission.dh_vap(T_new) * dm_ev
    ) / cp

    return dict(T_new=T_new, residual=T - T_new, dresidual=dprime, cp=cp,
                m_ev=m_ev, m_el=m_el, m_wv=m_wv, m_wl=m_wl,
                sat_e=sat_e, sat_w=sat_w)


def solve_equilibrium(
    mix: Mixture,
    emission: ThermoBackend,
    water: ThermoBackend,
    *,
    T_ambient: float,
    rho_ambient: float,
    mw_ambient_moist: float,
    T_guess: float | None = None,
    coeffs: Coefficients = COEFFS,
    T_bounds: tuple[float, float] = (1.0, 2000.0),
) -> Equilibrium:
    """
    Solve EQ 40-44 for temperature, phase split and density at one step.

    Parameters
    ----------
    mix : bulk composition and enthalpy content at this step.
    emission, water : property backends for the two condensable species.
    T_ambient, rho_ambient, mw_ambient_moist : ambient reference state used by
        the equation of state (EQ 10).
    T_guess : starting temperature; the previous step's value is the natural
        choice and roughly halves the iteration count.
    T_bounds : hard bracket.  The physical bounds are the all-condensed and
        all-vapour temperatures, which lie inside this by construction.

    Raises
    ------
    EquilibriumError
        If the residual does not change sign inside `T_bounds`, or Newton
        plus bisection fail to converge within the iteration cap.  The
        reference implementation returns silently in both cases.
    """
    mw_e = emission.substance.mw
    f = lambda T: _state(T, mix, emission, water, mw_e)   # noqa: E731

    lo, hi = T_bounds
    r_lo, r_hi = f(lo)["residual"], f(hi)["residual"]
    if r_lo * r_hi > 0.0:
        raise EquilibriumError(
            f"residual does not bracket a root on [{lo}, {hi}] "
            f"(f={r_lo:.3e}, {r_hi:.3e}); enthalpy or composition unphysical?"
        )
    if r_lo > 0.0:                       # orient so that f(lo) < 0 < f(hi)
        lo, hi = hi, lo

    # Safeguarded Newton ("rtsafe"): take a Newton step when it stays inside
    # the bracket and is reducing the interval fast enough, bisect otherwise.
    # Written so that landing exactly on the root, or on a bracket endpoint,
    # terminates instead of being rejected.
    T = T_guess if T_guess is not None and min(lo, hi) < T_guess < max(lo, hi) \
        else 0.5 * (lo + hi)
    st = f(T)
    dx_old = abs(hi - lo)
    dx = dx_old
    tol = coeffs.newton_tol
    max_iter = coeffs.newton_max_iter * 4

    for n_iter in range(1, max_iter + 1):
        r, d = st["residual"], st["dresidual"]

        bad_newton = (
            abs(d) < 1e-30
            or ((T - hi) * d - r) * ((T - lo) * d - r) > 0.0   # step leaves bracket
            or abs(2.0 * r) > abs(dx_old * d)                  # too slow
        )
        dx_old = dx
        if bad_newton:
            dx = 0.5 * (hi - lo)
            T_new = lo + dx
        else:
            dx = r / d
            T_new = T - dx

        if abs(dx) < tol or r == 0.0:
            T = T_new if r != 0.0 else T
            st = f(T)
            break

        T = T_new
        st = f(T)
        if st["residual"] < 0.0:
            lo = T
        else:
            hi = T
    else:
        raise EquilibriumError(
            f"no convergence after {max_iter} iterations "
            f"(last step {dx:.3e} K, residual {st['residual']:.3e})"
        )

    # -- equation of state, EQ 10 --------------------------------------
    alpha = mw_ambient_moist * (
        mix.m_dry_air / PHYS.MW_AIR
        + st["m_wv"] / PHYS.MW_WATER
        + st["m_ev"] / mw_e
    )
    gamma = (
        rho_ambient / water.rho_liquid(T) * st["m_wl"]
        + rho_ambient / emission.rho_liquid(T) * st["m_el"]
    )
    rho = rho_ambient * T_ambient / (alpha * T + gamma * T_ambient)

    return Equilibrium(
        T=T, rho=rho, cp=st["cp"],
        m_ev=st["m_ev"], m_el=st["m_el"], m_wv=st["m_wv"], m_wl=st["m_wl"],
        alpha=alpha, gamma=gamma, iterations=n_iter,
        saturated_emission=st["sat_e"], saturated_water=st["sat_w"],
    )
