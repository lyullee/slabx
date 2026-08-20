"""
Droplet rainout
===============

SLAB has none.  Once liquid is present in the cloud — 81 % of the mass in
the manual's ammonia jet, 88 % in its chlorine stack — it stays suspended
until it evaporates, however large the droplets or however slowly the cloud
moves.  Real two-phase releases lose part of that liquid to the ground,
where it forms a pool that then evaporates on its own timescale.  The
HSE/ADMLC review names rainout as one of the principal gaps in source-term
modelling, and it is the largest piece of physics missing from SLAB.

Why this can be added without inventing coefficients
-----------------------------------------------------
The chain from release conditions to settling rate is standard and each link
is measured rather than fitted:

1. **Droplet size.**  Aerodynamic break-up limits the largest stable droplet
   to a critical Weber number,

       d = We_c sigma / (rho_g u_rel^2),      We_c ~ 12

   with `We_c` from shock-tube and cross-flow break-up experiments and
   `sigma` the real surface tension, which CoolProp supplies.  Nothing here
   is tuned to dispersion data.

2. **Terminal velocity.**  A drag balance on that droplet.  Stokes below
   Re ~ 1, Newton above, with the standard blend.

3. **Removal rate.**  A droplet starting at a representative height in the
   cloud reaches the ground after ``h / v_t``; over a distance ``dx`` the
   cloud travels ``dx/U``.  So

       dm_liquid/dx = - (v_t / (U h)) m_liquid

   which is the one modelling choice, and it is the same first-order form
   used by every integral model that carries rainout.

Everything else follows: the mass that leaves reduces the cloud's inventory
and its buoyancy, and is accounted to a pool.

The predictions, registered before the comparison
--------------------------------------------------
R1  **No effect on single-phase releases.**  Burro is pure methane vapour;
    the Burro statistics must not move at all.  This is the negative control
    and if it fails nothing else here can be believed.

R2  **Effect concentrates in the two-phase decks.**  The manual's ammonia
    jet (81 % liquid) and chlorine stack (88 % liquid) must both change, and
    the ammonia jet — slower, larger droplets — by more.

R3  **Direction: concentrations fall near the source and the cloud is less
    dense.**  Removing liquid removes mass and removes the droplet-volume
    term from EQ 10, so the cloud is lighter and disperses further per unit
    of remaining material.

R4  **Mass is conserved.**  Everything that leaves the cloud arrives in the
    pool: ``m_cloud + m_pool = m_released`` to integration accuracy.

R5  **The effect scales with droplet size.**  A release through a smaller
    orifice at higher velocity makes smaller droplets, which settle more
    slowly; the rained-out fraction must fall monotonically as the exit
    velocity rises.

R2 and R5 are the discriminating ones.  A change that alters the pure-vapour
cases, or that does not respond to droplet size, is not modelling rainout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..coefficients import PHYS

__all__ = ["WEBER_CRITICAL", "DropletState", "droplet_diameter",
           "terminal_velocity", "rainout_rate", "evaporation_rate",
           "PREDICTIONS"]

#: Critical Weber number for aerodynamic break-up.  Values between 10 and 20
#: appear in the literature depending on how "break-up" is defined; 12 is the
#: usual choice for the largest stable droplet and is what UDM uses.
WEBER_CRITICAL = 12.0

#: Drag coefficient in the Newton regime for a sphere.
C_DRAG_SPHERE = 0.44

PREDICTIONS = (
    ("R1", "no effect on a single-phase release (negative control)"),
    ("R2", "effect concentrates in two-phase decks, largest for the slower jet"),
    ("R3", "concentrations fall near the source; the cloud is less dense"),
    ("R4", "cloud mass plus pool mass equals released mass"),
    ("R5", "rained-out fraction falls monotonically with exit velocity"),
)


@dataclass(frozen=True)
class DropletState:
    """Droplet size and settling rate for one cloud state."""

    diameter: float       #: representative droplet diameter [m]
    terminal_velocity: float  #: settling velocity in still gas [m/s]
    reynolds: float       #: droplet Reynolds number, to show which regime
    regime: str           #: "stokes", "intermediate" or "newton"


def droplet_diameter(surface_tension: float, rho_gas: float,
                     u_relative: float, *, weber: float = WEBER_CRITICAL,
                     d_max: float = 0.01) -> float:
    """
    Largest droplet that survives aerodynamic break-up, EQ We = We_c.

    `u_relative` is the slip between droplet and gas — at the orifice this is
    the jet velocity, since the droplets are formed there.  The result is
    capped at `d_max` because the criterion has no upper bound as the slip
    goes to zero, and a centimetre droplet is not a droplet.
    """
    if u_relative <= 0.0 or rho_gas <= 0.0:
        return d_max
    return min(weber * surface_tension / (rho_gas * u_relative**2), d_max)


def terminal_velocity(diameter: float, rho_liquid: float, rho_gas: float,
                      mu_gas: float = 1.8e-5) -> DropletState:
    """
    Settling velocity from a drag balance, with the regime chosen by Re.

    Stokes is solved directly; above Re = 1 the Newton form is iterated,
    since the drag coefficient and the velocity depend on each other.
    """
    if diameter <= 0.0 or rho_liquid <= rho_gas:
        return DropletState(diameter, 0.0, 0.0, "none")

    g = PHYS.GRAVITY
    dr = rho_liquid - rho_gas

    v = dr * g * diameter**2 / (18.0 * mu_gas)          # Stokes
    re = rho_gas * v * diameter / mu_gas
    if re <= 1.0:
        return DropletState(diameter, v, re, "stokes")

    # Newton / intermediate: iterate on the standard correlation
    for _ in range(50):
        cd = (24.0 / re) * (1.0 + 0.15 * re**0.687) if re < 1000.0 \
            else C_DRAG_SPHERE
        v_new = math.sqrt(4.0 * dr * g * diameter / (3.0 * cd * rho_gas))
        if abs(v_new - v) < 1e-9 * max(v_new, 1.0):
            v = v_new
            break
        v = 0.5 * (v + v_new)
        re = rho_gas * v * diameter / mu_gas
    regime = "newton" if re >= 1000.0 else "intermediate"
    return DropletState(diameter, v, re, regime)


def rainout_rate(m_liquid: float, v_terminal: float, u_cloud: float,
                 h_cloud: float) -> float:
    """
    Fractional loss of liquid per unit down-wind distance [1/m].

    ``-(v_t / (U h))``: a droplet crosses the cloud depth in ``h/v_t`` and the
    cloud covers ``U`` metres in that time.  First order in the liquid load,
    so the liquid decays exponentially with distance while the settling rate
    holds.
    """
    if m_liquid <= 0.0 or v_terminal <= 0.0 or u_cloud <= 0.0 or h_cloud <= 0.0:
        return 0.0
    return v_terminal / (u_cloud * h_cloud)


# ===========================================================================
# finite-rate evaporation
# ===========================================================================
#: Binary diffusivity of the vapour in air [m^2/s].  Order 2e-5 for the
#: light gases of interest; the lifetime scales inversely with it.
DIFFUSIVITY = 2.0e-5

#: Schmidt number for a light vapour in air.
SCHMIDT = 0.7


def evaporation_rate(diameter: float, y_surface: float, y_bulk: float,
                     rho_gas: float, rho_liquid: float, *,
                     reynolds: float = 0.0, diffusivity: float = DIFFUSIVITY,
                     schmidt: float = SCHMIDT) -> float:
    """
    Rate of change of the *squared* droplet diameter, ``-d(d^2)/dt`` [m^2/s].

    Mass-transfer controlled, not heat-transfer controlled:

        d(d^2)/dt = - (4 rho_g D Sh / rho_l) ln(1 + B_M)
        B_M = (Y_s - Y_inf) / (1 - Y_s)
        Sh  = 2 + 0.6 Re^(1/2) Sc^(1/3)                Ranz & Marshall (1952)

    with `y_surface` the saturated vapour mass fraction at the droplet's own
    temperature and `y_bulk` the mass fraction in the surrounding gas.

    Why the driver is a concentration difference
    ---------------------------------------------
    The obvious first choice — heat conduction into the droplet, driven by
    ``T_ambient - T_cloud`` — was tried and falsified.  It gives a positive
    feedback with the wrong sign: fewer droplets evaporate, so less latent
    heat is absorbed, so the cloud stays warm, so the temperature difference
    to *ambient* shrinks, so evaporation slows further.  On Desert Tortoise
    it left droplets alive at 300 m where SMEDIS puts them gone at 51, and
    the cloud at 299 K where SMEDIS measured 205.

    The physical driver is that the gas around the droplet is unsaturated.
    The droplet sits near its wet-bulb temperature and evaporates because
    vapour diffuses away from its surface, so the rate follows the
    concentration difference and is largest exactly where the cloud is
    coldest and most dilute — the opposite feedback, and the stabilising one.

    Why this exists at all
    ----------------------
    SLAB assumes local thermodynamic equilibrium: droplets evaporate the
    instant the surrounding gas is unsaturated, however large they are.  That
    is the assumption this relaxes.  Evaporation is rate-limited and
    condensation is not — vapour finds existing droplets quickly — so the
    constraint is one-sided: the liquid load may exceed what equilibrium
    would leave, never fall below it.
    """
    if diameter <= 0.0 or rho_liquid <= 0.0 or rho_gas <= 0.0:
        return 0.0
    y_s = min(max(y_surface, 0.0), 0.999)
    b_m = (y_s - min(max(y_bulk, 0.0), 1.0)) / (1.0 - y_s)
    if b_m <= 0.0:
        return 0.0                       # saturated: no net evaporation
    sh = 2.0 + 0.6 * math.sqrt(max(reynolds, 0.0)) * schmidt ** (1.0 / 3.0)
    return (4.0 * rho_gas * diffusivity * sh / rho_liquid) * math.log1p(b_m)
