"""
Added mass against Hall & Walker (2000)
========================================

    python3 examples/added_mass_liftoff.py

The discriminating experiment for a term whose entire content is a width
dependence: Hall & Walker varied a ground-level area source **64-fold in
width at fixed buoyancy flux** and reported when the plume lifts off.

Source geometry is Table 1 of AEAT/NOIL/27328006/001 (URAHFREP WP7), the
"wide" series, with length x/L = 3.43 held and width y/L running 0.448 to
28.66. L = 6.7 cm and z0 = 0.03 L.

Their criteria, quoted at 15 L and 30 L downwind:

    F / (W u^3) ~ 0.01     onset: the concentration maximum leaves the ground
    F / (W u^3) ~ 0.035    lift-off: ground concentration falls to 10-20 %
                           of the maximum

with "considerable scatter" noted. `F` is the buoyancy flux `g' Q` with
`g' = g (rho_a - rho_s) / rho_a` and `Q` the volumetric release rate; `W` is
the cross-wind source width.

Why this matters
----------------
`slabx` carries `k = C_A (rho_a/rho)(B/h)` with `C_A = 2/pi`, the
potential-flow value for a disc moving broadside. Nothing is fitted. DRIFT's
authors, on this same data, report that the term "suppressed lift-off too
much".

Predictions were registered in `docs/PREREG_added_mass.md` before this ran.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slabx.coefficients import COEFFS                       # noqa: E402
from slabx.core.plume import run_dispersion                 # noqa: E402
from slabx.core.source import EvaporatingPool               # noqa: E402
from slabx.scope import ScopeWarning                        # noqa: E402
from slabx.submodels.atmosphere import Atmosphere           # noqa: E402
from slabx.thermo.base import (                             # noqa: E402
    LegacyThermo, Substance, water_backend,
)

L = 0.067                    # Hall & Walker reference length [m]
U_REF = 1.0                  # wind speed at that height [m/s]
X_LENGTH = 3.43              # source length x/L, held fixed
GRAVITY = 9.80665

#: The "wide" series: width varies 64-fold at fixed length.
WIDTHS = {"G": 0.448, "H": 1.19, "D": 3.43, "I": 7.16, "J": 14.33,
          "K": 28.66}

#: A light gas standing in for the buoyancy-conserving tracer.
HELIUM = Substance(name="Helium", mw=0.004003, cp_vapour=5193.0,
                   cp_liquid=5193.0, dh_vap=20900.0, T_boil=4.2,
                   rho_liquid=125.0)


def atmosphere() -> Atmosphere:
    return Atmosphere(u_ref=U_REF, z_ref=L, T=293.0, rh=50.0,
                      z0=0.03 * L, stability="D")


def rate_for(F_over_Wu3: float, width_over_L: float, atm: Atmosphere,
             ) -> tuple[float, float]:
    """
    Mass release rate giving the requested dimensionless buoyancy flux.

    ``F = g' Q`` with ``g' = g (rho_a - rho_s)/rho_a``; the source density is
    the pure gas at ambient temperature, so ``Q = m_dot / rho_s`` and

        m_dot = F rho_s / g'
    """
    width = width_over_L * L
    F = F_over_Wu3 * width * U_REF ** 3

    rho_s = HELIUM.mw * 101325.0 / (8.3145 * 293.0)
    g_prime = GRAVITY * (atm.rho - rho_s) / atm.rho
    return F * rho_s / g_prime, width


def lift_off_metrics(group: str, F_over_Wu3: float, *, added_mass: bool,
                     c_added: float | None = None) -> dict | None:
    """
    Run one source and report where the cloud sits at 15 L and 30 L.

    Hall & Walker define lift-off by the ground concentration falling to
    10-20 % of the maximum, which needs a vertical profile. The shallow-layer
    equivalent is the centre height relative to the cloud depth: ``z_c`` at
    or below zero is a grounded cloud, and ``z_c`` of order ``h`` is one that
    has left the ground.
    """
    atm = atmosphere()
    rate, width = rate_for(F_over_Wu3, WIDTHS[group], atm)
    area = width * X_LENGTH * L

    coeffs = COEFFS
    if c_added is not None:
        coeffs = COEFFS.perturb(c_added_mass=c_added, name="fit")

    src = EvaporatingPool(substance=HELIUM, rate=rate, area=area,
                          duration=10_000.0)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ScopeWarning)
            traj, _ = run_dispersion(
                src, atm, LegacyThermo(HELIUM), water_backend(),
                x_max=30.0 * L, n_puff_steps=40, coeffs=coeffs,
                added_mass=added_mass,
            )
    except Exception:                                        # noqa: BLE001
        return None

    out = {}
    for tag, x in (("15L", 15.0 * L), ("30L", 30.0 * L)):
        z_c = float(np.interp(x, traj.x, traj.z_c))
        h = float(np.interp(x, traj.x, traj.h))
        out[tag] = z_c / max(h, 1e-12)
    return out


def main() -> int:
    print(f"\n{'=' * 74}")
    print("Added mass against Hall & Walker (2000), 'wide' series")
    print("z_c/h at 30 L; a grounded cloud sits near zero")
    print(f"{'=' * 74}\n")

    for F in (0.010, 0.035, 0.100):
        print(f"F/(W u^3) = {F:.3f}"
              f"{'   (onset)' if F == 0.010 else ''}"
              f"{'   (lift-off)' if F == 0.035 else ''}")
        print(f"  {'group':>6}{'y/L':>8}{'no added mass':>16}"
              f"{'C_A = 2/pi':>14}{'ratio':>9}")
        for g, y in WIDTHS.items():
            a = lift_off_metrics(g, F, added_mass=False)
            b = lift_off_metrics(g, F, added_mass=True)
            if a is None or b is None:
                print(f"  {g:>6}{y:8.3f}{'failed':>16}")
                continue
            r = a["30L"] / b["30L"] if b["30L"] > 1e-12 else float("inf")
            print(f"  {g:>6}{y:8.3f}{a['30L']:16.4f}{b['30L']:14.4f}"
                  f"{r:9.1f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
