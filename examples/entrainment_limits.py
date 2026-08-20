"""
Where does SLAB's entrainment function sit?
===========================================

An entrainment closure for a dense cloud has to reduce correctly to four
canonical flows, each of which has been measured **independently of any
dispersion trial**:

    passive dispersion       lambda_1 = 0.75   Sutton (1953), analytic
    stratified shear flow    lambda_2 = 2.5    Kato & Phillips (1969), tank
    weak free convection     lambda_3 = 0.37   Farmer (1975) / Bo Pedersen (1980)
    strong free convection   lambda_4 = 0.25   Deardorff et al. (1980), tank

collected by Nielsen & Jensen (Risoe, "A versatile entrainment function for
dense-gas dispersion"), who fit them with a single parameter-free form

    u_e / sqrt(e) = 1 / (0.25 + 3.3 Ri),   e^(3/2) = u_*^3 + 0.1 w_*^3
    Ri = g (rho - rho_a) h / (rho e)

Why this matters here
---------------------
It settles whether the gravity-front entrainment coefficient added to
`Coefficients.alpha_front` has to be fitted to dispersion data.  If SLAB's
closure already matches the canonical limits, the term is redundant; if it
misses them, the limits themselves say by how much — with no circularity
between calibration and validation.

Nielsen's own table lists an *early* SLAB (Morgan et al., 1983) whose
turbulence scale included the spreading front velocity ``u_f ~ sqrt(g' h)``.
Ermak (1990) replaced that with an ambient-stability formulation, which is
the version implemented here: `friction()` builds

    U_*^2 = U_mg*^2 + U_mh*^2 + U_t^2

in which the gravity velocity appears only through *ground friction*,
``(c_f V_g / 2)^2`` with ``c_f ~ 0.05``, i.e. suppressed by some three orders
of magnitude relative to ``g' h``.  So the term really was dropped.

Reading the output
------------------
Definitions differ between models — Nielsen warns explicitly that these
coefficients depend on the box-height convention.  His height is twice the
centre of gravity; SLAB's is ``h = sqrt(3) sigma``.  For a half-Gaussian the
two differ by 8.5 %, and lambda_1 scales roughly linearly with the
convention, so the reference values are rescaled here before comparison.
A residual factor of order 10 % is not meaningful; a factor of two is.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slabx.coefficients import COEFFS, PHYS                       # noqa: E402
from slabx.submodels.atmosphere import Atmosphere                 # noqa: E402
from slabx.submodels.entrainment import (                         # noqa: E402
    CloudLocal, entrainment, friction,
)

SQRT3 = math.sqrt(3.0)

#: Nielsen & Jensen, Table 2.  Measured in canonical flows, not in dispersion
#: trials — which is the whole point.
REFERENCE = {"lambda_1": 0.75, "lambda_2": 2.5, "lambda_3": 0.37, "lambda_4": 0.25}

#: h_SLAB / h_Nielsen for a half-Gaussian: sqrt(3) sigma vs 2 sigma sqrt(2/pi).
H_CONVENTION = math.sqrt(3.0) / (2.0 * math.sqrt(2.0 / math.pi))


def ideal_atmosphere(u_ref=5.0, z0=0.01):
    """Neutral, so the ambient contributes no stability of its own."""
    return Atmosphere(u_ref=u_ref, z_ref=10.0, T=290.0, rh=0.0, z0=z0,
                      stability="D")


def probe(atm, *, h, rho_ratio, dT_ground=0.0, u=None):
    """
    Vertical entrainment for an idealised cloud.

    The cloud is set to travel with the wind (no slip, so no top drag) and to
    have no gravity flow, isolating the balance between the ground-friction
    turbulence scale and the density stratification.  `dT_ground` switches on
    the convective term by making the cloud colder than the ground.
    """
    u_bar = atm.mean_wind_over(h, n=200)
    cl = CloudLocal(
        h=h, b_half=200.0, b_half_x=1.0, z_c=0.0,
        u=u if u is not None else u_bar,
        T=atm.T - dT_ground, rho=rho_ratio * atm.rho, cp=1010.0,
    )
    fr = friction(cl, atm, u_bar_ambient=u_bar)
    en = entrainment(cl, atm, fr, u_bar_ambient=u_bar)
    u_e = SQRT3 * en.w                       # EQ 35a applies sqrt(3) in EQ 2a
    Ri = (PHYS.GRAVITY * (cl.rho - atm.rho) * h
          / (cl.rho * max(fr.u_star**2, 1e-30)))
    return dict(u_e=u_e, u_star=fr.u_star, Ri=Ri, phi_h=fr.phi_h,
                u_t=math.sqrt(fr.u_t_sq), u_mg=fr.u_mg_star)


# ===========================================================================
def shear_branch(atm, h=2.0):
    """
    lambda_1 and lambda_2: no convection, so the only turbulence is shear.

    lambda_1 = u_e/u_*   as Ri -> 0
    lambda_2 = u_e Ri/u_* as Ri -> infinity
    """
    print(f"\n{'=' * 74}\nShear branch (isothermal cloud, h = {h} m)\n{'=' * 74}")
    print(f"{'rho/rho_a':>11}{'Ri':>11}{'u_e/u_*':>11}{'u_e Ri/u_*':>13}"
          f"{'phi_h':>9}")
    rows = []
    for r in (1.0 + 1e-9, 1.0001, 1.001, 1.01, 1.05, 1.2, 1.5, 2.0, 4.0):
        p = probe(atm, h=h, rho_ratio=r)
        ratio = p["u_e"] / p["u_star"]
        rows.append((p["Ri"], ratio))
        print(f"{r:11.4f}{p['Ri']:11.4g}{ratio:11.4f}"
              f"{ratio * p['Ri']:13.4f}{p['phi_h']:9.3f}")
    lam1 = rows[0][1]
    lam2 = rows[-1][0] * rows[-1][1]
    return lam1, lam2


#: SLAB's thermal velocity is not the convective velocity scale.  Its
#: definition (EQ 35e) is ``U_t^3 = C_t g dT V_H h / T``, while the standard
#: scale is ``w_*^3 = g h phi / (rho cp T)`` with the heat flux
#: ``phi = rho V_H cp dT``.  Substituting gives ``w_*^3 = g h V_H dT / T``,
#: so the two differ by exactly the coefficient C_t:
#:
#:     U_t = C_t^(1/3) w_* = 0.519 w_*
#:
#: Comparing ``u_e/U_t`` against a measured ``u_e/w_*`` would therefore
#: overstate SLAB's convective entrainment by a factor 1/0.519 = 1.93.
W_STAR_FROM_U_T = COEFFS.c_thermal ** (-1.0 / 3.0)


def convective_branch(atm, h=2.0, dT=60.0):
    """
    lambda_3 and lambda_4: convection from the ground with weak mean shear.

    The mean shear cannot be switched off in SLAB — the ground-friction term
    U_mg* is always present — so the pure-convection limits are approached
    rather than reached.  The wind is set as low as the surface-layer
    formulation allows and the residual shear contribution is reported, so
    the reader can see how far from the limit these numbers are.
    """
    print(f"\n{'=' * 74}\nConvective branch (cold cloud over warm ground, "
          f"h = {h} m, dT = {dT:.0f} K)\n{'=' * 74}")
    print(f"{'rho/rho_a':>11}{'Ri_w':>11}{'u_e/w_*':>11}{'u_e Ri_w/w_*':>14}"
          f"{'w_*/u_*':>10}")
    rows = []
    for r in (1.0 + 1e-9, 1.02, 1.1, 1.4, 2.0, 4.0, 10.0):
        p = probe(atm, h=h, rho_ratio=r, dT_ground=dT)
        w_star = p["u_t"] * W_STAR_FROM_U_T
        if w_star <= 0:
            continue
        Ri_w = PHYS.GRAVITY * (r - 1.0) / r * h / w_star**2
        ratio = p["u_e"] / w_star
        rows.append((Ri_w, ratio))
        print(f"{r:11.4f}{Ri_w:11.4g}{ratio:11.4f}{ratio * Ri_w:14.4f}"
              f"{w_star / p['u_mg']:10.2f}")
    lam3 = rows[0][1]
    lam4 = rows[-1][0] * rows[-1][1]
    return lam3, lam4


def height_dependence(atm):
    """
    SLAB's entrainment carries an explicit absolute-height factor.

    EQ 35a multiplies by ``U_r / U_a(h_top)`` with the reference height fixed
    at 4 m.  A similarity closure should depend on the cloud only through
    dimensionless groups, so this factor is a genuine departure: two clouds
    with identical Richardson number entrain at different rates purely
    because one is thinner.
    """
    print(f"\n{'=' * 74}\nAbsolute-height factor U_r / U_a(h)\n{'=' * 74}")
    print(f"{'h [m]':>8}{'U_r/U_a(h)':>13}{'u_e/u_* (Ri->0)':>18}")
    for h in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        f = (atm.dimensionless_profile(4.0)
             / atm.dimensionless_profile(h))
        p = probe(atm, h=h, rho_ratio=1.0 + 1e-9)
        print(f"{h:8.1f}{f:13.4f}{p['u_e'] / p['u_star']:18.4f}")


def main():
    atm = ideal_atmosphere()
    lam1, lam2 = shear_branch(atm)
    lam3, lam4 = convective_branch(ideal_atmosphere(u_ref=1.0))
    height_dependence(atm)

    print(f"\n{'=' * 74}\nAsymptotes vs canonical flows\n{'=' * 74}")
    print(f"{'limit':<26}{'SLAB':>10}{'measured':>12}{'rescaled':>11}"
          f"{'SLAB/ref':>11}")
    print("-" * 70)
    got = {"lambda_1": lam1, "lambda_2": lam2,
           "lambda_3": lam3, "lambda_4": lam4}
    names = {"lambda_1": "passive dispersion",
             "lambda_2": "stratified shear flow",
             "lambda_3": "weak free convection",
             "lambda_4": "strong free convection"}
    for k, v in got.items():
        ref = REFERENCE[k] * H_CONVENTION
        print(f"{names[k]:<26}{v:10.3f}{REFERENCE[k]:12.2f}{ref:11.3f}"
              f"{v / ref:11.2f}")
    print("\nrescaled = measured x h_SLAB/h_Nielsen = "
          f"x{H_CONVENTION:.3f} (box-height convention)")


if __name__ == "__main__":
    main()
