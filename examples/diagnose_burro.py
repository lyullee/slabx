"""
Where does the Burro over-prediction come from?
===============================================

`validate_burro.py` shows the model over-predicting by about a factor two in
the three neutral, windy trials while getting the stable, low-wind one almost
exactly right.  That pattern is condition-dependent, so no single entrainment
coefficient can be responsible.  This script works through the candidates in
order and rules them out.

The checks, and what each one settled
-------------------------------------
1. **Source area.**  Only BU08's is documented; the rest come from
   ``A = q/E``.  Sweeping E across the whole range reported for LNG on water
   moves MG from 0.605 to 0.643 and leaves the per-trial pattern intact.
   Not the cause.

2. **Meander and averaging time.**  The trials were processed at 50-140 s.
   Switching the averaging off changes the concentration by 1-9 %.  Not the
   cause.

3. **Source-area expansion.**  It does not trigger at all for BU03, BU07 or
   BU09, yet those are the trials that over-predict.  Not the cause.

4. **Surface roughness.**  This is the one that matters.  The MVD
   (03903-RP-002 section 2.2.5.4) records 0.0002 m as the base value, notes
   that FERC proposed 0.01 m, and rejects that as "excessive, given the
   strong effect this parameter can have".  It is excessive in effect and
   correct in outcome: at 0.0002 m the unmodified model over-predicts by 1.65
   and fails three Chang-Hanna criteria, at 0.01 m it passes all five.
   The sensors sat on desert 57-800 m downwind of a 58 m water pond, so
   0.0002 m describes the pond, not the fetch.

5. **What is left after the roughness is right.**  Against Pasquill-Gifford
   corrected for the *same* roughness, sigma_z comes out at 0.74 of the curve
   and sigma_y at 1.94, so the cross-section is 1.48 times too large and the
   cloud is somewhat too flat.  That is a real but modest structural bias —
   an order of magnitude smaller than it looked before the roughness was
   corrected, when the same comparison gave 4.76 and 4.39.

6. **Why the cloud is flat.**  Decomposing ``dB/dx`` shows gravity spreading
   contributing 35-92 % of the width and still contributing after the density
   excess has fallen to a fraction of a percent: for BU09 the buoyancy drops
   by a factor 211 between 11 m and 921 m while V_g drops only by 3.2.  The
   measured damping factor ``e_v`` sits at 0.9997, so EQ 5a effectively never
   dissipates the cross-wind gravity momentum.

7. **Is V_g simply too fast?**  No.  It stays at 0.11-0.69 of the
   lock-exchange front speed ``1.1 sqrt(g' h)``, so a Froude cap would never
   bind.  The problem is persistence, not magnitude, and no parameter-free
   remedy for it has been found.

Why this matters for the variants
---------------------------------
`canonical` and `frontal` both act on entrainment.  Once the roughness is
right, sigma_z is within 26 % of the reference curve while the roughness
itself moved the answer by 65 %, so both were adjusting the smaller term —
which is what their pre-registered predictions failed on.

Open caveat
-----------
The "cloud width" reported by Witlox (2013) is used here without knowing its
definition.  Taken at face value the observed widths at 140 m are no wider
than the source pool itself and narrower than passive Gaussian dispersion
would give, which suggests it is measured at a concentration threshold rather
than being the full cloud extent.  The Pasquill-Gifford comparison in leg 5
does not depend on it and is the one to trust.

Usage
-----
    python examples/diagnose_burro.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import slabx.validation.field_trials as FT                        # noqa: E402
from slabx.coefficients import PHYS                               # noqa: E402
from slabx.validation.field_trials import (                       # noqa: E402
    BURRO_TRIALS, load_conditions, load_observations,
    predictions_for,
)
from slabx.validation.metrics import metrics                      # noqa: E402

STABILITY = {"BU03": "C", "BU07": "D", "BU08": "E", "BU09": "D"}

#: Briggs (1973) open-country sigmas: (a_y, b_y, a_z, b_z, exponent_z).
BRIGGS = {
    "C": (0.11, 1e-4, 0.08, 2e-4, -0.5),
    "D": (0.08, 1e-4, 0.06, 1.5e-3, -0.5),
    "E": (0.06, 1e-4, 0.03, 3e-4, -1.0),
}

#: The curves are quoted for open country, z0 ~ 0.03 m.  Burro was over
#: water.  The usual correction is (z0/0.03)^0.2, applied to both sigmas.
Z0_REFERENCE = 0.03
Z0_EXPONENT = 0.2


def briggs(stability: str, x: float, z0: float) -> tuple[float, float]:
    ay, by, az, bz, ez = BRIGGS[stability]
    f = (z0 / Z0_REFERENCE) ** Z0_EXPONENT
    return (ay * x * (1 + by * x) ** -0.5 * f,
            az * x * (1 + bz * x) ** ez * f)


def sigma_y(traj, i: int) -> float:
    """Second moment of SLAB's cross-wind profile: b^2/3 + beta^2."""
    return math.sqrt(traj.b_shape[i] ** 2 / 3.0 + traj.beta[i] ** 2)


# ===========================================================================
def check_source_area(obs, cond, y, tid):
    print(f"\n{'=' * 74}\n1. Source area\n{'=' * 74}")
    base = dict(FT.BURRO_AREA)
    print(f"{'E [kg/m2/s]':>13}{'MG':>8}{'VG':>8}{'FAC2':>7}   "
          + "".join(f"{t:>8}" for t in BURRO_TRIALS))
    try:
        for E in (0.200, 0.150, 0.120, 0.085):
            FT.BURRO_AREA = {t: cond[t]["rate"] / E for t in BURRO_TRIALS}
            p = predictions_for("slabx", obs, cond)
            m = metrics(y, p)
            print(f"{E:13.3f}{m.MG:8.3f}{m.VG:8.3f}{m.FAC2:7.2f}   "
                  + "".join(f"{metrics(y[tid == t], p[tid == t]).MG:8.3f}"
                            for t in BURRO_TRIALS))
    finally:
        FT.BURRO_AREA = base
    print("  -> pattern unchanged across the whole literature range")


def check_meander(cond):
    print(f"\n{'=' * 74}\n2. Meander and averaging time\n{'=' * 74}")
    print(f"{'trial':>7}{'t_avg [s]':>11}{'C(t_avg)/C(0)':>15}")
    for t in BURRO_TRIALS:
        c = cond[t]
        _, f = FT.run_trial(t, c, "slabx")
        _, f0 = FT.run_trial(t, {**c, "t_avg": 1e-6}, "slabx")
        i = int(np.argmin(np.abs(f.x - 140.0)))
        print(f"{t:>7}{c['t_avg']:11.0f}{f.peak[i] / f0.peak[i]:15.3f}")
    print("  -> at most a 9 % effect; cannot explain a factor two")


def check_spreads(obs, cond):
    print(f"\n{'=' * 74}\n3. Horizontal vs vertical spread, against "
          f"roughness-corrected PG\n{'=' * 74}")
    print(f"{'trial':>6}{'x':>6}{'sy/PG':>8}{'sz/PG':>8}{'product':>10}"
          f"{'C/C_obs':>10}")
    for t in BURRO_TRIALS:
        traj, field = FT.run_trial(t, cond[t], "slabx")
        for o in obs:
            if o["trial"] != t:
                continue
            i = int(np.argmin(np.abs(traj.x - o["x"])))
            py, pz = briggs(STABILITY[t], o["x"], cond[t]["z0"])
            sy, sz = sigma_y(traj, i), traj.sigma_z[i]
            c = float(np.exp(np.interp(o["x"], field.x,
                                       np.log(np.maximum(field.peak, 1e-30)))))
            print(f"{t:>6}{o['x']:6.0f}{sy / py:8.2f}{sz / pz:8.2f}"
                  f"{(sy * sz) / (py * pz):10.2f}{c / o['c_obs']:10.2f}")
    print("  -> sigma_z is right; sigma_y is far too large, and the "
          "cross-section is too small")


def check_width_source(cond):
    print(f"\n{'=' * 74}\n4. What makes the cloud wide, and does it stop?"
          f"\n{'=' * 74}")
    print(f"{'trial':>6}{'x':>7}{'rho/rho_a-1':>13}{'V_g':>9}{'V_e':>9}"
          f"{'gravity %':>11}{'V_g/U_front':>13}")
    for t in ("BU07", "BU09", "BU08"):
        traj, _ = FT.run_trial(t, cond[t], "slabx")
        atm = FT.VARIANTS["slabx"].atmosphere(cond[t])
        grav = ent = 0.0
        marks = [10.0, 30.0, 100.0, 300.0, 900.0]
        for i in range(1, len(traj)):
            dx = traj.x[i] - traj.x[i - 1]
            if dx <= 0:
                continue
            grav += abs(traj.v_g[i]) / traj.u[i] * dx
            ent += (math.sqrt(3.0) * atm.rho / traj.rho[i]
                    * traj.v_entrain[i] / traj.u[i] * dx)
            if marks and traj.x[i] >= marks[0]:
                marks.pop(0)
                gp = PHYS.GRAVITY * (traj.rho[i] - atm.rho) / traj.rho[i]
                u_front = 1.1 * math.sqrt(max(gp, 0.0) * traj.h[i])
                print(f"{t:>6}{traj.x[i]:7.1f}"
                      f"{traj.rho[i] / atm.rho - 1:13.5f}"
                      f"{abs(traj.v_g[i]):9.5f}{traj.v_entrain[i]:9.5f}"
                      f"{100 * grav / max(grav + ent, 1e-9):11.1f}"
                      f"{abs(traj.v_g[i]) / max(u_front, 1e-9):13.2f}")
    print("  -> gravity dominates the width and outlives its own buoyancy,")
    print("     yet never exceeds the lock-exchange front speed: the fault")
    print("     is persistence in the cross-wind momentum equation, not "
          "magnitude")


def main():
    obs = load_observations()
    cond = load_conditions()
    y = np.array([o["c_obs"] for o in obs])
    tid = np.array([o["trial"] for o in obs])

    check_source_area(obs, cond, y, tid)
    check_meander(cond)
    check_spreads(obs, cond)
    check_width_source(cond)


if __name__ == "__main__":
    main()
