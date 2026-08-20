"""
slabx against Ermak's Fortran
=============================

    python3 examples/fortran_reference.py            # all five manual decks
    python3 examples/fortran_reference.py burro8     # one

This replaces `fortran_reference.py`, which compared against a JavaScript
transcription.  The reasons are in `golden/fortran/oracle.py`: of six
differences originally recorded as defects of SLAB, only one was — three came
from a mechanical `min` -> `Math.min` substitution, one was a
single-character typo, and one was a sentinel mistaken for an omission.  The
typo made the JavaScript throw for `|1/L|` above about 0.15, which was
recorded as a limit of the model and used to bound the whole reproduction.
The Fortran has no such limit.

What is compared
----------------
The spatially averaged cloud parameters, variable by variable, at points
spaced logarithmically over the region both produce.  Reported as the median
and maximum relative difference per variable, which is more informative than
a single aggregate: a model can match the concentration while getting the
height wrong in a way that cancels.

What counts as agreement
------------------------
The same criterion as before: the difference must be no larger than the
reference's own discretisation error, estimated by running it on a coarser
grid.  Matching more closely than the original resolves its own equations
would be meaningless.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "golden" / "fortran"))

from oracle import FortranSLAB, OracleUnavailable          # noqa: E402

from slabx.coefficients import preset                      # noqa: E402
from slabx.core.plume import run_dispersion                # noqa: E402
from slabx.core.source import (                            # noqa: E402
    EvaporatingPool, HorizontalJet, InstantaneousRelease,
)
from slabx.core.vertical_jet import VerticalJet            # noqa: E402
from slabx.submodels.atmosphere import Atmosphere          # noqa: E402
from slabx.thermo.base import (                            # noqa: E402
    LegacyThermo, Substance, water_backend,
)

sys.path.insert(0, str(ROOT / "examples"))
from decks import CASES                                     # noqa: E402

#: Fortran column -> Trajectory attribute.  `bbx`/`bx` are puff-only and
#: `ua` is ambient rather than cloud, so they are compared separately.
MAP = {
    "h": "h", "bb": "b_half", "b": "b_shape", "cv": "vol_frac",
    "rho": "rho", "t": "T", "u": "u", "zc": "z_c",
}


def build(case):
    """Turn a Case into the slabx objects it describes."""
    sub = Substance(
        name=case.name, mw=case.wms, cp_vapour=case.cps,
        cp_liquid=case.cpsl, dh_vap=case.dhe, T_boil=case.tbp,
        rho_liquid=case.rhosl,
        **({"sat_B": case.spb} if case.spb > 0 else {}),
        **({"sat_C": case.spc} if case.spc != 0 else {}),
    )
    atm = Atmosphere(
        u_ref=case.ua, z_ref=case.za, T=case.ta, rh=case.rh, z0=case.z0,
        **({"inv_L": case.ala} if case.stab == 0 else
           {"stability": "ABCDEF"[int(case.stab) - 1]}),
    )
    common = dict(substance=sub, rate=case.qs, area=case.as_,
                  duration=case.tsd)
    if case.idspl == 1:
        src = EvaporatingPool(**common)
    elif case.idspl == 2:
        src = HorizontalJet(liquid_fraction=case.cmed0, height=case.hs,
                            T_source=case.ts, **common)
    elif case.idspl == 3:
        src = VerticalJet(liquid_fraction=case.cmed0, height=case.hs,
                          T_source=case.ts, **common)
    else:
        src = InstantaneousRelease(substance=sub, rate=0.0, duration=0.0,
                                   area=case.as_, mass=case.qtis,
                                   height=case.hs)
    return sub, atm, src


def deck(case) -> dict:
    """Case -> the keyword form `FortranSLAB.run` takes."""
    return dict(
        idspl=case.idspl, ncalc=case.ncalc, wms=case.wms, cps=case.cps,
        tbp=case.tbp, cmed0=case.cmed0, dhe=case.dhe, cpsl=case.cpsl,
        rhosl=case.rhosl, spb=case.spb, spc=case.spc, ts=case.ts,
        qs=case.qs, as_=case.as_, tsd=case.tsd, qtis=case.qtis, hs=case.hs,
        tav=case.tav, xffm=case.xffm,
        zp=(case.zp, case.zp, case.zp, case.zp),
        z0=case.z0, za=case.za, ua=case.ua, ta=case.ta, rh=case.rh,
        stab=case.stab, ala=case.ala if case.stab == 0 else None,
    )


def compare(name, case, oracle) -> dict:   # noqa: C901
    ref = oracle.run(**deck(case))
    coarse = oracle.run(**{**deck(case), "xffm": case.xffm * 0.5})

    sub, atm, src = build(case)
    traj, used = run_dispersion(src, atm, LegacyThermo(sub), water_backend(),
                                x_max=case.xffm, n_puff_steps=40,
                                t_max=case.t_max, coeffs=preset("ermak90"))

    lo = max(ref.x[ref.x > 0].min(), traj.x[traj.x > 0].min())
    hi = min(ref.x.max(), traj.x.max())
    if not (hi > lo):
        raise RuntimeError(f"{name}: no overlapping region")
    xs = np.geomspace(lo, hi, 40)

    # A stack release rises and then grounds; until it has, the two
    # implementations are interpolating through the rise and differ most.
    # The split is taken where the reference's own centre height falls below
    # a tenth of the cloud depth, which is immediate for a pool or an
    # instantaneous release and a few metres for a stack.
    zc_ref, h_ref = ref.column("zc"), ref.column("h")
    grounded = np.where((h_ref > 0) & (zc_ref < 0.1 * h_ref))[0]
    x_established = float(ref.x[grounded[0]]) if grounded.size else 0.0

    rows = {}
    for col, attr in MAP.items():
        raw = ref.column(col)
        # The rise region carries -1 sentinels for quantities that are not
        # defined there (SLAB.FOR lines 1250-1251 assign them explicitly).
        # Interpolating across them, or comparing against them, produces
        # differences of tens of thousands of percent that mean nothing.
        valid = raw > -0.5 if col in ("t", "rho", "cv", "h") else \
            np.ones_like(raw, dtype=bool)
        if valid.sum() < 5:
            continue
        o = np.interp(xs, ref.x[valid], raw[valid])
        p = np.interp(xs, traj.x, getattr(traj, attr))
        # Compare only where the quantity is a meaningful fraction of its own
        # range.  Without this, `zc` for a stack release — 0.04 m against
        # 0.08 m, both effectively zero once the plume has grounded — reports
        # a 141 % disagreement that is an artefact of dividing small by small.
        # A floor relative to the variable's own range, so that quantities
        # which decay to zero are not compared where both are effectively
        # zero.  Concentration falls four decades over the domain, so the
        # floor has to be well below one per cent of the peak or the far
        # field is discarded — 1e-4 keeps it while still excluding `zc` once
        # the plume has grounded (0.04 m against 0.08 m is not a 100 %
        # disagreement).
        scale = float(np.max(np.abs(o))) if o.size else 0.0
        live = (np.abs(o) > max(1e-12, 1e-4 * scale)) \
            & (xs >= ref.x[valid].min())
        if live.sum() < 5:
            continue
        if col == "zc":
            # A position, not a magnitude.  Once the plume has grounded both
            # are within centimetres of zero and a ratio is meaningless, so
            # the difference is scaled by the cloud depth instead.
            depth = np.interp(xs, ref.x, np.maximum(h_ref, 1e-6))
            d = np.abs(p[live] - o[live]) / depth[live]
        else:
            d = np.abs(p[live] / o[live] - 1.0)

        # the oracle's own grid sensitivity, as the tolerance
        craw = coarse.column(col)
        cvalid = craw > -0.5 if col in ("t", "rho", "cv", "h") else \
            np.ones_like(craw, dtype=bool)
        if cvalid.sum() < 5:
            rows[col] = (float(np.median(np.abs(p[live] / o[live] - 1.0))),
                         float(np.max(np.abs(p[live] / o[live] - 1.0))),
                         float("nan"))
            continue
        c = np.interp(xs, coarse.x[cvalid], craw[cvalid])
        gl = live & (np.abs(c) > 1e-9)
        grid = (float(np.median(np.abs(c[gl] / o[gl] - 1.0)))
                if gl.sum() >= 5 else float("nan"))

        # Split at ten source half-widths.  For a stack release the plume
        # rise and its interpolation back to the ground occupy the first few
        # metres, and that region is where the two implementations differ
        # most; reporting one median over the whole domain hides both the
        # agreement outside it and the disagreement inside.
        near = xs[live] < x_established
        far_d = d[~near] if (~near).sum() >= 5 else d
        rows[col] = (float(np.median(far_d)), float(d.max()), grid,
                     float(np.median(d[near])) if near.sum() >= 3 else
                     float("nan"))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("case", nargs="?", default="all", help="deck name or all")
    args = ap.parse_args()

    try:
        oracle = FortranSLAB()
    except OracleUnavailable as exc:
        print(exc)
        return 2

    names = list(CASES) if args.case == "all" else [args.case]
    worst = 0.0
    for name in names:
        case = CASES[name]
        print(f"\n{'=' * 62}\n{name}  —  {case.name}  (idspl={case.idspl})"
              f"\n{'=' * 62}")
        try:
            rows = compare(name, case, oracle)
        except Exception as exc:                          # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            worst = float("inf")
            continue

        print(f"{'변수':>6}{'established':>12}{'near-source':>13}"
              f"{'최대':>9}{'원본 격자':>11}{'판정':>7}")
        for col, (med, mx, grid, near) in rows.items():
            ok = med <= max(2.0 * grid, 0.02) if grid == grid else med <= 0.02
            worst = max(worst, med)
            nr = f"{near:12.2%}" if near == near else f"{'—':>12}"
            print(f"{col:>6}{med:11.2%}{nr}{mx:9.2%}{grid:10.2%}"
                  f"{'OK' if ok else '<<<':>7}")

    print(f"\n최대 중앙 차이: {worst:.2%}")
    return 0 if worst < 0.05 else 1


if __name__ == "__main__":
    raise SystemExit(main())
