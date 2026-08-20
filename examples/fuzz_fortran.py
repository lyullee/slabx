"""
Random differential testing against Ermak's Fortran
====================================================

    python3 examples/fuzz_fortran.py            # 60 decks, seed 0
    python3 examples/fuzz_fortran.py 200 7      # 200 decks, seed 7

Why the oracle changed
----------------------
The earlier fuzzer compared against a JavaScript transcription.  That copy
throws for ``|1/L|`` above about 0.15 — it reads an undeclared a misspelling of it where
the Fortran reads the intended variable (SLAB.FOR line 236, ten occurrences against one) —
so decks in that region had to be discarded as "outside the oracle".  About a
third of the sample was lost that way, and the loss was mistakenly recorded
as a property of SLAB rather than of the port.

The Fortran has no such limit, so the stability axis is now sampled over its
full useful range instead of a narrow band around neutral.

What the five manual decks do not cover
---------------------------------------
They are the decks Ermak chose to publish, so they are where the model is
best behaved.  All five sit at ``|1/L| < 0.07``, none has a two-phase
vertical jet, and none has a source area small enough to make the initial
slab tall.  Five points cannot cover a 28-dimensional input space.

Sampling
--------
Materials are drawn as whole substances rather than by independently sampling
molecular weight and boiling point, which would produce fluids that do not
exist.  Half the decks give a stability class and half an inverse
Monin-Obukhov length, because the two take different paths through the
stability mapping.

Reporting
---------
Per-variable medians and tails, plus the decks where the two disagree by more
than half.  A differential test says the two implementations differ, not
which is wrong: attributing a difference needs a reading of the original.

What it has found so far
------------------------
* **The vertical-jet clock.**  The plume-to-puff transition row took its time
  from EQ 29b in the plume's own clock, while every other row carried the
  rise duration.  Four of ten sampled vertical jets raised
  "t must be non-decreasing"; the five manual decks did not, because their
  rise is negligible next to t_sd.  Fixed, pinned by
  `tests/test_vertical_jet.py::test_the_transition_row_carries_the_rise_duration`.

* **Dense, low-momentum releases (open).**  About one deck in six disagrees
  by more than half, and the tail is dominated by one family: a small source
  area with a high rate of a heavy material.  The signature is that the
  source expansion (EQ 30) does not fire — half-width 0.317 m against the
  original's 1.270 m for one chlorine pool — after which the in-place
  widening runs away and the h = 2B clamp returns a cloud tens of metres
  tall against the original's 1.4 m.

  This is the same failure already recorded as
  `tests/test_fuzz_fortran.py::test_dense_low_momentum_jet`, but reached
  through an evaporating pool rather than a two-phase jet, which places the
  cause in the source-expansion fixed point rather than in the jet source.
  Still unresolved; no prescription without a free parameter has been found.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "golden" / "fortran"))
sys.path.insert(0, str(ROOT / "examples"))

from oracle import FortranSLAB, OracleUnavailable            # noqa: E402

from decks import FLUIDS, Case                               # noqa: E402
from fortran_reference import MAP, build, deck               # noqa: E402

from slabx.coefficients import preset                        # noqa: E402
from slabx.core.plume import run_dispersion                  # noqa: E402
from slabx.scope import ScopeWarning                         # noqa: E402
from slabx.thermo.base import LegacyThermo, water_backend    # noqa: E402


def sample(rng: np.random.Generator) -> Case:
    """One deck.  Stability now spans the range the Fortran actually accepts."""
    f = FLUIDS[rng.integers(len(FLUIDS))]
    idspl = int(rng.choice([1, 2, 3, 4]))

    if rng.random() < 0.5:
        stab, ala = float(rng.integers(1, 7)), 0.0
    else:
        # The JavaScript threw above 0.15.  The Fortran does not, so the
        # band that used to be discarded is now the interesting part.
        stab, ala = 0.0, float(rng.uniform(-1.0, 1.0))

    z0 = float(10 ** rng.uniform(-4.0, -1.0))
    ua = float(rng.uniform(1.0, 12.0))
    za = float(rng.choice([2.0, 3.0, 10.0]))
    ta = float(rng.uniform(265.0, 315.0))
    rh = float(rng.uniform(0.0, 90.0))

    qs = float(10 ** rng.uniform(-0.5, 2.5))
    tsd = float(10 ** rng.uniform(1.0, 3.0))
    liquid = float(rng.uniform(0.0, 0.9)) if idspl in (2, 3) else 0.0
    hs = float(rng.uniform(0.0, 5.0)) if idspl in (2, 3) else 0.0

    if idspl == 4:
        qs, tsd = 0.0, 0.0
        qtis = float(10 ** rng.uniform(2.0, 5.0))
        area = float(max(10 ** rng.uniform(0.0, 3.5), qtis / 100.0))
    else:
        qtis = 0.0
        area = (float(10 ** rng.uniform(-3.0, 0.5)) if idspl in (2, 3)
                else float(10 ** rng.uniform(-2.0, 3.0)))

    return Case(
        name=f"{f.name}-{idspl}", idspl=idspl, ncalc=1,
        wms=f.wms, cps=f.cps, tbp=f.tbp, cmed0=liquid, dhe=f.dhe,
        cpsl=f.cpsl, rhosl=f.rhosl, spb=f.spb, spc=f.spc, ts=f.tbp,
        qs=qs, as_=area, tsd=tsd, qtis=qtis, hs=hs,
        tav=float(rng.choice([1.0, 10.0, 60.0, 600.0])),
        xffm=float(10 ** rng.uniform(2.0, 4.0)), zp=0.0,
        z0=z0, za=za, ua=ua, ta=ta, rh=rh, stab=stab, ala=ala,
        n=121, fluid=f.fluid,
    )


def compare(case: Case, oracle: FortranSLAB) -> dict | None:
    """Median relative difference per variable, or None if either side fails."""
    import warnings

    try:
        ref = oracle.run(**deck(case), timeout=30.0)
    except Exception:                                        # noqa: BLE001
        return {"__oracle_failed__": True}

    try:
        sub, atm, src = build(case)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ScopeWarning)
            traj, _ = run_dispersion(src, atm, LegacyThermo(sub),
                                     water_backend(), x_max=case.xffm,
                                     n_puff_steps=40, coeffs=preset("ermak90"))
    except Exception as exc:                                 # noqa: BLE001
        return {"__ours_failed__": type(exc).__name__}

    pos_r, pos_t = ref.x > 0, traj.x > 0
    if pos_r.sum() < 5 or pos_t.sum() < 5:
        return None
    lo = max(ref.x[pos_r].min(), traj.x[pos_t].min())
    hi = min(ref.x.max(), traj.x.max())
    if not hi > lo * 1.5:
        return None
    xs = np.geomspace(lo, hi, 25)

    out = {}
    for col, attr in MAP.items():
        raw = ref.column(col)
        valid = raw > -0.5 if col in ("t", "rho", "cv", "h") else \
            np.ones_like(raw, dtype=bool)
        if valid.sum() < 5:
            continue
        o = np.interp(xs, ref.x[valid], raw[valid])
        p = np.interp(xs, traj.x, getattr(traj, attr))
        scale = float(np.max(np.abs(o)))
        # Also stay inside the range the original actually defines.  It
        # leaves -1 sentinels through the plume-rise region (SLAB.FOR
        # L1250-1251), and interpolating across them from the first real
        # value compares our genuine dense cloud against something close to
        # ambient -- which reported a 222 % disagreement on two chlorine
        # jets that are in fact fine.  `fortran_reference.py` had this
        # guard; this harness did not.
        live = (np.abs(o) > max(1e-12, 1e-4 * scale)) \
            & (xs >= float(ref.x[valid].min()))
        if live.sum() < 5:
            continue
        if col == "zc":
            depth = np.interp(xs, ref.x, np.maximum(ref.column("h"), 1e-6))
            d = np.abs(p[live] - o[live]) / depth[live]
        else:
            d = np.abs(p[live] / o[live] - 1.0)
        out[col] = float(np.median(d))
    return out or None


def main(n: int = 60, seed: int = 0) -> int:
    try:
        oracle = FortranSLAB()
    except OracleUnavailable as exc:
        print(exc)
        return 2

    rng = np.random.default_rng(seed)
    per = defaultdict(list)
    compared = oracle_failed = ours_failed = skipped = 0
    worst: list[tuple[float, str, Case]] = []
    stable_decks = 0

    for i in range(n):
        case = sample(rng)
        if case.stab == 0.0 and abs(case.ala) > 0.15:
            stable_decks += 1

        res = compare(case, oracle)
        if res is None:
            skipped += 1
            continue
        if "__oracle_failed__" in res:
            oracle_failed += 1
            continue
        if "__ours_failed__" in res:
            ours_failed += 1
            print(f"  [{i}] ours raised {res['__ours_failed__']}: "
                  f"{case.name} 1/L={case.ala:+.3f} q={case.qs:.3g}")
            continue

        compared += 1
        for k, v in res.items():
            per[k].append(v)
        top = max(res.items(), key=lambda kv: kv[1])
        if top[1] > 0.5:
            worst.append((top[1], top[0], case))

    print(f"\n{'=' * 66}")
    print(f"표집 {n}   비교 {compared}   오라클 실패 {oracle_failed}   "
          f"우리 실패 {ours_failed}   중첩 없음 {skipped}")
    print(f"이전 오라클이라면 버려졌을 데크 (|1/L| > 0.15): {stable_decks}")
    print(f"{'=' * 66}")

    print(f"\n{'변수':>6}{'중앙':>10}{'p90':>10}{'p99':>10}{'n':>6}")
    for k in sorted(per, key=lambda k: float(np.median(per[k]))):
        v = np.array(per[k])
        print(f"{k:>6}{np.median(v):9.2%}{np.percentile(v, 90):10.2%}"
              f"{np.percentile(v, 99):10.2%}{len(v):6d}")

    if worst:
        print(f"\n50% 초과 {len(worst)}건 ({len(worst) / max(compared, 1):.1%})")
        for d, col, c in sorted(worst, reverse=True)[:6]:
            print(f"  {col:>4} {d:8.1%}  idspl={c.idspl} {c.name:>12} "
                  f"1/L={c.ala:+.3f} stab={c.stab:.0f} "
                  f"q={c.qs:.3g} A={c.as_:.3g} liq={c.cmed0:.2f}")
    else:
        print("\n50% 초과 없음")
    return 0


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    raise SystemExit(main(n, seed))
