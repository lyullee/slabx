"""
Burro trials: does the model beat the original?
===============================================

The question this answers is not "how does slabx compare with Phast".  It is
"is the 1990 model, run today, improved by any of the changes made to it" —
and every change is measured against the version of the model it is meant to
improve.

Surface roughness dominates everything else
-------------------------------------------
The report is produced twice, at the two roughness lengths the validation
database itself discusses.  At the MVD base value the original model
over-predicts by a factor 1.65 and fails Chang-Hanna; at the value FERC
proposed and the MVD rejected as "excessive", the same model passes on all
five measures without any change at all.  No physics change tested here
comes close to that in size, and any claim about a model change has to be
read against it.

Two baselines, deliberately
---------------------------
``slab90``  SLAB exactly as released: the stability-mapping typo, the
            unstable wind-profile transcription error and the rounded drag
            coefficient all present.
``slabx``   The same model with those three corrected and nothing else
            changed.

A *defect correction* is judged against `slab90`.  A *physics change* is
judged against `slabx`, because comparing it against the original would let
it take credit for the transcription fixes.  Each entry in `PREDICTIONS`
names its own baseline for that reason.

Phast appears in the point table for context only.  It is a different model
with a different source term; it is not the thing being tested.

Reading a verdict
-----------------
A variant is scored on *where* its effect appears, not on whether an
aggregate statistic improved.  The registered prediction says which trials
should move and which should not; a change that improves the wrong ones is
evidence against its mechanism however good the totals look.

Usage
-----
    python examples/validate_burro.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slabx.validation.field_trials import (                       # noqa: E402
    BURRO_AREA, BURRO_TRIALS, PREDICTIONS, ROUGHNESS, VARIANTS,
    load_conditions, validate,
)
from slabx.validation.metrics import check_acceptance, metrics    # noqa: E402

#: Effect below this counts as "no effect" when scoring a prediction.
NOISE = 0.02

STABILITY = {"BU03": "C", "BU07": "D", "BU08": "E", "BU09": "D"}

ORDER = ["slab90", "slabx", "canonical", "coolprop",
         "canonical+coolprop", "frontal"]


def point_table(obs, pred):
    print(f"\n{'=' * 82}\nPoint by point [mol %]\n{'=' * 82}")
    print(f"{'trial':>6}{'x [m]':>7}{'observed':>10}{'slab90':>9}"
          f"{'slabx':>9}{'canonical':>11}{'coolprop':>10}{'Phast':>9}")
    for i, o in enumerate(obs):
        ph = o["c_phast"]
        print(f"{o['trial']:>6}{o['x']:7.0f}{o['c_obs'] * 100:10.2f}"
              f"{pred['slab90'][i] * 100:9.2f}{pred['slabx'][i] * 100:9.2f}"
              f"{pred['canonical'][i] * 100:11.2f}"
              f"{pred['coolprop'][i] * 100:10.2f}"
              + (f"{ph * 100:9.2f}" if np.isfinite(ph) else f"{'-':>9}"))


def overall(y, pred, ph):
    print(f"\n{'=' * 82}\nOverall skill\n{'=' * 82}")
    for v in ORDER:
        m = metrics(y, pred[v])
        flag = "" if check_acceptance(m)["all"] else "   fails Chang-Hanna"
        print(f"{v:>20}  {m}{flag}")
    ok = np.isfinite(ph)
    print(f"{'Phast (context)':>20}  {metrics(y[ok], ph[ok])}")


def per_trial(y, tid, pred):
    print(f"\n{'=' * 82}\nPer trial: MG   (1 = unbiased, < 1 = over-predicts)"
          f"\n{'=' * 82}")
    cond = load_conditions()
    cols = ORDER[:5]
    print(f"{'trial':>6}{'n':>3}{'stab':>5}{'u':>6}"
          + "".join(f"{v:>20}" for v in cols))
    for t in BURRO_TRIALS:
        m = tid == t
        row = f"{t:>6}{m.sum():3d}{STABILITY[t]:>5}{cond[t]['u_ref']:6.2f}"
        for v in cols:
            row += f"{metrics(y[m], pred[v][m]).MG:20.3f}"
        print(row)


def verdicts(y, tid, pred, comparisons):
    print(f"\n{'=' * 82}\nPre-registered predictions\n{'=' * 82}")
    for name, p in PREDICTIONS.items():
        eff = {}
        for t in BURRO_TRIALS:
            m = tid == t
            a = metrics(y[m], pred[p.baseline][m]).MG
            b = metrics(y[m], pred[p.variant][m]).MG
            eff[t] = abs(b / a - 1.0)

        want = {t: eff[t] > NOISE for t in p.expect_effect_in}
        wont = {t: eff[t] <= NOISE for t in p.expect_no_effect_in}
        held = all(want.values()) and all(wont.values())

        c = comparisons[name]
        moved = [k for k in ("MG", "VG", "FAC2", "NMSE") if c.significant[k]]
        print(f"\n[{name}]  vs {p.baseline}   ->  "
              f"{'HELD' if held else 'FAILED'}")
        print(f"  {p.rationale}")
        print("  effect on MG:  "
              + "   ".join(f"{t} {eff[t]:+.1%}" for t in BURRO_TRIALS))
        if p.expect_effect_in:
            print(f"  wanted in {list(p.expect_effect_in)}: "
                  + ", ".join(f"{t} {'yes' if v else 'NO'}"
                              for t, v in want.items()))
        if p.expect_no_effect_in:
            print(f"  wanted none in {list(p.expect_no_effect_in)}: "
                  + ", ".join(f"{t} {'ok' if v else 'MOVED'}"
                              for t, v in wont.items()))
        print("  paired bootstrap: "
              + (f"significant in {moved}" if moved
                 else "no significant change in any measure"))


def main():
    for case in ("mvd_base", "ferc"):
        report(case)


def report(roughness: str):
    res = validate(roughness=roughness)
    obs, y, tid = res["observations"], res["y"], res["trial"]
    pred = res["predictions"]
    ph = np.array([o["c_phast"] for o in obs])

    print(f"\n{'#' * 82}\n# roughness case: {roughness}  "
          f"(z0 = {res['z0']} m)\n{'#' * 82}")
    print(f"source areas [m^2]: {BURRO_AREA}")
    print("\nvariants:")
    for n in ORDER:
        v = VARIANTS[n]
        legacy = []
        if v.legacy_stability_bug:
            legacy.append("stability typo")
        if v.legacy_unstable_profile:
            legacy.append("unstable profile")
        tag = f"  [{', '.join(legacy)}]" if legacy else ""
        real = "  real properties" if v.real_properties else ""
        print(f"  {n:<20} coeffs={v.coefficients}{real}{tag}")

    point_table(obs, pred)
    overall(y, pred, ph)
    per_trial(y, tid, pred)

    print(f"\n{'=' * 82}\nPaired bootstrap, each against its own baseline"
          f"\n{'=' * 82}")
    for c in res["comparisons"].values():
        print(f"\n{c.summary(keys=('MG', 'VG', 'FAC2', 'NMSE'))}")

    verdicts(y, tid, pred, res["comparisons"])


if __name__ == "__main__":
    main()
