"""
Ablation: real-fluid properties
===============================

Runs each manual example twice — once with Ermak's property model, once with
CoolProp reference equations of state — through an otherwise identical model,
and reports the difference.

This is the first change to `slabx` that alters physics rather than numerics,
so it is the first for which the reproduction work actually pays off: the
baseline is known to track the original implementation to a fraction of a
percent, which means a difference of a few percent here is attributable to
the property model and not to the port.

Reading the output
------------------
`|S0-S3|` is the effect of the change.  Compare it against the reproduction
error reported by `fortran_reference.py` for the same case — a change smaller
than that is not resolvable and should not be claimed.

Nothing here says the real properties are *better*.  They are more accurate
descriptions of the pure substances; whether that improves agreement with
experiment is a separate question, answered against field data, not against
the original code.

Known gap
---------
The *source* still uses the constant liquid density from the input deck:
`_PressurisedRelease.rho_source` reads `substance.rho_liquid` rather than
asking the backend.  That matters for a two-phase release — SLAB's ammonia
value is 12 % below reality — so the ammonia and chlorine numbers below
understate the effect of the change.  Wiring the backend into the source
term is the next step.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

from decks import CASES, Case, build_source          # noqa: E402
from slabx.coefficients import preset                           # noqa: E402
from slabx.core.plume import run_dispersion                     # noqa: E402
from slabx.post.concentration import concentration_field        # noqa: E402
from slabx.thermo.base import LegacyThermo, water_backend       # noqa: E402
from slabx.thermo.coolprop import CoolPropThermo, coolprop_water  # noqa: E402

#: Reproduction error per case, from `fortran_reference.py --case all`.
#: A property effect below this is inside the noise of the port itself.
REPRODUCTION = {
    "LNG": 0.0093, "LNG-neutral": 0.0076, "Ammonia-jet": 0.0881,
    "LNG-instant": 0.0049, "Chlorine-stack": 0.0130,
}

FIELDS = [
    ("vol_frac", "volume fraction"),
    ("T", "temperature [K]"),
    ("rho", "density [kg/m3]"),
    ("h", "cloud height [m]"),
    ("b_half", "half-width B [m]"),
    ("mass_frac_emission_liquid", "droplet fraction"),
]


def run(case: Case, *, real: bool):
    sub = case.substance()
    atm = case.atmosphere(legacy=False)
    src = build_source(case, sub)
    em = (CoolPropThermo(sub, fluid=case.fluid) if real
          else LegacyThermo(sub))
    wt = coolprop_water() if real else water_backend()
    traj, used = run_dispersion(
        src, atm, em, wt, x_max=case.xffm, t_max=case.t_max,
        n_puff_steps=60, coeffs=preset("ermak90"),
    )
    field = concentration_field(traj, atm, z="max", t_avg=case.tav,
                                t_release=case.tsd)
    return traj, field, used


def compare(case: Case) -> None:
    print(f"\n{'=' * 78}\n{case.name}   S0 legacy properties  ->  S3 CoolProp"
          f"\n{'=' * 78}")
    try:
        a, fa, ua = run(case, real=False)
        b, fb, ub = run(case, real=True)
    except Exception as exc:                                   # noqa: BLE001
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return

    lo = max(1.0, min(a.x.max(), b.x.max()) / 1000.0)
    xs = np.geomspace(max(lo, a.x[a.x > 0].min()),
                      min(a.x.max(), b.x.max()), 14)

    repro = REPRODUCTION.get(case.name)
    print(f"source half-width  S0 {ua.half_width:.3f}   S3 {ub.half_width:.3f} m")
    print(f"\n{'variable':<26}{'median |S0-S3|':>17}{'max':>10}"
          f"{'resolvable?':>14}")
    print("-" * 68)
    for attr, label in FIELDS:
        va = np.interp(xs, a.x, getattr(a, attr))
        vb = np.interp(xs, b.x, getattr(b, attr))
        ok = np.isfinite(va) & np.isfinite(vb) & (np.abs(va) > 1e-12)
        if ok.sum() < 3:
            continue
        d = np.abs(vb[ok] / va[ok] - 1.0)
        mark = ("-" if repro is None
                else "yes" if np.median(d) > repro else "below noise")
        print(f"{label:<26}{np.median(d) * 100:16.2f}%{d.max() * 100:9.2f}%"
              f"{mark:>14}")

    ca = np.interp(xs, fa.x, fa.peak)
    cb = np.interp(xs, fb.x, fb.peak)
    ok = ca > 0
    d = np.abs(cb[ok] / ca[ok] - 1.0)
    mark = ("-" if repro is None
            else "yes" if np.median(d) > repro else "below noise")
    print(f"{'centreline C(x,0,z)':<26}{np.median(d) * 100:16.2f}%"
          f"{d.max() * 100:9.2f}%{mark:>14}")

    if repro is not None:
        print(f"\nreproduction error for this case: {repro:.2%}")


def main() -> None:
    names = sys.argv[1:] or list(CASES)
    for n in names:
        compare(CASES[n])


if __name__ == "__main__":
    main()
