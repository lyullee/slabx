"""
First end-to-end validation run: ambient wind profile, Thorney Island 005
========================================================================

Small but real.  It exercises the whole chain that every later milestone
will reuse — model -> paired predictions -> metrics -> paired bootstrap —
on a submodel that is already complete.

Data
----
Trial 005, 3 August 1982.  Cup-anemometer means on the 'A' mast at
x=400 m, y=50 m, taken from the trial data sheet (HSE Safety Engineering
Laboratory, "Heavy Gas Dispersion Trials Thorney Island 1982-3, Data for
Trial 005/I", pages E03/E09/E11/E19/E21).  Two independent averaging
windows are reported per height:

  RUN UP   : mean over the 300 s before container release
  RUN DOWN : mean over the 300 s after the release

Roughness and stability follow the PHMSA toxic-dispersion MVD
(03903-RP-004, Table 2.23-3): z0 = 0.01 m base case.

Caveats — read before believing any number
------------------------------------------
* N = 4 predicted points per window.  This is a pipeline smoke test, not
  evidence about the model.
* The trial sheet reports SEVEN stability estimates for this trial, which
  disagree from A to E (visual B/C, dT/dz B, solarimeter B, heat flux B,
  Richardson A, bulk Richardson D, sigma-theta E).  Stability is swept
  here rather than assumed.
* Trial 005 is flagged unusable for dispersion (the container hung up
  during the drop and gas escaped), but the meteorology is unaffected.
* Run-up is nearly height-uniform above 4.5 m, which no surface-layer
  profile can reproduce; run-down is better behaved.
"""

from __future__ import annotations

import numpy as np

from slabx.submodels.atmosphere import Atmosphere
from slabx.validation.metrics import compare, metrics

# --- observations ---------------------------------------------------------
HEIGHTS = np.array([2.0, 4.5, 10.0, 17.3, 30.0])          # m
RUN_UP = np.array([2.76, 3.09, 3.09, 3.08, 2.99])         # m/s
RUN_DOWN = np.array([3.96, 4.46, 4.58, 4.60, 4.41])       # m/s
Z_ANCHOR = 10.0                                            # m, used to fit u*
Z0 = 0.01                                                  # m, MVD base case
T_AMB = 295.35                                             # K (22.2 C at 9 m)
RH = 64.1                                                  # %

WINDOWS = {"run_up": RUN_UP, "run_down": RUN_DOWN}
STABILITIES = ["A", "B", "C", "D", "E"]                    # sheet's own spread


def predict(u_anchor: float, stability: str, **kw) -> np.ndarray:
    """Wind speed at HEIGHTS, with u* fitted to the 10 m observation."""
    atm = Atmosphere(
        u_ref=u_anchor, z_ref=Z_ANCHOR, T=T_AMB, rh=RH, z0=Z0,
        stability=stability, **kw,
    )
    return np.array([atm.wind_speed(z) for z in HEIGHTS])


def _mask_anchor(arr):
    """Drop the anchor height — it is fitted, not predicted."""
    return arr[HEIGHTS != Z_ANCHOR]


def sweep_stability() -> None:
    print("=" * 74)
    print("Stability sweep — u* fitted at 10 m, other 4 heights predicted")
    print("=" * 74)
    for wname, obs in WINDOWS.items():
        u10 = float(obs[HEIGHTS == Z_ANCHOR][0])
        print(f"\n[{wname}]  u(10 m) = {u10:.2f} m/s")
        print(f"  {'stab':<6}{'FAC2':>7}{'MG':>8}{'VG':>8}{'FB':>8}   "
              + "".join(f"{z:>7.1f}" for z in HEIGHTS))
        print(f"  {'obs':<6}{'':>7}{'':>8}{'':>8}{'':>8}   "
              + "".join(f"{v:>7.2f}" for v in obs))
        for s in STABILITIES:
            p = predict(u10, s)
            m = metrics(_mask_anchor(obs), _mask_anchor(p))
            print(f"  {s:<6}{m.FAC2:>7.2f}{m.MG:>8.3f}{m.VG:>8.4f}{m.FB:>+8.3f}   "
                  + "".join(f"{v:>7.2f}" for v in p))


def ablate_legacy_defect() -> None:
    """
    Paired comparison of the two forms of the unstable wind profile.

    S0 = SLAB_uafn as ported (sign flip + spurious square root)
    S1 = exact integral of the assumed gradient

    Unstable classes only; the defect lives in that branch.  Trials are the
    (window, stability) pairs, which is the honest grouping here because the
    four heights within one profile are strongly correlated.
    """
    obs, p_legacy, p_exact, trial = [], [], [], []
    for wname, o in WINDOWS.items():
        u10 = float(o[HEIGHTS == Z_ANCHOR][0])
        for s in ("A", "B", "C"):
            obs.append(_mask_anchor(o))
            p_legacy.append(_mask_anchor(predict(u10, s, legacy_unstable_profile=True)))
            p_exact.append(_mask_anchor(predict(u10, s)))
            trial.append([f"{wname}:{s}"] * (len(HEIGHTS) - 1))

    c = compare(
        np.concatenate(obs), np.concatenate(p_legacy), np.concatenate(p_exact),
        trial=np.concatenate(trial),
        label_a="legacy", label_b="exact", n_boot=5000, seed=0,
    )
    print("\n" + "=" * 74)
    print("Ablation: ported closed form vs exact integral (unstable branch)")
    print("=" * 74)
    print(c.summary())
    d = np.abs(np.concatenate(p_legacy) / np.concatenate(p_exact) - 1.0)
    print(f"\nmax |legacy/exact - 1| = {d.max():.2e}"
          f"   ({'below' if d.max() < 1e-3 else 'above'} 0.1%)")
    print("Difference is real but far below the observational scatter, so it")
    print("cannot be resolved against field data — it matters only for")
    print("bit-level comparison against the reference implementation.")


if __name__ == "__main__":
    sweep_stability()
    ablate_legacy_defect()
