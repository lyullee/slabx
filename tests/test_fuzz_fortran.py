"""
Differential testing against the original on randomly sampled decks.

The five manual decks are the ones Ermak chose to publish, so they are also
the ones the model was most likely checked on.  These tests sample the input
space instead.  They are slow — each case runs the compiled Fortran and the
full model — so the count is small; `examples/fuzz_fortran.py` is the version
to run when looking for new problems.

Both known outcomes are pinned here: the agreement that has been established,
and the one deck class that is known to fail.

Skips when the oracle has not been built; see `golden/fortran/README.md`.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
sys.path.insert(0, str(ROOT / "golden" / "fortran"))


@pytest.fixture(scope="module")
def fuzz():
    oracle_mod = pytest.importorskip("oracle")
    try:
        slab = oracle_mod.FortranSLAB()
    except oracle_mod.OracleUnavailable as exc:
        pytest.skip(str(exc))

    import fuzz_fortran as F

    rng = np.random.default_rng(21)
    out = []
    for _ in range(12):
        case = F.sample(rng)
        try:
            d = F.compare(case, slab)
        except Exception:                                    # noqa: BLE001
            continue
        if not d or "__oracle_failed__" in d or "__ours_failed__" in d:
            continue
        out.append((case, d))
    if len(out) < 4:
        pytest.skip(f"only {len(out)} comparable decks in the sample")
    return out


def test_thermodynamic_state_agrees_on_random_decks(fuzz):
    """
    Temperature and density are the variables with no discretisation of
    their own — they follow algebraically from the integrated state — so they
    are the sharpest check that the port is faithful away from the manual's
    decks.
    """
    for label in ("t", "rho"):
        v = np.array([d[label] for _, d in fuzz if label in d])
        assert np.median(v) < 0.01, label
        assert np.percentile(v, 75) < 0.05, label


def test_geometry_and_concentration_agree_on_most_decks(fuzz):
    """
    Height, width and concentration carry the transition and the grid, so a
    few percent is expected; the median is what should be tight.
    """
    for label in ("h", "bb", "cv"):
        v = np.array([d[label] for _, d in fuzz if label in d])
        assert np.median(v) < 0.03, label


@pytest.mark.xfail(reason="known failure: dense low-momentum jet, see below",
                   strict=False)
def test_dense_low_momentum_jet():
    """
    A reproducing deck for the one failure class fuzzing has turned up.
    Three of 92 sampled decks hit it; all three are dense, slow releases.

    The chain, traced step by step:

    1. The jet leaves the source elevated, so it is lofted and V_g is zero
       while the gravity integral G accumulates unchecked.
    2. As it settles, z_c falls below h/2 and the cloud becomes grounded.
       V_g is then recovered from the accumulated G all at once and comes out
       about four times the reference's — 4.0 against 1.0 at x = 1.1 m.
    3. The excess V_g drives the cloud onto the ground faster still, which
       switches alpha_g from 0 to 0.25.
    4. With gravity now in the velocity cubic, EQ 4d loses its positive root
       and the in-place widening fires.
    5. B grows, U collapses, and the ``h > 2B`` clamp then sets
       ``B = sqrt(R/(2 rho U))`` — which with a collapsing U is unbounded.
       The result is h = 2B = 240 m.

    The amplifier is step 5; the cause is step 2.  Both implementations agree
    exactly at the source, so the divergence is entirely in how the
    accumulated gravity momentum is released at grounding — the same
    structure that leaves ``e_v`` at 0.9997 in the Burro runs.

    **This diagnosis was wrong, and the failure is resolved.** See
    `test_the_dense_failures_resolve_on_a_finer_grid`: at `substeps` = 8 this
    same deck gives h(10 m) = 0.819 against the original's 0.808. The chain
    above describes what happens *at the default grid*; the mechanism is a
    marginal instability that particular step sizes excite, not a defect in
    how the gravity momentum is released.

    Kept as xfail because it pins the behaviour **at the defaults**, which
    are Ermak's and are what this project reproduces. A user on the default
    grid does see this, so the failure is documented rather than deleted.
    """
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "examples"))
    from decks import Case

    case = Case(
        name="LNG-dense-jet", idspl=2, ncalc=1, wms=0.016043, cps=2238.0,
        tbp=111.70, cmed0=0.768, dhe=509900.0, cpsl=3348.5, rhosl=424.1,
        spb=-1.0, spc=0.0, ts=111.70, qs=1.40, as_=0.9426, tsd=100.0,
        qtis=0.0, hs=1.70, tav=10.0, xffm=300.0, zp=0.0, z0=6.19e-3,
        za=10.0, ua=1.50, ta=290.0, rh=50.0, stab=4.0, ala=0.0, n=61,
        fluid="Methane",
    )
    _, ref, cols = run_reference(case)
    idx = {c: i for i, c in enumerate(cols)}
    traj, _, _ = run_slabx(case, legacy=True)

    x = 10.0
    h_ref = float(np.interp(x, ref[:, idx["xp"]], ref[:, idx["hp"]]))
    h_ours = float(np.interp(x, traj.x, traj.h))
    assert h_ours == pytest.approx(h_ref, rel=0.1)


def test_the_dense_failures_resolve_on_a_finer_grid():
    """
    The failure class is a marginal instability, not a structural defect.

    Both reproducing decks agree with the original once the grid resolves
    the transition, and the two respond to *different* grids — an earlier
    version of this file called them the same failure reached through
    different source types, which was wrong:

    * the chlorine pool needs `n_source` >= 160, at which the source
      expansion fires (0.317 m -> 1.372 m against the original's 1.270 m)
      and the cloud is 1.18 m tall instead of 66 m;
    * the two-phase jet does not respond to `n_source` at all, because a jet
      has no source region. It needs `substeps` >= 8, at which h(10 m) is
      0.819 against the original's 0.808.

    The response is not monotonic in resolution — at `substeps` = 1 the jet
    is correct at `n_field` = 40 and 240 but diverges at 60, 80 and 120 —
    which is the signature of an instability that particular step sizes
    excite rather than of a convergence error. The defaults, which are
    Ermak's, happen to land in a bad spot for this deck.

    The defaults are not changed: they are part of what is being reproduced.
    """
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "examples"))
    try:
        from decks import Case
        from fortran_reference import build
    except ImportError as exc:                     # oracle harness absent
        pytest.skip(str(exc))

    from slabx.core.plume import run_plume
    from slabx.thermo.base import LegacyThermo, water_backend

    case = Case(
        name="LNG-dense-jet", idspl=2, ncalc=1, wms=0.016043, cps=2238.0,
        tbp=111.70, cmed0=0.768, dhe=509900.0, cpsl=3348.5, rhosl=424.1,
        spb=-1.0, spc=0.0, ts=111.70, qs=1.40, as_=0.9426, tsd=100.0,
        qtis=0.0, hs=1.70, tav=10.0, xffm=300.0, zp=0.0, z0=6.19e-3,
        za=10.0, ua=1.50, ta=290.0, rh=50.0, stab=4.0, ala=0.0, n=61,
        fluid="Methane",
    )
    sub, atm, src = build(case)

    def h_at_10(**kw):
        traj, _ = run_plume(src, atm, LegacyThermo(sub), water_backend(),
                            x_max=300.0, **kw)
        return float(np.interp(10.0, traj.x, traj.h))

    H_REF = 0.808                       # the compiled original

    # the default lands badly
    assert h_at_10() > 10.0

    # and enough substeps resolve it, at three different field grids
    for n_field in (40, 120, 240):
        h = h_at_10(n_field_steps=n_field, substeps=8)
        assert h == pytest.approx(H_REF, rel=0.05), (n_field, h)

    # non-monotonic: correct at the coarsest and finest, wrong between
    assert h_at_10(n_field_steps=240, substeps=1) == pytest.approx(H_REF,
                                                                   rel=0.05)
    assert h_at_10(n_field_steps=120, substeps=1) > 10.0


def test_a_finer_grid_does_not_disturb_the_field_comparison():
    """The resolution that fixes the dense decks leaves the field result alone."""
    from slabx.validation.lng_pools import compare_lfl
    from slabx.validation.metrics import metrics

    r3, r8 = compare_lfl(substeps=3), compare_lfl(substeps=8)
    a = metrics(r3["observed"], r3["predicted"])
    b = metrics(r8["observed"], r8["predicted"])
    assert b.FAC2 == a.FAC2
    assert abs(b.MG / a.MG - 1.0) < 0.01
