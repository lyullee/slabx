"""
Against Ermak's Fortran, not a transcription of it.

The project began by comparing with `SLAB_main.js`.  Six differences were
recorded as defects of SLAB; checking against `SLAB.FOR` showed that only one
was.  Three came from a mechanical ``min`` -> ``Math.min`` substitution
applied to the whole file — it also produced ``Math.minus`` and
``terMath.minated`` inside comments — one was a single-character typo (a misspelling of it
for the intended variable, appearing once against ten), and one was a deliberate sentinel
mistaken for an omission.

These tests skip when the binary has not been built:

    cd golden/fortran
    gfortran -std=legacy -O2 -fno-automatic -w -o slab slab.f
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "golden" / "fortran"))
sys.path.insert(0, str(ROOT / "examples"))

oracle_mod = pytest.importorskip("oracle")
FortranSLAB = oracle_mod.FortranSLAB
OracleUnavailable = oracle_mod.OracleUnavailable

from slabx.coefficients import COEFFS                     # noqa: E402


@pytest.fixture(scope="module")
def slab():
    try:
        return FortranSLAB()
    except OracleUnavailable as exc:
        pytest.skip(str(exc))


BURRO8 = dict(
    idspl=1, wms=0.016043, cps=2238.0, tbp=111.70, cmed0=0.0, dhe=509900.0,
    cpsl=3348.50, rhosl=424.10, ts=111.70, qs=116.93, as_=657.0, tsd=107.0,
    tav=10.0, xffm=1000.0, z0=2e-4, za=2.88, ua=1.92, ta=306.0, rh=4.6,
    stab=0.0, ala=0.0665,
)


# ===========================================================================
# the input format
# ===========================================================================
def test_precision_is_not_lost_to_the_read_format():
    """
    The format is ``f10.3``, but an explicit decimal point in the data
    overrides it.  Writing ``"%10.3f"`` truncates a molecular weight of
    0.016043 to 0.016, after which every downstream quantity is NaN — which
    is how this was found.
    """
    f10 = oracle_mod._f10
    assert float(f10(0.016043)) == pytest.approx(0.016043, rel=1e-9)
    assert float(f10(509900.0)) == pytest.approx(509900.0)
    assert float(f10(2e-4)) == pytest.approx(2e-4, rel=1e-6)
    assert all(len(f10(v)) <= 10
               for v in (0.016043, 509900.0, 2e-4, -1.0, 1e-7, 1234567.0))


def test_the_deck_is_one_value_per_line(tmp_path, slab):
    """The format carries no repeat count, so each value is its own record."""
    path = tmp_path / "input"
    slab.write_input(path, **BURRO8)
    lines = path.read_text().splitlines()
    assert len(lines) == 2 + 9 + 6 + 6 + 1 + 5 + 1 + 1
    assert float(lines[-1]) < 0                  # stop sentinel


def test_a_neutral_deck_needs_no_monin_obukhov_length(tmp_path, slab):
    with pytest.raises(ValueError, match="Monin-Obukhov"):
        slab.write_input(tmp_path / "a", **{**BURRO8, "ala": None})
    slab.write_input(tmp_path / "b", **{**BURRO8, "stab": 4.0, "ala": None})


# ===========================================================================
# the original runs
# ===========================================================================
def test_the_original_runs_and_produces_a_table(slab):
    r = slab.run(**BURRO8)
    assert r.table.shape[1] == 12
    assert len(r.table) > 30
    assert np.all(np.isfinite(r.table))
    assert r.x.max() > 500.0


def test_only_the_first_of_four_tables_is_parsed(slab):
    """
    `predict` holds four tables.  Taking every numeric line interleaves them;
    an early parser did exactly that and reported a factor-three
    disagreement that was purely a column misalignment.
    """
    r = slab.run(**BURRO8)
    assert r.raw.count("time averaged") >= 3      # the other three
    # density stays within a factor of two of ambient throughout
    rho = r.column("rho")
    assert np.all((rho > 1.0) & (rho < 2.0))


def test_the_stability_typo_is_absent_from_the_original(slab):
    """
    The JavaScript throws for |1/L| above about 0.15, reading an undeclared
    a misspelling of it.  The Fortran reads the intended variable (SLAB.FOR line 236) and runs.
    """
    for inv_L in (0.30, -0.30, 0.60):
        r = slab.run(**{**BURRO8, "ala": inv_L})
        assert len(r.table) > 20, inv_L
        assert np.all(np.isfinite(r.column("cv"))), inv_L


# ===========================================================================
# slabx against it
# ===========================================================================
def _slabx_burro8():
    from slabx.core.plume import run_dispersion
    from slabx.core.source import EvaporatingPool
    from slabx.submodels.atmosphere import Atmosphere
    from slabx.thermo.base import LegacyThermo, Substance, water_backend

    sub = Substance(name="LNG", mw=0.016043, cp_vapour=2238.0,
                    cp_liquid=3348.5, dh_vap=509900.0, T_boil=111.7,
                    rho_liquid=424.1)
    atm = Atmosphere(u_ref=1.92, z_ref=2.88, T=306.0, rh=4.6, z0=2e-4,
                     inv_L=0.0665)
    src = EvaporatingPool(substance=sub, rate=116.93, area=657.0,
                          duration=107.0)
    return run_dispersion(src, atm, LegacyThermo(sub), water_backend(),
                          x_max=1000.0, n_puff_steps=40)


def test_reproduction_against_the_original(slab):
    """
    Burro 8, variable by variable.  The median difference is well under one
    per cent on every quantity, which is the level of the original's own
    discretisation.
    """
    ref = slab.run(**BURRO8)
    traj, _ = _slabx_burro8()

    lo = max(ref.x[ref.x > 0].min(), traj.x[traj.x > 0].min())
    hi = min(ref.x.max(), traj.x.max())
    xs = np.geomspace(lo, hi, 40)

    for col, attr in (("h", "h"), ("bb", "b_half"), ("b", "b_shape"),
                      ("cv", "vol_frac"), ("rho", "rho"), ("t", "T"),
                      ("u", "u")):
        o = np.interp(xs, ref.x, ref.column(col))
        p = np.interp(xs, traj.x, getattr(traj, attr))
        live = np.abs(o) > 1e-9
        median = float(np.median(np.abs(p[live] / o[live] - 1.0)))
        assert median < 0.01, f"{col}: {median:.2%}"


def test_the_source_expansion_matches(slab):
    """
    The outer fixed point for the evaporating-pool source area lands on the
    same effective half-width: 12.82 m geometric expands to 31.5 m in both.
    """
    from slabx.core.plume import run_dispersion
    from slabx.core.source import EvaporatingPool
    from slabx.submodels.atmosphere import Atmosphere
    from slabx.thermo.base import LegacyThermo, Substance, water_backend

    sub = Substance(name="LNG", mw=0.016043, cp_vapour=2238.0,
                    cp_liquid=3348.5, dh_vap=509900.0, T_boil=111.7,
                    rho_liquid=424.1)
    atm = Atmosphere(u_ref=1.92, z_ref=2.88, T=306.0, rh=4.6, z0=2e-4,
                     inv_L=0.0665)
    src = EvaporatingPool(substance=sub, rate=116.93, area=657.0,
                          duration=107.0)
    _, used = run_dispersion(src, atm, LegacyThermo(sub), water_backend(),
                             x_max=1000.0, n_puff_steps=40)
    ref = slab.run(**BURRO8)
    assert used.half_width == pytest.approx(ref.column("bb")[0], rel=0.01)


def test_the_constants_are_ermaks(slab):
    """
    Cross-check on the numbers rather than the trajectory: every coefficient
    in `Coefficients` must be the literal in SLAB.FOR.  ``c_drag_top`` was
    0.0195 until this comparison showed the code has always used 0.02.
    """
    assert COEFFS.c_drag_top == 0.02          # L381
    assert COEFFS.a_entrain == 1.50           # L378  aa = 1.50
    assert COEFFS.c_mu_strat == 0.025         # L382  cri = .025
    assert COEFFS.h_entrain_ref == 4.0        # L466  hrf = 4.0
