"""
Reading the SMEDIS standardised trial files.

Twenty-eight trials across eight campaigns, in a common but not rigid
layout.  These tests pin the parser and the contents, because everything
downstream depends on the numbers being read out of the right columns.

The strongest check here is the cross-validation: the arc-wise maxima parsed
from the SMEDIS spreadsheets agree exactly with the independently published
Jack Rabbit III summary workbook for all ten trial-arc pairs they share.
"""

import csv
from pathlib import Path

import pytest

pytest.importorskip("xlrd")

from slabx.validation._data_access import ENV_VAR, data_root, require
from slabx.validation.smedis import (                             # noqa: E402
    MISSING, load_all_smedis, load_smedis,
)

#: The SMEDIS spreadsheets are third-party and not distributed; see
#: `validation/data/provenance/README.md`. They are looked for in the user's
#: own directory rather than inside the package, so that they cannot be
#: swept into a wheel or a commit by accident.
SMEDIS = data_root() / "smedis"

#: Every test here needs them, so the whole module skips when they are
#: absent rather than each test reporting the same missing directory.
pytestmark = pytest.mark.skipif(
    not SMEDIS.is_dir() or not any(SMEDIS.glob("*.xls")),
    reason=f"SMEDIS spreadsheets not present. Obtain them from their "
           f"official source and place them in {SMEDIS}, or set {ENV_VAR} "
           f"to the directory holding them. See "
           f"validation/data/provenance/ for the source of each file.",
)


@pytest.fixture(scope="module")
def trials():
    return load_all_smedis(SMEDIS)


# ===========================================================================
# what is there
# ===========================================================================
def test_all_twenty_eight_trials_parse(trials):
    assert len(trials) == 28
    assert {t.dataset for t in trials} >= {
        "Thorney Island", "Desert Tortoise", "FLADIS", "Prairie Grass",
        "BA-Hamburg",
    }


def test_obstructed_trials_are_identified(trials):
    """
    Six of the twenty-eight have box arrays, canyons, trenches, fences or
    curved walls.  The model has no representation of any of them, so they
    are excluded rather than silently run.
    """
    blocked = [t for t in trials if t.obstructed]
    assert len(blocked) == 6
    assert len(trials) - len(blocked) == 22
    for t in blocked:
        assert (t.raw("obstacles") or "").strip().lower() not in ("", "none")


def test_the_missing_sentinel_never_reaches_a_caller(trials):
    """
    -999 means "not measured".  A friction velocity of minus 999 that reaches
    a model is worse than no friction velocity at all.
    """
    for t in trials:
        for name in ("wind_speed", "friction_velocity", "roughness",
                     "monin_obukhov", "temperature"):
            v = t.value(name)
            assert v is None or v > MISSING + 1.0, (t.trial, name)


# ===========================================================================
# conditions
# ===========================================================================
def test_thorney_island_conditions():
    t = load_smedis(SMEDIS / "ti08.xls")
    assert t.dataset == "Thorney Island"
    assert t.instantaneous
    assert t.raw("substance") == "R12"
    assert t.value("wind_speed") == pytest.approx(2.4)
    assert t.value("friction_velocity") == pytest.approx(0.126)
    assert t.value("roughness") == pytest.approx(0.005)
    assert t.value("molecular_weight") == pytest.approx(47.2)


def test_measured_friction_velocity_is_available_for_most_trials(trials):
    """
    Which matters: u* is what the entrainment closure actually uses, and
    having it measured removes the roughness-length guesswork that dominated
    the Burro comparison.
    """
    have = [t for t in trials if t.value("friction_velocity") is not None]
    assert len(have) >= 15


# ===========================================================================
# arc-wise maxima
# ===========================================================================
def test_arcs_are_read_for_the_jet_trials():
    for name, n in (("dt1", 5), ("dt2", 6), ("fladis9", 4), ("pg8", 5)):
        arcs = load_smedis(SMEDIS / f"{name}.xls").arcs()
        assert len(arcs) == n, name
        for a in arcs:
            assert a["distance"] > 0 and a["concentration"] > 0


def test_arcs_agree_with_the_independent_jriii_workbook():
    """
    The cross-validation.  The SMEDIS spreadsheets and the Jack Rabbit III
    summary workbook were produced by different groups from the same trials;
    every trial-arc pair they share agrees to the digit.
    """
    published = {(r["trial"], float(r["arc_m"])): float(r["c_measured_ppm"])
                 for r in csv.DictReader(
                     open(require("jriii_arcmax_measured.csv")))}
    checked = 0
    for stem, trial in (("dt1", "DT1"), ("dt2", "DT2"), ("fladis9", "FL9"),
                        ("fladis16", "FL16"), ("fladis24", "FL24")):
        seen = set()
        for a in load_smedis(SMEDIS / f"{stem}.xls").arcs():
            d = a["distance"]
            if d in seen:
                continue
            seen.add(d)
            key = (trial, d)
            if key in published:
                assert a["concentration"] * 1e4 == pytest.approx(
                    published[key], rel=1e-6), key
                checked += 1
    assert checked >= 10


def test_prairie_grass_is_the_passive_limit():
    """
    A neutral-density tracer over flat ground: the limit a dense-gas model
    must reduce to, and the only trials here that test it.
    """
    t = load_smedis(SMEDIS / "pg8.xls")
    assert t.dataset == "Prairie Grass"
    arcs = t.arcs()
    assert [a["distance"] for a in arcs] == [50.0, 100.0, 200.0, 400.0, 800.0]
    # monotone decay, as any plume must
    c = [a["concentration"] for a in arcs]
    assert all(b < a for a, b in zip(c, c[1:]))


# ===========================================================================
# sensor-level data
# ===========================================================================
def test_instantaneous_trials_report_dose_and_arrival():
    """
    Thorney Island measures dose with arrival and departure times rather than
    a mean concentration — the instantaneous analogue, and something the
    puff-mode integrator has never been compared against.
    """
    s = load_smedis(SMEDIS / "ti08.xls").sensors()
    assert len(s) > 100
    keys = set().union(*(r.keys() for r in s))
    assert any("dose" in k for k in keys)
    assert any("arriv" in k for k in keys)


def test_continuous_trials_report_mean_concentration():
    s = load_smedis(SMEDIS / "dt1.xls").sensors()
    assert len(s) > 20
    keys = set().union(*(r.keys() for r in s))
    assert any("mean_c" in k for k in keys)
