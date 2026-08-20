"""
EEC propane jets: a dataset the model cannot fairly be tested on.

Nine trials from the EEC/Lathen campaign — pressurised propane through 4 to
15.5 mm nozzles at 0.11 to 3.0 kg/s.  They looked like the natural extension
of the two-phase jet evidence to a third substance.  They are not usable as a
quantitative test, and the reasons are worth recording so that nobody repeats
the attempt.

Reason 1: most of the trials have fences
-----------------------------------------
Five of the nine carry one or two upwind fences, which the model has no
representation of.  The dataset is designed as matched pairs — the same
release with and without an obstruction — and comparing them shows the fence
is not a second-order effect: the concentration at common sensors changes by
a median factor of 0.32 for EEC170/171, and individual sensors move by two
orders of magnitude.

That is a useful negative result in itself.  It is direct measured evidence
that the "no obstacles" limit in `scope.describe_scope()` is a real
restriction and not a formality.

Reason 2: the coordinate frame cannot be recovered
---------------------------------------------------
The sensor coordinates are in a site frame whose relation to the wind is not
recoverable from the file.  Sensor x-values are negative in every trial while
the release is at the origin, and the documented wind direction, ideal wind
direction and x-axis orientation do not reconcile: rotating each trial to
minimise the concentration-weighted cross-wind offset gives 147 degrees for
EEC170 but 63 to 64 for EEC550 and EEC560, and for EEC550 the best rotation
still leaves sensors upwind of the source.

Without the frame, downwind distance is unknown, and a concentration-versus-
distance comparison is meaningless.  Using radial distance from the source
instead gives a consistent 3 to 5-fold under-prediction — but sweeping the
two free source assumptions to their limits (expansion ratio 4 to 100,
liquid fraction 0.5 to 0.95) never closes even half the gap, which says the
discrepancy is in the geometry rather than the physics.

What is kept
------------
The conditions and sensor readings are registered, and the fence comparison
is run as a test, because it is measured evidence about the model's scope.
The concentration comparison is not.
"""

import csv
from pathlib import Path

import numpy as np
import pytest
from slabx.validation._data_access import require

DATA = (Path(__file__).resolve().parents[1] / "src" / "slabx" / "validation"
        / "data")


def _read(name):
    with open(require(name), newline="") as fh:
        return list(csv.DictReader(fh))


def _conditions():
    return {r["trial"]: r for r in _read("eec_conditions.csv")}


def test_nine_trials_five_with_fences():
    c = _conditions()
    assert len(c) == 9
    fenced = [t for t, r in c.items() if float(r["n_fences"]) > 0]
    assert len(fenced) == 5
    assert all(r["substance"] == "propane" for r in c.values())


def test_the_rates_span_a_factor_of_thirty():
    rates = [float(r["rate_kg_s"]) for r in _conditions().values()]
    assert min(rates) < 0.15 and max(rates) > 2.9


def test_fences_change_the_measured_concentration_substantially():
    """
    Matched pairs, same release, fence or no fence.  This is the measured
    justification for excluding obstructed geometries from the validated
    scope: the effect is a factor of three at the median and two orders of
    magnitude at individual sensors, not a correction.
    """
    obs = _read("eec_sensors.csv")

    def field(trial):
        return {(r["x_m"], r["y_m"], r["z_m"]): float(r["mean_C_pct"])
                for r in obs if r["trial"] == trial}

    a, b = field("EEC170"), field("EEC171")
    common = [k for k in a if k in b and a[k] > 1e-4 and b[k] > 1e-4]
    assert len(common) > 20
    ratio = np.array([b[k] / a[k] for k in common])
    assert np.median(ratio) < 0.5                 # a third of the concentration
    assert ratio.max() / ratio.min() > 4.0        # and highly variable


def test_the_coordinate_frame_does_not_reconcile():
    """
    Regression guard against re-attempting a distance comparison.  Rotating
    each trial to best align its plume with a downwind axis gives angles that
    differ by more than eighty degrees between trials of the same campaign,
    and for EEC550 the best available rotation still places sensors upwind of
    the release.
    """
    import math

    obs = _read("eec_sensors.csv")
    best = {}
    for trial in ("EEC170", "EEC550", "EEC560"):
        v = [(float(r["x_m"]), float(r["y_m"]), float(r["mean_C_pct"]))
             for r in obs if r["trial"] == trial]
        x = np.array([p[0] for p in v])
        y = np.array([p[1] for p in v])
        c = np.array([p[2] for p in v])
        m = c > 1e-3
        found = None
        for deg in range(0, 360):
            t = math.radians(deg)
            cross = -x * math.sin(t) + y * math.cos(t)
            score = float(np.average(np.abs(cross[m]), weights=c[m]))
            if found is None or score < found[0]:
                found = (score, deg)
        best[trial] = found[1]
    spread = max(best.values()) - min(best.values())
    assert spread > 60.0


def test_sensors_are_registered_even_though_unused():
    """
    Kept in the tree deliberately.  If the site geometry is ever recovered —
    the campaign reports would have it — the comparison becomes possible, and
    re-extracting the data should not be necessary.
    """
    obs = _read("eec_sensors.csv")
    assert len(obs) > 250
    assert {r["trial"] for r in obs} == set(_conditions())


# ===========================================================================
# the whole SMEDIS set, and why most of it cannot be used
# ===========================================================================
def test_the_inventory_records_a_reason_for_every_exclusion():
    """
    Twenty-eight SMEDIS trial files were reviewed.  Nine are usable; the
    other nineteen are excluded, and each carries the reason in the
    inventory rather than being silently dropped.

    The exclusions are not arbitrary: obstacles change the measured
    concentration by a factor of three (tested above), surface slopes of 4 to
    12 per cent have no representation in a flat-terrain model, and one
    wind-tunnel trial specifies a roughness of 5 m — two orders of magnitude
    above anything validated.
    """
    rows = _read("smedis_inventory.csv")
    assert len(rows) == 28
    usable = [r for r in rows if r["usable"] == "True"]
    assert len(usable) == 9
    for r in rows:
        if r["usable"] != "True":
            assert r["exclusion_reason"].strip(), r["file"]


def test_exclusions_group_into_three_causes():
    rows = _read("smedis_inventory.csv")
    reasons = [r["exclusion_reason"] for r in rows if r["usable"] != "True"]
    assert sum("obstacle" in x for x in reasons) >= 8
    assert sum("slope" in x for x in reasons) == 3
    assert sum("coordinate frame" in x for x in reasons) >= 4
    assert sum("roughness" in x for x in reasons) == 1
