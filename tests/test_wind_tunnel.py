"""
Wind-tunnel scale, and Froude similarity.

Every other deck in this project is field scale.  These three are four to six
orders of magnitude smaller, which asks a question none of the others can:
does the model depend on absolute size?

These runs sit outside the validated envelope by construction — tunnel
roughness is 3e-5 m and tunnel wind speeds are well below 0.5 m/s — so they
raise `ScopeWarning`.  That is the warning working, not a problem, and it is
filtered here rather than silenced globally.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore::slabx.scope.ScopeWarning")

from slabx.coefficients import COEFFS
from slabx.core.plume import run_dispersion
from slabx.core.source import EvaporatingPool
from slabx.submodels.atmosphere import Atmosphere
from slabx.thermo.base import LegacyThermo, water_backend
from slabx.validation.wind_tunnel import (
    froude_scaling, load_tunnel_trials,
)


def test_only_the_unobstructed_trials_are_marked_applicable():
    """The model has no representation of fences, dikes or slopes."""
    usable = load_tunnel_trials()
    every = load_tunnel_trials(applicable_only=False)
    assert len(usable) == 3
    assert len(every) > len(usable)
    assert all(t.obstruction == "none" for t in usable)
    assert any(t.obstruction != "none" for t in every)


def test_the_model_runs_at_tunnel_scale():
    """
    Rates of 1e-4 kg/s and reference heights of millimetres, against field
    decks of 100 kg/s and metres.  Nothing here asserts the answer is right —
    only that it is finite, bounded and decaying.
    """
    for t in load_tunnel_trials():
        atm = Atmosphere(u_ref=t.u_ref, z_ref=t.z_ref, T=293.0, rh=50.0,
                         z0=t.z0, stability="D")
        src = EvaporatingPool(substance=t.substance(), rate=t.rate,
                              area=0.0113, duration=1e4)
        traj, _ = run_dispersion(src, atm, LegacyThermo(t.substance()),
                                 water_backend(), x_max=10.0,
                                 n_field_steps=60, n_puff_steps=20)
        c = traj.vol_frac
        assert np.all(np.isfinite(c)) and np.all((c >= 0) & (c <= 1))
        assert np.all(np.diff(c[traj.x > 0.5]) <= 1e-12), t.trial


def test_the_model_is_not_froude_similar():
    """
    The finding.  Scale lengths by s, velocities by sqrt(s) and the rate by
    s^2.5 and the dimensionless field should not move; over three decades it
    moves by a factor 2.4.
    """
    c = froude_scaling()
    ratio = c / c[0]
    assert ratio[-1] > 2.0
    assert np.all(np.diff(ratio) > 0)        # monotone in scale


def test_the_hard_coded_reference_height_is_most_of_it():
    """
    EQ 35a evaluates the wind profile at a fixed 4 m.  Scaling that with the
    geometry takes the departure from 2.4 to 1.6, so it accounts for most but
    not all of the scale dependence.
    """
    fixed = froude_scaling()
    scaled = froude_scaling(scale_reference_height=True)
    assert scaled[-1] / scaled[0] < 0.75 * (fixed[-1] / fixed[0])
    assert scaled[-1] / scaled[0] > 1.2      # something else remains


def test_the_reference_height_default_is_ermaks():
    assert COEFFS.h_entrain_ref == 4.0


def test_removing_the_fixed_height_does_not_restore_froude_similarity():
    """
    The pre-registered prescription test, pinned. See `docs/PREREG_froude.md`.

    EQ 35a evaluates the entrainment velocity ratio at a fixed 4 m, an
    absolute length inside a similarity closure, and scaling a release by
    1000 changes the concentration by 2.39 where similarity requires 1.0.
    The registered prediction was that replacing that factor with a
    Richardson-number closure carrying no absolute length would bring the
    ratio inside [1/1.3, 1.3].

    It does not. Both published closures give about 1.5 — barely better than
    merely scaling the fixed height (1.57), and landing within 1 % of each
    other despite one being fitted to canonical flows and the other measured
    in a wind tunnel. The residual belongs to the rest of the model.
    """
    from slabx.coefficients import COEFFS

    scales = (1.0, 10.0, 100.0, 1000.0)

    def ratio(**kw):
        c = froude_scaling(scales, **kw)
        return float(c[-1] / c[0])

    fixed = ratio()
    scaled = ratio(scale_reference_height=True)
    robins = ratio(coeffs=COEFFS.perturb(entrainment_closure="robins",
                                         name="robins"))
    nielsen = ratio(coeffs=COEFFS.perturb(entrainment_closure="nielsen",
                                          name="nielsen"))

    assert fixed > 2.0, fixed
    assert 1.4 < scaled < 1.7, scaled

    # the prediction was < 1.3; both fail, and both agree with each other
    assert robins > 1.3, f"P-F1 unexpectedly passed: {robins:.3f}"
    assert nielsen > 1.3, f"P-F1 unexpectedly passed: {nielsen:.3f}"
    assert abs(robins - nielsen) < 0.05, (robins, nielsen)

    # negative control: nothing to change at unit scale
    for kw in ({}, {"scale_reference_height": True}):
        c = froude_scaling((1.0,), **kw)
        assert float(c[0] / c[0]) == pytest.approx(1.0)


def test_the_better_field_statistic_must_not_be_adopted():
    """
    The fourth time an aggregate pointed opposite to a mechanism test, and
    the first time it happened to a change the authors expected to work.

    The Robins closure gives FAC2 1.00 and MG 0.997 on the ten LNG pool
    trials — better than the original (0.90, 1.252) and better than
    Phast 8.4 (0.90, 0.996). Every point inside a factor of two.

    The mechanism it was introduced to fix was not fixed
    (`test_removing_the_fixed_height_does_not_restore_froude_similarity`).
    An aggregate improving while the mechanism stays broken is the signature
    this project exists to expose, so the default stays as Ermak wrote it.

    This test fails if the default ever changes.
    """
    from slabx.coefficients import COEFFS
    from slabx.validation.lng_pools import compare_lfl
    from slabx.validation.metrics import metrics

    assert COEFFS.entrainment_closure == "slab"

    base = compare_lfl()
    alt = compare_lfl(coeffs=COEFFS.perturb(entrainment_closure="robins",
                                            name="robins"))
    m0 = metrics(base["observed"], base["predicted"])
    m1 = metrics(alt["observed"], alt["predicted"])

    assert m1.FAC2 > m0.FAC2          # it really is better
    assert abs(m1.MG - 1.0) < abs(m0.MG - 1.0)
    assert m1.NMSE < m0.NMSE
    # and P-F2's criterion is blown: the change was meant to be < 5 %
    assert abs(m1.MG / m0.MG - 1.0) > 0.05
