"""
Input validation and the validated envelope.

Two kinds of protection, both aimed at the same failure: a plausible number
produced from an input nobody checked.

The first is validation — inputs that are physically impossible are refused.
The second is scope — inputs that are possible but outside the range this
package has compared against measurements produce a warning, not a refusal,
because a user may have no alternative and is entitled to the result as long
as they know what it is worth.
"""

import warnings

import numpy as np

import pytest

from slabx.core.source import EvaporatingPool
from slabx.scope import VALIDATED, ScopeWarning, check_scope, describe_scope
from slabx.submodels.atmosphere import Atmosphere
from slabx.thermo.base import Substance

METHANE = Substance(name="LNG", mw=0.016043, cp_vapour=2238.0,
                    cp_liquid=3348.5, dh_vap=509900.0, T_boil=111.7,
                    rho_liquid=424.1)

OK = dict(u_ref=3.0, z_ref=10.0, T=290.0, z0=0.01, stability="D")


# ===========================================================================
# impossible inputs are refused
# ===========================================================================
@pytest.mark.parametrize("bad,match", [
    (dict(u_ref=-3.0), "wind speed"),
    (dict(z0=-0.01), "roughness"),
    (dict(T=0.0), "temperature"),
    (dict(rh=150.0), "humidity"),
    (dict(rh=-5.0), "humidity"),
    (dict(z_ref=0.001), "reference height"),
    (dict(p=0.0), "pressure"),
])
def test_impossible_atmospheres_are_refused(bad, match):
    """
    Each of these used to produce a number.  A negative wind speed gave a
    negative friction velocity and a complete run; a reference height below
    the roughness length inverted the log profile and returned u* = 33 m/s.
    Neither raised anything.
    """
    with pytest.raises(ValueError, match=match):
        Atmosphere(**{**OK, **bad})


def test_calm_is_allowed():
    """Zero wind is degenerate but not impossible, and F-stability siting
    calculations sit close to it."""
    assert Atmosphere(**{**OK, "u_ref": 0.0}).u_star == 0.0


def test_impossible_sources_are_refused():
    for bad in (dict(rate=-10.0, area=100.0, duration=100.0),
                dict(rate=10.0, area=0.0, duration=100.0),
                dict(rate=10.0, area=-5.0, duration=100.0),
                dict(rate=10.0, area=100.0, duration=-50.0)):
        with pytest.raises(ValueError):
            EvaporatingPool(substance=METHANE, **bad)


# ===========================================================================
# the validated envelope
# ===========================================================================
def test_every_bound_carries_its_evidence():
    """A boundary without a reason is a guess; these are not."""
    for name, r in VALIDATED.items():
        assert r.low < r.high, name
        assert r.reason.strip() and r.evidence.strip(), name


def test_the_stability_bound_was_withdrawn():
    """
    Regression guard for a claim that had to be retracted.

    The envelope originally carried |1/L| < 0.15 on the grounds that SLAB
    itself could not run outside it: the JavaScript reference throws there,
    reading an undeclared a misspelling of it.  Comparing against Ermak's Fortran showed
    the original reads the correctly spelled variable (SLAB.FOR line 236) and
    uses the intended variable in all ten places, so the crash is a transcription defect
    and the model has no such limit.

    The bound is now a physical one on the Monin-Obukhov length, and a
    stability class of E or F must not trip it.
    """
    assert VALIDATED["inv_L"].high >= 1.0
    assert "transcription" in VALIDATED["inv_L"].reason
    assert "SLAB.FOR" in VALIDATED["inv_L"].evidence


@pytest.mark.parametrize("kw", [
    dict(u_ref=1.94, z_ref=3.0, T=306.0, z0=2e-4, stability="E"),   # Burro 8
    dict(u_ref=9.8, z_ref=10.0, T=290.0, z0=3e-4, stability="D"),   # MS35
    dict(u_ref=1.7, z_ref=10.0, T=288.0, z0=0.005, stability="D"),  # TI 009
])
def test_validated_trials_are_silent(kw):
    """
    The envelope has to contain the trials it was drawn from, or the warning
    becomes noise and users learn to ignore it.
    """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Atmosphere(**kw)
    assert not [x for x in w if issubclass(x.category, ScopeWarning)]


@pytest.mark.parametrize("kw,expect", [
    (dict(u_ref=3.0, z_ref=10.0, T=290.0, z0=0.003, inv_L=5.0), "inv_L"),
    (dict(u_ref=3.0, z_ref=10.0, T=290.0, z0=1.0, stability="D"), "z0"),
    (dict(u_ref=0.1, z_ref=10.0, T=290.0, z0=0.01, stability="F"), "u_ref"),
    (dict(u_ref=20.0, z_ref=10.0, T=290.0, z0=0.01, stability="D"), "u_ref"),
])
def test_out_of_scope_inputs_warn_but_still_run(kw, expect):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        atm = Atmosphere(**kw)
    issued = [x for x in w if issubclass(x.category, ScopeWarning)]
    assert len(issued) >= 1
    assert expect in str(issued[0].message)
    assert atm.rho > 0                       # and the run is still usable


def test_a_stability_class_suppresses_the_inv_L_check():
    """
    Specifying a class rather than a Monin-Obukhov length takes a different
    path in the reference, one that does not hit the undeclared identifier.
    """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Atmosphere(u_ref=3.0, z_ref=10.0, T=290.0, z0=0.003, stability="F")
    assert not [x for x in w if "inv_L" in str(x.message)]


def test_check_scope_returns_what_it_issued():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert check_scope(u_ref=0.1, z0=1.0) != []
        assert check_scope(u_ref=3.0, z0=0.01) == []


def test_the_summary_names_what_is_out_of_scope_entirely():
    text = describe_scope()
    for phrase in ("obstacles", "terrain", "Froude", "BuoyantRise",
                   "dense, slow two-phase jets"):
        assert phrase in text


def test_zero_wind_is_refused_with_a_reason_not_a_zero_division():
    """
    `Atmosphere` admits zero wind as degenerate but possible
    (`test_zero_wind_is_allowed`), and then `u* = 0`. The initial cloud
    height divides by `u*`, so a run at `u_ref = 0` used to surface
    `ZeroDivisionError: float division by zero` from inside the source
    construction -- an inconsistency between what the scope statement
    permits and what the integrator accepts.

    SLAB has no still-air limit, so refusing is right; doing it with a
    reason is what was missing.
    """
    from slabx.core.plume import run_dispersion
    from slabx.core.source import EvaporatingPool
    from slabx.thermo.base import LegacyThermo, Substance, water_backend

    sub = Substance(name="LNG", mw=0.016043, cp_vapour=2238.0,
                    cp_liquid=3348.5, dh_vap=509900.0, T_boil=111.7,
                    rho_liquid=424.1)
    src = EvaporatingPool(substance=sub, rate=116.93, area=700.0,
                          duration=107.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ScopeWarning)
        calm = Atmosphere(u_ref=0.0, z_ref=3.0, T=290.0, rh=50.0, z0=2e-4,
                          stability="E")
        assert calm.u_star == 0.0

        with pytest.raises(ValueError, match="friction velocity"):
            run_dispersion(src, calm, LegacyThermo(sub), water_backend(),
                           x_max=500.0, n_puff_steps=40)

        # and the smallest wind the envelope warns about still runs
        faint = Atmosphere(u_ref=0.1, z_ref=3.0, T=290.0, rh=50.0, z0=2e-4,
                           stability="E")
        traj, _ = run_dispersion(src, faint, LegacyThermo(sub),
                                 water_backend(), x_max=500.0,
                                 n_puff_steps=40)
        assert np.all(np.isfinite(traj.h))
