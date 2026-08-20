"""
Where the model has been validated, and where it has not
=======================================================

The dangerous failure for a dispersion model is not a crash — it is a
plausible number produced outside the range anyone has checked.  This module
makes the boundaries explicit and warns when a run crosses one.

The boundaries are not guesses.  Each one is the edge of a comparison
recorded elsewhere in this package, or a limitation established by
measurement:

``|1/L| < 0.15``  — **withdrawn**
    This bound was recorded on the belief that SLAB itself could not run
    outside it.  Comparing against Ermak's Fortran showed otherwise: the
    original reads the correctly spelled variable (SLAB.FOR line 236), and the
    crash belongs to the JavaScript transcription, which has a misspelling of it in that
    one place while using the intended variable in the ten others.  The model has no such
    limit and the bound is no longer applied.

``0.5 <= u_ref <= 12 m/s``
    The span of the validated field trials: 1.7 m/s at Thorney Island 009 to
    9.8 at Maplin Sands 35, with the LNG pool set reaching 10.5.  Below about
    0.5 m/s the cloud barely advects and the plume formulation degenerates.

``z0 <= 0.1 m``
    The validated trials span 2e-4 m (Burro, over water) to 0.04 m (FLADIS,
    over Danish farmland), with Prairie Grass at 6e-3, Thorney Island at
    5e-3 and Desert Tortoise at 3e-3 in between.  The upper end is above the
    0.03 m that 49 CFR 193.2059 prescribes for siting, so the regulatory
    value is covered — FLADIS is in fact where the model does best of the
    three compared.

    Above that, nothing.  Suburban and urban roughness is 0.1 to 2 m, an
    order of magnitude beyond FLADIS, and the objection is not only that the
    number is untested: roughness of that size is *produced by* buildings,
    which the model has no representation for at all.  Raising ``z0`` does
    not stand in for them (see below).

``scale``
    The model is not Froude-similar: scaling a release by 1000 in length
    changes the predicted concentration by 2.4 (see
    `validation/wind_tunnel`).  Wind-tunnel and field results must not be
    transferred through it.

What warnings are for
---------------------
They mark the edge of the evidence, not a mistake.  A user modelling an urban
release may have no alternative; what they must not do is assume the result
carries the same weight as one inside the validated range.  Warnings can be
silenced in the usual way, which is the point of using `warnings` rather than
printing.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

__all__ = ["ScopeWarning", "VALIDATED", "check_scope", "describe_scope"]


class ScopeWarning(UserWarning):
    """Issued when a run leaves the range that has been validated."""


@dataclass(frozen=True)
class Range:
    low: float
    high: float
    reason: str
    evidence: str


#: The validated envelope, with the evidence for each bound.
VALIDATED = {
    "inv_L": Range(
        -2.0, 2.0,
        "a physical bound on the Monin-Obukhov length rather than a "
        "validation bound; the earlier 0.15 limit was an artefact of the "
        "JavaScript transcription, not of SLAB (see module docstring)",
        "SLAB.FOR line 236; tests/test_scope.py",
    ),
    "u_ref": Range(
        0.5, 12.0,
        "the span of the validated trials (1.7 to 10.5 m/s); below ~0.5 m/s "
        "the plume formulation degenerates",
        "validation/lng_pools, validation/thorney_island",
    ),
    "z0": Range(
        1e-5, 0.1,
        "the validated trials span 2e-4 m (Burro, over water) to 0.04 m "
        "(FLADIS, over farmland), which covers the 0.03 m of "
        "49 CFR 193.2059; suburban and urban roughness (0.1-2 m) is "
        "untested, and is produced by obstacles the model cannot represent",
        "validation/lng_pools, validation/fladis, validation/prairie_grass",
    ),
}


def check_scope(*, inv_L: float | None = None, u_ref: float | None = None,
                z0: float | None = None, stacklevel: int = 3) -> list[str]:
    """
    Warn for each validated range the arguments fall outside.

    Returns the messages issued, so a caller that wants to handle them
    itself can do so without parsing warnings.
    """
    issued = []
    for name, value in (("inv_L", inv_L), ("u_ref", u_ref), ("z0", z0)):
        if value is None:
            continue
        r = VALIDATED[name]
        if not (r.low <= value <= r.high):
            msg = (f"{name} = {value:g} is outside the validated range "
                   f"[{r.low:g}, {r.high:g}]: {r.reason}. "
                   f"Evidence: {r.evidence}. The result may still be useful "
                   f"but carries no validation.")
            warnings.warn(msg, ScopeWarning, stacklevel=stacklevel)
            issued.append(msg)
    return issued


def describe_scope() -> str:
    """Human-readable summary, for a README or a run header."""
    lines = ["Validated envelope:"]
    for name, r in VALIDATED.items():
        lines.append(f"  {name:8} [{r.low:g}, {r.high:g}]   {r.reason}")
    lines += [
        "",
        "Outside the envelope entirely (no representation in the model):",
        "  obstacles, dikes, tanks, buildings, urban arrays",
        "    -- measured: an upwind fence changes the EEC propane jet",
        "       concentration by a median factor of 3 at common sensors",
        "       (tests/test_eec.py)",
        "  sloped or undulating terrain",
        "  transfer between wind-tunnel and field scale (not Froude-similar)",
        "  buoyant plume rise for light gases (raises "
        "BuoyantRiseNotImplemented)",
        "",
        "Known failure:",
        "  dense, slow two-phase jets -- the in-place widening runs away; see",
        "  tests/test_fuzz_fortran.py::test_dense_low_momentum_jet",
    ]
    return "\n".join(lines)
