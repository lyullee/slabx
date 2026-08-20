"""
Added mass in the vertical momentum balance
===========================================

A cloud that rises has to push ambient air out of the way, and the momentum
given to that air is part of what the buoyancy has to accelerate.  SLAB's
vertical equation (EQ 6) does not carry it, so a lofting cloud responds only
to its own mass.

Why the shape matters so much
-----------------------------
For a compact cloud the correction is modest.  For the flat pancake a dense
release actually becomes it is not: the air an oblate body displaces as it
moves along its short axis scales with its *width*, not its depth.  Treating
the cloud as an oblate spheroid of horizontal semi-axis B and vertical
semi-axis h/2 moving broadside, the added mass is ``(8/3) rho_a B^3`` against
a cloud volume of ``(2/3) pi B^2 h``, so

    k_added = C_A (rho_a / rho) (B / h),      C_A = 2 / pi

and the effective inertia is ``(1 + k_added)`` times the cloud's own.  A
cloud fifty times wider than it is deep is an order of magnitude harder to
lift than SLAB thinks.

What it changes
---------------
Without it, `slabx` lifts a buoyant hydrogen cloud to 259 m whether it starts
2 m wide or 100 m wide — the rise height varies by 0.2 % across a fiftyfold
change in width, which is the signature of the term being absent rather than
small.  DRIFT added exactly this term, reporting that it "has the effect of
suppressing lift-off for wide clouds".

The predictions, registered before the comparison
--------------------------------------------------
A1  **No effect on a cloud that never leaves the ground.**  The dense-gas
    field trials must not move at all; this is the negative control.
A2  **Suppression scales with the aspect ratio.**  A cloud wide relative to
    its depth must be held down much more than a compact one, so the rise
    height must become a strong function of width where it currently is not.
A3  **The compact limit is recovered.**  As ``B -> h`` the correction tends
    to a factor of order one, not to zero and not to something large — the
    term must not quietly rescale every buoyant release.
A4  **Direction: rise height falls, never rises.**

A2 is the discriminating one.  A term that suppressed all lift-off equally
would be a fudge on the buoyancy, not added mass.
"""

from __future__ import annotations

__all__ = ["C_ADDED_OBLATE", "added_mass_factor", "PREDICTIONS"]

#: Added-mass coefficient for an oblate body moving along its short axis,
#: from the potential-flow solution for a flat disc: 2/pi.
C_ADDED_OBLATE = 2.0 / 3.141592653589793

PREDICTIONS = (
    ("A1", "no effect on a cloud that never lofts (negative control)"),
    ("A2", "suppression scales with width-to-depth ratio"),
    ("A3", "the compact limit gives a factor of order one"),
    ("A4", "rise height falls, never rises"),
)


def added_mass_factor(b_half: float, h: float, rho_cloud: float,
                      rho_ambient: float, *,
                      coefficient: float = C_ADDED_OBLATE,
                      cap: float = 50.0) -> float:
    """
    ``k_added`` such that the effective vertical inertia is ``1 + k_added``.

    Capped because the oblate-spheroid result diverges as the cloud flattens,
    while a real cloud that thin is being sheared apart rather than rising as
    a body.  The cap bounds the model, not the physics.
    """
    if h <= 0.0 or rho_cloud <= 0.0 or b_half <= 0.0:
        return 0.0
    return min(coefficient * (rho_ambient / rho_cloud) * (b_half / h), cap)
