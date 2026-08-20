"""
Concentration field, meander and time averaging
===============================================

Turns the spatially averaged trajectory into what a sensor would record:
a three-dimensional, time-averaged volume fraction (section 2.6, EQ 13, 28,
48-55).

This is an **add-on**.  It consumes a `Trajectory` and imports nothing from
the integrators, so it can be developed, tested and replaced independently of
the core.  Everything it needs is already in the schema.

Three averages, in order
------------------------
1. *Ensemble* — everything SLAB produces is an ensemble average.  A single
   experiment may lie either side of it; this is why validation has to be
   statistical (section 2.6.1).
2. *Spatial* — undone here, by re-imposing similarity profiles on the
   cross-wind, vertical and (in puff mode) down-wind directions.
3. *Time* — applied last, over the averaging time `t_avg`, after the cloud
   width has been widened for meander.

Meander (EQ 48-53)
------------------
The conservation equations are solved with meander switched off, so their
width is the "instantaneous" one.  A real sensor at a fixed point sees the
cloud centreline wander, which broadens the apparent cloud and lowers the
peak.  SLAB adds this afterwards as an extra variance,

    sigma_m^2 = (F(t_av)/F(0))^2 - 1) * sigma_y0^2                EQ 50b
    beta_c^2  = beta^2 + sigma_m^2                                EQ 52b
    B_c       = sqrt(b^2 + 3 beta_c^2)                            EQ 53

with `sigma_y0` the trace-gas plume spread (EQ 51b).  Meander cannot exceed
what the cloud's own duration allows, so the averaging time entering `F` is
capped at the local cloud duration `t_cld = 2 Bx / U` — a short puff barely
meanders, a long one meanders like a plume.

Time averaging (EQ 55)
----------------------
In the plume region the signal is a square wave of length `t_sd`, so the time
average is simply ``min(t_sd / t_av, 1)``.  In the puff region EQ 55b gives
the closed form

    C3 = z1 erf(z1) - z2 erf(z2) + (exp(-z1^2) - exp(-z2^2)) / sqrt(pi)
    z1,2 = (b_x +/- U t_av / 2) / (sqrt(2) beta_x)

which is the down-wind profile convolved with the sampling window.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import erf

from ..coefficients import Coefficients, COEFFS
from ..core.trajectory import Mode, Trajectory
from ..submodels.atmosphere import Atmosphere

__all__ = [
    "ConcentrationField",
    "cloud_duration",
    "meander_width",
    "concentration_field",
    "centreline",
    "height_of_maximum",
]

SQRT3 = math.sqrt(3.0)
SQRT2 = math.sqrt(2.0)
SQRTPI = math.sqrt(math.pi)


# ===========================================================================
def cloud_duration(traj: Trajectory, t_release: float) -> np.ndarray:
    """
    Duration of the cloud passing a fixed point, ``t_cld``.

    ``2 Bx / U`` is the time the cloud takes to pass, but it is enforced
    non-increasing downwind and never shorter than the release
    (SLAB_editcc L3105): a cloud cannot appear to last longer at a distant
    receptor than at a near one.
    """
    Bx = np.where(traj.is_puff, traj.b_half_x, 0.5 * traj.u * t_release)
    t_pass = np.where(traj.u > 0, 2.0 * Bx / np.maximum(traj.u, 1e-12), t_release)

    out = np.empty_like(t_pass)
    out[-1] = max(t_pass[-1], t_release)
    for i in range(len(out) - 2, -1, -1):
        out[i] = max(t_release, min(out[i + 1], t_pass[i]))
    return out


#: What the two meander closures reproduce
#: ----------------------------------------
#: PHMSA's assessment of Phast 8.4 reports arc-wise statistics for each Burro
#: and Coyote trial at both 1 s and 50-140 s averaging.  Its predicted LFL
#: distances change by under 1 % between the two, so the ratio of the two
#: geometric mean biases is close to the *measured* peak-to-mean ratio:
#:
#:     BU08 (E)  1.21     CO6 (D)  1.17     BU09 (D)  1.35
#:     BU03 (C)  1.52     CO5 (C)  2.39     CO3 (C)   2.43
#:
#: Ordered by stability, and spanning a factor 2.08.  Against those six:
#:
#:     closure        range        bias   scatter   log-r
#:     ermak       1.03 - 1.20     0.69     1.30     0.75
#:     sigma_theta 1.07 - 3.17     1.25     1.28     0.76
#:
#: The correlations are indistinguishable at n = 6 and neither is significant.
#: What separates them is the dynamic range: EQ 50b compresses a factor-two
#: variation into a factor of 1.17, while the direction-fluctuation closure
#: reproduces its scale and overshoots.  For a model of *fluctuation* that is
#: the property that matters, so `sigma_theta` is offered — but the bias goes
#: the other way and six trials cannot settle which is better overall.
#:
#: Standard deviation of the horizontal wind direction by stability class
#: [degrees].  Measured quantities — from wind-vane and sonic records, not
#: from dispersion statistics — and the physical origin of plume meander: the
#: plume follows the wind, and over a long averaging window the wind wanders.
#: Conventional values (Slade 1968; Hanna, Briggs & Hosker 1982).  Note the
#: factor of ten from A to F.
SIGMA_THETA = {1: 25.0, 2: 20.0, 3: 15.0, 4: 10.0, 5: 5.0, 6: 2.5}


def sigma_theta(stability: float) -> float:
    """Wind-direction standard deviation [radians], interpolated in class."""
    s = min(max(stability, 1.0), 6.0)
    lo, hi = int(math.floor(s)), int(math.ceil(s))
    if lo == hi:
        return math.radians(SIGMA_THETA[lo])
    return math.radians(SIGMA_THETA[lo]
                        + (s - lo) * (SIGMA_THETA[hi] - SIGMA_THETA[lo]))


def meander_width(
    traj: Trajectory,
    atm: Atmosphere,
    *,
    t_avg: float,
    t_release: float,
    coeffs: Coefficients = COEFFS,
    closure: str = "ermak",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Effective half-width and beta with meander included (EQ 49-53).

    Returns ``(B_c, beta_c)``.  With ``t_avg = 0`` both reduce to the
    instantaneous values, which is the consistency check the reference does
    not make explicit.
    """
    c = coeffs
    t_cld = cloud_duration(traj, t_release)

    def F(t):                                       # EQ 49b / 36a
        return 0.08 * ((t + c.tau_min * np.exp(-t / c.tau_min))
                       / c.t_ref_avg) ** c.p_meander

    F0 = F(0.0)
    r_cf = math.sqrt((atm.u_star / atm.wind_speed(2.0)) / c.c_10)
    inv_La = atm.inv_L_ground
    s_stab = (1.0 - r_cf * c.L_a_ref * inv_La if inv_La < 0.0
              else 1.0 / (1.0 + r_cf * c.L_a_ref * inv_La))

    # EQ 51b: trace-gas plume spread, the solution of EQ 51a
    a_sig = 2.0 * F0 * s_stab / c.a2_horiz
    sigma_y0 = a_sig * (np.sqrt(1.0 + c.a2_horiz * (traj.x - traj.x[0])) - 1.0)

    if closure == "sigma_theta":
        # The plume wanders because the wind does, so the meander spread is
        # the direction fluctuation times the distance travelled, rather than
        # SLAB's own trace-gas plume spread.  Stability then enters through a
        # measured quantity that varies tenfold between classes A and F.
        sigma_y0 = sigma_theta(atm.s) * (traj.x - traj.x[0])
    elif closure != "ermak":
        raise ValueError(f"unknown meander closure {closure!r}")

    t_meander = np.minimum(t_cld, max(t_avg, 0.0))
    ratio = F(t_meander) / F0
    if closure == "sigma_theta":
        sigma_y0 = sigma_theta(atm.s) * (traj.x - traj.x[0])
    elif closure != "ermak":
        raise ValueError(f"unknown meander closure {closure!r}")
    sigma_m2 = np.maximum(ratio**2 - 1.0, 0.0) * sigma_y0**2   # EQ 50b

    beta_c = np.sqrt(traj.beta**2 + sigma_m2)                  # EQ 52b
    B_c = np.sqrt(traj.b_shape**2 + 3.0 * beta_c**2)           # EQ 53
    return B_c, beta_c


# ===========================================================================
def _fcx(traj, t_avg, t_release, coeffs):
    """
    Down-wind and time-averaging factor, EQ 55a (plume) and 55b (puff).

    The two regions have genuinely different forms, not a common core times a
    correction: the plume sees a square wave of length `t_sd`, the puff sees
    its own down-wind profile convolved with the sampling window.  They are
    therefore written out separately, as in SLAB_editcc (L3156, L3172).
    """
    out = np.empty(len(traj))
    h_over_sigma = traj.h / np.maximum(traj.sigma_z, 1e-12)
    b = np.maximum(traj.b_shape, 1e-12)
    tav = max(t_avg, 1e-12)

    pl = traj.is_plume
    if pl.any():
        square = min(t_release / tav, 1.0) if t_avg > 0 else 1.0
        out[pl] = (0.5 / (SQRT2 * SQRTPI)) * (
            traj.b_half[pl] * h_over_sigma[pl] * square * traj.vol_frac[pl] / b[pl]
        )

    pf = np.flatnonzero(traj.is_puff)
    if pf.size:
        bx = np.maximum(traj.b_shape_x[pf], 1e-12)
        bex = np.maximum(traj.beta_x[pf], 1e-12)
        half = 0.5 * traj.u[pf] * tav
        z1 = (bx + half) / (SQRT2 * bex)
        z2 = (bx - half) / (SQRT2 * bex)
        C3 = (z1 * erf(z1) - z2 * erf(z2)
              + (np.exp(-z1**2) - np.exp(-z2**2)) / SQRTPI)
        num = (0.5 / SQRTPI) * bex * traj.b_half_x[pf] * traj.b_half[pf] \
            * h_over_sigma[pf]
        den = bx * traj.u[pf] * tav * b[pf]
        # A stalled or fully-dispersed puff can leave the denominator at zero
        # or produce inf/inf in C3 far downwind; report zero rather than nan,
        # which would silently poison every statistic computed from the field.
        out[pf] = np.divide(num * C3 * traj.vol_frac[pf], den,
                            out=np.zeros_like(den), where=den > 0)
    out[~np.isfinite(out)] = 0.0
    return out


def _fcy(y, b, beta_c):
    """Cross-wind shape factor; 2 erf(b / (sqrt2 beta)) on the centreline."""
    beta_c = np.maximum(beta_c, 1e-12)
    return erf((y + b) / (SQRT2 * beta_c)) - erf((y - b) / (SQRT2 * beta_c))


def height_of_maximum(z_c, sigma, *, tol: float = 1e-12, max_iter: int = 30):
    """
    Height at which the vertical profile peaks, ``z_m`` (SLAB_editcc L3197).

    With ground reflection the profile is the sum of two Gaussians centred at
    ``+z_c`` and ``-z_c``.  Its maximum satisfies

        z = z_c (1 - E) / (1 + E),      E = exp(-2 z z_c / sigma^2)

    which has only the trivial root ``z = 0`` while ``z_c <= sigma`` — a cloud
    that reaches the ground peaks *at* the ground — and a positive root once
    the cloud has lifted clear.  The reference runs five Newton passes with no
    convergence test; this iterates to tolerance.

    Reporting the concentration here rather than at a fixed receptor height is
    what "maximum concentration along the centreline" means in SLAB's output.
    """
    z_c = np.asarray(z_c, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    z = np.where(z_c > sigma, z_c, 0.0)
    lifted = z_c > sigma
    if not np.any(lifted):
        return z
    for _ in range(max_iter):
        E = np.exp(-2.0 * z_c * z / sigma**2, where=lifted, out=np.ones_like(z))
        g = z_c * (1.0 - E) / (1.0 + E)
        gp = (4.0 * z_c**2 / sigma**2) * E / (1.0 + E) ** 2
        step = np.where(lifted, (g - z * gp) / (1.0 - gp) - z, 0.0)
        z = z + step
        if np.max(np.abs(step)) < tol:
            break
    return np.where(lifted, np.maximum(z, 0.0), 0.0)


def _fcz(z, z_c, sigma):
    """Vertical shape factor with ground reflection; 2 at the centre."""
    sigma = np.maximum(sigma, 1e-12)
    return (np.exp(-0.5 * ((z - z_c) / sigma) ** 2)
            + np.exp(-0.5 * ((z + z_c) / sigma) ** 2))


@dataclass(frozen=True)
class ConcentrationField:
    """Time-averaged volume fraction and the geometry it was built from."""

    x: np.ndarray
    """Down-wind distance [m]."""
    peak: np.ndarray
    """Centreline (y = 0) time-averaged volume fraction at `z`."""
    z: np.ndarray
    """Receptor height used at each x [m]."""
    t_avg: float
    """Concentration averaging time [s]."""
    B_c: np.ndarray
    """Effective half-width including meander [m] (EQ 53)."""
    beta_c: np.ndarray
    b_shape: np.ndarray
    z_c: np.ndarray
    sigma_z: np.ndarray
    t_cld: np.ndarray
    """Local cloud duration [s]."""

    def at_y(self, y) -> np.ndarray:
        """Cross-wind profile at every x, normalised to the centreline."""
        y = np.atleast_1d(np.asarray(y, dtype=float))[:, None]
        b = np.maximum(self.b_shape, 1e-12)[None, :]
        num = _fcy(y, b, self.beta_c[None, :])
        den = _fcy(0.0, np.maximum(self.b_shape, 1e-12), self.beta_c)
        return np.squeeze(self.peak[None, :] * num / den)

    def distance_to(self, level: float) -> float:
        """
        Furthest down-wind distance at which `level` is exceeded.

        Interpolated in log-concentration, which is how LFL and toxic
        endpoint distances are normally reported.
        """
        above = self.peak >= level
        if not above.any():
            return float("nan")
        i = int(np.flatnonzero(above)[-1])
        if i == len(self.peak) - 1:
            return float(self.x[i])
        c1, c2 = self.peak[i], self.peak[i + 1]
        if c1 <= 0 or c2 <= 0 or c1 == c2:
            return float(self.x[i])
        f = (math.log(c1) - math.log(level)) / (math.log(c1) - math.log(c2))
        return float(self.x[i] + f * (self.x[i + 1] - self.x[i]))


def concentration_field(
    traj: Trajectory,
    atm: Atmosphere,
    *,
    z: float | str = 0.0,
    t_avg: float = 0.0,
    t_release: float | None = None,
    coeffs: Coefficients = COEFFS,
    meander_closure: str = "ermak",
) -> ConcentrationField:
    """
    Time-averaged centreline volume fraction at height `z` (EQ 13/28 + 55).

    Parameters
    ----------
    z : receptor height [m], or ``"max"`` to evaluate at the height of
        maximum concentration (`height_of_maximum`).  Field data is usually
        reported at a fixed height; SLAB's own "maximum concentration along
        the centreline" table uses ``"max"``.  The two coincide only for a
        grounded cloud.
    t_avg : concentration averaging time [s].  ``0`` gives the instantaneous
        (meander-free) value.
    t_release : release duration [s]; taken from ``traj.meta`` when omitted.

    Notes
    -----
    The returned `peak` is the **arc-wise maximum** at that height — the
    quantity dispersion-model validation databases report — because the
    cross-wind profile is maximal on the centreline.
    """
    if t_release is None:
        t_release = float(traj.meta.get("release_duration", math.inf))
        if not math.isfinite(t_release):
            t_release = float(traj.t[-1])

    B_c, beta_c = meander_width(traj, atm, t_avg=t_avg, t_release=t_release,
                                closure=meander_closure,
                                coeffs=coeffs)
    sigma = np.maximum(traj.sigma_z, 1e-12)
    b = np.maximum(traj.b_shape, 1e-12)

    # EQ 13 / EQ 28 factorise into three independent shape factors.
    # Meander enters only through `beta_c` in the cross-wind factor: a
    # wandering centreline spreads the same material over a wider arc.
    z_eval = (height_of_maximum(traj.z_c, sigma) if isinstance(z, str)
              else np.full(len(traj), float(z)))
    fcx = _fcx(traj, t_avg, t_release, coeffs)
    peak = fcx * _fcy(0.0, b, beta_c) * _fcz(z_eval, traj.z_c, sigma)
    peak = np.clip(peak, 0.0, 1.0)

    return ConcentrationField(
        x=traj.x.copy(), peak=peak, z=z_eval, t_avg=t_avg,
        B_c=B_c, beta_c=beta_c, b_shape=traj.b_shape.copy(),
        z_c=traj.z_c.copy(), sigma_z=traj.sigma_z.copy(),
        t_cld=cloud_duration(traj, t_release),
    )


def centreline(
    traj: Trajectory, atm: Atmosphere, *, z: float = 0.0, t_avg: float = 0.0, **kw
) -> np.ndarray:
    """Shorthand for ``concentration_field(...).peak``."""
    return concentration_field(traj, atm, z=z, t_avg=t_avg, **kw).peak
