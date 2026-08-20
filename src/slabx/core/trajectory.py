"""
Trajectory: the output contract of the dispersion core
=====================================================

Everything downstream of the conservation equations — three-dimensional
concentration fields, meander, time averaging, dose, contours, uncertainty
studies, plotting — consumes a `Trajectory` and nothing else.  The core does
not import from `post`, `study` or `viz`; the dependency runs one way only.

**This schema is a contract.** Add-ons written against it must keep working,
so fields are only ever added, never removed or renamed.  Adding is cheap:
every quantity here is already computed while integrating, so recording it
costs nothing and saves add-ons from recomputing (or, worse, from
recomputing it slightly differently).

Contents
--------
The 26 arrays stored by the reference implementation (``SLAB_store``,
SLAB.FOR L2941 (subroutine store)) are all present, under readable names:

    JS      here                    JS      here
    ------  ----------------------  ------  ----------------------
    xp      x                       cmdap   mass_frac_dry_air
    bbp     b_half                  cmwp    mass_frac_water
    bp      b_shape                 cmwvp   mass_frac_water_vapour
    betap   beta                    cmevp   mass_frac_emission_vapour
    hp      h                       uap     u_ambient_mean
    zcp     z_c                     wcp     w_c
    up      u                       vgp     v_g
    tp      T                       ugp     u_g
    rhop    rho                     wp      w_entrain
    cmp     mass_frac               vp      v_entrain
    cvp     vol_frac                vxp     v_x_entrain
    timp    t                       qintp   mass_in_cloud
    bbxp    b_half_x                bxp     b_shape_x
    betaxp  beta_x

plus diagnostics that the reference discards but that the add-ons need:
in-cloud friction velocity, the stability function, the Monin-Obukhov
length, the specific heat, the liquid fractions and the flux terms.

Independent variable
--------------------
`x` and `t` are *both* always present and both monotonic.  In plume mode `x`
is the integration variable and `t` follows from EQ 29b; in puff mode `t` is
the integration variable and `x` is the centre-of-mass position `X_c`.  A
trajectory may contain both phases, joined at the transition (EQ 30); `mode`
records which set of equations produced each row.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, fields
from typing import Any, Mapping

import numpy as np

__all__ = ["Mode", "Trajectory", "SCHEMA_VERSION"]

#: Bump on any backwards-incompatible change.  Recorded in `meta`.
SCHEMA_VERSION = 1


class Mode(enum.IntEnum):
    """Which set of conservation equations produced a row."""

    PLUME = 0   #: steady-state, cross-wind averaged (EQ 1-10)
    PUFF = 1    #: transient, volume averaged (EQ 15-26)


# ---------------------------------------------------------------------------
# field groups, used for validation and for `to_dataframe`
# ---------------------------------------------------------------------------
_COORDS = ("x", "t")

_GEOMETRY = (
    "h", "z_c", "b_half", "b_shape", "beta",
    "b_half_x", "b_shape_x", "beta_x", "sigma_z",
)

_STATE = ("u", "T", "rho", "cp", "mass_frac", "vol_frac")

_VELOCITIES = ("v_g", "u_g", "w_c", "u_ambient_mean")

_SPECIES = (
    "mass_frac_dry_air", "mass_frac_water", "mass_frac_water_vapour",
    "mass_frac_emission_vapour", "mass_frac_water_liquid",
    "mass_frac_emission_liquid",
)

_ENTRAIN = ("w_entrain", "v_entrain", "v_x_entrain")

_DIAGNOSTIC = (
    "u_star_cloud", "phi_h", "inv_L_cloud",
    "f_u", "f_v", "f_w", "f_t",
    "mass_in_cloud", "R_flux", "rained_out",
)

ARRAY_FIELDS = _COORDS + _GEOMETRY + _STATE + _VELOCITIES + _SPECIES \
    + _ENTRAIN + _DIAGNOSTIC


@dataclass(frozen=True)
class Trajectory:
    """
    Spatially-averaged cloud properties along the dispersion path.

    All arrays have the same length ``n`` and are aligned row-by-row.  Units
    are SI throughout: m, s, K, kg/m^3, m/s, J/(kg K); mass and volume
    fractions are dimensionless in [0, 1].
    """

    # -- coordinates ---------------------------------------------------
    x: np.ndarray
    """Down-wind distance [m].  Puff rows carry the centre-of-mass X_c."""

    t: np.ndarray
    """Travel time from release [s]."""

    mode: np.ndarray
    """`Mode` per row (int8)."""

    # -- cloud geometry -------------------------------------------------
    h: np.ndarray
    """Cloud height [m] (EQ 14, h^2 = 3 sigma_z^2)."""

    z_c: np.ndarray
    """Centre-height parameter [m] (EQ 9). 0 for a grounded cloud."""

    b_half: np.ndarray
    """Cloud half-width B [m] (EQ 7)."""

    b_shape: np.ndarray
    """Half-width shape parameter b [m] (EQ 8)."""

    beta: np.ndarray
    """Gaussian half-width parameter beta [m]; B^2 = b^2 + 3 beta^2 (EQ 13)."""

    b_half_x: np.ndarray
    """Cloud half-length Bx [m] (EQ 25). 1.0 in plume mode."""

    b_shape_x: np.ndarray
    """Half-length shape parameter bx [m] (EQ 26)."""

    beta_x: np.ndarray
    """Gaussian half-length parameter beta_x [m] (EQ 31b)."""

    sigma_z: np.ndarray
    """Vertical spread sigma [m] used by the profile function C2 (EQ 13)."""

    # -- averaged state --------------------------------------------------
    u: np.ndarray
    """Cloud down-wind velocity U [m/s] (EQ 4)."""

    T: np.ndarray
    """Cloud temperature [K] (EQ 41)."""

    rho: np.ndarray
    """Cloud density [kg/m^3] (EQ 10)."""

    cp: np.ndarray
    """Cloud specific heat [J/(kg K)] (EQ 41)."""

    mass_frac: np.ndarray
    """Mass concentration m of released material (EQ 1a/15a)."""

    vol_frac: np.ndarray
    """Volume concentration C (EQ 12).  This is the primary output."""

    # -- velocities ------------------------------------------------------
    v_g: np.ndarray
    """Cross-wind gravity-flow velocity V_g [m/s] (EQ 5a)."""

    u_g: np.ndarray
    """Down-wind gravity-flow velocity U_g [m/s] (EQ 19a). Puff only."""

    w_c: np.ndarray
    """Rate of change of the centre height [m/s] (EQ 6/9)."""

    u_ambient_mean: np.ndarray
    """Depth-averaged ambient wind over the cloud [m/s] (EQ 4c)."""

    # -- species ---------------------------------------------------------
    mass_frac_dry_air: np.ndarray
    """Dry-air mass fraction (EQ 40b)."""

    mass_frac_water: np.ndarray
    """Total water mass fraction (EQ 40a)."""

    mass_frac_water_vapour: np.ndarray
    """Water present as vapour (EQ 40c)."""

    mass_frac_emission_vapour: np.ndarray
    """Released material present as vapour (EQ 40d)."""

    mass_frac_water_liquid: np.ndarray
    """Water present as droplets (EQ 40e)."""

    mass_frac_emission_liquid: np.ndarray
    """Released material present as droplets (EQ 40f)."""

    # -- entrainment (reduced; multiply by sqrt(3) for W_e, V_e) ---------
    w_entrain: np.ndarray
    """Reduced vertical entrainment velocity (EQ 35a)."""

    v_entrain: np.ndarray
    """Reduced cross-wind entrainment velocity (EQ 36a)."""

    v_x_entrain: np.ndarray
    """Reduced down-wind entrainment velocity (EQ 36b)."""

    # -- diagnostics -----------------------------------------------------
    u_star_cloud: np.ndarray
    """In-cloud friction velocity U* [m/s] (EQ 35e)."""

    phi_h: np.ndarray
    """Stability damping function phi_h at the cloud top (EQ 35c)."""

    inv_L_cloud: np.ndarray
    """In-cloud inverse Monin-Obukhov length [1/m] (EQ 35d)."""

    f_u: np.ndarray
    """Down-wind momentum flux [N/m] (EQ 38)."""

    f_v: np.ndarray
    """Cross-wind momentum flux [N/m] (EQ 39)."""

    f_w: np.ndarray
    """Vertical momentum flux [N/m]."""

    f_t: np.ndarray
    """Ground heat flux [J/(m s)] (EQ 37)."""

    mass_in_cloud: np.ndarray
    """Released mass contained in the cloud [kg] (EQ 29b integrand)."""

    R_flux: np.ndarray
    """R = rho U B h (plume) or rho Bx By h (puff); the master variable."""

    rained_out: np.ndarray
    """
    Released material deposited on the ground, in the same units as R.

    Zero unless the rainout submodel is active.  Recorded because the pool it
    forms is a hazard in its own right — it evaporates on its own timescale —
    and because ``airborne + rained_out`` is the check that the submodel
    conserves mass.
    """

    # -- provenance ------------------------------------------------------
    meta: Mapping[str, Any] = field(default_factory=dict)
    """
    Everything needed to reproduce the run: scenario, source, atmosphere,
    the full `Coefficients` dict, the thermodynamic backend, the integrator
    settings and its tolerances, the legacy flags, and `schema_version`.
    A Trajectory without provenance cannot enter an ablation study.
    """

    # ------------------------------------------------------------------
    def __post_init__(self):
        n = len(self.x)
        for f in fields(self):
            if f.name == "meta":
                continue
            arr = np.asarray(getattr(self, f.name))
            if arr.ndim != 1:
                raise ValueError(f"{f.name}: expected 1-D, got shape {arr.shape}")
            if len(arr) != n:
                raise ValueError(
                    f"{f.name}: length {len(arr)} != len(x) = {n}"
                )
            object.__setattr__(self, f.name, arr)
        if n == 0:
            raise ValueError("empty trajectory")
        if np.any(np.diff(self.x) < 0):
            raise ValueError("x must be non-decreasing")
        if np.any(np.diff(self.t) < 0):
            raise ValueError("t must be non-decreasing")
        meta = dict(self.meta)
        meta.setdefault("schema_version", SCHEMA_VERSION)
        object.__setattr__(self, "meta", meta)

    # -- basics ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self.x)

    def __repr__(self) -> str:
        return (
            f"<Trajectory n={len(self)} "
            f"x=[{self.x[0]:.3g}, {self.x[-1]:.3g}] m "
            f"t=[{self.t[0]:.3g}, {self.t[-1]:.3g}] s "
            f"modes={sorted({Mode(m).name for m in self.mode})}>"
        )

    # -- construction ---------------------------------------------------
    @classmethod
    def from_rows(cls, rows, meta: Mapping[str, Any] | None = None) -> "Trajectory":
        """
        Build from an iterable of per-step mappings.

        Missing optional fields are filled with NaN so that a partially
        instrumented core still produces a valid Trajectory; missing
        *required* fields raise.
        """
        rows = list(rows)
        if not rows:
            raise ValueError("no rows")
        required = set(_COORDS) | {"mode", "h", "b_half", "u", "T", "rho",
                                   "mass_frac", "vol_frac"}
        missing = required - set(rows[0])
        if missing:
            raise KeyError(f"missing required field(s): {sorted(missing)}")

        data = {}
        for name in ARRAY_FIELDS:
            data[name] = np.array([r.get(name, np.nan) for r in rows], dtype=float)
        data["mode"] = np.array([int(r["mode"]) for r in rows], dtype=np.int8)
        return cls(**data, meta=dict(meta or {}))

    # -- access ---------------------------------------------------------
    @property
    def is_plume(self) -> np.ndarray:
        return self.mode == Mode.PLUME

    @property
    def is_puff(self) -> np.ndarray:
        return self.mode == Mode.PUFF

    @property
    def transition_index(self) -> int | None:
        """Index of the first puff row, or None if the run stayed in one mode."""
        idx = np.flatnonzero(self.is_puff)
        return int(idx[0]) if idx.size and not self.is_puff[0] else None

    def at(self, x: float, *, fields_: tuple[str, ...] | None = None) -> dict:
        """
        Linearly interpolate the state to a down-wind distance.

        Interpolation is on the raw arrays; for quantities that vary over
        orders of magnitude (mass_frac, vol_frac) prefer `at_log`.
        """
        if not (self.x[0] <= x <= self.x[-1]):
            raise ValueError(f"x = {x} outside [{self.x[0]}, {self.x[-1]}]")
        names = fields_ or ARRAY_FIELDS
        return {k: float(np.interp(x, self.x, getattr(self, k))) for k in names}

    def at_log(self, x: float, name: str) -> float:
        """Interpolate a positive quantity in log space."""
        v = getattr(self, name)
        ok = np.isfinite(v) & (v > 0)
        if ok.sum() < 2:
            return float("nan")
        return float(np.exp(np.interp(x, self.x[ok], np.log(v[ok]))))

    def slice(self, mode: Mode) -> "Trajectory":
        """Return the sub-trajectory belonging to one dispersion mode."""
        sel = self.mode == mode
        if not sel.any():
            raise ValueError(f"no rows in {Mode(mode).name} mode")
        data = {f.name: getattr(self, f.name)[sel]
                for f in fields(self) if f.name != "meta"}
        return Trajectory(**data, meta=self.meta)

    # -- export ---------------------------------------------------------
    def to_dict(self) -> dict[str, np.ndarray]:
        return {f.name: getattr(self, f.name)
                for f in fields(self) if f.name != "meta"}

    def to_dataframe(self):
        """pandas DataFrame indexed by x, with `meta` in `df.attrs`."""
        import pandas as pd

        df = pd.DataFrame(self.to_dict()).set_index("x", drop=False)
        df.attrs.update(self.meta)
        return df

    def to_dataset(self):
        """xarray Dataset with x as the coordinate and `meta` as attributes."""
        import xarray as xr

        d = self.to_dict()
        x = d.pop("x")
        return xr.Dataset(
            {k: ("x", v) for k, v in d.items()},
            coords={"x": x},
            attrs={k: str(v) for k, v in self.meta.items()},
        )

    # -- verification ---------------------------------------------------
    def check_mass_conservation(
        self,
        expected: float | np.ndarray,
        *,
        rtol: float = 1e-6,
        tail_frac: float = 0.25,
    ) -> dict:
        """
        Tier-0 check: is the released mass accounted for in the cloud?

        Parameters
        ----------
        expected : either the total released mass [kg], or a per-row array of
            the mass released up to each row.  **Prefer the array form**: the
            scalar form can only be checked once the source has shut off and
            the cloud mass has plateaued, so it says nothing about the source
            region, which is exactly where mass errors originate.
        tail_frac : with the scalar form, the fraction of trailing rows over
            which the plateau is checked.

        Returns
        -------
        dict with ``max_rel_error``, ``passes`` and enough context to see
        *where* the check was applied.
        """
        m = np.asarray(self.mass_in_cloud, dtype=float)
        ok = np.isfinite(m)
        if not ok.any():
            return {"available": False}

        exp = np.asarray(expected, dtype=float)
        if exp.ndim == 0:
            if not 0.0 < tail_frac <= 1.0:
                raise ValueError("tail_frac must be in (0, 1]")
            n_tail = max(int(round(tail_frac * ok.sum())), 1)
            idx = np.flatnonzero(ok)[-n_tail:]
            ref = np.full(idx.shape, float(exp))
            where = f"last {n_tail} of {int(ok.sum())} rows (plateau)"
        else:
            if exp.shape != m.shape:
                raise ValueError(
                    f"expected array of length {m.size}, got {exp.size}"
                )
            idx = np.flatnonzero(ok & (exp > 0.0))
            ref = exp[idx]
            where = f"{idx.size} rows with released mass > 0"

        if idx.size == 0:
            return {"available": False}

        err = np.abs(m[idx] / ref - 1.0)
        worst = int(idx[int(np.argmax(err))])
        return {
            "available": True,
            "max_rel_error": float(err.max()),
            "passes": bool(err.max() <= rtol),
            "worst_at_x": float(self.x[worst]),
            "checked_over": where,
            "final_mass": float(m[ok][-1]),
        }
