"""
Statistical performance measures for dispersion model evaluation
================================================================

Implements the measure set that has become the de-facto standard in dense-gas
model evaluation, together with the acceptance criteria used by the LNG Model
Evaluation Protocol and its descendants.

References
----------
Hanna, S.R., Chang, J.C. & Strimaitis, D.G. (1993). Hazardous gas model
    evaluation with field observations. *Atmos. Environ.* 27A, 2265-2285.
Chang, J.C. & Hanna, S.R. (2004). Air quality model performance evaluation.
    *Meteorol. Atmos. Phys.* 87, 167-196.
Hanna, S. & Chang, J. (2012). Acceptance criteria for urban dispersion model
    evaluation. *Meteorol. Atmos. Phys.* 116, 133-146.
Ivings, M.J. et al. (2007, 2016). Evaluating vapor dispersion models for
    safety analysis of LNG facilities. FPRF.

Sign convention
---------------
Throughout, `obs` is the observation and `pred` the model.  FB and MG are
defined so that **> 1 (MG) or > 0 (FB) means the model under-predicts**,
following Hanna et al. (1993).  This is the convention used in the Phast and
DRIFT validation reports, so published values can be compared directly.

Paired comparison
-----------------
The reason this module exists is `compare()`.  Absolute skill is dominated by
source-term error, measurement error and the ensemble/single-realisation
mismatch, all of which are common to two model variants run on the same
trials and therefore cancel in the difference.  `compare()` resamples whole
*trials* (not individual points, which are correlated within a trial) and
reports the difference in each metric with a confidence interval.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "Metrics",
    "metrics",
    "ACCEPTANCE",
    "check_acceptance",
    "vg_min",
    "Comparison",
    "compare",
]


# ===========================================================================
# core measures
# ===========================================================================
@dataclass(frozen=True)
class Metrics:
    """Statistical performance measures for one model on one data set."""

    n: int
    """Number of paired points used (after screening)."""

    n_excluded: int
    """Points dropped because obs or pred fell below `floor`."""

    FB: float
    """Fractional bias, 2(Co-Cp)/(Co+Cp). 0 = unbiased, >0 = under-predicts."""

    MG: float
    """Geometric mean bias, exp(<ln Co - ln Cp>). 1 = unbiased, >1 = under."""

    NMSE: float
    """Normalised mean square error. 0 = perfect."""

    VG: float
    """Geometric variance, exp(<(ln Co - ln Cp)^2>). 1 = perfect."""

    FAC2: float
    """Fraction of predictions within a factor of 2 of observations."""

    FAC5: float
    """Fraction within a factor of 5."""

    R: float
    """Pearson correlation of obs and pred."""

    MRB: float
    """Mean relative bias, <2(Co-Cp)/(Co+Cp)>. SMEDIS / LNG MEP."""

    MRSE: float
    """Mean relative square error, <4(Co-Cp)^2/(Co+Cp)^2>. SMEDIS / LNG MEP."""

    def as_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"n={self.n:<4d} FAC2={self.FAC2:5.2f}  MG={self.MG:6.3f}  "
            f"VG={self.VG:6.3f}  FB={self.FB:+6.3f}  NMSE={self.NMSE:7.3f}"
        )


def _clean(obs, pred, floor):
    o = np.asarray(obs, dtype=float).ravel()
    p = np.asarray(pred, dtype=float).ravel()
    if o.shape != p.shape:
        raise ValueError(f"shape mismatch: obs {o.shape} vs pred {p.shape}")
    if o.size == 0:
        raise ValueError("empty input")
    keep = np.isfinite(o) & np.isfinite(p) & (o >= floor) & (p >= floor)
    return o[keep], p[keep], int((~keep).sum())


def metrics(
    obs: Sequence[float],
    pred: Sequence[float],
    *,
    floor: float = 1e-12,
) -> Metrics:
    """
    Compute all performance measures for one paired data set.

    Parameters
    ----------
    obs, pred : paired observations and predictions, same length.
    floor : points with obs or pred below this are excluded.  MG and VG are
        undefined for non-positive values, so a floor is unavoidable; Chang &
        Hanna recommend screening at a physically meaningful detection limit
        rather than at machine epsilon.  **Set this deliberately** — results
        are sensitive to it, and the chosen value must be reported.

    Notes
    -----
    FB and NMSE emphasise large concentrations; MG and VG treat the whole
    range logarithmically and are therefore the more informative pair when
    values span orders of magnitude.  Report both: FB or MG alone can be
    perfect through cancelling over- and under-predictions.
    """
    o, p, n_ex = _clean(obs, pred, floor)
    if o.size == 0:
        raise ValueError("no points survived screening; lower `floor`?")

    ob, pb = o.mean(), p.mean()
    ratio = p / o
    lr = np.log(o) - np.log(p)
    rel = 2.0 * (o - p) / (o + p)

    return Metrics(
        n=int(o.size),
        n_excluded=n_ex,
        FB=float(2.0 * (ob - pb) / (ob + pb)),
        MG=float(np.exp(lr.mean())),
        NMSE=float(np.mean((o - p) ** 2) / (ob * pb)),
        VG=float(np.exp(np.mean(lr**2))),
        FAC2=float(np.mean((ratio >= 0.5) & (ratio <= 2.0))),
        FAC5=float(np.mean((ratio >= 0.2) & (ratio <= 5.0))),
        R=float(np.corrcoef(o, p)[0, 1]) if o.size > 1 and o.std() > 0
        and p.std() > 0 else float("nan"),
        MRB=float(rel.mean()),
        MRSE=float(np.mean(rel**2)),
    )


def vg_min(mg: float) -> float:
    """
    Smallest VG attainable for a given MG, VG_min = exp((ln MG)^2).

    On an MG-VG diagram every model must lie on or above this parabola; the
    vertical distance from it is the part of the scatter that is *not*
    explained by systematic bias.
    """
    return float(np.exp(np.log(mg) ** 2))


# ===========================================================================
# acceptance criteria
# ===========================================================================
#: Published acceptance ranges.  ``None`` means the criterion is not applied.
ACCEPTANCE: dict[str, dict[str, tuple[float | None, float | None]]] = {
    # Chang & Hanna (2004), rural / open-terrain releases
    "chang_hanna_2004": {
        "FB": (-0.3, 0.3),
        "MG": (0.7, 1.3),
        "NMSE": (None, 4.0),
        "VG": (None, 1.6),
        "FAC2": (0.5, None),
    },
    # Hanna & Chang (2012), relaxed for built-up terrain
    "hanna_chang_2012_urban": {
        "FB": (-0.67, 0.67),
        "NMSE": (None, 6.0),
        "FAC2": (0.3, None),
    },
}


def check_acceptance(
    m: Metrics, criteria: str | Mapping = "chang_hanna_2004"
) -> dict[str, bool]:
    """
    Test a `Metrics` result against a named (or custom) criterion set.

    Returns one boolean per criterion plus ``"all"``.  A criterion whose
    metric is NaN counts as failed.
    """
    spec = ACCEPTANCE[criteria] if isinstance(criteria, str) else dict(criteria)
    out: dict[str, bool] = {}
    for key, (lo, hi) in spec.items():
        v = getattr(m, key)
        ok = np.isfinite(v)
        if ok and lo is not None:
            ok = bool(v >= lo)
        if ok and hi is not None:
            ok = bool(v <= hi)
        out[key] = bool(ok)
    out["all"] = all(out.values())
    return out


# ===========================================================================
# paired comparison of two model variants
# ===========================================================================
@dataclass(frozen=True)
class Comparison:
    """Result of a paired bootstrap comparison between two model variants."""

    label_a: str
    label_b: str
    n_trials: int
    n_points: int
    metrics_a: Metrics
    metrics_b: Metrics
    delta: dict[str, float]
    """Point estimate of metric(B) - metric(A)."""
    ci: dict[str, tuple[float, float]]
    """Bootstrap confidence interval on the difference."""
    significant: dict[str, bool]
    """True where the interval excludes zero."""
    level: float

    def summary(self, keys: Iterable[str] = ("FAC2", "MG", "VG", "FB", "NMSE")) -> str:
        w = max(len(self.label_a), len(self.label_b), 6)
        lines = [
            f"{'':14s} {self.label_a:>{w}s} {self.label_b:>{w}s} "
            f"{'delta':>9s} {f'{self.level:.0%} CI':>19s}  sig",
            "-" * (16 + 2 * w + 32),
        ]
        for k in keys:
            lo, hi = self.ci[k]
            lines.append(
                f"{k:14s} {getattr(self.metrics_a, k):{w}.3f} "
                f"{getattr(self.metrics_b, k):{w}.3f} {self.delta[k]:+9.3f} "
                f"[{lo:+8.3f},{hi:+8.3f}]  {'*' if self.significant[k] else ' '}"
            )
        lines.append(f"trials={self.n_trials}  points={self.n_points}")
        return "\n".join(lines)


def compare(
    obs: Sequence[float],
    pred_a: Sequence[float],
    pred_b: Sequence[float],
    *,
    trial: Sequence | None = None,
    label_a: str = "A",
    label_b: str = "B",
    keys: Sequence[str] = ("FAC2", "MG", "VG", "FB", "NMSE", "FAC5", "MRB", "MRSE"),
    n_boot: int = 10_000,
    level: float = 0.95,
    floor: float = 1e-12,
    seed: int | None = 0,
) -> Comparison:
    """
    Paired bootstrap comparison of two model variants on the same trials.

    Both variants must be evaluated against the *same* observations, in the
    same order.  Resampling is done at the level of `trial` (points within a
    trial are correlated, so resampling points would understate the interval);
    with `trial=None` every point is treated as its own trial, which is
    anti-conservative — pass real trial identifiers whenever they exist.

    The same resample is used for A and B in every replicate.  This is what
    makes the comparison powerful: source-term error, measurement error and
    ensemble mismatch are shared between the variants and cancel in the
    difference, so a small improvement is detectable even when neither
    variant's absolute skill is well determined.

    Returns
    -------
    Comparison
        `significant[k]` is True where the confidence interval on
        metric(B) - metric(A) excludes zero.
    """
    o = np.asarray(obs, dtype=float).ravel()
    a = np.asarray(pred_a, dtype=float).ravel()
    b = np.asarray(pred_b, dtype=float).ravel()
    if not (o.shape == a.shape == b.shape):
        raise ValueError("obs, pred_a and pred_b must have the same length")

    tr = np.arange(o.size) if trial is None else np.asarray(trial).ravel()
    if tr.shape != o.shape:
        raise ValueError("trial must have the same length as obs")
    groups = [np.flatnonzero(tr == t) for t in dict.fromkeys(tr.tolist())]

    ma, mb = metrics(o, a, floor=floor), metrics(o, b, floor=floor)
    delta = {k: getattr(mb, k) - getattr(ma, k) for k in keys}

    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {k: [] for k in keys}
    ng = len(groups)
    for _ in range(n_boot):
        pick = np.concatenate([groups[i] for i in rng.integers(0, ng, ng)])
        try:
            ra = metrics(o[pick], a[pick], floor=floor)
            rb = metrics(o[pick], b[pick], floor=floor)
        except ValueError:
            continue
        for k in keys:
            draws[k].append(getattr(rb, k) - getattr(ra, k))

    lo_q, hi_q = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    ci, sig = {}, {}
    for k in keys:
        d = np.asarray(draws[k], dtype=float)
        d = d[np.isfinite(d)]
        if d.size < 100:
            ci[k], sig[k] = (float("nan"), float("nan")), False
            continue
        lo, hi = float(np.percentile(d, lo_q)), float(np.percentile(d, hi_q))
        ci[k], sig[k] = (lo, hi), bool(lo > 0 or hi < 0)

    return Comparison(
        label_a=label_a,
        label_b=label_b,
        n_trials=ng,
        n_points=int(o.size),
        metrics_a=ma,
        metrics_b=mb,
        delta=delta,
        ci=ci,
        significant=sig,
        level=level,
    )
