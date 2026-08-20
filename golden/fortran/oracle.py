"""
Ermak's Fortran as the oracle
=============================

Runs the original ``SLAB.FOR`` (Ermak 1990, UCRL-MA-105607) and parses its
output, so that comparisons are against the model rather than against a
transcription of it.

Why this replaced the JavaScript
--------------------------------
The project began by comparing against ``SLAB_main.js``, a JavaScript port.
Six differences were recorded as defects of SLAB; checking against the
Fortran showed that only one was.  Three came from a mechanical
``min`` -> ``Math.min`` substitution applied to the whole file — it also
produced ``Math.minus`` and ``terMath.minated`` inside comments — one was a
single-character typo (``all`` for ``aal``, ten occurrences against one), and
one was a deliberate sentinel mistaken for an omission.

The typo mattered most: it made the JavaScript throw for ``|1/L|`` above
about 0.15, which was recorded as a limit of SLAB and used to bound the
validity of the whole reproduction.  The Fortran has no such limit.

Two traps in the input format
------------------------------
1. The read format is ``f10.3``, but Fortran lets an explicit decimal point
   in the data override the format's implied scale.  Writing ``"%10.3f"``
   truncates a molecular weight of 0.016043 to 0.016 and every subsequent
   calculation returns NaN.  `_f10` keeps as many digits as fit.

2. Each value occupies its own record — the format has no repeat count — so
   the file is one number per line, not one line per group.

Output
------
``predict`` holds four tables.  The one wanted here is the first, headed
``x  zc  h  bb  b  bbx  bx  cv  rho  t  u  ua``; the others are contour
parameters, a z-plane slice and the centreline maximum.  Parsing without
splitting on the headers silently interleaves them.

Building
--------
    gfortran -std=legacy -O2 -fno-automatic -w -o slab slab.f

``-std=legacy`` for the F77 constructs and ``-fno-automatic`` because the
code assumes locals persist between calls.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["FortranSLAB", "FortranResult", "OracleUnavailable"]

HERE = Path(__file__).resolve().parent

#: Columns of the spatially averaged table, in order.
COLUMNS = ("x", "zc", "h", "bb", "b", "bbx", "bx", "cv", "rho", "t", "u", "ua")

#: Header that marks the start of that table.
HEADER = "x        zc"


class OracleUnavailable(RuntimeError):
    """The compiled binary is missing, or the run failed."""


def _f10(x: float) -> str:
    """
    Ten characters, as many significant digits as fit.

    The format is ``f10.3``, but an explicit decimal point in the data wins.
    Writing ``"%10.3f"`` would truncate 0.016043 to 0.016 and poison the run.
    """
    target = float(f"{x:.10g}")
    for fmt in ("{:10.6f}", "{:10.5f}", "{:10.4f}", "{:10.3f}",
                "{:10.2f}", "{:10.1f}", "{:10.0f}"):
        s = fmt.format(x)
        if len(s) <= 10 and float(s) == target:
            return s
    for fmt in ("{:10.6f}", "{:10.4f}", "{:10.3f}", "{:10.1f}"):
        s = fmt.format(x)
        if len(s) <= 10:
            return s
    return f"{x:10.3E}"


@dataclass(frozen=True)
class FortranResult:
    """One run of the original code."""

    table: np.ndarray          #: (n, 12) spatially averaged cloud parameters
    stdout: str
    raw: str                   #: the whole `predict` file

    def column(self, name: str) -> np.ndarray:
        return self.table[:, COLUMNS.index(name)]

    @property
    def x(self) -> np.ndarray:
        return self.table[:, 0]

    def at(self, name: str, x: float) -> float:
        """Linear interpolation of one column at a down-wind distance."""
        return float(np.interp(x, self.x, self.column(name)))


class FortranSLAB:
    """
    Wrapper around the compiled original.

    Parameters
    ----------
    binary : path to the built executable.  Defaults to ``golden/fortran/slab``.
    """

    def __init__(self, binary: Path | str | None = None):
        self.binary = Path(binary) if binary else HERE / "slab"
        if not self.binary.exists():
            raise OracleUnavailable(
                f"{self.binary} not found. Build it with:\n"
                f"  cd {HERE} && gfortran -std=legacy -O2 -fno-automatic "
                f"-w -o slab slab.f"
            )

    # ------------------------------------------------------------------
    def write_input(self, path: Path, *, idspl: int, ncalc: int = 1,
                    wms: float, cps: float, tbp: float, cmed0: float,
                    dhe: float, cpsl: float, rhosl: float,
                    spb: float = -1.0, spc: float = 0.0,
                    ts: float, qs: float, as_: float, tsd: float,
                    qtis: float = 0.0, hs: float = 0.0,
                    tav: float = 10.0, xffm: float = 1000.0,
                    zp: tuple = (0.0, 0.0, 0.0, 0.0),
                    z0: float, za: float, ua: float, ta: float,
                    rh: float, stab: float, ala: float | None = None) -> None:
        """
        Write an input deck in the layout ``SLAB.FOR`` expects.

        One value per line.  ``ala`` is read only when ``stab`` is zero.  A
        negative ``z0`` terminates the run, so it is appended as a trailer.
        """
        lines = [f"{idspl:5d}", f"{ncalc:5d}"]
        lines += [_f10(v) for v in
                  (wms, cps, tbp, cmed0, dhe, cpsl, rhosl, spb, spc)]
        lines += [_f10(v) for v in (ts, qs, as_, tsd, qtis, hs)]
        lines += [_f10(v) for v in (tav, xffm, *zp)]
        lines.append(_f10(z0))
        lines += [_f10(v) for v in (za, ua, ta, rh, stab)]
        if stab == 0.0:
            if ala is None:
                raise ValueError("stab = 0 requires an inverse "
                                 "Monin-Obukhov length `ala`")
            lines.append(_f10(ala))
        lines.append(_f10(-1.0))                 # sentinel: stop
        path.write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------------
    def run(self, *, timeout: float = 60.0, **deck) -> FortranResult:
        """Run one deck in a scratch directory and parse the output."""
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            shutil.copy(self.binary, work / "slab")
            (work / "slab").chmod(0o755)
            self.write_input(work / "input", **deck)

            proc = subprocess.run(["./slab"], cwd=work, timeout=timeout,
                                  capture_output=True, text=True)
            out = work / "predict"
            if not out.exists():
                raise OracleUnavailable(
                    f"no `predict` written (rc={proc.returncode}): "
                    f"{proc.stderr[:300]}"
                )
            raw = out.read_text()

        return FortranResult(table=self._parse(raw), stdout=proc.stdout,
                             raw=raw)

    # ------------------------------------------------------------------
    @staticmethod
    def _parse(raw: str) -> np.ndarray:
        """
        Extract the spatially averaged table.

        `predict` holds four tables; taking every numeric line interleaves
        them and produces nonsense — an early version of this parser did
        exactly that and reported a factor-three disagreement that was purely
        a column misalignment.
        """
        lines = raw.splitlines()
        try:
            start = next(i for i, ln in enumerate(lines)
                         if ln.strip().startswith(HEADER))
        except StopIteration:
            raise OracleUnavailable("output has no spatially averaged table")

        rows = []
        for ln in lines[start + 1:]:
            parts = ln.split()
            if len(parts) != len(COLUMNS):
                break
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                break
        if len(rows) < 3:
            raise OracleUnavailable(f"table has only {len(rows)} rows")
        return np.array(rows)
