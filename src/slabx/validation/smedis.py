"""
Reading the SMEDIS standardised trial files
===========================================

The SMEDIS project distributed one spreadsheet per trial in a common layout:
a block of ``key | value`` rows for release and meteorological conditions,
then tabular blocks for sensors, each introduced by a ``#`` heading and a row
of column titles.

The layout is common but not rigid.  Keys are indented inconsistently, some
have trailing units and some do not, the same quantity appears under
different spellings between datasets, and missing values are ``-999``.  The
sensor blocks differ too: continuous releases report concentration, puff
releases report dose with arrival and departure times.  This module absorbs
that so the validation code can ask for a quantity by name.

What the files contain
----------------------
Twenty-eight trials across eight campaigns.  Twenty-two have no obstacles and
are therefore within this model's scope; the rest have box arrays, canyons,
trenches, fences or curved walls and are recorded but excluded.

Of the twenty-two, the ones that matter here are the ones that measure
something the model has not been tested against:

* **Thorney Island 008** — an instantaneous release with per-sensor *dose*,
  arrival time and departure time, plus a measured friction velocity.  The
  instantaneous source type has never been compared against sensor-level
  field data.
* **Desert Tortoise 1 and 2, FLADIS 9/16/24** — two-phase ammonia jets.
* **Prairie Grass 8 and 17** — passive tracer, the neutral-density limit.
* **BA-Hamburg and BA-TNO** — wind tunnel, at the small end of the scale.

Missing values
--------------
``-999`` means not measured and is returned as ``None`` rather than as a
number, because a friction velocity of minus 999 that reaches a model is
worse than no friction velocity at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["MISSING", "SmedisTrial", "load_smedis", "load_all_smedis"]

#: Sentinel used throughout the SMEDIS files for "not measured".
MISSING = -999.0

#: Aliases for quantities whose key differs between campaigns.
_ALIASES = {
    "wind_speed": ("site average windspeed", "site avg windspeed",
                   "windspeed", "wind speed"),
    "wind_ref_height": ("ref height for wind",),
    "friction_velocity": ("friction velocity",),
    "roughness": ("surface roughness",),
    "monin_obukhov": ("monin-obukov length", "monin-obukhov length"),
    "temperature": ("ambient temperature",),
    "humidity": ("relative humidity",),
    "stability": ("stability class",),
    "substance": ("substance",),
    "release_type": ("release type",),
    "obstacles": ("obstacle configuration",),
    "rate": ("release rate",),
    "duration": ("release duration",),
    "exit_temperature": ("exit temperature",),
    "exit_pressure": ("exit pressure",),
    "molecular_weight": ("effective mol wt",),
    "nozzle_diameter": ("nozzle diameter",),
    "initial_concentration": ("initial concentration",),
    "liquid_fraction": ("phase",),
    "release_height": ("z",),
    "length_scale": ("length scale",),
    "time_scale": ("time scale",),
}


def _clean(key: str) -> str:
    """Strip comment markers, units and whitespace from a key."""
    k = key.strip().lstrip("#").strip().lower()
    k = re.sub(r"\s*\([^)]*\)\s*$", "", k)      # trailing (m), (kg/s), ...
    return re.sub(r"\s+", " ", k).strip()


def _looks_like_header(row) -> bool:
    """
    A column-title row rather than a key/value pair.

    Distinguished by having two or more non-numeric cells: ``X(m) | Y(m) |
    height(m) | Dose`` is a header, while ``month | 9.0`` is a field.  Some
    files mark headers with a leading '#' and some do not, so the shape has
    to decide.
    """
    cells = [str(c).strip() for c in row if str(c).strip()]
    if len(cells) < 3:
        return False
    # Three or more non-numeric cells.  "X(m) | Y(m) | height(m) | Dose" is a
    # header; "substance | R12" is a field and must not be mistaken for one,
    # which would drop it out of `fields` entirely.
    return all(_number(c) is None for c in cells)


def _number(value) -> float | None:
    """Parse a cell to a float, mapping the -999 sentinel to None."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        v = float(value)
    else:
        s = str(value).strip().replace(",", ".")
        m = re.match(r"^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        if not m:
            return None
        v = float(m.group())
    return None if v <= MISSING + 1e-9 else v


@dataclass
class SmedisTrial:
    """One trial, with its conditions and sensor tables."""

    path: Path
    dataset: str
    trial: str
    fields: dict[str, str] = field(default_factory=dict)
    tables: dict[str, list[list]] = field(default_factory=dict)
    headers: dict[str, list[str]] = field(default_factory=dict)
    raw_headers: dict[str, list[str]] = field(default_factory=dict)
    raw_headers: dict[str, list[str]] = field(default_factory=dict)

    # -- conditions ----------------------------------------------------
    def raw(self, name: str) -> str | None:
        """Value under any of the aliases for `name`, as text."""
        for alias in _ALIASES.get(name, (name,)):
            for key, value in self.fields.items():
                if key.startswith(alias):
                    return value
        return None

    def value(self, name: str) -> float | None:
        """Value under any of the aliases for `name`, as a number."""
        return _number(self.raw(name))

    @property
    def obstructed(self) -> bool:
        o = (self.raw("obstacles") or "").strip().lower()
        return o not in ("", "none")

    @property
    def instantaneous(self) -> bool:
        return (self.raw("release_type") or "").strip().lower() == "puff"

    # -- tables --------------------------------------------------------
    def table(self, name: str) -> list[dict]:
        """
        One tabular block as a list of dicts keyed by its column titles.

        The header row is itself prefixed with ``#`` in most files, so it
        arrives as a separate section rather than as the table's first row;
        both layouts are handled.
        """
        want = name.lower()
        keys = [k for k in self.tables if want in k.lower()]
        if not keys:
            # Some files put the table under the *preceding* heading, e.g.
            # Thorney Island's sensor block follows a bare "# No Arcs".
            keys = [k for k, h in self.headers.items()
                    if any(want.split("_")[0] in c for c in h)]
        if not keys:
            return []
        key = keys[0]
        rows = self.tables[key]
        header = self.headers.get(key)
        if header is None and rows:
            header, rows = [_clean(str(h)) for h in rows[0]], rows[1:]
        if not header:
            return []
        # Column titles collapse to the same key once units are stripped —
        # "T(arriv.,s)" and "T(depart.,s)" both become "t" — so duplicates
        # are suffixed rather than silently overwriting each other.
        # Column titles collapse once units are stripped: "T(arriv.,s)" and
        # "T(depart.,s)" both become "t".  Recover the distinction from the
        # original titles rather than numbering them, since arrival and
        # departure are not interchangeable.
        raw_titles = self.raw_headers.get(key) or []
        seen: dict[str, int] = {}
        names = []
        for i, h in enumerate(header):
            if not h:
                names.append("")
                continue
            title = raw_titles[i].lower() if i < len(raw_titles) else ""
            if h == "t" and "arriv" in title:
                h = "arrival"
            elif h == "t" and "depart" in title:
                h = "departure"
            seen[h] = seen.get(h, 0) + 1
            names.append(h if seen[h] == 1 else f"{h}_{seen[h]}")

        out = []
        for row in rows:
            rec = {h: _number(v) for h, v in zip(names, row) if h}
            if any(v is not None for v in rec.values()):
                out.append(rec)
        return out

    def arcs(self) -> list[dict]:
        """
        Arc-wise maxima: distance, height, peak concentration, width.

        This is the quantity every model-evaluation protocol compares, and
        it is tabulated directly rather than left to be derived from the
        sensor grid.
        """
        out = []
        for r in self.table("arc_positions"):
            vals = [v for v in r.values() if v is not None]
            if len(vals) < 3:
                continue
            out.append({"distance": vals[0], "height": vals[1],
                        "concentration": vals[2],
                        "width": vals[3] if len(vals) > 3 else None})
        return out

    def sensors(self) -> list[dict]:
        """
        Concentration or dose sensors.

        Continuous releases report a mean concentration; instantaneous ones
        report a dose with arrival and departure times.
        """
        return self.table("concentration_sensor")


def _read_rows(path: Path) -> list[list]:
    import xlrd

    sheet = xlrd.open_workbook(path).sheet_by_index(0)
    return [[sheet.cell_value(i, j) for j in range(sheet.ncols)]
            for i in range(sheet.nrows)]


def load_smedis(path: str | Path) -> SmedisTrial:
    """Parse one SMEDIS trial spreadsheet."""
    path = Path(path)
    rows = _read_rows(path)

    trial = SmedisTrial(path=path, dataset="", trial="")
    section = None
    for row in rows:
        first = str(row[0]).strip()
        if not first:
            continue
        if first.startswith("#"):
            label = _clean(first)
            if label.startswith("dataset_reference"):
                trial.dataset = str(row[1]).strip()
            elif label.startswith("trial_identification"):
                trial.trial = str(row[1]).strip()
            elif section and section in trial.tables \
                    and not trial.tables[section] \
                    and section not in trial.headers \
                    and _looks_like_header(row):
                # A second '#' row straight after a table heading is the
                # column titles, not a new section.
                trial.headers[section] = [_clean(str(c)) for c in row]
                trial.raw_headers[section] = [str(c) for c in row]
            else:
                section = label
                trial.tables.setdefault(section, [])
            continue
        rest = [c for c in row[1:] if str(c).strip()]
        # A row whose first cell is a number belongs to the current table;
        # one whose first cell is text is a key/value pair.
        looks_tabular = _number(first) is not None
        if section and section in trial.tables and not trial.tables[section] \
                and section not in trial.headers and not looks_tabular \
                and _looks_like_header(row):
            trial.headers[section] = [_clean(str(c)) for c in row]
            trial.raw_headers[section] = [str(c) for c in row]
            continue
        if section and looks_tabular:
            trial.tables[section].append(list(row))
        elif rest:
            trial.fields[_clean(first)] = str(row[1]).strip()
    return trial


def load_all_smedis(directory: str | Path,
                    *, obstacle_free_only: bool = False) -> list[SmedisTrial]:
    """Every ``.xls`` in `directory`, optionally excluding obstructed trials."""
    trials = []
    for p in sorted(Path(directory).glob("*.xls")):
        try:
            t = load_smedis(p)
        except Exception:                                          # noqa: BLE001
            continue
        if t.dataset and not (obstacle_free_only and t.obstructed):
            trials.append(t)
    return trials
