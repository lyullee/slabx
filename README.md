# slabx

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22023625.svg)](https://doi.org/10.5281/zenodo.22023625)
[![PyPI](https://img.shields.io/pypi/v/slabx.svg)](https://pypi.org/project/slabx/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A Python reimplementation of **SLAB**, the dense-gas dispersion model of
Ermak (1990, UCRL-MA-105607), verified against the original Fortran and
evaluated against 38 field and wind-tunnel trials.

The implementation follows the published model formulation and the user's
manual, in a program structure of its own; individual results were then
compared against the original Fortran, which was also consulted for
implementation details the manual leaves open. It is therefore not a
clean-room implementation, and the comparison points are recorded in the
source as line references so that the checking is visible.

```python
from slabx.core.plume import run_dispersion
from slabx.core.source import EvaporatingPool
from slabx.post.concentration import concentration_field
from slabx.submodels.atmosphere import Atmosphere
from slabx.thermo.base import LegacyThermo, Substance, water_backend

LNG = Substance(name="LNG", mw=0.016043, cp_vapour=2238.0, cp_liquid=3348.5,
                dh_vap=509900.0, T_boil=111.7, rho_liquid=424.1)

atm = Atmosphere(u_ref=1.94, z_ref=3.0, T=290.0, rh=50.0, z0=2e-4,
                 stability="E")

# The pool area is the spill divided by the evaporation flux, not the
# footprint quoted in the manual, and the manual's worked example uses
# demonstration weather rather than the trial's. Both matter: the manual's
# 657 m2 with T = 306 K and 4.6 % humidity gives 386 m instead of 449.
src = EvaporatingPool(substance=LNG, rate=116.93, area=116.93 / 0.167,
                      duration=107.0)

traj, used = run_dispersion(src, atm, LegacyThermo(LNG), water_backend(),
                            x_max=2000.0, n_puff_steps=40)
field = concentration_field(traj, atm, z=1.0, t_avg=80.0, t_release=107.0)

print(f"{field.distance_to(0.05):.0f} m to LFL")     # 449; measured 455
```

`slabx.validation.lng_pools` sets all of this from the trial record, which
is the path the paper's numbers come from -- but it needs the observations,
which are not distributed here. See below.

## Install

```bash
pip install slabx                 # numpy, scipy
pip install "slabx[thermo]"       # adds CoolProp for real-fluid properties
```

From a clone, for development or to run the tests:

```bash
pip install -e ".[thermo,dev]"
```

Python 3.10+. `QUICKSTART.md` has a runnable scenario template and a
troubleshooting table.

## What it does

Four source types — evaporating pool, horizontal jet, vertical jet with plume
rise, instantaneous release — integrated as a shallow-layer model with the
plume-to-puff transition of the original.

Beyond the original, all off by default so the reproduction is never disturbed:

| | |
|---|---|
| `rainout=True` | droplet break-up, settling and removal |
| `kinetic_evaporation=True` | finite-rate droplet evaporation |
| `added_mass=True` | oblate-spheroid added mass on buoyant rise |
| `substrate=DRY_SOIL` | ground heat transfer |
| `CoolPropThermo` | real-fluid properties, and multicomponent mixtures |
| `meander_closure="sigma_theta"` | meander from measured wind-direction variability |

Thirty-eight coefficients are exposed as a frozen dataclass with presets, so
sensitivity studies do not need edits to the source.

## How far it has been checked

**Against the original Fortran.** The five decks of the manual agree to
0.05–1.48 % on every cloud variable, at the level of the original's own
discretisation error. A random differential test over 108 decks — four source
types, five materials, the full stability range — puts the median difference
at 0.09 % on temperature and 0.67 % on concentration.

The comparison is against Ermak's Fortran itself, compiled from source. Early
work used a JavaScript transcription and misattributed five of six differences
to SLAB as a result; `docs/02_VALIDATION_REFERENCE.md` §2.5 records what that
cost and how it was found.

**Against measurements.** Ten LNG pool trials (Burro, Coyote, Maplin Sands),
three Desert Tortoise and three FLADIS two-phase ammonia jets, fifteen Thorney
Island instantaneous releases, two Prairie Grass passive-dispersion trials,
and three wind-tunnel trials. Nineteen further SMEDIS trials are excluded with
a recorded reason for each.

On the ten LNG trials the model passes all five Chang–Hanna criteria
(FAC2 0.90, MG 1.25, NMSE 0.12).

**Where it fails.** `LIMITATIONS.md` is the honest list: a known runaway for
dense, low-momentum releases; a stability response that four independent
datasets say is too strong; a hard-coded reference height that breaks Froude
similarity; and two published datasets for the same trials that disagree by a
factor of two and reverse the recommended surface roughness.

`VALIDATION.md` has the numbers and the commands that reproduce them.

## Tests

```bash
pytest -q
python3 examples/validate_burro.py     # field comparison, pre-registered
python3 examples/diagnose_burro.py     # how the hypotheses were excluded
python3 -c "from slabx.scope import describe_scope; print(describe_scope())"
```

Tests that need the original SLAB source, an optional dependency, or a
third-party observation are skipped when the resource is absent, so the
counts depend on what you have installed and obtained. Nothing fails for
want of them.

The one `xfailed` is deliberate: a reproducible failure kept with its input
deck rather than hidden.

Comparing against the original needs its source, which is not redistributable
here; see `golden/fortran/README.md`. Those tests skip without it and
everything else runs.

## Documentation

| | |
|---|---|
| `QUICKSTART.md` | install, run, change a scenario |
| `LIMITATIONS.md` | what it cannot do, with evidence |
| `VALIDATION.md` | every comparison and how to reproduce it |
| `docs/01_THEORY.md` | governing equations and closures |
| `docs/02_VALIDATION_REFERENCE.md` | agreement with the original |
| `docs/03_VALIDATION_FIELD.md` | agreement with measurements |
| `docs/04_DEVELOPMENT.md` | defects found, and the method that found them |

## Citing

Cite the version you ran -- the version DOI fixes the files, the concept DOI
resolves to the latest:

> Lee, U. (2026). *slabx: a Python reimplementation of the SLAB dense-gas
> dispersion model*. Zenodo. Concept DOI
> [10.5281/zenodo.22023625](https://doi.org/10.5281/zenodo.22023625).

Each release has its own version DOI, listed on the Zenodo record under
*Versions*. Cite the one you ran rather than the number written here: a
release cannot record a DOI that Zenodo issues only after the release is
published, so any version number in this file is the version being
prepared, not the one you downloaded.

`CITATION.cff` carries the same in machine-readable form.

The model itself is Ermak's:

> Ermak, D.L. (1990) *User's Manual for SLAB: An Atmospheric Dispersion Model
> for Denser-than-Air Releases*. UCRL-MA-105607, Lawrence Livermore National
> Laboratory.

## Licence

MIT, for this implementation. `THIRD_PARTY_NOTICES.md` records the
third-party works relied on, none of which are included here. The original SLAB source is not included and
carries its own non-commercial terms; see `LICENSE` and
`golden/fortran/README.md`.
