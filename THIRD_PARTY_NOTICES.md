# Third-party material

This repository contains no third-party code or data. What follows records
the third-party works this project depends on, so that a reader can obtain
them and see what was relied on.

## The model

The model implemented here is described in

> Ermak, D.L. (1990) *User's Manual for SLAB: An Atmospheric Dispersion Model
> for Denser-than-Air Releases*. UCRL-MA-105607, Lawrence Livermore National
> Laboratory.

**The original Fortran source is not included.** Its header reserves rights
to the Regents of the University of California, notes that the work was
produced under Department of Energy sponsorship, and permits copying and
distribution only on two conditions: that no fee is charged, and that it is
not distributed as part of any commercial product. The second is a
non-commercial restriction, which no OSI-approved licence permits; including
the file would make this repository non-free whatever its own licence said.

`golden/fortran/README.md` explains how to obtain and build it. The oracle
tests skip without it.

### On the relationship between the two

This is a reimplementation, not a translation. The program structure,
data model, and coefficient handling are original; the physics follows the
published formulation and the user's manual.

It is **not** a clean-room implementation. The original Fortran was consulted
during development, both to settle implementation details the manual leaves
open and to compare results. Where a specific behaviour was checked against
it, the source records the line number, so that the checking is visible
rather than implied. Those references are citations of location, not
reproductions of code.

## Observations

None of the validation observations are redistributed here. Their sources,
the tables they came from, and the transformations applied are recorded in
`src/slabx/validation/data/provenance/`, which also explains where to obtain
them and where to put them.

In summary they come from: the EU SMEDIS project (via ADMLC); PHMSA's LNG
Model Evaluation Protocol validation database and its Phast 8.4
environmental assessment; the Jack Rabbit III model inter-comparison (via
ADMLC); Witlox et al. (2013) in *Chemical Engineering Transactions*; and
McQuaid & Roebuck (1985), EUR 10029 EN.

No permission to redistribute was sought from any of them. The material was
excluded rather than pursued.

## Dependencies

NumPy and SciPy (BSD-3-Clause), and optionally CoolProp (MIT). These are
declared in `pyproject.toml` and installed from their own distributions;
none of their code is vendored here.
