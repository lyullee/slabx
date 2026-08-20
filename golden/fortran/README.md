# The Fortran oracle

`slabx` compares itself against Ermak's original SLAB, compiled from source.
**That source is not distributed with this project.**

## Why not

`SLAB.FOR` carries the following terms (its own header, 1990):

> (C) Copyright 1990 the Regents of the University of California.
> All Rights Reserved.
>
> This work was produced under the sponsorship of the U.S. Department of
> Energy. The Government retains certain rights therein.
>
> The user may copy and distribute the SLAB program [...] provided that:
>   1. No fee may be charged for copying or distributing the SLAB program
>      or any part thereof.
>   2. The SLAB program may not be distributed, in whole or in part, as part
>      of any commercial product.

Condition 2 is a non-commercial restriction, which no OSI-approved licence
permits. Bundling the file would make this repository non-free regardless of
what its own LICENSE said, so the file is fetched by the user instead.

This is a statement of what the header says, not legal advice.

## Getting it

SLAB is distributed as an EPA alternative model. The source has been
available from:

* EPA SCRAM, the alternative-models listing
* Lawrence Livermore National Laboratory, UCRL-MA-105607 (Ermak 1990)

Place the file here as `slab.f`, with DOS line endings stripped:

    tr -d '\r' < /path/to/SLAB.FOR > golden/fortran/slab.f

## Building

    cd golden/fortran
    gfortran -std=legacy -O2 -fno-automatic -w -o slab slab.f

`-std=legacy` for the F77 constructs; `-fno-automatic` because the code
assumes locals persist between calls. Warnings are suppressed because the
code predates most of them and they are not actionable here.

## Checking it works

    python3 -m pytest tests/test_fortran_oracle.py -v

Without the binary these tests skip rather than fail, and every other test in
the suite runs normally — the oracle is used for the reproduction claim, not
for the physics.

## What it is used for

    python3 examples/fortran_reference.py        # the five manual decks
    python3 examples/fuzz_fortran.py 60 7        # random differential test

See `docs/02_VALIDATION_REFERENCE.md` §2.2a for the results and for the two
input-format traps that cost the most time (`f10.3` truncation, and the four
tables in `predict`).
