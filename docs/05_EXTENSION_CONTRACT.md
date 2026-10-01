# Extension contract

`slabx` is the material-independent dense-gas transport core.  An extension
may calculate a release upstream of SLAB, but the handoff must preserve the
quantities that the integrator actually conserves.

## Boundary of responsibility

The core owns the shallow-layer plume and puff equations, entrainment,
atmospheric transport, thermodynamic state updates, and concentration
post-processing.  A material package owns source physics that is not part of
SLAB: flashing, jet impact, pool formation, cryogenic air condensation, and
the construction of the resolved state at the handoff plane.

The handoff should be expressed as a `SourceModel` and, when a route boundary
must be recorded or exchanged, as a JSON-safe `SourceLedger`.  The ledger is a
record of a resolved state; it is not permission to invent a missing source
history.  Native adapters therefore reject unresolved liquid accumulation.

## Species-inventory rule

`SourceModel.total_mass` and `SourceModel.released_mass(t)` must use the same
emitted-species basis as the integrated species accumulator used by the
plume-to-puff transition.  For a source that has already entrained air before
the handoff, the following are different quantities:

- emitted-species mass, such as H2 mass;
- total carrier mass, such as H2 plus entrained air;
- the half-plume mixture flux stored as `R_flux`.

Carrier mass must be exposed separately when it is useful.  Returning carrier
mass from `total_mass` compares unlike inventories and can delay or suppress a
finite-release plume-to-puff transition.

## Extension checklist

An external source adapter should demonstrate that:

1. mass fractions sum to one on the stated basis;
2. mixture flux, density, velocity, and cross-sectional area are consistent;
3. `total_mass` and `released_mass` track the emitted species;
4. carrier inventory, if present, has a separate name;
5. temperature, heat capacity, geometry, and release duration are explicit;
6. no observation-derived residual is hidden in the source state.

The companion [`slabx-lh2`](https://github.com/lyullee/slabx-lh2) package is
the reference material-specific extension.  Its `PhysicalTransitionCloud`
implements a pre-diluted LH2 handoff while keeping H2 and carrier inventories
separate.
