"""
Turn a missing observation into a skip.

The validation observations are third-party and are not distributed with
slabx (see `validation/data/provenance/README.md`). Everything that does not
depend on them still runs, so a reader can verify the reproduction against
the original Fortran, the randomised differential test, the Froude scaling
and the ablations without obtaining anything.

`ObservationsUnavailable` is raised by `validation._data_access.require`.
It is caught at three points because a missing file surfaces differently
depending on where it is read: inside the test body, inside a fixture, or
while a module-level constant is being built at collection time. Handling it
here keeps the tests free of availability logic and makes the skip reason
name the file.

The hooks use pytest's `wrapper=True` form, which lets the wrapper catch what
the wrapped call raised. The older `hookwrapper=True` with
`outcome.get_result()` also works but pytest reports raising from it as a
teardown fault, which fills the run with warnings that say nothing about the
data.
"""

import pytest

from slabx.validation._data_access import ObservationsUnavailable


def _reason(exc: ObservationsUnavailable) -> str:
    return f"observation not distributed with slabx: {exc.name}"


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    try:
        return (yield)
    except ObservationsUnavailable as exc:
        pytest.skip(_reason(exc))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item):
    try:
        return (yield)
    except ObservationsUnavailable as exc:
        pytest.skip(_reason(exc))


@pytest.hookimpl(wrapper=True)
def pytest_fixture_setup(fixturedef, request):
    try:
        return (yield)
    except ObservationsUnavailable as exc:
        pytest.skip(_reason(exc))


@pytest.hookimpl(wrapper=True)
def pytest_collectstart(collector):
    """
    A module that reads an observation while being imported cannot be
    collected without it. Skipping the whole module is right: a collection
    error reads as a broken suite rather than as data the reader has not
    obtained.

    Only `ObservationsUnavailable` is intercepted. Other import failures --
    an optional dependency, the Fortran oracle harness -- are left to
    pytest's own handling, which already reports them as skips with their
    own reasons.
    """
    try:
        return (yield)
    except ObservationsUnavailable as exc:
        raise pytest.skip.Exception(
            _reason(exc), allow_module_level=True
        ) from exc
