"""
Locating the validation observations, which are not distributed with slabx.

Every observation used in this work came from a published source downloaded
from its official site, but none of it is redistributed here: the permissions
that exist cover reading and using rather than re-hosting, and in several
cases no licence was published at all. See
`validation/data/provenance/README.md` for the file-by-file position.

The code runs without any of it. Anything that needs an observation raises
`ObservationsUnavailable`, which the tests turn into a skip rather than a
failure, so a reader can verify everything that does not depend on
third-party data and see plainly which results do.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["ObservationsUnavailable", "require", "data_root", "ENV_VAR"]

#: Where the package's own files live -- the provenance manifests and the
#: exclusion inventory, both written for this work.
DATA = Path(__file__).parent / "data"
PROVENANCE = DATA / "provenance"

#: Where third-party observations are looked for.
#:
#: They are deliberately *not* sought inside the installed package. Putting
#: them there invites them into a wheel or a commit by accident, and the
#: whole point is that they are the user's copy rather than ours.
ENV_VAR = "SLABX_VALIDATION_DATA"


def data_root() -> Path:
    """
    Directory holding third-party observations.

    ``$SLABX_VALIDATION_DATA`` if set, otherwise ``./validation-data``
    relative to the working directory. Neither needs to exist; the callers
    raise `ObservationsUnavailable` when a file they want is not there.
    """
    import os

    env = os.environ.get(ENV_VAR)
    return Path(env).expanduser() if env else Path.cwd() / "validation-data"


class ObservationsUnavailable(FileNotFoundError):
    """A required observation file is not present.

    Carries the manifest entry for the file so the message says where the
    data came from rather than only that it is missing.
    """

    def __init__(self, name: str, hint: str = ""):
        self.name = name
        super().__init__(
            f"{name} is not distributed with slabx.\n"
            f"{hint}"
            f"Obtain the original yourself and place it at\n"
            f"    {data_root() / name}\n"
            f"or set {ENV_VAR} to the directory holding it.\n"
            f"`{PROVENANCE / 'variable_mapping.csv'}` gives the source "
            f"document, table and columns for each derived file, and "
            f"`source_manifest.csv` gives the download URL and the terms "
            f"as read on the date recorded there."
        )


def _hint_for(name: str) -> str:
    """The manifest line for this file, if the manifests are present."""
    try:
        import csv

        for row in csv.DictReader(open(PROVENANCE / "variable_mapping.csv")):
            if row["local_file"] == name:
                ds = row["dataset_id"]
                for src in csv.DictReader(
                        open(PROVENANCE / "source_manifest.csv")):
                    if src["dataset_id"] == ds:
                        return (f"Source: {src['document']} "
                                f"({src['publisher']}), "
                                f"{row['source_location']}\n"
                                f"  {src['official_url']}\n")
                return f"Source: {ds}, {row['source_location']}\n"
    except Exception:                                        # noqa: BLE001
        pass
    return ""


def require(name: str) -> Path:
    """
    Path to an observation file, or `ObservationsUnavailable`.

    Looks in the user's data directory first, then inside the package. The
    second is only so that a working copy with the observations in place
    keeps running; a published install has nothing there but the manifests.
    """
    for path in (data_root() / name, DATA / name):
        if path.exists():
            return path
    raise ObservationsUnavailable(name, _hint_for(name))
