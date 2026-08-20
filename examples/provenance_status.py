"""
Derive the publication verdict for each data file.

    python3 examples/provenance_status.py

Reads the two manifests in `src/slabx/validation/data/provenance/` and
regenerates `publication_status.csv`. Rights and scientific transformation
are kept in separate files on purpose; this is the only place they meet, and
the verdict is derived rather than typed by hand so it cannot drift from the
rights position it is supposed to follow.

Only our own work is published. Third-party observations are withheld
regardless of what their licence would allow — the decision was to ship code
and provenance rather than data, so a permissive source does not become
published merely because it could be. `UNRESOLVED` is likewise withhold, not
permitted-by-default.
"""

import csv
import sys
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "src/slabx/validation/data"
PROV = DATA / "provenance"


def verdict(dataset_id: str, sources: dict) -> tuple[str, str]:
    if dataset_id == "OWN":
        return "PUBLISH", "authored for this work"
    s = sources.get(dataset_id)
    if s is None:
        return "WITHHOLD", "source not registered in the manifest"
    r = s["redistribution"].strip().upper()
    if r == "NOT PERMITTED":
        return "WITHHOLD", s["basis"][:70]
    if r == "UNRESOLVED":
        return "WITHHOLD", "rights not established"
    # Nothing third-party is redistributed, whatever its licence allows.
    # A permissive source does not become published just because it could
    # be; the decision was to ship code and provenance only.
    return "WITHHOLD", "redistribution not sought; provenance given instead"


def main() -> int:
    sources = {r["dataset_id"]: r
               for r in csv.DictReader(open(PROV / "source_manifest.csv"))}
    rows = list(csv.DictReader(open(PROV / "variable_mapping.csv")))

    out = []
    for r in rows:
        v, why = verdict(r["dataset_id"], sources)
        out.append({**r, "publish_verdict": v, "verdict_basis": why})

    with open(PROV / "publication_status.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    counts = Counter(x["publish_verdict"] for x in out)
    print(f"\n{len(out)} files, {sum(int(x['n_rows']) for x in out)} rows\n")
    for v in ("PUBLISH", "REVIEW", "WITHHOLD"):
        n = counts.get(v, 0)
        rows_n = sum(int(x["n_rows"]) for x in out
                     if x["publish_verdict"] == v)
        print(f"  {v:<10} {n:2d} files {rows_n:5d} rows")

    print("\nnot withheld:")
    for x in out:
        if x["publish_verdict"] != "WITHHOLD":
            print(f"  {x['publish_verdict']:<8} {x['local_file']:<38} "
                  f"{x['verdict_basis'][:44]}")

    missing = [s["dataset_id"] for s in sources.values()
               if not s["downloaded_date"].strip()]
    if missing:
        print(f"\n`downloaded_date` still empty for {len(missing)} sources:")
        print("  " + ", ".join(missing))
        print("  These are the author's own records and are not inferred.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
