"""Run the pipeline on a fixed set of example inputs that cover the main cases.

Usage (from the project root):
    python scripts/run_examples.py

Writes output/<place>/ for each example, plus output/examples_summary.md / .json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import run as run_mod  # noqa: E402
from app.google_search import PlaceNotFound  # noqa: E402

EXAMPLES = [
    ("Koramangala, Bangalore", "Locality (neighbourhood)"),
    ("Bengaluru", "Large city"),
    ("Vidisha", "Small city"),
    ("560034", "Indian pincode"),
    ("Karnataka", "State"),
    ("Manhattan, New York", "US borough"),
    ("SW1A 1AA", "UK postcode"),
    ("東京", "Non-English input"),
    ("koramangla bangalore", "Typo in input"),
    ("Hawaii", "Islands (several parts)"),
    ("India", "Country"),
    ("27th Main Rd, HSR Layout, Bengaluru", "Street (pin only)"),
    ("asdkjhqwe zzxq", "Nonsense input"),
]


def main() -> None:
    rows = []
    for query, case in EXAMPLES:
        print(f"\n=== {case}: {query}")
        t0 = time.time()
        try:
            folder = run_mod.run(query)
            summary = json.loads((folder / "summary.json").read_text())
            summary["folder"] = str(folder.relative_to(run_mod.OUTPUT_DIR))
        except PlaceNotFound as exc:
            summary = {"query": query, "status": "not_found", "error": str(exc)}
        except Exception as exc:  # keep going; record the failure
            summary = {"query": query, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
        summary["case"] = case
        summary["total_s"] = round(time.time() - t0, 2)
        rows.append(summary)
        time.sleep(1)  # be gentle with Google

    out = run_mod.OUTPUT_DIR
    out.mkdir(exist_ok=True)
    (out / "examples_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    md = ["| Case | Input | Status | Google picked | Type | Zoom | Tiles | Polygons | Holes | Points | Area km² | Time s |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        md.append("| {case} | `{query}` | {status} | {resolved_name} | {place_type} | {zoom} | {tiles} | "
                  "{polygons} | {holes} | {points} | {area} | {total_s} |".format(
                      case=r["case"], query=r["query"], status=r["status"],
                      resolved_name=r.get("resolved_name", "-"), place_type=r.get("place_type", "-"),
                      zoom=r.get("zoom", "-"), tiles=r.get("tiles_downloaded", "-"),
                      polygons=r.get("polygons", "-"), holes=r.get("holes", "-"),
                      points=r.get("points", "-"), area=r.get("area_km2", "-"), total_s=r["total_s"]))
    (out / "examples_summary.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md))
    print(f"\nsaved -> {out / 'examples_summary.md'}")


if __name__ == "__main__":
    main()
