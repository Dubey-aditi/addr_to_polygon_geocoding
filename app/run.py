"""Run the pipeline and save the result + a step-by-step process report.

Usage:
    python -m app.run "Koramangala, Bangalore"

Creates:
    output/<place>/
    ├── <place>.geojson            final result (paste into geojson.io)
    └── process/
        ├── process.md             what happened at each step
        ├── 1_search_response.json raw JSPB response from the search API
        ├── 1_place.json           fields we picked out of it
        ├── 2_tiles/               every downloaded tile (z_x_y.png)
        ├── 2_tiles_stitched.png   all tiles glued together, with the tile grid
        ├── 3_mask.png             white = inside, black = outside
        └── 4_trace.png            traced outline + its points on top of the tiles
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests

from . import geo, google_tiles, trace
from .google_search import SEARCH_URL, parse_response, search_raw
from .polygon import NoBoundary, place_polygons, to_geojson

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
WHITE = 255
GRID_BGR = (230, 200, 150)     # light blue tile grid
LINE_BGR = (204, 85, 0)        # our outline (blue)
POINT_BGR = (0, 140, 255)      # our points (orange)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "place"


# ---------- images ----------------------------------------------------------

def on_white(bgra: np.ndarray) -> np.ndarray:
    """Put a transparent image on a white background so the gray is visible."""
    alpha = bgra[..., 3:4].astype(np.float32) / 255.0
    out = bgra[..., :3].astype(np.float32) * alpha + WHITE * (1 - alpha)
    return out.astype(np.uint8)


def draw_tile_grid(img: np.ndarray, mosaic: google_tiles.Mosaic) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    size = geo.TILE_SIZE
    for c in range(0, w + 1, size):
        cv2.line(out, (c, 0), (c, h), GRID_BGR, 1)
    for r in range(0, h + 1, size):
        cv2.line(out, (0, r), (w, r), GRID_BGR, 1)
    for r in range(0, h, size):
        for c in range(0, w, size):
            label = f"{mosaic.x0 + c // size},{mosaic.y0 + r // size}"
            cv2.putText(out, label, (c + 6, r + 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (120, 120, 120), 1, cv2.LINE_AA)
    return out


def draw_trace(img: np.ndarray, polygons, mosaic: google_tiles.Mosaic) -> np.ndarray:
    out = img.copy()
    ox, oy = mosaic.x0 * geo.TILE_SIZE, mosaic.y0 * geo.TILE_SIZE
    for poly in polygons:
        for ring in poly:
            pts = np.array([[x - ox, y - oy] for x, y in
                            (geo.latlng_to_pixel(lat, lng, mosaic.z) for lng, lat in ring)],
                           np.int32)
            cv2.polylines(out, [pts], True, LINE_BGR, 2, cv2.LINE_AA)
            for p in pts:
                cv2.circle(out, tuple(int(v) for v in p), 3, POINT_BGR, -1, cv2.LINE_AA)
    return out


# ---------- report ----------------------------------------------------------

def run(query: str, name: Optional[str] = None, max_tiles: int = 64,
        session: Optional[requests.Session] = None,
        data: Optional[list] = None,
        fetch: Optional[google_tiles.TileFetcher] = None) -> Path:
    """Run every step, saving what each one produced. `data`/`fetch` are for tests."""
    session = session or requests.Session()
    md = [f"# Process: `{query}`", ""]

    # Step 1: search
    t0 = time.time()
    if data is None:
        data = search_raw(query, session=session)
    place = parse_response(data)
    t_search = time.time() - t0

    folder = OUTPUT_DIR / slugify(name or (place.full_name or query).split(",")[0])
    proc = folder / "process"
    (proc / "2_tiles").mkdir(parents=True, exist_ok=True)
    (proc / "1_search_response.json").write_text(json.dumps(data, ensure_ascii=False))
    (proc / "1_place.json").write_text(json.dumps(place.to_dict(), indent=2, ensure_ascii=False))

    s, w, n, e = place.bbox or (None,) * 4
    md += [
        "## Step 1 · Search API: text → place ID + camera",
        "",
        f"- Request: `{SEARCH_URL}?tbm=map&hl=en&q={requests.utils.quote(query)}`",
        f"- Time: {t_search:.2f} s",
        f"- Raw response (JSPB): [1_search_response.json](1_search_response.json)",
        "",
        "| Field | JSPB path | Value |",
        "| --- | --- | --- |",
        f"| Place ID | `data[0][1][0][14][10]` | `{place.place_id}` |",
        f"| Full name | `data[0][1][0][14][18]` | {place.full_name} |",
        f"| Center (lat, lng) | `data[0][1][0][14][9]` | {place.lat}, {place.lng} |",
        f"| Type | `data[0][1][0][14][209]` → field 18 | {place.place_type} |",
        f"| Camera | `data[1]` | `{json.dumps(data[1]) if len(data) > 1 else None}` |",
        f"| Bounding box (from camera) | computed | S {s:.5f}, W {w:.5f}, N {n:.5f}, E {e:.5f} |"
        if place.bbox else "| Bounding box | computed | none |",
        "",
    ]

    # Steps 2–3: tiles + trace
    log: list = []
    fetch = fetch or google_tiles.http_fetcher(place.place_id, session=session)
    first_zoom = google_tiles.choose_zoom(place.bbox, max_tiles) if place.bbox else None
    t0 = time.time()
    try:
        polygons, mosaic = place_polygons(place, fetch, max_tiles=max_tiles, log=log)
        error = None
    except NoBoundary as exc:
        polygons, mosaic, error = [], None, str(exc)
    t_tiles = time.time() - t0

    hi, lo = google_tiles.place_id_to_u64(place.place_id)
    md += [
        "## Step 2 · Zoom + spotlit tiles",
        "",
        f"- Place ID as two 64-bit numbers: `{hi}` and `{lo}`",
        f"- First zoom chosen (fits ≤ {max_tiles} tiles): **{first_zoom}**",
        f"- Time (download + trace): {t_tiles:.2f} s",
        "",
        "| # | Zoom | Tile range (x0, y0 → x1, y1) | Tiles | What happened |",
        "| --- | --- | --- | --- | --- |",
    ]
    for i, a in enumerate(log, 1):
        x0, y0, x1, y1 = a["tile_range"]
        md.append(f"| {i} | {a['zoom']} | {x0}, {y0} → {x1}, {y1} | {a['tiles']} | {a['result']} |")
    md.append("")

    if mosaic is None:
        md += ["## Result", "", f"**No polygon:** {error}", ""]
        (proc / "process.md").write_text("\n".join(md))
        print(f"{place.full_name}: no boundary\nsaved -> {folder}")
        return folder

    size = geo.TILE_SIZE
    rows, cols = mosaic.image.shape[0] // size, mosaic.image.shape[1] // size
    for r in range(rows):
        for c in range(cols):
            tile = mosaic.image[r * size:(r + 1) * size, c * size:(c + 1) * size]
            cv2.imwrite(str(proc / "2_tiles" / f"{mosaic.z}_{mosaic.x0 + c}_{mosaic.y0 + r}.png"), tile)
    white = on_white(mosaic.image)
    cv2.imwrite(str(proc / "2_tiles_stitched.png"), draw_tile_grid(white, mosaic))

    mpp = geo.meters_per_pixel(place.lat, mosaic.z)
    sample_url = google_tiles.build_tile_url(place.place_id, mosaic.z, mosaic.x0, mosaic.y0)
    md += [
        f"- Final zoom **{mosaic.z}**, grid **{cols} × {rows} = {cols * rows} tiles**, "
        f"image **{cols * size} × {rows * size} px**",
        f"- 1 pixel ≈ **{mpp:.1f} m** at this zoom",
        f"- Example tile URL (top-left): `{sample_url}`",
        "- Every tile: [2_tiles/](2_tiles/) (named `zoom_x_y.png`)",
        "",
        "Stitched tiles on a white background (clear = inside, gray = outside, red = Google's border):",
        "",
        "![stitched tiles](2_tiles_stitched.png)",
        "",
    ]

    # Step 3: mask + contours
    mask = trace.inside_mask(mosaic.image)
    cv2.imwrite(str(proc / "3_mask.png"), mask)
    raw_contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    raw_points = sum(len(c) for c in raw_contours)
    final_points = sum(len(r) for p in polygons for r in p)
    inside_frac = float((mask > 0).mean())
    md += [
        "## Step 3 · Trace: image → border pixels",
        "",
        f"- Inside pixels: **{inside_frac:.1%}** of the image",
        f"- Outlines found by OpenCV: **{len(raw_contours)}** ({raw_points} edge pixels)",
        f"- After simplifying: **{final_points}** points",
        "",
        "Mask (white = inside, black = outside):",
        "",
        "![mask](3_mask.png)",
        "",
    ]

    cv2.imwrite(str(proc / "4_trace.png"), draw_trace(white, polygons, mosaic))
    md += [
        "## Step 4 · Pixels → lat/lng (Web Mercator)",
        "",
        "Our outline (blue) and its points (orange) on top of Google's tiles:",
        "",
        "![trace](4_trace.png)",
        "",
    ]
    if polygons:
        px, py = polygons[0][0][0]
        gx, gy = geo.latlng_to_pixel(py, px, mosaic.z)
        md += [
            "Example, first point:",
            "",
            "```text",
            f"pixel in image   = ({gx - mosaic.x0 * size:.1f}, {gy - mosaic.y0 * size:.1f})",
            f"global pixel     = ({gx:.1f}, {gy:.1f})  at zoom {mosaic.z}",
            f"lng, lat         = {px}, {py}",
            "```",
            "",
        ]

    # Result
    geojson = to_geojson(polygons, place) if polygons else None
    result_path = folder / f"{folder.name}.geojson"
    if geojson:
        result_path.write_text(json.dumps(geojson, indent=2))
    area_km2 = (mask > 0).sum() * (mpp ** 2) / 1e6
    md += [
        "## Result",
        "",
        f"- {len(polygons)} polygon(s), {final_points} points"
        + (f", {sum(len(p) - 1 for p in polygons)} hole(s)" if polygons else ""),
        f"- Area ≈ **{area_km2:,.1f} km²** (pixel count × pixel area, approximate)",
        f"- GeoJSON: [../{result_path.name}](../{result_path.name}) — paste into geojson.io",
        "",
    ]
    (proc / "process.md").write_text("\n".join(md))
    print(f"{place.full_name}: {len(polygons)} polygon(s), {final_points} points, zoom {mosaic.z}, "
          f"{cols * rows} tiles")
    print(f"saved -> {folder}")
    return folder


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Address text -> GeoJSON polygon + process report")
    ap.add_argument("query")
    ap.add_argument("--name", help="output folder name (default: place name)")
    ap.add_argument("--max-tiles", type=int, default=64)
    args = ap.parse_args()
    run(args.query, args.name, args.max_tiles)
