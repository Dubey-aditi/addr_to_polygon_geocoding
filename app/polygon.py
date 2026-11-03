"""Pipeline: address string -> GeoJSON polygon of Google's highlighted area."""
from __future__ import annotations

from typing import Optional

import requests

from . import geo, google_tiles, trace
from .google_search import Place, search


class NoBoundary(Exception):
    """Google found the place but shows only a pin, not an outline."""


def place_polygons(place: Place, fetch: Optional[google_tiles.TileFetcher] = None,
                   max_tiles: int = 64, max_expand: int = 4,
                   zoom_out_retries: int = 2, log: Optional[list] = None):
    """Return (polygons, mosaic) for a place, growing the tile area if needed.

    If `log` is a list, one entry per tile download is appended to it.
    """
    def note(z, x0, y0, x1, y1, result):
        if log is not None:
            log.append({"zoom": z, "tile_range": [x0, y0, x1, y1],
                        "tiles": (x1 - x0 + 1) * (y1 - y0 + 1), "result": result})

    if place.bbox is None:
        raise NoBoundary("No bounding box for this place")
    fetch = fetch or google_tiles.http_fetcher(place.place_id)
    z = google_tiles.choose_zoom(place.bbox, max_tiles)
    x0, y0, x1, y1 = geo.bbox_to_tile_range(place.bbox, z)

    # No gray anywhere means either "no boundary" or "our view is fully
    # inside a big area". Zoom out (with 1 tile padding) to tell them apart.
    for attempt in range(zoom_out_retries + 1):
        mosaic = google_tiles.fetch_mosaic(fetch, z, x0, y0, x1, y1)
        mask = trace.inside_mask(mosaic.image)
        if mask is not None:
            break
        if attempt == zoom_out_retries or z < 2:
            note(z, x0, y0, x1, y1, "no gray found -> no boundary")
            raise NoBoundary(f"Google shows no outline for '{place.full_name}'")
        note(z, x0, y0, x1, y1, "no gray found -> zoom out 2 levels")
        z -= 2
        x0, y0, x1, y1 = geo.bbox_to_tile_range(place.bbox, z)
        last = 2 ** z - 1
        x0, y0, x1, y1 = max(x0 - 1, 0), max(y0 - 1, 0), min(x1 + 1, last), min(y1 + 1, last)
    last = 2 ** z - 1

    for i in range(max_expand + 1):
        if i:
            mosaic = google_tiles.fetch_mosaic(fetch, z, x0, y0, x1, y1)
            mask = trace.inside_mask(mosaic.image)
        edges = trace.touches_edges(mask)
        grow = {
            "north": edges["north"] and y0 > 0,
            "south": edges["south"] and y1 < last,
            "west": edges["west"] and x0 > 0,
            "east": edges["east"] and x1 < last,
        }
        if not any(grow.values()):
            note(z, x0, y0, x1, y1, "area fully inside the image -> done")
            break
        note(z, x0, y0, x1, y1,
             "area touches " + ", ".join(k for k, v in grow.items() if v) + " edge -> add tiles")
        # Area continues beyond what we downloaded -> add one ring of tiles
        y0 -= grow["north"]
        y1 += grow["south"]
        x0 -= grow["west"]
        x1 += grow["east"]

    return trace.mask_to_polygons(mask, mosaic), mosaic


def to_geojson(polygons, place: Optional[Place] = None, include_properties: bool = False) -> dict:
    if len(polygons) == 1:
        geometry = {"type": "Polygon", "coordinates": polygons[0]}
    else:
        geometry = {"type": "MultiPolygon", "coordinates": polygons}
    props = {}
    if include_properties and place:
        props = {
            "name": place.full_name,
            "place_id": place.place_id,
            "place_type": place.place_type,
            "center": [place.lng, place.lat],
        }
    return {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": props, "geometry": geometry}],
    }


def get_polygon(query: str, session: Optional[requests.Session] = None,
                max_tiles: int = 64, include_properties: bool = False) -> dict:
    session = session or requests.Session()
    place = search(query, session=session)
    fetch = google_tiles.http_fetcher(place.place_id, session=session)
    polygons, _ = place_polygons(place, fetch, max_tiles=max_tiles)
    if not polygons:
        raise NoBoundary(f"Could not trace an outline for '{place.full_name}'")
    return to_geojson(polygons, place, include_properties)


if __name__ == "__main__":
    import json
    import sys

    q = " ".join(sys.argv[1:]) or "Koramangala, Bangalore"
    print(json.dumps(get_polygon(q), indent=2))
