"""Step 1: turn an address string into a Google place.

Uses the same internal call Google Maps makes when you press Enter:
    GET https://www.google.com/search?tbm=map&q=<query>
No cookies, API key or pb param are needed.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import requests

from . import jspb

SEARCH_URL = "https://www.google.com/search"
HEADERS = {
    "user-agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/156.0.0.0 Safari/537.36"
    ),
    "accept-language": "en-US,en;q=0.9",
}

# --- Our "schema": JSPB positions we reverse engineered ----------------------
RESULTS = (0, 1)          # data[0][1] -> list of result entries
PLACE_IN_ENTRY = 14       # entry[14] -> place block
F_CENTER = 9              # place[9]  -> [null, null, lat, lng]
F_PLACE_ID = 10           # place[10] -> "0x...:0x..."
F_FULL_NAME = 18          # place[18] -> "HSR Layout, Bengaluru, Karnataka"
F_TYPE_BLOB = 209         # place[209] -> base64 protobuf, field 18 = type
TYPE_BLOB_FIELD = 18
CAMERA = (1,)             # data[1] -> [[altitude_m, lng, lat], .., [w, h], fov]
CAMERA_FOV_DEG = 13.1     # vertical field of view Google uses


class PlaceNotFound(Exception):
    pass


@dataclass
class Place:
    place_id: str
    full_name: str | None
    lat: float
    lng: float
    place_type: str | None        # e.g. sublocality1, locality, postal_code
    bbox: tuple[float, float, float, float] | None  # (south, west, north, east)

    def to_dict(self) -> dict:
        return asdict(self)


def camera_to_bbox(camera: list) -> tuple[float, float, float, float] | None:
    """Convert Google's camera [[altitude_m, lng, lat], .., [w, h], fov] to a bbox.

    Visible ground height = 2 * altitude * tan(fov / 2); width scales with
    the screen aspect ratio.
    """
    alt = jspb.get(camera, (0, 0))
    lng = jspb.get(camera, (0, 1))
    lat = jspb.get(camera, (0, 2))
    w = jspb.get(camera, (2, 0), 1024)
    h = jspb.get(camera, (2, 1), 768)
    fov = jspb.get(camera, (3,), CAMERA_FOV_DEG)
    if None in (alt, lng, lat):
        return None
    height_m = 2 * alt * math.tan(math.radians(fov) / 2)
    width_m = height_m * w / h
    dlat = height_m / 2 / 111_320
    dlng = width_m / 2 / (111_320 * max(math.cos(math.radians(lat)), 1e-6))
    return (lat - dlat, lng - dlng, lat + dlat, lng + dlng)


def parse_place(entry: list, camera: list | None = None) -> Place | None:
    place = jspb.get(entry, (PLACE_IN_ENTRY,))
    place_id = jspb.get(place, (F_PLACE_ID,))
    if not place_id:
        return None
    blob = jspb.get(place, (F_TYPE_BLOB,))
    place_type = None
    if isinstance(blob, str):
        place_type = jspb.decode_proto_strings(blob).get(TYPE_BLOB_FIELD)
    return Place(
        place_id=place_id,
        full_name=jspb.get(place, (F_FULL_NAME,)),
        lat=jspb.get(place, (F_CENTER, 2)),
        lng=jspb.get(place, (F_CENTER, 3)),
        place_type=place_type,
        bbox=camera_to_bbox(camera) if camera else None,
    )


def parse_response(data: list) -> Place:
    """Pick Google's top result (what Maps jumps to)."""
    camera = jspb.get(data, CAMERA)
    for entry in jspb.get(data, RESULTS, []):
        place = parse_place(entry, camera)
        if place:
            return place
    raise PlaceNotFound("Google returned no single place for this query")


def search_raw(query: str, hl: str = "en", gl: str | None = None,
               session: requests.Session | None = None, timeout: float = 10) -> list:
    """Call the search API and return the parsed JSPB response."""
    params = {"tbm": "map", "hl": hl, "q": query}
    if gl:
        params["gl"] = gl
    http = session or requests
    resp = http.get(SEARCH_URL, params=params, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return jspb.loads(resp.text)


def search(query: str, hl: str = "en", gl: str | None = None,
           session: requests.Session | None = None, timeout: float = 10) -> Place:
    return parse_response(search_raw(query, hl, gl, session, timeout))


if __name__ == "__main__":
    import json
    import sys

    q = " ".join(sys.argv[1:]) or "Koramangala, Bangalore"
    print(json.dumps(search(q).to_dict(), indent=2))
