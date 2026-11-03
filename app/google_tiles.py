"""Step 2: download Google's "spotlit" tiles for a place.

When you select an area, Google Maps requests extra map tiles with the
"spotlit" layer. Without the base road layer, these tiles contain only:
    - transparent pixels    -> inside the area
    - light gray, low alpha -> outside the area (the dimming)
    - red pixels            -> the dashed border
Places with no boundary (shops, buildings) come back fully transparent
or with only a pin.

Reverse engineered minimal URL (pb = protobuf-in-URL, "!<field><type><value>"):
    !1m4!1m3!1i{z}!2i{x}!3i{y}          tile coordinates
    !2m2!1e2!2sspotlit                  layer = spotlit only (no roads)
    !3m6!5e1105!12m4!1e68!2m2!1sset!2sRoadmap   render style (required)
    !27m7!299174093m6!14m5!1m4!1m2!1y{hi}!2y{lo}!8b1   the place to highlight
The place ID "0xHI:0xLO" is sent as two unsigned 64-bit decimals.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
import requests

from . import geo
from .google_search import HEADERS

TILE_URL = "https://www.google.com/maps/vt/pb="
STYLE = "!3m6!5e1105!12m4!1e68!2m2!1sset!2sRoadmap"
MAX_ZOOM = 18

TileFetcher = Callable[[int, int, int], Optional[bytes]]


def place_id_to_u64(place_id: str) -> Tuple[str, str]:
    """'0x3bae...:0x1681...' -> ('4300397023059311661', '1621845134389849686')."""
    hi, lo = place_id.split(":")
    return str(int(hi, 16)), str(int(lo, 16))


def build_tile_url(place_id: str, z: int, x: int, y: int) -> str:
    hi, lo = place_id_to_u64(place_id)
    return (
        f"{TILE_URL}!1m4!1m3!1i{z}!2i{x}!3i{y}"
        f"!2m2!1e2!2sspotlit{STYLE}"
        f"!27m7!299174093m6!14m5!1m4!1m2!1y{hi}!2y{lo}!8b1"
    )


def choose_zoom(bbox, max_tiles: int = 64) -> int:
    """Highest zoom where the bbox fits in at most max_tiles tiles."""
    for z in range(MAX_ZOOM, -1, -1):
        x0, y0, x1, y1 = geo.bbox_to_tile_range(bbox, z)
        if (x1 - x0 + 1) * (y1 - y0 + 1) <= max_tiles:
            return z
    return 0


def http_fetcher(place_id: str, session: Optional[requests.Session] = None,
                 timeout: float = 10) -> TileFetcher:
    http = session or requests.Session()

    def fetch(z: int, x: int, y: int) -> Optional[bytes]:
        resp = http.get(build_tile_url(place_id, z, x, y), headers=HEADERS, timeout=timeout)
        if resp.status_code != 200 or not resp.headers.get("content-type", "").startswith("image"):
            return None
        return resp.content

    return fetch


def decode_png(data: Optional[bytes]) -> np.ndarray:
    """PNG bytes -> 256x256x4 BGRA array (empty tile if missing)."""
    empty = np.zeros((geo.TILE_SIZE, geo.TILE_SIZE, 4), np.uint8)
    if not data:
        return empty
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return empty
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


class Mosaic:
    """Stitched tiles: image + where it sits in the world at zoom z."""

    def __init__(self, image: np.ndarray, z: int, x0: int, y0: int):
        self.image = image          # H x W x 4 (BGRA)
        self.z = z
        self.x0 = x0                # top-left tile
        self.y0 = y0

    def pixel_to_latlng(self, px: float, py: float) -> Tuple[float, float]:
        return geo.pixel_to_latlng(self.x0 * geo.TILE_SIZE + px, self.y0 * geo.TILE_SIZE + py, self.z)


def fetch_mosaic(fetch: TileFetcher, z: int, x0: int, y0: int, x1: int, y1: int,
                 workers: int = 8) -> Mosaic:
    size = geo.TILE_SIZE
    cols, rows = x1 - x0 + 1, y1 - y0 + 1
    image = np.zeros((rows * size, cols * size, 4), np.uint8)
    coords = [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tiles = list(pool.map(lambda c: decode_png(fetch(z, c[0], c[1])), coords))
    for (x, y), tile in zip(coords, tiles):
        r, c = (y - y0) * size, (x - x0) * size
        image[r:r + size, c:c + size] = tile
    return Mosaic(image, z, x0, y0)
