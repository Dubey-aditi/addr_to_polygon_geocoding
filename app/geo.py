"""Web Mercator math: the projection Google Maps tiles use.

The world at zoom z is a square of (256 * 2**z) pixels.
Tile (x, y) covers pixels [x*256, (x+1)*256) and [y*256, (y+1)*256).
"""
from __future__ import annotations

import math
from typing import Tuple

TILE_SIZE = 256
MAX_LAT = 85.05112878  # Mercator cut-off


def world_size(z: int) -> int:
    return TILE_SIZE * (2 ** z)


def latlng_to_pixel(lat: float, lng: float, z: int) -> Tuple[float, float]:
    """Global pixel (x, y) for a lat/lng at zoom z."""
    lat = max(min(lat, MAX_LAT), -MAX_LAT)
    size = world_size(z)
    x = (lng + 180.0) / 360.0 * size
    s = math.sin(math.radians(lat))
    y = (0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * size
    return x, y


def pixel_to_latlng(x: float, y: float, z: int) -> Tuple[float, float]:
    """Inverse of latlng_to_pixel."""
    size = world_size(z)
    lng = x / size * 360.0 - 180.0
    n = math.pi * (1 - 2 * y / size)
    lat = math.degrees(math.atan(math.sinh(n)))
    return lat, lng


def latlng_to_tile(lat: float, lng: float, z: int) -> Tuple[int, int]:
    x, y = latlng_to_pixel(lat, lng, z)
    last = 2 ** z - 1
    return min(max(int(x // TILE_SIZE), 0), last), min(max(int(y // TILE_SIZE), 0), last)


def bbox_to_tile_range(bbox, z: int) -> Tuple[int, int, int, int]:
    """(south, west, north, east) -> (x0, y0, x1, y1) inclusive tile range."""
    south, west, north, east = bbox
    x0, y0 = latlng_to_tile(north, west, z)  # top-left
    x1, y1 = latlng_to_tile(south, east, z)  # bottom-right
    return x0, y0, x1, y1


def meters_per_pixel(lat: float, z: int) -> float:
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** z)
