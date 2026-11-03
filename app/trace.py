"""Step 3: turn a stitched spotlit image into polygon coordinates."""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from .google_tiles import Mosaic

MIN_DIM_FRACTION = 0.01   # below this there is no "outside" dimming -> no boundary
MIN_AREA_PX = 30          # drop specks smaller than this
MIN_PART_FRACTION = 0.005 # drop parts/holes smaller than 0.5% of the main area
SIMPLIFY_PX = 1.5         # Douglas-Peucker tolerance in pixels
DIM_MAX_ALPHA = 80        # Google's outside dimming is a faint gray (alpha ~27-42)
DIM_MAX_SPREAD = 30       # ...and neutral: its B, G, R values are close together
DIM_MAX_BRIGHT = 230      # ...and not white (label halos are white)


def inside_mask(image: np.ndarray) -> Optional[np.ndarray]:
    """255 where the pixel is inside the area, 0 outside.

    Outside = Google's dimming: faint, neutral gray. Inside = everything else
    (transparent pixels, the red border, and the place's pin and label).
    Returns None if the image has no dimming at all, i.e. no boundary.
    """
    b, g, r, a = (image[..., i].astype(np.int16) for i in range(4))
    spread = np.maximum(np.maximum(b, g), r) - np.minimum(np.minimum(b, g), r)
    dim = ((a > 0) & (a <= DIM_MAX_ALPHA) & (spread <= DIM_MAX_SPREAD)
           & (r < DIM_MAX_BRIGHT))
    if dim.mean() < MIN_DIM_FRACTION:
        return None
    mask = np.where(dim, 0, 255).astype(np.uint8)
    # Remove tiny noise (anti-aliasing, label dots)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def touches_edges(mask: np.ndarray) -> dict:
    """Which mosaic edges the inside region touches (area may continue there)."""
    return {
        "north": bool(mask[0, :].any()),
        "south": bool(mask[-1, :].any()),
        "west": bool(mask[:, 0].any()),
        "east": bool(mask[:, -1].any()),
    }


def _ring_to_lnglat(contour: np.ndarray, mosaic: Mosaic) -> List[List[float]]:
    ring = []
    for px, py in contour.reshape(-1, 2):
        lat, lng = mosaic.pixel_to_latlng(px + 0.5, py + 0.5)
        ring.append([round(lng, 7), round(lat, 7)])
    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def mask_to_polygons(mask: np.ndarray, mosaic: Mosaic) -> List[List[List[List[float]]]]:
    """Return a list of polygons; each polygon = [outer_ring, hole1, hole2, ...]."""
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]
    outers = [cv2.contourArea(c) for c, h in zip(contours, hierarchy) if h[3] == -1]
    min_area = max(MIN_AREA_PX, MIN_PART_FRACTION * max(outers, default=0))
    polygons = []
    for i, cnt in enumerate(contours):
        if hierarchy[i][3] != -1:          # this is a hole; handled with its parent
            continue
        if cv2.contourArea(cnt) < min_area:
            continue
        rings = [cv2.approxPolyDP(cnt, SIMPLIFY_PX, True)]
        child = hierarchy[i][2]
        while child != -1:
            if cv2.contourArea(contours[child]) >= min_area:
                rings.append(cv2.approxPolyDP(contours[child], SIMPLIFY_PX, True))
            child = hierarchy[child][0]
        polygon = [_ring_to_lnglat(r, mosaic) for r in rings if len(r) >= 3]
        if polygon:
            polygons.append(polygon)
    # Largest first
    polygons.sort(key=lambda p: -abs(_ring_area(p[0])))
    return polygons


def _ring_area(ring) -> float:
    return 0.5 * sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(ring, ring[1:]))
