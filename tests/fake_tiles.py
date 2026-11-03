"""A fake Google spotlit tile server for tests.

It draws tiles exactly like Google does: transparent inside the area,
light gray (low alpha) outside, red line on the border.
"""
import cv2
import numpy as np

from app import geo

DIM_BGRA = (170, 164, 158, 42)   # real value seen in Google tiles
RED_BGRA = (52, 68, 235, 255)


def make_fetcher(rings_latlng, calls=None):
    """rings_latlng: [outer, hole, ...], each a list of (lat, lng)."""

    def fetch(z, x, y):
        if calls is not None:
            calls.append((z, x, y))
        size = geo.TILE_SIZE
        ox, oy = x * size, y * size
        pts = []
        for ring in rings_latlng:
            p = [geo.latlng_to_pixel(lat, lng, z) for lat, lng in ring]
            pts.append(np.array([[px - ox, py - oy] for px, py in p], np.int32))
        inside = np.zeros((size, size), np.uint8)
        cv2.fillPoly(inside, pts, 255)  # even-odd: holes stay empty
        img = np.zeros((size, size, 4), np.uint8)
        img[inside == 0] = DIM_BGRA
        cv2.polylines(img, pts, True, RED_BGRA, 1)
        ok, png = cv2.imencode(".png", img)
        return png.tobytes()

    return fetch


def empty_fetcher(z, x, y):
    ok, png = cv2.imencode(".png", np.zeros((256, 256, 4), np.uint8))
    return png.tobytes()
