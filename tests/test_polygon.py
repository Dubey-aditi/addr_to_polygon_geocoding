import math

import pytest

from app import geo, google_tiles, trace
from app.google_search import Place
from app.polygon import NoBoundary, place_polygons, to_geojson
from tests.fake_tiles import empty_fetcher, make_fetcher

# A rough Koramangala-sized pentagon
SHAPE = [(12.950, 77.615), (12.948, 77.640), (12.930, 77.650), (12.922, 77.628), (12.932, 77.612)]


def make_place(bbox):
    return Place(place_id="0x3bae144ed898fc2d:0x1681f38e8c00ae56", full_name="Test area",
                 lat=12.936, lng=77.63, place_type="sublocality1", bbox=bbox)


def nearest_m(lat, lng, ring):
    """Distance in meters from (lat, lng) to the closest traced vertex."""
    best = math.inf
    for x, y in ring:
        dy = (y - lat) * 111_320
        dx = (x - lng) * 111_320 * math.cos(math.radians(lat))
        best = min(best, math.hypot(dx, dy))
    return best


def test_place_id_to_u64_matches_google():
    # Values copied from a real Google tile request
    assert google_tiles.place_id_to_u64("0x3bae144ed898fc2d:0x1681f38e8c00ae56") == (
        "4300397023059311661", "1621845134389849686")


def test_build_tile_url():
    url = google_tiles.build_tile_url("0x1:0x2", 15, 10, 20)
    assert "!1i15!2i10!3i20" in url and "!2sspotlit" in url and "!1y1!2y2" in url


def test_geo_roundtrip():
    x, y = geo.latlng_to_pixel(12.9352, 77.6245, 16)
    lat, lng = geo.pixel_to_latlng(x, y, 16)
    assert abs(lat - 12.9352) < 1e-9 and abs(lng - 77.6245) < 1e-9


def test_choose_zoom_small_vs_large():
    small = (12.92, 77.61, 12.95, 77.65)   # neighbourhood
    large = (11.5, 74.0, 18.5, 78.6)       # state
    assert google_tiles.choose_zoom(small) > google_tiles.choose_zoom(large)


def test_traces_known_shape():
    place = make_place((12.915, 77.605, 12.957, 77.657))
    polys, mosaic = place_polygons(place, make_fetcher([SHAPE]))
    assert len(polys) == 1
    outer = polys[0][0]
    assert outer[0] == outer[-1]                         # closed ring
    tol = 3 * geo.meters_per_pixel(12.93, mosaic.z)     # within 3 pixels
    for lat, lng in SHAPE:
        assert nearest_m(lat, lng, outer) < tol


def test_grows_when_bbox_too_small():
    calls = []
    place = make_place((12.935, 77.625, 12.940, 77.635))  # only the middle
    polys, _ = place_polygons(place, make_fetcher([SHAPE], calls), max_tiles=4)
    outer = polys[0][0]
    lats = [p[1] for p in outer]
    lngs = [p[0] for p in outer]
    assert min(lats) < 12.925 and max(lats) > 12.947
    assert min(lngs) < 77.615 and max(lngs) > 77.647


def test_hole_is_kept():
    hole = [(12.940, 77.628), (12.940, 77.636), (12.934, 77.636), (12.934, 77.628)]
    place = make_place((12.915, 77.605, 12.957, 77.657))
    polys, _ = place_polygons(place, make_fetcher([SHAPE, hole]))
    assert len(polys[0]) == 2   # outer + 1 hole


def test_no_boundary_raises():
    place = make_place((12.915, 77.605, 12.957, 77.657))
    with pytest.raises(NoBoundary):
        place_polygons(place, empty_fetcher)


def test_geojson_shape():
    out = to_geojson([[[[1, 2], [3, 4], [5, 6], [1, 2]]]])
    assert out["type"] == "FeatureCollection"
    assert out["features"][0]["geometry"]["type"] == "Polygon"
    assert out["features"][0]["properties"] == {}


def test_run_writes_process_report():
    import tempfile
    from pathlib import Path
    from app import run as run_mod
    from tests.test_google_search import HSR_PLACE, make_response

    old_dir, run_mod.OUTPUT_DIR = run_mod.OUTPUT_DIR, Path(tempfile.mkdtemp())
    hsr_shape = [(12.925, 77.630), (12.924, 77.660), (12.900, 77.662), (12.895, 77.632)]
    try:
        folder = run_mod.run("hsr layout", data=make_response(HSR_PLACE),
                             fetch=make_fetcher([hsr_shape]))
    finally:
        run_mod.OUTPUT_DIR = old_dir
    proc = folder / "process"
    for f in ["process.md", "1_search_response.json", "1_place.json",
              "2_tiles_stitched.png", "3_mask.png", "4_trace.png"]:
        assert (proc / f).exists(), f
    assert list((proc / "2_tiles").glob("*.png"))
    assert (folder / f"{folder.name}.geojson").exists()


def test_ignores_undimmed_tile_without_border():
    """Google sometimes returns one tile with no dimming; it has no red edge."""
    from tests.fake_tiles import empty_fetcher
    real = make_fetcher([SHAPE])
    place = make_place((12.915, 77.605, 12.957, 77.657))
    z = google_tiles.choose_zoom(place.bbox)
    x0, y0, _, _ = geo.bbox_to_tile_range(place.bbox, z)

    def fetch(zz, x, y):
        return empty_fetcher(zz, x, y) if (x, y) == (x0, y0) else real(zz, x, y)

    polys, _ = place_polygons(place, fetch)
    assert len(polys) == 1
