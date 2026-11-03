from app import jspb
from app.google_search import PlaceNotFound, camera_to_bbox, parse_response

import pytest


def make_place(**fields):
    """Build a place block with values at their real JSPB positions."""
    place = [None] * 210
    for idx, val in fields.items():
        place[int(idx.lstrip("f"))] = val
    return place


HSR_PLACE = make_place(
    f9=[None, None, 12.912118099999999, 77.6445548],
    f10="0x3bae1491bfdc6ecd:0xf232718439fbc879",
    f11="HSR Layout",
    f18="HSR Layout, Bengaluru, Karnataka",
    f89="/m/07vf48",
    f183=[None, ["HSR Layout", None, None, "Bengaluru", None, "Karnataka", "IN"]],
    f209="Cgpoc3IgbGF5b3V0kgEMc3VibG9jYWxpdHkx4AEA",  # real blob from Google
)
HSR_CAMERA = [[31111.8103251424, 77.6458227, 12.909237999999998], [0, 0, 0], [1024, 768], 13.1]


def make_response(place, camera=HSR_CAMERA):
    entry = [None] * 14 + [place]
    return [["hsr layout", [entry]], camera]


def test_loads_strips_xssi_prefix():
    assert jspb.loads(")]}'\n[1, 2]") == [1, 2]


def test_get_is_safe():
    assert jspb.get([[1, [2, 3]]], (0, 1, 1)) == 3
    assert jspb.get([[1]], (0, 5, 2), "x") == "x"
    assert jspb.get(None, (0,)) is None


def test_decode_type_blob():
    fields = jspb.decode_proto_strings("Cgpoc3IgbGF5b3V0kgEMc3VibG9jYWxpdHkx4AEA")
    assert fields[1] == "hsr layout"
    assert fields[18] == "sublocality1"


def test_parse_hsr_layout():
    p = parse_response(make_response(HSR_PLACE))
    assert p.place_id == "0x3bae1491bfdc6ecd:0xf232718439fbc879"
    assert p.full_name == "HSR Layout, Bengaluru, Karnataka"
    assert (p.lat, p.lng) == (12.912118099999999, 77.6445548)
    assert p.place_type == "sublocality1"


def test_bbox_contains_center_and_is_reasonable():
    s, w, n, e = camera_to_bbox(HSR_CAMERA)
    assert s < 12.912 < n and w < 77.644 < e
    # HSR Layout is a few km across; view should be roughly 5-15 km
    assert 0.04 < (n - s) < 0.15


def test_no_place_raises():
    with pytest.raises(PlaceNotFound):
        parse_response([["xyz", []], None])
