# Address → Polygon (Google Maps boundary)

Give it any place text (an address, locality, city, state, postcode or country) and it returns the **same area outline Google Maps draws**, as GeoJSON.

```bash
python -m app.polygon "Koramangala, Bangalore" > koramangala.geojson
```

Paste the output into [geojson.io](https://geojson.io) to see it on a map.

## How it works

Google Maps never sends the border as coordinates. It only **draws** it into map images. So we:

1. **Search:** call the internal search API Maps uses (`/search?tbm=map&q=...`). The response is protobuf sent as JSPB (JSON arrays with no keys). From it we take the **place ID** and the **camera view**, which gives the area to look at.
2. **Tiles:** download Google's **spotlit** map tiles for that place ID. They show the place clear, everything else in light gray, and a red dashed border.
3. **Trace:** stitch the tiles, mark gray as outside, trace the edge with OpenCV, and convert each edge pixel to lat/lng with Web Mercator math.

No API key, login or cookies are needed.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.9+.

## Usage

| Command | What you get |
| --- | --- |
| `python -m app.polygon "<place>"` | GeoJSON printed to the terminal |
| `python -m app.run "<place>"` | `output/<place>/` with the GeoJSON and a step-by-step report with images (`process/process.md`) |
| `python scripts/run_examples.py` | 13 example inputs covering the main cases, plus `output/examples_summary.md` |
| `pytest -q` | Tests (they don't call Google) |

## Project layout

```
app/
├── google_search.py   text → place ID + camera view (search API)
├── jspb.py            helpers to read JSPB arrays and base64 protobuf
├── geo.py             lat/lng ↔ pixel math (Web Mercator)
├── google_tiles.py    build tile URLs, choose zoom, download and stitch tiles
├── trace.py           stitched image → polygon rings
├── polygon.py         full pipeline → GeoJSON
└── run.py             same pipeline + saves a visual process report
scripts/run_examples.py
tests/
```

## Known limits

- **No outline in Google:** some countries (e.g. India, Japan) and pin-only places (streets, buildings, shops) → `NoBoundary` error
- **Ambiguous input** (e.g. `cafe`) or a bare US zip (`10001`) → `PlaceNotFound`. Add a city or country
- Areas crossing the 180° line (e.g. Fiji) aren't handled yet
- Accuracy depends on zoom: about 2–5 m per pixel for localities, much coarser for states
- This uses internal Google endpoints, which can change at any time and aren't covered by Google's terms. It's for learning and demonstration only
