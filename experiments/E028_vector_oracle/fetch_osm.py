"""Fetch E028 ground-truth vectors (roads, buildings) from OpenStreetMap via Overpass.

Data © OpenStreetMap contributors, ODbL. Stored outside the repo; not redistributed.

    uv run python experiments/E028_vector_oracle/fetch_osm.py --out "$E028_DATA"
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from pyproj import Transformer

# (south, west, north, east); about 3 km x 3 km each.
AREAS = {
    "innaloo": (-31.905, 115.785, -31.878, 115.817),  # tune: suburbs + Mitchell Fwy interchanges
    "murdoch": (-32.075, 115.840, -32.048, 115.872),  # test: Kwinana Fwy / South St, suburbs
    "northam": (-31.665, 116.655, -31.638, 116.687),  # test: regional town, rail, sheds
}
# Drivable public roads (roughly City-Scale's selection); service roads and tracks excluded.
ROAD_CLASSES = (
    "motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street"
    "|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link"
)
URL = "https://overpass-api.de/api/interpreter"
CRS = "EPSG:7850"  # GDA2020 / MGA zone 50, metres


def overpass(query: str) -> dict:
    data = urllib.parse.urlencode({"data": query}).encode()
    req = urllib.request.Request(URL, data=data, headers={"User-Agent": "ManySeg-E028/0.1 (research)"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)


def fetch(bbox: tuple[float, float, float, float]) -> dict:
    b = ",".join(map(str, bbox))
    q = (
        f'[out:json][timeout:120];(way["highway"~"^({ROAD_CLASSES})$"]({b});'
        f'way["building"]({b}););out body geom;'
    )
    return overpass(q)


def to_features(raw: dict) -> dict:
    tf = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    roads, buildings = [], []
    for el in raw["elements"]:
        if el["type"] != "way" or "geometry" not in el:
            continue
        xy = [list(tf.transform(p["lon"], p["lat"])) for p in el["geometry"]]
        tags = el.get("tags", {})
        if "building" in tags:
            if len(xy) >= 4 and xy[0] == xy[-1]:
                buildings.append({"id": el["id"], "coords": xy})
        elif tags.get("tunnel", "no") == "no" and tags.get("covered", "no") == "no":
            layer = int(tags.get("layer", "0")) if tags.get("layer", "0").lstrip("-").isdigit() else 0
            roads.append({
                "id": el["id"], "coords": xy, "highway": tags["highway"],
                "bridge": tags.get("bridge", "no") != "no", "layer": layer,
                "nodes": el["nodes"],
            })
    return {"crs": CRS, "roads": roads, "buildings": buildings}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, bbox in AREAS.items():
        path = args.out / f"{name}.json"
        if path.exists():
            continue
        raw = fetch(bbox)
        (args.out / f"{name}_raw.json").write_text(json.dumps(raw))
        feats = to_features(raw)
        feats["bbox_wgs84"] = bbox
        feats["osm_timestamp"] = raw["osm3s"]["timestamp_osm_base"]
        path.write_text(json.dumps(feats))
        print(name, len(feats["roads"]), "roads", len(feats["buildings"]), "buildings")
        time.sleep(5)


if __name__ == "__main__":
    main()
