"""Fetch 3DBAG CityJSON tiles (LoD2.2 roofs) for E031, verified against the published SHA-256.

Data: "© 3DBAG by tudelft3d and 3DGI", CC BY 4.0 (https://docs.3dbag.nl/en/copyright/).
Release v20250903; tile index from the 3DBAG WFS (layer BAG3D:tiles). Stored outside the repo.

    uv run python experiments/E031_lod2_oracle/fetch_3dbag.py --out "$E031_DATA"
"""

import argparse
import gzip
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

WFS = "https://data.3dbag.nl/api/BAG3D/wfs"
# (xmin, ymin, xmax, ymax) in RD New (EPSG:28992); the first 4 tiles of each are used
AREAS = {
    "delft": (84000, 446000, 86000, 448000),  # tune: historic centre and post-war housing
    "leidscherijn": (129000, 454000, 131000, 456000),  # test: new-build suburb (Utrecht)
    "staphorst": (202000, 516000, 204000, 518000),  # test: village, farmhouses, complex roofs
}
TILES_PER_AREA = 4
UA = {"User-Agent": "ManySeg-E031/0.1 (research)"}


def tiles(bbox) -> list[dict]:
    q = urllib.parse.urlencode({"service": "WFS", "version": "2.0.0", "request": "GetFeature",
                                "typeNames": "BAG3D:tiles", "outputFormat": "application/json",
                                "bbox": ",".join(map(str, bbox)) + ",EPSG:28992"})
    with urllib.request.urlopen(urllib.request.Request(f"{WFS}?{q}", headers=UA), timeout=120) as r:
        feats = json.load(r)["features"]
    return sorted((f["properties"] for f in feats), key=lambda p: p["tile_id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    manifest = {}
    for area, bbox in AREAS.items():
        d = args.out / area
        d.mkdir(parents=True, exist_ok=True)
        chosen = tiles(bbox)[:TILES_PER_AREA]
        manifest[area] = []
        for p in chosen:
            name = p["tile_id"].replace("/", "-") + ".city.json"
            path = d / name
            if not path.exists():
                for attempt in range(6):  # the object store occasionally times out the handshake
                    try:
                        req = urllib.request.Request(p["cj_download"], headers=UA)
                        with urllib.request.urlopen(req, timeout=600) as r:
                            gz = r.read()
                        if hashlib.sha256(gz).hexdigest() == p["cj_sha256"]:
                            break
                        print(p["tile_id"], "SHA-256 mismatch, retrying", flush=True)
                    except OSError as e:
                        print(p["tile_id"], "retry after", e, flush=True)
                    time.sleep(10 * (attempt + 1))
                else:
                    raise RuntimeError(f"{p['tile_id']}: download failed")
                path.write_bytes(gzip.decompress(gz))
                time.sleep(1)
            manifest[area].append({"tile_id": p["tile_id"], "file": name,
                                   "buildings": p["cj_nr_building"], "sha256_gz": p["cj_sha256"]})
            print(area, p["tile_id"], p["cj_nr_building"], "buildings", flush=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
