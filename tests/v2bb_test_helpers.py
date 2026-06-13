"""Isolated v2bb test helpers."""

import csv
import json

import scripts.v2bb_public_geometry_retrieval_feed_builder as engine


def dirs(tmp_path):
    return {"dataset_dir": tmp_path / "datasets", "output_dir": tmp_path / "outputs",
            "config_dir": tmp_path / "configs", "external_dir": tmp_path / "external",
            "docs_dir": tmp_path / "docs"}


def feature(target_type, target_id, geometry_type="Polygon", crs="EPSG:4326", point=False):
    coords = [-34.9, -8.1] if point else [[[-34.91, -8.11], [-34.90, -8.11], [-34.90, -8.10], [-34.91, -8.11]]]
    return {"type": "Feature", "properties": {"target_type": target_type, "target_id": target_id,
            "package_id": "PKG_34713b8aab96", "crs": crs},
            "geometry": {"type": "Point" if point else geometry_type, "coordinates": coords}}


def write_geojson(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
