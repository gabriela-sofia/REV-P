"""Helpers for isolated v2ba acquisition tests."""

from __future__ import annotations

import csv

import scripts.v2ba_minimal_real_geometry_acquisition_workbench as engine


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def dirs(tmp_path):
    return {
        "dataset_dir": tmp_path / "datasets",
        "output_dir": tmp_path / "outputs",
        "config_dir": tmp_path / "configs",
        "external_dir": tmp_path / "external",
        "docs_dir": tmp_path / "docs",
    }


def run(tmp_path, patch=None, event=None, mode="validate"):
    paths = dirs(tmp_path)
    if patch:
        write_csv(paths["external_dir"] / "patch_boundary_REC_00019" / "patch.csv", [patch])
    if event:
        write_csv(paths["external_dir"] / "event_polygon_REC_2022_05_24_30" / "event.csv", [event])
    code, summary = engine.run(mode, **paths)
    return paths, code, summary


def patch_bbox(crs="EPSG:3857", package_id="PKG_34713b8aab96"):
    return {
        "target_type": "patch_boundary", "target_id": "REC_00019", "package_id": package_id,
        "source_type": "bbox", "geometry_value": "0,0,10,10", "crs": crs,
        "provenance_note": "synthetic test fixture", "source_public": "true",
        "access_status": "public_or_project_access",
    }


def event_wkt(crs="EPSG:3857", point=False, package_id="PKG_34713b8aab96"):
    return {
        "target_type": "event_polygon", "target_id": "REC_2022_05_24_30", "package_id": package_id,
        "source_type": "wkt",
        "geometry_value": "POINT(1 2)" if point else "POLYGON((0 0,10 0,10 10,0 10,0 0))",
        "crs": crs, "provenance_note": "synthetic test fixture", "source_public": "true",
        "access_status": "public_or_project_access",
    }


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
