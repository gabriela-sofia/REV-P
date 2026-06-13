"""v2aw - geometry source template generation tests."""

from __future__ import annotations

import csv


def _run(engine, ds, tmp_path):
    code, summary = engine.run(dataset_dir=str(ds), output_dir=str(tmp_path / "out"),
                               config_dir=str(tmp_path / "cfg"))
    assert code == 0
    return summary


def _read(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_templates_are_generated(v2aw_engine, v2aw_dataset, tmp_path):
    ds = v2aw_dataset(n_recife=3)
    _run(v2aw_engine, ds, tmp_path)
    assert (ds / "v2aw_patch_geometry_sources_template.csv").is_file()
    assert (ds / "v2aw_event_geometry_sources_template.csv").is_file()


def test_recife_template_has_55_priority_rows(v2aw_engine, v2aw_dataset, tmp_path):
    ds = v2aw_dataset(n_recife=55)
    summary = _run(v2aw_engine, ds, tmp_path)
    rows = _read(ds / "v2aw_patch_geometry_sources_template.csv")
    assert len(rows) == 55
    assert summary["total_priority_patches"] == 55
    assert all(row["region"] == "Recife" and row["priority_rank"] == "1" for row in rows)


def test_patch_template_never_invents_geometry(v2aw_engine, v2aw_dataset, tmp_path):
    ds = v2aw_dataset(n_recife=4)
    _run(v2aw_engine, ds, tmp_path)
    rows = _read(ds / "v2aw_patch_geometry_sources_template.csv")
    assert all(row["source_type"] == "missing" for row in rows)
    assert all(row["geometry_value"] == "" and row["geometry_path"] == "" for row in rows)
    assert all(row["crs"] == "" for row in rows)


def test_templates_are_deterministic(v2aw_engine, v2aw_dataset, tmp_path):
    ds = v2aw_dataset(n_recife=5)
    _run(v2aw_engine, ds, tmp_path)
    names = ("v2aw_patch_geometry_sources_template.csv", "v2aw_event_geometry_sources_template.csv",
             "v2aw_geometry_source_validation_registry.csv", "v2aw_recife_p1_geometry_readiness.csv")
    first = {name: (ds / name).read_bytes() for name in names}
    first_summary = (tmp_path / "out" / "execution_reports" /
                     "v2aw_geometry_source_intake_summary.json").read_bytes()
    _run(v2aw_engine, ds, tmp_path)
    assert {name: (ds / name).read_bytes() for name in names} == first
    assert (tmp_path / "out" / "execution_reports" /
            "v2aw_geometry_source_intake_summary.json").read_bytes() == first_summary
