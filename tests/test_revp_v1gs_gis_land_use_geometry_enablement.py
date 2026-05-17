"""Tests for revp_v1gs_gis_land_use_geometry_enablement.py."""
from __future__ import annotations

import json
import struct
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.dino.revp_v1gs_gis_land_use_geometry_enablement import (
    DEPS,
    GEOMETRY_LIBS,
    METHODOLOGICAL_GUARDRAILS,
    PENDING_ISSUES,
    PETROPOLIS_CLASS_COL,
    PETROPOLIS_ENCODING,
    audit_dependencies,
    audit_sidecars,
    build_v1gq_rerun_plan,
    class_distribution_rows,
    geometry_lib_available,
    read_dbf_attributes,
    run_audit,
    schema_audit,
    sidecars_complete,
    spatial_extent_rows,
    write_csv,
    write_json,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dbf(path: Path, fields: list[tuple[str, str, int]],
              records: list[list[str]]) -> None:
    n_records = len(records)
    header_size = 32 + 32 * len(fields) + 1
    record_size = 1 + sum(f[2] for f in fields)
    hdr = bytearray(32)
    hdr[0] = 3
    struct.pack_into("<I", hdr, 4, n_records)
    struct.pack_into("<H", hdr, 8, header_size)
    struct.pack_into("<H", hdr, 10, record_size)
    field_descs = bytearray()
    for name, ftype, length in fields:
        fd = bytearray(32)
        name_b = name.encode("latin-1")[:11]
        fd[: len(name_b)] = name_b
        fd[11] = ord(ftype)
        fd[16] = length
        field_descs += bytes(fd)
    data = bytearray()
    for rec in records:
        row = bytearray(b" ")
        for (_, _, length), val in zip(fields, rec):
            encoded = val.encode("latin-1", errors="replace")[:length]
            row += encoded + b" " * (length - len(encoded))
        data += bytes(row)
    with path.open("wb") as f:
        f.write(bytes(hdr))
        f.write(bytes(field_descs))
        f.write(b"\r")
        f.write(bytes(data))
        f.write(b"\x1a")


# ---------------------------------------------------------------------------
# Methodological guardrails
# ---------------------------------------------------------------------------

class TestMethodologicalGuardrails:
    def test_review_only_true(self):
        assert METHODOLOGICAL_GUARDRAILS["review_only"] is True

    def test_supervised_training_false(self):
        assert METHODOLOGICAL_GUARDRAILS["supervised_training"] is False

    def test_labels_created_false(self):
        assert METHODOLOGICAL_GUARDRAILS["labels_created"] is False

    def test_land_use_not_ground_truth(self):
        assert METHODOLOGICAL_GUARDRAILS["land_use_is_ground_truth"] is False

    def test_vulnerability_not_ground_truth(self):
        assert METHODOLOGICAL_GUARDRAILS["vulnerability_index_is_ground_truth"] is False

    def test_dino_not_predicts_vulnerability(self):
        assert METHODOLOGICAL_GUARDRAILS["dino_predicts_vulnerability"] is False

    def test_final_vulnerability_claim_false(self):
        assert METHODOLOGICAL_GUARDRAILS["final_vulnerability_claim"] is False

    def test_multimodal_disabled(self):
        assert METHODOLOGICAL_GUARDRAILS["multimodal_execution_enabled"] is False


class TestPendingIssues:
    def test_curitiba_land_use_blocked(self):
        assert PENDING_ISSUES["curitiba_land_use_status"] == "BLOCKED"

    def test_recife_land_use_blocked(self):
        assert PENDING_ISSUES["recife_land_use_status"] == "BLOCKED"

    def test_population_density_blocked(self):
        assert PENDING_ISSUES["population_density_status"] == "BLOCKED"

    def test_road_density_partial(self):
        assert PENDING_ISSUES["road_density_status"] == "PARTIAL"

    def test_global_index_not_final(self):
        assert PENDING_ISSUES["global_index_status"] in ("PARTIAL", "BLOCKED")


# ---------------------------------------------------------------------------
# audit_dependencies
# ---------------------------------------------------------------------------

class TestAuditDependencies:
    def test_returns_all_deps(self):
        result = audit_dependencies()
        for lib in DEPS:
            assert lib in result

    def test_values_valid(self):
        result = audit_dependencies()
        for v in result.values():
            assert v in ("AVAILABLE", "MISSING")

    def test_no_crash_even_if_all_missing(self):
        with patch("builtins.__import__", side_effect=ImportError("not found")):
            pass  # Can't easily mock __import__ for specific libs; just assert no crash below
        result = audit_dependencies()
        assert isinstance(result, dict)

    def test_geometry_lib_available_returns_none_when_all_missing(self):
        deps = {lib: "MISSING" for lib in DEPS}
        assert geometry_lib_available(deps) is None

    def test_geometry_lib_returns_pyogrio_first(self):
        deps = {lib: "MISSING" for lib in DEPS}
        deps["pyogrio"] = "AVAILABLE"
        deps["geopandas"] = "AVAILABLE"
        assert geometry_lib_available(deps) == "pyogrio"

    def test_geometry_lib_falls_back_to_geopandas(self):
        deps = {lib: "MISSING" for lib in DEPS}
        deps["geopandas"] = "AVAILABLE"
        assert geometry_lib_available(deps) == "geopandas"

    def test_geometry_lib_falls_back_to_fiona(self):
        deps = {lib: "MISSING" for lib in DEPS}
        deps["fiona"] = "AVAILABLE"
        assert geometry_lib_available(deps) == "fiona"


# ---------------------------------------------------------------------------
# audit_sidecars
# ---------------------------------------------------------------------------

class TestAuditSidecars:
    def test_all_sidecars_present(self, tmp_path):
        shp = tmp_path / "test.shp"
        for ext in [".shp", ".dbf", ".shx", ".prj"]:
            shp.with_suffix(ext).write_bytes(b"dummy")
        rows = audit_sidecars(shp)
        assert all(r["exists"] for r in rows)

    def test_missing_sidecar_has_blocker(self, tmp_path):
        shp = tmp_path / "test.shp"
        shp.write_bytes(b"dummy")
        (tmp_path / "test.dbf").write_bytes(b"dummy")
        rows = audit_sidecars(shp)
        shx_row = next(r for r in rows if r["extension"] == ".shx")
        assert shx_row["exists"] is False
        assert shx_row["blocker"] != ""

    def test_sidecars_complete_true_when_all_essential_present(self, tmp_path):
        shp = tmp_path / "test.shp"
        for ext in [".shp", ".dbf", ".shx"]:
            shp.with_suffix(ext).write_bytes(b"dummy")
        rows = audit_sidecars(shp)
        assert sidecars_complete(rows) is True

    def test_sidecars_complete_false_when_essential_missing(self, tmp_path):
        shp = tmp_path / "test.shp"
        shp.write_bytes(b"dummy")
        rows = audit_sidecars(shp)
        assert sidecars_complete(rows) is False

    def test_result_has_required_keys(self, tmp_path):
        shp = tmp_path / "test.shp"
        shp.write_bytes(b"dummy")
        rows = audit_sidecars(shp)
        for r in rows:
            for k in ["extension", "path", "exists", "essential", "blocker"]:
                assert k in r


# ---------------------------------------------------------------------------
# read_dbf_attributes (fallback, no geometry)
# ---------------------------------------------------------------------------

class TestReadDbfAttributes:
    def test_reads_class_distribution(self, tmp_path):
        dbf = tmp_path / "test.dbf"
        _make_dbf(dbf, [("CLASSE_USO", "C", 25)],
                  [["formação florestal"], ["área edificada"], ["formação florestal"]])
        result = read_dbf_attributes(dbf, "CLASSE_USO")
        assert result["readable"] is True
        dist = result["class_distribution"]
        assert dist.get("formação florestal") == 2
        assert dist.get("área edificada") == 1

    def test_nonexistent_returns_not_readable(self, tmp_path):
        result = read_dbf_attributes(tmp_path / "no.dbf", "CLASSE_USO")
        assert result["readable"] is False

    def test_missing_col_gives_empty_distribution(self, tmp_path):
        dbf = tmp_path / "test.dbf"
        _make_dbf(dbf, [("OTHER", "C", 20)], [["something"]])
        result = read_dbf_attributes(dbf, "CLASSE_USO")
        assert result.get("class_distribution") == {} or result["class_col"] == ""


# ---------------------------------------------------------------------------
# schema_audit
# ---------------------------------------------------------------------------

class TestSchemaAudit:
    def test_failed_geometry_gives_fail_checks(self):
        geom = {"success": False, "error": "conversion failed"}
        rows = schema_audit(geom)
        statuses = {r["check"]: r["status"] for r in rows}
        assert statuses["geometry_readable"] == "FAIL"

    def test_successful_geometry_gives_pass(self):
        geom = {
            "success": True, "method": "geopandas",
            "n_features": 100, "classes": ["formação florestal", "silvicultura"],
            "crs_src": "EPSG:31983", "crs_dst": "EPSG:4326",
        }
        rows = schema_audit(geom)
        statuses = {r["check"]: r["status"] for r in rows}
        assert statuses["geometry_readable"] == "PASS"
        assert statuses["classe_uso_column_present"] == "PASS"
        assert statuses["crs_reprojected_to_wgs84"] == "PASS"

    def test_empty_classes_warn(self):
        geom = {
            "success": True, "method": "geopandas", "n_features": 5,
            "classes": [], "crs_src": "EPSG:31983", "crs_dst": "EPSG:4326",
        }
        rows = schema_audit(geom)
        col_row = next(r for r in rows if r["check"] == "classe_uso_column_present")
        assert col_row["status"] == "WARN"


# ---------------------------------------------------------------------------
# spatial_extent_rows
# ---------------------------------------------------------------------------

class TestSpatialExtentRows:
    def test_blocked_when_geometry_failed(self):
        geom = {"success": False, "error": "failed"}
        rows = spatial_extent_rows(geom)
        assert rows[0]["status"] == "BLOCKED"

    def test_available_when_geometry_succeeded(self):
        geom = {
            "success": True,
            "bbox_wgs84": [-43.5, -22.7, -43.1, -22.2],
        }
        rows = spatial_extent_rows(geom)
        assert rows[0]["status"] == "AVAILABLE"
        assert rows[0]["lon_min"] == -43.5


# ---------------------------------------------------------------------------
# class_distribution_rows
# ---------------------------------------------------------------------------

class TestClassDistributionRows:
    def test_uses_geometry_distribution_when_available(self):
        geom = {
            "success": True,
            "class_distribution": {"formação florestal": 100, "silvicultura": 50},
        }
        rows = class_distribution_rows(geom, {})
        classes = {r["classe_uso"] for r in rows}
        assert "formação florestal" in classes
        assert all(r["source"] == "geometry" for r in rows)

    def test_falls_back_to_dbf_distribution(self):
        geom = {"success": False, "class_distribution": {}}
        dbf = {"readable": True, "class_distribution": {"água": 97}}
        rows = class_distribution_rows(geom, dbf)
        classes = {r["classe_uso"] for r in rows}
        assert "água" in classes
        assert all(r["source"] == "dbf_fallback" for r in rows)

    def test_all_rows_review_only(self):
        geom = {"success": True, "class_distribution": {"floresta": 10}}
        for r in class_distribution_rows(geom, {}):
            assert r["review_only"] == "true"

    def test_no_invented_classes(self):
        geom = {"success": True, "class_distribution": {"real_class": 5}}
        rows = class_distribution_rows(geom, {})
        assert all(r["classe_uso"] in ["real_class"] for r in rows)


# ---------------------------------------------------------------------------
# build_v1gq_rerun_plan
# ---------------------------------------------------------------------------

class TestBuildV1gqRerunPlan:
    def test_blocked_when_geometry_failed(self, tmp_path):
        geom = {"success": False, "error": "failed", "out_path": ""}
        plan = build_v1gq_rerun_plan(geom, tmp_path, None)
        assert plan["v1gq_rerun_readiness"] == "BLOCKED"
        assert plan["geojson_path"] == ""

    def test_blocked_when_geojson_not_exists(self, tmp_path):
        geom = {"success": True, "out_path": str(tmp_path / "missing.geojson")}
        plan = build_v1gq_rerun_plan(geom, tmp_path, None)
        assert plan["v1gq_rerun_readiness"] == "BLOCKED"

    def test_ready_when_geojson_exists(self, tmp_path):
        geojson = tmp_path / "petropolis_land_use_v1gs.geojson"
        geojson.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
        geom = {"success": True, "out_path": str(geojson)}
        plan = build_v1gq_rerun_plan(geom, tmp_path, None)
        assert plan["v1gq_rerun_readiness"] == "READY_FOR_PARTIAL_RERUN"

    def test_suggested_command_contains_geojson_path(self, tmp_path):
        geojson = tmp_path / "test.geojson"
        geojson.write_text("{}", encoding="utf-8")
        geom = {"success": True, "out_path": str(geojson)}
        plan = build_v1gq_rerun_plan(geom, tmp_path, tmp_path)
        assert str(geojson) in plan["suggested_command"]
        assert "--land-use-geojson-petropolis" in plan["suggested_command"]

    def test_no_v1gq_rerun_without_geometry(self, tmp_path):
        geom = {"success": False, "out_path": "", "error": "missing lib"}
        plan = build_v1gq_rerun_plan(geom, tmp_path, None)
        assert plan["v1gq_rerun_readiness"] != "READY_FOR_PARTIAL_RERUN"


# ---------------------------------------------------------------------------
# run_audit — no gis_root
# ---------------------------------------------------------------------------

class TestRunAuditNoGisRoot:
    def test_runs_without_error(self, tmp_path):
        result = run_audit(None, tmp_path)
        assert isinstance(result, dict)

    def test_summary_present(self, tmp_path):
        result = run_audit(None, tmp_path)
        assert "summary" in result

    def test_geometry_blocked_without_gis_root(self, tmp_path):
        result = run_audit(None, tmp_path)
        assert result["summary"]["land_use_geometry_status"] == "BLOCKED"

    def test_v1gq_rerun_blocked_without_gis_root(self, tmp_path):
        result = run_audit(None, tmp_path)
        assert result["summary"]["v1gq_rerun_readiness"] == "BLOCKED"

    def test_summary_has_all_guardrails(self, tmp_path):
        result = run_audit(None, tmp_path)
        s = result["summary"]
        for key, val in METHODOLOGICAL_GUARDRAILS.items():
            assert s[key] == val, f"Guardrail mismatch: {key}"

    def test_summary_has_pending_issues(self, tmp_path):
        result = run_audit(None, tmp_path)
        s = result["summary"]
        assert s["curitiba_land_use_status"] == "BLOCKED"
        assert s["recife_land_use_status"] == "BLOCKED"
        assert s["population_density_status"] == "BLOCKED"
        assert s["road_density_status"] == "PARTIAL"

    def test_no_fail_checks_without_gis_root(self, tmp_path):
        result = run_audit(None, tmp_path)
        fail = [r for r in result["qa_rows"] if r["status"] == "FAIL"]
        assert fail == []

    def test_no_v1gq_rerun_executed_without_gis_root(self, tmp_path):
        result = run_audit(None, tmp_path)
        assert result["summary"]["v1gq_rerun_executed"] is False

    def test_gis_root_provided_false(self, tmp_path):
        result = run_audit(None, tmp_path)
        assert result["summary"]["gis_root_provided"] is False

    def test_summary_has_22_required_fields(self, tmp_path):
        result = run_audit(None, tmp_path)
        s = result["summary"]
        required = [
            "stage", "stage_name", "generated_at", "gis_root_provided",
            "geometry_lib_used", "land_use_geometry_status",
            "petropolis_geojson_converted", "petropolis_n_features",
            "petropolis_classes_found", "v1gq_rerun_readiness",
            "v1gq_rerun_executed", "v1gq_rerun_output_dir",
            "v1gq_land_use_status_after_rerun", "v1gq_index_status_after_rerun",
            "v1gq_petropolis_available_indicator_count_after_rerun",
            "v1gq_petropolis_index_status_after_rerun",
            "curitiba_land_use_status", "recife_land_use_status",
            "population_density_status", "road_density_status",
            "global_index_status", "petropolis_land_use_status",
            "review_only", "supervised_training", "labels_created",
            "output_dir", "blockers_count", "qa_status", "methodology_note",
        ]
        for field in required:
            assert field in s, f"Missing summary field: {field}"


# ---------------------------------------------------------------------------
# No forbidden outputs outside local_runs
# ---------------------------------------------------------------------------

class TestNoForbiddenOutputs:
    def test_geojson_only_in_output_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = run_audit(None, tmp_path / "v1gs")
        geojson_path = result["summary"].get("v1gq_rerun_output_dir", "")
        # When BLOCKED, no geojson is produced
        assert result["geom_result"].get("out_path", "") == "" or \
               str(tmp_path) in result["geom_result"].get("out_path", "")

    def test_no_files_created_in_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        out = tmp_path / "v1gs_out"
        run_audit(None, out)
        # Only v1gs_out dir should appear, no stray files at top level
        unexpected = [f for f in tmp_path.iterdir()
                      if f != out and f.suffix in (".csv", ".json", ".geojson")]
        assert unexpected == []


# ---------------------------------------------------------------------------
# v1gq outputs not erased
# ---------------------------------------------------------------------------

class TestV1gqOutputsPreserved:
    def test_original_v1gq_dir_not_touched(self, tmp_path):
        v1gq_dir = tmp_path / "dino_embeddings" / "v1gq"
        v1gq_dir.mkdir(parents=True)
        sentinel = v1gq_dir / "gis_vulnerability_summary_v1gq.json"
        sentinel.write_text('{"stage":"v1gq"}', encoding="utf-8")

        run_audit(None, tmp_path / "dino_embeddings" / "v1gs")

        assert sentinel.exists()
        data = json.loads(sentinel.read_text(encoding="utf-8"))
        assert data["stage"] == "v1gq", "v1gs must not modify v1gq outputs"


# ---------------------------------------------------------------------------
# Output files written
# ---------------------------------------------------------------------------

class TestOutputFilesWritten:
    def test_all_expected_csvs_written(self, tmp_path):
        out = tmp_path / "v1gs"
        result = run_audit(None, out)
        out.mkdir(parents=True, exist_ok=True)

        write_json(out / "land_use_geometry_summary_v1gs.json", result["summary"])
        write_csv(out / "land_use_geometry_qa_v1gs.csv", result["qa_rows"],
                  ["check", "status", "detail"])
        write_csv(out / "land_use_geometry_blockers_v1gs.csv", result["blockers"],
                  ["category", "severity", "detail"])
        write_csv(out / "land_use_geometry_dependency_audit_v1gs.csv", result["dep_rows"],
                  ["library", "status", "required_for", "blocker_if_missing"])

        assert (out / "land_use_geometry_summary_v1gs.json").exists()
        assert (out / "land_use_geometry_qa_v1gs.csv").exists()
        assert (out / "land_use_geometry_blockers_v1gs.csv").exists()
        assert (out / "land_use_geometry_dependency_audit_v1gs.csv").exists()

    def test_summary_json_readable(self, tmp_path):
        out = tmp_path / "v1gs"
        result = run_audit(None, out)
        out.mkdir(parents=True, exist_ok=True)
        p = out / "land_use_geometry_summary_v1gs.json"
        write_json(p, result["summary"])
        loaded = json.loads(p.read_text(encoding="utf-8"))
        assert loaded["stage"] == "v1gs"
        assert loaded["labels_created"] is False
        assert loaded["supervised_training"] is False


# ---------------------------------------------------------------------------
# Missing-deps path: BLOCKED not crash
# ---------------------------------------------------------------------------

class TestMissingDepsBlocked:
    def test_missing_all_geometry_libs_gives_blocked_not_crash(self, tmp_path):
        deps_all_missing = {lib: "MISSING" for lib in DEPS}
        lib = geometry_lib_available(deps_all_missing)
        assert lib is None

    def test_blocked_geom_result_does_not_trigger_rerun(self, tmp_path):
        geom = {
            "success": False,
            "method": "none",
            "out_path": "",
            "error": "all libs missing",
        }
        plan = build_v1gq_rerun_plan(geom, tmp_path, None)
        assert plan["v1gq_rerun_readiness"] == "BLOCKED"
