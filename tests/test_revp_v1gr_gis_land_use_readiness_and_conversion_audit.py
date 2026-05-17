"""Tests for revp_v1gr_gis_land_use_readiness_and_conversion_audit.py."""
from __future__ import annotations

import json
import struct
import tempfile
from pathlib import Path

import pytest

from scripts.dino.revp_v1gr_gis_land_use_readiness_and_conversion_audit import (
    CONVERSION_DEPS,
    DEPS,
    GIS_LAND_USE_HINTS,
    LAND_USE_EXTENSIONS,
    LAND_USE_SEARCH_TERMS,
    METHODOLOGICAL_GUARDRAILS,
    REGIONS,
    assess_region,
    audit_dependencies,
    build_class_mapping_table,
    build_conversion_plan,
    CLASS_MAPPING_ROWS,
    compute_v1gq_readiness,
    conversion_possible,
    extract_unique_classes,
    file_matches_land_use,
    inventory_land_use_files,
    map_class_to_score,
    parse_dbf,
    run_audit,
    write_csv,
    write_json,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dbf(path: Path, fields: list[tuple[str, str, int]],
              records: list[list[str]]) -> None:
    """Write a minimal DBF file for testing."""
    n_records = len(records)
    n_fields = len(fields)
    header_size = 32 + 32 * n_fields + 1
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

    def test_targets_created_false(self):
        assert METHODOLOGICAL_GUARDRAILS["targets_created"] is False

    def test_predictive_claims_false(self):
        assert METHODOLOGICAL_GUARDRAILS["predictive_claims"] is False

    def test_multimodal_disabled(self):
        assert METHODOLOGICAL_GUARDRAILS["multimodal_execution_enabled"] is False

    def test_land_use_is_not_ground_truth(self):
        assert METHODOLOGICAL_GUARDRAILS["land_use_is_ground_truth"] is False

    def test_all_guardrail_keys_present(self):
        required = {
            "review_only", "supervised_training", "labels_created",
            "targets_created", "predictive_claims",
            "multimodal_execution_enabled", "land_use_is_ground_truth",
        }
        assert required.issubset(METHODOLOGICAL_GUARDRAILS.keys())


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_regions_list(self):
        assert set(REGIONS) == {"Curitiba", "Petropolis", "Recife"}

    def test_land_use_extensions_present(self):
        for ext in [".shp", ".geojson", ".gpkg", ".csv", ".tif", ".tiff"]:
            assert ext in LAND_USE_EXTENSIONS

    def test_land_use_search_terms_present(self):
        for term in ["uso", "land_use", "cobertura", "lulc"]:
            assert term in LAND_USE_SEARCH_TERMS

    def test_conversion_deps_subset_of_deps(self):
        assert CONVERSION_DEPS.issubset(set(DEPS))

    def test_gis_hints_petropolis_has_path(self):
        assert len(GIS_LAND_USE_HINTS.get("petropolis", [])) > 0

    def test_gis_hints_curitiba_empty(self):
        assert GIS_LAND_USE_HINTS.get("curitiba", []) == []

    def test_gis_hints_recife_empty(self):
        assert GIS_LAND_USE_HINTS.get("recife", []) == []


# ---------------------------------------------------------------------------
# file_matches_land_use
# ---------------------------------------------------------------------------

class TestFileMatchesLandUse:
    def test_uso_shp_matches(self):
        assert file_matches_land_use("uso_solo.shp") is True

    def test_land_use_geojson_matches(self):
        assert file_matches_land_use("land_use_recife.geojson") is True

    def test_cobertura_tif_matches(self):
        assert file_matches_land_use("cobertura_terra.tif") is True

    def test_lulc_gpkg_matches(self):
        assert file_matches_land_use("lulc_curitiba.gpkg") is True

    def test_urban_csv_matches(self):
        assert file_matches_land_use("urban_classification.csv") is True

    def test_random_shp_no_match(self):
        assert file_matches_land_use("rivers.shp") is False

    def test_uso_txt_no_match(self):
        assert file_matches_land_use("uso_solo.txt") is False

    def test_npz_no_match(self):
        assert file_matches_land_use("embedding.npz") is False

    def test_tiff_cobertura_matches(self):
        assert file_matches_land_use("cobertura.tiff") is True

    def test_classification_csv_matches(self):
        assert file_matches_land_use("classification.csv") is True


# ---------------------------------------------------------------------------
# audit_dependencies
# ---------------------------------------------------------------------------

class TestAuditDependencies:
    def test_returns_dict(self):
        result = audit_dependencies()
        assert isinstance(result, dict)

    def test_all_deps_covered(self):
        result = audit_dependencies()
        for lib in DEPS:
            assert lib in result

    def test_values_are_available_or_missing(self):
        result = audit_dependencies()
        for v in result.values():
            assert v in ("AVAILABLE", "MISSING")

    def test_rasterio_available(self):
        result = audit_dependencies()
        assert result.get("rasterio") == "AVAILABLE"

    def test_pandas_available(self):
        result = audit_dependencies()
        assert result.get("pandas") == "AVAILABLE"


# ---------------------------------------------------------------------------
# parse_dbf
# ---------------------------------------------------------------------------

class TestParseDbf:
    def test_readable_on_synthetic_dbf(self, tmp_path):
        dbf = tmp_path / "test.dbf"
        _make_dbf(dbf, [("CLASSE_USO", "C", 30)], [["formação florestal"], ["água"]])
        result = parse_dbf(dbf)
        assert result["readable"] is True

    def test_fields_extracted(self, tmp_path):
        dbf = tmp_path / "test.dbf"
        _make_dbf(dbf, [("CLASSE_USO", "C", 30), ("AREA", "N", 10)], [["floresta", "100"]])
        result = parse_dbf(dbf)
        assert "CLASSE_USO" in result["fields"]

    def test_sample_records_read(self, tmp_path):
        dbf = tmp_path / "test.dbf"
        _make_dbf(dbf, [("CLASSE_USO", "C", 20)], [["área edificada"], ["silvicultura"]])
        result = parse_dbf(dbf)
        assert result["n_records"] == 2
        assert len(result["sample"]) == 2

    def test_nonexistent_file_not_readable(self, tmp_path):
        result = parse_dbf(tmp_path / "no.dbf")
        assert result["readable"] is False
        assert "error" in result

    def test_empty_file_not_readable(self, tmp_path):
        dbf = tmp_path / "empty.dbf"
        dbf.write_bytes(b"")
        result = parse_dbf(dbf)
        assert result["readable"] is False


# ---------------------------------------------------------------------------
# extract_unique_classes
# ---------------------------------------------------------------------------

class TestExtractUniqueClasses:
    def test_extracts_unique_values(self, tmp_path):
        dbf = tmp_path / "test.dbf"
        _make_dbf(dbf, [("CLASSE_USO", "C", 30)],
                  [["formação florestal"], ["água"], ["formação florestal"], ["silvicultura"]])
        classes = extract_unique_classes(dbf, "CLASSE_USO")
        assert set(classes) == {"formação florestal", "água", "silvicultura"}

    def test_returns_sorted(self, tmp_path):
        dbf = tmp_path / "test.dbf"
        _make_dbf(dbf, [("CLASSE_USO", "C", 30)],
                  [["zz"], ["aa"], ["mm"]])
        classes = extract_unique_classes(dbf, "CLASSE_USO")
        assert classes == sorted(classes)

    def test_missing_col_returns_empty(self, tmp_path):
        dbf = tmp_path / "test.dbf"
        _make_dbf(dbf, [("OTHER", "C", 20)], [["something"]])
        classes = extract_unique_classes(dbf, "CLASSE_USO")
        assert classes == []

    def test_nonexistent_returns_empty(self, tmp_path):
        classes = extract_unique_classes(tmp_path / "no.dbf", "CLASSE_USO")
        assert classes == []


# ---------------------------------------------------------------------------
# map_class_to_score
# ---------------------------------------------------------------------------

class TestMapClassToScore:
    def test_area_edificada_scores_3(self):
        assert map_class_to_score("área edificada") == 3

    def test_urban_keyword_scores_3(self):
        assert map_class_to_score("area urbana") == 3

    def test_built_keyword_scores_3(self):
        assert map_class_to_score("built-up area") == 3

    def test_area_antropizada_scores_2(self):
        assert map_class_to_score("área antropizada") == 2

    def test_pastagem_scores_2(self):
        assert map_class_to_score("pastagem") == 2

    def test_silvicultura_scores_2(self):
        assert map_class_to_score("silvicultura") == 2

    def test_formacao_nao_florestal_scores_2(self):
        assert map_class_to_score("formação não florestal") == 2

    def test_formacao_florestal_scores_1(self):
        assert map_class_to_score("formação florestal") == 1

    def test_agua_scores_1(self):
        assert map_class_to_score("água") == 1

    def test_water_scores_1(self):
        assert map_class_to_score("water body") == 1

    def test_vegetacao_scores_1(self):
        assert map_class_to_score("vegetação natural") == 1

    def test_unknown_class_review(self):
        assert map_class_to_score("totally_unknown_xyzabc") == "REVIEW"

    def test_desconhecido_review(self):
        assert map_class_to_score("desconhecido") == "REVIEW"

    def test_nulo_review(self):
        assert map_class_to_score("nulo") == "REVIEW"

    def test_no_invented_numeric_scores(self):
        unknown = "xyzqrstuvw_not_a_real_class_00099"
        result = map_class_to_score(unknown)
        assert result not in (1, 2, 3), "Unknown class must not receive an invented numeric score"

    def test_case_insensitive(self):
        assert map_class_to_score("SILVICULTURA") == 2

    def test_stripped_whitespace(self):
        assert map_class_to_score("  água  ") == 1


# ---------------------------------------------------------------------------
# build_class_mapping_table
# ---------------------------------------------------------------------------

class TestBuildClassMappingTable:
    def test_returns_list(self):
        table = build_class_mapping_table()
        assert isinstance(table, list)

    def test_has_19_entries(self):
        table = build_class_mapping_table()
        assert len(table) == 19

    def test_required_keys_present(self):
        table = build_class_mapping_table()
        required = {"class_pattern", "class_example", "score", "category", "review_only", "notes"}
        for row in table:
            assert required.issubset(row.keys()), f"Missing keys in: {row}"

    def test_all_review_only_true(self):
        for row in build_class_mapping_table():
            assert row["review_only"] == "true"

    def test_fbds_classes_covered(self):
        patterns = {r["class_pattern"] for r in build_class_mapping_table()}
        for cls in ["área edificada", "área antropizada", "formação florestal",
                    "formação não florestal", "silvicultura", "água"]:
            assert cls in patterns, f"FBDS class not in mapping: {cls}"

    def test_scores_are_1_2_3_or_review(self):
        for row in build_class_mapping_table():
            assert row["score"] in (1, 2, 3, "REVIEW"), f"Invalid score: {row['score']}"


# ---------------------------------------------------------------------------
# inventory_land_use_files
# ---------------------------------------------------------------------------

class TestInventoryLandUseFiles:
    def test_finds_matching_files(self, tmp_path):
        f = tmp_path / "uso_solo.shp"
        f.write_bytes(b"fake")
        result = inventory_land_use_files([tmp_path])
        assert any(r["filename"] == "uso_solo.shp" for r in result)

    def test_ignores_non_matching(self, tmp_path):
        (tmp_path / "rivers.shp").write_bytes(b"fake")
        result = inventory_land_use_files([tmp_path])
        assert result == []

    def test_finds_geojson(self, tmp_path):
        (tmp_path / "land_use_area.geojson").write_bytes(b"{}")
        result = inventory_land_use_files([tmp_path])
        assert len(result) == 1

    def test_nonexistent_root_skipped(self, tmp_path):
        fake = tmp_path / "nonexistent_dir"
        result = inventory_land_use_files([fake])
        assert result == []

    def test_no_duplicates_same_file(self, tmp_path):
        f = tmp_path / "uso.geojson"
        f.write_bytes(b"{}")
        result = inventory_land_use_files([tmp_path, tmp_path])
        names = [r["filename"] for r in result]
        assert names.count("uso.geojson") == 1

    def test_result_has_required_fields(self, tmp_path):
        (tmp_path / "cobertura.tif").write_bytes(b"fake")
        result = inventory_land_use_files([tmp_path])
        if result:
            for key in ["filename", "extension", "size_bytes", "full_path", "search_root"]:
                assert key in result[0]


# ---------------------------------------------------------------------------
# assess_region
# ---------------------------------------------------------------------------

class TestAssessRegion:
    def test_no_gis_root_returns_blocked(self):
        deps = audit_dependencies()
        result = assess_region("Curitiba", None, deps)
        assert result["coverage_status"] == "BLOCKED"
        assert "gis_root" in result["blocker_reason"]

    def test_curitiba_no_hints_blocked(self, tmp_path):
        deps = audit_dependencies()
        result = assess_region("Curitiba", tmp_path, deps)
        assert result["coverage_status"] == "BLOCKED"
        assert result["land_use_source_found"] is False

    def test_recife_no_hints_blocked(self, tmp_path):
        deps = audit_dependencies()
        result = assess_region("Recife", tmp_path, deps)
        assert result["coverage_status"] == "BLOCKED"

    def test_petropolis_hint_not_existing(self, tmp_path):
        deps = audit_dependencies()
        result = assess_region("Petropolis", tmp_path, deps)
        assert result["coverage_status"] == "BLOCKED"
        assert result["land_use_source_found"] is False

    def test_result_has_all_required_keys(self, tmp_path):
        deps = audit_dependencies()
        result = assess_region("Curitiba", tmp_path, deps)
        for key in ["region", "land_use_source_found", "source_path", "coverage_status",
                    "blocker_reason", "dbf_readable", "conversion_possible"]:
            assert key in result

    def test_no_gis_root_all_fields_present(self):
        deps = audit_dependencies()
        result = assess_region("Recife", None, deps)
        for key in ["region", "land_use_source_found", "coverage_status", "blocker_reason"]:
            assert key in result


# ---------------------------------------------------------------------------
# compute_v1gq_readiness
# ---------------------------------------------------------------------------

class TestComputeV1gqReadiness:
    def test_all_available(self):
        cov = [{"coverage_status": "AVAILABLE"} for _ in range(3)]
        assert compute_v1gq_readiness(cov) == "READY_FOR_V1GQ_RERUN"

    def test_one_partial(self):
        cov = [
            {"coverage_status": "PARTIAL"},
            {"coverage_status": "BLOCKED"},
            {"coverage_status": "BLOCKED"},
        ]
        assert compute_v1gq_readiness(cov) == "PARTIAL_READY"

    def test_all_blocked(self):
        cov = [{"coverage_status": "BLOCKED"} for _ in range(3)]
        assert compute_v1gq_readiness(cov) == "BLOCKED"

    def test_mixed_available_and_blocked(self):
        cov = [
            {"coverage_status": "AVAILABLE"},
            {"coverage_status": "BLOCKED"},
            {"coverage_status": "BLOCKED"},
        ]
        assert compute_v1gq_readiness(cov) == "PARTIAL_READY"


# ---------------------------------------------------------------------------
# build_conversion_plan
# ---------------------------------------------------------------------------

class TestBuildConversionPlan:
    def test_no_source_returns_blocked(self, tmp_path):
        deps = {"fiona": "MISSING", "geopandas": "MISSING", "pyogrio": "MISSING"}
        region_coverage = [{
            "region": "Curitiba",
            "land_use_source_found": False,
            "source_path": "",
            "blocker_reason": "no source",
            "conversion_possible": False,
        }]
        plan = build_conversion_plan(region_coverage, deps, tmp_path)
        assert plan[0]["plan_status"] == "BLOCKED"

    def test_result_has_required_keys(self, tmp_path):
        deps = {"fiona": "MISSING", "geopandas": "MISSING", "pyogrio": "MISSING"}
        region_coverage = [{
            "region": "Petropolis",
            "land_use_source_found": False,
            "source_path": "",
            "blocker_reason": "no source",
            "conversion_possible": False,
        }]
        plan = build_conversion_plan(region_coverage, deps, tmp_path)
        for key in ["region", "source_path", "output_path", "conversion_method",
                    "can_execute", "blocker", "plan_status"]:
            assert key in plan[0]

    def test_output_path_under_output_dir(self, tmp_path):
        deps = {"fiona": "AVAILABLE", "geopandas": "MISSING", "pyogrio": "MISSING"}
        region_coverage = [{
            "region": "Petropolis",
            "land_use_source_found": True,
            "source_path": str(tmp_path / "data" / "uso.shp"),
            "blocker_reason": "",
            "conversion_possible": True,
        }]
        plan = build_conversion_plan(region_coverage, deps, tmp_path / "out")
        if plan[0]["output_path"]:
            out = Path(plan[0]["output_path"])
            assert str(out).startswith(str(tmp_path / "out")), \
                "Conversion output must be under the output_dir"


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

    def test_all_regions_blocked_without_gis_root(self, tmp_path):
        result = run_audit(None, tmp_path)
        for reg in result["region_coverage"]:
            assert reg["coverage_status"] == "BLOCKED"

    def test_v1gq_readiness_blocked_without_gis_root(self, tmp_path):
        result = run_audit(None, tmp_path)
        assert result["summary"]["v1gq_rerun_readiness"] == "BLOCKED"

    def test_summary_has_required_fields(self, tmp_path):
        result = run_audit(None, tmp_path)
        s = result["summary"]
        required = [
            "stage", "stage_name", "generated_at", "gis_root_provided",
            "input_inventory_status", "dependency_status",
            "curitiba_land_use_status", "petropolis_land_use_status",
            "recife_land_use_status", "land_use_global_status",
            "v1gq_rerun_readiness", "blockers_count",
            "review_only", "supervised_training", "labels_created",
            "targets_created", "predictive_claims",
            "multimodal_execution_enabled", "land_use_is_ground_truth",
            "output_dir", "qa_status", "methodology_note",
        ]
        for field in required:
            assert field in s, f"Missing summary field: {field}"

    def test_guardrails_in_summary(self, tmp_path):
        result = run_audit(None, tmp_path)
        s = result["summary"]
        assert s["review_only"] is True
        assert s["supervised_training"] is False
        assert s["labels_created"] is False
        assert s["land_use_is_ground_truth"] is False
        assert s["multimodal_execution_enabled"] is False

    def test_class_mapping_in_result(self, tmp_path):
        result = run_audit(None, tmp_path)
        assert len(result["class_mapping"]) == 19

    def test_gis_root_provided_false(self, tmp_path):
        result = run_audit(None, tmp_path)
        assert result["summary"]["gis_root_provided"] is False

    def test_qa_rows_present(self, tmp_path):
        result = run_audit(None, tmp_path)
        assert len(result["qa_rows"]) > 0

    def test_no_fail_qa_checks(self, tmp_path):
        result = run_audit(None, tmp_path)
        fail_checks = [r for r in result["qa_rows"] if r["status"] == "FAIL"]
        assert fail_checks == [], f"Unexpected FAIL in QA: {fail_checks}"

    def test_conversion_results_all_skipped_or_blocked(self, tmp_path):
        result = run_audit(None, tmp_path)
        for cr in result["conversion_results"]:
            assert cr["status"] in ("SKIPPED", "BLOCKED"), \
                f"Expected SKIPPED/BLOCKED but got: {cr['status']}"

    def test_no_new_labels_in_summary(self, tmp_path):
        result = run_audit(None, tmp_path)
        note = result["summary"].get("methodology_note", "")
        assert "label" in note.lower() or "ground truth" in note.lower(), \
            "methodology_note should clarify no-label status"


# ---------------------------------------------------------------------------
# Output files created (integration-style)
# ---------------------------------------------------------------------------

class TestOutputFilesCreated:
    def test_main_writes_all_expected_csvs(self, tmp_path):
        result = run_audit(None, tmp_path / "v1gr")
        out = tmp_path / "v1gr"
        out.mkdir(parents=True, exist_ok=True)

        from scripts.dino.revp_v1gr_gis_land_use_readiness_and_conversion_audit import (
            write_csv, write_json,
        )
        write_json(out / "land_use_summary_v1gr.json", result["summary"])
        write_csv(out / "land_use_qa_v1gr.csv", result["qa_rows"],
                  ["check", "status", "detail"])
        write_csv(out / "land_use_class_mapping_candidate_v1gr.csv",
                  result["class_mapping"],
                  ["class_pattern", "class_example", "score", "category", "review_only", "notes"])
        write_csv(out / "land_use_region_coverage_v1gr.csv",
                  result["region_coverage"],
                  ["region", "coverage_status", "blocker_reason"])
        write_csv(out / "land_use_blockers_v1gr.csv",
                  result["blockers"],
                  ["category", "severity", "detail"])

        assert (out / "land_use_summary_v1gr.json").exists()
        assert (out / "land_use_qa_v1gr.csv").exists()
        assert (out / "land_use_class_mapping_candidate_v1gr.csv").exists()
        assert (out / "land_use_region_coverage_v1gr.csv").exists()
        assert (out / "land_use_blockers_v1gr.csv").exists()

    def test_summary_json_readable(self, tmp_path):
        result = run_audit(None, tmp_path / "v1gr")
        out = tmp_path / "v1gr"
        out.mkdir(parents=True, exist_ok=True)
        json_path = out / "land_use_summary_v1gr.json"
        json_path.write_text(
            __import__("json").dumps(result["summary"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        loaded = json.loads(json_path.read_text(encoding="utf-8"))
        assert loaded["stage"] == "v1gr"
        assert loaded["labels_created"] is False

    def test_class_mapping_csv_has_19_data_rows(self, tmp_path):
        result = run_audit(None, tmp_path)
        import csv as _csv
        out = tmp_path / "mapping.csv"
        write_csv(out, result["class_mapping"],
                  ["class_pattern", "class_example", "score", "category", "review_only", "notes"])
        with out.open("r", encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
        assert len(rows) == 19


# ---------------------------------------------------------------------------
# No forbidden outputs outside local_runs
# ---------------------------------------------------------------------------

class TestNoForbiddenOutputs:
    def test_conversion_outputs_under_output_dir(self, tmp_path):
        out_dir = tmp_path / "local_runs" / "v1gr"
        result = run_audit(None, out_dir)
        for cr in result["conversion_results"]:
            op = cr.get("output_path", "")
            if op:
                assert str(out_dir) in op, \
                    f"Conversion output outside output_dir: {op}"

    def test_run_audit_does_not_create_files_in_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        run_audit(None, tmp_path / "out")
        generated = list(tmp_path.glob("*.csv")) + list(tmp_path.glob("*.json"))
        assert generated == [], f"Unexpected files in cwd: {generated}"


# ---------------------------------------------------------------------------
# v1gq outputs untouched
# ---------------------------------------------------------------------------

class TestV1gqOutputsUntouched:
    def test_v1gq_dir_not_created_by_v1gr(self, tmp_path):
        v1gq_dir = tmp_path / "dino_embeddings" / "v1gq"
        v1gq_dir.mkdir(parents=True)
        sentinel = v1gq_dir / "sentinel_file.csv"
        sentinel.write_text("original content", encoding="utf-8")

        run_audit(None, tmp_path / "dino_embeddings" / "v1gr")

        assert sentinel.exists(), "v1gr must not delete v1gq outputs"
        assert sentinel.read_text(encoding="utf-8") == "original content", \
            "v1gr must not modify v1gq outputs"
