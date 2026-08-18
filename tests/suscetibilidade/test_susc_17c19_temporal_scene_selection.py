"""Testes do SUSC-17C19 - consolidacao temporal e selecao canonica de cenas Sentinel-2."""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"
DAT = ROOT / "datasets" / "suscetibilidade"

EXPECTED = [
    OUT / "SUSC_17C19_CONSOLIDACAO_TEMPORAL_E_SELECAO_CENAS_SENTINEL2_REPORT.md",
    OUT / "susc_17c19_event_temporal_lineage.csv",
    OUT / "susc_17c19_candidate_patch_temporal_binding.csv",
    OUT / "susc_17c19_sentinel2_scene_selection_criteria.csv",
    OUT / "susc_17c19_sentinel2_canonical_pre_event_scenes.csv",
    OUT / "susc_17c19_sentinel2_rejected_scene_candidates.csv",
    OUT / "susc_17c19_sentinel2_tile_request_manifest.csv",
    OUT / "susc_17c19_chirps_runtime_plan.csv",
    OUT / "susc_17c19_temporal_lineage_gap_audit.csv",
    OUT / "susc_17c19_no_leakage_audit.csv",
    OUT / "susc_17c19_gate_evaluation_matrix.csv",
    OUT / "susc_17c19_readiness_summary.json",
    OUT / "susc_17c19_promotion_blockers.csv",
    SCHEMAS / "susc_17c19_temporal_lineage_schema_v1.json",
    SCHEMAS / "susc_17c19_scene_selection_schema_v1.json",
    SCHEMAS / "susc_17c19_tile_request_schema_v1.json",
]


def _load_common():
    path = SCRIPTS / "susc_17c19_temporal_scene_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s17c19_temporal_scene_common", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_script(name: str, *args: str):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
    )


def test_build_gera_todos_artefatos_e_valida_byte_identico():
    result = run_script("build_susc_17c19_temporal_scene_selection.py")
    assert result.returncode == 0, result.stderr + result.stdout
    first = {path: path.read_bytes() for path in EXPECTED}
    result = run_script("build_susc_17c19_temporal_scene_selection.py")
    assert result.returncode == 0, result.stderr + result.stdout
    assert {path: path.read_bytes() for path in EXPECTED} == first
    result = run_script("validate_susc_17c19_temporal_scene_selection.py")
    assert result.returncode == 0, result.stderr + result.stdout


def test_schemas_validos_e_ids_deterministicos():
    s17c19 = _load_common()
    lineage_schema = json.loads((SCHEMAS / "susc_17c19_temporal_lineage_schema_v1.json").read_text(encoding="utf-8"))
    scene_schema = json.loads((SCHEMAS / "susc_17c19_scene_selection_schema_v1.json").read_text(encoding="utf-8"))
    tile_schema = json.loads((SCHEMAS / "susc_17c19_tile_request_schema_v1.json").read_text(encoding="utf-8"))
    for row in read_csv(OUT / "susc_17c19_event_temporal_lineage.csv"):
        assert s17c19._schema_violations(row, lineage_schema) == []
    for row in read_csv(OUT / "susc_17c19_sentinel2_canonical_pre_event_scenes.csv"):
        assert s17c19._schema_violations(row, scene_schema) == []
    for row in read_csv(OUT / "susc_17c19_sentinel2_tile_request_manifest.csv"):
        assert s17c19._schema_violations(row, tile_schema) == []
    assert [row["candidate_patch_temporal_binding_id"] for row in read_csv(OUT / "susc_17c19_candidate_patch_temporal_binding.csv")] == [
        f"S17C19_TBIND_{idx:04d}" for idx in range(1, 6)
    ]


def test_data_nao_parseada_do_event_id_e_fonte_upstream_usada():
    lineage = read_csv(OUT / "susc_17c19_event_temporal_lineage.csv")
    assert {row["parse_from_event_id"] for row in lineage} == {"false"}
    assert {row["upstream_source_file"] for row in lineage} == {
        "outputs_public/suscetibilidade/susc_17c4_extracted_reference_candidates.csv"
    }
    assert {row["event_period_start"] for row in lineage} == {"2022-05-24"}
    assert {row["event_period_end"] for row in lineage} == {"2022-05-30"}


def test_bindings_temporais_existem_para_5_patches():
    bindings = read_csv(OUT / "susc_17c19_candidate_patch_temporal_binding.csv")
    assert len(bindings) == 5
    for row in bindings:
        assert row["pre_event_window_start"] == "2022-04-24"
        assert row["pre_event_window_end"] == "2022-05-23"
        assert row["during_event_window_start"] == "2022-05-24"
        assert row["during_event_window_end"] == "2022-05-30"
        assert row["post_event_window_start"] == "2022-05-31"
        assert row["usable_for_pre_event_sensor_query"] == "true"


def test_cenas_canonicas_sao_pre_evento_e_uma_por_patch():
    canonical = read_csv(OUT / "susc_17c19_sentinel2_canonical_pre_event_scenes.csv")
    scenes = read_csv(OUT / "susc_17c18_sentinel2_candidate_scene_registry.csv")
    pre_event_patches = {s["candidate_patch_id"] for s in scenes if s["is_pre_event"] == "true" and s["is_post_event"] == "false"}
    canonical_patches = {row["candidate_patch_id"] for row in canonical}
    assert pre_event_patches == canonical_patches
    assert len(canonical) == len(canonical_patches)  # uma canonica por patch
    for row in canonical:
        assert row["selection_rank"] == "1"
        assert row["usable_for_pre_event_feature"] == "true"
        match = next(s for s in scenes if s["candidate_patch_id"] == row["candidate_patch_id"] and s["scene_or_item_id"] == row["scene_or_item_id"])
        assert match["is_pre_event"] == "true"
        assert match["is_post_event"] == "false"


def test_durante_e_pos_evento_bloqueadas_para_pre_evento():
    rejected = read_csv(OUT / "susc_17c19_sentinel2_rejected_scene_candidates.csv")
    during = [r for r in rejected if r["scene_time_class"] == "during_event_blocked"]
    post = [r for r in rejected if r["scene_time_class"] == "post_event_blocked"]
    assert len(during) > 0 and len(post) > 0
    assert {r["usable_for_pre_event_feature"] for r in rejected} == {"false"}
    leakage = read_csv(OUT / "susc_17c19_no_leakage_audit.csv")
    assert {r["uses_during_event_as_pre_event"] for r in leakage} == {"false"}
    assert {r["uses_post_event_as_pre_event"] for r in leakage} == {"false"}
    assert {r["passes_no_leakage"] for r in leakage} == {"true"}


def test_tile_e_embedding_e_chirps_nao_executados():
    tile = read_csv(OUT / "susc_17c19_sentinel2_tile_request_manifest.csv")
    chirps = read_csv(OUT / "susc_17c19_chirps_runtime_plan.csv")
    canonical = read_csv(OUT / "susc_17c19_sentinel2_canonical_pre_event_scenes.csv")
    assert {row["can_execute_now"] for row in tile} == {"false"}
    assert {row["can_execute_now"] for row in chirps} == {"false"}
    assert {row["tile_created"] for row in canonical} == {"false"}
    assert {row["downloaded"] for row in canonical} == {"false"}
    summary = json.loads((OUT / "susc_17c19_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["sentinel2_products_downloaded_count"] == 0
    assert summary["sentinel2_tiles_created_count"] == 0
    assert summary["embedding_tiles_created_count"] == 0
    assert summary["chirps_features_extracted_count"] == 0


def test_nada_vira_ground_reference_e_gaps_detectados():
    gates = read_csv(OUT / "susc_17c19_gate_evaluation_matrix.csv")
    assert {row["G4_vinculo_espacial_evento"] for row in gates} == {"false"}
    assert {row["G5_separacao_fenomeno"] for row in gates} == {"false"}
    assert {row["all_gates_passed_for_ground_reference"] for row in gates} == {"false"}
    gap = read_csv(OUT / "susc_17c19_temporal_lineage_gap_audit.csv")
    backfill = [r for r in gap if r["requires_temporal_backfill"] == "true"]
    assert len(backfill) >= 4
    ok = [r for r in gap if r["artifact_name"] == "susc_17c18_sensor_aoi_execution_plan.csv"]
    assert ok and ok[0]["requires_temporal_backfill"] == "false"


def test_score_guardrails_e_17b_inelegivel():
    summary = json.loads((OUT / "susc_17c19_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["score_v6_changed"] is False
    assert summary["score_v7_created"] is False
    assert summary["eligible_for_17b_now"] is False
    assert summary["eligible_for_score_v7"] is False
    assert summary["review_only"] is True
    assert summary["trainable"] is False
    assert summary["ground_truth"] is False
    assert summary["features_extracted_this_sprint"] == 0
    assert summary["ground_reference_candidates_created_count"] == 0
    assert not (DAT / "susc_score_v7_candidate_by_patch_v1.csv").exists()
    for target in [
        DAT / "susc_score_v6_candidate_by_patch_v1.csv",
        DAT / "susc_patches_official_v1.csv",
        DAT / "susc_patch_links_official_v1.csv",
    ]:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--", str(target.relative_to(ROOT))],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.stdout.strip() == ""
    blockers = read_csv(OUT / "susc_17c19_promotion_blockers.csv")
    assert any(row["blocker_type"] == "17b_blocked_until_event_reference_and_policy" for row in blockers)
    assert {row["blocks_17b"] for row in blockers} == {"true"}
