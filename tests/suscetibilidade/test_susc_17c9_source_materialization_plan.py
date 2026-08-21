"""Testes do SUSC-17C9 - plano de materializacao de artefatos fonte."""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

EXPECTED = [
    OUT / "SUSC_WORKTREE_AUDIT_BEFORE_17C9.md",
    OUT / "SUSC_17C9_MATERIALIZACAO_ARTEFATOS_FONTE_PATCHES_CANDIDATOS_REPORT.md",
    OUT / "susc_17c9_source_artifact_inventory.csv",
    OUT / "susc_17c9_pipeline_reuse_audit.csv",
    OUT / "susc_17c9_candidate_patch_input_requirements.csv",
    OUT / "susc_17c9_external_artifact_request_plan.csv",
    OUT / "susc_17c9_lightweight_export_plan.csv",
    OUT / "susc_17c9_embedding_tile_requirement_plan.csv",
    OUT / "susc_17c9_feature_extraction_unblock_matrix.csv",
    OUT / "susc_17c9_no_fake_feature_policy.json",
    OUT / "susc_17c9_readiness_summary.json",
    OUT / "susc_17c9_promotion_blockers.csv",
    SCHEMAS / "susc_17c9_source_artifact_inventory_schema_v1.json",
    SCHEMAS / "susc_17c9_pipeline_reuse_schema_v1.json",
]


def _load_common():
    path = SCRIPTS / "susc_17c9_source_materialization_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s17c9_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_script(name: str):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
    )


def test_build_gera_todos_artefatos_e_valida():
    result = run_script("build_susc_17c9_source_materialization_plan.py")
    assert result.returncode == 0, result.stderr + result.stdout
    for path in EXPECTED:
        assert path.exists(), path
    result = run_script("validate_susc_17c9_source_materialization_plan.py")
    assert result.returncode == 0, result.stderr + result.stdout


def test_schemas_validos():
    s17c9 = _load_common()
    source_schema = json.loads((SCHEMAS / "susc_17c9_source_artifact_inventory_schema_v1.json").read_text(encoding="utf-8"))
    pipeline_schema = json.loads((SCHEMAS / "susc_17c9_pipeline_reuse_schema_v1.json").read_text(encoding="utf-8"))
    for row in read_csv(OUT / "susc_17c9_source_artifact_inventory.csv"):
        assert s17c9._schema_violations(row, source_schema) == []
    for row in read_csv(OUT / "susc_17c9_pipeline_reuse_audit.csv"):
        assert s17c9._schema_violations(row, pipeline_schema) == []


def test_build_duas_vezes_byte_identico_e_ids_deterministicos():
    first = tuple(path.read_bytes() for path in EXPECTED)
    result = run_script("build_susc_17c9_source_materialization_plan.py")
    assert result.returncode == 0, result.stderr + result.stdout
    second = tuple(path.read_bytes() for path in EXPECTED)
    assert first == second
    source_ids = [r["source_artifact_id"] for r in read_csv(OUT / "susc_17c9_source_artifact_inventory.csv")]
    pipe_ids = [r["pipeline_audit_id"] for r in read_csv(OUT / "susc_17c9_pipeline_reuse_audit.csv")]
    assert source_ids == sorted(source_ids)
    assert pipe_ids == sorted(pipe_ids)
    assert len(source_ids) == len(set(source_ids))
    assert len(pipe_ids) == len(set(pipe_ids))


def test_sem_score_v7_score_v6_ou_dataset_oficial_alterado():
    assert not (ROOT / "datasets" / "suscetibilidade" / "susc_score_v7_candidate_by_patch_v1.csv").exists()
    summary = json.loads((OUT / "susc_17c9_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["score_v6_changed"] is False
    assert summary["score_v7_created"] is False
    assert summary["official_patch_created"] is False
    assert summary["official_patch_link_created"] is False
    for path in (
        "datasets/suscetibilidade/susc_score_v6_candidate_by_patch_v1.csv",
        "datasets/suscetibilidade/susc_features_by_patch_v1.csv",
    ):
        changed = subprocess.run(["git", "diff", "--name-only", "--", path], cwd=ROOT, text=True, capture_output=True, check=False)
        assert changed.stdout.strip() == ""


def test_sem_trainable_ground_truth_feature_real_ou_raster_bruto_commitado():
    for path in (
        OUT / "susc_17c9_source_artifact_inventory.csv",
        OUT / "susc_17c9_pipeline_reuse_audit.csv",
        OUT / "susc_17c9_candidate_patch_input_requirements.csv",
        OUT / "susc_17c9_embedding_tile_requirement_plan.csv",
        OUT / "susc_17c9_promotion_blockers.csv",
    ):
        for row in read_csv(path):
            assert row["review_only"] == "true"
            assert row["trainable"] == "false"
            assert row["ground_truth"] == "false"
    for row in read_csv(OUT / "susc_17c9_source_artifact_inventory.csv"):
        assert not (row["artifact_is_raw_heavy"] == "true" and row["artifact_is_committed"] == "true")
    summary = json.loads((OUT / "susc_17c9_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["features_extracted_this_sprint"] == 0
    assert summary["raw_raster_committed"] is False


def test_artefato_remoto_nao_marcado_como_local_e_pipeline_nao_executado():
    for row in read_csv(OUT / "susc_17c9_source_artifact_inventory.csv"):
        assert not (row["artifact_kind"] == "remote_asset_reference" and row["artifact_exists_local"] == "true")
    for row in read_csv(OUT / "susc_17c9_pipeline_reuse_audit.csv"):
        assert row["pipeline_current_output_type"] != "executed_candidate_feature_extraction"


def test_politica_anti_feature_falsa_bloqueia_casos_obrigatorios():
    policy = json.loads((OUT / "susc_17c9_no_fake_feature_policy.json").read_text(encoding="utf-8"))
    blocked = policy["blocked_uses"]
    for key in (
        "copy_neighbor_official_patch_value",
        "interpolation_without_method_and_provenance",
        "centroid_as_area_substitute",
        "protocolo_c_patch_as_susc_patch",
        "REC_00019_as_SUSC_patch",
        "charter_footprint_as_susceptibility_feature",
        "post_event_data_as_pre_event_feature",
        "synthetic_placeholder_as_real",
        "remote_artifact_not_downloaded_as_local",
        "existing_pipeline_as_executed_extraction",
        "promotion_to_17b_without_real_feature_provenance_qa",
    ):
        assert blocked[key] is True


def test_17b_score_v7_bloqueados_e_summary_reflete_contagens():
    summary = json.loads((OUT / "susc_17c9_readiness_summary.json").read_text(encoding="utf-8"))
    inventory = read_csv(OUT / "susc_17c9_source_artifact_inventory.csv")
    pipelines = read_csv(OUT / "susc_17c9_pipeline_reuse_audit.csv")
    assert summary["candidate_patch_count"] == len(read_csv(OUT / "susc_17c6_candidate_patch_grid.csv"))
    assert summary["source_artifacts_inventory_count"] == len(inventory)
    assert summary["pipelines_audited_count"] == len(pipelines)
    assert summary["eligible_for_17b_now"] is False
    assert summary["eligible_for_score_v7"] is False
    assert read_csv(OUT / "susc_17c9_promotion_blockers.csv")
