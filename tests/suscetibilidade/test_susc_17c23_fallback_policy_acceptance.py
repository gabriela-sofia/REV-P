"""Testes do SUSC-17C23 - politica de equivalencia Earth Search e matriz cientifica review-only."""

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
    OUT / "SUSC_17C23_POLITICA_EQUIVALENCIA_EARTH_SEARCH_MATRIZ_CIENTIFICA_REVIEW_ONLY_REPORT.md",
    OUT / "susc_17c23_earth_search_resolution_dossier.csv",
    OUT / "susc_17c23_fallback_equivalence_policy.csv",
    OUT / "susc_17c23_fallback_scene_equivalence_audit.csv",
    OUT / "susc_17c23_fallback_acceptance_decision.csv",
    OUT / "susc_17c23_candidate_patch_usage_policy.csv",
    OUT / "susc_17c23_candidate_patch_usage_decision.csv",
    OUT / "susc_17c23_scientific_review_only_feature_acceptance.csv",
    OUT / "susc_17c23_candidate_multimodal_feature_matrix.csv",
    OUT / "susc_17c23_observational_delta_review_matrix.csv",
    OUT / "susc_17c23_science_scope_limits.csv",
    OUT / "susc_17c23_training_score_17b_blockers.csv",
    OUT / "susc_17c23_no_leakage_audit.csv",
    OUT / "susc_17c23_gate_evaluation_matrix.csv",
    OUT / "susc_17c23_readiness_summary.json",
    OUT / "susc_17c23_promotion_blockers.csv",
    SCHEMAS / "susc_17c23_earth_search_resolution_schema_v1.json",
    SCHEMAS / "susc_17c23_candidate_patch_usage_schema_v1.json",
    SCHEMAS / "susc_17c23_scientific_feature_acceptance_schema_v1.json",
]


def _load_common():
    path = SCRIPTS / "susc_17c23_fallback_policy_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s17c23_fallback_policy_common", path)
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
        cwd=ROOT, text=True, capture_output=True, timeout=600, check=False,
    )


def test_build_offline_gera_todos_artefatos_e_valida():
    result = run_script("build_susc_17c23_fallback_policy_acceptance.py")
    assert result.returncode == 0, result.stderr + result.stdout
    first = {path: path.read_bytes() for path in EXPECTED}
    result = run_script("build_susc_17c23_fallback_policy_acceptance.py")
    assert result.returncode == 0, result.stderr + result.stdout
    assert {path: path.read_bytes() for path in EXPECTED} == first
    result = run_script("validate_susc_17c23_fallback_policy_acceptance.py")
    assert result.returncode == 0, result.stderr + result.stdout


def test_schemas_validos_e_ids_deterministicos():
    s17c23 = _load_common()
    dossier_schema = json.loads((SCHEMAS / "susc_17c23_earth_search_resolution_schema_v1.json").read_text(encoding="utf-8"))
    patch_schema = json.loads((SCHEMAS / "susc_17c23_candidate_patch_usage_schema_v1.json").read_text(encoding="utf-8"))
    sci_schema = json.loads((SCHEMAS / "susc_17c23_scientific_feature_acceptance_schema_v1.json").read_text(encoding="utf-8"))
    for row in read_csv(OUT / "susc_17c23_earth_search_resolution_dossier.csv"):
        assert s17c23._schema_violations(row, dossier_schema) == []
    for row in read_csv(OUT / "susc_17c23_candidate_patch_usage_decision.csv"):
        assert s17c23._schema_violations(row, patch_schema) == []
    for row in read_csv(OUT / "susc_17c23_scientific_review_only_feature_acceptance.csv"):
        assert s17c23._schema_violations(row, sci_schema) == []
    assert [row["earth_search_resolution_id"] for row in read_csv(OUT / "susc_17c23_earth_search_resolution_dossier.csv")] == [
        f"S17C23_ESRES_{idx:04d}" for idx in range(1, 6)
    ]


def test_dossier_existe_e_5_patches_equivalencia_resolvida():
    dossier = read_csv(OUT / "susc_17c23_earth_search_resolution_dossier.csv")
    assert len(dossier) == 5
    assert {r["equivalence_resolved"] for r in dossier} == {"true"}
    assert {r["accepted_for_scientific_review_only"] for r in dossier} == {"true"}
    assert {r["same_acquisition_date"] for r in dossier} == {"true"}
    assert {r["same_mgrs_tile"] for r in dossier} == {"true"}
    assert {r["hash_manifest_verified"] for r in dossier} == {"true"}
    assert {r["replay_verified"] for r in dossier} == {"true"}
    summary = json.loads((OUT / "susc_17c23_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["earth_search_equivalence_resolved"] is True
    assert summary["fallback_policy_review_blockers_count"] == 0


def test_l1c_l2a_registrada_como_limitacao():
    dossier = read_csv(OUT / "susc_17c23_earth_search_resolution_dossier.csv")
    assert all("l1c_vs_earth_search_l2a" in r["remaining_limitation"] for r in dossier)
    assert {r["cdse_processing_level"] for r in dossier} == {"L1C"}
    assert {r["earth_search_processing_level"] for r in dossier} == {"L2A"}
    limits = read_csv(OUT / "susc_17c23_science_scope_limits.csv")
    l1c = [r for r in limits if r["limit_name"] == "processing_level_l1c_vs_l2a"]
    assert l1c and l1c[0]["blocks_scientific_review_only"] == "false"


def test_earth_search_nao_vira_ground_reference_treino_score_17b():
    fb = read_csv(OUT / "susc_17c23_fallback_acceptance_decision.csv")
    assert {r["accepted_for_scientific_review_only"] for r in fb} == {"true"}
    assert {r["accepted_for_ground_reference"] for r in fb} == {"false"}
    assert {r["accepted_for_training"] for r in fb} == {"false"}
    assert {r["accepted_for_score_v7"] for r in fb} == {"false"}
    assert {r["accepted_for_17b"] for r in fb} == {"false"}
    assert {r["not_official_primary_source"] for r in fb} == {"true"}
    summary = json.loads((OUT / "susc_17c23_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["fallback_accepted_for_ground_reference_count"] == 0
    assert summary["fallback_accepted_for_training_count"] == 0
    assert summary["fallback_accepted_for_score_v7_count"] == 0
    assert summary["fallback_accepted_for_17b_count"] == 0


def test_40_features_scientific_review_only_e_matriz_sem_deltas():
    sci = read_csv(OUT / "susc_17c23_scientific_review_only_feature_acceptance.csv")
    accepted = [r for r in sci if r["acceptance_status"] == "accepted_scientific_review_only_sensor_feature"]
    assert len(accepted) == 40
    assert {r["usable_for_training"] for r in sci} == {"false"}
    assert {r["eligible_for_score_v7"] for r in sci} == {"false"}
    assert {r["eligible_for_17b"] for r in sci} == {"false"}
    matrix = read_csv(OUT / "susc_17c23_candidate_multimodal_feature_matrix.csv")
    assert len(matrix) == 5
    assert all(not col.startswith("delta_") for col in matrix[0].keys())
    for band in ["B03_mean", "NDVI_mean", "MNDWI_mean"]:
        assert band in matrix[0]
    assert {r["usable_for_science_now"] for r in matrix} == {"true"}
    assert {r["usable_for_training"] for r in matrix} == {"false"}


def test_deltas_mantidos_observacionais():
    dm = read_csv(OUT / "susc_17c23_observational_delta_review_matrix.csv")
    assert len(dm) == 5
    for col in ["delta_B03_mean", "delta_NDVI_mean", "delta_NDBI_mean"]:
        assert col in dm[0]
    assert {r["usable_as_observational_change_review_only"] for r in dm} == {"true"}
    assert {r["usable_as_pre_event_feature"] for r in dm} == {"false"}
    assert {r["usable_for_training"] for r in dm} == {"false"}


def test_patch_candidato_nao_vira_oficial_e_score_guardrails():
    patch = read_csv(OUT / "susc_17c23_candidate_patch_usage_decision.csv")
    assert len(patch) == 5
    assert {r["not_official_patch"] for r in patch} == {"true"}
    assert {r["not_official_patch_link"] for r in patch} == {"true"}
    assert {r["usage_decision_status"] for r in patch} == {"accepted_candidate_patch_scientific_review_only"}
    summary = json.loads((OUT / "susc_17c23_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["minimum_success_achieved"] is True
    assert summary["score_v6_changed"] is False
    assert summary["score_v7_created"] is False
    assert summary["eligible_for_17b_now"] is False
    assert summary["eligible_for_score_v7"] is False
    assert summary["ground_reference_candidates_created_count"] == 0
    assert summary["ground_truth_created"] is False
    assert summary["training_labels_created"] is False
    assert summary["official_patch_created"] is False
    assert not (DAT / "susc_score_v7_candidate_by_patch_v1.csv").exists()
    for target in [
        DAT / "susc_score_v6_candidate_by_patch_v1.csv",
        DAT / "susc_patches_official_v1.csv",
        DAT / "susc_patch_links_official_v1.csv",
    ]:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--", str(target.relative_to(ROOT))],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        assert result.stdout.strip() == ""
    ts = read_csv(OUT / "susc_17c23_training_score_17b_blockers.csv")
    assert {r["blocks_training"] for r in ts} == {"true"}
    assert {r["blocks_17b"] for r in ts} == {"true"}
