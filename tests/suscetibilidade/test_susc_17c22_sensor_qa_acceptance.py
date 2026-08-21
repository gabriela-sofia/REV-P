"""Testes do SUSC-17C22 - aceite formal automatizado do QA sensorial e feature matrix candidata."""

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
DAT = ROOT / "datasets" / "suscetibilidade"

EXPECTED = [
    OUT / "SUSC_17C22_ACEITE_FORMAL_QA_SENSORIAL_FEATURE_MATRIX_REPORT.md",
    OUT / "susc_17c22_sensor_qa_acceptance_policy.csv",
    OUT / "susc_17c22_pre_event_sensor_feature_acceptance.csv",
    OUT / "susc_17c22_observational_delta_acceptance.csv",
    OUT / "susc_17c22_candidate_patch_sensor_feature_matrix.csv",
    OUT / "susc_17c22_candidate_patch_observational_delta_matrix.csv",
    OUT / "susc_17c22_feature_provenance_trace.csv",
    OUT / "susc_17c22_fallback_policy_review.csv",
    OUT / "susc_17c22_science_readiness_gates.csv",
    OUT / "susc_17c22_training_and_score_blockers.csv",
    OUT / "susc_17c22_no_leakage_audit.csv",
    OUT / "susc_17c22_gate_evaluation_matrix.csv",
    OUT / "susc_17c22_readiness_summary.json",
    OUT / "susc_17c22_promotion_blockers.csv",
    SCHEMAS / "susc_17c22_feature_acceptance_schema_v1.json",
    SCHEMAS / "susc_17c22_feature_matrix_schema_v1.json",
    SCHEMAS / "susc_17c22_provenance_trace_schema_v1.json",
]


def _load_common():
    path = SCRIPTS / "susc_17c22_sensor_qa_acceptance_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s17c22_sensor_qa_acceptance_common", path)
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
    result = run_script("build_susc_17c22_sensor_qa_acceptance.py")
    assert result.returncode == 0, result.stderr + result.stdout
    first = {path: path.read_bytes() for path in EXPECTED}
    result = run_script("build_susc_17c22_sensor_qa_acceptance.py")
    assert result.returncode == 0, result.stderr + result.stdout
    assert {path: path.read_bytes() for path in EXPECTED} == first
    result = run_script("validate_susc_17c22_sensor_qa_acceptance.py")
    assert result.returncode == 0, result.stderr + result.stdout


def test_schemas_validos_e_ids_deterministicos():
    s17c22 = _load_common()
    acc_schema = json.loads((SCHEMAS / "susc_17c22_feature_acceptance_schema_v1.json").read_text(encoding="utf-8"))
    matrix_schema = json.loads((SCHEMAS / "susc_17c22_feature_matrix_schema_v1.json").read_text(encoding="utf-8"))
    prov_schema = json.loads((SCHEMAS / "susc_17c22_provenance_trace_schema_v1.json").read_text(encoding="utf-8"))
    for row in read_csv(OUT / "susc_17c22_pre_event_sensor_feature_acceptance.csv"):
        assert s17c22._schema_violations(row, acc_schema) == []
    for row in read_csv(OUT / "susc_17c22_candidate_patch_sensor_feature_matrix.csv"):
        assert s17c22._schema_violations(row, matrix_schema) == []
    for row in read_csv(OUT / "susc_17c22_feature_provenance_trace.csv"):
        assert s17c22._schema_violations(row, prov_schema) == []
    assert [row["feature_acceptance_id"] for row in read_csv(OUT / "susc_17c22_pre_event_sensor_feature_acceptance.csv")] == [
        f"S17C22_FEATACC_{idx:04d}" for idx in range(1, 41)
    ]


def test_40_features_pre_evento_avaliadas_e_aceitas():
    pre = read_csv(OUT / "susc_17c22_pre_event_sensor_feature_acceptance.csv")
    assert len(pre) == 40
    accepted = [r for r in pre if r["acceptance_status"] == "accepted_review_only_sensor_feature"]
    assert len(accepted) == 40
    assert {r["usable_as_review_only_sensor_feature"] for r in accepted} == {"true"}
    assert {r["usable_for_training"] for r in pre} == {"false"}
    assert {r["ground_truth"] for r in pre} == {"false"}
    assert {r["eligible_for_score_v7"] for r in pre} == {"false"}
    assert {r["usable_for_science_now"] for r in pre} == {"false"}


def test_matriz_sensorial_5_patches_sem_deltas():
    fm = read_csv(OUT / "susc_17c22_candidate_patch_sensor_feature_matrix.csv")
    assert len(fm) == 5
    assert all(not col.startswith("delta_") for col in fm[0].keys())
    for band in ["B03_mean", "B04_mean", "B08_mean", "B11_mean", "NDVI_mean", "NDWI_mean", "MNDWI_mean", "NDBI_mean"]:
        assert band in fm[0]
    assert {r["usable_for_science_now"] for r in fm} == {"false"}
    assert {r["eligible_for_score_v7"] for r in fm} == {"false"}


def test_40_deltas_aceitos_observacionais_e_nao_treino():
    delta = read_csv(OUT / "susc_17c22_observational_delta_acceptance.csv")
    assert len(delta) == 40
    accepted = [r for r in delta if r["acceptance_status"] == "accepted_observational_change_review_only"]
    assert len(accepted) == 40
    assert {r["usable_as_pre_event_feature"] for r in delta} == {"false"}
    assert {r["usable_for_training"] for r in delta} == {"false"}
    assert {r["ground_truth"] for r in delta} == {"false"}
    dm = read_csv(OUT / "susc_17c22_candidate_patch_observational_delta_matrix.csv")
    assert len(dm) == 5
    for delta_col in ["delta_B03_mean", "delta_NDVI_mean", "delta_MNDWI_mean"]:
        assert delta_col in dm[0]
    assert {r["usable_as_pre_event_feature"] for r in dm} == {"false"}


def test_pos_evento_nao_vira_pre_evento_e_delta_nao_label():
    leakage = read_csv(OUT / "susc_17c22_no_leakage_audit.csv")
    assert {r["uses_post_event_as_pre_event_feature"] for r in leakage} == {"false"}
    assert {r["uses_delta_as_training_label"] for r in leakage} == {"false"}
    assert {r["passes_no_leakage"] for r in leakage} == {"true"}
    summary = json.loads((OUT / "susc_17c22_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["post_event_used_as_pre_event_feature_count"] == 0
    assert summary["delta_used_as_training_label_count"] == 0
    assert summary["features_promoted_to_training_count"] == 0


def test_fallback_declarado_nao_ciencia_nao_ground_reference():
    fb = read_csv(OUT / "susc_17c22_fallback_policy_review.csv")
    assert {r["fallback_declared"] for r in fb} == {"true"}
    assert {r["can_be_used_for_review_only_sensor_feature"] for r in fb} == {"true"}
    assert {r["can_be_used_for_science_now"] for r in fb} == {"false"}
    assert {r["can_be_used_for_ground_reference"] for r in fb} == {"false"}
    assert {r["requires_external_policy_review"] for r in fb} == {"true"}
    summary = json.loads((OUT / "susc_17c22_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["fallback_accepted_for_science_now_count"] == 0
    assert summary["fallback_accepted_for_review_only_count"] == 5


def test_nada_vira_ground_reference():
    gates = read_csv(OUT / "susc_17c22_gate_evaluation_matrix.csv")
    assert {r["G4_vinculo_espacial_evento"] for r in gates} == {"false"}
    assert {r["G5_separacao_fenomeno"] for r in gates} == {"false"}
    assert {r["all_gates_passed_for_ground_reference"] for r in gates} == {"false"}
    summary = json.loads((OUT / "susc_17c22_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["ground_reference_candidates_created_count"] == 0


def test_score_guardrails_e_17b_inelegivel():
    summary = json.loads((OUT / "susc_17c22_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["minimum_success_achieved"] is True
    assert summary["score_v6_changed"] is False
    assert summary["score_v7_created"] is False
    assert summary["eligible_for_17b_now"] is False
    assert summary["eligible_for_score_v7"] is False
    assert summary["ground_truth_created"] is False
    assert summary["training_labels_created"] is False
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
    ts = read_csv(OUT / "susc_17c22_training_and_score_blockers.csv")
    assert {r["blocks_training"] for r in ts} == {"true"}
    assert {r["blocks_score_v7"] for r in ts} == {"true"}
    assert {r["blocks_17b"] for r in ts} == {"true"}
    blockers = read_csv(OUT / "susc_17c22_promotion_blockers.csv")
    assert any(row["blocker_type"] == "17b_blocked_until_event_reference_and_policy" for row in blockers)
