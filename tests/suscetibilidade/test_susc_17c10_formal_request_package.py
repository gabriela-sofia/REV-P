"""Testes do SUSC-17C10 - pacote de solicitacao formal."""

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
    OUT / "SUSC_WORKTREE_AUDIT_BEFORE_17C10.md",
    OUT / "SUSC_17C10_PACOTE_SOLICITACAO_FORMAL_REPORT.md",
    OUT / "susc_17c10_formal_request_registry.csv",
    OUT / "susc_17c10_provider_package_index.csv",
    OUT / "susc_17c10_required_fields_by_request.csv",
    OUT / "susc_17c10_request_to_blocker_traceability.csv",
    OUT / "susc_17c10_request_impact_matrix.csv",
    OUT / "susc_17c10_manual_submission_checklist.csv",
    OUT / "susc_17c10_request_message_templates.md",
    OUT / "susc_17c10_attachment_manifest.csv",
    OUT / "susc_17c10_response_intake_schema_plan.csv",
    OUT / "susc_17c10_readiness_summary.json",
    OUT / "susc_17c10_promotion_blockers.csv",
    SCHEMAS / "susc_17c10_formal_request_schema_v1.json",
    SCHEMAS / "susc_17c10_request_impact_schema_v1.json",
]


def _load_common():
    path = SCRIPTS / "susc_17c10_formal_request_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s17c10_common", path)
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
    result = run_script("build_susc_17c10_formal_request_package.py")
    assert result.returncode == 0, result.stderr + result.stdout
    for path in EXPECTED:
        assert path.exists(), path
    result = run_script("validate_susc_17c10_formal_request_package.py")
    assert result.returncode == 0, result.stderr + result.stdout


def test_schemas_validos():
    s17c10 = _load_common()
    request_schema = json.loads((SCHEMAS / "susc_17c10_formal_request_schema_v1.json").read_text(encoding="utf-8"))
    impact_schema = json.loads((SCHEMAS / "susc_17c10_request_impact_schema_v1.json").read_text(encoding="utf-8"))
    for row in read_csv(OUT / "susc_17c10_formal_request_registry.csv"):
        assert s17c10._schema_violations(row, request_schema) == []
    for row in read_csv(OUT / "susc_17c10_request_impact_matrix.csv"):
        assert s17c10._schema_violations(row, impact_schema) == []


def test_build_duas_vezes_byte_identico_e_ids_deterministicos():
    first = tuple(path.read_bytes() for path in EXPECTED)
    result = run_script("build_susc_17c10_formal_request_package.py")
    assert result.returncode == 0, result.stderr + result.stdout
    second = tuple(path.read_bytes() for path in EXPECTED)
    assert first == second
    ids = [r["formal_request_id"] for r in read_csv(OUT / "susc_17c10_formal_request_registry.csv")]
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))


def test_sem_score_v7_score_v6_ou_dataset_oficial_alterado():
    assert not (ROOT / "datasets" / "suscetibilidade" / "susc_score_v7_candidate_by_patch_v1.csv").exists()
    summary = json.loads((OUT / "susc_17c10_readiness_summary.json").read_text(encoding="utf-8"))
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


def test_sem_trainable_ground_truth_feature_real_raster_ou_envio():
    for row in read_csv(OUT / "susc_17c10_formal_request_registry.csv"):
        assert row["review_only"] == "true"
        assert row["trainable"] == "false"
        assert row["ground_truth"] == "false"
        assert row["request_status"] != "sent"
        assert row["do_not_mark_as_received"] == "true"
        assert row["submission_channel_reference"] == "not_available"
        assert "@" not in row["submission_channel_reference"]
    summary = json.loads((OUT / "susc_17c10_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["requests_marked_sent_count"] == 0
    assert summary["responses_received_count"] == 0
    assert summary["features_extracted_this_sprint"] == 0
    assert summary["raw_raster_committed"] is False


def test_p0_drm_p0_compdec_e_sgb_presentes():
    ids = {r["formal_request_id"] for r in read_csv(OUT / "susc_17c10_formal_request_registry.csv")}
    assert "P0_DRM_RJ_PKG_FR_PET_001" in ids
    assert "P0_COMPDEC_RECIFE_PKG_FR_REC_002" in ids
    assert "P1_SGB_CPRM_VECTOR_OR_COORDINATES" in ids


def test_modelos_portugues_e_checklist_nao_concluido():
    text = (OUT / "susc_17c10_request_message_templates.md").read_text(encoding="utf-8")
    assert "Prezadas(os)" in text
    assert "pesquisa academica" in text
    assert "nao enviar" in text
    assert "ground truth operacional" in text
    assert "@" not in text
    for row in read_csv(OUT / "susc_17c10_manual_submission_checklist.csv"):
        assert row["done"] == "false"


def test_17b_score_v7_bloqueados_e_summary_reflete_contagens():
    summary = json.loads((OUT / "susc_17c10_readiness_summary.json").read_text(encoding="utf-8"))
    registry = read_csv(OUT / "susc_17c10_formal_request_registry.csv")
    assert summary["formal_requests_count"] == len(registry)
    assert summary["p0_requests_count"] == 2
    assert summary["p1_requests_count"] == 5
    assert summary["p2_requests_count"] == 2
    assert summary["eligible_for_17b_now"] is False
    assert summary["eligible_for_score_v7"] is False
    assert read_csv(OUT / "susc_17c10_promotion_blockers.csv")
