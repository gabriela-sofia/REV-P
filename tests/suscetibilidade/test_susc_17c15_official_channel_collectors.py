"""Testes do SUSC-17C15 - coletores oficiais controlados."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"
DAT = ROOT / "datasets" / "suscetibilidade"

EXPECTED = [
    OUT / "SUSC_17C15_COLETORES_OFICIAIS_CONTROLADOS_REPORT.md",
    OUT / "susc_17c15_channel_capability_matrix.csv",
    OUT / "susc_17c15_public_collection_plan.csv",
    OUT / "susc_17c15_collection_execution_log.csv",
    OUT / "susc_17c15_collected_artifact_manifest.csv",
    OUT / "susc_17c15_collected_pdf_text_probe.csv",
    OUT / "susc_17c15_sensor_catalog_probe.csv",
    OUT / "susc_17c15_blocked_external_action_registry.csv",
    OUT / "susc_17c15_gate_evaluation_matrix.csv",
    OUT / "susc_17c15_accepted_ground_reference_candidates.csv",
    OUT / "susc_17c15_rejected_or_pending_artifacts.csv",
    OUT / "susc_17c15_anti_leakage_audit.csv",
    OUT / "susc_17c15_readiness_summary.json",
    OUT / "susc_17c15_promotion_blockers.csv",
    SCHEMAS / "susc_17c15_channel_capability_schema_v1.json",
    SCHEMAS / "susc_17c15_collected_artifact_manifest_schema_v1.json",
    SCHEMAS / "susc_17c15_gate_evaluation_schema_v1.json",
]


def _load_common():
    path = SCRIPTS / "susc_17c15_channel_collector_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s17c15_channel_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_script(name: str, *args: str):
    env = os.environ.copy()
    env.pop("SUSC_17C15_ALLOW_NETWORK", None)
    env.pop("SUSC_17C15_ALLOW_EXTERNAL_ACTION", None)
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
    )


def test_build_gera_todos_artefatos_e_valida():
    result = run_script("build_susc_17c15_official_channel_collectors.py")
    assert result.returncode == 0, result.stderr + result.stdout
    for path in EXPECTED:
        assert path.exists(), path
    result = run_script("validate_susc_17c15_official_channel_collectors.py")
    assert result.returncode == 0, result.stderr + result.stdout


def test_schemas_e_ids_deterministicos():
    s17c15 = _load_common()
    capability_schema = json.loads((SCHEMAS / "susc_17c15_channel_capability_schema_v1.json").read_text(encoding="utf-8"))
    manifest_schema = json.loads((SCHEMAS / "susc_17c15_collected_artifact_manifest_schema_v1.json").read_text(encoding="utf-8"))
    gate_schema = json.loads((SCHEMAS / "susc_17c15_gate_evaluation_schema_v1.json").read_text(encoding="utf-8"))
    for row in read_csv(OUT / "susc_17c15_channel_capability_matrix.csv"):
        assert s17c15._schema_violations(row, capability_schema) == []
    for row in read_csv(OUT / "susc_17c15_collected_artifact_manifest.csv"):
        assert s17c15._schema_violations(row, manifest_schema) == []
    for row in read_csv(OUT / "susc_17c15_gate_evaluation_matrix.csv"):
        assert s17c15._schema_violations(row, gate_schema) == []
    assert [row["capability_id"] for row in read_csv(OUT / "susc_17c15_channel_capability_matrix.csv")] == [
        f"S17C15_CAP_{idx:04d}" for idx in range(1, 10)
    ]


def test_drivers_expostos_e_rede_bloqueada_por_padrao():
    s17c15 = _load_common()
    assert set(s17c15.DRIVER_NAMES) == {
        "OfficialStaticDownloadDriver",
        "OfficialPageProbeDriver",
        "OfficialPdfTextProbeDriver",
        "OfficialDataPortalDriver",
        "OfficialSensorCatalogProbeDriver",
        "BlockedAuthenticatedPortalDriver",
    }
    for mode in ["collect-public", "probe"]:
        result = run_script("susc_17c15_official_channel_collector.py", mode)
        assert result.returncode == 2
        assert "blocked_network_not_enabled" in result.stdout
    log = read_csv(OUT / "susc_17c15_collection_execution_log.csv")
    assert {row["network_enabled"] for row in log} == {"false"}
    assert {row["external_effect"] for row in log} == {"false"}
    assert {row["artifact_collected"] for row in log} == {"false"}


def test_canais_controlados_sem_acao_externa():
    caps = read_csv(OUT / "susc_17c15_channel_capability_matrix.csv")
    blocked = read_csv(OUT / "susc_17c15_blocked_external_action_registry.csv")
    assert len(caps) == 9
    assert len([row for row in caps if row["can_collect_publicly"] == "true"]) == 2
    assert len(blocked) == 5
    assert {row["can_be_done_by_agent_later"] for row in blocked} == {"false"}
    assert {row["requires_human_or_institutional_auth"] for row in blocked} == {"true"}


def test_sem_pdf_aceito_sem_sensor_coletado_e_sem_manifesto_real():
    assert read_csv(OUT / "susc_17c15_collected_pdf_text_probe.csv") == []
    sensors = read_csv(OUT / "susc_17c15_sensor_catalog_probe.csv")
    assert len(sensors) == 2
    assert {row["metadata_available"] for row in sensors} == {"false"}
    assert {row["raster_downloaded"] for row in sensors} == {"false"}
    manifests = read_csv(OUT / "susc_17c15_collected_artifact_manifest.csv")
    assert {row["sha256"] for row in manifests} == {"not_available"}
    assert {row["artifact_local_path"] for row in manifests} == {"not_available"}
    assert {row["stored_in_outputs_public"] for row in manifests} == {"false"}


def test_gates_fail_closed_sem_feature_label_ou_ground_truth():
    gates = read_csv(OUT / "susc_17c15_gate_evaluation_matrix.csv")
    accepted = read_csv(OUT / "susc_17c15_accepted_ground_reference_candidates.csv")
    leakage = read_csv(OUT / "susc_17c15_anti_leakage_audit.csv")
    assert accepted == []
    assert {row["all_gates_passed"] for row in gates} == {"false"}
    assert {row["G4_vinculo_espacial"] for row in gates} == {"false"}
    assert {row["G5_separacao_fenomeno"] for row in gates} == {"false"}
    assert {row["G6_proveniencia_integridade"] for row in gates} == {"false"}
    assert {row["trainable"] for row in gates} == {"false"}
    assert {row["ground_truth"] for row in gates} == {"false"}
    assert {row["passes_anti_leakage"] for row in leakage} == {"true"}


def test_summary_e_guardrails_de_promocao():
    summary = json.loads((OUT / "susc_17c15_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["channels_evaluated_count"] == 9
    assert summary["channels_collectable_publicly_count"] == 2
    assert summary["channels_blocked_authenticated_count"] == 5
    assert summary["collection_attempts_count"] == 9
    assert summary["artifacts_collected_count"] == 0
    assert summary["artifacts_all_gates_passed_count"] == 0
    assert summary["accepted_ground_reference_candidates_count"] == 0
    assert summary["rejected_or_pending_artifacts_count"] == 9
    assert summary["external_submissions_performed_count"] == 0
    assert summary["protocols_opened_count"] == 0
    assert summary["responses_invented_count"] == 0
    assert summary["contacts_invented_count"] == 0
    assert summary["score_v6_changed"] is False
    assert summary["score_v7_created"] is False
    assert summary["eligible_for_17b_now"] is False
    assert summary["eligible_for_score_v7"] is False
    assert summary["review_only"] is True
    assert summary["trainable"] is False
    assert summary["ground_truth"] is False


def test_datasets_oficiais_e_score_v7_intocados():
    score_v7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"
    assert not score_v7.exists()
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
