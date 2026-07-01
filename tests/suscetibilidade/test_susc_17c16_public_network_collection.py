"""Testes do SUSC-17C16 - coleta publica com opt-in de rede."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"
DAT = ROOT / "datasets" / "suscetibilidade"

EXPECTED = [
    OUT / "SUSC_17C16_COLETA_PUBLICA_OPT_IN_REDE_CATALOGOS_REPORT.md",
    OUT / "susc_17c16_network_capability_update.csv",
    OUT / "susc_17c16_public_probe_log.csv",
    OUT / "susc_17c16_lightweight_collection_log.csv",
    OUT / "susc_17c16_collected_artifact_manifest.csv",
    OUT / "susc_17c16_chirps_catalog_probe.csv",
    OUT / "susc_17c16_sentinel2_catalog_probe.csv",
    OUT / "susc_17c16_blocked_heavy_raster_registry.csv",
    OUT / "susc_17c16_gate_evaluation_matrix.csv",
    OUT / "susc_17c16_accepted_ground_reference_candidates.csv",
    OUT / "susc_17c16_rejected_or_pending_artifacts.csv",
    OUT / "susc_17c16_anti_leakage_audit.csv",
    OUT / "susc_17c16_readiness_summary.json",
    OUT / "susc_17c16_promotion_blockers.csv",
    SCHEMAS / "susc_17c16_network_probe_schema_v1.json",
    SCHEMAS / "susc_17c16_catalog_probe_schema_v1.json",
    SCHEMAS / "susc_17c16_collected_artifact_manifest_schema_v1.json",
]


def _load_common():
    path = SCRIPTS / "susc_17c16_public_network_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s17c16_public_network_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_script(name: str, *args: str):
    env = os.environ.copy()
    env.pop("SUSC_17C16_ALLOW_NETWORK", None)
    env.pop("SUSC_17C16_ALLOW_EXTERNAL_ACTION", None)
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
    )


def test_build_offline_gera_todos_artefatos_e_valida():
    result = run_script("build_susc_17c16_public_network_collection.py")
    assert result.returncode == 0, result.stderr + result.stdout
    first = {path: path.read_bytes() for path in EXPECTED}
    result = run_script("build_susc_17c16_public_network_collection.py")
    assert result.returncode == 0, result.stderr + result.stdout
    assert {path: path.read_bytes() for path in EXPECTED} == first
    result = run_script("validate_susc_17c16_public_network_collection.py")
    assert result.returncode == 0, result.stderr + result.stdout


def test_schemas_validos_e_ids_deterministicos():
    s17c16 = _load_common()
    network_schema = json.loads((SCHEMAS / "susc_17c16_network_probe_schema_v1.json").read_text(encoding="utf-8"))
    catalog_schema = json.loads((SCHEMAS / "susc_17c16_catalog_probe_schema_v1.json").read_text(encoding="utf-8"))
    manifest_schema = json.loads((SCHEMAS / "susc_17c16_collected_artifact_manifest_schema_v1.json").read_text(encoding="utf-8"))
    for row in read_csv(OUT / "susc_17c16_public_probe_log.csv"):
        assert s17c16._schema_violations(row, network_schema) == []
    for row in read_csv(OUT / "susc_17c16_chirps_catalog_probe.csv") + read_csv(OUT / "susc_17c16_sentinel2_catalog_probe.csv"):
        assert s17c16._schema_violations(row, catalog_schema) == []
    assert read_csv(OUT / "susc_17c16_collected_artifact_manifest.csv") == []
    assert set(manifest_schema["required"]) == set(s17c16.MANIFEST_FIELDS)
    assert [row["network_capability_id"] for row in read_csv(OUT / "susc_17c16_network_capability_update.csv")] == [
        f"S17C16_NETCAP_{idx:04d}" for idx in range(1, 10)
    ]


def test_network_sem_opt_in_bloqueada_e_capabilities_nao_acessa_rede():
    for mode in ["probe-public", "collect-lightweight", "probe-sensor-catalog"]:
        result = run_script("susc_17c16_public_network_collector.py", mode)
        assert result.returncode == 2
        assert "blocked_network_not_enabled" in result.stdout
    result = run_script("susc_17c16_public_network_collector.py", "capabilities")
    assert result.returncode == 0
    assert "ChirpsCatalogProbeDriver" in result.stdout
    assert "CopernicusSentinel2CatalogProbeDriver" in result.stdout
    assert {row["attempted"] for row in read_csv(OUT / "susc_17c16_public_probe_log.csv")} == {"false"}


def test_raster_pesado_tile_e_embedding_bloqueados():
    heavy = read_csv(OUT / "susc_17c16_blocked_heavy_raster_registry.csv")
    sentinel = read_csv(OUT / "susc_17c16_sentinel2_catalog_probe.csv")
    chirps = read_csv(OUT / "susc_17c16_chirps_catalog_probe.csv")
    assert len(heavy) == 2
    assert {row["can_be_done_in_future_with_policy"] for row in heavy} == {"true"}
    assert {row["raster_downloaded"] for row in chirps + sentinel} == {"false"}
    assert {row["download_or_export_performed"] for row in chirps + sentinel} == {"false"}
    assert {row["tile_created"] for row in sentinel} == {"false"}
    summary = json.loads((OUT / "susc_17c16_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["heavy_raster_downloads_count"] == 0
    assert summary["raster_tiles_created_count"] == 0
    assert summary["dino_satmae_embeddings_created_count"] == 0


def test_chirps_e_sentinel_nao_viram_evento_observado():
    chirps = read_csv(OUT / "susc_17c16_chirps_catalog_probe.csv")
    sentinel = read_csv(OUT / "susc_17c16_sentinel2_catalog_probe.csv")
    gates = read_csv(OUT / "susc_17c16_gate_evaluation_matrix.csv")
    sensor_ids = {row["formal_request_id"] for row in chirps + sentinel}
    sensor_gates = [row for row in gates if row["formal_request_id"] in sensor_ids]
    assert {row["metadata_available"] for row in chirps + sentinel} == {"false"}
    assert {row["G4_vinculo_espacial"] for row in sensor_gates} == {"false"}
    assert {row["G5_separacao_fenomeno"] for row in sensor_gates} == {"false"}
    assert {row["all_gates_passed"] for row in sensor_gates} == {"false"}


def test_accepted_candidates_exigem_todos_os_gates_e_ficam_vazios():
    assert read_csv(OUT / "susc_17c16_accepted_ground_reference_candidates.csv") == []
    gates = read_csv(OUT / "susc_17c16_gate_evaluation_matrix.csv")
    assert {row["all_gates_passed"] for row in gates} == {"false"}
    assert {row["trainable"] for row in gates} == {"false"}
    assert {row["ground_truth"] for row in gates} == {"false"}
    rejected = read_csv(OUT / "susc_17c16_rejected_or_pending_artifacts.csv")
    assert len(rejected) == 9


def test_score_guardrails_e_17b_inelegivel():
    summary = json.loads((OUT / "susc_17c16_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["score_v6_changed"] is False
    assert summary["score_v7_created"] is False
    assert summary["eligible_for_17b_now"] is False
    assert summary["eligible_for_score_v7"] is False
    assert summary["review_only"] is True
    assert summary["trainable"] is False
    assert summary["ground_truth"] is False
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


def test_sem_submissao_protocolo_resposta_ou_label():
    summary = json.loads((OUT / "susc_17c16_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["external_submissions_performed_count"] == 0
    assert summary["protocols_opened_count"] == 0
    assert summary["responses_invented_count"] == 0
    assert summary["contacts_invented_count"] == 0
    assert summary["features_extracted_this_sprint"] == 0
    assert summary["lightweight_artifacts_collected_count"] == 0
    assert summary["artifacts_with_manifest_count"] == 0
