"""Testes do SUSC-17C31 - execucao da fila G4c/G4d/G5d (geometria oficial, SAR metadata, separacao hidrologica).

Os testes sao offline e deterministicos: usam os artefatos reais ja adquiridos
(ledger commitado). O build offline reproduz os derivados byte-identicos.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
DAT = ROOT / "datasets" / "suscetibilidade"
ARTIFACT_DIR = OUT / "susc_17c31_source_artifacts"
MAX_ARTIFACT = 1_000_000
FORBIDDEN = (".tif", ".tiff", ".zip", ".safe", ".nc", ".npz", ".npy", ".gz", ".jp2")

PLAN = OUT / "susc_17c31_target_execution_plan.csv"
ATTEMPTS = OUT / "susc_17c31_targeted_acquisition_attempts.csv"
MANIFEST = OUT / "susc_17c31_source_artifact_manifest.csv"
PARSED = OUT / "susc_17c31_parsed_artifact_index.csv"
GEOM = OUT / "susc_17c31_geometry_candidate_registry.csv"
G4C = OUT / "susc_17c31_g4c_patch_level_geometry_evaluation.csv"
G4D = OUT / "susc_17c31_g4d_patch_buffer_link_evaluation.csv"
PHSEP = OUT / "susc_17c31_phenomenon_separation_candidate_registry.csv"
G5D = OUT / "susc_17c31_g5d_hydrological_separation_evaluation.csv"
GRR = OUT / "susc_17c31_ground_reference_readiness_evaluation.csv"
SAR = OUT / "susc_17c31_sar_metadata_feasibility.csv"
TFC = OUT / "susc_17c31_technical_footprint_candidate_registry.csv"
LEAKAGE = OUT / "susc_17c31_no_leakage_audit.csv"
SUMMARY = OUT / "susc_17c31_readiness_summary.json"

EXPECTED = [
    OUT / "SUSC_17C31_EXECUCAO_FILA_G4C_G4D_G5D_REPORT.md",
    PLAN, ATTEMPTS, OUT / "susc_17c31_followed_link_registry.csv", MANIFEST, PARSED,
    GEOM, OUT / "susc_17c31_location_resolution.csv", G4C, G4D, PHSEP, G5D, GRR,
    SAR, TFC, OUT / "susc_17c31_subgate_update_matrix.csv",
    OUT / "susc_17c31_gate_transition_ledger.csv", LEAKAGE, SUMMARY,
    OUT / "susc_17c31_promotion_blockers.csv",
]


def rc(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as h:
        return list(csv.DictReader(h))


def run_script(name: str, *args: str):
    env = os.environ.copy()
    for v in ["SUSC_17C31_ALLOW_NETWORK", "SUSC_17C31_ALLOW_PUBLIC_DOWNLOAD",
              "SUSC_17C31_ALLOW_DEEP_SEARCH", "SUSC_17C31_ALLOW_WAYBACK",
              "SUSC_17C31_ALLOW_LIGHTWEIGHT_GEOSPATIAL", "SUSC_17C31_ALLOW_SAR_METADATA"]:
        env.pop(v, None)
    return subprocess.run([sys.executable, str(SCRIPTS / name), *args],
                          cwd=ROOT, env=env, text=True, capture_output=True, timeout=600, check=False)


def test_build_offline_byte_identico_e_valida():
    r = run_script("build_susc_17c31_execute_g4_g5_queue.py")
    assert r.returncode == 0, r.stderr + r.stdout
    first = {p: p.read_bytes() for p in EXPECTED}
    r = run_script("build_susc_17c31_execute_g4_g5_queue.py")
    assert r.returncode == 0, r.stderr + r.stdout
    assert {p: p.read_bytes() for p in EXPECTED} == first
    r = run_script("validate_susc_17c31_execute_g4_g5_queue.py")
    assert r.returncode == 0, r.stderr + r.stdout


def test_fila_17c30_consumida_e_tentativas_minimas():
    assert len(rc(PLAN)) >= 36
    attempts = rc(ATTEMPTS)
    assert len([a for a in attempts if a["network_enabled"] == "true"]) >= 50
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["acquisition_queue_consumed_count"] >= 36
    assert s["targeted_acquisition_attempts_count"] >= 50


def test_artefatos_reais_com_hash_e_leves():
    manifest = rc(MANIFEST)
    assert len(manifest) >= 5
    official_tech = [m for m in manifest if m["officiality_level"] in ("official", "official_institutional", "official_institutional_public_agency") or m["source_family"] == "asf_sentinel1"]
    assert len(official_tech) >= 3
    for m in manifest:
        path = ROOT / m["artifact_local_path"]
        assert path.exists(), m["artifact_local_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == m["sha256"]
        assert int(m["size_bytes"]) <= MAX_ARTIFACT
        assert m["artifact_type"] in ("html", "pdf", "txt", "csv", "json", "geojson", "kml")
    if ARTIFACT_DIR.exists():
        for p in ARTIFACT_DIR.glob("**/*"):
            if p.is_file():
                assert p.suffix.lower() not in FORBIDDEN, p.name


def test_parsed_geometry_e_phenomenon_existem():
    assert len([p for p in rc(PARSED) if p["parse_success"] == "true"]) >= 5
    geom = rc(GEOM)
    assert len(geom) >= 1
    assert any(g["geometry_candidate_type"] != "insufficient" for g in geom)
    assert len(rc(PHSEP)) >= 1
    assert len(rc(G4C)) >= 1 and len(rc(G4D)) >= 1 and len(rc(G5D)) >= 1


def test_sar_metadata_nao_vira_footprint():
    sar = rc(SAR)
    assert len(sar) >= 1
    for r in sar:
        assert r["requires_future_raster_processing"] == "true"
    for r in rc(TFC):
        assert r["requires_sentinel1_raster"] == "true"
        assert r["not_ground_truth"] == "true"
    for r in rc(LEAKAGE):
        assert r["uses_sar_metadata_as_footprint"] == "false"


def test_setor_de_risco_nao_vira_evento_observado():
    geom = rc(GEOM)
    # ha candidatos de risk_sector, todos marcados not_ground_truth e nao-evento
    for g in geom:
        assert g["not_ground_truth"] == "true"
    for r in rc(LEAKAGE):
        assert r["uses_risk_sector_as_observed_event"] == "false"
        assert r["uses_alert_as_occurrence"] == "false"
    # G4d nunca abre para risk sector
    g4d = {r["geometry_candidate_id"]: r for r in rc(G4D)}
    for g in geom:
        if g["geometry_candidate_type"] == "risk_sector_not_event":
            assert g4d[g["geometry_candidate_id"]]["G4d_patch_buffer_link_confirmed"] == "false"


def test_bairro_cidade_nao_vira_patch_level_sem_incerteza():
    for g in rc(GEOM):
        if g["geometry_candidate_type"] == "risk_sector_not_event":
            assert "bairro_level" in g["uncertainty_m"] or "3000" in g["uncertainty_m"]
    for r in rc(LEAKAGE):
        assert r["uses_city_or_bairro_as_patch_level_without_uncertainty"] == "false"
        assert r["invented_coordinate"] == "false"


def test_g5d_nao_abre_com_fenomeno_misto():
    g5d = rc(G5D)
    for r in g5d:
        if r["hydrological_signal"] == "true" and r["mass_movement_signal"] == "true":
            assert r["G5d_hydrological_separated"] == "false"
    # ha ao menos um bairro so-hidrologico com G5d true (Resultado A)
    assert any(r["G5d_hydrological_separated"] == "true" for r in g5d)
    # e ao menos um misto bloqueado
    assert any(r["hydrological_signal"] == "true" and r["mass_movement_signal"] == "true"
               and r["G5d_hydrological_separated"] == "false" for r in g5d)


def test_ground_truth_e_treino_nao_criados():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["ground_truth_created"] is False
    assert s["training_labels_created"] is False
    for r in rc(GRR):
        assert r["can_be_ground_truth"] == "false"
        assert r["can_be_training_label"] == "false"


def test_score_v6_intacto_e_score_v7_inexistente():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["score_v6_changed"] is False
    assert s["score_v7_created"] is False
    for target in [
        DAT / "susc_score_v6_candidate_by_patch_v1.csv",
        DAT / "susc_patches_official_v1.csv",
        DAT / "susc_patch_links_official_v1.csv",
    ]:
        r = subprocess.run(["git", "diff", "--name-only", "--", str(target.relative_to(ROOT))],
                           cwd=ROOT, text=True, capture_output=True, check=False)
        assert r.stdout.strip() == ""
    assert not (DAT / "susc_score_v7_candidate_by_patch_v1.csv").exists()


def test_17b_inelegivel_sem_ground_reference_aceito():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    accepted = s["accepted_ground_reference_candidate_review_only_count"]
    assert s["eligible_for_17b_now"] == (accepted > 0)
    # neste marco G4_full=0 -> nenhum GR aceito -> 17B bloqueado
    assert s["G4_full_true_count"] == 0
    assert accepted == 0
    assert s["eligible_for_17b_now"] is False


def test_acquisition_bloqueia_sem_env():
    r = run_script("susc_17c31_execute_g4_g5_queue.py", "run-targeted-acquisition")
    assert "BLOCKED" in (r.stdout + r.stderr)
