"""Testes do SUSC-17C29 - aquisicao de geometria local e separacao de fenomeno G4/G5."""

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
ARTIFACT_DIR = OUT / "susc_17c29_source_artifacts"
MAX_ARTIFACT = 500_000
OFFICIAL_LEVELS = {"official", "official_institutional", "official_institutional_public_agency"}

EXPECTED = [
    OUT / "SUSC_17C29_AQUISICAO_GEOMETRIA_LOCAL_FENOMENO_G4_G5_REPORT.md",
    OUT / "susc_17c29_local_search_plan.csv",
    OUT / "susc_17c29_local_source_acquisition_attempts.csv",
    OUT / "susc_17c29_local_followed_link_registry.csv",
    OUT / "susc_17c29_local_source_artifact_manifest.csv",
    OUT / "susc_17c29_local_parsed_artifact_index.csv",
    OUT / "susc_17c29_local_observed_event_candidates.csv",
    OUT / "susc_17c29_local_location_resolution.csv",
    OUT / "susc_17c29_local_phenomenon_classification.csv",
    OUT / "susc_17c29_g4_spatial_link_evaluation.csv",
    OUT / "susc_17c29_g5_phenomenon_evaluation.csv",
    OUT / "susc_17c29_ground_reference_candidate_evaluation.csv",
    OUT / "susc_17c29_local_artifact_scorecard.csv",
    OUT / "susc_17c29_evidence_graph_update_nodes.csv",
    OUT / "susc_17c29_evidence_graph_update_edges.csv",
    OUT / "susc_17c29_no_leakage_audit.csv",
    OUT / "susc_17c29_gate_evaluation_matrix.csv",
    OUT / "susc_17c29_readiness_summary.json",
    OUT / "susc_17c29_promotion_blockers.csv",
]


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_script(name: str, *args: str):
    env = os.environ.copy()
    for var in ["SUSC_17C29_ALLOW_NETWORK", "SUSC_17C29_ALLOW_PUBLIC_DOWNLOAD",
                "SUSC_17C29_ALLOW_DEEP_SEARCH", "SUSC_17C29_ALLOW_WAYBACK"]:
        env.pop(var, None)
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *args],
        cwd=ROOT, env=env, text=True, capture_output=True, timeout=600, check=False,
    )


def test_build_offline_byte_identico_e_valida():
    result = run_script("build_susc_17c29_local_geometry_phenomenon.py")
    assert result.returncode == 0, result.stderr + result.stdout
    first = {path: path.read_bytes() for path in EXPECTED}
    result = run_script("build_susc_17c29_local_geometry_phenomenon.py")
    assert result.returncode == 0, result.stderr + result.stdout
    assert {path: path.read_bytes() for path in EXPECTED} == first
    result = run_script("validate_susc_17c29_local_geometry_phenomenon.py")
    assert result.returncode == 0, result.stderr + result.stdout


def test_plano_e_tentativas_locais_minimas():
    plan = read_csv(OUT / "susc_17c29_local_search_plan.csv")
    assert len(plan) >= 40
    attempts = read_csv(OUT / "susc_17c29_local_source_acquisition_attempts.csv")
    assert len([a for a in attempts if a["network_enabled"] == "true"]) >= 40
    summary = json.loads((OUT / "susc_17c29_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["local_geometry_search_attempts_count"] >= 40


def test_artefatos_locais_oficiais_com_hash():
    manifests = read_csv(OUT / "susc_17c29_local_source_artifact_manifest.csv")
    official = [row for row in manifests if row["local_specific"] == "true" and row["officiality_level"] in OFFICIAL_LEVELS]
    assert len(official) >= 3
    assert any(row["officiality_level"] == "official_institutional_public_agency" for row in official)
    for row in manifests:
        path = ROOT / row["artifact_local_path"]
        assert path.exists(), row["artifact_local_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        assert int(row["size_bytes"]) <= MAX_ARTIFACT
        assert row["artifact_type"] in ("html", "pdf", "csv", "json", "txt")
    if ARTIFACT_DIR.exists():
        for path in ARTIFACT_DIR.glob("**/*"):
            if path.is_file():
                assert path.suffix.lower() not in (".tif", ".nc", ".zip", ".gz", ".npz", ".npy")


def test_artefatos_locais_sao_novos_nao_reusam_17c28():
    c29 = read_csv(OUT / "susc_17c29_local_source_artifact_manifest.csv")
    c28 = read_csv(OUT / "susc_17c28_deep_source_artifact_manifest.csv")
    c28_urls = {row["source_url"].split("?")[0] for row in c28}
    for row in c29:
        assert row["source_url"].split("?")[0] not in c28_urls, row["source_url"]


def test_parse_candidatos_geocodaveis_e_hidrologico():
    parsed = read_csv(OUT / "susc_17c29_local_parsed_artifact_index.csv")
    assert len([p for p in parsed if p["parse_success"] == "true"]) >= 3
    candidates = read_csv(OUT / "susc_17c29_local_observed_event_candidates.csv")
    assert len(candidates) >= 3
    assert len([c for c in candidates if c["geocodable_location"] == "true"]) >= 1
    phenomena = read_csv(OUT / "susc_17c29_local_phenomenon_classification.csv")
    assert len([p for p in phenomena if p["hydrological_specific"] == "true"]) >= 1
    g4 = read_csv(OUT / "susc_17c29_g4_spatial_link_evaluation.csv")
    g5 = read_csv(OUT / "susc_17c29_g5_phenomenon_evaluation.csv")
    assert len(g4) + len(g5) >= 3


def test_cidade_nao_vira_patch_level_sem_incerteza():
    locations = read_csv(OUT / "susc_17c29_local_location_resolution.csv")
    assert {row["patch_level_geometry"] for row in locations} == {"false"}
    assert {row["within_patch_or_buffer"] for row in locations} == {"false"}
    assert {row["resolved_coordinate"] for row in locations} == {"not_available_not_invented"}
    candidates = read_csv(OUT / "susc_17c29_local_observed_event_candidates.csv")
    assert {row["geometry_type"] for row in candidates} == {"none"}
    g4 = read_csv(OUT / "susc_17c29_g4_spatial_link_evaluation.csv")
    assert {row["G4_vinculo_espacial_evento"] for row in g4} == {"false"}


def test_fenomeno_misto_nao_vira_g5():
    phenomena = {p["local_observed_event_candidate_id"]: p for p in read_csv(OUT / "susc_17c29_local_phenomenon_classification.csv")}
    g5 = read_csv(OUT / "susc_17c29_g5_phenomenon_evaluation.csv")
    for row in g5:
        if phenomena[row["local_observed_event_candidate_id"]]["mixed_or_ambiguous"] == "true":
            assert row["G5_separacao_fenomeno"] == "false"
    assert any(p["mixed_or_ambiguous"] == "true" for p in phenomena.values())


def test_sensor_chirps_e_noticia_nao_viram_ground_reference():
    gr = read_csv(OUT / "susc_17c29_ground_reference_candidate_evaluation.csv")
    assert {row["can_be_ground_truth"] for row in gr} == {"false"}
    assert {row["can_be_training_label"] for row in gr} == {"false"}
    assert {row["can_be_ground_reference_candidate"] for row in gr} == {"false"}
    leakage = read_csv(OUT / "susc_17c29_no_leakage_audit.csv")
    assert {r["uses_sensor_as_event_observation"] for r in leakage} == {"false"}
    assert {r["uses_chirps_as_event_reference"] for r in leakage} == {"false"}
    assert {r["uses_news_as_ground_reference_without_official_support"] for r in leakage} == {"false"}
    assert {r["invented_coordinate"] for r in leakage} == {"false"}
    assert {r["uses_mixed_phenomenon_as_g5"] for r in leakage} == {"false"}
    assert {r["passes_no_leakage"] for r in leakage} == {"true"}


def test_score_guardrails_e_17b_bloqueado():
    summary = json.loads((OUT / "susc_17c29_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["minimum_success_achieved"] is True
    assert summary["ground_truth_created"] is False
    assert summary["training_labels_created"] is False
    assert summary["score_v6_changed"] is False
    assert summary["score_v7_created"] is False
    assert summary["eligible_for_17b_now"] is False
    assert summary["eligible_for_score_v7"] is False
    assert summary["G4_true_count"] == 0
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
    blockers = read_csv(OUT / "susc_17c29_promotion_blockers.csv")
    assert any(row["blocks_17b"] == "true" for row in blockers)
