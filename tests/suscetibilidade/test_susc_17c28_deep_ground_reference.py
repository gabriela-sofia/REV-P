"""Testes do SUSC-17C28 - aquisicao profunda oficial para G4/G5."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
DAT = ROOT / "datasets" / "suscetibilidade"
ARTIFACT_DIR = OUT / "susc_17c28_source_artifacts"
MAX_ARTIFACT = 500_000

EXPECTED = [
    OUT / "SUSC_17C28_AQUISICAO_PROFUNDA_ARTEFATOS_OFICIAIS_G4_G5_REPORT.md",
    OUT / "susc_17c28_expanded_search_plan.csv",
    OUT / "susc_17c28_deep_source_acquisition_attempts.csv",
    OUT / "susc_17c28_followed_link_registry.csv",
    OUT / "susc_17c28_deep_source_artifact_manifest.csv",
    OUT / "susc_17c28_deep_parsed_artifact_index.csv",
    OUT / "susc_17c28_specific_observed_event_candidates.csv",
    OUT / "susc_17c28_location_resolution.csv",
    OUT / "susc_17c28_phenomenon_classification.csv",
    OUT / "susc_17c28_g4_spatial_link_evaluation.csv",
    OUT / "susc_17c28_g5_phenomenon_evaluation.csv",
    OUT / "susc_17c28_ground_reference_candidate_evaluation.csv",
    OUT / "susc_17c28_official_artifact_scorecard.csv",
    OUT / "susc_17c28_evidence_graph_update_nodes.csv",
    OUT / "susc_17c28_evidence_graph_update_edges.csv",
    OUT / "susc_17c28_no_leakage_audit.csv",
    OUT / "susc_17c28_gate_evaluation_matrix.csv",
    OUT / "susc_17c28_readiness_summary.json",
    OUT / "susc_17c28_promotion_blockers.csv",
]


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_script(name: str, *args: str):
    env = os.environ.copy()
    env.pop("SUSC_17C28_ALLOW_NETWORK", None)
    env.pop("SUSC_17C28_ALLOW_PUBLIC_DOWNLOAD", None)
    env.pop("SUSC_17C28_ALLOW_DEEP_SEARCH", None)
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *args],
        cwd=ROOT, env=env, text=True, capture_output=True, timeout=600, check=False,
    )


def test_build_offline_byte_identico_e_valida():
    result = run_script("build_susc_17c28_deep_ground_reference.py")
    assert result.returncode == 0, result.stderr + result.stdout
    first = {path: path.read_bytes() for path in EXPECTED}
    result = run_script("build_susc_17c28_deep_ground_reference.py")
    assert result.returncode == 0, result.stderr + result.stdout
    assert {path: path.read_bytes() for path in EXPECTED} == first
    result = run_script("validate_susc_17c28_deep_ground_reference.py")
    assert result.returncode == 0, result.stderr + result.stdout


def test_plano_e_tentativas_profundas_minimas():
    plan = read_csv(OUT / "susc_17c28_expanded_search_plan.csv")
    assert len(plan) >= 30
    attempts = read_csv(OUT / "susc_17c28_deep_source_acquisition_attempts.csv")
    assert len([a for a in attempts if a["network_enabled"] == "true"]) >= 30
    summary = json.loads((OUT / "susc_17c28_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["deep_source_search_attempts_count"] >= 30


def test_artefatos_event_specific_oficiais_com_hash():
    manifests = read_csv(OUT / "susc_17c28_deep_source_artifact_manifest.csv")
    official = [
        row for row in manifests
        if row["event_specific"] == "true"
        and row["officiality_level"] in {"official", "official_institutional", "official_institutional_public_agency"}
    ]
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


def test_homepage_generica_nao_conta_como_event_specific():
    manifests = read_csv(OUT / "susc_17c28_deep_source_artifact_manifest.csv")
    homepage_rows = [row for row in manifests if row["source_url"].rstrip("/").endswith("agenciabrasil.ebc.com.br")]
    assert homepage_rows
    assert {row["event_specific"] for row in homepage_rows} == {"false"}


def test_parse_candidates_e_g4_g5_minimos():
    parsed = read_csv(OUT / "susc_17c28_deep_parsed_artifact_index.csv")
    assert len([p for p in parsed if p["parse_success"] == "true"]) >= 3
    observed = read_csv(OUT / "susc_17c28_specific_observed_event_candidates.csv")
    assert len(observed) >= 3
    locations = read_csv(OUT / "susc_17c28_location_resolution.csv")
    phenomena = read_csv(OUT / "susc_17c28_phenomenon_classification.csv")
    assert len(locations) >= 3
    assert len(phenomena) >= 3
    g4 = read_csv(OUT / "susc_17c28_g4_spatial_link_evaluation.csv")
    g5 = read_csv(OUT / "susc_17c28_g5_phenomenon_evaluation.csv")
    assert len(g4) + len(g5) >= 3
    assert {row["G4_vinculo_espacial_evento"] for row in g4} == {"false"}


def test_triagem_sensor_e_noticia_comercial_nao_viram_ground_reference():
    gr = read_csv(OUT / "susc_17c28_ground_reference_candidate_evaluation.csv")
    assert {row["can_be_ground_truth"] for row in gr} == {"false"}
    assert {row["can_be_training_label"] for row in gr} == {"false"}
    assert {row["can_be_ground_reference_candidate"] for row in gr} == {"false"}
    leakage = read_csv(OUT / "susc_17c28_no_leakage_audit.csv")
    assert {r["uses_sensor_as_event_observation"] for r in leakage} == {"false"}
    assert {r["uses_chirps_as_event_reference"] for r in leakage} == {"false"}
    assert {r["uses_news_as_ground_reference_without_official_support"] for r in leakage} == {"false"}
    assert {r["passes_no_leakage"] for r in leakage} == {"true"}


def test_score_guardrails_e_17b_bloqueado():
    summary = json.loads((OUT / "susc_17c28_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["minimum_success_achieved"] is True
    assert summary["ground_truth_created"] is False
    assert summary["training_labels_created"] is False
    assert summary["score_v6_changed"] is False
    assert summary["score_v7_created"] is False
    assert summary["eligible_for_17b_now"] is False
    assert summary["eligible_for_score_v7"] is False
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
    blockers = read_csv(OUT / "susc_17c28_promotion_blockers.csv")
    assert any(row["blocks_17b"] == "true" for row in blockers)
