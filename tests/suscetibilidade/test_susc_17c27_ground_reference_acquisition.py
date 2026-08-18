"""Testes do SUSC-17C27 - aquisicao dirigida de artefatos de evento observado para G4/G5."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
DAT = ROOT / "datasets" / "suscetibilidade"
ARTIFACT_DIR = OUT / "susc_17c27_source_artifacts"
MAX_ARTIFACT = 500_000

EXPECTED = [
    OUT / "SUSC_17C27_AQUISICAO_DIRIGIDA_ARTEFATOS_EVENTO_G4_G5_REPORT.md",
    OUT / "susc_17c27_acquisition_plan.csv",
    OUT / "susc_17c27_source_acquisition_attempts.csv",
    OUT / "susc_17c27_source_artifact_manifest.csv",
    OUT / "susc_17c27_parsed_artifact_index.csv",
    OUT / "susc_17c27_observed_event_candidates.csv",
    OUT / "susc_17c27_candidate_location_resolution.csv",
    OUT / "susc_17c27_phenomenon_classification.csv",
    OUT / "susc_17c27_g4_spatial_link_evaluation.csv",
    OUT / "susc_17c27_g5_phenomenon_evaluation.csv",
    OUT / "susc_17c27_ground_reference_candidate_evaluation.csv",
    OUT / "susc_17c27_evidence_graph_update_nodes.csv",
    OUT / "susc_17c27_evidence_graph_update_edges.csv",
    OUT / "susc_17c27_source_limitations.csv",
    OUT / "susc_17c27_no_leakage_audit.csv",
    OUT / "susc_17c27_gate_evaluation_matrix.csv",
    OUT / "susc_17c27_readiness_summary.json",
    OUT / "susc_17c27_promotion_blockers.csv",
]


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_script(name: str, *args: str):
    env = {}
    import os
    env = os.environ.copy()
    env.pop("SUSC_17C27_ALLOW_NETWORK", None)
    env.pop("SUSC_17C27_ALLOW_PUBLIC_DOWNLOAD", None)
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *args],
        cwd=ROOT, env=env, text=True, capture_output=True, timeout=600, check=False,
    )


def test_build_offline_byte_identico_e_valida():
    result = run_script("build_susc_17c27_ground_reference_acquisition.py")
    assert result.returncode == 0, result.stderr + result.stdout
    first = {path: path.read_bytes() for path in EXPECTED}
    result = run_script("build_susc_17c27_ground_reference_acquisition.py")
    assert result.returncode == 0, result.stderr + result.stdout
    assert {path: path.read_bytes() for path in EXPECTED} == first
    result = run_script("validate_susc_17c27_ground_reference_acquisition.py")
    assert result.returncode == 0, result.stderr + result.stdout


def test_15_query_packages_consumidos_e_15_tentativas():
    plan = read_csv(OUT / "susc_17c27_acquisition_plan.csv")
    assert len(plan) >= 15
    attempts = read_csv(OUT / "susc_17c27_source_acquisition_attempts.csv")
    assert len([a for a in attempts if a["attempted"] == "true"]) >= 15
    summary = json.loads((OUT / "susc_17c27_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["source_query_packages_consumed_count"] >= 15
    assert summary["source_acquisition_attempts_count"] >= 15


def test_pelo_menos_3_artefatos_reais_com_hash():
    manifests = read_csv(OUT / "susc_17c27_source_artifact_manifest.csv")
    assert len(manifests) >= 3
    for row in manifests:
        path = ROOT / row["artifact_local_path"]
        assert path.exists(), row["artifact_local_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        assert int(row["size_bytes"]) <= MAX_ARTIFACT
        assert row["artifact_type"] in ("html", "pdf", "csv", "json", "txt")
    if ARTIFACT_DIR.exists():
        for path in ARTIFACT_DIR.glob("**/*"):
            if path.is_file():
                assert path.suffix.lower() not in (".tif", ".nc", ".zip", ".gz", ".npz")


def test_parsing_observed_candidate_e_g4_g5():
    parsed = read_csv(OUT / "susc_17c27_parsed_artifact_index.csv")
    assert len([p for p in parsed if p["parse_success"] == "true"]) >= 3
    observed = read_csv(OUT / "susc_17c27_observed_event_candidates.csv")
    assert len(observed) >= 1
    g4 = read_csv(OUT / "susc_17c27_g4_spatial_link_evaluation.csv")
    g5 = read_csv(OUT / "susc_17c27_g5_phenomenon_evaluation.csv")
    assert len(g4) + len(g5) >= 1
    gr = read_csv(OUT / "susc_17c27_ground_reference_candidate_evaluation.csv")
    assert len(gr) >= 1


def test_noticia_nao_vira_ground_reference_e_sensor_nao_vira_evento():
    gr = read_csv(OUT / "susc_17c27_ground_reference_candidate_evaluation.csv")
    observed = {o["observed_event_candidate_id"]: o for o in read_csv(OUT / "susc_17c27_observed_event_candidates.csv")}
    for row in gr:
        if row["can_be_ground_reference_candidate"] == "true":
            assert observed[row["observed_event_candidate_id"]]["officiality_level"] == "official"
    leakage = read_csv(OUT / "susc_17c27_no_leakage_audit.csv")
    assert {r["uses_sensor_as_event_observation"] for r in leakage} == {"false"}
    assert {r["uses_chirps_as_event_reference"] for r in leakage} == {"false"}
    assert {r["uses_news_as_ground_reference_without_official_support"] for r in leakage} == {"false"}
    assert {r["passes_no_leakage"] for r in leakage} == {"true"}


def test_g4_g5_false_e_ground_truth_nao_criado():
    gr = read_csv(OUT / "susc_17c27_ground_reference_candidate_evaluation.csv")
    assert {r["can_be_ground_truth"] for r in gr} == {"false"}
    assert {r["can_be_training_label"] for r in gr} == {"false"}
    summary = json.loads((OUT / "susc_17c27_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["ground_truth_created"] is False
    assert summary["training_labels_created"] is False
    assert summary["ground_reference_candidates_created_count"] == 0


def test_17b_inelegivel_sem_accepted_ground_reference():
    summary = json.loads((OUT / "susc_17c27_readiness_summary.json").read_text(encoding="utf-8"))
    gr = read_csv(OUT / "susc_17c27_ground_reference_candidate_evaluation.csv")
    accepted = [r for r in gr if r["can_be_ground_reference_candidate"] == "true"]
    assert summary["accepted_ground_reference_candidate_count"] == len(accepted)
    assert summary["eligible_for_17b_now"] == (len(accepted) > 0)
    if not accepted:
        assert summary["eligible_for_17b_now"] is False
    assert summary["minimum_success_achieved"] is True


def test_score_guardrails_intactos():
    summary = json.loads((OUT / "susc_17c27_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["score_v6_changed"] is False
    assert summary["score_v7_created"] is False
    assert summary["official_patch_created"] is False
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
    blockers = read_csv(OUT / "susc_17c27_promotion_blockers.csv")
    assert any(row["blocker_type"] == "17b_blocked_until_accepted_ground_reference" for row in blockers)
