"""Testes do SUSC-17C32 - geocodificacao oficial de ocorrencias e vinculo patch-buffer para G4d.

Offline e deterministico: usa o ledger de geocodificacao ja commitado.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
DAT = ROOT / "datasets" / "suscetibilidade"

TARGETS = OUT / "susc_17c32_hydrological_occurrence_targets.csv"
ATTEMPTS = OUT / "susc_17c32_geocoding_attempts.csv"
POINTS = OUT / "susc_17c32_geocoded_point_registry.csv"
DIST = OUT / "susc_17c32_patch_buffer_distance_evaluation.csv"
G4D = OUT / "susc_17c32_g4d_patch_buffer_link_evaluation.csv"
G4F = OUT / "susc_17c32_g4_full_evaluation.csv"
GRR = OUT / "susc_17c32_ground_reference_readiness_evaluation.csv"
LEAKAGE = OUT / "susc_17c32_no_leakage_audit.csv"
SUMMARY = OUT / "susc_17c32_readiness_summary.json"
BLOCKERS = OUT / "susc_17c32_promotion_blockers.csv"

EXPECTED = [
    OUT / "SUSC_17C32_GEOCODIFICACAO_OCORRENCIAS_G4D_REPORT.md",
    TARGETS, ATTEMPTS, POINTS, DIST, G4D, G4F, GRR,
    OUT / "susc_17c32_subgate_update_matrix.csv",
    OUT / "susc_17c32_gate_transition_ledger.csv", LEAKAGE, SUMMARY, BLOCKERS,
]


def rc(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as h:
        return list(csv.DictReader(h))


def run_script(name: str, *args: str):
    env = os.environ.copy()
    for v in ["SUSC_17C32_ALLOW_NETWORK", "SUSC_17C32_ALLOW_GEOCODING"]:
        env.pop(v, None)
    return subprocess.run([sys.executable, str(SCRIPTS / name), *args],
                          cwd=ROOT, env=env, text=True, capture_output=True, timeout=600, check=False)


def test_build_offline_byte_identico_e_valida():
    r = run_script("build_susc_17c32_geocoding_g4d.py")
    assert r.returncode == 0, r.stderr + r.stdout
    first = {p: p.read_bytes() for p in EXPECTED}
    r = run_script("build_susc_17c32_geocoding_g4d.py")
    assert r.returncode == 0, r.stderr + r.stdout
    assert {p: p.read_bytes() for p in EXPECTED} == first
    r = run_script("validate_susc_17c32_geocoding_g4d.py")
    assert r.returncode == 0, r.stderr + r.stdout


def test_alvos_geocoding_e_pontos_minimos():
    assert len(rc(TARGETS)) >= 3
    attempts = rc(ATTEMPTS)
    assert len([a for a in attempts if a["network_enabled"] == "true"]) >= 3
    assert len(rc(POINTS)) >= 1
    assert len(rc(DIST)) >= 1
    assert len(rc(G4D)) >= 1


def test_geocoder_nao_oficial_e_incerteza_alta():
    for p in rc(POINTS):
        assert p["geocoder_is_official"] == "false"
        assert p["house_number_status"] == "anonymized_lgpd"
        assert p["precision_level"] == "street_level"
        assert int(p["uncertainty_m"]) >= 300


def test_bairro_osm_area_risco_nao_abrem_g4d():
    for r in rc(LEAKAGE):
        assert r["uses_bairro_centroid_as_g4d"] == "false"
        assert r["uses_osm_alone_to_open_g4d"] == "false"
        assert r["uses_risk_sector_to_open_g4d"] == "false"
        assert r["invented_coordinate"] == "false"
        assert r["passes_no_leakage"] == "true"


def test_g4d_so_abre_com_ponto_oficial_e_dentro_do_buffer():
    for r in rc(G4D):
        if r["G4d_patch_buffer_link_confirmed"] == "true":
            assert r["geometry_is_official_or_controlled_point"] == "true"
            assert r["within_patch_or_acceptable_buffer"] == "true"


def test_g4_full_requer_g4c_evento_e_g4d():
    for r in rc(G4F):
        if r["G4_full_spatial_patch_link"] == "true":
            assert r["G4c_event_geometry_present"] == "true"
            assert r["G4d_patch_buffer_link_confirmed"] == "true"


def test_resultado_honesto_g4d_bloqueado_neste_marco():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    # dados reais: ocorrencias hidrologicas distantes do patch + street-level -> G4d bloqueado
    assert s["G4d_true_count"] == 0
    assert s["G4_full_true_count"] == 0
    assert s["accepted_ground_reference_candidate_review_only_count"] == 0
    assert s["points_within_patch_buffer_count"] == 0
    assert s["result_class"] == "B_honest_block_spatial_mismatch"


def test_nao_redesenha_17c31_g5d_herdado():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    c31 = json.loads((OUT / "susc_17c31_readiness_summary.json").read_text(encoding="utf-8"))
    assert s["G5d_true_count_inherited_17c31"] == c31["G5d_true_count"]
    assert s["G5_full_true_count_inherited_17c31"] == c31["G5_full_true_count"]


def test_ground_truth_treino_nao_criados():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["ground_truth_created"] is False
    assert s["training_labels_created"] is False
    for r in rc(GRR):
        assert r["can_be_ground_truth"] == "false"
        assert r["can_be_training_label"] == "false"


def test_score_v6_intacto_e_v7_inexistente():
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


def test_17b_bloqueado_sem_ground_reference():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["eligible_for_17b_now"] is False


def test_geocoding_bloqueia_sem_env():
    r = run_script("susc_17c32_geocoding_g4d.py", "run-controlled-geocoding")
    assert "BLOCKED" in (r.stdout + r.stderr)
