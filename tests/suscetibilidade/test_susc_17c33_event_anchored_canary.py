"""Testes do SUSC-17C33 - event-anchored canary patch set e validacao observacional review-only."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
DAT = ROOT / "datasets" / "suscetibilidade"

REGISTRY = OUT / "susc_17c33_event_anchored_canary_patch_registry.csv"
GEOJSON = OUT / "susc_17c33_event_anchored_patch_geojson.geojson"
CONTROL = OUT / "susc_17c33_original_patch_mismatch_control.csv"
BINDING = OUT / "susc_17c33_pre_event_feature_binding.csv"
SCORE = OUT / "susc_17c33_score_v6_on_canary_patches.csv"
CONTRAST = OUT / "susc_17c33_observational_contrast_matrix.csv"
SUMMARY = OUT / "susc_17c33_review_only_validation_summary.json"

EXPECTED = [
    OUT / "SUSC_17C33_EVENT_ANCHORED_CANARY_PATCH_REPORT.md",
    REGISTRY, GEOJSON, CONTROL, BINDING, SCORE, CONTRAST, SUMMARY,
]


def rc(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as h:
        return list(csv.DictReader(h))


def run_script(name: str, *args: str):
    return subprocess.run([sys.executable, str(SCRIPTS / name), *args],
                          cwd=ROOT, text=True, capture_output=True, timeout=600, check=False)


def test_build_offline_byte_identico_e_valida():
    r = run_script("build_susc_17c33_event_anchored_canary.py")
    assert r.returncode == 0, r.stderr + r.stdout
    first = {p: p.read_bytes() for p in EXPECTED}
    r = run_script("build_susc_17c33_event_anchored_canary.py")
    assert r.returncode == 0, r.stderr + r.stdout
    assert {p: p.read_bytes() for p in EXPECTED} == first
    r = run_script("validate_susc_17c33_event_anchored_canary.py")
    assert r.returncode == 0, r.stderr + r.stdout


def test_canary_namespace_e_nao_substitui_original():
    reg = rc(REGISTRY)
    assert len(reg) >= 3
    grid_ids = {r["candidate_patch_id"] for r in rc(OUT / "susc_17c6_candidate_patch_grid.csv")}
    for r in reg:
        assert r["namespace"] == "event_anchored_canary_patch"
        assert r["event_anchored_canary_patch_id"] not in grid_ids
        assert r["replaces_original_patch"] == "false"


def test_canary_nao_e_ground_truth_nem_treino_e_hidrologico():
    for r in rc(REGISTRY):
        assert r["is_ground_truth"] == "false"
        assert r["is_training"] == "false"
        assert r["hydrological_signal"] == "true"
        assert int(r["uncertainty_m"]) >= 300  # centroide/ponto sem incerteza proibido


def test_original_patch_preservado_como_controle():
    control = rc(CONTROL)
    grid_ids = {r["candidate_patch_id"] for r in rc(OUT / "susc_17c6_candidate_patch_grid.csv")}
    control_ids = {r["original_patch_id"] for r in control}
    assert control_ids == grid_ids  # todos os originais viram controle, nenhum sumiu
    for r in control:
        assert r["control_role"] == "spatial_mismatch_control_candidate"
        assert r["replaces_by_canary"] == "false"


def test_ausencia_de_ocorrencia_nao_e_negativo_verdadeiro():
    for r in rc(CONTROL):
        assert r["absence_of_occurrence_is_true_negative"] == "false"


def test_score_v6_nao_recomputado_nem_alterado_e_sem_v7():
    for r in rc(SCORE):
        assert r["score_v6_recomputed"] == "false"
        assert r["score_v6_source_dataset_altered"] == "false"
        assert r["score_v7_created"] == "false"
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["score_v6_changed"] is False
    assert s["score_v7_created"] is False
    assert not (DAT / "susc_score_v7_candidate_by_patch_v1.csv").exists()
    r = subprocess.run(["git", "diff", "--name-only", "--", "datasets/suscetibilidade/susc_score_v6_candidate_by_patch_v1.csv"],
                       cwd=ROOT, text=True, capture_output=True, check=False)
    assert r.stdout.strip() == ""


def test_features_locais_nao_disponiveis_chirps_herdado():
    binding = rc(BINDING)
    for r in binding:
        assert r["ndvi_preevent"] == "not_available_requires_extraction"
        assert r["hand_m"] == "not_available_requires_extraction"
        assert r["chirps_source"] == "inherited_region_level_17c25_same_event"
        assert r["chirps_discriminates_local"] == "false"


def test_contraste_observacional_ancora_ocorrencia():
    contrast = rc(CONTRAST)
    assert len(contrast) >= 4
    controls = [r for r in contrast if r["entity_type"] == "spatial_mismatch_control_candidate"]
    canaries = [r for r in contrast if r["entity_type"] == "event_anchored_canary_patch"]
    assert controls and canaries
    for r in controls:
        assert r["has_official_hydrological_occurrence_within_buffer"] == "false"
        assert r["absence_is_true_negative"] == "false"
    for r in canaries:
        assert r["has_official_hydrological_occurrence_within_buffer"] == "true"


def test_geojson_valido():
    gj = json.loads(GEOJSON.read_text(encoding="utf-8"))
    assert gj["type"] == "FeatureCollection"
    assert len(gj["features"]) >= 3


def test_summary_guardrails_e_17b_bloqueado():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["original_patch_overwritten"] is False
    assert s["canary_is_ground_truth"] is False
    assert s["canary_is_training"] is False
    assert s["eligible_for_17b_now"] is False
    assert s["score_v6_computable_on_canary_count"] == 0
    assert s["minimum_success_achieved"] is True
