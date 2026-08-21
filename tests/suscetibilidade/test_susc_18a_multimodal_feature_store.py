"""Testes do SUSC-18A - feature store multimodal escalavel patch-level."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
DAT = ROOT / "datasets" / "suscetibilidade"

REGISTRY = OUT / "susc_18a_unified_patch_registry.csv"
CONTRACT = OUT / "susc_18a_multimodal_feature_family_contract.json"
INVENTORY = OUT / "susc_18a_feature_source_inventory.csv"
S2 = OUT / "susc_18a_sentinel2_feature_store.csv"
CHIRPS = OUT / "susc_18a_chirps_feature_store.csv"
TERRAIN = OUT / "susc_18a_terrain_topography_feature_store.csv"
HYDRO = OUT / "susc_18a_hydrology_proximity_feature_store.csv"
LANDCOVER = OUT / "susc_18a_landcover_urban_feature_store.csv"
MULTIMODAL = OUT / "susc_18a_multimodal_patch_feature_store.csv"
AVAIL = OUT / "susc_18a_feature_availability_matrix.csv"
LINEAGE = OUT / "susc_18a_feature_lineage_manifest.csv"
QFLAGS = OUT / "susc_18a_quality_flags.csv"
NEXT_STAGE = OUT / "susc_18a_next_stage_readiness.csv"
INTEG = OUT / "susc_18a_score_integrity_audit.csv"
LEAKAGE = OUT / "susc_18a_no_leakage_audit.csv"
SUMMARY = OUT / "susc_18a_readiness_summary.json"
BLOCKERS = OUT / "susc_18a_promotion_blockers.csv"
REPORT = OUT / "SUSC_18A_MULTIMODAL_FEATURE_STORE_REPORT.md"

SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"

EXPECTED = [
    REGISTRY, CONTRACT, INVENTORY, S2, CHIRPS, TERRAIN, HYDRO, LANDCOVER, MULTIMODAL, AVAIL,
    LINEAGE, QFLAGS, NEXT_STAGE, INTEG, LEAKAGE, SUMMARY, BLOCKERS, REPORT,
]


def rc(path):
    with path.open("r", encoding="utf-8", newline="") as h:
        return list(csv.DictReader(h))


def rj(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_script(name, *args):
    return subprocess.run([sys.executable, str(SCRIPTS / name), *args],
                          cwd=ROOT, text=True, capture_output=True, timeout=600, check=False)


def test_build_offline_byte_identico_e_valida():
    r = run_script("build_susc_18a_multimodal_feature_store.py")
    assert r.returncode == 0, r.stderr + r.stdout
    first = {p: p.read_bytes() for p in EXPECTED}
    r = run_script("build_susc_18a_multimodal_feature_store.py")
    assert r.returncode == 0, r.stderr + r.stdout
    assert {p: p.read_bytes() for p in EXPECTED} == first
    r = run_script("validate_susc_18a_multimodal_feature_store.py")
    assert r.returncode == 0, r.stderr + r.stdout


def test_contagens_minimas_de_patches():
    reg = rc(REGISTRY)
    official = [r for r in reg if r["patch_family"] == "official_population_patch"]
    canary = [r for r in reg if r["patch_family"] == "event_anchored_canary_patch"]
    control = [r for r in reg if r["patch_family"] == "spatial_mismatch_control_candidate"]
    assert len(official) >= 100
    assert len(canary) >= 11
    assert len(control) >= 5
    assert len(reg) >= 116


def test_contrato_de_familias_existe():
    c = rj(CONTRACT)
    assert c["feature_family_count"] >= 6
    for fam in ("sentinel2_spectral", "chirps_rainfall", "terrain_topography",
                "hydrology_proximity", "landcover_urban", "transparent_review_components"):
        assert fam in c["families"]


def test_feature_stores_existem():
    for p in (S2, CHIRPS, TERRAIN, HYDRO, LANDCOVER):
        assert p.exists()
        assert len(rc(p)) >= 116


def test_multimodal_store_existe_e_completo():
    mm = rc(MULTIMODAL)
    assert len(mm) >= 116
    for r in mm:
        assert r["review_only"] == "true"
        assert r["ground_truth"] == "false"
        assert r["training_label"] == "false"


def test_availability_matrix_existe():
    assert len(rc(AVAIL)) >= 116


def test_lineage_manifest_existe():
    lin = rc(LINEAGE)
    assert len(lin) >= 116
    for r in lin:
        assert r["is_review_only"] == "true"


def test_quality_flags_existem():
    assert len(rc(QFLAGS)) >= 116


def test_flow_acc_nao_e_equivalente():
    s = rj(SUMMARY)
    assert s["flow_acc_treated_as_equivalent"] is False
    for r in rc(TERRAIN):
        assert r["flow_acc_is_official_equivalent"] == "false"
        if r["patch_family"] == "event_anchored_canary_patch" and r["flow_acc_value"] != "not_available":
            assert r["flow_acc_replacement_contract_required"] == "true"


def test_ocorrencia_nao_entra_como_feature_causal():
    for r in rc(LEAKAGE):
        assert r["uses_occurrence_as_causal_feature"] == "false"
    for r in rc(MULTIMODAL):
        assert r["absence_of_occurrence_is_true_negative"] == "false"


def test_controle_nao_vira_negativo_verdadeiro():
    for r in rc(LEAKAGE):
        assert r["uses_control_as_negative_truth"] == "false"
        assert r["uses_absence_as_real_absence"] == "false"
        assert r["uses_canary_as_positive_label"] == "false"


def test_score_v6_oficial_intacto():
    r = subprocess.run(["git", "diff", "--name-only", "--", "datasets/suscetibilidade/susc_score_v6_candidate_by_patch_v1.csv"],
                       cwd=ROOT, text=True, capture_output=True, check=False)
    assert r.stdout.strip() == ""
    s = rj(SUMMARY)
    assert s["score_v6_original_file_changed"] is False


def test_score_v7_inexistente():
    assert not SCORE_V7.exists()
    assert rj(SUMMARY)["score_v7_created"] is False


def test_ground_truth_e_treino_nao_criados():
    s = rj(SUMMARY)
    assert s["ground_truth_created"] is False
    assert s["training_labels_created"] is False
    assert s["ground_truth"] is False
    assert s["trainable"] is False


def test_17b_nao_liberado():
    s = rj(SUMMARY)
    assert s["eligible_for_17b_now"] is False
    for r in rc(LEAKAGE):
        assert r["unlocks_17b"] == "false"
    ready = {r["readiness_target"]: r["is_ready"] for r in rc(NEXT_STAGE)}
    assert ready["blocked_for_17b"] == "false"


def test_no_leakage_passa():
    leak = rc(LEAKAGE)
    assert len(leak) >= 116
    for r in leak:
        assert r["passes_no_leakage"] == "true"
        assert r["hides_missing_feature"] == "false"
        assert r["imputes_missing_feature"] == "false"


def test_missingness_explicita_controles():
    # controles nao tem terrain/hydrology/landcover -> registrado, nao imputado
    mm = {r["source_patch_id"]: r for r in rc(MULTIMODAL)}
    reg = [r for r in rc(REGISTRY) if r["patch_family"] == "spatial_mismatch_control_candidate"]
    for r in reg:
        row = mm[r["source_patch_id"]]
        assert row["terrain_available"] == "false"
        assert row["HAND_value"] == "not_available"
        assert "terrain_topography" in row["missing_feature_families"]


def test_minimum_success_achieved():
    assert rj(SUMMARY)["minimum_success_achieved"] is True


def test_promotion_blockers_presentes():
    types = {r["blocker_type"] for r in rc(BLOCKERS)}
    for expected in ("not_ground_truth", "not_training_ready", "not_score_v7", "flow_acc_not_equivalent",
                     "documentary_anchor_not_label", "control_not_negative_truth", "17b_blocked",
                     "feature_missingness_explicit"):
        assert expected in types
