"""
SUSC-10A score v6 candidate tests.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCORE = ROOT / "datasets" / "suscetibilidade" / "susc_score_v6_candidate_by_patch_v1.csv"
MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_score_v6_candidate_manifest_v1.json"
REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_10A_score_v6_candidate_methodology_report.md"
SUMMARY = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_10A_score_v6_candidate_summary.csv"
TOP = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_10A_score_v6_candidate_top_patches.csv"
CONTRIB = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_10A_score_v6_candidate_feature_contributions.csv"

FLAGGED_OUT = {"curvature_laplacian_mean", "rain_3d_7d_ratio", "flow_acc_log_p75", "water_occurrence_patch"}


def _csv(p):
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_artifacts_exist():
    for p in (SCORE, MANIFEST, REPORT, SUMMARY, TOP, CONTRIB):
        assert p.exists(), p


def test_score_rows_300():
    rows = _csv(SCORE)
    assert len(rows) == 300


def test_score_in_unit_interval():
    for r in _csv(SCORE):
        v = float(r["susc_score_v6_candidate"])
        assert 0.0 <= v <= 1.0


def test_no_label_or_score_v5_as_feature():
    m = _manifest()
    assert m["uses_label_as_feature"] is False
    assert m["uses_score_v5_as_feature"] is False
    assert m["uses_proxy_target"] is False
    for f in m["features_used"]:
        assert not f.startswith(("score_", "label_", "proxy_v5_")), f


def test_flagged_features_excluded():
    m = _manifest()
    for f in FLAGGED_OUT:
        assert f not in m["features_used"]


def test_no_training_no_model():
    m = _manifest()
    assert m["allowed_for_training"] is False
    assert m["trained_model"] is False
    assert m["model_persisted"] is False


def test_no_ground_truth():
    m = _manifest()
    assert m["can_be_used_as_ground_truth"] is False
    for r in _csv(SCORE):
        assert r["can_be_used_as_ground_truth"] == "false"


def test_classes_present():
    classes = {r["susc_class_v6_candidate"] for r in _csv(SCORE)}
    assert classes <= {"low", "medium", "high"}
    assert "high" in classes and "low" in classes


def test_report_mandatory_statement():
    assert "Ele não é modelo supervisionado, não usa ground truth" in REPORT.read_text(encoding="utf-8")
