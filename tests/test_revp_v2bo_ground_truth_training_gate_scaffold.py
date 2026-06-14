"""Tests for revp_v2bo_ground_truth_training_gate_scaffold.py.

Covers: guardrails, scaffold rows with empty labels, the negative policy that
refuses absence-based negatives, the training gate staying blocked, fail-closed
on missing inputs, outputs created, and no private paths in the script.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "multimodal"
sys.path.insert(0, str(SCRIPTS_DIR))

from revp_v2bo_ground_truth_training_gate_scaffold import (  # noqa: E402
    METHODOLOGICAL_GUARDRAILS,
    SCAFFOLD_FIELDS,
    build_artifacts,
    build_scaffold_rows,
    negative_policy,
    label_policy,
    training_gate,
    write_artifacts,
)


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _feature_table(tmp_path: Path, patches) -> Path:
    rows = [{"canonical_patch_id": p, "region": r} for p, r in patches]
    path = tmp_path / "ft.csv"
    _write_csv(path, rows, ["canonical_patch_id", "region"])
    return path


# --------------------------------------------------------------------------- #
# Guardrails
# --------------------------------------------------------------------------- #

class TestGuardrails:
    def test_review_only(self):
        assert METHODOLOGICAL_GUARDRAILS["review_only"] is True

    def test_labels_created_false(self):
        assert METHODOLOGICAL_GUARDRAILS["labels_created"] is False

    def test_negative_from_absence_false(self):
        assert METHODOLOGICAL_GUARDRAILS["negative_from_absence"] is False

    def test_unknown_stays_unknown(self):
        assert METHODOLOGICAL_GUARDRAILS["unknown_stays_unknown"] is True

    def test_multimodal_disabled(self):
        assert METHODOLOGICAL_GUARDRAILS["multimodal_execution_enabled"] is False
        assert METHODOLOGICAL_GUARDRAILS["multimodal_training_enabled"] is False


# --------------------------------------------------------------------------- #
# Scaffold rows: empty labels, human review required
# --------------------------------------------------------------------------- #

class TestScaffoldRows:
    def _rows(self):
        patches = [
            {"canonical_patch_id": "REC_1", "region": "Recife"},
            {"canonical_patch_id": "CUR_1", "region": "Curitiba"},
        ]
        return build_scaffold_rows(patches)

    def test_one_row_per_patch(self):
        assert len(self._rows()) == 2

    def test_labels_empty(self):
        for r in self._rows():
            assert r["gt_patch_flood_observed"] == ""
            assert r["label_quality"] == "NO_LABEL"

    def test_training_blocked(self):
        for r in self._rows():
            assert r["allowed_for_training"] == "False"
            assert r["blocked_reason"] != ""

    def test_human_review_required(self):
        for r in self._rows():
            assert r["human_review_required"] == "True"

    def test_no_event_bound(self):
        for r in self._rows():
            assert r["candidate_event_id"] == ""

    def test_evidence_not_established(self):
        for r in self._rows():
            assert r["positive_evidence_status"] == "NOT_ESTABLISHED"
            assert r["negative_evidence_status"] == "NOT_ESTABLISHED"

    def test_dedup_patches(self):
        patches = [
            {"canonical_patch_id": "REC_1", "region": "Recife"},
            {"canonical_patch_id": "REC_1", "region": "Recife"},
        ]
        assert len(build_scaffold_rows(patches)) == 1

    def test_skips_blank_patch_id(self):
        patches = [{"canonical_patch_id": "", "region": "Recife"}]
        assert build_scaffold_rows(patches) == []


# --------------------------------------------------------------------------- #
# Negative policy refuses absence-based negatives
# --------------------------------------------------------------------------- #

class TestNegativePolicy:
    def test_zero_formal_negatives(self):
        assert negative_policy()["formal_negative_count"] == 0

    def test_negative_from_absence_false(self):
        assert negative_policy()["negative_from_absence"] is False

    def test_no_allowed_negative_types_now(self):
        assert negative_policy()["allowed_negative_types_now"] == []

    def test_core_rules_enforced(self):
        rules = {r["rule"]: r["enforced"] for r in negative_policy()["rules"]}
        assert rules["absence_of_evidence_is_not_negative"] is True
        assert rules["pseudo_absence_is_not_formal_negative"] is True
        assert rules["random_background_is_not_formal_negative"] is True
        assert rules["distance_from_anchor_is_not_formal_negative"] is True
        assert rules["unknown_stays_unknown"] is True


# --------------------------------------------------------------------------- #
# Label policy creates no label
# --------------------------------------------------------------------------- #

class TestLabelPolicy:
    def test_labels_created_false(self):
        assert label_policy()["labels_created"] is False

    def test_current_quality_no_label(self):
        assert label_policy()["current_label_quality_for_all_patches"] == "NO_LABEL"

    def test_embeddings_cannot_be_label(self):
        text = json.dumps(label_policy()).lower()
        assert "embedding" in text  # explicitly listed as not-a-label


# --------------------------------------------------------------------------- #
# Training gate stays blocked
# --------------------------------------------------------------------------- #

class TestTrainingGate:
    def test_gate_blocked(self):
        rows = build_scaffold_rows([{"canonical_patch_id": "P1", "region": "Recife"}])
        gate = training_gate(rows)
        assert gate["supervised_training_enabled"] is False
        assert gate["labels_created"] is False
        assert gate["formal_negative_count"] == 0
        assert gate["any_row_allowed_for_training"] is False

    def test_future_validation_is_grouped(self):
        rows = build_scaffold_rows([{"canonical_patch_id": "P1", "region": "Recife"}])
        gate = training_gate(rows)
        assert "group" in gate["future_validation_policy"]

    def test_future_baselines_are_light(self):
        rows = build_scaffold_rows([{"canonical_patch_id": "P1", "region": "Recife"}])
        gate = training_gate(rows)
        assert any("logistic" in b for b in gate["future_baselines_when_unblocked"])


# --------------------------------------------------------------------------- #
# Build artifacts / fail-closed / outputs
# --------------------------------------------------------------------------- #

class TestBuildArtifacts:
    def test_reads_feature_table(self, tmp_path):
        ft = _feature_table(tmp_path, [("REC_1", "Recife"), ("CUR_1", "Curitiba")])
        art = build_artifacts(ft, tmp_path / "missing.csv")
        assert art["summary"]["patch_source"] == "v2bn_feature_table"
        assert art["summary"]["candidate_patch_count"] == 2

    def test_fallback_to_manifest(self, tmp_path):
        manifest = tmp_path / "manifest.csv"
        _write_csv(manifest, [{"canonical_patch_id": "PET_1", "region": "Petropolis"}], ["canonical_patch_id", "region"])
        art = build_artifacts(tmp_path / "no_ft.csv", manifest)
        assert art["summary"]["patch_source"] == "v1fu_input_manifest"

    def test_empty_inputs_fail_closed(self, tmp_path):
        art = build_artifacts(tmp_path / "a.csv", tmp_path / "b.csv")
        assert art["summary"]["candidate_patch_count"] == 0
        assert art["summary"]["patch_source"] == "NONE"
        assert art["summary"]["qa_status"] == "PASS"

    def test_no_labels_created(self, tmp_path):
        ft = _feature_table(tmp_path, [("REC_1", "Recife")])
        art = build_artifacts(ft, tmp_path / "missing.csv")
        assert art["summary"]["labelled_patch_count"] == 0
        assert art["summary"]["training_allowed_count"] == 0


class TestOutputs:
    EXPECTED = [
        "gt_patch_registry_scaffold_v2bo.csv",
        "gt_label_policy_v2bo.json",
        "gt_negative_policy_v2bo.json",
        "gt_training_gate_v2bo.json",
        "gt_scaffold_qa_v2bo.csv",
        "gt_scaffold_report_v2bo.md",
    ]

    def _run(self, tmp_path):
        ft = _feature_table(tmp_path, [("REC_1", "Recife"), ("CUR_1", "Curitiba")])
        art = build_artifacts(ft, tmp_path / "missing.csv")
        out = tmp_path / "out"
        out.mkdir()
        write_artifacts(out, art)
        return out

    def test_all_files_created(self, tmp_path):
        out = self._run(tmp_path)
        for fname in self.EXPECTED:
            assert (out / fname).exists(), f"Missing {fname}"

    def test_scaffold_csv_schema(self, tmp_path):
        out = self._run(tmp_path)
        with (out / "gt_patch_registry_scaffold_v2bo.csv").open(encoding="utf-8") as f:
            fields = csv.DictReader(f).fieldnames
        assert fields == SCAFFOLD_FIELDS

    def test_scaffold_csv_labels_empty(self, tmp_path):
        out = self._run(tmp_path)
        with (out / "gt_patch_registry_scaffold_v2bo.csv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            assert r["gt_patch_flood_observed"] == ""
            assert r["allowed_for_training"] == "False"

    def test_no_forbidden_extensions(self, tmp_path):
        out = self._run(tmp_path)
        forbidden = {".npz", ".npy", ".parquet", ".tif", ".tiff", ".pt", ".pth", ".ckpt", ".safetensors"}
        for p in out.glob("*"):
            assert p.suffix.lower() not in forbidden

    def test_report_safe_language(self, tmp_path):
        out = self._run(tmp_path)
        text = (out / "gt_scaffold_report_v2bo.md").read_text(encoding="utf-8").lower()
        assert "no labels" in text
        assert "blocked" in text


class TestNoPrivatePaths:
    def test_no_private_path_in_script(self):
        user_prefix = "Users" + "\\" + "gabriela"
        text = (SCRIPTS_DIR / "revp_v2bo_ground_truth_training_gate_scaffold.py").read_text(encoding="utf-8", errors="replace")
        assert user_prefix not in text
        assert "/home/" not in text
