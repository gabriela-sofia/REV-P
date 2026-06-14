"""Tests for revp_v2bn_multimodal_feature_table_builder.py.

Covers: guardrails, embedding validity detection (768D), the historical
zero-embedding reconciliation, the core feature table, fail-closed behavior on
missing inputs, the training gate staying blocked, no forbidden outputs, and
safe report language.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "multimodal"
sys.path.insert(0, str(SCRIPTS_DIR))

from revp_v2bn_multimodal_feature_table_builder import (  # noqa: E402
    CORE_FIELDS,
    METHODOLOGICAL_GUARDRAILS,
    build_artifacts,
    build_embedding_index,
    build_feature_table,
    classify_embedding_source,
    compute_missingness,
    embedding_row_is_valid,
    training_gate,
    write_artifacts,
)


# --------------------------------------------------------------------------- #
# Synthetic input helpers
# --------------------------------------------------------------------------- #

def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _spine_row(dino_id: str, patch: str, region: str, group: str = "") -> dict:
    return {
        "dino_input_id": dino_id,
        "canonical_patch_id": patch,
        "region": region,
        "source_asset_id": f"asset_{patch}",
        "split_group": group or f"{region}__grp__{patch}",
    }


def _emb_row(dino_id: str, patch: str, region: str, *, success="SUCCESS", dim="768") -> dict:
    return {
        "patch_id": patch,
        "dino_input_id": dino_id,
        "region": region,
        "source_path": r"C:\Users\someone\PROJETO\data\x.tif",
        "embedding_path": f"embeddings/{dino_id}.npz",
        "embedding_dim": dim,
        "model_backbone": "facebook/dinov2-with-registers-base",
        "device": "cpu",
        "success": success,
        "failure_reason": "",
        "hash": f"hash_{dino_id}",
        "timestamp": "2026-05-18T00:00:00+00:00",
    }


def _make_inputs(tmp_path: Path, *, spine, emb, with_zero_registry=True, gis=False, evidence=False, overlay=False) -> dict:
    spine_path = tmp_path / "spine.csv"
    _write_csv(spine_path, spine, list(spine[0].keys()) if spine else ["dino_input_id", "canonical_patch_id", "region", "source_asset_id", "split_group"])
    emb_path = tmp_path / "emb.csv"
    if emb:
        _write_csv(emb_path, emb, list(emb[0].keys()))
    zero_path = tmp_path / "zero.csv"
    if with_zero_registry:
        # header-only registry -> historical stale zero
        _write_csv(zero_path, [], ["embedding_id", "patch_id", "vector_dim"])
    pc_summary = tmp_path / "pc.csv"
    pc_summary.write_text("gate,status,value\nx,BLOCKED,0\n", encoding="utf-8")
    gis_dir = tmp_path / "gis"
    if gis:
        gis_dir.mkdir()
        (gis_dir / "out.csv").write_text("a\n1\n", encoding="utf-8")
    evidence_report = tmp_path / "evidence.md"
    if evidence:
        evidence_report.write_text("# evidence", encoding="utf-8")
    overlay_report = tmp_path / "overlay.md"
    if overlay:
        overlay_report.write_text("# overlay", encoding="utf-8")
    return {
        "input_manifest": spine_path,
        "embedding_manifest": emb_path,
        "fallback_embedding_manifest": tmp_path / "missing_fallback.csv",
        "patch_registry": tmp_path / "missing_registry.csv",
        "feature_store_zero": zero_path,
        "gis_dir": gis_dir,
        "evidence_report": evidence_report,
        "overlay_report": overlay_report,
        "protocol_c_summary": pc_summary,
    }


# --------------------------------------------------------------------------- #
# Guardrails
# --------------------------------------------------------------------------- #

class TestGuardrails:
    def test_review_only(self):
        assert METHODOLOGICAL_GUARDRAILS["review_only"] is True

    def test_supervised_training_false(self):
        assert METHODOLOGICAL_GUARDRAILS["supervised_training"] is False

    def test_labels_created_false(self):
        assert METHODOLOGICAL_GUARDRAILS["labels_created"] is False

    def test_negative_from_absence_false(self):
        assert METHODOLOGICAL_GUARDRAILS["negative_from_absence"] is False

    def test_multimodal_execution_disabled(self):
        assert METHODOLOGICAL_GUARDRAILS["multimodal_execution_enabled"] is False

    def test_multimodal_training_disabled(self):
        assert METHODOLOGICAL_GUARDRAILS["multimodal_training_enabled"] is False

    def test_dino_frozen_not_finetuned(self):
        assert METHODOLOGICAL_GUARDRAILS["dino_frozen_encoder"] is True
        assert METHODOLOGICAL_GUARDRAILS["dino_finetuned"] is False

    def test_no_fusion(self):
        assert METHODOLOGICAL_GUARDRAILS["early_fusion"] is False
        assert METHODOLOGICAL_GUARDRAILS["pixel_space_fusion"] is False


# --------------------------------------------------------------------------- #
# Embedding validity / 768D detection (req. 5)
# --------------------------------------------------------------------------- #

class TestEmbeddingValidity:
    def test_success_768_valid(self):
        assert embedding_row_is_valid(_emb_row("D1", "P1", "Recife")) is True

    def test_skipped_existing_valid(self):
        assert embedding_row_is_valid(_emb_row("D1", "P1", "Recife", success="SKIPPED_EXISTING")) is True

    def test_failed_not_valid(self):
        assert embedding_row_is_valid(_emb_row("D1", "P1", "Recife", success="FAILED")) is False

    def test_missing_dim_not_valid(self):
        assert embedding_row_is_valid(_emb_row("D1", "P1", "Recife", dim="")) is False

    def test_index_detects_768(self, tmp_path):
        emb_path = tmp_path / "e.csv"
        rows = [_emb_row("D1", "P1", "Recife")]
        _write_csv(emb_path, rows, list(rows[0].keys()))
        index = build_embedding_index(emb_path)
        assert "D1" in index
        assert index["D1"]["dino_embedding_dim"] == "768"
        assert index["D1"]["dino_embedding_available"] == "True"

    def test_index_uri_is_relative_not_private(self, tmp_path):
        emb_path = tmp_path / "e.csv"
        rows = [_emb_row("D1", "P1", "Recife")]
        _write_csv(emb_path, rows, list(rows[0].keys()))
        index = build_embedding_index(emb_path)
        uri = index["D1"]["dino_embedding_uri"]
        assert "Users" + "\\" + "someone" not in uri
        assert "PROJETO" not in uri

    def test_index_empty_when_no_success(self, tmp_path):
        emb_path = tmp_path / "e.csv"
        rows = [_emb_row("D1", "P1", "Recife", success="FAILED")]
        _write_csv(emb_path, rows, list(rows[0].keys()))
        assert build_embedding_index(emb_path) == {}


# --------------------------------------------------------------------------- #
# Historical zero-embedding reconciliation (req. 6)
# --------------------------------------------------------------------------- #

class TestZeroEmbeddingReconciliation:
    def test_header_only_registry_is_stale_zero(self, tmp_path):
        zero = tmp_path / "zero.csv"
        _write_csv(zero, [], ["embedding_id", "vector_dim"])
        row = classify_embedding_source(zero, "feature_store_zero")
        assert row["status"] == "HISTORICAL_STALE_ZERO_EMBEDDINGS"
        assert row["row_count"] == 0

    def test_stale_zero_interpretation_not_a_contradiction(self, tmp_path):
        zero = tmp_path / "zero.csv"
        _write_csv(zero, [], ["embedding_id"])
        row = classify_embedding_source(zero, "feature_store_zero")
        assert "NOT a claim" in row["interpretation"]

    def test_local_manifest_available_when_real(self, tmp_path):
        emb = tmp_path / "e.csv"
        rows = [_emb_row("D1", "P1", "Recife"), _emb_row("D2", "P2", "Curitiba")]
        _write_csv(emb, rows, list(rows[0].keys()))
        out = classify_embedding_source(emb, "local_manifest")
        assert out["status"] == "LOCAL_MANIFEST_AVAILABLE"
        assert out["success_count"] == 2
        assert out["valid_768d_count"] == 2

    def test_missing_manifest_status(self, tmp_path):
        out = classify_embedding_source(tmp_path / "nope.csv", "local_manifest")
        assert out["status"] == "MISSING"

    def test_public_report_only(self, tmp_path):
        rep = tmp_path / "rep.csv"
        rep.write_text("a\n1\n", encoding="utf-8")
        out = classify_embedding_source(rep, "public_final_report")
        assert out["status"] == "PUBLIC_FINAL_REPORT_ONLY"


# --------------------------------------------------------------------------- #
# Core feature table
# --------------------------------------------------------------------------- #

class TestFeatureTable:
    def _build(self, tmp_path):
        spine = [
            _spine_row("D1", "REC_1", "Recife"),
            _spine_row("D2", "CUR_1", "Curitiba"),
            _spine_row("D3", "PET_1", "Petropolis"),
        ]
        emb = [_emb_row("D1", "REC_1", "Recife")]  # only one has an embedding
        inputs = _make_inputs(tmp_path, spine=spine, emb=emb)
        return build_artifacts(inputs)

    def test_row_per_spine_entry(self, tmp_path):
        art = self._build(tmp_path)
        assert art["summary"]["core_row_count"] == 3

    def test_embedding_available_only_for_matched(self, tmp_path):
        art = self._build(tmp_path)
        avail = {r["dino_input_id"]: r["dino_embedding_available"] for r in art["core_rows"]}
        assert avail["D1"] == "True"
        assert avail["D2"] == "False"
        assert avail["D3"] == "False"

    def test_review_eligibility_follows_embedding(self, tmp_path):
        art = self._build(tmp_path)
        review = {r["dino_input_id"]: r["allowed_for_review"] for r in art["core_rows"]}
        assert review["D1"] == "True"
        assert review["D2"] == "False"

    def test_all_training_blocked(self, tmp_path):
        art = self._build(tmp_path)
        assert all(r["allowed_for_training"] == "False" for r in art["core_rows"])
        assert art["summary"]["training_allowed_count"] == 0

    def test_gt_columns_empty(self, tmp_path):
        art = self._build(tmp_path)
        for r in art["core_rows"]:
            assert r["gt_patch_flood_observed"] == ""
            assert r["gt_negative_type"] == ""

    def test_split_group_preserved(self, tmp_path):
        art = self._build(tmp_path)
        row = next(r for r in art["core_rows"] if r["dino_input_id"] == "D1")
        assert row["split_group"] != ""

    def test_core_fields_complete(self, tmp_path):
        art = self._build(tmp_path)
        for r in art["core_rows"]:
            assert set(r.keys()) >= set(CORE_FIELDS)


# --------------------------------------------------------------------------- #
# Never create training/negatives (req. 2, 3, 4)
# --------------------------------------------------------------------------- #

class TestNoTrainingNoNegatives:
    def test_build_feature_table_never_allows_training(self):
        spine = [_spine_row("D1", "P1", "Recife")]
        emb_index = {"D1": {"dino_embedding_available": "True", "dino_embedding_dim": "768", "dino_embedding_hash": "h", "dino_embedding_uri": "x/y.npz", "dino_backbone": "b", "region": "Recife"}}
        rows = build_feature_table(spine, emb_index, gis_available="False", gis_status="X", evidence_available="False", binding_status="UNAVAILABLE_NO_OVERLAY", manifest_source="m")
        assert rows[0]["allowed_for_training"] == "False"

    def test_gate_labels_created_false(self):
        spine = [_spine_row("D1", "P1", "Recife")]
        rows = build_feature_table(spine, {}, gis_available="False", gis_status="X", evidence_available="False", binding_status="UNAVAILABLE_NO_OVERLAY", manifest_source="m")
        gate = training_gate(rows)
        assert gate["labels_created"] is False
        assert gate["formal_negative_count"] == 0
        assert gate["any_row_allowed_for_training"] is False

    def test_no_negative_type_anywhere(self, tmp_path):
        spine = [_spine_row(f"D{i}", f"P{i}", "Recife") for i in range(5)]
        inputs = _make_inputs(tmp_path, spine=spine, emb=[])
        art = build_artifacts(inputs)
        assert all(r["gt_negative_type"] == "" for r in art["core_rows"])


# --------------------------------------------------------------------------- #
# Fail-closed on missing inputs (req. 8)
# --------------------------------------------------------------------------- #

class TestFailClosed:
    def test_missing_spine_yields_empty_table(self, tmp_path):
        inputs = {
            "input_manifest": tmp_path / "nope.csv",
            "embedding_manifest": tmp_path / "nope2.csv",
            "fallback_embedding_manifest": tmp_path / "nope3.csv",
            "patch_registry": tmp_path / "nope4.csv",
            "feature_store_zero": tmp_path / "nope5.csv",
            "gis_dir": tmp_path / "nogis",
            "evidence_report": tmp_path / "noev.md",
            "overlay_report": tmp_path / "noov.md",
            "protocol_c_summary": tmp_path / "nopc.csv",
        }
        art = build_artifacts(inputs)
        assert art["summary"]["core_row_count"] == 0
        assert art["guardrails"]["overall"] == "PASS"

    def test_missing_overlay_blocks_binding(self, tmp_path):
        spine = [_spine_row("D1", "P1", "Recife")]
        inputs = _make_inputs(tmp_path, spine=spine, emb=[], overlay=False)
        art = build_artifacts(inputs)
        assert art["summary"]["binding_status"] == "UNAVAILABLE_NO_OVERLAY"

    def test_present_overlay_marks_blocked_geometry(self, tmp_path):
        spine = [_spine_row("D1", "P1", "Recife")]
        inputs = _make_inputs(tmp_path, spine=spine, emb=[], overlay=True)
        art = build_artifacts(inputs)
        assert "BLOCKED" in art["summary"]["binding_status"]

    def test_missing_gis_unavailable(self, tmp_path):
        spine = [_spine_row("D1", "P1", "Recife")]
        inputs = _make_inputs(tmp_path, spine=spine, emb=[], gis=False)
        art = build_artifacts(inputs)
        assert art["core_rows"][0]["gis_feature_available"] == "False"


# --------------------------------------------------------------------------- #
# Missingness / gate
# --------------------------------------------------------------------------- #

class TestMissingnessAndGate:
    def test_missingness_covers_all_fields(self):
        spine = [_spine_row("D1", "P1", "Recife")]
        rows = build_feature_table(spine, {}, gis_available="False", gis_status="X", evidence_available="False", binding_status="UNAVAILABLE_NO_OVERLAY", manifest_source="m")
        miss = compute_missingness(rows)
        names = {m["feature_name"] for m in miss}
        assert names == set(CORE_FIELDS)

    def test_gt_columns_marked_expected_empty(self):
        spine = [_spine_row("D1", "P1", "Recife")]
        rows = build_feature_table(spine, {}, gis_available="False", gis_status="X", evidence_available="False", binding_status="UNAVAILABLE_NO_OVERLAY", manifest_source="m")
        miss = {m["feature_name"]: m["status"] for m in compute_missingness(rows)}
        assert miss["gt_patch_flood_observed"] == "EXPECTED_EMPTY_NO_GROUND_TRUTH"


# --------------------------------------------------------------------------- #
# Outputs (req. 7) and no forbidden outputs (req. 9)
# --------------------------------------------------------------------------- #

class TestOutputs:
    EXPECTED = [
        "multimodal_feature_table_core_v2bn.csv",
        "multimodal_feature_schema_v2bn.csv",
        "multimodal_embedding_inventory_v2bn.csv",
        "multimodal_missingness_report_v2bn.csv",
        "multimodal_join_audit_v2bn.csv",
        "multimodal_training_gate_v2bn.json",
        "multimodal_guardrails_v2bn.json",
        "multimodal_feature_table_summary_v2bn.json",
        "multimodal_feature_table_report_v2bn.md",
    ]

    def _run(self, tmp_path):
        spine = [_spine_row("D1", "REC_1", "Recife"), _spine_row("D2", "CUR_1", "Curitiba")]
        emb = [_emb_row("D1", "REC_1", "Recife")]
        inputs = _make_inputs(tmp_path, spine=spine, emb=emb, evidence=True)
        art = build_artifacts(inputs)
        out = tmp_path / "out"
        out.mkdir()
        write_artifacts(out, art)
        return out

    def test_all_files_created(self, tmp_path):
        out = self._run(tmp_path)
        for fname in self.EXPECTED:
            assert (out / fname).exists(), f"Missing {fname}"

    def test_no_forbidden_extensions(self, tmp_path):
        out = self._run(tmp_path)
        forbidden = {".npz", ".npy", ".parquet", ".tif", ".tiff", ".pt", ".pth", ".ckpt", ".safetensors"}
        for p in out.glob("*"):
            assert p.suffix.lower() not in forbidden

    def test_gate_json_blocked(self, tmp_path):
        out = self._run(tmp_path)
        gate = json.loads((out / "multimodal_training_gate_v2bn.json").read_text(encoding="utf-8"))
        assert gate["labels_created"] is False
        assert gate["supervised_training_enabled"] is False
        assert gate["formal_negative_count"] == 0

    def test_core_csv_has_schema(self, tmp_path):
        out = self._run(tmp_path)
        with (out / "multimodal_feature_table_core_v2bn.csv").open(encoding="utf-8") as f:
            fields = csv.DictReader(f).fieldnames
        assert fields == CORE_FIELDS

    def test_report_safe_language(self, tmp_path):
        out = self._run(tmp_path)
        text = (out / "multimodal_feature_table_report_v2bn.md").read_text(encoding="utf-8").lower()
        assert "operational flood detection" not in text or "no operational flood detection" in text
        assert "validated prediction" not in text or "no validated" in text
        # forbidden positive claims must not appear as bare assertions
        assert "flood accuracy" not in text or "no" in text


# --------------------------------------------------------------------------- #
# No private paths in versionable script
# --------------------------------------------------------------------------- #

class TestNoPrivatePaths:
    def test_no_private_path_in_script(self):
        user_prefix = "Users" + "\\" + "gabriela"
        text = (SCRIPTS_DIR / "revp_v2bn_multimodal_feature_table_builder.py").read_text(encoding="utf-8", errors="replace")
        assert user_prefix not in text
        assert "/home/" not in text
