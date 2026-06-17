"""Testes para revp_v2ch_consolidacao_cadeia_curitiba.py.

Cobre a etapa de consolidação: inventaria a cadeia v2ca-v2cg, agrega critérios,
travas e testes, registra a prontidão metodológica, separa mudanças não
relacionadas e mantém ground truth, label, negativo formal e treino bloqueados.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "multimodal"
sys.path.insert(0, str(SCRIPTS_DIR))

from revp_v2ch_consolidacao_cadeia_curitiba import (  # noqa: E402
    CHAIN,
    METHODOLOGICAL_GUARDRAILS,
    build_artifacts,
    write_artifacts,
)

FORBIDDEN_OPERATIONAL_LANGUAGE = [
    "operational model", "flood accuracy",
    "ground truth validado", "label ready", "training ready",
]


def _build_env(tmp_path):
    """Create a synthetic chain environment (scripts/tests/docs/gates)."""
    scripts = tmp_path / "scripts" / "multimodal"
    tests = tmp_path / "tests"
    docs = tmp_path / "docs" / "metodologia_cientifica"
    templates = docs / "templates"
    gt = tmp_path / "gt"
    for d in (scripts, tests, templates, gt):
        d.mkdir(parents=True, exist_ok=True)

    from revp_v2ch_consolidacao_cadeia_curitiba import STAGE_META
    for sid in CHAIN:
        (scripts / f"revp_{sid}_x.py").write_text("# script\n", encoding="utf-8")
        (tests / f"test_revp_{sid}_x.py").write_text("def test_a():\n    assert True\n\ndef test_b():\n    assert True\n", encoding="utf-8")
        (docs / f"revp_{sid}_x.md").write_text("# doc\n", encoding="utf-8")
        sgt = gt / sid
        sgt.mkdir(parents=True, exist_ok=True)
        gate = {"blocked_reason": f"{sid}_blocked", "formal_labels_created": False,
                "allowed_for_training_count": 0, "can_train_supervised_model": False}
        if sid == "v2ca":
            gate.update({"curitiba_events_repaired": 2, "patches_with_boundary": 43,
                         "patch_event_bindings_created": 86})
        if sid == "v2cc":
            gate["external_evidence_package_created"] = True
        if sid == "v2ce":
            gate["official_data_request_dossier_created"] = True
        if sid == "v2cf":
            gate["qa_geometry_candidates_created"] = 0
        if sid == "v2cg":
            gate["dry_run_positive_candidates"] = 0
        (sgt / STAGE_META[sid]["gate"]).write_text(json.dumps(gate), encoding="utf-8")
        (sgt / STAGE_META[sid]["guardrail"]).write_text(
            json.dumps({"overall": "PASS", "checks": {"a": "PASS", "b": "PASS"}}), encoding="utf-8")
    (templates / "curitiba_external_event_evidence_template_v2cc.csv").write_text("event_id\n", encoding="utf-8")
    (templates / "curitiba_external_event_evidence_template_v2cc.md").write_text("# template\n", encoding="utf-8")
    return scripts, tests, docs, gt


_STATUS = [
    " M datasets/protocolo_c/v2bb_adjudication_decision_table.csv",
    " M docs/protocolo_c/v2bb_secondary_evidence_adjudication/README.md",
]


def _art(tmp_path, status=None):
    scripts, tests, docs, gt = _build_env(tmp_path)
    return build_artifacts(output_dir=tmp_path / "out", scripts_dir=scripts, tests_dir=tests,
                           docs_dir=docs, gt_dir=gt,
                           git_status_override=_STATUS if status is None else status)


# --------------------------------------------------------------------------- #
# 1-3. Runs / inventories
# --------------------------------------------------------------------------- #

def test_runs_with_minimal_inputs(tmp_path):
    art = _art(tmp_path)
    assert art["summary"]["phase"] == "v2ch"
    assert art["guardrails"]["overall"] == "PASS"


def test_inventories_chain_stages(tmp_path):
    art = _art(tmp_path)
    sids = {r["stage_id"] for r in art["stage_rows"]}
    assert sids == set(CHAIN)
    assert art["summary"]["stages_inventoried"] == 7


def test_inventories_scripts_tests_docs(tmp_path):
    art = _art(tmp_path)
    assert art["summary"]["scripts_found"] == 7
    assert art["summary"]["tests_found"] == 7
    assert art["summary"]["docs_found"] == 7


# --------------------------------------------------------------------------- #
# 4-5. Rollups
# --------------------------------------------------------------------------- #

def test_gate_rollup_works(tmp_path):
    art = _art(tmp_path)
    keys = {(r["stage_id"], r["gate_key"]) for r in art["gate_rollup"]}
    assert ("v2ca", "blocked_reason") in keys
    assert ("v2cg", "can_train_supervised_model") in keys


def test_guardrail_rollup_works(tmp_path):
    art = _art(tmp_path)
    assert len(art["guardrail_rollup"]) == 7
    assert all(g["overall"] == "PASS" for g in art["guardrail_rollup"])


# --------------------------------------------------------------------------- #
# 6-8. Scientific readiness
# --------------------------------------------------------------------------- #

def test_readiness_identifies_43_boundaries(tmp_path):
    art = _art(tmp_path)
    item = next(r for r in art["readiness"] if r["readiness_item"] == "curitiba_patch_boundaries_available")
    assert item["current_status"] == "PRESENT_43"


def test_readiness_identifies_absent_geometry(tmp_path):
    art = _art(tmp_path)
    item = next(r for r in art["readiness"] if r["readiness_item"] == "valid_event_geometry_available")
    assert item["current_status"] == "ABSENT"
    assert item["blocks_overlay"] == "true"


def test_readiness_overlay_executor_available_but_blocked(tmp_path):
    art = _art(tmp_path)
    item = next(r for r in art["readiness"] if r["readiness_item"] == "overlay_executor_available")
    assert item["current_status"] == "AVAILABLE_BUT_BLOCKED"


# --------------------------------------------------------------------------- #
# 9-10. Public artifact plan
# --------------------------------------------------------------------------- #

def test_public_plan_excludes_raw_local_runs(tmp_path):
    art = _art(tmp_path)
    assert all(p["contains_raw_local_runs"] == "false" for p in art["public_plan"])
    assert all("local_runs/" not in p["public_path"] for p in art["public_plan"])


def test_public_plan_excludes_downloaded_sources(tmp_path):
    art = _art(tmp_path)
    assert all("downloaded_sources" not in p["public_path"] for p in art["public_plan"])


# --------------------------------------------------------------------------- #
# 11-14. Commit manifest / unrelated changes
# --------------------------------------------------------------------------- #

def test_commit_manifest_only_intentional(tmp_path):
    art = _art(tmp_path)
    assert all(m["include_in_commit"] == "true" for m in art["manifest"])
    types = {m["file_type"] for m in art["manifest"]}
    assert {"script", "test", "doc", "template"}.issubset(types)


def test_commit_manifest_excludes_v2bb(tmp_path):
    art = _art(tmp_path)
    assert all("v2bb" not in m["file_path"] for m in art["manifest"])


def test_commit_manifest_excludes_protocolo_c(tmp_path):
    art = _art(tmp_path)
    assert all("docs/protocolo_c" not in m["file_path"] for m in art["manifest"])


def test_unrelated_registry_records_changes(tmp_path):
    art = _art(tmp_path)
    paths = {u["file_path"] for u in art["unrelated"]}
    assert any("v2bb" in p for p in paths)
    assert any("docs/protocolo_c" in p for p in paths)
    assert all(u["include_in_commit"] == "false" for u in art["unrelated"])
    assert all(u["change_category"] == "UNRELATED_WORKING_TREE_CHANGE" for u in art["unrelated"])


# --------------------------------------------------------------------------- #
# 15-16. README patch / commit message
# --------------------------------------------------------------------------- #

def test_readme_patch_created(tmp_path):
    art = _art(tmp_path)
    write_artifacts(tmp_path / "out", art, publish=False)
    readme = tmp_path / "out" / "curitiba_readme_patch_v2ch.md"
    assert readme.exists()
    assert "Curitiba" in readme.read_text(encoding="utf-8")


def test_commit_message_created(tmp_path):
    art = _art(tmp_path)
    write_artifacts(tmp_path / "out", art, publish=False)
    msg = tmp_path / "out" / "curitiba_commit_message_suggestion_v2ch.md"
    assert msg.exists()
    text = msg.read_text(encoding="utf-8")
    assert "docs: consolida cadeia de ground truth e estado metodológico do REV-P" in text
    assert "ground truth operacional segue bloqueado" in text


# --------------------------------------------------------------------------- #
# 17-19. Invariants
# --------------------------------------------------------------------------- #

def test_no_label_no_negative_no_training(tmp_path):
    art = _art(tmp_path)
    s = art["summary"]
    assert s["formal_labels_created"] is False
    assert s["formal_negatives_created"] is False
    assert s["allowed_for_training_count"] == 0
    assert s["supervised_training_enabled"] is False
    for r in art["stage_rows"]:
        assert r["creates_label"] == "false" and r["allows_training"] == "false"


# --------------------------------------------------------------------------- #
# 20-23. Safety / outputs / report / guardrails
# --------------------------------------------------------------------------- #

def test_no_private_absolute_paths(tmp_path):
    art = _art(tmp_path)
    write_artifacts(tmp_path / "out", art, publish=False)
    needles = ["C:\\Users", "/Users/", "/home/", "gabriela"]
    for p in (tmp_path / "out").rglob("*"):
        if p.is_file() and p.suffix in {".csv", ".json", ".md"}:
            text = p.read_text(encoding="utf-8")
            for n in needles:
                assert n not in text, f"{n!r} leaked into {p.name}"


def test_no_heavy_outputs(tmp_path):
    art = _art(tmp_path)
    write_artifacts(tmp_path / "out", art, publish=False)
    heavy = {".npz", ".npy", ".pt", ".pth", ".parquet", ".tif", ".tiff", ".shp", ".zip", ".gpkg"}
    for p in (tmp_path / "out").rglob("*"):
        if p.is_file():
            assert p.suffix.lower() not in heavy


def test_report_has_no_forbidden_claims(tmp_path):
    art = _art(tmp_path)
    write_artifacts(tmp_path / "out", art, publish=False)
    report = (tmp_path / "out" / "curitiba_release_report_v2ch.md").read_text(encoding="utf-8").lower()
    body = report.split("## nota de trava metodológica")[0]
    for claim in FORBIDDEN_OPERATIONAL_LANGUAGE:
        assert claim.lower() not in body
    assert "não reivindica uso operacional" in report


def test_generates_all_expected_outputs(tmp_path):
    art = _art(tmp_path)
    files = write_artifacts(tmp_path / "out", art, publish=False)
    expected = {
        "curitiba_chain_stage_inventory_v2ch.csv", "curitiba_chain_output_inventory_v2ch.csv",
        "curitiba_chain_gate_rollup_v2ch.csv", "curitiba_chain_guardrail_rollup_v2ch.csv",
        "curitiba_chain_test_rollup_v2ch.csv", "curitiba_scientific_readiness_rollup_v2ch.csv",
        "curitiba_public_artifact_plan_v2ch.csv", "curitiba_commit_file_manifest_v2ch.csv",
        "curitiba_commit_hygiene_checklist_v2ch.csv", "curitiba_unrelated_working_tree_changes_v2ch.csv",
        "curitiba_readme_patch_v2ch.md", "curitiba_commit_message_suggestion_v2ch.md",
        "curitiba_release_guardrails_v2ch.json", "curitiba_release_summary_v2ch.json",
        "curitiba_release_report_v2ch.md",
    }
    assert expected.issubset(set(files))


def test_guardrails_pass(tmp_path):
    art = _art(tmp_path)
    g = art["guardrails"]
    assert g["overall"] == "PASS"
    for key, verdict in g["checks"].items():
        assert verdict in {"PASS", "BLOCKED_EXPECTED"}, f"{key}={verdict}"
    assert g["checks"]["unrelated_v2bb_changes_not_in_commit_manifest"] == "PASS"
    assert METHODOLOGICAL_GUARDRAILS["labels_created"] is False
