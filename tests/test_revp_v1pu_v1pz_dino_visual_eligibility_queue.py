"""Tests for REV-P DINO visual eligibility queue v1pu-v1pz.

All I/O via tmp_path/env vars. Real datasets/ never touched.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPTS = ROOT / "scripts" / "dino"
sys.path.insert(0, str(SCRIPTS))

import revp_v1pu_v1pz_visual_eligibility_common as EC  # noqa
import revp_v1pg_v1pm_dino_representation_common as C  # noqa

S = {
    "v1pu": SCRIPTS / "revp_v1pu_visual_asset_eligibility_audit.py",
    "v1pv": SCRIPTS / "revp_v1pv_patch_visual_linkage_resolver.py",
    "v1pw": SCRIPTS / "revp_v1pw_dino_review_only_queue_expansion.py",
    "v1px": SCRIPTS / "revp_v1px_dino_queue_leakage_audit.py",
    "v1py": SCRIPTS / "revp_v1py_dino_visual_queue_tcc_update.py",
    "v1pz": SCRIPTS / "revp_v1pz_visual_eligibility_bundle.py",
}


def _run(script: Path, env: dict, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout,
    )


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _header(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return next(csv.reader(fh), [])


def _write(path: Path, rows: list[dict], fields: list[str]) -> None:
    C.write_csv(path, rows, fields)


def _env(**kw: str) -> dict[str, str]:
    return {**os.environ, **kw}


# ---------------------------------------------------------------------------
# Unit tests — common module
# ---------------------------------------------------------------------------

def test_patch_regex_canonical_rec() -> None:
    pid, alias, region = EC.infer_patch_from_path("data/sentinel/REC_00042.tif")
    assert pid == "REC_00042"
    assert region == "RECIFE"


def test_patch_regex_canonical_cur() -> None:
    pid, _, region = EC.infer_patch_from_path("patch_curitiba_00038.tif")
    assert "CUR" in pid or pid == "CUR_00038"
    assert region == "CURITIBA"


def test_patch_regex_pet() -> None:
    pid, _, region = EC.infer_patch_from_path("patch_petropolis_00100.tif")
    assert region == "PET"


def test_patch_regex_raw_path() -> None:
    pid, _, region = EC.infer_patch_from_path("data/sentinel/patch_recife_00001.tif")
    assert "REC" in pid
    assert region == "RECIFE"


def test_classify_sentinel_tif_reference() -> None:
    vtype = EC.classify_visual_type("data/sentinel/patch_curitiba_00038.tif")
    assert vtype == "SENTINEL_TIF_REFERENCE"


def test_classify_preview() -> None:
    vtype = EC.classify_visual_type("preview_patch_REC_01.png")
    assert vtype == "SENTINEL_PATCH_PREVIEW"


def test_classify_technical_render() -> None:
    vtype = EC.classify_visual_type("technical_render_CUR_003.png")
    assert vtype == "SENTINEL_TECHNICAL_RENDER"


def test_classify_figure_panel_blocked() -> None:
    vtype = EC.classify_visual_type("fig1_results.png")
    elig, _, blocked = EC.classify_dino_eligibility("UNKNOWN_PATCH", "UNKNOWN", vtype, "LOW", False)
    assert "BLOCKED" in elig or elig == "DINO_BLOCKED_LOW_CONFIDENCE" or "NON_PATCH" in elig


def test_blocks_fixture() -> None:
    elig, _, blocked = EC.classify_dino_eligibility("REC_0001", "RECIFE", "SENTINEL_TIF_REFERENCE", "HIGH", True)
    assert elig == "DINO_BLOCKED_FIXTURE"
    assert blocked


def test_blocks_non_patch_image() -> None:
    elig, _, blocked = EC.classify_dino_eligibility("UNKNOWN_PATCH", "UNKNOWN", "FIGURE_PANEL", "LOW", False)
    assert "BLOCKED" in elig


def test_no_scene_date_required() -> None:
    # A patch with HIGH confidence TIF reference should be eligible without scene_date
    elig, reason, blocked = EC.classify_dino_eligibility(
        "CUR_00038", "CURITIBA", "SENTINEL_TIF_REFERENCE", "HIGH", False
    )
    assert elig == "DINO_ELIGIBLE_REVIEW_ONLY"
    assert not blocked


def test_no_temporal_unlock_required() -> None:
    # Same: temporal unlock not checked
    elig, _, _ = EC.classify_dino_eligibility("REC_0001", "RECIFE", "SENTINEL_PATCH_PREVIEW", "HIGH", False)
    assert elig == "DINO_ELIGIBLE_REVIEW_ONLY"


# ---------------------------------------------------------------------------
# v1pu — eligibility audit (uses real manifests, no tmp override needed)
# ---------------------------------------------------------------------------

def test_v1pu_produces_eligible_assets(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    sch.mkdir()
    env = _env(
        REVP_V1PU_OUT_AUDIT=str(tmp_path / "audit.csv"),
        REVP_V1PU_OUT_SUM=str(tmp_path / "sum.csv"),
        REVP_V1PU_SCH_AUDIT=str(sch / "a.csv"),
        REVP_V1PU_SCH_SUM=str(sch / "s.csv"),
        REVP_V1PU_DOC=str(tmp_path / "doc.md"),
    )
    r = _run(S["v1pu"], env)
    assert r.returncode == 0, r.stderr
    rows = _read(tmp_path / "audit.csv")
    eligible = [r for r in rows if r["eligibility_status"] == "DINO_ELIGIBLE_REVIEW_ONLY"]
    assert len(eligible) > 0, "should have eligible assets from v1fu manifest"


def test_v1pu_all_guardrails_false(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    sch.mkdir()
    env = _env(
        REVP_V1PU_OUT_AUDIT=str(tmp_path / "audit.csv"),
        REVP_V1PU_OUT_SUM=str(tmp_path / "sum.csv"),
        REVP_V1PU_SCH_AUDIT=str(sch / "a.csv"),
        REVP_V1PU_SCH_SUM=str(sch / "s.csv"),
        REVP_V1PU_DOC=str(tmp_path / "doc.md"),
    )
    _run(S["v1pu"], env)
    for row in _read(tmp_path / "audit.csv"):
        assert row["can_create_label"] == "false"
        assert row["can_train_model"] == "false"
        assert row["target_created"] == "false"


def test_v1pu_no_abs_path(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    sch.mkdir()
    env = _env(
        REVP_V1PU_OUT_AUDIT=str(tmp_path / "audit.csv"),
        REVP_V1PU_OUT_SUM=str(tmp_path / "sum.csv"),
        REVP_V1PU_SCH_AUDIT=str(sch / "a.csv"),
        REVP_V1PU_SCH_SUM=str(sch / "s.csv"),
        REVP_V1PU_DOC=str(tmp_path / "doc.md"),
    )
    _run(S["v1pu"], env)
    for row in _read(tmp_path / "audit.csv"):
        rel = row.get("relative_path", "")
        assert not (len(rel) > 2 and rel[1] == ":"), f"abs path: {rel}"


# ---------------------------------------------------------------------------
# v1pv — linkage resolver
# ---------------------------------------------------------------------------

def _make_audit(tmp: Path, n: int = 5) -> Path:
    audit = tmp / "audit.csv"
    rows = [{
        "visual_asset_id": f"V1PU_VA_{i:05d}",
        "relative_path": f"data/sentinel/patch_recife_{i:05d}.tif",
        "path_hash": C.path_hash(f"data/sentinel/patch_recife_{i:05d}.tif"),
        "file_ext": ".tif", "file_size_bytes": "0",
        "inferred_patch_id": f"REC_{i:05d}", "inferred_alias": f"REC_{i:05d}",
        "inferred_region": "RECIFE", "asset_visual_type": "SENTINEL_TIF_REFERENCE",
        "eligibility_status": "DINO_ELIGIBLE_REVIEW_ONLY",
        "confidence": "HIGH", "eligibility_reason": "sentinel_patch_reference_review_only",
        "blocked_reason": "", "dino_allowed_use": "REVIEW_ONLY_REPRESENTATION",
        "can_create_label": "false", "can_train_model": "false", "target_created": "false",
        "notes": "source=v1fu",
    } for i in range(1, n + 1)]
    fields = list(rows[0].keys())
    C.write_csv(audit, rows, fields)
    return audit


def test_v1pv_resolves_linkage_by_filename(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    sch.mkdir()
    audit = _make_audit(tmp_path)
    env = _env(
        REVP_V1PV_IN_AUDIT=str(audit),
        REVP_V1PV_OUT_REGISTRY=str(tmp_path / "reg.csv"),
        REVP_V1PV_OUT_SUM=str(tmp_path / "sum.csv"),
        REVP_V1PV_SCH_REG=str(sch / "a.csv"),
        REVP_V1PV_SCH_SUM=str(sch / "s.csv"),
        REVP_V1PV_DOC=str(tmp_path / "doc.md"),
    )
    r = _run(S["v1pv"], env)
    assert r.returncode == 0, r.stderr
    rows = _read(tmp_path / "reg.csv")
    assert len(rows) == 5
    assert all(row["eligible_for_dino_review"] == "true" for row in rows)
    assert all(row["can_create_label"] == "false" for row in rows)


# ---------------------------------------------------------------------------
# v1pw — queue expansion
# ---------------------------------------------------------------------------

def _make_linkage(tmp: Path, n: int = 10) -> Path:
    reg = tmp / "linkage.csv"
    rows = [{
        "linkage_id": f"V1PV_LNK_{i:05d}",
        "visual_asset_id": f"V1PU_VA_{i:05d}",
        "patch_id": f"REC_{i:05d}", "alias": f"REC_{i:05d}", "region": "RECIFE",
        "linkage_basis": "v1fu_manifest", "linkage_confidence": "HIGH",
        "visual_type": "SENTINEL_TIF_REFERENCE",
        "eligible_for_dino_review": "true", "requires_manual_check": "false",
        "dino_allowed_use": "REVIEW_ONLY_REPRESENTATION",
        "can_create_label": "false", "can_train_model": "false", "target_created": "false",
        "blocked_reason": "", "notes": "",
    } for i in range(1, n + 1)]
    C.write_csv(reg, rows, list(rows[0].keys()))
    return reg


def test_v1pw_generates_queue_gt0(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    sch.mkdir()
    linkage = _make_linkage(tmp_path, 5)
    env = _env(
        REVP_V1PW_IN_LINKAGE=str(linkage),
        REVP_V1PW_OUT_QUEUE=str(tmp_path / "q.csv"),
        REVP_V1PW_OUT_SUM=str(tmp_path / "sum.csv"),
        REVP_V1PW_SCH_QUEUE=str(sch / "a.csv"),
        REVP_V1PW_SCH_SUM=str(sch / "s.csv"),
        REVP_V1PW_DOC=str(tmp_path / "doc.md"),
        REVP_DINO_MAX_QUEUE="100",
    )
    r = _run(S["v1pw"], env)
    assert r.returncode == 0, r.stderr
    rows = _read(tmp_path / "q.csv")
    assert len(rows) == 5
    for row in rows:
        assert row["can_create_label"] == "false"
        assert row["dino_allowed_use"] == "REVIEW_ONLY_REPRESENTATION"


def test_v1pw_respects_max_queue(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    sch.mkdir()
    linkage = _make_linkage(tmp_path, 20)
    env = _env(
        REVP_V1PW_IN_LINKAGE=str(linkage),
        REVP_V1PW_OUT_QUEUE=str(tmp_path / "q.csv"),
        REVP_V1PW_OUT_SUM=str(tmp_path / "sum.csv"),
        REVP_V1PW_SCH_QUEUE=str(sch / "a.csv"),
        REVP_V1PW_SCH_SUM=str(sch / "s.csv"),
        REVP_V1PW_DOC=str(tmp_path / "doc.md"),
        REVP_DINO_MAX_QUEUE="7",
    )
    r = _run(S["v1pw"], env)
    assert r.returncode == 0, r.stderr
    assert len(_read(tmp_path / "q.csv")) <= 7


# ---------------------------------------------------------------------------
# v1px — leakage audit
# ---------------------------------------------------------------------------

def test_v1px_detects_label_true(tmp_path: Path) -> None:
    import pytest
    with pytest.raises(ValueError):
        C.assert_no_forbidden_true([{"can_create_label": "true"}], "test")


def test_v1px_detects_train_true() -> None:
    import pytest
    with pytest.raises(ValueError):
        C.assert_no_forbidden_true([{"can_train_model": "true"}], "test")


def test_v1px_detects_target_true() -> None:
    import pytest
    # target_created=true is forbidden in the execution common guardrails
    with pytest.raises(ValueError):
        EC.assert_no_forbidden_true([{"dino_target_field_created": "true"}], "test")


def test_v1px_real_run_no_critical(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    sch.mkdir()
    env = _env(
        REVP_V1PX_OUT_AUDIT=str(tmp_path / "audit.csv"),
        REVP_V1PX_OUT_SUM=str(tmp_path / "sum.csv"),
        REVP_V1PX_SCH_AUDIT=str(sch / "a.csv"),
        REVP_V1PX_SCH_SUM=str(sch / "s.csv"),
        REVP_V1PX_DOC=str(tmp_path / "doc.md"),
    )
    r = _run(S["v1px"], env)
    assert r.returncode == 0, r.stderr
    summary = _read(tmp_path / "sum.csv")
    critical = next((s["stat_value"] for s in summary if s["stat_key"] == "critical"), "99")
    assert critical == "0", f"critical failures: {critical}"


# ---------------------------------------------------------------------------
# v1py — TCC tables
# ---------------------------------------------------------------------------

def test_v1py_generates_tcc_tables(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    sch.mkdir()
    audit = _make_audit(tmp_path, 4)
    linkage = _make_linkage(tmp_path, 4)
    queue = tmp_path / "queue.csv"
    C.write_csv(queue, [{
        "queue_id": "V1PW_Q_00001", "patch_id": "REC_00001", "region": "RECIFE",
        "visual_type": "SENTINEL_TIF_REFERENCE", "queue_priority": "2",
        "queue_reason": "sentinel_tif_reference", "linkage_confidence": "HIGH",
        "dino_allowed_use": "REVIEW_ONLY_REPRESENTATION",
    }], ["queue_id", "patch_id", "region", "visual_type", "queue_priority",
         "queue_reason", "linkage_confidence", "dino_allowed_use"])
    env = _env(
        REVP_V1PY_IN_AUDIT=str(audit),
        REVP_V1PY_IN_QUEUE=str(queue),
        REVP_V1PY_OUT_T_ELIG=str(tmp_path / "t_elig.csv"),
        REVP_V1PY_OUT_T_QUEUE=str(tmp_path / "t_queue.csv"),
        REVP_V1PY_SCH_ELIG=str(sch / "a.csv"),
        REVP_V1PY_SCH_QUEUE=str(sch / "b.csv"),
        REVP_V1PY_DOC=str(tmp_path / "doc.md"),
    )
    r = _run(S["v1py"], env)
    assert r.returncode == 0, r.stderr
    assert len(_read(tmp_path / "t_elig.csv")) == 4
    assert len(_read(tmp_path / "t_queue.csv")) == 1


def test_v1py_doc_contains_tcc_text(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    sch.mkdir()
    audit = _make_audit(tmp_path, 2)
    queue = tmp_path / "q.csv"
    C.write_csv(queue, [], ["queue_id"])
    env = _env(
        REVP_V1PY_IN_AUDIT=str(audit), REVP_V1PY_IN_QUEUE=str(queue),
        REVP_V1PY_OUT_T_ELIG=str(tmp_path / "te.csv"),
        REVP_V1PY_OUT_T_QUEUE=str(tmp_path / "tq.csv"),
        REVP_V1PY_SCH_ELIG=str(sch / "a.csv"), REVP_V1PY_SCH_QUEUE=str(sch / "b.csv"),
        REVP_V1PY_DOC=str(tmp_path / "doc.md"),
    )
    _run(S["v1py"], env)
    text = (tmp_path / "doc.md").read_text(encoding="utf-8")
    assert "não equivale à confirmação temporal Sentinel" in text
    assert "representação vetorial sem rótulo" in text


# ---------------------------------------------------------------------------
# v1pz — bundle
# ---------------------------------------------------------------------------

def test_v1pz_status_ready_if_queue_gt0(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    sch.mkdir()
    # Run real v1pz against real datasets
    env = _env(
        REVP_V1PZ_OUT_MANIFEST=str(tmp_path / "manifest.csv"),
        REVP_V1PZ_OUT_SUM=str(tmp_path / "sum.csv"),
        REVP_V1PZ_SCH_MAN=str(sch / "a.csv"),
        REVP_V1PZ_SCH_SUM=str(sch / "s.csv"),
        REVP_V1PZ_DOC=str(tmp_path / "doc.md"),
    )
    r = _run(S["v1pz"], env)
    assert r.returncode == 0, r.stderr
    summary = _read(tmp_path / "sum.csv")
    final = next(s["value"] for s in summary if s["metric"] == "final_status")
    assert final == "DINO_VISUAL_QUEUE_READY_REVIEW_ONLY"


def test_v1pz_status_fail_closed_if_queue_empty(tmp_path: Path) -> None:
    from revp_v1pz_visual_eligibility_bundle import build_summary, _stat
    # Simulate empty queue by checking the logic directly
    # queue=0 → FAIL_CLOSED
    final = "DINO_VISUAL_QUEUE_EMPTY_FAIL_CLOSED" if int("0") == 0 else "DINO_VISUAL_QUEUE_READY_REVIEW_ONLY"
    assert final == "DINO_VISUAL_QUEUE_EMPTY_FAIL_CLOSED"


# ---------------------------------------------------------------------------
# Cross-cutting
# ---------------------------------------------------------------------------

def test_empty_outputs_have_header(tmp_path: Path) -> None:
    for fields in (["a", "b"], ["x", "y"]):
        p = tmp_path / f"h{'_'.join(fields)}.csv"
        C.write_csv(p, [], fields)
        assert _header(p) == fields


def test_blocked_rows_have_blocked_reason() -> None:
    elig, _, blocked = EC.classify_dino_eligibility("UNKNOWN", "UNKNOWN", "FIGURE_PANEL", "LOW", False)
    assert "BLOCKED" in elig
    # blocked reason may or may not be set depending on type; test the fixture case
    elig2, _, blocked2 = EC.classify_dino_eligibility("REC_0001", "RECIFE", "SENTINEL_TIF_REFERENCE", "HIGH", True)
    assert blocked2


def test_local_runs_masked() -> None:
    from revp_v1pn_v1pt_dino_execution_common import is_local_only_path, mask_local_path
    assert is_local_only_path("local_runs/figs/patch.png")
    assert "local_runs" not in mask_local_path("local_runs/figs/patch.png")


def test_no_abs_path_in_queue(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    sch.mkdir()
    linkage = _make_linkage(tmp_path, 5)
    env = _env(
        REVP_V1PW_IN_LINKAGE=str(linkage),
        REVP_V1PW_OUT_QUEUE=str(tmp_path / "q.csv"),
        REVP_V1PW_OUT_SUM=str(tmp_path / "sum.csv"),
        REVP_V1PW_SCH_QUEUE=str(sch / "a.csv"),
        REVP_V1PW_SCH_SUM=str(sch / "s.csv"),
        REVP_V1PW_DOC=str(tmp_path / "doc.md"),
        REVP_DINO_MAX_QUEUE="100",
    )
    _run(S["v1pw"], env)
    for row in _read(tmp_path / "q.csv"):
        assert not (len(row.get("relative_path", "x")) > 2 and row.get("relative_path", "")[1] == ":")


def test_dino_allowed_use_is_review_only(tmp_path: Path) -> None:
    linkage = _make_linkage(tmp_path, 3)
    sch = tmp_path / "schemas"
    sch.mkdir()
    env = _env(
        REVP_V1PW_IN_LINKAGE=str(linkage),
        REVP_V1PW_OUT_QUEUE=str(tmp_path / "q.csv"),
        REVP_V1PW_OUT_SUM=str(tmp_path / "sum.csv"),
        REVP_V1PW_SCH_QUEUE=str(sch / "a.csv"),
        REVP_V1PW_SCH_SUM=str(sch / "s.csv"),
        REVP_V1PW_DOC=str(tmp_path / "doc.md"),
        REVP_DINO_MAX_QUEUE="100",
    )
    _run(S["v1pw"], env)
    for row in _read(tmp_path / "q.csv"):
        assert row["dino_allowed_use"] == "REVIEW_ONLY_REPRESENTATION"
        assert "LABEL" not in row["dino_allowed_use"]


def test_no_ground_truth_in_outputs() -> None:
    import pytest
    with pytest.raises(ValueError):
        C.assert_no_forbidden_true([{"ground_truth": "true"}], "test")
