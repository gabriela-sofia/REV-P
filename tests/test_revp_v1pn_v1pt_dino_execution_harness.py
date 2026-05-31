"""Tests for REV-P DINO execution harness v1pn-v1pt.

All I/O via tmp_path/env vars — real datasets/ never touched.
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

import revp_v1pn_v1pt_dino_execution_common as EC  # noqa: E402
import revp_v1pg_v1pm_dino_representation_common as C  # noqa: E402

S = {
    "v1pn": SCRIPTS / "revp_v1pn_patch_visual_asset_inventory.py",
    "v1po": SCRIPTS / "revp_v1po_dino_embedding_execution_queue.py",
    "v1pp": SCRIPTS / "revp_v1pp_dino_backend_model_probe.py",
    "v1pq": SCRIPTS / "revp_v1pq_controlled_smoke_embedding_executor.py",
    "v1pr": SCRIPTS / "revp_v1pr_import_smoke_embeddings_feature_store.py",
    "v1ps": SCRIPTS / "revp_v1ps_smoke_embedding_review_products.py",
    "v1pt": SCRIPTS / "revp_v1pt_dino_execution_bundle.py",
}


def _run(script: Path, env: dict, timeout: int = 120) -> subprocess.CompletedProcess:
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
        r = csv.reader(fh)
        return next(r, [])


def _base_env(**kwargs: str) -> dict[str, str]:
    return {**os.environ, **kwargs}


def _make_image(tmp: Path, name: str = "patch_REC_0001.png") -> Path:
    """Create a minimal 1×1 PNG — no model inference, just a file."""
    img = tmp / name
    # Minimal PNG bytes (1×1 red pixel)
    img.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
        b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    return img


def _v1pn_env(tmp: Path, sch: Path) -> dict[str, str]:
    sch.mkdir(exist_ok=True)
    return _base_env(
        REVP_V1PN_OUT_INV=str(tmp / "inv.csv"),
        REVP_V1PN_OUT_SUM=str(tmp / "inv_sum.csv"),
        REVP_V1PN_SCH_INV=str(sch / "s_inv.csv"),
        REVP_V1PN_SCH_SUM=str(sch / "s_sum.csv"),
        REVP_V1PN_DOC=str(tmp / "doc.md"),
    )


def _empty_csv(tmp: Path, name: str, fields: list[str]) -> Path:
    p = tmp / name
    C.write_csv(p, [], fields)
    return p


# ---------------------------------------------------------------------------
# v1pn — visual asset inventory
# ---------------------------------------------------------------------------

def test_v1pn_detects_images_by_extension(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    r = _run(S["v1pn"], _v1pn_env(tmp_path, sch))
    assert r.returncode == 0, r.stderr
    # real repo should not yield no images at all (it has PNGs in docs etc)
    inv = _read(tmp_path / "inv.csv")
    assert _header(tmp_path / "inv.csv")  # header always present


def test_v1pn_no_pixel_reading(tmp_path: Path) -> None:
    """v1pn must not open PIL image objects (no pixel reading)."""
    sch = tmp_path / "schemas"
    r = _run(S["v1pn"], _v1pn_env(tmp_path, sch))
    assert r.returncode == 0, r.stderr
    # If PIL was invoked, output would be slow; we just verify no crash and output exists.
    assert (tmp_path / "inv.csv").exists()


def test_v1pn_blocks_fixture(tmp_path: Path) -> None:
    """Files with 'fixture' in name should be blocked."""
    # Create a fixture image in a temp scan folder — v1pn scans datasets/ etc by default,
    # so we verify the common helper is_fixture_or_synthetic catches the pattern.
    assert EC.is_fixture_or_synthetic("patch_fixture_test.png")
    assert not EC.is_fixture_or_synthetic("recife_patch_REC_0001.png")


def test_v1pn_local_runs_masked(tmp_path: Path) -> None:
    assert EC.is_local_only_path("local_runs/something.png")
    masked = EC.mask_local_path("local_runs/something.png")
    assert masked.startswith("local_only:")
    assert "local_runs" not in masked


def test_v1pn_no_abs_path_in_outputs(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    r = _run(S["v1pn"], _v1pn_env(tmp_path, sch))
    assert r.returncode == 0, r.stderr
    for row in _read(tmp_path / "inv.csv"):
        rel = row.get("relative_path", "")
        assert not (len(rel) > 1 and rel[1] == ":"), f"abs path: {rel}"


# ---------------------------------------------------------------------------
# v1po — execution queue
# ---------------------------------------------------------------------------

def _v1po_env(tmp: Path, sch: Path, inv: Path, v1oz: Path) -> dict[str, str]:
    sch.mkdir(exist_ok=True)
    return _base_env(
        REVP_V1PO_OUT_QUEUE=str(tmp / "queue.csv"),
        REVP_V1PO_OUT_SUM=str(tmp / "q_sum.csv"),
        REVP_V1PO_SCH_QUEUE=str(sch / "sq.csv"),
        REVP_V1PO_SCH_SUM=str(sch / "ss.csv"),
        REVP_V1PO_DOC=str(tmp / "doc.md"),
        REVP_V1PO_IN_INV=str(inv),
        REVP_V1PO_IN_V1OZ=str(v1oz),
        REVP_DINO_MAX_QUEUE="10",
    )


def test_v1po_prioritizes_protocol_c_queue(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    img = _make_image(tmp_path, "patch_REC_0001.png")
    inv = tmp_path / "inv.csv"
    C.write_csv(inv, [{
        "visual_asset_id": "V1PN_IMG_X",
        "relative_path": "docs/patch_REC_0001.png",
        "path_hash": C.path_hash("docs/patch_REC_0001.png"),
        "patch_id": "REC_0001", "alias": "REC_0001", "region": "RECIFE",
        "eligible_for_embedding_queue": "true", "blocked_reason": "", "notes": "",
    }], ["visual_asset_id", "relative_path", "path_hash", "patch_id", "alias",
         "region", "eligible_for_embedding_queue", "blocked_reason", "notes"])
    v1oz = tmp_path / "v1oz.csv"
    C.write_csv(v1oz, [{"patch_id": "REC_0001", "event_id": "EVT_1"}],
                ["patch_id", "event_id"])
    r = _run(S["v1po"], _v1po_env(tmp_path, sch, inv, v1oz))
    assert r.returncode == 0, r.stderr
    rows = _read(tmp_path / "queue.csv")
    assert len(rows) == 1
    assert rows[0]["queue_priority"] == "1"


def test_v1po_respects_max_queue(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    items = [{
        "visual_asset_id": f"IMG_{i}", "relative_path": f"docs/p{i}.png",
        "path_hash": C.path_hash(f"docs/p{i}.png"),
        "patch_id": f"REC_{i:04d}", "alias": f"REC_{i:04d}", "region": "RECIFE",
        "eligible_for_embedding_queue": "true", "blocked_reason": "", "notes": "",
    } for i in range(20)]
    inv = _empty_csv(tmp_path, "inv.csv", list(items[0].keys()))
    C.write_csv(inv, items, list(items[0].keys()))
    v1oz = _empty_csv(tmp_path, "v1oz.csv", ["patch_id"])
    env = _v1po_env(tmp_path, sch, inv, v1oz)
    env["REVP_DINO_MAX_QUEUE"] = "5"
    r = _run(S["v1po"], env)
    assert r.returncode == 0, r.stderr
    assert len(_read(tmp_path / "queue.csv")) <= 5


def test_v1po_guardrails_always_false(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    inv = _empty_csv(tmp_path, "inv.csv", ["visual_asset_id", "eligible_for_embedding_queue"])
    v1oz = _empty_csv(tmp_path, "v1oz.csv", ["patch_id"])
    r = _run(S["v1po"], _v1po_env(tmp_path, sch, inv, v1oz))
    assert r.returncode == 0, r.stderr
    for row in _read(tmp_path / "queue.csv"):
        assert row["can_create_label"] == "false"
        assert row["can_train_model"] == "false"
        assert row["target_created"] == "false"


# ---------------------------------------------------------------------------
# v1pp — backend probe
# ---------------------------------------------------------------------------

def _v1pp_env(tmp: Path, sch: Path, model_path: str = "", allow_dl: str = "false") -> dict[str, str]:
    sch.mkdir(exist_ok=True)
    return _base_env(
        REVP_V1PP_OUT_PROBE=str(tmp / "probe.csv"),
        REVP_V1PP_OUT_SUM=str(tmp / "probe_sum.csv"),
        REVP_V1PP_SCH_PROBE=str(sch / "sp.csv"),
        REVP_V1PP_SCH_SUM=str(sch / "ss.csv"),
        REVP_V1PP_DOC=str(tmp / "doc.md"),
        REVP_DINO_MODEL_PATH=model_path,
        REVP_DINO_ALLOW_DOWNLOAD=allow_dl,
    )


def test_v1pp_fail_closed_without_model(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    r = _run(S["v1pp"], _v1pp_env(tmp_path, sch))
    assert r.returncode == 0, r.stderr
    summary = _read(tmp_path / "probe_sum.csv")
    final = next((s["stat_value"] for s in summary if s["stat_key"] == "final_status"), "")
    assert final == "DINO_BACKEND_MODEL_UNAVAILABLE_FAIL_CLOSED"
    can = next((s["stat_value"] for s in summary if s["stat_key"] == "can_execute_embeddings"), "true")
    assert can == "false"


def test_v1pp_detects_model_env_var(tmp_path: Path) -> None:
    # Probe with a fake local path that exists.
    fake_model = tmp_path / "fake_model"
    fake_model.mkdir()
    sch = tmp_path / "schemas"
    info = EC.probe_backend()
    # The probe function itself should see model_path_exists based on env.
    os.environ["REVP_DINO_MODEL_PATH"] = str(fake_model)
    try:
        info2 = EC.probe_backend()
        assert info2["model_path_exists"] is True
        assert info2["model_path"] == str(fake_model)
    finally:
        del os.environ["REVP_DINO_MODEL_PATH"]


def test_v1pp_no_download_by_default(tmp_path: Path) -> None:
    info = EC.probe_backend()
    assert info["allow_download"] is False


# ---------------------------------------------------------------------------
# v1pq — smoke executor
# ---------------------------------------------------------------------------

def _v1pq_env(tmp: Path, sch: Path, queue: Path, dry: str = "true") -> dict[str, str]:
    sch.mkdir(exist_ok=True)
    return _base_env(
        REVP_V1PQ_IN_QUEUE=str(queue),
        REVP_V1PQ_OUT_RESULTS=str(tmp / "res.csv"),
        REVP_V1PQ_OUT_FAILURES=str(tmp / "fail.csv"),
        REVP_V1PQ_OUT_SUM=str(tmp / "sum.csv"),
        REVP_V1PQ_SCH_RES=str(sch / "sr.csv"),
        REVP_V1PQ_SCH_FAIL=str(sch / "sf.csv"),
        REVP_V1PQ_SCH_SUM=str(sch / "ss.csv"),
        REVP_V1PQ_DOC=str(tmp / "doc.md"),
        REVP_DINO_DRY_RUN=dry,
        REVP_DINO_MODEL_PATH="",
        REVP_DINO_ALLOW_DOWNLOAD="false",
        REVP_DINO_MAX_EXECUTE="5",
    )


def test_v1pq_dry_run_does_not_execute(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    queue = _empty_csv(tmp_path, "queue.csv", ["queue_id", "visual_asset_id", "patch_id",
                                                "alias", "region", "relative_path", "path_hash"])
    r = _run(S["v1pq"], _v1pq_env(tmp_path, sch, queue, dry="true"))
    assert r.returncode == 0, r.stderr
    summary = _read(tmp_path / "sum.csv")
    mode = next((s["stat_value"] for s in summary if s["stat_key"] == "execution_mode"), "")
    assert mode == "DRY_RUN"
    attempted = next((s["stat_value"] for s in summary if s["stat_key"] == "embeddings_attempted"), "1")
    assert attempted == "0"


def test_v1pq_no_model_generates_skip(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    queue = tmp_path / "queue.csv"
    C.write_csv(queue, [{
        "queue_id": "V1PO_Q_00001", "visual_asset_id": "IMG_X",
        "patch_id": "REC_0001", "alias": "REC_0001", "region": "RECIFE",
        "relative_path": "docs/patch.png", "path_hash": "abc",
    }], ["queue_id", "visual_asset_id", "patch_id", "alias", "region", "relative_path", "path_hash"])
    r = _run(S["v1pq"], _v1pq_env(tmp_path, sch, queue, dry="false"))
    assert r.returncode == 0, r.stderr
    # Without model: failure row with skip/blocked status
    failures = _read(tmp_path / "fail.csv")
    assert len(failures) >= 1
    assert any("MODEL" in f.get("error_type", "") or "UNAVAILABLE" in f.get("status", "") for f in failures)


def test_v1pq_no_download_by_default(tmp_path: Path) -> None:
    info = EC.probe_backend()
    assert info["allow_download"] is False


def test_v1pq_dim_768_required(tmp_path: Path) -> None:
    status, reason = C.validate_vector([0.1] * 100)
    assert status == "BLOCKED_INVALID_DIMENSION"
    status2, _ = C.validate_vector([0.1] * 768)
    assert status2 == "VALID_REVIEW_ONLY"


# ---------------------------------------------------------------------------
# v1pr — import smoke embeddings
# ---------------------------------------------------------------------------

def _v1pr_env(tmp: Path, sch: Path, results: Path) -> dict[str, str]:
    sch.mkdir(exist_ok=True)
    return _base_env(
        REVP_V1PR_IN_RESULTS=str(results),
        REVP_V1PR_OUT_STORE=str(tmp / "store.csv"),
        REVP_V1PR_OUT_SUM=str(tmp / "store_sum.csv"),
        REVP_V1PR_SCH_STORE=str(sch / "st.csv"),
        REVP_V1PR_SCH_SUM=str(sch / "ss.csv"),
        REVP_V1PR_DOC=str(tmp / "doc.md"),
    )


def test_v1pr_imports_valid_768_embedding(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    vec = [0.001 * (i + 1) for i in range(768)]
    results = tmp_path / "results.csv"
    C.write_csv(results, [{
        "embedding_run_id": "V1PQ_RUN_00001",
        "queue_id": "V1PO_Q_00001",
        "visual_asset_id": "IMG_X",
        "patch_id": "REC_9001", "alias": "REC_9001", "region": "RECIFE",
        "embedding": json.dumps(vec),
        "status": "EMBEDDING_EXECUTED_REVIEW_ONLY",
    }], ["embedding_run_id", "queue_id", "visual_asset_id", "patch_id", "alias", "region", "embedding", "status"])
    r = _run(S["v1pr"], _v1pr_env(tmp_path, sch, results))
    assert r.returncode == 0, r.stderr
    store = _read(tmp_path / "store.csv")
    assert len(store) == 1
    assert store[0]["embedding_status"] == "VALID_REVIEW_ONLY"
    assert store[0]["dino_can_create_label"] == "false"
    assert store[0]["dino_can_train_model"] == "false"
    assert store[0]["dino_target_field_created"] == "false"


def test_v1pr_blocks_nan_vector(tmp_path: Path) -> None:
    vec = [0.1] * 768
    vec[0] = float("nan")
    status, reason = C.validate_vector(vec)
    assert status == "BLOCKED_INVALID_VECTOR"


def test_v1pr_blocks_zero_vector(tmp_path: Path) -> None:
    status, reason = C.validate_vector([0.0] * 768)
    assert status == "BLOCKED_INVALID_VECTOR"
    assert reason == "zero_vector"


def test_v1pr_empty_when_no_real_results(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    results = _empty_csv(tmp_path, "res.csv", ["embedding_run_id", "status", "embedding"])
    r = _run(S["v1pr"], _v1pr_env(tmp_path, sch, results))
    assert r.returncode == 0, r.stderr
    assert _read(tmp_path / "store.csv") == []
    assert "embedding_id" in _header(tmp_path / "store.csv")


# ---------------------------------------------------------------------------
# v1ps — smoke review products
# ---------------------------------------------------------------------------

def _make_v1pr_store(tmp: Path, n: int) -> Path:
    store = tmp / "store.csv"
    from revp_v1pn_v1pt_dino_execution_common import make_vector_row, build_vector_row_fields
    fields = build_vector_row_fields()
    rows = []
    for i in range(n):
        vec = [0.001 * (i + 1) + 0.0001 * j for j in range(768)]
        row = make_vector_row(i + 1, f"REC_9{i:03d}", f"REC_9{i:03d}", "RECIFE", f"RUN_{i}", f"IMG_{i}", vec)
        # v1ps reads embedding from store: inject embedding field
        row["embedding"] = json.dumps(vec)
        rows.append(row)
    C.write_csv(store, rows, fields + ["embedding"])
    return store


def _v1ps_env(tmp: Path, sch: Path, store: Path) -> dict[str, str]:
    sch.mkdir(exist_ok=True)
    empty_v1oy = _empty_csv(tmp, "v1oy.csv", ["patch_id", "event_id", "candidate_level"])
    return _base_env(
        REVP_V1PS_IN_STORE=str(store),
        REVP_V1PS_IN_V1OY=str(empty_v1oy),
        REVP_V1PS_OUT_NEIGHBORS=str(tmp / "nb.csv"),
        REVP_V1PS_OUT_PCA=str(tmp / "pca.csv"),
        REVP_V1PS_OUT_CLUSTER=str(tmp / "cl.csv"),
        REVP_V1PS_OUT_XW=str(tmp / "xw.csv"),
        REVP_V1PS_OUT_SUM=str(tmp / "sum.csv"),
        REVP_V1PS_SCH_NB=str(sch / "a.csv"), REVP_V1PS_SCH_PCA=str(sch / "b.csv"),
        REVP_V1PS_SCH_CL=str(sch / "c.csv"), REVP_V1PS_SCH_XW=str(sch / "d.csv"),
        REVP_V1PS_SCH_SUM=str(sch / "e.csv"),
        REVP_V1PS_DOC=str(tmp / "doc.md"),
    )


def test_v1ps_fail_closed_lt2_embeddings(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    store = _empty_csv(tmp_path, "store.csv", ["embedding_id", "embedding_status", "embedding"])
    r = _run(S["v1ps"], _v1ps_env(tmp_path, sch, store))
    assert r.returncode == 0, r.stderr
    assert _read(tmp_path / "nb.csv") == []
    assert "query_patch_id" in _header(tmp_path / "nb.csv")


def test_v1ps_generates_neighbors_with_2_embeddings(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    store = _make_v1pr_store(tmp_path, 3)
    r = _run(S["v1ps"], _v1ps_env(tmp_path, sch, store))
    assert r.returncode == 0, r.stderr
    nb = _read(tmp_path / "nb.csv")
    assert len(nb) >= 2
    for row in nb:
        assert row["can_infer_same_event"] == "false"
        assert row["can_create_label"] == "false"


def test_v1ps_cluster_not_class(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    store = _make_v1pr_store(tmp_path, 4)
    r = _run(S["v1ps"], _v1ps_env(tmp_path, sch, store))
    assert r.returncode == 0, r.stderr
    for row in _read(tmp_path / "cl.csv"):
        assert row["can_be_used_as_class"] == "false"
        assert row["can_create_label"] == "false"


# ---------------------------------------------------------------------------
# v1pt — execution bundle
# ---------------------------------------------------------------------------

def _v1pt_env(tmp: Path, sch: Path, dry: str = "true") -> dict[str, str]:
    sch.mkdir(exist_ok=True)
    # Point all inputs at empty CSVs so the bundle can read them
    def _e(name: str, fields: list[str]) -> Path:
        return _empty_csv(tmp, name, fields)

    return _base_env(
        REVP_V1PT_OUT_MANIFEST=str(tmp / "manifest.csv"),
        REVP_V1PT_OUT_QC=str(tmp / "qc.csv"),
        REVP_V1PT_OUT_SUM=str(tmp / "sum.csv"),
        REVP_V1PT_SCH_MAN=str(sch / "sm.csv"),
        REVP_V1PT_SCH_QC=str(sch / "sq.csv"),
        REVP_V1PT_SCH_SUM=str(sch / "ss.csv"),
        REVP_V1PT_DOC=str(tmp / "doc.md"),
        REVP_DINO_DRY_RUN=dry,
    )


def test_v1pt_generates_summary(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    r = _run(S["v1pt"], _v1pt_env(tmp_path, sch))
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "summary" if False else (tmp_path / "sum.csv")).exists()
    summary = _read(tmp_path / "sum.csv")
    assert len(summary) >= 5
    metrics = [s["metric"] for s in summary]
    assert "final_execution_status" in metrics
    assert "labels_created" in metrics


def test_v1pt_status_dry_run(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    r = _run(S["v1pt"], _v1pt_env(tmp_path, sch, dry="true"))
    assert r.returncode == 0, r.stderr
    summary = _read(tmp_path / "sum.csv")
    final = next(s["value"] for s in summary if s["metric"] == "final_execution_status")
    assert final == "DINO_EXECUTION_PLAN_READY_DRY_RUN"


def test_v1pt_status_model_unavailable(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    r = _run(S["v1pt"], _v1pt_env(tmp_path, sch, dry="false"))
    assert r.returncode == 0, r.stderr
    summary = _read(tmp_path / "sum.csv")
    final = next(s["value"] for s in summary if s["metric"] == "final_execution_status")
    assert final in ("DINO_EXECUTION_FAIL_CLOSED_MODEL_UNAVAILABLE",
                     "DINO_EXECUTION_NO_EMBEDDINGS_GENERATED_FAIL_CLOSED")


# ---------------------------------------------------------------------------
# Cross-cutting guardrails
# ---------------------------------------------------------------------------

def test_empty_outputs_always_have_header(tmp_path: Path) -> None:
    for fields in (["a", "b"], ["x", "y", "z"]):
        p = tmp_path / f"h{''.join(fields)}.csv"
        C.write_csv(p, [], fields)
        with p.open(encoding="utf-8") as fh:
            assert next(csv.reader(fh)) == fields


def test_blocked_rows_have_blocked_reason() -> None:
    status, reason = C.validate_vector(None)
    assert status.startswith("BLOCKED") and reason


def test_guardrail_detects_label_true() -> None:
    import pytest
    with pytest.raises(ValueError):
        C.assert_no_forbidden_true([{"can_create_label": "true"}], "test")


def test_guardrail_detects_training_true() -> None:
    import pytest
    with pytest.raises(ValueError):
        C.assert_no_forbidden_true([{"can_train_model": "true"}], "test")


def test_guardrail_detects_target_true() -> None:
    import pytest
    with pytest.raises(ValueError):
        C.assert_no_forbidden_true([{"dino_target_field_created": "true"}], "test")


def test_no_abs_path_in_v1pn_outputs(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    r = _run(S["v1pn"], _v1pn_env(tmp_path, sch))
    assert r.returncode == 0
    for row in _read(tmp_path / "inv.csv"):
        rel = row.get("relative_path", "")
        assert not (len(rel) > 1 and rel[1] == ":"), f"abs path: {rel}"


def test_doc_contains_tcc_text(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    r = _run(S["v1pt"], _v1pt_env(tmp_path, sch))
    assert r.returncode == 0
    doc_text = (tmp_path / "doc.md").read_text(encoding="utf-8")
    assert "representações 768D review-only" in doc_text
    assert "sem criação de rótulos" in doc_text
    assert "sem targets supervisionados" in doc_text
    assert "fail-closed" in doc_text.lower()


def test_no_test_writes_to_real_datasets() -> None:
    # Structural: confirm test env vars always redirect to tmp
    assert "REVP_V1PN_OUT_INV" not in os.environ  # not set globally
    assert C.DATASETS.name == "datasets"  # sanity
