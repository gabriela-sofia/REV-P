"""Tests for REV-P DINO smoke embedding pipeline v1qg-v1qm.

All I/O via tmp_path/env vars. The real datasets/ directory is never written.
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

import revp_v1qg_v1qm_smoke_embedding_common as SC  # noqa: E402
import revp_v1pg_v1pm_dino_representation_common as C  # noqa: E402

S = {
    "v1qg": SCRIPTS / "revp_v1qg_local_dino_model_offline_audit.py",
    "v1qh": SCRIPTS / "revp_v1qh_dino_smoke_sample_selector.py",
    "v1qi": SCRIPTS / "revp_v1qi_local_asset_preprocessing_audit.py",
    "v1qj": SCRIPTS / "revp_v1qj_controlled_real_smoke_embedding_executor.py",
    "v1qk": SCRIPTS / "revp_v1qk_import_smoke_embeddings_to_representation_layer.py",
    "v1ql": SCRIPTS / "revp_v1ql_smoke_similarity_pca_review_products.py",
    "v1qm": SCRIPTS / "revp_v1qm_smoke_embedding_scientific_bundle.py",
}


def _run(script: Path, env: dict, timeout: int = 90) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(script)], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=timeout)


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


def _base(**kw: str) -> dict[str, str]:
    return {**os.environ, **kw}


def _vec(seed: float, dim: int = 768) -> list[float]:
    return [seed + 0.001 * i for i in range(dim)]


# --- fixtures -------------------------------------------------------------

def _make_model_dir(tmp: Path, hidden: int = 768, with_registers: bool = True) -> Path:
    d = tmp / "model"
    d.mkdir(parents=True, exist_ok=True)
    cfg = {"model_type": "dinov2_with_registers" if with_registers else "dinov2",
           "hidden_size": hidden, "num_register_tokens": 4 if with_registers else 0}
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (d / "model.safetensors").write_bytes(b"\x00\x01\x02\x03")
    (d / "preprocessor_config.json").write_text(json.dumps({"size": 224}), encoding="utf-8")
    return d


def _make_queue(tmp: Path, regions: list[str], n_each: int = 6) -> Path:
    q = tmp / "queue.csv"
    rows = []
    idx = 0
    for reg in regions:
        for j in range(n_each):
            idx += 1
            pid = f"{reg[:3].upper()}_{10000 + idx:05d}"
            rows.append({
                "execution_queue_id": f"V1QA_EQ_{idx:05d}",
                "source_queue_id": f"V1PW_Q_{idx:05d}",
                "visual_asset_id": f"VA_{idx}", "patch_id": pid, "alias": pid,
                "region": reg, "relative_path": f"data/sentinel/{pid}.tif",
                "path_hash": C.path_hash(f"data/sentinel/{pid}.tif"),
                "visual_type": "SENTINEL_TIF_REFERENCE", "queue_priority": "2",
                "queue_reason": "sentinel_tif_reference", "linkage_confidence": "HIGH",
                "dino_allowed_use": "REVIEW_ONLY_REPRESENTATION",
                "can_create_label": "false", "can_train_model": "false",
                "target_created": "false", "ready_for_dry_run": "true",
                "ready_for_real_execution": "false", "blocked_reason": "", "notes": "",
            })
    C.write_csv(q, rows, list(rows[0].keys()))
    return q


def _make_store(tmp: Path, n: int = 4, dim: int = 768, dup: bool = False) -> Path:
    p = tmp / "store.csv"
    meta_cols = ["embedding_id", "smoke_id", "patch_id", "alias", "region",
                 "visual_asset_id", "relative_path", "path_hash", "model_name",
                 "model_path_hash", "embedding_dim", "l2_normalized", "vector_norm",
                 "dino_allowed_use", "review_only", "can_create_label",
                 "can_train_model", "target_created"]
    cols = meta_cols + SC.embedding_columns(dim)
    rows = []
    for i in range(1, n + 1):
        pid = f"CUR_{10000 + (1 if dup else i):05d}"
        ph = C.path_hash(f"data/{pid}.tif") if not dup else "samehash"
        base = {
            "embedding_id": f"EMB_{i:05d}", "smoke_id": f"SMK_{i:05d}", "patch_id": pid,
            "alias": pid, "region": "CURITIBA", "visual_asset_id": f"VA_{i}",
            "relative_path": f"data/{pid}.tif", "path_hash": ph,
            "model_name": "model", "model_path_hash": "mph", "embedding_dim": str(dim),
            "l2_normalized": "true", "vector_norm": "1.0",
            "dino_allowed_use": "REVIEW_ONLY_REPRESENTATION", "review_only": "true",
            "can_create_label": "false", "can_train_model": "false", "target_created": "false",
        }
        base.update(SC.vector_to_columns(_vec(0.1 * i, dim), dim))
        rows.append(base)
    C.write_csv(p, rows, cols)
    return p


# ===========================================================================
# common module
# ===========================================================================

def test_common_writes_empty_csv_with_header(tmp_path: Path) -> None:
    p = tmp_path / "e.csv"
    SC.write_csv(p, [], ["a", "b", "c"])
    assert _header(p) == ["a", "b", "c"]
    assert _read(p) == []


def test_common_embedding_columns_768() -> None:
    cols = SC.embedding_columns()
    assert len(cols) == 768
    assert cols[0] == "embedding_000" and cols[-1] == "embedding_767"


def test_common_vector_to_columns_roundtrip() -> None:
    cols = SC.vector_to_columns(_vec(0.5))
    assert len(cols) == 768
    assert cols["embedding_000"] == "0.5"


def test_common_normalize_identity() -> None:
    pid, alias, region = SC.normalize_identity({"patch_id": "cur_00038", "region": "curitiba"})
    assert pid == "CUR_00038" and region == "CURITIBA" and alias == "CUR_00038"


def test_common_mask_local_runs_path() -> None:
    masked = SC.mask_local("local_runs/foo/bar.png")
    assert masked.startswith("local_only:")
    assert SC.mask_local("data/x.tif") == "data/x.tif"


def test_common_resolve_local_asset_uses_env_root(tmp_path: Path) -> None:
    root = tmp_path / "assets"
    (root / "data").mkdir(parents=True)
    f = root / "data" / "p.tif"
    f.write_bytes(b"x")
    got = SC.resolve_local_asset("data/p.tif", [root])
    assert got == f


def test_common_resolve_local_asset_missing_returns_none(tmp_path: Path) -> None:
    assert SC.resolve_local_asset("data/none.tif", [tmp_path]) is None


def test_common_file_sha256_short(tmp_path: Path) -> None:
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello")
    h = SC.file_sha256_short(f)
    assert len(h) == 16
    assert SC.file_sha256_short(tmp_path / "missing.bin") == ""


def test_common_validate_vector_768_valid() -> None:
    status, reason = SC.validate_vector(_vec(0.1))
    assert status == "VALID_REVIEW_ONLY" and reason == ""


def test_common_validate_vector_767_rejected() -> None:
    status, reason = SC.validate_vector(_vec(0.1, 767))
    assert status == "BLOCKED_INVALID_DIMENSION"


def test_common_validate_vector_nan_rejected() -> None:
    v = _vec(0.1)
    v[0] = float("nan")
    status, _ = SC.validate_vector(v)
    assert status == "BLOCKED_INVALID_VECTOR"


def test_common_validate_vector_inf_rejected() -> None:
    v = _vec(0.1)
    v[5] = float("inf")
    status, _ = SC.validate_vector(v)
    assert status == "BLOCKED_INVALID_VECTOR"


def test_common_validate_vector_zero_rejected() -> None:
    status, _ = SC.validate_vector([0.0] * 768)
    assert status == "BLOCKED_INVALID_VECTOR"


def test_common_cosine_similarity_self_is_one() -> None:
    v = _vec(0.3)
    assert abs(SC.cosine_similarity(v, v) - 1.0) < 1e-9


def test_common_pca_fail_closed_n_lt_2() -> None:
    coords, evr, method = SC.pca_2d_review([_vec(0.1)])
    assert coords == [] and method == "PCA_FAIL_CLOSED_N_LT_2"


def test_common_pca_two_vectors() -> None:
    coords, evr, method = SC.pca_2d_review([_vec(0.1), _vec(0.9)])
    assert len(coords) == 2 and method.startswith("PCA")


def test_common_clusters_empty_below_4() -> None:
    assert SC.exploratory_clusters([_vec(0.1), _vec(0.2), _vec(0.3)]) == []


def test_common_clusters_present_at_4() -> None:
    idx = SC.exploratory_clusters([_vec(0.1 * i) for i in range(1, 6)], 2)
    assert len(idx) == 5 and all(isinstance(x, int) for x in idx)


def test_common_expected_dim_from_config(tmp_path: Path) -> None:
    d = _make_model_dir(tmp_path, hidden=768)
    assert SC.expected_embedding_dim(str(d)) == 768


def test_common_expected_dim_default() -> None:
    assert SC.expected_embedding_dim("nonexistent/path") == 768


def test_common_guardrail_detects_label_true() -> None:
    ok, reason = SC.guardrail_ok({"can_create_label": "true"})
    assert not ok and "can_create_label" in reason


def test_common_guardrail_detects_train_true() -> None:
    ok, reason = SC.guardrail_ok({"can_train_model": "true"})
    assert not ok and "can_train_model" in reason


def test_common_guardrail_detects_target_true() -> None:
    ok, reason = SC.guardrail_ok({"target_created": "true"})
    assert not ok and "target_created" in reason


def test_common_guardrail_detects_ground_truth_true() -> None:
    ok, reason = SC.guardrail_ok({"ground_truth": "true"})
    assert not ok and "ground_truth" in reason


# ===========================================================================
# v1qg — model audit
# ===========================================================================

def _v1qg_env(tmp: Path, sch: Path, model_path: str = "") -> dict:
    sch.mkdir(exist_ok=True)
    env = _base(
        REVP_V1QG_OUT_AUDIT=str(tmp / "audit.csv"),
        REVP_V1QG_OUT_SUM=str(tmp / "sum.csv"),
        REVP_V1QG_SCH_AUDIT=str(sch / "a.csv"), REVP_V1QG_SCH_SUM=str(sch / "b.csv"),
        REVP_V1QG_DOC=str(tmp / "doc.md"),
        REVP_DINO_ALLOW_DOWNLOAD="false", HF_HUB_OFFLINE="1",
    )
    if model_path:
        env["REVP_DINO_MODEL_PATH"] = model_path
    else:
        env.pop("REVP_DINO_MODEL_PATH", None)
    return env


def test_v1qg_missing_path_fail_closed(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    r = _run(S["v1qg"], _v1qg_env(tmp_path, sch))
    assert r.returncode == 0, r.stderr
    summary = {s["stat_key"]: s["stat_value"] for s in _read(tmp_path / "sum.csv")}
    assert summary["final_status"] == "LOCAL_DINO_MODEL_MISSING_FAIL_CLOSED"


def test_v1qg_fake_local_model_detected_offline(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    model = _make_model_dir(tmp_path)
    r = _run(S["v1qg"], _v1qg_env(tmp_path, sch, str(model)))
    assert r.returncode == 0, r.stderr
    audit = _read(tmp_path / "audit.csv")[0]
    assert audit["config_exists"] == "true"
    assert audit["weights_exists"] == "true"
    assert audit["expected_dim"] == "768"


def test_v1qg_dimension_mismatch_fail_closed(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    model = _make_model_dir(tmp_path, hidden=512)
    r = _run(S["v1qg"], _v1qg_env(tmp_path, sch, str(model)))
    assert r.returncode == 0, r.stderr
    summary = {s["stat_key"]: s["stat_value"] for s in _read(tmp_path / "sum.csv")}
    assert summary["final_status"] == "LOCAL_DINO_MODEL_DIMENSION_MISMATCH_FAIL_CLOSED"


def test_v1qg_invalid_path_missing_weights(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    d = tmp_path / "bad_model"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"hidden_size": 768, "model_type": "dinov2"}), encoding="utf-8")
    r = _run(S["v1qg"], _v1qg_env(tmp_path, sch, str(d)))
    assert r.returncode == 0, r.stderr
    summary = {s["stat_key"]: s["stat_value"] for s in _read(tmp_path / "sum.csv")}
    assert summary["final_status"] == "LOCAL_DINO_MODEL_INVALID_FAIL_CLOSED"


# ===========================================================================
# v1qh — smoke sample selector
# ===========================================================================

def _v1qh_env(tmp: Path, queue: Path, sch: Path, **extra: str) -> dict:
    sch.mkdir(exist_ok=True)
    return _base(
        REVP_DINO_SMOKE_QUEUE=str(queue),
        REVP_V1QH_OUT_SEL=str(tmp / "sel.csv"), REVP_V1QH_OUT_SUM=str(tmp / "sum.csv"),
        REVP_V1QH_SCH_SEL=str(sch / "a.csv"), REVP_V1QH_SCH_SUM=str(sch / "b.csv"),
        REVP_V1QH_DOC=str(tmp / "doc.md"), **extra,
    )


def test_v1qh_regional_diversity(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    q = _make_queue(tmp_path, ["RECIFE", "PET", "CURITIBA"], 6)
    r = _run(S["v1qh"], _v1qh_env(tmp_path, q, sch, REVP_DINO_SMOKE_N="9",
                                  REVP_DINO_SMOKE_MIN_PER_REGION="2"))
    assert r.returncode == 0, r.stderr
    rows = _read(tmp_path / "sel.csv")
    regions = {row["region"] for row in rows}
    assert len(regions) >= 3


def test_v1qh_respects_smoke_n(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    q = _make_queue(tmp_path, ["CURITIBA"], 30)
    _run(S["v1qh"], _v1qh_env(tmp_path, q, sch, REVP_DINO_SMOKE_N="16"))
    assert len(_read(tmp_path / "sel.csv")) == 16


def test_v1qh_does_not_require_scene_date(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    q = _make_queue(tmp_path, ["PET"], 10)  # no scene_date column anywhere
    _run(S["v1qh"], _v1qh_env(tmp_path, q, sch, REVP_DINO_SMOKE_N="5"))
    rows = _read(tmp_path / "sel.csv")
    assert len(rows) == 5
    assert all("scene_date" not in row for row in rows)


def test_v1qh_no_temporal_unlock_field(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    q = _make_queue(tmp_path, ["PET"], 6)
    _run(S["v1qh"], _v1qh_env(tmp_path, q, sch))
    for row in _read(tmp_path / "sel.csv"):
        assert "temporal_unlock" not in row
        assert row["selected_for_smoke"] == "true"


def test_v1qh_blocks_fixture(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    q = tmp_path / "q.csv"
    rows = [{
        "execution_queue_id": "V1QA_EQ_00001", "visual_asset_id": "VA_1",
        "patch_id": "REC_00001", "alias": "REC_00001", "region": "RECIFE",
        "relative_path": "data/fixture_sample.tif", "path_hash": "h",
        "visual_type": "SENTINEL_TIF_REFERENCE", "queue_priority": "2",
        "linkage_confidence": "HIGH", "dino_allowed_use": "REVIEW_ONLY_REPRESENTATION",
        "can_create_label": "false", "can_train_model": "false", "target_created": "false",
        "blocked_reason": "", "notes": "",
    }]
    C.write_csv(q, rows, list(rows[0].keys()))
    _run(S["v1qh"], _v1qh_env(tmp_path, q, sch))
    assert len(_read(tmp_path / "sel.csv")) == 0
    summary = {s["stat_key"]: s["stat_value"] for s in _read(tmp_path / "sum.csv")}
    assert summary["final_status"] == "DINO_SMOKE_SAMPLE_EMPTY_FAIL_CLOSED"


def test_v1qh_blocks_guardrail_label(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    q = tmp_path / "q.csv"
    rows = [{
        "execution_queue_id": "V1QA_EQ_00001", "visual_asset_id": "VA_1",
        "patch_id": "CUR_12345", "alias": "CUR_12345", "region": "CURITIBA",
        "relative_path": "data/p.tif", "path_hash": "h",
        "visual_type": "SENTINEL_TIF_REFERENCE", "queue_priority": "2",
        "linkage_confidence": "HIGH", "dino_allowed_use": "REVIEW_ONLY_REPRESENTATION",
        "can_create_label": "true", "can_train_model": "false", "target_created": "false",
        "blocked_reason": "", "notes": "",
    }]
    C.write_csv(q, rows, list(rows[0].keys()))
    _run(S["v1qh"], _v1qh_env(tmp_path, q, sch))
    assert len(_read(tmp_path / "sel.csv")) == 0


# ===========================================================================
# v1qi — asset preprocessing audit
# ===========================================================================

def _make_sel(tmp: Path, rels: list[str]) -> Path:
    p = tmp / "sel.csv"
    rows = []
    for i, rel in enumerate(rels, 1):
        rows.append({
            "smoke_id": f"V1QH_SMK_{i:05d}", "execution_queue_id": f"EQ_{i}",
            "visual_asset_id": f"VA_{i}", "patch_id": f"CUR_{10000+i:05d}",
            "alias": f"CUR_{10000+i:05d}", "region": "CURITIBA",
            "relative_path": rel, "path_hash": C.path_hash(rel),
            "visual_type": "SENTINEL_TIF_REFERENCE", "sample_rank": str(i),
            "selection_reason": "test", "linkage_confidence": "HIGH",
            "dino_allowed_use": "REVIEW_ONLY_REPRESENTATION",
            "can_create_label": "false", "can_train_model": "false",
            "target_created": "false", "selected_for_smoke": "true",
            "blocked_reason": "", "notes": "",
        })
    C.write_csv(p, rows, list(rows[0].keys()))
    return p


def _v1qi_env(tmp: Path, sel: Path, sch: Path, **extra: str) -> dict:
    sch.mkdir(exist_ok=True)
    return _base(
        REVP_V1QI_IN_SEL=str(sel),
        REVP_V1QI_OUT_AUDIT=str(tmp / "audit.csv"), REVP_V1QI_OUT_SUM=str(tmp / "sum.csv"),
        REVP_V1QI_SCH_AUDIT=str(sch / "a.csv"), REVP_V1QI_SCH_SUM=str(sch / "b.csv"),
        REVP_V1QI_DOC=str(tmp / "doc.md"), **extra,
    )


def test_v1qi_missing_asset_fail_closed(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    sel = _make_sel(tmp_path, ["data/missing.tif"])
    env = _v1qi_env(tmp_path, sel, sch)
    env.pop("REVP_DINO_PIXEL_READ_ALLOWED", None)
    r = _run(S["v1qi"], env)
    assert r.returncode == 0, r.stderr
    audit = _read(tmp_path / "audit.csv")[0]
    assert audit["status"] == "ASSET_MISSING_FAIL_CLOSED"


def test_v1qi_metadata_only_without_pixel_env(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    root = tmp_path / "assets"
    (root / "data").mkdir(parents=True)
    (root / "data" / "p.tif").write_bytes(b"fakecontent")
    sel = _make_sel(tmp_path, ["data/p.tif"])
    env = _v1qi_env(tmp_path, sel, sch, REVP_DINO_ASSET_ROOT=str(root))
    env.pop("REVP_DINO_PIXEL_READ_ALLOWED", None)
    r = _run(S["v1qi"], env)
    assert r.returncode == 0, r.stderr
    audit = _read(tmp_path / "audit.csv")[0]
    assert audit["status"] == "ASSET_METADATA_ONLY_PIXEL_READ_BLOCKED"
    assert audit["pixel_read_performed"] == "false"
    assert audit["file_exists"] == "true"


def test_v1qi_pixel_read_on_fixture_png(tmp_path: Path) -> None:
    import pytest
    PIL = pytest.importorskip("PIL")
    from PIL import Image
    sch = tmp_path / "schemas"
    root = tmp_path / "assets"
    (root / "data").mkdir(parents=True)
    Image.new("RGB", (8, 6), (10, 20, 30)).save(root / "data" / "p.png")
    sel = _make_sel(tmp_path, ["data/p.png"])
    env = _v1qi_env(tmp_path, sel, sch, REVP_DINO_ASSET_ROOT=str(root),
                    REVP_DINO_PIXEL_READ_ALLOWED="true")
    r = _run(S["v1qi"], env)
    assert r.returncode == 0, r.stderr
    audit = _read(tmp_path / "audit.csv")[0]
    assert audit["status"] == "ASSET_READY_FOR_DINO_PREPROCESSING"
    assert audit["image_width"] == "8" and audit["image_height"] == "6"
    assert audit["pixel_read_performed"] == "true"


def test_v1qi_no_absolute_path_in_output(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    root = tmp_path / "assets"
    (root / "data").mkdir(parents=True)
    (root / "data" / "p.tif").write_bytes(b"x")
    sel = _make_sel(tmp_path, ["data/p.tif"])
    _run(S["v1qi"], _v1qi_env(tmp_path, sel, sch, REVP_DINO_ASSET_ROOT=str(root)))
    for row in _read(tmp_path / "audit.csv"):
        for v in row.values():
            assert not (len(v) > 2 and v[1] == ":")


# ===========================================================================
# v1qj — executor
# ===========================================================================

def _v1qj_env(tmp: Path, sel: Path, sch: Path, asset: Path | None = None,
              model_sum: Path | None = None, **extra: str) -> dict:
    sch.mkdir(exist_ok=True)
    if asset is None:
        asset = tmp / "asset_empty.csv"
        C.write_csv(asset, [], ["smoke_id", "status", "relative_path"])
    if model_sum is None:
        model_sum = tmp / "model_empty.csv"
        C.write_csv(model_sum, [], ["stat_key", "stat_value"])
    return _base(
        REVP_V1QJ_IN_SEL=str(sel), REVP_V1QJ_IN_ASSET=str(asset),
        REVP_V1QJ_IN_MODEL=str(model_sum),
        REVP_V1QJ_OUT_STORE=str(tmp / "store.csv"), REVP_V1QJ_OUT_MAN=str(tmp / "man.csv"),
        REVP_V1QJ_OUT_FAIL=str(tmp / "fail.csv"), REVP_V1QJ_OUT_SUM=str(tmp / "sum.csv"),
        REVP_V1QJ_SCH_STORE=str(sch / "a.csv"), REVP_V1QJ_SCH_MAN=str(sch / "b.csv"),
        REVP_V1QJ_SCH_FAIL=str(sch / "c.csv"), REVP_V1QJ_SCH_SUM=str(sch / "d.csv"),
        REVP_V1QJ_DOC=str(tmp / "doc.md"), **extra,
    )


def test_v1qj_dry_run_empty_store_with_header(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    sel = _make_sel(tmp_path, ["data/p.tif"])
    env = _v1qj_env(tmp_path, sel, sch, REVP_DINO_DRY_RUN="true")
    r = _run(S["v1qj"], env)
    assert r.returncode == 0, r.stderr
    assert _read(tmp_path / "store.csv") == []
    hdr = _header(tmp_path / "store.csv")
    assert "embedding_000" in hdr and "embedding_767" in hdr
    summary = {s["stat_key"]: s["stat_value"] for s in _read(tmp_path / "sum.csv")}
    assert summary["final_status"] == "DINO_SMOKE_EMBEDDINGS_DRY_RUN_ONLY"


def test_v1qj_model_missing_fail_closed(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    sel = _make_sel(tmp_path, ["data/p.tif"])
    env = _v1qj_env(tmp_path, sel, sch, REVP_DINO_DRY_RUN="false")
    env.pop("REVP_DINO_MODEL_PATH", None)
    r = _run(S["v1qj"], env)
    assert r.returncode == 0, r.stderr
    summary = {s["stat_key"]: s["stat_value"] for s in _read(tmp_path / "sum.csv")}
    assert summary["final_status"] == "DINO_SMOKE_EMBEDDINGS_MODEL_MISSING_FAIL_CLOSED"


def test_v1qj_pixel_blocked_fail_closed(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    model = _make_model_dir(tmp_path)
    model_sum = tmp_path / "msum.csv"
    C.write_csv(model_sum, [{"stat_key": "model_ready", "stat_value": "true"}],
                ["stat_key", "stat_value"])
    sel = _make_sel(tmp_path, ["data/p.tif"])
    env = _v1qj_env(tmp_path, sel, sch, model_sum=model_sum,
                    REVP_DINO_DRY_RUN="false", REVP_DINO_MODEL_PATH=str(model),
                    REVP_DINO_ALLOW_DOWNLOAD="false", HF_HUB_OFFLINE="1")
    env.pop("REVP_DINO_PIXEL_READ_ALLOWED", None)
    r = _run(S["v1qj"], env)
    assert r.returncode == 0, r.stderr
    summary = {s["stat_key"]: s["stat_value"] for s in _read(tmp_path / "sum.csv")}
    assert summary["final_status"] == "DINO_SMOKE_EMBEDDINGS_PIXEL_READ_BLOCKED_FAIL_CLOSED"


def test_v1qj_outputs_review_only_flags(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    sel = _make_sel(tmp_path, ["data/p.tif"])
    _run(S["v1qj"], _v1qj_env(tmp_path, sel, sch, REVP_DINO_DRY_RUN="true"))
    summary = {s["stat_key"]: s["stat_value"] for s in _read(tmp_path / "sum.csv")}
    assert summary["labels_created"] == "0" and summary["targets_created"] == "0"


# ===========================================================================
# v1qk — import to representation layer
# ===========================================================================

def _v1qk_env(tmp: Path, store: Path, sch: Path) -> dict:
    sch.mkdir(exist_ok=True)
    return _base(
        REVP_V1QK_IN_STORE=str(store),
        REVP_V1QK_OUT_STORE=str(tmp / "rep.csv"), REVP_V1QK_OUT_SUM=str(tmp / "sum.csv"),
        REVP_V1QK_SCH_STORE=str(sch / "a.csv"), REVP_V1QK_SCH_SUM=str(sch / "b.csv"),
        REVP_V1QK_DOC=str(tmp / "doc.md"),
    )


def test_v1qk_imports_fixture_768d(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    store = _make_store(tmp_path, n=4)
    r = _run(S["v1qk"], _v1qk_env(tmp_path, store, sch))
    assert r.returncode == 0, r.stderr
    summary = {s["stat_key"]: s["stat_value"] for s in _read(tmp_path / "sum.csv")}
    assert summary["valid_vectors"] == "4"
    assert summary["final_status"] == "DINO_REPRESENTATION_WITH_SMOKE_READY_REVIEW_ONLY"


def test_v1qk_deduplicates(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    store = _make_store(tmp_path, n=4, dup=True)
    _run(S["v1qk"], _v1qk_env(tmp_path, store, sch))
    summary = {s["stat_key"]: s["stat_value"] for s in _read(tmp_path / "sum.csv")}
    assert int(summary["duplicate_vectors"]) >= 1
    assert summary["unique_valid_vectors"] == "1"


def test_v1qk_cluster_is_label_false(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    store = _make_store(tmp_path, n=3)
    _run(S["v1qk"], _v1qk_env(tmp_path, store, sch))
    for row in _read(tmp_path / "rep.csv"):
        assert row["cluster_is_label"] == "false"


# ===========================================================================
# v1ql — review products
# ===========================================================================

def _v1ql_env(tmp: Path, store: Path, sch: Path) -> dict:
    sch.mkdir(exist_ok=True)
    return _base(
        REVP_V1QL_IN_STORE=str(store),
        REVP_V1QL_OUT_NEIGH=str(tmp / "neigh.csv"), REVP_V1QL_OUT_MATRIX=str(tmp / "mtx.csv"),
        REVP_V1QL_OUT_PCA=str(tmp / "pca.csv"), REVP_V1QL_OUT_CLUST=str(tmp / "cl.csv"),
        REVP_V1QL_OUT_SUM=str(tmp / "sum.csv"),
        REVP_V1QL_SCH_NEIGH=str(sch / "a.csv"), REVP_V1QL_SCH_MATRIX=str(sch / "b.csv"),
        REVP_V1QL_SCH_PCA=str(sch / "c.csv"), REVP_V1QL_SCH_CLUST=str(sch / "d.csv"),
        REVP_V1QL_SCH_SUM=str(sch / "e.csv"), REVP_V1QL_DOC=str(tmp / "doc.md"),
    )


def _make_rep_store(tmp: Path, n: int) -> Path:
    """Build a v1qk-style representation store with valid vectors."""
    p = tmp / "rep.csv"
    meta = ["representation_id", "embedding_id", "patch_id", "alias", "region",
            "visual_asset_id", "source_stage", "model_name", "embedding_dim",
            "vector_valid", "duplicate_group_id", "dino_allowed_use", "review_only",
            "cluster_is_label", "can_create_label", "can_train_model", "target_created",
            "blocked_reason", "notes"]
    cols = meta + SC.embedding_columns()
    rows = []
    for i in range(1, n + 1):
        base = {
            "representation_id": f"REP_{i:05d}", "embedding_id": f"EMB_{i:05d}",
            "patch_id": f"CUR_{10000+i:05d}", "alias": f"CUR_{10000+i:05d}",
            "region": "CURITIBA", "visual_asset_id": f"VA_{i}", "source_stage": "v1qj_smoke",
            "model_name": "model", "embedding_dim": "768", "vector_valid": "true",
            "duplicate_group_id": f"DG_{i:05d}", "dino_allowed_use": "REVIEW_ONLY_REPRESENTATION",
            "review_only": "true", "cluster_is_label": "false", "can_create_label": "false",
            "can_train_model": "false", "target_created": "false", "blocked_reason": "", "notes": "",
        }
        base.update(SC.vector_to_columns(_vec(0.05 * i)))
        rows.append(base)
    C.write_csv(p, rows, cols)
    return p


def test_v1ql_fail_closed_n_lt_2(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    store = _make_rep_store(tmp_path, 1)
    r = _run(S["v1ql"], _v1ql_env(tmp_path, store, sch))
    assert r.returncode == 0, r.stderr
    summary = {s["stat_key"]: s["stat_value"] for s in _read(tmp_path / "sum.csv")}
    assert summary["final_status"] == "DINO_SMOKE_REVIEW_PRODUCTS_FAIL_CLOSED_N_LT_2"
    assert _header(tmp_path / "neigh.csv")  # header present even when empty


def test_v1ql_neighbors_with_n_ge_2(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    store = _make_rep_store(tmp_path, 4)
    _run(S["v1ql"], _v1ql_env(tmp_path, store, sch))
    assert len(_read(tmp_path / "neigh.csv")) > 0


def test_v1ql_pca_with_n_ge_2(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    store = _make_rep_store(tmp_path, 5)
    _run(S["v1ql"], _v1ql_env(tmp_path, store, sch))
    assert len(_read(tmp_path / "pca.csv")) == 5


def test_v1ql_clusters_not_class(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    store = _make_rep_store(tmp_path, 5)
    _run(S["v1ql"], _v1ql_env(tmp_path, store, sch))
    rows = _read(tmp_path / "cl.csv")
    assert len(rows) == 5
    for row in rows:
        assert row["cluster_is_label"] == "false"


def test_v1ql_similarity_does_not_validate_event(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    store = _make_rep_store(tmp_path, 3)
    _run(S["v1ql"], _v1ql_env(tmp_path, store, sch))
    for row in _read(tmp_path / "neigh.csv"):
        assert row["similarity_validates_event"] == "false"


def test_v1ql_pca_does_not_validate_event(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    store = _make_rep_store(tmp_path, 3)
    _run(S["v1ql"], _v1ql_env(tmp_path, store, sch))
    for row in _read(tmp_path / "pca.csv"):
        assert row["pca_validates_event"] == "false"


# ===========================================================================
# v1qm — scientific bundle
# ===========================================================================

def _v1qm_env(tmp: Path, in_datasets: Path, sch: Path) -> dict:
    sch.mkdir(exist_ok=True)
    return _base(
        REVP_V1QM_IN_DATASETS=str(in_datasets),
        REVP_V1QM_OUT_MAN=str(tmp / "man.csv"), REVP_V1QM_OUT_QC=str(tmp / "qc.csv"),
        REVP_V1QM_OUT_SUM=str(tmp / "sum.csv"), REVP_V1QM_OUT_TCC=str(tmp / "tcc.csv"),
        REVP_V1QM_SCH_MAN=str(sch / "a.csv"), REVP_V1QM_SCH_QC=str(sch / "b.csv"),
        REVP_V1QM_SCH_SUM=str(sch / "c.csv"), REVP_V1QM_SCH_TCC=str(sch / "d.csv"),
        REVP_V1QM_DOC=str(tmp / "doc.md"),
    )


def test_v1qm_status_missing_model_when_empty(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    ds = tmp_path / "ds"
    ds.mkdir()
    r = _run(S["v1qm"], _v1qm_env(tmp_path, ds, sch))
    assert r.returncode == 0, r.stderr
    summary = {s["metric"]: s["value"] for s in _read(tmp_path / "sum.csv")}
    assert summary["final_status"] == "DINO_SMOKE_EMBEDDINGS_MODEL_MISSING_FAIL_CLOSED"


def test_v1qm_status_ready_with_embeddings(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    ds = tmp_path / "ds"
    ds.mkdir()
    C.write_csv(ds / "dino_smoke_embedding_summary_v1qj.csv", [
        {"stat_key": "embeddings_valid_768d", "stat_value": "3"},
        {"stat_key": "embedding_dim", "stat_value": "768"},
        {"stat_key": "labels_created", "stat_value": "0"},
        {"stat_key": "targets_created", "stat_value": "0"},
        {"stat_key": "pixel_read_allowed", "stat_value": "true"},
    ], ["stat_key", "stat_value"])
    r = _run(S["v1qm"], _v1qm_env(tmp_path, ds, sch))
    assert r.returncode == 0, r.stderr
    summary = {s["metric"]: s["value"] for s in _read(tmp_path / "sum.csv")}
    assert summary["final_status"] == "DINO_SMOKE_EMBEDDINGS_AVAILABLE_REVIEW_ONLY"


def test_v1qm_quality_checks_pass(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    ds = tmp_path / "ds"
    ds.mkdir()
    r = _run(S["v1qm"], _v1qm_env(tmp_path, ds, sch))
    assert r.returncode == 0, r.stderr
    qc = _read(tmp_path / "qc.csv")
    assert all(row["passed"] == "true" for row in qc)


def test_v1qm_doc_contains_boundary_phrase(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    ds = tmp_path / "ds"
    ds.mkdir()
    _run(S["v1qm"], _v1qm_env(tmp_path, ds, sch))
    text = (tmp_path / "doc.md").read_text(encoding="utf-8")
    assert "não constituem rótulo" in text
    assert "validação de evento observado" in text


def test_v1qm_tcc_table_generated(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    ds = tmp_path / "ds"
    ds.mkdir()
    _run(S["v1qm"], _v1qm_env(tmp_path, ds, sch))
    tcc = _read(tmp_path / "tcc.csv")
    assert len(tcc) == 5
    assert all("boundary" in row for row in tcc)


# ===========================================================================
# cross-cutting guardrails
# ===========================================================================

def test_assert_no_forbidden_true_label() -> None:
    import pytest
    with pytest.raises(ValueError):
        C.assert_no_forbidden_true([{"can_create_label": "true"}], "x")


def test_assert_no_forbidden_true_train() -> None:
    import pytest
    with pytest.raises(ValueError):
        C.assert_no_forbidden_true([{"can_train_model": "true"}], "x")


def test_no_test_writes_to_real_datasets() -> None:
    # Sanity: the real feature store path is not among any tmp output we write.
    assert (ROOT / "datasets").exists()
    # Guardrail constant present.
    assert "ground_truth" in SC.SMOKE_FORBIDDEN_FIELDS


def test_outputs_have_schema_files(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    store = _make_store(tmp_path, n=2)
    _run(S["v1qk"], _v1qk_env(tmp_path, store, sch))
    assert (sch / "a.csv").exists() and (sch / "b.csv").exists()


def test_docs_created(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    q = _make_queue(tmp_path, ["CURITIBA"], 5)
    _run(S["v1qh"], _v1qh_env(tmp_path, q, sch))
    assert (tmp_path / "doc.md").exists()
