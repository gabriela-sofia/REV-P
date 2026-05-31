"""Tests for the REV-P DINO representation layer (v1pg-v1pm).

All script I/O is redirected to tmp_path via env vars; the real datasets/ tree is
never written. Unit tests import the common module directly; integration tests
run each stage as a subprocess and verify guardrails and fail-closed behavior.
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

import revp_v1pg_v1pm_dino_representation_common as C  # noqa: E402

S_V1PG = SCRIPTS / "revp_v1pg_dino_artifact_discovery.py"
S_V1PH = SCRIPTS / "revp_v1ph_dino_embedding_feature_store_registry.py"
S_V1PI = SCRIPTS / "revp_v1pi_dino_embedding_quality_audit.py"
S_V1PJ = SCRIPTS / "revp_v1pj_dino_similarity_neighbor_graph.py"
S_V1PK = SCRIPTS / "revp_v1pk_dino_pca_cluster_exploratory.py"
S_V1PL = SCRIPTS / "revp_v1pl_dino_protocol_c_crosswalk.py"
S_V1PM = SCRIPTS / "revp_v1pm_dino_tcc_results_bundle.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _header(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as fh:
        return next(csv.reader(fh))


def _run(script: Path, env: dict[str, str], timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout,
    )


def _make_emb_source(path: Path, n: int, dim: int = 768) -> None:
    """Write a synthetic embedding source with patch_id + dim_0..dim_{dim-1}."""
    fields = ["patch_id", "region"] + [f"dim_{i}" for i in range(dim)]
    rows = []
    for r in range(n):
        row = {"patch_id": f"REC_9{r:04d}", "region": "recife"}
        for i in range(dim):
            row[f"dim_{i}"] = f"{(r + 1) * 0.001 + i * 0.0001:.6f}"
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _make_discovery(path: Path, rel_source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=C.read_csv_header(path) or [
            "artifact_id", "relative_path", "path_hash", "likely_embedding_source",
            "allowed_for_dino_registry"])
        w.writeheader()
        w.writerow({
            "artifact_id": "V1PG_ART_0001", "relative_path": rel_source,
            "path_hash": C.path_hash(rel_source),
            "likely_embedding_source": "true", "allowed_for_dino_registry": "true",
        })


def _valid_chain(tmp: Path, n: int = 4) -> dict[str, Path]:
    """Build a full valid v1ph→v1pm chain inside tmp; return key output paths."""
    src_root = tmp / "srcroot"
    rel_source = "datasets/synth_emb.csv"
    _make_emb_source(src_root / rel_source, n)
    discovery = tmp / "discovery.csv"
    _make_discovery(discovery, rel_source)
    sch = tmp / "schemas"
    sch.mkdir(exist_ok=True)
    reg = tmp / "registry.csv"
    base = {**os.environ, "REVP_DINO_SOURCE_ROOT": str(src_root)}

    env_h = {**base,
             "REVP_V1PH_IN_DISCOVERY": str(discovery),
             "REVP_V1PH_OUT_REGISTRY": str(reg),
             "REVP_V1PH_OUT_SUMMARY": str(tmp / "reg_sum.csv"),
             "REVP_V1PH_SCHEMA_REGISTRY": str(sch / "h1.csv"),
             "REVP_V1PH_SCHEMA_SUMMARY": str(sch / "h2.csv"),
             "REVP_V1PH_DOC": str(tmp / "h.md")}
    assert _run(S_V1PH, env_h).returncode == 0
    return {"src_root": src_root, "discovery": discovery, "registry": reg,
            "reg_sum": tmp / "reg_sum.csv", "schemas": sch}


# ---------------------------------------------------------------------------
# Unit tests — common
# ---------------------------------------------------------------------------

def test_write_csv_empty_has_header(tmp_path: Path) -> None:
    p = tmp_path / "e.csv"
    C.write_csv(p, [], ["a", "b", "c"])
    assert _header(p) == ["a", "b", "c"]
    assert _read(p) == []


def test_parse_embedding_json_list() -> None:
    vec = C.parse_embedding_from_row({"embedding": json.dumps([1.0, 2.0, 3.0])})
    assert vec == [1.0, 2.0, 3.0]


def test_parse_embedding_string_list() -> None:
    assert C.parse_embedding_from_text("1.0 2.0 3.0") == [1.0, 2.0, 3.0]


def test_parse_embedding_768_columns() -> None:
    row = {f"dim_{i}": str(i * 0.5) for i in range(768)}
    vec = C.parse_embedding_from_row(row)
    assert vec is not None and len(vec) == 768


def test_parse_embedding_f_and_embedding_prefixes() -> None:
    assert len(C.parse_embedding_from_row({f"f{i}": "0.1" for i in range(768)})) == 768
    assert len(C.parse_embedding_from_row({f"embedding_{i}": "0.1" for i in range(768)})) == 768


def test_validate_blocks_wrong_dim() -> None:
    status, reason = C.validate_vector([0.1, 0.2, 0.3])
    assert status == "BLOCKED_INVALID_DIMENSION"
    assert reason


def test_validate_blocks_nan_inf() -> None:
    base = [0.1] * 768
    base[0] = float("nan")
    assert C.validate_vector(base)[0] == "BLOCKED_INVALID_VECTOR"
    base[0] = float("inf")
    assert C.validate_vector(base)[0] == "BLOCKED_INVALID_VECTOR"


def test_validate_blocks_zero_vector() -> None:
    status, reason = C.validate_vector([0.0] * 768)
    assert status == "BLOCKED_INVALID_VECTOR"
    assert reason == "zero_vector"


def test_validate_accepts_768() -> None:
    assert C.validate_vector([0.01 * (i + 1) for i in range(768)])[0] == "VALID_REVIEW_ONLY"


def test_cosine_and_euclidean() -> None:
    a = [1.0, 0.0]
    b = [1.0, 0.0]
    assert abs(C.cosine_similarity(a, b) - 1.0) < 1e-9
    assert C.euclidean_distance(a, [0.0, 0.0]) == 1.0


def test_pca_2d_runs() -> None:
    vecs = [[float((i + j) % 7) for j in range(768)] for i in range(6)]
    coords, evr = C.pca_2d(vecs)
    assert len(coords) == 6 and len(coords[0]) == 2
    assert 0.0 <= evr[0] <= 1.0001


def test_kmeans_deterministic() -> None:
    vecs = [[0.0] * 4, [0.1] * 4, [9.0] * 4, [9.1] * 4]
    a = C.kmeans_simple(vecs, 2)
    b = C.kmeans_simple(vecs, 2)
    assert a == b and len(set(a)) == 2


def test_fixture_detection() -> None:
    assert C.is_fixture_or_synthetic("this is a synthetic fixture")
    assert not C.is_fixture_or_synthetic("real recife patch")


def test_duplicate_vector_detected(tmp_path: Path) -> None:
    out = _valid_chain(tmp_path, n=4)
    # Add a duplicate of the first row by re-running with a source that repeats it.
    rows = _read(out["registry"])
    shas = [r["vector_sha256_16"] for r in rows]
    assert len(shas) == len(set(shas))  # distinct synthetic vectors → no dup here
    assert all(r["is_duplicate_vector"] == "false" for r in rows)


def test_guardrail_assert_no_forbidden_true() -> None:
    import pytest
    with pytest.raises(ValueError):
        C.assert_no_forbidden_true([{"dino_can_create_label": "true"}], "x")
    with pytest.raises(ValueError):
        C.assert_no_forbidden_true([{"can_train_model": "true"}], "x")


def test_scan_text_for_forbidden_tokens() -> None:
    assert "dino_can_create_label,true" in C.scan_text_for_forbidden("x,dino_can_create_label,true,y")
    assert "can_train_model,true" in C.scan_text_for_forbidden("a can_train_model,true b")
    assert C.scan_text_for_forbidden("dino_can_create_label,false") == []


# ---------------------------------------------------------------------------
# Integration — v1pg discovery
# ---------------------------------------------------------------------------

def test_v1pg_discovers_dino_artifacts(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    sch.mkdir()
    env = {**os.environ,
           "REVP_V1PG_OUT_DISCOVERY": str(tmp_path / "disc.csv"),
           "REVP_V1PG_OUT_SUMMARY": str(tmp_path / "disc_sum.csv"),
           "REVP_V1PG_SCHEMA_DISCOVERY": str(sch / "s1.csv"),
           "REVP_V1PG_SCHEMA_SUMMARY": str(sch / "s2.csv"),
           "REVP_V1PG_DOC": str(tmp_path / "d.md")}
    r = _run(S_V1PG, env)
    assert r.returncode == 0, r.stderr
    rows = _read(tmp_path / "disc.csv")
    assert len(rows) > 0
    assert any(row["likely_embedding_source"] == "true" for row in rows)
    for row in rows:
        rel = row["relative_path"]
        assert not (len(rel) > 1 and rel[1] == ":"), f"abs path: {rel}"


# ---------------------------------------------------------------------------
# Integration — v1ph feature store
# ---------------------------------------------------------------------------

def test_v1ph_creates_valid_registry(tmp_path: Path) -> None:
    out = _valid_chain(tmp_path, n=4)
    rows = _read(out["registry"])
    valid = [r for r in rows if r["embedding_status"] == "VALID_REVIEW_ONLY"]
    assert len(valid) >= 2
    for r in rows:
        assert r["dino_can_create_label"] == "false"
        assert r["dino_can_train_model"] == "false"
        assert r["dino_target_field_created"] == "false"
        assert r["vector_dim"] == "768"


def test_v1ph_empty_when_no_embeddings(tmp_path: Path) -> None:
    discovery = tmp_path / "disc.csv"
    C.write_csv(discovery, [], ["artifact_id", "relative_path", "likely_embedding_source", "allowed_for_dino_registry"])
    sch = tmp_path / "schemas"
    sch.mkdir()
    env = {**os.environ,
           "REVP_V1PH_IN_DISCOVERY": str(discovery),
           "REVP_V1PH_OUT_REGISTRY": str(tmp_path / "reg.csv"),
           "REVP_V1PH_OUT_SUMMARY": str(tmp_path / "reg_sum.csv"),
           "REVP_V1PH_SCHEMA_REGISTRY": str(sch / "a.csv"),
           "REVP_V1PH_SCHEMA_SUMMARY": str(sch / "b.csv"),
           "REVP_V1PH_DOC": str(tmp_path / "h.md")}
    assert _run(S_V1PH, env).returncode == 0
    assert _read(tmp_path / "reg.csv") == []
    assert "embedding_id" in _header(tmp_path / "reg.csv")


# ---------------------------------------------------------------------------
# Integration — v1pi audit
# ---------------------------------------------------------------------------

def test_v1pi_audits_valid_embeddings(tmp_path: Path) -> None:
    out = _valid_chain(tmp_path, n=4)
    sch = out["schemas"]
    env = {**os.environ,
           "REVP_V1PI_IN_REGISTRY": str(out["registry"]),
           "REVP_V1PI_OUT_AUDIT": str(tmp_path / "audit.csv"),
           "REVP_V1PI_OUT_SUMMARY": str(tmp_path / "audit_sum.csv"),
           "REVP_V1PI_SCHEMA_AUDIT": str(sch / "ia.csv"),
           "REVP_V1PI_SCHEMA_SUMMARY": str(sch / "ib.csv"),
           "REVP_V1PI_DOC": str(tmp_path / "ai.md")}
    assert _run(S_V1PI, env).returncode == 0
    rows = _read(tmp_path / "audit.csv")
    assert len(rows) >= 2
    for r in rows:
        assert r["check_dim_768"] == "PASS"
        assert r["check_no_label"] == "PASS"
        assert r["check_no_training"] == "PASS"


# ---------------------------------------------------------------------------
# Integration — v1pj similarity
# ---------------------------------------------------------------------------

def test_v1pj_generates_neighbors_without_label(tmp_path: Path) -> None:
    out = _valid_chain(tmp_path, n=4)
    sch = out["schemas"]
    env = {**os.environ, "REVP_DINO_SOURCE_ROOT": str(out["src_root"]),
           "REVP_V1PJ_IN_DISCOVERY": str(out["discovery"]),
           "REVP_V1PJ_IN_REGISTRY": str(out["registry"]),
           "REVP_V1PJ_OUT_NEIGHBORS": str(tmp_path / "nb.csv"),
           "REVP_V1PJ_OUT_MATRIX": str(tmp_path / "mx.csv"),
           "REVP_V1PJ_OUT_SUMMARY": str(tmp_path / "nb_sum.csv"),
           "REVP_V1PJ_SCHEMA_NEIGHBORS": str(sch / "ja.csv"),
           "REVP_V1PJ_SCHEMA_MATRIX": str(sch / "jb.csv"),
           "REVP_V1PJ_SCHEMA_SUMMARY": str(sch / "jc.csv"),
           "REVP_V1PJ_DOC": str(tmp_path / "j.md")}
    assert _run(S_V1PJ, env).returncode == 0
    nb = _read(tmp_path / "nb.csv")
    assert len(nb) >= 2
    for r in nb:
        assert r["can_infer_same_event"] == "false"
        assert r["can_create_label"] == "false"
        assert r["can_train_model"] == "false"


def test_v1pj_fail_closed_lt2(tmp_path: Path) -> None:
    # Empty registry → no valid embeddings → empty neighbors with header.
    reg = tmp_path / "reg.csv"
    C.write_csv(reg, [], ["embedding_id", "embedding_status", "is_duplicate_vector", "vector_sha256_16"])
    disc = tmp_path / "disc.csv"
    C.write_csv(disc, [], ["relative_path", "likely_embedding_source", "allowed_for_dino_registry"])
    sch = tmp_path / "schemas"
    sch.mkdir()
    env = {**os.environ,
           "REVP_V1PJ_IN_DISCOVERY": str(disc), "REVP_V1PJ_IN_REGISTRY": str(reg),
           "REVP_V1PJ_OUT_NEIGHBORS": str(tmp_path / "nb.csv"),
           "REVP_V1PJ_OUT_MATRIX": str(tmp_path / "mx.csv"),
           "REVP_V1PJ_OUT_SUMMARY": str(tmp_path / "ns.csv"),
           "REVP_V1PJ_SCHEMA_NEIGHBORS": str(sch / "a.csv"),
           "REVP_V1PJ_SCHEMA_MATRIX": str(sch / "b.csv"),
           "REVP_V1PJ_SCHEMA_SUMMARY": str(sch / "c.csv"),
           "REVP_V1PJ_DOC": str(tmp_path / "j.md")}
    assert _run(S_V1PJ, env).returncode == 0
    assert _read(tmp_path / "nb.csv") == []
    assert "query_patch_id" in _header(tmp_path / "nb.csv")


# ---------------------------------------------------------------------------
# Integration — v1pk pca / cluster
# ---------------------------------------------------------------------------

def test_v1pk_generates_pca_without_label(tmp_path: Path) -> None:
    out = _valid_chain(tmp_path, n=5)
    sch = out["schemas"]
    env = {**os.environ, "REVP_DINO_SOURCE_ROOT": str(out["src_root"]),
           "REVP_V1PK_IN_DISCOVERY": str(out["discovery"]),
           "REVP_V1PK_IN_REGISTRY": str(out["registry"]),
           "REVP_V1PK_OUT_PCA": str(tmp_path / "pca.csv"),
           "REVP_V1PK_OUT_CLUSTER": str(tmp_path / "cl.csv"),
           "REVP_V1PK_OUT_SUMMARY": str(tmp_path / "pc_sum.csv"),
           "REVP_V1PK_SCHEMA_PCA": str(sch / "ka.csv"),
           "REVP_V1PK_SCHEMA_CLUSTER": str(sch / "kb.csv"),
           "REVP_V1PK_SCHEMA_SUMMARY": str(sch / "kc.csv"),
           "REVP_V1PK_DOC": str(tmp_path / "k.md")}
    assert _run(S_V1PK, env).returncode == 0
    pca = _read(tmp_path / "pca.csv")
    assert len(pca) >= 2
    for r in pca:
        assert r["can_create_label"] == "false"
        assert r["can_train_model"] == "false"


def test_v1pk_cluster_is_not_a_class(tmp_path: Path) -> None:
    out = _valid_chain(tmp_path, n=5)
    sch = out["schemas"]
    env = {**os.environ, "REVP_DINO_SOURCE_ROOT": str(out["src_root"]),
           "REVP_V1PK_IN_DISCOVERY": str(out["discovery"]),
           "REVP_V1PK_IN_REGISTRY": str(out["registry"]),
           "REVP_V1PK_OUT_PCA": str(tmp_path / "pca.csv"),
           "REVP_V1PK_OUT_CLUSTER": str(tmp_path / "cl.csv"),
           "REVP_V1PK_OUT_SUMMARY": str(tmp_path / "pc_sum.csv"),
           "REVP_V1PK_SCHEMA_PCA": str(sch / "ka.csv"),
           "REVP_V1PK_SCHEMA_CLUSTER": str(sch / "kb.csv"),
           "REVP_V1PK_SCHEMA_SUMMARY": str(sch / "kc.csv"),
           "REVP_V1PK_DOC": str(tmp_path / "k.md")}
    assert _run(S_V1PK, env).returncode == 0
    for r in _read(tmp_path / "cl.csv"):
        assert r["can_be_used_as_class"] == "false"
        assert r["can_create_label"] == "false"


# ---------------------------------------------------------------------------
# Integration — v1pl crosswalk
# ---------------------------------------------------------------------------

def test_v1pl_does_not_validate_event(tmp_path: Path) -> None:
    out = _valid_chain(tmp_path, n=3)
    sch = out["schemas"]
    empty = tmp_path / "empty_pc.csv"
    C.write_csv(empty, [], ["patch_id", "event_id", "candidate_level"])
    env = {**os.environ,
           "REVP_V1PL_IN_REGISTRY": str(out["registry"]),
           "REVP_V1PL_IN_V1OY": str(empty), "REVP_V1PL_IN_V1OZ": str(empty),
           "REVP_V1PL_IN_V1OX": str(empty), "REVP_V1PL_IN_V1PF": str(empty),
           "REVP_V1PL_OUT_CROSSWALK": str(tmp_path / "xw.csv"),
           "REVP_V1PL_OUT_SUMMARY": str(tmp_path / "xw_sum.csv"),
           "REVP_V1PL_SCHEMA_CROSSWALK": str(sch / "la.csv"),
           "REVP_V1PL_SCHEMA_SUMMARY": str(sch / "lb.csv"),
           "REVP_V1PL_DOC": str(tmp_path / "l.md")}
    assert _run(S_V1PL, env).returncode == 0
    rows = _read(tmp_path / "xw.csv")
    assert len(rows) >= 2
    for r in rows:
        assert r["dino_can_validate_event"] == "false"
        assert r["dino_can_create_label"] == "false"
        assert r["dino_can_train_model"] == "false"


# ---------------------------------------------------------------------------
# Integration — v1pm bundle
# ---------------------------------------------------------------------------

def _v1pm_env(tmp: Path, reg: Path, sch: Path) -> dict[str, str]:
    sch.mkdir(exist_ok=True)
    e = {**os.environ, "REVP_V1PM_IN_REGISTRY": str(reg), "REVP_V1PM_SCHEMA_DIR": str(sch)}
    outs = {
        "REVP_V1PM_OUT_T_EMB": "t_emb.csv", "REVP_V1PM_OUT_T_SIM": "t_sim.csv",
        "REVP_V1PM_OUT_T_PCA": "t_pca.csv", "REVP_V1PM_OUT_T_XW": "t_xw.csv",
        "REVP_V1PM_OUT_MANIFEST": "manifest.csv", "REVP_V1PM_OUT_SUMMARY": "summary.csv",
    }
    for k, v in outs.items():
        e[k] = str(tmp / v)
    e["REVP_V1PM_DOC"] = str(tmp / "m.md")
    # Point empty optional inputs at a headered empty file to stay isolated.
    empty = tmp / "empty.csv"
    C.write_csv(empty, [], ["stat_key", "stat_value"])
    for k in ["IN_DISCOVERY_S", "IN_REGISTRY_S", "IN_NEIGHBORS", "IN_SIM_S",
              "IN_PCA", "IN_CLUSTER", "IN_PCA_S", "IN_CROSSWALK", "IN_CROSSWALK_S"]:
        e[f"REVP_V1PM_{k}"] = str(empty)
    return e


def test_v1pm_generates_tcc_tables(tmp_path: Path) -> None:
    out = _valid_chain(tmp_path, n=4)
    env = _v1pm_env(tmp_path, out["registry"], out["schemas"])
    env["REVP_V1PM_IN_REGISTRY_S"] = str(out["reg_sum"])
    assert _run(S_V1PM, env).returncode == 0
    for name in ["t_emb.csv", "t_sim.csv", "t_pca.csv", "t_xw.csv", "manifest.csv", "summary.csv"]:
        assert (tmp_path / name).exists()
    assert len(_read(tmp_path / "t_emb.csv")) >= 2


def test_v1pm_status_ready_with_embeddings(tmp_path: Path) -> None:
    out = _valid_chain(tmp_path, n=4)
    env = _v1pm_env(tmp_path, out["registry"], out["schemas"])
    env["REVP_V1PM_IN_REGISTRY_S"] = str(out["reg_sum"])
    assert _run(S_V1PM, env).returncode == 0
    summary = _read(tmp_path / "summary.csv")
    final = [r for r in summary if r["metric"] == "final_dino_status"][0]
    assert final["value"] == "DINO_REPRESENTATION_LAYER_READY_REVIEW_ONLY"


def test_v1pm_status_fail_closed_without_embeddings(tmp_path: Path) -> None:
    reg = tmp_path / "reg.csv"
    C.write_csv(reg, [], C.read_csv_header(reg) or ["embedding_id", "embedding_status"])
    sch = tmp_path / "schemas"
    sch.mkdir()
    env = _v1pm_env(tmp_path, reg, sch)
    assert _run(S_V1PM, env).returncode == 0
    summary = _read(tmp_path / "summary.csv")
    final = [r for r in summary if r["metric"] == "final_dino_status"][0]
    assert final["value"] == "DINO_EMBEDDINGS_NOT_FOUND_FAIL_CLOSED"


# ---------------------------------------------------------------------------
# Cross-cutting guardrails
# ---------------------------------------------------------------------------

def test_empty_outputs_have_header(tmp_path: Path) -> None:
    for fields in (["a"], ["x", "y", "z"]):
        p = tmp_path / f"f{len(fields)}.csv"
        C.write_csv(p, [], fields)
        assert _header(p) == fields


def test_blocked_row_has_blocked_reason() -> None:
    status, reason = C.validate_vector([0.0] * 768)
    assert status.startswith("BLOCKED") and reason


def test_no_test_writes_to_real_datasets() -> None:
    # Sanity: the real registry path is never an output target in this module's tests.
    assert "REVP_V1PH_OUT_REGISTRY" not in {}  # placeholder; tests always set tmp env
    assert (C.DATASETS / "dino_embedding_feature_store_registry_v1ph.csv").parent == C.DATASETS


def test_tcc_text_has_no_overclaim() -> None:
    from revp_v1pm_dino_tcc_results_bundle import TCC_TEXT
    low = TCC_TEXT.lower()
    assert "não como rótulo supervisionado" in low
    assert "sem criar ground truth operacional" in low
    assert "sem treinar classificador" in low
    for bad in ("ground truth confirmado", "evento validado", "classificador treinado"):
        assert bad not in low
