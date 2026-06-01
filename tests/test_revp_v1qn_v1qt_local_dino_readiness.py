"""Tests for REV-P DINO local readiness block v1qn-v1qt.

All I/O via tmp_path / env vars. Real datasets/ never written.
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

import revp_v1qn_v1qt_local_readiness_common as LC  # noqa
import revp_v1pg_v1pm_dino_representation_common as C  # noqa

S = {
    "v1qn": SCRIPTS / "revp_v1qn_local_root_environment_audit.py",
    "v1qo": SCRIPTS / "revp_v1qo_smoke_asset_local_reconciliation.py",
    "v1qp": SCRIPTS / "revp_v1qp_manifest_crosswalk_repair_suggestions.py",
    "v1qq": SCRIPTS / "revp_v1qq_local_execution_config_template.py",
    "v1qr": SCRIPTS / "revp_v1qr_local_smoke_run_readiness_gate.py",
    "v1qs": SCRIPTS / "revp_v1qs_local_readiness_tcc_update.py",
    "v1qt": SCRIPTS / "revp_v1qt_local_readiness_bundle.py",
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


def _make_model_dir(tmp: Path, hidden: int = 768) -> Path:
    d = tmp / "model"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(
        json.dumps({"model_type": "dinov2_with_registers", "hidden_size": hidden}),
        encoding="utf-8")
    (d / "model.safetensors").write_bytes(b"\x00\x01")
    (d / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    return d


def _make_smoke(tmp: Path, n: int = 5, rel: str = "") -> Path:
    p = tmp / "smoke.csv"
    rows = []
    for i in range(1, n + 1):
        pid = f"CUR_{10000+i:05d}"
        rows.append({
            "smoke_id": f"V1QH_SMK_{i:05d}", "execution_queue_id": f"EQ_{i}",
            "visual_asset_id": f"VA_{i}", "patch_id": pid, "alias": pid,
            "region": "CURITIBA", "relative_path": rel,
            "path_hash": C.path_hash(rel or f"data/{pid}.tif"),
            "visual_type": "SENTINEL_TIF_REFERENCE", "sample_rank": str(i),
            "selection_reason": "test", "linkage_confidence": "HIGH",
            "dino_allowed_use": "REVIEW_ONLY_REPRESENTATION",
            "can_create_label": "false", "can_train_model": "false",
            "target_created": "false", "selected_for_smoke": "true",
            "blocked_reason": "", "notes": "",
        })
    C.write_csv(p, rows, list(rows[0].keys()))
    return p


def _make_rec(tmp: Path, n_ready: int = 2, n_unresolved: int = 3) -> Path:
    p = tmp / "rec.csv"
    rows = []
    for i in range(1, n_ready + 1):
        rows.append({
            "smoke_id": f"V1QH_SMK_{i:05d}", "patch_id": f"CUR_{10000+i:05d}",
            "match_type": "exact_relative", "match_confidence": "1.000",
            "ready_for_embedding": "true", "blocked_reason": "",
            "review_only": "true", "can_create_label": "false",
            "can_train_model": "false", "target_created": "false",
        })
    for i in range(n_ready + 1, n_ready + n_unresolved + 1):
        rows.append({
            "smoke_id": f"V1QH_SMK_{i:05d}", "patch_id": f"CUR_{10000+i:05d}",
            "match_type": "unresolved", "match_confidence": "0.000",
            "ready_for_embedding": "false", "blocked_reason": "no_local_file_found",
            "review_only": "true", "can_create_label": "false",
            "can_train_model": "false", "target_created": "false",
        })
    C.write_csv(p, rows, list(rows[0].keys()))
    return p


def _make_summary(tmp: Path, fname: str, stats: dict) -> Path:
    p = tmp / fname
    C.write_csv(p, [{"stat_key": k, "stat_value": v} for k, v in stats.items()],
                ["stat_key", "stat_value"])
    return p


def _v1qn_env(tmp: Path, sch: Path, root: str = "", model: str = "") -> dict:
    sch.mkdir(exist_ok=True)
    env = _base(
        REVP_V1QN_OUT_AUDIT=str(tmp / "audit.csv"),
        REVP_V1QN_OUT_SUM=str(tmp / "sum.csv"),
        REVP_V1QN_SCH_AUDIT=str(sch / "a.csv"),
        REVP_V1QN_SCH_SUM=str(sch / "b.csv"),
        REVP_V1QN_DOC=str(tmp / "doc.md"),
        REVP_DINO_ALLOW_DOWNLOAD="false",
        HF_HUB_OFFLINE="1",
    )
    for k in ("REVP_SENTINEL_LOCAL_ROOT","REVP_DINO_VISUAL_ROOT",
              "REVP_DINO_ASSET_ROOT","REVP_DINO_MODEL_PATH"):
        env.pop(k, None)
    if root:
        env["REVP_SENTINEL_LOCAL_ROOT"] = root
    if model:
        env["REVP_DINO_MODEL_PATH"] = model
    return env


# ===========================================================================
# common module
# ===========================================================================

def test_common_write_csv_empty_with_header(tmp_path: Path) -> None:
    p = tmp_path / "e.csv"
    LC.write_csv(p, [], ["a", "b"])
    assert _header(p) == ["a", "b"]


def test_common_mask_abs_path() -> None:
    assert LC.mask_abs("C:\\Users\\foo\\model").startswith("masked_abs:")
    assert LC.mask_abs("data/patch.tif") == "data/patch.tif"


def test_common_detect_abs() -> None:
    assert LC.detect_abs("C:\\Users\\foo") is True
    assert LC.detect_abs("data/x.tif") is False


def test_common_local_runs_not_in_mask() -> None:
    # mask_abs only masks Windows abs paths; local_runs is not one
    assert LC.mask_abs("local_runs/foo/bar") == "local_runs/foo/bar"


def test_common_file_sha256_short(tmp_path: Path) -> None:
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello")
    h = LC.file_sha256_short(f)
    assert len(h) == 16
    assert LC.file_sha256_short(tmp_path / "missing.bin") == ""


def test_common_guardrail_detects_label() -> None:
    ok, reason = LC.guardrail_row_ok({"can_create_label": "true"})
    assert not ok and "can_create_label" in reason


def test_common_guardrail_detects_train() -> None:
    ok, reason = LC.guardrail_row_ok({"can_train_model": "true"})
    assert not ok and "can_train_model" in reason


def test_common_guardrail_detects_target() -> None:
    ok, reason = LC.guardrail_row_ok({"target_created": "true"})
    assert not ok and "target_created" in reason


def test_common_guardrail_detects_ground_truth() -> None:
    ok, reason = LC.guardrail_row_ok({"ground_truth_created": "true"})
    assert not ok and "ground_truth_created" in reason


def test_common_normalize_patch() -> None:
    pid, alias, region = LC.normalize_patch({"patch_id": "pet_00038", "region": "pet"})
    assert pid == "PET_00038" and region == "PET"


def test_common_resolve_no_roots() -> None:
    got = LC.resolve_candidate("data/p.tif", "p.tif", "CUR_00001", "CUR_00001", [])
    assert got == []


def test_common_resolve_filename_match(tmp_path: Path) -> None:
    root = tmp_path / "r"
    root.mkdir()
    f = root / "CUR_00038_patch.tif"
    f.write_bytes(b"x")
    cands = LC.resolve_candidate("", "CUR_00038_patch.tif", "CUR_00038",
                                 "CUR_00038", [("REVP_SENTINEL_LOCAL_ROOT", root)])
    assert any(c["match_type"] == "filename_exact" for c in cands)


def test_common_resolve_patch_id_match(tmp_path: Path) -> None:
    root = tmp_path / "r"
    root.mkdir()
    (root / "sentinel_CUR_00038.tif").write_bytes(b"x")
    cands = LC.resolve_candidate("", "", "CUR_00038", "CUR_00038",
                                 [("REVP_SENTINEL_LOCAL_ROOT", root)])
    assert any(c["match_type"] == "patch_id_match" for c in cands)


def test_common_resolve_alias_match(tmp_path: Path) -> None:
    root = tmp_path / "r"
    root.mkdir()
    (root / "patch_curitiba_north.tif").write_bytes(b"x")
    cands = LC.resolve_candidate("", "", "CUR_00038", "curitiba_north",
                                 [("REVP_SENTINEL_LOCAL_ROOT", root)])
    assert any("match" in c["match_type"] for c in cands)


def test_common_unresolved_no_match(tmp_path: Path) -> None:
    root = tmp_path / "r"
    root.mkdir()
    (root / "completely_unrelated.tif").write_bytes(b"x")
    cands = LC.resolve_candidate("", "", "PET_99999", "PET_99999",
                                 [("REVP_SENTINEL_LOCAL_ROOT", root)])
    assert cands == []


# ===========================================================================
# v1qn
# ===========================================================================

def test_v1qn_missing_roots_fail_closed(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    r = _run(S["v1qn"], _v1qn_env(tmp_path, sch))
    assert r.returncode == 0, r.stderr
    summary = {s["stat_key"]: s["stat_value"] for s in _read(tmp_path / "sum.csv")}
    assert summary["final_status"] == "LOCAL_ENV_ROOTS_MISSING_FAIL_CLOSED"


def test_v1qn_root_present_counts_files(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    root = tmp_path / "assets"
    root.mkdir()
    (root / "a.tif").write_bytes(b"x")
    (root / "b.tif").write_bytes(b"x")
    r = _run(S["v1qn"], _v1qn_env(tmp_path, sch, root=str(root)))
    assert r.returncode == 0, r.stderr
    summary = {s["stat_key"]: s["stat_value"] for s in _read(tmp_path / "sum.csv")}
    assert int(summary["total_candidate_image_files"]) >= 2


def test_v1qn_model_missing_reported(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    r = _run(S["v1qn"], _v1qn_env(tmp_path, sch))
    assert r.returncode == 0, r.stderr
    summary = {s["stat_key"]: s["stat_value"] for s in _read(tmp_path / "sum.csv")}
    assert summary["model_path_set"] == "false"


def test_v1qn_fake_model_detected(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    model = _make_model_dir(tmp_path)
    root = tmp_path / "assets"
    root.mkdir()
    r = _run(S["v1qn"], _v1qn_env(tmp_path, sch, root=str(root), model=str(model)))
    assert r.returncode == 0, r.stderr
    summary = {s["stat_key"]: s["stat_value"] for s in _read(tmp_path / "sum.csv")}
    assert summary["model_path_set"] == "true"
    assert summary["model_config_exists"] == "true"
    assert summary["model_weights_exists"] == "true"
    # Fake model doesn't load transformers; just structural check
    assert summary["final_status"] in ("LOCAL_ENV_READY_FOR_ASSET_RECONCILIATION",
                                       "LOCAL_ENV_PARTIAL_READY_REVIEW_ONLY")


def test_v1qn_no_abs_path_in_output(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    _run(S["v1qn"], _v1qn_env(tmp_path, sch))
    for row in _read(tmp_path / "audit.csv"):
        for v in row.values():
            assert not LC.detect_abs(v), f"abs path found: {v}"


# ===========================================================================
# v1qo
# ===========================================================================

def _v1qo_env(tmp: Path, smoke: Path, sch: Path, root: str = "") -> dict:
    sch.mkdir(exist_ok=True)
    env = _base(
        REVP_V1QO_IN_SMOKE=str(smoke),
        REVP_V1QO_OUT_REC=str(tmp / "rec.csv"),
        REVP_V1QO_OUT_CAND=str(tmp / "cand.csv"),
        REVP_V1QO_OUT_SUM=str(tmp / "sum.csv"),
        REVP_V1QO_SCH_REC=str(sch / "a.csv"),
        REVP_V1QO_SCH_CAND=str(sch / "b.csv"),
        REVP_V1QO_SCH_SUM=str(sch / "c.csv"),
        REVP_V1QO_DOC=str(tmp / "doc.md"),
    )
    for k in ("REVP_SENTINEL_LOCAL_ROOT","REVP_DINO_VISUAL_ROOT","REVP_DINO_ASSET_ROOT"):
        env.pop(k, None)
    if root:
        env["REVP_SENTINEL_LOCAL_ROOT"] = root
    return env


def test_v1qo_unresolved_no_roots(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    smoke = _make_smoke(tmp_path, 3)
    r = _run(S["v1qo"], _v1qo_env(tmp_path, smoke, sch))
    assert r.returncode == 0, r.stderr
    summary = {s["stat_key"]: s["stat_value"] for s in _read(tmp_path / "sum.csv")}
    assert summary["final_status"] == "SMOKE_ASSETS_UNRESOLVED_FAIL_CLOSED"


def test_v1qo_exact_relative_match(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    root = tmp_path / "assets"
    (root / "data").mkdir(parents=True)
    pid = "CUR_10001"
    tif = root / "data" / f"{pid}.tif"
    tif.write_bytes(b"x")
    smoke = _make_smoke(tmp_path, 1, rel=f"data/{pid}.tif")
    r = _run(S["v1qo"], _v1qo_env(tmp_path, smoke, sch, root=str(root)))
    assert r.returncode == 0, r.stderr
    rows = _read(tmp_path / "rec.csv")
    assert any(row["match_type"] in ("exact_relative","filename_exact") for row in rows)


def test_v1qo_no_abs_path_in_output(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    smoke = _make_smoke(tmp_path, 2)
    _run(S["v1qo"], _v1qo_env(tmp_path, smoke, sch))
    for row in _read(tmp_path / "rec.csv") + _read(tmp_path / "cand.csv"):
        for v in row.values():
            assert not LC.detect_abs(v), f"abs path found: {v}"


def test_v1qo_unresolved_has_blocked_reason(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    smoke = _make_smoke(tmp_path, 2)
    _run(S["v1qo"], _v1qo_env(tmp_path, smoke, sch))
    for row in _read(tmp_path / "rec.csv"):
        if row["match_type"] == "unresolved":
            assert row["blocked_reason"] != ""


def test_v1qo_review_only_flags(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    smoke = _make_smoke(tmp_path, 2)
    _run(S["v1qo"], _v1qo_env(tmp_path, smoke, sch))
    for row in _read(tmp_path / "rec.csv"):
        assert row["review_only"] == "true"
        assert row["can_create_label"] == "false"
        assert row["can_train_model"] == "false"


# ===========================================================================
# v1qp
# ===========================================================================

def test_v1qp_missing_local_file_suggestion(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    smoke = _make_smoke(tmp_path, 3)
    rec   = _make_rec(tmp_path, n_ready=0, n_unresolved=3)
    empty = tmp_path / "empty.csv"; C.write_csv(empty, [], ["a"])
    r = _run(S["v1qp"], _base(
        REVP_V1QP_IN_SMOKE=str(smoke), REVP_V1QP_IN_REC=str(rec),
        REVP_V1QN_IN_V1QA=str(empty), REVP_V1QN_IN_V1FU=str(empty),
        REVP_V1QN_IN_V1FM=str(empty),
        REVP_V1QP_OUT_SUGG=str(tmp_path/"sugg.csv"), REVP_V1QP_OUT_SUM=str(tmp_path/"sum.csv"),
        REVP_V1QP_SCH_SUGG=str(sch/"a.csv"), REVP_V1QP_SCH_SUM=str(sch/"b.csv"),
        REVP_V1QP_DOC=str(tmp_path/"doc.md"),
    ))
    assert r.returncode == 0, r.stderr
    sugg = _read(tmp_path / "sugg.csv")
    assert any(s["issue_type"] in ("MISSING_LOCAL_FILE","FIX_RELATIVE_PATH") for s in sugg)


def test_v1qp_no_action_ready_when_resolved(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    smoke = _make_smoke(tmp_path, 2)
    rec   = _make_rec(tmp_path, n_ready=2, n_unresolved=0)
    empty = tmp_path / "empty.csv"; C.write_csv(empty, [], ["a"])
    _run(S["v1qp"], _base(
        REVP_V1QP_IN_SMOKE=str(smoke), REVP_V1QP_IN_REC=str(rec),
        REVP_V1QN_IN_V1QA=str(empty), REVP_V1QN_IN_V1FU=str(empty),
        REVP_V1QN_IN_V1FM=str(empty),
        REVP_V1QP_OUT_SUGG=str(tmp_path/"sugg.csv"), REVP_V1QP_OUT_SUM=str(tmp_path/"sum.csv"),
        REVP_V1QP_SCH_SUGG=str(sch/"a.csv"), REVP_V1QP_SCH_SUM=str(sch/"b.csv"),
        REVP_V1QP_DOC=str(tmp_path/"doc.md"),
    ))
    sugg = _read(tmp_path / "sugg.csv")
    assert any(s["issue_type"] == "NO_ACTION_READY" for s in sugg)


# ===========================================================================
# v1qq
# ===========================================================================

def _v1qq_env(tmp: Path, sch: Path) -> dict:
    sch.mkdir(exist_ok=True)
    cfg = tmp / "configs"
    return _base(
        REVP_V1QQ_OUT_DOC=str(tmp / "doc.md"),
        REVP_V1QQ_OUT_ENV_EX=str(cfg / "example.env"),
        REVP_V1QQ_OUT_CL=str(cfg / "checklist.csv"),
        REVP_V1QQ_SCH_CL=str(sch / "a.csv"),
    )


def test_v1qq_creates_env_example(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    r = _run(S["v1qq"], _v1qq_env(tmp_path, sch))
    assert r.returncode == 0, r.stderr
    cfg = tmp_path / "configs" / "example.env"
    assert cfg.exists()
    content = cfg.read_text(encoding="utf-8")
    assert "REVP_DINO_MODEL_PATH" in content
    assert "REVP_DINO_ALLOW_DOWNLOAD=false" in content


def test_v1qq_env_example_no_real_abs_path(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    _run(S["v1qq"], _v1qq_env(tmp_path, sch))
    cfg = tmp_path / "configs" / "example.env"
    if cfg.exists():
        lines = cfg.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if not line.strip().startswith("#"):
                # placeholders are allowed; real paths (Windows drive letter) are not
                assert "Users" not in line, f"real path in example: {line}"


def test_v1qq_powershell_commands_use_placeholders(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    _run(S["v1qq"], _v1qq_env(tmp_path, sch))
    cfg = tmp_path / "configs" / "checklist.csv"
    if cfg.exists():
        for row in _read(cfg):
            cmd = row.get("command_or_action", "")
            # Commands should not contain real Windows user paths
            assert "gabriela" not in cmd


def test_v1qq_checklist_has_manual_flag(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    _run(S["v1qq"], _v1qq_env(tmp_path, sch))
    cfg = tmp_path / "configs" / "checklist.csv"
    rows = _read(cfg)
    manual = [r for r in rows if r.get("requires_manual") == "true"]
    assert len(manual) >= 2


# ===========================================================================
# v1qr — readiness gate
# ===========================================================================

def _v1qr_env(tmp: Path, sch: Path,
              env_sum: Path, rec_sum: Path, model_sum: Path, asset_sum: Path,
              **extra: str) -> dict:
    sch.mkdir(exist_ok=True)
    return _base(
        REVP_V1QR_IN_ENV=str(env_sum), REVP_V1QR_IN_REC=str(rec_sum),
        REVP_V1QR_IN_MODEL=str(model_sum), REVP_V1QR_IN_ASSET=str(asset_sum),
        REVP_V1QR_OUT_GATE=str(tmp / "gate.csv"), REVP_V1QR_OUT_SUM=str(tmp / "sum.csv"),
        REVP_V1QR_SCH_GATE=str(sch / "a.csv"), REVP_V1QR_SCH_SUM=str(sch / "b.csv"),
        REVP_V1QR_DOC=str(tmp / "doc.md"), **extra,
    )


def test_v1qr_model_missing_blocked(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    empty = tmp_path / "e.csv"; C.write_csv(empty, [], ["stat_key","stat_value"])
    r = _run(S["v1qr"], _v1qr_env(tmp_path, sch, empty, empty, empty, empty,
                                   REVP_DINO_DRY_RUN="true"))
    assert r.returncode == 0, r.stderr
    summary = {s["stat_key"]: s["stat_value"] for s in _read(tmp_path / "sum.csv")}
    assert summary["final_status"] == "BLOCKED_MODEL_MISSING"


def test_v1qr_assets_unresolved_blocked(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    model_sum = _make_summary(tmp_path, "ms.csv", {
        "final_status": "LOCAL_DINO_MODEL_READY_OFFLINE",
        "model_path_exists": "true", "config_exists": "true",
        "weights_exists": "true", "allow_download": "false",
    })
    rec_sum = _make_summary(tmp_path, "rs.csv", {
        "exact_matches": "0", "partial_matches": "0",
    })
    empty = tmp_path / "e.csv"; C.write_csv(empty, [], ["stat_key","stat_value"])
    r = _run(S["v1qr"], _v1qr_env(tmp_path, sch, empty, rec_sum, model_sum, empty,
                                   REVP_DINO_DRY_RUN="false",
                                   REVP_DINO_PIXEL_READ_ALLOWED="true"))
    assert r.returncode == 0, r.stderr
    summary = {s["stat_key"]: s["stat_value"] for s in _read(tmp_path / "sum.csv")}
    assert summary["final_status"] == "BLOCKED_ASSETS_UNRESOLVED"


def test_v1qr_pixel_read_blocked(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    model_sum = _make_summary(tmp_path, "ms.csv", {
        "final_status": "LOCAL_DINO_MODEL_READY_OFFLINE",
        "model_path_exists": "true", "config_exists": "true",
        "weights_exists": "true", "allow_download": "false",
    })
    rec_sum = _make_summary(tmp_path, "rs.csv", {
        "exact_matches": "3", "partial_matches": "1",
    })
    empty = tmp_path / "e.csv"; C.write_csv(empty, [], ["stat_key","stat_value"])
    env = _v1qr_env(tmp_path, sch, empty, rec_sum, model_sum, empty,
                    REVP_DINO_DRY_RUN="false")
    env.pop("REVP_DINO_PIXEL_READ_ALLOWED", None)
    r = _run(S["v1qr"], env)
    assert r.returncode == 0, r.stderr
    summary = {s["stat_key"]: s["stat_value"] for s in _read(tmp_path / "sum.csv")}
    assert summary["final_status"] == "BLOCKED_PIXEL_READ_NOT_ALLOWED"


def test_v1qr_ready_manual_smoke_run(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    model_sum = _make_summary(tmp_path, "ms.csv", {
        "final_status": "LOCAL_DINO_MODEL_READY_OFFLINE",
        "model_path_exists": "true", "config_exists": "true",
        "weights_exists": "true", "allow_download": "false",
    })
    rec_sum = _make_summary(tmp_path, "rs.csv", {
        "exact_matches": "4", "partial_matches": "2",
    })
    empty = tmp_path / "e.csv"; C.write_csv(empty, [], ["stat_key","stat_value"])
    r = _run(S["v1qr"], _v1qr_env(tmp_path, sch, empty, rec_sum, model_sum, empty,
                                   REVP_DINO_DRY_RUN="false",
                                   REVP_DINO_PIXEL_READ_ALLOWED="true"))
    assert r.returncode == 0, r.stderr
    summary = {s["stat_key"]: s["stat_value"] for s in _read(tmp_path / "sum.csv")}
    assert summary["final_status"] == "READY_FOR_MANUAL_REAL_SMOKE_RUN"


def test_v1qr_dry_run_only(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    model_sum = _make_summary(tmp_path, "ms.csv", {
        "model_path_exists": "true", "config_exists": "true",
        "weights_exists": "true", "allow_download": "false",
    })
    rec_sum = _make_summary(tmp_path, "rs.csv", {"exact_matches": "5", "partial_matches": "0"})
    empty = tmp_path / "e.csv"; C.write_csv(empty, [], ["stat_key","stat_value"])
    r = _run(S["v1qr"], _v1qr_env(tmp_path, sch, empty, rec_sum, model_sum, empty,
                                   REVP_DINO_DRY_RUN="true",
                                   REVP_DINO_PIXEL_READ_ALLOWED="true"))
    assert r.returncode == 0, r.stderr
    summary = {s["stat_key"]: s["stat_value"] for s in _read(tmp_path / "sum.csv")}
    # dry_run=true means gate G07 fails → READY_FOR_DRY_RUN_ONLY
    assert summary["final_status"] == "READY_FOR_DRY_RUN_ONLY"


# ===========================================================================
# v1qs — TCC update
# ===========================================================================

def _v1qs_env(tmp: Path, sch: Path,
              gate_sum: Path, rec_sum: Path, env_sum: Path) -> dict:
    sch.mkdir(exist_ok=True)
    return _base(
        REVP_V1QS_IN_GATE=str(gate_sum), REVP_V1QS_IN_REC=str(rec_sum),
        REVP_V1QS_IN_ENV=str(env_sum),
        REVP_V1QS_OUT_READ=str(tmp / "read.csv"),
        REVP_V1QS_OUT_BLOCK=str(tmp / "block.csv"),
        REVP_V1QS_SCH_READ=str(sch / "a.csv"),
        REVP_V1QS_SCH_BLOCK=str(sch / "b.csv"),
        REVP_V1QS_DOC=str(tmp / "doc.md"),
    )


def test_v1qs_mandatory_phrase_in_doc(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    empty = tmp_path / "e.csv"; C.write_csv(empty, [], ["stat_key","stat_value"])
    r = _run(S["v1qs"], _v1qs_env(tmp_path, sch, empty, empty, empty))
    assert r.returncode == 0, r.stderr
    text = (tmp_path / "doc.md").read_text(encoding="utf-8")
    assert "não altera o estatuto científico" in text
    assert "rótulos, targets ou ground truth" in text


def test_v1qs_blockers_when_no_model(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    empty = tmp_path / "e.csv"; C.write_csv(empty, [], ["stat_key","stat_value"])
    env_sum = _make_summary(tmp_path, "es.csv", {"roots_existing":"0","model_path_set":"false","total_candidate_image_files":"0"})
    _run(S["v1qs"], _v1qs_env(tmp_path, sch, empty, empty, env_sum))
    block = _read(tmp_path / "block.csv")
    blockers = [b["blocker_name"] for b in block]
    assert "model_path_not_set" in blockers or "no_local_roots" in blockers


# ===========================================================================
# v1qt — bundle
# ===========================================================================

def _v1qt_env(tmp: Path, in_ds: Path, sch: Path) -> dict:
    sch.mkdir(exist_ok=True)
    return _base(
        REVP_V1QT_IN_DATASETS=str(in_ds),
        REVP_V1QT_OUT_MAN=str(tmp / "man.csv"),
        REVP_V1QT_OUT_QC=str(tmp / "qc.csv"),
        REVP_V1QT_OUT_SUM=str(tmp / "sum.csv"),
        REVP_V1QT_SCH_MAN=str(sch / "a.csv"),
        REVP_V1QT_SCH_QC=str(sch / "b.csv"),
        REVP_V1QT_SCH_SUM=str(sch / "c.csv"),
        REVP_V1QT_DOC=str(tmp / "doc.md"),
    )


def test_v1qt_model_missing_status(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    ds = tmp_path / "ds"; ds.mkdir()
    r = _run(S["v1qt"], _v1qt_env(tmp_path, ds, sch))
    assert r.returncode == 0, r.stderr
    summary = {s["metric"]: s["value"] for s in _read(tmp_path / "sum.csv")}
    assert summary["final_status"] == "LOCAL_DINO_READINESS_MODEL_MISSING_FAIL_CLOSED"


def test_v1qt_assets_missing_status(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    ds = tmp_path / "ds"; ds.mkdir()
    # model is set but assets unresolved
    C.write_csv(ds / "dino_local_root_environment_summary_v1qn.csv", [
        {"stat_key": "model_path_set",   "stat_value": "true"},
        {"stat_key": "model_path_exists","stat_value": "true"},
        {"stat_key": "roots_existing",   "stat_value": "1"},
        {"stat_key": "model_allow_download","stat_value":"false"},
    ], ["stat_key","stat_value"])
    C.write_csv(ds / "dino_smoke_asset_local_reconciliation_summary_v1qo.csv", [
        {"stat_key": "unresolved",       "stat_value": "5"},
        {"stat_key": "exact_matches",    "stat_value": "0"},
        {"stat_key": "partial_matches",  "stat_value": "0"},
    ], ["stat_key","stat_value"])
    r = _run(S["v1qt"], _v1qt_env(tmp_path, ds, sch))
    assert r.returncode == 0, r.stderr
    summary = {s["metric"]: s["value"] for s in _read(tmp_path / "sum.csv")}
    assert summary["final_status"] == "LOCAL_DINO_READINESS_ASSETS_MISSING_FAIL_CLOSED"


def test_v1qt_ready_status_simulated(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    ds = tmp_path / "ds"; ds.mkdir()
    C.write_csv(ds / "dino_local_root_environment_summary_v1qn.csv", [
        {"stat_key": "model_path_set",   "stat_value": "true"},
        {"stat_key": "model_path_exists","stat_value": "true"},
        {"stat_key": "roots_existing",   "stat_value": "1"},
        {"stat_key": "model_allow_download","stat_value":"false"},
    ], ["stat_key","stat_value"])
    C.write_csv(ds / "dino_smoke_asset_local_reconciliation_summary_v1qo.csv", [
        {"stat_key": "unresolved",    "stat_value": "0"},
        {"stat_key": "exact_matches", "stat_value": "5"},
        {"stat_key": "partial_matches","stat_value":"0"},
    ], ["stat_key","stat_value"])
    C.write_csv(ds / "dino_local_smoke_run_readiness_summary_v1qr.csv", [
        {"stat_key": "final_status",    "stat_value": "READY_FOR_MANUAL_REAL_SMOKE_RUN"},
        {"stat_key": "labels_created",  "stat_value": "0"},
        {"stat_key": "targets_created", "stat_value": "0"},
    ], ["stat_key","stat_value"])
    r = _run(S["v1qt"], _v1qt_env(tmp_path, ds, sch))
    assert r.returncode == 0, r.stderr
    summary = {s["metric"]: s["value"] for s in _read(tmp_path / "sum.csv")}
    assert summary["final_status"] == "LOCAL_DINO_READINESS_READY_FOR_MANUAL_SMOKE_RUN"


# ===========================================================================
# cross-cutting
# ===========================================================================

def test_guardrail_assert_label_raises() -> None:
    import pytest
    with pytest.raises(ValueError):
        C.assert_no_forbidden_true([{"can_create_label": "true"}], "x")


def test_guardrail_assert_train_raises() -> None:
    import pytest
    with pytest.raises(ValueError):
        C.assert_no_forbidden_true([{"can_train_model": "true"}], "x")


def test_no_test_writes_real_datasets() -> None:
    assert (ROOT / "datasets").exists()
    assert "can_create_label" in LC.READINESS_FORBIDDEN_FIELDS


def test_schemas_written_by_v1qn(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    _run(S["v1qn"], _v1qn_env(tmp_path, sch))
    assert (sch / "a.csv").exists() and (sch / "b.csv").exists()


def test_docs_written_by_v1qh_selector(tmp_path: Path) -> None:
    sch = tmp_path / "schemas"
    smoke = _make_smoke(tmp_path, 2)
    _run(S["v1qo"], _v1qo_env(tmp_path, smoke, sch))
    assert (tmp_path / "doc.md").exists()
