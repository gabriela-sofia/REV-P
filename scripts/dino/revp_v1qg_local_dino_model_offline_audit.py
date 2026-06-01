"""REV-P v1qg — Local DINOv2 model offline audit.

Audits a locally available DINOv2 (with-registers) model directory WITHOUT
downloading anything and WITHOUT running inference. Reads config only.

Gates honoured:
  * REVP_DINO_MODEL_PATH must exist;
  * REVP_DINO_ALLOW_DOWNLOAD must be false;
  * HF_HUB_OFFLINE should be 1.
"""
from __future__ import annotations

import argparse
from typing import Any

from revp_v1qg_v1qm_smoke_embedding_common import (
    DATASETS, DOCS, EXPECTED_DINO_DIM, SCHEMAS,
    _p, assert_no_forbidden_true, env_str, path_hash,
    require_no_abs_paths, safe_model_probe, write_csv, write_doc, write_schema,
)

OUT_AUDIT = _p("REVP_V1QG_OUT_AUDIT", DATASETS / "dino_local_model_offline_audit_v1qg.csv")
OUT_SUM = _p("REVP_V1QG_OUT_SUM", DATASETS / "dino_local_model_offline_summary_v1qg.csv")
SCH_AUDIT = _p("REVP_V1QG_SCH_AUDIT", SCHEMAS / "dino_local_model_offline_audit_v1qg_schema.csv")
SCH_SUM = _p("REVP_V1QG_SCH_SUM", SCHEMAS / "dino_local_model_offline_summary_v1qg_schema.csv")
DOC = _p("REVP_V1QG_DOC", DOCS / "revp_v1qg_local_dino_model_offline_audit.md")

AUDIT_FIELDS = [
    "check_id", "model_path_hash", "model_source", "model_path_exists",
    "config_exists", "weights_exists", "processor_exists",
    "transformers_available", "torch_available", "offline_mode",
    "allow_download", "expected_dim", "detected_dim", "model_ready",
    "status", "blocked_reason", "notes",
]
SUM_FIELDS = ["stat_key", "stat_value"]


def audit() -> tuple[dict[str, Any], str]:
    model_path = env_str("REVP_DINO_MODEL_PATH", "")
    info = safe_model_probe(model_path or None)

    expected = EXPECTED_DINO_DIM
    detected = int(info.get("detected_dim", 0) or 0)

    # Determine status, fail-closed by default.
    blocked = ""
    if not model_path or not info["model_path_exists"]:
        status = "LOCAL_DINO_MODEL_MISSING_FAIL_CLOSED"
        blocked = "model_path_missing_or_unset"
        ready = False
    elif info["allow_download"]:
        status = "LOCAL_DINO_MODEL_INVALID_FAIL_CLOSED"
        blocked = "allow_download_true_not_offline_safe"
        ready = False
    elif not info["config_exists"] or not info["weights_exists"]:
        status = "LOCAL_DINO_MODEL_INVALID_FAIL_CLOSED"
        blocked = "missing_config_or_weights"
        ready = False
    elif detected and detected != expected:
        status = "LOCAL_DINO_MODEL_DIMENSION_MISMATCH_FAIL_CLOSED"
        blocked = f"detected_dim={detected}_expected={expected}"
        ready = False
    elif not info["transformers_available"]:
        status = "LOCAL_DINO_MODEL_INVALID_FAIL_CLOSED"
        blocked = "transformers_unavailable"
        ready = False
    else:
        status = "LOCAL_DINO_MODEL_READY_OFFLINE"
        ready = True

    arch_note = "dinov2_with_registers" if info["is_dinov2_with_registers"] \
        else ("dinov2" if info["is_dinov2"] else info["model_type"])

    row: dict[str, Any] = {
        "check_id": "V1QG_CHK_00001",
        "model_path_hash": path_hash(model_path) if model_path else "",
        "model_source": "LOCAL_PATH" if model_path else "NONE",
        "model_path_exists": str(info["model_path_exists"]).lower(),
        "config_exists": str(info["config_exists"]).lower(),
        "weights_exists": str(info["weights_exists"]).lower(),
        "processor_exists": str(info["processor_exists"]).lower(),
        "transformers_available": str(info["transformers_available"]).lower(),
        "torch_available": str(info["torch_available"]).lower(),
        "offline_mode": str(info["offline_mode"]).lower(),
        "allow_download": str(info["allow_download"]).lower(),
        "expected_dim": str(expected),
        "detected_dim": str(detected),
        "model_ready": str(ready).lower(),
        "status": status,
        "blocked_reason": blocked,
        "notes": f"arch={arch_note};config_loadable={str(info['config_loadable']).lower()}",
    }
    return row, status


def run() -> None:
    row, status = audit()
    audit_rows = [row]
    summary = [
        {"stat_key": "model_path_set", "stat_value": str(bool(env_str("REVP_DINO_MODEL_PATH", ""))).lower()},
        {"stat_key": "model_path_exists", "stat_value": row["model_path_exists"]},
        {"stat_key": "config_exists", "stat_value": row["config_exists"]},
        {"stat_key": "weights_exists", "stat_value": row["weights_exists"]},
        {"stat_key": "transformers_available", "stat_value": row["transformers_available"]},
        {"stat_key": "torch_available", "stat_value": row["torch_available"]},
        {"stat_key": "offline_mode", "stat_value": row["offline_mode"]},
        {"stat_key": "allow_download", "stat_value": row["allow_download"]},
        {"stat_key": "expected_dim", "stat_value": row["expected_dim"]},
        {"stat_key": "detected_dim", "stat_value": row["detected_dim"]},
        {"stat_key": "model_ready", "stat_value": row["model_ready"]},
        {"stat_key": "final_status", "stat_value": status},
    ]
    for label, rows in (("v1qg_audit", audit_rows), ("v1qg_summary", summary)):
        require_no_abs_paths(rows, label)
        assert_no_forbidden_true(rows, label)
    write_csv(OUT_AUDIT, audit_rows, AUDIT_FIELDS)
    write_csv(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCH_AUDIT, AUDIT_FIELDS, "v1qg_local_model_offline_audit")
    write_schema(SCH_SUM, SUM_FIELDS, "v1qg_local_model_offline_summary")
    write_doc(DOC, "v1qg — Local DINOv2 Model Offline Audit", [
        "## Objetivo",
        "Auditar um modelo DINOv2 (with registers) local de forma offline, sem baixar "
        "nada e sem rodar inferência. Apenas presença de config/pesos/processor e "
        "leitura de hidden_size.",
        "## Gates",
        "Exige REVP_DINO_MODEL_PATH existente, REVP_DINO_ALLOW_DOWNLOAD=false e "
        "HF_HUB_OFFLINE=1. Qualquer falha resulta em status fail-closed.",
        "## Status",
        f"**{status}**. expected_dim=768.",
    ])
    print(f"[v1qg] status={status}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1qg local dino model offline audit").parse_args()
    run()
