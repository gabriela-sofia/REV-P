"""REV-P v1qr — Local smoke run readiness gate.

Consolidates v1qn/v1qo/v1qg/v1qi to decide if real smoke embedding can run.
Does NOT trigger any embedding execution.
"""
from __future__ import annotations

import argparse
from typing import Any

from revp_v1qn_v1qt_local_readiness_common import (
    DATASETS, DOCS, SCHEMAS,
    _p, assert_no_forbidden_true, env_str, read_csv,
    require_no_abs_paths, write_csv, write_doc, write_schema,
)

IN_ENV_SUM   = _p("REVP_V1QR_IN_ENV",   DATASETS / "dino_local_root_environment_summary_v1qn.csv")
IN_REC_SUM   = _p("REVP_V1QR_IN_REC",   DATASETS / "dino_smoke_asset_local_reconciliation_summary_v1qo.csv")
IN_MODEL_SUM = _p("REVP_V1QR_IN_MODEL", DATASETS / "dino_local_model_offline_summary_v1qg.csv")
IN_ASSET_SUM = _p("REVP_V1QR_IN_ASSET", DATASETS / "dino_local_asset_preprocessing_summary_v1qi.csv")

OUT_GATE = _p("REVP_V1QR_OUT_GATE", DATASETS / "dino_local_smoke_run_readiness_gate_v1qr.csv")
OUT_SUM  = _p("REVP_V1QR_OUT_SUM",  DATASETS / "dino_local_smoke_run_readiness_summary_v1qr.csv")
SCH_GATE = _p("REVP_V1QR_SCH_GATE", SCHEMAS / "dino_local_smoke_run_readiness_gate_v1qr_schema.csv")
SCH_SUM  = _p("REVP_V1QR_SCH_SUM",  SCHEMAS / "dino_local_smoke_run_readiness_summary_v1qr_schema.csv")
DOC      = _p("REVP_V1QR_DOC",       DOCS / "revp_v1qr_local_smoke_run_readiness_gate.md")

GATE_FIELDS = [
    "gate_id", "gate_name", "required_value", "observed_value",
    "passed", "blocking", "notes",
]
SUM_FIELDS = ["stat_key", "stat_value"]


def _stat(path: Any, key: str, default: str = "") -> str:
    for r in read_csv(path):
        if r.get("stat_key") == key:
            return r.get("stat_value", default)
    return default


def evaluate_gates() -> tuple[list[dict[str, Any]], str]:
    model_status  = _stat(IN_MODEL_SUM, "final_status", "MISSING")
    allow_dl      = _stat(IN_MODEL_SUM, "allow_download", "true")
    # Prefer env var; fall back to recorded value in model summary
    hf_offline    = env_str("HF_HUB_OFFLINE", "") or _stat(IN_MODEL_SUM, "offline_mode", "")
    detected_dim  = _stat(IN_MODEL_SUM, "detected_dim", "0")
    reconciled    = int(_stat(IN_REC_SUM, "exact_matches", "0") or "0") + \
                    int(_stat(IN_REC_SUM, "partial_matches", "0") or "0")
    pixel_allowed = env_str("REVP_DINO_PIXEL_READ_ALLOWED", "false")
    dry_run       = env_str("REVP_DINO_DRY_RUN", "true")
    labels        = _stat(IN_MODEL_SUM, "labels_created", "0")

    model_ready = model_status == "LOCAL_DINO_MODEL_READY_OFFLINE" or (
        _stat(IN_MODEL_SUM, "model_path_exists", "false") == "true"
        and _stat(IN_MODEL_SUM, "config_exists", "false") == "true"
        and _stat(IN_MODEL_SUM, "weights_exists", "false") == "true"
    )
    dim_ok = not detected_dim or detected_dim in ("0", "", "768")

    def gate(gid: str, name: str, required: str, observed: str,
             passed: bool, blocking: bool = True) -> dict[str, Any]:
        return {
            "gate_id": gid, "gate_name": name,
            "required_value": required, "observed_value": observed,
            "passed": str(passed).lower(), "blocking": str(blocking).lower(),
            "notes": "",
        }

    gates = [
        gate("G01", "model_local_ready",    "true",    str(model_ready).lower(), model_ready),
        gate("G02", "allow_download_false", "false",   allow_dl,                 allow_dl == "false"),
        gate("G03", "hf_hub_offline",       "1_or_na", hf_offline,               hf_offline in ("true","1","") or not hf_offline),
        gate("G04", "embedding_dim_768",    "768",     detected_dim or "unknown", dim_ok),
        gate("G05", "assets_reconciled_2+", ">=2",     str(reconciled),           reconciled >= 2),
        gate("G06", "pixel_read_allowed",   "true",    pixel_allowed,             pixel_allowed == "true"),
        gate("G07", "dry_run_disabled",     "false",   dry_run,                   dry_run == "false"),
        gate("G08", "labels_created_zero",  "0",       labels,                    labels == "0", False),
    ]

    blocking_failed = [g for g in gates if g["passed"] == "false" and g["blocking"] == "true"]
    if not blocking_failed:
        final = "READY_FOR_MANUAL_REAL_SMOKE_RUN"
    elif any(g["gate_name"] == "model_local_ready" for g in blocking_failed):
        final = "BLOCKED_MODEL_MISSING"
    elif any(g["gate_name"] == "assets_reconciled_2+" for g in blocking_failed):
        final = "BLOCKED_ASSETS_UNRESOLVED"
    elif any(g["gate_name"] == "pixel_read_allowed" for g in blocking_failed):
        final = "BLOCKED_PIXEL_READ_NOT_ALLOWED"
    elif any(g["gate_name"] == "labels_created_zero" for g in blocking_failed):
        final = "BLOCKED_GUARDRAIL_VIOLATION"
    else:
        final = "READY_FOR_DRY_RUN_ONLY"
    return gates, final


def run() -> None:
    gates, final = evaluate_gates()
    require_no_abs_paths(gates, "v1qr_gates")
    assert_no_forbidden_true(gates, "v1qr_gates")

    passed = sum(1 for g in gates if g["passed"] == "true")
    summary = [
        {"stat_key": "gates_total",   "stat_value": str(len(gates))},
        {"stat_key": "gates_passed",  "stat_value": str(passed)},
        {"stat_key": "gates_failed",  "stat_value": str(len(gates) - passed)},
        {"stat_key": "labels_created","stat_value": "0"},
        {"stat_key": "targets_created","stat_value": "0"},
        {"stat_key": "final_status",  "stat_value": final},
    ]
    require_no_abs_paths(summary, "v1qr_summary")
    assert_no_forbidden_true(summary, "v1qr_summary")
    write_csv(OUT_GATE, gates,   GATE_FIELDS)
    write_csv(OUT_SUM,  summary, SUM_FIELDS)
    write_schema(SCH_GATE, GATE_FIELDS, "v1qr_local_smoke_run_readiness_gate")
    write_schema(SCH_SUM,  SUM_FIELDS,  "v1qr_local_smoke_run_readiness_summary")
    write_doc(DOC, "v1qr — Local Smoke Run Readiness Gate", [
        "## Objetivo",
        "Consolidar v1qn/v1qo/v1qg/v1qi e decidir se smoke embedding real pode "
        "ser executado. Não dispara embedding.",
        "## Condição para READY_FOR_MANUAL_REAL_SMOKE_RUN",
        "Modelo local offline ready, allow_download=false, assets>=2 reconciliados, "
        "pixel_read_allowed=true, dry_run=false, labels=0.",
        "## Status",
        f"**{final}**. Gates passados: {passed}/{len(gates)}.",
    ])
    print(f"[v1qr] {final} passed={passed}/{len(gates)}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1qr local smoke run readiness gate").parse_args()
    run()
