"""REV-P v1pp — DINO backend/model availability probe.

Detects local Python environment without downloading anything by default.
REVP_DINO_ALLOW_DOWNLOAD=false (default). Fail-closed if no local model.
"""
from __future__ import annotations

import argparse
from typing import Any

from revp_v1pn_v1pt_dino_execution_common import (
    DATASETS, DOCS, SCHEMAS,
    _p, assert_no_forbidden_true, probe_backend, require_no_abs_paths,
    write_csv, write_doc, write_schema,
)

OUT_PROBE = _p("REVP_V1PP_OUT_PROBE", DATASETS / "dino_backend_model_probe_v1pp.csv")
OUT_SUM = _p("REVP_V1PP_OUT_SUM", DATASETS / "dino_backend_model_probe_summary_v1pp.csv")
SCH_PROBE = _p("REVP_V1PP_SCH_PROBE", SCHEMAS / "dino_backend_model_probe_v1pp_schema.csv")
SCH_SUM = _p("REVP_V1PP_SCH_SUM", SCHEMAS / "dino_backend_model_probe_summary_v1pp_schema.csv")
DOC = _p("REVP_V1PP_DOC", DOCS / "revp_v1pp_dino_backend_model_probe.md")

PROBE_FIELDS = [
    "probe_id", "component", "available", "version_or_value",
    "required_for_execution", "status", "blocked_reason", "notes",
]
SUM_FIELDS = ["stat_key", "stat_value"]


def build_probe(info: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def _row(i: int, comp: str, ok: bool, ver: str, req: bool, blocked: str = "") -> dict[str, Any]:
        return {
            "probe_id": f"V1PP_P{i:03d}",
            "component": comp,
            "available": str(ok).lower(),
            "version_or_value": ver,
            "required_for_execution": str(req).lower(),
            "status": "AVAILABLE" if ok else ("REQUIRED_MISSING" if req else "OPTIONAL_MISSING"),
            "blocked_reason": blocked if not ok and req else "",
            "notes": "",
        }

    nok, nver = info["numpy"]
    rows.append(_row(1, "numpy", nok, nver, True))
    pok, pver = info["pil"]
    rows.append(_row(2, "PIL", pok, pver, True))
    tok, tver = info["torch"]
    rows.append(_row(3, "torch", tok, tver, False))
    trk, trver = info["transformers"]
    rows.append(_row(4, "transformers", trk, trver, False))
    tik, tiver = info["timm"]
    rows.append(_row(5, "timm", tik, tiver, False))
    mp = info["model_path"]
    rows.append(_row(6, "model_path_env", bool(mp), mp or "not_set", False,
                     "" if mp else "REVP_DINO_MODEL_PATH not set"))
    rows.append(_row(7, "model_path_exists", info["model_path_exists"],
                     mp if info["model_path_exists"] else "path_does_not_exist", False))
    mn = info["model_name"]
    rows.append(_row(8, "model_name_env", bool(mn), mn or "not_set", False))
    rows.append(_row(9, "allow_download", info["allow_download"],
                     str(info["allow_download"]).lower(), False))
    return rows


def build_summary(info: dict[str, Any]) -> list[dict[str, str]]:
    model_src = "local_path" if info["model_path_exists"] else ("download_if_allowed" if info["model_name"] else "none")
    return [
        {"stat_key": "can_execute_embeddings", "stat_value": str(info["can_execute"]).lower()},
        {"stat_key": "model_source", "stat_value": model_src},
        {"stat_key": "download_allowed", "stat_value": str(info["allow_download"]).lower()},
        {"stat_key": "numpy_available", "stat_value": str(info["numpy"][0]).lower()},
        {"stat_key": "pil_available", "stat_value": str(info["pil"][0]).lower()},
        {"stat_key": "torch_available", "stat_value": str(info["torch"][0]).lower()},
        {"stat_key": "transformers_available", "stat_value": str(info["transformers"][0]).lower()},
        {"stat_key": "final_status", "stat_value": info["final_status"]},
    ]


def run() -> None:
    info = probe_backend()
    probe_rows = build_probe(info)
    summary = build_summary(info)
    require_no_abs_paths(probe_rows, "v1pp_probe")
    assert_no_forbidden_true(probe_rows, "v1pp_probe")
    write_csv(OUT_PROBE, probe_rows, PROBE_FIELDS)
    write_csv(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCH_PROBE, PROBE_FIELDS, "v1pp_dino_backend_model_probe")
    write_schema(SCH_SUM, SUM_FIELDS, "v1pp_dino_backend_model_probe_summary")
    write_doc(DOC, "v1pp — DINO Backend/Model Probe", [
        "## Objetivo",
        "Detectar ambiente local sem baixar nada por default. "
        "REVP_DINO_ALLOW_DOWNLOAD=false (padrão).",
        "## Fail-closed",
        "Se modelo não existir localmente e download não autorizado, "
        "status = DINO_BACKEND_MODEL_UNAVAILABLE_FAIL_CLOSED.",
        f"## Resultado",
        f"Final status: {info['final_status']}. "
        f"can_execute={info['can_execute']}.",
    ])
    print(f"[v1pp] status={info['final_status']} can_execute={info['can_execute']}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1pp dino backend model probe").parse_args()
    run()
