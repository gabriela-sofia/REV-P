"""REV-P v1qn — Local root environment audit.

Audits configured env roots and the local DINOv2 model directory without
reading pixels or downloading anything. Reports which roots exist, how many
candidate image files each contains, and whether the model directory looks valid.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from revp_v1qn_v1qt_local_readiness_common import (
    DATASETS, DOCS, SCHEMAS, IMAGE_EXTS,
    _p, assert_no_forbidden_true, env_str,
    path_hash, read_json_safe, require_no_abs_paths,
    write_csv, write_doc, write_schema,
)

OUT_AUDIT = _p("REVP_V1QN_OUT_AUDIT", DATASETS / "dino_local_root_environment_audit_v1qn.csv")
OUT_SUM   = _p("REVP_V1QN_OUT_SUM",   DATASETS / "dino_local_root_environment_summary_v1qn.csv")
SCH_AUDIT = _p("REVP_V1QN_SCH_AUDIT", SCHEMAS / "dino_local_root_environment_audit_v1qn_schema.csv")
SCH_SUM   = _p("REVP_V1QN_SCH_SUM",   SCHEMAS / "dino_local_root_environment_summary_v1qn_schema.csv")
DOC       = _p("REVP_V1QN_DOC",        DOCS / "revp_v1qn_local_root_environment_audit.md")

AUDIT_FIELDS = [
    "audit_id", "env_var", "configured", "path_exists",
    "candidate_files_tif", "candidate_files_png", "candidate_files_all_image",
    "path_hash", "role", "notes",
]
SUM_FIELDS = ["stat_key", "stat_value"]

ROOT_ROLES = {
    "REVP_SENTINEL_LOCAL_ROOT": "sentinel_tif_root",
    "REVP_DINO_VISUAL_ROOT":    "visual_asset_root",
    "REVP_DINO_ASSET_ROOT":     "generic_asset_root",
    "REVP_DINO_SOURCE_ROOT":    "embedding_source_root",
}


def _count_ext(root: Path, ext: str) -> int:
    try:
        return sum(1 for _ in root.rglob(f"*{ext}"))
    except Exception:
        return 0


def audit_roots() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, (env, role) in enumerate(ROOT_ROLES.items(), 1):
        val = env_str(env)
        configured = bool(val)
        p = Path(val) if val else None
        exists = p.exists() if p else False
        tif = (_count_ext(p, ".tif") + _count_ext(p, ".tiff")) if (exists and p) else 0
        png = _count_ext(p, ".png") if (exists and p) else 0
        all_img = sum(_count_ext(p, e) for e in IMAGE_EXTS) if (exists and p) else 0
        rows.append({
            "audit_id": f"V1QN_ROOT_{i:02d}",
            "env_var": env, "configured": str(configured).lower(),
            "path_exists": str(exists).lower(),
            "candidate_files_tif": str(tif),
            "candidate_files_png": str(png),
            "candidate_files_all_image": str(all_img),
            "path_hash": path_hash(val) if val else "",
            "role": role, "notes": "",
        })
    return rows


def audit_model() -> dict[str, Any]:
    mp = env_str("REVP_DINO_MODEL_PATH")
    allow_dl = env_str("REVP_DINO_ALLOW_DOWNLOAD", "false")
    hf_offline = env_str("HF_HUB_OFFLINE", "")
    if not mp:
        return {"model_path_set": False, "model_path_exists": False,
                "config_exists": False, "weights_exists": False,
                "processor_exists": False, "allow_download": allow_dl,
                "hf_hub_offline": hf_offline, "model_path_hash": "",
                "detected_hidden_size": "", "model_type": ""}
    p = Path(mp)
    exists = p.exists() and p.is_dir()
    cfg_path = p / "config.json"
    cfg_exists = cfg_path.exists()
    weights = exists and any(
        any(p.glob(g)) for g in ("*.safetensors", "*.bin", "*.pt", "*.pth")
    )
    proc_names = ("preprocessor_config.json", "processor_config.json",
                  "image_processor_config.json")
    proc = exists and any((p / n).exists() for n in proc_names)
    hidden = ""
    mtype = ""
    if cfg_exists:
        cfg = read_json_safe(cfg_path)
        if isinstance(cfg, dict):
            hidden = str(cfg.get("hidden_size", ""))
            mtype = str(cfg.get("model_type", ""))
    return {"model_path_set": True, "model_path_exists": exists,
            "config_exists": cfg_exists, "weights_exists": weights,
            "processor_exists": proc, "allow_download": allow_dl,
            "hf_hub_offline": hf_offline, "model_path_hash": path_hash(mp),
            "detected_hidden_size": hidden, "model_type": mtype}


def run() -> None:
    root_rows = audit_roots()
    model = audit_model()

    roots_exist = sum(1 for r in root_rows if r["path_exists"] == "true")
    roots_configured = sum(1 for r in root_rows if r["configured"] == "true")
    total_imgs = sum(int(r["candidate_files_all_image"]) for r in root_rows)
    model_ready = (model["model_path_exists"] and model["config_exists"]
                   and model["weights_exists"] and model["allow_download"] == "false")

    if not roots_configured:
        final = "LOCAL_ENV_ROOTS_MISSING_FAIL_CLOSED"
    elif not model["model_path_set"]:
        final = "LOCAL_ENV_MODEL_MISSING_FAIL_CLOSED"
    elif roots_exist > 0 and model_ready:
        final = "LOCAL_ENV_READY_FOR_ASSET_RECONCILIATION"
    else:
        final = "LOCAL_ENV_PARTIAL_READY_REVIEW_ONLY"

    summary = [
        {"stat_key": "roots_configured",             "stat_value": str(roots_configured)},
        {"stat_key": "roots_existing",               "stat_value": str(roots_exist)},
        {"stat_key": "total_candidate_image_files",  "stat_value": str(total_imgs)},
        {"stat_key": "model_path_set",               "stat_value": str(model["model_path_set"]).lower()},
        {"stat_key": "model_path_exists",            "stat_value": str(model["model_path_exists"]).lower()},
        {"stat_key": "model_config_exists",          "stat_value": str(model["config_exists"]).lower()},
        {"stat_key": "model_weights_exists",         "stat_value": str(model["weights_exists"]).lower()},
        {"stat_key": "model_allow_download",         "stat_value": model["allow_download"]},
        {"stat_key": "hf_hub_offline",               "stat_value": model["hf_hub_offline"]},
        {"stat_key": "detected_hidden_size",         "stat_value": model["detected_hidden_size"]},
        {"stat_key": "model_type",                   "stat_value": model["model_type"]},
        {"stat_key": "labels_created",               "stat_value": "0"},
        {"stat_key": "targets_created",              "stat_value": "0"},
        {"stat_key": "final_status",                 "stat_value": final},
    ]
    for rows, label in ((root_rows, "v1qn_audit"), (summary, "v1qn_summary")):
        require_no_abs_paths(rows, label)
        assert_no_forbidden_true(rows, label)
    write_csv(OUT_AUDIT, root_rows, AUDIT_FIELDS)
    write_csv(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCH_AUDIT, AUDIT_FIELDS, "v1qn_local_root_environment_audit")
    write_schema(SCH_SUM, SUM_FIELDS, "v1qn_local_root_environment_summary")
    write_doc(DOC, "v1qn — Local Root Environment Audit", [
        "## Objetivo",
        "Auditar env roots e modelo DINOv2 local sem ler pixels nem baixar nada. "
        "Verifica existência de roots, contagem de candidatos por extensão, e "
        "presença de config/pesos/processor no diretório de modelo.",
        "## Status",
        f"**{final}**. Roots existentes: {roots_exist}/{roots_configured}. "
        f"Imagens candidatas: {total_imgs}. Modelo configurado: {model['model_path_set']}.",
    ])
    print(f"[v1qn] {final} roots_exist={roots_exist} imgs={total_imgs} "
          f"model_set={model['model_path_set']}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1qn local root environment audit").parse_args()
    run()
