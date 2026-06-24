"""MV2 pre-unification controlled resume.

This script creates lightweight contracts, manifests, seeds, policies, and
readiness reports for the MV2-16 dry-run decision without opening training,
silver labels, formal negatives, supervised splits, downloads, public crops, or
operational claims.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MV2_WORKTREE = Path(r"C:\Users\gabriela\Documents\REV-P-mv2-01-reconciliado")

EXEC_REPORTS = PROJECT_ROOT / "outputs_public" / "execution_reports"
CONTRACT_DIR = PROJECT_ROOT / "outputs_public" / "mv2_pre_unification_contracts"
SEED_DIR = PROJECT_ROOT / "outputs_public" / "mv2_pre_unification_seed"
METADATA_DIR = PROJECT_ROOT / "outputs_public" / "mv2_pre_unification_metadata"
CROP_DIR = PROJECT_ROOT / "outputs_public" / "mv2_pre_unification_crop_policy"
SCL_DIR = PROJECT_ROOT / "outputs_public" / "mv2_pre_unification_scl_qa"
OBS_DIR = PROJECT_ROOT / "outputs_public" / "mv2_pre_unification_observational_gate"
POLICY_DIR = PROJECT_ROOT / "outputs_public" / "mv2_pre_unification_policies"
SCHEMA_DIR = PROJECT_ROOT / "datasets" / "schemas"
CONFIG_EXAMPLE = PROJECT_ROOT / "configs" / "api_config.example.json"
LOCAL_CONFIG = PROJECT_ROOT / "configs" / "api_config.local.json"

ALLOWED_STATES = {
    "BLOCKED_NO_CONFIG",
    "BLOCKED_NO_TEMPORAL_WINDOW",
    "BLOCKED_NO_SENSOR_LINEAGE",
    "BLOCKED_NO_PRODUCT_ID",
    "BLOCKED_NO_NATIVE_RASTER",
    "METADATA_ONLY_READY",
    "METADATA_CONFIRMED",
    "CROP_AUTHORIZATION_READY",
    "LOCAL_ONLY_RASTER_READY",
    "SCL_QA_READY",
}

FORBIDDEN_STATES = {
    "SUPERVISED_READY",
    "SILVER_READY",
    "NEGATIVE_READY",
    "TRAINING_READY",
    "SANDBOX_READY",
    "DAY10_UNLOCKED",
}

DENY_TOKENS = {
    "mv2_13",
    "mv2_14",
    "mv2_15",
    "local_only",
    "raster",
    "rasters",
    "crop",
    "crops",
    "geotiff",
    ".tif",
    ".tiff",
    ".npz",
    "private",
    "quarantine",
    "credentials",
    "api_config.local.json",
    ".env",
}

ALLOW_TOKENS = {
    "mv2_12",
    "data_readiness",
    "spectral_reconstruction",
    "mv2_spectral_reconstruction",
    "mv2_data_05",
    "temporal_window",
    "sensor_family",
    "lineage",
    "metadata_probe",
    "readiness",
}

MV2_12_DATA_READINESS_RELPATHS = [
    "scripts/mv2_12_build_download_readiness.py",
    "scripts/mv2_12_build_missing_data_matrix.py",
    "scripts/mv2_12_data_readiness_common.py",
    "scripts/mv2_12_event_geometry_backlog.py",
    "scripts/mv2_12_scan_local_data_candidates.py",
    "scripts/mv2_12_sentinel_native_raster_backlog.py",
    "scripts/mv2_12_validate_no_heavy_public_outputs.py",
    "tests/test_mv2_12_data_readiness.py",
    "outputs_public/mv2_data_readiness/MV2_12_DATA_READINESS_REPORT.md",
    "outputs_public/mv2_data_readiness/MV2_12_EXECUTIVE_SUMMARY.md",
    "outputs_public/mv2_data_readiness/mv2_12_data_readiness_summary.json",
    "outputs_public/mv2_data_readiness/mv2_12_download_readiness.csv",
    "outputs_public/mv2_data_readiness/mv2_12_event_geometry_backlog.csv",
    "outputs_public/mv2_data_readiness/mv2_12_local_recovery_candidates.csv",
    "outputs_public/mv2_data_readiness/mv2_12_missing_data_matrix.csv",
    "outputs_public/mv2_data_readiness/mv2_12_sentinel_native_raster_backlog.csv",
]

CONTRACTS: dict[str, dict[str, Any]] = {
    "patch_binding_record": {
        "required": ["patch_id", "asset_id", "binding_status", "review_status"],
        "fields": {
            "patch_id": "str",
            "asset_id": "str",
            "city": "str|null",
            "aoi_wgs84": "GeoJSON Polygon|null",
            "aoi_crs": "str|null",
            "aoi_bbox_native": "[xmin,ymin,xmax,ymax]|null",
            "aoi_area_m2": "float|null",
            "binding_status": "NONE|WEAK|MEDIUM|STRONG|INVALID|CONFLICT",
            "source_ref": "str|null",
            "review_status": "PENDING|REVIEWED|BLOCKED|CONFLICT",
        },
    },
    "temporal_window_record": {
        "required": ["patch_id", "asset_id", "promotion_status", "review_status"],
        "fields": {
            "patch_id": "str",
            "asset_id": "str",
            "temporal_window_start": "datetime|null",
            "temporal_window_end": "datetime|null",
            "temporal_window_source": "str|null",
            "source_ref": "str|null",
            "temporal_evidence_status": "EMPTY|WEAK|MEDIUM|STRONG|CONFLICT|INVALID",
            "promotion_status": "BLOCKED_EMPTY|BLOCKED_WEAK|PROMOTED_PROBE_READY|CONFLICT",
            "review_status": "PENDING|REVIEWED|BLOCKED",
        },
    },
    "source_sensor_lineage_record": {
        "required": ["patch_id", "asset_id", "sensor_family", "spectral_eligible", "support_only"],
        "fields": {
            "patch_id": "str",
            "asset_id": "str",
            "slot_id": "str|null",
            "evidence_id": "str|null",
            "asset_ref": "str|null",
            "source_asset_ref": "str|null",
            "sensor_family": "SENTINEL_2|SENTINEL_1|DINO_DERIVED|PNG_RENDER|NPZ_EMBEDDING|UNKNOWN|CONFLICT",
            "spectral_eligible": "bool",
            "support_only": "bool",
            "blocked_reason": "str|null",
        },
    },
    "scene_binding_record": {
        "required": ["patch_id", "asset_id", "provider", "binding_status", "consensus_status"],
        "fields": {
            "patch_id": "str",
            "asset_id": "str",
            "provider": "GEE|CDSE_STAC|CDSE_ODATA|MANUAL",
            "collection": "str|null",
            "scene_id": "str|null",
            "product_id": "str|null",
            "mgrs_tile": "str|null",
            "datatake_identifier": "str|null",
            "datetime_utc": "str|null",
            "cloudy_pixel_percentage": "float|null",
            "nodata_pixel_percentage": "float|null",
            "thin_cirrus_percentage": "float|null",
            "geometry": "GeoJSON|null",
            "bbox": "list|null",
            "odata_id": "str|null",
            "odata_name": "str|null",
            "odata_s3path": "str|null",
            "odata_geofootprint": "GeoJSON|null",
            "binding_status": "NONE|WEAK|MEDIUM|STRONG|CONFLICT|INVALID",
            "consensus_status": "NO_CALL|NO_MATCH|SINGLE_PROVIDER|MULTI_PROVIDER_AGREE|CONFLICT",
        },
    },
    "local_raster_manifest_record": {
        "required": ["patch_id", "asset_id", "product_id", "is_public", "crop_status", "scl_qa_status"],
        "fields": {
            "patch_id": "str",
            "asset_id": "str",
            "product_id": "str",
            "local_only_path": "str|null",
            "sha256": "str|null",
            "size_bytes": "int|null",
            "raster_type": "GEOTIFF|COG|OTHER|UNKNOWN",
            "is_public": "false",
            "crop_status": "NOT_AUTHORIZED|AUTHORIZED|DONE|INVALID",
            "scl_qa_status": "NOT_RUN|READY|DONE|FAILED",
            "blocked_reason": "str|null",
        },
    },
    "gate_status_record": {
        "required": ["patch_id", "asset_id", "day10_status", "claim_level"],
        "fields": {
            "patch_id": "str",
            "asset_id": "str",
            "gate_a_temporal_spectral": "BLOCKED|READY_METADATA_ONLY|METADATA_CONFIRMED|CROP_AUTHORIZED|SCL_QA_DONE",
            "gate_b_observational": "BLOCKED|CONTEXTUAL|FOOTPRINT_READY|ADJUDICATION_READY|EVIDENCE_CONFIRMED",
            "gate_c_negatives": "BLOCKED|POLICY_READY|CANDIDATE_ONLY|FORMAL_READY",
            "gate_d_antileakage": "BLOCKED|POLICY_READY|SPLIT_READY",
            "day10_status": "BLOCKED|PARTIAL_LOCAL_ONLY|READY_REVIEW_ONLY",
            "day18_22_status": "BLOCKED|REVIEW_ONLY|READY_FOR_REVIEW",
            "claim_level": "NONE|READINESS|METADATA_ONLY|LOCAL_SPECTRAL_BASELINE|OBSERVATIONAL_REVIEW",
        },
    },
}


def run_git(args: list[str], cwd: Path = PROJECT_ROOT) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    return result.stdout.strip()


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def rel(path: Path, root: Path = PROJECT_ROOT) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_required(record: dict[str, Any], required: list[str]) -> None:
    missing = [field for field in required if record.get(field) in (None, "")]
    if missing:
        raise ValueError("missing required fields: " + ",".join(missing))


def classify_sensor_family(value: str) -> dict[str, Any]:
    family = (value or "UNKNOWN").strip().upper()
    mapping = {
        "SENTINEL_2": ("SENTINEL_2_ELIGIBLE", True, False, ""),
        "SENTINEL_1": ("SENTINEL_1_SUPPORT_ONLY", False, True, "SENTINEL_1_NAO_E_BASELINE_OPTICO_S2"),
        "DINO_DERIVED": ("DINO_DERIVED_BLOCKED", False, True, "DINO_DERIVED_NAO_E_RASTER_ESPECTRAL"),
        "PNG_RENDER": ("PNG_RENDER_BLOCKED", False, True, "PNG_RENDER_NAO_E_RASTER_ESPECTRAL"),
        "NPZ_EMBEDDING": ("NPZ_EMBEDDING_BLOCKED", False, True, "NPZ_NAO_E_RASTER_ESPECTRAL"),
        "CONFLICT": ("CONFLICT_BLOCKED", False, True, "SENSOR_LINEAGE_CONFLICT"),
        "UNKNOWN": ("UNKNOWN_BLOCKED", False, True, "BLOCKED_NO_SENSOR_LINEAGE"),
    }
    status, eligible, support_only, reason = mapping.get(family, mapping["UNKNOWN"])
    return {
        "sensor_family": family if family in mapping else "UNKNOWN",
        "lineage_classification": status,
        "spectral_eligible": eligible,
        "support_only": support_only,
        "blocked_reason": reason,
    }


def temporal_promotion_status(row: dict[str, Any]) -> str:
    start = str(row.get("temporal_window_start") or "").strip()
    end = str(row.get("temporal_window_end") or "").strip()
    source = str(row.get("temporal_window_source") or "").strip()
    source_ref = str(row.get("source_ref") or "").strip()
    if not start or not end or not source or not source_ref:
        return "BLOCKED_EMPTY"
    return "PROMOTED_PROBE_READY"


def can_authorize_crop(row: dict[str, Any]) -> tuple[bool, str]:
    if not row.get("product_id"):
        return False, "NOT_AUTHORIZED_NO_PRODUCT_ID"
    if row.get("sensor_family") != "SENTINEL_2":
        return False, "NOT_AUTHORIZED_NO_SENSOR"
    if not row.get("temporal_window_start") or not row.get("temporal_window_end"):
        return False, "NOT_AUTHORIZED_NO_TEMPORAL_WINDOW"
    if not row.get("aoi_wgs84") and not row.get("bbox"):
        return False, "NOT_AUTHORIZED_NO_AOI"
    if row.get("consensus_status") == "CONFLICT":
        return False, "NOT_AUTHORIZED_CONFLICT"
    return True, "AUTHORIZED_METADATA_ONLY"


def local_raster_manifest_guard(row: dict[str, Any]) -> None:
    if str(row.get("is_public")).lower() != "false":
        raise ValueError("raster manifest must keep is_public=false")
    path = str(row.get("local_only_path") or "")
    if "outputs_public" in path.replace("\\", "/").lower():
        raise ValueError("raster path cannot be under outputs_public")


def day10_gate(row: dict[str, Any]) -> str:
    if row.get("crop_status") != "AUTHORIZED" and row.get("crop_status") != "DONE":
        return "BLOCKED"
    if row.get("scl_qa_status") != "DONE":
        return "BLOCKED"
    return "READY_REVIEW_ONLY"


def unknown_is_negative(value: str) -> bool:
    return False if (value or "").strip().upper() == "UNKNOWN" else value.strip().upper() == "FORMAL_NEGATIVE"


def observational_silver_status(row: dict[str, Any]) -> str:
    if not row.get("geometry_wgs84"):
        return "BLOCKED_INSUFFICIENT_EVIDENCE"
    if row.get("adjudication_status") != "ADJUDICATED":
        return "BLOCKED_INSUFFICIENT_EVIDENCE"
    if not row.get("uncertainty_class"):
        return "BLOCKED_INSUFFICIENT_EVIDENCE"
    if row.get("evidence_sources") == "TEXTUAL_ANCHOR_ONLY":
        return "TEXTUAL_ANCHOR_ONLY"
    return "DIGITIZED_PENDING_ADJUDICATION"


def git_state(root: Path = PROJECT_ROOT) -> dict[str, str]:
    return {
        "status_short": run_git(["status", "--short"], root),
        "status": run_git(["status"], root),
        "branch": run_git(["branch", "--show-current"], root),
        "top_commit": run_git(["log", "--oneline", "-1"], root),
        "last_commits": run_git(["log", "--oneline", "-5"], root),
        "worktrees": run_git(["worktree", "list"], root),
        "staged": run_git(["diff", "--cached", "--name-only"], root),
    }


def categorize(path: str) -> tuple[str, bool, str]:
    low = path.lower().replace("\\", "/")
    normalized = path.replace("\\", "/")
    if normalized in MV2_12_DATA_READINESS_RELPATHS:
        return "MV2_12_DATA_READINESS", True, ""
    if any(tok in low for tok in ["mv2_13", "mv2_14", "mv2_15", "cronograma_engine"]):
        return "MV2_13_OR_LATER", False, "fora_do_escopo_mv2_12"
    if any(tok in low for tok in ["local_only", "private", "quarantine", ".env", "api_config.local.json"]):
        return "PRIVATE_OR_HEAVY", False, "privado_ou_config_local"
    if any(low.endswith(ext) for ext in [".tif", ".tiff", ".geotiff", ".npz", ".npy", ".jp2", ".safe"]):
        return "PRIVATE_OR_HEAVY", False, "binario_ou_pesado"
    for idx in range(1, 6):
        if f"mv2_data_0{idx}" in low or f"data_0{idx}" in low:
            return f"DATA_0{idx}", True, ""
    if "mv2_12" in low and "data_readiness" in low:
        return "MV2_12_DATA_READINESS", True, ""
    if "mv2_12" in low or "mv2_spectral_reconstruction" in low:
        return "MV2_12_SPECTRAL", True, ""
    if any(tok in low for tok in ALLOW_TOKENS):
        return "UNKNOWN", True, ""
    if any(tok in low for tok in DENY_TOKENS):
        return "PRIVATE_OR_HEAVY", False, "denylist"
    return "UNKNOWN", False, "sem_assinatura_allowlist"


def collect_candidate_paths() -> list[Path]:
    roots = ["scripts", "tests", "datasets/schemas", "outputs_public"]
    paths: list[Path] = []
    for root in roots:
        base = PROJECT_ROOT / root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file():
                low = rel(path).lower()
                if any(tok in low for tok in (ALLOW_TOKENS | {"mv2_13", "mv2_14", "mv2_15", "data_01", "data_02", "data_03", "data_04", "data_05"})):
                    paths.append(path)
    return sorted(set(paths))


def write_initial_audit(stamp: str) -> Path:
    state = git_state(PROJECT_ROOT)
    mv2_state = git_state(MV2_WORKTREE) if MV2_WORKTREE.exists() else {}
    status_lines = state["status_short"].splitlines()
    data05 = PROJECT_ROOT / "outputs_public" / "mv2_data_temporal_window_intake"
    mv212_data = PROJECT_ROOT / "outputs_public" / "mv2_data_readiness"
    mv212_spectral = MV2_WORKTREE / "outputs_public" / "mv2_spectral_reconstruction"
    risk_13 = [line for line in status_lines if any(tok in line.lower() for tok in ["mv2_13", "mv2_14", "mv2_15"])]
    local_heavy = [line for line in status_lines if any(tok in line.lower() for tok in ["local_only", ".tif", ".tiff", ".npz", "crop"])]
    out = EXEC_REPORTS / f"revp_pre_unification_initial_audit_{stamp}.md"
    text = f"""# Auditoria inicial pre-unificacao REV-P

## Estado Git atual
- branch atual: {state['branch']}
- top commit: {state['top_commit']}
- staged files: {len([x for x in state['staged'].splitlines() if x.strip()])}
- entradas untracked/modificadas relevantes: {len(status_lines)}

## Worktrees detectados
```text
{state['worktrees']}
```

## MV2 worktree esperado
- path: {MV2_WORKTREE}
- existe: {MV2_WORKTREE.exists()}
- branch: {mv2_state.get('branch', 'NAO_VERIFICADO')}
- top commit: {mv2_state.get('top_commit', 'NAO_VERIFICADO')}
- staged files: {len([x for x in mv2_state.get('staged', '').splitlines() if x.strip()])}

## Presenca de blocos
- DATA-05: {'PRESENTE' if data05.exists() else 'AUSENTE'} ({rel(data05)})
- MV2-12 Data Readiness: {'PRESENTE' if mv212_data.exists() else 'AUSENTE'} ({rel(mv212_data)})
- MV2-12 Spectral Reconstruction no MV2 worktree: {'PRESENTE' if mv212_spectral.exists() else 'AUSENTE'} ({mv212_spectral})

## Riscos detectados
- arquivos MV2-13/MV2-14/MV2-15 misturados: {len(risk_13)}
- risco local_only/rasters/crops/pesados em status: {len(local_heavy)}
- acao segura: excluir MV2-13+ da consolidacao e manter rasters/crops fora de outputs_public.

## Status curto capturado
```text
{state['status_short']}
```

## Ultimos commits
```text
{state['last_commits']}
```
"""
    write_text(out, text)
    return out


def write_inventory(stamp: str) -> Path:
    rows: list[dict[str, Any]] = []
    for path in collect_candidate_paths():
        path_rel = rel(path)
        category, include, deny = categorize(path_rel)
        rows.append(
            {
                "path": path_rel,
                "exists": str(path.exists()).lower(),
                "size_bytes": path.stat().st_size if path.exists() else "",
                "sha256": sha256(path),
                "category": category,
                "worktree": "REV-P",
                "include_candidate": str(include).lower(),
                "deny_reason": deny,
            }
        )
    out = EXEC_REPORTS / f"revp_pre_unification_file_inventory_{stamp}.csv"
    write_csv(out, ["path", "exists", "size_bytes", "sha256", "category", "worktree", "include_candidate", "deny_reason"], rows)
    return out


def copy_mv2_12_data_readiness(stamp: str) -> tuple[Path, Path, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    included = 0
    refused = 0
    for relpath in MV2_12_DATA_READINESS_RELPATHS:
        src = PROJECT_ROOT / relpath
        dst = MV2_WORKTREE / relpath
        before = sha256(src)
        category, include, deny = categorize(relpath)
        refusal = deny
        copied = False
        if not MV2_WORKTREE.exists():
            refusal = "worktree_mv2_ausente"
        elif not src.exists():
            refusal = "origem_ausente"
        elif not include or category != "MV2_12_DATA_READINESS":
            refusal = refusal or "fora_allowlist_mv2_12_data_readiness"
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied = True
            included += 1
        if not copied:
            refused += 1
        rows.append(
            {
                "source_path": relpath,
                "destination_path": str(dst),
                "source_sha256_before": before,
                "destination_sha256_after": sha256(dst),
                "size_bytes": src.stat().st_size if src.exists() else "",
                "category": category,
                "copied": str(copied).lower(),
                "deny_reason": refusal,
            }
        )
    manifest = EXEC_REPORTS / f"revp_mv2_12_consolidation_manifest_{stamp}.csv"
    write_csv(
        manifest,
        [
            "source_path",
            "destination_path",
            "source_sha256_before",
            "destination_sha256_after",
            "size_bytes",
            "category",
            "copied",
            "deny_reason",
        ],
        rows,
    )
    spectral_count = len(list((MV2_WORKTREE / "outputs_public" / "mv2_spectral_reconstruction").glob("*"))) if MV2_WORKTREE.exists() else 0
    report = EXEC_REPORTS / f"revp_mv2_12_consolidation_report_{stamp}.md"
    write_text(
        report,
        f"""# Consolidacao MV2-12

- worktree MV2 alvo: {MV2_WORKTREE}
- arquivos Data Readiness copiados: {included}
- arquivos recusados: {refused}
- arquivos Spectral Reconstruction presentes no worktree MV2: {spectral_count}
- MV2-13/14/15: intocados; apenas inventariados como fora de escopo.
- staging: nenhum arquivo foi stageado.
- risco residual: worktrees permanecem com arquivos nao rastreados para revisao humana.

Manifesto: {rel(manifest)}
""",
    )
    return manifest, report, {"mv2_12_copied": included, "mv2_12_refused": refused, "mv2_12_spectral_files": spectral_count}


def write_contracts() -> list[Path]:
    written: list[Path] = []
    rows = []
    for name, spec in CONTRACTS.items():
        schema = {
            "schema_id": f"revp_mv2_pre_unification_{name}",
            "description": "Lightweight pre-unification contract; no supervised training or public raster output.",
            "required_fields": spec["required"],
            "fields": spec["fields"],
            "forbidden_states": sorted(FORBIDDEN_STATES),
            "allowed_execution_states": sorted(ALLOWED_STATES),
        }
        out = SCHEMA_DIR / f"schema_mv2_pre_unification_{name}.json"
        write_json(out, schema)
        written.append(out)
        rows.append({"contract": name, "schema_path": rel(out), "required_fields": "|".join(spec["required"])})
    manifest = CONTRACT_DIR / "revp_pre_unification_contract_manifest.csv"
    write_csv(manifest, ["contract", "schema_path", "required_fields"], rows)
    written.append(manifest)
    return written


def select_seed_rows() -> list[dict[str, str]]:
    batch = read_csv(PROJECT_ROOT / "outputs_public" / "mv2_data_corpus_metadata_probe" / "mv2_data_04_corpus_metadata_batch.csv")
    gate = read_csv(PROJECT_ROOT / "outputs_public" / "mv2_data_temporal_window_intake" / "mv2_data_05_temporal_promotion_gate.csv")
    gate_by_target = {row.get("target_id", ""): row for row in gate}
    selected = [row for row in batch if str(row.get("selected_for_probe", "")).lower() == "true"] or batch
    return selected[:10], gate_by_target


def write_temporal_seed() -> dict[str, Any]:
    selected, gate_by_target = select_seed_rows()
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(selected, 1):
        gate = gate_by_target.get(row.get("target_id", ""), {})
        start = gate.get("temporal_window_start", "")
        end = gate.get("temporal_window_end", "")
        promotion = "PROMOTED_PROBE_READY" if start and end and gate.get("source_ref") else "BLOCKED_NO_TEMPORAL_WINDOW"
        rows.append(
            {
                "seed_rank": idx,
                "patch_id": row.get("patch_id", ""),
                "asset_id": row.get("asset_id", ""),
                "city": row.get("region", ""),
                "bbox": row.get("bbox", ""),
                "crs": row.get("crs", ""),
                "temporal_window_start": start,
                "temporal_window_end": end,
                "temporal_window_source": gate.get("temporal_window_source", ""),
                "source_ref": gate.get("source_ref", ""),
                "review_status": gate.get("review_status", "PENDING_HUMAN_FILL"),
                "promotion_status": promotion,
                "blocked_reason": "" if promotion == "PROMOTED_PROBE_READY" else "BLOCKED_NO_TEMPORAL_WINDOW",
            }
        )
    write_csv(
        SEED_DIR / "revp_temporal_window_seed_10.csv",
        [
            "seed_rank",
            "patch_id",
            "asset_id",
            "city",
            "bbox",
            "crs",
            "temporal_window_start",
            "temporal_window_end",
            "temporal_window_source",
            "source_ref",
            "review_status",
            "promotion_status",
            "blocked_reason",
        ],
        rows,
    )
    summary = {
        "seed_targets": len(rows),
        "promoted_probe_ready": sum(1 for row in rows if row["promotion_status"] == "PROMOTED_PROBE_READY"),
        "blocked_no_temporal_window": sum(1 for row in rows if row["promotion_status"] == "BLOCKED_NO_TEMPORAL_WINDOW"),
    }
    write_json(SEED_DIR / "revp_temporal_window_seed_10_summary.json", summary)
    write_text(
        SEED_DIR / "revp_temporal_window_seed_10_report.md",
        f"""# Seed temporal inicial

- targets selecionados: {summary['seed_targets']}
- promovidos para probe-ready: {summary['promoted_probe_ready']}
- bloqueados por janela temporal ausente: {summary['blocked_no_temporal_window']}
- criterio: batch DATA-04 selecionado, sem inventar janela temporal.
""",
    )
    return summary


def write_sensor_lineage_seed() -> dict[str, Any]:
    seed = read_csv(SEED_DIR / "revp_temporal_window_seed_10.csv")
    resolution = read_csv(PROJECT_ROOT / "outputs_public" / "tables" / "revp_temporal_sensor_family_resolution_mv1.csv")
    resolution_by_patch = {row.get("canonical_patch_id", ""): row for row in resolution}
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for idx, row in enumerate(seed, 1):
        source = resolution_by_patch.get(row.get("patch_id", ""), {})
        family = source.get("resolved_required_sensor_family", "")
        if "sentinel_2" in family.lower():
            sensor = "SENTINEL_2"
        elif "sentinel_1" in family.lower():
            sensor = "SENTINEL_1"
        elif "dino" in family.lower():
            sensor = "DINO_DERIVED"
        elif "png" in family.lower():
            sensor = "PNG_RENDER"
        elif "npz" in family.lower():
            sensor = "NPZ_EMBEDDING"
        elif "conflict" in family.lower():
            sensor = "CONFLICT"
        else:
            sensor = "UNKNOWN"
        cls = classify_sensor_family(sensor)
        counts[cls["lineage_classification"]] += 1
        rows.append(
            {
                "seed_rank": idx,
                "patch_id": row.get("patch_id", ""),
                "asset_id": row.get("asset_id", ""),
                "slot_id": source.get("required_slot_index", ""),
                "evidence_id": source.get("evidence_asset_ids", ""),
                "asset_ref": source.get("evidence_asset_ids", ""),
                "source_asset_ref": "",
                "sensor_family": cls["sensor_family"],
                "lineage_classification": cls["lineage_classification"],
                "spectral_eligible": str(cls["spectral_eligible"]).lower(),
                "support_only": str(cls["support_only"]).lower(),
                "blocked_reason": cls["blocked_reason"],
            }
        )
    write_csv(
        SEED_DIR / "revp_source_sensor_lineage_seed_10.csv",
        [
            "seed_rank",
            "patch_id",
            "asset_id",
            "slot_id",
            "evidence_id",
            "asset_ref",
            "source_asset_ref",
            "sensor_family",
            "lineage_classification",
            "spectral_eligible",
            "support_only",
            "blocked_reason",
        ],
        rows,
    )
    summary = {
        "seed_targets": len(rows),
        "sentinel_2_eligible": counts.get("SENTINEL_2_ELIGIBLE", 0),
        "sentinel_1_support_only": counts.get("SENTINEL_1_SUPPORT_ONLY", 0),
        "unknown_blocked": counts.get("UNKNOWN_BLOCKED", 0),
        "dino_derived_blocked": counts.get("DINO_DERIVED_BLOCKED", 0),
        "png_render_blocked": counts.get("PNG_RENDER_BLOCKED", 0),
        "npz_embedding_blocked": counts.get("NPZ_EMBEDDING_BLOCKED", 0),
        "conflict_blocked": counts.get("CONFLICT_BLOCKED", 0),
    }
    write_json(SEED_DIR / "revp_source_sensor_lineage_seed_10_summary.json", summary)
    write_text(
        SEED_DIR / "revp_source_sensor_lineage_seed_10_report.md",
        f"""# Seed source sensor lineage

- targets: {summary['seed_targets']}
- Sentinel-2 eligible: {summary['sentinel_2_eligible']}
- Sentinel-1 support-only: {summary['sentinel_1_support_only']}
- unknown/blocked: {summary['unknown_blocked']}
- regra: sem cadeia de origem rastreavel, sensor fica bloqueado.
""",
    )
    return summary


def ensure_config_example() -> None:
    data = {
        "allow_network": False,
        "allow_metadata_calls": False,
        "allow_raster_download": False,
        "allow_canary_download": False,
        "providers": {
            "GEE": {"enabled": False, "project_id_env": "REV_P_GEE_PROJECT_ID"},
            "CDSE_STAC": {"enabled": False, "base_url": "https://stac.dataspace.copernicus.eu/v1"},
            "CDSE_ODATA": {"enabled": False, "base_url": "https://catalogue.dataspace.copernicus.eu/odata/v1"},
        },
    }
    write_json(CONFIG_EXAMPLE, data)


def metadata_preflight() -> dict[str, Any]:
    ensure_config_example()
    if not LOCAL_CONFIG.exists():
        summary = {
            "config_local_present": False,
            "preflight_status": "BLOCKED_NO_CONFIG",
            "allow_network": False,
            "allow_metadata_calls": False,
            "allow_raster_download": False,
            "allow_canary_download": False,
            "metadata_calls_executed": 0,
            "downloads_executed": 0,
        }
    else:
        cfg = json.loads(LOCAL_CONFIG.read_text(encoding="utf-8"))
        allow_network = bool(cfg.get("allow_network"))
        allow_metadata = bool(cfg.get("allow_metadata_calls"))
        status = "METADATA_ONLY_READY" if allow_network and allow_metadata and not cfg.get("allow_raster_download") else "BLOCKED_NO_CONFIG"
        summary = {
            "config_local_present": True,
            "preflight_status": status,
            "allow_network": allow_network,
            "allow_metadata_calls": allow_metadata,
            "allow_raster_download": bool(cfg.get("allow_raster_download")),
            "allow_canary_download": bool(cfg.get("allow_canary_download")),
            "metadata_calls_executed": 0,
            "downloads_executed": 0,
        }
    write_json(SEED_DIR / "revp_metadata_preflight_summary.json", summary)
    write_text(
        SEED_DIR / "revp_metadata_preflight_report.md",
        f"""# Metadata-only preflight

- config local presente: {summary['config_local_present']}
- status: {summary['preflight_status']}
- chamadas metadata executadas: 0
- downloads executados: 0
""",
    )
    return summary


def write_metadata_probe_outputs(preflight: dict[str, Any]) -> dict[str, Any]:
    seed = read_csv(SEED_DIR / "revp_temporal_window_seed_10.csv")
    lineage = {row.get("asset_id", ""): row for row in read_csv(SEED_DIR / "revp_source_sensor_lineage_seed_10.csv")}
    rows = []
    for row in seed:
        lin = lineage.get(row.get("asset_id", ""), {})
        eligible = row.get("temporal_window_start") and row.get("temporal_window_end") and lin.get("lineage_classification") == "SENTINEL_2_ELIGIBLE"
        can_call = eligible and preflight.get("preflight_status") == "METADATA_ONLY_READY"
        rows.append(
            {
                "patch_id": row.get("patch_id", ""),
                "asset_id": row.get("asset_id", ""),
                "provider": "NO_CALL",
                "scene_id": "",
                "product_id": "",
                "datetime_utc": "",
                "mgrs_tile": "",
                "binding_status": "NONE",
                "consensus_status": "NO_CALL",
                "blocked_reason": "" if can_call else "BLOCKED_PRECONDITIONS_METADATA_ONLY",
            }
        )
    fields = ["patch_id", "asset_id", "provider", "scene_id", "product_id", "datetime_utc", "mgrs_tile", "binding_status", "consensus_status", "blocked_reason"]
    for name in ["gee", "stac", "odata"]:
        write_csv(METADATA_DIR / f"revp_seed_10_{name}_metadata_probe.csv", fields, rows)
    write_csv(METADATA_DIR / "revp_seed_10_lineage_consensus.csv", fields, rows)
    summary = {
        "targets": len(rows),
        "metadata_calls_executed": 0,
        "downloads_executed": 0,
        "rasters_created": 0,
        "crops_created": 0,
        "confirmed_lineage": 0,
        "no_call": len(rows),
    }
    write_json(METADATA_DIR / "revp_seed_10_metadata_probe_summary.json", summary)
    write_text(
        METADATA_DIR / "revp_seed_10_metadata_probe_report.md",
        """# Metadata probe seed 10

- resultado: NO_CALL para todos os targets.
- motivo: precondicoes fail-closed nao atendidas.
- downloads: 0
- rasters: 0
- crops: 0
""",
    )
    return summary


def write_crop_policy() -> dict[str, Any]:
    seed = read_csv(SEED_DIR / "revp_temporal_window_seed_10.csv")
    lineage = {row.get("asset_id", ""): row for row in read_csv(SEED_DIR / "revp_source_sensor_lineage_seed_10.csv")}
    rows = []
    counts: Counter[str] = Counter()
    for row in seed:
        lin = lineage.get(row.get("asset_id", ""), {})
        auth_row = {
            "product_id": "",
            "sensor_family": lin.get("sensor_family", "UNKNOWN"),
            "temporal_window_start": row.get("temporal_window_start", ""),
            "temporal_window_end": row.get("temporal_window_end", ""),
            "bbox": row.get("bbox", ""),
            "consensus_status": "NO_CALL",
        }
        _, status = can_authorize_crop(auth_row)
        counts[status] += 1
        rows.append(
            {
                "patch_id": row.get("patch_id", ""),
                "asset_id": row.get("asset_id", ""),
                "authorization_status": status,
                "product_id": "",
                "downloads_executed": 0,
                "blocked_reason": status,
            }
        )
    write_csv(CROP_DIR / "revp_crop_authorization_candidates.csv", ["patch_id", "asset_id", "authorization_status", "product_id", "downloads_executed", "blocked_reason"], rows)
    summary = {"targets": len(rows), "authorized_metadata_only": counts.get("AUTHORIZED_METADATA_ONLY", 0), "not_authorized": len(rows) - counts.get("AUTHORIZED_METADATA_ONLY", 0), "downloads_executed": 0}
    write_json(CROP_DIR / "revp_crop_authorization_summary.json", summary)
    write_text(CROP_DIR / "revp_crop_authorization_report.md", "# Crop authorization\n\n- nenhum crop foi baixado ou criado.\n- autorizacoes dependem de product_id, S2, janela temporal, AOI e consenso.")
    return summary


def write_scl_qa() -> dict[str, Any]:
    rows = []
    for row in read_csv(SEED_DIR / "revp_temporal_window_seed_10.csv"):
        rows.append(
            {
                "patch_id": row.get("patch_id", ""),
                "asset_id": row.get("asset_id", ""),
                "scl_qa_status": "NOT_RUN_NO_LOCAL_RASTER",
                "cloud_local_ratio": "",
                "shadow_local_ratio": "",
                "valid_local_ratio": "",
                "non_masked_pixel_count": "",
                "scl_class_histogram": "",
                "blocked_reason": "BLOCKED_NO_NATIVE_RASTER",
            }
        )
    write_csv(SCL_DIR / "revp_scl_qa_readiness.csv", ["patch_id", "asset_id", "scl_qa_status", "cloud_local_ratio", "shadow_local_ratio", "valid_local_ratio", "non_masked_pixel_count", "scl_class_histogram", "blocked_reason"], rows)
    summary = {"targets": len(rows), "scl_qa_done": 0, "not_run_no_local_raster": len(rows)}
    write_json(SCL_DIR / "revp_scl_qa_summary.json", summary)
    write_text(SCL_DIR / "revp_scl_qa_report.md", "# SCL local QA\n\n- modulo preparado como contrato de metricas.\n- execucao bloqueada sem raster local-only manifestado.")
    return summary


def write_observational_gate() -> None:
    event_fields = [
        "event_id",
        "city",
        "event_type",
        "event_window_start",
        "event_window_end",
        "geometry_status",
        "geometry_wgs84",
        "crs",
        "uncertainty_class",
        "uncertainty_reason",
        "buffer_uncertainty_m",
        "source_ref",
        "evidence_sources",
        "adjudication_status",
        "silver_candidate",
        "silver_status",
        "blocked_reason",
    ]
    write_csv(OBS_DIR / "event_evidence_record.csv", event_fields, [])
    write_csv(OBS_DIR / "event_geometry_backlog.csv", ["event_id", "city", "geometry_status", "next_action", "blocked_reason"], [])
    write_csv(OBS_DIR / "adjudication_template.csv", ["event_id", "reviewer", "adjudication_status", "decision_basis", "review_date"], [])
    write_text(OBS_DIR / "uncertainty_policy.md", "# Uncertainty policy\n\nEventos sem geometria, incerteza e adjudicacao permanecem bloqueados. SILVER_HIGH, GOLD e SUPERVISED_READY sao proibidos nesta execucao.")


def write_policies() -> None:
    write_text(POLICY_DIR / "revp_formal_negative_policy.md", "# Formal negative policy\n\nUnknown nao e negativo. Ausencia de evidencia nao e classe 0. Curitiba nao e negativo automatico. Negativo formal exige pareamento por contexto, cidade, hidrologia, sensor e janela.")
    write_text(POLICY_DIR / "revp_antileakage_split_policy.md", "# Anti-leakage split policy\n\nSplit futuro deve ser por evento, cidade ou bloco espacial; nunca random patch split. Cenas Sentinel iguais ou vizinhas nao podem vazar entre train/test.")
    write_text(POLICY_DIR / "revp_unknown_is_not_negative_guardrail.md", "# Unknown is not negative\n\nUNKNOWN e estado bloqueado/revisavel, nao negativo formal e nao classe treinavel.")


def write_readiness(stamp: str, summaries: dict[str, Any]) -> tuple[Path, Path]:
    data05 = json.loads((PROJECT_ROOT / "outputs_public" / "mv2_data_temporal_window_intake" / "mv2_data_05_summary.json").read_text(encoding="utf-8"))
    decision = "READY_FOR_MV2_16_DRY_RUN"
    if data05.get("total_input_rows", 0) == 0 or summaries["mv2_12"].get("mv2_12_copied", 0) == 0:
        decision = "NOT_READY_FOR_MV2_16"
    if summaries["temporal"].get("promoted_probe_ready", 0) and summaries["lineage"].get("sentinel_2_eligible", 0) and summaries["preflight"].get("preflight_status") == "METADATA_ONLY_READY":
        decision = "READY_FOR_MV2_16_METADATA_ONLY"
    summary = {
        "data05_closed_as_intake": True,
        "data05_promoted_windows": data05.get("temporal_promoted_strong", 0) + data05.get("temporal_promoted_partial", 0),
        "mv2_12_consolidated_for_review": summaries["mv2_12"].get("mv2_12_copied", 0) > 0,
        "seed_targets": summaries["temporal"].get("seed_targets", 0),
        "seed_temporal_valid": summaries["temporal"].get("promoted_probe_ready", 0),
        "sentinel_2_eligible": summaries["lineage"].get("sentinel_2_eligible", 0),
        "metadata_calls_executed": summaries["metadata"].get("metadata_calls_executed", 0),
        "downloads_executed": summaries["metadata"].get("downloads_executed", 0),
        "rasters_created": summaries["metadata"].get("rasters_created", 0),
        "crops_created": summaries["metadata"].get("crops_created", 0),
        "gate_a_temporal_spectral": "BLOCKED",
        "gate_b_observational": "GEOMETRY_BACKLOG_READY",
        "gate_c_negatives": "POLICY_READY",
        "gate_d_antileakage": "POLICY_READY",
        "mv2_16_decision": decision,
    }
    json_out = EXEC_REPORTS / f"revp_pre_unification_readiness_summary_{stamp}.json"
    write_json(json_out, summary)
    report = EXEC_REPORTS / f"revp_pre_unification_readiness_report_{stamp}.md"
    write_text(
        report,
        f"""# Readiness pre-unificacao MV2

## Frente A
- MV2-12 consolidado para revisao: {summary['mv2_12_consolidated_for_review']}
- arquivos MV2-12 Data Readiness copiados: {summaries['mv2_12'].get('mv2_12_copied', 0)}
- MV2-13/14/15 preservados: true

## Frente B
- DATA-05 fechado como intake: true
- janelas promovidas: {summary['data05_promoted_windows']}
- seed targets: {summary['seed_targets']}
- targets com janela temporal valida: {summary['seed_temporal_valid']}
- Sentinel-2 eligible: {summary['sentinel_2_eligible']}
- chamadas metadata-only: {summary['metadata_calls_executed']}
- downloads/rasters/crops: {summary['downloads_executed']}/{summary['rasters_created']}/{summary['crops_created']}

## Gates
- Gate A temporal-espectral: {summary['gate_a_temporal_spectral']}
- Gate B observacional: {summary['gate_b_observational']}
- Gate C negativos: {summary['gate_c_negatives']}
- Gate D anti-leakage: {summary['gate_d_antileakage']}

## Decisao MV2-16
- {summary['mv2_16_decision']}
""",
    )
    return report, json_out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", default=now_stamp())
    args = parser.parse_args(argv)
    stamp = args.timestamp

    write_initial_audit(stamp)
    write_inventory(stamp)
    _, _, mv2_12_summary = copy_mv2_12_data_readiness(stamp)
    write_contracts()
    temporal_summary = write_temporal_seed()
    lineage_summary = write_sensor_lineage_seed()
    preflight_summary = metadata_preflight()
    metadata_summary = write_metadata_probe_outputs(preflight_summary)
    crop_summary = write_crop_policy()
    scl_summary = write_scl_qa()
    write_observational_gate()
    write_policies()
    write_readiness(
        stamp,
        {
            "mv2_12": mv2_12_summary,
            "temporal": temporal_summary,
            "lineage": lineage_summary,
            "preflight": preflight_summary,
            "metadata": metadata_summary,
            "crop": crop_summary,
            "scl": scl_summary,
        },
    )
    print(f"[mv2_pre_unification] concluido stamp={stamp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
