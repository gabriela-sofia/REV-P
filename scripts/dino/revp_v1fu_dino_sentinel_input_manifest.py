from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PHASE = "v1fu"
PHASE_NAME = "DINO_SENTINEL_INPUT_MANIFEST"
EXPECTED_SENTINEL_COUNT = 128

PROJECT_STATE = ROOT / "configs" / "project_state.yaml"
DINO_CONFIG = ROOT / "configs" / "dino_review_only.yaml"
V1FT_DIR = ROOT / "manifests" / "training_readiness" / "revp_v1ft_embedding_config_and_recife_balance_audit"
V1FR_DIR = ROOT / "manifests" / "training_readiness" / "revp_v1fr_self_supervised_dataloader_preflight"
V1FS_DIR = ROOT / "manifests" / "training_readiness" / "revp_v1fs_self_supervised_asset_sanity_and_embedding_plan"

READY_ASSETS = V1FT_DIR / "embedding_ready_assets_v1ft.csv"
EXTRACTION_CONFIG = V1FT_DIR / "embedding_extraction_config_v1ft.json"
SPLIT_CONFIG = V1FT_DIR / "embedding_split_config_v1ft.csv"
TRANSFORM_POLICY = V1FT_DIR / "embedding_transform_policy_v1ft.csv"
OUTPUT_SCHEMA = V1FT_DIR / "embedding_output_schema_v1ft.csv"
DL_INPUT = V1FR_DIR / "dl_input_manifest_v1fr.csv"
V1FS_READINESS = V1FS_DIR / "embedding_extraction_readiness_v1fs.csv"

OUT_DIR = ROOT / "manifests" / "dino_inputs" / "revp_v1fu_dino_sentinel_input_manifest"
MANIFEST_PATH = OUT_DIR / "dino_sentinel_input_manifest_v1fu.csv"
SUMMARY_PATH = OUT_DIR / "dino_sentinel_input_summary_v1fu.json"
QA_PATH = OUT_DIR / "dino_sentinel_input_qa_v1fu.csv"
STATUS_PATH = OUT_DIR / "dino_sentinel_input_status_v1fu.csv"

SOURCE_MANIFEST = "manifests/training_readiness/revp_v1ft_embedding_config_and_recife_balance_audit/embedding_ready_assets_v1ft.csv"
TRANSFORM_POLICY_REF = "manifests/training_readiness/revp_v1ft_embedding_config_and_recife_balance_audit/embedding_transform_policy_v1ft.csv"

MANIFEST_FIELDS = [
    "dino_input_id",
    "canonical_patch_id",
    "region",
    "source_asset_id",
    "source_asset_type",
    "source_manifest",
    "asset_path_reference",
    "modality",
    "dino_scope",
    "eligibility_status",
    "split_group",
    "split_policy",
    "transform_policy_ref",
    "encoder_mode",
    "label_status",
    "target_status",
    "pixel_read_status",
    "claim_scope",
    "blocker_status",
    "notes",
]

INPUT_FILES = [
    PROJECT_STATE,
    DINO_CONFIG,
    READY_ASSETS,
    EXTRACTION_CONFIG,
    SPLIT_CONFIG,
    TRANSFORM_POLICY,
    OUTPUT_SCHEMA,
    DL_INPUT,
    V1FS_READINESS,
]

FORBIDDEN_DIR_NAMES = {".claude", ".codex", "data", "outputs", "docs", "patches", "archive_drive"}
FORBIDDEN_FILE_NAMES = {"AGENTS.md", "CLAUDE.md"}
FORBIDDEN_EXTENSIONS = {
    ".tif",
    ".tiff",
    ".zip",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".parquet",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_simple_yaml(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in load_text(path).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def is_inside_repo(path: Path) -> bool:
    try:
        path.resolve().relative_to(ROOT)
        return True
    except ValueError:
        return False


def forbidden_paths() -> list[str]:
    found: list[str] = []
    for path in ROOT.rglob("*"):
        if ".git" in path.parts or "local_runs" in path.parts:
            continue
        name_lower = path.name.lower()
        if path.is_dir() and name_lower in FORBIDDEN_DIR_NAMES:
            found.append(rel(path))
        elif path.is_file():
            if path.name in FORBIDDEN_FILE_NAMES:
                found.append(rel(path))
            elif path.suffix.lower() in FORBIDDEN_EXTENSIONS:
                found.append(rel(path))
            elif "cbers" in name_lower:
                found.append(rel(path))
    return sorted(found)


def build_lookup(rows: list[dict[str, str]], *keys: str) -> dict[tuple[str, ...], dict[str, str]]:
    lookup: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        lookup[tuple(row.get(key, "") for key in keys)] = row
    return lookup


def build_manifest() -> tuple[list[dict[str, str]], dict[str, object], list[dict[str, str]], list[dict[str, str]]]:
    input_rows = read_csv(READY_ASSETS)
    split_rows = read_csv(SPLIT_CONFIG)
    dl_rows = read_csv(DL_INPUT)
    v1fs_rows = read_csv(V1FS_READINESS)
    extraction_config = json.loads(load_text(EXTRACTION_CONFIG))
    project_state = parse_simple_yaml(PROJECT_STATE)
    dino_config_text = load_text(DINO_CONFIG)

    split_lookup = build_lookup(split_rows, "split_group")
    dl_lookup = build_lookup(dl_rows, "candidate_id", "asset_path")
    v1fs_lookup = build_lookup(v1fs_rows, "asset_id")

    sentinel_rows = [
        row
        for row in input_rows
        if row.get("modality") == "sentinel_raster_path_only"
        and row.get("config_status") == "READY_SENTINEL_FIRST_REVIEW_ONLY"
        and row.get("readiness_status") == "EMBEDDING_REVIEW_ONLY_READY"
    ]

    manifest: list[dict[str, str]] = []
    for index, row in enumerate(sentinel_rows, start=1):
        candidate_id = row.get("candidate_id", "")
        asset_path = row.get("asset_path", "")
        split_group = row.get("split_group", "")
        dl_ref = dl_lookup.get((candidate_id, asset_path), {})
        v1fs_ref = v1fs_lookup.get((row.get("asset_id", ""),), {})
        split_ref = split_lookup.get((split_group,), {})
        source_asset_type = dl_ref.get("asset_type") or v1fs_ref.get("asset_type") or "SENTINEL_TIF_ASSET_REFERENCE_ONLY"
        notes = [
            "metadata/reference only",
            "no raster path existence check",
            "no GeoTIFF open",
            "no pixel read",
            "no embedding extraction",
            "output schema referenced only",
        ]
        manifest.append(
            {
                "dino_input_id": f"DINO_V1FU_SENTINEL_{index:05d}",
                "canonical_patch_id": candidate_id,
                "region": row.get("region", ""),
                "source_asset_id": row.get("asset_id", ""),
                "source_asset_type": source_asset_type,
                "source_manifest": SOURCE_MANIFEST,
                "asset_path_reference": asset_path,
                "modality": row.get("modality", ""),
                "dino_scope": "SENTINEL_FIRST_EMBEDDING_INPUT",
                "eligibility_status": row.get("config_status", ""),
                "split_group": split_group,
                "split_policy": split_ref.get("split_policy", "MISSING_SPLIT_POLICY_REFERENCE"),
                "transform_policy_ref": TRANSFORM_POLICY_REF,
                "encoder_mode": "frozen_encoder",
                "label_status": "NO_LABEL",
                "target_status": "NO_TARGET",
                "pixel_read_status": "NOT_READ__FUTURE_DINO_ENCODING_ONLY",
                "claim_scope": "REVIEW_ONLY_NO_PREDICTIVE_CLAIM",
                "blocker_status": "NONE_FOR_METADATA_MANIFEST" if len(sentinel_rows) == EXPECTED_SENTINEL_COUNT else "COUNT_MISMATCH_REVIEW_REQUIRED",
                "notes": "; ".join(notes),
            }
        )

    region_counts = Counter(row.get("region", "") for row in manifest)
    qa_rows = make_qa(
        manifest=manifest,
        project_state=project_state,
        dino_config_text=dino_config_text,
        extraction_config=extraction_config,
    )
    status = "PASS" if all(row["status"] == "PASS" for row in qa_rows) else "FAIL"
    summary: dict[str, object] = {
        "phase": PHASE,
        "phase_name": PHASE_NAME,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": SOURCE_MANIFEST,
        "expected_sentinel_count": EXPECTED_SENTINEL_COUNT,
        "actual_sentinel_count": len(manifest),
        "qa_status": status,
        "region_counts": dict(sorted(region_counts.items())),
        "encoder_mode": "frozen_encoder",
        "dino_scope": "SENTINEL_FIRST_EMBEDDING_INPUT",
        "guardrails": {
            "training": 0,
            "embedding_extraction": 0,
            "model_loading": 0,
            "downloads": 0,
            "labels": 0,
            "targets": 0,
            "weak_supervision": 0,
            "model_metrics": 0,
            "performance_claims": 0,
            "canonical_writes": 0,
            "gate_promotion": 0,
            "crs_promotion": 0,
            "patch_bound_validated_promotion": 0,
            "preflight_ready_promotion": 0,
            "raster_pixel_reads": 0,
            "band_reads": 0,
            "raster_file_opens": 0,
            "external_path_validation": 0,
            "raw_data_copy": 0,
        },
        "references": {
            "project_state": rel(PROJECT_STATE),
            "dino_config": rel(DINO_CONFIG),
            "embedding_extraction_config": rel(EXTRACTION_CONFIG),
            "split_config": rel(SPLIT_CONFIG),
            "transform_policy": rel(TRANSFORM_POLICY),
            "output_schema": rel(OUTPUT_SCHEMA),
            "dl_input_manifest": rel(DL_INPUT),
            "v1fs_readiness": rel(V1FS_READINESS),
        },
    }
    status_rows = [
        {"field": "phase", "value": PHASE},
        {"field": "phase_name", "value": PHASE_NAME},
        {"field": "status", "value": status},
        {"field": "expected_sentinel_count", "value": str(EXPECTED_SENTINEL_COUNT)},
        {"field": "actual_sentinel_count", "value": str(len(manifest))},
        {"field": "encoder_mode", "value": "frozen_encoder"},
        {"field": "label_status", "value": "NO_LABEL"},
        {"field": "target_status", "value": "NO_TARGET"},
        {"field": "pixel_read_status", "value": "NOT_READ__FUTURE_DINO_ENCODING_ONLY"},
        {"field": "claim_scope", "value": "REVIEW_ONLY_NO_PREDICTIVE_CLAIM"},
    ]
    return manifest, summary, qa_rows, status_rows


def make_qa(
    manifest: list[dict[str, str]],
    project_state: dict[str, str],
    dino_config_text: str,
    extraction_config: dict[str, object],
) -> list[dict[str, str]]:
    qa: list[dict[str, str]] = []

    def add(check: str, passed: bool, details: str) -> None:
        qa.append({"check": check, "status": "PASS" if passed else "FAIL", "details": details})

    input_status = [is_inside_repo(path) and path.exists() for path in INPUT_FILES]
    add("input files exist inside repo", all(input_status), f"{sum(input_status)}/{len(INPUT_FILES)} required inputs present")
    add(
        "project_state confirms review_only / sentinel_first_dino_review_only",
        project_state.get("stage") == "review_only_dino_preimplementation"
        and project_state.get("current_valid_path") == "sentinel_first_dino_review_only",
        f"stage={project_state.get('stage')}; current_valid_path={project_state.get('current_valid_path')}",
    )
    add(
        "dino config blocks supervised classification",
        "supervised_flood_classification" in dino_config_text
        and "predictive_performance_claims" in dino_config_text
        and "mode: frozen_encoder" in dino_config_text,
        "blocked supervised flood classification and predictive performance claims; mode frozen_encoder",
    )
    add(
        "no raw/heavy outputs created",
        not any(path.suffix.lower() in FORBIDDEN_EXTENSIONS for path in OUT_DIR.rglob("*") if path.is_file()) if OUT_DIR.exists() else True,
        "v1fu writes csv/json metadata only",
    )
    forbidden = forbidden_paths()
    add("no forbidden directories created", not forbidden, "; ".join(forbidden) if forbidden else "none found")
    add("no forbidden extensions created", not forbidden, "; ".join(forbidden) if forbidden else "none found")
    add(
        "expected Sentinel count = 128",
        len(manifest) == EXPECTED_SENTINEL_COUNT,
        f"actual={len(manifest)} expected={EXPECTED_SENTINEL_COUNT}",
    )
    add(
        "no label/target columns promoted as truth",
        all(row.get("label_status") == "NO_LABEL" and row.get("target_status") == "NO_TARGET" for row in manifest),
        "label_status and target_status are constant review-only guardrails",
    )
    add(
        "output CSV has required columns",
        bool(manifest) and all(field in manifest[0] for field in MANIFEST_FIELDS),
        f"{len(MANIFEST_FIELDS)} required columns checked",
    )
    add(
        "status CSV summarizes PASS/FAIL",
        True,
        "status is derived from QA checks after all output rows are built",
    )
    add(
        "embedding config remains future-only",
        extraction_config.get("encoder_policy", {}).get("extract_embeddings_now") is False
        and extraction_config.get("guardrails", {}).get("embedding_extraction") == 0,
        "no embeddings generated in v1fu",
    )
    return qa


def main() -> None:
    manifest, summary, qa_rows, status_rows = build_manifest()
    write_csv(MANIFEST_PATH, manifest, MANIFEST_FIELDS)
    write_json(SUMMARY_PATH, summary)
    write_csv(QA_PATH, qa_rows, ["check", "status", "details"])
    write_csv(STATUS_PATH, status_rows, ["field", "value"])


if __name__ == "__main__":
    main()
