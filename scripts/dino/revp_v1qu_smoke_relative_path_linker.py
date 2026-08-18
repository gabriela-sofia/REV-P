"""REV-P v1qu — Smoke sample relative-path linker (metadata-only, no pixel reads).

Problem addressed: `dino_smoke_sample_selection_v1qh.csv` selects 32 real patches
(quota + diversity across CUR/PET/REC) but its `relative_path` column was left
empty at generation time, so every row in v1qi (asset audit) fails closed with
`empty_relative_path`. This script resolves, for each smoke row, the expected
local Sentinel filename under the naming convention already used in
`PROJETO/data/sentinel/` (`patch_<region>_<patch_number>.tif`) and writes a NEW
linked artifact — it never rewrites v1qh in place (historical outputs are never
modified; see CLAUDE.md forbidden operations).

Boundary: this script only calls `Path.exists()` (a filesystem stat) to report
`expected_file_found`. It never opens a file, never reads a pixel, and never
sets `pixel_read_allowed`/`pixel_read_performed` — those remain the exclusive
responsibility of v1qi, gated by REVP_DINO_PIXEL_READ_ALLOWED.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from revp_v1qg_v1qm_smoke_embedding_common import (
    DATASETS, DOCS, SCHEMAS,
    _p, assert_no_forbidden_true, normalize_region, path_hash, read_csv,
    require_no_abs_paths, write_csv, write_doc, write_schema,
)

IN_SEL = _p("REVP_V1QU_IN_SEL", DATASETS / "dino_smoke_sample_selection_v1qh.csv")
OUT_LINKED = _p("REVP_V1QU_OUT_LINKED", DATASETS / "dino_smoke_sample_selection_linked_v1qu.csv")
OUT_SUM = _p("REVP_V1QU_OUT_SUM", DATASETS / "dino_smoke_relative_path_linker_summary_v1qu.csv")
SCH_LINKED = _p("REVP_V1QU_SCH_LINKED", SCHEMAS / "dino_smoke_sample_selection_linked_v1qu_schema.csv")
SCH_SUM = _p("REVP_V1QU_SCH_SUM", SCHEMAS / "dino_smoke_relative_path_linker_summary_v1qu_schema.csv")
DOC = _p("REVP_V1QU_DOC", DOCS / "revp_v1qu_smoke_relative_path_linker.md")

# Region code (as stored in v1qh) -> folder-naming token used under
# REV-P/patches/thumbnails/<token>/<token>_<numero>__rgb_preview.png.
# These PNGs are the project's own established B4/B3/B2 natural-composite render
# (percentile stretch + gray-world + gamma; see
# local_runs/treino_exploratorio_diagnostico_v5_chuva_real/generate_patch_rgb_previews.py
# in PROJETO), reused here so v1qj's PIL-based loader can read a 3-channel 8-bit
# image instead of the raw 6-band float64 BigTIFF (which PIL cannot open at all).
_REGION_TO_FILENAME_TOKEN = {
    "CURITIBA": "curitiba",
    "PET": "petropolis",
    "RECIFE": "recife",
}

LINKED_FIELDS = [
    "smoke_id", "execution_queue_id", "visual_asset_id", "patch_id", "alias",
    "region", "relative_path", "path_hash", "visual_type", "sample_rank",
    "selection_reason", "linkage_confidence", "dino_allowed_use",
    "can_create_label", "can_train_model", "target_created", "selected_for_smoke",
    "expected_filename", "expected_file_found", "link_status", "blocked_reason", "notes",
]
SUM_FIELDS = ["stat_key", "stat_value"]


def _numeric_suffix(patch_id: str) -> str:
    """Extract the trailing digit run from a patch_id like CUR_00038 -> 00038."""
    digits = "".join(ch for ch in patch_id if ch.isdigit())
    return digits[-5:] if len(digits) >= 5 else digits.zfill(5)


def _expected_filename(patch_id: str, region: str) -> str:
    token = _REGION_TO_FILENAME_TOKEN.get(region, "")
    if not token:
        return ""
    return f"{token}/{token}_{_numeric_suffix(patch_id)}__rgb_preview.png"


def link_rows(rows: list[dict[str, str]], sentinel_root: Path | None) -> tuple[list[dict[str, Any]], dict[str, int]]:
    out: list[dict[str, Any]] = []
    counts = {"linked": 0, "unmapped_region": 0, "file_not_found": 0}

    for r in rows:
        patch = (r.get("patch_id", "") or "").upper()
        region = normalize_region(r.get("region", ""))
        expected = _expected_filename(patch, region)
        row: dict[str, Any] = dict(r)
        row["expected_filename"] = expected
        row["expected_file_found"] = "false"
        row["blocked_reason"] = ""
        row["notes"] = ""

        if not expected:
            row["relative_path"] = ""
            row["path_hash"] = r.get("path_hash", "")
            row["link_status"] = "REGION_NOT_MAPPED_FAIL_CLOSED"
            row["blocked_reason"] = "region_not_in_filename_map"
            counts["unmapped_region"] += 1
            out.append(row)
            continue

        found = False
        if sentinel_root is not None:
            try:
                found = (sentinel_root / expected).exists() and (sentinel_root / expected).is_file()
            except Exception:
                found = False
        row["expected_file_found"] = str(found).lower()

        if not found:
            row["relative_path"] = ""
            row["path_hash"] = r.get("path_hash", "")
            row["link_status"] = "EXPECTED_FILE_NOT_FOUND_FAIL_CLOSED"
            row["blocked_reason"] = "expected_filename_not_on_disk"
            counts["file_not_found"] += 1
            out.append(row)
            continue

        row["relative_path"] = expected
        row["path_hash"] = path_hash(expected)
        row["link_status"] = "LINKED_RELATIVE_PATH_READY"
        counts["linked"] += 1
        out.append(row)

    return out, counts


def run() -> None:
    rows = read_csv(IN_SEL)
    root_env = _p("REVP_SENTINEL_LOCAL_ROOT", Path(""))
    sentinel_root = root_env if str(root_env) and root_env.exists() else None

    linked, counts = link_rows(rows, sentinel_root)
    require_no_abs_paths(linked, "v1qu_linked")
    assert_no_forbidden_true(linked, "v1qu_linked")

    summary = [
        {"stat_key": "smoke_rows", "stat_value": str(len(rows))},
        {"stat_key": "sentinel_root_configured", "stat_value": str(sentinel_root is not None).lower()},
        {"stat_key": "linked_relative_path_ready", "stat_value": str(counts["linked"])},
        {"stat_key": "unmapped_region", "stat_value": str(counts["unmapped_region"])},
        {"stat_key": "expected_file_not_found", "stat_value": str(counts["file_not_found"])},
        {"stat_key": "pixel_reads_performed", "stat_value": "0"},
        {"stat_key": "final_status",
         "stat_value": "LINKING_READY_FOR_V1QI" if counts["linked"] > 0 else "LINKING_FAIL_CLOSED_NO_MATCHES"},
    ]
    require_no_abs_paths(summary, "v1qu_summary")

    write_csv(OUT_LINKED, linked, LINKED_FIELDS)
    write_csv(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCH_LINKED, LINKED_FIELDS, "v1qu_smoke_sample_selection_linked")
    write_schema(SCH_SUM, SUM_FIELDS, "v1qu_smoke_relative_path_linker_summary")
    write_doc(DOC, "v1qu — Smoke Sample Relative-Path Linker", [
        "## Objetivo",
        "Resolver o `relative_path` esperado (convenção `patch_<regiao>_<numero>.tif`) "
        "para as 32 linhas de v1qh, sem reescrever o artefato histórico v1qh e sem "
        "ler nenhum pixel.",
        "## Limite metodológico",
        "Apenas `Path.exists()` (stat de arquivo) é usado para reportar "
        "`expected_file_found`. Nenhum arquivo é aberto; nenhum pixel é lido. "
        "`pixel_read_allowed`/`pixel_read_performed` continuam sendo responsabilidade "
        "exclusiva de v1qi, controlada por REVP_DINO_PIXEL_READ_ALLOWED.",
        "## Resultado",
        f"Linhas: {len(rows)}. Linkadas: {counts['linked']}. "
        f"Região não mapeada: {counts['unmapped_region']}. "
        f"Arquivo esperado ausente: {counts['file_not_found']}.",
    ])
    print(f"[v1qu] rows={len(rows)} linked={counts['linked']} "
          f"unmapped={counts['unmapped_region']} not_found={counts['file_not_found']}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1qu smoke relative-path linker").parse_args()
    run()
