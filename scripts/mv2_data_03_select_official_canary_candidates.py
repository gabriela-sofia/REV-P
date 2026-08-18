"""MV2-DATA-03 Tarefa 2 — Selecionar canary candidates oficiais.

Lê o registry oficial e seleciona no máximo 2 âncoras com lineage Sentinel completo,
confirmando que são âncoras oficiais (namespace REFPATCH_*/Protocolo-C) e NÃO patch do
corpus (PET_*/CUR_*/REC_*). Marca uso como TECHNICAL_CANARY_ONLY e
can_use_for_corpus_day10=false.

Saída: outputs_public/mv2_data_live_api_canary/mv2_data_03_official_canary_candidates.csv
"""

from __future__ import annotations

import re
from pathlib import Path

from mv2_data_03_common import (
    MAX_CANARY_CANDIDATES,
    PROJECT_ROOT,
    PUBLIC_DIR,
    clean,
    corpus_patch_ids,
    ensure_public_dir,
    load_official_strong_anchors,
    write_csv,
)

TILE_RE = re.compile(r"T\d{2}[A-Z]{3}")

COLUMNS = [
    "canary_candidate_id",
    "source_registry",
    "official_anchor_id",
    "official_refpatch_id",
    "corpus_patch_id",
    "corpus_asset_id",
    "region",
    "scene_id",
    "product_id",
    "scene_date",
    "gee_collection",
    "mgrs_tile",
    "crs",
    "bands",
    "cloud_cover",
    "canary_role",
    "can_use_for_corpus_day10",
    "blocking_reason",
    "notes",
]


def build_rows() -> list[dict[str, object]]:
    corpus = corpus_patch_ids()
    rows: list[dict[str, object]] = []
    for i, a in enumerate(load_official_strong_anchors(), start=1):
        if len(rows) >= MAX_CANARY_CANDIDATES:
            break
        scene = clean(a.get("scene_id_sanitized"))
        refpatch = clean(a.get("reference_patch_id"))
        # Confirma que é âncora oficial e NÃO um patch do corpus.
        is_corpus = refpatch in corpus or bool(re.match(r"^(PET|CUR|REC)_\d", refpatch))
        tile_match = TILE_RE.search(scene)
        tile = tile_match.group(0) if tile_match else ""
        rows.append(
            {
                "canary_candidate_id": f"MV2_DATA_03_CC_{i:02d}",
                "source_registry": "datasets/official_anchor_sentinel_patch_registry.csv",
                "official_anchor_id": clean(a.get("anchor_id")),
                "official_refpatch_id": refpatch,
                "corpus_patch_id": "",  # canary oficial NÃO é patch do corpus
                "corpus_asset_id": "",
                "region": clean(a.get("region")),
                "scene_id": scene,
                "product_id": scene,
                "scene_date": clean(a.get("scene_date")),
                "gee_collection": clean(a.get("gee_collection")),
                "mgrs_tile": tile,  # extraído do scene_id documentado, não auto-join por zona
                "crs": clean(a.get("crs")),
                "bands": clean(a.get("bands_available")),
                "cloud_cover": clean(a.get("cloud_cover_metadata")),
                "canary_role": "TECHNICAL_CANARY_ONLY",
                "can_use_for_corpus_day10": "false",
                "blocking_reason": (
                    "namespace_corpus_detectado" if is_corpus else "official_anchor_protocolo_c_nao_corpus"
                ),
                "notes": "lineage documentado; tile vem do scene_id oficial, nunca de zona/cidade",
            }
        )
    return rows


def main() -> None:
    ensure_public_dir()
    rows = build_rows()
    out = PUBLIC_DIR / "mv2_data_03_official_canary_candidates.csv"
    write_csv(out, COLUMNS, rows)
    print(
        f"[mv2_data_03_candidates] {len(rows)} candidates oficiais (TECHNICAL_CANARY_ONLY) | "
        f"saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
