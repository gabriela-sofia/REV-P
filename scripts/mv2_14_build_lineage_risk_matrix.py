"""MV2-14 Tarefa 11 — Matriz de risco do lineage GEE.

Saída: outputs_public/mv2_gee_lineage_recovery/mv2_14_lineage_risk_matrix.csv
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from mv2_14_gee_lineage_common import (
    OUTPUT_DIR,
    PROJECT_ROOT,
    RISK_TYPES,
    clean,
    ensure_output_dir,
    read_csv_rows,
    write_csv,
)

COLUMNS = [
    "risk_id", "related_asset_id", "related_patch_id", "related_scene_id",
    "severity", "risk_type", "evidence", "mitigation", "blocks_next_step",
]

RISK_SPEC = {
    "RISK_SCENE_WITHOUT_PATCH": ("ALTA", "cenas com scene_id sem patch/asset", "10 cenas T23KPR sem patch do corpus; review_required, auto_join=false", "true"),
    "RISK_PATCH_WITHOUT_SCENE": ("ALTA", "patches com geometria sem scene_id", "128 bindings espaciais sem cena recuperada", "true"),
    "RISK_TILE_AUTO_JOIN": ("CRITICA", "unir cena a patch por tile/zona", "Petropolis em CANDIDATE_REVIEW só por zona; auto_joins_performed=0", "true"),
    "RISK_FILENAME_INFERENCE": ("MEDIA", "inferir scene_id de filename fraco", "candidatos textuais marcados confidence=LOW; nunca viram verdade", "true"),
    "RISK_FUTURE_DATETIME": ("ALTA", "datetime futuro aceito como válido", "datas 2026-* descartadas como INVALID_FUTURE", "true"),
    "RISK_CITY_REGION_PROXY": ("ALTA", "cidade/região como vínculo", "região nunca estabelece lineage; no máximo CANDIDATE_REVIEW", "true"),
    "RISK_CLOUD_COVER_MISSING": ("MEDIA", "cloud_cover ausente", "cloud_cover ausente nos 128 bindings do corpus", "true"),
    "RISK_EXPORT_PROVENANCE_MISSING": ("ALTA", "sem proveniência de export GEE", "0 export task vinculado a patch do corpus", "true"),
    "RISK_STAC_BEFORE_LINEAGE": ("CRITICA", "STAC antes de fechar lineage", "can_stac_dry_run=0; can_stac_real=0", "true"),
    "RISK_DAY10_FALSE_UNLOCK": ("CRITICA", "Dia 10 desbloqueado sem raster", "can_unlock_day10_now=false em todos", "true"),
    "RISK_RASTER_DOWNLOAD_WITHOUT_BINDING": ("CRITICA", "baixar raster sem binding forte", "0 binding forte; download bloqueado", "true"),
    "RISK_PNG_AS_RASTER": ("ALTA", "PNG/NPZ/render como raster nativo", "has_native_raster=false sempre", "true"),
    "RISK_UNKNOWN_AS_NEGATIVE": ("CRITICA", "ausência como negativo", "80 NOT_FOUND nunca viram negativo", "true"),
}


def build_rows() -> list[dict[str, object]]:
    lineage = read_csv_rows(OUTPUT_DIR / "mv2_14_lineage_binding_matrix.csv")
    dist = Counter(clean(r.get("lineage_status")) for r in lineage)
    rows: list[dict[str, object]] = []
    for i, risk in enumerate(RISK_TYPES, start=1):
        sev, ev, mit, blocks = RISK_SPEC[risk]
        # Enriquece evidência com contagem real onde aplicável.
        if risk == "RISK_PATCH_WITHOUT_SCENE":
            ev = f"{len(lineage)} bindings espaciais sem cena recuperada"
        if risk == "RISK_CITY_REGION_PROXY":
            ev = f"{dist.get('LINEAGE_CANDIDATE_REVIEW', 0)} em CANDIDATE_REVIEW (hipótese região+zona, nunca vínculo)"
        if risk == "RISK_UNKNOWN_AS_NEGATIVE":
            ev = f"{dist.get('LINEAGE_NOT_FOUND', 0)} NOT_FOUND nunca viram negativo"
        rows.append({
            "risk_id": f"MV2_14_RISK_{i:02d}",
            "related_asset_id": "ALL",
            "related_patch_id": "ALL",
            "related_scene_id": "ALL",
            "severity": sev,
            "risk_type": risk,
            "evidence": ev,
            "mitigation": mit,
            "blocks_next_step": blocks,
        })
    return rows


def main() -> None:
    ensure_output_dir()
    rows = build_rows()
    out = OUTPUT_DIR / "mv2_14_lineage_risk_matrix.csv"
    write_csv(out, COLUMNS, rows)
    print(f"[mv2_14_risk] riscos: {len(rows)}")
    print(f"[mv2_14_risk] saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
