"""MV2-14 Tarefa 13 — Validador de guardrails do GEE Lineage Recovery.

Sai com código != 0 se qualquer trava fail-closed for violada.
"""

from __future__ import annotations

import json
import sys

from mv2_14_gee_lineage_common import (
    OUTPUT_DIR,
    PROJECT_ROOT,
    classify_datetime,
    clean,
    read_csv_rows,
)

HEAVY_EXT = {".tif", ".tiff", ".jp2", ".safe", ".img", ".cog", ".npy", ".npz",
             ".png", ".jpg", ".jpeg", ".zip", ".gpkg", ".shp", ".nc", ".hdf"}
ALLOWED_EXT = {".csv", ".json", ".md", ".txt"}
MAX_BYTES = 2_000_000
TEMPLATE_FILL_FIELDS = ["scene_id", "acquisition_datetime", "tile", "cloud_cover",
                        "gee_collection", "gee_export_task", "gee_script_path_or_name"]


def find_violations() -> list[str]:
    v: list[str] = []
    if not OUTPUT_DIR.exists():
        return [f"diretório ausente: {OUTPUT_DIR}"]

    # 13-14. Só leve; nenhum bruto pesado; nenhum caminho absoluto privado.
    for p in sorted(OUTPUT_DIR.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(PROJECT_ROOT).as_posix()
        ext = p.suffix.lower()
        if ext in HEAVY_EXT:
            v.append(f"bruto pesado em outputs_public: {rel}")
        elif ext not in ALLOWED_EXT:
            v.append(f"extensão não permitida: {rel} ({ext})")
        if p.stat().st_size > MAX_BYTES:
            v.append(f"arquivo acima de {MAX_BYTES} bytes: {rel}")
        if ext in ALLOWED_EXT:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            if "C:/Users" in txt or "C:\\Users" in txt:
                v.append(f"caminho absoluto privado em {p.name}")

    summary_path = OUTPUT_DIR / "mv2_14_lineage_summary.json"
    if not summary_path.exists():
        return v + ["summary JSON ausente"]
    s = json.loads(summary_path.read_text(encoding="utf-8"))

    # 1-7. Contadores fail-closed.
    for key in ["stac_real_executed", "downloads_executed", "crops_created",
                "native_rasters_created", "spectral_features_created",
                "labels_created", "silver_created", "gold_created", "negatives_created",
                "auto_joins_performed"]:
        if s.get(key, 0) != 0:
            v.append(f"guardrail violado: {key}={s.get(key)} (deve ser 0)")
    if s.get("can_train") is not False:
        v.append("guardrail violado: can_train deve ser false")

    # 8. Dia 10 não desbloqueado sem raster nativo validado.
    day10 = read_csv_rows(OUTPUT_DIR / "mv2_14_day10_post_lineage_gate.csv")
    for d in day10:
        if clean(d.get("can_unlock_day10_now")) == "true" and clean(d.get("has_native_raster")) != "true":
            v.append(f"day10 desbloqueado sem raster nativo: {clean(d.get('day10_post_lineage_id'))}")

    # 9. Nenhum datetime futuro aceito como válido na matriz de lineage.
    lineage = read_csv_rows(OUTPUT_DIR / "mv2_14_lineage_binding_matrix.csv")
    for lr in lineage:
        dt = clean(lr.get("recovered_datetime"))
        if dt and classify_datetime(dt[:10])[1] == "INVALID_FUTURE":
            if clean(lr.get("lineage_status")) not in {"LINEAGE_INVALID"}:
                v.append(f"datetime futuro aceito: {clean(lr.get('lineage_binding_id'))}")

    # 10. Nenhuma cena T23KPR auto-vinculada por tile/zona.
    scene_review = read_csv_rows(OUTPUT_DIR / "mv2_14_strong_scene_anchor_review.csv")
    for r in scene_review:
        if clean(r.get("auto_join_allowed")) == "true":
            v.append(f"cena auto-vinculada por tile: {clean(r.get('scene_anchor_id'))}")

    # 11. CANDIDATE_REVIEW (vínculo por região/zona) nunca recupera scene_id automaticamente.
    for lr in lineage:
        if clean(lr.get("lineage_status")) == "LINEAGE_CANDIDATE_REVIEW" and clean(lr.get("recovered_scene_id")):
            v.append(f"region/zone proxy recuperou scene_id: {clean(lr.get('lineage_binding_id'))}")
        # STAC real review só com lineage forte.
        if clean(lr.get("stac_readiness_after_lineage")) == "READY_FOR_STAC_REAL_REVIEW_REQUIRED":
            if clean(lr.get("lineage_status")) != "LINEAGE_RECOVERED_STRONG":
                v.append(f"STAC real review sem lineage forte: {clean(lr.get('lineage_binding_id'))}")

    # 15. Template manual não vem com lineage inventado.
    template = read_csv_rows(OUTPUT_DIR / "mv2_14_manual_lineage_template.csv")
    for t in template:
        for f in TEMPLATE_FILL_FIELDS:
            if clean(t.get(f)):
                v.append(f"template com lineage pré-preenchido em {f}: {clean(t.get('patch_id'))}")
                break

    # 16. Summary coerente com CSVs.
    if s.get("lineage_not_found", -1) + s.get("lineage_candidate_review", -1) + \
       s.get("lineage_recovered_strong", 0) + s.get("lineage_recovered_partial", 0) + \
       s.get("lineage_conflict", 0) + s.get("lineage_invalid", 0) != len(lineage):
        v.append("summary incoerente com a matriz de lineage")

    return v


def main() -> int:
    v = find_violations()
    if v:
        print("[mv2_14_validate] VIOLACOES:")
        for x in v:
            print(f"  - {x}")
        return 1
    n = sum(1 for p in OUTPUT_DIR.rglob("*") if p.is_file())
    print(f"[mv2_14_validate] OK — {n} arquivos leves; guardrails fail-closed preservados")
    return 0


if __name__ == "__main__":
    sys.exit(main())
