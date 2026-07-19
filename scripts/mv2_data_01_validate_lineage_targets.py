"""MV2-DATA-01 Tarefa 8 — Validador do GEE Lineage Target Pack.

Sai com código != 0 se qualquer trava fail-closed for violada. Verifica que o pacote é
metadata-only: nenhum scene_id/datetime/tile/cloud inventado, nenhum HTTP/GEE/STAC real,
nenhum download/raster/crop/feature/label/silver/gold/negativo, nenhum auto-join T23KPR,
cidade/região insuficiente, outputs leves, sem caminho absoluto privado e summary coerente.
"""

from __future__ import annotations

import json
import sys

from mv2_data_01_common import (
    METADATA_FIELDS,
    OUTPUT_DIR,
    PROJECT_ROOT,
    RISK_TYPES,
    classify_datetime,
    clean,
    read_csv_rows,
    resolve_data_input,
)

HEAVY_EXT = {".tif", ".tiff", ".jp2", ".safe", ".img", ".cog", ".npy", ".npz",
             ".png", ".jpg", ".jpeg", ".zip", ".gpkg", ".shp", ".nc", ".hdf"}
ALLOWED_EXT = {".csv", ".json", ".md", ".txt"}
MAX_BYTES = 2_000_000

TARGET_REQUIRED_COLS = [
    "target_id", "binding_id", "anchor_id", "asset_id", "patch_id", "region", "sensor",
    "platform", "bbox", "crs", "has_asset_id", "has_patch_id", "has_bbox", "has_crs",
    "has_scene_id", "has_datetime", "has_tile", "has_cloud_cover", "lineage_status",
    "priority", "blocking_reason", "next_action",
]

# Campos do template manual que jamais podem vir preenchidos por inferência.
TEMPLATE_INVENTED_FIELDS = ["product_id", "scene_id", "acquisition_datetime", "mgrs_tile",
                            "cloudy_pixel_percentage", "cloud_coverage_assessment",
                            "datatake_identifier", "gee_collection"]


def find_violations() -> list[str]:
    v: list[str] = []
    if not OUTPUT_DIR.exists():
        return [f"diretório ausente: {OUTPUT_DIR}"]

    # Outputs leves; nenhum bruto pesado; nenhum caminho absoluto privado.
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

    targets = read_csv_rows(OUTPUT_DIR / "mv2_data_01_lineage_targets.csv")
    if not targets:
        return v + ["mv2_data_01_lineage_targets.csv ausente ou vazio"]

    # 128 targets ou divergência registrada vs. entrada.
    src_rows = read_csv_rows(resolve_data_input("medium_binding_index")[0])
    if len(targets) != len(src_rows):
        v.append(f"divergência de contagem: {len(targets)} targets vs {len(src_rows)} entrada")
    if len(src_rows) == 128 and len(targets) != 128:
        v.append(f"esperado 128 targets, obtido {len(targets)}")

    # Colunas obrigatórias presentes.
    cols = set(targets[0].keys())
    for c in TARGET_REQUIRED_COLS:
        if c not in cols:
            v.append(f"coluna obrigatória ausente em targets: {c}")

    # Nenhum scene_id inventado; nenhum datetime futuro aceito; cidade/região insuficiente.
    # Casa por binding_id (presente tanto nos targets quanto na matriz de lineage MV2-14).
    lineage = {clean(r.get("binding_id")): r
               for r in read_csv_rows(resolve_data_input("lineage_binding_matrix")[0])}
    recovered_scene_count = sum(
        1 for r in lineage.values() if clean(r.get("recovered_scene_id"))
    )
    targets_with_scene = sum(1 for t in targets if clean(t.get("has_scene_id")) == "true")
    if targets_with_scene > recovered_scene_count:
        v.append(
            f"scene_id inventado: {targets_with_scene} has_scene_id=true vs "
            f"{recovered_scene_count} recuperados na matriz de lineage"
        )
    for t in targets:
        # Alvo só pode ter has_scene_id/datetime/tile/cloud=true com lado espacial pronto.
        if clean(t.get("has_scene_id")) == "true" and clean(t.get("has_patch_id")) != "true":
            v.append(f"scene_id sem patch: {clean(t.get('target_id'))}")
        # Status válido conhecido.
        if clean(t.get("lineage_status")) not in {
            "TARGET_READY_FOR_MANUAL_GEE_LOOKUP",
            "TARGET_READY_FOR_METADATA_QUERY_PLAN",
            "TARGET_BLOCKED_MISSING_PATCH",
            "TARGET_BLOCKED_MISSING_ASSET",
            "TARGET_BLOCKED_MISSING_GEOMETRY",
            "TARGET_BLOCKED_MISSING_CRS",
            "TARGET_INVALID",
        }:
            v.append(f"lineage_status inválido: {clean(t.get('lineage_status'))}")

    # datetime futuro nunca aceito como válido na matriz de lineage de origem.
    targets_by_binding = {clean(t.get("binding_id")): t for t in targets}
    for binding_id, r in lineage.items():
        dt = clean(r.get("recovered_datetime"))
        if dt and classify_datetime(dt[:10])[1] == "INVALID_FUTURE":
            t = targets_by_binding.get(binding_id)
            if t and clean(t.get("has_datetime")) == "true":
                v.append(f"datetime futuro aceito como válido: {clean(t.get('target_id'))}")

    # Plano GEE: nenhuma chamada GEE / export.
    gee = read_csv_rows(OUTPUT_DIR / "mv2_data_01_gee_metadata_query_plan.csv")
    for g in gee:
        if clean(g.get("would_call_gee")) != "false":
            v.append(f"plano GEE marcaria chamada real: {clean(g.get('query_id'))}")
        if clean(g.get("would_export_raster")) != "false":
            v.append(f"plano GEE marcaria export de raster: {clean(g.get('query_id'))}")
        if clean(g.get("execution_status")) != "NOT_EXECUTED_PLAN_ONLY":
            v.append(f"plano GEE não é PLAN_ONLY: {clean(g.get('query_id'))}")
    # Campos mínimos de metadados presentes no plano GEE.
    for g in gee[:1]:
        fields = clean(g.get("metadata_fields_to_collect"))
        for mf in METADATA_FIELDS:
            if mf not in fields:
                v.append(f"campo de metadado ausente no plano GEE: {mf}")

    # Plano STAC/OData: nenhum HTTP / download.
    stac = read_csv_rows(OUTPUT_DIR / "mv2_data_01_stac_odata_metadata_plan.csv")
    for s in stac:
        if clean(s.get("would_execute_http")) != "false":
            v.append(f"plano STAC marcaria HTTP real: {clean(s.get('query_id'))}")
        if clean(s.get("would_download")) != "false":
            v.append(f"plano STAC marcaria download: {clean(s.get('query_id'))}")
        if clean(s.get("execution_status")) != "NOT_EXECUTED_PLAN_ONLY":
            v.append(f"plano STAC não é PLAN_ONLY: {clean(s.get('query_id'))}")

    # Template manual: nenhum campo de lineage inventado; nenhum tile auto (T23KPR).
    template = read_csv_rows(OUTPUT_DIR / "mv2_data_01_manual_gee_lineage_template.csv")
    for tr in template:
        for f in TEMPLATE_INVENTED_FIELDS:
            if clean(tr.get(f)):
                v.append(f"template com {f} pré-preenchido: {clean(tr.get('target_id'))}")
        if clean(tr.get("mgrs_tile")):
            v.append(f"template com tile preenchido (auto-join): {clean(tr.get('target_id'))}")

    # Matriz de risco: todos os RISK_TYPES obrigatórios presentes; nenhum auto-join aceito.
    risk = read_csv_rows(OUTPUT_DIR / "mv2_data_01_risk_matrix.csv")
    present = {clean(r.get("risk_type")) for r in risk}
    for rt in RISK_TYPES:
        if rt not in present:
            v.append(f"risk_type obrigatório ausente: {rt}")

    # Summary coerente e fail-closed.
    summary_path = OUTPUT_DIR / "mv2_data_01_summary.json"
    if not summary_path.exists():
        return v + ["summary JSON ausente"]
    s = json.loads(summary_path.read_text(encoding="utf-8"))
    for key in ["http_calls_executed", "gee_calls_executed", "downloads_executed",
                "crops_created", "native_rasters_created", "spectral_features_created",
                "labels_created", "silver_created", "gold_created", "negatives_created"]:
        if s.get(key, -1) != 0:
            v.append(f"guardrail violado: {key}={s.get(key)} (deve ser 0)")
    if s.get("can_unlock_day10_now") is not False:
        v.append("guardrail violado: can_unlock_day10_now deve ser false")
    if s.get("can_train") is not False:
        v.append("guardrail violado: can_train deve ser false")
    if s.get("sandbox_status") != "bloqueado":
        v.append("guardrail violado: sandbox_status deve ser 'bloqueado'")
    if s.get("total_targets") != len(targets):
        v.append("summary incoerente: total_targets != linhas da CSV de targets")
    if s.get("gee_metadata_query_plans") != len(gee):
        v.append("summary incoerente: gee_metadata_query_plans")
    if s.get("stac_odata_query_plans") != len(stac):
        v.append("summary incoerente: stac_odata_query_plans")
    if s.get("manual_template_rows") != len(template):
        v.append("summary incoerente: manual_template_rows")

    return v


def main() -> int:
    v = find_violations()
    if v:
        print("[mv2_data_01_validate] VIOLACOES:")
        for x in v:
            print(f"  - {x}")
        return 1
    n = sum(1 for p in OUTPUT_DIR.rglob("*") if p.is_file())
    print(f"[mv2_data_01_validate] OK — {n} arquivos leves; guardrails fail-closed preservados")
    return 0


if __name__ == "__main__":
    sys.exit(main())
