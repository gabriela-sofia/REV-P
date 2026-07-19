"""MV2-14 Tarefa 4 — Reconcilia candidatos de lineage com os 128 bindings MEDIUM.

Matching SÓ por evidência explícita (asset_id/patch_id/export task/scene). Cidade/região
ou tile sozinhos NUNCA estabelecem vínculo: no máximo CANDIDATE_REVIEW. Datetime futuro
é inválido. Conflito quando candidatos incompatíveis aparecem para o mesmo alvo.

Saída: outputs_public/mv2_gee_lineage_recovery/mv2_14_lineage_binding_matrix.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_14_gee_lineage_common import (
    OUTPUT_DIR,
    PROJECT_ROOT,
    REGION_ZONE,
    classify_datetime,
    clean,
    ensure_output_dir,
    read_csv_rows,
    write_csv,
)

COLUMNS = [
    "lineage_binding_id", "target_id", "binding_id", "anchor_id", "asset_id",
    "patch_id", "region", "recovered_scene_id", "recovered_datetime",
    "recovered_tile", "recovered_cloud_cover", "recovered_sensor",
    "export_task_or_source", "evidence_source", "evidence_snippet", "lineage_status",
    "confidence", "conflict_status", "invalid_reason", "stac_readiness_after_lineage",
    "day10_status_after_lineage", "blocking_reason", "next_action",
]


def _zone_of_crs(crs: str) -> str:
    # EPSG:327NN -> zona UTM NN (índices 8:10) -> prefixo de tile MGRS "T<NN>"
    crs = clean(crs)
    if crs.startswith("EPSG:327") and len(crs) >= 10:
        return f"T{crs[8:10]}"
    return ""


def build_rows() -> list[dict[str, object]]:
    targets = read_csv_rows(OUTPUT_DIR / "mv2_14_medium_binding_index.csv")
    candidates = read_csv_rows(OUTPUT_DIR / "mv2_14_lineage_candidates.csv")

    # Índices de candidatos por vínculo explícito.
    cand_by_asset: dict[str, list[dict]] = {}
    cand_by_patch: dict[str, list[dict]] = {}
    cand_zone_present: set[str] = set()
    for c in candidates:
        a = clean(c.get("inferred_asset_id"))
        p = clean(c.get("inferred_patch_id"))
        tile = clean(c.get("inferred_tile"))
        if a:
            cand_by_asset.setdefault(a, []).append(c)
        if p:
            cand_by_patch.setdefault(p, []).append(c)
        if tile:
            cand_zone_present.add(tile[:3])  # ex.: "T23"

    rows: list[dict[str, object]] = []
    for i, t in enumerate(targets, start=1):
        asset_id = clean(t.get("asset_id"))
        patch_id = clean(t.get("patch_id"))
        region = clean(t.get("region"))
        crs = clean(t.get("crs"))
        zone = _zone_of_crs(crs)

        # Match explícito por asset_id ou patch_id do corpus.
        explicit = cand_by_asset.get(asset_id, []) + cand_by_patch.get(patch_id, [])

        recovered = {"scene": "", "dt": "", "tile": "", "cloud": "", "sensor": "", "src": "", "ev": ""}
        status = "LINEAGE_NOT_FOUND"
        confidence = "NONE"
        conflict = "NONE"
        invalid_reason = ""

        if explicit:
            scenes = {clean(c.get("normalized_value")) for c in explicit if clean(c.get("normalized_value"))}
            future = [c for c in explicit if classify_datetime(clean(c.get("inferred_datetime"))[:10])[1] == "INVALID_FUTURE"]
            if len(scenes) > 1:
                status, conflict, confidence = "LINEAGE_CONFLICT", "CONFLICT", "LOW"
            elif future:
                status, invalid_reason, confidence = "LINEAGE_INVALID", "future_datetime_in_candidate", "LOW"
            else:
                c0 = explicit[0]
                recovered.update({
                    "scene": clean(c0.get("normalized_value")),
                    "dt": clean(c0.get("inferred_datetime")),
                    "tile": clean(c0.get("inferred_tile")),
                    "cloud": clean(c0.get("inferred_cloud_cover")),
                    "sensor": clean(c0.get("inferred_sensor")),
                    "src": clean(c0.get("source_path_or_ref")),
                    "ev": clean(c0.get("evidence_snippet")),
                })
                has_temporal = recovered["dt"] and classify_datetime(recovered["dt"][:10])[1] == "VALID_SINGLE"
                if recovered["scene"] and has_temporal:
                    status, confidence = "LINEAGE_RECOVERED_STRONG", "HIGH"
                else:
                    status, confidence = "LINEAGE_RECOVERED_PARTIAL", "MEDIUM"
        elif region in REGION_ZONE and zone and zone in cand_zone_present:
            # Existem cenas candidatas locais para a MESMA zona/região — porém o
            # vínculo por região+zona é HIPÓTESE: exige revisão humana por patch.
            # Nada é recuperado automaticamente (recovered_scene_id fica vazio).
            status, confidence = "LINEAGE_CANDIDATE_REVIEW", "LOW"
            recovered["ev"] = f"cenas_candidatas_locais_zona_{zone}_regiao_{region}_HIPOTESE_revisao_humana"
        else:
            status = "LINEAGE_NOT_FOUND"

        # Prontidão STAC pós-lineage.
        if status == "LINEAGE_RECOVERED_STRONG":
            stac_ready = "READY_FOR_STAC_REAL_REVIEW_REQUIRED"
            block = "lineage_forte;requer_revisao_antes_de_STAC_real"
            action = "revisar_e_autorizar_STAC_real_em_etapa_futura_sem_executar_agora"
        elif status == "LINEAGE_RECOVERED_PARTIAL":
            stac_ready = "BLOCKED_NO_DATETIME" if not recovered["dt"] else "BLOCKED_NO_CLOUD_COVER"
            block = "lineage_parcial;faltam_campos_temporais"
            action = "recuperar_campos_temporais_faltantes"
        elif status == "LINEAGE_CANDIDATE_REVIEW":
            stac_ready = "BLOCKED_NO_SCENE_ID"
            block = "cena_nao_recuperada_por_patch;hipotese_regiao_zona_requer_revisao"
            action = "revisar_humano_quais_cenas_T23_correspondem_a_cada_patch_Petropolis"
        elif status == "LINEAGE_CONFLICT":
            stac_ready, block, action = "BLOCKED_CONFLICT", "candidatos_incompativeis", "resolver_conflito_humano"
        elif status == "LINEAGE_INVALID":
            stac_ready, block, action = "BLOCKED_INVALID", invalid_reason, "descartar_evidencia_invalida"
        else:
            stac_ready = "BLOCKED_NO_LINEAGE"
            block = "sem_cena_local_para_zona/regiao_deste_patch"
            action = "recuperar_historico_GEE_de_export_deste_patch_especifico"

        # Dia 10 pós-lineage: nunca desbloqueado (sem raster nativo nesta etapa).
        if status == "LINEAGE_RECOVERED_STRONG":
            day10 = "DAY10_READY_FOR_RASTER_DOWNLOAD_REVIEW"
        else:
            day10 = "DAY10_BLOCKED_NO_NATIVE_RASTER"

        rows.append({
            "lineage_binding_id": f"MV2_14_LB_{i:05d}",
            "target_id": clean(t.get("target_id")),
            "binding_id": clean(t.get("binding_id")),
            "anchor_id": clean(t.get("anchor_id")),
            "asset_id": asset_id,
            "patch_id": patch_id,
            "region": region,
            "recovered_scene_id": recovered["scene"],
            "recovered_datetime": recovered["dt"],
            "recovered_tile": recovered["tile"],
            "recovered_cloud_cover": recovered["cloud"],
            "recovered_sensor": recovered["sensor"],
            "export_task_or_source": recovered["src"],
            "evidence_source": recovered["src"] or "none",
            "evidence_snippet": recovered["ev"],
            "lineage_status": status,
            "confidence": confidence,
            "conflict_status": conflict,
            "invalid_reason": invalid_reason,
            "stac_readiness_after_lineage": stac_ready,
            "day10_status_after_lineage": day10,
            "blocking_reason": block,
            "next_action": action,
        })
    return rows


def main() -> None:
    ensure_output_dir()
    rows = build_rows()
    out = OUTPUT_DIR / "mv2_14_lineage_binding_matrix.csv"
    write_csv(out, COLUMNS, rows)
    from collections import Counter
    dist = Counter(r["lineage_status"] for r in rows)
    print(f"[mv2_14_reconcile] bindings: {len(rows)} | {dict(dist)}")
    print(f"[mv2_14_reconcile] saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
