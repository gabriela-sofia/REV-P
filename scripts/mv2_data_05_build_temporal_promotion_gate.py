"""MV2-DATA-05 Tarefa 4 — Gate de promoção temporal.

Converte a validação de evidência em decisão de promoção: só janelas STRONG/PARTIAL
habilitam o GEE/STAC metadata probe (em DATA-06). OData ainda exige product_id/scene_id.
can_unlock_day10_now=false sempre.

Saída: outputs_public/mv2_data_temporal_window_intake/mv2_data_05_temporal_promotion_gate.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_data_05_common import (
    PROJECT_ROOT,
    PUBLIC_DIR,
    clean,
    ensure_public_dir,
    read_csv_rows,
    write_csv,
)

COLUMNS = [
    "promotion_id",
    "target_id",
    "asset_id",
    "patch_id",
    "region",
    "temporal_window_start",
    "temporal_window_end",
    "temporal_window_source",
    "source_ref",
    "review_status",
    "promotion_status",
    "can_enable_gee_metadata_probe",
    "can_enable_stac_metadata_probe",
    "can_enable_odata_probe",
    "can_unlock_day10_now",
    "blocking_reason",
    "next_action",
]

# evidence_status -> promotion_status
EVIDENCE_TO_PROMOTION = {
    "TEMPORAL_EVIDENCE_VALID_STRONG": "TEMPORAL_WINDOW_PROMOTED_STRONG",
    "TEMPORAL_EVIDENCE_VALID_PARTIAL": "TEMPORAL_WINDOW_PROMOTED_PARTIAL",
    "TEMPORAL_EVIDENCE_REVIEW_REQUIRED": "TEMPORAL_WINDOW_REVIEW_REQUIRED",
    "TEMPORAL_EVIDENCE_EMPTY": "TEMPORAL_WINDOW_BLOCKED_EMPTY",
    "TEMPORAL_EVIDENCE_CONFLICT": "TEMPORAL_WINDOW_BLOCKED_CONFLICT",
    "TEMPORAL_EVIDENCE_INVALID": "TEMPORAL_WINDOW_BLOCKED_INVALID",
}


def build_rows() -> list[dict[str, object]]:
    evidence = read_csv_rows(PUBLIC_DIR / "mv2_data_05_temporal_evidence_validation.csv")
    rows: list[dict[str, object]] = []
    for i, e in enumerate(evidence, start=1):
        status = clean(e.get("evidence_status"))
        promotion = EVIDENCE_TO_PROMOTION.get(status, "TEMPORAL_WINDOW_BLOCKED_INVALID")
        promoted = promotion in {"TEMPORAL_WINDOW_PROMOTED_STRONG", "TEMPORAL_WINDOW_PROMOTED_PARTIAL"}

        if promoted:
            blocking = ""
            nxt = "habilitar_gee_stac_metadata_probe_em_DATA-06_com_esta_janela"
        elif promotion == "TEMPORAL_WINDOW_REVIEW_REQUIRED":
            blocking = clean(e.get("blocking_reason")) or "evidencia_insuficiente"
            nxt = "corrigir_no_correction_template_e_reenviar"
        else:
            blocking = clean(e.get("blocking_reason")) or "janela_nao_promovel"
            nxt = "preencher_janela_temporal_com_evidencia_no_correction_template"

        rows.append({
            "promotion_id": f"MV2_DATA_05_PROM_{i:03d}",
            "target_id": clean(e.get("target_id")),
            "asset_id": clean(e.get("asset_id")),
            "patch_id": clean(e.get("patch_id")),
            "region": clean(e.get("region")),
            "temporal_window_start": clean(e.get("temporal_window_start")),
            "temporal_window_end": clean(e.get("temporal_window_end")),
            "temporal_window_source": clean(e.get("temporal_window_source")),
            "source_ref": clean(e.get("source_ref")),
            "review_status": clean(e.get("review_status")),
            "promotion_status": promotion,
            # GEE/STAC só com janela promovida (strong/partial).
            "can_enable_gee_metadata_probe": "true" if promoted else "false",
            "can_enable_stac_metadata_probe": "true" if promoted else "false",
            # OData ainda exige product_id/scene_id (não resolvido neste marco).
            "can_enable_odata_probe": "false",
            "can_unlock_day10_now": "false",
            "blocking_reason": blocking,
            "next_action": nxt,
        })
    return rows


def main() -> None:
    ensure_public_dir()
    rows = build_rows()
    out = PUBLIC_DIR / "mv2_data_05_temporal_promotion_gate.csv"
    write_csv(out, COLUMNS, rows)
    from collections import Counter

    dist = Counter(str(r["promotion_status"]) for r in rows)
    print(
        f"[mv2_data_05_gate] {len(rows)} | "
        + " ".join(f"{k.replace('TEMPORAL_WINDOW_', '')}={v}" for k, v in sorted(dist.items()))
        + f" | saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
