"""MV2-DATA-04 Tarefa 2 — Selecionar batch de corpus targets (metadata-only).

Seleciona no máximo 15 targets do corpus, balanceado por região (até 5 cada), preferindo
maior prioridade. Exclui qualquer âncora oficial Protocolo-C e qualquer target sem bbox/CRS.

Saída: outputs_public/mv2_data_corpus_metadata_probe/mv2_data_04_corpus_metadata_batch.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_data_04_common import (
    BATCH_REGIONS,
    MAX_BATCH,
    MAX_PER_REGION,
    PROJECT_ROOT,
    PUBLIC_DIR,
    clean,
    crs_status,
    ensure_public_dir,
    load_corpus_targets,
    official_canary_ids,
    write_csv,
)

BATCH_ID = "MV2_DATA_04_BATCH_01"

COLUMNS = [
    "batch_id",
    "target_id",
    "asset_id",
    "patch_id",
    "region",
    "bbox",
    "crs",
    "priority",
    "selected_for_probe",
    "selection_reason",
    "is_official_canary",
    "can_probe_metadata",
    "blocking_reason",
]

# Ordem de prioridade (P0 > P1 > P2) para escolher dentro de cada região.
PRIORITY_RANK = {
    "P0_HIGH_SENSOR_CONFIRMED_QUERY": 0,
    "P1_MEDIUM_LOCAL_CANDIDATE_REVIEW": 1,
    "P2_LOW_NO_LOCAL_SCENE_QUERY": 2,
}


def build_rows() -> list[dict[str, object]]:
    targets = load_corpus_targets()
    canary = official_canary_ids()

    # Agrupa por região, mantendo só targets elegíveis (bbox+CRS, não-canary).
    by_region: dict[str, list[dict[str, str]]] = {r: [] for r in BATCH_REGIONS}
    for t in targets:
        region = clean(t.get("region"))
        if region not in by_region:
            continue
        is_canary = clean(t.get("patch_id")) in canary or clean(t.get("target_id")) in canary
        has_geo = clean(t.get("has_bbox")) == "true" and crs_status(clean(t.get("crs"))) != "ABSENT"
        if is_canary or not has_geo:
            continue
        by_region[region].append(t)

    for region in by_region:
        by_region[region].sort(key=lambda r: (PRIORITY_RANK.get(clean(r.get("priority")), 9),
                                              clean(r.get("target_id"))))

    # Seleção balanceada: até MAX_PER_REGION por região, respeitando MAX_BATCH total.
    selected: list[dict[str, str]] = []
    for region in BATCH_REGIONS:
        for t in by_region[region][:MAX_PER_REGION]:
            if len(selected) >= MAX_BATCH:
                break
            selected.append(t)

    rows: list[dict[str, object]] = []
    for t in selected:
        rows.append({
            "batch_id": BATCH_ID,
            "target_id": clean(t.get("target_id")),
            "asset_id": clean(t.get("asset_id")),
            "patch_id": clean(t.get("patch_id")),
            "region": clean(t.get("region")),
            "bbox": clean(t.get("bbox")),
            "crs": clean(t.get("crs")),
            "priority": clean(t.get("priority")),
            "selected_for_probe": "true",
            "selection_reason": "regiao_balanceada_ate_5_maior_prioridade_primeiro",
            "is_official_canary": "false",
            # Espacialmente pronto para preparar probe; execução ainda depende de janela temporal.
            "can_probe_metadata": "true",
            "blocking_reason": "probe_so_executa_com_janela_temporal_e_config",
        })
    return rows


def main() -> None:
    ensure_public_dir()
    rows = build_rows()
    out = PUBLIC_DIR / "mv2_data_04_corpus_metadata_batch.csv"
    write_csv(out, COLUMNS, rows)
    from collections import Counter

    dist = Counter(str(r["region"]) for r in rows)
    print(
        f"[mv2_data_04_batch] {len(rows)} targets | "
        + " ".join(f"{k}={v}" for k, v in sorted(dist.items()))
        + f" | saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
