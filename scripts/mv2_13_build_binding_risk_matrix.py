"""MV2-13 Tarefa 12 — Matriz de risco de binding.

Emite riscos obrigatórios derivados da matriz de binding, garantindo que cada tipo
de risco previsto pelo marco esteja representado (ainda que como risco mitigado/controlado).

Saída: outputs_public/mv2_sentinel_binding/mv2_13_binding_risk_matrix.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_13_sentinel_binding_common import (
    OUTPUT_DIR,
    PROJECT_ROOT,
    RISK_TYPES,
    clean,
    ensure_output_dir,
    read_csv_rows,
    write_csv,
)

COLUMNS = [
    "risk_id", "related_binding_id", "related_anchor_id", "severity",
    "risk_type", "description", "evidence", "mitigation", "blocks_next_step",
]

# Descrição/mitigação por tipo de risco. counts são preenchidos a partir da matriz.
RISK_SPEC = {
    "RISK_WEAK_FILENAME_INFERENCE": (
        "MEDIA", "Vínculo por filename/texto fraco poderia inflar binding.",
        "promover binding fraco a forte por nome", "binding fraco nunca vira STAC real; regra explícita no reconcile", "true"),
    "RISK_CITY_PROXY_BINDING": (
        "ALTA", "Usar cidade como proxy de binding asset/patch.",
        "cidade/região como vínculo", "cidade/região nunca basta para binding medio/forte", "true"),
    "RISK_REGION_PROXY_LABEL": (
        "ALTA", "Usar região como proxy de label/classe.",
        "região como label", "nenhum label é criado; região é só contexto", "true"),
    "RISK_PNG_AS_RASTER": (
        "ALTA", "Tratar PNG/render como raster Sentinel nativo.",
        "is_native_sentinel_raster a partir de PNG", "is_native só de raster_readable; PNG/NPZ marcados false", "true"),
    "RISK_NPZ_AS_SPECTRAL": (
        "ALTA", "Tratar NPZ/DINOv2 como feature espectral.",
        "NPZ como espectral", "NPZ/DINOv2 nunca é raster nativo; feature espectral=0", "true"),
    "RISK_MISSING_CRS": (
        "MEDIA", "CRS ausente impede AOI/STAC.",
        "crs_status=ABSENT", "STAC real bloqueado sem CRS", "true"),
    "RISK_MISSING_GEOMETRY": (
        "ALTA", "Geometria ausente impede AOI.",
        "geometry_status=ABSENT", "STAC real bloqueado sem geometria", "true"),
    "RISK_MISSING_SCENE_ID": (
        "ALTA", "scene_id ausente impede STAC real.",
        "128 patches espaciais sem scene_id", "dry-run só rascunho; real bloqueado", "true"),
    "RISK_MISSING_DATETIME": (
        "ALTA", "datetime ausente/inválido impede Trilha A e STAC real.",
        "datetime ABSENT ou INVALID_FUTURE", "datas futuras descartadas; real bloqueado", "true"),
    "RISK_STAC_WITHOUT_PATCH": (
        "ALTA", "STAC sem patch_id (cenas soltas).",
        "10 anchors strong-scene sem patch", "STAC bloqueado NO_PATCH", "true"),
    "RISK_STAC_WITHOUT_ASSET": (
        "ALTA", "STAC sem asset_id oficial.",
        "anchors sem asset", "STAC bloqueado NO_ASSET", "true"),
    "RISK_DAY10_FALSE_UNLOCK": (
        "CRITICA", "Marcar Dia 10 desbloqueado sem raster nativo.",
        "0 raster nativo no corpus", "can_unlock_day10_now=false enquanto raster ausente", "true"),
    "RISK_TRAINING_BEFORE_SILVER": (
        "CRITICA", "Treinar antes de silver/positivos/negativos formais.",
        "silver=0, negativos=0", "can_train=false; sandbox bloqueado", "true"),
    "RISK_UNKNOWN_AS_NEGATIVE": (
        "CRITICA", "Tratar ausência de evidência como negativo/classe 0.",
        "BINDING_NONE=ausência", "ausência nunca vira negativo; negatives_created=0", "true"),
}


def build_rows() -> list[dict[str, object]]:
    binding = read_csv_rows(OUTPUT_DIR / "mv2_13_binding_matrix.csv")
    # Evidência quantitativa por risco a partir da matriz.
    n_no_scene = sum(1 for b in binding if not clean(b.get("scene_id")) and clean(b.get("binding_strength")) == "BINDING_MEDIUM")
    n_no_patch = sum(1 for b in binding if not clean(b.get("patch_id")) and clean(b.get("scene_id")))
    n_invalid = sum(1 for b in binding if clean(b.get("conflict_status")) == "INVALID")
    n_none = sum(1 for b in binding if clean(b.get("binding_strength")) == "BINDING_NONE")
    extra = {
        "RISK_MISSING_SCENE_ID": f"{n_no_scene} MEDIUM sem scene_id",
        "RISK_STAC_WITHOUT_PATCH": f"{n_no_patch} cenas sem patch",
        "RISK_MISSING_DATETIME": f"{n_invalid} datetime invalido futuro",
        "RISK_UNKNOWN_AS_NEGATIVE": f"{n_none} BINDING_NONE (ausencia, nunca negativo)",
    }

    rows: list[dict[str, object]] = []
    for i, risk in enumerate(RISK_TYPES, start=1):
        sev, desc, ev_default, mit, blocks = RISK_SPEC[risk]
        rows.append(
            {
                "risk_id": f"MV2_13_RISK_{i:02d}",
                "related_binding_id": "ALL",
                "related_anchor_id": "ALL",
                "severity": sev,
                "risk_type": risk,
                "description": desc,
                "evidence": extra.get(risk, ev_default),
                "mitigation": mit,
                "blocks_next_step": blocks,
            }
        )
    return rows


def main() -> None:
    ensure_output_dir()
    rows = build_rows()
    out = OUTPUT_DIR / "mv2_13_binding_risk_matrix.csv"
    write_csv(out, COLUMNS, rows)
    print(f"[mv2_13_risk] riscos: {len(rows)}")
    print(f"[mv2_13_risk] saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
