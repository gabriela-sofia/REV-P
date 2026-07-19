"""MV2-12 — Backlog de geometrias de evento e bloqueios do gate G4.

Lê os 9 candidatos de evento do Protocolo C
(``docs/protocolo_c/v2aq_event_geometry_patch_link/geojson_candidates/``) e o estado
do produto Charter758 (Recife mai/2022), e produz um backlog objetivo do que falta
para cada evento poder suportar G4 (overlay/positivo formal).

Regras: município inteiro NÃO é geometria de evento observado; ausência de footprint
NÃO vira negativo; nenhum evento é promovido a ground truth aqui.

Saída: outputs_public/mv2_data_readiness/mv2_12_event_geometry_backlog.csv
"""

from __future__ import annotations

import json
from pathlib import Path

from mv2_12_data_readiness_common import (
    CHARTER_DIR,
    EVENT_GEOJSON_DIR,
    OUTPUT_DIR,
    REGION_PHENOMENON,
    ensure_output_dir,
    write_csv,
)

COLUMNS = [
    "event_id",
    "region",
    "phenomenon_type",
    "event_date",
    "current_geometry_status",
    "current_crs_status",
    "source_document",
    "has_observed_footprint",
    "requires_digitization",
    "requires_external_download",
    "can_support_g4",
    "can_support_silver",
    "blocking_reason",
    "next_action",
]


def _event_date_from_id(candidate_id: str) -> str:
    # CTB_2022_01_15_16 -> 2022-01-15..16 ; PET_2022_02_15 -> 2022-02-15
    parts = candidate_id.split("_")[1:]
    if len(parts) >= 3:
        base = f"{parts[0]}-{parts[1]}-{parts[2]}"
        if len(parts) >= 4:
            return f"{base}..{parts[3]}"
        return base
    return "DESCONHECIDO"


def _charter_state() -> dict[str, bool]:
    """Estado do produto Charter758 para REC_2022_05_24_30."""
    png_dir = CHARTER_DIR / "raw"
    has_png = png_dir.exists() and any(
        p.suffix.lower() == ".png" for p in png_dir.glob("*.png")
    )
    digitized = (
        CHARTER_DIR
        / "derived"
        / "event_polygon_REC_2022_05_24_30_charter758_digitized_candidate.geojson"
    )
    has_real_vector = False
    if digitized.exists():
        data = json.loads(digitized.read_text(encoding="utf-8"))
        feats = data.get("features", []) if isinstance(data, dict) else []
        has_real_vector = any(f.get("geometry") for f in feats)
    return {"has_png": has_png, "has_real_vector": has_real_vector}


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    charter = _charter_state()
    files = sorted(EVENT_GEOJSON_DIR.glob("v2aq_event_geometry_*.geojson"))
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        feats = data.get("features", []) if isinstance(data, dict) else []
        props = feats[0].get("properties", {}) if feats else {}
        geom = feats[0].get("geometry") if feats else None
        candidate_id = props.get("candidate_id", path.stem)
        region = props.get("region", "DESCONHECIDO")
        geom_status = props.get("geometry_status", "UNKNOWN")
        has_footprint = geom is not None

        is_recife_charter = candidate_id == "REC_2022_05_24_30"
        source_doc = "datasets/protocolo_c/v2an_spatial_anchor_registry.csv"
        requires_external = False
        if is_recife_charter:
            source_doc = (
                "Copernicus EMS EMSR601 / Charter758 (raster PNG baixado; vetor não)"
            )
            requires_external = True  # vetor oficial exige acesso pago/institucional

        # Próxima ação por tipo de status de geometria.
        if geom_status == "OFFICIAL_MAP_DIGITIZATION_REQUIRED":
            next_action = "digitalizar_footprint_de_mapa_oficial_com_CRS_e_incerteza"
            requires_digitization = True
        elif geom_status == "TEXTUAL_ANCHOR_ONLY":
            next_action = "obter_mapa_ou_pontos_oficiais_antes_de_digitalizar"
            requires_digitization = True
        elif geom_status == "INSUFFICIENT_GEOMETRY":
            next_action = "adquirir_fonte_oficial_de_geometria_inexistente_localmente"
            requires_digitization = True
        else:
            next_action = "revisar_status_de_geometria"
            requires_digitization = True

        if is_recife_charter and charter["has_png"] and not charter["has_real_vector"]:
            next_action = (
                "digitalizar_poligono_de_inundacao_do_PNG_Charter758_em_CRS_definido"
            )

        blocking = (
            "geometry=null em todos os candidatos; ancora textual/mapa não digitalizado"
        )

        rows.append(
            {
                "event_id": candidate_id,
                "region": region,
                "phenomenon_type": REGION_PHENOMENON.get(region, "DESCONHECIDO"),
                "event_date": _event_date_from_id(candidate_id),
                "current_geometry_status": geom_status,
                "current_crs_status": "CRS_AUSENTE" if not has_footprint else "CRS_PRESENTE",
                "source_document": source_doc,
                "has_observed_footprint": str(has_footprint).lower(),
                "requires_digitization": str(requires_digitization).lower(),
                "requires_external_download": str(requires_external).lower(),
                # Sem geometria observada não há suporte a G4 nem a silver.
                "can_support_g4": "false",
                "can_support_silver": "false",
                "blocking_reason": blocking,
                "next_action": next_action,
            }
        )
    return rows


def main() -> None:
    ensure_output_dir()
    rows = build_rows()
    out = OUTPUT_DIR / "mv2_12_event_geometry_backlog.csv"
    write_csv(out, COLUMNS, rows)
    n_g4 = sum(1 for r in rows if r["can_support_g4"] == "true")
    print(f"[mv2_12_event_geometry] eventos no backlog: {len(rows)}")
    print(f"[mv2_12_event_geometry] eventos que suportam G4 agora: {n_g4}")
    print(f"[mv2_12_event_geometry] saida: {Path(out).relative_to(EVENT_GEOJSON_DIR.parents[3]).as_posix()}")


if __name__ == "__main__":
    main()
