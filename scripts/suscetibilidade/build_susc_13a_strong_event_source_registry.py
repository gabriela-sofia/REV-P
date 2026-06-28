"""
SUSC-13A strong observed-event source registry (review-only).

Registers prioritized official/technical targets for stronger observed
flood/inundation evidence. No URL is invented as a direct download; direct/light
downloads are only allowed when an explicit file URL is known.

Writes:
  - manifests/suscetibilidade/susc_13a_strong_event_source_registry_v1.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_io import write_csv, rel  # noqa: E402

REGISTRY = ROOT / "manifests" / "suscetibilidade" / "susc_13a_strong_event_source_registry_v1.csv"

FIELDS = [
    "source_id",
    "region",
    "institution",
    "source_name",
    "priority",
    "expected_geometry",
    "expected_temporal_precision",
    "url",
    "url_is_direct_download",
    "download_allowed",
    "manual_search_required",
    "expected_evidence_level",
    "notes",
]

SOURCES = [
    ("recife", "Defesa Civil Recife (CODECIR)", "ocorrencias oficiais de alagamento com coordenada", "high", "point|csv|geojson", "day|period", "https://www2.recife.pe.gov.br/codecir", "strong_observed_flood_point|moderate_official_occurrence_point", "prioridade para ocorrencia ocorrida, data e lat/lon; risco nao serve como evento observado"),
    ("recife", "APAC", "alagamentos/inundacoes observados e bacias afetadas", "high", "polygon|bbox|csv", "day|period", "https://www.apac.pe.gov.br", "strong_observed_flood_polygon|moderate_official_flood_bbox", "buscar mancha/area efetivamente inundada; alerta hidrometeorologico sozinho nao serve"),
    ("recife", "Prefeitura do Recife Dados Abertos", "bases abertas de alagamento/ocorrencia", "high", "csv|geojson|shapefile_zip", "day|period", "http://dados.recife.pe.gov.br", "strong_observed_flood_point|moderate_official_occurrence_point", "preferir CSV/GeoJSON oficial com data e coordenadas"),
    ("recife", "CPRM/SGB", "relatorios tecnicos de inundacao/risco", "medium", "polygon|pdf|shapefile_zip", "day|period|unknown", "https://www.sgb.gov.br", "moderate_official_flood_bbox|weak_risk_area_context", "setor de risco nao e evento observado; footprint observado precisa estar explicito"),
    ("recife", "S2iD Defesa Civil Nacional", "registros oficiais de desastre", "medium", "point|csv|pdf", "day|period", "https://s2id.mi.gov.br", "moderate_official_occurrence_point", "registro municipal/federal ajuda temporalmente; precisa de geometria explicita para patch"),
    ("petropolis", "CPRM/SGB", "footprint ou relatorio tecnico 2022 Petropolis", "high", "polygon|bbox|pdf|shapefile_zip", "day|period", "https://www.sgb.gov.br", "strong_observed_flood_polygon|moderate_official_flood_bbox", "buscar area atingida em 2022; setor de risco isolado fica fraco"),
    ("petropolis", "Defesa Civil Petropolis", "ocorrencias oficiais 2022 com coordenada", "high", "point|csv|geojson", "day|period", "https://www.petropolis.rj.gov.br", "strong_observed_flood_point|moderate_official_occurrence_point", "precisa coordenada ou geometria e data do evento"),
    ("petropolis", "Prefeitura de Petropolis", "boletins e dados abertos de ocorrencias", "medium", "csv|pdf|geojson", "day|period", "https://www.petropolis.rj.gov.br", "moderate_official_occurrence_point", "boletim sem coordenada fica documental"),
    ("petropolis", "INEA/RJ", "mancha/area atingida por inundacao", "high", "polygon|geojson|shapefile_zip", "day|period", "https://www.inea.rj.gov.br", "strong_observed_flood_polygon|moderate_official_flood_bbox", "preferir vetor oficial com CRS e periodo"),
    ("petropolis", "S2iD Defesa Civil Nacional", "registros oficiais de desastre 2022", "medium", "point|csv|pdf", "day|period", "https://s2id.mi.gov.br", "moderate_official_occurrence_point", "registro sem geometria fica contexto documental"),
    ("curitiba", "GeoCuritiba/IPPUC", "ocorrencias/areas de alagamento georreferenciadas", "high", "geojson|shapefile_zip|csv", "day|period|unknown", "https://geocuritiba.ippuc.org.br", "strong_observed_flood_polygon|moderate_official_occurrence_point", "risco/suscetibilidade nao vira evento; buscar camada de ocorrencia"),
    ("curitiba", "Defesa Civil Curitiba/PR", "ocorrencias oficiais de alagamento com coordenada", "high", "point|csv|geojson", "day|period", "https://www.curitiba.pr.gov.br/defesacivil", "strong_observed_flood_point|moderate_official_occurrence_point", "precisa data e coordenada de ocorrencia"),
    ("curitiba", "Prefeitura de Curitiba", "dados abertos de alagamento/inundacao", "high", "csv|geojson|shapefile_zip", "day|period", "https://www.curitiba.pr.gov.br", "strong_observed_flood_point|moderate_official_occurrence_point", "preferir arquivo aberto direto com geometria"),
    ("curitiba", "IAT/Aguas Parana", "mancha, bacia ou registro hidrologico", "medium", "polygon|csv|pdf|shapefile_zip", "day|period|unknown", "https://www.iat.pr.gov.br", "moderate_official_flood_bbox", "cota/estacao sem footprint fica administrativo"),
    ("curitiba", "S2iD Defesa Civil Nacional", "registros oficiais de desastre", "medium", "point|csv|pdf", "day|period", "https://s2id.mi.gov.br", "moderate_official_occurrence_point", "ajuda como fonte oficial, mas exige geometria explicita para patch"),
]


def build_rows() -> list[dict]:
    rows = []
    for i, (region, inst, name, priority, geom, temporal, url, level, notes) in enumerate(SOURCES):
        rows.append(
            {
                "source_id": f"S13ASRC_{i:03d}",
                "region": region,
                "institution": inst,
                "source_name": name,
                "priority": priority,
                "expected_geometry": geom,
                "expected_temporal_precision": temporal,
                "url": url,
                "url_is_direct_download": "false",
                "download_allowed": "false",
                "manual_search_required": "true",
                "expected_evidence_level": level,
                "notes": notes,
            }
        )
    return rows


def main() -> int:
    rows = build_rows()
    write_csv(REGISTRY, rows, FIELDS)
    by_region: dict[str, int] = {}
    for row in rows:
        by_region[row["region"]] = by_region.get(row["region"], 0) + 1
    print(f"registry -> {rel(REGISTRY)} ({len(rows)} sources)")
    print("by region:", by_region)
    print("No invented direct URL. Manual search required. review-only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
