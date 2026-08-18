"""
SUSC-09B External Source Registry Builder (review-only)

Builds a prioritized registry of official/public vector sources per region. URLs
are institutional references; direct file URLs are only included when already
known/traceable in the repo. No download happens here.

Writes:
  - manifests/suscetibilidade/susc_09b_external_source_registry_v1.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_io import write_csv, rel  # noqa: E402

REGISTRY = ROOT / "manifests" / "suscetibilidade" / "susc_09b_external_source_registry_v1.csv"

# (region, institution, dataset/document, reference_url, expected_geometry, direct_url)
# direct_url left empty unless a traceable direct file is known; institutional
# reference_url documents the official source for manual acquisition.
SOURCES = [
    ("curitiba", "GeoCuritiba/IPPUC", "areas de risco / alagamento / hidrografia",
     "https://geocuritiba.ippuc.org.br", "geojson|shapefile", ""),
    ("curitiba", "Prefeitura de Curitiba", "dados abertos alagamento",
     "https://www.curitiba.pr.gov.br", "geojson|csv", ""),
    ("curitiba", "Defesa Civil Curitiba/PR", "ocorrencias inundacao",
     "https://www.curitiba.pr.gov.br/defesacivil", "csv|pdf", ""),
    ("recife", "APAC", "alagamento/inundacao bacias",
     "https://www.apac.pe.gov.br", "shapefile|geojson", ""),
    ("recife", "Prefeitura do Recife", "dados abertos alagamento",
     "http://dados.recife.pe.gov.br", "geojson|csv", ""),
    ("recife", "Defesa Civil Recife/PE", "ocorrencias coordenadas",
     "https://www2.recife.pe.gov.br/codecir", "csv|pdf", ""),
    ("petropolis", "SGB/CPRM", "setores de risco / relatorios 2022",
     "https://www.sgb.gov.br", "shapefile|pdf", ""),
    ("petropolis", "Defesa Civil Petropolis/RJ", "ocorrencias inundacao 2022",
     "https://www.petropolis.rj.gov.br", "csv|pdf", ""),
    ("petropolis", "INEA/RJ", "mancha de inundacao / areas atingidas",
     "https://www.inea.rj.gov.br", "shapefile|geojson", ""),
    ("multi", "ANA/Hidroweb", "estacoes/cota fluviometrica",
     "https://www.snirh.gov.br/hidroweb", "csv", ""),
    ("multi", "CEMADEN", "ocorrencias lat/lon",
     "https://www.cemaden.gov.br", "csv", ""),
    ("multi", "INMET", "estacoes meteorologicas coordenadas",
     "https://portal.inmet.gov.br", "csv", ""),
    ("multi", "MapBiomas", "agua/uso do solo (manifesto leve, se houver)",
     "https://brasil.mapbiomas.org", "geojson|csv", ""),
]

FIELDS = ["source_id", "region", "target_institution", "target_dataset_or_document",
          "reference_url", "direct_url", "expected_geometry_type", "source_tier",
          "download_allowed", "requires_manual_review", "can_be_ground_truth",
          "allowed_for_training", "review_only", "notes"]


def main() -> int:
    rows = []
    for i, (region, inst, dataset, ref, geom, direct) in enumerate(SOURCES):
        rows.append({
            "source_id": f"EXTSRC_{i:03d}", "region": region,
            "target_institution": inst, "target_dataset_or_document": dataset,
            "reference_url": ref, "direct_url": direct, "expected_geometry_type": geom,
            "source_tier": "official",
            "download_allowed": "true" if direct else "false",
            "requires_manual_review": "true",
            "can_be_ground_truth": "false", "allowed_for_training": "false", "review_only": "true",
            "notes": "institutional reference; direct file acquisition is manual unless direct_url present",
        })
    write_csv(REGISTRY, rows, FIELDS)
    print(f"registry -> {rel(REGISTRY)} ({len(rows)} official sources, "
          f"{sum(1 for r in rows if r['direct_url'])} with direct_url)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
