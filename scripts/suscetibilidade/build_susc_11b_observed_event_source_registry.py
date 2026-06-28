"""
SUSC-11B Observed Event Source Registry Builder (review-only)

Builds a prioritized registry of official/technical/public target sources of
OBSERVED flood/inundation events per region (Recife, Petropolis, Curitiba).
URLs are institutional references; a direct file URL is only recorded when it is
already traceable in the repo (none invented here). No download happens here.

Writes:
  - manifests/suscetibilidade/susc_11b_observed_event_source_registry_v1.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_io import write_csv, rel  # noqa: E402

REGISTRY = ROOT / "manifests" / "suscetibilidade" / "susc_11b_observed_event_source_registry_v1.csv"

# (region, institution, source_name, source_type, priority, expected_event_type,
#  expected_geometry_type, url, url_is_direct_download, download_allowed, notes)
# url_is_direct_download stays "false" unless a traceable direct file is known.
# The institutional url documents the official source for manual acquisition.
SOURCES = [
    # --- Recife ---
    ("recife", "APAC", "APAC alagamento/inundacao bacias", "state", "high",
     "urban_flooding", "polygon|shapefile", "https://www.apac.pe.gov.br", "false", "false",
     "agencia estadual de aguas e clima; mancha/bacias"),
    ("recife", "Defesa Civil Recife (CODECIR)", "ocorrencias coordenadas", "municipal", "high",
     "urban_flooding", "point|csv", "https://www2.recife.pe.gov.br/codecir", "false", "false",
     "ocorrencias municipais com coordenada"),
    ("recife", "Prefeitura do Recife (Dados Abertos)", "dados abertos alagamento", "municipal", "high",
     "urban_flooding", "geojson|csv", "http://dados.recife.pe.gov.br", "false", "false",
     "portal de dados abertos municipal"),
    ("recife", "CEMADEN", "ocorrencias/alertas lat-lon", "federal", "medium",
     "flood", "point|csv", "https://www.cemaden.gov.br", "false", "false",
     "monitoramento nacional; alerta != ocorrencia"),
    ("recife", "CPRM/SGB", "setores de risco / relatorios", "federal", "medium",
     "landslide_related_flood", "polygon|pdf", "https://www.sgb.gov.br", "false", "false",
     "setor de risco != evento observado"),
    ("recife", "ANA/Hidroweb", "cota fluviometrica Capibaribe", "federal", "medium",
     "inundation", "point|csv", "https://www.snirh.gov.br/hidroweb", "false", "false",
     "estacoes fluviometricas; cota != footprint"),
    ("recife", "S2iD (Defesa Civil Nacional)", "registros de desastre S2iD", "federal", "medium",
     "flood", "point|csv", "https://s2id.mi.gov.br", "false", "false",
     "registro nacional de desastres"),
    ("recife", "Charter/midia tecnica", "mapas de mancha rastreaveis (apoio secundario)", "news", "low",
     "flood", "polygon", "https://disasterscharter.org", "false", "false",
     "apoio secundario; digitalizacao candidata != footprint validado"),
    # --- Petropolis ---
    ("petropolis", "CPRM/SGB", "setores de risco / relatorios 2022", "federal", "high",
     "landslide_related_flood", "polygon|pdf", "https://www.sgb.gov.br", "false", "false",
     "relatorios tecnicos oficiais do desastre 2022"),
    ("petropolis", "Defesa Civil Petropolis/RJ", "ocorrencias inundacao 2022", "municipal", "high",
     "flash_flood", "point|csv", "https://www.petropolis.rj.gov.br", "false", "false",
     "ocorrencias municipais"),
    ("petropolis", "INEA/RJ", "mancha de inundacao / areas atingidas", "state", "high",
     "inundation", "polygon|geojson", "https://www.inea.rj.gov.br", "false", "false",
     "instituto estadual do ambiente"),
    ("petropolis", "Prefeitura de Petropolis", "dados abertos / boletins", "municipal", "medium",
     "flash_flood", "csv|pdf", "https://www.petropolis.rj.gov.br", "false", "false",
     "boletins municipais"),
    ("petropolis", "CEMADEN", "ocorrencias/alertas lat-lon", "federal", "medium",
     "flash_flood", "point|csv", "https://www.cemaden.gov.br", "false", "false",
     "monitoramento nacional; alerta != ocorrencia"),
    ("petropolis", "ANA/Hidroweb/SACE", "cota/vazao bacia", "federal", "medium",
     "inundation", "point|csv", "https://www.snirh.gov.br/hidroweb", "false", "false",
     "estacoes; cota != footprint"),
    ("petropolis", "S2iD (Defesa Civil Nacional)", "registros de desastre S2iD", "federal", "medium",
     "flash_flood", "point|csv", "https://s2id.mi.gov.br", "false", "false",
     "registro nacional de desastres"),
    # --- Curitiba ---
    ("curitiba", "GeoCuritiba/IPPUC", "areas de alagamento / hidrografia", "municipal", "high",
     "urban_flooding", "geojson|shapefile", "https://geocuritiba.ippuc.org.br", "false", "false",
     "geoportal municipal"),
    ("curitiba", "IPPUC", "dados urbanisticos / alagamento", "municipal", "high",
     "urban_flooding", "geojson|csv", "https://ippuc.org.br", "false", "false",
     "instituto de pesquisa e planejamento urbano"),
    ("curitiba", "Prefeitura de Curitiba (Dados Abertos)", "dados abertos alagamento", "municipal", "high",
     "urban_flooding", "geojson|csv", "https://www.curitiba.pr.gov.br", "false", "false",
     "portal de dados abertos municipal"),
    ("curitiba", "Defesa Civil Curitiba/PR", "ocorrencias inundacao", "municipal", "high",
     "urban_flooding", "point|csv", "https://www.curitiba.pr.gov.br/defesacivil", "false", "false",
     "ocorrencias municipais"),
    ("curitiba", "IAT/Aguas Parana", "mancha/bacia hidrografica", "state", "medium",
     "inundation", "polygon|geojson", "https://www.iat.pr.gov.br", "false", "false",
     "instituto agua e terra"),
    ("curitiba", "CEMADEN", "ocorrencias/alertas lat-lon", "federal", "medium",
     "urban_flooding", "point|csv", "https://www.cemaden.gov.br", "false", "false",
     "monitoramento nacional; alerta != ocorrencia"),
    ("curitiba", "ANA/Hidroweb", "cota fluviometrica", "federal", "medium",
     "inundation", "point|csv", "https://www.snirh.gov.br/hidroweb", "false", "false",
     "estacoes; cota != footprint"),
    ("curitiba", "S2iD (Defesa Civil Nacional)", "registros de desastre S2iD", "federal", "medium",
     "urban_flooding", "point|csv", "https://s2id.mi.gov.br", "false", "false",
     "registro nacional de desastres"),
]

FIELDS = ["source_id", "region", "institution", "source_name", "source_type",
          "priority", "expected_event_type", "expected_geometry_type", "url",
          "url_is_direct_download", "download_allowed", "max_size_mb",
          "requires_manual_review", "can_be_ground_truth", "allowed_for_training",
          "review_only", "notes"]


def build_rows() -> list[dict]:
    rows = []
    for i, (region, inst, name, stype, prio, etype, geom, url, direct, dl, notes) in enumerate(SOURCES):
        rows.append({
            "source_id": f"OBSSRC_{i:03d}", "region": region, "institution": inst,
            "source_name": name, "source_type": stype, "priority": prio,
            "expected_event_type": etype, "expected_geometry_type": geom, "url": url,
            "url_is_direct_download": direct, "download_allowed": dl, "max_size_mb": "100",
            "requires_manual_review": "true", "can_be_ground_truth": "false",
            "allowed_for_training": "false", "review_only": "true", "notes": notes,
        })
    return rows


def main() -> int:
    rows = build_rows()
    write_csv(REGISTRY, rows, FIELDS)
    n_direct = sum(1 for r in rows if r["url_is_direct_download"] == "true")
    by_region = {}
    for r in rows:
        by_region[r["region"]] = by_region.get(r["region"], 0) + 1
    print(f"registry -> {rel(REGISTRY)} ({len(rows)} target sources, {n_direct} direct-download)")
    print("by region:", by_region)
    print("No invented URL. Direct download only when traceable. review-only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
