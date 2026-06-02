"""REV-P v1sg — Official source endpoint registry.

Creates a registry of official endpoints and download candidates.
Does NOT download yet; only builds the queue. Review-only.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

from revp_v1sg_v1sz_official_download_common import (
    DATASETS, DOCS, SCHEMAS, _p, guardrail_row, read_csv_safe,
    write_csv_with_header, write_doc, write_schema_for,
    is_allowed_domain, domain_from_url, classify_source_family,
    default_allowed_sources, load_allowed_sources_config,
    forbidden_guardrail_scan,
)

ROOT = Path(__file__).resolve().parents[2]

IN_CONFIG = _p("REVP_V1SG_IN_CONFIG", ROOT / "configs" / "revp_official_source_endpoints_v1sg.csv")
OUT_REGISTRY = _p("REVP_V1SG_OUT_REGISTRY", DATASETS / "protocol_c_official_source_endpoint_registry_v1sg.csv")
OUT_SUMMARY = _p("REVP_V1SG_OUT_SUMMARY", DATASETS / "protocol_c_official_source_endpoint_summary_v1sg.csv")
SCHEMA_REG = _p("REVP_V1SG_SCHEMA_REG", SCHEMAS / "protocol_c_official_source_endpoint_registry_v1sg_schema.csv")
SCHEMA_SUM = _p("REVP_V1SG_SCHEMA_SUM", SCHEMAS / "protocol_c_official_source_endpoint_summary_v1sg_schema.csv")
DOC = _p("REVP_V1SG_DOC", DOCS / "revp_v1sg_official_source_endpoint_registry.md")

REG_FIELDS = [
    "endpoint_id", "source_name", "source_family", "base_url", "domain",
    "endpoint_type", "region_scope", "hazard_scope", "expected_file_types",
    "download_strategy", "requires_manual_url", "allowed_domain", "robots_status",
    "rate_limit_seconds", "enabled_by_default", "priority", "review_only",
    "blocked_reason", "notes",
]
SUM_FIELDS = ["stat_key", "stat_value"]

_SEEDS = [
    ("INMET_HIST", "INMET dados historicos anuais", "https://portal.inmet.gov.br/uploads/dadoshistoricos/", "portal.inmet.gov.br",
     "FILE_LISTING", "ALL", "HYDROMETEOROLOGICAL", "ZIP/CSV", "ANNUAL_ZIP_DOWNLOAD", "false", "true", "P0"),
    ("ANA_HIDROWEB", "ANA HidroWeb series", "https://www.snirh.gov.br/hidroweb/seriesHistoricas", "www.snirh.gov.br",
     "WEB_FORM", "ALL", "HYDROMETEOROLOGICAL", "CSV", "MANUAL_QUERY_OR_API", "true", "true", "P0"),
    ("ANA_TELEMETRIA", "ANA Telemetria", "https://telemetriaws1.ana.gov.br/ServiceANA.asmx", "telemetriaws1.ana.gov.br",
     "SOAP_API", "ALL", "HYDROMETEOROLOGICAL", "XML/CSV", "API_CALL", "false", "true", "P1"),
    ("CEMADEN_PORTAL", "CEMADEN alertas", "https://www.gov.br/cemaden/pt-br", "www.gov.br",
     "WEB_PORTAL", "ALL", "HYDROMETEOROLOGICAL", "HTML/PDF", "MANUAL_DISCOVERY", "true", "false", "P1"),
    ("SGB_RIGEO", "SGB/CPRM RIGEO", "https://rigeo.sgb.gov.br/", "rigeo.sgb.gov.br",
     "WEB_PORTAL", "PET", "GEOLOGICAL", "PDF/ZIP", "MANUAL_DISCOVERY", "true", "false", "P1"),
    ("IBGE_MUNICIPIOS", "IBGE limites municipais", "https://servicodados.ibge.gov.br/api/v3/malhas/municipios/", "servicodados.ibge.gov.br",
     "REST_API", "ALL", "TERRITORIAL", "JSON/GEOJSON", "API_CALL", "false", "true", "P2"),
    ("DEFESA_CIVIL_PE", "Defesa Civil PE/Recife", "", "",
     "UNKNOWN", "RECIFE", "FLOOD", "PDF/HTML", "CONFIG_REQUIRED", "true", "false", "P0"),
    ("DEFESA_CIVIL_RJ_PET", "Defesa Civil RJ/Petropolis", "", "",
     "UNKNOWN", "PET", "LANDSLIDE", "PDF/HTML", "CONFIG_REQUIRED", "true", "false", "P0"),
    ("DEFESA_CIVIL_PR_CWB", "Defesa Civil PR/Curitiba", "", "",
     "UNKNOWN", "CURITIBA", "FLOOD", "PDF/HTML", "CONFIG_REQUIRED", "true", "false", "P1"),
    ("DIARIO_OFICIAL_PE", "Diario Oficial PE", "", "",
     "UNKNOWN", "RECIFE", "INSTITUTIONAL", "PDF/HTML", "CONFIG_REQUIRED", "true", "false", "P1"),
    ("DIARIO_OFICIAL_RJ", "Diario Oficial RJ", "", "",
     "UNKNOWN", "PET", "INSTITUTIONAL", "PDF/HTML", "CONFIG_REQUIRED", "true", "false", "P1"),
]


def build_rows() -> list[dict[str, Any]]:
    user_config = load_allowed_sources_config(IN_CONFIG)
    user_urls = {r.get("source_name", ""): r.get("base_url", "") for r in user_config}

    rows: list[dict[str, Any]] = []
    for i, seed in enumerate(_SEEDS):
        eid, name, url, domain, etype, region, hazard, ftypes, strategy, manual, enabled, prio = seed
        # override URL from config if available
        if name in user_urls and user_urls[name]:
            url = user_urls[name]
            domain = domain_from_url(url) or domain

        allowed = "true" if (url and is_allowed_domain(url)) else ("false" if url else "NOT_CONFIGURED")
        blocked = ""
        if not url:
            blocked = "ENDPOINT_CONFIG_REQUIRED"
        elif allowed == "false":
            blocked = "ENDPOINT_DOMAIN_NOT_ALLOWED"

        row = {
            "endpoint_id": f"V1SG_EP_{i:03d}",
            "source_name": name, "source_family": classify_source_family(name),
            "base_url": url, "domain": domain,
            "endpoint_type": etype, "region_scope": region, "hazard_scope": hazard,
            "expected_file_types": ftypes, "download_strategy": strategy,
            "requires_manual_url": manual, "allowed_domain": allowed,
            "robots_status": "NOT_CHECKED", "rate_limit_seconds": "2",
            "enabled_by_default": enabled, "priority": prio,
            "review_only": "true", "blocked_reason": blocked, "notes": "",
        }
        rows.append(row)
    return rows


def run(datasets: Path | None = None) -> dict[str, Any]:
    rows = build_rows()
    write_csv_with_header(OUT_REGISTRY, rows, REG_FIELDS)
    write_schema_for(SCHEMA_REG, REG_FIELDS, "v1sg_endpoint_registry")

    ready = sum(1 for r in rows if not r["blocked_reason"])
    config_req = sum(1 for r in rows if r["blocked_reason"] == "ENDPOINT_CONFIG_REQUIRED")
    summary = [
        {"stat_key": "endpoints_total", "stat_value": str(len(rows))},
        {"stat_key": "endpoints_ready", "stat_value": str(ready)},
        {"stat_key": "endpoints_config_required", "stat_value": str(config_req)},
        {"stat_key": "stage", "stat_value": "v1sg"},
    ]
    write_csv_with_header(OUT_SUMMARY, summary, SUM_FIELDS)
    write_schema_for(SCHEMA_SUM, SUM_FIELDS, "v1sg_summary")

    write_doc(DOC, "v1sg — Official Source Endpoint Registry", [
        "## Objetivo",
        "Registrar endpoints oficiais e candidatos a download. Nao baixa ainda.",
        "## Resultado",
        f"Total: {len(rows)}. Ready: {ready}. Config required: {config_req}.",
    ])
    print(f"[v1sg] endpoints={len(rows)} ready={ready} config_req={config_req}")
    return {"endpoints": len(rows), "ready": ready, "config_req": config_req}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1sg endpoint registry").parse_args()
    run()
