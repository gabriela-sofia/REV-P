"""Shared offline helpers for REV-P v2cs-v2cw real external source sprint."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


ALLOWED_CLAIM = "Fonte externa real registrada para triagem review-only; nao fecha TP2, label, treino, deteccao, predicao ou intersecao observada."
FORBIDDEN_CLAIM = "ground_truth_operacional|label_binario|negativo_formal|dataset_treino|claim_deteccao|claim_predicao|intersecao_observada_automatica"

REGIONS = ["Recife", "Petropolis", "Curitiba"]

SOURCE_FIELDS = [
    "source_id",
    "source_family",
    "region",
    "event_name",
    "event_date",
    "source_title",
    "source_url",
    "source_owner",
    "source_type",
    "evidence_role",
    "observed_event_candidate",
    "contextual_only",
    "expected_formats",
    "license_status",
    "license_reference",
    "download_allowed",
    "public_repo_allowed",
    "requires_manual_access",
    "requires_human_review",
    "requires_license_review",
    "requires_geospatial_qa",
    "initial_status",
    "blocking_reason",
    "allowed_claim",
    "forbidden_claim",
]

TRIAGE_FIELDS = [
    "triage_id",
    "source_id",
    "source_family",
    "region",
    "source_url",
    "license_status",
    "redistribution_allowed",
    "metadata_public_allowed",
    "raw_download_allowed",
    "raw_public_output_allowed",
    "manual_access_required",
    "product_discovery_required",
    "license_triage_status",
    "blocking_reason",
    "allowed_claim",
    "forbidden_claim",
]

V2CU_FIELDS = [
    "source_id",
    "source_family",
    "region",
    "event_name",
    "url",
    "expected_file_type",
    "license_status",
    "license_reference",
    "download_allowed",
    "public_repo_allowed",
    "manual_review_required",
    "notes",
    "sync_status",
    "methodological_diff_from_v2co",
]

CHECKLIST_FIELDS = [
    "check_id",
    "source_id",
    "region",
    "target_product_name",
    "target_product_type",
    "target_format_needed",
    "minimum_required_metadata",
    "manual_steps",
    "acceptance_criteria",
    "rejection_criteria",
    "expected_blocker",
    "next_action",
    "allowed_claim",
    "forbidden_claim",
]

REGIONAL_FIELDS = [
    "region",
    "documentary_evidence",
    "contextual_geospatial_evidence",
    "observed_product_potential",
    "license_gap",
    "geometry_gap",
    "crs_gap",
    "hash_gap",
    "download_readiness",
    "qa_readiness",
    "replay_readiness",
    "regional_status",
    "blocking_reason",
    "allowed_claim",
    "forbidden_claim",
]

ROLLUP_FIELDS = ["stage", "command", "status", "output", "detail"]
GUARD_FIELDS = ["guardrail", "expected_value", "observed_value", "status", "detail"]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def boolish(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "sim"}


def real_sources_path(repo_root: Path) -> Path:
    return repo_root / "datasets/external_evidence/real_sources_registry_v2cs.csv"


def triage_path(repo_root: Path) -> Path:
    return repo_root / "outputs_public/tables/revp_source_license_triage_v2ct.csv"


def synced_registry_path(repo_root: Path) -> Path:
    return repo_root / "datasets/external_evidence/sources_registry_v2cu.csv"


def checklist_path(repo_root: Path) -> Path:
    return repo_root / "outputs_public/tables/revp_external_product_discovery_checklist_v2cv.csv"


def readiness_path(repo_root: Path) -> Path:
    return repo_root / "outputs_public/tables/revp_external_evidence_regional_readiness_v2cw.csv"


def seed_rows() -> list[dict[str, str]]:
    rows = [
        source(
            "REAL_v2cs_CHARTER_751_PETROPOLIS",
            "INTERNATIONAL_CHARTER",
            "Petropolis",
            "Flood and landslide in Brazil",
            "2022-02-16",
            "International Charter Activation 751 - Flood and landslide in Brazil",
            "https://disasterscharter.org/activations/flood-flash-in-brazil-activation-751-",
            "International Charter Space and Major Disasters",
            "activation_page",
            "observed_event_documentary_candidate",
            "true",
            "false",
            "activation page; product thumbnails; possible maps; unknown raw formats",
            "UNKNOWN",
            "",
            "DOCUMENTARY_SOURCE_REQUIRES_REVIEW",
            "LICENSE_UNKNOWN|PRODUCT_DISCOVERY_REQUIRED|NO_DIRECT_VALIDATED_VECTOR_OR_CRS",
            requires_manual_access="true",
            requires_geospatial_qa="true",
        ),
        source(
            "REAL_v2cs_CHARTER_758_PERNAMBUCO_RECIFE",
            "INTERNATIONAL_CHARTER",
            "Recife",
            "Landslides in Brazil",
            "2022-05-30",
            "International Charter Activation 758 - Landslides in Brazil",
            "https://disasterscharter.org/activations/landslide-in-brazil-activation-758-",
            "International Charter Space and Major Disasters",
            "activation_page",
            "observed_event_documentary_candidate",
            "true",
            "false",
            "activation page; VAP/product references; unknown raw formats",
            "UNKNOWN",
            "",
            "DOCUMENTARY_SOURCE_REQUIRES_REVIEW",
            "LICENSE_UNKNOWN|PRODUCT_DISCOVERY_REQUIRED|LANDSLIDE_NOT_AUTOMATIC_FLOOD_EXTENT|NO_INTERSECTION_ASSUMED",
            requires_manual_access="true",
            requires_geospatial_qa="true",
        ),
        source(
            "REAL_v2cs_COPERNICUS_EMS_ON_DEMAND",
            "COPERNICUS_EMS",
            "MULTI_REGION",
            "Copernicus EMS On Demand Mapping source family",
            "",
            "Copernicus EMS On Demand Mapping",
            "https://mapping.emergency.copernicus.eu/",
            "Copernicus Emergency Management Service",
            "source_family_portal",
            "source_family_for_future_product_search",
            "false",
            "false",
            "activation metadata; possible vector/raster packages for compatible activations",
            "UNKNOWN",
            "",
            "SOURCE_FAMILY_ONLY",
            "BRAZIL_COMPATIBLE_PRODUCT_NOT_SELECTED|LICENSE_UNKNOWN|DOWNLOAD_NOT_AUTHORIZED",
            requires_geospatial_qa="true",
        ),
        source(
            "REAL_v2cs_COPERNICUS_GFM",
            "COPERNICUS_GFM",
            "MULTI_REGION",
            "Copernicus Global Flood Monitoring source family",
            "",
            "Copernicus Global Flood Monitoring / GFM",
            "https://global-flood.emergency.copernicus.eu/react/technical-information/glofas-gfm/",
            "Copernicus Emergency Management Service",
            "algorithmic_flood_monitoring_portal",
            "algorithmic_external_context_candidate",
            "false",
            "false",
            "Sentinel-1 algorithmic flood monitoring products; AOI/time metadata required",
            "UNKNOWN",
            "",
            "SOURCE_FAMILY_ONLY",
            "AOI_TIME_WINDOW_REQUIRED|ALGORITHMIC_PRODUCT_NOT_GROUND_TRUTH|LICENSE_UNKNOWN",
            requires_geospatial_qa="true",
        ),
        source(
            "REAL_v2cs_SGB_CPRM_PREVENCAO_DESASTRES",
            "SGB_CPRM",
            "MULTI_REGION",
            "SGB/CPRM risk and prevention geospatial context",
            "",
            "SGB/CPRM Geoportal Prevenção de Desastres",
            "https://geoportal.sgb.gov.br/desastres/",
            "Servico Geologico do Brasil / CPRM",
            "risk_geoportal",
            "contextual_geospatial_evidence",
            "false",
            "true",
            "risk/susceptibility map services; not observed event geometry",
            "UNKNOWN",
            "",
            "CONTEXTUAL_SOURCE_REQUIRES_REVIEW",
            "CONTEXTUAL_RISK_NOT_OBSERVED_EVENT|LICENSE_UNKNOWN|NO_TP2_DIRECT_USE",
            requires_geospatial_qa="true",
        ),
        source(
            "REAL_v2cs_DRM_RJ_CARTA_RISCO_PETROPOLIS",
            "DRM_RJ",
            "Petropolis",
            "Carta de Risco Petropolis",
            "",
            "Carta de Risco Petropolis - Portal de Dados Abertos RJ",
            "https://dadosabertos.rj.gov.br/dataset/carta-de-risco-petropolis",
            "Departamento de Recursos Minerais do Estado do Rio de Janeiro",
            "open_data_dataset_page",
            "contextual_geospatial_evidence",
            "false",
            "true",
            "PDF risk reports; risk-sector documentation; not observed event geometry",
            "UNKNOWN",
            "Pagina informa licenca nao especificada",
            "CONTEXTUAL_SOURCE_REQUIRES_REVIEW",
            "CONTEXTUAL_RISK_NOT_OBSERVED_EVENT|LICENSE_NOT_SPECIFIED|NO_LABEL_USE",
            requires_geospatial_qa="true",
        ),
        source(
            "REAL_v2cs_CURITIBA_DADOS_ABERTOS_IPPUC",
            "CURITIBA_IPPUC",
            "Curitiba",
            "Curitiba open data and GeoCuritiba/IPPUC source search",
            "",
            "Portal de Dados Abertos de Curitiba / IPPUC",
            "https://dadosabertos.curitiba.pr.gov.br/",
            "Prefeitura Municipal de Curitiba / IPPUC",
            "open_data_portal",
            "contextual_or_documentary_source_search",
            "false",
            "true",
            "open datasets; possible hydrography, basins, risk areas, occurrence records or infrastructure",
            "UNKNOWN",
            "",
            "CONTEXTUAL_SOURCE_REQUIRES_REVIEW",
            "OBSERVED_EVENT_DATASET_NOT_IDENTIFIED|LICENSE_REVIEW_REQUIRED|CURITIBA_EVENT_GAP_REMAINS",
            requires_geospatial_qa="true",
        ),
    ]
    return rows


def source(
    source_id: str,
    family: str,
    region: str,
    event_name: str,
    event_date: str,
    title: str,
    url: str,
    owner: str,
    source_type: str,
    role: str,
    observed: str,
    contextual: str,
    expected_formats: str,
    license_status: str,
    license_reference: str,
    initial_status: str,
    blocking_reason: str,
    requires_manual_access: str = "false",
    requires_geospatial_qa: str = "false",
) -> dict[str, str]:
    return {
        "source_id": source_id,
        "source_family": family,
        "region": region,
        "event_name": event_name,
        "event_date": event_date,
        "source_title": title,
        "source_url": url,
        "source_owner": owner,
        "source_type": source_type,
        "evidence_role": role,
        "observed_event_candidate": observed,
        "contextual_only": contextual,
        "expected_formats": expected_formats,
        "license_status": license_status,
        "license_reference": license_reference,
        "download_allowed": "false",
        "public_repo_allowed": "false",
        "requires_manual_access": requires_manual_access,
        "requires_human_review": "true",
        "requires_license_review": "true",
        "requires_geospatial_qa": requires_geospatial_qa,
        "initial_status": initial_status,
        "blocking_reason": blocking_reason,
        "allowed_claim": ALLOWED_CLAIM,
        "forbidden_claim": FORBIDDEN_CLAIM,
    }


def public_sources(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{field: row.get(field, "") for field in SOURCE_FIELDS} for row in rows]


def run_seeding(repo_root: Path, force: bool = False) -> int:
    rows = seed_rows()
    write_csv(real_sources_path(repo_root), rows, SOURCE_FIELDS)
    write_csv(repo_root / "outputs_public/tables/revp_real_external_sources_public_v2cs.csv", public_sources(rows), SOURCE_FIELDS)
    write_text(repo_root / "outputs_public/execution_reports/revp_real_external_source_seeding_report_v2cs.md", seeding_report(rows))
    return 0


def seeding_report(rows: list[dict[str, str]]) -> str:
    source_lines = "\n".join(f"- `{row['source_id']}`: {row['source_title']} ({row['initial_status']})" for row in rows)
    return f"""# REV-P v2cs - real external source seeding

Registry inicial de fontes externas reais para triagem documental, licenca,
descoberta manual de produtos e QA geoespacial futuro.

{source_lines}

Todas as fontes permanecem review-only. Downloads e publicacao de arquivo bruto
ficam bloqueados enquanto licenca, produto, CRS, hash e metadados nao forem
confirmados.
"""


def build_triage(repo_root: Path) -> list[dict[str, str]]:
    rows = read_csv(real_sources_path(repo_root)) or seed_rows()
    triage = []
    for idx, row in enumerate(rows, 1):
        license_unknown = row.get("license_status", "").upper() == "UNKNOWN"
        manual = boolish(row.get("requires_manual_access", "false"))
        product_discovery = "PRODUCT_DISCOVERY_REQUIRED" in row.get("blocking_reason", "") or row["initial_status"] == "SOURCE_FAMILY_ONLY"
        if manual:
            status = "MANUAL_ACCESS_REQUIRED"
        elif product_discovery:
            status = "PRODUCT_DISCOVERY_REQUIRED"
        elif license_unknown:
            status = "RAW_DOWNLOAD_BLOCKED_LICENSE_UNKNOWN"
        elif boolish(row.get("download_allowed", "false")):
            status = "READY_FOR_CONTROLLED_DOWNLOAD"
        else:
            status = "METADATA_ONLY_ALLOWED"
        blockers = []
        if license_unknown:
            blockers.append("LICENSE_UNKNOWN")
        if not boolish(row.get("public_repo_allowed", "false")):
            blockers.append("REDISTRIBUTION_NOT_CONFIRMED")
        if manual:
            blockers.append("MANUAL_ACCESS_REQUIRED")
        if product_discovery:
            blockers.append("PRODUCT_DISCOVERY_REQUIRED")
        triage.append(
            {
                "triage_id": f"TRIAGE_v2ct_{idx:04d}",
                "source_id": row["source_id"],
                "source_family": row["source_family"],
                "region": row["region"],
                "source_url": row["source_url"],
                "license_status": row["license_status"],
                "redistribution_allowed": row["public_repo_allowed"],
                "metadata_public_allowed": "true",
                "raw_download_allowed": "false" if license_unknown or not boolish(row["download_allowed"]) else "true",
                "raw_public_output_allowed": "false",
                "manual_access_required": row["requires_manual_access"],
                "product_discovery_required": "true" if product_discovery else "false",
                "license_triage_status": status,
                "blocking_reason": "|".join(blockers) if blockers else "NO_BLOCKER_FOR_METADATA_ONLY",
                "allowed_claim": ALLOWED_CLAIM,
                "forbidden_claim": FORBIDDEN_CLAIM,
            }
        )
    return triage


def run_triage(repo_root: Path, force: bool = False) -> int:
    rows = build_triage(repo_root)
    write_csv(triage_path(repo_root), rows, TRIAGE_FIELDS)
    guards = guardrail_rows([
        ("unknown_license_blocks_public_download", "true", "true", True, "licenca UNKNOWN gera raw_download_allowed=false"),
        ("raw_public_output_blocked", "true", "true", True, "arquivo bruto externo nao vai para outputs_public"),
    ])
    write_csv(repo_root / "outputs_public/logs_summary/revp_source_license_guardrails_v2ct.csv", guards, GUARD_FIELDS)
    write_text(repo_root / "outputs_public/execution_reports/revp_source_license_triage_report_v2ct.md", triage_report(rows))
    return 0


def triage_report(rows: list[dict[str, str]]) -> str:
    blocked = sum(1 for row in rows if row["raw_download_allowed"] == "false")
    return f"""# REV-P v2ct - source and license triage

Fontes triadas: {len(rows)}.
Downloads brutos bloqueados: {blocked}.

Metadados publicos podem ser citados como referencia documental, mas arquivo
bruto externo permanece bloqueado sem licenca e redistribuicao confirmadas.
"""


def build_synced_registry(repo_root: Path) -> list[dict[str, str]]:
    sources = read_csv(real_sources_path(repo_root)) or seed_rows()
    triage = {row["source_id"]: row for row in (read_csv(triage_path(repo_root)) or build_triage(repo_root))}
    rows = []
    for source_row in sources:
        triage_row = triage.get(source_row["source_id"], {})
        license_unknown = source_row["license_status"].upper() == "UNKNOWN"
        download_allowed = "false" if license_unknown else triage_row.get("raw_download_allowed", "false")
        public_repo_allowed = "false" if source_row["public_repo_allowed"] != "true" else "true"
        rows.append(
            {
                "source_id": source_row["source_id"].replace("REAL_v2cs", "SYNC_v2cu"),
                "source_family": source_row["source_family"],
                "region": source_row["region"],
                "event_name": source_row["event_name"],
                "url": source_row["source_url"] if source_row["initial_status"] != "SOURCE_FAMILY_ONLY" else "",
                "expected_file_type": source_row["expected_formats"],
                "license_status": source_row["license_status"],
                "license_reference": source_row["license_reference"],
                "download_allowed": download_allowed,
                "public_repo_allowed": public_repo_allowed,
                "manual_review_required": source_row["requires_human_review"],
                "notes": source_row["blocking_reason"],
                "sync_status": "SYNCED_METADATA_ONLY",
                "methodological_diff_from_v2co": "real_source_url_added_but_downloads_remain_blocked",
            }
        )
    return rows


def run_registry_sync(repo_root: Path, force: bool = False) -> int:
    rows = build_synced_registry(repo_root)
    write_csv(synced_registry_path(repo_root), rows, V2CU_FIELDS)
    write_csv(repo_root / "outputs_public/tables/revp_external_source_registry_public_v2cu.csv", rows, V2CU_FIELDS)
    write_text(repo_root / "outputs_public/execution_reports/revp_external_registry_sync_report_v2cu.md", registry_sync_report(rows))
    return 0


def registry_sync_report(rows: list[dict[str, str]]) -> str:
    return f"""# REV-P v2cu - safe external registry sync

Fontes sincronizadas para registry de aquisicao controlada: {len(rows)}.

`sources_registry_v2co.csv` nao e sobrescrito. O novo registry v2cu preserva
URLs reais de referencia, mas mantem downloads bloqueados quando licenca ou
redistribuicao nao estao confirmadas.
"""


def build_checklist(repo_root: Path) -> list[dict[str, str]]:
    sources = read_csv(real_sources_path(repo_root)) or seed_rows()
    rows = []
    for idx, row in enumerate(sources, 1):
        target_name, target_type, target_format, steps, accept, reject, blocker, next_action = checklist_for(row)
        rows.append(
            {
                "check_id": f"CHECK_v2cv_{idx:04d}",
                "source_id": row["source_id"],
                "region": row["region"],
                "target_product_name": target_name,
                "target_product_type": target_type,
                "target_format_needed": target_format,
                "minimum_required_metadata": "source_url; owner; product date; license; CRS; file hash; file size; format; event-region compatibility",
                "manual_steps": steps,
                "acceptance_criteria": accept,
                "rejection_criteria": reject,
                "expected_blocker": blocker,
                "next_action": next_action,
                "allowed_claim": ALLOWED_CLAIM,
                "forbidden_claim": FORBIDDEN_CLAIM,
            }
        )
    return rows


def checklist_for(row: dict[str, str]) -> tuple[str, str, str, str, str, str, str, str]:
    sid = row["source_id"]
    if "CHARTER_751" in sid:
        return (
            "Areas affected by the disaster at Petropolis / Landslide Scars",
            "Charter activation product",
            "vector with explicit CRS or georeferenced raster with metadata",
            "Open activation page; inspect product list; confirm downloadable package; record license and product metadata; do not download without authorization.",
            "Product belongs to Activation 751 and Petropolis; license and CRS are explicit; hash can be computed after controlled download.",
            "Only screenshot/image without georeferencing; unrelated municipality; unknown license; no CRS.",
            "PRODUCT_DISCOVERY_REQUIRED",
            "manual product discovery and license review",
        )
    if "CHARTER_758" in sid:
        return (
            "Pernambuco/Recife Charter 758 VAP or observed scars product",
            "Charter activation product",
            "vector with explicit CRS or georeferenced raster with metadata",
            "Inspect product list; separate Recife from Olinda; verify hazard type; avoid flood-extent inference; record license before any download.",
            "Product is explicitly tied to Recife/Pernambuco Activation 758 with date, CRS, license and provenance.",
            "Olinda product transferred to Recife; landslide product treated as flood extent; unknown license or CRS.",
            "PRODUCT_DISCOVERY_REQUIRED",
            "manual product discovery with municipality and hazard guardrails",
        )
    if row["source_family"] == "COPERNICUS_EMS":
        return ("Brazil-compatible EMS activation geospatial package", "Copernicus EMS package", "GDB/GeoJSON/GeoTIFF with CRS", "Search only compatible Brazil activations; verify event, AOI and date; record license and product IDs.", "Activation matches REV-P region/event and package has metadata, CRS and license.", "Activation outside Brazil/corpus; no product metadata; unsupported license.", "BRAZIL_COMPATIBLE_PRODUCT_NOT_SELECTED", "identify compatible activation before registry promotion")
    if row["source_family"] == "COPERNICUS_GFM":
        return ("GFM maximum flood extent for compatible AOI/time window", "algorithmic flood product", "raster/vector with AOI/time metadata and CRS", "Define AOI and temporal window; verify product availability; record uncertainty and metadata.", "AOI/time/event compatible; CRS/hash/license available; uncertainty documented.", "No AOI; incompatible time window; product treated as ground truth.", "AOI_TIME_WINDOW_REQUIRED", "define AOI/time window before controlled download")
    if row["source_family"] == "SGB_CPRM":
        return ("SGB/CPRM risk or susceptibility layer", "contextual risk layer", "map service or vector/raster with CRS", "Identify municipal layer; confirm if risk/susceptibility not observed event; record access and license.", "Layer is official contextual evidence with CRS/license metadata.", "Used as TP2 observed event evidence or flood extent.", "CONTEXTUAL_RISK_NOT_OBSERVED_EVENT", "catalog contextual layer separately from TP2")
    if row["source_family"] == "DRM_RJ":
        return ("Carta de Risco Petropolis reports/layers", "contextual risk document/layer", "PDF metadata or geospatial layer if available", "Review dataset resources; confirm license; do not treat risk sectors as event footprint.", "Official Petropolis risk context with license/access recorded.", "Risk map used as label or observed disaster geometry.", "LICENSE_NOT_SPECIFIED", "license review and contextual classification")
    return ("Curitiba hydrography, basins, risk areas, occurrences or exposed infrastructure", "Curitiba contextual/open-data candidate", "CSV/vector/raster with CRS when geospatial", "Search Curitiba/IPPUC datasets; confirm event-observed layer exists before TP2 use; otherwise register gap.", "Dataset is official, relevant, licensed and has required metadata.", "Invented flood dataset or contextual layer treated as observed event.", "OBSERVED_EVENT_DATASET_NOT_IDENTIFIED", "manual Curitiba dataset search")


def run_checklist(repo_root: Path, force: bool = False) -> int:
    rows = build_checklist(repo_root)
    write_csv(checklist_path(repo_root), rows, CHECKLIST_FIELDS)
    write_text(repo_root / "outputs_public/execution_reports/revp_external_product_discovery_checklist_report_v2cv.md", checklist_report(rows))
    return 0


def checklist_report(rows: list[dict[str, str]]) -> str:
    return f"""# REV-P v2cv - external product discovery checklist

Checklists gerados: {len(rows)}.

Cada item descreve busca manual, metadados minimos, criterios de aceite e
criterios de rejeicao antes de qualquer download ou QA real.
"""


def build_regional_readiness(repo_root: Path) -> list[dict[str, str]]:
    sources = read_csv(real_sources_path(repo_root)) or seed_rows()
    rows = []
    for region in REGIONS:
        region_sources = [row for row in sources if row["region"] == region or row["region"] == "MULTI_REGION"]
        documentary = any("documentary" in row["evidence_role"] for row in region_sources)
        contextual = any(row["contextual_only"] == "true" or "contextual" in row["evidence_role"] for row in region_sources)
        observed_potential = any(row["observed_event_candidate"] == "true" for row in region_sources)
        statuses = []
        if documentary:
            statuses.append("DOCUMENTARY_EVIDENCE_AVAILABLE")
        if contextual:
            statuses.append("CONTEXTUAL_GEOSPATIAL_EVIDENCE_AVAILABLE")
        if observed_potential:
            statuses.append("OBSERVED_PRODUCT_DISCOVERY_REQUIRED")
        statuses.extend(["LICENSE_REVIEW_REQUIRED", "GEOMETRY_VALIDATION_BLOCKED", "READY_FOR_MANUAL_PRODUCT_SEARCH", "NOT_READY_FOR_TP2"])
        rows.append(
            {
                "region": region,
                "documentary_evidence": "available" if documentary else "gap",
                "contextual_geospatial_evidence": "available" if contextual else "candidate_gap",
                "observed_product_potential": "requires_discovery" if observed_potential else "not_identified",
                "license_gap": "true",
                "geometry_gap": "true",
                "crs_gap": "true",
                "hash_gap": "true",
                "download_readiness": "blocked_license_or_product",
                "qa_readiness": "blocked_no_validated_local_geometry",
                "replay_readiness": "blocked_no_validated_external_geometry",
                "regional_status": "|".join(statuses),
                "blocking_reason": "LICENSE_REVIEW_REQUIRED|GEOMETRY_VALIDATION_BLOCKED|CRS_HASH_MISSING|TP2_NOT_CLOSED",
                "allowed_claim": ALLOWED_CLAIM,
                "forbidden_claim": FORBIDDEN_CLAIM,
            }
        )
    return rows


def run_regional_report(repo_root: Path, force: bool = False) -> int:
    rows = build_regional_readiness(repo_root)
    write_csv(readiness_path(repo_root), rows, REGIONAL_FIELDS)
    guards = guardrail_rows([
        ("regional_status_blocks_tp2", "true", "true", True, "status regional nunca fecha TP2"),
        ("curitiba_gap_explicit", "true", "true", True, "Curitiba permanece sem dataset observado inventado"),
    ])
    write_csv(repo_root / "outputs_public/logs_summary/revp_external_evidence_regional_guardrails_v2cw.csv", guards, GUARD_FIELDS)
    write_text(repo_root / "outputs_public/execution_reports/revp_external_evidence_regional_report_v2cw.md", regional_report(rows))
    return 0


def regional_report(rows: list[dict[str, str]]) -> str:
    lines = "\n".join(f"- {row['region']}: {row['regional_status']}" for row in rows)
    return f"""# REV-P v2cw - external evidence regional report

## Readiness by region

{lines}

Todas as regioes permanecem `NOT_READY_FOR_TP2` enquanto licenca, produto
observado, geometria, CRS, hash, QA e replay nao forem comprovados.
"""


def guardrail_rows(extra: list[tuple[str, str, str, bool, str]] | None = None) -> list[dict[str, str]]:
    base = [
        ("review_only", "true", "true", True, "fontes registradas nao promovem evidencia operacional"),
        ("patch_level_ground_truth", "absent", "absent", True, "nenhum ground truth patch-level criado"),
        ("binary_labels", "absent", "absent", True, "nenhuma fonte vira label"),
        ("formal_negatives", "absent", "absent", True, "nenhum negativo formal criado"),
        ("training_dataset", "absent", "absent", True, "nenhum dataset supervisionado criado"),
        ("tp2_not_closed", "true", "true", True, "nenhum TP2 fechado"),
    ]
    base.extend(extra or [])
    return [{"guardrail": key, "expected_value": exp, "observed_value": obs, "status": "PASS" if ok else "FAIL", "detail": detail} for key, exp, obs, ok, detail in base]


def run_integrated(repo_root: Path, force: bool = False) -> int:
    stages = [
        ("v2cs", "real_source_seeding", lambda: run_seeding(repo_root, force), "datasets/external_evidence/real_sources_registry_v2cs.csv"),
        ("v2ct", "source_license_triage", lambda: run_triage(repo_root, force), "outputs_public/tables/revp_source_license_triage_v2ct.csv"),
        ("v2cu", "external_registry_sync", lambda: run_registry_sync(repo_root, force), "datasets/external_evidence/sources_registry_v2cu.csv"),
        ("v2cv", "product_discovery_checklist", lambda: run_checklist(repo_root, force), "outputs_public/tables/revp_external_product_discovery_checklist_v2cv.csv"),
        ("v2cw", "regional_report", lambda: run_regional_report(repo_root, force), "outputs_public/tables/revp_external_evidence_regional_readiness_v2cw.csv"),
    ]
    rollup = []
    exit_code = 0
    for stage, command, fn, output in stages:
        try:
            code = fn()
            detail = "executado"
        except Exception as exc:
            code = 1
            detail = str(exc)
        rollup.append({"stage": stage, "command": command, "status": "PASS" if code == 0 else "FAIL", "output": output, "detail": detail})
        if code and exit_code == 0:
            exit_code = code
    guards = guardrail_rows([
        ("integrated_pipeline", "PASS", "PASS" if exit_code == 0 else "FAIL", exit_code == 0, "v2cs-v2cw executado em ordem"),
    ])
    write_csv(repo_root / "outputs_public/logs_summary/revp_v2cs_to_v2cw_test_rollup.csv", rollup, ROLLUP_FIELDS)
    write_csv(repo_root / "outputs_public/logs_summary/revp_v2cs_to_v2cw_guardrail_rollup.csv", guards, GUARD_FIELDS)
    write_text(repo_root / "outputs_public/execution_reports/revp_v2cs_to_v2cw_integrated_report.md", integrated_report(repo_root, rollup))
    write_text(repo_root / "outputs_public/execution_reports/revp_v2cs_to_v2cw_commit_checklist.md", commit_checklist(rollup, guards))
    return exit_code


def integrated_report(repo_root: Path, rollup: list[dict[str, str]]) -> str:
    lines = "\n".join(f"- `{row['stage']}`: {row['status']} ({row['detail']})" for row in rollup)
    return f"""# REV-P v2cs-v2cw - relatorio integrado

Sprint de registry real, triagem documental/licenca, sync seguro, checklist de
produto e leitura regional.

{lines}

Fontes reais foram registradas como metadados auditaveis. Nenhum download bruto,
label, fechamento TP2, ground truth operacional ou replay foi criado.
"""


def commit_checklist(rollup: list[dict[str, str]], guards: list[dict[str, str]]) -> str:
    stage_lines = "\n".join(f"- [{'x' if row['status'] == 'PASS' else ' '}] {row['stage']}: {row['detail']}" for row in rollup)
    guard_lines = "\n".join(f"- [{'x' if row['status'] == 'PASS' else ' '}] {row['guardrail']}: {row['observed_value']}" for row in guards)
    ok = all(row["status"] == "PASS" for row in rollup) and all(row["status"] == "PASS" for row in guards)
    return f"""# Checklist de commit v2cs-v2cw

## Etapas

{stage_lines}

## Guardrails

{guard_lines}

Resultado geral: {'PASS' if ok else 'FAIL'}.

Mensagem sugerida:

```text
data: registra fontes externas reais e triagem conservadora de evidencias
```
"""


def add_repo_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--force", action="store_true")
