"""REV-P v1qu — Official external evidence source requirement registry.

Builds the registry of external source requirements per region / hazard /
evidence need. Marks sources as SOURCE_REQUIRED_NOT_LOCAL when they are not
present locally — never invents that a source exists. Review-only; no labels,
no targets, no operational ground truth.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from revp_v1lj_v1lq_common import DATASETS, DOCS, SCHEMAS
from revp_v1qu_v1qz_ground_reference_common import (
    OFFICIAL_CIVIL_DEFENSE,
    OFFICIAL_GEOLOGICAL,
    OFFICIAL_GOVERNMENT_PUBLICATION,
    OFFICIAL_HYDROMETEOROLOGICAL,
    SCIENTIFIC_DATASET,
    _p,
    assert_clean_rows,
    classify_source_family,
    guardrail_row,
    load_existing_protocol_c_context,
    normalize_region,
    write_csv_with_header,
    write_doc,
    write_schema_safe,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_REQUIREMENTS = _p("REVP_V1QU_OUT_REQUIREMENTS", DATASETS / "protocol_c_official_evidence_source_requirements_v1qu.csv")
OUT_GAP = _p("REVP_V1QU_OUT_GAP", DATASETS / "protocol_c_official_evidence_source_gap_summary_v1qu.csv")
SCHEMA_REQUIREMENTS = _p("REVP_V1QU_SCHEMA_REQUIREMENTS", SCHEMAS / "protocol_c_official_evidence_source_requirements_v1qu_schema.csv")
SCHEMA_GAP = _p("REVP_V1QU_SCHEMA_GAP", SCHEMAS / "protocol_c_official_evidence_source_gap_summary_v1qu_schema.csv")
DOC = _p("REVP_V1QU_DOC", DOCS / "revp_v1qu_official_evidence_source_requirement_registry.md")

REQUIREMENT_FIELDS = [
    "requirement_id",
    "region",
    "hazard_type",
    "evidence_need",
    "preferred_source_family",
    "preferred_source_name",
    "source_priority",
    "expected_evidence_type",
    "temporal_precision_needed",
    "spatial_precision_needed",
    "license_or_access_note",
    "collection_action",
    "collection_status",
    "blocks_c3",
    "blocks_c4",
    "review_only",
    "can_create_operational_label",
    "can_train_model",
    "target_created",
    "ground_truth_operational",
    "notes",
]

GAP_FIELDS = ["stat_key", "stat_value"]

# (region, hazard_type, evidence_need, source_family, source_name, priority,
#  expected_evidence_type, temporal_needed, spatial_needed, blocks_c3, blocks_c4)
_REQUIREMENT_SEEDS: list[tuple[str, ...]] = [
    # --- Recife: chuva / nivel / ocorrencia / Diario Oficial / Defesa Civil ---
    ("RECIFE", "FLOOD", "rainfall_intensity", OFFICIAL_HYDROMETEOROLOGICAL, "INMET / BDMEP", "P0", "rain_gauge_series", "DAY", "ADMINISTRATIVE", "true", "false"),
    ("RECIFE", "FLOOD", "rainfall_alert", OFFICIAL_HYDROMETEOROLOGICAL, "CEMADEN", "P0", "hydromet_alert", "DAY", "ADMINISTRATIVE", "true", "false"),
    ("RECIFE", "FLOOD", "river_level_discharge", OFFICIAL_HYDROMETEOROLOGICAL, "ANA / HidroWeb", "P1", "station_level_series", "DAY", "POINT", "true", "false"),
    ("RECIFE", "FLOOD", "field_occurrence_record", OFFICIAL_CIVIL_DEFENSE, "Defesa Civil Recife (SEDEC)", "P0", "occurrence_bulletin", "DAY", "ADDRESS", "true", "false"),
    ("RECIFE", "FLOOD", "institutional_recognition", OFFICIAL_GOVERNMENT_PUBLICATION, "Diario Oficial (situacao de emergencia)", "P1", "decree_recognition", "DAY", "ADMINISTRATIVE", "false", "false"),
    # --- Petropolis: Defesa Civil / SGB-CPRM / CEMADEN / INMET / Diario Oficial ---
    ("PET", "LANDSLIDE", "field_occurrence_record", OFFICIAL_CIVIL_DEFENSE, "Defesa Civil Petropolis (COMPDEC)", "P0", "occurrence_bulletin", "DAY", "ADDRESS", "true", "false"),
    ("PET", "LANDSLIDE", "mass_movement_mapping", OFFICIAL_GEOLOGICAL, "SGB / CPRM", "P0", "technical_post_disaster_report", "DAY", "POINT", "true", "false"),
    ("PET", "LANDSLIDE", "rainfall_alert", OFFICIAL_HYDROMETEOROLOGICAL, "CEMADEN", "P0", "hydromet_alert", "DAY", "ADMINISTRATIVE", "true", "false"),
    ("PET", "LANDSLIDE", "rainfall_intensity", OFFICIAL_HYDROMETEOROLOGICAL, "INMET / BDMEP", "P1", "rain_gauge_series", "DAY", "ADMINISTRATIVE", "true", "false"),
    ("PET", "LANDSLIDE", "institutional_recognition", OFFICIAL_GOVERNMENT_PUBLICATION, "Diario Oficial (situacao de emergencia)", "P1", "decree_recognition", "DAY", "ADMINISTRATIVE", "false", "false"),
    # --- Curitiba: Defesa Civil / INMET / ANA / CEMADEN / Diario Oficial ---
    ("CURITIBA", "FLOOD", "field_occurrence_record", OFFICIAL_CIVIL_DEFENSE, "Defesa Civil Curitiba", "P0", "occurrence_bulletin", "DAY", "ADDRESS", "true", "false"),
    ("CURITIBA", "FLOOD", "rainfall_intensity", OFFICIAL_HYDROMETEOROLOGICAL, "INMET / BDMEP", "P0", "rain_gauge_series", "DAY", "ADMINISTRATIVE", "true", "false"),
    ("CURITIBA", "FLOOD", "river_level_discharge", OFFICIAL_HYDROMETEOROLOGICAL, "ANA / HidroWeb", "P1", "station_level_series", "DAY", "POINT", "true", "false"),
    ("CURITIBA", "FLOOD", "rainfall_alert", OFFICIAL_HYDROMETEOROLOGICAL, "CEMADEN", "P1", "hydromet_alert", "DAY", "ADMINISTRATIVE", "true", "false"),
    ("CURITIBA", "FLOOD", "institutional_recognition", OFFICIAL_GOVERNMENT_PUBLICATION, "Diario Oficial (situacao de emergencia)", "P2", "decree_recognition", "DAY", "ADMINISTRATIVE", "false", "false"),
    # --- Territorial context (not an event label) ---
    ("RECIFE", "CONTEXT", "territorial_base", SCIENTIFIC_DATASET, "IBGE / MapBiomas (contexto territorial)", "P2", "territorial_layer", "YEAR", "ADMINISTRATIVE", "false", "false"),
    ("PET", "CONTEXT", "territorial_base", SCIENTIFIC_DATASET, "IBGE / MapBiomas (contexto territorial)", "P2", "territorial_layer", "YEAR", "ADMINISTRATIVE", "false", "false"),
    ("CURITIBA", "CONTEXT", "territorial_base", SCIENTIFIC_DATASET, "IBGE / MapBiomas (contexto territorial)", "P2", "territorial_layer", "YEAR", "ADMINISTRATIVE", "false", "false"),
]


def _local_source_present(source_name: str, context: dict[str, list[dict[str, str]]]) -> bool:
    """Heuristic: does any locally-known source candidate match this family?"""
    family = classify_source_family(source_name)
    for row in context.get("source_inventory", []):
        cand = (row.get("candidate_source_name", "") + " " + row.get("candidate_source_type", ""))
        if family != "UNKNOWN_SOURCE" and classify_source_family(cand) == family:
            if str(row.get("is_fixture_or_synthetic", "")).strip().lower() != "true":
                return True
    return False


def build_rows(context: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, seed in enumerate(_REQUIREMENT_SEEDS):
        (region, hazard, need, family, name, prio, etype,
         tneed, sneed, b_c3, b_c4) = seed
        present = _local_source_present(name, context)
        if present:
            collection_status = "SOURCE_PARTIALLY_LOCAL_NEEDS_REVIEW"
            action = "REVIEW_LOCAL_CANDIDATE_THEN_CONFIRM_OFFICIAL"
        else:
            collection_status = "SOURCE_REQUIRED_NOT_LOCAL"
            action = "MANUAL_EXTERNAL_COLLECTION_REQUIRED"
        row = {
            "requirement_id": f"V1QU_REQ_{i:04d}",
            "region": normalize_region(region),
            "hazard_type": hazard,
            "evidence_need": need,
            "preferred_source_family": family,
            "preferred_source_name": name,
            "source_priority": prio,
            "expected_evidence_type": etype,
            "temporal_precision_needed": tneed,
            "spatial_precision_needed": sneed,
            "license_or_access_note": "ACCESS_NOT_VERIFIED_LOCALLY",
            "collection_action": action,
            "collection_status": collection_status,
            "blocks_c3": b_c3,
            "blocks_c4": b_c4,
            "notes": "",
        }
        row.update(guardrail_row())
        rows.append(row)
    return rows


def run(datasets: Path | None = None) -> dict[str, Any]:
    context = load_existing_protocol_c_context(datasets)
    rows = build_rows(context)
    assert_clean_rows(rows, "v1qu_requirements")

    write_csv_with_header(OUT_REQUIREMENTS, rows, REQUIREMENT_FIELDS)
    write_schema_safe(SCHEMA_REQUIREMENTS, REQUIREMENT_FIELDS, "v1qu_source_requirements")

    total = len(rows)
    missing = sum(1 for r in rows if r["collection_status"] == "SOURCE_REQUIRED_NOT_LOCAL")
    partial = sum(1 for r in rows if r["collection_status"] == "SOURCE_PARTIALLY_LOCAL_NEEDS_REVIEW")
    blocks_c3 = sum(1 for r in rows if r["blocks_c3"] == "true")
    by_region: dict[str, int] = {}
    for r in rows:
        by_region[r["region"]] = by_region.get(r["region"], 0) + 1

    gap_rows = [
        {"stat_key": "total_requirements", "stat_value": str(total)},
        {"stat_key": "source_required_not_local", "stat_value": str(missing)},
        {"stat_key": "source_partially_local", "stat_value": str(partial)},
        {"stat_key": "requirements_blocking_c3", "stat_value": str(blocks_c3)},
    ]
    for region, n in sorted(by_region.items()):
        gap_rows.append({"stat_key": f"region_{region.lower()}", "stat_value": str(n)})
    gap_rows.append({"stat_key": "stage", "stat_value": "v1qu"})
    gap_rows.append({"stat_key": "scan_status", "stat_value": "NO_INTERNET_NO_DOWNLOAD"})
    write_csv_with_header(OUT_GAP, gap_rows, GAP_FIELDS)
    write_schema_safe(SCHEMA_GAP, GAP_FIELDS, "v1qu_gap_summary")

    write_doc(
        DOC,
        "v1qu — Official Evidence Source Requirement Registry",
        [
            "## Objetivo",
            "Definir requisitos de fonte externa por regiao, tipo de ameaca e necessidade de "
            "evidencia. Fontes nao presentes localmente sao marcadas SOURCE_REQUIRED_NOT_LOCAL. "
            "Nao inventa que uma fonte existe; nao usa internet; nao baixa dados.",
            "## Familias de fonte priorizadas",
            "CEMADEN, ANA/HidroWeb, INMET/BDMEP (hidrometeorologicas), SGB/CPRM (geologica), "
            "Defesa Civil municipal/estadual, Diario Oficial (publicacao governamental), "
            "IBGE/MapBiomas (contexto territorial, nunca label de evento).",
            "## Resultado",
            f"Total de requisitos: {total}. SOURCE_REQUIRED_NOT_LOCAL: {missing}. "
            f"Parcialmente local: {partial}. Requisitos que bloqueiam C3: {blocks_c3}.",
            "## Guardrails",
            "review_only=true em todas as linhas. Nenhuma fonte midiatica ou social fecha o "
            "gate C3 sozinha. Nenhum requisito cria label, target ou ground truth operacional.",
        ],
    )
    print(f"[v1qu] requirements={total} missing={missing} partial={partial}")
    return {"total": total, "missing": missing, "partial": partial}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1qu source requirement registry").parse_args()
    run()
