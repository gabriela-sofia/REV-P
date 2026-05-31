"""REV-P v1ov — Ground reference observed event registry.

Builds a normalized registry of observed events from v1ou output and
existing datasets. Does not invent events; generates empty registry with
header if no sufficient evidence exists.

can_be_used_as_ground_truth=false, can_train_model=false,
can_create_operational_label=false — always.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from revp_v1lj_v1lq_common import DATASETS, DOCS, SCHEMAS, read_csv
from revp_v1ou_v1pa_common import (
    _p,
    assert_no_forbidden_true,
    classify_evidence_use,
    classify_source_reliability,
    classify_spatial_precision,
    classify_temporal_precision,
    normalize_event_date,
    normalize_region,
    require_no_abs_paths_in_rows,
    write_csv_safe,
    write_doc,
    write_schema_safe,
)

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Output paths (env-overridable)
# ---------------------------------------------------------------------------

OUT_REGISTRY = _p("REVP_V1OV_OUT_REGISTRY", DATASETS / "recife_ground_reference_observed_event_registry_v1ov.csv")
OUT_SUMMARY = _p("REVP_V1OV_OUT_SUMMARY", DATASETS / "recife_ground_reference_observed_event_summary_v1ov.csv")
SCHEMA_REGISTRY = _p("REVP_V1OV_SCHEMA_REGISTRY", SCHEMAS / "recife_ground_reference_observed_event_registry_v1ov_schema.csv")
SCHEMA_SUMMARY = _p("REVP_V1OV_SCHEMA_SUMMARY", SCHEMAS / "recife_ground_reference_observed_event_summary_v1ov_schema.csv")
DOC = _p("REVP_V1OV_DOC", DOCS / "revp_v1ov_ground_reference_observed_event_registry.md")
IN_V1OU = _p("REVP_V1OV_IN_V1OU", DATASETS / "recife_external_evidence_source_inventory_v1ou.csv")

REGISTRY_FIELDS = [
    "event_id",
    "region",
    "event_type",
    "event_date_iso",
    "event_date_status",
    "event_time_precision",
    "location_text",
    "latitude",
    "longitude",
    "spatial_precision_level",
    "source_candidate_id",
    "source_type",
    "source_name",
    "source_reliability_level",
    "observed_event_status",
    "can_be_used_as_ground_truth",
    "can_train_model",
    "can_create_operational_label",
    "allowed_use",
    "blocked_reason",
    "notes",
]

SUMMARY_FIELDS = ["stat_key", "stat_value"]

# Observed event status values
OBS_CONFIRMED = "OBSERVED_EVENT_CONFIRMED_REVIEW_ONLY"
OBS_PROBABLE = "OBSERVED_EVENT_PROBABLE_REVIEW_ONLY"
OBS_CONTEXTUAL = "CONTEXTUAL_EVIDENCE_ONLY"
OBS_BLOCKED = "BLOCKED_INSUFFICIENT_EVIDENCE"
OBS_FIXTURE = "BLOCKED_FIXTURE_OR_SYNTHETIC"

# ---------------------------------------------------------------------------
# Seed events from existing datasets (Recife-focused)
# ---------------------------------------------------------------------------

# These are derived from inspecting the existing registries:
#   - event_evidence_dossier_registry.csv (DOS_REC_2021, DOS_REC_2022)
#   - manual_external_evidence_needed_registry.csv (REC_2022_05_24_30)
#   - observed_event_reference_gap_registry.csv (REC_2022_05_24_30)
# All are in DOSSIER_OPEN / NOT_ACQUIRED status — contextual only.

SEED_RECIFE_EVENTS = [
    {
        "event_id": "EVENT_REC_2021_INUNDACAO_DOSSIER",
        "region": "RECIFE",
        "event_type": "INUNDACAO_URBANA",
        "candidate_period": "2021-05/2021-07",
        "location_text": "Recife, PE",
        "latitude": "",
        "longitude": "",
        "source_id": "DOS_REC_2021",
        "source_type": "DOSSIER_EVENT_TARGET",
        "source_name": "event_evidence_dossier_registry.csv",
        "dossier_status": "DOSSIER_OPEN",
        "current_blocker": "LICENSE_UNKNOWN",
        "notes": (
            "Candidato a evento de inundação urbana em Recife 2021 identificado no "
            "dossier de eventos. G1 MISSING, G3 MISSING, G4 MISSING. Licença desconhecida. "
            "Não confirmado por fonte institucional."
        ),
    },
    {
        "event_id": "EVENT_REC_2022_INUNDACAO_DOSSIER",
        "region": "RECIFE",
        "event_type": "INUNDACAO_URBANA",
        "candidate_period": "2022",
        "location_text": "Recife, PE",
        "latitude": "",
        "longitude": "",
        "source_id": "DOS_REC_2022",
        "source_type": "DOSSIER_EVENT_TARGET",
        "source_name": "event_evidence_dossier_registry.csv",
        "dossier_status": "DOSSIER_OPEN",
        "current_blocker": "LICENSE_UNKNOWN",
        "notes": (
            "Candidato a evento de inundação urbana em Recife 2022 identificado no "
            "dossier de eventos. G1 MISSING, G3 MISSING, G4 MISSING. Licença desconhecida. "
            "Evento específico não identificado no período anual."
        ),
    },
    {
        "event_id": "EVENT_REC_2022_05_24_30_DECRETO",
        "region": "RECIFE",
        "event_type": "INUNDACAO_URBANA",
        "candidate_period": "2022-05-24/2022-05-30",
        "location_text": "Recife, PE — Decreto nº 35.669/2022",
        "latitude": "",
        "longitude": "",
        "source_id": "NEED_REC_001",
        "source_type": "GOVERNMENT_DECREE_TARGET",
        "source_name": "Decreto nº 35.669/2022 — Situação de emergência Recife",
        "dossier_status": "NOT_ACQUIRED",
        "current_blocker": "NOT_ACQUIRED",
        "notes": (
            "Referência a decreto municipal de situação de emergência (Recife, maio 2022). "
            "Não adquirido. Boletins COMPDEC associados: FORMAL_REQUEST_REQUIRED. "
            "Origem: manual_external_evidence_needed_registry.csv NEED_REC_001."
        ),
    },
    {
        "event_id": "EVENT_REC_FLOOD_PLACEHOLDER_PE3D",
        "region": "RECIFE",
        "event_type": "INUNDACAO_URBANA",
        "candidate_period": "",
        "location_text": "Recife, PE — contexto topográfico PE3D",
        "latitude": "",
        "longitude": "",
        "source_id": "recife_pe3d_mde",
        "source_type": "TOPOGRAPHIC_CONTEXT",
        "source_name": "PE3D/MDE Recife — external_evidence_registry.csv",
        "dossier_status": "EXISTS_PARTIAL",
        "current_blocker": "NO_EVENT_DATE",
        "notes": (
            "Contexto topográfico PE3D para Recife. Não é evidência de evento — é contexto "
            "geomorfológico. Sem data de evento associada. patch_event_reference_link_registry "
            "classifica como CONTEXTUAL_ONLY. rec_01_flood_search_pe3d placeholder."
        ),
    },
]


def _make_event_row(seed: dict[str, Any], seq: int) -> dict[str, Any]:
    """Build a normalized event registry row from a seed dict."""
    date_iso, date_status = normalize_event_date(seed.get("candidate_period", ""))
    time_prec = classify_temporal_precision(date_status)
    lat = seed.get("latitude", "")
    lon = seed.get("longitude", "")
    loc = seed.get("location_text", "")
    spatial_prec = classify_spatial_precision(loc, lat, lon)
    source_rel = classify_source_reliability(seed.get("source_name", ""), seed.get("source_type", ""))

    # Build evidence row for classify_evidence_use
    evidence_row = {
        "candidate_source_name": seed.get("source_name", ""),
        "candidate_date_raw": seed.get("candidate_period", ""),
        "candidate_location_raw": loc,
        "region": seed.get("region", ""),
        "current_blocker": seed.get("current_blocker", ""),
        "dossier_status": seed.get("dossier_status", ""),
        "confidence_preliminary": "",
    }
    allowed_use = classify_evidence_use(evidence_row)

    # Determine observed_event_status
    blocker = seed.get("current_blocker", "").upper()
    if allowed_use == "BLOCKED_FIXTURE_OR_SYNTHETIC":
        obs_status = OBS_FIXTURE
        blocked_reason = allowed_use
    elif blocker in ("NOT_ACQUIRED", "FORMAL_REQUEST_REQUIRED"):
        # Source completely absent — insufficient evidence
        obs_status = OBS_BLOCKED
        blocked_reason = f"SOURCE_{blocker}"
    elif blocker == "LICENSE_UNKNOWN" and date_iso:
        # Event is documented internally (dossier) but external source license unknown.
        # Still contextual: we know the candidate event exists, just can't confirm it.
        obs_status = OBS_CONTEXTUAL
        blocked_reason = "SOURCE_LICENSE_UNKNOWN_CONTEXTUAL_ONLY"
    elif allowed_use.startswith("BLOCKED_NO"):
        obs_status = OBS_BLOCKED
        blocked_reason = allowed_use
    elif not date_iso:
        obs_status = OBS_CONTEXTUAL
        blocked_reason = "NO_CONFIRMED_EVENT_DATE"
    else:
        obs_status = OBS_CONTEXTUAL
        blocked_reason = "EVIDENCE_NOT_CONFIRMED_BY_OFFICIAL_SOURCE"

    return {
        "event_id": seed["event_id"],
        "region": normalize_region(seed.get("region", "")),
        "event_type": seed.get("event_type", "UNKNOWN"),
        "event_date_iso": date_iso,
        "event_date_status": date_status,
        "event_time_precision": time_prec,
        "location_text": loc,
        "latitude": lat,
        "longitude": lon,
        "spatial_precision_level": spatial_prec,
        "source_candidate_id": seed.get("source_id", ""),
        "source_type": seed.get("source_type", ""),
        "source_name": seed.get("source_name", ""),
        "source_reliability_level": source_rel,
        "observed_event_status": obs_status,
        "can_be_used_as_ground_truth": "false",
        "can_train_model": "false",
        "can_create_operational_label": "false",
        "allowed_use": allowed_use,
        "blocked_reason": blocked_reason,
        "notes": seed.get("notes", ""),
    }


def run() -> None:
    # Load v1ou inventory for supplemental context (not strictly required)
    v1ou_rows = read_csv(IN_V1OU) if IN_V1OU.exists() else []
    v1ou_recife = [r for r in v1ou_rows if r.get("region") == "RECIFE" and r.get("allowed_for_event_registry") == "true"]

    rows: list[dict[str, Any]] = []
    for i, seed in enumerate(SEED_RECIFE_EVENTS):
        row = _make_event_row(seed, i)
        rows.append(row)

    # Enforce guardrails
    assert_no_forbidden_true(rows, "v1ov_registry")
    require_no_abs_paths_in_rows(rows, "v1ov_registry")

    write_csv_safe(OUT_REGISTRY, rows, REGISTRY_FIELDS)
    write_schema_safe(SCHEMA_REGISTRY, REGISTRY_FIELDS, "v1ov_ground_reference_observed_event_registry")

    # Counts
    confirmed = sum(1 for r in rows if r["observed_event_status"] == OBS_CONFIRMED)
    probable = sum(1 for r in rows if r["observed_event_status"] == OBS_PROBABLE)
    contextual = sum(1 for r in rows if r["observed_event_status"] == OBS_CONTEXTUAL)
    blocked = sum(1 for r in rows if r["observed_event_status"] == OBS_BLOCKED)
    total = len(rows)

    status = "NO_OBSERVED_EVENT_CONFIRMED" if confirmed == 0 and probable == 0 else "EVENTS_FOUND_REVIEW_ONLY"

    summary_rows = [
        {"stat_key": "total_events", "stat_value": str(total)},
        {"stat_key": "observed_event_confirmed_review_only", "stat_value": str(confirmed)},
        {"stat_key": "observed_event_probable_review_only", "stat_value": str(probable)},
        {"stat_key": "contextual_evidence_only", "stat_value": str(contextual)},
        {"stat_key": "blocked_insufficient_evidence", "stat_value": str(blocked)},
        {"stat_key": "can_be_used_as_ground_truth_any", "stat_value": "false"},
        {"stat_key": "can_train_model_any", "stat_value": "false"},
        {"stat_key": "can_create_operational_label_any", "stat_value": "false"},
        {"stat_key": "v1ou_recife_sources_loaded", "stat_value": str(len(v1ou_recife))},
        {"stat_key": "registry_status", "stat_value": status},
        {"stat_key": "stage", "stat_value": "v1ov"},
    ]
    write_csv_safe(OUT_SUMMARY, summary_rows, SUMMARY_FIELDS)
    write_schema_safe(SCHEMA_SUMMARY, SUMMARY_FIELDS, "v1ov_summary")

    write_doc(
        DOC,
        "v1ov — Ground Reference Observed Event Registry",
        [
            "## Objetivo",
            "Construir registro normalizado de eventos observados a partir de candidatos "
            "identificados em v1ou e datasets existentes. Não inventa eventos. Separa evento "
            "observado de evidência contextual.",
            "## Resultado",
            f"Total de eventos no registro: {total}. "
            f"Confirmados review-only: {confirmed}. "
            f"Prováveis review-only: {probable}. "
            f"Contextual apenas: {contextual}. "
            f"Bloqueados insuficiente: {blocked}. "
            f"Status: {status}.",
            "## Guardrails",
            "can_be_used_as_ground_truth=false, can_train_model=false, "
            "can_create_operational_label=false em todos os registros.",
            "## Por que eventos permanecem contextuais",
            "Os eventos de Recife identificados em DOS_REC_2021, DOS_REC_2022 e "
            "REC_2022_05_24_30 têm dossier aberto com G1 MISSING (confirmação institucional "
            "ausente), G3 MISSING (alinhamento temporal não confirmado) e G4 MISSING "
            "(geometria de área afetada ausente). O decreto nº 35.669/2022 não foi adquirido. "
            "Boletins COMPDEC exigem pedido formal. Nenhum evento pode ser promovido a "
            "referência observacional confirmada.",
        ],
    )

    print(f"[v1ov] {total} events registered: {confirmed} confirmed, {probable} probable, "
          f"{contextual} contextual, {blocked} blocked")
    print(f"[v1ov] Status: {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v1ov ground reference observed event registry")
    parser.parse_args()
    run()
