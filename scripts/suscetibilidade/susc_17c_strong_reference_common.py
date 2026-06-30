"""SUSC-17C Strong Reference Acquisition Canary (review-only, fail-closed).

Pipeline to move from "insufficient strong references" (the SUSC-17A deadlock:
65 strong patch-links from only 2 footprints, Curitiba-only, 0 event dates)
toward dated, verifiable strong candidates. It anchors on official/documentary
dated events and plans a Sentinel-1/SAR technical-footprint path. It does NOT
build the 17B benchmark, does NOT execute heavy raster, does NOT require GEE
credentials.

It NEVER changes score v6, creates score v7, training data, supervised labels,
models, ground truth or true negatives. Observed footprints stay candidate
evidence. Alerts are never occurrences; risk areas are never occurred events;
administrative records without fine geometry never become strong. No event,
date, geometry, source or coordinate is invented: missing values use controlled
absence tokens (``not_available`` / ``unknown`` / ``insufficient`` / ``blocked``).
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_io import read_csv, write_csv, write_json, write_markdown  # noqa: E402

DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

AUDIT = OUT / "SUSC_WORKTREE_AUDIT_BEFORE_17C.md"
CANDIDATE_SCHEMA = SCHEMAS / "susc_17c_candidate_event_schema_v1.json"
SAR_SCHEMA = SCHEMAS / "susc_17c_sar_feasibility_schema_v1.json"

# --- inputs (read-only) ---------------------------------------------------- #
EVENTS_13A = DAT / "susc_13a_strong_observed_events_parsed_v1.csv"
WINDOW_PLAN = DAT / "susc_16a_sentinel_event_window_plan_v1.csv"
SAR_CANDIDATES_16A = DAT / "susc_16a_local_sar_flood_footprint_candidates_v1.csv"
FOOTPRINT_CATALOG = DAT / "susc_16a_observed_footprint_catalog_v1.csv"
EXTERNAL_REGISTRY = ROOT / "datasets" / "external_evidence_registry.csv"
REGISTRY_17A = OUT / "susc_17a_reference_evidence_registry.csv"
SUMMARY_17A = OUT / "susc_17a_reference_evidence_summary.json"
SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"

GEE_STUB = "scripts/suscetibilidade/susc_16a_gee_sentinel1_flood_mapping_stub.js"
STAC_STUB = "scripts/suscetibilidade/susc_16a_stac_sentinel1_query_stub.py"

# --- outputs --------------------------------------------------------------- #
SOURCE_INVENTORY = OUT / "susc_17c_source_inventory.csv"
TARGET_PACK = OUT / "susc_17c_candidate_event_target_pack.csv"
DATE_RESOLUTION = OUT / "susc_17c_event_date_resolution.csv"
SAR_FEASIBILITY = OUT / "susc_17c_sar_feasibility_pack.csv"
PRIORITY_CANARY = OUT / "susc_17c_priority_canary_events.csv"
SENTINEL_TASK_PLAN = OUT / "susc_17c_sentinel1_task_plan.json"
HUMAN_QA_QUEUE = OUT / "susc_17c_human_qa_queue.csv"
SUMMARY = OUT / "susc_17c_strong_reference_acquisition_summary.json"
REPORT = OUT / "SUSC_17C_STRONG_REFERENCE_ACQUISITION_CANARY_REPORT.md"

REQUIRED_INPUTS = [AUDIT, CANDIDATE_SCHEMA, SAR_SCHEMA, EVENTS_13A, WINDOW_PLAN, SCORE_V6]

MANDATORY = (
    "O SUSC-17C prepara a aquisicao de referencias fortes datadas e verificaveis "
    "review-only. Ele ancora em evento oficial datado e planeja o caminho "
    "Sentinel-1/SAR como produtor de footprint tecnico, nao como score. Nao cria "
    "benchmark 17B, nao altera o score v6 oficial, nao cria score v7, nao cria "
    "treino, modelo, label ou ground truth, nao transforma footprint em ground "
    "truth, nao transforma ausencia documental em negativo, nao chama alerta de "
    "ocorrencia e nao chama area de risco de evento ocorrido."
)

SAR_PRECISIONS = {"exact_day", "date_range"}
PRIORITY_TARGET = 5  # select 3..5 canary events

PMASK = "permanent_water_mask;HAND_mask;slope_mask;urban_ambiguity_guard"
RECOMMENDED_METHOD = "sentinel1_grd_log_ratio_pre_post_change_detection_then_candidate_polygonization_then_human_qa"
RECOMMENDED_POLARIZATIONS = "VV;VH"

DECISION_VALUES = (
    "accepted;rejected;ambiguous;needs_more_source;phenomenon_mismatch;"
    "temporal_mismatch;spatial_mismatch"
)

FLOOD_PHENOMENA = {"urban_flooding", "flash_flood", "flood", "alagamento", "enxurrada", "inundacao"}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _require_inputs() -> None:
    missing = [p for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("; ".join(str(p.relative_to(ROOT)) for p in missing))


def _na(value) -> str:
    value = ("" if value is None else str(value)).strip()
    return value if value else "not_available"


def _parse_iso(value: str) -> _dt.date | None:
    value = (value or "").strip()
    try:
        return _dt.date.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def _institution_source_id(institution: str) -> str:
    inst = (institution or "").strip().lower()
    if "international charter" in inst:
        return "SRC_INTERNATIONAL_CHARTER"
    if "defesa civil" in inst:
        return "SRC_DEFESA_CIVIL"
    if "inmet" in inst:
        return "SRC_INMET"
    if "inea" in inst:
        return "SRC_INEA_RJ"
    if "cprm" in inst or "sgb" in inst:
        return "SRC_SGB_CPRM"
    return "SRC_OFFICIAL_UNRESOLVED"


def _occurrence_status_13a(row: dict) -> str:
    if row.get("is_alert") == "true":
        return "alert"
    if row.get("is_risk_area") == "true":
        return "risk_area"
    if row.get("is_administrative_only") == "true":
        return "administrative"
    if row.get("is_observed_event") == "true":
        return "observed_occurrence"
    return "unknown"


def _date_precision_from_temporal(temporal_precision: str, has_date: bool) -> str:
    tp = (temporal_precision or "").strip().lower()
    if not has_date:
        return "not_available"
    if tp in ("day", "explicit", "exact", "day_explicit", "exact_day"):
        return "exact_day"
    if tp in ("period", "range", "date_range"):
        return "date_range"
    if tp in ("month", "month_only"):
        return "month_only"
    return "unknown"


# --------------------------------------------------------------------------- #
# 1. source inventory
# --------------------------------------------------------------------------- #

def _source_rows() -> list[dict]:
    """Sources actually present in project artifacts. Missing priority sources
    (CEMADEN, APAC, ANA/Hidroweb, S2iD) are NOT fabricated here; they are listed
    as acquisition gaps in the report instead."""
    S = lambda **k: {  # noqa: E731
        "has_event_date": "false", "has_geometry": "false", "has_point": "false",
        "has_polygon": "false", "has_bbox": "false", "has_address_only": "false",
        "has_alert_only": "false", "has_risk_area_only": "false",
        "usable_for_event_record": "false", "usable_for_source_footprint": "false",
        "usable_for_sar_anchor": "false", "access_status": "present_local",
        "notes": "", **k}
    return [
        S(source_id="SRC_INTERNATIONAL_CHARTER", source_name="International Charter Space and Major Disasters",
          source_type="technical_remote_sensing_flood_map", city_or_region="recife",
          authority_level="international_technical", has_event_date="true", has_geometry="true",
          has_polygon="true", has_bbox="true", usable_for_event_record="true",
          usable_for_source_footprint="true", usable_for_sar_anchor="true", source_artifact=_rel(EVENTS_13A),
          notes="evento datado 2022-05-24 com geometria oficial coarse (neighborhood); ancora forte review-only"),
        S(source_id="SRC_DEFESA_CIVIL", source_name="Defesa Civil municipal/estadual",
          source_type="official_civil_defense_occurrence", city_or_region="recife;petropolis",
          authority_level="official_municipal_state", has_event_date="true", has_geometry="true",
          has_point="true", has_bbox="true", has_address_only="true", has_risk_area_only="true",
          usable_for_event_record="true", usable_for_source_footprint="false", usable_for_sar_anchor="true",
          source_artifact=_rel(EVENTS_13A),
          notes="ocorrencias datadas; precisao neighborhood; inclui registros de area de risco (nao evento)"),
        S(source_id="SRC_INMET", source_name="INMET", source_type="meteorological_administrative_alert",
          city_or_region="recife", authority_level="federal_meteorological", has_alert_only="true",
          has_address_only="false", usable_for_event_record="false", usable_for_source_footprint="false",
          usable_for_sar_anchor="false", source_artifact=_rel(EVENTS_13A),
          notes="registros administrativos/alerta; nao sao ocorrencia de evento"),
        S(source_id="SRC_INEA_RJ", source_name="INEA/RJ", source_type="state_environmental_agency",
          city_or_region="recife", authority_level="state_environmental", has_geometry="true",
          has_polygon="true", usable_for_event_record="false", usable_for_source_footprint="false",
          usable_for_sar_anchor="false", source_artifact=_rel(EVENTS_13A),
          notes="poligono sem data resolvida; contexto"),
        S(source_id="SRC_SGB_CPRM", source_name="SGB/CPRM (Servico Geologico do Brasil)",
          source_type="geological_terrain_reference", city_or_region="recife;curitiba;petropolis",
          authority_level="federal_geological", has_geometry="true", usable_for_event_record="false",
          usable_for_source_footprint="false", usable_for_sar_anchor="false", source_artifact=_rel(EXTERNAL_REGISTRY),
          notes="cartas geologicas/terreno; contexto de suscetibilidade, nao evento datado"),
        S(source_id="SRC_GEOCURITIBA_IPPUC", source_name="GeoCuritiba / IPPUC / Defesa Civil Curitiba",
          source_type="municipal_spatial_data_infrastructure", city_or_region="curitiba",
          authority_level="official_municipal", has_geometry="true", usable_for_event_record="false",
          usable_for_source_footprint="false", usable_for_sar_anchor="false", source_artifact=_rel(EXTERNAL_REGISTRY),
          notes="MDT/MDS/curvas e dados defesa civil Curitiba; sem evento datado com geometria fina aqui"),
        S(source_id="SRC_ESIG_EMLURB", source_name="ESIG/EMLURB Recife (macrodrenagem)",
          source_type="municipal_hydrography", city_or_region="recife", authority_level="official_municipal",
          has_geometry="true", usable_for_event_record="false", usable_for_source_footprint="false",
          usable_for_sar_anchor="false", source_artifact=_rel(EXTERNAL_REGISTRY),
          notes="hidrografia/drenagem; contexto"),
        S(source_id="SRC_16A_SENTINEL_WINDOW_PLAN", source_name="SUSC-16A Sentinel event window plan",
          source_type="internal_sar_window_plan", city_or_region="recife", authority_level="internal_derived",
          has_event_date="true", has_bbox="true", usable_for_event_record="true",
          usable_for_source_footprint="false", usable_for_sar_anchor="true", source_artifact=_rel(WINDOW_PLAN),
          notes="161 janelas datadas Recife, S1-elegiveis, com pre/pos e AOI; backbone SAR"),
        S(source_id="SRC_16A_FOOTPRINT_CATALOG", source_name="SUSC-16A observed footprint catalog",
          source_type="internal_footprint_catalog", city_or_region="recife;curitiba;petropolis",
          authority_level="internal_derived", has_geometry="true", has_polygon="true", has_point="true",
          has_bbox="true", usable_for_event_record="true", usable_for_source_footprint="true",
          usable_for_sar_anchor="false", source_artifact=_rel(FOOTPRINT_CATALOG),
          notes="12 footprints elegiveis sem data; geometria coarse"),
        S(source_id="SRC_16A_SAR_CANDIDATES", source_name="SUSC-16A local SAR flood footprint candidates",
          source_type="internal_sar_canary", city_or_region="unknown", authority_level="internal_derived",
          usable_for_event_record="false", usable_for_source_footprint="false", usable_for_sar_anchor="true",
          access_status="blocked_no_raster", source_artifact=_rel(SAR_CANDIDATES_16A),
          notes="3 candidatos SAR BLOCKED_NO_OFFLINE_RASTER_PARSER; sem execucao raster"),
        S(source_id="SRC_17A_REGISTRY", source_name="SUSC-17A reference evidence registry",
          source_type="internal_reference_protocol", city_or_region="curitiba;recife;petropolis",
          authority_level="internal_derived", has_geometry="true", usable_for_event_record="true",
          usable_for_source_footprint="true", usable_for_sar_anchor="false", source_artifact=_rel(REGISTRY_17A),
          notes="75 registros review-only; 65 fortes patch-linked sem data (deadlock 17B)"),
    ]


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


SOURCE_FIELDS = [
    "source_id", "source_name", "source_type", "city_or_region", "authority_level",
    "has_event_date", "has_geometry", "has_point", "has_polygon", "has_bbox",
    "has_address_only", "has_alert_only", "has_risk_area_only",
    "usable_for_event_record", "usable_for_source_footprint", "usable_for_sar_anchor",
    "source_artifact", "access_status", "notes",
]


def build_source_inventory() -> int:
    _require_inputs()
    rows = sorted(_source_rows(), key=lambda r: r["source_id"])
    rows = [{f: _na(r.get(f)) for f in SOURCE_FIELDS} for r in rows]
    write_csv(SOURCE_INVENTORY, rows, SOURCE_FIELDS)
    print(f"source inventory -> {_rel(SOURCE_INVENTORY)} ({len(rows)} rows)")
    return 0


# --------------------------------------------------------------------------- #
# 2/3. candidate event target pack + date resolution
# --------------------------------------------------------------------------- #

TARGET_FIELDS = [
    "candidate_event_id", "source_id", "city", "region", "phenomenon_type",
    "event_date_candidate", "event_date_precision", "date_resolution_status",
    "geometry_status", "geometry_type", "has_official_geometry", "has_patch_link",
    "has_pre_event_window", "has_post_event_window", "occurrence_status",
    "sar_candidate", "strong_reference_candidate", "expected_strong_class",
    "blocking_reason", "priority_score", "review_only", "trainable", "ground_truth",
]

DATE_FIELDS = [
    "candidate_event_id", "raw_date", "resolved_event_start_date",
    "resolved_event_end_date", "event_date_precision", "pre_event_window_start",
    "pre_event_window_end", "post_event_window_start", "post_event_window_end",
    "date_resolution_status", "date_blocking_reason", "source_artifact",
]


def _windows(start: _dt.date | None):
    if start is None:
        return ("not_available",) * 4
    pre_start = (start - _dt.timedelta(days=30)).isoformat()
    pre_end = (start - _dt.timedelta(days=7)).isoformat()
    post_start = start.isoformat()
    post_end = (start + _dt.timedelta(days=7)).isoformat()
    return pre_start, pre_end, post_start, post_end


def _strong_and_sar(precision: str, has_official_geometry: bool, sar_feasible: bool,
                    occurrence: str) -> tuple[str, str, str]:
    """Returns (sar_candidate, strong_reference_candidate, expected_strong_class)."""
    dated = precision in SAR_PRECISIONS
    is_occurrence = occurrence == "observed_occurrence"
    sar_candidate = "true" if (dated and sar_feasible and is_occurrence) else "false"
    # strong candidate: (date + official geometry) OR (date + SAR feasibility), and occurrence.
    strong = is_occurrence and dated and (has_official_geometry or sar_feasible)
    if not strong:
        return sar_candidate, "false", "blocked"
    if has_official_geometry:
        expected = "official_observed_event_polygon"
    else:
        expected = "technical_remote_sensing_flood_footprint"
    return sar_candidate, "true", expected


def _candidate_events() -> tuple[list[dict], list[dict]]:
    targets: list[dict] = []
    dates: list[dict] = []

    # (A) classified observed/risk/alert/admin events from SUSC-13A.
    for row in read_csv(EVENTS_13A):
        eid = f"S17C_E_{row['event_id']}"
        occurrence = _occurrence_status_13a(row)
        raw_date = (row.get("event_date") or "").strip()
        start = _parse_iso(raw_date) or _parse_iso(row.get("event_period_start", ""))
        end = _parse_iso(row.get("event_period_end", "")) or start
        precision = _date_precision_from_temporal(row.get("temporal_precision", ""), start is not None)
        geometry_type = (row.get("geometry_type") or "").strip()
        # neighborhood precision is NOT fine geometry (rule 8): coarse only.
        has_official_geometry = bool(geometry_type) and geometry_type.lower() in ("polygon", "point_set") \
            and occurrence == "observed_occurrence"
        geometry_status = (
            "official_coarse_polygon" if geometry_type.lower() == "polygon" and has_official_geometry
            else "point_set" if geometry_type.lower() == "point_set" and has_official_geometry
            else "not_available")
        sar_feasible = precision in SAR_PRECISIONS and occurrence == "observed_occurrence"
        sar_c, strong_c, expected = _strong_and_sar(precision, has_official_geometry, sar_feasible, occurrence)
        win = _windows(start if precision in SAR_PRECISIONS else None)
        has_pre = "true" if win[0] != "not_available" else "false"
        has_post = "true" if win[2] != "not_available" else "false"
        blocking = _blocking_reason(occurrence, precision, has_official_geometry, sar_feasible)
        priority = _priority_score(occurrence, precision, has_official_geometry, sar_feasible, source_event_count=1)
        targets.append({
            "candidate_event_id": eid,
            "source_id": _institution_source_id(row.get("source_institution", "")),
            "city": row.get("municipality") or row.get("region", ""),
            "region": row.get("region", ""),
            "phenomenon_type": _na(row.get("event_type")),
            "event_date_candidate": _na(raw_date),
            "event_date_precision": precision,
            "date_resolution_status": "resolved" if start is not None else "blocked",
            "geometry_status": geometry_status,
            "geometry_type": _na(geometry_type),
            "has_official_geometry": "true" if has_official_geometry else "false",
            "has_patch_link": "false",
            "has_pre_event_window": has_pre,
            "has_post_event_window": has_post,
            "occurrence_status": occurrence,
            "sar_candidate": sar_c,
            "strong_reference_candidate": strong_c,
            "expected_strong_class": expected,
            "blocking_reason": blocking,
            "priority_score": f"{priority:.3f}",
            "review_only": "true", "trainable": "false", "ground_truth": "false",
        })
        dates.append(_date_row(eid, raw_date, start, end, precision, _rel(EVENTS_13A)))

    # (B) dated Recife SAR windows from the SUSC-16A sentinel event window plan.
    for row in read_csv(WINDOW_PLAN):
        wid = row["event_window_id"]
        eid = f"S17C_W_{wid}"
        raw_date = (row.get("event_date") or "").strip()
        start = _parse_iso(raw_date)
        precision = "exact_day" if start is not None else "not_available"
        s1_ok = row.get("eligible_for_sentinel1") == "true"
        occurrence = "observed_occurrence"  # backed by official occurrence count on that date
        sar_feasible = precision in SAR_PRECISIONS and s1_ok
        # no fine official geometry: AOI bbox at neighborhood scale (rule 8).
        sar_c, strong_c, expected = _strong_and_sar(precision, False, sar_feasible, occurrence)
        win = _windows(start)
        try:
            sec = int(row.get("source_event_count") or 0)
        except ValueError:
            sec = 0
        priority = _priority_score(occurrence, precision, False, sar_feasible, source_event_count=sec)
        targets.append({
            "candidate_event_id": eid,
            "source_id": "SRC_16A_SENTINEL_WINDOW_PLAN",
            "city": row.get("municipality") or row.get("region", ""),
            "region": row.get("region", ""),
            "phenomenon_type": "urban_flooding",
            "event_date_candidate": _na(raw_date),
            "event_date_precision": precision,
            "date_resolution_status": "resolved" if start is not None else "blocked",
            "geometry_status": "aoi_bbox_only",
            "geometry_type": "bbox",
            "has_official_geometry": "false",
            "has_patch_link": "false",
            "has_pre_event_window": "true" if win[0] != "not_available" else "false",
            "has_post_event_window": "true" if win[2] != "not_available" else "false",
            "occurrence_status": occurrence,
            "sar_candidate": sar_c,
            "strong_reference_candidate": strong_c,
            "expected_strong_class": expected,
            "blocking_reason": _blocking_reason(occurrence, precision, False, sar_feasible),
            "priority_score": f"{priority:.3f}",
            "review_only": "true", "trainable": "false", "ground_truth": "false",
        })
        dates.append(_date_row(eid, raw_date, start, start, precision, _rel(WINDOW_PLAN)))

    targets.sort(key=lambda r: r["candidate_event_id"])
    dates.sort(key=lambda r: r["candidate_event_id"])
    return targets, dates


def _date_row(eid, raw_date, start, end, precision, artifact) -> dict:
    if start is None:
        return {
            "candidate_event_id": eid, "raw_date": _na(raw_date),
            "resolved_event_start_date": "not_available",
            "resolved_event_end_date": "not_available",
            "event_date_precision": precision,
            "pre_event_window_start": "not_available", "pre_event_window_end": "not_available",
            "post_event_window_start": "not_available", "post_event_window_end": "not_available",
            "date_resolution_status": "blocked",
            "date_blocking_reason": "no_resolvable_event_date", "source_artifact": artifact,
        }
    pre_s, pre_e, post_s, post_e = _windows(start)
    return {
        "candidate_event_id": eid, "raw_date": _na(raw_date),
        "resolved_event_start_date": start.isoformat(),
        "resolved_event_end_date": (end or start).isoformat(),
        "event_date_precision": precision,
        "pre_event_window_start": pre_s, "pre_event_window_end": pre_e,
        "post_event_window_start": post_s, "post_event_window_end": post_e,
        "date_resolution_status": "resolved",
        "date_blocking_reason": "not_available", "source_artifact": artifact,
    }


def _blocking_reason(occurrence, precision, has_official_geometry, sar_feasible) -> str:
    if occurrence == "alert":
        return "alert_only_not_occurrence"
    if occurrence == "risk_area":
        return "risk_area_not_event"
    if occurrence == "administrative":
        return "administrative_record_without_fine_geometry"
    if occurrence != "observed_occurrence":
        return "not_a_confirmed_observed_occurrence"
    if precision not in SAR_PRECISIONS:
        return "no_usable_event_date"
    if not has_official_geometry and not sar_feasible:
        return "needs_official_geometry_or_sar_execution"
    return "not_available"


def _priority_score(occurrence, precision, has_official_geometry, sar_feasible, source_event_count) -> float:
    if occurrence != "observed_occurrence" or precision not in SAR_PRECISIONS:
        return 0.0
    score = 0.4 if sar_feasible else 0.0
    score += 0.3 if has_official_geometry else 0.0
    score += min(0.3, 0.02 * max(0, source_event_count))
    return round(min(1.0, score), 3)


def build_target_pack_and_dates() -> int:
    targets, dates = _candidate_events()
    write_csv(TARGET_PACK, targets, TARGET_FIELDS)
    write_csv(DATE_RESOLUTION, dates, DATE_FIELDS)
    print(f"target pack -> {_rel(TARGET_PACK)} ({len(targets)} rows); dates -> {_rel(DATE_RESOLUTION)}")
    return 0


# --------------------------------------------------------------------------- #
# 4. SAR feasibility pack
# --------------------------------------------------------------------------- #

SAR_FIELDS = [
    "candidate_event_id", "city", "event_date", "pre_event_window", "post_event_window",
    "requires_sentinel1_pre", "requires_sentinel1_post", "requires_patch_intersection",
    "requires_permanent_water_mask", "requires_hand_slope_masks",
    "requires_urban_ambiguity_guard", "sar_feasibility_status", "sar_blocking_reason",
    "recommended_polarizations", "recommended_method", "expected_output_class",
    "review_only", "trainable", "ground_truth",
]


def build_sar_feasibility() -> int:
    dates_by_id = {r["candidate_event_id"]: r for r in read_csv(DATE_RESOLUTION)}
    rows = []
    for tgt in read_csv(TARGET_PACK):
        eid = tgt["candidate_event_id"]
        dr = dates_by_id.get(eid, {})
        feasible = tgt["sar_candidate"] == "true"
        pre = (f"{dr.get('pre_event_window_start')}/{dr.get('pre_event_window_end')}"
               if dr.get("pre_event_window_start", "not_available") != "not_available" else "not_available")
        post = (f"{dr.get('post_event_window_start')}/{dr.get('post_event_window_end')}"
                if dr.get("post_event_window_start", "not_available") != "not_available" else "not_available")
        if feasible:
            status, blocking, expected = "feasible", "not_available", "technical_remote_sensing_flood_footprint"
        else:
            status, expected = "blocked", "blocked"
            blocking = tgt["blocking_reason"] if tgt["blocking_reason"] != "not_available" else "not_sar_feasible"
        rows.append({
            "candidate_event_id": eid,
            "city": tgt["city"],
            "event_date": tgt["event_date_candidate"],
            "pre_event_window": pre,
            "post_event_window": post,
            "requires_sentinel1_pre": "true",
            "requires_sentinel1_post": "true",
            "requires_patch_intersection": "true",
            "requires_permanent_water_mask": "true",
            "requires_hand_slope_masks": "true",
            "requires_urban_ambiguity_guard": "true",
            "sar_feasibility_status": status,
            "sar_blocking_reason": blocking,
            "recommended_polarizations": RECOMMENDED_POLARIZATIONS,
            "recommended_method": RECOMMENDED_METHOD,
            "expected_output_class": expected,
            "review_only": "true", "trainable": "false", "ground_truth": "false",
        })
    rows.sort(key=lambda r: r["candidate_event_id"])
    write_csv(SAR_FEASIBILITY, rows, SAR_FIELDS)
    print(f"sar feasibility -> {_rel(SAR_FEASIBILITY)} ({len(rows)} rows)")
    return 0


# --------------------------------------------------------------------------- #
# 5. priority canary events (3..5)
# --------------------------------------------------------------------------- #

PRIORITY_FIELDS = [
    "priority_rank", "candidate_event_id", "city", "event_date", "source_id",
    "why_selected", "next_action", "expected_strong_class", "blocking_risk",
    "qa_required", "review_only", "trainable", "ground_truth",
]


def _select_priority(targets: list[dict]) -> list[dict]:
    eligible = [t for t in targets if t["sar_candidate"] == "true"
                and t["phenomenon_type"] in FLOOD_PHENOMENA
                and t["occurrence_status"] == "observed_occurrence"]
    # rank: official geometry first, then priority_score, then id for stability.
    eligible.sort(key=lambda t: (
        t["has_official_geometry"] != "true", -float(t["priority_score"]), t["candidate_event_id"]))
    return eligible[:PRIORITY_TARGET]


def build_priority_canary() -> list[dict]:
    targets = read_csv(TARGET_PACK)
    selected = _select_priority(targets)
    regions = {t["region"] for t in selected}
    rows = []
    for rank, t in enumerate(selected, start=1):
        diverse = len(regions) > 1
        if t["has_official_geometry"] == "true":
            why = "evento datado com geometria oficial (footprint tecnico/coarse) e fenomeno de inundacao"
            nxt = "verificar geometria oficial e QA humano; opcionalmente confirmar via SAR"
        else:
            why = "evento datado, S1-elegivel, fenomeno de inundacao; sem geometria oficial fina"
            nxt = "executar Sentinel-1/SAR canary para produzir footprint tecnico candidato"
        rows.append({
            "priority_rank": str(rank),
            "candidate_event_id": t["candidate_event_id"],
            "city": t["city"],
            "event_date": t["event_date_candidate"],
            "source_id": t["source_id"],
            "why_selected": why,
            "next_action": nxt,
            "expected_strong_class": t["expected_strong_class"],
            "blocking_risk": ("regiao concentrada (sem diversidade)" if not diverse else "not_available")
            + ("; depende de execucao SAR" if t["has_official_geometry"] != "true" else ""),
            "qa_required": "true",
            "review_only": "true", "trainable": "false", "ground_truth": "false",
        })
    write_csv(PRIORITY_CANARY, rows, PRIORITY_FIELDS)
    print(f"priority canary -> {_rel(PRIORITY_CANARY)} ({len(rows)} rows)")
    return rows


# --------------------------------------------------------------------------- #
# 6. sentinel-1 task plan + 7. human QA queue
# --------------------------------------------------------------------------- #

def build_sentinel_task_plan(priority_rows: list[dict]) -> int:
    sar_by_id = {r["candidate_event_id"]: r for r in read_csv(SAR_FEASIBILITY)}
    tasks = []
    for row in priority_rows:
        eid = row["candidate_event_id"]
        sar = sar_by_id.get(eid, {})
        tasks.append({
            "canary_event_id": eid,
            "aoi_source": "susc_16a_sentinel_event_window_plan_v1.aoi_bbox" if eid.startswith("S17C_W_")
            else "susc_13a_event_bbox",
            "event_date": row["event_date"],
            "pre_window": sar.get("pre_event_window", "not_available"),
            "post_window": sar.get("post_event_window", "not_available"),
            "collection": "COPERNICUS/S1_GRD",
            "polarizations": ["VV", "VH"],
            "orbit_pass_policy": "prefer_same_relative_orbit_and_pass_for_pre_post_pairing",
            "change_detection_method": "log_ratio_pre_post_then_threshold_then_candidate_polygonization",
            "masks": ["permanent_water_mask", "HAND_mask", "slope_mask", "urban_ambiguity_guard"],
            "outputs_expected": {
                "expected_output_class": sar.get("expected_output_class", "blocked"),
                "artifact": "candidate_polygon_geojson_review_only",
                "ground_truth": False,
            },
            "fail_closed_conditions": [
                "no_gee_or_stac_credential -> status blocked, no execution",
                "no_sentinel1_scene_in_window -> status blocked",
                "no_pre_or_post_pair -> status blocked",
                "human_qa_required_before_any_strong_reference",
                "no_raster_download_committed",
            ],
        })
    plan = {
        "protocol": "susc_17c_sentinel1_task_plan",
        "execution": "planned_only_not_executed",
        "requires_credentials": False,
        "compatible_stubs": [GEE_STUB, STAC_STUB],
        "review_only": True,
        "trainable": False,
        "ground_truth": False,
        "score_v6_changed": False,
        "score_v7_created": False,
        "technical_footprint_created_count": 0,
        "tasks": tasks,
    }
    write_json(SENTINEL_TASK_PLAN, plan)
    print(f"sentinel task plan -> {_rel(SENTINEL_TASK_PLAN)} ({len(tasks)} tasks)")
    return 0


QA_FIELDS = [
    "qa_item_id", "candidate_event_id", "evidence_type", "source_artifact",
    "qa_question", "expected_decision_values", "default_status", "blocking_if_unreviewed",
    "review_only", "trainable", "ground_truth",
]


def build_human_qa_queue(priority_rows: list[dict]) -> int:
    rows = []
    for row in priority_rows:
        eid = row["candidate_event_id"]
        evidence_type = ("technical_remote_sensing_flood_footprint_planned"
                         if row["expected_strong_class"] == "technical_remote_sensing_flood_footprint"
                         else "official_observed_event_geometry")
        rows.append({
            "qa_item_id": f"QA_{eid}",
            "candidate_event_id": eid,
            "evidence_type": evidence_type,
            "source_artifact": _rel(PRIORITY_CANARY),
            "qa_question": (
                "O candidato representa uma inundacao observada real na data e no local, "
                "com geometria/footprint compativel com patch-level, sem ser alerta nem area de risco?"),
            "expected_decision_values": DECISION_VALUES,
            "default_status": "needs_review",
            "blocking_if_unreviewed": "true",
            "review_only": "true", "trainable": "false", "ground_truth": "false",
        })
    rows.sort(key=lambda r: r["qa_item_id"])
    write_csv(HUMAN_QA_QUEUE, rows, QA_FIELDS)
    print(f"human qa queue -> {_rel(HUMAN_QA_QUEUE)} ({len(rows)} rows)")
    return 0


# --------------------------------------------------------------------------- #
# 8. summary + report
# --------------------------------------------------------------------------- #

def _readiness(priority_count, sar_candidate_count, date_resolved_count, official_geometry_count) -> str:
    if priority_count == 0:
        return "BLOCKED_NO_CANARY_EVENTS"
    if sar_candidate_count == 0 and official_geometry_count == 0:
        return "BLOCKED_NEEDS_GEOMETRY" if date_resolved_count else "BLOCKED_NEEDS_DATED_EVENTS"
    if sar_candidate_count > 0:
        return "SAR_CANARY_READY"
    return "STRONG_REFERENCE_ACQUISITION_READY"


def build_summary(priority_rows: list[dict]) -> int:
    targets = read_csv(TARGET_PACK)
    dates = read_csv(DATE_RESOLUTION)
    qa = read_csv(HUMAN_QA_QUEUE)
    sources = read_csv(SOURCE_INVENTORY)
    precision_counts = Counter(t["event_date_precision"] for t in targets)
    date_resolved = sum(1 for d in dates if d["date_resolution_status"] == "resolved")
    official_geometry = sum(1 for t in targets if t["has_official_geometry"] == "true")
    sar_candidate = sum(1 for t in targets if t["sar_candidate"] == "true")
    strong_candidate = sum(1 for t in targets if t["strong_reference_candidate"] == "true")
    priority_count = len(priority_rows)
    regions_priority = sorted({r["city"].strip().lower() for r in priority_rows})
    blocks_17b = []
    if len(regions_priority) < 2:
        blocks_17b.append("strong_candidates_concentrated_in_one_region")
    blocks_17b.append("no_sar_footprint_executed_yet_technical_footprint_created_count_0")
    if official_geometry == 0:
        blocks_17b.append("no_fine_official_geometry_available")
    summary = {
        "total_sources": len(sources),
        "total_candidate_events": len(targets),
        "date_resolved_count": date_resolved,
        "exact_day_count": precision_counts.get("exact_day", 0),
        "date_range_count": precision_counts.get("date_range", 0),
        "month_only_count": precision_counts.get("month_only", 0),
        "unknown_date_count": precision_counts.get("unknown", 0) + precision_counts.get("not_available", 0),
        "official_geometry_count": official_geometry,
        "sar_candidate_count": sar_candidate,
        "strong_reference_candidate_count": strong_candidate,
        "priority_canary_count": priority_count,
        "human_qa_items_count": len(qa),
        "strong_reference_created_count": 0,
        "technical_footprint_created_count": 0,
        "eligible_for_17b_now": False,
        "blocks_17b": blocks_17b,
        "priority_regions": regions_priority,
        "review_only": True,
        "trainable": False,
        "ground_truth": False,
        "score_v6_changed": False,
        "score_v7_created": False,
        "readiness_status": _readiness(priority_count, sar_candidate, date_resolved, official_geometry),
        "next_recommended_milestone": _next_milestone(priority_count, sar_candidate, official_geometry),
    }
    write_json(SUMMARY, summary)
    print(f"summary -> {_rel(SUMMARY)}")
    return 0


def _next_milestone(priority_count, sar_candidate, official_geometry) -> str:
    if priority_count == 0:
        return "SUSC-17C2 Official Geometry Acquisition (sem canaries SAR viaveis)"
    if sar_candidate > 0:
        return ("SUSC-17C2 Sentinel-1/SAR Footprint Execution para produzir footprints "
                "tecnicos datados a partir dos canaries priorizados")
    if official_geometry > 0:
        return "SUSC-17C2 Official Geometry Acquisition para recuperar geometria oficial fina"
    return "SUSC-17C2 Official Geometry Acquisition"


def build_report() -> int:
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    priority = read_csv(PRIORITY_CANARY)
    bullets = "\n".join(
        f"{r['priority_rank']}. `{r['candidate_event_id']}` ({r['city']}, {r['event_date']}, {r['source_id']}) "
        f"-> {r['expected_strong_class']}; {r['next_action']}"
        for r in priority)
    report = f"""# SUSC-17C - Strong Reference Acquisition Canary (review-only)

Status: review-only. `trainable=false`; `ground_truth=false`; `score_v6_changed=false`; `score_v7_created=false`; `strong_reference_created_count=0`; `technical_footprint_created_count=0`; `readiness_status={s['readiness_status']}`.

{MANDATORY}

## 1. Por que o 17B estava bloqueado apos o 17A

O 17A consolidou 65 referencias fortes patch-linked, mas vindas de apenas 2 footprints, em Curitiba, com 0 `event_date`. Sem datas nao ha janela pre/pos para avaliacao temporal; sem diversidade de fontes e regioes, um mini-benchmark de evento seria fragil. O 17C ataca exatamente esse gargalo.

## 2. Estrategia: de registros fracos para registros fortes

A esteira liga inventario de fontes -> fila de eventos candidatos -> resolucao de datas -> viabilidade SAR -> priorizacao de 3-5 canaries -> plano Sentinel-1 -> fila de QA humano. Nada vira benchmark, score v7, treino ou ground truth nesta etapa.

## 3. Por que evento oficial datado e a ancora

A data oficial define a janela temporal pre/pos que torna a deteccao de mudanca (SAR) e a avaliacao patch-level possiveis. Sem data, o caso fica bloqueado. Por isso o 17C so promove candidatos com `exact_day` ou `date_range`.

## 4. Por que SAR entra como produtor de footprint tecnico, nao como score

O Sentinel-1/SAR produz uma geometria de inundacao candidata (footprint tecnico) por deteccao de mudanca pre/pos, com mascaras de agua permanente, HAND/slope e guarda de ambiguidade urbana. Esse footprint e evidencia observacional candidata para avaliar o score v6 - nunca um score, nunca ground truth, sempre com QA humano.

## 5. Fontes internas aproveitadas

{s['total_sources']} fontes inventariadas (somente reais). Destaques: International Charter (evento datado 2022-05-24 com geometria oficial coarse), SUSC-16A sentinel event window plan (161 janelas Recife datadas, S1-elegiveis), Defesa Civil, e os artefatos internos 16A/16C/17A. Fontes prioritarias ausentes nos artefatos (CEMADEN, APAC, ANA/Hidroweb, S2iD autonomo) ficam como lacuna de aquisicao, nao foram fabricadas.

## 6. Quantos candidatos foram encontrados

{s['total_candidate_events']} eventos candidatos no target pack.

## 7. Quantos tem data utilizavel

{s['date_resolved_count']} com data resolvida (`exact_day`={s['exact_day_count']}, `date_range`={s['date_range_count']}, `month_only`={s['month_only_count']}, `unknown/not_available`={s['unknown_date_count']}).

## 8. Quantos sao viaveis para SAR

{s['sar_candidate_count']} candidatos SAR; {s['official_geometry_count']} com geometria oficial; {s['strong_reference_candidate_count']} candidatos fortes (data + geometria oficial OU data + viabilidade SAR), todos review-only.

## 9. Quais 3-5 eventos foram priorizados

{bullets}

## 10. O que ainda bloqueia o benchmark 17B

{json.dumps(s['blocks_17b'], ensure_ascii=False)}

Nenhum footprint tecnico foi realmente produzido (`technical_footprint_created_count=0`): o SAR foi planejado, nao executado. A diversidade regional dos candidatos fortes ainda e baixa.

## 11. Proximo passo

{s['next_recommended_milestone']}. Se o gargalo for execucao, seguir para `SUSC-17C2 Sentinel-1/SAR Footprint Execution`; se for geometria oficial, seguir para `SUSC-17C2 Official Geometry Acquisition`; `SUSC-17B Event-Based Mini-Benchmark` so quando houver datas, diversidade e patch-links fortes suficientes.
"""
    write_markdown(REPORT, report)
    print(f"report -> {_rel(REPORT)}")
    return 0


def run_all() -> int:
    build_source_inventory()
    build_target_pack_and_dates()
    build_sar_feasibility()
    priority_rows = build_priority_canary()
    build_sentinel_task_plan(priority_rows)
    build_human_qa_queue(priority_rows)
    build_summary(priority_rows)
    build_report()
    return 0


# --------------------------------------------------------------------------- #
# Validation (fail-closed)
# --------------------------------------------------------------------------- #

def schema_violations(row: dict, schema: dict) -> list[str]:
    out = []
    props = schema.get("properties", {})
    for field in schema.get("required", []):
        if not str(row.get(field, "")).strip():
            out.append(f"required empty: {field}")
    for field, spec in props.items():
        if field not in row:
            continue
        value = row.get(field, "")
        if "const" in spec and value != spec["const"]:
            out.append(f"{field}={value} must be {spec['const']}")
        if "enum" in spec and value not in spec["enum"]:
            out.append(f"{field}={value} not in enum")
    return out


def target_row_violations(row: dict) -> list[str]:
    """Fail-closed guard for a candidate-event row (used by validator + tests)."""
    out = []
    if row.get("review_only") != "true":
        out.append("review_only not true")
    if row.get("trainable") != "false":
        out.append("trainable true")
    if row.get("ground_truth") != "false":
        out.append("ground_truth true")
    if row.get("eligible_for_score_v7_future", "false") == "true":
        out.append("eligible_for_score_v7_future true")
    if row.get("score_v7_created", "false") == "true":
        out.append("score_v7_created true")
    if row.get("official_score_changed", "false") == "true":
        out.append("official_score_changed true")
    precision = row.get("event_date_precision", "")
    occ = row.get("occurrence_status", "")
    has_geom = row.get("has_official_geometry") == "true"
    sar = row.get("sar_candidate") == "true"
    strong = row.get("strong_reference_candidate") == "true"
    # rule 11: no date -> no SAR candidate
    if sar and precision not in SAR_PRECISIONS:
        out.append("sar_candidate without usable date")
    # rule 12: month_only/unknown/not_available never strong
    if strong and precision not in SAR_PRECISIONS:
        out.append("strong candidate without usable date")
    # rule 13: alert never occurrence
    if occ == "alert" and row.get("occurrence_status") == "observed_occurrence":
        out.append("alert marked as occurrence")
    # rules 13/14: alert/risk never produce strong/sar/occurrence
    if occ in ("alert", "risk_area") and (sar or strong):
        out.append(f"{occ} produced sar/strong candidate")
    # rule 16: administrative without fine geometry never strong
    if occ == "administrative" and strong and not has_geom:
        out.append("administrative without fine geometry marked strong")
    # rule 18: strong requires (date+official geometry) OR (date+sar feasibility)
    if strong:
        dated = precision in SAR_PRECISIONS
        if not (dated and (has_geom or sar)):
            out.append("strong candidate lacks date+geometry or date+sar")
    # rule 17: technical footprint only as expected output, never ground truth here
    if row.get("expected_strong_class") == "technical_remote_sensing_flood_footprint" \
            and row.get("ground_truth") != "false":
        out.append("technical footprint marked ground truth")
    return out


def validate() -> int:
    errors: list[str] = []
    required_files = [
        AUDIT, CANDIDATE_SCHEMA, SAR_SCHEMA, SOURCE_INVENTORY, TARGET_PACK,
        DATE_RESOLUTION, SAR_FEASIBILITY, PRIORITY_CANARY, SENTINEL_TASK_PLAN,
        HUMAN_QA_QUEUE, SUMMARY, REPORT,
    ]
    for path in required_files:
        if not path.exists():
            errors.append(f"missing: {path.relative_to(ROOT)}")
    if errors:
        print("FAILED: SUSC-17C validation")
        for error in errors:
            print("  ERROR:", error)
        return 1

    candidate_schema = json.loads(CANDIDATE_SCHEMA.read_text(encoding="utf-8"))
    sar_schema = json.loads(SAR_SCHEMA.read_text(encoding="utf-8"))
    targets = read_csv(TARGET_PACK)
    sar_rows = read_csv(SAR_FEASIBILITY)
    priority = read_csv(PRIORITY_CANARY)
    qa = read_csv(HUMAN_QA_QUEUE)

    # unique + deterministic IDs
    for label, rows, key in (
        ("target", targets, "candidate_event_id"),
        ("sar", sar_rows, "candidate_event_id"),
        ("priority", priority, "candidate_event_id"),
        ("qa", qa, "qa_item_id"),
    ):
        ids = [r[key] for r in rows]
        if len(ids) != len(set(ids)):
            errors.append(f"{label}: ids not unique")
        if ids != sorted(ids):
            errors.append(f"{label}: ids not sorted (non-deterministic)")

    for line_no, row in enumerate(targets, start=2):
        for v in schema_violations(row, candidate_schema):
            errors.append(f"{TARGET_PACK.name}:{line_no}: {v}")
        for v in target_row_violations(row):
            errors.append(f"{TARGET_PACK.name}:{line_no}: {v}")

    for line_no, row in enumerate(sar_rows, start=2):
        for v in schema_violations(row, sar_schema):
            errors.append(f"{SAR_FEASIBILITY.name}:{line_no}: {v}")
        # rule 11: SAR feasible only with usable date/windows
        if row.get("sar_feasibility_status") == "feasible" and row.get("pre_event_window") == "not_available":
            errors.append(f"{SAR_FEASIBILITY.name}:{line_no}: feasible without window")
        if row.get("expected_output_class") not in ("technical_remote_sensing_flood_footprint", "blocked"):
            errors.append(f"{SAR_FEASIBILITY.name}:{line_no}: bad expected_output_class")
        if row.get("ground_truth") != "false":
            errors.append(f"{SAR_FEASIBILITY.name}:{line_no}: ground_truth true")

    # priority canary: 3..5, all viable or blocked-with-reason, all have QA item
    if priority and not (3 <= len(priority) <= 5):
        errors.append(f"priority canary count {len(priority)} outside 3..5")
    qa_ids = {q["candidate_event_id"] for q in qa}
    target_by_id = {t["candidate_event_id"]: t for t in targets}
    for row in priority:
        eid = row["candidate_event_id"]
        if eid not in qa_ids:
            errors.append(f"priority {eid} has no QA item")  # rule 19
        if row.get("qa_required") != "true":
            errors.append(f"priority {eid} qa_required not true")
        tgt = target_by_id.get(eid, {})
        if tgt.get("sar_candidate") != "true" and not row.get("blocking_risk"):
            errors.append(f"priority {eid} not viable and no blocking risk")

    # QA queue: technical footprint never produced without QA (rule 18 of validator spec)
    for row in qa:
        if row.get("default_status") != "needs_review":
            errors.append("qa default_status not needs_review")
        if row.get("ground_truth") != "false":
            errors.append("qa ground_truth true")

    # sentinel task plan + summary governance
    plan = json.loads(SENTINEL_TASK_PLAN.read_text(encoding="utf-8"))
    if plan.get("ground_truth") is not False or plan.get("score_v7_created") is not False:
        errors.append("task plan governance flag wrong")
    if plan.get("technical_footprint_created_count") not in (0, "0"):
        errors.append("task plan created a technical footprint (should be 0)")
    for task in plan.get("tasks", []):
        if task.get("outputs_expected", {}).get("ground_truth") is not False:
            errors.append("task output marked ground truth")
        if task.get("outputs_expected", {}).get("expected_output_class") not in (
                "technical_remote_sensing_flood_footprint", "blocked"):
            errors.append("task expected_output_class invalid")

    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if summary.get("score_v7_created") is not False:
        errors.append("summary score_v7_created true")
    if summary.get("score_v6_changed") is not False:
        errors.append("summary score_v6_changed true")
    if summary.get("trainable") is not False or summary.get("ground_truth") is not False:
        errors.append("summary trainable/ground_truth wrong")
    if summary.get("strong_reference_created_count") not in (0, "0"):
        errors.append("summary strong_reference_created_count != 0")
    if summary.get("technical_footprint_created_count") not in (0, "0"):
        errors.append("summary technical_footprint_created_count != 0")
    if summary.get("readiness_status") not in (
            "STRONG_REFERENCE_ACQUISITION_READY", "SAR_CANARY_READY",
            "BLOCKED_NEEDS_DATED_EVENTS", "BLOCKED_NEEDS_GEOMETRY",
            "BLOCKED_INSUFFICIENT_DIVERSITY", "BLOCKED_NO_CANARY_EVENTS"):
        errors.append("summary readiness_status invalid")
    if summary.get("total_candidate_events") != len(targets):
        errors.append("summary total_candidate_events mismatch")
    if summary.get("priority_canary_count") != len(priority):
        errors.append("summary priority_canary_count mismatch")
    # rule 20: if no candidates, must mark blocked honestly (not fabricate)
    if not priority and not str(summary.get("readiness_status", "")).startswith("BLOCKED"):
        errors.append("no canaries but readiness not BLOCKED")

    # cross guards shared with the SUSC chain
    if SCORE_V7.exists():
        errors.append("score v7 artifact exists")
    if not SCORE_V6.exists():
        errors.append("score v6 official source missing")
    if MANDATORY not in REPORT.read_text(encoding="utf-8"):
        errors.append("mandatory report phrase missing")

    if errors:
        print("FAILED: SUSC-17C validation")
        for error in errors:
            print("  ERROR:", error)
        return 1
    print("PASSED: SUSC-17C validations passed")
    return 0
