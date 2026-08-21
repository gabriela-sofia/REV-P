"""Shared helpers for SUSC-15B forensic precision rescue.

The SUSC-15B line is deliberately fail-closed: it audits existing public
artifacts and records manual/deep-source blockers, but does not infer
coordinates, promote bairro-only evidence, create ground truth, train models, or
create score v7.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
MAN = ROOT / "manifests" / "suscetibilidade"

GEOCODED_15A = DAT / "susc_15a_precision_geocoded_occurrences_v1.csv"
LINKAGE_15A = DAT / "susc_15a_precision_event_patch_linkage_v1.csv"
SUMMARY_15A = OUT / "SUSC_15A_precision_score_event_diagnostics_summary.json"

HIDDEN_AUDIT = OUT / "SUSC_15B_hidden_coordinate_and_id_audit.csv"
DISCOVERY_REPORT = OUT / "SUSC_15B_nonobvious_layer_discovery_report.md"
DOWNLOAD_MANIFEST = OUT / "SUSC_15B_precision_source_download_manifest.csv"
PARSE_AUDIT = OUT / "SUSC_15B_precision_source_parse_audit.csv"
JOIN_AUDIT = OUT / "SUSC_15B_occurrence_id_official_join_audit.csv"
CATALOG = DAT / "susc_15b_precision_event_catalog_v1.csv"
LINKAGE = DAT / "susc_15b_precision_event_patch_linkage_v1.csv"
READINESS = OUT / "SUSC_15B_precision_readiness.md"
REPORT = OUT / "SUSC_15B_forensic_precision_rescue_report.md"

PRECISE_TIERS = {
    "T0_official_event_polygon",
    "T1_official_event_point",
    "T2_official_address_point_or_parcel",
    "T3_official_intersection_point",
    "T4_official_house_number_linear_reference",
}
WEAK_TIERS = {
    "T5_official_street_segment_candidate",
    "T6_neighborhood_context_only",
    "T7_rejected_or_non_flood",
}
MANDATORY = (
    "SUSC-15B e review-only: nao aceita bairro-only como sucesso, nao aceita "
    "street segment sem numero/intersecao como calibracao, nao cria ground "
    "truth, nao libera treino supervisionado e nao cria score v7 automatico."
)


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    ensure_dir(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_markdown(path: Path, text: str) -> None:
    ensure_dir(path)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def has_geometry(row: dict) -> bool:
    return bool(
        (row.get("lat") and row.get("lon"))
        or row.get("wkt")
        or row.get("bbox")
        or row.get("geojson_ref")
    )


def is_calibration_eligible(row: dict) -> bool:
    tier = row.get("precision_tier", "")
    if tier not in PRECISE_TIERS or not has_geometry(row):
        return False
    if tier == "T4_official_house_number_linear_reference":
        try:
            return float(row.get("uncertainty_m") or 9999) <= 100 and row.get("is_ambiguous") != "true"
        except ValueError:
            return False
    return True


def tier_counts(rows: list[dict]) -> Counter:
    return Counter(row.get("precision_tier", "") for row in rows)


def summary_15a() -> dict:
    summary = read_json(SUMMARY_15A)
    geocoded = read_csv(GEOCODED_15A)
    links = read_csv(LINKAGE_15A)
    eval_links = [row for row in links if row.get("can_be_used_for_observational_evaluation") == "true"]
    return {
        "precision_events_total": summary.get("precision_events_total", len(geocoded)),
        "calibration_eligible_events": summary.get("calibration_eligible_events", 0),
        "precision_patch_links": summary.get("precision_patch_links", len(eval_links)),
        "unique_patches_observational": summary.get(
            "unique_patches_observational",
            len({row.get("patch_id", "") for row in eval_links if row.get("patch_id")}),
        ),
        "ready_for_score_v7": bool(summary.get("ready_for_score_v7", False)),
        "score_v7_created": bool(summary.get("score_v7_created", False)),
    }
