"""
SUSC-13C-LIVE universal parser of live-acquired official sources.

Reuses the SUSC-13B universal parser primitives on the files acquired live into
the SUSC-13C directory (CKAN/ArcGIS/WFS/HTML). Fails closed: risk, alert,
administrative and documentary context never become observed events; strong
levels require explicit geometry plus date/period.

Writes:
  - datasets/suscetibilidade/susc_13c_live_observed_events_parsed_v1.csv
  - outputs_public/suscetibilidade/SUSC_13C_live_parse_audit.csv
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
import parse_susc_13b_auto_observed_events as p13b  # noqa: E402
from susc_io import read_csv, rel, write_csv  # noqa: E402

DROP = ROOT / "datasets" / "suscetibilidade" / "observed_event_sources_susc13c_live"
DL_MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_13c_live_download_manifest_v1.csv"
PARSED = ROOT / "datasets" / "suscetibilidade" / "susc_13c_live_observed_events_parsed_v1.csv"
AUDIT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_13C_live_parse_audit.csv"

OBSERVED_LEVELS = {"strong_observed_flood_polygon", "strong_observed_flood_point",
                   "moderate_official_occurrence_point", "moderate_official_flood_bbox"}
# A source only becomes an observed FLOOD event if its text carries a real flood
# term. Broad discovery keywords (e.g. "ocorrencia", "drenagem") match unrelated
# datasets (diamond occurrences, drainage geochemistry); those are downgraded.
FLOOD_TERMS = ("alagament", "inundac", "inundaç", "enchente", "enxurrada",
               "transbordament", "area alagada", "área alagada", "mancha de inund",
               "flood", "alagad")


def _is_flood_relevant(row: dict) -> bool:
    text = " ".join([row.get("source_title", ""), row.get("source_url", ""),
                     row.get("local_file_path", ""), row.get("notes", "")]).lower()
    return any(t in text for t in FLOOD_TERMS)


def _downgrade_non_flood(row: dict) -> dict:
    """Keep fail-closed: an observed level without explicit flood semantics is not
    a flood event. Downgrade to documentary_only (real dataset, not flood)."""
    if row.get("evidence_level") in OBSERVED_LEVELS and not _is_flood_relevant(row):
        row["evidence_level"] = "documentary_only"
        row["is_observed_event"] = "false"
        row["is_risk_area"] = "false"
        row["is_alert"] = "false"
        row["is_administrative_only"] = "false"
        row["source_confidence"] = "low"
        row["classification_reason"] = ("rebaixado: casou apenas keyword generica de descoberta "
                                        "(ex.: ocorrencia/drenagem) sem termo explicito de "
                                        "alagamento/inundacao; nao e evento de cheia")
    return row


def main() -> int:
    print("=" * 60)
    print("SUSC-13C-LIVE Universal Observed Event Parser")
    print("=" * 60)
    audit: list[dict] = []
    rows: list[dict] = []

    file_meta: dict[str, dict] = {}
    if DL_MANIFEST.exists():
        for r in read_csv(DL_MANIFEST):
            fp = (r.get("file_path") or "").strip()
            if fp:
                file_meta[fp] = r

    files = sorted(p for p in DROP.rglob("*") if p.is_file() and p.name != "README.md") if DROP.exists() else []
    for p in files:
        meta = file_meta.get(rel(p), {})
        # event_id prefix must read SUSC13C; p13b._parse_file emits SUSC13B_ -> remap below
        new = p13b._parse_file(
            p, meta.get("download_id", "LIVE"), meta.get("source_url", rel(p)),
            meta.get("institution", p.name), meta.get("content_ext", p.suffix.lstrip(".")),
            audit, len(rows))
        for row in new:
            row["event_id"] = row["event_id"].replace("SUSC13B_", "SUSC13C_")
            row["notes"] = "parseado do SUSC-13C live; " + row.get("notes", "")
            _downgrade_non_flood(row)
        rows.extend(new)

    if not files:
        audit.append(p13b._audit("", "", "", "no_files_to_parse", "no_observed_events",
                                 0, 0, "", "diretorio de aquisicao live vazio"))

    write_csv(PARSED, rows, p13b.FIELDS)
    write_csv(AUDIT, audit, p13b.AUDIT_FIELDS)
    print(f"parsed rows: {len(rows)}")
    print("by evidence_level:", dict(Counter(r["evidence_level"] for r in rows)) or "none")
    print(f"files seen: {len(files)} | audit rows: {len(audit)}")
    print("review-only. No ground truth. No training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
