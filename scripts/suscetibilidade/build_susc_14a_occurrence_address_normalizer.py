"""
SUSC-14A FASE 5a - normalize official occurrence records that lack coordinates.

Reads the official occurrence tables already acquired by SUSC-13C (Defesa Civil
Recife ``atendimentos-YYYY.csv``: date / type / address / neighborhood, but no
lat/lon) plus the SUSC-13C consolidated catalog records that are text-only / no
geometry. Keeps only genuine flood occurrences (alagamento / inundacao /
enchente / enxurrada / transbordamento); structural inspections, landslide-only,
social registry, alerts and prevention are rejected (blocked_non_flood). Emits a
light normalized occurrence table (the heavy raw CSVs are never committed).

Writes:
  - datasets/suscetibilidade/susc_14a_normalized_official_occurrences_v1.csv
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
import susc_14a_geocoding_common as g14  # noqa: E402
from susc_io import read_csv, rel, write_csv  # noqa: E402

LOCAL_DIR = ROOT / "datasets" / "suscetibilidade" / "observed_event_sources_susc13c_live"
CATALOG_13C = ROOT / "datasets" / "suscetibilidade" / "susc_13c_consolidated_observed_event_catalog_v1.csv"
PARSED_13C = ROOT / "datasets" / "suscetibilidade" / "susc_13c_live_observed_events_parsed_v1.csv"
OUT = ROOT / "datasets" / "suscetibilidade" / "susc_14a_normalized_official_occurrences_v1.csv"

FIELDS = [
    "occurrence_id", "region", "municipality", "source_file", "source_lineage",
    "event_date", "event_year", "raw_occurrence_text", "raw_address",
    "raw_neighborhood", "normalized_street", "normalized_neighborhood",
    "is_flood_occurrence", "rejection_reason", "requires_manual_review",
    "can_be_ground_truth", "allowed_for_training", "review_only",
]

DATE_RE = re.compile(r"(20\d{2})[-/](\d{1,2})[-/](\d{1,2})")
YEAR_RE = re.compile(r"(20\d{2})")
MAX_FLOOD_ROWS = 20000  # cap on emitted flood occurrences (keeps artifact light)


def _norm_date(raw: str) -> tuple[str, str]:
    m = DATE_RE.search(raw or "")
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}", m.group(1)
    y = YEAR_RE.search(raw or "")
    return "", (y.group(1) if y else "")


def _gov(row: dict) -> dict:
    row["requires_manual_review"] = "true"
    row["can_be_ground_truth"] = "false"
    row["allowed_for_training"] = "false"
    row["review_only"] = "true"
    return row


def _file_year(path: Path) -> str:
    m = YEAR_RE.search(path.stem)
    return m.group(1) if m else ""


def _from_atendimentos(path: Path, region: str) -> tuple[list[dict], int, int]:
    rows = g14.read_brazilian_csv(path)
    file_year = _file_year(path)
    kept: list[dict] = []
    rejected = 0
    for r in rows:
        occ = g14.header_pick(r, "ocorrencia", "tipo_da_acao", "solicitacao")
        tipo = g14.header_pick(r, "tipo_da_acao", "ocorrencia")
        addr = g14.header_pick(r, "endereco", "logradouro")
        hood = g14.header_pick(r, "bairro", "localidade", "regional")
        date_raw = g14.header_pick(r, "data", "data_da_acao")
        text = " ".join([occ, tipo, g14.header_pick(r, "solicitacao")])
        if not g14.is_flood_text(text, addr):
            rejected += 1
            continue
        edate, eyear = _norm_date(date_raw)
        # authoritative year fallback is the file name (atendimentos-YYYY.csv);
        # never let a month name leak into event_year.
        ano_col = g14.header_pick(r, "ano")
        if not eyear:
            eyear = ano_col if YEAR_RE.fullmatch(ano_col or "") else file_year
        kept.append(_gov({
            "region": region, "municipality": g14.MUNICIPALITY.get(region, region),
            "source_file": rel(path), "source_lineage": "defesa_civil_recife_atendimentos",
            "event_date": edate, "event_year": eyear,
            "raw_occurrence_text": text.strip()[:160], "raw_address": addr[:160],
            "raw_neighborhood": hood[:80], "normalized_street": g14.normalize_street(addr),
            "normalized_neighborhood": g14.normalize_text(hood),
            "is_flood_occurrence": "true", "rejection_reason": "",
        }))
    return kept, rejected, len(rows)


def _from_catalog_text_only(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for r in rows:
        if r.get("preferred_record", "true") == "false":
            continue
        has_geom = bool(r.get("bbox") or (r.get("lat") and r.get("lon")) or r.get("wkt") or r.get("geojson_ref"))
        if has_geom:
            continue  # already has geometry; not a geocoding target
        title = r.get("source_title", "")
        if not g14.is_flood_text(title, r.get("event_type", "")):
            continue
        out.append(_gov({
            "region": (r.get("region") or "").lower(),
            "municipality": r.get("municipality", ""),
            "source_file": r.get("local_file_path", ""), "source_lineage": "susc_13c_catalog_text_only",
            "event_date": r.get("event_date", ""),
            "event_year": (r.get("event_date", "")[:4] or r.get("event_period_start", "")[:4]),
            "raw_occurrence_text": title[:160], "raw_address": "",
            "raw_neighborhood": "", "normalized_street": "", "normalized_neighborhood": "",
            "is_flood_occurrence": "true", "rejection_reason": "",
        }))
    return out


def main() -> int:
    print("=" * 60)
    print("SUSC-14A Occurrence Address Normalizer")
    print("=" * 60)
    occurrences: list[dict] = []
    total_rows = 0
    total_rejected = 0

    if LOCAL_DIR.exists():
        for p in sorted(LOCAL_DIR.rglob("*.csv")):
            low = g14.normalize_text(p.name)
            if "atendimento" not in low and "ocorrencia" not in low and "alagament" not in low:
                continue
            region = "recife"  # Defesa Civil Recife atendimentos
            kept, rej, nrows = _from_atendimentos(p, region)
            occurrences.extend(kept)
            total_rows += nrows
            total_rejected += rej
            print(f"  {p.name}: {nrows} linhas -> {len(kept)} cheia, {rej} nao-cheia")

    if CATALOG_13C.exists():
        occurrences.extend(_from_catalog_text_only(read_csv(CATALOG_13C)))

    if len(occurrences) > MAX_FLOOD_ROWS:
        occurrences = occurrences[:MAX_FLOOD_ROWS]

    for i, o in enumerate(occurrences):
        o["occurrence_id"] = f"S14AOCC_{i:06d}"
    occurrences = [{k: o.get(k, "") for k in FIELDS} for o in occurrences]
    write_csv(OUT, occurrences, FIELDS)

    by_region = Counter(o["region"] for o in occurrences)
    by_year = Counter(o["event_year"] for o in occurrences if o["event_year"])
    print(f"flood occurrences kept: {len(occurrences)} | rejected non-flood: {total_rejected} of {total_rows} rows")
    print(f"by region: {dict(by_region)}")
    print(f"by year: {dict(sorted(by_year.items()))}")
    print(f"-> {rel(OUT)}")
    print("review-only. No ground truth. No training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
