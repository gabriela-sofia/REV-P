"""SUSC-15A FASE 2 - audit official occurrence address content."""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
import susc_14a_geocoding_common as g14  # noqa: E402
from susc_io import read_csv, write_csv, write_markdown  # noqa: E402

DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
OCC = DAT / "susc_14a_geocoded_official_occurrences_v1.csv"
AUDIT = OUT / "SUSC_15A_official_occurrence_address_content_audit.csv"
REPORT = OUT / "SUSC_15A_official_occurrence_address_content_report.md"

FIELDS = [
    "audit_id", "source_path", "source_kind", "row_count", "columns", "address_columns",
    "has_address", "has_number", "has_neighborhood", "has_complement",
    "has_spatial_text_reference", "has_cross_street", "has_canal_rio_ponte",
    "has_hidden_latlon", "has_hidden_xy_utm", "has_street_or_cadastral_code",
    "has_joinable_official_id", "precision_rank", "example_event_id", "review_only",
]


def _read_header(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            return next(csv.reader(handle))
    except (OSError, StopIteration):
        return []


def _b(v: bool) -> str:
    return str(bool(v)).lower()


def _number(text: str) -> str:
    match = re.search(r"\b(?:n|no|num|numero)?\s*(\d{1,6}[a-z]?)\b", g14.normalize_text(text))
    return match.group(1) if match else ""


def _cross(text: str) -> str:
    low = g14.normalize_text(text)
    return _b(any(x in low for x in ("esquina com", "cruzamento com", " com rua ", " proximo ao cruzamento")))


def _rank(row: dict) -> str:
    raw = row.get("raw_address", "")
    street = row.get("normalized_street", "")
    hood = row.get("normalized_neighborhood", "")
    has_street = bool(street)
    has_num = bool(_number(raw))
    has_cross = _cross(raw) == "true"
    low = g14.normalize_text(raw)
    landmark = any(x in low for x in ("proximo", "em frente", "ao lado", "ponte", "canal", "rio"))
    if has_street and has_num:
        return "has_street_and_number"
    if has_street and has_cross:
        return "has_street_and_cross_street"
    if has_street and landmark:
        return "has_street_and_landmark"
    if has_street and hood:
        return "has_street_and_neighborhood"
    if hood:
        return "neighborhood_only"
    return "insufficient_address"


def _source_row(path: Path, source_kind: str, rows: list[dict] | None = None) -> dict:
    rows = rows if rows is not None else (read_csv(path) if path.exists() else [])
    header = list(rows[0].keys()) if rows else _read_header(path)
    low_cols = [g14.normalize_text(c) for c in header]
    joined_cols = " ".join(low_cols)
    address_cols = [c for c in header if any(k in g14.normalize_text(c) for k in ("endereco", "logradouro", "rua", "address", "local"))]
    sample_text = " ".join(" ".join(str(v) for v in row.values()) for row in rows[:200])
    sample_low = g14.normalize_text(sample_text)
    ranks = Counter(_rank(row) for row in rows if "raw_address" in row)
    return {
        "audit_id": "",
        "source_path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "source_kind": source_kind,
        "row_count": len(rows),
        "columns": json.dumps(header, ensure_ascii=False),
        "address_columns": json.dumps(address_cols, ensure_ascii=False),
        "has_address": _b(bool(address_cols) or any(k in sample_low for k in ("rua", "avenida", "travessa", "estrada"))),
        "has_number": _b(bool(re.search(r"\b\d{1,6}[a-z]?\b", sample_low))),
        "has_neighborhood": _b("bairro" in joined_cols or "normalized_neighborhood" in header or "raw_neighborhood" in header),
        "has_complement": _b(any(k in sample_low for k in ("lote", "quadra", "bloco", "casa", "apto", "proximo", "frente"))),
        "has_spatial_text_reference": _b(any(k in sample_low for k in ("ponte", "canal", "rio", "cruzamento", "esquina"))),
        "has_cross_street": _b(any(k in sample_low for k in ("esquina com", "cruzamento com", " com rua "))),
        "has_canal_rio_ponte": _b(any(k in sample_low for k in ("canal", "rio", "ponte", "riacho"))),
        "has_hidden_latlon": _b(any(k in joined_cols for k in ("latitude", "longitude", "lat", "lon", "lng"))),
        "has_hidden_xy_utm": _b(any(k in joined_cols for k in ("utm", "coord_x", "coord_y", "x", "y"))),
        "has_street_or_cadastral_code": _b(any(k in joined_cols for k in ("codigo", "codlog", "cod_logradouro", "inscricao", "cadastro"))),
        "has_joinable_official_id": _b(any(k in joined_cols for k in ("id", "codigo", "protocolo", "processo"))),
        "precision_rank": ranks.most_common(1)[0][0] if ranks else "source_level_inventory",
        "example_event_id": next((row.get("occurrence_id", "") for row in rows if row.get("occurrence_id")), ""),
        "review_only": "true",
    }


def main() -> int:
    print("=" * 60)
    print("SUSC-15A Official Occurrence Address Content Audit")
    print("=" * 60)
    occs = read_csv(OCC) if OCC.exists() else []
    rows = [_source_row(OCC, "official_occurrence_catalog", occs)]
    rank_counts = Counter()
    example_by_rank = {}
    for occ in occs:
        rank = _rank(occ)
        rank_counts[rank] += 1
        example_by_rank.setdefault(rank, occ.get("occurrence_id", ""))
    for rank, count in rank_counts.most_common():
        rows.append({
            "audit_id": "", "source_path": str(OCC.relative_to(ROOT)).replace("\\", "/"),
            "source_kind": "precision_rank", "row_count": count, "columns": "[]",
            "address_columns": "[]", "has_address": "true", "has_number": _b(rank == "has_street_and_number"),
            "has_neighborhood": _b(rank != "insufficient_address"), "has_complement": "false",
            "has_spatial_text_reference": _b("landmark" in rank or "cross" in rank),
            "has_cross_street": _b("cross" in rank), "has_canal_rio_ponte": "false",
            "has_hidden_latlon": "false", "has_hidden_xy_utm": "false",
            "has_street_or_cadastral_code": "false", "has_joinable_official_id": "true",
            "precision_rank": rank, "example_event_id": example_by_rank.get(rank, ""),
            "review_only": "true",
        })
    for path in sorted(DAT.glob("susc_14*_*.csv")):
        if path == OCC:
            continue
        rows.append(_source_row(path, "susc_prior_stage_dataset"))
    for idx, row in enumerate(rows):
        row["audit_id"] = f"S15AAUD_{idx:04d}"
    write_csv(AUDIT, rows, FIELDS)
    write_markdown(REPORT, f"""# SUSC-15A - auditoria de conteudo de endereco

Status: review-only. A auditoria nao cria coordenadas, ground truth, treino ou score v7.

- Ocorrencias oficiais avaliadas: **{len(occs)}**
- Ranking de precisao: **{dict(rank_counts)}**
- Fontes/tabelas auditadas: **{len(rows)}**

Bairro-only e endereco sem numero/intersecao permanecem insuficientes para
calibracao fina. Numeros e referencias textuais sao tratados como candidatos
para cruzamento com bases oficiais, nunca como autorizacao para geocoding generico.
""")
    print(f"audit rows: {len(rows)} | ranks: {dict(rank_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
