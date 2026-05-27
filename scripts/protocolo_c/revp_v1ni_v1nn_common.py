"""Shared metadata-only helpers for Protocol C v1ni-v1nn."""

from __future__ import annotations

import csv
import hashlib
import os
import re
from pathlib import Path
from typing import Any

from revp_v1lj_v1lq_common import DATASETS, DOCS, LOCAL_ROOT, SCHEMAS, read_csv, write_csv, write_schema


INBOX = Path(os.environ.get("REVP_OFFICIAL_NEGATIVE_INBOX", str(LOCAL_ROOT / "official_negative_response_inbox")))

ABSOLUTE_PATH_RE = re.compile(r"[A-Za-z]:[\\/]|\\\\")
DATE_RE = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-](?:20)?\d{2}|20\d{2})\b")
LOCALITY_RE = re.compile(
    r"\b(?:petropolis|petr[oó]polis|moinho preto|alto da serra|quitandinha|mosel[la]|rua teresa|"
    r"valparaiso|valpara[ií]so|serra velha|sargento boening|vila felipe|bairro|rua|avenida|estrada)\b",
    re.IGNORECASE,
)
PHENOMENON_RE = re.compile(
    r"\b(?:deslizamento|escorregamento|movimento de massa|instabilidade|risco geol[oó]gico|"
    r"dano geol[oó]gico|encosta|barreira)\b",
    re.IGNORECASE,
)
NEGATIVE_RE = re.compile(
    r"\b(?:sem ocorr[eê]ncia|sem risco|sem instabilidade|sem dano geol[oó]gico|sem indicio|sem ind[ií]cio|"
    r"aus[eê]ncia de|n[aã]o foram observad[ao]s|area est[aá]vel|[aá]rea est[aá]vel|estabilidade)\b",
    re.IGNORECASE,
)
SOURCE_RE = re.compile(r"\b(?:defesa civil|prefeitura|sgb|cprm|drm|org[aã]o|secretaria|coordenadoria|laudo|auto|vistoria)\b", re.IGNORECASE)
ADDRESS_RE = re.compile(r"\b(?:rua|avenida|av\.|estrada|travessa|servid[aã]o|bairro|coordenad[ao]|lat(?:itude)?|lon(?:gitude)?)\b", re.IGNORECASE)


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_join(values: list[str]) -> str:
    clean = [str(value).strip() for value in values if str(value).strip()]
    return ";".join(clean)


def require_no_public_abs_paths(paths: list[Path]) -> None:
    offenders: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if ABSOLUTE_PATH_RE.search(text):
            offenders.append(str(path.relative_to(DATASETS.parent)))
    if offenders:
        raise RuntimeError("public output contains absolute path: " + ";".join(offenders))


def read_event_rows() -> list[dict[str, str]]:
    rows = read_csv(DATASETS / "ground_reference_event_registry.csv")
    return [row for row in rows if row.get("c_level") == "C3_EVENT_PATCH_LINKED" or row.get("can_be_ground_reference_event") == "true"]


def read_gap_rows() -> list[dict[str, str]]:
    rows = read_csv(DATASETS / "c4_gate_gap_analysis_registry.csv")
    return [row for row in rows if row.get("requires_negative_evidence") == "true" or "FORMAL_NEGATIVES_ZERO" in row.get("c4_blocking_reason", "")]


def read_linkage_by_event() -> dict[str, dict[str, str]]:
    return {row.get("event_id", ""): row for row in read_csv(DATASETS / "event_patch_linkage_registry.csv")}


def c3_event_count() -> int:
    rows = read_event_rows()
    return len(rows) if rows else 9


def formal_negative_count_from_adjudication() -> int:
    rows = read_csv(DATASETS / "strict_formal_negative_adjudication_registry.csv")
    return sum(1 for row in rows if row.get("can_be_formal_negative") == "true")


def gazette_formal_negative_count() -> int:
    rows = read_csv(DATASETS / "protocol_c_gazette_negative_resolution_summary.csv")
    if not rows:
        return 0
    try:
        return int(rows[0].get("formal_negative_count", "0") or "0")
    except ValueError:
        return 0


def write_doc(path: Path, title: str, paragraphs: list[str]) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    body = "# " + title + "\n\n" + "\n\n".join(paragraphs).strip() + "\n"
    path.write_text(body, encoding="utf-8")


def write_outputs(rows_by_path: list[tuple[Path, list[dict[str, Any]], list[str]]], schemas: list[tuple[Path, list[str], str]], docs: list[Path]) -> None:
    for path, rows, fields in rows_by_path:
        write_csv(path, rows, fields)
    for path, fields, prefix in schemas:
        write_schema(path, fields, prefix)
    require_no_public_abs_paths([path for path, _, _ in rows_by_path] + [path for path, _, _ in schemas] + docs)


def public_inbox_label(path: Path) -> str:
    return "official_negative_response_inbox/" + path.name


def lightweight_text(path: Path) -> str:
    if path.suffix.lower() in {".txt", ".csv", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")[:200_000]
    if path.suffix.lower() == ".pdf":
        return path.read_bytes()[:200_000].decode("latin-1", errors="ignore")
    return ""


def prevalidate_text(text: str) -> dict[str, str]:
    has_negative = bool(NEGATIVE_RE.search(text))
    has_date = bool(DATE_RE.search(text))
    has_locality = bool(LOCALITY_RE.search(text))
    has_phenomenon = bool(PHENOMENON_RE.search(text))
    has_source = bool(SOURCE_RE.search(text))
    has_address = bool(ADDRESS_RE.search(text))
    if not text.strip():
        status = "INSUFFICIENT_FIELDS"
    elif not has_negative:
        status = "REJECTED_NO_EXPLICIT_NEGATIVE_SEMANTICS"
    elif not (has_locality and has_address):
        status = "REJECTED_NO_SPATIAL_SPECIFICITY"
    elif not has_date:
        status = "REJECTED_NO_TEMPORAL_COMPATIBILITY"
    elif has_negative and has_date and has_locality and has_phenomenon and has_source and has_address:
        status = "POTENTIAL_FORMAL_NEGATIVE_NEEDS_ADJUDICATION"
    else:
        status = "INSUFFICIENT_FIELDS"
    return {
        "contains_explicit_negative_semantics": bool_text(has_negative),
        "contains_date": bool_text(has_date),
        "contains_locality": bool_text(has_locality),
        "contains_phenomenon": bool_text(has_phenomenon),
        "contains_official_source": bool_text(has_source),
        "contains_coordinate_address_or_bairro": bool_text(has_address),
        "prevalidation_status": status,
    }
