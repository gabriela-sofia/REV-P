"""Shared utilities for Protocol C v1na-v1nh gazette negative search."""

from __future__ import annotations

import csv
import hashlib
import html
import math
import re
import urllib.parse
import urllib.request
from datetime import date, datetime
from pathlib import Path
from typing import Any

from revp_v1lj_v1lq_common import (
    DATASETS,
    DOCS,
    LOCAL_ROOT,
    SCHEMAS,
    contains_any,
    fetch_url,
    official_domain,
    public_hash,
    read_csv,
    sanitize_public_text,
    write_csv,
    write_schema,
)
from revp_v1lr_v1lz_common import PHENOMENON_TERMS, extract_pages, split_sentences
from revp_v1mu_v1mz_common import haversine_m


RAW = LOCAL_ROOT / "v1na" / "raw"
BASE = "https://www.petropolis.rj.gov.br"
OLD_GAZETTE_ROOT = BASE + "/pmp/index.php/servicos-cidadao/diario-oficial"
MONTH_CATEGORIES = {
    "janeiro": "271-janeiro",
    "fevereiro": "270-fevereiro",
    "marco": "273-marco",
    "abril": "269-abril",
    "maio": "272-maio",
    "junho": "277-junho",
}
TARGET_START = date(2022, 2, 15)
TARGET_END = date(2022, 4, 30)
EXTENDED_END = date(2022, 6, 30)

GAZETTE_ADMIN_TERMS = [
    "interdicao",
    "interdiÃ§Ã£o",
    "desinterdicao",
    "desinterdiÃ§Ã£o",
    "liberacao",
    "liberaÃ§Ã£o",
    "vistoria",
    "laudo",
    "auto de vistoria",
    "defesa civil",
    "risco geologico",
    "risco geolÃ³gico",
    "habitavel",
    "habitÃ¡vel",
    "habitabilidade",
    "imovel liberado",
    "imÃ³vel liberado",
    "deslizamento",
    "movimento de massa",
    "escorregamento",
    "barreira",
    "encosta",
    "moinho preto",
    "mosela",
    "quitandinha",
    "alto da serra",
    "rua teresa",
    "valparaiso",
    "valparaÃ­so",
    "sargento boening",
    "vila felipe",
    "pontilhao",
    "pontilhÃ£o",
    "serra velha",
]
STRICT_NEGATIVE_TERMS = [
    "sem risco geologico",
    "sem risco geolÃ³gico",
    "sem ocorrencia",
    "sem ocorrÃªncia",
    "sem instabilidade",
    "sem indicio de deslizamento",
    "sem indÃ­cio de deslizamento",
    "sem indicios de deslizamento",
    "sem indÃ­cios de deslizamento",
    "sem indicio de movimento de massa",
    "sem indÃ­cio de movimento de massa",
    "sem dano geologico",
    "sem dano geolÃ³gico",
    "sem dano estrutural",
    "nao foram observadas feicoes",
    "nÃ£o foram observadas feiÃ§Ãµes",
    "ausencia de indicios de movimento de massa",
    "ausÃªncia de indÃ­cios de movimento de massa",
]
RELEASE_TERMS = [
    "desinterdicao",
    "desinterdiÃ§Ã£o",
    "liberacao",
    "liberaÃ§Ã£o",
    "habitavel",
    "habitÃ¡vel",
    "habitabilidade",
]
LOW_RISK_TERMS = ["baixo risco", "risco baixo", "baixa prioridade"]
ADDRESS_TERMS = [" rua ", " avenida ", " av. ", " estrada ", " travessa ", " servidao ", " servidÃ£o ", " praca ", " praÃ§a ", " lote "]
LOCALITY_TERMS = [
    "petropolis",
    "petrÃ³polis",
    "moinho preto",
    "mosela",
    "quitandinha",
    "alto da serra",
    "rua teresa",
    "valparaiso",
    "valparaÃ­so",
    "sargento boening",
    "vila felipe",
    "pontilhao",
    "pontilhÃ£o",
    "serra velha",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_pt_date(text: str) -> date | None:
    text = html.unescape(text)
    match = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](20\d{2}|\d{2})", text)
    if match:
        day, month, year = int(match.group(1)), int(match.group(2)), match.group(3)
        year_i = int(year) if len(year) == 4 else 2000 + int(year)
        try:
            return date(year_i, month, day)
        except ValueError:
            return None
    months = {
        "janeiro": 1,
        "fevereiro": 2,
        "marco": 3,
        "marÃ§o": 3,
        "abril": 4,
        "maio": 5,
        "junho": 6,
    }
    match = re.search(r"(\d{1,2})\s+de\s+([a-zÃ§]+)\s+de\s+(20\d{2})", text.casefold())
    if match and match.group(2) in months:
        try:
            return date(int(match.group(3)), months[match.group(2)], int(match.group(1)))
        except ValueError:
            return None
    return None


def issue_number(text: str) -> str:
    match = re.search(r"\b(?:n[.Âºo]*\s*)?(\d{4})\b", html.unescape(text), flags=re.IGNORECASE)
    return match.group(1) if match else ""


def download_url(path_or_url: str) -> str:
    return urllib.parse.urljoin(BASE, html.unescape(path_or_url).replace("&amp;", "&"))


def month_category_urls(extended: bool = True) -> list[str]:
    names = ["fevereiro", "marco", "abril"]
    if extended:
        names.extend(["janeiro", "maio", "junho"])
    return [f"{OLD_GAZETTE_ROOT}/category/{MONTH_CATEGORIES[name]}" for name in names]


def discover_gazette_issue_links(extended: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for category_url in month_category_urls(extended=extended):
        result = fetch_url(category_url, RAW, len(rows) + 1, timeout=25, max_bytes=4_000_000)
        body = result.get("body_sample", "")
        if result.get("status") == "DOWNLOAD_OK":
            stored = result.get("stored_label", "")
            if stored:
                path = RAW / stored
                try:
                    body = path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    pass
        if result.get("status") != "DOWNLOAD_OK":
            rows.append(
                {
                    "issue_id": f"GAZETTEISSUE_V1NA_INDEX_{public_hash(category_url, 8)}",
                    "issue_date": "",
                    "issue_number": "",
                    "official_domain": official_domain(category_url),
                    "issue_kind": "CATEGORY_INDEX",
                    "source_url_hash": public_hash(category_url),
                    "download_url_hash": "",
                    "issue_discovery_status": result.get("status", "DOWNLOAD_FAIL"),
                    "is_real_issue": "false",
                    "private_path_removed": "true",
                    "_download_url": "",
                }
            )
            continue
        for match in re.finditer(r'href=["\']([^"\']*task=download\.send[^"\']*)["\']', body, flags=re.IGNORECASE):
            href = html.unescape(match.group(1))
            url = download_url(href)
            parsed = urllib.parse.urlparse(url)
            params = urllib.parse.parse_qs(parsed.query.replace("&amp;", "&"))
            did = (params.get("id") or [""])[0]
            catid = (params.get("catid") or [""])[0]
            if not did or not catid or did in seen:
                continue
            snippet = re.sub(r"<[^>]+>", " ", html.unescape(body[max(0, match.start() - 1100) : match.start() + 260]))
            snippet = re.sub(r"\s+", " ", snippet).strip()
            dt = parse_pt_date(snippet)
            if not dt:
                continue
            in_window = TARGET_START <= dt <= (EXTENDED_END if extended else TARGET_END)
            if not in_window:
                continue
            seen.add(did)
            num = issue_number(snippet)
            rows.append(
                {
                    "issue_id": f"GAZETTEISSUE_V1NA_{did}",
                    "issue_date": dt.isoformat(),
                    "issue_number": num,
                    "official_domain": official_domain(url),
                    "issue_kind": "ISSUE_DOWNLOAD_LINK",
                    "source_url_hash": public_hash(category_url),
                    "download_url_hash": public_hash(url),
                    "issue_discovery_status": "DISCOVERED_REAL_ISSUE_LINK",
                    "is_real_issue": "true",
                    "private_path_removed": "true",
                    "_download_url": url,
                }
            )
    if not rows:
        rows.append(
            {
                "issue_id": "GAZETTEISSUE_V1NA_NONE",
                "issue_date": "",
                "issue_number": "",
                "official_domain": "none",
                "issue_kind": "NONE",
                "source_url_hash": public_hash("none"),
                "download_url_hash": "",
                "issue_discovery_status": "NO_REAL_ISSUE_LINK_DISCOVERED",
                "is_real_issue": "false",
                "private_path_removed": "true",
                "_download_url": "",
            }
        )
    return rows


def crawl_and_download_gazettes(extended: bool = True, max_issues: int = 90) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    RAW.mkdir(parents=True, exist_ok=True)
    issue_rows = discover_gazette_issue_links(extended=extended)
    manifest: list[dict[str, Any]] = []
    real = [r for r in issue_rows if r.get("is_real_issue") == "true"][:max_issues]
    for idx, row in enumerate(real, 1):
        url = row.pop("_download_url", "")
        result = fetch_url(url, RAW, 1000 + idx, timeout=35, max_bytes=10_000_000)
        stored = result.get("stored_label", "")
        digest = ""
        size = int(result.get("bytes", 0) or 0)
        if stored:
            src = RAW / stored
            suffix = src.suffix if src.exists() else ".bin"
            dst = RAW / f"gazette_issue_{row['issue_date']}_{row['issue_number'] or idx}_{public_hash(url, 8)}{suffix}"
            if src.exists() and src != dst:
                src.replace(dst)
            if dst.exists():
                digest = sha256_file(dst)
                size = dst.stat().st_size
        manifest.append(
            {
                "download_id": f"GAZETTEDL_V1NA_{idx:04d}",
                "issue_id": row["issue_id"],
                "official_domain": row["official_domain"],
                "url_hash": row["download_url_hash"],
                "file_type": result.get("file_type", "UNKNOWN"),
                "acquisition_status": result.get("status", "DOWNLOAD_FAIL"),
                "byte_count": str(size),
                "sha256": digest[:24],
                "raw_storage_policy": "RAW_ONLY_LOCAL_RUNS",
                "private_path_removed": "true",
            }
        )
    for row in issue_rows:
        row.pop("_download_url", None)
    if not manifest:
        manifest.append(
            {
                "download_id": "GAZETTEDL_V1NA_NONE",
                "issue_id": "none",
                "official_domain": "none",
                "url_hash": public_hash("none"),
                "file_type": "NONE",
                "acquisition_status": "NO_REAL_ISSUE_DOWNLOAD",
                "byte_count": "0",
                "sha256": "",
                "raw_storage_policy": "RAW_ONLY_LOCAL_RUNS",
                "private_path_removed": "true",
            }
        )
    return issue_rows, manifest


def raw_gazette_documents() -> list[Path]:
    if not RAW.exists():
        return []
    return sorted(p for p in RAW.rglob("*") if p.is_file() and p.suffix.lower() in {".pdf", ".html", ".htm", ".txt"})


def issue_id_from_path(path: Path) -> str:
    match = re.search(r"gazette_issue_(\d{4}-\d{2}-\d{2})_([^_]+)_", path.name)
    if not match:
        return f"GAZETTEISSUE_LOCAL_{public_hash(path.name, 8)}"
    number = re.sub(r"\D", "", match.group(2))
    for row in read_csv(DATASETS / "petropolis_official_gazette_issue_registry.csv"):
        if row.get("issue_date") == match.group(1) and (not number or row.get("issue_number") == number):
            return row.get("issue_id") or f"GAZETTEISSUE_LOCAL_{public_hash(path.name, 8)}"
    return f"GAZETTEISSUE_LOCAL_{public_hash(path.name, 8)}"


def extract_gazette_text() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    docs = [p for p in raw_gazette_documents() if p.name.startswith("gazette_issue_")]
    doc_rows: list[dict[str, Any]] = []
    page_rows: list[dict[str, Any]] = []
    for didx, path in enumerate(docs, 1):
        issue_id = issue_id_from_path(path)
        pages, method, status = extract_pages(path)
        page_count = len(pages)
        text_chars = sum(len(text) for _, text in pages)
        doc_id = f"GAZETTEDOC_V1NB_{didx:04d}_{public_hash(path.name, 8)}"
        doc_rows.append(
            {
                "document_id": doc_id,
                "issue_id": issue_id,
                "document_kind": path.suffix.lower().replace(".", "").upper() or "UNKNOWN",
                "page_count": str(page_count),
                "extraction_method": method,
                "extraction_status": status,
                "text_char_count": str(text_chars),
                "ocr_status": "OCR_NOT_NEEDED" if status == "TEXT_EXTRACTED" else "OCR_NOT_AVAILABLE_OR_NOT_RUN",
                "private_path_removed": "true",
            }
        )
        for page_no, text in pages:
            page_rows.append(
                {
                    "page_text_id": f"GAZETTEPAGE_V1NB_{len(page_rows)+1:05d}",
                    "document_id": doc_id,
                    "issue_id": issue_id,
                    "page": str(page_no),
                    "text_char_count": str(len(text)),
                    "has_keyword_hit": str(contains_any(text, GAZETTE_ADMIN_TERMS + STRICT_NEGATIVE_TERMS)).lower(),
                    "page_text_sample": sanitize_public_text(text, 420),
                    "extraction_method": method,
                    "private_path_removed": "true",
                }
            )
    if not doc_rows:
        doc_rows.append({"document_id": "GAZETTEDOC_V1NB_NONE", "issue_id": "none", "document_kind": "NONE", "page_count": "0", "extraction_method": "none", "extraction_status": "NO_GAZETTE_DOCUMENTS_FOUND", "text_char_count": "0", "ocr_status": "OCR_NOT_RUN", "private_path_removed": "true"})
    if not page_rows:
        page_rows.append({"page_text_id": "GAZETTEPAGE_V1NB_NONE", "document_id": doc_rows[0]["document_id"], "issue_id": doc_rows[0]["issue_id"], "page": "0", "text_char_count": "0", "has_keyword_hit": "false", "page_text_sample": "NO_GAZETTE_PAGE_TEXT_FOUND", "extraction_method": doc_rows[0]["extraction_method"], "private_path_removed": "true"})
    return doc_rows, page_rows


def segment_gazette_acts() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    segments: list[dict[str, Any]] = []
    hits: list[dict[str, Any]] = []
    for path in [p for p in raw_gazette_documents() if p.name.startswith("gazette_issue_")]:
        issue_id = issue_id_from_path(path)
        pages, method, _status = extract_pages(path)
        for page_no, text in pages:
            sentences = split_sentences(text)
            for idx, sentence in enumerate(sentences):
                context = " ".join(sentences[max(0, idx - 1) : min(len(sentences), idx + 2)])
                terms = matched_terms(context)
                if not terms:
                    continue
                act_id = f"GAZETTEACT_V1NC_{len(segments)+1:05d}"
                segments.append(
                    {
                        "act_id": act_id,
                        "issue_id": issue_id,
                        "issue_date": issue_date(issue_id),
                        "page": str(page_no),
                        "act_type": act_type(context),
                        "context": sanitize_public_text(context, 620),
                        "matched_terms": "|".join(terms[:8]),
                        "extraction_method": method,
                        "private_path_removed": "true",
                    }
                )
                for term in terms:
                    hits.append(
                        {
                            "hit_id": f"GAZETTEHIT_V1NC_{len(hits)+1:05d}",
                            "act_id": act_id,
                            "issue_id": issue_id,
                            "page": str(page_no),
                            "matched_term": sanitize_public_text(term, 120),
                            "term_class": term_class(term),
                            "private_path_removed": "true",
                        }
                    )
    if not segments:
        segments.append({"act_id": "GAZETTEACT_V1NC_NONE", "issue_id": "none", "issue_date": "", "page": "0", "act_type": "NONE", "context": "NO_ADMINISTRATIVE_ACT_KEYWORD_HIT_FOUND", "matched_terms": "", "extraction_method": "none", "private_path_removed": "true"})
    if not hits:
        hits.append({"hit_id": "GAZETTEHIT_V1NC_NONE", "act_id": segments[0]["act_id"], "issue_id": segments[0]["issue_id"], "page": "0", "matched_term": "none", "term_class": "NONE", "private_path_removed": "true"})
    return segments, hits


def matched_terms(text: str) -> list[str]:
    out = []
    low = text.casefold()
    for term in GAZETTE_ADMIN_TERMS + STRICT_NEGATIVE_TERMS + RELEASE_TERMS + LOW_RISK_TERMS:
        if term.casefold() in low and term not in out:
            out.append(term)
    if re.search(r"\bliberad[oa]s?\b", low) and "liberado" not in out:
        out.append("liberado")
    return out


def term_class(term: str) -> str:
    if contains_any(term, STRICT_NEGATIVE_TERMS):
        return "STRICT_NEGATIVE"
    if contains_any(term, LOW_RISK_TERMS):
        return "LOW_RISK_CONTEXT_ONLY"
    if contains_any(term, RELEASE_TERMS) or term == "liberado":
        return "RELEASE_CONTEXT"
    return "ADMIN_CONTEXT"


def act_type(text: str) -> str:
    if contains_any(text, STRICT_NEGATIVE_TERMS):
        return "EXPLICIT_NEGATIVE_CONTEXT"
    if contains_any(text, ["desinterdicao", "desinterdiÃ§Ã£o"]):
        return "DESINTERDICAO"
    if contains_any(text, ["interdicao", "interdiÃ§Ã£o"]):
        return "INTERDICAO"
    if contains_any(text, ["vistoria", "laudo", "auto de vistoria"]):
        return "VISTORIA_LAUDO"
    if contains_any(text, ["habitabilidade", "habitavel", "habitÃ¡vel"]):
        return "HABITABILIDADE"
    return "ADMINISTRATIVE_CONTEXT"


def contains_release_context(text: str) -> bool:
    low = text.casefold()
    return contains_any(text, RELEASE_TERMS) or bool(re.search(r"\bliberad[oa]s?\b", low))


def issue_date(issue_id: str) -> str:
    for row in read_csv(DATASETS / "petropolis_official_gazette_issue_registry.csv"):
        if row.get("issue_id") == issue_id:
            return row.get("issue_date", "")
    return ""


def phrase_flags(text: str, issue_dt: str = "") -> dict[str, str]:
    padded = f" {text.casefold()} "
    return {
        "has_date": str(bool(issue_dt or re.search(r"\b\d{1,2}/\d{1,2}/20\d{2}\b", text))).lower(),
        "has_location": str(contains_any(text, LOCALITY_TERMS) or any(term in padded for term in ADDRESS_TERMS)).lower(),
        "has_precise_address": str(any(term in padded for term in ADDRESS_TERMS)).lower(),
        "has_coordinate": str(bool(re.search(r"[-+]?\d{1,3}[.,]\d{4,}", text))).lower(),
        "has_phenomenon": str(contains_any(text, PHENOMENON_TERMS + ["risco geologico", "risco geolÃ³gico", "encosta", "barreira"])).lower(),
    }


def mine_negative_semantics_from_acts() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    for row in read_csv(DATASETS / "administrative_act_segment_registry.csv"):
        if row.get("act_id") == "GAZETTEACT_V1NC_NONE":
            continue
        text = row.get("context", "")
        issue_dt = row.get("issue_date", "")
        explicit = contains_any(text, STRICT_NEGATIVE_TERMS)
        release = contains_release_context(text)
        low_risk = contains_any(text, LOW_RISK_TERMS)
        flags = phrase_flags(text, issue_dt)
        if explicit and flags["has_date"] == "true" and flags["has_location"] == "true" and flags["has_phenomenon"] == "true":
            decision = "FORMAL_NEGATIVE_CANDIDATE_REVIEW"
        elif release and not explicit:
            decision = "ADMINISTRATIVE_RELEASE_REVIEW_ONLY"
        elif low_risk:
            decision = "CONTEXT_ONLY"
        elif explicit and flags["has_date"] != "true":
            decision = "BLOCKED_NO_DATE"
        elif explicit and flags["has_location"] != "true":
            decision = "BLOCKED_NO_LOCATION"
        elif explicit and flags["has_phenomenon"] != "true":
            decision = "BLOCKED_NOT_PHENOMENON_SPECIFIC"
        elif explicit:
            decision = "BLOCKED_NOT_EXPLICIT"
        else:
            continue
        out = {
            "candidate_id": f"GAZETTENEG_V1ND_{len(candidates)+1:05d}" if decision == "FORMAL_NEGATIVE_CANDIDATE_REVIEW" else f"GAZETTECTX_V1ND_{len(rejections)+1:05d}",
            "act_id": row.get("act_id"),
            "issue_id": row.get("issue_id"),
            "issue_date": issue_dt,
            "page": row.get("page"),
            "phrase_or_context": sanitize_public_text(text, 620),
            "explicit_negative_statement_gate": "PASS" if explicit else "FAIL",
            "date_gate": "PASS" if flags["has_date"] == "true" else "FAIL",
            "location_gate": "PASS" if flags["has_location"] == "true" else "FAIL",
            "phenomenon_specific_gate": "PASS" if flags["has_phenomenon"] == "true" else "FAIL",
            "decision": decision,
            "can_create_operational_label": "false",
            "can_train_model": "false",
        }
        if decision == "FORMAL_NEGATIVE_CANDIDATE_REVIEW":
            candidates.append(out)
        else:
            rejections.append(out)
    if not candidates:
        candidates.append({"candidate_id": "GAZETTENEG_V1ND_NONE", "act_id": "none", "issue_id": "none", "issue_date": "", "page": "0", "phrase_or_context": "NO_EXPLICIT_GAZETTE_NEGATIVE_CANDIDATE_FOUND", "explicit_negative_statement_gate": "FAIL", "date_gate": "FAIL", "location_gate": "FAIL", "phenomenon_specific_gate": "FAIL", "decision": "BLOCKED_NOT_EXPLICIT", "can_create_operational_label": "false", "can_train_model": "false"})
    if not rejections:
        rejections.append({"candidate_id": "GAZETTEREJECT_V1ND_NONE", "act_id": "none", "issue_id": "none", "issue_date": "", "page": "0", "phrase_or_context": "NO_GAZETTE_SEMANTIC_REJECTION_CONTEXT_FOUND", "explicit_negative_statement_gate": "FAIL", "date_gate": "FAIL", "location_gate": "FAIL", "phenomenon_specific_gate": "FAIL", "decision": "NO_REJECTION_CONTEXT", "can_create_operational_label": "false", "can_train_model": "false"})
    return candidates, rejections


def extract_coordinate(text: str) -> tuple[str, str] | None:
    nums = [n.replace(",", ".") for n in re.findall(r"[-+]?\d{1,3}[.,]\d{4,}", text)]
    vals: list[float] = []
    for n in nums:
        try:
            vals.append(float(n))
        except ValueError:
            pass
    for a, b in zip(vals, vals[1:]):
        if -23.0 <= a <= -22.0 and -44.0 <= b <= -42.0:
            return f"{a:.6f}", f"{b:.6f}"
        if -23.0 <= b <= -22.0 and -44.0 <= a <= -42.0:
            return f"{b:.6f}", f"{a:.6f}"
    return None


def nearest_positive_distance(lat: str, lon: str) -> str:
    if not lat or not lon:
        return ""
    best = math.inf
    for row in read_csv(DATASETS / "ground_reference_event_registry.csv"):
        try:
            best = min(best, haversine_m(float(lat), float(lon), float(row.get("latitude", "")), float(row.get("longitude", ""))))
        except Exception:
            continue
    return f"{best:.1f}" if math.isfinite(best) else ""


def geocode_gazette_candidates() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cand in read_csv(DATASETS / "gazette_negative_semantics_candidate_registry.csv"):
        if cand.get("decision") != "FORMAL_NEGATIVE_CANDIDATE_REVIEW":
            continue
        text = cand.get("phrase_or_context", "")
        coord = extract_coordinate(text)
        flags = phrase_flags(text, cand.get("issue_date", ""))
        if coord:
            status = "EXPLICIT_COORDINATE"
            precise_gate = "PASS"
            lat, lon = coord
        elif flags["has_precise_address"] == "true":
            status = "PRECISE_ADDRESS_GEOCODE_PENDING"
            precise_gate = "FAIL"
            lat = lon = ""
        else:
            status = "REVIEW_AREA_ONLY"
            precise_gate = "FAIL"
            lat = lon = ""
        dist = nearest_positive_distance(lat, lon) if coord else ""
        buffer_gate = "PASS" if dist and float(dist) >= 1000 else "FAIL"
        patch_gate = "PASS" if precise_gate == "PASS" and buffer_gate == "PASS" else "FAIL"
        rows.append(
            {
                "geocode_id": f"GAZETTEGEO_V1NE_{len(rows)+1:05d}",
                "candidate_id": cand.get("candidate_id"),
                "address_or_locality_status": status,
                "latitude": lat,
                "longitude": lon,
                "precise_location_gate": "PASS" if flags["has_precise_address"] == "true" else "FAIL",
                "coordinate_or_geocodable_address_gate": precise_gate,
                "review_area_only_flag": "true" if status == "REVIEW_AREA_ONLY" else "false",
                "nearest_positive_distance_m": dist,
                "positive_buffer_exclusion_gate": buffer_gate,
                "patch_extractability_gate": patch_gate,
                "private_path_removed": "true",
            }
        )
    if not rows:
        rows.append({"geocode_id": "GAZETTEGEO_V1NE_NONE", "candidate_id": "none", "address_or_locality_status": "NO_STRONG_GAZETTE_NEGATIVE_TO_GEOCODE", "latitude": "", "longitude": "", "precise_location_gate": "FAIL", "coordinate_or_geocodable_address_gate": "FAIL", "review_area_only_flag": "false", "nearest_positive_distance_m": "", "positive_buffer_exclusion_gate": "FAIL", "patch_extractability_gate": "FAIL", "private_path_removed": "true"})
    matrix = {
        "matrix_id": "GAZETTE_NEG_SPATIAL_SPECIFICITY_V1NE",
        "candidate_count": str(len([r for r in rows if r["candidate_id"] != "none"])),
        "precise_location_pass_count": str(sum(1 for r in rows if r["precise_location_gate"] == "PASS")),
        "coordinate_or_geocodable_address_pass_count": str(sum(1 for r in rows if r["coordinate_or_geocodable_address_gate"] == "PASS")),
        "patch_extractability_pass_count": str(sum(1 for r in rows if r["patch_extractability_gate"] == "PASS")),
        "decision": "GAZETTE_NEGATIVE_SPATIAL_REVIEW_READY" if any(r["coordinate_or_geocodable_address_gate"] == "PASS" for r in rows) else "GAZETTE_NEGATIVE_LOCATION_BLOCKED",
        "remaining_blocker": "NONE" if any(r["coordinate_or_geocodable_address_gate"] == "PASS" for r in rows) else "NO_COORDINATE_OR_GEOCODABLE_ADDRESS",
    }
    return rows, matrix


def adjudicate_gazette_negatives() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    geo = {r.get("candidate_id"): r for r in read_csv(DATASETS / "gazette_negative_address_geocoding_registry.csv")}
    rows: list[dict[str, Any]] = []
    for cand in read_csv(DATASETS / "gazette_negative_semantics_candidate_registry.csv"):
        if cand.get("candidate_id") == "GAZETTENEG_V1ND_NONE":
            continue
        g = geo.get(cand.get("candidate_id"), {})
        explicit = cand.get("explicit_negative_statement_gate") == "PASS"
        phen = cand.get("phenomenon_specific_gate") == "PASS"
        date_gate = cand.get("date_gate") == "PASS"
        loc = cand.get("location_gate") == "PASS"
        precise = g.get("precise_location_gate") == "PASS"
        geocode = g.get("coordinate_or_geocodable_address_gate") == "PASS"
        buffer_gate = g.get("positive_buffer_exclusion_gate") == "PASS"
        patch_gate = g.get("patch_extractability_gate") == "PASS"
        leakage = buffer_gate and patch_gate
        if not explicit:
            decision = "BLOCKED_NOT_EXPLICIT"
        elif not phen:
            decision = "BLOCKED_NO_PHENOMENON"
        elif not date_gate:
            decision = "BLOCKED_NO_DATE"
        elif not loc or not precise:
            decision = "BLOCKED_LOCATION_WEAK"
        elif not geocode:
            decision = "BLOCKED_GEOCODING_WEAK"
        elif not buffer_gate:
            decision = "BLOCKED_POSITIVE_BUFFER"
        elif not patch_gate:
            decision = "BLOCKED_PATCH_EXTRACTABILITY"
        elif not leakage:
            decision = "NEGATIVE_REVIEW_ONLY"
        else:
            decision = "FORMAL_NEGATIVE_CANDIDATE"
        rows.append(
            {
                "candidate_id": f"FORMAL_GAZETTENEG_V1NF_{len(rows)+1:05d}",
                "source_candidate_id": cand.get("candidate_id"),
                "official_gazette_gate": "PASS",
                "administrative_act_gate": "PASS",
                "explicit_negative_statement_gate": "PASS" if explicit else "FAIL",
                "phenomenon_specific_gate": "PASS" if phen else "FAIL",
                "date_gate": "PASS" if date_gate else "FAIL",
                "precise_location_gate": "PASS" if precise else "FAIL",
                "coordinate_or_geocodable_address_gate": "PASS" if geocode else "FAIL",
                "independent_area_gate": "PASS" if loc and precise else "FAIL",
                "positive_buffer_exclusion_gate": "PASS" if buffer_gate else "FAIL",
                "patch_extractability_gate": "PASS" if patch_gate else "FAIL",
                "leakage_precheck_gate": "PASS" if leakage else "FAIL",
                "decision": decision,
                "can_create_operational_label": "false",
                "can_train_model": "false",
            }
        )
    if not rows:
        rows.append({"candidate_id": "FORMAL_GAZETTENEG_V1NF_NONE", "source_candidate_id": "none", "official_gazette_gate": "PASS", "administrative_act_gate": "FAIL", "explicit_negative_statement_gate": "FAIL", "phenomenon_specific_gate": "FAIL", "date_gate": "FAIL", "precise_location_gate": "FAIL", "coordinate_or_geocodable_address_gate": "FAIL", "independent_area_gate": "FAIL", "positive_buffer_exclusion_gate": "FAIL", "patch_extractability_gate": "FAIL", "leakage_precheck_gate": "FAIL", "decision": "BLOCKED_NOT_EXPLICIT", "can_create_operational_label": "false", "can_train_model": "false"})
    formal = sum(1 for r in rows if r["decision"] == "FORMAL_NEGATIVE_CANDIDATE")
    matrix = {"matrix_id": "FORMAL_GAZETTE_NEG_GATE_V1NF", "formal_negative_count": str(formal), "review_or_blocked_count": str(len(rows) - formal), "decision": "FORMAL_NEGATIVE_CANDIDATE" if formal else "NO_FORMAL_GAZETTE_NEGATIVE", "remaining_blocker": "NONE" if formal else "NO_EXPLICIT_GAZETTE_NEGATIVE_WITH_PRECISE_LOCATION_PATCH_LEAKAGE", "can_create_operational_label": "false", "can_train_model": "false"}
    return rows, matrix


def c4_after_gazette_negatives() -> tuple[dict[str, Any], dict[str, Any]]:
    formal_pos = sum(1 for r in read_csv(DATASETS / "formal_positive_label_candidate_registry.csv") if r.get("decision") == "FORMAL_POSITIVE_PATCH_CANDIDATE")
    gate = read_csv(DATASETS / "formal_negative_gazette_gate_matrix.csv")
    formal_neg = int((gate[0].get("formal_negative_count", "0") if gate else "0") or 0)
    if formal_neg <= 0:
        decision = "C4_BLOCKED_NO_FORMAL_NEGATIVES"
        blocker = "NO_FORMAL_NEGATIVES"
    else:
        decision = "C4_OPERATIONAL_READY"
        blocker = "NONE"
    c4 = {"decision_id": "C4_GAZETTE_NEG_RECHECK_V1NG", "formal_positive_count": str(formal_pos), "formal_negative_count": str(formal_neg), "negative_provenance_gate": "PASS" if formal_neg else "FAIL", "patch_extractability_gate": "PASS" if formal_neg else "FAIL", "split_leakage_gate": "PASS" if formal_neg else "FAIL", "decision": decision, "remaining_blocker": blocker, "can_create_operational_label": "true" if decision == "C4_OPERATIONAL_READY" else "false", "can_train_model": "false"}
    readiness = {"readiness_id": "C4_LABEL_READINESS_GAZETTE_V1NG", "formal_positive_count": str(formal_pos), "formal_negative_count": str(formal_neg), "positive_gate": "PASS" if formal_pos else "FAIL", "negative_gate": "PASS" if formal_neg else "FAIL", "split_leakage_gate": c4["split_leakage_gate"], "decision": decision, "remaining_blocker": blocker, "can_create_operational_label": c4["can_create_operational_label"], "can_train_model": "false"}
    return c4, readiness


def gazette_summary() -> dict[str, Any]:
    issues = read_csv(DATASETS / "petropolis_official_gazette_issue_registry.csv")
    downloads = read_csv(DATASETS / "petropolis_official_gazette_download_manifest.csv")
    pages = read_csv(DATASETS / "petropolis_gazette_page_text_inventory.csv")
    acts = read_csv(DATASETS / "administrative_act_segment_registry.csv")
    candidates = read_csv(DATASETS / "gazette_negative_semantics_candidate_registry.csv")
    geocoded = read_csv(DATASETS / "gazette_negative_address_geocoding_registry.csv")
    formal_matrix = read_csv(DATASETS / "formal_negative_gazette_gate_matrix.csv")
    c4 = read_csv(DATASETS / "c4_recheck_after_gazette_negatives.csv")
    formal_count = int((formal_matrix[0].get("formal_negative_count", "0") if formal_matrix else "0") or 0)
    c4_decision = c4[0].get("decision", "C4_BLOCKED_NO_FORMAL_NEGATIVES") if c4 else "C4_BLOCKED_NO_FORMAL_NEGATIVES"
    best = next((r["candidate_id"] for r in candidates if r.get("candidate_id") != "GAZETTENEG_V1ND_NONE" and r.get("decision") == "FORMAL_NEGATIVE_CANDIDATE_REVIEW"), "none")
    issue_count = sum(1 for r in issues if r.get("is_real_issue") == "true")
    dl_count = sum(1 for r in downloads if r.get("acquisition_status") == "DOWNLOAD_OK")
    return {
        "summary_id": "PROTOCOL_C_GAZETTE_NEGATIVE_RESOLUTION_V1NH",
        "issues_discovered": str(issue_count),
        "issues_downloaded": str(dl_count),
        "pages_extracted": str(sum(1 for r in pages if r.get("page_text_id") != "GAZETTEPAGE_V1NB_NONE")),
        "administrative_acts_segmented": str(sum(1 for r in acts if r.get("act_id") != "GAZETTEACT_V1NC_NONE")),
        "negative_candidates": str(sum(1 for r in candidates if r.get("candidate_id") != "GAZETTENEG_V1ND_NONE")),
        "geocoded_candidates": str(sum(1 for r in geocoded if r.get("candidate_id") != "none")),
        "formal_negative_count": str(formal_count),
        "c4_decision": c4_decision,
        "best_negative_candidate": best,
        "remaining_blocker": "NONE" if formal_count else "NO_FORMAL_NEGATIVES",
        "next_single_technical_action": "obtain or request an official 2022 gazette/administrative act with explicit no-risk/no-occurrence statement, precise address or coordinate, phenomenon, patch extractability, and leakage clearance",
        "can_create_operational_label": "true" if c4_decision == "C4_OPERATIONAL_READY" else "false",
        "can_train_model": "false",
    }


def write_simple_doc(path: Path, title: str, lines: list[str]) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    path.write_text("# " + title + "\n\n" + "\n".join(f"- {sanitize_public_text(line, 620)}" for line in lines) + "\n", encoding="utf-8")
