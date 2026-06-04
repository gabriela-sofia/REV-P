#!/usr/bin/env python3
"""v1uq Petropolis phenomenon separation deep audit.

Local-only PDF text audit. It never promotes document evidence to geometry,
ground reference, ground truth, overlay, or labels.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict

try:
    import pypdf  # type: ignore
except Exception:  # pragma: no cover
    pypdf = None

try:
    import pdfplumber  # type: ignore
except Exception:  # pragma: no cover
    pdfplumber = None

PROTOCOL_VERSION = "v1uq"
DATASET_DIR = "datasets/protocolo_c"
DOCS_DIR = "docs/metodologia_cientifica"
CONFIG_DIR = "configs/protocolo_c"
V1UP_RAW_DIR = "local_only/protocolo_c/petropolis_public_geometry/raw/v1up"
LOCAL_STAGING_DIR = "local_only/protocolo_c/petropolis_phenomenon_audit/staging/v1uq"
LOCAL_REPORTS_DIR = "local_only/protocolo_c/petropolis_phenomenon_audit/reports/v1uq"
MAX_STATUS = "PETROPOLIS_PHENOMENON_SEPARATION_EVIDENCE_FOR_REVIEW"

TEXT_COLUMNS = [
    "text_extract_id", "event_id", "asset_id", "source_id", "pdf_sha256",
    "page_count", "extraction_backend", "extraction_status",
    "extracted_text_local_hash", "total_chars", "pages_with_text",
    "contains_flood_terms", "contains_landslide_terms",
    "contains_geodata_terms", "contains_date_terms",
    "can_create_ground_reference", "can_create_training_label", "notes",
]

STRUCTURE_COLUMNS = [
    "structure_id", "event_id", "asset_id", "page_number", "has_text",
    "likely_map_page", "likely_table_page", "likely_figure_page",
    "likely_annex_page", "geodata_reference_signal",
    "phenomenon_signal", "notes",
]

TERM_COLUMNS = [
    "term_index_id", "event_id", "asset_id", "page_number",
    "phenomenon_class", "matched_term_class", "matched_terms_hash",
    "term_count", "evidence_strength", "is_context_only",
    "can_support_phenomenon_separation", "notes",
]

PAGE_COLUMNS = [
    "page_evidence_id", "event_id", "asset_id", "page_number",
    "dominant_phenomenon_class", "flood_signal_strength",
    "landslide_signal_strength", "mixed_signal_strength",
    "locality_signal_strength", "date_signal_strength",
    "geodata_signal_strength", "is_map_page", "is_context_only",
    "evidence_role", "can_support_phenomenon_gate",
    "can_create_ground_reference", "notes",
]

LOCALITY_COLUMNS = [
    "locality_audit_id", "event_id", "asset_id", "page_number",
    "locality_token_hash", "locality_class", "locality_signal_strength",
    "linked_phenomenon_class", "can_support_contextual_review",
    "can_support_overlay", "notes",
]

DATE_COLUMNS = [
    "date_linkage_id", "event_id", "asset_id", "page_number",
    "date_signal_class", "date_signal_strength", "event_specificity",
    "temporal_link_status", "can_support_temporal_gate", "notes",
]

DECISION_COLUMNS = [
    "decision_id", "event_id", "flood_signal", "landslide_signal",
    "mixed_signal", "separation_status", "locality_linkage_status",
    "temporal_linkage_status", "geodata_status", "evidence_strength",
    "can_advance_to_geometry_search", "can_advance_to_overlay_preflight",
    "can_create_ground_reference", "can_create_training_label",
    "main_blocker", "required_next_action", "notes",
]

MISSING_GEODATA_COLUMNS = [
    "missing_geodata_id", "event_id", "asset_id", "page_number",
    "signal_class", "signal_strength", "referenced_artifact_type",
    "public_path_hint", "can_be_resolved_by_public_search",
    "recommended_next_query", "notes",
]

EVENT_STATUS_COLUMNS = [
    "event_id", "v1uq_status", "phenomenon_separation_status",
    "documentary_evidence_strength", "has_missing_geodata_signal",
    "has_observed_geometry", "ground_truth_operational",
    "can_create_ground_reference", "can_create_training_label",
    "can_advance_to_overlay_preflight", "main_blocker",
    "recommended_next_action", "notes",
]

BLOCKER_COLUMNS = [
    "blocker_id", "event_id", "gate", "gate_status", "blocking_reason",
    "ground_truth_operational", "can_create_ground_reference",
    "can_create_training_label", "notes",
]

NEXT_ACTION_COLUMNS = [
    "action_id", "event_id", "action_type", "priority", "description",
    "target", "status", "notes",
]

MANIFEST_COLUMNS = [
    "artifact_id", "artifact_path", "artifact_type", "protocol_version",
    "sha256_prefix", "file_size_bytes", "is_versionable", "reason",
]

V1UQ_ARTIFACTS = [
    "configs/protocolo_c/v1uq_petropolis_pdf_text_policy.yaml",
    "configs/protocolo_c/v1uq_petropolis_phenomenon_terms.yaml",
    "configs/protocolo_c/v1uq_petropolis_locality_terms.yaml",
    "configs/protocolo_c/v1uq_petropolis_date_linkage_policy.yaml",
    "configs/protocolo_c/v1uq_petropolis_decision_policy.yaml",
    "configs/protocolo_c/v1uq_petropolis_missing_geodata_policy.yaml",
    "datasets/protocolo_c/v1uq_petropolis_pdf_text_extraction_registry.csv",
    "datasets/protocolo_c/v1uq_petropolis_pdf_structure_inventory.csv",
    "datasets/protocolo_c/v1uq_petropolis_phenomenon_term_index.csv",
    "datasets/protocolo_c/v1uq_petropolis_page_level_evidence_registry.csv",
    "datasets/protocolo_c/v1uq_petropolis_locality_term_audit.csv",
    "datasets/protocolo_c/v1uq_petropolis_event_date_linkage_audit.csv",
    "datasets/protocolo_c/v1uq_petropolis_phenomenon_separation_decision_matrix.csv",
    "datasets/protocolo_c/v1uq_petropolis_missing_geodata_signal_audit.csv",
    "datasets/protocolo_c/v1uq_petropolis_event_status_registry.csv",
    "datasets/protocolo_c/v1uq_petropolis_ground_reference_blocker_matrix.csv",
    "datasets/protocolo_c/v1uq_next_actions_registry.csv",
    "datasets/protocolo_c/v1uq_versionable_artifacts_manifest.csv",
    "docs/metodologia_cientifica/protocolo_c_v1uq_petropolis_phenomenon_separation_deep_audit.md",
    "docs/metodologia_cientifica/protocolo_c_relatorio_v1uq_petropolis_phenomenon_separation_deep_audit.md",
    "docs/metodologia_cientifica/protocolo_c_status_atual_v1uq.md",
]

FLOOD_TERMS = ["inundacao", "alagamento", "enchente", "enxurrada", "transbordamento", "cheia", "inundado"]
LANDSLIDE_TERMS = ["deslizamento", "escorregamento", "movimento de massa", "corrida de massa", "queda de barreira", "instabilidade de encosta"]
RISK_TERMS = ["risco geologico", "suscetibilidade", "setor de risco", "area de risco", "risco"]
HYDROMET_TERMS = ["chuva", "precipitacao", "pluviometr", "hidrometeorologico"]
GEODATA_TERMS = ["geodados", "shapefile", "base cartografica", "sig", "arcgis", "qgis", "anexo digital", "camada", "raster", "vetor", "coordenada", "mapa"]
MAP_TERMS = ["mapa", "carta", "cartografia", "croqui", "figura"]
TABLE_TERMS = ["tabela", "quadro", "planilha"]
ANNEX_TERMS = ["anexo", "apendice"]
LOCALITY_TERMS = [
    "petropolis", "quitandinha", "rio quitandinha", "centro", "morin",
    "alto da serra", "bingen", "cascatinha", "itaipava", "correas",
    "corrêas", "nogueira", "mosella", "moinho preto", "valparaiso",
    "serra velha", "sargento boening", "rua teresa",
]


def norm_text(value):
    normalized = unicodedata.normalize("NFKD", value or "")
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


def bool_text(value):
    return "true" if bool(value) else "false"


def write_csv(path, columns, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_text(path, lines):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_terms(terms):
    return hashlib.sha256("|".join(sorted(set(terms))).encode("utf-8")).hexdigest()[:16] if terms else ""


def strength(count):
    if count >= 5:
        return "STRONG"
    if count >= 2:
        return "MODERATE"
    if count == 1:
        return "WEAK"
    return "NONE"


def term_matches(text, terms):
    lower = norm_text(text)
    return [t for t in terms if norm_text(t) in lower]


def count_terms(text, terms):
    lower = norm_text(text)
    total = 0
    matched = []
    for term in terms:
        n = lower.count(norm_text(term))
        if n:
            total += n
            matched.append(term)
    return total, matched


def classify_page(text):
    flood_count, flood = count_terms(text, FLOOD_TERMS)
    land_count, land = count_terms(text, LANDSLIDE_TERMS)
    risk_count, risk = count_terms(text, RISK_TERMS)
    hydro_count, hydro = count_terms(text, HYDROMET_TERMS)
    if flood_count and land_count:
        cls = "MIXED_HYDRO_GEO"
    elif flood_count and any(norm_text(t) in {"alagamento", "alagamentos"} for t in flood):
        cls = "URBAN_FLOODING"
    elif flood_count and any(norm_text(t) in {"enxurrada", "transbordamento", "cheia"} for t in flood):
        cls = "FLASH_FLOOD_OR_RUNOFF"
    elif flood_count:
        cls = "FLOOD_OR_INUNDATION"
    elif land_count:
        cls = "LANDSLIDE_OR_MASS_MOVEMENT"
    elif risk_count:
        cls = "RISK_OR_SUSCEPTIBILITY_CONTEXT"
    elif hydro_count:
        cls = "HYDROMET_CONTEXT"
    else:
        cls = "NO_PHENOMENON_SIGNAL"
    return cls, flood_count, land_count, risk_count, hydro_count, flood + land + risk + hydro


def date_signal(text, event_id):
    lower = norm_text(text)
    if event_id == "PET_2022_02_15":
        exact = ["15/02/2022", "15-02-2022", "15 de fevereiro de 2022"]
        month = ["fevereiro de 2022", "fev/2022", "02/2022"]
        if any(t in lower for t in exact):
            return "EXACT_EVENT_DATE", "STRONG", "EVENT_SPECIFIC", "EXACT_EVENT_DATE", "true"
        if any(t in lower for t in month):
            return "EVENT_MONTH", "MODERATE", "EVENT_MONTH", "EVENT_MONTH", "true"
        if "2022" in lower:
            return "EVENT_YEAR", "WEAK", "YEAR_ONLY", "EVENT_YEAR", "false"
    if event_id == "PET_2024_03_21_28":
        exact = ["21/03/2024", "28/03/2024", "21-28/03/2024", "21 a 28 de marco de 2024", "21 a 28 de março de 2024"]
        month = ["marco de 2024", "março de 2024", "03/2024"]
        if any(norm_text(t) in lower for t in exact):
            return "EXACT_EVENT_DATE", "STRONG", "EVENT_SPECIFIC", "EXACT_EVENT_DATE", "true"
        if any(norm_text(t) in lower for t in month):
            return "EVENT_MONTH", "MODERATE", "EVENT_MONTH", "EVENT_MONTH", "true"
        if "2024" in lower:
            return "EVENT_YEAR", "WEAK", "YEAR_ONLY", "EVENT_YEAR", "false"
    if any(t in lower for t in ["relatorio", "relatório", "visita tecnica", "vistoria"]):
        return "POST_EVENT_REPORT", "WEAK", "DOCUMENT_CONTEXT", "POST_EVENT_REPORT", "false"
    return "NO_TEMPORAL_LINK", "NONE", "NONE", "NO_TEMPORAL_LINK", "false"


def pdf_candidates(local_only_dir=None):
    raw_dir = local_only_dir or V1UP_RAW_DIR
    downloads = load_csv(os.path.join(DATASET_DIR, "v1up_petropolis_download_manifest.csv"))
    for row in downloads:
        if row.get("downloaded") != "true":
            continue
        path = os.path.join(raw_dir, row.get("safe_filename", ""))
        if row.get("format_hint") == "pdf" and os.path.exists(path):
            yield {
                "event_id": row.get("event_id", ""),
                "asset_id": f"PDF_{row.get('download_id', '')}",
                "source_id": row.get("source_id", ""),
                "path": path,
                "zip_member": "",
                "bytes": open(path, "rb").read(),
            }
        elif row.get("format_hint") == "zip" and os.path.exists(path):
            try:
                with zipfile.ZipFile(path) as zf:
                    for name in zf.namelist():
                        if name.lower().endswith(".pdf"):
                            yield {
                                "event_id": row.get("event_id", ""),
                                "asset_id": f"PDF_{row.get('download_id', '')}_{hashlib.sha1(name.encode('utf-8')).hexdigest()[:10]}",
                                "source_id": row.get("source_id", ""),
                                "path": path,
                                "zip_member": name,
                                "bytes": zf.read(name),
                            }
            except Exception:
                continue


def extract_pages_from_bytes(data):
    if pypdf is not None:
        try:
            reader = pypdf.PdfReader(io.BytesIO(data))
            return "pypdf", [(idx + 1, page.extract_text() or "") for idx, page in enumerate(reader.pages)], "EXTRACTED"
        except Exception:
            return "pypdf", [], "EXTRACTION_FAILED"
    if pdfplumber is not None:
        try:
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                return "pdfplumber", [(idx + 1, page.extract_text() or "") for idx, page in enumerate(pdf.pages)], "EXTRACTED"
        except Exception:
            return "pdfplumber", [], "EXTRACTION_FAILED"
    return "PDF_BACKEND_MISSING", [], "PDF_BACKEND_MISSING"


def local_page_text_path(asset_id):
    return os.path.join(LOCAL_STAGING_DIR, f"{asset_id}_page_text.json")


def run_pdf_text_extractor(local_only_dir=None, dry_run=False):
    os.makedirs(LOCAL_STAGING_DIR, exist_ok=True)
    os.makedirs(LOCAL_REPORTS_DIR, exist_ok=True)
    rows = []
    for idx, candidate in enumerate(pdf_candidates(local_only_dir)):
        data = candidate["bytes"]
        pdf_hash = sha256_bytes(data)
        backend, pages, status = ("DRY_RUN", [], "DRY_RUN") if dry_run else extract_pages_from_bytes(data)
        page_payload = [{"page_number": n, "text": text} for n, text in pages]
        text_blob = json.dumps(page_payload, ensure_ascii=False)
        text_hash = sha256_bytes(text_blob.encode("utf-8")) if page_payload else ""
        if page_payload:
            write_text(local_page_text_path(candidate["asset_id"]), [text_blob])
        all_text = "\n".join(text for _, text in pages)
        rows.append({
            "text_extract_id": f"TEXT_v1uq_{idx:04d}",
            "event_id": candidate["event_id"],
            "asset_id": candidate["asset_id"],
            "source_id": candidate["source_id"],
            "pdf_sha256": pdf_hash,
            "page_count": str(len(pages)),
            "extraction_backend": backend,
            "extraction_status": status,
            "extracted_text_local_hash": text_hash,
            "total_chars": str(len(all_text)),
            "pages_with_text": str(sum(1 for _, text in pages if text.strip())),
            "contains_flood_terms": bool_text(bool(term_matches(all_text, FLOOD_TERMS))),
            "contains_landslide_terms": bool_text(bool(term_matches(all_text, LANDSLIDE_TERMS))),
            "contains_geodata_terms": bool_text(bool(term_matches(all_text, GEODATA_TERMS))),
            "contains_date_terms": bool_text("2022" in norm_text(all_text) or "2024" in norm_text(all_text) or bool(re.search(r"\d{2}/\d{2}/\d{4}", all_text))),
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "notes": "Full extracted text stored only in local_only; public CSV stores hashes and counts.",
        })
    out = os.path.join(DATASET_DIR, "v1uq_petropolis_pdf_text_extraction_registry.csv")
    write_csv(out, TEXT_COLUMNS, rows)
    print(f"[v1uq text extractor] rows={len(rows)} -> {out}")
    return rows


def load_page_payload(asset_id):
    path = local_page_text_path(asset_id)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.loads(f.read() or "[]")


def all_pages_from_text_registry():
    for row in load_csv(os.path.join(DATASET_DIR, "v1uq_petropolis_pdf_text_extraction_registry.csv")):
        for page in load_page_payload(row.get("asset_id", "")):
            yield row, page


def run_pdf_structure_inventory():
    rows = []
    seq = 0
    for row, page in all_pages_from_text_registry():
        text = page.get("text", "")
        cls, flood, land, risk, hydro, _ = classify_page(text)
        rows.append({
            "structure_id": f"STRUCT_v1uq_{seq:05d}",
            "event_id": row.get("event_id", ""),
            "asset_id": row.get("asset_id", ""),
            "page_number": str(page.get("page_number", "")),
            "has_text": bool_text(bool(text.strip())),
            "likely_map_page": bool_text(bool(term_matches(text, MAP_TERMS))),
            "likely_table_page": bool_text(bool(term_matches(text, TABLE_TERMS))),
            "likely_figure_page": bool_text("figura" in norm_text(text)),
            "likely_annex_page": bool_text(bool(term_matches(text, ANNEX_TERMS))),
            "geodata_reference_signal": bool_text(bool(term_matches(text, GEODATA_TERMS))),
            "phenomenon_signal": cls,
            "notes": "Structure inferred from text signals only; no image extraction or OCR.",
        })
        seq += 1
    out = os.path.join(DATASET_DIR, "v1uq_petropolis_pdf_structure_inventory.csv")
    write_csv(out, STRUCTURE_COLUMNS, rows)
    print(f"[v1uq structure] rows={len(rows)} -> {out}")
    return rows


def run_phenomenon_term_indexer():
    rows = []
    seq = 0
    for row, page in all_pages_from_text_registry():
        text = page.get("text", "")
        cls, flood_count, land_count, risk_count, hydro_count, matched = classify_page(text)
        total = flood_count + land_count + risk_count + hydro_count
        context = cls in {"RISK_OR_SUSCEPTIBILITY_CONTEXT", "HYDROMET_CONTEXT"}
        rows.append({
            "term_index_id": f"TERM_v1uq_{seq:05d}",
            "event_id": row.get("event_id", ""),
            "asset_id": row.get("asset_id", ""),
            "page_number": str(page.get("page_number", "")),
            "phenomenon_class": cls,
            "matched_term_class": "flood|landslide|risk|hydromet",
            "matched_terms_hash": hash_terms(matched),
            "term_count": str(total),
            "evidence_strength": strength(total),
            "is_context_only": bool_text(context),
            "can_support_phenomenon_separation": bool_text(cls in {"FLOOD_OR_INUNDATION", "URBAN_FLOODING", "FLASH_FLOOD_OR_RUNOFF", "LANDSLIDE_OR_MASS_MOVEMENT", "MIXED_HYDRO_GEO"}),
            "notes": "Terms hashed; no raw page text in versionable output.",
        })
        seq += 1
    out = os.path.join(DATASET_DIR, "v1uq_petropolis_phenomenon_term_index.csv")
    write_csv(out, TERM_COLUMNS, rows)
    print(f"[v1uq term index] rows={len(rows)} -> {out}")
    return rows


def run_page_level_evidence_builder():
    structures = {(r["asset_id"], r["page_number"]): r for r in load_csv(os.path.join(DATASET_DIR, "v1uq_petropolis_pdf_structure_inventory.csv"))}
    terms = {(r["asset_id"], r["page_number"]): r for r in load_csv(os.path.join(DATASET_DIR, "v1uq_petropolis_phenomenon_term_index.csv"))}
    rows = []
    seq = 0
    for row, page in all_pages_from_text_registry():
        key = (row["asset_id"], str(page.get("page_number", "")))
        term = terms.get(key, {})
        struct = structures.get(key, {})
        text = page.get("text", "")
        loc_count = len(term_matches(text, LOCALITY_TERMS))
        date_cls, date_strength, _, _, _ = date_signal(text, row.get("event_id", ""))
        geo_count = len(term_matches(text, GEODATA_TERMS))
        cls = term.get("phenomenon_class", "NO_PHENOMENON_SIGNAL")
        rows.append({
            "page_evidence_id": f"PAGE_v1uq_{seq:05d}",
            "event_id": row.get("event_id", ""),
            "asset_id": row.get("asset_id", ""),
            "page_number": str(page.get("page_number", "")),
            "dominant_phenomenon_class": cls,
            "flood_signal_strength": strength(count_terms(text, FLOOD_TERMS)[0]),
            "landslide_signal_strength": strength(count_terms(text, LANDSLIDE_TERMS)[0]),
            "mixed_signal_strength": "STRONG" if cls == "MIXED_HYDRO_GEO" else "NONE",
            "locality_signal_strength": strength(loc_count),
            "date_signal_strength": date_strength,
            "geodata_signal_strength": strength(geo_count),
            "is_map_page": struct.get("likely_map_page", "false"),
            "is_context_only": term.get("is_context_only", "false"),
            "evidence_role": "PHENOMENON_REVIEW" if term.get("can_support_phenomenon_separation") == "true" else "CONTEXT_OR_NONE",
            "can_support_phenomenon_gate": term.get("can_support_phenomenon_separation", "false"),
            "can_create_ground_reference": "false",
            "notes": "Page-level evidence only; PDF page is not geometry.",
        })
        seq += 1
    out = os.path.join(DATASET_DIR, "v1uq_petropolis_page_level_evidence_registry.csv")
    write_csv(out, PAGE_COLUMNS, rows)
    print(f"[v1uq page evidence] rows={len(rows)} -> {out}")
    return rows


def run_locality_term_audit():
    page_by_key = {(r["asset_id"], r["page_number"]): r for r in load_csv(os.path.join(DATASET_DIR, "v1uq_petropolis_page_level_evidence_registry.csv"))}
    rows = []
    seq = 0
    for row, page in all_pages_from_text_registry():
        text = page.get("text", "")
        matches = term_matches(text, LOCALITY_TERMS)
        if not matches:
            continue
        page_number = str(page.get("page_number", ""))
        linked = page_by_key.get((row["asset_id"], page_number), {}).get("dominant_phenomenon_class", "NO_PHENOMENON_SIGNAL")
        rows.append({
            "locality_audit_id": f"LOC_v1uq_{seq:05d}",
            "event_id": row.get("event_id", ""),
            "asset_id": row.get("asset_id", ""),
            "page_number": page_number,
            "locality_token_hash": hash_terms(matches),
            "locality_class": "PETROPOLIS_LOCALITY_TEXT_SIGNAL",
            "locality_signal_strength": strength(len(matches)),
            "linked_phenomenon_class": linked,
            "can_support_contextual_review": "true",
            "can_support_overlay": "false",
            "notes": "Textual locality only; no geocoding and no coordinates inferred.",
        })
        seq += 1
    out = os.path.join(DATASET_DIR, "v1uq_petropolis_locality_term_audit.csv")
    write_csv(out, LOCALITY_COLUMNS, rows)
    print(f"[v1uq locality] rows={len(rows)} -> {out}")
    return rows


def run_event_date_linkage_audit():
    rows = []
    seq = 0
    for row, page in all_pages_from_text_registry():
        cls, sig_strength, specificity, status, can_support = date_signal(page.get("text", ""), row.get("event_id", ""))
        if cls == "NO_TEMPORAL_LINK":
            continue
        rows.append({
            "date_linkage_id": f"DATE_v1uq_{seq:05d}",
            "event_id": row.get("event_id", ""),
            "asset_id": row.get("asset_id", ""),
            "page_number": str(page.get("page_number", "")),
            "date_signal_class": cls,
            "date_signal_strength": sig_strength,
            "event_specificity": specificity,
            "temporal_link_status": status,
            "can_support_temporal_gate": can_support,
            "notes": "Year-only dates are not treated as exact event dates.",
        })
        seq += 1
    out = os.path.join(DATASET_DIR, "v1uq_petropolis_event_date_linkage_audit.csv")
    write_csv(out, DATE_COLUMNS, rows)
    print(f"[v1uq date linkage] rows={len(rows)} -> {out}")
    return rows


def run_phenomenon_separation_decision_matrix():
    pages = load_csv(os.path.join(DATASET_DIR, "v1uq_petropolis_page_level_evidence_registry.csv"))
    locs = load_csv(os.path.join(DATASET_DIR, "v1uq_petropolis_locality_term_audit.csv"))
    dates = load_csv(os.path.join(DATASET_DIR, "v1uq_petropolis_event_date_linkage_audit.csv"))
    missing = load_csv(os.path.join(DATASET_DIR, "v1uq_petropolis_missing_geodata_signal_audit.csv"))
    rows = []
    for idx, event_id in enumerate(["PET_2022_02_15", "PET_2024_03_21_28"]):
        ev_pages = [p for p in pages if p["event_id"] == event_id]
        classes = Counter(p["dominant_phenomenon_class"] for p in ev_pages)
        flood = sum(classes[c] for c in ["FLOOD_OR_INUNDATION", "URBAN_FLOODING", "FLASH_FLOOD_OR_RUNOFF", "MIXED_HYDRO_GEO"])
        land = sum(classes[c] for c in ["LANDSLIDE_OR_MASS_MOVEMENT", "MIXED_HYDRO_GEO"])
        mixed = classes["MIXED_HYDRO_GEO"]
        separated_pages = classes["LANDSLIDE_OR_MASS_MOVEMENT"] + classes["FLOOD_OR_INUNDATION"] + classes["URBAN_FLOODING"] + classes["FLASH_FLOOD_OR_RUNOFF"]
        ev_locs = [l for l in locs if l["event_id"] == event_id]
        ev_dates = [d for d in dates if d["event_id"] == event_id and d["can_support_temporal_gate"] == "true"]
        ev_missing = [m for m in missing if m["event_id"] == event_id]
        if flood and land and separated_pages >= 2:
            sep = "PHENOMENON_SEPARATION_PARTIAL_TEXTUAL" if mixed else "PHENOMENON_SEPARATION_STRONG_TEXTUAL"
        elif mixed and not separated_pages:
            sep = "MIXED_PHENOMENON_NO_CLEAR_SEPARATION"
        elif land:
            sep = "LANDSLIDE_DOMINANT_CONTEXT"
        elif flood:
            sep = "FLOOD_DOMINANT_CONTEXT"
        elif any(p["is_context_only"] == "true" for p in ev_pages):
            sep = "RISK_CONTEXT_ONLY"
        else:
            sep = "INSUFFICIENT_TEXTUAL_EVIDENCE"
        evidence_strength = "STRONG" if separated_pages >= 5 and ev_locs and ev_dates else "MODERATE" if separated_pages else "WEAK"
        geodata_status = "MISSING_GEODATA_SIGNAL_PRESENT" if ev_missing else "NO_GEODATA_IN_PUBLIC_DOWNLOAD"
        next_action = "v1ur - Petropolis Public Geodata Path Recovery" if ev_missing else "v1ur - Petropolis Geometry Search from Missing Geodata Signals"
        rows.append({
            "decision_id": f"DEC_v1uq_{idx:04d}",
            "event_id": event_id,
            "flood_signal": bool_text(flood),
            "landslide_signal": bool_text(land),
            "mixed_signal": bool_text(mixed),
            "separation_status": sep,
            "locality_linkage_status": "LOCALITY_TEXT_LINK_PRESENT" if ev_locs else "LOCALITY_LINK_MISSING",
            "temporal_linkage_status": "TEMPORAL_TEXT_LINK_PRESENT" if ev_dates else "TEMPORAL_LINK_MISSING_OR_YEAR_ONLY",
            "geodata_status": geodata_status,
            "evidence_strength": evidence_strength,
            "can_advance_to_geometry_search": bool_text(bool(ev_missing) or sep.startswith("PHENOMENON_SEPARATION")),
            "can_advance_to_overlay_preflight": "false",
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "main_blocker": "GEOMETRY_STILL_MISSING",
            "required_next_action": next_action,
            "notes": "Textual phenomenon decision only; no geometry or ground reference created.",
        })
    out = os.path.join(DATASET_DIR, "v1uq_petropolis_phenomenon_separation_decision_matrix.csv")
    write_csv(out, DECISION_COLUMNS, rows)
    print(f"[v1uq decision] rows={len(rows)} -> {out}")
    return rows


def run_missing_geodata_signal_audit():
    rows = []
    seq = 0
    for row, page in all_pages_from_text_registry():
        text = page.get("text", "")
        matches = term_matches(text, GEODATA_TERMS)
        if not matches:
            continue
        signal_class = "MAP_OR_CARTOGRAPHY_REFERENCE" if any(norm_text(t) in {"mapa", "base cartografica"} for t in matches) else "DIGITAL_GEODATA_REFERENCE"
        artifact = "map_or_pdf_cartography" if signal_class == "MAP_OR_CARTOGRAPHY_REFERENCE" else "geodata_or_sig_asset"
        rows.append({
            "missing_geodata_id": f"MISSGEO_v1uq_{seq:05d}",
            "event_id": row.get("event_id", ""),
            "asset_id": row.get("asset_id", ""),
            "page_number": str(page.get("page_number", "")),
            "signal_class": signal_class,
            "signal_strength": strength(len(matches)),
            "referenced_artifact_type": artifact,
            "public_path_hint": "SGB_RIGEO_PUBLIC_SOURCE_RECHECK",
            "can_be_resolved_by_public_search": "true",
            "recommended_next_query": "Petropolis SGB RIGeo SIG shapefile geodados anexo digital",
            "notes": "Signal only; no missing file invented and no formal request required.",
        })
        seq += 1
    out = os.path.join(DATASET_DIR, "v1uq_petropolis_missing_geodata_signal_audit.csv")
    write_csv(out, MISSING_GEODATA_COLUMNS, rows)
    print(f"[v1uq missing geodata] rows={len(rows)} -> {out}")
    return rows


def run_event_status_updater():
    decisions = load_csv(os.path.join(DATASET_DIR, "v1uq_petropolis_phenomenon_separation_decision_matrix.csv"))
    missing_rows = load_csv(os.path.join(DATASET_DIR, "v1uq_petropolis_missing_geodata_signal_audit.csv"))
    missing_by_event = Counter(r.get("event_id", "") for r in missing_rows)
    status_rows = []
    blockers = []
    for dec in decisions:
        sep = dec["separation_status"]
        if sep == "PHENOMENON_SEPARATION_STRONG_TEXTUAL":
            status = "PHENOMENON_SEPARATION_STRONG_TEXTUAL_NO_GEOMETRY"
        elif sep == "PHENOMENON_SEPARATION_PARTIAL_TEXTUAL":
            status = "PHENOMENON_SEPARATION_PARTIAL_TEXTUAL_NO_GEOMETRY"
        elif sep == "MIXED_PHENOMENON_NO_CLEAR_SEPARATION":
            status = "MIXED_PHENOMENON_STILL_BLOCKED"
        elif sep == "RISK_CONTEXT_ONLY":
            status = "RISK_CONTEXT_ONLY_NO_GEOMETRY"
        elif dec["geodata_status"] == "MISSING_GEODATA_SIGNAL_PRESENT":
            status = "MISSING_GEODATA_PUBLIC_SEARCH_REQUIRED"
        elif sep == "INSUFFICIENT_TEXTUAL_EVIDENCE":
            status = "DOCUMENT_ONLY_NO_GEOMETRY"
        else:
            status = "GEOMETRY_STILL_MISSING"
        has_missing_geodata = missing_by_event[dec["event_id"]] > 0 or dec["geodata_status"] == "MISSING_GEODATA_SIGNAL_PRESENT"
        next_action = "v1ur - Petropolis Public Geodata Path Recovery" if has_missing_geodata else dec["required_next_action"]
        status_rows.append({
            "event_id": dec["event_id"],
            "v1uq_status": status,
            "phenomenon_separation_status": sep,
            "documentary_evidence_strength": dec["evidence_strength"],
            "has_missing_geodata_signal": bool_text(has_missing_geodata),
            "has_observed_geometry": "false",
            "ground_truth_operational": "false",
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "can_advance_to_overlay_preflight": "false",
            "main_blocker": "GEOMETRY_STILL_MISSING",
            "recommended_next_action": next_action,
            "notes": "v1uq status only; prior registries unchanged.",
        })
        for gate, ok, reason in [
            ("phenomenon_textual_separation", sep.startswith("PHENOMENON_SEPARATION"), "phenomenon_separation_incomplete"),
            ("documentary_temporal_link", dec["temporal_linkage_status"] == "TEMPORAL_TEXT_LINK_PRESENT", "temporal_link_missing_or_year_only"),
            ("locality_text_link", dec["locality_linkage_status"] == "LOCALITY_TEXT_LINK_PRESENT", "locality_link_missing"),
            ("observed_geometry", False, "geometry_still_missing"),
            ("overlay_preflight", False, "overlay_forbidden_in_v1uq"),
            ("ground_reference", False, "ground_reference_forbidden_in_v1uq"),
            ("training_label", False, "training_label_forbidden_in_v1uq"),
        ]:
            blockers.append({
                "blocker_id": f"BLOCK_v1uq_{dec['event_id']}_{gate}",
                "event_id": dec["event_id"],
                "gate": gate,
                "gate_status": "PASS_REVIEW_ONLY" if ok else "BLOCKED",
                "blocking_reason": "" if ok else reason,
                "ground_truth_operational": "false",
                "can_create_ground_reference": "false",
                "can_create_training_label": "false",
                "notes": "Ground reference remains blocked by design.",
            })
    write_csv(os.path.join(DATASET_DIR, "v1uq_petropolis_event_status_registry.csv"), EVENT_STATUS_COLUMNS, status_rows)
    write_csv(os.path.join(DATASET_DIR, "v1uq_petropolis_ground_reference_blocker_matrix.csv"), BLOCKER_COLUMNS, blockers)
    print(f"[v1uq event status] rows={len(status_rows)} blockers={len(blockers)}")
    return status_rows


def write_policy_configs():
    configs = {
        "v1uq_petropolis_pdf_text_policy.yaml": [
            "protocol_version: v1uq",
            "full_text_storage: local_only",
            "ocr_massive_allowed: false",
            "preferred_backend: pypdf",
            "fallback_backend: pdfplumber_if_available",
        ],
        "v1uq_petropolis_phenomenon_terms.yaml": [
            "protocol_version: v1uq",
            "flood_terms: [inundacao, alagamento, enchente, enxurrada, transbordamento, cheia, inundado]",
            "landslide_terms: [deslizamento, escorregamento, movimento de massa, corrida de massa, queda de barreira, instabilidade de encosta]",
            "context_terms: [risco geologico, suscetibilidade]",
        ],
        "v1uq_petropolis_locality_terms.yaml": [
            "protocol_version: v1uq",
            "geocoding_allowed: false",
            "locality_terms: [Petropolis, Quitandinha, Rio Quitandinha, Centro, Morin, Alto da Serra, Bingen, Cascatinha, Itaipava, Correas, Nogueira]",
        ],
        "v1uq_petropolis_date_linkage_policy.yaml": [
            "protocol_version: v1uq",
            "year_only_exact_event: false",
            "accepted_statuses: [EXACT_EVENT_DATE, EVENT_MONTH, EVENT_YEAR, POST_EVENT_REPORT, NO_TEMPORAL_LINK]",
        ],
        "v1uq_petropolis_decision_policy.yaml": [
            "protocol_version: v1uq",
            "status_max: PETROPOLIS_PHENOMENON_SEPARATION_EVIDENCE_FOR_REVIEW",
            "can_create_ground_reference: false",
            "can_create_training_label: false",
            "overlay_allowed: false",
        ],
        "v1uq_petropolis_missing_geodata_policy.yaml": [
            "protocol_version: v1uq",
            "invent_missing_file_allowed: false",
            "formal_request_required: false",
            "public_search_first: true",
        ],
    }
    for name, lines in configs.items():
        write_text(os.path.join(CONFIG_DIR, name), lines)


def choose_next_action(status_rows):
    if any(r["has_missing_geodata_signal"] == "true" for r in status_rows):
        return "v1ur - Petropolis Public Geodata Path Recovery"
    if any("TEXTUAL_NO_GEOMETRY" in r["v1uq_status"] for r in status_rows):
        return "v1ur - Petropolis Geometry Search from Missing Geodata Signals"
    return "v1ur - Curitiba Event Registry and Public Source Discovery"


def run_completion_report():
    write_policy_configs()
    text_rows = load_csv(os.path.join(DATASET_DIR, "v1uq_petropolis_pdf_text_extraction_registry.csv"))
    pages = load_csv(os.path.join(DATASET_DIR, "v1uq_petropolis_page_level_evidence_registry.csv"))
    locs = load_csv(os.path.join(DATASET_DIR, "v1uq_petropolis_locality_term_audit.csv"))
    dates = load_csv(os.path.join(DATASET_DIR, "v1uq_petropolis_event_date_linkage_audit.csv"))
    missing = load_csv(os.path.join(DATASET_DIR, "v1uq_petropolis_missing_geodata_signal_audit.csv"))
    status_rows = load_csv(os.path.join(DATASET_DIR, "v1uq_petropolis_event_status_registry.csv"))
    next_action = choose_next_action(status_rows)
    next_rows = [{
        "action_id": "ACT_v1uq_0000",
        "event_id": "PET_2022_02_15" if "Petropolis" in next_action else "",
        "action_type": "PROGRAMMING_DEEPENING",
        "priority": "1",
        "description": next_action,
        "target": "PET",
        "status": "PENDING",
        "notes": "Selected from v1uq document-audit gates; still non-operational.",
    }]
    write_csv(os.path.join(DATASET_DIR, "v1uq_next_actions_registry.csv"), NEXT_ACTION_COLUMNS, next_rows)
    manifest = []
    for idx, path in enumerate(V1UQ_ARTIFACTS):
        exists = os.path.exists(path)
        manifest.append({
            "artifact_id": f"ART_v1uq_{idx:04d}",
            "artifact_path": path.replace("\\", "/"),
            "artifact_type": "config" if path.startswith("configs/") else "doc" if path.startswith("docs/") else "dataset",
            "protocol_version": PROTOCOL_VERSION,
            "sha256_prefix": sha256_file(path)[:16] if exists else "MISSING",
            "file_size_bytes": str(os.path.getsize(path) if exists else 0),
            "is_versionable": bool_text(exists),
            "reason": "Safe v1uq engineering artifact" if exists else "File not found",
        })
    write_csv(os.path.join(DATASET_DIR, "v1uq_versionable_artifacts_manifest.csv"), MANIFEST_COLUMNS, manifest)
    class_counts = Counter(p["dominant_phenomenon_class"] for p in pages)
    flood_pages = sum(class_counts[c] for c in ["FLOOD_OR_INUNDATION", "URBAN_FLOODING", "FLASH_FLOOD_OR_RUNOFF", "MIXED_HYDRO_GEO"])
    land_pages = sum(class_counts[c] for c in ["LANDSLIDE_OR_MASS_MOVEMENT", "MIXED_HYDRO_GEO"])
    method = [
        "# Protocolo C v1uq - Petropolis Phenomenon Separation Deep Audit",
        "",
        "## Engineering Scope",
        "- Audits official PDF text extracted from v1up local raw artifacts.",
        "- Stores full extracted text only under local_only.",
        "- Public registries contain hashes, counts, page numbers, and gate statuses only.",
        "- Does not OCR, geocode, vectorize PDF maps, execute overlay, or create labels.",
    ]
    report = [
        "# Relatorio tecnico v1uq - Petropolis Phenomenon Separation Deep Audit",
        "",
        f"- pdfs_audited: {len(text_rows)}",
        f"- pages_with_text: {sum(int(r.get('pages_with_text') or 0) for r in text_rows)}",
        f"- pages_with_flood_signal: {flood_pages}",
        f"- pages_with_landslide_signal: {land_pages}",
        f"- pages_with_mixed_signal: {class_counts['MIXED_HYDRO_GEO']}",
        f"- locality_signal_rows: {len(locs)}",
        f"- temporal_signal_rows: {len(dates)}",
        f"- missing_geodata_signal_rows: {len(missing)}",
        f"- next_programming_step: {next_action}",
        "",
        "## Guardrails",
        "- ground_truth_operational=false",
        "- can_create_ground_reference=false",
        "- can_create_training_label=false",
        "- can_reopen_protocol_b=false",
        "- dino_usage=SUPPORT_ONLY",
        "- no_overlay_executed=true",
        "- no_coordinates_invented=true",
        "- patch_bound_truth=false",
        "- operational_validation=false",
        "- public_official_discovery=true",
        "- document_text_audit_only=true",
    ]
    pet2022 = next((r for r in status_rows if r["event_id"] == "PET_2022_02_15"), {})
    pet2024 = next((r for r in status_rows if r["event_id"] == "PET_2024_03_21_28"), {})
    status_doc = [
        "# Status Atual - Protocolo C v1uq",
        "",
        f"status_max={MAX_STATUS}",
        f"PET_2022_02_15={pet2022.get('v1uq_status', '')}",
        f"PET_2024_03_21_28={pet2024.get('v1uq_status', '')}",
        f"pdfs_audited={len(text_rows)}",
        f"pages_with_flood_signal={flood_pages}",
        f"pages_with_landslide_signal={land_pages}",
        f"pages_with_mixed_signal={class_counts['MIXED_HYDRO_GEO']}",
        f"recommended_next_action={next_action}",
        "ground_truth_operational=false",
        "can_create_ground_reference=false",
        "can_create_training_label=false",
        "can_reopen_protocol_b=false",
        "dino_usage=SUPPORT_ONLY",
        "no_overlay_executed=true",
        "no_coordinates_invented=true",
        "patch_bound_truth=false",
        "operational_validation=false",
        "public_official_discovery=true",
        "document_text_audit_only=true",
    ]
    write_text(os.path.join(DOCS_DIR, "protocolo_c_v1uq_petropolis_phenomenon_separation_deep_audit.md"), method)
    write_text(os.path.join(DOCS_DIR, "protocolo_c_relatorio_v1uq_petropolis_phenomenon_separation_deep_audit.md"), report)
    write_text(os.path.join(DOCS_DIR, "protocolo_c_status_atual_v1uq.md"), status_doc)
    print(f"[v1uq completion] next_action={next_action}")
    return {
        "pdfs_audited": len(text_rows),
        "flood_pages": flood_pages,
        "landslide_pages": land_pages,
        "mixed_pages": class_counts["MIXED_HYDRO_GEO"],
        "missing_geodata_signals": len(missing),
        "next_action": next_action,
    }


def parser_for(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--timeout", type=int, default=30)
    p.add_argument("--local-only-dir", default="")
    return p


def main_for(kind):
    args = parser_for(f"v1uq {kind}").parse_args()
    if kind == "pdf_text_extractor":
        return run_pdf_text_extractor(local_only_dir=args.local_only_dir or None, dry_run=args.dry_run)
    if kind == "pdf_structure_inventory":
        return run_pdf_structure_inventory()
    if kind == "phenomenon_term_indexer":
        return run_phenomenon_term_indexer()
    if kind == "page_level_evidence_builder":
        return run_page_level_evidence_builder()
    if kind == "locality_term_audit":
        return run_locality_term_audit()
    if kind == "event_date_linkage_audit":
        return run_event_date_linkage_audit()
    if kind == "phenomenon_separation_decision_matrix":
        return run_phenomenon_separation_decision_matrix()
    if kind == "missing_geodata_signal_audit":
        return run_missing_geodata_signal_audit()
    if kind == "event_status_updater":
        return run_event_status_updater()
    if kind == "completion_report":
        return run_completion_report()
    raise ValueError(kind)
