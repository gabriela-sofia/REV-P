#!/usr/bin/env python3
"""v2bj Recife intake-result consolidation and candidate-gate reconciliation, fail-closed.

Consolidates the real evidence acquired by the v2bi manual intake (Charter 758 map,
APAC accumulated precipitation, ANA Capibaribe river level) plus the INMET raw
availability for Recife and its regional proxies, then reconciles the Recife
candidate gates C0-C7 WITHOUT any promotion. No ground truth, label, negative or
training is produced. River level is not precipitation; precipitation is not flood
extent; a landslide scar map is not flood extent; a regional proxy is not the local
Recife station; a CRS-less raster map is not a vector geometry.
"""

import argparse
import csv
import datetime as dt
import hashlib
import os
import re
import zipfile

DATASET_DIR = os.environ.get("DATASET_DIR", "datasets/protocolo_c")
DOCS_DIR = os.environ.get("V2BJ_DOCS_DIR", "docs/protocolo_c/v2bj_recife_candidate_gate_reconciliation")
INMET_ZIP = os.environ.get("V2BJ_INMET_ZIP", "data/external_raw/inmet/historical/inmet_2022.zip")
CHARTER_CACHE = os.environ.get("V2BJ_CHARTER_CACHE",
                               "docs/protocolo_c/v2bi_recife_charter_temporal_intake/evidence_cache/manual_charter_758")
REFRESH_V2BI = os.environ.get("V2BJ_REFRESH_V2BI", "1") == "1"

CANDIDATE_ID = "REC_2022_05_24_30"
PRODUCT_ID = "CH758_RECIFE_20220602_001"

INVARIANTS = {
    "can_create_ground_truth": "false", "can_create_patch_truth": "false", "can_create_label": "false",
    "can_create_negative": "false", "can_train_model": "false",
    "charter_map_is_not_vector_geometry": "true", "charter_geometry_requires_human_review": "true",
    "crs_missing_blocks_geometry_promotion": "true", "river_level_is_not_precipitation": "true",
    "precipitation_is_not_flood_extent": "true", "landslide_scar_is_not_flood_extent": "true",
    "regional_proxy_is_not_local_recife": "true", "a301_empty_precip_not_substituted": "true",
    "olinda_product_is_not_recife": "true", "raw_data_versioned": "false",
}

# INMET BDMEP automatic stations relevant to the Recife window (code, name, uf, role).
INMET_STATIONS = [
    ("A301", "RECIFE", "PE", "LOCAL_RECIFE"),
    ("A357", "PALMARES", "PE", "REGIONAL_PROXY"),
    ("A328", "SURUBIM", "PE", "REGIONAL_PROXY"),
    ("A320", "JOAO PESSOA", "PB", "REGIONAL_PROXY"),
]

OUTPUTS = {
    "summary": "v2bj_recife_intake_result_summary.csv",
    "inmet": "v2bj_inmet_proxy_availability_audit.csv",
    "gates": "v2bj_recife_candidate_gate_reconciliation.csv",
    "queue": "v2bj_recife_candidate_reference_queue.csv",
    "guardrail": "v2bj_guardrail_regression.csv",
    "manifest": "v2bj_orchestrator_manifest.csv",
}

# v2bi live outputs consumed here.
V2BI = {
    "cache": "v2bi_manual_intake_cache_inventory.csv", "charter_audit": "v2bi_charter_file_audit.csv",
    "charter_readiness": "v2bi_charter_candidate_geometry_readiness.csv",
    "temporal_cache": "v2bi_temporal_series_cache_inventory.csv", "parse": "v2bi_apac_cemaden_series_parse_report.csv",
    "metrics": "v2bi_recife_temporal_metrics.csv", "gates": "v2bi_recife_protocol_gate_update.csv",
}
V2AZ_QUEUE = "v2az_recife_gap_review_queue.csv"


def parse_args(argv=None):
    return argparse.ArgumentParser(description="v2bj orchestrator").parse_args(argv)


def dataset_path(name):
    return os.path.join(DATASET_DIR, name)


def doc_path(*parts):
    return os.path.join(DOCS_DIR, *parts)


def with_invariants(row):
    return {**row, **INVARIANTS}


def clean(value):
    return str(value or "").strip()


def is_true(value):
    return clean(value).lower() == "true"


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing empty output: {path}")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recife_window():
    for row in load_csv(dataset_path(V2AZ_QUEUE)):
        if row.get("candidate_id") == CANDIDATE_ID:
            return row.get("window_start", ""), row.get("window_end", ""), row.get("event_date", "")
    return "", "", ""


def refresh_v2bi():
    """Regenerate the v2bi intake outputs from the live (real) cache so that this
    consolidation reflects whatever manual evidence is actually present."""
    if not REFRESH_V2BI:
        return "SKIPPED"
    try:
        import revp_v2bi_common as v2bi
    except ImportError:
        import scripts.protocolo_c.revp_v2bi_common as v2bi
    v2bi.run_orchestrator()
    return "REFRESHED"


# --------------------------------------------------------------------------- #
# Charter metadata extracted from the cached activation page (when present).
# --------------------------------------------------------------------------- #

def charter_metadata_html():
    if not os.path.isdir(CHARTER_CACHE):
        return None
    for name in sorted(os.listdir(CHARTER_CACHE)):
        if name.lower().endswith((".html", ".htm")):
            return os.path.join(CHARTER_CACHE, name)
    return None


def extract_charter_facts():
    """Read feature type, license and product date for the Recife product from the
    cached activation HTML. Fail-closed to UNKNOWN when the file is absent."""
    facts = {"feature_type_candidate": "UNKNOWN", "license_terms": "UNKNOWN",
             "product_date": "UNKNOWN", "source_html_present": "false"}
    path = charter_metadata_html()
    if not path:
        return facts
    facts["source_html_present"] = "true"
    try:
        html = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return facts
    # Anchor on the Recife product slug inside the embedded JSON payload, not on the
    # visible card, so the license and acquisition date belong to the Recife product.
    anchor = html.find("landslides-scars-in-recife")
    if anchor < 0:
        slug = re.search(r"vapArticleSlug[^a-z]*([a-z0-9-]*recife[a-z0-9-]*)", html)
        if not slug:
            return facts
        anchor = slug.start()
    facts["feature_type_candidate"] = "LANDSLIDE_SCARS" if "landslides-scars-in-recife" in html \
        else "UNKNOWN"
    block = html[anchor:anchor + 900]
    lic = re.search(r"vapSourcesCopyrights[^A-Za-z]*([^\"\\]{6,180})", block)
    if lic:
        terms = lic.group(1).replace("<br>", " ").replace("\\n", " ").replace("�", "(c)")
        facts["license_terms"] = clean(re.sub(r"\s+", " ", terms))
    acq = re.search(r"vapAcquired[^0-9]*([0-9]{10,13})", block)
    if acq:
        ms = int(acq.group(1))
        facts["product_date"] = dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).date().isoformat()
    return facts


# --------------------------------------------------------------------------- #
# INMET raw availability audit (data availability, not a promoted series).
# --------------------------------------------------------------------------- #

def _inmet_member(names, code):
    for name in names:
        if f"_{code}_" in name.upper() and name.upper().endswith(".CSV"):
            return name
    return ""


def _inmet_precip_window(text, start, end):
    lines = text.splitlines()
    header_idx = next((i for i, line in enumerate(lines) if line.upper().startswith("DATA;")), -1)
    if header_idx < 0:
        return 0, 0, 0.0
    rows, nonempty, total = 0, 0, 0.0
    s = start.replace("-", "/")
    e = end.replace("-", "/")
    for line in lines[header_idx + 1:]:
        if not line.strip():
            continue
        cols = line.split(";")
        if len(cols) < 3:
            continue
        day = cols[0].strip()
        if not (s <= day <= e):
            continue
        rows += 1
        raw = cols[2].strip().replace(",", ".")
        if raw == "":
            continue
        try:
            total += float(raw)
            nonempty += 1
        except ValueError:
            pass
    return rows, nonempty, round(total, 2)


def run_audit_inmet_proxy_availability(args=None):
    start, end, _ = recife_window()
    rows = []
    present = os.path.exists(INMET_ZIP)
    names = []
    if present:
        try:
            with zipfile.ZipFile(INMET_ZIP) as archive:
                names = archive.namelist()
        except (zipfile.BadZipFile, OSError):
            present = False
    for i, (code, name, uf, role) in enumerate(INMET_STATIONS, 1):
        member = _inmet_member(names, code) if present else ""
        if member:
            with zipfile.ZipFile(INMET_ZIP) as archive:
                text = archive.read(member).decode("latin-1", errors="replace")
            win_rows, precip_records, precip_total = _inmet_precip_window(text, start, end)
            if precip_records == 0:
                coverage = "PRECIP_FULL_GAP"
            elif precip_records >= win_rows * 0.8 and win_rows:
                coverage = "PRECIP_AVAILABLE"
            else:
                coverage = "PRECIP_PARTIAL"
            raw_present = "true"
        else:
            win_rows = precip_records = 0
            precip_total = 0.0
            coverage = "RAW_NOT_PRESENT" if not present else "STATION_NOT_FOUND"
            raw_present = "false"
        if code == "A301":
            note = ("Local Recife station precipitation column empty for the window; "
                    "not usable as local rainfall and not substituted by any proxy.") if coverage == "PRECIP_FULL_GAP" \
                else "Local Recife station; precipitation availability audited only, not promoted."
        elif code == "A320":
            note = "Regional proxy in Joao Pessoa/PB (different city and state); never a local Recife substitute."
        else:
            note = "Regional proxy in PE interior; comparison only, never a local Recife substitute."
        rows.append(with_invariants({
            "inmet_audit_id": f"INMETAV_v2bj_{i:03d}", "station_code": code, "station_name": name, "uf": uf,
            "station_role": role, "raw_present": raw_present, "window_start": start, "window_end": end,
            "window_rows": str(win_rows), "precip_records_in_window": str(precip_records),
            "precip_total_mm_window": f"{precip_total:.2f}", "coverage_status": coverage,
            "usable_as_recife_local_rainfall": "false", "note": note,
        }))
    write_csv(dataset_path(OUTPUTS["inmet"]), rows)
    return rows


# --------------------------------------------------------------------------- #
# Intake result summary: one row per real evidence track.
# --------------------------------------------------------------------------- #

def _charter_state():
    audit = load_csv(dataset_path(V2BI["charter_audit"]))
    map_present = any(r.get("audit_status") in {"PREVIEW_ONLY_FOUND", "MAP_ONLY_FOUND"} for r in audit)
    vector_present = any(is_true(r.get("vector_candidate_found")) for r in audit)
    readiness = load_csv(dataset_path(V2BI["charter_readiness"]))
    status = readiness[0]["updated_candidate_status"] if readiness else "NO_FILE_AVAILABLE"
    return map_present, vector_present, status


def _temporal_files_by_source():
    found = {}
    for row in load_csv(dataset_path(V2BI["temporal_cache"])):
        src = row.get("source_candidate", "UNKNOWN")
        found.setdefault(src, []).append(row)
    return found


def run_build_recife_intake_result_summary(args=None):
    facts = extract_charter_facts()
    map_present, vector_present, _ = _charter_state()
    temporal = _temporal_files_by_source()
    parse = load_csv(dataset_path(V2BI["parse"]))
    parsed_ok = any(r.get("parse_status") == "PARSED" for r in parse)
    inmet_audit = load_csv(dataset_path(OUTPUTS["inmet"]))
    a301 = next((r for r in inmet_audit if r["station_code"] == "A301"), {})

    rows = []

    def add(track, source, institution, kind, present, parseable, dated, feature, role, supports, advances, limitation):
        rows.append(with_invariants({
            "intake_item_id": f"INTK_v2bj_{len(rows) + 1:03d}", "evidence_track": track, "source": source,
            "institution": institution, "item_kind": kind, "file_present": present,
            "machine_parseable": parseable, "dated_in_window": dated, "feature_or_variable": feature,
            "role": role, "supports_gate": supports, "advances_to": advances, "limitation": limitation,
        }))

    add("CHARTER", "International Charter Activation 758", "CENAD / CNES / Airbus DS",
        "raster_map_png", "true" if map_present else "false", "false", "unknown",
        facts["feature_type_candidate"], "OFFICIAL_DISASTER_MAP",
        "C3_SPATIAL_ANCHOR|C4_CANDIDATE_GEOMETRY",
        "MAP_HUMAN_REVIEW" if map_present else "MANUAL_DOWNLOAD_REQUIRED",
        ("Full-resolution map raster present but carries no machine-readable CRS and no vector; "
         f"license: {facts['license_terms']}; feature is landslide scars, not flood extent."))
    add("CHARTER", "International Charter Activation 758 (page metadata)", "disasterscharter.org",
        "metadata_html", facts["source_html_present"], "true", "unknown",
        facts["feature_type_candidate"], "PROVENANCE_LICENSE_RECORD", "C0_PROVENANCE|C4_CANDIDATE_GEOMETRY",
        "PROVENANCE_REVIEW", f"Source/license/feature extracted from cached page; product date {facts['product_date']}.")
    add("CHARTER", "International Charter Activation 758 (vector product)", "CENAD",
        "vector_geometry", "true" if vector_present else "false", "false", "unknown",
        facts["feature_type_candidate"], "GEOMETRY_SOURCE",
        "C4_CANDIDATE_GEOMETRY", "VECTOR_HUMAN_REVIEW" if vector_present else "REQUEST_REQUIRED",
        "No public vector/CRS exposed; must be requested from CENAD/Charter." if not vector_present
        else "Vector candidate present; CRS and geometry require validation and human review.")

    apac = temporal.get("APAC", [])
    add("TEMPORAL", "APAC monthly accumulated precipitation (May 2022)", "APAC-PE",
        "pdf_bulletin", "true" if apac else "false", "false", "true" if apac else "unknown",
        "monthly_accumulated_precipitation", "TEMPORAL_CONTEXT", "C1_TEMPORALITY",
        "TEMPORALITY_CONTEXT_REVIEW",
        "Monthly aggregate PDF; establishes event-month magnitude but is not a parseable sub-daily series.")
    ana = temporal.get("ANA_HIDROWEB", [])
    add("TEMPORAL", "ANA HidroWeb Capibaribe - Sao Lourenco da Mata (39187800)", "ANA",
        "river_level_series", "true" if ana else "false", "false", "true" if ana else "unknown",
        "river_level_cota", "HYDROLOGICAL_PROXY_NEARBY_RMR", "C1_TEMPORALITY|C2_VALID_SERIES_OR_STATION",
        "TEMPORALITY_AND_STATION_REVIEW",
        ("Dated in-window river stage from an RMR station upstream of Recife; river level is not "
         "precipitation and not flood extent; not the local Recife rainfall station."))
    add("TEMPORAL", "Cemaden pluviometers Recife/RMR (May 2022)", "Cemaden",
        "precipitation_series", "true" if temporal.get("CEMADEN") else "false",
        "true" if parsed_ok else "false", "unknown", "precipitation", "LOCAL_RAINFALL_CANDIDATE",
        "C1_TEMPORALITY|C2_VALID_SERIES_OR_STATION",
        "PARSE_REVIEW" if temporal.get("CEMADEN") else "MANUAL_DOWNLOAD_REQUIRED",
        "Interactive map selection with e-mail-delivered link; not retrievable non-interactively.")

    add("INMET_PROXY", "INMET BDMEP A301 Recife", "INMET", "precipitation_series",
        a301.get("raw_present", "false"), "false",
        "true" if a301.get("precip_records_in_window", "0") != "0" else "false",
        "precipitation", "LOCAL_RECIFE_STATION", "C1_TEMPORALITY|C2_VALID_SERIES_OR_STATION",
        "AVAILABILITY_DOCUMENTED",
        f"Local Recife precipitation availability: {a301.get('coverage_status', 'UNKNOWN')}; "
        "not usable as local rainfall when empty and never substituted by a regional proxy.")

    write_csv(dataset_path(OUTPUTS["summary"]), rows)
    return rows


# --------------------------------------------------------------------------- #
# Gate reconciliation (no promotion).
# --------------------------------------------------------------------------- #

def _v2bi_gate(gate_id):
    for row in load_csv(dataset_path(V2BI["gates"])):
        if row.get("candidate_id") == CANDIDATE_ID and row.get("gate_id") == gate_id:
            return row
    return {}


def reconcile_gates():
    map_present, _, charter_status = _charter_state()
    summary = load_csv(dataset_path(OUTPUTS["summary"]))
    ana_present = any(r["source"].startswith("ANA HidroWeb") and is_true(r["file_present"]) for r in summary)
    apac_present = any(r["source"].startswith("APAC") and is_true(r["file_present"]) for r in summary)
    parsed_ready = any(r.get("temporal_status") == "TEMPORAL_EVIDENCE_READY_FOR_REVIEW"
                       for r in load_csv(dataset_path(V2BI["metrics"])))

    dated_temporal = ana_present or apac_present or parsed_ready

    gates = {}
    gates["C0_PROVENANCE"] = (
        "PASS_FOR_REVIEW" if (map_present or charter_status != "NO_FILE_AVAILABLE") else "PENDING",
        "Charter 758 official mapping source documented (page, license, sources)." if map_present
        else "Provenance pending evidence file.",
        "CHARTER_758|ANA|APAC" if map_present else "", "REVIEW_PROVENANCE_RECORD")
    gates["C1_TEMPORALITY"] = (
        "TEMPORALITY_SUPPORTED_FOR_HUMAN_REVIEW" if dated_temporal else "PENDING",
        "Dated in-window evidence present (ANA Capibaribe level / APAC May accumulation)." if dated_temporal
        else "No dated in-window temporal evidence parsed.",
        "ANA_HIDROWEB|APAC" if dated_temporal else "", "HUMAN_REVIEW_TEMPORALITY")
    if parsed_ready:
        c2 = ("PASS_FOR_HUMAN_REVIEW_ONLY", "Parsed local precipitation series covers the window.",
              "CEMADEN_OR_APAC_SERIES", "HUMAN_REVIEW_LOCAL_SERIES")
    elif ana_present:
        c2 = ("PARTIAL_FOR_HUMAN_REVIEW",
              "ANA river-level station metadata sufficient for review; local Recife rainfall station "
              "(A301) precipitation empty and Cemaden series pending.",
              "ANA_HIDROWEB:39187800", "ACQUIRE_LOCAL_RAINFALL_SERIES")
    else:
        c2 = ("BLOCKED", "No valid local rainfall series or reviewable station.",
              "", "ACQUIRE_LOCAL_RAINFALL_SERIES")
    gates["C2_VALID_SERIES_OR_STATION"] = c2
    c3 = _v2bi_gate("C3_SPATIAL_ANCHOR")
    gates["C3_SPATIAL_ANCHOR"] = (
        c3.get("updated_status", "PASS") if c3 else "PASS",
        "Charter Recife product confirms the spatial anchor.", "CH758_RECIFE_20220602_001",
        "REVIEW_PRODUCT_GEOMETRY")
    if charter_status == "CANDIDATE_GEOMETRY_READY_FOR_HUMAN_REVIEW":
        c4 = ("PASS_FOR_HUMAN_REVIEW_ONLY", "Validated vector, CRS and Recife bbox.",
              "CHARTER_VECTOR", "HUMAN_REVIEW_GEOMETRY")
    elif map_present:
        c4 = ("MAP_PRESENT_PENDING_VECTOR_CRS",
              "Full-resolution Charter map raster present and reviewable; no machine-readable CRS and "
              "no vector, so geometry promotion stays blocked.", "CHARTER_MAP_RASTER",
              "REQUEST_CHARTER_VECTOR_CRS_FROM_CENAD")
    else:
        c4 = ("PENDING_VECTOR_CRS", "No Charter map or vector available.", "",
              "MANUAL_DOWNLOAD_OR_REQUEST_CHARTER_PRODUCT")
    gates["C4_CANDIDATE_GEOMETRY"] = c4
    gates["C5_HUMAN_REVIEW"] = (
        "PENDING", "Intake inputs assembled; awaiting human review decision.", "", "EXECUTE_HUMAN_REVIEW")
    c1_ok = dated_temporal
    c2_ok = c2[0] in {"PASS_FOR_HUMAN_REVIEW_ONLY", "PARTIAL_FOR_HUMAN_REVIEW"}
    c4_reviewable = c4[0] in {"PASS_FOR_HUMAN_REVIEW_ONLY", "MAP_PRESENT_PENDING_VECTOR_CRS"}
    if c1_ok and c2_ok and c4_reviewable:
        gates["C6_CANDIDATE_REFERENCE"] = (
            "CANDIDATE_REFERENCE_PENDING_HUMAN_REVIEW",
            "C1 supported, C2 reviewable (proxy/partial), C3 pass, C4 reviewable; reference decision "
            "deferred to human review and gated on resolving the local rainfall series.",
            "INTAKE_BUNDLE", "HUMAN_REVIEW_CANDIDATE_REFERENCE")
    else:
        gates["C6_CANDIDATE_REFERENCE"] = (
            "BLOCKED", "Upstream gates not sufficiently reviewable.", "", "RESOLVE_UPSTREAM_GATES")
    gates["C7_FINAL_GROUND_TRUTH"] = (
        "BLOCKED", "Final ground truth prohibited; intake is review-only.", "",
        "NONE_FINAL_GROUND_TRUTH_PROHIBITED")
    return gates


def run_reconcile_recife_candidate_gates(args=None):
    gates = reconcile_gates()
    packet = next((r for r in load_csv(dataset_path(V2AZ_QUEUE)) if r.get("candidate_id") == CANDIDATE_ID), {})
    rows = []
    for gate_id in ("C0_PROVENANCE", "C1_TEMPORALITY", "C2_VALID_SERIES_OR_STATION", "C3_SPATIAL_ANCHOR",
                    "C4_CANDIDATE_GEOMETRY", "C5_HUMAN_REVIEW", "C6_CANDIDATE_REFERENCE", "C7_FINAL_GROUND_TRUTH"):
        status, reason, evidence, action = gates[gate_id]
        v2bi_row = _v2bi_gate(gate_id)
        rows.append(with_invariants({
            "candidate_id": CANDIDATE_ID, "recife_package_id": packet.get("review_packet_id", "ARP_v2az_0005"),
            "event_patch_package_id": packet.get("event_patch_package_id", "FACT_v2at_0005"), "gate_id": gate_id,
            "v2bi_status": v2bi_row.get("updated_status", ""), "reconciled_status": status,
            "promotion_allowed": "false", "evidence_used": evidence, "reconciliation_reason": reason,
            "human_action_required": action,
        }))
    write_csv(dataset_path(OUTPUTS["gates"]), rows)
    return rows


# --------------------------------------------------------------------------- #
# Candidate reference queue.
# --------------------------------------------------------------------------- #

def run_build_recife_candidate_reference_queue(args=None):
    gates = {r["gate_id"]: r["reconciled_status"] for r in load_csv(dataset_path(OUTPUTS["gates"]))}
    rows = []
    for packet in load_csv(dataset_path(V2AZ_QUEUE)):
        candidate = packet.get("candidate_id", "")
        if candidate == CANDIDATE_ID:
            reference_status = gates.get("C6_CANDIDATE_REFERENCE", "BLOCKED")
            blocker = ("LOCAL_RAINFALL_SERIES_AND_CHARTER_VECTOR_CRS_PENDING"
                       if reference_status.startswith("CANDIDATE_REFERENCE") else "UPSTREAM_GATES_UNRESOLVED")
            actions = "REQUEST_CHARTER_VECTOR_CRS_FROM_CENAD|ACQUIRE_CEMADEN_APAC_LOCAL_SERIES|EXECUTE_HUMAN_REVIEW"
            note = ("Recife May 2022 has a reviewable Charter landslide-scar map and dated hydrological "
                    "context; reference decision is deferred to human review, not promoted.")
        else:
            reference_status = "BLOCKED_NO_INTAKE"
            blocker = "NO_CHARTER_PRODUCT_OR_INTAKE_FOR_THIS_EVENT"
            actions = "OUT_OF_CURRENT_INTAKE_SCOPE"
            note = "No v2bi intake evidence acquired for this event; remains fail-closed."
        rows.append(with_invariants({
            "candidate_id": candidate, "recife_package_id": packet.get("review_packet_id", ""),
            "region": packet.get("region", ""), "event_date": packet.get("event_date", ""),
            "reference_status": reference_status, "c0": gates.get("C0_PROVENANCE", "") if candidate == CANDIDATE_ID else "",
            "c1": gates.get("C1_TEMPORALITY", "") if candidate == CANDIDATE_ID else "",
            "c2": gates.get("C2_VALID_SERIES_OR_STATION", "") if candidate == CANDIDATE_ID else "",
            "c3": gates.get("C3_SPATIAL_ANCHOR", "") if candidate == CANDIDATE_ID else "",
            "c4": gates.get("C4_CANDIDATE_GEOMETRY", "") if candidate == CANDIDATE_ID else "",
            "c5": gates.get("C5_HUMAN_REVIEW", "") if candidate == CANDIDATE_ID else "",
            "c6": reference_status if candidate == CANDIDATE_ID else "",
            "c7": "BLOCKED", "blocker_remaining": blocker, "required_human_actions": actions, "note": note,
        }))
    write_csv(dataset_path(OUTPUTS["queue"]), rows)
    return rows


# --------------------------------------------------------------------------- #
# Review packets (markdown).
# --------------------------------------------------------------------------- #

def run_generate_recife_candidate_review_packets(args=None):
    summary = load_csv(dataset_path(OUTPUTS["summary"]))
    gates = load_csv(dataset_path(OUTPUTS["gates"]))
    inmet = load_csv(dataset_path(OUTPUTS["inmet"]))
    start, end, event_date = recife_window()
    lines = [f"# Recife candidate status after intake - {CANDIDATE_ID}", "",
             f"Event date: `{event_date}` | Window: `{start}` to `{end}` | Package: `ARP_v2az_0005`.", "",
             "Review-only consolidation. No ground truth, label, negative or training is produced.", "",
             "## Intake evidence", "",
             "| track | source | present | parseable | feature/variable | role | limitation |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in summary:
        lines.append(f"| {r['evidence_track']} | {r['source']} | {r['file_present']} | "
                     f"{r['machine_parseable']} | {r['feature_or_variable']} | {r['role']} | {r['limitation']} |")
    lines += ["", "## Gate reconciliation (no promotion)", "",
              "| gate | v2bi | reconciled | human action |", "| --- | --- | --- | --- |"]
    for r in gates:
        lines.append(f"| {r['gate_id']} | {r['v2bi_status']} | {r['reconciled_status']} | {r['human_action_required']} |")
    lines += ["", "## INMET availability (data audit only)", "",
              "| station | role | coverage | precip records in window | usable as local rainfall |",
              "| --- | --- | --- | --- | --- |"]
    for r in inmet:
        lines.append(f"| {r['station_code']} {r['station_name']} | {r['station_role']} | {r['coverage_status']} | "
                     f"{r['precip_records_in_window']} | {r['usable_as_recife_local_rainfall']} |")
    lines += ["", "## What human review must resolve", "",
              "1. Request the Charter 758 Recife vector product and CRS from CENAD/Charter (C4).",
              "2. Acquire a local Recife rainfall series (Cemaden/APAC) for the window (C2); A301 precipitation is empty.",
              "3. Confirm the Charter feature is landslide scars, not flood extent, before any geometry use.",
              "4. Keep Olinda products separate; do not transfer to Recife.", ""]
    write_text(doc_path("candidate_review_packets", f"{CANDIDATE_ID}.md"), "\n".join(lines))
    return [{"packet": f"{CANDIDATE_ID}.md"}]


def run_generate_readme(args=None):
    queue = load_csv(dataset_path(OUTPUTS["queue"]))
    primary = next((r for r in queue if r["candidate_id"] == CANDIDATE_ID), {})
    reference = primary.get("reference_status", "BLOCKED")
    write_text(doc_path("README.md"), f"""# v2bj Recife Candidate Gate Reconciliation

Consolidacao dos resultados reais do intake v2bi (Charter 758, APAC, ANA HidroWeb) e da
auditoria de disponibilidade INMET, com reconciliacao dos gates C0-C7 de Recife. Nada e
promovido: o status de referencia do candidato `{CANDIDATE_ID}` e `{reference}`.

Charter Recife: mapa raster em resolucao plena presente para revisao, sem vetor e sem CRS
legivel -> C4 permanece pendente de vetor/CRS. Feicao = landslide scars (nao flood extent).
Temporal: ANA Capibaribe (Sao Lourenco da Mata, RMR) fornece cota datada na janela como
contexto hidrologico; APAC mensal como contexto; serie local de chuva de Recife continua
pendente (A301 com precipitacao vazia, Cemaden por download manual).

C3 permanece PASS. C7 permanece BLOCKED. Ground truth final, labels, negativos e treino = 0.
River level nao e precipitacao; precipitacao nao e flood extent; proxy regional nao e a
estacao local de Recife.
""")
    return [{"readme": "README.md"}]


# --------------------------------------------------------------------------- #
# Guardrail regression.
# --------------------------------------------------------------------------- #

def run_guardrail_regression(args=None):
    forbidden = {"can_create_ground_truth", "can_create_patch_truth", "can_create_label",
                 "can_create_negative", "can_train_model", "raw_data_versioned"}
    rows = []
    for number, name in enumerate((OUTPUTS["summary"], OUTPUTS["inmet"], OUTPUTS["gates"], OUTPUTS["queue"]), 1):
        data = load_csv(dataset_path(name))
        violations = sum(clean(r.get(field)).lower() == "true" for r in data for field in forbidden)
        rows.append({"regression_id": f"GR_v2bj_{number:03d}", "check": f"forbidden_flags::{name}",
                     "detail": "no forbidden invariant is true", "violation_count": str(violations),
                     "status": "PASS" if not violations else "FAIL"})
    gate_rows = load_csv(dataset_path(OUTPUTS["gates"]))
    c7 = [r for r in gate_rows if r["gate_id"] == "C7_FINAL_GROUND_TRUTH"]
    c7_ok = bool(c7) and all(r["reconciled_status"] == "BLOCKED" for r in c7)
    rows.append({"regression_id": "GR_v2bj_005", "check": "c7_blocked", "detail": "C7 stays BLOCKED",
                 "violation_count": "0" if c7_ok else "1", "status": "PASS" if c7_ok else "FAIL"})
    promo = sum(clean(r.get("promotion_allowed")).lower() == "true" for r in gate_rows)
    rows.append({"regression_id": "GR_v2bj_006", "check": "no_promotion", "detail": "no gate promotes",
                 "violation_count": str(promo), "status": "PASS" if not promo else "FAIL"})
    inmet = load_csv(dataset_path(OUTPUTS["inmet"]))
    substituted = sum(clean(r.get("usable_as_recife_local_rainfall")).lower() == "true" for r in inmet)
    rows.append({"regression_id": "GR_v2bj_007", "check": "proxy_not_substituted",
                 "detail": "no INMET proxy marked usable as local rainfall",
                 "violation_count": str(substituted), "status": "PASS" if not substituted else "FAIL"})
    if any(r["status"] != "PASS" for r in rows):
        raise ValueError("v2bj guardrail regression failed")
    write_csv(dataset_path(OUTPUTS["guardrail"]), rows)
    return rows


def _steps():
    return [
        ("audit_inmet_proxy_availability", run_audit_inmet_proxy_availability, dataset_path(OUTPUTS["inmet"])),
        ("build_recife_intake_result_summary", run_build_recife_intake_result_summary, dataset_path(OUTPUTS["summary"])),
        ("reconcile_recife_candidate_gates", run_reconcile_recife_candidate_gates, dataset_path(OUTPUTS["gates"])),
        ("build_recife_candidate_reference_queue", run_build_recife_candidate_reference_queue, dataset_path(OUTPUTS["queue"])),
        ("generate_recife_candidate_review_packets", run_generate_recife_candidate_review_packets,
         doc_path("candidate_review_packets", f"{CANDIDATE_ID}.md")),
        ("generate_readme", run_generate_readme, doc_path("README.md")),
        ("run_guardrail_regression", run_guardrail_regression, dataset_path(OUTPUTS["guardrail"])),
    ]


def ensure_structure():
    for folder in (DOCS_DIR, doc_path("candidate_review_packets")):
        os.makedirs(folder, exist_ok=True)


def run_orchestrator(args=None):
    ensure_structure()
    refresh_status = refresh_v2bi()
    v2bi_gates = dataset_path(V2BI["gates"])
    manifest = [{"step_order": "0", "step_name": "refresh_v2bi_intake", "status": refresh_status,
                 "output": v2bi_gates.replace("\\", "/"),
                 "output_hash": sha256(v2bi_gates)[:16] if os.path.exists(v2bi_gates) else "",
                 "notes": "Regenerates v2bi outputs from the live intake cache."}]
    for number, (name, function, path) in enumerate(_steps(), 1):
        function(args)
        manifest.append({"step_order": str(number), "step_name": name, "status": "OK",
                         "output": path.replace("\\", "/"), "output_hash": sha256(path)[:16],
                         "notes": "Fail-closed reconciliation; no promotion, no ground truth."})
    write_csv(dataset_path(OUTPUTS["manifest"]), manifest)
    return manifest


if __name__ == "__main__":
    run_orchestrator(parse_args())
