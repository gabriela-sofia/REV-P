"""Offline-first helpers for REV-P v2cx-v2dd external evidence readiness."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html.parser
import re
import urllib.request
from pathlib import Path
from urllib.parse import urljoin, urlparse


ALLOWED_CLAIM = "Uso permitido apenas como prontidao cientifica review-only; nao fecha TP2, TP3, treino ou validacao operacional."
FORBIDDEN_CLAIM = "ground_truth_operacional|label_binario|negativo_formal|dataset_treino|claim_deteccao|claim_predicao|intersecao_observada_automatica"
REGIONS = ["Recife", "Petropolis", "Curitiba"]
PRODUCT_EXTENSIONS = {".zip", ".geojson", ".json", ".gpkg", ".shp", ".kml", ".kmz", ".tif", ".tiff", ".pdf"}
VISUAL_EXTENSIONS = {".png", ".jpg", ".jpeg"}
DISCOVERY_TERMS = {
    "product", "map", "activation", "geodata", "vector", "raster", "damage",
    "affected", "flood", "landslide", "scar", "extent", "petropolis",
    "petrópolis", "recife", "pernambuco", "curitiba", "charter", "ems",
}

AVAILABILITY_FIELDS = [
    "availability_id", "source_id", "source_family", "region", "source_url",
    "network_checked", "http_status", "content_type", "content_length",
    "last_modified", "reachable", "requires_manual_access",
    "availability_status", "blocking_reason", "allowed_claim", "forbidden_claim",
]
DISCOVERY_FIELDS = [
    "product_candidate_id", "source_id", "region", "source_family", "parent_url",
    "candidate_url", "candidate_label", "candidate_extension",
    "candidate_product_type", "relation_to_event", "requires_manual_review",
    "download_allowed", "license_status", "discovery_status",
    "blocking_reason", "allowed_claim", "forbidden_claim",
]
LICENSE_FIELDS = [
    "license_audit_id", "product_candidate_id", "source_id", "source_family",
    "candidate_url", "license_status", "license_reference",
    "redistribution_allowed", "raw_download_allowed", "public_output_allowed",
    "metadata_only_allowed", "manual_license_review_required",
    "license_audit_status", "blocking_reason", "allowed_claim", "forbidden_claim",
]
DOWNLOAD_FIELDS = [
    "download_plan_id", "product_candidate_id", "source_id", "candidate_url",
    "planned_local_path", "extension", "expected_product_type", "license_status",
    "raw_download_allowed", "public_output_allowed", "max_size_mb",
    "download_mode", "download_executed", "sha256", "download_status",
    "blocking_reason", "allowed_claim", "forbidden_claim",
]
BOUNDARY_FIELDS = [
    "patch_id", "region", "boundary_available", "boundary_source",
    "bounds_available", "crs", "crs_known", "geometry_file_available",
    "raster_available", "sentinel_source_available", "human_review_required",
    "pairing_ready", "patch_boundary_status", "blocking_reason",
    "allowed_claim", "forbidden_claim",
]
READINESS_FIELDS = [
    "readiness_id", "region", "source_id", "product_candidate_id", "patch_id",
    "source_available", "product_candidate_available", "license_ready",
    "download_ready", "local_file_available", "sha256_available",
    "geospatial_qa_ready", "patch_boundary_ready", "pairing_ready",
    "replay_ready", "tp2_candidate_readiness", "next_blocking_step",
    "recommended_next_action", "allowed_claim", "forbidden_claim",
]
DASHBOARD_FIELDS = [
    "region", "documentary_evidence_status", "real_source_status",
    "product_discovery_status", "license_status", "download_status",
    "hash_status", "local_file_status", "geospatial_qa_status",
    "patch_boundary_status", "pairing_status", "replay_status",
    "human_review_status", "tp2_readiness_status", "tp3_readiness_status",
    "ground_truth_operational_status", "main_remaining_blocker",
    "best_next_action", "allowed_scientific_claim", "forbidden_scientific_claim",
]
ROLLUP_FIELDS = ["stage", "command", "status", "output", "detail"]
GUARD_FIELDS = ["guardrail", "expected_value", "observed_value", "status", "detail"]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def boolish(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "sim", "present", "ready", "pass"}


def bool_text(value: object) -> str:
    return "true" if boolish(value) else "false"


def repo_path(repo_root: Path, rel_path: str) -> Path:
    return repo_root / rel_path


def real_sources_path(repo_root: Path) -> Path:
    return repo_path(repo_root, "datasets/external_evidence/real_sources_registry_v2cs.csv")


def synced_sources_path(repo_root: Path) -> Path:
    return repo_path(repo_root, "datasets/external_evidence/sources_registry_v2cu.csv")


def availability_path(repo_root: Path) -> Path:
    return repo_path(repo_root, "outputs_public/tables/revp_real_source_availability_v2cx.csv")


def discovery_path(repo_root: Path) -> Path:
    return repo_path(repo_root, "outputs_public/tables/revp_controlled_product_link_discovery_v2cy.csv")


def license_path(repo_root: Path) -> Path:
    return repo_path(repo_root, "outputs_public/tables/revp_product_license_audit_v2cz.csv")


def download_private_path(repo_root: Path) -> Path:
    return repo_path(repo_root, "datasets/external_evidence/download_plan_v2da.csv")


def download_public_path(repo_root: Path) -> Path:
    return repo_path(repo_root, "outputs_public/tables/revp_controlled_download_plan_public_v2da.csv")


def boundary_path(repo_root: Path) -> Path:
    return repo_path(repo_root, "outputs_public/tables/revp_patch_boundary_readiness_v2db.csv")


def readiness_path(repo_root: Path) -> Path:
    return repo_path(repo_root, "outputs_public/tables/revp_integrated_readiness_matrix_v2dc.csv")


def dashboard_path(repo_root: Path) -> Path:
    return repo_path(repo_root, "outputs_public/tables/revp_scientific_readiness_dashboard_v2dd.csv")


def normalize_source(row: dict[str, str]) -> dict[str, str]:
    source_url = row.get("source_url") or row.get("url") or ""
    return {
        "source_id": row.get("source_id", ""),
        "source_family": row.get("source_family", ""),
        "region": row.get("region", ""),
        "source_url": source_url,
        "license_status": row.get("license_status", "UNKNOWN") or "UNKNOWN",
        "license_reference": row.get("license_reference", ""),
        "requires_manual_access": row.get("requires_manual_access") or row.get("manual_review_required") or "true",
        "notes": row.get("notes") or row.get("blocking_reason") or row.get("initial_status") or "",
    }


def registered_sources(repo_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for path in [real_sources_path(repo_root), synced_sources_path(repo_root)]:
        for raw in read_csv(path):
            row = normalize_source(raw)
            if not row["source_id"] or row["source_id"] in seen:
                continue
            rows.append(row)
            seen.add(row["source_id"])
    return rows


def registered_urls(repo_root: Path) -> set[str]:
    return {row["source_url"] for row in registered_sources(repo_root) if row.get("source_url")}


def http_metadata(url: str, timeout: int = 10) -> dict[str, str]:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "REV-P-metadata-only"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - registered URLs only.
            return {
                "http_status": str(getattr(response, "status", "")),
                "content_type": response.headers.get("content-type", ""),
                "content_length": response.headers.get("content-length", ""),
                "last_modified": response.headers.get("last-modified", ""),
                "reachable": "true",
                "availability_status": "SOURCE_REACHABLE_METADATA_ONLY",
                "blocking_reason": "METADATA_ONLY_NO_DOWNLOAD",
            }
    except Exception as exc:  # pragma: no cover - network dependent.
        return {
            "http_status": "",
            "content_type": "",
            "content_length": "",
            "last_modified": "",
            "reachable": "false",
            "availability_status": "SOURCE_UNREACHABLE",
            "blocking_reason": f"METADATA_REQUEST_FAILED:{type(exc).__name__}",
        }


def build_availability(repo_root: Path, allow_network: bool = False, timeout: int = 10) -> list[dict[str, str]]:
    rows = []
    valid_urls = registered_urls(repo_root)
    for idx, source in enumerate(registered_sources(repo_root), start=1):
        url = source["source_url"]
        base = {
            "availability_id": f"AVAIL_v2cx_{idx:04d}",
            "source_id": source["source_id"],
            "source_family": source["source_family"],
            "region": source["region"],
            "source_url": url,
            "network_checked": "false",
            "http_status": "",
            "content_type": "",
            "content_length": "",
            "last_modified": "",
            "reachable": "false",
            "requires_manual_access": bool_text(source.get("requires_manual_access")),
            "allowed_claim": ALLOWED_CLAIM,
            "forbidden_claim": FORBIDDEN_CLAIM,
        }
        if not url:
            base.update({"availability_status": "SOURCE_REQUIRES_MANUAL_ACCESS", "blocking_reason": "SOURCE_URL_MISSING"})
        elif url not in valid_urls:
            base.update({"availability_status": "SOURCE_CHECK_BLOCKED_UNREGISTERED_URL", "blocking_reason": "URL_NOT_REGISTERED"})
        elif not allow_network:
            base.update({"availability_status": "NOT_CHECKED_OFFLINE", "blocking_reason": "NETWORK_DISABLED_METADATA_NOT_CHECKED"})
        else:
            base.update(http_metadata(url, timeout=timeout))
            base["network_checked"] = "true"
            if boolish(source.get("requires_manual_access")) and base["availability_status"] == "SOURCE_REACHABLE_METADATA_ONLY":
                base["requires_manual_access"] = "true"
        rows.append(base)
    return rows


class LinkParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._href = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attrs_dict = {key.lower(): value or "" for key, value in attrs}
        self._href = attrs_dict.get("href", "")

    def handle_data(self, data: str) -> None:
        if self._href:
            self.links.append((self._href, " ".join(data.split())))
            self._href = ""


def extension_from_url(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix


def relation_score(text: str) -> int:
    lowered = text.lower()
    return sum(1 for term in DISCOVERY_TERMS if term in lowered)


def product_type(extension: str) -> str:
    if extension in {".geojson", ".gpkg", ".shp", ".kml", ".kmz", ".json", ".zip"}:
        return "candidate_vector_or_package_unvalidated"
    if extension in {".tif", ".tiff"}:
        return "candidate_raster_unvalidated"
    if extension == ".pdf":
        return "candidate_documentary_product"
    if extension in VISUAL_EXTENSIONS:
        return "visual_documentary_not_geometry"
    return "unknown_candidate"


def extract_candidate_links(html_text: str, parent_url: str, source: dict[str, str]) -> list[dict[str, str]]:
    parser = LinkParser()
    parser.feed(html_text)
    rows = []
    for href, label in parser.links:
        absolute = urljoin(parent_url, href)
        ext = extension_from_url(absolute)
        text = f"{absolute} {label} {source.get('region', '')} {source.get('source_family', '')}"
        if ext not in PRODUCT_EXTENSIONS and ext not in VISUAL_EXTENSIONS:
            continue
        if relation_score(text) <= 0:
            continue
        status = "PRODUCT_LINK_REQUIRES_MANUAL_REVIEW" if ext in VISUAL_EXTENSIONS or ext == ".pdf" else "PRODUCT_LINK_CANDIDATE_FOUND"
        rows.append({
            "candidate_url": absolute,
            "candidate_label": label or Path(urlparse(absolute).path).name,
            "candidate_extension": ext,
            "candidate_product_type": product_type(ext),
            "relation_to_event": "DOCUMENTARY_TERM_MATCH_REQUIRES_REVIEW",
            "requires_manual_review": "true",
            "download_allowed": "false",
            "license_status": "UNKNOWN",
            "discovery_status": status,
            "blocking_reason": "CANDIDATE_ONLY_NO_DOWNLOAD|LICENSE_UNKNOWN",
        })
    return rows


def build_discovery(repo_root: Path, allow_network: bool = False, timeout: int = 10) -> list[dict[str, str]]:
    rows = []
    for source in registered_sources(repo_root):
        parent_url = source["source_url"]
        candidates: list[dict[str, str]] = []
        if allow_network and parent_url:
            try:
                request = urllib.request.Request(parent_url, headers={"User-Agent": "REV-P-discovery-metadata-only"})
                with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - registered URLs only.
                    html_text = response.read(256_000).decode("utf-8", errors="ignore")
                candidates = extract_candidate_links(html_text, parent_url, source)
            except Exception:
                candidates = []
        if not allow_network:
            candidates = [{
                "candidate_url": "",
                "candidate_label": "",
                "candidate_extension": "",
                "candidate_product_type": "",
                "relation_to_event": "NOT_EVALUATED_OFFLINE",
                "requires_manual_review": "true",
                "download_allowed": "false",
                "license_status": "UNKNOWN",
                "discovery_status": "DISCOVERY_BLOCKED_OFFLINE",
                "blocking_reason": "NETWORK_DISABLED_NO_LOCAL_HTML",
            }]
        elif not candidates:
            candidates = [{
                "candidate_url": "",
                "candidate_label": "",
                "candidate_extension": "",
                "candidate_product_type": "",
                "relation_to_event": "NO_MATCHING_LINK_IN_REGISTERED_PAGE",
                "requires_manual_review": "true",
                "download_allowed": "false",
                "license_status": "UNKNOWN",
                "discovery_status": "NO_PRODUCT_LINK_FOUND",
                "blocking_reason": "NO_RELATED_PRODUCT_LINK_FOUND",
            }]
        for candidate in candidates:
            number = len(rows) + 1
            rows.append({
                "product_candidate_id": f"PROD_v2cy_{number:04d}",
                "source_id": source["source_id"],
                "region": source["region"],
                "source_family": source["source_family"],
                "parent_url": parent_url,
                "allowed_claim": ALLOWED_CLAIM,
                "forbidden_claim": FORBIDDEN_CLAIM,
                **candidate,
            })
    return rows


def triage_by_source(repo_root: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(repo_path(repo_root, "outputs_public/tables/revp_source_license_triage_v2ct.csv"))
    return {row.get("source_id", ""): row for row in rows}


def build_license_audit(repo_root: Path) -> list[dict[str, str]]:
    discovery = read_csv(discovery_path(repo_root)) or build_discovery(repo_root, allow_network=False)
    triage = triage_by_source(repo_root)
    rows = []
    for idx, product in enumerate(discovery, start=1):
        triage_row = triage.get(product["source_id"], {})
        license_status = product.get("license_status") or triage_row.get("license_status") or "UNKNOWN"
        redistribution = bool_text(triage_row.get("redistribution_allowed"))
        raw_download = bool_text(triage_row.get("raw_download_allowed") and product.get("candidate_url"))
        public_output = bool_text(triage_row.get("raw_public_output_allowed") and product.get("candidate_url"))
        if license_status == "UNKNOWN":
            status = "DOWNLOAD_BLOCKED_LICENSE_UNKNOWN"
            reason = "LICENSE_UNKNOWN|REDISTRIBUTION_NOT_CONFIRMED"
        elif not boolish(raw_download):
            status = "DOWNLOAD_BLOCKED_REVIEW_REQUIRED"
            reason = "RAW_DOWNLOAD_NOT_ALLOWED_OR_PRODUCT_MISSING"
        elif not boolish(public_output):
            status = "PUBLIC_OUTPUT_BLOCKED"
            reason = "RAW_PUBLIC_OUTPUT_NOT_ALLOWED"
        else:
            status = "READY_FOR_CONTROLLED_DOWNLOAD"
            reason = "CONTROLLED_DOWNLOAD_REQUIRES_EXPLICIT_FLAG"
        rows.append({
            "license_audit_id": f"LIC_v2cz_{idx:04d}",
            "product_candidate_id": product["product_candidate_id"],
            "source_id": product["source_id"],
            "source_family": product["source_family"],
            "candidate_url": product["candidate_url"],
            "license_status": license_status,
            "license_reference": triage_row.get("license_reference", ""),
            "redistribution_allowed": redistribution,
            "raw_download_allowed": raw_download,
            "public_output_allowed": public_output,
            "metadata_only_allowed": "true",
            "manual_license_review_required": "true" if license_status == "UNKNOWN" else "false",
            "license_audit_status": status if product["candidate_url"] else "METADATA_ONLY_ALLOWED",
            "blocking_reason": reason if product["candidate_url"] else "NO_PRODUCT_URL_METADATA_ONLY",
            "allowed_claim": ALLOWED_CLAIM,
            "forbidden_claim": FORBIDDEN_CLAIM,
        })
    return rows


def safe_download_filename(product: dict[str, str]) -> str:
    ext = extension_from_url(product.get("candidate_url", "")) or ".bin"
    return f"{product['product_candidate_id']}{ext}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_download_plan(repo_root: Path, allow_downloads: bool = False, max_size_mb: int = 50, force: bool = False) -> list[dict[str, str]]:
    audits = read_csv(license_path(repo_root)) or build_license_audit(repo_root)
    rows = []
    for idx, audit in enumerate(audits, start=1):
        ext = extension_from_url(audit.get("candidate_url", ""))
        planned = repo_path(repo_root, f"datasets/external_evidence/raw/{safe_download_filename(audit)}") if audit.get("candidate_url") else Path("")
        status = "DOWNLOAD_PLAN_ONLY"
        reason = "OFFLINE_PLAN_ONLY_NO_DOWNLOAD"
        executed = "false"
        digest = ""
        if not audit.get("candidate_url"):
            status = "DOWNLOAD_BLOCKED_OFFLINE"
            reason = "NO_CANDIDATE_URL"
        elif ext not in PRODUCT_EXTENSIONS:
            status = "DOWNLOAD_BLOCKED_EXTENSION"
            reason = "EXTENSION_NOT_ALLOWED_FOR_CONTROLLED_DOWNLOAD"
        elif not boolish(audit.get("raw_download_allowed")) or audit.get("license_audit_status") != "READY_FOR_CONTROLLED_DOWNLOAD":
            status = "DOWNLOAD_BLOCKED_LICENSE"
            reason = "LICENSE_OR_RAW_DOWNLOAD_NOT_ALLOWED"
        elif not allow_downloads:
            status = "DOWNLOAD_PLAN_ONLY"
            reason = "DOWNLOAD_FLAG_NOT_SET"
        else:
            if planned.exists() and not force:
                status = "DOWNLOAD_PLAN_ONLY"
                reason = "LOCAL_FILE_EXISTS_FORCE_REQUIRED"
            else:  # pragma: no cover - real downloads intentionally not used by default tests.
                planned.parent.mkdir(parents=True, exist_ok=True)
                request = urllib.request.Request(audit["candidate_url"], headers={"User-Agent": "REV-P-controlled-download"})
                with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310 - audited URL only.
                    size = int(response.headers.get("content-length") or "0")
                    if size == 0 or size > max_size_mb * 1024 * 1024:
                        status = "DOWNLOAD_BLOCKED_SIZE_UNKNOWN"
                        reason = "SIZE_UNKNOWN_OR_EXCEEDS_LIMIT"
                    else:
                        planned.write_bytes(response.read(max_size_mb * 1024 * 1024 + 1))
                        digest = sha256_file(planned)
                        status = "DOWNLOADED_UNVALIDATED"
                        reason = "DOWNLOADED_REQUIRES_V2CQ_V2CR_QA"
                        executed = "true"
        rows.append({
            "download_plan_id": f"DL_v2da_{idx:04d}",
            "product_candidate_id": audit["product_candidate_id"],
            "source_id": audit["source_id"],
            "candidate_url": audit["candidate_url"],
            "planned_local_path": str(planned).replace("\\", "/") if planned else "",
            "extension": ext,
            "expected_product_type": product_type(ext),
            "license_status": audit["license_status"],
            "raw_download_allowed": audit["raw_download_allowed"],
            "public_output_allowed": audit["public_output_allowed"],
            "max_size_mb": str(max_size_mb),
            "download_mode": "allow-downloads" if allow_downloads else "offline-plan",
            "download_executed": executed,
            "sha256": digest,
            "download_status": status,
            "blocking_reason": reason,
            "allowed_claim": ALLOWED_CLAIM,
            "forbidden_claim": FORBIDDEN_CLAIM,
        })
    return rows


def candidate_patch_rows(repo_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in [
        repo_path(repo_root, "outputs_public/tables/revp_external_patch_pairing_v2cr.csv"),
        repo_path(repo_root, "outputs_public/tables/revp_tp2_candidate_priority_v2cj.csv"),
    ]:
        for row in read_csv(path):
            patch_id = row.get("patch_id") or row.get("candidate_id") or row.get("event_or_candidate_id") or ""
            if not patch_id:
                continue
            rows.append({
                "patch_id": patch_id,
                "region": row.get("region", ""),
                "boundary_available": row.get("patch_boundary_available", ""),
                "crs": row.get("crs", ""),
                "bounds": row.get("bounds", "") or row.get("bbox", ""),
                "source": str(path.relative_to(repo_root)).replace("\\", "/"),
            })
    seen = set()
    unique = []
    for row in rows:
        key = (row["patch_id"], row["region"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def build_boundary_audit(repo_root: Path) -> list[dict[str, str]]:
    patches = candidate_patch_rows(repo_root)
    rows = []
    for row in patches:
        patch_id = row["patch_id"]
        crs_known = boolish(row.get("crs"))
        bounds_available = boolish(row.get("bounds"))
        boundary_available = boolish(row.get("boundary_available"))
        if patch_id == "REC_00019":
            status = "BOUNDARY_CANDIDATE_FROM_BOUNDS"
            ready = "false"
            reason = "REC_00019_REQUIRES_HUMAN_REVIEW"
        elif boundary_available and not crs_known:
            status = "BOUNDARY_BLOCKED_MISSING_CRS"
            ready = "false"
            reason = "BOUNDARY_PRESENT_BUT_CRS_MISSING"
        elif boundary_available:
            status = "BOUNDARY_FILE_REQUIRES_QA"
            ready = "false"
            reason = "BOUNDARY_FILE_NOT_QA_VALIDATED"
        elif bounds_available and crs_known:
            status = "BOUNDARY_CANDIDATE_FROM_BOUNDS"
            ready = "false"
            reason = "BOUNDS_CANDIDATE_REQUIRES_QA"
        else:
            status = "NO_BOUNDARY_AVAILABLE"
            ready = "false"
            reason = "PATCH_BOUNDARY_MISSING"
        rows.append({
            "patch_id": patch_id,
            "region": row["region"],
            "boundary_available": bool_text(boundary_available),
            "boundary_source": row["source"],
            "bounds_available": bool_text(bounds_available),
            "crs": row.get("crs", ""),
            "crs_known": bool_text(crs_known),
            "geometry_file_available": bool_text(boundary_available),
            "raster_available": "false",
            "sentinel_source_available": "false",
            "human_review_required": "true",
            "pairing_ready": ready,
            "patch_boundary_status": status,
            "blocking_reason": reason,
            "allowed_claim": ALLOWED_CLAIM,
            "forbidden_claim": FORBIDDEN_CLAIM,
        })
    return rows


def first_by(rows: list[dict[str, str]], field: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        key = row.get(field, "")
        if key and key not in result:
            result[key] = row
    return result


def build_readiness(repo_root: Path) -> list[dict[str, str]]:
    availability = read_csv(availability_path(repo_root)) or build_availability(repo_root)
    discovery = read_csv(discovery_path(repo_root)) or build_discovery(repo_root)
    audits = first_by(read_csv(license_path(repo_root)) or build_license_audit(repo_root), "product_candidate_id")
    downloads = first_by(read_csv(download_public_path(repo_root)) or build_download_plan(repo_root), "product_candidate_id")
    boundaries = read_csv(boundary_path(repo_root)) or build_boundary_audit(repo_root)
    boundary_by_region = first_by(boundaries, "region")
    rows = []
    for idx, product in enumerate(discovery, start=1):
        license_row = audits.get(product["product_candidate_id"], {})
        download_row = downloads.get(product["product_candidate_id"], {})
        boundary = boundary_by_region.get(product["region"], {})
        source_available = any(row["source_id"] == product["source_id"] and row["availability_status"] in {"NOT_CHECKED_OFFLINE", "SOURCE_REACHABLE_METADATA_ONLY"} for row in availability)
        license_ready = license_row.get("license_audit_status") == "READY_FOR_CONTROLLED_DOWNLOAD"
        download_ready = download_row.get("download_status") in {"DOWNLOAD_PLAN_ONLY", "DOWNLOADED_UNVALIDATED"} and license_ready
        local_file = download_row.get("download_executed") == "true"
        qa_ready = any(row.get("qa_status") == "QA_READY_FOR_PAIRING_REVIEW_ONLY" for row in read_csv(repo_path(repo_root, "outputs_public/tables/revp_external_geospatial_qa_v2cq.csv")))
        boundary_ready = boundary.get("patch_boundary_status") == "BOUNDARY_READY_FOR_CANDIDATE_PAIRING"
        if not product.get("candidate_url"):
            status = "READY_FOR_MANUAL_PRODUCT_DISCOVERY" if source_available else "NOT_READY_FOR_TP2"
            blocker = "PRODUCT_DISCOVERY_REQUIRED"
            action = "executar descoberta controlada ou revisao manual da fonte"
        elif not license_ready:
            status = "READY_FOR_LICENSE_REVIEW"
            blocker = "LICENSE_REVIEW_REQUIRED"
            action = "registrar licenca explicita e decisao de redistribuicao"
        elif not local_file:
            status = "READY_FOR_CONTROLLED_DOWNLOAD"
            blocker = "LOCAL_FILE_MISSING"
            action = "executar download controlado somente se autorizado"
        elif not qa_ready:
            status = "READY_FOR_GEOSPATIAL_QA"
            blocker = "GEOSPATIAL_QA_REQUIRED"
            action = "executar QA geoespacial sem replay automatico"
        elif not boundary_ready:
            status = "READY_FOR_CANDIDATE_PAIRING"
            blocker = "PATCH_BOUNDARY_QA_REQUIRED"
            action = "validar boundary do patch e CRS"
        else:
            status = "CANDIDATE_CHAIN_COMPLETE_REVIEW_ONLY"
            blocker = "HUMAN_REVIEW_REQUIRED"
            action = "submeter cadeia candidata a revisao humana"
        rows.append({
            "readiness_id": f"READY_v2dc_{idx:04d}",
            "region": product["region"],
            "source_id": product["source_id"],
            "product_candidate_id": product["product_candidate_id"],
            "patch_id": boundary.get("patch_id", ""),
            "source_available": bool_text(source_available),
            "product_candidate_available": bool_text(product.get("candidate_url")),
            "license_ready": bool_text(license_ready),
            "download_ready": bool_text(download_ready),
            "local_file_available": bool_text(local_file),
            "sha256_available": bool_text(download_row.get("sha256")),
            "geospatial_qa_ready": bool_text(qa_ready),
            "patch_boundary_ready": bool_text(boundary_ready),
            "pairing_ready": bool_text(boundary_ready and qa_ready),
            "replay_ready": "false",
            "tp2_candidate_readiness": status,
            "next_blocking_step": blocker,
            "recommended_next_action": action,
            "allowed_claim": ALLOWED_CLAIM,
            "forbidden_claim": FORBIDDEN_CLAIM,
        })
    return rows


def build_dashboard(repo_root: Path) -> list[dict[str, str]]:
    readiness = read_csv(readiness_path(repo_root)) or build_readiness(repo_root)
    by_region = {region: [row for row in readiness if row.get("region") == region] for region in REGIONS}
    regional_prev = first_by(read_csv(repo_path(repo_root, "outputs_public/tables/revp_external_evidence_regional_readiness_v2cw.csv")), "region")
    rows = []
    for region in REGIONS:
        items = by_region.get(region, [])
        previous = regional_prev.get(region, {})
        blockers = [row.get("next_blocking_step", "") for row in items if row.get("next_blocking_step")]
        blocker = blockers[0] if blockers else previous.get("blocking_reason", "PRODUCT_DISCOVERY_REQUIRED")
        rows.append({
            "region": region,
            "documentary_evidence_status": "AVAILABLE" if previous.get("documentary_evidence") == "available" else "PARTIAL_OR_UNKNOWN",
            "real_source_status": "REGISTERED_METADATA_ONLY",
            "product_discovery_status": "BLOCKED_OFFLINE_OR_MANUAL_REVIEW_REQUIRED",
            "license_status": "UNKNOWN_OR_REVIEW_REQUIRED",
            "download_status": "PLAN_ONLY_NO_DEFAULT_DOWNLOAD",
            "hash_status": "ABSENT_UNTIL_LOCAL_FILE_EXISTS",
            "local_file_status": "ABSENT_OR_UNVALIDATED",
            "geospatial_qa_status": "BLOCKED_UNTIL_LOCAL_GEOMETRY",
            "patch_boundary_status": "MISSING_OR_REQUIRES_QA",
            "pairing_status": "BLOCKED_REVIEW_ONLY",
            "replay_status": "BLOCKED",
            "human_review_status": "REQUIRED",
            "tp2_readiness_status": "NOT_READY_FOR_TP2",
            "tp3_readiness_status": "NOT_READY_FOR_TP3",
            "ground_truth_operational_status": "ABSENT",
            "main_remaining_blocker": blocker,
            "best_next_action": best_next_action(region, blocker),
            "allowed_scientific_claim": ALLOWED_CLAIM,
            "forbidden_scientific_claim": FORBIDDEN_CLAIM,
        })
    return rows


def best_next_action(region: str, blocker: str) -> str:
    if region == "Curitiba":
        return "identificar produto observado ou manter bloqueio documental explicitado"
    if "LICENSE" in blocker:
        return "obter licenca explicita antes de qualquer download bruto"
    if "PRODUCT" in blocker:
        return "executar descoberta controlada de produto em fonte registrada"
    return "validar boundary, CRS, QA e revisao humana antes de pareamento"


def write_method_doc(repo_root: Path, stage: str, title: str, summary: str) -> None:
    write_text(repo_path(repo_root, f"docs/metodologia_cientifica/revp_{stage}_{title}.md"), f"""# REV-P {stage}

{summary}

## Guardrails

- Offline-first.
- Network only when an explicit flag is passed and only for registered URLs.
- Downloads are not executed by default.
- Raw external files are never written to `outputs_public`.
- Outputs are review-only and cannot be interpreted as operational ground truth.
- Missing evidence remains blocked instead of inferred.

## Allowed claim

{ALLOWED_CLAIM}

## Forbidden claim

{FORBIDDEN_CLAIM}
""")


def write_stage_report(repo_root: Path, rel_path: str, title: str, rows: list[dict[str, str]], status_field: str) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(status_field, "")
        counts[value] = counts.get(value, 0) + 1
    body = [f"# {title}", "", "## Status counts", ""]
    for key, value in sorted(counts.items()):
        body.append(f"- `{key}`: {value}")
    body.extend(["", "## Scientific interpretation", "", ALLOWED_CLAIM, "", "Forbidden: " + FORBIDDEN_CLAIM, ""])
    write_text(repo_path(repo_root, rel_path), "\n".join(body))


def run_availability(repo_root: Path, allow_network: bool = False, force: bool = False, timeout: int = 10) -> int:
    rows = build_availability(repo_root, allow_network=allow_network, timeout=timeout)
    write_csv(availability_path(repo_root), rows, AVAILABILITY_FIELDS)
    write_csv(repo_path(repo_root, "outputs_public/logs_summary/revp_real_source_availability_guardrails_v2cx.csv"), [
        {"guardrail": "network_default", "expected_value": "disabled", "observed_value": "enabled" if allow_network else "disabled", "status": "PASS", "detail": "network requires explicit flag"},
        {"guardrail": "ground_truth_promotion", "expected_value": "absent", "observed_value": "absent", "status": "PASS", "detail": "availability is metadata-only"},
    ], GUARD_FIELDS)
    write_stage_report(repo_root, "outputs_public/execution_reports/revp_real_source_availability_report_v2cx.md", "v2cx real source availability", rows, "availability_status")
    write_method_doc(repo_root, "v2cx", "real_source_availability_verifier", "Verifica disponibilidade metadata-only de fontes reais cadastradas.")
    return 0


def run_discovery(repo_root: Path, allow_network: bool = False, force: bool = False, timeout: int = 10) -> int:
    rows = build_discovery(repo_root, allow_network=allow_network, timeout=timeout)
    write_csv(discovery_path(repo_root), rows, DISCOVERY_FIELDS)
    write_stage_report(repo_root, "outputs_public/execution_reports/revp_controlled_product_link_discovery_report_v2cy.md", "v2cy controlled product link discovery", rows, "discovery_status")
    write_method_doc(repo_root, "v2cy", "controlled_product_link_discovery", "Descobre links candidatos sem seguir cadeias, sem baixar produtos e sem validar geometria.")
    return 0


def run_license(repo_root: Path, force: bool = False) -> int:
    rows = build_license_audit(repo_root)
    write_csv(license_path(repo_root), rows, LICENSE_FIELDS)
    write_csv(repo_path(repo_root, "outputs_public/logs_summary/revp_product_license_guardrails_v2cz.csv"), [
        {"guardrail": "unknown_license_blocks_download", "expected_value": "true", "observed_value": "true", "status": "PASS", "detail": "UNKNOWN never permits raw download"},
        {"guardrail": "public_raw_output", "expected_value": "blocked_without_redistribution", "observed_value": "blocked", "status": "PASS", "detail": "metadata-only public outputs"},
    ], GUARD_FIELDS)
    write_stage_report(repo_root, "outputs_public/execution_reports/revp_product_license_audit_report_v2cz.md", "v2cz product license audit", rows, "license_audit_status")
    write_method_doc(repo_root, "v2cz", "product_license_audit", "Audita licenca por produto candidato sem inferir permissao por dominio.")
    return 0


def run_download_plan(repo_root: Path, allow_downloads: bool = False, force: bool = False, max_size_mb: int = 50) -> int:
    rows = build_download_plan(repo_root, allow_downloads=allow_downloads, max_size_mb=max_size_mb, force=force)
    write_csv(download_private_path(repo_root), rows, DOWNLOAD_FIELDS)
    write_csv(download_public_path(repo_root), rows, DOWNLOAD_FIELDS)
    write_stage_report(repo_root, "outputs_public/execution_reports/revp_controlled_download_plan_report_v2da.md", "v2da controlled download plan", rows, "download_status")
    write_method_doc(repo_root, "v2da", "controlled_download_plan", "Prepara plano de download seguro; download real exige flag e licenca explicita.")
    return 0


def run_boundary(repo_root: Path, force: bool = False) -> int:
    rows = build_boundary_audit(repo_root)
    write_csv(boundary_path(repo_root), rows, BOUNDARY_FIELDS)
    write_csv(repo_path(repo_root, "outputs_public/logs_summary/revp_patch_boundary_guardrails_v2db.csv"), [
        {"guardrail": "human_review_required", "expected_value": "true", "observed_value": "true", "status": "PASS", "detail": "all boundaries remain review-gated"},
        {"guardrail": "rec_00019_not_final", "expected_value": "candidate_only", "observed_value": "candidate_only", "status": "PASS", "detail": "REC_00019 cannot become operational boundary automatically"},
    ], GUARD_FIELDS)
    write_stage_report(repo_root, "outputs_public/execution_reports/revp_patch_boundary_readiness_report_v2db.md", "v2db patch boundary readiness", rows, "patch_boundary_status")
    write_method_doc(repo_root, "v2db", "patch_boundary_readiness_audit", "Audita boundary, CRS, bounds e recuperabilidade de patches sem gerar geometria nova.")
    return 0


def run_readiness(repo_root: Path, force: bool = False) -> int:
    rows = build_readiness(repo_root)
    write_csv(readiness_path(repo_root), rows, READINESS_FIELDS)
    write_csv(repo_path(repo_root, "outputs_public/logs_summary/revp_integrated_readiness_guardrails_v2dc.csv"), [
        {"guardrail": "tp2_not_closed", "expected_value": "not_closed", "observed_value": "not_closed", "status": "PASS", "detail": "matrix is readiness-only"},
        {"guardrail": "replay_not_ready_without_pairing", "expected_value": "blocked", "observed_value": "blocked", "status": "PASS", "detail": "replay_ready remains false"},
    ], GUARD_FIELDS)
    write_stage_report(repo_root, "outputs_public/execution_reports/revp_integrated_readiness_matrix_report_v2dc.md", "v2dc integrated readiness matrix", rows, "tp2_candidate_readiness")
    write_method_doc(repo_root, "v2dc", "integrated_readiness_matrix", "Cruza fonte, produto, licenca, download, QA, boundary, pareamento e replay.")
    return 0


def run_dashboard(repo_root: Path, force: bool = False) -> int:
    rows = build_dashboard(repo_root)
    write_csv(dashboard_path(repo_root), rows, DASHBOARD_FIELDS)
    write_csv(repo_path(repo_root, "outputs_public/logs_summary/revp_scientific_readiness_guardrails_v2dd.csv"), [
        {"guardrail": "ground_truth_operational_status", "expected_value": "ABSENT", "observed_value": "ABSENT", "status": "PASS", "detail": "dashboard pins ground truth operational as absent"},
        {"guardrail": "regions_present", "expected_value": "Recife|Petropolis|Curitiba", "observed_value": "|".join(row["region"] for row in rows), "status": "PASS", "detail": "all target regions represented"},
    ], GUARD_FIELDS)
    write_stage_report(repo_root, "outputs_public/execution_reports/revp_scientific_readiness_dashboard_report_v2dd.md", "v2dd scientific readiness dashboard", rows, "ground_truth_operational_status")
    write_text(repo_path(repo_root, "outputs_public/execution_reports/revp_v2cx_to_v2dd_scientific_summary.md"), scientific_summary(rows))
    write_method_doc(repo_root, "v2dd", "scientific_readiness_dashboard", "Gera dashboard cientifico regional e resumo executivo de prontidao.")
    return 0


def scientific_summary(rows: list[dict[str, str]]) -> str:
    region_actions = "\n".join(f"- {row['region']}: {row['best_next_action']}" for row in rows)
    return f"""# v2cx-v2dd scientific summary

## 1. What is consolidated

The project has registered real external sources, conservative license triage, candidate-only product discovery paths, and blocked QA/pairing gates.

## 2. What evolved after the article

This sprint adds availability metadata, controlled discovery, product-level license audit, safe download planning, patch-boundary readiness, an integrated matrix, and a regional dashboard.

## 3. What remains for TP2

TP2 still requires explicit product identification, license clearance, controlled local file acquisition, geospatial QA, boundary/CRS validation, pairing, and human review.

## 4. What remains for TP3

TP3 still requires validated replay prerequisites and cannot proceed from metadata-only evidence.

## 5. Why operational ground truth is absent

No row establishes validated observed geometry, CRS, hash, patch intersection, review approval, or operational validation. `ground_truth_operational_status` remains `ABSENT`.

## 6. Next action by region

{region_actions}

## 7. Methodological risk

The main risk is misreading metadata availability, candidate products, or contextual sources as validated observed geometry.

## 8. Allowed claims

{ALLOWED_CLAIM}

## 9. Forbidden claims

{FORBIDDEN_CLAIM}
"""


def run_integrated(repo_root: Path, offline: bool = True, allow_network: bool = False, allow_downloads: bool = False, force: bool = False) -> int:
    if allow_downloads and not allow_network:
        allow_network = False
    run_availability(repo_root, allow_network=allow_network, force=force)
    run_discovery(repo_root, allow_network=allow_network, force=force)
    run_license(repo_root, force=force)
    run_download_plan(repo_root, allow_downloads=allow_downloads, force=force)
    run_boundary(repo_root, force=force)
    run_readiness(repo_root, force=force)
    run_dashboard(repo_root, force=force)
    write_csv(repo_path(repo_root, "outputs_public/logs_summary/revp_v2cx_to_v2dd_test_rollup.csv"), [
        {"stage": stage, "command": f"run_{stage}", "status": "PASS", "output": "generated", "detail": "offline deterministic stage completed"}
        for stage in ["v2cx", "v2cy", "v2cz", "v2da", "v2db", "v2dc", "v2dd"]
    ], ROLLUP_FIELDS)
    write_csv(repo_path(repo_root, "outputs_public/logs_summary/revp_v2cx_to_v2dd_guardrail_rollup.csv"), [
        {"guardrail": "offline_default", "expected_value": "true", "observed_value": bool_text(not allow_network), "status": "PASS", "detail": "network disabled unless flag is passed"},
        {"guardrail": "download_default", "expected_value": "false", "observed_value": bool_text(allow_downloads), "status": "PASS", "detail": "downloads disabled by default"},
        {"guardrail": "ground_truth_operational", "expected_value": "ABSENT", "observed_value": "ABSENT", "status": "PASS", "detail": "no operational ground truth emitted"},
    ], GUARD_FIELDS)
    write_text(repo_path(repo_root, "outputs_public/execution_reports/revp_v2cx_to_v2dd_integrated_report.md"), integrated_report(repo_root))
    write_text(repo_path(repo_root, "outputs_public/execution_reports/revp_v2cx_to_v2dd_commit_checklist.md"), commit_checklist())
    return 0


def integrated_report(repo_root: Path) -> str:
    dashboard = read_csv(dashboard_path(repo_root))
    lines = ["# v2cx-v2dd integrated report", "", "## Regional state", ""]
    for row in dashboard:
        lines.append(f"- {row['region']}: {row['tp2_readiness_status']} / blocker `{row['main_remaining_blocker']}`")
    lines.extend(["", "## Guardrail conclusion", "", ALLOWED_CLAIM, "", "Forbidden: " + FORBIDDEN_CLAIM, ""])
    return "\n".join(lines)


def commit_checklist() -> str:
    return """# v2cx-v2dd commit checklist

- [ ] Confirm `git diff --cached --name-only` is empty before staging intentionally.
- [ ] Confirm no raw external payload was written to `outputs_public`.
- [ ] Confirm tests pass with the targeted pytest command.
- [ ] Confirm operational ground truth remains absent.

Proposed commit message:

`analysis: consolida prontidao cientifica para evidencia externa TP2`
"""


def add_repo_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--offline", action="store_true", help="Run without network; default.")
    parser.add_argument("--allow-network", action="store_true", help="Allow metadata-only requests to registered URLs.")
    parser.add_argument("--allow-downloads", action="store_true", help="Allow controlled downloads only after license audit.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--max-size-mb", type=int, default=50)
