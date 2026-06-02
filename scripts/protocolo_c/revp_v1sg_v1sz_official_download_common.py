"""Shared helpers for REV-P Protocol C v1sg-v1sz official data acquisition.

Controlled downloads from official public sources. Never creates labels,
targets, operational ground truth or formal negatives. All outputs review-only.
Downloads respect rate limits, timeouts, max size and domain allowlists.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

from revp_v1qu_v1qz_ground_reference_common import (  # noqa: F401
    DATASETS, DOCS, SCHEMAS,
    assert_clean_rows, safe_relpath,
    write_csv_with_header as _write_csv, write_doc, write_schema_safe,
)

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Env var helpers
# ---------------------------------------------------------------------------

def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()

def _env_bool(name: str, default: bool = False) -> bool:
    return _env(name, str(default)).lower() in ("true", "1", "yes")

def _env_int(name: str, default: int) -> int:
    try: return int(_env(name, str(default)))
    except ValueError: return default

def _env_float(name: str, default: float) -> float:
    try: return float(_env(name, str(default)))
    except ValueError: return default

def _p(env: str, default: Path) -> Path:
    return Path(os.environ[env]) if env in os.environ else default

def force_queue_only() -> bool:
    """Queue-only is the default; only an explicit enable flag turns it off."""
    default = not _env_bool("REVP_ENABLE_OFFICIAL_DOWNLOADS")
    return _env_bool("REVP_DOWNLOAD_FORCE_QUEUE_ONLY", default)

def downloads_enabled() -> bool:
    return _env_bool("REVP_ENABLE_OFFICIAL_DOWNLOADS") and not force_queue_only()

def network_allowed() -> bool:
    """No network access at all unless downloads are explicitly enabled."""
    return downloads_enabled()

def download_mode() -> str:
    return _env("REVP_DOWNLOAD_MODE", "minimal")

def max_gb() -> float:
    return _env_float("REVP_DOWNLOAD_MAX_GB", 2.0)

def max_files() -> int:
    return _env_int("REVP_DOWNLOAD_MAX_FILES", 20)

def max_bytes_per_file() -> int:
    return _env_int("REVP_DOWNLOAD_MAX_BYTES_PER_FILE", 250 * 1024 * 1024)

def connect_timeout_sec() -> int:
    return _env_int("REVP_DOWNLOAD_CONNECT_TIMEOUT_SECONDS", 15)

def read_timeout_sec() -> int:
    return _env_int("REVP_DOWNLOAD_READ_TIMEOUT_SECONDS", 60)

def timeout_sec() -> int:
    return _env_int("REVP_DOWNLOAD_TIMEOUT_SECONDS", read_timeout_sec())

def retries() -> int:
    return max(0, _env_int("REVP_DOWNLOAD_RETRIES", 2))

def rate_limit_sec() -> float:
    return _env_float("REVP_DOWNLOAD_RATE_LIMIT_SECONDS", 2.0)

def raw_root() -> Path:
    return Path(_env("REVP_EXTERNAL_RAW_ROOT", str(ROOT / "data" / "external_raw")))

def cache_root() -> Path:
    return Path(_env("REVP_EXTERNAL_CACHE_ROOT", str(ROOT / "data" / "external_cache")))

def force_redownload() -> bool:
    return _env_bool("REVP_FORCE_REDOWNLOAD")

# ---------------------------------------------------------------------------
# Path/guardrail helpers
# ---------------------------------------------------------------------------

ABS_PATH_RE = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]")
_FORBIDDEN_LITERAL = "local" + "_runs"

GUARDRAIL_FIELDS = [
    "review_only", "can_create_operational_label", "can_train_model",
    "target_created", "ground_truth_operational", "formal_negative",
    "dino_validates_event", "absence_as_negative",
]

FORBIDDEN_TRUE = [
    "can_create_operational_label", "can_train_model", "target_created",
    "ground_truth_operational", "formal_negative", "dino_validates_event",
    "absence_as_negative",
]

def guardrail_row() -> dict[str, str]:
    return {
        "review_only": "true", "can_create_operational_label": "false",
        "can_train_model": "false", "target_created": "false",
        "ground_truth_operational": "false", "formal_negative": "false",
        "dino_validates_event": "false", "absence_as_negative": "false",
    }

def detect_absolute_path(text: str) -> bool:
    return bool(ABS_PATH_RE.search(str(text)))

def detect_forbidden_literal_exposure(text: str) -> bool:
    return _FORBIDDEN_LITERAL in str(text).lower()

def mask_local_path(text: str) -> str:
    s = re.sub(r"[A-Za-z]:[\\/][^\s,;\"']*", "[PATH_REDACTED]", str(text))
    s = re.sub(r"(?i)" + _FORBIDDEN_LITERAL + r"[\\/][^\s,;\"']*", "[LOCAL_REDACTED]", s)
    return re.sub(r"(?i)gabriela", "[USER_REDACTED]", s)

def hash_short(value: str, n: int = 16) -> str:
    return hashlib.sha256(str(value).encode("utf-8", errors="ignore")).hexdigest()[:n]

def sha256_file_short(path: Path, n: int = 16) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:n]
    except Exception:
        return ""

def sha256_text_short(text: str, n: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:n]

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def forbidden_guardrail_scan(rows: list[dict[str, Any]], label: str) -> None:
    for i, row in enumerate(rows):
        for f in FORBIDDEN_TRUE:
            if str(row.get(f, "false")).strip().lower() == "true":
                raise ValueError(f"GUARDRAIL in {label} row {i}: {f}=true")
        for k, v in row.items():
            if detect_absolute_path(str(v)):
                raise ValueError(f"Abs path in {label} row {i} field {k!r}")
            if detect_forbidden_literal_exposure(str(v)):
                raise ValueError(f"Forbidden literal in {label} row {i} field {k!r}")

def read_csv_safe(path: Path | str) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists(): return []
    try:
        with p.open(encoding="utf-8-sig", errors="replace", newline="") as fh:
            return list(csv.DictReader(fh))
    except Exception: return []

def write_csv_with_header(path: Path | str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    _write_csv(Path(path), rows, fields)

def write_json_safe(path: Path | str, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")

def write_schema_for(path: Path, fields: list[str], prefix: str) -> None:
    write_schema_safe(path, fields, prefix)

# ---------------------------------------------------------------------------
# URL / domain / source helpers
# ---------------------------------------------------------------------------

ALLOWED_DOMAINS = frozenset([
    "portal.inmet.gov.br", "bdmep.inmet.gov.br", "tempo.inmet.gov.br",
    "alert-as.inmet.gov.br", "www.gov.br", "gov.br",
    "www.snirh.gov.br", "snirh.gov.br",
    "telemetriaws1.ana.gov.br", "www.ana.gov.br", "ana.gov.br",
    "cemaden.gov.br", "www.gov.br/cemaden",
    "sgb.gov.br", "www.sgb.gov.br", "cprm.gov.br", "www.cprm.gov.br",
    "ibge.gov.br", "servicodados.ibge.gov.br",
    "rigeo.sgb.gov.br",
])

def domain_from_url(url: str) -> str:
    return (urllib.parse.urlparse(str(url)).netloc or "").lower()

def is_allowed_domain(url: str) -> bool:
    d = domain_from_url(url)
    if d in ALLOWED_DOMAINS:
        return True
    return d.endswith(".gov.br")

def safe_url(url: str) -> str:
    """Normalize and validate URL. Return empty if disallowed domain."""
    s = str(url or "").strip()
    if not s: return ""
    if not is_allowed_domain(s): return ""
    return s

def normalize_url(url: str) -> str:
    s = str(url or "").strip()
    s = re.sub(r"[?&](utm_[^=&]+|fbclid|gclid)=[^&]*", "", s)
    return s.rstrip("/?&#")

def infer_region_from_text(text: str) -> str:
    lo = str(text or "").lower()
    if any(k in lo for k in ("recife", "pernambuco")): return "RECIFE"
    if any(k in lo for k in ("petropolis", "petrópolis")): return "PET"
    if any(k in lo for k in ("curitiba", "parana", "paraná")): return "CURITIBA"
    return "UNKNOWN"

def infer_source_from_url(url: str) -> str:
    lo = str(url or "").lower()
    if "inmet" in lo: return "INMET"
    if "ana.gov" in lo or "hidroweb" in lo or "snirh" in lo: return "ANA"
    if "cemaden" in lo: return "CEMADEN"
    if "sgb" in lo or "cprm" in lo: return "SGB_CPRM"
    if "ibge" in lo: return "IBGE"
    return "UNKNOWN"

def infer_year_from_url(url: str) -> str:
    m = re.search(r"(20\d{2})", str(url or ""))
    return m.group(1) if m else ""

def infer_hazard_type(text: str) -> str:
    lo = str(text or "").lower()
    if any(k in lo for k in ("chuva", "precipit", "pluvio", "rain", "flood", "inundacao", "inundação")):
        return "FLOOD"
    if any(k in lo for k in ("desliz", "landslide", "massa", "escorreg")):
        return "LANDSLIDE"
    return "HYDROMETEOROLOGICAL"

def classify_source_family(name: str) -> str:
    lo = str(name or "").lower()
    if any(k in lo for k in ("inmet", "bdmep", "cemaden", "ana", "hidroweb", "pluvio", "meteorol")):
        return "OFFICIAL_HYDROMETEOROLOGICAL"
    if any(k in lo for k in ("sgb", "cprm", "geolog")):
        return "OFFICIAL_GEOLOGICAL"
    if any(k in lo for k in ("defesa civil",)):
        return "OFFICIAL_CIVIL_DEFENSE"
    if any(k in lo for k in ("diario oficial", "diário oficial", "decreto")):
        return "OFFICIAL_GOVERNMENT_PUBLICATION"
    if any(k in lo for k in ("ibge", "mapbiomas")):
        return "SCIENTIFIC_DATASET"
    return "UNKNOWN_SOURCE"

def classify_document_type(url: str, content_type: str = "") -> str:
    lo = (str(url) + " " + str(content_type)).lower()
    if ".zip" in lo: return "ZIP"
    if ".csv" in lo: return "CSV"
    if ".pdf" in lo: return "PDF"
    if ".json" in lo or "geojson" in lo: return "JSON"
    if ".xlsx" in lo or "excel" in lo: return "XLSX"
    if "html" in lo: return "HTML"
    return "UNKNOWN"

def build_download_id(source: str, year: str, region: str, idx: int) -> str:
    return f"DL_{source}_{year}_{region}_{idx:04d}"

# ---------------------------------------------------------------------------
# HTTP helpers (urllib-based, no external deps required)
# ---------------------------------------------------------------------------

_USER_AGENT = "REV-P-official-data-acquisition/1.0 (academic-research; rate-limited)"
_LAST_REQUEST_TIME: dict[str, float] = {}

# Status sentinels for non-HTTP outcomes (kept out of the 1xx-5xx range).
_ST_NETWORK_DISABLED = -3
_ST_REDIRECT_BLOCKED = -2
_ST_DOMAIN_BLOCKED = -1


class _AllowlistRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse redirects that leave the official-domain allowlist."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not is_allowed_domain(newurl):
            raise urllib.error.HTTPError(newurl, code, "redirect_blocked", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _build_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(_AllowlistRedirectHandler())


def rate_limited_get(url: str, timeout: int | None = None, max_bytes: int = 0) -> tuple[int, bytes, str]:
    """Rate-limited GET. Returns (status, body, content_type).

    Fail-closed: never touches the network unless downloads are enabled, and
    only to allowlisted domains. Redirects off the allowlist are blocked.
    Negative status sentinels signal disabled/blocked outcomes.
    """
    if not network_allowed():
        return (_ST_NETWORK_DISABLED, b"", "")
    if not is_allowed_domain(url):
        return (_ST_DOMAIN_BLOCKED, b"", "")

    timeout = timeout or read_timeout_sec()
    domain = domain_from_url(url)
    opener = _build_opener()
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})

    last_status = 0
    for _ in range(retries() + 1):
        now = time.time()
        wait = rate_limit_sec() - (now - _LAST_REQUEST_TIME.get(domain, 0))
        if wait > 0:
            time.sleep(wait)
        _LAST_REQUEST_TIME[domain] = time.time()
        try:
            with opener.open(req, timeout=timeout) as resp:
                ct = resp.headers.get("Content-Type", "")
                if max_bytes > 0:
                    data = resp.read(max_bytes + 1)
                    if len(data) > max_bytes:
                        return (resp.status, b"", ct)  # over per-file cap
                else:
                    data = resp.read()
                return (resp.status, data, ct)
        except urllib.error.HTTPError as e:
            if getattr(e, "reason", "") == "redirect_blocked":
                return (_ST_REDIRECT_BLOCKED, b"", "")
            last_status = e.code
        except Exception:
            last_status = 0
    return (last_status, b"", "")

def download_file(url: str, dest: Path, timeout: int | None = None,
                  max_bytes: int = 0) -> dict[str, Any]:
    """Download a single file to ``dest``. Idempotent and fail-closed.

    Existing files are hashed, never re-fetched (unless REVP_FORCE_REDOWNLOAD).
    Zero-byte leftovers from an interrupted run are flagged for review.
    """
    ensure_dir(dest.parent)
    if max_bytes <= 0:
        max_bytes = max_bytes_per_file()

    if dest.exists() and not force_redownload():
        size = dest.stat().st_size
        if size <= 0:
            return {"downloaded": "false", "download_attempted": "false",
                    "download_status": "PARTIAL_OR_EMPTY_FILE_FAIL_CLOSED",
                    "file_sha256_short": "", "file_size_bytes": "0",
                    "http_status": "", "content_type": ""}
        return {"downloaded": "false", "download_attempted": "false",
                "download_status": "ALREADY_EXISTS_HASHED",
                "file_sha256_short": sha256_file_short(dest),
                "file_size_bytes": str(size), "http_status": "", "content_type": ""}

    if not network_allowed():
        return {"downloaded": "false", "download_attempted": "false",
                "download_status": "DOWNLOAD_DISABLED_QUEUE_ONLY",
                "file_sha256_short": "", "file_size_bytes": "0",
                "http_status": "", "content_type": ""}
    if not is_allowed_domain(url):
        return {"downloaded": "false", "download_attempted": "true",
                "download_status": "DOMAIN_NOT_ALLOWED_FAIL_CLOSED",
                "file_sha256_short": "", "file_size_bytes": "0",
                "http_status": "", "content_type": ""}

    status, data, ct = rate_limited_get(url, timeout, max_bytes)
    if status == 200 and data:
        dest.write_bytes(data)
        return {"downloaded": "true", "download_attempted": "true",
                "download_status": "DOWNLOADED_OK",
                "file_sha256_short": sha256_file_short(dest),
                "file_size_bytes": str(len(data)),
                "http_status": str(status), "content_type": ct}
    if status == 200 and not data:
        return {"downloaded": "false", "download_attempted": "true",
                "download_status": "SKIPPED_MAX_SIZE_LIMIT",
                "file_sha256_short": "", "file_size_bytes": "0",
                "http_status": str(status), "content_type": ct}
    if status == _ST_REDIRECT_BLOCKED:
        reason = "REDIRECT_BLOCKED_FAIL_CLOSED"
    elif status == _ST_DOMAIN_BLOCKED:
        reason = "DOMAIN_NOT_ALLOWED_FAIL_CLOSED"
    elif status == _ST_NETWORK_DISABLED:
        reason = "DOWNLOAD_DISABLED_QUEUE_ONLY"
    else:
        reason = "DOWNLOAD_FAILED_FAIL_CLOSED"
    return {"downloaded": "false", "download_attempted": "true",
            "download_status": reason, "file_sha256_short": "",
            "file_size_bytes": "0",
            "http_status": str(status) if status > 0 else "", "content_type": ct}

def download_text(url: str, timeout: int | None = None,
                  max_bytes: int = 500_000) -> tuple[str, int]:
    """Download text content. Returns (text, http_status)."""
    status, data, _ = rate_limited_get(url, timeout, max_bytes)
    if status == 200 and data:
        return data.decode("utf-8", errors="replace"), status
    return "", status

def robots_policy_check(url: str) -> str:
    """Best-effort robots.txt check. Returns ALLOWED/DISALLOWED/UNKNOWN."""
    parsed = urllib.parse.urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    try:
        text, status = download_text(robots_url, timeout=10, max_bytes=100_000)
        if status != 200:
            return "ROBOTS_NOT_FOUND_ASSUME_ALLOWED"
        path = parsed.path or "/"
        # Naive check: if Disallow matches our path
        for line in text.splitlines():
            line = line.strip()
            if line.lower().startswith("disallow:"):
                disallowed = line.split(":", 1)[1].strip()
                if disallowed and path.startswith(disallowed):
                    return "DISALLOWED"
        return "ALLOWED"
    except Exception:
        return "ROBOTS_CHECK_FAILED_ASSUME_ALLOWED"

def load_allowed_sources_config(path: Path | None = None) -> list[dict[str, str]]:
    if path and path.exists():
        return read_csv_safe(path)
    return []

def default_allowed_sources() -> list[dict[str, str]]:
    return [
        {"source_name": "INMET", "base_url": "https://portal.inmet.gov.br/uploads/dadoshistoricos/", "domain": "portal.inmet.gov.br", "enabled": "true"},
        {"source_name": "ANA_SNIRH", "base_url": "https://www.snirh.gov.br/hidroweb/", "domain": "www.snirh.gov.br", "enabled": "true"},
        {"source_name": "ANA_TELEMETRIA", "base_url": "https://telemetriaws1.ana.gov.br/", "domain": "telemetriaws1.ana.gov.br", "enabled": "true"},
        {"source_name": "CEMADEN", "base_url": "https://www.gov.br/cemaden/pt-br", "domain": "www.gov.br", "enabled": "false"},
        {"source_name": "SGB_CPRM", "base_url": "https://rigeo.sgb.gov.br/", "domain": "rigeo.sgb.gov.br", "enabled": "false"},
        {"source_name": "IBGE_LIMITES", "base_url": "https://servicodados.ibge.gov.br/api/v3/malhas/municipios/", "domain": "servicodados.ibge.gov.br", "enabled": "true"},
    ]
