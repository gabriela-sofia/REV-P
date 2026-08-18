"""
SUSC safe downloader (offline-safe, optional network).

Light, controlled downloads of public/official files when a direct URL is given.
Hard limits: never raster, never >100MB, no API keys, no aggressive scraping,
small timeout, few retries. Every attempt is recorded (url, status, size, sha256,
reason). If urllib is unavailable or network fails, the attempt is recorded as a
failure and never raises.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from susc_io import ensure_dir, sha256_bytes  # noqa: E402

MAX_BYTES = 100 * 1024 * 1024  # 100 MB hard cap
TIMEOUT = 20
RETRIES = 2

ALLOWED_EXTS = {".csv", ".geojson", ".json", ".kml", ".kmz", ".wkt", ".txt", ".pdf",
                ".zip"}  # zip only for vector packs, capped + recorded
RASTER_EXTS = {".tif", ".tiff", ".geotiff", ".jp2", ".img", ".nc", ".hdf", ".grib"}


def _classify_ext(url: str) -> str:
    low = url.lower().split("?")[0]
    for e in RASTER_EXTS:
        if low.endswith(e):
            return "raster_blocked"
    for e in ALLOWED_EXTS:
        if low.endswith(e):
            return e
    return "unknown_ext"


def safe_download(url: str, dest_dir, filename=None) -> dict:
    """Attempt a single safe download. Returns a manifest-style dict; never raises."""
    rec: dict[str, object] = {
        "url": url, "download_status": "not_attempted", "http_status_or_error": "",
        "file_path": "", "file_size_bytes": "", "sha256": "",
        "content_ext": _classify_ext(url), "requires_manual_review": "true",
        "notes": "",
    }
    ext = rec["content_ext"]
    if ext == "raster_blocked":
        rec["download_status"] = "blocked_raster"
        rec["notes"] = "raster never downloaded"
        return rec
    if ext == "unknown_ext":
        rec["download_status"] = "blocked_unknown_extension"
        rec["notes"] = "extension not in allowlist"
        return rec
    if not (url.startswith("http://") or url.startswith("https://")):
        rec["download_status"] = "not_attempted_no_direct_url"
        return rec

    try:
        import urllib.request
    except Exception as e:  # pragma: no cover
        rec["download_status"] = "failed_no_urllib"
        rec["http_status_or_error"] = str(e)[:80]
        return rec

    last_err = ""
    for _ in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "REV-P-SUSC/1.0"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                clen = resp.headers.get("Content-Length")
                if clen and int(clen) > MAX_BYTES:
                    rec["download_status"] = "blocked_too_large"
                    rec["http_status_or_error"] = f"content-length {clen}"
                    return rec
                data = resp.read(MAX_BYTES + 1)
                if len(data) > MAX_BYTES:
                    rec["download_status"] = "blocked_too_large"
                    rec["http_status_or_error"] = "stream exceeded 100MB"
                    return rec
            dest_dir = ensure_dir(dest_dir)
            name = filename or url.split("?")[0].rstrip("/").split("/")[-1] or "download.bin"
            out = Path(dest_dir) / name
            out.write_bytes(data)
            rec.update({
                "download_status": "downloaded",
                "http_status_or_error": "200",
                "file_path": str(out),
                "file_size_bytes": len(data),
                "sha256": sha256_bytes(data),
                "notes": "light controlled download",
            })
            return rec
        except Exception as e:  # urllib.error.* and network errors
            last_err = str(e)[:120]
            continue
    rec["download_status"] = "failed"
    rec["http_status_or_error"] = last_err or "unknown_error"
    return rec
