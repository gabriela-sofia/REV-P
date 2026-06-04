#!/usr/bin/env python3
"""
v1uj — Focused Artifact Downloader

Le os registries v1uj (Copernicus, CKAN, S2iD, RIGeo), monta o download
manifest e baixa apenas artefatos permitidos para:
  local_only/protocolo_c/focused_public_artifacts/raw/v1uj/<source>/<event>/

Calcula SHA256. Nunca versiona bruto. Respeita max MB, allowlist de dominios e
filtros de extensao. Sem login, sem bypass. Default dry_run sem --download.
GeoSGB e metadata-only (nao baixa features). PDF deeplinks sao leads (nao baixa).
"""

import argparse
from collections import defaultdict
import csv
import hashlib
import os
import re
from urllib.parse import unquote
from urllib.parse import urlparse

try:
    import urllib.request
    HAS_URLLIB = True
except ImportError:
    HAS_URLLIB = False

try:
    import yaml
except ImportError:
    yaml = None

PROTOCOL_VERSION = "v1uj"

MANIFEST_COLUMNS = [
    "download_id", "source_tag", "event_id", "url", "resource_format",
    "extension", "domain", "download_status", "sha256", "file_size_bytes",
    "local_path_hash", "blocking_reason", "notes", "safe_filename",
    "url_sha1_12", "local_target_hash", "existing_file_status",
    "collision_status",
]

COLLISION_AUDIT_COLUMNS = [
    "collision_id", "event_id", "source_id", "url", "original_basename",
    "normalized_basename", "current_local_target", "url_sha1_12",
    "proposed_safe_filename", "collision_group", "collision_status",
    "required_action", "notes",
]

ALLOWED_EXTENSIONS = {
    ".zip", ".shp", ".shx", ".dbf", ".prj", ".gpkg", ".geojson", ".gml",
    ".kml", ".kmz", ".csv", ".xlsx", ".xls", ".pdf", ".json", ".xml",
}
BLOCKED_EXTENSIONS = {".exe", ".bat", ".cmd", ".ps1", ".sh", ".py", ".dll", ".msi"}

# registry_path -> (url_field, format_field, candidate_predicate)
ADAPTERS = {
    # Copernicus: so baixa se permitido E nao marcado explicitamente como
    # nao-event-specific (evita ingerir produtos de ativacoes off-target).
    "copernicus": ("product_url", "format_hint",
                   lambda r: r.get("download_allowed") == "true"
                             and r.get("is_event_specific") != "false"),
    "ckan": ("resource_url", "resource_format",
             lambda r: r.get("is_geospatial_candidate") == "true"
                       and r.get("is_contextual_only") != "true"),
    "s2id": ("resource_url", "resource_format",
             lambda r: r.get("record_class") == "table_with_coordinates_candidate"),
    "rigeo": ("bitstream_url", "bitstream_extension",
              lambda r: r.get("download_allowed") == "true"),
}


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_yaml(path):
    if yaml is None or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_allowed_domains(path):
    cfg = load_yaml(path)
    domains = set()
    for group in cfg.get("allowed_domains", {}).values():
        if isinstance(group, list):
            domains.update(group)
    return domains


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sha1_12(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def hash_path(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def ext_from(url, fmt_hint):
    ext = os.path.splitext(urlparse(url).path)[1].lower()
    if not ext and fmt_hint:
        f = fmt_hint.strip().lower()
        ext = f if f.startswith(".") else "." + f
    return ext


def original_basename(url, fmt_hint=""):
    parsed = urlparse(url)
    basename = unquote(os.path.basename(parsed.path or "")).strip()
    if not basename:
        ext = ext_from(url, fmt_hint)
        basename = "artifact" + (ext if ext else "")
    return basename


def sanitize_filename_component(value, fallback, max_len=80):
    text = (value or "").strip() or fallback
    text = unquote(text)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    text = text or fallback
    if len(text) <= max_len:
        return text
    root, ext = os.path.splitext(text)
    if ext and len(ext) < 12:
        return root[:max_len - len(ext)].rstrip("._-") + ext
    return text[:max_len].rstrip("._-")


def safe_filename_for_candidate(cand, seq=0):
    url = cand["url"]
    url_hash = sha1_12(url)
    event_id = sanitize_filename_component(cand.get("event_id"), "unbound", 48)
    source_id = sanitize_filename_component(cand.get("source_id"), cand["source_tag"], 40)
    resource_ref = (cand.get("resource_id") or cand.get("record_id")
                    or cand.get("artifact_id") or f"artifact_{seq:04d}")
    resource_ref = sanitize_filename_component(resource_ref, f"artifact_{seq:04d}", 48)
    basename = sanitize_filename_component(
        original_basename(url, cand.get("resource_format", "")), "artifact", 96)
    return f"{event_id}__{source_id}__{resource_ref}__{url_hash}__{basename}"


def safe_relative_path(local_only_dir, source_tag, event_id, safe_filename):
    return os.path.join(local_only_dir, source_tag, event_id or "unbound",
                        safe_filename)


def public_relpath(path):
    return os.path.relpath(path).replace("\\", "/")


def download_file(url, dest, timeout=60):
    if not HAS_URLLIB:
        return False, "NO_URLLIB"
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "REV-P-Academic-Research/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
        return True, "OK"
    except Exception as e:
        return False, str(e)[:120]


def build_candidates(registries):
    """Normaliza registries em candidatos de download. Funcao pura-ish."""
    cands = []
    for source_tag, rows in registries.items():
        url_field, fmt_field, pred = ADAPTERS[source_tag]
        for idx, r in enumerate(rows):
            url = (r.get(url_field) or "").strip()
            if not url or not pred(r):
                continue
            record_id = (r.get("ckan_record_id") or r.get("copernicus_record_id")
                         or r.get("s2id_record_id") or r.get("rigeo_record_id")
                         or r.get("record_id") or "")
            resource_id = (r.get("resource_id") or r.get("product_id")
                           or r.get("bitstream_id") or "")
            cands.append({
                "source_tag": source_tag,
                "source_id": source_tag,
                "event_id": r.get("event_id", ""),
                "url": url,
                "resource_format": r.get(fmt_field, ""),
                "record_id": record_id,
                "resource_id": resource_id,
                "artifact_id": f"{source_tag}_{idx:04d}",
            })
    return cands


def dedupe_by_url(candidates):
    unique = []
    seen = set()
    for cand in candidates:
        url = cand["url"]
        if url in seen:
            continue
        seen.add(url)
        unique.append(cand)
    return unique


def collision_groups(candidates, local_only_dir):
    basename_groups = defaultdict(set)
    target_groups = defaultdict(set)
    for cand in candidates:
        basename = original_basename(cand["url"], cand.get("resource_format", ""))
        normalized = sanitize_filename_component(basename.lower(), "artifact", 120)
        legacy_target = os.path.join(
            local_only_dir, cand["source_tag"], cand.get("event_id") or "unbound", basename)
        basename_groups[(cand.get("event_id", ""), normalized)].add(cand["url"])
        target_groups[public_relpath(legacy_target)].add(cand["url"])
    collision_keys = {}
    seq = 0
    for key, urls in sorted(basename_groups.items()):
        if len(urls) > 1:
            seq += 1
            collision_keys[("basename", key)] = f"COLLISION_GROUP_{seq:04d}"
    for key, urls in sorted(target_groups.items()):
        if len(urls) > 1:
            seq += 1
            collision_keys[("target", key)] = f"COLLISION_GROUP_{seq:04d}"
    return basename_groups, target_groups, collision_keys


def write_collision_audit(candidates, local_only_dir, out_path):
    basename_groups, target_groups, group_ids = collision_groups(candidates, local_only_dir)
    rows = []
    seen_urls = defaultdict(int)
    for seq, cand in enumerate(candidates):
        url = cand["url"]
        basename = original_basename(url, cand.get("resource_format", ""))
        normalized = sanitize_filename_component(basename.lower(), "artifact", 120)
        legacy_target = os.path.join(
            local_only_dir, cand["source_tag"], cand.get("event_id") or "unbound", basename)
        legacy_rel = public_relpath(legacy_target)
        basename_key = (cand.get("event_id", ""), normalized)
        target_key = legacy_rel
        basename_collision = len(basename_groups[basename_key]) > 1
        target_collision = len(target_groups[target_key]) > 1
        duplicate_url = seen_urls[url] > 0
        seen_urls[url] += 1
        if target_collision:
            group = group_ids.get(("target", target_key), "")
        elif basename_collision:
            group = group_ids.get(("basename", basename_key), "")
        else:
            group = ""

        exists = os.path.exists(legacy_target)
        existing_note = ""
        if exists:
            existing_note = f"legacy_target_exists_sha256={sha256_file(legacy_target)[:16]}"

        if duplicate_url:
            status = "DUPLICATE_CANDIDATE_SAME_URL"
            required = "DEDUPLICATE_BY_URL"
        elif target_collision or basename_collision:
            status = "COLLISION_DETECTED"
            required = "DOWNLOAD_TO_SAFE_FILENAME"
        else:
            status = "NO_COLLISION"
            required = "USE_SAFE_FILENAME"

        safe_name = safe_filename_for_candidate(cand, seq)
        rows.append({
            "collision_id": f"COLL_{PROTOCOL_VERSION}_{seq:04d}",
            "event_id": cand.get("event_id", ""),
            "source_id": cand.get("source_id", cand["source_tag"]),
            "url": url,
            "original_basename": basename,
            "normalized_basename": normalized,
            "current_local_target": legacy_rel,
            "url_sha1_12": sha1_12(url),
            "proposed_safe_filename": safe_name,
            "collision_group": group,
            "collision_status": status,
            "required_action": required,
            "notes": existing_note,
        })

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLLISION_AUDIT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description="v1uj — Focused Artifact Downloader")
    parser.add_argument("--copernicus", default="datasets/protocolo_c/v1uj_copernicus_ems_registry.csv")
    parser.add_argument("--ckan", default="datasets/protocolo_c/v1uj_ckan_resource_registry.csv")
    parser.add_argument("--s2id", default="datasets/protocolo_c/v1uj_s2id_resource_registry.csv")
    parser.add_argument("--rigeo", default="datasets/protocolo_c/v1uj_rigeo_bitstream_registry.csv")
    parser.add_argument("--allowed-domains", default="configs/protocolo_c/v1ui_allowed_domains.yaml")
    parser.add_argument("--local-only-dir", default="local_only/protocolo_c/focused_public_artifacts/raw/v1uj")
    parser.add_argument("--out", default="datasets/protocolo_c/v1uj_focused_download_manifest.csv")
    parser.add_argument("--collision-audit-out", default="datasets/protocolo_c/v1uj_download_collision_audit.csv")
    parser.add_argument("--allow-web", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-download-mb", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    registries = {
        "copernicus": load_csv(args.copernicus),
        "ckan": load_csv(args.ckan),
        "s2id": load_csv(args.s2id),
        "rigeo": load_csv(args.rigeo),
    }
    allowed = load_allowed_domains(args.allowed_domains)
    raw_candidates = build_candidates(registries)
    collision_rows = write_collision_audit(
        raw_candidates, args.local_only_dir, args.collision_audit_out)
    candidates = dedupe_by_url(raw_candidates)
    do_download = args.allow_web and args.download

    rows = []
    seq = 0
    total_bytes = 0
    content_hash_to_url = {}
    collision_by_url = {}
    for row in collision_rows:
        if row["url"] not in collision_by_url or row["collision_status"] == "COLLISION_DETECTED":
            collision_by_url[row["url"]] = row["collision_status"]

    for cand in candidates:
        url = cand["url"]
        source_tag = cand["source_tag"]
        event_id = cand["event_id"] or "unbound"
        ext = ext_from(url, cand["resource_format"])
        domain = urlparse(url).hostname or ""
        domain_ok = any(domain.endswith(d) for d in allowed) if allowed else False
        ext_ok = ext in ALLOWED_EXTENSIONS and ext not in BLOCKED_EXTENSIONS
        safe_name = safe_filename_for_candidate(cand, seq)
        url_hash = sha1_12(url)
        dest = safe_relative_path(args.local_only_dir, source_tag, event_id, safe_name)

        dl_status = "DRY_RUN"
        sha = ""
        size = 0
        blocking = ""
        existing_file_status = ""
        collision_status = collision_by_url.get(url, "NO_COLLISION")

        if not domain_ok:
            dl_status = "BLOCKED_DOMAIN"
            blocking = "domain_not_allowed"
        elif not ext_ok:
            dl_status = "BLOCKED_EXTENSION"
            blocking = "extension_not_allowed"
        elif do_download:
            if os.path.exists(dest):
                size = os.path.getsize(dest)
                sha = sha256_file(dest)
                dl_status = "ALREADY_EXISTS_SAME_URL_SAME_HASH"
                existing_file_status = dl_status
                content_hash_to_url.setdefault(sha, url)
            else:
                ok, msg = download_file(url, dest, args.timeout)
                if ok:
                    size = os.path.getsize(dest)
                    if size > args.max_download_mb * 1024 * 1024:
                        os.remove(dest)
                        size = 0
                        dl_status = "SIZE_LIMIT"
                        blocking = "exceeds_max_mb"
                    else:
                        sha = sha256_file(dest)
                        previous_url = content_hash_to_url.get(sha)
                        if previous_url and previous_url != url:
                            dl_status = "DUPLICATE_CONTENT_DIFFERENT_URL"
                            existing_file_status = dl_status
                        else:
                            dl_status = "DOWNLOAD_OK"
                        content_hash_to_url.setdefault(sha, url)
                        total_bytes += size
                else:
                    msg_l = msg.lower()
                    if "ssl" in msg_l:
                        dl_status = "SSL_ERROR"
                    elif "timed out" in msg_l or "timeout" in msg_l:
                        dl_status = "TIMEOUT"
                    elif "http error" in msg_l or "403" in msg_l or "404" in msg_l or "500" in msg_l:
                        dl_status = "HTTP_ERROR"
                    else:
                        dl_status = "HTTP_ERROR"
                    blocking = "download_failed"
                    existing_file_status = msg[:80]
        elif args.allow_web:
            dl_status = "PLANNED"  # --allow-web sem --download

        rows.append({
            "download_id": f"FDL_{PROTOCOL_VERSION}_{seq:04d}",
            "source_tag": source_tag, "event_id": cand["event_id"],
            "url": url, "resource_format": cand["resource_format"],
            "extension": ext, "domain": domain,
            "download_status": dl_status, "sha256": sha,
            "file_size_bytes": str(size), "local_path_hash": hash_path(url),
            "blocking_reason": blocking, "notes": "",
            "safe_filename": safe_name, "url_sha1_12": url_hash,
            "local_target_hash": hash_path(public_relpath(dest)),
            "existing_file_status": existing_file_status,
            "collision_status": collision_status,
        })
        seq += 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    downloaded = sum(1 for r in rows if r["download_status"] == "DOWNLOAD_OK")
    duplicates = sum(1 for r in rows if r["download_status"] == "DUPLICATE_CONTENT_DIFFERENT_URL")
    already = sum(1 for r in rows if r["download_status"] == "ALREADY_EXISTS_SAME_URL_SAME_HASH")
    collisions = sum(1 for r in collision_rows if r["collision_status"] == "COLLISION_DETECTED")
    print(f"[Focused Artifact Downloader v1uj] {len(rows)} candidates | "
          f"downloaded={downloaded} | duplicate_content={duplicates} | "
          f"already_same_url={already} | {total_bytes/1024/1024:.1f}MB")
    print(f"  collision_audit_rows={len(collision_rows)} | collisions_detected={collisions}")
    print(f"  never_version_raw=true | no_login_or_auth=true")
    print(f"\nManifest: {args.out}")
    print(f"Collision audit: {args.collision_audit_out}")


if __name__ == "__main__":
    main()
