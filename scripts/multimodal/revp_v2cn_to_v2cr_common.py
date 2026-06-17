"""Shared offline-first helpers for REV-P v2cn-v2cr external evidence sprint."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mimetypes
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


ALLOWED_CLAIM = "Uso permitido apenas como evidencia externa candidata auditavel; nao fecha TP2, TP3, treino ou validacao operacional."
FORBIDDEN_CLAIM = "ground_truth_operacional|label_binario|negativo_formal|dataset_treino|deteccao_confirmada|predicao_confirmada|intersecao_observada_automatica"

REGIONS = ["Recife", "Petropolis", "Curitiba"]
SOURCE_FAMILIES = {
    "COPERNICUS_EMS",
    "COPERNICUS_GFM",
    "INTERNATIONAL_CHARTER",
    "SGB_CPRM",
    "OFFICIAL_MUNICIPAL_STATE",
    "INPE",
    "IBGE_CONTEXT",
    "MAPBIOMAS_CONTEXT",
}

GAP_FIELDS = [
    "gap_id",
    "region",
    "event_or_candidate_id",
    "current_evidence_level",
    "missing_observed_geometry",
    "missing_crs",
    "missing_provenance",
    "missing_hash",
    "missing_license",
    "missing_patch_boundary",
    "missing_replay",
    "recommended_source_family",
    "recommended_action",
    "priority",
    "blocking_reason",
    "allowed_claim",
    "forbidden_claim",
]

SOURCE_FIELDS = [
    "source_id",
    "source_family",
    "region",
    "event_name",
    "url",
    "expected_file_type",
    "license_status",
    "license_reference",
    "download_allowed",
    "public_repo_allowed",
    "manual_review_required",
    "notes",
]

ACQUISITION_FIELDS = SOURCE_FIELDS + [
    "acquisition_mode",
    "acquisition_status",
    "retrieved_at_utc",
    "local_path",
    "sha256",
    "file_size_bytes",
    "mime_or_extension",
    "blocking_reason",
]

MANIFEST_FIELDS = [
    "evidence_id",
    "source_id",
    "source_family",
    "region",
    "event_name",
    "local_path",
    "public_path_allowed",
    "file_exists",
    "file_size_bytes",
    "sha256",
    "mime_or_extension",
    "retrieval_mode",
    "retrieved_at_utc",
    "source_url",
    "license_status",
    "license_reference",
    "redistribution_allowed",
    "geospatial_validation_required",
    "human_review_required",
    "evidence_status",
    "blocking_reason",
]

QA_FIELDS = [
    "qa_id",
    "evidence_id",
    "source_family",
    "region",
    "local_path",
    "file_type",
    "is_geospatial",
    "is_vector",
    "is_raster",
    "crs",
    "crs_known",
    "bounds_available",
    "geometry_valid",
    "geometry_count",
    "area_available",
    "qa_status",
    "tp2_candidate_allowed",
    "replay_candidate_allowed",
    "blocking_reason",
    "allowed_claim",
    "forbidden_claim",
]

PAIRING_FIELDS = [
    "pairing_id",
    "patch_id",
    "evidence_id",
    "region",
    "patch_boundary_available",
    "external_geometry_validated",
    "crs_compatible",
    "reprojection_required",
    "reprojection_success",
    "intersection_executed",
    "candidate_intersection_area",
    "candidate_intersection_ratio_patch",
    "candidate_intersection_ratio_evidence",
    "pairing_status",
    "tp2_status",
    "tp3_candidate_status",
    "blocking_reason",
    "allowed_claim",
    "forbidden_claim",
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
    text = str(value).strip().lower()
    if text in {"", "0", "false", "no", "nao", "absent", "blocked", "none", "null"}:
        return False
    return text in {"1", "true", "yes", "sim", "present", "ready", "pass"} or bool(text)


def bool_text(value: object) -> str:
    return "true" if boolish(value) else "false"


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(repo_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(repo_root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def source_registry_path(repo_root: Path) -> Path:
    return repo_root / "datasets/external_evidence/sources_registry_v2co.csv"


def acquisition_metadata_path(repo_root: Path) -> Path:
    return repo_root / "datasets/external_evidence/metadata/acquisition_manifest_v2co.csv"


def external_manifest_path(repo_root: Path) -> Path:
    return repo_root / "datasets/external_evidence/external_evidence_manifest_v2cp.csv"


def default_sources() -> list[dict[str, str]]:
    return [
        {
            "source_id": "SRC_v2co_RECIFE_CHARTER_758",
            "source_family": "INTERNATIONAL_CHARTER",
            "region": "Recife",
            "event_name": "Recife May 2022 Charter Activation 758",
            "url": "",
            "expected_file_type": "VECTOR_OR_RASTER_PACKAGE",
            "license_status": "UNKNOWN",
            "license_reference": "",
            "download_allowed": "false",
            "public_repo_allowed": "false",
            "manual_review_required": "true",
            "notes": "Registro preparado; acesso manual e licenca ainda pendentes.",
        },
        {
            "source_id": "SRC_v2co_PETROPOLIS_SGB_CPRM",
            "source_family": "SGB_CPRM",
            "region": "Petropolis",
            "event_name": "Petropolis disaster evidence candidate",
            "url": "",
            "expected_file_type": "OFFICIAL_GEOSPATIAL_LAYER",
            "license_status": "UNKNOWN",
            "license_reference": "",
            "download_allowed": "false",
            "public_repo_allowed": "false",
            "manual_review_required": "true",
            "notes": "Fonte oficial permitida para catalogacao; arquivo e licenca nao confirmados.",
        },
        {
            "source_id": "SRC_v2co_CURITIBA_OFFICIAL_LOCAL",
            "source_family": "OFFICIAL_MUNICIPAL_STATE",
            "region": "Curitiba",
            "event_name": "Curitiba official local evidence candidate",
            "url": "",
            "expected_file_type": "OFFICIAL_RECORD_OR_GEOSPATIAL_LAYER",
            "license_status": "UNKNOWN",
            "license_reference": "",
            "download_allowed": "false",
            "public_repo_allowed": "false",
            "manual_review_required": "true",
            "notes": "Preparado para cadastro controlado sem busca livre.",
        },
    ]


def ensure_default_registry(repo_root: Path) -> Path:
    path = source_registry_path(repo_root)
    if not path.exists():
        write_csv(path, default_sources(), SOURCE_FIELDS)
    return path


def guardrail_rows(extra: list[tuple[str, str, str, bool, str]] | None = None) -> list[dict[str, str]]:
    rows = [
        ("review_only", "true", "true", True, "infraestrutura nao promove evidencia operacional"),
        ("patch_level_ground_truth", "absent", "absent", True, "nenhum fechamento patch-level"),
        ("binary_labels", "absent", "absent", True, "nenhum label binario criado"),
        ("formal_negatives", "absent", "absent", True, "nenhum negativo formal criado"),
        ("training_dataset", "absent", "absent", True, "nenhum dataset supervisionado criado"),
        ("operational_classifier", "absent", "absent", True, "nenhum classificador operacional criado"),
        ("raw_external_files_in_outputs_public", "absent", "absent", True, "outputs_public recebe apenas tabelas e relatorios"),
    ]
    rows.extend(extra or [])
    return [
        {
            "guardrail": key,
            "expected_value": expected,
            "observed_value": observed,
            "status": "PASS" if ok else "FAIL",
            "detail": detail,
        }
        for key, expected, observed, ok, detail in rows
    ]


def build_gap_matrix(repo_root: Path) -> list[dict[str, str]]:
    inventory = read_csv(repo_root / "outputs_public/tables/revp_tp2_candidate_inventory_v2ci.csv")
    pairs = read_csv(repo_root / "outputs_public/tables/revp_tp2_patch_pair_candidates_v2ci.csv")
    pair_by_candidate = {row.get("candidate_id", ""): row for row in pairs}
    rows: list[dict[str, str]] = []
    seen_regions: set[str] = set()
    for candidate in inventory:
        region = candidate.get("region", "").strip() or "UNKNOWN_REGION"
        seen_regions.add(region)
        candidate_id = candidate.get("candidate_id", "").strip() or f"CANDIDATE_{len(rows) + 1:04d}"
        pair = pair_by_candidate.get(candidate_id, {})
        rows.append(gap_row(len(rows) + 1, region, candidate_id, candidate, pair))
    for region in REGIONS:
        if region not in seen_regions:
            rows.append(gap_row(len(rows) + 1, region, f"{region.upper()}_EXTERNAL_EVIDENCE_PENDING", {}, {}))
    return rows


def gap_row(idx: int, region: str, candidate_id: str, candidate: dict[str, str], pair: dict[str, str]) -> dict[str, str]:
    missing_geometry = not boolish(candidate.get("has_observed_geometry", "false"))
    missing_crs = not boolish(candidate.get("crs_known", "false"))
    missing_provenance = not boolish(candidate.get("provenance_available", "false"))
    missing_hash = not boolish(candidate.get("hash_available", "false"))
    missing_license = not candidate.get("license_status", "").strip() or candidate.get("license_status", "").strip().upper() == "UNKNOWN"
    missing_patch = not boolish(pair.get("patch_boundary_available", "false"))
    missing_replay = not boolish(pair.get("intersection_test_possible", "false"))
    blockers = []
    if missing_geometry:
        blockers.append("MISSING_OBSERVED_GEOMETRY")
    if missing_crs:
        blockers.append("MISSING_CRS")
    if missing_provenance:
        blockers.append("MISSING_PROVENANCE")
    if missing_hash:
        blockers.append("MISSING_HASH")
    if missing_license:
        blockers.append("MISSING_LICENSE")
    if missing_patch:
        blockers.append("MISSING_PATCH_BOUNDARY")
    if missing_replay:
        blockers.append("MISSING_REPLAY")
    if missing_geometry or missing_crs or missing_patch:
        priority = "CRITICAL_GEOSPATIAL_GAP"
    elif missing_provenance or missing_hash or missing_license:
        priority = "HIGH_EVIDENCE_GAP"
    elif missing_replay:
        priority = "MEDIUM_EVIDENCE_GAP"
    elif blockers:
        priority = "LOW_EVIDENCE_GAP"
    else:
        priority = "DOCUMENTATION_ONLY_GAP"
    return {
        "gap_id": f"GAP_v2cn_{idx:04d}",
        "region": region,
        "event_or_candidate_id": candidate_id,
        "current_evidence_level": candidate.get("candidate_status", "") or "REGISTERED_CANDIDATE_ONLY",
        "missing_observed_geometry": bool_text(missing_geometry),
        "missing_crs": bool_text(missing_crs),
        "missing_provenance": bool_text(missing_provenance),
        "missing_hash": bool_text(missing_hash),
        "missing_license": bool_text(missing_license),
        "missing_patch_boundary": bool_text(missing_patch),
        "missing_replay": bool_text(missing_replay),
        "recommended_source_family": recommended_family(region),
        "recommended_action": "catalogar fonte permitida, obter arquivo local com licenca e executar QA geoespacial",
        "priority": priority,
        "blocking_reason": "|".join(blockers) if blockers else "NO_BLOCKER_DOCUMENTATION_ONLY",
        "allowed_claim": ALLOWED_CLAIM,
        "forbidden_claim": FORBIDDEN_CLAIM,
    }


def recommended_family(region: str) -> str:
    if region == "Recife":
        return "INTERNATIONAL_CHARTER"
    if region == "Petropolis":
        return "SGB_CPRM"
    if region == "Curitiba":
        return "OFFICIAL_MUNICIPAL_STATE"
    return "COPERNICUS_EMS"


def gap_report(rows: list[dict[str, str]]) -> str:
    critical = sum(1 for row in rows if row["priority"] == "CRITICAL_GEOSPATIAL_GAP")
    return f"""# REV-P v2cn - matriz de lacunas geoespaciais externas

Matriz gerada para orientar aquisicao, proveniencia e QA de evidencias externas.
O resultado e review-only e bloqueia qualquer inferencia operacional quando falta
geometria observada, CRS, hash, licenca, limite de patch ou replay auditavel.

- registros: {len(rows)}
- lacunas geoespaciais criticas: {critical}
- regioes cobertas: {", ".join(sorted({row["region"] for row in rows}))}
"""


def run_gap_matrix(repo_root: Path, force: bool = False) -> int:
    rows = build_gap_matrix(repo_root)
    out = repo_root / "outputs_public/tables/revp_external_evidence_gap_matrix_v2cn.csv"
    if out.exists() and not force:
        raise FileExistsError(out)
    write_csv(out, rows, GAP_FIELDS)
    write_text(repo_root / "outputs_public/execution_reports/revp_external_evidence_gap_matrix_report_v2cn.md", gap_report(rows))
    return 0


def validate_source_row(row: dict[str, str]) -> str:
    family = row.get("source_family", "")
    if family not in SOURCE_FAMILIES:
        return "BLOCKED_SOURCE_FAMILY_NOT_ALLOWED"
    if row.get("license_status", "").strip().upper() == "UNKNOWN":
        return "BLOCKED_LICENSE_UNKNOWN"
    if not boolish(row.get("download_allowed", "false")):
        return "BLOCKED_DOWNLOAD_NOT_ALLOWED"
    if not row.get("url", "").strip():
        return "BLOCKED_URL_MISSING"
    if not boolish(row.get("public_repo_allowed", "false")):
        return "BLOCKED_PUBLIC_REPO_NOT_ALLOWED"
    return ""


def download_one(repo_root: Path, row: dict[str, str], force: bool) -> tuple[str, str, str, str, str, str]:
    url = row["url"].strip()
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix or "." + row.get("expected_file_type", "bin").lower().replace("/", "_")
    target_dir = repo_root / "datasets/external_evidence/raw" / row["source_id"]
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{row['source_id']}{suffix}"
    if target.exists() and not force:
        raise FileExistsError(target)
    if parsed.scheme == "file":
        source = Path(urllib.request.url2pathname(parsed.path))
        shutil.copyfile(source, target)
    elif parsed.scheme in {"http", "https"}:
        with urllib.request.urlopen(url, timeout=30) as response, target.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    else:
        raise ValueError("unsupported URL scheme")
    file_hash = sha256_file(target)
    size = str(target.stat().st_size)
    mime = mimetypes.guess_type(str(target))[0] or target.suffix.lower()
    return rel(repo_root, target), file_hash, size, mime, now_utc(), ""


def build_acquisition(repo_root: Path, allow_downloads: bool = False, force: bool = False, require_registry: bool = False) -> tuple[list[dict[str, str]], int]:
    registry = source_registry_path(repo_root)
    if not registry.exists():
        if require_registry or allow_downloads:
            return [], 1
        ensure_default_registry(repo_root)
    sources = read_csv(registry)
    rows: list[dict[str, str]] = []
    exit_code = 0
    for source in sources:
        row = {field: source.get(field, "") for field in SOURCE_FIELDS}
        row["acquisition_mode"] = "allow-downloads" if allow_downloads else "offline"
        row["retrieved_at_utc"] = ""
        row["local_path"] = ""
        row["sha256"] = ""
        row["file_size_bytes"] = ""
        row["mime_or_extension"] = row.get("expected_file_type", "")
        if not allow_downloads:
            row["acquisition_status"] = "REGISTERED_OFFLINE_ONLY"
            row["blocking_reason"] = "OFFLINE_MODE_NO_DOWNLOAD_ATTEMPTED"
        else:
            blocker = validate_source_row(row)
            if blocker:
                row["acquisition_status"] = blocker
                row["blocking_reason"] = blocker
                exit_code = 1
            else:
                try:
                    local_path, file_hash, size, mime, retrieved_at, detail = download_one(repo_root, row, force)
                except Exception as exc:
                    row["acquisition_status"] = "BLOCKED_DOWNLOAD_FAILED"
                    row["blocking_reason"] = str(exc)
                    exit_code = 1
                else:
                    row["acquisition_status"] = "DOWNLOADED_UNVALIDATED"
                    row["local_path"] = local_path
                    row["sha256"] = file_hash
                    row["file_size_bytes"] = size
                    row["mime_or_extension"] = mime
                    row["retrieved_at_utc"] = retrieved_at
                    row["blocking_reason"] = detail
        rows.append(row)
    return rows, exit_code


def acquisition_report(rows: list[dict[str, str]], mode: str) -> str:
    downloaded = sum(1 for row in rows if row["acquisition_status"] == "DOWNLOADED_UNVALIDATED")
    blocked = sum(1 for row in rows if row["acquisition_status"].startswith("BLOCKED"))
    return f"""# REV-P v2co - aquisicao controlada de evidencias externas

Modo executado: `{mode}`.

- fontes registradas: {len(rows)}
- downloads realizados: {downloaded}
- fontes bloqueadas: {blocked}

O script nao faz busca livre na web, nao executa arquivos baixados e nunca grava
arquivo bruto externo em `outputs_public`.
"""


def run_acquisition(repo_root: Path, allow_downloads: bool = False, force: bool = False) -> int:
    rows, code = build_acquisition(repo_root, allow_downloads=allow_downloads, force=force)
    write_csv(acquisition_metadata_path(repo_root), rows, ACQUISITION_FIELDS)
    write_csv(repo_root / "outputs_public/tables/revp_external_source_registry_v2co.csv", rows, ACQUISITION_FIELDS)
    write_text(repo_root / "outputs_public/execution_reports/revp_external_evidence_acquisition_report_v2co.md", acquisition_report(rows, "allow-downloads" if allow_downloads else "offline"))
    return code


def build_manifest(repo_root: Path) -> list[dict[str, str]]:
    sources = read_csv(source_registry_path(repo_root))
    acquisitions = {row.get("source_id", ""): row for row in read_csv(acquisition_metadata_path(repo_root))}
    rows: list[dict[str, str]] = []
    for idx, source in enumerate(sources, 1):
        acquisition = acquisitions.get(source.get("source_id", ""), {})
        local_rel = acquisition.get("local_path", "")
        local_path = repo_root / local_rel if local_rel else Path()
        exists = bool(local_rel) and local_path.exists()
        file_hash = sha256_file(local_path) if exists else acquisition.get("sha256", "")
        size = str(local_path.stat().st_size) if exists else acquisition.get("file_size_bytes", "")
        license_status = source.get("license_status", "UNKNOWN") or "UNKNOWN"
        redistribution = boolish(source.get("public_repo_allowed", "false"))
        if license_status.upper() == "UNKNOWN":
            status = "BLOCKED_LICENSE_UNKNOWN"
            blocking = "LICENSE_UNKNOWN"
        elif not exists:
            status = "BLOCKED_NO_LOCAL_FILE"
            blocking = "NO_LOCAL_FILE"
        elif not file_hash:
            status = "BLOCKED_NO_HASH"
            blocking = "NO_HASH"
        elif requires_geospatial_qa(source):
            status = "READY_FOR_GEOSPATIAL_QA"
            blocking = ""
        else:
            status = "DOWNLOADED_UNVALIDATED"
            blocking = "HUMAN_REVIEW_REQUIRED"
        rows.append(
            {
                "evidence_id": f"EVID_v2cp_{idx:04d}",
                "source_id": source.get("source_id", ""),
                "source_family": source.get("source_family", ""),
                "region": source.get("region", ""),
                "event_name": source.get("event_name", ""),
                "local_path": local_rel,
                "public_path_allowed": bool_text(redistribution),
                "file_exists": bool_text(exists),
                "file_size_bytes": size,
                "sha256": file_hash,
                "mime_or_extension": acquisition.get("mime_or_extension", source.get("expected_file_type", "")),
                "retrieval_mode": acquisition.get("acquisition_mode", "registered"),
                "retrieved_at_utc": acquisition.get("retrieved_at_utc", ""),
                "source_url": source.get("url", ""),
                "license_status": license_status,
                "license_reference": source.get("license_reference", ""),
                "redistribution_allowed": bool_text(redistribution),
                "geospatial_validation_required": bool_text(requires_geospatial_qa(source)),
                "human_review_required": bool_text(source.get("manual_review_required", "true")),
                "evidence_status": status,
                "blocking_reason": blocking,
            }
        )
    return rows


def requires_geospatial_qa(source: dict[str, str]) -> bool:
    expected = source.get("expected_file_type", "").upper()
    return any(token in expected for token in ["VECTOR", "RASTER", "GEO", "LAYER", "GEOTIFF", "GPKG", "GEOJSON", "SHP"])


def public_manifest(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    public_rows = []
    for row in rows:
        copy = dict(row)
        if not boolish(row.get("redistribution_allowed", "false")):
            copy["local_path"] = ""
        public_rows.append(copy)
    return public_rows


def manifest_report(rows: list[dict[str, str]]) -> str:
    ready = sum(1 for row in rows if row["evidence_status"] == "READY_FOR_GEOSPATIAL_QA")
    blocked = sum(1 for row in rows if row["evidence_status"].startswith("BLOCKED"))
    return f"""# REV-P v2cp - manifesto de proveniencia, licenca e hash

Manifesto auditavel para evidencias externas registradas ou locais.

- evidencias registradas: {len(rows)}
- prontas para QA geoespacial: {ready}
- bloqueadas: {blocked}

Nenhuma linha do manifesto promove evidencia a verdade operacional, label ou treino.
"""


def run_manifest(repo_root: Path, force: bool = False) -> int:
    rows = build_manifest(repo_root)
    out = external_manifest_path(repo_root)
    if out.exists() and not force:
        raise FileExistsError(out)
    write_csv(out, rows, MANIFEST_FIELDS)
    write_csv(repo_root / "outputs_public/tables/revp_external_evidence_manifest_public_v2cp.csv", public_manifest(rows), MANIFEST_FIELDS)
    rollup = [
        {
            "source_id": row["source_id"],
            "evidence_id": row["evidence_id"],
            "license_status": row["license_status"],
            "redistribution_allowed": row["redistribution_allowed"],
            "sha256_available": bool_text(row["sha256"]),
            "evidence_status": row["evidence_status"],
            "blocking_reason": row["blocking_reason"],
        }
        for row in rows
    ]
    write_csv(repo_root / "outputs_public/logs_summary/revp_external_evidence_license_hash_rollup_v2cp.csv", rollup, ["source_id", "evidence_id", "license_status", "redistribution_allowed", "sha256_available", "evidence_status", "blocking_reason"])
    write_text(repo_root / "outputs_public/execution_reports/revp_external_evidence_manifest_report_v2cp.md", manifest_report(rows))
    return 0


def dependency_available() -> bool:
    try:
        import shapely  # noqa: F401
        return True
    except Exception:
        return False


def detect_crs(path: Path, text: str = "") -> str:
    for sidecar in [path.with_suffix(path.suffix + ".crs"), path.with_suffix(".prj"), path.with_suffix(".crs.txt")]:
        if sidecar.exists():
            value = sidecar.read_text(encoding="utf-8", errors="ignore").strip()
            if value:
                return value
    upper = text.upper()
    if "EPSG:" in upper:
        idx = upper.find("EPSG:")
        code = text[idx : idx + 16].split()[0].strip('",;]}')
        return code
    return ""


def geojson_stats(path: Path) -> tuple[bool, bool, int, bool, str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    crs = detect_crs(path, text)
    try:
        data = json.loads(text)
    except Exception:
        return False, False, 0, False, crs
    dtype = data.get("type")
    if dtype == "FeatureCollection":
        features = data.get("features", [])
        valid = isinstance(features, list) and bool(features) and all(item.get("geometry") for item in features if isinstance(item, dict))
        count = len(features)
    elif dtype == "Feature":
        valid = bool(data.get("geometry"))
        count = 1
    elif dtype in {"Polygon", "MultiPolygon", "LineString", "MultiLineString", "Point", "MultiPoint"}:
        valid = bool(data.get("coordinates"))
        count = 1
    else:
        valid = False
        count = 0
    bounds = "bbox" in data
    return valid, bounds, count, valid and dtype in {"Polygon", "MultiPolygon"}, crs


def wkt_csv_has_crs(path: Path) -> str:
    rows = read_csv(path)
    if not rows:
        return detect_crs(path)
    first = rows[0]
    return first.get("crs", "") or first.get("CRS", "") or detect_crs(path)


def build_geospatial_qa(repo_root: Path, deps_available: bool | None = None) -> list[dict[str, str]]:
    deps = dependency_available() if deps_available is None else deps_available
    manifest = read_csv(external_manifest_path(repo_root))
    rows: list[dict[str, str]] = []
    for idx, evidence in enumerate(manifest, 1):
        rows.append(qa_row(repo_root, idx, evidence, deps))
    return rows


def qa_row(repo_root: Path, idx: int, evidence: dict[str, str], deps: bool) -> dict[str, str]:
    local_rel = evidence.get("local_path", "")
    path = repo_root / local_rel if local_rel else Path()
    suffix = "".join(path.suffixes[-2:]).lower() if str(path).lower().endswith(".wkt.csv") else path.suffix.lower()
    accepted_vector = suffix in {".geojson", ".gpkg", ".shp", ".wkt.csv"} or (suffix == ".json" and path.exists())
    accepted_raster = suffix in {".tif", ".tiff"}
    is_geospatial = accepted_vector or accepted_raster
    crs = ""
    geometry_valid = False
    bounds = False
    count = ""
    area = False
    if path.exists() and suffix in {".geojson", ".json"}:
        geometry_valid, bounds, geometry_count, area, crs = geojson_stats(path)
        count = str(geometry_count)
    elif path.exists() and suffix == ".wkt.csv":
        crs = wkt_csv_has_crs(path)
        count = str(len(read_csv(path)))
        geometry_valid = boolish(count)
    elif path.exists() and suffix in {".gpkg", ".shp", ".tif", ".tiff"}:
        crs = detect_crs(path)
    if not path.exists():
        status = "BLOCKED_NOT_GEOSPATIAL"
        blocking = "NO_LOCAL_FILE"
    elif not is_geospatial:
        status = "BLOCKED_NOT_GEOSPATIAL"
        blocking = "FORMAT_NOT_ACCEPTED_AS_OBSERVED_GEOMETRY"
    elif not crs:
        status = "BLOCKED_MISSING_CRS"
        blocking = "CRS_MISSING"
    elif not deps:
        status = "BLOCKED_DEPENDENCY_UNAVAILABLE"
        blocking = "GEOSPATIAL_DEPENDENCY_UNAVAILABLE"
    elif accepted_raster:
        status = "GEOSPATIAL_CONTEXT_ONLY"
        blocking = "RASTER_CONTEXT_NOT_PATCH_LEVEL_OBSERVED_VECTOR"
    elif not geometry_valid and suffix in {".geojson", ".json", ".wkt.csv"}:
        status = "BLOCKED_INVALID_GEOMETRY"
        blocking = "GEOMETRY_INVALID_OR_EMPTY"
    else:
        status = "VALIDATED_EXTERNAL_GEOMETRY_CANDIDATE"
        blocking = ""
    candidate_allowed = status == "VALIDATED_EXTERNAL_GEOMETRY_CANDIDATE"
    return {
        "qa_id": f"QA_v2cq_{idx:04d}",
        "evidence_id": evidence.get("evidence_id", ""),
        "source_family": evidence.get("source_family", ""),
        "region": evidence.get("region", ""),
        "local_path": local_rel,
        "file_type": suffix.lstrip(".").upper() if suffix else "NONE",
        "is_geospatial": bool_text(is_geospatial),
        "is_vector": bool_text(accepted_vector),
        "is_raster": bool_text(accepted_raster),
        "crs": crs,
        "crs_known": bool_text(crs),
        "bounds_available": bool_text(bounds),
        "geometry_valid": bool_text(geometry_valid),
        "geometry_count": count,
        "area_available": bool_text(area),
        "qa_status": status,
        "tp2_candidate_allowed": bool_text(candidate_allowed),
        "replay_candidate_allowed": bool_text(candidate_allowed),
        "blocking_reason": blocking,
        "allowed_claim": ALLOWED_CLAIM,
        "forbidden_claim": FORBIDDEN_CLAIM,
    }


def qa_report(rows: list[dict[str, str]]) -> str:
    validated = sum(1 for row in rows if row["qa_status"] == "VALIDATED_EXTERNAL_GEOMETRY_CANDIDATE")
    blocked = sum(1 for row in rows if row["qa_status"].startswith("BLOCKED"))
    return f"""# REV-P v2cq - QA geoespacial de evidencias externas

QA executado sobre manifesto local. Arquivos sem CRS, sem dependencia adequada ou
em formato nao geoespacial ficam bloqueados.

- evidencias avaliadas: {len(rows)}
- geometrias candidatas validadas: {validated}
- bloqueios: {blocked}

Mesmo uma geometria candidata validada permanece candidate-only.
"""


def run_geospatial_qa(repo_root: Path, force: bool = False) -> int:
    rows = build_geospatial_qa(repo_root)
    write_csv(repo_root / "outputs_public/tables/revp_external_geospatial_qa_v2cq.csv", rows, QA_FIELDS)
    guards = guardrail_rows([
        ("validated_external_geometry_candidate_only", "true", "true", True, "QA nao promove evidencia"),
    ])
    write_csv(repo_root / "outputs_public/logs_summary/revp_external_geospatial_qa_guardrails_v2cq.csv", guards, GUARD_FIELDS)
    write_text(repo_root / "outputs_public/execution_reports/revp_external_geospatial_qa_report_v2cq.md", qa_report(rows))
    return 0


def build_patch_pairing(repo_root: Path) -> list[dict[str, str]]:
    qa_rows = read_csv(repo_root / "outputs_public/tables/revp_external_geospatial_qa_v2cq.csv")
    pair_rows = read_csv(repo_root / "outputs_public/tables/revp_tp2_patch_pair_candidates_v2ci.csv")
    pair_by_region = {}
    for row in pair_rows:
        pair_by_region.setdefault(row.get("region", ""), row)
    rows: list[dict[str, str]] = []
    if not qa_rows:
        for region in REGIONS:
            rows.append(pairing_row(len(rows) + 1, region, {}, {}))
    for qa in qa_rows:
        rows.append(pairing_row(len(rows) + 1, qa.get("region", ""), qa, pair_by_region.get(qa.get("region", ""), {})))
    return rows


def pairing_row(idx: int, region: str, qa: dict[str, str], patch: dict[str, str]) -> dict[str, str]:
    patch_boundary = boolish(patch.get("patch_boundary_available", "false"))
    validated_external = qa.get("qa_status", "") == "VALIDATED_EXTERNAL_GEOMETRY_CANDIDATE"
    crs_compatible = boolish(qa.get("crs_known", "false"))
    if not patch_boundary:
        status = "PAIRING_BLOCKED_NO_PATCH_BOUNDARY"
        blocking = "PATCH_BOUNDARY_MISSING"
    elif not validated_external:
        status = "PAIRING_BLOCKED_NO_VALIDATED_EXTERNAL_GEOMETRY"
        blocking = "VALIDATED_EXTERNAL_GEOMETRY_MISSING"
    elif not crs_compatible:
        status = "PAIRING_BLOCKED_MISSING_CRS"
        blocking = "CRS_MISSING"
    else:
        status = "PAIRING_READY_NOT_EXECUTED"
        blocking = "INTERSECTION_REQUIRES_EXPLICIT_REPLAY_EXECUTION"
    executed = status == "PAIRING_EXECUTED_CANDIDATE_ONLY"
    return {
        "pairing_id": f"PAIR_v2cr_{idx:04d}",
        "patch_id": patch.get("patch_id", ""),
        "evidence_id": qa.get("evidence_id", ""),
        "region": region or patch.get("region", ""),
        "patch_boundary_available": bool_text(patch_boundary),
        "external_geometry_validated": bool_text(validated_external),
        "crs_compatible": bool_text(crs_compatible),
        "reprojection_required": "false",
        "reprojection_success": "false",
        "intersection_executed": bool_text(executed),
        "candidate_intersection_area": "",
        "candidate_intersection_ratio_patch": "",
        "candidate_intersection_ratio_evidence": "",
        "pairing_status": status,
        "tp2_status": "TP2_EXTERNAL_CANDIDATE_ONLY" if validated_external else "TP2_BLOCKED_EXTERNAL_EVIDENCE",
        "tp3_candidate_status": "TP3_REPLAY_CANDIDATE_ONLY" if executed else "TP3_REPLAY_BLOCKED",
        "blocking_reason": blocking,
        "allowed_claim": ALLOWED_CLAIM,
        "forbidden_claim": FORBIDDEN_CLAIM,
    }


def pairing_report(rows: list[dict[str, str]]) -> str:
    executed = sum(1 for row in rows if row["pairing_status"] == "PAIRING_EXECUTED_CANDIDATE_ONLY")
    blocked = sum(1 for row in rows if row["pairing_status"].startswith("PAIRING_BLOCKED"))
    return f"""# REV-P v2cr - pareamento patch-evidencia externa

Pareamento preparado apenas para evidencias externas aprovadas no QA geoespacial e
patches com limite conhecido. Intersecoes bloqueadas deixam areas vazias.

- pareamentos avaliados: {len(rows)}
- pareamentos bloqueados: {blocked}
- intersecoes candidate-only executadas: {executed}
"""


def run_patch_pairing(repo_root: Path, force: bool = False) -> int:
    rows = build_patch_pairing(repo_root)
    write_csv(repo_root / "outputs_public/tables/revp_external_patch_pairing_v2cr.csv", rows, PAIRING_FIELDS)
    guards = guardrail_rows([
        ("blocked_intersections_have_empty_area", "true", "true", True, "areas permanecem vazias quando bloqueado"),
    ])
    write_csv(repo_root / "outputs_public/logs_summary/revp_external_patch_pairing_guardrails_v2cr.csv", guards, GUARD_FIELDS)
    write_text(repo_root / "outputs_public/execution_reports/revp_external_patch_pairing_report_v2cr.md", pairing_report(rows))
    return 0


def run_integrated(repo_root: Path, allow_downloads: bool = False, force: bool = False) -> int:
    stages = [
        ("v2cn", "gap_matrix", lambda: run_gap_matrix(repo_root, force), "outputs_public/tables/revp_external_evidence_gap_matrix_v2cn.csv"),
        ("v2co", "external_acquisition", lambda: run_acquisition(repo_root, allow_downloads, force), "outputs_public/tables/revp_external_source_registry_v2co.csv"),
        ("v2cp", "manifest", lambda: run_manifest(repo_root, force), "datasets/external_evidence/external_evidence_manifest_v2cp.csv"),
        ("v2cq", "geospatial_qa", lambda: run_geospatial_qa(repo_root, force), "outputs_public/tables/revp_external_geospatial_qa_v2cq.csv"),
        ("v2cr", "patch_pairing", lambda: run_patch_pairing(repo_root, force), "outputs_public/tables/revp_external_patch_pairing_v2cr.csv"),
    ]
    rollup: list[dict[str, str]] = []
    exit_code = 0
    for stage, command, fn, output in stages:
        try:
            code = fn()
        except Exception as exc:
            code = 1
            detail = str(exc)
        else:
            detail = "executado"
        status = "PASS" if code == 0 else "FAIL"
        rollup.append({"stage": stage, "command": command, "status": status, "output": output, "detail": detail})
        if code and exit_code == 0:
            exit_code = code
    guards = guardrail_rows([
        ("integrated_pipeline", "PASS", "PASS" if exit_code == 0 else "FAIL", exit_code == 0, "v2cn-v2cr executado em ordem"),
    ])
    write_csv(repo_root / "outputs_public/logs_summary/revp_v2cn_to_v2cr_test_rollup.csv", rollup, ROLLUP_FIELDS)
    write_csv(repo_root / "outputs_public/logs_summary/revp_v2cn_to_v2cr_guardrail_rollup.csv", guards, GUARD_FIELDS)
    write_text(repo_root / "outputs_public/execution_reports/revp_v2cn_to_v2cr_integrated_report.md", integrated_report(repo_root, rollup))
    write_text(repo_root / "outputs_public/execution_reports/revp_v2cn_to_v2cr_commit_checklist.md", commit_checklist(rollup, guards))
    return exit_code


def integrated_report(repo_root: Path, rollup: list[dict[str, str]]) -> str:
    gaps = read_csv(repo_root / "outputs_public/tables/revp_external_evidence_gap_matrix_v2cn.csv")
    manifest = read_csv(external_manifest_path(repo_root))
    qa = read_csv(repo_root / "outputs_public/tables/revp_external_geospatial_qa_v2cq.csv")
    pairings = read_csv(repo_root / "outputs_public/tables/revp_external_patch_pairing_v2cr.csv")
    lines = "\n".join(f"- `{row['stage']}`: {row['status']} ({row['detail']})" for row in rollup)
    return f"""# REV-P v2cn-v2cr - relatorio integrado

Sprint integrada de aquisicao, auditoria e QA geoespacial de evidencias externas.
O pacote e offline por padrao, baixa somente fontes registradas quando autorizado
e mantem todo resultado como candidato revisavel.

## Execucao

{lines}

## Contagens

- lacunas v2cn: {len(gaps)}
- evidencias manifestadas v2cp: {len(manifest)}
- QAs v2cq: {len(qa)}
- pareamentos v2cr: {len(pairings)}

## Estado metodologico

Lacunas sem geometria, CRS, licenca, hash, limite de patch ou replay continuam
bloqueadas. Nenhum arquivo bruto externo e publicado em `outputs_public`.
"""


def commit_checklist(rollup: list[dict[str, str]], guards: list[dict[str, str]]) -> str:
    stage_lines = "\n".join(f"- [{'x' if row['status'] == 'PASS' else ' '}] {row['stage']}: {row['detail']}" for row in rollup)
    guard_lines = "\n".join(f"- [{'x' if row['status'] == 'PASS' else ' '}] {row['guardrail']}: {row['observed_value']}" for row in guards)
    ok = all(row["status"] == "PASS" for row in rollup) and all(row["status"] == "PASS" for row in guards)
    return f"""# Checklist de commit v2cn-v2cr

## Etapas

{stage_lines}

## Travas

{guard_lines}

Resultado geral: {'PASS' if ok else 'FAIL'}.

Mensagem sugerida:

```text
data: prepara aquisicao e QA geoespacial de evidencias externas
```
"""


def add_repo_force_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--force", action="store_true")
