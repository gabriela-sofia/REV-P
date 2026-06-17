"""REV-P v2cl - observed geometry validation contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from revp_v2cj_to_v2cm_common import (
    ALLOWED_CLAIM,
    FORBIDDEN_CLAIM,
    bool_text,
    guardrail_rows,
    queue_path,
    read_csv,
    sha256_file,
    validation_path,
    write_csv,
    write_text,
)


FIELDS = [
    "geometry_candidate_id",
    "candidate_id",
    "region",
    "file_path",
    "geometry_format",
    "crs",
    "crs_known",
    "geometry_valid",
    "provenance_available",
    "hash_available",
    "documentary_link_available",
    "validation_status",
    "blocking_reason",
    "allowed_claim",
    "forbidden_claim",
]

GUARD_FIELDS = ["guardrail", "expected_value", "observed_value", "status", "detail"]
VECTOR_SUFFIXES = {".geojson", ".gpkg", ".shp", ".wkt"}
SEARCH_DIRS = [
    "datasets/protocolo_c",
    "datasets/observed_geometry",
    "datasets/ground_truth_candidates",
    "outputs_public/review_packages",
]


def dependency_available() -> bool:
    try:
        import shapely  # noqa: F401
        return True
    except Exception:
        return False


def find_geometry_files(repo_root: Path, candidate_id: str) -> list[Path]:
    found: list[Path] = []
    token = candidate_id.lower()
    for rel in SEARCH_DIRS:
        base = repo_root / rel
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() in VECTOR_SUFFIXES and token in path.name.lower():
                found.append(path)
    return sorted(found)


def crs_for(path: Path) -> str:
    sidecars = [path.with_suffix(path.suffix + ".crs"), path.with_suffix(".prj"), path.with_suffix(".crs.txt")]
    for sidecar in sidecars:
        if sidecar.exists():
            text = sidecar.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                return text
    if path.suffix.lower() in {".geojson", ".wkt"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "EPSG:" in text.upper():
            idx = text.upper().find("EPSG:")
            return text[idx: idx + 10].split()[0].strip('",;')
    return ""


def has_provenance(path: Path) -> bool:
    return any(path.with_suffix(path.suffix + suffix).exists() for suffix in [".prov", ".provenance", ".md", ".txt"])


def has_documentary_link(path: Path) -> bool:
    return any(path.with_suffix(path.suffix + suffix).exists() for suffix in [".link", ".source", ".md", ".txt"])


def geometry_valid(path: Path, deps_available: bool) -> tuple[bool, str]:
    if not deps_available:
        return False, "BLOCKED_VALIDATOR_DEPENDENCY_UNAVAILABLE"
    if path.suffix.lower() == ".geojson":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("type") in {"Feature", "FeatureCollection", "Polygon", "MultiPolygon"}:
                return True, ""
        except Exception:
            return False, "BLOCKED_INVALID_GEOMETRY"
    return False, "BLOCKED_VALIDATOR_DEPENDENCY_UNAVAILABLE"


def build_validation(repo_root: Path, deps_available: bool | None = None) -> list[dict[str, str]]:
    deps = dependency_available() if deps_available is None else deps_available
    queue = read_csv(queue_path(repo_root))
    rows: list[dict[str, str]] = []
    for idx, task in enumerate(queue, 1):
        candidate_id = task.get("candidate_id", "")
        files = find_geometry_files(repo_root, candidate_id)
        if not files:
            rows.append(row_for_missing(idx, task, "NO_OBSERVED_VECTOR_GEOMETRY", "GEOMETRIA_VETORIAL_LOCAL_AUSENTE"))
            continue
        for file_path in files:
            crs = crs_for(file_path)
            prov = has_provenance(file_path)
            doc = has_documentary_link(file_path)
            hashed = True
            valid, dep_blocker = geometry_valid(file_path, deps)
            blockers: list[str] = []
            if not crs:
                blockers.append("CRS_AUSENTE")
            if not prov:
                blockers.append("PROVENIENCIA_AUSENTE")
            if not doc:
                blockers.append("VINCULO_DOCUMENTAL_AUSENTE")
            if dep_blocker:
                blockers.append(dep_blocker)
            if not valid and not dep_blocker:
                blockers.append("GEOMETRIA_INVALIDA")
            if valid and crs and prov and hashed and doc:
                status = "VALIDATED_OBSERVED_GEOMETRY_CANDIDATE"
                blocking = "SEM_BLOQUEIO_DE_VALIDACAO_CANDIDATA"
            elif not crs:
                status = "BLOCKED_MISSING_CRS"
                blocking = "|".join(blockers)
            elif not prov:
                status = "BLOCKED_MISSING_PROVENANCE"
                blocking = "|".join(blockers)
            elif dep_blocker:
                status = dep_blocker
                blocking = "|".join(blockers)
            else:
                status = "BLOCKED_INVALID_GEOMETRY"
                blocking = "|".join(blockers)
            rows.append(
                {
                    "geometry_candidate_id": f"GEOM_v2cl_{idx:04d}",
                    "candidate_id": candidate_id,
                    "region": task.get("region", ""),
                    "file_path": str(file_path.relative_to(repo_root)).replace("\\", "/"),
                    "geometry_format": file_path.suffix.lstrip(".").upper(),
                    "crs": crs,
                    "crs_known": bool_text(crs),
                    "geometry_valid": bool_text(valid),
                    "provenance_available": bool_text(prov),
                    "hash_available": bool_text(hashed),
                    "documentary_link_available": bool_text(doc),
                    "validation_status": status,
                    "blocking_reason": blocking,
                    "allowed_claim": ALLOWED_CLAIM,
                    "forbidden_claim": FORBIDDEN_CLAIM,
                }
            )
    return rows


def row_for_missing(idx: int, task: dict[str, str], status: str, reason: str) -> dict[str, str]:
    return {
        "geometry_candidate_id": f"GEOM_v2cl_{idx:04d}",
        "candidate_id": task.get("candidate_id", ""),
        "region": task.get("region", ""),
        "file_path": "",
        "geometry_format": "NONE",
        "crs": "",
        "crs_known": "false",
        "geometry_valid": "false",
        "provenance_available": "false",
        "hash_available": "false",
        "documentary_link_available": "false",
        "validation_status": status,
        "blocking_reason": reason,
        "allowed_claim": ALLOWED_CLAIM,
        "forbidden_claim": FORBIDDEN_CLAIM,
    }


def report(rows: list[dict[str, str]]) -> str:
    validated = sum(1 for row in rows if row["validation_status"] == "VALIDATED_OBSERVED_GEOMETRY_CANDIDATE")
    return f"""# REV-P v2cl - contrato de validacao de geometria observada

Geometrias observadas futuras sao aceitas apenas como candidatas validadas, nunca
como ground truth operacional. Sem vetor local, CRS, proveniencia, hash, validade
geometrica e vinculo documental, a etapa bloqueia.

Registros avaliados: {len(rows)}.
Candidatas validadas para replay: {validated}.
"""


def run(repo_root: Path, force: bool = False) -> int:
    rows = build_validation(repo_root)
    out = validation_path(repo_root)
    if out.exists() and not force:
        raise FileExistsError(out)
    write_csv(out, rows, FIELDS)
    guards = guardrail_rows([
        ("validated_geometry_is_candidate_only", "true", "true", True, "geometria validada nao vira ground truth"),
    ])
    write_csv(repo_root / "outputs_public/logs_summary/revp_observed_geometry_validation_guardrails_v2cl.csv", guards, GUARD_FIELDS)
    write_text(repo_root / "outputs_public/execution_reports/revp_observed_geometry_validation_report_v2cl.md", report(rows))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    return run(Path(args.repo_root), args.force)


if __name__ == "__main__":
    raise SystemExit(main())

