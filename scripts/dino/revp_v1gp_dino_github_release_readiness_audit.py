from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1gp"

VERSIONABLE_DIRS = ["scripts/dino", "tests", "docs", "configs"]
VERSIONABLE_EXTRA = ["README.md"]

FORBIDDEN_EXTENSIONS = {".npz", ".npy", ".tif", ".tiff", ".vrt"}
FORBIDDEN_PATTERNS_EXTRA = re.compile(
    r"\.(aux\.xml|geojson)$", re.IGNORECASE
)

_PRIVATE_USER = "gabriel" + "a"
PRIVATE_PATH_PATTERNS = [
    re.compile(r"C:\\Users\\" + _PRIVATE_USER, re.IGNORECASE),
    re.compile(r"C:/Users/" + _PRIVATE_USER, re.IGNORECASE),
    re.compile(r"/Users/[a-zA-Z]"),
    re.compile(r"/home/[a-zA-Z]"),
]

REQUIRED_DOCS = [
    "docs/dino_sentinel_embedding_protocol.md",
    "docs/dino_sentinel_scientific_evidence_summary.md",
    "docs/dino_command_registry.md",
]

README_PATH = "README.md"
README_DINO_MARKERS = [
    "dino_sentinel_embedding_protocol",
    "dino_sentinel_scientific_evidence_summary",
    "dino_command_registry",
]

REQUIRED_SCRIPTS = [
    "scripts/dino/revp_v1gn_dino_execution_health_monitor.py",
    "scripts/dino/revp_v1go_dino_pipeline_orchestrator.py",
]

REQUIRED_TESTS = [
    "tests/test_revp_v1gn_v1go_dino_operational_hardening.py",
]

COMMAND_REGISTRY_PATH = "docs/dino_command_registry.md"
COMMAND_REGISTRY_VERSIONS = ["v1gn", "v1go", "v1gp"]

METHODOLOGICAL_PROTECTIONS = [
    "review_only",
    "supervised_training",
    "labels_created",
    "predictive_claims",
    "multimodal",
]

METHODOLOGY_SEARCH_FILES = [
    "docs/dino_sentinel_embedding_protocol.md",
    "docs/dino_sentinel_scientific_evidence_summary.md",
    "docs/dino_command_registry.md",
    "configs/dino_review_only.yaml",
    "configs/project_state.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="REV-P v1gp DINO GitHub release readiness audit."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(
                f"Output directory already exists: {path}. Use --force."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def collect_versionable_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for dir_name in VERSIONABLE_DIRS:
        d = root / dir_name
        if d.is_dir():
            files.extend(
                p for p in d.rglob("*")
                if p.is_file() and "__pycache__" not in p.parts
            )
    for extra in VERSIONABLE_EXTRA:
        p = root / extra
        if p.is_file():
            files.append(p)
    return sorted(set(files))


def is_forbidden_file(path: Path) -> bool:
    name_lower = path.name.lower()
    if path.suffix.lower() in FORBIDDEN_EXTENSIONS:
        return True
    if name_lower.endswith(".aux.xml"):
        return True
    return False


def check_forbidden_artifacts(root: Path) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    local_runs = root / "local_runs"
    git_dir = root / ".git"
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        try:
            p.relative_to(local_runs)
            continue
        except ValueError:
            pass
        try:
            p.relative_to(git_dir)
            continue
        except ValueError:
            pass
        if is_forbidden_file(p):
            rel = p.relative_to(root).as_posix()
            issues.append({"file": rel, "reason": "forbidden_extension", "status": "FAIL"})
    return issues


def check_private_paths(root: Path, files: list[Path]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    for path in files:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for pattern in PRIVATE_PATH_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                rel = path.relative_to(root).as_posix()
                issues.append({
                    "file": rel,
                    "pattern": pattern.pattern,
                    "match_count": str(len(matches)),
                    "status": "FAIL",
                })
                break
    return issues


def build_methodology_matrix(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for rel in METHODOLOGY_SEARCH_FILES:
        path = root / rel
        if not path.is_file():
            row: dict[str, str] = {"file": rel, "exists": "no"}
            for term in METHODOLOGICAL_PROTECTIONS:
                row[term] = "absent"
            rows.append(row)
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace").lower()
        except Exception:
            text = ""
        row = {"file": rel, "exists": "yes"}
        for term in METHODOLOGICAL_PROTECTIONS:
            row[term] = "present" if term.lower() in text else "absent"
        rows.append(row)
    return rows


def check_docs_coverage(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for rel in REQUIRED_DOCS:
        p = root / rel
        rows.append({"doc": rel, "exists": "yes" if p.is_file() else "no"})
    readme = root / README_PATH
    if readme.is_file():
        try:
            text = readme.read_text(encoding="utf-8", errors="replace")
        except Exception:
            text = ""
        for marker in README_DINO_MARKERS:
            present = marker in text
            rows.append({
                "doc": f"README.md -> {marker}",
                "exists": "yes" if present else "no",
            })
    else:
        rows.append({"doc": "README.md", "exists": "no"})
    return rows


def check_operational_coverage(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for rel in REQUIRED_SCRIPTS + REQUIRED_TESTS:
        p = root / rel
        rows.append({"item": rel, "kind": "file", "exists": "yes" if p.is_file() else "no"})
    registry = root / COMMAND_REGISTRY_PATH
    if registry.is_file():
        try:
            text = registry.read_text(encoding="utf-8", errors="replace")
        except Exception:
            text = ""
        for version in COMMAND_REGISTRY_VERSIONS:
            present = f"## {version}" in text or f"v1gn" in text or version in text
            present = version in text
            rows.append({
                "item": f"{COMMAND_REGISTRY_PATH} mentions {version}",
                "kind": "registry_entry",
                "exists": "yes" if present else "no",
            })
    else:
        rows.append({"item": COMMAND_REGISTRY_PATH, "kind": "registry_file", "exists": "no"})
    return rows


def collect_file_inventory(root: Path, files: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for p in files:
        rel = p.relative_to(root).as_posix()
        rows.append({
            "file": rel,
            "size_bytes": str(p.stat().st_size),
            "extension": p.suffix,
        })
    return rows


def git_tracked_files(root: Path) -> list[str] | None:
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except Exception:
        pass
    return None


def determine_readiness(
    forbidden_issues: list[dict[str, str]],
    private_path_issues: list[dict[str, str]],
    docs_coverage: list[dict[str, str]],
    operational_coverage: list[dict[str, str]],
    methodology_matrix: list[dict[str, str]],
) -> tuple[str, list[str]]:
    blockers: list[str] = []
    review_notes: list[str] = []

    if forbidden_issues:
        for issue in forbidden_issues:
            blockers.append(f"Forbidden artifact outside local_runs/: {issue['file']}")

    if private_path_issues:
        for issue in private_path_issues:
            blockers.append(f"Private path in versionable file: {issue['file']} (pattern: {issue['pattern']})")

    for row in docs_coverage:
        if row.get("exists") == "no":
            doc = row.get("doc", "")
            if doc in REQUIRED_DOCS:
                blockers.append(f"Required doc missing: {doc}")
            elif "README.md ->" in doc:
                blockers.append(f"README.md does not reference: {doc.split(' -> ')[1]}")
            elif doc == "README.md":
                blockers.append("README.md missing")

    for row in operational_coverage:
        if row.get("exists") == "no":
            kind = row.get("kind", "")
            item = row.get("item", "")
            if kind == "file":
                blockers.append(f"Required script/test missing: {item}")
            elif kind == "registry_entry":
                review_notes.append(f"Command registry: {item}")
            elif kind == "registry_file":
                blockers.append(f"Command registry file missing: {item}")

    critical_protections = ["review_only", "supervised_training", "predictive_claims", "multimodal"]
    for protection in critical_protections:
        found_in_any = any(
            row.get(protection) == "present"
            for row in methodology_matrix
        )
        if not found_in_any:
            blockers.append(f"Methodological protection not documented anywhere: {protection}")

    if blockers:
        return "BLOCKED", blockers
    if review_notes:
        return "READY_WITH_REVIEW_NOTES", review_notes
    return "READY_FOR_LOCAL_COMMIT", []


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    prepare(output_dir, args.force)

    ts = datetime.now(timezone.utc).isoformat()

    print("[v1gp] Collecting versionable files...")
    versionable_files = collect_versionable_files(ROOT)

    print("[v1gp] Checking forbidden artifacts outside local_runs/...")
    forbidden_issues = check_forbidden_artifacts(ROOT)

    print("[v1gp] Checking private paths in versionable files...")
    private_path_issues = check_private_paths(ROOT, versionable_files)

    print("[v1gp] Building methodology matrix...")
    methodology_matrix = build_methodology_matrix(ROOT)

    print("[v1gp] Checking documentation coverage...")
    docs_coverage = check_docs_coverage(ROOT)

    print("[v1gp] Checking operational coverage...")
    operational_coverage = check_operational_coverage(ROOT)

    print("[v1gp] Collecting file inventory...")
    file_inventory = collect_file_inventory(ROOT, versionable_files)

    print("[v1gp] Querying git tracked files (optional)...")
    tracked = git_tracked_files(ROOT)
    git_available = tracked is not None

    print("[v1gp] Determining readiness status...")
    readiness_status, notes = determine_readiness(
        forbidden_issues,
        private_path_issues,
        docs_coverage,
        operational_coverage,
        methodology_matrix,
    )

    summary = {
        "version": "v1gp",
        "generated_at": ts,
        "readiness_status": readiness_status,
        "notes": notes,
        "git_available": git_available,
        "versionable_files_count": len(versionable_files),
        "forbidden_issues_count": len(forbidden_issues),
        "private_path_issues_count": len(private_path_issues),
        "docs_coverage_pass": sum(1 for r in docs_coverage if r.get("exists") == "yes"),
        "docs_coverage_total": len(docs_coverage),
        "operational_coverage_pass": sum(1 for r in operational_coverage if r.get("exists") == "yes"),
        "operational_coverage_total": len(operational_coverage),
        "guardrails": {
            "review_only": True,
            "supervised_training": False,
            "labels_created": False,
            "predictive_claims": False,
            "multimodal_hold": True,
        },
    }

    summary_path = output_dir / "release_readiness_summary_v1gp.json"
    write_json(summary_path, summary)
    print(f"[v1gp] Summary: {summary_path}")

    qa_rows = [
        {"check": "forbidden_artifacts", "status": "PASS" if not forbidden_issues else "FAIL", "count": len(forbidden_issues)},
        {"check": "private_paths", "status": "PASS" if not private_path_issues else "FAIL", "count": len(private_path_issues)},
        {"check": "docs_coverage", "status": "PASS" if all(r.get("exists") == "yes" for r in docs_coverage if r.get("doc", "") in REQUIRED_DOCS) else "FAIL", "count": len(docs_coverage)},
        {"check": "operational_coverage", "status": "PASS" if all(r.get("exists") == "yes" for r in operational_coverage if r.get("kind") == "file") else "FAIL", "count": len(operational_coverage)},
        {"check": "methodology_matrix", "status": "PASS" if not any(row.get("review_only") == "absent" and row.get("exists") == "yes" for row in methodology_matrix) else "REVIEW", "count": len(methodology_matrix)},
        {"check": "overall_readiness", "status": readiness_status, "count": len(notes)},
    ]
    write_csv(
        output_dir / "release_readiness_qa_v1gp.csv",
        qa_rows,
        ["check", "status", "count"],
    )

    write_csv(
        output_dir / "release_readiness_files_v1gp.csv",
        file_inventory,
        ["file", "size_bytes", "extension"],
    )

    methodology_fields = ["file", "exists"] + METHODOLOGICAL_PROTECTIONS
    write_csv(
        output_dir / "release_readiness_methodology_matrix_v1gp.csv",
        methodology_matrix,
        methodology_fields,
    )

    blockers_rows: list[dict[str, str]] = []
    for issue in forbidden_issues:
        blockers_rows.append({"category": "forbidden_artifact", "detail": issue["file"], "severity": "FAIL"})
    for issue in private_path_issues:
        blockers_rows.append({"category": "private_path", "detail": issue["file"], "severity": "FAIL"})
    for note in notes:
        blockers_rows.append({"category": "note", "detail": note, "severity": "REVIEW"})
    write_csv(
        output_dir / "release_readiness_blockers_v1gp.csv",
        blockers_rows,
        ["category", "detail", "severity"],
    )

    print(f"\n[v1gp] Readiness status: {readiness_status}")
    if notes:
        for note in notes:
            print(f"  - {note}")
    print(f"[v1gp] Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
