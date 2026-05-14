from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PHASE = "v1fv"
PHASE_NAME = "DINO_LOCAL_ASSET_PREFLIGHT"

DEFAULT_INPUT_MANIFEST = ROOT / "manifests" / "dino_inputs" / "revp_v1fu_dino_sentinel_input_manifest" / "dino_sentinel_input_manifest_v1fu.csv"
DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "dino_asset_preflight" / "v1fv"

PREFLIGHT_CSV = "dino_local_asset_preflight_v1fv.csv"
SUMMARY_JSON = "dino_local_asset_preflight_summary_v1fv.json"
QA_CSV = "dino_local_asset_preflight_qa_v1fv.csv"

REQUIRED_INPUT_COLUMNS = [
    "dino_input_id",
    "canonical_patch_id",
    "region",
    "asset_path_reference",
    "source_manifest",
    "modality",
    "eligibility_status",
]

OUTPUT_FIELDS = [
    "dino_input_id",
    "canonical_patch_id",
    "region",
    "asset_path_reference",
    "resolved_status",
    "resolved_path_private",
    "candidate_count",
    "file_extension",
    "future_pixel_read_allowed",
    "pixel_read_status",
    "embedding_status",
    "notes",
]

FORBIDDEN_REPO_DIRS = {"data", "outputs", "docs"}
FORBIDDEN_HEAVY_EXTENSIONS = {
    ".npy",
    ".npz",
    ".parquet",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".index",
}
FUTURE_RASTER_EXTENSIONS = {".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="REV-P v1fv DINO local asset preflight.")
    parser.add_argument("--private-project-root", required=True, help="Private PROJETO root used only for local path resolution.")
    parser.add_argument("--input-manifest", default=str(DEFAULT_INPUT_MANIFEST), help="v1fu DINO Sentinel input manifest.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Local non-versioned output directory.")
    parser.add_argument("--force", action="store_true", help="Replace output directory if it already exists.")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


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


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def is_local_runs_ignored() -> bool:
    gitignore = ROOT / ".gitignore"
    if not gitignore.exists():
        return False
    lines = [line.strip() for line in gitignore.read_text(encoding="utf-8").splitlines()]
    return "local_runs/" in lines or "local_runs" in lines


def forbidden_repo_artifacts() -> list[str]:
    found: list[str] = []
    for path in ROOT.rglob("*"):
        if ".git" in path.parts or "local_runs" in path.parts:
            continue
        if path.is_dir() and path.name in FORBIDDEN_REPO_DIRS:
            found.append(rel(path))
        elif path.is_file() and path.suffix.lower() in FORBIDDEN_HEAVY_EXTENSIONS:
            found.append(rel(path))
    return sorted(found)


def find_by_basename(private_root: Path, basename: str) -> list[Path]:
    if not basename:
        return []
    matches: list[Path] = []
    for candidate in private_root.rglob(basename):
        if candidate.is_file():
            matches.append(candidate.resolve())
    return sorted(matches, key=lambda p: str(p).lower())


def resolve_reference(asset_path_reference: str, private_root: Path) -> tuple[str, list[Path], str]:
    reference = asset_path_reference.strip()
    if not reference:
        return "INVALID_REFERENCE", [], "empty asset_path_reference"

    ref_path = Path(reference)
    candidates: list[Path] = []
    notes: list[str] = []

    if ref_path.is_absolute():
        if ref_path.exists() and ref_path.is_file():
            return "FOUND", [ref_path.resolve()], "absolute reference exists; file not opened"
        notes.append("absolute reference missing; basename search attempted")
    else:
        rooted = private_root / ref_path
        if rooted.exists() and rooted.is_file():
            return "FOUND", [rooted.resolve()], "relative reference resolved under private root; file not opened"
        notes.append("relative reference missing under private root; basename search attempted")

    basename_matches = find_by_basename(private_root, ref_path.name)
    unique_matches = list(dict.fromkeys(basename_matches))
    if len(unique_matches) == 1:
        return "FOUND", unique_matches, "; ".join(notes + ["one basename candidate found; file not opened"])
    if len(unique_matches) > 1:
        return "AMBIGUOUS", unique_matches, "; ".join(notes + ["multiple basename candidates found; no selection made"])
    return "MISSING", [], "; ".join(notes + ["no basename candidate found"])


def build_preflight(input_rows: list[dict[str, str]], private_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in input_rows:
        reference = row.get("asset_path_reference", "")
        status, candidates, note = resolve_reference(reference, private_root)
        extension = Path(reference).suffix.lower()
        future_pixel_read_allowed = "YES" if status == "FOUND" and extension in FUTURE_RASTER_EXTENSIONS else "NO"
        rows.append(
            {
                "dino_input_id": row.get("dino_input_id", ""),
                "canonical_patch_id": row.get("canonical_patch_id", ""),
                "region": row.get("region", ""),
                "asset_path_reference": reference,
                "resolved_status": status,
                "resolved_path_private": str(candidates[0]) if status == "FOUND" and len(candidates) == 1 else "",
                "candidate_count": str(len(candidates)),
                "file_extension": extension,
                "future_pixel_read_allowed": future_pixel_read_allowed,
                "pixel_read_status": "NOT_READ__PREFLIGHT_ONLY",
                "embedding_status": "NOT_EXTRACTED",
                "notes": note,
            }
        )
    return rows


def no_phase_outputs_under_manifests() -> bool:
    matches = list((ROOT / "manifests").rglob("*v1fv*")) if (ROOT / "manifests").exists() else []
    return len(matches) == 0


def make_qa(
    input_manifest: Path,
    input_rows: list[dict[str, str]],
    output_rows: list[dict[str, str]],
    private_root: Path,
    structural_error: str,
) -> list[dict[str, str]]:
    qa: list[dict[str, str]] = []

    def add(check: str, passed: bool, details: str) -> None:
        qa.append({"check": check, "status": "PASS" if passed else "FAIL", "details": details})

    input_columns = set(input_rows[0].keys()) if input_rows else set()
    add("input manifest exists", input_manifest.exists(), rel(input_manifest))
    add("input manifest has 128 rows", len(input_rows) == 128, f"rows={len(input_rows)}")
    add("input manifest has required columns", all(column in input_columns for column in REQUIRED_INPUT_COLUMNS), ",".join(sorted(input_columns)))
    add("private project root exists", private_root.exists() and private_root.is_dir(), str(private_root))
    add("local_runs/ is gitignored", is_local_runs_ignored(), ".gitignore contains local_runs/")
    add("no outputs written under manifests/ for this phase", no_phase_outputs_under_manifests(), "no manifests/**/*v1fv* found")
    add("no data/, outputs/, docs/ created", not any((ROOT / name).exists() for name in FORBIDDEN_REPO_DIRS), "repo root checked")
    forbidden = forbidden_repo_artifacts()
    add("no forbidden heavy/model artifacts created", not forbidden, "; ".join(forbidden) if forbidden else "none found")
    add("script does not read raster pixels", True, "path resolution uses exists/is_file/rglob basename only")
    add(
        "local output contains pixel_read_status = NOT_READ__PREFLIGHT_ONLY",
        bool(output_rows) and {row.get("pixel_read_status") for row in output_rows} == {"NOT_READ__PREFLIGHT_ONLY"},
        "constant preflight-only status",
    )
    add(
        "label_status/target_status are not promoted",
        "label_status" not in OUTPUT_FIELDS and "target_status" not in OUTPUT_FIELDS,
        "output schema excludes label and target fields",
    )
    add("no structural error", structural_error == "", structural_error or "none")
    return qa


def prepare_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Re-run with --force to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def run(private_root: Path, input_manifest: Path, output_dir: Path, force: bool) -> int:
    prepare_output_dir(output_dir, force)

    structural_error = ""
    input_rows: list[dict[str, str]] = []
    output_rows: list[dict[str, str]] = []
    if not input_manifest.exists():
        structural_error = f"input manifest not found: {input_manifest}"
    else:
        input_rows = read_csv(input_manifest)

    if not private_root.exists() or not private_root.is_dir():
        structural_error = structural_error or f"private project root not found: {private_root}"
    elif input_rows:
        missing_columns = [column for column in REQUIRED_INPUT_COLUMNS if column not in input_rows[0]]
        if missing_columns:
            structural_error = f"input manifest missing columns: {', '.join(missing_columns)}"
        else:
            output_rows = build_preflight(input_rows, private_root)

    counts = Counter(row.get("resolved_status", "INVALID_REFERENCE") for row in output_rows)
    qa_rows = make_qa(input_manifest, input_rows, output_rows, private_root, structural_error)
    qa_status = "PASS" if all(row["status"] == "PASS" for row in qa_rows) else "FAIL"
    summary = {
        "phase": PHASE,
        "phase_name": PHASE_NAME,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_manifest": rel(input_manifest),
        "total_inputs": len(input_rows),
        "found_count": counts.get("FOUND", 0),
        "missing_count": counts.get("MISSING", 0),
        "ambiguous_count": counts.get("AMBIGUOUS", 0),
        "invalid_reference_count": counts.get("INVALID_REFERENCE", 0),
        "private_paths_redacted_from_git": True,
        "pixel_read": False,
        "embeddings_extracted": False,
        "ready_for_v1fw": counts.get("FOUND", 0) > 0 and structural_error == "",
        "qa_status": qa_status,
        "structural_error": structural_error,
    }

    write_csv(output_dir / PREFLIGHT_CSV, output_rows, OUTPUT_FIELDS)
    write_json(output_dir / SUMMARY_JSON, summary)
    write_csv(output_dir / QA_CSV, qa_rows, ["check", "status", "details"])

    if structural_error:
        print(structural_error, file=sys.stderr)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if qa_status == "PASS" else 2


def main() -> int:
    args = parse_args()
    return run(
        private_root=Path(args.private_project_root),
        input_manifest=Path(args.input_manifest),
        output_dir=Path(args.output_dir),
        force=args.force,
    )


if __name__ == "__main__":
    raise SystemExit(main())
