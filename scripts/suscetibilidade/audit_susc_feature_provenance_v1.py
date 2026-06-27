"""
SUSC-04 Feature Provenance Audit (read-only scanner)

Scans REV-P and PROJETO source trees for evidence of WHERE each strong
susceptibility feature came from, HOW it was computed, WHICH public source or
script supports it, its unit, expected scientific direction, and the confidence
status it should receive.

The scanner is bounded, precise, and read-only:
  - prunes heavy directories (figures, archives, worktrees, caches, raw outputs);
  - reads only text/code/config files under a size cap; never opens binaries;
  - EXCLUDES the SUSC layer itself from evidence (provenance must come from the
    upstream pipeline, not from our own derived artifacts);
  - matches feature/source tokens on WORD BOUNDARIES (no substring false hits);
  - claims script evidence only when the EXACT column name (or a known spec
    function) appears in a non-SUSC .py file.

Outputs:
  - manifests/suscetibilidade/susc_feature_provenance_scan_v1.json   (raw scan, Etapa 2)
  - manifests/suscetibilidade/susc_feature_provenance_audit_v1.csv   (audit, Etapa 3)
  - outputs_public/suscetibilidade/SUSC_04_score_v6_feature_decision_matrix.csv (Etapa 5)

Governance: every feature stays can_be_ground_truth=false, allowed_for_training=false,
review_only=true. The audit does NOT train a model, does NOT create ground truth,
does NOT compute score v6. If provenance is not found it is marked 'unresolved'
(never invented).
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJETO = ROOT.parent / "PROJETO"

DIRECTION_PATH = (
    ROOT / "schemas" / "suscetibilidade" / "susc_feature_scientific_direction_v1.json"
)
SCHEMA_PATH = ROOT / "schemas" / "suscetibilidade" / "susc_features_schema_v1.json"
MANIFEST_PATH = (
    ROOT / "manifests" / "suscetibilidade" / "susc_features_provenance_manifest_v1.csv"
)
CSV_PATH = ROOT / "datasets" / "suscetibilidade" / "susc_features_by_patch_v1.csv"

SCAN_JSON = ROOT / "manifests" / "suscetibilidade" / "susc_feature_provenance_scan_v1.json"
AUDIT_CSV = ROOT / "manifests" / "suscetibilidade" / "susc_feature_provenance_audit_v1.csv"
DECISION_CSV = (
    ROOT / "outputs_public" / "suscetibilidade" / "SUSC_04_score_v6_feature_decision_matrix.csv"
)

# --- scan configuration (bounded) ---
SCAN_ROOTS = [
    ROOT / "scripts",
    ROOT / "datasets",
    ROOT / "manifests",
    ROOT / "outputs_public",
    ROOT / "configs",
    ROOT / "docs",
    PROJETO / "src",
    PROJETO / "scripts",
    PROJETO / "docs",
    PROJETO / "data",
]

PRUNE_DIR_NAMES = {
    ".git", ".claude", "worktrees", "__pycache__", "node_modules",
    "local_runs", "local_only", "outputs", "patches", "archive_drive",
    ".venv", "venv", ".pytest_cache", "tmp", "suscetibilidade",
}
PRUNE_DIR_PREFIXES = ("figures", "figuras", "tmp_")
TEXT_EXTS = {".py", ".md", ".csv", ".json", ".txt", ".yaml", ".yml", ".cfg", ".ini", ".toml"}
MAX_FILE_BYTES = 3_000_000

# Specific public source tokens (word-boundary matched, case-insensitive).
PUBLIC_SOURCE_TOKENS = [
    "MapBiomas", "CHIRPS", "JRC", "SRTM", "Topodata", "PE3D", "ANA", "BHO",
    "Sentinel", "GEE", "MERIT", "Copernicus", "GlobalSurfaceWater",
    "GeoCuritiba", "earthengine", "MDE", "MDT",
]

# Known in-repo spec/computation function or module hints per feature. Presence of
# any of these (word-boundary) in a non-SUSC .py counts as script evidence even if
# the exact aggregated column name is absent.
SPEC_HINTS = {
    "ndvi_mean": ["ndvi_spec"],
    "mndwi_mean": ["mndwi_spec"],
    "ndbi_mean": ["ndbi_spec"],
    "s1_vv_mean_clean": ["features_sar", "s1_vv"],
    "s1_vh_mean_clean": ["features_sar", "s1_vh"],
    "s1_vv_minus_vh_mean_clean": ["s1_vv_minus_vh"],
    "distance_to_water_mean": ["drainage_distance"],
    "flow_acc_log_p75": ["flow_acc_log_p75"],
}

EPSG_RE = re.compile(r"EPSG:\s*\d{4,6}", re.IGNORECASE)
RES_RE = re.compile(r"\b(10|20|30|90|250|500)\s?m\b")
UNIT_RE = re.compile(r"\b(mm|dB|degrees|meters|metros|fraction)\b")
DEF_RE = re.compile(r"def\s+(\w+)\s*\(")
TEMPORAL_RE = re.compile(r"reference_date|window|multitemporal|cubo|2022")

MONOTONIC = {"higher_increases", "lower_increases", "higher_decreases", "lower_decreases"}
PHYSICAL_GROUPS = {
    "topography", "topography_hydrology", "hydrology", "land_use",
    "spectral_index", "precipitation",
}
DINO_GROUPS = PHYSICAL_GROUPS | {"sar"}
SPGAM_GROUPS = {
    "topography", "topography_hydrology", "hydrology", "land_use",
    "spectral_index", "precipitation",
}
VERIFIED = {"verified_script_and_source", "verified_script_only", "verified_dataset_only"}


def _should_prune(dirname: str) -> bool:
    if dirname in PRUNE_DIR_NAMES:
        return True
    return any(dirname.startswith(p) for p in PRUNE_DIR_PREFIXES)


def iter_text_files():
    seen: set[Path] = set()
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not _should_prune(d)]
            for fn in filenames:
                p = Path(dirpath) / fn
                if p.suffix.lower() not in TEXT_EXTS:
                    continue
                if "suscetibilidade" in p.parts:
                    continue
                if p in seen:
                    continue
                try:
                    if p.stat().st_size > MAX_FILE_BYTES:
                        continue
                except OSError:
                    continue
                seen.add(p)
                yield p


def build_scan_index():
    contents: dict[Path, str] = {}
    for p in iter_text_files():
        try:
            contents[p] = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
    return contents


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return "PROJETO/" + str(p.relative_to(PROJETO)).replace("\\", "/")


def _word_re(token: str) -> re.Pattern:
    return re.compile(r"(?<![A-Za-z0-9_])" + re.escape(token) + r"(?![A-Za-z0-9_])")


def scan_feature(feature: str, contents: dict[Path, str]) -> dict:
    """Precise per-feature evidence scan.

    Script evidence requires the EXACT column name or a known spec hint (word
    boundary) in a non-SUSC .py file. Public-source evidence is tied to those
    computation files.
    """
    script_tokens = [feature] + SPEC_HINTS.get(feature, [])
    script_res = [_word_re(t) for t in script_tokens]
    exact_re = _word_re(feature)

    script_paths: list[str] = []
    dataset_paths: list[str] = []
    doc_paths: list[str] = []
    func_names: set[str] = set()
    public_sources_in_script: set[str] = set()
    public_sources_colocated: set[str] = set()
    public_sources_documented: set[str] = set()
    source_paths: list[str] = []
    units: set[str] = set()
    crs: set[str] = set()
    resolutions: set[str] = set()
    temporal = False

    src_res = [(s, _word_re(s)) for s in PUBLIC_SOURCE_TOKENS]
    proximity = 5

    for p, text in contents.items():
        ext = p.suffix.lower()
        r = rel(p)

        is_script_hit = ext == ".py" and any(rx.search(text) for rx in script_res)
        is_exact_hit = bool(exact_re.search(text))

        if is_script_hit:
            script_paths.append(r)
            lines = text.splitlines()
            hit_lines = [
                i for i, ln in enumerate(lines)
                if any(rx.search(ln) for rx in script_res)
            ]
            for m in DEF_RE.finditer(text):
                name = m.group(1)
                if feature in name or any(h in name for h in SPEC_HINTS.get(feature, [])):
                    func_names.add(f"{name} @ {r}")
            tied_here = False
            for s, rx in src_res:
                if not rx.search(text):
                    continue
                # source counts as TIED only if near a feature-token line
                near = any(
                    any(rx.search(lines[j]) for j in range(max(0, i - proximity),
                        min(len(lines), i + proximity + 1)))
                    for i in hit_lines
                )
                if near:
                    public_sources_in_script.add(s)
                    tied_here = True
                else:
                    public_sources_colocated.add(s)
            if tied_here:
                source_paths.append(r)
            for m in EPSG_RE.findall(text):
                crs.add(m.replace(" ", ""))
            for m in RES_RE.findall(text):
                resolutions.add(f"{m}m")
            for m in UNIT_RE.findall(text):
                units.add(m)
            if TEMPORAL_RE.search(text):
                temporal = True
        elif is_exact_hit:
            if ext in {".csv", ".json"}:
                dataset_paths.append(r)
            elif ext == ".md":
                doc_paths.append(r)
                for s, rx in src_res:
                    if rx.search(text):
                        public_sources_documented.add(s)

    def uniq(seq):
        out = []
        for x in seq:
            if x not in out:
                out.append(x)
        return out

    return {
        "feature_name": feature,
        "script_search_tokens": script_tokens,
        "found_in_scripts": bool(script_paths),
        "script_paths": uniq(script_paths)[:12],
        "function_names_if_detected": sorted(func_names)[:12],
        "source_dataset_paths": uniq(dataset_paths)[:12],
        "doc_paths": uniq(doc_paths)[:8],
        "public_source_in_script": sorted(public_sources_in_script),
        "public_source_colocated_unverified": sorted(public_sources_colocated),
        "public_source_documented_only": sorted(public_sources_documented),
        "source_paths": uniq(source_paths)[:8],
        "unit_detected": sorted(units),
        "crs_detected_if_any": sorted(crs),
        "resolution_detected_if_any": sorted(resolutions),
        "temporal_reference_detected": temporal,
    }


def observed_stats(col_values: list[str]):
    vals = []
    missing = 0
    for v in col_values:
        s = v.strip()
        if s == "" or s.lower() in {"na", "nan", "none", "null"}:
            missing += 1
            continue
        try:
            vals.append(float(s))
        except ValueError:
            pass
    if not vals:
        return None, None, None, missing, False
    return min(vals), max(vals), sum(vals) / len(vals), missing, True


def provenance_status(scan: dict, exists: bool, in_schema: bool, in_manifest: bool) -> str:
    has_script = scan["found_in_scripts"]
    has_source = bool(scan["public_source_in_script"])
    documented = bool(scan["doc_paths"]) or bool(scan["public_source_documented_only"])
    if has_script and has_source:
        return "verified_script_and_source"
    if has_script:
        return "verified_script_only"
    if exists and (in_schema or in_manifest):
        return "verified_dataset_only"
    if documented:
        return "documented_only"
    return "unresolved"


def decide(meta, status, exists, numeric):
    direction = meta["expected_direction_for_flood_susceptibility"]
    group = meta["scientific_group"]
    verified = status in VERIFIED
    monotonic = direction in MONOTONIC

    safe_v6 = bool(
        exists and verified and monotonic
        and group in (PHYSICAL_GROUPS | {"sar"})
        and group != "interaction"
    )
    safe_spgam = bool(
        exists and numeric and monotonic and group in SPGAM_GROUPS and verified
    )
    safe_dino = bool(exists and numeric and group in DINO_GROUPS)

    role_map = {
        "topography": "physical_core",
        "topography_hydrology": "physical_core",
        "hydrology": "hydrological_core",
        "land_use": "urbanization_core",
        "interaction": "urbanization_core",
        "spectral_index": "spectral_support",
        "sar": "sar_support",
        "precipitation": "rainfall_trigger",
    }
    role = role_map.get(group, "proxy_only")
    if not exists or status == "unresolved":
        role = "exclude_until_audited"

    if not exists or status == "unresolved":
        weight = "blocked"
    elif direction == "not_scored":
        weight = "not_weighted"
    elif direction in {"ambiguous", "non_monotonic"}:
        weight = "low"
    elif group in {"topography_hydrology", "hydrology"} and verified:
        weight = "high"
    elif group in {"topography", "land_use", "spectral_index"} and verified:
        weight = "high" if safe_v6 else "medium"
    else:
        weight = "medium"

    blocker = ""
    if not exists:
        blocker = "ABSENT_FROM_SUSC03"
    elif status == "unresolved":
        blocker = "UNRESOLVED_PROVENANCE"
    elif group == "interaction":
        blocker = "COMPOSITE_V5_INTERACTION_AUDIT_FORMULA_FIRST"
    elif direction in {"ambiguous", "non_monotonic"}:
        blocker = "AMBIGUOUS_DIRECTION"
    elif direction == "not_scored":
        blocker = "NOT_SCORED"
    elif not verified:
        blocker = "PROVENANCE_NOT_VERIFIED"
    elif not safe_v6:
        blocker = "DIRECTION_OR_GROUP_NOT_ELIGIBLE"

    return {
        "safe_for_score_v6": safe_v6,
        "safe_for_spgam_baseline": safe_spgam,
        "safe_for_dino_comparison": safe_dino,
        "role": role,
        "weight_class": weight,
        "blocker": blocker,
    }


def main() -> int:
    print("=" * 60)
    print("SUSC-04 Feature Provenance Audit (read-only scanner)")
    print("=" * 60)

    for p in (DIRECTION_PATH, SCHEMA_PATH, MANIFEST_PATH, CSV_PATH):
        if not p.exists():
            print(f"STOP: required input not found: {p}")
            return 1

    with DIRECTION_PATH.open("r", encoding="utf-8") as f:
        direction = json.load(f)
    dir_feats = {f["feature_name"]: f for f in direction["features"]}

    with SCHEMA_PATH.open("r", encoding="utf-8") as f:
        schema = json.load(f)
    schema_cols = {f["source_column"]: f for f in schema["features"]}

    manifest_cols: set[str] = set()
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            manifest_cols.add(r["source_column"])

    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        data = list(reader)
    col_index = {c: i for i, c in enumerate(header)}

    print("\n[1/4] Scanning source trees (bounded, precise)...")
    contents = build_scan_index()
    print(f"  read {len(contents)} text/code/config files (SUSC layer excluded)")

    scans = {}
    audit_rows = []
    decision_rows = []

    print("[2/4] Auditing features...")
    for feature, meta in dir_feats.items():
        scan = scan_feature(feature, contents)
        scans[feature] = scan

        exists = feature in col_index
        in_schema = feature in schema_cols
        in_manifest = feature in manifest_cols

        if exists:
            vals = [data[r][col_index[feature]] for r in range(len(data))]
            omin, omax, omean, missing, numeric = observed_stats(vals)
        else:
            omin = omax = omean = None
            missing = ""
            numeric = False

        status = provenance_status(scan, exists, in_schema, in_manifest)
        dec = decide(meta, status, exists, numeric)

        # Manual review is required when provenance is weak OR when the source
        # attribution is ambiguous (more than one candidate source detected near
        # the feature). The scan surfaces candidate sources; a human confirms the
        # definitive one before any scoring use.
        requires_manual = (
            status in {"unresolved", "documented_only", "conflicting_sources"}
            or not scan["public_source_in_script"]
            or len(scan["public_source_in_script"]) > 1
        )

        public_source = ";".join(scan["public_source_in_script"]) or (
            "documented:" + ";".join(scan["public_source_documented_only"])
            if scan["public_source_documented_only"] else "unresolved"
        )
        crs_res = ";".join(scan["crs_detected_if_any"] + scan["resolution_detected_if_any"]) or "unresolved"
        temporal_ref = "detected" if scan["temporal_reference_detected"] else "unresolved"

        audit_rows.append({
            "feature_name": feature,
            "feature_group": meta["scientific_group"],
            "source_column": feature if exists else "NOT_IN_SUSC03",
            "exists_in_susc03": str(exists).lower(),
            "scientific_direction": meta["expected_direction_for_flood_susceptibility"],
            "expected_unit": meta["unit_expected"],
            "observed_min": "" if omin is None else round(omin, 6),
            "observed_max": "" if omax is None else round(omax, 6),
            "observed_mean": "" if omean is None else round(omean, 6),
            "missing_count": missing,
            "provenance_status": status,
            "script_found": str(scan["found_in_scripts"]).lower(),
            "script_paths": ";".join(scan["script_paths"]) or "",
            "source_found": str(bool(scan["public_source_in_script"])).lower(),
            "source_paths": ";".join(scan["source_paths"]) or "",
            "public_source": public_source,
            "crs_or_resolution": crs_res,
            "temporal_reference": temporal_ref,
            "requires_manual_review": str(requires_manual).lower(),
            "safe_for_score_v6": str(dec["safe_for_score_v6"]).lower(),
            "safe_for_spgam_baseline": str(dec["safe_for_spgam_baseline"]).lower(),
            "safe_for_dino_comparison": str(dec["safe_for_dino_comparison"]).lower(),
            "can_be_ground_truth": "false",
            "allowed_for_training": "false",
            "review_only": "true",
            "audit_notes": meta.get("notes", ""),
        })

        decision_rows.append({
            "feature_name": feature,
            "include_in_score_v6": str(dec["safe_for_score_v6"]).lower(),
            "include_in_spgam_baseline": str(dec["safe_for_spgam_baseline"]).lower(),
            "include_in_dino_comparison": str(dec["safe_for_dino_comparison"]).lower(),
            "role": dec["role"],
            "direction": meta["expected_direction_for_flood_susceptibility"],
            "weight_class": dec["weight_class"],
            "reason": meta["expected_reason"],
            "blocker": dec["blocker"],
        })

    print("[3/4] Writing scan JSON + audit CSV...")
    SCAN_JSON.parent.mkdir(parents=True, exist_ok=True)
    with SCAN_JSON.open("w", encoding="utf-8") as f:
        json.dump({
            "schema_id": "SUSC-04-SCAN",
            "files_scanned": len(contents),
            "governance": {"allowed_for_training": False, "review_only": True,
                           "can_create_ground_truth": False},
            "features": scans,
        }, f, indent=2, ensure_ascii=False)
        f.write("\n")

    with AUDIT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(audit_rows)

    print("[4/4] Writing score v6 decision matrix...")
    DECISION_CSV.parent.mkdir(parents=True, exist_ok=True)
    with DECISION_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(decision_rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(decision_rows)

    status_counts = Counter(r["provenance_status"] for r in audit_rows)
    v6 = sum(1 for r in audit_rows if r["safe_for_score_v6"] == "true")
    spgam = sum(1 for r in audit_rows if r["safe_for_spgam_baseline"] == "true")
    dino = sum(1 for r in audit_rows if r["safe_for_dino_comparison"] == "true")
    print("\nProvenance status:")
    for k, v in sorted(status_counts.items()):
        print(f"  {k}: {v}")
    print(f"safe_for_score_v6: {v6}/{len(audit_rows)}")
    print(f"safe_for_spgam_baseline: {spgam}/{len(audit_rows)}")
    print(f"safe_for_dino_comparison: {dino}/{len(audit_rows)}")
    print("\nAudit complete. No ground truth created. score v6 NOT computed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
