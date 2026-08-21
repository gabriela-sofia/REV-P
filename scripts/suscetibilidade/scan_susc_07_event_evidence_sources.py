"""
SUSC-07A Event Evidence Source Scanner (read-only, offline)

Scans the repository (no internet) for documentary/event evidence that could be
compared against susceptibility patches. Records, per file, the matched terms, a
region guess, an evidence-type guess, and a candidate status. This is an
inventory step: it does NOT create ground truth and does NOT link events to
patches.

Outputs:
  - outputs_public/suscetibilidade/SUSC_07A_event_source_scan.csv
  - outputs_public/suscetibilidade/SUSC_07A_event_source_scan_summary.json
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())

SCAN_ROOTS = [
    ROOT / "outputs_public",
    ROOT / "docs",
    ROOT / "datasets",
    ROOT / "manifests",
    ROOT / "scripts",
    ROOT / "configs",
]

OUT_CSV = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_07A_event_source_scan.csv"
OUT_SUMMARY = (
    ROOT / "outputs_public" / "suscetibilidade" / "SUSC_07A_event_source_scan_summary.json"
)

PRUNE_DIR_NAMES = {
    ".git", ".claude", "worktrees", "__pycache__", "node_modules",
    "local_runs", "local_only", "patches", ".venv", "venv", ".pytest_cache",
}
PRUNE_DIR_PREFIXES = ("figures", "figuras", "tmp_")
TEXT_EXTS = {".csv", ".json", ".md", ".txt", ".yaml", ".yml"}
MAX_FILE_BYTES = 3_000_000

KEYWORDS = [
    "enchente", "alagamento", "inundação", "inundacao", "flood", "flooding",
    "evento", "Defesa Civil", "CEMADEN", "ANA", "APAC", "CPRM", "SGB",
    "GeoCuritiba", "Petrópolis", "Petropolis", "Recife", "Curitiba",
    "Protocolo C", "protocolo_c", "eventos candidatos", "ocorrência", "ocorrencia",
    "calamidade", "emergência", "emergencia",
]

REGION_HINTS = {
    "recife": ["Recife", "REC", "Pernambuco", "APAC", "Olinda"],
    "petropolis": ["Petrópolis", "Petropolis", "PET", "Serrana"],
    "curitiba": ["Curitiba", "CUR", "GeoCuritiba", "Paraná", "Parana"],
}

EVENT_TYPE_HINTS = {
    "flood": ["enchente", "alagamento", "inundação", "inundacao", "flood", "flooding"],
    "mass_movement": ["deslizamento", "movimento de massa", "landslide", "escorregamento"],
    "disaster_declaration": ["calamidade", "emergência", "emergencia", "decreto"],
}

DATE_RE = re.compile(r"\b(19|20)\d{2}[-/.]?(0?[1-9]|1[0-2])?[-/.]?(0?[1-9]|[12]\d|3[01])?\b")
GEOM_RE = re.compile(
    r"\b(bbox|geometry|geometria|polygon|poligono|lat(itude)?|lon(gitude)?|coord|wkt|geojson|epsg)\b",
    re.IGNORECASE,
)


def _prune(name: str) -> bool:
    return name in PRUNE_DIR_NAMES or any(name.startswith(p) for p in PRUNE_DIR_PREFIXES)


def iter_files():
    seen = set()
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if not _prune(d)]
            for fn in fns:
                p = Path(dp) / fn
                if p.suffix.lower() not in TEXT_EXTS:
                    continue
                # avoid self-reference to SUSC-07 outputs / catalog
                if p.name.lower().startswith(("susc_07", "susc_07a")):
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


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT)).replace("\\", "/")


def classify_status(matched, region_guess, has_date, has_geom, path):
    rl = path.lower()
    flood_terms = {"enchente", "alagamento", "inundação", "inundacao", "flood", "flooding"}
    event_terms = flood_terms | {"evento", "ocorrência", "ocorrencia", "calamidade",
                                  "emergência", "emergencia"}
    has_event_term = any(m in event_terms for m in matched)
    is_methodology = any(k in rl for k in ("schema", "validate", "policy", "ontolog",
                                           "guardrail", "metodolog"))
    if is_methodology and not has_event_term:
        return "methodological_reference"
    if has_event_term and (has_date or has_geom) and region_guess != "unknown":
        return "candidate_evidence_source"
    if has_event_term and region_guess != "unknown":
        return "documentary_context"
    if has_event_term:
        return "needs_manual_review"
    if is_methodology:
        return "methodological_reference"
    return "not_event_evidence"


def main() -> int:
    print("=" * 60)
    print("SUSC-07A Event Evidence Source Scanner (offline)")
    print("=" * 60)

    kw_lower = [(k, k.lower()) for k in KEYWORDS]
    rows = []
    files = list(iter_files())
    for p in files:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        tl = text.lower()
        matched = [k for k, kl in kw_lower if kl in tl]
        if not matched:
            continue

        region_guess = "unknown"
        for reg, hints in REGION_HINTS.items():
            if any(h.lower() in tl for h in hints):
                region_guess = reg if region_guess == "unknown" else "multi"
        et_guess = "unknown"
        for et, hints in EVENT_TYPE_HINTS.items():
            if any(h in tl for h in hints):
                et_guess = et
                break
        has_date = bool(DATE_RE.search(text))
        has_geom = bool(GEOM_RE.search(text))
        status = classify_status(matched, region_guess, has_date, has_geom, rel(p))

        rows.append({
            "file_path": rel(p),
            "matched_terms": ";".join(sorted(set(matched))),
            "region_guess": region_guess,
            "evidence_type_guess": et_guess,
            "has_date_guess": str(has_date).lower(),
            "has_geometry_guess": str(has_geom).lower(),
            "candidate_status": status,
            "notes": "offline keyword scan; review-only; not ground truth",
        })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    import csv
    fields = ["file_path", "matched_terms", "region_guess", "evidence_type_guess",
              "has_date_guess", "has_geometry_guess", "candidate_status", "notes"]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    summary = {
        "schema_id": "SUSC-07A-SCAN",
        "files_scanned": len(files),
        "files_with_matches": len(rows),
        "status_counts": dict(Counter(r["candidate_status"] for r in rows)),
        "region_counts": dict(Counter(r["region_guess"] for r in rows)),
        "governance": {"allowed_for_training": False, "review_only": True,
                       "can_create_ground_truth": False},
        "note": "Inventory only. Evidence != occurrence per patch != ground truth.",
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\nfiles scanned: {len(files)} | with matches: {len(rows)}")
    print("status:", summary["status_counts"])
    print("region:", summary["region_counts"])
    print("\nInventory only. No ground truth. review-only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
