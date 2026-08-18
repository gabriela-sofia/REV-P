"""
SUSC-10D DINO Embedding Discovery (read-only, offline)

Scans the repository for DINO/embedding artifacts and determines whether any
LOADABLE per-patch embedding vectors exist (light, with patch_id). Records every
candidate honestly; if no loadable vectors exist, downstream produces an absence
report (never fails).

Writes:
  - outputs_public/suscetibilidade/SUSC_10D_dino_embedding_discovery.csv
  - outputs_public/suscetibilidade/SUSC_10D_dino_embedding_discovery_summary.json
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_io import read_csv, write_csv, write_json, rel  # noqa: E402

OUT_CSV = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_10D_dino_embedding_discovery.csv"
OUT_SUMMARY = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_10D_dino_embedding_discovery_summary.json"

SCAN_ROOTS = [ROOT / "datasets", ROOT / "outputs_public", ROOT / "manifests",
              ROOT / "scripts", ROOT / "docs"]
PRUNE = {".git", ".claude", "worktrees", "__pycache__", "node_modules", "local_runs",
         "local_only", ".venv", "venv", ".pytest_cache"}
PRUNE_PREFIX = ("figures", "figuras", "tmp_")
KEY = ("dino", "embedding")
VECTOR_HINT_COLS = ("embedding", "vector", "dim_0", "emb_0", "vec_0", "feature_0")
MAX_LIGHT_BYTES = 25 * 1024 * 1024


def _prune(n):
    return n in PRUNE or any(n.startswith(p) for p in PRUNE_PREFIX)


def iter_files():
    seen = set()
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if not _prune(d)]
            for fn in fns:
                p = Path(dp) / fn
                low = fn.lower()
                if not any(k in low for k in KEY) and p.suffix.lower() not in {".npz", ".npy"}:
                    continue
                if p.name.lower().startswith("susc_10d"):
                    continue
                if p in seen:
                    continue
                seen.add(p)
                yield p


def classify(p: Path):
    ext = p.suffix.lower()
    try:
        size = p.stat().st_size
    except OSError:
        size = -1
    if ext in {".npz", ".npy"}:
        return {"type_guess": "binary_embedding", "n_rows": "", "has_patch_id": "false",
                "has_vector_column": "true", "loadable_per_patch_vectors": "false",
                "notes": "binary vectors not loaded (git-ignored/heavy or external); review-only"}
    if ext == ".csv":
        try:
            rows = read_csv(p)
        except Exception:
            rows = []
        cols = set(rows[0].keys()) if rows else set()
        has_pid = "patch_id" in cols
        vec_col = any(c in cols for c in VECTOR_HINT_COLS)
        stats_only = any(c.startswith("vector_") for c in cols) and not vec_col
        # loadable only if a real per-row vector token exists AND there are rows
        loadable = bool(rows and has_pid and vec_col and size <= MAX_LIGHT_BYTES
                        and any(str(r.get("embedding", r.get("vector", ""))).strip() for r in rows))
        if loadable:
            tg = "loadable_per_patch_vectors"
        elif vec_col:
            tg = "vector_column_present_but_empty"
        elif stats_only:
            tg = "vector_stats_registry_only"
        else:
            tg = "dino_metadata_registry"
        return {"type_guess": tg, "n_rows": len(rows), "has_patch_id": str(has_pid).lower(),
                "has_vector_column": str(vec_col).lower(),
                "loadable_per_patch_vectors": str(loadable).lower(),
                "notes": "csv scanned; vectors loadable only with per-row vector + rows + light size"}
    if ext == ".json":
        return {"type_guess": "dino_metadata_json", "n_rows": "", "has_patch_id": "unknown",
                "has_vector_column": "unknown", "loadable_per_patch_vectors": "false",
                "notes": "json metadata, not loaded as vectors"}
    if ext == ".py":
        return {"type_guess": "dino_script", "n_rows": "", "has_patch_id": "false",
                "has_vector_column": "false", "loadable_per_patch_vectors": "false",
                "notes": "code, not data"}
    return {"type_guess": "other", "n_rows": "", "has_patch_id": "false",
            "has_vector_column": "false", "loadable_per_patch_vectors": "false", "notes": ""}


def main() -> int:
    print("=" * 60)
    print("SUSC-10D DINO Embedding Discovery (offline)")
    print("=" * 60)

    rows = []
    for p in sorted(iter_files()):
        info = classify(p)
        info["file_path"] = rel(p)
        try:
            info["size_bytes"] = str(p.stat().st_size)
        except OSError:
            info["size_bytes"] = "-1"
        rows.append(info)

    fields = ["file_path", "type_guess", "size_bytes", "n_rows", "has_patch_id",
              "has_vector_column", "loadable_per_patch_vectors", "notes"]
    write_csv(OUT_CSV, rows, fields)

    loadable = [r for r in rows if r["loadable_per_patch_vectors"] == "true"]
    summary = {
        "stage": "SUSC-10D DINO discovery",
        "files_found": len(rows),
        "type_counts": dict(Counter(r["type_guess"] for r in rows)),
        "loadable_per_patch_vectors_available": bool(loadable),
        "loadable_files": [r["file_path"] for r in loadable],
        "dino_usage": "representational_diagnostic_only_never_classifier",
        "can_be_used_as_ground_truth": False, "allowed_for_training": False, "review_only": True,
        "note": "Heavy DINO vectors (.npz/.npy) are git-ignored / external; only metadata in repo.",
    }
    write_json(OUT_SUMMARY, summary)

    print(f"\nfiles found: {len(rows)} | loadable per-patch vectors: {bool(loadable)}")
    print("types:", summary["type_counts"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
