"""
SUSC-03 Migration Script

Migrates a light, governed, auditable subset of the operational matrix
PROJETO/data/dataset_final.csv into REV-P, selecting ONLY the columns
catalogued in the SUSC-01 schema and provenance manifest.

Governance guarantees enforced here (fail-closed):
  - no column with allowed_for_training=true is migrated;
  - no column with can_be_used_as_ground_truth=true is migrated;
  - heuristic labels and v5 scores are migrated as documented heuristics,
    NEVER as ground truth;
  - the original string tokens are preserved verbatim (no float reformatting),
    keeping the migration faithful and auditable.

This script does NOT delete anything, does NOT move raw data, does NOT
unlock supervised training, and does NOT create ground truth.

The SUSC-03 matrix is a review-only tabular artifact of attributes
associated with urban flood susceptibility. It does NOT constitute
occurrence ground truth.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
PROJETO_DATASET = ROOT.parent / "PROJETO" / "data" / "dataset_final.csv"

SCHEMA_PATH = ROOT / "schemas" / "suscetibilidade" / "susc_features_schema_v1.json"
MANIFEST_PATH = (
    ROOT / "manifests" / "suscetibilidade" / "susc_features_provenance_manifest_v1.csv"
)

OUT_CSV = ROOT / "datasets" / "suscetibilidade" / "susc_features_by_patch_v1.csv"
OUT_MANIFEST = (
    ROOT
    / "manifests"
    / "suscetibilidade"
    / "susc_features_by_patch_v1_artifact_manifest.json"
)

EXPECTED_ROWS = 300

# Migration is relative to REV-P root; reported with forward slashes for portability.
REL_SOURCE = "PROJETO/data/dataset_final.csv"
REL_SCHEMA = "schemas/suscetibilidade/susc_features_schema_v1.json"
REL_MANIFEST = "manifests/suscetibilidade/susc_features_provenance_manifest_v1.csv"
REL_ARTIFACT = "datasets/suscetibilidade/susc_features_by_patch_v1.csv"


def _fail(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"  STOP: {msg}")
    print("\nMigration aborted (fail-closed). No artifact written.")
    sys.exit(1)


def load_schema() -> dict:
    if not SCHEMA_PATH.exists():
        _fail(f"schema not found: {SCHEMA_PATH}")
    with SCHEMA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_manifest_columns() -> set[str]:
    if not MANIFEST_PATH.exists():
        _fail(f"provenance manifest not found: {MANIFEST_PATH}")
    cols: set[str] = set()
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cols.add(row["source_column"])
    return cols


def select_columns(schema: dict, manifest_cols: set[str]) -> list[dict]:
    """Return the ordered list of catalogued feature records to migrate.

    Fail-closed on any feature flagged for training or ground truth, or any
    feature absent from the provenance manifest.
    """
    selected: list[dict] = []
    for feat in schema.get("features", []):
        name = feat["name"]
        src = feat["source_column"]

        if feat.get("allowed_for_training") is True:
            _fail(f"feature '{name}' has allowed_for_training=true")
        if feat.get("can_be_used_as_ground_truth") is True:
            _fail(f"feature '{name}' has can_be_used_as_ground_truth=true")
        if src not in manifest_cols:
            _fail(f"feature '{name}' source_column '{src}' missing from manifest")

        selected.append(feat)
    return selected


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    print("=" * 60)
    print("SUSC-03 Dataset Migration (review-only, fail-closed)")
    print("=" * 60)

    if not PROJETO_DATASET.exists():
        _fail(f"source dataset not found: {PROJETO_DATASET}")

    schema = load_schema()
    manifest_cols = load_manifest_columns()
    selected = select_columns(schema, manifest_cols)
    selected_cols = [f["source_column"] for f in selected]

    print(f"\n[1/4] Catalogued columns selected: {len(selected_cols)}")

    # Read source preserving original string tokens (no float reformatting).
    with PROJETO_DATASET.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        header_index = {name: i for i, name in enumerate(header)}

        missing = [c for c in selected_cols if c not in header_index]
        if missing:
            _fail(f"source dataset missing catalogued columns: {missing}")

        col_idx = [header_index[c] for c in selected_cols]
        rows: list[list[str]] = []
        for raw in reader:
            rows.append([raw[i] for i in col_idx])

    print(f"[2/4] Rows read from source: {len(rows)}")
    if len(rows) != EXPECTED_ROWS:
        _fail(f"expected {EXPECTED_ROWS} rows, found {len(rows)}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(selected_cols)
        writer.writerows(rows)

    digest = sha256_file(OUT_CSV)
    print(f"[3/4] Wrote {REL_ARTIFACT} ({len(rows)} rows x {len(selected_cols)} cols)")
    print(f"        sha256: {digest}")

    group_counts: dict[str, int] = {}
    for feat in selected:
        group_counts[feat["group"]] = group_counts.get(feat["group"], 0) + 1

    artifact = {
        "artifact": REL_ARTIFACT,
        "source": REL_SOURCE,
        "schema": REL_SCHEMA,
        "provenance_manifest": REL_MANIFEST,
        "rows": len(rows),
        "columns": len(selected_cols),
        "sha256": digest,
        "allowed_for_training": False,
        "review_only": True,
        "can_be_used_as_ground_truth": False,
        "scientific_status": "susceptibility_features_not_event_ground_truth",
        "scientific_disclaimer": (
            "A matriz SUSC-03 e um artefato tabular review-only de atributos "
            "associados a suscetibilidade urbana a enchentes. Ela nao constitui "
            "ground truth de ocorrencia, nao desbloqueia treinamento "
            "supervisionado e nao autoriza afirmacoes de evento observado por patch."
        ),
        "column_list": selected_cols,
        "group_counts": group_counts,
    }

    with OUT_MANIFEST.open("w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"[4/4] Wrote artifact manifest: {OUT_MANIFEST.relative_to(ROOT)}")
    print("\nMigration complete. Susceptibility features != confirmed occurrence.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
