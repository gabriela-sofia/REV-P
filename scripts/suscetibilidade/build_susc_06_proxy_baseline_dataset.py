"""
SUSC-06A Proxy Baseline Dataset Builder

Assembles a light, review-only dataset for the interpretable GAM/SPGAM-style
proxy baseline. Selects only feature cards with status ready_for_methodology
(plus an optional usable_with_caution group), excludes proxy_only /
blocked_until_recomputed / unresolved, and carries the available proxy TARGETS
(score / heuristic_label / proxy_v5) WITHOUT ever treating them as ground truth.

Reads:
  - datasets/suscetibilidade/susc_features_by_patch_v1.csv
  - outputs_public/suscetibilidade/feature_cards/susc_feature_cards_v1.json
  - outputs_public/suscetibilidade/feature_cards/SUSC_05_feature_cards_summary.csv
  - outputs_public/suscetibilidade/SUSC_04_score_v6_feature_decision_matrix.csv
  - schemas/suscetibilidade/susc_06_proxy_baseline_schema_v1.json

Writes:
  - datasets/suscetibilidade/susc_06_proxy_baseline_dataset_v1.csv
  - manifests/suscetibilidade/susc_06_proxy_baseline_dataset_manifest_v1.json

Governance: allowed_for_training=false, can_be_used_as_ground_truth=false,
review_only=true. The source matrix is NEVER modified. Proxy targets are NOT
ground truth. Susceptibility != confirmed flood occurrence.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MATRIX = ROOT / "datasets" / "suscetibilidade" / "susc_features_by_patch_v1.csv"
CARDS_JSON = (
    ROOT / "outputs_public" / "suscetibilidade" / "feature_cards" / "susc_feature_cards_v1.json"
)
CARDS_SUMMARY = (
    ROOT / "outputs_public" / "suscetibilidade" / "feature_cards" / "SUSC_05_feature_cards_summary.csv"
)
DECISION = (
    ROOT / "outputs_public" / "suscetibilidade" / "SUSC_04_score_v6_feature_decision_matrix.csv"
)
SCHEMA = ROOT / "schemas" / "suscetibilidade" / "susc_06_proxy_baseline_schema_v1.json"

OUT_CSV = ROOT / "datasets" / "suscetibilidade" / "susc_06_proxy_baseline_dataset_v1.csv"
OUT_MANIFEST = (
    ROOT / "manifests" / "suscetibilidade" / "susc_06_proxy_baseline_dataset_manifest_v1.json"
)

IDENTITY_COLS = ["patch_id", "regiao", "reference_date"]
PROXY_TARGET_GROUPS = {"score", "heuristic_label", "proxy_v5"}


def main() -> int:
    print("=" * 60)
    print("SUSC-06A Proxy Baseline Dataset Builder")
    print("=" * 60)

    for p in (MATRIX, CARDS_JSON, CARDS_SUMMARY, DECISION, SCHEMA):
        if not p.exists():
            print(f"STOP: required input not found: {p}")
            return 1

    with CARDS_JSON.open("r", encoding="utf-8") as f:
        cards = json.load(f)["cards"]

    ready = [c["feature_name"] for c in cards if c["status"] == "ready_for_methodology"]
    caution = [c["feature_name"] for c in cards if c["status"] == "usable_with_caution"]
    excluded = [
        c["feature_name"] for c in cards
        if c["status"] in {"proxy_only", "blocked_until_recomputed", "unresolved"}
    ]

    # proxy targets = heuristic cards that exist as columns in the matrix
    with MATRIX.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    present = set(header)
    col_idx = {c: i for i, c in enumerate(header)}

    proxy_targets = [
        c["feature_name"] for c in cards
        if c["feature_group"] in PROXY_TARGET_GROUPS and c["feature_name"] in present
    ]

    # final selection: identity + ready (+ caution) features that exist + proxy targets
    feat_used = [f for f in ready if f in present]
    caution_used = [f for f in caution if f in present]
    keep_cols = (
        [c for c in IDENTITY_COLS if c in present]
        + feat_used
        + caution_used
        + proxy_targets
    )
    # de-dup preserving order
    seen, ordered = set(), []
    for c in keep_cols:
        if c not in seen:
            seen.add(c)
            ordered.append(c)

    idx = [col_idx[c] for c in ordered]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(ordered)
        for r in rows:
            w.writerow([r[i] for i in idx])

    print(f"\n[ok] dataset: {len(rows)} rows x {len(ordered)} cols -> {OUT_CSV.name}")
    print(f"  features (ready): {len(feat_used)}")
    print(f"  features (caution, optional): {len(caution_used)}")
    print(f"  proxy targets available: {len(proxy_targets)}")
    print(f"  excluded (proxy_only/blocked/unresolved): {len(excluded)}")

    manifest = {
        "artifact": "datasets/suscetibilidade/susc_06_proxy_baseline_dataset_v1.csv",
        "source": "datasets/suscetibilidade/susc_features_by_patch_v1.csv",
        "rows": len(rows),
        "columns": len(ordered),
        "identity_columns": [c for c in IDENTITY_COLS if c in present],
        "features_used": feat_used,
        "optional_features_caution": caution_used,
        "optional_features_excluded": excluded,
        "proxy_targets_available": proxy_targets,
        "ground_truth_targets": [],
        "allowed_for_training": False,
        "can_be_used_as_ground_truth": False,
        "review_only": True,
        "scientific_status": "proxy_baseline_dataset_not_event_ground_truth",
        "scientific_disclaimer": (
            "Dataset review-only para baseline interpretavel ajustado contra proxies "
            "internos. NAO e ground truth de ocorrencia; nao desbloqueia treino."
        ),
    }
    with OUT_MANIFEST.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"[ok] manifest -> {OUT_MANIFEST.name}")
    print("\nSource matrix untouched. Proxy targets are NOT ground truth.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
