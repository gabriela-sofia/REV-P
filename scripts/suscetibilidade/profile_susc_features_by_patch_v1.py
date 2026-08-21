"""
SUSC-03 Exploratory Profiling (programmatic diagnostics, NO model training)

Generates lightweight, auditable exploratory diagnostics over the migrated
susceptibility matrix datasets/suscetibilidade/susc_features_by_patch_v1.csv.

Outputs (outputs_public/suscetibilidade/):
  - SUSC_03_feature_profile_by_group.csv     feature counts per group
  - SUSC_03_missingness_report.csv           missingness per column
  - SUSC_03_numeric_summary.csv              mean/median/std/min/max
  - SUSC_03_region_summary.csv               distribution per region
  - SUSC_03_feature_correlation_screen.csv   physical/orbital features vs scores

This is exploratory profiling ONLY. It does NOT train any model, does NOT
create ground truth, and does NOT treat heuristic labels or v5 scores as
validated targets. Susceptibility features != confirmed occurrence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())

SCHEMA_PATH = ROOT / "schemas" / "suscetibilidade" / "susc_features_schema_v1.json"
CSV_PATH = ROOT / "datasets" / "suscetibilidade" / "susc_features_by_patch_v1.csv"
OUT_DIR = ROOT / "outputs_public" / "suscetibilidade"

COLLINEARITY_THRESHOLD = 0.90

# Profile grouping vocabulary (14 buckets). Schema groups outside this set are
# mapped deterministically: proxy_v5 -> heuristic_label (heuristic binary flags),
# cbers_* -> unknown (legacy, not migrated). The fold is documented in the report.
PROFILE_GROUP_MAP = {
    "patch_identity": "patch_identity",
    "geometry": "geometry",
    "sentinel2_bands": "sentinel2_bands",
    "spectral_index": "spectral_index",
    "topography": "topography",
    "hydrology": "hydrology",
    "precipitation": "precipitation",
    "sar": "sar",
    "land_use": "land_use",
    "interaction": "interaction",
    "proxy_v5": "heuristic_label",
    "score": "score",
    "heuristic_label": "heuristic_label",
    "qa": "qa",
    "cbers_linkage": "unknown",
    "cbers_refresh": "unknown",
}

PROFILE_VOCAB = [
    "patch_identity",
    "geometry",
    "sentinel2_bands",
    "spectral_index",
    "topography",
    "hydrology",
    "precipitation",
    "sar",
    "land_use",
    "interaction",
    "score",
    "heuristic_label",
    "qa",
    "unknown",
]

# Physical / orbital feature groups screened against heuristic scores.
PHYSICAL_GROUPS = {
    "sentinel2_bands",
    "spectral_index",
    "topography",
    "hydrology",
    "precipitation",
    "sar",
    "land_use",
    "interaction",
}


def main() -> int:
    print("=" * 60)
    print("SUSC-03 Exploratory Profiling (no model training)")
    print("=" * 60)

    if not CSV_PATH.exists():
        print(f"STOP: migrated matrix not found: {CSV_PATH}")
        print("       Run migrate_dataset_final_to_revp.py first.")
        return 1
    if not SCHEMA_PATH.exists():
        print(f"STOP: schema not found: {SCHEMA_PATH}")
        return 1

    with SCHEMA_PATH.open("r", encoding="utf-8") as f:
        schema = json.load(f)
    feat_meta = {f["source_column"]: f for f in schema["features"]}

    df = pd.read_csv(CSV_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ----- 1. feature profile by group -----
    rows = []
    for col in df.columns:
        meta = feat_meta.get(col, {})
        schema_group = meta.get("group", "unknown")
        profile_group = PROFILE_GROUP_MAP.get(schema_group, "unknown")
        rows.append(
            {
                "column": col,
                "schema_group": schema_group,
                "profile_group": profile_group,
                "dtype": meta.get("dtype", "unknown"),
            }
        )
    feat_df = pd.DataFrame(rows)
    group_profile = (
        feat_df.groupby("profile_group")
        .agg(
            n_features=("column", "count"),
            schema_groups=("schema_group", lambda s: ";".join(sorted(set(s)))),
        )
        .reindex(PROFILE_VOCAB)
        .fillna({"n_features": 0, "schema_groups": ""})
    )
    group_profile["n_features"] = group_profile["n_features"].astype(int)
    group_profile = group_profile.reset_index().rename(columns={"index": "profile_group"})
    group_profile.to_csv(
        OUT_DIR / "SUSC_03_feature_profile_by_group.csv", index=False
    )
    print(f"\n[1/5] feature_profile_by_group.csv ({len(group_profile)} groups)")

    # ----- 2. missingness report -----
    miss_rows = []
    n = len(df)
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        miss_rows.append(
            {
                "column": col,
                "schema_group": feat_meta.get(col, {}).get("group", "unknown"),
                "n_rows": n,
                "n_missing": n_missing,
                "pct_missing": round(100.0 * n_missing / n, 4) if n else 0.0,
                "requires_provenance_audit": feat_meta.get(col, {}).get(
                    "requires_provenance_audit", False
                ),
            }
        )
    pd.DataFrame(miss_rows).to_csv(
        OUT_DIR / "SUSC_03_missingness_report.csv", index=False
    )
    print(f"[2/5] missingness_report.csv ({len(miss_rows)} columns)")

    # ----- 3. numeric summary -----
    numeric_cols = [
        c for c in df.columns
        if feat_meta.get(c, {}).get("dtype") in {"float", "integer"}
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    num_rows = []
    for col in numeric_cols:
        s = df[col].dropna()
        num_rows.append(
            {
                "column": col,
                "schema_group": feat_meta.get(col, {}).get("group", "unknown"),
                "count": int(s.count()),
                "mean": float(s.mean()) if len(s) else np.nan,
                "median": float(s.median()) if len(s) else np.nan,
                "std": float(s.std()) if len(s) else np.nan,
                "min": float(s.min()) if len(s) else np.nan,
                "max": float(s.max()) if len(s) else np.nan,
            }
        )
    pd.DataFrame(num_rows).to_csv(
        OUT_DIR / "SUSC_03_numeric_summary.csv", index=False
    )
    print(f"[3/5] numeric_summary.csv ({len(num_rows)} numeric columns)")

    # ----- 4. region summary -----
    region_rows = []
    if "regiao" in df.columns:
        key_scores = [
            c for c in df.columns
            if feat_meta.get(c, {}).get("group") == "score"
        ]
        for region, g in df.groupby("regiao"):
            row = {"regiao": region, "n_patches": int(len(g))}
            for sc in key_scores:
                if pd.api.types.is_numeric_dtype(g[sc]):
                    row[f"{sc}__mean"] = float(g[sc].mean())
            region_rows.append(row)
    pd.DataFrame(region_rows).to_csv(
        OUT_DIR / "SUSC_03_region_summary.csv", index=False
    )
    print(f"[4/5] region_summary.csv ({len(region_rows)} regions)")

    # ----- 5. correlation screen: physical/orbital features vs scores -----
    phys_cols = [
        c for c in numeric_cols
        if feat_meta.get(c, {}).get("group") in PHYSICAL_GROUPS
    ]
    score_cols = [
        c for c in numeric_cols
        if feat_meta.get(c, {}).get("group") == "score"
    ]
    corr_rows = []
    for feat_col in phys_cols:
        for score_col in score_cols:
            pair = df[[feat_col, score_col]].dropna()
            if len(pair) < 3 or pair[feat_col].std() == 0 or pair[score_col].std() == 0:
                corr = np.nan
            else:
                corr = float(pair[feat_col].corr(pair[score_col]))
            corr_rows.append(
                {
                    "feature": feat_col,
                    "feature_group": feat_meta.get(feat_col, {}).get("group"),
                    "score": score_col,
                    "pearson_corr": round(corr, 6) if corr == corr else np.nan,
                    "abs_corr": round(abs(corr), 6) if corr == corr else np.nan,
                    "collinearity_alert": bool(
                        corr == corr and abs(corr) >= COLLINEARITY_THRESHOLD
                    ),
                }
            )
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(
        OUT_DIR / "SUSC_03_feature_correlation_screen.csv", index=False
    )
    n_alerts = int(corr_df["collinearity_alert"].sum()) if len(corr_df) else 0
    print(
        f"[5/5] feature_correlation_screen.csv "
        f"({len(corr_df)} pairs, {n_alerts} collinearity alerts >= {COLLINEARITY_THRESHOLD})"
    )

    print("\nProfiling complete. NO model was trained.")
    print("Susceptibility features != confirmed occurrence.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
