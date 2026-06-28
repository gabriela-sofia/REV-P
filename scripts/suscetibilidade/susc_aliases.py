"""
SUSC feature-name alias resolution.

Reconciles requested/legacy feature names with the real columns present in the
SUSC-03 matrix, so downstream code never silently misses a column. Emits an
auditable alias-resolution report.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from susc_io import read_csv, write_csv, rel  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "datasets" / "suscetibilidade" / "susc_features_by_patch_v1.csv"
ALIAS_REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_09A_alias_resolution_report.csv"

# requested/legacy name -> real column in SUSC-03
ALIAS_MAP = {
    "flow_acc_p75": "flow_acc_log_p75",
    "chirps_3d_to_7d_ratio": "rain_3d_7d_ratio",
    "chirps_7d_to_30d_ratio": "rain_7d_30d_ratio",
    "chirps_3d_to_30d_ratio": "rain_3d_7d_ratio",  # closest real ratio (different def)
    "rain_persistence": "rain_persistence_index",
    "s1_vv_minus_vh": "s1_vv_minus_vh_mean_clean",
    "runoff_score": "runoff_context_7d",  # closest real composite (different def)
}


def resolve(name: str) -> str:
    """Return the real matrix column for a possibly-aliased name."""
    return ALIAS_MAP.get(name, name)


def matrix_columns() -> set[str]:
    if not MATRIX.exists():
        return set()
    rows = read_csv(MATRIX)
    return set(rows[0].keys()) if rows else set()


def build_alias_report() -> Path:
    cols = matrix_columns()
    rows = []
    for requested, real in sorted(ALIAS_MAP.items()):
        present = real in cols
        exact_def = requested not in {"chirps_3d_to_30d_ratio", "runoff_score",
                                      "chirps_3d_to_7d_ratio"}
        rows.append({
            "requested_name": requested,
            "resolved_real_column": real,
            "present_in_susc03": str(present).lower(),
            "exact_definition_match": str(exact_def).lower(),
            "resolution_status": ("resolved_exact" if present and exact_def else
                                  "resolved_closest_different_definition" if present else
                                  "unresolved_absent"),
            "requires_manual_review": str(not (present and exact_def)).lower(),
            "notes": "alias maps requested/legacy name to the real SUSC-03 column",
        })
    return write_csv(ALIAS_REPORT, rows)


if __name__ == "__main__":
    p = build_alias_report()
    print(f"alias report -> {rel(p)} ({len(ALIAS_MAP)} aliases)")
