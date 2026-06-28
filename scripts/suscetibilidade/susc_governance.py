"""
SUSC shared governance constants and helpers.

Every SUSC artifact is review-only and must NEVER unlock training or assert
ground truth. These helpers apply and verify those invariants uniformly.
"""

from __future__ import annotations

ALLOWED_FOR_TRAINING = False
CAN_BE_GROUND_TRUTH = False
REVIEW_ONLY = True

GOVERNANCE_COLUMNS = {
    "can_be_ground_truth": "false",
    "allowed_for_training": "false",
    "review_only": "true",
}
# Some artifacts use the longer name for the GT flag.
GT_ALIASES = ("can_be_ground_truth", "can_be_used_as_ground_truth")


def apply_governance(row: dict, gt_key: str = "can_be_ground_truth") -> dict:
    """Stamp governance columns onto a row (in place) and return it."""
    row[gt_key] = "false"
    row["allowed_for_training"] = "false"
    row["review_only"] = "true"
    return row


def _gt_value(row: dict):
    for k in GT_ALIASES:
        if k in row:
            return str(row[k]).strip().lower()
    return None


def row_violates_governance(row: dict) -> bool:
    gt = _gt_value(row)
    if gt is not None and gt == "true":
        return True
    if "allowed_for_training" in row and str(row["allowed_for_training"]).strip().lower() == "true":
        return True
    if "review_only" in row and str(row["review_only"]).strip().lower() != "true":
        return True
    return False


def violations(rows: list[dict]) -> list[int]:
    """Return indices of rows that violate governance."""
    return [i for i, r in enumerate(rows) if row_violates_governance(r)]


def assert_clean(rows: list[dict], label: str = "rows") -> list[str]:
    """Return a list of error strings (empty if clean)."""
    bad = violations(rows)
    if bad:
        return [f"{label}: {len(bad)} governance violation(s) at rows {bad[:10]}"]
    return []


MANDATORY_NON_GT_PHRASES = (
    "não cria ground truth",
    "review-only",
    "não desbloqueia treinamento",
)
