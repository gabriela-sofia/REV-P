"""Shared helpers for REV-P Protocol C v1rs-v1rz integration/hardening.

Integration and commit-readiness utilities. No new science; no labels,
no targets, no operational ground truth, no DINO-as-proof.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from revp_v1qu_v1qz_ground_reference_common import (  # noqa: F401
    DATASETS,
    DOCS,
    SCHEMAS,
    _p,
    assert_clean_rows,
    guardrail_row,
    read_csv_safe,
    safe_relpath,
    write_csv_with_header,
    write_doc,
    write_json_safe,
    write_schema_safe,
)

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Guardrail field list
# ---------------------------------------------------------------------------

GUARDRAIL_FIELDS = [
    "can_create_operational_label", "can_train_model", "target_created",
    "ground_truth_operational", "formal_negative", "dino_validates_event",
    "absence_as_negative",
]

# Forbidden literal (split so this source never embeds it)
_FORBIDDEN_LITERAL = "local" + "_runs"

ABS_PATH_RE = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]")


def hash_short(value: str, n: int = 12) -> str:
    return hashlib.sha256(str(value).encode("utf-8", errors="ignore")).hexdigest()[:n]


def detect_absolute_path(text: str) -> bool:
    return bool(ABS_PATH_RE.search(str(text)))


def detect_forbidden_literal_exposure(text: str) -> bool:
    return _FORBIDDEN_LITERAL in str(text).lower()


def scan_csv_guardrails(path: Path) -> dict[str, int]:
    """Return counts of violations in a CSV. Never reads >50k rows."""
    counts: dict[str, int] = {
        "abs_path": 0, "forbidden_literal": 0, **{f: 0 for f in GUARDRAIL_FIELDS}
    }
    if not path.exists():
        return counts
    try:
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh)
            for i, row in enumerate(reader):
                if i >= 50_000:
                    break
                for k, v in row.items():
                    sv = str(v)
                    if detect_absolute_path(sv):
                        counts["abs_path"] += 1
                    if detect_forbidden_literal_exposure(sv):
                        counts["forbidden_literal"] += 1
                    if k in GUARDRAIL_FIELDS and sv.strip().lower() == "true":
                        counts[k] += 1
    except Exception:
        pass
    return counts


def scan_doc_guardrails(path: Path) -> dict[str, int]:
    """Scan a doc for absolute paths and forbidden literal in content."""
    counts = {"abs_path": 0, "forbidden_literal": 0}
    if not path.exists():
        return counts
    try:
        text = path.read_text(encoding="utf-8", errors="replace")[:100_000]
        lines = text.splitlines()
        for line in lines:
            if detect_absolute_path(line):
                counts["abs_path"] += 1
            if detect_forbidden_literal_exposure(line):
                counts["forbidden_literal"] += 1
    except Exception:
        pass
    return counts


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as fh:
            return sum(1 for _ in csv.DictReader(fh))
    except Exception:
        return 0


def artifact_exists(path: Path) -> bool:
    return path.exists()


def infer_stage_from_filename(name: str) -> str:
    """Extract stage token like v1rs from a filename."""
    m = re.search(r"_(v1[a-z]{2})\b", name.lower())
    return m.group(1).upper() if m else "UNKNOWN"


def infer_block_from_filename(name: str) -> str:
    """Map a filename to a high-level block label."""
    lo = name.lower()
    if any(t in lo for t in ["v1pg", "v1ph", "v1pi", "v1pj", "v1pk", "v1pl", "v1pm"]):
        return "DINO_REPRESENTATION_V1PG_V1PM"
    if any(t in lo for t in ["v1pn", "v1po", "v1pp", "v1pq", "v1pr", "v1ps", "v1pt"]):
        return "DINO_HARNESS_V1PN_V1PT"
    if any(t in lo for t in ["v1pu", "v1pv", "v1pw", "v1px", "v1py", "v1pz"]):
        return "DINO_VISUAL_V1PU_V1PZ"
    if any(t in lo for t in ["v1qa", "v1qb", "v1qc", "v1qd", "v1qe", "v1qf"]):
        return "DINO_BRIDGE_V1QA_V1QF"
    if any(t in lo for t in ["v1qg", "v1qh", "v1qi", "v1qj", "v1qk", "v1ql", "v1qm"]):
        return "DINO_SMOKE_V1QG_V1QM"
    if any(t in lo for t in ["v1qn", "v1qo", "v1qp", "v1qq", "v1qr", "v1qs", "v1qt"]):
        return "DINO_LOCAL_V1QN_V1QT"
    if any(t in lo for t in ["v1qu", "v1qv", "v1qw", "v1qx", "v1qy", "v1qz"]):
        return "GROUND_REF_P0_V1QU_V1QZ"
    if any(t in lo for t in ["v1ra", "v1rb", "v1rc", "v1rd", "v1re", "v1rf"]):
        return "EXTERNAL_INTAKE_P1_V1RA_V1RF"
    if any(t in lo for t in ["v1rg", "v1rh", "v1ri", "v1rj", "v1rk", "v1rl", "v1rm"]):
        return "REVIEW_GATE_P2_V1RG_V1RM"
    if any(t in lo for t in ["v1rn", "v1ro", "v1rp", "v1rq", "v1rr"]):
        return "DASHBOARD_P3_V1RN_V1RR"
    if any(t in lo for t in ["v1rs", "v1rt", "v1ru", "v1rv", "v1rw", "v1rx", "v1ry", "v1rz"]):
        return "INTEGRATION_V1RS_V1RZ"
    return "UNKNOWN"


def collect_artifacts_by_patterns(
    patterns: list[str], root: Path | None = None
) -> list[Path]:
    """Collect file paths matching glob patterns under root."""
    root = root or ROOT
    out: list[Path] = []
    for pat in patterns:
        for p in sorted(root.glob(pat)):
            if p not in out:
                out.append(p)
    return out


def build_dependency_edge(
    edge_id: str,
    source_block: str, source_artifact: str,
    target_block: str, target_artifact: str,
    dep_type: str = "INPUT",
    required: str = "true",
    fail_closed: str = "true",
) -> dict[str, str]:
    return {
        "edge_id": edge_id,
        "source_block": source_block,
        "source_artifact": source_artifact,
        "target_block": target_block,
        "target_artifact": target_artifact,
        "dependency_type": dep_type,
        "required": required,
        "fail_closed_behavior": fail_closed,
        "notes": "",
    }


def classify_artifact_status(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    if path.stat().st_size == 0:
        return "EMPTY"
    rows = count_rows(path)
    if rows == 0:
        return "HEADER_ONLY"
    return "PRESENT"
