from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V1GH = ROOT / "scripts" / "dino" / "revp_v1gh_dino_longitudinal_structural_diagnostics.py"
V1GI = ROOT / "scripts" / "dino" / "revp_v1gi_dino_structural_provenance_tracker.py"
V1GJ = ROOT / "scripts" / "dino" / "revp_v1gj_multimodal_readiness_audit.py"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def test_v1gh_longitudinal_outputs(tmp_path: Path) -> None:
    out = tmp_path / "local_runs" / "dino_embeddings" / "v1gh"
    result = subprocess.run([sys.executable, str(V1GH), "--output-dir", str(out), "--force"], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr + result.stdout
    summary = json.loads((out / "longitudinal_summary.json").read_text(encoding="utf-8"))
    assert summary["qa_status"] == "PASS"
    assert summary["embeddings_analyzed"] > 0
    assert read_csv(out / "longitudinal_neighbor_persistence.csv")
    assert read_csv(out / "longitudinal_review_priority.csv")
    assert {row["review_priority_is_not_label"] for row in read_csv(out / "longitudinal_review_priority.csv")} == {"true"}


def test_v1gi_provenance_traceability(tmp_path: Path) -> None:
    out = tmp_path / "local_runs" / "dino_embeddings" / "v1gi"
    result = subprocess.run([sys.executable, str(V1GI), "--output-dir", str(out), "--force"], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr + result.stdout
    summary = json.loads((out / "provenance_summary.json").read_text(encoding="utf-8"))
    assert summary["qa_status"] == "PASS"
    provenance = read_csv(out / "structural_provenance_index.csv")
    assert provenance
    assert {row["label_status"] for row in provenance} == {"NO_LABEL"}
    assert read_csv(out / "patch_diagnostic_history.csv")
    assert read_csv(out / "review_traceability.csv")


def test_v1gj_multimodal_readiness_disabled_and_blocked(tmp_path: Path) -> None:
    out = tmp_path / "local_runs" / "dino_embeddings" / "v1gj"
    result = subprocess.run([sys.executable, str(V1GJ), "--output-dir", str(out), "--force"], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr + result.stdout
    guardrails = json.loads((out / "multimodal_guardrails.json").read_text(encoding="utf-8"))
    assert guardrails["multimodal_execution_enabled"] is False
    assert guardrails["multimodal_training_enabled"] is False
    summary = json.loads((out / "multimodal_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["multimodal_readiness_status"] == "HOLD"
    assert summary["qa_status"] == "PASS"
    blockers = read_csv(out / "multimodal_blockers.csv")
    assert blockers
    assert any(row["blocker"] == "crs_consistency_known" for row in blockers)
