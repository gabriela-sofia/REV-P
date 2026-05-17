from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "dino" / "revp_v1gk_dino_pipeline_reproducibility_audit.py"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def test_v1gk_reproducibility_audit(tmp_path: Path) -> None:
    out = tmp_path / "local_runs" / "dino_embeddings" / "v1gk"
    result = subprocess.run([sys.executable, str(SCRIPT), "--output-dir", str(out), "--force"], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr + result.stdout
    summary = json.loads((out / "reproducibility_summary.json").read_text(encoding="utf-8"))
    assert summary["qa_status"] == "PASS"
    assert summary["script_count"] >= 15
    assert summary["local_runs_ignored"] is True
    assert summary["forbidden_versioned_artifacts"] == []
    qa = read_csv(out / "reproducibility_qa.csv")
    assert {row["status"] for row in qa} == {"PASS"}
    audit = read_csv(out / "reproducibility_audit.csv")
    assert any(row["audit_item"] == "v1gk" and row["status"] == "PASS" for row in audit)


def test_final_dino_docs_and_guardrails_exist() -> None:
    summary = ROOT / "docs" / "dino_sentinel_scientific_evidence_summary.md"
    registry = ROOT / "docs" / "dino_command_registry.md"
    protocol = ROOT / "docs" / "dino_sentinel_embedding_protocol.md"
    assert summary.exists()
    assert registry.exists()
    assert protocol.exists()
    summary_text = summary.read_text(encoding="utf-8").lower()
    registry_text = registry.read_text(encoding="utf-8").lower()
    protocol_text = protocol.read_text(encoding="utf-8").lower()
    assert "does not create labels" in summary_text
    assert "not a scientific label" in summary_text
    assert "multimodal_execution_enabled=false" in registry_text
    assert "local_runs/" in registry_text
    assert "v1gk" in protocol_text
    assert "review_only" in protocol_text
