from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "multimodal"))

from revp_v2ck_digitization_package_builder import ACCEPTED, REJECTED, main  # noqa: E402


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    write_csv(root / "outputs_public/tables/revp_tp2_candidate_priority_v2cj.csv",
              ["rank", "candidate_id", "region", "event_name", "source_name", "evidence_type",
               "review_priority_score", "priority_class", "main_strength", "main_blocker",
               "recommended_next_action", "allowed_claim", "forbidden_claim", "tp2_status", "tp3_ready"],
              [{"rank": "1", "candidate_id": "A", "region": "Recife", "event_name": "A",
                "source_name": "Charter", "evidence_type": "EVIDENCIA_VISUAL",
                "review_priority_score": "55", "priority_class": "MEDIUM_REVIEW_PRIORITY",
                "main_strength": "visual", "main_blocker": "crs", "recommended_next_action": "digitar",
                "allowed_claim": "review", "forbidden_claim": "ground_truth", "tp2_status": "TP2_BLOCKED",
                "tp3_ready": "false"}])
    write_csv(root / "outputs_public/tables/revp_tp2_candidate_inventory_v2ci.csv",
              ["candidate_id", "source_reference"],
              [{"candidate_id": "A", "source_reference": "local/ref"}])
    return root


def rows(tmp_path: Path) -> list[dict[str, str]]:
    root = repo(tmp_path)
    assert main(["--repo-root", str(root), "--force"]) == 0
    return read_csv(root / "outputs_public/tables/revp_digitization_task_queue_v2ck.csv")


def test_task_queue_exists(tmp_path: Path) -> None:
    root = repo(tmp_path)
    main(["--repo-root", str(root), "--force"])
    assert (root / "outputs_public/tables/revp_digitization_task_queue_v2ck.csv").exists()


def test_visual_formats_rejected(tmp_path: Path) -> None:
    row = rows(tmp_path)[0]
    assert "PNG_ISOLADO" in row["rejected_output_formats"]
    assert "JPEG_ISOLADO" in row["rejected_output_formats"]
    assert "PDF_SEM_GEORREFERENCIAMENTO" in row["rejected_output_formats"]


def test_vector_formats_accepted(tmp_path: Path) -> None:
    row = rows(tmp_path)[0]
    for token in ["GeoJSON", "GPKG", "Shapefile", "CSV_WKT_COM_CRS_EXPLICITO"]:
        assert token in row["accepted_output_formats"]


def test_every_item_requires_human_review(tmp_path: Path) -> None:
    assert all(row["requires_human_review"] == "true" for row in rows(tmp_path))


def test_report_declares_no_digitization_executed(tmp_path: Path) -> None:
    root = repo(tmp_path)
    main(["--repo-root", str(root), "--force"])
    text = (root / "outputs_public/execution_reports/revp_digitization_package_report_v2ck.md").read_text()
    assert "nao digitaliza automaticamente" in text


def test_manual_protocol_created(tmp_path: Path) -> None:
    root = repo(tmp_path)
    main(["--repo-root", str(root), "--force"])
    assert (root / "outputs_public/execution_reports/revp_digitization_manual_protocol_v2ck.md").exists()


def test_no_ground_truth_or_label_created(tmp_path: Path) -> None:
    root = repo(tmp_path)
    main(["--repo-root", str(root), "--force"])
    text = (root / "outputs_public/execution_reports/revp_digitization_package_report_v2ck.md").read_text()
    assert "nao cria ground truth" in text
    assert "nao cria label" in text


def test_constants_include_expected_formats() -> None:
    assert "GeoJSON" in ACCEPTED
    assert "DESCRICAO_TEXTUAL" in REJECTED

