"""REV-P v2ck - manual digitization/georeferencing package builder."""
from __future__ import annotations

import argparse
from pathlib import Path

from revp_v2cj_to_v2cm_common import priority_path, queue_path, read_csv, write_csv, write_text


FIELDS = [
    "task_id",
    "candidate_id",
    "region",
    "source_name",
    "source_reference",
    "input_evidence_type",
    "requires_georeferencing",
    "requires_digitization",
    "requires_crs_assignment",
    "requires_human_review",
    "minimum_required_outputs",
    "accepted_output_formats",
    "rejected_output_formats",
    "quality_checks_required",
    "blocking_reason",
    "next_action",
]

ACCEPTED = "GeoJSON|GPKG|Shapefile|CSV_WKT_COM_CRS_EXPLICITO"
REJECTED = "PNG_ISOLADO|JPEG_ISOLADO|PDF_SEM_GEORREFERENCIAMENTO|DESCRICAO_TEXTUAL|LINK_SEM_ARQUIVO_LOCAL|COORDENADA_INFERIDA_SEM_FONTE"


def build_queue(repo_root: Path) -> list[dict[str, str]]:
    priority = read_csv(priority_path(repo_root))
    inv = {row.get("candidate_id", ""): row for row in read_csv(repo_root / "outputs_public/tables/revp_tp2_candidate_inventory_v2ci.csv")}
    rows: list[dict[str, str]] = []
    for idx, row in enumerate(priority, 1):
        source = inv.get(row.get("candidate_id", ""), {})
        evidence_type = row.get("evidence_type", "")
        requires_digitization = evidence_type in {"EVIDENCIA_VISUAL", "EVIDENCIA_DOCUMENTAL", "GEOMETRIA_CANDIDATA"}
        rows.append(
            {
                "task_id": f"TASK_v2ck_{idx:04d}",
                "candidate_id": row.get("candidate_id", ""),
                "region": row.get("region", ""),
                "source_name": row.get("source_name", ""),
                "source_reference": source.get("source_reference", ""),
                "input_evidence_type": evidence_type,
                "requires_georeferencing": "true",
                "requires_digitization": "true" if requires_digitization else "false",
                "requires_crs_assignment": "true",
                "requires_human_review": "true",
                "minimum_required_outputs": "vetor_observado|crs_explicito|proveniencia|hash|vinculo_documental",
                "accepted_output_formats": ACCEPTED,
                "rejected_output_formats": REJECTED,
                "quality_checks_required": "validade_geometrica|crs|proveniencia|hash|escopo_municipal|tipo_fenomeno",
                "blocking_reason": row.get("main_blocker", ""),
                "next_action": "digitalizacao_manual_revisada" if row.get("priority_class") != "BLOCKED_FOR_REVIEW" else "aguardar_evidencia_melhor",
            }
        )
    return rows


def report(rows: list[dict[str, str]]) -> str:
    return f"""# REV-P v2ck - pacote de digitalizacao/georreferenciamento manual

Este pacote prepara trabalho humano. Ele nao digitaliza automaticamente, nao
transforma imagem em vetor por inferencia, nao cria ground truth e nao cria label.

Total de tarefas: {len(rows)}.
Formatos futuros aceitos: {ACCEPTED}.
Formatos rejeitados como geometria validada: {REJECTED}.
"""


def manual_protocol() -> str:
    return f"""# Protocolo manual v2ck

Para cada tarefa, o revisor deve produzir geometria vetorial local com CRS
explicito, proveniencia, hash e vinculo documental. Saidas aceitas: {ACCEPTED}.
Saidas rejeitadas: {REJECTED}.

Nenhuma coordenada pode ser inferida sem fonte. Nenhum arquivo visual isolado vale
como geometria observada validada.
"""


def run(repo_root: Path, force: bool = False) -> int:
    rows = build_queue(repo_root)
    out = queue_path(repo_root)
    if out.exists() and not force:
        raise FileExistsError(out)
    write_csv(out, rows, FIELDS)
    write_text(repo_root / "outputs_public/execution_reports/revp_digitization_package_report_v2ck.md", report(rows))
    write_text(repo_root / "outputs_public/execution_reports/revp_digitization_manual_protocol_v2ck.md", manual_protocol())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    return run(Path(args.repo_root), args.force)


if __name__ == "__main__":
    raise SystemExit(main())
