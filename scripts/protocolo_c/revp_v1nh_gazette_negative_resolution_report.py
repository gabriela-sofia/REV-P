"""REV-P v1nh - final gazette negative resolution report."""

from __future__ import annotations

import argparse
import json

from revp_v1lj_v1lq_common import DATASETS, DOCS, SCHEMAS, write_csv, write_schema
from revp_v1na_v1nh_common import gazette_summary


OUT_SUMMARY = DATASETS / "protocol_c_gazette_negative_resolution_summary.csv"
SCHEMA = SCHEMAS / "protocol_c_gazette_negative_resolution_summary_schema.csv"
DOC_METHOD = DOCS / "protocolo_c_diario_oficial_negativos_v1na_v1nh.md"
DOC_REPORT = DOCS / "protocolo_c_relatorio_diario_oficial_negativos_v1na_v1nh.md"
FIELDS = ["summary_id", "issues_discovered", "issues_downloaded", "pages_extracted", "administrative_acts_segmented", "negative_candidates", "geocoded_candidates", "formal_negative_count", "c4_decision", "best_negative_candidate", "remaining_blocker", "next_single_technical_action", "can_create_operational_label", "can_train_model"]


def write_docs(summary: dict[str, str]) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    DOC_METHOD.write_text(
        "# Protocolo C - Diario Oficial e negativos v1na-v1nh\n\n"
        "Rota dirigida ao Diario Oficial antigo de Petropolis para 2022. Listagem nao conta como edicao. "
        "Ausencia de edicao, baixo risco e desinterdicao sem justificativa explicita nao criam negativo formal.\n",
        encoding="utf-8",
    )
    DOC_REPORT.write_text(
        "# Relatorio v1na-v1nh - Diario Oficial\n\n"
        f"- Edicoes reais descobertas: {summary['issues_discovered']}.\n"
        f"- Edicoes baixadas: {summary['issues_downloaded']}.\n"
        f"- Paginas extraidas: {summary['pages_extracted']}.\n"
        f"- Atos segmentados: {summary['administrative_acts_segmented']}.\n"
        f"- Candidatos negativos: {summary['negative_candidates']}.\n"
        f"- Candidatos geocodificados: {summary['geocoded_candidates']}.\n"
        f"- Negativos formais: {summary['formal_negative_count']}.\n"
        f"- C4: {summary['c4_decision']}.\n"
        f"- Bloqueador: {summary['remaining_blocker']}.\n"
        f"- Proxima acao: {summary['next_single_technical_action']}.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--emit-summary", action="store_true")
    args = parser.parse_args()
    if OUT_SUMMARY.exists() and not args.force:
        print(json.dumps({"stage": "v1nh", "status": "EXISTING_OUTPUTS_PRESERVED"}))
        return
    summary = gazette_summary()
    if args.force or args.emit_summary:
        write_csv(OUT_SUMMARY, [summary], FIELDS)
        write_schema(SCHEMA, FIELDS, "v1nh gazette negative resolution summary")
        write_docs(summary)
    print(json.dumps({"stage": "v1nh", **summary}, indent=2))


if __name__ == "__main__":
    main()
