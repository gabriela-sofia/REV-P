"""REV-P v1qs — TCC local readiness update.

Generates TCC-ready tables describing the local execution readiness status
of the DINO smoke pipeline. Encodes the mandatory methodological phrase.
"""
from __future__ import annotations

import argparse
from typing import Any

from revp_v1qn_v1qt_local_readiness_common import (
    DATASETS, DOCS, READINESS_PHRASE, SCHEMAS,
    _p, assert_no_forbidden_true, read_csv,
    require_no_abs_paths, write_csv, write_doc, write_schema,
)

IN_GATE_SUM = _p("REVP_V1QS_IN_GATE",  DATASETS / "dino_local_smoke_run_readiness_summary_v1qr.csv")
IN_REC_SUM  = _p("REVP_V1QS_IN_REC",   DATASETS / "dino_smoke_asset_local_reconciliation_summary_v1qo.csv")
IN_ENV_SUM  = _p("REVP_V1QS_IN_ENV",   DATASETS / "dino_local_root_environment_summary_v1qn.csv")

OUT_READ    = _p("REVP_V1QS_OUT_READ",  DATASETS / "dino_tcc_table_local_readiness_v1qs.csv")
OUT_BLOCK   = _p("REVP_V1QS_OUT_BLOCK", DATASETS / "dino_tcc_table_local_blockers_v1qs.csv")
SCH_READ    = _p("REVP_V1QS_SCH_READ",  SCHEMAS / "dino_tcc_table_local_readiness_v1qs_schema.csv")
SCH_BLOCK   = _p("REVP_V1QS_SCH_BLOCK", SCHEMAS / "dino_tcc_table_local_blockers_v1qs_schema.csv")
DOC         = _p("REVP_V1QS_DOC",        DOCS / "revp_v1qs_local_readiness_tcc_update.md")

READ_FIELDS  = ["row_id", "indicator", "value", "scientific_reading", "boundary"]
BLOCK_FIELDS = ["blocker_id", "blocker_name", "current_status", "action_required",
                "impact_on_tcc", "review_only"]


def _stat(path: Any, key: str, default: str = "0") -> str:
    for r in read_csv(path):
        if r.get("stat_key") == key:
            return r.get("stat_value", default)
    return default


def run() -> None:
    gate_status  = _stat(IN_GATE_SUM, "final_status", "MISSING")
    gates_passed = _stat(IN_GATE_SUM, "gates_passed", "0")
    gates_total  = _stat(IN_GATE_SUM, "gates_total", "0")
    reconciled   = str(int(_stat(IN_REC_SUM, "exact_matches", "0") or "0") +
                       int(_stat(IN_REC_SUM, "partial_matches", "0") or "0"))
    unresolved   = _stat(IN_REC_SUM, "unresolved", "0")
    roots_exist  = _stat(IN_ENV_SUM, "roots_existing", "0")
    imgs_found   = _stat(IN_ENV_SUM, "total_candidate_image_files", "0")
    model_set    = _stat(IN_ENV_SUM, "model_path_set", "false")
    boundary = "representacao_visual_review_only_nao_valida_evento"

    read_rows: list[dict[str, Any]] = [
        {"row_id": "V1QS_R01", "indicator": "Roots configurados existentes",
         "value": roots_exist, "scientific_reading": "Diretórios locais de assets disponíveis",
         "boundary": boundary},
        {"row_id": "V1QS_R02", "indicator": "Imagens candidatas encontradas",
         "value": imgs_found, "scientific_reading": "Arquivos TIF/PNG locais visíveis ao pipeline",
         "boundary": boundary},
        {"row_id": "V1QS_R03", "indicator": "Assets smoke reconciliados",
         "value": reconciled, "scientific_reading": "Patches com arquivo local identificado",
         "boundary": boundary},
        {"row_id": "V1QS_R04", "indicator": "Modelo local configurado",
         "value": model_set, "scientific_reading": "REVP_DINO_MODEL_PATH set e verificado",
         "boundary": boundary},
        {"row_id": "V1QS_R05", "indicator": "Gates de prontidão aprovados",
         "value": f"{gates_passed}/{gates_total}",
         "scientific_reading": "Critérios para execução smoke real satisfeitos",
         "boundary": boundary},
        {"row_id": "V1QS_R06", "indicator": "Status final prontidão local",
         "value": gate_status, "scientific_reading": "Resultado auditável da etapa v1qn-v1qt",
         "boundary": boundary},
    ]

    block_rows: list[dict[str, Any]] = []
    if model_set != "true":
        block_rows.append({"blocker_id": "V1QS_BK01", "blocker_name": "model_path_not_set",
                            "current_status": "MISSING", "review_only": "true",
                            "action_required": "Set REVP_DINO_MODEL_PATH to local DINOv2 dir",
                            "impact_on_tcc": "sem modelo local, embedding real impossível"})
    if int(unresolved or "0") > 0:
        block_rows.append({"blocker_id": "V1QS_BK02", "blocker_name": "unresolved_assets",
                            "current_status": f"unresolved={unresolved}", "review_only": "true",
                            "action_required": "Set REVP_SENTINEL_LOCAL_ROOT with TIF location",
                            "impact_on_tcc": "sem TIFs locais, embedding real impossível"})
    if int(roots_exist or "0") == 0:
        block_rows.append({"blocker_id": "V1QS_BK03", "blocker_name": "no_local_roots",
                            "current_status": "MISSING", "review_only": "true",
                            "action_required": "Configure at least one REVP_*_LOCAL_ROOT env var",
                            "impact_on_tcc": "sem roots, resolução de assets impossível"})
    if not block_rows:
        block_rows.append({"blocker_id": "V1QS_BK00", "blocker_name": "no_blockers",
                            "current_status": gate_status, "review_only": "true",
                            "action_required": "none", "impact_on_tcc": "pipeline pronto"})

    for rows, label in ((read_rows, "v1qs_read"), (block_rows, "v1qs_block")):
        require_no_abs_paths(rows, label)
        assert_no_forbidden_true(rows, label)

    write_csv(OUT_READ,  read_rows,  READ_FIELDS)
    write_csv(OUT_BLOCK, block_rows, BLOCK_FIELDS)
    write_schema(SCH_READ,  READ_FIELDS,  "v1qs_tcc_table_local_readiness")
    write_schema(SCH_BLOCK, BLOCK_FIELDS, "v1qs_tcc_table_local_blockers")
    write_doc(DOC, "v1qs — TCC Local Readiness Update", [
        "## Frase metodológica obrigatória",
        READINESS_PHRASE,
        "## Tabelas TCC",
        "Indicadores de prontidão local e bloqueadores para execução smoke real.",
        "## Status",
        f"**{gate_status}**. Gates passados: {gates_passed}/{gates_total}. "
        f"Assets reconciliados: {reconciled}. Bloqueadores: {len(block_rows)}.",
    ])
    print(f"[v1qs] gate={gate_status} read_rows={len(read_rows)} blockers={len(block_rows)}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1qs local readiness tcc update").parse_args()
    run()
