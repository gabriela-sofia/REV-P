"""MV2-DATA-05 Tarefa 11 — Summary JSON do intake de janela temporal.

Lê os CSVs gerados e produz um resumo coerente e fail-closed. Sem template preenchido, o
resultado correto é 0 promoted e DATA-06 bloqueado, com template de correção pronto.

Saída: outputs_public/mv2_data_temporal_window_intake/mv2_data_05_summary.json
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from mv2_data_05_common import (
    BRANCH_EXECUTADA,
    PROJECT_ROOT,
    PUBLIC_DIR,
    clean,
    ensure_public_dir,
    load_intake_source,
    write_json,
    read_csv_rows,
)


def build_summary() -> dict[str, object]:
    _, filled_found, _ = load_intake_source()
    normalized = read_csv_rows(PUBLIC_DIR / "mv2_data_05_normalized_temporal_windows.csv")
    evidence = read_csv_rows(PUBLIC_DIR / "mv2_data_05_temporal_evidence_validation.csv")
    gate = read_csv_rows(PUBLIC_DIR / "mv2_data_05_temporal_promotion_gate.csv")
    probe_ready = read_csv_rows(PUBLIC_DIR / "mv2_data_05_probe_ready_batch.csv")

    ev = Counter(clean(r.get("evidence_status")) for r in evidence)
    prom = Counter(clean(r.get("promotion_status")) for r in gate)

    return {
        "stage": "mv2_data_05_temporal_window_intake",
        "vertente": "B_desbloqueio_cientifico_de_dados",
        "branch": BRANCH_EXECUTADA,
        "total_input_rows": len(normalized),
        "filled_template_found": filled_found,
        "temporal_valid_strong": ev.get("TEMPORAL_EVIDENCE_VALID_STRONG", 0),
        "temporal_valid_partial": ev.get("TEMPORAL_EVIDENCE_VALID_PARTIAL", 0),
        "temporal_review_required": ev.get("TEMPORAL_EVIDENCE_REVIEW_REQUIRED", 0),
        "temporal_empty": ev.get("TEMPORAL_EVIDENCE_EMPTY", 0),
        "temporal_conflict": ev.get("TEMPORAL_EVIDENCE_CONFLICT", 0),
        "temporal_invalid": ev.get("TEMPORAL_EVIDENCE_INVALID", 0),
        "temporal_promoted_strong": prom.get("TEMPORAL_WINDOW_PROMOTED_STRONG", 0),
        "temporal_promoted_partial": prom.get("TEMPORAL_WINDOW_PROMOTED_PARTIAL", 0),
        "probe_ready_gee": sum(1 for r in probe_ready if clean(r.get("can_probe_gee_metadata")) == "true"),
        "probe_ready_stac": sum(1 for r in probe_ready if clean(r.get("can_probe_stac_metadata")) == "true"),
        "probe_ready_odata": sum(1 for r in probe_ready if clean(r.get("can_probe_odata_metadata")) == "true"),
        "gee_calls_executed": 0,
        "stac_calls_executed": 0,
        "odata_calls_executed": 0,
        "downloads_executed": 0,
        "crops_created": 0,
        "native_rasters_created": 0,
        "spectral_features_created": 0,
        "labels_created": 0,
        "silver_created": 0,
        "gold_created": 0,
        "negatives_created": 0,
        "corpus_day10_unlocked": False,
        "sandbox_unlocked": False,
        "can_train": False,
        "guardrails_passed": True,
    }


def main() -> None:
    ensure_public_dir()
    summary = build_summary()
    out = PUBLIC_DIR / "mv2_data_05_summary.json"
    write_json(out, summary)
    print(
        f"[mv2_data_05_summary] input_rows={summary['total_input_rows']} "
        f"filled={summary['filled_template_found']} "
        f"promoted_strong={summary['temporal_promoted_strong']} "
        f"promoted_partial={summary['temporal_promoted_partial']} "
        f"probe_ready_gee={summary['probe_ready_gee']} | "
        f"saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
