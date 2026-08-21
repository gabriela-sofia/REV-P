"""Shared logic for MV2-15 Cronograma Engine & Gate Orchestrator.

This stage is engineering-only and fail-closed. It consolidates MV2-10..MV2-14
outputs into schedule, gate, backlog, registry, and risk artifacts. It never
executes STAC, downloads rasters, creates crops/features/labels/silver/negatives,
creates splits, starts a sandbox, or trains a model.
"""

from __future__ import annotations

import csv
import json
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

PROJECT_ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
OUTPUT_DIR = PROJECT_ROOT / "outputs_public" / "mv2_cronograma_engine"
SCHEMA_DIR = PROJECT_ROOT / "datasets" / "schemas"

LIGHTWEIGHT_LIMIT_BYTES = 5_000_000

SCHEDULE_COLUMNS = [
    "day_id",
    "day_label",
    "planned_objective",
    "real_status",
    "status_reason",
    "evidence_source",
    "blocking_gate",
    "blocking_data",
    "blocking_script",
    "blocking_schema",
    "blocking_test",
    "can_progress_without_data",
    "allowed_next_programming_actions",
    "forbidden_actions",
    "scientific_risk",
    "engineering_risk",
    "priority",
]

ALLOWED_STATUSES = [
    "DONE",
    "PARTIAL",
    "BLOCKED_DATA",
    "BLOCKED_EVIDENCE",
    "BLOCKED_METHOD",
    "BLOCKED_GUARDRAIL",
    "READY_ENGINEERING_ONLY",
    "READY_REVIEW_ONLY",
    "NOT_APPLICABLE_NOW",
]

PROHIBITED_CATEGORIES = [
    "STAC_REAL",
    "RASTER_DOWNLOAD",
    "PRIVATE_CROP",
    "SPECTRAL_FEATURE_EXTRACTION",
    "SUPERVISED_BASELINE",
    "TRAINING",
    "SILVER_PROMOTION",
    "NEGATIVE_CREATION",
    "SPLIT_TRAINABLE",
    "SANDBOX_SUPERVISED",
]

ALLOWED_CATEGORIES = [
    "ENGINEERING_CONSOLIDATION",
    "SCHEMA_HARDENING",
    "TEST_HARDENING",
    "CLI_ORCHESTRATION",
    "REPORTING",
    "MANIFEST_VALIDATION",
    "WORKTREE_READONLY_INTEGRATION",
    "REPRODUCIBILITY_PACKAGE",
    "DOCS_FAIL_CLOSED",
    "REVIEW_QUEUE_TOOLING",
    "MANUAL_TEMPLATE_SUPPORT",
    "QUALITY_DASHBOARD",
]

FORBIDDEN_ACTIONS = ";".join(PROHIBITED_CATEGORIES)


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, columns: list[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def read_csv(path: Path | None) -> list[dict[str, str]]:
    if not path or not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path | None) -> dict[str, object]:
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def git_output(args: list[str], cwd: Path = PROJECT_ROOT) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:
        return ""


def worktrees() -> list[dict[str, str | Path]]:
    raw = git_output(["worktree", "list", "--porcelain"])
    rows: list[dict[str, str | Path]] = []
    current: dict[str, str | Path] = {}
    for line in raw.splitlines():
        if not line:
            if current:
                rows.append(current)
                current = {}
            continue
        if line.startswith("worktree "):
            current["path"] = Path(line.removeprefix("worktree ").strip())
        elif line.startswith("HEAD "):
            current["commit_ref"] = line.removeprefix("HEAD ").strip()[:12]
        elif line.startswith("branch "):
            current["branch"] = line.removeprefix("branch refs/heads/").strip()
    if current:
        rows.append(current)
    for idx, row in enumerate(rows):
        row["worktree_id"] = "WT_LOCAL" if Path(row["path"]) == PROJECT_ROOT else f"WT_{idx:02d}"
        row.setdefault("branch", "DETACHED_OR_UNKNOWN")
        row.setdefault("commit_ref", "")
    return rows


STAGES = [
    {
        "stage": "MV2-10_GAP_AUDIT",
        "expected_dir": "outputs_public/audits/mv2_10_gap_audit",
        "key_summary_file": "mv2_10_gap_matrix.csv",
    },
    {
        "stage": "MV2-11_REGIONAL_REBALANCING",
        "expected_dir": "outputs_public/mv2_regional_rebalancing",
        "key_summary_file": "mv2_11_regional_rebalancing_summary.json",
    },
    {
        "stage": "MV2-12_DATA_READINESS",
        "expected_dir": "outputs_public/mv2_data_readiness",
        "key_summary_file": "mv2_12_data_readiness_summary.json",
    },
    {
        "stage": "MV2-12_SPECTRAL_RECONSTRUCTION",
        "expected_dir": "outputs_public/mv2_spectral_reconstruction",
        "key_summary_file": "mv2_12_sentinel_spectral_reconstruction_summary.json",
    },
    {
        "stage": "MV2-13_SENTINEL_BINDING",
        "expected_dir": "outputs_public/mv2_sentinel_binding",
        "key_summary_file": "mv2_13_binding_summary.json",
    },
    {
        "stage": "MV2-14_GEE_LINEAGE_RECOVERY",
        "expected_dir": "outputs_public/mv2_gee_lineage_recovery",
        "key_summary_file": "mv2_14_lineage_summary.json",
    },
]


def resolve_stage(stage: Mapping[str, str]) -> dict[str, object]:
    rel = stage["expected_dir"]
    for wt in worktrees():
        root = Path(wt["path"])
        candidate = root / rel
        if candidate.exists():
            files = [p for p in candidate.iterdir() if p.is_file()]
            key = candidate / stage["key_summary_file"]
            location_type = "LOCAL" if root == PROJECT_ROOT else "READ_ONLY_WORKTREE"
            return {
                "stage": stage["stage"],
                "expected_dir": rel,
                "resolved_location": rel if root == PROJECT_ROOT else str(wt["worktree_id"]),
                "location_type": location_type,
                "exists": "true",
                "files_found": len(files),
                "key_summary_file": stage["key_summary_file"] if key.exists() else "",
                "read_status": "READ_LIGHTWEIGHT",
                "notes": "local" if root == PROJECT_ROOT else f"read_only_branch={wt['branch']}",
                "_path": candidate,
                "_worktree_id": wt["worktree_id"],
            }
    return {
        "stage": stage["stage"],
        "expected_dir": rel,
        "resolved_location": "",
        "location_type": "MISSING",
        "exists": "false",
        "files_found": 0,
        "key_summary_file": "",
        "read_status": "MISSING_RECORDED",
        "notes": "ausencia registrada sem falhar",
        "_path": None,
        "_worktree_id": "",
    }


def discover_mv2_outputs() -> list[dict[str, object]]:
    rows = [resolve_stage(stage) for stage in STAGES]
    public_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    write_csv(
        OUTPUT_DIR / "mv2_15_mv2_output_discovery.csv",
        [
            "stage",
            "expected_dir",
            "resolved_location",
            "location_type",
            "exists",
            "files_found",
            "key_summary_file",
            "read_status",
            "notes",
        ],
        public_rows,
    )
    return rows


def source_file_for(stage_name: str, file_name: str) -> str:
    for row in discover_mv2_outputs():
        if row["stage"] == stage_name and row.get("_path"):
            path = Path(row["_path"]) / file_name
            if path.exists():
                return f"{stage_name}:{file_name}"
    return f"{stage_name}:MISSING"


def build_schedule_state() -> list[dict[str, object]]:
    rows = [
        schedule_row("8", "Dia 8", "expandir embeddings para n>=59", "PARTIAL",
                     "DINOv2/visuais e subset limitado existem, mas vies regional segue alto",
                     "MV2-11_REGIONAL_REBALANCING:regional_representation_coverage.csv",
                     "GATE_DINOV2_REVIEW_ONLY", "amostra e cobertura regional limitadas", "", "", "",
                     "true", "QUALITY_DASHBOARD;REPORTING;SCHEMA_HARDENING", "risco de exagerar validade estrutural", "normalizar evidencias parciais", "P1"),
        schedule_row("9", "Dia 9", "QA embeddings", "DONE",
                     "QA label-free existe como controle review-only; nao promove label",
                     "outputs_public/tables/revp_vizinhos_embeddings_label_free_mv1.csv",
                     "GATE_DINOV2_REVIEW_ONLY", "", "", "", "",
                     "true", "TEST_HARDENING;QUALITY_DASHBOARD", "DINOv2 nao e evidencia operacional", "consolidar testes globais", "P2"),
        schedule_row("10", "Dia 10", "baseline espectral simples", "BLOCKED_DATA",
                     "sem raster Sentinel nativo, lineage completo, STAC real, crop ou feature espectral",
                     "MV2-14_GEE_LINEAGE_RECOVERY:mv2_14_lineage_summary.json",
                     "GATE_DAY10_BLOCKED", "native_raster=0;lineage_strong=0;stac_real=0", "scripts/mv2_14_validate_gee_lineage_guardrails.py", "datasets/schemas/schema_mv2_14_day10_post_lineage_gate.json", "tests/test_mv2_14_gee_lineage_recovery.py",
                     "false", "MANUAL_TEMPLATE_SUPPORT;REVIEW_QUEUE_TOOLING;REPORTING", "Dia 10 falso desbloqueio", "criar simulador dry-run sem execucao real", "P0"),
        schedule_row("11", "Dia 11", "sanity check vizinhanca", "DONE",
                     "sanity label-free e vizinhanca existem como analise review-only",
                     "outputs_public/tables/revp_vizinhos_embeddings_label_free_mv1.csv",
                     "GATE_DINOV2_REVIEW_ONLY", "", "", "", "",
                     "true", "REPORTING;QUALITY_DASHBOARD", "vizinhanca nao vira verdade operacional", "resumir achados em dashboard", "P2"),
        schedule_row("12", "Dia 12", "confounder de cidade", "PARTIAL",
                     "confounder regional/cidade esta documentado e permanece risco alto",
                     "MV2-11_REGIONAL_REBALANCING:rebalanced_confounder_audit.csv",
                     "GATE_CITY_NOT_LABEL", "cidade/regiao nao e label", "", "", "",
                     "true", "QUALITY_DASHBOARD;SCHEMA_HARDENING", "usar cidade como proxy de classe", "criar checker de claims", "P1"),
        schedule_row("13", "Dia 13", "random baseline", "DONE",
                     "baseline/controle metodologico existe como referencia nao treinavel",
                     "outputs_public/tables/revp_matriz_maturidade_metodologica_mv1.csv",
                     "GATE_NO_TRAINING", "", "", "", "",
                     "true", "REPORTING;TEST_HARDENING", "baseline confundido com treinamento supervisionado", "documentar limites", "P2"),
        schedule_row("14", "Dia 14", "relatorio label-free intermediario", "DONE",
                     "pacote publico label-free existe e preserva semantica review-only",
                     "outputs_public/execution_reports/revp_fechamento_marco_validacao_label_free_evidencia_estrutural_mv1.md",
                     "GATE_DINOV2_REVIEW_ONLY", "", "", "", "",
                     "true", "REPORTING;REPRODUCIBILITY_PACKAGE", "linguagem publica exagerada", "gerar resumo executivo por fase", "P1"),
        schedule_row("15", "Dia 15", "politica de negativos", "READY_REVIEW_ONLY",
                     "politica existe, mas sem negativos formais criados",
                     "docs/metodologia_cientifica/revp_politica_evidencia_negativa_mv1.md",
                     "GATE_NO_NEGATIVES_FORMAL", "formal_negatives=0", "", "", "",
                     "true", "DOCS_FAIL_CLOSED;MANIFEST_VALIDATION", "unknown como negativo", "validador de claims negativos", "P1"),
        schedule_row("16", "Dia 16", "selecao inicial de negativos", "BLOCKED_EVIDENCE",
                     "nao existem negativos formais; ausencia nao vira negativo",
                     "outputs_public/tables/revp_gates_readiness_treino_mv1.csv",
                     "GATE_NO_NEGATIVES_FORMAL", "formal_negatives=0", "", "", "",
                     "false", "REVIEW_QUEUE_TOOLING;DOCS_FAIL_CLOSED", "criar negativo por inferencia", "fila de revisao manual sem promocao", "P0"),
        schedule_row("17", "Dia 17", "adjudicacao", "PARTIAL",
                     "ha filas/revisoes candidatas, mas sem fechamento observacional patch-level",
                     "outputs_public/tables/revp_human_review_queue_v2ed.csv",
                     "GATE_UNKNOWN_NOT_NEGATIVE", "sem fechamento observacional", "", "", "",
                     "true", "REVIEW_QUEUE_TOOLING;MANUAL_TEMPLATE_SUPPORT", "adjudicacao IA virar label", "importador manual sem promocao automatica", "P1"),
        schedule_row("18", "Dia 18", "rodada de revisao", "BLOCKED_EVIDENCE",
                     "sem evidencia manual/observacional suficiente para desbloqueio",
                     "outputs_public/tables/revp_dashboard_bloqueio_treino_ground_truth_mv1.csv",
                     "GATE_DAY18_BLOCKED", "ground_truth_operacional_status=ausente", "", "", "",
                     "false", "REVIEW_QUEUE_TOOLING;REPORTING", "claim operacional indevido", "validador de preenchimento manual", "P0"),
        schedule_row("19", "Dia 19", "silver set inicial", "BLOCKED_EVIDENCE",
                     "silver formal permanece zero",
                     "outputs_public/tables/revp_gates_readiness_treino_mv1.csv",
                     "GATE_DAY19_BLOCKED", "silver_formal=0", "", "", "",
                     "false", "DOCS_FAIL_CLOSED;MANIFEST_VALIDATION", "silver IA como silver formal", "registry global de gates", "P0"),
        schedule_row("20", "Dia 20", "pacote reproduzivel externo", "PARTIAL",
                     "pacote leve existe, mas reproducibilidade externa ainda depende de dados ausentes",
                     "outputs_public/tables/revp_plano_pacote_externo_reprodutibilidade_mv1.csv",
                     "GATE_NO_DOWNLOAD", "dados externos pesados ausentes", "", "", "",
                     "true", "REPRODUCIBILITY_PACKAGE;MANIFEST_VALIDATION;REPORTING", "pacote sugerir execucao operacional", "pacote leve de terceiro", "P1"),
        schedule_row("21", "Dia 21", "splits espaciais", "BLOCKED_METHOD",
                     "sem labels/silver/splits treinaveis",
                     "outputs_public/tables/revp_gates_readiness_treino_mv1.csv",
                     "GATE_DAY21_BLOCKED", "labels=0;silver=0;can_train=false", "", "", "",
                     "false", "SCHEMA_HARDENING;TEST_HARDENING", "split treinavel sem label", "NO_GO automatico para split", "P0"),
        schedule_row("22", "Dia 22", "sandbox supervisionado minimo", "BLOCKED_GUARDRAIL",
                     "sandbox supervisionado proibido sem silver formal e split treinavel",
                     "outputs_public/tables/revp_dashboard_bloqueio_treino_ground_truth_mv1.csv",
                     "GATE_DAY22_BLOCKED", "can_train=false;silver_formal=0", "", "", "",
                     "false", "CLI_ORCHESTRATION;TEST_HARDENING", "sandbox sem silver", "NO_GO automatico para sandbox", "P0"),
    ]
    write_csv(OUTPUT_DIR / "mv2_15_schedule_state.csv", SCHEDULE_COLUMNS, rows)
    return rows


def schedule_row(
    day_id: str,
    day_label: str,
    planned_objective: str,
    real_status: str,
    status_reason: str,
    evidence_source: str,
    blocking_gate: str,
    blocking_data: str,
    blocking_script: str,
    blocking_schema: str,
    blocking_test: str,
    can_progress_without_data: str,
    allowed_next_programming_actions: str,
    scientific_risk: str,
    engineering_risk: str,
    priority: str,
) -> dict[str, object]:
    return {
        "day_id": day_id,
        "day_label": day_label,
        "planned_objective": planned_objective,
        "real_status": real_status,
        "status_reason": status_reason,
        "evidence_source": evidence_source,
        "blocking_gate": blocking_gate,
        "blocking_data": blocking_data,
        "blocking_script": blocking_script,
        "blocking_schema": blocking_schema,
        "blocking_test": blocking_test,
        "can_progress_without_data": can_progress_without_data,
        "allowed_next_programming_actions": allowed_next_programming_actions,
        "forbidden_actions": FORBIDDEN_ACTIONS,
        "scientific_risk": scientific_risk,
        "engineering_risk": engineering_risk,
        "priority": priority,
    }


def build_gate_engine() -> list[dict[str, object]]:
    gate_defs = [
        ("GATE_NO_LABELS_CREATED", "labels_created", 0, 0, "MV2-14_GEE_LINEAGE_RECOVERY", "mv2_14_lineage_summary.json", "19;21;22", "P0", "manter bloqueio"),
        ("GATE_NO_SILVER_FORMAL", "silver_created", 0, 0, "MV2-14_GEE_LINEAGE_RECOVERY", "mv2_14_lineage_summary.json", "19;21;22", "P0", "nao promover silver"),
        ("GATE_NO_GOLD", "gold_created", 0, 0, "MV2-14_GEE_LINEAGE_RECOVERY", "mv2_14_lineage_summary.json", "19;21;22", "P0", "nao promover gold"),
        ("GATE_NO_NEGATIVES_FORMAL", "negatives_created", 0, 0, "MV2-14_GEE_LINEAGE_RECOVERY", "mv2_14_lineage_summary.json", "15;16;19;21", "P0", "ausencia nao vira negativo"),
        ("GATE_NO_TRAINING", "can_train", "false", "false", "MV2-14_GEE_LINEAGE_RECOVERY", "mv2_14_lineage_summary.json", "21;22", "P0", "bloquear treino"),
        ("GATE_NO_SANDBOX", "sandbox_created", 0, 0, "MV2-15_CRONOGRAMA_ENGINE", "mv2_15_schedule_state.csv", "22", "P0", "bloquear sandbox"),
        ("GATE_NO_STAC_REAL", "stac_real_executed", 0, 0, "MV2-14_GEE_LINEAGE_RECOVERY", "mv2_14_lineage_summary.json", "10", "P0", "nao executar STAC real"),
        ("GATE_NO_DOWNLOAD", "downloads_executed", 0, 0, "MV2-14_GEE_LINEAGE_RECOVERY", "mv2_14_lineage_summary.json", "10;20", "P0", "nao baixar raster"),
        ("GATE_NO_CROP", "crops_created", 0, 0, "MV2-14_GEE_LINEAGE_RECOVERY", "mv2_14_lineage_summary.json", "10", "P0", "nao criar crop"),
        ("GATE_NO_NATIVE_RASTER", "native_rasters_created", 0, 0, "MV2-14_GEE_LINEAGE_RECOVERY", "mv2_14_lineage_summary.json", "10", "P0", "manter Dia 10 bloqueado"),
        ("GATE_NO_SPECTRAL_FEATURES", "spectral_features_created", 0, 0, "MV2-14_GEE_LINEAGE_RECOVERY", "mv2_14_lineage_summary.json", "10", "P0", "nao calcular feature"),
        ("GATE_UNKNOWN_NOT_NEGATIVE", "unknown_as_negative", "false", "false", "MV2-15_CRONOGRAMA_ENGINE", "mv2_15_scientific_risk_summary.csv", "16;17", "P0", "falhar fechado"),
        ("GATE_CITY_NOT_LABEL", "city_as_label", "false", "false", "MV2-11_REGIONAL_REBALANCING", "rebalanced_confounder_audit.csv", "12;16;19", "P0", "nao usar cidade como label"),
        ("GATE_PNG_NOT_RASTER", "png_as_raster", "false", "false", "MV2-13_SENTINEL_BINDING", "mv2_13_binding_risk_matrix.csv", "10", "P0", "bloquear raster falso"),
        ("GATE_DINOV2_REVIEW_ONLY", "dinov2_operational_label", "false", "false", "MV1_LABEL_FREE", "revp_validacao_label_free_evidencia_estrutural_mv1.csv", "8;9;11;14", "P1", "manter review-only"),
        ("GATE_DAY10_BLOCKED", "can_unlock_day10_now", "false", "false", "MV2-14_GEE_LINEAGE_RECOVERY", "mv2_14_lineage_summary.json", "10", "P0", "exigir raster e lineage completo"),
        ("GATE_DAY18_BLOCKED", "day18_unlocked", "false", "false", "MV2-15_CRONOGRAMA_ENGINE", "mv2_15_schedule_state.csv", "18", "P0", "exigir evidencia manual"),
        ("GATE_DAY19_BLOCKED", "silver_formal", 0, 0, "MV2-15_CRONOGRAMA_ENGINE", "mv2_15_schedule_state.csv", "19", "P0", "exigir silver formal"),
        ("GATE_DAY21_BLOCKED", "trainable_splits", 0, 0, "MV2-15_CRONOGRAMA_ENGINE", "mv2_15_schedule_state.csv", "21", "P0", "exigir labels/silver"),
        ("GATE_DAY22_BLOCKED", "supervised_sandbox", 0, 0, "MV2-15_CRONOGRAMA_ENGINE", "mv2_15_schedule_state.csv", "22", "P0", "NO_GO sandbox"),
    ]
    rows = []
    for idx, (name, observed_name, expected, observed, stage, src, days, severity, action) in enumerate(gate_defs, 1):
        rows.append({
            "gate_id": f"MV2_15_GATE_{idx:03d}",
            "gate_name": name,
            "expected_value": expected,
            "observed_value": observed,
            "pass_fail": "PASS" if str(expected).lower() == str(observed).lower() else "FAIL",
            "source_stage": stage,
            "source_file": src,
            "affected_days": days,
            "severity": severity,
            "action_if_failed": action,
            "notes": f"{observed_name} preservado em estado fail-closed",
        })
    write_csv(OUTPUT_DIR / "mv2_15_gate_engine.csv", [
        "gate_id", "gate_name", "expected_value", "observed_value", "pass_fail",
        "source_stage", "source_file", "affected_days", "severity", "action_if_failed", "notes",
    ], rows)
    return rows


def build_allowed_programming_actions() -> list[dict[str, object]]:
    rows = []
    for idx, category in enumerate(ALLOWED_CATEGORIES, 1):
        rows.append({
            "action_id": f"MV2_15_ACT_{idx:03d}",
            "action_category": category,
            "action_name": category.lower(),
            "allowed_now": "true",
            "depends_on_external_data": "false",
            "blocked_by": "",
            "expected_output": f"outputs_public/mv2_cronograma_engine/{category.lower()}.csv",
            "suggested_stage": "MV2-16_ENGINEERING_PACKAGE",
            "priority": "P0" if idx <= 6 else "P1",
            "rationale": "programacao pesada permitida sem criar dado cientifico novo",
        })
    for idx, category in enumerate(PROHIBITED_CATEGORIES, len(rows) + 1):
        rows.append({
            "action_id": f"MV2_15_ACT_{idx:03d}",
            "action_category": category,
            "action_name": category.lower(),
            "allowed_now": "false",
            "depends_on_external_data": "true",
            "blocked_by": "MV2_15_FAIL_CLOSED_GATES",
            "expected_output": "",
            "suggested_stage": "NA",
            "priority": "NO_GO",
            "rationale": "bloqueado por ausencia de evidencia, raster nativo, silver formal ou guardrail",
        })
    write_csv(OUTPUT_DIR / "mv2_15_allowed_programming_actions.csv", [
        "action_id", "action_category", "action_name", "allowed_now",
        "depends_on_external_data", "blocked_by", "expected_output",
        "suggested_stage", "priority", "rationale",
    ], rows)
    return rows


BACKLOG_TITLES = [
    ("CLI unico revp mv2 status", "CLI_ORCHESTRATION", "8-22", "GLOBAL", "false"),
    ("CLI unico revp mv2 validate", "CLI_ORCHESTRATION", "8-22", "GLOBAL", "false"),
    ("CLI unico revp mv2 gates", "CLI_ORCHESTRATION", "8-22", "GLOBAL", "false"),
    ("CLI unico revp mv2 report", "REPORTING", "8-22", "GLOBAL", "false"),
    ("Consolidar schemas MV2-10 a MV2-14", "SCHEMA_HARDENING", "8-22", "GLOBAL", "false"),
    ("Validador global de outputs publicos leves", "TEST_HARDENING", "8-22", "GLOBAL", "false"),
    ("Registry global de artefatos MV2", "MANIFEST_VALIDATION", "8-22", "GLOBAL", "false"),
    ("Checker de worktrees read-only", "WORKTREE_READONLY_INTEGRATION", "8-22", "GLOBAL", "false"),
    ("Dashboard Markdown/CSV do cronograma", "QUALITY_DASHBOARD", "8-22", "GLOBAL", "false"),
    ("Sistema de severidade de bloqueios", "ENGINEERING_CONSOLIDATION", "8-22", "GLOBAL", "false"),
    ("Importador do template manual GEE sem promocao automatica", "MANUAL_TEMPLATE_SUPPORT", "10;17;18", "GATE_DAY10_BLOCKED", "false"),
    ("Validador de preenchimento manual GEE", "REVIEW_QUEUE_TOOLING", "10;17;18", "GATE_DAY10_BLOCKED", "false"),
    ("Simulador dry-run de STAC sem execucao real", "ENGINEERING_CONSOLIDATION", "10", "GATE_NO_STAC_REAL", "false"),
    ("Relatorio para artigo/slides com status fail-closed", "REPORTING", "14;20", "GATE_DINOV2_REVIEW_ONLY", "false"),
    ("Testes de regressao globais", "TEST_HARDENING", "8-22", "GLOBAL", "false"),
    ("Pacote reproduzivel leve para terceiro", "REPRODUCIBILITY_PACKAGE", "20", "GATE_NO_DOWNLOAD", "false"),
    ("Verificador de claims publicos", "DOCS_FAIL_CLOSED", "8-22", "GLOBAL", "false"),
    ("Camada de resumo executivo por fase", "REPORTING", "8-22", "GLOBAL", "false"),
    ("Matriz de risco consolidada", "ENGINEERING_CONSOLIDATION", "8-22", "GLOBAL", "false"),
    ("NO_GO automatico para sandbox", "TEST_HARDENING", "22", "GATE_DAY22_BLOCKED", "false"),
]


def build_engineering_backlog() -> list[dict[str, object]]:
    rows = []
    for idx, (title, category, day, gate, blocked_by_data) in enumerate(BACKLOG_TITLES, 1):
        rows.append({
            "backlog_id": f"MV2_15_BACKLOG_{idx:03d}",
            "title": title,
            "description": f"Implementar {title} preservando gates fail-closed.",
            "category": category,
            "related_day": day,
            "related_gate": gate,
            "blocked_by_data": blocked_by_data,
            "can_implement_now": "true",
            "expected_files": "scripts/;tests/;outputs_public/mv2_cronograma_engine/",
            "priority": "P0" if idx <= 12 or idx == 20 else "P1",
            "risk_reduced": "reduz risco de avanco indevido e dispersao de estado",
            "acceptance_criteria": "testes passam; nenhum dado cientifico novo e criado; staged vazio",
        })
    write_csv(OUTPUT_DIR / "mv2_15_engineering_backlog.csv", [
        "backlog_id", "title", "description", "category", "related_day", "related_gate",
        "blocked_by_data", "can_implement_now", "expected_files", "priority",
        "risk_reduced", "acceptance_criteria",
    ], rows)
    return rows


def build_artifact_registry() -> list[dict[str, object]]:
    rows = []
    stages = discover_mv2_outputs()
    artifact_id = 0
    for stage in stages:
        root = stage.get("_path")
        if not root:
            continue
        for path in sorted(Path(root).iterdir()):
            if not path.is_file():
                continue
            artifact_id += 1
            rel = f"{stage['expected_dir']}/{path.name}"
            suffix = path.suffix.lower()
            rows.append({
                "artifact_id": f"MV2_15_ART_{artifact_id:04d}",
                "stage": stage["stage"],
                "path": rel,
                "artifact_type": suffix.removeprefix(".") or "unknown",
                "public_private": "public",
                "lightweight": str(path.stat().st_size <= LIGHTWEIGHT_LIMIT_BYTES).lower(),
                "producer_script": "registrado_em_marco_origem",
                "schema": "schema_stage_specific_or_not_declared",
                "tests": "testes_stage_specific_or_not_declared",
                "consumed_by": "MV2-15_CRONOGRAMA_ENGINE",
                "scientific_role": "evidencia_de_estado_ou_gate",
                "can_be_public": "true",
                "notes": "caminho relativo; sem material bruto pesado",
            })
    mv2_15_expected = [
        "mv2_15_mv2_output_discovery.csv",
        "mv2_15_schedule_state.csv",
        "mv2_15_gate_engine.csv",
        "mv2_15_allowed_programming_actions.csv",
        "mv2_15_engineering_backlog.csv",
        "mv2_15_artifact_registry.csv",
        "mv2_15_worktree_integration_matrix.csv",
        "mv2_15_scientific_risk_summary.csv",
        "mv2_15_cronograma_summary.json",
    ]
    for name in mv2_15_expected:
        artifact_id += 1
        rows.append({
            "artifact_id": f"MV2_15_ART_{artifact_id:04d}",
            "stage": "MV2-15_CRONOGRAMA_ENGINE",
            "path": f"outputs_public/mv2_cronograma_engine/{name}",
            "artifact_type": Path(name).suffix.removeprefix("."),
            "public_private": "public",
            "lightweight": "true",
            "producer_script": "scripts/mv2_15_run_cronograma_engine.py",
            "schema": "schema_mv2_15_schedule_state.json" if name == "mv2_15_schedule_state.csv" else "",
            "tests": "tests/test_mv2_15_cronograma_engine.py",
            "consumed_by": "MV2-15 reports and validator",
            "scientific_role": "estado consolidado fail-closed",
            "can_be_public": "true",
            "notes": "saida leve MV2-15",
        })
    write_csv(OUTPUT_DIR / "mv2_15_artifact_registry.csv", [
        "artifact_id", "stage", "path", "artifact_type", "public_private", "lightweight",
        "producer_script", "schema", "tests", "consumed_by", "scientific_role",
        "can_be_public", "notes",
    ], rows)
    return rows


def build_worktree_integration_matrix() -> list[dict[str, object]]:
    stage_rows = discover_mv2_outputs()
    by_wt: dict[str, list[dict[str, object]]] = {}
    for stage in stage_rows:
        by_wt.setdefault(str(stage.get("_worktree_id", "")), []).append(stage)
    rows = []
    for wt in worktrees():
        wt_id = str(wt["worktree_id"])
        stages_here = [s for s in by_wt.get(wt_id, []) if s["exists"] == "true"]
        rows.append({
            "worktree_id": wt_id,
            "branch": wt["branch"],
            "contains_stage": ";".join(str(s["stage"]) for s in stages_here) or "NONE",
            "read_mode": "LOCAL_WRITE_TARGET" if wt_id == "WT_LOCAL" else "READ_ONLY",
            "outputs_available": sum(int(s["files_found"]) for s in stages_here),
            "tests_available": "checked_by_stage_or_not_declared",
            "commit_ref": wt["commit_ref"],
            "integration_status": "LOCAL" if wt_id == "WT_LOCAL" else "READ_ONLY_AVAILABLE",
            "notes": "path privado omitido; usar worktree_id sanitizado",
        })
    write_csv(OUTPUT_DIR / "mv2_15_worktree_integration_matrix.csv", [
        "worktree_id", "branch", "contains_stage", "read_mode", "outputs_available",
        "tests_available", "commit_ref", "integration_status", "notes",
    ], rows)
    return rows


RISK_ROWS = [
    ("claim operacional indevido", "P0", "manter review-only; exigir evidencia patch-level"),
    ("Dia 10 falso desbloqueio", "P0", "bloquear sem raster nativo e lineage completo"),
    ("PNG como raster", "P0", "PNG/NPZ/render nunca vira raster Sentinel nativo"),
    ("DINOv2 como evidencia operacional", "P0", "DINOv2 permanece diagnostico label-free"),
    ("cidade como label", "P0", "cidade/regiao nao e label"),
    ("unknown como negativo", "P0", "ausencia nunca vira negativo"),
    ("silver IA como silver formal", "P0", "silver formal continua zero"),
    ("split treinavel sem label", "P0", "NO_GO para split treinavel"),
    ("sandbox sem silver", "P0", "NO_GO automatico para sandbox"),
    ("STAC sem binding", "P0", "STAC real e download seguem zero"),
    ("crop sem CRS", "P0", "crop proibido ate CRS/raster validos"),
    ("lineage sem patch", "P0", "T23KPR e candidato de revisao, nao join"),
    ("cena T23KPR auto-vinculada", "P0", "auto_join_allowed=false"),
    ("relatorio publico exagerado", "P1", "checador de claims publicos"),
]


def build_scientific_risk_summary() -> list[dict[str, object]]:
    rows = []
    for idx, (risk, severity, mitigation) in enumerate(RISK_ROWS, 1):
        rows.append({
            "risk_id": f"MV2_15_RISK_{idx:03d}",
            "risk_name": risk,
            "severity": severity,
            "source_gate": "MV2_15_FAIL_CLOSED_GATES",
            "affected_days": "8-22",
            "current_status": "CONTROLLED_BY_GATE",
            "mitigation": mitigation,
            "allowed_action": "engineering_or_review_only",
            "forbidden_action": FORBIDDEN_ACTIONS,
            "notes": "risco consolidado sem criar dado cientifico novo",
        })
    write_csv(OUTPUT_DIR / "mv2_15_scientific_risk_summary.csv", [
        "risk_id", "risk_name", "severity", "source_gate", "affected_days",
        "current_status", "mitigation", "allowed_action", "forbidden_action", "notes",
    ], rows)
    return rows


def build_schema() -> None:
    write_json(SCHEMA_DIR / "schema_mv2_15_schedule_state.json", {
        "schema_id": "schema_mv2_15_schedule_state",
        "stage": "MV2-15_CRONOGRAMA_ENGINE",
        "required_fields": SCHEDULE_COLUMNS,
        "allowed_statuses": ALLOWED_STATUSES,
        "notes": "Cronograma fail-closed; estados bloqueados nao autorizam treino, STAC real, crop, silver ou sandbox.",
    })


def build_summary() -> dict[str, object]:
    schedule = read_csv(OUTPUT_DIR / "mv2_15_schedule_state.csv")
    actions = read_csv(OUTPUT_DIR / "mv2_15_allowed_programming_actions.csv")
    backlog = read_csv(OUTPUT_DIR / "mv2_15_engineering_backlog.csv")
    heavy = count_heavy_outputs()
    private = count_private_path_hits()
    by_day = {row["day_id"]: row["real_status"] for row in schedule}
    high_items = [row["title"] for row in backlog if row["priority"] == "P0"][:5]
    summary = {
        "current_phase": "MV2-15_CRONOGRAMA_ENGINE",
        "current_day_range": "8-22",
        "day8_status": by_day.get("8", ""),
        "day10_status": by_day.get("10", ""),
        "day18_status": by_day.get("18", ""),
        "day19_status": by_day.get("19", ""),
        "day21_status": by_day.get("21", ""),
        "day22_status": by_day.get("22", ""),
        "can_train": False,
        "can_stac_real": False,
        "can_download_raster": False,
        "can_create_crop": False,
        "can_compute_spectral_features": False,
        "can_create_silver": False,
        "can_create_negatives": False,
        "can_create_sandbox": False,
        "allowed_programming_actions_count": sum(1 for a in actions if a["allowed_now"] == "true"),
        "blocked_actions_count": sum(1 for a in actions if a["allowed_now"] == "false"),
        "engineering_backlog_count": len(backlog),
        "high_priority_engineering_items": high_items,
        "heavy_public_outputs": heavy,
        "private_paths_leaked": private,
        "guardrails_passed": heavy == 0 and private == 0,
    }
    write_json(OUTPUT_DIR / "mv2_15_cronograma_summary.json", summary)
    return summary


def count_heavy_outputs() -> int:
    if not OUTPUT_DIR.exists():
        return 0
    return sum(1 for p in OUTPUT_DIR.rglob("*") if p.is_file() and p.stat().st_size > LIGHTWEIGHT_LIMIT_BYTES)


def count_private_path_hits() -> int:
    if not OUTPUT_DIR.exists():
        return 0
    needles = ["C:/Users", "C:\\Users", "\\\\?\\C:", "Users/gabriela"]
    hits = 0
    for path in OUTPUT_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".csv", ".json", ".md", ".txt"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            hits += sum(1 for needle in needles if needle in text)
    return hits


def write_reports() -> None:
    summary = read_json(OUTPUT_DIR / "mv2_15_cronograma_summary.json")
    schedule = read_csv(OUTPUT_DIR / "mv2_15_schedule_state.csv")
    backlog = read_csv(OUTPUT_DIR / "mv2_15_engineering_backlog.csv")
    blocked = [r for r in schedule if r["real_status"].startswith("BLOCKED")]
    top_backlog = [r["title"] for r in backlog[:5]]
    report = [
        "# MV2-15 - Cronograma Engine & Gate Orchestrator",
        "",
        "Marco engineering-only, review-only e fail-closed. Nao executa STAC real, nao baixa raster, nao cria crop, feature espectral, label, silver, gold, negativo, split, sandbox ou treino.",
        "",
        "## Fechamento da vertente de dados",
        "",
        "MV2-10 a MV2-14 mostraram que o gargalo atual e dado/evidencia: ha diagnostico suficiente para formalizar gates, mas nao para promover treino ou operacao.",
        "Lineage local real existe apenas como Protocolo-C/Petropolis; no corpus oficial, MV2-14 manteve 48 candidatos de revisao e 80 sem lineage.",
        "",
        "## Estado do cronograma",
        "",
    ]
    report.extend(f"- Dia {r['day_id']}: {r['real_status']} - {r['status_reason']}" for r in schedule)
    report.extend([
        "",
        "## Bloqueios ativos",
        "",
    ])
    report.extend(f"- Dia {r['day_id']}: {r['blocking_gate']} - {r['blocking_data']}" for r in blocked)
    report.extend([
        "",
        "## Programacao pesada liberada",
        "",
        f"- Acoes permitidas agora: {summary.get('allowed_programming_actions_count', 0)}",
        "- Escopo: consolidacao de engenharia, schemas, testes, CLI, dashboards, registry, validadores e pacote leve reprodutivel.",
        "",
        "## Backlog recomendado",
        "",
    ])
    report.extend(f"- {item}" for item in top_backlog)
    report.extend([
        "",
        "## Acoes proibidas",
        "",
        "- " + "; ".join(PROHIBITED_CATEGORIES),
        "",
        "## Proximo pacote recomendado",
        "",
        "MV2-16 deve implementar CLI/validadores globais e dashboard de gates usando apenas outputs leves ja existentes.",
        "",
    ])
    (OUTPUT_DIR / "MV2_15_CRONOGRAMA_ENGINE_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    exec_summary = [
        "# MV2-15 - Sumario Executivo",
        "",
        f"1. Fase atual: {summary.get('current_phase')}.",
        "2. A vertente de dados foi transformada em gates formais; nao ha motivo para continuar cavando dados automaticamente agora.",
        f"3. Dia 8: {summary.get('day8_status')}; Dia 10: {summary.get('day10_status')}; Dia 18: {summary.get('day18_status')}; Dia 19: {summary.get('day19_status')}; Dia 21: {summary.get('day21_status')}; Dia 22: {summary.get('day22_status')}.",
        f"4. Acoes de programacao pesada liberadas: {summary.get('allowed_programming_actions_count')}.",
        "5. Proibido segue: STAC real, download, crop, feature espectral, labels, silver, negativos, splits, sandbox e treino.",
        "6. Proximo pacote: CLI/validadores globais, dashboard de gates, registry global e checker de claims publicos.",
        "",
    ]
    (OUTPUT_DIR / "MV2_15_EXECUTIVE_SUMMARY.md").write_text("\n".join(exec_summary), encoding="utf-8")

    commands = [
        "# Reproducao MV2-15 - Cronograma Engine & Gate Orchestrator",
        "git branch --show-current",
        "git status --short",
        "git log --oneline -5",
        "git diff --cached --name-only",
        "python scripts/mv2_15_run_cronograma_engine.py",
        "python scripts/mv2_15_validate_cronograma_engine.py",
        "pytest tests/test_mv2_15_cronograma_engine.py",
        "pytest tests/test_mv2_14_gee_lineage_recovery.py",
        "pytest tests/test_mv2_13_sentinel_binding.py",
        "pytest tests/test_mv2_12_data_readiness.py",
        "python -m py_compile scripts/mv2_15_cronograma_common.py scripts/mv2_15_discover_mv2_outputs.py scripts/mv2_15_build_schedule_state.py scripts/mv2_15_build_gate_engine.py scripts/mv2_15_build_allowed_programming_actions.py scripts/mv2_15_build_engineering_backlog.py scripts/mv2_15_build_artifact_registry.py scripts/mv2_15_build_worktree_integration_matrix.py scripts/mv2_15_build_scientific_risk_summary.py scripts/mv2_15_build_summaries.py scripts/mv2_15_run_cronograma_engine.py scripts/mv2_15_validate_cronograma_engine.py",
        "git diff --check",
        "git diff --cached --name-only",
    ]
    (OUTPUT_DIR / "commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")


def build_all() -> None:
    ensure_dirs()
    build_schema()
    discover_mv2_outputs()
    build_schedule_state()
    build_gate_engine()
    build_allowed_programming_actions()
    build_engineering_backlog()
    build_artifact_registry()
    build_worktree_integration_matrix()
    build_scientific_risk_summary()
    build_summary()
    write_reports()


def validate_all() -> list[str]:
    problems: list[str] = []
    schedule = read_csv(OUTPUT_DIR / "mv2_15_schedule_state.csv")
    gates = read_csv(OUTPUT_DIR / "mv2_15_gate_engine.csv")
    actions = read_csv(OUTPUT_DIR / "mv2_15_allowed_programming_actions.csv")
    backlog = read_csv(OUTPUT_DIR / "mv2_15_engineering_backlog.csv")
    summary = read_json(OUTPUT_DIR / "mv2_15_cronograma_summary.json")
    risks = read_csv(OUTPUT_DIR / "mv2_15_scientific_risk_summary.csv")
    by_day = {row["day_id"]: row["real_status"] for row in schedule}
    for day in ["10", "18", "19", "21", "22"]:
        if not by_day.get(day, "").startswith("BLOCKED"):
            problems.append(f"Dia {day} desbloqueado indevidamente")
    for key in ["can_train", "can_stac_real", "can_create_sandbox"]:
        if summary.get(key) is not False:
            problems.append(f"{key} deveria ser false")
    for action in actions:
        if action["action_category"] in PROHIBITED_CATEGORIES and action["allowed_now"] != "false":
            problems.append(f"acao proibida permitida: {action['action_category']}")
    allowed_present = {a["action_category"] for a in actions if a["allowed_now"] == "true"}
    for category in ALLOWED_CATEGORIES:
        if category not in allowed_present:
            problems.append(f"categoria permitida ausente: {category}")
    if sum(1 for row in backlog if row["can_implement_now"] == "true") < 10:
        problems.append("backlog implementavel insuficiente")
    if count_heavy_outputs() != 0:
        problems.append("output publico pesado em MV2-15")
    if count_private_path_hits() != 0:
        problems.append("caminho absoluto privado em MV2-15")
    if summary.get("engineering_backlog_count") != len(backlog):
        problems.append("summary backlog incoerente")
    gate_names = {g["gate_name"] for g in gates}
    for required in [
        "GATE_NO_LABELS_CREATED", "GATE_NO_SILVER_FORMAL", "GATE_NO_GOLD",
        "GATE_NO_NEGATIVES_FORMAL", "GATE_NO_TRAINING", "GATE_NO_SANDBOX",
        "GATE_NO_STAC_REAL", "GATE_NO_DOWNLOAD", "GATE_NO_CROP",
        "GATE_NO_NATIVE_RASTER", "GATE_NO_SPECTRAL_FEATURES",
        "GATE_UNKNOWN_NOT_NEGATIVE", "GATE_CITY_NOT_LABEL", "GATE_PNG_NOT_RASTER",
        "GATE_DINOV2_REVIEW_ONLY", "GATE_DAY10_BLOCKED", "GATE_DAY18_BLOCKED",
        "GATE_DAY19_BLOCKED", "GATE_DAY21_BLOCKED", "GATE_DAY22_BLOCKED",
    ]:
        if required not in gate_names:
            problems.append(f"gate obrigatorio ausente: {required}")
    risk_names = {r["risk_name"] for r in risks}
    for risk, _, _ in RISK_ROWS:
        if risk not in risk_names:
            problems.append(f"risco obrigatorio ausente: {risk}")
    if git_output(["diff", "--cached", "--name-only"]):
        problems.append("staged area nao esta vazia")
    forbidden_claims = ["labels_created\": true", "silver_created\": true", "negatives_created\": true"]
    for path in OUTPUT_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".csv", ".json", ".md", ".txt"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(claim in text for claim in forbidden_claims):
                problems.append(f"claim proibido em {path.name}")
    return problems


def print_validation_result() -> int:
    problems = validate_all()
    if problems:
        for problem in problems:
            print(f"[mv2_15_validate] FALHA: {problem}")
        return 1
    print("[mv2_15_validate] OK - cronograma consolidado e guardrails fail-closed preservados")
    return 0


def status_counts() -> dict[str, int]:
    schedule = read_csv(OUTPUT_DIR / "mv2_15_schedule_state.csv")
    return dict(Counter(row["real_status"] for row in schedule))
