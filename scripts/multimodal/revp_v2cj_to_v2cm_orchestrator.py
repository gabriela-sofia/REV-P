"""REV-P v2cj-v2cm integrated orchestrator."""
from __future__ import annotations

import argparse
from pathlib import Path

import revp_v2cj_tp2_candidate_prioritization as v2cj
import revp_v2ck_digitization_package_builder as v2ck
import revp_v2cl_observed_geometry_validator as v2cl
import revp_v2cm_patch_event_replay_engine as v2cm
from revp_v2cj_to_v2cm_common import guardrail_rows, read_csv, write_csv, write_text


ROLLUP_FIELDS = ["stage", "command", "status", "output", "detail"]
GUARD_FIELDS = ["guardrail", "expected_value", "observed_value", "status", "detail"]


def run(repo_root: Path, force: bool = False) -> int:
    stages = [
        ("v2cj", "priorizacao", v2cj.run, "outputs_public/tables/revp_tp2_candidate_priority_v2cj.csv"),
        ("v2ck", "pacote_digitizacao", v2ck.run, "outputs_public/tables/revp_digitization_task_queue_v2ck.csv"),
        ("v2cl", "validacao_geometria", v2cl.run, "outputs_public/tables/revp_observed_geometry_validation_v2cl.csv"),
        ("v2cm", "replay_bloqueavel", v2cm.run, "outputs_public/tables/revp_patch_event_replay_v2cm.csv"),
    ]
    rows: list[dict[str, str]] = []
    exit_code = 0
    for stage, label, fn, output in stages:
        try:
            code = fn(repo_root, force)
            ok = code == 0
        except Exception as exc:
            ok = False
            code = 1
            detail = str(exc)
        else:
            detail = "executado"
        rows.append(
            {
                "stage": stage,
                "command": label,
                "status": "PASS" if ok else "FAIL",
                "output": output,
                "detail": detail,
            }
        )
        if code:
            exit_code = code
            break
    write_csv(repo_root / "outputs_public/logs_summary/revp_v2cj_to_v2cm_test_rollup.csv", rows, ROLLUP_FIELDS)
    guards = guardrail_rows([
        ("pipeline_integrado_executado", "PASS", "PASS" if exit_code == 0 else "FAIL", exit_code == 0, "v2cj-v2cm executado em ordem"),
    ])
    write_csv(repo_root / "outputs_public/logs_summary/revp_v2cj_to_v2cm_guardrail_rollup.csv", guards, GUARD_FIELDS)
    write_text(repo_root / "outputs_public/execution_reports/revp_v2cj_to_v2cm_integrated_report.md", integrated_report(repo_root, rows))
    write_text(repo_root / "outputs_public/execution_reports/revp_v2cj_to_v2cm_commit_checklist.md", checklist(rows, guards))
    return exit_code


def integrated_report(repo_root: Path, rows: list[dict[str, str]]) -> str:
    priority = read_csv(repo_root / "outputs_public/tables/revp_tp2_candidate_priority_v2cj.csv")
    queue = read_csv(repo_root / "outputs_public/tables/revp_digitization_task_queue_v2ck.csv")
    validation = read_csv(repo_root / "outputs_public/tables/revp_observed_geometry_validation_v2cl.csv")
    replay = read_csv(repo_root / "outputs_public/tables/revp_patch_event_replay_v2cm.csv")
    blocked_replay = sum(1 for row in replay if row.get("replay_status", "").startswith("REPLAY_BLOCKED"))
    lines = "\n".join(f"- `{row['stage']}`: {row['status']} ({row['detail']})" for row in rows)
    return f"""# REV-P v2cj-v2cm - relatorio integrado

Pacote integrado de priorizacao TP2, fila de digitalizacao, validacao geometrica
e replay bloqueavel. O pacote avanca infraestrutura de revisao, mas nao fecha
ground truth operacional, nao cria label e nao cria treino.

## Execucao

{lines}

## Contagens

- prioridades v2cj: {len(priority)}
- tarefas v2ck: {len(queue)}
- validacoes v2cl: {len(validation)}
- replays v2cm: {len(replay)}
- replays bloqueados: {blocked_replay}

## Estado metodologico

O pacote segue review-only. Replays bloqueados mantem area e razoes vazias. Claims
operacionais, deteccao, predicao, labels e negativos formais permanecem bloqueados.
"""


def checklist(rows: list[dict[str, str]], guards: list[dict[str, str]]) -> str:
    all_pass = all(row["status"] == "PASS" for row in rows) and all(row["status"] == "PASS" for row in guards)
    row_lines = "\n".join(f"- [{'x' if row['status'] == 'PASS' else ' '}] {row['stage']}: {row['detail']}" for row in rows)
    guard_lines = "\n".join(f"- [{'x' if row['status'] == 'PASS' else ' '}] {row['guardrail']}: {row['observed_value']}" for row in guards)
    return f"""# Checklist de commit v2cj-v2cm

## Etapas

{row_lines}

## Travas

{guard_lines}

Resultado geral: {'PASS' if all_pass else 'FAIL'}.

Mensagem sugerida:

```text
analysis: prepara priorizacao TP2 e replay bloqueavel sem ground truth operacional
```
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    return run(Path(args.repo_root), args.force)


if __name__ == "__main__":
    raise SystemExit(main())

