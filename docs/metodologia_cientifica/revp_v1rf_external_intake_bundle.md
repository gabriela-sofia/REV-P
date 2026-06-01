# v1rf — External Intake Bundle (P1)

## Objetivo

Consolidar v1ra-v1re num pacote auditavel: manifest, QC, resumo cientifico e tabela TCC, e resolver o estado do intake externo. Funciona mesmo sem intake preenchido (inputs vazios mantem header).

## Status final

final_status=EXTERNAL_INTAKE_TASK_BOARD_READY. QC checks: 15 (falharam: 0).

## Quality checks

task board exists, intake template exists, intake validation exists, event candidates review-only, event-patch links review-only, no operational label, no training target, no ground truth operational, no formal negative, no absence-as-negative, no DINO-as-proof, no absolute paths, no local_runs, blocked rows have blocked_reason.

## Declaracao obrigatoria

A camada v1ra-v1rf organiza a coleta e ingestao manual de evidencia externa, mas nao cria ground truth operacional. Documentos externos podem gerar candidatos review-only e vinculos evento-patch para revisao, sem produzir rotulos, targets supervisionados ou negativos formais por ausencia.
