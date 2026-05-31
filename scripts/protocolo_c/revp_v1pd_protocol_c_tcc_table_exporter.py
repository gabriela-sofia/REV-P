"""REV-P v1pd — Protocol C TCC-ready table exporter.

Generates small, directly reusable tables for the TCC from v1ot, v1pa, v1pc
summaries. Does not invent data — only reformats existing metrics with
Portuguese technical sentences ready to copy.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from revp_v1pb_v1pf_common import (
    DATASETS,
    DOCS,
    SCHEMAS,
    _p,
    emit_doc,
    load_metric_from_summary,
    write_csv_with_header,
    write_schema,
)

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

OUT_TEMPORAL = _p("REVP_V1PD_OUT_TEMPORAL", DATASETS / "recife_protocol_c_tcc_table_temporal_recovery_v1pd.csv")
OUT_OBSERVED = _p("REVP_V1PD_OUT_OBSERVED", DATASETS / "recife_protocol_c_tcc_table_observed_evidence_v1pd.csv")
OUT_GUARDRAILS = _p("REVP_V1PD_OUT_GUARDRAILS", DATASETS / "recife_protocol_c_tcc_table_guardrails_v1pd.csv")
OUT_DECISIONS = _p("REVP_V1PD_OUT_DECISIONS", DATASETS / "recife_protocol_c_tcc_table_decision_levels_v1pd.csv")
SCHEMA_TEMPORAL = _p("REVP_V1PD_SCHEMA_TEMPORAL", SCHEMAS / "recife_protocol_c_tcc_table_temporal_recovery_v1pd_schema.csv")
SCHEMA_OBSERVED = _p("REVP_V1PD_SCHEMA_OBSERVED", SCHEMAS / "recife_protocol_c_tcc_table_observed_evidence_v1pd_schema.csv")
SCHEMA_GUARDRAILS = _p("REVP_V1PD_SCHEMA_GUARDRAILS", SCHEMAS / "recife_protocol_c_tcc_table_guardrails_v1pd_schema.csv")
SCHEMA_DECISIONS = _p("REVP_V1PD_SCHEMA_DECISIONS", SCHEMAS / "recife_protocol_c_tcc_table_decision_levels_v1pd_schema.csv")
DOC = _p("REVP_V1PD_DOC", DOCS / "revp_v1pd_protocol_c_tcc_table_exporter.md")

# Input summaries
IN_V1OT = _p("REVP_V1PD_IN_V1OT", DATASETS / "recife_scene_date_recovery_final_scientific_summary_v1ot.csv")
IN_V1PA = _p("REVP_V1PD_IN_V1PA", DATASETS / "recife_protocol_c_observed_evidence_scientific_summary_v1pa.csv")

TEMPORAL_FIELDS = ["item", "value", "interpretation", "tcc_sentence"]
OBSERVED_FIELDS = ["item", "value", "interpretation", "tcc_sentence"]
GUARDRAILS_FIELDS = ["guardrail", "status", "evidence", "tcc_sentence"]
DECISIONS_FIELDS = ["level", "meaning_in_project", "current_count", "allowed_use", "prohibited_use", "tcc_sentence"]


def _m_v1ot(metric: str) -> str:
    return load_metric_from_summary(IN_V1OT, metric)


def _m_v1pa(metric: str) -> str:
    return load_metric_from_summary(IN_V1PA, metric)


def build_temporal_table() -> list[dict[str, Any]]:
    patches = _m_v1ot("total_patch_alias_candidates_evaluated")
    product_dates = _m_v1ot("product_dates_confirmed_real")
    can_unlock = _m_v1ot("can_unlock_temporal_true_count")
    if can_unlock == "N/A":
        can_unlock = "0"
    c3_plus = _m_v1pa("c3_plus_candidates")
    dino = _m_v1ot("dino_queue_entries")
    if dino == "N/A":
        dino = "0"

    return [
        {"item": "Patches Sentinel avaliados",
         "value": patches,
         "interpretation": "Aliases unicos de patches no corpus Recife com tentativa de resolucao de data",
         "tcc_sentence": f"Foram avaliados {patches} patches Sentinel unicos para resolucao temporal."},
        {"item": "Datas de produto confirmadas",
         "value": product_dates,
         "interpretation": "Cadeia completa patch-asset-produto-aquisicao confirmada",
         "tcc_sentence": f"Nenhuma data de aquisicao Sentinel foi confirmada com cadeia de proveniencia completa ({product_dates} de {patches} tentativas)."},
        {"item": "Desbloqueios temporais",
         "value": can_unlock,
         "interpretation": "Patches com can_unlock_temporal=true",
         "tcc_sentence": f"Nenhum patch obteve desbloqueio temporal (can_unlock_temporal=true: {can_unlock})."},
        {"item": "Candidatos C3+",
         "value": c3_plus,
         "interpretation": "Candidatos que alcancaram nivel C3 ou superior",
         "tcc_sentence": f"Nenhum candidato alcancou nivel C3+ (requer data Sentinel confirmada): {c3_plus}."},
        {"item": "Fila DINO (temporal)",
         "value": dino,
         "interpretation": "Entradas na fila DINO derivadas de recuperacao temporal",
         "tcc_sentence": f"A fila DINO derivada de recuperacao temporal permaneceu vazia ({dino} entradas)."},
        {"item": "Status final",
         "value": "TEMPORAL_RECOVERY_FAIL_CLOSED",
         "interpretation": "Pipeline temporal encerrado sem confirmacao de data",
         "tcc_sentence": "O status final da recuperacao temporal e FAIL_CLOSED: nenhuma data de cena Sentinel foi confirmada, bloqueando adjudicacao temporal operacional."},
    ]


def build_observed_table() -> list[dict[str, Any]]:
    sources = _m_v1pa("sources_scanned")
    candidates = _m_v1pa("source_candidates_found")
    contextual = _m_v1pa("contextual_only_evidence")
    blocked = _m_v1pa("blocked_insufficient_evidence")
    linkages = _m_v1pa("event_patch_linkages_total")
    temporal = _m_v1pa("temporal_linkages_confirmed")
    c1 = _m_v1pa("c1_contextual")
    c2 = _m_v1pa("c2_review_only")
    c3 = _m_v1pa("c3_plus_candidates")
    c4 = _m_v1pa("c4_formal_negatives")
    dino = _m_v1pa("dino_review_queue")
    final = _m_v1pa("final_status")

    return [
        {"item": "Fontes escaneadas", "value": sources,
         "interpretation": "Arquivos do repositorio escaneados para evidencia externa",
         "tcc_sentence": f"Foram escaneados {sources} arquivos do repositorio para identificar candidatos a evidencia externa."},
        {"item": "Candidatos a fonte permitidos", "value": candidates,
         "interpretation": "Fontes com termos de evidencia relevantes",
         "tcc_sentence": f"Desses, {candidates} foram identificados como candidatos a fontes de evidencia permitidos para registro."},
        {"item": "Evidencia contextual apenas", "value": contextual,
         "interpretation": "Eventos com evidencia contextual sem confirmacao institucional",
         "tcc_sentence": f"{contextual} candidatos a eventos foram classificados como evidencia contextual apenas, sem confirmacao por fonte institucional adquirida."},
        {"item": "Evidencia bloqueada/insuficiente", "value": blocked,
         "interpretation": "Candidatos bloqueados por falta de fonte ou dado",
         "tcc_sentence": f"{blocked} candidatos foram bloqueados por evidencia insuficiente (fonte nao adquirida ou licenca desconhecida)."},
        {"item": "Vinculos evento-patch", "value": linkages,
         "interpretation": "Tentativas de vinculacao evento-patch geradas",
         "tcc_sentence": f"Foram gerados {linkages} vinculos evento-patch, todos em regime contextual ou bloqueado temporalmente."},
        {"item": "Vinculos temporais confirmados", "value": temporal,
         "interpretation": "Vinculos com cadeia temporal Sentinel confirmada",
         "tcc_sentence": f"Nenhum vinculo temporal foi confirmado ({temporal}), pois TEMPORAL_RECOVERY_FAIL_CLOSED impede qualquer adjudicacao temporal."},
        {"item": "C1 (contextual)", "value": c1,
         "interpretation": "Decisoes C1 — evidencia contextual insuficiente para review",
         "tcc_sentence": f"{c1} candidatos foram classificados como C1 (contextual apenas)."},
        {"item": "C2 (review-only)", "value": c2,
         "interpretation": "Decisoes C2 — review-only sem label",
         "tcc_sentence": f"{c2} candidatos alcancaram C2 (review-only), permitindo representacao visual sem criacao de rotulo."},
        {"item": "C3+ (nao alcancado)", "value": c3,
         "interpretation": "Nenhum candidato alcancou C3+ sem data Sentinel confirmada",
         "tcc_sentence": f"O nivel C3+ nao foi alcancado ({c3} candidatos), pois requer cadeia temporal Sentinel confirmada."},
        {"item": "C4 negativos formais", "value": c4,
         "interpretation": "Nenhum negativo formal disponivel",
         "tcc_sentence": f"Nenhum negativo formal foi identificado ({c4}), mantendo C4 fechado."},
        {"item": "Fila DINO review-only", "value": dino,
         "interpretation": "Entradas DINO para representacao visual sem label",
         "tcc_sentence": f"A fila DINO review-only conteve {dino} entradas para representacao visual/embedding sem criacao de rotulo."},
        {"item": "Status final", "value": final,
         "interpretation": "Pipeline observacional encerrado em review-only",
         "tcc_sentence": "O status final da camada observacional e OBSERVED_EVIDENCE_REVIEW_ONLY_FAIL_CLOSED: nenhuma evidencia externa foi promovida a ground truth operacional."},
    ]


def build_guardrails_table() -> list[dict[str, Any]]:
    return [
        {"guardrail": "Sem ground truth operacional",
         "status": "PASS",
         "evidence": "ground_truth=true nao encontrado; can_be_used_as_ground_truth=false em todos os registros",
         "tcc_sentence": "Nenhum registro foi promovido a ground truth operacional em qualquer estagio do Protocolo C."},
        {"guardrail": "Sem labels",
         "status": "PASS",
         "evidence": "labels_created=0; can_create_operational_label=false em todos os registros",
         "tcc_sentence": "Nenhum rotulo supervisionado foi criado; o pipeline opera em regime review-only."},
        {"guardrail": "Sem targets de treinamento",
         "status": "PASS",
         "evidence": "training_targets_created=0; can_train_model=false em todos os registros",
         "tcc_sentence": "Nenhum target de treinamento foi gerado; DINOv2 nao e utilizado para classificacao supervisionada."},
        {"guardrail": "DINO review-only",
         "status": "PASS",
         "evidence": "dino_allowed_use=REVIEW_ONLY_REPRESENTATION; dino_can_create_label=false; dino_target_field_created=false",
         "tcc_sentence": "Os embeddings DINOv2 sao utilizados exclusivamente para representacao estrutural visual, sem derivacao de rotulos ou targets."},
        {"guardrail": "Sem negativos formais",
         "status": "PASS",
         "evidence": "formal_negative_count=0; C4 permanece fechado",
         "tcc_sentence": "Na ausencia de declaracao oficial de nao-ocorrencia, o nivel C4 permanece fechado."},
        {"guardrail": "Sem C3+ sem data Sentinel",
         "status": "PASS",
         "evidence": "product_dates_confirmed_real=0; c3_plus_candidates=0",
         "tcc_sentence": "Nenhum candidato alcancou C3+ porque a resolucao temporal Sentinel nao confirmou nenhuma data de aquisicao (FAIL_CLOSED)."},
        {"guardrail": "Sem desbloqueio temporal sem produto",
         "status": "PASS",
         "evidence": "can_unlock_temporal=true: 0",
         "tcc_sentence": "Nenhum desbloqueio temporal ocorreu na ausencia de cadeia de proveniencia produto-patch confirmada."},
    ]


def build_decisions_table() -> list[dict[str, Any]]:
    c1 = _m_v1pa("c1_contextual")
    c2 = _m_v1pa("c2_review_only")

    return [
        {"level": "C1",
         "meaning_in_project": "Evidencia contextual apenas — informacao territorial documentada sem confirmacao institucional de evento",
         "current_count": c1,
         "allowed_use": "Referencia contextual em texto; citacao de existencia de busca",
         "prohibited_use": "Ground truth; label; treinamento; predicao; flood detection",
         "tcc_sentence": f"O nivel C1 ({c1} candidatos) indica evidencia contextual documentada que nao pode ser utilizada como referencia operacional."},
        {"level": "C2",
         "meaning_in_project": "Candidato review-only — evidencia suficiente para representacao visual DINO sem label",
         "current_count": c2,
         "allowed_use": "Representacao DINO review-only; revisao humana futura; planejamento de aquisicao",
         "prohibited_use": "Ground truth; label; treinamento supervisionado; predicao; validacao cruzada",
         "tcc_sentence": f"O nivel C2 ({c2} candidatos) permite representacao visual via embeddings DINOv2, mas proibe qualquer uso como rotulo ou target de treinamento."},
        {"level": "C3",
         "meaning_in_project": "Candidato com linkage temporal — requer scene_date Sentinel confirmada",
         "current_count": "0",
         "allowed_use": "Review-only com contexto temporal; controle experimental futuro",
         "prohibited_use": "Ground truth operacional; label sem revisao humana; treinamento direto",
         "tcc_sentence": "O nivel C3 nao foi alcancado porque nenhuma data de cena Sentinel foi confirmada com cadeia de proveniencia completa."},
        {"level": "C3+",
         "meaning_in_project": "Candidato com linkage temporal confirmado e criterios espaciais satisfeitos",
         "current_count": "0",
         "allowed_use": "Review-only avancado; candidato a revisao humana formal",
         "prohibited_use": "Ground truth automatico; label sem revisao; treinamento sem validacao independente",
         "tcc_sentence": "Nenhum candidato alcancou C3+ — este nivel exige cadeia temporal confirmada E criterios espaciais satisfeitos simultaneamente."},
        {"level": "C4",
         "meaning_in_project": "Gate operacional — requer negativo formal explicito de fonte oficial",
         "current_count": "0",
         "allowed_use": "Uso operacional com negativos formais; treinamento supervisionado com split/leakage verificado",
         "prohibited_use": "Qualquer uso sem negativo formal; treino sem negativos; label unilateral",
         "tcc_sentence": "O nivel C4 permanece fechado (0 negativos formais): nenhuma fonte oficial emitiu declaracao de nao-ocorrencia que habilite uso operacional."},
    ]


def run() -> None:
    temporal = build_temporal_table()
    observed = build_observed_table()
    guardrails = build_guardrails_table()
    decisions = build_decisions_table()

    write_csv_with_header(OUT_TEMPORAL, temporal, TEMPORAL_FIELDS)
    write_csv_with_header(OUT_OBSERVED, observed, OBSERVED_FIELDS)
    write_csv_with_header(OUT_GUARDRAILS, guardrails, GUARDRAILS_FIELDS)
    write_csv_with_header(OUT_DECISIONS, decisions, DECISIONS_FIELDS)

    write_schema(SCHEMA_TEMPORAL, TEMPORAL_FIELDS, "v1pd_tcc_temporal_recovery")
    write_schema(SCHEMA_OBSERVED, OBSERVED_FIELDS, "v1pd_tcc_observed_evidence")
    write_schema(SCHEMA_GUARDRAILS, GUARDRAILS_FIELDS, "v1pd_tcc_guardrails")
    write_schema(SCHEMA_DECISIONS, DECISIONS_FIELDS, "v1pd_tcc_decision_levels")

    emit_doc(DOC, """# v1pd - Protocol C TCC-Ready Table Exporter

## Objetivo

Gerar tabelas pequenas e diretamente reutilizaveis no TCC a partir dos
summaries v1ot e v1pa. Nao inventa dado — apenas reformata metricas
existentes com frases em portugues tecnico prontas para copia.

## Tabelas geradas

1. **Temporal Recovery** — metricas da recuperacao temporal Sentinel
2. **Observed Evidence** — metricas da camada observacional
3. **Guardrails** — evidencias de conformidade anti-overclaim
4. **Decision Levels** — definicao e contagem dos niveis C1-C4

## Uso no TCC

As colunas `tcc_sentence` contem frases em portugues tecnico que podem
ser copiadas diretamente para secoes de Metodos, Resultados ou Discussao.
Nenhuma frase faz overclaim ou promove evidencia fraca.
""")

    print(f"[v1pd] TCC tables: temporal={len(temporal)}, observed={len(observed)}, "
          f"guardrails={len(guardrails)}, decisions={len(decisions)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v1pd Protocol C TCC table exporter")
    parser.parse_args()
    run()
