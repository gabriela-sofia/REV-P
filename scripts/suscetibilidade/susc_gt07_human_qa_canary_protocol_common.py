"""SUSC-GT07 - Protocolo de QA Humano para Canarios Observacionais.

Le os canarios selecionados pelo SUSC-GT06 e prepara, sem executar nada, o protocolo de
QA (Quality Assurance = controle de qualidade) humano: fila de revisao, formulario futuro,
checklist por canario, matriz de decisao, criterios de aceitacao/rejeicao e bloqueios. E um
pacote offline e review-only: NAO busca internet, NAO consulta STAC nem API, NAO roda SAR
nem GEE, NAO baixa Sentinel-1/2 nem raster, NAO abre imagem externa, NAO cria footprint,
NAO cria geometria real, NAO geocodifica, NAO executa QA real, NAO marca qa_status como
accepted nem rejected, NAO altera o score_v6, NAO cria score_v7, NAO treina modelo e NAO
altera as decisoes/datas/geometrias/selecao dos marcos anteriores. Nada e promovido a
ground truth nem a positive_strong.

Principio metodologico explicito:
  - QA humano e etapa OBRIGATORIA antes de qualquer promocao observacional forte;
  - QA humano NAO e treino;
  - QA humano NAO e ground truth automatico;
  - QA humano so avalia aderencia e qualidade da evidencia;
  - sem imagem/footprint executado, o QA real NAO pode ser concluido;
  - este marco cria a ESTRUTURA de revisao, nao a revisao final;
  - canarios continuam review-only;
  - footprint futuro pos-evento sera REFERENCIA de avaliacao, nunca feature pre-evento.

Ver [[susc_gt06_sar_canary_preparation]]. Identificadores tecnicos em ingles sao mantidos
por compatibilidade; o relatorio publico explica cada campo em portugues.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_io import (  # noqa: E402
    ensure_dir,
    rel,
    sha256_file,
    write_csv,
    write_json,
    write_markdown,
)
import susc_gt06_sar_canary_preparation_common as gt06  # noqa: E402

gt05 = gt06.gt05
gt02 = gt06.gt02

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS_DIR = ROOT / "schemas" / "suscetibilidade"

GT06_PACK = OUT / "susc_gt06_pacote_canarios_sar.csv"
GT06_CANARY = OUT / "susc_gt06_canarios_prioritarios.csv"
GT06_ELIG = OUT / "susc_gt06_elegibilidade_canarios_sar.csv"
GT06_REQ = OUT / "susc_gt06_requisitos_sar_futuro.csv"
GT06_PLAN = OUT / "susc_gt06_plano_execucao_sar_futura.csv"
GT06_EXCL = OUT / "susc_gt06_criterios_exclusao_sar.csv"
GT06_MANIFEST = OUT / "susc_gt06_manifesto.json"

REQUIRED_INPUTS = [GT06_PACK, GT06_CANARY, GT06_ELIG, GT06_REQ, GT06_PLAN, GT06_EXCL, GT06_MANIFEST]

REPORT = OUT / "SUSC_GT07_PROTOCOLO_QA_HUMANO_CANARIOS_OBSERVACIONAIS.md"
QUEUE_CSV = OUT / "susc_gt07_fila_qa_canarios.csv"
FORM_CSV = OUT / "susc_gt07_formulario_qa_canarios.csv"
CHECKLIST_CSV = OUT / "susc_gt07_checklist_qa_por_canario.csv"
MATRIX_CSV = OUT / "susc_gt07_matriz_decisao_qa.csv"
CRITERIA_CSV = OUT / "susc_gt07_criterios_aceitacao_rejeicao.csv"
BLOCKERS_CSV = OUT / "susc_gt07_bloqueios_qa.csv"
SUMMARY_CSV = OUT / "susc_gt07_resumo_qa_canarios.csv"
MANIFEST_JSON = OUT / "susc_gt07_manifesto.json"

SCHEMA_QUEUE = SCHEMAS_DIR / "susc_gt07_qa_queue.schema.json"
SCHEMA_FORM = SCHEMAS_DIR / "susc_gt07_qa_form.schema.json"
SCHEMA_MATRIX = SCHEMAS_DIR / "susc_gt07_qa_decision_matrix.schema.json"
SCHEMA_SUMMARY = SCHEMAS_DIR / "susc_gt07_qa_summary.schema.json"

PUBLIC_ARTIFACTS = [
    REPORT, QUEUE_CSV, FORM_CSV, CHECKLIST_CSV, MATRIX_CSV, CRITERIA_CSV,
    BLOCKERS_CSV, SUMMARY_CSV,
]
REQUIRED_OUTPUTS = PUBLIC_ARTIFACTS + [
    MANIFEST_JSON, SCHEMA_QUEUE, SCHEMA_FORM, SCHEMA_MATRIX, SCHEMA_SUMMARY,
]

MARCO = "SUSC-GT07"
TITULO_PUBLICO = "Protocolo de QA Humano para Canários Observacionais"
TITULO_H1_TOKEN = "QA Humano para Canários Observacionais"

PUBLIC_FORBIDDEN_RE = gt06.PUBLIC_FORBIDDEN_RE
OPERATIONAL_FORECAST_RE = gt06.OPERATIONAL_FORECAST_RE

GLOBAL_GUARDRAILS = {
    "eligible_for_training": "false",
    "eligible_for_ground_truth": "false",
    "trainable": "false",
    "score_v7_candidate": "false",
    "review_only": "true",
}

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
QA_PREP_STATUSES = [
    "pronto_para_qa_futuro",
    "aguardando_footprint_futuro",
    "aguardando_geometria_ou_aoi",
    "aguardando_separacao_fenomenologica",
    "bloqueado_para_qa",
    "fora_da_fila_qa",
]
QA_STATUS_FUTURE_ALLOWED = [
    "accepted", "rejected", "needs_more_source", "ambiguous",
    "phenomenon_mismatch", "temporal_mismatch", "spatial_mismatch",
    "no_data", "not_reviewed",
]
CURRENT_QA_ALLOWED = ["not_reviewed", "pending_review", "prepared_for_future_review"]
DECISION_OPTIONS = [
    "aceitar_como_referencia_observacional_forte_review_only",
    "manter_como_referencia_provisoria",
    "rejeitar_por_incompatibilidade_fenomenologica",
    "rejeitar_por_incompatibilidade_temporal",
    "rejeitar_por_incompatibilidade_espacial",
    "bloquear_por_no_data",
    "bloquear_por_falta_de_fonte",
    "bloquear_por_falta_de_footprint",
    "marcar_como_ambiguo",
    "manter_apenas_como_contexto_documental",
]
CHECKLIST_GROUPS = [
    "fenomeno", "temporalidade", "espacialidade", "fonte", "footprint_futuro",
    "exclusoes", "leakage", "incerteza", "decisao",
]

FLOOD_COMPATIBLE = gt06.FLOOD_COMPATIBLE
MIXED_PHENOMENA = gt06.MIXED_PHENOMENA
LANDSLIDE_PURE = gt06.LANDSLIDE_PURE
PETROPOLIS_BLOCKING = gt02.PETROPOLIS_BLOCKING_PHENOMENA
USABLE_WINDOW = gt06.USABLE_WINDOW

_clean = gt06._clean
_present = gt06._present

# ---------------------------------------------------------------------------
# Conteudo fixo: checklist, matriz de decisao, criterios
# ---------------------------------------------------------------------------
# (grupo, item_pt, required_for_acceptance, evidence_needed_pt, fail_closed_if_missing, blocks_acceptance)
CHECKLIST_ITEMS = [
    ("fenomeno", "Confirmar que o evento e alagamento/inundacao/enxurrada, nao deslizamento/risco/alerta", "true", "descricao do evento e fonte", "true", "true"),
    ("temporalidade", "Confirmar compatibilidade da data com a janela pre/pos-evento", "true", "data do evento e janela do GT04", "true", "true"),
    ("temporalidade", "Registrar se a data e explicita ou inferida", "false", "temporal_precision do GT04", "false", "false"),
    ("espacialidade", "Confirmar relacao espacial plausivel entre patch/AOI e a ocorrencia", "true", "patch_id/geometria do GT05", "true", "true"),
    ("espacialidade", "Avaliar se ponto/poligono/bbox/geometria parcial e suficiente", "true", "geometry_type do GT05", "true", "true"),
    ("fonte", "Classificar a fonte (oficial/institucional/tecnica/documental fraca)", "true", "source_authority", "true", "true"),
    ("footprint_futuro", "Avaliar plausibilidade do footprint SAR futuro (quando existir)", "true", "footprint SAR pos-evento futuro", "true", "true"),
    ("footprint_futuro", "Verificar que o footprint nao e confundido com agua permanente", "true", "mascara de agua permanente futura", "true", "true"),
    ("exclusoes", "Verificar agua permanente", "true", "mascara de agua permanente futura", "true", "true"),
    ("exclusoes", "Verificar sombra SAR / layover", "true", "checagem SAR futura", "true", "true"),
    ("exclusoes", "Verificar speckle", "false", "checagem SAR futura", "false", "false"),
    ("exclusoes", "Verificar vegetacao ambigua", "true", "checagem espectral futura", "true", "false"),
    ("exclusoes", "Verificar superficie urbana ambigua", "true", "checagem urbana futura", "true", "false"),
    ("exclusoes", "Verificar declividade/HAND incompativel", "true", "filtros HAND/slope futuros", "true", "true"),
    ("exclusoes", "Verificar ausencia de observacao", "true", "artefato visual futuro", "true", "true"),
    ("exclusoes", "Verificar fenomeno misto (inundacao vs deslizamento)", "true", "separacao fenomenologica", "true", "true"),
    ("leakage", "Confirmar que footprint pos-evento NAO e usado como feature pre-evento", "true", "auditoria de leakage temporal", "true", "true"),
    ("incerteza", "Revisar uncertainty_m e qualidade do vinculo", "true", "uncertainty_m e patch_link_quality", "true", "false"),
    ("decisao", "Registrar decisao review-only (aceitar/rejeitar/ambiguo) sem criar ground truth", "true", "formulario de QA", "true", "true"),
]

# (condition_group, condition_pt, decision_option, qa_status_future, allowed_use, forbidden_use, can_enable_evaluation)
DECISION_RULES = [
    ("aceitacao", "fenomeno, temporal, espacial, fonte e footprint compativeis e QA aceito",
     "aceitar_como_referencia_observacional_forte_review_only", "accepted",
     "avaliacao review-only da aderencia observacional",
     "treino; ground truth supervisionado; score_v7; alteracao do score_v6", "true"),
    ("provisorio", "evento documentado, mas footprint ainda ausente",
     "manter_como_referencia_provisoria", "not_reviewed",
     "calibracao review-only futura, sem avaliacao patch-level forte",
     "treino; ground truth; score_v7", "false"),
    ("rejeicao_fenomeno", "fenomeno incompativel (deslizamento, chuva generica, risco, alerta)",
     "rejeitar_por_incompatibilidade_fenomenologica", "phenomenon_mismatch",
     "descartar como referencia observacional", "treino; ground truth; score_v7", "false"),
    ("rejeicao_temporal", "data incompativel com a janela pre/pos-evento",
     "rejeitar_por_incompatibilidade_temporal", "temporal_mismatch",
     "descartar por incompatibilidade temporal", "treino; ground truth; score_v7", "false"),
    ("rejeicao_espacial", "patch/AOI sem relacao espacial plausivel com a ocorrencia",
     "rejeitar_por_incompatibilidade_espacial", "spatial_mismatch",
     "descartar por incompatibilidade espacial", "treino; ground truth; score_v7", "false"),
    ("bloqueio_no_data", "impossivel decidir por nuvem/sombra/ambiguidade/ausencia de imagem",
     "bloquear_por_no_data", "no_data", "aguardar dados", "treino; ground truth; score_v7", "false"),
    ("bloqueio_fonte", "fonte insuficiente para sustentar referencia",
     "bloquear_por_falta_de_fonte", "needs_more_source", "buscar fonte melhor", "treino; ground truth; score_v7", "false"),
    ("bloqueio_footprint", "footprint SAR/artefato visual ausente",
     "bloquear_por_falta_de_footprint", "not_reviewed", "aguardar footprint futuro", "treino; ground truth; score_v7", "false"),
    ("ambiguidade", "evidencia ambigua entre inundacao e outro fenomeno/artefato",
     "marcar_como_ambiguo", "ambiguous", "revisar novamente com mais evidencia", "treino; ground truth; score_v7", "false"),
    ("contexto", "util apenas como documentacao, sem potencial patch-level",
     "manter_apenas_como_contexto_documental", "not_reviewed", "manter como contexto documental", "treino; ground truth; score_v7", "false"),
]

# (criterion_type, criterion_pt, required_for, severity, accepted_value, rejected_value, ambiguous_value)
CRITERIA = [
    ("fenomeno", "O evento e alagamento/inundacao/enxurrada?", "aceitacao", "alta",
     "inundacao/alagamento confirmado", "deslizamento/risco/alerta", "fenomeno indefinido"),
    ("temporal", "A data e compativel com a janela pre/pos-evento?", "aceitacao", "alta",
     "data dentro da janela", "data fora da janela", "data imprecisa"),
    ("espacial", "O patch/AOI tem relacao espacial plausivel com a ocorrencia?", "aceitacao", "alta",
     "relacao espacial plausivel", "sem relacao espacial", "relacao espacial incerta"),
    ("fonte", "A fonte e oficial/institucional/tecnica?", "aceitacao", "media",
     "fonte oficial/institucional/tecnica", "fonte documental fraca", "fonte parcial"),
    ("footprint", "O footprint futuro e plausivel e nao confundido com agua permanente?", "aceitacao", "alta",
     "footprint plausivel e delimitado", "footprint implausivel/confundido", "footprint incerto"),
    ("exclusoes", "As exclusoes (agua permanente, sombra, vegetacao, urbano, HAND) foram checadas?", "aceitacao", "alta",
     "exclusoes verificadas e ok", "exclusao critica presente", "exclusao nao verificada"),
    ("leakage", "O footprint pos-evento NAO e usado como feature pre-evento?", "aceitacao", "alta",
     "sem leakage temporal", "leakage temporal detectado", "risco de leakage a revisar"),
    ("incerteza", "A incerteza espacial (uncertainty_m) foi revisada?", "aceitacao", "media",
     "incerteza aceitavel", "incerteza excessiva", "incerteza desconhecida"),
    ("decisao", "A decisao final e review-only, sem criar ground truth?", "aceitacao", "alta",
     "decisao review-only registrada", "tentativa de criar ground truth", "decisao pendente"),
]

# Campos boolean/check do formulario que devem ficar como placeholder neste marco.
FORM_CHECK_FIELDS = [
    "visual_artifact_available", "footprint_available", "official_geometry_available",
    "event_date_confirmed", "phenomenon_confirmed", "spatial_match_confirmed",
    "temporal_match_confirmed", "source_quality_confirmed", "permanent_water_checked",
    "sar_shadow_layover_checked", "urban_ambiguity_checked", "vegetation_ambiguity_checked",
    "hand_slope_consistency_checked", "leakage_check_passed", "uncertainty_m_reviewed",
]


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _run_git(args: list[str]) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return r.stdout.strip() if r.returncode == 0 else ""


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        return list(csv.DictReader(f))


def require_inputs() -> None:
    missing = [rel(p) for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("entradas GT06 ausentes: " + "; ".join(missing))


def _seq(x: str) -> str:
    import re
    m = re.search(r"(\d+)$", x)
    return m.group(1) if m else x


# ---------------------------------------------------------------------------
# Classificacao do estado de preparacao QA (pura)
# ---------------------------------------------------------------------------
def classify_qa_prep(fields: dict, selected: bool) -> str:
    if not selected:
        return "fora_da_fila_qa"
    pheno = _clean(fields.get("phenomenon_type")).lower()
    region = _clean(fields.get("region")).lower()
    city = _clean(fields.get("city")).lower()
    petro = "petropolis" in f"{region} {city}"
    petro_block = petro and (pheno in PETROPOLIS_BLOCKING or not _present(fields.get("phenomenon_type")))
    if petro_block or pheno in MIXED_PHENOMENA or pheno in LANDSLIDE_PURE:
        return "aguardando_separacao_fenomenologica"
    has_patch = _present(fields.get("patch_id"))
    has_geom_aoi = has_patch or _clean(fields.get("geometry_acquisition_status")).lower() in (
        "geometria_parcial_resolvivel", "geometria_forte_existente") or _present(fields.get("geometry_type"))
    if not has_geom_aoi:
        return "aguardando_geometria_ou_aoi"
    has_date = _present(fields.get("resolved_event_date"))
    has_window = _clean(fields.get("window_status")).lower() in USABLE_WINDOW
    if not has_date or not has_window:
        return "bloqueado_para_qa"
    pheno_ok = pheno in FLOOD_COMPATIBLE or _present(fields.get("phenomenon_type"))
    if has_geom_aoi and has_date and has_window and pheno_ok and has_patch:
        return "pronto_para_qa_futuro"
    return "aguardando_footprint_futuro"


def _footprint_available(fields: dict) -> bool:
    # Neste marco nenhum footprint foi executado; footprint so existiria se
    # geometry_acquisition_status fosse geometria_forte_existente (nao ocorre aqui).
    return _clean(fields.get("geometry_acquisition_status")).lower() == "geometria_forte_existente"


# ---------------------------------------------------------------------------
# Construcao das linhas
# ---------------------------------------------------------------------------
def build_rows() -> dict:
    pack = {p["sar_target_id"]: p for p in _read_csv(GT06_PACK)}
    canarios = _read_csv(GT06_CANARY)
    # mapear canary -> sar_target_id
    selected = [pack[c["sar_target_id"]] for c in canarios if c["sar_target_id"] in pack]
    selected.sort(key=lambda p: (p.get("canary_selection_rank", "99"), p["sar_target_id"]))

    queue, form, checklist, blockers = [], [], [], []
    b_idx = 1

    for p in selected:
        tid = p["target_id"]
        qid = f"QAQ_{_seq(p['sar_target_id'])}"
        prep = classify_qa_prep(p, selected=True)
        footprint_av = _footprint_available(p)
        visual_av = footprint_av  # sem execucao, nenhum artefato visual disponivel
        qa_now = footprint_av or visual_av  # sempre false neste marco
        req_artifacts = "footprint_sar_pos_evento;artefato_visual;mascara_agua_permanente"
        blocking = "aguardando footprint SAR / artefato visual do evento (execucao futura)" if not qa_now else ""

        queue.append({
            "qa_queue_id": qid,
            "canary_id": next((c["canary_id"] for c in canarios if c["sar_target_id"] == p["sar_target_id"]), ""),
            "sar_target_id": p["sar_target_id"], "target_id": tid,
            "event_id": p.get("event_id", "not_available"), "patch_id": p.get("patch_id", "not_available"),
            "city": p.get("city", "not_available"), "region": p.get("region", "not_available"),
            "phenomenon_type": p.get("phenomenon_type", "not_available"),
            "resolved_event_date": p.get("resolved_event_date", "not_available"),
            "temporal_precision": p.get("temporal_precision", ""),
            "pre_event_window_start": p.get("pre_event_window_start", ""),
            "pre_event_window_end": p.get("pre_event_window_end", ""),
            "post_event_window_start": p.get("post_event_window_start", ""),
            "post_event_window_end": p.get("post_event_window_end", ""),
            "source_authority": p.get("source_authority", "not_available"),
            "geometry_type": p.get("geometry_type", "not_available"),
            "geometry_acquisition_status": p.get("geometry_acquisition_status", ""),
            "sar_canary_eligibility": p.get("sar_canary_eligibility", ""),
            "sar_canary_priority_score": p.get("sar_canary_priority_score", ""),
            "qa_preparation_status": prep,
            "current_qa_status": "prepared_for_future_review",
            "future_qa_required": "true",
            "required_future_artifacts": req_artifacts,
            "qa_can_be_completed_now": _b(qa_now),
            "qa_blocking_reason": blocking,
            "remains_review_only": "true", "trainable": "false",
            "eligible_for_training": "false", "eligible_for_ground_truth": "false",
            "score_v7_candidate": "false",
            "rationale_pt": ("canario pronto para QA futuro: dados minimos presentes, mas QA real "
                             "so conclui com footprint/artefato visual; permanece review-only, sem treino"),
        })

        # formulario: placeholders (QA real nao executado)
        form.append({
            "qa_form_id": f"QAF_{_seq(p['sar_target_id'])}", "qa_queue_id": qid,
            "reviewer_id": "a_preencher", "review_date": "a_preencher",
            **{f: "nao_verificado" for f in FORM_CHECK_FIELDS},
            "reviewer_decision": "a_preencher",
            "reviewer_confidence": "a_preencher",
            "reviewer_notes": "a_preencher",
            "qa_status_after_review": "not_reviewed",
        })

        # checklist por canario
        for grp, item, req_acc, ev, fail_closed, blocks in CHECKLIST_ITEMS:
            checklist.append({
                "checklist_id": f"CHK_{len(checklist)+1:05d}", "qa_queue_id": qid,
                "checklist_group": grp, "checklist_item_pt": item,
                "required_for_acceptance": req_acc, "current_status": "pendente",
                "evidence_needed_pt": ev, "fail_closed_if_missing": fail_closed,
                "blocks_acceptance": blocks,
                "notes_pt": "aguardando QA humano futuro; sem execucao neste marco",
            })

        # bloqueios de QA
        blockers.append({
            "qa_blocker_id": f"QBL_{b_idx:05d}", "qa_queue_id": qid,
            "blocker_type": "footprint_ou_artefato_visual_ausente", "severity": "alta",
            "field": "footprint_available",
            "reason_pt": "QA real exige footprint SAR/artefato visual do evento, ainda nao executado",
            "prevents_qa_completion_now": "true", "prevents_future_acceptance": "false",
            "prevents_positive_strong": "true", "prevents_calibration": "true",
            "how_to_resolve_pt": "buscar cena Sentinel-1 e gerar footprint em marco futuro (sem executar agora)",
        })
        b_idx += 1

    # matriz de decisao (geral)
    matrix = []
    for i, (grp, cond, opt, qsf, allowed, forbidden, can_eval) in enumerate(DECISION_RULES, start=1):
        matrix.append({
            "decision_rule_id": f"DR_{i:03d}", "condition_group": grp, "condition_pt": cond,
            "decision_option": opt, "qa_status_future": qsf,
            "allowed_use_after_decision": allowed, "forbidden_use_after_decision": forbidden,
            "can_enable_evaluation_review_only": can_eval,
            "can_enable_training": "false", "can_enable_ground_truth": "false",
            "can_enable_score_v7": "false",
            "fail_closed_outcome_pt": "na duvida, bloquear/rejeitar; nunca promover a ground truth ou treino",
        })

    # criterios de aceitacao/rejeicao
    criteria = []
    for i, (ctype, cpt, reqf, sev, acc, rej, amb) in enumerate(CRITERIA, start=1):
        criteria.append({
            "criterion_id": f"CR_{i:03d}", "criterion_type": ctype, "criterion_pt": cpt,
            "required_for": reqf, "severity": sev,
            "accepted_value_pt": acc, "rejected_value_pt": rej, "ambiguous_value_pt": amb,
            "blocks_positive_strong_future": "true", "blocks_calibration_future": _b(ctype in ("fenomeno", "footprint", "leakage", "decisao")),
            "blocks_benchmark_future": "true",
        })

    return {
        "queue": queue, "form": form, "checklist": checklist, "matrix": matrix,
        "criteria": criteria, "blockers": blockers, "selected_count": len(selected),
    }


def _b(v) -> str:
    return "true" if v else "false"


# ---------------------------------------------------------------------------
# Resumo
# ---------------------------------------------------------------------------
def summary_rows(queue: list[dict]) -> list[dict]:
    order = {s: i for i, s in enumerate(QA_PREP_STATUSES)}
    agg = {}
    for q in queue:
        s = q["qa_preparation_status"]
        a = agg.setdefault(s, {"count": 0, "now": 0, "future": 0, "pending": 0})
        a["count"] += 1
        a["now"] += q["qa_can_be_completed_now"] == "true"
        a["future"] += q["future_qa_required"] == "true"
        a["pending"] += q["current_qa_status"] in ("pending_review", "prepared_for_future_review", "not_reviewed")
    rows = []
    for s in sorted(agg, key=lambda x: order.get(x, 99)):
        a = agg[s]
        rows.append({
            "qa_preparation_status": s, "count": str(a["count"]),
            "qa_can_be_completed_now_count": str(a["now"]),
            "future_qa_required_count": str(a["future"]),
            "accepted_now_count": "0", "rejected_now_count": "0",
            "pending_review_count": str(a["pending"]),
            "training_allowed_count": "0", "ground_truth_allowed_count": "0",
            "score_v7_candidate_count": "0",
        })
    return rows


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
_BOOL = {"enum": ["true", "false"]}
_FALSE = {"const": "false"}
_TRUE = {"const": "true"}


def schema_queue_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "susc_gt07_qa_queue.schema.json",
        "title": "SUSC-GT07 Fila de QA", "type": "object",
        "required": ["qa_queue_id", "sar_target_id", "target_id", "qa_preparation_status",
                     "current_qa_status", "qa_can_be_completed_now", "remains_review_only",
                     "trainable", "eligible_for_training", "eligible_for_ground_truth",
                     "score_v7_candidate", "rationale_pt"],
        "properties": {
            "qa_queue_id": {"type": "string", "pattern": r"^QAQ_\d+$"},
            "sar_target_id": {"type": "string", "pattern": r"^SAR_\d+$"},
            "target_id": {"type": "string", "pattern": r"^TGT_\d+$"},
            "qa_preparation_status": {"enum": QA_PREP_STATUSES},
            "current_qa_status": {"enum": CURRENT_QA_ALLOWED},
            "qa_can_be_completed_now": _BOOL,
            "remains_review_only": _TRUE, "trainable": _FALSE,
            "eligible_for_training": _FALSE, "eligible_for_ground_truth": _FALSE,
            "score_v7_candidate": _FALSE, "rationale_pt": {"type": "string", "minLength": 1},
        },
    }


def schema_form_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "susc_gt07_qa_form.schema.json",
        "title": "SUSC-GT07 Formulario de QA", "type": "object",
        "required": ["qa_form_id", "qa_queue_id", "qa_status_after_review"],
        "properties": {
            "qa_form_id": {"type": "string", "pattern": r"^QAF_\d+$"},
            "qa_queue_id": {"type": "string", "pattern": r"^QAQ_\d+$"},
            "qa_status_after_review": {"enum": QA_STATUS_FUTURE_ALLOWED},
            "reviewer_decision": {"type": "string"},
        },
    }


def schema_matrix_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "susc_gt07_qa_decision_matrix.schema.json",
        "title": "SUSC-GT07 Matriz de Decisao QA", "type": "object",
        "required": ["decision_rule_id", "decision_option", "qa_status_future",
                     "can_enable_training", "can_enable_ground_truth", "can_enable_score_v7"],
        "properties": {
            "decision_rule_id": {"type": "string", "pattern": r"^DR_\d+$"},
            "decision_option": {"enum": DECISION_OPTIONS},
            "qa_status_future": {"enum": QA_STATUS_FUTURE_ALLOWED},
            "can_enable_evaluation_review_only": _BOOL,
            "can_enable_training": _FALSE, "can_enable_ground_truth": _FALSE,
            "can_enable_score_v7": _FALSE,
        },
    }


def schema_summary_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "susc_gt07_qa_summary.schema.json",
        "title": "SUSC-GT07 Resumo de QA", "type": "object",
        "required": ["qa_preparation_status", "count", "accepted_now_count",
                     "rejected_now_count", "training_allowed_count",
                     "ground_truth_allowed_count", "score_v7_candidate_count"],
        "properties": {
            "qa_preparation_status": {"enum": QA_PREP_STATUSES},
            "count": {"type": "string", "pattern": r"^\d+$"},
            "qa_can_be_completed_now_count": {"type": "string", "pattern": r"^\d+$"},
            "future_qa_required_count": {"type": "string", "pattern": r"^\d+$"},
            "accepted_now_count": {"const": "0"},
            "rejected_now_count": {"const": "0"},
            "pending_review_count": {"type": "string", "pattern": r"^\d+$"},
            "training_allowed_count": {"const": "0"},
            "ground_truth_allowed_count": {"const": "0"},
            "score_v7_candidate_count": {"const": "0"},
        },
    }


# ---------------------------------------------------------------------------
# Manifesto / proximo marco
# ---------------------------------------------------------------------------
def _next_milestone(queue: list[dict], blockers: list[dict]) -> str:
    if not queue:
        return "GT08 - Preparacao de Pacote de Revisao Visual"
    footprint_blocked = sum(1 for b in blockers if b["blocker_type"] == "footprint_ou_artefato_visual_ausente")
    qa_now = sum(1 for q in queue if q["qa_can_be_completed_now"] == "true")
    if footprint_blocked >= len(queue) and qa_now == 0:
        return "GT08 - Busca Controlada de Cenas Sentinel-1 sem Download"
    if qa_now >= max(1, len(queue) // 2):
        return "GT08 - Execucao Manual de QA Externo"
    return "GT08 - Preparacao de Pacote de Revisao Visual"


def manifest_obj(bundle: dict) -> dict:
    queue = bundle["queue"]
    dist = {}
    for q in queue:
        dist[q["qa_preparation_status"]] = dist.get(q["qa_preparation_status"], 0) + 1
    artifacts = [
        {"path": rel(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in PUBLIC_ARTIFACTS + [SCHEMA_QUEUE, SCHEMA_FORM, SCHEMA_MATRIX, SCHEMA_SUMMARY]
        if p.exists()
    ]
    return {
        "marco": MARCO, "titulo_publico": TITULO_PUBLICO,
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "gt06_source_commit": _run_git(["log", "-1", "--format=%h", "--", rel(GT06_PACK)]) or "unknown",
        "canarios_na_fila": len(queue),
        "distribuicao_qa_prep": dist,
        "qa_executado": 0, "accepted_now": 0, "rejected_now": 0,
        "positive_strong_promovidos": 0, "footprint_criado": 0, "sar_executado": 0,
        "sentinel_baixado": 0, "internet_usada": 0,
        "global_guardrails": GLOBAL_GUARDRAILS,
        "training_true_count": sum(1 for q in queue if q["eligible_for_training"] == "true"),
        "ground_truth_true_count": sum(1 for q in queue if q["eligible_for_ground_truth"] == "true"),
        "score_v7_candidate_true_count": sum(1 for q in queue if q["score_v7_candidate"] == "true"),
        "current_qa_accepted_count": sum(1 for q in queue if q["current_qa_status"] == "accepted"),
        "current_qa_rejected_count": sum(1 for q in queue if q["current_qa_status"] == "rejected"),
        "qa_can_be_completed_now_count": sum(1 for q in queue if q["qa_can_be_completed_now"] == "true"),
        "score_v6_changed": gt02.gt01._score_v6_changed(),
        "score_v7_created": gt02.gt01._score_v7_exists(),
        "proximo_marco": _next_milestone(queue, bundle["blockers"]),
        "artifacts": artifacts,
    }


# ---------------------------------------------------------------------------
# Relatorio publico (portugues)
# ---------------------------------------------------------------------------
def report_text(manifest: dict, bundle: dict, summary: list[dict]) -> str:
    queue = bundle["queue"]
    if queue:
        linhas_fila = "\n".join(
            f"| {q['qa_queue_id']} | {q['event_id']} | {q['patch_id']} | {q['city']}/{q['region']} | "
            f"{q['resolved_event_date']} | {q['qa_preparation_status']} | {q['current_qa_status']} |"
            for q in queue
        )
        bloco_fila = f"""| qa_queue_id | event_id | patch_id | cidade/bairro | data | preparacao | status atual |
| --- | --- | --- | --- | --- | --- | --- |
{linhas_fila}"""
    else:
        bloco_fila = "Nenhum canario na fila de QA (selecao vazia no GT06)."
    linhas_sum = "\n".join(
        f"| {r['qa_preparation_status']} | {r['count']} | {r['qa_can_be_completed_now_count']} | "
        f"{r['pending_review_count']} | {r['accepted_now_count']} | {r['rejected_now_count']} |"
        for r in summary
    )
    n_petro = sum(1 for q in queue if q["region"] == "petropolis" or q["city"] == "Petropolis")

    return f"""# {MARCO} - {TITULO_PUBLICO}

## 1. Escopo do marco

Este marco prepara o **protocolo de QA humano** (QA = Quality Assurance, controle de
qualidade) para os canários selecionados no GT06, **sem executar o QA real** e sem marcar
nenhum canário como aceito/rejeitado. É um pacote **offline e review-only** (uso restrito a
revisão): não busca internet, não consulta STAC nem API, não roda SAR nem GEE, não baixa
Sentinel nem raster, não abre imagem externa, não cria footprint, não cria geometria real,
não executa QA de verdade, não altera o `score_v6`, não cria `score_v7`, não treina modelo
e não promove nada a ground truth nem a `positive_strong`.

## 2. Relação com GT01 a GT06

O GT01 definiu a política; o GT02 aplicou-a; o GT03 montou a fila; o GT04 resolveu datas; o
GT05 preparou geometria; o GT06 selecionou os canários. O GT07 monta a **estrutura de
revisão humana** desses canários.

## 3. Por que QA humano é necessário

QA humano é etapa **obrigatória** antes de qualquer promoção observacional forte: só o
revisor humano confirma compatibilidade fenomenológica, temporal e espacial, qualidade da
fonte e do footprint futuro, exclusões e ausência de leakage temporal. Sem QA, nenhuma
evidência vira referência forte.

## 4. Por que este marco não executa QA real

O QA real exige o **artefato visual/footprint** do evento, que só existirá após a execução
SAR futura. Sem imagem, o revisor não pode concluir a avaliação. Este marco cria a fila, o
formulário, o checklist, a matriz de decisão e os critérios — mas a revisão final fica para
um marco posterior.

## 5. Entradas usadas

Pacote e canários prioritários do GT06 (mais elegibilidade, requisitos e exclusões), lidos
dos outputs versionados sem regeração. Canários na fila: **{manifest['canarios_na_fila']}**.

## 6. Canários incluídos na fila

{bloco_fila}

## 7. Estados de preparação QA

`pronto_para_qa_futuro` (dados mínimos presentes, falta artefato visual/footprint),
`aguardando_footprint_futuro`, `aguardando_geometria_ou_aoi`,
`aguardando_separacao_fenomenologica`, `bloqueado_para_qa` e `fora_da_fila_qa`. Distribuição:

| qa_preparation_status | canários | QA possível agora | pendentes | aceitos agora | rejeitados agora |
| --- | --- | --- | --- | --- | --- |
{linhas_sum}

## 8. Formulário de QA futuro

`susc_gt07_formulario_qa_canarios.csv` traz uma linha por canário com todos os campos de
revisão (fenômeno, temporalidade, espacialidade, fonte, exclusões, leakage, incerteza,
decisão) **em branco/placeholder** (`a_preencher`/`nao_verificado`), porque o QA real não
foi executado. O `qa_status_after_review` inicial é `not_reviewed`.

## 9. Checklist por canário

`susc_gt07_checklist_qa_por_canario.csv` detalha os itens por grupo (`fenomeno`,
`temporalidade`, `espacialidade`, `fonte`, `footprint_futuro`, `exclusoes`, `leakage`,
`incerteza`, `decisao`), todos com `current_status=pendente`.

## 10. Critérios de aceitação, rejeição e ambiguidade

`susc_gt07_criterios_aceitacao_rejeicao.csv` define, por critério, o valor que leva a
aceitar, rejeitar ou marcar como ambíguo, sempre bloqueando `positive_strong` e benchmark
futuros até a revisão.

## 11. Matriz de decisão QA

`susc_gt07_matriz_decisao_qa.csv` mapeia condições para as opções de decisão futura
(`decision_option`). **Nenhuma** regra habilita treino, ground truth supervisionado ou
`score_v7` (`can_enable_training=false`, `can_enable_ground_truth=false`,
`can_enable_score_v7=false`). Apenas `aceitar_como_referencia_observacional_forte_review_only`
habilita avaliação review-only futura.

## 12. Bloqueios atuais

Em `susc_gt07_bloqueios_qa.csv`: todos os canários estão bloqueados para conclusão do QA por
**ausência de footprint SAR/artefato visual** (`prevents_qa_completion_now=true`), que
depende de execução futura.

## 13. Como o QA futuro poderá liberar avaliação review-only

Quando o revisor humano aceitar um canário
(`aceitar_como_referencia_observacional_forte_review_only`), ele liberará **apenas**
avaliação review-only da aderência observacional — nunca treino, ground truth ou `score_v7`.

## 14. Por que QA não libera treino nem ground truth supervisionado

Aceitar um canário no QA significa que a evidência é uma referência observacional de
qualidade suficiente para **revisão**, não um rótulo supervisionado. Todas as linhas mantêm
`eligible_for_training=false`, `eligible_for_ground_truth=false` e `score_v7_candidate=false`.

## 15. Tratamento de leakage temporal

O checklist e os critérios exigem confirmar que o **footprint pós-evento não é usado como
feature pré-evento** — o item de `leakage` bloqueia a aceitação se essa checagem falhar.

## 16. Petrópolis e eventos mistos

Canários de Petrópolis ou eventos mistos ({n_petro} na fila) entram como
`aguardando_separacao_fenomenologica` e não seguem para aceitação sem separar inundação de
deslizamento.

## 17. Confirmação explícita dos bloqueios

Este marco **não** usou internet, **não** consultou STAC/API, **não** executou SAR
(`sar_executado={manifest['sar_executado']}`), **não** rodou GEE, **não** baixou raster nem
Sentinel (`sentinel_baixado={manifest['sentinel_baixado']}`), **não** criou footprint
(`footprint_criado={manifest['footprint_criado']}`), **não** criou geometria real, **não**
executou QA real (`qa_executado={manifest['qa_executado']}`), **não** marcou nenhum canário
como aceito/rejeitado (`accepted_now={manifest['accepted_now']}`,
`rejected_now={manifest['rejected_now']}`), **não** treinou modelo, **não** produziu ground
truth, **não** criou `score_v7`, **não** alterou o `score_v6`
(`score_v6_changed={str(manifest['score_v6_changed']).lower()}`) e **não** promoveu nenhum
canário a `positive_strong`
(`positive_strong_promovidos={manifest['positive_strong_promovidos']}`). Contagens de
controle: `eligible_for_training=true` → {manifest['training_true_count']};
`eligible_for_ground_truth=true` → {manifest['ground_truth_true_count']};
`score_v7_candidate=true` → {manifest['score_v7_candidate_true_count']}.

O REV-P não prevê enchentes operacionalmente: produz análise estrutural review-only com
evidência observacional auditável.

## 18. Próximo passo recomendado

**{manifest['proximo_marco']}**. Como o QA está bloqueado pela ausência de footprint/artefato
visual, o próximo passo natural é a busca controlada de cenas Sentinel-1 (sem download) para
viabilizar o footprint técnico futuro que permitirá concluir o QA.
"""


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_all() -> dict:
    require_inputs()
    ensure_dir(OUT)
    ensure_dir(SCHEMAS_DIR)

    bundle = build_rows()
    summary = summary_rows(bundle["queue"])

    write_csv(QUEUE_CSV, bundle["queue"])
    write_csv(FORM_CSV, bundle["form"])
    write_csv(CHECKLIST_CSV, bundle["checklist"])
    write_csv(MATRIX_CSV, bundle["matrix"])
    write_csv(CRITERIA_CSV, bundle["criteria"])
    write_csv(BLOCKERS_CSV, bundle["blockers"])
    write_csv(SUMMARY_CSV, summary)

    write_json(SCHEMA_QUEUE, schema_queue_obj())
    write_json(SCHEMA_FORM, schema_form_obj())
    write_json(SCHEMA_MATRIX, schema_matrix_obj())
    write_json(SCHEMA_SUMMARY, schema_summary_obj())

    manifest = manifest_obj(bundle)
    write_markdown(REPORT, report_text(manifest, bundle, summary))
    manifest["artifacts"] = [
        {"path": rel(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in PUBLIC_ARTIFACTS + [SCHEMA_QUEUE, SCHEMA_FORM, SCHEMA_MATRIX, SCHEMA_SUMMARY]
        if p.exists()
    ]
    write_json(MANIFEST_JSON, manifest)
    return manifest


# ---------------------------------------------------------------------------
# Validacao
# ---------------------------------------------------------------------------
def _public_paths() -> list[Path]:
    return sorted((p for p in REQUIRED_OUTPUTS if p.suffix.lower() in {".csv", ".json", ".md"} and p.exists()),
                  key=lambda p: rel(p))


def public_text_violations_text(text: str) -> list[str]:
    return sorted({m.group(0) for m in PUBLIC_FORBIDDEN_RE.finditer(text)})


def operational_forecast_violations(text: str) -> list[str]:
    return sorted({m.group(0).strip() for m in OPERATIONAL_FORECAST_RE.finditer(text)})


def _report_h1(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def title_is_portuguese(h1: str) -> bool:
    return TITULO_H1_TOKEN in h1


def validate_outputs(manifest: dict) -> list[str]:
    errors = [f"missing:{rel(p)}" for p in REQUIRED_OUTPUTS if not p.exists()]
    if errors:
        return errors

    if manifest["score_v6_changed"]:
        errors.append("score_v6_changed_forbidden")
    if manifest["score_v7_created"]:
        errors.append("score_v7_created_forbidden")
    if manifest["positive_strong_promovidos"] != 0:
        errors.append("positive_strong_promovido_forbidden")
    if manifest["qa_executado"] != 0 or manifest["sar_executado"] != 0 or manifest["footprint_criado"] != 0:
        errors.append("qa_ou_sar_executado_forbidden")
    if manifest["current_qa_accepted_count"] != 0 or manifest["current_qa_rejected_count"] != 0:
        errors.append("qa_status_accepted_ou_rejected_forbidden")

    queue = _read_csv(QUEUE_CSV)
    form = {f["qa_queue_id"]: f for f in _read_csv(FORM_CSV)}
    matrix = _read_csv(MATRIX_CSV)
    summary = _read_csv(SUMMARY_CSV)
    checklist = _read_csv(CHECKLIST_CSV)

    if len(queue) > 5:
        errors.append(f"fila_maior_que_5:{len(queue)}")

    for q in queue:
        qid = q["qa_queue_id"]
        for fld in ("eligible_for_training", "eligible_for_ground_truth", "trainable", "score_v7_candidate"):
            if q.get(fld) != "false":
                errors.append(f"fila:{qid}:{fld}_nao_false")
        if q.get("remains_review_only") != "true":
            errors.append(f"fila:{qid}:review_only_nao_true")
        if q.get("current_qa_status") in ("accepted", "rejected"):
            errors.append(f"fila:{qid}:current_qa_status_proibido:{q.get('current_qa_status')}")
        if q.get("current_qa_status") not in CURRENT_QA_ALLOWED:
            errors.append(f"fila:{qid}:current_qa_status_fora_enum")
        if q.get("qa_preparation_status") not in QA_PREP_STATUSES:
            errors.append(f"fila:{qid}:prep_status_fora_enum")
        blob = q.get("rationale_pt", "").lower()
        if "ground truth" in blob and "nao" not in blob and "não" not in blob:
            errors.append(f"fila:{qid}:rationale_ground_truth")
        if "positive_strong" in blob:
            errors.append(f"fila:{qid}:rationale_positive_strong")
        # qa_can_be_completed_now=true exige footprint/visual disponivel no formulario
        if q.get("qa_can_be_completed_now") == "true":
            f = form.get(qid, {})
            if f.get("footprint_available") != "true" and f.get("visual_artifact_available") != "true":
                errors.append(f"fila:{qid}:qa_now_sem_footprint_visual")

    # matriz nunca habilita treino/ground truth/score_v7
    for r in matrix:
        for fld in ("can_enable_training", "can_enable_ground_truth", "can_enable_score_v7"):
            if r.get(fld) != "false":
                errors.append(f"matriz:{r.get('decision_rule_id')}:{fld}_nao_false")
        if r.get("decision_option") not in DECISION_OPTIONS:
            errors.append(f"matriz:{r.get('decision_rule_id')}:opcao_invalida")
        if r.get("qa_status_future") not in QA_STATUS_FUTURE_ALLOWED:
            errors.append(f"matriz:{r.get('decision_rule_id')}:status_future_invalido")

    # checklist deve conter todos os grupos obrigatorios e o item de leakage
    grupos = {c["checklist_group"] for c in checklist}
    for g in CHECKLIST_GROUPS:
        if g not in grupos:
            errors.append(f"checklist:grupo_ausente:{g}")
    if not any(c["checklist_group"] == "leakage" for c in checklist):
        errors.append("checklist:leakage_ausente")

    # formulario: placeholders, sem decisao real; qa_status_after_review not_reviewed
    for f in form.values():
        if f.get("qa_status_after_review") in ("accepted", "rejected"):
            errors.append(f"form:{f.get('qa_form_id')}:decisao_real_proibida")

    # resumo: accepted/rejected/training/gt/score_v7 sempre 0
    for r in summary:
        for fld in ("accepted_now_count", "rejected_now_count", "training_allowed_count",
                    "ground_truth_allowed_count", "score_v7_candidate_count"):
            if r.get(fld) != "0":
                errors.append(f"resumo:{r['qa_preparation_status']}:{fld}_nao_zero")

    # texto publico
    for path in _public_paths():
        txt = path.read_text(encoding="utf-8", errors="ignore")
        if operational_forecast_violations(txt):
            errors.append(f"{rel(path)}:claim_previsao_operacional")
        if public_text_violations_text(txt):
            errors.append(f"{rel(path)}:vocabulario_publico_proibido")
    report_txt = REPORT.read_text(encoding="utf-8", errors="ignore")
    if "review-only" not in report_txt.lower() and "review_only" not in report_txt.lower():
        errors.append("relatorio_omite_review_only")
    if "não executa o qa real" not in report_txt.lower() and "sem executar o qa real" not in report_txt.lower():
        errors.append("relatorio_nao_declara_qa_nao_executado")
    if not title_is_portuguese(_report_h1(report_txt)):
        errors.append("titulo_principal_nao_portugues")

    ignored = gt02.gt01.git_ignored(REQUIRED_OUTPUTS)
    for p in REQUIRED_OUTPUTS:
        if rel(p) in ignored:
            errors.append(f"artefato_ignorado_pelo_git:{rel(p)}")

    return errors


def validate() -> int:
    manifest = build_all()
    errors = validate_outputs(manifest)
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1
    print(
        "GT07 protocolo QA validado: "
        f"fila={manifest['canarios_na_fila']} distribuicao={manifest['distribuicao_qa_prep']} "
        f"accepted_now={manifest['accepted_now']} rejected_now={manifest['rejected_now']} "
        f"qa_executado={manifest['qa_executado']} proximo={manifest['proximo_marco']} "
        f"score_v6_changed={str(manifest['score_v6_changed']).lower()}"
    )
    return 0


def run_all() -> int:
    manifest = build_all()
    print(
        "GT07 protocolo QA gerado: "
        f"fila={manifest['canarios_na_fila']} distribuicao={manifest['distribuicao_qa_prep']} "
        f"proximo={manifest['proximo_marco']}"
    )
    return 0
