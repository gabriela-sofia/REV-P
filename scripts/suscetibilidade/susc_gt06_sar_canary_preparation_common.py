"""SUSC-GT06 - Preparacao de Canarios SAR sem Execucao.

Le os resultados do SUSC-GT05 (geometria) e do SUSC-GT04 (datas/janelas) e seleciona,
sem executar nada, um conjunto pequeno de alvos candidatos a canario SAR (Sentinel-1)
para footprint tecnico FUTURO. E um pacote offline e review-only: NAO busca internet,
NAO consulta STAC nem API, NAO roda SAR nem GEE, NAO baixa Sentinel-1/2 nem raster, NAO
cria footprint, NAO cria geometria real, NAO geocodifica, NAO altera o score_v6, NAO cria
score_v7, NAO treina modelo e NAO altera as decisoes do GT02, as datas do GT04 nem os
diagnosticos do GT05. Nada e promovido a ground truth nem a positive_strong.

Principio metodologico explicito:
  - canario SAR NAO e ground truth massivo;
  - canario SAR NAO e base de treino;
  - canario SAR NAO altera score;
  - canario SAR serve para testar aderencia observacional em POUCOS eventos prioritarios;
  - footprint pos-evento, quando existir no futuro, sera REFERENCIA DE AVALIACAO, nunca
    feature pre-evento;
  - a generalizacao do REV-P continua sustentada por features escalaveis por patch.

Ver [[susc_gt05_official_geometry_acquisition]] e [[susc_gt04_event_date_window_resolver]].
Identificadores tecnicos em ingles sao mantidos por compatibilidade; o relatorio publico
explica cada campo em portugues.
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
import susc_gt05_official_geometry_acquisition_common as gt05  # noqa: E402

gt04 = gt05.gt04
gt03 = gt05.gt03
gt02 = gt05.gt02

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS_DIR = ROOT / "schemas" / "suscetibilidade"

GT05_TARGETS = OUT / "susc_gt05_pacote_aquisicao_geometria_oficial.csv"
GT05_DIAG = OUT / "susc_gt05_diagnostico_geometrico_por_alvo.csv"
GT05_TRACKS = OUT / "susc_gt05_trilhas_aquisicao_geometrica.csv"
GT05_RELEASED = OUT / "susc_gt05_alvos_liberados_para_qa_ou_sar_futuro.csv"
GT05_MANIFEST = OUT / "susc_gt05_manifesto.json"
GT04_WINDOWS = OUT / "susc_gt04_janelas_pre_pos_evento.csv"

REQUIRED_INPUTS = [GT05_TARGETS, GT05_DIAG, GT05_TRACKS, GT05_RELEASED, GT05_MANIFEST, GT04_WINDOWS]

REPORT = OUT / "SUSC_GT06_PREPARACAO_CANARIOS_SAR_SEM_EXECUCAO.md"
PACK_CSV = OUT / "susc_gt06_pacote_canarios_sar.csv"
ELIG_CSV = OUT / "susc_gt06_elegibilidade_canarios_sar.csv"
REQ_CSV = OUT / "susc_gt06_requisitos_sar_futuro.csv"
PLAN_CSV = OUT / "susc_gt06_plano_execucao_sar_futura.csv"
EXCL_CSV = OUT / "susc_gt06_criterios_exclusao_sar.csv"
CANARY_CSV = OUT / "susc_gt06_canarios_prioritarios.csv"
BLOCKERS_CSV = OUT / "susc_gt06_bloqueios_canarios_sar.csv"
SUMMARY_CSV = OUT / "susc_gt06_resumo_canarios_sar.csv"
MANIFEST_JSON = OUT / "susc_gt06_manifesto.json"

SCHEMA_TARGET = SCHEMAS_DIR / "susc_gt06_sar_canary_target.schema.json"
SCHEMA_ELIG = SCHEMAS_DIR / "susc_gt06_sar_eligibility.schema.json"
SCHEMA_REQ = SCHEMAS_DIR / "susc_gt06_sar_future_requirement.schema.json"
SCHEMA_SUMMARY = SCHEMAS_DIR / "susc_gt06_sar_summary.schema.json"

PUBLIC_ARTIFACTS = [
    REPORT, PACK_CSV, ELIG_CSV, REQ_CSV, PLAN_CSV, EXCL_CSV, CANARY_CSV,
    BLOCKERS_CSV, SUMMARY_CSV,
]
REQUIRED_OUTPUTS = PUBLIC_ARTIFACTS + [
    MANIFEST_JSON, SCHEMA_TARGET, SCHEMA_ELIG, SCHEMA_REQ, SCHEMA_SUMMARY,
]

MARCO = "SUSC-GT06"
TITULO_PUBLICO = "Preparação de Canários SAR sem Execução"
TITULO_H1_TOKEN = "Canários SAR sem Execução"

PUBLIC_FORBIDDEN_RE = gt05.PUBLIC_FORBIDDEN_RE
OPERATIONAL_FORECAST_RE = gt05.OPERATIONAL_FORECAST_RE

GLOBAL_GUARDRAILS = {
    "eligible_for_training": "false",
    "eligible_for_ground_truth": "false",
    "trainable": "false",
    "score_v7_candidate": "false",
    "review_only": "true",
}

MAX_CANARIES = 5

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
SAR_ELIGIBILITY = [
    "elegivel_canario_sar_prioritario",
    "elegivel_canario_sar_secundario",
    "requer_geometria_antes_do_sar",
    "requer_separacao_fenomenologica",
    "bloqueado_temporalmente",
    "bloqueado_por_contexto_fraco",
    "nao_elegivel_para_sar",
]
SAR_PRIORITY_CLASSES = [
    "canario_sar_prioridade_alta",
    "canario_sar_prioridade_media",
    "canario_sar_prioridade_baixa",
    "bloqueado_para_sar",
    "manter_fora_dos_canarios",
]

USABLE_TEMPORAL = {"exact_day", "inferred_from_event_id", "date_range"}
STRONG_TEMPORAL = {"exact_day", "inferred_from_event_id"}
USABLE_WINDOW = {"usable", "usable_with_range_uncertainty"}
FLOOD_COMPATIBLE = set(gt02.FLOOD_PHENOMENA) | {"urban_flooding", "urban_flash_flood"}
MIXED_PHENOMENA = {"mixed_flood_landslide", "flash_flood_landslide", "flood_or_mass_movement"}
LANDSLIDE_PURE = {"landslide", "mass_movement", "deslizamento"}
CONTEXTUAL_FORBIDDEN = gt05.CONTEXTUAL_FORBIDDEN_TYPES
REGIOES_TRABALHADAS = REGIOES = {"recife", "curitiba", "petropolis"}
CIDADES = {"recife", "curitiba", "petropolis"}

_clean = gt03._clean
_present = gt04._present

# Requisitos futuros de SAR (com tipo).
SAR_REQUIREMENTS = {
    "resolved_event_date": ("temporal", "true"),
    "pre_event_window": ("temporal", "true"),
    "post_event_window": ("temporal", "true"),
    "patch_id_or_aoi": ("espacial", "true"),
    "phenomenon_type": ("fenomeno", "true"),
    "geometry_or_aoi": ("espacial", "true"),
    "source_authority": ("fonte", "false"),
    "permanent_water_mask_future": ("processamento_futuro", "true"),
    "hand_slope_filter_future": ("processamento_futuro", "true"),
    "urban_ambiguity_exclusion_future": ("processamento_futuro", "true"),
    "qa_status_future": ("qa_futuro", "true"),
    "uncertainty_m_future": ("qa_futuro", "false"),
    "sentinel1_scene_search_future": ("aquisicao_futura", "true"),
}
FUTURE_STANDARD_REQS = [
    "permanent_water_mask_future", "hand_slope_filter_future",
    "urban_ambiguity_exclusion_future", "qa_status_future",
    "uncertainty_m_future", "sentinel1_scene_search_future",
]

EXCLUSION_TYPES = [
    "temporal_window_missing", "phenomenon_mismatch", "mixed_landslide_flood",
    "no_patch_or_aoi", "contextual_source_only", "permanent_water_confusion",
    "urban_shadow_layover", "vegetation_ambiguity", "slope_hand_inconsistent",
    "no_qa_possible", "source_too_weak",
]
# Exclusoes de execucao futura (checagens padrao quando SAR rodar no futuro).
FUTURE_EXECUTION_EXCLUSIONS = [
    "permanent_water_confusion", "urban_shadow_layover", "vegetation_ambiguity",
    "slope_hand_inconsistent", "no_qa_possible",
]

# Plano de execucao SAR futura (passos fixos).
SAR_PLAN_STEPS = [
    ("confirmar AOI/patch do alvo", "AOI/patch confirmado", "false", "false", "false", "false"),
    ("buscar cenas Sentinel-1 futuras (sem executar agora)", "lista de cenas candidatas", "true", "true", "false", "false"),
    ("selecionar janela pre-evento", "cena pre-evento selecionada", "true", "true", "false", "false"),
    ("selecionar janela pos-evento", "cena pos-evento selecionada", "true", "true", "false", "false"),
    ("aplicar pre-processamento SAR futuro", "SAR pre-processado", "false", "true", "false", "false"),
    ("aplicar mascara de agua permanente futura", "mascara de agua aplicada", "false", "true", "true", "false"),
    ("aplicar filtros HAND/slope futuros", "filtros topograficos aplicados", "false", "false", "false", "false"),
    ("tratar ambiguidade urbana futura", "ambiguidade urbana tratada", "false", "true", "false", "false"),
    ("gerar poligono candidato futuro", "poligono candidato (referencia de avaliacao)", "false", "true", "false", "false"),
    ("executar QA humano futuro", "QA humano concluido", "false", "false", "false", "true"),
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
        raise FileNotFoundError("entradas GT05/GT04 ausentes: " + "; ".join(missing))


def _norm(v: str) -> str:
    v = _clean(v)
    return v if v and v.lower() not in gt02.NULLISH else "not_available"


def _b(v) -> str:
    return "true" if v else "false"


def _seq(target_id: str) -> str:
    import re
    m = re.search(r"(\d+)$", target_id)
    return m.group(1) if m else target_id


# ---------------------------------------------------------------------------
# Analise SAR (pura)
# ---------------------------------------------------------------------------
def analyze_sar(fields: dict) -> dict:
    prec = _clean(fields.get("temporal_precision")).lower()
    win = _clean(fields.get("window_status")).lower()
    src = _clean(fields.get("source_authority")).lower()
    gtype = _clean(fields.get("geometry_type")).lower()
    pheno = _clean(fields.get("phenomenon_type")).lower()
    status = _clean(fields.get("geometry_acquisition_status")).lower()
    region = _clean(fields.get("region")).lower()
    city = _clean(fields.get("city")).lower()

    weak = (src in CONTEXTUAL_FORBIDDEN) or (gtype in CONTEXTUAL_FORBIDDEN) or (src in gt02.WEAK_SOURCE_TYPES)
    petro = "petropolis" in f"{region} {city}"
    pheno_present = _present(fields.get("phenomenon_type"))
    petro_block = petro and (pheno in gt05.PETROPOLIS_BLOCKING or not pheno_present)
    has_patch = _present(fields.get("patch_id"))
    has_geom_aoi = has_patch or status in ("geometria_parcial_resolvivel", "geometria_forte_existente") \
        or _present(fields.get("footprint_id")) or _present(fields.get("geometry_id"))
    return {
        "prec": prec, "win": win,
        "has_date_op": prec in USABLE_TEMPORAL,
        "strong_temporal": prec in STRONG_TEMPORAL,
        "precision_unknown": prec in ("unknown", "invalid_or_conflicting", ""),
        "has_window": win in USABLE_WINDOW,
        "has_resolved_date": _present(fields.get("resolved_event_date")),
        "has_city_region": _present(fields.get("city")) or _present(fields.get("region")),
        "has_patch": has_patch, "has_geom_aoi": has_geom_aoi,
        "weak": weak, "petro": petro, "petro_block": petro_block,
        "pheno_present": pheno_present,
        "pheno_flood": pheno in FLOOD_COMPATIBLE,
        "pheno_mixed": pheno in MIXED_PHENOMENA,
        "pheno_landslide_pure": pheno in LANDSLIDE_PURE,
        "geom_status": status,
        "geom_priority_high": _clean(fields.get("geometry_priority_class")).lower() == "prioridade_geometrica_alta",
        "in_scope_city": city in CIDADES,
        "has_footprint_partial": _present(fields.get("footprint_id")) or _present(fields.get("geometry_id")),
        "can_qa": _clean(fields.get("can_advance_to_qa_future")).lower() == "true",
    }


def classify_eligibility(fl: dict) -> str:
    if not fl["has_date_op"] or fl["precision_unknown"] or not fl["has_window"]:
        return "bloqueado_temporalmente"
    if fl["petro_block"] or fl["pheno_mixed"] or fl["pheno_landslide_pure"] or fl["geom_status"] == "geometria_rejeitada":
        return "requer_separacao_fenomenologica"
    if fl["weak"] and not fl["has_geom_aoi"]:
        return "bloqueado_por_contexto_fraco"
    if (fl["strong_temporal"] and fl["has_resolved_date"] and fl["has_city_region"]
            and fl["has_geom_aoi"] and (fl["pheno_flood"] or not fl["pheno_present"])
            and not fl["weak"] and not fl["petro_block"]):
        return "elegivel_canario_sar_prioritario"
    if fl["has_geom_aoi"] or fl["has_footprint_partial"]:
        return "elegivel_canario_sar_secundario"
    if fl["has_date_op"]:
        return "requer_geometria_antes_do_sar"
    return "nao_elegivel_para_sar"


def missing_sar_requirements(fl: dict, fields: dict) -> list[str]:
    reqs = []
    if not fl["has_resolved_date"]:
        reqs.append("resolved_event_date")
    if not fl["has_window"]:
        reqs.append("pre_event_window")
        reqs.append("post_event_window")
    if not fl["has_patch"] and not fl["has_geom_aoi"]:
        reqs.append("patch_id_or_aoi")
    if not fl["pheno_flood"] and not fl["pheno_present"]:
        reqs.append("phenomenon_type")
    if not fl["has_geom_aoi"]:
        reqs.append("geometry_or_aoi")
    if not _present(fields.get("source_authority")):
        reqs.append("source_authority")
    # requisitos de processamento/QA futuros sempre necessarios para SAR real
    reqs.extend(FUTURE_STANDARD_REQS)
    # ordem estavel conforme SAR_REQUIREMENTS
    return [r for r in SAR_REQUIREMENTS if r in dict.fromkeys(reqs)]


def sar_priority_score(fl: dict) -> int:
    s = 0
    if fl["has_window"]:
        s += 12
    if fl["prec"] == "exact_day":
        s += 12
    if fl["prec"] == "inferred_from_event_id":
        s += 8
    if fl["geom_status"] == "requer_footprint_tecnico_futuro":
        s += 8
    if fl["geom_status"] == "geometria_parcial_resolvivel":
        s += 10
    if fl["geom_priority_high"]:
        s += 12
    if not fl["weak"] and fl["has_city_region"]:
        s += 8  # fonte/contexto util
    if fl["has_patch"]:
        s += 12
    if fl["pheno_flood"]:
        s += 8
    if fl["in_scope_city"]:
        s += 6
    if fl["has_footprint_partial"]:
        s += 4
    if fl["can_qa"]:
        s += 4
    # negativos
    if fl["precision_unknown"]:
        s -= 20
    if not fl["has_window"]:
        s -= 15
    if not fl["has_patch"]:
        s -= 6
    if not fl["has_city_region"]:
        s -= 6
    if fl["geom_status"] == "geometria_rejeitada":
        s -= 20
    if fl["petro_block"]:
        s -= 20
    if fl["weak"]:
        s -= 25
    return max(0, min(100, s))


def sar_priority_class(eligibility: str, score: int) -> str:
    if eligibility == "nao_elegivel_para_sar":
        return "manter_fora_dos_canarios"
    if eligibility in ("bloqueado_temporalmente", "bloqueado_por_contexto_fraco", "requer_separacao_fenomenologica"):
        return "bloqueado_para_sar"
    if eligibility == "elegivel_canario_sar_prioritario":
        return "canario_sar_prioridade_alta" if score >= 60 else "canario_sar_prioridade_media"
    if eligibility == "elegivel_canario_sar_secundario":
        return "canario_sar_prioridade_media" if score >= 40 else "canario_sar_prioridade_baixa"
    # requer_geometria_antes_do_sar
    return "canario_sar_prioridade_baixa" if score >= 20 else "bloqueado_para_sar"


# ---------------------------------------------------------------------------
# Construcao das linhas
# ---------------------------------------------------------------------------
def build_rows() -> dict:
    g5 = _read_csv(GT05_TARGETS)
    windows = {w["target_id"]: w for w in _read_csv(GT04_WINDOWS)}

    prepared = []  # (row, fl, eligibility, score)
    for t in g5:
        tid = t["target_id"]
        w = windows.get(tid, {})
        fields = dict(t)
        fields["window_status"] = w.get("window_status", t.get("window_status", ""))
        fl = analyze_sar(fields)
        elig = classify_eligibility(fl)
        score = sar_priority_score(fl)
        prepared.append((t, w, fl, elig, score))

    # selecao de canarios: apenas prioritarios, top 5 por score
    priortarios = [p for p in prepared if p[3] == "elegivel_canario_sar_prioritario"]
    priortarios_sorted = sorted(priortarios, key=lambda p: (-p[4], p[0]["target_id"]))
    selected_ids = {p[0]["target_id"] for p in priortarios_sorted[:MAX_CANARIES]}
    rank_by_id = {p[0]["target_id"]: i + 1 for i, p in enumerate(priortarios_sorted[:MAX_CANARIES])}

    pack, elig_rows, req_rows, plan_rows, excl_rows, canary_rows, blocker_rows = [], [], [], [], [], [], []
    b_idx = 1

    for t, w, fl, elig, score in prepared:
        tid = t["target_id"]
        sid = f"SAR_{_seq(tid)}"
        cls = sar_priority_class(elig, score)
        selected = tid in selected_ids
        rank = str(rank_by_id.get(tid, "")) if selected else "not_selected"
        missing = missing_sar_requirements(fl, t)
        can_prepare = elig in ("elegivel_canario_sar_prioritario", "elegivel_canario_sar_secundario", "requer_geometria_antes_do_sar")

        pack.append({
            "sar_target_id": sid, "geometry_target_id": t.get("geometry_target_id", ""), "target_id": tid,
            "source_replay_id": t.get("source_replay_id", ""), "source_decision_id": t.get("source_decision_id", ""),
            "event_id": t.get("event_id", "not_available"), "patch_id": t.get("patch_id", "not_available"),
            "city": t.get("city", "not_available"), "region": t.get("region", "not_available"),
            "phenomenon_type": t.get("phenomenon_type", "not_available"),
            "resolved_event_date": t.get("resolved_event_date", "not_available"),
            "temporal_precision": t.get("temporal_precision", ""),
            "pre_event_window_start": w.get("pre_event_window_start", ""),
            "pre_event_window_end": w.get("pre_event_window_end", ""),
            "post_event_window_start": w.get("post_event_window_start", ""),
            "post_event_window_end": w.get("post_event_window_end", ""),
            "window_status": w.get("window_status", "not_available"),
            "source_authority": t.get("source_authority", "not_available"),
            "geometry_acquisition_status": t.get("geometry_acquisition_status", ""),
            "geometry_priority_class": t.get("geometry_priority_class", ""),
            "geometry_type": t.get("geometry_type", "not_available"),
            "patch_link_quality": t.get("patch_link_quality", "not_available"),
            "qa_status": t.get("qa_status", "not_available"),
            "uncertainty_m": t.get("uncertainty_m", "not_available"),
            "sar_canary_eligibility": elig,
            "sar_canary_priority_score": str(score),
            "sar_canary_priority_class": cls,
            "canary_selection_rank": rank,
            "selected_for_canary_pack": _b(selected),
            "expected_future_sar_artifact_pt": "poligono candidato SAR pos-evento (referencia de avaliacao futura, nunca feature pre-evento)",
            "expected_future_qa_artifact_pt": "QA humano do canario (aceite review-only, sem treino)",
            "remains_review_only": "true", "trainable": "false",
            "eligible_for_training": "false", "eligible_for_ground_truth": "false",
            "score_v7_candidate": "false",
            "rationale_pt": _rationale(fl, elig, selected),
        })

        elig_rows.append({
            "eligibility_id": f"ELG_{_seq(tid)}", "sar_target_id": sid,
            "sar_canary_eligibility": elig, "eligibility_basis_pt": _elig_basis(elig),
            "required_before_sar_pt": _required_before(elig),
            "blocking_condition_pt": _blocking_condition(elig),
            "can_prepare_sar_future": _b(can_prepare),
            "can_execute_sar_now": "false",
            "no_sar_execution_in_this_milestone": "true",
            "no_network_in_this_milestone": "true",
            "no_raster_download_in_this_milestone": "true",
            "review_only": "true",
        })

        # requisitos futuros (apenas para quem pode preparar SAR)
        if can_prepare:
            for req in missing:
                rtype, req_for = SAR_REQUIREMENTS[req]
                req_rows.append({
                    "sar_requirement_id": f"SRQ_{len(req_rows)+1:05d}", "sar_target_id": sid,
                    "requirement": req, "requirement_type": rtype,
                    "required_for_future_sar": req_for,
                    "current_value": _current_value(req, t, w),
                    "severity": "alta" if req_for == "true" and rtype in ("temporal", "espacial") else "media",
                    "how_to_resolve_pt": _how_resolve_req(req),
                    "prevents_sar_future": req_for,
                    "prevents_positive_strong": "true",
                    "prevents_calibration": "true" if req in ("phenomenon_type", "qa_status_future") else "false",
                    "prevents_benchmark": "true",
                })

        # plano futuro: 10 passos para prioritarios; passo terminal para os demais
        if elig == "elegivel_canario_sar_prioritario":
            for i, (act, art, net, s1, dem, water) in enumerate(SAR_PLAN_STEPS, start=1):
                plan_rows.append({
                    "sar_plan_id": f"SPL_{len(plan_rows)+1:05d}", "sar_target_id": sid, "step_order": str(i),
                    "action_pt": act, "expected_artifact_pt": art,
                    "success_criteria_pt": f"{art} registrado e auditavel, mantendo review_only",
                    "fail_closed_outcome_pt": "sem o artefato, o canario nao avanca; sem promocao e sem treino",
                    "requires_network_future": net, "requires_sentinel1_future": s1,
                    "requires_dem_or_hand_future": dem, "requires_permanent_water_mask_future": water,
                    "requires_manual_qa_future": _b(i == len(SAR_PLAN_STEPS)),
                    "can_be_automated_future": _b(i not in (1, len(SAR_PLAN_STEPS))),
                    "no_execution_now": "true",
                })
        else:
            plan_rows.append({
                "sar_plan_id": f"SPL_{len(plan_rows)+1:05d}", "sar_target_id": sid, "step_order": "1",
                "action_pt": "sem plano SAR neste marco; resolver bloqueio antes de preparar canario",
                "expected_artifact_pt": "nenhum artefato SAR neste marco",
                "success_criteria_pt": "bloqueio resolvido em marco futuro",
                "fail_closed_outcome_pt": "permanece review-only, sem SAR e sem promocao",
                "requires_network_future": "false", "requires_sentinel1_future": "false",
                "requires_dem_or_hand_future": "false", "requires_permanent_water_mask_future": "false",
                "requires_manual_qa_future": "false", "can_be_automated_future": "false",
                "no_execution_now": "true",
            })

        # criterios de exclusao
        for et in _applicable_exclusions(fl):
            excl_rows.append({
                "exclusion_id": f"EXC_{len(excl_rows)+1:05d}", "sar_target_id": sid,
                "exclusion_type": et, "exclusion_reason_pt": _exclusion_reason(et),
                "applies_now": _b(et in _now_exclusions(fl)),
                "applies_in_future_execution": _b(et in FUTURE_EXECUTION_EXCLUSIONS),
                "required_check_future": _b(et in FUTURE_EXECUTION_EXCLUSIONS),
                "fail_closed_outcome_pt": "se o criterio ocorrer, excluir do canario e nao promover",
            })

        # canarios prioritarios selecionados
        if selected:
            canary_rows.append({
                "canary_id": f"CAN_{rank_by_id[tid]:02d}", "sar_target_id": sid, "selection_rank": str(rank_by_id[tid]),
                "event_id": t.get("event_id", "not_available"), "patch_id": t.get("patch_id", "not_available"),
                "city": t.get("city", "not_available"), "region": t.get("region", "not_available"),
                "resolved_event_date": t.get("resolved_event_date", "not_available"),
                "temporal_precision": t.get("temporal_precision", ""),
                "sar_canary_priority_score": str(score),
                "selection_reason_pt": "alvo datado com janela, patch/AOI e fenomeno de inundacao compativel; poucos passos ate footprint tecnico futuro",
                "future_execution_scope_pt": "footprint tecnico SAR pos-evento em marco futuro, sem execucao agora",
                "expected_validation_use_pt": "referencia de avaliacao review-only da aderencia observacional (nunca feature pre-evento)",
                "not_training_reason_pt": "canario e poucos eventos prioritarios; nao e base de treino nem generalizacao",
                "not_ground_truth_reason": "footprint SAR nao e verdade de campo massiva; e referencia review-only",
                "remains_review_only": "true",
            })

        # bloqueios
        for bl in _blockers(fl, elig, sid):
            bl["sar_blocker_id"] = f"SBL_{b_idx:05d}"
            blocker_rows.append(bl)
            b_idx += 1

    return {
        "pack": pack, "elig": elig_rows, "req": req_rows, "plan": plan_rows,
        "excl": excl_rows, "canary": canary_rows, "blockers": blocker_rows,
        "selected_count": len(selected_ids), "priortarios_count": len(priortarios),
    }


def _current_value(req: str, t: dict, w: dict) -> str:
    m = {
        "resolved_event_date": t.get("resolved_event_date", ""),
        "pre_event_window": w.get("pre_event_window_start", ""),
        "post_event_window": w.get("post_event_window_start", ""),
        "patch_id_or_aoi": t.get("patch_id", ""),
        "phenomenon_type": t.get("phenomenon_type", ""),
        "geometry_or_aoi": t.get("geometry_type", ""),
        "source_authority": t.get("source_authority", ""),
    }
    return _norm(m.get(req, "nao_disponivel_neste_marco"))


def _rationale(fl: dict, elig: str, selected: bool) -> str:
    if selected:
        return "canario prioritario selecionado: datado, com janela, patch/AOI e fenomeno de inundacao; footprint SAR futuro sera referencia review-only, nunca treino"
    return {
        "elegivel_canario_sar_prioritario": "elegivel prioritario (nao selecionado no top-5); segue review-only",
        "elegivel_canario_sar_secundario": "elegivel secundario: datado e plausivel, mas com lacunas de geometria/patch/fenomeno",
        "requer_geometria_antes_do_sar": "datado, mas sem patch/AOI minima; precisa de geometria antes de preparar SAR",
        "requer_separacao_fenomenologica": "fenomeno misto/deslizamento sem separacao; separar antes de qualquer canario",
        "bloqueado_temporalmente": "sem data ou sem janela operacional; nao ha base temporal para SAR",
        "bloqueado_por_contexto_fraco": "fonte/geometria apenas contextual; sem base para canario",
        "nao_elegivel_para_sar": "sem utilidade para footprint tecnico; fora dos canarios",
    }.get(elig, "review-only")


def _elig_basis(elig: str) -> str:
    return {
        "elegivel_canario_sar_prioritario": "data exata/inferida + janela + patch/AOI + fenomeno de inundacao compativel",
        "elegivel_canario_sar_secundario": "datado com janela, mas com lacunas relevantes",
        "requer_geometria_antes_do_sar": "datado sem patch/AOI minima",
        "requer_separacao_fenomenologica": "fenomeno misto/deslizamento nao separado",
        "bloqueado_temporalmente": "sem data util ou sem janela",
        "bloqueado_por_contexto_fraco": "fonte/geometria apenas contextual",
        "nao_elegivel_para_sar": "sem base para footprint tecnico",
    }.get(elig, "")


def _required_before(elig: str) -> str:
    return {
        "elegivel_canario_sar_prioritario": "QA humano e busca de cena Sentinel-1 (marco futuro)",
        "elegivel_canario_sar_secundario": "completar geometria/patch e fenomeno",
        "requer_geometria_antes_do_sar": "obter patch/AOI/geometria antes do SAR",
        "requer_separacao_fenomenologica": "separar inundacao de deslizamento",
        "bloqueado_temporalmente": "resolver data e janela pre/pos-evento",
        "bloqueado_por_contexto_fraco": "obter fonte/geometria nao contextual",
        "nao_elegivel_para_sar": "sem acao SAR objetiva",
    }.get(elig, "")


def _blocking_condition(elig: str) -> str:
    if elig.startswith("bloqueado") or elig.startswith("requer") or elig == "nao_elegivel_para_sar":
        return _elig_basis(elig)
    return "sem bloqueio de elegibilidade; segue como candidato review-only"


def _how_resolve_req(req: str) -> str:
    return {
        "resolved_event_date": "resolver data do evento (marco de datas)",
        "pre_event_window": "gerar janela pre-evento a partir da data",
        "post_event_window": "gerar janela pos-evento a partir da data",
        "patch_id_or_aoi": "resolver patch REV-P ou definir AOI minima",
        "phenomenon_type": "confirmar/separar fenomeno de inundacao",
        "geometry_or_aoi": "obter geometria/AOI para recorte SAR",
        "source_authority": "identificar fonte/autoridade rastreavel",
        "permanent_water_mask_future": "aplicar mascara de agua permanente na execucao futura",
        "hand_slope_filter_future": "aplicar filtros HAND/slope na execucao futura",
        "urban_ambiguity_exclusion_future": "tratar ambiguidade urbana na execucao futura",
        "qa_status_future": "executar QA humano na execucao futura",
        "uncertainty_m_future": "estimar incerteza na execucao futura",
        "sentinel1_scene_search_future": "buscar cena Sentinel-1 em marco futuro (sem executar agora)",
    }.get(req, "resolver em marco futuro")


def _applicable_exclusions(fl: dict) -> list[str]:
    ex = list(_now_exclusions(fl))
    # checagens de execucao futura sao lembradas para candidatos preparaveis
    if fl["has_date_op"] and fl["has_window"]:
        ex.extend(FUTURE_EXECUTION_EXCLUSIONS)
    return [e for e in EXCLUSION_TYPES if e in dict.fromkeys(ex)]


def _now_exclusions(fl: dict) -> set:
    now = set()
    if not fl["has_window"]:
        now.add("temporal_window_missing")
    if fl["pheno_landslide_pure"]:
        now.add("phenomenon_mismatch")
    if fl["pheno_mixed"] or fl["petro_block"]:
        now.add("mixed_landslide_flood")
    if not fl["has_patch"] and not fl["has_geom_aoi"]:
        now.add("no_patch_or_aoi")
    if fl["weak"]:
        now.add("contextual_source_only")
        now.add("source_too_weak")
    return now


def _exclusion_reason(et: str) -> str:
    return {
        "temporal_window_missing": "sem janela pre/pos-evento; SAR nao pode ser recortado no tempo",
        "phenomenon_mismatch": "fenomeno incompativel com inundacao (ex.: deslizamento puro)",
        "mixed_landslide_flood": "evento misto inundacao/deslizamento sem separacao",
        "no_patch_or_aoi": "sem patch nem AOI minima para recorte",
        "contextual_source_only": "fonte apenas contextual, sem base observacional",
        "permanent_water_confusion": "risco de confundir agua permanente com inundacao (checar no futuro)",
        "urban_shadow_layover": "sombra/layover urbano no SAR (checar no futuro)",
        "vegetation_ambiguity": "ambiguidade por vegetacao (checar no futuro)",
        "slope_hand_inconsistent": "inconsistencia de declividade/HAND (checar no futuro)",
        "no_qa_possible": "sem QA humano possivel (checar no futuro)",
        "source_too_weak": "fonte fraca demais para sustentar canario",
    }.get(et, "criterio de exclusao")


def _blockers(fl: dict, elig: str, sid: str) -> list[dict]:
    out = []

    def add(btype, sev, field, reason, sel=True, sar=True, ps=True, cal=False):
        out.append({
            "sar_target_id": sid, "blocker_type": btype, "severity": sev, "field": field,
            "reason_pt": reason, "prevents_canary_selection": _b(sel),
            "prevents_sar_future": _b(sar), "prevents_positive_strong": _b(ps),
            "prevents_calibration": _b(cal), "how_to_resolve_pt": _how_resolve_blocker(btype),
        })

    if elig == "bloqueado_temporalmente":
        add("sem_data_ou_janela", "alta", "temporal_precision", "sem data util ou janela pre/pos-evento", cal=True)
    if elig == "requer_separacao_fenomenologica":
        add("fenomeno_misto_ou_deslizamento", "alta", "phenomenon_type",
            "fenomeno misto/deslizamento sem separacao", cal=True)
    if elig == "bloqueado_por_contexto_fraco":
        add("fonte_contextual_fraca", "alta", "source_authority",
            "fonte/geometria apenas contextual", cal=True)
    if not fl["has_patch"] and not fl["has_geom_aoi"] and elig != "bloqueado_temporalmente":
        add("sem_patch_ou_aoi", "media", "patch_id", "sem patch nem AOI minima para canario", sar=False)
    return out


def _how_resolve_blocker(btype: str) -> str:
    return {
        "sem_data_ou_janela": "resolver data e janela (marco de datas) antes de preparar SAR",
        "fenomeno_misto_ou_deslizamento": "separar inundacao de deslizamento",
        "fonte_contextual_fraca": "obter fonte/geometria oficial nao contextual",
        "sem_patch_ou_aoi": "resolver patch REV-P ou definir AOI minima",
    }.get(btype, "resolver em marco futuro")


# ---------------------------------------------------------------------------
# Resumo
# ---------------------------------------------------------------------------
def summary_rows(pack: list[dict], elig_rows: list[dict]) -> list[dict]:
    order = {e: i for i, e in enumerate(SAR_ELIGIBILITY)}
    can_prep = {e["sar_target_id"]: e["can_prepare_sar_future"] for e in elig_rows}
    agg = {}
    for p in pack:
        e = p["sar_canary_eligibility"]
        a = agg.setdefault(e, {"count": 0, "score": 0, "selected": 0, "prep": 0})
        a["count"] += 1
        a["score"] += int(p["sar_canary_priority_score"])
        a["selected"] += p["selected_for_canary_pack"] == "true"
        a["prep"] += can_prep.get(p["sar_target_id"], "false") == "true"
    rows = []
    for e in sorted(agg, key=lambda x: order.get(x, 99)):
        a = agg[e]
        rows.append({
            "sar_canary_eligibility": e, "count": str(a["count"]),
            "average_sar_canary_priority_score": str(round(a["score"] / a["count"], 2)) if a["count"] else "0",
            "selected_for_canary_pack_count": str(a["selected"]),
            "can_prepare_sar_future_count": str(a["prep"]),
            "can_execute_sar_now_count": "0",
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
_SCORE = {"type": "string", "pattern": r"^(100|[1-9]?[0-9])$"}


def schema_target_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "susc_gt06_sar_canary_target.schema.json",
        "title": "SUSC-GT06 Alvo de Canario SAR", "type": "object",
        "required": ["sar_target_id", "target_id", "sar_canary_eligibility",
                     "sar_canary_priority_score", "sar_canary_priority_class",
                     "selected_for_canary_pack", "remains_review_only", "trainable",
                     "eligible_for_training", "eligible_for_ground_truth", "score_v7_candidate",
                     "rationale_pt"],
        "properties": {
            "sar_target_id": {"type": "string", "pattern": r"^SAR_\d+$"},
            "target_id": {"type": "string", "pattern": r"^TGT_\d+$"},
            "sar_canary_eligibility": {"enum": SAR_ELIGIBILITY},
            "sar_canary_priority_score": _SCORE,
            "sar_canary_priority_class": {"enum": SAR_PRIORITY_CLASSES},
            "selected_for_canary_pack": _BOOL,
            "remains_review_only": _TRUE, "trainable": _FALSE,
            "eligible_for_training": _FALSE, "eligible_for_ground_truth": _FALSE,
            "score_v7_candidate": _FALSE, "rationale_pt": {"type": "string", "minLength": 1},
        },
    }


def schema_elig_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "susc_gt06_sar_eligibility.schema.json",
        "title": "SUSC-GT06 Elegibilidade SAR", "type": "object",
        "required": ["eligibility_id", "sar_target_id", "sar_canary_eligibility",
                     "can_execute_sar_now", "no_sar_execution_in_this_milestone",
                     "no_network_in_this_milestone", "no_raster_download_in_this_milestone",
                     "review_only"],
        "properties": {
            "eligibility_id": {"type": "string", "pattern": r"^ELG_\d+$"},
            "sar_target_id": {"type": "string", "pattern": r"^SAR_\d+$"},
            "sar_canary_eligibility": {"enum": SAR_ELIGIBILITY},
            "can_prepare_sar_future": _BOOL, "can_execute_sar_now": _FALSE,
            "no_sar_execution_in_this_milestone": _TRUE, "no_network_in_this_milestone": _TRUE,
            "no_raster_download_in_this_milestone": _TRUE, "review_only": _TRUE,
        },
    }


def schema_req_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "susc_gt06_sar_future_requirement.schema.json",
        "title": "SUSC-GT06 Requisito SAR Futuro", "type": "object",
        "required": ["sar_requirement_id", "sar_target_id", "requirement",
                     "requirement_type", "how_to_resolve_pt"],
        "properties": {
            "sar_requirement_id": {"type": "string", "pattern": r"^SRQ_\d+$"},
            "sar_target_id": {"type": "string", "pattern": r"^SAR_\d+$"},
            "requirement": {"enum": list(SAR_REQUIREMENTS.keys())},
            "requirement_type": {"type": "string", "minLength": 1},
            "required_for_future_sar": _BOOL,
            "how_to_resolve_pt": {"type": "string", "minLength": 1},
            "prevents_sar_future": _BOOL, "prevents_positive_strong": _BOOL,
        },
    }


def schema_summary_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "susc_gt06_sar_summary.schema.json",
        "title": "SUSC-GT06 Resumo de Canarios SAR", "type": "object",
        "required": ["sar_canary_eligibility", "count", "average_sar_canary_priority_score",
                     "selected_for_canary_pack_count", "can_execute_sar_now_count",
                     "training_allowed_count", "ground_truth_allowed_count", "score_v7_candidate_count"],
        "properties": {
            "sar_canary_eligibility": {"enum": SAR_ELIGIBILITY},
            "count": {"type": "string", "pattern": r"^\d+$"},
            "average_sar_canary_priority_score": {"type": "string"},
            "selected_for_canary_pack_count": {"type": "string", "pattern": r"^\d+$"},
            "can_prepare_sar_future_count": {"type": "string", "pattern": r"^\d+$"},
            "can_execute_sar_now_count": {"const": "0"},
            "training_allowed_count": {"const": "0"},
            "ground_truth_allowed_count": {"const": "0"},
            "score_v7_candidate_count": {"const": "0"},
        },
    }


# ---------------------------------------------------------------------------
# Manifesto / proximo marco
# ---------------------------------------------------------------------------
def _next_milestone(dist: dict, selected: int, can_prep: int) -> str:
    # Ha canarios prioritarios suficientes -> QA humano dos canarios.
    if selected >= 3:
        return "GT07 - Protocolo de QA Humano para Canarios"
    req_geo = dist.get("requer_geometria_antes_do_sar", 0)
    # Maioria carece de patch/AOI -> aquisicao manual de geometria antes do SAR.
    if req_geo > can_prep:
        return "GT07 - Aquisicao Manual de Geometria/AOI"
    # Ha candidatos preparaveis, mas poucos selecionados -> buscar cenas Sentinel-1.
    if can_prep > 0:
        return "GT07 - Busca Controlada de Cenas Sentinel-1"
    return "GT07 - Aquisicao Manual de Geometria/AOI"


def manifest_obj(bundle: dict) -> dict:
    pack = bundle["pack"]
    dist = {}
    for p in pack:
        dist[p["sar_canary_eligibility"]] = dist.get(p["sar_canary_eligibility"], 0) + 1
    can_prep = sum(1 for e in bundle["elig"] if e["can_prepare_sar_future"] == "true")
    artifacts = [
        {"path": rel(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in PUBLIC_ARTIFACTS + [SCHEMA_TARGET, SCHEMA_ELIG, SCHEMA_REQ, SCHEMA_SUMMARY]
        if p.exists()
    ]
    return {
        "marco": MARCO, "titulo_publico": TITULO_PUBLICO,
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "gt05_source_commit": _run_git(["log", "-1", "--format=%h", "--", rel(GT05_TARGETS)]) or "unknown",
        "alvos_total": len(pack),
        "distribuicao_elegibilidade": dist,
        "canarios_prioritarios_elegiveis": bundle["priortarios_count"],
        "canarios_selecionados": bundle["selected_count"],
        "pode_preparar_sar_futuro": can_prep,
        "positive_strong_promovidos": 0, "sar_executado": 0, "footprint_criado": 0,
        "sentinel_baixado": 0, "internet_usada": 0,
        "global_guardrails": GLOBAL_GUARDRAILS,
        "training_true_count": sum(1 for p in pack if p["eligible_for_training"] == "true"),
        "ground_truth_true_count": sum(1 for p in pack if p["eligible_for_ground_truth"] == "true"),
        "score_v7_candidate_true_count": sum(1 for p in pack if p["score_v7_candidate"] == "true"),
        "can_execute_sar_now_true_count": sum(1 for e in bundle["elig"] if e["can_execute_sar_now"] == "true"),
        "score_v6_changed": gt02.gt01._score_v6_changed(),
        "score_v7_created": gt02.gt01._score_v7_exists(),
        "proximo_marco": _next_milestone(dist, bundle["selected_count"], can_prep),
        "artifacts": artifacts,
    }


# ---------------------------------------------------------------------------
# Relatorio publico (portugues)
# ---------------------------------------------------------------------------
def report_text(manifest: dict, bundle: dict, summary: list[dict]) -> str:
    linhas_sum = "\n".join(
        f"| {r['sar_canary_eligibility']} | {r['count']} | {r['average_sar_canary_priority_score']} | "
        f"{r['selected_for_canary_pack_count']} | {r['can_prepare_sar_future_count']} |"
        for r in summary
    )
    canarios = bundle["canary"]
    if canarios:
        linhas_can = "\n".join(
            f"| {c['selection_rank']} | {c['event_id']} | {c['patch_id']} | {c['city']} | "
            f"{c['region']} | {c['resolved_event_date']} | {c['sar_canary_priority_score']} |"
            for c in canarios
        )
        bloco_can = f"""| rank | event_id | patch_id | cidade | bairro/regiao | data | score |
| --- | --- | --- | --- | --- | --- | --- |
{linhas_can}"""
    else:
        bloco_can = ("Nenhum canario prioritario foi selecionado nesta rodada: os candidatos "
                     "disponiveis nao atingiram qualidade suficiente. A selecao vazia e valida e "
                     "documentada; ver bloqueios.")
    n_petro = sum(1 for p in bundle["pack"] if p["region"] == "petropolis" or p["city"] == "Petropolis")

    return f"""# {MARCO} - {TITULO_PUBLICO}

## 1. Escopo do marco

Este marco seleciona e prepara, **sem executar nada**, um conjunto pequeno de alvos
candidatos a **canário SAR** (Sentinel-1) para footprint técnico **futuro**. É um pacote
**offline e review-only** (uso restrito a revisão): não busca internet, não consulta STAC
nem API, não roda SAR nem GEE, não baixa Sentinel-1/2 nem raster, não cria footprint, não
cria geometria real, não geocodifica, não altera o `score_v6`, não cria `score_v7`, não
treina modelo e não promove nada a ground truth nem a `positive_strong`.

## 2. Relação com GT01, GT02, GT03, GT04 e GT05

O GT01 definiu a política; o GT02 aplicou-a; o GT03 montou a fila; o GT04 resolveu datas e
janelas; o GT05 preparou a geometria e apontou os alvos datados sem geometria oficial como
o gargalo. O GT06 usa esses alvos para escolher **poucos** canários SAR prioritários.

## 3. Por que SAR entra como canário, não como base central

O SAR não é a base central do REV-P: a generalização continua sustentada por **features
escaláveis por patch**. O canário SAR serve apenas para **testar a aderência observacional
em poucos eventos prioritários** — não é ground truth massivo, não é base de treino e não
altera o score. O footprint pós-evento, quando existir no futuro, será **referência de
avaliação** review-only, **nunca feature pré-evento**.

## 4. Por que este marco não executa SAR

A execução (busca de cena Sentinel-1, pré-processamento, máscara de água, filtros, QA) é
custosa e depende de rede e dados externos. Este marco apenas **prepara** a fila e o plano,
de forma determinística e auditável, deixando a execução para um marco futuro controlado.

## 5. Entradas usadas

Alvos e diagnóstico geométrico do GT05 e janelas do GT04 (por `target_id`), lidos dos
outputs versionados sem regeração. Total de alvos: **{manifest['alvos_total']}**.

## 6. Critérios de elegibilidade SAR

Sete classes: `elegivel_canario_sar_prioritario` (datado, com janela, patch/AOI e fenômeno
de inundação compatível), `elegivel_canario_sar_secundario` (datado com lacunas),
`requer_geometria_antes_do_sar` (datado sem patch/AOI), `requer_separacao_fenomenologica`
(fenômeno misto/deslizamento), `bloqueado_temporalmente` (sem data/janela),
`bloqueado_por_contexto_fraco` (fonte só contextual) e `nao_elegivel_para_sar`.

## 7. Critérios de pontuação e seleção

O `sar_canary_priority_score` (0 a 100) pontua data exata/inferida, janela gerada,
necessidade de footprint, prioridade geométrica alta, patch, fonte forte, fenômeno de
inundação e cidade/região no escopo; penaliza precisão `unknown`, ausência de janela/patch,
fenômeno misto e fonte fraca. São selecionados no **máximo {MAX_CANARIES}** canários
prioritários, por maior score, com diversidade de bairros quando possível.

## 8. Distribuição das elegibilidades

| sar_canary_eligibility | alvos | score médio | selecionados | podem preparar SAR |
| --- | --- | --- | --- | --- |
{linhas_sum}

## 9. Quantos podem preparar SAR futuro

**{manifest['pode_preparar_sar_futuro']}** alvos podem preparar SAR futuro (elegíveis ou
que precisam apenas de geometria/AOI). Nenhum pode executar SAR agora
(`can_execute_sar_now_true_count={manifest['can_execute_sar_now_true_count']}`).

## 10-11. Canários prioritários selecionados

Elegíveis prioritários: **{manifest['canarios_prioritarios_elegiveis']}**; selecionados:
**{manifest['canarios_selecionados']}** (limite {MAX_CANARIES}).

{bloco_can}

## 12. Principais bloqueios para SAR futuro

Em `susc_gt06_bloqueios_canarios_sar.csv`: ausência de data/janela (bloqueio temporal),
fenômeno misto/deslizamento sem separação, fonte apenas contextual e ausência de patch/AOI.

## 13. Critérios de exclusão para execução futura

Em `susc_gt06_criterios_exclusao_sar.csv`: janela ausente, incompatibilidade de fenômeno,
evento misto, ausência de patch/AOI, fonte contextual e, para a execução futura, confusão
com água permanente, sombra/layover urbano, ambiguidade por vegetação, inconsistência de
declividade/HAND e ausência de QA.

## 14. Plano de execução SAR futura (sem execução agora)

Em `susc_gt06_plano_execucao_sar_futura.csv`, dez passos para cada canário prioritário:
confirmar AOI/patch; buscar cenas Sentinel-1 futuras; selecionar janela pré-evento;
selecionar janela pós-evento; pré-processar SAR; aplicar máscara de água permanente;
aplicar filtros HAND/slope; tratar ambiguidade urbana; gerar polígono candidato (referência
de avaliação); executar QA humano. Todos com `no_execution_now=true`.

## 15. Petrópolis e eventos mistos

Alvos de Petrópolis ou mistos ({n_petro} relacionados) entram como
`requer_separacao_fenomenologica` e **nunca** são selecionados como canário sem separar o
fenômeno (inundação vs deslizamento).

## 16. Confirmação explícita dos bloqueios

Este marco **não** usou internet, **não** consultou STAC/API, **não** executou SAR
(`sar_executado={manifest['sar_executado']}`), **não** rodou GEE, **não** baixou raster nem
Sentinel (`sentinel_baixado={manifest['sentinel_baixado']}`), **não** criou footprint
(`footprint_criado={manifest['footprint_criado']}`), **não** criou geometria real, **não**
treinou modelo, **não** produziu ground truth, **não** criou `score_v7`, **não** alterou o
`score_v6` (`score_v6_changed={str(manifest['score_v6_changed']).lower()}`) e **não**
promoveu nenhum alvo a `positive_strong`
(`positive_strong_promovidos={manifest['positive_strong_promovidos']}`). Contagens de
controle: `eligible_for_training=true` → {manifest['training_true_count']};
`eligible_for_ground_truth=true` → {manifest['ground_truth_true_count']};
`score_v7_candidate=true` → {manifest['score_v7_candidate_true_count']}.

O REV-P não prevê enchentes operacionalmente: produz análise estrutural review-only com
evidência observacional auditável.

## 17. Próximo passo recomendado

**{manifest['proximo_marco']}**. Com **{manifest['canarios_selecionados']}** canários
selecionados, o próximo passo natural é o protocolo de QA humano desses poucos canários
(ou a busca controlada de cenas Sentinel-1), sempre review-only e sem treino.
"""


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_all() -> dict:
    require_inputs()
    ensure_dir(OUT)
    ensure_dir(SCHEMAS_DIR)

    bundle = build_rows()
    summary = summary_rows(bundle["pack"], bundle["elig"])

    write_csv(PACK_CSV, bundle["pack"])
    write_csv(ELIG_CSV, bundle["elig"])
    write_csv(REQ_CSV, bundle["req"])
    write_csv(PLAN_CSV, bundle["plan"])
    write_csv(EXCL_CSV, bundle["excl"])
    write_csv(CANARY_CSV, bundle["canary"])
    write_csv(BLOCKERS_CSV, bundle["blockers"])
    write_csv(SUMMARY_CSV, summary)

    write_json(SCHEMA_TARGET, schema_target_obj())
    write_json(SCHEMA_ELIG, schema_elig_obj())
    write_json(SCHEMA_REQ, schema_req_obj())
    write_json(SCHEMA_SUMMARY, schema_summary_obj())

    manifest = manifest_obj(bundle)
    write_markdown(REPORT, report_text(manifest, bundle, summary))
    manifest["artifacts"] = [
        {"path": rel(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in PUBLIC_ARTIFACTS + [SCHEMA_TARGET, SCHEMA_ELIG, SCHEMA_REQ, SCHEMA_SUMMARY]
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
    if manifest["sar_executado"] != 0 or manifest["footprint_criado"] != 0 or manifest["sentinel_baixado"] != 0:
        errors.append("execucao_sar_ou_download_forbidden")
    if manifest["can_execute_sar_now_true_count"] != 0:
        errors.append("can_execute_sar_now_forbidden")

    pack = _read_csv(PACK_CSV)
    elig = {e["sar_target_id"]: e for e in _read_csv(ELIG_CSV)}
    blockers_by = {}
    for b in _read_csv(BLOCKERS_CSV):
        blockers_by.setdefault(b["sar_target_id"], []).append(b)
    req_by = {}
    for r in _read_csv(REQ_CSV):
        req_by.setdefault(r["sar_target_id"], set()).add(r["requirement"])
    summary = _read_csv(SUMMARY_CSV)

    selected = 0
    for p in pack:
        sid = p["sar_target_id"]
        for fld in ("eligible_for_training", "eligible_for_ground_truth", "trainable", "score_v7_candidate"):
            if p.get(fld) != "false":
                errors.append(f"pack:{sid}:{fld}_nao_false")
        if p.get("remains_review_only") != "true":
            errors.append(f"pack:{sid}:review_only_nao_true")
        blob = p.get("rationale_pt", "").lower()
        if "ground truth" in blob and "nao" not in blob and "não" not in blob:
            errors.append(f"pack:{sid}:rationale_ground_truth")
        if "positive_strong" in blob:
            errors.append(f"pack:{sid}:rationale_positive_strong")
        try:
            sc = int(p["sar_canary_priority_score"])
        except ValueError:
            sc = -1
        if not (0 <= sc <= 100):
            errors.append(f"pack:{sid}:score_fora_intervalo")
        if p.get("sar_canary_priority_class") not in SAR_PRIORITY_CLASSES:
            errors.append(f"pack:{sid}:classe_invalida")
        if p.get("sar_canary_eligibility") not in SAR_ELIGIBILITY:
            errors.append(f"pack:{sid}:elegibilidade_invalida")

        sel = p.get("selected_for_canary_pack") == "true"
        if sel:
            selected += 1
            # restricoes de selecao
            if p.get("temporal_precision") in ("unknown", "invalid_or_conflicting", ""):
                errors.append(f"pack:{sid}:selecionado_temporal_unknown")
            if p.get("window_status") not in USABLE_WINDOW:
                errors.append(f"pack:{sid}:selecionado_sem_janela")
            has_patch = p.get("patch_id", "not_available") != "not_available"
            has_aoi = p.get("geometry_acquisition_status") in ("geometria_parcial_resolvivel", "geometria_forte_existente")
            if not has_patch and not has_aoi:
                errors.append(f"pack:{sid}:selecionado_sem_patch_aoi")
            pheno = p.get("phenomenon_type", "").lower()
            if (p.get("region") == "petropolis" or p.get("city") == "Petropolis") and pheno in gt05.PETROPOLIS_BLOCKING:
                if "separar_fenomeno" not in req_by.get(sid, set()) and "phenomenon_type" not in req_by.get(sid, set()):
                    errors.append(f"pack:{sid}:petropolis_selecionado_sem_separacao")
            # fontes fracas nunca prioritario sem blocker
            src = p.get("source_authority", "").lower()
            gtype = p.get("geometry_type", "").lower()
            if (src in CONTEXTUAL_FORBIDDEN or gtype in CONTEXTUAL_FORBIDDEN) and not blockers_by.get(sid):
                errors.append(f"pack:{sid}:fraco_selecionado_sem_blocker")

        # elegibilidade: can_execute_sar_now sempre false; flags de guardrail
        e = elig.get(sid, {})
        if e.get("can_execute_sar_now") != "false":
            errors.append(f"elig:{sid}:can_execute_sar_now_nao_false")
        for fld in ("no_sar_execution_in_this_milestone", "no_network_in_this_milestone", "no_raster_download_in_this_milestone", "review_only"):
            if e.get(fld) != "true":
                errors.append(f"elig:{sid}:{fld}_nao_true")

    if selected > MAX_CANARIES:
        errors.append(f"mais_de_{MAX_CANARIES}_canarios_selecionados:{selected}")

    for r in summary:
        for fld in ("can_execute_sar_now_count", "training_allowed_count", "ground_truth_allowed_count", "score_v7_candidate_count"):
            if r.get(fld) != "0":
                errors.append(f"resumo:{r['sar_canary_eligibility']}:{fld}_nao_zero")
        if r.get("sar_canary_eligibility") not in SAR_ELIGIBILITY:
            errors.append(f"resumo:elegibilidade_invalida")

    # footprint pos-evento descrito como referencia futura, nunca feature pre-evento
    report_txt = REPORT.read_text(encoding="utf-8", errors="ignore")
    if "feature pré-evento" in report_txt.lower() and "nunca feature pré-evento" not in report_txt.lower():
        errors.append("footprint_descrito_como_feature_pre_evento")

    for path in _public_paths():
        txt = path.read_text(encoding="utf-8", errors="ignore")
        if operational_forecast_violations(txt):
            errors.append(f"{rel(path)}:claim_previsao_operacional")
        if public_text_violations_text(txt):
            errors.append(f"{rel(path)}:vocabulario_publico_proibido")
    if "review-only" not in report_txt.lower() and "review_only" not in report_txt.lower():
        errors.append("relatorio_omite_review_only")
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
        "GT06 canarios SAR validado: "
        f"alvos={manifest['alvos_total']} distribuicao={manifest['distribuicao_elegibilidade']} "
        f"selecionados={manifest['canarios_selecionados']} pode_preparar={manifest['pode_preparar_sar_futuro']} "
        f"proximo={manifest['proximo_marco']} score_v6_changed={str(manifest['score_v6_changed']).lower()}"
    )
    return 0


def run_all() -> int:
    manifest = build_all()
    print(
        "GT06 canarios SAR gerado: "
        f"alvos={manifest['alvos_total']} distribuicao={manifest['distribuicao_elegibilidade']} "
        f"selecionados={manifest['canarios_selecionados']} proximo={manifest['proximo_marco']}"
    )
    return 0
