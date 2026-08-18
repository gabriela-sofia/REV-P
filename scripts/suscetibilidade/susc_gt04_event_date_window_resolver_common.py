"""SUSC-GT04 - Resolvedor de Datas e Janelas Pre/Pos-Evento.

Le o pacote de alvos do SUSC-GT03 (e, quando necessario, o replay do SUSC-GT02) e
resolve, infere ou classifica datas candidatas usando APENAS os artefatos ja
existentes no repositorio. Para cada alvo, atribui uma precisao temporal, um score
de confianca temporal (0 a 100) e, quando a data e suficientemente precisa, gera
janelas pre-evento e pos-evento para aquisicao FUTURA.

Este marco NAO busca internet, NAO baixa dado novo, NAO roda SAR, NAO roda GEE, NAO
cria footprint, NAO altera o score_v6, NAO cria score_v7, NAO treina modelo e NAO
altera as decisoes do GT02 nem as prioridades do GT03 (apenas as le). Nenhuma janela
e usada para executar SAR aqui: o output apenas prepara metadata. Nada e promovido a
ground truth nem a positive_strong.

Ver [[susc_gt03_strong_evidence_target_pack]], [[susc_gt02_ground_reference_replay]] e
[[susc_gt01_ground_reference_policy]]. Identificadores tecnicos em ingles sao mantidos
por compatibilidade com os marcos anteriores, schemas e validadores; o relatorio
publico explica cada campo em portugues.
"""

from __future__ import annotations

import csv
import datetime as _dt
import re
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
import susc_gt03_strong_evidence_target_pack_common as gt03  # noqa: E402

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS_DIR = ROOT / "schemas" / "suscetibilidade"

# Entradas GT03 (principais) e GT02 (auxiliares) -- lidas, nunca regeradas.
GT03_TARGETS = OUT / "susc_gt03_pacote_alvos_evidencia_forte.csv"
GT03_MISSING = OUT / "susc_gt03_requisitos_faltantes_por_alvo.csv"
GT03_TRACKS = OUT / "susc_gt03_trilhas_aquisicao.csv"
GT03_PLAN = OUT / "susc_gt03_plano_aquisicao_por_alvo.csv"
GT03_MANIFEST = OUT / "susc_gt03_manifesto.json"
GT02_REPLAY = OUT / "susc_gt02_registro_replay_evidencias.csv"

REQUIRED_INPUTS = [GT03_TARGETS, GT03_MISSING, GT03_TRACKS, GT03_PLAN, GT03_MANIFEST, GT02_REPLAY]

# Saidas publicas (portugues)
REPORT = OUT / "SUSC_GT04_RESOLVEDOR_DATAS_JANELAS_PRE_POS_EVENTO.md"
DATES_CSV = OUT / "susc_gt04_datas_resolvidas_por_alvo.csv"
WINDOWS_CSV = OUT / "susc_gt04_janelas_pre_pos_evento.csv"
DIAG_CSV = OUT / "susc_gt04_diagnostico_temporal.csv"
BLOCKERS_CSV = OUT / "susc_gt04_bloqueios_temporais.csv"
SUMMARY_CSV = OUT / "susc_gt04_resumo_precisao_temporal.csv"
RELEASED_CSV = OUT / "susc_gt04_alvos_liberados_para_proxima_aquisicao.csv"
MANIFEST_JSON = OUT / "susc_gt04_manifesto.json"

SCHEMA_DATE = SCHEMAS_DIR / "susc_gt04_resolved_event_date.schema.json"
SCHEMA_WINDOW = SCHEMAS_DIR / "susc_gt04_event_window.schema.json"
SCHEMA_DIAG = SCHEMAS_DIR / "susc_gt04_temporal_diagnostic.schema.json"
SCHEMA_SUMMARY = SCHEMAS_DIR / "susc_gt04_temporal_summary.schema.json"

PUBLIC_ARTIFACTS = [
    REPORT, DATES_CSV, WINDOWS_CSV, DIAG_CSV, BLOCKERS_CSV, SUMMARY_CSV, RELEASED_CSV,
]
REQUIRED_OUTPUTS = PUBLIC_ARTIFACTS + [
    MANIFEST_JSON, SCHEMA_DATE, SCHEMA_WINDOW, SCHEMA_DIAG, SCHEMA_SUMMARY,
]

MARCO = "SUSC-GT04"
TITULO_PUBLICO = "Resolvedor de Datas e Janelas Pré/Pós-Evento"
TITULO_H1_TOKEN = "Datas e Janelas Pré/Pós-Evento"

PUBLIC_FORBIDDEN_RE = gt03.PUBLIC_FORBIDDEN_RE
OPERATIONAL_FORECAST_RE = gt03.OPERATIONAL_FORECAST_RE

GLOBAL_GUARDRAILS = {
    "eligible_for_training": "false",
    "eligible_for_ground_truth": "false",
    "trainable": "false",
    "score_v7_candidate": "false",
    "review_only": "true",
}

# ---------------------------------------------------------------------------
# Precisao temporal / confianca
# ---------------------------------------------------------------------------
TEMPORAL_PRECISIONS = [
    "exact_day",
    "inferred_from_event_id",
    "publication_date_only",
    "date_range",
    "month_only",
    "year_only",
    "unknown",
    "invalid_or_conflicting",
]
CONFIDENCE_BY_PRECISION = {
    "exact_day": 100,
    "inferred_from_event_id": 75,
    "date_range": 80,
    "publication_date_only": 45,
    "month_only": 30,
    "year_only": 15,
    "unknown": 0,
    "invalid_or_conflicting": 0,
}
USABLE_WINDOW_PRECISIONS = {"exact_day", "inferred_from_event_id", "date_range"}

PRE_DAYS = 30
POST_DAYS = 7

# Campos textuais varridos por padroes de data (hierarquia de confianca).
DATE_TEXT_FIELDS = [
    "event_date", "event_date_candidate", "source_date", "date",
    "publication_date", "event_id", "candidate_event_id", "source_artifact",
    "source_url", "filename", "notes", "blocking_reason", "rationale_pt",
]

# ---------------------------------------------------------------------------
# Regex de datas
# ---------------------------------------------------------------------------
_FULL_RE = re.compile(r"(?<!\d)(19|20)(\d{2})[-_/.](\d{1,2})[-_/.](\d{1,2})(?!\d)")
_MONTH_RE = re.compile(r"(?<!\d)(19|20)(\d{2})[-_/.](\d{1,2})(?!\d)")
_YEAR_RE = re.compile(r"(?<!\d)(19|20)(\d{2})(?!\d)")


def _valid_ymd(y: int, m: int, d: int) -> bool:
    try:
        _dt.date(y, m, d)
        return True
    except ValueError:
        return False


def extract_full_dates(text: str) -> tuple[list[str], bool]:
    """Retorna (datas_completas_validas_iso, houve_padrao_invalido)."""
    valid, invalid_found = [], False
    for mth in _FULL_RE.finditer(text or ""):
        y = int(mth.group(1) + mth.group(2))
        mo, da = int(mth.group(3)), int(mth.group(4))
        if _valid_ymd(y, mo, da):
            valid.append(f"{y:04d}-{mo:02d}-{da:02d}")
        else:
            invalid_found = True
    return valid, invalid_found


def extract_month(text: str) -> str | None:
    m = _MONTH_RE.search(text or "")
    if not m:
        return None
    y, mo = int(m.group(1) + m.group(2)), int(m.group(3))
    if 1 <= mo <= 12:
        return f"{y:04d}-{mo:02d}"
    return None


def extract_year(text: str) -> str | None:
    m = _YEAR_RE.search(text or "")
    return (m.group(1) + m.group(2)) if m else None


# ---------------------------------------------------------------------------
# Resolvedor de data (puro)
# ---------------------------------------------------------------------------
def _present(v) -> bool:
    return gt03._clean(v).lower() not in gt03.gt02.NULLISH


def resolve_date(fields: dict) -> dict:
    """Resolve a melhor data a partir dos campos disponiveis, seguindo a
    hierarquia de confianca. Puro e deterministico."""
    def g(k):
        return gt03._clean(fields.get(k))

    # date_range explicito
    rng_start, rng_end = g("date_range_start"), g("date_range_end")
    if rng_start and rng_end:
        vs, _ = extract_full_dates(rng_start)
        ve, _ = extract_full_dates(rng_end)
        if vs and ve and vs[0] <= ve[0]:
            return _result("date_range", "date_range", "campo_date_range_explicito",
                           f"{vs[0]}..{ve[0]}", resolved="", rng=(vs[0], ve[0]))

    # 1) event_date explicito
    for field, method, precision, inferred in [
        ("event_date", "campo_event_date_explicito", "exact_day", False),
        ("event_date_candidate", "campo_event_date_candidate_explicito", "exact_day", False),
    ]:
        if _present(fields.get(field)):
            valid, invalid = extract_full_dates(g(field))
            if len(set(valid)) > 1:
                return _result("invalid_or_conflicting", field, method, g(field), conflict=True)
            if valid:
                return _result(precision, field, method, g(field), resolved=valid[0], inferred=inferred)
            if invalid:
                return _result("invalid_or_conflicting", field, method, g(field), conflict=True)

    # 2) source_date / date com sinal de evento (tratado como campo explicito fraco)
    for field in ("source_date", "date"):
        if _present(fields.get(field)):
            valid, _ = extract_full_dates(g(field))
            if valid:
                return _result("exact_day", field, "campo_data_com_sinal_de_evento", g(field),
                               resolved=valid[0], confidence_override=70)

    # 3) inferencia por event_id / candidate_event_id
    for field in ("event_id", "candidate_event_id"):
        if _present(fields.get(field)):
            valid, _ = extract_full_dates(g(field))
            if len(set(valid)) > 1:
                return _result("invalid_or_conflicting", field, "inferencia_de_event_id", g(field), conflict=True)
            if valid:
                return _result("inferred_from_event_id", field, "inferencia_de_event_id",
                               g(field), resolved=valid[0], inferred=True)

    # 4) data em source_artifact / filename (inferida)
    for field in ("source_artifact", "filename"):
        if _present(fields.get(field)):
            valid, _ = extract_full_dates(g(field))
            if valid:
                return _result("inferred_from_event_id", field, "extraida_de_nome_de_arquivo",
                               g(field), resolved=valid[0], inferred=True, confidence_override=60)

    # 5) data de publicacao
    for field in ("publication_date", "source_url"):
        if _present(fields.get(field)):
            valid, _ = extract_full_dates(g(field))
            if valid:
                return _result("publication_date_only", field, "data_de_publicacao_da_fonte",
                               g(field), resolved=valid[0])

    # 6) mes/ano parcial em qualquer campo textual
    for field in DATE_TEXT_FIELDS:
        if not _present(fields.get(field)):
            continue
        mo = extract_month(g(field))
        if mo:
            return _result("month_only", field, "mes_ano_parcial", mo)
    for field in DATE_TEXT_FIELDS:
        if not _present(fields.get(field)):
            continue
        yr = extract_year(g(field))
        if yr:
            return _result("year_only", field, "ano_parcial", yr)

    return _result("unknown", "", "nenhuma_data_util", "")


def _result(precision, field, method, value, resolved="", inferred=False,
            conflict=False, rng=("", ""), confidence_override=None) -> dict:
    conf = CONFIDENCE_BY_PRECISION[precision]
    if confidence_override is not None:
        conf = confidence_override
    return {
        "temporal_precision": precision,
        "date_source_field": field or "none",
        "date_source_value": value or "none",
        "date_resolution_method": method,
        "resolved_event_date": resolved,
        "date_range_start": rng[0],
        "date_range_end": rng[1],
        "temporal_confidence_score": conf,
        "date_is_inferred": "true" if inferred else "false",
        "date_is_publication_only": "true" if precision == "publication_date_only" else "false",
        "date_conflict_detected": "true" if conflict else "false",
        "temporal_blocking_reason": _temporal_blocking_reason(precision),
    }


def _temporal_blocking_reason(precision: str) -> str:
    return {
        "exact_day": "",
        "inferred_from_event_id": "",
        "date_range": "",
        "publication_date_only": "data apenas de publicacao da fonte, nao do evento; nao libera SAR",
        "month_only": "apenas mes/ano conhecido; resolucao insuficiente para janela operacional",
        "year_only": "apenas ano conhecido; resolucao insuficiente para janela operacional",
        "unknown": "nenhuma data util encontrada nos artefatos existentes",
        "invalid_or_conflicting": "datas invalidas ou conflitantes entre fontes",
    }.get(precision, "")


# ---------------------------------------------------------------------------
# Janelas pre/pos-evento
# ---------------------------------------------------------------------------
def _iso(d: _dt.date) -> str:
    return d.isoformat()


def _parse(s: str) -> _dt.date | None:
    try:
        return _dt.date.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def build_window(res: dict) -> dict:
    precision = res["temporal_precision"]
    base = {
        "resolved_event_date": res["resolved_event_date"] or "not_available",
        "temporal_precision": precision,
        "pre_event_window_start": "", "pre_event_window_end": "",
        "post_event_window_start": "", "post_event_window_end": "",
        "window_policy": "", "window_status": "",
        "usable_for_sar_candidate_future": "false",
        "usable_for_official_geometry_search_future": "false",
        "usable_for_qa_future": "false",
        "window_blocking_reason": "",
        "no_sar_execution_in_this_milestone": "true",
        "no_network_in_this_milestone": "true",
        "review_only": "true",
    }
    if precision in ("exact_day", "inferred_from_event_id"):
        d = _parse(res["resolved_event_date"])
        if d:
            base.update({
                "pre_event_window_start": _iso(d - _dt.timedelta(days=PRE_DAYS)),
                "pre_event_window_end": _iso(d - _dt.timedelta(days=1)),
                "post_event_window_start": _iso(d),
                "post_event_window_end": _iso(d + _dt.timedelta(days=POST_DAYS)),
                "window_policy": f"pre=[-{PRE_DAYS}d,-1d] pos=[0,+{POST_DAYS}d]",
                "window_status": "usable",
                "usable_for_sar_candidate_future": "true",
                "usable_for_official_geometry_search_future": "true",
                "usable_for_qa_future": "true",
            })
            return base
    if precision == "date_range":
        ds, de = _parse(res["date_range_start"]), _parse(res["date_range_end"])
        if ds and de:
            base.update({
                "resolved_event_date": res["date_range_start"],
                "pre_event_window_start": _iso(ds - _dt.timedelta(days=PRE_DAYS)),
                "pre_event_window_end": _iso(ds - _dt.timedelta(days=1)),
                "post_event_window_start": _iso(ds),
                "post_event_window_end": _iso(de + _dt.timedelta(days=POST_DAYS)),
                "window_policy": f"pre=[range_start-{PRE_DAYS}d,-1d] pos=[range_start,range_end+{POST_DAYS}d]",
                "window_status": "usable_with_range_uncertainty",
                "usable_for_sar_candidate_future": "true",
                "usable_for_official_geometry_search_future": "true",
                "usable_for_qa_future": "true",
            })
            return base
    if precision == "publication_date_only":
        base.update({
            "window_policy": "sem janela SAR; data apenas de publicacao",
            "window_status": "usable_for_documentary_context_only",
            "usable_for_qa_future": "true",
            "window_blocking_reason": "data de publicacao nao e data do evento; nao libera SAR",
        })
        return base
    # month_only / year_only / unknown / invalid_or_conflicting
    base.update({
        "window_policy": "sem janela operacional",
        "window_status": "blocked_temporal_resolution_insufficient",
        "window_blocking_reason": _temporal_blocking_reason(precision) or "resolucao temporal insuficiente",
    })
    return base


# ---------------------------------------------------------------------------
# Leitura das entradas
# ---------------------------------------------------------------------------
def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        return list(csv.DictReader(f))


def require_inputs() -> None:
    missing = [rel(p) for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("entradas GT03/GT02 ausentes: " + "; ".join(missing))


# ---------------------------------------------------------------------------
# Construcao das linhas
# ---------------------------------------------------------------------------
def build_rows() -> dict:
    targets = _read_csv(GT03_TARGETS)
    replay_by_id = {r["replay_id"]: r for r in _read_csv(GT02_REPLAY)}

    dates, windows, diagnostics, blockers, released = [], [], [], [], []
    b_idx = 1

    for t in targets:
        rp = replay_by_id.get(t.get("source_replay_id", ""), {})
        # campos de data a partir do GT03 (normalizado) + GT02 (original)
        fields = {
            "event_date": _prefer(rp.get("event_date"), t.get("event_date")),
            "event_id": _prefer(rp.get("event_id"), t.get("event_id")),
            "candidate_event_id": t.get("event_id", ""),
            "blocking_reason": rp.get("blocking_reason", ""),
            "rationale_pt": t.get("rationale_pt", ""),
            "source_authority": t.get("source_authority", ""),
        }
        res = resolve_date(fields)
        win = build_window(res)

        target_id = t["target_id"]
        precision = res["temporal_precision"]
        had_missing_date = "event_date" in t.get("missing_requirements", "")
        resolved_now = precision in USABLE_WINDOW_PRECISIONS

        dates.append({
            "target_id": target_id,
            "source_replay_id": t.get("source_replay_id", ""),
            "source_decision_id": t.get("source_decision_id", ""),
            "event_id": t.get("event_id", "not_available"),
            "candidate_event_id": t.get("event_id", "not_available"),
            "patch_id": t.get("patch_id", "not_available"),
            "city": t.get("city", "not_available"),
            "region": t.get("region", "not_available"),
            "assigned_state_gt02": t.get("assigned_state_gt02", ""),
            "priority_class_gt03": t.get("priority_class", ""),
            "event_date_original": t.get("event_date", "not_available"),
            "resolved_event_date": res["resolved_event_date"] or "not_available",
            "date_range_start": res["date_range_start"] or "not_available",
            "date_range_end": res["date_range_end"] or "not_available",
            "temporal_precision": precision,
            "temporal_confidence_score": str(res["temporal_confidence_score"]),
            "date_source_field": res["date_source_field"],
            "date_source_value": res["date_source_value"],
            "date_resolution_method": res["date_resolution_method"],
            "date_is_inferred": res["date_is_inferred"],
            "date_is_publication_only": res["date_is_publication_only"],
            "date_conflict_detected": res["date_conflict_detected"],
            "temporal_blocking_reason": res["temporal_blocking_reason"],
            "remains_review_only": "true",
            "trainable": "false",
            "eligible_for_training": "false",
            "eligible_for_ground_truth": "false",
            "score_v7_candidate": "false",
            "rationale_pt": _date_rationale(res),
        })

        windows.append({"window_id": f"WIN_{_seq(target_id)}", "target_id": target_id, **win})

        # diagnostico
        still_missing = _still_missing(t.get("missing_requirements", ""), resolved_now)
        next_track = _next_track(still_missing)
        diagnostics.append({
            "diagnostic_id": f"DIA_{_seq(target_id)}",
            "target_id": target_id,
            "had_missing_event_date_in_gt03": "true" if had_missing_date else "false",
            "event_date_resolved_now": "true" if resolved_now else "false",
            "resolution_level": _resolution_level(precision),
            "still_missing_requirements": ";".join(still_missing) if still_missing else "none",
            "next_required_track": next_track,
            "can_advance_to_geometry_acquisition": "true" if resolved_now else "false",
            "can_advance_to_sar_feasibility_future": "true" if win["usable_for_sar_candidate_future"] == "true" else "false",
            "can_advance_to_qa_future": "true" if precision not in ("unknown", "invalid_or_conflicting") else "false",
            "prevents_positive_strong": "true" if still_missing else "false",
            "prevents_calibration": "true" if still_missing else "false",
            "prevents_benchmark": "true",
            "reason_pt": _diag_reason(precision, resolved_now, still_missing),
        })

        # bloqueios temporais
        if precision not in USABLE_WINDOW_PRECISIONS:
            blockers.append({
                "temporal_blocker_id": f"TBL_{b_idx:05d}",
                "target_id": target_id,
                "blocker_type": _blocker_type(precision),
                "severity": "alta" if precision in ("unknown", "invalid_or_conflicting") else "media",
                "source_field": res["date_source_field"],
                "source_value": res["date_source_value"],
                "reason_pt": res["temporal_blocking_reason"] or "resolucao temporal insuficiente",
                "prevents_window_generation": "true" if precision != "publication_date_only" else "false",
                "prevents_sar_candidate_future": "true",
                "prevents_positive_strong": "true",
                "how_to_resolve_pt": _how_to_resolve(precision),
            })
            b_idx += 1

        # liberados para proxima aquisicao
        if resolved_now:
            released.append({
                "released_target_id": f"REL_{_seq(target_id)}",
                "target_id": target_id,
                "resolved_event_date": res["resolved_event_date"] or res["date_range_start"] or "not_available",
                "temporal_precision": precision,
                "priority_class_gt03": t.get("priority_class", ""),
                "next_recommended_milestone": "GT05 - Pacote de Aquisicao de Geometria Oficial",
                "next_recommended_action_pt": _released_action(next_track),
                "still_missing_requirements": ";".join(still_missing) if still_missing else "none",
                "release_type": "data_resolvida_liberada_para_geometria",
                "remains_review_only": "true",
                "not_ground_truth_reason": "data resolvida review-only; nao e verdade de campo nem habilita treino",
            })

    return {
        "dates": dates, "windows": windows, "diagnostics": diagnostics,
        "blockers": blockers, "released": released,
    }


def _prefer(a, b) -> str:
    a, b = gt03._clean(a), gt03._clean(b)
    if a and a.lower() not in gt03.gt02.NULLISH:
        return a
    return b


def _seq(target_id: str) -> str:
    m = re.search(r"(\d+)$", target_id)
    return m.group(1) if m else target_id


def _still_missing(missing_str: str, resolved_now: bool) -> list[str]:
    reqs = [r for r in missing_str.split(";") if r and r != "none"]
    if resolved_now and "event_date" in reqs:
        reqs = [r for r in reqs if r != "event_date"]
    return reqs


def _next_track(still_missing: list[str]) -> str:
    if "geometry_type" in still_missing:
        return "obter_geometria_oficial"
    if "patch_link_quality" in still_missing or "patch_id" in still_missing:
        return "resolver_patch_link"
    if "uncertainty_m" in still_missing:
        return "estimar_incerteza_espacial"
    if "qa_status" in still_missing:
        return "executar_qa_humano"
    if "phenomenon_type" in still_missing:
        return "separar_fenomeno"
    return "nenhuma_pendencia_forte"


def _resolution_level(precision: str) -> str:
    return {
        "exact_day": "resolvida_exata",
        "inferred_from_event_id": "resolvida_inferida",
        "date_range": "resolvida_intervalo",
        "publication_date_only": "apenas_publicacao",
        "month_only": "parcial_mes",
        "year_only": "parcial_ano",
        "unknown": "nao_resolvida",
        "invalid_or_conflicting": "conflito",
    }.get(precision, "nao_resolvida")


def _blocker_type(precision: str) -> str:
    return {
        "publication_date_only": "apenas_data_publicacao",
        "month_only": "precisao_mes_apenas",
        "year_only": "precisao_ano_apenas",
        "unknown": "data_ausente",
        "invalid_or_conflicting": "data_invalida_ou_conflitante",
    }.get(precision, "resolucao_insuficiente")


def _how_to_resolve(precision: str) -> str:
    return {
        "publication_date_only": "buscar data do evento (nao da publicacao) em registro documental existente",
        "month_only": "refinar dia do evento a partir de registro documental ou event_id",
        "year_only": "refinar mes e dia do evento a partir de registro documental",
        "unknown": "enriquecimento documental manual para obter a data do evento",
        "invalid_or_conflicting": "reconciliar datas conflitantes e validar a data correta do evento",
    }.get(precision, "revisar a fonte de data")


def _released_action(next_track: str) -> str:
    return {
        "obter_geometria_oficial": "avancar para aquisicao de geometria oficial usando a janela temporal",
        "resolver_patch_link": "avancar para resolucao de patch_link com a data resolvida",
        "estimar_incerteza_espacial": "avancar para estimativa de incerteza espacial",
        "executar_qa_humano": "avancar para QA humano com a data resolvida",
        "separar_fenomeno": "avancar para separacao de fenomeno",
        "nenhuma_pendencia_forte": "revisar caso a caso; poucas pendencias fortes restantes",
    }.get(next_track, "avancar conforme requisito faltante")


def _date_rationale(res: dict) -> str:
    p = res["temporal_precision"]
    if p == "exact_day":
        return "data completa extraida de campo explicito; util para janela e aquisicao futura"
    if p == "inferred_from_event_id":
        return "data completa inferida do identificador do evento; marcada como inferida, nao confirmada"
    if p == "date_range":
        return "intervalo temporal com inicio e fim; janela gerada com incerteza de intervalo"
    if p == "publication_date_only":
        return "apenas data de publicacao da fonte; serve para contexto documental, nao libera SAR"
    if p == "invalid_or_conflicting":
        return "datas invalidas ou conflitantes; nao gera janela ate reconciliacao"
    if p in ("month_only", "year_only"):
        return "precisao parcial; insuficiente para janela operacional"
    return "nenhuma data util encontrada nos artefatos existentes"


def _diag_reason(precision: str, resolved_now: bool, still_missing: list[str]) -> str:
    if resolved_now:
        falta = ", ".join(still_missing) if still_missing else "nenhuma pendencia forte"
        return f"data resolvida ({precision}); pendencias restantes: {falta}"
    return f"data nao resolvida em nivel operacional ({precision}); segue como gargalo temporal"


# ---------------------------------------------------------------------------
# Resumo por precisao
# ---------------------------------------------------------------------------
def summary_rows(dates: list[dict], windows: list[dict]) -> list[dict]:
    order = {p: i for i, p in enumerate(TEMPORAL_PRECISIONS)}
    win_by_target = {w["target_id"]: w for w in windows}
    agg = {}
    for d in dates:
        p = d["temporal_precision"]
        a = agg.setdefault(p, {"count": 0, "conf": 0, "usable": 0, "blocked": 0})
        a["count"] += 1
        a["conf"] += int(d["temporal_confidence_score"])
        w = win_by_target.get(d["target_id"], {})
        if w.get("window_status") in ("usable", "usable_with_range_uncertainty"):
            a["usable"] += 1
        else:
            a["blocked"] += 1
    rows = []
    for p in sorted(agg, key=lambda x: order.get(x, 99)):
        a = agg[p]
        rows.append({
            "temporal_precision": p,
            "count": str(a["count"]),
            "average_temporal_confidence_score": str(round(a["conf"] / a["count"], 2)) if a["count"] else "0",
            "usable_window_count": str(a["usable"]),
            "blocked_window_count": str(a["blocked"]),
            "eligible_for_training_count": "0",
            "eligible_for_ground_truth_count": "0",
            "score_v7_candidate_count": "0",
        })
    return rows


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
_BOOL = {"enum": ["true", "false"]}
_FALSE = {"const": "false"}
_DATE_OR_NA = {"type": "string", "pattern": r"^(\d{4}-\d{2}-\d{2}|not_available)$"}
_SCORE = {"type": "string", "pattern": r"^(100|[1-9]?[0-9])$"}


def schema_date_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "susc_gt04_resolved_event_date.schema.json",
        "title": "SUSC-GT04 Data de Evento Resolvida",
        "type": "object",
        "required": [
            "target_id", "temporal_precision", "temporal_confidence_score",
            "resolved_event_date", "date_is_inferred", "remains_review_only",
            "trainable", "eligible_for_training", "eligible_for_ground_truth",
            "score_v7_candidate",
        ],
        "properties": {
            "target_id": {"type": "string", "pattern": r"^TGT_\d+$"},
            "temporal_precision": {"enum": TEMPORAL_PRECISIONS},
            "temporal_confidence_score": _SCORE,
            "resolved_event_date": _DATE_OR_NA,
            "date_range_start": _DATE_OR_NA,
            "date_range_end": _DATE_OR_NA,
            "date_is_inferred": _BOOL,
            "date_is_publication_only": _BOOL,
            "date_conflict_detected": _BOOL,
            "remains_review_only": {"const": "true"},
            "trainable": _FALSE,
            "eligible_for_training": _FALSE,
            "eligible_for_ground_truth": _FALSE,
            "score_v7_candidate": _FALSE,
        },
    }


def schema_window_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "susc_gt04_event_window.schema.json",
        "title": "SUSC-GT04 Janela de Evento",
        "type": "object",
        "required": [
            "window_id", "target_id", "temporal_precision", "window_status",
            "usable_for_sar_candidate_future", "no_sar_execution_in_this_milestone",
            "no_network_in_this_milestone", "review_only",
        ],
        "properties": {
            "window_id": {"type": "string", "pattern": r"^WIN_\d+$"},
            "target_id": {"type": "string", "pattern": r"^TGT_\d+$"},
            "temporal_precision": {"enum": TEMPORAL_PRECISIONS},
            "pre_event_window_start": {"type": "string"},
            "pre_event_window_end": {"type": "string"},
            "post_event_window_start": {"type": "string"},
            "post_event_window_end": {"type": "string"},
            "window_status": {"enum": [
                "usable", "usable_with_range_uncertainty",
                "usable_for_documentary_context_only",
                "blocked_temporal_resolution_insufficient",
            ]},
            "usable_for_sar_candidate_future": _BOOL,
            "usable_for_official_geometry_search_future": _BOOL,
            "usable_for_qa_future": _BOOL,
            "no_sar_execution_in_this_milestone": {"const": "true"},
            "no_network_in_this_milestone": {"const": "true"},
            "review_only": {"const": "true"},
        },
    }


def schema_diag_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "susc_gt04_temporal_diagnostic.schema.json",
        "title": "SUSC-GT04 Diagnostico Temporal",
        "type": "object",
        "required": [
            "diagnostic_id", "target_id", "event_date_resolved_now",
            "resolution_level", "next_required_track", "reason_pt",
        ],
        "properties": {
            "diagnostic_id": {"type": "string", "pattern": r"^DIA_\d+$"},
            "target_id": {"type": "string", "pattern": r"^TGT_\d+$"},
            "had_missing_event_date_in_gt03": _BOOL,
            "event_date_resolved_now": _BOOL,
            "resolution_level": {"type": "string", "minLength": 1},
            "can_advance_to_geometry_acquisition": _BOOL,
            "can_advance_to_sar_feasibility_future": _BOOL,
            "can_advance_to_qa_future": _BOOL,
            "prevents_positive_strong": _BOOL,
            "reason_pt": {"type": "string", "minLength": 1},
        },
    }


def schema_summary_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "susc_gt04_temporal_summary.schema.json",
        "title": "SUSC-GT04 Resumo de Precisao Temporal",
        "type": "object",
        "required": [
            "temporal_precision", "count", "average_temporal_confidence_score",
            "usable_window_count", "blocked_window_count",
            "eligible_for_training_count", "eligible_for_ground_truth_count",
            "score_v7_candidate_count",
        ],
        "properties": {
            "temporal_precision": {"enum": TEMPORAL_PRECISIONS},
            "count": {"type": "string", "pattern": r"^\d+$"},
            "average_temporal_confidence_score": {"type": "string"},
            "usable_window_count": {"type": "string", "pattern": r"^\d+$"},
            "blocked_window_count": {"type": "string", "pattern": r"^\d+$"},
            "eligible_for_training_count": {"const": "0"},
            "eligible_for_ground_truth_count": {"const": "0"},
            "score_v7_candidate_count": {"const": "0"},
        },
    }


# ---------------------------------------------------------------------------
# Manifesto / proximo marco
# ---------------------------------------------------------------------------
def _next_milestone(dist: dict, total: int) -> str:
    usable = sum(dist.get(p, 0) for p in USABLE_WINDOW_PRECISIONS)
    unknown_pub = dist.get("unknown", 0) + dist.get("publication_date_only", 0) + \
        dist.get("month_only", 0) + dist.get("year_only", 0) + dist.get("invalid_or_conflicting", 0)
    if total and usable >= 0.4 * total:
        return "GT05 - Pacote de Aquisicao de Geometria Oficial"
    if usable >= 30:
        return "GT05 - Preparacao de Canarios SAR sem Execucao"
    if unknown_pub > usable:
        return "GT05 - Enriquecimento Documental Manual"
    return "GT05 - Pacote de Aquisicao de Geometria Oficial"


def manifest_obj(bundle: dict) -> dict:
    dates = bundle["dates"]
    dist = {}
    for d in dates:
        dist[d["temporal_precision"]] = dist.get(d["temporal_precision"], 0) + 1
    usable_windows = sum(1 for w in bundle["windows"]
                         if w["window_status"] in ("usable", "usable_with_range_uncertainty"))
    artifacts = [
        {"path": rel(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in PUBLIC_ARTIFACTS + [SCHEMA_DATE, SCHEMA_WINDOW, SCHEMA_DIAG, SCHEMA_SUMMARY]
        if p.exists()
    ]
    return {
        "marco": MARCO,
        "titulo_publico": TITULO_PUBLICO,
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "gt03_source_commit": _run_git(["log", "-1", "--format=%h", "--", rel(GT03_TARGETS)]) or "unknown",
        "alvos_total": len(dates),
        "distribuicao_precisao": dist,
        "datas_resolvidas_exatas": dist.get("exact_day", 0),
        "datas_inferidas": dist.get("inferred_from_event_id", 0),
        "datas_intervalo": dist.get("date_range", 0),
        "apenas_publicacao": dist.get("publication_date_only", 0),
        "unknown": dist.get("unknown", 0),
        "invalidas_ou_conflitantes": dist.get("invalid_or_conflicting", 0),
        "janelas_geradas": usable_windows,
        "janelas_bloqueadas": len(dates) - usable_windows,
        "alvos_liberados": len(bundle["released"]),
        "positive_strong_promovidos": 0,
        "global_guardrails": GLOBAL_GUARDRAILS,
        "training_true_count": sum(1 for d in dates if d["eligible_for_training"] == "true"),
        "ground_truth_true_count": sum(1 for d in dates if d["eligible_for_ground_truth"] == "true"),
        "score_v7_candidate_true_count": sum(1 for d in dates if d["score_v7_candidate"] == "true"),
        "score_v6_changed": gt03.gt02.gt01._score_v6_changed(),
        "score_v7_created": gt03.gt02.gt01._score_v7_exists(),
        "proximo_marco": _next_milestone(dist, len(dates)),
        "artifacts": artifacts,
    }


def _run_git(args: list[str]) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return r.stdout.strip() if r.returncode == 0 else ""


# ---------------------------------------------------------------------------
# Relatorio publico (portugues)
# ---------------------------------------------------------------------------
def _example(dates: list[dict], precision: str):
    return next((d for d in dates if d["temporal_precision"] == precision), None)


def report_text(manifest: dict, bundle: dict, summary: list[dict]) -> str:
    dates = bundle["dates"]
    linhas_prec = "\n".join(
        f"| {r['temporal_precision']} | {r['count']} | {r['average_temporal_confidence_score']} | "
        f"{r['usable_window_count']} | {r['blocked_window_count']} |"
        for r in summary
    )

    def ex(precision, titulo):
        d = _example(dates, precision)
        if not d:
            return f"- **{titulo}** (`{precision}`): nenhum alvo neste nivel."
        return (f"- **{titulo}** (`{precision}`): alvo `{d['target_id']}`, event_id `{d['event_id']}`, "
                f"data resolvida `{d['resolved_event_date']}`, confianca {d['temporal_confidence_score']}, "
                f"método `{d['date_resolution_method']}`.")

    return f"""# {MARCO} - {TITULO_PUBLICO}

## 1. Escopo do marco

Este marco resolve, infere ou classifica **datas de evento** e gera **janelas
pré-evento e pós-evento** para aquisição futura, usando apenas os artefatos que já
existem no repositório. É um resolvedor **review-only** (uso restrito a revisão): não
busca internet, não baixa dado novo, não roda SAR nem GEE, não cria footprint, não
altera o `score_v6`, não cria `score_v7`, não treina modelo e não promove nada a ground
truth nem a `positive_strong`. As janelas geradas são apenas metadata; nenhuma é usada
para executar SAR agora.

## 2. Relação com GT01, GT02 e GT03

O GT01 definiu a política, o GT02 aplicou-a às evidências existentes e o GT03 montou o
pacote de alvos, apontando a **ausência de `event_date`** (data do evento) como maior
gargalo. O GT04 ataca exatamente esse gargalo, lendo os alvos do GT03 e o replay do GT02.

## 3. Por que resolver data vem antes de SAR, geometria e QA

Sem a data do evento não há janela temporal pré/pós-evento; sem janela não é possível
buscar geometria oficial por período, preparar canários SAR nem contextualizar o QA
humano. A data é o requisito **upstream** que desbloqueia os demais.

## 4. Entradas usadas

Alvos e requisitos do GT03 e o replay do GT02 (campos originais), lidos dos outputs
versionados sem regeração. Total de alvos processados: **{manifest['alvos_total']}**.

## 5. Hierarquia de fontes de data

Ordem de confiança: (1) `event_date` explícito; (2) `event_date_candidate`; (3)
`date`/`source_date` com sinal de evento; (4) data inferida de
`event_id`/`candidate_event_id`; (5) data em nome de arquivo/`source_artifact`; (6) data
de publicação; (7) ano/mês parcial; (8) `unknown` (desconhecida). Cada resolução
registra o campo-fonte, o valor-fonte, o método, a precisão e a confiança.

## 6. Categorias de precisão temporal

`exact_day` (dia exato de campo explícito), `inferred_from_event_id` (dia inferido do
identificador — marcado como inferido, não confirmado), `publication_date_only` (apenas
data de publicação, **não** libera SAR), `date_range` (intervalo início–fim),
`month_only`, `year_only`, `unknown` e `invalid_or_conflicting` (datas inválidas ou
conflitantes).

## 7. Política de janelas pré/pós-evento

Para `exact_day`/`inferred_from_event_id`: pré-evento = [data−{PRE_DAYS} dias, data−1] e
pós-evento = [data, data+{POST_DAYS} dias]. Para `date_range`: pré = [início−{PRE_DAYS},
início−1] e pós = [início, fim+{POST_DAYS}], com incerteza de intervalo. Para
`publication_date_only`: sem janela SAR, apenas contexto documental. Para
`month_only`/`year_only`/`unknown`/`invalid_or_conflicting`: sem janela operacional,
bloqueada por resolução insuficiente.

## 8-13. Distribuição da resolução

- Datas resolvidas exatas (`exact_day`): **{manifest['datas_resolvidas_exatas']}**.
- Datas inferidas (`inferred_from_event_id`): **{manifest['datas_inferidas']}**.
- Intervalos (`date_range`): **{manifest['datas_intervalo']}**.
- Apenas publicação (`publication_date_only`): **{manifest['apenas_publicacao']}**.
- Desconhecidas (`unknown`): **{manifest['unknown']}**.
- Inválidas/conflitantes (`invalid_or_conflicting`): **{manifest['invalidas_ou_conflitantes']}**.
- Janelas geradas (utilizáveis): **{manifest['janelas_geradas']}**.
- Janelas bloqueadas: **{manifest['janelas_bloqueadas']}**.

| temporal_precision | alvos | confiança média | janelas utilizáveis | janelas bloqueadas |
| --- | --- | --- | --- | --- |
{linhas_prec}

## 14. Exemplos de resoluções

{ex("exact_day", "Dia exato")}
{ex("inferred_from_event_id", "Inferida do identificador")}
{ex("publication_date_only", "Apenas publicação")}
{ex("unknown", "Desconhecida")}
{ex("invalid_or_conflicting", "Inválida/conflitante")}

## 15. Confirmação explícita dos bloqueios

Este marco **não** usou internet, **não** executou SAR nem GEE, **não** baixou raster,
**não** criou footprint, **não** treinou modelo, **não** produziu ground truth
supervisionado, **não** criou `score_v7`, **não** alterou o `score_v6`
(`score_v6_changed={str(manifest['score_v6_changed']).lower()}`) e **não** promoveu
nenhum alvo a `positive_strong`
(`positive_strong_promovidos={manifest['positive_strong_promovidos']}`). Contagens de
controle: `eligible_for_training=true` → {manifest['training_true_count']};
`eligible_for_ground_truth=true` → {manifest['ground_truth_true_count']};
`score_v7_candidate=true` → {manifest['score_v7_candidate_true_count']}.

O REV-P não prevê enchentes operacionalmente: produz análise estrutural review-only com
evidência observacional auditável.

## 16. Próximo passo recomendado

**{manifest['proximo_marco']}**. Com **{manifest['alvos_liberados']}** alvos liberados
por terem data resolvida em nível operacional, o próximo gargalo passa a ser a geometria
oficial: a maioria dos alvos datados ainda precisa de ponto/polígono/bbox para virar
candidato forte no futuro.
"""


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_all() -> dict:
    require_inputs()
    ensure_dir(OUT)
    ensure_dir(SCHEMAS_DIR)

    bundle = build_rows()
    summary = summary_rows(bundle["dates"], bundle["windows"])

    write_csv(DATES_CSV, bundle["dates"])
    write_csv(WINDOWS_CSV, bundle["windows"])
    write_csv(DIAG_CSV, bundle["diagnostics"])
    write_csv(BLOCKERS_CSV, bundle["blockers"])
    write_csv(SUMMARY_CSV, summary)
    write_csv(RELEASED_CSV, bundle["released"])

    write_json(SCHEMA_DATE, schema_date_obj())
    write_json(SCHEMA_WINDOW, schema_window_obj())
    write_json(SCHEMA_DIAG, schema_diag_obj())
    write_json(SCHEMA_SUMMARY, schema_summary_obj())

    manifest = manifest_obj(bundle)
    write_markdown(REPORT, report_text(manifest, bundle, summary))
    manifest["artifacts"] = [
        {"path": rel(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in PUBLIC_ARTIFACTS + [SCHEMA_DATE, SCHEMA_WINDOW, SCHEMA_DIAG, SCHEMA_SUMMARY]
        if p.exists()
    ]
    write_json(MANIFEST_JSON, manifest)
    return manifest


# ---------------------------------------------------------------------------
# Validacao
# ---------------------------------------------------------------------------
_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


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

    dates = _read_csv(DATES_CSV)
    windows = {w["target_id"]: w for w in _read_csv(WINDOWS_CSV)}
    blockers = _read_csv(BLOCKERS_CSV)
    summary = _read_csv(SUMMARY_CSV)
    blockers_by_target = {}
    for b in blockers:
        blockers_by_target.setdefault(b["target_id"], []).append(b)

    for d in dates:
        tid = d["target_id"]
        for fld in ("eligible_for_training", "eligible_for_ground_truth", "trainable", "score_v7_candidate"):
            if d.get(fld) != "false":
                errors.append(f"data:{tid}:{fld}_nao_false")
        if d.get("remains_review_only") != "true":
            errors.append(f"data:{tid}:review_only_nao_true")
        blob = d.get("rationale_pt", "").lower()
        if "ground truth" in blob and "nao" not in blob and "não" not in blob:
            errors.append(f"data:{tid}:rationale_afirma_ground_truth")
        if "positive_strong" in blob:
            errors.append(f"data:{tid}:rationale_afirma_positive_strong")
        # confidence 0..100
        try:
            sc = int(d["temporal_confidence_score"])
        except ValueError:
            sc = -1
        if not (0 <= sc <= 100):
            errors.append(f"data:{tid}:confidence_fora_intervalo:{d['temporal_confidence_score']}")
        prec = d["temporal_precision"]
        if prec not in TEMPORAL_PRECISIONS:
            errors.append(f"data:{tid}:precisao_fora_enum:{prec}")
        # exact_day exige data ISO valida
        if prec == "exact_day" and not _ISO_RE.match(d["resolved_event_date"]):
            errors.append(f"data:{tid}:exact_day_sem_data_iso")
        # inferred exige date_is_inferred=true
        if prec == "inferred_from_event_id" and d["date_is_inferred"] != "true":
            errors.append(f"data:{tid}:inferred_sem_flag")
        # date_range exige inicio e fim validos
        if prec == "date_range":
            if not (_ISO_RE.match(d["date_range_start"]) and _ISO_RE.match(d["date_range_end"])):
                errors.append(f"data:{tid}:date_range_sem_limites")

        w = windows.get(tid, {})
        # publication_date_only nao libera SAR
        if prec == "publication_date_only" and w.get("usable_for_sar_candidate_future") == "true":
            errors.append(f"window:{tid}:publication_libera_sar")
        # month/year/unknown nao pode ter janela operacional sem blocking_reason
        if prec in ("month_only", "year_only", "unknown", "invalid_or_conflicting"):
            if w.get("window_status") in ("usable", "usable_with_range_uncertainty"):
                errors.append(f"window:{tid}:janela_operacional_indevida")
            if not w.get("window_blocking_reason", "").strip():
                errors.append(f"window:{tid}:sem_window_blocking_reason")
        # coerencia das janelas utilizaveis
        if w.get("window_status") in ("usable", "usable_with_range_uncertainty"):
            ps, pe = w["pre_event_window_start"], w["pre_event_window_end"]
            qs, qe = w["post_event_window_start"], w["post_event_window_end"]
            if ps and pe and ps > pe:
                errors.append(f"window:{tid}:pre_start_maior_que_end")
            if qs and qe and qs > qe:
                errors.append(f"window:{tid}:post_start_maior_que_end")
            if pe and qs and qs < pe:
                errors.append(f"window:{tid}:post_start_antes_do_pre_end")

    # resumo: contagens proibidas sempre 0
    for r in summary:
        for fld in ("eligible_for_training_count", "eligible_for_ground_truth_count", "score_v7_candidate_count"):
            if r.get(fld) != "0":
                errors.append(f"resumo:{r['temporal_precision']}:{fld}_nao_zero")

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
    if not title_is_portuguese(_report_h1(report_txt)):
        errors.append("titulo_principal_nao_portugues")

    # artefatos nao ignorados pelo git
    ignored = gt03.gt02.gt01.git_ignored(REQUIRED_OUTPUTS)
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
        "GT04 resolvedor temporal validado: "
        f"alvos={manifest['alvos_total']} precisao={manifest['distribuicao_precisao']} "
        f"janelas={manifest['janelas_geradas']} bloqueadas={manifest['janelas_bloqueadas']} "
        f"liberados={manifest['alvos_liberados']} proximo={manifest['proximo_marco']} "
        f"score_v6_changed={str(manifest['score_v6_changed']).lower()}"
    )
    return 0


def run_all() -> int:
    manifest = build_all()
    print(
        "GT04 resolvedor temporal gerado: "
        f"alvos={manifest['alvos_total']} precisao={manifest['distribuicao_precisao']} "
        f"janelas={manifest['janelas_geradas']} proximo={manifest['proximo_marco']}"
    )
    return 0
