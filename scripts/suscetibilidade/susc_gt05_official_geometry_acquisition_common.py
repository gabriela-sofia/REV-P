"""SUSC-GT05 - Pacote de Aquisicao de Geometria Oficial.

Le os resultados do SUSC-GT04 (datas e janelas) e recupera do SUSC-GT03/GT02 os
campos geometricos, para classificar, priorizar e planejar a AQUISICAO FUTURA de
geometria oficial (ou geometria observacional forte) dos alvos ja datados. E um
pacote offline e review-only: NAO busca internet, NAO baixa dado novo, NAO consulta
API, NAO geocodifica, NAO roda SAR nem GEE, NAO baixa raster, NAO cria footprint,
NAO cria geometria real, NAO altera o score_v6, NAO cria score_v7, NAO treina modelo
e NAO altera as decisoes do GT02 nem as datas do GT04. Nada e promovido a ground truth
nem a positive_strong.

Observacao metodologica: a qualificacao oficial da geometria (evidence_class, ex.
official_observed_event_polygon) nao foi propagada ate os outputs versionados do
GT02/GT03; ali a geometria aparece apenas com o tipo bruto (Polygon, bbox, point).
Fail-closed: tipos brutos nao sao tratados como geometria oficial forte; entram como
geometria_parcial_resolvivel, candidata a confirmacao oficial futura.

Ver [[susc_gt04_event_date_window_resolver]] e [[susc_gt03_strong_evidence_target_pack]].
Identificadores tecnicos em ingles sao mantidos por compatibilidade com os marcos
anteriores, schemas e validadores; o relatorio publico explica cada campo em portugues.
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
import susc_gt04_event_date_window_resolver_common as gt04  # noqa: E402

gt03 = gt04.gt03
gt02 = gt03.gt02

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS_DIR = ROOT / "schemas" / "suscetibilidade"

# Entradas GT04 (principais) + GT03 (geometria) -- lidas, nunca regeradas.
GT04_DATES = OUT / "susc_gt04_datas_resolvidas_por_alvo.csv"
GT04_WINDOWS = OUT / "susc_gt04_janelas_pre_pos_evento.csv"
GT04_DIAG = OUT / "susc_gt04_diagnostico_temporal.csv"
GT04_RELEASED = OUT / "susc_gt04_alvos_liberados_para_proxima_aquisicao.csv"
GT04_MANIFEST = OUT / "susc_gt04_manifesto.json"
GT03_TARGETS = OUT / "susc_gt03_pacote_alvos_evidencia_forte.csv"

REQUIRED_INPUTS = [GT04_DATES, GT04_WINDOWS, GT04_DIAG, GT04_RELEASED, GT04_MANIFEST, GT03_TARGETS]

# Saidas publicas (portugues)
REPORT = OUT / "SUSC_GT05_PACOTE_AQUISICAO_GEOMETRIA_OFICIAL.md"
TARGETS_CSV = OUT / "susc_gt05_pacote_aquisicao_geometria_oficial.csv"
DIAG_CSV = OUT / "susc_gt05_diagnostico_geometrico_por_alvo.csv"
TRACKS_CSV = OUT / "susc_gt05_trilhas_aquisicao_geometrica.csv"
PRIORITY_CSV = OUT / "susc_gt05_priorizacao_geometrica.csv"
PLAN_CSV = OUT / "susc_gt05_plano_aquisicao_geometria_por_alvo.csv"
BLOCKERS_CSV = OUT / "susc_gt05_bloqueios_geometricos.csv"
SUMMARY_CSV = OUT / "susc_gt05_resumo_geometrico.csv"
RELEASED_CSV = OUT / "susc_gt05_alvos_liberados_para_qa_ou_sar_futuro.csv"
MANIFEST_JSON = OUT / "susc_gt05_manifesto.json"

SCHEMA_TARGET = SCHEMAS_DIR / "susc_gt05_geometry_acquisition_target.schema.json"
SCHEMA_DIAG = SCHEMAS_DIR / "susc_gt05_geometry_diagnostic.schema.json"
SCHEMA_TRACK = SCHEMAS_DIR / "susc_gt05_geometry_track.schema.json"
SCHEMA_SUMMARY = SCHEMAS_DIR / "susc_gt05_geometry_summary.schema.json"

PUBLIC_ARTIFACTS = [
    REPORT, TARGETS_CSV, DIAG_CSV, TRACKS_CSV, PRIORITY_CSV, PLAN_CSV,
    BLOCKERS_CSV, SUMMARY_CSV, RELEASED_CSV,
]
REQUIRED_OUTPUTS = PUBLIC_ARTIFACTS + [
    MANIFEST_JSON, SCHEMA_TARGET, SCHEMA_DIAG, SCHEMA_TRACK, SCHEMA_SUMMARY,
]

MARCO = "SUSC-GT05"
TITULO_PUBLICO = "Pacote de Aquisição de Geometria Oficial"
TITULO_H1_TOKEN = "Aquisição de Geometria Oficial"

PUBLIC_FORBIDDEN_RE = gt04.PUBLIC_FORBIDDEN_RE
OPERATIONAL_FORECAST_RE = gt04.OPERATIONAL_FORECAST_RE

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
GEOMETRY_STATUSES = [
    "geometria_forte_existente",
    "geometria_parcial_resolvivel",
    "geometria_contextual_insuficiente",
    "geometria_ausente",
    "requer_footprint_tecnico_futuro",
    "geometria_rejeitada",
]
GEOMETRY_PRIORITY_CLASSES = [
    "prioridade_geometrica_alta",
    "prioridade_geometrica_media",
    "prioridade_geometrica_baixa",
    "bloqueado_por_geometria_insuficiente",
    "manter_apenas_contexto",
]
GEOMETRY_TRACKS = [
    "buscar_poligono_oficial",
    "buscar_ponto_oficial",
    "resolver_endereco_oficial",
    "consultar_base_setorial_oficial_futura",
    "preparar_footprint_sar_futuro",
    "estimar_incerteza_geometrica",
    "validar_patch_link",
    "separar_fenomeno",
    "manter_como_contexto",
]

# Geometria forte (oficial/tecnica). official_bbox permitido se preciso.
STRONG_GEOMETRY_TYPES = set(gt02.STRONG_GEOMETRY_TYPES) | {"official_bbox"}
# Tipos que NUNCA sao geometria forte (contexto/aproximacao).
CONTEXTUAL_FORBIDDEN_TYPES = {
    "city_level_record", "municipal_context", "street_neighborhood_context",
    "risk_area_not_event", "alert_only", "news_only", "documentary_context_only",
    "susceptibility_map_only", "administrative_record_without_geometry",
    "centroid_guess", "neighborhood_centroid", "municipality_centroid",
}
# Tipos brutos de geometria vetorial real (nao oficiais, mas resolviveis).
REAL_GEOMETRY_HINTS = {
    "polygon", "multipolygon", "point", "point_set", "bbox", "polygon_bbox",
    "linestring", "geometrycollection",
}
USABLE_TEMPORAL = {"exact_day", "inferred_from_event_id", "date_range"}
REGIOES_TRABALHADAS = {"recife", "curitiba", "petropolis"}
PETROPOLIS_BLOCKING = gt02.PETROPOLIS_BLOCKING_PHENOMENA
FLOOD_PHENOMENA = gt02.FLOOD_PHENOMENA
STRONG_PATCH_LINK = gt02.STRONG_PATCH_LINK_QUALITY
QA_ACCEPTED = gt02.QA_ACCEPTED

_clean = gt03._clean
_present = gt04._present
_num = gt02._num

# Requisitos geometricos (para positive_strong futuro).
GEOMETRY_REQUIREMENTS = ["geometry_type", "patch_id", "patch_link_quality", "uncertainty_m", "qa_status"]

TRACK_INFO = {
    "buscar_poligono_oficial": (
        "buscar poligono oficial do evento em base ja conhecida (sem consulta agora)",
        "evento datado com fonte forte e area candidata",
        "poligono oficial associado ao evento"),
    "buscar_ponto_oficial": (
        "buscar ponto/coordenada oficial da ocorrencia (chamado, atendimento)",
        "ocorrencia com fonte oficial e possivel coordenada",
        "ponto oficial georreferenciado"),
    "resolver_endereco_oficial": (
        "resolver endereco/rua/intersecao para geometria (sem geocodificar agora)",
        "registro com endereco rastreavel, sem lat/lon",
        "geometria derivada de endereco oficial"),
    "consultar_base_setorial_oficial_futura": (
        "consultar base setorial oficial no futuro (Defesa Civil/SGB/APAC/IPPUC/CEMADEN/S2iD)",
        "fonte setorial identificada",
        "geometria/atributo oficial da base setorial"),
    "preparar_footprint_sar_futuro": (
        "preparar alvo de footprint tecnico SAR para execucao futura (sem rodar SAR)",
        "data/janela operacional e area candidata",
        "especificacao de footprint SAR"),
    "estimar_incerteza_geometrica": (
        "estimar incerteza geometrica (uncertainty_m) do candidato",
        "geometria ou candidato disponivel",
        "uncertainty_m quantificada"),
    "validar_patch_link": (
        "validar/fortalecer o vinculo entre geometria e patch REV-P",
        "geometria e grade de patches",
        "patch_link_quality forte auditavel"),
    "separar_fenomeno": (
        "separar o fenomeno (inundacao vs deslizamento) antes de qualquer promocao",
        "descricao do evento",
        "phenomenon_type separado e compativel"),
    "manter_como_contexto": (
        "manter apenas como contexto documental, sem uso patch-level",
        "registro contextual",
        "documentacao contextual preservada"),
}


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
        raise FileNotFoundError("entradas GT04/GT03 ausentes: " + "; ".join(missing))


# ---------------------------------------------------------------------------
# Analise geometrica (pura)
# ---------------------------------------------------------------------------
def analyze_geometry(fields: dict) -> dict:
    gtype = _clean(fields.get("geometry_type")).lower()
    src = _clean(fields.get("source_authority")).lower()
    plq = _clean(fields.get("patch_link_quality")).lower()
    pheno = _clean(fields.get("phenomenon_type")).lower()
    region = _clean(fields.get("region")).lower()
    precision = _clean(fields.get("temporal_precision")).lower()

    contextual = (gtype in CONTEXTUAL_FORBIDDEN_TYPES) or (src in CONTEXTUAL_FORBIDDEN_TYPES) \
        or (src in gt02.WEAK_SOURCE_TYPES)
    geom_strong = gtype in STRONG_GEOMETRY_TYPES
    has_real_geometry = (gtype in REAL_GEOMETRY_HINTS) or geom_strong
    petro = "petropolis" in f"{region} {_clean(fields.get('city')).lower()}"
    petro_block = petro and (pheno in PETROPOLIS_BLOCKING or not _present(fields.get("phenomenon_type")))

    return {
        "has_geometry_id": _present(fields.get("geometry_id")),
        "has_footprint_id": _present(fields.get("footprint_id")),
        "has_patch_id": _present(fields.get("patch_id")),
        "has_geometry_type": _present(fields.get("geometry_type")),
        "geom_strong": geom_strong,
        "has_real_geometry": has_real_geometry,
        "contextual": contextual,
        "has_patch_link": _present(fields.get("patch_link_quality")),
        "patch_link_strong": plq in STRONG_PATCH_LINK,
        "has_uncertainty": _num(fields.get("uncertainty_m")) is not None,
        "qa_ok": _clean(fields.get("qa_status")).lower() in QA_ACCEPTED,
        "has_source_strong": _present(fields.get("source_authority")) and not contextual,
        "pheno_flood": pheno in FLOOD_PHENOMENA,
        "in_scope": region in REGIOES_TRABALHADAS,
        "has_date_operational": precision in USABLE_TEMPORAL,
        "precision_unknown": precision in ("unknown", "invalid_or_conflicting", ""),
        "petro": petro,
        "petro_block": petro_block,
    }


def classify_status(fl: dict) -> str:
    if fl["petro_block"]:
        return "geometria_rejeitada"
    if fl["contextual"] and not fl["has_real_geometry"]:
        return "geometria_contextual_insuficiente"
    if fl["geom_strong"] and (fl["has_patch_id"] or fl["has_patch_link"]):
        return "geometria_forte_existente"
    if fl["has_real_geometry"]:
        return "geometria_parcial_resolvivel"
    if fl["has_date_operational"]:
        return "requer_footprint_tecnico_futuro"
    if fl["has_footprint_id"] or fl["has_geometry_id"] or fl["has_patch_id"]:
        return "geometria_parcial_resolvivel"
    return "geometria_ausente"


def missing_geometry(fl: dict) -> list[str]:
    checks = {
        "geometry_type": fl["geom_strong"],
        "patch_id": fl["has_patch_id"],
        "patch_link_quality": fl["patch_link_strong"],
        "uncertainty_m": fl["has_uncertainty"],
        "qa_status": fl["qa_ok"],
    }
    return [k for k in GEOMETRY_REQUIREMENTS if not checks[k]]


def geometry_tracks(fl: dict, status: str) -> list[str]:
    tracks = []
    if fl["contextual"] and not fl["has_real_geometry"]:
        tracks.append("manter_como_contexto")
    if fl["has_real_geometry"] and not fl["geom_strong"]:
        tracks.append("buscar_poligono_oficial")
    if status in ("geometria_ausente", "requer_footprint_tecnico_futuro") and fl["has_source_strong"]:
        tracks.append("buscar_ponto_oficial")
    if not fl["has_real_geometry"] and fl["has_source_strong"]:
        tracks.append("resolver_endereco_oficial")
    if fl["has_source_strong"]:
        tracks.append("consultar_base_setorial_oficial_futura")
    if fl["has_date_operational"] and not fl["geom_strong"]:
        tracks.append("preparar_footprint_sar_futuro")
    if not fl["has_uncertainty"] and (fl["has_real_geometry"] or fl["has_footprint_id"]):
        tracks.append("estimar_incerteza_geometrica")
    if (fl["has_patch_id"] or fl["has_patch_link"]) and not fl["patch_link_strong"]:
        tracks.append("validar_patch_link")
    if fl["petro_block"] or (fl["petro"] and not fl["pheno_flood"]):
        tracks.append("separar_fenomeno")
    if not tracks:
        tracks.append("manter_como_contexto")
    return [t for t in GEOMETRY_TRACKS if t in dict.fromkeys(tracks)]


def geometry_priority_score(fl: dict, precision: str, priority_gt03: str, missing: list[str]) -> int:
    s = 0
    if fl["has_date_operational"]:
        s += 12
    if precision == "exact_day":
        s += 10
    if precision == "inferred_from_event_id":
        s += 7
    if fl["has_source_strong"]:
        s += 10
    if fl["has_geometry_id"] or fl["has_footprint_id"]:
        s += 8
    if fl["geom_strong"]:
        s += 12
    elif fl["has_real_geometry"]:
        s += 8
    if fl["has_patch_id"]:
        s += 8
    if fl["has_patch_link"]:
        s += 6
    if fl["pheno_flood"]:
        s += 6
    if fl["in_scope"]:
        s += 6
    if priority_gt03 == "prioridade_media":
        s += 6
    if len(missing) <= 2:
        s += 8
    # negativos
    if not fl["has_geometry_type"]:
        s -= 8
    if fl["contextual"]:
        s -= 40
    if fl["petro_block"]:
        s -= 20
    if not fl["has_patch_id"]:
        s -= 4
    if not fl["has_patch_link"]:
        s -= 4
    if fl["precision_unknown"]:
        s -= 10
    return max(0, min(100, s))


def geometry_priority_class(fl: dict, status: str, score: int) -> str:
    if fl["contextual"] and not fl["has_real_geometry"]:
        return "manter_apenas_contexto"
    if status == "geometria_ausente" and not fl["has_date_operational"] and not fl["has_source_strong"]:
        return "bloqueado_por_geometria_insuficiente"
    teto_alta = not fl["petro_block"]
    has_geom = fl["geom_strong"] or fl["has_real_geometry"]
    has_patch = fl["has_patch_id"] or fl["has_patch_link"]
    if teto_alta and fl["has_date_operational"] and fl["has_source_strong"] and has_geom and has_patch and score >= 60:
        return "prioridade_geometrica_alta"
    if fl["has_date_operational"] and (fl["has_source_strong"] or fl["has_real_geometry"]) and score >= 40:
        return "prioridade_geometrica_media"
    if score >= 20:
        return "prioridade_geometrica_baixa"
    return "bloqueado_por_geometria_insuficiente"


def can_become_strong(fl: dict, missing: list[str]) -> bool:
    if fl["contextual"] or fl["petro_block"]:
        return False
    if not (fl["has_date_operational"] or fl["has_source_strong"] or fl["has_real_geometry"]):
        return False
    return len(missing) > 0


# ---------------------------------------------------------------------------
# Construcao das linhas
# ---------------------------------------------------------------------------
def build_rows() -> dict:
    dates = {d["target_id"]: d for d in _read_csv(GT04_DATES)}
    windows = {w["target_id"]: w for w in _read_csv(GT04_WINDOWS)}
    targets = {t["target_id"]: t for t in _read_csv(GT03_TARGETS)}

    order = list(dates.keys())  # ordem estavel = ordem do GT04
    out_targets, out_diag, out_tracks, out_plan, out_blockers, out_released = [], [], [], [], [], []
    b_idx = 1

    for tid in order:
        d = dates[tid]
        w = windows.get(tid, {})
        t = targets.get(tid, {})
        precision = d.get("temporal_precision", "")
        fields = {
            "geometry_type": t.get("geometry_type", ""),
            "geometry_id": t.get("geometry_id", ""),
            "footprint_id": t.get("footprint_id", ""),
            "patch_id": t.get("patch_id", ""),
            "patch_link_quality": t.get("patch_link_quality", ""),
            "uncertainty_m": t.get("uncertainty_m", ""),
            "qa_status": t.get("qa_status", ""),
            "source_authority": t.get("source_authority", ""),
            "phenomenon_type": t.get("phenomenon_type", ""),
            "city": t.get("city", ""), "region": t.get("region", ""),
            "temporal_precision": precision,
        }
        fl = analyze_geometry(fields)
        status = classify_status(fl)
        missing = missing_geometry(fl)
        tracks = geometry_tracks(fl, status)
        prio_gt03 = t.get("priority_class", "")
        score = geometry_priority_score(fl, precision, prio_gt03, missing)
        cls = geometry_priority_class(fl, status, score)
        can_strong = can_become_strong(fl, missing)
        can_qa = status in ("geometria_forte_existente", "geometria_parcial_resolvivel") and not fl["petro_block"]
        can_sar = (status in ("requer_footprint_tecnico_futuro", "geometria_parcial_resolvivel")
                   and fl["has_date_operational"] and not fl["precision_unknown"] and not fl["petro_block"])
        gtid = f"GEO_{_seq(tid)}"

        out_targets.append({
            "geometry_target_id": gtid, "target_id": tid,
            "source_replay_id": d.get("source_replay_id", ""),
            "source_decision_id": d.get("source_decision_id", ""),
            "event_id": d.get("event_id", "not_available"),
            "footprint_id": _norm(fields["footprint_id"]),
            "geometry_id": _norm(fields["geometry_id"]),
            "patch_id": _norm(fields["patch_id"]),
            "city": _norm(fields["city"]), "region": _norm(fields["region"]),
            "phenomenon_type": _norm(fields["phenomenon_type"]),
            "resolved_event_date": d.get("resolved_event_date", "not_available"),
            "temporal_precision": precision,
            "window_status": w.get("window_status", "not_available"),
            "source_authority": _norm(fields["source_authority"]),
            "geometry_type": _norm(fields["geometry_type"]),
            "patch_link_quality": _norm(fields["patch_link_quality"]),
            "uncertainty_m": _norm(fields["uncertainty_m"]),
            "qa_status": _norm(fields["qa_status"]),
            "assigned_state_gt02": t.get("assigned_state_gt02", ""),
            "priority_class_gt03": prio_gt03,
            "geometry_acquisition_status": status,
            "geometry_priority_score": str(score),
            "geometry_priority_class": cls,
            "geometry_acquisition_tracks": ";".join(tracks) if tracks else "nenhuma",
            "missing_geometry_requirements": ";".join(missing) if missing else "none",
            "expected_geometry_artifact_pt": _expected_artifact(tracks),
            "can_advance_to_qa_future": "true" if can_qa else "false",
            "can_advance_to_sar_future": "true" if can_sar else "false",
            "can_become_positive_strong_after_geometry_and_qa": "true" if can_strong else "false",
            "remains_review_only": "true", "trainable": "false",
            "eligible_for_training": "false", "eligible_for_ground_truth": "false",
            "score_v7_candidate": "false",
            "rationale_pt": _rationale(fl, status, cls),
        })

        rej_reason = "fenomeno misto/deslizamento sem separacao da mancha de inundacao" if fl["petro_block"] else "none"
        out_diag.append({
            "geometry_diagnostic_id": f"GDG_{_seq(tid)}", "geometry_target_id": gtid, "target_id": tid,
            "has_geometry_id": _b(fl["has_geometry_id"]), "has_footprint_id": _b(fl["has_footprint_id"]),
            "has_patch_id": _b(fl["has_patch_id"]), "has_geometry_type": _b(fl["has_geometry_type"]),
            "has_strong_geometry_type": _b(fl["geom_strong"]), "has_patch_link_quality": _b(fl["has_patch_link"]),
            "has_uncertainty_m": _b(fl["has_uncertainty"]), "has_qa_status": _b(fl["qa_ok"]),
            "geometry_is_contextual_only": _b(fl["contextual"] and not fl["has_real_geometry"]),
            "geometry_is_official_candidate": _b(fl["has_real_geometry"] and not fl["geom_strong"]),
            "geometry_is_technical_candidate": _b(status == "requer_footprint_tecnico_futuro"),
            "geometry_rejection_reason": rej_reason,
            "next_required_geometry_action_pt": _next_action(status, tracks),
            "prevents_positive_strong": _b(bool(missing)),
            "prevents_evaluation": _b(status not in ("geometria_forte_existente",)),
            "prevents_calibration": _b(bool(missing)),
            "prevents_benchmark": "true",
        })

        for tr in tracks:
            obj, inp, outp = TRACK_INFO[tr]
            out_tracks.append({
                "geometry_track_id": f"GTK_{len(out_tracks)+1:05d}", "geometry_target_id": gtid,
                "acquisition_track": tr, "objective_pt": obj, "required_input_pt": inp,
                "expected_output_pt": outp,
                "success_criteria_pt": f"{outp} obtido e auditavel, mantendo review_only",
                "fail_closed_outcome_pt": "sem o artefato, o alvo permanece no estado atual; sem promocao",
                "no_network_in_this_milestone": "true", "no_geocoding_in_this_milestone": "true",
                "no_sar_execution_in_this_milestone": "true", "no_geometry_created_in_this_milestone": "true",
                "review_only": "true",
            })

        if cls in ("prioridade_geometrica_alta", "prioridade_geometrica_media", "prioridade_geometrica_baixa") and tracks:
            for order_i, tr in enumerate([x for x in tracks if x != "manter_como_contexto"] or tracks, start=1):
                obj, inp, outp = TRACK_INFO[tr]
                out_plan.append({
                    "plan_id": f"GPL_{len(out_plan)+1:05d}", "geometry_target_id": gtid,
                    "step_order": str(order_i), "action_pt": obj, "expected_artifact_pt": outp,
                    "success_criteria_pt": f"{outp} registrado e auditavel",
                    "fail_closed_outcome_pt": "sem o artefato, permanece review-only sem promocao",
                    "manual_review_required": "true",
                    "can_be_automated_future": _b(tr in ("buscar_poligono_oficial", "buscar_ponto_oficial", "consultar_base_setorial_oficial_futura", "preparar_footprint_sar_futuro")),
                    "requires_network_future": _b(tr in ("buscar_poligono_oficial", "buscar_ponto_oficial", "resolver_endereco_oficial", "consultar_base_setorial_oficial_futura")),
                    "requires_official_source_future": _b(tr in ("buscar_poligono_oficial", "buscar_ponto_oficial", "consultar_base_setorial_oficial_futura")),
                    "requires_sar_future": _b(tr == "preparar_footprint_sar_futuro"),
                })
        else:
            out_plan.append({
                "plan_id": f"GPL_{len(out_plan)+1:05d}", "geometry_target_id": gtid, "step_order": "1",
                "action_pt": "manter como contexto/bloqueado; sem aquisicao geometrica neste marco",
                "expected_artifact_pt": "nenhum artefato de aquisicao neste marco",
                "success_criteria_pt": "registro preservado como contexto auditavel",
                "fail_closed_outcome_pt": "permanece review-only, sem promocao e sem treino",
                "manual_review_required": "true", "can_be_automated_future": "false",
                "requires_network_future": "false", "requires_official_source_future": "false",
                "requires_sar_future": "false",
            })

        for bl in _blockers(fl, status, cls, gtid):
            bl["geometry_blocker_id"] = f"GBL_{b_idx:05d}"
            out_blockers.append(bl)
            b_idx += 1

        if can_qa or can_sar:
            rel_type = "geometria_util_liberada_para_qa" if can_qa else "datado_sem_geometria_liberado_para_sar_futuro"
            out_released.append({
                "released_geometry_target_id": f"GRL_{_seq(tid)}", "geometry_target_id": gtid, "target_id": tid,
                "release_type": rel_type,
                "next_recommended_milestone": "GT06 - Protocolo de QA Humano" if can_qa else "GT06 - Preparacao de Canarios SAR sem Execucao",
                "next_recommended_action_pt": "avancar para QA humano da geometria existente" if can_qa else "avancar para preparacao de footprint SAR com a janela temporal",
                "resolved_event_date": d.get("resolved_event_date", "not_available"),
                "geometry_acquisition_status": status, "geometry_priority_class": cls,
                "still_missing_requirements": ";".join(missing) if missing else "none",
                "remains_review_only": "true",
                "not_ground_truth_reason": "geometria review-only; nao e verdade de campo nem habilita treino",
            })

    return {
        "targets": out_targets, "diag": out_diag, "tracks": out_tracks, "plan": out_plan,
        "blockers": out_blockers, "released": out_released,
    }


def _b(v) -> str:
    return "true" if v else "false"


def _norm(v: str) -> str:
    v = _clean(v)
    return v if v and v.lower() not in gt02.NULLISH else "not_available"


def _seq(target_id: str) -> str:
    import re
    m = re.search(r"(\d+)$", target_id)
    return m.group(1) if m else target_id


def _expected_artifact(tracks: list[str]) -> str:
    primary = tracks[0] if tracks else "manter_como_contexto"
    return {
        "buscar_poligono_oficial": "poligono oficial do evento",
        "buscar_ponto_oficial": "ponto oficial georreferenciado",
        "resolver_endereco_oficial": "geometria derivada de endereco oficial",
        "consultar_base_setorial_oficial_futura": "atributo/geometria de base setorial oficial",
        "preparar_footprint_sar_futuro": "especificacao de footprint SAR (sem execucao)",
        "estimar_incerteza_geometrica": "uncertainty_m estimada",
        "validar_patch_link": "patch_link forte auditavel",
        "separar_fenomeno": "fenomeno separado (inundacao vs deslizamento)",
        "manter_como_contexto": "documentacao contextual mantida",
    }.get(primary, "revisao_manual")


def _next_action(status: str, tracks: list[str]) -> str:
    return {
        "geometria_forte_existente": "validar patch_link e submeter a QA humano",
        "geometria_parcial_resolvivel": "buscar confirmacao de geometria oficial e estimar incerteza",
        "geometria_contextual_insuficiente": "manter como contexto; sem uso patch-level",
        "geometria_ausente": "buscar ponto/poligono oficial ou preparar footprint SAR futuro",
        "requer_footprint_tecnico_futuro": "preparar footprint tecnico SAR com a janela temporal",
        "geometria_rejeitada": "separar o fenomeno antes de qualquer aquisicao",
    }.get(status, "revisar caso a caso")


def _rationale(fl: dict, status: str, cls: str) -> str:
    if fl["petro_block"]:
        return "Petropolis com fenomeno misto/deslizamento sem separacao: geometria rejeitada ate separar o fenomeno"
    if fl["contextual"] and not fl["has_real_geometry"]:
        return "apenas contexto (cidade/bairro/municipio/area de risco); nao serve para patch-level"
    if status == "geometria_forte_existente":
        return "geometria forte oficial ja existente; ainda pode faltar QA/incerteza; nao promove estado"
    if status == "geometria_parcial_resolvivel":
        return "geometria vetorial existe mas nao confirmada como oficial; resolvivel a geometria oficial futura"
    if status == "requer_footprint_tecnico_futuro":
        return "datado sem geometria oficial; candidato a footprint tecnico SAR em marco futuro (sem executar SAR)"
    return "sinal geometrico ausente ou insuficiente; permanece review-only"


def _blockers(fl: dict, status: str, cls: str, gtid: str) -> list[dict]:
    out = []

    def add(btype, sev, field, reason, ph=False, ps=True, ev=False, cal=False):
        out.append({
            "geometry_target_id": gtid, "blocker_type": btype, "severity": sev, "field": field,
            "reason_pt": reason, "prevents_geometry_priority_high": _b(ph),
            "prevents_positive_strong": _b(ps), "prevents_evaluation": _b(ev),
            "prevents_calibration": _b(cal),
            "how_to_resolve_pt": _how_resolve(btype),
        })

    if fl["contextual"] and not fl["has_real_geometry"]:
        add("geometria_contextual_proibida", "alta", "geometry_type",
            "tipo contextual/aproximado nunca e geometria forte; manter como contexto",
            ph=True, ps=True, ev=True, cal=True)
    if fl["petro_block"]:
        add("phenomenon_mismatch_or_mixed_event", "alta", "phenomenon_type",
            "Petropolis com fenomeno misto/deslizamento sem separacao", ph=True, ps=True, ev=True, cal=True)
    if status == "geometria_ausente":
        add("geometria_ausente", "media", "geometry_type",
            "sem sinal geometrico util para aquisicao objetiva", ph=True, ps=True, cal=True)
    if status == "requer_footprint_tecnico_futuro":
        add("geometria_oficial_ausente", "media", "geometry_type",
            "datado sem geometria oficial; depende de footprint SAR futuro", ph=False, ps=True, cal=True)
    return out


def _how_resolve(btype: str) -> str:
    return {
        "geometria_contextual_proibida": "buscar geometria oficial (ponto/poligono); contexto sozinho nao serve",
        "phenomenon_mismatch_or_mixed_event": "separar inundacao de deslizamento antes de adquirir geometria",
        "geometria_ausente": "buscar ponto/poligono oficial em base setorial ou preparar footprint SAR futuro",
        "geometria_oficial_ausente": "preparar footprint tecnico SAR (marco futuro) ou buscar geometria oficial",
    }.get(btype, "revisar a fonte geometrica")


# ---------------------------------------------------------------------------
# Priorizacao agregada e resumo
# ---------------------------------------------------------------------------
def priorizacao_rows(targets: list[dict]) -> list[dict]:
    agg = {}
    for t in targets:
        key = (t["source_authority"], t["region"], t["geometry_type"], t["geometry_acquisition_status"])
        a = agg.setdefault(key, {"count": 0, "alta": 0, "media": 0, "baixa": 0, "blocked": 0, "context": 0, "missing": {}})
        a["count"] += 1
        cls = t["geometry_priority_class"]
        a["alta"] += cls == "prioridade_geometrica_alta"
        a["media"] += cls == "prioridade_geometrica_media"
        a["baixa"] += cls == "prioridade_geometrica_baixa"
        a["blocked"] += cls == "bloqueado_por_geometria_insuficiente"
        a["context"] += cls == "manter_apenas_contexto"
        for req in t["missing_geometry_requirements"].split(";"):
            if req and req != "none":
                a["missing"][req] = a["missing"].get(req, 0) + 1
    rows = []
    for i, (key, a) in enumerate(sorted(agg.items()), start=1):
        src, region, gtype, status = key
        main_gap = max(a["missing"], key=a["missing"].get) if a["missing"] else "none"
        rows.append({
            "grouping_id": f"GRP_{i:04d}", "source_authority": src, "city": "", "region": region,
            "geometry_type": gtype, "geometry_acquisition_status": status,
            "target_count": str(a["count"]), "high_priority_count": str(a["alta"]),
            "medium_priority_count": str(a["media"]), "low_priority_count": str(a["baixa"]),
            "blocked_count": str(a["blocked"]), "context_only_count": str(a["context"]),
            "main_geometry_gap_pt": _gap_pt(main_gap),
            "recommended_next_action_pt": _gap_action(main_gap, status),
        })
    return rows


def _gap_pt(req: str) -> str:
    return {
        "geometry_type": "falta geometria oficial forte",
        "patch_id": "falta patch correspondente",
        "patch_link_quality": "falta vinculo forte com patch",
        "uncertainty_m": "falta incerteza espacial",
        "qa_status": "falta QA humano aceito",
        "none": "sem lacuna geometrica dominante",
    }.get(req, "revisar lacunas caso a caso")


def _gap_action(req: str, status: str) -> str:
    if status == "requer_footprint_tecnico_futuro":
        return "preparar footprint SAR futuro para obter geometria"
    return {
        "geometry_type": "buscar geometria oficial (ponto/poligono)",
        "patch_link_quality": "validar/fortalecer patch_link",
        "uncertainty_m": "estimar incerteza geometrica",
        "qa_status": "encaminhar para QA humano",
        "none": "revisar caso a caso",
    }.get(req, "revisar lacunas caso a caso")


def summary_rows(targets: list[dict]) -> list[dict]:
    order = {s: i for i, s in enumerate(GEOMETRY_STATUSES)}
    agg = {}
    for t in targets:
        st = t["geometry_acquisition_status"]
        a = agg.setdefault(st, {"count": 0, "score": 0, "qa": 0, "sar": 0, "strong": 0})
        a["count"] += 1
        a["score"] += int(t["geometry_priority_score"])
        a["qa"] += t["can_advance_to_qa_future"] == "true"
        a["sar"] += t["can_advance_to_sar_future"] == "true"
        a["strong"] += t["can_become_positive_strong_after_geometry_and_qa"] == "true"
    rows = []
    for st in sorted(agg, key=lambda s: order.get(s, 99)):
        a = agg[st]
        rows.append({
            "geometry_acquisition_status": st, "count": str(a["count"]),
            "average_geometry_priority_score": str(round(a["score"] / a["count"], 2)) if a["count"] else "0",
            "can_advance_to_qa_future_count": str(a["qa"]),
            "can_advance_to_sar_future_count": str(a["sar"]),
            "can_become_positive_strong_after_geometry_and_qa_count": str(a["strong"]),
            "training_allowed_count": "0", "ground_truth_allowed_count": "0",
            "score_v7_candidate_count": "0",
        })
    return rows


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
_BOOL = {"enum": ["true", "false"]}
_FALSE = {"const": "false"}
_SCORE = {"type": "string", "pattern": r"^(100|[1-9]?[0-9])$"}


def schema_target_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "susc_gt05_geometry_acquisition_target.schema.json",
        "title": "SUSC-GT05 Alvo de Aquisicao Geometrica",
        "type": "object",
        "required": [
            "geometry_target_id", "target_id", "geometry_acquisition_status",
            "geometry_priority_score", "geometry_priority_class",
            "can_become_positive_strong_after_geometry_and_qa", "remains_review_only",
            "trainable", "eligible_for_training", "eligible_for_ground_truth",
            "score_v7_candidate", "rationale_pt",
        ],
        "properties": {
            "geometry_target_id": {"type": "string", "pattern": r"^GEO_\d+$"},
            "target_id": {"type": "string", "pattern": r"^TGT_\d+$"},
            "geometry_acquisition_status": {"enum": GEOMETRY_STATUSES},
            "geometry_priority_score": _SCORE,
            "geometry_priority_class": {"enum": GEOMETRY_PRIORITY_CLASSES},
            "geometry_acquisition_tracks": {"type": "string"},
            "missing_geometry_requirements": {"type": "string"},
            "can_advance_to_qa_future": _BOOL,
            "can_advance_to_sar_future": _BOOL,
            "can_become_positive_strong_after_geometry_and_qa": _BOOL,
            "remains_review_only": {"const": "true"},
            "trainable": _FALSE, "eligible_for_training": _FALSE,
            "eligible_for_ground_truth": _FALSE, "score_v7_candidate": _FALSE,
            "rationale_pt": {"type": "string", "minLength": 1},
        },
    }


def schema_diag_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "susc_gt05_geometry_diagnostic.schema.json",
        "title": "SUSC-GT05 Diagnostico Geometrico",
        "type": "object",
        "required": ["geometry_diagnostic_id", "geometry_target_id", "target_id",
                     "has_strong_geometry_type", "next_required_geometry_action_pt"],
        "properties": {
            "geometry_diagnostic_id": {"type": "string", "pattern": r"^GDG_\d+$"},
            "geometry_target_id": {"type": "string", "pattern": r"^GEO_\d+$"},
            "target_id": {"type": "string", "pattern": r"^TGT_\d+$"},
            "has_geometry_id": _BOOL, "has_footprint_id": _BOOL, "has_patch_id": _BOOL,
            "has_geometry_type": _BOOL, "has_strong_geometry_type": _BOOL,
            "has_patch_link_quality": _BOOL, "has_uncertainty_m": _BOOL, "has_qa_status": _BOOL,
            "geometry_is_contextual_only": _BOOL, "geometry_is_official_candidate": _BOOL,
            "geometry_is_technical_candidate": _BOOL,
            "next_required_geometry_action_pt": {"type": "string", "minLength": 1},
            "prevents_positive_strong": _BOOL,
        },
    }


def schema_track_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "susc_gt05_geometry_track.schema.json",
        "title": "SUSC-GT05 Trilha de Aquisicao Geometrica",
        "type": "object",
        "required": ["geometry_track_id", "geometry_target_id", "acquisition_track",
                     "objective_pt", "no_network_in_this_milestone",
                     "no_geocoding_in_this_milestone", "no_sar_execution_in_this_milestone",
                     "no_geometry_created_in_this_milestone", "review_only"],
        "properties": {
            "geometry_track_id": {"type": "string", "pattern": r"^GTK_\d+$"},
            "geometry_target_id": {"type": "string", "pattern": r"^GEO_\d+$"},
            "acquisition_track": {"enum": GEOMETRY_TRACKS},
            "objective_pt": {"type": "string", "minLength": 1},
            "no_network_in_this_milestone": {"const": "true"},
            "no_geocoding_in_this_milestone": {"const": "true"},
            "no_sar_execution_in_this_milestone": {"const": "true"},
            "no_geometry_created_in_this_milestone": {"const": "true"},
            "review_only": {"const": "true"},
        },
    }


def schema_summary_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "susc_gt05_geometry_summary.schema.json",
        "title": "SUSC-GT05 Resumo Geometrico",
        "type": "object",
        "required": ["geometry_acquisition_status", "count", "average_geometry_priority_score",
                     "training_allowed_count", "ground_truth_allowed_count", "score_v7_candidate_count"],
        "properties": {
            "geometry_acquisition_status": {"enum": GEOMETRY_STATUSES},
            "count": {"type": "string", "pattern": r"^\d+$"},
            "average_geometry_priority_score": {"type": "string"},
            "can_advance_to_qa_future_count": {"type": "string", "pattern": r"^\d+$"},
            "can_advance_to_sar_future_count": {"type": "string", "pattern": r"^\d+$"},
            "can_become_positive_strong_after_geometry_and_qa_count": {"type": "string", "pattern": r"^\d+$"},
            "training_allowed_count": {"const": "0"},
            "ground_truth_allowed_count": {"const": "0"},
            "score_v7_candidate_count": {"const": "0"},
        },
    }


# ---------------------------------------------------------------------------
# Manifesto / proximo marco
# ---------------------------------------------------------------------------
def _next_milestone(dist: dict, total: int) -> str:
    forte = dist.get("geometria_forte_existente", 0)
    parcial = dist.get("geometria_parcial_resolvivel", 0)
    sar = dist.get("requer_footprint_tecnico_futuro", 0)
    ausente_ctx = dist.get("geometria_ausente", 0) + dist.get("geometria_contextual_insuficiente", 0)
    if total and (forte + parcial) >= 0.5 * total and (forte + parcial) >= sar:
        return "GT06 - Protocolo de QA Humano"
    if sar >= parcial and sar > 0:
        return "GT06 - Preparacao de Canarios SAR sem Execucao"
    if ausente_ctx > (forte + parcial + sar):
        return "GT06 - Aquisicao Documental Manual Orientada"
    return "GT06 - Protocolo de QA Humano"


def manifest_obj(bundle: dict) -> dict:
    targets = bundle["targets"]
    dist = {}
    for t in targets:
        dist[t["geometry_acquisition_status"]] = dist.get(t["geometry_acquisition_status"], 0) + 1
    artifacts = [
        {"path": rel(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in PUBLIC_ARTIFACTS + [SCHEMA_TARGET, SCHEMA_DIAG, SCHEMA_TRACK, SCHEMA_SUMMARY]
        if p.exists()
    ]
    return {
        "marco": MARCO, "titulo_publico": TITULO_PUBLICO,
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "gt04_source_commit": _run_git(["log", "-1", "--format=%h", "--", rel(GT04_DATES)]) or "unknown",
        "alvos_total": len(targets),
        "distribuicao_geometrica": dist,
        "geometria_forte_existente": dist.get("geometria_forte_existente", 0),
        "geometria_parcial_resolvivel": dist.get("geometria_parcial_resolvivel", 0),
        "requer_footprint_tecnico_futuro": dist.get("requer_footprint_tecnico_futuro", 0),
        "geometria_contextual_insuficiente": dist.get("geometria_contextual_insuficiente", 0),
        "geometria_ausente": dist.get("geometria_ausente", 0),
        "geometria_rejeitada": dist.get("geometria_rejeitada", 0),
        "alvos_liberados": len(bundle["released"]),
        "positive_strong_promovidos": 0,
        "geometria_criada": 0,
        "global_guardrails": GLOBAL_GUARDRAILS,
        "training_true_count": sum(1 for t in targets if t["eligible_for_training"] == "true"),
        "ground_truth_true_count": sum(1 for t in targets if t["eligible_for_ground_truth"] == "true"),
        "score_v7_candidate_true_count": sum(1 for t in targets if t["score_v7_candidate"] == "true"),
        "score_v6_changed": gt02.gt01._score_v6_changed(),
        "score_v7_created": gt02.gt01._score_v7_exists(),
        "proximo_marco": _next_milestone(dist, len(targets)),
        "artifacts": artifacts,
    }


# ---------------------------------------------------------------------------
# Relatorio publico (portugues)
# ---------------------------------------------------------------------------
def _example(targets, status):
    return next((t for t in targets if t["geometry_acquisition_status"] == status), None)


def report_text(manifest: dict, bundle: dict, summary: list[dict]) -> str:
    targets = bundle["targets"]
    linhas_sum = "\n".join(
        f"| {r['geometry_acquisition_status']} | {r['count']} | {r['average_geometry_priority_score']} | "
        f"{r['can_advance_to_qa_future_count']} | {r['can_advance_to_sar_future_count']} |"
        for r in summary
    )
    n_petro = sum(1 for t in targets if t["region"] == "petropolis")

    def ex(status, titulo):
        t = _example(targets, status)
        if not t:
            return f"- **{titulo}** (`{status}`): nenhum alvo neste estado."
        return (f"- **{titulo}** (`{status}`): alvo `{t['geometry_target_id']}`, event_id `{t['event_id']}`, "
                f"geometry_type `{t['geometry_type']}`, classe `{t['geometry_priority_class']}`, "
                f"trilhas `{t['geometry_acquisition_tracks']}`.")

    return f"""# {MARCO} - {TITULO_PUBLICO}

## 1. Escopo do marco

Este marco classifica, prioriza e planeja a **aquisição futura de geometria oficial**
(ou geometria observacional forte) para os alvos já datados pelo GT04. É um pacote
**offline e review-only** (uso restrito a revisão): não busca internet, não baixa dado
novo, não consulta API, não geocodifica, não roda SAR nem GEE, não baixa raster, não
cria footprint, **não cria geometria real**, não altera o `score_v6`, não cria
`score_v7`, não treina modelo e não promove nada a ground truth nem a `positive_strong`.

## 2. Relação com GT01, GT02, GT03 e GT04

O GT01 definiu a política; o GT02 aplicou-a; o GT03 montou a fila de alvos; o GT04
resolveu as datas e janelas. O GT05 usa as datas do GT04 e recupera do GT03/GT02 os
campos geométricos (tipo de geometria, patch, vínculo, fonte) para atacar o próximo
gargalo: a **geometria oficial**.

## 3. Por que geometria vem depois da data e antes do QA forte

Sem data não há janela; com data, o passo seguinte é ter uma geometria confiável (ponto,
polígono ou footprint) para vincular ao patch. O QA humano forte só faz sentido sobre
uma geometria já candidata; por isso a geometria vem antes do QA definitivo.

## 4. Por que o GT05 não busca internet, não geocodifica e não cria footprint

Este é um marco de **planejamento determinístico**: monta a fila e o plano a partir do
que já existe no repositório. A aquisição real (polígono oficial, ponto oficial,
geocodificação, footprint SAR) fica para marcos seguintes, com execução controlada.

## 5. Entradas usadas

Datas e janelas do GT04 e campos geométricos do GT03 (recuperados por `target_id`), lidos
dos outputs versionados sem regeração. Total de alvos: **{manifest['alvos_total']}**.

## 6. Classes de estado geométrico

`geometria_forte_existente` (geometria oficial forte + vínculo mínimo),
`geometria_parcial_resolvivel` (ponto/endereço/vetor bruto convertível a oficial),
`geometria_contextual_insuficiente` (só cidade/bairro/área de risco),
`geometria_ausente` (sem sinal útil), `requer_footprint_tecnico_futuro` (datado sem
geometria, candidato a SAR) e `geometria_rejeitada` (incompatível ou fenômeno misto).

## 7. Tipos de geometria forte e tipos proibidos

Fortes: `official_observed_event_polygon/point`, `technical_remote_sensing_flood_footprint`,
`official_polygon/point`, `technical_remote_sensing_polygon` e `official_bbox` com
precisão. **Nunca** fortes: `city_level_record`, `municipal_context`,
`street_neighborhood_context`, `risk_area_not_event`, `alert_only`, `news_only`,
`documentary_context_only`, `susceptibility_map_only`,
`administrative_record_without_geometry` e centroides (`centroid_guess`,
`neighborhood_centroid`, `municipality_centroid`). Observação: nos outputs versionados do
GT02/GT03 a geometria aparece com o tipo bruto (Polygon, bbox, point), sem a qualificação
oficial; por isso, fail-closed, ela entra como parcial resolvível, não como forte.

## 8. Critérios de priorização geométrica

O `geometry_priority_score` (0 a 100) pontua positivamente data operacional, precisão
`exact_day`/`inferred_from_event_id`, fonte forte, geometria/footprint/patch existentes,
fenômeno de inundação, cidade/região no escopo e poucos requisitos faltantes; pontua
negativamente geometria ausente/contextual, tipos proibidos, Petrópolis misto, ausência
de patch e precisão temporal `unknown`. A priorização **não** promove estado.

## 9-13. Distribuição dos estados geométricos

- `geometria_forte_existente`: **{manifest['geometria_forte_existente']}**.
- `geometria_parcial_resolvivel`: **{manifest['geometria_parcial_resolvivel']}**.
- `requer_footprint_tecnico_futuro`: **{manifest['requer_footprint_tecnico_futuro']}**.
- `geometria_contextual_insuficiente`: **{manifest['geometria_contextual_insuficiente']}**.
- `geometria_ausente`: **{manifest['geometria_ausente']}**.
- `geometria_rejeitada`: **{manifest['geometria_rejeitada']}**.

| geometry_acquisition_status | alvos | score médio | podem ir a QA | podem ir a SAR |
| --- | --- | --- | --- | --- |
{linhas_sum}

## 14. Principais bloqueios geométricos

Registrados em `susc_gt05_bloqueios_geometricos.csv`: geometria oficial ausente nos
alvos datados (dependem de footprint SAR futuro), tipos contextuais proibidos (só
contexto) e Petrópolis com fenômeno misto (bloqueio `phenomenon_mismatch_or_mixed_event`).

## 15. Petrópolis e eventos mistos

Alvos de Petrópolis ({n_petro} no total) com deslizamento ou fenômeno misto são
classificados como `geometria_rejeitada`, exigem a trilha `separar_fenomeno` e **nunca**
recebem prioridade geométrica alta sem essa separação.

## 16. Exemplos de alvos

{ex("geometria_forte_existente", "Geometria forte existente")}
{ex("geometria_parcial_resolvivel", "Geometria parcial resolvível")}
{ex("geometria_contextual_insuficiente", "Geometria contextual insuficiente")}
{ex("geometria_ausente", "Geometria ausente")}
{ex("requer_footprint_tecnico_futuro", "Requer footprint técnico futuro")}
{ex("geometria_rejeitada", "Geometria rejeitada")}

## 17. Confirmação explícita dos bloqueios

Este marco **não** usou internet, **não** geocodificou, **não** executou SAR nem GEE,
**não** baixou raster, **não** criou footprint, **não** criou geometria real
(`geometria_criada={manifest['geometria_criada']}`), **não** treinou modelo, **não**
produziu ground truth, **não** criou `score_v7`, **não** alterou o `score_v6`
(`score_v6_changed={str(manifest['score_v6_changed']).lower()}`) e **não** promoveu
nenhum alvo a `positive_strong`
(`positive_strong_promovidos={manifest['positive_strong_promovidos']}`). Contagens de
controle: `eligible_for_training=true` → {manifest['training_true_count']};
`eligible_for_ground_truth=true` → {manifest['ground_truth_true_count']};
`score_v7_candidate=true` → {manifest['score_v7_candidate_true_count']}.

O REV-P não prevê enchentes operacionalmente: produz análise estrutural review-only com
evidência observacional auditável.

## 18. Próximo passo recomendado

**{manifest['proximo_marco']}**. Com **{manifest['alvos_liberados']}** alvos liberados
para QA ou footprint SAR futuro, o próximo gargalo é obter/validar geometria: os alvos
datados sem geometria oficial devem seguir para a preparação de canários SAR (sem
execução), enquanto os que já têm geometria parcial devem seguir para QA humano.
"""


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_all() -> dict:
    require_inputs()
    ensure_dir(OUT)
    ensure_dir(SCHEMAS_DIR)

    bundle = build_rows()
    summary = summary_rows(bundle["targets"])
    priorizacao = priorizacao_rows(bundle["targets"])

    write_csv(TARGETS_CSV, bundle["targets"])
    write_csv(DIAG_CSV, bundle["diag"])
    write_csv(TRACKS_CSV, bundle["tracks"])
    write_csv(PRIORITY_CSV, priorizacao)
    write_csv(PLAN_CSV, bundle["plan"])
    write_csv(BLOCKERS_CSV, bundle["blockers"])
    write_csv(SUMMARY_CSV, summary)
    write_csv(RELEASED_CSV, bundle["released"])

    write_json(SCHEMA_TARGET, schema_target_obj())
    write_json(SCHEMA_DIAG, schema_diag_obj())
    write_json(SCHEMA_TRACK, schema_track_obj())
    write_json(SCHEMA_SUMMARY, schema_summary_obj())

    manifest = manifest_obj(bundle)
    write_markdown(REPORT, report_text(manifest, bundle, summary))
    manifest["artifacts"] = [
        {"path": rel(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in PUBLIC_ARTIFACTS + [SCHEMA_TARGET, SCHEMA_DIAG, SCHEMA_TRACK, SCHEMA_SUMMARY]
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
    if manifest["geometria_criada"] != 0:
        errors.append("geometria_criada_forbidden")

    targets = _read_csv(TARGETS_CSV)
    blockers = _read_csv(BLOCKERS_CSV)
    tracks = _read_csv(TRACKS_CSV)
    summary = _read_csv(SUMMARY_CSV)
    blockers_by = {}
    for b in blockers:
        blockers_by.setdefault(b["geometry_target_id"], []).append(b)
    tracks_by = {}
    for tr in tracks:
        tracks_by.setdefault(tr["geometry_target_id"], set()).add(tr["acquisition_track"])

    for t in targets:
        gid = t["geometry_target_id"]
        for fld in ("eligible_for_training", "eligible_for_ground_truth", "trainable", "score_v7_candidate"):
            if t.get(fld) != "false":
                errors.append(f"target:{gid}:{fld}_nao_false")
        if t.get("remains_review_only") != "true":
            errors.append(f"target:{gid}:review_only_nao_true")
        blob = t.get("rationale_pt", "").lower()
        if "ground truth" in blob and "nao" not in blob and "não" not in blob:
            errors.append(f"target:{gid}:rationale_ground_truth")
        if "positive_strong" in blob:
            errors.append(f"target:{gid}:rationale_positive_strong")
        # score 0..100
        try:
            sc = int(t["geometry_priority_score"])
        except ValueError:
            sc = -1
        if not (0 <= sc <= 100):
            errors.append(f"target:{gid}:score_fora_intervalo:{t['geometry_priority_score']}")
        if t.get("geometry_priority_class") not in GEOMETRY_PRIORITY_CLASSES:
            errors.append(f"target:{gid}:classe_invalida:{t.get('geometry_priority_class')}")
        if t.get("geometry_acquisition_status") not in GEOMETRY_STATUSES:
            errors.append(f"target:{gid}:status_invalido:{t.get('geometry_acquisition_status')}")
        # tipo contextual proibido nunca geometria_forte_existente
        gtype = t.get("geometry_type", "").lower()
        src = t.get("source_authority", "").lower()
        is_forbidden = gtype in CONTEXTUAL_FORBIDDEN_TYPES or src in CONTEXTUAL_FORBIDDEN_TYPES
        if is_forbidden and t.get("geometry_acquisition_status") == "geometria_forte_existente":
            errors.append(f"target:{gid}:contextual_virou_forte")
        # tipos fracos nunca prioridade alta sem blocker
        if is_forbidden and t.get("geometry_priority_class") == "prioridade_geometrica_alta" and not blockers_by.get(gid):
            errors.append(f"target:{gid}:fraco_alta_sem_blocker")
        # Petropolis mixed nunca alta sem separar_fenomeno
        pheno = t.get("phenomenon_type", "").lower()
        if t.get("region") == "petropolis" and pheno in PETROPOLIS_BLOCKING:
            if t.get("geometry_priority_class") == "prioridade_geometrica_alta":
                errors.append(f"target:{gid}:petropolis_alta")
            if "separar_fenomeno" not in tracks_by.get(gid, set()):
                errors.append(f"target:{gid}:petropolis_sem_separar")
        # temporal unknown nao liberado para SAR futuro
        if t.get("temporal_precision") in ("unknown", "invalid_or_conflicting", "") and t.get("can_advance_to_sar_future") == "true":
            errors.append(f"target:{gid}:unknown_liberado_sar")
        # can_become exige requisitos faltantes
        if t.get("can_become_positive_strong_after_geometry_and_qa") == "true":
            if t.get("missing_geometry_requirements") in ("", "none"):
                errors.append(f"target:{gid}:can_become_sem_missing")

    # resumo: contagens proibidas 0
    for r in summary:
        for fld in ("training_allowed_count", "ground_truth_allowed_count", "score_v7_candidate_count"):
            if r.get(fld) != "0":
                errors.append(f"resumo:{r['geometry_acquisition_status']}:{fld}_nao_zero")
        if r.get("geometry_acquisition_status") not in GEOMETRY_STATUSES:
            errors.append(f"resumo:status_invalido:{r.get('geometry_acquisition_status')}")

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
        "GT05 aquisicao geometrica validada: "
        f"alvos={manifest['alvos_total']} distribuicao={manifest['distribuicao_geometrica']} "
        f"liberados={manifest['alvos_liberados']} proximo={manifest['proximo_marco']} "
        f"score_v6_changed={str(manifest['score_v6_changed']).lower()}"
    )
    return 0


def run_all() -> int:
    manifest = build_all()
    print(
        "GT05 aquisicao geometrica gerada: "
        f"alvos={manifest['alvos_total']} distribuicao={manifest['distribuicao_geometrica']} "
        f"proximo={manifest['proximo_marco']}"
    )
    return 0
