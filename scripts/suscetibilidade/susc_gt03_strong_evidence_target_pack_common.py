"""SUSC-GT03 - Pacote de Alvos para Aquisicao de Evidencia Forte.

Le os resultados do SUSC-GT02 (replay review-only) e transforma os registros
provisorios/no_data em uma FILA AUDITAVEL de alvos de aquisicao futura de evidencia
forte. Nao promove nenhum estado: continua tudo review-only, sem treino, sem ground
truth, sem score_v7. Este marco NAO busca internet, NAO baixa dado novo, NAO roda SAR,
NAO roda GEE, NAO cria footprint, NAO altera o score_v6 e NAO altera as decisoes do
GT02 (apenas as le).

Diagnostica, por alvo, quais requisitos faltam para virar `positive_strong` no futuro,
qual trilha de aquisicao e necessaria e uma prioridade deterministica de 0 a 100.

Ver [[susc_gt01_ground_reference_policy]] e [[susc_gt02_ground_reference_replay]].
Identificadores tecnicos em ingles sao mantidos por compatibilidade com o GT01/GT02,
schemas e validadores; o relatorio publico explica cada campo em portugues.
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
# Importado APENAS para reutilizar constantes/regex (o import nao grava arquivos).
import susc_gt02_ground_reference_replay_common as gt02  # noqa: E402

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS_DIR = ROOT / "schemas" / "suscetibilidade"

# Entradas: outputs versionados do GT02 (nunca regeramos o GT02 aqui).
GT02_REPLAY = OUT / "susc_gt02_registro_replay_evidencias.csv"
GT02_DECISIONS = OUT / "susc_gt02_decisoes_referencia_por_patch.csv"
GT02_BLOCKERS = OUT / "susc_gt02_bloqueios.csv"
GT02_SUMMARY = OUT / "susc_gt02_resumo_estados.csv"
GT02_MANIFEST = OUT / "susc_gt02_manifesto.json"
GT02_INPUTS = [GT02_REPLAY, GT02_DECISIONS, GT02_BLOCKERS, GT02_SUMMARY, GT02_MANIFEST]

# Saidas publicas (portugues)
REPORT = OUT / "SUSC_GT03_PACOTE_ALVOS_AQUISICAO_EVIDENCIA_FORTE.md"
TARGETS_CSV = OUT / "susc_gt03_pacote_alvos_evidencia_forte.csv"
MISSING_CSV = OUT / "susc_gt03_requisitos_faltantes_por_alvo.csv"
TRACKS_CSV = OUT / "susc_gt03_trilhas_aquisicao.csv"
SOURCES_CSV = OUT / "susc_gt03_priorizacao_fontes.csv"
PLAN_CSV = OUT / "susc_gt03_plano_aquisicao_por_alvo.csv"
PRIORITY_SUMMARY_CSV = OUT / "susc_gt03_resumo_prioridades.csv"
BLOCKERS_CSV = OUT / "susc_gt03_bloqueios.csv"
MANIFEST_JSON = OUT / "susc_gt03_manifesto.json"

SCHEMA_TARGET = SCHEMAS_DIR / "susc_gt03_evidence_target.schema.json"
SCHEMA_MISSING = SCHEMAS_DIR / "susc_gt03_missing_requirement.schema.json"
SCHEMA_TRACK = SCHEMAS_DIR / "susc_gt03_acquisition_track.schema.json"
SCHEMA_PRIORITY = SCHEMAS_DIR / "susc_gt03_priority_summary.schema.json"

PUBLIC_ARTIFACTS = [
    REPORT, TARGETS_CSV, MISSING_CSV, TRACKS_CSV, SOURCES_CSV, PLAN_CSV,
    PRIORITY_SUMMARY_CSV, BLOCKERS_CSV,
]
REQUIRED_OUTPUTS = PUBLIC_ARTIFACTS + [
    MANIFEST_JSON, SCHEMA_TARGET, SCHEMA_MISSING, SCHEMA_TRACK, SCHEMA_PRIORITY,
]

MARCO = "SUSC-GT03"
TITULO_PUBLICO = "Pacote de Alvos para Aquisição de Evidência Forte"
TITULO_H1_TOKEN = "Aquisição de Evidência Forte"

PUBLIC_FORBIDDEN_RE = gt02.PUBLIC_FORBIDDEN_RE
OPERATIONAL_FORECAST_RE = gt02.OPERATIONAL_FORECAST_RE  # ja trata negacao acentuada

GLOBAL_GUARDRAILS = {
    "eligible_for_training": "false",
    "eligible_for_ground_truth": "false",
    "trainable": "false",
    "score_v7_candidate": "false",
    "review_only": "true",
}

# ---------------------------------------------------------------------------
# Trilhas de aquisicao e classes de prioridade
# ---------------------------------------------------------------------------
ACQUISITION_TRACKS = [
    "resolver_data_evento",
    "obter_geometria_oficial",
    "produzir_footprint_tecnico_sar",
    "estimar_incerteza_espacial",
    "executar_qa_humano",
    "separar_fenomeno",
    "resolver_patch_link",
    "descartar_ou_rebaixar",
]
PRIORITY_CLASSES = [
    "prioridade_alta",
    "prioridade_media",
    "prioridade_baixa",
    "bloqueado_sem_acao_imediata",
    "descartar_como_contexto",
]
REGIOES_TRABALHADAS = {"recife", "curitiba", "petropolis"}

# reuso das regras do GT02
STRONG_GEOMETRY_TYPES = gt02.STRONG_GEOMETRY_TYPES
STRONG_PATCH_LINK_QUALITY = gt02.STRONG_PATCH_LINK_QUALITY
QA_ACCEPTED = gt02.QA_ACCEPTED
FLOOD_PHENOMENA = gt02.FLOOD_PHENOMENA
WEAK_SOURCE_TYPES = gt02.WEAK_SOURCE_TYPES
PETROPOLIS_BLOCKING_PHENOMENA = gt02.PETROPOLIS_BLOCKING_PHENOMENA
STATE_NAMES = gt02.STATE_NAMES

_present = gt02._present
_num = gt02._num
_clean = gt02._clean

# Objetivo/entrada/saida de cada trilha (portugues) para a tabela de trilhas.
TRACK_INFO = {
    "resolver_data_evento": (
        "resolver a data (ou janela) do evento observado",
        "registro documental/oficial com data ja existente no acervo",
        "event_date e janela pre/pos-evento definidas"),
    "obter_geometria_oficial": (
        "obter geometria oficial do evento (ponto, poligono ou bbox)",
        "artefato oficial/observado ja existente no acervo",
        "geometry_type forte associada ao evento"),
    "produzir_footprint_tecnico_sar": (
        "definir alvo de footprint tecnico SAR (sem executar SAR neste marco)",
        "evento datado e area candidata ja conhecida",
        "especificacao de footprint SAR para execucao futura"),
    "estimar_incerteza_espacial": (
        "estimar a incerteza espacial em metros e a qualidade do vinculo",
        "geometria/vinculo ja disponivel para medir incerteza",
        "uncertainty_m e patch_link_quality quantificados"),
    "executar_qa_humano": (
        "submeter o candidato a QA humano e registrar aceite",
        "candidato ja normalizado pelo GT02",
        "qa_status aceito"),
    "separar_fenomeno": (
        "separar o fenomeno (flood/flash_flood/river_flood/landslide/misto)",
        "descricao do evento ja existente",
        "phenomenon_type compativel e separado da mancha de inundacao"),
    "resolver_patch_link": (
        "resolver o vinculo forte entre geometria/evento e patch REV-P",
        "geometria e grade de patches ja existentes",
        "patch_link_quality forte auditavel"),
    "descartar_ou_rebaixar": (
        "manter apenas como contexto documental, sem promocao patch-level",
        "registro de fonte fraca ja existente",
        "documentacao contextual preservada, sem patch positivo"),
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
    missing = [rel(p) for p in GT02_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("entradas GT02 ausentes: " + "; ".join(missing))


# ---------------------------------------------------------------------------
# Analise de um registro (flags + fonte fraca + petropolis)
# ---------------------------------------------------------------------------
def _detect_weak(fields: dict) -> tuple[bool, str]:
    for key in ("evidence_source_kind", "source_type_guess", "source_authority",
                "geometry_type", "phenomenon_type"):
        v = _clean(fields.get(key)).lower()
        if v in WEAK_SOURCE_TYPES:
            return True, v
    return False, "generic"


def _is_petropolis(fields: dict) -> bool:
    blob = f"{_clean(fields.get('region'))} {_clean(fields.get('city'))}".lower()
    return "petropolis" in blob or "petrópolis" in blob


def analyze(fields: dict) -> dict:
    """Deriva flags de presenca/forca a partir dos campos normalizados do GT02."""
    geom = _clean(fields.get("geometry_type")).lower()
    plq = _clean(fields.get("patch_link_quality")).lower()
    pheno = _clean(fields.get("phenomenon_type")).lower()
    weak, kind = _detect_weak(fields)
    petro = _is_petropolis(fields)
    petro_block = petro and (pheno in PETROPOLIS_BLOCKING_PHENOMENA or not _present(fields.get("phenomenon_type")))
    flags = {
        "has_event": _present(fields.get("event_id")),
        "has_footprint": _present(fields.get("footprint_id")) or _present(fields.get("geometry_id")),
        "has_patch": _present(fields.get("patch_id")),
        "has_date": _present(fields.get("event_date")),
        "has_source": _present(fields.get("source_authority")) and not weak,
        "has_geom_any": _present(fields.get("geometry_type")),
        "geom_strong": geom in STRONG_GEOMETRY_TYPES,
        "has_plq": _present(fields.get("patch_link_quality")),
        "plq_strong": plq in STRONG_PATCH_LINK_QUALITY,
        "has_unc": _num(fields.get("uncertainty_m")) is not None,
        "qa_ok": _clean(fields.get("qa_status")).lower() in QA_ACCEPTED,
        "pheno_present": _present(fields.get("phenomenon_type")) and pheno != "insufficient_type",
        "pheno_flood": pheno in FLOOD_PHENOMENA,
        "in_worked_region": _clean(fields.get("region")).lower() in REGIOES_TRABALHADAS,
        "weak": weak,
        "source_kind": kind,
        "petro": petro,
        "petro_block": petro_block,
    }
    return flags


# requisito -> (severity, prevents_positive_strong, como_resolver_pt)
REQUIREMENT_INFO = {
    "event_date": ("alta", "true", "resolver data do evento em registro oficial/documental (trilha resolver_data_evento)"),
    "source_authority": ("alta", "true", "identificar fonte/autoridade rastreavel do evento"),
    "geometry_type": ("alta", "true", "obter geometria oficial (ponto/poligono/bbox) do evento (trilha obter_geometria_oficial)"),
    "patch_id": ("media", "true", "resolver o patch REV-P correspondente (trilha resolver_patch_link)"),
    "patch_link_quality": ("alta", "true", "produzir vinculo espacial forte com o patch (trilha resolver_patch_link)"),
    "uncertainty_m": ("media", "true", "estimar incerteza espacial em metros (trilha estimar_incerteza_espacial)"),
    "qa_status": ("alta", "true", "executar QA humano e registrar aceite (trilha executar_qa_humano)"),
    "phenomenon_type": ("alta", "true", "confirmar e separar o fenomeno de inundacao (trilha separar_fenomeno)"),
    "pre_event_window": ("media", "false", "definir janela pre-evento para footprint SAR futuro (trilha produzir_footprint_tecnico_sar)"),
    "post_event_window": ("media", "false", "definir janela pos-evento para footprint SAR futuro (trilha produzir_footprint_tecnico_sar)"),
}
STRONG_REQUIREMENTS = [
    "event_date", "source_authority", "geometry_type", "patch_id",
    "patch_link_quality", "uncertainty_m", "qa_status", "phenomenon_type",
]


def compute_missing(flags: dict) -> list[str]:
    checks = {
        "event_date": flags["has_date"],
        "source_authority": flags["has_source"],
        "geometry_type": flags["geom_strong"],
        "patch_id": flags["has_patch"],
        "patch_link_quality": flags["plq_strong"],
        "uncertainty_m": flags["has_unc"],
        "qa_status": flags["qa_ok"],
        "phenomenon_type": flags["pheno_flood"],
    }
    return [k for k in STRONG_REQUIREMENTS if not checks[k]]


def acquisition_tracks(flags: dict) -> list[str]:
    tracks = []
    if flags["weak"]:
        tracks.append("descartar_ou_rebaixar")
    if not flags["has_date"]:
        tracks.append("resolver_data_evento")
    if not flags["geom_strong"] and (flags["has_event"] or flags["has_source"] or flags["has_geom_any"]):
        tracks.append("obter_geometria_oficial")
    if flags["has_date"] and not flags["geom_strong"] and (flags["in_worked_region"] or flags["has_footprint"]):
        tracks.append("produzir_footprint_tecnico_sar")
    if not flags["has_unc"] or not flags["plq_strong"]:
        tracks.append("estimar_incerteza_espacial")
    if not flags["qa_ok"]:
        tracks.append("executar_qa_humano")
    if flags["petro_block"] or (flags["petro"] and not flags["pheno_flood"]):
        tracks.append("separar_fenomeno")
    if not flags["plq_strong"] and (flags["has_event"] or flags["has_geom_any"] or flags["has_footprint"]):
        tracks.append("resolver_patch_link")
    # ordem estavel e sem duplicatas
    return [t for t in ACQUISITION_TRACKS if t in dict.fromkeys(tracks)]


def priority_score(flags: dict, assigned_state: str, missing: list[str]) -> int:
    s = 0
    # positivos
    if flags["has_event"]:
        s += 12
    if flags["has_date"]:
        s += 14
    if flags["in_worked_region"]:
        s += 8
    if flags["has_source"]:
        s += 10
    if flags["has_geom_any"]:
        s += 6
    if flags["geom_strong"]:
        s += 6
    if flags["has_patch"]:
        s += 8
    if flags["has_footprint"]:
        s += 6
    if flags["pheno_flood"]:
        s += 8
    if assigned_state == "positive_provisional":
        s += 12
    if len(missing) <= 2:
        s += 10
    # negativos
    if flags["weak"]:
        s -= 40
    if not flags["has_date"]:
        s -= 8
    if not flags["has_geom_any"]:
        s -= 8
    if not flags["has_patch"]:
        s -= 6
    if flags["petro_block"]:
        s -= 20
    if assigned_state == "no_data" and len(missing) >= 6:
        s -= 10
    return max(0, min(100, s))


def priority_class(flags: dict, score: int, missing: list[str]) -> str:
    if flags["weak"]:
        return "descartar_como_contexto"
    # criticos demais ausentes -> bloqueado
    sem_ancora = not (flags["has_event"] or flags["has_footprint"] or flags["has_patch"])
    if sem_ancora or (not flags["has_date"] and not flags["has_geom_any"] and not flags["has_patch"]):
        return "bloqueado_sem_acao_imediata"
    # petropolis misto nunca vira alta
    teto_alta = not flags["petro_block"]
    if teto_alta and score >= 65 and len(missing) <= 2:
        return "prioridade_alta"
    if score >= 45:
        return "prioridade_media"
    if score >= 20:
        return "prioridade_baixa"
    return "bloqueado_sem_acao_imediata"


def can_become_strong(flags: dict, missing: list[str]) -> bool:
    if flags["weak"] or flags["petro_block"]:
        return False
    if not flags["has_event"]:
        return False
    if not (flags["has_footprint"] or flags["has_patch"] or flags["has_geom_any"] or flags["has_date"]):
        return False
    return len(missing) > 0  # ainda falta algo, mas e adquirivel


# ---------------------------------------------------------------------------
# Construcao dos alvos
# ---------------------------------------------------------------------------
def _target_kind(flags: dict, assigned_state: str) -> str:
    if flags["weak"]:
        return "fonte_fraca_contextual"
    if assigned_state == "positive_provisional":
        return "referencia_provisoria_para_promocao"
    if assigned_state == "no_data" and (flags["has_event"] or flags["has_footprint"] or flags["has_patch"]):
        return "no_data_com_sinal_observacional"
    return "no_data_sem_acao_clara"


def _expected_next_artifact(tracks: list[str]) -> str:
    primary = tracks[0] if tracks else "descartar_ou_rebaixar"
    return {
        "resolver_data_evento": "event_date e janela pre/pos-evento resolvidas",
        "obter_geometria_oficial": "geometria oficial (ponto/poligono/bbox) do evento",
        "produzir_footprint_tecnico_sar": "especificacao de footprint SAR (alvo, sem execucao)",
        "estimar_incerteza_espacial": "uncertainty_m estimada",
        "executar_qa_humano": "qa_status humano aceito",
        "separar_fenomeno": "fenomeno separado (flood/flash_flood/landslide/misto)",
        "resolver_patch_link": "patch_link forte auditavel",
        "descartar_ou_rebaixar": "documentacao contextual mantida (sem patch-level)",
    }.get(primary, "revisao_manual")


def _rationale_pt(flags: dict, assigned_state: str, priority_cls: str, missing: list[str]) -> str:
    if flags["weak"]:
        return f"fonte fraca ({flags['source_kind']}): serve apenas como contexto documental; nunca vira positive_strong"
    if flags["petro_block"]:
        return "Petropolis com fenomeno misto/deslizamento sem separacao: exige separar_fenomeno antes de qualquer promocao"
    base = f"estado GT02={assigned_state}; faltam {len(missing)} requisito(s) para referencia forte"
    if priority_cls == "prioridade_alta":
        return base + "; faltam poucos requisitos fechaveis"
    if priority_cls == "bloqueado_sem_acao_imediata":
        return base + "; campos criticos demais ausentes para acao imediata"
    return base + "; promissor porem incompleto, permanece review-only"


def build_targets() -> dict:
    replay = _read_csv(GT02_REPLAY)
    decisions = _read_csv(GT02_DECISIONS)
    blockers = _read_csv(GT02_BLOCKERS)
    # blockers de fenomeno por replay_id (cruzamento auditavel)
    petro_blocked_ids = {b["replay_id"] for b in blockers if b["blocker_type"] == "phenomenon_mismatch_or_mixed_event"}

    targets, missing_rows, track_rows, plan_rows, blocker_rows = [], [], [], [], []
    t_idx = m_idx = k_idx = p_idx = b_idx = 1

    for i, r in enumerate(replay):
        d = decisions[i] if i < len(decisions) else {}
        fields = {
            "event_id": r.get("event_id", ""), "footprint_id": r.get("footprint_id", ""),
            "geometry_id": r.get("geometry_id", ""), "patch_id": r.get("patch_id", ""),
            "city": r.get("city", ""), "region": r.get("region", ""),
            "phenomenon_type": r.get("phenomenon_type", ""),
            "source_authority": r.get("source_authority", ""),
            "geometry_type": r.get("geometry_type", ""),
            "patch_link_quality": r.get("patch_link_quality", ""),
            "qa_status": r.get("qa_status", ""), "event_date": r.get("event_date", ""),
            "uncertainty_m": r.get("uncertainty_m", ""),
        }
        flags = analyze(fields)
        assigned = r.get("assigned_state", "")
        missing = compute_missing(flags)
        tracks = acquisition_tracks(flags)
        score = priority_score(flags, assigned, missing)
        cls = priority_class(flags, score, missing)
        can_strong = can_become_strong(flags, missing)
        target_id = f"TGT_{t_idx:04d}"
        t_idx += 1

        targets.append({
            "target_id": target_id,
            "source_replay_id": r.get("replay_id", ""),
            "source_decision_id": d.get("decision_id", ""),
            "event_id": _norm(fields["event_id"]),
            "footprint_id": _norm(fields["footprint_id"]),
            "geometry_id": _norm(fields["geometry_id"]),
            "patch_id": _norm(fields["patch_id"]),
            "city": _norm(fields["city"]),
            "region": _norm(fields["region"]),
            "phenomenon_type": _norm(fields["phenomenon_type"]),
            "assigned_state_gt02": assigned,
            "source_authority": _norm(fields["source_authority"]),
            "geometry_type": _norm(fields["geometry_type"]),
            "patch_link_quality": _norm(fields["patch_link_quality"]),
            "qa_status": _norm(fields["qa_status"]),
            "event_date": _norm(fields["event_date"]),
            "uncertainty_m": _norm(fields["uncertainty_m"]),
            "target_kind": _target_kind(flags, assigned),
            "priority_score": str(score),
            "priority_class": cls,
            "acquisition_tracks": ";".join(tracks) if tracks else "nenhuma",
            "missing_requirements": ";".join(missing) if missing else "none",
            "expected_next_artifact": _expected_next_artifact(tracks),
            "can_become_positive_strong_after_acquisition": "true" if can_strong else "false",
            "remains_review_only": "true",
            "trainable": "false",
            "eligible_for_training": "false",
            "eligible_for_ground_truth": "false",
            "score_v7_candidate": "false",
            "rationale_pt": _rationale_pt(flags, assigned, cls, missing),
        })

        # requisitos faltantes (fortes + janelas se trilha SAR)
        req_list = list(missing)
        if "produzir_footprint_tecnico_sar" in tracks:
            req_list += ["pre_event_window", "post_event_window"]
        for req in req_list:
            sev, prevents_ps, how = REQUIREMENT_INFO[req]
            missing_rows.append({
                "missing_requirement_id": f"MRQ_{m_idx:05d}",
                "target_id": target_id,
                "requirement": req,
                "required_for_positive_strong": "true" if req in STRONG_REQUIREMENTS else "false",
                "current_value": _norm(fields.get(req, "")) if req in fields else "nao_disponivel_no_replay_gt02",
                "severity": sev,
                "how_to_resolve_pt": how,
                "prevents_positive_strong": prevents_ps,
                "prevents_calibration": "true" if req in ("qa_status", "phenomenon_type", "geometry_type") else "false",
                "prevents_benchmark": "true",
            })
            m_idx += 1

        # trilhas
        for tr in tracks:
            obj, inp, outp = TRACK_INFO[tr]
            track_rows.append({
                "track_id": f"TRK_{k_idx:05d}",
                "target_id": target_id,
                "acquisition_track": tr,
                "objective_pt": obj,
                "required_input_pt": inp,
                "expected_output_pt": outp,
                "no_network_in_this_milestone": "true",
                "no_sar_execution_in_this_milestone": "true",
                "review_only": "true",
                "blocker_if_not_done": _blocker_if_not_done(tr),
            })
            k_idx += 1

        # plano operacional (passos) para alvos acionaveis; terminal para os demais
        if cls in ("prioridade_alta", "prioridade_media", "prioridade_baixa") and tracks:
            for order, tr in enumerate(tracks, start=1):
                obj, inp, outp = TRACK_INFO[tr]
                plan_rows.append({
                    "plan_id": f"PLN_{p_idx:05d}",
                    "target_id": target_id,
                    "step_order": str(order),
                    "action_pt": obj,
                    "expected_artifact_pt": outp,
                    "success_criteria_pt": f"{outp} registrada e auditavel, mantendo review_only",
                    "fail_closed_outcome_pt": "sem o artefato, o alvo permanece no estado atual; nao ha promocao",
                    "manual_review_required": "true",
                    "can_be_automated_future": "true" if tr not in ("executar_qa_humano", "separar_fenomeno") else "false",
                })
                p_idx += 1
        else:
            plan_rows.append({
                "plan_id": f"PLN_{p_idx:05d}",
                "target_id": target_id,
                "step_order": "1",
                "action_pt": "manter como contexto/bloqueado; sem acao de aquisicao neste marco",
                "expected_artifact_pt": "nenhum artefato de aquisicao neste marco",
                "success_criteria_pt": "registro preservado como contexto auditavel",
                "fail_closed_outcome_pt": "permanece review-only, sem promocao e sem treino",
                "manual_review_required": "true",
                "can_be_automated_future": "false",
            })
            p_idx += 1

        # bloqueios do alvo
        for bl in _target_blockers(flags, cls, target_id):
            bl["blocker_id"] = f"BLK_{b_idx:05d}"
            blocker_rows.append(bl)
            b_idx += 1

    return {
        "targets": targets, "missing": missing_rows, "tracks": track_rows,
        "plan": plan_rows, "blockers": blocker_rows,
        "petro_blocked_ids": petro_blocked_ids,
    }


def _norm(v: str) -> str:
    v = _clean(v)
    return v if v else "not_available"


def _blocker_if_not_done(track: str) -> str:
    return {
        "resolver_data_evento": "sem data, nao ha janela pre/pos-evento e nao vira positive_strong",
        "obter_geometria_oficial": "sem geometria oficial forte, nao vira positive_strong",
        "produzir_footprint_tecnico_sar": "sem footprint tecnico, o vinculo forte fica indisponivel",
        "estimar_incerteza_espacial": "sem incerteza, a qualidade do vinculo fica indefinida",
        "executar_qa_humano": "sem QA aceito, nao vira positive_strong nem calibracao",
        "separar_fenomeno": "sem separar o fenomeno, a calibracao fica bloqueada",
        "resolver_patch_link": "sem vinculo forte, nao ha avaliacao patch-level",
        "descartar_ou_rebaixar": "permanece apenas como contexto documental",
    }.get(track, "bloqueio nao especificado")


def _target_blockers(flags, cls, target_id) -> list[dict]:
    out = []

    def add(btype, sev, reason, ph=False, ps=True, ev=False, cal=False):
        out.append({
            "target_id": target_id, "blocker_type": btype, "severity": sev,
            "reason_pt": reason,
            "prevents_priority_high": "true" if ph else "false",
            "prevents_positive_strong": "true" if ps else "false",
            "prevents_evaluation": "true" if ev else "false",
            "prevents_calibration": "true" if cal else "false",
        })

    if flags["weak"]:
        add("fonte_fraca_contexto", "alta",
            f"fonte fraca ({flags['source_kind']}) nao gera patch positivo forte; manter como contexto",
            ph=True, ps=True, ev=True, cal=True)
    if flags["petro_block"]:
        add("phenomenon_mismatch_or_mixed_event", "alta",
            "Petropolis com fenomeno misto/deslizamento sem separacao da mancha de inundacao",
            ph=True, ps=True, ev=True, cal=True)
    if cls == "bloqueado_sem_acao_imediata":
        add("campos_criticos_ausentes", "alta",
            "faltam campos criticos demais (evento/data/geometria/patch) para acao imediata",
            ph=True, ps=True, ev=True, cal=True)
    return out


# ---------------------------------------------------------------------------
# Resumos
# ---------------------------------------------------------------------------
def priorizacao_fontes_rows(targets: list[dict]) -> list[dict]:
    agg = {}
    for t in targets:
        src = t["source_authority"]
        a = agg.setdefault(src, {"count": 0, "alta": 0, "media": 0, "baixa": 0, "blocked": 0, "missing": {}})
        a["count"] += 1
        cls = t["priority_class"]
        if cls == "prioridade_alta":
            a["alta"] += 1
        elif cls == "prioridade_media":
            a["media"] += 1
        elif cls == "prioridade_baixa":
            a["baixa"] += 1
        else:
            a["blocked"] += 1
        for req in t["missing_requirements"].split(";"):
            if req and req != "none":
                a["missing"][req] = a["missing"].get(req, 0) + 1
    rows = []
    for src in sorted(agg):
        a = agg[src]
        main_missing = max(a["missing"], key=a["missing"].get) if a["missing"] else "none"
        rows.append({
            "source_authority": src,
            "source_type_guess": _source_type_guess(src),
            "target_count": str(a["count"]),
            "high_priority_count": str(a["alta"]),
            "medium_priority_count": str(a["media"]),
            "low_priority_count": str(a["baixa"]),
            "blocked_count": str(a["blocked"]),
            "main_missing_requirement": main_missing,
            "recommended_next_action_pt": _recommended_action(main_missing),
        })
    return rows


def _source_type_guess(src: str) -> str:
    s = src.lower()
    if s in WEAK_SOURCE_TYPES:
        return s
    if "defesa_civil" in s:
        return "defesa_civil"
    if "official" in s or "oficial" in s:
        return "artefato_oficial_local"
    if s in ("", "not_available"):
        return "sem_fonte"
    return "fonte_generica"


def _recommended_action(main_missing: str) -> str:
    return {
        "event_date": "priorizar resolucao de datas e janelas pre/pos-evento",
        "geometry_type": "priorizar obtencao de geometria oficial",
        "qa_status": "priorizar protocolo de QA humano",
        "uncertainty_m": "priorizar estimativa de incerteza espacial",
        "patch_link_quality": "priorizar resolucao de patch_link forte",
        "phenomenon_type": "priorizar separacao de fenomeno",
        "none": "sem requisito dominante; revisar caso a caso",
    }.get(main_missing, "revisar requisitos faltantes caso a caso")


def resumo_prioridades_rows(targets: list[dict]) -> list[dict]:
    order = {c: i for i, c in enumerate(PRIORITY_CLASSES)}
    agg = {}
    for t in targets:
        cls = t["priority_class"]
        a = agg.setdefault(cls, {"count": 0, "score_sum": 0, "can": 0})
        a["count"] += 1
        a["score_sum"] += int(t["priority_score"])
        a["can"] += 1 if t["can_become_positive_strong_after_acquisition"] == "true" else 0
    rows = []
    for cls in sorted(agg, key=lambda c: order.get(c, 99)):
        a = agg[cls]
        rows.append({
            "priority_class": cls,
            "count": str(a["count"]),
            "average_priority_score": str(round(a["score_sum"] / a["count"], 2)) if a["count"] else "0",
            "can_become_positive_strong_after_acquisition_count": str(a["can"]),
            "training_allowed_count": "0",
            "ground_truth_allowed_count": "0",
            "score_v7_candidate_count": "0",
        })
    return rows


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
_BOOL = {"enum": ["true", "false"]}
_FALSE = {"const": "false"}


def schema_target_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "susc_gt03_evidence_target.schema.json",
        "title": "SUSC-GT03 Alvo de Evidencia",
        "type": "object",
        "required": [
            "target_id", "source_replay_id", "assigned_state_gt02", "target_kind",
            "priority_score", "priority_class", "can_become_positive_strong_after_acquisition",
            "remains_review_only", "trainable", "eligible_for_training",
            "eligible_for_ground_truth", "score_v7_candidate", "rationale_pt",
        ],
        "properties": {
            "target_id": {"type": "string", "pattern": "^TGT_\\d+$"},
            "source_replay_id": {"type": "string", "pattern": "^RP_\\d+$"},
            "assigned_state_gt02": {"enum": STATE_NAMES},
            "target_kind": {"type": "string", "minLength": 1},
            "priority_score": {"type": "string", "pattern": "^(100|[1-9]?[0-9])$"},
            "priority_class": {"enum": PRIORITY_CLASSES},
            "acquisition_tracks": {"type": "string"},
            "missing_requirements": {"type": "string"},
            "can_become_positive_strong_after_acquisition": _BOOL,
            "remains_review_only": {"const": "true"},
            "trainable": _FALSE,
            "eligible_for_training": _FALSE,
            "eligible_for_ground_truth": _FALSE,
            "score_v7_candidate": _FALSE,
            "rationale_pt": {"type": "string", "minLength": 1},
        },
    }


def schema_missing_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "susc_gt03_missing_requirement.schema.json",
        "title": "SUSC-GT03 Requisito Faltante",
        "type": "object",
        "required": [
            "missing_requirement_id", "target_id", "requirement",
            "required_for_positive_strong", "severity", "how_to_resolve_pt",
            "prevents_positive_strong",
        ],
        "properties": {
            "missing_requirement_id": {"type": "string", "pattern": "^MRQ_\\d+$"},
            "target_id": {"type": "string", "pattern": "^TGT_\\d+$"},
            "requirement": {"enum": list(REQUIREMENT_INFO.keys()) + [
                "observability", "exclusion_check"]},
            "required_for_positive_strong": _BOOL,
            "current_value": {"type": "string"},
            "severity": {"enum": ["alta", "media", "baixa"]},
            "how_to_resolve_pt": {"type": "string", "minLength": 1},
            "prevents_positive_strong": _BOOL,
            "prevents_calibration": _BOOL,
            "prevents_benchmark": _BOOL,
        },
    }


def schema_track_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "susc_gt03_acquisition_track.schema.json",
        "title": "SUSC-GT03 Trilha de Aquisicao",
        "type": "object",
        "required": [
            "track_id", "target_id", "acquisition_track", "objective_pt",
            "no_network_in_this_milestone", "no_sar_execution_in_this_milestone",
            "review_only",
        ],
        "properties": {
            "track_id": {"type": "string", "pattern": "^TRK_\\d+$"},
            "target_id": {"type": "string", "pattern": "^TGT_\\d+$"},
            "acquisition_track": {"enum": ACQUISITION_TRACKS},
            "objective_pt": {"type": "string", "minLength": 1},
            "required_input_pt": {"type": "string"},
            "expected_output_pt": {"type": "string"},
            "no_network_in_this_milestone": {"const": "true"},
            "no_sar_execution_in_this_milestone": {"const": "true"},
            "review_only": {"const": "true"},
            "blocker_if_not_done": {"type": "string"},
        },
    }


def schema_priority_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "susc_gt03_priority_summary.schema.json",
        "title": "SUSC-GT03 Resumo de Prioridades",
        "type": "object",
        "required": [
            "priority_class", "count", "average_priority_score",
            "can_become_positive_strong_after_acquisition_count",
            "training_allowed_count", "ground_truth_allowed_count",
            "score_v7_candidate_count",
        ],
        "properties": {
            "priority_class": {"enum": PRIORITY_CLASSES},
            "count": {"type": "string", "pattern": "^\\d+$"},
            "average_priority_score": {"type": "string"},
            "can_become_positive_strong_after_acquisition_count": {"type": "string", "pattern": "^\\d+$"},
            "training_allowed_count": {"const": "0"},
            "ground_truth_allowed_count": {"const": "0"},
            "score_v7_candidate_count": {"const": "0"},
        },
    }


# ---------------------------------------------------------------------------
# Manifesto / proximo marco
# ---------------------------------------------------------------------------
def _dominant_gargalo(targets: list[dict]) -> str:
    counts = {}
    datados = 0
    for t in targets:
        if t["can_become_positive_strong_after_acquisition"] != "true":
            continue
        if t["event_date"] != "not_available":
            datados += 1
        for req in t["missing_requirements"].split(";"):
            if req and req != "none":
                counts[req] = counts.get(req, 0) + 1
    return _next_milestone(counts, datados)


def _next_milestone(missing_counts: dict, datados: int) -> str:
    # Data e requisito upstream: sem event_date nao ha janela para SAR nem base
    # para QA. Se mais alvos promissores carecem de data do que ja tem data,
    # o gargalo dominante e a resolucao de datas.
    ed = missing_counts.get("event_date", 0)
    qa = missing_counts.get("qa_status", 0)
    if ed > datados and ed > 0:
        return "GT04 - Resolvedor de Datas e Janelas Pre/Pos-Evento"
    if datados >= 30:
        return "GT04 - Preparacao de Canarios SAR sem Execucao"
    if qa > 0:
        return "GT04 - Protocolo de QA Humano"
    return "GT04 - Resolvedor de Datas e Janelas Pre/Pos-Evento"


def manifest_obj(bundle: dict) -> dict:
    targets = bundle["targets"]
    dist = {}
    for t in targets:
        dist[t["priority_class"]] = dist.get(t["priority_class"], 0) + 1
    gt02_head = _run_git(["log", "-1", "--format=%h", "--", rel(GT02_REPLAY)]) or "unknown"
    artifacts = [
        {"path": rel(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in PUBLIC_ARTIFACTS + [SCHEMA_TARGET, SCHEMA_MISSING, SCHEMA_TRACK, SCHEMA_PRIORITY]
        if p.exists()
    ]
    return {
        "marco": MARCO,
        "titulo_publico": TITULO_PUBLICO,
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "gt02_source_commit": gt02_head,
        "alvos_total": len(targets),
        "distribuicao_prioridades": dist,
        "positive_strong_promovidos": 0,
        "can_become_strong_count": sum(1 for t in targets if t["can_become_positive_strong_after_acquisition"] == "true"),
        "requisitos_faltantes_linhas": len(bundle["missing"]),
        "trilhas_linhas": len(bundle["tracks"]),
        "bloqueios_linhas": len(bundle["blockers"]),
        "global_guardrails": GLOBAL_GUARDRAILS,
        "training_true_count": sum(1 for t in targets if t["eligible_for_training"] == "true"),
        "ground_truth_true_count": sum(1 for t in targets if t["eligible_for_ground_truth"] == "true"),
        "score_v7_candidate_true_count": sum(1 for t in targets if t["score_v7_candidate"] == "true"),
        "score_v6_changed": gt02.gt01._score_v6_changed(),
        "score_v7_created": gt02.gt01._score_v7_exists(),
        "proximo_marco": _dominant_gargalo(targets),
        "artifacts": artifacts,
    }


# ---------------------------------------------------------------------------
# Relatorio publico (portugues)
# ---------------------------------------------------------------------------
def _top_targets(targets: list[dict], n=10) -> list[dict]:
    ordem = {c: i for i, c in enumerate(PRIORITY_CLASSES)}
    return sorted(targets, key=lambda t: (ordem.get(t["priority_class"], 99), -int(t["priority_score"])))[:n]


def report_text(manifest, bundle, priorities, sources) -> str:
    targets = bundle["targets"]
    linhas_prior = "\n".join(
        f"| {r['priority_class']} | {r['count']} | {r['average_priority_score']} | "
        f"{r['can_become_positive_strong_after_acquisition_count']} | {r['training_allowed_count']} |"
        for r in priorities
    )
    # requisitos faltantes agregados
    req_counts = {}
    for m in bundle["missing"]:
        req_counts[m["requirement"]] = req_counts.get(m["requirement"], 0) + 1
    linhas_req = "\n".join(
        f"| {k} | {v} |" for k, v in sorted(req_counts.items(), key=lambda kv: -kv[1])
    )
    # trilhas agregadas
    trk_counts = {}
    for tr in bundle["tracks"]:
        trk_counts[tr["acquisition_track"]] = trk_counts.get(tr["acquisition_track"], 0) + 1
    linhas_trk = "\n".join(
        f"| {k} | {v} |" for k, v in sorted(trk_counts.items(), key=lambda kv: -kv[1])
    )
    linhas_top = "\n".join(
        f"| {t['target_id']} | {t['priority_class']} | {t['priority_score']} | "
        f"{t['region']} | {t['assigned_state_gt02']} | {t['acquisition_tracks']} |"
        for t in _top_targets(targets)
    )
    linhas_src = "\n".join(
        f"| {r['source_authority']} | {r['target_count']} | {r['high_priority_count']} | "
        f"{r['blocked_count']} | {r['main_missing_requirement']} |"
        for r in sources
    )
    n_petro = sum(1 for t in targets if t["region"] == "petropolis")

    return f"""# {MARCO} - {TITULO_PUBLICO}

## 1. Escopo do marco

Este marco transforma o resultado do replay do SUSC-GT02 em uma **fila auditável de
alvos** para aquisição futura de evidência forte. Ele **não** promove nenhum estado,
**não** busca internet, **não** baixa dado novo, **não** roda SAR nem GEE, **não** cria
footprint, **não** altera o `score_v6`, **não** cria `score_v7`, **não** treina modelo e
**não** altera as decisões do GT02 (apenas as lê). Tudo permanece **review-only** (uso
restrito a revisão).

## 2. Relação com GT01 e GT02

O GT01 definiu a política (estados, requisitos, bloqueios). O GT02 aplicou a política aos
artefatos existentes e concluiu que o acervo atual tem 86 `positive_provisional`, 471
`no_data` e **0** `positive_strong`. O GT03 lê exatamente esses registros e diagnostica,
por alvo, **o que falta** para cada um virar candidato forte no futuro.

## 3. Por que o GT03 ainda não calibra o score

Sem referência forte (`positive_strong`) não há base para calibração nem benchmark. O
GT03 apenas organiza a aquisição futura; calibrar agora seria promover evidência
insuficiente a verdade — proibido pela política.

## 4. Por que o GT03 não busca internet nem roda SAR

Este é um marco de **planejamento determinístico**: monta a fila e o plano a partir do
que já existe no repositório. Aquisição real (datas, geometria oficial, footprint SAR,
QA humano) fica para os marcos seguintes, com execução controlada e auditável.

## 5. Entradas usadas

Foram lidos os outputs versionados do GT02 (sem regeração): registro de replay,
decisões por patch, bloqueios, resumo de estados e manifesto. Total de alvos gerados:
**{manifest['alvos_total']}** (um por registro de replay do GT02).

## 6. Critérios de priorização

O `priority_score` é determinístico, de 0 a 100. Pontua positivamente a presença de
evento (`event_id`), data (`event_date`), cidade/região já trabalhada (Recife, Curitiba,
Petrópolis), fonte forte (`source_authority`), geometria (parcial ou forte), patch,
footprint e fenômeno de inundação, além de já ser `positive_provisional` no GT02 e de
faltarem poucos requisitos. Pontua negativamente fontes fracas (alerta, área de risco,
notícia, contexto municipal, mapa de suscetibilidade), ausência total de
data/geometria/patch, fenômeno misto sem separação e `no_data` sem ação clara. A
priorização **não** promove estado: mesmo um alvo de prioridade alta continua
`review_only` e não treinável.

## 7. Distribuição das prioridades

| priority_class | alvos | score médio | podem virar forte após aquisição | treino permitido |
| --- | --- | --- | --- | --- |
{linhas_prior}

Leitura pública: *priority_class* é a classe de prioridade; *podem virar forte após
aquisição* conta alvos que, adquiridos os requisitos, poderiam alcançar `positive_strong`
no futuro; *treino permitido* é **sempre zero** por política.

## 8. Principais requisitos faltantes

| requisito | ocorrências |
| --- | --- |
{linhas_req}

## 9. Trilhas de aquisição propostas

| trilha | alvos |
| --- | --- |
{linhas_trk}

Cada trilha explica objetivo, entrada e saída esperada em
`susc_gt03_trilhas_aquisicao.csv`, sempre com `no_network_in_this_milestone=true` e
`no_sar_execution_in_this_milestone=true`.

## 10. Top alvos por prioridade

| target_id | classe | score | região | estado GT02 | trilhas |
| --- | --- | --- | --- | --- | --- |
{linhas_top}

## 11. Alvos bloqueados e por quê

Alvos em `bloqueado_sem_acao_imediata` ou `descartar_como_contexto` estão listados em
`susc_gt03_bloqueios.csv` com o motivo em português: fonte fraca (apenas contexto),
ausência de campos críticos (evento/data/geometria/patch) ou fenômeno misto sem
separação. Nenhum deles é tratado como negativo.

## 12. Petrópolis e eventos mistos

Alvos de Petrópolis ({n_petro} no total) com fenômeno de deslizamento ou misto exigem a
trilha `separar_fenomeno` antes de qualquer promoção e **nunca** recebem prioridade alta
sem essa separação. A separação obrigatória cobre flood, flash_flood, river_flood,
landslide, mixed_flood_landslide e insufficient_type.

## 13. Confirmação explícita dos bloqueios

Este marco **não** treinou modelo, **não** produziu ground truth supervisionado,
**não** criou `score_v7`, **não** alterou o `score_v6`
(`score_v6_changed={str(manifest['score_v6_changed']).lower()}`), **não** executou SAR,
**não** usou internet e **não** promoveu nenhum alvo a `positive_strong`
(`positive_strong_promovidos={manifest['positive_strong_promovidos']}`). Contagens de
controle: `eligible_for_training=true` → {manifest['training_true_count']};
`eligible_for_ground_truth=true` → {manifest['ground_truth_true_count']};
`score_v7_candidate=true` → {manifest['score_v7_candidate_true_count']}.

O REV-P não prevê enchentes operacionalmente: produz análise estrutural review-only com
evidência observacional auditável.

## 14. Próximo passo recomendado

**{manifest['proximo_marco']}**. O maior gargalo do acervo é a ausência de datas de
evento na maioria dos alvos promissores; resolver datas e janelas pré/pós-evento
desbloqueia as trilhas seguintes (geometria oficial, footprint SAR e QA humano).

| fonte | alvos | prioridade alta | bloqueados | requisito dominante |
| --- | --- | --- | --- | --- |
{linhas_src}
"""


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_all() -> dict:
    require_inputs()
    ensure_dir(OUT)
    ensure_dir(SCHEMAS_DIR)

    bundle = build_targets()
    sources = priorizacao_fontes_rows(bundle["targets"])
    priorities = resumo_prioridades_rows(bundle["targets"])

    write_csv(TARGETS_CSV, bundle["targets"])
    write_csv(MISSING_CSV, bundle["missing"])
    write_csv(TRACKS_CSV, bundle["tracks"])
    write_csv(SOURCES_CSV, sources)
    write_csv(PLAN_CSV, bundle["plan"])
    write_csv(PRIORITY_SUMMARY_CSV, priorities)
    write_csv(BLOCKERS_CSV, bundle["blockers"])

    write_json(SCHEMA_TARGET, schema_target_obj())
    write_json(SCHEMA_MISSING, schema_missing_obj())
    write_json(SCHEMA_TRACK, schema_track_obj())
    write_json(SCHEMA_PRIORITY, schema_priority_obj())

    manifest = manifest_obj(bundle)
    write_markdown(REPORT, report_text(manifest, bundle, priorities, sources))
    manifest["artifacts"] = [
        {"path": rel(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in PUBLIC_ARTIFACTS + [SCHEMA_TARGET, SCHEMA_MISSING, SCHEMA_TRACK, SCHEMA_PRIORITY]
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

    targets = _read_csv(TARGETS_CSV)
    missing = _read_csv(MISSING_CSV)
    tracks = _read_csv(TRACKS_CSV)
    blockers = _read_csv(BLOCKERS_CSV)
    priorities = _read_csv(PRIORITY_SUMMARY_CSV)

    missing_by_target = {}
    for m in missing:
        missing_by_target.setdefault(m["target_id"], []).append(m)
    tracks_by_target = {}
    for tr in tracks:
        tracks_by_target.setdefault(tr["target_id"], set()).add(tr["acquisition_track"])
    blockers_by_target = {}
    for b in blockers:
        blockers_by_target.setdefault(b["target_id"], []).append(b)

    for t in targets:
        tid = t["target_id"]
        for fld in ("eligible_for_training", "eligible_for_ground_truth", "trainable", "score_v7_candidate"):
            if t.get(fld) != "false":
                errors.append(f"target:{tid}:{fld}_nao_false")
        if t.get("remains_review_only") != "true":
            errors.append(f"target:{tid}:review_only_nao_true")
        if t.get("assigned_state_gt02") not in STATE_NAMES:
            errors.append(f"target:{tid}:estado_gt02_fora_enum")
        # nada pode ter virado ground truth / positive_strong por causa do GT03
        blob = t.get("rationale_pt", "").lower()
        if "ground truth" in blob and "nao" not in blob and "não" not in blob:
            errors.append(f"target:{tid}:rationale_afirma_ground_truth")
        # priority_score em 0..100
        try:
            sc = int(t["priority_score"])
        except ValueError:
            sc = -1
        if not (0 <= sc <= 100):
            errors.append(f"target:{tid}:priority_score_fora_intervalo:{t['priority_score']}")
        if t.get("priority_class") not in PRIORITY_CLASSES:
            errors.append(f"target:{tid}:priority_class_invalida:{t.get('priority_class')}")
        # can_become=true exige requisitos faltantes explicitados
        if t.get("can_become_positive_strong_after_acquisition") == "true":
            if not missing_by_target.get(tid):
                errors.append(f"target:{tid}:can_become_sem_requisitos")
            if t.get("missing_requirements") in ("", "none"):
                errors.append(f"target:{tid}:can_become_sem_missing_list")
        # Petropolis mixed/landslide nunca alta sem trilha separar_fenomeno
        pheno = t.get("phenomenon_type", "").lower()
        if t.get("region") == "petropolis" and pheno in PETROPOLIS_BLOCKING_PHENOMENA:
            if t.get("priority_class") == "prioridade_alta":
                errors.append(f"target:{tid}:petropolis_mixed_alta")
            if "separar_fenomeno" not in tracks_by_target.get(tid, set()):
                errors.append(f"target:{tid}:petropolis_sem_trilha_separar")
        # fonte fraca nunca alta sem blocker
        src_kind = t.get("source_authority", "").lower()
        geom = t.get("geometry_type", "").lower()
        is_weak = src_kind in WEAK_SOURCE_TYPES or geom in WEAK_SOURCE_TYPES
        if is_weak and t.get("priority_class") == "prioridade_alta" and not blockers_by_target.get(tid):
            errors.append(f"target:{tid}:fonte_fraca_alta_sem_blocker")

    # requisitos: schema de campos e nao vazios
    for m in missing:
        if not m.get("how_to_resolve_pt", "").strip():
            errors.append(f"missing:{m['missing_requirement_id']}:how_to_resolve_vazio")

    # resumo: contagens proibidas sempre 0
    for r in priorities:
        for fld in ("training_allowed_count", "ground_truth_allowed_count", "score_v7_candidate_count"):
            if r.get(fld) != "0":
                errors.append(f"resumo:{r['priority_class']}:{fld}_nao_zero")
        if r.get("priority_class") not in PRIORITY_CLASSES:
            errors.append(f"resumo:priority_class_invalida:{r.get('priority_class')}")

    # texto publico: vocabulario e claim proibido em todos; review-only exigido no relatorio
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
        "GT03 pacote de alvos validado: "
        f"alvos={manifest['alvos_total']} prioridades={manifest['distribuicao_prioridades']} "
        f"podem_virar_forte={manifest['can_become_strong_count']} "
        f"positive_strong_promovidos={manifest['positive_strong_promovidos']} "
        f"proximo={manifest['proximo_marco']} "
        f"score_v6_changed={str(manifest['score_v6_changed']).lower()}"
    )
    return 0


def run_all() -> int:
    manifest = build_all()
    print(
        "GT03 pacote de alvos gerado: "
        f"alvos={manifest['alvos_total']} prioridades={manifest['distribuicao_prioridades']} "
        f"proximo={manifest['proximo_marco']}"
    )
    return 0
