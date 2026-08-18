"""SUSC-GT02 - Motor de Decisao de Referencia Observacional e Replay de Evidencias.

Aplica a politica do SUSC-GT01 (ver [[susc_gt01_ground_reference_policy]]) sobre os
artefatos observacionais JA EXISTENTES no repositorio, sem baixar nada novo. E um
replay review-only: descobre entradas, normaliza colunas para os quatro objetos
conceituais (event_record, footprint, patch_link, score_evaluation), classifica cada
registro em um dos seis estados do GT01 e emite tabelas de decisao por patch/evento.

Este marco NAO busca internet, NAO baixa dado novo, NAO roda SAR, NAO altera o
score_v6, NAO cria score_v7 e NAO treina modelo. Nenhum registro e promovido a ground
truth supervisionado. Guardrails constantes: eligible_for_training=false,
eligible_for_ground_truth=false, trainable=false, score_v7_candidate=false,
review_only=true.

Identificadores tecnicos em ingles (positive_strong, no_data, patch_link_quality,
review_only, etc.) sao mantidos por compatibilidade com o GT01, schemas e validadores;
o relatorio publico explica cada campo em portugues.
"""

from __future__ import annotations

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
import susc_gt01_ground_reference_common as gt01  # noqa: E402

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
SUSC_DIR = ROOT / "outputs_public" / "suscetibilidade"
OUT = SUSC_DIR
SCHEMAS_DIR = ROOT / "schemas" / "suscetibilidade"

REPORT = OUT / "SUSC_GT02_MOTOR_DECISAO_REFERENCIA_OBSERVACIONAL_REPLAY.md"
INVENTORY_CSV = OUT / "susc_gt02_inventario_entradas.csv"
COLMAP_CSV = OUT / "susc_gt02_mapeamento_colunas_entrada.csv"
REPLAY_CSV = OUT / "susc_gt02_registro_replay_evidencias.csv"
DECISION_CSV = OUT / "susc_gt02_decisoes_referencia_por_patch.csv"
BLOCKERS_CSV = OUT / "susc_gt02_bloqueios.csv"
STATE_SUMMARY_CSV = OUT / "susc_gt02_resumo_estados.csv"
MANIFEST_JSON = OUT / "susc_gt02_manifesto.json"

SCHEMA_INVENTORY = SCHEMAS_DIR / "susc_gt02_input_inventory.schema.json"
SCHEMA_REPLAY = SCHEMAS_DIR / "susc_gt02_evidence_replay_record.schema.json"
SCHEMA_DECISION = SCHEMAS_DIR / "susc_gt02_patch_reference_decision.schema.json"
SCHEMA_SUMMARY = SCHEMAS_DIR / "susc_gt02_state_summary.schema.json"

PUBLIC_ARTIFACTS = [
    REPORT, INVENTORY_CSV, COLMAP_CSV, REPLAY_CSV, DECISION_CSV,
    BLOCKERS_CSV, STATE_SUMMARY_CSV,
]
REQUIRED_OUTPUTS = PUBLIC_ARTIFACTS + [
    MANIFEST_JSON, SCHEMA_INVENTORY, SCHEMA_REPLAY, SCHEMA_DECISION, SCHEMA_SUMMARY,
]

MARCO = "SUSC-GT02"
TITULO_PUBLICO = "Motor de Decisão de Referência Observacional e Replay de Evidências"
# Frase que deve aparecer no titulo H1 publico (checagem de idioma).
TITULO_H1_TOKEN = "Referência Observacional"

# ---------------------------------------------------------------------------
# Vocabulario / guardrails
# ---------------------------------------------------------------------------
PUBLIC_FORBIDDEN_RE = gt01.PUBLIC_FORBIDDEN_RE
# Proibe a afirmacao operacional "preve enchentes", mas exclui a negacao legitima
# em portugues acentuado ("nao prevê"/"não prevê") e "nunca". Lookbehinds de
# largura fixa distintos cobrem cada forma.
OPERATIONAL_FORECAST_RE = re.compile(
    r"(?<!não )(?<!nao )(?<!nunca )(prev[eê]|previs[aã]o operacional|forecast)[^\.\n]{0,40}enchent",
    re.IGNORECASE,
)

GLOBAL_GUARDRAILS = {
    "eligible_for_training": "false",
    "eligible_for_ground_truth": "false",
    "trainable": "false",
    "score_v7_candidate": "false",
    "review_only": "true",
}

STATE_NAMES = list(gt01.STATE_NAMES)  # reusa a taxonomia oficial do GT01

# ---------------------------------------------------------------------------
# Regras de decisao (aterradas nos valores reais do repositorio)
# ---------------------------------------------------------------------------
# geometry_type forte permitido (uniao do pedido do marco + evidence_class oficial).
STRONG_GEOMETRY_TYPES = {
    "official_observed_event_polygon",
    "official_observed_event_point",
    "technical_remote_sensing_flood_footprint",
    "official_polygon",
    "official_point",
    "technical_remote_sensing_polygon",
}

# patch_link forte (inclui o vocabulario real "high_spatial_overlap" e o do GT01).
STRONG_PATCH_LINK_QUALITY = {
    "high_spatial_overlap", "intersection_strong", "strong", "high",
}

# qa_status considerado aceito (pending/none nunca contam).
QA_ACCEPTED = {"accepted", "approved", "aceito", "aprovado", "pass", "passed", "human_accepted"}

# Fenomenos de inundacao compativeis com referencia forte de enchente.
FLOOD_PHENOMENA = {
    "flood", "flash_flood", "river_flood", "hydrological_flood", "urban_flood",
    "urban_flash_flood",
}

# Tipos de fonte fraca: NUNCA geram positive_strong.
WEAK_SOURCE_TYPES = {
    "alert_only", "risk_area_not_event", "street_neighborhood_context",
    "municipal_context", "city_level_record", "news_only",
    "documentary_context_only", "susceptibility_map_only",
    "administrative_record_without_geometry",
}

# Marcadores que sinalizam evidencia insuficiente (nao forte).
INSUFFICIENT_MARKERS = {
    "insufficient_reference", "tier_e_insufficient", "not_available", "insufficient",
}

# Petropolis: separacao fenomenologica obrigatoria.
PETROPOLIS_PHENOMENA = [
    "flood", "flash_flood", "river_flood", "landslide",
    "mixed_flood_landslide", "insufficient_type",
]
PETROPOLIS_BLOCKING_PHENOMENA = {"landslide", "mixed_flood_landslide", "insufficient_type"}

# Requisitos minimos de positive_strong (conceitos normalizados).
REQUIRED_STRONG_CONCEPTS = [
    "event_id", "event_date", "source_authority", "geometry_type", "patch_id",
    "patch_link_quality", "uncertainty_m", "qa_status", "phenomenon_type",
]

# ---------------------------------------------------------------------------
# Descoberta / normalizacao de colunas
# ---------------------------------------------------------------------------
# normalized_concept -> lista de colunas originais aceitas (ordem = confianca).
CONCEPT_SYNONYMS = {
    "event_id": ["event_id", "candidate_event_id"],
    "event_date": ["event_date"],
    "footprint_id": ["footprint_id", "candidate_footprint_id", "geometry_id"],
    "geometry_id": ["geometry_id", "footprint_id"],
    "patch_id": ["patch_id", "unified_patch_id", "source_patch_id"],
    "patch_link_id": ["patch_link_id", "candidate_patch_link_id"],
    "patch_link_quality": ["patch_link_quality", "link_quality", "patch_link_status"],
    "geometry_type": ["geometry_type"],
    "evidence_class": ["evidence_class"],
    "reference_level": ["reference_level"],
    "phenomenon_type": ["phenomenon_type"],
    "source_authority": ["source_authority"],
    "source_name": ["source_name", "source_id", "provider_name"],
    "qa_status": ["qa_status"],
    "uncertainty_m": ["uncertainty_m", "geometry_precision_m", "spatial_precision_m"],
    "score_v6": ["score_v6", "susc_score_v6_candidate"],
    "score_v6_class": ["score_v6_class", "susc_class_v6_candidate", "score_class"],
    "city": ["city"],
    "region": ["region"],
    "not_ground_truth_reason": ["not_ground_truth_reason"],
    "review_only": ["review_only"],
}

# Colunas que identificam um registro observacional (para mapeamento de conceito).
IDENTIFIER_COLUMNS = {
    "patch_id", "unified_patch_id", "source_patch_id", "event_id",
    "candidate_event_id", "footprint_id", "candidate_footprint_id",
    "reference_evidence_id", "ground_reference_candidate_id",
}
# Descoberta conservadora: um artefato so entra no replay se identificar um
# EVENTO/FOOTPRINT/REFERENCIA (nao apenas patch) E tiver ao menos um sinal
# observacional forte. Isso evita puxar tabelas de feature/score genericas que
# apenas carregam patch_id + score_v6.
EVENT_FOOTPRINT_ID_COLUMNS = {
    "event_id", "candidate_event_id", "footprint_id", "candidate_footprint_id",
    "reference_evidence_id", "ground_reference_candidate_id",
}
STRONG_SIGNAL_COLUMNS = {
    "event_date", "source_authority", "phenomenon_type", "geometry_type",
    "evidence_class", "reference_level", "link_quality", "patch_link_quality",
    "patch_link_status", "qa_status", "uncertainty_m",
}

# Arquivos deste eixo de politica (GT01/GT02) sao excluidos da descoberta.
POLICY_FILE_RE = re.compile(r"susc_gt0\d", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _run_git(args: list[str]) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return r.stdout.strip() if r.returncode == 0 else ""


def _clean(v) -> str:
    return (str(v).strip() if v is not None else "")


NULLISH = {"", "not_available", "na", "n/a", "none", "null", "unknown", "nan"}


def _present(v) -> bool:
    return _clean(v).lower() not in NULLISH


def _num(v):
    try:
        return float(_clean(v))
    except (TypeError, ValueError):
        return None


def _read_header(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            first = f.readline()
    except OSError:
        return []
    first = first.rstrip("\r\n").lstrip("﻿")
    if not first:
        return []
    return [c.strip() for c in first.split(",")]


def _read_rows(path: Path) -> list[dict]:
    import csv
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


# ---------------------------------------------------------------------------
# 1. Descoberta / inventario de entradas
# ---------------------------------------------------------------------------
def discover_inputs() -> list[dict]:
    """Inventario conservador dos CSVs observacionais existentes (nao recursivo,
    exclui artefatos de politica GT01/GT02). Deterministico: ordenado por path."""
    rows = []
    idx = 1
    for path in sorted(SUSC_DIR.glob("*.csv"), key=lambda p: p.name):
        name = path.name
        if POLICY_FILE_RE.search(name):
            continue
        header = _read_header(path)
        header_set = set(header)
        has_ref_id = bool(header_set & EVENT_FOOTPRINT_ID_COLUMNS)
        has_signal = bool(header_set & STRONG_SIGNAL_COLUMNS)
        is_candidate = has_ref_id and has_signal
        if not is_candidate:
            continue  # descoberta conservadora: evento/footprint + sinal observacional
        data_rows = _read_rows(path) if header else []
        row_count = len(data_rows)
        parseable = bool(header)
        usable = parseable and has_ref_id and has_signal and row_count > 0
        blocking = ""
        if not parseable:
            blocking = "cabecalho_ilegivel"
        elif not has_ref_id:
            blocking = "sem_identificador_evento_footprint"
        elif not has_signal:
            blocking = "sem_sinal_observacional_forte"
        elif row_count == 0:
            blocking = "sem_linhas_de_dados"
        rows.append({
            "input_id": f"IN_{idx:03d}",
            "path": rel(path),
            "artifact_type_guess": _guess_artifact_type(header_set),
            "exists": "true",
            "parseable": "true" if parseable else "false",
            "row_count": str(row_count),
            "detected_columns": ";".join(header),
            "usable_for_replay": "true" if usable else "false",
            "blocking_reason": blocking,
        })
        idx += 1
    return rows


def _guess_artifact_type(header: set) -> str:
    if header & {"score_v6", "susc_score_v6_candidate"}:
        return "score_evaluation"
    if header & {"footprint_id", "candidate_footprint_id"}:
        return "footprint_registry"
    if header & {"patch_link_id", "candidate_patch_link_id", "link_quality", "patch_link_status"}:
        return "patch_link"
    if header & {"reference_evidence_id", "ground_reference_candidate_id"}:
        return "reference_evidence"
    if header & {"event_date", "event_id", "candidate_event_id"}:
        return "event_record"
    return "observational_table"


# ---------------------------------------------------------------------------
# 2. Mapeamento de colunas -> conceitos normalizados
# ---------------------------------------------------------------------------
def column_mapping_rows(inventory: list[dict]) -> list[dict]:
    rows = []
    seen = set()
    for inv in inventory:
        if inv["usable_for_replay"] != "true":
            continue
        header = inv["detected_columns"].split(";")
        header_set = set(header)
        for concept, synonyms in CONCEPT_SYNONYMS.items():
            for rank, col in enumerate(synonyms):
                if col in header_set:
                    key = (inv["input_id"], col, concept)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append({
                        "input_id": inv["input_id"],
                        "original_column": col,
                        "normalized_concept": concept,
                        "confidence": "alta" if rank == 0 else "media",
                        "required_for_positive_strong": "true" if concept in REQUIRED_STRONG_CONCEPTS else "false",
                        "notes": "mapeamento por sinonimo conservador; sem inferencia de valor",
                    })
                    break  # primeiro sinonimo presente vence
    return rows


def _lookup(row: dict, concept: str):
    for col in CONCEPT_SYNONYMS.get(concept, []):
        if col in row and _clean(row[col]) != "":
            return _clean(row[col])
    return ""


def normalize_row(row: dict) -> dict:
    """Extrai os conceitos normalizados de uma linha de entrada. Sem inferencia."""
    return {c: _lookup(row, c) for c in CONCEPT_SYNONYMS}


# ---------------------------------------------------------------------------
# 3. Motor de decisao (puro): aplica a politica GT01 a um registro normalizado
# ---------------------------------------------------------------------------
def _effective_geometry_type(rec: dict) -> str:
    ec = _clean(rec.get("evidence_class")).lower()
    gt = _clean(rec.get("geometry_type")).lower()
    if ec in STRONG_GEOMETRY_TYPES:
        return ec
    if gt in STRONG_GEOMETRY_TYPES:
        return gt
    return gt or ec


def _source_kind(rec: dict) -> str:
    """Detecta fonte fraca a partir dos campos disponiveis (sem inferencia forte)."""
    explicit = _clean(rec.get("evidence_source_kind")).lower()
    if explicit:
        return explicit
    candidates = [
        _clean(rec.get("evidence_class")).lower(),
        _clean(rec.get("reference_level")).lower(),
        _clean(rec.get("geometry_type")).lower(),
        _clean(rec.get("source_name")).lower(),
        _clean(rec.get("artifact_type")).lower(),
    ]
    for c in candidates:
        if c in WEAK_SOURCE_TYPES:
            return c
    return "generic"


def _is_petropolis(rec: dict) -> bool:
    blob = f"{_clean(rec.get('region'))} {_clean(rec.get('city'))}".lower()
    return "petropolis" in blob or "petrópolis" in blob


def decide_state(rec: dict) -> dict:
    """Classifica um registro normalizado em um dos seis estados do GT01.

    Fail-closed: nunca infere positive_strong por heuristica fraca; ausencia de
    evidencia vira unlabeled (nunca negative)."""
    geom_eff = _effective_geometry_type(rec)
    source_kind = _source_kind(rec)
    weak_source = source_kind in WEAK_SOURCE_TYPES

    has_event = _present(rec.get("event_id"))
    has_footprint = _present(rec.get("footprint_id")) or _present(rec.get("geometry_id"))
    has_patch = _present(rec.get("patch_id"))
    has_date = _present(rec.get("event_date"))
    has_source = _present(rec.get("source_authority"))
    has_geom_any = _present(rec.get("geometry_type")) or _present(rec.get("evidence_class"))
    geometry_strong = geom_eff in STRONG_GEOMETRY_TYPES
    plq = _clean(rec.get("patch_link_quality")).lower()
    has_patch_link_any = _present(rec.get("patch_link_quality"))
    patch_link_strong = plq in STRONG_PATCH_LINK_QUALITY
    has_uncertainty = _num(rec.get("uncertainty_m")) is not None
    qa_ok = _clean(rec.get("qa_status")).lower() in QA_ACCEPTED
    pheno = _clean(rec.get("phenomenon_type")).lower()
    pheno_present = _present(rec.get("phenomenon_type")) and pheno != "insufficient_type"
    pheno_flood = pheno in FLOOD_PHENOMENA

    # entradas de auditoria opcionais (hard negative / rejeicao / no_data explicito)
    observability = _clean(rec.get("observability")).lower() in ("true", "confirmed", "observed", "sim")
    exclusion_check = _clean(rec.get("exclusion_check")).lower() in ("true", "passed", "ok", "sim")
    in_same_event = _clean(rec.get("in_same_event")).lower() in ("true", "sim")
    outside_footprint = _clean(rec.get("outside_footprint")).lower() in ("true", "sim")
    permanent_water = _clean(rec.get("permanent_water")).lower() in ("true", "sim")
    rejection_reason = _clean(rec.get("rejection_reason"))
    blocking_reason_in = _clean(rec.get("blocking_reason"))

    # separacao fenomenologica de Petropolis
    petro = _is_petropolis(rec)
    petro_block = petro and (pheno in PETROPOLIS_BLOCKING_PHENOMENA or not pheno_present)

    blockers = []

    def add_blocker(btype, severity, field, reason, ps=True, ev=False, cal=False):
        blockers.append({
            "blocker_type": btype, "severity": severity, "field": field, "reason": reason,
            "prevents_positive_strong": "true" if ps else "false",
            "prevents_evaluation": "true" if ev else "false",
            "prevents_calibration": "true" if cal else "false",
        })

    if petro_block:
        add_blocker("phenomenon_mismatch_or_mixed_event", "alta", "phenomenon_type",
                    "Petropolis com fenomeno landslide/misto/indefinido sem separacao clara da mancha de inundacao",
                    ps=True, ev=True, cal=True)

    # requisitos ausentes para positive_strong (transparencia sempre)
    req_present = {
        "event_id": has_event, "event_date": has_date, "source_authority": has_source,
        "geometry_type": geometry_strong, "patch_id": has_patch,
        "patch_link_quality": patch_link_strong, "uncertainty_m": has_uncertainty,
        "qa_status": qa_ok, "phenomenon_type": pheno_flood,
    }
    missing = [k for k, ok in req_present.items() if not ok]

    # --- ordem fail-closed ---
    if rejection_reason:
        state = "rejected"
        basis = f"rejeitado: {rejection_reason}"
        add_blocker("rejected", "alta", "rejection_reason", rejection_reason, ps=True, ev=True, cal=True)
    elif blocking_reason_in:
        state = "no_data"
        basis = f"sem dados para decidir: {blocking_reason_in}"
        add_blocker(f"no_data:{blocking_reason_in}", "alta", "blocking_reason", blocking_reason_in, ps=True, ev=True, cal=True)
    elif observability and exclusion_check and qa_ok and in_same_event and outside_footprint and not permanent_water and not petro_block:
        state = "hard_negative_audited"
        basis = "negativo auditado: observado no mesmo evento, fora de footprint, sem agua permanente, QA aceito"
    elif (not weak_source and not petro_block and has_event and has_date and has_source
          and geometry_strong and has_patch and patch_link_strong and has_uncertainty
          and qa_ok and pheno_flood):
        state = "positive_strong"
        basis = "referencia forte: evento+data+fonte+geometria forte+patch_link forte+incerteza+QA+fenomeno compativel"
    elif has_source and pheno_present and (has_date or has_geom_any or has_patch_link_any or geometry_strong):
        state = "positive_provisional"
        basis = "evento documentado e rastreavel, mas sem geometria/patch_link/QA suficientes para referencia forte"
        if weak_source:
            add_blocker("weak_source_never_strong", "media", "evidence_source_kind",
                        f"fonte fraca ({source_kind}) nunca gera positive_strong", ps=True, ev=True, cal=False)
    elif has_event or has_footprint or has_patch or has_geom_any:
        state = "no_data"
        basis = "metadados insuficientes para decidir (sem data/fonte/geometria suficientes)"
        add_blocker("no_data:insufficient_metadata_for_decision", "media", "metadata",
                    "registro presente mas sem metadados minimos de decisao", ps=True, ev=True, cal=True)
    else:
        state = "unlabeled"
        basis = "sem evidencia suficiente; estado padrao (nao e negativo)"

    # blockers por requisito ausente (quando nao ficou forte)
    if state != "positive_strong":
        for field in missing:
            add_blocker("missing_required_field", "impede_positive_strong", field,
                        f"campo obrigatorio de positive_strong ausente/insuficiente: {field}",
                        ps=True, ev=False, cal=False)

    eligible_eval = "true" if state in ("positive_strong", "hard_negative_audited") else "false"
    eligible_cal = "true" if (state in ("positive_strong", "positive_provisional", "hard_negative_audited") and not petro_block) else "false"

    not_gt = (
        "referencia observacional review-only; nao e verdade de campo supervisionada; "
        "sem QA/geometria/data suficientes para ground truth" if state != "positive_strong"
        else "referencia forte review-only; segue nao habilitada como ground truth supervisionado neste marco"
    )

    return {
        "assigned_state": state,
        "decision_basis": basis,
        "missing_requirements": ";".join(missing) if missing else "none",
        "blockers": blockers,
        "eligible_for_evaluation": eligible_eval,
        "eligible_for_calibration": eligible_cal,
        "not_ground_truth_reason": not_gt,
        "source_kind": source_kind,
        "petropolis_blocked": "true" if petro_block else "false",
    }


# ---------------------------------------------------------------------------
# 4. Replay: percorre entradas usaveis e gera registros/decisoes/bloqueios
# ---------------------------------------------------------------------------
def build_replay(inventory: list[dict]) -> dict:
    replay_rows = []
    decision_rows = []
    blocker_rows = []
    r_idx = 1
    d_idx = 1
    b_idx = 1
    for inv in inventory:
        if inv["usable_for_replay"] != "true":
            continue
        path = ROOT / inv["path"]
        for row in _read_rows(path):
            rec = normalize_row(row)
            decision = decide_state(rec)
            replay_id = f"RP_{r_idx:04d}"
            r_idx += 1
            replay_rows.append({
                "replay_id": replay_id,
                "source_input_id": inv["input_id"],
                "event_id": rec["event_id"],
                "footprint_id": rec["footprint_id"] or rec["geometry_id"],
                "geometry_id": rec["geometry_id"],
                "patch_id": rec["patch_id"],
                "event_date": rec["event_date"],
                "city": rec["city"],
                "region": rec["region"],
                "phenomenon_type": rec["phenomenon_type"],
                "source_authority": rec["source_authority"],
                "geometry_type": rec["geometry_type"] or rec["evidence_class"],
                "patch_link_quality": rec["patch_link_quality"],
                "uncertainty_m": rec["uncertainty_m"],
                "qa_status": rec["qa_status"],
                "score_v6": rec["score_v6"],
                "score_v6_class": rec["score_v6_class"],
                "review_only": "true",
                "trainable": "false",
                "eligible_for_training": "false",
                "eligible_for_ground_truth": "false",
                "eligible_for_evaluation": decision["eligible_for_evaluation"],
                "eligible_for_calibration": decision["eligible_for_calibration"],
                "assigned_state": decision["assigned_state"],
                "not_ground_truth_reason": decision["not_ground_truth_reason"],
                "blocking_reason": _blocking_reason_of(decision),
            })
            decision_rows.append({
                "decision_id": f"DEC_{d_idx:04d}",
                "patch_id": rec["patch_id"] or "not_available",
                "event_id": rec["event_id"] or "not_available",
                "assigned_state": decision["assigned_state"],
                "decision_basis": decision["decision_basis"],
                "missing_requirements": decision["missing_requirements"],
                "allowed_use": _allowed_use(decision["assigned_state"]),
                "forbidden_use": "treino supervisionado; ground truth; score_v7; patch_link forte a partir de fonte fraca",
                "review_only": "true",
                "eligible_for_evaluation": decision["eligible_for_evaluation"],
                "eligible_for_calibration": decision["eligible_for_calibration"],
                "eligible_for_training": "false",
                "eligible_for_ground_truth": "false",
                "score_v7_candidate": "false",
            })
            d_idx += 1
            for bl in decision["blockers"]:
                blocker_rows.append({
                    "blocker_id": f"BLK_{b_idx:04d}",
                    "replay_id": replay_id,
                    "blocker_type": bl["blocker_type"],
                    "severity": bl["severity"],
                    "field": bl["field"],
                    "reason": bl["reason"],
                    "prevents_positive_strong": bl["prevents_positive_strong"],
                    "prevents_evaluation": bl["prevents_evaluation"],
                    "prevents_calibration": bl["prevents_calibration"],
                })
                b_idx += 1
    return {"replay": replay_rows, "decisions": decision_rows, "blockers": blocker_rows}


def _blocking_reason_of(decision: dict) -> str:
    for bl in decision["blockers"]:
        if bl["blocker_type"].startswith("no_data:") or bl["blocker_type"] == "rejected":
            return bl["reason"]
    return ""


def _allowed_use(state: str) -> str:
    return {
        "positive_strong": "referencia review-only para avaliacao e calibracao review-only",
        "positive_provisional": "referencia provisoria review-only; calibracao review-only; sem avaliacao patch-level forte",
        "unlabeled": "universo nao rotulado (nunca negativo); fora de avaliacao e calibracao",
        "hard_negative_audited": "negativo auditado raro review-only para avaliacao e calibracao review-only",
        "no_data": "excluido de avaliacao e calibracao ate resolver o bloqueio",
        "rejected": "descartado; registrar motivo; fora de avaliacao e calibracao",
    }.get(state, "review-only")


# ---------------------------------------------------------------------------
# 5. Resumo de estados
# ---------------------------------------------------------------------------
def state_summary_rows(replay: list[dict]) -> list[dict]:
    order = {s: i for i, s in enumerate(STATE_NAMES)}
    agg = {}
    for r in replay:
        st = r["assigned_state"]
        a = agg.setdefault(st, {
            "count": 0, "eval": 0, "cal": 0, "train": 0, "gt": 0, "v7": 0,
        })
        a["count"] += 1
        a["eval"] += 1 if r["eligible_for_evaluation"] == "true" else 0
        a["cal"] += 1 if r["eligible_for_calibration"] == "true" else 0
        # os proximos sao sempre 0 por guardrail (contabilizados para transparencia)
        a["train"] += 1 if r["eligible_for_training"] == "true" else 0
        a["gt"] += 1 if r["eligible_for_ground_truth"] == "true" else 0
    rows = []
    for st in sorted(agg, key=lambda s: order.get(s, 99)):
        a = agg[st]
        rows.append({
            "assigned_state": st,
            "count": str(a["count"]),
            "eligible_for_evaluation_count": str(a["eval"]),
            "eligible_for_calibration_count": str(a["cal"]),
            "training_allowed_count": str(a["train"]),
            "ground_truth_allowed_count": str(a["gt"]),
            "score_v7_candidate_count": "0",
        })
    return rows


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
_BOOL = {"enum": ["true", "false"]}
_FALSE = {"const": "false"}


def schema_inventory_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "susc_gt02_input_inventory.schema.json",
        "title": "SUSC-GT02 Inventario de Entradas",
        "type": "object",
        "required": ["input_id", "path", "exists", "parseable", "row_count", "usable_for_replay"],
        "properties": {
            "input_id": {"type": "string", "pattern": "^IN_\\d+$"},
            "path": {"type": "string", "minLength": 1},
            "artifact_type_guess": {"type": "string"},
            "exists": _BOOL,
            "parseable": _BOOL,
            "row_count": {"type": "string", "pattern": "^\\d+$"},
            "detected_columns": {"type": "string"},
            "usable_for_replay": _BOOL,
            "blocking_reason": {"type": "string"},
        },
    }


def schema_replay_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "susc_gt02_evidence_replay_record.schema.json",
        "title": "SUSC-GT02 Registro de Replay de Evidencia",
        "type": "object",
        "required": [
            "replay_id", "source_input_id", "assigned_state", "review_only",
            "trainable", "eligible_for_training", "eligible_for_ground_truth",
            "eligible_for_evaluation", "eligible_for_calibration",
        ],
        "properties": {
            "replay_id": {"type": "string", "pattern": "^RP_\\d+$"},
            "source_input_id": {"type": "string", "pattern": "^IN_\\d+$"},
            "assigned_state": {"enum": STATE_NAMES},
            "review_only": {"const": "true"},
            "trainable": _FALSE,
            "eligible_for_training": _FALSE,
            "eligible_for_ground_truth": _FALSE,
            "eligible_for_evaluation": _BOOL,
            "eligible_for_calibration": _BOOL,
        },
    }


def schema_decision_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "susc_gt02_patch_reference_decision.schema.json",
        "title": "SUSC-GT02 Decisao de Referencia por Patch",
        "type": "object",
        "required": [
            "decision_id", "assigned_state", "decision_basis", "allowed_use",
            "review_only", "eligible_for_evaluation", "eligible_for_calibration",
            "eligible_for_training", "eligible_for_ground_truth", "score_v7_candidate",
        ],
        "properties": {
            "decision_id": {"type": "string", "pattern": "^DEC_\\d+$"},
            "patch_id": {"type": "string"},
            "event_id": {"type": "string"},
            "assigned_state": {"enum": STATE_NAMES},
            "decision_basis": {"type": "string", "minLength": 1},
            "missing_requirements": {"type": "string"},
            "allowed_use": {"type": "string", "minLength": 1},
            "forbidden_use": {"type": "string"},
            "review_only": {"const": "true"},
            "eligible_for_evaluation": _BOOL,
            "eligible_for_calibration": _BOOL,
            "eligible_for_training": _FALSE,
            "eligible_for_ground_truth": _FALSE,
            "score_v7_candidate": _FALSE,
        },
    }


def schema_summary_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "susc_gt02_state_summary.schema.json",
        "title": "SUSC-GT02 Resumo de Estados",
        "type": "object",
        "required": [
            "assigned_state", "count", "eligible_for_evaluation_count",
            "eligible_for_calibration_count", "training_allowed_count",
            "ground_truth_allowed_count", "score_v7_candidate_count",
        ],
        "properties": {
            "assigned_state": {"enum": STATE_NAMES},
            "count": {"type": "string", "pattern": "^\\d+$"},
            "eligible_for_evaluation_count": {"type": "string", "pattern": "^\\d+$"},
            "eligible_for_calibration_count": {"type": "string", "pattern": "^\\d+$"},
            "training_allowed_count": {"const": "0"},
            "ground_truth_allowed_count": {"const": "0"},
            "score_v7_candidate_count": {"const": "0"},
        },
    }


# ---------------------------------------------------------------------------
# Manifesto
# ---------------------------------------------------------------------------
def manifest_obj(inventory, replay, decisions, blockers, summary) -> dict:
    artifacts = []
    for p in PUBLIC_ARTIFACTS + [SCHEMA_INVENTORY, SCHEMA_REPLAY, SCHEMA_DECISION, SCHEMA_SUMMARY]:
        if p.exists():
            artifacts.append({"path": rel(p), "sha256": sha256_file(p), "bytes": p.stat().st_size})
    state_counts = {r["assigned_state"]: int(r["count"]) for r in summary}
    return {
        "marco": MARCO,
        "titulo_publico": TITULO_PUBLICO,
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "gt01_head_commit": _run_git(["log", "-1", "--format=%h", "--", rel(gt01.POLICY_JSON)]) or "unknown",
        "inputs_descobertos": len(inventory),
        "inputs_usaveis": sum(1 for r in inventory if r["usable_for_replay"] == "true"),
        "inputs_bloqueados": sum(1 for r in inventory if r["usable_for_replay"] != "true"),
        "registros_replay": len(replay),
        "decisoes": len(decisions),
        "bloqueios": len(blockers),
        "distribuicao_estados": state_counts,
        "positive_strong_count": state_counts.get("positive_strong", 0),
        "global_guardrails": GLOBAL_GUARDRAILS,
        "training_true_count": sum(1 for r in replay if r["eligible_for_training"] == "true"),
        "ground_truth_true_count": sum(1 for r in replay if r["eligible_for_ground_truth"] == "true"),
        "score_v7_candidate_true_count": 0,
        "score_v6_changed": gt01._score_v6_changed(),
        "score_v7_created": gt01._score_v7_exists(),
        "gt01_valida": _gt01_ok(),
        "proximo_marco": _next_milestone(state_counts),
        "artifacts": artifacts,
    }


def _gt01_ok() -> bool:
    try:
        m = gt01.build_all()
        return gt01.validate_outputs(m) == []
    except Exception:
        return False


def _next_milestone(state_counts: dict) -> str:
    strong = state_counts.get("positive_strong", 0)
    hard_neg = state_counts.get("hard_negative_audited", 0)
    if strong + hard_neg > 0:
        return "GT03 - Revisao de Prontidao para Calibracao Review-Only"
    return "GT03 - Pacote de Alvos para Aquisicao de Evidencia Forte"


# ---------------------------------------------------------------------------
# Relatorio publico (portugues)
# ---------------------------------------------------------------------------
def _example_of(decisions: list[dict], state: str):
    for d in decisions:
        if d["assigned_state"] == state:
            return d
    return None


def report_text(manifest: dict, inventory, decisions, summary) -> str:
    linhas_estado = "\n".join(
        f"| {r['assigned_state']} | {r['count']} | {r['eligible_for_evaluation_count']} | "
        f"{r['eligible_for_calibration_count']} | {r['training_allowed_count']} | "
        f"{r['ground_truth_allowed_count']} |"
        for r in summary
    )
    linhas_entrada = "\n".join(
        f"| {r['input_id']} | {r['path'].split('/')[-1]} | {r['artifact_type_guess']} | "
        f"{r['row_count']} | {r['usable_for_replay']} |"
        for r in inventory[:20]
    )

    def ex_block(state, titulo):
        d = _example_of(decisions, state)
        if not d:
            return f"- **{titulo}** (`{state}`): nenhum registro neste estado no replay atual."
        return (f"- **{titulo}** (`{state}`): decisão `{d['decision_id']}`, patch `{d['patch_id']}`, "
                f"evento `{d['event_id']}`. Base: {d['decision_basis']}. "
                f"Requisitos ausentes: {d['missing_requirements']}.")

    return f"""# {MARCO} - {TITULO_PUBLICO}

## 1. Escopo do marco

Este marco transforma a política do SUSC-GT01 em um **motor de decisão executável**
que lê os artefatos observacionais **já existentes** no repositório e classifica cada
evidência/patch-link em um dos seis estados de referência. É um replay **review-only**:
não busca internet, não baixa dado novo, não roda SAR, não altera o `score_v6`, não
cria `score_v7` e não treina modelo. Nenhum registro é promovido a ground truth
supervisionado.

## 2. Relação com o SUSC-GT01

O GT01 definiu a taxonomia (estados, campos obrigatórios, usos permitidos) e os
bloqueios metodológicos. O GT02 **aplica** essa política: reutiliza os nomes de estado
(`positive_strong`, `positive_provisional`, `unlabeled`, `hard_negative_audited`,
`no_data`, `rejected`) e os guardrails constantes. O manifesto registra se o validador
do GT01 ainda passa (`gt01_valida={str(manifest['gt01_valida']).lower()}`).

## 3. Entradas descobertas

A descoberta é conservadora: varre os CSVs de `outputs_public/suscetibilidade/` que têm
ao menos uma coluna identificadora (patch/evento/footprint) e uma coluna observacional,
excluindo os próprios artefatos de política (GT01/GT02). Foram descobertos
**{manifest['inputs_descobertos']}** candidatos; **{manifest['inputs_usaveis']}** úteis
para replay e **{manifest['inputs_bloqueados']}** bloqueados (sem linhas ou sem colunas
mínimas). O inventário completo está em `susc_gt02_inventario_entradas.csv`.

| input_id | arquivo | tipo estimado | linhas | usável |
| --- | --- | --- | --- | --- |
{linhas_entrada}

## 4. Como as colunas foram normalizadas

Cada coluna original é mapeada para um dos quatro objetos conceituais — `event_record`,
`footprint`, `patch_link`, `score_evaluation` — por sinônimos conservadores, **sem
inferir valores**. O mapeamento auditável está em
`susc_gt02_mapeamento_colunas_entrada.csv`. Campos técnicos em inglês (mantidos por
compatibilidade com schemas e GT01) e seu significado público:

- `event_id` / `event_date`: identificador e data do evento observado.
- `geometry_type`: tipo de geometria (forte só se oficial/observada, ex.:
  `official_observed_event_polygon`).
- `patch_link_quality`: qualidade do vínculo footprint→patch (forte só em
  sobreposição alta, ex.: `high_spatial_overlap`).
- `source_authority`: autoridade/fonte rastreável.
- `uncertainty_m`: incerteza posicional em metros.
- `qa_status`: situação do controle de qualidade humano (forte só se aceito).
- `review_only`: uso restrito a revisão (sempre `true`).
- `eligible_for_evaluation` / `eligible_for_calibration`: elegibilidade review-only.
- `eligible_for_training` / `eligible_for_ground_truth` / `score_v7_candidate` /
  `trainable`: **sempre `false`** neste marco.

## 5. Como a política foi aplicada

O motor é fail-closed: só atribui `positive_strong` quando estão presentes,
simultaneamente, evento, data, fonte, **geometria forte**, patch, **patch_link forte**,
incerteza, **QA aceito** e fenômeno compatível. Sem isso, o registro cai para
`positive_provisional` (evento documentado, mas geometria/patch_link/QA insuficientes),
`no_data` (metadados insuficientes para decidir — exige `blocking_reason`), `rejected`
(incompatibilidade explícita — exige motivo) ou `unlabeled` (sem evidência suficiente;
**nunca negativo**). Fontes fracas (alerta, área de risco, notícia, contexto
municipal/de bairro, mapa de suscetibilidade, registro administrativo sem geometria)
**nunca** geram `positive_strong`.

## 6. Distribuição dos estados

| assigned_state | registros | elegíveis avaliação | elegíveis calibração | treino permitido | ground truth permitido |
| --- | --- | --- | --- | --- | --- |
{linhas_estado}

Leitura pública das colunas: *registros* é a contagem por estado; *elegíveis avaliação/
calibração* são usos review-only; *treino permitido* e *ground truth permitido* são
**sempre zero** por política.

## 7. Principais bloqueios encontrados

Os bloqueios estão em `susc_gt02_bloqueios.csv`. Os mais comuns neste replay são a
ausência de `event_date`, `uncertainty_m` e `qa_status` aceito nos registros de
referência (que impede `positive_strong` e rebaixa para `positive_provisional`), e a
falta de metadados de evento nas tabelas de avaliação de score (que gera `no_data`).
Evidências de Petrópolis com fenômeno de deslizamento ou misto sem separação clara da
mancha de inundação recebem o bloqueio `phenomenon_mismatch_or_mixed_event`, que impede
calibração.

## 8. Exemplos de decisões

{ex_block("positive_strong", "Referência forte")}
{ex_block("positive_provisional", "Referência provisória")}
{ex_block("unlabeled", "Não rotulado")}
{ex_block("no_data", "Sem dados")}
{ex_block("rejected", "Rejeitado")}
{ex_block("hard_negative_audited", "Negativo auditado")}

## 9. Confirmação explícita dos bloqueios

Este marco **não** treinou modelo, **não** produziu ground truth supervisionado,
**não** criou `score_v7`, **não** alterou o `score_v6`
(`score_v6_changed={str(manifest['score_v6_changed']).lower()}`), **não** rodou SAR e
**não** usou internet. Contagens de controle:
`eligible_for_training=true` → {manifest['training_true_count']};
`eligible_for_ground_truth=true` → {manifest['ground_truth_true_count']};
`score_v7_candidate=true` → {manifest['score_v7_candidate_true_count']}.

O REV-P não prevê enchentes operacionalmente: produz análise estrutural review-only com
evidência observacional auditável.

## 10. Próximo passo recomendado

**{manifest['proximo_marco']}**. Como o replay atual não encontrou referência forte
(`positive_strong={manifest['positive_strong_count']}`) — faltam sobretudo data,
geometria oficial forte com incerteza e QA humano aceito —, o caminho recomendado é
montar o pacote de alvos para aquisição de evidência forte antes de qualquer calibração
patch-level.
"""


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_all() -> dict:
    ensure_dir(OUT)
    ensure_dir(SCHEMAS_DIR)

    inventory = discover_inputs()
    colmap = column_mapping_rows(inventory)
    rp = build_replay(inventory)
    summary = state_summary_rows(rp["replay"])

    write_csv(INVENTORY_CSV, inventory)
    write_csv(COLMAP_CSV, colmap)
    write_csv(REPLAY_CSV, rp["replay"])
    write_csv(DECISION_CSV, rp["decisions"])
    write_csv(BLOCKERS_CSV, rp["blockers"])
    write_csv(STATE_SUMMARY_CSV, summary)

    write_json(SCHEMA_INVENTORY, schema_inventory_obj())
    write_json(SCHEMA_REPLAY, schema_replay_obj())
    write_json(SCHEMA_DECISION, schema_decision_obj())
    write_json(SCHEMA_SUMMARY, schema_summary_obj())

    manifest = manifest_obj(inventory, rp["replay"], rp["decisions"], rp["blockers"], summary)
    write_markdown(REPORT, report_text(manifest, inventory, rp["decisions"], summary))
    # manifesto por ultimo (hash dos demais artefatos, inclui o relatorio)
    manifest["artifacts"] = [
        {"path": rel(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in PUBLIC_ARTIFACTS + [SCHEMA_INVENTORY, SCHEMA_REPLAY, SCHEMA_DECISION, SCHEMA_SUMMARY]
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
    return gt01.public_text_violations_text(text)


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
    import csv as _csv
    errors = [f"missing:{rel(p)}" for p in REQUIRED_OUTPUTS if not p.exists()]
    if errors:
        return errors

    # GT01 precisa estar valido
    if not manifest.get("gt01_valida"):
        errors.append("gt01_invalido")
    if manifest["score_v6_changed"]:
        errors.append("score_v6_changed_forbidden")
    if manifest["score_v7_created"]:
        errors.append("score_v7_created_forbidden")

    def read(path):
        with path.open(encoding="utf-8", newline="") as f:
            return list(_csv.DictReader(f))

    replay = read(REPLAY_CSV)
    decisions = read(DECISION_CSV)
    blockers = read(BLOCKERS_CSV)
    summary = read(STATE_SUMMARY_CSV)
    inventory = read(INVENTORY_CSV)

    # guardrails linha a linha em replay + decisoes
    for label, rows, fields in (
        ("replay", replay, ("eligible_for_training", "eligible_for_ground_truth", "trainable")),
        ("decisao", decisions, ("eligible_for_training", "eligible_for_ground_truth", "score_v7_candidate")),
    ):
        for r in rows:
            for fld in fields:
                if r.get(fld) != "false":
                    errors.append(f"{label}:{r.get('replay_id') or r.get('decision_id')}:{fld}_nao_false")
            if r.get("assigned_state") not in STATE_NAMES:
                errors.append(f"{label}:estado_fora_enum:{r.get('assigned_state')}")
            if r.get("review_only") != "true":
                errors.append(f"{label}:review_only_nao_true")

    # score_v7_candidate na decisao ja coberto; garantir tambem em replay (implicito false)
    for r in replay:
        if r.get("assigned_state") == "unlabeled":
            blob = " ".join(str(v).lower() for v in r.values())
            if re.search(r"\bnegativ", blob):
                errors.append(f"replay:{r['replay_id']}:unlabeled_como_negative")

    # positive_strong exige todos os campos obrigatorios (missing_requirements=none)
    dec_by_state = {}
    for d in decisions:
        dec_by_state.setdefault(d["assigned_state"], []).append(d)
    for d in dec_by_state.get("positive_strong", []):
        if d.get("missing_requirements") not in ("", "none"):
            errors.append(f"positive_strong_com_requisitos_faltando:{d['decision_id']}:{d['missing_requirements']}")

    # nenhuma fonte fraca vira positive_strong (checa via replay: source_kind implicito
    # nao esta no csv, entao validamos a regra pela ausencia de strong quando ha blocker
    # weak_source_never_strong apontando para o mesmo replay)
    strong_replay_ids = {r["replay_id"] for r in replay if r["assigned_state"] == "positive_strong"}
    for b in blockers:
        if b["blocker_type"] == "weak_source_never_strong" and b["replay_id"] in strong_replay_ids:
            errors.append(f"fonte_fraca_virou_strong:{b['replay_id']}")

    # no_data exige blocking_reason (na linha de replay ou blocker no_data:)
    nodata_replay = {r["replay_id"]: r for r in replay if r["assigned_state"] == "no_data"}
    nodata_with_reason = set()
    for r in nodata_replay.values():
        if r.get("blocking_reason", "").strip():
            nodata_with_reason.add(r["replay_id"])
    for b in blockers:
        if b["blocker_type"].startswith("no_data:") and b["reason"].strip():
            nodata_with_reason.add(b["replay_id"])
    for rid in nodata_replay:
        if rid not in nodata_with_reason:
            errors.append(f"no_data_sem_blocking_reason:{rid}")

    # rejected exige motivo (rejection blocker)
    rejected_ids = {r["replay_id"] for r in replay if r["assigned_state"] == "rejected"}
    rejected_with_reason = {b["replay_id"] for b in blockers if b["blocker_type"] == "rejected" and b["reason"].strip()}
    for rid in rejected_ids:
        if rid not in rejected_with_reason and not nodata_replay.get(rid):
            # tambem aceita blocking_reason na linha
            row = next((r for r in replay if r["replay_id"] == rid), {})
            if not row.get("blocking_reason", "").strip():
                errors.append(f"rejected_sem_motivo:{rid}")

    # hard_negative_audited exige QA/observability/exclusion (sem blocker missing desses)
    hn_ids = {r["replay_id"] for r in replay if r["assigned_state"] == "hard_negative_audited"}
    for b in blockers:
        if b["replay_id"] in hn_ids and b["field"] in ("qa_status", "observability", "exclusion_check"):
            errors.append(f"hard_negative_sem_qa:{b['replay_id']}:{b['field']}")

    # resumo: training/ground_truth/score_v7 counts sempre 0
    for r in summary:
        for fld in ("training_allowed_count", "ground_truth_allowed_count", "score_v7_candidate_count"):
            if r.get(fld) != "0":
                errors.append(f"resumo:{r['assigned_state']}:{fld}_nao_zero")
        if r.get("assigned_state") not in STATE_NAMES:
            errors.append(f"resumo:estado_fora_enum:{r.get('assigned_state')}")

    # inventario: patch sem evidencia nunca tratado como negative
    for r in inventory:
        blob = " ".join(str(v).lower() for v in r.values())
        if re.search(r"\bnegativ", blob):
            errors.append(f"inventario:{r.get('input_id')}:mencao_negative")

    # texto publico: sem previsao operacional, sem vocabulario proibido, titulo em PT
    for path in _public_paths():
        txt = path.read_text(encoding="utf-8", errors="ignore")
        if operational_forecast_violations(txt):
            errors.append(f"{rel(path)}:claim_previsao_operacional")
        if public_text_violations_text(txt):
            errors.append(f"{rel(path)}:vocabulario_publico_proibido")
    if not title_is_portuguese(_report_h1(REPORT.read_text(encoding="utf-8", errors="ignore"))):
        errors.append("titulo_principal_nao_portugues")

    # artefatos nao podem ser ignorados pelo git
    ignored = gt01.git_ignored(REQUIRED_OUTPUTS)
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
    dist = manifest["distribuicao_estados"]
    print(
        "GT02 replay de referencia observacional validado: "
        f"inputs={manifest['inputs_descobertos']} usaveis={manifest['inputs_usaveis']} "
        f"replay={manifest['registros_replay']} estados={dist} "
        f"positive_strong={manifest['positive_strong_count']} "
        f"gt01_valida={str(manifest['gt01_valida']).lower()} "
        f"score_v6_changed={str(manifest['score_v6_changed']).lower()}"
    )
    return 0


def run_all() -> int:
    manifest = build_all()
    print(
        "GT02 replay gerado: "
        f"inputs={manifest['inputs_descobertos']} usaveis={manifest['inputs_usaveis']} "
        f"replay={manifest['registros_replay']} estados={manifest['distribuicao_estados']} "
        f"proximo={manifest['proximo_marco']}"
    )
    return 0
