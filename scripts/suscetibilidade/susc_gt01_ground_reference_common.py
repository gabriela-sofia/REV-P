"""SUSC-GT01 - Ground Reference Policy & Evidence State Taxonomy.

Formaliza no codigo a politica oficial de referencia observacional por patch do
REV-P. O REV-P NAO trata "ground truth" como rotulo binario absoluto de
alagou/nao alagou. A politica separa quatro objetos distintos:

  1. event_record   -> registro historico/documental do evento
  2. footprint      -> geometria observacional/tecnica (poligono, ponto, bbox,
                       footprint SAR, area de risco, alerta, contexto documental)
  3. patch_link     -> vinculo auditavel entre footprint/evento e patch
  4. score_evaluation -> comparacao review-only entre score/features e evidencia

e define seis estados de referencia observacional por patch. Esta base e
puramente metodologica: nao busca internet, nao baixa raster, nao roda SAR, nao
cria embeddings, nao altera score_v6, nao cria score_v7 e nao treina modelo.

Guardrails globais deste marco (constantes, nao derivadas):
  eligible_for_training      = false  (sempre)
  eligible_for_ground_truth  = false  (sempre)
  score_v7_candidate         = false  (sempre)
  trainable                  = false  (sempre)
  review_only                = true   (sempre)
  eligible_for_evaluation    -> true apenas para positive_strong,
                                hard_negative_audited (ou aceitos por politica)
  eligible_for_calibration   -> true apenas em regime review-only
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
    read_csv,
    rel,
    sha256_file,
    write_csv,
    write_json,
    write_markdown,
)

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS_DIR = ROOT / "schemas" / "suscetibilidade"

REPORT = OUT / "SUSC_GT01_GROUND_REFERENCE_POLICY.md"
TAXONOMY_CSV = OUT / "susc_gt01_evidence_state_taxonomy.csv"
POLICY_JSON = OUT / "susc_gt01_ground_reference_policy.json"
TRANSITIONS_CSV = OUT / "susc_gt01_transition_rules.csv"
NOT_GT_CSV = OUT / "susc_gt01_not_ground_truth_reasons.csv"
DECISION_CSV = OUT / "susc_gt01_review_only_decision_table.csv"
MANIFEST_JSON = OUT / "susc_gt01_manifest.json"

SCHEMA_POLICY = SCHEMAS_DIR / "susc_gt01_ground_reference_policy.schema.json"
SCHEMA_STATE = SCHEMAS_DIR / "susc_gt01_evidence_state.schema.json"
SCHEMA_DECISION = SCHEMAS_DIR / "susc_gt01_patch_reference_decision.schema.json"

# Artefatos publicos (sem o manifesto, que contem os hashes dos demais)
PUBLIC_ARTIFACTS = [
    REPORT, TAXONOMY_CSV, POLICY_JSON, TRANSITIONS_CSV, NOT_GT_CSV, DECISION_CSV,
]
REQUIRED_OUTPUTS = PUBLIC_ARTIFACTS + [
    MANIFEST_JSON, SCHEMA_POLICY, SCHEMA_STATE, SCHEMA_DECISION,
]

# ---------------------------------------------------------------------------
# Guardrails / vocabulario
# ---------------------------------------------------------------------------
PUBLIC_FORBIDDEN_RE = re.compile(r"\b(?:agentic|agente|codex|llm|ia)\b", re.IGNORECASE)
# REV-P nao "preve enchentes" operacionalmente. Detecta a afirmacao operacional,
# permitindo a negacao explicita ("nao preve enchentes").
OPERATIONAL_FORECAST_RE = re.compile(
    r"(?<!n[ao]o )(?<!nunca )(prev[eê]|previs[aã]o operacional|forecast)[^\.\n]{0,40}enchent",
    re.IGNORECASE,
)

MARCO = "SUSC-GT01"
TITULO_PUBLICO = "Ground Reference Policy & Evidence State Taxonomy"
PROXIMO_MARCO = "calibracao candidata review-only + canarios observacionais SAR + QA humano"

# Guardrails constantes deste marco (nunca derivados).
GLOBAL_GUARDRAILS = {
    "eligible_for_training": "false",
    "eligible_for_ground_truth": "false",
    "score_v7_candidate": "false",
    "trainable": "false",
    "review_only": "true",
    "supervised_training_allowed": "false",
    "score_v6_altered": "false",
    "score_v7_created": "false",
    "network_access": "false",
    "raster_download": "false",
}

# Motivos permitidos de no_data (blocking_reason obrigatorio).
NO_DATA_BLOCKING_REASONS = [
    "cloud", "sar_shadow", "layover", "speckle", "vegetation",
    "urban_ambiguity", "no_image", "bad_source", "insufficient_geometry",
]

# Forcas de patch_link (ordenadas da mais fraca para a mais forte).
PATCH_LINK_STRENGTHS = [
    "none", "insufficient", "region_period", "buffer", "bbox_overlap",
    "intersection_strong",
]
STRONG_PATCH_LINK = "intersection_strong"

# Fenomenos que Petropolis deve separar explicitamente.
PETROPOLIS_PHENOMENA = ["flood", "flash_flood", "landslide", "mixed_flood_landslide"]

# ---------------------------------------------------------------------------
# Fonte de verdade: os seis estados
# ---------------------------------------------------------------------------
# Cada estado declara: definicao, uso permitido, elegibilidades e campos exigidos.
STATES = [
    {
        "state": "positive_strong",
        "is_default": "false",
        "is_negative": "false",
        "definition": (
            "Evento real com data compativel, fonte rastreavel, geometria forte, "
            "patch_link claro, incerteza documentada e QA aceito."
        ),
        "allowed_use": (
            "referencia observacional review-only; elegivel para avaliacao e "
            "calibracao review-only"
        ),
        "eligible_for_evaluation": "true",
        "eligible_for_calibration": "true",
        "required_fields": [
            "event_date", "geometry_type", "source_authority",
            "patch_link_quality", "uncertainty_m", "qa_status",
        ],
        "blocking_reason_required": "false",
    },
    {
        "state": "positive_provisional",
        "is_default": "false",
        "is_negative": "false",
        "definition": (
            "Evento bem documentado e plausivel, mas sem geometria suficiente "
            "para patch-level forte."
        ),
        "allowed_use": (
            "referencia review-only provisoria; calibracao review-only permitida; "
            "avaliacao patch-level nao habilitada por falta de geometria forte"
        ),
        "eligible_for_evaluation": "false",
        "eligible_for_calibration": "true",
        "required_fields": [
            "event_date", "source_authority", "documentation_quality",
        ],
        "blocking_reason_required": "false",
    },
    {
        "state": "unlabeled",
        "is_default": "true",
        "is_negative": "false",
        "definition": (
            "Sem evidencia suficiente. Estado padrao de todo patch. NAO e negativo: "
            "ausencia de registro nao e negativo verdadeiro."
        ),
        "allowed_use": (
            "universo nao rotulado (unlabeled_background); nunca tratado como "
            "negativo; nao entra em avaliacao nem calibracao"
        ),
        "eligible_for_evaluation": "false",
        "eligible_for_calibration": "false",
        "required_fields": [],
        "blocking_reason_required": "false",
    },
    {
        "state": "hard_negative_audited",
        "is_default": "false",
        "is_negative": "true",
        "definition": (
            "Caso raro: patch observado no mesmo evento, fora de footprint, sem "
            "exclusoes relevantes, com comportamento seco e QA aceito."
        ),
        "allowed_use": (
            "negativo auditado raro review-only; elegivel para avaliacao e "
            "calibracao review-only; exige auditoria explicita"
        ),
        "eligible_for_evaluation": "true",
        "eligible_for_calibration": "true",
        "required_fields": [
            "observability", "exclusion_check", "qa_status",
        ],
        "blocking_reason_required": "false",
    },
    {
        "state": "no_data",
        "is_default": "false",
        "is_negative": "false",
        "definition": (
            "Impossivel decidir por nuvem, sombra SAR, layover, speckle, vegetacao, "
            "ambiguidade urbana, ausencia de imagem, fonte ruim ou geometria "
            "insuficiente."
        ),
        "allowed_use": (
            "excluido da avaliacao e da calibracao; exige blocking_reason; nunca "
            "vira positivo nem negativo"
        ),
        "eligible_for_evaluation": "false",
        "eligible_for_calibration": "false",
        "required_fields": ["blocking_reason"],
        "blocking_reason_required": "true",
    },
    {
        "state": "rejected",
        "is_default": "false",
        "is_negative": "false",
        "definition": (
            "Evidencia rejeitada por incompatibilidade de fenomeno, temporalidade, "
            "localizacao, fonte ou duplicidade."
        ),
        "allowed_use": (
            "evidencia descartada; fora de avaliacao e calibracao; registrada com "
            "rejection_reason auditavel"
        ),
        "eligible_for_evaluation": "false",
        "eligible_for_calibration": "false",
        "required_fields": ["rejection_reason"],
        "blocking_reason_required": "false",
    },
]

STATE_NAMES = [s["state"] for s in STATES]
DEFAULT_STATE = "unlabeled"
# Estados explicitamente aceitos por politica para avaliacao review-only.
EVALUATION_ELIGIBLE_STATES = {"positive_strong", "hard_negative_audited"}

# Campos exigidos por estado (fonte para o validador do proprio codigo).
REQUIRED_FIELDS_BY_STATE = {s["state"]: list(s["required_fields"]) for s in STATES}

# ---------------------------------------------------------------------------
# Regras de transicao (permitidas e proibidas)
# ---------------------------------------------------------------------------
# from_state, trigger, to_state, guard, allowed, review_only
TRANSITIONS = [
    ("any", "criacao_do_patch", "unlabeled",
     "todo patch nasce unlabeled; nunca negative", "true", "true"),
    ("unlabeled", "evento_documentado_forte + geometria_forte + patch_link_forte + qa_aceito",
     "positive_strong",
     "exige event_date, geometry_type, source_authority, patch_link_quality, uncertainty_m, qa_status",
     "true", "true"),
    ("unlabeled", "evento_bem_documentado_sem_geometria_forte", "positive_provisional",
     "documentacao forte mas geometria insuficiente para patch-level forte", "true", "true"),
    ("positive_provisional", "geometria_oficial_adquirida + patch_link_forte + qa_aceito",
     "positive_strong", "promocao apenas com geometria forte e QA", "true", "true"),
    ("unlabeled", "mesmo_evento + fora_de_footprint + sem_exclusoes + seco + qa_aceito",
     "hard_negative_audited",
     "negativo auditado raro; exige observability, exclusion_check, qa_status", "true", "true"),
    ("unlabeled", "cloud|sar_shadow|layover|speckle|vegetation|urban_ambiguity|no_image|bad_source|insufficient_geometry",
     "no_data", "exige blocking_reason", "true", "true"),
    ("any", "incompatibilidade_fenomeno|temporal|localizacao|fonte|duplicidade", "rejected",
     "exige rejection_reason auditavel", "true", "true"),
    # --- transicoes PROIBIDAS (allowed=false) ---
    ("alert_only", "alerta_vira_evento_observado", "positive_strong",
     "alerta nao pode virar evento observado nem patch_link forte", "false", "true"),
    ("risk_area_not_event", "area_de_risco_vira_evento", "positive_strong",
     "area de risco nao e evento ocorrido", "false", "true"),
    ("news_report", "noticia_vira_patch_link_forte", "positive_strong",
     "noticia nao vira patch_link forte", "false", "true"),
    ("municipality_or_neighborhood_affected", "municipio_bairro_vira_patch_atingido", "positive_strong",
     "municipio/bairro atingido nao vira patch atingido", "false", "true"),
    ("post_event_footprint", "footprint_pos_evento_vira_feature_pre_evento", "positive_strong",
     "footprint pos-evento nao vira feature pre-evento", "false", "true"),
    ("unlabeled", "ausencia_de_registro_vira_negativo", "hard_negative_audited",
     "ausencia de registro nao e negativo; hard_negative exige patch observado no evento + QA",
     "false", "true"),
]

# ---------------------------------------------------------------------------
# Razoes pelas quais evidencia NAO e ground truth supervisionado
# ---------------------------------------------------------------------------
# reason_id, category, statement, applies_to
NOT_GROUND_TRUTH_REASONS = [
    ("NGT-01", "conceito", "suscetibilidade nao e evento observado", "score_v6/features"),
    ("NGT-02", "conceito", "score nao e ground truth", "score_v6/score_v7"),
    ("NGT-03", "fonte", "noticia/alerta/area de risco nao e patch positivo", "event_record/footprint"),
    ("NGT-04", "logica", "ausencia de registro nao e negativo verdadeiro", "unlabeled"),
    ("NGT-05", "sensor", "Sentinel-1/SAR pode gerar/confirmar footprint mas nao vira verdade automatica", "footprint"),
    ("NGT-06", "fonte", "area de risco nao pode virar evento ocorrido", "risk_area_not_event"),
    ("NGT-07", "geometria", "municipio/bairro atingido nao vira patch atingido", "patch_link"),
    ("NGT-08", "temporal", "footprint pos-evento nao vira feature pre-evento", "footprint/features"),
    ("NGT-09", "qualidade", "evidencia observacional so e referencia review-only com qualidade suficiente", "score_evaluation"),
    ("NGT-10", "fonte", "alerta nao pode virar evento observado", "alert_only"),
    ("NGT-11", "escopo", "REV-P nao preve enchentes operacionalmente; produz analise estrutural review-only", "projeto"),
    ("NGT-12", "fenomeno", "Petropolis exige separacao entre flood, flash_flood, landslide e mixed_flood_landslide", "event_record"),
]

# ---------------------------------------------------------------------------
# Tabela de decisao review-only por tipo de evidencia
# ---------------------------------------------------------------------------
# input_evidence_type, can_promote_to, max_patch_link_strength,
# eligible_for_evaluation, eligible_for_calibration, decision, rationale
DECISION_ROWS = [
    ("official_polygon", "positive_strong", "intersection_strong", "true", "true",
     "promocao_forte_permitida_com_qa",
     "geometria oficial forte permite patch_link por intersecao quando QA aceito"),
    ("official_point", "positive_strong", "buffer", "false", "true",
     "provisorio_ou_forte_conforme_buffer_e_qa",
     "ponto oficial exige buffer/incerteza; forte apenas com QA e patch_link claro"),
    ("official_bbox", "positive_provisional", "bbox_overlap", "false", "true",
     "provisorio_sem_geometria_fina",
     "bbox nao entrega geometria fina para patch-level forte"),
    ("sar_footprint", "positive_provisional", "bbox_overlap", "false", "true",
     "tecnico_review_only_nunca_verdade_automatica",
     "footprint SAR e tecnico/pos-evento; nao vira verdade automatica nem feature pre-evento"),
    ("risk_area", "unlabeled", "insufficient", "false", "false",
     "nao_promove_area_de_risco_nao_e_evento",
     "area de risco nao e evento ocorrido; nunca patch_link forte"),
    ("alert_only", "unlabeled", "insufficient", "false", "false",
     "nao_promove_alerta_nao_e_evento",
     "alerta nao vira evento observado nem patch_link forte"),
    ("news_report", "positive_provisional", "region_period", "false", "false",
     "documental_fraco_nao_forte",
     "noticia sustenta contexto documental; nunca patch_link forte"),
    ("municipality_affected", "unlabeled", "region_period", "false", "false",
     "nao_promove_municipio_nao_e_patch",
     "municipio atingido nao vira patch atingido"),
    ("neighborhood_affected", "unlabeled", "region_period", "false", "false",
     "nao_promove_bairro_nao_e_patch",
     "bairro atingido nao vira patch atingido"),
    ("documentary_context", "positive_provisional", "region_period", "false", "false",
     "contexto_documental_review_only",
     "contexto documental sustenta provisorio; nunca forte sozinho"),
    ("post_event_footprint", "no_data", "none", "false", "false",
     "nao_vira_feature_pre_evento",
     "footprint pos-evento nao vira feature pre-evento; excluido como feature"),
]


# ---------------------------------------------------------------------------
# Utilidades git / numeros
# ---------------------------------------------------------------------------
def _run_git(args: list[str]) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return r.stdout.strip() if r.returncode == 0 else ""


def git_ignored(paths: list[Path]) -> set[str]:
    if not paths:
        return set()
    rels = [rel(p) for p in paths]
    r = subprocess.run(
        ["git", "check-ignore", "-z", "--stdin"], cwd=ROOT,
        input=("\0".join(rels) + "\0").encode("utf-8"), capture_output=True, check=False,
    )
    out = r.stdout.decode("utf-8", errors="ignore")
    return {piece.replace("\\", "/") for piece in out.split("\0") if piece.strip()}


def _score_v6_changed() -> bool:
    score_v6 = ROOT / "datasets" / "suscetibilidade" / "susc_score_v6_candidate_by_patch_v1.csv"
    return bool(_run_git(["diff", "--name-only", "--", rel(score_v6)]))


def _score_v7_exists() -> bool:
    return (ROOT / "datasets" / "suscetibilidade" / "susc_score_v7_candidate_by_patch_v1.csv").exists()


# ---------------------------------------------------------------------------
# Geradores de linhas
# ---------------------------------------------------------------------------
def taxonomy_rows() -> list[dict]:
    rows = []
    for s in STATES:
        rows.append({
            "state": s["state"],
            "is_default": s["is_default"],
            "is_negative": s["is_negative"],
            "definition": s["definition"],
            "allowed_use": s["allowed_use"],
            "required_fields": ";".join(s["required_fields"]) or "none",
            "blocking_reason_required": s["blocking_reason_required"],
            "eligible_for_evaluation": s["eligible_for_evaluation"],
            "eligible_for_calibration": s["eligible_for_calibration"],
            # guardrails constantes por estado
            "eligible_for_training": GLOBAL_GUARDRAILS["eligible_for_training"],
            "eligible_for_ground_truth": GLOBAL_GUARDRAILS["eligible_for_ground_truth"],
            "score_v7_candidate": GLOBAL_GUARDRAILS["score_v7_candidate"],
            "trainable": GLOBAL_GUARDRAILS["trainable"],
            "review_only": GLOBAL_GUARDRAILS["review_only"],
        })
    return rows


def transition_rows() -> list[dict]:
    return [{
        "from_state": f, "trigger": t, "to_state": to, "guard": g,
        "allowed": allowed, "review_only": ro,
    } for f, t, to, g, allowed, ro in TRANSITIONS]


def not_ground_truth_rows() -> list[dict]:
    return [{
        "reason_id": rid, "category": cat, "statement": st, "applies_to": ap,
        "review_only": "true",
    } for rid, cat, st, ap in NOT_GROUND_TRUTH_REASONS]


def decision_rows() -> list[dict]:
    return [{
        "input_evidence_type": ev, "can_promote_to": promo,
        "max_patch_link_strength": strength,
        "eligible_for_evaluation": eval_ok, "eligible_for_calibration": calib_ok,
        "score_v7_candidate": GLOBAL_GUARDRAILS["score_v7_candidate"],
        "eligible_for_training": GLOBAL_GUARDRAILS["eligible_for_training"],
        "eligible_for_ground_truth": GLOBAL_GUARDRAILS["eligible_for_ground_truth"],
        "review_only": GLOBAL_GUARDRAILS["review_only"],
        "decision": dec, "rationale": rat,
    } for ev, promo, strength, eval_ok, calib_ok, dec, rat in DECISION_ROWS]


# ---------------------------------------------------------------------------
# Objetos JSON: policy, schemas, manifest
# ---------------------------------------------------------------------------
def policy_obj() -> dict:
    return {
        "marco": MARCO,
        "titulo_publico": TITULO_PUBLICO,
        "descricao": (
            "Politica oficial de referencia observacional por patch. Separa "
            "event_record, footprint, patch_link e score_evaluation. Estado padrao "
            "unlabeled (nao negativo). Base metodologica review-only; sem treino, "
            "sem ground truth supervisionado, sem score_v7."
        ),
        "objects": {
            "event_record": "registro historico/documental do evento (data, cidade, fenomeno, fonte, autoridade, precisao temporal)",
            "footprint": "geometria observacional/tecnica (poligono, ponto, bbox, footprint SAR, area de risco, alerta, contexto documental)",
            "patch_link": "vinculo auditavel footprint/evento->patch (intersecao, bbox overlap, buffer, mesma regiao/periodo ou insuficiente)",
            "score_evaluation": "comparacao review-only entre score/features do patch e a evidencia; nao treina, nao altera score, nao cria score_v7",
        },
        "global_guardrails": GLOBAL_GUARDRAILS,
        "default_state": DEFAULT_STATE,
        "states": [
            {
                "state": s["state"],
                "definition": s["definition"],
                "allowed_use": s["allowed_use"],
                "is_default": s["is_default"] == "true",
                "is_negative": s["is_negative"] == "true",
                "required_fields": s["required_fields"],
                "blocking_reason_required": s["blocking_reason_required"] == "true",
                "eligible_for_evaluation": s["eligible_for_evaluation"] == "true",
                "eligible_for_calibration": s["eligible_for_calibration"] == "true",
                "eligible_for_training": False,
                "eligible_for_ground_truth": False,
                "score_v7_candidate": False,
                "trainable": False,
                "review_only": True,
            }
            for s in STATES
        ],
        "evaluation_eligible_states": sorted(EVALUATION_ELIGIBLE_STATES),
        "no_data_blocking_reasons": NO_DATA_BLOCKING_REASONS,
        "patch_link_strengths": PATCH_LINK_STRENGTHS,
        "strong_patch_link": STRONG_PATCH_LINK,
        "non_promotable_to_strong_patch_link": [
            "alert_only", "risk_area", "risk_area_not_event", "news_report",
            "municipality_affected", "neighborhood_affected", "post_event_footprint",
        ],
        "petropolis_phenomenon_separation": {
            "required": True,
            "phenomena": PETROPOLIS_PHENOMENA,
            "rule": "Petropolis exige separacao explicita entre flood, flash_flood, landslide e mixed_flood_landslide antes de qualquer promocao",
        },
        "operational_forecast_claim": {
            "revp_operational_flood_forecast": False,
            "statement": "REV-P nao preve enchentes operacionalmente; produz analise estrutural review-only",
        },
        "prepares_next_milestones": [
            "calibracao candidata review-only",
            "canarios observacionais SAR",
            "QA humano",
            "matriz multimodal escalavel",
            "benchmark futuro (nao autorizado neste marco)",
        ],
    }


def schema_policy_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "susc_gt01_ground_reference_policy.schema.json",
        "title": "SUSC-GT01 Ground Reference Policy",
        "type": "object",
        "required": [
            "marco", "default_state", "global_guardrails", "states",
            "petropolis_phenomenon_separation", "operational_forecast_claim",
        ],
        "properties": {
            "marco": {"const": MARCO},
            "default_state": {"const": "unlabeled"},
            "global_guardrails": {
                "type": "object",
                "required": [
                    "eligible_for_training", "eligible_for_ground_truth",
                    "score_v7_candidate", "trainable", "review_only",
                ],
                "properties": {
                    "eligible_for_training": {"const": "false"},
                    "eligible_for_ground_truth": {"const": "false"},
                    "score_v7_candidate": {"const": "false"},
                    "trainable": {"const": "false"},
                    "review_only": {"const": "true"},
                },
            },
            "states": {
                "type": "array",
                "minItems": 6,
                "items": {"$ref": "susc_gt01_evidence_state.schema.json"},
            },
            "operational_forecast_claim": {
                "type": "object",
                "properties": {
                    "revp_operational_flood_forecast": {"const": False},
                },
            },
        },
    }


def schema_state_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "susc_gt01_evidence_state.schema.json",
        "title": "SUSC-GT01 Evidence State",
        "type": "object",
        "required": [
            "state", "definition", "allowed_use", "is_default", "is_negative",
            "required_fields", "eligible_for_evaluation", "eligible_for_calibration",
            "eligible_for_training", "eligible_for_ground_truth",
            "score_v7_candidate", "trainable", "review_only",
        ],
        "properties": {
            "state": {"enum": STATE_NAMES},
            "definition": {"type": "string", "minLength": 1},
            "allowed_use": {"type": "string", "minLength": 1},
            "is_default": {"type": "boolean"},
            "is_negative": {"type": "boolean"},
            "required_fields": {"type": "array", "items": {"type": "string"}},
            "eligible_for_evaluation": {"type": "boolean"},
            "eligible_for_calibration": {"type": "boolean"},
            "eligible_for_training": {"const": False},
            "eligible_for_ground_truth": {"const": False},
            "score_v7_candidate": {"const": False},
            "trainable": {"const": False},
            "review_only": {"const": True},
        },
        "allOf": [
            {
                "if": {"properties": {"state": {"const": "positive_strong"}}},
                "then": {
                    "properties": {
                        "required_fields": {
                            "allOf": [
                                {"contains": {"const": "event_date"}},
                                {"contains": {"const": "geometry_type"}},
                                {"contains": {"const": "source_authority"}},
                                {"contains": {"const": "patch_link_quality"}},
                                {"contains": {"const": "uncertainty_m"}},
                                {"contains": {"const": "qa_status"}},
                            ]
                        }
                    }
                },
            },
            {
                "if": {"properties": {"state": {"const": "hard_negative_audited"}}},
                "then": {
                    "properties": {
                        "required_fields": {
                            "allOf": [
                                {"contains": {"const": "observability"}},
                                {"contains": {"const": "exclusion_check"}},
                                {"contains": {"const": "qa_status"}},
                            ]
                        }
                    }
                },
            },
            {
                "if": {"properties": {"state": {"const": "no_data"}}},
                "then": {
                    "properties": {
                        "required_fields": {"contains": {"const": "blocking_reason"}}
                    }
                },
            },
        ],
    }


def schema_decision_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "susc_gt01_patch_reference_decision.schema.json",
        "title": "SUSC-GT01 Patch Reference Decision",
        "type": "object",
        "required": [
            "input_evidence_type", "can_promote_to", "max_patch_link_strength",
            "eligible_for_evaluation", "eligible_for_calibration",
            "score_v7_candidate", "review_only", "decision",
        ],
        "properties": {
            "input_evidence_type": {"type": "string", "minLength": 1},
            "can_promote_to": {"enum": STATE_NAMES},
            "max_patch_link_strength": {"enum": PATCH_LINK_STRENGTHS},
            "eligible_for_evaluation": {"enum": ["true", "false"]},
            "eligible_for_calibration": {"enum": ["true", "false"]},
            "score_v7_candidate": {"const": "false"},
            "eligible_for_training": {"const": "false"},
            "eligible_for_ground_truth": {"const": "false"},
            "review_only": {"const": "true"},
            "decision": {"type": "string", "minLength": 1},
            "rationale": {"type": "string", "minLength": 1},
        },
        "not_promotable_to_strong_patch_link": [
            "alert_only", "risk_area", "risk_area_not_event", "news_report",
            "municipality_affected", "neighborhood_affected", "post_event_footprint",
        ],
    }


def manifest_obj() -> dict:
    artifacts = []
    for p in PUBLIC_ARTIFACTS + [SCHEMA_POLICY, SCHEMA_STATE, SCHEMA_DECISION]:
        if p.exists():
            artifacts.append({
                "path": rel(p),
                "sha256": sha256_file(p),
                "bytes": p.stat().st_size,
            })
    return {
        "marco": MARCO,
        "titulo_publico": TITULO_PUBLICO,
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "states": STATE_NAMES,
        "default_state": DEFAULT_STATE,
        "global_guardrails": GLOBAL_GUARDRAILS,
        "score_v6_changed": _score_v6_changed(),
        "score_v7_created": _score_v7_exists(),
        "artifacts": artifacts,
        "proximo_marco": PROXIMO_MARCO,
    }


# ---------------------------------------------------------------------------
# Relatorio principal
# ---------------------------------------------------------------------------
def report_text() -> str:
    tax = taxonomy_rows()
    linhas_estados = "\n".join(
        f"| {r['state']} | {'padrao' if r['is_default'] == 'true' else '-'} | "
        f"{'sim' if r['is_negative'] == 'true' else 'nao'} | {r['eligible_for_evaluation']} | "
        f"{r['eligible_for_calibration']} | {r['required_fields']} |"
        for r in tax
    )
    linhas_dec = "\n".join(
        f"| {r['input_evidence_type']} | {r['can_promote_to']} | {r['max_patch_link_strength']} | "
        f"{r['eligible_for_evaluation']} | {r['decision']} |"
        for r in decision_rows()
    )
    return f"""# {MARCO} - {TITULO_PUBLICO}

Politica oficial e validavel de referencia observacional por patch do REV-P. Este
marco e puramente metodologico: nao busca internet, nao baixa raster, nao roda
SAR, nao cria embeddings, nao altera o score_v6, nao cria score_v7 e nao treina
modelo. Ele formaliza no codigo a taxonomia de estados de evidencia e as regras
que impedem transformar suscetibilidade, score, alerta, area de risco ou noticia
em rotulo de campo.

## 1. Por que o REV-P nao usa ground truth binario agora

O REV-P nao trata "ground truth" como rotulo binario absoluto de alagou/nao
alagou. Rotular patches como positivo/negativo exigiria geometria de ocorrencia
confiavel, separacao de fenomeno, janela temporal pre-evento e QA — que hoje nao
existem de forma consistente. Rotular sem isso produziria vazamento, negativos
falsos e overclaim. Enquanto essa base nao esta pronta, a referencia observacional
so pode ser usada como **evidencia review-only**, nunca como verdade supervisionada.

## 2. Quatro objetos distintos

A arquitetura correta separa quatro objetos, que nunca se confundem:

- **event_record** — registro historico/documental do evento: data, cidade, tipo
  de fenomeno, fonte, autoridade, precisao temporal.
- **footprint** — geometria observacional ou tecnica: poligono oficial, ponto
  oficial, bbox, footprint SAR, area de risco, alerta, contexto documental.
- **patch_link** — vinculo auditavel entre footprint/evento e patch: intersecao,
  bbox overlap, buffer, mesma regiao/periodo ou insuficiente.
- **score_evaluation** — comparacao review-only entre score/features do patch e a
  evidencia observacional. Nao treina modelo, nao altera o score oficial, nao cria
  score v7.

E as distincoes conceituais que a politica protege:

- **suscetibilidade** e uma propriedade estrutural do terreno; **risco** combina
  suscetibilidade com exposicao/vulnerabilidade; **evento observado** e um fato
  datado; **footprint** e geometria; **patch_link** e um vinculo auditavel;
  **reference evidence** e evidencia review-only de qualidade suficiente; e
  **ground truth supervisionado** e rotulo de treino — que este marco nao produz.

## 3. Estados de referencia observacional

| Estado | Default | Negativo | eval review-only | calib review-only | Campos exigidos |
| --- | --- | --- | --- | --- | --- |
{linhas_estados}

O detalhe completo esta em `susc_gt01_evidence_state_taxonomy.csv` e
`susc_gt01_ground_reference_policy.json`.

## 4. Por que o estado padrao e unlabeled

Todo patch nasce **unlabeled**. Ausencia de registro nao e negativo verdadeiro:
um patch sem evidencia documentada apenas nao tem evidencia suficiente — nao ha
prova de que ficou seco. Tratar o universo nao rotulado como negativo criaria
negativos falsos em massa. Por isso o universo sem evidencia e
`unlabeled_background`, nunca negative.

## 5. Por que hard negative e raro

**hard_negative_audited** so existe num caso raro e auditado: um patch observado
no mesmo evento, fora do footprint, sem exclusoes relevantes (nuvem, sombra,
ambiguidade), com comportamento seco e QA aceito. Exige `observability`,
`exclusion_check` e `qa_status`. Sem essa auditoria explicita, o patch permanece
unlabeled — nunca negativo por omissao.

## 6. Por que score_evaluation e review-only

A comparacao entre score/features e evidencia observacional e exploratoria e
review-only. Ela nao valida o modelo, nao vira benchmark, nao corrige o score_v6
e nao autoriza score_v7. `eligible_for_evaluation` so pode ser true para
positive_strong e hard_negative_audited (ou casos explicitamente aceitos por
politica); `eligible_for_calibration` so pode ser true em regime review-only.

## 7. Quais evidencias podem ou nao promover um patch

A tabela de decisao (`susc_gt01_review_only_decision_table.csv`) fixa o teto de
promocao por tipo de evidencia:

| Evidencia | Pode promover a | patch_link maximo | eval review-only | Decisao |
| --- | --- | --- | --- | --- |
{linhas_dec}

Regras nao-negociaveis: **alerta** nao pode virar evento observado nem patch_link
forte; **area de risco** nao pode virar evento ocorrido; **noticia** nao vira
patch_link forte; **municipio/bairro atingido** nao vira patch atingido;
**footprint pos-evento** nao vira feature pre-evento; e **Sentinel-1/SAR** pode
gerar ou confirmar footprint, mas nao vira verdade automatica.

## 8. Quais bloqueios impedem treino supervisionado

Sao constantes deste marco, nunca derivadas: `eligible_for_training=false`,
`eligible_for_ground_truth=false`, `score_v7_candidate=false`, `trainable=false`,
`review_only=true`. Nenhuma linha de nenhum artefato pode habilitar treino,
ground truth supervisionado ou score_v7 oficial. As razoes estao catalogadas em
`susc_gt01_not_ground_truth_reasons.csv`.

## 9. Separacao de fenomeno em Petropolis

Petropolis exige separacao explicita entre `flood`, `flash_flood`, `landslide` e
`mixed_flood_landslide`. Enquanto o fenomeno estiver misto, nenhum patch da regiao
pode ser promovido: o caso permanece contextual/bloqueado.

## 10. Como isso prepara os proximos marcos

Esta base metodologica executavel (schemas + policy + validador + testes +
relatorio) deixa o projeto pronto para: **calibracao candidata** review-only;
**canarios SAR** observacionais; **QA humano** dos estados; a **matriz multimodal
escalavel**; e um **benchmark futuro** — que permanece nao autorizado enquanto a
amostra e a geometria oficial nao forem suficientes.

REV-P nao preve enchentes operacionalmente: produz analise estrutural review-only
com evidencia observacional auditavel.
"""


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_all() -> dict:
    ensure_dir(OUT)
    ensure_dir(SCHEMAS_DIR)

    write_csv(TAXONOMY_CSV, taxonomy_rows())
    write_csv(TRANSITIONS_CSV, transition_rows())
    write_csv(NOT_GT_CSV, not_ground_truth_rows())
    write_csv(DECISION_CSV, decision_rows())
    write_json(POLICY_JSON, policy_obj())
    write_json(SCHEMA_POLICY, schema_policy_obj())
    write_json(SCHEMA_STATE, schema_state_obj())
    write_json(SCHEMA_DECISION, schema_decision_obj())
    write_markdown(REPORT, report_text())
    # manifesto por ultimo: hash dos demais artefatos ja escritos
    manifest = manifest_obj()
    write_json(MANIFEST_JSON, manifest)
    return manifest


# ---------------------------------------------------------------------------
# Validacao
# ---------------------------------------------------------------------------
def _public_paths() -> list[Path]:
    paths = [p for p in REQUIRED_OUTPUTS if p.suffix.lower() in {".csv", ".json", ".md"} and p.exists()]
    return sorted(dict.fromkeys(paths), key=lambda p: rel(p))


def public_text_violations_text(text: str) -> list[str]:
    return sorted({m.group(0) for m in PUBLIC_FORBIDDEN_RE.finditer(text)})


def operational_forecast_violations(text: str) -> list[str]:
    return sorted({m.group(0).strip() for m in OPERATIONAL_FORECAST_RE.finditer(text)})


def validate_outputs(manifest: dict) -> list[str]:
    errors = [f"missing:{rel(p)}" for p in REQUIRED_OUTPUTS if not p.exists()]
    if errors:
        return errors  # sem artefatos nao ha o que validar

    # --- guardrails do manifesto/policy ---
    if manifest["score_v6_changed"]:
        errors.append("score_v6_changed_forbidden")
    if manifest["score_v7_created"]:
        errors.append("score_v7_created_forbidden")
    for k, v in GLOBAL_GUARDRAILS.items():
        if manifest["global_guardrails"].get(k) != v:
            errors.append(f"guardrail_alterado:{k}")

    # --- taxonomia: todos os estados definidos e com allowed_use ---
    tax = read_csv(TAXONOMY_CSV)
    tax_by_state = {r["state"]: r for r in tax}
    for state in STATE_NAMES:
        if state not in tax_by_state:
            errors.append(f"estado_ausente_na_taxonomia:{state}")
    for r in tax:
        if not r.get("definition", "").strip():
            errors.append(f"taxonomy:{r['state']}:sem_definicao")
        if not r.get("allowed_use", "").strip():
            errors.append(f"taxonomy:{r['state']}:sem_allowed_use")
        # guardrails constantes por linha
        if r.get("eligible_for_training") != "false":
            errors.append(f"taxonomy:{r['state']}:eligible_for_training_nao_false")
        if r.get("eligible_for_ground_truth") != "false":
            errors.append(f"taxonomy:{r['state']}:eligible_for_ground_truth_nao_false")
        if r.get("score_v7_candidate") != "false":
            errors.append(f"taxonomy:{r['state']}:score_v7_candidate_nao_false")
        if r.get("trainable") != "false":
            errors.append(f"taxonomy:{r['state']}:trainable_nao_false")
        if r.get("review_only") != "true":
            errors.append(f"taxonomy:{r['state']}:review_only_nao_true")
        # eligible_for_evaluation apenas nos estados aceitos por politica
        if r.get("eligible_for_evaluation") == "true" and r["state"] not in EVALUATION_ELIGIBLE_STATES:
            errors.append(f"taxonomy:{r['state']}:eval_indevida")
        # eligible_for_calibration exige review_only=true (regime review-only)
        if r.get("eligible_for_calibration") == "true" and r.get("review_only") != "true":
            errors.append(f"taxonomy:{r['state']}:calibracao_fora_de_review_only")

    # unlabeled: default e nunca negativo
    unl = tax_by_state.get("unlabeled", {})
    if unl.get("is_default") != "true":
        errors.append("unlabeled_nao_e_default")
    if unl.get("is_negative") != "false":
        errors.append("unlabeled_tratado_como_negative")

    # positive_strong: campos obrigatorios
    ps_fields = set(tax_by_state.get("positive_strong", {}).get("required_fields", "").split(";"))
    for f in ("event_date", "geometry_type", "source_authority", "patch_link_quality", "uncertainty_m", "qa_status"):
        if f not in ps_fields:
            errors.append(f"positive_strong_sem_campo:{f}")

    # hard_negative_audited: campos obrigatorios
    hn_fields = set(tax_by_state.get("hard_negative_audited", {}).get("required_fields", "").split(";"))
    for f in ("observability", "exclusion_check", "qa_status"):
        if f not in hn_fields:
            errors.append(f"hard_negative_sem_campo:{f}")

    # no_data: exige blocking_reason
    nd = tax_by_state.get("no_data", {})
    if "blocking_reason" not in set(nd.get("required_fields", "").split(";")):
        errors.append("no_data_sem_blocking_reason")
    if nd.get("blocking_reason_required") != "true":
        errors.append("no_data_blocking_reason_required_nao_true")

    # rejected: exige rejection_reason
    if "rejection_reason" not in set(tax_by_state.get("rejected", {}).get("required_fields", "").split(";")):
        errors.append("rejected_sem_rejection_reason")

    # --- transicoes: alerta/area de risco nunca chegam a patch_link forte ---
    for r in transition_rows():
        if r["from_state"] in ("alert_only", "risk_area_not_event") and r["to_state"] == "positive_strong" and r["allowed"] == "true":
            errors.append(f"transicao_forte_indevida:{r['from_state']}")

    # --- tabela de decisao ---
    dec = decision_rows()
    NON_STRONG = {"alert_only", "risk_area", "risk_area_not_event", "news_report",
                  "municipality_affected", "neighborhood_affected", "post_event_footprint"}
    for r in dec:
        if r["score_v7_candidate"] != "false":
            errors.append(f"decision:{r['input_evidence_type']}:score_v7_candidate_nao_false")
        if r["eligible_for_training"] != "false":
            errors.append(f"decision:{r['input_evidence_type']}:eligible_for_training_nao_false")
        if r["eligible_for_ground_truth"] != "false":
            errors.append(f"decision:{r['input_evidence_type']}:eligible_for_ground_truth_nao_false")
        if r["input_evidence_type"] in NON_STRONG:
            if r["max_patch_link_strength"] == STRONG_PATCH_LINK:
                errors.append(f"decision:{r['input_evidence_type']}:patch_link_forte_indevido")
            if r["can_promote_to"] == "positive_strong":
                errors.append(f"decision:{r['input_evidence_type']}:promocao_forte_indevida")
        if r["eligible_for_evaluation"] == "true" and r["can_promote_to"] not in EVALUATION_ELIGIBLE_STATES:
            errors.append(f"decision:{r['input_evidence_type']}:eval_sem_estado_elegivel")

    # --- razoes not-ground-truth: nao vazias ---
    ngt = not_ground_truth_rows()
    if len(ngt) < 6:
        errors.append("not_ground_truth_reasons_insuficientes")
    for r in ngt:
        if not r.get("statement", "").strip():
            errors.append(f"ngt:{r.get('reason_id')}:statement_vazio")

    # --- nenhum artefato pode afirmar que REV-P preve enchentes operacionalmente ---
    for path in _public_paths():
        txt = path.read_text(encoding="utf-8", errors="ignore")
        fc = operational_forecast_violations(txt)
        if fc:
            errors.append(f"{rel(path)}:claim_previsao_operacional:{','.join(fc)}")
        vocab = public_text_violations_text(txt)
        if vocab:
            errors.append(f"{rel(path)}:vocabulario_publico_proibido:{','.join(vocab)}")

    # --- artefatos nao podem ser ignorados pelo git ---
    ignored = git_ignored(REQUIRED_OUTPUTS)
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
        f"GT01 politica de referencia observacional validada: "
        f"estados={len(STATE_NAMES)} default={DEFAULT_STATE} "
        f"eval_elegiveis={sorted(EVALUATION_ELIGIBLE_STATES)} "
        f"score_v6_changed={str(manifest['score_v6_changed']).lower()} "
        f"score_v7_created={str(manifest['score_v7_created']).lower()}"
    )
    return 0


def run_all() -> int:
    manifest = build_all()
    print(
        f"GT01 politica de referencia observacional gerada: "
        f"estados={len(STATE_NAMES)} artefatos={len(manifest['artifacts'])} "
        f"default={DEFAULT_STATE}"
    )
    return 0
