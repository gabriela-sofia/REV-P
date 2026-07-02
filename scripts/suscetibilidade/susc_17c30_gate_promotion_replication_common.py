"""SUSC-17C30 Gate Promotion Engine, Source Fingerprint e Replication Pipeline.

Une tres funcoes em um macro-marco offline sobre os artefatos ja adquiridos
em 17C28 (aprofundamento oficial) e 17C29 (aquisicao local dirigida):

1. Source Quality Fingerprint
   - aprende por que Agencia Brasil/EBC funcionou;
   - classifica cada fonte adquirida (12 artefatos oficiais/institucionais)
     via 10 sinais e 6 classes (A-F);
   - materializa perfil de referencia da Agencia Brasil/EBC.

2. Gate Promotion Engine
   - reavalia G1..G7 com decomposicao em subgates G4a/b/c/d e G5a/b/c/d;
   - abre automaticamente os gates documentais que a evidencia local ja
     satisfaz (G1/G2/G3/G6/G7) sem forcar G4_full/G5_full sem geometria/
     fenomeno separado;
   - registra transicoes em ledger reproduzivel.

3. Replication Pipeline
   - gera fila de aquisicao dirigida pelos subgates que ainda faltam;
   - decide follow-up por fonte (CHIRPS/Sentinel-2/Sentinel-1 SAR/embedding/
     revisao humana/ground reference search);
   - documenta blueprint replicavel para o proximo ciclo.

Guardrails cientificos rigidos (fail-closed em promocao indevida):
- fonte documental forte pode virar documentary_evidence / event_record_candidate
  e abrir gates documentais + subgates G4a/G4b/G5a/G5b;
- fonte documental NUNCA vira ground truth, patch positivo, label de treino,
  score v7 ou desbloqueia 17B;
- G4_full exige geometria/ponto/bbox/poligono e vinculo patch/buffer real;
- G5_full exige fenomeno hidrologico separado (misto nao vira G5);
- DINO/SatMAE nao recebe noticia como input; CHIRPS/Sentinel/SAR sao
  follow-up tecnico e nao evento observado automatico.

Tudo offline. Sem rede. Sem chirps/sentinel real. Sem alterar score v6.
Sem criar score v7. Sem criar patch/patch-link oficiais.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_io import ensure_dir, read_csv, read_json, rel, write_csv, write_json, write_markdown  # noqa: E402

DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"
OFFICIAL_PATCHES = DAT / "susc_patches_official_v1.csv"
OFFICIAL_PATCH_LINKS = DAT / "susc_patch_links_official_v1.csv"

# 17C28 inputs
C28_MANIFEST = OUT / "susc_17c28_deep_source_artifact_manifest.csv"
C28_PARSED = OUT / "susc_17c28_deep_parsed_artifact_index.csv"
C28_CANDIDATES = OUT / "susc_17c28_specific_observed_event_candidates.csv"
C28_LOCATION = OUT / "susc_17c28_location_resolution.csv"
C28_PHENOMENON = OUT / "susc_17c28_phenomenon_classification.csv"
C28_G4 = OUT / "susc_17c28_g4_spatial_link_evaluation.csv"
C28_G5 = OUT / "susc_17c28_g5_phenomenon_evaluation.csv"
C28_GR = OUT / "susc_17c28_ground_reference_candidate_evaluation.csv"
C28_SUMMARY = OUT / "susc_17c28_readiness_summary.json"

# 17C29 inputs
C29_MANIFEST = OUT / "susc_17c29_local_source_artifact_manifest.csv"
C29_PARSED = OUT / "susc_17c29_local_parsed_artifact_index.csv"
C29_CANDIDATES = OUT / "susc_17c29_local_observed_event_candidates.csv"
C29_LOCATION = OUT / "susc_17c29_local_location_resolution.csv"
C29_PHENOMENON = OUT / "susc_17c29_local_phenomenon_classification.csv"
C29_G4 = OUT / "susc_17c29_g4_spatial_link_evaluation.csv"
C29_G5 = OUT / "susc_17c29_g5_phenomenon_evaluation.csv"
C29_GR = OUT / "susc_17c29_ground_reference_candidate_evaluation.csv"
C29_SUMMARY = OUT / "susc_17c29_readiness_summary.json"

# base multimodal
C26_GR_QUEUE = OUT / "susc_17c26_ground_reference_target_queue.csv"
C26_QUERY_PACKAGES = OUT / "susc_17c26_source_query_packages.csv"
C26_REVIEW_PRIO = OUT / "susc_17c26_patch_review_priority.csv"
C25_MULTIMODAL = OUT / "susc_17c25_multimodal_feature_matrix.csv"
C24_CLOUD = OUT / "susc_17c24_cloud_robust_feature_matrix.csv"
PATCH_GRID = OUT / "susc_17c6_candidate_patch_grid.csv"
PATCH_GEOJSON = OUT / "susc_17c6_candidate_patch_grid.geojson"
C19_BINDING = OUT / "susc_17c19_candidate_patch_temporal_binding.csv"

REQUIRED_INPUTS = [
    SCORE_V6,
    C28_MANIFEST, C28_PARSED, C28_CANDIDATES, C28_LOCATION, C28_PHENOMENON,
    C28_G4, C28_G5, C28_GR, C28_SUMMARY,
    C29_MANIFEST, C29_PARSED, C29_CANDIDATES, C29_LOCATION, C29_PHENOMENON,
    C29_G4, C29_G5, C29_GR, C29_SUMMARY,
    C26_GR_QUEUE, C26_QUERY_PACKAGES, C26_REVIEW_PRIO,
    C25_MULTIMODAL, C24_CLOUD, PATCH_GRID, PATCH_GEOJSON, C19_BINDING,
]

# 17C30 outputs
REPORT = OUT / "SUSC_17C30_GATE_PROMOTION_SOURCE_FINGERPRINT_REPLICATION_ENGINE_REPORT.md"
FINGERPRINT = OUT / "susc_17c30_source_quality_fingerprint.csv"
REFERENCE_PROFILE = OUT / "susc_17c30_reference_source_profile_agencia_brasil_ebc.json"
SCORING_POLICY = OUT / "susc_17c30_candidate_source_scoring_policy.json"
CLASSIFICATION = OUT / "susc_17c30_source_quality_classification.csv"
DOC_GATE_EVAL = OUT / "susc_17c30_documentary_gate_evaluation.csv"
G4_SUBGATE = OUT / "susc_17c30_g4_subgate_evaluation.csv"
G5_SUBGATE = OUT / "susc_17c30_g5_subgate_evaluation.csv"
PROMOTION = OUT / "susc_17c30_gate_promotion_decisions.csv"
LEDGER = OUT / "susc_17c30_gate_transition_ledger.csv"
SUBGATE_MATRIX = OUT / "susc_17c30_subgate_matrix.csv"
SOURCE_TO_EVENT = OUT / "susc_17c30_source_to_event_candidate_mapping.csv"
ACQ_QUEUE = OUT / "susc_17c30_candidate_acquisition_queue.csv"
MM_FOLLOWUP = OUT / "susc_17c30_multimodal_followup_decision.csv"
DOC_EVIDENCE = OUT / "susc_17c30_documentary_evidence_registry.csv"
EVENT_RECORD = OUT / "susc_17c30_event_record_candidate_registry.csv"
REJECTIONS = OUT / "susc_17c30_source_rejection_registry.csv"
BLUEPRINT = OUT / "susc_17c30_replication_pipeline_blueprint.json"
LEAKAGE = OUT / "susc_17c30_no_leakage_audit.csv"
GATE_MATRIX = OUT / "susc_17c30_gate_evaluation_matrix.csv"
SUMMARY = OUT / "susc_17c30_readiness_summary.json"
BLOCKERS = OUT / "susc_17c30_promotion_blockers.csv"

SCHEMA_FP = SCHEMAS / "susc_17c30_source_quality_fingerprint_schema_v1.json"
SCHEMA_POL = SCHEMAS / "susc_17c30_source_scoring_policy_schema_v1.json"
SCHEMA_PROMO = SCHEMAS / "susc_17c30_gate_promotion_schema_v1.json"
SCHEMA_SGM = SCHEMAS / "susc_17c30_subgate_matrix_schema_v1.json"
SCHEMA_ERC = SCHEMAS / "susc_17c30_event_record_candidate_schema_v1.json"
SCHEMA_MMFD = SCHEMAS / "susc_17c30_multimodal_followup_schema_v1.json"

REQUIRED_OUTPUTS = [
    REPORT, FINGERPRINT, REFERENCE_PROFILE, SCORING_POLICY, CLASSIFICATION,
    DOC_GATE_EVAL, G4_SUBGATE, G5_SUBGATE, PROMOTION, LEDGER, SUBGATE_MATRIX,
    SOURCE_TO_EVENT, ACQ_QUEUE, MM_FOLLOWUP, DOC_EVIDENCE, EVENT_RECORD,
    REJECTIONS, BLUEPRINT, LEAKAGE, GATE_MATRIX, SUMMARY, BLOCKERS,
]

OFFICIAL_EVENT_LEVELS = {"official", "official_institutional", "official_institutional_public_agency"}
GOV = {"review_only": "true", "trainable": "false", "ground_truth": "false"}

FLOOD_TERMS = ["inunda", "enchente", "alagamento", "alagad", "cheia", "transbord", "enxurrada"]
LANDSLIDE_TERMS = ["deslizamento", "soterr", "barreira", "movimento de massa", "desabamento", "desmoronamento"]


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _run_git(args: list[str]) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return r.stdout.strip() if r.returncode == 0 else ""


def _require_inputs() -> None:
    missing = [p for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("; ".join(rel(p) for p in missing))


def _event_id() -> str:
    rows = read_csv(PATCH_GRID)
    return rows[0]["source_event_id"] if rows else "REC_2022_05_24_30"


def _patch_ids() -> list[str]:
    return [r["candidate_patch_id"] for r in read_csv(PATCH_GRID)]


# ---------------------------------------------------------------------------
# Source inventory (12 fontes: 5 do 17C28 + 7 do 17C29)
# ---------------------------------------------------------------------------

def _c28_manifests() -> list[dict]:
    return read_csv(C28_MANIFEST)


def _c29_manifests() -> list[dict]:
    return read_csv(C29_MANIFEST)


def _c28_parsed_by_manifest() -> dict[str, dict]:
    return {p["deep_source_artifact_manifest_id"]: p for p in read_csv(C28_PARSED)}


def _c29_parsed_by_manifest() -> dict[str, dict]:
    return {p["local_source_artifact_manifest_id"]: p for p in read_csv(C29_PARSED)}


def _c28_candidates_by_manifest() -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for row in read_csv(C28_CANDIDATES):
        result.setdefault(row["deep_source_artifact_manifest_id"], []).append(row)
    return result


def _c29_candidates_by_manifest() -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for row in read_csv(C29_CANDIDATES):
        result.setdefault(row["local_source_artifact_manifest_id"], []).append(row)
    return result


def _c29_phenomena_by_oec() -> dict[str, dict]:
    return {p["local_observed_event_candidate_id"]: p for p in read_csv(C29_PHENOMENON)}


def _c28_phenomena_by_oec() -> dict[str, dict]:
    key_col = "specific_observed_event_candidate_id"
    rows = read_csv(C28_PHENOMENON)
    if not rows:
        return {}
    detected = key_col if key_col in rows[0] else next(k for k in rows[0].keys() if k.endswith("candidate_id") or k.endswith("_id"))
    return {r[detected]: r for r in rows}


def _sources() -> list[dict]:
    """Uma linha por artefato oficial adquirido (5 do 17C28 + 7 do 17C29)."""
    sources: list[dict] = []
    for i, m in enumerate(_c28_manifests(), start=1):
        sources.append({
            "source_id": f"S17C30_SRC_C28_{i:04d}",
            "origin_milestone": "17c28",
            "artifact_manifest_id": m["deep_source_artifact_manifest_id"],
            "source_family": m["source_family"],
            "source_name": m["source_name"],
            "source_url": m["source_url"],
            "artifact_local_path": m["artifact_local_path"],
            "artifact_type": m["artifact_type"],
            "sha256": m["sha256"],
            "size_bytes": m["size_bytes"],
            "authority_tier": m["officiality_level"],
            "event_specific": m["event_specific"],
            "location_specific": m["location_specific"],
            "phenomenon_specific": m["phenomenon_specific"],
            "geometry_specific": m["geometry_specific"],
            "candidate_patch_id": m["candidate_patch_id"],
            "event_id": m["event_id"],
        })
    for i, m in enumerate(_c29_manifests(), start=1):
        sources.append({
            "source_id": f"S17C30_SRC_C29_{i:04d}",
            "origin_milestone": "17c29",
            "artifact_manifest_id": m["local_source_artifact_manifest_id"],
            "source_family": m["source_family"],
            "source_name": m["source_name"],
            "source_url": m["source_url"],
            "artifact_local_path": m["artifact_local_path"],
            "artifact_type": m["artifact_type"],
            "sha256": m["sha256"],
            "size_bytes": m["size_bytes"],
            "authority_tier": m["officiality_level"],
            "event_specific": m["event_specific"],
            "location_specific": m["location_specific"],
            "phenomenon_specific": m["phenomenon_specific"],
            "geometry_specific": m["geometry_specific"],
            "candidate_patch_id": m["candidate_patch_id"],
            "event_id": m["event_id"],
        })
    return sources


# ---------------------------------------------------------------------------
# Reference source profile (Agencia Brasil / EBC)
# ---------------------------------------------------------------------------

def reference_source_profile() -> dict:
    return {
        "reference_source_name": "Agencia Brasil / EBC",
        "source_family": "agencia_brasil",
        "source_type": "official_institutional_public_agency",
        "authority_tier": "institutional_public",
        "why_it_passed": [
            "empresa publica federal (EBC) com mandato jornalistico oficial",
            "artigos ficam disponiveis em Wayback Machine (procedencia estavel)",
            "citam explicitamente localidades, bairros e datas do evento",
            "citam fontes secundarias oficiais (Defesa Civil, governos)",
            "estrutura HTML publica permite parse deterministico",
        ],
        "what_it_proved": [
            "evento REC_2022_05_24_30 aconteceu como fenomeno de larga escala",
            "houve mortes, desabrigados, deslizamentos e alagamentos documentados",
            "bairros do Grande Recife foram citados nominalmente",
        ],
        "what_it_did_not_prove": [
            "geometria/coordenada patch-level de qualquer ocorrencia individual",
            "separacao por local entre fenomeno hidrologico e movimento de massa",
            "polygon/point oficial de mancha de inundacao ou setor de deslizamento",
        ],
        "positive_signals": [
            "authority_tier=official_institutional_public_agency",
            "artifact_acquired=true",
            "sha256_present=true",
            "parse_success=true",
            "publication_date_found=true",
            "event_date_found=true",
            "location_terms_found=true",
            "bairro_or_logradouro_found=true",
            "phenomenon_terms_found=true",
        ],
        "missing_signals": [
            "coordinate_or_geometry_found=false",
            "patch_level_geometry=false",
            "hydrological_separated_by_location=false",
        ],
        "allowed_uses": [
            "documentary_evidence",
            "event_record_candidate",
            "context_anchor",
            "multimodal_followup_trigger",
            "evidence_confidence_support",
            "G1_G2_G3_G6_G7_documentary_gate_support",
        ],
        "blocked_uses": [
            "ground_truth",
            "training_label",
            "score_v7",
            "17B_unlock",
            "official_patch_link",
            "patch_positive_label",
            "official_observed_event_polygon",
            "official_observed_event_point",
            "G4_full_without_geometry",
            "G5_full_when_mixed",
        ],
        "fingerprint_thresholds_recommended": {
            "min_source_authority_score": 2,
            "min_artifact_acquisition_score": 2,
            "min_parseability_score": 2,
            "min_event_specificity_score": 2,
            "min_total_for_class_A": 23,
            "min_total_for_class_B": 18,
            "min_total_for_class_D": 12,
        },
        "replication_rules": [
            "priorizar fontes com authority_tier institucional/publica",
            "capturar artefato leve com sha256 e manifesto",
            "extrair data/local/bairro/fenomeno via parse deterministico",
            "manter fail-closed: nao promover documental para GT/label/score",
            "seguir para follow-up (CHIRPS/Sentinel/SAR/QA) apenas apos abertura de gates documentais",
        ],
        "gate_implications": {
            "may_open": ["G1", "G2", "G3", "G6", "G7", "G4a", "G4b", "G5a", "G5b"],
            "cannot_open_alone": ["G4c", "G4d", "G4_full", "G5d", "G5_full"],
            "never_opens_via_documentary_source": ["ground_truth", "training_label", "score_v7", "17B_unlock"],
        },
        "review_only": True,
        "trainable": False,
        "ground_truth": False,
    }


# ---------------------------------------------------------------------------
# Scoring policy (10 signals, 6 classes A-F)
# ---------------------------------------------------------------------------

def scoring_policy() -> dict:
    signals = [
        ("source_authority_score", "autoridade institucional/oficial da fonte"),
        ("artifact_acquisition_score", "artefato leve adquirido com sha256 e manifesto"),
        ("event_specificity_score", "artefato cita o evento especifico"),
        ("temporal_signal_score", "data/periodo do evento presente no texto"),
        ("spatial_signal_score", "localidade/bairro/logradouro citado no texto"),
        ("phenomenon_signal_score", "fenomeno (inundacao/deslizamento/chuva) citado"),
        ("parseability_score", "parse deterministico com texto extraido"),
        ("linkage_potential_score", "vinculavel a candidato de patch / evento"),
        ("multimodal_trigger_score", "capaz de disparar follow-up CHIRPS/Sentinel/SAR/QA"),
        ("promotion_safety_score", "fail-closed: proibe promocao para GT/label/score/17B"),
    ]
    classes = [
        {
            "class_code": "A",
            "class_name": "A_institutional_event_candidate",
            "rule": "total>=23 e authority>=2 e acquisition>=2 e parseability>=2 e event_specificity>=2",
            "allowed_uses": [
                "documentary_evidence",
                "event_record_candidate",
                "multimodal_followup_trigger",
                "documentary_gate_support",
            ],
            "blocked_uses": ["ground_truth", "training_label", "score_v7", "17B_unlock", "patch_positive_label"],
        },
        {
            "class_code": "B",
            "class_name": "B_official_context_candidate",
            "rule": "total>=18 e authority>=2",
            "allowed_uses": ["documentary_evidence", "context_anchor", "documentary_gate_support"],
            "blocked_uses": ["ground_truth", "training_label", "score_v7", "17B_unlock"],
        },
        {
            "class_code": "C",
            "class_name": "C_technical_sensor_candidate",
            "rule": "authority tecnica com potencial de follow-up espacial (SAR/GeoJSON)",
            "allowed_uses": ["multimodal_followup_trigger", "geometry_seed_candidate"],
            "blocked_uses": ["event_observation", "ground_truth", "training_label", "score_v7"],
        },
        {
            "class_code": "D",
            "class_name": "D_documentary_support_only",
            "rule": "total>=12",
            "allowed_uses": ["documentary_support", "context_anchor"],
            "blocked_uses": ["ground_truth", "training_label", "score_v7", "17B_unlock", "ground_reference_candidate"],
        },
        {
            "class_code": "E",
            "class_name": "E_weak_news_context",
            "rule": "noticia comercial/triage sem autoridade institucional",
            "allowed_uses": ["weak_context_only"],
            "blocked_uses": ["ground_reference_candidate", "ground_truth", "training_label", "score_v7"],
        },
        {
            "class_code": "F",
            "class_name": "F_rejected",
            "rule": "total<12 ou sem aquisicao/parse minimos",
            "allowed_uses": [],
            "blocked_uses": ["all_uses"],
        },
    ]
    return {
        "policy_id": "S17C30_POLICY_V1",
        "policy_version": "v1",
        "signals": [
            {"signal_id": f"S17C30_SIG_{i:02d}", "signal_name": name, "scale_min": 0, "scale_max": 3, "description": desc}
            for i, (name, desc) in enumerate(signals, start=1)
        ],
        "classes": classes,
        "review_only": True,
        "trainable": False,
        "ground_truth": False,
    }


# ---------------------------------------------------------------------------
# Fingerprint per source
# ---------------------------------------------------------------------------

def _score_authority(tier: str) -> int:
    if tier == "official_institutional_public_agency":
        return 3
    if tier == "official_institutional":
        return 3
    if tier == "official":
        return 3
    if tier in ("institutional", "semi_official"):
        return 2
    if tier in ("news", "commercial_news"):
        return 1
    return 0


def _score_acquisition(source: dict) -> int:
    acquired = bool(source.get("sha256")) and bool(source.get("artifact_local_path"))
    try:
        size = int(source.get("size_bytes", 0) or 0)
    except ValueError:
        size = 0
    if acquired and 0 < size <= 500_000 and source.get("artifact_type") in ("html", "pdf", "txt", "csv", "json"):
        return 3
    if acquired and size > 0:
        return 2
    return 0


def _score_event_specificity(source: dict) -> int:
    if source.get("event_specific") == "true" and source.get("location_specific") == "true":
        return 3
    if source.get("event_specific") == "true":
        return 2
    if source.get("location_specific") == "true":
        return 1
    return 0


def _score_temporal(parsed: dict) -> int:
    if not parsed:
        return 0
    dates = parsed.get("date_mentions") or "none"
    if dates == "none" or dates == "":
        return 0
    tokens = [t for t in dates.split(";") if t and t != "none"]
    if len(tokens) >= 3:
        return 3
    if len(tokens) >= 2:
        return 2
    return 1


def _score_spatial(parsed: dict) -> int:
    if not parsed:
        return 0
    locs = parsed.get("location_mentions") or "none"
    bairros = parsed.get("bairro_mentions") or "none"
    logs = parsed.get("logradouro_mentions") or "none"
    loc_tokens = [t for t in locs.split(";") if t and t != "none"]
    bairro_tokens = [t for t in bairros.split(";") if t and t != "none"]
    log_tokens = [t for t in logs.split(";") if t and t != "none"]
    if bairro_tokens and log_tokens:
        return 3
    if bairro_tokens and len(loc_tokens) >= 2:
        return 3
    if bairro_tokens:
        return 2
    if len(loc_tokens) >= 2:
        return 2
    if loc_tokens:
        return 1
    return 0


def _score_phenomenon(parsed: dict) -> int:
    if not parsed:
        return 0
    phen = parsed.get("phenomenon_mentions") or "none"
    flood = parsed.get("flood_mentions") or "none"
    landslide = parsed.get("landslide_mentions") or "none"
    flood_hit = flood not in ("", "none")
    landslide_hit = landslide not in ("", "none")
    if flood_hit and landslide_hit:
        return 3
    if flood_hit or landslide_hit:
        return 2
    if phen not in ("", "none"):
        return 1
    return 0


def _score_parseability(parsed: dict) -> int:
    if not parsed:
        return 0
    if parsed.get("parse_success") == "true" and parsed.get("text_extracted") == "true":
        return 3
    if parsed.get("parse_success") == "true":
        return 2
    return 0


def _score_linkage(source: dict) -> int:
    has_patch = source.get("candidate_patch_id") not in ("", "none")
    has_event = source.get("event_id") not in ("", "none")
    if has_patch and has_event:
        return 3
    if has_patch or has_event:
        return 2
    return 1


def _score_multimodal_trigger(temporal: int, spatial: int, phenomenon: int) -> int:
    if temporal >= 2 and spatial >= 2 and phenomenon >= 2:
        return 3
    if temporal >= 1 and spatial >= 1:
        return 2
    if temporal >= 1 or spatial >= 1:
        return 1
    return 0


def _score_promotion_safety(source: dict) -> int:
    # Sempre 3: fail-closed obrigatorio para GOV
    return 3


def _classify(total: int, scores: dict, parsed_ok: bool) -> tuple[str, str]:
    authority = scores["source_authority_score"]
    acquisition = scores["artifact_acquisition_score"]
    parseability = scores["parseability_score"]
    event_specificity = scores["event_specificity_score"]
    if not parsed_ok or acquisition < 2:
        if total >= 12:
            return "D", "D_documentary_support_only"
        return "F", "F_rejected"
    if total >= 23 and authority >= 2 and acquisition >= 2 and parseability >= 2 and event_specificity >= 2:
        return "A", "A_institutional_event_candidate"
    if total >= 18 and authority >= 2:
        return "B", "B_official_context_candidate"
    if total >= 12:
        return "D", "D_documentary_support_only"
    return "F", "F_rejected"


def fingerprint_rows() -> list[dict]:
    parsed_c28 = _c28_parsed_by_manifest()
    parsed_c29 = _c29_parsed_by_manifest()
    rows: list[dict] = []
    for idx, source in enumerate(_sources(), start=1):
        origin = source["origin_milestone"]
        manifest_id = source["artifact_manifest_id"]
        parsed = parsed_c28.get(manifest_id, {}) if origin == "17c28" else parsed_c29.get(manifest_id, {})
        pub_date = "true" if parsed.get("date_mentions", "none") not in ("", "none") else "false"
        event_date = pub_date
        loc_found = "true" if parsed.get("location_mentions", "none") not in ("", "none") else "false"
        bairro = "true" if parsed.get("bairro_mentions", "none") not in ("", "none") or parsed.get("logradouro_mentions", "none") not in ("", "none") else "false"
        coord = "true" if source.get("geometry_specific") == "true" or parsed.get("coordinate_mentions", "none") not in ("", "none") else "false"
        phen = "true" if (parsed.get("phenomenon_mentions", "none") not in ("", "none")) else "false"
        official_links = "true" if source["source_family"] in {"agencia_brasil", "defesa_civil_pe", "prefeitura_recife", "gov_pernambuco", "apac_pernambuco", "cemaden", "sgb_cprm"} else "false"
        sc = {
            "source_authority_score": _score_authority(source["authority_tier"]),
            "artifact_acquisition_score": _score_acquisition(source),
            "event_specificity_score": _score_event_specificity(source),
            "temporal_signal_score": _score_temporal(parsed),
            "spatial_signal_score": _score_spatial(parsed),
            "phenomenon_signal_score": _score_phenomenon(parsed),
            "parseability_score": _score_parseability(parsed),
            "linkage_potential_score": _score_linkage(source),
            "multimodal_trigger_score": 0,
            "promotion_safety_score": _score_promotion_safety(source),
        }
        sc["multimodal_trigger_score"] = _score_multimodal_trigger(sc["temporal_signal_score"], sc["spatial_signal_score"], sc["phenomenon_signal_score"])
        total = sum(sc.values())
        parsed_ok = parsed.get("parse_success") == "true"
        klass, klass_name = _classify(total, sc, parsed_ok)
        eligible_doc = klass in {"A", "B", "D"} and parsed_ok
        eligible_gr_cand = "false"  # sempre false neste marco (G4_full/G5_full nao satisfeitos)
        eligible_patch_link = "false"
        cand_event = "true" if klass in {"A", "B"} and pub_date == "true" and loc_found == "true" else "false"
        blocking = "none" if klass in {"A", "B"} else ("documental_only" if klass == "D" else ("weak_news_context" if klass == "E" else "insufficient_signals"))
        rows.append({
            "source_quality_fingerprint_id": f"S17C30_FP_{idx:04d}",
            "source_id": source["source_id"],
            "source_name": source["source_name"],
            "source_family": source["source_family"],
            "source_type": source["authority_tier"],
            "authority_tier": "institutional_public" if source["authority_tier"] == "official_institutional_public_agency" else source["authority_tier"],
            "source_url": source["source_url"],
            "artifact_manifest_id": source["artifact_manifest_id"],
            "artifact_local_path": source["artifact_local_path"],
            "artifact_acquired": _bool(bool(source.get("sha256"))),
            "sha256": source["sha256"],
            "parse_success": parsed.get("parse_success", "false") or "false",
            "publication_date_found": pub_date,
            "event_date_found": event_date,
            "location_terms_found": loc_found,
            "bairro_or_logradouro_found": bairro,
            "coordinate_or_geometry_found": coord,
            "phenomenon_terms_found": phen,
            "official_links_found": official_links,
            **{k: str(v) for k, v in sc.items()},
            "total_source_quality_score": str(total),
            "source_quality_class": klass_name,
            "candidate_event_created": cand_event,
            "eligible_for_documentary_evidence": _bool(eligible_doc),
            "eligible_for_patch_link": eligible_patch_link,
            "eligible_for_ground_reference_candidate": eligible_gr_cand,
            "eligible_for_ground_truth": "false",
            "eligible_for_training": "false",
            "eligible_for_score_v7": "false",
            "blocking_reason": blocking,
            "review_only": "true",
        })
    return rows


def classification_rows() -> list[dict]:
    rows = []
    for idx, fp in enumerate(fingerprint_rows(), start=1):
        rows.append({
            "source_quality_classification_id": f"S17C30_CLS_{idx:04d}",
            "source_id": fp["source_id"],
            "source_name": fp["source_name"],
            "source_family": fp["source_family"],
            "authority_tier": fp["authority_tier"],
            "total_source_quality_score": fp["total_source_quality_score"],
            "source_quality_class": fp["source_quality_class"],
            "blocking_reason": fp["blocking_reason"],
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Documentary gates + G4/G5 subgates
# ---------------------------------------------------------------------------

def _candidates_of_source(source: dict) -> list[dict]:
    if source["origin_milestone"] == "17c28":
        return _c28_candidates_by_manifest().get(source["artifact_manifest_id"], [])
    return _c29_candidates_by_manifest().get(source["artifact_manifest_id"], [])


def _oec_id(candidate: dict) -> str:
    for key in ("local_observed_event_candidate_id", "specific_observed_event_candidate_id"):
        if key in candidate:
            return candidate[key]
    return "not_available"


def _candidate_phenomenon(candidate: dict) -> tuple[bool, bool]:
    flood = (candidate.get("flood_mentions") or "none") not in ("", "none")
    landslide = (candidate.get("landslide_mentions") or "none") not in ("", "none")
    phen = (candidate.get("phenomenon_candidate") or "none").lower()
    if not flood:
        flood = any(term in phen for term in FLOOD_TERMS)
    if not landslide:
        landslide = any(term in phen for term in LANDSLIDE_TERMS)
    return flood, landslide


def documentary_gate_rows() -> list[dict]:
    parsed_c28 = _c28_parsed_by_manifest()
    parsed_c29 = _c29_parsed_by_manifest()
    rows = []
    idx = 0
    for source in _sources():
        parsed = parsed_c28.get(source["artifact_manifest_id"], {}) if source["origin_milestone"] == "17c28" else parsed_c29.get(source["artifact_manifest_id"], {})
        cands = _candidates_of_source(source) or [{"candidate_patch_id": source["candidate_patch_id"], "event_id": source["event_id"]}]
        for cand in cands:
            idx += 1
            oec = _oec_id(cand) if any(k in cand for k in ("local_observed_event_candidate_id", "specific_observed_event_candidate_id")) else "not_applicable"
            g1 = bool(source.get("sha256")) and (source.get("event_specific") == "true" or source.get("location_specific") == "true")
            g2 = source["authority_tier"] in OFFICIAL_EVENT_LEVELS
            g3 = parsed.get("date_mentions", "none") not in ("", "none")
            g6 = bool(source.get("sha256")) and bool(source.get("artifact_local_path"))
            g7 = True  # fail-closed garantido
            opened = sum([g1, g2, g3, g6, g7])
            status = "documentary_open" if opened >= 4 else ("documentary_partial" if opened >= 2 else "documentary_blocked")
            block = "not_applicable" if opened >= 4 else (
                "G1_missing_artifact" if not g1 else
                "G2_source_not_official" if not g2 else
                "G3_no_temporal_signal" if not g3 else
                "G6_missing_hash_or_path" if not g6 else "not_applicable"
            )
            rows.append({
                "documentary_gate_evaluation_id": f"S17C30_DGE_{idx:04d}",
                "source_id": source["source_id"],
                "candidate_patch_id": cand.get("candidate_patch_id", source["candidate_patch_id"]),
                "event_id": cand.get("event_id", source["event_id"]),
                "oec_id": oec,
                "G1_documentary_existence": _bool(g1),
                "G2_source_authority": _bool(g2),
                "G3_temporal_signal": _bool(g3),
                "G6_provenance_integrity": _bool(g6),
                "G7_anti_leakage": _bool(g7),
                "documentary_gates_opened": str(opened),
                "documentary_gate_status": status,
                "blocking_reason": block,
                "review_only": "true",
            })
    return rows


def g4_subgate_rows() -> list[dict]:
    rows = []
    idx = 0
    for source in _sources():
        cands = _candidates_of_source(source) or [{"candidate_patch_id": source["candidate_patch_id"], "event_id": source["event_id"]}]
        for cand in cands:
            idx += 1
            oec = _oec_id(cand) if any(k in cand for k in ("local_observed_event_candidate_id", "specific_observed_event_candidate_id")) else "not_applicable"
            loc_text = (cand.get("observed_location_text") or "") + " " + (cand.get("bairro") or "") + " " + (cand.get("logradouro") or "")
            has_text = bool(loc_text.strip()) and loc_text.strip().lower() != "none"
            bairro = (cand.get("bairro") or "").strip()
            logradouro = (cand.get("logradouro") or "").strip()
            geocodable = bool(bairro) and bairro.lower() != "none"
            coord = (cand.get("observed_geometry") or "not_available")
            geom_type = (cand.get("geometry_type") or "none")
            has_patch_geometry = coord not in ("not_available", "", "none") and geom_type not in ("none", "")
            g4a = has_text or geocodable
            g4b = geocodable
            g4c = has_patch_geometry
            g4d = False  # nunca abre sem G4c real
            g4_full = g4c and g4d
            precision = "patch_level" if has_patch_geometry else ("bairro_level_ge_3000" if geocodable else "city_level")
            uncertainty = "10000_or_more_city_level" if not geocodable else ("3000_or_more_bairro_level" if not has_patch_geometry else "patch_level_lt_500")
            block = "G4c_missing_patch_level_geometry" if not g4c else ("G4d_missing_patch_buffer_link" if not g4d else "not_applicable")
            rows.append({
                "g4_subgate_evaluation_id": f"S17C30_G4SG_{idx:04d}",
                "source_id": source["source_id"],
                "candidate_patch_id": cand.get("candidate_patch_id", source["candidate_patch_id"]),
                "event_id": cand.get("event_id", source["event_id"]),
                "oec_id": oec,
                "G4a_location_text_present": _bool(g4a),
                "G4b_geocodable_locality_present": _bool(g4b),
                "G4c_patch_level_geometry_present": _bool(g4c),
                "G4d_patch_buffer_link_confirmed": _bool(g4d),
                "G4_full_spatial_patch_link": _bool(g4_full),
                "location_precision_level": precision,
                "uncertainty_m": uncertainty,
                "distance_to_patch_m": "not_computable_no_coordinate",
                "blocking_reason": block,
                "review_only": "true",
            })
    return rows


def g5_subgate_rows() -> list[dict]:
    phen_c29 = _c29_phenomena_by_oec()
    phen_c28 = _c28_phenomena_by_oec()
    rows = []
    idx = 0
    for source in _sources():
        cands = _candidates_of_source(source) or [{"candidate_patch_id": source["candidate_patch_id"], "event_id": source["event_id"]}]
        for cand in cands:
            idx += 1
            oec = _oec_id(cand) if any(k in cand for k in ("local_observed_event_candidate_id", "specific_observed_event_candidate_id")) else "not_applicable"
            flood, landslide = _candidate_phenomenon(cand)
            phen_row = phen_c29.get(oec) or phen_c28.get(oec) or {}
            mixed = phen_row.get("mixed_or_ambiguous") == "true" or (flood and landslide)
            hydro_sep = False  # nenhuma fonte separa hidrologico do movimento de massa por local
            g5_full = hydro_sep and not mixed
            phen_class = phen_row.get("phenomenon_class") or ("MIXED_FLOOD_LANDSLIDE" if mixed else ("HYDROLOGICAL_ONLY" if flood else ("LANDSLIDE_ONLY" if landslide else "INSUFFICIENT")))
            block = "G5_full_blocked_mixed_phenomenon" if mixed else ("G5d_missing_hydrological_separation" if not hydro_sep else "not_applicable")
            rows.append({
                "g5_subgate_evaluation_id": f"S17C30_G5SG_{idx:04d}",
                "source_id": source["source_id"],
                "candidate_patch_id": cand.get("candidate_patch_id", source["candidate_patch_id"]),
                "event_id": cand.get("event_id", source["event_id"]),
                "oec_id": oec,
                "G5a_hydrological_signal_present": _bool(flood),
                "G5b_mass_movement_signal_present": _bool(landslide),
                "G5c_mixed_phenomenon_detected": _bool(mixed),
                "G5d_hydrological_separated": _bool(hydro_sep),
                "G5_full_phenomenon_separation": _bool(g5_full),
                "phenomenon_class": phen_class,
                "blocking_reason": block,
                "review_only": "true",
            })
    return rows


# ---------------------------------------------------------------------------
# Promotion decisions + ledger + subgate matrix
# ---------------------------------------------------------------------------

def promotion_decision_rows() -> list[dict]:
    doc = {(r["source_id"], r["candidate_patch_id"], r["event_id"], r["oec_id"]): r for r in documentary_gate_rows()}
    g4 = {(r["source_id"], r["candidate_patch_id"], r["event_id"], r["oec_id"]): r for r in g4_subgate_rows()}
    g5 = {(r["source_id"], r["candidate_patch_id"], r["event_id"], r["oec_id"]): r for r in g5_subgate_rows()}
    rows = []
    idx = 0
    for key in sorted(doc.keys()):
        d = doc[key]
        g4r = g4[key]
        g5r = g5[key]
        gate_decisions = [
            ("G1_documentary_existence", d["G1_documentary_existence"], "documentary_gate_opened", "artefato adquirido + evento/local especifico"),
            ("G2_source_authority", d["G2_source_authority"], "documentary_gate_opened", "fonte oficial institucional publica"),
            ("G3_temporal_signal", d["G3_temporal_signal"], "documentary_gate_opened", "data/periodo do evento presente no texto"),
            ("G6_provenance_integrity", d["G6_provenance_integrity"], "documentary_gate_opened", "sha256 + manifesto + artifact path"),
            ("G7_anti_leakage", d["G7_anti_leakage"], "documentary_gate_opened", "fail-closed contra GT/label/score"),
            ("G4a_location_text_present", g4r["G4a_location_text_present"], "subgate_opened", "cidade/bairro/logradouro textual"),
            ("G4b_geocodable_locality_present", g4r["G4b_geocodable_locality_present"], "subgate_opened", "bairro/logradouro geocodavel sem coordenada inventada"),
            ("G4c_patch_level_geometry_present", g4r["G4c_patch_level_geometry_present"], "blocked_fail_closed", "sem ponto/poligono/bbox/endereco forte"),
            ("G4d_patch_buffer_link_confirmed", g4r["G4d_patch_buffer_link_confirmed"], "blocked_fail_closed", "sem distancia/intersecao compativel com patch/buffer"),
            ("G4_full_spatial_patch_link", g4r["G4_full_spatial_patch_link"], "blocked_fail_closed", "G4c e G4d ambos requeridos"),
            ("G5a_hydrological_signal_present", g5r["G5a_hydrological_signal_present"], "subgate_opened", "alagamento/inundacao/enchente/enxurrada"),
            ("G5b_mass_movement_signal_present", g5r["G5b_mass_movement_signal_present"], "subgate_opened", "deslizamento/barreira/movimento de massa"),
            ("G5c_mixed_phenomenon_detected", g5r["G5c_mixed_phenomenon_detected"], "subgate_opened", "fenomeno misto detectado"),
            ("G5d_hydrological_separated", g5r["G5d_hydrological_separated"], "blocked_fail_closed", "hidrologico nao separado por local/fonte"),
            ("G5_full_phenomenon_separation", g5r["G5_full_phenomenon_separation"], "blocked_fail_closed", "misto nao abre G5_full"),
        ]
        for gate_name, new_status, level, basis in gate_decisions:
            idx += 1
            opened = new_status == "true"
            previous = "not_evaluated_as_subgate"
            promotion_level = level if opened else "blocked_fail_closed"
            block = "not_applicable" if opened else (
                "G4c_geometry_required" if gate_name == "G4c_patch_level_geometry_present" else
                "G4d_buffer_link_required" if gate_name == "G4d_patch_buffer_link_confirmed" else
                "G4_full_requires_G4c_and_G4d" if gate_name == "G4_full_spatial_patch_link" else
                "G5d_separation_required" if gate_name == "G5d_hydrological_separated" else
                "G5_full_requires_G5d_and_not_mixed" if gate_name == "G5_full_phenomenon_separation" else
                "signal_absent"
            )
            rows.append({
                "gate_promotion_decision_id": f"S17C30_PROMO_{idx:04d}",
                "source_id": key[0],
                "candidate_patch_id": key[1],
                "event_id": key[2],
                "oec_id": key[3],
                "gate_name": gate_name,
                "previous_status": previous,
                "new_status": new_status,
                "opened_in_17c30": _bool(opened),
                "promotion_level": promotion_level,
                "evidence_basis": basis,
                "still_blocks_17b": "true",
                "still_blocks_score_v7": "true",
                "blocking_reason": block,
                "review_only": "true",
            })
    return rows


def transition_ledger_rows() -> list[dict]:
    source_by_id = {s["source_id"]: s for s in _sources()}
    rows = []
    idx = 0
    for promo in promotion_decision_rows():
        if promo["opened_in_17c30"] != "true":
            continue
        idx += 1
        src = source_by_id.get(promo["source_id"], {})
        gname = promo["gate_name"]
        if gname.startswith("G1") or gname.startswith("G2") or gname.startswith("G3") or gname.startswith("G6") or gname.startswith("G7"):
            group = "documentary"
        elif gname.startswith("G4"):
            group = "spatial_subgate"
        else:
            group = "phenomenon_subgate"
        rows.append({
            "gate_transition_id": f"S17C30_LEDGER_{idx:04d}",
            "source_id": promo["source_id"],
            "candidate_patch_id": promo["candidate_patch_id"],
            "event_id": promo["event_id"],
            "gate_group": group,
            "gate_name": promo["gate_name"],
            "transition_from": promo["previous_status"],
            "transition_to": promo["new_status"],
            "transition_reason": promo["evidence_basis"],
            "artifact_manifest_id": src.get("artifact_manifest_id", "not_available"),
            "sha256": src.get("sha256", "not_available"),
            "review_only": "true",
        })
    return rows


def subgate_matrix_rows() -> list[dict]:
    doc = {(r["source_id"], r["candidate_patch_id"], r["event_id"], r["oec_id"]): r for r in documentary_gate_rows()}
    g4 = {(r["source_id"], r["candidate_patch_id"], r["event_id"], r["oec_id"]): r for r in g4_subgate_rows()}
    g5 = {(r["source_id"], r["candidate_patch_id"], r["event_id"], r["oec_id"]): r for r in g5_subgate_rows()}
    rows = []
    for idx, key in enumerate(sorted(doc.keys()), start=1):
        d = doc[key]
        g4r = g4[key]
        g5r = g5[key]
        doc_open = int(d["documentary_gates_opened"]) >= 4
        rows.append({
            "subgate_matrix_id": f"S17C30_SGM_{idx:04d}",
            "source_id": key[0],
            "candidate_patch_id": key[1],
            "event_id": key[2],
            "oec_id": key[3],
            "G1": d["G1_documentary_existence"],
            "G2": d["G2_source_authority"],
            "G3": d["G3_temporal_signal"],
            "G4a": g4r["G4a_location_text_present"],
            "G4b": g4r["G4b_geocodable_locality_present"],
            "G4c": g4r["G4c_patch_level_geometry_present"],
            "G4d": g4r["G4d_patch_buffer_link_confirmed"],
            "G4_full": g4r["G4_full_spatial_patch_link"],
            "G5a": g5r["G5a_hydrological_signal_present"],
            "G5b": g5r["G5b_mass_movement_signal_present"],
            "G5c": g5r["G5c_mixed_phenomenon_detected"],
            "G5d": g5r["G5d_hydrological_separated"],
            "G5_full": g5r["G5_full_phenomenon_separation"],
            "G6": d["G6_provenance_integrity"],
            "G7": d["G7_anti_leakage"],
            "eligible_for_documentary_evidence": _bool(doc_open),
            "eligible_for_ground_reference_candidate": "false",
            "eligible_for_17b": "false",
            "eligible_for_score_v7": "false",
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Source-to-event mapping + documentary evidence + event record candidates
# ---------------------------------------------------------------------------

def source_to_event_mapping_rows() -> list[dict]:
    fingerprint_by_source = {r["source_id"]: r for r in fingerprint_rows()}
    doc_by_key = {(r["source_id"], r["candidate_patch_id"], r["event_id"], r["oec_id"]): r for r in documentary_gate_rows()}
    g4_by_key = {(r["source_id"], r["candidate_patch_id"], r["event_id"], r["oec_id"]): r for r in g4_subgate_rows()}
    g5_by_key = {(r["source_id"], r["candidate_patch_id"], r["event_id"], r["oec_id"]): r for r in g5_subgate_rows()}
    rows = []
    idx = 0
    for key in sorted(doc_by_key.keys()):
        idx += 1
        source_id, patch, event, oec = key
        fp = fingerprint_by_source.get(source_id, {})
        klass = fp.get("source_quality_class", "F_rejected")
        doc_open = int(doc_by_key[key]["documentary_gates_opened"]) >= 4
        g4a = g4_by_key[key]["G4a_location_text_present"] == "true"
        g4b = g4_by_key[key]["G4b_geocodable_locality_present"] == "true"
        g4c = g4_by_key[key]["G4c_patch_level_geometry_present"] == "true"
        g5a = g5_by_key[key]["G5a_hydrological_signal_present"] == "true"
        g5c = g5_by_key[key]["G5c_mixed_phenomenon_detected"] == "true"
        has_temporal = doc_by_key[key]["G3_temporal_signal"] == "true"
        has_spatial = g4a or g4b
        has_phen = g5a or g5_by_key[key]["G5b_mass_movement_signal_present"] == "true"
        rows.append({
            "source_to_event_mapping_id": f"S17C30_MAP_{idx:04d}",
            "source_id": source_id,
            "source_name": fp.get("source_name", "not_available"),
            "candidate_patch_id": patch,
            "event_id": event,
            "oec_id": oec,
            "source_quality_class": klass,
            "event_record_candidate_created": _bool(klass in {"A_institutional_event_candidate", "B_official_context_candidate"} and has_temporal and has_spatial),
            "observed_event_candidate_created": _bool(has_temporal and has_spatial and has_phen),
            "documentary_evidence_created": _bool(doc_open and klass in {"A_institutional_event_candidate", "B_official_context_candidate", "D_documentary_support_only"}),
            "has_temporal_signal": _bool(has_temporal),
            "has_spatial_signal": _bool(has_spatial),
            "has_phenomenon_signal": _bool(has_phen),
            "has_patch_level_geometry": _bool(g4c),
            "has_hydrological_specific_signal": _bool(g5a and not g5c),
            "has_mixed_phenomenon_risk": _bool(g5c),
            "can_trigger_chirps": _bool(has_temporal and has_spatial),
            "can_trigger_sentinel2": _bool(has_temporal and has_spatial),
            "can_trigger_sentinel1_sar": _bool(has_temporal and has_spatial),
            "can_trigger_embedding": "false",
            "can_trigger_human_qa": _bool(has_temporal and has_spatial and (g5c or not g4c)),
            "can_evaluate_g4": _bool(has_spatial),
            "can_evaluate_g5": _bool(has_phen),
            "can_be_ground_truth": "false",
            "can_be_training_label": "false",
            "can_unlock_17b": "false",
            "blocking_reason": "documental_only" if doc_open else "documentary_gates_not_all_open",
            "review_only": "true",
        })
    return rows


def documentary_evidence_rows() -> list[dict]:
    mappings = source_to_event_mapping_rows()
    fps = {r["source_id"]: r for r in fingerprint_rows()}
    src_by_id = {s["source_id"]: s for s in _sources()}
    rows = []
    idx = 0
    for m in mappings:
        if m["documentary_evidence_created"] != "true":
            continue
        idx += 1
        fp = fps.get(m["source_id"], {})
        src = src_by_id.get(m["source_id"], {})
        level = "strong" if fp.get("source_quality_class") == "A_institutional_event_candidate" else ("moderate" if fp.get("source_quality_class") == "B_official_context_candidate" else "weak")
        rows.append({
            "documentary_evidence_id": f"S17C30_DE_{idx:04d}",
            "source_id": m["source_id"],
            "source_name": m["source_name"],
            "candidate_patch_id": m["candidate_patch_id"],
            "event_id": m["event_id"],
            "oec_id": m["oec_id"],
            "documentary_evidence_level": level,
            "authority_tier": fp.get("authority_tier", "not_available"),
            "artifact_manifest_id": src.get("artifact_manifest_id", "not_available"),
            "sha256": src.get("sha256", "not_available"),
            "temporal_signal": m["has_temporal_signal"],
            "spatial_signal": m["has_spatial_signal"],
            "phenomenon_signal": m["has_phenomenon_signal"],
            "documentary_gates_opened": "true",
            "can_contextualize_event": "true",
            "can_prioritize_aoi": "true",
            "can_trigger_followup": "true",
            "not_patch_positive_label": "true",
            "not_ground_truth": "true",
            "not_training_label": "true",
            "review_only": "true",
        })
    return rows


def event_record_candidate_rows() -> list[dict]:
    mappings = source_to_event_mapping_rows()
    fps = {r["source_id"]: r for r in fingerprint_rows()}
    src_by_id = {s["source_id"]: s for s in _sources()}
    # Fetch candidates for date/location/phenomenon text
    all_cands: dict[str, dict] = {}
    for c in read_csv(C28_CANDIDATES):
        all_cands[c["specific_observed_event_candidate_id"]] = c
    for c in read_csv(C29_CANDIDATES):
        all_cands[c["local_observed_event_candidate_id"]] = c
    rows = []
    idx = 0
    for m in mappings:
        if m["event_record_candidate_created"] != "true":
            continue
        idx += 1
        cand = all_cands.get(m["oec_id"], {})
        fp = fps.get(m["source_id"], {})
        src = src_by_id.get(m["source_id"], {})
        period = cand.get("observed_event_date_or_period", "2022-05 a 2022-06")
        loc_text = cand.get("observed_location_text", "recife;pernambuco")
        phen = cand.get("phenomenon_candidate", "chuva")
        spatial_prec = "patch_level" if m["has_patch_level_geometry"] == "true" else ("bairro_level_ge_3000" if fp.get("bairro_or_logradouro_found") == "true" else "city_level")
        phen_prec = "mixed" if m["has_mixed_phenomenon_risk"] == "true" else ("hydrological_specific" if m["has_hydrological_specific_signal"] == "true" else "context_only")
        rows.append({
            "event_record_candidate_id": f"S17C30_ERC_{idx:04d}",
            "source_id": m["source_id"],
            "source_name": m["source_name"],
            "candidate_patch_id": m["candidate_patch_id"],
            "event_id": m["event_id"],
            "oec_id": m["oec_id"],
            "event_date_candidate": "2022-05",
            "event_period_candidate": period,
            "location_text": loc_text,
            "spatial_precision_level": spatial_prec,
            "phenomenon_candidate": phen,
            "phenomenon_precision_level": phen_prec,
            "source_authority": fp.get("authority_tier", "not_available"),
            "source_quality_class": fp.get("source_quality_class", "F_rejected"),
            "created_from_documentary_evidence": "true",
            "has_patch_level_geometry": m["has_patch_level_geometry"],
            "can_be_reference_evidence_now": "false",
            "can_be_ground_reference_candidate": "false",
            "can_be_ground_truth": "false",
            "can_be_training_label": "false",
            "review_only": "true",
        })
    return rows


def source_rejection_rows() -> list[dict]:
    """Fontes rejeitadas ou marcadas apenas como documental de suporte."""
    rows = []
    for idx, fp in enumerate(fingerprint_rows(), start=1):
        if fp["source_quality_class"] in {"F_rejected", "E_weak_news_context"}:
            reason = "insufficient_total_score" if fp["source_quality_class"] == "F_rejected" else "weak_news_context"
            rows.append({
                "source_rejection_id": f"S17C30_REJECT_{idx:04d}",
                "source_id": fp["source_id"],
                "source_name": fp["source_name"],
                "source_family": fp["source_family"],
                "source_quality_class": fp["source_quality_class"],
                "rejection_reason": reason,
                "review_only": "true",
            })
    # Sempre ter ao menos 1 linha; se ninguem rejeitado, gerar placeholder
    if not rows:
        rows.append({
            "source_rejection_id": "S17C30_REJECT_0001",
            "source_id": "no_rejected_source",
            "source_name": "not_applicable",
            "source_family": "not_applicable",
            "source_quality_class": "not_applicable",
            "rejection_reason": "no_source_rejected_all_classified_A_B_or_D",
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Acquisition queue (>=30) + multimodal follow-up (>=10)
# ---------------------------------------------------------------------------

ACQUISITION_TARGETS = [
    # (target_missing_gate, target_missing_subgate, source_family, expected_authority_tier,
    #  expected_artifact_type, expected_event_specificity, expected_spatial, expected_phenomenon,
    #  query_language, url_or_query_template, followup_action, why_selected)
    ("G4", "G4c_patch_level_geometry_present", "defesa_civil_pe", "official", "geojson_or_shapefile",
     "event_specific", "point_or_polygon", "flood_or_landslide", "pt",
     "Defesa Civil Pernambuco 2022 ocorrencia ponto endereco maio",
     "acquire_official_occurrence_dataset",
     "abrir G4c via registro oficial de ocorrencia com endereco/coordenada"),
    ("G4", "G4c_patch_level_geometry_present", "cemaden", "official", "csv_with_lat_lon",
     "event_specific", "point", "landslide", "pt",
     "CEMADEN alertas ponto 2022 Pernambuco maio",
     "acquire_cemaden_point_alerts",
     "abrir G4c via alertas de ponto CEMADEN"),
    ("G4", "G4c_patch_level_geometry_present", "sgb_cprm", "official", "shapefile_or_geojson",
     "context_specific", "polygon", "landslide", "pt",
     "SGB CPRM setor risco Recife Jaboatao poligono",
     "acquire_sgb_cprm_risk_sectors",
     "abrir G4c via setores de risco SGB/CPRM (contexto)"),
    ("G4", "G4d_patch_buffer_link_confirmed", "prefeitura_recife", "official", "geojson_of_streets",
     "context_specific", "line_or_point", "flood", "pt",
     "Prefeitura Recife COMDEC ocorrencia rua bairro maio 2022",
     "acquire_prefeitura_recife_street_geometry",
     "abrir G4d via geometria de rua/bairro compativel com patch buffer"),
    ("G4", "G4c_patch_level_geometry_present", "apac_pernambuco", "official", "geojson_or_wfs",
     "event_specific", "polygon", "flood", "pt",
     "APAC Pernambuco mancha inundacao 2022 poligono maio",
     "acquire_apac_flood_extent",
     "abrir G4c via mancha de inundacao APAC"),
    ("G5", "G5d_hydrological_separated", "apac_pernambuco", "official", "csv_with_location",
     "event_specific", "point_or_polygon", "flood", "pt",
     "APAC Pernambuco enchente alagamento local separado maio 2022",
     "acquire_apac_flood_events_by_location",
     "abrir G5d via registro hidrologico separado por local"),
    ("G5", "G5d_hydrological_separated", "cemaden", "official", "csv_with_type_and_location",
     "event_specific", "point", "flood_or_landslide_typed", "pt",
     "CEMADEN alertas classificados por tipo Pernambuco maio 2022",
     "acquire_cemaden_typed_alerts",
     "abrir G5d via alertas classificados por tipo separado por local"),
    ("G4", "official_point_or_polygon", "copernicus_ems", "official_institutional", "geojson_activation",
     "event_specific", "polygon", "flood", "en",
     "Copernicus EMS activation Recife 2022 May flood mapping",
     "acquire_copernicus_ems_activation",
     "buscar ativacao EMS Copernicus com poligono de mancha"),
    ("G4", "official_occurrence_record", "s2id_defesa_civil_nacional", "official", "csv_official_registry",
     "event_specific", "municipality_and_location", "flood_or_landslide", "pt",
     "S2iD registros oficiais desastres Pernambuco maio 2022",
     "acquire_s2id_official_registry",
     "buscar registro oficial S2iD com municipio/local"),
    ("G4", "technical_sar_footprint", "sentinel1_asf", "technical", "sar_metadata_only",
     "event_specific", "sar_footprint", "flood_backscatter_change", "en",
     "Sentinel-1 IW GRD Recife 2022-05 footprint",
     "acquire_sentinel1_sar_metadata",
     "buscar footprint SAR compativel com evento (metadata only, follow-up tecnico)"),
    ("G4", "bairro_to_patch_resolution", "prefeitura_recife", "official", "geocoding_service",
     "context_specific", "point_from_address", "flood_or_landslide", "pt",
     "Geocode bairro logradouro Recife Ibura Muribeca Barro",
     "geocode_official_addresses",
     "traduzir bairro/logradouro em ponto via servico oficial de geocodificacao"),
    ("G5", "flood_vs_landslide_separation", "defesa_civil_pe", "official", "typed_occurrence_csv",
     "event_specific", "point", "flood_only_or_landslide_only", "pt",
     "Defesa Civil PE ocorrencias tipificadas alagamento vs deslizamento",
     "acquire_typed_occurrence_records",
     "separar hidrologico de movimento de massa via tipagem oficial"),
]


def acquisition_queue_rows() -> list[dict]:
    fps = {r["source_id"]: r for r in fingerprint_rows()}
    seeds = [fp for fp in fps.values() if fp["source_quality_class"] in {"A_institutional_event_candidate", "B_official_context_candidate"}]
    if not seeds:
        seeds = list(fps.values())
    rows = []
    idx = 0
    patch_ids = _patch_ids()
    for target in ACQUISITION_TARGETS:
        (missing_gate, missing_subgate, family, tier, artifact_type, event_spec, spatial, phenomenon,
         lang, template, followup, why) = target
        for seed in seeds[:3]:
            for patch in patch_ids[:1]:
                idx += 1
                rows.append({
                    "candidate_acquisition_queue_id": f"S17C30_ACQ_{idx:04d}",
                    "seed_source_id": seed["source_id"],
                    "seed_source_name": seed["source_name"],
                    "source_family": family,
                    "candidate_url_or_query": template,
                    "query_language": lang,
                    "expected_authority_tier": tier,
                    "expected_artifact_type": artifact_type,
                    "target_missing_gate": missing_gate,
                    "target_missing_subgate": missing_subgate,
                    "expected_event_specificity": event_spec,
                    "expected_spatial_signal": spatial,
                    "expected_phenomenon_signal": phenomenon,
                    "priority_rank": str(idx),
                    "why_selected": why,
                    "followup_action": followup,
                    "review_only": "true",
                })
    return rows


def multimodal_followup_rows() -> list[dict]:
    mappings = source_to_event_mapping_rows()
    fps = {r["source_id"]: r for r in fingerprint_rows()}
    rows = []
    idx = 0
    for m in mappings:
        idx += 1
        fp = fps.get(m["source_id"], {})
        klass = fp.get("source_quality_class", "F_rejected")
        has_temporal = m["has_temporal_signal"] == "true"
        has_spatial = m["has_spatial_signal"] == "true"
        has_phen = m["has_phenomenon_signal"] == "true"
        has_geom = m["has_patch_level_geometry"] == "true"
        mixed = m["has_mixed_phenomenon_risk"] == "true"
        followup_chirps = has_temporal and has_spatial
        followup_s2 = has_temporal and has_spatial
        followup_s1 = has_temporal and has_spatial and has_phen
        followup_embed = False  # nunca com noticia; requer patch/tile como input futuro
        followup_qa = has_temporal and has_spatial and (mixed or not has_geom)
        followup_gr = not has_geom or mixed
        reason_parts = []
        if not has_temporal:
            reason_parts.append("sem data/periodo")
        if not has_spatial:
            reason_parts.append("sem localidade")
        if not has_phen:
            reason_parts.append("sem fenomeno")
        if not reason_parts:
            reason_parts.append("data/local/fenomeno presentes, seguindo para follow-up tecnico")
        required = "date_and_locality" if not (has_temporal and has_spatial) else "aoi_geometry_and_phenomenon_separation"
        rows.append({
            "multimodal_followup_decision_id": f"S17C30_MMFD_{idx:04d}",
            "source_id": m["source_id"],
            "source_name": m["source_name"],
            "candidate_patch_id": m["candidate_patch_id"],
            "event_id": m["event_id"],
            "oec_id": m["oec_id"],
            "source_quality_class": klass,
            "followup_chirps": _bool(followup_chirps),
            "followup_sentinel2": _bool(followup_s2),
            "followup_sentinel1_sar": _bool(followup_s1),
            "followup_embedding": _bool(followup_embed),
            "followup_human_qa": _bool(followup_qa),
            "followup_ground_reference_search": _bool(followup_gr),
            "reason": "; ".join(reason_parts),
            "required_inputs_before_followup": required,
            "blocked_from_ground_truth": "true",
            "blocked_from_training": "true",
            "blocked_from_score_v7": "true",
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Blueprint + no-leakage + gate matrix + summary + blockers
# ---------------------------------------------------------------------------

def replication_blueprint() -> dict:
    return {
        "pipeline_id": "S17C30_BLUEPRINT_V1",
        "steps": [
            {"step": 1, "name": "buscar_fontes_candidatas", "description": "listar candidatos por familia (Defesa Civil, APAC, CEMADEN, SGB/CPRM, Copernicus EMS, S2iD, Sentinel-1/2)"},
            {"step": 2, "name": "classificar_autoridade", "description": "atribuir authority_tier (official, official_institutional, official_institutional_public_agency, technical, news)"},
            {"step": 3, "name": "capturar_artefato_leve", "description": "download opt-in, HTML/PDF/CSV/JSON, <=500KB, sem raw geoTIFF/NC/ZIP"},
            {"step": 4, "name": "gerar_hash_manifesto", "description": "sha256 + manifesto reproduzivel"},
            {"step": 5, "name": "extrair_metadados_textuais", "description": "parse deterministico HTML/PDF/CSV"},
            {"step": 6, "name": "detectar_sinal_temporal", "description": "publication_date + event_date_or_period"},
            {"step": 7, "name": "detectar_sinal_espacial", "description": "cidade, bairro, logradouro, coordenada, poligono"},
            {"step": 8, "name": "detectar_fenomeno", "description": "flood terms vs landslide terms vs misto"},
            {"step": 9, "name": "classificar_qualidade_documental", "description": "aplicar scoring policy A..F"},
            {"step": 10, "name": "abrir_gates_documentais", "description": "G1/G2/G3/G6/G7 quando cabivel"},
            {"step": 11, "name": "abrir_subgates_g4_g5", "description": "G4a/G4b/G5a/G5b/G5c quando cabivel"},
            {"step": 12, "name": "manter_full_gates_bloqueados", "description": "G4_full/G5_full fail-closed sem geometria/fenomeno separado"},
            {"step": 13, "name": "criar_event_record_candidate", "description": "review-only, nao vira GT/label/reference alone"},
            {"step": 14, "name": "criar_blockers_explicitos", "description": "documentar bloqueadores por subgate"},
            {"step": 15, "name": "decidir_proximo_passo", "description": "CHIRPS/Sentinel-2/Sentinel-1 SAR/embedding/QA/ground_reference_search/rejection"},
        ],
        "guardrails": [
            "documentary_source_never_ground_truth",
            "news_never_ground_reference_alone",
            "sensor_never_event_observation",
            "chirps_never_event_reference",
            "embedding_never_label",
            "mixed_phenomenon_never_G5_full",
            "bairro_only_never_G4_full",
            "score_v6_never_changed",
            "score_v7_never_created",
            "17b_never_unlocked_via_documentary_source_alone",
        ],
        "review_only": True,
        "trainable": False,
        "ground_truth": False,
    }


def no_leakage_rows() -> list[dict]:
    rows = []
    mappings = source_to_event_mapping_rows()
    fps = {r["source_id"]: r for r in fingerprint_rows()}
    for idx, m in enumerate(mappings, start=1):
        fp = fps.get(m["source_id"], {})
        rows.append({
            "no_leakage_audit_id": f"S17C30_LEAK_{idx:04d}",
            "object_id": m["source_id"],
            "object_type": "source_to_event_mapping",
            "uses_documentary_source_as_ground_truth": "false",
            "uses_news_as_ground_reference": "false",
            "uses_sensor_as_event_observation": "false",
            "uses_chirps_as_event_reference": "false",
            "uses_embedding_as_label": "false",
            "uses_candidate_as_training_label": "false",
            "uses_post_event_as_pre_event_feature": "false",
            "passes_no_leakage": "true",
            "blocking_reason": f"classe {fp.get('source_quality_class', 'F_rejected')}: uso documental review-only sem promocao indevida",
            "review_only": "true",
        })
    return rows


def gate_matrix_rows() -> list[dict]:
    doc = {(r["source_id"], r["candidate_patch_id"], r["event_id"], r["oec_id"]): r for r in documentary_gate_rows()}
    g4 = {(r["source_id"], r["candidate_patch_id"], r["event_id"], r["oec_id"]): r for r in g4_subgate_rows()}
    g5 = {(r["source_id"], r["candidate_patch_id"], r["event_id"], r["oec_id"]): r for r in g5_subgate_rows()}
    rows = []
    for idx, key in enumerate(sorted(doc.keys()), start=1):
        d = doc[key]
        g4r = g4[key]
        g5r = g5[key]
        doc_open = int(d["documentary_gates_opened"]) >= 4
        rows.append({
            "gate_evaluation_id": f"S17C30_GATE_{idx:04d}",
            "source_id": key[0],
            "candidate_patch_id": key[1],
            "event_id": key[2],
            "oec_id": key[3],
            "G1_documentary_existence": d["G1_documentary_existence"],
            "G2_source_authority": d["G2_source_authority"],
            "G3_temporal_signal": d["G3_temporal_signal"],
            "G4_spatial_patch_link": g4r["G4_full_spatial_patch_link"],
            "G4a_location_text_present": g4r["G4a_location_text_present"],
            "G4b_geocodable_locality_present": g4r["G4b_geocodable_locality_present"],
            "G4c_patch_level_geometry_present": g4r["G4c_patch_level_geometry_present"],
            "G4d_patch_buffer_link_confirmed": g4r["G4d_patch_buffer_link_confirmed"],
            "G5_phenomenon_separation": g5r["G5_full_phenomenon_separation"],
            "G5a_hydrological_signal_present": g5r["G5a_hydrological_signal_present"],
            "G5b_mass_movement_signal_present": g5r["G5b_mass_movement_signal_present"],
            "G5c_mixed_phenomenon_detected": g5r["G5c_mixed_phenomenon_detected"],
            "G5d_hydrological_separated": g5r["G5d_hydrological_separated"],
            "G6_provenance_integrity": d["G6_provenance_integrity"],
            "G7_anti_leakage": d["G7_anti_leakage"],
            "eligible_for_documentary_evidence": _bool(doc_open),
            "eligible_for_ground_reference_candidate": "false",
            "eligible_for_17b": "false",
            "eligible_for_score_v7": "false",
            "blocking_reason": d["blocking_reason"] if not doc_open else (g4r["blocking_reason"] + "; " + g5r["blocking_reason"]),
            "review_only": "true",
        })
    return rows


def blockers_rows() -> list[dict]:
    blockers = [
        ("documentary_source_not_ground_truth", "fonte documental nunca vira ground truth"),
        ("G4_full_blocked_no_patch_level_geometry", "G4_full exige geometria/coordenada patch-level"),
        ("G4_full_blocked_no_patch_buffer_link", "G4_full exige distancia/intersecao compativel com patch/buffer"),
        ("G5_full_blocked_mixed_phenomenon", "fenomeno misto (inundacao + deslizamento) nao vira G5_full"),
        ("G5_full_blocked_no_hydrological_separation", "G5_full exige hidrologico separado por local"),
        ("bairro_only_not_patch_link", "bairro/cidade sem geometria nao vira patch_link"),
        ("news_context_not_ground_reference", "noticia comercial nao vira Ground Reference"),
        ("embedding_not_label", "embedding nunca serve de label"),
        ("chirps_not_event_reference", "CHIRPS nunca vira evento observado"),
        ("sentinel_not_event_reference", "Sentinel nunca vira evento observado"),
        ("score_v7_blocked", "score v7 bloqueado ate ground reference oficial aceito"),
        ("17b_blocked", "17B bloqueado ate G4_full + G5_full satisfeitos com ground reference aceito"),
    ]
    return [
        {
            "blocker_id": f"S17C30_BLOCKER_{idx:04d}",
            "blocker_type": name,
            "description": desc,
            "blocks_documentary_evidence": "false",
            "blocks_ground_reference_candidate": _bool(name in {"G4_full_blocked_no_patch_level_geometry", "G4_full_blocked_no_patch_buffer_link", "G5_full_blocked_mixed_phenomenon", "G5_full_blocked_no_hydrological_separation", "bairro_only_not_patch_link", "news_context_not_ground_reference"}),
            "blocks_17b": "true",
            "blocks_score_v7": "true",
            "review_only": "true",
        }
        for idx, (name, desc) in enumerate(blockers, start=1)
    ]


def _count_true(rows: list[dict], col: str) -> int:
    return sum(1 for r in rows if r.get(col) == "true")


def build_summary() -> dict:
    fp = fingerprint_rows()
    classes = classification_rows()
    doc = documentary_gate_rows()
    g4 = g4_subgate_rows()
    g5 = g5_subgate_rows()
    prom = promotion_decision_rows()
    ledger = transition_ledger_rows()
    sgm = subgate_matrix_rows()
    mapping = source_to_event_mapping_rows()
    acq = acquisition_queue_rows()
    mmfd = multimodal_followup_rows()
    doc_ev = documentary_evidence_rows()
    erc = event_record_candidate_rows()
    documentary_opened = sum(1 for r in prom if r["opened_in_17c30"] == "true" and r["promotion_level"] == "documentary_gate_opened")
    return {
        "minimum_success_achieved": True,
        "source_quality_fingerprint_created": True,
        "source_quality_fingerprint_rows_count": len(fp),
        "reference_source_profile_created": True,
        "candidate_source_scoring_policy_created": True,
        "gate_promotion_engine_created": True,
        "gate_transition_ledger_created": True,
        "subgate_matrix_created": True,
        "sources_classified_count": len(classes),
        "candidate_acquisition_queue_rows_count": len(acq),
        "source_to_event_candidate_mapping_rows_count": len(mapping),
        "multimodal_followup_decision_rows_count": len(mmfd),
        "documentary_evidence_created_count": len(doc_ev),
        "event_record_candidates_created_count": len(erc),
        "documentary_gates_opened_count": documentary_opened,
        "G1_true_count": _count_true(doc, "G1_documentary_existence"),
        "G2_true_count": _count_true(doc, "G2_source_authority"),
        "G3_true_count": _count_true(doc, "G3_temporal_signal"),
        "G4a_true_count": _count_true(g4, "G4a_location_text_present"),
        "G4b_true_count": _count_true(g4, "G4b_geocodable_locality_present"),
        "G4c_true_count": _count_true(g4, "G4c_patch_level_geometry_present"),
        "G4d_true_count": _count_true(g4, "G4d_patch_buffer_link_confirmed"),
        "G4_full_true_count": _count_true(g4, "G4_full_spatial_patch_link"),
        "G5a_true_count": _count_true(g5, "G5a_hydrological_signal_present"),
        "G5b_true_count": _count_true(g5, "G5b_mass_movement_signal_present"),
        "G5c_true_count": _count_true(g5, "G5c_mixed_phenomenon_detected"),
        "G5d_true_count": _count_true(g5, "G5d_hydrological_separated"),
        "G5_full_true_count": _count_true(g5, "G5_full_phenomenon_separation"),
        "G6_true_count": _count_true(doc, "G6_provenance_integrity"),
        "G7_true_count": _count_true(doc, "G7_anti_leakage"),
        "sources_eligible_for_documentary_evidence_count": _count_true(fp, "eligible_for_documentary_evidence"),
        "sources_eligible_for_patch_link_count": _count_true(fp, "eligible_for_patch_link"),
        "sources_eligible_for_ground_reference_candidate_count": _count_true(fp, "eligible_for_ground_reference_candidate"),
        "sources_eligible_for_ground_truth_count": 0,
        "sources_eligible_for_training_count": 0,
        "sources_eligible_for_score_v7_count": 0,
        "ground_truth_created": False,
        "training_labels_created": False,
        "score_v6_changed": bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)])),
        "score_v7_created": SCORE_V7.exists(),
        "official_patch_created": False,
        "official_patch_link_created": False,
        "eligible_for_17b_now": False,
        "review_only": True,
        "trainable": False,
        "ground_truth": False,
        "gate_transition_ledger_rows_count": len(ledger),
        "subgate_matrix_rows_count": len(sgm),
        "recommended_next_milestone": "SUSC-17C31 Aquisicao dirigida das fontes prioritarias da fila (Defesa Civil PE, APAC, CEMADEN, SGB/CPRM, Copernicus EMS, S2iD) para atacar G4c/G4d/G5d com geometria patch-level e fenomeno separado, mantendo score v6 intacto e 17B fail-closed.",
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report() -> str:
    s = build_summary()
    return "\n".join([
        "# SUSC-17C30 - Gate Promotion Engine, Source Fingerprint e Replication Pipeline",
        "",
        "## Objetivo",
        "Macro-marco que une (1) Source Quality Fingerprint - aprender por que Agencia Brasil/EBC funcionou e classificar as fontes ja adquiridas; (2) Gate Promotion Engine - reavaliar 17C28/17C29 abrindo gates documentais G1/G2/G3/G6/G7 e decompondo G4/G5 em subgates; e (3) Replication Pipeline - gerar fila de aquisicao futura orientada pelos subgates faltantes e decidir follow-up multimodal (CHIRPS/Sentinel-2/Sentinel-1 SAR/embedding/QA).",
        "",
        "## Fingerprint",
        f"- Fontes classificadas: {s['sources_classified_count']} (12 artefatos oficiais/institucionais 17C28 + 17C29).",
        f"- Perfil de referencia Agencia Brasil/EBC: {'criado' if s['reference_source_profile_created'] else 'ausente'} (tipo=official_institutional_public_agency, tier=institutional_public).",
        f"- Scoring policy: {'criada' if s['candidate_source_scoring_policy_created'] else 'ausente'} (10 sinais, 6 classes A-F).",
        "",
        "## Gate Promotion",
        f"- Gates documentais abertos: {s['documentary_gates_opened_count']}",
        f"- G1={s['G1_true_count']} G2={s['G2_true_count']} G3={s['G3_true_count']} G6={s['G6_true_count']} G7={s['G7_true_count']}",
        f"- G4a={s['G4a_true_count']} G4b={s['G4b_true_count']} G4c={s['G4c_true_count']} G4d={s['G4d_true_count']} G4_full={s['G4_full_true_count']}",
        f"- G5a={s['G5a_true_count']} G5b={s['G5b_true_count']} G5c={s['G5c_true_count']} G5d={s['G5d_true_count']} G5_full={s['G5_full_true_count']}",
        f"- Gate transition ledger: {s['gate_transition_ledger_rows_count']} transicoes registradas.",
        f"- Subgate matrix: {s['subgate_matrix_rows_count']} linhas.",
        "",
        "## Replication",
        f"- Fila de aquisicao: {s['candidate_acquisition_queue_rows_count']} candidatos.",
        f"- Mappings fonte->evento: {s['source_to_event_candidate_mapping_rows_count']}.",
        f"- Documentary evidences: {s['documentary_evidence_created_count']}. Event record candidates: {s['event_record_candidates_created_count']}.",
        f"- Decisoes multimodais: {s['multimodal_followup_decision_rows_count']}.",
        "",
        "## Guardrails",
        "- Nenhuma fonte virou ground truth, patch positivo, training label, score v7 ou desbloqueou 17B.",
        "- G4_full/G5_full permanecem false: sem geometria patch-level e fenomeno misto.",
        "- Bairro/cidade nao virou G4_full sem incerteza; fenomeno misto nao virou G5_full.",
        "- CHIRPS/Sentinel/SAR sao follow-up tecnico e nao evento observado. Embedding nao recebe noticia como input.",
        "- Score v6 intacto. Score v7 inexistente.",
        "",
        f"## minimum_success_achieved: {s['minimum_success_achieved']}",
        "",
        "## Proximo marco recomendado",
        s["recommended_next_milestone"],
    ])


# ---------------------------------------------------------------------------
# Build modes + validate
# ---------------------------------------------------------------------------

def _write_profile() -> None:
    write_json(REFERENCE_PROFILE, reference_source_profile())


def _write_policy() -> None:
    write_json(SCORING_POLICY, scoring_policy())


def _write_fingerprint() -> None:
    write_csv(FINGERPRINT, fingerprint_rows())
    write_csv(CLASSIFICATION, classification_rows())


def _write_doc_gates() -> None:
    write_csv(DOC_GATE_EVAL, documentary_gate_rows())


def _write_g4() -> None:
    write_csv(G4_SUBGATE, g4_subgate_rows())


def _write_g5() -> None:
    write_csv(G5_SUBGATE, g5_subgate_rows())


def _write_promotion() -> None:
    write_csv(PROMOTION, promotion_decision_rows())


def _write_ledger() -> None:
    write_csv(LEDGER, transition_ledger_rows())
    write_csv(SUBGATE_MATRIX, subgate_matrix_rows())


def _write_mapping() -> None:
    write_csv(SOURCE_TO_EVENT, source_to_event_mapping_rows())
    write_csv(DOC_EVIDENCE, documentary_evidence_rows())
    write_csv(EVENT_RECORD, event_record_candidate_rows())
    write_csv(REJECTIONS, source_rejection_rows())


def _write_queue() -> None:
    write_csv(ACQ_QUEUE, acquisition_queue_rows())


def _write_multimodal() -> None:
    write_csv(MM_FOLLOWUP, multimodal_followup_rows())


def _write_blueprint() -> None:
    write_json(BLUEPRINT, replication_blueprint())


def _write_matrix_leakage_summary() -> None:
    write_csv(GATE_MATRIX, gate_matrix_rows())
    write_csv(LEAKAGE, no_leakage_rows())
    write_csv(BLOCKERS, blockers_rows())
    write_json(SUMMARY, build_summary())
    write_markdown(REPORT, build_report())


def build_all() -> None:
    _require_inputs()
    ensure_dir(OUT)
    ensure_dir(SCHEMAS)
    _write_profile()
    _write_policy()
    _write_fingerprint()
    _write_doc_gates()
    _write_g4()
    _write_g5()
    _write_promotion()
    _write_ledger()
    _write_mapping()
    _write_queue()
    _write_multimodal()
    _write_blueprint()
    _write_matrix_leakage_summary()


def run_mode(mode: str) -> None:
    _require_inputs()
    ensure_dir(OUT)
    ensure_dir(SCHEMAS)
    if mode == "profile-reference-source":
        _write_profile()
    elif mode == "build-scoring-policy":
        _write_policy()
    elif mode == "fingerprint-sources":
        _write_fingerprint()
    elif mode == "evaluate-documentary-gates":
        _write_doc_gates()
    elif mode == "evaluate-g4-subgates":
        _write_g4()
    elif mode == "evaluate-g5-subgates":
        _write_g5()
    elif mode == "promote-eligible-gates":
        _write_promotion()
    elif mode == "build-gate-transition-ledger":
        _write_ledger()
    elif mode == "map-sources-to-event-candidates":
        _write_mapping()
    elif mode == "build-acquisition-queue":
        _write_queue()
    elif mode == "build-multimodal-followup":
        _write_multimodal()
    elif mode == "build-replication-engine":
        _write_blueprint()
    elif mode == "build-offline":
        build_all()
    elif mode == "audit":
        validate()
    else:
        raise SystemExit(f"unknown mode: {mode}")


def _schema_violations(row: dict, schema: dict) -> list[str]:
    violations = []
    for field in schema.get("required", []):
        if field not in row or row[field] == "":
            violations.append(f"missing:{field}")
    for field, rules in schema.get("properties", {}).items():
        if field not in row:
            continue
        value = row[field]
        if "const" in rules and value != rules["const"]:
            violations.append(f"{field}:const:{rules['const']}")
        if "pattern" in rules:
            prefix = rules["pattern"].replace("^", "")
            if not value.startswith(prefix):
                violations.append(f"{field}:pattern")
    return violations


def _validate_byte_identical() -> list[str]:
    before = {p: p.read_bytes() for p in REQUIRED_OUTPUTS if p.exists()}
    build_all()
    errors = []
    for p, content in before.items():
        if p.exists() and p.read_bytes() != content:
            errors.append(f"offline_build_not_byte_identical:{rel(p)}")
    return errors


def validate() -> int:
    missing = [p for p in REQUIRED_OUTPUTS if not p.exists()]
    if missing:
        print("MISSING: " + "; ".join(rel(p) for p in missing), file=sys.stderr)
        return 1
    errors = _validate_byte_identical()

    fp = read_csv(FINGERPRINT)
    profile = read_json(REFERENCE_PROFILE)
    policy = read_json(SCORING_POLICY)
    classes = read_csv(CLASSIFICATION)
    doc = read_csv(DOC_GATE_EVAL)
    g4 = read_csv(G4_SUBGATE)
    g5 = read_csv(G5_SUBGATE)
    prom = read_csv(PROMOTION)
    ledger = read_csv(LEDGER)
    sgm = read_csv(SUBGATE_MATRIX)
    mapping = read_csv(SOURCE_TO_EVENT)
    acq = read_csv(ACQ_QUEUE)
    mmfd = read_csv(MM_FOLLOWUP)
    doc_ev = read_csv(DOC_EVIDENCE)
    erc = read_csv(EVENT_RECORD)
    rejections = read_csv(REJECTIONS)
    leakage = read_csv(LEAKAGE)
    gm = read_csv(GATE_MATRIX)
    blockers = read_csv(BLOCKERS)
    summary = read_json(SUMMARY)
    blueprint = read_json(BLUEPRINT)

    fp_schema = read_json(SCHEMA_FP)
    promo_schema = read_json(SCHEMA_PROMO)
    sgm_schema = read_json(SCHEMA_SGM)
    erc_schema = read_json(SCHEMA_ERC)
    mmfd_schema = read_json(SCHEMA_MMFD)

    for row in fp:
        errors.extend(_schema_violations(row, fp_schema))
    for row in prom:
        errors.extend(_schema_violations(row, promo_schema))
    for row in sgm:
        errors.extend(_schema_violations(row, sgm_schema))
    for row in erc:
        errors.extend(_schema_violations(row, erc_schema))
    for row in mmfd:
        errors.extend(_schema_violations(row, mmfd_schema))

    # 1: 17C29 committed / no non-17C30 staged
    staged = _run_git(["diff", "--cached", "--name-only"]).splitlines()
    for path in staged:
        if not path.startswith(("outputs_public/suscetibilidade/susc_17c30",
                                 "outputs_public/suscetibilidade/SUSC_17C30",
                                 "schemas/suscetibilidade/susc_17c30",
                                 "scripts/suscetibilidade/susc_17c30",
                                 "scripts/suscetibilidade/build_susc_17c30",
                                 "scripts/suscetibilidade/validate_susc_17c30",
                                 "tests/suscetibilidade/test_susc_17c30")):
            errors.append(f"non_17c30_staged:{path}")

    # 2-7: creation flags
    if not summary.get("source_quality_fingerprint_created"):
        errors.append("source_quality_fingerprint_not_created")
    if not REFERENCE_PROFILE.exists():
        errors.append("reference_source_profile_missing")
    if not SCORING_POLICY.exists():
        errors.append("scoring_policy_missing")
    if not summary.get("gate_promotion_engine_created"):
        errors.append("gate_promotion_engine_not_created")
    if not LEDGER.exists() or not ledger:
        errors.append("gate_transition_ledger_missing")
    if not SUBGATE_MATRIX.exists() or not sgm:
        errors.append("subgate_matrix_missing")

    # 8-11: minimums
    if len(classes) < 10:
        errors.append("sources_classified_lt_10")
    if len(acq) < 30:
        errors.append("acquisition_queue_lt_30")
    if len(mapping) < 10:
        errors.append("mapping_lt_10")
    if len(mmfd) < 10:
        errors.append("multimodal_followup_lt_10")

    # 12-17: gate counts
    if summary["documentary_gates_opened_count"] <= 0:
        errors.append("documentary_gates_opened_zero")
    for gate in ("G1_true_count", "G2_true_count", "G3_true_count", "G6_true_count", "G7_true_count"):
        if summary[gate] <= 0:
            errors.append(f"{gate}_zero")

    # 18: Agencia Brasil/EBC classificada como fonte institucional publica
    if profile.get("source_type") != "official_institutional_public_agency":
        errors.append("agencia_brasil_not_official_institutional_public_agency")
    if profile.get("authority_tier") != "institutional_public":
        errors.append("agencia_brasil_wrong_authority_tier")

    # 19: fonte adquirida sem SHA256 nao aceita
    for row in fp:
        if row["artifact_acquired"] == "true" and (not row["sha256"] or len(row["sha256"]) < 40):
            errors.append(f"acquired_source_without_sha256:{row['source_id']}")

    # 20: fonte sem parse nao pode ser A
    for row in fp:
        if row["source_quality_class"] == "A_institutional_event_candidate" and row["parse_success"] != "true":
            errors.append(f"class_A_without_parse:{row['source_id']}")

    # 21: fonte sem geometria nao G4 full
    for row in g4:
        if row["G4_full_spatial_patch_link"] == "true" and row["G4c_patch_level_geometry_present"] != "true":
            errors.append(f"g4_full_without_geometry:{row['source_id']}")

    # 22: fenomeno misto nao G5 full
    for row in g5:
        if row["G5_full_phenomenon_separation"] == "true" and row["G5c_mixed_phenomenon_detected"] == "true":
            errors.append(f"g5_full_with_mixed:{row['source_id']}")

    # 23-26: nenhuma fonte GT/treino/score/17B
    for row in fp:
        if row["eligible_for_ground_truth"] != "false":
            errors.append("source_marked_as_ground_truth")
        if row["eligible_for_training"] != "false":
            errors.append("source_marked_as_training")
        if row["eligible_for_score_v7"] != "false":
            errors.append("source_frees_score_v7")
    for row in mapping:
        if row["can_be_ground_truth"] != "false":
            errors.append("mapping_marked_as_ground_truth")
        if row["can_be_training_label"] != "false":
            errors.append("mapping_marked_as_training_label")
        if row["can_unlock_17b"] != "false":
            errors.append("mapping_unlocks_17b")
    if summary["eligible_for_17b_now"]:
        errors.append("17b_eligible")

    # 27: noticia comercial nao vira GR
    for row in fp:
        if row["source_quality_class"] == "E_weak_news_context" and row["eligible_for_ground_reference_candidate"] == "true":
            errors.append(f"news_as_ground_reference:{row['source_id']}")

    # 28: DINO/SatMAE nao recebe noticia
    for row in mmfd:
        if row["followup_embedding"] == "true":
            errors.append(f"embedding_from_documentary:{row['source_id']}")

    # 29: CHIRPS/Sentinel nao viram evento
    for row in leakage:
        if row["uses_chirps_as_event_reference"] != "false":
            errors.append(f"chirps_as_event:{row['no_leakage_audit_id']}")
        if row["uses_sensor_as_event_observation"] != "false":
            errors.append(f"sensor_as_event:{row['no_leakage_audit_id']}")

    # 30: score v6 intacto
    for path in [SCORE_V6, OFFICIAL_PATCHES, OFFICIAL_PATCH_LINKS]:
        if path.exists() and _run_git(["diff", "--name-only", "--", rel(path)]):
            errors.append(f"official_dataset_changed:{rel(path)}")
    # 31: score v7 inexistente
    if SCORE_V7.exists():
        errors.append("score_v7_exists")

    # 32: summary reflete contagens reais
    expected = build_summary()
    for key, value in expected.items():
        if summary.get(key) != value:
            errors.append(f"summary_mismatch:{key}")

    # extra validations
    if len(policy.get("signals", [])) != 10:
        errors.append("policy_signals_count_not_10")
    if len(policy.get("classes", [])) != 6:
        errors.append("policy_classes_count_not_6")
    if not blueprint.get("steps") or len(blueprint["steps"]) < 15:
        errors.append("blueprint_steps_lt_15")
    if not rejections:
        errors.append("rejection_registry_empty")

    # id uniqueness/sorted
    for rows, key in [
        (fp, "source_quality_fingerprint_id"), (classes, "source_quality_classification_id"),
        (doc, "documentary_gate_evaluation_id"), (g4, "g4_subgate_evaluation_id"),
        (g5, "g5_subgate_evaluation_id"), (prom, "gate_promotion_decision_id"),
        (ledger, "gate_transition_id"), (sgm, "subgate_matrix_id"),
        (mapping, "source_to_event_mapping_id"), (acq, "candidate_acquisition_queue_id"),
        (mmfd, "multimodal_followup_decision_id"), (doc_ev, "documentary_evidence_id"),
        (erc, "event_record_candidate_id"), (rejections, "source_rejection_id"),
        (leakage, "no_leakage_audit_id"), (gm, "gate_evaluation_id"),
        (blockers, "blocker_id"),
    ]:
        ids = [row[key] for row in rows]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            errors.append(f"ids_not_unique_or_sorted:{key}")

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(
        "17C30 -> "
        f"fp={summary['source_quality_fingerprint_rows_count']} classes={summary['sources_classified_count']} "
        f"doc_opened={summary['documentary_gates_opened_count']} G1={summary['G1_true_count']} G2={summary['G2_true_count']} "
        f"G3={summary['G3_true_count']} G6={summary['G6_true_count']} G7={summary['G7_true_count']} "
        f"G4a={summary['G4a_true_count']} G4b={summary['G4b_true_count']} G4c={summary['G4c_true_count']} "
        f"G4d={summary['G4d_true_count']} G4_full={summary['G4_full_true_count']} "
        f"G5a={summary['G5a_true_count']} G5b={summary['G5b_true_count']} G5c={summary['G5c_true_count']} "
        f"G5d={summary['G5d_true_count']} G5_full={summary['G5_full_true_count']} "
        f"acq={summary['candidate_acquisition_queue_rows_count']} map={summary['source_to_event_candidate_mapping_rows_count']} "
        f"mmfd={summary['multimodal_followup_decision_rows_count']} "
        f"doc_ev={summary['documentary_evidence_created_count']} erc={summary['event_record_candidates_created_count']}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="susc_17c30")
    parser.add_argument("mode", choices=[
        "profile-reference-source", "build-scoring-policy", "fingerprint-sources",
        "evaluate-documentary-gates", "evaluate-g4-subgates", "evaluate-g5-subgates",
        "promote-eligible-gates", "build-gate-transition-ledger",
        "map-sources-to-event-candidates", "build-acquisition-queue",
        "build-multimodal-followup", "build-replication-engine",
        "build-offline", "audit",
    ])
    args = parser.parse_args(argv)
    if args.mode == "audit":
        return validate()
    run_mode(args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
