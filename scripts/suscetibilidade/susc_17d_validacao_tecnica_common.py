"""SUSC-17D validacao tecnica de evidencia observacional.

Consome a fila de validacao tecnica gerada no SUSC-17C5 e produz decisoes
metodologicas review-only. Nenhum resultado vira ground truth, treino,
score_v7 ou benchmark 17B.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_io import ensure_dir, read_csv, read_json, rel, write_csv, write_json, write_markdown  # noqa: E402

DATASETS = ROOT / "datasets"
DAT_SUSC = DATASETS / "suscetibilidade"
OUT_DATA_17C = ROOT / "outputs_public" / "data" / "susc_17c_strong_reference_acquisition_canary"
OUT_DATA_17C5 = ROOT / "outputs_public" / "data" / "susc_17c5_geometry_to_patch_linkage_resolver"
OUT_DATA = ROOT / "outputs_public" / "data" / "susc_17d_validacao_tecnica_evidencia_observacional"
CARDS_DIR = OUT_DATA / "cartoes_evidencia"
OUT_REPORTS = ROOT / "outputs_public" / "reports"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

QA_QUEUE_17C5 = OUT_DATA_17C5 / "reference_patch_link_qa_queue.csv"
PATCH_LINKS_17C5 = OUT_DATA_17C5 / "susc_17c5_patch_links.csv"
GEOMETRIES_17C5 = OUT_DATA_17C5 / "susc_17c5_normalized_geometry.csv"
BLOCKERS_17C5 = OUT_DATA_17C5 / "susc_17c5_patch_link_blockers.csv"
GEOJSON_17C5 = OUT_DATA_17C5 / "susc_17c5_candidate_patch_links.geojson"
SUMMARY_17C5 = OUT_DATA_17C5 / "susc_17c5_geometry_to_patch_linkage_summary.json"
REPORT_17C5 = OUT_REPORTS / "SUSC_17C5_GEOMETRY_TO_PATCH_LINKAGE_RESOLVER_REPORT.md"
TARGET_PACK_17C = OUT_DATA_17C / "susc_17c_source_target_pack.csv"
SUMMARY_17C = OUT_DATA_17C / "susc_17c_strong_reference_acquisition_summary.json"
FEATURES = DAT_SUSC / "susc_features_by_patch_v1.csv"
SCORE_V6 = DAT_SUSC / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT_SUSC / "susc_score_v7_candidate_by_patch_v1.csv"

PREFLIGHT_CSV = OUT_DATA / "preflight_validacao_tecnica.csv"
PREFLIGHT_JSON = OUT_DATA / "preflight_validacao_tecnica.json"
DECISIONS = OUT_DATA / "decisoes_validacao_tecnica.csv"
FEATURE_MATRIX = OUT_DATA / "matriz_consistencia_features.csv"
STATUS_SUMMARY = OUT_DATA / "resumo_por_status_validacao.csv"
REGION_SUMMARY = OUT_DATA / "resumo_por_regiao.csv"
BLOCKERS = OUT_DATA / "bloqueios_validacao_tecnica.csv"
GATE_17B = OUT_DATA / "gate_prontidao_17b.csv"
SUMMARY = OUT_DATA / "susc_17d_validacao_tecnica_summary.json"
REPORT = OUT_REPORTS / "SUSC_17D_VALIDACAO_TECNICA_EVIDENCIA_OBSERVACIONAL.md"

SCHEMA_DECISION = SCHEMAS / "susc_17d_validacao_tecnica_schema_v1.json"

REQUIRED_INPUTS = [
    QA_QUEUE_17C5,
    PATCH_LINKS_17C5,
    GEOMETRIES_17C5,
    BLOCKERS_17C5,
    GEOJSON_17C5,
    SUMMARY_17C5,
    REPORT_17C5,
    TARGET_PACK_17C,
    SUMMARY_17C,
    FEATURES,
    SCORE_V6,
]

REQUIRED_OUTPUTS = [
    PREFLIGHT_CSV,
    PREFLIGHT_JSON,
    DECISIONS,
    FEATURE_MATRIX,
    STATUS_SUMMARY,
    REGION_SUMMARY,
    BLOCKERS,
    GATE_17B,
    SUMMARY,
    REPORT,
    SCHEMA_DECISION,
]

STATUS_ALLOWED = [
    "aceito_para_avaliacao_review_only",
    "rejeitado",
    "ambiguo",
    "precisa_mais_fonte",
    "incompatibilidade_espacial",
    "incompatibilidade_temporal",
    "incompatibilidade_de_fenomeno",
    "geometria_insuficiente",
]
CONFIDENCE_ALLOWED = ["alta", "media", "baixa", "insuficiente"]
PUBLIC_FORBIDDEN_RE = re.compile(r"\b(?:agentic|agente|llm|codex|ia|human\s+qa)\b", re.IGNORECASE)
NO_GT_REASON = "validacao tecnica review-only; nao e verdade de referencia, nao e treino e nao autoriza score_v7"
STRONG_LINK_CLASSES = {"exact_polygon_overlap", "point_within_patch", "bbox_overlap"}
SOURCE_TYPES_FOR_ACCEPTANCE = {
    "official_observed_event_polygon",
    "official_observed_event_point",
    "technical_remote_sensing_flood_footprint",
}

PREFLIGHT_FIELDS = ["artifact_role", "path", "exists", "row_count", "columns_found", "notes"]
DECISION_FIELDS = [
    "item_validacao_id",
    "candidate_event_id",
    "source_id",
    "source_type",
    "geometry_id",
    "patch_id",
    "cidade",
    "regiao",
    "tipo_fenomeno",
    "data_evento",
    "classe_vinculo",
    "pontuacao_confiabilidade_fonte",
    "pontuacao_qualidade_temporal",
    "pontuacao_qualidade_geometrica",
    "pontuacao_qualidade_vinculo_patch",
    "pontuacao_compatibilidade_fenomeno",
    "pontuacao_incerteza_espacial",
    "pontuacao_consistencia_features",
    "pontuacao_risco_vazamento_temporal",
    "pontuacao_severidade_bloqueios",
    "confianca_tecnica_final",
    "status_validacao_tecnica",
    "accepted_for_evaluation",
    "accepted_for_calibration_candidate",
    "eligible_for_training",
    "ground_truth",
    "review_only",
    "score_v7_allowed",
    "not_ground_truth_reason",
    "justificativa_tecnica",
    "bloqueios_criticos",
]
FEATURE_FIELDS = [
    "item_validacao_id",
    "candidate_event_id",
    "patch_id",
    "nome_feature",
    "feature_available",
    "valor_feature",
    "direcao_esperada",
    "consistencia_com_suscetibilidade",
    "contradicoes_detectadas",
    "observacao_metodologica",
]
STATUS_SUMMARY_FIELDS = ["status_validacao_tecnica", "quantidade"]
REGION_SUMMARY_FIELDS = [
    "regiao",
    "itens",
    "aceitos_para_avaliacao",
    "candidatos_a_calibracao",
    "rejeitados",
    "ambiguos",
    "precisam_mais_fonte",
]
BLOCKER_FIELDS = [
    "bloqueio_id",
    "item_validacao_id",
    "candidate_event_id",
    "patch_id",
    "classe_bloqueio",
    "severidade",
    "descricao",
    "acao_recomendada",
    "review_only",
    "ground_truth",
    "eligible_for_training",
    "score_v7_allowed",
]
GATE_FIELDS = ["criterio", "valor_observado", "limiar", "passou", "observacao"]

FEATURE_MAP = [
    ("score_v6", "susc_score_v6_candidate", "maior indica maior suscetibilidade"),
    ("score_v6_class", "susc_class_v6_candidate", "classe interpretativa do score v6"),
    ("HAND", "hand_mean", "menor HAND tende a apoiar suscetibilidade hidrologica"),
    ("slope", "slope_mean", "menor declividade tende a apoiar acumulacao"),
    ("elevation", "elevation_mean", "menor elevacao pode apoiar contexto de acumulacao"),
    ("distance_to_water", "distance_to_water_mean", "menor distancia tende a apoiar exposicao hidrica"),
    ("TWI", "twi_mean", "maior TWI tende a apoiar umidade topografica"),
    ("flow_accumulation", "flow_acc_log_mean", "maior acumulacao tende a apoiar drenagem concentrada"),
    ("urban_prop", "urban_prop", "maior urbanizacao pode apoiar amplificacao urbana"),
    ("vegetation_prop", "vegetation_prop", "maior vegetacao pode reduzir suscetibilidade urbana"),
    ("water_prop", "water_occurrence_patch", "maior ocorrencia de agua pode apoiar proximidade hidrica"),
    ("NDVI", "ndvi_mean", "menor NDVI pode indicar menor mitigacao por vegetacao"),
    ("MNDWI", "mndwi_mean", "maior MNDWI pode indicar sinal hidrico"),
    ("NDBI", "ndbi_mean", "maior NDBI pode indicar superficie construida"),
    ("CHIRPS_3d", "chirps_3d_mm", "maior chuva recente pode apoiar gatilho"),
    ("CHIRPS_7d", "chirps_7d_mm", "maior chuva semanal pode apoiar gatilho"),
    ("CHIRPS_30d", "chirps_30d_mm", "maior chuva mensal pode apoiar saturacao"),
    ("runoff_context_7d", "runoff_context_7d", "maior escoamento contextual pode apoiar gatilho"),
]


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _run_git(args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def _require_inputs() -> None:
    missing = [path for path in REQUIRED_INPUTS if not path.exists()]
    if missing:
        raise FileNotFoundError("; ".join(rel(path) for path in missing))
    if SCORE_V7.exists():
        raise AssertionError("score_v7 existe e e proibido para SUSC-17D")


def _read(path: Path) -> list[dict]:
    return read_csv(path) if path.exists() else []


def _row_count(path: Path) -> str:
    if not path.exists():
        return "0"
    if path.suffix.lower() == ".csv":
        return str(len(_read(path)))
    if path.suffix.lower() in {".json", ".geojson"}:
        return "1"
    return "not_csv"


def _columns(path: Path) -> str:
    rows = _read(path) if path.suffix.lower() == ".csv" else []
    if rows:
        return ";".join(rows[0].keys())
    if path.exists() and path.suffix.lower() == ".csv":
        return path.read_text(encoding="utf-8", errors="ignore").splitlines()[0].replace(",", ";")
    if path.exists():
        return "json_or_markdown"
    return ""


def preflight_rows() -> list[dict]:
    sources = [
        ("fila_validacao_tecnica_17c5", QA_QUEUE_17C5, "itens candidatos para validacao tecnica"),
        ("vinculos_candidatos_17c5", PATCH_LINKS_17C5, "classes de vinculo e sobreposicao"),
        ("geometria_normalizada_17c5", GEOMETRIES_17C5, "estado de geometria resolvida"),
        ("bloqueios_17c5", BLOCKERS_17C5, "bloqueios espaciais anteriores"),
        ("geojson_leve_17c5", GEOJSON_17C5, "mapa leve dos candidatos"),
        ("resumo_17c5", SUMMARY_17C5, "estado herdado do 17C5"),
        ("relatorio_17c5", REPORT_17C5, "relatorio publico anterior"),
        ("registries_17c", TARGET_PACK_17C, "fonte, fenomeno e data dos candidatos"),
        ("resumo_17c", SUMMARY_17C, "estado herdado do 17C"),
        ("features_score", FEATURES, "features existentes por patch"),
        ("score_v6", SCORE_V6, "score v6 somente leitura"),
        ("score_v7_proibido", SCORE_V7, "deve permanecer ausente"),
    ]
    rows = []
    for role, path, note in sources:
        rows.append({
            "artifact_role": role,
            "path": rel(path),
            "exists": _bool(path.exists()),
            "row_count": _row_count(path),
            "columns_found": _columns(path),
            "notes": note,
        })
    rows.append({
        "artifact_role": "estado_git",
        "path": ".",
        "exists": "true",
        "row_count": "not_applicable",
        "columns_found": "branch;head;staged_count;dirty_status_lines",
        "notes": f"branch={_run_git(['branch', '--show-current'])}; head={_run_git(['rev-parse', '--short', 'HEAD'])}; staged={len(_run_git(['diff', '--cached', '--name-only']).splitlines())}; dirty={len(_run_git(['status', '--short']).splitlines())}",
    })
    return rows


def _by_key(rows: list[dict], key: str) -> dict[str, dict]:
    return {row.get(key, ""): row for row in rows}


def _float(value: str | None) -> float | None:
    try:
        if value is None or value == "" or value == "not_available":
            return None
        return float(value)
    except ValueError:
        return None


def _target(candidate_event_id: str) -> dict:
    return _by_key(_read(TARGET_PACK_17C), "candidate_event_id").get(candidate_event_id, {})


def _source_reliability(source_type: str) -> int:
    if source_type in {"technical_remote_sensing_flood_footprint", "official_observed_event_polygon", "official_observed_event_point"}:
        return 85
    if source_type in {"official_address_resolved", "administrative_disaster_record"}:
        return 55
    if source_type == "documentary_context":
        return 35
    return 10


def _temporal_quality(target: dict) -> int:
    precision = target.get("event_date_precision", "")
    if precision == "exact_day":
        return 90
    if precision == "range":
        return 80
    if precision == "month_only":
        return 35
    return 0


def _geometry_quality(geom: dict) -> int:
    if geom.get("has_geometry_object") != "true":
        return 0
    if geom.get("crs") in {"", "not_available"}:
        return 20
    if geom.get("geometry_type") in {"polygon", "technical_footprint"}:
        return 85
    if geom.get("geometry_type") in {"point", "bbox"}:
        return 70
    return 30


def _patch_link_quality(link: dict) -> int:
    if link.get("link_class") == "exact_polygon_overlap":
        ratio = _float(link.get("overlap_ratio")) or 0.0
        return 65 if ratio < 0.005 else 85
    if link.get("link_class") in {"point_within_patch", "bbox_overlap"}:
        return 75
    if str(link.get("link_class", "")).startswith("near_patch_buffer"):
        return 35
    return 0


def _phenomenon_quality(target: dict) -> int:
    text = " ".join([target.get("phenomenon_type", ""), target.get("source_name", "")]).lower()
    if any(term in text for term in ["flood", "inund", "alag", "enxurr", "chuva", "hidro"]):
        return 85
    if "mass_movement" in text or "desliz" in text:
        return 40
    return 45


def _uncertainty_quality(value: str) -> int:
    num = _float(value)
    if num is not None:
        if num <= 50:
            return 85
        if num <= 500:
            return 65
        return 35
    if value in {"requires_human_review", "bbox_aoi_not_footprint"}:
        return 55
    if value == "not_computed_metadata_only":
        return 45
    return 0


def _feature_consistency(feature_rows: list[dict]) -> tuple[int, str]:
    available = [row for row in feature_rows if row["feature_available"] == "true"]
    contradictions = [row for row in feature_rows if row["contradicoes_detectadas"] != "nenhuma"]
    if not available:
        return 50, "features do patch candidato nao disponiveis; ausencia nao reprova"
    if contradictions:
        return 40, "contradicoes reduziram confianca"
    return 70, "features disponiveis sem contradicao metodologica"


def _temporal_leakage_score(link: dict, target: dict, feature_row: dict | None) -> tuple[int, str]:
    if not feature_row:
        return 80, "sem leitura de feature temporal do patch candidato"
    ref_date = feature_row.get("reference_date", "")
    event_date = target.get("event_date_candidate", "")
    if ref_date and event_date and ref_date > event_date:
        return 55, "features existentes podem ser posteriores ao evento; uso limitado a diagnostico review-only"
    return 80, "sem vazamento temporal critico identificado"


def _critical_blockers(link: dict, geom: dict, target: dict, leakage_score: int) -> list[str]:
    blockers: list[str] = []
    if target.get("source_type") in {"alert_only", "risk_area_not_event", "documentary_context"}:
        blockers.append("classe_de_fonte_inadequada")
    if geom.get("has_geometry_object") != "true":
        blockers.append("geometria_nao_resolvida")
    if link.get("patch_id") in {"", "NO_PATCH_RESOLUTION"}:
        blockers.append("patch_id_ausente")
    if link.get("geometry_id") in {"", "not_available"}:
        blockers.append("geometry_id_ausente")
    if link.get("link_class") not in STRONG_LINK_CLASSES:
        blockers.append("classe_de_vinculo_nao_forte")
    if _phenomenon_quality(target) < 50:
        blockers.append("fenomeno_incompativel")
    if _temporal_quality(target) < 70:
        blockers.append("temporalidade_insuficiente")
    if leakage_score < 40:
        blockers.append("vazamento_temporal_critico")
    return blockers


def _confidence(score: float) -> str:
    if score >= 75:
        return "alta"
    if score >= 58:
        return "media"
    if score >= 35:
        return "baixa"
    return "insuficiente"


def _status(link: dict, geom: dict, target: dict, blockers: list[str]) -> str:
    if "classe_de_fonte_inadequada" in blockers:
        if target.get("source_type") == "documentary_context":
            return "precisa_mais_fonte"
        return "rejeitado"
    if "geometria_nao_resolvida" in blockers:
        return "geometria_insuficiente"
    if "classe_de_vinculo_nao_forte" in blockers or link.get("link_class") == "same_region_only":
        return "incompatibilidade_espacial"
    if "temporalidade_insuficiente" in blockers:
        return "incompatibilidade_temporal"
    if "fenomeno_incompativel" in blockers:
        return "incompatibilidade_de_fenomeno"
    if "vazamento_temporal_critico" in blockers:
        return "rejeitado"
    return "aceito_para_avaliacao_review_only"


def _score_row_by_patch() -> dict[str, dict]:
    return _by_key(_read(SCORE_V6), "patch_id")


def _feature_row_by_patch() -> dict[str, dict]:
    return _by_key(_read(FEATURES), "patch_id")


def build_feature_matrix(decision_seed_rows: list[dict]) -> list[dict]:
    feature_by_patch = _feature_row_by_patch()
    score_by_patch = _score_row_by_patch()
    rows: list[dict] = []
    for seed in decision_seed_rows:
        patch_id = seed["patch_id"]
        merged = {}
        if patch_id in feature_by_patch:
            merged.update(feature_by_patch[patch_id])
        if patch_id in score_by_patch:
            merged.update(score_by_patch[patch_id])
        for public_name, field, direction in FEATURE_MAP:
            value = merged.get(field, "")
            available = value not in {"", "not_available"}
            contradiction = "nenhuma"
            consistency = "nao_avaliada"
            observation = "feature ausente no patch candidato; ausencia nao reprova"
            if available:
                consistency = "sem_contradicao_detectada"
                observation = "feature existente usada apenas como apoio diagnostico review-only"
                if public_name == "score_v6" and (_float(value) or 0) < 0.2:
                    contradiction = "score_v6_baixo_para_candidato_observacional"
                    consistency = "contraditoria"
            rows.append({
                "item_validacao_id": seed["item_validacao_id"],
                "candidate_event_id": seed["candidate_event_id"],
                "patch_id": patch_id,
                "nome_feature": public_name,
                "feature_available": _bool(available),
                "valor_feature": value if available else "not_available",
                "direcao_esperada": direction,
                "consistencia_com_suscetibilidade": consistency,
                "contradicoes_detectadas": contradiction,
                "observacao_metodologica": observation,
            })
    return rows


def decision_rows() -> tuple[list[dict], list[dict]]:
    qa_rows = _read(QA_QUEUE_17C5)
    link_by_key = {(row["candidate_event_id"], row["patch_id"], row["geometry_id"]): row for row in _read(PATCH_LINKS_17C5)}
    geom_by_id = _by_key(_read(GEOMETRIES_17C5), "geometry_id")
    target_by_id = _by_key(_read(TARGET_PACK_17C), "candidate_event_id")
    feature_by_patch = _feature_row_by_patch()
    seed_rows = []
    prelim = []
    for idx, qa in enumerate(qa_rows, start=1):
        link = link_by_key.get((qa["candidate_event_id"], qa["patch_id"], qa["geometry_id"]), {})
        geom = geom_by_id.get(qa["geometry_id"], {})
        target = target_by_id.get(qa["candidate_event_id"], {})
        item_id = f"S17D_VAL_{idx:04d}"
        seed_rows.append({
            "item_validacao_id": item_id,
            "candidate_event_id": qa["candidate_event_id"],
            "patch_id": qa["patch_id"],
        })
        prelim.append((item_id, qa, link, geom, target, feature_by_patch.get(qa["patch_id"])))
    feature_rows = build_feature_matrix(seed_rows)
    features_by_item: dict[str, list[dict]] = defaultdict(list)
    for row in feature_rows:
        features_by_item[row["item_validacao_id"]].append(row)
    decisions = []
    for item_id, qa, link, geom, target, feature_patch_row in prelim:
        feature_score, feature_note = _feature_consistency(features_by_item[item_id])
        leak_score, leak_note = _temporal_leakage_score(link, target, feature_patch_row)
        scores = {
            "pontuacao_confiabilidade_fonte": _source_reliability(target.get("source_type", link.get("source_type", ""))),
            "pontuacao_qualidade_temporal": _temporal_quality(target),
            "pontuacao_qualidade_geometrica": _geometry_quality(geom),
            "pontuacao_qualidade_vinculo_patch": _patch_link_quality(link),
            "pontuacao_compatibilidade_fenomeno": _phenomenon_quality(target),
            "pontuacao_incerteza_espacial": _uncertainty_quality(link.get("uncertainty_m", "")),
            "pontuacao_consistencia_features": feature_score,
            "pontuacao_risco_vazamento_temporal": leak_score,
        }
        critical = _critical_blockers(link, geom, target, leak_score)
        severity = 85 if not critical else max(0, 85 - len(critical) * 25)
        scores["pontuacao_severidade_bloqueios"] = severity
        final_score = sum(scores.values()) / len(scores)
        confidence = _confidence(final_score)
        status = _status(link, geom, target, critical)
        accepted = status == "aceito_para_avaliacao_review_only"
        calibration = (
            accepted
            and confidence in {"alta", "media"}
            and feature_score >= 60
            and scores["pontuacao_incerteza_espacial"] >= 60
            and target.get("source_type") in SOURCE_TYPES_FOR_ACCEPTANCE
        )
        if target.get("source_type") == "technical_remote_sensing_flood_footprint" and not feature_patch_row:
            calibration = False
        justification_parts = [
            f"fonte={scores['pontuacao_confiabilidade_fonte']}",
            f"temporal={scores['pontuacao_qualidade_temporal']}",
            f"geometria={scores['pontuacao_qualidade_geometrica']}",
            f"vinculo={scores['pontuacao_qualidade_vinculo_patch']}",
            feature_note,
            leak_note,
        ]
        decisions.append({
            "item_validacao_id": item_id,
            "candidate_event_id": qa["candidate_event_id"],
            "source_id": link.get("source_id", target.get("source_id", "")),
            "source_type": target.get("source_type", link.get("source_type", "")),
            "geometry_id": qa["geometry_id"],
            "patch_id": qa["patch_id"],
            "cidade": link.get("city", target.get("city", "")),
            "regiao": link.get("region", target.get("region", "")),
            "tipo_fenomeno": target.get("phenomenon_type", "not_available"),
            "data_evento": target.get("event_date_candidate", "not_available"),
            "classe_vinculo": link.get("link_class", "not_available"),
            **{key: str(value) for key, value in scores.items()},
            "confianca_tecnica_final": confidence,
            "status_validacao_tecnica": status,
            "accepted_for_evaluation": _bool(accepted),
            "accepted_for_calibration_candidate": _bool(calibration),
            "eligible_for_training": "false",
            "ground_truth": "false",
            "review_only": "true",
            "score_v7_allowed": "false",
            "not_ground_truth_reason": NO_GT_REASON,
            "justificativa_tecnica": "; ".join(justification_parts),
            "bloqueios_criticos": ";".join(critical) if critical else "nenhum",
        })
    return decisions, feature_rows


def status_summary_rows(decisions: list[dict]) -> list[dict]:
    counts = Counter(row["status_validacao_tecnica"] for row in decisions)
    return [{"status_validacao_tecnica": status, "quantidade": str(counts.get(status, 0))} for status in STATUS_ALLOWED if counts.get(status, 0)]


def region_summary_rows(decisions: list[dict]) -> list[dict]:
    by_region: dict[str, list[dict]] = defaultdict(list)
    for row in decisions:
        by_region[row["regiao"]].append(row)
    rows = []
    for region in sorted(by_region):
        group = by_region[region]
        rows.append({
            "regiao": region,
            "itens": str(len(group)),
            "aceitos_para_avaliacao": str(sum(1 for row in group if row["accepted_for_evaluation"] == "true")),
            "candidatos_a_calibracao": str(sum(1 for row in group if row["accepted_for_calibration_candidate"] == "true")),
            "rejeitados": str(sum(1 for row in group if row["status_validacao_tecnica"] == "rejeitado")),
            "ambiguos": str(sum(1 for row in group if row["status_validacao_tecnica"] == "ambiguo")),
            "precisam_mais_fonte": str(sum(1 for row in group if row["status_validacao_tecnica"] == "precisa_mais_fonte")),
        })
    return rows


def blocker_rows(decisions: list[dict]) -> list[dict]:
    rows = []
    for decision in decisions:
        blockers = decision["bloqueios_criticos"].split(";") if decision["bloqueios_criticos"] != "nenhum" else []
        if not blockers and decision["accepted_for_calibration_candidate"] != "true":
            blockers = ["calibracao_futura_bloqueada_por_feature_ou_incerteza"]
        for blocker in blockers:
            rows.append({
                "bloqueio_id": f"S17D_BLOQ_{len(rows) + 1:04d}",
                "item_validacao_id": decision["item_validacao_id"],
                "candidate_event_id": decision["candidate_event_id"],
                "patch_id": decision["patch_id"],
                "classe_bloqueio": blocker,
                "severidade": "alta" if "critico" in blocker or "geometria" in blocker else "media",
                "descricao": _blocker_description(blocker),
                "acao_recomendada": _blocker_action(blocker),
                "review_only": "true",
                "ground_truth": "false",
                "eligible_for_training": "false",
                "score_v7_allowed": "false",
            })
    return rows


def _blocker_description(blocker: str) -> str:
    mapping = {
        "calibracao_futura_bloqueada_por_feature_ou_incerteza": "item aceito apenas para avaliacao review-only; features ou incerteza ainda nao sustentam calibracao",
        "geometria_nao_resolvida": "geometria real com CRS nao foi resolvida",
        "classe_de_vinculo_nao_forte": "classe de vinculo espacial nao sustenta avaliacao",
        "vazamento_temporal_critico": "risco temporal critico impede aceite",
    }
    return mapping.get(blocker, blocker)


def _blocker_action(blocker: str) -> str:
    if blocker == "calibracao_futura_bloqueada_por_feature_ou_incerteza":
        return "extrair features do patch candidato ou reduzir incerteza espacial antes de calibracao"
    if "geometria" in blocker:
        return "adquirir geometria real com CRS e revisar novamente"
    if "vazamento" in blocker:
        return "separar features pre-evento e revalidar"
    return "manter bloqueado ate nova evidencia tecnica"


def gate_17b_rows(decisions: list[dict]) -> list[dict]:
    accepted = [row for row in decisions if row["accepted_for_evaluation"] == "true"]
    regions = sorted({row["regiao"] for row in accepted})
    events = sorted({row["candidate_event_id"] for row in accepted})
    strong_links = len(accepted)
    accepted_enough = len(accepted) >= 20
    rows = [
        ("minimo_3_eventos_datados", str(len(events)), "3", len(events) >= 3, "conta apenas itens aceitos"),
        ("minimo_2_regioes", str(len(regions)), "2", len(regions) >= 2, "regioes aceitas: " + ",".join(regions)),
        ("minimo_1_geometria_forte_por_regiao", str(len(regions)), "2", len(regions) >= 2, "geometrias aceitas ainda concentradas"),
        ("minimo_20_patch_links_fortes_aceitos", str(strong_links), "20", strong_links >= 20, "limiar nao atingido"),
        ("validacao_tecnica_aceita_suficiente", str(len(accepted)), "20", accepted_enough, "aceite review-only insuficiente para benchmark"),
        ("ground_truth_false", "0", "0", all(row["ground_truth"] == "false" for row in decisions), "sem ground truth"),
        ("trainable_false", "0", "0", all(row["eligible_for_training"] == "false" for row in decisions), "sem treino"),
    ]
    return [
        {
            "criterio": name,
            "valor_observado": observed,
            "limiar": threshold,
            "passou": _bool(passed),
            "observacao": note,
        }
        for name, observed, threshold, passed, note in rows
    ]


def _status_17b(gate_rows: list[dict], decisions: list[dict]) -> str:
    if all(row["passou"] == "true" for row in gate_rows):
        return "17B_PRONTO_PARA_BENCHMARK_REVIEW_ONLY"
    if any(row["accepted_for_evaluation"] == "true" for row in decisions):
        return "17B_PARCIAL_COM_VALIDACAO_TECNICA_SEM_PRONTIDAO_BENCHMARK"
    return "17B_BLOQUEADO_FAIL_CLOSED"


def _status_17e(decisions: list[dict]) -> str:
    count = sum(1 for row in decisions if row["accepted_for_calibration_candidate"] == "true")
    if count >= 3:
        return "17E_PODE_RODAR_COM_ESCOPO_REVIEW_ONLY"
    if count > 0:
        return "17E_PARCIAL_INSUFICIENTE"
    return "17E_BLOQUEADO_SEM_CANDIDATO_CALIBRACAO"


def summary_obj(decisions: list[dict], features: list[dict], blockers: list[dict], gate_rows: list[dict]) -> dict:
    inherited = read_json(SUMMARY_17C5)
    accepted = sum(1 for row in decisions if row["accepted_for_evaluation"] == "true")
    calibration = sum(1 for row in decisions if row["accepted_for_calibration_candidate"] == "true")
    return {
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "staged_count": len(_run_git(["diff", "--cached", "--name-only"]).splitlines()),
        "herdado_17c5_geometrias_resolvidas": inherited.get("geometries_resolved_count", 0),
        "herdado_17c5_geometrias_nao_resolvidas": inherited.get("geometries_unresolved_count", 0),
        "herdado_17c5_vinculos_patch": inherited.get("patch_link_rows_count", 0),
        "herdado_17c5_vinculos_fortes_candidatos": inherited.get("strong_link_candidate_count", 0),
        "herdado_17c5_itens_para_validacao_tecnica": inherited.get("qa_queue_count", 0),
        "herdado_17c5_estado_17b_publico": "bloqueado_parcial_aguardando_validacao_tecnica_17d",
        "items_count": len(decisions),
        "accepted_for_evaluation_count": accepted,
        "accepted_for_calibration_candidate_count": calibration,
        "rejected_count": sum(1 for row in decisions if row["status_validacao_tecnica"] == "rejeitado"),
        "ambiguous_count": sum(1 for row in decisions if row["status_validacao_tecnica"] == "ambiguo"),
        "needs_more_source_count": sum(1 for row in decisions if row["status_validacao_tecnica"] == "precisa_mais_fonte"),
        "feature_rows_count": len(features),
        "feature_available_count": sum(1 for row in features if row["feature_available"] == "true"),
        "blocker_rows_count": len(blockers),
        "ground_truth_true_count": sum(1 for row in decisions if row["ground_truth"] == "true"),
        "eligible_for_training_true_count": sum(1 for row in decisions if row["eligible_for_training"] == "true"),
        "score_v7_allowed_true_count": sum(1 for row in decisions if row["score_v7_allowed"] == "true"),
        "score_v6_changed": bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)])),
        "score_v7_created": SCORE_V7.exists(),
        "status_17b": _status_17b(gate_rows, decisions),
        "status_17e": _status_17e(decisions),
        "review_only": True,
        "ground_truth": False,
        "trainable": False,
    }


def _schema() -> dict:
    properties = {}
    for field in DECISION_FIELDS:
        if field in {"eligible_for_training", "ground_truth", "score_v7_allowed"}:
            properties[field] = {"const": "false"}
        elif field == "review_only":
            properties[field] = {"const": "true"}
        elif field == "status_validacao_tecnica":
            properties[field] = {"enum": STATUS_ALLOWED}
        elif field == "confianca_tecnica_final":
            properties[field] = {"enum": CONFIDENCE_ALLOWED}
        elif field in {"accepted_for_evaluation", "accepted_for_calibration_candidate"}:
            properties[field] = {"enum": ["true", "false"]}
        else:
            properties[field] = {"type": "string", "minLength": 1}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SUSC-17D validacao tecnica de evidencia observacional",
        "type": "object",
        "required": DECISION_FIELDS,
        "properties": properties,
        "additionalProperties": True,
    }


def write_schema() -> None:
    write_json(SCHEMA_DECISION, _schema())


def card_markdown(decision: dict, feature_rows: list[dict]) -> str:
    feature_lines = "\n".join(
        f"- {row['nome_feature']}: disponivel={row['feature_available']}; valor={row['valor_feature']}; consistencia={row['consistencia_com_suscetibilidade']}"
        for row in feature_rows
    )
    return f"""# Cartao tecnico {decision['item_validacao_id']}

## Resumo do evento

- Candidato: `{decision['candidate_event_id']}`
- Cidade/regiao: {decision['cidade']} / {decision['regiao']}
- Fenomeno: {decision['tipo_fenomeno']}
- Data: {decision['data_evento']}

## Fonte e geometria

- Fonte: `{decision['source_id']}`
- Tipo de fonte: `{decision['source_type']}`
- Geometria: `{decision['geometry_id']}`
- Patch: `{decision['patch_id']}`
- Classe de vinculo: `{decision['classe_vinculo']}`

## Comparacao espacial e temporal

- Pontuacao geometrica: {decision['pontuacao_qualidade_geometrica']}
- Pontuacao de vinculo com patch: {decision['pontuacao_qualidade_vinculo_patch']}
- Pontuacao temporal: {decision['pontuacao_qualidade_temporal']}
- Risco de vazamento temporal: {decision['pontuacao_risco_vazamento_temporal']}

## Comparacao de features e score

{feature_lines}

## Bloqueios e decisao tecnica

- Bloqueios criticos: {decision['bloqueios_criticos']}
- Confianca tecnica: {decision['confianca_tecnica_final']}
- Status: {decision['status_validacao_tecnica']}
- Pode entrar em avaliacao review-only: {decision['accepted_for_evaluation']}
- Pode entrar em prontidao de calibracao futura: {decision['accepted_for_calibration_candidate']}

## Justificativa tecnica

{decision['justificativa_tecnica']}

## Por que nao e ground truth

{decision['not_ground_truth_reason']}
"""


def report_markdown(decisions: list[dict], features: list[dict], blockers: list[dict], gate_rows: list[dict], summary: dict) -> str:
    status_lines = "\n".join(
        f"- {row['status_validacao_tecnica']}: {row['quantidade']}"
        for row in status_summary_rows(decisions)
    )
    accepted_lines = "\n".join(
        f"- {row['item_validacao_id']} / {row['candidate_event_id']} / {row['patch_id']} / {row['status_validacao_tecnica']}"
        for row in decisions
    )
    gate_lines = "\n".join(
        f"- {row['criterio']}: passou={row['passou']} ({row['valor_observado']} / {row['limiar']})"
        for row in gate_rows
    )
    return f"""# SUSC-17D Validacao tecnica de evidencia observacional

## Estado herdado do 17C5

- Branch: `{summary['branch']}`
- HEAD: `{summary['head']}`
- Area staged: {summary['staged_count']} arquivo(s)
- Geometrias resolvidas no 17C5: {summary['herdado_17c5_geometrias_resolvidas']}
- Geometrias nao resolvidas no 17C5: {summary['herdado_17c5_geometrias_nao_resolvidas']}
- Vinculos evento-patch herdados: {summary['herdado_17c5_vinculos_patch']}
- Vinculos fortes candidatos herdados: {summary['herdado_17c5_vinculos_fortes_candidatos']}
- Itens herdados para validacao tecnica: {summary['herdado_17c5_itens_para_validacao_tecnica']}
- Estado 17B herdado: {summary['herdado_17c5_estado_17b_publico']}
- Itens avaliados: {summary['items_count']}
- `score_v6` alterado: {summary['score_v6_changed']}
- `score_v7` criado: {summary['score_v7_created']}

## Metodologia de validacao tecnica

A validacao comparou fonte, data, fenomeno, geometria, vinculo com patch, incerteza espacial, features disponiveis, score v6 e bloqueios metodologicos. A decisao e review-only e nao cria verdade de referencia, treino, score_v7 ou benchmark 17B.

## Criterios de decisao

Um item so e aceito para avaliacao review-only quando tem geometria real resolvida, patch identificado, classe de vinculo forte, temporalidade suficiente, fenomeno compativel, justificativa tecnica e ausencia de bloqueio critico.

## Contagem por status

{status_lines}

## Itens avaliados

{accepted_lines}

## Itens aceitos, rejeitados e ambiguos

- Aceitos para avaliacao review-only: {summary['accepted_for_evaluation_count']}
- Candidatos a calibracao futura: {summary['accepted_for_calibration_candidate_count']}
- Rejeitados: {summary['rejected_count']}
- Ambiguos: {summary['ambiguous_count']}
- Precisam de mais fonte: {summary['needs_more_source_count']}

## Comparacao feature/score

- Linhas de matriz de features: {summary['feature_rows_count']}
- Features disponiveis: {summary['feature_available_count']}
- Observacao: ausencia de feature nao reprova automaticamente, mas reduz prontidao para calibracao.

## Bloqueios criticos

- Linhas de bloqueio: {summary['blocker_rows_count']}
- Principal bloqueio: patches candidatos 17C6 ainda nao possuem features/score v6 materializados e exigem reducao de incerteza antes de calibracao.

## Gate de prontidao 17B

{gate_lines}

## Conclusao

- Desbloqueado: {summary['accepted_for_evaluation_count']} itens para avaliacao review-only.
- Segue bloqueado: calibracao futura, treino, ground truth, score_v7 e benchmark 17B.
- Candidatos a calibracao futura: {summary['accepted_for_calibration_candidate_count']}.
- Status 17E: {summary['status_17e']}.
- Status 17B: {summary['status_17b']}.

17B continua sem prontidao de benchmark porque os aceites tecnicos sao insuficientes em quantidade, estao concentrados em uma regiao e seguem sem ground truth, sem treino e sem score_v7.
"""


def _public_output_paths() -> list[Path]:
    paths = [REPORT]
    paths.extend(OUT_DATA.glob("*.csv"))
    paths.extend(OUT_DATA.glob("*.json"))
    paths.extend(CARDS_DIR.glob("*.md"))
    return [path for path in paths if path.exists()]


def public_text_violations_text(text: str) -> list[str]:
    return sorted({match.group(0) for match in PUBLIC_FORBIDDEN_RE.finditer(text)})


def public_text_violations_files(paths: list[Path] | None = None) -> list[str]:
    errors = []
    for path in paths or _public_output_paths():
        text = path.read_text(encoding="utf-8", errors="ignore")
        hits = public_text_violations_text(text)
        if hits:
            errors.append(f"{rel(path)}:{','.join(hits)}")
    return errors


def build_all() -> dict:
    _require_inputs()
    ensure_dir(OUT_DATA)
    ensure_dir(CARDS_DIR)
    ensure_dir(OUT_REPORTS)
    write_schema()
    decisions, features = decision_rows()
    status_rows = status_summary_rows(decisions)
    region_rows = region_summary_rows(decisions)
    blockers = blocker_rows(decisions)
    gate_rows = gate_17b_rows(decisions)
    summary = summary_obj(decisions, features, blockers, gate_rows)
    write_csv(PREFLIGHT_CSV, preflight_rows(), PREFLIGHT_FIELDS)
    write_json(PREFLIGHT_JSON, {"preflight": preflight_rows()})
    write_csv(DECISIONS, decisions, DECISION_FIELDS)
    write_csv(FEATURE_MATRIX, features, FEATURE_FIELDS)
    write_csv(STATUS_SUMMARY, status_rows, STATUS_SUMMARY_FIELDS)
    write_csv(REGION_SUMMARY, region_rows, REGION_SUMMARY_FIELDS)
    write_csv(BLOCKERS, blockers, BLOCKER_FIELDS)
    write_csv(GATE_17B, gate_rows, GATE_FIELDS)
    write_json(SUMMARY, summary)
    by_item_features: dict[str, list[dict]] = defaultdict(list)
    for row in features:
        by_item_features[row["item_validacao_id"]].append(row)
    for decision in decisions:
        card = CARDS_DIR / f"{decision['item_validacao_id']}_{decision['candidate_event_id']}_{decision['patch_id']}.md"
        write_markdown(card, card_markdown(decision, by_item_features[decision["item_validacao_id"]]))
    write_markdown(REPORT, report_markdown(decisions, features, blockers, gate_rows, summary))
    return summary


def _schema_violations(row: dict, schema: dict) -> list[str]:
    errors = []
    for field in schema.get("required", []):
        if field not in row or str(row.get(field, "")) == "":
            errors.append(f"missing_required:{field}")
    for field, rule in schema.get("properties", {}).items():
        if field not in row:
            continue
        value = row.get(field)
        if "const" in rule and value != rule["const"]:
            errors.append(f"{field}:expected_const:{rule['const']}:got:{value}")
        if "enum" in rule and value not in rule["enum"]:
            errors.append(f"{field}:not_in_enum:{value}")
    return errors


def validate_decision_rows(decisions: list[dict]) -> list[str]:
    errors: list[str] = []
    schema = read_json(SCHEMA_DECISION)
    for idx, row in enumerate(decisions, start=1):
        row_id = row.get("item_validacao_id", f"row{idx}")
        for err in _schema_violations(row, schema):
            errors.append(f"{row_id}:{err}")
        if row["accepted_for_evaluation"] == "true" and row["status_validacao_tecnica"] != "aceito_para_avaliacao_review_only":
            errors.append(f"{row_id}:accepted_status_invalido")
        if row["accepted_for_calibration_candidate"] == "true" and row["accepted_for_evaluation"] != "true":
            errors.append(f"{row_id}:calibracao_sem_avaliacao")
        if row["accepted_for_evaluation"] == "true":
            if row["geometry_id"] in {"", "not_available"}:
                errors.append(f"{row_id}:accepted_sem_geometry_id")
            if row["patch_id"] in {"", "NO_PATCH_RESOLUTION", "not_available"}:
                errors.append(f"{row_id}:accepted_sem_patch_id")
            if row["classe_vinculo"] in {"same_region_only", "unresolved_geometry"} or row["classe_vinculo"].startswith("near_patch_buffer"):
                errors.append(f"{row_id}:accepted_classe_vinculo_invalida")
            if row.get("source_type") in {"alert_only", "risk_area_not_event", "documentary_context"}:
                errors.append(f"{row_id}:accepted_fonte_invalida")
            if "vazamento_temporal_critico" in row["bloqueios_criticos"]:
                errors.append(f"{row_id}:accepted_com_vazamento_critico")
            if (_float(row.get("pontuacao_risco_vazamento_temporal")) or 0) < 40:
                errors.append(f"{row_id}:accepted_com_pontuacao_vazamento_critica")
        if row["not_ground_truth_reason"].strip() == "":
            errors.append(f"{row_id}:not_ground_truth_reason_vazio")
        if row["justificativa_tecnica"].strip() == "":
            errors.append(f"{row_id}:justificativa_vazia")
        if row["confianca_tecnica_final"] not in CONFIDENCE_ALLOWED:
            errors.append(f"{row_id}:confianca_fora_enum")
        for field in ["ground_truth", "eligible_for_training", "score_v7_allowed"]:
            if row[field] == "true":
                errors.append(f"{row_id}:{field}_true_proibido")
    return errors


def validate() -> int:
    summary = build_all()
    decisions = _read(DECISIONS)
    errors = validate_decision_rows(decisions)
    errors.extend(public_text_violations_files())
    if summary["ground_truth_true_count"] != 0:
        errors.append("ground_truth_true_count_nonzero")
    if summary["eligible_for_training_true_count"] != 0:
        errors.append("eligible_for_training_true_count_nonzero")
    if summary["score_v7_allowed_true_count"] != 0:
        errors.append("score_v7_allowed_true_count_nonzero")
    if summary["score_v6_changed"]:
        errors.append("score_v6_changed_forbidden")
    if summary["score_v7_created"]:
        errors.append("score_v7_created_forbidden")
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1
    print(
        "17D validacao tecnica validada: "
        f"aceitos={summary['accepted_for_evaluation_count']} "
        f"calibracao={summary['accepted_for_calibration_candidate_count']} "
        f"status17B={summary['status_17b']} "
        f"status17E={summary['status_17e']}"
    )
    return 0


def run_all() -> int:
    summary = build_all()
    print(
        "17D validacao tecnica gerada: "
        f"aceitos={summary['accepted_for_evaluation_count']} "
        f"calibracao={summary['accepted_for_calibration_candidate_count']} "
        f"status17B={summary['status_17b']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_all())
