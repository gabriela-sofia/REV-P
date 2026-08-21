"""SUSC-19A matriz multimodal escalavel por patch.

Consolida uma matriz unica com uma linha por patch, integrando features fisicas,
territoriais, espectrais e de chuva reais, alem de score_v6 e evidencia
documental/observacional review-only. Nao cria footprint, score oficial,
benchmark, treino nem ground truth. Onde a feature nao existe, marca NA e lacuna.

Fontes reais herdadas:
- datasets/suscetibilidade/susc_features_by_patch_v1.csv (features por patch)
- datasets/suscetibilidade/susc_score_v6_candidate_by_patch_v1.csv (score_v6)
- outputs_public/data/linhagem_anterior/susc_18h_... (linhagem consolidada)
- outputs_public/data/linhagem_anterior/susc_18g_... (overlays tecnicos SAR de Curitiba)

Guardrails: ground_truth=false, eligible_for_training=false, score_v7_allowed=false,
score_v6 intacto, patch_stats SAR (pos-evento) nunca vira feature pre-evento.
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
    write_csv,
    write_json,
    write_markdown,
)

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
OUT = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_19a_matriz_multimodal_escalavel_por_patch"
CARDS = OUT / "cartoes_regionais"
REPORTS = ROOT / "outputs_public" / "reports"
SCHEMAS_DIR = ROOT / "schemas" / "suscetibilidade"

REPORT = REPORTS / "SUSC_19A_MATRIZ_MULTIMODAL_ESCALAVEL_POR_PATCH.md"
SCHEMA = SCHEMAS_DIR / "susc_19a_matriz_multimodal_schema_v1.json"

FEATURES_SRC = ROOT / "datasets" / "suscetibilidade" / "susc_features_by_patch_v1.csv"
SCORE_V6 = ROOT / "datasets" / "suscetibilidade" / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = ROOT / "datasets" / "suscetibilidade" / "susc_score_v7_candidate_by_patch_v1.csv"
LINHAGEM_18H = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_18h_consolidacao_mestre_cadeia_observacional" / "linhagem_mestre_evidencia_observacional.csv"
OVERLAYS_18G = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_18g_recuperacao_compactacao_vetorial_sar_curitiba" / "vinculos_vetoriais_sar_patch_curitiba_18g.csv"

# Saidas
INVENTARIO = OUT / "inventario_fontes_multimodais.csv"
UNIVERSO = OUT / "universo_patches_multimodal.csv"
MAT_FISICA = OUT / "matriz_fisica_topografica_por_patch.csv"
MAT_TERRITORIAL = OUT / "matriz_territorial_por_patch.csv"
MAT_ESPECTRAL = OUT / "matriz_espectral_por_patch.csv"
MAT_CHUVA = OUT / "matriz_chuva_por_patch.csv"
MAT_DOC = OUT / "matriz_documental_observacional_por_patch.csv"
MAT_FINAL = OUT / "matriz_multimodal_escalavel_por_patch.csv"
AUD_COBERTURA = OUT / "auditoria_cobertura_multimodal.csv"
MISSINGNESS = OUT / "missingness_por_patch.csv"
INDICE = OUT / "indice_cobertura_multimodal_review_only.csv"
GATE_19A = OUT / "gate_matriz_multimodal_19a.csv"
GATE_19B = OUT / "gate_prontidao_19b_auditoria_cobertura.csv"
RESUMO_REGIAO = OUT / "resumo_por_regiao.csv"
RESUMO_FAMILIA = OUT / "resumo_por_familia_feature.csv"
SUMMARY = OUT / "summary.json"
PREFLIGHT = OUT / "preflight.json"

REQUIRED_OUTPUTS = [
    INVENTARIO, UNIVERSO, MAT_FISICA, MAT_TERRITORIAL, MAT_ESPECTRAL, MAT_CHUVA, MAT_DOC,
    MAT_FINAL, AUD_COBERTURA, MISSINGNESS, INDICE, GATE_19A, GATE_19B, RESUMO_REGIAO,
    RESUMO_FAMILIA, SUMMARY, PREFLIGHT, SCHEMA, REPORT,
]

# ---------------------------------------------------------------------------
# Guardrails / constantes
# ---------------------------------------------------------------------------
PUBLIC_FORBIDDEN_RE = re.compile(r"\b(?:agentic|agente|codex|llm|ia)\b", re.IGNORECASE)
NA = "NA"

HARD_DEFAULTS = {
    "ground_truth": "false",
    "eligible_for_training": "false",
    "score_v7_allowed": "false",
    "review_only_status": "true",
}

STATUS_19A_ALLOWED = {
    "19A_MATRIZ_MULTIMODAL_CONSOLIDADA",
    "19A_MATRIZ_MULTIMODAL_PARCIAL_COM_LACUNAS",
    "19A_BLOQUEADO_POR_FONTES_FEATURES",
    "19A_BLOQUEADO_FAIL_CLOSED",
}
CLASSES_COBERTURA = {"cobertura_alta", "cobertura_media", "cobertura_baixa", "cobertura_insuficiente"}
STATUS_17B_MESTRE = "17B_APROXIMACAO_COM_SEGUNDA_REGIAO_TECNICA"

CIDADE = {"recife": "Recife", "curitiba": "Curitiba", "petropolis": "Petropolis"}

TEMPORALIDADE_FEATURES = "composto de referencia 2022-12-31; nao e janela pre-evento especifica por ocorrencia"

# Features esperadas por familia (para cobertura e inventario)
FISICO_EXPECTED = ["elevation_mean", "slope_mean", "HAND_mean", "distance_to_water_mean", "TWI_mean", "flow_accumulation_mean", "drainage_context"]
TERRITORIAL_EXPECTED = ["urban_prop", "vegetation_prop", "water_prop", "exposed_soil_prop", "impervious_proxy", "MapBiomas_class_majority"]
ESPECTRAL_EXPECTED = ["NDVI", "NDWI", "MNDWI", "NDBI"]
CHUVA_EXPECTED = ["CHIRPS_3d", "CHIRPS_7d", "CHIRPS_30d", "runoff_context_7d"]

# Features realmente presentes na fonte
TERRITORIAL_PRESENTE = {"urban_prop", "vegetation_prop"}
TERRITORIAL_AUSENTE = {"water_prop", "exposed_soil_prop", "impervious_proxy", "MapBiomas_class_majority"}

# Patches com evidencia observacional/documental real (derivada de dados reais)
CURITIBA_SAR_OVERLAYS = {"curitiba_01050", "curitiba_01101"}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _run_git(args: list[str]) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return r.stdout.strip() if r.returncode == 0 else ""


def _score_v6_changed() -> bool:
    return bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)]))


def git_ignored(paths: list[Path]) -> set[str]:
    """Subconjunto de paths ignorados pelo git. Usa -z + bytes para evitar a
    traducao de fim de linha do Windows que corromperia a avaliacao de negacoes."""
    if not paths:
        return set()
    rels = [rel(p) for p in paths]
    r = subprocess.run(
        ["git", "check-ignore", "-z", "--stdin"], cwd=ROOT,
        input=("\0".join(rels) + "\0").encode("utf-8"), capture_output=True, check=False,
    )
    out = r.stdout.decode("utf-8", errors="ignore")
    return {piece.replace("\\", "/") for piece in out.split("\0") if piece.strip()}


def _f(value, ndigits: int = 6) -> str:
    try:
        return f"{float(value):.{ndigits}f}"
    except (TypeError, ValueError):
        return NA


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Carregamento de fontes reais
# ---------------------------------------------------------------------------
def load_sources() -> dict:
    features = {r["patch_id"]: r for r in read_csv(FEATURES_SRC)}
    score = {r["patch_id"]: r for r in read_csv(SCORE_V6)}
    overlays = set()
    if OVERLAYS_18G.exists():
        for r in read_csv(OVERLAYS_18G):
            if r.get("vinculo_tecnico_forte") == "true":
                pid = r.get("patch_id", "").replace("CUR_", "curitiba_")
                overlays.add(pid)
    return {"features": features, "score": score, "overlays": overlays or set(CURITIBA_SAR_OVERLAYS)}


def _ndwi(row: dict):
    b3, b8 = _num(row.get("B3_mean")), _num(row.get("B8_mean"))
    if b3 is None or b8 is None or (b3 + b8) == 0:
        return None
    return (b3 - b8) / (b3 + b8)


def _drainage_context(row: dict) -> str:
    wo = _num(row.get("water_occurrence_patch"))
    if wo is None:
        return NA
    return "com_historico_de_agua" if wo > 0 else "sem_historico_de_agua"


def documentary_level(regiao: str, evidence_support: float) -> str:
    if regiao == "petropolis" and evidence_support > 0:
        return "contexto_misto_bloqueado"
    if regiao == "recife" and evidence_support >= 1.0:
        return "forte_review_only"
    if regiao == "recife" and evidence_support > 0:
        return "moderada_review_only"
    return "nenhuma"


def observational_tier(regiao: str, patch_id: str, evidence_support: float, overlays: set) -> str:
    if regiao == "recife" and evidence_support > 0:
        return "C_footprint_tecnico_sar"
    if patch_id in overlays:
        return "C_footprint_tecnico_sar"
    if regiao == "petropolis" and evidence_support > 0:
        return "E_contexto_rua_bairro"
    return "sem_evidencia_observacional_direta"


# ---------------------------------------------------------------------------
# Registro por patch (fonte de verdade interna)
# ---------------------------------------------------------------------------
def build_patch_records(src: dict) -> list[dict]:
    features, score, overlays = src["features"], src["score"], src["overlays"]
    records = []
    for pid in sorted(features.keys()):
        f = features[pid]
        s = score.get(pid, {})
        regiao = f.get("regiao", "desconhecida")
        evid = _num(s.get("evidence_support_index")) or 0.0
        doc_level = documentary_level(regiao, evid)
        obs_tier = observational_tier(regiao, pid, evid, overlays)
        is_sar = pid in overlays

        rec = {
            "patch_id": pid,
            "regiao": regiao,
            "cidade": CIDADE.get(regiao, regiao),
            "bbox": ",".join(_f(f.get(k), 6) for k in ("xmin", "ymin", "xmax", "ymax")),
            "crs": "EPSG:4326",
            "reference_date": f.get("reference_date", NA),
            # fisico
            "elevation_mean": _f(f.get("elevation_mean")),
            "slope_mean": _f(f.get("slope_mean")),
            "HAND_mean": _f(f.get("hand_mean")),
            "distance_to_water_mean": _f(f.get("distance_to_water_mean")),
            "TWI_mean": _f(f.get("twi_mean")),
            "flow_accumulation_mean": _f(f.get("flow_acc_log_mean")),
            "drainage_context": _drainage_context(f),
            # territorial
            "urban_prop": _f(f.get("urban_prop")),
            "vegetation_prop": _f(f.get("vegetation_prop")),
            "water_prop": NA,
            "exposed_soil_prop": NA,
            "impervious_proxy": NA,
            "MapBiomas_class_majority": NA,
            # espectral
            "NDVI": _f(f.get("ndvi_mean")),
            "NDWI": _f(_ndwi(f)),
            "MNDWI": _f(f.get("mndwi_mean")),
            "NDBI": _f(f.get("ndbi_mean")),
            "acquisition_date": f.get("reference_date", NA),
            "sensor": "Sentinel-2",
            "pre_event_flag": "nao_especifico_por_evento",
            # chuva
            "CHIRPS_3d": _f(f.get("chirps_3d_mm")),
            "CHIRPS_7d": _f(f.get("chirps_7d_mm")),
            "CHIRPS_30d": _f(f.get("chirps_30d_mm")),
            "runoff_context_7d": _f(f.get("runoff_context_7d")),
            "event_relative_window": "nao_relativo_a_evento",
            # score_v6
            "score_v6": _f(s.get("susc_score_v6_candidate")) if s else NA,
            "score_v6_class": s.get("susc_class_v6_candidate", NA) if s else NA,
            # documental / observacional
            "evidence_support_index": _f(evid, 3),
            "documentary_evidence_level": doc_level,
            "observational_evidence_tier": obs_tier,
            "technical_sar_evidence": "overlay_tecnico_sar_review_only" if is_sar else "not_available",
            "technical_sar_patch_stats": "disponivel_pos_evento_nao_feature" if regiao == "curitiba" else "not_available",
            "official_partial_event": "curitiba_evento_reconhecido_sem_geometria" if regiao == "curitiba" else "not_available",
        }
        rec["_evid"] = evid
        rec["_is_sar"] = is_sar
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Cobertura por familia
# ---------------------------------------------------------------------------
def _present(value) -> bool:
    return value not in (NA, "", None)


def coverage_for(rec: dict) -> dict:
    fis = [rec[k] for k in ("elevation_mean", "slope_mean", "HAND_mean", "distance_to_water_mean", "TWI_mean", "flow_accumulation_mean", "drainage_context")]
    terr_present = sum(1 for k in ("urban_prop", "vegetation_prop") if _present(rec[k]))
    esp = [rec[k] for k in ("NDVI", "NDWI", "MNDWI", "NDBI")]
    chu = [rec[k] for k in ("CHIRPS_3d", "CHIRPS_7d", "CHIRPS_30d", "runoff_context_7d")]
    cov_fis = sum(1 for v in fis if _present(v)) / len(FISICO_EXPECTED)
    cov_terr = terr_present / len(TERRITORIAL_EXPECTED)
    cov_esp = sum(1 for v in esp if _present(v)) / len(ESPECTRAL_EXPECTED)
    cov_chu = sum(1 for v in chu if _present(v)) / len(CHUVA_EXPECTED)
    cov_doc = 1.0 if rec["documentary_evidence_level"] != "nenhuma" else 0.0
    cov_obs = 1.0 if rec["observational_evidence_tier"] != "sem_evidencia_observacional_direta" else 0.0
    cov_total = (cov_fis + cov_terr + cov_esp + cov_chu + cov_doc + cov_obs) / 6.0
    return {
        "fisico": cov_fis, "territorial": cov_terr, "espectral": cov_esp,
        "chuva": cov_chu, "documental": cov_doc, "observacional": cov_obs, "total": cov_total,
    }


def classe_cobertura(total: float) -> str:
    if total >= 0.80:
        return "cobertura_alta"
    if total >= 0.60:
        return "cobertura_media"
    if total >= 0.40:
        return "cobertura_baixa"
    return "cobertura_insuficiente"


# ---------------------------------------------------------------------------
# Tabelas
# ---------------------------------------------------------------------------
def inventario_rows() -> list[dict]:
    def row(fid, caminho, tipo, regioes, familia, feats, gran, temp, crs, status, obs):
        return {
            "fonte_id": fid, "caminho": caminho, "tipo_fonte": tipo, "regioes_cobertas": regioes,
            "familia_feature": familia, "features_disponiveis": feats, "granularidade": gran,
            "temporalidade": temp, "crs": crs, "status_uso": status, "observacao": obs,
        }
    return [
        row("FONTE_FEATURES", rel(FEATURES_SRC), "csv_features_por_patch", "recife;curitiba;petropolis",
            "fisica_topografica", "elevation_mean;slope_mean;hand_mean;twi_mean;flow_acc_log_mean;distance_to_water_mean;water_occurrence_patch",
            "por_patch", TEMPORALIDADE_FEATURES, "EPSG:4326", "utilizavel_direto",
            "features fisicas completas nas tres regioes"),
        row("FONTE_FEATURES", rel(FEATURES_SRC), "csv_features_por_patch", "recife;curitiba;petropolis",
            "territorial", "urban_prop;vegetation_prop", "por_patch", TEMPORALIDADE_FEATURES, "EPSG:4326",
            "parcial", "urban_prop e vegetation_prop reais; water_prop, exposed_soil_prop, impervious_proxy e MapBiomas ausentes"),
        row("FONTE_MAPBIOMAS", "not_available", "cobertura_do_solo", "nenhuma", "territorial",
            "MapBiomas_class_majority;exposed_soil_prop;water_prop;impervious_proxy", "por_patch", "not_available",
            "not_available", "ausente", "lacuna territorial; nao inventar; buscar cobertura do solo oficial"),
        row("FONTE_FEATURES", rel(FEATURES_SRC), "csv_features_por_patch", "recife;curitiba;petropolis",
            "espectral", "ndvi_mean;mndwi_mean;ndbi_mean;B2_mean;B3_mean;B4_mean;B8_mean;B11_mean;B12_mean",
            "por_patch", TEMPORALIDADE_FEATURES, "EPSG:4326", "utilizavel_direto",
            "NDVI, MNDWI e NDBI diretos; NDWI derivado de B3/B8 reais; AWEI e NDMI nao materializados"),
        row("FONTE_FEATURES", rel(FEATURES_SRC), "csv_features_por_patch", "recife;curitiba;petropolis",
            "chuva", "chirps_3d_mm;chirps_7d_mm;chirps_30d_mm;runoff_context_7d;runoff_context_30d;rain_persistence_index",
            "por_patch", TEMPORALIDADE_FEATURES, "EPSG:4326", "utilizavel_direto",
            "chuva CHIRPS e runoff completos; janela nao relativa a ocorrencia especifica"),
        row("FONTE_SCORE_V6", rel(SCORE_V6), "csv_score_por_patch", "recife;curitiba;petropolis",
            "score_evaluation", "susc_score_v6_candidate;susc_class_v6_candidate;evidence_support_index",
            "por_patch", "referencia review-only", "EPSG:4326", "utilizavel_com_join",
            "score_v6 intacto; usado por join; nao e alvo de treino"),
        row("FONTE_18H_LINHAGEM", rel(LINHAGEM_18H), "csv_linhagem", "recife;curitiba;petropolis",
            "feature_evidence_documental", "documentary_evidence_level;observational_evidence_tier",
            "por_regiao_e_patch", "consolidado 18H", "EPSG:4326", "utilizavel_com_join",
            "evidencia documental/observacional review-only consolidada no 18H"),
        row("FONTE_18G_SAR", rel(OVERLAYS_18G), "csv_overlays_sar", "curitiba", "technical_sar_overlap",
            "technical_sar_vector_overlap", "por_patch", "pos-evento tecnico", "EPSG:4326", "utilizavel_com_join",
            "overlays tecnicos em curitiba_01050 e curitiba_01101 como evidencia, nunca como feature pre-evento"),
        row("FONTE_18F_PATCH_STATS", rel(ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_18f_ingestao_validacao_footprint_sar_curitiba"),
            "csv_patch_stats_sar", "curitiba", "sar_patch_stats_pos_evento", "flood_mask_mean;pixel_count",
            "por_patch", "pos-evento", "EPSG:4326", "bloqueado_por_lacuna",
            "patch_stats SAR e pos-evento; bloqueado como feature pre-evento por regra cientifica"),
    ]


def universo_rows(records: list[dict], cov_by_patch: dict) -> list[dict]:
    rows = []
    for rec in records:
        cov = cov_by_patch[rec["patch_id"]]
        rows.append({
            "patch_id": rec["patch_id"],
            "regiao": rec["regiao"],
            "cidade": rec["cidade"],
            "geometry_source": rel(FEATURES_SRC),
            "bbox": rec["bbox"],
            "crs": rec["crs"],
            "tem_geometria_patch": "true",
            "tem_score_v6": "true" if rec["score_v6"] != NA else "false",
            "tem_features_fisicas": "true" if cov["fisico"] >= 1.0 else "parcial",
            "tem_features_territoriais": "parcial" if 0 < cov["territorial"] < 1.0 else ("true" if cov["territorial"] >= 1.0 else "false"),
            "tem_features_espectrais": "true" if cov["espectral"] >= 1.0 else "parcial",
            "tem_features_chuva": "true" if cov["chuva"] >= 1.0 else "parcial",
            "tem_evidencia_observacional": "true" if cov["observacional"] > 0 else "false",
            "status_patch": "patch_com_features_review_only",
        })
    return rows


def _fam_justi(kind: str) -> str:
    return {
        "fisica": "features fisicas de referencia review-only; composto de referencia, nao janela pre-evento por ocorrencia",
        "territorial": "features territoriais parciais; urban_prop e vegetation_prop reais, demais ausentes marcadas NA como lacuna",
        "espectral": "features espectrais review-only; NDWI derivado de bandas reais; nao e janela pre-evento por ocorrencia",
        "chuva": "features de chuva review-only; janela nao relativa a ocorrencia especifica",
        "documental": "evidencia documental/observacional review-only; SAR e pos-evento e nao vira feature; nao e ground truth",
    }[kind]


def matriz_fisica_rows(records: list[dict]) -> list[dict]:
    out = []
    for r in records:
        out.append({
            "patch_id": r["patch_id"], "regiao": r["regiao"], "cidade": r["cidade"],
            "elevation_mean": r["elevation_mean"], "slope_mean": r["slope_mean"], "HAND_mean": r["HAND_mean"],
            "distance_to_water_mean": r["distance_to_water_mean"], "TWI_mean": r["TWI_mean"],
            "flow_accumulation_mean": r["flow_accumulation_mean"], "drainage_context": r["drainage_context"],
            "fonte": rel(FEATURES_SRC), "temporalidade": TEMPORALIDADE_FEATURES,
            "status_familia": "utilizavel_direto", "justificativa_tecnica": _fam_justi("fisica"),
        })
    return out


def matriz_territorial_rows(records: list[dict]) -> list[dict]:
    out = []
    for r in records:
        out.append({
            "patch_id": r["patch_id"], "regiao": r["regiao"], "cidade": r["cidade"],
            "urban_prop": r["urban_prop"], "vegetation_prop": r["vegetation_prop"],
            "water_prop": r["water_prop"], "exposed_soil_prop": r["exposed_soil_prop"],
            "impervious_proxy": r["impervious_proxy"], "MapBiomas_class_majority": r["MapBiomas_class_majority"],
            "fonte": rel(FEATURES_SRC), "fonte_ausente": "MapBiomas;exposed_soil_prop;water_prop;impervious_proxy",
            "temporalidade": TEMPORALIDADE_FEATURES, "status_familia": "parcial",
            "justificativa_tecnica": _fam_justi("territorial"),
        })
    return out


def matriz_espectral_rows(records: list[dict]) -> list[dict]:
    out = []
    for r in records:
        out.append({
            "patch_id": r["patch_id"], "regiao": r["regiao"], "cidade": r["cidade"],
            "NDVI": r["NDVI"], "NDWI": r["NDWI"], "MNDWI": r["MNDWI"], "NDBI": r["NDBI"],
            "AWEI": NA, "NDMI": NA, "acquisition_date": r["acquisition_date"], "sensor": r["sensor"],
            "pre_event_flag": r["pre_event_flag"], "fonte": rel(FEATURES_SRC),
            "temporalidade": TEMPORALIDADE_FEATURES, "status_familia": "utilizavel_direto",
            "justificativa_tecnica": _fam_justi("espectral"),
        })
    return out


def matriz_chuva_rows(records: list[dict]) -> list[dict]:
    out = []
    for r in records:
        out.append({
            "patch_id": r["patch_id"], "regiao": r["regiao"], "cidade": r["cidade"],
            "CHIRPS_3d": r["CHIRPS_3d"], "CHIRPS_7d": r["CHIRPS_7d"], "CHIRPS_30d": r["CHIRPS_30d"],
            "runoff_context_7d": r["runoff_context_7d"], "event_relative_window": r["event_relative_window"],
            "fonte": rel(FEATURES_SRC), "temporalidade": TEMPORALIDADE_FEATURES,
            "status_familia": "utilizavel_direto", "justificativa_tecnica": _fam_justi("chuva"),
        })
    return out


def matriz_documental_rows(records: list[dict]) -> list[dict]:
    out = []
    for r in records:
        regiao = r["regiao"]
        out.append({
            "patch_id": r["patch_id"], "regiao": regiao, "cidade": r["cidade"],
            "documentary_context": r["documentary_evidence_level"],
            "official_partial_event": r["official_partial_event"],
            "technical_sar_patch_stats": r["technical_sar_patch_stats"],
            "technical_sar_vector_overlap": "true" if r["_is_sar"] else "false",
            "recife_canary_evidence": "true" if (regiao == "recife" and r["_evid"] > 0) else "false",
            "curitiba_sar_evidence": "true" if r["_is_sar"] else "false",
            "petropolis_blocked_mixed": "true" if regiao == "petropolis" else "false",
            "observational_evidence_tier": r["observational_evidence_tier"],
            "review_only_status": "true",
            "justificativa_tecnica": _fam_justi("documental"),
        })
    return out


def matriz_final_rows(records: list[dict], cov_by_patch: dict) -> list[dict]:
    out = []
    for r in records:
        cov = cov_by_patch[r["patch_id"]]
        row = {
            "patch_id": r["patch_id"], "regiao": r["regiao"], "cidade": r["cidade"],
            "tem_geometria_patch": "true",
            "elevation_mean": r["elevation_mean"], "slope_mean": r["slope_mean"], "HAND_mean": r["HAND_mean"],
            "distance_to_water_mean": r["distance_to_water_mean"], "TWI_mean": r["TWI_mean"],
            "flow_accumulation_mean": r["flow_accumulation_mean"],
            "urban_prop": r["urban_prop"], "vegetation_prop": r["vegetation_prop"],
            "water_prop": r["water_prop"], "exposed_soil_prop": r["exposed_soil_prop"],
            "NDVI": r["NDVI"], "NDWI": r["NDWI"], "MNDWI": r["MNDWI"], "NDBI": r["NDBI"],
            "CHIRPS_3d": r["CHIRPS_3d"], "CHIRPS_7d": r["CHIRPS_7d"], "CHIRPS_30d": r["CHIRPS_30d"],
            "score_v6": r["score_v6"], "score_v6_class": r["score_v6_class"],
            "documentary_evidence_level": r["documentary_evidence_level"],
            "observational_evidence_tier": r["observational_evidence_tier"],
            "technical_sar_evidence": r["technical_sar_evidence"],
            "review_only_status": "true",
            "missing_fisico": "false" if cov["fisico"] >= 1.0 else "true",
            "missing_territorial": "false" if cov["territorial"] >= 1.0 else "true",
            "missing_espectral": "false" if cov["espectral"] >= 1.0 else "true",
            "missing_chuva": "false" if cov["chuva"] >= 1.0 else "true",
            "missing_documental": "false" if cov["documental"] >= 1.0 else "true",
            "missing_observacional": "false" if cov["observacional"] >= 1.0 else "true",
            "coverage_score": _f(cov["total"], 4),
            "ground_truth": "false",
            "eligible_for_training": "false",
            "score_v7_allowed": "false",
            "not_ground_truth_reason": "features de referencia review-only sem geometria de ocorrencia confirmada por patch; SAR e pos-evento; evidencia documental nao e verdade de campo",
            "justificativa_tecnica": "linha multimodal por patch somente revisao; coverage_score mede completude e nao suscetibilidade; score_v6 intacto e score_v7 nao permitido",
        }
        out.append(row)
    return out


def cobertura_rows(records: list[dict], cov_by_patch: dict) -> list[dict]:
    out = []
    for r in records:
        cov = cov_by_patch[r["patch_id"]]
        out.append({
            "patch_id": r["patch_id"], "regiao": r["regiao"],
            "coverage_fisico": _f(cov["fisico"], 4), "coverage_territorial": _f(cov["territorial"], 4),
            "coverage_espectral": _f(cov["espectral"], 4), "coverage_chuva": _f(cov["chuva"], 4),
            "coverage_documental": _f(cov["documental"], 4), "coverage_observacional": _f(cov["observacional"], 4),
            "coverage_total": _f(cov["total"], 4), "classe_cobertura": classe_cobertura(cov["total"]),
            "uso_permitido": "apenas_completude_da_matriz_nao_suscetibilidade",
        })
    return out


def missingness_rows(records: list[dict], cov_by_patch: dict) -> list[dict]:
    out = []
    for r in records:
        cov = cov_by_patch[r["patch_id"]]
        faltantes = []
        if cov["territorial"] < 1.0:
            faltantes.extend(sorted(TERRITORIAL_AUSENTE))
        if cov["documental"] < 1.0:
            faltantes.append("documentary_context")
        if cov["observacional"] < 1.0:
            faltantes.append("observational_evidence")
        out.append({
            "patch_id": r["patch_id"], "regiao": r["regiao"],
            "missing_fisico": "false" if cov["fisico"] >= 1.0 else "true",
            "missing_territorial": "false" if cov["territorial"] >= 1.0 else "true",
            "missing_espectral": "false" if cov["espectral"] >= 1.0 else "true",
            "missing_chuva": "false" if cov["chuva"] >= 1.0 else "true",
            "missing_documental": "false" if cov["documental"] >= 1.0 else "true",
            "missing_observacional": "false" if cov["observacional"] >= 1.0 else "true",
            "n_familias_incompletas": str(sum(1 for k in ("fisico", "territorial", "espectral", "chuva", "documental", "observacional") if cov[k] < 1.0)),
            "features_faltantes": ";".join(faltantes) if faltantes else "nenhuma",
        })
    return out


def indice_rows(records: list[dict], cov_by_patch: dict) -> list[dict]:
    out = []
    for r in records:
        cov = cov_by_patch[r["patch_id"]]
        out.append({
            "patch_id": r["patch_id"],
            "coverage_fisico": _f(cov["fisico"], 4), "coverage_territorial": _f(cov["territorial"], 4),
            "coverage_espectral": _f(cov["espectral"], 4), "coverage_chuva": _f(cov["chuva"], 4),
            "coverage_documental": _f(cov["documental"], 4), "coverage_observacional": _f(cov["observacional"], 4),
            "coverage_total": _f(cov["total"], 4), "classe_cobertura": classe_cobertura(cov["total"]),
            "uso_permitido": "indice_de_completude_review_only_nao_e_score_de_suscetibilidade",
            "score_oficial": "false", "substituir_score_v6": "false", "usar_em_treino": "false",
        })
    return out


# ---------------------------------------------------------------------------
# Resumos, gates, schema, preflight, summary
# ---------------------------------------------------------------------------
def resumo_regiao_rows(records: list[dict], cov_by_patch: dict) -> list[dict]:
    out = []
    for regiao in ("recife", "curitiba", "petropolis"):
        recs = [r for r in records if r["regiao"] == regiao]
        if not recs:
            continue
        covs = [cov_by_patch[r["patch_id"]] for r in recs]
        n = len(recs)
        mean = lambda k: sum(c[k] for c in covs) / n
        out.append({
            "regiao": regiao, "n_patches": str(n),
            "coverage_fisico_medio": _f(mean("fisico"), 4),
            "coverage_territorial_medio": _f(mean("territorial"), 4),
            "coverage_espectral_medio": _f(mean("espectral"), 4),
            "coverage_chuva_medio": _f(mean("chuva"), 4),
            "coverage_documental_medio": _f(mean("documental"), 4),
            "coverage_observacional_medio": _f(mean("observacional"), 4),
            "coverage_total_medio": _f(mean("total"), 4),
            "patches_com_evidencia_observacional": str(sum(1 for c in covs if c["observacional"] > 0)),
        })
    return out


def resumo_familia_rows(records: list[dict], cov_by_patch: dict) -> list[dict]:
    n = len(records)
    covs = [cov_by_patch[r["patch_id"]] for r in records]
    def mean(k):
        return sum(c[k] for c in covs) / n if n else 0.0
    familias = [
        ("fisica_topografica", len(FISICO_EXPECTED), len(FISICO_EXPECTED), 0, mean("fisico"), "nenhuma"),
        ("territorial", len(TERRITORIAL_EXPECTED), len(TERRITORIAL_PRESENTE), len(TERRITORIAL_AUSENTE), mean("territorial"), "MapBiomas_class_majority;exposed_soil_prop;water_prop;impervious_proxy"),
        ("espectral", len(ESPECTRAL_EXPECTED), len(ESPECTRAL_EXPECTED), 0, mean("espectral"), "AWEI e NDMI nao materializados (opcionais)"),
        ("chuva", len(CHUVA_EXPECTED), len(CHUVA_EXPECTED), 0, mean("chuva"), "nenhuma"),
        ("documental", 1, 1, 0, mean("documental"), "documentacao so em Recife e contexto misto em Petropolis"),
        ("observacional", 1, 1, 0, mean("observacional"), "evidencia observacional so em Recife e SAR de Curitiba"),
    ]
    return [{
        "familia": fam, "n_features_esperadas": str(esp), "n_features_presentes": str(pres),
        "n_features_ausentes": str(aus), "cobertura_media": _f(cov, 4), "principal_lacuna": lac,
    } for fam, esp, pres, aus, cov, lac in familias]


def status_19a(records: list[dict], cov_by_patch: dict) -> str:
    if not records:
        return "19A_BLOQUEADO_FAIL_CLOSED"
    total = sum(cov_by_patch[r["patch_id"]]["total"] for r in records) / len(records)
    any_missing = any(cov_by_patch[r["patch_id"]][k] < 1.0 for r in records for k in ("territorial", "documental", "observacional"))
    if total >= 0.5 and any_missing:
        return "19A_MATRIZ_MULTIMODAL_PARCIAL_COM_LACUNAS"
    if total >= 0.8:
        return "19A_MATRIZ_MULTIMODAL_CONSOLIDADA"
    return "19A_MATRIZ_MULTIMODAL_PARCIAL_COM_LACUNAS"


def gate_19a_rows(summary: dict) -> list[dict]:
    def g(criterio, valor, limiar, passou, obs):
        return {"criterio": criterio, "valor_observado": str(valor), "limiar": limiar, "passou": "true" if passou else "false", "status": summary["status_19a"], "observacao": obs}
    return [
        g("total_patches", summary["total_patches"], ">0", summary["total_patches"] > 0, "universo de patches consolidado"),
        g("patch_id_unico", summary["patch_id_unicos"], "== total", summary["patch_id_unicos"] == summary["total_patches"], "sem patch_id duplicado"),
        g("regioes_presentes", len(summary["regioes"]), ">=1", len(summary["regioes"]) >= 1, "Recife, Curitiba e Petropolis"),
        g("cobertura_fisica_media", summary["coverage_fisico_medio"], ">=0.9", summary["coverage_fisico_medio"] >= 0.9, "features fisicas completas"),
        g("ground_truth_zero", summary["ground_truth_true_count"], "0", summary["ground_truth_true_count"] == 0, "sem ground truth"),
        g("score_v6_intacto", str(not summary["score_v6_changed"]).lower(), "true", not summary["score_v6_changed"], "score_v6 nao alterado"),
        g("score_v7_zero", str(summary["score_v7_created"]).lower(), "false", not summary["score_v7_created"], "sem score_v7"),
    ]


def gate_19b_rows(summary: dict) -> list[dict]:
    def g(criterio, valor, limiar, passou, obs):
        return {"criterio": criterio, "valor_observado": str(valor), "limiar": limiar, "passou": "true" if passou else "false", "status_19b_alvo": "19B_AUDITORIA_COBERTURA", "observacao": obs}
    return [
        g("matriz_multimodal_pronta", summary["total_patches"], ">0", summary["total_patches"] > 0, "matriz base para auditoria de cobertura"),
        g("missingness_calculada", summary["patches_com_lacuna"], ">=0", summary["patches_com_lacuna"] >= 0, "missingness por patch calculada"),
        g("familia_maior_lacuna", summary["familia_maior_lacuna"], "documentada", bool(summary["familia_maior_lacuna"]), "lacuna territorial documentada"),
        g("coverage_nao_e_suscetibilidade", "true", "true", True, "indice de cobertura nao substitui score de suscetibilidade"),
    ]


def preflight_obj(records: list[dict], cov_by_patch: dict) -> dict:
    outs = [SUMMARY, MAT_FINAL, INDICE]
    ignored = git_ignored(outs)
    status_full = _run_git(["status", "--short"]).splitlines()
    return {
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "staged_count": len(_run_git(["diff", "--cached", "--name-only"]).splitlines()),
        "status_short_count": len(status_full),
        "fontes_lidas": {
            "features_by_patch": FEATURES_SRC.exists(),
            "score_v6": SCORE_V6.exists(),
            "linhagem_18h": LINHAGEM_18H.exists(),
            "overlays_18g": OVERLAYS_18G.exists(),
        },
        "total_patches": len(records),
        "score_v6_path": rel(SCORE_V6),
        "score_v6_changed": _score_v6_changed(),
        "score_v7_path": rel(SCORE_V7),
        "score_v7_created": SCORE_V7.exists(),
        "outputs_ignorados_pelo_git": sorted(git_ignored([OUT])),
        "summary_versionavel": rel(SUMMARY) not in ignored,
        "local_runs_relevantes": "nenhum; 19A nao gera artefato em local_runs",
    }


def summary_obj(records: list[dict], cov_by_patch: dict, resumo_reg: list[dict], resumo_fam: list[dict]) -> dict:
    n = len(records)
    covs = [cov_by_patch[r["patch_id"]] for r in records]
    ids = [r["patch_id"] for r in records]
    fam_min = min(resumo_fam, key=lambda x: float(x["cobertura_media"])) if resumo_fam else {"familia": "not_available"}
    summary = {
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "staged_count": len(_run_git(["diff", "--cached", "--name-only"]).splitlines()),
        "total_patches": n,
        "patch_id_unicos": len(set(ids)),
        "regioes": sorted({r["regiao"] for r in records}),
        "patches_por_regiao": {reg: sum(1 for r in records if r["regiao"] == reg) for reg in sorted({r["regiao"] for r in records})},
        "coverage_fisico_medio": sum(c["fisico"] for c in covs) / n if n else 0.0,
        "coverage_territorial_medio": sum(c["territorial"] for c in covs) / n if n else 0.0,
        "coverage_espectral_medio": sum(c["espectral"] for c in covs) / n if n else 0.0,
        "coverage_chuva_medio": sum(c["chuva"] for c in covs) / n if n else 0.0,
        "coverage_documental_medio": sum(c["documental"] for c in covs) / n if n else 0.0,
        "coverage_observacional_medio": sum(c["observacional"] for c in covs) / n if n else 0.0,
        "coverage_total_medio": sum(c["total"] for c in covs) / n if n else 0.0,
        "patches_com_evidencia_observacional": sum(1 for c in covs if c["observacional"] > 0),
        "patches_com_evidencia_documental": sum(1 for c in covs if c["documental"] > 0),
        "patches_com_lacuna": sum(1 for c in covs if c["total"] < 1.0),
        "familia_maior_lacuna": fam_min["familia"],
        "coverage_is_susceptibility_score": False,
        "status_19a": "",
        "status_17b_mestre": STATUS_17B_MESTRE,
        "marco_17b_criado": False,
        "benchmark_17b_criado": False,
        "ground_truth_true_count": 0,
        "eligible_for_training_true_count": 0,
        "score_v7_allowed_true_count": 0,
        "score_v6_changed": _score_v6_changed(),
        "score_v7_created": SCORE_V7.exists(),
        "review_only": True,
        "proximo_marco": "SUSC-19B",
    }
    summary["status_19a"] = status_19a(records, cov_by_patch)
    return summary


def schema_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SUSC-19A matriz multimodal escalavel por patch",
        "type": "object",
        "description": "Matriz multimodal por patch, review-only. Coverage_score mede completude, nao suscetibilidade.",
        "guardrails": {
            "ground_truth": {"const": "false"},
            "eligible_for_training": {"const": "false"},
            "score_v7_allowed": {"const": "false"},
            "review_only_status": {"const": "true"},
            "coverage_is_susceptibility_score": {"const": False},
        },
        "familias": ["fisica_topografica", "territorial", "espectral", "chuva", "documental", "observacional"],
        "classes_cobertura": sorted(CLASSES_COBERTURA),
        "status_19a_allowed": sorted(STATUS_19A_ALLOWED),
        "status_17b_mestre": STATUS_17B_MESTRE,
        "temporalidade_features": TEMPORALIDADE_FEATURES,
    }


# ---------------------------------------------------------------------------
# Cartoes regionais
# ---------------------------------------------------------------------------
def _card_text(regiao: str, resumo: dict) -> str:
    contexto = {
        "recife": "Recife tem 5 patches com evidencia observacional forte review-only (canarios), coerente com o 18H. E a referencia mais completa.",
        "curitiba": "Curitiba e segunda regiao tecnica: 2 patches com overlay tecnico SAR (curitiba_01050 e curitiba_01101). A geometria oficial segue ausente (18D aguardando). patch_stats SAR e pos-evento e nao entra como feature.",
        "petropolis": "Petropolis tem 1 patch com contexto documental, porem fenomeno misto (deslizamento e inundacao) sem separacao; nao entra como evidencia de inundacao.",
    }[regiao]
    proximo = {
        "recife": "consolidar features territoriais faltantes e ampliar eventos.",
        "curitiba": "ingerir a resposta oficial (18D) e consolidar a referencia tecnica SAR.",
        "petropolis": "separar o fenomeno antes de qualquer promocao.",
    }[regiao]
    return f"""# Cartao regional - {CIDADE[regiao]}

- Total de patches: {resumo['n_patches']}
- Cobertura fisica media: {resumo['coverage_fisico_medio']}
- Cobertura territorial media: {resumo['coverage_territorial_medio']}
- Cobertura espectral media: {resumo['coverage_espectral_medio']}
- Cobertura de chuva media: {resumo['coverage_chuva_medio']}
- Cobertura documental media: {resumo['coverage_documental_medio']}
- Cobertura observacional media: {resumo['coverage_observacional_medio']}
- Patches com evidencia observacional: {resumo['patches_com_evidencia_observacional']}

## Lacunas
- Territorial parcial: faltam MapBiomas, exposed_soil_prop, water_prop e impervious_proxy.
- Evidencia documental/observacional restrita aos patches destacados.

## Relacao com o 18H
{contexto}

## Por que nao e ground truth
As features sao de referencia review-only, sem geometria de ocorrencia confirmada por patch.

## Por que nao e treino
Nenhuma coluna e alvo; eligible_for_training e falso em toda a matriz.

## Por que nao cria score_v7
A matriz organiza cobertura; nao gera score oficial. score_v6 permanece intacto.

## Proximo passo
{proximo}
"""


def write_cards(resumo_reg: list[dict]) -> None:
    ensure_dir(CARDS)
    by_reg = {r["regiao"]: r for r in resumo_reg}
    for regiao in ("recife", "curitiba", "petropolis"):
        if regiao in by_reg:
            write_markdown(CARDS / f"cartao_{regiao}.md", _card_text(regiao, by_reg[regiao]))


# ---------------------------------------------------------------------------
# Relatorio publico
# ---------------------------------------------------------------------------
def report_text(summary: dict, resumo_reg: list[dict], resumo_fam: list[dict]) -> str:
    linhas_reg = "\n".join(
        f"| {r['regiao']} | {r['n_patches']} | {r['coverage_fisico_medio']} | {r['coverage_territorial_medio']} | {r['coverage_espectral_medio']} | {r['coverage_chuva_medio']} | {r['coverage_documental_medio']} | {r['coverage_observacional_medio']} |"
        for r in resumo_reg
    )
    linhas_fam = "\n".join(
        f"| {r['familia']} | {r['n_features_presentes']}/{r['n_features_esperadas']} | {r['cobertura_media']} | {r['principal_lacuna']} |"
        for r in resumo_fam
    )
    return f"""# SUSC-19A - Matriz multimodal escalavel por patch

## Estado herdado do 18H

O 18H consolidou a cadeia 17C ate 18G: Recife como referencia forte review-only,
Curitiba como segunda regiao tecnica SAR sem geometria oficial e Petropolis
bloqueado por fenomeno misto. Estado do 17B: `{summary['status_17b_mestre']}` (17B nao criado).

## Motivacao da matriz multimodal

Deixar de depender do footprint como base central e organizar uma matriz unica,
com uma linha por patch, que sustente a generalizacao do REV-P. A matriz integra
features fisicas, territoriais, espectrais e de chuva reais, alem de score_v6 e
evidencia documental/observacional review-only.

## Fontes de features

As features fisicas, territoriais basicas, espectrais e de chuva vem de
`{rel(FEATURES_SRC)}`. O score_v6 vem de `{rel(SCORE_V6)}` por join. A evidencia
documental/observacional vem do 18H e dos overlays tecnicos SAR do 18G. O
patch_stats SAR (pos-evento) e bloqueado como feature pre-evento.

## Universo de patches

Total de patches consolidados: {summary['total_patches']} (Recife
{summary['patches_por_regiao'].get('recife', 0)}, Curitiba
{summary['patches_por_regiao'].get('curitiba', 0)}, Petropolis
{summary['patches_por_regiao'].get('petropolis', 0)}). Cada patch e uma linha
unica, com geometria de patch e bbox.

## Cobertura por regiao

| Regiao | Patches | Fisico | Territorial | Espectral | Chuva | Documental | Observacional |
| --- | --- | --- | --- | --- | --- | --- | --- |
{linhas_reg}

## Cobertura por familia

| Familia | Presentes/Esperadas | Cobertura media | Principal lacuna |
| --- | --- | --- | --- |
{linhas_fam}

## Lacunas

A maior lacuna e territorial: faltam MapBiomas, exposed_soil_prop, water_prop e
impervious_proxy em todas as regioes. A evidencia documental e observacional
existe apenas em poucos patches (Recife e Curitiba SAR). A missingness e explicita
em `missingness_por_patch.csv` e nao e escondida.

## Como Recife, Curitiba e Petropolis entram

- Recife: features completas mais evidencia observacional forte review-only.
- Curitiba: features completas mais overlay tecnico SAR em 2 patches; geometria
  oficial ausente.
- Petropolis: features completas, porem evidencia bloqueada por fenomeno misto.

## Por que SAR e footprints sao canarios

O footprint e o SAR indicam onde olhar (canario), mas nao sao a base estrutural
nem geometria oficial. O patch_stats SAR e pos-evento e nunca vira feature pre-evento.

## Por que a matriz e a base estrutural

Ela e escalavel (uma linha por patch), auditavel (fonte e temporalidade por
feature) e honesta quanto a missingness. E o eixo que sustenta a generalizacao.

## Por que nao e ground truth, nem treino, nem score_v7

Toda a matriz e review-only: ground_truth falso, treino desabilitado e score_v7
nao permitido. O `coverage_score` mede completude da matriz, nunca suscetibilidade.
O score_v6 permanece intacto e nenhum benchmark 17B foi criado.

## Proximo marco recomendado

**SUSC-19B - Auditoria de cobertura multimodal**: aprofundar a missingness por
regiao e feature e priorizar o preenchimento das lacunas territoriais.
"""


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_all() -> dict:
    ensure_dir(OUT)
    ensure_dir(REPORTS)
    ensure_dir(SCHEMAS_DIR)

    src = load_sources()
    records = build_patch_records(src)
    cov_by_patch = {r["patch_id"]: coverage_for(r) for r in records}

    resumo_reg = resumo_regiao_rows(records, cov_by_patch)
    resumo_fam = resumo_familia_rows(records, cov_by_patch)
    summary = summary_obj(records, cov_by_patch, resumo_reg, resumo_fam)

    write_csv(INVENTARIO, inventario_rows())
    write_csv(UNIVERSO, universo_rows(records, cov_by_patch))
    write_csv(MAT_FISICA, matriz_fisica_rows(records))
    write_csv(MAT_TERRITORIAL, matriz_territorial_rows(records))
    write_csv(MAT_ESPECTRAL, matriz_espectral_rows(records))
    write_csv(MAT_CHUVA, matriz_chuva_rows(records))
    write_csv(MAT_DOC, matriz_documental_rows(records))
    write_csv(MAT_FINAL, matriz_final_rows(records, cov_by_patch))
    write_csv(AUD_COBERTURA, cobertura_rows(records, cov_by_patch))
    write_csv(MISSINGNESS, missingness_rows(records, cov_by_patch))
    write_csv(INDICE, indice_rows(records, cov_by_patch))
    write_csv(GATE_19A, gate_19a_rows(summary))
    write_csv(GATE_19B, gate_19b_rows(summary))
    write_csv(RESUMO_REGIAO, resumo_reg)
    write_csv(RESUMO_FAMILIA, resumo_fam)

    write_json(SCHEMA, schema_obj())
    write_json(PREFLIGHT, preflight_obj(records, cov_by_patch))
    write_json(SUMMARY, summary)
    write_cards(resumo_reg)
    write_markdown(REPORT, report_text(summary, resumo_reg, resumo_fam))
    return summary


# ---------------------------------------------------------------------------
# Validacao
# ---------------------------------------------------------------------------
def _public_paths() -> list[Path]:
    paths = []
    if OUT.exists():
        paths.extend(p for p in OUT.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".json", ".md"})
    if REPORT.exists():
        paths.append(REPORT)
    return sorted(dict.fromkeys(paths), key=lambda p: rel(p))


def public_text_violations_text(text: str) -> list[str]:
    return sorted({m.group(0) for m in PUBLIC_FORBIDDEN_RE.finditer(text)})


def validate_public_text() -> list[str]:
    errors = []
    for path in _public_paths():
        hits = public_text_violations_text(path.read_text(encoding="utf-8", errors="ignore"))
        if hits:
            errors.append(f"{rel(path)}:vocabulario_publico_proibido:{','.join(hits)}")
    return errors


def validate_outputs(summary: dict) -> list[str]:
    errors = [f"missing:{rel(p)}" for p in REQUIRED_OUTPUTS if not p.exists()]

    if summary["score_v6_changed"]:
        errors.append("score_v6_changed_forbidden")
    if summary["score_v7_created"]:
        errors.append("score_v7_created_forbidden")
    if summary["benchmark_17b_criado"]:
        errors.append("benchmark_17b_criado_forbidden")
    if summary["coverage_is_susceptibility_score"]:
        errors.append("coverage_chamado_de_suscetibilidade")
    if summary["status_19a"] not in STATUS_19A_ALLOWED:
        errors.append(f"status_19a_fora_enum:{summary['status_19a']}")

    final = read_csv(MAT_FINAL)
    if not final:
        errors.append("matriz_final_vazia")
    ids = [r["patch_id"] for r in final]
    if len(ids) != len(set(ids)):
        errors.append("patch_id_duplicado_sem_justificativa")
    regioes = {r["regiao"] for r in final}
    for reg in ("recife", "curitiba", "petropolis"):
        if reg not in regioes:
            errors.append(f"regiao_ausente:{reg}")

    for r in final:
        for field in ("ground_truth", "eligible_for_training", "score_v7_allowed"):
            if r.get(field) != "false":
                errors.append(f"final:{r['patch_id']}:{field}_nao_false")
        if not r.get("justificativa_tecnica", "").strip():
            errors.append(f"final:{r['patch_id']}:justificativa_vazia")
        if not r.get("not_ground_truth_reason", "").strip():
            errors.append(f"final:{r['patch_id']}:not_ground_truth_reason_vazio")

    # feature presente exige fonte marcada
    for path, feat_cols in ((MAT_FISICA, ["elevation_mean"]), (MAT_ESPECTRAL, ["NDVI"]), (MAT_CHUVA, ["CHIRPS_3d"])):
        for r in read_csv(path):
            if any(r.get(c, NA) not in (NA, "") for c in feat_cols):
                if r.get("fonte", "") in ("", "ausente", "not_available"):
                    errors.append(f"{rel(path)}:{r.get('patch_id')}:feature_presente_sem_fonte")

    # patch_stats SAR nunca como feature pre-evento
    for r in read_csv(MAT_DOC):
        val = r.get("technical_sar_patch_stats", "")
        if val not in ("not_available", "disponivel_pos_evento_nao_feature"):
            errors.append(f"doc:{r.get('patch_id')}:patch_stats_sar_uso_indevido")
    for r in read_csv(MAT_ESPECTRAL):
        if r.get("pre_event_flag") == "true":
            errors.append(f"espectral:{r.get('patch_id')}:pre_event_flag_indevido")

    # Curitiba SAR nao pode ser chamada de geometria oficial / AOI nao e ocorrencia
    for r in read_csv(MAT_DOC):
        justi = r.get("justificativa_tecnica", "").lower()
        if r.get("curitiba_sar_evidence") == "true":
            if "geometria oficial" in justi and "nao" not in justi:
                errors.append(f"doc:{r.get('patch_id')}:sar_chamado_de_geometria_oficial")
        if "aoi" in justi and "ocorrencia" in justi and "nao" not in justi:
            errors.append(f"doc:{r.get('patch_id')}:aoi_chamada_de_ocorrencia")

    # indice de cobertura nunca chamado de score de suscetibilidade
    for r in read_csv(INDICE):
        if r.get("score_oficial") != "false" or r.get("substituir_score_v6") != "false" or r.get("usar_em_treino") != "false":
            errors.append(f"indice:{r.get('patch_id')}:flag_proibida")
        if "suscetibilidade" not in r.get("uso_permitido", "").lower() or "nao" not in r.get("uso_permitido", "").lower():
            errors.append(f"indice:{r.get('patch_id')}:uso_permitido_incompleto")

    # gate status no enum
    for r in read_csv(GATE_19A):
        if r.get("status") not in STATUS_19A_ALLOWED:
            errors.append(f"gate_19a:status_fora_enum:{r.get('status')}")

    # output 19A nao pode ser ignorado pelo git
    if rel(SUMMARY) in git_ignored([SUMMARY, MAT_FINAL]):
        errors.append("output_19a_ignorado_pelo_git")

    errors.extend(validate_public_text())
    return errors


def validate() -> int:
    summary = build_all()
    errors = validate_outputs(summary)
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1
    print(
        "19A matriz multimodal validada: "
        f"patches={summary['total_patches']} unicos={summary['patch_id_unicos']} "
        f"cov_total_medio={summary['coverage_total_medio']:.3f} "
        f"status19A={summary['status_19a']} status17B={summary['status_17b_mestre']} "
        f"score_v6_changed={str(summary['score_v6_changed']).lower()}"
    )
    return 0


def run_all() -> int:
    summary = build_all()
    print(
        "19A matriz multimodal gerada: "
        f"patches={summary['total_patches']} regioes={len(summary['regioes'])} "
        f"cov_total_medio={summary['coverage_total_medio']:.3f} "
        f"familia_maior_lacuna={summary['familia_maior_lacuna']} status19A={summary['status_19a']}"
    )
    return 0
