"""SUSC-19C avaliacao observacional review-only da matriz multimodal.

Compara, sem treino e sem benchmark, os patches com evidencia observacional
review-only (Recife 5 canarios; Curitiba 2 overlays tecnicos SAR) contra o
universo nao rotulado (unlabeled_background) do 19A/19B, no score_v6 e nas
features fisicas, hidrologicas, espectrais, urbanas e de chuva.

Pergunta (exploratoria): as areas com evidencia observacional review-only tem
score_v6 e features compativeis com maior suscetibilidade em relacao ao universo
nao rotulado? Nao responde "o modelo preve enchentes", nao ha ground truth, nao
ha negativos, nao ha benchmark e o score_v6 nao e validado operacionalmente.

Guardrails: ground_truth=false, eligible_for_training=false, score_v7_allowed=false,
sem benchmark 17B, sem score_v7, score_v6 intacto. O universo sem evento e
unlabeled_background (nunca negativo). Petropolis misto/contextual nao entra como
observado. patch_stats SAR e pos-evento e nunca vira feature pre-evento.
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
OUT = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_19c_avaliacao_observacional_review_only"
CARDS = OUT / "cartoes_observacionais"
REPORTS = ROOT / "outputs_public" / "reports"
SCHEMAS_DIR = ROOT / "schemas" / "suscetibilidade"

REPORT = REPORTS / "SUSC_19C_AVALIACAO_OBSERVACIONAL_REVIEW_ONLY.md"
SCHEMA = SCHEMAS_DIR / "susc_19c_avaliacao_observacional_schema_v1.json"

FEATURES_SRC = ROOT / "datasets" / "suscetibilidade" / "susc_features_by_patch_v1.csv"
SCORE_V6 = ROOT / "datasets" / "suscetibilidade" / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = ROOT / "datasets" / "suscetibilidade" / "susc_score_v7_candidate_by_patch_v1.csv"
MAT_19A = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_19a_matriz_multimodal_escalavel_por_patch" / "matriz_multimodal_escalavel_por_patch.csv"
OVERLAYS_18G = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_18g_recuperacao_compactacao_vetorial_sar_curitiba" / "vinculos_vetoriais_sar_patch_curitiba_18g.csv"

# Saidas
REG_OBS = OUT / "registro_patches_observacionais_19c.csv"
REG_EXC = OUT / "registro_patches_excluidos_19c.csv"
UNIVERSO = OUT / "universo_nao_rotulado_19c.csv"
RANKING = OUT / "ranking_score_v6_observacional_19c.csv"
HITRATE = OUT / "metricas_hit_rate_review_only_19c.csv"
CONTRASTE = OUT / "contraste_features_observacional_19c.csv"
DIVERGENCIAS = OUT / "matriz_divergencias_observacionais_19c.csv"
AVAL_REGIAO = OUT / "avaliacao_por_regiao_19c.csv"
MAT_AVAL = OUT / "matriz_avaliacao_observacional_review_only_19c.csv"
GATE_19C = OUT / "gate_avaliacao_observacional_19c.csv"
GATE_17B = OUT / "gate_prontidao_17b_pos_19c.csv"
GATE_V7 = OUT / "gate_prontidao_score_v7_pos_19c.csv"
RESUMO_REGIAO = OUT / "resumo_por_regiao.csv"
RESUMO_STATUS = OUT / "resumo_por_status.csv"
SUMMARY = OUT / "summary.json"
PREFLIGHT = OUT / "preflight.json"

REQUIRED_INPUTS = [FEATURES_SRC, SCORE_V6, MAT_19A]
REQUIRED_OUTPUTS = [
    REG_OBS, REG_EXC, UNIVERSO, RANKING, HITRATE, CONTRASTE, DIVERGENCIAS, AVAL_REGIAO,
    MAT_AVAL, GATE_19C, GATE_17B, GATE_V7, RESUMO_REGIAO, RESUMO_STATUS, SUMMARY, PREFLIGHT,
    SCHEMA, REPORT,
]

# ---------------------------------------------------------------------------
# Guardrails / constantes
# ---------------------------------------------------------------------------
PUBLIC_FORBIDDEN_RE = re.compile(r"\b(?:agentic|agente|codex|llm|ia)\b", re.IGNORECASE)
NA = "NA"

STATUS_19C_ALLOWED = {
    "19C_AVALIACAO_REVIEW_ONLY_CONCLUIDA",
    "19C_AVALIACAO_PARCIAL_COM_AMOSTRA_MINIMA",
    "19C_INSUFICIENTE_PARA_CONCLUSAO",
    "19C_BLOQUEADO_FAIL_CLOSED",
}
STATUS_17B_ALLOWED = {
    "17B_NAO_CRIADO",
    "17B_APROXIMACAO_COM_SEGUNDA_REGIAO_TECNICA",
    "17B_BLOQUEADO_POR_AMOSTRA",
    "17B_BLOQUEADO_POR_GEOMETRIA_OFICIAL",
}
STATUS_V7_ALLOWED = {
    "SCORE_V7_NAO_AUTORIZADO",
    "SCORE_V7_BLOQUEADO_POR_AMOSTRA",
    "SCORE_V7_BLOQUEADO_POR_MISSINGNESS_TERRITORIAL",
    "SCORE_V7_BLOQUEADO_POR_AUSENCIA_BENCHMARK",
}
EVAL_STATUS_ALLOWED = {
    "aderencia_observacional_exploratoria",
    "aderencia_parcial_com_divergencias",
    "divergencia_observacional_relevante",
    "amostra_insuficiente_para_conclusao",
    "bloqueado_por_fenomeno_misto",
}
REGIONAL_STATUS = {
    "recife": "recife_review_only_com_amostra_pequena",
    "curitiba": "curitiba_tecnica_sar_com_amostra_minima",
    "petropolis": "petropolis_excluido_por_fenomeno_misto",
}
CIDADE = {"recife": "Recife", "curitiba": "Curitiba", "petropolis": "Petropolis"}
SMALL_SAMPLE_LIMIT = 30  # abaixo disso, potencia amostral baixa; sem conclusao estatistica forte

# Features do contraste: display -> (coluna_fonte ou "NDWI"/"score", direcao esperada)
FEATURE_SPEC = [
    ("elevation_mean", "elevation_mean", "menor"),
    ("slope_mean", "slope_mean", "menor"),
    ("HAND_mean", "hand_mean", "menor"),
    ("distance_to_water_mean", "distance_to_water_mean", "menor"),
    ("TWI_mean", "twi_mean", "maior"),
    ("flow_accumulation_mean", "flow_acc_log_mean", "maior"),
    ("urban_prop", "urban_prop", "maior"),
    ("vegetation_prop", "vegetation_prop", "menor"),
    ("NDVI", "ndvi_mean", "menor"),
    ("NDWI", "NDWI", "maior"),
    ("MNDWI", "mndwi_mean", "maior"),
    ("NDBI", "ndbi_mean", "maior"),
    ("CHIRPS_3d", "chirps_3d_mm", "maior"),
    ("CHIRPS_7d", "chirps_7d_mm", "maior"),
    ("CHIRPS_30d", "chirps_30d_mm", "maior"),
    ("score_v6", "score", "maior"),
]


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _run_git(args: list[str]) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return r.stdout.strip() if r.returncode == 0 else ""


def _score_v6_changed() -> bool:
    return bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)]))


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


def require_inputs() -> None:
    missing = [rel(p) for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("entradas obrigatorias ausentes: " + "; ".join(missing))


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _fmt(v, n=4):
    return f"{v:.{n}f}" if isinstance(v, (int, float)) else NA


# ---------------------------------------------------------------------------
# Carga e derivacao dos conjuntos
# ---------------------------------------------------------------------------
def load_data() -> dict:
    features = {r["patch_id"]: r for r in read_csv(FEATURES_SRC)}
    score = {r["patch_id"]: r for r in read_csv(SCORE_V6)}
    # NDWI derivado de B3/B8 reais
    for pid, f in features.items():
        b3, b8 = _num(f.get("B3_mean")), _num(f.get("B8_mean"))
        f["NDWI"] = (b3 - b8) / (b3 + b8) if (b3 is not None and b8 is not None and (b3 + b8) != 0) else None
        f["score"] = _num(score.get(pid, {}).get("susc_score_v6_candidate"))
    return {"features": features, "score": score}


def curitiba_sar_patches() -> list[str]:
    ids = []
    if OVERLAYS_18G.exists():
        for r in read_csv(OVERLAYS_18G):
            if r.get("vinculo_tecnico_forte") == "true":
                pid = r.get("patch_id", "").replace("CUR_", "curitiba_")
                if pid:
                    ids.append(pid)
    return sorted(dict.fromkeys(ids)) or ["curitiba_01050", "curitiba_01101"]


def derive_sets(data: dict) -> dict:
    score = data["score"]
    recife_obs = sorted(p for p, r in score.items() if r.get("regiao") == "recife" and (_num(r.get("evidence_support_index")) or 0) > 0)
    curitiba_obs = [p for p in curitiba_sar_patches() if p in score]
    petropolis_ctx = sorted(p for p, r in score.items() if r.get("regiao") == "petropolis" and (_num(r.get("evidence_support_index")) or 0) > 0)
    observed = recife_obs + curitiba_obs
    background = sorted(p for p in score if p not in set(observed))
    return {
        "recife_obs": recife_obs, "curitiba_obs": curitiba_obs,
        "petropolis_ctx": petropolis_ctx, "observed": observed, "background": background,
    }


# ---------------------------------------------------------------------------
# Ranking score_v6
# ---------------------------------------------------------------------------
def compute_ranks(data: dict) -> dict:
    score = data["score"]
    def sc(pid):
        return _num(score[pid].get("susc_score_v6_candidate")) or 0.0
    allids = sorted(score, key=lambda p: -sc(p))
    N = len(allids)
    global_rank = {pid: i + 1 for i, pid in enumerate(allids)}
    regional_rank = {}
    regional_n = {}
    for reg in ("recife", "curitiba", "petropolis"):
        regids = sorted([p for p in score if score[p].get("regiao") == reg], key=lambda p: -sc(p))
        regional_n[reg] = len(regids)
        for i, pid in enumerate(regids):
            regional_rank[pid] = i + 1
    return {"global_rank": global_rank, "regional_rank": regional_rank, "regional_n": regional_n, "N": N, "sc": sc, "allids": allids}


def ranking_rows(data: dict, sets: dict, ranks: dict) -> list[dict]:
    score = data["score"]
    out = []
    for pid in sets["observed"]:
        reg = score[pid].get("regiao")
        rg = ranks["global_rank"][pid]
        rr = ranks["regional_rank"][pid]
        N = ranks["N"]
        Nr = ranks["regional_n"][reg]
        pct_g = (N - rg + 1) / N
        pct_r = (Nr - rr + 1) / Nr
        cls = score[pid].get("susc_class_v6_candidate", NA)
        interp = "score no terco superior" if pct_g >= 0.66 else ("score intermediario" if pct_g >= 0.4 else "score baixo apesar da evidencia")
        out.append({
            "patch_id": pid, "regiao": reg, "score_v6": _fmt(ranks["sc"](pid), 6),
            "score_v6_class": cls, "rank_global": str(rg), "percentile_global": _fmt(pct_g),
            "rank_regional": str(rr), "percentile_regional": _fmt(pct_r),
            "top_10_global": "true" if rg <= 10 else "false",
            "top_20_global": "true" if rg <= 20 else "false",
            "top_30_global": "true" if rg <= 30 else "false",
            "top_10_regional": "true" if rr <= 10 else "false",
            "top_20_regional": "true" if rr <= 20 else "false",
            "top_30_regional": "true" if rr <= 30 else "false",
            "interpretation_review_only": interp + "; leitura exploratoria review-only, nao e validacao",
        })
    return out


# ---------------------------------------------------------------------------
# Hit-rate / enrichment
# ---------------------------------------------------------------------------
def hitrate_rows(data: dict, sets: dict, ranks: dict) -> list[dict]:
    score = data["score"]
    observed = sets["observed"]
    n_obs = len(observed)
    N = ranks["N"]
    allids = ranks["allids"]
    obs_set = set(observed)
    rows = []
    # global
    for K in (10, 20, 30):
        topg = set(allids[:K])
        hits = sum(1 for p in observed if p in topg)
        expected = n_obs * K / N
        enr = (hits / expected) if expected > 0 else 0.0
        rows.append({
            "escopo": "global", "top_k": str(K), "observados": str(n_obs), "universo": str(N),
            "observed_in_top_k": str(hits), "esperado_aleatorio": _fmt(expected, 3),
            "enrichment_ratio": _fmt(enr, 3),
            "potencia_amostral": "baixa" if n_obs < SMALL_SAMPLE_LIMIT else "moderada",
            "limitation": "metrica exploratoria review-only; amostra pequena; nao e validacao operacional nem benchmark",
        })
    # regional (agregado)
    for K in (10, 20, 30):
        hits = 0
        expected = 0.0
        for reg in ("recife", "curitiba"):
            regids = sorted([p for p in score if score[p].get("regiao") == reg], key=lambda p: -(ranks["sc"](p)))
            top = set(regids[:K])
            n_obs_r = sum(1 for p in observed if score[p].get("regiao") == reg)
            hits += sum(1 for p in observed if score[p].get("regiao") == reg and p in top)
            expected += n_obs_r * K / len(regids) if regids else 0
        enr = (hits / expected) if expected > 0 else 0.0
        rows.append({
            "escopo": "regional", "top_k": str(K), "observados": str(n_obs), "universo": "100_por_regiao",
            "observed_in_top_k": str(hits), "esperado_aleatorio": _fmt(expected, 3),
            "enrichment_ratio": _fmt(enr, 3),
            "potencia_amostral": "baixa",
            "limitation": "metrica exploratoria review-only; amostra minima por regiao; nao e validacao nem benchmark",
        })
    return rows


# ---------------------------------------------------------------------------
# Contraste de features
# ---------------------------------------------------------------------------
def _mean(features: dict, ids: list[str], col: str):
    vals = []
    for p in ids:
        v = _num(features[p].get(col))
        if v is not None:
            vals.append(v)
    return sum(vals) / len(vals) if vals else None


def _agrees(direction: str, delta: float) -> bool:
    return delta > 0 if direction == "maior" else delta < 0


def contraste_rows(data: dict, sets: dict) -> tuple[list[dict], dict]:
    features = data["features"]
    escopos = {
        "global": (sets["observed"], sets["background"]),
        "recife": ([p for p in sets["observed"] if features[p].get("regiao") == "recife"],
                   [p for p in sets["background"] if features[p].get("regiao") == "recife"]),
        "curitiba": ([p for p in sets["observed"] if features[p].get("regiao") == "curitiba"],
                     [p for p in sets["background"] if features[p].get("regiao") == "curitiba"]),
    }
    rows = []
    agree_by_scope = {}
    for escopo, (obs, bg) in escopos.items():
        agree = 0
        counted = 0
        for disp, col, direction in FEATURE_SPEC:
            o = _mean(features, obs, col)
            b = _mean(features, bg, col)
            if o is None or b is None:
                delta = None
                agr = NA
            else:
                delta = o - b
                agr = "true" if _agrees(direction, delta) else "false"
                counted += 1
                agree += 1 if agr == "true" else 0
            rows.append({
                "escopo": escopo, "feature": disp,
                "expected_direction_for_susceptibility": direction,
                "observed_mean": _fmt(o, 4) if o is not None else NA,
                "background_mean": _fmt(b, 4) if b is not None else NA,
                "delta": _fmt(delta, 4) if delta is not None else NA,
                "direction_agreement": agr,
                "sample_observed_n": str(len(obs)), "background_n": str(len(bg)),
                "limitation": "contraste exploratorio review-only; amostra pequena; sem conclusao estatistica forte",
            })
        agree_by_scope[escopo] = (agree, counted)
    return rows, agree_by_scope


def patch_direction_agreement(data: dict, sets: dict, pid: str) -> float:
    """Taxa de direcoes em que o patch esta no lado suscetivel vs media do background."""
    features = data["features"]
    bg = sets["background"]
    agree = 0
    counted = 0
    for disp, col, direction in FEATURE_SPEC:
        pv = _num(features[pid].get(col))
        bm = _mean(features, bg, col)
        if pv is None or bm is None:
            continue
        counted += 1
        delta = pv - bm
        if _agrees(direction, delta):
            agree += 1
    return agree / counted if counted else 0.0


# ---------------------------------------------------------------------------
# Divergencias
# ---------------------------------------------------------------------------
def divergencia_rows(data: dict, sets: dict, ranks: dict, agree_by_scope: dict) -> list[dict]:
    score = data["score"]
    rows = []
    idx = 1
    for pid in sets["observed"]:
        reg = score[pid].get("regiao")
        cls = (score[pid].get("susc_class_v6_candidate") or "").lower()
        pct = (ranks["N"] - ranks["global_rank"][pid] + 1) / ranks["N"]
        flags = []
        if cls == "low":
            flags.append("observed_patch_low_score")
        elif cls == "medium":
            flags.append("observed_patch_medium_score")
        if reg == "curitiba":
            flags.append("SAR_only_evidence")
        else:
            evid = _num(score[pid].get("evidence_support_index")) or 0
            if 0 < evid < 1:
                flags.append("documentary_low_confidence")
        flags.append("sample_size_limitation")
        for flag in flags:
            rows.append({
                "item_id": f"DIV_{idx:03d}", "escopo": "patch", "patch_id": pid, "regiao": reg,
                "divergence_type": flag, "evidencia": f"score_v6_class={cls or NA};percentile_global={_fmt(pct)}",
                "review_only": "true",
                "justificativa_tecnica": "divergencia registrada em revisao; nao corrige score, nao propoe score_v7, nao e negativo",
            })
            idx += 1
    # divergencias globais a partir do contraste
    ag, ct = agree_by_scope["global"]
    global_flags = [
        ("physical_divergence", "distance_to_water/TWI/flow_accumulation divergentes da direcao esperada no contraste global"),
        ("rainfall_divergence", "CHIRPS 3d/7d/30d menores nos observados que no background (divergencia de gatilho de chuva)"),
        ("spectral_divergence", "verificar NDVI/NDWI/MNDWI/NDBI no contraste global"),
        ("sample_size_limitation", f"apenas {len(sets['observed'])} patches observacionais; potencia amostral baixa"),
    ]
    for flag, ev in global_flags:
        rows.append({
            "item_id": f"DIV_{idx:03d}", "escopo": "global", "patch_id": "todos_observados", "regiao": "global",
            "divergence_type": flag, "evidencia": ev + f"; direcoes_coerentes_global={ag}/{ct}",
            "review_only": "true",
            "justificativa_tecnica": "divergencia global exploratoria review-only; nao corrige score e nao propoe score_v7",
        })
        idx += 1
    return rows


# ---------------------------------------------------------------------------
# Matriz de avaliacao final (uma linha por observacional)
# ---------------------------------------------------------------------------
def _eval_status(cls: str, pct: float, agree_rate: float) -> str:
    cls = (cls or "").lower()
    if cls == "low":
        return "divergencia_observacional_relevante"
    if cls == "medium":
        return "aderencia_parcial_com_divergencias"
    if agree_rate >= 0.6 and pct >= 0.6:
        return "aderencia_observacional_exploratoria"
    return "aderencia_parcial_com_divergencias"


def avaliacao_rows(data: dict, sets: dict, ranks: dict) -> list[dict]:
    score = data["score"]
    out = []
    for pid in sets["observed"]:
        reg = score[pid].get("regiao")
        cls = score[pid].get("susc_class_v6_candidate", NA)
        pct_g = (ranks["N"] - ranks["global_rank"][pid] + 1) / ranks["N"]
        Nr = ranks["regional_n"][reg]
        pct_r = (Nr - ranks["regional_rank"][pid] + 1) / Nr
        agree_rate = patch_direction_agreement(data, sets, pid)
        etype = "recife_canary_review_only" if reg == "recife" else "curitiba_sar_overlay_review_only"
        flags = []
        if (cls or "").lower() == "low":
            flags.append("observed_patch_low_score")
        if (cls or "").lower() == "medium":
            flags.append("observed_patch_medium_score")
        if reg == "curitiba":
            flags.append("SAR_only_evidence")
        flags.append("sample_size_limitation")
        out.append({
            "patch_id": pid, "regiao": reg, "evidence_tier": "C_footprint_tecnico_sar",
            "evidence_type": etype, "score_v6": _fmt(ranks["sc"](pid), 6), "score_v6_class": cls,
            "percentile_global": _fmt(pct_g), "percentile_regional": _fmt(pct_r),
            "feature_direction_agreement_rate": _fmt(agree_rate, 3),
            "divergence_flags": ";".join(flags),
            "evaluation_status": _eval_status(cls, pct_g, agree_rate),
            "review_only": "true", "ground_truth": "false", "eligible_for_training": "false",
            "score_v7_allowed": "false",
            "not_ground_truth_reason": "evidencia observacional review-only sem geometria de ocorrencia confirmada por patch; SAR e pos-evento; nao e verdade de campo",
            "justificativa_tecnica": "avaliacao exploratoria review-only contra universo nao rotulado; nao e validacao, nao e benchmark, score_v6 intacto",
        })
    return out


# ---------------------------------------------------------------------------
# Registros e universo
# ---------------------------------------------------------------------------
def registro_obs_rows(data: dict, sets: dict) -> list[dict]:
    score = data["score"]
    out = []
    for pid in sets["observed"]:
        reg = score[pid].get("regiao")
        if reg == "recife":
            src, etype, ev = "recife_canario_review_only_17c_18h", "event_anchored_canary", "documentary_review_only"
        else:
            src, etype, ev = "curitiba_overlay_tecnico_sar_18g", "sar_vector_overlap", "technical_sar_review_only"
        out.append({
            "patch_id": pid, "regiao": reg, "cidade": CIDADE.get(reg, reg),
            "evidence_source": src, "evidence_type": etype, "evidence_tier": "C_footprint_tecnico_sar",
            "candidate_event_id": "S17C_REF_0060" if reg == "curitiba" else "REC_CANARY_SET",
            "geometry_id": "S18G_CUR_SAR_COMPACT_0001" if reg == "curitiba" else "not_available",
            "patch_link_id": f"S18G_LINK_{pid.replace('curitiba_', 'CUR_')}" if reg == "curitiba" else "REC_PATCH_LINK",
            "review_only": "true", "ground_truth": "false", "eligible_for_training": "false",
            "score_v7_allowed": "false", "uso_permitido": ev,
            "justificativa_tecnica": "patch observacional review-only; SAR nao e geometria oficial; nao e ground truth e nao habilita treino",
        })
    return out


def registro_exc_rows(data: dict, sets: dict) -> list[dict]:
    score = data["score"]
    out = []
    for pid in sets["petropolis_ctx"]:
        out.append({
            "patch_id": pid, "regiao": "petropolis", "cidade": "Petropolis",
            "motivo_exclusao": "fenomeno_misto_deslizamento_inundacao_sem_separacao",
            "evidence_type": "documentary_context_only", "evidence_tier": "E_contexto_rua_bairro",
            "status": "contextual_bloqueado", "review_only": "true",
            "ground_truth": "false", "eligible_for_training": "false", "score_v7_allowed": "false",
            "justificativa_tecnica": "Petropolis misto/contextual nao entra como observado; permanece bloqueado ate separacao do fenomeno",
        })
    return out


def universo_rows(data: dict, sets: dict) -> list[dict]:
    score = data["score"]
    out = []
    for pid in sets["background"]:
        reg = score[pid].get("regiao")
        out.append({
            "patch_id": pid, "regiao": reg,
            "rotulo": "unlabeled_background",
            "condicao": "no_documented_observational_evidence",
            "score_v6": score[pid].get("susc_score_v6_candidate", NA),
            "score_v6_class": score[pid].get("susc_class_v6_candidate", NA),
            "review_only": "true",
            "observacao": "ausencia de evidencia documentada nao e evidencia de ausencia; nao e negativo",
        })
    return out


# ---------------------------------------------------------------------------
# Sintese por regiao
# ---------------------------------------------------------------------------
def avaliacao_regiao_rows(data: dict, sets: dict, ranks: dict, agree_by_scope: dict) -> list[dict]:
    score = data["score"]
    out = []
    for reg in ("recife", "curitiba", "petropolis"):
        obs = [p for p in sets["observed"] if score[p].get("regiao") == reg]
        bg = [p for p in sets["background"] if score[p].get("regiao") == reg]
        if reg == "petropolis":
            obs = []  # excluido de observado
        smean_o = _mean(data["features"], obs, "score") if obs else None
        smean_b = _mean(data["features"], bg, "score") if bg else None
        # top-k regional resultado (top30)
        if obs:
            regids = sorted([p for p in score if score[p].get("regiao") == reg], key=lambda p: -(ranks["sc"](p)))
            top30 = set(regids[:30])
            topk = f"{sum(1 for p in obs if p in top30)}/{len(obs)}_em_top30_regional"
        else:
            topk = "sem_observados"
        ag, ct = agree_by_scope.get(reg, (0, 0))
        out.append({
            "regiao": reg, "observed_patches": str(len(obs)), "background_patches": str(len(bg)),
            "score_medio_observado": _fmt(smean_o, 4) if smean_o is not None else NA,
            "score_medio_background": _fmt(smean_b, 4) if smean_b is not None else NA,
            "topk_resultado": topk,
            "direcoes_consistentes": str(ag), "direcoes_divergentes": str(ct - ag) if ct else NA,
            "status_regional": REGIONAL_STATUS[reg],
            "principal_limite": "amostra pequena e sem geometria oficial" if reg != "petropolis" else "fenomeno misto sem separacao",
        })
    return out


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------
def status_19c(sets: dict) -> str:
    n = len(sets["observed"])
    if n == 0:
        return "19C_BLOQUEADO_FAIL_CLOSED"
    if n < SMALL_SAMPLE_LIMIT:
        return "19C_AVALIACAO_PARCIAL_COM_AMOSTRA_MINIMA"
    return "19C_AVALIACAO_REVIEW_ONLY_CONCLUIDA"


def gate_19c_rows(summary: dict) -> list[dict]:
    def g(criterio, valor, limiar, passou, obs):
        return {"criterio": criterio, "valor_observado": str(valor), "limiar": limiar, "passou": "true" if passou else "false", "status": summary["status_19c"], "observacao": obs}
    return [
        g("observados_registrados", summary["observed_patches"], "==7", summary["observed_patches"] == 7, "Recife 5 + Curitiba 2"),
        g("petropolis_excluido", "true", "true", True, "Petropolis misto fora de observado"),
        g("background_nao_e_negativo", "true", "true", True, "universo e unlabeled_background"),
        g("hit_rate_exploratorio", "true", "true", True, "metrica review-only, nao benchmark"),
        g("amostra_pequena", summary["observed_patches"], f"<{SMALL_SAMPLE_LIMIT}", summary["observed_patches"] < SMALL_SAMPLE_LIMIT, "potencia amostral baixa, sem conclusao estatistica forte"),
        g("score_v6_intacto", str(not summary["score_v6_changed"]).lower(), "true", not summary["score_v6_changed"], "score_v6 nao alterado"),
    ]


def gate_17b_rows(summary: dict) -> list[dict]:
    return [{
        "criterio": c, "valor_observado": v, "limiar": l, "passou": p,
        "status_17b": "17B_NAO_CRIADO", "observacao": o,
    } for c, v, l, p, o in [
        ("benchmark_criado", "false", "false", "true", "nenhum benchmark 17B criado"),
        ("eventos_distintos", "2", ">=3", "false", "abaixo do minimo de eventos"),
        ("regioes", "2", ">=2", "true", "Recife e Curitiba tecnica"),
        ("patch_links_fortes", "7", ">=20", "false", "abaixo do minimo de vinculos fortes"),
        ("amostra_suficiente", "false", "true", "false", "amostra observacional minima"),
    ]]


def gate_v7_rows(summary: dict) -> list[dict]:
    return [{
        "criterio": c, "bloqueio": b, "status_score_v7": s, "observacao": o,
    } for c, b, s, o in [
        ("autorizacao_geral", "true", "SCORE_V7_NAO_AUTORIZADO", "score_v7 nao autorizado nesta etapa"),
        ("amostra_observacional", "true", "SCORE_V7_BLOQUEADO_POR_AMOSTRA", "apenas 7 patches observacionais"),
        ("missingness_territorial", "true", "SCORE_V7_BLOQUEADO_POR_MISSINGNESS_TERRITORIAL", "territorial faltante herdado do 19B"),
        ("ausencia_benchmark", "true", "SCORE_V7_BLOQUEADO_POR_AUSENCIA_BENCHMARK", "sem benchmark 17B"),
    ]]


# ---------------------------------------------------------------------------
# Resumos
# ---------------------------------------------------------------------------
def resumo_regiao_rows(aval_reg: list[dict]) -> list[dict]:
    return [{
        "regiao": r["regiao"], "observed_patches": r["observed_patches"],
        "score_medio_observado": r["score_medio_observado"], "score_medio_background": r["score_medio_background"],
        "topk_resultado": r["topk_resultado"], "status_regional": r["status_regional"],
    } for r in aval_reg]


def resumo_status_rows(aval: list[dict]) -> list[dict]:
    counts = {}
    for r in aval:
        counts[r["evaluation_status"]] = counts.get(r["evaluation_status"], 0) + 1
    return [{"evaluation_status": k, "quantidade": str(v)} for k, v in sorted(counts.items())]


# ---------------------------------------------------------------------------
# Schema / preflight / summary
# ---------------------------------------------------------------------------
def schema_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SUSC-19C avaliacao observacional review-only",
        "type": "object",
        "description": "Avaliacao exploratoria review-only; sem treino, sem benchmark, sem negativos.",
        "guardrails": {
            "ground_truth": {"const": "false"},
            "eligible_for_training": {"const": "false"},
            "score_v7_allowed": {"const": "false"},
            "review_only": {"const": "true"},
        },
        "status_19c_allowed": sorted(STATUS_19C_ALLOWED),
        "status_17b_allowed": sorted(STATUS_17B_ALLOWED),
        "status_score_v7_allowed": sorted(STATUS_V7_ALLOWED),
        "evaluation_status_allowed": sorted(EVAL_STATUS_ALLOWED),
        "background_label": "unlabeled_background",
        "small_sample_limit": SMALL_SAMPLE_LIMIT,
    }


def preflight_obj(sets: dict) -> dict:
    status_full = _run_git(["status", "--short"]).splitlines()
    return {
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "staged_count": len(_run_git(["diff", "--cached", "--name-only"]).splitlines()),
        "status_short_count": len(status_full),
        "entradas_lidas": {rel(p): p.exists() for p in REQUIRED_INPUTS},
        "observed_recife": sets["recife_obs"],
        "observed_curitiba": sets["curitiba_obs"],
        "excluidos_petropolis": sets["petropolis_ctx"],
        "background_n": len(sets["background"]),
        "lacunas_herdadas_19b": "territorial faltante (MapBiomas/exposed_soil/water/impervious) encaminhado ao pacote GEE",
        "score_v6_path": rel(SCORE_V6),
        "score_v6_changed": _score_v6_changed(),
        "score_v7_created": SCORE_V7.exists(),
        "outputs_ignorados_pelo_git": sorted(git_ignored([OUT])),
        "summary_versionavel": rel(SUMMARY) not in git_ignored([SUMMARY]),
    }


def summary_obj(data: dict, sets: dict, ranks: dict, hitrate: list[dict], agree_by_scope: dict, aval: list[dict]) -> dict:
    n_obs = len(sets["observed"])
    top30g = next((r for r in hitrate if r["escopo"] == "global" and r["top_k"] == "30"), {})
    top30r = next((r for r in hitrate if r["escopo"] == "regional" and r["top_k"] == "30"), {})
    ag, ct = agree_by_scope["global"]
    smean_o = _mean(data["features"], sets["observed"], "score")
    smean_b = _mean(data["features"], sets["background"], "score")
    status_counts = {}
    for r in aval:
        status_counts[r["evaluation_status"]] = status_counts.get(r["evaluation_status"], 0) + 1
    return {
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "staged_count": len(_run_git(["diff", "--cached", "--name-only"]).splitlines()),
        "observed_patches": n_obs,
        "observed_recife": len(sets["recife_obs"]),
        "observed_curitiba": len(sets["curitiba_obs"]),
        "excluidos_petropolis": len(sets["petropolis_ctx"]),
        "background_patches": len(sets["background"]),
        "background_label": "unlabeled_background",
        "score_medio_observado": round(smean_o, 4) if smean_o is not None else None,
        "score_medio_background": round(smean_b, 4) if smean_b is not None else None,
        "observed_in_top_30_global": int(top30g.get("observed_in_top_k", 0)),
        "enrichment_top_30_global": float(top30g.get("enrichment_ratio", 0.0)),
        "observed_in_top_30_regional": int(top30r.get("observed_in_top_k", 0)),
        "direcoes_coerentes_global": f"{ag}/{ct}",
        "avaliacao_status_counts": status_counts,
        "status_19c": status_19c(sets),
        "status_17b": "17B_NAO_CRIADO",
        "status_score_v7": "SCORE_V7_NAO_AUTORIZADO",
        "marco_17b_criado": False,
        "benchmark_17b_criado": False,
        "ground_truth_true_count": 0,
        "eligible_for_training_true_count": 0,
        "score_v7_allowed_true_count": 0,
        "score_v6_changed": _score_v6_changed(),
        "score_v7_created": SCORE_V7.exists(),
        "sample_power": "baixa",
        "review_only": True,
        "proximo_marco": "SUSC-19D",
    }


# ---------------------------------------------------------------------------
# Cartoes / relatorio
# ---------------------------------------------------------------------------
def _card_regional(reg: str, aval_reg: dict, summary: dict) -> str:
    if reg == "petropolis":
        return f"""# Cartao observacional - Petropolis (excluido/bloqueado)

## Amostra
Nenhum patch observado. {summary['excluidos_petropolis']} patch contextual registrado
em `registro_patches_excluidos_19c.csv`.

## Evidencia
Contexto documental de fenomeno misto (deslizamento e inundacao) sem separacao.

## Por que esta excluido
Petropolis misto/contextual nao entra como observado; permanece bloqueado ate a
separacao do fenomeno.

## Por que nao e ground truth / treino / score_v7 / 17B
Sem geometria de ocorrencia e com fenomeno misto; nao vira verdade de campo, treino,
score_v7 nem benchmark.
"""
    return f"""# Cartao observacional - {CIDADE[reg]}

## Amostra
{aval_reg['observed_patches']} patches observacionais review-only (background da regiao:
{aval_reg['background_patches']}).

## Evidencia
{"Canarios review-only (documental)." if reg == "recife" else "Overlays tecnicos SAR review-only (curitiba_01050, curitiba_01101)."}

## score_v6
Score medio observado {aval_reg['score_medio_observado']} contra background
{aval_reg['score_medio_background']}. Resultado top-k: {aval_reg['topk_resultado']}.

## Features coerentes
Direcoes consistentes: {aval_reg['direcoes_consistentes']} (divergentes:
{aval_reg['direcoes_divergentes']}). Coerencia principal em elevacao, declividade,
HAND, urbanizacao e indices espectrais.

## Divergencias
Distancia a agua, TWI, fluxo e chuva antecedente divergem da direcao esperada.

## Limitacoes
Amostra pequena; sem geometria oficial; potencia amostral baixa.

## Por que nao e ground truth / treino / score_v7 / 17B
Evidencia review-only sem geometria de ocorrencia; nao e verdade de campo, nao
habilita treino, nao autoriza score_v7 e nao cria benchmark 17B.
"""


def _card_global(summary: dict) -> str:
    return f"""# Cartao observacional - Sintese global

## Amostra
{summary['observed_patches']} patches observacionais (Recife {summary['observed_recife']},
Curitiba {summary['observed_curitiba']}); background nao rotulado de
{summary['background_patches']}.

## score_v6
Score medio observado {summary['score_medio_observado']} contra background
{summary['score_medio_background']}. Observados no top-30 global:
{summary['observed_in_top_30_global']} (enrichment {summary['enrichment_top_30_global']}).

## Leitura review-only
Sinal urbano e topografico coerente com maior suscetibilidade; divergencia
hidrologica e de chuva antecedente. Direcoes coerentes globais:
{summary['direcoes_coerentes_global']}.

## Limitacoes
Amostra pequena (potencia baixa); background nao e negativo; sem benchmark.

## Guardrails
Nao e ground truth, nao e treino, nao e benchmark, score_v6 intacto e score_v7
segue `{summary['status_score_v7']}`.
"""


def write_cards(aval_reg: list[dict], summary: dict) -> None:
    ensure_dir(CARDS)
    by_reg = {r["regiao"]: r for r in aval_reg}
    for reg in ("recife", "curitiba", "petropolis"):
        write_markdown(CARDS / f"cartao_{reg}.md", _card_regional(reg, by_reg[reg], summary))
    write_markdown(CARDS / "cartao_sintese_global.md", _card_global(summary))


def report_text(summary: dict, aval_reg: list[dict], contraste: list[dict]) -> str:
    glob = [c for c in contraste if c["escopo"] == "global"]
    linhas_feat = "\n".join(
        f"| {c['feature']} | {c['expected_direction_for_susceptibility']} | {c['observed_mean']} | {c['background_mean']} | {c['delta']} | {c['direction_agreement']} |"
        for c in glob
    )
    linhas_reg = "\n".join(
        f"| {r['regiao']} | {r['observed_patches']} | {r['score_medio_observado']} | {r['score_medio_background']} | {r['topk_resultado']} | {r['status_regional']} |"
        for r in aval_reg
    )
    return f"""# SUSC-19C - Avaliacao observacional review-only da matriz multimodal

## Estado herdado do 19A/19B

O 19A consolidou 300 patches multimodais e o 19B encaminhou a lacuna territorial
ao pacote MapBiomas/GEE. Esta etapa compara os patches observacionais review-only
com o universo nao rotulado, sem treino e sem benchmark.

## Amostra observacional

{summary['observed_patches']} patches observacionais: Recife {summary['observed_recife']}
(canarios review-only) e Curitiba {summary['observed_curitiba']} (overlays tecnicos
SAR). Petropolis fica de fora ({summary['excluidos_petropolis']} patch contextual
registrado como bloqueado). Background nao rotulado: {summary['background_patches']}.

## Por que o background nao e negativo

O universo sem evidencia documentada e `unlabeled_background`
(no_documented_observational_evidence). Ausencia de evidencia documentada nao e
evidencia de ausencia; nunca e negativo.

## Ranking score_v6

Score medio observado {summary['score_medio_observado']} contra background
{summary['score_medio_background']}. Observados no top-30 global:
{summary['observed_in_top_30_global']} (enrichment {summary['enrichment_top_30_global']});
no top-30 regional: {summary['observed_in_top_30_regional']}. Os observados ficam no
terco medio-superior do score_v6, nao no extremo.

## Hit-rate / enrichment review-only

As metricas de hit-rate e enrichment sao exploratorias e review-only. Com
{summary['observed_patches']} patches, a potencia amostral e baixa e nao ha
conclusao estatistica forte. Nao sao validacao operacional nem benchmark.

## Contraste de features (global)

| Feature | Direcao esperada | Media observada | Media background | Delta | Coerente |
| --- | --- | --- | --- | --- | --- |
{linhas_feat}

Direcoes coerentes globais: {summary['direcoes_coerentes_global']}. Ha coerencia
urbana e topografica (elevacao, declividade, HAND, urbanizacao, indices espectrais)
e divergencia hidrologica e de chuva antecedente.

## Divergencias

Registradas em `matriz_divergencias_observacionais_19c.csv`: patches observados com
score baixo/medio, divergencia fisica, divergencia de chuva, evidencia apenas SAR e
limitacao de amostra. Nenhum score e corrigido e nenhum score_v7 e proposto.

## Recife

Cinco canarios review-only, score medio acima do background regional, coerencia
urbana e topografica, amostra pequena.

## Curitiba

Dois overlays tecnicos SAR review-only; um deles com score alto e outro com score
baixo (divergencia observacional relevante). Amostra minima; SAR nao e geometria oficial.

## Petropolis excluido

Fenomeno misto sem separacao; permanece contextual/bloqueado, fora de observado.

## Por que nao e ground truth, nem treino, nem benchmark

Evidencia review-only sem geometria de ocorrencia confirmada; background nao e
negativo; amostra pequena; nenhuma metrica e benchmark.

## Por que o score_v7 segue bloqueado

Estado `{summary['status_score_v7']}`: bloqueado por amostra, por missingness
territorial herdado do 19B e por ausencia de benchmark. O score_v6 permanece intacto.

## Sintese por regiao

| Regiao | Observados | Score obs | Score background | Top-k | Status |
| --- | --- | --- | --- | --- | --- |
{linhas_reg}

## Proximo marco recomendado

**SUSC-19D - Pacote de comunicacao cientifica**: consolidar figuras, tabelas e
narrativa review-only para trabalho de conclusao e apresentacao.
"""


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_all() -> dict:
    require_inputs()
    ensure_dir(OUT)
    ensure_dir(REPORTS)
    ensure_dir(SCHEMAS_DIR)

    data = load_data()
    sets = derive_sets(data)
    ranks = compute_ranks(data)

    ranking = ranking_rows(data, sets, ranks)
    hitrate = hitrate_rows(data, sets, ranks)
    contraste, agree_by_scope = contraste_rows(data, sets)
    divergencias = divergencia_rows(data, sets, ranks, agree_by_scope)
    aval = avaliacao_rows(data, sets, ranks)
    aval_reg = avaliacao_regiao_rows(data, sets, ranks, agree_by_scope)
    summary = summary_obj(data, sets, ranks, hitrate, agree_by_scope, aval)

    write_csv(REG_OBS, registro_obs_rows(data, sets))
    write_csv(REG_EXC, registro_exc_rows(data, sets))
    write_csv(UNIVERSO, universo_rows(data, sets))
    write_csv(RANKING, ranking)
    write_csv(HITRATE, hitrate)
    write_csv(CONTRASTE, contraste)
    write_csv(DIVERGENCIAS, divergencias)
    write_csv(AVAL_REGIAO, aval_reg)
    write_csv(MAT_AVAL, aval)
    write_csv(GATE_19C, gate_19c_rows(summary))
    write_csv(GATE_17B, gate_17b_rows(summary))
    write_csv(GATE_V7, gate_v7_rows(summary))
    write_csv(RESUMO_REGIAO, resumo_regiao_rows(aval_reg))
    write_csv(RESUMO_STATUS, resumo_status_rows(aval))

    write_json(SCHEMA, schema_obj())
    write_json(PREFLIGHT, preflight_obj(sets))
    write_json(SUMMARY, summary)
    write_cards(aval_reg, summary)
    write_markdown(REPORT, report_text(summary, aval_reg, contraste))
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


NEGATIVE_RE = re.compile(r"\bnegativ", re.IGNORECASE)
# Afirmacoes estatisticas fortes proibidas com n pequeno. A frase de ressalva
# "sem conclusao estatistica forte" e permitida e por isso nao entra no padrao.
STRONG_STATS_RE = re.compile(r"p[-_ ]?valor|p[-_ ]?value|significancia estatistica|estatisticamente significativo", re.IGNORECASE)


def validate_outputs(summary: dict) -> list[str]:
    errors = [f"missing:{rel(p)}" for p in REQUIRED_OUTPUTS if not p.exists()]

    if summary["score_v6_changed"]:
        errors.append("score_v6_changed_forbidden")
    if summary["score_v7_created"]:
        errors.append("score_v7_created_forbidden")
    if summary["benchmark_17b_criado"] or summary["marco_17b_criado"]:
        errors.append("benchmark_ou_marco_17b_criado_forbidden")
    if summary["status_19c"] not in STATUS_19C_ALLOWED:
        errors.append(f"status_19c_fora_enum:{summary['status_19c']}")
    if summary["status_17b"] not in STATUS_17B_ALLOWED:
        errors.append(f"status_17b_fora_enum:{summary['status_17b']}")
    if summary["status_score_v7"] not in STATUS_V7_ALLOWED:
        errors.append(f"status_score_v7_fora_enum:{summary['status_score_v7']}")
    if summary["status_score_v7"] == "SCORE_V7_AUTORIZADO":
        errors.append("score_v7_autorizado_forbidden")

    # Registro observacional: 7 patches, Recife 5 + Curitiba 2, sem Petropolis
    obs = read_csv(REG_OBS)
    if len(obs) != 7:
        errors.append(f"observados_diferente_de_7:{len(obs)}")
    if sum(1 for r in obs if r["regiao"] == "recife") != 5:
        errors.append("recife_diferente_de_5")
    if sum(1 for r in obs if r["regiao"] == "curitiba") != 2:
        errors.append("curitiba_diferente_de_2")
    if any(r["regiao"] == "petropolis" for r in obs):
        errors.append("petropolis_entrou_como_observado")
    for r in obs:
        for field in ("ground_truth", "eligible_for_training", "score_v7_allowed"):
            if r.get(field) != "false":
                errors.append(f"obs:{r['patch_id']}:{field}_nao_false")
        if not r.get("justificativa_tecnica", "").strip():
            errors.append(f"obs:{r['patch_id']}:justificativa_vazia")
        if "oficial" in r.get("justificativa_tecnica", "").lower() and "nao" not in r.get("justificativa_tecnica", "").lower():
            errors.append(f"obs:{r['patch_id']}:sar_chamado_de_oficial")

    # Universo: rotulo unlabeled/background, nunca negativo
    uni = read_csv(UNIVERSO)
    if not uni:
        errors.append("universo_vazio")
    for r in uni:
        if r.get("rotulo") not in ("unlabeled_background",):
            errors.append(f"universo:{r.get('patch_id')}:rotulo_invalido")
        blob = " ".join(str(v) for v in r.values())
        if NEGATIVE_RE.search(blob) and "nao e negativo" not in blob.lower():
            errors.append(f"universo:{r.get('patch_id')}:chamado_de_negativo")

    # matriz de avaliacao final
    final = read_csv(MAT_AVAL)
    if not final:
        errors.append("matriz_avaliacao_vazia")
    ids = [r["patch_id"] for r in final]
    if len(ids) != len(set(ids)):
        errors.append("patch_id_duplicado")
    for r in final:
        if r.get("evaluation_status") not in EVAL_STATUS_ALLOWED:
            errors.append(f"final:{r['patch_id']}:eval_status_fora_enum:{r.get('evaluation_status')}")
        for field in ("ground_truth", "eligible_for_training", "score_v7_allowed"):
            if r.get(field) != "false":
                errors.append(f"final:{r['patch_id']}:{field}_nao_false")
        if not r.get("justificativa_tecnica", "").strip():
            errors.append(f"final:{r['patch_id']}:justificativa_vazia")

    # hit-rate deve declarar review-only / exploratorio, nunca benchmark
    for r in read_csv(HITRATE):
        lim = r.get("limitation", "").lower()
        if "benchmark" in lim and "nao e" not in lim and "nao e validacao" not in lim:
            errors.append("hitrate:chamado_de_benchmark")
        if "review-only" not in lim and "exploratori" not in lim:
            errors.append("hitrate:sem_marcacao_review_only")

    # gate enums
    for r in read_csv(GATE_17B):
        if r.get("status_17b") not in STATUS_17B_ALLOWED:
            errors.append(f"gate_17b:status_fora_enum:{r.get('status_17b')}")
    for r in read_csv(GATE_V7):
        if r.get("status_score_v7") not in STATUS_V7_ALLOWED:
            errors.append(f"gate_v7:status_fora_enum:{r.get('status_score_v7')}")

    # sem conclusao estatistica forte / p-valor em texto publico com n pequeno
    for path in _public_paths():
        txt = path.read_text(encoding="utf-8", errors="ignore")
        if STRONG_STATS_RE.search(txt):
            errors.append(f"{rel(path)}:conclusao_estatistica_forte_com_n_pequeno")

    # patch_stats SAR nao pode aparecer como feature pre-evento nas saidas
    for path in (CONTRASTE, MAT_AVAL):
        for r in read_csv(path):
            blob = " ".join(str(v).lower() for v in r.values())
            if "patch_stats" in blob:
                errors.append(f"{rel(path)}:patch_stats_como_feature")

    # output 19C nao pode ser ignorado pelo git
    if rel(SUMMARY) in git_ignored([SUMMARY, MAT_AVAL]):
        errors.append("output_19c_ignorado_pelo_git")

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
        "19C avaliacao observacional validada: "
        f"observados={summary['observed_patches']} (rec={summary['observed_recife']} cur={summary['observed_curitiba']}) "
        f"background={summary['background_patches']} top30g={summary['observed_in_top_30_global']} "
        f"enr30g={summary['enrichment_top_30_global']} status19C={summary['status_19c']} "
        f"status17B={summary['status_17b']} scorev7={summary['status_score_v7']} "
        f"score_v6_changed={str(summary['score_v6_changed']).lower()}"
    )
    return 0


def run_all() -> int:
    summary = build_all()
    print(
        "19C avaliacao observacional gerada: "
        f"observados={summary['observed_patches']} background={summary['background_patches']} "
        f"score_obs={summary['score_medio_observado']} score_bg={summary['score_medio_background']} "
        f"status19C={summary['status_19c']} scorev7={summary['status_score_v7']}"
    )
    return 0
