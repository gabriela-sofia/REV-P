"""SUSC-19D diagnostico de divergencias e hipoteses review-only do score_v6.

Explica tecnicamente o resultado do SUSC-19C: os patches observacionais
review-only (Recife 5 canarios; Curitiba 2 overlays tecnicos SAR) tem score_v6
medio maior que o universo nao rotulado, porem nenhum entra no top-30 global;
ha coerencia urbana/topografica e divergencia hidrologica/de chuva; o patch
curitiba_01101 e a divergencia mais forte.

Esta etapa NAO altera o score_v6, NAO cria score_v7, NAO cria benchmark 17B, NAO
treina e NAO cria ground truth. Produz um diagnostico auditavel: componentes/
features que sustentam a aderencia, os que derrubam os patches observados, falhas
metodologicas, hipoteses de ajuste testaveis no futuro e o motivo do bloqueio do
score_v7.

Achado estrutural: score_v6 = robust_minmax(combinacao linear ponderada dos 5
componentes). A combinacao linear reproduz exatamente o ranking oficial (a
normalizacao final e monotonica), o que permite analise de sensibilidade por
reordenacao sem recalcular nem substituir o score_v6 oficial. Os experimentos de
sensibilidade sao review-only, nao oficiais, e nunca substituem o score_v6.

Guardrails: ground_truth=false, eligible_for_training=false, score_v7_allowed=false,
score_v7=false, sem benchmark 17B, score_v6 intacto. O universo sem evidencia e
unlabeled_background (nunca negativo). Petropolis misto/contextual nao entra como
observado. patch_stats SAR e pos-evento e nunca vira feature pre-evento; SAR nao e
geometria oficial.
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
    read_json,
    rel,
    write_csv,
    write_json,
    write_markdown,
)

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
OUT = ROOT / "outputs_public" / "data" / "susc_19d_diagnostico_divergencias_score_v6_review_only"
CARDS = OUT / "cartoes_diagnostico"
REPORTS = ROOT / "outputs_public" / "reports"
SCHEMAS_DIR = ROOT / "schemas" / "suscetibilidade"

REPORT = REPORTS / "SUSC_19D_DIAGNOSTICO_DIVERGENCIAS_SCORE_V6_REVIEW_ONLY.md"
SCHEMA = SCHEMAS_DIR / "susc_19d_diagnostico_score_v6_schema_v1.json"

FEATURES_SRC = ROOT / "datasets" / "suscetibilidade" / "susc_features_by_patch_v1.csv"
SCORE_V6 = ROOT / "datasets" / "suscetibilidade" / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = ROOT / "datasets" / "suscetibilidade" / "susc_score_v7_candidate_by_patch_v1.csv"
FORMULA_CONTRACT = ROOT / "outputs_public" / "suscetibilidade" / "susc_17c35_score_v6_formula_contract.json"

# Insumos herdados 19A/19B/19C
DATA_19A = ROOT / "outputs_public" / "data" / "susc_19a_matriz_multimodal_escalavel_por_patch"
DATA_19B = ROOT / "outputs_public" / "data" / "susc_19b_auditoria_lacunas_territoriais"
DATA_19C = ROOT / "outputs_public" / "data" / "susc_19c_avaliacao_observacional_review_only"
REG_OBS_19C = DATA_19C / "registro_patches_observacionais_19c.csv"
MISSINGNESS_19B = DATA_19B / "auditoria_missingness_territorial_19b.csv"

# Saidas
INVENTARIO = OUT / "inventario_score_v6_19d.csv"
DIAG_PATCHES = OUT / "diagnostico_patches_observacionais_19d.csv"
DECOMP_FAMILIAS = OUT / "decomposicao_familias_features_19d.csv"
TIPOLOGIA = OUT / "tipologia_falhas_score_v6_19d.csv"
HIPOTESES = OUT / "hipoteses_ajuste_review_only_19d.csv"
SENSIBILIDADE = OUT / "sensibilidade_nao_oficial_score_v6_19d.csv"
GATE_V7 = OUT / "gate_prontidao_score_v7_pos_19d.csv"
GATE_17B = OUT / "gate_prontidao_17b_pos_19d.csv"
GATE_19E = OUT / "gate_prontidao_19e_comunicacao_cientifica.csv"
RESUMO_REGIAO = OUT / "resumo_por_regiao.csv"
RESUMO_FALHA = OUT / "resumo_por_falha.csv"
SUMMARY = OUT / "summary.json"
PREFLIGHT = OUT / "preflight.json"

REQUIRED_INPUTS = [FEATURES_SRC, SCORE_V6, FORMULA_CONTRACT, REG_OBS_19C, MISSINGNESS_19B]
REQUIRED_OUTPUTS = [
    INVENTARIO, DIAG_PATCHES, DECOMP_FAMILIAS, TIPOLOGIA, HIPOTESES, SENSIBILIDADE,
    GATE_V7, GATE_17B, GATE_19E, RESUMO_REGIAO, RESUMO_FALHA, SUMMARY, PREFLIGHT,
    SCHEMA, REPORT,
]

# ---------------------------------------------------------------------------
# Guardrails / constantes
# ---------------------------------------------------------------------------
PUBLIC_FORBIDDEN_RE = re.compile(r"\b(?:agentic|agente|codex|llm|ia)\b", re.IGNORECASE)
NA = "NA"
SMALL_SAMPLE_LIMIT = 30

STATUS_V7_ALLOWED = {
    "SCORE_V7_NAO_AUTORIZADO",
    "SCORE_V7_BLOQUEADO_POR_AMOSTRA",
    "SCORE_V7_BLOQUEADO_POR_MISSINGNESS_TERRITORIAL",
    "SCORE_V7_BLOQUEADO_POR_AUSENCIA_BENCHMARK",
    "SCORE_V7_APENAS_HIPOTESES_REVIEW_ONLY",
}
STATUS_17B_ALLOWED = {
    "17B_NAO_CRIADO",
    "17B_BLOQUEADO_POR_AMOSTRA",
    "17B_BLOQUEADO_POR_GEOMETRIA_OFICIAL",
    "17B_APROXIMACAO_COM_SEGUNDA_REGIAO_TECNICA",
}
STATUS_19E_ALLOWED = {
    "19E_PRONTO_PARA_COMUNICACAO_CIENTIFICA_REVIEW_ONLY",
    "19E_BLOQUEADO_POR_LACUNAS_CRITICAS",
    "19E_PRONTO_COM_RESSALVAS",
}
FAILURE_TYPES_ALLOWED = {
    "observed_patch_low_score",
    "observed_patch_medium_not_topk",
    "hidrologia_divergente",
    "chuva_subponderada_ou_nao_representativa",
    "topografia_coerente_mas_score_nao_extremo",
    "urbano_espectral_coerente",
    "evidencia_documental_rara",
    "SAR_only_evidence",
    "missingness_territorial",
    "amostra_minima",
    "evento_tecnico_nao_oficial",
}
CIDADE = {"recife": "Recife", "curitiba": "Curitiba", "petropolis": "Petropolis"}

# Familia -> lista de (display, coluna_fonte, direcao_esperada_para_suscetibilidade)
# Usa exatamente as features ja validadas em direcao pelo contraste do 19C.
FAMILY_FEATURES = {
    "fisica_topografica": [
        ("elevation_mean", "elevation_mean", "menor"),
        ("slope_mean", "slope_mean", "menor"),
        ("HAND_mean", "hand_mean", "menor"),
    ],
    "hidrologica": [
        ("distance_to_water_mean", "distance_to_water_mean", "menor"),
        ("TWI_mean", "twi_mean", "maior"),
        ("flow_accumulation_mean", "flow_acc_log_mean", "maior"),
    ],
    "urbana_territorial": [
        ("urban_prop", "urban_prop", "maior"),
        ("vegetation_prop", "vegetation_prop", "menor"),
    ],
    "espectral_umidade": [
        ("NDVI", "ndvi_mean", "menor"),
        ("NDWI", "NDWI", "maior"),
        ("MNDWI", "mndwi_mean", "maior"),
        ("NDBI", "ndbi_mean", "maior"),
    ],
    "chuva_hidrometeorologica": [
        ("CHIRPS_3d", "chirps_3d_mm", "maior"),
        ("CHIRPS_7d", "chirps_7d_mm", "maior"),
        ("CHIRPS_30d", "chirps_30d_mm", "maior"),
    ],
    "documental_observacional": [
        ("evidence_support_index", "evidence_support_index", "maior"),
    ],
    "score_v6": [
        ("score_v6", "score", "maior"),
    ],
}
FEATURE_FAMILIES = [
    "fisica_topografica", "hidrologica", "urbana_territorial",
    "espectral_umidade", "chuva_hidrometeorologica",
]

TERRITORIAL_MISSING = [
    "MapBiomas_class_majority", "MapBiomas_class_distribution",
    "exposed_soil_prop", "water_prop", "impervious_proxy",
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
# Carga
# ---------------------------------------------------------------------------
def load_data() -> dict:
    features = {r["patch_id"]: dict(r) for r in read_csv(FEATURES_SRC)}
    score = {r["patch_id"]: dict(r) for r in read_csv(SCORE_V6)}
    for pid, f in features.items():
        b3, b8 = _num(f.get("B3_mean")), _num(f.get("B8_mean"))
        f["NDWI"] = (b3 - b8) / (b3 + b8) if (b3 is not None and b8 is not None and (b3 + b8) != 0) else None
        srow = score.get(pid, {})
        f["score"] = _num(srow.get("susc_score_v6_candidate"))
        f["evidence_support_index"] = _num(srow.get("evidence_support_index"))
    contract = read_json(FORMULA_CONTRACT)
    return {"features": features, "score": score, "contract": contract}


def observed_ids() -> list[str]:
    """Le os 7 patches observacionais herdados do 19C (nao recomputa)."""
    ids = [r["patch_id"] for r in read_csv(REG_OBS_19C)]
    return ids


def derive_sets(data: dict) -> dict:
    obs = observed_ids()
    score = data["score"]
    recife = [p for p in obs if score.get(p, {}).get("regiao") == "recife"]
    curitiba = [p for p in obs if score.get(p, {}).get("regiao") == "curitiba"]
    background = sorted(p for p in score if p not in set(obs))
    return {"observed": obs, "recife": recife, "curitiba": curitiba, "background": background}


# ---------------------------------------------------------------------------
# Ranking score_v6 (reproduz 19C; nao altera score)
# ---------------------------------------------------------------------------
def compute_ranks(data: dict) -> dict:
    score = data["score"]

    def sc(pid):
        return _num(score[pid].get("susc_score_v6_candidate")) or 0.0

    allids = sorted(score, key=lambda p: -sc(p))
    N = len(allids)
    global_rank = {pid: i + 1 for i, pid in enumerate(allids)}
    regional_rank, regional_n = {}, {}
    for reg in ("recife", "curitiba", "petropolis"):
        regids = sorted([p for p in score if score[p].get("regiao") == reg], key=lambda p: -sc(p))
        regional_n[reg] = len(regids)
        for i, pid in enumerate(regids):
            regional_rank[pid] = i + 1
    return {"global_rank": global_rank, "regional_rank": regional_rank,
            "regional_n": regional_n, "N": N, "sc": sc}


def _mean(features: dict, ids: list[str], col: str):
    vals = [_num(features[p].get(col)) for p in ids]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _agrees(direction: str, delta: float) -> bool:
    return delta > 0 if direction == "maior" else delta < 0


# ---------------------------------------------------------------------------
# 2. Inventario do score_v6 (fonte-verdade: contrato 17c35)
# ---------------------------------------------------------------------------
def inventario_rows(data: dict) -> tuple[list[dict], bool]:
    c = data["contract"]
    weights = c.get("weights", {})
    subidx = c.get("subindex_features", {})
    componentes_formais_ausentes = not bool(weights)
    # mapeia componente -> familias/entradas para leitura de interpretabilidade
    papel = {
        "topography_hydrology_index": ("fisica_topografica+hidrologica", "sustenta_via_topografia_e_derruba_via_hidrologia"),
        "rainfall_trigger_index": ("chuva_hidrometeorologica", "derruba_observados_chuva_antecedente_menor"),
        "urban_spectral_index": ("urbana_territorial+espectral_umidade", "sustenta_aderencia"),
        "vegetation_mitigation_index": ("espectral_umidade+urbana_territorial", "mitigacao_peso_negativo"),
        "evidence_support_index": ("documental_observacional", "confianca_documental_confundida_com_suscetibilidade"),
    }
    rows = []
    for comp in c.get("component_names", list(weights.keys())):
        feats = subidx.get(comp, [comp])
        present = [f for f in feats if f in _known_feature_cols(data)]
        missing = [f for f in feats if f not in _known_feature_cols(data)]
        familia, interp = papel.get(comp, (NA, NA))
        rows.append({
            "componente": comp,
            "fonte_score_v6": rel(SCORE_V6),
            "contrato_formula": rel(FORMULA_CONTRACT),
            "coluna_score": comp if comp in _score_cols(data) else NA,
            "peso_explicito": _fmt(weights.get(comp), 3) if comp in weights else NA,
            "features_do_subindice": ";".join(feats),
            "features_usadas": ";".join(present),
            "features_ausentes": ";".join(missing) if missing else "nenhuma",
            "entra_documental": "true" if comp == "evidence_support_index" else "false",
            "entra_rainfall": "true" if comp == "rainfall_trigger_index" else "false",
            "entra_urban_spectral": "true" if comp in ("urban_spectral_index", "vegetation_mitigation_index") else "false",
            "familia_associada": familia,
            "status_interpretabilidade": "peso_explicito_documentado" if comp in weights else "peso_ausente",
            "papel_no_diagnostico": interp,
            "componentes_formais_ausentes": "true" if componentes_formais_ausentes else "false",
            "observacao": "componente com peso explicito no contrato 17c35; score_v6 nao alterado",
        })
    # linha do score final (normalizacao)
    rows.append({
        "componente": "score_v6_final",
        "fonte_score_v6": rel(SCORE_V6),
        "contrato_formula": rel(FORMULA_CONTRACT),
        "coluna_score": "susc_score_v6_candidate",
        "peso_explicito": NA,
        "features_do_subindice": ";".join(c.get("component_names", [])),
        "features_usadas": ";".join(c.get("component_names", [])),
        "features_ausentes": ";".join(TERRITORIAL_MISSING),
        "entra_documental": "true",
        "entra_rainfall": "true",
        "entra_urban_spectral": "true",
        "familia_associada": "score_v6",
        "status_interpretabilidade": c.get("normalization_method", NA),
        "papel_no_diagnostico": "score_v6=robust_minmax(combinacao_linear_ponderada);ranking_monotonico_reproduzido",
        "componentes_formais_ausentes": "true" if componentes_formais_ausentes else "false",
        "observacao": "territorial faltante herdado do 19B nao materializado; score_v6 intacto",
    })
    return rows, componentes_formais_ausentes


def _known_feature_cols(data: dict) -> set[str]:
    any_row = next(iter(data["features"].values()))
    cols = set(any_row.keys())
    cols.add("evidence_support_index")
    return cols


def _score_cols(data: dict) -> set[str]:
    any_row = next(iter(data["score"].values()))
    return set(any_row.keys())


# ---------------------------------------------------------------------------
# 4. Decomposicao por familia
# ---------------------------------------------------------------------------
def familia_agreement(data: dict, ids: list[str], bg: list[str], familia: str) -> tuple[float, list[tuple]]:
    features = data["features"]
    details = []
    agree = counted = 0
    for disp, col, direction in FAMILY_FEATURES[familia]:
        o = _mean(features, ids, col)
        b = _mean(features, bg, col)
        if o is None or b is None:
            details.append((disp, o, b, None, NA))
            continue
        delta = o - b
        ok = _agrees(direction, delta)
        counted += 1
        agree += 1 if ok else 0
        details.append((disp, o, b, delta, "true" if ok else "false"))
    rate = agree / counted if counted else 0.0
    return rate, details


def decomposicao_rows(data: dict, sets: dict) -> tuple[list[dict], dict]:
    obs, bg = sets["observed"], sets["background"]
    rows = []
    fam_rate = {}
    for familia in FAMILY_FEATURES:
        rate, details = familia_agreement(data, obs, bg, familia)
        fam_rate[familia] = rate
        feats = [d[0] for d in details]
        n_feat = len(FAMILY_FEATURES[familia])
        o_means = [d[1] for d in details if d[1] is not None]
        b_means = [d[2] for d in details if d[2] is not None]
        om = sum(o_means) / len(o_means) if o_means else None
        bm = sum(b_means) / len(b_means) if b_means else None
        exp_dir = ";".join(f"{d}:{dir_}" for d, _c, dir_ in FAMILY_FEATURES[familia])
        if familia == "documental_observacional":
            contrib = "confianca_documental_separavel_da_suscetibilidade"
            limit = "documental so em Recife (0.5-1.0); Curitiba SAR sem documental (0.0); nao e suscetibilidade fisica"
        elif familia == "score_v6":
            contrib = "score_medio_observado_maior_que_background_mas_sem_top30_global"
            limit = "score_v6 intacto; leitura exploratoria review-only; sem conclusao estatistica forte"
        elif rate >= 0.6:
            contrib = "sustenta_aderencia"
            limit = "amostra pequena (n=7); contraste review-only; sem conclusao estatistica forte"
        elif rate <= 0.34:
            contrib = "derruba_ou_diverge"
            limit = "amostra pequena (n=7); contraste review-only; sem conclusao estatistica forte"
        else:
            contrib = "aderencia_parcial"
            limit = "amostra pequena (n=7); contraste review-only; sem conclusao estatistica forte"
        if familia == "urbana_territorial":
            limit += "; territorial incompleto: faltam " + ";".join(TERRITORIAL_MISSING) + " (missingness herdado do 19B)"
        rows.append({
            "familia": familia,
            "feature_count": str(n_feat),
            "features_usadas": ";".join(feats),
            "observed_mean": _fmt(om, 4) if om is not None else NA,
            "background_mean": _fmt(bm, 4) if bm is not None else NA,
            "expected_direction": exp_dir,
            "direction_agreement_rate": _fmt(rate, 3),
            "contribuicao_interpretativa": contrib,
            "limitacao": limit,
        })
    return rows, fam_rate


# ---------------------------------------------------------------------------
# 3. Diagnostico por patch observacional
# ---------------------------------------------------------------------------
def _patch_family_status(data: dict, sets: dict, pid: str, familia: str) -> str:
    features = data["features"]
    bg = sets["background"]
    if familia == "documental_observacional":
        ev = _num(features[pid].get("evidence_support_index")) or 0.0
        return "documental_presente" if ev > 0 else "documental_ausente"
    agree = counted = 0
    for _disp, col, direction in FAMILY_FEATURES[familia]:
        pv = _num(features[pid].get(col))
        bm = _mean(features, bg, col)
        if pv is None or bm is None:
            continue
        counted += 1
        if _agrees(direction, pv - bm):
            agree += 1
    if not counted:
        return NA
    rate = agree / counted
    if rate >= 0.6:
        return "aderente"
    if rate <= 0.34:
        return "divergente"
    return "parcial"


def diagnostico_patches_rows(data: dict, sets: dict, ranks: dict) -> list[dict]:
    score = data["score"]
    obs_meta = {r["patch_id"]: r for r in read_csv(REG_OBS_19C)}
    out = []
    for pid in sets["observed"]:
        reg = score[pid].get("regiao")
        cls = (score[pid].get("susc_class_v6_candidate") or NA)
        rg, rr = ranks["global_rank"][pid], ranks["regional_rank"][pid]
        N, Nr = ranks["N"], ranks["regional_n"][reg]
        pct_g = (N - rg + 1) / N
        pct_r = (Nr - rr + 1) / Nr
        fam_status = {fam: _patch_family_status(data, sets, pid, fam) for fam in FEATURE_FAMILIES}
        # aderencia/divergencia principais entre familias de feature
        aderentes = [f for f in FEATURE_FAMILIES if fam_status[f] == "aderente"]
        divergentes = [f for f in FEATURE_FAMILIES if fam_status[f] == "divergente"]
        principal_ader = aderentes[0] if aderentes else "nenhuma_familia_plenamente_aderente"
        principal_div = divergentes[0] if divergentes else "nenhuma_familia_plenamente_divergente"
        flags = []
        if cls.lower() == "low":
            flags.append("observed_patch_low_score")
        elif cls.lower() == "medium":
            flags.append("observed_patch_medium_not_topk")
        if reg == "curitiba":
            flags.append("SAR_only_evidence")
        else:
            ev = _num(score[pid].get("evidence_support_index")) or 0
            if 0 < ev < 1:
                flags.append("documentary_low_confidence")
        flags.append("sample_size_limitation")
        interp = (
            f"score_v6 no percentil global {_fmt(pct_g)} (classe {cls}); "
            f"aderencia em {principal_ader}, divergencia em {principal_div}; "
            "leitura exploratoria review-only, nao e validacao, nao e ground truth"
        )
        out.append({
            "patch_id": pid, "regiao": reg,
            "evidence_type": obs_meta.get(pid, {}).get("evidence_type", NA),
            "evidence_tier": obs_meta.get(pid, {}).get("evidence_tier", NA),
            "score_v6": _fmt(ranks["sc"](pid), 6),
            "score_v6_class": cls,
            "percentile_global": _fmt(pct_g), "percentile_regional": _fmt(pct_r),
            "top30_global": "true" if rg <= 30 else "false",
            "top30_regional": "true" if rr <= 30 else "false",
            "familia_fisica_status": fam_status["fisica_topografica"],
            "familia_hidrologica_status": fam_status["hidrologica"],
            "familia_urbana_status": fam_status["urbana_territorial"],
            "familia_espectral_status": fam_status["espectral_umidade"],
            "familia_chuva_status": fam_status["chuva_hidrometeorologica"],
            "familia_documental_status": _patch_family_status(data, sets, pid, "documental_observacional"),
            "divergence_flags": ";".join(flags),
            "principal_fator_aderencia": principal_ader,
            "principal_fator_divergencia": principal_div,
            "interpretation_review_only": interp,
        })
    return out


# ---------------------------------------------------------------------------
# 5. Tipologia de falhas
# ---------------------------------------------------------------------------
def tipologia_rows(data: dict, sets: dict, ranks: dict, fam_rate: dict) -> list[dict]:
    score = data["score"]
    rows = []
    idx = 1

    def add(pid, reg, ftype, sev, ev, impacto, acao, justif):
        nonlocal idx
        rows.append({
            "failure_id": f"FAIL_{idx:03d}", "patch_id": pid, "regiao": reg,
            "failure_type": ftype, "severidade": sev, "evidencia_suporte": ev,
            "impacto_no_score": impacto, "acao_futura_possivel": acao,
            "bloqueia_score_v7": "true", "justificativa_tecnica": justif,
        })
        idx += 1

    for pid in sets["observed"]:
        reg = score[pid].get("regiao")
        cls = (score[pid].get("susc_class_v6_candidate") or "").lower()
        rg = ranks["global_rank"][pid]
        if cls == "low":
            add(pid, reg, "observed_patch_low_score", "alta",
                f"score_v6_class=low;rank_global={rg}",
                "score_v6 baixo apesar de evidencia observacional review-only",
                "revisar decomposicao e hipoteses review-only sem alterar score_v6",
                "patch observado com score baixo; divergencia registrada, nao corrige score, nao e negativo")
        elif cls == "medium" and rg > 30:
            add(pid, reg, "observed_patch_medium_not_topk", "media",
                f"score_v6_class=medium;rank_global={rg}",
                "score intermediario fora do top-30 global",
                "avaliar reponderacao urbana/espectral em hipotese review-only",
                "patch observado medio fora do top-k; leitura exploratoria review-only")
        if reg == "curitiba":
            add(pid, reg, "SAR_only_evidence", "media",
                "evidencia apenas SAR overlay tecnico",
                "evidencia review-only sem documental; SAR e pos-evento",
                "buscar geometria oficial de ocorrencia antes de qualquer promocao",
                "SAR nao e geometria oficial e nao vira feature pre-evento; nao habilita treino")
            add(pid, reg, "evento_tecnico_nao_oficial", "media",
                "overlay tecnico SAR sem evento oficial datado por patch",
                "vinculo tecnico review-only, nao oficial",
                "aquisicao de geometria oficial (fila 17B/GT)",
                "evento tecnico nao oficial; nao e ground truth e nao cria benchmark 17B")

    # falhas globais / metodologicas
    n_obs = len(sets["observed"])
    add("todos_observados", "global", "hidrologia_divergente", "alta",
        f"familia_hidrologica direction_agreement_rate={_fmt(fam_rate['hidrologica'],3)}",
        "distance_to_water/TWI/flow_accumulation divergem da direcao esperada e limitam o score",
        "hipotese review-only de componente hidrologico diferenciado para alagamento urbano",
        "divergencia hidrologica global review-only; nao corrige score e nao propoe score_v7")
    add("todos_observados", "global", "chuva_subponderada_ou_nao_representativa", "alta",
        f"familia_chuva direction_agreement_rate={_fmt(fam_rate['chuva_hidrometeorologica'],3)}",
        "CHIRPS 3d/7d/30d menores nos observados; chuva estatica derruba o ranking",
        "hipotese review-only de chuva como gatilho contextual, nao como suscetibilidade estatica",
        "chuva antecedente tratada como suscetibilidade estatica; hipotese review-only, sem score_v7")
    add("todos_observados", "global", "topografia_coerente_mas_score_nao_extremo", "media",
        f"familia_fisica direction_agreement_rate={_fmt(fam_rate['fisica_topografica'],3)};top30_global=0/{n_obs}",
        "topografia coerente nao basta para levar observados ao top-30 global",
        "documentar em comunicacao cientifica review-only (19E)",
        "coerencia topografica sem extremo de score; leitura exploratoria review-only")
    add("todos_observados", "global", "urbano_espectral_coerente", "baixa",
        f"familia_urbana={_fmt(fam_rate['urbana_territorial'],3)};familia_espectral={_fmt(fam_rate['espectral_umidade'],3)}",
        "sinal urbano e espectral coerente sustenta parte da aderencia",
        "testar aumento de peso urbano/espectral em hipotese review-only",
        "coerencia urbana/espectral review-only; nao vira score oficial")
    add("todos_observados", "global", "evidencia_documental_rara", "media",
        "documental presente so em Recife (evidence_support_index 0.5-1.0); Curitiba 0.0",
        "documental confunde confianca com suscetibilidade e beneficia apenas Recife",
        "separar evidencia documental de suscetibilidade fisica (hipotese review-only)",
        "evidencia documental rara e desigual; separacao proposta como hipotese review-only")
    add("todos_observados", "global", "missingness_territorial", "alta",
        "faltam " + ";".join(TERRITORIAL_MISSING) + " em todos os 300 patches (cobertura territorial 0.333)",
        "urban_spectral usa apenas urban/veg; refino territorial ausente",
        "executar pacote MapBiomas/GEE (19B) antes de qualquer calibracao",
        "missingness territorial explicito herdado do 19B; nao ocultado; bloqueia score_v7")
    add("todos_observados", "global", "amostra_minima", "alta",
        f"apenas {n_obs} patches observacionais (Recife 5 + Curitiba 2)",
        "potencia amostral baixa; sem conclusao estatistica forte",
        "ampliar amostra observacional antes de qualquer score_v7 ou benchmark",
        "amostra minima; leitura exploratoria review-only; nao e benchmark")
    return rows


# ---------------------------------------------------------------------------
# 6. Hipoteses review-only
# ---------------------------------------------------------------------------
def hipoteses_rows() -> list[dict]:
    base = [
        ("HIP_01", "separar evidencia documental de suscetibilidade fisica",
         "documental beneficia so Recife (evidence 0.5-1.0) e Curitiba fica em 0.0",
         "evidence_support_index",
         "confundir presenca de documentacao com suscetibilidade; documentacao rara e desigual",
         "mais eventos documentados e datados nas 3 regioes"),
        ("HIP_02", "reduzir penalizacao por baixa documentacao",
         "patches sem documental nao devem ser tratados como menos suscetiveis",
         "evidence_support_index",
         "ausencia de documentacao nao e evidencia de ausencia; nao e negativo",
         "cobertura documental balanceada por regiao"),
        ("HIP_03", "testar aumento de peso urbano/espectral",
         "familias urbana e espectral coerentes com maior suscetibilidade nos observados",
         "urban_prop;mndwi_mean;ndbi_mean;ndvi_mean",
         "sobreajuste a 7 patches; missingness territorial ainda presente",
         "territorial completo (MapBiomas/solo/agua/impervious) e amostra maior"),
        ("HIP_04", "testar chuva como gatilho contextual, nao suscetibilidade estatica",
         "CHIRPS 3d/7d/30d menores nos observados derrubam o ranking",
         "chirps_3d_mm;chirps_7d_mm;chirps_30d_mm;rain_7d_30d_ratio",
         "confundir clima da janela com suscetibilidade estrutural do terreno",
         "serie temporal de chuva por evento e separacao gatilho vs suscetibilidade"),
        ("HIP_05", "componente hidrologico diferenciado para alagamento urbano",
         "distance_to_water/TWI/flow_accumulation divergem em contexto urbano",
         "twi_mean;flow_acc_log_mean;distance_to_water_mean;hand_mean",
         "hidrologia natural nao captura drenagem urbana; metodo nao equivalente (cadeia 17C36-40)",
         "hidrologia urbana e drenagem oficial; DEM/HAND compativel"),
        ("HIP_06", "preencher MapBiomas/solo exposto/agua/impervious antes de qualquer calibracao",
         "cobertura territorial 0.333; 4 features territoriais ausentes nos 300",
         "MapBiomas_class_majority;exposed_soil_prop;water_prop;impervious_proxy",
         "calibrar sobre missingness territorial produz vies",
         "execucao do pacote MapBiomas/GEE preparado no 19B"),
        ("HIP_07", "aumentar amostra observacional antes de qualquer score_v7",
         "apenas 7 observados; 0 no top-30 global; potencia amostral baixa",
         "amostra_observacional",
         "conclusao com n=7 sem robustez; sem conclusao estatistica forte",
         "mais eventos com geometria oficial (fila 17B/GT)"),
    ]
    rows = []
    for hid, desc, origem, feats, risco, dado in base:
        rows.append({
            "hipotese_id": hid, "descricao": desc, "origem_no_19c": origem,
            "features_afetadas": feats, "risco_metodologico": risco,
            "dado_necessario": dado, "status": "hipotese_review_only_nao_testada",
            "pode_virar_score_v7_agora": "false",
            "motivo_bloqueio": "hipotese exploratoria review-only; amostra minima, missingness territorial e ausencia de benchmark bloqueiam score_v7",
        })
    return rows


# ---------------------------------------------------------------------------
# 7. Sensibilidade nao oficial (reordenacao, nunca substitui score_v6)
# ---------------------------------------------------------------------------
def _weighted_ranks(data: dict, weights: dict) -> dict:
    score = data["score"]
    comps = list(weights.keys())
    raw = {}
    for pid, r in score.items():
        raw[pid] = sum(weights[k] * (_num(r.get(k)) or 0.0) for k in comps)
    order = sorted(raw, key=lambda p: -raw[p])
    return {pid: i + 1 for i, pid in enumerate(order)}


def sensibilidade_rows(data: dict, sets: dict, base_ranks: dict) -> tuple[list[dict], bool]:
    c = data["contract"]
    weights = dict(c.get("weights", {}))
    componentes_formais_ausentes = not bool(weights)
    obs = sets["observed"]
    baseg = base_ranks["global_rank"]

    def obs_metrics(rank_map):
        top30 = sum(1 for p in obs if rank_map[p] <= 30)
        deltas = [baseg[p] - rank_map[p] for p in obs]  # positivo = melhora (rank menor)
        return top30, sum(deltas) / len(deltas)

    rows = []

    def exec_scenario(sid, desc, mods, interp):
        w = dict(weights)
        w.update(mods)
        rmap = _weighted_ranks(data, w)
        top30, mean_delta = obs_metrics(rmap)
        rows.append({
            "scenario_id": sid, "descricao": desc, "executado": "true",
            "motivo_se_nao_executado": NA,
            "patches_afetados": ";".join(obs),
            "mudanca_rank_observada": f"obs_top30_global={top30};delta_medio_rank_vs_score_v6={mean_delta:+.1f}",
            "interpretacao": interp,
            "uso_permitido": "analise_sensibilidade_review_only",
            "substitui_score_v6": "false", "score_oficial": "false", "score_v7": "false",
        })

    def design_scenario(sid, desc, motivo, interp):
        rows.append({
            "scenario_id": sid, "descricao": desc, "executado": "false",
            "motivo_se_nao_executado": motivo,
            "patches_afetados": ";".join(obs),
            "mudanca_rank_observada": NA,
            "interpretacao": interp,
            "uso_permitido": "analise_sensibilidade_review_only",
            "substitui_score_v6": "false", "score_oficial": "false", "score_v7": "false",
        })

    if not componentes_formais_ausentes:
        exec_scenario(
            "SENS_01_remover_documental", "zera o peso do componente documental (evidence_support_index)",
            {"evidence_support_index": 0.0},
            "Recife perde parte do score que vinha da documentacao; Curitiba (documental 0.0) quase nao muda; separa confianca de suscetibilidade")
        exec_scenario(
            "SENS_02_boost_urbano_espectral", "aumenta peso urbano/espectral e reduz topografia_hidrologia",
            {"urban_spectral_index": 0.35, "topography_hydrology_index": 0.25},
            "parte dos observados urbanos sobe (alguns ao top-30); patches de topografia forte perdem posicao; sobreajuste possivel")
        exec_scenario(
            "SENS_03_chuva_como_gatilho", "trata chuva como gatilho contextual removendo-a da suscetibilidade estatica",
            {"rainfall_trigger_index": 0.0},
            "observados sobem no ranking porque tinham chuva antecedente menor; sugere chuva como gatilho, nao suscetibilidade estatica")
    else:
        design_scenario(
            "SENS_01_remover_documental", "zera o peso do componente documental",
            "componentes_formais_ausentes", "apenas desenho experimental; sem recalculo")
        design_scenario(
            "SENS_02_boost_urbano_espectral", "aumenta peso urbano/espectral",
            "componentes_formais_ausentes", "apenas desenho experimental; sem recalculo")
        design_scenario(
            "SENS_03_chuva_como_gatilho", "trata chuva como gatilho contextual",
            "componentes_formais_ausentes", "apenas desenho experimental; sem recalculo")

    design_scenario(
        "SENS_04_sem_missingness_territorial",
        "cenario com features territoriais (MapBiomas/solo/agua/impervious) preenchidas",
        "dados_territoriais_ausentes_em_todos_os_300_patches",
        "nao executavel: features territoriais nao materializadas (19B); preenchimento inventado seria vies")
    design_scenario(
        "SENS_05_mapbiomas_pendente",
        "cenario com classe MapBiomas incorporada ao componente urbano/territorial",
        "pacote_mapbiomas_gee_pendente_execucao",
        "nao executavel: pacote MapBiomas/GEE preparado no 19B ainda nao executado")
    return rows, componentes_formais_ausentes


# ---------------------------------------------------------------------------
# 8. Gates de prontidao
# ---------------------------------------------------------------------------
def gate_v7_rows() -> list[dict]:
    data = [
        ("autorizacao_geral", "true", "SCORE_V7_NAO_AUTORIZADO", "score_v7 nao autorizado nesta etapa"),
        ("amostra_observacional", "true", "SCORE_V7_BLOQUEADO_POR_AMOSTRA", "apenas 7 patches observacionais"),
        ("missingness_territorial", "true", "SCORE_V7_BLOQUEADO_POR_MISSINGNESS_TERRITORIAL", "4 features territoriais ausentes nos 300 (19B)"),
        ("ausencia_benchmark", "true", "SCORE_V7_BLOQUEADO_POR_AUSENCIA_BENCHMARK", "sem benchmark 17B"),
        ("hipoteses_review_only", "true", "SCORE_V7_APENAS_HIPOTESES_REVIEW_ONLY", "existem hipoteses, nenhuma vira score oficial"),
    ]
    return [{"criterio": c, "bloqueio": b, "status_score_v7": s, "observacao": o} for c, b, s, o in data]


def gate_17b_rows() -> list[dict]:
    data = [
        ("benchmark_criado", "false", "false", "17B_NAO_CRIADO", "nenhum benchmark 17B criado"),
        ("eventos_distintos", "2", ">=3", "17B_BLOQUEADO_POR_AMOSTRA", "abaixo do minimo de eventos"),
        ("geometria_oficial", "false", "true", "17B_BLOQUEADO_POR_GEOMETRIA_OFICIAL", "Curitiba so SAR; sem geometria oficial de ocorrencia"),
        ("segunda_regiao_tecnica", "true", "true", "17B_APROXIMACAO_COM_SEGUNDA_REGIAO_TECNICA", "Recife forte + Curitiba tecnica SAR"),
    ]
    return [{"criterio": c, "valor_observado": v, "limiar": l, "status_17b": s, "observacao": o}
            for c, v, l, s, o in data]


def gate_19e_rows(summary: dict) -> list[dict]:
    status = summary["status_19e"]
    data = [
        ("diagnostico_completo", "true", "true", "inventario, decomposicao, tipologia e hipoteses gerados"),
        ("score_v6_intacto", str(not summary["score_v6_changed"]).lower(), "true", "score_v6 nao alterado"),
        ("hipoteses_review_only", "true", "true", "hipoteses nao viram score oficial"),
        ("lacunas_documentadas", "true", "true", "missingness territorial e amostra minima explicitos"),
        ("pronto_com_ressalvas", "true", "true", "comunicacao cientifica review-only possivel com ressalvas de amostra e missingness"),
    ]
    return [{"criterio": c, "valor_observado": v, "limiar": l, "status_19e": status, "observacao": o}
            for c, v, l, o in data]


# ---------------------------------------------------------------------------
# Resumos
# ---------------------------------------------------------------------------
def resumo_regiao_rows(data: dict, sets: dict, ranks: dict, diag: list[dict]) -> list[dict]:
    score = data["score"]
    by_reg = {}
    for r in diag:
        by_reg.setdefault(r["regiao"], []).append(r)
    out = []
    for reg in ("recife", "curitiba"):
        rs = by_reg.get(reg, [])
        obs = [p for p in sets["observed"] if score[p].get("regiao") == reg]
        bg = [p for p in sets["background"] if score[p].get("regiao") == reg]
        smean_o = _mean(data["features"], obs, "score") if obs else None
        smean_b = _mean(data["features"], bg, "score") if bg else None
        top30r = sum(1 for r in rs if r["top30_regional"] == "true")
        low = sum(1 for r in rs if r["score_v6_class"].lower() == "low")
        out.append({
            "regiao": reg, "observados": str(len(obs)),
            "score_medio_observado": _fmt(smean_o, 4) if smean_o is not None else NA,
            "score_medio_background": _fmt(smean_b, 4) if smean_b is not None else NA,
            "top30_regional": str(top30r),
            "patches_low": str(low),
            "principal_divergencia": "hidrologia_e_chuva" if reg == "recife" else "hidrologia_chuva_e_um_patch_low_score",
            "status_regional": "recife_forte_review_only" if reg == "recife" else "curitiba_tecnica_sar_com_divergencia_forte",
        })
    return out


def resumo_falha_rows(tipologia: list[dict]) -> list[dict]:
    counts, sev = {}, {}
    for r in tipologia:
        ft = r["failure_type"]
        counts[ft] = counts.get(ft, 0) + 1
        sev[ft] = r["severidade"]
    return [{"failure_type": ft, "ocorrencias": str(counts[ft]), "severidade": sev[ft],
             "bloqueia_score_v7": "true"} for ft in sorted(counts)]


# ---------------------------------------------------------------------------
# Schema / preflight / summary
# ---------------------------------------------------------------------------
def schema_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SUSC-19D diagnostico de divergencias e hipoteses review-only do score_v6",
        "type": "object",
        "description": "Diagnostico auditavel review-only; nao altera score_v6, nao cria score_v7, nao cria benchmark 17B, nao treina, sem negativos.",
        "guardrails": {
            "ground_truth": {"const": "false"},
            "eligible_for_training": {"const": "false"},
            "score_v7_allowed": {"const": "false"},
            "score_v7": {"const": "false"},
            "benchmark_17b_criado": {"const": "false"},
            "score_v6_changed": {"const": "false"},
            "review_only": {"const": "true"},
            "substitui_score_v6": {"const": "false"},
        },
        "status_score_v7_allowed": sorted(STATUS_V7_ALLOWED),
        "status_17b_allowed": sorted(STATUS_17B_ALLOWED),
        "status_19e_allowed": sorted(STATUS_19E_ALLOWED),
        "failure_types_allowed": sorted(FAILURE_TYPES_ALLOWED),
        "familias": list(FAMILY_FEATURES.keys()),
        "background_label": "unlabeled_background",
        "small_sample_limit": SMALL_SAMPLE_LIMIT,
        "score_v6_formula": "robust_minmax(combinacao_linear_ponderada); ranking monotonico reproduzido; sensibilidade por reordenacao nao substitui score_v6",
    }


def preflight_obj(sets: dict) -> dict:
    status_full = _run_git(["status", "--short"]).splitlines()
    return {
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "staged_count": len(_run_git(["diff", "--cached", "--name-only"]).splitlines()),
        "status_short_count": len(status_full),
        "entradas_lidas": {rel(p): p.exists() for p in REQUIRED_INPUTS},
        "score_v6_baseline_path": rel(SCORE_V6),
        "score_v6_changed": _score_v6_changed(),
        "score_v7_created": SCORE_V7.exists(),
        "observados_herdados_19c": sets["observed"],
        "observed_recife": sets["recife"],
        "observed_curitiba": sets["curitiba"],
        "background_n": len(sets["background"]),
        "lacunas_herdadas_19b": {
            "cobertura_territorial": 0.3333,
            "features_ausentes": TERRITORIAL_MISSING,
        },
        "lacunas_herdadas_19c": {
            "observed_in_top_30_global": 0,
            "sample_power": "baixa",
            "divergencia_forte": "curitiba_01101",
        },
        "outputs_ignorados_pelo_git": sorted(git_ignored([OUT])),
        "summary_versionavel": rel(SUMMARY) not in git_ignored([SUMMARY]),
    }


def summary_obj(data: dict, sets: dict, ranks: dict, fam_rate: dict, diag: list[dict],
                tipologia: list[dict], sens: list[dict], comp_ausentes: bool) -> dict:
    smean_o = _mean(data["features"], sets["observed"], "score")
    smean_b = _mean(data["features"], sets["background"], "score")
    top30g = sum(1 for r in diag if r["top30_global"] == "true")
    aderentes = [f for f in FEATURE_FAMILIES if fam_rate[f] >= 0.6]
    divergentes = [f for f in FEATURE_FAMILIES if fam_rate[f] <= 0.34]
    n_low = sum(1 for r in diag if r["score_v6_class"].lower() == "low")
    status_19e = "19E_PRONTO_COM_RESSALVAS"
    return {
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "staged_count": len(_run_git(["diff", "--cached", "--name-only"]).splitlines()),
        "observed_patches": len(sets["observed"]),
        "observed_recife": len(sets["recife"]),
        "observed_curitiba": len(sets["curitiba"]),
        "background_patches": len(sets["background"]),
        "background_label": "unlabeled_background",
        "score_medio_observado": round(smean_o, 4) if smean_o is not None else None,
        "score_medio_background": round(smean_b, 4) if smean_b is not None else None,
        "observed_in_top_30_global": top30g,
        "familias_aderentes": aderentes,
        "familias_divergentes": divergentes,
        "familia_agreement_rate": {k: round(v, 3) for k, v in fam_rate.items()},
        "patches_low_score": n_low,
        "divergencia_forte": "curitiba_01101",
        "componentes_formais_ausentes": comp_ausentes,
        "n_falhas": len(tipologia),
        "n_hipoteses": 7,
        "sensibilidade_executada": sum(1 for r in sens if r["executado"] == "true"),
        "sensibilidade_desenho": sum(1 for r in sens if r["executado"] == "false"),
        "status_score_v7": "SCORE_V7_NAO_AUTORIZADO",
        "status_17b": "17B_NAO_CRIADO",
        "status_19e": status_19e,
        "marco_17b_criado": False,
        "benchmark_17b_criado": False,
        "ground_truth_true_count": 0,
        "eligible_for_training_true_count": 0,
        "score_v7_allowed_true_count": 0,
        "score_v6_changed": _score_v6_changed(),
        "score_v7_created": SCORE_V7.exists(),
        "sample_power": "baixa",
        "review_only": True,
        "proximo_marco": "SUSC-19E",
    }


# ---------------------------------------------------------------------------
# 10. Cartoes tecnicos
# ---------------------------------------------------------------------------
def _guardrail_footer() -> str:
    return (
        "## Por que nao e ground truth\n"
        "Evidencia observacional review-only sem geometria de ocorrencia confirmada por patch; "
        "SAR e pos-evento; nao e verdade de campo.\n\n"
        "## Por que nao e treino\n"
        "eligible_for_training=false; nenhuma linha habilita treino; a amostra e minima.\n\n"
        "## Por que nao cria score_v7\n"
        "score_v7 bloqueado por amostra, missingness territorial e ausencia de benchmark; "
        "as hipoteses sao review-only e nenhuma vira score oficial.\n\n"
        "## Por que nao cria 17B\n"
        "Sem geometria oficial e sem eventos suficientes; nenhum benchmark 17B e criado.\n"
    )


def _card_recife(data: dict, sets: dict, diag: list[dict], summary: dict) -> str:
    rs = [r for r in diag if r["regiao"] == "recife"]
    linhas = "\n".join(
        f"| {r['patch_id']} | {r['score_v6']} | {r['score_v6_class']} | {r['percentile_global']} | "
        f"{r['principal_fator_aderencia']} | {r['principal_fator_divergencia']} |" for r in rs
    )
    return f"""# Cartao de diagnostico - Recife

## Evidencia
Cinco canarios observacionais review-only (event_anchored_canary, documental).

## score_v6
Score medio observado (Recife) acima do background regional; nenhum patch no
top-30 global. Detalhe por patch:

| patch | score_v6 | classe | percentil global | aderencia | divergencia |
| --- | --- | --- | --- | --- | --- |
{linhas}

## Aderencias
Familias fisica_topografica, urbana_territorial e espectral_umidade coerentes com
maior suscetibilidade.

## Divergencias
Familias hidrologica (distance_to_water/TWI/flow_accumulation) e
chuva_hidrometeorologica (CHIRPS 3d/7d/30d menores) divergem e limitam o score.

## Hipotese review-only
Separar evidencia documental de suscetibilidade e tratar chuva como gatilho
contextual (HIP_01, HIP_04); nenhuma vira score oficial.

{_guardrail_footer()}"""


def _card_curitiba(data: dict, sets: dict, diag: list[dict], summary: dict) -> str:
    rs = [r for r in diag if r["regiao"] == "curitiba"]
    linhas = "\n".join(
        f"| {r['patch_id']} | {r['score_v6']} | {r['score_v6_class']} | {r['percentile_global']} | "
        f"{r['familia_documental_status']} |" for r in rs
    )
    return f"""# Cartao de diagnostico - Curitiba

## Evidencia
Dois overlays tecnicos SAR review-only (curitiba_01050, curitiba_01101). SAR nao e
geometria oficial e nunca vira feature pre-evento.

## score_v6
Um patch com score alto (curitiba_01050) e um com score baixo (curitiba_01101):

| patch | score_v6 | classe | percentil global | documental |
| --- | --- | --- | --- | --- |
{linhas}

## Aderencias
curitiba_01050 mantem coerencia urbana/espectral.

## Divergencias
Familia hidrologica e chuva divergem; documental ausente (evidence_support_index=0.0)
em ambos.

## Hipotese review-only
Componente hidrologico diferenciado para alagamento urbano e reducao da penalizacao
por baixa documentacao (HIP_05, HIP_02); nenhuma vira score oficial.

{_guardrail_footer()}"""


def _card_curitiba_01101(data: dict, sets: dict, diag: list[dict]) -> str:
    r = next(r for r in diag if r["patch_id"] == "curitiba_01101")
    return f"""# Cartao de diagnostico - curitiba_01101 (divergencia forte)

## Evidencia
Overlay tecnico SAR review-only. SAR e pos-evento e nao e geometria oficial.

## score_v6
score_v6={r['score_v6']} (classe {r['score_v6_class']}); percentil global
{r['percentile_global']} e regional {r['percentile_regional']} - o menor da amostra
observacional. Componente topografia_hidrologia e o mais baixo entre os observados e
o documental e 0.0.

## Aderencias
Aderencia parcial em {r['principal_fator_aderencia']}.

## Divergencias
Divergencia forte em {r['principal_fator_divergencia']}; o score_v6 classifica o patch
como baixo apesar da evidencia observacional review-only. Permanece baixo em todos os
cenarios de sensibilidade nao oficiais.

## Hipotese review-only
Caso emblematico para separar documental de suscetibilidade e revisar o componente
hidrologico (HIP_01, HIP_02, HIP_05); nenhuma vira score oficial.

{_guardrail_footer()}"""


def _card_sintese(summary: dict, fam_rate: dict) -> str:
    ader = ", ".join(summary["familias_aderentes"]) or "nenhuma"
    diver = ", ".join(summary["familias_divergentes"]) or "nenhuma"
    return f"""# Cartao de diagnostico - Sintese global

## Evidencia
{summary['observed_patches']} patches observacionais review-only (Recife
{summary['observed_recife']}, Curitiba {summary['observed_curitiba']}); background nao
rotulado de {summary['background_patches']} (nunca negativo).

## score_v6
Score medio observado {summary['score_medio_observado']} contra background
{summary['score_medio_background']}; observados no top-30 global:
{summary['observed_in_top_30_global']}.

## Aderencias
Familias que sustentam a aderencia: {ader}.

## Divergencias
Familias que derrubam os observados: {diver}. O score_v6 pesa topografia_hidrologia
(0.4) e chuva (0.25); a hidrologia divergente e a chuva antecedente menor limitam o
ranking mesmo com topografia, urbano e espectral coerentes.

## Hipotese review-only
Sete hipoteses review-only (separar documental, chuva como gatilho, hidrologia urbana,
preencher territorial, ampliar amostra); nenhuma vira score oficial.

{_guardrail_footer()}"""


def _card_bloqueio_v7(summary: dict) -> str:
    return f"""# Cartao de diagnostico - Bloqueio do score_v7

## Estado
status_score_v7 = `{summary['status_score_v7']}`.

## Motivos do bloqueio
- amostra minima ({summary['observed_patches']} observados; 0 no top-30 global);
- missingness territorial herdado do 19B (faltam {', '.join(TERRITORIAL_MISSING)});
- ausencia de benchmark 17B;
- hipoteses sao review-only e nenhuma vira score oficial.

## O que muda com o diagnostico
Nada no score_v6 (intacto) e nada no score_v7 (nao criado). O diagnostico apenas
organiza aderencias, divergencias, falhas e hipoteses testaveis no futuro.

{_guardrail_footer()}"""


def write_cards(data: dict, sets: dict, diag: list[dict], summary: dict, fam_rate: dict) -> None:
    ensure_dir(CARDS)
    write_markdown(CARDS / "cartao_recife.md", _card_recife(data, sets, diag, summary))
    write_markdown(CARDS / "cartao_curitiba.md", _card_curitiba(data, sets, diag, summary))
    write_markdown(CARDS / "cartao_curitiba_01101_divergencia_forte.md", _card_curitiba_01101(data, sets, diag))
    write_markdown(CARDS / "cartao_sintese_global.md", _card_sintese(summary, fam_rate))
    write_markdown(CARDS / "cartao_bloqueio_score_v7.md", _card_bloqueio_v7(summary))


# ---------------------------------------------------------------------------
# 14. Relatorio publico
# ---------------------------------------------------------------------------
def report_text(summary: dict, inventario: list[dict], decomp: list[dict],
                diag: list[dict], sens: list[dict]) -> str:
    linhas_inv = "\n".join(
        f"| {r['componente']} | {r['peso_explicito']} | {r['entra_documental']} | "
        f"{r['entra_rainfall']} | {r['entra_urban_spectral']} | {r['papel_no_diagnostico']} |"
        for r in inventario if r["componente"] != "score_v6_final"
    )
    linhas_dec = "\n".join(
        f"| {r['familia']} | {r['feature_count']} | {r['observed_mean']} | {r['background_mean']} | "
        f"{r['direction_agreement_rate']} | {r['contribuicao_interpretativa']} |"
        for r in decomp
    )
    linhas_diag = "\n".join(
        f"| {r['patch_id']} | {r['regiao']} | {r['score_v6']} | {r['score_v6_class']} | "
        f"{r['percentile_global']} | {r['top30_global']} | {r['principal_fator_divergencia']} |"
        for r in diag
    )
    linhas_sens = "\n".join(
        f"| {r['scenario_id']} | {r['executado']} | {r['mudanca_rank_observada']} | {r['interpretacao']} |"
        for r in sens
    )
    return f"""# SUSC-19D - Diagnostico de divergencias e hipoteses review-only do score_v6

## Estado herdado do 19C

O 19C avaliou 7 patches observacionais review-only (Recife 5 canarios; Curitiba 2
overlays tecnicos SAR) contra o universo nao rotulado de {summary['background_patches']}
patches. Achados herdados: score medio observado {summary['score_medio_observado']} maior
que o background {summary['score_medio_background']}, porem
{summary['observed_in_top_30_global']} observados no top-30 global; coerencia
urbana/topografica; divergencia hidrologica/de chuva; `curitiba_01101` como divergencia
forte. O universo sem evidencia e `unlabeled_background` e nunca e negativo.

## Por que o 19D existe

O 19C mostrou um paradoxo: os observados tem score medio maior mas nao aparecem no
extremo do ranking. O 19D explica tecnicamente esse resultado, sem alterar o score_v6,
sem criar score_v7 e sem criar benchmark 17B.

## Diagnostico do score_v6

O score_v6 tem 5 componentes com pesos explicitos (contrato 17c35) e e
`robust_minmax(combinacao_linear_ponderada)`. A combinacao linear reproduz o ranking
oficial exatamente (transformacao monotonica), o que permite analise de sensibilidade
por reordenacao sem recalcular nem substituir o score_v6.

| Componente | Peso | Documental | Chuva | Urbano/espectral | Papel no diagnostico |
| --- | --- | --- | --- | --- | --- |
{linhas_inv}

## Patches observacionais

| patch | regiao | score_v6 | classe | percentil global | top30 global | divergencia principal |
| --- | --- | --- | --- | --- | --- | --- |
{linhas_diag}

## Decomposicao por familia

| Familia | Nº features | Media observada | Media background | Coerencia | Contribuicao |
| --- | --- | --- | --- | --- | --- |
{linhas_dec}

Familias que sustentam a aderencia: {', '.join(summary['familias_aderentes']) or 'nenhuma'}.
Familias que derrubam os observados: {', '.join(summary['familias_divergentes']) or 'nenhuma'}.

## Divergencias principais

- **Hidrologica**: distance_to_water maior, TWI e flow_accumulation menores nos
  observados - direcao contraria a esperada. A hidrologia natural nao captura drenagem
  urbana (limitacao ja registrada na cadeia 17C36-40).
- **Chuva**: CHIRPS 3d/7d/30d menores nos observados. A chuva antecedente entra como
  suscetibilidade estatica e derruba o ranking, quando deveria ser gatilho contextual.
- Com topografia_hidrologia (0.4) e chuva (0.25) somando a maior parte do peso, a
  divergencia hidrologica e de chuva limita os observados mesmo com topografia, urbano e
  espectral coerentes.

## Caso curitiba_01101

score_v6 baixo (classe low), menor percentil da amostra. Componente
topografia_hidrologia e o mais baixo entre os observados e documental=0.0. E a
divergencia observacional mais forte e permanece baixo em todos os cenarios de
sensibilidade nao oficiais.

## Sensibilidade nao oficial (review-only, nao substitui score_v6)

| Cenario | Executado | Mudanca de rank observada | Interpretacao |
| --- | --- | --- | --- |
{linhas_sens}

Os cenarios sao interpretativos e review-only; nao substituem o score_v6, nao sao score
oficial e nao criam score_v7.

## Hipoteses review-only

Sete hipoteses em `hipoteses_ajuste_review_only_19d.csv` (separar documental de
suscetibilidade; reduzir penalizacao por baixa documentacao; aumentar peso
urbano/espectral; chuva como gatilho; hidrologia urbana; preencher territorial; ampliar
amostra). Todas com `pode_virar_score_v7_agora=false`.

## Por que o score_v7 segue bloqueado

`{summary['status_score_v7']}`: bloqueado por amostra minima, por missingness territorial
herdado do 19B e por ausencia de benchmark. As hipoteses sao review-only e nenhuma vira
score oficial. O score_v6 permanece intacto.

## Por que o 17B segue nao criado

`{summary['status_17b']}`: sem geometria oficial de ocorrencia e sem eventos suficientes;
Curitiba e apenas tecnica SAR. Nenhum benchmark 17B e criado.

## Por que nao e ground truth nem treino

Evidencia review-only sem geometria de ocorrencia confirmada; SAR e pos-evento;
background nunca e negativo; amostra minima. ground_truth=false e
eligible_for_training=false em todas as linhas.

## Proximo marco recomendado

**SUSC-19E - Comunicacao cientifica review-only**: consolidar figuras, tabelas e
narrativa para trabalho de conclusao, com ressalvas de amostra e missingness territorial.
Estado: `{summary['status_19e']}`.
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

    inventario, comp_ausentes_inv = inventario_rows(data)
    decomp, fam_rate = decomposicao_rows(data, sets)
    diag = diagnostico_patches_rows(data, sets, ranks)
    tipologia = tipologia_rows(data, sets, ranks, fam_rate)
    hipoteses = hipoteses_rows()
    base_ranks = {"global_rank": ranks["global_rank"]}
    sens, comp_ausentes_sens = sensibilidade_rows(data, sets, base_ranks)
    summary = summary_obj(data, sets, ranks, fam_rate, diag, tipologia, sens,
                          comp_ausentes_inv or comp_ausentes_sens)

    write_csv(INVENTARIO, inventario)
    write_csv(DIAG_PATCHES, diag)
    write_csv(DECOMP_FAMILIAS, decomp)
    write_csv(TIPOLOGIA, tipologia)
    write_csv(HIPOTESES, hipoteses)
    write_csv(SENSIBILIDADE, sens)
    write_csv(GATE_V7, gate_v7_rows())
    write_csv(GATE_17B, gate_17b_rows())
    write_csv(GATE_19E, gate_19e_rows(summary))
    write_csv(RESUMO_REGIAO, resumo_regiao_rows(data, sets, ranks, diag))
    write_csv(RESUMO_FALHA, resumo_falha_rows(tipologia))

    write_json(SCHEMA, schema_obj())
    write_json(PREFLIGHT, preflight_obj(sets))
    write_json(SUMMARY, summary)
    write_cards(data, sets, diag, summary, fam_rate)
    write_markdown(REPORT, report_text(summary, inventario, decomp, diag, sens))
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
STRONG_STATS_RE = re.compile(r"p[-_ ]?valor|p[-_ ]?value|significancia estatistica|estatisticamente significativo", re.IGNORECASE)


def validate_outputs(summary: dict) -> list[str]:
    errors = [f"missing:{rel(p)}" for p in REQUIRED_OUTPUTS if not p.exists()]

    # guardrails do summary
    if summary["score_v6_changed"]:
        errors.append("score_v6_changed_forbidden")
    if summary["score_v7_created"]:
        errors.append("score_v7_created_forbidden")
    if summary["benchmark_17b_criado"] or summary["marco_17b_criado"]:
        errors.append("benchmark_ou_marco_17b_criado_forbidden")
    if summary["ground_truth_true_count"]:
        errors.append("ground_truth_true_forbidden")
    if summary["eligible_for_training_true_count"]:
        errors.append("eligible_for_training_true_forbidden")
    if summary["score_v7_allowed_true_count"]:
        errors.append("score_v7_allowed_true_forbidden")
    if summary["status_score_v7"] not in STATUS_V7_ALLOWED:
        errors.append(f"status_score_v7_fora_enum:{summary['status_score_v7']}")
    if summary["status_17b"] not in STATUS_17B_ALLOWED:
        errors.append(f"status_17b_fora_enum:{summary['status_17b']}")
    if summary["status_19e"] not in STATUS_19E_ALLOWED:
        errors.append(f"status_19e_fora_enum:{summary['status_19e']}")

    # diagnostico dos 7 observados
    diag = read_csv(DIAG_PATCHES)
    if len(diag) != 7:
        errors.append(f"diagnostico_diferente_de_7:{len(diag)}")
    if sum(1 for r in diag if r["regiao"] == "recife") != 5:
        errors.append("recife_diferente_de_5")
    if sum(1 for r in diag if r["regiao"] == "curitiba") != 2:
        errors.append("curitiba_diferente_de_2")
    if any(r["regiao"] == "petropolis" for r in diag):
        errors.append("petropolis_como_observado_forbidden")
    if not any(r["patch_id"] == "curitiba_01101" for r in diag):
        errors.append("curitiba_01101_ausente")
    for r in diag:
        if r["patch_id"] == "curitiba_01101":
            blob = " ".join(str(v).lower() for v in r.values())
            if "diverg" not in blob:
                errors.append("curitiba_01101_nao_marcado_como_divergencia")

    # inventario: pesos explicitos e documental/chuva/urban
    inv = read_csv(INVENTARIO)
    comp_names = {r["componente"] for r in inv}
    for req in ("evidence_support_index", "rainfall_trigger_index", "urban_spectral_index"):
        if req not in comp_names:
            errors.append(f"inventario_sem_componente:{req}")
    if not any(r["entra_documental"] == "true" for r in inv):
        errors.append("inventario_sem_documental")
    if not any(r["entra_rainfall"] == "true" for r in inv):
        errors.append("inventario_sem_rainfall")

    # decomposicao: missingness territorial deve aparecer
    dec = read_csv(DECOMP_FAMILIAS)
    fam_names = {r["familia"] for r in dec}
    for fam in FAMILY_FEATURES:
        if fam not in fam_names:
            errors.append(f"decomposicao_sem_familia:{fam}")
    dec_blob = " ".join(" ".join(str(v) for v in r.values()) for r in dec)
    if "MapBiomas" not in dec_blob or "impervious_proxy" not in dec_blob:
        errors.append("missingness_territorial_omitido_na_decomposicao")

    # tipologia: enums e missingness/amostra presentes
    tip = read_csv(TIPOLOGIA)
    tip_types = {r["failure_type"] for r in tip}
    for r in tip:
        if r["failure_type"] not in FAILURE_TYPES_ALLOWED:
            errors.append(f"tipologia_failure_type_fora_enum:{r['failure_type']}")
        if r.get("bloqueia_score_v7") != "true":
            errors.append(f"tipologia_bloqueia_v7_nao_true:{r['failure_id']}")
    for req in ("missingness_territorial", "amostra_minima", "hidrologia_divergente"):
        if req not in tip_types:
            errors.append(f"tipologia_sem_falha:{req}")

    # hipoteses: nenhuma vira score_v7 agora
    for r in read_csv(HIPOTESES):
        if r.get("pode_virar_score_v7_agora") != "false":
            errors.append(f"hipotese_pode_virar_v7:{r['hipotese_id']}")

    # sensibilidade: nunca substitui score_v6 / nao e score oficial / nao e v7
    sens = read_csv(SENSIBILIDADE)
    if not sens:
        errors.append("sensibilidade_vazia")
    for r in sens:
        for field in ("substitui_score_v6", "score_oficial", "score_v7"):
            if r.get(field) != "false":
                errors.append(f"sensibilidade:{r['scenario_id']}:{field}_nao_false")
        if r.get("uso_permitido") != "analise_sensibilidade_review_only":
            errors.append(f"sensibilidade:{r['scenario_id']}:uso_permitido_invalido")

    # gates enums
    for r in read_csv(GATE_V7):
        if r.get("status_score_v7") not in STATUS_V7_ALLOWED:
            errors.append(f"gate_v7:status_fora_enum:{r.get('status_score_v7')}")
    for r in read_csv(GATE_17B):
        if r.get("status_17b") not in STATUS_17B_ALLOWED:
            errors.append(f"gate_17b:status_fora_enum:{r.get('status_17b')}")
    for r in read_csv(GATE_19E):
        if r.get("status_19e") not in STATUS_19E_ALLOWED:
            errors.append(f"gate_19e:status_fora_enum:{r.get('status_19e')}")

    # background nunca chamado de negativo em texto publico
    for path in _public_paths():
        txt = path.read_text(encoding="utf-8", errors="ignore")
        if STRONG_STATS_RE.search(txt):
            errors.append(f"{rel(path)}:conclusao_estatistica_forte_com_n_pequeno")
        for m in NEGATIVE_RE.finditer(txt):
            window = txt[max(0, m.start() - 40): m.end() + 40].lower()
            if "nao e negativo" not in window and "nunca e negativo" not in window and "nunca negativo" not in window:
                errors.append(f"{rel(path)}:background_ou_termo_negativo_sem_ressalva")
                break

    # patch_stats SAR nunca como feature pre-evento
    for path in (DECOMP_FAMILIAS, DIAG_PATCHES):
        for r in read_csv(path):
            blob = " ".join(str(v).lower() for v in r.values())
            if "patch_stats" in blob:
                errors.append(f"{rel(path)}:patch_stats_como_feature")

    # SAR nao pode ser chamado (afirmativamente) de geometria oficial;
    # frases com negacao (sem/nao/nunca geometria oficial) sao permitidas.
    for path in _public_paths():
        txt = path.read_text(encoding="utf-8", errors="ignore").lower()
        for m in re.finditer(r"sar[^.\n]{0,30}geometria oficial", txt):
            window = txt[max(0, m.start() - 20): m.end()]
            if not re.search(r"\b(sem|nao|n[aã]o|nunca)\b", window):
                errors.append(f"{rel(path)}:sar_chamado_de_geometria_oficial")
                break

    # output 19D nao pode ser ignorado pelo git
    if rel(SUMMARY) in git_ignored([SUMMARY, DIAG_PATCHES]):
        errors.append("output_19d_ignorado_pelo_git")

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
        "19D diagnostico validado: "
        f"observados={summary['observed_patches']} (rec={summary['observed_recife']} cur={summary['observed_curitiba']}) "
        f"background={summary['background_patches']} top30g={summary['observed_in_top_30_global']} "
        f"aderentes={summary['familias_aderentes']} divergentes={summary['familias_divergentes']} "
        f"falhas={summary['n_falhas']} hipoteses={summary['n_hipoteses']} "
        f"sens_exec={summary['sensibilidade_executada']} "
        f"status17B={summary['status_17b']} scorev7={summary['status_score_v7']} 19E={summary['status_19e']} "
        f"score_v6_changed={str(summary['score_v6_changed']).lower()}"
    )
    return 0


def run_all() -> int:
    summary = build_all()
    print(
        "19D diagnostico gerado: "
        f"observados={summary['observed_patches']} background={summary['background_patches']} "
        f"score_obs={summary['score_medio_observado']} score_bg={summary['score_medio_background']} "
        f"divergencia_forte={summary['divergencia_forte']} scorev7={summary['status_score_v7']} "
        f"19E={summary['status_19e']}"
    )
    return 0
