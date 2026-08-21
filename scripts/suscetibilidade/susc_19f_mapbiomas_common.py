"""SUSC-19F execucao controlada MapBiomas/GEE e ingestao territorial.

Executa de fato o pacote MapBiomas/GEE preparado no SUSC-19B, recupera o CSV leve
de proporcoes territoriais por patch e atualiza a matriz multimodal com features
territoriais reais para os 300 patches. Se o Earth Engine nao estiver autenticado,
registra o bloqueio tecnico exato e mantem tudo reexecutavel; nunca inventa valores.

Regras cientificas: uso/cobertura do solo e feature territorial escalavel, nao
evidencia de evento. MapBiomas nao vira ground truth. SAR pos-evento nunca entra
como feature territorial. coverage_score e completude, nao suscetibilidade.
background/unlabeled nunca e negativo. score_v6 permanece baseline oficial e intacto.

Guardrails: ground_truth=false, eligible_for_training=false, score_v7_allowed=false,
score_oficial=false, substituir_score_v6=false, usar_em_treino=false, sem benchmark 17B.

Este modulo concentra caminhos, mapeamento de classes, ingestao, cobertura, build e
validacao. A execucao GEE fica em run_susc_19f_mapbiomas_gee.py (escreve local_runs);
a atualizacao de tasks fica em check_susc_19f_mapbiomas_tasks.py. O build le o CSV
local (se existir) e produz os outputs publicos; nunca escreve raster nem credenciais.
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
OUT = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_19f_execucao_mapbiomas_gee_ingestao_territorial"
CARDS = OUT / "cartoes_regionais"
REPORTS = ROOT / "outputs_public" / "reports"
SCHEMAS_DIR = ROOT / "schemas" / "suscetibilidade"

REPORT = REPORTS / "SUSC_19F_EXECUCAO_MAPBIOMAS_GEE_INGESTAO_TERRITORIAL.md"
SCHEMA = SCHEMAS_DIR / "susc_19f_mapbiomas_schema_v1.json"

SCORE_V6 = ROOT / "datasets" / "suscetibilidade" / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = ROOT / "datasets" / "suscetibilidade" / "susc_score_v7_candidate_by_patch_v1.csv"

DATA = ROOT / "outputs_public" / "data" / "linhagem_anterior"
PKG_19B = DATA / "susc_19b_auditoria_lacunas_territoriais" / "pacote_gee"
MANIFEST_19B = PKG_19B / "gee_export_manifest_19b.csv"
MAT_19A = DATA / "susc_19a_matriz_multimodal_escalavel_por_patch" / "matriz_multimodal_escalavel_por_patch.csv"
MAT_19B = DATA / "susc_19b_auditoria_lacunas_territoriais" / "matriz_multimodal_19b_atualizada.csv"
SUM_19A = DATA / "susc_19a_matriz_multimodal_escalavel_por_patch" / "summary.json"
SUM_19B = DATA / "susc_19b_auditoria_lacunas_territoriais" / "summary.json"

# Saida local (git-ignored) da execucao GEE
LOCAL_RUN = ROOT / "local_runs" / "suscetibilidade" / "19f_mapbiomas_territorial" / "resultados_gee"
LOCAL_CSV = LOCAL_RUN / "mapbiomas_patch_landcover_2022.csv"
LOCAL_META = LOCAL_RUN / "mapbiomas_execution_metadata.json"
LOCAL_TASKS = LOCAL_RUN / "mapbiomas_tasks.csv"
LOCAL_LOG = LOCAL_RUN / "execution_log.txt"

# Saidas publicas
AUD_AMBIENTE = OUT / "auditoria_ambiente_gee_19f.csv"
STATUS_TASKS = OUT / "status_tasks_mapbiomas_19f.csv"
RECUPERACAO = OUT / "recuperacao_export_mapbiomas_19f.csv"
MAT_TERRITORIAL = OUT / "matriz_territorial_mapbiomas_19f.csv"
MAT_MULTIMODAL = OUT / "matriz_multimodal_19f_atualizada.csv"
COMPARACAO = OUT / "comparacao_cobertura_19a_19b_19f.csv"
AUD_COBERTURA = OUT / "auditoria_cobertura_pos_19f.csv"
GATE_TERRITORIAL = OUT / "gate_territorial_19f.csv"
GATE_V7 = OUT / "gate_prontidao_score_v7_pos_19f.csv"
GATE_17B = OUT / "gate_prontidao_17b_pos_19f.csv"
GATE_19G = OUT / "gate_prontidao_19g_pos_19f.csv"
RESUMO_REGIAO = OUT / "resumo_por_regiao.csv"
RESUMO_FEATURE = OUT / "resumo_por_feature.csv"
SUMMARY = OUT / "summary.json"
PREFLIGHT = OUT / "preflight.json"

REQUIRED_INPUTS = [MANIFEST_19B, MAT_19A, SUM_19A, SUM_19B]
REQUIRED_OUTPUTS = [
    AUD_AMBIENTE, STATUS_TASKS, RECUPERACAO, MAT_TERRITORIAL, MAT_MULTIMODAL,
    COMPARACAO, AUD_COBERTURA, GATE_TERRITORIAL, GATE_V7, GATE_17B, GATE_19G,
    RESUMO_REGIAO, RESUMO_FEATURE, SUMMARY, PREFLIGHT, SCHEMA, REPORT,
]

# ---------------------------------------------------------------------------
# MapBiomas Colecao 9 - parametros herdados do pacote 19B
# ---------------------------------------------------------------------------
MAPBIOMAS_COLLECTION = "projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_integration_v1"
MAPBIOMAS_YEAR = 2022
MAPBIOMAS_SCALE = 30
LANDCOVER_SOURCE = "mapbiomas_collection9_gee"

# Classes agregadas (identicas ao pacote 19B)
CLASSES_AGUA = [33, 31]
CLASSES_SOLO_EXPOSTO = [23, 25]
CLASSES_URBANO = [24]
CLASSES_VEGETACAO = [1, 3, 4, 5, 6, 49, 10, 11, 12]

# Rotulos legiveis das classes mais comuns (apoio a cartoes; CSV guarda o codigo)
MAPBIOMAS_LABELS = {
    3: "formacao_florestal", 4: "formacao_savanica", 5: "mangue", 6: "floresta_alagavel",
    11: "campo_alagado", 12: "formacao_campestre", 15: "pastagem", 21: "mosaico_agropecuaria",
    23: "praia_duna", 24: "area_urbanizada", 25: "area_nao_vegetada", 31: "aquicultura",
    33: "rio_lago_oceano", 49: "restinga_arborizada",
}

TERRITORIAL_TARGET_FEATURES = [
    "mapbiomas_class_majority", "mapbiomas_class_distribution",
    "water_prop", "exposed_soil_prop", "impervious_proxy",
    "landcover_source", "landcover_reference_year",
]

# ---------------------------------------------------------------------------
# Guardrails / constantes
# ---------------------------------------------------------------------------
PUBLIC_FORBIDDEN_RE = re.compile(r"\b(?:agentic|agente|codex|llm|ia)\b", re.IGNORECASE)
# Padroes de credencial/token que nunca podem aparecer em output publico.
SECRET_RE = re.compile(r"(refresh_token|client_secret|private_key|-----BEGIN|access_token|Bearer\s+[A-Za-z0-9._-]{12,})", re.IGNORECASE)
NA = "NA"
EXPECTED_PATCHES = 300

STATUS_19F_ALLOWED = {
    "19F_TERRITORIAL_PREENCHIDO_MAPBIOMAS_GEE",
    "19F_TAREFAS_GEE_INICIADAS_AGUARDANDO_CONCLUSAO",
    "19F_EXPORT_CONCLUIDO_NAO_RECUPERADO",
    "19F_BLOQUEADO_POR_AUTENTICACAO",
    "19F_BLOQUEADO_POR_PERMISSAO",
    "19F_BLOQUEADO_FAIL_CLOSED",
}
STATUS_V7_ALLOWED = {
    "SCORE_V7_NAO_AUTORIZADO",
    "SCORE_V7_BLOQUEADO_POR_AMOSTRA",
    "SCORE_V7_BLOQUEADO_POR_AUSENCIA_BENCHMARK",
    "SCORE_V7_BLOQUEADO_POR_VALIDACAO_INSUFICIENTE",
}
STATUS_17B_ALLOWED = {
    "17B_NAO_CRIADO",
    "17B_BLOQUEADO_POR_AMOSTRA",
    "17B_BLOQUEADO_POR_GEOMETRIA_OFICIAL",
    "17B_APROXIMACAO_COM_SEGUNDA_REGIAO_TECNICA",
}
STATUS_19G_ALLOWED = {
    "19G_PRONTO_APOS_TERRITORIAL",
    "19G_PRONTO_COM_RESSALVAS",
    "19G_BLOQUEADO_POR_TERRITORIAL_INCOMPLETO",
}
ENV_STATUS_ALLOWED = {
    "gee_autenticado_pronto", "gee_nao_autenticado", "biblioteca_ausente",
    "permissao_insuficiente", "ambiente_indeterminado",
}
CIDADE = {"recife": "Recife", "curitiba": "Curitiba", "petropolis": "Petropolis"}
# Familias 19A e numero de features esperadas (para coverage de completude)
FAMILIA_TERRITORIAL_FEATURES = 6  # urban_prop, vegetation_prop + 4 MapBiomas


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


def load_manifest() -> list[dict]:
    return read_csv(MANIFEST_19B)


# ---------------------------------------------------------------------------
# Classe majoritaria / proporcoes a partir do histograma de pixels
# ---------------------------------------------------------------------------
def proportions_from_hist(hist: dict) -> dict:
    """Recebe {classe_str: contagem_pixels} e devolve proporcoes agregadas."""
    counts = {}
    total = 0.0
    for k, v in hist.items():
        try:
            c = int(float(k))
            n = float(v)
        except (TypeError, ValueError):
            continue
        counts[c] = counts.get(c, 0.0) + n
        total += n
    if total <= 0:
        return {}

    def prop(classes):
        return sum(counts.get(c, 0.0) for c in classes) / total

    majority = max(counts, key=lambda c: counts[c]) if counts else None
    dist = {str(c): round(counts[c] / total, 6) for c in sorted(counts, key=lambda c: -counts[c])}
    return {
        "mapbiomas_class_majority": majority,
        "mapbiomas_class_majority_label": MAPBIOMAS_LABELS.get(majority, f"classe_{majority}"),
        "mapbiomas_class_distribution": dist,
        "water_prop": round(prop(CLASSES_AGUA), 6),
        "exposed_soil_prop": round(prop(CLASSES_SOLO_EXPOSTO), 6),
        "impervious_proxy": round(prop(CLASSES_URBANO), 6),
        "vegetation_prop_mapbiomas": round(prop(CLASSES_VEGETACAO), 6),
        "pixel_count": round(total, 3),
    }


# ---------------------------------------------------------------------------
# 2. Auditoria de ambiente GEE (le metadata da execucao; nao roda GEE aqui)
# ---------------------------------------------------------------------------
def env_audit_rows() -> tuple[list[dict], str]:
    meta = read_json(LOCAL_META) if LOCAL_META.exists() else {}
    env = meta.get("environment", {}) if isinstance(meta, dict) else {}
    status = meta.get("env_status", "ambiente_indeterminado") if isinstance(meta, dict) else "ambiente_indeterminado"
    if status not in ENV_STATUS_ALLOWED:
        status = "ambiente_indeterminado"
    checks = [
        ("python_version", env.get("python_version", NA), "python 3.x disponivel"),
        ("ee_library", env.get("ee_library", NA), "biblioteca earthengine-api"),
        ("earthengine_cli", env.get("earthengine_cli", NA), "CLI earthengine"),
        ("ee_initialize", env.get("ee_initialize", NA), "ee.Initialize() sem credencial impressa"),
        ("internet", env.get("internet", NA), "conectividade com o Earth Engine"),
        ("export_permission", env.get("export_permission", NA), "permissao de leitura/compute (getInfo)"),
        ("writable_folder", env.get("writable_folder", NA), "pasta local gravavel para CSV leve"),
    ]
    rows = [{
        "verificacao": c, "resultado": str(v), "criterio": crit,
        "env_status": status,
        "observacao": "sem impressao de credenciais; apenas indicadores booleanos",
    } for c, v, crit in checks]
    return rows, status


# ---------------------------------------------------------------------------
# 4/5. Tasks e recuperacao do export
# ---------------------------------------------------------------------------
def status_tasks_rows() -> list[dict]:
    if LOCAL_TASKS.exists():
        rows = read_csv(LOCAL_TASKS)
        if rows:
            return rows
    return [{
        "task_id": NA, "task_type": "synchronous_getInfo", "status": "nao_iniciado",
        "patches_processados": "0", "observacao": "nenhuma task GEE registrada; execucao nao concluida",
    }]


def recuperacao_rows(csv_exists: bool) -> list[dict]:
    data = [
        ("A_arquivo_local", "true" if csv_exists else "false",
         rel(LOCAL_CSV) if csv_exists else NA,
         "CSV leve recuperado localmente pela API" if csv_exists else "arquivo local ausente"),
        ("B_download_ee_api", "true" if csv_exists else "false",
         "ee.getInfo() paginado" if csv_exists else NA,
         "download via API do Earth Engine (getInfo), sem Drive" if csv_exists
         else "nao recuperado por API"),
        ("C_task_export_concluida", "false", NA,
         "nao utilizada; recuperacao por API dispensa export assincrono"),
        ("D_drive_api_local", "false", NA, "nao utilizada"),
        ("E_bloqueio_com_instrucao", "false" if csv_exists else "true", NA,
         "recuperado" if csv_exists else
         "reexecutar run_susc_19f_mapbiomas_gee.py apos autenticar (earthengine authenticate); "
         "ou rodar o .js do 19B no Code Editor e baixar o CSV do Drive para " + rel(LOCAL_CSV)),
    ]
    return [{
        "rota": r, "utilizada_ou_disponivel": u, "referencia": ref, "instrucao_ou_observacao": obs,
    } for r, u, ref, obs in data]


# ---------------------------------------------------------------------------
# 6. Ingestao territorial
# ---------------------------------------------------------------------------
def load_local_territorial() -> dict:
    """Le o CSV local do MapBiomas (se existir). Retorna {patch_id: row}."""
    if not LOCAL_CSV.exists():
        return {}
    out = {}
    for r in read_csv(LOCAL_CSV):
        pid = r.get("patch_id", "").strip()
        if pid:
            out[pid] = r
    return out


def matriz_territorial_rows(manifest: list[dict], terr: dict) -> tuple[list[dict], dict]:
    """Se houver CSV real, monta a matriz territorial; senao, cabecalho vazio."""
    fieldnames = [
        "patch_id", "regiao", "mapbiomas_class_majority", "mapbiomas_class_majority_label",
        "mapbiomas_class_distribution", "water_prop", "exposed_soil_prop", "impervious_proxy",
        "vegetation_prop_mapbiomas", "pixel_count", "landcover_source", "landcover_reference_year",
        "territorial_status", "fonte_real", "ground_truth", "eligible_for_training",
        "score_v7_allowed", "review_only", "observacao",
    ]
    rows = []
    filled = 0
    for m in manifest:
        pid = m["patch_id"]
        reg = m.get("regiao", NA)
        t = terr.get(pid)
        if t and _num(t.get("water_prop")) is not None:
            filled += 1
            rows.append({
                "patch_id": pid, "regiao": reg,
                "mapbiomas_class_majority": t.get("mapbiomas_class_majority", NA),
                "mapbiomas_class_majority_label": t.get("mapbiomas_class_majority_label", NA),
                "mapbiomas_class_distribution": t.get("mapbiomas_class_distribution", NA),
                "water_prop": t.get("water_prop", NA),
                "exposed_soil_prop": t.get("exposed_soil_prop", NA),
                "impervious_proxy": t.get("impervious_proxy", NA),
                "vegetation_prop_mapbiomas": t.get("vegetation_prop_mapbiomas", NA),
                "pixel_count": t.get("pixel_count", NA),
                "landcover_source": LANDCOVER_SOURCE,
                "landcover_reference_year": str(MAPBIOMAS_YEAR),
                "territorial_status": "preenchido_mapbiomas_real",
                "fonte_real": "true",
                "ground_truth": "false", "eligible_for_training": "false",
                "score_v7_allowed": "false", "review_only": "true",
                "observacao": "uso/cobertura do solo review-only; nao e ground truth e nao habilita treino",
            })
        else:
            rows.append({
                "patch_id": pid, "regiao": reg,
                "mapbiomas_class_majority": NA, "mapbiomas_class_majority_label": NA,
                "mapbiomas_class_distribution": NA, "water_prop": NA, "exposed_soil_prop": NA,
                "impervious_proxy": NA, "vegetation_prop_mapbiomas": NA, "pixel_count": NA,
                "landcover_source": NA, "landcover_reference_year": NA,
                "territorial_status": "aguardando_export",
                "fonte_real": "false",
                "ground_truth": "false", "eligible_for_training": "false",
                "score_v7_allowed": "false", "review_only": "true",
                "observacao": "sem CSV real do MapBiomas; feature nao preenchida (nao inventada)",
            })
    meta = {"filled": filled, "fieldnames": fieldnames}
    return rows, meta


# ---------------------------------------------------------------------------
# 7. Matriz multimodal atualizada
# ---------------------------------------------------------------------------
def _cobertura_familia_presentes(row_terr: dict) -> int:
    """Territorial presente = urban_prop + vegetation_prop (sempre) + 4 MapBiomas se reais."""
    base = 2  # urban_prop e vegetation_prop vem do 19A
    if row_terr and row_terr.get("fonte_real") == "true":
        base += 4
    return base


def matriz_multimodal_rows(terr_rows: list[dict], ingested: bool) -> tuple[list[dict], dict]:
    base = read_csv(MAT_19B) if MAT_19B.exists() else read_csv(MAT_19A)
    base_by_id = {r["patch_id"]: r for r in base}
    terr_by_id = {r["patch_id"]: r for r in terr_rows}
    out = []
    cov_terr_vals = []
    for pid, brow in base_by_id.items():
        row = dict(brow)
        trow = terr_by_id.get(pid, {})
        presentes = _cobertura_familia_presentes(trow)
        cov_terr = presentes / FAMILIA_TERRITORIAL_FEATURES
        cov_terr_vals.append(cov_terr)
        if ingested and trow.get("fonte_real") == "true":
            row["mapbiomas_class_majority"] = trow.get("mapbiomas_class_majority", NA)
            row["water_prop"] = trow.get("water_prop", NA)
            row["exposed_soil_prop"] = trow.get("exposed_soil_prop", NA)
            row["impervious_proxy"] = trow.get("impervious_proxy", NA)
            row["landcover_source"] = LANDCOVER_SOURCE
            row["landcover_reference_year"] = str(MAPBIOMAS_YEAR)
            row["territorial_fonte_real"] = "true"
        else:
            row.setdefault("mapbiomas_class_majority", NA)
            row.setdefault("water_prop", NA)
            row.setdefault("exposed_soil_prop", NA)
            row.setdefault("impervious_proxy", NA)
            row.setdefault("landcover_source", NA)
            row.setdefault("landcover_reference_year", NA)
            row["territorial_fonte_real"] = "false"
        row["coverage_territorial"] = _fmt(cov_terr, 4)
        row["coverage_score_is_susceptibility"] = "false"
        # hard defaults
        row["ground_truth"] = "false"
        row["eligible_for_training"] = "false"
        row["score_v7_allowed"] = "false"
        row["score_oficial"] = "false"
        row["substituir_score_v6"] = "false"
        row["usar_em_treino"] = "false"
        row["review_only"] = "true"
        out.append(row)
    cov_terr_medio = sum(cov_terr_vals) / len(cov_terr_vals) if cov_terr_vals else 0.0
    return out, {"coverage_territorial_medio": cov_terr_medio, "n": len(out)}


# ---------------------------------------------------------------------------
# 8. Comparacao / auditoria de cobertura
# ---------------------------------------------------------------------------
def comparacao_rows(ctx: dict, cov_terr_19f: float, filled: int) -> list[dict]:
    t19a = ctx["cov_terr_19a"]
    data = [
        ("cobertura_territorial", _fmt(t19a, 4), _fmt(ctx["cov_terr_19b"], 4), _fmt(cov_terr_19f, 4),
         "territorial passa de 2/6 para 6/6 nos patches preenchidos" if filled else "territorial inalterado (sem CSV real)"),
        ("cobertura_total", _fmt(ctx["cov_total_19a"], 4), _fmt(ctx["cov_total_19a"], 4),
         _fmt(_cov_total_19f(ctx, cov_terr_19f), 4),
         "cobertura total recalculada apenas como completude; nao e suscetibilidade"),
        ("patches_preenchidos", "0", "0", str(filled),
         "patches com features territoriais reais do MapBiomas"),
        ("features_destravadas", "0", "0",
         str(len([c for c in ("mapbiomas_class_majority", "water_prop", "exposed_soil_prop", "impervious_proxy") ]) if filled else 0),
         "features territoriais destravadas quando ha CSV real"),
    ]
    return [{
        "metrica": m, "valor_19a": a, "valor_19b": b, "valor_19f": f, "observacao": o,
    } for m, a, b, f, o in data]


def _cov_total_19f(ctx: dict, cov_terr_19f: float) -> float:
    # media das 6 familias: fisico, espectral, chuva (1.0), documental, observacional (herdadas), territorial (nova)
    fam = [ctx["cov_fisico"], ctx["cov_espectral"], ctx["cov_chuva"],
           ctx["cov_documental"], ctx["cov_observacional"], cov_terr_19f]
    return sum(fam) / len(fam)


def auditoria_cobertura_rows(ctx: dict, cov_terr_19f: float, filled: int) -> list[dict]:
    familias = [
        ("fisica_topografica", ctx["cov_fisico"], "completa", "nenhuma"),
        ("espectral_umidade", ctx["cov_espectral"], "completa", "nenhuma"),
        ("chuva_hidrometeorologica", ctx["cov_chuva"], "completa", "nenhuma"),
        ("territorial", cov_terr_19f, "completa" if filled == EXPECTED_PATCHES else ("parcial" if filled else "aguardando_export"),
         "nenhuma" if filled == EXPECTED_PATCHES else ";".join(TERRITORIAL_TARGET_FEATURES)),
        ("documental", ctx["cov_documental"], "rara", "documental so em Recife"),
        ("observacional", ctx["cov_observacional"], "rara", "evidencia so em Recife e SAR de Curitiba"),
    ]
    return [{
        "familia": f, "cobertura_media": _fmt(c, 4), "status_cobertura": st,
        "principal_lacuna": lac, "coverage_is_susceptibility": "false",
        "observacao": "cobertura e completude, nao suscetibilidade",
    } for f, c, st, lac in familias]


# ---------------------------------------------------------------------------
# 9. Gates
# ---------------------------------------------------------------------------
def status_19f(env_status: str, csv_exists: bool, filled: int) -> str:
    if csv_exists and filled == EXPECTED_PATCHES:
        return "19F_TERRITORIAL_PREENCHIDO_MAPBIOMAS_GEE"
    if csv_exists and filled > 0:
        return "19F_TERRITORIAL_PREENCHIDO_MAPBIOMAS_GEE"
    if env_status == "gee_nao_autenticado":
        return "19F_BLOQUEADO_POR_AUTENTICACAO"
    if env_status == "permissao_insuficiente":
        return "19F_BLOQUEADO_POR_PERMISSAO"
    if env_status == "biblioteca_ausente":
        return "19F_BLOQUEADO_FAIL_CLOSED"
    return "19F_BLOQUEADO_FAIL_CLOSED"


def gate_territorial_rows(summary: dict) -> list[dict]:
    def g(criterio, valor, limiar, passou, obs):
        return {"criterio": criterio, "valor_observado": str(valor), "limiar": limiar,
                "passou": "true" if passou else "false", "status_19f": summary["status_19f"], "observacao": obs}
    filled = summary["patches_preenchidos"]
    return [
        g("csv_real_recuperado", str(summary["csv_recuperado"]).lower(), "true", summary["csv_recuperado"],
          "CSV leve do MapBiomas recuperado localmente" if summary["csv_recuperado"] else "sem CSV real (fail-closed)"),
        g("patches_preenchidos", filled, f"=={EXPECTED_PATCHES}", filled == EXPECTED_PATCHES,
          "features territoriais reais por patch"),
        g("mapbiomas_nao_e_ground_truth", "true", "true", True, "uso do solo nao vira ground truth"),
        g("coverage_nao_e_suscetibilidade", "true", "true", True, "coverage_score e completude"),
        g("score_v6_intacto", str(not summary["score_v6_changed"]).lower(), "true", not summary["score_v6_changed"],
          "score_v6 permanece baseline oficial"),
        g("sem_valor_inventado", "true", "true", True, "features vazias quando nao ha CSV real"),
    ]


def gate_v7_rows() -> list[dict]:
    data = [
        ("autorizacao_geral", "true", "SCORE_V7_NAO_AUTORIZADO", "score_v7 nao autorizado nesta etapa"),
        ("amostra_observacional", "true", "SCORE_V7_BLOQUEADO_POR_AMOSTRA", "apenas 7 patches observacionais"),
        ("ausencia_benchmark", "true", "SCORE_V7_BLOQUEADO_POR_AUSENCIA_BENCHMARK", "sem benchmark 17B"),
        ("validacao_insuficiente", "true", "SCORE_V7_BLOQUEADO_POR_VALIDACAO_INSUFICIENTE", "territorial preenchido nao valida score"),
    ]
    return [{"criterio": c, "bloqueio": b, "status_score_v7": s, "observacao": o} for c, b, s, o in data]


def gate_17b_rows() -> list[dict]:
    data = [
        ("benchmark_criado", "false", "false", "17B_NAO_CRIADO", "nenhum benchmark 17B criado"),
        ("geometria_oficial", "false", "true", "17B_BLOQUEADO_POR_GEOMETRIA_OFICIAL", "Curitiba so SAR; sem geometria oficial"),
        ("segunda_regiao_tecnica", "true", "true", "17B_APROXIMACAO_COM_SEGUNDA_REGIAO_TECNICA", "Recife forte + Curitiba tecnica SAR"),
    ]
    return [{"criterio": c, "valor_observado": v, "limiar": l, "status_17b": s, "observacao": o}
            for c, v, l, s, o in data]


def gate_19g_rows(summary: dict) -> list[dict]:
    filled = summary["patches_preenchidos"]
    if filled == EXPECTED_PATCHES:
        status = "19G_PRONTO_APOS_TERRITORIAL"
    elif filled > 0:
        status = "19G_PRONTO_COM_RESSALVAS"
    else:
        status = "19G_BLOQUEADO_POR_TERRITORIAL_INCOMPLETO"
    data = [
        ("territorial_preenchido", str(filled), f"=={EXPECTED_PATCHES}", "features territoriais reais ingeridas"),
        ("score_v6_intacto", str(not summary["score_v6_changed"]).lower(), "true", "baseline preservado"),
        ("sem_score_novo", "true", "true", "nenhum score novo criado em 19F"),
    ]
    return [{"criterio": c, "valor_observado": v, "limiar": l, "status_19g": status, "observacao": o}
            for c, v, l, o in data]


# ---------------------------------------------------------------------------
# Resumos
# ---------------------------------------------------------------------------
def resumo_regiao_rows(terr_rows: list[dict]) -> list[dict]:
    by_reg = {}
    for r in terr_rows:
        by_reg.setdefault(r["regiao"], []).append(r)
    out = []
    for reg in ("curitiba", "petropolis", "recife"):
        rs = by_reg.get(reg, [])
        filled = [r for r in rs if r["fonte_real"] == "true"]
        imp = [_num(r.get("impervious_proxy")) for r in filled]
        imp = [v for v in imp if v is not None]
        wat = [_num(r.get("water_prop")) for r in filled]
        wat = [v for v in wat if v is not None]
        out.append({
            "regiao": reg, "patches": str(len(rs)), "patches_preenchidos": str(len(filled)),
            "impervious_proxy_medio": _fmt(sum(imp) / len(imp), 4) if imp else NA,
            "water_prop_medio": _fmt(sum(wat) / len(wat), 4) if wat else NA,
            "territorial_status": "preenchido" if filled else "aguardando_export",
            "observacao": "uso do solo review-only; nao e ground truth",
        })
    return out


def resumo_feature_rows(terr_rows: list[dict]) -> list[dict]:
    filled = [r for r in terr_rows if r["fonte_real"] == "true"]
    n = len(filled)
    feats = ["water_prop", "exposed_soil_prop", "impervious_proxy", "vegetation_prop_mapbiomas"]
    out = []
    for f in feats:
        vals = [_num(r.get(f)) for r in filled]
        vals = [v for v in vals if v is not None]
        out.append({
            "feature": f, "fonte": LANDCOVER_SOURCE if n else NA,
            "patches_preenchidos": str(n),
            "media": _fmt(sum(vals) / len(vals), 4) if vals else NA,
            "minimo": _fmt(min(vals), 4) if vals else NA,
            "maximo": _fmt(max(vals), 4) if vals else NA,
            "fonte_real": "true" if n else "false",
        })
    out.append({
        "feature": "mapbiomas_class_majority", "fonte": LANDCOVER_SOURCE if n else NA,
        "patches_preenchidos": str(n),
        "media": NA, "minimo": NA, "maximo": NA, "fonte_real": "true" if n else "false",
    })
    return out


# ---------------------------------------------------------------------------
# Contexto herdado
# ---------------------------------------------------------------------------
def load_context() -> dict:
    s19a = read_json(SUM_19A)
    s19b = read_json(SUM_19B)
    return {
        "cov_fisico": s19a.get("coverage_fisico_medio", 1.0),
        "cov_espectral": s19a.get("coverage_espectral_medio", 1.0),
        "cov_chuva": s19a.get("coverage_chuva_medio", 1.0),
        "cov_documental": s19a.get("coverage_documental_medio", 0.02),
        "cov_observacional": s19a.get("coverage_observacional_medio", 0.0267),
        "cov_terr_19a": s19a.get("coverage_territorial_medio", 0.3333),
        "cov_terr_19b": s19b.get("coverage_territorial_medio_19b", 0.3333),
        "cov_total_19a": s19a.get("coverage_total_medio", 0.5633),
        "total_patches": s19a.get("total_patches", EXPECTED_PATCHES),
    }


# ---------------------------------------------------------------------------
# Schema / preflight / summary
# ---------------------------------------------------------------------------
def schema_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SUSC-19F execucao MapBiomas/GEE e ingestao territorial",
        "type": "object",
        "description": "Execucao controlada MapBiomas/GEE; feature territorial review-only; sem GT/treino/score_v7/benchmark; score_v6 intacto; nunca inventa valores.",
        "guardrails": {
            "ground_truth": {"const": "false"},
            "eligible_for_training": {"const": "false"},
            "score_v7_allowed": {"const": "false"},
            "score_oficial": {"const": "false"},
            "substituir_score_v6": {"const": "false"},
            "usar_em_treino": {"const": "false"},
            "score_v6_changed": {"const": "false"},
            "review_only": {"const": "true"},
        },
        "status_19f_allowed": sorted(STATUS_19F_ALLOWED),
        "status_score_v7_allowed": sorted(STATUS_V7_ALLOWED),
        "status_17b_allowed": sorted(STATUS_17B_ALLOWED),
        "status_19g_allowed": sorted(STATUS_19G_ALLOWED),
        "env_status_allowed": sorted(ENV_STATUS_ALLOWED),
        "mapbiomas_collection": MAPBIOMAS_COLLECTION,
        "mapbiomas_year": MAPBIOMAS_YEAR,
        "landcover_source": LANDCOVER_SOURCE,
        "coverage_is_susceptibility": False,
        "territorial_target_features": TERRITORIAL_TARGET_FEATURES,
    }


def preflight_obj(ctx: dict, env_status: str, csv_exists: bool) -> dict:
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
        "pacote_gee_19b": rel(PKG_19B),
        "manifest_patches": len(load_manifest()),
        "local_run_dir": rel(LOCAL_RUN),
        "local_csv_existe": csv_exists,
        "env_status": env_status,
        "cobertura_territorial_19b": ctx["cov_terr_19b"],
        "outputs_ignorados_pelo_git": sorted(git_ignored([OUT])),
        "summary_versionavel": rel(SUMMARY) not in git_ignored([SUMMARY]),
    }


def summary_obj(ctx: dict, env_status: str, csv_exists: bool, filled: int,
                cov_terr_19f: float, n_matriz: int) -> dict:
    return {
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "staged_count": len(_run_git(["diff", "--cached", "--name-only"]).splitlines()),
        "env_status": env_status,
        "earth_engine_autenticado": env_status == "gee_autenticado_pronto",
        "csv_recuperado": csv_exists,
        "total_patches": ctx["total_patches"],
        "patches_matriz": n_matriz,
        "patches_preenchidos": filled,
        "cobertura_territorial_19a": round(ctx["cov_terr_19a"], 4),
        "cobertura_territorial_19b": round(ctx["cov_terr_19b"], 4),
        "cobertura_territorial_19f": round(cov_terr_19f, 4),
        "cobertura_total_19a": round(ctx["cov_total_19a"], 4),
        "cobertura_total_19f": round(_cov_total_19f(ctx, cov_terr_19f), 4),
        "mapbiomas_collection": MAPBIOMAS_COLLECTION,
        "mapbiomas_year": MAPBIOMAS_YEAR,
        "landcover_source": LANDCOVER_SOURCE if filled else NA,
        "coverage_is_susceptibility": False,
        "status_19f": status_19f(env_status, csv_exists, filled),
        "status_score_v7": "SCORE_V7_NAO_AUTORIZADO",
        "status_17b": "17B_NAO_CRIADO",
        "marco_17b_criado": False,
        "benchmark_17b_criado": False,
        "ground_truth_true_count": 0,
        "eligible_for_training_true_count": 0,
        "score_v7_allowed_true_count": 0,
        "score_v6_changed": _score_v6_changed(),
        "score_v7_created": SCORE_V7.exists(),
        "review_only": True,
        "proximo_marco": "SUSC-19G (recalibracao candidata review-only apos territorial)" if filled == EXPECTED_PATCHES
        else "reexecutar 19F (MapBiomas/GEE) ate territorial completo",
    }


# ---------------------------------------------------------------------------
# 11. Cartoes regionais
# ---------------------------------------------------------------------------
def _card_regiao(reg: str, ctx: dict, resumo_reg: dict, csv_exists: bool) -> str:
    if resumo_reg and resumo_reg.get("fonte_real_disponivel"):
        pass
    filled = int(resumo_reg.get("patches_preenchidos", "0")) if resumo_reg else 0
    imp = resumo_reg.get("impervious_proxy_medio", NA) if resumo_reg else NA
    wat = resumo_reg.get("water_prop_medio", NA) if resumo_reg else NA
    resultado = (
        f"MapBiomas 19F: {filled} patches preenchidos (colecao 9, ano {MAPBIOMAS_YEAR}); "
        f"impervious_proxy medio {imp}, water_prop medio {wat}."
        if filled else
        "MapBiomas 19F: nenhum patch preenchido nesta regiao (sem CSV real; feature nao inventada)."
    )
    lacunas = "nenhuma (territorial completo)" if filled else ";".join(TERRITORIAL_TARGET_FEATURES)
    return f"""# Cartao territorial - {CIDADE.get(reg, reg)}

## Cobertura territorial 19A/19B
Antes do 19F a cobertura territorial era {_fmt(ctx['cov_terr_19a'], 4)} (2 de 6 features:
urban_prop e vegetation_prop). O 19B manteve {_fmt(ctx['cov_terr_19b'], 4)} e preparou o
pacote MapBiomas/GEE.

## Resultado MapBiomas 19F
{resultado}

## Classes majoritarias
Classe majoritaria por patch registrada em `matriz_territorial_mapbiomas_19f.csv`
(codigo MapBiomas e rotulo). {"Predominio urbano nas areas construidas." if filled else "Aguardando CSV real."}

## water_prop / exposed_soil / impervious
Proporcoes derivadas do histograma de classes MapBiomas por patch (agua, solo exposto,
area urbanizada). {"Valores reais ingeridos." if filled else "Nao preenchidos (nao inventados)."}

## Lacunas restantes
{lacunas}

## Por que nao e ground truth
Uso/cobertura do solo e feature territorial escalavel review-only; MapBiomas nao e
verdade de campo de enchente e nao vira ground truth.

## Por que nao e treino
eligible_for_training=false; a feature territorial nao habilita treino supervisionado.

## Por que nao cria score_v7
score_v7 permanece bloqueado (amostra, benchmark e validacao); preencher territorial
nao autoriza score_v7 e nao altera o score_v6.
"""


def write_cards(ctx: dict, resumo_regiao: list[dict], csv_exists: bool) -> None:
    ensure_dir(CARDS)
    by_reg = {r["regiao"]: r for r in resumo_regiao}
    for reg in ("recife", "curitiba", "petropolis"):
        write_markdown(CARDS / f"cartao_{reg}.md", _card_regiao(reg, ctx, by_reg.get(reg, {}), csv_exists))


# ---------------------------------------------------------------------------
# 15. Relatorio publico
# ---------------------------------------------------------------------------
def report_text(ctx: dict, summary: dict, env_rows: list[dict], comp: list[dict], aud: list[dict]) -> str:
    linhas_comp = "\n".join(
        f"| {r['metrica']} | {r['valor_19a']} | {r['valor_19b']} | {r['valor_19f']} |" for r in comp
    )
    linhas_aud = "\n".join(
        f"| {r['familia']} | {r['cobertura_media']} | {r['status_cobertura']} | {r['principal_lacuna']} |" for r in aud
    )
    env_status = summary["env_status"]
    return f"""# SUSC-19F - Execucao MapBiomas/GEE e ingestao territorial

## Estado herdado do 19B/19E

O 19B identificou a lacuna territorial (MapBiomas/solo exposto/agua/impermeabilizacao)
nos 300 patches e preparou o pacote MapBiomas/GEE. O 19E consolidou a comunicacao
review-only. Esta etapa executa o pacote e ingere as features territoriais reais,
sem criar score novo e sem alterar o score_v6.

## Ambiente GEE

Status do ambiente: `{env_status}`. Verificacoes de Python, biblioteca earthengine,
CLI, `ee.Initialize()` (sem impressao de credenciais), conectividade, permissao de
compute e pasta gravavel em `auditoria_ambiente_gee_19f.csv`.

## Execucao MapBiomas

Colecao `{MAPBIOMAS_COLLECTION}`, ano {MAPBIOMAS_YEAR}, escala {MAPBIOMAS_SCALE} m. Para
cada patch calcula-se o histograma de classes (`frequencyHistogram`) e derivam-se a
classe majoritaria, `water_prop`, `exposed_soil_prop` e `impervious_proxy`. Nao se baixa
raster pesado; exporta-se apenas CSV leve.

## Export / recuperacao

Rotas de recuperacao em `recuperacao_export_mapbiomas_19f.csv`. CSV leve recuperado:
`{str(summary['csv_recuperado']).lower()}`. Patches preenchidos:
{summary['patches_preenchidos']} de {summary['total_patches']}.

## Ingestao territorial

Quando ha CSV real, `matriz_territorial_mapbiomas_19f.csv` recebe as features
territoriais reais e `matriz_multimodal_19f_atualizada.csv` atualiza a cobertura de
completude. urban_prop e vegetation_prop do 19A sao preservados. Nenhum valor e
inventado: sem CSV real, a feature fica vazia e o status e `aguardando_export`.

## Cobertura antes/depois

| Metrica | 19A | 19B | 19F |
| --- | --- | --- | --- |
{linhas_comp}

| Familia | Cobertura | Status | Principal lacuna |
| --- | --- | --- | --- |
{linhas_aud}

## Lacunas restantes

{"Nenhuma lacuna territorial: as 6 features estao completas." if summary['patches_preenchidos'] == ctx['total_patches'] else "Territorial ainda incompleto; reexecutar o pacote MapBiomas/GEE."}

## Por que nao e ground truth

Uso/cobertura do solo e feature territorial escalavel review-only; MapBiomas nao e
verdade de campo de enchente e nunca vira ground truth.

## Por que nao e treino

eligible_for_training=false; a feature territorial nao habilita treino supervisionado
e nao cria rotulo nem target.

## Por que nao e score_v7

O score_v7 permanece `{summary['status_score_v7']}`. Preencher territorial nao valida
score, nao cria benchmark e nao altera o score_v6, que segue baseline oficial.

## Proximo marco recomendado

{summary['proximo_marco']}.
"""


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_all() -> dict:
    require_inputs()
    ensure_dir(OUT)
    ensure_dir(REPORTS)
    ensure_dir(SCHEMAS_DIR)

    ctx = load_context()
    manifest = load_manifest()
    env_rows, env_status = env_audit_rows()
    csv_exists = LOCAL_CSV.exists()
    terr = load_local_territorial()

    terr_rows, terr_meta = matriz_territorial_rows(manifest, terr)
    filled = terr_meta["filled"]
    ingested = filled > 0
    mm_rows, mm_meta = matriz_multimodal_rows(terr_rows, ingested)
    cov_terr_19f = mm_meta["coverage_territorial_medio"]

    summary = summary_obj(ctx, env_status, csv_exists, filled, cov_terr_19f, mm_meta["n"])
    resumo_regiao = resumo_regiao_rows(terr_rows)

    write_csv(AUD_AMBIENTE, env_rows)
    write_csv(STATUS_TASKS, status_tasks_rows())
    write_csv(RECUPERACAO, recuperacao_rows(csv_exists))
    write_csv(MAT_TERRITORIAL, terr_rows, fieldnames=terr_meta["fieldnames"])
    write_csv(MAT_MULTIMODAL, mm_rows)
    write_csv(COMPARACAO, comparacao_rows(ctx, cov_terr_19f, filled))
    write_csv(AUD_COBERTURA, auditoria_cobertura_rows(ctx, cov_terr_19f, filled))
    write_csv(GATE_TERRITORIAL, gate_territorial_rows(summary))
    write_csv(GATE_V7, gate_v7_rows())
    write_csv(GATE_17B, gate_17b_rows())
    write_csv(GATE_19G, gate_19g_rows(summary))
    write_csv(RESUMO_REGIAO, resumo_regiao)
    write_csv(RESUMO_FEATURE, resumo_feature_rows(terr_rows))

    write_json(SCHEMA, schema_obj())
    write_json(PREFLIGHT, preflight_obj(ctx, env_status, csv_exists))
    write_json(SUMMARY, summary)
    write_cards(ctx, resumo_regiao, csv_exists)
    write_markdown(REPORT, report_text(ctx, summary, env_rows,
                                       comparacao_rows(ctx, cov_terr_19f, filled),
                                       auditoria_cobertura_rows(ctx, cov_terr_19f, filled)))
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
        txt = path.read_text(encoding="utf-8", errors="ignore")
        hits = public_text_violations_text(txt)
        if hits:
            errors.append(f"{rel(path)}:vocabulario_publico_proibido:{','.join(hits)}")
        if SECRET_RE.search(txt):
            errors.append(f"{rel(path)}:credencial_ou_token_em_output_publico")
    return errors


NEGATIVE_RE = re.compile(r"\bnegativ", re.IGNORECASE)


def validate_outputs(summary: dict) -> list[str]:
    errors = [f"missing:{rel(p)}" for p in REQUIRED_OUTPUTS if not p.exists()]

    # guardrails do summary
    if summary["score_v6_changed"]:
        errors.append("score_v6_changed_forbidden")
    if summary["score_v7_created"]:
        errors.append("score_v7_created_forbidden")
    if summary["benchmark_17b_criado"] or summary["marco_17b_criado"]:
        errors.append("benchmark_ou_marco_17b_criado_forbidden")
    for f in ("ground_truth_true_count", "eligible_for_training_true_count", "score_v7_allowed_true_count"):
        if summary[f]:
            errors.append(f"{f}_forbidden")
    if summary["status_19f"] not in STATUS_19F_ALLOWED:
        errors.append(f"status_19f_fora_enum:{summary['status_19f']}")
    if summary["status_score_v7"] not in STATUS_V7_ALLOWED:
        errors.append(f"status_score_v7_fora_enum:{summary['status_score_v7']}")
    if summary["status_17b"] not in STATUS_17B_ALLOWED:
        errors.append(f"status_17b_fora_enum:{summary['status_17b']}")
    if summary["coverage_is_susceptibility"]:
        errors.append("coverage_chamado_de_suscetibilidade")

    # ambiente: status dentro do enum; sem credenciais
    env = read_csv(AUD_AMBIENTE)
    if not env:
        errors.append("auditoria_ambiente_vazia")
    for r in env:
        if r.get("env_status") not in ENV_STATUS_ALLOWED:
            errors.append(f"env_status_fora_enum:{r.get('env_status')}")

    # matriz territorial: patch_id unico, nao vazia, sem valor inventado
    terr = read_csv(MAT_TERRITORIAL)
    if not terr:
        errors.append("matriz_territorial_vazia")
    ids = [r["patch_id"] for r in terr]
    if len(ids) != len(set(ids)):
        errors.append("matriz_territorial_patch_id_duplicado")
    for r in terr:
        # feature marcada presente exige fonte real
        has_val = _num(r.get("water_prop")) is not None or _num(r.get("impervious_proxy")) is not None
        if has_val and r.get("fonte_real") != "true":
            errors.append(f"territorial:{r['patch_id']}:feature_presente_sem_fonte_real")
        if r.get("fonte_real") != "true":
            # sem fonte real, features territoriais MapBiomas devem estar vazias (NA)
            for f in ("water_prop", "exposed_soil_prop", "impervious_proxy", "mapbiomas_class_majority"):
                if _num(r.get(f)) is not None:
                    errors.append(f"territorial:{r['patch_id']}:{f}_inventado_sem_fonte")
        for field in ("ground_truth", "eligible_for_training", "score_v7_allowed"):
            if r.get(field) != "false":
                errors.append(f"territorial:{r['patch_id']}:{field}_nao_false")

    # CSV declarado recuperado exige arquivo real
    if summary["csv_recuperado"] and not LOCAL_CSV.exists():
        errors.append("csv_declarado_recuperado_sem_arquivo")
    # patches preenchidos exige CSV real
    if summary["patches_preenchidos"] > 0 and not LOCAL_CSV.exists():
        errors.append("patches_preenchidos_sem_csv_real")

    # matriz multimodal: nao vazia, patch_id unico, hard defaults, coverage nao e suscetibilidade
    mm = read_csv(MAT_MULTIMODAL)
    if not mm:
        errors.append("matriz_multimodal_vazia")
    mmids = [r["patch_id"] for r in mm]
    if len(mmids) != len(set(mmids)):
        errors.append("matriz_multimodal_patch_id_duplicado")
    if summary["patches_matriz"] and len(mm) != EXPECTED_PATCHES:
        errors.append(f"matriz_multimodal_diferente_de_300:{len(mm)}")
    for r in mm:
        for field in ("ground_truth", "eligible_for_training", "score_v7_allowed",
                      "score_oficial", "substituir_score_v6", "usar_em_treino"):
            if r.get(field) != "false":
                errors.append(f"multimodal:{r['patch_id']}:{field}_nao_false")
        if r.get("coverage_score_is_susceptibility") != "false":
            errors.append(f"multimodal:{r['patch_id']}:coverage_como_suscetibilidade")

    # SAR pos-evento nunca como feature territorial (nas matrizes)
    for path in (MAT_TERRITORIAL, MAT_MULTIMODAL):
        for r in read_csv(path):
            for k, v in r.items():
                kl = k.lower()
                if ("water_prop" in kl or "exposed_soil" in kl or "impervious" in kl
                        or "mapbiomas" in kl) and "sar" in str(v).lower():
                    errors.append(f"{rel(path)}:sar_como_feature_territorial")
                    break

    # gates enums
    for r in read_csv(GATE_TERRITORIAL):
        if r.get("status_19f") not in STATUS_19F_ALLOWED:
            errors.append(f"gate_territorial:status_fora_enum:{r.get('status_19f')}")
    for r in read_csv(GATE_V7):
        if r.get("status_score_v7") not in STATUS_V7_ALLOWED:
            errors.append(f"gate_v7:status_fora_enum:{r.get('status_score_v7')}")
    for r in read_csv(GATE_17B):
        if r.get("status_17b") not in STATUS_17B_ALLOWED:
            errors.append(f"gate_17b:status_fora_enum:{r.get('status_17b')}")
    for r in read_csv(GATE_19G):
        if r.get("status_19g") not in STATUS_19G_ALLOWED:
            errors.append(f"gate_19g:status_fora_enum:{r.get('status_19g')}")

    # coverage nao pode ser chamado de score de suscetibilidade no texto publico
    for path in _public_paths():
        txt = path.read_text(encoding="utf-8", errors="ignore").lower()
        if re.search(r"coverage[_ ]?score[^.\n]{0,30}suscetibilidade", txt) and "nao e suscetibilidade" not in txt and "e completude" not in txt:
            errors.append(f"{rel(path)}:coverage_como_suscetibilidade_texto")
        for m in NEGATIVE_RE.finditer(txt):
            window = txt[max(0, m.start() - 40): m.end() + 40]
            if "nao e negativo" not in window and "nunca e negativo" not in window and "nunca negativo" not in window:
                errors.append(f"{rel(path)}:termo_negativo_sem_ressalva")
                break

    # output 19F nao pode ser ignorado pelo git
    if rel(SUMMARY) in git_ignored([SUMMARY, MAT_TERRITORIAL]):
        errors.append("output_19f_ignorado_pelo_git")

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
        "19F execucao MapBiomas validada: "
        f"env={summary['env_status']} csv={str(summary['csv_recuperado']).lower()} "
        f"preenchidos={summary['patches_preenchidos']}/{summary['total_patches']} "
        f"cov_terr {summary['cobertura_territorial_19a']}->{summary['cobertura_territorial_19f']} "
        f"status19F={summary['status_19f']} scorev7={summary['status_score_v7']} 17B={summary['status_17b']} "
        f"score_v6_changed={str(summary['score_v6_changed']).lower()}"
    )
    return 0


def run_all() -> int:
    summary = build_all()
    print(
        "19F execucao MapBiomas gerada: "
        f"env={summary['env_status']} csv={str(summary['csv_recuperado']).lower()} "
        f"preenchidos={summary['patches_preenchidos']} status19F={summary['status_19f']}"
    )
    return 0
