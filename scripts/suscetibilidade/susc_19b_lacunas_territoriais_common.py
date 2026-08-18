"""SUSC-19B auditoria e preenchimento de lacunas territoriais multimodais.

Aprofunda o missingness territorial do 19A e tenta preencher, com dados reais, as
features territoriais faltantes (MapBiomas, exposed_soil, water_prop, impervious).
Se nao houver fonte local utilizavel para os 300 patches, gera um pacote executavel
de extracao MapBiomas/GEE, sem inventar valores.

Auditoria das fontes locais (real):
- susc_18a_landcover_urban_feature_store.csv: 300 linhas "oficiais" apenas
  reembrulham urban_prop/vegetation_prop de susc_features_by_patch_v1; water_prop
  not_available; sem MapBiomas nem exposed_soil.
- susc_17c34/17c35_landcover_urban_features.csv: apenas canarios (11), fora do
  universo base de 300.
- mapbiomas_col10_1_...xlsx: nivel estado/bioma, em quarentena (local_only).
Conclusao: sem fonte local utilizavel para os alvos territoriais dos 300 patches
=> pacote MapBiomas/GEE.

Guardrails: territorial e feature de suscetibilidade escalavel (nao evidencia de
evento); SAR pos-evento nunca vira feature territorial; coverage_score mede
completude, nao suscetibilidade; score_v6 intacto; nada de ground truth/treino/score_v7.
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
OUT = ROOT / "outputs_public" / "data" / "susc_19b_auditoria_lacunas_territoriais"
PACOTE_GEE = OUT / "pacote_gee"
CARDS = OUT / "cartoes_regionais"
REPORTS = ROOT / "outputs_public" / "reports"
SCHEMAS_DIR = ROOT / "schemas" / "suscetibilidade"

REPORT = REPORTS / "SUSC_19B_AUDITORIA_LACUNAS_TERRITORIAIS.md"
SCHEMA = SCHEMAS_DIR / "susc_19b_lacunas_territoriais_schema_v1.json"

# Fontes herdadas / reais
OUT_19A = ROOT / "outputs_public" / "data" / "susc_19a_matriz_multimodal_escalavel_por_patch"
MAT_19A = OUT_19A / "matriz_multimodal_escalavel_por_patch.csv"
INDICE_19A = OUT_19A / "indice_cobertura_multimodal_review_only.csv"
FEATURES_SRC = ROOT / "datasets" / "suscetibilidade" / "susc_features_by_patch_v1.csv"
SCORE_V6 = ROOT / "datasets" / "suscetibilidade" / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = ROOT / "datasets" / "suscetibilidade" / "susc_score_v7_candidate_by_patch_v1.csv"
LANDCOVER_18A = ROOT / "outputs_public" / "suscetibilidade" / "susc_18a_landcover_urban_feature_store.csv"
LANDCOVER_17C35 = ROOT / "outputs_public" / "suscetibilidade" / "susc_17c35_landcover_urban_features.csv"
LANDCOVER_17C34 = ROOT / "outputs_public" / "suscetibilidade" / "susc_17c34_landcover_urban_features.csv"
MAPBIOMAS_XLSX = ROOT / "local_only" / "evidencias_externas_quarentena" / "fontes_nacionais" / "mapbiomas_col10_1_cobertura_dhn250_estado_bioma.xlsx"

# Saidas
AUDITORIA_MISS = OUT / "auditoria_missingness_territorial_19b.csv"
INVENTARIO = OUT / "inventario_fontes_territoriais_19b.csv"
MAT_EXTRAIDA = OUT / "matriz_territorial_extraida_19b.csv"
FILA = OUT / "fila_extracao_territorial_19b.csv"
MAT_19B = OUT / "matriz_multimodal_19b_atualizada.csv"
COBERTURA_POS = OUT / "auditoria_cobertura_pos_19b.csv"
COMPARACAO = OUT / "comparacao_cobertura_19a_19b.csv"
GATE_19B = OUT / "gate_territorial_19b.csv"
GATE_19C = OUT / "gate_prontidao_19c_avaliacao_observacional.csv"
RESUMO_REGIAO = OUT / "resumo_por_regiao.csv"
RESUMO_FEATURE = OUT / "resumo_por_feature.csv"
SUMMARY = OUT / "summary.json"
PREFLIGHT = OUT / "preflight.json"

GEE_JS = PACOTE_GEE / "gee_mapbiomas_patch_landcover_19b.js"
GEE_MD = PACOTE_GEE / "gee_mapbiomas_patch_landcover_19b.md"
GEE_MANIFEST = PACOTE_GEE / "gee_export_manifest_19b.csv"
GEE_SCHEMA = PACOTE_GEE / "expected_outputs_schema_19b.json"

REQUIRED_INPUTS = [MAT_19A, INDICE_19A, FEATURES_SRC, SCORE_V6]
REQUIRED_OUTPUTS = [
    AUDITORIA_MISS, INVENTARIO, MAT_EXTRAIDA, FILA, MAT_19B, COBERTURA_POS, COMPARACAO,
    GATE_19B, GATE_19C, RESUMO_REGIAO, RESUMO_FEATURE, SUMMARY, PREFLIGHT, SCHEMA, REPORT,
    GEE_JS, GEE_MD, GEE_MANIFEST, GEE_SCHEMA,
]

# ---------------------------------------------------------------------------
# Guardrails / constantes
# ---------------------------------------------------------------------------
PUBLIC_FORBIDDEN_RE = re.compile(r"\b(?:agentic|agente|codex|llm|ia)\b", re.IGNORECASE)
NA = "NA"

STATUS_19B_ALLOWED = {
    "19B_TERRITORIAL_PREENCHIDO_COM_FONTE_LOCAL",
    "19B_PACOTE_MAPBIOMAS_GEE_PRONTO",
    "19B_TERRITORIAL_PARCIAL_COM_FILAS_EXECUTAVEIS",
    "19B_BLOQUEADO_POR_FONTE_TERRITORIAL",
    "19B_BLOQUEADO_FAIL_CLOSED",
}
STATUS_USO_ALLOWED = {"utilizavel_direto", "utilizavel_com_extracao", "pacote_externo_necessario", "insuficiente", "ausente"}
STATUS_17B_MESTRE = "17B_APROXIMACAO_COM_SEGUNDA_REGIAO_TECNICA"

# Alvos territoriais desta sprint
TERRITORIAL_TARGETS = ["MapBiomas_class_majority", "MapBiomas_class_distribution", "exposed_soil_prop", "water_prop", "impervious_proxy", "landcover_source", "landcover_reference_year"]
# Features territoriais totais (para cobertura, coerente com 19A: 6 esperadas)
TERRITORIAL_EXPECTED = ["urban_prop", "vegetation_prop", "water_prop", "exposed_soil_prop", "impervious_proxy", "MapBiomas_class_majority"]
TERRITORIAL_PRESENTE_19A = {"urban_prop", "vegetation_prop"}

CIDADE = {"recife": "Recife", "curitiba": "Curitiba", "petropolis": "Petropolis"}

MAPBIOMAS_COLLECTION = "projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_integration_v1"
MAPBIOMAS_YEAR = 2022


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


# ---------------------------------------------------------------------------
# Inventario de fontes territoriais (auditoria real)
# ---------------------------------------------------------------------------
def inventario_rows() -> list[dict]:
    def row(fid, caminho, tipo, regiao, ano, crs, res, fmt, feats, status, motivo):
        return {
            "fonte_id": fid, "caminho": caminho, "tipo": tipo, "regiao_coberta": regiao,
            "ano": ano, "crs": crs, "resolucao": res, "formato": fmt,
            "features_possiveis": feats, "status_uso": status, "motivo": motivo,
        }
    return [
        row("LOCAL_18A_STORE", rel(LANDCOVER_18A) if LANDCOVER_18A.exists() else "not_available",
            "tabela_landcover_por_patch", "recife;curitiba;petropolis (300 base)", "not_available", "EPSG:4326",
            "por_patch", "csv", "urban_prop;vegetation_prop", "insuficiente",
            "as 300 linhas oficiais apenas reembrulham urban_prop/vegetation_prop de susc_features_by_patch_v1; water_prop not_available; sem MapBiomas nem exposed_soil"),
        row("LOCAL_17C35", rel(LANDCOVER_17C35) if LANDCOVER_17C35.exists() else "not_available",
            "tabela_landcover_canario", "recife (canarios)", "not_available", "EPSG:4326", "por_patch", "csv",
            "urban_prop;vegetation_prop;water_prop", "insuficiente",
            "cobre apenas 11 canarios (S17C33), fora do universo base de 300 patches"),
        row("LOCAL_17C34", rel(LANDCOVER_17C34) if LANDCOVER_17C34.exists() else "not_available",
            "tabela_landcover_canario", "recife (canarios)", "not_available", "EPSG:4326", "por_patch", "csv",
            "urban_prop;vegetation_prop;water_prop;built_up_proxy_ndbi", "insuficiente",
            "cobre apenas canarios; nao cobre os 300 patches base"),
        row("LOCAL_NDBI_PROXY", rel(FEATURES_SRC), "proxy_espectral_ndbi", "recife;curitiba;petropolis (300 base)",
            "2022", "EPSG:4326", "por_patch", "csv", "impervious_proxy_via_ndbi", "insuficiente",
            "built_up_proxy_ndbi e proxy espectral derivado de NDBI; nao substitui landcover MapBiomas nem preenche exposed_soil/water/classe"),
        row("LOCAL_MAPBIOMAS_XLSX", "quarentena_local_only_nao_versionavel", "planilha_estado_bioma",
            "nacional (estado/bioma)", "colecao_10_1", "not_available", "1:250000", "xlsx",
            "cobertura_por_estado_bioma", "insuficiente",
            "nivel estado/bioma (DHN250), nao por patch; arquivo em quarentena local_only, nao versionavel e nao copiavel"),
        row("MAPBIOMAS_GEE", "pacote_gee/gee_mapbiomas_patch_landcover_19b.js", "colecao_raster_gee",
            "recife;curitiba;petropolis (300 base)", str(MAPBIOMAS_YEAR), "EPSG:4326", "30m", "earth_engine_asset",
            "MapBiomas_class_majority;MapBiomas_class_distribution;exposed_soil_prop;water_prop;impervious_proxy",
            "pacote_externo_necessario",
            "MapBiomas Colecao 9 via Earth Engine cobre os 300 patches; requer execucao externa autenticada pelo usuario"),
    ]


def fonte_local_utilizavel(inv: list[dict]) -> bool:
    return any(r["status_uso"] in {"utilizavel_direto", "utilizavel_com_extracao"} for r in inv)


# ---------------------------------------------------------------------------
# Auditoria de missingness territorial + matriz base
# ---------------------------------------------------------------------------
def load_19a_matrix() -> list[dict]:
    return read_csv(MAT_19A)


def load_19a_indice() -> dict:
    return {r["patch_id"]: r for r in read_csv(INDICE_19A)}


def auditoria_missingness_rows(matrix19a: list[dict]) -> list[dict]:
    out = []
    for r in matrix19a:
        faltas = [c for c in ("water_prop", "exposed_soil_prop", "impervious_proxy", "MapBiomas_class_majority") if r.get(c, NA) in (NA, "", "not_available")]
        out.append({
            "patch_id": r["patch_id"], "regiao": r["regiao"],
            "urban_prop": r.get("urban_prop", NA), "vegetation_prop": r.get("vegetation_prop", NA),
            "water_prop": r.get("water_prop", NA), "exposed_soil_prop": r.get("exposed_soil_prop", NA),
            "impervious_proxy": NA, "MapBiomas_class_majority": NA, "MapBiomas_distribution": NA,
            "status_missingness": "territorial_parcial_urban_veg_presentes_faltam_mapbiomas_solo_agua_impervious" if faltas else "territorial_completo",
            "acao_minima": "executar pacote MapBiomas/GEE para classe, solo exposto, agua e impervious" if faltas else "nenhuma",
        })
    return out


# ---------------------------------------------------------------------------
# Matriz territorial extraida (sem fonte local => tudo NA, extracao pendente)
# ---------------------------------------------------------------------------
def matriz_extraida_rows(matrix19a: list[dict], local_ok: bool) -> list[dict]:
    out = []
    for r in matrix19a:
        if local_ok:
            # Ramo nao acionado nesta execucao (sem fonte local utilizavel).
            continue
        out.append({
            "patch_id": r["patch_id"], "regiao": r["regiao"],
            "landcover_source": "not_available",
            "landcover_reference_year": "not_available",
            "urban_prop_19b": r.get("urban_prop", NA),
            "vegetation_prop_19b": r.get("vegetation_prop", NA),
            "water_prop_19b": NA,
            "exposed_soil_prop_19b": NA,
            "impervious_proxy_19b": NA,
            "MapBiomas_class_majority": NA,
            "MapBiomas_class_distribution": NA,
            "extraction_method": "pacote_gee_pendente",
            "source_path": "not_available",
            "confidence": "sem_fonte_local",
            "justificativa_tecnica": "sem fonte territorial local utilizavel para os 300 patches; features MapBiomas/solo/agua/impervious permanecem NA e a extracao fica pendente no pacote GEE; nao inventar valores",
        })
    return out


# ---------------------------------------------------------------------------
# Fila executavel de extracao
# ---------------------------------------------------------------------------
def fila_rows() -> list[dict]:
    manifest_out = "local_runs/susc_19b/mapbiomas_patch_landcover_19b_export.csv"
    def t(tid, feat, fonte, prio):
        return {
            "task_id": tid, "feature_alvo": feat, "patch_scope": "300 patches (recife;curitiba;petropolis)",
            "fonte_sugerida": fonte, "expected_input_path": rel(GEE_MANIFEST),
            "expected_output_path": manifest_out,
            "command_hint": "abrir gee_mapbiomas_patch_landcover_19b.js no editor do Earth Engine, ajustar YEAR/COLLECTION/PATCHES_ASSET e exportar CSV",
            "criterio_sucesso": "CSV com patch_id, classe majoritaria e proporcoes para os 300 patches",
            "prioridade": prio,
        }
    return [
        t("T19B_01", "MapBiomas_class_majority;MapBiomas_class_distribution", "MapBiomas Colecao 9 (GEE)", "P0"),
        t("T19B_02", "water_prop", "MapBiomas classe agua (GEE)", "P0"),
        t("T19B_03", "exposed_soil_prop", "MapBiomas classe solo exposto (GEE)", "P1"),
        t("T19B_04", "impervious_proxy", "MapBiomas classe area urbanizada / built (GEE)", "P1"),
        t("T19B_05", "landcover_source;landcover_reference_year", "metadados MapBiomas Colecao 9 ano 2022", "P2"),
    ]


# ---------------------------------------------------------------------------
# Matriz multimodal 19B atualizada (mantem 19A + colunas territoriais/estado)
# ---------------------------------------------------------------------------
EXTRA_19B_FIELDS = [
    "MapBiomas_class_majority", "MapBiomas_class_distribution", "impervious_proxy_territorial",
    "landcover_source", "landcover_reference_year", "territorial_fill_status",
    "score_oficial", "substituir_score_v6", "usar_em_treino",
]


def matriz_19b_rows(matrix19a: list[dict], local_ok: bool) -> tuple[list[dict], list[str]]:
    base_fields = list(matrix19a[0].keys()) if matrix19a else []
    fields = base_fields + [f for f in EXTRA_19B_FIELDS if f not in base_fields]
    out = []
    for r in matrix19a:
        row = dict(r)
        # Sem fonte local utilizavel: territorial-alvo permanece NA (preserva 19A).
        row["MapBiomas_class_majority"] = NA
        row["MapBiomas_class_distribution"] = NA
        row["impervious_proxy_territorial"] = NA
        row["landcover_source"] = "not_available"
        row["landcover_reference_year"] = "not_available"
        row["territorial_fill_status"] = "pendente_pacote_gee" if not local_ok else "preenchido_com_fonte_local"
        row["score_oficial"] = "false"
        row["substituir_score_v6"] = "false"
        row["usar_em_treino"] = "false"
        out.append(row)
    return out, fields


# ---------------------------------------------------------------------------
# Cobertura pos-19B e comparacao 19A/19B
# ---------------------------------------------------------------------------
def _cov_territorial(row_extraida: dict) -> float:
    presentes = set(TERRITORIAL_PRESENTE_19A)
    for feat, col in (("water_prop", "water_prop_19b"), ("exposed_soil_prop", "exposed_soil_prop_19b"),
                      ("impervious_proxy", "impervious_proxy_19b"), ("MapBiomas_class_majority", "MapBiomas_class_majority")):
        if row_extraida.get(col, NA) not in (NA, "", "not_available"):
            presentes.add(feat)
    return len(presentes) / len(TERRITORIAL_EXPECTED)


def cobertura_pos_rows(matrix19a: list[dict], extraida: list[dict], indice19a: dict) -> list[dict]:
    ext_by = {r["patch_id"]: r for r in extraida}
    out = []
    for r in matrix19a:
        pid = r["patch_id"]
        cov19a = indice19a.get(pid, {})
        terr_before = float(cov19a.get("coverage_territorial", 0.3333))
        ext = ext_by.get(pid)
        terr_after = _cov_territorial(ext) if ext else terr_before
        # cobertura total recalculada trocando somente o componente territorial (6 familias)
        total_before = float(cov19a.get("coverage_total", 0.0))
        total_after = total_before + (terr_after - terr_before) / 6.0
        out.append({
            "patch_id": pid, "regiao": r["regiao"],
            "coverage_territorial_19a": f"{terr_before:.4f}",
            "coverage_territorial_19b": f"{terr_after:.4f}",
            "coverage_total_19a": f"{total_before:.4f}",
            "coverage_total_19b": f"{total_after:.4f}",
            "territorial_delta": f"{terr_after - terr_before:.4f}",
            "lacuna_remanescente": "MapBiomas;exposed_soil_prop;water_prop;impervious_proxy" if terr_after < 1.0 else "nenhuma",
        })
    return out


def comparacao_rows(cobertura_pos: list[dict]) -> list[dict]:
    regs = {}
    for r in cobertura_pos:
        reg = r["regiao"]
        regs.setdefault(reg, []).append(r)
    out = []
    for reg in sorted(regs):
        rows = regs[reg]
        n = len(rows)
        tb = sum(float(x["coverage_territorial_19a"]) for x in rows) / n
        ta = sum(float(x["coverage_territorial_19b"]) for x in rows) / n
        cb = sum(float(x["coverage_total_19a"]) for x in rows) / n
        ca = sum(float(x["coverage_total_19b"]) for x in rows) / n
        out.append({
            "regiao": reg, "n_patches": str(n),
            "coverage_territorial_19a": f"{tb:.4f}", "coverage_territorial_19b": f"{ta:.4f}",
            "coverage_total_19a": f"{cb:.4f}", "coverage_total_19b": f"{ca:.4f}",
            "territorial_delta": f"{ta - tb:.4f}",
            "features_que_destravam": "MapBiomas_class_majority;exposed_soil_prop;water_prop;impervious_proxy",
        })
    return out


# ---------------------------------------------------------------------------
# Resumos
# ---------------------------------------------------------------------------
def resumo_regiao_rows(comparacao: list[dict], auditoria: list[dict]) -> list[dict]:
    faltas_by_reg = {}
    for r in auditoria:
        faltas_by_reg.setdefault(r["regiao"], 0)
        if "faltam" in r["status_missingness"]:
            faltas_by_reg[r["regiao"]] += 1
    out = []
    for c in comparacao:
        reg = c["regiao"]
        out.append({
            "regiao": reg, "n_patches": c["n_patches"],
            "coverage_territorial_19a": c["coverage_territorial_19a"],
            "coverage_territorial_19b": c["coverage_territorial_19b"],
            "territorial_delta": c["territorial_delta"],
            "patches_com_lacuna_territorial": str(faltas_by_reg.get(reg, 0)),
            "acao_minima": "executar pacote MapBiomas/GEE",
        })
    return out


def resumo_feature_rows(matrix19a: list[dict], local_ok: bool) -> list[dict]:
    n = len(matrix19a)
    def r(feature, presente_19a, presente_19b, fonte, status):
        return {
            "feature": feature, "n_patches": str(n),
            "presentes_19a": str(presente_19a), "presentes_19b": str(presente_19b),
            "fonte": fonte, "status": status,
        }
    pres_urban = sum(1 for x in matrix19a if x.get("urban_prop", NA) not in (NA, "", "not_available"))
    pres_veg = sum(1 for x in matrix19a if x.get("vegetation_prop", NA) not in (NA, "", "not_available"))
    return [
        r("urban_prop", pres_urban, pres_urban, rel(FEATURES_SRC), "presente"),
        r("vegetation_prop", pres_veg, pres_veg, rel(FEATURES_SRC), "presente"),
        r("water_prop", 0, 0, "not_available", "pendente_pacote_gee"),
        r("exposed_soil_prop", 0, 0, "not_available", "pendente_pacote_gee"),
        r("impervious_proxy", 0, 0, "not_available", "pendente_pacote_gee"),
        r("MapBiomas_class_majority", 0, 0, "not_available", "pendente_pacote_gee"),
        r("MapBiomas_class_distribution", 0, 0, "not_available", "pendente_pacote_gee"),
    ]


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------
def gate_19b_rows(summary: dict) -> list[dict]:
    def g(criterio, valor, limiar, passou, obs):
        return {"criterio": criterio, "valor_observado": str(valor), "limiar": limiar, "passou": "true" if passou else "false", "status": summary["status_19b"], "observacao": obs}
    return [
        g("fonte_local_utilizavel", str(summary["fonte_local_utilizavel"]).lower(), "documentado", True, "auditoria real das fontes territoriais locais"),
        g("pacote_gee_pronto", str(summary["pacote_gee_pronto"]).lower(), "true", summary["pacote_gee_pronto"], "script, manifesto e schema do MapBiomas/GEE"),
        g("fila_executavel", summary["fila_tasks"], ">=1", summary["fila_tasks"] >= 1, "tarefas de extracao territorial"),
        g("territorial_nao_inventado", "true", "true", True, "features territoriais faltantes permanecem NA"),
        g("score_v6_intacto", str(not summary["score_v6_changed"]).lower(), "true", not summary["score_v6_changed"], "score_v6 nao alterado"),
        g("ground_truth_zero", summary["ground_truth_true_count"], "0", summary["ground_truth_true_count"] == 0, "sem ground truth"),
    ]


def gate_19c_rows(summary: dict) -> list[dict]:
    def g(criterio, valor, limiar, passou, obs):
        return {"criterio": criterio, "valor_observado": str(valor), "limiar": limiar, "passou": "true" if passou else "false", "status_19c_alvo": "19C_AVALIACAO_OBSERVACIONAL_REVIEW_ONLY", "observacao": obs}
    return [
        g("matriz_multimodal_disponivel", summary["total_patches"], ">0", summary["total_patches"] > 0, "matriz 19B por patch"),
        g("cobertura_documentada", "true", "true", True, "cobertura antes/depois registrada"),
        g("territorial_em_fila", summary["fila_tasks"], ">=1", summary["fila_tasks"] >= 1, "territorial encaminhado, nao bloqueia avaliacao observacional review-only"),
        g("coverage_nao_e_suscetibilidade", "true", "true", True, "indice de cobertura nao e score de suscetibilidade"),
    ]


# ---------------------------------------------------------------------------
# Pacote GEE / MapBiomas
# ---------------------------------------------------------------------------
def gee_js_text() -> str:
    return f"""// SUSC-19B - Extracao MapBiomas por patch (Earth Engine).
// Somente revisao. Nao contem credenciais. Nao baixa raster pesado.
// Ajuste os parametros editaveis abaixo e exporte o CSV leve.

// ===== PARAMETROS EDITAVEIS =====
var COLLECTION_ASSET = '{MAPBIOMAS_COLLECTION}';
var YEAR = {MAPBIOMAS_YEAR};                 // ano de referencia MapBiomas
var SCALE = 30;                              // resolucao MapBiomas (m)
var PATCHES_ASSET = 'users/SEU_USUARIO/susc_patches_300';  // FeatureCollection com os 300 patches (patch_id, geometria)
var EXPORT_NAME = 'mapbiomas_patch_landcover_19b_export';

// Classes MapBiomas agregadas (editavel conforme legenda oficial):
var CLASSES_AGUA = [33, 31];                 // rio/lago/oceano e aquicultura
var CLASSES_SOLO_EXPOSTO = [23, 25];         // praia/duna e area nao vegetada
var CLASSES_URBANO = [24];                   // area urbanizada
var CLASSES_VEGETACAO = [1, 3, 4, 5, 6, 49, 10, 11, 12]; // formacoes naturais

// ===== CARGA =====
var patches = ee.FeatureCollection(PATCHES_ASSET);
var mapbiomas = ee.Image(COLLECTION_ASSET).select('classification_' + YEAR);

// ===== PROPORCAO DE CLASSES POR PATCH =====
function proporcoes(feature) {{
  var hist = mapbiomas.reduceRegion({{
    reducer: ee.Reducer.frequencyHistogram(),
    geometry: feature.geometry(),
    scale: SCALE,
    maxPixels: 1e9
  }}).get('classification_' + YEAR);
  hist = ee.Dictionary(hist);
  var total = ee.Number(hist.values().reduce(ee.Reducer.sum()));
  function prop(classes) {{
    var soma = ee.List(classes).iterate(function(c, acc) {{
      c = ee.Number(c).format();
      return ee.Number(acc).add(ee.Number(hist.get(c, 0)));
    }}, 0);
    return ee.Number(soma).divide(total);
  }}
  // classe majoritaria
  var keys = hist.keys();
  var vals = ee.Array(hist.values());
  var idxMax = vals.argmax().get(0);
  var classeMajoritaria = ee.Number.parse(keys.get(idxMax));
  return feature.set({{
    'mapbiomas_year': YEAR,
    'mapbiomas_class_majority': classeMajoritaria,
    'water_prop': prop(CLASSES_AGUA),
    'exposed_soil_prop': prop(CLASSES_SOLO_EXPOSTO),
    'impervious_proxy': prop(CLASSES_URBANO),
    'vegetation_prop_mapbiomas': prop(CLASSES_VEGETACAO),
    'class_distribution_json': hist,
    'pixel_count': total,
    'review_only': true
  }});
}}

var resultado = patches.map(proporcoes);

// ===== EXPORT CSV LEVE (sem raster) =====
Export.table.toDrive({{
  collection: resultado,
  description: EXPORT_NAME,
  fileFormat: 'CSV',
  selectors: ['patch_id', 'mapbiomas_year', 'mapbiomas_class_majority', 'water_prop',
              'exposed_soil_prop', 'impervious_proxy', 'vegetation_prop_mapbiomas',
              'class_distribution_json', 'pixel_count', 'review_only']
}});
"""


def gee_md_text() -> str:
    return f"""# Pacote MapBiomas/GEE - SUSC-19B

Extracao das features territoriais faltantes para os 300 patches, somente revisao.

## O que o script faz

1. Carrega os 300 patches como `FeatureCollection` (asset editavel `PATCHES_ASSET`).
2. Seleciona a banda `classification_{MAPBIOMAS_YEAR}` da colecao MapBiomas.
3. Calcula a proporcao de classes por patch com `frequencyHistogram`.
4. Deriva `mapbiomas_class_majority`, `water_prop`, `exposed_soil_prop`,
   `impervious_proxy` e a distribuicao de classes.
5. Exporta um CSV leve (sem raster) com uma linha por patch.

## Parametros editaveis

- `COLLECTION_ASSET`: colecao MapBiomas (padrao Colecao 9).
- `YEAR`: ano de referencia (padrao {MAPBIOMAS_YEAR}).
- `SCALE`: resolucao em metros (padrao 30).
- `PATCHES_ASSET`: sua `FeatureCollection` com `patch_id` e geometria dos 300 patches.
- Listas de classes (agua, solo exposto, urbano, vegetacao) conforme a legenda oficial.

## Como preparar os patches

Use `gee_export_manifest_19b.csv` (patch_id + bbox) para construir a
`FeatureCollection` no Earth Engine ou para gerar um asset via upload.

## Restricoes

- Nao contem credenciais; a autenticacao e feita pelo usuario no Earth Engine.
- Nao baixa raster pesado; exporta apenas CSV leve.
- Resultado e somente revisao: nao e ground truth, nao habilita treino e nao cria score_v7.
"""


def gee_manifest_rows(matrix19a: list[dict], features_by_id: dict) -> list[dict]:
    out = []
    for r in matrix19a:
        pid = r["patch_id"]
        f = features_by_id.get(pid, {})
        out.append({
            "patch_id": pid, "regiao": r["regiao"],
            "xmin": f.get("xmin", NA), "ymin": f.get("ymin", NA),
            "xmax": f.get("xmax", NA), "ymax": f.get("ymax", NA),
            "crs": "EPSG:4326",
            "mapbiomas_collection": MAPBIOMAS_COLLECTION,
            "mapbiomas_year": str(MAPBIOMAS_YEAR),
            "expected_output": "mapbiomas_patch_landcover_19b_export.csv",
        })
    return out


def gee_schema_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SUSC-19B saida esperada do pacote MapBiomas/GEE",
        "type": "object",
        "description": "CSV leve, uma linha por patch, somente revisao.",
        "required": ["patch_id", "mapbiomas_year", "mapbiomas_class_majority"],
        "properties": {
            "patch_id": {"type": "string"},
            "mapbiomas_year": {"const": MAPBIOMAS_YEAR},
            "mapbiomas_class_majority": {"type": "integer"},
            "water_prop": {"type": "number", "minimum": 0, "maximum": 1},
            "exposed_soil_prop": {"type": "number", "minimum": 0, "maximum": 1},
            "impervious_proxy": {"type": "number", "minimum": 0, "maximum": 1},
            "class_distribution_json": {"type": "string"},
            "pixel_count": {"type": "number"},
            "review_only": {"const": True},
        },
        "guardrails": {"ground_truth": False, "eligible_for_training": False, "score_v7_allowed": False},
    }


def write_pacote_gee(matrix19a: list[dict], features_by_id: dict) -> None:
    ensure_dir(PACOTE_GEE)
    write_markdown(GEE_JS, gee_js_text())
    write_markdown(GEE_MD, gee_md_text())
    write_csv(GEE_MANIFEST, gee_manifest_rows(matrix19a, features_by_id))
    write_json(GEE_SCHEMA, gee_schema_obj())


# ---------------------------------------------------------------------------
# Cartoes / relatorio / schema / preflight / summary
# ---------------------------------------------------------------------------
def _card_text(regiao: str, resumo: dict) -> str:
    return f"""# Cartao territorial - {CIDADE[regiao]}

## Lacuna 19A
Territorial parcial: presentes urban_prop e vegetation_prop; faltam water_prop,
exposed_soil_prop, impervious_proxy e MapBiomas_class_majority.

## Fontes encontradas
Nenhuma fonte local cobre os alvos territoriais dos patches base. As tabelas de
landcover locais reembrulham urban/vegetation ou cobrem apenas canarios; a
planilha MapBiomas e por estado/bioma e esta em quarentena.

## Preenchimento 19B
Sem preenchimento local. Cobertura territorial de {resumo['coverage_territorial_19a']}
mantida ({resumo['territorial_delta']} de variacao). Extracao encaminhada ao pacote MapBiomas/GEE.

## Lacunas restantes
MapBiomas_class_majority, exposed_soil_prop, water_prop e impervious_proxy.

## Acao minima
{resumo['acao_minima']} para os {resumo['n_patches']} patches da regiao.

## Por que nao e ground truth
Features territoriais de suscetibilidade escalavel, sem geometria de ocorrencia.

## Por que nao e treino
eligible_for_training e falso em toda a matriz.

## Por que nao cria score_v7
A sprint organiza cobertura territorial; nao gera score oficial. score_v6 intacto.
"""


def write_cards(resumo_reg: list[dict]) -> None:
    ensure_dir(CARDS)
    by_reg = {r["regiao"]: r for r in resumo_reg}
    for regiao in ("recife", "curitiba", "petropolis"):
        if regiao in by_reg:
            write_markdown(CARDS / f"cartao_{regiao}.md", _card_text(regiao, by_reg[regiao]))


def schema_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SUSC-19B auditoria e preenchimento de lacunas territoriais",
        "type": "object",
        "description": "Auditoria territorial e pacote MapBiomas/GEE. Somente revisao.",
        "guardrails": {
            "ground_truth": {"const": "false"},
            "eligible_for_training": {"const": "false"},
            "score_v7_allowed": {"const": "false"},
            "score_oficial": {"const": "false"},
            "substituir_score_v6": {"const": "false"},
            "usar_em_treino": {"const": "false"},
        },
        "territorial_targets": TERRITORIAL_TARGETS,
        "status_19b_allowed": sorted(STATUS_19B_ALLOWED),
        "status_uso_allowed": sorted(STATUS_USO_ALLOWED),
        "status_17b_mestre": STATUS_17B_MESTRE,
        "mapbiomas_collection": MAPBIOMAS_COLLECTION,
        "mapbiomas_year": MAPBIOMAS_YEAR,
    }


def preflight_obj(matrix19a: list[dict], inv: list[dict]) -> dict:
    fontes = {r["fonte_id"]: r["status_uso"] for r in inv}
    status_full = _run_git(["status", "--short"]).splitlines()
    return {
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "staged_count": len(_run_git(["diff", "--cached", "--name-only"]).splitlines()),
        "status_short_count": len(status_full),
        "entradas_lidas": {rel(p): p.exists() for p in REQUIRED_INPUTS},
        "total_patches_19a": len(matrix19a),
        "fontes_territoriais_locais": fontes,
        "fonte_local_utilizavel": fonte_local_utilizavel(inv),
        "score_v6_path": rel(SCORE_V6),
        "score_v6_changed": _score_v6_changed(),
        "score_v7_created": SCORE_V7.exists(),
        "outputs_ignorados_pelo_git": sorted(git_ignored([OUT])),
        "summary_versionavel": rel(SUMMARY) not in git_ignored([SUMMARY]),
    }


def summary_obj(matrix19a: list[dict], inv: list[dict], extraida: list[dict], comparacao: list[dict], fila: list[dict]) -> dict:
    local_ok = fonte_local_utilizavel(inv)
    pacote_pronto = all(p.exists() for p in (GEE_JS, GEE_MD, GEE_MANIFEST, GEE_SCHEMA)) if not local_ok else True
    if local_ok:
        status = "19B_TERRITORIAL_PREENCHIDO_COM_FONTE_LOCAL"
    elif pacote_pronto:
        status = "19B_PACOTE_MAPBIOMAS_GEE_PRONTO"
    else:
        status = "19B_BLOQUEADO_POR_FONTE_TERRITORIAL"
    terr_before = sum(float(c["coverage_territorial_19a"]) * int(c["n_patches"]) for c in comparacao) / max(1, len(matrix19a))
    terr_after = sum(float(c["coverage_territorial_19b"]) * int(c["n_patches"]) for c in comparacao) / max(1, len(matrix19a))
    return {
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "staged_count": len(_run_git(["diff", "--cached", "--name-only"]).splitlines()),
        "total_patches": len(matrix19a),
        "regioes": sorted({r["regiao"] for r in matrix19a}),
        "fonte_local_utilizavel": local_ok,
        "pacote_gee_pronto": pacote_pronto,
        "fila_tasks": len(fila),
        "territorial_features_preenchidas_localmente": 0,
        "coverage_territorial_medio_19a": round(terr_before, 4),
        "coverage_territorial_medio_19b": round(terr_after, 4),
        "coverage_territorial_delta": round(terr_after - terr_before, 4),
        "lacunas_territoriais_remanescentes": ["MapBiomas_class_majority", "MapBiomas_class_distribution", "exposed_soil_prop", "water_prop", "impervious_proxy"],
        "coverage_is_susceptibility_score": False,
        "status_19b": status,
        "status_17b_mestre": STATUS_17B_MESTRE,
        "marco_17b_criado": False,
        "benchmark_17b_criado": False,
        "ground_truth_true_count": 0,
        "eligible_for_training_true_count": 0,
        "score_v7_allowed_true_count": 0,
        "score_v6_changed": _score_v6_changed(),
        "score_v7_created": SCORE_V7.exists(),
        "review_only": True,
        "proximo_marco": "SUSC-19C",
    }


def report_text(summary: dict, comparacao: list[dict], inv: list[dict]) -> str:
    linhas_cmp = "\n".join(
        f"| {c['regiao']} | {c['n_patches']} | {c['coverage_territorial_19a']} | {c['coverage_territorial_19b']} | {c['territorial_delta']} |"
        for c in comparacao
    )
    linhas_fontes = "\n".join(
        f"| {r['fonte_id']} | {r['status_uso']} | {r['motivo']} |" for r in inv
    )
    return f"""# SUSC-19B - Auditoria e preenchimento de lacunas territoriais

## Estado herdado do 19A

O 19A consolidou 300 patches com fisico, espectral e chuva completos, mas
territorial parcial (apenas urban_prop e vegetation_prop). Estado do 17B:
`{summary['status_17b_mestre']}` (17B nao criado).

## Lacuna territorial real

Faltam, nos 300 patches base: MapBiomas_class_majority, MapBiomas_class_distribution,
exposed_soil_prop, water_prop e impervious_proxy. A cobertura territorial media
permanece em {summary['coverage_territorial_medio_19a']} (2 de 6 features esperadas).

## Fontes encontradas

| Fonte | Status | Motivo |
| --- | --- | --- |
{linhas_fontes}

## Houve preenchimento local?

Nao. Nenhuma fonte local cobre os alvos territoriais dos 300 patches base: as
tabelas de landcover reembrulham urban/vegetation ou cobrem apenas canarios, e a
planilha MapBiomas e por estado/bioma e esta em quarentena. Nenhum valor foi
inventado.

## Pacote MapBiomas/GEE

Como nao ha fonte local utilizavel, foi criado um pacote executavel em
`pacote_gee/`: script Earth Engine, documentacao, manifesto de export e schema de
saida esperada. O pacote nao contem credenciais e nao baixa raster pesado.

## Impacto na cobertura multimodal

| Regiao | Patches | Territorial 19A | Territorial 19B | Delta |
| --- | --- | --- | --- | --- |
{linhas_cmp}

A cobertura territorial nao muda nesta sprint porque o preenchimento depende da
execucao externa do pacote MapBiomas/GEE. A fila `fila_extracao_territorial_19b.csv`
lista as tarefas executaveis.

## Lacunas restantes

MapBiomas_class_majority, MapBiomas_class_distribution, exposed_soil_prop,
water_prop e impervious_proxy, em todas as regioes.

## Por que nao e ground truth, nem treino, nem score_v7

As features territoriais sao de suscetibilidade escalavel, sem geometria de
ocorrencia. eligible_for_training e falso; nenhum score oficial e criado; o
coverage mede completude, nao suscetibilidade; o score_v6 permanece intacto.

## Proximo marco recomendado

**SUSC-19C - Avaliacao observacional review-only**: comparar Recife e Curitiba
tecnica com score e features, apos encaminhar a extracao territorial via MapBiomas/GEE.
"""


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_all() -> dict:
    require_inputs()
    ensure_dir(OUT)
    ensure_dir(REPORTS)
    ensure_dir(SCHEMAS_DIR)

    matrix19a = load_19a_matrix()
    indice19a = load_19a_indice()
    features_by_id = {r["patch_id"]: r for r in read_csv(FEATURES_SRC)}
    inv = inventario_rows()
    local_ok = fonte_local_utilizavel(inv)

    auditoria = auditoria_missingness_rows(matrix19a)
    extraida = matriz_extraida_rows(matrix19a, local_ok)
    fila = fila_rows()
    mat19b, mat19b_fields = matriz_19b_rows(matrix19a, local_ok)
    cobertura_pos = cobertura_pos_rows(matrix19a, extraida, indice19a)
    comparacao = comparacao_rows(cobertura_pos)
    resumo_reg = resumo_regiao_rows(comparacao, auditoria)
    resumo_feat = resumo_feature_rows(matrix19a, local_ok)

    # pacote GEE (necessario quando nao ha fonte local utilizavel)
    write_pacote_gee(matrix19a, features_by_id)

    summary = summary_obj(matrix19a, inv, extraida, comparacao, fila)

    write_csv(AUDITORIA_MISS, auditoria)
    write_csv(INVENTARIO, inventario_rows())
    write_csv(MAT_EXTRAIDA, extraida, ["patch_id", "regiao", "landcover_source", "landcover_reference_year", "urban_prop_19b", "vegetation_prop_19b", "water_prop_19b", "exposed_soil_prop_19b", "impervious_proxy_19b", "MapBiomas_class_majority", "MapBiomas_class_distribution", "extraction_method", "source_path", "confidence", "justificativa_tecnica"])
    write_csv(FILA, fila)
    write_csv(MAT_19B, mat19b, mat19b_fields)
    write_csv(COBERTURA_POS, cobertura_pos)
    write_csv(COMPARACAO, comparacao)
    write_csv(GATE_19B, gate_19b_rows(summary))
    write_csv(GATE_19C, gate_19c_rows(summary))
    write_csv(RESUMO_REGIAO, resumo_reg)
    write_csv(RESUMO_FEATURE, resumo_feat)

    write_json(SCHEMA, schema_obj())
    write_json(PREFLIGHT, preflight_obj(matrix19a, inv))
    write_json(SUMMARY, summary)
    write_cards(resumo_reg)
    write_markdown(REPORT, report_text(summary, comparacao, inv))
    return summary


# ---------------------------------------------------------------------------
# Validacao
# ---------------------------------------------------------------------------
def _public_paths() -> list[Path]:
    paths = []
    if OUT.exists():
        paths.extend(p for p in OUT.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".json", ".md", ".js"})
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
    if summary["status_19b"] not in STATUS_19B_ALLOWED:
        errors.append(f"status_19b_fora_enum:{summary['status_19b']}")

    # matriz final 19B
    final = read_csv(MAT_19B)
    if not final:
        errors.append("matriz_final_vazia")
    ids = [r["patch_id"] for r in final]
    if len(ids) != len(set(ids)):
        errors.append("patch_id_duplicado")
    for r in final:
        for field in ("ground_truth", "eligible_for_training", "score_v7_allowed", "score_oficial", "substituir_score_v6", "usar_em_treino"):
            if r.get(field) not in ("false", None) and field in r and r.get(field) != "false":
                errors.append(f"final:{r['patch_id']}:{field}_nao_false")
        if not r.get("justificativa_tecnica", "").strip():
            errors.append(f"final:{r['patch_id']}:justificativa_vazia")

    # auditoria le os 300 patches
    aud = read_csv(AUDITORIA_MISS)
    if len(aud) != summary["total_patches"]:
        errors.append(f"auditoria_missingness_incompleta:{len(aud)}!={summary['total_patches']}")

    # extracao: feature territorial presente exige fonte real (sem invencao)
    for r in read_csv(MAT_EXTRAIDA):
        src = r.get("landcover_source", "not_available")
        for col in ("water_prop_19b", "exposed_soil_prop_19b", "impervious_proxy_19b", "MapBiomas_class_majority", "MapBiomas_class_distribution"):
            val = r.get(col, NA)
            if val not in (NA, "", "not_available") and src in ("not_available", "", "ausente"):
                errors.append(f"extraida:{r.get('patch_id')}:{col}_presente_sem_fonte")

    # inventario: status_uso no enum
    for r in read_csv(INVENTARIO):
        if r.get("status_uso") not in STATUS_USO_ALLOWED:
            errors.append(f"inventario:status_uso_fora_enum:{r.get('status_uso')}")

    # pacote GEE obrigatorio quando nao ha fonte local
    if not summary["fonte_local_utilizavel"]:
        for p in (GEE_JS, GEE_MD, GEE_MANIFEST, GEE_SCHEMA):
            if not p.exists():
                errors.append(f"pacote_gee_ausente:{rel(p)}")

    # coverage nunca chamado de score de suscetibilidade (comparacao/cobertura)
    for path in (COBERTURA_POS, COMPARACAO):
        header = read_csv(path)
        for r in header:
            for k in r:
                if "suscetibilidade" in k.lower():
                    errors.append(f"{rel(path)}:coluna_suscetibilidade_indevida:{k}")

    # SAR pos-evento nao pode virar feature territorial
    for r in read_csv(MAT_EXTRAIDA):
        blob = " ".join(str(v).lower() for v in r.values())
        if "patch_stats" in blob or "sar" in blob and "footprint" in blob:
            errors.append(f"extraida:{r.get('patch_id')}:sar_como_feature_territorial")

    # gate status no enum
    for r in read_csv(GATE_19B):
        if r.get("status") not in STATUS_19B_ALLOWED:
            errors.append(f"gate_19b:status_fora_enum:{r.get('status')}")

    # output 19B nao pode ser ignorado pelo git
    if rel(SUMMARY) in git_ignored([SUMMARY, MAT_19B, GEE_JS]):
        errors.append("output_19b_ignorado_pelo_git")

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
        "19B auditoria territorial validada: "
        f"patches={summary['total_patches']} fonte_local={str(summary['fonte_local_utilizavel']).lower()} "
        f"pacote_gee={str(summary['pacote_gee_pronto']).lower()} fila={summary['fila_tasks']} "
        f"terr_19a={summary['coverage_territorial_medio_19a']} terr_19b={summary['coverage_territorial_medio_19b']} "
        f"status19B={summary['status_19b']} score_v6_changed={str(summary['score_v6_changed']).lower()}"
    )
    return 0


def run_all() -> int:
    summary = build_all()
    print(
        "19B auditoria territorial gerada: "
        f"patches={summary['total_patches']} fonte_local={str(summary['fonte_local_utilizavel']).lower()} "
        f"pacote_gee={str(summary['pacote_gee_pronto']).lower()} status19B={summary['status_19b']}"
    )
    return 0
