"""SUSC-GT09 - Esteira Canaria de Revisao Visual, Pareamento Pre/Pos e Pre-Execucao SAR.

Marco composto que transforma os 5 canarios Recife em uma esteira completa de
pre-execucao observacional, sem executar SAR nem baixar raster. Consolida sete
subsistemas:
  A) seletor do melhor par Sentinel-1 pre/pos por canario (ranking);
  B) dossie tecnico-visual por canario;
  C) inventario de ativos visuais/geometricos locais JA versionados (sem gerar imagem);
  D) contrato de renderizacao futura (sem execucao);
  E) dry-run logico do pipeline SAR futuro (sem dados raster);
  F) handoff objetivo para QA humano (le o GT07);
  G) readiness gate para avaliacao observacional review-only;
  + guardrails de leakage temporal (footprint pos-evento e referencia, nunca feature).

E um pacote offline e review-only: NAO usa rede, NAO consulta STAC/API remota, NAO baixa
Sentinel/raster, NAO abre nem gera imagem, NAO roda SAR/GEE, NAO cria footprint, NAO cria
geometria real, NAO executa QA, NAO marca accepted/rejected, NAO altera o score_v6, NAO
cria score_v7, NAO treina modelo e NAO promove nada a positive_strong nem a ground truth.

Ver [[susc_gt08_sentinel1_readiness]] e [[susc_gt07_human_qa_canary_protocol]].
Identificadores tecnicos em ingles sao mantidos por compatibilidade; o relatorio publico
explica cada campo em portugues.
"""

from __future__ import annotations

import csv
import datetime as _dt
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
import susc_gt08_sentinel1_readiness_common as gt08  # noqa: E402

gt07 = gt08.gt07
gt02 = gt08.gt02

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS_DIR = ROOT / "schemas" / "suscetibilidade"

GT08_TARGETS = OUT / "susc_gt08_alvos_sentinel1_canarios.csv"
GT08_AOI = OUT / "susc_gt08_aoi_resolvido_canarios.csv"
GT08_SCENES = OUT / "susc_gt08_cenas_candidatas_locais_sentinel1.csv"
GT08_PAIRING = OUT / "susc_gt08_contrato_pareamento_pre_pos.csv"
GT08_MANIFEST = OUT / "susc_gt08_manifesto.json"
GT07_QUEUE = OUT / "susc_gt07_fila_qa_canarios.csv"
GT07_CHECKLIST = OUT / "susc_gt07_checklist_qa_por_canario.csv"

REQUIRED_INPUTS = [GT08_TARGETS, GT08_AOI, GT08_SCENES, GT08_PAIRING, GT08_MANIFEST,
                   GT07_QUEUE, GT07_CHECKLIST]

REPORT = OUT / "SUSC_GT09_ESTEIRA_CANARIA_REVISAO_VISUAL_PRE_EXECUCAO_SAR.md"
PAIRS_CSV = OUT / "susc_gt09_pares_pre_pos_canarios.csv"
ALT_PAIRS_CSV = OUT / "susc_gt09_pares_alternativos_canarios.csv"
DOSSIER_CSV = OUT / "susc_gt09_dossie_canarios.csv"
ASSETS_CSV = OUT / "susc_gt09_inventario_ativos_locais.csv"
RENDER_CSV = OUT / "susc_gt09_contrato_renderizacao_futura.csv"
DRYRUN_CSV = OUT / "susc_gt09_dry_run_pipeline_sar_futuro.csv"
HANDOFF_CSV = OUT / "susc_gt09_handoff_qa_canarios.csv"
GATE_CSV = OUT / "susc_gt09_readiness_gate_observacional.csv"
LEAKAGE_CSV = OUT / "susc_gt09_guardrails_leakage_evidencia.csv"
BLOCKERS_CSV = OUT / "susc_gt09_bloqueios_esteira_canaria.csv"
SUMMARY_CSV = OUT / "susc_gt09_resumo_esteira_canaria.csv"
MANIFEST_JSON = OUT / "susc_gt09_manifesto.json"

SCHEMA_PAIR = SCHEMAS_DIR / "susc_gt09_pre_post_pair.schema.json"
SCHEMA_DOSSIER = SCHEMAS_DIR / "susc_gt09_canary_dossier.schema.json"
SCHEMA_ASSET = SCHEMAS_DIR / "susc_gt09_visual_asset_inventory.schema.json"
SCHEMA_RENDER = SCHEMAS_DIR / "susc_gt09_future_render_contract.schema.json"
SCHEMA_DRYRUN = SCHEMAS_DIR / "susc_gt09_sar_dry_run_step.schema.json"
SCHEMA_GATE = SCHEMAS_DIR / "susc_gt09_observational_readiness_gate.schema.json"
SCHEMA_SUMMARY = SCHEMAS_DIR / "susc_gt09_summary.schema.json"

PUBLIC_ARTIFACTS = [
    REPORT, PAIRS_CSV, ALT_PAIRS_CSV, DOSSIER_CSV, ASSETS_CSV, RENDER_CSV, DRYRUN_CSV,
    HANDOFF_CSV, GATE_CSV, LEAKAGE_CSV, BLOCKERS_CSV, SUMMARY_CSV,
]
REQUIRED_OUTPUTS = PUBLIC_ARTIFACTS + [
    MANIFEST_JSON, SCHEMA_PAIR, SCHEMA_DOSSIER, SCHEMA_ASSET, SCHEMA_RENDER,
    SCHEMA_DRYRUN, SCHEMA_GATE, SCHEMA_SUMMARY,
]

MARCO = "SUSC-GT09"
TITULO_PUBLICO = "Esteira Canária de Revisão Visual, Pareamento Pré/Pós e Pré-Execução SAR"
TITULO_H1_TOKEN = "Esteira Canária de Revisão Visual"

PUBLIC_FORBIDDEN_RE = gt08.PUBLIC_FORBIDDEN_RE
OPERATIONAL_FORECAST_RE = gt08.OPERATIONAL_FORECAST_RE

GLOBAL_GUARDRAILS = {
    "eligible_for_training": "false", "eligible_for_ground_truth": "false",
    "trainable": "false", "score_v7_candidate": "false", "review_only": "true",
}

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
ASSET_CLASSES = [
    "ativo_visual_existente", "ativo_geometrico_existente",
    "apenas_metadado", "nenhum_ativo_visual",
]
GATE_STATES = [
    "pronto_para_renderizacao_visual_futura",
    "pronto_para_download_controlado_futuro",
    "pronto_para_qa_visual_futuro",
    "bloqueado_por_par_pre_pos",
    "bloqueado_por_aoi",
    "bloqueado_por_ativo_visual",
    "bloqueado_por_politica_review_only",
]
HANDOFF_STATUS = [
    "desbloqueavel_apos_visualizacao", "depende_de_footprint",
    "depende_de_qa_humano", "ja_disponivel",
]

# Paineis do contrato de renderizacao futura (subsistema D).
RENDER_PANELS = [
    ("painel_pre_sar", "painel da cena SAR pre-evento"),
    ("painel_pos_sar", "painel da cena SAR pos-evento"),
    ("diferenca_pre_pos", "painel de diferenca pre/pos"),
    ("overlay_aoi_patch", "overlay do AOI/patch"),
    ("overlay_agua_permanente_futura", "overlay de agua permanente (futuro)"),
    ("overlay_hand_slope_futuro", "overlay HAND/slope (futuro)"),
    ("overlay_exclusoes_urbanas_futuras", "overlay de exclusoes urbanas (futuro)"),
    ("mascara_candidata_futura", "mascara candidata (futuro)"),
    ("painel_qa", "painel de QA humano"),
]

# Etapas do dry-run do pipeline SAR futuro (subsistema E).
# (etapa, input_esperado, output_esperado, sucesso, fail_closed)
DRY_RUN_STEPS = [
    ("confirmar_aoi", "bbox do patch", "AOI confirmado", "AOI valido e intersecta cenas", "sem AOI valido, parar"),
    ("confirmar_par_pre_pos", "par selecionado GT09", "par confirmado", "pre e pos com scene_id/product_id", "sem par, parar"),
    ("validar_metadados_grd_iw_vv_vh", "metadados das cenas", "metadados validados", "GRD/IW/VV+VH confirmados", "produto/modo/pol invalido, parar"),
    ("planejar_calibracao_radiometrica", "cenas GRD futuras", "plano de calibracao", "calibracao especificada", "sem cena, parar"),
    ("planejar_speckle_filtering", "SAR calibrado futuro", "plano de speckle", "filtro especificado", "sem SAR, parar"),
    ("planejar_razao_pre_pos", "pre e pos futuros", "plano de razao pre/pos", "razao especificada", "sem par, parar"),
    ("planejar_mascara_agua_permanente", "mascara de agua futura", "plano de mascara", "mascara especificada", "sem mascara, marcar no_data"),
    ("planejar_filtro_hand_slope", "DEM/HAND futuro", "plano de filtro topografico", "filtro especificado", "sem HAND, marcar incerteza"),
    ("planejar_exclusao_urbana", "camada urbana futura", "plano de exclusao urbana", "exclusao especificada", "sem camada, marcar ambiguidade"),
    ("planejar_poligono_candidato", "resultado SAR futuro", "poligono candidato (referencia de avaliacao)", "poligono especificado", "sem resultado, parar"),
    ("planejar_qa_humano", "poligono candidato futuro", "plano de QA humano", "QA especificado", "sem QA, nao promover"),
]

# Grupos de checklist do GT07 e como o handoff os trata (subsistema F).
HANDOFF_GROUP_MAP = {
    "fenomeno": ("desbloqueavel_apos_visualizacao", "visualizacao ajuda a confirmar inundacao vs outro fenomeno"),
    "temporalidade": ("desbloqueavel_apos_visualizacao", "datas e janelas ja conhecidas; visualizacao confirma"),
    "espacialidade": ("desbloqueavel_apos_visualizacao", "overlay AOI/patch permite conferir relacao espacial"),
    "fonte": ("ja_disponivel", "fonte/autoridade ja registrada"),
    "footprint_futuro": ("depende_de_footprint", "exige footprint SAR pos-evento futuro"),
    "exclusoes": ("depende_de_footprint", "exige footprint e mascaras futuras"),
    "leakage": ("depende_de_qa_humano", "auditoria de leakage exige revisao humana"),
    "incerteza": ("depende_de_footprint", "incerteza depende do footprint futuro"),
    "decisao": ("depende_de_qa_humano", "decisao final exige QA humano"),
}

# Inventario de ativos locais (subsistema C).
ASSET_EXTS_VISUAL = {"png", "jpg", "jpeg", "webp", "svg", "tif", "tiff"}
ASSET_EXTS_GEOM = {"geojson", "wkt"}
ASSET_KEYWORDS = ("recife", "canary", "canario", "s18a", "patch", "preview", "sentinel", "pe3d", "mde", "risco")

FLOOD_COMPATIBLE = gt08.FLOOD_COMPATIBLE
_clean = gt08._clean
_present = gt08._present
USABLE_WINDOW = gt08.gt06.USABLE_WINDOW


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
    missing = [rel(p) for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("entradas GT08/GT07 ausentes: " + "; ".join(missing))


def _seq(x: str) -> str:
    m = re.search(r"(\d+)$", x)
    return m.group(1) if m else x


def _b(v) -> str:
    return "true" if v else "false"


def _scene_date(dt_str: str) -> str:
    s = _clean(dt_str)
    return s[:10] if len(s) >= 10 else ""


def _days_between(a: str, b: str):
    try:
        return abs((_dt.date.fromisoformat(a) - _dt.date.fromisoformat(b)).days)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# A) Seletor de par pre/pos
# ---------------------------------------------------------------------------
def _pair_score(pre: dict, post: dict) -> tuple[int, list[str]]:
    reasons = []
    s = 0
    if (pre["orbit_path"], pre["orbit_frame"], pre["orbit_direction"]) == (post["orbit_path"], post["orbit_frame"], post["orbit_direction"]):
        s += 50
        reasons.append("mesma orbita/path/frame")
    if "VV" in pre["polarization"] and "VH" in pre["polarization"] and "VV" in post["polarization"] and "VH" in post["polarization"]:
        s += 20
        reasons.append("VV+VH em ambas")
    # proximidade temporal ao evento (quanto menor o gap, melhor); janelas garantem lado correto
    s += 15  # ambas ja dentro das janelas (pre/pos validas)
    reasons.append("pre na janela pre e pos na janela pos")
    return s, reasons


def select_pairs(pre_list: list[dict], post_list: list[dict], event_date: str):
    """Ranqueia todos os pares pre x pos; retorna (melhor, alternativos, todos_ranqueados)."""
    pairs = []
    for pre in pre_list:
        for post in post_list:
            score, reasons = _pair_score(pre, post)
            pd = _scene_date(pre["acquisition_datetime"])
            qd = _scene_date(post["acquisition_datetime"])
            pairs.append({
                "pre": pre, "post": post, "score": score, "reasons": reasons,
                "delta_pre": _days_between(pd, event_date), "delta_post": _days_between(qd, event_date),
                "same_orbit": score >= 50,
            })
    # ordenar por score desc, depois menor delta total, depois scene_id estavel
    def keyf(p):
        dt = (p["delta_pre"] or 999) + (p["delta_post"] or 999)
        return (-p["score"], dt, p["pre"]["scene_id"], p["post"]["scene_id"])
    pairs.sort(key=keyf)
    best = pairs[0] if pairs else None
    alternatives = pairs[1:] if len(pairs) > 1 else []
    return best, alternatives, pairs


# ---------------------------------------------------------------------------
# C) Inventario de ativos locais (versionados)
# ---------------------------------------------------------------------------
def inventory_assets() -> list[dict]:
    tracked = _run_git(["ls-files", "outputs_public"]).splitlines()
    rows = []
    idx = 1
    for path in sorted(tracked):
        low = path.lower()
        ext = low.rsplit(".", 1)[-1] if "." in low else ""
        if ext not in (ASSET_EXTS_VISUAL | ASSET_EXTS_GEOM):
            continue
        if not any(k in low for k in ASSET_KEYWORDS):
            continue
        if ext in ASSET_EXTS_VISUAL:
            acls = "ativo_visual_existente"
        else:
            acls = "ativo_geometrico_existente"
        # associacao heuristica a canario por indice rec_0000N
        m = re.search(r"canary_rec_0*([1-9]\d*)", low)
        if m:
            assoc = f"canario_indice_{int(m.group(1))}"
            method, conf = "indice_canario_recife_ordenado", "media"
        else:
            assoc = "recife_regiao"
            method, conf = "regiao", "baixa"
        rows.append({
            "asset_id": f"AST_{idx:04d}", "path": path, "asset_type": ext,
            "asset_class": acls, "associated_canary": assoc,
            "association_method": method, "association_confidence": conf,
            "image_generated_in_this_milestone": "false",
            "downloaded_in_this_milestone": "false", "review_only": "true",
        })
        idx += 1
    return rows


def _canary_asset_counts(assets: list[dict]) -> dict:
    counts = {}
    for a in assets:
        m = re.match(r"canario_indice_(\d+)", a["associated_canary"])
        if m and a["asset_class"] == "ativo_visual_existente":
            counts[int(m.group(1))] = counts.get(int(m.group(1)), 0) + 1
    counts["_region_visual"] = sum(1 for a in assets if a["asset_class"] == "ativo_visual_existente")
    counts["_region_geom"] = sum(1 for a in assets if a["asset_class"] == "ativo_geometrico_existente")
    return counts


# ---------------------------------------------------------------------------
# G) Readiness gate
# ---------------------------------------------------------------------------
def gate_state(aoi_ok: bool, has_pair: bool, pair_has_ids: bool, has_window: bool,
               has_source_pheno: bool) -> str:
    if not aoi_ok:
        return "bloqueado_por_aoi"
    if not has_pair or not pair_has_ids:
        return "bloqueado_por_par_pre_pos"
    if aoi_ok and has_pair and pair_has_ids and has_window and has_source_pheno:
        return "pronto_para_renderizacao_visual_futura"
    return "bloqueado_por_politica_review_only"


# ---------------------------------------------------------------------------
# Construcao das linhas
# ---------------------------------------------------------------------------
def build_rows() -> dict:
    targets = _read_csv(GT08_TARGETS)
    aoi = {a["sentinel1_target_id"]: a for a in _read_csv(GT08_AOI)}
    scenes = _read_csv(GT08_SCENES)
    queue = {q["target_id"]: q for q in _read_csv(GT07_QUEUE)}
    checklist_groups = sorted({c["checklist_group"] for c in _read_csv(GT07_CHECKLIST)})

    scenes_by_target = {}
    for s in scenes:
        scenes_by_target.setdefault(s["sentinel1_target_id"], []).append(s)

    assets = inventory_assets()
    asset_counts = _canary_asset_counts(assets)
    region_visual = asset_counts.get("_region_visual", 0)

    pairs_rows, alt_rows, dossier_rows, render_rows, dryrun_rows = [], [], [], [], []
    handoff_rows, gate_rows, leakage_rows, blocker_rows = [], [], [], []
    b_idx = 1

    for t in targets:
        sid = t["sentinel1_target_id"]
        tid = t["target_id"]
        rank = _seq(sid)
        a = aoi.get(sid, {})
        q = queue.get(tid, {})
        event_date = _clean(t.get("resolved_event_date"))
        pre_list = [s for s in scenes_by_target.get(sid, []) if s["scene_classification"] == "pre_event_candidate"]
        post_list = [s for s in scenes_by_target.get(sid, []) if s["scene_classification"] == "post_event_candidate"]
        best, alternatives, _all = select_pairs(pre_list, post_list, event_date)

        aoi_ok = _clean(a.get("sufficient_for_search")) == "true" or _clean(t.get("aoi_status")) == "aoi_forte_por_bbox_ou_geometria_patch"
        has_window = _present(t.get("pre_event_window", "").split("..")[0])
        has_source_pheno = _present(t.get("phenomenon_type")) and _present(q.get("source_authority") or "defesa_civil")
        pair_has_ids = bool(best) and _present(best["pre"]["scene_id"]) and _present(best["post"]["scene_id"])

        # A) par selecionado
        if best:
            pre, post = best["pre"], best["post"]
            pairs_rows.append({
                "pair_id": f"PP_{rank}", "sentinel1_target_id": sid, "target_id": tid,
                "patch_id": t.get("patch_id", "not_available"), "event_id": t.get("event_id", "not_available"),
                "event_date": event_date or "not_available",
                "pre_scene_id": pre["scene_id"], "pre_product_id": pre["product_id"],
                "pre_acquisition": pre["acquisition_datetime"], "pre_polarization": pre["polarization"],
                "post_scene_id": post["scene_id"], "post_product_id": post["product_id"],
                "post_acquisition": post["acquisition_datetime"], "post_polarization": post["polarization"],
                "orbit_path": pre["orbit_path"], "orbit_frame": pre["orbit_frame"],
                "orbit_direction": pre["orbit_direction"],
                "same_orbit_pair": _b(best["same_orbit"]),
                "delta_pre_event_days": str(best["delta_pre"]) if best["delta_pre"] is not None else "not_available",
                "delta_event_post_days": str(best["delta_post"]) if best["delta_post"] is not None else "not_available",
                "pair_score": str(best["score"]),
                "aoi_intersection": _b(_clean(pre.get("aoi_overlap")) == "true" and _clean(post.get("aoi_overlap")) == "true"),
                "selection_reason_pt": "; ".join(best["reasons"]),
                "selected": "true",
                "metadata_only": "true", "no_download_in_this_milestone": "true", "review_only": "true",
                "post_event_scene_role_pt": "referencia de avaliacao futura, nunca feature pre-evento",
            })
            for j, alt in enumerate(alternatives, start=1):
                ap, aq = alt["pre"], alt["post"]
                alt_rows.append({
                    "alt_pair_id": f"AP_{rank}_{j:02d}", "sentinel1_target_id": sid, "target_id": tid,
                    "rank": str(j + 1), "pre_scene_id": ap["scene_id"], "post_scene_id": aq["scene_id"],
                    "same_orbit_pair": _b(alt["same_orbit"]), "pair_score": str(alt["score"]),
                    "why_not_selected_pt": "score menor que o par selecionado" if alt["score"] < best["score"] else "empate resolvido por menor delta temporal/scene_id",
                    "metadata_only": "true", "no_download_in_this_milestone": "true", "review_only": "true",
                })
        else:
            pairs_rows.append({
                "pair_id": f"PP_{rank}", "sentinel1_target_id": sid, "target_id": tid,
                "patch_id": t.get("patch_id", "not_available"), "event_id": t.get("event_id", "not_available"),
                "event_date": event_date or "not_available",
                "pre_scene_id": "not_available", "pre_product_id": "not_available",
                "pre_acquisition": "not_available", "pre_polarization": "not_available",
                "post_scene_id": "not_available", "post_product_id": "not_available",
                "post_acquisition": "not_available", "post_polarization": "not_available",
                "orbit_path": "not_available", "orbit_frame": "not_available", "orbit_direction": "not_available",
                "same_orbit_pair": "false", "delta_pre_event_days": "not_available",
                "delta_event_post_days": "not_available", "pair_score": "0",
                "aoi_intersection": "false", "selection_reason_pt": "sem par pre/pos valido",
                "selected": "false", "metadata_only": "true", "no_download_in_this_milestone": "true",
                "review_only": "true",
                "post_event_scene_role_pt": "referencia de avaliacao futura, nunca feature pre-evento",
            })

        # B) dossie
        cvis = asset_counts.get(int(rank[-1]) if rank[-1].isdigit() else 0, 0)
        dossier_rows.append({
            "dossier_id": f"DOS_{rank}", "sentinel1_target_id": sid, "target_id": tid,
            "patch_id": t.get("patch_id", "not_available"), "aoi_bbox": t.get("aoi_bbox", "not_available"),
            "city": t.get("city", "not_available"), "region": t.get("region", "not_available"),
            "event_date": event_date or "not_available", "phenomenon_type": t.get("phenomenon_type", "not_available"),
            "source_authority": q.get("source_authority", "defesa_civil_recife_occurrence"),
            "pre_scene_id": best["pre"]["scene_id"] if best else "not_available",
            "post_scene_id": best["post"]["scene_id"] if best else "not_available",
            "delta_pre_event_days": str(best["delta_pre"]) if best and best["delta_pre"] is not None else "not_available",
            "delta_event_post_days": str(best["delta_post"]) if best and best["delta_post"] is not None else "not_available",
            "pair_selection_reason_pt": "; ".join(best["reasons"]) if best else "sem par",
            "local_visual_assets_for_canary": str(cvis),
            "requisitos_revisao_visual_pt": "painel pre/pos, overlay AOI/patch, diferenca pre/pos",
            "requisitos_qa_futuro_pt": "footprint SAR pos-evento, mascaras de exclusao, QA humano",
            "remains_review_only": "true", "trainable": "false",
            "eligible_for_training": "false", "eligible_for_ground_truth": "false",
            "score_v7_candidate": "false",
            "rationale_pt": "dossie tecnico-visual review-only; par pre/pos localizado; sem download nem SAR",
        })

        # D) contrato de renderizacao futura
        for panel, desc in RENDER_PANELS:
            render_rows.append({
                "render_contract_id": f"RC_{len(render_rows)+1:05d}", "sentinel1_target_id": sid,
                "panel": panel, "panel_description_pt": desc,
                "renderizacao_executada": "false", "raster_baixado": "false", "footprint_criado": "false",
                "requires_scene_download_future": _b(panel in ("painel_pre_sar", "painel_pos_sar", "diferenca_pre_pos", "mascara_candidata_futura")),
                "requires_footprint_future": _b(panel in ("mascara_candidata_futura", "painel_qa")),
                "input_esperado_pt": "cenas SAR/overlays futuros" if "sar" in panel or "mascara" in panel else "geometria/AOI local",
                "review_only": "true",
            })

        # E) dry-run do pipeline SAR futuro
        for order, (step, inp, outp, ok, fc) in enumerate(DRY_RUN_STEPS, start=1):
            dryrun_rows.append({
                "dry_run_id": f"DR_{len(dryrun_rows)+1:05d}", "sentinel1_target_id": sid,
                "step_order": str(order), "step": step,
                "input_esperado_pt": inp, "output_esperado_pt": outp,
                "criterio_sucesso_pt": ok, "criterio_fail_closed_pt": fc,
                "executado_agora": "false",
                "requires_raster_future": _b(order >= 4 and order <= 10),
                "review_only": "true",
            })

        # F) handoff QA (por grupo de checklist do GT07)
        for grp in checklist_groups:
            status, why = HANDOFF_GROUP_MAP.get(grp, ("depende_de_qa_humano", "requer revisao humana"))
            handoff_rows.append({
                "handoff_id": f"HO_{len(handoff_rows)+1:05d}", "sentinel1_target_id": sid,
                "checklist_group": grp, "handoff_status": status, "reason_pt": why,
                "unblockable_after_visualization": _b(status == "desbloqueavel_apos_visualizacao"),
                "depends_on_footprint": _b(status == "depende_de_footprint"),
                "depends_on_qa_human": _b(status == "depende_de_qa_humano"),
                "future_decision_possible_pt": "aceitar/rejeitar/ambiguo apos QA humano",
                "accepted_future_is_review_only": "true",
                "current_qa_status": "prepared_for_future_review",
                "review_only": "true",
            })

        # G) readiness gate
        gs = gate_state(aoi_ok, bool(best), pair_has_ids, has_window, has_source_pheno)
        gate_rows.append({
            "gate_id": f"GT_{rank}", "sentinel1_target_id": sid, "target_id": tid,
            "gate_state": gs,
            "aoi_forte": _b(aoi_ok), "par_selecionado": _b(bool(best)),
            "scene_ids_presentes": _b(pair_has_ids), "janela_valida": _b(has_window),
            "fonte_fenomeno_presentes": _b(has_source_pheno),
            "can_advance_to_download_controlled_future": _b(pair_has_ids),
            "can_advance_to_qa_visual_future": _b(region_visual > 0),
            "local_visual_assets_region": str(region_visual),
            "remains_review_only": "true", "trainable": "false",
            "eligible_for_training": "false", "eligible_for_ground_truth": "false",
            "score_v7_candidate": "false",
            "rationale_pt": _gate_rationale(gs),
        })

        # guardrails de leakage
        leakage_rows.append({
            "leakage_id": f"LK_{rank}", "sentinel1_target_id": sid,
            "post_event_scene_id": best["post"]["scene_id"] if best else "not_available",
            "post_scene_used_as_pre_event_feature": "false",
            "footprint_is_evaluation_reference_only": "true",
            "pre_event_features_from_pre_window_only": "true",
            "post_event_only_for_reference_not_feature": "true",
            "leakage_check_status": "sem_leakage_por_construcao",
            "reason_pt": "cena pos-evento e referencia de avaliacao futura; nunca entra como feature pre-evento",
            "review_only": "true",
        })

        # bloqueios
        if gs.startswith("bloqueado"):
            blocker_rows.append({
                "esteira_blocker_id": f"EB_{b_idx:05d}", "sentinel1_target_id": sid,
                "blocker_type": gs, "severity": "alta",
                "reason_pt": _gate_rationale(gs),
                "prevents_render_future": "true", "prevents_positive_strong": "true",
                "how_to_resolve_pt": "resolver AOI/par pre-pos antes de avancar",
            })
            b_idx += 1

    return {
        "pairs": pairs_rows, "alt": alt_rows, "dossier": dossier_rows, "assets": assets,
        "render": render_rows, "dryrun": dryrun_rows, "handoff": handoff_rows,
        "gate": gate_rows, "leakage": leakage_rows, "blockers": blocker_rows,
        "region_visual": region_visual, "region_geom": asset_counts.get("_region_geom", 0),
    }


def _gate_rationale(gs: str) -> str:
    return {
        "pronto_para_renderizacao_visual_futura": "AOI forte, par pre/pos com scene_id/product_id, janela valida, fonte/fenomeno presentes; pronto para renderizacao visual futura review-only",
        "bloqueado_por_aoi": "AOI insuficiente para a esteira",
        "bloqueado_por_par_pre_pos": "sem par pre/pos valido com scene_id/product_id",
        "bloqueado_por_ativo_visual": "sem ativo visual local para revisao",
        "bloqueado_por_politica_review_only": "faltam requisitos de politica review-only",
    }.get(gs, "estado do gate")


# ---------------------------------------------------------------------------
# Resumo
# ---------------------------------------------------------------------------
def summary_rows(bundle: dict) -> list[dict]:
    order = {s: i for i, s in enumerate(GATE_STATES)}
    agg = {}
    for g in bundle["gate"]:
        s = g["gate_state"]
        a = agg.setdefault(s, {"count": 0, "download": 0, "qa_visual": 0})
        a["count"] += 1
        a["download"] += g["can_advance_to_download_controlled_future"] == "true"
        a["qa_visual"] += g["can_advance_to_qa_visual_future"] == "true"
    rows = []
    for s in sorted(agg, key=lambda x: order.get(x, 99)):
        a = agg[s]
        rows.append({
            "gate_state": s, "count": str(a["count"]),
            "can_advance_to_download_controlled_future_count": str(a["download"]),
            "can_advance_to_qa_visual_future_count": str(a["qa_visual"]),
            "render_executada_count": "0", "raster_baixado_count": "0",
            "footprint_criado_count": "0", "qa_executado_count": "0",
            "training_allowed_count": "0", "ground_truth_allowed_count": "0",
            "score_v7_candidate_count": "0",
        })
    return rows


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
_BOOL = {"enum": ["true", "false"]}
_FALSE = {"const": "false"}
_TRUE = {"const": "true"}


def schema_pair_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "susc_gt09_pre_post_pair.schema.json", "title": "SUSC-GT09 Par Pre/Pos", "type": "object",
        "required": ["pair_id", "sentinel1_target_id", "selected", "same_orbit_pair",
                     "metadata_only", "no_download_in_this_milestone", "review_only"],
        "properties": {
            "pair_id": {"type": "string", "pattern": r"^PP_\d+$"},
            "sentinel1_target_id": {"type": "string", "pattern": r"^S1_\d+$"},
            "selected": _BOOL, "same_orbit_pair": _BOOL, "aoi_intersection": _BOOL,
            "metadata_only": _TRUE, "no_download_in_this_milestone": _TRUE, "review_only": _TRUE,
        },
    }


def schema_dossier_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "susc_gt09_canary_dossier.schema.json", "title": "SUSC-GT09 Dossie Canario", "type": "object",
        "required": ["dossier_id", "sentinel1_target_id", "remains_review_only", "trainable",
                     "eligible_for_training", "eligible_for_ground_truth", "score_v7_candidate", "rationale_pt"],
        "properties": {
            "dossier_id": {"type": "string", "pattern": r"^DOS_\d+$"},
            "sentinel1_target_id": {"type": "string", "pattern": r"^S1_\d+$"},
            "remains_review_only": _TRUE, "trainable": _FALSE, "eligible_for_training": _FALSE,
            "eligible_for_ground_truth": _FALSE, "score_v7_candidate": _FALSE,
            "rationale_pt": {"type": "string", "minLength": 1},
        },
    }


def schema_asset_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "susc_gt09_visual_asset_inventory.schema.json", "title": "SUSC-GT09 Inventario de Ativos", "type": "object",
        "required": ["asset_id", "path", "asset_class", "image_generated_in_this_milestone",
                     "downloaded_in_this_milestone", "review_only"],
        "properties": {
            "asset_id": {"type": "string", "pattern": r"^AST_\d+$"},
            "path": {"type": "string", "minLength": 1},
            "asset_class": {"enum": ASSET_CLASSES},
            "image_generated_in_this_milestone": _FALSE,
            "downloaded_in_this_milestone": _FALSE, "review_only": _TRUE,
        },
    }


def schema_render_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "susc_gt09_future_render_contract.schema.json", "title": "SUSC-GT09 Contrato de Renderizacao", "type": "object",
        "required": ["render_contract_id", "sentinel1_target_id", "panel",
                     "renderizacao_executada", "raster_baixado", "footprint_criado", "review_only"],
        "properties": {
            "render_contract_id": {"type": "string", "pattern": r"^RC_\d+$"},
            "sentinel1_target_id": {"type": "string", "pattern": r"^S1_\d+$"},
            "panel": {"type": "string", "minLength": 1},
            "renderizacao_executada": _FALSE, "raster_baixado": _FALSE,
            "footprint_criado": _FALSE, "review_only": _TRUE,
        },
    }


def schema_dryrun_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "susc_gt09_sar_dry_run_step.schema.json", "title": "SUSC-GT09 Etapa Dry-Run SAR", "type": "object",
        "required": ["dry_run_id", "sentinel1_target_id", "step", "executado_agora", "review_only"],
        "properties": {
            "dry_run_id": {"type": "string", "pattern": r"^DR_\d+$"},
            "sentinel1_target_id": {"type": "string", "pattern": r"^S1_\d+$"},
            "step": {"type": "string", "minLength": 1},
            "executado_agora": _FALSE, "review_only": _TRUE,
        },
    }


def schema_gate_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "susc_gt09_observational_readiness_gate.schema.json", "title": "SUSC-GT09 Readiness Gate", "type": "object",
        "required": ["gate_id", "sentinel1_target_id", "gate_state", "remains_review_only",
                     "trainable", "eligible_for_training", "eligible_for_ground_truth", "score_v7_candidate"],
        "properties": {
            "gate_id": {"type": "string", "pattern": r"^GT_\d+$"},
            "sentinel1_target_id": {"type": "string", "pattern": r"^S1_\d+$"},
            "gate_state": {"enum": GATE_STATES},
            "remains_review_only": _TRUE, "trainable": _FALSE, "eligible_for_training": _FALSE,
            "eligible_for_ground_truth": _FALSE, "score_v7_candidate": _FALSE,
        },
    }


def schema_summary_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "susc_gt09_summary.schema.json", "title": "SUSC-GT09 Resumo", "type": "object",
        "required": ["gate_state", "count", "render_executada_count", "raster_baixado_count",
                     "footprint_criado_count", "qa_executado_count", "training_allowed_count",
                     "ground_truth_allowed_count", "score_v7_candidate_count"],
        "properties": {
            "gate_state": {"enum": GATE_STATES}, "count": {"type": "string", "pattern": r"^\d+$"},
            "render_executada_count": {"const": "0"}, "raster_baixado_count": {"const": "0"},
            "footprint_criado_count": {"const": "0"}, "qa_executado_count": {"const": "0"},
            "training_allowed_count": {"const": "0"}, "ground_truth_allowed_count": {"const": "0"},
            "score_v7_candidate_count": {"const": "0"},
        },
    }


# ---------------------------------------------------------------------------
# Manifesto / proximo marco
# ---------------------------------------------------------------------------
def _next_milestone(bundle: dict) -> str:
    gates = bundle["gate"]
    bloqueados = [g for g in gates if g["gate_state"].startswith("bloqueado")]
    prontos = [g for g in gates if g["gate_state"] == "pronto_para_renderizacao_visual_futura"]
    if bloqueados:
        return "GT10 - Resolucao de Bloqueios AOI/Par"
    if bundle["region_visual"] > 0 and prontos:
        return "GT10 - Renderizacao Visual Local Controlada"
    if prontos:
        return "GT10 - Download Controlado Sentinel-1"
    if bundle["region_visual"] > 0:
        return "GT10 - QA Manual Externo"
    return "GT10 - Resolucao de Bloqueios AOI/Par"


def manifest_obj(bundle: dict) -> dict:
    gates = bundle["gate"]
    dist = {}
    for g in gates:
        dist[g["gate_state"]] = dist.get(g["gate_state"], 0) + 1
    selected_pairs = sum(1 for p in bundle["pairs"] if p["selected"] == "true")
    same_orbit = sum(1 for p in bundle["pairs"] if p["selected"] == "true" and p["same_orbit_pair"] == "true")
    artifacts = [
        {"path": rel(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in PUBLIC_ARTIFACTS + [SCHEMA_PAIR, SCHEMA_DOSSIER, SCHEMA_ASSET, SCHEMA_RENDER, SCHEMA_DRYRUN, SCHEMA_GATE, SCHEMA_SUMMARY]
        if p.exists()
    ]
    return {
        "marco": MARCO, "titulo_publico": TITULO_PUBLICO,
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "gt08_source_commit": _run_git(["log", "-1", "--format=%h", "--", rel(GT08_TARGETS)]) or "unknown",
        "canarios": len(gates), "distribuicao_gate": dist,
        "pares_selecionados": selected_pairs, "pares_mesma_orbita": same_orbit,
        "pares_alternativos": len(bundle["alt"]),
        "ativos_locais_total": len(bundle["assets"]),
        "ativos_visuais": bundle["region_visual"], "ativos_geometricos": bundle["region_geom"],
        "render_executada": 0, "raster_baixado": 0, "footprint_criado": 0, "geometria_criada": 0,
        "sar_executado": 0, "qa_executado": 0, "imagem_gerada": 0, "download_executado": 0,
        "internet_usada": 0, "positive_strong_promovidos": 0,
        "accepted_atual": 0, "rejected_atual": 0,
        "global_guardrails": GLOBAL_GUARDRAILS,
        "training_true_count": sum(1 for g in gates if g["eligible_for_training"] == "true"),
        "ground_truth_true_count": sum(1 for g in gates if g["eligible_for_ground_truth"] == "true"),
        "score_v7_candidate_true_count": sum(1 for g in gates if g["score_v7_candidate"] == "true"),
        "dry_run_executado_agora_count": sum(1 for d in bundle["dryrun"] if d["executado_agora"] == "true"),
        "score_v6_changed": gt02.gt01._score_v6_changed(),
        "score_v7_created": gt02.gt01._score_v7_exists(),
        "proximo_marco": _next_milestone(bundle),
        "artifacts": artifacts,
    }


# ---------------------------------------------------------------------------
# Relatorio publico (portugues)
# ---------------------------------------------------------------------------
def report_text(manifest: dict, bundle: dict, summary: list[dict]) -> str:
    linhas_par = "\n".join(
        f"| {p['pair_id']} | {p['patch_id']} | {p['pre_scene_id'][:32]} | {p['post_scene_id'][:32]} | "
        f"{p['same_orbit_pair']} | {p['pair_score']} |"
        for p in bundle["pairs"]
    )
    linhas_gate = "\n".join(
        f"| {r['gate_state']} | {r['count']} | {r['can_advance_to_download_controlled_future_count']} | "
        f"{r['can_advance_to_qa_visual_future_count']} |"
        for r in summary
    )

    return f"""# {MARCO} - {TITULO_PUBLICO}

## 1. Escopo do marco

Este marco composto transforma os 5 canários Recife em uma **esteira completa de
pré-execução observacional**: melhor par Sentinel-1 pré/pós por canário, dossiê
técnico-visual, inventário de ativos locais, contrato de renderização futura, dry-run do
pipeline SAR, handoff para QA humano, readiness gate e guardrails de leakage. É **offline
e review-only**: não usa internet, não consulta STAC/API remota, não baixa Sentinel nem
raster, não abre nem gera imagem, não roda SAR/GEE, não cria footprint, não cria geometria
real, não executa QA, não altera o `score_v6`, não cria `score_v7`, não treina modelo e não
promove nada a ground truth nem a `positive_strong`.

## 2. Relação com GT01 a GT08

O GT08 encontrou, offline, AOI forte e um catálogo Sentinel-1 local real com pares pré/pós
para os 5 canários. O GT09 consolida essa fronteira em uma esteira auditável, **sem
cruzá-la**. O SAR continua sendo canário review-only, não a base central do dataset.

## 3. Canários e pares pré/pós selecionados

Entraram **{manifest['canarios']}** canários; **{manifest['pares_selecionados']}** com par
pré/pós selecionado, sendo **{manifest['pares_mesma_orbita']}** de mesma órbita/path/frame.
Pares alternativos: **{manifest['pares_alternativos']}**.

| par | patch | cena pré | cena pós | mesma órbita | score |
| --- | --- | --- | --- | --- | --- |
{linhas_par}

O par é escolhido preferindo mesma órbita/path/frame, polarização VV+VH e menor distância
temporal ao evento; os demais pares ficam como alternativos em
`susc_gt09_pares_alternativos_canarios.csv`.

## 4. Dossiê por canário

`susc_gt09_dossie_canarios.csv` reúne, por canário, patch/AOI, bairro, data, fonte,
fenômeno, cenas pré/pós, deltas temporais (pré→evento, evento→pós), razão do par e
requisitos para revisão visual e QA futuro.

## 5. Inventário de ativos locais

Foram inventariados **{manifest['ativos_locais_total']}** ativos **já versionados**
(**{manifest['ativos_visuais']}** visuais e **{manifest['ativos_geometricos']}**
geométricos), sem gerar nenhuma imagem nem baixar nada. Incluem prévias de canário
(pré/pós) e camadas GeoJSON de bbox de patch.

## 6. Contrato de renderização futura

`susc_gt09_contrato_renderizacao_futura.csv` define nove painéis por canário (pré SAR, pós
SAR, diferença pré/pós, overlay AOI/patch, overlays de água permanente, HAND/slope e
exclusões urbanas, máscara candidata e painel de QA), todos com `renderizacao_executada=false`,
`raster_baixado=false` e `footprint_criado=false`.

## 7. Dry-run do pipeline SAR futuro

`susc_gt09_dry_run_pipeline_sar_futuro.csv` traz onze etapas lógicas por canário (confirmar
AOI; confirmar par; validar GRD/IW/VV/VH; calibração; speckle; razão pré/pós; máscara de
água; HAND/slope; exclusão urbana; polígono candidato; QA humano), cada uma com input,
output, critério de sucesso e critério fail-closed. **Nenhuma** foi executada
(`executado_agora=false`).

## 8. Handoff para QA humano

`susc_gt09_handoff_qa_canarios.csv` mapeia cada grupo do checklist do GT07: quais itens
ficam **desbloqueáveis após visualização**, quais **dependem de footprint** e quais
**dependem de QA humano**. Nenhum canário é marcado como aceito/rejeitado; `accepted`
futuro só liberaria avaliação review-only.

## 9. Readiness gate observacional

| gate_state | canários | podem baixar (futuro) | podem QA visual (futuro) |
| --- | --- | --- | --- |
{linhas_gate}

Um canário só é `pronto_para_renderizacao_visual_futura` com AOI forte, par pré/pós com
scene_id/product_id, janela válida, fonte/fenômeno e guardrails review-only.

## 10. Guardrails de leakage

`susc_gt09_guardrails_leakage_evidencia.csv` registra, por canário, que a **cena pós-evento
é referência de avaliação futura, nunca feature pré-evento**
(`post_scene_used_as_pre_event_feature=false`), e que features pré-evento só vêm da janela
pré.

## 11. Por que nada foi baixado ou processado

Este marco é de **pré-execução**: prepara a esteira a partir do que já está versionado, sem
rede, download, SAR ou geração de imagem. A execução fica para marcos futuros controlados.

## 12. Confirmação explícita

**Não** houve internet, STAC/API remota, download
(`download_executado={manifest['download_executado']}`), SAR/GEE, imagem gerada
(`imagem_gerada={manifest['imagem_gerada']}`), footprint criado
(`footprint_criado={manifest['footprint_criado']}`), geometria criada, QA executado
(`qa_executado={manifest['qa_executado']}`), accepted/rejected atual
(`accepted_atual={manifest['accepted_atual']}`, `rejected_atual={manifest['rejected_atual']}`),
alteração do `score_v6` (`score_v6_changed={str(manifest['score_v6_changed']).lower()}`) nem
promoção a `positive_strong`
(`positive_strong_promovidos={manifest['positive_strong_promovidos']}`). Etapas de dry-run
executadas agora: {manifest['dry_run_executado_agora_count']}. Contagens de controle:
`eligible_for_training=true` → {manifest['training_true_count']};
`eligible_for_ground_truth=true` → {manifest['ground_truth_true_count']};
`score_v7_candidate=true` → {manifest['score_v7_candidate_true_count']}.

O REV-P não prevê enchentes operacionalmente: produz análise estrutural review-only com
evidência observacional auditável, e nenhuma evidência vira ground truth supervisionado sem
política, QA e incerteza.

## 13. Próximo passo recomendado

**{manifest['proximo_marco']}**. Com AOI forte, pares pré/pós de mesma órbita e ativos
visuais locais já disponíveis para os canários de Recife, o próximo passo é a renderização
visual local controlada, mantendo tudo review-only e sem download.
"""


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_all() -> dict:
    require_inputs()
    ensure_dir(OUT)
    ensure_dir(SCHEMAS_DIR)

    bundle = build_rows()
    summary = summary_rows(bundle)

    write_csv(PAIRS_CSV, bundle["pairs"])
    write_csv(ALT_PAIRS_CSV, bundle["alt"])
    write_csv(DOSSIER_CSV, bundle["dossier"])
    write_csv(ASSETS_CSV, bundle["assets"])
    write_csv(RENDER_CSV, bundle["render"])
    write_csv(DRYRUN_CSV, bundle["dryrun"])
    write_csv(HANDOFF_CSV, bundle["handoff"])
    write_csv(GATE_CSV, bundle["gate"])
    write_csv(LEAKAGE_CSV, bundle["leakage"])
    write_csv(BLOCKERS_CSV, bundle["blockers"])
    write_csv(SUMMARY_CSV, summary)

    write_json(SCHEMA_PAIR, schema_pair_obj())
    write_json(SCHEMA_DOSSIER, schema_dossier_obj())
    write_json(SCHEMA_ASSET, schema_asset_obj())
    write_json(SCHEMA_RENDER, schema_render_obj())
    write_json(SCHEMA_DRYRUN, schema_dryrun_obj())
    write_json(SCHEMA_GATE, schema_gate_obj())
    write_json(SCHEMA_SUMMARY, schema_summary_obj())

    manifest = manifest_obj(bundle)
    write_markdown(REPORT, report_text(manifest, bundle, summary))
    manifest["artifacts"] = [
        {"path": rel(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in PUBLIC_ARTIFACTS + [SCHEMA_PAIR, SCHEMA_DOSSIER, SCHEMA_ASSET, SCHEMA_RENDER, SCHEMA_DRYRUN, SCHEMA_GATE, SCHEMA_SUMMARY]
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
    for k in ("render_executada", "raster_baixado", "footprint_criado", "geometria_criada",
              "sar_executado", "qa_executado", "imagem_gerada", "download_executado",
              "internet_usada", "accepted_atual", "rejected_atual"):
        if manifest[k] != 0:
            errors.append(f"{k}_forbidden")
    if manifest["dry_run_executado_agora_count"] != 0:
        errors.append("dry_run_executado_agora_forbidden")

    pairs = _read_csv(PAIRS_CSV)
    dossier = _read_csv(DOSSIER_CSV)
    assets = _read_csv(ASSETS_CSV)
    render = _read_csv(RENDER_CSV)
    dryrun = _read_csv(DRYRUN_CSV)
    handoff = _read_csv(HANDOFF_CSV)
    gate = _read_csv(GATE_CSV)
    leakage = _read_csv(LEAKAGE_CSV)
    summary = _read_csv(SUMMARY_CSV)

    # guardrails de nao-treino em dossie e gate
    for label, rows in (("dossie", dossier), ("gate", gate)):
        for r in rows:
            for fld in ("eligible_for_training", "eligible_for_ground_truth", "trainable", "score_v7_candidate"):
                if r.get(fld) != "false":
                    errors.append(f"{label}:{fld}_nao_false")
            if r.get("remains_review_only") != "true":
                errors.append(f"{label}:review_only_nao_true")

    # pares: selecionado exige pre e pos com scene_id/product_id + overlap AOI + janela
    for p in pairs:
        if p.get("selected") == "true":
            for fld in ("pre_scene_id", "pre_product_id", "post_scene_id", "post_product_id"):
                if not _present(p.get(fld)) or p.get(fld) == "not_available":
                    errors.append(f"par:{p['pair_id']}:{fld}_ausente")
            if p.get("aoi_intersection") != "true":
                errors.append(f"par:{p['pair_id']}:sem_overlap_aoi")
        if p.get("metadata_only") != "true" or p.get("no_download_in_this_milestone") != "true":
            errors.append(f"par:{p['pair_id']}:download_flag_incorreta")
        # cena pos nunca descrita como feature pre-evento
        if "feature pre-evento" in p.get("post_event_scene_role_pt", "").lower() and "nunca" not in p.get("post_event_scene_role_pt", "").lower():
            errors.append(f"par:{p['pair_id']}:pos_como_feature_pre")

    # inventario: nada gerado nem baixado
    for a in assets:
        if a.get("image_generated_in_this_milestone") != "false" or a.get("downloaded_in_this_milestone") != "false":
            errors.append(f"ativo:{a.get('asset_id')}:gerado_ou_baixado")
        if a.get("asset_class") not in ASSET_CLASSES:
            errors.append(f"ativo:{a.get('asset_id')}:classe_invalida")

    # render: nada executado/baixado/footprint
    for r in render:
        for fld in ("renderizacao_executada", "raster_baixado", "footprint_criado"):
            if r.get(fld) != "false":
                errors.append(f"render:{r.get('render_contract_id')}:{fld}_nao_false")

    # dry-run: nada executado agora
    for d in dryrun:
        if d.get("executado_agora") != "false":
            errors.append(f"dryrun:{d.get('dry_run_id')}:executado_agora_forbidden")

    # handoff: sem accepted/rejected atual; accepted futuro review-only
    for h in handoff:
        if h.get("current_qa_status") in ("accepted", "rejected"):
            errors.append(f"handoff:{h.get('handoff_id')}:qa_status_proibido")
        if h.get("accepted_future_is_review_only") != "true":
            errors.append(f"handoff:{h.get('handoff_id')}:accepted_futuro_nao_review_only")
        if h.get("handoff_status") not in HANDOFF_STATUS:
            errors.append(f"handoff:{h.get('handoff_id')}:status_invalido")

    # gate: estado no enum; pronto exige aoi/par/janela
    for g in gate:
        if g.get("gate_state") not in GATE_STATES:
            errors.append(f"gate:{g.get('gate_id')}:estado_invalido")
        if g.get("gate_state") == "pronto_para_renderizacao_visual_futura":
            if g.get("aoi_forte") != "true" or g.get("par_selecionado") != "true" or g.get("scene_ids_presentes") != "true":
                errors.append(f"gate:{g.get('gate_id')}:pronto_sem_requisitos")

    # leakage: pos nunca como feature pre-evento
    for lk in leakage:
        if lk.get("post_scene_used_as_pre_event_feature") != "false":
            errors.append(f"leakage:{lk.get('leakage_id')}:pos_como_feature_pre")
        if lk.get("footprint_is_evaluation_reference_only") != "true":
            errors.append(f"leakage:{lk.get('leakage_id')}:footprint_nao_referencia")

    # resumo: contagens proibidas 0
    for r in summary:
        for fld in ("render_executada_count", "raster_baixado_count", "footprint_criado_count",
                    "qa_executado_count", "training_allowed_count", "ground_truth_allowed_count",
                    "score_v7_candidate_count"):
            if r.get(fld) != "0":
                errors.append(f"resumo:{r['gate_state']}:{fld}_nao_zero")

    # texto publico
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
        "GT09 esteira canaria validada: "
        f"canarios={manifest['canarios']} gate={manifest['distribuicao_gate']} "
        f"pares={manifest['pares_selecionados']} mesma_orbita={manifest['pares_mesma_orbita']} "
        f"ativos={manifest['ativos_locais_total']} proximo={manifest['proximo_marco']} "
        f"score_v6_changed={str(manifest['score_v6_changed']).lower()}"
    )
    return 0


def run_all() -> int:
    manifest = build_all()
    print(
        "GT09 esteira canaria gerada: "
        f"canarios={manifest['canarios']} gate={manifest['distribuicao_gate']} "
        f"pares={manifest['pares_selecionados']} ativos={manifest['ativos_locais_total']} "
        f"proximo={manifest['proximo_marco']}"
    )
    return 0
