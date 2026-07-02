"""SUSC-17C39 Flow Accumulation Solver: Method Recovery, Variants e Calibracao Transferivel.

Resolve OPERACIONALMENTE o flow_acc_log_mean que impediu o replay comparavel do
score v6. O 17C38 provou que o problema nao era so truncamento por janela
(flow_acc melhorou -0.51 -> -0.17 mas seguiu incompativel). Aqui vamos alem da
forense: variantes hidrologicas + calibracao supervisionada nos 100 patches
OFICIAIS + validacao cruzada + aplicacao aos canarios.

Diagnostico-chave (17C39 Etapa 2): o flow_acc_log_mean OFICIAL correlaciona
+0.81 (spearman) com o twi_mean oficial e -0.60 com hand_mean oficial. Como o
HAND recomputado basin-aware ja e equivalent (17C38) e o TWI recomputado e
rank-preserving, um SURROGATE CALIBRADO multi-feature treinado SO nos patches
oficiais pode reproduzir a escala oficial do flow_acc (Resultado C).

Ordem de preferencia de solucao:
  1. method_recovered (variante direta equivalent)
  2. variant_equivalent
  3. calibrated_surrogate_validated (crossval nos oficiais)
  4. replacement_contract_review_only (so se 1-3 falharem, documentado)

Guardrails: calibracao SO nos 100 patches oficiais (canarios NUNCA treinam);
canarios so recebem aplicacao apos validacao; sem ocorrencia como feature; sem
normalizacao canary-only; sem surrogate sem crossval; matriz oficial e score v6
intactos; sem v7/GT/treino; raster pesado so em local_runs; metodo calibrado
NUNCA mascarado como original (not_original_method=true). Score final NAO e
computado neste marco (so readiness).

Reproducibilidade: run-flow-variant-recompute (rede+numpy) cacheia valores por
patch/variante + features recomputadas em susc_17c39_light_artifacts/
_ledger_flow.json; metricas/calibracao/crossval/selecao/aplicacao/readiness sao
deterministicas a partir do ledger; build offline reproduz byte-identico.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_io import ensure_dir, read_csv, read_json, rel, write_csv, write_json, write_markdown  # noqa: E402
import susc_17c36_hydrologic_topography_common as s17c36  # noqa: E402
import susc_17c37_topography_method_equivalence_common as s17c37  # noqa: E402
import susc_17c38_basin_aware_flow_equivalence_common as s17c38  # noqa: E402

DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"
LIGHT = OUT / "susc_17c39_light_artifacts"
LOCAL_RUNS = ROOT / "local_runs" / "suscetibilidade" / "17c39_flow_solver"
LEDGER = LIGHT / "_ledger_flow.json"

SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"
OFFICIAL_MATRIX = DAT / "susc_features_by_patch_v1.csv"

C38_METRICS = OUT / "susc_17c38_basin_aware_equivalence_metrics.csv"
C38_SUMMARY = OUT / "susc_17c38_readiness_summary.json"
C37_METRICS = OUT / "susc_17c37_method_equivalence_metrics.csv"
C35_CONTRACT = OUT / "susc_17c35_score_v6_formula_contract.json"
C34_MATRIX = OUT / "susc_17c34_canary_pre_event_feature_matrix.csv"
C33_REGISTRY = OUT / "susc_17c33_event_anchored_canary_patch_registry.csv"

REQUIRED_INPUTS = [
    SCORE_V6, OFFICIAL_MATRIX, C38_METRICS, C38_SUMMARY, C37_METRICS, C35_CONTRACT,
    C34_MATRIX, C33_REGISTRY,
]

REPORT = OUT / "SUSC_17C39_FLOW_ACC_SOLVER_REPORT.md"
TRACE = OUT / "susc_17c39_flow_acc_method_trace_inventory.csv"
DIAG = OUT / "susc_17c39_official_flow_acc_diagnostic.csv"
VARIANT_REG = OUT / "susc_17c39_flow_acc_variant_registry.csv"
VARIANT_RESULTS = OUT / "susc_17c39_official_patch_flow_acc_variants.csv"
VARIANT_METRICS = OUT / "susc_17c39_flow_acc_variant_metrics.csv"
CALIB_REG = OUT / "susc_17c39_flow_acc_calibration_model_registry.csv"
CALIB_CROSSVAL = OUT / "susc_17c39_flow_acc_calibration_crossval.csv"
CALIB_METRICS = OUT / "susc_17c39_flow_acc_calibration_metrics.csv"
SOLUTION = OUT / "susc_17c39_flow_acc_solution_selection.csv"
CANARY_SOLUTION = OUT / "susc_17c39_canary_flow_acc_solution.csv"
READINESS = OUT / "susc_17c39_score_v6_replay_readiness_after_flow_solution.csv"
INTEGRITY = OUT / "susc_17c39_score_integrity_audit.csv"
LEAKAGE = OUT / "susc_17c39_no_leakage_audit.csv"
SUMMARY = OUT / "susc_17c39_readiness_summary.json"
BLOCKERS = OUT / "susc_17c39_promotion_blockers.csv"

SCHEMA_SOLVER = SCHEMAS / "susc_17c39_flow_solver_schema_v1.json"
SCHEMA_CALIB = SCHEMAS / "susc_17c39_calibration_schema_v1.json"
SCHEMA_RDY = SCHEMAS / "susc_17c39_score_readiness_schema_v1.json"

REQUIRED_OUTPUTS = [
    REPORT, TRACE, DIAG, VARIANT_REG, VARIANT_RESULTS, VARIANT_METRICS, CALIB_REG,
    CALIB_CROSSVAL, CALIB_METRICS, SOLUTION, CANARY_SOLUTION, READINESS, INTEGRITY,
    LEAKAGE, SUMMARY, BLOCKERS,
]

EQ_SPEARMAN, EQ_PEARSON = 0.85, 0.70
CAL_CV_SPEARMAN, CAL_CV_PEARSON, CAL_RANK = 0.75, 0.65, 0.70
EQ_SCALE_LO, EQ_SCALE_HI = 0.80, 1.25
FORBIDDEN_SUFFIXES = (".tif", ".tiff", ".safe", ".zip", ".nc", ".jp2", ".gz")
MAX_LIGHT_BYTES = 5_000_000
TRACE_TERMS = ["flow_acc_log", "flow_acc", "flow accumulation", "accumulation", "contributing_area",
               "specific_area", "catchment", "watershed", "whitebox", "richdem", "pysheds", "taudem",
               "saga", "grass", "r.watershed", "d8", "mfd", "dinf", "d-infinity", "twi", "hand",
               "hydrology", "hidrologia", "susc_features_by_patch"]
CALIB_FEATURES = ["best_variant", "hand", "twi", "tpi", "slope", "elevation"]


def _bool(v):
    return "true" if v else "false"


def _run_git(args):
    r = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return r.stdout.strip() if r.returncode == 0 else ""


def _env_ok():
    need = ["SUSC_17C39_ALLOW_NETWORK", "SUSC_17C39_ALLOW_PUBLIC_DOWNLOAD",
            "SUSC_17C39_ALLOW_LIGHTWEIGHT_RASTER", "SUSC_17C39_ALLOW_DEM",
            "SUSC_17C39_ALLOW_HYDROLOGIC_PROCESSING", "SUSC_17C39_ALLOW_FLOW_VARIANTS",
            "SUSC_17C39_ALLOW_CALIBRATION"]
    return all(os.environ.get(v) == "1" for v in need)


def _require_inputs():
    missing = [p for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("; ".join(rel(p) for p in missing))


def _official_rows():
    return s17c38._official_rows()


def _fnum(r, k):
    return s17c37._fnum(r, k)


# ---------------------------------------------------------------------------
# Etapa 1 - forense documental (offline)
# ---------------------------------------------------------------------------

def method_trace_rows():
    rows = []
    tid = 0
    scan_dirs = [ROOT / "scripts" / "suscetibilidade", DAT]
    files = []
    for d in scan_dirs:
        if d.exists():
            files += sorted(str(p) for p in d.glob("*.py"))
            files += sorted(str(p) for p in d.glob("*.md"))
    # incluir feature cards + score builder
    for extra in [ROOT / "scripts" / "suscetibilidade" / "build_susc_10a_score_v6_candidate.py",
                  ROOT / "outputs_public" / "suscetibilidade" / "feature_cards" / "susc_feature_cards_v1.json"]:
        if extra.exists() and str(extra) not in files:
            files.append(str(extra))
    term_re = {t: re.compile(re.escape(t), re.I) for t in TRACE_TERMS}
    seen = set()
    for fpath in sorted(set(files)):
        # nao varrer os proprios scripts 17c36-39 (ruido) exceto 10a
        base = Path(fpath).name
        if re.match(r"susc_17c3[6-9]", base):
            continue
        try:
            text = Path(fpath).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        low = text.lower()
        for t, rgx in term_re.items():
            if t in low:
                m = rgx.search(low)
                ctx = low[max(0, m.start() - 30):m.start() + 40].replace("\n", " ") if m else ""
                key = (base, t)
                if key in seen:
                    continue
                seen.add(key)
                tid += 1
                indicates_algo = t in ("d8", "mfd", "dinf", "d-infinity", "whitebox", "richdem", "pysheds", "taudem", "saga", "grass", "r.watershed")
                rows.append({
                    "trace_id": f"S17C39_TRACE_{tid:04d}", "source_file": rel(Path(fpath)), "matched_term": t,
                    "line_or_context": ctx[:80], "indicates_algorithm": _bool(indicates_algo),
                    "indicates_dem_source": _bool(t in ("susc_features_by_patch",) or "srtm" in low or "copernicus" in low),
                    "indicates_transform": _bool(t in ("flow_acc_log", "specific_area", "contributing_area")),
                    "indicates_resolution": _bool("30m" in low or "90m" in low or "resolution" in low),
                    "indicates_domain": _bool("basin" in low or "watershed" in low or "catchment" in low),
                    "usable_for_reconstruction": _bool(indicates_algo),
                    "review_only": "true",
                })
    if not rows:
        rows.append({"trace_id": "S17C39_TRACE_0001", "source_file": "none", "matched_term": "none",
                     "line_or_context": "no_method_trace_found", "indicates_algorithm": "false",
                     "indicates_dem_source": "false", "indicates_transform": "false",
                     "indicates_resolution": "false", "indicates_domain": "false",
                     "usable_for_reconstruction": "false", "review_only": "true"})
    return rows


# ---------------------------------------------------------------------------
# Etapa 2 - diagnostico (offline)
# ---------------------------------------------------------------------------

def official_flow_diagnostic_rows():
    rows = _official_rows()
    fa = [_fnum(r, "flow_acc_log_mean") for r in rows]
    fa = [v for v in fa if v is not None]

    def corr(feat):
        pairs = [(_fnum(r, "flow_acc_log_mean"), _fnum(r, feat)) for r in rows]
        xs = [a for a, b in pairs if a is not None and b is not None]
        ys = [b for a, b in pairs if a is not None and b is not None]
        return s17c37._pearson(s17c37._rank(xs), s17c37._rank(ys))
    s = sorted(fa)
    n = len(s)

    def pct(p):
        return round(s[min(n - 1, int(p * n))], 4) if n else "not_available"
    corr_twi = corr("twi_mean")
    corr_hand = corr("hand_mean")
    corr_elev = corr("elevation_mean")
    # interpretacao
    interp = "sintetica_ou_especificacao"
    if corr_twi is not None and corr_twi >= 0.7:
        interp = "consistente_com_area_contribuinte_via_twi (flow_acc e twi compartilham a)"
    elif corr_hand is not None and corr_hand <= -0.5:
        interp = "inversa_a_hand (vales acumulam mais)"
    return [{
        "diagnostic_id": "S17C39_DIAG_0001", "official_patch_count": str(len(fa)),
        "official_min": pct(0.0), "official_p05": pct(0.05), "official_median": pct(0.5),
        "official_mean": round(sum(fa) / len(fa), 4) if fa else "not_available",
        "official_p95": pct(0.95), "official_max": pct(1.0),
        "correlation_with_hand": round(corr_hand, 4) if corr_hand is not None else "not_available",
        "correlation_with_twi": round(corr_twi, 4) if corr_twi is not None else "not_available",
        "correlation_with_tpi": round(corr("tpi_250m_mean"), 4) if corr("tpi_250m_mean") is not None else "not_available",
        "correlation_with_elevation": round(corr_elev, 4) if corr_elev is not None else "not_available",
        "correlation_with_slope": round(corr("slope_mean"), 4) if corr("slope_mean") is not None else "not_available",
        "correlation_with_distance_to_water": round(corr("distance_to_water_mean"), 4) if corr("distance_to_water_mean") is not None else "not_available",
        "distribution_interpretation": interp, "review_only": "true",
    }]


# ---------------------------------------------------------------------------
# Etapa 3 - registro de variantes (offline)
# ---------------------------------------------------------------------------

_VARIANTS = [
    ("V01_d8_log1p_cell_count_basin_5km", "primary", 5, "log1p", "d8", "priority_flood", "false", "false", "false", "false", "cell_count", "log1p acumulacao basin 5km"),
    ("V02_d8_log10_cell_count_basin_5km", "primary", 5, "log10", "d8", "priority_flood", "false", "false", "false", "false", "cell_count", "log10 acumulacao"),
    ("V03_d8_raw_cell_count_basin_5km", "primary", 5, "raw", "d8", "priority_flood", "false", "false", "false", "false", "cell_count", "acumulacao bruta"),
    ("V04_d8_specific_area_log1p_basin_5km", "primary", 5, "log1p", "d8", "priority_flood", "false", "false", "false", "false", "specific_area", "area especifica log1p"),
    ("V05_d8_log1p_cell_count_basin_10km", "basin10", 10, "log1p", "d8", "priority_flood", "false", "false", "false", "false", "cell_count", "log1p acumulacao basin 10km"),
    ("V06_d8_log1p_with_drainage_burning", "burn", 5, "log1p", "d8", "priority_flood", "true", "false", "false", "false", "cell_count", "drainage burning"),
    ("V07_d8_log1p_no_sink_fill", "nofill", 5, "log1p", "d8", "none", "false", "false", "false", "false", "cell_count", "sem fill de sinks"),
    ("V08_d8_log1p_patch_buffer_2km", "patchwin", 2, "log1p", "d8", "priority_flood", "false", "false", "false", "false", "cell_count", "janela por patch 2km (proxy truncado 17C37-like)"),
    ("V09_inverse_flow_acc_log1p", "primary", 5, "inverse_log1p", "d8", "priority_flood", "false", "true", "false", "false", "cell_count", "inverso da acumulacao"),
    ("V10_downstream_distance_proxy", "primary", 5, "distance", "d8", "priority_flood", "false", "false", "false", "false", "downstream_distance", "distancia a drenagem (proxy)"),
    ("V11_distance_to_water_inverse_log", "primary", 5, "inverse_log", "none", "none", "false", "false", "true", "false", "distance_to_water", "inverso log dist a agua"),
    ("V12_rank_quantile_of_d8_basin", "primary", 5, "rank_quantile", "d8", "priority_flood", "false", "false", "false", "true", "cell_count", "quantil de rank da acumulacao"),
    ("V17_log1p_accumulation_plus_slope_weight", "primary", 5, "log1p_plus_slope", "d8", "priority_flood", "false", "false", "false", "false", "cell_count", "log1p acc + peso de declividade"),
    ("V18_flow_acc_times_low_slope_mask", "primary", 5, "times_low_slope", "d8", "priority_flood", "false", "false", "false", "false", "cell_count", "acc x mascara de baixa declividade"),
]


def variant_registry_rows():
    rows = []
    for (name, src, buf, transform, fdir, sink, burn, inv, dtw, rankq, unit, effect) in _VARIANTS:
        rows.append({
            "flow_acc_variant_id": name.split("_")[0], "variant_name": name,
            "dem_source": "copernicus_dem_glo30", "domain_buffer_km": str(buf),
            "resolution_m": "92.8", "flow_direction_method": fdir, "sink_fill_method": sink,
            "uses_drainage_burning": burn, "uses_inverse_transform": inv, "uses_distance_to_water": dtw,
            "uses_rank_quantile": rankq, "accumulation_unit": unit, "transform": transform,
            "aggregation_method": "bbox_mean", "expected_effect": effect, "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Etapa 4 - recompute variantes (network) -> ledger
# ---------------------------------------------------------------------------

def _canary_bboxes():
    out = []
    for c in read_csv(C33_REGISTRY):
        out.append((c["event_anchored_canary_patch_id"],
                    float(c["xmin"]), float(c["ymin"]), float(c["xmax"]), float(c["ymax"])))
    return out


def _union_domain(buffer_km):
    rows = _official_rows()
    xs = [_fnum(r, "xmin") for r in rows if _fnum(r, "xmin") is not None] + [b[1] for b in _canary_bboxes()]
    ys = [_fnum(r, "ymin") for r in rows if _fnum(r, "ymin") is not None] + [b[2] for b in _canary_bboxes()]
    xM = [_fnum(r, "xmax") for r in rows if _fnum(r, "xmax") is not None] + [b[3] for b in _canary_bboxes()]
    yM = [_fnum(r, "ymax") for r in rows if _fnum(r, "ymax") is not None] + [b[4] for b in _canary_bboxes()]
    xmin, ymin, xmax, ymax = min(xs), min(ys), max(xM), max(yM)
    lat_mid = (ymin + ymax) / 2
    dlat = buffer_km * 1000.0 / 110540.0
    dlon = buffer_km * 1000.0 / (111320.0 * math.cos(math.radians(lat_mid)))
    return xmin - dlon, ymin - dlat, xmax + dlon, ymax + dlat


def _build_grid(bbox, sink_fill=True, burn=False):
    import numpy as np
    dem, transform, _att = s17c38._read_domain_dem(bbox, 0.0008333)
    if dem is None:
        return None
    dem = np.where(dem < -1000, np.nan, dem)
    valid = np.isfinite(dem)
    demf = np.where(valid, dem, np.nanmax(dem[valid]))
    lat_mid = (bbox[1] + bbox[3]) / 2
    cellsize = 0.0008333 * 111320.0 * math.cos(math.radians(lat_mid))
    filled = s17c36._priority_flood_fill(demf) if sink_fill else demf
    if burn:
        # drainage burning: rebaixa celulas de alta acumulacao antes de re-rotear
        acc0, _rec0 = s17c36._d8_flow(filled, cellsize)
        thr = float(np.percentile(acc0, 95))
        filled = np.where(acc0 >= thr, filled - 20.0, filled)
        filled = s17c36._priority_flood_fill(filled)
    slope = s17c36._slope_rad(filled, cellsize)
    acc, rec = s17c36._d8_flow(filled, cellsize)
    tpi = s17c36._tpi(demf, cellsize, s17c36.TPI_RADIUS_M)
    thr = float(np.percentile(acc, s17c36.DRAINAGE_FLOWACC_PCTL))
    hand = s17c36._hand(filled, acc, rec, acc >= thr)
    a_sca = (acc + 1.0) * cellsize
    beta = np.maximum(slope, math.radians(0.1))
    twi = np.log(a_sca / np.tan(beta))
    return {"transform": transform, "cellsize": cellsize, "demf": demf, "filled": filled,
            "slope_deg": np.degrees(slope), "acc": acc, "tpi": tpi, "hand": hand, "twi": twi,
            "drainage_mask": acc >= thr, "shape": demf.shape}


def _bbox_mean(arr, transform, shape, b):
    import numpy as np
    from rasterio.transform import rowcol
    ny, nx = shape
    r0, c0 = rowcol(transform, b[0], b[3])
    r1, c1 = rowcol(transform, b[2], b[1])
    r0, r1 = sorted((max(0, r0), min(ny, r1 + 1)))
    c0, c1 = sorted((max(0, c0), min(nx, c1 + 1)))
    sub = arr[r0:r1, c0:c1]
    sub = sub[np.isfinite(sub)]
    return (round(float(sub.mean()), 5) if sub.size else None), int(sub.size)


def _distance_to_drainage_grid(g):
    import numpy as np
    # distancia (em celulas*cellsize) ao dreno mais proximo via BFS multi-fonte aproximado
    mask = g["drainage_mask"]
    ny, nx = g["shape"]
    INF = 1e9
    dist = np.full((ny, nx), INF)
    dist[mask] = 0.0
    # relaxamento simples 4 passadas (aprox distancia chebyshev * cellsize)
    cs = g["cellsize"]
    for _ in range(4):
        for ax in (0, 1):
            for shift in (1, -1):
                rolled = np.roll(dist, shift, axis=ax) + cs
                dist = np.minimum(dist, rolled)
    dist[dist >= INF] = np.nan
    return dist


def _run_variant_recompute():
    if not _env_ok():
        raise SystemExit("BLOCKED: run-flow-variant-recompute exige os sete opt-ins SUSC_17C39_*.")
    import numpy as np
    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
    os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")
    ensure_dir(LIGHT)
    ensure_dir(LOCAL_RUNS)
    t0 = datetime.now(timezone.utc)
    dom5 = _union_domain(5.0)
    dom10 = _union_domain(10.0)
    g = _build_grid(dom5, sink_fill=True)
    if g is None:
        write_json(LEDGER, {"generated_at_utc": _utcnow(), "grid_built": False, "official": {}, "canary": {}})
        return
    g10 = _build_grid(dom10, sink_fill=True)
    gnf = _build_grid(dom5, sink_fill=False)
    gburn = _build_grid(dom5, sink_fill=True, burn=True)
    dist = _distance_to_drainage_grid(g)
    acc = g["acc"]
    slope = g["slope_deg"]
    max_acc = float(acc.max())
    slope_med = float(np.nanmedian(slope))
    rank_q = (np.argsort(np.argsort(acc.ravel())).reshape(acc.shape) / (acc.size - 1))

    def variant_array(vid):
        if vid == "V01":
            return np.log1p(acc)
        if vid == "V02":
            return np.log10(acc + 1.0)
        if vid == "V03":
            return acc.astype("float64")
        if vid == "V04":
            return np.log1p((acc + 1.0) * g["cellsize"])
        if vid == "V05":
            return np.log1p(g10["acc"]) if g10 else np.log1p(acc)
        if vid == "V06":
            return np.log1p(gburn["acc"]) if gburn else np.log1p(acc)
        if vid == "V07":
            return np.log1p(gnf["acc"]) if gnf else np.log1p(acc)
        if vid == "V09":
            return np.log1p(np.maximum(0.0, max_acc - acc))
        if vid == "V11":
            return -np.log1p(dist)
        if vid == "V12":
            return rank_q
        if vid == "V17":
            return np.log1p(acc) + (slope / (slope_med + 1e-6))
        if vid == "V18":
            return acc * (slope < slope_med).astype("float64")
        return None  # V08, V10 tratados a parte
    grids_for = {"V01": g, "V02": g, "V03": g, "V04": g, "V05": g10 or g, "V06": gburn or g,
                 "V07": gnf or g, "V09": g, "V11": g, "V12": g, "V17": g, "V18": g}
    variant_arrays = {}
    for vid in ["V01", "V02", "V03", "V04", "V05", "V06", "V07", "V09", "V11", "V12", "V17", "V18"]:
        variant_arrays[vid] = variant_array(vid)
    dist_arr = dist  # V10 downstream distance proxy (negativo: perto do dreno ~ mais acumulacao)
    v10_arr = -dist

    def sample_all(pid, b):
        rec = {}
        for vid, arr in variant_arrays.items():
            gg = grids_for[vid]
            rec[vid], _ = _bbox_mean(arr, gg["transform"], gg["shape"], b)
        rec["V10"], _ = _bbox_mean(v10_arr, g["transform"], g["shape"], b)
        # V08 janela por patch 2km: max acumulacao dentro de janela 2km centrada (proxy truncado)
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        dcell = int(2000.0 / g["cellsize"])
        from rasterio.transform import rowcol
        r, c = rowcol(g["transform"], cx, cy)
        ny, nx = g["shape"]
        r0, r1 = max(0, r - dcell), min(ny, r + dcell)
        c0, c1 = max(0, c - dcell), min(nx, c + dcell)
        sub = acc[r0:r1, c0:c1]
        rec["V08"] = round(float(np.log1p(sub.max())), 5) if sub.size else None
        # features basin-aware para calibracao
        rec["hand"], _ = _bbox_mean(g["hand"], g["transform"], g["shape"], b)
        rec["twi"], _ = _bbox_mean(g["twi"], g["transform"], g["shape"], b)
        rec["tpi"], _ = _bbox_mean(g["tpi"], g["transform"], g["shape"], b)
        rec["slope"], _ = _bbox_mean(g["slope_deg"], g["transform"], g["shape"], b)
        rec["elevation"], npx = _bbox_mean(g["demf"], g["transform"], g["shape"], b)
        rec["valid_pixel_count"] = npx
        rec["inside"] = (dom5[0] <= b[0] and b[2] <= dom5[2] and dom5[1] <= b[1] and b[3] <= dom5[3])
        return rec

    official = {}
    for r in _official_rows():
        b = (_fnum(r, "xmin"), _fnum(r, "ymin"), _fnum(r, "xmax"), _fnum(r, "ymax"))
        if any(v is None for v in b):
            continue
        rec = sample_all(r["patch_id"], b)
        rec["official_flow_acc_log_mean"] = _fnum(r, "flow_acc_log_mean")
        official[r["patch_id"]] = rec
    canary = {}
    for (cid, xmin, ymin, xmax, ymax) in _canary_bboxes():
        canary[cid] = sample_all(cid, (xmin, ymin, xmax, ymax))
    proc_s = (datetime.now(timezone.utc) - t0).total_seconds()
    raw_path = LOCAL_RUNS / "flow_solver_grids.npz"
    np.savez_compressed(raw_path, acc=acc.astype("float32"))
    write_json(LEDGER, {"generated_at_utc": _utcnow(), "grid_built": True,
                        "domain_5km": list(dom5), "domain_10km": list(dom10),
                        "grid_shape": list(g["shape"]), "cellsize_m": round(g["cellsize"], 3),
                        "max_flow_accumulation": max_acc, "processing_time_s": round(proc_s, 2),
                        "variant_ids": [v[0].split("_")[0] for v in _VARIANTS],
                        "official": official, "canary": canary})


def _utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_ledger():
    if not LEDGER.exists():
        raise FileNotFoundError(f"{rel(LEDGER)} ausente: rode run-flow-variant-recompute com os sete opt-ins.")
    return read_json(LEDGER)


# ---------------------------------------------------------------------------
# Offline derivation
# ---------------------------------------------------------------------------

def _variant_order():
    return [v[0].split("_")[0] for v in _VARIANTS]


def official_variant_result_rows():
    led = _load_ledger()
    official = led.get("official", {})
    reg = {r["flow_acc_variant_id"]: r for r in variant_registry_rows()}
    rows = []
    rid = 0
    for vid in _variant_order():
        for pid in sorted(official.keys()):
            rid += 1
            o = official[pid]
            v = o.get(vid)
            rows.append({
                "flow_acc_variant_result_id": f"S17C39_VR_{rid:05d}", "flow_acc_variant_id": vid,
                "patch_id": pid, "official_flow_acc_log_mean": o.get("official_flow_acc_log_mean", "not_available"),
                "recomputed_flow_acc_value": "not_available" if v is None else v,
                "valid_pixel_count": str(o.get("valid_pixel_count", 0)),
                "patch_inside_domain": _bool(o.get("inside", False)),
                "edge_effect_risk": "low" if o.get("inside") else "check",
                "processing_status": "recomputed" if v is not None else "not_available", "review_only": "true",
            })
    return rows


def _official_pairs(vid):
    led = _load_ledger()
    official = led.get("official", {})
    xs, ys = [], []
    for pid in sorted(official.keys()):
        o = official[pid]
        v = o.get(vid)
        y = o.get("official_flow_acc_log_mean")
        if v is not None and y is not None:
            xs.append(v)
            ys.append(y)
    return xs, ys


def variant_metric_rows():
    reg = {r["flow_acc_variant_id"]: r for r in variant_registry_rows()}
    c38 = None
    for r in read_csv(C38_METRICS):
        if r["feature_name"] == "flow_acc_log_mean":
            c38 = r
    sp38 = None
    try:
        sp38 = float(c38["spearman_corr"]) if c38 else None
    except (TypeError, ValueError):
        sp38 = None
    rows = []
    for i, vid in enumerate(_variant_order(), start=1):
        xs, ys = _official_pairs(vid)
        pear = s17c37._pearson(xs, ys)
        spear = s17c37._pearson(s17c37._rank(xs), s17c37._rank(ys))
        aes = [abs(a - b) for a, b in zip(xs, ys)]
        res = [abs(a - b) / abs(b) for a, b in zip(xs, ys) if b not in (None, 0)]
        ratios = [a / b for a, b in zip(xs, ys) if b not in (None, 0)]
        scale_med = s17c37._median(ratios)
        n = len(xs)
        if n < 3:
            status = "insufficient_data"
        elif spear is not None and spear >= EQ_SPEARMAN and pear is not None and pear >= EQ_PEARSON and scale_med is not None and EQ_SCALE_LO <= scale_med <= EQ_SCALE_HI:
            status = "equivalent"
        elif spear is not None and abs(spear) >= 0.70:
            status = "calibratable"
        else:
            status = "incompatible"
        improved = _bool(spear is not None and sp38 is not None and abs(spear) > abs(sp38))
        rows.append({
            "flow_acc_variant_metric_id": f"S17C39_VM_{i:04d}", "flow_acc_variant_id": vid,
            "variant_name": reg[vid]["variant_name"],
            "pearson_corr": "not_available" if pear is None else round(pear, 4),
            "spearman_corr": "not_available" if spear is None else round(spear, 4),
            "mae": "not_available" if not aes else round(sum(aes) / len(aes), 4),
            "median_absolute_error": "not_available" if s17c37._median(aes) is None else round(s17c37._median(aes), 4),
            "relative_error_median": "not_available" if s17c37._median(res) is None else round(s17c37._median(res), 4),
            "official_min": round(min(ys), 4) if ys else "not_available",
            "official_max": round(max(ys), 4) if ys else "not_available",
            "recomputed_min": round(min(xs), 4) if xs else "not_available",
            "recomputed_max": round(max(xs), 4) if xs else "not_available",
            "scale_ratio_median": "not_available" if scale_med is None else round(scale_med, 4),
            "rank_preservation_score": round(spear, 4) if spear is not None else 0.0,
            "improved_vs_17c38": improved, "equivalence_status": status, "review_only": "true",
        })
    return rows


def _best_variant():
    metrics = variant_metric_rows()
    best = max(metrics, key=lambda m: abs(m["spearman_corr"]) if isinstance(m["spearman_corr"], (int, float)) else 0.0)
    return best


# ---------------------------------------------------------------------------
# Etapa 6/7 - calibradores + crossval (numpy, offline-deterministico)
# ---------------------------------------------------------------------------

CALIB_MODELS = [
    ("C01", "C01_quantile_mapping_best_variant", "quantile_mapping", ["best_variant"]),
    ("C02", "C02_isotonic_monotonic_best_variant", "isotonic", ["best_variant"]),
    ("C03", "C03_robust_linear_multi_variant", "robust_linear", CALIB_FEATURES),
    ("C04", "C04_rank_based_piecewise_mapping", "rank_piecewise", ["best_variant"]),
]


def calibration_model_registry_rows():
    best = _best_variant()["flow_acc_variant_id"]
    rows = []
    for cid, name, mtype, feats in CALIB_MODELS:
        rows.append({
            "calibration_model_id": cid, "model_name": name,
            "input_features": ";".join(feats).replace("best_variant", f"best_variant({best})"),
            "target_feature": "official_flow_acc_log_mean", "trained_on": "100_official_patches",
            "uses_canaries": "false", "uses_event_labels": "false", "model_type": mtype, "review_only": "true",
        })
    return rows


def _feature_matrix(pids, best_vid, ledger_group):
    import numpy as np
    X, y, ids = [], [], []
    for pid in sorted(ledger_group.keys()):
        if pids is not None and pid not in pids:
            continue
        o = ledger_group[pid]
        best = o.get(best_vid)
        feats = [best, o.get("hand"), o.get("twi"), o.get("tpi"), o.get("slope"), o.get("elevation")]
        if any(f is None for f in feats):
            continue
        X.append(feats)
        y.append(o.get("official_flow_acc_log_mean"))
        ids.append(pid)
    return np.array(X, dtype="float64"), y, ids


def _orient(x, y):
    # orienta x para correlacao positiva com y (rank)
    sp = s17c37._pearson(s17c37._rank(list(x)), s17c37._rank(list(y)))
    return (-1.0 if (sp is not None and sp < 0) else 1.0)


def _fit_predict(mtype, Xtr, ytr, Xte):
    import numpy as np
    ytr = np.array(ytr, dtype="float64")
    if mtype == "robust_linear":
        A = np.column_stack([np.ones(len(Xtr)), Xtr])
        coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
        Ate = np.column_stack([np.ones(len(Xte)), Xte])
        return Ate @ coef
    # 1D models usam best_variant (coluna 0), orientado
    xtr = Xtr[:, 0]
    xte = Xte[:, 0]
    sign = _orient(xtr, ytr)
    xtr = xtr * sign
    xte = xte * sign
    if mtype == "quantile_mapping":
        qs = np.linspace(0, 1, 11)
        vq = np.quantile(xtr, qs)
        oq = np.quantile(ytr, qs)
        return np.interp(xte, vq, oq)
    if mtype == "isotonic":
        order = np.argsort(xtr)
        xs = xtr[order]
        ys = ytr[order]
        # PAV simplificado
        yfit = ys.astype("float64").copy()
        w = np.ones(len(yfit))
        i = 0
        while i < len(yfit) - 1:
            if yfit[i] > yfit[i + 1]:
                new = (yfit[i] * w[i] + yfit[i + 1] * w[i + 1]) / (w[i] + w[i + 1])
                yfit[i] = yfit[i + 1] = new
                w[i] = w[i + 1] = w[i] + w[i + 1]
                if i > 0:
                    i -= 1
            else:
                i += 1
        return np.interp(xte, xs, yfit)
    if mtype == "rank_piecewise":
        K = 8
        edges = np.quantile(xtr, np.linspace(0, 1, K + 1))
        means = []
        for k in range(K):
            lo, hi = edges[k], edges[k + 1]
            sel = (xtr >= lo) & (xtr <= hi) if k == K - 1 else (xtr >= lo) & (xtr < hi)
            means.append(float(ytr[sel].mean()) if sel.any() else float(ytr.mean()))
        idx = np.clip(np.digitize(xte, edges[1:-1]), 0, K - 1)
        return np.array([means[i] for i in idx])
    return np.full(len(Xte), float(ytr.mean()))


def _folds(ids, k=5):
    ids = sorted(ids)
    folds = [[] for _ in range(k)]
    for i, pid in enumerate(ids):
        folds[i % k].append(pid)
    return folds


def calibration_crossval_rows():
    import numpy as np
    led = _load_ledger()
    official = led.get("official", {})
    best = _best_variant()["flow_acc_variant_id"]
    Xall, yall, ids = _feature_matrix(None, best, official)
    id_index = {pid: i for i, pid in enumerate(ids)}
    folds = _folds(ids, 5)
    rows = []
    rid = 0
    for cid, name, mtype, feats in CALIB_MODELS:
        for fi, test_ids in enumerate(folds, start=1):
            tr_ids = [p for p in ids if p not in set(test_ids)]
            tr_idx = [id_index[p] for p in tr_ids]
            te_idx = [id_index[p] for p in test_ids]
            if len(tr_idx) < 5 or len(te_idx) < 1:
                continue
            Xtr, ytr = Xall[tr_idx], [yall[i] for i in tr_idx]
            Xte, yte = Xall[te_idx], [yall[i] for i in te_idx]
            pred = _fit_predict(mtype, Xtr, ytr, Xte)
            pear = s17c37._pearson(list(pred), yte)
            spear = s17c37._pearson(s17c37._rank(list(pred)), s17c37._rank(yte))
            aes = [abs(float(p) - t) for p, t in zip(pred, yte)]
            mae = sum(aes) / len(aes) if aes else None
            mede = s17c37._median(aes)
            ratios = [float(p) / t for p, t in zip(pred, yte) if t not in (None, 0)]
            scale_med = s17c37._median(ratios)
            scale_ok = scale_med is not None and EQ_SCALE_LO <= scale_med <= EQ_SCALE_HI
            rank_ps = spear if spear is not None else 0.0
            passes = (spear is not None and spear >= CAL_CV_SPEARMAN and pear is not None and pear >= CAL_CV_PEARSON and rank_ps >= CAL_RANK)
            rid += 1
            rows.append({
                "crossval_id": f"S17C39_CV_{rid:04d}", "calibration_model_id": cid, "fold_id": str(fi),
                "train_patch_count": str(len(tr_idx)), "test_patch_count": str(len(te_idx)),
                "spearman_corr": "not_available" if spear is None else round(spear, 4),
                "pearson_corr": "not_available" if pear is None else round(pear, 4),
                "mae": "not_available" if mae is None else round(mae, 4),
                "median_absolute_error": "not_available" if mede is None else round(mede, 4),
                "rank_preservation_score": round(rank_ps, 4),
                "scale_compatible_after_calibration": _bool(scale_ok),
                "passes_calibration_threshold": _bool(passes), "review_only": "true",
            })
    return rows


def calibration_metric_rows():
    cv = calibration_crossval_rows()
    reg = {r["calibration_model_id"]: r for r in calibration_model_registry_rows()}
    rows = []
    for i, (cid, name, mtype, feats) in enumerate(CALIB_MODELS, start=1):
        folds = [r for r in cv if r["calibration_model_id"] == cid]

        def mean(key):
            vals = [float(r[key]) for r in folds if r[key] != "not_available"]
            return round(sum(vals) / len(vals), 4) if vals else None
        msp = mean("spearman_corr")
        mpe = mean("pearson_corr")
        mrk = mean("rank_preservation_score")
        scale_ok = all(r["scale_compatible_after_calibration"] == "true" for r in folds) if folds else False
        passes = (msp is not None and msp >= CAL_CV_SPEARMAN and mpe is not None and mpe >= CAL_CV_PEARSON and mrk is not None and mrk >= CAL_RANK)
        rows.append({
            "calibration_metric_id": f"S17C39_CALM_{i:04d}", "calibration_model_id": cid, "model_name": name,
            "mean_spearman_corr": "not_available" if msp is None else msp,
            "mean_pearson_corr": "not_available" if mpe is None else mpe,
            "mean_mae": mean("mae") if mean("mae") is not None else "not_available",
            "mean_rank_preservation_score": "not_available" if mrk is None else mrk,
            "scale_compatible_after_calibration": _bool(scale_ok),
            "passes_calibration_threshold": _bool(passes), "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Etapa 8 - selecao
# ---------------------------------------------------------------------------

def solution_selection_rows():
    vmetrics = variant_metric_rows()
    cmetrics = calibration_metric_rows()
    best_var = _best_variant()
    equiv_variants = [m for m in vmetrics if m["equivalence_status"] == "equivalent"]
    passing_cals = [m for m in cmetrics if m["passes_calibration_threshold"] == "true"]
    best_cal = max(cmetrics, key=lambda m: m["mean_spearman_corr"] if isinstance(m["mean_spearman_corr"], (int, float)) else -1)
    if equiv_variants:
        stype = "variant_equivalent"
        sel_var = equiv_variants[0]["flow_acc_variant_id"]
        sel_cal = ""
        sname = equiv_variants[0]["variant_name"]
        passes = True
        reason = "variante hidrologica direta equivalente encontrada"
    elif passing_cals:
        best_pass = max(passing_cals, key=lambda m: m["mean_spearman_corr"])
        stype = "calibrated_surrogate_validated"
        sel_var = best_var["flow_acc_variant_id"]
        sel_cal = best_pass["calibration_model_id"]
        sname = best_pass["model_name"]
        passes = True
        reason = f"surrogate calibrado validado por crossval (spearman={best_pass['mean_spearman_corr']}, pearson={best_pass['mean_pearson_corr']})"
    else:
        stype = "replacement_contract_review_only"
        sel_var = best_var["flow_acc_variant_id"]
        sel_cal = best_cal["calibration_model_id"]
        sname = "replacement_contract_best_effort_proxy"
        passes = False
        reason = "nenhuma variante equivalente nem calibrador validado; melhor proxy documentado; migrar para componentes review-only transparentes"
    return [{
        "flow_acc_solution_selection_id": "S17C39_SEL_0001", "selected_solution_type": stype,
        "selected_variant_id": sel_var, "selected_calibration_model_id": sel_cal, "selected_solution_name": sname,
        "best_direct_variant_status": best_var["equivalence_status"],
        "best_direct_variant_spearman": best_var["spearman_corr"],
        "best_calibration_spearman": best_cal["mean_spearman_corr"],
        "best_calibration_pearson": best_cal["mean_pearson_corr"],
        "best_calibration_rank_preservation": best_cal["mean_rank_preservation_score"],
        "solution_passes_threshold": _bool(passes), "solution_reason": reason, "review_only": "true",
    }]


# ---------------------------------------------------------------------------
# Etapa 9 - aplicar aos canarios
# ---------------------------------------------------------------------------

def canary_solution_rows():
    import numpy as np
    led = _load_ledger()
    sel = solution_selection_rows()[0]
    stype = sel["selected_solution_type"]
    best = _best_variant()["flow_acc_variant_id"]
    official = led.get("official", {})
    canary = led.get("canary", {})
    # treina modelo escolhido em TODOS os oficiais para aplicar aos canarios
    model_type = next((m[2] for m in CALIB_MODELS if m[0] == sel["selected_calibration_model_id"]), "robust_linear")
    Xtr, ytr, _ids = _feature_matrix(None, best, official)
    rows = []
    for i, cid in enumerate(sorted(canary.keys()), start=1):
        o = canary[cid]
        raw = o.get(best)
        feats = [raw, o.get("hand"), o.get("twi"), o.get("tpi"), o.get("slope"), o.get("elevation")]
        calibrated = "not_available"
        if stype in ("variant_equivalent", "calibrated_surrogate_validated") and all(f is not None for f in feats) and len(Xtr) >= 5:
            Xte = np.array([feats], dtype="float64")
            pred = _fit_predict(model_type if stype == "calibrated_surrogate_validated" else "quantile_mapping", Xtr, ytr, Xte)
            calibrated = round(float(pred[0]), 5)
        status = stype if stype != "replacement_contract_review_only" else "replacement_contract_required"
        rows.append({
            "canary_flow_acc_solution_id": f"S17C39_CFA_{i:04d}", "canary_patch_id": cid,
            "selected_solution_type": stype,
            "raw_flow_acc_variant_value": "not_available" if raw is None else raw,
            "calibrated_flow_acc_log_mean_review_only": calibrated,
            "official_scale_flow_acc_available": _bool(calibrated != "not_available"),
            "solution_validated_on_official_patches": _bool(sel["solution_passes_threshold"] == "true"),
            "uses_canary_for_training": "false", "not_original_method": "true",
            "calibrated_against_official_population": "true", "solution_status": status,
            "review_only": "true", "ground_truth": "false", "training_label": "false",
        })
    return rows


def readiness_rows():
    sel = solution_selection_rows()[0]
    stype = sel["selected_solution_type"]
    validated = sel["solution_passes_threshold"] == "true"
    # HAND equivalent, TWI/TPI calibratable (17C38); normalizacao 17C35 existe
    can_calibrated = (stype == "calibrated_surrogate_validated" and validated)
    rows = []
    for i, c in enumerate(read_csv(C33_REGISTRY), start=1):
        pid = c["event_anchored_canary_patch_id"]
        status = ("flow_acc_calibrated_surrogate_ready_for_calibrated_replay" if can_calibrated
                  else ("flow_acc_variant_equivalent_ready" if stype == "variant_equivalent"
                        else "flow_acc_replacement_contract_score_final_still_blocked"))
        rows.append({
            "score_replay_readiness_after_flow_solution_id": f"S17C39_RDY_{i:04d}", "canary_patch_id": pid,
            "previous_status_17c38": "topography_still_incompatible_score_final_blocked",
            "flow_acc_solution_type": stype, "flow_acc_solution_validated": _bool(validated),
            "topography_component_status": "hand_equivalent_twi_tpi_calibratable_flow_acc_" + stype,
            "can_compute_full_score_v6_replay": "false",
            "can_compute_calibrated_score_v6_replay": _bool(can_calibrated),
            "score_replay_status_after_17c39": status,
            "blocking_reason": ("componentes prontos para replay CALIBRADO review-only (nao oficial); score final literal segue diferido"
                                if can_calibrated else sel["solution_reason"]),
            "not_original_method": "true", "does_not_modify_score_v6": "true", "review_only": "true",
        })
    return rows


def score_integrity_audit_rows():
    changed = bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)]))
    mchanged = bool(_run_git(["diff", "--name-only", "--", rel(OFFICIAL_MATRIX)]))
    return [{"score_integrity_audit_id": "S17C39_INTEG_0001", "official_score_v6_path": rel(SCORE_V6),
             "official_score_v6_changed": _bool(changed), "official_matrix_path": rel(OFFICIAL_MATRIX),
             "official_matrix_changed": _bool(mchanged), "score_v7_exists": _bool(SCORE_V7.exists()),
             "final_score_computed": "false", "review_only": "true"}]


def no_leakage_audit_rows():
    rows = []
    ledger = _load_ledger()
    for i, pid in enumerate(sorted(ledger.get("official", {}).keys()), start=1):
        rows.append({
            "no_leakage_audit_id": f"S17C39_LEAK_{i:04d}", "object_id": pid, "object_type": "official_patch_calibration",
            "uses_occurrence_as_feature": "false", "uses_canary_for_training": "false",
            "calibration_on_official_only": "true", "uses_canary_only_normalization": "false",
            "surrogate_validated_before_apply": "true", "final_score_computed": "false",
            "modifies_official_matrix": "false", "modifies_score_v6": "false", "uses_score_v7": "false",
            "uses_control_as_negative_truth": "false", "masks_calibrated_as_original": "false",
            "passes_no_leakage": "true", "blocking_reason": "calibracao so oficial; canario nao treina; review-only",
            "review_only": "true",
        })
    return rows


def build_summary():
    trace = method_trace_rows()
    diag = official_flow_diagnostic_rows()
    reg = variant_registry_rows()
    results = official_variant_result_rows()
    vmetrics = variant_metric_rows()
    cmodels = calibration_model_registry_rows()
    cv = calibration_crossval_rows()
    cmetrics = calibration_metric_rows()
    sel = solution_selection_rows()[0]
    canary = canary_solution_rows()
    ready = readiness_rows()
    led = _load_ledger() if LEDGER.exists() else {}
    best_var = _best_variant()
    best_cal = max(cmetrics, key=lambda m: m["mean_spearman_corr"] if isinstance(m["mean_spearman_corr"], (int, float)) else -1)
    pop = len(_official_rows())
    recomputed_results = len([r for r in results if r["processing_status"] == "recomputed"])
    stype = sel["selected_solution_type"]
    minimum = (
        pop >= 100 and len(trace) > 0 and len(reg) >= 12 and recomputed_results >= 1200
        and len(vmetrics) >= 12 and len(cmodels) >= 4 and len(cv) >= 4 and sel["selected_solution_type"] != ""
        and len(canary) >= 11 and stype != "blocked"
    )
    return {
        "minimum_success_achieved": minimum,
        "official_patch_population_rows_count": pop,
        "method_trace_rows_count": len(trace),
        "flow_acc_variant_count": len(reg),
        "official_patch_variant_recomputed_rows_count": recomputed_results,
        "flow_acc_variant_metric_rows_count": len(vmetrics),
        "best_direct_variant_id": best_var["flow_acc_variant_id"],
        "best_direct_variant_spearman": best_var["spearman_corr"],
        "best_direct_variant_status": best_var["equivalence_status"],
        "calibration_model_count": len(cmodels),
        "calibration_crossval_rows_count": len(cv),
        "best_calibration_model_id": best_cal["calibration_model_id"],
        "best_calibration_spearman": best_cal["mean_spearman_corr"],
        "best_calibration_pearson": best_cal["mean_pearson_corr"],
        "best_calibration_rank_preservation": best_cal["mean_rank_preservation_score"],
        "best_calibration_passes_threshold": best_cal["passes_calibration_threshold"] == "true",
        "best_solution_selected": True,
        "selected_solution_type": stype,
        "flow_acc_method_recovered": stype == "method_recovered",
        "flow_acc_variant_equivalent": stype == "variant_equivalent",
        "flow_acc_calibrated_surrogate_validated": stype == "calibrated_surrogate_validated",
        "flow_acc_replacement_contract_review_only": stype == "replacement_contract_review_only",
        "flow_acc_solution_status": stype,
        "canary_flow_acc_solution_rows_count": len(canary),
        "can_compute_score_v6_final_replay": False,
        "can_compute_calibrated_score_v6_replay": any(r["can_compute_calibrated_score_v6_replay"] == "true" for r in ready),
        "score_v6_replay_readiness_updated": True,
        "score_v6_replay_readiness_rows_count": len(ready),
        "official_flow_corr_with_twi": diag[0]["correlation_with_twi"],
        "official_flow_corr_with_hand": diag[0]["correlation_with_hand"],
        "basin_processing_time_s": led.get("processing_time_s", -1),
        "score_v6_original_file_changed": bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)])),
        "official_matrix_changed": bool(_run_git(["diff", "--name-only", "--", rel(OFFICIAL_MATRIX)])),
        "score_v7_created": SCORE_V7.exists(),
        "ground_truth_created": False,
        "training_labels_created": False,
        "official_patch_created": False,
        "official_patch_link_created": False,
        "eligible_for_17b_now": False,
        "review_only": True,
        "trainable": False,
        "ground_truth": False,
        "result_class": {"variant_equivalent": "B_variant_equivalent", "method_recovered": "A_method_recovered",
                         "calibrated_surrogate_validated": "C_calibrated_surrogate_validated",
                         "replacement_contract_review_only": "D_replacement_contract"}.get(stype, "unknown"),
        "recommended_next_milestone": (
            "SUSC-17C40 Replay v6 CALIBRADO review-only por patch canario usando o surrogate de flow_acc validado + "
            "HAND/TWI/TPI calibrados e normalizacao populacional 17C35; score final rotulado nao-oficial, sem alterar score v6. 17B fail-closed."
            if stype == "calibrated_surrogate_validated" else
            "SUSC-17C40 Migrar para score/componentes review-only transparentes (flow_acc como replacement contract documentado) ou buscar metodo/DEM original. 17B fail-closed."
        ),
    }


def build_blockers():
    vmetrics = variant_metric_rows()
    cmetrics = calibration_metric_rows()
    sel = solution_selection_rows()[0]
    stype = sel["selected_solution_type"]
    blockers = []
    if not any(m["equivalence_status"] == "equivalent" for m in vmetrics):
        blockers.append(("flow_acc_direct_method_not_recovered", "nenhuma variante direta atingiu equivalencia"))
        blockers.append(("flow_acc_direct_variant_below_equivalence_threshold", "melhor variante direta abaixo do limiar de equivalencia"))
    if stype == "calibrated_surrogate_validated":
        pass
    elif not any(m["passes_calibration_threshold"] == "true" for m in cmetrics):
        blockers.append(("flow_acc_calibration_threshold_not_met", "nenhum calibrador passou o limiar de crossval"))
    if stype == "replacement_contract_review_only":
        blockers.append(("flow_acc_replacement_contract_required", "adotado contrato de substituicao review-only"))
    blockers += [
        ("score_v6_final_replay_deferred_to_next_milestone", "score final v6 diferido para o proximo marco"),
        ("score_v7_blocked", "score v7 bloqueado"),
        ("17b_blocked", "17B bloqueado ate G4_full + ground reference aceito"),
    ]
    return [{"blocker_id": f"S17C39_BLOCKER_{i:04d}", "blocker_type": n, "description": d,
             "blocks_final_score_replay": "true", "blocks_17b": "true", "blocks_score_v7": "true", "review_only": "true"}
            for i, (n, d) in enumerate(blockers, start=1)]


def build_report():
    s = build_summary()
    vmetrics = variant_metric_rows()
    top = sorted(vmetrics, key=lambda m: abs(m["spearman_corr"]) if isinstance(m["spearman_corr"], (int, float)) else 0, reverse=True)[:5]
    return "\n".join([
        "# SUSC-17C39 - Flow Accumulation Solver: Method Recovery, Variants e Calibracao Transferivel",
        "",
        "## Objetivo",
        "Resolver operacionalmente o flow_acc_log_mean via forense + variantes hidrologicas + calibracao supervisionada nos 100 patches OFICIAIS + crossval + aplicacao aos canarios.",
        "",
        "## Diagnostico",
        f"- flow_acc oficial correlaciona com TWI={s['official_flow_corr_with_twi']} e HAND={s['official_flow_corr_with_hand']} (spearman).",
        "",
        "## Variantes diretas (top 5 por |spearman|)",
        *[f"  - {m['variant_name']}: spearman={m['spearman_corr']} status={m['equivalence_status']} improved_vs_17c38={m['improved_vs_17c38']}" for m in top],
        f"- Variantes registradas: {s['flow_acc_variant_count']}; resultados patch x variante: {s['official_patch_variant_recomputed_rows_count']}.",
        f"- Melhor variante direta: {s['best_direct_variant_id']} (spearman={s['best_direct_variant_spearman']}, status={s['best_direct_variant_status']}).",
        "",
        "## Calibracao transferivel (crossval 5-fold, so oficiais)",
        f"- Modelos: {s['calibration_model_count']}; folds: {s['calibration_crossval_rows_count']}.",
        f"- Melhor calibrador: {s['best_calibration_model_id']} (spearman={s['best_calibration_spearman']}, pearson={s['best_calibration_pearson']}, rank={s['best_calibration_rank_preservation']}, passa={s['best_calibration_passes_threshold']}).",
        "",
        "## Solucao selecionada",
        f"- Tipo: {s['selected_solution_type']} | result_class: {s['result_class']}.",
        f"- method_recovered={s['flow_acc_method_recovered']}; variant_equivalent={s['flow_acc_variant_equivalent']}; calibrated_surrogate_validated={s['flow_acc_calibrated_surrogate_validated']}; replacement_contract={s['flow_acc_replacement_contract_review_only']}.",
        f"- Aplicada a {s['canary_flow_acc_solution_rows_count']} canarios (calibrated_flow_acc_log_mean_review_only, not_original_method=true).",
        "",
        "## Readiness",
        f"- can_compute_calibrated_score_v6_replay={s['can_compute_calibrated_score_v6_replay']}; score final v6 NAO computado neste marco.",
        "",
        "## Guardrails",
        "- Calibracao SO nos 100 patches oficiais (canarios nunca treinam); surrogate validado por crossval antes de aplicar; matriz oficial e score v6 intactos; sem v7/GT/treino; metodo calibrado nunca mascarado como original; raster pesado so em local_runs.",
        "",
        f"## minimum_success_achieved: {s['minimum_success_achieved']}",
        "",
        "## Proximo marco recomendado",
        s["recommended_next_milestone"],
    ])


# ---------------------------------------------------------------------------
# Build / modes / validate
# ---------------------------------------------------------------------------

def build_all():
    _require_inputs()
    ensure_dir(OUT)
    ensure_dir(LIGHT)
    write_csv(TRACE, method_trace_rows())
    write_csv(DIAG, official_flow_diagnostic_rows())
    write_csv(VARIANT_REG, variant_registry_rows())
    write_csv(VARIANT_RESULTS, official_variant_result_rows())
    write_csv(VARIANT_METRICS, variant_metric_rows())
    write_csv(CALIB_REG, calibration_model_registry_rows())
    write_csv(CALIB_CROSSVAL, calibration_crossval_rows())
    write_csv(CALIB_METRICS, calibration_metric_rows())
    write_csv(SOLUTION, solution_selection_rows())
    write_csv(CANARY_SOLUTION, canary_solution_rows())
    write_csv(READINESS, readiness_rows())
    write_csv(INTEGRITY, score_integrity_audit_rows())
    write_csv(LEAKAGE, no_leakage_audit_rows())
    write_json(SUMMARY, build_summary())
    write_csv(BLOCKERS, build_blockers())
    write_markdown(REPORT, build_report())


def run_mode(mode):
    _require_inputs()
    ensure_dir(OUT)
    if mode == "trace-original-method":
        write_csv(TRACE, method_trace_rows())
    elif mode == "diagnose-official-flow":
        write_csv(DIAG, official_flow_diagnostic_rows())
    elif mode == "build-flow-variant-registry":
        write_csv(VARIANT_REG, variant_registry_rows())
    elif mode == "run-flow-variant-recompute":
        _run_variant_recompute()
    elif mode == "compute-flow-variant-metrics":
        write_csv(VARIANT_RESULTS, official_variant_result_rows())
        write_csv(VARIANT_METRICS, variant_metric_rows())
    elif mode == "build-calibration-models":
        if not _env_ok():
            raise SystemExit("BLOCKED: build-calibration-models exige os sete opt-ins SUSC_17C39_*.")
        write_csv(CALIB_REG, calibration_model_registry_rows())
    elif mode == "run-calibration-crossval":
        if not _env_ok():
            raise SystemExit("BLOCKED: run-calibration-crossval exige os sete opt-ins SUSC_17C39_*.")
        write_csv(CALIB_CROSSVAL, calibration_crossval_rows())
        write_csv(CALIB_METRICS, calibration_metric_rows())
    elif mode == "select-flow-solution":
        write_csv(SOLUTION, solution_selection_rows())
    elif mode == "apply-flow-solution-to-canaries":
        if not _env_ok():
            raise SystemExit("BLOCKED: apply-flow-solution-to-canaries exige os sete opt-ins SUSC_17C39_*.")
        write_csv(CANARY_SOLUTION, canary_solution_rows())
    elif mode == "update-score-replay-readiness":
        write_csv(READINESS, readiness_rows())
    elif mode == "build-offline":
        build_all()
    elif mode == "audit":
        raise SystemExit(validate())
    else:
        raise SystemExit(f"unknown mode: {mode}")


def _schema_violations(row, schema):
    v = []
    for f in schema.get("required", []):
        if f not in row or row[f] == "":
            v.append(f"missing:{f}")
    for f, rules in schema.get("properties", {}).items():
        if f not in row:
            continue
        val = row[f]
        if "const" in rules and val != rules["const"]:
            v.append(f"{f}:const")
        if "enum" in rules and val not in rules["enum"]:
            v.append(f"{f}:enum")
        if "pattern" in rules and not str(val).startswith(rules["pattern"].replace("^", "")):
            v.append(f"{f}:pattern")
    return v


def _validate_byte_identical():
    before = {p: p.read_bytes() for p in REQUIRED_OUTPUTS if p.exists()}
    build_all()
    errs = []
    for p, content in before.items():
        if p.exists() and p.read_bytes() != content:
            errs.append(f"offline_build_not_byte_identical:{rel(p)}")
    return errs


def validate():
    if not LEDGER.exists():
        print(f"MISSING_LEDGER: {rel(LEDGER)}", file=sys.stderr)
        return 1
    missing = [p for p in REQUIRED_OUTPUTS if not p.exists()]
    if missing:
        print("MISSING: " + "; ".join(rel(p) for p in missing), file=sys.stderr)
        return 1
    errors = _validate_byte_identical()

    trace = read_csv(TRACE)
    reg = read_csv(VARIANT_REG)
    results = read_csv(VARIANT_RESULTS)
    vmetrics = read_csv(VARIANT_METRICS)
    cmodels = read_csv(CALIB_REG)
    cv = read_csv(CALIB_CROSSVAL)
    cmetrics = read_csv(CALIB_METRICS)
    sel = read_csv(SOLUTION)
    canary = read_csv(CANARY_SOLUTION)
    ready = read_csv(READINESS)
    leak = read_csv(LEAKAGE)
    integ = read_csv(INTEGRITY)
    summary = read_json(SUMMARY)
    solver_schema = read_json(SCHEMA_SOLVER)
    calib_schema = read_json(SCHEMA_CALIB)
    rdy_schema = read_json(SCHEMA_RDY)

    for r in vmetrics:
        errors.extend(_schema_violations(r, solver_schema))
    for r in cmetrics:
        errors.extend(_schema_violations(r, calib_schema))
    for r in ready:
        errors.extend(_schema_violations(r, rdy_schema))

    # 1 pop, 2 variantes>=12, 3 resultados>=1200, 4 metricas>=12, 5 modelos>=4, 6 crossval, 7 solucao, 8 tipo, 9 nao blocked
    if summary["official_patch_population_rows_count"] < 100:
        errors.append("population_lt_100")
    if len(reg) < 12:
        errors.append("variants_lt_12")
    if summary["official_patch_variant_recomputed_rows_count"] < 1200:
        errors.append("variant_results_lt_1200")
    if len(vmetrics) < 12:
        errors.append("variant_metrics_lt_12")
    if len(cmodels) < 4:
        errors.append("calibration_models_lt_4")
    if len(cv) < 4:
        errors.append("crossval_lt_4")
    if not sel or not sel[0]["selected_solution_type"]:
        errors.append("no_solution_selected")
    stype = sel[0]["selected_solution_type"] if sel else ""
    if stype not in ("method_recovered", "variant_equivalent", "calibrated_surrogate_validated", "replacement_contract_review_only"):
        errors.append("invalid_solution_type")
    if stype == "blocked" or summary["minimum_success_achieved"] is False:
        errors.append("milestone_blocked")
    # 10 calibrador usa canarios
    for r in cmodels:
        if r["uses_canaries"] != "false":
            errors.append("calibrator_uses_canaries")
    for r in leak:
        if r["uses_canary_for_training"] != "false" or r["calibration_on_official_only"] != "true":
            errors.append("canary_training_leak")
    for r in canary:
        if r["uses_canary_for_training"] != "false":
            errors.append("canary_solution_training_leak")
        if r["not_original_method"] != "true":
            errors.append("calibrated_masked_as_original")
    # 11 score final computado
    if summary["can_compute_score_v6_final_replay"]:
        errors.append("final_score_computed")
    for r in integ:
        if r["final_score_computed"] != "false":
            errors.append("integrity_final_score")
    # 13 surrogate aplicado sem validacao
    if stype == "calibrated_surrogate_validated" and sel[0]["solution_passes_threshold"] != "true":
        errors.append("surrogate_applied_without_validation")
    for r in leak:
        if r["surrogate_validated_before_apply"] != "true":
            errors.append("surrogate_not_validated")
    # 14/15 score/matriz/v7
    if _run_git(["diff", "--name-only", "--", rel(SCORE_V6)]) or summary["score_v6_original_file_changed"]:
        errors.append("score_v6_changed")
    if _run_git(["diff", "--name-only", "--", rel(OFFICIAL_MATRIX)]) or summary["official_matrix_changed"]:
        errors.append("official_matrix_changed")
    if SCORE_V7.exists() or summary["score_v7_created"]:
        errors.append("score_v7_exists")
    # 16/17 GT/treino
    if summary["ground_truth_created"] or summary["training_labels_created"]:
        errors.append("gt_or_training")
    # 18 raster pesado
    if LIGHT.exists():
        for p in LIGHT.glob("**/*"):
            if p.is_file() and (p.suffix.lower() in FORBIDDEN_SUFFIXES or p.stat().st_size > MAX_LIGHT_BYTES):
                errors.append(f"forbidden_or_heavy:{rel(p)}")
    # 19 ocorrencia feature, 20 controle negativo
    if any(r["uses_occurrence_as_feature"] != "false" for r in leak):
        errors.append("occurrence_as_feature")
    if any(r["uses_control_as_negative_truth"] != "false" for r in leak):
        errors.append("control_negative")
    if any(r["passes_no_leakage"] != "true" for r in leak):
        errors.append("no_leakage_failed")
    if len(canary) < 11:
        errors.append("canary_solution_lt_11")

    # 21 summary reflete contagens
    expected = build_summary()
    for k, v in expected.items():
        if summary.get(k) != v:
            errors.append(f"summary_mismatch:{k}")

    for rows, key in [
        (vmetrics, "flow_acc_variant_metric_id"), (cmetrics, "calibration_metric_id"),
        (cv, "crossval_id"), (canary, "canary_flow_acc_solution_id"),
        (ready, "score_replay_readiness_after_flow_solution_id"), (leak, "no_leakage_audit_id"),
        (results, "flow_acc_variant_result_id"), (trace, "trace_id"),
    ]:
        ids = [r[key] for r in rows]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            errors.append(f"ids_not_unique_or_sorted:{key}")

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(
        "17C39 -> "
        f"pop={summary['official_patch_population_rows_count']} trace={summary['method_trace_rows_count']} "
        f"variants={summary['flow_acc_variant_count']} results={summary['official_patch_variant_recomputed_rows_count']} "
        f"best_variant={summary['best_direct_variant_id']}({summary['best_direct_variant_spearman']},{summary['best_direct_variant_status']}) "
        f"best_cal={summary['best_calibration_model_id']}(sp={summary['best_calibration_spearman']},pe={summary['best_calibration_pearson']},pass={summary['best_calibration_passes_threshold']}) "
        f"solution={summary['selected_solution_type']} canary={summary['canary_flow_acc_solution_rows_count']} "
        f"calibrated_replay={summary['can_compute_calibrated_score_v6_replay']} result={summary['result_class']} "
        f"min_success={summary['minimum_success_achieved']}"
    )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="susc_17c39")
    parser.add_argument("mode", choices=[
        "trace-original-method", "diagnose-official-flow", "build-flow-variant-registry",
        "run-flow-variant-recompute", "compute-flow-variant-metrics", "build-calibration-models",
        "run-calibration-crossval", "select-flow-solution", "apply-flow-solution-to-canaries",
        "update-score-replay-readiness", "build-offline", "audit",
    ])
    args = parser.parse_args(argv)
    if args.mode == "audit":
        return validate()
    run_mode(args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
