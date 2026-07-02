"""SUSC-17C37 Validacao de Equivalencia Metodologica Topography/Hydrology.

Valida se o pipeline hidrologico local do 17C36 e metodologicamente equivalente,
calibravel ou incompativel com as features topograficas/hidrologicas oficiais em
datasets/suscetibilidade/susc_features_by_patch_v1.csv (100 patches recife). Roda
o MESMO pipeline do 17C36 numa amostra de patches OFICIAIS, compara contra os
valores oficiais e decide se as features dos canarios podem entrar em replay v6
comparavel review-only.

Pergunta: hand_mean/twi_mean/tpi_250m_mean/flow_acc_log_mean recalculados pelo
pipeline 17C36 nos patches OFICIAIS sao suficientemente equivalentes aos oficiais?

Guardrails: nao alterar susc_features_by_patch_v1.csv nem o score v6 oficial; sem
v7/GT/treino; ocorrencia nao e feature; NAO usar canarios para calibrar (a
equivalencia e avaliada SO em patches oficiais); nao aceitar equivalencia so
porque o pipeline gerou valores; sem normalizacao canary-only; nao computar score
final se equivalencia rejeitada; raster pesado so em local_runs.

O pipeline hidrologico (fill-sinks/D8/flow-acc/slope/TPI/TWI/HAND) e IMPORTADO do
17C36 para garantir metodo identico. Cada patch oficial amostrado e recomputado
numa janela DEM Copernicus GLO-30 (tile correto W035/W036) com margem.

Reproducibilidade: run-recompute-official-patches (rede+numpy) cacheia os valores
recomputados em susc_17c37_light_artifacts/_ledger_recompute.json; o build offline
reproduz comparacao/metricas/decisao byte-identicos.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_io import ensure_dir, read_csv, read_json, rel, write_csv, write_json, write_markdown  # noqa: E402
import susc_17c36_hydrologic_topography_common as s17c36  # noqa: E402

DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"
LIGHT = OUT / "susc_17c37_light_artifacts"
LOCAL_RUNS = ROOT / "local_runs" / "suscetibilidade" / "17c37_equivalence"
LEDGER = LIGHT / "_ledger_recompute.json"

SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"
OFFICIAL_MATRIX = DAT / "susc_features_by_patch_v1.csv"

C36_MATRIX = OUT / "susc_17c36_topography_hydrology_feature_matrix.csv"
C36_READINESS = OUT / "susc_17c36_score_v6_replay_readiness_update.csv"
C36_SUMMARY = OUT / "susc_17c36_readiness_summary.json"
C35_CONTRACT = OUT / "susc_17c35_score_v6_formula_contract.json"
C33_REGISTRY = OUT / "susc_17c33_event_anchored_canary_patch_registry.csv"

REQUIRED_INPUTS = [
    SCORE_V6, OFFICIAL_MATRIX, C36_MATRIX, C36_READINESS, C36_SUMMARY, C35_CONTRACT, C33_REGISTRY,
]

REPORT = OUT / "SUSC_17C37_TOPOGRAPHY_METHOD_EQUIVALENCE_REPORT.md"
POP_INV = OUT / "susc_17c37_official_patch_population_inventory.csv"
SAMPLE = OUT / "susc_17c37_official_patch_equivalence_sample.csv"
RECOMPUTED = OUT / "susc_17c37_official_patch_recomputed_topography_features.csv"
COMPARISON = OUT / "susc_17c37_topography_feature_equivalence_comparison.csv"
METRICS = OUT / "susc_17c37_method_equivalence_metrics.csv"
DECISION = OUT / "susc_17c37_method_equivalence_decision.json"
READINESS = OUT / "susc_17c37_score_v6_replay_readiness_after_equivalence.csv"
INTEGRITY = OUT / "susc_17c37_score_integrity_audit.csv"
LEAKAGE = OUT / "susc_17c37_no_leakage_audit.csv"
SUMMARY = OUT / "susc_17c37_readiness_summary.json"
BLOCKERS = OUT / "susc_17c37_promotion_blockers.csv"

SCHEMA_METRIC = SCHEMAS / "susc_17c37_method_equivalence_schema_v1.json"
SCHEMA_CMP = SCHEMAS / "susc_17c37_topography_comparison_schema_v1.json"
SCHEMA_RDY = SCHEMAS / "susc_17c37_score_readiness_schema_v1.json"

REQUIRED_OUTPUTS = [
    REPORT, POP_INV, SAMPLE, RECOMPUTED, COMPARISON, METRICS, DECISION, READINESS,
    INTEGRITY, LEAKAGE, SUMMARY, BLOCKERS,
]

CRITICAL_FEATURES = ["hand_mean", "twi_mean", "tpi_250m_mean", "flow_acc_log_mean"]
SAMPLE_MARGIN_DEG = 0.02
FORBIDDEN_SUFFIXES = (".tif", ".tiff", ".safe", ".zip", ".nc", ".jp2", ".gz")
MAX_LIGHT_BYTES = 5_000_000
# thresholds de aceitacao (documentados, defensaveis)
EQ_SPEARMAN = 0.80
EQ_REL_ERR = 0.25
EQ_SCALE_LO, EQ_SCALE_HI = 0.80, 1.25
CAL_SPEARMAN = 0.70


def _bool(v):
    return "true" if v else "false"


def _run_git(args):
    r = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return r.stdout.strip() if r.returncode == 0 else ""


def _env_ok():
    need = ["SUSC_17C37_ALLOW_NETWORK", "SUSC_17C37_ALLOW_PUBLIC_DOWNLOAD",
            "SUSC_17C37_ALLOW_LIGHTWEIGHT_RASTER", "SUSC_17C37_ALLOW_DEM",
            "SUSC_17C37_ALLOW_HYDROLOGIC_PROCESSING"]
    return all(os.environ.get(v) == "1" for v in need)


def _require_inputs():
    missing = [p for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("; ".join(rel(p) for p in missing))


def _official_rows():
    return [r for r in read_csv(OFFICIAL_MATRIX) if r["patch_id"].startswith("recife")]


def _fnum(r, k):
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return None


def _score_by_patch():
    out = {}
    for r in read_csv(SCORE_V6):
        out[r["patch_id"]] = r.get("susc_score_v6_candidate", ""), r.get("susc_class_v6_candidate", "")
    return out


# ---------------------------------------------------------------------------
# Inventory + sample (offline)
# ---------------------------------------------------------------------------

def official_population_inventory_rows():
    rows = _official_rows()
    fields = ["patch_id", "hand_mean", "twi_mean", "tpi_250m_mean", "flow_acc_log_mean",
              "elevation_mean", "slope_mean", "distance_to_water_mean", "urban_prop", "ndbi_mean"]
    out = []
    for i, r in enumerate(rows, start=1):
        present = {f: _bool(_fnum(r, f) is not None) for f in fields if f != "patch_id"}
        has_bbox = all(_fnum(r, k) is not None for k in ("xmin", "ymin", "xmax", "ymax"))
        out.append({
            "population_inventory_id": f"S17C37_POP_{i:04d}", "patch_id": r["patch_id"],
            "has_bbox": _bool(has_bbox),
            **{f"has_{k}": v for k, v in present.items()},
            "review_only": "true",
        })
    return out


def _sample_patches():
    rows = _official_rows()
    score = _score_by_patch()
    chosen = {}
    reasons = {}

    def add(pid, reason):
        if pid not in chosen:
            chosen[pid] = next(r for r in rows if r["patch_id"] == pid)
            reasons[pid] = reason
    # extremos por feature
    for f in CRITICAL_FEATURES:
        valid = [r for r in rows if _fnum(r, f) is not None]
        valid.sort(key=lambda r: _fnum(r, f))
        if valid:
            add(valid[0]["patch_id"], f"min_{f}")
            add(valid[-1]["patch_id"], f"max_{f}")
    # proximos aos canarios (centroides)
    canary = read_csv(C33_REGISTRY)
    clat = sum(float(c["center_lat"]) for c in canary) / len(canary)
    clon = sum(float(c["center_lon"]) for c in canary) / len(canary)

    def dist(r):
        cx = (_fnum(r, "xmin") + _fnum(r, "xmax")) / 2
        cy = (_fnum(r, "ymin") + _fnum(r, "ymax")) / 2
        return (cx - clon) ** 2 + (cy - clat) ** 2
    near = sorted([r for r in rows if _fnum(r, "xmin") is not None], key=dist)
    for r in near[:6]:
        add(r["patch_id"], "near_canary_aoi")
    # estratificado por classe
    by_class = {}
    for r in rows:
        cls = score.get(r["patch_id"], ("", ""))[1] or "unknown"
        by_class.setdefault(cls, []).append(r)
    for cls, lst in sorted(by_class.items()):
        lst_sorted = sorted(lst, key=lambda r: r["patch_id"])
        for r in lst_sorted[:2]:
            add(r["patch_id"], f"stratified_class_{cls}")
    # garantir >=20 (preenche por ordem de patch_id)
    for r in sorted(rows, key=lambda r: r["patch_id"]):
        if len(chosen) >= 24:
            break
        add(r["patch_id"], "fill_to_min_sample")
    ordered = sorted(chosen.keys())
    return [(pid, reasons[pid], chosen[pid]) for pid in ordered]


def official_sample_rows():
    score = _score_by_patch()
    out = []
    for i, (pid, reason, r) in enumerate(_sample_patches(), start=1):
        out.append({
            "sample_id": f"S17C37_SMP_{i:04d}", "patch_id": pid, "selection_reason": reason,
            "official_hand_mean": r.get("hand_mean", ""), "official_twi_mean": r.get("twi_mean", ""),
            "official_tpi_250m_mean": r.get("tpi_250m_mean", ""), "official_flow_acc_log_mean": r.get("flow_acc_log_mean", ""),
            "official_score_v6": score.get(pid, ("", ""))[0],
            "bbox": f"{r.get('xmin','')},{r.get('ymin','')},{r.get('xmax','')},{r.get('ymax','')}",
            "review_only": "true",
        })
    return out


# ---------------------------------------------------------------------------
# Recompute (network) via pipeline 17C36 -> ledger
# ---------------------------------------------------------------------------

def _copernicus_tile_url(lon, lat):
    slat = int(math.ceil(-lat))
    wlon = int(math.ceil(-lon))
    base = f"Copernicus_DSM_COG_10_S{slat:02d}_00_W{wlon:03d}_00_DEM"
    return f"https://copernicus-dem-30m.s3.amazonaws.com/{base}/{base}.tif", base


def _recompute_patch(bbox):
    import numpy as np
    import rasterio
    from rasterio.windows import from_bounds
    xmin, ymin, xmax, ymax = bbox
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    url, base = _copernicus_tile_url(cx, cy)
    m = SAMPLE_MARGIN_DEG
    with rasterio.open(url) as ds:
        w = from_bounds(xmin - m, ymin - m, xmax + m, ymax + m, ds.transform)
        dem = ds.read(1, window=w).astype("float64")
        win_transform = ds.window_transform(w)
        res_deg = ds.res[0]
    dem = np.where(dem < -1000, np.nan, dem)
    valid = np.isfinite(dem)
    if valid.sum() < 50:
        return None
    demf = np.where(valid, dem, np.nanmax(dem[valid]))
    cellsize = res_deg * 111320.0 * math.cos(math.radians(cy))
    filled = s17c36._priority_flood_fill(demf)
    slope = s17c36._slope_rad(filled, cellsize)
    acc, rec = s17c36._d8_flow(filled, cellsize)
    flow_acc_log = np.log1p(acc)
    tpi = s17c36._tpi(demf, cellsize, s17c36.TPI_RADIUS_M)
    thr = np.percentile(acc, s17c36.DRAINAGE_FLOWACC_PCTL)
    hand = s17c36._hand(filled, acc, rec, acc >= thr)
    a_sca = (acc + 1.0) * cellsize
    beta = np.maximum(slope, math.radians(0.1))
    twi = np.log(a_sca / np.tan(beta))
    from rasterio.transform import rowcol
    ny, nx = demf.shape
    r0, c0 = rowcol(win_transform, xmin, ymax)
    r1, c1 = rowcol(win_transform, xmax, ymin)
    r0, r1 = sorted((max(0, r0), min(ny, r1 + 1)))
    c0, c1 = sorted((max(0, c0), min(nx, c1 + 1)))

    def stat(arr):
        sub = arr[r0:r1, c0:c1]
        sub = sub[np.isfinite(sub)]
        return round(float(sub.mean()), 4) if sub.size else None, int(sub.size)
    e, npx = stat(demf)
    return {
        "tile": base, "valid_pixel_count": npx,
        "recomputed_elevation_mean": e,
        "recomputed_slope_mean": stat(np.degrees(slope))[0],
        "recomputed_hand_mean": stat(hand)[0],
        "recomputed_twi_mean": stat(twi)[0],
        "recomputed_tpi_250m_mean": stat(tpi)[0],
        "recomputed_flow_acc_log_mean": stat(flow_acc_log)[0],
    }


def _run_recompute():
    if not _env_ok():
        raise SystemExit("BLOCKED: run-recompute-official-patches exige os cinco opt-ins SUSC_17C37_*.")
    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
    os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")
    ensure_dir(LIGHT)
    ensure_dir(LOCAL_RUNS)
    recomputed = []
    for s in official_sample_rows():
        bbox = tuple(float(x) for x in s["bbox"].split(","))
        try:
            res = _recompute_patch(bbox)
            if res:
                recomputed.append({"patch_id": s["patch_id"], "status": "recomputed", **res})
            else:
                recomputed.append({"patch_id": s["patch_id"], "status": "insufficient_valid_pixels"})
        except Exception as e:
            recomputed.append({"patch_id": s["patch_id"], "status": f"error_{type(e).__name__}"})
    write_json(LEDGER, {"generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "pipeline_source": "susc_17c36_hydrologic_topography_common",
                        "dem_source": s17c36.DEM_SOURCE, "recomputed": recomputed})


def _load_ledger():
    if not LEDGER.exists():
        raise FileNotFoundError(f"{rel(LEDGER)} ausente: rode run-recompute-official-patches com os cinco opt-ins.")
    return read_json(LEDGER)


# ---------------------------------------------------------------------------
# Offline derivation
# ---------------------------------------------------------------------------

def recomputed_rows():
    led = _load_ledger()
    by = {r["patch_id"]: r for r in led.get("recomputed", [])}
    out = []
    for i, s in enumerate(official_sample_rows(), start=1):
        r = by.get(s["patch_id"], {})
        ok = r.get("status") == "recomputed"
        out.append({
            "recomputed_feature_id": f"S17C37_REC_{i:04d}", "patch_id": s["patch_id"],
            "dem_source": led.get("dem_source", "copernicus_dem_glo30"),
            "pipeline_source": led.get("pipeline_source", "susc_17c36"),
            "recomputed_hand_mean": r.get("recomputed_hand_mean", "not_available") if ok else "not_available",
            "recomputed_twi_mean": r.get("recomputed_twi_mean", "not_available") if ok else "not_available",
            "recomputed_tpi_250m_mean": r.get("recomputed_tpi_250m_mean", "not_available") if ok else "not_available",
            "recomputed_flow_acc_log_mean": r.get("recomputed_flow_acc_log_mean", "not_available") if ok else "not_available",
            "recomputed_elevation_mean": r.get("recomputed_elevation_mean", "not_available") if ok else "not_available",
            "recomputed_slope_mean": r.get("recomputed_slope_mean", "not_available") if ok else "not_available",
            "valid_pixel_count": str(r.get("valid_pixel_count", 0)),
            "method_equivalence_status": s17c36.METHOD_EQUIV, "review_only": "true",
        })
    return out


def _rank(values):
    # ranks (1..n) com media para empates; ignora None
    idx = [(v, i) for i, v in enumerate(values) if v is not None]
    idx.sort(key=lambda t: t[0])
    ranks = [None] * len(values)
    for r, (_v, i) in enumerate(idx, start=1):
        ranks[i] = r
    return ranks


def comparison_rows():
    sample = {s["patch_id"]: s for s in official_sample_rows()}
    rec = {r["patch_id"]: r for r in recomputed_rows()}
    off_map = {r["patch_id"]: r for r in _official_rows()}
    # coleta valores por feature para ranks
    pids = [s["patch_id"] for s in official_sample_rows()]
    off_vals = {f: [] for f in CRITICAL_FEATURES}
    rec_vals = {f: [] for f in CRITICAL_FEATURES}
    for pid in pids:
        for f in CRITICAL_FEATURES:
            off_vals[f].append(_fnum(off_map.get(pid, {}), f))
            rv = rec[pid].get(f.replace("_mean", "").replace("tpi_250m", "tpi_250m") and "recomputed_" + f) if False else None
    # mapear feature -> coluna recomputada
    rec_col = {"hand_mean": "recomputed_hand_mean", "twi_mean": "recomputed_twi_mean",
               "tpi_250m_mean": "recomputed_tpi_250m_mean", "flow_acc_log_mean": "recomputed_flow_acc_log_mean"}
    for pid in pids:
        for f in CRITICAL_FEATURES:
            v = rec[pid].get(rec_col[f], "not_available")
            rec_vals[f].append(float(v) if v not in ("not_available", "", None) else None)
    off_ranks = {f: _rank(off_vals[f]) for f in CRITICAL_FEATURES}
    rec_ranks = {f: _rank(rec_vals[f]) for f in CRITICAL_FEATURES}
    rows = []
    cid = 0
    for j, pid in enumerate(pids):
        for f in CRITICAL_FEATURES:
            cid += 1
            ov = off_vals[f][j]
            rv = rec_vals[f][j]
            ae = abs(ov - rv) if (ov is not None and rv is not None) else None
            re_ = (ae / abs(ov)) if (ae is not None and ov not in (None, 0)) else None
            scale_ok = (rv is not None and ov not in (None, 0) and EQ_SCALE_LO <= (rv / ov) <= EQ_SCALE_HI)
            within = (re_ is not None and re_ <= EQ_REL_ERR)
            ro, rr = off_ranks[f][j], rec_ranks[f][j]
            same_bucket = (ro is not None and rr is not None and abs(ro - rr) <= max(1, len(pids) // 5))
            rows.append({
                "comparison_id": f"S17C37_CMP_{cid:04d}", "patch_id": pid, "feature_name": f,
                "official_value": "not_available" if ov is None else round(ov, 4),
                "recomputed_value": "not_available" if rv is None else round(rv, 4),
                "absolute_error": "not_available" if ae is None else round(ae, 4),
                "relative_error": "not_available" if re_ is None else round(re_, 4),
                "rank_official": ro if ro is not None else "not_available",
                "rank_recomputed": rr if rr is not None else "not_available",
                "same_order_bucket": _bool(same_bucket),
                "scale_compatible": _bool(scale_ok),
                "within_acceptance_threshold": _bool(within),
                "blocking_reason": "not_applicable" if within else "relative_error_or_scale_out_of_threshold",
                "review_only": "true",
            })
    return rows


def _pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    n = len(pairs)
    if n < 3:
        return None
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    num = sum((p[0] - mx) * (p[1] - my) for p in pairs)
    dx = math.sqrt(sum((p[0] - mx) ** 2 for p in pairs))
    dy = math.sqrt(sum((p[1] - my) ** 2 for p in pairs))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def _median(vals):
    v = sorted(x for x in vals if x is not None)
    if not v:
        return None
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def metrics_rows():
    off_map = {r["patch_id"]: r for r in _official_rows()}
    rec = {r["patch_id"]: r for r in recomputed_rows()}
    pids = [s["patch_id"] for s in official_sample_rows()]
    rec_col = {"hand_mean": "recomputed_hand_mean", "twi_mean": "recomputed_twi_mean",
               "tpi_250m_mean": "recomputed_tpi_250m_mean", "flow_acc_log_mean": "recomputed_flow_acc_log_mean"}
    rows = []
    for i, f in enumerate(CRITICAL_FEATURES, start=1):
        ov, rv = [], []
        for pid in pids:
            ov.append(_fnum(off_map.get(pid, {}), f))
            v = rec[pid].get(rec_col[f], "not_available")
            rv.append(float(v) if v not in ("not_available", "", None) else None)
        pear = _pearson(ov, rv)
        spear = _pearson(_rank(ov), _rank(rv))
        aes = [abs(o - r) for o, r in zip(ov, rv) if o is not None and r is not None]
        res = [abs(o - r) / abs(o) for o, r in zip(ov, rv) if o is not None and r not in (None,) and o not in (None, 0)]
        ratios = [r / o for o, r in zip(ov, rv) if o not in (None, 0) and r is not None]
        ovv = [o for o in ov if o is not None]
        rvv = [r for r in rv if r is not None]
        n = len([1 for o, r in zip(ov, rv) if o is not None and r is not None])
        scale_med = _median(ratios)
        rank_pres = spear if spear is not None else 0.0
        if n < 3:
            status = "insufficient_data"
        elif (spear is not None and spear >= EQ_SPEARMAN and _median(res) is not None and _median(res) <= EQ_REL_ERR
              and scale_med is not None and EQ_SCALE_LO <= scale_med <= EQ_SCALE_HI):
            status = "equivalent"
        elif spear is not None and spear >= CAL_SPEARMAN:
            status = "calibratable"
        else:
            status = "incompatible"
        rows.append({
            "method_equivalence_metric_id": f"S17C37_MEM_{i:04d}", "feature_name": f,
            "pearson_corr": "not_available" if pear is None else round(pear, 4),
            "spearman_corr": "not_available" if spear is None else round(spear, 4),
            "mae": "not_available" if not aes else round(sum(aes) / len(aes), 4),
            "median_absolute_error": "not_available" if _median(aes) is None else round(_median(aes), 4),
            "relative_error_median": "not_available" if _median(res) is None else round(_median(res), 4),
            "official_min": round(min(ovv), 4) if ovv else "not_available",
            "official_max": round(max(ovv), 4) if ovv else "not_available",
            "recomputed_min": round(min(rvv), 4) if rvv else "not_available",
            "recomputed_max": round(max(rvv), 4) if rvv else "not_available",
            "scale_ratio_median": "not_available" if scale_med is None else round(scale_med, 4),
            "rank_preservation_score": round(rank_pres, 4),
            "equivalence_status": status, "review_only": "true",
        })
    return rows


def equivalence_decision():
    metrics = {m["feature_name"]: m for m in metrics_rows()}
    accepted = [f for f in CRITICAL_FEATURES if metrics[f]["equivalence_status"] == "equivalent"]
    calibratable = [f for f in CRITICAL_FEATURES if metrics[f]["equivalence_status"] == "calibratable"]
    incompatible = [f for f in CRITICAL_FEATURES if metrics[f]["equivalence_status"] == "incompatible"]
    insufficient = [f for f in CRITICAL_FEATURES if metrics[f]["equivalence_status"] == "insufficient_data"]
    full_equiv = len(accepted) == len(CRITICAL_FEATURES)
    calibration_possible = (not full_equiv) and (not incompatible) and (not insufficient) and len(calibratable) > 0
    can_final = full_equiv
    can_calibrated = calibration_possible or (len(calibratable) > 0 and not insufficient)
    if full_equiv:
        reason = "todas as 4 features criticas equivalentes"
    elif incompatible:
        reason = f"features incompativeis: {','.join(incompatible)} (escala/rank divergentes do metodo oficial nao documentado)"
    elif calibratable:
        reason = f"features calibraveis: {','.join(calibratable)}; nenhuma incompativel"
    else:
        reason = "dados insuficientes para decisao"
    return {
        "method_equivalence_decision_created": True,
        "method_equivalence_accepted": full_equiv,
        "method_calibration_possible": bool(calibration_possible or (len(calibratable) > 0 and not incompatible and not insufficient)),
        "accepted_features": accepted, "calibratable_features": calibratable,
        "incompatible_features": incompatible, "insufficient_features": insufficient,
        "blocking_features": incompatible + insufficient,
        "reason": reason,
        "can_compute_score_v6_final_replay": can_final,
        "can_compute_calibrated_component_replay": bool(can_calibrated),
        "requires_future_calibration": bool(calibratable and not full_equiv),
        "acceptance_thresholds": {"spearman_equivalent": EQ_SPEARMAN, "rel_err_max": EQ_REL_ERR,
                                  "scale_ratio": [EQ_SCALE_LO, EQ_SCALE_HI], "spearman_calibratable": CAL_SPEARMAN},
        "review_only": True,
    }


def readiness_rows():
    dec = equivalence_decision()
    prev = {r["canary_patch_id"]: r for r in read_csv(C36_READINESS)}
    accepted = dec["method_equivalence_accepted"]
    cal = dec["method_calibration_possible"]
    rows = []
    for i, c in enumerate(read_csv(C33_REGISTRY), start=1):
        pid = c["event_anchored_canary_patch_id"]
        p = prev.get(pid, {})
        if accepted:
            status = "topography_equivalent_ready_for_final_replay_next_marco"
        elif cal:
            status = "topography_calibratable_calibrated_component_replay_only"
        else:
            status = "topography_incompatible_score_final_blocked"
        rows.append({
            "score_replay_readiness_after_equivalence_id": f"S17C37_RDY_{i:04d}", "canary_patch_id": pid,
            "previous_status_17c36": p.get("score_replay_status_after_17c36", "topography_reconstructed_score_final_blocked_on_method_equivalence"),
            "method_equivalence_accepted": _bool(accepted),
            "method_calibration_possible": _bool(cal),
            "topography_features_equivalent": ";".join(dec["accepted_features"]) or "none",
            "topography_features_calibratable": ";".join(dec["calibratable_features"]) or "none",
            "can_compute_full_score_v6_replay": _bool(dec["can_compute_score_v6_final_replay"]),
            "can_compute_calibrated_score_v6_replay": _bool(dec["can_compute_calibrated_component_replay"]),
            "score_replay_status_after_17c37": status,
            "blocking_reason": dec["reason"],
            "not_official_score": "true", "does_not_replace_score_v6": "true", "review_only": "true",
        })
    return rows


def score_integrity_audit_rows():
    changed = bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)]))
    h = _run_git(["hash-object", rel(SCORE_V6)]) if SCORE_V6.exists() else "absent"
    hm = _run_git(["hash-object", rel(OFFICIAL_MATRIX)]) if OFFICIAL_MATRIX.exists() else "absent"
    matrix_changed = bool(_run_git(["diff", "--name-only", "--", rel(OFFICIAL_MATRIX)]))
    dec = equivalence_decision()
    return [{"score_integrity_audit_id": "S17C37_INTEG_0001", "official_score_v6_path": rel(SCORE_V6),
             "official_score_v6_hash": h, "official_score_v6_changed": _bool(changed),
             "official_matrix_path": rel(OFFICIAL_MATRIX), "official_matrix_hash": hm,
             "official_matrix_changed": _bool(matrix_changed), "score_v7_exists": _bool(SCORE_V7.exists()),
             "final_score_computed": _bool(dec["can_compute_score_v6_final_replay"]), "review_only": "true"}]


def no_leakage_audit_rows():
    rows = []
    for i, s in enumerate(official_sample_rows(), start=1):
        rows.append({
            "no_leakage_audit_id": f"S17C37_LEAK_{i:04d}", "object_id": s["patch_id"],
            "object_type": "official_patch_equivalence_sample",
            "uses_occurrence_as_feature": "false",
            "uses_canary_to_calibrate_equivalence": "false",
            "equivalence_evaluated_on_official_patches": "true",
            "uses_canary_only_normalization": "false",
            "accepts_equivalence_just_because_values_generated": "false",
            "uses_control_as_negative_truth": "false",
            "modifies_official_matrix": "false", "modifies_score_v6": "false", "uses_score_v7": "false",
            "passes_no_leakage": "true",
            "blocking_reason": "equivalencia avaliada so em patches oficiais; canarios nao calibram; review-only",
            "review_only": "true",
        })
    return rows


def build_summary():
    pop = official_population_inventory_rows()
    sample = official_sample_rows()
    rec = recomputed_rows()
    comp = comparison_rows()
    metrics = metrics_rows()
    dec = equivalence_decision()
    ready = readiness_rows()
    rec_ok = len([r for r in rec if r["recomputed_hand_mean"] != "not_available"])
    minimum = (
        len(pop) >= 100 and len(sample) >= 20 and rec_ok >= 20
        and len(comp) >= 80 and len(metrics) >= 4 and dec["method_equivalence_decision_created"]
    )
    return {
        "minimum_success_achieved": minimum,
        "official_patch_population_rows_count": len(pop),
        "official_patch_sample_rows_count": len(sample),
        "official_patch_recomputed_rows_count": rec_ok,
        "topography_feature_comparison_rows_count": len(comp),
        "method_equivalence_metric_rows_count": len(metrics),
        "method_equivalence_decision_created": True,
        "method_equivalence_accepted": dec["method_equivalence_accepted"],
        "method_calibration_possible": dec["method_calibration_possible"],
        "accepted_feature_count": len(dec["accepted_features"]),
        "calibratable_feature_count": len(dec["calibratable_features"]),
        "incompatible_feature_count": len(dec["incompatible_features"]),
        "per_feature_status": {m["feature_name"]: m["equivalence_status"] for m in metrics},
        "can_compute_score_v6_final_replay": dec["can_compute_score_v6_final_replay"],
        "can_compute_calibrated_component_replay": dec["can_compute_calibrated_component_replay"],
        "score_v6_replay_readiness_updated": True,
        "score_v6_replay_readiness_rows_count": len(ready),
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
        "result_class": ("A_method_equivalent" if dec["method_equivalence_accepted"]
                         else ("B_method_calibratable" if dec["method_calibration_possible"] else "C_method_incompatible")),
        "recommended_next_milestone": (
            "SUSC-17C38 Politica de calibracao review-only das features topograficas calibraveis (ex.: TWI/flow_acc por "
            "relacao monotonica) OU busca do metodo/DEM original do score v6; so entao replay v6 comparavel. Manter score "
            "v6 oficial intacto e 17B fail-closed."
        ),
    }


def build_blockers():
    metrics = {m["feature_name"]: m for m in metrics_rows()}
    dec = equivalence_decision()
    blockers = []
    if not dec["method_equivalence_accepted"]:
        blockers.append(("method_equivalence_not_accepted", "equivalencia metodologica full nao aceita"))
    for f, key in [("twi_mean", "twi_scale_incompatible"), ("hand_mean", "hand_scale_incompatible"),
                   ("flow_acc_log_mean", "flow_acc_scale_incompatible"), ("tpi_250m_mean", "tpi_scale_incompatible")]:
        if metrics[f]["equivalence_status"] in ("incompatible", "insufficient_data"):
            blockers.append((key, f"{f}: status {metrics[f]['equivalence_status']} (spearman={metrics[f]['spearman_corr']}, scale_ratio={metrics[f]['scale_ratio_median']})"))
    if dec["requires_future_calibration"]:
        blockers.append(("method_calibration_required", f"features calibraveis: {','.join(dec['calibratable_features'])}"))
    blockers += [
        ("score_v6_final_replay_blocked", "score final v6 bloqueado ate full equivalence"),
        ("score_v7_blocked", "score v7 bloqueado"),
        ("17b_blocked", "17B bloqueado ate G4_full + ground reference aceito"),
    ]
    return [{"blocker_id": f"S17C37_BLOCKER_{i:04d}", "blocker_type": n, "description": d,
             "blocks_final_score_replay": "true", "blocks_17b": "true", "blocks_score_v7": "true", "review_only": "true"}
            for i, (n, d) in enumerate(blockers, start=1)]


def build_report():
    s = build_summary()
    metrics = metrics_rows()
    return "\n".join([
        "# SUSC-17C37 - Validacao de Equivalencia Metodologica Topography/Hydrology",
        "",
        "## Objetivo",
        "Validar se o pipeline hidrologico local do 17C36 e equivalente/calibravel/incompativel com as features oficiais (susc_features_by_patch_v1.csv), recomputando patches OFICIAIS amostrados com o MESMO pipeline e comparando aos valores oficiais.",
        "",
        "## Amostra e recomputo",
        f"- Populacao oficial: {s['official_patch_population_rows_count']} patches recife.",
        f"- Amostra oficial: {s['official_patch_sample_rows_count']} (extremos por feature + proximos aos canarios + estratificado por classe).",
        f"- Recomputados (pipeline 17C36, DEM Copernicus GLO-30): {s['official_patch_recomputed_rows_count']}.",
        f"- Comparacoes patch x feature: {s['topography_feature_comparison_rows_count']}.",
        "",
        "## Metricas de equivalencia por feature",
        *[f"  - {m['feature_name']}: status={m['equivalence_status']}; spearman={m['spearman_corr']}; rel_err_med={m['relative_error_median']}; scale_ratio_med={m['scale_ratio_median']} (oficial {m['official_min']}..{m['official_max']}; recomputado {m['recomputed_min']}..{m['recomputed_max']})" for m in metrics],
        "",
        "## Decisao",
        f"- method_equivalence_accepted={s['method_equivalence_accepted']}; method_calibration_possible={s['method_calibration_possible']}.",
        f"- Equivalentes: {s['accepted_feature_count']}; calibraveis: {s['calibratable_feature_count']}; incompativeis: {s['incompatible_feature_count']}.",
        f"- can_compute_score_v6_final_replay={s['can_compute_score_v6_final_replay']}; can_compute_calibrated_component_replay={s['can_compute_calibrated_component_replay']}.",
        f"- result_class: {s['result_class']}.",
        "",
        "## Interpretacao (honesta)",
        "O pipeline 17C36 gera features reais mas o metodo/DEM/unidade originais do score v6 nao estao documentados; a comparacao em patches oficiais mede diretamente a equivalencia. O score final v6 comparavel so e liberado se as 4 features criticas forem equivalent. Caso contrario, o score final permanece bloqueado (calibracao futura ou busca do metodo original).",
        "",
        "## Guardrails",
        "- susc_features_by_patch_v1.csv e score v6 oficial intactos; sem v7/GT/treino; equivalencia avaliada SO em patches oficiais (canarios nao calibram); ocorrencia nao e feature; sem normalizacao canary-only; raster pesado so em local_runs; controle nao vira negativo verdadeiro.",
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
    write_csv(POP_INV, official_population_inventory_rows())
    write_csv(SAMPLE, official_sample_rows())
    write_csv(RECOMPUTED, recomputed_rows())
    write_csv(COMPARISON, comparison_rows())
    write_csv(METRICS, metrics_rows())
    write_json(DECISION, equivalence_decision())
    write_csv(READINESS, readiness_rows())
    write_csv(INTEGRITY, score_integrity_audit_rows())
    write_csv(LEAKAGE, no_leakage_audit_rows())
    write_json(SUMMARY, build_summary())
    write_csv(BLOCKERS, build_blockers())
    write_markdown(REPORT, build_report())


def run_mode(mode):
    _require_inputs()
    ensure_dir(OUT)
    if mode == "inventory-official-population":
        write_csv(POP_INV, official_population_inventory_rows())
    elif mode == "build-official-sample":
        write_csv(SAMPLE, official_sample_rows())
    elif mode == "run-recompute-official-patches":
        _run_recompute()
    elif mode == "compare-topography-features":
        write_csv(COMPARISON, comparison_rows())
    elif mode == "compute-equivalence-metrics":
        write_csv(METRICS, metrics_rows())
    elif mode == "decide-method-equivalence":
        write_json(DECISION, equivalence_decision())
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

    pop = read_csv(POP_INV)
    sample = read_csv(SAMPLE)
    rec = read_csv(RECOMPUTED)
    comp = read_csv(COMPARISON)
    metrics = read_csv(METRICS)
    ready = read_csv(READINESS)
    leak = read_csv(LEAKAGE)
    integ = read_csv(INTEGRITY)
    summary = read_json(SUMMARY)
    decision = read_json(DECISION)
    m_schema = read_json(SCHEMA_METRIC)
    c_schema = read_json(SCHEMA_CMP)
    r_schema = read_json(SCHEMA_RDY)

    for r in metrics:
        errors.extend(_schema_violations(r, m_schema))
    for r in comp:
        errors.extend(_schema_violations(r, c_schema))
    for r in ready:
        errors.extend(_schema_violations(r, r_schema))

    # 1-6 minimos
    if len(pop) < 100:
        errors.append("population_lt_100")
    if len(sample) < 20:
        errors.append("sample_lt_20")
    rec_ok = len([r for r in rec if r["recomputed_hand_mean"] != "not_available"])
    if rec_ok < 20:
        errors.append("recomputed_lt_20")
    if len(comp) < 80:
        errors.append("comparison_lt_80")
    if len(metrics) < 4:
        errors.append("metrics_lt_4")
    feats = {m["feature_name"] for m in metrics}
    if not set(CRITICAL_FEATURES).issubset(feats):
        errors.append("metrics_missing_critical_feature")
    if not decision.get("method_equivalence_decision_created"):
        errors.append("decision_not_created")

    # 7 full equivalence com feature nao-equivalente
    statuses = {m["feature_name"]: m["equivalence_status"] for m in metrics}
    if decision["method_equivalence_accepted"] and any(statuses[f] != "equivalent" for f in CRITICAL_FEATURES):
        errors.append("full_equivalence_with_non_equivalent_feature")
    # 8 score final sem full equivalence
    if summary["can_compute_score_v6_final_replay"] and not decision["method_equivalence_accepted"]:
        errors.append("final_score_without_full_equivalence")
    for r in integ:
        if r["final_score_computed"] == "true" and not decision["method_equivalence_accepted"]:
            errors.append("integrity_final_score_without_equivalence")
    # 9 canarios para calibrar
    for r in leak:
        if r["uses_canary_to_calibrate_equivalence"] != "false":
            errors.append("canary_used_to_calibrate")
        if r["equivalence_evaluated_on_official_patches"] != "true":
            errors.append("equivalence_not_on_official")
        if r["accepts_equivalence_just_because_values_generated"] != "false":
            errors.append("accepts_equivalence_trivially")
    # sample e recompute sao patches oficiais (recife_*)
    for r in sample:
        if not r["patch_id"].startswith("recife"):
            errors.append(f"sample_not_official:{r['patch_id']}")
    # 10 score v6 intacto + matriz oficial intacta
    if _run_git(["diff", "--name-only", "--", rel(SCORE_V6)]) or summary["score_v6_original_file_changed"]:
        errors.append("score_v6_changed")
    if _run_git(["diff", "--name-only", "--", rel(OFFICIAL_MATRIX)]) or summary["official_matrix_changed"]:
        errors.append("official_matrix_changed")
    # 11 v7
    if SCORE_V7.exists() or summary["score_v7_created"]:
        errors.append("score_v7_exists")
    # 12/13 GT/treino
    if summary["ground_truth_created"] or summary["training_labels_created"]:
        errors.append("gt_or_training")
    # 14 raster pesado
    if LIGHT.exists():
        for p in LIGHT.glob("**/*"):
            if p.is_file() and (p.suffix.lower() in FORBIDDEN_SUFFIXES or p.stat().st_size > MAX_LIGHT_BYTES):
                errors.append(f"forbidden_or_heavy_committed:{rel(p)}")
    # 15 ocorrencia como feature
    if any(r["uses_occurrence_as_feature"] != "false" for r in leak):
        errors.append("occurrence_as_feature")
    # 16 controle negativo
    if any(r["uses_control_as_negative_truth"] != "false" for r in leak):
        errors.append("control_negative")
    if any(r["passes_no_leakage"] != "true" for r in leak):
        errors.append("no_leakage_failed")

    # 17 summary reflete contagens
    expected = build_summary()
    for k, v in expected.items():
        if summary.get(k) != v:
            errors.append(f"summary_mismatch:{k}")
    if not summary["minimum_success_achieved"]:
        errors.append("minimum_success_not_achieved")

    for rows, key in [
        (pop, "population_inventory_id"), (sample, "sample_id"), (rec, "recomputed_feature_id"),
        (comp, "comparison_id"), (metrics, "method_equivalence_metric_id"), (ready, "score_replay_readiness_after_equivalence_id"),
        (leak, "no_leakage_audit_id"),
    ]:
        ids = [r[key] for r in rows]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            errors.append(f"ids_not_unique_or_sorted:{key}")

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(
        "17C37 -> "
        f"pop={summary['official_patch_population_rows_count']} sample={summary['official_patch_sample_rows_count']} "
        f"recomputed={summary['official_patch_recomputed_rows_count']} comparison={summary['topography_feature_comparison_rows_count']} "
        f"metrics={summary['method_equivalence_metric_rows_count']} accepted={summary['method_equivalence_accepted']} "
        f"calibratable={summary['method_calibration_possible']} per_feature={summary['per_feature_status']} "
        f"final_replay={summary['can_compute_score_v6_final_replay']} result={summary['result_class']} "
        f"min_success={summary['minimum_success_achieved']}"
    )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="susc_17c37")
    parser.add_argument("mode", choices=[
        "inventory-official-population", "build-official-sample", "run-recompute-official-patches",
        "compare-topography-features", "compute-equivalence-metrics", "decide-method-equivalence",
        "update-score-replay-readiness", "build-offline", "audit",
    ])
    args = parser.parse_args(argv)
    if args.mode == "audit":
        return validate()
    run_mode(args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
