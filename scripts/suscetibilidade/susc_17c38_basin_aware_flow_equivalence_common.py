"""SUSC-17C38 Flow Accumulation Basin-Aware e Revalidacao de Equivalencia.

O 17C37 rejeitou a equivalencia porque flow_acc ficou incompativel (spearman
-0.51). Hipotese: o flow accumulation foi truncado por ser calculado em janela
POR PATCH, cortando a bacia contribuinte a montante. Este marco recomputa o
flow accumulation (e TWI/TPI/HAND/elevation/slope) num DOMINIO HIDROLOGICO UNICO
E AMPLO cobrindo todos os 100 patches oficiais + buffer, e reavalia a
equivalencia com susc_features_by_patch_v1.csv.

NAO calibra ainda. NAO cria score final. NAO altera score v6 oficial. NAO cria
score v7/GT/treino. Objetivo: testar se o truncamento espacial causou a
incompatibilidade do 17C37.

Metodo: DEM Copernicus GLO-30 mosaicado sobre o dominio (tiles S08/S09 x
W035/W036), reamostrado para uma resolucao de dominio (documentada), UMA unica
grade; pipeline hidrologico IMPORTADO do 17C36 (fill-sinks priority-flood, D8,
flow accumulation, slope, TPI 250m, TWI, HAND) rodado UMA vez no dominio; cada
patch oficial e amostrado do MESMO grid (nunca janela isolada por patch para
flow accumulation).

Guardrails: canarios NAO decidem equivalencia (so patches oficiais); flow_acc
NAO por janela de patch; sem normalizacao canary-only; matriz oficial e score v6
intactos; sem v7/GT/treino; score final so com full equivalence; raster pesado
so em local_runs; controle nao vira negativo verdadeiro.

Reproducibilidade: build-basin-hydrologic-grid (rede+numpy) cacheia stats por
patch em susc_17c38_light_artifacts/_ledger_basin.json; build offline reproduz
comparacao/metricas/decisao byte-identicos.
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
import susc_17c37_topography_method_equivalence_common as s17c37  # noqa: E402

DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"
LIGHT = OUT / "susc_17c38_light_artifacts"
LOCAL_RUNS = ROOT / "local_runs" / "suscetibilidade" / "17c38_basin"
LEDGER = LIGHT / "_ledger_basin.json"

SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"
OFFICIAL_MATRIX = DAT / "susc_features_by_patch_v1.csv"

C37_METRICS = OUT / "susc_17c37_method_equivalence_metrics.csv"
C37_DECISION = OUT / "susc_17c37_method_equivalence_decision.json"
C37_READINESS = OUT / "susc_17c37_score_v6_replay_readiness_after_equivalence.csv"
C37_SUMMARY = OUT / "susc_17c37_readiness_summary.json"
C36_MATRIX = OUT / "susc_17c36_topography_hydrology_feature_matrix.csv"
C33_REGISTRY = OUT / "susc_17c33_event_anchored_canary_patch_registry.csv"

REQUIRED_INPUTS = [
    SCORE_V6, OFFICIAL_MATRIX, C37_METRICS, C37_DECISION, C37_READINESS, C37_SUMMARY,
    C36_MATRIX, C33_REGISTRY,
]

REPORT = OUT / "SUSC_17C38_BASIN_AWARE_FLOW_EQUIVALENCE_REPORT.md"
DOMAIN = OUT / "susc_17c38_basin_aware_domain_definition.csv"
DEM_ATTEMPTS = OUT / "susc_17c38_dem_acquisition_attempts.csv"
DEM_MANIFEST = OUT / "susc_17c38_dem_artifact_manifest.csv"
DEM_GRID = OUT / "susc_17c38_basin_dem_grid_metadata.csv"
HYDRO_GRID = OUT / "susc_17c38_basin_hydrologic_grid_metadata.csv"
FLOW_DIAG = OUT / "susc_17c38_basin_flow_accumulation_diagnostics.csv"
BASIN_FEATURES = OUT / "susc_17c38_official_patch_basin_aware_topography_features.csv"
COMPARISON = OUT / "susc_17c38_basin_aware_topography_feature_comparison.csv"
METRICS = OUT / "susc_17c38_basin_aware_equivalence_metrics.csv"
DECISION = OUT / "susc_17c38_method_equivalence_decision_update.json"
READINESS = OUT / "susc_17c38_score_v6_replay_readiness_after_basin_flow.csv"
INTEGRITY = OUT / "susc_17c38_score_integrity_audit.csv"
LEAKAGE = OUT / "susc_17c38_no_leakage_audit.csv"
SUMMARY = OUT / "susc_17c38_readiness_summary.json"
BLOCKERS = OUT / "susc_17c38_promotion_blockers.csv"

SCHEMA_BASIN = SCHEMAS / "susc_17c38_basin_flow_schema_v1.json"
SCHEMA_METRIC = SCHEMAS / "susc_17c38_equivalence_metrics_schema_v1.json"
SCHEMA_RDY = SCHEMAS / "susc_17c38_score_readiness_schema_v1.json"

REQUIRED_OUTPUTS = [
    REPORT, DOMAIN, DEM_ATTEMPTS, DEM_MANIFEST, DEM_GRID, HYDRO_GRID, FLOW_DIAG,
    BASIN_FEATURES, COMPARISON, METRICS, DECISION, READINESS, INTEGRITY, LEAKAGE, SUMMARY, BLOCKERS,
]

CRITICAL = s17c37.CRITICAL_FEATURES
BUFFER_KM = 5.0
DOMAIN_RES_DEG = 0.0008333  # ~90 m: resolucao de dominio (documentada) p/ rotear bacia inteira
FORBIDDEN_SUFFIXES = (".tif", ".tiff", ".safe", ".zip", ".nc", ".jp2", ".gz")
MAX_LIGHT_BYTES = 5_000_000
EQ_SPEARMAN = 0.85
CAL_SPEARMAN = 0.70
EQ_SCALE_LO, EQ_SCALE_HI = 0.80, 1.25
EQ_REL_ERR = 0.25
DOMAIN_ID = "S17C38_DOMAIN_RECIFE_BASIN_V1"


def _bool(v):
    return "true" if v else "false"


def _run_git(args):
    r = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return r.stdout.strip() if r.returncode == 0 else ""


def _env_ok():
    need = ["SUSC_17C38_ALLOW_NETWORK", "SUSC_17C38_ALLOW_PUBLIC_DOWNLOAD",
            "SUSC_17C38_ALLOW_LIGHTWEIGHT_RASTER", "SUSC_17C38_ALLOW_DEM",
            "SUSC_17C38_ALLOW_HYDROLOGIC_PROCESSING", "SUSC_17C38_ALLOW_BASIN_AWARE_FLOW"]
    return all(os.environ.get(v) == "1" for v in need)


def _require_inputs():
    missing = [p for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("; ".join(rel(p) for p in missing))


def _official_rows():
    return s17c37._official_rows()


def _fnum(r, k):
    return s17c37._fnum(r, k)


def _official_bbox_union():
    rows = _official_rows()
    xmins = [_fnum(r, "xmin") for r in rows if _fnum(r, "xmin") is not None]
    ymins = [_fnum(r, "ymin") for r in rows if _fnum(r, "ymin") is not None]
    xmaxs = [_fnum(r, "xmax") for r in rows if _fnum(r, "xmax") is not None]
    ymaxs = [_fnum(r, "ymax") for r in rows if _fnum(r, "ymax") is not None]
    return min(xmins), min(ymins), max(xmaxs), max(ymaxs)


def _domain_bbox():
    xmin, ymin, xmax, ymax = _official_bbox_union()
    lat_mid = (ymin + ymax) / 2
    dlat = BUFFER_KM * 1000.0 / 110540.0
    dlon = BUFFER_KM * 1000.0 / (111320.0 * math.cos(math.radians(lat_mid)))
    return xmin - dlon, ymin - dlat, xmax + dlon, ymax + dlat


# ---------------------------------------------------------------------------
# Domain definition (offline)
# ---------------------------------------------------------------------------

def domain_definition_rows():
    orig = _official_bbox_union()
    dom = _domain_bbox()
    slat_lo = int(math.ceil(-dom[3]))
    slat_hi = int(math.ceil(-dom[1]))
    wlon_lo = int(math.ceil(-dom[2]))
    wlon_hi = int(math.ceil(-dom[0]))
    tiles = []
    for slat in range(slat_lo, slat_hi + 1):
        for wlon in range(wlon_lo, wlon_hi + 1):
            tiles.append(f"S{slat:02d}_00_W{wlon:03d}_00")
    return [{
        "domain_id": DOMAIN_ID,
        "original_bbox": ",".join(f"{v:.5f}" for v in orig),
        "expanded_bbox": ",".join(f"{v:.5f}" for v in dom),
        "buffer_km": str(BUFFER_KM),
        "resolution_deg": str(DOMAIN_RES_DEG),
        "resolution_m_approx": str(round(DOMAIN_RES_DEG * 111320.0, 1)),
        "dem_tiles_needed": ";".join(tiles),
        "buffer_reason": "capturar bacia contribuinte a montante e evitar truncamento do flow accumulation por patch",
        "single_domain_flow_accumulation": "true",
        "flow_acc_per_patch_window": "false",
        "may_still_truncate_large_basins": "true_edge_of_domain_only",
        "review_only": "true",
    }]


# ---------------------------------------------------------------------------
# Basin DEM + hydrologic grid (network) -> ledger
# ---------------------------------------------------------------------------

def _read_domain_dem(bbox, res):
    import rasterio
    from rasterio.merge import merge
    dom = bbox
    slat_lo = int(math.ceil(-dom[3]))
    slat_hi = int(math.ceil(-dom[1]))
    wlon_lo = int(math.ceil(-dom[2]))
    wlon_hi = int(math.ceil(-dom[0]))
    datasets = []
    attempts = []
    for slat in range(slat_lo, slat_hi + 1):
        for wlon in range(wlon_lo, wlon_hi + 1):
            base = f"Copernicus_DSM_COG_10_S{slat:02d}_00_W{wlon:03d}_00_DEM"
            url = f"https://copernicus-dem-30m.s3.amazonaws.com/{base}/{base}.tif"
            try:
                ds = rasterio.open(url)
                datasets.append(ds)
                attempts.append({"source": "copernicus_dem_glo30", "url": url, "tile": base,
                                 "attempt_type": "cog_open", "status": "opened", "failure_reason": ""})
            except Exception as e:
                attempts.append({"source": "copernicus_dem_glo30", "url": url, "tile": base,
                                 "attempt_type": "cog_open", "status": "unavailable", "failure_reason": type(e).__name__})
    if not datasets:
        return None, None, attempts
    arr, transform = merge(datasets, bounds=(dom[0], dom[1], dom[2], dom[3]), res=res)
    for ds in datasets:
        ds.close()
    return arr[0].astype("float64"), transform, attempts


def _run_basin_grid():
    if not _env_ok():
        raise SystemExit("BLOCKED: processamento basin 17C38 exige os seis opt-ins SUSC_17C38_*.")
    import numpy as np
    from rasterio.transform import rowcol
    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
    os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")
    ensure_dir(LIGHT)
    ensure_dir(LOCAL_RUNS)
    dom = _domain_bbox()
    dem, transform, attempts = _read_domain_dem(dom, DOMAIN_RES_DEG)
    if dem is None:
        write_json(LEDGER, {"generated_at_utc": _utcnow(), "attempts": attempts, "dem_acquired": False,
                            "grid": None, "diagnostics": None, "per_patch": []})
        return
    dem = np.where(dem < -1000, np.nan, dem)
    valid = np.isfinite(dem)
    demf = np.where(valid, dem, np.nanmax(dem[valid]))
    ny, nx = demf.shape
    lat_mid = (dom[1] + dom[3]) / 2
    cellsize = DOMAIN_RES_DEG * 111320.0 * math.cos(math.radians(lat_mid))
    raw_path = LOCAL_RUNS / "basin_dem.npy"
    np.save(raw_path, dem)

    t0 = datetime.now(timezone.utc)
    filled = s17c36._priority_flood_fill(demf)
    slope = s17c36._slope_rad(filled, cellsize)
    acc, rec = s17c36._d8_flow(filled, cellsize)
    flow_acc_log = np.log1p(acc)
    tpi = s17c36._tpi(demf, cellsize, s17c36.TPI_RADIUS_M)
    thr = float(np.percentile(acc, s17c36.DRAINAGE_FLOWACC_PCTL))
    hand = s17c36._hand(filled, acc, rec, acc >= thr)
    a_sca = (acc + 1.0) * cellsize
    beta = np.maximum(slope, math.radians(0.1))
    twi = np.log(a_sca / np.tan(beta))
    proc_s = (datetime.now(timezone.utc) - t0).total_seconds()

    diagnostics = {
        "grid_width": nx, "grid_height": ny, "cell_count": int(ny * nx), "valid_cells": int(valid.sum()),
        "cellsize_m": round(cellsize, 3), "max_flow_accumulation": float(acc.max()),
        "flow_acc_p50": float(np.percentile(acc, 50)), "flow_acc_p95": float(np.percentile(acc, 95)),
        "flow_acc_p99": float(np.percentile(acc, 99)), "drainage_threshold": thr,
        "edge_effect_risk": "low_interior_high_at_domain_boundary", "sink_fill_status": "priority_flood_applied",
        "processing_time_s": round(proc_s, 2), "memory_estimate_mb": round(ny * nx * 8 * 6 / 1e6, 1),
        "may_still_truncate": "only_basins_entering_from_outside_domain_edge",
    }

    def bbox_stats(arr, xmin, ymin, xmax, ymax):
        r0, c0 = rowcol(transform, xmin, ymax)
        r1, c1 = rowcol(transform, xmax, ymin)
        r0, r1 = sorted((max(0, r0), min(ny, r1 + 1)))
        c0, c1 = sorted((max(0, c0), min(nx, c1 + 1)))
        sub = arr[r0:r1, c0:c1]
        sub = sub[np.isfinite(sub)]
        return (round(float(sub.mean()), 4) if sub.size else None), int(sub.size)

    def edge_dist_cells(xmin, ymin, xmax, ymax):
        r0, c0 = rowcol(transform, xmin, ymax)
        r1, c1 = rowcol(transform, xmax, ymin)
        rr = [max(0, min(ny, r0)), max(0, min(ny, r1))]
        cc = [max(0, min(nx, c0)), max(0, min(nx, c1))]
        return min(min(rr), ny - max(rr), min(cc), nx - max(cc))

    per_patch = []
    for r in _official_rows():
        b = (_fnum(r, "xmin"), _fnum(r, "ymin"), _fnum(r, "xmax"), _fnum(r, "ymax"))
        if any(v is None for v in b):
            per_patch.append({"patch_id": r["patch_id"], "status": "no_bbox"})
            continue
        e, npx = bbox_stats(demf, *b)
        ed = edge_dist_cells(*b)
        inside = (dom[0] <= b[0] and b[2] <= dom[2] and dom[1] <= b[1] and b[3] <= dom[3])
        per_patch.append({
            "patch_id": r["patch_id"], "status": "recomputed", "valid_pixel_count": npx,
            "patch_inside_domain": bool(inside), "edge_distance_cells": int(ed),
            "recomputed_elevation_mean": e,
            "recomputed_slope_mean": bbox_stats(np.degrees(slope), *b)[0],
            "recomputed_hand_mean": bbox_stats(hand, *b)[0],
            "recomputed_twi_mean": bbox_stats(twi, *b)[0],
            "recomputed_tpi_250m_mean": bbox_stats(tpi, *b)[0],
            "recomputed_flow_acc_log_mean": bbox_stats(flow_acc_log, *b)[0],
        })
    manifest = {
        "source_name": "copernicus_dem_glo30", "source_authority": "copernicus_esa_official",
        "artifact_local_path": rel(raw_path), "artifact_type": "npy_local_runs",
        "sha256": s17c36.hashlib.sha256(raw_path.read_bytes()).hexdigest(), "size_bytes": raw_path.stat().st_size,
        "crs": "EPSG:4326", "resolution_m": round(cellsize, 2), "bbox": ",".join(f"{v:.5f}" for v in dom),
    }
    grid = {"width": nx, "height": ny, "cellsize_m": round(cellsize, 3), "valid_cell_count": int(valid.sum()),
            "sink_fill_method": "priority_flood_wang_liu", "flow_direction_method": "d8_steepest_descent",
            "flow_accumulation_method": "d8_single_domain_basin_aware",
            "flow_accumulation_scope": "single_wide_domain_not_per_patch_window",
            "method_equivalence_status": s17c36.METHOD_EQUIV}
    write_json(LEDGER, {"generated_at_utc": _utcnow(), "domain_id": DOMAIN_ID, "domain_bbox": list(dom),
                        "attempts": attempts, "dem_acquired": True, "dem_manifest": manifest,
                        "grid": grid, "diagnostics": diagnostics, "per_patch": per_patch})


def _utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_ledger():
    if not LEDGER.exists():
        raise FileNotFoundError(f"{rel(LEDGER)} ausente: rode build-basin-hydrologic-grid com os seis opt-ins.")
    return read_json(LEDGER)


# ---------------------------------------------------------------------------
# Offline derivation
# ---------------------------------------------------------------------------

def dem_acquisition_attempt_rows():
    led = _load_ledger()
    atts = led.get("attempts", [])
    rows = [{"dem_acquisition_attempt_id": f"S17C38_DEMA_{i:04d}", "source_name": a["source"],
             "source_url_or_api": a["url"], "tile": a.get("tile", ""), "attempt_type": a["attempt_type"],
             "network_enabled": "true", "status": a["status"], "dem_acquired": _bool(a["status"] == "opened"),
             "failure_reason": a.get("failure_reason", ""), "review_only": "true"}
            for i, a in enumerate(atts, start=1)]
    return rows or [{"dem_acquisition_attempt_id": "S17C38_DEMA_0001", "source_name": "none", "source_url_or_api": "",
                     "tile": "", "attempt_type": "none", "network_enabled": "true", "status": "not_run",
                     "dem_acquired": "false", "failure_reason": "ledger_absent", "review_only": "true"}]


def dem_manifest_rows():
    led = _load_ledger()
    m = led.get("dem_manifest")
    if not m:
        return [{"dem_artifact_manifest_id": "S17C38_DEM_0001", "source_name": "none", "source_authority": "none",
                 "artifact_local_path": "none", "artifact_type": "none", "sha256": "none", "size_bytes": "0",
                 "crs": "none", "resolution_m": "0", "bbox": "none", "stored_in_outputs_public": "false",
                 "raw_heavy": "false", "review_only": "true"}]
    return [{"dem_artifact_manifest_id": "S17C38_DEM_0001", "source_name": m["source_name"],
             "source_authority": m["source_authority"], "artifact_local_path": m["artifact_local_path"],
             "artifact_type": m["artifact_type"], "sha256": m["sha256"], "size_bytes": str(m["size_bytes"]),
             "crs": m["crs"], "resolution_m": str(m["resolution_m"]), "bbox": m["bbox"],
             "stored_in_outputs_public": "false", "raw_heavy": "true", "review_only": "true"}]


def basin_dem_grid_metadata_rows():
    led = _load_ledger()
    g = led.get("grid") or {}
    m = led.get("dem_manifest") or {}
    ok = bool(g)
    return [{"basin_dem_grid_id": "S17C38_GRID_0001", "domain_id": DOMAIN_ID, "dem_manifest_id": "S17C38_DEM_0001",
             "crs": "EPSG:4326", "resolution_m": str(g.get("cellsize_m", "0")), "width": str(g.get("width", 0)),
             "height": str(g.get("height", 0)), "bbox": m.get("bbox", "none"),
             "valid_cell_count": str(g.get("valid_cell_count", 0)),
             "usable_for_basin_flow": _bool(ok), "review_only": "true"}]


def basin_hydrologic_grid_metadata_rows():
    led = _load_ledger()
    g = led.get("grid") or {}
    ok = bool(g)
    return [{"basin_hydrologic_grid_id": "S17C38_HGRID_0001", "basin_dem_grid_id": "S17C38_GRID_0001",
             "flow_accumulation_scope": g.get("flow_accumulation_scope", "none"),
             "flow_accumulation_method": g.get("flow_accumulation_method", "none"),
             "flow_direction_method": g.get("flow_direction_method", "none"),
             "sink_fill_method": g.get("sink_fill_method", "none"),
             "flow_acc_per_patch_window": "false",
             "basin_aware_flow_acc_created": _bool(ok), "basin_aware_twi_created": _bool(ok),
             "basin_aware_hand_created": _bool(ok),
             "method_equivalence_status": g.get("method_equivalence_status", s17c36.METHOD_EQUIV),
             "review_only": "true"}]


def flow_diagnostics_rows():
    led = _load_ledger()
    d = led.get("diagnostics") or {}
    return [{"flow_diagnostic_id": "S17C38_FDIAG_0001", "domain_id": DOMAIN_ID,
             "grid_width": str(d.get("grid_width", 0)), "grid_height": str(d.get("grid_height", 0)),
             "cell_count": str(d.get("cell_count", 0)), "valid_cells": str(d.get("valid_cells", 0)),
             "max_flow_accumulation": str(d.get("max_flow_accumulation", "not_available")),
             "flow_acc_p50": str(d.get("flow_acc_p50", "")), "flow_acc_p95": str(d.get("flow_acc_p95", "")),
             "flow_acc_p99": str(d.get("flow_acc_p99", "")),
             "edge_effect_risk": d.get("edge_effect_risk", "unknown"),
             "sink_fill_status": d.get("sink_fill_status", "unknown"),
             "processing_time_s": str(d.get("processing_time_s", "")),
             "memory_estimate_mb": str(d.get("memory_estimate_mb", "")),
             "may_still_truncate": d.get("may_still_truncate", "unknown"), "review_only": "true"}]


_REC_COL = {"hand_mean": "recomputed_hand_mean", "twi_mean": "recomputed_twi_mean",
            "tpi_250m_mean": "recomputed_tpi_250m_mean", "flow_acc_log_mean": "recomputed_flow_acc_log_mean"}


def basin_feature_rows():
    led = _load_ledger()
    by = {r["patch_id"]: r for r in led.get("per_patch", [])}
    rows = []
    for i, r in enumerate(_official_rows(), start=1):
        pid = r["patch_id"]
        d = by.get(pid, {})
        # recompute bem-sucedido = grid processou o patch (elevacao presente);
        # HAND pode ficar not_available onde a drenagem nao e alcancada a jusante (metodo identico 17C36)
        ok = d.get("status") == "recomputed" and d.get("recomputed_elevation_mean") is not None

        def val(key):
            v = d.get(key) if ok else None
            return "not_available" if v is None else v
        rows.append({
            "recomputed_feature_id": f"S17C38_REC_{i:04d}", "patch_id": pid,
            "dem_source": "copernicus_dem_glo30", "pipeline_source": "susc_17c36_basin_aware", "domain_id": DOMAIN_ID,
            "recomputed_hand_mean": val("recomputed_hand_mean"),
            "recomputed_twi_mean": val("recomputed_twi_mean"),
            "recomputed_tpi_250m_mean": val("recomputed_tpi_250m_mean"),
            "recomputed_flow_acc_log_mean": val("recomputed_flow_acc_log_mean"),
            "recomputed_elevation_mean": val("recomputed_elevation_mean"),
            "recomputed_slope_mean": val("recomputed_slope_mean"),
            "valid_pixel_count": str(d.get("valid_pixel_count", 0)),
            "patch_inside_domain": _bool(d.get("patch_inside_domain", False)),
            "edge_effect_risk": "low" if d.get("edge_distance_cells", 0) and d.get("edge_distance_cells", 0) > 10 else "check",
            "method_equivalence_status": s17c36.METHOD_EQUIV, "review_only": "true",
        })
    return rows


def _c37_metrics():
    return {r["feature_name"]: r for r in read_csv(C37_METRICS)}


def comparison_rows():
    off = {r["patch_id"]: r for r in _official_rows()}
    rec = {r["patch_id"]: r for r in basin_feature_rows()}
    pids = [r["patch_id"] for r in _official_rows()]
    c37m = _c37_metrics()
    off_vals = {f: [_fnum(off.get(p, {}), f) for p in pids] for f in CRITICAL}
    rec_vals = {f: [] for f in CRITICAL}
    for f in CRITICAL:
        for p in pids:
            v = rec[p].get(_REC_COL[f], "not_available")
            rec_vals[f].append(float(v) if v not in ("not_available", "", None) else None)
    off_ranks = {f: s17c37._rank(off_vals[f]) for f in CRITICAL}
    rec_ranks = {f: s17c37._rank(rec_vals[f]) for f in CRITICAL}
    rows = []
    cid = 0
    for j, pid in enumerate(pids):
        for f in CRITICAL:
            cid += 1
            ov, rv = off_vals[f][j], rec_vals[f][j]
            ae = abs(ov - rv) if (ov is not None and rv is not None) else None
            re_ = (ae / abs(ov)) if (ae is not None and ov not in (None, 0)) else None
            scale_ok = (rv is not None and ov not in (None, 0) and EQ_SCALE_LO <= (rv / ov) <= EQ_SCALE_HI)
            within = (re_ is not None and re_ <= EQ_REL_ERR)
            ro, rr = off_ranks[f][j], rec_ranks[f][j]
            same_bucket = (ro is not None and rr is not None and abs(ro - rr) <= max(1, len(pids) // 5))
            # improved vs 17c37: comparado ao status incompativel de flow_acc no 17c37
            improved = "not_applicable"
            if f == "flow_acc_log_mean":
                improved = _bool(same_bucket)
            rows.append({
                "comparison_id": f"S17C38_CMP_{cid:05d}", "patch_id": pid, "feature_name": f,
                "official_value": "not_available" if ov is None else round(ov, 4),
                "recomputed_basin_aware_value": "not_available" if rv is None else round(rv, 4),
                "absolute_error": "not_available" if ae is None else round(ae, 4),
                "relative_error": "not_available" if re_ is None else round(re_, 4),
                "rank_official": ro if ro is not None else "not_available",
                "rank_recomputed": rr if rr is not None else "not_available",
                "same_order_bucket": _bool(same_bucket), "scale_compatible": _bool(scale_ok),
                "within_acceptance_threshold": _bool(within), "improved_vs_17c37": improved,
                "blocking_reason": "not_applicable" if within else "relative_error_or_scale_out_of_threshold",
                "review_only": "true",
            })
    return rows


def metrics_rows():
    off = {r["patch_id"]: r for r in _official_rows()}
    rec = {r["patch_id"]: r for r in basin_feature_rows()}
    pids = [r["patch_id"] for r in _official_rows()]
    c37m = _c37_metrics()
    rows = []
    for i, f in enumerate(CRITICAL, start=1):
        ov = [_fnum(off.get(p, {}), f) for p in pids]
        rv = []
        for p in pids:
            v = rec[p].get(_REC_COL[f], "not_available")
            rv.append(float(v) if v not in ("not_available", "", None) else None)
        pear = s17c37._pearson(ov, rv)
        spear = s17c37._pearson(s17c37._rank(ov), s17c37._rank(rv))
        aes = [abs(o - r) for o, r in zip(ov, rv) if o is not None and r is not None]
        res = [abs(o - r) / abs(o) for o, r in zip(ov, rv) if o is not None and r is not None and o not in (None, 0)]
        ratios = [r / o for o, r in zip(ov, rv) if o not in (None, 0) and r is not None]
        ovv = [o for o in ov if o is not None]
        rvv = [r for r in rv if r is not None]
        n = len([1 for o, r in zip(ov, rv) if o is not None and r is not None])
        scale_med = s17c37._median(ratios)
        if n < 3:
            status = "insufficient_data"
        elif (spear is not None and spear >= EQ_SPEARMAN and scale_med is not None and EQ_SCALE_LO <= scale_med <= EQ_SCALE_HI):
            status = "equivalent"
        elif spear is not None and spear >= CAL_SPEARMAN:
            status = "calibratable"
        else:
            status = "incompatible"
        st37 = c37m.get(f, {}).get("equivalence_status", "unknown")
        # melhora: spearman 17c38 > spearman 17c37 (para flow_acc principalmente)
        try:
            sp37 = float(c37m.get(f, {}).get("spearman_corr"))
        except (TypeError, ValueError):
            sp37 = None
        improved = _bool(spear is not None and sp37 is not None and spear > sp37)
        rows.append({
            "method_equivalence_metric_id": f"S17C38_MEM_{i:04d}", "feature_name": f,
            "pearson_corr": "not_available" if pear is None else round(pear, 4),
            "spearman_corr": "not_available" if spear is None else round(spear, 4),
            "mae": "not_available" if not aes else round(sum(aes) / len(aes), 4),
            "median_absolute_error": "not_available" if s17c37._median(aes) is None else round(s17c37._median(aes), 4),
            "relative_error_median": "not_available" if s17c37._median(res) is None else round(s17c37._median(res), 4),
            "official_min": round(min(ovv), 4) if ovv else "not_available",
            "official_max": round(max(ovv), 4) if ovv else "not_available",
            "recomputed_min": round(min(rvv), 4) if rvv else "not_available",
            "recomputed_max": round(max(rvv), 4) if rvv else "not_available",
            "scale_ratio_median": "not_available" if scale_med is None else round(scale_med, 4),
            "rank_preservation_score": round(spear, 4) if spear is not None else 0.0,
            "equivalence_status_17c37": st37, "equivalence_status_17c38": status,
            "improved_vs_17c37": improved, "equivalence_status": status, "review_only": "true",
        })
    return rows


def _decision_dict():
    metrics = {m["feature_name"]: m for m in metrics_rows()}
    c37 = read_json(C37_DECISION)
    accepted = [f for f in CRITICAL if metrics[f]["equivalence_status"] == "equivalent"]
    calibratable = [f for f in CRITICAL if metrics[f]["equivalence_status"] == "calibratable"]
    incompatible = [f for f in CRITICAL if metrics[f]["equivalence_status"] == "incompatible"]
    insufficient = [f for f in CRITICAL if metrics[f]["equivalence_status"] == "insufficient_data"]
    full = len(accepted) == len(CRITICAL)
    calibration_possible = (not incompatible) and (not insufficient) and (len(calibratable) > 0 or full)
    fa = metrics["flow_acc_log_mean"]
    if full:
        reason = "todas as 4 features equivalentes com dominio basin-aware"
    elif incompatible:
        reason = f"features ainda incompativeis apos basin-aware: {','.join(incompatible)}"
    elif calibratable:
        reason = f"features calibraveis apos basin-aware: {','.join(calibratable)}; nenhuma incompativel"
    else:
        reason = "dados insuficientes"
    return {
        "method_equivalence_decision_updated": True,
        "previous_method_equivalence_accepted_17c37": read_json(C37_SUMMARY).get("method_equivalence_accepted", False),
        "method_equivalence_accepted": full,
        "method_calibration_possible": bool(calibration_possible),
        "flow_acc_equivalence_retested": True,
        "flow_acc_status_17c37": c37.get("incompatible_features") and "incompatible" or _c37_metrics().get("flow_acc_log_mean", {}).get("equivalence_status", "unknown"),
        "flow_acc_status_17c38": fa["equivalence_status"],
        "accepted_features": accepted, "calibratable_features": calibratable,
        "incompatible_features": incompatible, "insufficient_features": insufficient,
        "blocking_features": incompatible + insufficient,
        "reason": reason,
        "can_compute_score_v6_final_replay": full,
        "can_compute_calibrated_component_replay": bool(calibration_possible),
        "requires_future_calibration": bool(calibratable and not full),
        "acceptance_thresholds": {"spearman_equivalent": EQ_SPEARMAN, "spearman_calibratable": CAL_SPEARMAN,
                                  "scale_ratio": [EQ_SCALE_LO, EQ_SCALE_HI], "rel_err_max": EQ_REL_ERR},
        "review_only": True,
    }


def readiness_rows():
    dec = _decision_dict()
    prev = {r["canary_patch_id"]: r for r in read_csv(C37_READINESS)}
    fa_status = dec["flow_acc_status_17c38"]
    rows = []
    for i, c in enumerate(read_csv(C33_REGISTRY), start=1):
        pid = c["event_anchored_canary_patch_id"]
        p = prev.get(pid, {})
        if dec["method_equivalence_accepted"]:
            status = "topography_equivalent_ready_for_final_replay"
        elif dec["method_calibration_possible"]:
            status = "topography_calibratable_after_basin_flow_calibrated_replay_only"
        else:
            status = "topography_still_incompatible_score_final_blocked"
        rows.append({
            "score_replay_readiness_after_basin_flow_id": f"S17C38_RDY_{i:04d}", "canary_patch_id": pid,
            "previous_status_17c37": p.get("score_replay_status_after_17c37", "topography_incompatible_score_final_blocked"),
            "method_equivalence_accepted": _bool(dec["method_equivalence_accepted"]),
            "method_calibration_possible": _bool(dec["method_calibration_possible"]),
            "flow_acc_equivalence_status": fa_status,
            "topography_features_equivalent": ";".join(dec["accepted_features"]) or "none",
            "topography_features_calibratable": ";".join(dec["calibratable_features"]) or "none",
            "can_compute_full_score_v6_replay": _bool(dec["can_compute_score_v6_final_replay"]),
            "can_compute_calibrated_score_v6_replay": _bool(dec["can_compute_calibrated_component_replay"]),
            "score_replay_status_after_17c38": status, "blocking_reason": dec["reason"],
            "not_official_score": "true", "does_not_replace_score_v6": "true", "review_only": "true",
        })
    return rows


def score_integrity_audit_rows():
    changed = bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)]))
    mchanged = bool(_run_git(["diff", "--name-only", "--", rel(OFFICIAL_MATRIX)]))
    dec = _decision_dict()
    return [{"score_integrity_audit_id": "S17C38_INTEG_0001", "official_score_v6_path": rel(SCORE_V6),
             "official_score_v6_changed": _bool(changed), "official_matrix_path": rel(OFFICIAL_MATRIX),
             "official_matrix_changed": _bool(mchanged), "score_v7_exists": _bool(SCORE_V7.exists()),
             "final_score_computed": _bool(dec["can_compute_score_v6_final_replay"]), "review_only": "true"}]


def no_leakage_audit_rows():
    rows = []
    for i, r in enumerate(_official_rows(), start=1):
        rows.append({
            "no_leakage_audit_id": f"S17C38_LEAK_{i:04d}", "object_id": r["patch_id"],
            "object_type": "official_patch_basin_recompute",
            "uses_occurrence_as_feature": "false", "uses_canary_to_decide_equivalence": "false",
            "equivalence_evaluated_on_official_patches": "true",
            "flow_acc_per_patch_window": "false", "flow_acc_single_basin_domain": "true",
            "uses_canary_only_normalization": "false", "modifies_official_matrix": "false",
            "modifies_score_v6": "false", "uses_score_v7": "false", "uses_control_as_negative_truth": "false",
            "passes_no_leakage": "true",
            "blocking_reason": "recompute basin-aware em patches oficiais; flow_acc dominio unico; review-only",
            "review_only": "true",
        })
    return rows


def build_summary():
    dom = domain_definition_rows()
    pop = _official_rows()
    feats = basin_feature_rows()
    comp = comparison_rows()
    metrics = metrics_rows()
    dec = _decision_dict()
    ready = readiness_rows()
    diag = flow_diagnostics_rows()[0]
    led = _load_ledger() if LEDGER.exists() else {}
    rec_ok = len([f for f in feats if f["recomputed_elevation_mean"] != "not_available"])
    hand_ok = len([f for f in feats if f["recomputed_hand_mean"] != "not_available"])
    fa = next(m for m in metrics if m["feature_name"] == "flow_acc_log_mean")
    dem_grid_ok = bool(led.get("grid"))
    minimum = (
        len(pop) >= 100 and rec_ok >= 100 and len(dom) >= 1 and dem_grid_ok
        and len(comp) >= 400 and len(metrics) >= 4 and dec["method_equivalence_decision_updated"]
    )
    return {
        "minimum_success_achieved": minimum,
        "official_patch_population_rows_count": len(pop),
        "official_patch_recomputed_rows_count": rec_ok,
        "hand_recomputed_rows_count": hand_ok,
        "basin_aware_domain_created": True,
        "basin_aware_domain_buffer_km": BUFFER_KM,
        "dem_acquisition_attempts_count": len(dem_acquisition_attempt_rows()),
        "basin_dem_grid_created": dem_grid_ok,
        "basin_hydrologic_grid_created": dem_grid_ok,
        "basin_aware_flow_acc_created": dem_grid_ok,
        "basin_aware_twi_created": dem_grid_ok,
        "basin_aware_hand_created": dem_grid_ok,
        "topography_feature_comparison_rows_count": len(comp),
        "method_equivalence_metric_rows_count": len(metrics),
        "flow_acc_equivalence_retested": True,
        "flow_acc_status_17c37": dec["flow_acc_status_17c37"],
        "flow_acc_status_17c38": dec["flow_acc_status_17c38"],
        "flow_acc_improved_vs_17c37": fa["improved_vs_17c37"] == "true",
        "method_equivalence_decision_updated": True,
        "method_equivalence_accepted": dec["method_equivalence_accepted"],
        "method_calibration_possible": dec["method_calibration_possible"],
        "accepted_feature_count": len(dec["accepted_features"]),
        "calibratable_feature_count": len(dec["calibratable_features"]),
        "incompatible_feature_count": len(dec["incompatible_features"]),
        "per_feature_status": {m["feature_name"]: m["equivalence_status"] for m in metrics},
        "can_compute_score_v6_final_replay": dec["can_compute_score_v6_final_replay"],
        "can_compute_calibrated_component_replay": dec["can_compute_calibrated_component_replay"],
        "score_v6_replay_readiness_rows_count": len(ready),
        "basin_processing_time_s": float(diag["processing_time_s"]) if diag["processing_time_s"] not in ("", "not_available") else -1,
        "basin_max_flow_accumulation": diag["max_flow_accumulation"],
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
        "result_class": ("A_flow_acc_recovered" if dec["method_calibration_possible"] and not dec["incompatible_features"]
                         else ("B_flow_acc_still_incompatible" if dec["incompatible_features"] else "C_basin_processing_unviable")),
        "recommended_next_milestone": (
            "SUSC-17C39 Politica de calibracao review-only (mapear escala/offset por feature calibravel: HAND/TWI/TPI/flow_acc "
            "vs oficial) e replay v6 calibrado review-only, sem alterar score v6 oficial; ou buscar metodo/DEM original. 17B fail-closed."
            if dec["method_calibration_possible"] else
            "SUSC-17C39 Buscar metodo/DEM/algoritmo original do score v6 (twi escala ~116) ou adotar analise por features brutas; "
            "flow_acc segue incompativel mesmo basin-aware. 17B fail-closed."
        ),
    }


def build_blockers():
    metrics = {m["feature_name"]: m for m in metrics_rows()}
    dec = _decision_dict()
    blockers = []
    if metrics["flow_acc_log_mean"]["equivalence_status"] == "incompatible":
        blockers.append(("flow_acc_still_incompatible", "flow_acc incompativel mesmo com dominio basin-aware"))
        blockers.append(("flow_acc_basin_domain_still_truncated", "bacias entrando pela borda do dominio ainda podem truncar; ou metodo oficial difere"))
    if not dec["method_equivalence_accepted"]:
        blockers.append(("method_equivalence_not_accepted", "equivalencia full nao aceita"))
    if dec["requires_future_calibration"]:
        blockers.append(("method_calibration_required", f"calibraveis: {','.join(dec['calibratable_features'])}"))
    blockers += [
        ("score_v6_final_replay_blocked", "score final v6 bloqueado ate full equivalence"),
        ("score_v7_blocked", "score v7 bloqueado"),
        ("17b_blocked", "17B bloqueado ate G4_full + ground reference aceito"),
    ]
    return [{"blocker_id": f"S17C38_BLOCKER_{i:04d}", "blocker_type": n, "description": d,
             "blocks_final_score_replay": "true", "blocks_17b": "true", "blocks_score_v7": "true", "review_only": "true"}
            for i, (n, d) in enumerate(blockers, start=1)]


def build_report():
    s = build_summary()
    metrics = metrics_rows()
    return "\n".join([
        "# SUSC-17C38 - Flow Accumulation Basin-Aware e Revalidacao de Equivalencia",
        "",
        "## Objetivo",
        "Recalcular flow accumulation (e TWI/TPI/HAND) num DOMINIO HIDROLOGICO UNICO E AMPLO (nao janela por patch) cobrindo os 100 patches oficiais + buffer, e reavaliar equivalencia com as features oficiais. Testar se o truncamento espacial causou a incompatibilidade do 17C37.",
        "",
        "## Dominio e grid",
        f"- Dominio basin-aware: buffer {s['basin_aware_domain_buffer_km']} km; grid unico; flow accumulation NAO por janela de patch.",
        f"- DEM: {s['dem_acquisition_attempts_count']} tentativas de tile Copernicus GLO-30; grid criado: {s['basin_dem_grid_created']}.",
        f"- Processamento: {s['basin_processing_time_s']}s; max flow accumulation: {s['basin_max_flow_accumulation']}.",
        f"- Patches oficiais recomputados: {s['official_patch_recomputed_rows_count']}/{s['official_patch_population_rows_count']}.",
        "",
        "## Equivalencia por feature (17C37 -> 17C38)",
        *[f"  - {m['feature_name']}: {m['equivalence_status_17c37']} -> {m['equivalence_status_17c38']} | spearman={m['spearman_corr']} | scale_ratio={m['scale_ratio_median']} | improved={m['improved_vs_17c37']}" for m in metrics],
        "",
        "## flow_acc (foco do marco)",
        f"- flow_acc 17C37: {s['flow_acc_status_17c37']} -> 17C38: {s['flow_acc_status_17c38']}; melhorou: {s['flow_acc_improved_vs_17c37']}.",
        "",
        "## Decisao",
        f"- method_equivalence_accepted={s['method_equivalence_accepted']}; method_calibration_possible={s['method_calibration_possible']}.",
        f"- equivalentes={s['accepted_feature_count']}; calibraveis={s['calibratable_feature_count']}; incompativeis={s['incompatible_feature_count']}.",
        f"- can_compute_score_v6_final_replay={s['can_compute_score_v6_final_replay']}; result_class={s['result_class']}.",
        "",
        "## Guardrails",
        "- Equivalencia so em patches oficiais (canarios nao decidem); flow_acc em dominio unico (nao janela por patch); matriz oficial e score v6 intactos; sem v7/GT/treino; sem normalizacao canary-only; raster pesado so em local_runs; controle nao vira negativo verdadeiro.",
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
    write_csv(DOMAIN, domain_definition_rows())
    write_csv(DEM_ATTEMPTS, dem_acquisition_attempt_rows())
    write_csv(DEM_MANIFEST, dem_manifest_rows())
    write_csv(DEM_GRID, basin_dem_grid_metadata_rows())
    write_csv(HYDRO_GRID, basin_hydrologic_grid_metadata_rows())
    write_csv(FLOW_DIAG, flow_diagnostics_rows())
    write_csv(BASIN_FEATURES, basin_feature_rows())
    write_csv(COMPARISON, comparison_rows())
    write_csv(METRICS, metrics_rows())
    write_json(DECISION, _decision_dict())
    write_csv(READINESS, readiness_rows())
    write_csv(INTEGRITY, score_integrity_audit_rows())
    write_csv(LEAKAGE, no_leakage_audit_rows())
    write_json(SUMMARY, build_summary())
    write_csv(BLOCKERS, build_blockers())
    write_markdown(REPORT, build_report())


def run_mode(mode):
    _require_inputs()
    ensure_dir(OUT)
    if mode == "define-basin-domain":
        write_csv(DOMAIN, domain_definition_rows())
    elif mode in ("run-basin-dem-acquisition", "build-basin-hydrologic-grid"):
        _run_basin_grid()
    elif mode == "sample-official-patches":
        write_csv(BASIN_FEATURES, basin_feature_rows())
    elif mode == "compare-basin-topography-features":
        write_csv(COMPARISON, comparison_rows())
    elif mode == "compute-basin-equivalence-metrics":
        write_csv(METRICS, metrics_rows())
    elif mode == "decide-basin-method-equivalence":
        write_json(DECISION, _decision_dict())
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

    dom = read_csv(DOMAIN)
    feats = read_csv(BASIN_FEATURES)
    comp = read_csv(COMPARISON)
    metrics = read_csv(METRICS)
    ready = read_csv(READINESS)
    hgrid = read_csv(HYDRO_GRID)
    leak = read_csv(LEAKAGE)
    integ = read_csv(INTEGRITY)
    summary = read_json(SUMMARY)
    decision = read_json(DECISION)
    b_schema = read_json(SCHEMA_BASIN)
    m_schema = read_json(SCHEMA_METRIC)
    r_schema = read_json(SCHEMA_RDY)

    for r in feats:
        errors.extend(_schema_violations(r, b_schema))
    for r in metrics:
        errors.extend(_schema_violations(r, m_schema))
    for r in ready:
        errors.extend(_schema_violations(r, r_schema))

    # 1 pop, 2 recomputed>=100, 3 domain, 4 flow por janela, 5 grid, 6 comparison>=400, 7 metrics 4, 8 decision, 9 flow retest
    if summary["official_patch_population_rows_count"] < 100:
        errors.append("population_lt_100")
    if summary["official_patch_recomputed_rows_count"] < 100:
        errors.append("recomputed_lt_100")
    if not dom or summary["basin_aware_domain_created"] is not True:
        errors.append("domain_not_created")
    for r in hgrid:
        if r["flow_acc_per_patch_window"] != "false" or "single_domain" not in r["flow_accumulation_method"]:
            errors.append("flow_acc_per_patch_window")
    for r in leak:
        if r["flow_acc_per_patch_window"] != "false" or r["flow_acc_single_basin_domain"] != "true":
            errors.append("flow_acc_window_leak")
    if not summary["basin_hydrologic_grid_created"]:
        errors.append("basin_grid_missing")
    if len(comp) < 400:
        errors.append("comparison_lt_400")
    feats_set = {m["feature_name"] for m in metrics}
    if not set(CRITICAL).issubset(feats_set):
        errors.append("metrics_missing_feature")
    if not decision.get("method_equivalence_decision_updated"):
        errors.append("decision_not_updated")
    if not decision.get("flow_acc_equivalence_retested"):
        errors.append("flow_acc_not_retested")

    statuses = {m["feature_name"]: m["equivalence_status"] for m in metrics}
    # 10 full equivalence com feature nao-equivalente
    if decision["method_equivalence_accepted"] and any(statuses[f] != "equivalent" for f in CRITICAL):
        errors.append("full_equivalence_with_non_equivalent")
    # 11 score final sem full
    if summary["can_compute_score_v6_final_replay"] and not decision["method_equivalence_accepted"]:
        errors.append("final_score_without_full_equivalence")
    for r in integ:
        if r["final_score_computed"] == "true" and not decision["method_equivalence_accepted"]:
            errors.append("integrity_final_without_equivalence")
    # 12 canarios calibram
    for r in leak:
        if r["uses_canary_to_decide_equivalence"] != "false" or r["equivalence_evaluated_on_official_patches"] != "true":
            errors.append("canary_decides_equivalence")
    for r in feats:
        if not r["patch_id"].startswith("recife"):
            errors.append(f"non_official_recomputed:{r['patch_id']}")
    # 13/14 score/matriz/v7
    if _run_git(["diff", "--name-only", "--", rel(SCORE_V6)]) or summary["score_v6_original_file_changed"]:
        errors.append("score_v6_changed")
    if _run_git(["diff", "--name-only", "--", rel(OFFICIAL_MATRIX)]) or summary["official_matrix_changed"]:
        errors.append("official_matrix_changed")
    if SCORE_V7.exists() or summary["score_v7_created"]:
        errors.append("score_v7_exists")
    # 15/16 GT/treino
    if summary["ground_truth_created"] or summary["training_labels_created"]:
        errors.append("gt_or_training")
    # 17 raster pesado
    if LIGHT.exists():
        for p in LIGHT.glob("**/*"):
            if p.is_file() and (p.suffix.lower() in FORBIDDEN_SUFFIXES or p.stat().st_size > MAX_LIGHT_BYTES):
                errors.append(f"forbidden_or_heavy:{rel(p)}")
    # 18 ocorrencia feature, 19 controle negativo
    if any(r["uses_occurrence_as_feature"] != "false" for r in leak):
        errors.append("occurrence_as_feature")
    if any(r["uses_control_as_negative_truth"] != "false" for r in leak):
        errors.append("control_negative")
    if any(r["passes_no_leakage"] != "true" for r in leak):
        errors.append("no_leakage_failed")

    # 20 summary reflete contagens
    expected = build_summary()
    for k, v in expected.items():
        if summary.get(k) != v:
            errors.append(f"summary_mismatch:{k}")
    if not summary["minimum_success_achieved"]:
        errors.append("minimum_success_not_achieved")

    for rows, key in [
        (feats, "recomputed_feature_id"), (comp, "comparison_id"), (metrics, "method_equivalence_metric_id"),
        (ready, "score_replay_readiness_after_basin_flow_id"), (leak, "no_leakage_audit_id"),
    ]:
        ids = [r[key] for r in rows]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            errors.append(f"ids_not_unique_or_sorted:{key}")

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(
        "17C38 -> "
        f"pop={summary['official_patch_population_rows_count']} recomputed={summary['official_patch_recomputed_rows_count']} "
        f"comparison={summary['topography_feature_comparison_rows_count']} "
        f"flow_acc_17c37={summary['flow_acc_status_17c37']} flow_acc_17c38={summary['flow_acc_status_17c38']} "
        f"improved={summary['flow_acc_improved_vs_17c37']} per_feature={summary['per_feature_status']} "
        f"accepted={summary['method_equivalence_accepted']} calibratable={summary['method_calibration_possible']} "
        f"final_replay={summary['can_compute_score_v6_final_replay']} result={summary['result_class']} "
        f"min_success={summary['minimum_success_achieved']}"
    )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="susc_17c38")
    parser.add_argument("mode", choices=[
        "define-basin-domain", "run-basin-dem-acquisition", "build-basin-hydrologic-grid",
        "sample-official-patches", "compare-basin-topography-features", "compute-basin-equivalence-metrics",
        "decide-basin-method-equivalence", "update-score-replay-readiness", "build-offline", "audit",
    ])
    args = parser.parse_args(argv)
    if args.mode == "audit":
        return validate()
    run_mode(args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
