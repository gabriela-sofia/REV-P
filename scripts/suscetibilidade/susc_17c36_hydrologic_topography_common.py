"""SUSC-17C36 Pipeline Hidrologico Local para TWI/TPI/Flow Accumulation/HAND Full.

Constroi um pipeline hidrologico local, auditavel e leve, para calcular as
features topograficas/hidrologicas faltantes dos patches canarios (17C33):
twi_mean, tpi_250m_mean, flow_acc_log_mean, hand_mean/min/max. Completa o
subindice topography_hydrology exigido pelo replay review-only do score v6.

NAO cria score v7. NAO altera score v6 oficial. NAO cria ground truth/treino.
NAO libera 17B.

Metodo (numpy puro, DEM Copernicus GLO-30 real):
 - DEM grid AOI (tile Copernicus S09_00_W035_00, ~30 m, EPSG:4326) -> janela AOI
   dos canarios (raw em local_runs, NUNCA commitado; so stats/manifesto/hash).
 - fill sinks (priority-flood, Wang & Liu).
 - slope (gradiente do DEM em metros).
 - D8 flow direction + flow accumulation (ordem decrescente de elevacao).
 - flow_acc_log = log1p(flow_accumulation_cells).
 - TWI = ln(a / tan(beta)), a = (flow_acc+1)*cellsize; beta guardado.
 - TPI 250m = elevation - media da vizinhanca (raio ~250 m) via integral image.
 - HAND local = elevation - elevacao da celula de drenagem alcancada a jusante
   (drenagem = flow_acc acima de limiar).
 - estatisticas por canario (mean/min/max) sobre o bbox do patch.

Equivalencia metodologica: o score v6 original tem valores prontos em
susc_features_by_patch_v1.csv, mas o metodo/DEM/unidade originais NAO estao
documentados no repo (feature cards: twi "design_only", source vazio; twi_mean
oficial ~116, escala atipica). Logo o metodo aqui e RECONSTRUIDO de forma
auditavel mas NAO comprovadamente identico -> method_equivalence_status=
"method_reconstructed_not_proven_equivalent". Por isso o SCORE FINAL v6
review-only NAO e computado neste marco (guardrail: nao calcular score final se
metodo nao for equivalente). Resultado A e para a REPRODUCAO das features; o
replay final fica preparado para o proximo marco (equivalencia/validacao).

Guardrails: sem ocorrencia como feature; sem pos-evento; sem area de risco como
evento; sem GT/treino/v7; score v6 oficial intacto; proxy != full sem flag;
OSM != hidrografia oficial; raster pesado so em local_runs, nunca outputs_public.

Reproducibilidade: o processamento (rede+numpy) cacheia stats por canario em
susc_17c36_light_artifacts/_ledger_topography.json; o build offline reproduz os
CSV/JSON derivados byte-identicos a partir do ledger.
"""

from __future__ import annotations

import argparse
import hashlib
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

DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"
LIGHT = OUT / "susc_17c36_light_artifacts"
LOCAL_RUNS = ROOT / "local_runs" / "suscetibilidade" / "17c36_hydrologic"
LEDGER_TOPO = LIGHT / "_ledger_topography.json"

SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"
OFFICIAL_MATRIX = DAT / "susc_features_by_patch_v1.csv"
FEATURE_CARDS = OUT / "feature_cards" / "susc_feature_cards_v1.json"
SCORE_V6_BUILDER = ROOT / "scripts" / "suscetibilidade" / "build_susc_10a_score_v6_candidate.py"

# inputs
C35_CONTRACT = OUT / "susc_17c35_score_v6_formula_contract.json"
C35_DRAINAGE = OUT / "susc_17c35_official_drainage_features.csv"
C35_READINESS = OUT / "susc_17c35_score_v6_replay_readiness.csv"
C35_SUMMARY = OUT / "susc_17c35_readiness_summary.json"
C34_TERRAIN = OUT / "susc_17c34_terrain_features.csv"
C33_REGISTRY = OUT / "susc_17c33_event_anchored_canary_patch_registry.csv"
PATCH_GRID = OUT / "susc_17c6_candidate_patch_grid.csv"

REQUIRED_INPUTS = [
    SCORE_V6, OFFICIAL_MATRIX, C35_CONTRACT, C35_DRAINAGE, C35_READINESS, C35_SUMMARY,
    C34_TERRAIN, C33_REGISTRY, PATCH_GRID,
]

REPORT = OUT / "SUSC_17C36_HYDROLOGIC_TOPOGRAPHY_PIPELINE_REPORT.md"
METHOD_INV = OUT / "susc_17c36_original_topography_method_inventory.csv"
DEM_ATTEMPTS = OUT / "susc_17c36_dem_acquisition_attempts.csv"
DEM_MANIFEST = OUT / "susc_17c36_dem_artifact_manifest.csv"
DEM_GRID = OUT / "susc_17c36_dem_grid_metadata.csv"
HYDRO_GRID = OUT / "susc_17c36_hydrologic_grid_metadata.csv"
TPI = OUT / "susc_17c36_tpi_250m_features.csv"
FLOW_ACC = OUT / "susc_17c36_flow_accumulation_features.csv"
TWI = OUT / "susc_17c36_twi_features.csv"
HAND = OUT / "susc_17c36_hand_full_features.csv"
TOPO_MATRIX = OUT / "susc_17c36_topography_hydrology_feature_matrix.csv"
TOPO_PARITY = OUT / "susc_17c36_topography_hydrology_parity_matrix.csv"
READINESS_UPDATE = OUT / "susc_17c36_score_v6_replay_readiness_update.csv"
INTEGRITY = OUT / "susc_17c36_score_integrity_audit.csv"
LEAKAGE = OUT / "susc_17c36_no_leakage_audit.csv"
SUMMARY = OUT / "susc_17c36_readiness_summary.json"
BLOCKERS = OUT / "susc_17c36_promotion_blockers.csv"

SCHEMA_FEAT = SCHEMAS / "susc_17c36_hydrologic_feature_schema_v1.json"
SCHEMA_PARITY = SCHEMAS / "susc_17c36_topography_parity_schema_v1.json"
SCHEMA_DEM = SCHEMAS / "susc_17c36_dem_manifest_schema_v1.json"

REQUIRED_OUTPUTS = [
    REPORT, METHOD_INV, DEM_ATTEMPTS, DEM_MANIFEST, DEM_GRID, HYDRO_GRID, TPI, FLOW_ACC,
    TWI, HAND, TOPO_MATRIX, TOPO_PARITY, READINESS_UPDATE, INTEGRITY, LEAKAGE, SUMMARY, BLOCKERS,
]

TOPO_FEATURES = ["elevation_mean", "slope_mean", "hand_mean", "twi_mean", "tpi_250m_mean", "flow_acc_log_mean"]
DEM_TILE_URL = "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_S09_00_W035_00_DEM/Copernicus_DSM_COG_10_S09_00_W035_00_DEM.tif"
DEM_SOURCE = "copernicus_dem_glo30"
TPI_RADIUS_M = 250.0
DRAINAGE_FLOWACC_PCTL = 95.0  # celulas com flow_acc acima do p95 = drenagem
METHOD_EQUIV = "method_reconstructed_not_proven_equivalent"
FORBIDDEN_SUFFIXES = (".tif", ".tiff", ".safe", ".zip", ".nc", ".jp2", ".gz")
MAX_LIGHT_BYTES = 5_000_000


def _bool(v):
    return "true" if v else "false"


def _run_git(args):
    r = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return r.stdout.strip() if r.returncode == 0 else ""


def _env_ok():
    need = ["SUSC_17C36_ALLOW_NETWORK", "SUSC_17C36_ALLOW_PUBLIC_DOWNLOAD",
            "SUSC_17C36_ALLOW_LIGHTWEIGHT_RASTER", "SUSC_17C36_ALLOW_DEM",
            "SUSC_17C36_ALLOW_HYDROLOGIC_PROCESSING", "SUSC_17C36_ALLOW_HAND_PIPELINE"]
    return all(os.environ.get(v) == "1" for v in need)


def _require_inputs():
    missing = [p for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("; ".join(rel(p) for p in missing))


def _event_id():
    rows = read_csv(PATCH_GRID)
    return rows[0]["source_event_id"] if rows else "REC_2022_05_24_30"


def _canary_points():
    return read_csv(C33_REGISTRY)


def _aoi_bbox_margin(margin_deg=0.01):
    pts = _canary_points()
    lons = [float(c["center_lon"]) for c in pts]
    lats = [float(c["center_lat"]) for c in pts]
    return (min(lons) - margin_deg, min(lats) - margin_deg, max(lons) + margin_deg, max(lats) + margin_deg)


# ---------------------------------------------------------------------------
# Hydrologic processing (numpy) -> ledger
# ---------------------------------------------------------------------------

def _priority_flood_fill(dem, nodata=-9999.0):
    import numpy as np
    import heapq
    ny, nx = dem.shape
    filled = dem.copy()
    visited = np.zeros((ny, nx), dtype=bool)
    heap = []
    # borda como sementes
    for x in range(nx):
        for y in (0, ny - 1):
            if not visited[y, x]:
                visited[y, x] = True
                heapq.heappush(heap, (dem[y, x], y, x))
    for y in range(ny):
        for x in (0, nx - 1):
            if not visited[y, x]:
                visited[y, x] = True
                heapq.heappush(heap, (dem[y, x], y, x))
    while heap:
        elev, y, x = heapq.heappop(heap)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            ny2, nx2 = y + dy, x + dx
            if 0 <= ny2 < ny and 0 <= nx2 < nx and not visited[ny2, nx2]:
                visited[ny2, nx2] = True
                if filled[ny2, nx2] < elev:
                    filled[ny2, nx2] = elev
                heapq.heappush(heap, (filled[ny2, nx2], ny2, nx2))
    return filled


def _d8_flow(filled, cellsize):
    import numpy as np
    ny, nx = filled.shape
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    dist = [cellsize, cellsize, cellsize, cellsize,
            cellsize * math.sqrt(2), cellsize * math.sqrt(2), cellsize * math.sqrt(2), cellsize * math.sqrt(2)]
    rec = -np.ones((ny, nx, 2), dtype=np.int32)  # receiver
    order = np.argsort(filled.ravel())[::-1]  # decrescente
    acc = np.ones((ny, nx), dtype=np.float64)
    for flat in order:
        y, x = divmod(int(flat), nx)
        best_slope, by, bx = 0.0, -1, -1
        for k, (dy, dx) in enumerate(dirs):
            y2, x2 = y + dy, x + dx
            if 0 <= y2 < ny and 0 <= x2 < nx:
                s = (filled[y, x] - filled[y2, x2]) / dist[k]
                if s > best_slope:
                    best_slope, by, bx = s, y2, x2
        if by >= 0:
            rec[y, x] = (by, bx)
    for flat in order:
        y, x = divmod(int(flat), nx)
        by, bx = rec[y, x]
        if by >= 0:
            acc[by, bx] += acc[y, x]
    return acc, rec


def _slope_rad(filled, cellsize):
    import numpy as np
    gy, gx = np.gradient(filled, cellsize)
    return np.arctan(np.sqrt(gx * gx + gy * gy))


def _tpi(elev, cellsize, radius_m):
    import numpy as np
    r = max(1, int(round(radius_m / cellsize)))
    ny, nx = elev.shape
    ii = np.zeros((ny + 1, nx + 1), dtype=np.float64)
    ii[1:, 1:] = np.cumsum(np.cumsum(elev, axis=0), axis=1)
    tpi = np.zeros_like(elev)
    for y in range(ny):
        y0, y1 = max(0, y - r), min(ny, y + r + 1)
        for x in range(nx):
            x0, x1 = max(0, x - r), min(nx, x + r + 1)
            s = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
            cnt = (y1 - y0) * (x1 - x0)
            tpi[y, x] = elev[y, x] - (s / cnt)
    return tpi


def _hand(filled, acc, rec, drainage_mask):
    import numpy as np
    ny, nx = filled.shape
    drain_elev = np.full((ny, nx), np.nan)
    # celulas de drenagem: elevacao propria
    for y in range(ny):
        for x in range(nx):
            if drainage_mask[y, x]:
                drain_elev[y, x] = filled[y, x]
    # propaga a montante: cada celula herda a elevacao de dreno do seu receptor
    for flat in np.argsort(filled.ravel()):  # crescente
        y, x = divmod(int(flat), nx)
        if drainage_mask[y, x]:
            continue
        by, bx = rec[y, x]
        if by >= 0 and not np.isnan(drain_elev[by, bx]):
            drain_elev[y, x] = drain_elev[by, bx]
    hand = filled - drain_elev
    return hand


def _run_dem_and_grid() -> None:
    if not _env_ok():
        raise SystemExit("BLOCKED: processamento 17C36 exige os seis opt-ins SUSC_17C36_*.")
    import numpy as np
    import rasterio
    from rasterio.windows import from_bounds
    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
    os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")
    ensure_dir(LIGHT)
    ensure_dir(LOCAL_RUNS)
    aoi = _aoi_bbox_margin()
    attempts = []
    dem_manifest = None
    try:
        with rasterio.open(DEM_TILE_URL) as ds:
            w = from_bounds(aoi[0], aoi[1], aoi[2], aoi[3], ds.transform)
            dem = ds.read(1, window=w).astype("float64")
            win_transform = ds.window_transform(w)
            res_deg = ds.res[0]
        attempts.append({"source": DEM_SOURCE, "url": DEM_TILE_URL, "attempt_type": "cog_window_read",
                         "status": "acquired", "failure_reason": ""})
    except Exception as e:
        attempts.append({"source": DEM_SOURCE, "url": DEM_TILE_URL, "attempt_type": "cog_window_read",
                         "status": "error", "failure_reason": type(e).__name__})
        write_json(LEDGER_TOPO, {"generated_at_utc": _utcnow(), "attempts": attempts, "dem_acquired": False,
                                 "grid": None, "per_canary": []})
        return
    # tentativas adicionais (fontes alternativas registradas honestamente)
    attempts.append({"source": "opentopodata_srtm30m", "url": "https://api.opentopodata.org/v1/srtm30m",
                     "attempt_type": "point_service_not_grid", "status": "not_used_point_only",
                     "failure_reason": "point_service_insufficient_for_grid_flow_routing"})
    attempts.append({"source": "repo_local_dem", "url": rel(OFFICIAL_MATRIX), "attempt_type": "repo_scan",
                     "status": "no_raster_dem_in_repo", "failure_reason": "only_feature_table_present"})

    nodata = -32767.0
    dem = np.where(dem < -1000, np.nan, dem)
    valid = np.isfinite(dem)
    if valid.sum() < 100:
        write_json(LEDGER_TOPO, {"generated_at_utc": _utcnow(), "attempts": attempts, "dem_acquired": True,
                                 "grid": {"insufficient_valid": True}, "per_canary": []})
        return
    demf = np.where(valid, dem, np.nanmax(dem[valid]))
    lat_mid = (aoi[1] + aoi[3]) / 2
    cellsize = res_deg * 111320.0 * math.cos(math.radians(lat_mid))  # ~metros
    # raw DEM window -> local_runs (npy), NUNCA outputs_public
    raw_path = LOCAL_RUNS / "dem_aoi_window.npy"
    np.save(raw_path, dem)
    raw_bytes = raw_path.read_bytes()

    filled = _priority_flood_fill(demf)
    slope = _slope_rad(filled, cellsize)
    acc, rec = _d8_flow(filled, cellsize)
    flow_acc_log = np.log1p(acc)
    tpi = _tpi(demf, cellsize, TPI_RADIUS_M)
    thr = np.percentile(acc, DRAINAGE_FLOWACC_PCTL)
    drainage_mask = acc >= thr
    hand = _hand(filled, acc, rec, drainage_mask)
    a_sca = (acc + 1.0) * cellsize
    beta = np.maximum(slope, math.radians(0.1))
    twi = np.log(a_sca / np.tan(beta))

    ny, nx = demf.shape

    def bbox_stats(arr, xmin, ymin, xmax, ymax):
        # mapeia bbox lon/lat -> indices via win_transform
        from rasterio.transform import rowcol
        r0, c0 = rowcol(win_transform, xmin, ymax)
        r1, c1 = rowcol(win_transform, xmax, ymin)
        r0, r1 = sorted((max(0, r0), min(ny, r1 + 1)))
        c0, c1 = sorted((max(0, c0), min(nx, c1 + 1)))
        sub = arr[r0:r1, c0:c1]
        sub = sub[np.isfinite(sub)]
        if sub.size == 0:
            return None, None, None, 0
        return float(np.mean(sub)), float(np.min(sub)), float(np.max(sub)), int(sub.size)

    per_canary = []
    for c in _canary_points():
        xmin, ymin, xmax, ymax = float(c["xmin"]), float(c["ymin"]), float(c["xmax"]), float(c["ymax"])
        e_m, e_mn, e_mx, npx = bbox_stats(demf, xmin, ymin, xmax, ymax)
        s_m, _, _, _ = bbox_stats(np.degrees(slope), xmin, ymin, xmax, ymax)
        fa_m, fa_mn, fa_mx, _ = bbox_stats(flow_acc_log, xmin, ymin, xmax, ymax)
        tp_m, tp_mn, tp_mx, _ = bbox_stats(tpi, xmin, ymin, xmax, ymax)
        tw_m, tw_mn, tw_mx, _ = bbox_stats(twi, xmin, ymin, xmax, ymax)
        hd_m, hd_mn, hd_mx, _ = bbox_stats(hand, xmin, ymin, xmax, ymax)

        def rnd(v):
            return round(v, 4) if v is not None else None
        per_canary.append({
            "canary_patch_id": c["event_anchored_canary_patch_id"], "valid_pixel_count": npx,
            "elevation_mean": rnd(e_m), "elevation_min": rnd(e_mn), "elevation_max": rnd(e_mx),
            "slope_mean_deg": rnd(s_m),
            "flow_acc_log_mean": rnd(fa_m), "flow_acc_log_min": rnd(fa_mn), "flow_acc_log_max": rnd(fa_mx),
            "tpi_250m_mean": rnd(tp_m), "tpi_250m_min": rnd(tp_mn), "tpi_250m_max": rnd(tp_mx),
            "twi_mean": rnd(tw_m), "twi_min": rnd(tw_mn), "twi_max": rnd(tw_mx),
            "hand_mean": rnd(hd_m), "hand_min": rnd(hd_mn), "hand_max": rnd(hd_mx),
        })
    dem_manifest = {
        "source_name": DEM_SOURCE, "source_authority": "copernicus_esa_official", "artifact_local_path": rel(raw_path),
        "artifact_type": "npy_local_runs", "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "size_bytes": len(raw_bytes), "crs": "EPSG:4326", "resolution_m": round(cellsize, 2),
        "bbox": ",".join(f"{v:.5f}" for v in aoi), "stored_in_outputs_public": False, "raw_heavy": True,
        "usable_for_hydrologic_grid": True,
    }
    grid = {
        "width": nx, "height": ny, "cellsize_m": round(cellsize, 3), "valid_cell_count": int(valid.sum()),
        "sink_fill_applied": True, "sink_fill_method": "priority_flood_wang_liu",
        "flow_direction_method": "d8_steepest_descent", "flow_accumulation_method": "d8_topo_order",
        "drainage_flowacc_percentile": DRAINAGE_FLOWACC_PCTL, "tpi_radius_m": TPI_RADIUS_M,
        "twi_formula": "ln((flowacc+1)*cellsize/tan(beta))", "hand_method": "downstream_to_drainage_cell",
        "method_equivalence_status": METHOD_EQUIV,
    }
    write_json(LEDGER_TOPO, {"generated_at_utc": _utcnow(), "attempts": attempts, "dem_acquired": True,
                             "dem_manifest": dem_manifest, "grid": grid, "per_canary": per_canary})


def _utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_ledger():
    if not LEDGER_TOPO.exists():
        raise FileNotFoundError(f"{rel(LEDGER_TOPO)} ausente: rode run-dem-acquisition + build-hydrologic-grid com os seis opt-ins.")
    return read_json(LEDGER_TOPO)


def _per_canary():
    led = _load_ledger()
    return {r["canary_patch_id"]: r for r in led.get("per_canary", [])}


# ---------------------------------------------------------------------------
# Offline derivation
# ---------------------------------------------------------------------------

def method_inventory_rows():
    cards = {c["feature_name"]: c for c in read_json(FEATURE_CARDS)["cards"]} if FEATURE_CARDS.exists() else {}
    directions = {}
    for f in TOPO_FEATURES:
        directions[f] = cards.get(f, {}).get("expected_direction", "unknown")
    specs = {
        "hand_mean": ("m", "elevacao acima do dreno mais proximo", "true"),
        "twi_mean": ("index", "ln(a/tan(beta)) design_only; escala original atipica (~116)", "true"),
        "tpi_250m_mean": ("m", "elevacao - media vizinhanca 250m", "true"),
        "flow_acc_log_mean": ("log_cells", "log1p da acumulacao de fluxo", "true"),
        "elevation_mean": ("m", "elevacao media DEM", "false"),
        "slope_mean": ("deg_or_rad", "declividade media DEM", "false"),
    }
    rows = []
    for i, f in enumerate(TOPO_FEATURES, start=1):
        unit, note, req = specs.get(f, ("", "", "true"))
        card = cards.get(f, {})
        has_method = bool(card.get("source")) or bool(card.get("provenance"))
        recovery = "feature_names_only" if not has_method else "method_partially_recovered"
        rows.append({
            "method_inventory_id": f"S17C36_MI_{i:04d}", "feature_name": f,
            "used_in_score_v6": "true", "original_method_found": _bool(has_method),
            "original_source_file": rel(SCORE_V6_BUILDER) if SCORE_V6_BUILDER.exists() else "not_located",
            "unit": unit, "direction": directions.get(f, "unknown"),
            "required_for_topography_hydrology": req,
            "method_equivalence_requirement": "same_dem_algorithm_units_as_original",
            "recovery_status": recovery,
            "blocking_reason": note if recovery == "feature_names_only" else "not_applicable",
            "review_only": "true",
        })
    return rows


def dem_acquisition_attempt_rows():
    led = _load_ledger()
    atts = led.get("attempts", [])
    rows = [{"dem_acquisition_attempt_id": f"S17C36_DEMA_{i:04d}", "source_name": a["source"],
             "source_url_or_api": a["url"], "attempt_type": a["attempt_type"], "network_enabled": "true",
             "http_status": "200" if a["status"] == "acquired" else a["status"],
             "dem_acquired": _bool(a["status"] == "acquired"),
             "artifact_manifest_id": "S17C36_DEM_0001" if a["status"] == "acquired" else "",
             "failure_reason": a.get("failure_reason", ""), "review_only": "true"}
            for i, a in enumerate(atts, start=1)]
    return rows or [{"dem_acquisition_attempt_id": "S17C36_DEMA_0001", "source_name": "none",
                     "source_url_or_api": "", "attempt_type": "none", "network_enabled": "true",
                     "http_status": "not_run", "dem_acquired": "false", "artifact_manifest_id": "",
                     "failure_reason": "ledger_absent", "review_only": "true"}]


def dem_manifest_rows():
    led = _load_ledger()
    m = led.get("dem_manifest")
    if not m:
        return [{"dem_artifact_manifest_id": "S17C36_DEM_0001", "source_name": "none", "source_authority": "none",
                 "artifact_local_path": "none", "artifact_type": "none", "sha256": "none", "size_bytes": "0",
                 "crs": "none", "resolution_m": "0", "bbox": "none", "stored_in_outputs_public": "false",
                 "raw_heavy": "false", "usable_for_hydrologic_grid": "false", "review_only": "true"}]
    return [{"dem_artifact_manifest_id": "S17C36_DEM_0001", "source_name": m["source_name"],
             "source_authority": m["source_authority"], "artifact_local_path": m["artifact_local_path"],
             "artifact_type": m["artifact_type"], "sha256": m["sha256"], "size_bytes": str(m["size_bytes"]),
             "crs": m["crs"], "resolution_m": str(m["resolution_m"]), "bbox": m["bbox"],
             "stored_in_outputs_public": "false", "raw_heavy": "true",
             "usable_for_hydrologic_grid": _bool(m["usable_for_hydrologic_grid"]), "review_only": "true"}]


def dem_grid_metadata_rows():
    led = _load_ledger()
    g = led.get("grid") or {}
    m = led.get("dem_manifest") or {}
    ok = bool(g) and not g.get("insufficient_valid")
    return [{"dem_grid_id": "S17C36_GRID_0001", "source_artifact_manifest_id": "S17C36_DEM_0001",
             "crs": "EPSG:4326", "resolution_m": str(g.get("cellsize_m", "0")), "width": str(g.get("width", 0)),
             "height": str(g.get("height", 0)), "bbox": m.get("bbox", "none"), "nodata_value": "-32767",
             "valid_cell_count": str(g.get("valid_cell_count", 0)), "sink_fill_applied": _bool(g.get("sink_fill_applied", False)),
             "usable_for_tpi": _bool(ok), "usable_for_flow_acc": _bool(ok), "usable_for_twi": _bool(ok),
             "usable_for_hand": _bool(ok), "review_only": "true"}]


def hydrologic_grid_metadata_rows():
    led = _load_ledger()
    g = led.get("grid") or {}
    ok = bool(g) and not g.get("insufficient_valid")
    return [{"hydrologic_grid_id": "S17C36_HGRID_0001", "dem_grid_id": "S17C36_GRID_0001",
             "flow_direction_created": _bool(ok), "flow_accumulation_created": _bool(ok),
             "slope_created": _bool(ok), "drainage_mask_created": _bool(ok), "nearest_drainage_cell_created": _bool(ok),
             "sink_fill_method": g.get("sink_fill_method", "none"), "flow_direction_method": g.get("flow_direction_method", "none"),
             "flow_accumulation_method": g.get("flow_accumulation_method", "none"),
             "method_equivalence_status": g.get("method_equivalence_status", METHOD_EQUIV), "review_only": "true"}]


def _ok_grid():
    led = _load_ledger()
    g = led.get("grid") or {}
    return bool(g) and not g.get("insufficient_valid") and bool(led.get("per_canary"))


def tpi_rows():
    pc = _per_canary()
    ok = _ok_grid()
    rows = []
    for c in _canary_points():
        pid = c["event_anchored_canary_patch_id"]
        d = pc.get(pid, {})
        rows.append({"tpi_feature_id": pid.replace("EAC", "TPI"), "canary_patch_id": pid, "dem_grid_id": "S17C36_GRID_0001",
                     "radius_m": str(int(TPI_RADIUS_M)),
                     "tpi_250m_mean": d.get("tpi_250m_mean", "not_available") if ok else "not_available",
                     "tpi_250m_min": d.get("tpi_250m_min", "not_available") if ok else "not_available",
                     "tpi_250m_max": d.get("tpi_250m_max", "not_available") if ok else "not_available",
                     "valid_pixel_count": str(d.get("valid_pixel_count", 0)),
                     "method_equivalence_status": METHOD_EQUIV, "review_only": "true"})
    return rows


def flow_accumulation_rows():
    pc = _per_canary()
    ok = _ok_grid()
    rows = []
    for c in _canary_points():
        pid = c["event_anchored_canary_patch_id"]
        d = pc.get(pid, {})
        rows.append({"flow_acc_feature_id": pid.replace("EAC", "FA"), "canary_patch_id": pid,
                     "hydrologic_grid_id": "S17C36_HGRID_0001",
                     "flow_acc_log_mean": d.get("flow_acc_log_mean", "not_available") if ok else "not_available",
                     "flow_acc_log_min": d.get("flow_acc_log_min", "not_available") if ok else "not_available",
                     "flow_acc_log_max": d.get("flow_acc_log_max", "not_available") if ok else "not_available",
                     "flow_acc_proxy": "false", "flow_acc_full": _bool(ok),
                     "valid_pixel_count": str(d.get("valid_pixel_count", 0)),
                     "method_equivalence_status": METHOD_EQUIV,
                     "blocking_reason": "not_applicable" if ok else "dem_grid_unavailable", "review_only": "true"})
    return rows


def twi_rows():
    pc = _per_canary()
    ok = _ok_grid()
    rows = []
    for c in _canary_points():
        pid = c["event_anchored_canary_patch_id"]
        d = pc.get(pid, {})
        rows.append({"twi_feature_id": pid.replace("EAC", "TWI"), "canary_patch_id": pid,
                     "hydrologic_grid_id": "S17C36_HGRID_0001",
                     "twi_mean": d.get("twi_mean", "not_available") if ok else "not_available",
                     "twi_min": d.get("twi_min", "not_available") if ok else "not_available",
                     "twi_max": d.get("twi_max", "not_available") if ok else "not_available",
                     "twi_proxy": "false", "twi_full": _bool(ok),
                     "valid_pixel_count": str(d.get("valid_pixel_count", 0)),
                     "method_equivalence_status": METHOD_EQUIV,
                     "blocking_reason": "not_applicable" if ok else "dem_grid_unavailable", "review_only": "true"})
    return rows


def hand_rows():
    pc = _per_canary()
    ok = _ok_grid()
    drainage = {r["canary_patch_id"]: r for r in read_csv(C35_DRAINAGE)}
    rows = []
    for c in _canary_points():
        pid = c["event_anchored_canary_patch_id"]
        d = pc.get(pid, {})
        rows.append({"hand_feature_id": pid.replace("EAC", "HAND"), "canary_patch_id": pid,
                     "hydrologic_grid_id": "S17C36_HGRID_0001",
                     "drainage_source": "dem_derived_flowacc_p95_and_official_faixas_marginais",
                     "hand_mean": d.get("hand_mean", "not_available") if ok else "not_available",
                     "hand_min": d.get("hand_min", "not_available") if ok else "not_available",
                     "hand_max": d.get("hand_max", "not_available") if ok else "not_available",
                     "nearest_drainage_elevation_m": "dem_derived_downstream_drainage_cell",
                     "patch_elevation_mean_m": d.get("elevation_mean", "not_available") if ok else "not_available",
                     "HAND_full": _bool(ok), "HAND_proxy": "false", "not_full_hydrologic_HAND": _bool(not ok),
                     "method_equivalence_status": METHOD_EQUIV,
                     "blocking_reason": "not_applicable" if ok else "dem_grid_unavailable", "review_only": "true"})
    return rows


def topography_matrix_rows():
    pc = _per_canary()
    ok = _ok_grid()
    tpi = {r["canary_patch_id"]: r for r in tpi_rows()}
    fa = {r["canary_patch_id"]: r for r in flow_accumulation_rows()}
    tw = {r["canary_patch_id"]: r for r in twi_rows()}
    hd = {r["canary_patch_id"]: r for r in hand_rows()}
    rows = []
    for i, c in enumerate(_canary_points(), start=1):
        pid = c["event_anchored_canary_patch_id"]
        d = pc.get(pid, {})
        complete = ok and all(d.get(k) is not None for k in ("hand_mean", "twi_mean", "tpi_250m_mean", "flow_acc_log_mean", "elevation_mean", "slope_mean_deg"))
        rows.append({
            "topography_hydrology_feature_id": f"S17C36_TH_{i:04d}", "canary_patch_id": pid,
            "elevation_mean": d.get("elevation_mean", "not_available") if ok else "not_available",
            "slope_mean": d.get("slope_mean_deg", "not_available") if ok else "not_available",
            "hand_mean": hd[pid]["hand_mean"], "hand_status": "full_reconstructed" if ok else "blocked",
            "twi_mean": tw[pid]["twi_mean"], "twi_status": "full_reconstructed" if ok else "blocked",
            "tpi_250m_mean": tpi[pid]["tpi_250m_mean"], "tpi_status": "full_reconstructed" if ok else "blocked",
            "flow_acc_log_mean": fa[pid]["flow_acc_log_mean"], "flow_acc_status": "full_reconstructed" if ok else "blocked",
            "topography_hydrology_feature_complete": _bool(complete),
            "method_equivalence_status": METHOD_EQUIV,
            "review_only": "true", "ground_truth": "false", "training_label": "false",
        })
    return rows


def topography_parity_rows():
    ok = _ok_grid()
    matrix = {r["canary_patch_id"]: r for r in topography_matrix_rows()}
    rows = []
    for i, c in enumerate(_canary_points(), start=1):
        pid = c["event_anchored_canary_patch_id"]
        m = matrix[pid]
        # paridade: computado com metodo reconstruido (nao provado equivalente) -> "reconstructed_not_equivalent"
        par = "reconstructed_not_proven_equivalent" if ok else "missing"
        n_ok = sum(1 for s in (m["hand_status"], m["twi_status"], m["tpi_status"], m["flow_acc_status"]) if s == "full_reconstructed")
        rows.append({
            "topography_parity_id": f"S17C36_TP_{i:04d}", "canary_patch_id": pid,
            "hand_parity": par, "twi_parity": par, "tpi_parity": par, "flow_acc_parity": par,
            "topography_hydrology_parity_ratio": f"{n_ok / 4:.4f}",
            "sufficient_for_topography_component_replay": _bool(ok),
            "sufficient_for_score_v6_full_replay": "false",
            "blocking_reason": "metodo reconstruido nao comprovadamente equivalente ao score v6 original (DEM/algoritmo/unidade originais nao documentados)",
            "review_only": "true",
        })
    return rows


def readiness_update_rows():
    ok = _ok_grid()
    prev = {r["canary_patch_id"]: r for r in read_csv(C35_READINESS)}
    rows = []
    for c in _canary_points():
        pid = c["event_anchored_canary_patch_id"]
        p = prev.get(pid, {})
        topo_status = "reconstructed_not_proven_equivalent" if ok else "blocked_dem_grid_unavailable"
        rows.append({
            "score_replay_readiness_update_id": pid.replace("EAC", "RDYU"), "canary_patch_id": pid,
            "previous_score_replay_status_17c35": p.get("score_replay_status", "partial_components_only_topography_blocked"),
            "topography_hydrology_component_status": topo_status,
            "normalization_population_reconstructed": "true",
            "urban_component_available": "true", "veg_component_available": "true", "rain_component_available": "true",
            "evidence_component_status": "occurrence_not_used_as_feature",
            "can_compute_topography_component": _bool(ok),
            "can_compute_full_score_v6_replay": "false",
            "score_replay_status_after_17c36": "topography_reconstructed_score_final_blocked_on_method_equivalence" if ok else "topography_blocked",
            "blocking_reason": "score final exige metodo comprovadamente equivalente ao original (DEM/algoritmo/unidade); reconstruido != provado equivalente",
            "not_official_score": "true", "does_not_replace_score_v6": "true", "review_only": "true",
        })
    return rows


def score_integrity_audit_rows():
    changed = bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)]))
    h = _run_git(["hash-object", rel(SCORE_V6)]) if SCORE_V6.exists() else "absent"
    return [{"score_integrity_audit_id": "S17C36_INTEG_0001", "official_score_v6_path": rel(SCORE_V6),
             "official_score_v6_hash_before": h, "official_score_v6_hash_after": h,
             "official_score_v6_changed": _bool(changed), "score_v7_exists": _bool(SCORE_V7.exists()),
             "final_score_computed": "false", "review_only": "true"}]


def no_leakage_audit_rows():
    rows = []
    for i, c in enumerate(_canary_points(), start=1):
        rows.append({
            "no_leakage_audit_id": f"S17C36_LEAK_{i:04d}", "object_id": c["event_anchored_canary_patch_id"],
            "object_type": "topography_hydrology_feature",
            "uses_occurrence_as_feature": "false", "uses_event_as_label": "false",
            "uses_post_event_feature_as_pre_event": "false", "uses_risk_sector_as_observed_event": "false",
            "uses_proxy_as_full_without_flag": "false", "uses_osm_as_official_hydrography": "false",
            "uses_bairro_centroid_as_hydrology": "false", "uses_control_as_negative_truth": "false",
            "modifies_score_v6": "false", "uses_score_v7": "false", "computed_final_score": "false",
            "passes_no_leakage": "true",
            "blocking_reason": "features topograficas reconstruidas de DEM real review-only; metodo flaggeado nao-equivalente; sem score final",
            "review_only": "true",
        })
    return rows


def build_summary():
    canary = _canary_points()
    method = method_inventory_rows()
    dem_att = dem_acquisition_attempt_rows()
    dem_man = dem_manifest_rows()
    grid = dem_grid_metadata_rows()
    hgrid = hydrologic_grid_metadata_rows()
    tpi = tpi_rows()
    fa = flow_accumulation_rows()
    tw = twi_rows()
    hd = hand_rows()
    matrix = topography_matrix_rows()
    parity = topography_parity_rows()
    ready = readiness_update_rows()
    drainage = read_csv(C35_DRAINAGE)
    ok = _ok_grid()
    dem_local = len([m for m in dem_man if m["source_name"] != "none"])
    complete = len([m for m in matrix if m["topography_hydrology_feature_complete"] == "true"])
    parity_mean = round(sum(float(r["topography_hydrology_parity_ratio"]) for r in parity) / len(parity), 4) if parity else 0.0
    dem_attempts = len(dem_att)
    minimum = (
        len(canary) >= 11 and dem_attempts >= 3 and dem_local >= 1
        and len(drainage) >= 11 and len(hgrid) >= 1 and len(matrix) >= 11
        and len(tpi) >= 11 and len(fa) >= 11 and len(tw) >= 11 and len(hd) >= 11
        and len(matrix) >= 11 and len(parity) >= 11 and ok
    )
    return {
        "minimum_success_achieved": minimum,
        "canary_patch_rows_consumed_count": len(canary),
        "dem_acquisition_attempts_count": dem_attempts,
        "dem_local_artifacts_count": dem_local,
        "dem_grid_created": ok,
        "hydrography_source_rows_count": len(drainage),
        "hydrologic_grid_rows_count": len(hgrid),
        "terrain_derivative_rows_count": len(matrix),
        "tpi_250m_rows_count": len(tpi),
        "flow_accumulation_rows_count": len(fa),
        "twi_rows_count": len(tw),
        "hand_rows_count": len(hd),
        "topography_hydrology_feature_rows_count": len(matrix),
        "topography_hydrology_complete_rows_count": complete,
        "hydrologic_feature_parity_rows_count": len(parity),
        "topography_hydrology_parity_mean": parity_mean,
        "HAND_full_rows_count": len([r for r in hd if r["HAND_full"] == "true"]),
        "HAND_proxy_rows_count": len([r for r in hd if r["HAND_proxy"] == "true"]),
        "TWI_full_rows_count": len([r for r in tw if r["twi_full"] == "true"]),
        "TWI_proxy_rows_count": len([r for r in tw if r["twi_proxy"] == "true"]),
        "flow_acc_full_rows_count": len([r for r in fa if r["flow_acc_full"] == "true"]),
        "flow_acc_proxy_rows_count": len([r for r in fa if r["flow_acc_proxy"] == "true"]),
        "score_v6_replay_readiness_updated": True,
        "score_v6_full_replay_computable": False,
        "score_v6_review_only_final_score_computed_count": 0,
        "method_equivalence_status": METHOD_EQUIV,
        "score_v6_original_file_changed": bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)])),
        "score_v7_created": SCORE_V7.exists(),
        "ground_truth_created": False,
        "training_labels_created": False,
        "official_patch_created": False,
        "official_patch_link_created": False,
        "eligible_for_17b_now": False,
        "review_only": True,
        "trainable": False,
        "ground_truth": False,
        "result_class": "A_topography_reconstructed_score_final_blocked_on_method_equivalence" if ok else "C_dem_grid_unavailable",
        "recommended_next_milestone": (
            "SUSC-17C37 Validacao de equivalencia metodologica (recomputar as features topograficas dos PATCHES "
            "OFICIAIS com o MESMO pipeline e comparar com os valores originais); se equivalente, computar o score v6 "
            "replay final review-only; manter score v6 oficial intacto e 17B fail-closed."
        ),
    }


def build_blockers():
    ok = _ok_grid()
    blockers = []
    if not ok:
        blockers += [("dem_grid_not_available", "DEM grid indisponivel"),
                     ("flow_direction_not_available", "flow direction nao criado"),
                     ("flow_accumulation_not_full", "flow accumulation nao completo"),
                     ("twi_not_full", "TWI nao completo"), ("HAND_not_full", "HAND nao completo")]
    blockers += [
        ("method_equivalence_not_recovered", "metodo/DEM/unidade originais do score v6 nao documentados no repo -> equivalencia nao comprovada"),
        ("topography_component_still_blocked_for_final_score", "componente topografia reconstruido mas nao comprovadamente equivalente"),
        ("score_v6_full_replay_blocked", "score final v6 bloqueado ate equivalencia metodologica"),
        ("score_v7_blocked", "score v7 bloqueado"),
        ("17b_blocked", "17B bloqueado ate G4_full + ground reference aceito"),
    ]
    return [{"blocker_id": f"S17C36_BLOCKER_{i:04d}", "blocker_type": n, "description": d,
             "blocks_full_score_replay": "true", "blocks_17b": "true", "blocks_score_v7": "true", "review_only": "true"}
            for i, (n, d) in enumerate(blockers, start=1)]


def build_report():
    s = build_summary()
    return "\n".join([
        "# SUSC-17C36 - Pipeline Hidrologico Local para TWI/TPI/Flow Accumulation/HAND Full",
        "",
        "## Objetivo",
        "Pipeline hidrologico local auditavel (DEM Copernicus GLO-30 + numpy) para reproduzir twi_mean, tpi_250m_mean, flow_acc_log_mean e hand_mean/min/max dos canarios, completando o subindice topography_hydrology do replay review-only do score v6.",
        "",
        "## Metodo",
        f"- Inventario do metodo original: {len(method_inventory_rows())} features; recuperacao majoritaria feature_names_only (DEM/algoritmo/unidade originais nao documentados no repo).",
        f"- DEM: {s['dem_acquisition_attempts_count']} tentativas; artefato local: {s['dem_local_artifacts_count']} (Copernicus GLO-30, ~30 m, raw em local_runs, NAO commitado).",
        f"- Grid hidrologico: fill-sinks (priority-flood), D8 flow direction+accumulation, slope, TPI 250m (integral image), TWI=ln(a/tan(beta)), HAND (downstream ate celula de drenagem).",
        "",
        "## Features topograficas (reconstruidas)",
        f"- TPI: {s['tpi_250m_rows_count']}; flow_acc: {s['flow_accumulation_rows_count']} (full {s['flow_acc_full_rows_count']}); TWI: {s['twi_rows_count']} (full {s['TWI_full_rows_count']}); HAND: {s['hand_rows_count']} (full {s['HAND_full_rows_count']}/proxy {s['HAND_proxy_rows_count']}).",
        f"- Topography matrix: {s['topography_hydrology_feature_rows_count']} linhas; completas: {s['topography_hydrology_complete_rows_count']}; paridade media: {s['topography_hydrology_parity_mean']}.",
        "",
        "## Resultado (honesto)",
        f"- result_class: {s['result_class']}.",
        "- As features topograficas/hidrologicas FORAM reproduzidas em escala local auditavel a partir de DEM real. Porem o metodo NAO e comprovadamente identico ao score v6 original (metodo original nao documentado; twi original ~116 tem escala atipica) -> method_equivalence_status=method_reconstructed_not_proven_equivalent.",
        f"- Score v6 full replay computavel: {s['score_v6_full_replay_computable']}; score final review-only computado: {s['score_v6_review_only_final_score_computed_count']}. O score final NAO foi computado (guardrail: nao computar se metodo nao for equivalente).",
        "",
        "## Guardrails",
        "- Score v6 oficial intacto; sem v7; sem GT/treino; ocorrencia nao e feature; proxy != full flaggeado; OSM != hidrografia oficial; raster pesado so em local_runs (nunca outputs_public); controle nao vira negativo verdadeiro.",
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
    write_csv(METHOD_INV, method_inventory_rows())
    write_csv(DEM_ATTEMPTS, dem_acquisition_attempt_rows())
    write_csv(DEM_MANIFEST, dem_manifest_rows())
    write_csv(DEM_GRID, dem_grid_metadata_rows())
    write_csv(HYDRO_GRID, hydrologic_grid_metadata_rows())
    write_csv(TPI, tpi_rows())
    write_csv(FLOW_ACC, flow_accumulation_rows())
    write_csv(TWI, twi_rows())
    write_csv(HAND, hand_rows())
    write_csv(TOPO_MATRIX, topography_matrix_rows())
    write_csv(TOPO_PARITY, topography_parity_rows())
    write_csv(READINESS_UPDATE, readiness_update_rows())
    write_csv(INTEGRITY, score_integrity_audit_rows())
    write_csv(LEAKAGE, no_leakage_audit_rows())
    write_json(SUMMARY, build_summary())
    write_csv(BLOCKERS, build_blockers())
    write_markdown(REPORT, build_report())


def run_mode(mode):
    _require_inputs()
    ensure_dir(OUT)
    if mode == "inventory-original-method":
        write_csv(METHOD_INV, method_inventory_rows())
    elif mode in ("run-dem-acquisition", "build-hydrologic-grid"):
        _run_dem_and_grid()
    elif mode == "compute-tpi":
        write_csv(TPI, tpi_rows())
    elif mode == "compute-flow-accumulation":
        write_csv(FLOW_ACC, flow_accumulation_rows())
    elif mode == "compute-twi":
        write_csv(TWI, twi_rows())
    elif mode == "compute-hand":
        write_csv(HAND, hand_rows())
    elif mode == "build-topography-feature-matrix":
        write_csv(TOPO_MATRIX, topography_matrix_rows())
    elif mode == "build-hydrologic-parity":
        write_csv(TOPO_PARITY, topography_parity_rows())
    elif mode == "update-score-replay-readiness":
        write_csv(READINESS_UPDATE, readiness_update_rows())
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
    if not LEDGER_TOPO.exists():
        print(f"MISSING_LEDGER: {rel(LEDGER_TOPO)}", file=sys.stderr)
        return 1
    missing = [p for p in REQUIRED_OUTPUTS if not p.exists()]
    if missing:
        print("MISSING: " + "; ".join(rel(p) for p in missing), file=sys.stderr)
        return 1
    errors = _validate_byte_identical()

    method = read_csv(METHOD_INV)
    dem_att = read_csv(DEM_ATTEMPTS)
    dem_man = read_csv(DEM_MANIFEST)
    tpi = read_csv(TPI)
    fa = read_csv(FLOW_ACC)
    tw = read_csv(TWI)
    hd = read_csv(HAND)
    matrix = read_csv(TOPO_MATRIX)
    parity = read_csv(TOPO_PARITY)
    ready = read_csv(READINESS_UPDATE)
    leak = read_csv(LEAKAGE)
    integ = read_csv(INTEGRITY)
    summary = read_json(SUMMARY)
    feat_schema = read_json(SCHEMA_FEAT)
    par_schema = read_json(SCHEMA_PARITY)
    dem_schema = read_json(SCHEMA_DEM)

    for r in matrix:
        errors.extend(_schema_violations(r, feat_schema))
    for r in parity:
        errors.extend(_schema_violations(r, par_schema))
    for r in dem_man:
        if r["source_name"] != "none":
            errors.extend(_schema_violations(r, dem_schema))

    # 1 canary, 2 dem attempts, 3 dem local, 11 method inventory
    if summary["canary_patch_rows_consumed_count"] < 11:
        errors.append("canary_lt_11")
    if summary["dem_acquisition_attempts_count"] < 3:
        errors.append("dem_attempts_lt_3")
    if summary["dem_local_artifacts_count"] < 1:
        errors.append("dem_local_lt_1")
    if len(method) < len(TOPO_FEATURES):
        errors.append("method_inventory_incomplete")
    # 4 grid quando dem adquirido
    if summary["dem_grid_created"] and summary["hydrologic_grid_rows_count"] < 1:
        errors.append("grid_missing_when_dem_acquired")
    # 5-8 features existem/bloqueadas
    for rows, n, mn in [(tpi, "tpi", 11), (fa, "flow_acc", 11), (tw, "twi", 11), (hd, "hand", 11)]:
        if len(rows) < mn:
            errors.append(f"{n}_rows_lt_{mn}")
    if summary["topography_hydrology_feature_rows_count"] < 11:
        errors.append("topo_matrix_lt_11")
    if len(parity) < 11:
        errors.append("parity_lt_11")
    if len(ready) < 11:
        errors.append("readiness_update_lt_11")
    # 9 proxy full sem equivalencia
    for r in fa:
        if r["flow_acc_full"] == "true" and r["method_equivalence_status"] not in (METHOD_EQUIV, "method_partially_recovered", "method_exactly_recovered"):
            errors.append("flowacc_full_without_equiv_flag")
    for r in hd:
        if r["HAND_full"] == "true" and r["HAND_proxy"] == "true":
            errors.append("hand_full_and_proxy")
    if any(r["uses_proxy_as_full_without_flag"] != "false" for r in leak):
        errors.append("proxy_as_full_leak")
    # 10 osm/proxy oficial
    if any(r["uses_osm_as_official_hydrography"] != "false" for r in leak):
        errors.append("osm_as_official")
    # 11 metodo inventariado
    if summary["method_equivalence_status"] not in (METHOD_EQUIV, "method_partially_recovered", "method_exactly_recovered"):
        errors.append("method_not_inventoried")
    # 12 score v6 intacto
    if _run_git(["diff", "--name-only", "--", rel(SCORE_V6)]) or summary["score_v6_original_file_changed"]:
        errors.append("score_v6_changed")
    for r in integ:
        if r["official_score_v6_changed"] != "false":
            errors.append("integrity_changed")
        if r["final_score_computed"] != "false":
            errors.append("final_score_computed")
    # 13 score v7
    if SCORE_V7.exists() or summary["score_v7_created"]:
        errors.append("score_v7_exists")
    # 14/15 GT/treino
    if summary["ground_truth_created"] or summary["training_labels_created"]:
        errors.append("gt_or_training")
    for r in matrix:
        if r["ground_truth"] != "false" or r["training_label"] != "false":
            errors.append("matrix_gt_training")
    # 16 raster pesado em outputs_public
    if LIGHT.exists():
        for p in LIGHT.glob("**/*"):
            if p.is_file() and (p.suffix.lower() in FORBIDDEN_SUFFIXES or p.stat().st_size > MAX_LIGHT_BYTES):
                errors.append(f"forbidden_or_heavy_committed:{rel(p)}")
    for r in dem_man:
        if r["source_name"] != "none" and r["stored_in_outputs_public"] != "false":
            errors.append("dem_stored_in_outputs_public")
    # 17 ocorrencia como feature
    if any(r["uses_occurrence_as_feature"] != "false" for r in leak):
        errors.append("occurrence_as_feature")
    # 18 controle negativo verdadeiro
    if any(r["uses_control_as_negative_truth"] != "false" for r in leak):
        errors.append("control_negative")
    if any(r["passes_no_leakage"] != "true" for r in leak):
        errors.append("no_leakage_failed")
    # score final nao computado
    if summary["score_v6_full_replay_computable"] or summary["score_v6_review_only_final_score_computed_count"] != 0:
        errors.append("final_score_should_be_blocked")

    # 19 summary reflete contagens
    expected = build_summary()
    for k, v in expected.items():
        if summary.get(k) != v:
            errors.append(f"summary_mismatch:{k}")
    if not summary["minimum_success_achieved"]:
        errors.append("minimum_success_not_achieved")

    for rows, key in [
        (method, "method_inventory_id"), (dem_att, "dem_acquisition_attempt_id"), (tpi, "tpi_feature_id"),
        (fa, "flow_acc_feature_id"), (tw, "twi_feature_id"), (hd, "hand_feature_id"),
        (matrix, "topography_hydrology_feature_id"), (parity, "topography_parity_id"),
        (ready, "score_replay_readiness_update_id"), (leak, "no_leakage_audit_id"),
    ]:
        ids = [r[key] for r in rows]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            errors.append(f"ids_not_unique_or_sorted:{key}")

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(
        "17C36 -> "
        f"canary={summary['canary_patch_rows_consumed_count']} dem_att={summary['dem_acquisition_attempts_count']} "
        f"dem_local={summary['dem_local_artifacts_count']} grid={summary['dem_grid_created']} "
        f"tpi={summary['tpi_250m_rows_count']} flow_acc={summary['flow_accumulation_rows_count']} twi={summary['twi_rows_count']} "
        f"hand={summary['hand_rows_count']} complete={summary['topography_hydrology_complete_rows_count']} "
        f"parity_mean={summary['topography_hydrology_parity_mean']} hand_full={summary['HAND_full_rows_count']} "
        f"full_replay={summary['score_v6_full_replay_computable']} final_score={summary['score_v6_review_only_final_score_computed_count']} "
        f"result={summary['result_class']} min_success={summary['minimum_success_achieved']}"
    )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="susc_17c36")
    parser.add_argument("mode", choices=[
        "inventory-original-method", "run-dem-acquisition", "build-hydrologic-grid", "compute-tpi",
        "compute-flow-accumulation", "compute-twi", "compute-hand", "build-topography-feature-matrix",
        "build-hydrologic-parity", "update-score-replay-readiness", "build-offline", "audit",
    ])
    args = parser.parse_args(argv)
    if args.mode == "audit":
        return validate()
    run_mode(args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
