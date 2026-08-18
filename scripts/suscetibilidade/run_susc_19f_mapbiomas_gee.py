"""Executor GEE do SUSC-19F: extrai features territoriais MapBiomas por patch.

Roda o pacote MapBiomas/GEE do 19B contra os 300 patches reais (bbox do manifesto),
calcula o histograma de classes por patch via Earth Engine e recupera um CSV leve
localmente pela API (getInfo, sem Drive, sem raster). Nunca imprime credenciais.

Se o Earth Engine nao estiver autenticado ou a biblioteca faltar, registra o bloqueio
tecnico exato em execution_log.txt e mapbiomas_execution_metadata.json e mantem tudo
reexecutavel. Nao inventa valores: sem extracao real, nenhum CSV de resultado e escrito.

Saidas (todas em local_runs/, git-ignored):
- mapbiomas_patch_landcover_2022.csv
- mapbiomas_execution_metadata.json
- mapbiomas_tasks.csv
- execution_log.txt
"""

from __future__ import annotations

import json
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import susc_19f_mapbiomas_common as s19f  # noqa: E402
from susc_io import ensure_dir, write_csv  # noqa: E402

CHUNK = 20  # patches por chamada getInfo


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Logger:
    def __init__(self, path: Path):
        self.path = path
        ensure_dir(path.parent)
        self.lines: list[str] = []

    def log(self, msg: str) -> None:
        line = f"{_now()} {msg}"
        self.lines.append(line)
        print(line)

    def flush(self) -> None:
        self.path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def audit_environment(log: Logger) -> dict:
    env = {
        "python_version": platform.python_version(),
        "ee_library": "ausente",
        "earthengine_cli": "ausente",
        "ee_initialize": "nao_testado",
        "internet": "nao_testado",
        "export_permission": "nao_testado",
        "writable_folder": "false",
    }
    status = "ambiente_indeterminado"

    # pasta gravavel
    try:
        ensure_dir(s19f.LOCAL_RUN)
        probe = s19f.LOCAL_RUN / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        env["writable_folder"] = "true"
    except OSError:
        env["writable_folder"] = "false"

    env["earthengine_cli"] = "presente" if shutil.which("earthengine") else "ausente"

    try:
        import ee  # noqa: F401
        env["ee_library"] = getattr(ee, "__version__", "presente")
    except ImportError:
        env["ee_library"] = "ausente"
        status = "biblioteca_ausente"
        log.log("ERRO: biblioteca earthengine-api ausente")
        return {"environment": env, "env_status": status}

    # Initialize sem imprimir credenciais
    try:
        import ee
        ee.Initialize()
        env["ee_initialize"] = "ok"
        env["internet"] = "true"
    except Exception as exc:  # noqa: BLE001
        env["ee_initialize"] = "falha"
        msg = type(exc).__name__
        low = str(exc).lower()
        if "authenticat" in low or "credential" in low or "not been used" in low:
            status = "gee_nao_autenticado"
        elif "permission" in low or "forbidden" in low or "403" in low:
            status = "permissao_insuficiente"
        else:
            status = "ambiente_indeterminado"
        log.log(f"ERRO: ee.Initialize falhou ({msg}); status={status}")
        return {"environment": env, "env_status": status}

    # teste de compute leve (getInfo) - permissao de leitura
    try:
        import ee
        val = ee.Number(1).add(1).getInfo()
        env["export_permission"] = "true" if val == 2 else "indeterminado"
        status = "gee_autenticado_pronto"
        log.log("ambiente GEE autenticado e pronto")
    except Exception as exc:  # noqa: BLE001
        env["export_permission"] = "false"
        low = str(exc).lower()
        status = "permissao_insuficiente" if ("permission" in low or "403" in low) else "ambiente_indeterminado"
        log.log(f"ERRO: getInfo de teste falhou ({type(exc).__name__}); status={status}")
    return {"environment": env, "env_status": status}


def extract_histograms(manifest: list[dict], log: Logger) -> tuple[list[dict], list[dict]]:
    """Retorna (linhas_resultado, tasks). Uma linha por patch com proporcoes reais."""
    import ee

    coll = s19f.MAPBIOMAS_COLLECTION
    year = s19f.MAPBIOMAS_YEAR
    scale = s19f.MAPBIOMAS_SCALE
    band = f"classification_{year}"
    img = ee.Image(coll).select(band)

    results: list[dict] = []
    tasks: list[dict] = []
    n = len(manifest)
    for start in range(0, n, CHUNK):
        chunk = manifest[start:start + CHUNK]
        feats = []
        for m in chunk:
            rect = ee.Geometry.Rectangle([
                float(m["xmin"]), float(m["ymin"]), float(m["xmax"]), float(m["ymax"]),
            ])
            hist = img.reduceRegion(
                reducer=ee.Reducer.frequencyHistogram(),
                geometry=rect, scale=scale, maxPixels=int(1e9),
            ).get(band)
            feats.append(ee.Feature(None, {"patch_id": m["patch_id"], "hist": hist}))
        fc = ee.FeatureCollection(feats)
        t0 = time.time()
        chunk_ok = False
        info = None
        for attempt in range(3):
            try:
                info = fc.getInfo()
                chunk_ok = True
                break
            except Exception as exc:  # noqa: BLE001
                log.log(f"chunk {start}-{start+len(chunk)} tentativa {attempt+1} falhou: {type(exc).__name__}")
                time.sleep(3 * (attempt + 1))
        dt = round(time.time() - t0, 1)
        tasks.append({
            "task_id": f"getinfo_chunk_{start:04d}",
            "task_type": "synchronous_getInfo",
            "status": "completed" if chunk_ok else "failed",
            "patches_processados": str(len(chunk)) if chunk_ok else "0",
            "observacao": f"chunk {start}-{start+len(chunk)} em {dt}s",
        })
        if not chunk_ok or info is None:
            log.log(f"chunk {start}-{start+len(chunk)} FALHOU apos retries")
            continue
        for feat in info.get("features", []):
            props = feat.get("properties", {})
            pid = props.get("patch_id")
            hist = props.get("hist") or {}
            derived = s19f.proportions_from_hist(hist)
            if not derived:
                log.log(f"patch {pid} sem pixels; ignorado")
                continue
            reg = next((m["regiao"] for m in chunk if m["patch_id"] == pid), "")
            results.append({
                "patch_id": pid,
                "regiao": reg,
                "mapbiomas_year": year,
                "mapbiomas_class_majority": derived["mapbiomas_class_majority"],
                "mapbiomas_class_majority_label": derived["mapbiomas_class_majority_label"],
                "mapbiomas_class_distribution": json.dumps(derived["mapbiomas_class_distribution"], separators=(",", ":")),
                "water_prop": derived["water_prop"],
                "exposed_soil_prop": derived["exposed_soil_prop"],
                "impervious_proxy": derived["impervious_proxy"],
                "vegetation_prop_mapbiomas": derived["vegetation_prop_mapbiomas"],
                "pixel_count": derived["pixel_count"],
                "landcover_source": s19f.LANDCOVER_SOURCE,
                "review_only": "true",
            })
        log.log(f"chunk {start}-{start+len(chunk)} ok: {len(info.get('features', []))} patches em {dt}s")
    return results, tasks


def main() -> int:
    ensure_dir(s19f.LOCAL_RUN)
    log = Logger(s19f.LOCAL_LOG)
    log.log(f"SUSC-19F executor iniciado; manifesto={s19f.rel(s19f.MANIFEST_19B)}")

    audit = audit_environment(log)
    env_status = audit["env_status"]
    meta = {
        "executed_at": _now(),
        "mapbiomas_collection": s19f.MAPBIOMAS_COLLECTION,
        "mapbiomas_year": s19f.MAPBIOMAS_YEAR,
        "scale": s19f.MAPBIOMAS_SCALE,
        "env_status": env_status,
        "environment": audit["environment"],
        "patches_expected": s19f.EXPECTED_PATCHES,
        "patches_extracted": 0,
        "csv_path": s19f.rel(s19f.LOCAL_CSV),
        "note": "recuperacao por API (getInfo); sem Drive; sem raster; sem credenciais impressas",
    }

    if env_status != "gee_autenticado_pronto":
        meta["blocked_reason"] = env_status
        s19f.LOCAL_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        write_csv(s19f.LOCAL_TASKS, [{
            "task_id": s19f.NA, "task_type": "synchronous_getInfo", "status": "blocked",
            "patches_processados": "0", "observacao": f"bloqueado: {env_status}; reexecutar apos resolver",
        }])
        log.log(f"BLOQUEADO: {env_status}. Nenhum CSV escrito (sem inventar valores).")
        log.flush()
        return 0

    manifest = s19f.load_manifest()
    log.log(f"extraindo MapBiomas para {len(manifest)} patches (chunks de {CHUNK})")
    results, tasks = extract_histograms(manifest, log)

    meta["patches_extracted"] = len(results)
    write_csv(s19f.LOCAL_TASKS, tasks or [{
        "task_id": s19f.NA, "task_type": "synchronous_getInfo", "status": "no_tasks",
        "patches_processados": "0", "observacao": "nenhuma task",
    }])

    if results:
        fields = ["patch_id", "regiao", "mapbiomas_year", "mapbiomas_class_majority",
                  "mapbiomas_class_majority_label", "mapbiomas_class_distribution", "water_prop",
                  "exposed_soil_prop", "impervious_proxy", "vegetation_prop_mapbiomas",
                  "pixel_count", "landcover_source", "review_only"]
        write_csv(s19f.LOCAL_CSV, results, fieldnames=fields)
        log.log(f"CSV leve escrito: {s19f.rel(s19f.LOCAL_CSV)} ({len(results)} patches)")
    else:
        log.log("nenhum resultado extraido; CSV nao escrito (sem inventar valores)")

    s19f.LOCAL_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log.log(f"metadata escrito: {s19f.rel(s19f.LOCAL_META)}; extraidos={len(results)}")
    log.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
