"""MV2-14 Tarefa 2 — Extrai candidatos de lineage de fontes leves.

Extrai scene_id, product_id, collection GEE, tile, datas, cloud_cover, export task,
asset/patch, região e CRS — sempre como CANDIDATO (nunca verdade). CSVs estruturados
(scene searches, registries) são parseados linha-a-linha; demais arquivos por regex.

Saída: outputs_public/mv2_gee_lineage_recovery/mv2_14_lineage_candidates.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

from mv2_14_gee_lineage_common import (
    COLLECTION_RE,
    OUTPUT_DIR,
    PROJECT_ROOT,
    S2_SCENE_RE,
    TILE_RE,
    classify_datetime,
    clean,
    ensure_output_dir,
    is_excluded,
    iter_scannable_files,
    rel_to_root,
    resolve_scan_roots,
    write_csv,
)

COLUMNS = [
    "candidate_id", "source_id", "source_path_or_ref", "candidate_type", "raw_value",
    "normalized_value", "inferred_sensor", "inferred_tile", "inferred_datetime",
    "inferred_cloud_cover", "inferred_asset_id", "inferred_patch_id", "inferred_region",
    "extraction_method", "extraction_confidence", "evidence_snippet", "notes",
]

MAX_BYTES = 262_144
HARD_SKIP = 5_000_000

SCENE_COLS = ["scene_id_sanitized", "scene_id", "scene_id_like", "product_id_sanitized"]
DATE_COLS = ["scene_date", "acquisition_datetime", "acquisition_date", "datetime"]
CLOUD_COLS = ["cloud_cover_metadata", "cloud_cover", "cloud_metadata_global", "cloud_cover_percent"]
TILE_COLS = ["mgrs_tile", "tile_id", "tile"]
COLL_COLS = ["gee_collection", "collection"]
REGION_COLS = ["region", "regiao", "municipality", "cidade"]
ASSET_COLS = ["asset_id"]
PATCH_COLS = ["reference_patch_id", "patch_id"]


def _first(row: dict, cols: list[str]) -> str:
    for c in cols:
        if c in row and clean(row[c]):
            return clean(row[c])
    return ""


def _sensor_from(scene: str, collection: str) -> str:
    if S2_SCENE_RE.search(scene) or "S2" in collection.upper():
        return "S2_MSI"
    if "S1" in collection.upper():
        return "S1_SAR"
    return ""


def _extract_csv(path: Path, src_id: str, ref: str, out: list, start_idx: int) -> int:
    idx = start_idx
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as h:
            reader = csv.DictReader(h)
            cols = reader.fieldnames or []
            if not any(c in cols for c in SCENE_COLS):
                return idx  # CSV sem coluna de cena explícita
            for row in reader:
                scene = _first(row, SCENE_COLS)
                if not scene or not S2_SCENE_RE.search(scene):
                    continue
                date_raw = _first(row, DATE_COLS)
                dt_norm, dt_status = classify_datetime(date_raw[:10]) if date_raw else ("", "ABSENT")
                tile = _first(row, TILE_COLS) or (TILE_RE.search(scene).group(0) if TILE_RE.search(scene) else "")
                coll = _first(row, COLL_COLS)
                idx += 1
                out.append({
                    "candidate_id": f"MV2_14_CAND_{idx:05d}",
                    "source_id": src_id,
                    "source_path_or_ref": ref,
                    "candidate_type": "scene_id",
                    "raw_value": scene,
                    "normalized_value": scene,
                    "inferred_sensor": _sensor_from(scene, coll),
                    "inferred_tile": tile,
                    "inferred_datetime": dt_norm if dt_status in {"VALID_SINGLE"} else date_raw[:10],
                    "inferred_cloud_cover": _first(row, CLOUD_COLS),
                    "inferred_asset_id": _first(row, ASSET_COLS),
                    "inferred_patch_id": _first(row, PATCH_COLS),
                    "inferred_region": _first(row, REGION_COLS),
                    "extraction_method": "csv_structured_row",
                    "extraction_confidence": "HIGH" if dt_status == "VALID_SINGLE" else "MEDIUM",
                    "evidence_snippet": f"collection={coll};datetime_status={dt_status}"[:120],
                    "notes": "candidato; namespace pode ser anchor protocolo-C, nao corpus",
                })
    except OSError:
        return idx
    return idx


def _extract_text(src_id: str, ref: str, txt: str, out: list, start_idx: int) -> int:
    idx = start_idx
    coll = COLLECTION_RE.search(txt)
    for m in set(s.group(0) for s in S2_SCENE_RE.finditer(txt)):
        tile = TILE_RE.search(m)
        idx += 1
        out.append({
            "candidate_id": f"MV2_14_CAND_{idx:05d}",
            "source_id": src_id,
            "source_path_or_ref": ref,
            "candidate_type": "scene_id",
            "raw_value": m,
            "normalized_value": m,
            "inferred_sensor": _sensor_from(m, coll.group(0) if coll else ""),
            "inferred_tile": tile.group(0) if tile else "",
            "inferred_datetime": "",
            "inferred_cloud_cover": "",
            "inferred_asset_id": "",
            "inferred_patch_id": "",
            "inferred_region": "",
            "extraction_method": "text_regex",
            "extraction_confidence": "LOW",
            "evidence_snippet": (f"collection={coll.group(0)}" if coll else "scene_id_in_text")[:120],
            "notes": "candidato textual; nunca vira verdade sem fonte estruturada",
        })
    return idx


def extract() -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    idx = 0
    src_idx = 0
    for root, _ in resolve_scan_roots():
        for path in iter_scannable_files(root):
            if not path.is_file() or is_excluded(path):
                continue
            if path.suffix.lower() not in {".csv", ".py", ".md", ".txt", ".json", ".yaml", ".yml"}:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > HARD_SKIP:
                continue
            try:
                with path.open("r", encoding="utf-8", errors="ignore") as h:
                    head = h.read(MAX_BYTES)
            except OSError:
                continue
            if not S2_SCENE_RE.search(head):
                continue
            src_idx += 1
            src_id = f"MV2_14_SRC_{src_idx:04d}"
            ref = rel_to_root(path)
            if path.suffix.lower() == ".csv" and size <= MAX_BYTES:
                new_idx = _extract_csv(path, src_id, ref, out, idx)
                if new_idx == idx:
                    new_idx = _extract_text(src_id, ref, head, out, idx)
                idx = new_idx
            else:
                idx = _extract_text(src_id, ref, head, out, idx)
    return out


def main() -> None:
    ensure_output_dir()
    rows = extract()
    out = OUTPUT_DIR / "mv2_14_lineage_candidates.csv"
    write_csv(out, COLUMNS, rows)
    scenes = len(set(str(r["normalized_value"]) for r in rows))
    tiles = sorted(set(str(r["inferred_tile"]) for r in rows if r["inferred_tile"]))
    print(f"[mv2_14_candidates] candidatos: {len(rows)} | cenas distintas: {scenes} | tiles: {tiles}")
    print(f"[mv2_14_candidates] saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
