"""
SUSC-07B Refined Patch x Event Overlay (review-only)

Uses the REAL extracted coordinates to test spatial adherence against patch
bboxes. Polygon/bbox geometries -> bbox_overlap with patch bboxes; point sets ->
point-in-bbox (near_patch_buffer_candidate). Where no usable event geometry
exists, the SUSC-07A regional/documentary classification is preserved. Nothing
becomes ground truth.

Reads:
  - datasets/suscetibilidade/susc_features_by_patch_v1.csv
  - datasets/suscetibilidade/susc_07b_event_evidence_geometry_enriched_v1.csv
  - datasets/suscetibilidade/susc_07b_real_event_coordinates_v1.csv
  - outputs_public/suscetibilidade/SUSC_07A_patch_event_overlay_candidates.csv

Writes:
  - outputs_public/suscetibilidade/SUSC_07B_refined_patch_event_overlay.csv
  - outputs_public/suscetibilidade/SUSC_07B_refined_patch_event_overlay_summary.csv
  - outputs_public/suscetibilidade/SUSC_07B_refined_overlay_limitations.json
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())

MATRIX = ROOT / "datasets" / "suscetibilidade" / "susc_features_by_patch_v1.csv"
ENRICHED = ROOT / "datasets" / "suscetibilidade" / "susc_07b_event_evidence_geometry_enriched_v1.csv"
COORDS = ROOT / "datasets" / "suscetibilidade" / "susc_07b_real_event_coordinates_v1.csv"
OVERLAY_07A = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_07A_patch_event_overlay_candidates.csv"

OUT_OVERLAY = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_07B_refined_patch_event_overlay.csv"
OUT_SUMMARY = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_07B_refined_patch_event_overlay_summary.csv"
OUT_LIMITS = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_07B_refined_overlay_limitations.json"

PROXY_COL = "score_evento_enchente_potencial_v5_core"
BBOX_COLS = ["xmin", "ymin", "xmax", "ymax"]
REGION_MAP = {"recife": "recife", "petropolis": "petropolis", "curitiba": "curitiba"}

OVERLAY_FIELDS = [
    "patch_id", "region", "susceptibility_proxy", "evidence_id", "coordinate_id",
    "evidence_status", "spatial_relation", "temporal_relation", "overlay_confidence",
    "geometry_source", "supporting_points", "can_be_used_as_ground_truth",
    "allowed_for_training", "review_only", "interpretation", "limitation",
]


def load_csv(p):
    with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        return list(csv.DictReader(f))


def _num(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def parse_bbox(s):
    try:
        xs = [float(v) for v in s.split(";")]
        return xs if len(xs) == 4 else None
    except (ValueError, AttributeError):
        return None


def bbox_overlap(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def iter_geojson_points(path):
    try:
        obj = json.loads((ROOT / path).read_text(encoding="utf-8", errors="ignore"))
    except (OSError, ValueError):
        return []
    pts = []

    def walk(c):
        if isinstance(c, list):
            if c and isinstance(c[0], (int, float)) and len(c) >= 2:
                pts.append((float(c[0]), float(c[1])))
            else:
                for x in c:
                    walk(x)

    feats = []
    if isinstance(obj, dict) and obj.get("type") == "FeatureCollection":
        feats = [ft.get("geometry") for ft in obj.get("features", [])]
    elif isinstance(obj, dict) and obj.get("type") == "Feature":
        feats = [obj.get("geometry")]
    elif isinstance(obj, dict):
        feats = [obj]
    for g in feats:
        if g:
            walk(g.get("coordinates", []))
    return pts


def iter_csv_points(path):
    p = ROOT / path
    pts = []
    try:
        with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            cols = {c.lower(): c for c in (reader.fieldnames or [])}
            latc = next((cols[k] for k in cols if k in ("lat", "latitude")), None)
            lonc = next((cols[k] for k in cols if k in ("lon", "long", "longitude")), None)
            if not latc or not lonc:
                return pts
            for r in reader:
                lon, lat = _num(r.get(lonc)), _num(r.get(latc))
                if lon is not None and lat is not None:
                    pts.append((lon, lat))
    except OSError:
        pass
    return pts


def main() -> int:
    print("=" * 60)
    print("SUSC-07B Refined Patch x Event Overlay (review-only)")
    print("=" * 60)

    for p in (MATRIX, ENRICHED, COORDS):
        if not p.exists():
            print(f"STOP: required input not found: {p}")
            return 1

    patches = load_csv(MATRIX)
    coords = load_csv(COORDS)

    has_bbox = all(c in patches[0].keys() for c in BBOX_COLS)
    if not has_bbox:
        print("STOP: patch matrix has no bbox columns.")
        return 1

    # patch bboxes by region
    region_patches = defaultdict(list)
    for r in patches:
        bb = [_num(r[c]) for c in BBOX_COLS]
        if all(v is not None for v in bb):
            region_patches[r.get("regiao", "")].append((r["patch_id"], bb, _num(r.get(PROXY_COL))))

    rows = []
    # process each extracted geometry against its region's patches
    for c in coords:
        region = REGION_MAP.get(c["region"], c["region"])
        patches_here = region_patches.get(region, [])
        gtype = c["geometry_type"]
        ev_id = c["evidence_id"]
        coord_id = c["coordinate_id"]
        src = c["source_name"]
        geom_bbox = parse_bbox(c["bbox"])

        if gtype == "polygon" and geom_bbox:
            for pid, pbb, proxy in patches_here:
                if bbox_overlap(geom_bbox, pbb):
                    rows.append(_row(pid, region, proxy, ev_id, coord_id,
                                     "bbox_overlap", "low_spatial", src, "",
                                     "Sobreposicao de bbox entre poligono de evento rastreavel e bbox do patch.",
                                     "bbox e aproximacao; poligono e candidato digitalizado review-only; NAO confirma evento no patch."))
        elif gtype in {"point", "point_set"}:
            # point-in-bbox using the real points from the source file
            src_path = c["source_path_or_url"]
            if c["geojson_ref"]:
                pts = iter_geojson_points(src_path)
            elif src_path.lower().endswith((".csv", ".tsv")):
                pts = iter_csv_points(src_path)
            else:
                pts = []
            if not pts and c["lat"] and c["lon"]:
                pts = [(float(c["lon"]), float(c["lat"]))]
            # count points per patch
            counts = {}
            for (lon, lat) in pts:
                for pid, pbb, proxy in patches_here:
                    if pbb[0] <= lon <= pbb[2] and pbb[1] <= lat <= pbb[3]:
                        counts[pid] = counts.get(pid, 0) + 1
            proxy_by_pid = {pid: proxy for pid, _, proxy in patches_here}
            for pid, n in sorted(counts.items()):
                rows.append(_row(pid, region, proxy_by_pid.get(pid), ev_id, coord_id,
                                 "near_patch_buffer_candidate", "low_spatial", src, str(n),
                                 f"{n} ponto(s) de fonte rastreavel dentro do bbox do patch.",
                                 "pontos de localidade/risco != footprint de evento; aderencia espacial review-only; NAO e ground truth."))

    # always preserve the SUSC-07A regional/documentary rows (per documentary
    # evidence); the refined spatial rows above are an ADDITIONAL layer, not a
    # replacement, so documentary context is never lost.
    if OVERLAY_07A.exists():
        for a in load_csv(OVERLAY_07A):
            rows.append(_row("REGION_LEVEL_NO_PATCH_RESOLUTION", a["region"],
                             None, a.get("evidence_id", ""),
                             "", a.get("spatial_relation", "same_region_only"),
                             a.get("temporal_relation", "undated"), "", "",
                             "Associacao regional/documental preservada do SUSC-07A (camada documental).",
                             "Sem geometria de evento utilizavel para esta evidencia; nao e overlap de patch."))

    OUT_OVERLAY.parent.mkdir(parents=True, exist_ok=True)
    with OUT_OVERLAY.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OVERLAY_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    # summary
    rel_counts = Counter(r["spatial_relation"] for r in rows)
    by_region_rel = defaultdict(Counter)
    for r in rows:
        by_region_rel[r["region"]][r["spatial_relation"]] += 1
    summary_rows = []
    for region in sorted(by_region_rel):
        rc = by_region_rel[region]
        summary_rows.append({
            "region": region,
            "n_patches": len(region_patches.get(region, [])),
            "exact_patch_overlap": rc.get("exact_patch_overlap", 0),
            "bbox_overlap": rc.get("bbox_overlap", 0),
            "near_patch_buffer_candidate": rc.get("near_patch_buffer_candidate", 0),
            "same_region_period": rc.get("same_region_period", 0),
            "same_region_only": rc.get("same_region_only", 0),
            "documentary_context_only": rc.get("documentary_context_only", 0),
            "insufficient_for_patch_link": rc.get("insufficient_for_patch_link", 0),
            "review_only": "true",
        })
    with OUT_SUMMARY.open("w", encoding="utf-8", newline="") as f:
        fields = list(summary_rows[0].keys()) if summary_rows else ["region"]
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(summary_rows)

    unique_patches = len({r["patch_id"] for r in rows if r["patch_id"] != "REGION_LEVEL_NO_PATCH_RESOLUTION"})
    limitations = {
        "overlay_name": "SUSC-07B refined patch x event overlay",
        "patch_geometry_available": True,
        "event_geometry_available": True,
        "geometry_sources": "traceable local GeoJSON/CSV (Charter758, Defesa Civil, CPRM/official registries)",
        "spatial_relation_counts": dict(rel_counts),
        "n_overlay_rows": len(rows),
        "n_unique_patches_with_spatial_relation": unique_patches,
        "exact_patch_overlap": rel_counts.get("exact_patch_overlap", 0),
        "bbox_overlap": rel_counts.get("bbox_overlap", 0),
        "near_patch_buffer_candidate": rel_counts.get("near_patch_buffer_candidate", 0),
        "can_be_used_as_ground_truth": False,
        "allowed_for_training": False,
        "review_only": True,
        "scientific_status": "spatial_adherence_from_traceable_geometry_not_patch_ground_truth",
        "key_limitations": [
            "Geometria de evento e candidata/digitalizada ou pontos de risco rastreaveis, NAO footprint validado.",
            "Pontos de localidade/risco da Defesa Civil != ocorrencia confirmada por patch.",
            "Sobreposicao de bbox e aproximacao, nao geometria exata patch-evento.",
            "Vinculo evidencia-coordenada e em nivel de regiao (requires_manual_review).",
            "Nenhuma relacao aqui e ground truth ou rotulo de treino.",
        ],
        "prepares": "SUSC-08: validacao humana da aderencia e criterio de referencia sob revisao.",
    }
    OUT_LIMITS.write_text(json.dumps(limitations, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\noverlay rows: {len(rows)} | unique patches with spatial relation: {unique_patches}")
    print("spatial relations:", dict(rel_counts))
    print("\nSpatial adherence from traceable geometry. No patch ground truth. review-only.")
    return 0


def _row(pid, region, proxy, ev_id, coord_id, rel, temporal, src, supp, interp, limit):
    return {
        "patch_id": pid, "region": region,
        "susceptibility_proxy": "" if proxy is None else round(proxy, 6),
        "evidence_id": ev_id, "coordinate_id": coord_id,
        "evidence_status": "geometry_enriched" if coord_id else "regional_documentary",
        "spatial_relation": rel, "temporal_relation": temporal,
        "overlay_confidence": "low", "geometry_source": src, "supporting_points": supp,
        "can_be_used_as_ground_truth": "false", "allowed_for_training": "false",
        "review_only": "true", "interpretation": interp, "limitation": limit,
    }


if __name__ == "__main__":
    sys.exit(main())
