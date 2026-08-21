"""
SUSC-07A Patch x Event Overlay Builder (review-only)

Tests documentary/spatial adherence between susceptibility patches and the event
evidence catalog. Because the catalog has NO usable coordinates, no exact patch
overlap is possible: the overlay produces only regional/documentary associations,
each with an explicit confidence and limitation. NOTHING becomes ground truth.

Reads:
  - datasets/suscetibilidade/susc_features_by_patch_v1.csv
  - datasets/suscetibilidade/susc_07_event_evidence_catalog_v1.csv
  - outputs_public/suscetibilidade/SUSC_06B_event_overlay_readiness_matrix.csv

Writes:
  - outputs_public/suscetibilidade/SUSC_07A_patch_event_overlay_candidates.csv
  - outputs_public/suscetibilidade/SUSC_07A_patch_event_overlay_summary.csv
  - outputs_public/suscetibilidade/SUSC_07A_patch_event_overlay_limitations.json
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())

MATRIX = ROOT / "datasets" / "suscetibilidade" / "susc_features_by_patch_v1.csv"
CATALOG = ROOT / "datasets" / "suscetibilidade" / "susc_07_event_evidence_catalog_v1.csv"
READINESS = (
    ROOT / "outputs_public" / "suscetibilidade" / "SUSC_06B_event_overlay_readiness_matrix.csv"
)

OUT_CAND = (
    ROOT / "outputs_public" / "suscetibilidade" / "SUSC_07A_patch_event_overlay_candidates.csv"
)
OUT_SUMMARY = (
    ROOT / "outputs_public" / "suscetibilidade" / "SUSC_07A_patch_event_overlay_summary.csv"
)
OUT_LIMITS = (
    ROOT / "outputs_public" / "suscetibilidade" / "SUSC_07A_patch_event_overlay_limitations.json"
)

PROXY_COL = "score_evento_enchente_potencial_v5_core"
BBOX_COLS = ["xmin", "ymin", "xmax", "ymax"]

CAND_FIELDS = [
    "patch_id", "region", "susceptibility_proxy", "evidence_id", "evidence_status",
    "spatial_relation", "temporal_relation", "overlay_confidence",
    "can_be_used_as_ground_truth", "allowed_for_training", "review_only",
    "interpretation", "limitation",
]


def load_csv(path):
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        return list(csv.DictReader(f))


def _num(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def classify_spatial(evidence):
    region = evidence["region"]
    status = evidence["evidence_status"]
    if region == "unknown" or status == "insufficient_for_patch_event_link":
        return "insufficient_for_patch_link"
    if status == "documentary_context_only":
        return "documentary_context_only"
    # event has region but no usable coordinate -> region-level association
    if evidence["date_or_period"] != "unknown":
        return "same_region_period"
    return "same_region_only"


def main() -> int:
    print("=" * 60)
    print("SUSC-07A Patch x Event Overlay (review-only)")
    print("=" * 60)

    for p in (MATRIX, CATALOG):
        if not p.exists():
            print(f"STOP: required input not found: {p}")
            return 1

    patches = load_csv(MATRIX)
    catalog = load_csv(CATALOG)

    # patch geometry availability (bbox exists in SUSC-03 matrix)
    matrix_cols = set(patches[0].keys()) if patches else set()
    has_bbox = all(c in matrix_cols for c in BBOX_COLS)

    # region proxy aggregates
    region_patches = defaultdict(list)
    for r in patches:
        region_patches[r.get("regiao", "")].append(r)
    region_proxy_mean = {}
    for reg, rs in region_patches.items():
        vals = [v for v in (_num(r.get(PROXY_COL)) for r in rs) if v is not None]
        region_proxy_mean[reg] = round(sum(vals) / len(vals), 6) if vals else None

    # overlay candidates: one region-level association per evidence (no coordinate
    # exists, so no patch is singled out -> patch_id is a region-level marker).
    cand_rows = []
    for ev in catalog:
        region = ev["region"]
        spatial = classify_spatial(ev)
        temporal = "dated" if ev["date_or_period"] != "unknown" else "undated"
        if spatial == "insufficient_for_patch_link":
            conf = "insufficient"
        elif spatial == "documentary_context_only":
            conf = "documentary_only"
        else:
            conf = "low"
        proxy = region_proxy_mean.get(region)
        cand_rows.append({
            "patch_id": "REGION_LEVEL_NO_PATCH_RESOLUTION",
            "region": region,
            "susceptibility_proxy": f"region_mean={proxy}" if proxy is not None else "NA",
            "evidence_id": ev["evidence_id"],
            "evidence_status": ev["evidence_status"],
            "spatial_relation": spatial,
            "temporal_relation": temporal,
            "overlay_confidence": conf,
            "can_be_used_as_ground_truth": "false",
            "allowed_for_training": "false",
            "review_only": "true",
            "interpretation": (
                "Associacao em nivel de REGIAO entre evidencia documental e patches "
                "da mesma regiao; NAO e overlap espacial de patch."
            ),
            "limitation": (
                "Sem coordenada/bbox do evento (catalogo can_link_to_patch=false); "
                "nao identifica patch especifico; nao e ocorrencia confirmada por patch."
            ),
        })

    OUT_CAND.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CAND.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CAND_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(cand_rows)

    # per-region summary
    summary_rows = []
    ev_by_region = Counter(ev["region"] for ev in catalog)
    rel_by_region = defaultdict(Counter)
    for c in cand_rows:
        rel_by_region[c["region"]][c["spatial_relation"]] += 1
    for region in sorted(set(list(region_patches.keys()) + list(ev_by_region.keys()))):
        if region in ("", None):
            continue
        summary_rows.append({
            "region": region,
            "n_patches": len(region_patches.get(region, [])),
            "n_evidence": ev_by_region.get(region, 0),
            "region_proxy_mean": region_proxy_mean.get(region, "NA"),
            "same_region_period": rel_by_region[region].get("same_region_period", 0),
            "same_region_only": rel_by_region[region].get("same_region_only", 0),
            "documentary_context_only": rel_by_region[region].get("documentary_context_only", 0),
            "insufficient_for_patch_link": rel_by_region[region].get("insufficient_for_patch_link", 0),
            "exact_patch_overlap": 0,
            "bbox_overlap": 0,
            "review_only": "true",
        })
    with OUT_SUMMARY.open("w", encoding="utf-8", newline="") as f:
        fields = list(summary_rows[0].keys()) if summary_rows else ["region"]
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(summary_rows)

    rel_counts = Counter(c["spatial_relation"] for c in cand_rows)
    limitations = {
        "overlay_name": "SUSC-07A patch x event overlay",
        "patch_geometry_available": has_bbox,
        "event_geometry_available": False,
        "exact_patch_overlap_possible": False,
        "reason_no_patch_overlap": (
            "O catalogo de evidencia nao possui coordenada/bbox utilizavel "
            "(can_link_to_patch=false em todos); apenas associacao regional/documental."
        ),
        "spatial_relation_counts": dict(rel_counts),
        "n_overlay_candidates": len(cand_rows),
        "highest_adherence_relation": "same_region_period",
        "can_be_used_as_ground_truth": False,
        "allowed_for_training": False,
        "review_only": True,
        "scientific_status": "regional_documentary_association_not_patch_ground_truth",
        "key_limitations": [
            "Nenhum overlap espacial de patch e possivel sem coordenada do evento.",
            "Associacao e em nivel de regiao; nao identifica patch especifico.",
            "Evidencia documental != ocorrencia confirmada por patch != suscetibilidade != ground truth supervisionado.",
            "Periodo/data do evento existe so para parte das evidencias.",
        ],
        "prepares": "SUSC-07B (aquisicao de geometria/coordenada de evento) e SUSC-08.",
    }
    OUT_LIMITS.write_text(json.dumps(limitations, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\noverlay candidates: {len(cand_rows)}")
    print("spatial relations:", dict(rel_counts))
    print(f"patch bbox available: {has_bbox} | event geometry available: False")
    print("\nRegional/documentary association only. No patch overlap. No ground truth.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
