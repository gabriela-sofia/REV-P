"""
SUSC-09A Human Review Workspace Builder (review-only)

Creates an auditable workspace so a person can review the SUSC-08 spatial/
documentary cases: an index, a prefilled form, GeoJSON layers (patch bbox, event
geometry, review cases), per-case dossiers and simple SVG maps. It does NOT
execute or replace human review and does NOT create ground truth.

Reads SUSC-08 cases, SUSC-07B coordinates/enriched, the matrix, the refined
overlay, the SUSC-08 form and the feature cards.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_io import read_csv, write_csv, write_json, write_markdown, ensure_dir, rel  # noqa: E402
from susc_geometry import parse_bbox  # noqa: E402

CASES = ROOT / "datasets" / "suscetibilidade" / "susc_08_spatiotemporal_review_cases_v1.csv"
COORDS = ROOT / "datasets" / "suscetibilidade" / "susc_07b_real_event_coordinates_v1.csv"
MATRIX = ROOT / "datasets" / "suscetibilidade" / "susc_features_by_patch_v1.csv"

WS = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_09A_human_review_workspace"
REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_09A_human_review_workspace_report.md"

BBOX_COLS = ["xmin", "ymin", "xmax", "ymax"]
MANDATORY = ("O SUSC-09A não executa nem substitui a revisão humana. Ele cria um "
             "workspace auditável para que uma pessoa revise os casos espaciais/"
             "documentais, preservando o status review-only e impedindo que aderência "
             "espacial seja confundida com ground truth.")

PRE_REVIEW_MAP = {
    ("moderate_candidate", "tcc_example_with_caution"): "candidate_for_tcc_with_caution",
    ("weak_contextual", "context_only"): "context_only",
    ("insufficient", "acquire_geometry_first"): "blocked_pending_geometry",
}
FORBIDDEN_PRE_REVIEW = {"approved", "strong_candidate", "ground_truth", "training_ready"}


def machine_pre_review(case):
    if case.get("known_conflicts") == "geometry_source_disagreement":
        return "blocked_source_conflict"
    return PRE_REVIEW_MAP.get(
        (case.get("case_strength_pre_review"), case.get("recommended_use")),
        "needs_human_review",
    )


def patch_bbox(patch_row):
    try:
        return [float(patch_row[c]) for c in BBOX_COLS]
    except (KeyError, ValueError, TypeError):
        return None


def bbox_polygon(bbox):
    xmin, ymin, xmax, ymax = bbox
    return [[[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]]]


def svg_for_case(case, pbbox, ebbox):
    """Simple SVG showing patch bbox (blue) and event bbox (orange) in a shared frame."""
    boxes = [b for b in (pbbox, ebbox) if b]
    if not boxes:
        return (
            '<svg viewBox="0 0 400 200" xmlns="http://www.w3.org/2000/svg">'
            '<rect width="400" height="200" fill="#f6f6f6"/>'
            f'<text x="200" y="100" text-anchor="middle" font-size="13" fill="#444">'
            f'{case["case_id"]}: sem geometria (review-only)</text></svg>'
        )
    lons = [b[0] for b in boxes] + [b[2] for b in boxes]
    lats = [b[1] for b in boxes] + [b[3] for b in boxes]
    lo, hi = min(lons), max(lons)
    la, ha = min(lats), max(lats)
    pad = max((hi - lo), (ha - la), 1e-4) * 0.15
    lo, hi, la, ha = lo - pad, hi + pad, la - pad, ha + pad
    W, H = 480, 320

    def sx(x):
        return (x - lo) / (hi - lo) * (W - 40) + 20

    def sy(y):
        return H - ((y - la) / (ha - la) * (H - 40) + 20)  # invert lat

    def rect(b, color, label):
        x, y = sx(b[0]), sy(b[3])
        w = sx(b[2]) - sx(b[0])
        h = sy(b[1]) - sy(b[3])
        return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                f'fill="{color}" fill-opacity="0.25" stroke="{color}" stroke-width="2"/>'
                f'<text x="{x:.1f}" y="{y-4:.1f}" font-size="11" fill="{color}">{label}</text>')

    parts = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">',
             f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    if pbbox:
        parts.append(rect(pbbox, "#1f6feb", f'patch {case["patch_id"]}'))
    if ebbox:
        parts.append(rect(ebbox, "#e8730a", "event geom (review-only)"))
    parts.append(f'<text x="{W/2:.0f}" y="{H-8}" text-anchor="middle" font-size="11" '
                 f'fill="#666">{case["case_id"]} — {case["spatial_relation"]} — review-only, NOT ground truth</text>')
    parts.append("</svg>")
    return "".join(parts)


def main() -> int:
    print("=" * 60)
    print("SUSC-09A Human Review Workspace Builder (review-only)")
    print("=" * 60)

    for p in (CASES, COORDS, MATRIX):
        if not p.exists():
            print(f"STOP: required input not found: {p}")
            return 1

    cases = read_csv(CASES)
    coords = {c["source_name"]: c for c in read_csv(COORDS)}
    patches = {r["patch_id"]: r for r in read_csv(MATRIX)}

    ensure_dir(WS)
    ensure_dir(WS / "forms")
    ensure_dir(WS / "geojson")
    ensure_dir(WS / "case_dossiers")
    ensure_dir(WS / "maps_svg")

    index_rows = []
    case_features = []
    patch_features = []
    event_features = []
    seen_patches = set()
    n_svg = 0
    n_dossier = 0

    for case in cases:
        cid = case["case_id"]
        pre = machine_pre_review(case)
        assert pre not in FORBIDDEN_PRE_REVIEW
        patch_row = patches.get(case["patch_id"])
        pbbox = patch_bbox(patch_row) if patch_row else None
        coord = coords.get(case["source_name"])
        ebbox = parse_bbox(coord["bbox"]) if coord and coord.get("bbox") else None

        # geojson layers
        if pbbox and case["patch_id"] not in seen_patches:
            seen_patches.add(case["patch_id"])
            patch_features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": bbox_polygon(pbbox)},
                "properties": {"patch_id": case["patch_id"], "region": case["region"],
                               "review_only": True, "can_be_ground_truth": False},
            })
        if ebbox:
            event_features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": bbox_polygon(ebbox)},
                "properties": {"case_id": cid, "source": case["source_name"],
                               "geometry_kind": "review_only_traceable_extent",
                               "can_be_ground_truth": False},
            })
        if pbbox:
            case_features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": bbox_polygon(pbbox)},
                "properties": {"case_id": cid, "patch_id": case["patch_id"],
                               "spatial_relation": case["spatial_relation"],
                               "machine_pre_review": pre, "review_only": True,
                               "can_be_ground_truth": False},
            })

        # svg map
        svg = svg_for_case(case, pbbox, ebbox)
        (WS / "maps_svg" / f"{cid}.svg").write_text(svg, encoding="utf-8")
        n_svg += 1

        # dossier
        dossier = "\n".join([
            f"# {cid} — {case['patch_id']} ({case['region']})",
            "",
            "> AVISO: aderência espacial review-only. NÃO é ground truth de enchente por patch. "
            "Mesmo aprovação humana não cria ground truth automaticamente.",
            "",
            f"- **Região:** {case['region']}",
            f"- **Patch:** {case['patch_id']}",
            f"- **Relação espacial:** {case['spatial_relation']}",
            f"- **evidence_id:** {case['evidence_id']}",
            f"- **coordinate_id:** {case['coordinate_id']}",
            f"- **Fonte:** {case['source_name']}  ({case['source_path_or_url']})",
            f"- **Geometria:** {case['geometry_type']}",
            f"- **Data/período:** {case['date_or_period']}",
            f"- **Score/proxy:** {case['susceptibility_proxy']}",
            f"- **Features físicas:** {case['top_physical_features']}",
            f"- **Features hidrológicas:** {case['top_hydrological_features']}",
            f"- **Features espectrais:** {case['top_spectral_features']}",
            f"- **Interpretação:** aderência espacial review-only ({case['spatial_relation']}).",
            f"- **Limitações:** {case['limitations']}",
            f"- **Conflito conhecido:** {case['known_conflicts']}",
            f"- **machine_pre_review:** `{pre}`",
            f"- **Mapa:** `maps_svg/{cid}.svg`",
            "",
            "## Perguntas para o revisor humano",
            "",
            case["review_questions"],
            "",
            "## Campos a preencher (no form)",
            "",
            "source_verified, geometry_verified, temporal_match_verified, patch_relation_verified, "
            "conflict_checked, approved_for_tcc_example, approved_for_score_v6_context, "
            "approved_for_ground_truth(=false), final_case_strength, review_notes",
            "",
            "> Governança: can_be_ground_truth=false · allowed_for_training=false · review_only=true",
            "",
        ])
        write_markdown(WS / "case_dossiers" / f"{cid}.md", dossier)
        n_dossier += 1

        index_rows.append({
            "case_id": cid, "region": case["region"], "patch_id": case["patch_id"],
            "spatial_relation": case["spatial_relation"],
            "case_strength_pre_review": case["case_strength_pre_review"],
            "machine_pre_review": pre,
            "has_patch_bbox": str(bool(pbbox)).lower(),
            "has_event_geometry": str(bool(ebbox)).lower(),
            "dossier": f"case_dossiers/{cid}.md", "map_svg": f"maps_svg/{cid}.svg",
            "requires_human_review": "true",
            "can_be_ground_truth": "false", "allowed_for_training": "false", "review_only": "true",
        })

    # write index csv + md
    write_csv(WS / "SUSC_09A_review_index.csv", index_rows)
    idx_md = ["# SUSC-09A — Índice de Revisão Humana", "",
              f"> {MANDATORY}", "",
              f"Total de casos: **{len(index_rows)}**. Cada caso tem dossiê e mapa SVG.", "",
              "| case_id | região | patch | relação | machine_pre_review | dossiê | mapa |",
              "|---------|--------|-------|---------|--------------------|--------|------|"]
    for r in index_rows:
        idx_md.append(f"| {r['case_id']} | {r['region']} | {r['patch_id']} | {r['spatial_relation']} | "
                      f"`{r['machine_pre_review']}` | [{r['dossier']}]({r['dossier']}) | [svg]({r['map_svg']}) |")
    write_markdown(WS / "SUSC_09A_review_index.md", "\n".join(idx_md))

    # geojson layers
    def fc(features):
        return {"type": "FeatureCollection", "properties": {"review_only": True,
                "can_be_ground_truth": False}, "features": features}
    write_json(WS / "geojson" / "SUSC_09A_review_cases.geojson", fc(case_features))
    write_json(WS / "geojson" / "SUSC_09A_patch_bbox_layer.geojson", fc(patch_features))
    write_json(WS / "geojson" / "SUSC_09A_event_geometry_layer.geojson", fc(event_features))

    # prefilled form + instructions
    form_fields = ["case_id", "reviewer", "review_date", "source_verified", "geometry_verified",
                   "temporal_match_verified", "patch_relation_verified", "conflict_checked",
                   "approved_for_tcc_example", "approved_for_score_v6_context",
                   "approved_for_ground_truth", "final_case_strength", "review_notes"]
    form_rows = [{**{k: "" for k in form_fields}, "case_id": r["case_id"],
                  "approved_for_ground_truth": "false"} for r in index_rows]
    write_csv(WS / "forms" / "SUSC_09A_human_review_form_prefill.csv", form_rows, form_fields)
    write_markdown(WS / "forms" / "SUSC_09A_human_review_instructions.md",
                   "# SUSC-09A — Instruções de Revisão\n\n"
                   f"> {MANDATORY}\n\n"
                   "Abra cada dossiê em `case_dossiers/`, veja o mapa em `maps_svg/`, e preencha "
                   "`SUSC_09A_human_review_form_prefill.csv`. **`approved_for_ground_truth` deve "
                   "permanecer `false`** — aprovação humana não cria ground truth automaticamente.\n")

    # workspace manifest
    write_json(WS / "SUSC_09A_workspace_manifest.json", {
        "workspace": rel(WS), "n_cases": len(index_rows),
        "n_dossiers": n_dossier, "n_svg_maps": n_svg,
        "geojson_layers": ["SUSC_09A_review_cases.geojson", "SUSC_09A_patch_bbox_layer.geojson",
                           "SUSC_09A_event_geometry_layer.geojson"],
        "n_patch_features": len(patch_features), "n_event_features": len(event_features),
        "allowed_for_training": False, "can_be_used_as_ground_truth": False, "review_only": True,
        "scientific_status": "human_review_workspace_not_patch_ground_truth",
    })

    # report
    write_markdown(REPORT, "\n".join([
        "# SUSC-09A — Workspace de Revisão Humana", "",
        f"> {MANDATORY}", "",
        f"Workspace em `{rel(WS)}` com **{len(index_rows)} casos**, {n_dossier} dossiês, "
        f"{n_svg} mapas SVG e 3 camadas GeoJSON (patch bbox: {len(patch_features)} features; "
        f"event geometry: {len(event_features)} features).", "",
        "## Conteúdo", "",
        "- `SUSC_09A_review_index.md/.csv` — índice navegável dos casos.",
        "- `case_dossiers/<case_id>.md` — dossiê por caso (identificação, geometria, features, limitações, perguntas).",
        "- `maps_svg/<case_id>.svg` — mapa simples patch×evento (review-only).",
        "- `geojson/` — camadas para QGIS: patch bbox, geometria de evento rastreável, casos.",
        "- `forms/` — formulário pré-preenchido (`approved_for_ground_truth=false`) + instruções.",
        "", "## Governança", "",
        "Todos os artefatos: `can_be_ground_truth=false`, `allowed_for_training=false`, `review_only=true`. "
        "O workspace organiza a revisão; não a executa nem cria ground truth.", "",
        "## Limitações", "",
        "Mapas SVG e bboxes são auxílio visual review-only; geometria de evento é candidata/rastreável, "
        "não footprint validado. Aderência espacial ≠ ocorrência confirmada por patch.", "",
        "> A matriz de suscetibilidade ≠ ocorrência confirmada de enchente.", "",
    ]))

    print(f"\nworkspace: {rel(WS)} | cases: {len(index_rows)} | dossiers: {n_dossier} | svg: {n_svg}")
    print(f"geojson features: patch={len(patch_features)} event={len(event_features)} cases={len(case_features)}")
    print("\nWorkspace only organizes human review. No ground truth. review-only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
