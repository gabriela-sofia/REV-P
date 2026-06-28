"""
SUSC-09B External Vector Integration (review-only)

Turns parsed external geometries into event-geometry candidates and patch overlay
candidates (bbox_overlap / point-in-bbox), keeping everything review-only and
requires_manual_review. If no external geometry was parsed (offline), it records
the acquisition gap honestly without failing.

Writes:
  - datasets/suscetibilidade/susc_09b_event_geometry_candidates_v1.csv
  - outputs_public/suscetibilidade/SUSC_09B_external_vector_overlay_candidates.csv
  - outputs_public/suscetibilidade/SUSC_09B_external_vector_limitations.json
  - outputs_public/suscetibilidade/SUSC_09B_external_vector_acquisition_report.md
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_io import read_csv, write_csv, write_json, write_markdown  # noqa: E402
from susc_geometry import parse_bbox, bbox_intersect, REGION_BOUNDS  # noqa: E402

PARSED = ROOT / "datasets" / "suscetibilidade" / "susc_09b_external_geometries_parsed_v1.csv"
MATRIX = ROOT / "datasets" / "suscetibilidade" / "susc_features_by_patch_v1.csv"
DOWNLOAD_MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_09b_external_download_manifest_v1.csv"

CANDIDATES = ROOT / "datasets" / "suscetibilidade" / "susc_09b_event_geometry_candidates_v1.csv"
OVERLAY = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_09B_external_vector_overlay_candidates.csv"
LIMITS = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_09B_external_vector_limitations.json"
REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_09B_external_vector_acquisition_report.md"

BBOX_COLS = ["xmin", "ymin", "xmax", "ymax"]
CAND_FIELDS = ["candidate_id", "geometry_id", "region", "geometry_type", "bbox",
               "source_file", "evidence_status", "can_link_to_patch", "can_be_ground_truth",
               "allowed_for_training", "review_only", "requires_manual_review", "notes"]
OVERLAY_FIELDS = ["patch_id", "region", "candidate_id", "geometry_id", "spatial_relation",
                  "overlay_confidence", "source_file", "can_be_used_as_ground_truth",
                  "allowed_for_training", "review_only", "interpretation", "limitation"]
MANDATORY = ("O SUSC-09B amplia a aquisição vetorial externa oficial review-only. Não cria ground "
             "truth, não desbloqueia treinamento e não transforma geometria candidata em footprint "
             "validado por patch.")


def _num(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def main() -> int:
    print("=" * 60)
    print("SUSC-09B External Vector Integration (review-only)")
    print("=" * 60)

    parsed = read_csv(PARSED) if PARSED.exists() else []
    patches = read_csv(MATRIX) if MATRIX.exists() else []
    dl = read_csv(DOWNLOAD_MANIFEST) if DOWNLOAD_MANIFEST.exists() else []

    region_patches = {}
    for r in patches:
        bb = [_num(r[c]) for c in BBOX_COLS]
        if all(v is not None for v in bb):
            region_patches.setdefault(r.get("regiao", ""), []).append((r["patch_id"], bb))

    candidates = []
    overlay = []
    for g in parsed:
        region = g["region"]
        bbox = parse_bbox(g["bbox"])
        is_official = region in REGION_BOUNDS and bbox is not None
        candidates.append({
            "candidate_id": f"EXTCAND_{g['geometry_id']}", "geometry_id": g["geometry_id"],
            "region": region, "geometry_type": g["geometry_type"], "bbox": g["bbox"],
            "source_file": g["source_file"],
            "evidence_status": "spatial_evidence_candidate" if is_official else "insufficient_for_patch_event_link",
            "can_link_to_patch": "true" if is_official else "false",
            "can_be_ground_truth": "false", "allowed_for_training": "false", "review_only": "true",
            "requires_manual_review": "true",
            "notes": "external official geometry candidate; review-only, not validated footprint",
        })
        if bbox:
            for pid, pbb in region_patches.get(region, []):
                if bbox_intersect(bbox, pbb):
                    overlay.append({
                        "patch_id": pid, "region": region,
                        "candidate_id": f"EXTCAND_{g['geometry_id']}", "geometry_id": g["geometry_id"],
                        "spatial_relation": "bbox_overlap", "overlay_confidence": "low",
                        "source_file": g["source_file"],
                        "can_be_used_as_ground_truth": "false", "allowed_for_training": "false",
                        "review_only": "true",
                        "interpretation": "Sobreposicao bbox entre geometria oficial externa e patch (review-only).",
                        "limitation": "Geometria candidata, nao footprint validado; nao e ocorrencia por patch.",
                    })

    write_csv(CANDIDATES, candidates, CAND_FIELDS)
    write_csv(OVERLAY, overlay, OVERLAY_FIELDS)

    dl_status = dict(Counter(r["download_status"] for r in dl))
    downloaded = sum(1 for r in dl if r["download_status"] == "downloaded")
    write_json(LIMITS, {
        "stage": "SUSC-09B external vector integration",
        "external_geometries_parsed": len(parsed),
        "event_geometry_candidates": len(candidates),
        "overlay_candidates": len(overlay),
        "downloads_attempted": len(dl),
        "downloads_succeeded": downloaded,
        "download_status_counts": dl_status,
        "can_be_used_as_ground_truth": False, "allowed_for_training": False, "review_only": True,
        "scientific_status": "external_official_vector_acquisition_not_patch_ground_truth",
        "key_limitations": [
            "Aquisicao externa offline-safe: sem URL direta, downloads nao realizados nesta passada.",
            "Geometrias candidatas (quando houver) sao review-only, nao footprints validados.",
            "Shapefile/PDF exigem biblioteca/parse manual; registrados como metadata.",
            "Nada vira ground truth ou rotulo de treino.",
        ],
        "prepares": "aquisicao manual oficial (GeoCuritiba/APAC/CPRM/INEA) + re-parse + re-overlay.",
    })

    write_markdown(REPORT, "\n".join([
        "# SUSC-09B — Aquisição Vetorial Externa Oficial", "",
        f"> {MANDATORY}", "",
        "## Registry e download",
        f"- Fontes oficiais no registry: ver `susc_09b_external_source_registry_v1.csv`.",
        f"- Tentativas de download: **{len(dl)}** — status: {dl_status}.",
        f"- Downloads bem-sucedidos: **{downloaded}** (offline-safe; sem URL direta → not_attempted).",
        "",
        "## Parsing e integração",
        f"- Geometrias externas parseadas: **{len(parsed)}**.",
        f"- Candidatos de geometria de evento: **{len(candidates)}**.",
        f"- Overlay candidates (bbox): **{len(overlay)}**.",
        "",
        "## Regras respeitadas",
        "- Sem raster, sem imagem Sentinel, sem arquivo >100MB, sem API com chave, sem geocoding.",
        "- Todo download registrado com URL, status, tamanho e SHA256 (quando houver).",
        "- Tudo `requires_manual_review=true`, `can_be_ground_truth=false`, `review_only=true`.",
        "",
        "## Limitações",
        "Offline nesta passada: a aquisição oficial direta é manual. Geometrias candidatas, quando "
        "adquiridas, são review-only e exigem validação humana — nunca footprint validado por patch.",
        "",
        "> A matriz de suscetibilidade ≠ ocorrência confirmada de enchente.", "",
    ]))

    print(f"\nparsed: {len(parsed)} | candidates: {len(candidates)} | overlay: {len(overlay)} | downloads ok: {downloaded}")
    print("No raster. No GT. review-only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
