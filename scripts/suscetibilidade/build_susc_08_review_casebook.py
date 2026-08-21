"""
SUSC-08 Spatiotemporal Review Casebook Builder (review-only)

Organizes the real spatial cases from SUSC-07B into an auditable human-review
package. Classifies each case by preliminary strength (never strong unless an
official validated footprint exists), records known conflicts, recommended use,
review questions and limitations. Adds regional/documentary context cases
(esp. Petropolis 2022) and the Curitiba acquisition gap.

Reads:
  - datasets/suscetibilidade/susc_features_by_patch_v1.csv
  - datasets/suscetibilidade/susc_07b_real_event_coordinates_v1.csv
  - datasets/suscetibilidade/susc_07b_event_evidence_geometry_enriched_v1.csv
  - outputs_public/suscetibilidade/SUSC_07B_refined_patch_event_overlay.csv
  - outputs_public/suscetibilidade/SUSC_07B_refined_patch_event_overlay_summary.csv
  - outputs_public/suscetibilidade/SUSC_07B_refined_overlay_limitations.json
  - outputs_public/suscetibilidade/SUSC_06B_feature_action_matrix.csv
  - outputs_public/suscetibilidade/feature_cards/susc_feature_cards_v1.json

Writes:
  - datasets/suscetibilidade/susc_08_spatiotemporal_review_cases_v1.csv
  - outputs_public/suscetibilidade/SUSC_08_review_casebook.md
  - outputs_public/suscetibilidade/SUSC_08_review_casebook_summary.csv
  - outputs_public/suscetibilidade/SUSC_08_manual_review_checklist.md
  - outputs_public/suscetibilidade/SUSC_08_manual_review_form.csv

Governance: review_only, no ground truth, no training. Spatial adherence is NOT
confirmed occurrence per patch.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())

MATRIX = ROOT / "datasets" / "suscetibilidade" / "susc_features_by_patch_v1.csv"
COORDS = ROOT / "datasets" / "suscetibilidade" / "susc_07b_real_event_coordinates_v1.csv"
OVERLAY = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_07B_refined_patch_event_overlay.csv"
DECISION = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_06B_feature_action_matrix.csv"
CARDS = ROOT / "outputs_public" / "suscetibilidade" / "feature_cards" / "susc_feature_cards_v1.json"

OUT_CASES = ROOT / "datasets" / "suscetibilidade" / "susc_08_spatiotemporal_review_cases_v1.csv"
OUT_CASEBOOK = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_08_review_casebook.md"
OUT_SUMMARY = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_08_review_casebook_summary.csv"
OUT_CHECKLIST = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_08_manual_review_checklist.md"
OUT_FORM = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_08_manual_review_form.csv"

PHYS = ["slope_mean", "elevation_mean", "hand_mean", "tpi_250m_mean"]
HYDRO = ["distance_to_water_mean", "twi_mean", "flow_acc_log_mean", "water_occurrence_patch"]
SPECTRAL = ["ndbi_mean", "mndwi_mean", "ndvi_mean"]
PROXY_COL = "score_evento_enchente_potencial_v5_core"

CASE_FIELDS = [
    "case_id", "patch_id", "region", "spatial_relation", "evidence_id", "coordinate_id",
    "geometry_type", "source_name", "source_path_or_url", "date_or_period",
    "susceptibility_proxy", "top_physical_features", "top_hydrological_features",
    "top_spectral_features", "coordinate_confidence", "temporal_confidence",
    "spatial_confidence", "evidence_patch_link_strength", "known_conflicts",
    "case_strength_pre_review", "recommended_use", "can_be_ground_truth",
    "allowed_for_training", "review_only", "requires_human_review",
    "review_questions", "limitations",
]

REVIEW_QUESTIONS = (
    "1) fonte oficial/tecnica/derivada? 2) coordenada e evento/setor de risco/estacao/"
    "poligono candidato/contexto? 3) data compativel? 4) geometria dentro/proxima do patch? "
    "5) proxy alto? 6) features fisicas explicam? 7) conflito entre fontes? 8) uso forte/cautela/"
    "contexto? 9) deve ficar bloqueado? 10) respeita review-only?"
)


def load_csv(p):
    with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        return list(csv.DictReader(f))


def feats_str(patch_row, names):
    parts = []
    for n in names:
        v = patch_row.get(n, "")
        if v not in ("", None):
            try:
                parts.append(f"{n}={round(float(v), 4)}")
            except ValueError:
                parts.append(f"{n}={v}")
    return "; ".join(parts)


def classify(spatial_relation, source_name, date_or_period):
    src = source_name.lower()
    is_context = any(k in src for k in ("aoi_context", "digitization", "risk_areas_context"))
    is_self_boundary = "patch_boundary" in src
    is_risk_points = any(k in src for k in ("defesa_civil", "risk_locations"))
    is_official = any(k in src for k in ("official", "cprm", "ground_reference"))
    weak_temporal = (not date_or_period) or date_or_period in ("unknown", "")

    if is_self_boundary:
        return "weak_contextual", "geometry_is_patch_self_boundary", "context_only", "low"
    if is_context:
        return "weak_contextual", "geometry_is_context_not_event", "context_only", "low"
    if spatial_relation == "near_patch_buffer_candidate" and is_risk_points:
        conflict = "geometry_source_disagreement"  # Charter758 x Defesa Civil known conflict
        strength = "moderate_candidate"
        return strength, conflict, "tcc_example_with_caution", "medium"
    if spatial_relation == "near_patch_buffer_candidate" and is_official:
        conflict = "event_to_patch_link_requires_review"
        strength = "moderate_candidate"
        if weak_temporal:
            conflict = "missing_or_weak_temporal_link"
        return strength, conflict, "tcc_example_with_caution", "medium"
    if spatial_relation == "bbox_overlap":
        return "weak_contextual", "geometry_is_context_not_event", "context_only", "low"
    return "insufficient", "none", "acquire_geometry_first", "low"


def main() -> int:
    print("=" * 60)
    print("SUSC-08 Spatiotemporal Review Casebook Builder (review-only)")
    print("=" * 60)

    for p in (MATRIX, COORDS, OVERLAY):
        if not p.exists():
            print(f"STOP: required input not found: {p}")
            return 1

    patches = {r["patch_id"]: r for r in load_csv(MATRIX)}
    coords_by_src = {}
    for c in load_csv(COORDS):
        coords_by_src[c["source_name"]] = c
    overlay = load_csv(OVERLAY)

    spatial = [r for r in overlay if r["spatial_relation"] in
               ("bbox_overlap", "near_patch_buffer_candidate", "exact_patch_overlap")]
    regional = [r for r in overlay if r["spatial_relation"] in
                ("same_region_period", "same_region_only")]
    documentary = [r for r in overlay if r["spatial_relation"] == "documentary_context_only"]

    cases = []
    cid = 0

    # 1) spatial cases (the real 9)
    for r in spatial:
        cid += 1
        patch = patches.get(r["patch_id"], {})
        coord = coords_by_src.get(r["geometry_source"], {})
        date = coord.get("date_or_period", "unknown")
        strength, conflict, rec_use, spatial_conf = classify(
            r["spatial_relation"], r["geometry_source"], date)
        temporal_conf = "low" if date in ("", "unknown") else "medium"
        cases.append({
            "case_id": f"CASE_{cid:03d}",
            "patch_id": r["patch_id"], "region": r["region"],
            "spatial_relation": r["spatial_relation"],
            "evidence_id": r.get("evidence_id", ""), "coordinate_id": coord.get("coordinate_id", ""),
            "geometry_type": coord.get("geometry_type", ""),
            "source_name": r["geometry_source"], "source_path_or_url": coord.get("source_path_or_url", ""),
            "date_or_period": date,
            "susceptibility_proxy": patch.get(PROXY_COL, r.get("susceptibility_proxy", "")),
            "top_physical_features": feats_str(patch, PHYS),
            "top_hydrological_features": feats_str(patch, HYDRO),
            "top_spectral_features": feats_str(patch, SPECTRAL),
            "coordinate_confidence": "traceable_review_only",
            "temporal_confidence": temporal_conf,
            "spatial_confidence": spatial_conf,
            "evidence_patch_link_strength": "regional_manual" if coord.get("requires_manual_review") == "true" else "direct",
            "known_conflicts": conflict,
            "case_strength_pre_review": strength,
            "recommended_use": rec_use,
            "can_be_ground_truth": "false", "allowed_for_training": "false", "review_only": "true",
            "requires_human_review": "true",
            "review_questions": REVIEW_QUESTIONS,
            "limitations": (
                "Aderencia espacial review-only; NAO e ocorrencia confirmada por patch. "
                + ("Ponto de risco != footprint de evento. " if "risk" in r["geometry_source"].lower() or "defesa" in r["geometry_source"].lower() else "")
                + ("Geometria e contexto/AOI, nao footprint de evento. " if conflict == "geometry_is_context_not_event" else "")
                + ("Fonte e a propria fronteira do patch (quase circular). " if conflict == "geometry_is_patch_self_boundary" else "")
            ),
        })

    # 2) Petropolis 2022 regional context (secondary), up to 3
    pet_regional = [r for r in regional if r["region"] == "petropolis"][:3]
    for r in pet_regional:
        cid += 1
        cases.append({
            "case_id": f"CASE_{cid:03d}", "patch_id": "REGION_LEVEL_petropolis",
            "region": "petropolis", "spatial_relation": r["spatial_relation"],
            "evidence_id": r.get("evidence_id", ""), "coordinate_id": "",
            "geometry_type": "", "source_name": "SUSC-07A regional/documentary",
            "source_path_or_url": "", "date_or_period": r.get("temporal_relation", "dated"),
            "susceptibility_proxy": "", "top_physical_features": "", "top_hydrological_features": "",
            "top_spectral_features": "", "coordinate_confidence": "none",
            "temporal_confidence": "medium", "spatial_confidence": "region_only",
            "evidence_patch_link_strength": "regional", "known_conflicts": "none",
            "case_strength_pre_review": "weak_contextual", "recommended_use": "context_only",
            "can_be_ground_truth": "false", "allowed_for_training": "false", "review_only": "true",
            "requires_human_review": "true", "review_questions": REVIEW_QUESTIONS,
            "limitations": "Associacao regional datada (Petropolis 2022); contexto, nao overlap de patch.",
        })

    # 3) Curitiba acquisition gap
    cid += 1
    cur_doc = next((r for r in documentary if r["region"] == "curitiba"), None)
    cases.append({
        "case_id": f"CASE_{cid:03d}", "patch_id": "REGION_LEVEL_curitiba", "region": "curitiba",
        "spatial_relation": "insufficient_for_patch_link",
        "evidence_id": cur_doc.get("evidence_id", "") if cur_doc else "", "coordinate_id": "",
        "geometry_type": "", "source_name": "no_real_coordinate_extracted",
        "source_path_or_url": "", "date_or_period": "unknown", "susceptibility_proxy": "",
        "top_physical_features": "", "top_hydrological_features": "", "top_spectral_features": "",
        "coordinate_confidence": "none", "temporal_confidence": "low", "spatial_confidence": "none",
        "evidence_patch_link_strength": "none", "known_conflicts": "none",
        "case_strength_pre_review": "insufficient", "recommended_use": "acquire_geometry_first",
        "can_be_ground_truth": "false", "allowed_for_training": "false", "review_only": "true",
        "requires_human_review": "true", "review_questions": REVIEW_QUESTIONS,
        "limitations": "Curitiba sem coordenada real extraida; lacuna de aquisicao (GeoCuritiba/IPPUC).",
    })

    # write cases CSV
    OUT_CASES.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CASES.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CASE_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(cases)

    # summary CSV
    by_strength = Counter(c["case_strength_pre_review"] for c in cases)
    by_rel = Counter(c["spatial_relation"] for c in cases)
    with OUT_SUMMARY.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["metric", "key", "count"])
        for k, v in sorted(by_strength.items()):
            w.writerow(["case_strength_pre_review", k, v])
        for k, v in sorted(by_rel.items()):
            w.writerow(["spatial_relation", k, v])
        w.writerow(["total_cases", "all", len(cases)])

    # casebook MD
    lines = [
        "# SUSC-08 — Casebook de Revisão Espaço-Temporal",
        "",
        "> O SUSC-08 não cria ground truth de enchente por patch. Aderência espacial é review-only, "
        "NÃO ocorrência confirmada. Todos os casos exigem revisão humana.",
        "",
        f"Total de casos: **{len(cases)}** "
        f"({sum(1 for c in cases if c['spatial_relation'] in ('bbox_overlap','near_patch_buffer_candidate'))} espaciais reais + contexto/lacuna).",
        "",
        "## Resumo por força preliminar",
        "",
        "| força | nº |", "|-------|----|",
    ]
    for k in ("strong_candidate", "moderate_candidate", "weak_contextual", "blocked_conflict", "insufficient"):
        lines.append(f"| {k} | {by_strength.get(k, 0)} |")
    lines.append("")
    for c in cases:
        lines.append(f"## {c['case_id']} — {c['patch_id']} ({c['region']}) `[{c['case_strength_pre_review']}]`")
        lines.append("")
        lines.append(f"- **Relação espacial:** {c['spatial_relation']} · **fonte:** {c['source_name']}")
        lines.append(f"- **Geometria:** {c['geometry_type']} · **coord_id:** {c['coordinate_id']} · **data:** {c['date_or_period']}")
        lines.append(f"- **Proxy suscetibilidade:** {c['susceptibility_proxy']}")
        if c["top_physical_features"]:
            lines.append(f"- **Físicas:** {c['top_physical_features']}")
            lines.append(f"- **Hidrológicas:** {c['top_hydrological_features']}")
            lines.append(f"- **Espectrais:** {c['top_spectral_features']}")
        lines.append(f"- **Confiança:** coord={c['coordinate_confidence']} · temporal={c['temporal_confidence']} · espacial={c['spatial_confidence']}")
        lines.append(f"- **Vínculo evidência↔patch:** {c['evidence_patch_link_strength']} · **conflitos:** {c['known_conflicts']}")
        lines.append(f"- **Uso recomendado:** {c['recommended_use']} · **requires_human_review:** {c['requires_human_review']}")
        lines.append(f"- **Limitações:** {c['limitations']}")
        lines.append(f"- **Governança:** ground_truth=false · training=false · review_only=true")
        lines.append("")
    OUT_CASEBOOK.write_text("\n".join(lines), encoding="utf-8")

    # manual review checklist
    OUT_CHECKLIST.write_text(
        "# SUSC-08 — Checklist de Revisão Manual\n\n"
        "> Mesmo a aprovação humana NÃO transforma automaticamente um caso em ground truth. "
        "Aderência espacial permanece review-only.\n\n"
        "Para cada `case_id` do casebook, responda:\n\n"
        "1. A fonte da coordenada é oficial, técnica ou derivada?\n"
        "2. A coordenada representa evento observado, setor de risco, estação, polígono candidato ou contexto?\n"
        "3. Há data/período compatível com a evidência?\n"
        "4. A geometria cai dentro ou próxima do patch?\n"
        "5. O patch possui alta/média suscetibilidade proxy?\n"
        "6. As features físicas do patch explicam a aderência?\n"
        "7. Existe conflito entre fontes (ex.: Charter 758 × Defesa Civil)?\n"
        "8. A associação pode ser usada em texto como exemplo forte, exemplo com cautela ou só contexto?\n"
        "9. O caso deve permanecer bloqueado?\n"
        "10. A conclusão respeita review-only?\n\n"
        "Registre as respostas em `SUSC_08_manual_review_form.csv`. "
        "`approved_for_ground_truth` deve permanecer `false`.\n",
        encoding="utf-8",
    )

    # manual review form (pre-filled, awaiting human)
    form_fields = [
        "case_id", "reviewer", "review_date", "source_verified", "geometry_verified",
        "temporal_match_verified", "patch_relation_verified", "conflict_checked",
        "approved_for_tcc_example", "approved_for_score_v6_context",
        "approved_for_ground_truth", "final_case_strength", "review_notes",
    ]
    with OUT_FORM.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=form_fields, lineterminator="\n")
        w.writeheader()
        for c in cases:
            w.writerow({
                "case_id": c["case_id"], "reviewer": "", "review_date": "",
                "source_verified": "", "geometry_verified": "", "temporal_match_verified": "",
                "patch_relation_verified": "", "conflict_checked": "",
                "approved_for_tcc_example": "", "approved_for_score_v6_context": "",
                "approved_for_ground_truth": "false", "final_case_strength": "",
                "review_notes": "",
            })

    print(f"\ncases: {len(cases)} | by strength: {dict(by_strength)}")
    print(f"spatial real (bbox+near): {sum(1 for c in cases if c['spatial_relation'] in ('bbox_overlap','near_patch_buffer_candidate'))}")
    print("\nHuman review package. No ground truth. review-only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
