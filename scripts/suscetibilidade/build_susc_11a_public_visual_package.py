"""
SUSC-11A Public Visual Package (review-only)

Builds a TCC/article communication package: SVG figures (pure Python, no
dependency), CSV tables, GeoJSON, slide content and a manifest. The figures are
review-only — not ground truth, not operational prediction, not confirmed
occurrence per patch.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_io import read_csv, read_json, write_csv, write_json, write_markdown, ensure_dir, rel  # noqa: E402

OUTS = ROOT / "outputs_public" / "suscetibilidade"
SCORE = ROOT / "datasets" / "suscetibilidade" / "susc_score_v6_candidate_by_patch_v1.csv"
BY_REGION = OUTS / "SUSC_10A_score_v6_candidate_by_region.csv"
TOP = OUTS / "SUSC_10A_score_v6_candidate_top_patches.csv"
SCEN = OUTS / "SUSC_10C_score_sensitivity_scenarios.csv"
RANK = OUTS / "SUSC_10B_score_rank_shift_by_patch.csv"
CASES = ROOT / "datasets" / "suscetibilidade" / "susc_08_spatiotemporal_review_cases_v1.csv"
DINO_SUMMARY = OUTS / "SUSC_10D_dino_score_diagnostics_summary.json"

PKG = OUTS / "SUSC_11A_public_visual_package"
MANDATORY = ("Este pacote visual comunica resultados review-only de suscetibilidade. As figuras não "
             "representam ground truth, não mostram predição operacional de enchente e não confirmam "
             "ocorrência por patch.")
FOOTER = "review-only — NÃO ground truth"


def svg(body, w=520, h=320, title=""):
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">'
            f'<rect width="{w}" height="{h}" fill="#ffffff"/>'
            f'<text x="{w/2:.0f}" y="20" text-anchor="middle" font-size="14" fill="#222">{title}</text>'
            f'{body}'
            f'<text x="{w/2:.0f}" y="{h-6}" text-anchor="middle" font-size="10" fill="#999">{FOOTER}</text>'
            f'</svg>')


def bar_chart(labels, values, title, color="#1f6feb"):
    w, h = 520, 320
    n = len(values)
    if n == 0:
        return svg('<text x="260" y="160" text-anchor="middle" fill="#888">sem dados</text>', w, h, title)
    mx = max(values) or 1
    bw = (w - 80) / n
    bars = []
    for i, v in enumerate(values):
        bh = (v / mx) * (h - 90)
        x = 50 + i * bw
        y = h - 40 - bh
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw*0.8:.1f}" height="{bh:.1f}" fill="{color}"/>')
        bars.append(f'<text x="{x+bw*0.4:.1f}" y="{h-26}" text-anchor="middle" font-size="9" fill="#555" '
                    f'transform="rotate(0 {x+bw*0.4:.1f} {h-26})">{labels[i]}</text>')
    return svg("".join(bars), w, h, title)


def histogram(values, title, bins=10, color="#2da44e"):
    w, h = 520, 320
    if not values:
        return svg('<text x="260" y="160" text-anchor="middle" fill="#888">sem dados</text>', w, h, title)
    lo, hi = min(values), max(values)
    rng = (hi - lo) or 1
    counts = [0] * bins
    for v in values:
        b = min(bins - 1, int((v - lo) / rng * bins))
        counts[b] += 1
    return bar_chart([f"{lo + i*rng/bins:.2f}" for i in range(bins)], counts, title, color)


def scatter(xs, ys, title, xlabel="v5", ylabel="v6"):
    w, h = 520, 320
    if not xs:
        return svg('<text x="260" y="160" text-anchor="middle" fill="#888">sem dados</text>', w, h, title)
    pts = []
    for x, y in zip(xs, ys):
        px = 50 + x * (w - 80)
        py = h - 40 - y * (h - 90)
        pts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2" fill="#8957e5" fill-opacity="0.5"/>')
    diag = f'<line x1="50" y1="{h-40}" x2="{w-30}" y2="40" stroke="#ccc" stroke-dasharray="4"/>'
    axis = (f'<text x="{w/2:.0f}" y="{h-26}" text-anchor="middle" font-size="10" fill="#777">{xlabel}</text>'
            f'<text x="16" y="{h/2:.0f}" font-size="10" fill="#777" transform="rotate(-90 16 {h/2:.0f})">{ylabel}</text>')
    return svg(diag + "".join(pts) + axis, w, h, title)


def main() -> int:
    print("=" * 60)
    print("SUSC-11A Public Visual Package (review-only)")
    print("=" * 60)
    if not SCORE.exists():
        print("STOP: score v6 not found")
        return 1

    ensure_dir(PKG / "figures")
    ensure_dir(PKG / "tables")
    ensure_dir(PKG / "geojson")
    ensure_dir(PKG / "slides_content")

    score = read_csv(SCORE)
    vals = [float(r["susc_score_v6_candidate"]) for r in score]

    figs = {}
    figs["fig_01_score_v6_distribution"] = histogram(vals, "Distribuição do score v6 (review-only)")
    if BY_REGION.exists():
        reg = read_csv(BY_REGION)
        figs["fig_02_score_v6_by_region"] = bar_chart(
            [r["regiao"] for r in reg], [float(r["score_mean"]) for r in reg],
            "Score v6 médio por região (review-only)")
    if TOP.exists():
        top = read_csv(TOP)[:15]
        figs["fig_03_top15_patches"] = bar_chart(
            [r["patch_id"].replace("recife_", "rec").replace("petropolis_", "pet").replace("curitiba_", "cur") for r in top],
            [float(r["susc_score_v6_candidate"]) for r in top], "Top 15 patches por score v6", "#bf3989")
    if SCEN.exists():
        sc = read_csv(SCEN)
        figs["fig_04_sensitivity_by_scenario"] = bar_chart(
            [r["scenario"][:6] for r in sc], [float(r["class_agreement_vs_baseline"]) for r in sc],
            "Sensibilidade: concordância de classe vs baseline", "#d29922")
    if RANK.exists():
        rk = read_csv(RANK)
        figs["fig_05_rank_shift_v6_vs_v5"] = scatter(
            [float(r["score_v5_proxy"]) for r in rk], [float(r["score_v6"]) for r in rk],
            "Score v6 vs proxy v5 (review-only, não validação)")
    # fig 06 spatial cases
    spatial_counts = Counter()
    if CASES.exists():
        for c in read_csv(CASES):
            if c["spatial_relation"] in {"bbox_overlap", "near_patch_buffer_candidate"}:
                spatial_counts[c["spatial_relation"]] += 1
    figs["fig_06_spatial_evidence_cases"] = bar_chart(
        list(spatial_counts.keys()) or ["nenhum"], list(spatial_counts.values()) or [0],
        "Casos espaciais review-only (07B/08)", "#e8730a")
    # fig 07 pipeline
    stages = ["01-02 schema", "03 matriz", "04 proven.", "05 cards", "06A/B baseline",
              "07A/B overlay", "08 review", "09A/B work.", "10A score v6", "10B-D anal."]
    seg = []
    for i, s in enumerate(stages):
        x = 20 + i * 49
        seg.append(f'<rect x="{x}" y="140" width="44" height="40" rx="4" fill="#dbeafe" stroke="#1f6feb"/>'
                   f'<text x="{x+22}" y="164" text-anchor="middle" font-size="7" fill="#1f3a8a">{s}</text>')
        if i < len(stages) - 1:
            seg.append(f'<line x1="{x+44}" y1="160" x2="{x+49}" y2="160" stroke="#1f6feb"/>')
    figs["fig_07_pipeline_susc_01_10"] = svg("".join(seg), 520, 320, "Pipeline metodológico SUSC-01→10")
    # fig 08 concept matrix
    concept = (
        '<rect x="40" y="60" width="130" height="60" rx="6" fill="#e6f4ea" stroke="#2da44e"/>'
        '<text x="105" y="95" text-anchor="middle" font-size="11" fill="#14532d">Suscetibilidade<tspan x="105" dy="14">(predisposição)</tspan></text>'
        '<rect x="195" y="60" width="130" height="60" rx="6" fill="#fff4e5" stroke="#e8730a"/>'
        '<text x="260" y="95" text-anchor="middle" font-size="11" fill="#7c2d12">Evento observado<tspan x="260" dy="14">(documental)</tspan></text>'
        '<rect x="350" y="60" width="130" height="60" rx="6" fill="#fde8e8" stroke="#cf222e"/>'
        '<text x="415" y="95" text-anchor="middle" font-size="11" fill="#7f1d1d">Ground truth<tspan x="415" dy="14">(NÃO existe)</tspan></text>'
        '<text x="260" y="160" text-anchor="middle" font-size="13" fill="#222">≠ &#160;&#160;&#160; ≠</text>'
        '<text x="260" y="200" text-anchor="middle" font-size="11" fill="#555">aderência espacial review-only não fecha a cadeia</text>')
    figs["fig_08_concept_matrix"] = svg(concept, 520, 320, "Suscetibilidade ≠ evento ≠ ground truth")
    # fig 09 dino
    dino_absent = True
    if DINO_SUMMARY.exists():
        dino_absent = str(read_json(DINO_SUMMARY).get("dino_status", "")).startswith("ABSENT")
    figs["fig_09_dino_diagnostics"] = svg(
        f'<text x="260" y="150" text-anchor="middle" font-size="13" fill="#666">'
        f'{"DINO ausente: sem vetores carregáveis por patch" if dino_absent else "DINO diagnóstico disponível"}'
        f'</text><text x="260" y="175" text-anchor="middle" font-size="10" fill="#999">placeholder honesto</text>',
        520, 320, "Diagnóstico DINO (camada representacional)")

    for name, content in figs.items():
        (PKG / "figures" / f"{name}.svg").write_text(content, encoding="utf-8")

    # tables
    write_csv(PKG / "tables" / "top_patches_for_tcc.csv",
              read_csv(TOP)[:15] if TOP.exists() else [],
              list(read_csv(TOP)[0].keys()) if TOP.exists() and read_csv(TOP) else ["patch_id"])
    write_csv(PKG / "tables" / "score_v6_by_region.csv",
              read_csv(BY_REGION) if BY_REGION.exists() else [],
              list(read_csv(BY_REGION)[0].keys()) if BY_REGION.exists() and read_csv(BY_REGION) else ["regiao"])
    # cases for tcc with caution (moderate)
    caution = []
    if CASES.exists():
        for c in read_csv(CASES):
            if c.get("recommended_use") == "tcc_example_with_caution":
                caution.append({"case_id": c["case_id"], "patch_id": c["patch_id"], "region": c["region"],
                                "spatial_relation": c["spatial_relation"], "source_name": c["source_name"],
                                "case_strength_pre_review": c["case_strength_pre_review"],
                                "known_conflicts": c["known_conflicts"], "review_only": "true"})
    write_csv(PKG / "tables" / "cases_for_tcc_with_caution.csv", caution,
              ["case_id", "patch_id", "region", "spatial_relation", "source_name",
               "case_strength_pre_review", "known_conflicts", "review_only"])
    write_csv(PKG / "tables" / "limitations_table.csv", [
        {"limitation": "Score v6 é determinístico candidato, não modelo treinado nem predição", "scope": "10A"},
        {"limitation": "Correlação v6×v5 não é validação supervisionada", "scope": "10B"},
        {"limitation": "Evidência espacial é review-only, não ocorrência confirmada", "scope": "07B/08"},
        {"limitation": "DINO ausente (sem vetores carregáveis por patch)", "scope": "10D"},
        {"limitation": "Aquisição externa oficial pendente (offline)", "scope": "09B"},
        {"limitation": "Pesos/thresholds do score são escolha metodológica documentada", "scope": "10A/10C"},
    ], ["limitation", "scope"])
    write_csv(PKG / "tables" / "methodology_artifacts_index.csv", [
        {"stage": "SUSC-03", "artifact": "susc_features_by_patch_v1.csv"},
        {"stage": "SUSC-05", "artifact": "feature_cards/susc_feature_cards_v1.json"},
        {"stage": "SUSC-06B", "artifact": "SUSC_06B_feature_action_matrix.csv"},
        {"stage": "SUSC-07B", "artifact": "SUSC_07B_refined_patch_event_overlay.csv"},
        {"stage": "SUSC-08", "artifact": "susc_08_spatiotemporal_review_cases_v1.csv"},
        {"stage": "SUSC-09A", "artifact": "SUSC_09A_human_review_workspace/"},
        {"stage": "SUSC-10A", "artifact": "susc_score_v6_candidate_by_patch_v1.csv"},
        {"stage": "SUSC-10B", "artifact": "SUSC_10B_score_comparison_report.md"},
        {"stage": "SUSC-10C", "artifact": "SUSC_10C_score_sensitivity_report.md"},
        {"stage": "SUSC-10D", "artifact": "SUSC_10D_dino_score_diagnostics_report.md"},
    ], ["stage", "artifact"])

    # geojson: copy patch bbox layer from 09A if present, else build minimal from top patches
    src_layer = OUTS / "SUSC_09A_human_review_workspace" / "geojson" / "SUSC_09A_patch_bbox_layer.geojson"
    if src_layer.exists():
        write_json(PKG / "geojson" / "SUSC_11A_patch_bbox_layer.geojson", read_json(src_layer))

    # slides
    slides = {
        "slide_01_problem": ("Problema", "Suscetibilidade urbana a enchentes em 3 regiões, sem ground truth de ocorrência.",
                             "fig_08_concept_matrix", "Distinguir suscetibilidade, evento e ground truth.", True),
        "slide_02_pipeline": ("Pipeline", "Cadeia auditável SUSC-01→10, review-only.",
                              "fig_07_pipeline_susc_01_10", "Cada etapa é rastreável e governada.", False),
        "slide_03_features": ("Features", "19 features físicas/hidro/espectrais aprovadas pelos gates.",
                             "fig_02_score_v6_by_region", "Sem labels/proxy/score v5 como feature.", False),
        "slide_04_score_v6": ("Score v6 candidato", "Score determinístico explicável em [0,1], classes low/medium/high.",
                             "fig_01_score_v6_distribution", "Não é predição operacional.", True),
        "slide_05_spatial_evidence": ("Evidência espacial", "9 relações espaciais review-only (07B/08).",
                                     "fig_06_spatial_evidence_cases", "Aderência espacial ≠ ocorrência por patch.", True),
        "slide_06_dino": ("DINO", "Camada representacional complementar; ausente nesta passada.",
                         "fig_09_dino_diagnostics", "DINO nunca como classificador.", True),
        "slide_07_limitations": ("Limitações", "Sem GT, sem treino, sem footprint validado; aquisição externa pendente.",
                                "fig_05_rank_shift_v6_vs_v5", "Honestidade metodológica.", True),
        "slide_08_conclusion": ("Conclusão", "Infraestrutura review-only robusta e auditável para o TCC.",
                               "fig_03_top15_patches", "Próximo: validação humana + aquisição oficial.", False),
    }
    for name, (title, msg, fig, text, alert) in slides.items():
        write_markdown(PKG / "slides_content" / f"{name}.md", "\n".join([
            f"# {title}", "", f"**Mensagem principal:** {msg}", "",
            f"**Figura sugerida:** `figures/{fig}.svg`", "",
            f"**Texto curto:** {text}", "",
            ("> ⚠️ Alerta metodológico: review-only; não é ground truth nem predição de enchente." if alert else ""),
            "",
        ]))

    # README + manifest
    write_markdown(PKG / "README.md", "\n".join([
        "# SUSC-11A — Pacote Visual Público (review-only)", "",
        f"> {MANDATORY}", "",
        f"- **figures/**: {len(figs)} figuras SVG.",
        "- **tables/**: 5 tabelas (top patches, por região, casos com cautela, limitações, índice de artefatos).",
        "- **geojson/**: camada de bbox de patch (de SUSC-09A).",
        "- **slides_content/**: 8 slides (título, mensagem, figura, texto, alerta).", "",
        "Toda figura traz o rodapé `review-only — NÃO ground truth`.", "",
        "> A matriz de suscetibilidade ≠ ocorrência confirmada de enchente.", "",
    ]))
    write_json(PKG / "SUSC_11A_visual_package_manifest.json", {
        "package": rel(PKG), "n_figures": len(figs), "figures": sorted(figs.keys()),
        "n_tables": 5, "n_slides": len(slides),
        "figure_backend": "pure_svg",
        "can_be_used_as_ground_truth": False, "allowed_for_training": False, "review_only": True,
        "scientific_status": "public_visual_communication_review_only_not_ground_truth",
    })

    print(f"\npackage: {rel(PKG)} | figures: {len(figs)} | tables: 5 | slides: {len(slides)}")
    print("review-only communication package. No ground truth.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
