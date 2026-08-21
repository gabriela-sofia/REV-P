"""
SUSC-05 Feature Card Builder

Transforms the audited susceptibility features (SUSC-01..04) into explainable
methodological units ("feature cards"). Each card consolidates formula/derivation,
interpretation, expected direction, source, limitations and future role.

Reads:
  - datasets/suscetibilidade/susc_features_by_patch_v1.csv
  - schemas/suscetibilidade/susc_features_schema_v1.json
  - schemas/suscetibilidade/susc_feature_scientific_direction_v1.json
  - manifests/suscetibilidade/susc_features_provenance_manifest_v1.csv
  - manifests/suscetibilidade/susc_feature_provenance_audit_v1.csv
  - outputs_public/suscetibilidade/SUSC_04_score_v6_feature_decision_matrix.csv

Writes:
  - outputs_public/suscetibilidade/feature_cards/susc_feature_cards_v1.json
  - outputs_public/suscetibilidade/feature_cards/SUSC_05_feature_cards_catalog.md
  - outputs_public/suscetibilidade/feature_cards/SUSC_05_feature_cards_summary.csv

This script does NOT compute any score, does NOT train any model, and does NOT
create ground truth. Heuristic scores/labels are carded as proxy_only, never as
truth. Susceptibility features != confirmed flood occurrence.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())

CSV_PATH = ROOT / "datasets" / "suscetibilidade" / "susc_features_by_patch_v1.csv"
SCHEMA_PATH = ROOT / "schemas" / "suscetibilidade" / "susc_features_schema_v1.json"
DIRECTION_PATH = (
    ROOT / "schemas" / "suscetibilidade" / "susc_feature_scientific_direction_v1.json"
)
CARD_SCHEMA_PATH = ROOT / "schemas" / "suscetibilidade" / "susc_feature_card_schema_v1.json"
PROV_MANIFEST = (
    ROOT / "manifests" / "suscetibilidade" / "susc_features_provenance_manifest_v1.csv"
)
AUDIT_CSV = ROOT / "manifests" / "suscetibilidade" / "susc_feature_provenance_audit_v1.csv"
DECISION_CSV = (
    ROOT / "outputs_public" / "suscetibilidade" / "SUSC_04_score_v6_feature_decision_matrix.csv"
)

OUT_DIR = ROOT / "outputs_public" / "suscetibilidade" / "feature_cards"
OUT_JSON = OUT_DIR / "susc_feature_cards_v1.json"
OUT_CATALOG = OUT_DIR / "SUSC_05_feature_cards_catalog.md"
OUT_SUMMARY = OUT_DIR / "SUSC_05_feature_cards_summary.csv"

# Measured features that MUST have a card (real names + requested aliases).
OBLIGATORY = [
    "elevation_mean", "elevation_std", "slope_mean", "slope_std",
    "hand_mean", "twi_mean", "tpi_250m_mean", "curvature_laplacian_mean",
    "distance_to_water_mean", "water_occurrence_patch",
    "flow_acc_log_mean", "flow_acc_log_p75",
    "chirps_3d_mm", "chirps_7d_mm", "chirps_30d_mm",
    "chirps_3d_to_30d_ratio", "chirps_7d_to_30d_ratio",
    "rain_3d_7d_ratio", "rain_7d_30d_ratio", "rain_persistence_index",
    "runoff_context_7d", "runoff_context_30d", "runoff_score",
    "s1_vv_mean_clean", "s1_vh_mean_clean", "s1_vv_minus_vh_mean_clean",
    "ndvi_mean", "mndwi_mean", "ndbi_mean",
    "urban_prop", "vegetation_prop",
    "urban_water_interaction", "urban_drainage_interaction",
]

# Per-feature formula/derivation. Where the in-repo computation is design-only or
# external, the derivation states the standard definition and flags audit.
FORMULA = {
    "elevation_mean": "Media da elevacao (DEM) sobre os pixels do patch.",
    "elevation_std": "Desvio-padrao da elevacao (DEM) dentro do patch.",
    "slope_mean": "Media da declividade derivada do DEM (gradiente do terreno).",
    "slope_std": "Desvio-padrao da declividade dentro do patch.",
    "hand_mean": "Height Above Nearest Drainage: altura vertical de cada pixel acima do nivel da drenagem mais proxima; media no patch. Spec design_only em features_topo_hydro.py.",
    "twi_mean": "Topographic Wetness Index: ln(a / tan(b)), a=area de contribuicao, b=declividade; media no patch. Spec design_only.",
    "tpi_250m_mean": "Topographic Position Index: elevacao do pixel menos media da vizinhanca (janela 250m); media no patch.",
    "curvature_laplacian_mean": "Curvatura laplaciana: segunda derivada da superficie do DEM; media no patch (negativo=concavo).",
    "distance_to_water_mean": "Distancia euclidiana de cada pixel a hidrografia/drenagem mais proxima; media no patch.",
    "water_occurrence_patch": "Fracao da area do patch com ocorrencia historica de agua superficial (provavel JRC Global Surface Water).",
    "flow_acc_log_mean": "Log da acumulacao de fluxo (roteamento do DEM); media no patch.",
    "flow_acc_log_p75": "Percentil 75 do log da acumulacao de fluxo no patch. Nome solicitado 'flow_acc_p75'.",
    "chirps_3d_mm": "Precipitacao acumulada CHIRPS em 3 dias relativos a reference_date.",
    "chirps_7d_mm": "Precipitacao acumulada CHIRPS em 7 dias.",
    "chirps_30d_mm": "Precipitacao acumulada CHIRPS em 30 dias.",
    "chirps_3d_to_30d_ratio": "Razao solicitada CHIRPS 3d/30d. AUSENTE da matriz SUSC-03; real mais proxima rain_3d_7d_ratio (definicao diferente).",
    "chirps_7d_to_30d_ratio": "Razao solicitada CHIRPS 7d/30d. AUSENTE da matriz; real mais proxima rain_7d_30d_ratio.",
    "rain_3d_7d_ratio": "Razao chuva 3 dias / 7 dias (concentracao recente).",
    "rain_7d_30d_ratio": "Razao chuva 7 dias / 30 dias (recente vs antecedente).",
    "rain_persistence_index": "Indice composto de persistencia da chuva ao longo do tempo. Nome solicitado 'rain_persistence'. Formula a auditar.",
    "runoff_context_7d": "Contexto de escoamento de 7 dias combinando chuva e terreno. Formula composta a auditar.",
    "runoff_context_30d": "Contexto de escoamento de 30 dias. Formula composta a auditar.",
    "runoff_score": "Nome solicitado inexistente na matriz; reais sao runoff_context_7d/30d.",
    "s1_vv_mean_clean": "Media do retroespalhamento Sentinel-1 VV (limpo) no patch. Metodo de limpeza/pareamento temporal a auditar.",
    "s1_vh_mean_clean": "Media do retroespalhamento Sentinel-1 VH (limpo) no patch.",
    "s1_vv_minus_vh_mean_clean": "Diferenca de polarizacao VV-VH. Nome solicitado 's1_vv_minus_vh'.",
    "ndvi_mean": "NDVI = (NIR-RED)/(NIR+RED); media no patch. Formula confirmada em features_optical.py ndvi_spec().",
    "mndwi_mean": "MNDWI = (GREEN-SWIR)/(GREEN+SWIR); media. Confirmada em features_optical.py mndwi_spec().",
    "ndbi_mean": "NDBI = (SWIR-NIR)/(SWIR+NIR); media. Confirmada em features_optical.py ndbi_spec().",
    "urban_prop": "Fracao do patch classificada como urbano/construido. Fonte publica indeterminada (MapBiomas vs classificacao GEE).",
    "vegetation_prop": "Fracao do patch com cobertura vegetal. Mesma fonte indeterminada de urban_prop.",
    "urban_water_interaction": "Termo composto v5: proporcao urbana combinada com proximidade/ocorrencia de agua. Formula a auditar.",
    "urban_drainage_interaction": "Termo composto v5: proporcao urbana combinada com proximidade a drenagem. Formula a auditar.",
}

# Group-level relation strings (SPGAM / Baixo Jaguaribe / DINO).
SPGAM_REL = {
    "topography": "Condicionante direto no SPGAM (declividade, elevacao, orientacao de vertentes).",
    "topography_hydrology": "Condicionante hidro-topografico alinhado ao SPGAM (proximidade vertical/horizontal a drenagem).",
    "hydrology": "Condicionante hidrologico (distancia a drenagem, fluxo) alinhado ao SPGAM.",
    "land_use": "Impermeabilizacao/uso do solo: NDBI/BU e cobertura vegetal sao condicionantes no SPGAM.",
    "spectral_index": "Indices espectrais (NDBI, NDVI) usados como condicionantes no SPGAM.",
    "precipitation": "Gatilho pluviometrico; o SPGAM foca condicionantes estaticos, a chuva entra como modulador temporal.",
    "sar": "Nao usado no SPGAM classico (que e optico/topografico).",
    "interaction": "Termo composto que combina condicionantes do SPGAM (urbanizacao x agua/drenagem).",
}
JAGUARIBE_REL = {
    "topography": "Contexto de relevo complementar ao mapeamento de areas inundaveis.",
    "topography_hydrology": "Apoia a delimitacao de areas baixas/inundaveis do estudo.",
    "hydrology": "Proximidade a agua/drenagem alinhada a logica cheia/seca do estudo.",
    "land_use": "Uso/cobertura do solo e variavel do estudo do Baixo Jaguaribe.",
    "spectral_index": "Sentinel-2 (indices) usado para validacao espectral no estudo.",
    "precipitation": "Precipitacao/cota fluviometrica como contexto hidrologico do estudo.",
    "sar": "Nucleo do metodo: retroespalhamento Sentinel-1, limiar cheia/seca, cubo multitemporal.",
    "interaction": "Combina uso do solo e agua, ambos presentes no estudo.",
}
DINO_REL = (
    "DINOv2 compara padroes visuais/espaciais latentes como camada complementar de "
    "generalizacao espacial; NAO substitui esta metrica fisica, NAO e detector e NAO "
    "e ground truth."
)


def load_csv(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    print("=" * 60)
    print("SUSC-05 Feature Card Builder")
    print("=" * 60)

    for p in (CSV_PATH, SCHEMA_PATH, DIRECTION_PATH, PROV_MANIFEST, AUDIT_CSV, DECISION_CSV):
        if not p.exists():
            print(f"STOP: required input not found: {p}")
            return 1

    with SCHEMA_PATH.open("r", encoding="utf-8") as f:
        schema = json.load(f)
    schema_by_col = {f["source_column"]: f for f in schema["features"]}

    with DIRECTION_PATH.open("r", encoding="utf-8") as f:
        direction = json.load(f)
    dir_by_name = {f["feature_name"]: f for f in direction["features"]}

    prov_by_col = {r["source_column"]: r for r in load_csv(PROV_MANIFEST)}
    audit_by_name = {r["feature_name"]: r for r in load_csv(AUDIT_CSV)}
    decision_by_name = {r["feature_name"]: r for r in load_csv(DECISION_CSV)}

    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))
    present_cols = set(header)

    def b(x):
        return str(x).strip().lower() == "true"

    cards = []

    def make_measured_card(name):
        meta = dir_by_name.get(name, {})
        audit = audit_by_name.get(name, {})
        dec = decision_by_name.get(name, {})
        sch = schema_by_col.get(name, {})
        prov = prov_by_col.get(name, {})
        group = meta.get("scientific_group") or sch.get("group", "unknown")
        exists = name in present_cols
        status_prov = audit.get("provenance_status", "unresolved")

        safe_v6 = b(audit.get("safe_for_score_v6", "false"))
        safe_spgam = b(audit.get("safe_for_spgam_baseline", "false"))
        safe_dino = b(audit.get("safe_for_dino_comparison", "false"))
        requires_manual = b(audit.get("requires_manual_review", "true"))
        direction_val = meta.get("expected_direction_for_flood_susceptibility", "not_scored")

        if not exists:
            status = "blocked_until_recomputed"
        elif status_prov == "unresolved":
            status = "unresolved"
        elif group == "interaction":
            status = "proxy_only"
        elif safe_v6:
            status = "ready_for_methodology"
        else:
            status = "usable_with_caution"

        source_files = []
        sp = audit.get("script_paths", "")
        if sp:
            source_files = sp.split(";")[:6]
        public_source = audit.get("public_source", "unresolved")
        input_data = [s for s in public_source.replace("documented:", "").split(";") if s and s != "unresolved"]

        limitations = []
        if requires_manual:
            limitations.append("Fonte definitiva por feature requires_manual_review (atribuicao por proximidade no scan SUSC-04).")
        if status_prov == "verified_script_only":
            limitations.append("Fonte publica apenas documentada, nao amarrada por proximidade ao script de computacao.")
        if not exists:
            limitations.append("Coluna AUSENTE da matriz SUSC-03; exige recomputacao ou uso da coluna real equivalente.")
        if direction_val in {"ambiguous", "non_monotonic"}:
            limitations.append("Direcao nao-monotonica/ambigua; nao usar com peso alto.")
        if group == "interaction":
            limitations.append("Termo composto v5; auditar formula antes de pesar.")
        if prov and str(prov.get("public_source_known", "")).strip().lower() != "true":
            limitations.append("Manifesto de proveniencia: public_source_known=false (fonte publica nao confirmada).")
        note = sch.get("notes") or meta.get("notes")
        if note:
            limitations.append(note)

        card = {
            "feature_name": name,
            "feature_group": group,
            "concept": meta.get("concept") or sch.get("interpretation", name),
            "formula_or_derivation": FORMULA.get(name, "Derivacao a documentar."),
            "input_data": input_data or ["unresolved"],
            "source_files": source_files,
            "source_columns": [name] if exists else [],
            "unit": meta.get("unit_expected") or sch.get("unit") or "unknown",
            "spatial_resolution": audit.get("crs_or_resolution", "unresolved"),
            "temporal_reference": audit.get("temporal_reference", "unresolved"),
            "expected_direction": direction_val,
            "scientific_rationale": meta.get("expected_reason", "A documentar."),
            "relation_to_flood_susceptibility": _susc_relation(direction_val),
            "relation_to_spgam_study": SPGAM_REL.get(group, "Relacao a documentar."),
            "relation_to_baixo_jaguaribe_study": JAGUARIBE_REL.get(group, "Relacao a documentar."),
            "relation_to_dino": DINO_REL,
            "known_limitations": limitations or ["Nenhuma limitacao adicional registrada."],
            "safe_for_score_v6": safe_v6,
            "safe_for_spgam_baseline": safe_spgam,
            "safe_for_dino_comparison": safe_dino,
            "can_be_ground_truth": False,
            "allowed_for_training": False,
            "review_only": True,
            "requires_manual_review": requires_manual,
            "status": status,
            "role": dec.get("role", "exclude_until_audited"),
            "weight_class": dec.get("weight_class", "blocked"),
        }
        return card

    def make_heuristic_card(sch):
        name = sch["source_column"]
        group = sch["group"]
        kind = {
            "score": "Score composto heuristico v5 (NAO ground truth, NAO validado contra evento).",
            "heuristic_label": "Label heuristico derivado de score por limiar (NAO ground truth).",
            "proxy_v5": "Flag binaria proxy do sistema de score v5 (heuristica, NAO ground truth).",
        }[group]
        exists = name in present_cols
        return {
            "feature_name": name,
            "feature_group": group,
            "concept": sch.get("interpretation", name),
            "formula_or_derivation": kind,
            "input_data": ["score_v5_system"],
            "source_files": [],
            "source_columns": [name] if exists else [],
            "unit": sch.get("unit") or "unknown",
            "spatial_resolution": "unresolved",
            "temporal_reference": "reference_date_2022-12-31",
            "expected_direction": "not_scored",
            "scientific_rationale": "Heuristica derivada de condicionantes; descreve predisposicao modelada, nao ocorrencia.",
            "relation_to_flood_susceptibility": "Resultado heuristico de suscetibilidade; NUNCA evidencia de evento observado.",
            "relation_to_spgam_study": "Analogo ao output de probabilidade do SPGAM, porem heuristico e NAO calibrado/validado.",
            "relation_to_baixo_jaguaribe_study": "Sem equivalente direto; o estudo valida por evidencia hidrologica, ausente aqui.",
            "relation_to_dino": DINO_REL,
            "known_limitations": [
                "Heuristico, NAO ground truth.",
                "Derivado de scores/limiar regional (p75), nao de evento observado.",
                sch.get("notes", ""),
            ],
            "safe_for_score_v6": False,
            "safe_for_spgam_baseline": False,
            "safe_for_dino_comparison": False,
            "can_be_ground_truth": False,
            "allowed_for_training": False,
            "review_only": True,
            "requires_manual_review": True,
            "status": "proxy_only",
            "role": "proxy_only",
            "weight_class": "not_weighted",
        }

    # 1) measured/obligatory cards
    for name in OBLIGATORY:
        cards.append(make_measured_card(name))

    # 2) heuristic scores / labels / proxies (proxy_only)
    for sch in schema["features"]:
        if sch["group"] in {"score", "heuristic_label", "proxy_v5"}:
            cards.append(make_heuristic_card(sch))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # JSON
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump({
            "schema_id": "SUSC-05-FEATURE-CARDS",
            "card_schema": "schemas/suscetibilidade/susc_feature_card_schema_v1.json",
            "governance": {"allowed_for_training": False, "review_only": True,
                           "can_create_ground_truth": False},
            "n_cards": len(cards),
            "cards": cards,
        }, f, indent=2, ensure_ascii=False)
        f.write("\n")

    # summary CSV
    summ_fields = [
        "feature_name", "feature_group", "status", "expected_direction",
        "safe_for_score_v6", "safe_for_spgam_baseline", "safe_for_dino_comparison",
        "role", "weight_class", "requires_manual_review",
        "can_be_ground_truth", "allowed_for_training", "review_only",
    ]
    with OUT_SUMMARY.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summ_fields, lineterminator="\n")
        w.writeheader()
        for c in cards:
            w.writerow({k: c[k] for k in summ_fields})

    # catalog MD
    lines = [
        "# SUSC-05 — Catalogo de Feature Cards Cientificos",
        "",
        "> O SUSC-05 nao calcula score, nao treina modelo e nao cria ground truth. "
        "Cada card e uma unidade metodologica explicavel. Suscetibilidade != ocorrencia confirmada.",
        "",
        f"Total de cards: **{len(cards)}**.",
        "",
    ]
    by_status = {}
    for c in cards:
        by_status.setdefault(c["status"], []).append(c["feature_name"])
    lines.append("## Resumo por status")
    lines.append("")
    lines.append("| status | nº | features |")
    lines.append("|--------|----|----------|")
    for st in ["ready_for_methodology", "usable_with_caution", "proxy_only",
               "blocked_until_recomputed", "unresolved"]:
        names = by_status.get(st, [])
        shown = ", ".join(names[:12]) + (" ..." if len(names) > 12 else "")
        lines.append(f"| {st} | {len(names)} | {shown} |")
    lines.append("")

    for c in cards:
        lines.append(f"## {c['feature_name']}  `({c['status']})`")
        lines.append("")
        lines.append(f"- **Grupo:** {c['feature_group']} · **Papel:** {c['role']} · **Peso:** {c['weight_class']}")
        lines.append(f"- **Conceito:** {c['concept']}")
        lines.append(f"- **Formula/derivacao:** {c['formula_or_derivation']}")
        lines.append(f"- **Unidade:** {c['unit']} · **Resolucao/CRS:** {c['spatial_resolution']} · **Referencia temporal:** {c['temporal_reference']}")
        lines.append(f"- **Direcao esperada:** {c['expected_direction']}")
        lines.append(f"- **Racional:** {c['scientific_rationale']}")
        lines.append(f"- **Relacao com suscetibilidade:** {c['relation_to_flood_susceptibility']}")
        lines.append(f"- **SPGAM/INPE:** {c['relation_to_spgam_study']}")
        lines.append(f"- **Baixo Jaguaribe/UFC:** {c['relation_to_baixo_jaguaribe_study']}")
        lines.append(f"- **DINO:** {c['relation_to_dino']}")
        lines.append(f"- **Fontes (candidatas):** {', '.join(c['input_data'])}")
        lines.append(f"- **Limitacoes:** {' | '.join(l for l in c['known_limitations'] if l)}")
        lines.append(
            f"- **Governanca:** score_v6={c['safe_for_score_v6']} · spgam={c['safe_for_spgam_baseline']} · "
            f"dino={c['safe_for_dino_comparison']} · ground_truth={c['can_be_ground_truth']} · "
            f"training={c['allowed_for_training']} · review_only={c['review_only']} · "
            f"requires_manual_review={c['requires_manual_review']}"
        )
        lines.append("")
    OUT_CATALOG.write_text("\n".join(lines), encoding="utf-8")

    # summary print
    from collections import Counter
    print(f"\n[ok] {len(cards)} cards written")
    print("Status:", dict(Counter(c["status"] for c in cards)))
    print(f"safe_for_score_v6: {sum(1 for c in cards if c['safe_for_score_v6'])}")
    print(f"safe_for_spgam_baseline: {sum(1 for c in cards if c['safe_for_spgam_baseline'])}")
    print(f"safe_for_dino_comparison: {sum(1 for c in cards if c['safe_for_dino_comparison'])}")
    # governance invariant
    bad = [c["feature_name"] for c in cards if c["can_be_ground_truth"] or c["allowed_for_training"] or not c["review_only"]]
    print("governance violations:", bad or "NONE")
    print("\nNo score computed. No model trained. No ground truth created.")
    return 0


def _susc_relation(direction):
    base = {
        "higher_increases": "Valores mais altos tendem a AUMENTAR a suscetibilidade.",
        "lower_increases": "Valores mais baixos tendem a AUMENTAR a suscetibilidade.",
        "higher_decreases": "Valores mais altos tendem a REDUZIR a suscetibilidade.",
        "lower_decreases": "Valores mais baixos tendem a REDUZIR a suscetibilidade.",
        "ambiguous": "Evidencia complementar sem direcao monotonica; nao isolar como condicionante.",
        "non_monotonic": "Relacao nao-monotonica com a suscetibilidade.",
        "not_scored": "Nao pontuado como condicionante direto.",
    }.get(direction, "Relacao a documentar.")
    return base


if __name__ == "__main__":
    sys.exit(main())
