"""
SUSC-13B-AUTO observational readiness assessment (review-only).

Counts strong/moderate events, temporal/geometry coverage and strong/moderate
patch links from the consolidated catalog and linkage, then reports whether the
observational layer is ready for SUSC-12A/12B/12C re-evaluation and whether a
score v7 would be justified. It NEVER creates score v7 — readiness only.

Writes:
  - outputs_public/suscetibilidade/SUSC_13B_auto_observational_readiness.csv
  - outputs_public/suscetibilidade/SUSC_13B_auto_observational_readiness.md
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_io import read_csv, write_csv, write_markdown  # noqa: E402

CATALOG = ROOT / "datasets" / "suscetibilidade" / "susc_13b_auto_consolidated_observed_event_catalog_v1.csv"
LINKAGE = ROOT / "datasets" / "suscetibilidade" / "susc_13b_auto_event_patch_linkage_v1.csv"
READINESS_CSV = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_13B_auto_observational_readiness.csv"
READINESS_MD = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_13B_auto_observational_readiness.md"

STRONG = {"strong_observed_flood_polygon", "strong_observed_flood_point"}
MODERATE = {"moderate_official_occurrence_point", "moderate_official_flood_bbox"}
OBSERVED = STRONG | MODERATE


def _has_geom(e: dict) -> bool:
    return bool(e.get("bbox") or (e.get("lat") and e.get("lon")) or e.get("wkt") or e.get("geojson_ref"))


def _has_date(e: dict) -> bool:
    return bool(e.get("event_date") or e.get("event_period_start"))


def main() -> int:
    print("=" * 60)
    print("SUSC-13B-AUTO Observational Readiness")
    print("=" * 60)
    events = read_csv(CATALOG) if CATALOG.exists() else []
    links = read_csv(LINKAGE) if LINKAGE.exists() else []
    # only preferred (non-duplicate) records count toward readiness
    pref = [e for e in events if e.get("preferred_record", "true") != "false"]

    strong_events = sum(1 for e in pref if e["evidence_level"] in STRONG)
    moderate_events = sum(1 for e in pref if e["evidence_level"] in MODERATE)
    observed_events = strong_events + moderate_events
    events_with_date = sum(1 for e in pref if e["evidence_level"] in OBSERVED and _has_date(e))
    events_with_geometry = sum(1 for e in pref if e["evidence_level"] in OBSERVED and _has_geom(e))

    strong_links = sum(1 for r in links if r.get("linkage_confidence") == "strong")
    moderate_links = sum(1 for r in links if r.get("linkage_confidence") == "moderate")
    strong_mod_links = strong_links + moderate_links
    eval_links = sum(1 for r in links if r.get("can_be_used_for_observational_evaluation") == "true")
    regions_covered = sorted({e["region"] for e in pref if e["evidence_level"] in OBSERVED and _has_geom(e)})

    ready_12a = observed_events >= 10 and strong_mod_links >= 10 and events_with_date >= 10
    ready_12b = strong_mod_links >= 10 and len(regions_covered) >= 1
    ready_12c = strong_mod_links >= 20 and len(regions_covered) >= 2
    # score v7 readiness is the strictest gate; it stays a recommendation only.
    single_source_dominant = False
    if pref:
        from collections import Counter
        src = Counter(e.get("source_institution", "") for e in pref if e["evidence_level"] in OBSERVED)
        if src:
            top = max(src.values())
            single_source_dominant = top >= 0.8 * sum(src.values()) and sum(src.values()) > 0
    ready_v7 = (strong_mod_links >= 20 and events_with_geometry >= 20 and
                events_with_date >= 20 and len(regions_covered) >= 2 and not single_source_dominant)

    metrics = [
        ("strong_events_count", strong_events),
        ("moderate_events_count", moderate_events),
        ("observed_events_count", observed_events),
        ("events_with_date_count", events_with_date),
        ("events_with_geometry_count", events_with_geometry),
        ("strong_patch_links_count", strong_links),
        ("moderate_patch_links_count", moderate_links),
        ("strong_or_moderate_patch_links_count", strong_mod_links),
        ("observational_evaluation_links_count", eval_links),
        ("regions_covered", "|".join(regions_covered) if regions_covered else "none"),
        ("regions_covered_count", len(regions_covered)),
        ("single_source_dominant", str(single_source_dominant).lower()),
        ("ready_for_12a_temporal", str(ready_12a).lower()),
        ("ready_for_12b_feature_contrast", str(ready_12b).lower()),
        ("ready_for_12c_proxy_calibration", str(ready_12c).lower()),
        ("ready_for_score_v7", str(ready_v7).lower()),
    ]
    rows = [{"metric": k, "value": v, "review_only": "true"} for k, v in metrics]
    write_csv(READINESS_CSV, rows, ["metric", "value", "review_only"])

    def yn(b):
        return "PRONTO" if b else "BLOQUEADO"

    md = f"""# SUSC-13B-AUTO - Prontidao observacional

Status: **review-only** | `can_be_ground_truth=false` | `allowed_for_training=false`

## Contagens (apenas registros preferenciais, sem duplicatas)
- Eventos fortes: **{strong_events}**
- Eventos moderados: **{moderate_events}**
- Eventos observados (forte+moderado): **{observed_events}**
- Eventos observados com data/periodo: **{events_with_date}**
- Eventos observados com geometria: **{events_with_geometry}**
- Links patch-evento fortes: **{strong_links}**
- Links patch-evento moderados: **{moderate_links}**
- Links fortes+moderados: **{strong_mod_links}**
- Links permitidos para avaliacao observacional: **{eval_links}**
- Regioes cobertas: **{", ".join(regions_covered) if regions_covered else "nenhuma"}**

## Prontidao
- SUSC-12A (temporal): **{yn(ready_12a)}** — exige >=10 eventos com data e >=10 links fortes/moderados.
- SUSC-12B (contraste de features): **{yn(ready_12b)}** — exige >=10 patches fortes/moderados e controles na regiao.
- SUSC-12C (calibracao de proxy): **{yn(ready_12c)}** — exige >=20 patches fortes/moderados e >=2 regioes.
- Score v7: **{yn(ready_v7)}** — exige >=20 patches, data/geometria suficientes, 2 regioes e evidencia nao concentrada numa unica fonte fraca.

## Conclusao
{"A camada observacional automatica ainda nao atinge os limiares minimos; o gargalo segue sendo a ausencia de fonte oficial com data + geometria explicita acessivel automaticamente (offline e o modo padrao)." if not ready_12a else "Limiar de 12A atingido; revisar manualmente antes de qualquer reexecucao."}

Esta etapa **nao cria score v7** e nao altera a matriz SUSC-03. Mesmo se PRONTO,
qualquer avanco exige revisao humana e permanece review-only.
"""
    write_markdown(READINESS_MD, md)

    print(f"observed events: {observed_events} (strong {strong_events}, moderate {moderate_events})")
    print(f"strong/moderate links: {strong_mod_links} | regions: {regions_covered}")
    print(f"ready 12a={ready_12a} 12b={ready_12b} 12c={ready_12c} v7={ready_v7}")
    print("review-only. Score v7 NOT created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
