"""SUSC-20R -- correlacao entre indice ONI real (ENOS) e o colapso de AUC prospectivo 2026."""
from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[4] / (
    "outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/results")

# Valores ONI reais (JFM..MJJ), extraidos da tabela ggweather.com/enso/oni.htm em 2026-08-02.
# Linha do ano X-(X+1) da fonte cobre JJA(X) ate MJJ(X+1) -- os meses JFM..MJJ de um ano
# civil Y vem da linha rotulada "(Y-1)-Y".
ONI_JAN_JUL = {
    # linha fonte "2022-2023": JFM=-0.29 FMA=-0.02 MAM=0.27 AMJ=0.57 MJJ=0.84
    2023: {"JFM": -0.29, "FMA": -0.02, "MAM": 0.27, "AMJ": 0.57, "MJJ": 0.84},
    # linha fonte "2023-2024": JFM=1.62 FMA=1.26 MAM=0.82 AMJ=0.49 MJJ=0.22
    2024: {"JFM": 1.62, "FMA": 1.26, "MAM": 0.82, "AMJ": 0.49, "MJJ": 0.22},
    # linha fonte "2024-2025": JFM=-0.24 FMA=-0.06 MAM=0.02 AMJ=-0.02 MJJ=-0.04
    2025: {"JFM": -0.24, "FMA": -0.06, "MAM": 0.02, "AMJ": -0.02, "MJJ": -0.04},
    # 2026: NAO PUBLICADO na fonte consultada (tabela vai so ate OND/2025=-0.55) -- vazio de
    # proposito, nao estimado.
    2026: {},
}

# Referencia (SUSC-20Q, dado real ja commitado): AUC de teste do walk-forward multi-corte e
# magnitude/sinal do coeficiente Firth de chuva por ano (rain_decay_index_api_chirps).
WALK_FORWARD_TEST_AUC = {2024: 0.6282, 2025: 0.6652, 2026: 0.5246}
RAIN_DECAY_COEF_BY_YEAR = {2023: 0.5881, 2024: 0.4349, 2025: 0.6312, 2026: 0.0320}
RAIN_DECAY_P_BY_YEAR = {2023: 0.0002, 2024: 0.0022, 2025: 0.0000, 2026: 0.7948}


def mean_oni_jan_jul(year: int) -> float | None:
    vals = ONI_JAN_JUL.get(year, {})
    if not vals:
        return None
    return round(sum(vals.values()) / len(vals), 4)


def build_comparison_table() -> list[dict]:
    rows = []
    for year in [2023, 2024, 2025, 2026]:
        rows.append({
            "year": year,
            "oni_jan_jul_mean": mean_oni_jan_jul(year),
            "oni_publicado": mean_oni_jan_jul(year) is not None,
            "walk_forward_test_auc": WALK_FORWARD_TEST_AUC.get(year),
            "rain_decay_coef": RAIN_DECAY_COEF_BY_YEAR.get(year),
            "rain_decay_p": RAIN_DECAY_P_BY_YEAR.get(year),
        })
    return rows


def main() -> int:
    out = RESULTS
    out.mkdir(parents=True, exist_ok=True)
    rows = build_comparison_table()
    for r in rows:
        print(json.dumps(r))

    # leitura descritiva -- n=2 anos com ONI publicado E auc de teste (2024, 2025); 2026 sem
    # ONI publicado. Sem teste de correlacao formal (n insuficiente pra inferencia).
    leitura = {
        "n_anos_com_oni_e_auc_teste": sum(
            1 for r in rows if r["oni_publicado"] and r["walk_forward_test_auc"] is not None),
        "ano_com_oni_mais_extremo_jan_jul": max(
            (r for r in rows if r["oni_publicado"]), key=lambda r: abs(r["oni_jan_jul_mean"]))["year"],
        "esse_ano_generalizou_bem": None,
        "oni_2026_publicado": mean_oni_jan_jul(2026) is not None,
        "nota": ("2024 teve o ONI jan-jul mais extremo (El Nino forte, media ~0,88) dos anos "
                 "com dado publicado, e generalizou bem (AUC teste=0,6282). 2025 foi "
                 "ENOS-neutro (media ~-0,07) e tambem generalizou bem (AUC teste=0,6652). "
                 "2026 nao tem ONI jan-jul publicado na fonte consultada -- a hipotese de "
                 "anomalia ENOS explicando o colapso de 2026 fica SEM SUPORTE nos dados "
                 "disponiveis: o ano com maior anomalia ENOS real (2024) generalizou melhor "
                 "que o ano ENOS-neutro seguinte (2025 generalizou ainda melhor), o que vai "
                 "na direcao contraria de 'anomalia ENOS = colapso'."),
    }
    ano_extremo = leitura["ano_com_oni_mais_extremo_jan_jul"]
    leitura["esse_ano_generalizou_bem"] = bool(
        WALK_FORWARD_TEST_AUC.get(ano_extremo, 0) >= 0.60)
    print(json.dumps(leitura, indent=2, ensure_ascii=False))

    report = {"fonte": "https://ggweather.com/enso/oni.htm (ONI v5 NOAA/CPC, consultado 2026-08-02)",
              "comparacao": rows, "leitura": leitura}
    (out / "v20r_oni_enso_correlation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nresultados -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
