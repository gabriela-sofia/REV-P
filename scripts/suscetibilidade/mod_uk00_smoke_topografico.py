"""
MOD-UK-00 -- Teste de fumaca do pipeline ingles, so com features topograficas.

*** ISTO NAO E RESULTADO CIENTIFICO. E teste de encanamento. ***

Distincao que precisa ficar explicita, porque confundi-la seria grave:

  MOD-UK-00 (este)  teste de fumaca. Roda o modelo com as 4 features
                    topograficas que ja estao no disco, para provar que a
                    cadeia FT-UK-01 -> 02 -> 03 -> modelo funciona ponta a
                    ponta e que os sinais tem direcao fisica correta.
                    NAO usa as features de chuva. NAO e o modelo do artigo.

  MOD-UK-01         o modelo de verdade: 6 features (as 4 daqui mais as duas
                    de chuva), bootstrap de 1000, e so roda depois de
                    cumpridos os pre-requisitos declarados. Continua bloqueado.

Por que rodar um teste de fumaca faz sentido: um pipeline com sete etapas
(mosaico, reprojecao, derivadas, rasterizacao, distancia, pareamento,
amostragem) pode produzir uma tabela com forma correta e conteudo errado.
Coeficiente com sinal invertido, feature trocada de coluna, grupo vazando
entre folds -- nada disso aparece olhando o CSV. Aparece aqui.

CRITERIOS DE APROVACAO, fixados ANTES de rodar:
  A. AUC agrupada entre 0,60 e 0,95. Abaixo de 0,60 sugere feature quebrada;
     acima de 0,95 sugere vazamento de grupo ou rotulo circular.
  B. `hand_m` com coeficiente NEGATIVO. Quanto mais alto acima da drenagem,
     menor a chance de inundar. Sinal positivo aqui significa erro de sinal
     ou de coluna.
  C. Nenhum fold com AUC = 1,0 exato -- indicaria particao degenerada.
  D. Diferenca entre AUC no treino e na validacao agrupada menor que 0,15.

Falhar qualquer um dos quatro e motivo para investigar o pipeline, nao para
ajustar o modelo.

Uso:
    python scripts/suscetibilidade/mod_uk00_smoke_topografico.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

# shim obrigatorio: firthlogist x scikit-learn >=1.6 (ver susc_firth_shim.py)
import sys as _sys
_sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import susc_firth_shim  # noqa: F401,E402

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
ENTRADA = RUNS / "ft-uk-03-amostra" / "amostra_candidata_uk_v1.csv"
OUT = RUNS / "mod-uk-00-smoke"

FEATURES = ["elevation_m", "slope_deg", "hand_m", "twi_dinf"]
N_FOLDS = 5
SEMENTE = 20260807


def main() -> int:
    from firthlogist import FirthLogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("===MOD-UK-00 (TESTE DE FUMACA, NAO E RESULTADO)===")

    df = pd.read_csv(ENTRADA).dropna(subset=FEATURES + ["label", "grupo_cv"])
    y = df["label"].to_numpy().astype(int)
    X = df[FEATURES].to_numpy(dtype=float)
    grupos = df["grupo_cv"].to_numpy()

    n_eventos = int(df.loc[df.label == 1, "rec_grp_id"].nunique())
    print(f"N_PONTOS={len(df):,} pos={int(y.sum()):,} neg={int((1-y).sum()):,}")
    print(f"N_EVENTOS_INDEPENDENTES={n_eventos} N_GRUPOS_CV={df.grupo_cv.nunique()}")
    print(f"EPV_POR_EVENTO={n_eventos/len(FEATURES):.1f} (features={len(FEATURES)})")

    # padronizacao: sem ela, elevacao em metros e TWI adimensional entram em
    # escalas incomparaveis e o Firth converge devagar
    mu, sd = X.mean(axis=0), X.std(axis=0)
    Xz = (X - mu) / np.where(sd == 0, 1, sd)

    gkf = GroupKFold(n_splits=N_FOLDS)
    aucs_val, aucs_tr = [], []
    pred = np.full(len(y), np.nan)
    for k, (tr, te) in enumerate(gkf.split(Xz, y, groups=grupos), 1):
        m = FirthLogisticRegression(max_iter=500, skip_pvals=True, skip_ci=True)
        m.fit(Xz[tr], y[tr])
        p_te = m.predict_proba(Xz[te])[:, 1]
        p_tr = m.predict_proba(Xz[tr])[:, 1]
        pred[te] = p_te
        a_val = roc_auc_score(y[te], p_te) if len(np.unique(y[te])) > 1 else np.nan
        a_tr = roc_auc_score(y[tr], p_tr)
        aucs_val.append(a_val)
        aucs_tr.append(a_tr)
        print(f"  fold {k}: n_teste={len(te):5d} grupos={len(set(grupos[te])):4d} "
              f"AUC_val={a_val:.4f} AUC_treino={a_tr:.4f}")

    auc_val = float(np.nanmean(aucs_val))
    auc_tr = float(np.nanmean(aucs_tr))
    print(f"AUC_CV_AGRUPADA={auc_val:.4f} (dp={np.nanstd(aucs_val):.4f})")
    print(f"AUC_TREINO_MEDIA={auc_tr:.4f}  gap={auc_tr-auc_val:+.4f}")

    modelo = FirthLogisticRegression(max_iter=500, skip_pvals=False, skip_ci=True)
    modelo.fit(Xz, y)
    # firthlogist devolve pvals_ com o intercepto junto; alinhamos pelo fim
    pv = np.asarray(getattr(modelo, "pvals_", [np.nan] * len(FEATURES)), dtype=float)
    pv = pv[-len(FEATURES):] if pv.size >= len(FEATURES) else np.full(len(FEATURES), np.nan)
    coefs = pd.DataFrame({"feature": FEATURES,
                          "coef_padronizado": np.asarray(modelo.coef_).ravel()[:len(FEATURES)],
                          "p_valor": pv})
    print("---COEFICIENTES (padronizados)---")
    for _, r in coefs.iterrows():
        print(f"  {r.feature:14s} {r.coef_padronizado:+.4f}  p={r.p_valor:.3g}")
    coefs.to_csv(OUT / "coeficientes_smoke.csv", index=False)

    # ------------------------------------------------------------------
    # criterios de aprovacao, verificados na ordem em que foram declarados
    # ------------------------------------------------------------------
    coef_hand = float(coefs.loc[coefs.feature == "hand_m", "coef_padronizado"].iloc[0])
    checagens = {
        "A_auc_na_faixa": bool(0.60 <= auc_val <= 0.95),
        "B_hand_negativo": bool(coef_hand < 0),
        "C_nenhum_fold_perfeito": bool(not any(a >= 0.999 for a in aucs_val if a == a)),
        "D_gap_treino_val_menor_015": bool((auc_tr - auc_val) < 0.15),
    }
    print("---CHECAGENS---")
    for k, v in checagens.items():
        print(f"  {'PASSOU' if v else 'FALHOU'}  {k}")
    aprovado = all(checagens.values())
    print(f"VEREDICTO={'PIPELINE_COERENTE' if aprovado else 'INVESTIGAR_PIPELINE'}")

    df.assign(pred_cv=pred).to_csv(OUT / "predicoes_smoke.csv", index=False)
    (OUT / "resumo.json").write_text(json.dumps({
        "natureza": "TESTE DE FUMACA -- nao e resultado cientifico",
        "features": FEATURES,
        "n_pontos": int(len(df)), "n_eventos": n_eventos,
        "auc_cv_agrupada": round(auc_val, 4),
        "auc_treino": round(auc_tr, 4),
        "auc_por_fold": [round(float(a), 4) for a in aucs_val],
        "coef_hand_m": round(coef_hand, 4),
        "checagens": checagens,
        "aprovado": aprovado,
        "faltando": "duas features de chuva (FT-UK-04, depende do CHIRPS)",
        "segundos": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("===END===")
    return 0 if aprovado else 4


if __name__ == "__main__":
    raise SystemExit(main())
