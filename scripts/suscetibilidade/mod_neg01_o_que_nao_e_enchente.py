"""
MOD-NEG-01 -- O que NAO e enchente: comparacao entre duas definicoes de negativo.

A PERGUNTA CIENTIFICA:
o REV-P nasceu sem negativo formal. Todo negativo do Protocolo C era construido
por AUSENCIA de registro -- "nao ha noticia de enchente aqui, logo aqui nao
inundou". A frente externa trouxe, pela primeira vez, negativo por OBSERVACAO:
area que um analista declarou ter examinado e onde nao detectou inundacao.

A pergunta que este script responde e: **as duas definicoes ensinam a mesma
coisa ao modelo?** Se ensinam, a fragilidade do Protocolo C era teorica. Se nao
ensinam, a diferenca e mensuravel e precisa aparecer no artigo.

POR QUE O DESENHO INGENUO NAO SERVE:
comparar os dois tipos de negativo contra os MESMOS positivos e impossivel --
eles vem de regioes diferentes (Inglaterra x ativacoes CEMS mundiais).
Qualquer diferenca observada seria confundida com geografia.

O DESENHO QUE RESPONDE, em tres partes:

  A. CARACTERIZACAO. Como cada tipo de negativo se distribui no espaco de
     features? Se o negativo por exclusao for sistematicamente mais alto e
     mais seco que o observado, o vies esta quantificado.

  B. DOIS MODELOS INDEPENDENTES. Um treinado so na Inglaterra (negativo por
     exclusao, agrupado por evento) e outro so no CEMS (negativo por
     observacao, agrupado por AOI). Se os coeficientes concordarem em SINAL,
     a fisica e a mesma. Se discordarem, uma das definicoes esta ensinando
     outra coisa.

  C. APLICACAO CRUZADA. Cada modelo avaliado sobre o dado do outro. E o teste
     mais duro: um modelo que so aprendeu "onde nao ha registro" deve degradar
     ao encontrar "onde foi olhado e estava seco".

O QUE ISSO ENSINA -- e o motivo de o script existir:
o valor nao esta em qual modelo tem AUC maior. Esta em descobrir se a definicao
de NAO-ENCHENTE muda o que o modelo entende por enchente. Um classificador so
sabe o que e um evento na medida em que sabe o que nao e.

LIMITACAO DECLARADA: `twi_dinf` nao entra. As tabelas CEMS sairam com TWI
inteiramente NaN (o `pysheds` nao estava no ambiente do projeto; corrigido em
09/08/2026, mas as tabelas ainda nao foram refeitas). Rodar com 3 features e
uma escolha consciente de comparabilidade -- as duas fontes precisam ter as
MESMAS colunas, senao a comparacao mede feature faltando, nao definicao de
negativo.

NAO faz: nao ajusta hiperparametro, nao seleciona feature por desempenho, nao
declara vencedor.

Uso:
    python scripts/suscetibilidade/mod_neg01_o_que_nao_e_enchente.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
import susc_firth_shim  # noqa: F401,E402

REPO = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
RUNS = REPO / "local_runs"
ENTRADA = RUNS / "ds-01-multirregiao" / "dataset_multirregiao_v1.csv"
OUT = RUNS / "mod-neg-01"

# 3 features: as unicas com cobertura nas DUAS fontes. Ver limitacao no topo.
FEATURES = ["elevation_m", "slope_deg", "hand_m"]
N_FOLDS = 5
N_BOOT = 400
SEMENTE = 20260809


def alinhar(v, n: int):
    v = np.asarray(v, dtype=float).ravel()
    return v if v.size == n else (v[-n:] if v.size > n else np.full(n, np.nan))


def ajustar(X, y):
    from firthlogist import FirthLogisticRegression

    m = FirthLogisticRegression(max_iter=500, skip_pvals=True, skip_ci=True)
    m.fit(X, y)
    return m


def main() -> int:
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEMENTE)
    t0 = time.time()
    print("===MOD-NEG-01: O QUE NAO E ENCHENTE===")

    df = pd.read_csv(ENTRADA)
    df = df[df["classe"].isin([0, 1])].dropna(subset=FEATURES)

    # separacao por procedencia do negativo
    exc = df[(df.tipo_negativo == "por_exclusao_qualificada")
             | ((df.classe == 1) & (df.regiao == "UK_noroeste"))]
    obs = df[(df.tipo_negativo == "por_observacao")
             | ((df.classe == 1) & (df.regiao.str.startswith("analogo_")))]
    obs = obs[obs.regiao.str.startswith("analogo_") | (obs.tipo_negativo == "por_observacao")]
    obs = obs[obs[FEATURES].notna().all(axis=1)]

    print(f"EXCLUSAO  n={len(exc):,} pos={int((exc.classe==1).sum()):,} "
          f"neg={int((exc.classe==0).sum()):,} grupos={exc.grupo_cv.nunique()}")
    print(f"OBSERVACAO n={len(obs):,} pos={int((obs.classe==1).sum()):,} "
          f"neg={int((obs.classe==0).sum()):,} grupos={obs.grupo_cv.nunique()}")
    if len(obs) < 100 or (obs.classe == 0).sum() < 30:
        print("ABORTADO: dado de observacao insuficiente para comparar")
        return 2

    # TRAVA DE EPV -- mesma regra do mod_uk01. Ate 09/08/2026 este script nao a
    # tinha, e rodou com EPV 1,67 (5 grupos / 3 features) produzindo um AUC que
    # media dificuldade de validacao, nao definicao de negativo. Scripts do
    # mesmo projeto nao podem ter criterios diferentes de suficiencia amostral.
    EPV_MINIMO = 10
    for nome, sub in (("exclusao", exc), ("observacao", obs)):
        epv = sub["grupo_cv"].nunique() / len(FEATURES)
        print(f"  EPV_{nome}={epv:.1f} (grupos/features, minimo={EPV_MINIMO})")
        if epv < EPV_MINIMO:
            print(f"ABORTADO: EPV de '{nome}' abaixo do minimo. Aumente o numero "
                  f"de grupos ou reduza features. Nao vou comparar modelos com "
                  f"suficiencia amostral incompativel.")
            return 3

    # ---------------- A. caracterizacao dos dois negativos ----------------
    print("---A. COMO SAO OS DOIS NEGATIVOS---")
    carac = {}
    for nome, sub in (("exclusao", exc[exc.classe == 0]), ("observacao", obs[obs.classe == 0])):
        d = {f: round(float(sub[f].median()), 2) for f in FEATURES}
        carac[nome] = d
        print(f"  negativo por {nome:11s} " + "  ".join(f"{k}={v}" for k, v in d.items()))
    print("  positivos de referencia:")
    for nome, sub in (("UK", exc[exc.classe == 1]), ("CEMS", obs[obs.classe == 1])):
        d = {f: round(float(sub[f].median()), 2) for f in FEATURES}
        carac[f"pos_{nome}"] = d
        print(f"    {nome:11s} " + "  ".join(f"{k}={v}" for k, v in d.items()))

    # contraste: distancia do negativo ao seu proprio positivo, por feature
    print("---CONTRASTE (negativo menos positivo, na mesma fonte)---")
    contraste = {}
    for nome, sub in (("exclusao", exc), ("observacao", obs)):
        d = {f: round(float(sub.loc[sub.classe == 0, f].median()
                           - sub.loc[sub.classe == 1, f].median()), 2)
             for f in FEATURES}
        contraste[nome] = d
        print(f"  {nome:11s} " + "  ".join(f"{k}={v:+}" for k, v in d.items()))

    # ---------------- B. dois modelos independentes ----------------
    print("---B. DOIS MODELOS, MESMAS FEATURES---")
    modelos, resultados = {}, {}
    for nome, sub in (("exclusao", exc), ("observacao", obs)):
        y = sub["classe"].to_numpy().astype(int)
        X = sub[FEATURES].to_numpy(dtype=float)
        g = sub["grupo_cv"].to_numpy()
        mu, sd = X.mean(0), X.std(0)
        sd = np.where(sd == 0, 1, sd)
        Xz = (X - mu) / sd

        n_g = len(np.unique(g))
        folds = min(N_FOLDS, n_g)
        aucs = []
        if folds >= 2:
            for tr, te in GroupKFold(n_splits=folds).split(Xz, y, groups=g):
                if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
                    continue
                p = ajustar(Xz[tr], y[tr]).predict_proba(Xz[te])[:, 1]
                aucs.append(roc_auc_score(y[te], p))
        m = ajustar(Xz, y)
        coef = alinhar(m.coef_, len(FEATURES))
        modelos[nome] = {"modelo": m, "mu": mu, "sd": sd}
        auc = float(np.mean(aucs)) if aucs else float("nan")
        resultados[nome] = {"auc_cv": auc, "folds": len(aucs), "grupos": int(n_g),
                            "coef": {f: round(float(c), 4)
                                     for f, c in zip(FEATURES, coef)}}
        print(f"  [{nome}] AUC_CV_agrupada={auc:.4f} ({len(aucs)} folds, "
              f"{n_g} grupos)")
        for f, c in zip(FEATURES, coef):
            print(f"      {f:14s} {c:+.4f}")

    # concordancia de sinal -- a pergunta fisica
    sinais = {f: (np.sign(resultados["exclusao"]["coef"][f])
                  == np.sign(resultados["observacao"]["coef"][f]))
              for f in FEATURES}
    concordam = sum(sinais.values())
    print(f"  CONCORDANCIA_DE_SINAL={concordam}/{len(FEATURES)} " +
          "; ".join(f"{f}:{'=' if s else 'X'}" for f, s in sinais.items()))

    # ---------------- C. aplicacao cruzada ----------------
    print("---C. CADA MODELO SOBRE O DADO DO OUTRO---")
    cruzado = {}
    for treino, teste, sub_teste in (("exclusao", "observacao", obs),
                                     ("observacao", "exclusao", exc)):
        info = modelos[treino]
        Xz = (sub_teste[FEATURES].to_numpy(dtype=float) - info["mu"]) / info["sd"]
        y = sub_teste["classe"].to_numpy().astype(int)
        p = info["modelo"].predict_proba(Xz)[:, 1]
        auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
        proprio = resultados[treino]["auc_cv"]
        queda = proprio - auc
        cruzado[f"{treino}_em_{teste}"] = {"auc": round(float(auc), 4),
                                           "auc_proprio": round(proprio, 4),
                                           "queda": round(float(queda), 4)}
        print(f"  treinado em {treino:11s} -> avaliado em {teste:11s} "
              f"AUC={auc:.4f}  (proprio={proprio:.4f}, queda={queda:+.4f})")

    (OUT / "resumo.json").write_text(json.dumps({
        "pergunta": ("As duas definicoes de negativo -- por exclusao e por "
                     "observacao -- ensinam a mesma coisa ao modelo?"),
        "features": FEATURES,
        "limitacao": ("twi_dinf excluida: tabelas CEMS sairam com TWI NaN por "
                      "ausencia do pysheds no ambiente. As duas fontes precisam "
                      "das MESMAS colunas para a comparacao medir definicao de "
                      "negativo, nao feature faltando."),
        "caracterizacao": carac,
        "contraste_neg_menos_pos": contraste,
        "modelos": {k: {kk: vv for kk, vv in v.items()} for k, v in resultados.items()},
        "concordancia_de_sinal": {f: bool(s) for f, s in sinais.items()},
        "aplicacao_cruzada": cruzado,
        "aviso": ("Nenhum modelo aqui e o modelo do artigo. O objetivo e medir "
                  "o efeito da DEFINICAO de negativo, nao maximizar desempenho."),
        "segundos": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
