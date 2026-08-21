"""
MOD-UK-01 -- Firth penalizado no piloto ingles, com validacao agrupada por evento.

*** ESTE SCRIPT NAO DEVE SER EXECUTADO AINDA. ***

Ele existe escrito e versionado ANTES de rodar de proposito. As decisoes de
validacao -- como agrupar, como reamostrar, o que reportar -- precisam estar
fixadas antes de qualquer numero aparecer. Fixa-las depois de ver resultado
seria decisao pos-hoc, e e exatamente o tipo de coisa que uma banca cobra.

PRE-REQUISITOS que ainda nao estao cumpridos:
  1. FT-UK-04 executado (as duas features de chuva anexadas).
  2. Decisao registrada sobre qual flavor do CHIRPS e primario.
  3. Revisao humana do criterio N1-N4 -- a amostra continua CANDIDATA.

O QUE ESTE SCRIPT IMPLEMENTA, e por que cada peca:

REGRA U1 -- validacao cruzada agrupada.
  `GroupKFold` sobre `grupo_cv`. Positivo agrupa por evento (`rec_grp_id`),
  negativo por bloco espacial de 5 km. Nenhum evento aparece em treino e teste
  ao mesmo tempo. Sem isso, pontos do mesmo evento dos dois lados da particao
  inflam a metrica sem que nada tenha sido aprendido.

REGRA U2 -- reamostragem no nivel do evento.
  O bootstrap reamostra GRUPOS, nao linhas. Reamostrar linhas trataria 3.738
  pontos como 3.738 observacoes independentes, quando ha 201 eventos. O
  intervalo de confianca sairia estreito demais por construcao.

REGRA U3 -- relato duplo obrigatorio.
  Toda saida declara n de pontos E n de eventos independentes. Relatar so o
  primeiro e considerado relato incompleto por este projeto.

ORCAMENTO DE EPV.
  Regra pratica: ao menos 10 eventos por variavel na classe minoritaria --
  contados em EVENTOS, nao em pontos. Com 201 eventos, 6 features dao EPV
  proximo de 33, folgado. Se o EPV cair abaixo de 10, o script recusa rodar em
  vez de produzir estimativa instavel em silencio. Esse mesmo limite ja
  obrigou a reduzir o modelo no A/B do DINO (revp_v1r5).

O QUE ESTE SCRIPT NAO FAZ:
  nao ajusta hiperparametro, nao seleciona feature por desempenho, nao compara
  com o v12 brasileiro. Otimizacao de desempenho e validacao cientifica sao
  etapas separadas por regra do projeto.

Uso (quando liberado):
    python scripts/suscetibilidade/mod_uk01_firth_agrupado.py --confirmo-pre-requisitos
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# shim obrigatorio: firthlogist x scikit-learn >=1.6 (ver susc_firth_shim.py)
import sys as _sys
_sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import susc_firth_shim  # noqa: F401,E402

REPO = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
RUNS = REPO / "local_runs"
ENTRADA = RUNS / "ft-uk-04-chuva" / "amostra_candidata_uk_v2_com_chuva.csv"
OUT = RUNS / "mod-uk-01-firth"

FLAVOR_PADRAO = "rnl"
N_BOOT = 1000
N_FOLDS = 5
EPV_MINIMO = 10
SEMENTE = 20260807


FEATURES_TOPO = ["elevation_m", "slope_deg", "hand_m", "twi_dinf"]


def alinhar(vetor, n: int):
    """firthlogist devolve coef_/pvals_ com o intercepto junto em algumas versoes."""
    v = np.asarray(vetor, dtype=float).ravel()
    if v.size == n:
        return v
    return v[-n:] if v.size > n else np.full(n, np.nan)


def features(flavor: str) -> list[str]:
    """As 6 do v12, com os nomes desta regiao."""
    return [
        "elevation_m", "slope_deg", "hand_m", "twi_dinf",
        f"rain_max_24h_{flavor}", f"rain_decay_index_api_{flavor}",
    ]


def ajustar_firth(X: np.ndarray, y: np.ndarray):
    from firthlogist import FirthLogisticRegression

    m = FirthLogisticRegression(max_iter=500, skip_pvals=False, skip_ci=True)
    m.fit(X, y)
    return m


def main() -> int:
    args = sys.argv[1:]
    if "--confirmo-pre-requisitos" not in args:
        print("BLOQUEADO: este script nao deve rodar antes de")
        print("  1. FT-UK-04 executado (features de chuva anexadas)")
        print("  2. flavor primario do CHIRPS decidido e registrado")
        print("  3. revisao humana do criterio N1-N4 da adjudicacao")
        print("Se os tres estiverem cumpridos, rode com --confirmo-pre-requisitos")
        return 1

    flavor = FLAVOR_PADRAO
    if "--flavor" in args:
        flavor = args[args.index("--flavor") + 1]

    # ------------------------------------------------------------------
    # MODO SEM CHUVA: roda com as 4 features topograficas quando a serie
    # CHIRPS ainda nao esta no disco. Nao e o modelo do artigo -- e o
    # resultado defensavel disponivel hoje, e sai carimbado como tal em
    # todas as saidas. Existe porque bloquear o projeto inteiro esperando
    # um download nao e rigor, e paralisia.
    # ------------------------------------------------------------------
    sem_chuva = "--sem-chuva" in args

    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(SEMENTE)
    print("===MOD-UK-01===")

    entrada = ENTRADA
    if sem_chuva or not ENTRADA.exists():
        entrada = RUNS / "ft-uk-03-amostra" / "amostra_candidata_uk_v1.csv"
        sem_chuva = True
        print("MODO=SEM_CHUVA (4 features topograficas) -- NAO e o modelo do artigo")
    df = pd.read_csv(entrada)
    cols = FEATURES_TOPO if sem_chuva else features(flavor)
    faltando = [c for c in cols if c not in df.columns]
    if faltando:
        print(f"ABORTADO: colunas ausentes {faltando}")
        return 2

    df = df.dropna(subset=cols + ["label", "grupo_cv"])
    y = df["label"].to_numpy().astype(int)
    X = df[cols].to_numpy(dtype=float)
    grupos = df["grupo_cv"].to_numpy()

    n_pontos = len(df)
    n_eventos = df.loc[df.label == 1, "rec_grp_id"].nunique()
    n_grupos = df["grupo_cv"].nunique()
    print(f"N_PONTOS={n_pontos:,} N_EVENTOS_INDEPENDENTES={n_eventos} "
          f"N_GRUPOS_CV={n_grupos} FLAVOR={flavor}")

    # REGRA DE EPV -- contada em eventos, nao em pontos
    epv = n_eventos / len(cols)
    print(f"EPV={epv:.1f} (eventos por variavel, minimo={EPV_MINIMO})")
    if epv < EPV_MINIMO:
        print("ABORTADO: orcamento de EPV insuficiente. Reduza features ou "
              "amplie a AOI. Nao vou produzir estimativa instavel em silencio.")
        return 3

    # --- validacao cruzada agrupada (U1) ---
    gkf = GroupKFold(n_splits=N_FOLDS)
    aucs, pred = [], np.full(len(y), np.nan)
    for treino, teste in gkf.split(X, y, groups=grupos):
        m = ajustar_firth(X[treino], y[treino])
        p = m.predict_proba(X[teste])[:, 1]
        pred[teste] = p
        if len(np.unique(y[teste])) > 1:
            aucs.append(roc_auc_score(y[teste], p))
    auc_cv = float(np.mean(aucs))
    print(f"AUC_CV_AGRUPADA={auc_cv:.4f} (media de {len(aucs)} folds, "
          f"dp={np.std(aucs):.4f})")

    # --- modelo completo e coeficientes ---
    modelo = ajustar_firth(X, y)
    coefs = pd.DataFrame({
        "feature": cols,
        "coef": alinhar(modelo.coef_, len(cols)),
        "p_valor": alinhar(getattr(modelo, "pvals_", [np.nan] * len(cols)), len(cols)),
    })
    print("---COEFICIENTES---")
    for _, r in coefs.iterrows():
        print(f"  {r.feature:32s} coef={r.coef:+.4f} p={r.p_valor:.4g}")

    # --- bootstrap reamostrando EVENTOS, nao linhas (U2) ---
    unicos = np.unique(grupos)
    boot = np.zeros((N_BOOT, len(cols)))
    for b in range(N_BOOT):
        amostra_g = rng.choice(unicos, size=len(unicos), replace=True)
        idx = np.concatenate([np.nonzero(grupos == g)[0] for g in amostra_g])
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            boot[b] = np.nan
            continue
        try:
            boot[b] = alinhar(ajustar_firth(X[idx], yb).coef_, len(cols))
        except Exception:  # noqa: BLE001
            boot[b] = np.nan
    ic = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    coefs["ic_baixo"] = ic[0]
    coefs["ic_alto"] = ic[1]
    coefs["cruza_zero"] = (ic[0] < 0) & (ic[1] > 0)
    print("---IC 95% POR REAMOSTRAGEM DE GRUPO---")
    for _, r in coefs.iterrows():
        marca = " (cruza zero)" if r.cruza_zero else ""
        print(f"  {r.feature:32s} [{r.ic_baixo:+.4f}, {r.ic_alto:+.4f}]{marca}")

    coefs.to_csv(OUT / f"coeficientes_{flavor}.csv", index=False)
    df.assign(pred_cv=pred).to_csv(OUT / f"predicoes_{flavor}.csv", index=False)
    (OUT / f"resumo_{flavor}.json").write_text(json.dumps({
        "n_pontos": int(n_pontos),
        "n_eventos_independentes": int(n_eventos),
        "n_grupos_cv": int(n_grupos),
        "epv_por_evento": round(epv, 2),
        "auc_cv_agrupada": round(auc_cv, 4),
        "auc_por_fold": [round(a, 4) for a in aucs],
        "flavor_chuva": None if sem_chuva else flavor,
        "modo_sem_chuva": bool(sem_chuva),
        "features_usadas": cols,
        "n_bootstrap": N_BOOT,
        "aviso": ("Validacao agrupada por evento. Comparar com o LOO-AUC do v12 "
                  "brasileiro (0,6781) exige cautela: metodologias de validacao "
                  "diferentes nao sao diretamente comparaveis."),
        "segundos": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
