"""
MOD-MEC-02 -- O pool fluvial depois da harmonizacao completa, e o que ele responde.

O QUE MUDOU DESDE O MOD-MEC-01:

o mec01 rodou com 5.834 pontos e 1.492 grupos, de duas fontes. Nao por escolha:
so 21 AOIs ingremes tinham derivacao harmonizada, e o piloto ingles ficara de
fora por incompatibilidade de chave -- o `ter02` casava ponto e raster pelo
nome da AOI, e no UK o grupo e o evento. Depois de derivar as 119 AOIs e o
recorte ingles na mesma cadeia, o pool passa a 33.071 pontos, 1.991 grupos e
tres fontes, com a planicie deixando de ser 1.680 pontos para ser 28.684.

ISSO IMPORTA PARA UMA PERGUNTA QUE FICOU ABERTA NO MEC01:

o leave-one-source-out mostrou que o modelo treinado so nas AOIs tropicais e
aplicado a Curitiba caia para 0,4880 -- nivel de acaso -- enquanto o caminho
inverso funcionava. Ficaram duas hipoteses, e o registro da epoca diz
explicitamente que nao dava para escolher entre elas com o dado de entao:

    (i)  amostra pequena e nao representativa do lado planicie
    (ii) incompatibilidade real entre serra tropical e planalto subtropical

Harmonizar as 97 AOIs de planicie e o que separa as duas. Se o LOSO recuperar,
era (i). Se continuar em torno de 0,50, e (ii), e ai o achado negativo passa a
ser sobre o fenomeno e nao sobre o conjunto.

O PROBLEMA QUE ESTE SCRIPT PRECISA TRATAR, e o mec01 nao tratava:

a hierarquia de negativo do `ds01` diz, sem ressalva, que negativo por
AUSENCIA nao entra em modelo -- ausencia de registro nao e registro de
ausencia. Os 442 negativos de Curitiba sao ausencia: existe chamado no 156 e o
assunto nao era hidrologico. No mec01 isso passou porque o `tipo_negativo`
vinha como texto cru da fonte, sem normalizacao contra a hierarquia, entao a
regra nunca chegou a ser avaliada.

Nao da para simplesmente repetir. Este script roda as DUAS versoes e reporta
lado a lado:

    ESTRITO    so negativo observado e por exclusao qualificada. E o que a
               hierarquia autoriza. Curitiba entra so com positivos, e o LOSO
               dela fica indefinido -- declarado, nao escondido.
    AMPLIADO   inclui o negativo por ausencia, com a proporcao reportada. E a
               composicao do mec01, agora com o desvio explicito.

Se as duas concordarem, o resultado do mec01 sobrevive a regra. Se divergirem,
a diferenca e a medida do quanto o negativo por ausencia estava carregando.

ANALISE NOVA: TRANSFERENCIA ENTRE CLASSES DE RELEVO. Com 4.387 pontos de serra
e 28.684 de planicie na mesma cadeia, da para treinar num relevo e testar no
outro. E a pergunta de transferencia do planejamento -- terreno nao
representado no treino -- medida em vez de suposta.

CRITERIOS: os mesmos de ext_criterios_de_acerto_v1.md, importados do
mod_serra01. Nao sao reajustados aqui.

NAO faz: nao ajusta hiperparametro, nao seleciona feature por desempenho, nao
mistura cadeia, e nao escolhe entre ESTRITO e AMPLIADO -- reporta os dois.

Uso:
    python scripts/suscetibilidade/mod_mec02_fluvial_pool_expandido.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import susc_firth_shim  # noqa: F401,E402
from ds03_esquema_alvo import VERSAO  # noqa: E402
from mod_serra01_ingreme_2features import (  # noqa: E402
    AUC_MAX, AUC_MIN, AUC_SUSPEITA, EPV_MINIMO, GAP_MAX, N_BOOT, SEMENTE,
    ajustar, padronizar,
)

REPO = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
RUNS = REPO / "local_runs"
ENTRADA = RUNS / "ds-05-tabela-unica" / f"tabela_unica_pool_fluvial_{VERSAO}.csv"
OUT = RUNS / "mod-mec-02"

FEATURES = ["elevation_m", "slope_deg", "hand_m", "twi_dinf"]
SINAL_EXIGIDO = {"hand_m": -1, "twi_dinf": +1}
N_FOLDS = 5

# a hierarquia do ds01, aplicada e nao so citada
NEGATIVO_ADMITIDO_ESTRITO = ("observado", "exclusao_qualificada")

# numero do mec01, guardado para comparacao direta
MEC01 = {"n": 5834, "grupos": 1492, "auc_cv": 0.7673, "gap": -0.0011,
         "loso_cems_para_curitiba": 0.4880}


def treinar_e_avaliar(d: pd.DataFrame, rng) -> dict:
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    y = d.classe.to_numpy().astype(int)
    X = d[FEATURES].to_numpy(dtype=float)
    g = d.grupo_cv.to_numpy()
    Xz, _, _ = padronizar(X)

    aucs, aucs_tr = [], []
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(Xz, y, groups=g):
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        m = ajustar(Xz[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(Xz[te])[:, 1]))
        aucs_tr.append(roc_auc_score(y[tr], m.predict_proba(Xz[tr])[:, 1]))
    auc = float(np.mean(aucs))
    gap = float(np.mean(aucs_tr) - auc)

    coef = ajustar(Xz, y).coef_.ravel()[:len(FEATURES)]
    grupos = d.grupo_cv.unique()
    idx = {gr: np.flatnonzero((d.grupo_cv == gr).to_numpy()) for gr in grupos}
    boots = []
    for _ in range(N_BOOT):
        esc = rng.choice(grupos, size=len(grupos), replace=True)
        linhas = np.concatenate([idx[gr] for gr in esc])
        ys = y[linhas]
        if len(np.unique(ys)) < 2:
            continue
        Xs, _, _ = padronizar(X[linhas])
        try:
            boots.append(ajustar(Xs, ys).coef_.ravel()[:len(FEATURES)])
        except Exception:  # noqa: BLE001
            continue
    boots = np.array(boots)
    ic = (np.percentile(boots, [2.5, 97.5], axis=0) if len(boots) > 30
          else np.full((2, len(FEATURES)), np.nan))

    falhas = []
    if auc >= AUC_SUSPEITA:
        falhas.append(f"AUC {auc:.4f} >= {AUC_SUSPEITA} (vazamento)")
    elif not (AUC_MIN <= auc <= AUC_MAX):
        falhas.append(f"AUC {auc:.4f} fora de [{AUC_MIN}, {AUC_MAX}]")
    if abs(gap) > GAP_MAX:
        falhas.append(f"gap {gap:+.4f} acima de {GAP_MAX}")
    for f, exigido in SINAL_EXIGIDO.items():
        i = FEATURES.index(f)
        if np.sign(coef[i]) != exigido:
            falhas.append(f"{f} com sinal invertido")
        if not np.isnan(ic[0, i]) and ic[0, i] * ic[1, i] <= 0:
            falhas.append(f"{f} com IC95 cruzando zero")

    return {"n": int(len(d)), "grupos": int(d.grupo_cv.nunique()),
            "pos": int(y.sum()), "neg": int(len(y) - y.sum()),
            "epv": round(d.grupo_cv.nunique() / len(FEATURES), 1),
            "auc_cv": round(auc, 4), "folds": len(aucs),
            "auc_min": round(min(aucs), 4), "auc_max": round(max(aucs), 4),
            "gap": round(gap, 4),
            "coef": {f: round(float(c), 4) for f, c in zip(FEATURES, coef)},
            "ic95": {f: [round(float(ic[0, i]), 4), round(float(ic[1, i]), 4)]
                     for i, f in enumerate(FEATURES)},
            "falhas": falhas,
            "veredito": "COERENTE_COM_CRITERIOS" if not falhas else "FORA_DOS_CRITERIOS"}


def transferir(d: pd.DataFrame, coluna: str) -> list[dict]:
    """Treina fora de um estrato e testa dentro dele. Nao e CV: e transferencia."""
    from sklearn.metrics import roc_auc_score

    y = d.classe.to_numpy().astype(int)
    X = d[FEATURES].to_numpy(dtype=float)
    saida = []
    for valor in sorted(d[coluna].dropna().unique()):
        te = (d[coluna] == valor).to_numpy()
        tr = ~te
        linha = {"estrato": str(valor), "coluna": coluna,
                 "n_teste": int(te.sum()),
                 "grupos_teste": int(d.loc[te, "grupo_cv"].nunique()),
                 "pos_teste": int(y[te].sum()),
                 "neg_teste": int(te.sum() - y[te].sum())}
        if len(np.unique(y[te])) < 2:
            linha["auc"] = None
            linha["motivo"] = ("estrato sem as duas classes: AUC indefinida. "
                               "Nao e falha do modelo, e ausencia de contraste")
        elif len(np.unique(y[tr])) < 2 or tr.sum() < 50:
            linha["auc"] = None
            linha["motivo"] = "treino sem as duas classes ou pequeno demais"
        else:
            mu, sd = X[tr].mean(0), X[tr].std(0)
            sd = np.where(sd == 0, 1, sd)
            m = ajustar((X[tr] - mu) / sd, y[tr])
            linha["auc"] = round(float(roc_auc_score(
                y[te], m.predict_proba((X[te] - mu) / sd)[:, 1])), 4)
        saida.append(linha)
    return saida


def contraste_interno(d: pd.DataFrame, fonte: str) -> dict:
    """O conjunto de teste tem contraste para ser separado por terreno?

    POR QUE ESTA CHECAGEM PRECEDE QUALQUER LEITURA DO LOSO:

    AUC de 0,50 num conjunto de teste tem duas causas possiveis e elas levam a
    conclusoes opostas. Ou o modelo nao transfere para aquele contexto, ou o
    conjunto de teste nao tem contraste que qualquer modelo pudesse achar. As
    duas produzem exatamente o mesmo numero.

    Distinguir e barato: mede-se, DENTRO da fonte, o AUC de cada feature
    isolada. Se nem a melhor feature sozinha separa as classes ali, entao nao
    ha o que transferir para -- e o 0,50 e propriedade do rotulo daquela fonte,
    nao evidencia sobre o modelo. Sem isto, um achado sobre o dado seria
    apresentado como achado sobre o fenomeno.
    """
    from sklearn.metrics import roc_auc_score

    s = d[d.fonte == fonte]
    y = s.classe.to_numpy().astype(int)
    if len(np.unique(y)) < 2:
        return {"fonte": fonte, "avaliavel": False,
                "motivo": "a fonte nao tem as duas classes"}
    r = {"fonte": fonte, "avaliavel": True, "n": int(len(s)),
         "pos": int(y.sum()), "neg": int(len(y) - y.sum()),
         "nivel_negativo": s.loc[s.classe == 0, "nivel_negativo"]
         .value_counts().to_dict(), "por_feature": {}}
    for f in FEATURES:
        v = s[f].to_numpy(dtype=float)
        a = float(roc_auc_score(y, v))
        r["por_feature"][f] = {
            "auc_bruto": round(a, 4),
            # AUC e simetrico: 0,35 separa tanto quanto 0,65, com sinal oposto
            "separacao": round(abs(a - 0.5) * 2, 4),
            "mediana_pos": round(float(np.nanmedian(v[y == 1])), 3),
            "mediana_neg": round(float(np.nanmedian(v[y == 0])), 3)}
    r["melhor_separacao"] = round(
        max(x["separacao"] for x in r["por_feature"].values()), 4)

    # Separacao fraca e direcao errada sao coisas diferentes, e so a segunda
    # contradiz o modelo. Com HAND o esperado e coeficiente negativo, isto e,
    # positivo mais BAIXO sobre a drenagem, o que da AUC bruto abaixo de 0,5.
    esperado = {"hand_m": "<", "twi_dinf": ">", "elevation_m": "<", "slope_deg": "<"}
    concorda = {}
    for f, sinal in esperado.items():
        a = r["por_feature"][f]["auc_bruto"]
        concorda[f] = (a < 0.5) if sinal == "<" else (a > 0.5)
    r["direcao_esperada"] = concorda
    r["direcao_concorda_em"] = f"{sum(concorda.values())}/{len(concorda)}"
    return r


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEMENTE)
    t0 = time.time()
    print("===MOD-MEC-02: POOL FLUVIAL APOS HARMONIZACAO COMPLETA===")

    if not ENTRADA.exists():
        print(f"ABORTADO: {ENTRADA} ausente. Rode ds05 antes.")
        return 1
    base = pd.read_csv(ENTRADA, low_memory=False)
    base = (base[base.classe.isin([0, 1])]
            .dropna(subset=FEATURES + ["grupo_cv"]).reset_index(drop=True))

    print(f"POOL n={len(base):,} grupos={base.grupo_cv.nunique():,} "
          f"fontes={base.fonte.nunique()}")
    comp = (base.groupby("fonte")
            .agg(n=("classe", "size"), pos=("classe", "sum"),
                 grupos=("grupo_cv", "nunique"))
            .assign(neg=lambda x: x.n - x.pos).sort_values("n", ascending=False))
    print(comp[["n", "pos", "neg", "grupos"]].to_string())
    print(f"relevo: {base.classe_relevo.value_counts().to_dict()}")

    neg = base[base.classe == 0]
    prop = (neg.nivel_negativo.value_counts(normalize=True) * 100).round(1).to_dict()
    print(f"\ncomposicao do negativo (a hierarquia do ds01 exige reportar): {prop}")

    resultados: dict = {"entrada": str(ENTRADA.relative_to(REPO)),
                        "mec01_para_comparacao": MEC01,
                        "proporcao_do_negativo_pct": prop, "variantes": {}}

    for nome, filtro in (
            ("ESTRITO", base.nivel_negativo.isin(
                list(NEGATIVO_ADMITIDO_ESTRITO) + ["nao_aplicavel"])),
            ("AMPLIADO", pd.Series(True, index=base.index))):
        d = base[filtro].reset_index(drop=True)
        print(f"\n{'='*70}\n--- VARIANTE {nome} ---")
        if nome == "ESTRITO":
            print("  negativo por ausencia excluido, conforme a hierarquia do ds01")
        else:
            print("  inclui negativo por ausencia; e a composicao do mec01")

        epv = d.grupo_cv.nunique() / len(FEATURES)
        if epv < EPV_MINIMO:
            print(f"  ABORTADO: EPV {epv:.1f} abaixo de {EPV_MINIMO}")
            resultados["variantes"][nome] = {"abortado": f"EPV {epv:.1f}"}
            continue

        r = treinar_e_avaliar(d, rng)
        print(f"  n={r['n']:,} grupos={r['grupos']:,} pos={r['pos']:,} "
              f"neg={r['neg']:,} EPV={r['epv']}")
        print(f"  AUC_CV={r['auc_cv']:.4f} (folds={r['folds']}, "
              f"{r['auc_min']:.4f}-{r['auc_max']:.4f}) gap={r['gap']:+.4f}")
        for f in FEATURES:
            lo, hi = r["ic95"][f]
            marca = "  <-- IC CRUZA ZERO" if lo * hi <= 0 else ""
            print(f"    {f:14s} coef={r['coef'][f]:+8.4f} "
                  f"IC95=[{lo:+7.4f},{hi:+7.4f}]{marca}")

        r["leave_one_source_out"] = transferir(d, "fonte")
        print("  --- leave-one-source-out ---")
        for x in r["leave_one_source_out"]:
            if x["auc"] is None:
                print(f"    sem {x['estrato']:12s} -> INDEFINIDO ({x['motivo'][:52]})")
            else:
                print(f"    sem {x['estrato']:12s} -> AUC={x['auc']:.4f} "
                      f"(n={x['n_teste']:,}, pos={x['pos_teste']:,})")

        r["leave_one_relevo_out"] = transferir(d, "classe_relevo")
        print("  --- transferencia entre classes de relevo ---")
        for x in r["leave_one_relevo_out"]:
            if x["auc"] is None:
                print(f"    treinar fora de {x['estrato']:18s} -> INDEFINIDO")
            else:
                print(f"    treinar fora de {x['estrato']:18s} -> AUC={x['auc']:.4f} "
                      f"(n={x['n_teste']:,})")

        for x in r["falhas"]:
            print(f"  REPROVA: {x}")
        print(f"  VEREDITO={r['veredito']}")
        resultados["variantes"][nome] = r

    # ---- a pergunta que estava em aberto ----
    print(f"\n{'='*70}\n--- A PERGUNTA DO MEC01: CEMS TROPICAL -> CURITIBA ---")
    amp = resultados["variantes"].get("AMPLIADO", {})
    alvo = next((x for x in amp.get("leave_one_source_out", [])
                 if x["estrato"] == "curitiba" and x["auc"] is not None), None)
    contraste = contraste_interno(base, "curitiba")
    resultados["contraste_interno_curitiba"] = contraste
    if contraste.get("avaliavel"):
        print("  contraste interno de Curitiba, por feature isolada:")
        for f, x in contraste["por_feature"].items():
            print(f"    {f:14s} AUC_bruto={x['auc_bruto']:.4f} "
                  f"separacao={x['separacao']:.4f}  "
                  f"mediana pos={x['mediana_pos']} neg={x['mediana_neg']}")
        print(f"    melhor separacao por feature isolada: "
              f"{contraste['melhor_separacao']:.4f}  |  direcao esperada em "
              f"{contraste['direcao_concorda_em']} features")

    if alvo is None:
        print("  indisponivel nesta execucao")
        resultados["resposta_loso_curitiba"] = None
    else:
        antes, agora = MEC01["loso_cems_para_curitiba"], alvo["auc"]
        # o treino cresce a cada harmonizacao, entao a descricao dele e
        # calculada e nao escrita: uma frase fixa aqui envelheceria em silencio
        treino = base[base.fonte != "curitiba"]
        planicie = int((treino.classe_relevo == "PLANO_OU_ONDULADO").sum())
        fontes_treino = sorted(treino.fonte.unique())
        print(f"\n  no mec01, treino de 2 fontes e 1.680 pontos de planicie: "
              f"AUC={antes:.4f}")
        print(f"  agora, treino de {len(fontes_treino)} fontes "
              f"({', '.join(fontes_treino)}) e {planicie:,} de planicie: "
              f"AUC={agora:.4f}")

        # A ordem das checagens importa: so faz sentido perguntar se o modelo
        # transfere depois de saber que ha para onde transferir.
        sem_contraste = (contraste.get("avaliavel")
                         and contraste["melhor_separacao"] < 0.20)
        so_ausencia = set(contraste.get("nivel_negativo", {})) == {"ausencia"}

        if sem_contraste or so_ausencia:
            hipotese = "iii"
            leitura = (
                "nenhuma das duas hipoteses do mec01 pode ser decidida com este "
                "teste. O negativo de Curitiba e por AUSENCIA de registro, e "
                "nenhuma feature isolada separa as classes la dentro "
                f"(melhor separacao {contraste.get('melhor_separacao')}). Um "
                "conjunto de teste sem contraste produz AUC de 0,50 qualquer que "
                "seja o modelo: o numero e propriedade do rotulo de Curitiba, "
                "nao evidencia sobre transferencia entre climas. O que a "
                f"harmonizacao mostrou foi que levar a planicie de treino de "
                f"1.680 para {planicie:,} pontos, em {len(fontes_treino)} "
                "fontes, nao muda o resultado -- consistente com o problema "
                "estar do lado do teste")
        elif agora >= 0.65:
            hipotese = "i"
            leitura = ("o colapso vinha da amostra de planicie, nao de "
                       "incompatibilidade entre climas: harmonizar recuperou")
        elif agora <= 0.55:
            hipotese = "ii"
            leitura = ("harmonizar a planicie nao recuperou, e o conjunto de "
                       "teste tem contraste. A incompatibilidade nao era de "
                       "amostra")
        else:
            hipotese = "indefinida"
            leitura = ("valor na faixa em que as duas hipoteses explicam; "
                       "nao se decide aqui")
        print(f"  HIPOTESE={hipotese}")
        print(f"  LEITURA: {leitura}")
        resultados["resposta_loso_curitiba"] = {
            "antes_mec01": antes, "agora": agora,
            "hipotese": hipotese, "leitura": leitura,
            "transferencia_por_relevo_e_o_teste_valido": (
                "a transferencia entre serra e planicie, medida em conjuntos "
                "com contraste, e o teste que efetivamente responde a pergunta "
                "de generalizacao -- ver leave_one_relevo_out")}

    resultados["segundos"] = round(time.time() - t0, 1)
    resultados["semente"] = SEMENTE
    resultados["nao_e"] = (
        "nao e validacao operacional nem autoriza uso preditivo: e um ajuste "
        "com validacao agrupada sobre uma tabela de pontos harmonizada")
    (OUT / "resultado.json").write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"\nGRAVADO={OUT}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
