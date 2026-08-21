"""
MOD-MEC-04 -- bootstrap de coeficiente RETOMAVEL, mesmo resultado do N_BOOT=600
que o `mod_mec04_elevacao_relativa.py` faria de uma vez, so que em lotes.

POR QUE ESTE SCRIPT EXISTE, E O QUE ELE NAO E:

nao e um metodo novo. E a MESMA conta de `avaliar()` no `mod_mec04_elevacao_
relativa.py`, quebrada em lotes porque o ambiente que a executou (sandbox
desta sessao) mata qualquer chamada acima de ~170-180 s, e 600 ajustes de
Firth por conjunto sobre ~63 mil linhas passam de 20 minutos. Rodar este
script localmente, sem esse limite, e desnecessario -- e o mesmo resultado.

COMO GARANTE SER O MESMO RESULTADO, NAO OUTRO:

o `avaliar()` original instancia `rng = np.random.default_rng(SEMENTE)` uma
vez por conjunto (TERRENO, COMPLETO) e consome dele, em ordem, um resample de
grupos por rodada de bootstrap. Como os dois conjuntos usam a MESMA semente e
o MESMO `grupo_cv`, a sequencia de reamostras e identica entre os dois -- so
os coeficientes ajustados mudam, porque as features mudam. Este script
recria essa mesma sequencia com um UNICO gerador (`SEMENTE`), gravando o
estado do gerador em disco a cada lote e retomando exatamente dali na
chamada seguinte. O resultado final -- os 600 coeficientes por conjunto -- e
bit-a-bit o mesmo que uma execucao unica produziria, porque
`numpy.random.Generator.bit_generator.state` captura a posicao inteira do
fluxo, nao uma re-semeadura.

O QUE ACONTECE NO FINAL: quando os dois conjuntos atingem N_BOOT_ALVO
draws, o script recalcula o IC95 dos coeficientes (percentil 2,5-97,5) e
SUBSTITUI so os campos `ic95` e `falhas`/`veredito` (que dependem do IC) em
`local_runs/mod-mec-04/resultado.json` -- os campos que ja eram fidelidade
plena (auc_cv, leave_one_source_out, transferencia_relevo, que usam o
n_boot=2000 proprio de `transferir()`, nao afetado por N_BOOT) ficam
intocados.

Uso (repetir ate ver "N_BOOT_ALVO atingido" no console):
    python scripts/suscetibilidade/mod_mec04_bootstrap_resumavel.py
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import susc_firth_shim  # noqa: F401,E402
from mod_mec04_elevacao_relativa import (  # noqa: E402
    CONJUNTOS, ENTRADA, NEGATIVO_ADMITIDO, OUT, SINAL_EXIGIDO,
)
from mod_serra01_ingreme_2features import (  # noqa: E402
    AUC_MAX, AUC_MIN, AUC_SUSPEITA, GAP_MAX, N_BOOT as N_BOOT_ORIGINAL,
    SEMENTE, ajustar, padronizar,
)

N_BOOT_ALVO = N_BOOT_ORIGINAL  # 600 -- mesma meta do mod_mec03/mod_mec04
ORCAMENTO_SEGUNDOS = 140  # margem sob o limite observado do sandbox (~170s)

CKPT = OUT / "boot_checkpoint.pkl"
RESULTADO = OUT / "resultado.json"


def carregar_dados() -> pd.DataFrame:
    d = pd.read_csv(ENTRADA, low_memory=False)
    d = d[d.classe.isin([0, 1]) & d.nivel_negativo.isin(NEGATIVO_ADMITIDO)].copy()
    todas = sorted({f for feats in CONJUNTOS.values() for f in feats})
    d = d.dropna(subset=todas + ["grupo_cv"]).reset_index(drop=True)
    return d


def estado_inicial(d: pd.DataFrame) -> dict:
    grupos = d.grupo_cv.unique()
    return {
        "rng_state": np.random.default_rng(SEMENTE).bit_generator.state,
        "boots": {nome: [] for nome in CONJUNTOS},
        "n_feito": 0,
        "grupos": grupos.tolist(),
    }


def carregar_checkpoint(d: pd.DataFrame) -> dict:
    if CKPT.exists():
        with CKPT.open("rb") as f:
            return pickle.load(f)
    return estado_inicial(d)


def salvar_checkpoint(estado: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with CKPT.open("wb") as f:
        pickle.dump(estado, f)


def main() -> int:
    t0 = time.time()
    if not ENTRADA.exists():
        print(f"ABORTADO: {ENTRADA} ausente.")
        return 1

    d = carregar_dados()
    estado = carregar_checkpoint(d)
    if estado["n_feito"] >= N_BOOT_ALVO:
        print(f"N_BOOT_ALVO atingido ({estado['n_feito']}/{N_BOOT_ALVO}) -- "
              "nada a fazer. Rode o passo de FECHAMENTO (--fechar).")
        if "--fechar" in sys.argv:
            return fechar(d, estado)
        return 0

    y_full = d.classe.to_numpy().astype(int)
    grupos = np.array(estado["grupos"])
    idx = {g: np.flatnonzero((d.grupo_cv == g).to_numpy()) for g in grupos}

    rng = np.random.default_rng(SEMENTE)
    rng.bit_generator.state = estado["rng_state"]

    X_por_conjunto = {}
    for nome, feats in CONJUNTOS.items():
        X = d[feats].to_numpy(dtype=float)
        X_por_conjunto[nome] = X

    # IMPORTANTE: `n_feito` conta TENTATIVAS (uma por rng.choice()), nao so
    # draws bem-sucedidas -- e exatamente o que `for _ in range(N_BOOT)` conta
    # no avaliar() original. Uma reamostra degenerada (uma classe so) ainda
    # consome uma unidade do orcamento de N_BOOT, do contrario a sequencia de
    # rng.choice() divergiria da que uma execucao unica teria produzido.
    feitos_nesta_chamada = 0
    while estado["n_feito"] < N_BOOT_ALVO and time.time() - t0 < ORCAMENTO_SEGUNDOS:
        esc = rng.choice(grupos, size=len(grupos), replace=True)
        linhas = np.concatenate([idx[g] for g in esc])
        ys = y_full[linhas]
        if len(np.unique(ys)) >= 2:
            for nome, X in X_por_conjunto.items():
                try:
                    Xs, _, _ = padronizar(X[linhas])
                    coef = ajustar(Xs, ys).coef_.ravel()[:X.shape[1]]
                    estado["boots"][nome].append(coef)
                except Exception as e:  # noqa: BLE001
                    print(f"   [{nome}] draw {estado['n_feito']} falhou: {e}")
        estado["n_feito"] += 1
        estado["rng_state"] = rng.bit_generator.state
        feitos_nesta_chamada += 1

    salvar_checkpoint(estado)
    seg = round(time.time() - t0, 1)
    print(f"lote: +{feitos_nesta_chamada} draws em {seg}s -- "
          f"total {estado['n_feito']}/{N_BOOT_ALVO}")

    if estado["n_feito"] >= N_BOOT_ALVO:
        print("N_BOOT_ALVO atingido nesta chamada.")
        return fechar(d, estado)

    restantes = N_BOOT_ALVO - estado["n_feito"]
    print(f"restam {restantes} draws -- rode de novo para continuar")
    return 0


def fechar(d: pd.DataFrame, estado: dict) -> int:
    """Recalcula IC95/falhas/veredito com os 600 draws e atualiza resultado.json."""
    if not RESULTADO.exists():
        print(f"ABORTADO: {RESULTADO} ausente -- rode mod_mec04_elevacao_relativa.py "
              "uma vez primeiro (ele grava auc_cv/LOSO, que este script nao recalcula).")
        return 2
    saida = json.loads(RESULTADO.read_text(encoding="utf-8"))

    for nome, feats in CONJUNTOS.items():
        boots = np.array(estado["boots"][nome])
        n_ok = len(boots)
        ic = (np.percentile(boots, [2.5, 97.5], axis=0) if n_ok > 30
              else np.full((2, len(feats)), np.nan))
        coef = saida["conjuntos"][nome]["coef"]
        falhas = [f for f in saida["conjuntos"][nome]["falhas"]
                  if "IC95 cruzando zero" not in f and "sinal invertido" not in f]
        for i, f in enumerate(feats):
            saida["conjuntos"][nome]["ic95"][f] = [
                round(float(ic[0, i]), 4), round(float(ic[1, i]), 4)]
            if f in SINAL_EXIGIDO:
                if np.sign(coef[f]) != SINAL_EXIGIDO[f]:
                    falhas.append(f"{f} com sinal invertido")
                if not np.isnan(ic[0, i]) and ic[0, i] * ic[1, i] <= 0:
                    falhas.append(f"{f} com IC95 cruzando zero")
        saida["conjuntos"][nome]["falhas"] = falhas
        saida["conjuntos"][nome]["veredito"] = (
            "COERENTE_COM_CRITERIOS" if not falhas else "FORA_DOS_CRITERIOS")
        saida["conjuntos"][nome]["n_boot_coeficiente"] = n_ok

    saida["bootstrap_coeficiente"] = {
        "n_boot_alvo": N_BOOT_ALVO,
        "n_boot_efetivo": estado["n_feito"],
        "metodo": "retomavel em lotes (mod_mec04_bootstrap_resumavel.py); "
                  "mesma semente e mesma sequencia de reamostra que uma "
                  "execucao unica de avaliar() com N_BOOT=600 produziria",
        "fidelidade": "PLENA -- N_BOOT_ALVO atingido" if estado["n_feito"] >= N_BOOT_ALVO
                      else f"PARCIAL -- {estado['n_feito']}/{N_BOOT_ALVO}",
    }
    RESULTADO.write_text(
        json.dumps(saida, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"FECHADO -- IC95 com N_BOOT={estado['n_feito']} gravado em {RESULTADO}")
    for nome in CONJUNTOS:
        print(f"\n--- {nome} ---")
        for f, (lo, hi) in saida["conjuntos"][nome]["ic95"].items():
            marca = "  <-- IC CRUZA ZERO" if lo * hi <= 0 else ""
            print(f"  {f:20s} IC95=[{lo:+7.4f},{hi:+7.4f}]{marca}")
        print(f"  VEREDITO={saida['conjuntos'][nome]['veredito']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
